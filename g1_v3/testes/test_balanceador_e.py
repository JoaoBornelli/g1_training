"""Testes do DR por nível do `Curriculo` de g1_v3/balanceador.py (E6 do plano; §6.3 "Fase E: núcleos puros
(29/10)", `dr()` e `nivel_passou`; §6.8; §12.1; PD-25, PD-28, PD-38, PD-46). Escritos antes do código: `k_dr`,
`dr()` e `nivel_passou` ainda não existem na classe, então os casos falham até a lane E6.

Convenções: o currículo é o de `test_balanceador.py`, com `fecha` do mesmo jeito (registra n = 10 por célula aberta
e fecha a iteração). O "portão do PEGAR nível 0" é o primeiro fecho em que `nivel_passou("PEGAR")` ≥ 0. Os valores
do DR vêm de `DR()` dos knobs: μ dos pads 0,6 → piso 0,5 em passos de 0,025; atraso 1 → 2 no portão.
"""
import dataclasses
import unittest

from g1_v3 import balanceador
from g1_v3.knobs import ATIVO, DR

B = ATIVO.balanceador
N = ATIVO.niveis
COM = ATIVO.comando
TER = ATIVO.terreno
DRK = DR()
MU0 = DRK.atrito_pad[0]  # 0,6: borda de baixo do μ dos pads antes do portão (§6.8)
PASSO_MU = DRK.passo_mu_pad  # 0,025 por nível
MU_MIN = DRK.mu_pad_min  # 0,5: piso do μ
TOL = 1e-9


def novo(k_dr=DRK):
    return balanceador.Curriculo(B, N, COM, TER, k_dr=k_dr)


def cid(cur, tarefa, tipo):
    # o id é o índice em `celulas`; o teste acha pela (tarefa, tipo), como em test_balanceador.py
    for i, c in enumerate(cur.celulas):
        if c.tarefa == tarefa and c.tipo == tipo:
            return i
    raise AssertionError(f"célula {tarefa}/{tipo} não existe")


def fecha(cur, falha=()):
    # sucesso total em toda célula aberta; as de `falha` registram sucesso 0 (n = 10)
    for i, c in enumerate(cur.celulas):
        if c.aberta:
            cur.registra(i, 10, 0 if i in falha else 10)
    cur.fecha_iteracao()


def ate(cur, condicao, falha=(), limite=200):
    # fecha até `condicao()` valer; falha se não valer dentro do limite
    for _ in range(limite):
        fecha(cur, falha)
        if condicao():
            return
    raise AssertionError("a condição não aconteceu dentro do limite")


def so_tipos_python(v):
    # §6.3: o estado() e o dr() só têm int, float, bool, str, list, dict (e None)
    if isinstance(v, dict):
        return all(isinstance(k, str) and so_tipos_python(x) for k, x in v.items())
    if isinstance(v, list):
        return all(so_tipos_python(x) for x in v)
    return isinstance(v, (int, float, bool, str, type(None)))


class TestEstadoInicialDoDR(unittest.TestCase):

    def test_dr_comeca_com_atrito_desligado_e_atraso_inicial(self):
        # E6, linha "Teste": o DR começa com atrito do pad e da laje desligados, e atraso_max = atraso_inicio[1]
        d = novo().dr()
        self.assertFalse(d["atrito_pad"])
        self.assertFalse(d["atrito_laje"])
        self.assertEqual(d["atraso_max"], DRK.atraso_inicio[1])
        self.assertFalse(d["vies_imu"])
        self.assertFalse(d["forca_caixa"])
        self.assertEqual(d["rfi"], 0.0)
        self.assertAlmostEqual(d["mu_min_pad"], MU0, places=12)  # §6.3: atrito_pad[0] antes do portão
        self.assertEqual(list(d["com_caixa"]), [0, 0])

    def test_sem_k_dr_o_dr_devolve_vazio(self):
        # E6, §6.3 (assinatura): sem `k_dr`, `dr()` devolve {} e o currículo segue como antes
        self.assertEqual(novo(k_dr=None).dr(), {})


class TestPortaoDoPegarNivel0(unittest.TestCase):

    def test_nivel_passou_do_pegar_acompanha_o_portao(self):
        # E6, linha "Teste": `nivel_passou("PEGAR")` é −1 antes do portão, 0 no portão do nível 0, e nível − 1
        # quando a célula sai do portão
        cur = novo()
        pegar = cid(cur, "PEGAR", "pegar")
        self.assertEqual(cur.nivel_passou("PEGAR"), -1)
        ate(cur, lambda: cur.nivel_passou("PEGAR") >= 0)
        self.assertEqual(cur.nivel("PEGAR"), 0)
        self.assertEqual(cur.nivel_passou("PEGAR"), 0)
        ate(cur, lambda: cur.nivel("PEGAR") == 1)  # primeira subida, it 50
        self.assertEqual(cur.nivel_passou("PEGAR"), 1)
        ate(cur, lambda: cur.celulas[pegar].s < B.portao, falha=(pegar,))
        self.assertEqual(cur.nivel("PEGAR"), 1)  # o nível não desce
        self.assertEqual(cur.nivel_passou("PEGAR"), 0)

    def test_portao_do_pegar_0_liga_atrito_dos_pads_e_da_laje_e_atraso_2(self):
        # E6, linha "Teste": o PEGAR no portão do nível 0 liga o atrito dos pads e da laje e põe o atraso em 2
        cur = novo()
        antes = None
        for _ in range(20):
            antes = cur.dr()
            fecha(cur)
            if cur.nivel_passou("PEGAR") >= 0:
                break
        else:
            self.fail("o PEGAR nível 0 não passou no portão")
        self.assertFalse(antes["atrito_pad"])
        self.assertFalse(antes["atrito_laje"])
        self.assertEqual(antes["atraso_max"], DRK.atraso_inicio[1])
        depois = cur.dr()
        self.assertTrue(depois["atrito_pad"])
        self.assertTrue(depois["atrito_laje"])
        self.assertEqual(depois["atraso_max"], DRK.atraso_portao[1])  # 2 passos (PD-28)
        self.assertEqual(depois["atraso_max"], 2)

    def test_vies_imu_liga_com_reto_e_parado_do_andar(self):
        # E6, §6.3 (docstring de `dr`): "vies_imu" liga com reto e parado do ANDAR no portão, no mesmo fecho que
        # abre a fatia B
        cur = novo()
        reto, parado = cid(cur, "ANDAR", "reto"), cid(cur, "ANDAR", "parado")
        antes = None
        for _ in range(20):
            antes = cur.dr()
            fecha(cur)
            if cur.celulas[reto].s >= B.portao and cur.celulas[parado].s >= B.portao:
                break
        self.assertFalse(antes["vies_imu"])
        self.assertTrue(cur.dr()["vies_imu"])

    def test_forca_caixa_e_rfi_ligam_na_primeira_subida_de_nivel(self):
        # E6, §6.3 (docstring de `dr`): "forca_caixa" e "rfi" ligam na primeira subida de nível do PEGAR ou do
        # CARREGAR; rfi passa de 0 para rfi[0]
        cur = novo()
        antes = None
        for _ in range(100):
            antes = cur.dr()
            fecha(cur)
            if cur.nivel("PEGAR") + cur.nivel("CARREGAR") > 0:
                break
        self.assertFalse(antes["forca_caixa"])
        self.assertEqual(antes["rfi"], 0.0)
        depois = cur.dr()
        self.assertTrue(depois["forca_caixa"])
        self.assertAlmostEqual(depois["rfi"], DRK.rfi[0], places=12)

    def test_com_caixa_segue_os_portoes_com_a_chave_ligada(self):
        # E6, §6.8 e PD-38: com `com_caixa` ligada, [xy, z] vai de [0, 0] para frac0 no portão do PEGAR 0 e para
        # frac_max na primeira subida de nível
        cur = novo(dataclasses.replace(DRK, com_caixa=True))
        self.assertEqual(list(cur.dr()["com_caixa"]), [0, 0])
        ate(cur, lambda: cur.nivel_passou("PEGAR") >= 0)
        self.assertEqual(list(cur.dr()["com_caixa"]), [DRK.com_caixa_frac0[0], DRK.com_caixa_frac0[1]])
        ate(cur, lambda: cur.nivel("PEGAR") + cur.nivel("CARREGAR") > 0)
        self.assertEqual(list(cur.dr()["com_caixa"]), [DRK.com_caixa_frac_max[0], DRK.com_caixa_frac_max[1]])


class TestMuDosPads(unittest.TestCase):

    def test_cada_nivel_desce_o_mu_0_025_ate_0_5_e_nada_sobe(self):
        # E6, linha "Teste": cada nível seguinte desce o μ mínimo em 0,025 até 0,5; nada sobe
        cur = novo()
        mu_ant, soma_ant = MU0, 0
        for _ in range(400):
            fecha(cur)
            d = cur.dr()
            soma = cur.nivel("PEGAR") + cur.nivel("CARREGAR")
            mu = d["mu_min_pad"]
            self.assertLessEqual(mu, mu_ant + TOL, f"it {cur.it}: o μ subiu")
            self.assertGreaterEqual(mu, MU_MIN - TOL, f"it {cur.it}: o μ passou do piso")
            if d["atrito_pad"] and soma > soma_ant:
                self.assertAlmostEqual(mu, max(MU_MIN, mu_ant - PASSO_MU * (soma - soma_ant)), places=9,
                                       msg=f"it {cur.it}: a subida de nível não desceu 0,025")
            mu_ant, soma_ant = mu, soma
        self.assertGreaterEqual(soma_ant, 4)  # PEGAR chega ao nível 4 em it 200 (test_balanceador.py)
        self.assertAlmostEqual(mu_ant, MU_MIN, places=9)

    def test_nada_desce_com_falhas_depois_do_portao(self):
        # E6, linha "Teste": nada desce. Com blocos de sucesso e falha, os flags não voltam, o atraso e o rfi não
        # caem e o μ não sobe
        cur = novo()
        pegar = cid(cur, "PEGAR", "pegar")
        ate(cur, lambda: cur.nivel_passou("PEGAR") >= 0)
        anterior = cur.dr()
        for bloco in range(12):
            falha = (pegar,) if bloco % 2 else ()
            for _ in range(30):
                fecha(cur, falha)
                atual = cur.dr()
                for chave in ("atrito_pad", "atrito_laje", "forca_caixa", "vies_imu"):
                    self.assertTrue(atual[chave] or not anterior[chave], f"{chave} voltou a False")
                self.assertGreaterEqual(atual["atraso_max"], anterior["atraso_max"])
                self.assertGreaterEqual(atual["rfi"], anterior["rfi"])
                self.assertLessEqual(atual["mu_min_pad"], anterior["mu_min_pad"] + TOL)
                anterior = atual


class TestEstadoDoDR(unittest.TestCase):

    def test_dr_volta_igual_pelo_estado_e_carrega_e_segue_igual(self):
        # E6, linha "Teste": `dr()` volta igual pelo `estado()`/`carrega()`, e as duas cópias seguem iguais
        cur = novo()
        pegar = cid(cur, "PEGAR", "pegar")
        for i in range(120):
            fecha(cur, falha=(pegar,) if i % 30 < 10 else ())
        est = cur.estado()
        self.assertTrue(so_tipos_python(est))
        self.assertTrue(so_tipos_python(cur.dr()))
        cur2 = novo()
        cur2.carrega(est)
        self.assertEqual(cur2.dr(), cur.dr())
        for i in range(60):
            falha = (pegar,) if i % 30 < 10 else ()
            fecha(cur, falha)
            fecha(cur2, falha)
            self.assertEqual(cur2.dr(), cur.dr(), f"fecha {i} depois da retomada")


if __name__ == "__main__":
    unittest.main()
