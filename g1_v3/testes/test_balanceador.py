"""Testes de g1_v3/balanceador.py (B0 do plano, §6.3; S §11.1, §11.2; PD-12, PD-13). Escritos antes do código.

Convenções (§6.3): `celula_id` é o índice em `Curriculo.celulas`, na ordem da B3 (ANDAR 0..5, CARREGAR 6..9,
PEGAR 10, BOTAR 11). A EMA: n > 0 → x = n_sucesso/n; com n_medidas = 0, s = x; senão s += α(x − s); s começa
em 0. `registra` com n = 10 e n_sucesso = 10 é sucesso total na iteração; `fecha` chama `fecha_iteracao`.
A carência de subida (`iters_min_nivel`) é 50 its; nos casos de fatia e de célula ela é desligada com
`dataclasses.replace`, para o teste não depender dela.
"""
import dataclasses
import math
import unittest

import torch

from g1_v3 import balanceador
from g1_v3.knobs import ATIVO

B = ATIVO.balanceador
N = ATIVO.niveis
COM = ATIVO.comando
TER = ATIVO.terreno
TAREFAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")
PORTAO = B.portao  # 0,75 (S §11.2)
ALFA = B.alfa      # 0,05 (PD-13)
SEM_CARENCIA = dataclasses.replace(B, iters_min_nivel=10 ** 6)


def novo(k_bal=B, sub_final=True):
    # 13/11, M3: por padrão o PEGAR e o BOTAR já estão no último subnível (o nível 0 de antes); os testes do
    # subnível passam `sub_final=False`
    cur = balanceador.Curriculo(k_bal, N, COM, TER)
    if sub_final:
        est = cur.estado()
        for t in ("PEGAR", "BOTAR"):
            est["tarefas"][t]["sub"] = len(N.subnivel_extra) - 1
        cur.carrega(est)
    return cur


def cid(cur, tarefa, tipo):
    # o id é o índice em `celulas`; o teste acha pela (tarefa, tipo), não por número fixo
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


def so_tipos_python(v):
    # §6.3: o estado() só tem int, float, bool, str, list, dict (e None)
    if isinstance(v, dict):
        return all(isinstance(k, str) and so_tipos_python(x) for k, x in v.items())
    if isinstance(v, list):
        return all(so_tipos_python(x) for x in v)
    return isinstance(v, (int, float, bool, str, type(None)))


class TestCarenciaDaTarefaNova(unittest.TestCase):

    def test_pegar_conta_a_carencia_da_abertura_da_fatia_b(self):
        # 13/11, A8: a fatia B abre na it 61 (> iters_min_nivel); o PEGAR, com sucesso total desde a 1ª medida, só sobe
        # de nível na it 61 + iters_min_nivel. Antes, `_ult` valia 0 e a 1ª medida (it 62) subia o nível sem carência.
        cur = novo()
        cur.it = B.iters_min_nivel + 10
        for tp in ("reto", "parado"):
            cid_ = cid(cur, "ANDAR", tp)
            cur.celulas[cid_].s, cur.celulas[cid_].n_medidas = 1.0, 1
        cur.fecha_iteracao()
        abriu = cur.it
        self.assertGreater(abriu, B.iters_min_nivel)
        self.assertTrue(cur.celulas[cid(cur, "PEGAR", "pegar")].aberta)
        self.assertEqual((cur._ult["PEGAR"], cur._ult["CARREGAR"]), (abriu, abriu))
        pegar = cid(cur, "PEGAR", "pegar")
        while cur.it < abriu + B.iters_min_nivel - 1:
            cur.registra(pegar, 10, 10)
            cur.fecha_iteracao()
            self.assertEqual(cur.nivel("PEGAR"), 0, f"it {cur.it}")
        cur.registra(pegar, 10, 10)
        cur.fecha_iteracao()
        self.assertEqual(cur.nivel("PEGAR"), 1)

    def test_botar_conta_a_carencia_da_abertura_da_fatia_c(self):
        # 13/11, A8: a fatia C abre quando o PEGAR passa no portão; `_ult` do BOTAR vale essa iteração
        cur = novo()
        cur.it = B.iters_min_nivel + 10
        cur._abre_cel("PEGAR", "pegar")
        cur.celulas[cid(cur, "PEGAR", "pegar")].s = 1.0
        cur.fecha_iteracao()
        self.assertTrue(cur.celulas[cid(cur, "BOTAR", "botar")].aberta)
        self.assertEqual(cur._ult["BOTAR"], cur.it)


class TestFracoes(unittest.TestCase):

    def test_sem_deficit_da_um_terco_para_cada(self):
        # S §11.1: F = piso + (1 − n·piso)(D + ε)/Σ(D + ε); com D = 0 e n = 3, 1/3 cada.
        f = balanceador.fracoes(torch.zeros(3, dtype=torch.float64), B.piso_tarefa, B.eps)
        for v in f.tolist():
            self.assertAlmostEqual(v, 1 / 3, delta=1e-6)

    def test_deficit_dominante_respeita_o_piso_e_soma_um(self):
        # Um déficit 0,75 e dois zeros: todo F ≥ piso (0,20) e a soma é 1 (S §11.1, piso de tarefa).
        f = balanceador.fracoes(torch.tensor([0.75, 0.0, 0.0], dtype=torch.float64),
                                B.piso_tarefa, B.eps)
        self.assertTrue(bool((f >= B.piso_tarefa - 1e-9).all().item()))
        self.assertAlmostEqual(f.sum().item(), 1.0, delta=1e-6)
        self.assertGreater(f[0].item(), f[1].item())


class TestAberturaInicial(unittest.TestCase):

    def test_so_o_andar_abre_no_comeco(self):
        # S §11.1: "só o ANDAR abre no começo (fase 0, 100%)".
        f = novo().fracoes_fatias()
        self.assertTrue(torch.allclose(f.double(), torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
                                       atol=1e-9))

    def test_nivel_e_degrau_zero_no_comeco(self):
        # S §11.2: o nível começa em 0 (ANDAR, PEGAR); o degrau do refino começa em 0 (§11.5).
        c = novo()
        for tarefa in TAREFAS:
            with self.subTest(tarefa=tarefa):
                self.assertEqual(c.degrau(tarefa), 0)
        self.assertEqual(c.nivel("ANDAR"), 0)
        self.assertEqual(c.nivel("PEGAR"), 0)


class TestAberturaDeFatiaB(unittest.TestCase):

    def test_B_so_com_reto_e_parado_no_portao_e_baixo_junto(self):
        # S §11.1 e 11/11 (§5.1): a B abre quando a média de reto e parado do ANDAR passa no portão (0,75). A
        # célula seguinte (baixo) abre pela média das abertas do ANDAR: o mesmo momento aqui.
        cur = novo(SEM_CARENCIA)
        reto, parado = cid(cur, "ANDAR", "reto"), cid(cur, "ANDAR", "parado")
        baixo = cid(cur, "ANDAR", "baixo")
        # fecha 1: reto passa (s = 1 na primeira medida), parado não (s = 0)
        fecha(cur, falha={parado})
        self.assertGreaterEqual(cur.celulas[reto].s, PORTAO)
        self.assertLess(cur.celulas[parado].s, PORTAO)
        self.assertEqual(cur.fracoes_fatias()[1].item(), 0.0)
        self.assertFalse(cur.celulas[baixo].aberta)
        primeiro = None
        for k in range(2, 80):
            fecha(cur)
            passam = (cur.celulas[reto].s + cur.celulas[parado].s) / 2 >= PORTAO
            aberta_b = cur.fracoes_fatias()[1].item() > 0
            self.assertEqual(aberta_b, passam, f"fecha {k}: a B abriu fora do critério")
            self.assertEqual(cur.celulas[baixo].aberta, passam, f"fecha {k}: baixo fora do critério")
            if aberta_b:
                primeiro = k
                break
        # parado sobe 1 − 0,95^j depois da 1ª medida (j = fechas depois da 1ª). A média (1 + parado) / 2 ≥ 0,75
        # pede parado ≥ 0,5: j = 14, então a B abre no 15º fecho.
        esperado = 1 + math.ceil(math.log(1 - (2 * PORTAO - 1)) / math.log(1 - ALFA))
        self.assertEqual(primeiro, esperado)

    def test_media_das_celulas_abre_a_fatia_b(self):
        # 11/11 (§5.1): a B abre pela média de reto e parado no portão. Com reto 0,60 e parado 0,95, a média
        # é 0,775 e abre; com parado 0,80, a média é 0,70 e não abre.
        for parado, abre in ((0.95, True), (0.80, False)):
            with self.subTest(parado=parado):
                cur = novo(SEM_CARENCIA)
                cur.celulas[cid(cur, "ANDAR", "reto")].s = 0.60
                cur.celulas[cid(cur, "ANDAR", "parado")].s = parado
                cur.fecha_iteracao()
                self.assertEqual(cur.celulas[cid(cur, "CARREGAR", "reto")].aberta, abre)

    def test_rampa_da_fatia_b_entra_com_teto_e_sobe_por_degrau(self):
        # 11/11 (§5.3): a B entra com fração ≤ rampa_fatia_inicio (0,05) e o teto sobe rampa_fatia_passo (0,02) a
        # cada rampa_fatia_its (12) fechos. Sem medida, o s das células nascidas em 0 fica em 0: o déficit da B
        # é 0,75 e a fração sem a rampa passa de 0,5. O excedente vai para o ANDAR, então a soma continua 1.
        cur = novo(SEM_CARENCIA)
        while cur.fracoes_fatias()[1].item() == 0:
            fecha(cur)
        self.assertAlmostEqual(cur.fracoes_fatias()[1].item(), 0.05)
        self.assertAlmostEqual(cur.fracoes_fatias()[0].item(), 0.95)
        self.assertAlmostEqual(cur.fracoes_fatias()[2].item(), 0.0)
        self.assertAlmostEqual(cur.fracoes_fatias().sum().item(), 1.0, delta=1e-9)
        for _ in range(B.rampa_fatia_its):
            cur.fecha_iteracao()
        self.assertAlmostEqual(cur.fracoes_fatias()[1].item(), 0.07)
        self.assertAlmostEqual(cur.fracoes_fatias()[0].item(), 0.93)
        self.assertAlmostEqual(cur.fracoes_fatias().sum().item(), 1.0, delta=1e-9)

    def test_fracao_da_fatia_c_entra_com_teto_no_fecho_da_abertura(self):
        # 11/11 (§5.3): a C entra com fração 0,05, o mesmo teto da B. A 1ª medida do PEGAR (s = 1, célula
        # nascida em 0 e medida no fecho seguinte) abre a C; com 3 fatias abertas, o piso dá 0,20 e o teto corta.
        cur = novo(SEM_CARENCIA)
        while cur.fracoes_fatias()[1].item() == 0:  # abre a B
            fecha(cur)
        fecha(cur)
        self.assertGreater(cur.fracoes_fatias()[2].item(), 0.0, "a C deveria abrir no fecho seguinte à B")
        self.assertAlmostEqual(cur.fracoes_fatias()[2].item(), 0.05)


class TestAberturaDeFatiaC(unittest.TestCase):

    def test_C_so_com_pegar_nivel_0_no_portao(self):
        # §11.1 e §11.2: a C abre quando o PEGAR nível 0 passa no portão (0,75). Sem carência, para o
        # nível não subir e mudar o critério no meio do caminho.
        cur = novo(SEM_CARENCIA)
        pegar = cid(cur, "PEGAR", "pegar")
        while cur.fracoes_fatias()[1].item() == 0:  # abre a B com sucesso total
            fecha(cur)
        k_b = cur.it
        self.assertTrue(cur.celulas[pegar].aberta, "o PEGAR nível 0 deveria abrir junto com a B")
        # primeira medida do PEGAR falha (s = 0); depois, sucesso total
        fecha(cur, falha={pegar})
        self.assertFalse(cur.fracoes_fatias()[2].item() > 0)
        primeiro = None
        for _ in range(60):
            fecha(cur)
            passa = cur.celulas[pegar].s >= PORTAO
            aberta_c = cur.fracoes_fatias()[2].item() > 0
            self.assertEqual(aberta_c, passa, f"fecha {cur.it}: C fora do critério")
            if aberta_c:
                primeiro = cur.it
                break
        esperado = k_b + 1 + math.ceil(math.log(1 - PORTAO) / math.log(1 - ALFA))
        self.assertEqual(primeiro, esperado)


class TestSubidaDeNivel(unittest.TestCase):

    def test_primeira_subida_so_depois_de_50_its(self):
        # PD-12: "mínimo de 50 its entre subidas": a 1ª subida do PEGAR conta desde a abertura da fatia B (fecha 1;
        # 13/11, A8; o ANDAR conta desde it = 0).
        cur = novo()
        for k in range(1, 51):
            fecha(cur)
            self.assertEqual(cur.nivel("PEGAR"), 0, f"subiu cedo, fecha {k}")
        fecha(cur)  # fecha 51
        self.assertEqual(cur.nivel("PEGAR"), 1)

    def test_EMA_recomeca_na_primeira_medida_depois_da_subida(self):
        # PD-12/§6.3: depois da subida, a 1ª medida dá s = x (não o EMA do s antigo).
        cur = novo()
        pegar = cid(cur, "PEGAR", "pegar")
        for _ in range(51):  # a fatia B abre na fecha 1 (13/11, A8); a subida sai na 51
            fecha(cur)
        self.assertEqual(cur.nivel("PEGAR"), 1)
        fecha(cur, falha={pegar})  # 1ª medida depois da subida: x = 0
        self.assertAlmostEqual(cur.celulas[pegar].s, 0.0, places=12)
        fecha(cur)  # próxima medida: s = 0 + α(1 − 0) = 0,05
        self.assertAlmostEqual(cur.celulas[pegar].s, ALFA, places=9)

    def test_proxima_subida_espera_mais_50_its(self):
        # Carência de 50 its entre subidas: nível = min((it − 1) // 50, 4) com sucesso total (a fatia B abre na it 1).
        cur = novo()
        for k in range(1, 202):
            fecha(cur)
            self.assertEqual(cur.nivel("PEGAR"), min((k - 1) // 50, 4), f"fecha {k}")

    def test_nivel_sobe_pela_media_das_velocidades_mesmo_com_uma_em_0_70(self):
        # 11/11 (§5.1): a subida exige a média das células de velocidade no portão, não todas: (0,70 + 4·0,80) / 5 = 0,78
        cur = novo(dataclasses.replace(B, iters_min_nivel=0))
        for tp in balanceador.VELOCIDADE["ANDAR"]:
            c = cur.celulas[cid(cur, "ANDAR", tp)]
            c.aberta, c.s = True, 0.80
        cur.celulas[cid(cur, "ANDAR", "reto")].s = 0.70
        cur.fecha_iteracao()
        self.assertEqual(cur.nivel("ANDAR"), 1)


class TestSubnivel(unittest.TestCase):
    # 13/11, M3: o nível 0 do PEGAR e do BOTAR tem 3 subníveis; o índice do nível só sobe depois do último

    def sobe_ate(self, cur, tarefa, sub, limite=400):
        # fecha com sucesso total até o subnível da tarefa valer `sub`; devolve o it da subida
        for _ in range(limite):
            fecha(cur)
            if cur.subnivel(tarefa) == sub:
                return cur.it
        raise AssertionError("o subnível não subiu")

    def test_pegar_sobe_0_1_2_com_carencia_e_o_nivel_fica_0(self):
        cur = novo(sub_final=False)
        self.assertEqual(cur.subnivel("PEGAR"), 0)
        self.assertEqual(cur.nivel_passou("PEGAR"), -1)
        it1 = self.sobe_ate(cur, "PEGAR", 1)
        self.assertEqual(cur.nivel("PEGAR"), 0)
        self.assertEqual(cur.nivel_passou("PEGAR"), -1)
        self.assertEqual(cur._ult["PEGAR"], it1)  # a subida grava `_ult`
        self.assertEqual(cur.celulas[cid(cur, "PEGAR", "pegar")].n_medidas, 0)  # e zera n_medidas
        it2 = self.sobe_ate(cur, "PEGAR", 2)
        self.assertEqual(it2 - it1, B.iters_min_nivel)  # a carência entre as subidas
        self.assertEqual(cur.nivel("PEGAR"), 0)
        # C1: no fecho da subida o `s` ainda é o do subnível anterior (no portão), mas sem medida nova o nível 0 não
        # passou e a trava do DR (atrito, `com_caixa`) fica desligada
        self.assertEqual(cur.nivel_passou("PEGAR"), -1)
        self.assertFalse(cur._dr["portao0"])
        fecha(cur)  # a primeira medida no último subnível, com s no portão: o nível 0 passou
        self.assertEqual(cur.nivel_passou("PEGAR"), 0)
        self.assertTrue(cur._dr["portao0"])
        for _ in range(B.iters_min_nivel):
            fecha(cur)
        self.assertEqual(cur.nivel("PEGAR"), 1)
        self.assertEqual(cur.subnivel("PEGAR"), 2)

    def test_botar_tem_o_mesmo_caminho_e_as_outras_tarefas_valem_o_ultimo(self):
        cur = novo(sub_final=False)
        self.assertEqual(cur.subnivel("BOTAR"), 0)
        self.assertEqual(cur.subnivel("ANDAR"), 2)
        self.assertEqual(cur.subnivel("CARREGAR"), 2)
        self.sobe_ate(cur, "BOTAR", 1)
        self.assertEqual(cur.nivel("BOTAR"), 0)
        self.assertEqual(cur.nivel_passou("BOTAR"), -1)
        self.sobe_ate(cur, "BOTAR", 2)
        self.assertEqual(cur.nivel("BOTAR"), 0)
        fecha(cur)
        self.assertEqual(cur.nivel_passou("BOTAR"), 0)

    def test_a_fatia_c_abre_no_subnivel_0_do_pegar(self):
        cur = novo(sub_final=False)
        for _ in range(3):
            fecha(cur)
        self.assertEqual(cur.subnivel("PEGAR"), 0)
        self.assertTrue(cur.celulas[cid(cur, "BOTAR", "botar")].aberta)

    def test_estado_leva_o_subnivel_e_estado_antigo_vale_o_ultimo(self):
        cur = novo(sub_final=False)
        self.sobe_ate(cur, "PEGAR", 1)
        est = cur.estado()
        self.assertEqual((est["tarefas"]["PEGAR"]["sub"], est["tarefas"]["BOTAR"]["sub"]), (1, 0))
        self.assertTrue(so_tipos_python(est))
        cur2 = novo(sub_final=False)
        cur2.carrega(est)
        self.assertEqual((cur2.subnivel("PEGAR"), cur2.subnivel("BOTAR")), (1, 0))
        self.assertEqual(cur2.estado(), est)
        for t in TAREFAS:
            del est["tarefas"][t]["sub"]
        cur2.carrega(est)
        self.assertEqual((cur2.subnivel("PEGAR"), cur2.subnivel("BOTAR")), (2, 2))


class TestRetomada(unittest.TestCase):

    def test_carregado_20_its_depois_de_uma_subida_espera_mais_30(self):
        # §6.3: a subida em 51 (13/11, A8); carregado em 70, o próximo nível sai em 101 (50 its depois da subida),
        # e não em 121. O estado é restaurado com `carrega`, e o contador `it` vem junto.
        cur = novo()
        for _ in range(70):
            fecha(cur)
        self.assertEqual(cur.nivel("PEGAR"), 1)
        estado = cur.estado()
        self.assertTrue(so_tipos_python(estado))
        cur2 = novo()
        cur2.carrega(estado)
        self.assertEqual(cur2.estado(), estado)
        self.assertEqual(cur2.it, 70)
        for k in range(71, 102):
            fecha(cur)
            fecha(cur2)
            self.assertEqual(cur2.nivel("PEGAR"), cur.nivel("PEGAR"), f"fecha {k}")
            if k == 100:
                self.assertEqual(cur2.nivel("PEGAR"), 1, "retomada recomeçou a carência")
        self.assertEqual(cur2.nivel("PEGAR"), 2)


class TestRefino(unittest.TestCase):

    def test_refino_so_depois_do_ultimo_nivel_e_100_its_no_portao_e_nunca_desce(self):
        # §6.3 / S §11.5: o refino começa com o último nível aberto (PEGAR 4, em it 201; 13/11, A8) e 100 its
        # seguidas no portão. Depois, falhar não faz o degrau descer.
        cur = novo()
        pegar = cid(cur, "PEGAR", "pegar")
        for _ in range(201):
            fecha(cur)
        self.assertEqual(cur.nivel("PEGAR"), 4)
        self.assertEqual(cur.degrau("PEGAR"), 0)
        for _ in range(50):  # até it 251: 50 its depois do último nível, ainda não são 100
            fecha(cur)
        self.assertEqual(cur.degrau("PEGAR"), 0)
        for _ in range(55):  # até it 306: 100 its no portão depois do it 201
            fecha(cur)
        self.assertGreaterEqual(cur.degrau("PEGAR"), 1)
        self.assertEqual(cur.nivel("PEGAR"), 4)  # não passa do último nível
        degrau = cur.degrau("PEGAR")
        for _ in range(40):  # falha de verdade: o degrau não desce
            fecha(cur, falha={pegar})
            self.assertGreaterEqual(cur.degrau("PEGAR"), degrau)


class TestNivelDeTerreno(unittest.TestCase):

    def test_terreno_sobe_uma_linha_por_carencia_ate_o_teto_e_o_refino_espera(self):
        # B3 e §6.3 (29/10): a linha do gerador sobe dentro da célula `terreno`, pela regra dos níveis;
        # teto 12 no ANDAR (linhas − 1) e 9 no CARREGAR (o maior `linha_max_carregar`, PD-20). O refino
        # do ANDAR espera o último nível de terreno.
        cur = novo()
        teto_andar = TER.linhas - 1
        teto_carregar = max(v for _nome, v in TER.linha_max_carregar)
        ter_andar = cid(cur, "ANDAR", "terreno")
        historico = []
        for _ in range(2000):
            antes = cur.nivel_terreno("ANDAR")
            fecha(cur)
            depois = cur.nivel_terreno("ANDAR")
            self.assertIn(depois - antes, (0, 1), f"it {cur.it}: o terreno pulou linha")
            if depois > antes:
                historico.append(cur.it)
            if depois > 0:
                self.assertTrue(cur.celulas[ter_andar].aberta, "linha de terreno sem a célula aberta")
            if depois < teto_andar:
                self.assertEqual(cur.degrau("ANDAR"), 0, f"it {cur.it}: refino antes do último terreno")
            self.assertLessEqual(cur.nivel_terreno("CARREGAR"), teto_carregar)
            self.assertEqual(cur.nivel_terreno("PEGAR"), 0)
        self.assertEqual(cur.nivel_terreno("ANDAR"), teto_andar)
        self.assertEqual(cur.nivel_terreno("CARREGAR"), teto_carregar)
        for a, b in zip(historico, historico[1:]):
            self.assertGreaterEqual(b - a, B.iters_min_nivel, "subida de terreno sem carência")
        self.assertGreaterEqual(cur.degrau("ANDAR"), 1)  # com sucesso total, o refino chega


class TestSorteioDeFatias(unittest.TestCase):

    def test_sorteio_com_fracoes_05_03_02_fica_a_1_por_cento(self):
        # §11.1: sorteia_fatias respeita as frações. A fração é trocada na instância (sem gancho novo).
        cur = novo()
        cur.fracoes_fatias = lambda: torch.tensor([0.5, 0.3, 0.2])
        gerador = torch.Generator().manual_seed(0)
        amostras = cur.sorteia_fatias(100_000, gerador)
        freq = torch.bincount(amostras, minlength=3).double() / 100_000
        for i, f in enumerate((0.5, 0.3, 0.2)):
            with self.subTest(fatia=i):
                self.assertAlmostEqual(freq[i].item(), f, delta=0.01)


class TestSorteioSemTerreno(unittest.TestCase):

    def test_sem_tipo_terreno_nunca_sorteia_a_celula_terreno_do_andar(self):
        # §6.4 `v3_celula` (29/10): fora do episódio de terreno o `twist` sorteia com `sem_tipo="terreno"`; a
        # célula `terreno` aberta nunca sai. Sem `sem_tipo`, a mesma célula aparece.
        cur = novo()
        ter = cid(cur, "ANDAR", "terreno")
        cur.celulas[ter].aberta = True
        gerador = torch.Generator().manual_seed(0)
        sem = cur.sorteia_celulas("ANDAR", 2000, gerador, sem_tipo="terreno")
        self.assertNotIn(ter, sem.tolist())
        com = cur.sorteia_celulas("ANDAR", 2000, gerador)
        self.assertIn(ter, com.tolist())


class TestIdCelula(unittest.TestCase):

    def test_id_do_terreno_de_cada_tarefa_e_do_reto_do_carregar(self):
        # §6.3: ANDAR 0..5 (terreno = 5), CARREGAR 6..9 (reto = 6, terreno = 9)
        cur = novo()
        self.assertEqual(cur.id_celula("ANDAR", "terreno"), 5)
        self.assertEqual(cur.id_celula("CARREGAR", "terreno"), 9)
        self.assertEqual(cur.id_celula("CARREGAR", "reto"), 6)
        self.assertEqual(cur.celulas[cur.id_celula("CARREGAR", "terreno")].tipo, "terreno")


@unittest.skipUnless(torch.cuda.is_available(), "sem CUDA: o teste de device roda no pod")
class TestDevice(unittest.TestCase):

    def test_sorteios_devolvem_no_device_do_gerador_cuda(self):
        cur = novo()
        cur.celulas[cid(cur, "ANDAR", "terreno")].aberta = True
        gerador = torch.Generator(device="cuda").manual_seed(0)
        fatias = cur.sorteia_fatias(16, gerador)
        self.assertEqual(fatias.device.type, "cuda")
        sem = cur.sorteia_celulas("ANDAR", 16, gerador)
        self.assertEqual(sem.device.type, "cuda")
        com = cur.sorteia_celulas("ANDAR", 16, gerador, sem_tipo="terreno")
        self.assertEqual(com.device.type, "cuda")


class TestEstado(unittest.TestCase):

    def test_carrega_devolve_o_mesmo_estado_com_o_contador(self):
        # §6.3: estado() → carrega() é identidade, inclusive o contador `it`.
        c = novo()
        for _ in range(20):
            c.fecha_iteracao()
        self.assertEqual(c.it, 20)
        c2 = novo()
        c2.carrega(c.estado())
        self.assertEqual(c2.it, 20)
        self.assertEqual(c2.estado(), c.estado())
        self.assertTrue(torch.allclose(c2.fracoes_fatias(), c.fracoes_fatias()))
        for tarefa in TAREFAS:
            with self.subTest(tarefa=tarefa):
                self.assertEqual(c2.nivel(tarefa), c.nivel(tarefa))
                self.assertEqual(c2.degrau(tarefa), c.degrau(tarefa))
                ids1, f1 = c.fracoes_celulas(tarefa)
                ids2, f2 = c2.fracoes_celulas(tarefa)
                self.assertEqual(ids1, ids2)
                self.assertTrue(torch.allclose(f1, f2))

    def test_it_abertura_vai_ao_estado_e_volta_no_carrega(self):
        # 11/11 (§5.3): a rampa precisa do fecho em que cada fatia abriu; o estado leva isso ao checkpoint
        c = novo()
        fecha(c)  # a B abre no 1º fecho, com sucesso total
        self.assertEqual(c.estado()["it_abertura"], [0, 1, None])
        c2 = novo()
        c2.carrega(c.estado())
        self.assertEqual(c2.estado()["it_abertura"], [0, 1, None])
        self.assertTrue(torch.allclose(c2.fracoes_fatias(), c.fracoes_fatias()))

    def test_estado_sem_it_abertura_carrega_o_padrao(self):
        # 11/11 (§5.3): um checkpoint sem a chave carrega com a rampa de fora (só o ANDAR abre na 0)
        c = novo()
        est = c.estado()
        del est["it_abertura"]
        c.carrega(est)
        self.assertEqual(c.estado()["it_abertura"], [0, None, None])


if __name__ == "__main__":
    unittest.main()
