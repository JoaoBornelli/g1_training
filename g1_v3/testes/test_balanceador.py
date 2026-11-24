"""Testes de g1_v3/balanceador.py (B0 do plano, §6.3; S §11.1, §11.2; PD-12, PD-13). Escritos antes do código.

Convenções (§6.3): `celula_id` é o índice em `Curriculo.celulas`, na ordem da B3 (ANDAR 0..1, CARREGAR 2..5,
PEGAR 6, BOTAR 7). A EMA: n > 0 → x = soma/n; com n_medidas = 0, s = x; senão s += α(x − s); s começa
em 0. `registra` com n = 10 e soma = 10 é sucesso total na iteração; `fecha` chama `fecha_iteracao`.
16/11: a tarefa sobe depois de `iters_nivel` (50) fechos seguidos no portão; nos casos de fatia e de célula a subida
é desligada com `dataclasses.replace`, para o teste não depender dela. A fatia B abre com o ANDAR no nível
`nivel_abre_b` (5): `novo(andar=NIVEL_B)` põe o ANDAR nesse nível pelo estado.
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
PORTAO = B.portao  # 0,75 (S §11.2): as tarefas fora do ANDAR
PORTAO_ANDAR = B.portao_andar  # 0,80 (16/11, A4)
ALFA = B.alfa      # 0,05 (PD-13)
NIVEL_B = B.nivel_abre_b  # 5 (16/11, B1)
SEM_CARENCIA = dataclasses.replace(B, iters_nivel=10 ** 6)


def novo(k_bal=B, sub_final=True, andar=0):
    # 13/11, M3: por padrão o PEGAR e o BOTAR já estão no último subnível (o nível 0 de antes); os testes do
    # subnível passam `sub_final=False`. 16/11: `andar` é o nível do ANDAR; com `NIVEL_B` a fatia B abre no 1º fecho
    cur = balanceador.Curriculo(k_bal, N, COM, TER)
    est = cur.estado()
    if sub_final:
        for t in ("PEGAR", "BOTAR"):
            est["tarefas"][t]["sub"] = len(N.subnivel_extra) - 1
    est["tarefas"]["ANDAR"]["nivel"] = andar
    cur.carrega(est)
    return cur


def cid(cur, tarefa, tipo):
    # o id é o índice em `celulas`; o teste acha pela (tarefa, tipo), não por número fixo
    for i, c in enumerate(cur.celulas):
        if c.tarefa == tarefa and c.tipo == tipo:
            return i
    raise AssertionError(f"célula {tarefa}/{tipo} não existe")


def fecha(cur, falha=(), notas=None):
    # sucesso total em toda célula aberta; as de `falha` registram 0 (n = 10); `notas` dá a nota (0 a 1) por id de célula
    notas = notas or {}
    for i, c in enumerate(cur.celulas):
        if c.aberta:
            cur.registra(i, 10, 10 * notas.get(i, 0.0 if i in falha else 1.0))
    cur.fecha_iteracao()


def so_tipos_python(v):
    # §6.3: o estado() só tem int, float, bool, str, list, dict (e None)
    if isinstance(v, dict):
        return all(isinstance(k, str) and so_tipos_python(x) for k, x in v.items())
    if isinstance(v, list):
        return all(so_tipos_python(x) for x in v)
    return isinstance(v, (int, float, bool, str, type(None)))


class TestContagemDaTarefaNova(unittest.TestCase):

    def test_pegar_conta_as_50_seguidas_da_abertura_da_fatia_b(self):
        # 16/11, A4 (cobre o A8 de 13/11): a fatia B abre no 1º fecho com o ANDAR no nível 5; o PEGAR, com sucesso total
        # desde a 1ª medida, só sobe depois de 50 fechos seguidos medidos. A contagem da tarefa nova começa em 0.
        cur = novo(andar=NIVEL_B)
        pegar = cid(cur, "PEGAR", "pegar")
        fecha(cur)
        self.assertTrue(cur.celulas[pegar].aberta)
        self.assertEqual(cur._seguidas["PEGAR"], 0)
        for k in range(2, 2 + B.iters_nivel - 1):
            fecha(cur)
            self.assertEqual(cur.nivel("PEGAR"), 0, f"fecho {k}")
        fecha(cur)
        self.assertEqual(cur.nivel("PEGAR"), 1)

    def test_botar_abre_com_o_pegar_no_portao_e_conta_da_abertura(self):
        # a fatia C abre quando o PEGAR passa no portão; a contagem do BOTAR começa em 0 na abertura
        cur = novo(andar=NIVEL_B)
        cur._abre_cel("PEGAR", "pegar")
        cur.celulas[cid(cur, "PEGAR", "pegar")].s = 1.0
        cur.fecha_iteracao()
        self.assertTrue(cur.celulas[cid(cur, "BOTAR", "botar")].aberta)
        self.assertEqual(cur._seguidas["BOTAR"], 0)


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

    def test_B_abre_no_fecho_seguinte_ao_nivel_5_do_andar(self):
        # 16/11, B1: a B abre quando o ANDAR chega ao nível `nivel_abre_b`, no fecho seguinte ao da subida. Com a nota
        # do ANDAR em 0,85 em todo fecho o nível sobe no fecho 50·k; o nível 5 sai no fecho 250 e a B abre no 251.
        cur = novo()
        andar = cid(cur, "ANDAR", "andar")
        for k in range(1, 252):
            fecha(cur, notas={andar: 0.85})
            self.assertEqual(cur.celulas[cid(cur, "CARREGAR", "reto")].aberta, k >= 251, f"fecho {k}")
            self.assertEqual(cur.celulas[cid(cur, "PEGAR", "pegar")].aberta, k >= 251, f"fecho {k}")
        self.assertEqual(cur.nivel("ANDAR"), NIVEL_B)

    def test_media_alta_sem_o_nivel_nao_abre_a_fatia_b(self):
        # a B não abre pelo `s` do ANDAR: com a nota em 1,0 e o nível abaixo de 5, a CARREGAR reto fica fechada
        cur = novo(SEM_CARENCIA, andar=NIVEL_B - 1)
        for _ in range(20):
            fecha(cur)
        self.assertFalse(cur.celulas[cid(cur, "CARREGAR", "reto")].aberta)
        self.assertEqual(cur.fracoes_fatias()[1].item(), 0.0)

    def test_rampa_sobe_0_02_a_cada_12_fechos_sem_olhar_a_nota_do_andar(self):
        # 18/11, T9: a B entra com teto 0,05 e o teto sobe 0,02 a cada `rampa_fatia_its` fechos, contados da abertura.
        # A nota do ANDAR não entra: 0,85 (no portão), 0,75 e 0,60 dão a mesma rampa. A soma continua 1.
        for nota in (0.85, 0.75, 0.60):
            with self.subTest(nota=nota):
                cur = novo(SEM_CARENCIA, andar=NIVEL_B)
                notas = {cid(cur, "ANDAR", "andar"): nota, cid(cur, "CARREGAR", "reto"): 0.0, cid(cur, "PEGAR", "pegar"): 0.0}
                fecha(cur, notas=notas)  # o fecho 1 abre a B
                abriu = cur.it
                for _ in range(60):
                    fecha(cur, notas=notas)
                    degraus = (cur.it - abriu) // B.rampa_fatia_its
                    f = cur.fracoes_fatias()
                    self.assertAlmostEqual(f[1].item(), B.rampa_fatia_inicio + B.rampa_fatia_passo * degraus, msg=f"fecho {cur.it}")
                    self.assertAlmostEqual(f.sum().item(), 1.0, delta=1e-9)

    def test_rampa_para_na_parte_igual_e_nao_desce(self):
        # 18/11, T9: a B sobe até 1/2 e fica lá, com a nota do ANDAR em 0,60 e a B em s = 0 o tempo todo
        cur = novo(SEM_CARENCIA, andar=NIVEL_B)
        notas = {cid(cur, "ANDAR", "andar"): 0.6, cid(cur, "CARREGAR", "reto"): 0.0, cid(cur, "PEGAR", "pegar"): 0.0}
        anterior = 0.0
        for _ in range(40 * B.rampa_fatia_its):
            fecha(cur, notas=notas)
            f = cur.fracoes_fatias()
            self.assertGreaterEqual(f[1].item(), anterior - 1e-9, f"fecho {cur.it}: a fração da B desceu")
            self.assertLessEqual(f[1].item(), 0.5 + 1e-9)
            anterior = f[1].item()
        self.assertAlmostEqual(anterior, 0.5, delta=1e-9)

    def test_fatias_abertas_dividem_igual_e_a_nova_tira_das_outras_em_partes_iguais(self):
        # 18/11, T9: com a rampa no fim, 2 fatias dão 1/2 cada e 3 dão 1/3 cada, com qualquer nota. Com a C no começo
        # da rampa (teto 0,05), o ANDAR e a B dividem o resto: 0,475 cada.
        casos = (((0, 40, 0), False, (0.5, 0.5, 0.0)), ((0, 40, 40), True, (1 / 3, 1 / 3, 1 / 3)),
                 ((0, 40, 0), True, (0.475, 0.475, 0.05)))
        for rampa, abre_c, esperado in casos:
            with self.subTest(rampa=rampa, abre_c=abre_c):
                cur = novo()
                est = cur.estado()
                abertas = [("ANDAR", "andar"), ("CARREGAR", "reto"), ("PEGAR", "pegar")] + ([("BOTAR", "botar")] if abre_c else [])
                for tarefa, tipo in abertas:
                    est["celulas"][cid(cur, tarefa, tipo)]["aberta"] = True
                est["celulas"][cid(cur, "ANDAR", "andar")]["s"] = 0.85
                est["it_abertura"] = [0, 1, 1 if abre_c else None]
                est["rampa"] = list(rampa)
                cur.carrega(est)
                for v, e in zip(cur.fracoes_fatias().tolist(), esperado):
                    self.assertAlmostEqual(v, e, delta=1e-9)

    def test_fracao_da_fatia_c_entra_com_teto_no_fecho_da_abertura(self):
        # 11/11 (§5.3): a C entra com fração 0,05, o mesmo teto da B. A 1ª medida do PEGAR (s = 1, célula
        # nascida em 0 e medida no fecho seguinte) abre a C; com 3 fatias abertas, o piso dá 0,20 e o teto corta.
        cur = novo(SEM_CARENCIA, andar=NIVEL_B)
        while cur.fracoes_fatias()[1].item() == 0:  # abre a B
            fecha(cur)
            self.assertLess(cur.it, 200, "a B não abriu")
        fecha(cur)
        self.assertGreater(cur.fracoes_fatias()[2].item(), 0.0, "a C deveria abrir no fecho seguinte à B")
        self.assertAlmostEqual(cur.fracoes_fatias()[2].item(), 0.05)


class TestAberturaDeFatiaC(unittest.TestCase):

    def test_C_so_com_pegar_nivel_0_no_portao(self):
        # §11.1 e §11.2: a C abre quando o PEGAR nível 0 passa no portão (0,75). Sem carência, para o
        # nível não subir e mudar o critério no meio do caminho.
        cur = novo(SEM_CARENCIA, andar=NIVEL_B)
        pegar = cid(cur, "PEGAR", "pegar")
        while cur.fracoes_fatias()[1].item() == 0:  # abre a B com sucesso total
            fecha(cur)
            self.assertLess(cur.it, 200, "a B não abriu")
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
        # 16/11, A4: 50 fechos seguidos no portão. O PEGAR conta desde a abertura da fatia B (fecho 1) e é medido
        # a partir do fecho 2: a subida sai no fecho 51.
        cur = novo(andar=NIVEL_B)
        for k in range(1, 51):
            fecha(cur)
            self.assertEqual(cur.nivel("PEGAR"), 0, f"subiu cedo, fecha {k}")
        fecha(cur)  # fecha 51
        self.assertEqual(cur.nivel("PEGAR"), 1)

    def test_EMA_recomeca_na_primeira_medida_depois_da_subida(self):
        # PD-12/§6.3: depois da subida, a 1ª medida dá s = x (não o EMA do s antigo).
        cur = novo(andar=NIVEL_B)
        pegar = cid(cur, "PEGAR", "pegar")
        for _ in range(51):  # a fatia B abre na fecha 1; a subida sai na 51
            fecha(cur)
        self.assertEqual(cur.nivel("PEGAR"), 1)
        fecha(cur, falha={pegar})  # 1ª medida depois da subida: x = 0
        self.assertAlmostEqual(cur.celulas[pegar].s, 0.0, places=12)
        fecha(cur)  # próxima medida: s = 0 + α(1 − 0) = 0,05
        self.assertAlmostEqual(cur.celulas[pegar].s, ALFA, places=9)

    def test_proxima_subida_espera_mais_50_its(self):
        # 50 fechos seguidos entre subidas: nível = min((it − 1) // 50, 4) com sucesso total (a fatia B abre na it 1).
        cur = novo(andar=NIVEL_B)
        for k in range(1, 202):
            fecha(cur)
            self.assertEqual(cur.nivel("PEGAR"), min((k - 1) // 50, 4), f"fecha {k}")

    def test_nivel_sobe_pela_media_das_velocidades_mesmo_com_uma_em_0_70(self):
        # 11/11 (§5.1): a subida exige a média das células de velocidade no portão, não todas: (0,70 + 2·0,80) / 3 = 0,767
        # no CARREGAR (o ANDAR tem uma célula de velocidade só)
        cur = novo(dataclasses.replace(B, iters_nivel=1))
        for tp in balanceador.VELOCIDADE["CARREGAR"]:
            c = cur.celulas[cid(cur, "CARREGAR", tp)]
            c.aberta, c.s = True, 0.80
        cur.celulas[cid(cur, "CARREGAR", "reto")].s = 0.70
        cur.fecha_iteracao()
        self.assertEqual(cur.nivel("CARREGAR"), 1)

    def test_estatua_com_nota_abaixo_do_portao_do_andar_nao_sobe_o_nivel_e_a_B_fica_fechada(self):
        # 16/11, A1 e A4: a estátua com faixa de 0,5 tira 0,69 no núcleo da Unitree, abaixo do portão 0,80. O nível do
        # ANDAR fica em 0 por 3000 fechos e a B fica fechada. Com 0,79 (acima do portão 0,75 das outras tarefas) também.
        for nota in (0.69, 0.79):
            with self.subTest(nota=nota):
                cur = novo()
                andar = cid(cur, "ANDAR", "andar")
                for _ in range(3000):
                    fecha(cur, notas={andar: nota})
                self.assertEqual(cur.nivel("ANDAR"), 0)
                self.assertEqual(cur._seguidas["ANDAR"], 0)
                self.assertFalse(cur.celulas[cid(cur, "CARREGAR", "reto")].aberta)
                self.assertEqual(cur.fracoes_fatias()[1].item(), 0.0)

    def test_nota_0_85_sobe_o_nivel_a_cada_50_fechos_ate_10(self):
        # 16/11, A4: a cada 50 fechos seguidos com s ≥ 0,8 o nível sobe 1, até o máximo (10); depois fica
        cur = novo()
        andar = cid(cur, "ANDAR", "andar")
        self.assertEqual(cur.max_nivel["ANDAR"], 10)
        for k in range(1, 701):
            fecha(cur, notas={andar: 0.85})
            self.assertEqual(cur.nivel("ANDAR"), min(k // 50, 10), f"fecho {k}")

    def test_contagem_zera_com_uma_queda_e_conta_da_volta(self):
        # 16/11, A4: nota 0,85 por 40 fechos, 0,0 por 2, 0,85 depois. No fecho 50 o nível não sobe; ele sobe só
        # depois de 50 fechos seguidos com s ≥ 0,8, contados da volta (a EMA leva alguns fechos para voltar a 0,8)
        cur = novo()
        andar = cid(cur, "ANDAR", "andar")
        for _ in range(40):
            fecha(cur, notas={andar: 0.85})
        for _ in range(2):
            fecha(cur, notas={andar: 0.0})
        self.assertLess(cur.celulas[andar].s, PORTAO_ANDAR)
        seguidas = 0
        while cur.nivel("ANDAR") == 0:
            fecha(cur, notas={andar: 0.85})
            seguidas = seguidas + 1 if cur.celulas[andar].s >= PORTAO_ANDAR else 0
            self.assertLess(cur.it, 200, "o nível não subiu")
            if cur.it == 50:
                self.assertEqual(cur.nivel("ANDAR"), 0)
        self.assertEqual(seguidas, B.iters_nivel)
        self.assertGreater(cur.it, 50 + 1)

    def test_pegar_sobe_pela_mesma_regra_com_portao_0_75_o_subnivel_e_depois_o_nivel(self):
        # 16/11, P8: o PEGAR com s = 0,76 (≥ 0,75, abaixo de 0,80) por 50 fechos seguidos sobe o subnível (0 → 1) e,
        # depois de mais 50, o nível
        cur = novo(sub_final=False, andar=NIVEL_B)
        pegar = cid(cur, "PEGAR", "pegar")
        fecha(cur)  # o fecho 1 abre a B
        for k in range(2, 51):
            cur.registra(pegar, 100, 76)
            cur.fecha_iteracao()
            self.assertEqual(cur.subnivel("PEGAR"), 0, f"fecho {k}")
        cur.registra(pegar, 100, 76)
        cur.fecha_iteracao()  # fecho 51: 50 fechos medidos
        self.assertEqual((cur.subnivel("PEGAR"), cur.nivel("PEGAR")), (1, 0))
        for k in range(52, 101):
            cur.registra(pegar, 100, 76)
            cur.fecha_iteracao()
            self.assertEqual(cur.nivel("PEGAR"), 0, f"fecho {k}")
        cur.registra(pegar, 100, 76)
        cur.fecha_iteracao()  # fecho 101
        self.assertEqual(cur.nivel("PEGAR"), 1)


class TestSubnivel(unittest.TestCase):
    # 13/11, M3; 16/11: o nível 0 do PEGAR e do BOTAR tem 2 subníveis (sai a laje de 0,75 m); o índice do nível só sobe
    # depois do último

    def sobe_ate(self, cur, tarefa, sub, limite=400):
        # fecha com sucesso total até o subnível da tarefa valer `sub`; devolve o it da subida
        for _ in range(limite):
            fecha(cur)
            if cur.subnivel(tarefa) == sub:
                return cur.it
        raise AssertionError("o subnível não subiu")

    def test_pegar_sobe_0_1_com_50_seguidas_e_o_nivel_fica_0(self):
        cur = novo(sub_final=False, andar=NIVEL_B)
        self.assertEqual(cur.subnivel("PEGAR"), 0)
        self.assertEqual(cur.nivel_passou("PEGAR"), -1)
        self.sobe_ate(cur, "PEGAR", 1)
        self.assertEqual(cur.nivel("PEGAR"), 0)
        # C1: no fecho da subida o `s` ainda é o do subnível anterior (no portão), mas sem medida nova o nível 0 não
        # passou e a trava do DR (atrito, `com_caixa`) fica desligada
        self.assertEqual(cur.nivel_passou("PEGAR"), -1)
        self.assertFalse(cur._dr["portao0"])
        self.assertEqual(cur.celulas[cid(cur, "PEGAR", "pegar")].n_medidas, 0)  # a subida zera n_medidas
        fecha(cur)  # a primeira medida no último subnível, com s no portão: o nível 0 passou
        self.assertEqual(cur.nivel_passou("PEGAR"), 0)
        self.assertTrue(cur._dr["portao0"])
        for _ in range(B.iters_nivel - 2):
            fecha(cur)
            self.assertEqual(cur.nivel("PEGAR"), 0)
        fecha(cur)
        self.assertEqual(cur.nivel("PEGAR"), 1)
        self.assertEqual(cur.subnivel("PEGAR"), 1)

    def test_botar_tem_o_mesmo_caminho_e_as_outras_tarefas_valem_o_ultimo(self):
        cur = novo(sub_final=False, andar=NIVEL_B)
        self.assertEqual(cur.subnivel("BOTAR"), 0)
        self.assertEqual(cur.subnivel("ANDAR"), 1)
        self.assertEqual(cur.subnivel("CARREGAR"), 1)
        self.sobe_ate(cur, "BOTAR", 1)
        self.assertEqual(cur.nivel("BOTAR"), 0)
        self.assertEqual(cur.nivel_passou("BOTAR"), -1)
        fecha(cur)
        self.assertEqual(cur.nivel_passou("BOTAR"), 0)

    def test_a_fatia_c_abre_no_subnivel_0_do_pegar(self):
        cur = novo(sub_final=False, andar=NIVEL_B)
        for _ in range(3):
            fecha(cur)
        self.assertEqual(cur.subnivel("PEGAR"), 0)
        self.assertTrue(cur.celulas[cid(cur, "BOTAR", "botar")].aberta)

    def test_estado_leva_o_subnivel_e_estado_antigo_vale_o_ultimo(self):
        cur = novo(sub_final=False, andar=NIVEL_B)
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
        self.assertEqual((cur2.subnivel("PEGAR"), cur2.subnivel("BOTAR")), (1, 1))


class TestRetomada(unittest.TestCase):

    def test_carregado_20_its_depois_de_uma_subida_espera_mais_30(self):
        # §6.3: a subida em 51; carregado em 70, o próximo nível sai em 101 (50 fechos seguidos depois da subida),
        # e não em 121. O estado é restaurado com `carrega`, e o contador `it` e a contagem `seguidas` vêm junto.
        cur = novo(andar=NIVEL_B)
        for _ in range(70):
            fecha(cur)
        self.assertEqual(cur.nivel("PEGAR"), 1)
        estado = cur.estado()
        self.assertTrue(so_tipos_python(estado))
        self.assertEqual(estado["tarefas"]["PEGAR"]["seguidas"], 19)
        cur2 = novo()
        cur2.carrega(estado)
        self.assertEqual(cur2.estado(), estado)
        self.assertEqual(cur2.it, 70)
        for k in range(71, 102):
            fecha(cur)
            fecha(cur2)
            self.assertEqual(cur2.nivel("PEGAR"), cur.nivel("PEGAR"), f"fecha {k}")
            if k == 100:
                self.assertEqual(cur2.nivel("PEGAR"), 1, "retomada recomeçou a contagem")
        self.assertEqual(cur2.nivel("PEGAR"), 2)


class TestRefino(unittest.TestCase):

    def test_refino_so_depois_do_ultimo_nivel_e_100_its_no_portao_e_nunca_desce(self):
        # §6.3 / S §11.5: o refino começa com o último nível aberto (PEGAR 4, em it 201; 13/11, A8) e 100 its
        # seguidas no portão. Depois, falhar não faz o degrau descer.
        cur = novo(andar=NIVEL_B)
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
        cur = novo(andar=NIVEL_B)
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
            self.assertGreaterEqual(b - a, B.iters_nivel, "subida de terreno sem as 50 seguidas")
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
        # §6.3 (16/11): ANDAR 0..1 (andar = 0, terreno = 1), CARREGAR 2..5 (reto = 2, terreno = 5)
        cur = novo()
        self.assertEqual(cur.id_celula("ANDAR", "andar"), 0)
        self.assertEqual(cur.id_celula("ANDAR", "terreno"), 1)
        self.assertEqual(cur.id_celula("CARREGAR", "terreno"), 5)
        self.assertEqual(cur.id_celula("CARREGAR", "reto"), 2)
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
        c = novo(andar=NIVEL_B)
        fecha(c)  # a B abre no 1º fecho, com o ANDAR no nível 5
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

    def test_rampa_e_seguidas_vao_ao_estado_e_estado_sem_elas_carrega_o_padrao(self):
        # 16/11: o estado leva os degraus da rampa e a contagem de fechos seguidos; sem as chaves valem 0
        c = novo(SEM_CARENCIA, andar=NIVEL_B)
        for _ in range(40):
            fecha(c, notas={cid(c, "ANDAR", "andar"): 0.85, cid(c, "CARREGAR", "reto"): 0.0, cid(c, "PEGAR", "pegar"): 0.0})
        est = c.estado()
        self.assertEqual(est["rampa"], [0, 3, 0])
        self.assertGreater(est["tarefas"]["ANDAR"]["seguidas"], 0)
        self.assertTrue(so_tipos_python(est))
        c2 = novo()
        c2.carrega(est)
        self.assertEqual(c2.estado(), est)
        del est["rampa"]
        for t in TAREFAS:
            del est["tarefas"][t]["seguidas"]
        c2.carrega(est)
        self.assertEqual(c2.estado()["rampa"], [0, 0, 0])
        self.assertEqual([c2.estado()["tarefas"][t]["seguidas"] for t in TAREFAS], [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
