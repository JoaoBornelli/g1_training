"""Testes da fase H1 do `balanceador.py` (plano §12.4 H1; §6.3 "Fase H: núcleos puros (30/10)"; §6.9; P1, P5).

`no_degrau` é a fórmula do degrau numa função só; `degrau_global` é o menor degrau entre as 4 tarefas (PD-41);
o `dr()` do refino deriva dele. Escritos antes do código: `no_degrau` e `degrau_global` ainda não existem.
Os degraus e as travas entram por `estado()` → edição → `carrega()`, como no `test_balanceador_e.py`.
"""
import unittest

import torch

from g1_v3 import balanceador
from g1_v3.knobs import ATIVO, DR

B = ATIVO.balanceador
DRK = DR()
TAREFAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")
DEG = B.degraus  # 5
TOL = 9


def novo():
    return balanceador.Curriculo(B, ATIVO.niveis, ATIVO.comando, ATIVO.terreno, k_dr=DRK)


def com_degraus(degraus, subida=False, portao0=False):
    # degraus: (ANDAR, PEGAR, CARREGAR, BOTAR); as travas entram pelo estado, como no checkpoint
    cur = novo()
    est = cur.estado()
    for t, d in zip(TAREFAS, degraus):
        est["tarefas"][t]["degrau"] = d
    est["dr"] = {"portao0": portao0, "vies": False, "subida": subida}
    cur2 = novo()
    cur2.carrega(est)
    return cur2


def so_tipos_python(v):
    if isinstance(v, dict):
        return all(isinstance(k, str) and so_tipos_python(x) for k, x in v.items())
    if isinstance(v, list):
        return all(so_tipos_python(x) for x in v)
    return isinstance(v, (int, float, bool, str, type(None)))


class TestNoDegrau(unittest.TestCase):

    def test_extremos_e_meio_pela_formula_da_6_9(self):
        # §12.4 H1: d = 0 → ini; d = 5 → fim; d = 3 → ini + 0,6·(fim − ini)
        self.assertAlmostEqual(balanceador.no_degrau(50.0, 10.0, 0, 5), 50.0, places=TOL)
        self.assertAlmostEqual(balanceador.no_degrau(50.0, 10.0, 5, 5), 10.0, places=TOL)
        self.assertAlmostEqual(balanceador.no_degrau(50.0, 10.0, 3, 5), 26.0, places=TOL)  # laje_joelho_N
        self.assertAlmostEqual(balanceador.no_degrau(30.0, 15.0, 3, 5), 21.0, places=TOL)  # incl_lim
        self.assertAlmostEqual(balanceador.no_degrau(1.0, 0.25, 3, 5), 0.55, places=TOL)  # chegada
        self.assertAlmostEqual(balanceador.no_degrau(1.0, 0.35, 3, 5), 0.61, places=TOL)  # sigma_pouso
        self.assertAlmostEqual(balanceador.no_degrau(0.0, 1.0, 3, 5), 0.6, places=TOL)  # o fator 0 → 1

    def test_degrau_tensor_devolve_tensor_por_env(self):
        # §6.9: com `d` tensor (E,), o resultado é (E,): [50, 26, 10] para d = [0, 3, 5]
        d = torch.tensor([0, 3, 5], dtype=torch.long)
        out = balanceador.no_degrau(50.0, 10.0, d, 5)
        self.assertEqual(tuple(out.shape), (3,))
        self.assertTrue(torch.allclose(out, torch.tensor([50.0, 26.0, 10.0], dtype=out.dtype), atol=1e-6))

    def test_devolve_tipo_do_python_com_d_inteiro(self):
        # §12.4 H1: "só tipos do Python" (o estado e o dr() saem daqui)
        self.assertIsInstance(balanceador.no_degrau(0.02, 0.015, 5, 5), float)


class TestDegrauGlobal(unittest.TestCase):

    def test_menor_degrau_entre_as_quatro_tarefas(self):
        # §12.4 H1 (P5, PD-41): (ANDAR, PEGAR, CARREGAR, BOTAR) = (5, 3, 4, 0) → 0; (5, 3, 4, 2) → 2; (5, 5, 5, 5) → 5
        self.assertEqual(com_degraus((5, 3, 4, 0)).degrau_global(), 0)
        self.assertEqual(com_degraus((5, 3, 4, 2)).degrau_global(), 2)
        self.assertEqual(com_degraus((5, 5, 5, 5)).degrau_global(), 5)

    def test_sem_degrau_nenhum_e_zero(self):
        self.assertEqual(novo().degrau_global(), 0)


class TestDrDoRefino(unittest.TestCase):

    def test_g0_sem_travas_fica_no_inicio(self):
        # §12.4 H1: g = 0 sem travas: pd, forca_motor e forca_motor_bracos [1, 1]; solref [0,02; 0,02];
        # rfi 0; atraso_max 1
        d = com_degraus((0, 0, 0, 0)).dr()
        self.assertEqual(d["pd"], [1.0, 1.0])
        self.assertEqual(d["forca_motor"], [1.0, 1.0])
        self.assertEqual(d["forca_motor_bracos"], [1.0, 1.0])
        self.assertEqual(d["solref"], [0.02, 0.02])
        self.assertEqual(d["rfi"], 0.0)
        self.assertEqual(d["atraso_max"], DRK.atraso_inicio[1])

    def test_g0_com_portao0_e_subida_liga_rfi_e_atraso_2(self):
        # §12.4 H1: g = 0 com "portao0" e "subida": rfi 0,05, atraso_max 2
        d = com_degraus((0, 0, 0, 0), subida=True, portao0=True).dr()
        self.assertAlmostEqual(d["rfi"], 0.05, places=TOL)
        self.assertEqual(d["atraso_max"], 2)

    def test_g2_com_travas_mantem_atraso_2(self):
        # §12.4 H1: g = 2 com as travas: atraso_max 2 (abaixo de degrau_atraso_refino = 3)
        d = com_degraus((2, 2, 2, 2), subida=True, portao0=True).dr()
        self.assertEqual(d["atraso_max"], 2)

    def test_g3_com_travas(self):
        # §12.4 H1: g = 3 (degraus (5, 3, 4, 3)) com as travas
        d = com_degraus((5, 3, 4, 3), subida=True, portao0=True).dr()
        self.assertAlmostEqual(d["pd"][0], 0.94, places=TOL)
        self.assertAlmostEqual(d["pd"][1], 1.06, places=TOL)
        self.assertAlmostEqual(d["forca_motor"][0], 0.94, places=TOL)
        self.assertAlmostEqual(d["forca_motor"][1], 1.06, places=TOL)
        self.assertAlmostEqual(d["forca_motor_bracos"][0], 0.88, places=TOL)
        self.assertAlmostEqual(d["forca_motor_bracos"][1], 1.06, places=TOL)
        self.assertAlmostEqual(d["solref"][0], 0.017, places=TOL)
        self.assertAlmostEqual(d["solref"][1], 0.026, places=TOL)
        self.assertAlmostEqual(d["rfi"], 0.08, places=TOL)
        self.assertEqual(d["atraso_max"], 3)

    def test_g5_fecha_nas_pontas_dos_pares(self):
        # §12.4 H1: g = 5: pd e forca_motor [0,9; 1,1], forca_motor_bracos [0,8; 1,1], solref [0,015; 0,03],
        # rfi 0,1, atraso_max 3
        d = com_degraus((5, 5, 5, 5), subida=True, portao0=True).dr()
        self.assertAlmostEqual(d["pd"][0], 0.9, places=TOL)
        self.assertAlmostEqual(d["pd"][1], 1.1, places=TOL)
        self.assertAlmostEqual(d["forca_motor"][0], 0.9, places=TOL)
        self.assertAlmostEqual(d["forca_motor"][1], 1.1, places=TOL)
        self.assertAlmostEqual(d["forca_motor_bracos"][0], 0.8, places=TOL)
        self.assertAlmostEqual(d["forca_motor_bracos"][1], 1.1, places=TOL)
        self.assertAlmostEqual(d["solref"][0], 0.015, places=TOL)
        self.assertAlmostEqual(d["solref"][1], 0.03, places=TOL)
        self.assertAlmostEqual(d["rfi"], 0.1, places=TOL)
        self.assertEqual(d["atraso_max"], 3)

    def test_forca_caixa_segue_a_trava_sem_escala(self):
        # P15: "forca_caixa" é a trava "subida", sem escala pelo degrau
        self.assertFalse(com_degraus((5, 5, 5, 5), subida=False).dr()["forca_caixa"])
        self.assertTrue(com_degraus((0, 0, 0, 0), subida=True).dr()["forca_caixa"])

    def test_de_g_para_g_mais_1_o_lo_nao_sobe_e_o_hi_nao_desce(self):
        # §12.4 H1 (PD-24): de g para g + 1, o lo não sobe e o hi não desce, em todos os pares do refino
        for chave in ("pd", "forca_motor", "forca_motor_bracos", "solref"):
            anterior = None
            for g in range(DEG + 1):
                par = com_degraus((g, g, g, g), subida=True, portao0=True).dr()[chave]
                if anterior is not None:
                    self.assertLessEqual(par[0], anterior[0] + 1e-12, f"{chave} g={g}: lo subiu")
                    self.assertGreaterEqual(par[1], anterior[1] - 1e-12, f"{chave} g={g}: hi desceu")
                anterior = par
        # rfi é escalar (a fração de τ_max com a trava "subida"): só sobe de g para g + 1
        rfis = [com_degraus((g, g, g, g), subida=True).dr()["rfi"] for g in range(DEG + 1)]
        self.assertEqual(rfis, sorted(rfis))

    def test_so_tipos_do_python_e_carrega_repete_o_dr(self):
        # §12.4 H1: só tipos do Python; `carrega(estado())` dá o mesmo `dr()`
        cur = com_degraus((5, 3, 4, 3), subida=True, portao0=True)
        self.assertTrue(so_tipos_python(cur.dr()))
        cur2 = novo()
        cur2.carrega(cur.estado())
        self.assertEqual(cur2.dr(), cur.dr())


if __name__ == "__main__":
    unittest.main()
