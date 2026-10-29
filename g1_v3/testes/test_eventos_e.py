"""Testes do núcleo puro sorteia_topo_e_massa de g1_v3/eventos.py (E1 do plano; §6.3 "Fase E: núcleos puros
(29/10)" e §12.1; S §7, §11.2). Escritos antes do código: o ImportError é esperado até a lane E1.

Convenções: fatia é índice em FATIAS (0 ANDAR, 1 B, 2 C); `nivel_pegar` é o índice em `Niveis.pegar` (0..4), com
h_k e m_k = `Niveis.pegar[k]`; m_min = `Cena.caixa_massa_base`. Tensores sintéticos, gerador de semente 0 e
E = 20 000 amostras: com esse tamanho, as bordas de uma faixa uniforme ficam a menos de 0,002 m do sorteio.
"""
import unittest

import torch

from g1_v3 import cena, eventos
from g1_v3.env_cfg import make_env_cfg
from g1_v3.eventos import sorteia_topo_e_massa
from g1_v3.knobs import ATIVO

NIV = ATIVO.niveis
CENA = ATIVO.cena
ANDAR, B, C = 0, 1, 2
E = 20_000
H0 = NIV.pegar[0][0]  # 0,55 m: topo do nível 0, teto das fatias B e C
M_MIN = CENA.caixa_massa_base  # 1,0 kg
TOPO_MIN_C = NIV.topo_min_c  # 0,30 m (§6.2)
BORDA = 0.002  # folga para "o sorteio chegou à borda" com E = 20 000 (espaçamento ~1e-5 m)


def sorteia(fatia, nivel_pegar=0, nivel_massa_c=-1, origem_z=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.full((E,), fatia, dtype=torch.long)
    z = torch.full((E,), float(origem_z))
    return sorteia_topo_e_massa(f, nivel_pegar, nivel_massa_c, z, NIV, CENA, g)


class TestSorteiaTopoEMassa(unittest.TestCase):

    def test_topo_w_soma_a_origem_do_chao_local(self):
        # E1, linha "Teste": com a origem a +1,4 m, o topo da laje fica a 1,4 + h (topo_w = origem_z + topo)
        for fatia in (ANDAR, B, C):
            with self.subTest(fatia=fatia):
                topo, topo_w, _ = sorteia(fatia, nivel_pegar=2, nivel_massa_c=1, origem_z=1.4)
                self.assertTrue(torch.allclose(topo_w, topo + 1.4, atol=1e-5))

    def test_fatia_c_nunca_sorteia_topo_abaixo_do_minimo_no_nivel_4(self):
        # E1, linha "Teste": a fatia C nunca sorteia topo < topo_min_c (0,30 m), com o PEGAR no nível 4
        topo, _, _ = sorteia(C, nivel_pegar=4)
        self.assertGreaterEqual(topo.min().item(), TOPO_MIN_C - 1e-6)
        self.assertLessEqual(topo.max().item(), H0 + 1e-6)
        # h_4 = 0,04 fica abaixo do piso, então o mínimo sorteado tem de colar no piso (o max() está ativo)
        self.assertLess(topo.min().item(), TOPO_MIN_C + BORDA)

    def test_fatia_b_no_nivel_k_fica_entre_h_k_e_h_0_e_massa_entre_1_e_m_k(self):
        # E1, linha "Teste": a fatia B no nível k fica em [h_k; h_0] e a massa em [1; m_k]
        for k in range(len(NIV.pegar)):
            h_k, m_k = NIV.pegar[k]
            with self.subTest(nivel=k):
                topo, _, massa = sorteia(B, nivel_pegar=k)
                self.assertGreaterEqual(topo.min().item(), h_k - 1e-6)
                self.assertLessEqual(topo.max().item(), H0 + 1e-6)
                self.assertGreaterEqual(massa.min().item(), M_MIN - 1e-6)
                self.assertLessEqual(massa.max().item(), m_k + 1e-6)
                # as duas bordas são alcançadas (k = 0 é degenerado: topo = h_0 e massa = 1)
                self.assertLessEqual(topo.min().item(), h_k + BORDA)
                self.assertGreaterEqual(massa.max().item(), m_k - BORDA)

    def test_fatia_c_com_nivel_massa_menos_1_da_massa_1(self):
        # E1, linha "Teste": na fatia C com nivel_massa_c = −1, a massa é m_min = 1 kg
        _, _, massa = sorteia(C, nivel_pegar=3, nivel_massa_c=-1)
        self.assertTrue(torch.allclose(massa, torch.full_like(massa, M_MIN)))

    def test_fatia_c_massa_vai_ate_m_j_do_nivel_massa(self):
        # E1, §6.3 (assinatura): na C, massa U[m_min; m_j] com j = nivel_massa_c; j = 2 dá m_j = 3 kg
        _, _, massa = sorteia(C, nivel_pegar=4, nivel_massa_c=2)
        m_j = NIV.pegar[2][1]
        self.assertGreaterEqual(massa.min().item(), M_MIN - 1e-6)
        self.assertLessEqual(massa.max().item(), m_j + 1e-6)
        self.assertGreaterEqual(massa.max().item(), m_j - BORDA)

    def test_andar_topo_zero_massa_minima_e_topo_w_na_origem(self):
        # E1, linha "Teste": o ANDAR dá topo 0 (a cena estaciona), com massa m_min; topo_w fica na origem
        topo, topo_w, massa = sorteia(ANDAR, nivel_pegar=4, origem_z=1.4)
        self.assertTrue(bool((topo == 0).all()))
        self.assertTrue(torch.allclose(massa, torch.full_like(massa, M_MIN)))
        self.assertTrue(torch.allclose(topo_w, torch.full_like(topo_w, 1.4)))


class TestAtritoPadNoCfg(unittest.TestCase):
    """S1 (revisão E11): o `asset_cfg` do `atrito_pad` tem de chegar resolvido pelos `params` do cfg, com só as
    geoms dos pads. Sem isso `dr.geom_friction` põe μ = 1,0 em todas as geoms do robô a cada reset. Confere o cfg
    de `make_env_cfg()`; não constrói env. Os nomes são literais, então o `fullmatch` casa só com eles."""

    def test_atrito_pad_recebe_so_as_geoms_dos_pads(self):
        termo = make_env_cfg().events["atrito_pad"]
        self.assertIs(termo.func, eventos.atrito_pad)
        asset = termo.params["asset_cfg"]
        self.assertEqual(asset.name, "robot")
        self.assertEqual(tuple(asset.geom_names), cena.PALM_PAD_GEOMS)


if __name__ == "__main__":
    unittest.main()
