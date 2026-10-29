"""Testes da laje de destino do BOTAR: `eventos.sorteia_destino` (F2 do plano; §6.3 "Fase F: núcleos puros
(29/10)", §12.2 F2; S §8.2, §11.2; PD-47, [C38]). Escritos antes do código: o AttributeError de `sorteia_destino`
é esperado até a lane F2. A geometria de `alvos.laje_destino` já existe (fase B): esses casos devem passar hoje.

Convenções: tensores sintéticos float32, CPU, sem env. `gerador` é `torch.Generator` semeado. Os casos de geometria
usam recuo e θ fixos; o canto mais perto é o de menor x no frame da base, e a laje é quadrada com meia aresta
`Cena.prateleira_meia_xy` (a mesma nos dois eixos).
"""
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from g1_v3 import alvos, estimador, eventos
from g1_v3.knobs import ATIVO

CENA = ATIVO.cena
NIV = ATIVO.niveis
N = 20_000  # amostras: as bordas de uma faixa uniforme ficam a menos de 0,002 m do sorteio
BORDA = CENA.destino_borda  # 0,15 m
MEIA = CENA.prateleira_meia_xy  # 0,30 m
TOL_MM = 1e-3  # tolerância da geometria (1 mm), PD-47
GIRO_MAX = math.radians(CENA.destino_giro_max_graus)  # 10°


def sorteia(nivel, n=N, seed=0):
    g = torch.Generator().manual_seed(seed)
    return eventos.sorteia_destino(nivel, n, NIV, CENA, g)


class SorteioDestino(unittest.TestCase):
    def test_limites_dos_sorteios(self):
        # Teste (F2): θ em ±10°, y em ±0,10 m; longe em [0; jitter_x_max[nivel]]; topo em [botar[nivel]; botar[0]].
        for nivel in range(len(NIV.botar)):
            theta, y, longe, topo = sorteia(nivel)
            self.assertLessEqual(theta.abs().max().item(), GIRO_MAX + 1e-6)
            self.assertLessEqual(y.abs().max().item(), CENA.destino_jitter_y + 1e-6)
            self.assertGreaterEqual(longe.min().item(), 0.0)
            self.assertLessEqual(longe.max().item(), CENA.jitter_x_max[nivel] + 1e-6)
            self.assertGreaterEqual(topo.min().item(), NIV.botar[nivel] - 1e-6)
            self.assertLessEqual(topo.max().item(), NIV.botar[0] + 1e-6)

    def test_sorteio_cobre_a_faixa_inteira(self):
        # Teste (F2): com E = 20 000 o sorteio chega perto das duas bordas de θ e de y (uniforme, não degenerado).
        nivel = 3  # jitter_x_max = 0,15 m, botar = 0,15 m
        theta, y, longe, topo = sorteia(nivel)
        self.assertLess(theta.min().item(), -0.9 * GIRO_MAX)
        self.assertGreater(theta.max().item(), 0.9 * GIRO_MAX)
        self.assertLess(y.min().item(), -0.9 * CENA.destino_jitter_y)
        self.assertGreater(y.max().item(), 0.9 * CENA.destino_jitter_y)
        self.assertLess(longe.max().item(), CENA.jitter_x_max[nivel] + 1e-6)
        self.assertGreater(longe.max().item(), 0.9 * CENA.jitter_x_max[nivel])
        self.assertGreater(topo.max().item(), 0.9 * NIV.botar[0])

    def test_mesma_semente_mesmo_sorteio(self):
        # Teste (F2): o sorteio é determinístico com o gerador semeado (o teste de reset depende disso).
        a = sorteia(2, n=100, seed=7)
        b = sorteia(2, n=100, seed=7)
        for x, z in zip(a, b):
            self.assertTrue(torch.equal(x, z))


class GeometriaDaLaje(unittest.TestCase):
    # Os casos usam `alvos.laje_destino` direto: o núcleo é da fase B, e este teste confirma a conta do F2.

    def laje(self, recuo, theta, y):
        th = torch.tensor([theta], dtype=torch.float64)
        yy = torch.tensor([y], dtype=torch.float64)
        rc = torch.tensor([recuo], dtype=torch.float64)
        alvo, centro = alvos.laje_destino(rc, th, yy, MEIA, BORDA)
        return alvo[0].double(), centro[0].double()

    def test_theta_zero_centro_e_borda(self):
        # Teste (F2): θ = 0 e y = 0 põem o centro a recuo + 0,15 m e a borda perto a recuo − 0,15 m.
        for recuo in (0.40, 0.60):
            alvo, centro = self.laje(recuo, 0.0, 0.0)
            self.assertAlmostEqual(centro[0].item(), recuo + 0.15, delta=TOL_MM)
            self.assertAlmostEqual(centro[0].item() - MEIA, recuo - 0.15, delta=TOL_MM)

    def test_theta_dez_graus_canto_mais_perto(self):
        # Teste (F2): θ = 10° põe o canto mais perto a recuo − 0,15 m (± 1 mm), como no caso θ = 0.
        for recuo in (0.40, 0.60):
            for sinal in (1.0, -1.0):
                th = sinal * GIRO_MAX
                _, centro = self.laje(recuo, th, 0.0)
                c, s = math.cos(th), math.sin(th)
                # cantos da laje: centro + R(θ)·(±meia, ±meia); o de menor x é o canto perto da base
                xs = [centro[0].item() + c * dx - s * dy for dx in (-MEIA, MEIA) for dy in (-MEIA, MEIA)]
                self.assertAlmostEqual(min(xs), recuo - 0.15, delta=TOL_MM)

    def test_alvo_a_0_15_da_borda_perto_no_frame_da_laje(self):
        # Teste (F2): o alvo fica a 0,15 m da borda perto, no frame da laje, para todo θ e y.
        for recuo in (0.40, 0.60):
            for theta in (-GIRO_MAX, -0.05, 0.0, 0.05, GIRO_MAX):
                for y in (-0.10, 0.0, 0.10):
                    alvo, centro = self.laje(recuo, theta, y)
                    c, s = math.cos(theta), math.sin(theta)
                    dx, dy = (alvo - centro).tolist()
                    # frame da laje: gira o vetor centro→alvo por −θ
                    local_x = c * dx + s * dy
                    local_y = -s * dx + c * dy
                    # borda perto é a de x_local = −meia; o alvo tem de estar a 0,15 m dela
                    self.assertAlmostEqual(local_x + MEIA, BORDA, delta=TOL_MM)
                    self.assertAlmostEqual(local_y, 0.0, delta=TOL_MM)


class _Cena(dict):  # o `env.scene`: `scene["robot"]` e `scene.env_origins`
    env_origins = None


class BaseEYawDaLajeDestino(unittest.TestCase):
    # F9 R3: sai a cópia de base → mundo de `eventos` (usa `alvos.base_para_mundo`) e o rumo sai de `estimador.yaw`.

    def test_nao_ha_copia_de_base_para_mundo_em_eventos(self):
        # Antes do conserto: `eventos._base_para_mundo` existe e o teste falha.
        self.assertFalse(hasattr(eventos, "_base_para_mundo"))

    def test_rumo_da_laje_vem_do_quaternion_e_nao_de_heading_w(self):
        # Antes do conserto: o rumo vem de `heading_w` (falso, zero) e o teste falha com yaw 0 em vez de 0,5.
        psi = 0.5
        quat = torch.tensor([[math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)]], dtype=torch.float32)
        robo = SimpleNamespace(root_link_pos_w=torch.zeros(1, 3), root_link_quat_w=quat, heading_w=torch.zeros(1))
        cena = _Cena(robot=SimpleNamespace(data=robo))
        cena.env_origins = torch.zeros(1, 3)
        env = SimpleNamespace(
            v3_cur=SimpleNamespace(nivel=lambda nome: 0), v3_gerador=torch.Generator().manual_seed(0),
            device="cpu", scene=cena, v3_topo=torch.zeros(1), v3_topo_destino=torch.zeros(1),
            v3_meia=torch.tensor([0.1]), v3_ref=SimpleNamespace(recuo=lambda topo, meia: torch.full((1,), 0.5)))
        sorteio = (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.full((1,), 0.5))  # θ = 0, y = 0, longe = 0
        capturado = {}

        def laje(env_, ids, topo_w, xy=None, yaw=None):
            capturado["yaw"] = yaw

        with mock.patch.object(eventos, "sorteia_destino", return_value=sorteio), \
                mock.patch.object(eventos, "_laje", side_effect=laje):
            eventos.posiciona_destino(env, torch.tensor([0]))
        esperado = estimador.yaw(quat).double()
        self.assertTrue(torch.allclose(capturado["yaw"], esperado, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
