"""Testes da tarefa E2 do plano (§12.1, `estimador.py`): reduz_simetria_cubo, visivel, EstimadorCaixa e
alvo_no_frame_da_base (§6.3). Escritos antes do código: o import falha até a lane E2 criar as funções.

Convenções: quaternion wxyz; câmera com o eixo óptico em +x do próprio frame (plano, §6.2 `Estimador`,
eixo x do site d435); comprimentos em metros; ângulos em graus nas tabelas e em rad nas funções.
Tensores sintéticos em CPU; nenhum env nem simulador.
"""
import math
import unittest

import torch

from g1_v3.estimador import (EstimadorCaixa, Odometria, alvo_no_frame_da_base, do_frame_da_base, reduz_simetria_cubo,
                             visivel)
from g1_v3.knobs import ATIVO

K = ATIVO.estimador
DEV = "cpu"
F = torch.float32
Q_ID = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=F)
POS_ZERO = torch.zeros(1, 3, dtype=F)
MEIA = torch.full((1,), 0.10, dtype=F)


def quat_de_yaw(psi):
    z = torch.zeros_like(psi)
    return torch.stack([torch.cos(psi / 2), z, z, torch.sin(psi / 2)], dim=-1)


def quat_de_z(graus):
    """Quaternion wxyz de uma rotação de `graus` em torno de z (E=1)."""
    a = math.radians(graus) / 2
    return torch.tensor([[math.cos(a), 0.0, 0.0, math.sin(a)]], dtype=F)


def caixa_em(x):
    """9 pontos de uma caixa de 10 cm centrada em (x, 0, 0): centro e os 8 cantos, (1, 9, 3)."""
    h = 0.05
    cantos = [[x + sx * h, sy * h, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return torch.tensor([[[x, 0.0, 0.0]] + cantos], dtype=F)


def magnitude(sigma_pos=0.0, perda=0.0):
    """Magnitude corrente do ruído da caixa (formato a confirmar: chaves = campos de `Estimador`)."""
    return {"sigma_pos": sigma_pos, "perda": perda}


class TestReduzSimetriaCubo(unittest.TestCase):
    def test_identidade_da_zero(self):
        # Teste E2: identidade → 0
        out = reduz_simetria_cubo(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=F))
        self.assertLess(float(out.abs().max()), 1e-6)

    def test_90_graus_em_z_da_zero(self):
        # Teste E2: 90° em z → 0 (é simetria do cubo)
        out = reduz_simetria_cubo(quat_de_z(90.0))
        self.assertLess(float(out.abs().max()), 1e-5)

    def test_100_graus_em_z_da_10_graus_em_z(self):
        # Teste E2: 100° em z → 10° em z
        out = reduz_simetria_cubo(quat_de_z(100.0))
        self.assertAlmostEqual(float(out[0, 2]), math.radians(10.0), places=5)
        self.assertLess(float(out[0, :2].abs().max()), 1e-5)

    def test_30_graus_em_z_da_30_graus_em_z(self):
        # Teste E2: 30° em z → 30°
        out = reduz_simetria_cubo(quat_de_z(30.0))
        self.assertAlmostEqual(float(out[0, 2]), math.radians(30.0), places=5)

    def test_180_graus_em_z_da_zero(self):
        # Borda (além do plano): 180° em z também é simetria do cubo. Quaternion (−1,0,0,0) é a mesma rotação.
        out = reduz_simetria_cubo(torch.tensor([[-1.0, 0.0, 0.0, 0.0]], dtype=F))
        self.assertLess(float(out.abs().max()), 1e-5)


class TestVisivel(unittest.TestCase):
    def _ve(self, x, capsulas=None, raios=None):
        if capsulas is None:
            capsulas = torch.zeros(1, 0, 2, 3, dtype=F)
            raios = torch.zeros(0, dtype=F)
        return bool(visivel(POS_ZERO, Q_ID, caixa_em(x), capsulas, raios, K)[0])

    def test_caixa_a_1_m_na_frente_e_visivel(self):
        # Teste E2: caixa a 1 m à frente da câmera sem cápsula → visível
        self.assertTrue(self._ve(1.0))

    def test_caixa_atras_nao_e_visivel(self):
        # Teste E2: atrás da câmera → não
        self.assertFalse(self._ve(-1.0))

    def test_caixa_a_20_cm_nao_e_visivel(self):
        # Teste E2: a 0,20 m → não (alcance mínimo de 0,28 m)
        self.assertFalse(self._ve(0.20))

    def test_capsula_de_braco_no_segmento_bloqueia(self):
        # Teste E2: cápsula de braço no segmento câmera–caixa → não (cobre os 9 pontos)
        capsula = torch.tensor([[[[0.5, -0.5, 0.0], [0.5, 0.5, 0.0]]]], dtype=F)  # (1, K=1, 2, 3)
        raios = torch.tensor([0.10], dtype=F)  # (K,)
        self.assertFalse(self._ve(1.0, capsula, raios))

    def test_capsula_fora_do_segmento_nao_bloqueia(self):
        # Borda: a mesma cápsula deslocada 50 cm em y não cruza a linha de visada → visível
        capsula = torch.tensor([[[[0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]]], dtype=F)
        raios = torch.tensor([0.10], dtype=F)
        self.assertTrue(self._ve(1.0, capsula, raios))


class TestEstimadorCaixa(unittest.TestCase):
    def test_leitura_do_instante_t_sai_em_t_mais_33_ms(self):
        # Teste E2: leitura do instante t entregue em t + 33 ms. A caixa anda 1 m/s em x (x = t); a leitura
        # de τ aparece em t = τ + latencia_s. Com a grade de 30 Hz, a saída de t cai em (t−0,033−1/30; t−0,033].
        est = EstimadorCaixa(1, DEV, K, None)  # gerador None: sem ruído, medida determinística
        odo = Odometria(1, DEV, K, None)
        ids = torch.arange(1)
        est.reinicia(ids)
        odo.reinicia(ids, POS_ZERO, Q_ID)
        vis = torch.ones(1, dtype=torch.bool)
        saidas = {}
        for k in range(41):
            t = 0.02 * k
            est.passo(t, torch.tensor([[t, 0.0, 0.0]], dtype=F), Q_ID, MEIA, vis, magnitude())
            saidas[k] = float(est.saida_b(odo, Q_ID)[0, 0])
        x = saidas[20]  # t = 0,40 s
        self.assertGreater(x, 0.40 - K.latencia_s - 1.0 / K.taxa_hz - 1e-4)
        self.assertLessEqual(x, 0.40 - K.latencia_s + 1e-4)
        self.assertGreater(abs(x - 0.40), 0.03)  # não é a leitura do instante atual

    def test_sem_leitura_a_idade_cresce_0_02_por_passo_e_para_em_2_s(self):
        # Teste E2: sem leitura a idade cresce 0,02 s por passo e para em 2 s (idade_max)
        est = EstimadorCaixa(1, DEV, K, None)
        odo = Odometria(1, DEV, K, None)
        ids = torch.arange(1)
        est.reinicia(ids)
        odo.reinicia(ids, POS_ZERO, Q_ID)
        nao_vis = torch.zeros(1, dtype=torch.bool)
        idades = []
        for k in range(1, 201):
            est.passo(0.02 * k, POS_ZERO, Q_ID, MEIA, nao_vis, magnitude())
            idades.append(float(est.saida_b(odo, Q_ID)[0, 7]))
        for a, b in zip(idades, idades[1:]):
            if a < K.idade_max - 1e-6:
                self.assertAlmostEqual(b - a, 0.02, places=5)
        self.assertLessEqual(max(idades), K.idade_max + 1e-6)
        self.assertAlmostEqual(idades[-1], K.idade_max, places=5)
        self.assertAlmostEqual(idades[149], K.idade_max, places=5)  # já parada no passo 150

    def test_ruido_de_posicao_tem_sigma_0_5_cm_em_10000_amostras(self):
        # Teste E2: ruído de posição com σ 0,5 cm medido em 10 000 amostras, a 10%
        n = 10_000
        est = EstimadorCaixa(n, DEV, K, torch.Generator().manual_seed(0))
        odo = Odometria(n, DEV, K, None)
        ids = torch.arange(n)
        est.reinicia(ids)
        odo.reinicia(ids, torch.zeros(n, 3), Q_ID.expand(n, 4).clone())
        verdade = torch.tensor([0.5, 0.2, 0.3], dtype=F).expand(n, 3).clone()
        vis = torch.ones(n, dtype=torch.bool)
        meia = torch.full((n,), 0.10, dtype=F)
        for k in range(6):  # leituras chegam depois da latência; a última já é uma leitura ruidosa
            est.passo(0.02 * k, verdade, Q_ID.expand(n, 4), meia, vis, magnitude(sigma_pos=0.005))
        erro = est.saida_b(odo, Q_ID.expand(n, 4))[:, :3] - verdade
        for eixo in (0, 1):
            self.assertAlmostEqual(float(erro[:, eixo].std()), 0.005, delta=0.0005)


class TestAlvoNoFrameDaBase(unittest.TestCase):
    def test_alvo_girado_pelo_rumo_sem_ruido(self):
        # Teste E2: alvo_no_frame_da_base com Odometria sem ruído (gerador None) devolve o alvo girado pelo rumo
        n, psi = 4, 0.5
        odo = Odometria(n, DEV, K, None)
        pos = torch.tensor([1.0, -0.5, 0.0], dtype=F).expand(n, 3).clone()
        odo.reinicia(torch.arange(n), pos, quat_de_yaw(torch.full((n,), psi, dtype=F)))
        alvo = torch.tensor([2.0, 0.3, 0.2], dtype=F).expand(n, 3).clone()
        out = alvo_no_frame_da_base(alvo, odo)
        c, s = math.cos(psi), math.sin(psi)
        dx, dy = 1.0, 0.8  # alvo − pos no mundo
        esperado = torch.tensor([c * dx + s * dy, -s * dx + c * dy, 0.2], dtype=F)
        self.assertLess(float((out - esperado).abs().max()), 1e-5)


class TestIdadeDepoisDaLeitura(unittest.TestCase):
    def test_idade_cresce_0_02_por_passo_depois_de_leitura_valida(self):
        # M9 (revisão E11): depois de uma leitura válida entregue, a idade vale t − instante da leitura e cresce
        # 0,02 s por passo, até idade_max. Lê-se a idade logo após a entrega, não só sem leitura desde o reset.
        est = EstimadorCaixa(1, DEV, K, None)
        odo = Odometria(1, DEV, K, None)
        ids = torch.arange(1)
        est.reinicia(ids)
        odo.reinicia(ids, POS_ZERO, Q_ID)
        vis = torch.ones(1, dtype=torch.bool)
        nao_vis = torch.zeros(1, dtype=torch.bool)
        for k in range(20):  # leituras válidas até t = 0,38 s
            est.passo(0.02 * k, POS_ZERO, Q_ID, MEIA, vis, magnitude())
        idades = []
        for k in range(20, 220):  # sem leitura nova; a última amostra válida já foi entregue em ~0,42 s
            est.passo(0.02 * k, POS_ZERO, Q_ID, MEIA, nao_vis, magnitude())
            idades.append(float(est.saida_b(odo, Q_ID)[0, 7]))
        # a leitura entregue dá idade pequena (não idade_max): a idade veio da leitura, não do reset
        self.assertLess(idades[3], 0.15)
        for a, b in zip(idades[3:], idades[4:]):
            if b < K.idade_max - 1e-6:  # o passo que cruza idade_max é cortado: não cresce 0,02
                self.assertAlmostEqual(b - a, 0.02, places=5)
        self.assertAlmostEqual(idades[-1], K.idade_max, places=5)


def palmas_em(meio, graus=0.0):
    """Palmas a 20 cm do meio, com o eixo girado `graus` em z a partir de +y: (1, 2, 3)."""
    a = math.radians(90.0 + graus)
    d = torch.tensor([math.cos(a), math.sin(a), 0.0], dtype=F) * 0.2
    m = torch.tensor(meio, dtype=F)
    return torch.stack([m - d, m + d])[None]


def passo_k(est, k, caixa, vis, palmas):
    """Um passo de 20 ms (t = 0,02·k), sem ruído."""
    est.passo(0.02 * k, caixa, Q_ID, MEIA, torch.tensor([vis]), magnitude(), palmas_w=palmas)


class TestCaixaPresaAsPalmas(unittest.TestCase):
    # D1 (dono, 30/10): sem leitura, a caixa presa às palmas anda com o meio e gira com o rumo do eixo delas.
    # Leitura até k=20 (t≈0,4 s, entregue até ≈0,44 s); palmas paradas até k=25; depois andam 10 cm em x (k=50).

    def test_presa_a_5_cm_anda_com_o_meio_e_idade_cresce(self):
        est = EstimadorCaixa(1, DEV, K, None)
        odo = Odometria(1, DEV, K, None)
        odo.reinicia(torch.arange(1), POS_ZERO, Q_ID)
        caixa = torch.tensor([[0.0, 0.05, 0.0]], dtype=F)
        idades = {}
        for k in range(51):
            pm = palmas_em((0.1 * min(1.0, max(0.0, (k - 25) / 25)), 0.0, 0.0))
            passo_k(est, k, caixa, k <= 20, pm)
            idades[k] = float(est.saida_b(odo, Q_ID)[0, 7])
        s = est.saida_b(odo, Q_ID)[0]
        self.assertAlmostEqual(float(s[0]), 0.10, delta=1e-5)
        self.assertAlmostEqual(float(s[1]), 0.05, delta=1e-5)
        self.assertAlmostEqual(idades[50] - idades[26], 0.02 * 24, places=4)

    def test_caixa_a_1_m_fica_parada(self):
        est = EstimadorCaixa(1, DEV, K, None)
        odo = Odometria(1, DEV, K, None)
        odo.reinicia(torch.arange(1), POS_ZERO, Q_ID)
        caixa = torch.tensor([[1.0, 0.0, 0.0]], dtype=F)
        for k in range(51):
            pm = palmas_em((0.1 * min(1.0, max(0.0, (k - 25) / 25)), 0.0, 0.0))
            passo_k(est, k, caixa, k <= 20, pm)
        s = est.saida_b(odo, Q_ID)[0]
        self.assertAlmostEqual(float(s[0]), 1.0, delta=1e-5)
        self.assertAlmostEqual(float(s[1]), 0.0, delta=1e-5)

    def test_presa_sem_leitura_gira_com_o_eixo_das_palmas(self):
        est = EstimadorCaixa(1, DEV, K, None)
        odo = Odometria(1, DEV, K, None)
        odo.reinicia(torch.arange(1), POS_ZERO, Q_ID)
        meio = (0.2, 0.1, 0.0)
        off = (0.03, 0.04, 0.0)  # |off| = 0,05 < meia + presa_margem
        caixa = torch.tensor([[meio[0] + off[0], meio[1] + off[1], 0.0]], dtype=F)
        for k in range(51):
            passo_k(est, k, caixa, k <= 20, palmas_em(meio, 0.0 if k <= 25 else 30.0))
        s = est.saida_b(odo, Q_ID)[0]
        c, sn = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
        esperado = (meio[0] + c * off[0] - sn * off[1], meio[1] + sn * off[0] + c * off[1])
        self.assertAlmostEqual(float(s[0]), esperado[0], delta=1e-4)
        self.assertAlmostEqual(float(s[1]), esperado[1], delta=1e-4)
        self.assertAlmostEqual(float(s[5]), math.radians(30.0), delta=1e-4)

    def test_palmas_none_fica_parada(self):
        est = EstimadorCaixa(1, DEV, K, None)
        odo = Odometria(1, DEV, K, None)
        odo.reinicia(torch.arange(1), POS_ZERO, Q_ID)
        caixa = torch.tensor([[0.0, 0.05, 0.0]], dtype=F)
        for k in range(51):
            pm = palmas_em((0.1 * min(1.0, max(0.0, (k - 25) / 25)), 0.0, 0.0))
            passo_k(est, k, caixa, k <= 20, None)
        s = est.saida_b(odo, Q_ID)[0]
        self.assertAlmostEqual(float(s[0]), 0.0, delta=1e-5)
        self.assertAlmostEqual(float(s[1]), 0.05, delta=1e-5)

    def test_presa_com_nova_leitura_volta_a_ser_a_leitura(self):
        est = EstimadorCaixa(1, DEV, K, None)
        odo = Odometria(1, DEV, K, None)
        odo.reinicia(torch.arange(1), POS_ZERO, Q_ID)
        caixa = torch.tensor([[0.0, 0.05, 0.0]], dtype=F)
        for k in range(51):
            pm = palmas_em((0.1 * min(1.0, max(0.0, (k - 25) / 25)), 0.0, 0.0))
            passo_k(est, k, caixa, k <= 20, pm)
        nova = torch.tensor([[0.3, 0.05, 0.0]], dtype=F)
        for k in range(51, 71):  # caixa vista de novo em x = 0,3 m (k = 51..60); a leitura nova chega até ≈1,23 s
            passo_k(est, k, nova, k <= 60, palmas_em((0.1, 0.0, 0.0)))
        s = est.saida_b(odo, Q_ID)[0]
        self.assertAlmostEqual(float(s[0]), 0.3, delta=1e-5)
        self.assertAlmostEqual(float(s[1]), 0.05, delta=1e-5)


class TestDoFrameDaBase(unittest.TestCase):
    def test_do_frame_da_base_inverte_alvo_no_frame_da_base(self):
        # S5a (revisão E11): do_frame_da_base(alvo_no_frame_da_base(p, odo), odo) = p, com rumo e posição não nulos.
        n, psi = 4, 0.7
        odo = Odometria(n, DEV, K, None)
        odo.reinicia(torch.arange(n), torch.tensor([0.3, -1.2, 0.1], dtype=F).expand(n, 3).clone(),
                     quat_de_yaw(torch.full((n,), psi, dtype=F)))
        p = torch.tensor([[2.0, 0.3, 0.2], [-1.0, 0.5, 1.1], [0.0, 0.0, 0.0], [3.0, -2.0, 0.6]], dtype=F)
        volta = do_frame_da_base(alvo_no_frame_da_base(p, odo), odo)
        self.assertLess(float((volta - p).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
