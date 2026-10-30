"""Testes da lane H5 da fase H (`estimador.py`; plano §6.3 "Fase H", §12.4 H5; P21): vieses por episódio (posição,
meia aresta e orientação lognormal), a leitura de 1° por quatérnio e as rajadas de perda. Escritos antes do código.

Convenções: quaternion wxyz; comprimentos em metros; ângulos em graus nas tabelas. Tensores sintéticos em CPU;
nenhum env nem simulador.
"""
import dataclasses
import unittest

import torch

from g1_v3.estimador import EstimadorCaixa, Odometria
from g1_v3.knobs import ATIVO

K = ATIVO.estimador
DEV = "cpu"
F = torch.float32
Q_ID = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=F)
PASSOS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]  # 5 leituras de 30 Hz já entregues depois da latência de 33 ms


def _run(est, odo, n, instantes, pos, meia, mag_de_t=lambda t: {}, vis=True):
    # `passo` em cada instante (caixa parada em `pos`) e devolve a saída de `saida_b` depois do último passo
    visivel = torch.full((n,), vis, dtype=torch.bool)
    quat = Q_ID.expand(n, 4).clone()
    for t in instantes:
        est.passo(t, pos, quat, meia, visivel, mag_de_t(t))
    return est.saida_b(odo, quat)


class TestViesDePosicaoEMeia(unittest.TestCase):

    def setUp(self):
        self.n = 4000
        self.gerador = torch.Generator().manual_seed(0)
        self.est = EstimadorCaixa(self.n, DEV, K, self.gerador)
        self.odo = Odometria(self.n, DEV, K, None)
        self.ids = torch.arange(self.n)
        self.est.reinicia(self.ids)
        self.odo.reinicia(self.ids, torch.zeros(self.n, 3), Q_ID.expand(self.n, 4).clone())
        self.pos = torch.zeros(self.n, 3, dtype=F)
        self.meia = torch.full((self.n,), 0.10, dtype=F)

    def test_vies_de_posicao_uniforme_em_mais_ou_menos_2_cm(self):
        # Posição (caixa parada, σ 0): erro por eixo U[±0,02]; média ~0, desvio 0,02/√3 = 0,0115 m
        mag = {"vies_pos": 0.02}
        out = _run(self.est, self.odo, self.n, PASSOS, self.pos, self.meia, lambda t: mag)
        erro = out[:, :3]
        self.assertLessEqual(float(erro.abs().max()), 0.02 + 1e-6)
        self.assertLess(float(erro.mean(0).abs().max()), 0.001)
        self.assertAlmostEqual(float(erro.std()), 0.0115, delta=0.001)

    def test_vies_de_posicao_e_constante_no_episodio_e_muda_no_reinicia(self):
        mag = {"vies_pos": 0.02}
        out1 = _run(self.est, self.odo, self.n, PASSOS, self.pos, self.meia, lambda t: mag)
        out2 = _run(self.est, self.odo, self.n, [0.12, 0.14, 0.16], self.pos, self.meia, lambda t: mag)
        self.assertTrue(torch.allclose(out1[:, :3], out2[:, :3], atol=1e-6))  # mesmo viés nas leituras seguintes
        self.est.reinicia(self.ids)
        out3 = _run(self.est, self.odo, self.n, PASSOS, self.pos, self.meia, lambda t: mag)
        self.assertFalse(torch.allclose(out1[:, :3], out3[:, :3]))  # novo episódio, novo viés

    def test_vies_de_meia_aresta_uniforme_em_mais_ou_menos_2_cm(self):
        mag = {"vies_meia": 0.02}
        out = _run(self.est, self.odo, self.n, PASSOS, self.pos, self.meia, lambda t: mag)
        desvio = out[:, 6] - 0.10
        self.assertLessEqual(float(desvio.abs().max()), 0.02 + 1e-6)
        self.assertLess(abs(float(desvio.mean())), 0.001)
        self.assertAlmostEqual(float(desvio.std()), 0.0115, delta=0.001)

    def test_sem_gerador_o_vies_de_posicao_nao_muda_nada(self):
        est = EstimadorCaixa(3, DEV, K, None)
        odo = Odometria(3, DEV, K, None)
        est.reinicia(torch.arange(3))
        odo.reinicia(torch.arange(3), torch.zeros(3, 3), Q_ID.expand(3, 4).clone())
        out = _run(est, odo, 3, PASSOS, torch.zeros(3, 3, dtype=F), torch.full((3,), 0.10, dtype=F),
                   lambda t: {"vies_pos": 0.02, "vies_meia": 0.02})
        self.assertTrue(torch.allclose(out[:, :3], torch.zeros(3, 3)))
        self.assertTrue(torch.allclose(out[:, 6], torch.full((3,), 0.10)))


class TestViesDeOrientacao(unittest.TestCase):

    def setUp(self):
        self.n = 10_000
        self.est = EstimadorCaixa(self.n, DEV, K, torch.Generator().manual_seed(0))
        self.odo = Odometria(self.n, DEV, K, None)
        self.ids = torch.arange(self.n)
        self.est.reinicia(self.ids)
        self.odo.reinicia(self.ids, torch.zeros(self.n, 3), Q_ID.expand(self.n, 4).clone())
        self.pos = torch.zeros(self.n, 3, dtype=F)
        self.meia = torch.full((self.n,), 0.10, dtype=F)

    def _angulo_graus(self, out):
        return torch.rad2deg(out[:, 3:6].norm(dim=-1))

    def test_vies_de_orientacao_lognormal_com_mediana_5_3_e_p90_15_3(self):
        # P21a: θ = mediana·exp(σ_ln·z), σ_ln = ln(p90/mediana)/1,2816; o eixo é uniforme na esfera
        mag = {"ori_mediana_graus": 5.3, "ori_p90_graus": 15.3}
        out = _run(self.est, self.odo, self.n, PASSOS, self.pos, self.meia, lambda t: mag)
        ang = self._angulo_graus(out)
        self.assertAlmostEqual(float(torch.quantile(ang, 0.5)), 5.3, delta=0.2)
        self.assertAlmostEqual(float(torch.quantile(ang, 0.9)), 15.3, delta=0.8)

    def test_vies_de_orientacao_e_constante_no_episodio(self):
        mag = {"ori_mediana_graus": 5.3, "ori_p90_graus": 15.3}
        out1 = _run(self.est, self.odo, self.n, PASSOS, self.pos, self.meia, lambda t: mag)
        out2 = _run(self.est, self.odo, self.n, [0.12, 0.14, 0.16], self.pos, self.meia, lambda t: mag)
        self.assertTrue(torch.allclose(self._angulo_graus(out1), self._angulo_graus(out2), atol=1e-3))

    def test_so_a_leitura_1_grau_por_eixo_da_mediana_1_54_e_muda_a_cada_leitura(self):
        # P21b: χ₃ de σ 1° tem mediana 1,538°; cada leitura sorteia de novo
        mag = {"ori_leitura_graus": 1.0}
        out1 = _run(self.est, self.odo, self.n, PASSOS, self.pos, self.meia, lambda t: mag)
        self.assertAlmostEqual(float(torch.quantile(self._angulo_graus(out1), 0.5)), 1.54, delta=0.1)
        out2 = _run(self.est, self.odo, self.n, [0.12, 0.14, 0.16], self.pos, self.meia, lambda t: mag)
        self.assertFalse(torch.allclose(out1[:, 3:6], out2[:, 3:6]))

    def test_mediana_zero_da_erro_zero_e_sem_nan(self):
        for mag in ({"ori_mediana_graus": 0.0, "ori_p90_graus": 0.0}, {"ori_mediana_graus": 0.0, "ori_p90_graus": 15.3}):
            out = _run(self.est, self.odo, self.n, PASSOS, self.pos, self.meia, lambda t: mag)
            self.assertTrue(bool(torch.isfinite(out).all()))
            self.assertLess(float(self._angulo_graus(out).max()), 1e-6)


class TestRajadaDePerda(unittest.TestCase):
    # Gilbert-Elliott (P21e): 0,5 s fixos (rajada_s (0,5; 0,5)); taxa 10⁶/s só no passo de t = 0,02, 0 nos outros

    def _idades(self, gerador, taxa_no_passo):
        k = dataclasses.replace(K, rajada_s=(0.5, 0.5))
        est = EstimadorCaixa(1, DEV, k, gerador)
        odo = Odometria(1, DEV, k, None)
        est.reinicia(torch.arange(1))
        odo.reinicia(torch.arange(1), torch.zeros(1, 3), Q_ID.expand(1, 4).clone())
        quat = Q_ID.expand(1, 4).clone()
        vis = torch.ones(1, dtype=torch.bool)
        meia = torch.full((1,), 0.10, dtype=F)
        idades = {}
        for i in range(31):  # t = 0, 0,02, …, 0,60
            t = 0.02 * i
            mag = {"rajada_taxa": taxa_no_passo} if i == 1 else {}
            est.passo(t, torch.zeros(1, 3), quat, meia, vis, mag)
            idades[i] = float(est.saida_b(odo, quat)[0, 7])
        return idades[25], idades[30]  # t = 0,50 e t = 0,60

    def test_rajada_abre_no_passo_com_taxa_e_derruba_as_amostras_ate_0_52_s(self):
        idade_050, idade_060 = self._idades(torch.Generator().manual_seed(0), 1e6)
        self.assertGreaterEqual(idade_050, 0.48)  # sem amostra válida desde a rajada
        self.assertLess(idade_060, 0.1)  # a primeira válida é a de 0,533 s

    def test_sem_taxa_nao_ha_rajada(self):
        idade_050, _ = self._idades(torch.Generator().manual_seed(0), 0.0)
        self.assertLess(idade_050, 0.1)

    def test_sem_gerador_nao_ha_rajada(self):
        idade_050, _ = self._idades(None, 1e6)
        self.assertLess(idade_050, 0.1)


if __name__ == "__main__":
    unittest.main()
