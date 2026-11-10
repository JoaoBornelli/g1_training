"""Testes de g1_v3/potencial.py (B0 do plano, §6.3). Escritos antes do código: o ImportError é esperado.

Os números são os do plano (B0) e da spec (§8.2, §9.1, §9.2, §19). Os escalares (σ, Δ, γ, F_ref)
vêm do knobs, para o teste não carregar número próprio.
"""
import math
import unittest

import torch

from g1_v3 import potencial
from g1_v3.knobs import ATIVO

R = ATIVO.recompensa
C = ATIVO.comando
T = torch.float64


def t(*valores):
    return torch.tensor(list(valores), dtype=T)


class TestNucleo(unittest.TestCase):

    def test_em_d_igual_sigma_vale_e_menos_1(self):
        # [C20]: exp(−(d/σ)²); em d = σ o núcleo vale e⁻¹.
        self.assertAlmostEqual(potencial.nucleo(t(0.3), 0.3).item(), math.exp(-1), places=9)


class TestRampa(unittest.TestCase):

    def test_pontos_da_rampa_pela_inclinacao(self):
        # S §9.1: g = clip(1 − (x − x_lim)/Δ, 0, 1); Δ = 15° na inclinação (knobs.alvos).
        x_lim, delta = 15.0, ATIVO.alvos.delta_incl_graus
        self.assertAlmostEqual(potencial.rampa(t(x_lim), x_lim, delta).item(), 1.0, places=9)
        self.assertAlmostEqual(potencial.rampa(t(x_lim + delta), x_lim, delta).item(), 0.0, places=9)
        self.assertAlmostEqual(potencial.rampa(t(x_lim + delta / 2), x_lim, delta).item(), 0.5, places=9)
        # clip nas duas bordas: abaixo do limite vale 1, longe acima vale 0
        self.assertAlmostEqual(potencial.rampa(t(0.0), x_lim, delta).item(), 1.0, places=9)
        self.assertAlmostEqual(potencial.rampa(t(60.0), x_lim, delta).item(), 0.0, places=9)


class TestCustoAperto(unittest.TestCase):

    def test_valor_em_F_ref(self):
        # Caso 9 (decisão do dono, 30/10): linear, min(F)/F_ref. Com F = F_ref vale 1,0.
        v = potencial.custo_aperto(t(49.0), t(49.0), R.F_ref)
        self.assertAlmostEqual(v.item(), 1.0, delta=1e-6)

    def test_valor_e_derivada_em_117_5_N(self):
        # Caso 9: 117,5/49 = 2,398; derivada constante 1/49 = 0,0204 por N, igual em 49 e em 117,5 N
        # (a forma 1 − exp saturava: 0,0019 por N em 117,5 N, canal morto).
        v = potencial.custo_aperto(t(117.5), t(117.5), R.F_ref)
        self.assertAlmostEqual(v.item(), 117.5 / 49.0, delta=1e-5)
        h = 0.5
        for f0 in (49.0, 117.5):
            mais = potencial.custo_aperto(t(f0 + h), t(f0 + h), R.F_ref)
            menos = potencial.custo_aperto(t(f0 - h), t(f0 - h), R.F_ref)
            deriv = ((mais - menos) / (2 * h)).item()
            self.assertAlmostEqual(deriv, 1.0 / 49.0, delta=1e-5)

    def test_usa_a_face_mais_fraca(self):
        # Caso 9: o custo é pelo mínimo das duas faces, e a face mais forte não entra.
        v = potencial.custo_aperto(t(60.0), t(100.0), 49.0)
        self.assertAlmostEqual(v.item(), 60.0 / 49.0, delta=1e-6)
        v = potencial.custo_aperto(t(100.0), t(60.0), 49.0)
        self.assertAlmostEqual(v.item(), 60.0 / 49.0, delta=1e-6)


class TestFatorPouso(unittest.TestCase):

    def test_pouso_a_0_9_ms_com_fundo_na_laje(self):
        # S §9.1 (A10): exp(−(v/σ)²) com σ 1,0 m/s e fundo a 2 cm (≤ 5 cm); 0,9 m/s dá 0,44 (0,445 ± 0,001).
        sigma, altura, inicio = R.sigma_pouso[0], R.altura_pouso, R.altura_pouso_inicio
        v = potencial.fator_pouso(t(0.9), t(0.02), sigma, altura, inicio)
        self.assertAlmostEqual(v.item(), 0.445, delta=0.001)

    def test_fora_da_altura_do_pouso_vale_1(self):
        # S §9.1 (D3): com o fundo a 10 cm (= altura_inicio) o fator ainda não atua.
        sigma, altura, inicio = R.sigma_pouso[0], R.altura_pouso, R.altura_pouso_inicio
        v = potencial.fator_pouso(t(0.9), t(0.10), sigma, altura, inicio)
        self.assertAlmostEqual(v.item(), 1.0, places=9)

    def test_rampa_entre_10_e_5_cm_meio_caminho_a_7_5_cm(self):
        # S §9.1 (D3): entre 10 cm e 5 cm o fator é gradual. Com fundo a 7,5 cm, s = 0,5:
        # 1 − 0,5·(1 − e^−0,81) = 0,7225 (± 0,001).
        sigma, altura, inicio = R.sigma_pouso[0], R.altura_pouso, R.altura_pouso_inicio
        v = potencial.fator_pouso(t(0.9), t(0.075), sigma, altura, inicio)
        self.assertAlmostEqual(v.item(), 0.7225, delta=0.001)

    def test_acima_de_10_cm_vale_1_exato(self):
        # S §9.1 (D3): fundo a 12 cm, acima de altura_inicio: s = 0 e o fator é 1.0 exato.
        sigma, altura, inicio = R.sigma_pouso[0], R.altura_pouso, R.altura_pouso_inicio
        v = potencial.fator_pouso(t(0.9), t(0.12), sigma, altura, inicio)
        self.assertEqual(v.item(), 1.0)

    def test_continuidade_na_altura_inicio(self):
        # S §9.1 (D3): sem degrau. Com fundo 1e−6 abaixo de altura_inicio, o fator fica em 1 ± 1e−4.
        sigma, altura, inicio = R.sigma_pouso[0], R.altura_pouso, R.altura_pouso_inicio
        v = potencial.fator_pouso(t(0.9), t(inicio - 1e-6), sigma, altura, inicio)
        self.assertAlmostEqual(v.item(), 1.0, delta=1e-4)


class TestExcessoDeJunta(unittest.TestCase):

    def test_zero_a_90_por_cento_meio_a_97_5_meio_a_1_no_batente(self):
        # S §9.2: limite mole a 0,95 do curso; excesso = relu(frac − 0,95)/0,05.
        # Uma junta com centro 0 e meio curso 1: frac = |q|.
        q = torch.tensor([[0.90], [0.975], [1.0]], dtype=T)
        out = potencial.excesso_de_junta(q, t(0.0), t(1.0), R.limite_mole).reshape(-1)
        self.assertAlmostEqual(out[0].item(), 0.0, places=9)
        self.assertAlmostEqual(out[1].item(), 0.5, places=9)
        self.assertAlmostEqual(out[2].item(), 1.0, places=9)

    def test_soma_entre_juntas(self):
        # S §9.2: o custo é Σ_j excesso_j; duas juntas no batente somam 2 (aqui 1 + 0,5).
        q = torch.tensor([[1.0, 0.975]], dtype=T)
        out = potencial.excesso_de_junta(q, t(0.0, 0.0), t(1.0, 1.0), R.limite_mole).reshape(-1)
        self.assertAlmostEqual(out[0].item(), 1.5, places=9)


class TestCurso(unittest.TestCase):

    def test_centro_0_meio_1_e_o_valor_absoluto(self):
        # §12.4 H1 (P13): potencial.curso(q, 0, 1) = |q|
        q = torch.tensor([[-0.5, 0.0, 0.5, 1.0]], dtype=T)
        out = potencial.curso(q, t(0.0, 0.0, 0.0, 0.0), t(1.0, 1.0, 1.0, 1.0))
        self.assertTrue(torch.allclose(out, q.abs(), atol=1e-9))

    def test_um_e_um_virgula_zero_um_no_batente(self):
        # §12.4 H1: centro + 1,01·meio → 1,01; o batente do modelo é 1 (|q − c| = meio)
        centro, meio = t(0.3), t(0.5)
        q = torch.tensor([[0.3 + 1.01 * 0.5], [0.3 + 0.5]], dtype=T)
        out = potencial.curso(q, centro, meio).reshape(-1)
        self.assertAlmostEqual(out[0].item(), 1.01, places=9)
        self.assertAlmostEqual(out[1].item(), 1.0, places=9)

    def test_excesso_de_junta_segue_igual_pela_funcao(self):
        # §12.4 H1: `excesso_de_junta` passa a chamar `curso`; o resultado não muda (o teste do batente acima)
        q = torch.tensor([[0.975, 1.0]], dtype=T)
        a = potencial.excesso_de_junta(q, t(0.0, 0.0), t(1.0, 1.0), R.limite_mole)
        b = (torch.relu(potencial.curso(q, t(0.0, 0.0), t(1.0, 1.0)) - R.limite_mole).sum(dim=-1)
             / (1.0 - R.limite_mole))
        self.assertTrue(torch.allclose(a, b, atol=1e-12))


class TestRastreioNormalizado(unittest.TestCase):
    """S §19, conflito 5; PD-49. A função recebe erro² e v² (plano §6.3)."""

    def _r(self, erro, c, v):
        return potencial.rastreio_normalizado(
            t(erro * erro), t(c), t(v * v), R.sigma_rastreio, R.sigma_zero_lin,
            C.limiar_canal, R.piso_rastreio)

    def test_seguindo_o_comando_vale_1(self):
        # c 0,5 e erro 0: (1 − e⁻¹)/(1 − e⁻¹) = 1.
        self.assertAlmostEqual(self._r(0.0, 0.5, 0.5).item(), 1.0, places=9)

    def test_parado_com_comando_vale_0(self):
        # erro = c (o robô parado): numerador zero.
        self.assertAlmostEqual(self._r(0.5, 0.5, 0.0).item(), 0.0, places=9)

    def test_metade_da_velocidade_com_comando_pequeno(self):
        # c 0,05 e v = c/2 → erro = 0,025. Dá 0,749; o plano pede 0,75 ± 0,01.
        self.assertAlmostEqual(self._r(0.025, 0.05, 0.025).item(), 0.75, delta=0.01)

    def test_canal_sem_comando_parado_vale_0(self):
        # c 0,01 < limiar 0,02 e v 0: exp(0) − 1 = 0.
        self.assertAlmostEqual(self._r(0.0, 0.01, 0.0).item(), 0.0, places=9)

    def test_canal_sem_comando_com_deslize_pequeno(self):
        # PD-49: c 0,01, v 0,05, σ₀ 0,10 → exp(−0,25) − 1 = −0,221.
        self.assertAlmostEqual(self._r(0.0, 0.01, 0.05).item(), -0.221, delta=0.001)

    def test_canal_sem_comando_com_deslize_grande(self):
        # PD-49: c 0,01, v 0,5, σ₀ 0,10 → exp(−25) − 1 = −1,000.
        self.assertAlmostEqual(self._r(0.0, 0.01, 0.5).item(), -1.0, delta=0.001)

    def test_piso_menos_um_e_respeitado(self):
        # c 0,02 (no limite do canal) e erro grande: a fórmula cai abaixo de −1 e o piso corta.
        v = self._r(1.0, 0.02, 0.0)
        self.assertAlmostEqual(v.item(), R.piso_rastreio, places=9)


class TestLAlvo(unittest.TestCase):

    def test_comprimento_do_passo_pelos_quatro_pontos(self):
        # S §19, conflito 6: |v|·clamp(2·L_min/|v|, 0,8, 1,4)/2, com L_min 0,20.
        # 0,25 → 0,175 (teto 1,4); 0,5 → 0,20 (0,8); 1,0 → 0,40; 1,5 → 0,60.
        casos = ((0.25, 0.175), (0.5, 0.20), (1.0, 0.40), (1.5, 0.60))
        for v, esperado in casos:
            with self.subTest(v=v):
                out = potencial.L_alvo(t(v), R.L_min, R.fator_T)
                self.assertAlmostEqual(out.item(), esperado, places=6)


if __name__ == "__main__":
    unittest.main()
