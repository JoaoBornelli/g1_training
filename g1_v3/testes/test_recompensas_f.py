"""Testes do pouso devagar e do invariante depois do apoio no BOTAR (F4 do plano; §6.3 "Fase F: núcleos puros
(29/10)", §12.2 F4; S §9.1; D3 da Parte 0). Escritos antes do código: a F4 não cria núcleo novo, então estes casos
passam hoje (são as peças que o `progresso` do BOTAR vai usar, com `fator_pouso`, `portoes_do_alvo` e
`phi_do_alvo`, já na fase E). Estes casos são a guarda da F4: não podem quebrar quando ela entrar.

Convenções: tensores sintéticos, E = 1, sem env. `a` é a meia aresta da caixa (E,); `sigma` é o σ do degrau 0,
`Recompensa.sigma_pouso[0]` = 1,0 m/s.
"""
import math
import unittest

import torch

from g1_v3 import potencial, recompensas
from g1_v3.knobs import ATIVO

A = ATIVO.alvos
B2, B3 = 6, 7
R = ATIVO.recompensa


def t(x, dtype=torch.float32):
    return torch.tensor([x], dtype=dtype)


class PousoDevagar(unittest.TestCase):
    def test_pouso_a_0_9_m_s_abaixo_da_altura_da_0_44(self):
        # Teste (F4): pousar a 0,9 m/s com o fundo abaixo de altura_pouso dá fator 0,44 (± 0,01).
        fator = potencial.fator_pouso(t(0.9), t(0.03), R.sigma_pouso[0], R.altura_pouso, R.altura_pouso_inicio)
        self.assertAlmostEqual(fator.item(), 0.44, delta=0.01)


class InvarianteDepoisDoApoio(unittest.TestCase):
    def test_deslocamento_de_3_cm_depois_do_apoio_da_meio(self):
        # Teste (F4): no B3, deslocar 3 cm depois de apoiar leva a rampa do `desloc` a 0,5 (rampa(0,03; 0,02; 0,02)).
        indice = t(B3, torch.long)
        zeros = torch.zeros(1)
        portoes = recompensas.portoes_do_alvo(
            indice, zeros, t(0.03), zeros, t(30.0), torch.ones(1, 2, dtype=torch.bool), zeros, zeros, zeros,
            torch.tensor([True]), t(0.1), A)
        self.assertAlmostEqual(portoes["desloc"].item(), 0.5, delta=1e-5)


class PhiDoB2(unittest.TestCase):
    def test_phi_no_b2_com_fator_0_44_e_0_44_vezes_o_nucleo(self):
        # Teste (F4): phi_do_alvo no B2 com o fator 0,44 dá 0,44 × o núcleo (portões todos 1 com as condições B2).
        indice = t(B2, torch.long)
        zeros = torch.zeros(1)
        portoes = recompensas.portoes_do_alvo(
            indice, zeros, zeros, zeros, t(30.0), torch.ones(1, 2, dtype=torch.bool), zeros, zeros, zeros,
            torch.tensor([True]), t(0.1), A)
        d, sigma = t(0.2), t(0.5)
        phi = recompensas.phi_do_alvo(indice, d, sigma, t(0.44), portoes)
        nucleo = math.exp(-((0.2 / 0.5) ** 2))
        self.assertAlmostEqual(phi.item(), 0.44 * nucleo, delta=1e-5)

    def test_phi_com_indice_menos_um_e_zero(self):
        # Teste (F4): −1 (nenhum alvo) dá Φ = 0 (PD-36, §6.6: nunca NaN).
        indice = t(-1, torch.long)
        zeros = torch.zeros(1)
        portoes = recompensas.portoes_do_alvo(
            indice, zeros, zeros, zeros, t(30.0), torch.ones(1, 2, dtype=torch.bool), zeros, zeros, zeros,
            torch.tensor([True]), t(0.1), A)
        phi = recompensas.phi_do_alvo(indice, t(0.2), t(0.5), t(0.44), portoes)
        self.assertEqual(phi.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
