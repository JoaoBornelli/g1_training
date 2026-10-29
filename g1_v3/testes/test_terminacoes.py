"""Testes dos núcleos de terminação da D6 (terminacoes.py). Escritos antes do código: o ImportError é esperado.

Os casos são os da linha "Teste" de D6 (plano §11), com os knobs `Terminacao` (S §10). Tensores sintéticos.
"""
import math
import unittest

import torch

from g1_v3 import terminacoes
from g1_v3.knobs import ATIVO, TAREFAS

K = ATIVO.terminacao
F = torch.float32
ANDAR = TAREFAS.index("ANDAR")  # 0
PEGAR = TAREFAS.index("PEGAR")  # 1


def t(*valores):
    return torch.tensor(valores, dtype=F)


def queda(tarefa, incl_rad, z_pelve, z_chao_pelve, z_joelho=0.5, z_chao_joelho=0.0):
    return terminacoes.n_queda(torch.tensor([tarefa]), t(incl_rad), t(z_pelve), t(z_chao_pelve),
                               t(z_joelho), t(z_chao_joelho), K).tolist()


class TestQueda(unittest.TestCase):
    def test_inclinacao_09_rad_no_andar_cai(self):
        # 0,9 rad > incl_marcha (0,8 rad, S §10).
        self.assertEqual(queda(ANDAR, 0.9, 0.78, 0.0), [True])

    def test_inclinacao_09_rad_no_pegar_nao_cai(self):
        # 0,9 rad = 51,6° < incl_manip_graus (70°): na manipulação a queda é só por inclinação de 70°.
        self.assertEqual(queda(PEGAR, 0.9, 0.78, 0.0), [False])
        self.assertLess(0.9 * 180 / math.pi, K.incl_manip_graus)

    def test_pelve_a_015_do_terreno_com_origem_do_subterreno_em_14_cai(self):
        # z_pelve_w = 1,55; chão do subterreno a 1,40 m: 0,15 m < base_min (0,20 m) → cai.
        self.assertEqual(queda(ANDAR, 0.0, 1.55, 1.40), [True])

    def test_mesma_pelve_com_chao_no_mundo_nao_cai(self):
        # Com z_chao_pelve = 0 (o mundo), a mesma pelve está a 1,55 m e não cai: a conta é do terreno.
        self.assertEqual(queda(ANDAR, 0.0, 1.55, 0.0), [False])


if __name__ == "__main__":
    unittest.main()
