"""Testes dos núcleos de terminação da D6 (terminacoes.py). Escritos antes do código: o ImportError é esperado.

Os casos são os da linha "Teste" de D6 (plano §11), com os knobs `Terminacao` (S §10). Tensores sintéticos.
"""
import math
import types
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


class TestFimDaCadeiaNaTravessia(unittest.TestCase):
    """27/11, T19 §13.3: o episódio de terreno do ANDAR acaba (`time_out`) quando a base passa de `dist_travessia` da origem da célula."""

    def _fim(self, fatia, coluna, d, passo, max_ep=1000, fim=False):
        origem = torch.tensor([[10.0, -5.0, 0.0]])
        base = origem + torch.tensor([[d * 0.6, d * 0.8, 0.3]])  # a distância no plano xy é `d`; o z não entra
        env = types.SimpleNamespace(
            episode_length_buf=torch.tensor([passo]), max_episode_length=max_ep, v3_fim=torch.tensor([fim]),
            v3_fatia=torch.tensor([fatia]),
            scene=_Cena(origem, torch.tensor([coluna]), base))
        return bool(terminacoes.fim_da_cadeia(env)[0])

    def test_andar_em_terreno_com_d_3_1_m_no_passo_10_termina(self):
        self.assertTrue(self._fim(fatia=0, coluna=2, d=3.1, passo=10))

    def test_d_2_9_m_nao_termina(self):
        self.assertFalse(self._fim(fatia=0, coluna=2, d=2.9, passo=10))

    def test_no_plano_nao_termina_mesmo_com_d_5_m(self):
        self.assertFalse(self._fim(fatia=0, coluna=0, d=5.0, passo=10))

    def test_fatia_b_com_coluna_de_terreno_nao_termina_com_d_5_m(self):
        self.assertFalse(self._fim(fatia=1, coluna=2, d=5.0, passo=10))

    def test_no_passo_2_nao_termina_com_d_5_m(self):
        # a guarda do mjlab: no reset a posição lida ainda é a velha
        self.assertFalse(self._fim(fatia=0, coluna=2, d=5.0, passo=2))

    def test_as_regras_de_hoje_seguem(self):
        self.assertTrue(self._fim(fatia=0, coluna=0, d=0.0, passo=1000, max_ep=1000))  # `max_episode_length`
        self.assertTrue(self._fim(fatia=0, coluna=0, d=0.0, passo=10, fim=True))  # `v3_fim`


class _Cena:
    # `env.scene["robot"].data.root_link_pos_w`, `env.scene["terrain"].terrain_types` e `env.scene.env_origins`, como a cena do mjlab
    def __init__(self, origem, coluna, base):
        self.env_origins = origem
        self._itens = {"robot": types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=base)),
                       "terrain": types.SimpleNamespace(terrain_types=coluna)}

    def __getitem__(self, nome):
        return self._itens[nome]


if __name__ == "__main__":
    unittest.main()
