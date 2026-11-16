"""Teste do instante do empurrão em `eventos.empurrao` (defeito (b) da triagem do smoke, 30/10; D7; PD-15).

O buffer `v3_t_empurrao` é float32 (`curriculo.py:75`); a conta de `episode_length_buf` sai em float64. Escrever
float64 num buffer float32 quebra o `index_put` no primeiro empurrão. Env falso com `types.SimpleNamespace`.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from g1_v3 import eventos


class Empurrao(unittest.TestCase):
    def test_instante_grava_no_dtype_do_buffer(self):
        # antes do conserto: RuntimeError (index put Float × Double); depois: segundos float32 desde o começo do episódio
        env = SimpleNamespace(v3_t_empurrao=torch.full((4,), float("-inf")),
                              episode_length_buf=torch.tensor([100, 0, 250, 7]), step_dt=0.02)
        ids = torch.tensor([0, 2])
        with mock.patch.object(eventos, "push_by_setting_velocity"):
            eventos.empurrao(env, ids, velocity_range={})
        self.assertEqual(env.v3_t_empurrao.dtype, torch.float32)
        self.assertTrue(torch.allclose(env.v3_t_empurrao[ids], torch.tensor([2.0, 5.0])))
        self.assertTrue(bool(torch.isinf(env.v3_t_empurrao[[1, 3]]).all()))  # os outros envs ficam como estavam

    def test_empurrao_nao_muda_a_fase_do_relogio(self):
        # 16/11, R: a fase é o tempo do episódio sobre 0,8 s, sem recomeço no empurrão
        env = SimpleNamespace(v3_t_empurrao=torch.full((4,), float("-inf")),
                              episode_length_buf=torch.zeros(4, dtype=torch.long), step_dt=0.02,
                              v3_fase_marcha=torch.full((4,), 0.6))
        with mock.patch.object(eventos, "push_by_setting_velocity"):
            eventos.empurrao(env, torch.tensor([0, 2]), velocity_range={})
        self.assertTrue(torch.allclose(env.v3_fase_marcha, torch.full((4,), 0.6)))


if __name__ == "__main__":
    unittest.main()
