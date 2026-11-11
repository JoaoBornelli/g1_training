"""Testes da fase H6 de g1_v3/curriculo.py (plano §12.4 H6; §6.7; P23; S §11).

O log do termo `v3` ganha `atraso_max` e `mu_min_pad` (do `dr()`, que ele já lia) e `degrau_global`.
Os degraus entram por `estado()` → edição → `carrega()`, como no `test_balanceador_h.py`. Escritos antes do código:
o log ainda não tem as três chaves (KeyError).
"""
import types
import unittest

import torch

from g1_v3 import curriculo
from g1_v3.knobs import ATIVO

TAREFAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")


def env_falso(tarefas, n=None):
    n = len(tarefas) if n is None else n
    env = types.SimpleNamespace(num_envs=n, device="cpu", common_step_counter=0)
    env.v3_tarefa = torch.tensor(tarefas, dtype=torch.long)
    return env


class CurriculoHTest(unittest.TestCase):

    def test_log_traz_degraus_degrau_global_atraso_e_mu(self):
        # §12.4 H6: degraus (ANDAR, PEGAR, CARREGAR, BOTAR) = (5, 3, 4, 2) entram pelo estado.
        # degrau_global = min = 2; g = 2 < 3, sem portão0: atraso_max no teto inicial (11/11: 0, era 1, com atraso_inicio (0, 0)); mu_min_pad = atrito_pad[0] = 0,6.
        env = env_falso([0, 1, 2, 3])
        curriculo.garante_v3(env)
        est = env.v3_cur.estado()
        for t, d in zip(TAREFAS, (5, 3, 4, 2)):
            est["tarefas"][t]["degrau"] = d
        env.v3_cur.carrega(est)
        log = curriculo.v3(env, torch.arange(4))
        self.assertEqual(log["degrau_ANDAR"], 5)
        self.assertEqual(log["degrau_PEGAR"], 3)
        self.assertEqual(log["degrau_CARREGAR"], 4)
        self.assertEqual(log["degrau_BOTAR"], 2)
        self.assertEqual(log["degrau_global"], 2)
        self.assertEqual(log["atraso_max"], 0)
        self.assertAlmostEqual(log["mu_min_pad"], 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
