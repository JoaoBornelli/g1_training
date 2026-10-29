"""Testes de g1_v3/metricas.py da fase G (G4; §6.3 "Fase G", viab. D10): `fracao_fora_da_plataforma`.

Env falso com `types.SimpleNamespace`. `scene` é um dict (como no mjlab, `env.scene["robot"]`) com o atributo
`env_origins`, como `env.scene.env_origins`. O terreno é o `scene["terrain"].terrain_types` (coluna > 0 é
episódio de terreno).
"""
import types
import unittest

import torch

from g1_v3 import metricas
from g1_v3.comando import FATIA_B
from g1_v3.knobs import TAREFAS

CARREGAR = TAREFAS.index("CARREGAR")
ANDAR = TAREFAS.index("ANDAR")


class Cena(dict):
    pass


def env_falso(tarefa, fatia, terreno):
    """Env com `n` envs. A posição da base é `env.scene["robot"].data.root_link_pos_w`, trocada pelo teste."""
    n = len(tarefa)
    robo = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=torch.zeros(n, 3)))
    cena = Cena(robot=robo, terrain=types.SimpleNamespace(terrain_types=torch.tensor(terreno)))
    cena.env_origins = torch.zeros(n, 3)
    return types.SimpleNamespace(num_envs=n, device="cpu", scene=cena,
                                 v3_tarefa=torch.tensor(tarefa), v3_fatia=torch.tensor(fatia))


class MetricasGTest(unittest.TestCase):

    def test_fracao_fora_da_plataforma_so_conta_carregar_de_terreno_b(self):
        # G4: 4 envs; só o env 0 é CARREGAR, fatia B, em terreno. A base dele fica fora do quadrado de 2 m
        # (meio lado 1 m) em 1 de 4 passos → 0,25. Os outros 3 não têm o evento e não entram na média (M5).
        env = env_falso(tarefa=[CARREGAR, CARREGAR, ANDAR, CARREGAR],
                        fatia=[FATIA_B, 0, FATIA_B, FATIA_B], terreno=[1, 0, 1, 0])
        m = metricas.fracao_fora_da_plataforma(None, env)
        passos = [[0.5, 0.0], [0.5, 0.0], [1.5, 0.0], [0.5, 0.0]]  # o env 0 sai do quadrado só no passo 3
        for xy in passos:
            env.scene["robot"].data.root_link_pos_w = torch.tensor([[*xy, 0.8], [1.5, 0.0, 0.8],
                                                                    [1.5, 0.0, 0.8], [1.5, 0.0, 0.8]])
            out = m(env)
        self.assertAlmostEqual(float(out.mean()), 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
