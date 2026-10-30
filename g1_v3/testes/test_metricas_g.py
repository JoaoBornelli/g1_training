"""Testes de g1_v3/metricas.py da fase G (G4; §6.3 "Fase G", viab. D10): `fracao_fora_da_plataforma`.

Env falso com `types.SimpleNamespace`. `scene` é um dict (como no mjlab, `env.scene["robot"]`) com o atributo
`env_origins`, como `env.scene.env_origins`. O terreno é o `scene["terrain"].terrain_types` (coluna > 0 é
episódio de terreno). "Fora" é o chão sob a pelve (z da pelve − `altura_pelve.heights[:, 0]`) a mais de 1 cm do z
da origem (G7 P1).

Ordem do mjlab (`envs/manager_based_rl_env.py`, `managers/metrics_manager.py`): num passo, `compute()` chama o
`__call__` e só depois o `reset(env_ids)` roda; o reset do manager lê o valor "last" antes do `func.reset`. Os
testes seguem essa ordem: passos → `reset` → `__call__`. O valor por episódio sai no `reset`, em
`extras["log"]["Episodio/fracao_fora_da_plataforma"]`; o `__call__` devolve a flag "teve passo de evento".
"""
import types
import unittest

import torch

from g1_v3 import metricas
from g1_v3.knobs import FATIAS, TAREFAS

CARREGAR = TAREFAS.index("CARREGAR")
ANDAR = TAREFAS.index("ANDAR")
FATIA_B = FATIAS.index("B")
PELVE = 0.8  # z da pelve sobre o plano (m)
CHAVE = "Episodio/fracao_fora_da_plataforma"


class Cena(dict):
    pass


def env_falso(tarefa, fatia, terreno):
    """Env com `n` envs. A base (pelve) é `env.scene["robot"].data.root_link_pos_w`; a altura é `altura_pelve`."""
    n = len(tarefa)
    robo = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=torch.zeros(n, 3)))
    altura = types.SimpleNamespace(data=types.SimpleNamespace(heights=torch.zeros(n, 4)))
    cena = Cena(robot=robo, terrain=types.SimpleNamespace(terrain_types=torch.tensor(terreno)), altura_pelve=altura)
    cena.env_origins = torch.zeros(n, 3)
    return types.SimpleNamespace(num_envs=n, device="cpu", scene=cena, extras={"log": {}},
                                 v3_tarefa=torch.tensor(tarefa), v3_fatia=torch.tensor(fatia))


def pisa(env, chao):
    """Pelve a PELVE sobre o `chao` (m) de cada env: o `altura_pelve` mede a distância da pelve ao chão."""
    chao = torch.tensor(chao, dtype=torch.float)
    env.scene["robot"].data.root_link_pos_w[:, 2] = PELVE
    env.scene["altura_pelve"].data.heights[:, 0] = PELVE - chao


class MetricasGTest(unittest.TestCase):

    def test_pelve_sobre_o_plano_nao_conta(self):
        # Chão = origem, base longe do centro no xy: o critério é o chão, e não a distância no xy (S1 a).
        env = env_falso(tarefa=[CARREGAR], fatia=[FATIA_B], terreno=[1])
        env.scene["robot"].data.root_link_pos_w[:, 0] = 1.5
        m = metricas.fracao_fora_da_plataforma(None, env)
        pisa(env, [0.0])
        m(env)
        m.reset(torch.tensor([0]))
        self.assertAlmostEqual(float(env.extras["log"][CHAVE][0]), 0.0, places=6)

    def test_pelve_sobre_o_primeiro_degrau_conta(self):
        # Chão = origem − 0,15 m (o primeiro degrau): fora em todo passo do episódio.
        env = env_falso(tarefa=[CARREGAR], fatia=[FATIA_B], terreno=[1])
        m = metricas.fracao_fora_da_plataforma(None, env)
        pisa(env, [-0.15])
        m(env)
        m.reset(torch.tensor([0]))
        self.assertAlmostEqual(float(env.extras["log"][CHAVE][0]), 1.0, places=6)

    def test_log_e_a_media_dos_episodios_que_fecham(self):
        # env 0: fora em 1 de 5 passos → 0,2. env 1: fora em 3 de 5 → 0,6. env 2: ANDAR, sem evento, fora em
        # todo passo: não entra (se entrasse, a média seria 0,27). O log é [0,2; 0,6]: a média por episódio é 0,4.
        env = env_falso(tarefa=[CARREGAR, CARREGAR, ANDAR], fatia=[FATIA_B, FATIA_B, FATIA_B],
                        terreno=[1, 1, 1])
        m = metricas.fracao_fora_da_plataforma(None, env)
        chao_por_passo = [[-0.15, 0.0, -0.15],
                          [0.0, -0.15, -0.15],
                          [0.0, -0.15, -0.15],
                          [0.0, -0.15, -0.15],
                          [0.0, 0.0, -0.15]]
        for chao in chao_por_passo:
            pisa(env, chao)
            out = m(env)
        self.assertTrue(torch.equal(out, torch.tensor([1.0, 1.0, 0.0])), out)  # flag: teve evento
        m.reset(torch.arange(3))
        self.assertTrue(torch.allclose(env.extras["log"][CHAVE], torch.tensor([0.2, 0.6])),
                        env.extras["log"][CHAVE])

    def test_lote_sem_evento_nao_loga(self):
        # só ANDAR no lote: nenhum episódio com evento, e a chave não sai
        env = env_falso(tarefa=[ANDAR], fatia=[FATIA_B], terreno=[1])
        m = metricas.fracao_fora_da_plataforma(None, env)
        pisa(env, [-0.15])
        m(env)
        m.reset(torch.tensor([0]))
        self.assertNotIn(CHAVE, env.extras["log"])


if __name__ == "__main__":
    unittest.main()
