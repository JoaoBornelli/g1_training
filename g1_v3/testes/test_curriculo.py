"""Testes de g1_v3/curriculo.py (D4 do plano, §6.4, §6.6, §6.3 `Curriculo`; S §11). Escritos antes do código.

O env é falso (SimpleNamespace) com `num_envs`, `device` e `common_step_counter`, como no mjlab e no
`g1_limpo/curriculo.py`. A iteração tem `passos_por_iteracao` passos (knob, 24; D4 e §6.2).
Índices de TAREFAS (knobs): 0 ANDAR, 1 PEGAR, 2 CARREGAR, 3 BOTAR.
"""
import types
import unittest

import torch

from g1_v3 import curriculo
from g1_v3.knobs import ATIVO

PASSOS = ATIVO.balanceador.passos_por_iteracao  # 24 (§6.2)


def env_falso(tarefas, n=None):
    n = len(tarefas) if n is None else n
    env = types.SimpleNamespace(num_envs=n, device="cpu", common_step_counter=0)
    env.v3_tarefa = torch.tensor(tarefas, dtype=torch.long)
    return env


class CurriculoTest(unittest.TestCase):

    def test_mascara_por_tarefa(self):
        env = env_falso([0, 1, 2, 3])
        m = curriculo.mascara(env, ("ANDAR", "CARREGAR"))
        self.assertEqual(m.dtype, torch.bool)  # §6.6: bool, para o torch.where (nunca multiplicar)
        self.assertEqual(m.tolist(), [True, False, True, False])

    def test_garante_v3_celula_nasce_sem_celula(self):
        # S3: v3_celula nasce em −1 (§6.4: sem célula), como v3_alvo_idx. Em 0, o 1º reset fechava um
        # segmento fantasma no `reto` (comando.py:334-341).
        env = env_falso([0, 1, 2, 3])
        curriculo.garante_v3(env)
        self.assertEqual(env.v3_celula.tolist(), [-1, -1, -1, -1])

    def test_garante_v3_nao_recria_o_curriculo(self):
        env = env_falso([0, 1, 2, 3])
        curriculo.garante_v3(env)
        cur = env.v3_cur
        curriculo.garante_v3(env)
        self.assertIs(env.v3_cur, cur)

    def test_fecha_iteracao_uma_vez_por_iteracao_e_nao_por_reset(self):
        env = env_falso([0, 0, 0, 0])
        curriculo.garante_v3(env)
        todos = torch.arange(4)
        curriculo.v3(env, todos)  # primeira chamada fixa a iteração corrente
        it0 = env.v3_cur.it
        curriculo.v3(env, torch.tensor([1, 2]))  # reset no meio da iteração: não fecha
        self.assertEqual(env.v3_cur.it, it0)
        env.common_step_counter = PASSOS  # uma iteração depois
        curriculo.v3(env, todos)
        self.assertEqual(env.v3_cur.it, it0 + 1)
        curriculo.v3(env, todos)  # outro reset na mesma iteração
        self.assertEqual(env.v3_cur.it, it0 + 1)
        env.common_step_counter = 2 * PASSOS
        curriculo.v3(env, todos)
        self.assertEqual(env.v3_cur.it, it0 + 2)

    def test_garante_v3_cria_rumo_da_cena_zerado_por_env(self):
        # G4 / §6.4: v3_rumo_cena é (E,) float, 0 no começo (a lane G1 o escreve no posiciona_cena)
        env = env_falso([0, 1, 2, 3])
        curriculo.garante_v3(env)
        self.assertEqual(tuple(env.v3_rumo_cena.shape), (4,))
        self.assertEqual(env.v3_rumo_cena.dtype, torch.float32)
        self.assertTrue((env.v3_rumo_cena == 0).all())

    def test_log_traz_o_nivel_de_terreno_de_andar_e_carregar(self):
        # G4 / §6.7 e §6.3 "Fase G": o log ganha `nivel_terreno_<tarefa>` para ANDAR e CARREGAR, pelo
        # `cur.nivel_terreno`. O nível de CARREGAR é fixado em 3 para o log não ser o zero de sempre.
        env = env_falso([0, 0, 0, 0])
        curriculo.garante_v3(env)
        env.v3_cur._nivel_ter["CARREGAR"] = 3
        log = curriculo.v3(env, torch.arange(4))
        self.assertEqual(log["nivel_terreno_ANDAR"], 0)
        self.assertEqual(log["nivel_terreno_CARREGAR"], 3)


if __name__ == "__main__":
    unittest.main()
