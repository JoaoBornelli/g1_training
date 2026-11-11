"""Registro das tasks da v3 no mjlab (D10 do plano; §6.5, PD-1).

O registro acontece no IMPORT deste módulo: `import g1_v3.tarefa` deixa `Mjlab-G1-V3` e a
`Mjlab-G1-V3-SemPercepcao` disponíveis para o `mjlab.scripts.train` e o `play`. Os módulos puros
(knobs, alvos, potencial...) não importam este arquivo (PD-1).

⚠ `experiment_name` "g1_v3" é o que nomeia a pasta dos logs. O `g1_v3/container/entrada.sh` não passa
`--agent.experiment-name`, então a pasta depende deste valor. O `load_run` default é `.*`: sem o nome
próprio, um `--agent.resume` casaria com a run de outro pacote (mesmo motivo do `g1_limpo/train.py`).
"""
from __future__ import annotations

import dataclasses

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from g1_v3 import algoritmo
from g1_v3.env_cfg import make_env_cfg
from g1_v3.knobs import ATIVO, Knobs
from g1_v3.runner import RunnerV3

TASK_ID = "Mjlab-G1-V3"
# ⚠ Só para a corrida de custo do I5: sem o lidar, o `mapa` sai do ator (`observacoes.grupos`). Substitui a
# chave `PERCEPCAO` do notebook antigo.
TASK_ID_SEM_PERCEPCAO = "Mjlab-G1-V3-SemPercepcao"
EXPERIMENT = "g1_v3"


def rl_cfg():
    """O PPO do mjlab com a classe de algoritmo da v3 (`algoritmo.CAMINHO`, a vantagem normalizada por tarefa).

    ⚠ `obs_normalization` e `entropy_coef` são os do mjlab (True e 0,01), como no g1_limpo (dono, 11/11; antes: False e
    0 a 0,005). Medido na poc9: sem o normalizador o ator recebe a caixa com desvio 0,07 e a ação anterior com 0,82, e
    a marcha ficou no nível 0 da it 4600 à 11231 (`s_ANDAR_reto` 0,65 → 0,79). Os termos categóricos e o mapa ficam
    fora do normalizador (`runner._FIXOS`, `algoritmo.fixa_normalizador`). `num_steps_per_env` vem do
    `Balanceador.passos_por_iteracao`: é a iteração que o balanceador fecha (`v3.curriculo`).
    """
    cfg = unitree_g1_ppo_runner_cfg()
    cfg.experiment_name = EXPERIMENT
    cfg.algorithm.class_name = algoritmo.CAMINHO
    cfg.num_steps_per_env = ATIVO.balanceador.passos_por_iteracao
    return cfg


def _sem_percepcao(k: Knobs = ATIVO) -> Knobs:
    return dataclasses.replace(k, percepcao=dataclasses.replace(k.percepcao, ligada=False))


# O `runner_cls` NÃO é cosmético: sem ele o estado do currículo não vai para o checkpoint (ver `runner.py`).
register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=rl_cfg(),
    runner_cls=RunnerV3,
)

register_mjlab_task(
    task_id=TASK_ID_SEM_PERCEPCAO,
    env_cfg=make_env_cfg(_sem_percepcao()),
    play_env_cfg=make_env_cfg(_sem_percepcao(), play=True),
    rl_cfg=rl_cfg(),
    runner_cls=RunnerV3,
)
