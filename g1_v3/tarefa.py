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

    ⚠ `obs_normalization` False no ator e no crítico: a v3 não tem normalizador (o `algoritmo.py` não tem
    `_fixa_one_hot`; plano D9). `num_steps_per_env` vem do `Balanceador.passos_por_iteracao`: é a iteração
    que o balanceador fecha (`v3.curriculo`).
    """
    cfg = unitree_g1_ppo_runner_cfg()
    cfg.experiment_name = EXPERIMENT
    cfg.actor.obs_normalization = False
    cfg.critic.obs_normalization = False
    cfg.algorithm.class_name = algoritmo.CAMINHO
    # O std é aprendido: fica onde a vantagem (que o baixa) e a entropia (que o sobe) se anulam. Medido com a renda em
    # escada: 0,005 → +0,03/100 its (poc6, 0,38 → 0,73); 0 → −0,013/100 its (poc7, 0,37 → 0,24); 0,003 → +0,021/100 its
    # (retomada da 9600) e +0,052/100 its (poc8, 0,24 → 0,47). A pega leve perde o contato com o std acima de ~0,3
    # (duas palmas 72 → 38 % na poc8). 0 de novo (11/11) para isolar o portão de firmeza (`knobs.margem_pega`) a partir
    # do model_9300, que treinou com 0. Subir só depois de a pega ficar firme. Antes: 0,01 na poc2 e no g1_limpo.
    cfg.algorithm.entropy_coef = 0.0
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
