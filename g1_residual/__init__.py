"""Registra a task do experimento residual. Pacote IRMÃO, não parte do multi-tarefa.

    import g1_residual        # o import é o que registra

`experiment_name` PRÓPRIO (`g1_residual_pegar`). Motivo igual ao do
`g1_multitask/__init__.py`: o `load_run` default é o regex `.*`, e a ação aqui tem
**49** números contra 29 do multi-tarefa. Nome compartilhado deixaria um
`--agent.resume True` casar com a run errada e tentar carregar uma cabeça de 29 numa
de 49 — `size mismatch` na saída do ator.

Mudar o espaço de ação é **Categoria C** pela tabela da §15: run do zero. Já era o
caso, porque isto é experimento separado, mas vale escrever para ninguém tentar
warm-start do checkpoint monolítico.
"""
from mjlab.tasks.registry import register_mjlab_task

from g1_multitask.runner import MultitaskRunner
from g1_training.rl_cfg import lift_box_ppo_runner_cfg

from .env_pegar import build_env_pegar

TASK_ID = "Mjlab-Residual-Pegar-Unitree-G1"
EXPERIMENT = "g1_residual_pegar"


def _rl_cfg():
    cfg = lift_box_ppo_runner_cfg(run_name="residual_pegar")
    cfg.experiment_name = EXPERIMENT
    return cfg


register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=build_env_pegar(play=False),
    play_env_cfg=build_env_pegar(play=True),
    rl_cfg=_rl_cfg(),
    # O `MultitaskRunner` continua servindo, e por dois motivos que não mudaram: ele
    # congela o normalizador nos 17 canais de comando, e ele faz o round-trip do
    # estado do currículo no checkpoint. A ação ser diferente não afeta nenhum dos
    # dois — o congelamento é sobre OBS, e o currículo é sobre `success_buf`.
    runner_cls=MultitaskRunner,
)
