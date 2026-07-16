"""Registra as tasks do projeto (skills: Stand, Lift, Place-futuro).

MESMO `experiment_name` (`g1_lifting_box`, ver rl_cfg.py) pra todas -> warm-start
entre skills via `--agent.resume --agent.load-run` (obs/ação shape idêntico,
contrato garantido em `base_env.py`). Task IDs preservados do pacote anterior
(`src/G1_lifting_box`) pra nenhum comando/notebook precisar mudar.
"""
from mjlab.tasks.registry import register_mjlab_task

from .rl_cfg import lift_box_ppo_runner_cfg
from .skills.lift.configs import ACTIVE as _LIFT_ACTIVE
from .skills.lift.env import build_lift_env
from .skills.stand.configs import BASELINE as _STAND_BASELINE, STEP_RECOVERY as _STAND_STEP
from .skills.stand.env import build_stand_env

# nome da task -> (builder, knobs ativos, run_name pro log)
_TASKS = {
    "Stand": (build_stand_env, _STAND_BASELINE, "stand"),
    "Stand-Step": (build_stand_env, _STAND_STEP, "stand_step"),
    "Lift": (build_lift_env, _LIFT_ACTIVE, "lift"),
}

for _name, (_builder, _knobs, _run_name) in _TASKS.items():
    register_mjlab_task(
        task_id=f"Mjlab-Lift-Box-Unitree-G1-{_name}",
        env_cfg=_builder(_knobs, play=False),
        play_env_cfg=_builder(_knobs, play=True),
        rl_cfg=lift_box_ppo_runner_cfg(run_name=_run_name),
    )
