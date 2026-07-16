"""Smoke da task REGISTRADA (roda direto, do venv do projeto):

    python smoke.py

Confirma que o pacote registra as 3 skills (Stand/Stand-Step/Lift), que os
knobs de cada config batem, que a fundação (slip dos pés) está SEMPRE
presente independente da skill, e que os 3 envs instanciam + resetam + dão
1 step (exercitando os accessors de física usados nos rewards: push_force,
subtree_com, site_lin_vel_w).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

import g1_training  # noqa: F401  (o import dispara o register_mjlab_task)
from g1_training.skills.lift.configs import ACTIVE as lift_active
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

TASKS = sorted(t for t in list_tasks() if "Lift-Box" in t)
print("tasks registradas:", TASKS)
assert "Mjlab-Lift-Box-Unitree-G1-Stand" in TASKS
assert "Mjlab-Lift-Box-Unitree-G1-Stand-Step" in TASKS
assert "Mjlab-Lift-Box-Unitree-G1-Lift" in TASKS

stand_cfg = load_env_cfg("Mjlab-Lift-Box-Unitree-G1-Stand")
step_cfg = load_env_cfg("Mjlab-Lift-Box-Unitree-G1-Stand-Step")
lift_cfg = load_env_cfg("Mjlab-Lift-Box-Unitree-G1-Lift")

# --- FUNDAÇÃO: slip dos pés é SEMPRE-ON, em TODA skill (não é knob de config) ---
for name, cfg in (("Stand", stand_cfg), ("Stand-Step", step_cfg), ("Lift", lift_cfg)):
    assert "feet_slip" in cfg.rewards, f"{name}: feet_slip ausente (deveria ser fundação fixa)"
    assert cfg.rewards["feet_slip"].weight == -0.1, f"{name}: peso do feet_slip mudou"
print("OK: feet_slip presente e com o mesmo peso nas 3 skills")

# --- toggle de recompensa de TAREFA (só na Lift) ---
print("Stand rewards:", list(stand_cfg.rewards))
print("Stand-Step rewards:", list(step_cfg.rewards))
print("Lift rewards:", list(lift_cfg.rewards))
for t in ("reaching", "grasp", "lift", "sustain_precise", "com_balance", "table_contact"):
    assert t not in stand_cfg.rewards, f"{t} vazou no Stand"
    assert t not in step_cfg.rewards, f"{t} vazou no Stand-Step"
    assert t in lift_cfg.rewards, f"{t} ausente na Lift"

# --- postura escopada por skill ---
assert stand_cfg.rewards["posture"].weight == 0.5, "Stand: posture deveria ser 0.5"
assert step_cfg.rewards["posture"].weight == 0.5, "Stand-Step: posture deveria ser 0.5"
assert lift_cfg.rewards["posture"].weight == 0.25, "Lift: posture deveria ser 0.25"

# --- fundação NUNCA mexida (upright cheio; nunca mais baixar, ver knobs.py) ---
assert stand_cfg.rewards["upright"].weight == 1.0
assert lift_cfg.rewards["upright"].weight == 1.0

# --- anti-dinâmica afrouxada SÓ no Stand-Step ---
assert step_cfg.rewards["action_rate_l2"].weight == -0.03
assert stand_cfg.rewards["action_rate_l2"].weight == -0.1

# --- push mais forte só no Stand-Step (treino); push_force só lá também ---
assert tuple(step_cfg.events["push_robot"].params["velocity_range"]["x"]) == (-0.8, 0.8)
assert tuple(stand_cfg.events["push_robot"].params["velocity_range"]["x"]) == (-0.5, 0.5)
assert "push_force" in step_cfg.events and step_cfg.events["push_force"].mode == "step"
assert "push_force" not in stand_cfg.events

# --- alvos por-mão (reaching) + gate de orientação (lift) vindos dos knobs ---
assert lift_cfg.rewards["reaching"].params.get("lateral_offset") == 0.10
assert lift_cfg.rewards["lift"].params.get("upright_gate_deg") == 10.0

# --- FIX DE GEOMETRIA 07-16: caixa na BORDA da mesa, não mais no centro ---
assert lift_active.scene.box_xy == (0.30, 0.0), \
    "config ativo da Lift não está com a caixa na borda (ver configs/c2026_07_16_box_edge.py)"
print(f"OK: config ativo da Lift = caixa em x={lift_active.scene.box_xy[0]} (borda da mesa)")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _build_and_step(task_id: str, label: str):
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
    env.reset()
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    env.step(act)
    print(f"OK {label}: instancia + reseta + 1 step")
    return env


_build_and_step("Mjlab-Lift-Box-Unitree-G1-Stand", "Stand")
_build_and_step("Mjlab-Lift-Box-Unitree-G1-Stand-Step", "Stand-Step (push_force)")
_build_and_step("Mjlab-Lift-Box-Unitree-G1-Lift", "Lift (reaching/grasp/lift/com_balance/feet_slip)")

print("\nOK smoke completo: 3 skills registradas, fundação (feet_slip) sempre presente,")
print("knobs por-config conferem, caixa na borda, os 3 envs sobem e dão 1 step.")
