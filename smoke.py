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
for t in ("reaching", "grasp", "lift", "sustain_precise", "com_balance", "table_contact",
          "box_shake"):
    assert t not in stand_cfg.rewards, f"{t} vazou no Stand"
    assert t not in step_cfg.rewards, f"{t} vazou no Stand-Step"
    assert t in lift_cfg.rewards, f"{t} ausente na Lift"

# --- postura da Stand/Stand-Step (baseline retrofitado, não é o que se tuna agora) ---
assert stand_cfg.rewards["posture"].weight == 0.5, "Stand: posture deveria ser 0.5"
assert step_cfg.rewards["posture"].weight == 0.5, "Stand-Step: posture deveria ser 0.5"

# --- upright é GUARDA, não knob de fine-tune (nunca baixar sem motivo forte — ver
#     knobs.py: afrouxar pra "liberar reach" já degradou o treino inteiro em 07-15) ---
assert stand_cfg.rewards["upright"].weight == 1.0
assert lift_cfg.rewards["upright"].weight == 1.0, \
    "upright da Lift saiu de 1.0 — se foi de propósito, ok; se foi acidente, ver 07-15"

# --- anti-dinâmica afrouxada SÓ no Stand-Step ---
assert step_cfg.rewards["action_rate_l2"].weight == -0.03
assert stand_cfg.rewards["action_rate_l2"].weight == -0.1

# --- push mais forte só no Stand-Step (treino); push_force só lá também ---
assert tuple(step_cfg.events["push_robot"].params["velocity_range"]["x"]) == (-0.8, 0.8)
assert tuple(stand_cfg.events["push_robot"].params["velocity_range"]["x"]) == (-0.5, 0.5)
assert "push_force" in step_cfg.events and step_cfg.events["push_force"].mode == "step"
assert "push_force" not in stand_cfg.events

# --- pesos/params de TAREFA da Lift: são KNOBS (tunados a cada fine-tune, ver
#     configs/), então aqui NÃO travamos valor nenhum — só provamos que o env
#     aplicou exatamente o que o config ATIVO diz (wiring correto, não um número
#     esquecido hardcoded em algum lugar). Se você mudar um knob, este bloco
#     acompanha sozinho; se ele falhar, o bug está no wiring (env.py), não no knob. ---
r, s = lift_active.reward, lift_active.scene
_lateral = r.lateral_offset if r.lateral_offset is not None else s.box_half[1]
assert lift_cfg.rewards["posture"].weight == r.posture
assert lift_cfg.rewards["reaching"].weight == r.reaching
assert lift_cfg.rewards["reaching"].params.get("lateral_offset") == _lateral
assert lift_cfg.rewards["grasp"].weight == r.grasp
assert lift_cfg.rewards["lift"].weight == r.lift
assert lift_cfg.rewards["lift"].params.get("upright_std") == r.upright_std
assert lift_cfg.rewards["sustain_precise"].weight == r.sustain_precise
assert lift_cfg.rewards["back_penalty"].weight == r.back
assert lift_cfg.rewards["table_contact"].weight == r.table_contact
assert lift_cfg.rewards["com_balance"].weight == r.com_balance
assert lift_cfg.rewards["com_balance"].params.get("forward_margin") == r.com_margin
assert lift_cfg.rewards["box_shake"].weight == r.box_shake
print("OK: pesos/params de tarefa da Lift refletem o config ATIVO (wiring correto)")

# --- FIX DE GEOMETRIA 07-16: caixa na BORDA da mesa, não mais no centro ---
assert lift_active.scene.box_xy == (0.30, 0.0), \
    "config ativo da Lift não está com a caixa na borda (ver configs/c2026_07_16_box_edge.py)"
print(f"OK: config ativo da Lift = caixa em x={lift_active.scene.box_xy[0]} (borda da mesa)")

# --- DR de posição da caixa (offset xy, 07-16): reset_box precisa refletir os
#     ranges do knob, e a MESA nunca varia (pose_range sempre {}) ---
jx, jy = lift_active.scene.box_jitter_x, lift_active.scene.box_jitter_y
expected_range = {"x": tuple(jx), "y": tuple(jy)} if (any(jx) or any(jy)) else {}
assert lift_cfg.events["reset_box"].params["pose_range"] == expected_range, \
    f"reset_box pose_range {lift_cfg.events['reset_box'].params['pose_range']} != {expected_range}"
assert lift_cfg.events["reset_table"].params["pose_range"] == {}, "mesa não deveria variar"
print(f"OK: jitter da caixa x={jx} y={jy} (mesa fixa)")

# --- push sob carga (generalize): push_robot mais forte + push_force na Lift de treino ---
if lift_active.push.force_enabled:
    assert "push_force" in lift_cfg.events, "force_enabled mas push_force ausente na Lift"
    assert lift_cfg.events["push_force"].mode == "step"
assert tuple(lift_cfg.events["push_robot"].params["velocity_range"]["x"]) == tuple(lift_active.push.x)
print(f"OK: push_robot x={lift_active.push.x}, push_force={'on' if lift_active.push.force_enabled else 'off'}")

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
lift_env = _build_and_step("Mjlab-Lift-Box-Unitree-G1-Lift",
                           "Lift (reaching/grasp/lift/com_balance/feet_slip)")

if any(jx) or any(jy):
    # confirma o jitter DE VERDADE (não só o wiring): reseta de novo e observa
    # a posição da caixa variar entre os 4 envs (cada um sorteia seu próprio offset).
    lift_env.reset()
    box_xy = lift_env.scene["box"].data.root_link_pos_w[:, :2] - lift_env.scene.env_origins[:, :2]
    spread = (box_xy.max(dim=0).values - box_xy.min(dim=0).values)
    assert (spread > 0.0).any(), "jitter configurado mas caixa nasceu na mesma pose em todo env"
    print(f"OK: jitter observado no reset -> spread x={spread[0]:.3f}m y={spread[1]:.3f}m entre envs")

print("\nOK smoke completo: 3 skills registradas, fundação (feet_slip) sempre presente,")
print("knobs por-config conferem, caixa na borda, os 3 envs sobem e dão 1 step.")
