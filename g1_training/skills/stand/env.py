"""A skill STAND: ficar de pé parado (a fundação de equilíbrio pura).

FIX DA MULETA (2026-07-15, ver memória [[g1-lift-box-task]]): caixa/mesa ficam
FORA de alcance útil — sem isso o robô aprende a se ESCORAR na mesa (ao
alcance da base, altura de peito) pra satisfazer o upright sem equilibrar de
verdade; prior inútil (e prejudicial) pro warm-start da Lift, que precisa de
um "ficar de pé" que sabe recuperar sozinho.
"""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import events as base_events
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...base_env import build_base_env
from .knobs import StandKnobs

_BOX_HALF = (0.10, 0.10, 0.10)
_SHELF_HALF = (0.30, 0.30, 0.02)       # prateleira fina mocap (longe no Stand)
_RESET_BASE_POSE_RANGE = {
    "x": (-0.10, 0.0), "y": (-0.10, 0.10), "z": (0.01, 0.05), "yaw": (-0.2, 0.2),
}


def build_stand_env(knobs: StandKnobs, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = build_base_env(
        play=play,
        box_pos=(0.50, 0.0, _BOX_HALF[2]),     # chão, ~0.5m à frente (fora de alcance útil)
        table_pos=(5.0, 0.0, 0.53),             # prateleira bem longe: nada pra escorar
        box_half=_BOX_HALF, shelf_half=_SHELF_HALF,
        reset_base_pose_range=_RESET_BASE_POSE_RANGE,
        posture_weight=knobs.foundation.posture_weight,
        posture_joints=list(knobs.foundation.posture_joints),
    )
    cfg.rewards["action_rate_l2"].weight = knobs.foundation.action_rate_l2
    cfg.rewards["body_ang_vel"].weight = knobs.foundation.body_ang_vel
    cfg.rewards["angular_momentum"].weight = knobs.foundation.angular_momentum

    if "push_robot" in cfg.events:  # removido no play por build_base_env
        vr = cfg.events["push_robot"].params["velocity_range"]
        vr["x"] = knobs.push.x
        vr["y"] = knobs.push.y

    # FORÇA SUSTENTADA (Stand-Step): apply_body_impulse segura a força na
    # pelvis por duration_s — é a MESMA perturbação que o peso da caixa vai
    # impor na Lift (força contínua deslocando o CoM). Só no treino.
    if knobs.push.force_enabled and not play:
        cfg.events["push_force"] = EventTermCfg(
            func=base_events.apply_body_impulse, mode="step",
            params={
                "force_range": knobs.push.force_range,
                "torque_range": (0.0, 0.0),
                "duration_s": knobs.push.force_duration_s,
                "cooldown_s": knobs.push.force_cooldown_s,
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            },
        )
    return cfg
