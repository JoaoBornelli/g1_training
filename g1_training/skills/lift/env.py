"""A skill LIFT: pegar a caixa e levá-la ao alvo de sustentação (bimanual).

Cadeia MONOTÔNICA (anti-vale) de recompensa: reaching -> grasp -> lift ->
sustain_precise, com anti-hacks (back_penalty, table_contact, com_balance) e
fundação escopada (upright cheio, postura só perna+cintura — braços livres
pra manipular). Ver `rewards.py` pro racional de cada termo e a memória
[[g1-lift-box-task]] pro histórico completo de descobertas.
"""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...base_env import BACK_SENSORS, BODY_TABLE_SENSOR, FOOT_SITES, PALM_SENSORS, build_base_env
from ...common.robot import PALM_SITES
from . import rewards as R
from .knobs import LiftKnobs

_RESET_BASE_POSE_RANGE = {
    "x": (-0.10, 0.0), "y": (-0.10, 0.10), "z": (0.01, 0.05), "yaw": (-0.2, 0.2),
}
_POSTURE_JOINTS = [".*(hip|knee|ankle|waist).*"]  # braços LIVRES pra manipular


def build_lift_env(knobs: LiftKnobs, play: bool = False) -> ManagerBasedRlEnvCfg:
    s = knobs.scene
    table_top = 2 * s.table_half[2]
    box_z = s.box_z if s.box_z is not None else table_top + s.box_half[2]
    box_pos = (s.box_xy[0], s.box_xy[1], box_z)
    table_pos = (s.table_xy[0], s.table_xy[1], s.table_half[2])

    jitter = s.box_xy_jitter
    box_pose_range = {"x": (-jitter, jitter), "y": (-jitter, jitter)} if jitter > 0 else {}

    cfg = build_base_env(
        play=play,
        box_pos=box_pos, table_pos=table_pos,
        box_half=s.box_half, box_mass=s.box_mass,
        table_half=s.table_half, table_mass=s.table_mass,
        box_pose_range=box_pose_range,
        reset_base_pose_range=_RESET_BASE_POSE_RANGE,
        posture_weight=knobs.reward.posture,
        posture_joints=_POSTURE_JOINTS,
    )
    cfg.rewards["upright"].weight = knobs.reward.upright
    cfg.rewards["action_rate_l2"].weight = knobs.foundation.action_rate_l2
    cfg.rewards["body_ang_vel"].weight = knobs.foundation.body_ang_vel
    cfg.rewards["angular_momentum"].weight = knobs.foundation.angular_momentum

    if "push_robot" in cfg.events:
        vr = cfg.events["push_robot"].params["velocity_range"]
        vr["x"] = knobs.push.x
        vr["y"] = knobs.push.y

    cmd = cfg.commands["lift_target"]
    cmd.target_x = knobs.command.target_x
    cmd.target_y = knobs.command.target_y
    cmd.target_z = knobs.command.target_z

    palm_asset = SceneEntityCfg("robot", site_names=PALM_SITES)
    grasp_sensors = dict(palm_sensors=PALM_SENSORS, back_sensors=BACK_SENSORS)
    r = knobs.reward
    lateral_offset = r.lateral_offset if r.lateral_offset is not None else s.box_half[1]

    cfg.rewards["reaching"] = RewardTermCfg(
        func=R.reaching_reward, weight=r.reaching,
        params={"std_coarse": r.std_coarse, "std_fine": r.std_fine, "object_name": "box",
                "asset_cfg": palm_asset, "lateral_offset": lateral_offset},
    )
    cfg.rewards["grasp"] = RewardTermCfg(
        func=R.grasp_reward, weight=r.grasp, params={**grasp_sensors},
    )
    cfg.rewards["lift"] = RewardTermCfg(
        func=R.lift_reward, weight=r.lift,
        params={"object_name": "box", "command_name": "lift_target", "rest_z": box_z,
                "upright_std": r.upright_std, **grasp_sensors},
    )
    cfg.rewards["sustain_precise"] = RewardTermCfg(
        func=R.sustain_precise_reward, weight=r.sustain_precise,
        params={"std": r.sustain_std, "object_name": "box", "command_name": "lift_target",
                **grasp_sensors},
    )
    cfg.rewards["back_penalty"] = RewardTermCfg(
        func=R.back_penalty, weight=r.back, params={"back_sensors": BACK_SENSORS},
    )
    cfg.rewards["table_contact"] = RewardTermCfg(
        func=R.table_contact_penalty, weight=r.table_contact,
        params={"sensor_name": BODY_TABLE_SENSOR},
    )
    cfg.rewards["com_balance"] = RewardTermCfg(
        func=R.com_over_feet_penalty, weight=r.com_balance,
        params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
                "forward_margin": r.com_margin},
    )
    cfg.rewards["box_shake"] = RewardTermCfg(
        func=R.box_shake_penalty, weight=r.box_shake, params={"object_name": "box"},
    )
    return cfg
