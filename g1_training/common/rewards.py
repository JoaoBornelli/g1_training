"""Recompensas de FUNDAÇÃO (equilíbrio) compartilhadas por TODAS as skills.

Diferente de `skills/lift/rewards.py` (recompensa de TAREFA, só existe na Lift),
este módulo é a parte "ficar de pé" que toda skill herda via `base_env.py`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

FOOT_SITES = ("left_foot", "right_foot")
FEET_CONTACT_SENSOR = "feet_ground_contact"  # já existe na cena (herdado do flat env G1)


def feet_slip_standing(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg,
                       sensor_name: str = FEET_CONTACT_SENSOR) -> torch.Tensor:
    """Penaliza pé ESCORREGANDO (vel. xy enquanto em contato) — SEMPRE ligado.

    É o `feet_slip` do mjlab (mjlab/tasks/velocity/mdp/rewards.py) DECAPADO do
    gate de comando de marcha (`active = total_command > threshold`): aqui não
    existe comando de velocidade, então o gate SEMPRE zeraria a penalidade —
    era por isso que o termo original não sobrevivia na task parada. Motivo de
    existir (user, 07-16): o robô abre um "espacato" (base larga) escorregando
    os pés pra alargar o apoio sem dar um passo de verdade — punir o próprio
    escorregão ataca o MECANISMO, não o sintoma (upright/posture não veem isso).
    Fundação: aplicado incondicionalmente em toda skill via base_env, não é
    knob de fine-tune (não desliga por config).
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()               # [B, 2]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
    vel_xy_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)           # [B, 2]
    return torch.sum(vel_xy_sq * in_contact, dim=1)


def foot_flat_penalty(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """FALLBACK (não wireado ainda): pune o pé fora do paralelo ao chão.

    Preparado pra caso o slip (acima) não baste pra fechar o espacato — o
    user pediu pra tentar isso SÓ SE o slip não resolver (07-16). Projeta a
    gravidade no frame do PÉ (mesmo padrão do reward `upright`, mas na site
    do pé em vez do tronco): pé plano no chão -> componente xy ~0.
    """
    asset: Entity = env.scene[asset_cfg.name]
    from mjlab.utils.lab_api.math import quat_apply_inverse
    foot_quat = asset.data.site_quat_w[:, asset_cfg.site_ids]      # [B, 2, 4]
    gravity_w = asset.data.gravity_vec_w                           # [3]
    gravity_b = quat_apply_inverse(foot_quat, gravity_w)            # [B, 2, 3]
    xy_sq = torch.sum(torch.square(gravity_b[..., :2]), dim=-1)     # [B, 2]
    return torch.sum(xy_sq, dim=1)
