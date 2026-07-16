"""Eventos da Levanta-Caixa — por ora, só o workaround do reset de juntas.

BUG upstream (mjlab, diagnosticado 2026-07-15 no T4): `reset_joints_by_offset`
indexa `asset.data.soft_joint_pos_limits[env_ids]`, mas quando o env não tem
NENHUM evento de DR de startup (removemos foot_friction/encoder_bias/base_com),
o campo fica por-MODELO — shape (1, nj, 2) — e não por-env (B, nj, 2). Qualquer
env_id > 0 então indexa fora do range → device-side assert no CUDA (aflora
async, parecendo vir de outro lugar). As tasks padrão nunca exercitam esse
caminho porque todas têm DR de startup, que expande os campos por-mundo.

Esta versão é idêntica à do mjlab, com UMA mudança: expande os soft limits pra
por-env quando vierem colapsados (expand = view com stride 0, sem cópia). Se a
DR de startup voltar um dia (sim-to-real), o campo já vem expandido e o guard
vira no-op — seguro manter pra sempre.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp import events as mjlab_events
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def reset_joints_by_offset(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
    env_ids = mjlab_events.resolve_env_ids(env, env_ids)

    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert soft_joint_pos_limits is not None
    # >>> o fix: sem DR de startup o campo vem (1, nj, 2); expande pra (B, nj, 2)
    if soft_joint_pos_limits.shape[0] == 1 and env.num_envs > 1:
        soft_joint_pos_limits = soft_joint_pos_limits.expand(env.num_envs, -1, -1)

    joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_pos += mjlab_events.sample_uniform(*position_range, joint_pos.shape, env.device)
    joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel += mjlab_events.sample_uniform(*velocity_range, joint_vel.shape, env.device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids,
    )
