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


def _write_root(asset: Entity, positions, orientations, env_ids, device) -> None:
    n = len(env_ids)
    asset.write_root_link_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids)


def reset_scene_with_rehearsal(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    box_pose_range: dict[str, tuple[float, float]],
    rehearsal_fraction: float,
    far_x: float = 5.0,
    box_far_z: float = 0.10,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
) -> None:
    """Reset COMBINADO caixa+mesa com REHEARSAL anti-esquecimento entre skills.

    A maioria dos envs spawna normal (caixa na mesa + jitter de `box_pose_range`,
    mesa no lugar). Uma FRAÇÃO (`rehearsal_fraction`) manda caixa E MESA pra LONGE
    (~`far_x` m à frente, fora de alcance) → cena LIMPA de "só ficar de pé": nada
    na frente pra alcançar nem esbarrar. Nesses envs os rewards de tarefa zeram
    sozinhos (nada pra pegar) e só a fundação (upright/postura/slip) + o push ficam
    ativos → praticam ficar de pé/recuperar empurrão. A obs (box_pos_b longe) é o
    sinal → a política aprende a condicionar "sem caixa = fico de pé", sem mascarar
    reward. Caixa e mesa usam a MESMA máscara por-env (por isso um evento só) e
    ficam offsetadas entre si quando longe (não se sobrepõem). Ver [[g1-lift-box-task]]."""
    env_ids = mjlab_events.resolve_env_ids(env, env_ids)
    n = len(env_ids)
    device = env.device
    origins = env.scene.env_origins[env_ids]
    is_stand = torch.rand(n, device=device) < rehearsal_fraction   # MESMA máscara p/ os 2

    # --- CAIXA: normal = repouso + jitter; rehearsal = longe, no chão ---
    box: Entity = env.scene[box_cfg.name]
    box_root = box.data.default_root_state[env_ids].clone()
    off = torch.zeros(n, 3, device=device)
    for i, key in enumerate(("x", "y", "z")):
        if key in box_pose_range:
            off[:, i] = mjlab_events.sample_uniform(
                box_pose_range[key][0], box_pose_range[key][1], (n,), device)
    box_pos = box_root[:, 0:3] + off + origins
    box_pos[is_stand, 0] = origins[is_stand, 0] + far_x
    box_pos[is_stand, 1] = origins[is_stand, 1]
    box_pos[is_stand, 2] = box_far_z
    _write_root(box, box_pos, box_root[:, 3:7], env_ids, device)

    # --- PRATELEIRA (MOCAP): normal = no lugar; rehearsal = longe (offset da caixa
    #     p/ não sobrepor). MOCAP => write_mocap_pose_to_sim (não write_root_link), sem
    #     velocidade (corpo cinemático). ---
    table: Entity = env.scene[table_cfg.name]
    tab_root = table.data.default_root_state[env_ids].clone()
    tab_pos = tab_root[:, 0:3] + origins
    tab_pos[is_stand, 0] = origins[is_stand, 0] + far_x + 1.5      # 1.5 m além da caixa
    tab_pos[is_stand, 1] = origins[is_stand, 1]
    table.write_mocap_pose_to_sim(
        torch.cat([tab_pos, tab_root[:, 3:7]], dim=-1), env_ids=env_ids)
