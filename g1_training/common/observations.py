"""Observações da Levanta-Caixa — o CONTRATO sim-to-real.

O ACTOR só vê o que o robô REAL mede + a pose que a percepção entrega:
propriocepção (herdada da velocity) + ALVO e CAIXA em frame da base + torque de
junta. NUNCA força de contato da palma (rubber hands não têm sensor de força) —
isso é privilégio do critic (entra no brick 5, com os sensores).

Tudo em frame da BASE (relativo ao robô), nunca em coordenadas de MUNDO: o mundo
inclui o env_origin (que varia por-ambiente) -> obs em mundo não generaliza.
(orientação da caixa fica pra Fase 2 — a preensão precisa dela; posição basta agora.)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def _to_base_frame(env: "ManagerBasedRlEnv", vec_w):
    """Leva um vetor do frame do MUNDO pro frame da BASE do robô."""
    robot: Entity = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_link_quat_w, vec_w)


def object_pos_b(env: "ManagerBasedRlEnv", object_name: str):
    """Posição do objeto relativa à base (frame do robô), [B, 3]."""
    robot: Entity = env.scene["robot"]
    obj: Entity = env.scene[object_name]
    return _to_base_frame(env, obj.data.root_link_pos_w - robot.data.root_link_pos_w)


def target_pos_b(env: "ManagerBasedRlEnv", command_name: str):
    """Alvo de sustentação (do comando) relativo à base, [B, 3]."""
    robot: Entity = env.scene["robot"]
    target_w = env.command_manager.get_term(command_name).command[:, 0:3]
    return _to_base_frame(env, target_w - robot.data.root_link_pos_w)


def command_phase(env: "ManagerBasedRlEnv", command_name: str):
    """O bit de fase do comando (reservado; 0=hold), [B, 1]."""
    return env.command_manager.get_term(command_name).command[:, 3:4]


def joint_torque(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _ROBOT):
    """Torque por atuador (actuator_force) — o sinal do hardware real, [B, n_act].

    O ACTOR infere contato daqui (no robô real não há sensor de força na palma)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force[:, asset_cfg.actuator_ids]
