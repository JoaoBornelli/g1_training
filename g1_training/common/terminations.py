"""Terminações extras da Levanta-Caixa.

`nonfinite_state` é DEFESA EM PROFUNDIDADE (não é o fix do NaN de 2026-07-15 —
esse foi voltar o solver pra pyramidal/impratio=1): se QUALQUER mundo divergir
pra NaN/Inf por qualquer motivo futuro, o episódio termina e o env reseta, em
vez de a obs suja chegar no check_nan do rsl_rl e derrubar o treino inteiro.
Régua no TB: `Episode_Termination/nonfinite` deve ficar ~0; se subir, a física
está doente e é pra investigar, não conviver.

Caveat: é contenção, não cura — se o reset não sanear o mundo, aquele env pode
ciclar NaN→reset→NaN; ainda assim o custo fica confinado a 1 env.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_ENTITIES = ("robot", "box", "table")


def nonfinite_state(env: "ManagerBasedRlEnv",
                    entity_names: tuple[str, ...] = _ENTITIES) -> torch.Tensor:
    """[B] bool: True onde a pose/estado de alguma entidade tem NaN/Inf."""
    bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in entity_names:
        asset: Entity = env.scene[name]
        bad |= ~torch.isfinite(asset.data.root_link_pos_w).all(dim=-1)
        joint_pos = asset.data.joint_pos
        if joint_pos is not None and joint_pos.numel():
            bad |= ~torch.isfinite(joint_pos).all(dim=-1)
    return bad
