"""O empurrão. Evento FIXO do fabricante, com uma janela livre no início.

O eixo `push` do currículo SAIU na reforma de 07/08 (§7). Com ele saíram a escada de
5 níveis, o `push_fator` por env, o `push_nivel`, a força sustentada
(`empurrao_sustentado`) e a célula `(PARADO, PUSH)` do orquestrador.

O que resta é o `push_by_setting_velocity` do fabricante
(`mjlab/envs/mdp/events.py`), com magnitude fixa, em todos os envs, sempre ligado —
exatamente como o cfg de velocity dele faz.

**Por que o invólucro existe.** Só pela janela livre. Ele é a única coisa que este
módulo acrescenta ao evento do fabricante.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import events as base_events
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_ROBO = SceneEntityCfg("robot")

JANELA_LIVRE_S = 0.5
"""Segundos no início do episódio em que NENHUM empurrão age.

⚠️ Não é folga arbitrária. As tarefas de `SPAWN_SEGURANDO` nascem com as palmas
apenas TOCANDO a caixa, com força normal zero. O `pregrasp.py` mede que, com ação
nula, a caixa cai 22 cm em 0,5 s. Um empurrão dentro desse intervalo torna o episódio
não-ganhável antes de a política ter tido a chance de fechar a preensão — e o
currículo leria isso como incompetência."""


def empurrao(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = _ROBO,
) -> None:
    """`push_by_setting_velocity` do fabricante, calado nos primeiros `JANELA_LIVRE_S`.

    A assinatura é a mesma do evento do fabricante, então o `velocity_range` do cfg de
    velocity entra sem tradução. A dinâmica também é a mesma — a função de dentro é a
    dele, chamada com o subconjunto de envs que já passou da janela.

    Use com `mode="interval"`."""
    env_ids = resolve_env_ids(env, env_ids)
    fora = env.episode_length_buf * env.step_dt >= JANELA_LIVRE_S
    ids = env_ids[fora[env_ids]]
    if len(ids) == 0:
        return
    base_events.push_by_setting_velocity(
        env, ids, velocity_range=velocity_range, asset_cfg=asset_cfg)
