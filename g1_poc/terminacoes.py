"""As terminações próprias do g1_poc (§12).

`time_out` e `fell_over` vêm da fundação e não aparecem aqui. Ela traz um terceiro,
o `out_of_terrain_bounds`, que o `env_cfg` remove: o terreno é plano e a mobília tem
pose absoluta. São 4 no total, e o smoke confere o número.

Princípio: **terminar em vez de penalizar.** Uma trajetória inválida acaba; ela não
paga multa. É o que a tarefa `tracking` do mjlab faz, e ela troca quatro penalidades
por duas terminações:

    table_contact  −1,5   ->  contato_ilegal
    box_shake      −0,15  ->  precise_ori
    back_penalty   −0,5   ->  reaching bimanual
    com_balance    −2,0   ->  upright + fell_over
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def caixa_largada(
    env: ManagerBasedRlEnv,
    object_name: str,
    z_min: float,
    dist_max: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """A caixa caiu, ou ela escapou das duas palmas.

    ⚠ Só vale DEPOIS de o elo `pegar` fechar. Antes disso a caixa está na
    prateleira e longe das mãos de propósito — terminar ali encerraria o episódio
    no primeiro passo.
    """
    obj: Entity = env.scene[object_name]
    robot: Entity = env.scene[asset_cfg.name]
    caixa = obj.data.root_link_pos_w
    palmas = robot.data.site_pos_w[:, asset_cfg.site_ids]        # [B,2,3]
    dist = torch.norm(palmas - caixa.unsqueeze(1), dim=-1)       # [B,2]
    caiu = caixa[:, 2] < z_min
    escapou = (dist > dist_max).all(dim=-1)
    ja_pegou = getattr(env, "poc_success", torch.zeros_like(caiu, dtype=torch.float)) > 0.5
    return (caiu | escapou) & ja_pegou


def contato_ilegal(
    env: ManagerBasedRlEnv, sensor_name: str, limiar_N: float
) -> torch.Tensor:
    """A pelve, o tronco ou a coxa toca a prateleira com força acima do limiar.

    Escorar o corpo na prateleira é o hack medido no repo: o robô alcança embaixo
    sem agachar, e o `com_balance` não pega, porque ele estica a coxa e mantém o
    CoM atrás.

    O antebraço, a mão e o pé NÃO estão no sensor. Numa pega a 0,04 m o antebraço
    passa perto do tampo, e esse contato é normal.
    """
    f = env.scene[sensor_name].data.force
    assert f is not None, f"sensor '{sensor_name}' precisa do field 'force'."
    return torch.norm(f, dim=-1).amax(dim=-1) > limiar_N
