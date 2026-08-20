"""As observações do g1_poc.

Contrato (§5): 112 canais no ator, 125 no crítico. Os 13 privilegiados são o
`base_lin_vel` (3) mais os 10 de força/caixa deste arquivo — o `env_cfg` o tira do ator e
o devolve ao crítico.

Os quatro canais de caixa são ZERADOS quando `caixa_valida` é 0. Sem isso, "não
existe caixa" e "a caixa está exatamente no alvo" produziriam a mesma observação.

⚠ NÃO existe clamp de distância. O bit faz o trabalho: na forma de locomoção os
canais são zero, e é exatamente o que o robô real vê quando não há caixa.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


# --------------------------------------------------------------------- auxiliares
def _bit(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """[B,1] — o bit `caixa_valida`, direto do comando."""
    return env.command_manager.get_term(command_name).command[:, 9:10]


def _para_base(env: ManagerBasedRlEnv, vec_w: torch.Tensor) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_link_quat_w, vec_w)


def alvos_das_palmas(
    env: ManagerBasedRlEnv,
    object_name: str,
    lateral_offset: float,
) -> torch.Tensor:
    """[B,2,3] — o ponto que cada palma deve alcançar, em MUNDO.

    O alvo NÃO é o centro da caixa: é a FACE lateral de cada lado. Com o centro,
    o `reaching` estagnava com UMA mão na face próxima, sem gradiente para o
    abraço. O offset roda com a caixa. (Vem do `_palm_dists_sq` da skill Lift.)
    """
    obj: Entity = env.scene[object_name]
    caixa = obj.data.root_link_pos_w
    off = torch.zeros_like(caixa)
    off[:, 1] = lateral_offset
    off = quat_apply(obj.data.root_link_quat_w, off)
    return torch.stack((caixa + off, caixa - off), dim=1)   # E -> +y, D -> -y


# ---------------------------------------------------------------- ator (112)
def palmas_para_caixa(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    lateral_offset: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """[B,6] — o vetor de cada palma até a sua face, no frame da BASE.

    Zerado quando `caixa_valida` é 0.
    """
    robot: Entity = env.scene[asset_cfg.name]
    palmas = robot.data.site_pos_w[:, asset_cfg.site_ids]          # [B,2,3] (E, D)
    alvos = alvos_das_palmas(env, object_name, lateral_offset)      # [B,2,3]
    d_w = alvos - palmas
    d_b = torch.stack(
        (_para_base(env, d_w[:, 0]), _para_base(env, d_w[:, 1])), dim=1)
    return d_b.reshape(d_b.shape[0], -1) * _bit(env, command_name)


def caixa_para_alvo(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
) -> torch.Tensor:
    """[B,3] — o vetor da caixa até o alvo, no frame da BASE.

    O alvo vive em MUNDO. A observação é egocêntrica, portanto a conversão é por
    passo. A recompensa continua ancorada no mundo — só a OBSERVAÇÃO é do corpo.
    Zerado quando `caixa_valida` é 0.
    """
    obj: Entity = env.scene[object_name]
    alvo_w = env.command_manager.get_term(command_name).command[:, 0:3]
    return _para_base(env, alvo_w - obj.data.root_link_pos_w) * _bit(env, command_name)


def fatia_comando(
    env: ManagerBasedRlEnv, command_name: str, lo: int, hi: int
) -> torch.Tensor:
    """[B, hi-lo] — uma fatia crua do comando (`face_alvo`, `dir_alvo`, o bit).

    As fatias `face_alvo` e `dir_alvo` já são zeradas dentro do comando quando o
    bit é 0. Ver `CaixaAlvoCommand._update_command`.
    """
    return env.command_manager.get_term(command_name).command[:, lo:hi]


# ------------------------------------------------------------- crítico (+10)
def forca_palmas(env: ManagerBasedRlEnv, sensores: tuple[str, str]) -> torch.Tensor:
    """[B,2] — a magnitude da força de contato de cada palma. Privilegiado."""
    saida = []
    for nome in sensores:
        f = env.scene[nome].data.force
        assert f is not None, f"sensor '{nome}' precisa do field 'force'."
        saida.append(torch.norm(f, dim=-1).sum(dim=-1, keepdim=True))
    return torch.cat(saida, dim=-1)


def forca_apoio(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """[B,1] — a componente vertical da força de apoio da prateleira. Privilegiado."""
    f = env.scene[sensor_name].data.force
    assert f is not None
    return f[..., 2].abs().sum(dim=-1, keepdim=True)


def vel_caixa(env: ManagerBasedRlEnv, object_name: str) -> torch.Tensor:
    """[B,6] — velocidade linear e angular da caixa, em MUNDO. Privilegiado."""
    obj: Entity = env.scene[object_name]
    return torch.cat(
        (obj.data.root_link_lin_vel_w, obj.data.root_link_ang_vel_w), dim=-1)


def topo_prateleira(env: ManagerBasedRlEnv, meia_z: float) -> torch.Tensor:
    """[B,1] — a altura do topo da prateleira. Privilegiado.

    O ator NÃO vê isto. Ele infere a altura de pega pela posição da caixa, que
    repousa em cima. É o que o robô real consegue.
    """
    mesa: Entity = env.scene["table"]
    return (mesa.data.root_link_pos_w[:, 2] + meia_z).unsqueeze(-1)


def face_normal_b(env: ManagerBasedRlEnv, command_name: str, object_name: str) -> torch.Tensor:
    """[B,3] — a normal ATUAL da face alvo, no frame da BASE.

    O ator vê o DESEJADO (`dir_alvo`) e não via o ATUAL: a orientação da caixa só
    era recuperável pela diferença dos dois vetores palma→face, dominada pela
    distância. O `reorientar` fecha por `Δθ < 20°` e o `precise_ori` paga por Δθ —
    sem este canal a coordenada é invisível (auditoria T18, 20/08).

    No deploy a percepção JÁ entrega esta grandeza: é a mesma orientação medida que
    preenche `face_alvo`/`dir_alvo` (§21.2, "os medidos").

    Zera com `caixa_valida = 0`, como os outros canais de caixa.
    """
    cmd = env.command_manager.get_term(command_name)
    obj = env.scene[object_name]
    robot = env.scene["robot"]
    face_b = cmd.command[:, 3:6]                                   # face, frame da caixa
    normal_w = quat_apply(obj.data.root_link_quat_w, face_b)
    normal_base = quat_apply_inverse(robot.data.root_link_quat_w, normal_w)
    return normal_base * cmd.command[:, 9:10]
