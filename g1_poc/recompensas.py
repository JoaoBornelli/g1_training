"""Os 5 termos de tarefa do g1_poc (§8.2).

Os 13 termos de fundação vêm do `velocity` do mjlab, sem uma linha aqui.

Desenho:
    staged      = reaching × (1 + bringing)          — o anti-hack é a FORMA
    precise_pos = exp(−‖caixa − alvo‖² / 0,05²)
    precise_ori = reaching × exp(−Δθ² / 0,40²)
    squeeze     = tanh( min(F_n_esq, F_n_dir) / F_ref )
    joint_vel_hinge = (|v| − v_max)⁺²

Os quatro primeiros multiplicam por `caixa_valida`. **Isto é obrigatório**: com o
bit em 0 os canais da caixa são zerados, e um vetor zerado dá exp(0) = 1. Sem a
multiplicação, "não existe caixa" pagaria o valor MÁXIMO.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1_poc.observacoes import alvos_das_palmas

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


# --------------------------------------------------------------------- auxiliares
def _valida(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).command[:, 9]


def _erro_pos_sq(env: ManagerBasedRlEnv, command_name: str, object_name: str):
    obj: Entity = env.scene[object_name]
    alvo = env.command_manager.get_term(command_name).command[:, 0:3]
    return torch.sum(torch.square(alvo - obj.data.root_link_pos_w), dim=-1)


def _reaching(
    env: ManagerBasedRlEnv,
    object_name: str,
    lateral_offset: float,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """O `reaching` BIMANUAL.

    O `reaching` do `lift_cube` do mjlab mede UM site (um braço com garra). Aqui
    cada palma mira a SUA face lateral, e a média das duas distâncias entra no
    kernel: uma mão atrasada derruba o gradiente, portanto as duas se aproximam
    juntas. O máximo do termo é a pose PRÉ-GRASP, com as mãos flanqueando a caixa.
    """
    robot: Entity = env.scene[asset_cfg.name]
    palmas = robot.data.site_pos_w[:, asset_cfg.site_ids]          # [B,2,3]
    alvos = alvos_das_palmas(env, object_name, lateral_offset)      # [B,2,3]
    d2 = torch.sum(torch.square(palmas - alvos), dim=-1)           # [B,2]
    return torch.exp(-d2.mean(dim=-1) / std**2)


# ----------------------------------------------------------------- os 5 termos
def staged(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    reaching_std: float,
    lateral_offset: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """`reaching × (1 + bringing)` — a forma do `lift_cube` do mjlab.

    O `bringing` só paga ATRAVÉS do `reaching`. Levar a caixa ao alvo sem as mãos
    nela não paga. O anti-hack é a forma, e não uma penalidade.

    O σ do `bringing` é VARIÁVEL, e é a única mudança que fazemos numa função do
    mjlab. O σ fixo de 0,30 m já está saturado no nível 0, onde a caixa sobe
    0,17 m: exp(−0,17²/0,30²) = 0,72 contra 1,00 no alvo. Com
    σ = distância comandada, o termo cobre o percurso todo.
    """
    reaching = _reaching(env, object_name, lateral_offset, reaching_std, asset_cfg)
    std = env.poc_dist_inicial
    bringing = torch.exp(-_erro_pos_sq(env, command_name, object_name) / std**2)
    return reaching * (1.0 + bringing) * _valida(env, command_name)


def precise_pos(
    env: ManagerBasedRlEnv, command_name: str, object_name: str, std: float
) -> torch.Tensor:
    """Gaussiana APERTADA no alvo. É o termo que paga por a caixa ESTAR no lugar.

    O treino atual não tem nenhum termo assim no `pegar`: o `box_at_peito` foi
    retirado dele pelo ADR-0001, e sobrou só o progresso escalar.
    """
    err = _erro_pos_sq(env, command_name, object_name)
    return torch.exp(-err / std**2) * _valida(env, command_name)


def precise_ori(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    std: float,
    lateral_offset: float,
    reaching_std: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """`reaching × exp(−Δθ²/σ²)`.

    Δθ é o ângulo entre a normal da face alvo, em MUNDO, e o `dir_alvo`. A
    simetria do cubo se resolve sozinha: girar em torno da normal não move o vetor.

    No nível 0 o `dir_alvo` é a normal ATUAL, portanto o termo pede "erga sem
    torcer". É isto que substitui o `box_shake`: erguer torto deixa de pagar, em
    vez de custar.
    """
    cmd = env.command_manager.get_term(command_name)
    theta = cmd.erro_ang()
    reaching = _reaching(env, object_name, lateral_offset, reaching_std, asset_cfg)
    return reaching * torch.exp(-torch.square(theta) / std**2) * _valida(env, command_name)


def squeeze(
    env: ManagerBasedRlEnv,
    command_name: str,
    palm_sensors: tuple[str, str],
    massa_attr: str,
    mu: float,
) -> torch.Tensor:
    """`tanh( min(F_n_esq, F_n_dir) / F_ref )` — o termo que faltava.

    **Este é o termo mais importante do arquivo.** Sem ele o treino repete a falha
    de hoje.

    O diagnóstico: o `lift` de hoje paga +0,34/s por centímetro de subida no nível
    0. O gradiente existe e é grande. O robô mesmo assim não subiu 1 cm. O motivo é
    que o gradiente está na coordenada ERRADA:

        d(recompensa)/d(altura da caixa)  = grande
        d(recompensa)/d(força de aperto)  = ZERO

    A política só age em alvos de junta. Antes de a força de atrito vencer o peso a
    caixa não se move, e nenhuma recompensa muda. É um degrau, não uma rampa.

    O `reaching` não conserta isto: a palma não penetra a caixa, portanto ele satura
    no contato.

    A força de palma cresce de forma CONTÍNUA com a penetração comandada. Portanto
    este termo tem derivada positiva em toda a faixa de 0 N a F_ref.

    ANTI-HACK: usa só a componente NORMAL ao pad. Apertar a caixa para BAIXO contra
    a prateleira gera força TANGENCIAL, e não normal — o ADR-0001 registrou esse
    risco. A projeção o fecha sem precisar de um segundo termo.

    O `min` das duas palmas exige aperto SIMÉTRICO: uma palma sozinha vale zero.
    """
    forcas = []
    for nome in palm_sensors:
        dados = env.scene[nome].data
        f = dados.force
        n = dados.normal
        assert f is not None, f"sensor '{nome}' precisa do field 'force'."
        assert n is not None, f"sensor '{nome}' precisa do field 'normal'."
        # componente ao longo da normal do contato, somada nos slots
        f_n = torch.sum(f * n, dim=-1).abs().sum(dim=-1)
        forcas.append(f_n)
    f_min = torch.minimum(forcas[0], forcas[1])
    # F_ref = m·g / (2·μ). A massa é POR ENV (a DR de carga a sorteia).
    massa = getattr(env, massa_attr)
    f_ref = (massa * 9.81 / (2.0 * mu)).clamp(min=1e-3)
    return torch.tanh(f_min / f_ref) * _valida(env, command_name)


def joint_vel_hinge(
    env: ManagerBasedRlEnv, max_vel: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """Penalidade de DOBRADIÇA sobre a velocidade de junta.

    Grátis abaixo de `max_vel`; quadrática acima. É a forma do
    `joint_velocity_hinge_penalty` do `lift_cube`, reescrita aqui para não importar
    `mjlab.tasks.manipulation`, que registra tasks como efeito colateral.

    O currículo aperta o PESO deste termo, de −0,01 para −1,00. É aqui que a
    qualidade de pose se conserta, e é DEPOIS da tarefa.
    """
    robot: Entity = env.scene[asset_cfg.name]
    v = robot.data.joint_vel[:, asset_cfg.joint_ids]
    excesso = (v.abs() - max_vel).clamp_min(0.0)
    return torch.square(excesso).sum(dim=-1)
