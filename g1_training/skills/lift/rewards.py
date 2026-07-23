"""Recompensas da Levanta-Caixa — desenho MONOTÔNICO (anti-vale), Degrau 4 revisado.

Cadeia sem vale de recompensa (o vale antigo travava o robô pairando perto sem pegar):
- reaching: aproxima as 2 palmas da caixa (2 escalas). SEMPRE ligada — o gate de
  contato antigo (×(1−both_contact)) ZERAVA no toque, então pegar CUSTAVA recompensa.
  Agora fica ligada até tocar; o lift assume a partir da pega.
- lift: preensão × PROGRESSO de altura (0 na mesa → 1 no alvo). Contínuo: pegar-na-mesa
  dá 0 (sem hack de segurar parado), erguer sobe monotônico. Exige preensão → arremessar
  não conta.
- sustain_precise: preensão × gaussiana APERTADA no alvo (segurar PARADO no alvo).
- back_penalty / table_contact: anti-hacks (verso na caixa / apoiar corpo na mesa).

Contato pode entrar aqui (reward NÃO roda no robô real) — nunca na obs do actor.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def height_kernel(err_sq, std: float):
    """Gaussiana sobre o erro² de posição caixa->alvo."""
    return torch.exp(-err_sq / (std ** 2))


def reaching_kernel(d2_left, d2_right, std: float):
    """Produto das 2 gaussianas de distância palma->caixa: uma mão atrasada
    derruba o gradiente -> força as duas a se aproximarem juntas."""
    return torch.exp(-(d2_left + d2_right) / (std ** 2))


def _contact(env: "ManagerBasedRlEnv", sensor_name: str):
    """Contato booleano por env (float 0/1) a partir de um ContactSensor."""
    found = env.scene[sensor_name].data.found
    assert found is not None, f"sensor '{sensor_name}' precisa do field 'found'."
    return (found > 0).any(dim=-1).float()


def _palm_dists_sq(env: "ManagerBasedRlEnv", object_name: str, asset_cfg: SceneEntityCfg,
                   lateral_offset: float = 0.0):
    """(d2_left, d2_right): distância² de cada palma (site) ao seu ALVO na caixa.

    lateral_offset=0 -> ambas miram o CENTRO (comportamento antigo; estagnava com
    UMA mão na face próxima, sem gradiente pro abraço). lateral_offset>0 -> ALVOS
    POR MÃO: palma esquerda -> face esquerda (+y no frame da CAIXA), direita -> face
    direita (-y). O máximo do reaching vira a pose PRÉ-GRASP (mãos flanqueando) ->
    degrau denso entre aproximar e prensar. O offset roda com a caixa (quat)."""
    robot: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]
    palm = robot.data.site_pos_w[:, asset_cfg.site_ids]           # [B, 2, 3] (L, R)
    box = obj.data.root_link_pos_w                               # [B, 3]
    if lateral_offset > 0.0:
        off = torch.zeros_like(box)
        off[:, 1] = lateral_offset
        off = quat_apply(obj.data.root_link_quat_w, off)          # frame da caixa
        targets = torch.stack((box + off, box - off), dim=1)      # [B, 2, 3] L→+y, R→−y
    else:
        targets = box.unsqueeze(1).expand(-1, palm.shape[1], -1)
    d2 = torch.sum(torch.square(palm - targets), dim=-1)          # [B, 2]
    return d2[:, 0], d2[:, 1]


def _grasp(env, palm_sensors, back_sensors):
    """Preensão: as 2 palmas tocando e NENHUM verso. NÃO exige caixa fora do apoio —
    pra o gradiente de erguer ser contínuo desde o instante da pega."""
    cL = _contact(env, palm_sensors[0]); cR = _contact(env, palm_sensors[1])
    vL = _contact(env, back_sensors[0]); vR = _contact(env, back_sensors[1])
    return cL * cR * (1.0 - vL) * (1.0 - vR)


def _box_target_err_sq(env, object_name, command_name):
    obj: Entity = env.scene[object_name]
    target = env.command_manager.get_term(command_name).command[:, 0:3]  # world (env_origin cancela)
    return torch.sum(torch.square(target - obj.data.root_link_pos_w), dim=-1)


def reaching_reward(env, std_coarse, std_fine, object_name, asset_cfg, lateral_offset=0.0):
    """Aproximação em 2 escalas, SEMPRE ligada (grossa mantém gradiente de longe; fina
    premia proximidade). Sem gate de contato: pegar não custa mais recompensa (o vale
    antigo travava o robô pairando). Com lateral_offset>0, cada palma mira SUA face
    da caixa (pose pré-grasp = máximo do termo) — ver _palm_dists_sq."""
    dL, dR = _palm_dists_sq(env, object_name, asset_cfg, lateral_offset)
    return 0.5 * reaching_kernel(dL, dR, std_coarse) + 0.5 * reaching_kernel(dL, dR, std_fine)


def grasp_reward(env, palm_sensors, back_sensors):
    """Bônus por PREENSÃO estabelecida (2 palmas tocando, NENHUM verso) — recompensa o
    TOQUE em si: o degrau entre 'mãos na caixa' (reaching) e 'erguendo' (lift). Sem ele,
    pegar dá 0 até a caixa subir → o robô não tem gradiente pra aprender a pegar. 0/1."""
    return _grasp(env, palm_sensors, back_sensors)


def lift_reward(env, object_name, command_name, palm_sensors, back_sensors, rest_z,
                upright_std: float = 0.1, rest_z_attr: str | None = None):
    """Preensão × ORIENTAÇÃO × PROGRESSO de altura (0 na mesa -> 1 no alvo). Contínuo e
    monotônico, exige preensão (mata arremesso). O fator de ORIENTAÇÃO (alinhamento do
    eixo-up da caixa com o mundo) FECHA o hack de TOMBAR: deitar a caixa subia o centro
    de graça; tombada -> fator cai -> lift cai.

    KERNEL SUAVE (2026-07-16, substitui o corte duro em 10°): `exp(-(1-cos θ)/upright_std)`,
    onde cos θ = box_up_z (1=reta, 0=deitada). O corte duro só PUNIA a inclinação (0 além
    de 10°, gradiente nenhum de 11°..45°); o kernel dá gradiente em TODA a faixa puxando
    pra vertical -> a política é ativamente empurrada a manter nível, o que a força a
    PEGAR QUADRADO na face (não na quina) pra conseguir erguer reto. Diagnóstico (user,
    confirmado no play 07-16): grab rápido pega na quina -> caixa roda na mão -> chega
    tortа -> gate zerava o lift. std=0.1: fator 0.86@10°, 0.55@20°, 0.26@30°, 0.06@45°
    (demandante mas graduado; menor std = mais exigente)."""
    obj: Entity = env.scene[object_name]
    box_z = obj.data.root_link_pos_w[:, 2]
    target_z = env.command_manager.get_term(command_name).command[:, 2]
    # rest_z (altura de repouso da caixa = zero do progresso). ESCALAR por default;
    # com PLR de altura vira POR-ENV (cada altura tem seu zero) via buffer env.plr_rest_z
    # — senão as alturas baixas ficariam sem gradiente de lift (a caixa nasce abaixo do
    # rest_z fixo → progresso clampado em 0 na maior parte do erguer). Ver curriculums.py.
    rz = getattr(env, rest_z_attr) if rest_z_attr is not None else rest_z
    progress = torch.clamp((box_z - rz) / (target_z - rz), 0.0, 1.0)
    world_up = torch.zeros(box_z.shape[0], 3, device=box_z.device)
    world_up[:, 2] = 1.0
    box_up = quat_apply(obj.data.root_link_quat_w, world_up)   # eixo-up da caixa no mundo
    upright = torch.exp(-(1.0 - box_up[:, 2]) / upright_std)   # 1=reta, cai suave c/ tombo
    return _grasp(env, palm_sensors, back_sensors) * upright * progress


def sustain_precise_reward(env, std, object_name, command_name, palm_sensors, back_sensors):
    """Preensão × gaussiana APERTADA do erro caixa->alvo: recompensa segurar PARADO no
    alvo (precisão final). std pequeno -> ~0 fora do alvo (sem hack de segurar na mesa)."""
    err_sq = _box_target_err_sq(env, object_name, command_name)
    return _grasp(env, palm_sensors, back_sensors) * height_kernel(err_sq, std)


def back_penalty(env, back_sensors):
    """Penalidade por tocar a caixa com o verso (gradiente 'vire a palma')."""
    return _contact(env, back_sensors[0]) + _contact(env, back_sensors[1])


def table_contact_penalty(env, sensor_name):
    """Pune QUALQUER parte do corpo (exceto as mãos) tocando a mesa = apoiar-se.

    As mãos são pads ('*_pad') e o sensor mira só '*_collision' (coxa/antebraço/
    tronco/etc.) → as mãos ficam livres pra pegar a caixa. Boolean (0/1 por env):
    encostar coxa/antebraço na mesa já pune, sem depender de força/tempo."""
    return _contact(env, sensor_name)


def com_over_feet_penalty(env, asset_cfg, forward_margin=0.05):
    """Pune o CoM de CORPO INTEIRO derivando pra FRENTE do apoio dos pés (dump de peso).

    CoM REAL = `subtree_com` no corpo-raiz (soma ponderada por massa de TODO o robô,
    incluindo braços/tronco) — NÃO a pelvis, que é gameável: o robô mantém a pelvis
    atrás (sobre os pés, satisfazendo o upright que só mede o tronco) enquanto joga a
    massa pra frente e se escora com as mãos na caixa. O CoM real anda com a massa →
    não dá pra enganar. Só pune deriva pra FRENTE (recuar/lateral = ok). Codifica
    "auto-sustentado": se a caixa sumisse, CoM sobre os pés = continua de pé. A mão
    pode tocar/guiar (não penaliza contato); só o PESO indo pra frente é punido.

    Reward-only: o CoM é quantidade PRIVILEGIADA do sim; NÃO entra na obs do actor →
    o contrato sim-to-real fica intacto (o robô real não precisa medir CoM; a política
    aprende a manter o equilíbrio pela propriocepção). +X = frente (o robô encara +X)."""
    robot: Entity = env.scene[asset_cfg.name]
    com = robot.data.data.subtree_com[:, robot.data.indexing.root_body_id]  # [B,3] CoM real
    feet_center = robot.data.site_pos_w[:, asset_cfg.site_ids].mean(dim=1)  # [B,3] centro dos pés
    fwd = com[:, 0] - feet_center[:, 0] - forward_margin                    # +X = frente
    return torch.clamp(fwd, min=0.0) ** 2                                   # só frente, quadrático


def box_shake_penalty(env, object_name):
    """Pune SACUDIR a caixa: soma dos quadrados da velocidade ANGULAR da caixa.

    Alvo = manuseio violento (girar/balançar a caixa na mão), confirmado no play
    07-16 (grab rápido -> caixa roda na mão). Quadrático: sacudida grande custa
    caro, movimento estável passa barato. Só a ANGULAR — a linear NÃO, porque
    erguer é mover a caixa pra cima (penalizar linear brigaria com o lift).
    Complementa o kernel de orientação (kernel = não terminar torta; este = não
    ficar rodando no caminho). Reward-only: dinâmica da caixa é privilegiada do
    sim, NÃO entra na obs do actor -> contrato sim-to-real intacto."""
    obj: Entity = env.scene[object_name]
    return torch.sum(torch.square(obj.data.root_link_ang_vel_w), dim=-1)


def joint_deviation_l1(env, asset_cfg):
    """Penaliza |ângulo − keyframe default| (L1) das juntas em asset_cfg — pra HIP_ROLL/YAW.

    Ataca a 'perna esticada pro lado' / espacate: no play (model_15700) a hip_roll esquerda
    ficava aberta +53.7° (default 0) enquanto a direita agachava — base assimétrica, pose
    final não-natural. Por que L1 e não a posture existente: a posture é GAUSSIANA (exp(-dev²
    /std²)) → a 53° (~0.94 rad, std 0.5) ela SATURA (~0.03) e não tem mais gradiente pra
    rebocar. L1 dá gradiente CONSTANTE em qualquer desvio → puxa de volta mesmo de 53°.

    Escopo hip ROLL/YAW só (via joint_names no asset_cfg) — deixa hip/knee/ankle PITCH LIVRES,
    que são o agachar. Molde real-G1: IsaacLab joint_deviation_l1 (hip −0.1) / unitree_rl_gym
    hip_pos. feet_slip NÃO resolve isso (o splay é hip_roll estático com pé plantado → slip 0,
    invisível pro feet_slip)."""
    robot: Entity = env.scene[asset_cfg.name]
    dev = (robot.data.joint_pos[:, asset_cfg.joint_ids]
           - robot.data.default_joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.abs(dev), dim=-1)
