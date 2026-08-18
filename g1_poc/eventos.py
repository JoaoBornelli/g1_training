"""Os eventos de reset do g1_poc.

Três eventos, e a ORDEM no dict importa (o event manager percorre por ordem de
inserção):

    1. reset_cena    põe a prateleira na altura sorteada e a caixa em cima dela
    2. carga_caixa   sorteia a carga e a aplica como força externa
    3. afasta_cena   sobe a mobília 5 m nos envs da forma de LOCOMOÇÃO

⚠ Todos leem `env.poc_manipula`, que é escrito pelo CURRÍCULO. No mjlab a ordem no
reset é currículo → eventos → comando. Portanto o currículo é o único lugar de onde
os eventos conseguem ler a forma do episódio.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _forma(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """[n] bool — True onde o episódio é de manipulação."""
    manipula = getattr(env, "poc_manipula", None)
    if manipula is None:
        return torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    return manipula[env_ids]


def reset_cena(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    topo_piso: float,
    topo_teto: float,
    jitter_z: float,
    meia_z: float,
    caixa_meia_z: float,
    caixa_xy: tuple[float, float],
    prateleira_xy: tuple[float, float],
    jitter_x: tuple[float, float],
    jitter_y: tuple[float, float],
    jitter_yaw_deg: float,
) -> None:
    """Põe a prateleira na altura sorteada, e a caixa em repouso em cima dela.

    A altura é sorteada na faixa `[topo_piso, topo_teto]`. No esqueleto a faixa é
    degenerada (nível 0 → 0,55 fixo); o currículo a alarga no passo 4.

    Grava `env.poc_topo` — o crítico o observa, e o `botar` o usará.
    """
    n = len(env_ids)
    dev = env.device
    caixa: Entity = env.scene["box"]
    mesa: Entity = env.scene["table"]
    origem = env.scene.env_origins[env_ids]

    # --- a altura do topo ---
    topo = topo_piso + (topo_teto - topo_piso) * torch.rand(n, device=dev)
    topo = topo + (2.0 * torch.rand(n, device=dev) - 1.0) * jitter_z
    topo = topo.clamp(min=topo_piso)
    if not hasattr(env, "poc_topo"):
        env.poc_topo = torch.zeros(env.num_envs, device=dev)
    env.poc_topo[env_ids] = topo

    # --- a prateleira (mocap: pose direta, sem velocidade) ---
    pose_mesa = torch.zeros(n, 7, device=dev)
    pose_mesa[:, 0] = origem[:, 0] + prateleira_xy[0]
    pose_mesa[:, 1] = origem[:, 1] + prateleira_xy[1]
    pose_mesa[:, 2] = topo - meia_z
    pose_mesa[:, 3] = 1.0
    mesa.write_mocap_pose_to_sim(pose_mesa, env_ids=env_ids)

    # --- a caixa, em repouso no topo ---
    dx = jitter_x[0] + (jitter_x[1] - jitter_x[0]) * torch.rand(n, device=dev)
    dy = jitter_y[0] + (jitter_y[1] - jitter_y[0]) * torch.rand(n, device=dev)
    yaw_max = math.radians(jitter_yaw_deg)
    yaw = (2.0 * torch.rand(n, device=dev) - 1.0) * yaw_max
    zeros = torch.zeros(n, device=dev)
    quat = quat_from_euler_xyz(zeros, zeros, yaw)

    pose_caixa = torch.zeros(n, 7, device=dev)
    pose_caixa[:, 0] = origem[:, 0] + caixa_xy[0] + dx
    pose_caixa[:, 1] = origem[:, 1] + caixa_xy[1] + dy
    pose_caixa[:, 2] = topo + caixa_meia_z
    pose_caixa[:, 3:7] = quat
    caixa.write_root_link_pose_to_sim(pose_caixa, env_ids=env_ids)
    caixa.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


def carga_caixa(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    faixa_kg: tuple[float, float],
    massa_base: float,
) -> None:
    """Sorteia a carga da caixa e a aplica como FORÇA EXTERNA vertical.

    ⚠ NUNCA `dr.body_mass` nem `dr.pseudo_inertia`: os dois corrompem a heap
    (CUDA illegal memory access). Está medido no repositório.

    Consequência declarada: a caixa de 5 kg tem a inércia de 1 kg. A DR endurece a
    estática, e não a dinâmica.

    Grava `env.poc_massa` — o `squeeze` a usa para calcular `F_ref = m·g/(2μ)`.
    """
    n = len(env_ids)
    dev = env.device
    caixa: Entity = env.scene["box"]

    if not hasattr(env, "poc_massa"):
        env.poc_massa = torch.full((env.num_envs,), massa_base, device=dev)
    kg = faixa_kg[0] + (faixa_kg[1] - faixa_kg[0]) * torch.rand(n, device=dev)
    env.poc_massa[env_ids] = kg

    forcas = torch.zeros(n, 1, 3, device=dev)
    forcas[:, 0, 2] = -(kg - massa_base) * 9.81
    caixa.write_external_wrench_to_sim(
        forces=forcas, torques=torch.zeros_like(forcas), env_ids=env_ids
    )


def afasta_cena(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    altura: float,
    meia_z: float,
    caixa_meia_z: float,
    caixa_xy: tuple[float, float],
    prateleira_xy: tuple[float, float],
) -> None:
    """Sobe a caixa e a prateleira nos envs da forma de LOCOMOÇÃO.

    Duas razões, e as duas importam:

    1. A prateleira sai da frente. O robô anda sem obstáculo. Sem isto, mandá-lo
       andar o faz bater na mesa.
    2. O robô não alcança a caixa. Portanto ele não coleta `reaching` de graça
       durante um episódio de locomoção.

    A política NÃO vê este movimento: com `caixa_valida = 0` os quatro canais de
    caixa são zerados. É o mesmo estado do robô real quando não existe caixa.
    """
    manipula = _forma(env, env_ids)
    ids = env_ids[~manipula]
    if len(ids) == 0:
        return
    n = len(ids)
    dev = env.device
    caixa: Entity = env.scene["box"]
    mesa: Entity = env.scene["table"]

    origem = env.scene.env_origins[ids]

    # ⚠ NÃO ler a pose aqui. O `reset_cena` acabou de escrever a pose da caixa e da
    # prateleira, e as grandezas derivadas só são recalculadas no próximo
    # `forward()`. Ler devolveria a pose ANTERIOR ao reset — o gotcha que o
    # `g1_multitask/events.py` documenta e que o `afasta_cena` de lá pisou.
    # A pose é RECONSTRUÍDA das mesmas entradas do `reset_cena`.
    pose_mesa = torch.zeros(n, 7, device=dev)
    pose_mesa[:, 0] = origem[:, 0] + prateleira_xy[0]
    pose_mesa[:, 1] = origem[:, 1] + prateleira_xy[1]
    pose_mesa[:, 2] = altura - meia_z
    pose_mesa[:, 3] = 1.0
    mesa.write_mocap_pose_to_sim(pose_mesa, env_ids=ids)

    pose_caixa = torch.zeros(n, 7, device=dev)
    pose_caixa[:, 0] = origem[:, 0] + caixa_xy[0]
    pose_caixa[:, 1] = origem[:, 1] + caixa_xy[1]
    pose_caixa[:, 2] = altura + caixa_meia_z
    pose_caixa[:, 3] = 1.0
    caixa.write_root_link_pose_to_sim(pose_caixa, env_ids=ids)
    caixa.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=ids)
