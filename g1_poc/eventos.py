"""Os eventos de reset do g1_poc.

Três eventos, e a ORDEM no dict importa (o event manager percorre por ordem de
inserção):

    1. reset_cena    põe a prateleira na altura sorteada e a caixa em cima dela
    2. carga_caixa   sorteia a carga e a aplica como força externa
    3. afasta_cena   sobe a mobília 5 m nos envs da forma de LOCOMOÇÃO

⚠ Todos leem `env.poc_manipula`, que é escrito pelo CURRÍCULO. `reset_cena` e
`carga_caixa` também leem `env.poc_nivel`, escrito pelo mesmo currículo, que roda
inteiro antes dos eventos — o `getattr` com `None` é defesa, não necessidade
(medido em 20/08).

⚠ NÃO toque no `afasta_cena`. Medido em 20/08: a caixa afastada fica apoiada na
laje a 5 m (z = 5,099 após 120 passos, mesmo com 5 kg de wrench).

No mjlab a ordem no reset é currículo → eventos → comando. Portanto o currículo é
o único lugar de onde os eventos conseguem ler a forma do episódio.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

_ROBOT_CFG = SceneEntityCfg("robot")

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
    topo_min_por_nivel: tuple[float, ...],
    topo_teto: float,
    jitter_z: float,
    meia_z: float,
    caixa_meia_z: float,
    caixa_xy: tuple[float, float],
    prateleira_xy: tuple[float, float],
    jitter_x_max_por_nivel: tuple[float, ...],
    jitter_y: tuple[float, float],
    jitter_yaw_deg: float,
) -> None:
    """Põe a prateleira na altura sorteada, e a caixa em repouso em cima dela.

    A altura (PISO) é sorteada por CÉLULA do nível; o TETO é 0,55 m em todos os
    níveis (§10.1). O jitter x também vem da célula do nível. O jitter z e o y
    são fixos.

    Grava `env.poc_topo` — o crítico o observa, e o `botar` o usará.
    """
    n = len(env_ids)
    dev = env.device
    caixa: Entity = env.scene["box"]
    mesa: Entity = env.scene["table"]
    origem = env.scene.env_origins[env_ids]

    # --- a altura do topo, pela CÉLULA do nível (§10.1) ---
    # Só o PISO da faixa desce com o nível; o teto é 0,55 m em todos. No nível 0 a
    # faixa é degenerada em 0,55 — a cena de antes da tabela, número por número.
    # ⚠ `poc_nivel` já existe aqui mesmo no primeiro reset (o currículo roda
    # inteiro ANTES dos eventos); o `getattr` é defensivo, não necessário.
    nivel = getattr(env, "poc_nivel", None)
    if nivel is None:
        piso = torch.full((n,), topo_min_por_nivel[0], device=dev)
        jx_max = torch.full((n,), jitter_x_max_por_nivel[0], device=dev)
    else:
        piso = torch.tensor(topo_min_por_nivel, device=dev)[nivel[env_ids]]
        jx_max = torch.tensor(jitter_x_max_por_nivel, device=dev)[nivel[env_ids]]
    topo = piso + (topo_teto - piso) * torch.rand(n, device=dev)
    topo = topo + (2.0 * torch.rand(n, device=dev) - 1.0) * jitter_z
    topo = torch.maximum(topo, piso)
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
    # o jitter x APERTA com o nível: no topo a 0,04 m o alcance acaba em ~0,45 m
    dx = jx_max * torch.rand(n, device=dev)
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
    carga_max_por_nivel: tuple[float, ...],
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
    # o TETO vem da célula do nível; o PISO é sempre `massa_base` (§10.1)
    nivel = getattr(env, "poc_nivel", None)
    if nivel is None:
        teto = torch.full((n,), carga_max_por_nivel[0], device=dev)
    else:
        teto = torch.tensor(carga_max_por_nivel, device=dev)[nivel[env_ids]]
    kg = massa_base + (teto - massa_base).clamp(min=0.0) * torch.rand(n, device=dev)
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


def entrega_do_navegador(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    faixa: dict,
) -> None:
    """Dá velocidade residual de base SÓ aos envs da forma de manipulação (§11.1).

    ⚠ Antes de 21/08 isto vivia no `velocity_range` do `reset_base`, que é GLOBAL.
    Portanto TODO episódio nascia com um empurrão, inclusive os de locomoção, e
    desde a iteração 0, com uma política que ainda não anda.

    E o controlador de forma amplifica o erro: com a locomoção morrendo em 35
    passos, ele sorteia 91% de locomoção para equalizar a fatia de transições
    (correto, e medido: `frac_loco_sorteio = 0,9128`). Portanto o empurrão caía 91%
    das vezes justamente na habilidade que não funcionava.

    A §11.1 sempre pediu o contrário: a velocidade residual existe para treinar a
    ENTREGA DO NAVEGADOR, que só acontece antes de um elo de manipulação. No robô
    real o navegador entrega um robô com velocidade e com erro de rumo; um episódio
    de locomoção começa do zero, como no `velocity` do fabricante.

    Roda DEPOIS do `reset_base` (ordem do dict): ele põe a pose e zera a velocidade,
    e este termo a reescreve onde é devido.
    """
    manipula = _forma(env, env_ids)
    ids = env_ids[manipula]
    if len(ids) == 0:
        return
    n = len(ids)
    dev = env.device
    robot: Entity = env.scene["robot"]

    vel = torch.zeros(n, 6, device=dev)
    for i, chave in enumerate(("x", "y", "z")):
        lo, hi = faixa.get(chave, (0.0, 0.0))
        vel[:, i] = lo + (hi - lo) * torch.rand(n, device=dev)
    for i, chave in enumerate(("roll", "pitch", "yaw")):
        lo, hi = faixa.get(chave, (0.0, 0.0))
        vel[:, 3 + i] = lo + (hi - lo) * torch.rand(n, device=dev)
    robot.write_root_link_velocity_to_sim(vel, env_ids=ids)


def reset_base_por_forma(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    faixa_loco: dict,
    faixa_manipula: dict,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> None:
    """O `reset_base` do fabricante, com faixa de pose POR FORMA (21/08).

    Substitui `cfg.events["reset_base"]`. Ele não reimplementa nada: chama o
    `reset_root_state_uniform` do mjlab DUAS vezes, em subconjuntos disjuntos, com
    faixas diferentes. Portanto a matemática, a soma de `env_origins` e o
    tratamento de base flutuante continuam sendo os do fabricante.

    Motivo, e é o defeito central do bloco 2:

    O fabricante sorteia `yaw: (-3.14, 3.14)` — o CÍRCULO INTEIRO. Com
    `heading_command=True` e `rel_heading_envs = 0,30`, o ωz comandado vem do erro
    de rumo. Com ±3,14 esse erro cobre toda a faixa, e o robô TEM de aprender a
    girar desde a iteração 0.

    Nós sorteávamos ±0,2 rad, porque a mobília tem pose ABSOLUTA e o robô precisa
    nascer olhando para ela. Consequência: o erro de rumo era sempre minúsculo, o
    `track_angular_velocity` era satisfeito sem fazer nada, e o canal de yaw nunca
    foi exercitado. Medido na it 1216: ωz real de ~1,5 rad/s com comando ZERO, e
    `track_angular_velocity` 51× abaixo do linear COM PESO IGUAL.

    Na LOCOMOÇÃO o `afasta_cena` sobe a mobília 5 m. Não existe nada com que
    alinhar o rumo. Portanto ali o ±3,14 é de graça — e é a receita do fabricante,
    que já está comprovada.

    ⚠ Este termo tem de vir ANTES do `entrega_do_navegador` no dict de eventos. O
    `reset_root_state_uniform` zera a velocidade da base, e a entrega a reescreve.
    Trocar a ordem apagaria a entrega em silêncio.
    """
    manipula = _forma(env, env_ids)
    for ids, faixa in ((env_ids[~manipula], faixa_loco),
                       (env_ids[manipula], faixa_manipula)):
        if len(ids) == 0:
            continue
        envs_mdp.reset_root_state_uniform(
            env, ids, pose_range=faixa, velocity_range={}, asset_cfg=asset_cfg)
