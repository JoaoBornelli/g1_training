"""Os eventos de reset. FASE F0/F1: posicionar a cena pelo nível.

⚠ UM evento por entidade. Dois eventos que escrevem a pose da MESMA entidade no
MESMO reset não se somam: o segundo APAGA o primeiro, sem erro e sem log. Portanto
`posiciona_cena` faz a prateleira E a caixa, e não existe um segundo evento tocando
nenhuma das duas.

⚠ Este evento LÊ `env.limpo_nivel`, escrito pelo termo de currículo. No reset o
mjlab roda CURRÍCULO -> EVENTOS -> COMANDO, portanto o nível já está lá.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from g1_limpo.curriculo import garante_elo, garante_nivel

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["posiciona_cena", "afasta_cena", "carga_caixa", "trava_robo",
           "segura_caixa", "orientacao_de_nascimento", "POSE_TRAVADA",
           "reset_base_por_elo", "avanca_elo_no_viewer"]


def reset_base_por_elo(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    elos_que_andam: tuple[int, ...],
    faixa_loco: dict,
    faixa_manipula: dict,
    velocidade: dict | None = None,
) -> None:
    """Reseta a pose da base com a faixa de yaw que o ELO daquele env pede.

        elo que ANDA        yaw ±3,14   a mobília está a +5 m, não há com que alinhar
        elo de MANIPULAÇÃO  yaw ±0,2    mobília de pose absoluta, à frente do robô

    ⚠ O ±0,2 GLOBAL foi o defeito central de um bloco medido: o erro de rumo era
    sempre minúsculo, o `track_angular_velocity` era satisfeito sem o robô fazer nada,
    e o canal de guinada nunca foi exercitado. Quando a política derivou para o giro,
    ela não tinha autoridade para sair.

    ⚠ E o ±3,14 GLOBAL é o defeito espelhado: um env de manipulação nasceria de costas
    para a prateleira, e a tarefa viraria sorte de sorteio.

    ⚠ ELE NÃO REIMPLEMENTA NADA. O evento do fabricante já aceita `env_ids`, portanto
    isto é um DESPACHANTE: chama `reset_root_state_uniform` uma vez por subconjunto.
    Reescrever a amostragem à mão perderia o `default_root_state`, o `env_origins` e o
    `quat_mul` com a orientação default — três coisas que o fabricante faz e que um
    transcritor esquece.
    """
    if len(env_ids) == 0:
        return
    elo = garante_elo(env)[env_ids]
    anda = torch.isin(elo, torch.tensor(elos_que_andam, device=env.device))
    for mascara, faixa in ((anda, faixa_loco), (~anda, faixa_manipula)):
        ids = env_ids[mascara]
        if len(ids):
            reset_root_state_uniform(env, ids, pose_range=dict(faixa),
                                     velocity_range=dict(velocidade or {}))


def _topo_por_nivel(env, env_ids, topo_min, topo_teto, jitter_z, n) -> torch.Tensor:
    """A altura do TOPO da prateleira, sorteada pela célula do nível.

    Só o PISO da faixa desce com o nível; o teto é o mesmo em todos. Consequência
    deliberada: **cada nível CONTÉM o anterior**, e a altura fácil nunca desaparece
    do treino no instante da promoção — que é o defeito da tabela discreta.
    """
    dev = env.device
    nivel = garante_nivel(env)[env_ids]
    piso = torch.tensor(topo_min, device=dev)[nivel]
    topo = piso + (topo_teto - piso) * torch.rand(n, device=dev)
    topo = topo + (2.0 * torch.rand(n, device=dev) - 1.0) * jitter_z
    return torch.maximum(topo, piso)      # o jitter nunca desce abaixo do piso


def orientacao_de_nascimento(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    voltas_max: tuple[int, ...],
    eixo_vertical: tuple[bool, ...],
    desalinho_max_deg: tuple[float, ...],
) -> torch.Tensor:
    """O quatérnion de nascimento da caixa, por QUARTOS DE VOLTA da célula do nível.

    O `reorientar` pede que a face MARCADA fique normal ao robô. Ela é constante,
    portanto a dificuldade está aqui: quantos quartos de volta a face marcada está da
    frente no instante do reset.

        0 voltas   a face marcada já está à frente, só torta pelo desalinho
        1 volta    uma das 4 faces ADJACENTES

    ⚠ TETO DE UMA VOLTA. A face marcada nunca nasce do lado OPOSTO: o robô só precisa
    aprender a girar no máximo 90°.

    ⚠ O sorteio é uniforme em `0..voltas_max`, portanto **cada nível CONTÉM o
    anterior** — o caso fácil nunca desaparece do treino.

    ⚠ O eixo VERTICAL (Y) só entra quando a célula o permite, e a razão é FÍSICA:
    girar em Z é PIVOTAR sobre a laje, e dá para empurrar com uma mão; girar em Y é
    TOMBAR, e exige erguer uma aresta de um cubo de 20 cm. São manipulações
    diferentes, e a segunda é muito mais difícil.
    """
    n = len(env_ids)
    dev = env.device
    nivel = garante_nivel(env)[env_ids]

    vmax = torch.tensor(voltas_max, device=dev)[nivel].float()
    vert = torch.tensor(eixo_vertical, device=dev)[nivel]
    desal = torch.deg2rad(torch.tensor(desalinho_max_deg, device=dev))[nivel]

    # uniforme em 0..vmax, inclusive
    voltas = (torch.rand(n, device=dev) * (vmax + 1.0)).floor().clamp(max=1.0)
    usa_y = vert & (torch.rand(n, device=dev) < 0.5)
    sinal = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    ang = voltas * (math.pi / 2.0) * sinal      # no máximo ±90°

    zeros = torch.zeros(n, device=dev)
    pitch = torch.where(usa_y, ang, zeros)
    yaw = torch.where(usa_y, zeros, ang)
    # o desalinho residual entra sempre, e é a tarefa do nível 0
    yaw = yaw + (2.0 * torch.rand(n, device=dev) - 1.0) * desal

    if not hasattr(env, "limpo_voltas"):
        env.limpo_voltas = torch.zeros(env.num_envs, device=dev)
    env.limpo_voltas[env_ids] = voltas
    return quat_from_euler_xyz(zeros, pitch, yaw)


def posiciona_cena(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    topo_min: tuple[float, ...],
    jitter_x_max: tuple[float, ...],
    topo_teto: float,
    jitter_z: float,
    prateleira_xy: tuple[float, float],
    prateleira_meia_z: float,
    caixa_xy: tuple[float, float],
    caixa_jitter_y: tuple[float, float],
    caixa_meia_z: float,
    voltas_max: tuple[int, ...],
    eixo_vertical: tuple[bool, ...],
    desalinho_max_deg: tuple[float, ...],
) -> None:
    """Põe a prateleira na altura sorteada e a caixa em repouso em cima dela.

    Publica `env.limpo_topo` — o topo REAL de cada env. Ele é o zero de progresso do
    erguer, e sem ele "subir 5 cm" significaria coisas diferentes em níveis
    diferentes.
    """
    n = len(env_ids)
    dev = env.device
    caixa: Entity = env.scene["box"]
    mesa: Entity = env.scene["table"]
    origem = env.scene.env_origins[env_ids]

    topo = _topo_por_nivel(env, env_ids, topo_min, topo_teto, jitter_z, n)
    if not hasattr(env, "limpo_topo"):
        env.limpo_topo = torch.zeros(env.num_envs, device=dev)
    env.limpo_topo[env_ids] = topo

    # --- a prateleira. É MOCAP: pose direta, sem velocidade. A pose é o CENTRO do
    #     corpo, portanto ela fica `meia_z` ABAIXO do topo pedido.
    #     Layout do quat: (w, x, y, z) -> `pose[:, 3] = 1.0` é identidade.
    pose_mesa = torch.zeros(n, 7, device=dev)
    pose_mesa[:, 0] = origem[:, 0] + prateleira_xy[0]
    pose_mesa[:, 1] = origem[:, 1] + prateleira_xy[1]
    pose_mesa[:, 2] = topo - prateleira_meia_z
    pose_mesa[:, 3] = 1.0
    mesa.write_mocap_pose_to_sim(pose_mesa, env_ids=env_ids)

    # --- a caixa, em repouso no topo.
    #     ⚠ o jitter em x é de UM LADO só (0 .. jx_max): ele afasta a caixa do robô,
    #     nunca a aproxima. E ele APERTA com o nível, porque com o topo a 0,04 m as
    #     poses de pega só existem até x relativo ~0,45 m.
    nivel = garante_nivel(env)[env_ids]
    jx = torch.tensor(jitter_x_max, device=dev)[nivel]
    dx = jx * torch.rand(n, device=dev)
    dy = caixa_jitter_y[0] + (caixa_jitter_y[1] - caixa_jitter_y[0]) * torch.rand(n, device=dev)

    pose_caixa = torch.zeros(n, 7, device=dev)
    pose_caixa[:, 0] = origem[:, 0] + caixa_xy[0] + dx
    pose_caixa[:, 1] = origem[:, 1] + caixa_xy[1] + dy
    pose_caixa[:, 2] = topo + caixa_meia_z
    # ⚠ A ORIENTAÇÃO não é mais um jitter de yaw: ela é o eixo do `reorientar`,
    # sorteada em QUARTOS DE VOLTA pela célula do nível.
    pose_caixa[:, 3:7] = orientacao_de_nascimento(
        env, env_ids, voltas_max=voltas_max, eixo_vertical=eixo_vertical,
        desalinho_max_deg=desalinho_max_deg)
    caixa.write_root_link_pose_to_sim(pose_caixa, env_ids=env_ids)
    caixa.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev),
                                          env_ids=env_ids)


def afasta_cena(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    afasta_z: float,
    prateleira_xy: tuple[float, float],
    prateleira_meia_z: float,
    caixa_meia_z: float,
) -> None:
    """Sobe a mobília, para a forma de LOCOMOÇÃO pura. O chão fica livre.

    É a razão de o yaw de reset poder ser ±3,14 na locomoção: não sobra mobília de
    pose absoluta com que alinhar o rumo.

    ⚠ FASE F0/F1: chamado com TODOS os envs pelo `inspeciona.py`, para conferir. Na
    F2 ele passa a ser gateado pelo slot `ANDAR` do one-hot.
    """
    n = len(env_ids)
    dev = env.device
    mesa: Entity = env.scene["table"]
    caixa: Entity = env.scene["box"]
    origem = env.scene.env_origins[env_ids]

    pose = torch.zeros(n, 7, device=dev)
    pose[:, 0] = origem[:, 0] + prateleira_xy[0]
    pose[:, 1] = origem[:, 1] + prateleira_xy[1]
    pose[:, 2] = afasta_z - prateleira_meia_z
    pose[:, 3] = 1.0
    mesa.write_mocap_pose_to_sim(pose, env_ids=env_ids)

    pose_c = pose.clone()
    pose_c[:, 2] = afasta_z + caixa_meia_z    # apoiada na laje erguida
    caixa.write_root_link_pose_to_sim(pose_c, env_ids=env_ids)
    caixa.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev),
                                          env_ids=env_ids)

    if hasattr(env, "limpo_topo"):
        env.limpo_topo[env_ids] = afasta_z


def carga_caixa(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    carga_max: tuple[float, ...],
    massa_base: float,
) -> None:
    """Sorteia a carga da caixa e a aplica como FORÇA EXTERNA vertical.

    ⚠ NUNCA `dr.body_mass` nem `dr.pseudo_inertia`: os dois corrompem a heap
    (CUDA illegal memory access). Está MEDIDO no repositório, e é o mesmo tipo de
    defeito do `base_com`.

    ⚠ CONSEQUÊNCIA DECLARADA: a caixa de 5 kg fica com a INÉRCIA de 1 kg. A
    randomização endurece a ESTÁTICA, e não a dinâmica. É o preço de não poder tocar
    a massa de verdade.

    Publica `env.limpo_massa` em **kg**, e não o peso em newtons. A massa é a
    grandeza primitiva: o `unload` deriva `m·g` dela, e o `squeeze` deriva
    `F_ref = m·g/(2µ)`. Publicar newtons obrigaria um dos dois a desfazer a conta, e
    é assim que se erra um fator 9,81 em silêncio.
    """
    n = len(env_ids)
    dev = env.device
    caixa: Entity = env.scene["box"]
    nivel = garante_nivel(env)[env_ids]

    # o TETO vem da célula do nível; o PISO é sempre a massa do geom
    teto = torch.tensor(carga_max, device=dev)[nivel]
    kg = massa_base + (teto - massa_base).clamp(min=0.0) * torch.rand(n, device=dev)

    if not hasattr(env, "limpo_massa"):
        env.limpo_massa = torch.full((env.num_envs,), massa_base, device=dev)
    env.limpo_massa[env_ids] = kg

    # ⚠ `forces` tem shape (N, num_bodies, 3). A caixa tem 1 body.
    forcas = torch.zeros(n, 1, 3, device=dev)
    forcas[:, 0, 2] = -(kg - massa_base) * 9.81   # o resto já vem da massa do geom
    caixa.write_external_wrench_to_sim(
        forces=forcas, torques=torch.zeros_like(forcas), env_ids=env_ids)


# A pose em que o robô fica TRAVADO na inspeção, relativa à origem do env. Ela é
# CANÔNICA e compartilhada: o `segura_caixa` a usa para saber onde fica o peito.
POSE_TRAVADA = (0.0, 0.0, 0.80)


def trava_robo(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    pose: tuple[float, float, float] = POSE_TRAVADA,
) -> None:
    """TRAVA o robô numa pose CONHECIDA. SÓ PARA INSPEÇÃO — nunca no treino.

    Existe porque a revisão visual pede a cena PARADA: sem isto, um robô sem
    política cai em meio segundo e não dá para conferir alvo nenhum.

    ⚠ Ele NÃO preserva a pose sorteada pelo `reset_base` — ele impõe a
    `POSE_TRAVADA`. É deliberado: uma pose conhecida e igual em todos os envs é o
    que torna a conferência visual comparável, e é o que permite ao `segura_caixa`
    saber onde fica o peito sem ler pose nenhuma.
    """
    n = len(env_ids)
    dev = env.device
    robot: Entity = env.scene["robot"]
    origem = env.scene.env_origins[env_ids]

    p = torch.zeros(n, 7, device=dev)
    p[:, 0] = origem[:, 0] + pose[0]
    p[:, 1] = origem[:, 1] + pose[1]
    p[:, 2] = pose[2]
    p[:, 3] = 1.0
    robot.write_root_link_pose_to_sim(p, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev),
                                          env_ids=env_ids)
    robot.write_joint_velocity_to_sim(
        torch.zeros(n, robot.data.joint_pos.shape[-1], device=dev), env_ids=env_ids)
    # ⚠ As JUNTAS também são pinadas. Sem isto o robô "trava" a base mas desmonta as
    # pernas, e a pose deixa de ser comparável entre envs.
    robot.write_joint_position_to_sim(
        robot.data.default_joint_pos[env_ids], env_ids=env_ids)


def segura_caixa(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    peito_b: tuple[float, float, float] = (0.25, 0.00, 0.15),
) -> None:
    """Põe a caixa na altura do PEITO, no frame da base. SÓ PARA INSPEÇÃO.

    Os elos `CARREGAR` e `BOTAR` começam com a caixa JÁ nas mãos, porque no treino
    eles são o 2º elo de uma cadeia e o robô a pegou no 1º. Para INSPECIONAR esses
    dois é preciso pôr a caixa lá sem que ninguém a pegue.

    ⚠ Isto é colocação GEOMÉTRICA, e não uma pega: nada segura a caixa, e ela cai se
    a física rodar. Com o robô travado e o episódio curto isso não importa — o que se
    quer ver é o ALVO, e o alvo do `BOTAR` depende do fundo da caixa.

    ⚠ Na F4 este evento ganha uso de treino: ele é o que permite treinar uma cadeia
    começando pelo 2º elo, que é o `reset_segurando` do g1_multitask.

    ⚠⚠ ARMADILHA MEDIDA EM 25/08, e ela custa uma sessão a quem não souber:
    **`robot.data.root_link_pos_w` está OBSOLETO dentro de um evento de reset.** O
    `reset_base` escreveu na simulação, mas os buffers de `data` só são recomputados
    no forward seguinte. Lendo a pose aqui, a caixa foi para `(0.25, 0, 0.94)` nos
    TRÊS envs — sem a origem do env, porque a leitura devolveu o keyframe cru.

    Portanto este evento **não lê pose nenhuma**. Ele usa a `POSE_TRAVADA`, que é
    a pose que o `trava_robo` impõe, e soma o `peito_b` a ela. Determinístico, e sem
    depender da ordem de recomputação dos buffers.
    """
    n = len(env_ids)
    dev = env.device
    caixa: Entity = env.scene["box"]
    origem = env.scene.env_origins[env_ids]

    pose = torch.zeros(n, 7, device=dev)
    pose[:, 0] = origem[:, 0] + POSE_TRAVADA[0] + peito_b[0]
    pose[:, 1] = origem[:, 1] + POSE_TRAVADA[1] + peito_b[1]
    pose[:, 2] = POSE_TRAVADA[2] + peito_b[2]
    pose[:, 3] = 1.0
    caixa.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    caixa.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev),
                                          env_ids=env_ids)


def avanca_elo_no_viewer(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    nome_do_comando: str = "alvo_caixa",
) -> None:
    """Dispara o avanço de elo à mão. SÓ PARA O VISUALIZADOR.

    ⚠ POR QUE UM EVENTO DE INTERVALO, e não uma chamada no laço do viewer: o
    `run_play` do mjlab roda o próprio laço e não expõe um gancho por passo. O evento de
    intervalo é o mesmo idioma que o `trava_robo` já usa, e por isso não exige
    reescrever o `run_play` — que foi o motivo pelo qual este caminho ficou como no-op
    na primeira tentativa.

    ⚠ IDEMPOTENTE por construção: o `_avanca_elo_force` só avança quem tem elo seguinte
    na cadeia, e marca `fechou` no último. Portanto disparar de novo não faz nada. É o
    que permite ao evento repetir sem estragar o que se está olhando.

    ⚠ E ele NÃO faz nada num env sem cadeia (`CADEIA_NENHUMA`): o `--avanca-elo` só tem
    sentido com uma cadeia forçada, e o inspetor recusa a combinação inválida.
    """
    del env_ids
    termo = env.command_manager.get_term(nome_do_comando)
    todos = torch.arange(env.num_envs, device=env.device)
    termo.forca_avanco(todos)


def entrega_tarefa_no_viewer(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    elo_novo: int,
    cena: dict,
    entrega_apos_s: float,
    nome_do_comando: str = "alvo_caixa",
) -> None:
    """A cena do `pegar` POSTA, o robô PARADO, e a tarefa chegando aos N segundos.

    ⚠⚠ SÓ PARA O VISUALIZADOR, e ela simula o DEPLOY: a caixa está na laje à vista do
    robô desde o começo, o robô está de pé com comando de velocidade ZERO, e um
    operador manda "pega a caixa". No treino isto NÃO existe — o elo é sorteado no
    reset e nunca troca no meio.

    ⚠ ELA RODA A CADA PASSO (`interval_range_s = (dt, dt)`), que é o idioma do
    `trava_robo`. O `run_play` do mjlab roda o próprio laço e não expõe gancho por
    passo; sem o evento de intervalo este caminho seria um no-op, que foi como o
    `--avanca-elo` nasceu quebrado.

    DUAS COISAS, e cada uma existe por um motivo:

    1. **A MOBÍLIA VOLTA.** No `ANDAR` o termo de comando manda a laje a +5 m com a
       caixa em cima, e ele roda DEPOIS dos eventos de reset (`currículo -> eventos ->
       comando`) — portanto um `posiciona_cena` no reset seria desfeito. O teste é
       `caixa guardada`, e não um contador: ele é idempotente por construção (uma vez
       posta, a caixa deixa de estar guardada) e se refaz sozinho depois de um reset.

    2. **A ENTREGA**, por `episode_length_buf`, portanto POR ENV. Um cronômetro global
       entregaria a tarefa no meio do episódio de quem resetou fora de fase.

    ⚠⚠ O TWIST **NÃO** É ZERADO AQUI, e a primeira versão o zerava — estava errado. O
    `reset()` chama `command_manager.compute(dt=0.0)` e NÃO roda evento de intervalo,
    portanto a PRIMEIRA observação de todo episódio saía com comando de até 2 m/s
    (`cmd_obs_max = 1,97` medido). A política dava o primeiro passo contra "ande a
    2 m/s" e depois tinha de frear o que ela mesma começou: deriva lateral lenta no
    viewer. Quem zera é o `_zera_twist_nos_parados`, que roda DENTRO da passada do
    command manager e por isso cobre o reset — o `env_cfg` põe o `ANDAR` em
    `elos_parados` neste modo.
    """
    from g1_limpo.comando import ANDAR

    del env_ids
    termo = env.command_manager.get_term(nome_do_comando)
    anda = termo._elo == ANDAR
    if not bool(anda.any()):
        return

    # 1. a mobília. `guardada` é o próprio estado, e não um contador de passos.
    z = (env.scene["box"].data.root_link_pos_w[:, 2]
         - env.scene.env_origins[:, 2])
    posta = anda & (z > 4.0)
    if bool(posta.any()):
        posiciona_cena(env, posta.nonzero().flatten(), **cena)

    # 2. a entrega, por env.
    t = env.episode_length_buf.float() * env.step_dt
    entrega = anda & (t >= entrega_apos_s)
    if bool(entrega.any()):
        termo.recebe_tarefa(entrega.nonzero().flatten(), int(elo_novo))
