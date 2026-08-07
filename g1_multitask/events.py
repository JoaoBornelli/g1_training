"""Eventos de reset próprios do multi-tarefa.

Quatro:

  - `reset_segurando` — a condição de spawn "segurando" das 3 tarefas que começam com
    a caixa nas mãos (`parado c/ caixa`, `andar c/ caixa`, `botar`);
  - `afasta_cena` — tira a prateleira do caminho de quem anda;
  - `payload_por_nivel` — o eixo `peso` do currículo virando força na caixa (S1);
  - `jitter_yaw_caixa` — devolve o jitter de yaw que o `reset_box` fazia (S1).

Por que o `reset_segurando` existe: medido em 30/07, sem ele essas 3 tarefas não têm
caminho de aquisição nenhum. Todos os termos de tarefa dão exatamente 0.0 no reset — a
caixa nasce em cima da prateleira, e `reaching`/`grasp`/`lift` são gateados só no
`pegar` pela §6b. O robô não tem gradiente pra pegar a caixa, e o único termo que
pontuaria exige uma preensão que ele nunca vai estabelecer. O doc pede a condição na §3
("'andar com caixa' não é tarefa nova; é a tarefa 'andar' com condição de spawn
'segurando'") e na §4 (linha do gatilho), mas ela não existia no código.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1_training.common.robot import PALM_SITES

from . import tasks as T

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_PALMAS = SceneEntityCfg("robot", site_names=list(PALM_SITES))
_GRAVIDADE = 9.81


def reset_segurando(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    pose_bracos: dict[str, float],
    tarefas: tuple[int, ...],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> None:
    """Põe os braços na pose de pré-grasp e a caixa entre as palmas.

    Só nos envs cuja tarefa sorteada está em `tarefas`. Lê `env.tarefa_sorteada`, que
    o currículo escreveu 6 linhas antes (`:554` contra `:560`) — é por isso que o
    sorteio mora no currículo e não no termo de comando, que roda depois (`:581`).

    ⚠️ **As palmas apenas TOCAM a caixa; não a apertam.** Força normal zero, atrito
    zero, e com ação nula a caixa escorrega 22 cm em 0.5 s. Isso é correto: segurar é
    o que a tarefa ensina. Ver `pregrasp.py`.

    **Ordem interna importa.** Escreve as juntas, força um `forward()`, e só então lê
    a posição das palmas. Sem o `forward()` os `site_pos_w` estariam STALE — a
    escrita mexeu em `qpos`, e a cinemática só roda no passo seguinte. É o mesmo
    gotcha que quebrava o `dir_alvo` do `reorientar` (ver `commands.py`), aqui evitado
    com um forward explícito em vez de resolução preguiçosa, porque um evento de reset
    não tem "próximo ponto de leitura"."""
    if env_ids is None or len(env_ids) == 0:
        return
    quais = torch.tensor(tarefas, device=env.device)
    mask = (env.tarefa_sorteada[env_ids].unsqueeze(-1) == quais).any(dim=-1)
    ids = env_ids[mask]
    if len(ids) == 0:
        return

    robo: Entity = env.scene[robot_cfg.name]
    caixa: Entity = env.scene[box_cfg.name]

    # 1. juntas do braço = a pose autorada. Velocidade zero: o braço não deve nascer
    #    em movimento, senão o 1º passo já joga a caixa.
    q = robo.data.joint_pos[ids].clone()
    for nome, valor in pose_bracos.items():
        idx, _ = robo.find_joints([nome])
        q[:, idx] = valor
    dq = torch.zeros_like(q)
    robo.write_joint_state_to_sim(q, dq, env_ids=ids)

    # 2. cinemática atualizada, senão o passo 3 lê palma na pose ANTIGA
    env.sim.forward()

    # 3. caixa no ponto médio das palmas, parada. A orientação vem do que o
    #    `reset_box` já sorteou (jitter de yaw de ±15°), então o `reorientar` e o
    #    `pegar` continuam vendo a mesma distribuição de spawn.
    _PALMAS.resolve(env.scene)
    meio = robo.data.site_pos_w[:, _PALMAS.site_ids].mean(dim=1)[ids]
    estado = torch.cat(
        [meio, caixa.data.root_link_quat_w[ids], torch.zeros(len(ids), 6,
                                                             device=env.device)],
        dim=-1,
    )
    caixa.write_root_state_to_sim(estado, env_ids=ids)


def payload_dr(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    box_mass: float,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> None:
    """A DR de carga da caixa, POR ENV, em dois níveis.

    | nível de DR | massa da caixa |
    |---|---|
    | 0 | `1 kg` fixo |
    | 1 | `U(1, 5)` kg |

    É o `apply_box_payload` da Lift (`g1_training/common/events.py:190`) com uma
    diferença só: lá a massa efetiva sai de um `weight_range` GLOBAL; aqui o TETO sai
    do booleano que o currículo abriu para a tarefa daquele env. A física é a mesma —
    `write_external_wrench_to_sim` com força dirigida em −z no COM da caixa.

    ⚠️ **O peso NÃO é eixo de currículo.** Ele não tem célula, não tem EMA e não tem
    portão próprio, porque o sucesso não é atrelado à massa — o critério é "fez o que
    tinha que fazer". O booleano `env.dr_peso` vira `True` no PRIMEIRO evento da
    tarefa, antes de o eixo específico avançar.

    ⚠️ **Massa sorteada em `U(piso, teto)`, não fixada no teto.** Com valor fixo, o
    nível 1 não CONTÉM o nível 0: a carga leve desapareceria do treino no momento em
    que a DR alargasse. É o sorteio que torna verdadeira a afirmação de que o nível 1
    contém o nível 0.

    ⚠️ **É PESO, e não INÉRCIA.** `dr.body_mass` corrompe a heap (CUDA illegal
    access), portanto a força externa é a única saída correta. A consequência fica
    registrada: a caixa de 5 kg tem inércia de 1 kg. A DR randomiza carga
    **estática**; ela NÃO é evidência de robustez a payload em movimento."""
    if env_ids is None or len(env_ids) == 0:
        return
    n = len(env_ids)
    dev = env.device
    box: Entity = env.scene[box_cfg.name]

    pesos = torch.tensor(T.LEVELS["peso"], device=dev)
    piso = pesos[0]
    largo = getattr(env, "dr_peso", None)
    teto = torch.where(largo[env_ids], pesos[1], piso) if largo is not None \
        else piso.expand(n)
    m = piso + torch.rand(n, device=dev) * (teto - piso)
    env.peso_amostrado[env_ids] = m

    num_bodies = (len(box_cfg.body_ids) if isinstance(box_cfg.body_ids, list)
                  else box.num_bodies)
    fz = -(m - box_mass) * _GRAVIDADE       # extra pra baixo; m < box_mass => pra cima
    forces = torch.zeros(n, num_bodies, 3, device=dev)
    forces[:, :, 2] = fz.unsqueeze(-1)
    box.write_external_wrench_to_sim(
        forces, torch.zeros_like(forces), env_ids=env_ids, body_ids=box_cfg.body_ids)


def jitter_cena(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    yaw_max_rad: float,
    mesa_xy_max: float = 0.0,
    mesa_yaw_max_rad: float = 0.0,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
) -> None:
    """Jitter de spawn da caixa e da prateleira, num evento só (S1 + S12).

    Faz três coisas, e elas partilham o mesmo `forward()`:

      1. **yaw da caixa** (S1) — devolve o ±15° que o `reset_box` fazia;
      2. **xy da prateleira E da caixa** (S12) — o MESMO delta nas duas;
      3. **yaw da prateleira** (S12).

    ⚠️ **O delta xy é compartilhado, e isso não é detalhe.** O `reset_scene_plr`
    posiciona a caixa relativa ao xy NOMINAL, não ao da mesa. Deslocar só a mesa
    deixaria a caixa pendurada no ar fora dela, e o `box_shake` puniria a queda.

    ⚠️ O yaw da mesa NÃO entra no delta compartilhado: girar em torno do centro dela
    não move a caixa, que repousa no centro.

    ⚠️ A prateleira é MOCAP (`get_shelf_spec` -> `auto_wrap_fixed_base_mocap`),
    portanto `write_mocap_pose_to_sim`, nunca `write_root_state_to_sim`.

    --- o que a parte de yaw da caixa resolve (S1) ---

    ⚠️ **Existe por causa de uma perda silenciosa.** A S1 troca o `reset_box` pelo
    `reset_scene_plr`, e os dois não são equivalentes na ORIENTAÇÃO: o
    `reset_scene_plr` escreve o quaternion de `default_root_state`
    (`common/events.py:170`), ou seja identidade. Sem este evento, a caixa passaria a
    nascer sempre alinhada, e o jitter de ±15° sumiria sem erro e sem log.

    Duas coisas dependem dele, e as duas estão documentadas no código:
      - o `reset_segurando` diz que a orientação "vem do que o `reset_box` já sorteou
        (jitter de yaw de ±15°), então o `reorientar` e o `pegar` continuam vendo a
        mesma distribuição de spawn";
      - o `env.py` chama o jitter de yaw de GERAL, e não de filtro de tarefa.

    O ângulo não é número novo: sai de `knobs.Scene.box_jitter_yaw_deg`, o mesmo que o
    `build_base_env` já recebia.

    ⚠️ **Roda DEPOIS do `reset_cena`**, senão o `reset_scene_plr` apaga o quaternion.

    ⚠️ **E precisa de `forward()` antes de ler.** A API só escreve o estado inteiro
    (13 números), então pra trocar o quaternion é preciso reler a posição — e logo
    depois do `reset_scene_plr` ela está STALE. Medido em 05/08 com o evento sem o
    forward: a caixa do `pegar` ia parar em `z = 0.035` com `desvio_xy = 1.50`, ou
    seja o evento reescrevia a pose do episódio ANTERIOR e o `reset_cena` era
    desfeito em silêncio. Com o forward, `folga = 0.0004 m`.

    É o mesmo gotcha, e a mesma solução, do `reset_segurando` logo acima. O custo é um
    `forward()` a mais por reset; o `reset_segurando` já paga um."""
    if env_ids is None or len(env_ids) == 0:
        return
    n = len(env_ids)
    dev = env.device
    box: Entity = env.scene[box_cfg.name]
    mesa: Entity = env.scene[table_cfg.name]

    env.sim.forward()       # sem isto, `root_link_pos_w` é a pose do episódio anterior

    def _quat_yaw(ang: torch.Tensor) -> torch.Tensor:
        q = torch.zeros(len(ang), 4, device=dev)        # mjlab usa wxyz
        q[:, 0] = torch.cos(0.5 * ang)
        q[:, 3] = torch.sin(0.5 * ang)
        return q

    # delta xy COMPARTILHADO: a caixa acompanha a prateleira, senão ela fica no ar
    delta = torch.zeros(n, 3, device=dev)
    if mesa_xy_max > 0.0:
        delta[:, :2] = (torch.rand(n, 2, device=dev) * 2.0 - 1.0) * mesa_xy_max

    # --- caixa: pose + delta, orientação nova ---
    yaw_box = (torch.rand(n, device=dev) * 2.0 - 1.0) * yaw_max_rad
    estado = torch.cat(
        [box.data.root_link_pos_w[env_ids] + delta, _quat_yaw(yaw_box),
         torch.zeros(n, 6, device=dev)],
        dim=-1,
    )
    box.write_root_state_to_sim(estado, env_ids=env_ids)

    # --- prateleira (mocap): mesmo delta xy, yaw próprio ---
    yaw_mesa = (torch.rand(n, device=dev) * 2.0 - 1.0) * mesa_yaw_max_rad
    mesa.write_mocap_pose_to_sim(
        torch.cat([mesa.data.root_link_pos_w[env_ids] + delta, _quat_yaw(yaw_mesa)],
                  dim=-1),
        env_ids=env_ids)


def afasta_cena(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    tarefas_com_prateleira: tuple[int, ...],
    tarefas_com_caixa: tuple[int, ...],
    shelf_half_z: float,
    box_half_z: float,
    distancia: float = 5.0,
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> None:
    """Tira a prateleira do caminho de quem anda, e a caixa junto quando ela não serve.

    **O problema, medido no `play` em 30/07.** A prateleira mora em `x = 0.50` com
    meia-extensão `0.30`, então ela ocupa **x de 0.20 a 0.80** com o topo em 0.55 m —
    altura de joelho. E o destino do `andar` é
    `pos_robô + (d·cos(head), d·sin(head), 0)` com `d` de 0.3 a 2.0 m. Com heading
    perto de zero, **o robô anda direto contra ela**. Ela não faz parte da tarefa
    `andar`; é obstáculo acidental.

    Não é bug do experimento residual: a run monolítica tem o mesmo, e só não
    apareceu porque ela nunca chegou a abrir o `andar`.

    **Por que 5 m e não 30.** O `object_pos_b` entra na obs e o normalizador dele
    APRENDE. Um valor de 30 m envenena a estatística — mesma classe de estrago do bug
    de índice do normalizador (§9b). O maior valor que a obs já vê é o destino do
    `andar` a 2.0 m, então 5 m é 2,5x isso: fora do alcance de um episódio e dentro da
    faixa que o normalizador aguenta.

    **Por que a caixa vai JUNTO e não cai.** Movendo só a prateleira, a caixa
    despenca. E o `box_shake` **não** é gateado no `andar` (a §6b só o desliga no
    `reorientar`), então ele puniria a queda. Com o par junto a caixa continua
    apoiada e o `box_shake` fica em zero.

    ⚠️ **Roda ANTES do `reset_segurando`.** As tarefas de `SPAWN_SEGURANDO` não estão
    em `tarefas_com_caixa`, então a caixa delas é afastada aqui e o `reset_segurando`
    a traz de volta para as palmas. Invertendo a ordem, a caixa sairia das mãos.

    A prateleira é mocap (`get_shelf_spec` -> `auto_wrap_fixed_base_mocap`): corpo
    cinemático, posicionável por-env, e **flutua em qualquer z sem tocar o chão**.
    Daí `write_mocap_pose_to_sim` em vez de `write_root_state_to_sim`.

    ⚠️ **A altura é ALVO ABSOLUTO, não deslocamento relativo.** Duas versões relativas
    falharam, e pelo mesmo motivo de fundo: **`root_link_pos_w` de corpo mocap não
    reflete a escrita sem um `forward()`.** Ver o comentário no corpo da função."""
    if env_ids is None or len(env_ids) == 0:
        return
    mesa: Entity = env.scene[table_cfg.name]
    caixa: Entity = env.scene[box_cfg.name]
    tarefa = env.tarefa_sorteada[env_ids]

    def _fora(quais: tuple[int, ...]) -> torch.Tensor:
        q = torch.tensor(quais, device=env.device)
        return env_ids[~(tarefa.unsqueeze(-1) == q).any(dim=-1)]

    # ⚠️ PARA CIMA, não para o lado. Os envs do mjlab ficam lado a lado em x e y, com
    # espaçamento menor que 5 m — deslocar em x fazia a prateleira de cada env
    # MATERIALIZAR DENTRO DO ROBÔ de um env vizinho e derrubar ele. Medido no smoke em
    # 30/07: os checks de posição passavam (a prateleira estava no lugar pedido) e o
    # `rel_z` da caixa dava +0.145 — que era a PELVE caindo, não a caixa subindo. O
    # acumulador de contribuição contava 960 de 1920 pares, exatamente metade, porque
    # os envs morriam no meio da coleta.
    #
    # Em z não há vizinho: as origens dos envs diferem em x e y, nunca em z. E a
    # prateleira é mocap (corpo cinemático que flutua sem tocar o chão), então ela
    # sustenta a caixa lá em cima igual sustenta aqui embaixo.
    # ⚠️ ALVO ABSOLUTO, não deslocamento relativo. Idempotente por construção, e não lê
    # nada que possa estar velho.
    #
    # As duas tentativas relativas falharam, e pelo MESMO motivo de fundo:
    # **`root_link_pos_w` de corpo mocap não reflete a escrita sem um `forward()`.**
    #
    #   1ª: prateleira na nominal + 5, caixa na atual + 5. O `level_jitter_z` de ±2 cm
    #       fazia as duas somas darem alturas diferentes -> caixa pendurada, e caía.
    #   2ª: as duas na atual + 5, com guarda "quem já subiu não sobe". A guarda LIA a
    #       prateleira, que estava velha, então ela errava: `folga = +9.74 m` com
    #       deslocamento de 5 m (caixa somou duas vezes, prateleira uma) e o check
    #       lendo `z = 0.53` numa prateleira que no sim estava a 5.53.
    #
    # Escrevendo alvo absoluto os dois problemas desaparecem: rodar o evento dez vezes
    # dá o mesmo resultado que rodar uma, e o jitter da nominal deixa de participar.
    origem = env.scene.env_origins
    z_mesa = origem[:, 2] + distancia
    # o pé da caixa encosta no topo da prateleira: topo = centro + shelf_half_z
    z_caixa = z_mesa + shelf_half_z + box_half_z

    ids_mesa = _fora(tarefas_com_prateleira)
    if len(ids_mesa) > 0:
        # XY da prateleira = XY da CAIXA, não o nominal. O `reset_table` sorteia jitter
        # no xy dela, e o `reset_box` põe a caixa relativa a esse xy jitterado — os dois
        # jitters são CORRELACIONADOS. Forçando o nominal eu quebrava a correlação e os
        # desvios somavam: medido `desvio_xy = 0.392` contra 0.197 do `pegar`, ou seja
        # 0.20 da caixa mais 0.19 da prateleira.
        #
        # Pondo a prateleira embaixo da caixa o desvio vira zero, e nestas tarefas o xy
        # exato não importa — só importa que a caixa esteja apoiada.
        pos = torch.stack([caixa.data.root_link_pos_w[ids_mesa, 0],
                           caixa.data.root_link_pos_w[ids_mesa, 1],
                           z_mesa[ids_mesa]], dim=-1)
        quat = torch.zeros(len(ids_mesa), 4, device=env.device)
        quat[:, 0] = 1.0        # a prateleira nasce sem rotação (`env.py` passa só pos)
        mesa.write_mocap_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids_mesa)

    ids_caixa = _fora(tarefas_com_caixa)
    if len(ids_caixa) > 0:
        # o XY da caixa é lido, e isso é seguro: nada aqui escreve XY dela, então a
        # leitura é idempotente e o jitter de spawn (±0.20 em x, ±0.18 em y, dentro da
        # pegada de 0.30) sobrevive.
        pos = torch.stack([caixa.data.root_link_pos_w[ids_caixa, 0],
                           caixa.data.root_link_pos_w[ids_caixa, 1],
                           z_caixa[ids_caixa]], dim=-1)
        estado = torch.cat([pos, caixa.data.root_link_quat_w[ids_caixa],
                            torch.zeros(len(ids_caixa), 6, device=env.device)], dim=-1)
        caixa.write_root_state_to_sim(estado, env_ids=ids_caixa)
