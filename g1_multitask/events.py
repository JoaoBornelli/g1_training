"""Eventos de reset próprios do multi-tarefa.

Dois: a condição de spawn "segurando" das 3 tarefas que começam com a caixa nas mãos
(`parado c/ caixa`, `andar c/ caixa`, `botar`), e afastar a prateleira das tarefas que
não a usam — ela fica NO CAMINHO de quem anda.

Por que ela existe: medido em 30/07, sem ela essas 3 tarefas não têm caminho de
aquisição nenhum. Todos os termos de tarefa dão exatamente 0.0 no reset — a caixa
nasce em cima da prateleira, e `reaching`/`grasp`/`lift` são gateados só no `pegar`
pela §6b. O robô não tem gradiente pra pegar a caixa, e o único termo que pontuaria
exige uma preensão que ele nunca vai estabelecer. O doc pede a condição na §3
("'andar com caixa' não é tarefa nova; é a tarefa 'andar' com condição de spawn
'segurando'") e na §4 (linha do gatilho), mas ela não existia no código.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1_training.common.robot import PALM_SITES

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_PALMAS = SceneEntityCfg("robot", site_names=list(PALM_SITES))


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


def afasta_cena(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    tarefas_com_prateleira: tuple[int, ...],
    tarefas_com_caixa: tuple[int, ...],
    pos_prateleira: tuple[float, float, float],
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

    ⚠️ `pos_prateleira` vem por parâmetro porque **`mocap_pose` não é legível** — o
    `EntityData` só expõe a ESCRITA (`write_mocap_pose_to_sim`). Então não dá para ler
    a pose atual e somar um deslocamento; a posição de longe é montada do zero, a
    partir da nominal (relativa à origem do env) que o `env.py` passa."""
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
    desloca = torch.tensor([0.0, 0.0, distancia], device=env.device)
    nominal = torch.tensor(pos_prateleira, device=env.device)

    ids_mesa = _fora(tarefas_com_prateleira)
    if len(ids_mesa) > 0:
        # a prateleira nasce sem rotação (`env.py` passa só `pos=`), então quat = w=1
        alvo = env.scene.env_origins[ids_mesa] + nominal + desloca
        quat = torch.zeros(len(ids_mesa), 4, device=env.device)
        quat[:, 0] = 1.0
        mesa.write_mocap_pose_to_sim(torch.cat([alvo, quat], dim=-1),
                                     env_ids=ids_mesa)

    ids_caixa = _fora(tarefas_com_caixa)
    if len(ids_caixa) > 0:
        estado = torch.cat([caixa.data.root_link_pos_w[ids_caixa] + desloca,
                            caixa.data.root_link_quat_w[ids_caixa],
                            torch.zeros(len(ids_caixa), 6, device=env.device)], dim=-1)
        caixa.write_root_state_to_sim(estado, env_ids=ids_caixa)
