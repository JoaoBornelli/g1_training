"""Eventos de reset próprios do multi-tarefa.

Um só, por enquanto: a condição de spawn "segurando" das 3 tarefas que começam com a
caixa nas mãos (`parado c/ caixa`, `andar c/ caixa`, `botar`).

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
