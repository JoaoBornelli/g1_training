"""Observações novas do multi-tarefa — 20 números em cima do contrato herdado.

O que o `g1_training/common/observations.py` já entrega e é reusado sem mudança:
`target_pos_b`, `object_pos_b`, `joint_torque`. Em especial o `target_pos_b`, que
lê `command[:, 0:3]` — e é por isso que `alvo_pos` fica em `[0:3]` no layout novo.

O que entra aqui:

    object_rot_b     6   🚧 a ÚNICA dependência dura do desenho — sem ela o
                         `reorientar` é impossível
    face_alvo        3   fatia do comando
    dir_alvo         3   fatia do comando
    task_onehot      8   fatia do comando

Tudo em frame da BASE, nunca em mundo: o mundo inclui o `env_origin`, que varia
por ambiente, então obs em mundo não generaliza.

⚠️ **Contrato sim-to-real.** O ator só vê o que o robô real mede mais a pose que a
percepção entrega. Nada de força de contato da palma, nada de velocidade da caixa,
nada de CoM — isso é privilégio do crítico. `object_rot_b` passa no critério porque
uma câmera entrega orientação de objeto; velocidade angular da caixa não passaria.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def object_rot_b(env: "ManagerBasedRlEnv", object_name: str) -> torch.Tensor:
    """Orientação do objeto em frame da base, representação 6D. [B, 6]

    São as DUAS PRIMEIRAS COLUNAS da matriz de rotação (Zhou et al.), não um
    quaternion. Motivo: o quaternion tem dupla cobertura (`q` e `−q` são a mesma
    rotação) e ângulos de Euler têm descontinuidade — as duas coisas dão salto na
    entrada da rede sem que nada tenha acontecido no mundo. A 6D é contínua em
    todo SO(3), e a terceira coluna é recuperável pelo produto vetorial, então
    não se perde informação.

    Por que a orientação COMPLETA e não só o erro de ângulo: o erro é escalar e
    basta pra pontuar, mas pra saber COMO empurrar a política precisa ver onde as
    outras faces estão. O comentário do `common/observations.py` já sinalizava
    isso ("orientação da caixa fica pra Fase 2") — chegou a Fase 2."""
    robot: Entity = env.scene["robot"]
    obj: Entity = env.scene[object_name]
    eixos = torch.zeros(2, 3, device=obj.data.root_link_quat_w.device)
    eixos[0, 0] = 1.0                                   # e_x do corpo da caixa
    eixos[1, 1] = 1.0                                   # e_y do corpo da caixa
    colunas = []
    for i in range(2):
        e = eixos[i].expand(obj.data.root_link_quat_w.shape[0], 3)
        col_w = quat_apply(obj.data.root_link_quat_w, e)
        colunas.append(quat_apply_inverse(robot.data.root_link_quat_w, col_w))
    return torch.cat(colunas, dim=-1)


def twist_cmd(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """O twist comandado — `[vx, vy, ωz]` em frame da base. [B, 3]  (S10)

    **Passa no contrato sim-to-real.** Ele não é medição: é CALCULADO a partir do
    alvo, por código determinístico que roda igual na simulação e na pilha de
    navegação do robô. O que o robô real recebe da navegação é exatamente isto.

    **Por que vale a pena entrar na obs.** Sem ele, a política tem de aprender a
    função de perfil por dentro — a rampa, a trava por rumo, o morto. Com ele,
    ajustar o perfil deixa de ser re-aprendizado e vira Categoria A.

    ⚠️ **Tem de estar em `CANAIS_CONGELADOS`.** Na Fase 0 só o `parado` roda e o
    twist fica constante em zero. O `EmpiricalNormalization` calcula `_std = 0` no
    primeiro update, e quando o `andar` abre o valor entra ~100× amplificado. É o
    mesmo caso do `target_pos_b` que o `runner.py` já trata.

    ⚠️ `vy` é sempre zero desde a S5 — o perfil gira para apontar e anda para frente.
    O canal fica assim mesmo: tirá-lo mudaria a largura da obs de novo se a marcha
    lateral voltar, e mudança de largura custa o zero."""
    return env.command_manager.get_term("twist").command[:, :3]


def target_pos_b_gateado(
    env: "ManagerBasedRlEnv", command_name: str, tarefas_zeradas: tuple[int, ...]
) -> torch.Tensor:
    """`target_pos_b` zerado nas tarefas que não têm alvo de posição. [B, 3]  (S10)

    Decisão de arquitetura da S5/S10: o ator **não recebe alvo de posição** nas
    tarefas de locomoção. Ele recebe velocidade e velocidade angular para rastrear,
    que é o que a pilha de navegação entrega no robô real. `botar` e `reorientar`
    continuam vendo o alvo, porque ali ele é a pose de um objeto, não um waypoint.

    ⚠️ **Só o ATOR usa esta função.** O crítico continua com o `target_pos_b` cheio,
    inclusive no `andar`: o retorno depende de `chegou`, e sem o alvo o crítico não
    distingue "a 5 cm" de "a 2 m". É privilégio legítimo — o crítico é descartado e
    não vai para o robô.

    O campo continua com 3 números em vez de sumir. Largura de observação é contrato:
    encolher aqui e crescer depois custa o zero duas vezes."""
    from g1_training.common import observations as base_obs

    cheio = base_obs.target_pos_b(env, command_name=command_name)
    quais = torch.tensor(tarefas_zeradas, device=cheio.device)
    zera = (env.active_task.unsqueeze(-1) == quais).any(dim=-1)
    return torch.where(zera.unsqueeze(-1), torch.zeros_like(cheio), cheio)


def command_slice(
    env: "ManagerBasedRlEnv", command_name: str, lo: int, hi: int
) -> torch.Tensor:
    """Uma fatia do vetor de comando. [B, hi-lo]

    Uma função cobre `face_alvo`, `dir_alvo` e `task_onehot` — três termos de obs
    que só diferem nos índices. Escrever três funções idênticas seria triplicar o
    mesmo `return`."""
    return env.command_manager.get_term(command_name).command[:, lo:hi]
