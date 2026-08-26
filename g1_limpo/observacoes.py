"""O que a política vê a mais que a receita do fabricante.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Só `mjlab`, que é framework.

CONTRATO DO LAYOUT: canal novo entra sempre POR ÚLTIMO, e nos DOIS grupos, na MESMA
ordem. Assim migrar um checkpoint é um APPEND de colunas, e nunca uma inserção no
meio — uma inserção no meio desloca todo peso da primeira camada em silêncio, e a
política sai andando de lado sem uma linha de erro.

ESCOPO DESTA FASE (F2): o one-hot dos 5 elos. Os canais da caixa (posição do alvo,
face pedida, erro angular) entram na F3, depois deste, pelo mesmo contrato.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["um_de_cinco", "caixa_no_frame_da_base", "N_SLOTS", "N_CAIXA"]

N_CAIXA = 8

N_SLOTS = 5


def um_de_cinco(env: "ManagerBasedRlEnv", command_name: str,
                canal_do_elo: int) -> torch.Tensor:
    """O one-hot do elo corrente: 5 slots, um por estado.

        0 ANDAR   1 REORIENTAR   2 PEGAR   3 CARREGAR   4 BOTAR

    ⚠ ELE É LIDO DO COMANDO, POR PASSO, e não de um buffer de reset. Isso é o que
    permite o elo TROCAR DENTRO do episódio na F4 sem reset e sem resample — e era a
    única incompatibilidade real entre a máquina de elo do `g1_poc` e o one-hot do
    `g1_multitask`. Ela é de uma linha, e é esta.

    ⚠ SEM `noise` E SEM `scale`, e isso é decisão. Ruído num one-hot produziria
    frações entre slots, isto é, estados que não existem; e `scale` num canal que já
    está em [0,1] só desalinharia a escala contra o normalizador.

    ⚠ O one-hot NÃO leva o crédito do andar. O `g1_poc` já tinha o equivalente
    funcional — o bit `caixa_valida` mais o twist forçado a zero — e não andou. A
    razão de engenharia dele é outra, e basta: ele diz QUAL objetivo está ativo, e
    gateia os sete termos de tarefa da F3, que sem gate pagariam o máximo com os
    canais de caixa zerados, porque `exp(0) = 1`.
    """
    comando = env.command_manager.get_command(command_name)
    assert comando is not None, f"comando '{command_name}' não existe"
    elo = comando[:, canal_do_elo].long().clamp(0, N_SLOTS - 1)
    return torch.nn.functional.one_hot(elo, num_classes=N_SLOTS).float()


def caixa_no_frame_da_base(env, command_name: str) -> torch.Tensor:
    """Os canais da caixa, TODOS no frame da base. 8 canais.

        [0:3]  caixa − base, no frame da base
        [3:6]  alvo  − base, no frame da base
        [6]    erro angular da face pedida, em radianos
        [7]    valida

    ⚠ TUDO NO FRAME DA BASE, e não em mundo. Coordenada de mundo carrega a ORIGEM DO
    ENV, que é diferente em cada um dos 4096 — a política teria de aprender 4096
    deslocamentos. E carrega o rumo: o mesmo problema geométrico visto de dois yaws
    daria dois vetores diferentes. No frame da base o problema é o mesmo em todo env.

    ⚠ O σ NÃO ENTRA AQUI, e é decisão. Ele diz "este env é fácil ou difícil", e a
    política condicionaria a ação à forma da RECOMPENSA em vez de à tarefa. σ é
    moldagem; a observação é estado do mundo.

    ⚠ O erro angular entra em RADIANOS e sem normalizar. Ele já vive em [0, π].
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse

    from g1_limpo.comando import ALVO, ANG, VALIDA

    cmd = env.command_manager.get_command(command_name)
    robo = env.scene["robot"]
    p, q = robo.data.root_link_pos_w, robo.data.root_link_quat_w
    caixa_b = quat_apply_inverse(q, env.scene["box"].data.root_link_pos_w - p)
    alvo_b = quat_apply_inverse(q, cmd[:, ALVO] - p)
    return torch.cat([caixa_b, alvo_b,
                      cmd[:, ANG].unsqueeze(-1),
                      cmd[:, VALIDA].unsqueeze(-1)], dim=-1)
