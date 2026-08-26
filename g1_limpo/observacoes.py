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

__all__ = ["um_de_cinco", "N_SLOTS"]

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
