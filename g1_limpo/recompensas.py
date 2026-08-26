"""Os termos de recompensa PRÓPRIOS do g1_limpo.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Só `mjlab`, que é framework.

Na F1 este arquivo é pequeno de propósito: a locomoção do fabricante é a fundação, e
tudo que ela já entrega fica como está. Aqui vive só o que o molde NÃO tem, e cada
item traz o defeito medido que o justifica.

Os sete incentivos de manipulação entram na F3.
"""
from __future__ import annotations

import torch

from mjlab.tasks.velocity.mdp import feet_swing_height

__all__ = ["AlturaDeBalanco"]


class AlturaDeBalanco(feet_swing_height):
    """O `feet_swing_height` do fabricante, com o `reset` que falta.

    ⚠ ISTO É UM BUG DO MOLDE, e ele é silencioso. O termo do fabricante acumula
    `peak_heights` por pé e só zera no PRIMEIRO CONTATO. Mas
    `reward_manager.py:174` só registra um termo de classe em `_class_term_cfgs` —
    a lista dos que recebem `reset(env_ids)` — quando a classe TEM um método
    `reset`. O `feet_swing_height` não tem.

    Consequência: quando o episódio termina com um pé no ar (isto é, toda vez que o
    robô CAI), o pico daquele pé sobrevive ao reset e entra no episódio seguinte. O
    `Metrics/peak_height_mean` então INFLA com queda, e o painel mostra "o passo está
    subindo" exatamente quando o robô está caindo mais.

    Foi assim que um bloco leu `peak_height` em alta durante 5000 iterações com o robô
    imóvel: a altura vinha do vôo da queda, e não de passo nenhum.

    O conserto tem três linhas e nenhum número.
    """

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.peak_heights[env_ids] = 0.0
