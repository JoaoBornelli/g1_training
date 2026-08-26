"""Os relógios do currículo. FASE F0/F1: só o dono do NÍVEL.

Três relógios, e eles não se falam:

    A. `command_vel`  agendado por passo global. É o do fabricante, e fica como está.
    B. `forma`        a fatia locomoção × cadeia.        -> F5
    C. `nivel`        a dificuldade do objetivo.          -> aqui (F6 põe o passeio)

⚠ ORDEM NO DICT É CONTRATO, e ela tem teste. No reset o mjlab roda
CURRÍCULO -> EVENTOS -> COMANDO. Portanto:

    o `nivel` escreve `env.limpo_nivel`  (currículo, primeiro)
    os eventos posicionam a cena LENDO esse nível
    o comando sorteia o alvo LENDO esse mesmo nível

Invertida, a coisa quebra em silêncio: num bug medido em 20/08 a probabilidade de
subir caía de `p` para `0,7·p`, o ponto fixo do nível saía de 0,5 para 0,714, e um
episódio de LOCOMOÇÃO rebaixava o nível em 70% das vezes.

ESCOPO DESTA FASE: o nível é FIXO (o `forcado` do knob, ou 0). O passeio aleatório
±1 por sucesso entra na F6 — e quando entrar, ele muda SÓ este arquivo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["nivel", "garante_nivel"]


def garante_nivel(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Cria `env.limpo_nivel` se ele ainda não existe, e devolve o buffer.

    ⚠ Ele precisa existir antes do PRIMEIRO reset, porque os eventos o leem. O
    currículo roda inteiro antes dos eventos, portanto chamar isto no termo de
    currículo é suficiente — mas os consumidores usam esta mesma função, e não um
    `getattr` solto, para que a criação fique num lugar só.
    """
    if not hasattr(env, "limpo_nivel"):
        env.limpo_nivel = torch.zeros(env.num_envs, dtype=torch.long,
                                      device=env.device)
    return env.limpo_nivel


def nivel(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    n_niveis: int,
    forcado: int | None = None,
) -> float:
    """Escreve o nível dos envs que resetaram. Devolve a média, para o log.

    F0/F1: fixo. F6 troca o corpo desta função pelo passeio ±1 por sucesso, e a
    assinatura não muda.
    """
    buf = garante_nivel(env)
    if forcado is not None:
        buf[env_ids] = int(max(0, min(n_niveis - 1, forcado)))
    # sem `forcado`, o nível permanece onde está. Na F6 é aqui que o passeio entra:
    #     sobe = sucesso ; desce = ~sucesso ; buf = clamp(buf + sobe - desce, 0, N-1)
    return float(buf.float().mean())
