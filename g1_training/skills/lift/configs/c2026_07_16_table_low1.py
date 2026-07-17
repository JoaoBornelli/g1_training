"""Currículo de altura, degrau 1: prateleira (e caixa) 10cm mais BAIXAS.

Primeiro passo rumo ao chão — a tese do projeto: agachar EMERGE de ter que
alcançar a caixa mais baixo, não é prescrito.

Com a PRATELEIRA MOCAP (2026-07-16), baixar a altura é só `shelf_top` — por-env,
runtime, sem recompilar (era o limite da mesa free-body):
- shelf_top 0.55 -> 0.45  => caixa repousa 0.65 -> 0.55 (box_z = shelf_top + box_half)
- target_z fica ABSOLUTO em 0.78-0.85 (decisão do user) => o lift cresce (~23-30cm):
  pega mais embaixo, ergue até o mesmo ponto comandado.
- com_balance -2.0 (mais liberdade de inclinar pra alcançar).

NOTA: config de referência do degrau 1. Quando for rodar o currículo de verdade,
provavelmente basear no `rehearsal` (herda push+jitter+rehearsal) e sortear
`shelf_top` num range por-env (multi-altura sai de graça com a prateleira mocap).
"""
from dataclasses import replace

from ..knobs import LiftKnobs
from .c2026_07_16_box_edge import KNOBS as _PREV

KNOBS = replace(
    _PREV,
    scene=replace(_PREV.scene, shelf_top=0.45),
    command=replace(_PREV.command, target_z=(0.78, 0.85)),
    reward=replace(_PREV.reward, com_balance=-2.0),
)
