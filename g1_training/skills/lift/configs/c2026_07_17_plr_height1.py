"""Currículo de altura via PLR — degrau 1 (2026-07-17): alturas [0.55, 0.45].

Warm-start do checkpoint `lift_6550` (altura normal, prateleira 0.55 → caixa 0.65).
PRIMEIRO degrau rumo ao chão com Prioritized Level Replay:
- mantém a altura JÁ DOMINADA (0.55) no sorteio pra não esquecer, e
- introduz a NOVA (0.45, −10cm).

O PLR (rank-based) garante um piso mínimo às duas e concentra o esforço sozinho na
que o robô for pior. `target_z` fica ABSOLUTO (0.78–0.85, decisão do user): prateleira
mais baixa = lift MAIOR = mais difícil → o PLR naturalmente foca na 0.45. Base =
`generalize` (herda jitter de posição da caixa + push sob carga + toda a cadeia de
reward). Ver common/curriculums.py e [[g1-lift-box-task]].

PRÓXIMO DEGRAU = só fazer append da altura nova na lista, warm-start deste checkpoint:
    plr=Plr(shelf_levels=(0.55, 0.45, 0.35))
(o rehearsal das anteriores sai de graça — elas continuam no sorteio).
"""
from dataclasses import replace

from ..knobs import Plr
from .c2026_07_16_generalize import KNOBS as _PREV

KNOBS = replace(
    _PREV,
    scene=replace(_PREV.scene, shelf_top=0.55),   # init saudável (= altura mais alta da lista)
    plr=Plr(shelf_levels=(0.55, 0.45)),
)
