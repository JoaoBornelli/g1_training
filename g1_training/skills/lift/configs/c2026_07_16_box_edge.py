"""Fine-tune 2026-07-16: caixa movida da CENTRO da mesa pra BORDA.

Causa raiz investigada (user reportou 07-16: robô ainda se apoiando na caixa
e agora abrindo um "espacato" pra alcançar, mesmo com gate 10°+CoM+slip):
com a caixa no centro (x=0.50) e a mesa indo de x=0.20 a x=0.80, alcançar o
centro exige esticar por CIMA de 30cm de tampo — a pose correta (em pé, sem
escorar) fisicamente NÃO alcança lá. Mover a caixa pra BORDA da mesa encurta
a distância de alcance em ~0.20m sem mudar mais nada (mesa, altura, massa
intactas) — ataca a geometria, não empilha mais penalidade em cima do sintoma.

box_xy: novo centro = borda_da_mesa + meia-largura_da_caixa
      = (table_xy[0] - table_half[0]) + box_half[0] = (0.50-0.30)+0.10 = 0.30

box_xy_jitter (user 07-16): a caixa não deve nascer sempre no MESMO pixel da
mesa — 10cm de variação uniforme em x/y a cada reset (via `reset_box`,
pose_range soma à posição de repouso) evita que a política decore uma pose
fixa de aproximação; generaliza a preensão pra uma vizinhança da borda.
"""
from dataclasses import replace

from ..knobs import LiftKnobs
from .c2026_07_15_gate_and_com import KNOBS as _PREV

KNOBS = replace(_PREV, scene=replace(_PREV.scene, box_xy=(0.30, 0.0), box_xy_jitter=0.10))
