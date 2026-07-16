"""Currículo de altura, degrau 1 (2026-07-16): mesa+caixa 10cm mais BAIXAS.

Warm-start do checkpoint do c2026_07_16_box_edge (que chegou a lift 0.60 /
sustain 0.48 / erro 6cm na it 4000). Primeiro passo rumo ao chão — a tese do
projeto: agachar EMERGE de ter que alcançar a caixa mais baixo, não é prescrito.

O que muda (só a VERTICAL; xy/pega/rewards intactos, warm-start-safe):
- table_half[2] 0.275 -> 0.225  => topo da mesa 0.55 -> 0.45, caixa repousa 0.65 -> 0.55
- target_z 0.78-0.85 -> 0.68-0.75  (baixa o MESMO 10cm)

target baixa junto com a caixa de propósito: a distância de erguer fica a MESMA
(~13-20cm acima do repouso), então o robô só precisa dobrar mais pra pegar —
isola a variável do currículo (altura de pega) da dificuldade de erguer.

NÃO faz rehearsal (altura fixa, não sorteada) — decisão pendente com o user;
se esquecer a altura anterior, o passo é randomizar a altura da mesa por-env
(exige amostragem coordenada caixa<->mesa no reset). Ver [[g1-lift-box-task]].
"""
from dataclasses import replace

from ..knobs import LiftKnobs
from .c2026_07_16_box_edge import KNOBS as _PREV

KNOBS = replace(
    _PREV,
    scene=replace(_PREV.scene, table_half=(0.30, 0.30, 0.225)),
    command=replace(_PREV.command, target_z=(0.68, 0.75)),
)
