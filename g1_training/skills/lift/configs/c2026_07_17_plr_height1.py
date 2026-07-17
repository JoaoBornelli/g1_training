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

TUNING APLICADO 2026-07-17 (experimento validado no play — o robô executa bem e
generalizou até 0.25 sem retreinar):
- table_contact −0.8 → −1.5: encarece escorar a coxa na prateleira (o hack de alcançar
  embaixo sem agachar). com_balance não pega (ele estica a coxa e mantém o CoM atrás).
- arm_vel −0.002: freio de velocidade das juntas do braço (joint_vel_l2) — o robô ia
  rápido demais pegar/levar e sacudia a caixa. (posture 0.8 foi TESTADO e DESCARTADO:
  briga com o squat — posture-raw caiu apesar do peso maior.)

TODO próximos passos (discutidos, ainda não aqui): (a) migrar o freio de velocidade
p/ ACELERAÇÃO/TORQUE (mais elegante, ataca a causa do jerk/pulo); (b) DR de MASSA da
caixa (0.5–5kg) nestas 2 alturas antes de descer o currículo; (c) o "pulo ao levantar"
é a PERNA explodindo — vai precisar de freio no corpo/anti-pulo, não só no braço.
"""
from dataclasses import replace

from ..knobs import Plr
from .c2026_07_16_generalize import KNOBS as _PREV

KNOBS = replace(
    _PREV,
    scene=replace(_PREV.scene, shelf_top=0.55),   # init saudável (= altura mais alta da lista)
    reward=replace(_PREV.reward, table_contact=-1.5, arm_vel=-0.002),
    plr=Plr(shelf_levels=(0.55, 0.45)),
)
