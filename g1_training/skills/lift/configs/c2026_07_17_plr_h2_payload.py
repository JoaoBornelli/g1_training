"""Currículo de altura — degrau 2 (2026-07-17): alturas [0.55, 0.45, 0.35, 0.25] + payload.

Adiciona DUAS novas alturas (0.35 e 0.25) ao PLR de uma vez (as 2 dominadas continuam no
sorteio, anti-esquecimento) mantendo o PESO variável (payload 0.5–5kg). Cada episódio
sorteia altura (PLR) E peso. É um salto de 2 degraus — se travar (score/fell_over das
baixas não cede), reduzir pra 1 altura nova por vez.

MECANISMO TESTADO: `write_external_wrench_to_sim` (o mesmo primitivo do
`apply_external_force_torque` que o mjlab ship/testa como payload), NÃO o
`dr.body_mass`/`dr.pseudo_inertia` — que são código órfão/não-testado do mjlab e
CORROMPEM a heap (CUDA illegal memory access). Ver [[g1-lift-box-task]] e
[[feedback-use-tested-manufacturer-reference]].

Warm-start do checkpoint do payload nas 2 alturas (h1_payload). O peso NÃO está na obs;
o robô infere pela pega/torque das juntas (`joint_torque` ESTÁ na obs). Simula só o PESO
(força a sustentar), não a inércia — o que domina no lift/hold.

MOVIMENTO CONTROLADO/ESTÁVEL: liga a limitação de ACELERAÇÃO (`joint_acc`, corpo todo —
pune movimento explosivo, o "pulo" da perna) junto com a de VELOCIDADE (`arm_vel`,
herdado). `joint_acc=-2.5e-7` é o start (convenção IsaacLab; joint_acc_l2 não é shipado
pelo mjlab mas é reward-math validado, não corrompe — ver [[feedback-use-tested-manufacturer-reference]]).
WATCH: se `grasp`/`lift` caírem = over-damping, afrouxa; se `Episode_Reward/joint_acc` ~0
= peso fraco, sobe pra −1e-6.

Herda do plr_height1: table_contact −1.5, arm_vel −0.002. Ver [[g1-lift-box-task]].
"""
from dataclasses import replace

from ..knobs import Plr
from .c2026_07_17_plr_height1 import KNOBS as _PREV

KNOBS = replace(
    _PREV,
    scene=replace(_PREV.scene,
        box_weight_range=(0.5, 5.0),
        shelf_top=0.55,
    ),
    reward=replace(_PREV.reward, joint_acc=-2.5e-7),   # acel (corpo todo) + arm_vel herdado
    plr=Plr(shelf_levels=(0.55, 0.45, 0.35, 0.25)),
)
