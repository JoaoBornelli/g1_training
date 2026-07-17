"""DR de MASSA da caixa (eixo 2 do currículo) — 2026-07-17.

Antes de descer o currículo de altura, testa robustez ao PESO: caixa de **0.5–5kg**
nas alturas **[0.55, 0.45]** que o robô já domina (mesma lista de PLR do plr_height1).
Cada env sorteia uma massa no startup via `dr.pseudo_inertia` (massa E inércia escalam
juntas — caixa de 5kg tem inércia de 5kg). Distribuição log-uniforme no range.

Warm-start do checkpoint do `plr_height1` tunado. A massa NÃO está na obs (só
`box_pos_b`) → warm-start PURO; o robô aprende a sustentar pesos variados pela
propriocepção/torque das juntas (o mesmo sinal que terá no robô real). O peso desloca
o CoM → é equilíbrio sob carga variável, exatamente o desafio da task.

NOTA: este é o 1º evento de DR de STARTUP re-adicionado (os antigos foot_friction/
base_com estavam OFF). Ele expande body_mass/inertia por-mundo, mas NÃO expande
soft_joint_pos_limits → o fix do reset_joints_by_offset continua necessário e válido.

Herda do plr_height1: alturas [0.55,0.45], table_contact −1.5, arm_vel −0.002.
Ver common/curriculums.py, common/box.py e [[g1-lift-box-task]].
"""
from dataclasses import replace

from .c2026_07_17_plr_height1 import KNOBS as _PREV

KNOBS = replace(
    _PREV,
    scene=replace(_PREV.scene, box_mass_range=(0.5, 5.0)),
)
