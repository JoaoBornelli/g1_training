"""PESO variável da caixa (payload) — 2026-07-17, nas 2 alturas já sabidas.

Robustez a diferentes PESOS de caixa (0.5–5kg) nas alturas [0.55, 0.45] que o robô
já domina. Cada episódio sorteia um peso efetivo e aplica uma força extra −z na caixa.

MECANISMO TESTADO: `write_external_wrench_to_sim` (o mesmo primitivo do
`apply_external_force_torque` que o mjlab ship/testa como payload), NÃO o
`dr.body_mass`/`dr.pseudo_inertia` — que são código órfão/não-testado do mjlab e
CORROMPEM a heap (CUDA illegal memory access). Ver [[g1-lift-box-task]] e
[[feedback-use-tested-manufacturer-reference]].

Warm-start do `model_7300` (PLR alturas 0.55+0.45). O peso NÃO está na obs; o robô
infere pela pega/torque das juntas (`joint_torque` ESTÁ na obs) — como fará no robô
real. Simula só o PESO (força a sustentar), não a inércia — o que domina no lift/hold.

Herda do plr_height1: alturas [0.55,0.45], table_contact −1.5, arm_vel −0.002.
"""
from dataclasses import replace

from .c2026_07_17_plr_height1 import KNOBS as _PREV

KNOBS = replace(
    _PREV,
    scene=replace(_PREV.scene, box_weight_range=(0.5, 5.0)),
)
