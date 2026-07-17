# ACTIVE = config que a task REGISTRADA usa (treino sem override E play.py).
# Aponta pro que está sendo trabalhado AGORA.
# Cadeia: gate_and_com -> box_edge -> generalize (hardening) -> plr_height1 (PLR alturas
# [0.55,0.45] + table_contact −1.5 + arm_vel −0.002) -> plr_h1_payload (peso 0.5–5kg via
# força payload testada) -> plr_h2_payload (degrau 2: alturas [0.55,0.45,0.35,0.25] + peso
# + joint_acc: movimento controlado/estável). `rehearsal` cross-skill DROPADO. (mass DR via
# dr.body_mass foi tentado e REVERTIDO: API não-testada corrompe a heap.)
from .c2026_07_17_plr_h2_payload import KNOBS as ACTIVE
