# ACTIVE = config que a task REGISTRADA usa (treino sem override E play.py).
# Aponta pro que está sendo trabalhado AGORA.
# Cadeia: gate_and_com -> box_edge -> generalize (hardening) -> plr_height1 (PLR alturas
# [0.55,0.45] + table_contact −1.5 + arm_vel −0.002) -> plr_h1_payload (peso 0.5–5kg via
# força payload testada) -> plr_h2_payload (degrau 2: alturas [0.55,0.45,0.35,0.25] + peso
# + joint_acc) -> smooth_impact (qualidade de movimento: #1 action_scale 0.8 estrutural +
# #3 anti-impacto na mesa via soft_landing testado; MESMAS 4 alturas, warm-startable).
# `rehearsal` cross-skill DROPADO. (mass DR via dr.body_mass foi REVERTIDO: corrompe heap.)
from .c2026_07_18_smooth_impact import KNOBS as ACTIVE
