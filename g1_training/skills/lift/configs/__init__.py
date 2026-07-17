# ACTIVE = config que a task REGISTRADA usa (treino sem override E play.py).
# Aponta pro que está sendo trabalhado AGORA.
# Cadeia: gate_and_com -> box_edge -> generalize (hardening) -> plr_height1 (currículo
# de altura via PLR: alturas [0.55, 0.45] + table_contact −1.5 + arm_vel −0.002) ->
# plr_h1_mass_dr (eixo 2: DR de massa da caixa 0.5–5kg nas 2 alturas conhecidas).
# `rehearsal` (cross-skill ficar-de-pé) foi DROPADO — redundante com skills congeladas.
from .c2026_07_17_plr_h1_mass_dr import KNOBS as ACTIVE
