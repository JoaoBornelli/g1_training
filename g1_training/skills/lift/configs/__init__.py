# ACTIVE = config que a task REGISTRADA usa (treino sem override E play.py).
# Aponta pro que está sendo trabalhado AGORA.
# Cadeia: gate_and_com -> box_edge -> generalize (hardening) -> plr_height1 (currículo
# de altura via PLR: alturas [0.55, 0.45] + table_contact −1.5 + arm_vel −0.002) ->
# plr_h1_payload (eixo 2: PESO variável da caixa 0.5–5kg via força payload testada,
# nas 2 alturas). `rehearsal` (cross-skill ficar-de-pé) DROPADO — redundante c/ skills
# congeladas. (mass DR via dr.body_mass foi tentado e REVERTIDO: API não-testada corrompe.)
from .c2026_07_17_plr_h1_payload import KNOBS as ACTIVE
