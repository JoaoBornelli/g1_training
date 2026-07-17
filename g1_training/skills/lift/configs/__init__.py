# ACTIVE = config que a task REGISTRADA usa (treino sem override E play.py).
# Aponta pro que está sendo trabalhado AGORA.
# Cadeia: gate_and_com -> box_edge -> generalize (hardening) -> plr_height1 (currículo
# de altura via Prioritized Level Replay: alturas [0.55, 0.45], rehearsal MULTI-ALTURA
# por-env com foco adaptativo na difícil). `rehearsal` (cross-skill ficar-de-pé) foi
# DROPADO — redundante com a arquitetura de skills congeladas (stand é política própria).
from .c2026_07_17_plr_height1 import KNOBS as ACTIVE
