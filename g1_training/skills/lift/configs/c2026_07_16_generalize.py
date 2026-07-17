"""Fine-tune 2026-07-16: GENERALIZAR posição da caixa + empurrão sob carga.

Warm-start do checkpoint bom do box_edge (4700-derivado, que já pega/ergue reto
na altura original). Agora que ele PEGA, endurecer antes de baixar a mesa:

1) POSIÇÃO da caixa mais variada (era ±0.10 simétrico do box_edge):
   - x = OFFSET (0.0, +0.20) -> centro da caixa 0.30..0.50. ASSIMÉTRICO de propósito:
     box_xy=0.30 é a borda da frente da mesa; -x jogaria a caixa pra fora e ela cairia.
     +0.20 é o limite prático de alcance (0.50 era o "longe demais" original).
   - y = OFFSET (-0.18, +0.18) -> centro -0.18..0.18 (quase a largura toda da mesa, no
     alcance bimanual). Simétrico (mesa é simétrica em y).

2) EMPURRÃO sob carga (não-gated, igual ao stand_step): push_robot ±0.8 + push_force
   (apply_body_impulse ±50N na pelvis). Não-gated mas o hold domina o episódio (pega em
   ~1-2s, segura ~18s) -> a maioria dos empurrões cai COM a caixa na mão -> treina
   equilíbrio sob o CoM deslocado pelo peso ("melhorar o stand-step" carregando a caixa).

Muda 3 coisas de robustez de uma vez (posição + push forte + push_force) — hardening
combinado, de propósito; atribuição fina não importa aqui (todas puxam "generalizar").
"""
from dataclasses import replace

from ..knobs import LiftKnobs
from .c2026_07_16_box_edge import KNOBS as _PREV

KNOBS = replace(_PREV,
    scene=replace(_PREV.scene, box_jitter_x=(0.0, 0.20), box_jitter_y=(-0.18, 0.18)),
    push=replace(_PREV.push, x=(-0.6, 0.6), y=(-0.6, 0.6), force_enabled=True),
)
