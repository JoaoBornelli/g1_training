"""Stand-Step: passo protetivo emergente sob empurrão forte.

DRIVER-ONLY (2026-07-15, ver [[g1-lift-box-task]]): os 4 termos de marcha do
mjlab (foot_clearance/air_time/foot_slip/soft_landing) são gateados pelo
comando `twist` — numa task sem comando de velocidade dão KeyError ou ficam
inertes. O que CRIA o passo aqui é o DRIVER (push forte + upright + penalidades
anti-dinâmica afrouxadas), que não depende de comando nenhum. Números =
retrofit exato do `gait_recovery=True` original."""
from ..knobs import Foundation, Push, StandKnobs

KNOBS = StandKnobs(
    foundation=Foundation(action_rate_l2=-0.03, body_ang_vel=-0.01, angular_momentum=-0.005),
    push=Push(x=(-0.8, 0.8), y=(-0.8, 0.8), force_enabled=True),
)
