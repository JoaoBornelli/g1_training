"""Knobs (números tunáveis) da skill STAND — ficar de pé parado.

Cada treino salvo = uma instância em `configs/<nome>.py`. `baseline` = de pé
calmo; `step_recovery` = passo protetivo emergente sob empurrão forte (ver
`configs/step_recovery.py` pro racional)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Push:
    x: tuple[float, float] = (-0.5, 0.5)
    y: tuple[float, float] = (-0.5, 0.5)
    force_enabled: bool = False               # apply_body_impulse sustentado (só Step)
    force_range: tuple[float, float] = (-50.0, 50.0)
    force_duration_s: tuple[float, float] = (0.3, 3.0)
    force_cooldown_s: tuple[float, float] = (1.5, 3.0)


@dataclass
class Foundation:
    action_rate_l2: float = -0.1
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    posture_weight: float = 0.5
    posture_joints: tuple[str, ...] = (".*",)  # Stand: corpo TODO (parado, sem manipular)


@dataclass
class Train:
    entropy_coef: float = 0.01


@dataclass
class StandKnobs:
    foundation: Foundation = field(default_factory=Foundation)
    push: Push = field(default_factory=Push)
    train: Train = field(default_factory=Train)
