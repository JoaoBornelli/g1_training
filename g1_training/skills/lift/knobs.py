"""Knobs (números tunáveis) da skill LIFT — pegar a caixa e levar ao alvo.

Cada fine-tune = uma instância salva em `configs/<AAAA_MM_DD_nome>.py` (só os
campos que mudam vs. os defaults abaixo). Trocar o treino ATIVO = editar
`configs/__init__.py:ACTIVE` (1 linha). Voltar a um treino antigo = apontar
ACTIVE pra ele; puxar um valor = importar o config antigo direto.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scene:
    box_xy: tuple[float, float] = (0.50, 0.0)
    box_z: float | None = None          # None = repousa no topo da mesa (calculado)
    box_half: tuple[float, float, float] = (0.10, 0.10, 0.10)
    box_mass: float = 1.0
    table_xy: tuple[float, float] = (0.50, 0.0)
    table_half: tuple[float, float, float] = (0.30, 0.30, 0.275)
    table_mass: float = 20.0


@dataclass
class Command:
    target_x: tuple[float, float] = (0.40, 0.50)
    target_y: tuple[float, float] = (-0.05, 0.05)
    target_z: tuple[float, float] = (0.78, 0.85)   # acima do topo da mesa => erguer


@dataclass
class Reward:
    # cadeia MONOTÔNICA: reaching (sempre on) -> grasp (bônus de toque) ->
    # lift (progresso de altura) -> sustain_precise (segurar no alvo).
    reaching: float = 1.0
    std_coarse: float = 1.0
    std_fine: float = 0.25
    lateral_offset: float | None = None  # None = box_half[1] (alvo por-mão, meia-largura)
    grasp: float = 0.5
    lift: float = 2.0
    upright_gate_deg: float = 10.0       # ⚠ gradiente some além do gate; alargar se lift ~0
    sustain_precise: float = 1.0
    sustain_std: float = 0.05
    back: float = -0.5
    table_contact: float = -0.5
    com_balance: float = -5.0
    com_margin: float = 0.05
    # fundação escopada por skill:
    upright: float = 1.0   # ⚠ NUNCA baixar sem motivo forte — tentado e revertido 07-15
                            #    (afrouxar upright/posture pra "liberar reach" degradou o
                            #    treino inteiro; fundação de pé não é knob de reach).
    posture: float = 0.25


@dataclass
class Foundation:
    action_rate_l2: float = -0.1
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02


@dataclass
class Push:
    x: tuple[float, float] = (-0.5, 0.5)
    y: tuple[float, float] = (-0.5, 0.5)


@dataclass
class Train:
    entropy_coef: float = 0.01
    num_envs: int = 4096
    max_iterations: int = 3000


@dataclass
class LiftKnobs:
    scene: Scene = field(default_factory=Scene)
    command: Command = field(default_factory=Command)
    reward: Reward = field(default_factory=Reward)
    foundation: Foundation = field(default_factory=Foundation)
    push: Push = field(default_factory=Push)
    train: Train = field(default_factory=Train)
