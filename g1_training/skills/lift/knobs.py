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
    # DR de posição: OFFSET (lo, hi) em metros somado a box_xy em CADA reset. (0,0)=off.
    # x costuma ser ASSIMÉTRICO (só +, pra dentro da mesa): box_xy fica na borda da
    # frente, então -x jogaria a caixa pra fora e ela cairia. y é simétrico (mesa simétrica).
    box_jitter_x: tuple[float, float] = (0.0, 0.0)
    box_jitter_y: tuple[float, float] = (0.0, 0.0)
    # REHEARSAL anti-esquecimento: fração dos envs spawna a caixa a `rehearsal_far_x`
    # metros (fora de alcance, no chão) → esses envs só praticam ficar de pé. 0=off.
    rehearsal_fraction: float = 0.0
    rehearsal_far_x: float = 5.0
    box_half: tuple[float, float, float] = (0.10, 0.10, 0.10)
    box_mass: float = 1.0
    # PESO variável da caixa (payload) — massa EFETIVA total em kg. None = off (usa box_mass
    # real). Setado (lo, hi): cada episódio sorteia um peso e aplica força −z extra na caixa
    # (evento apply_box_payload, mode=reset). Mecanismo TESTADO (write_external_wrench_to_sim),
    # NÃO o dr.body_mass/pseudo_inertia (não-testado, corrompe heap). Só peso, não inércia.
    box_weight_range: tuple[float, float] | None = None
    # PRATELEIRA mocap (fina, flutuante): `shelf_top` = altura do topo (onde a caixa
    # repousa em cima). É o EIXO do currículo de altura — baixar shelf_top desce a
    # caixa rumo ao chão, por-env em runtime, SEM recompilar (era o limite da mesa).
    # box repousa em z = shelf_top + box_half[2] (calculado no env).
    table_xy: tuple[float, float] = (0.50, 0.0)
    shelf_top: float = 0.55             # altura de repouso da caixa (topo da prateleira)
    shelf_half_xy: float = 0.30
    shelf_half_z: float = 0.02          # fina (plateleira, não paredão)


@dataclass
class Command:
    # target_x era (0.40, 0.50) -- MAIS LONGE que o próprio box_xy (0.30, borda
    # da mesa): depois de pegar num alcance confortável, o alvo pedia pra
    # empurrar a caixa ainda mais pra frente (user 07-16: "muito longe, não vai
    # conseguir alcançar"). Trazido pra dentro do alcance de pega -- o robô pode
    # puxar a caixa em direção ao corpo enquanto ergue (mais alavanca).
    target_x: tuple[float, float] = (0.20, 0.30)
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
    upright_std: float = 0.1             # kernel suave de orientação (menor = mais exigente)
    sustain_precise: float = 1.0
    sustain_std: float = 0.05
    back: float = -0.5
    table_contact: float = -0.8
    com_balance: float = -3.0
    com_margin: float = 0.08
    box_shake: float = -0.15             # pune vel. angular² da caixa (sacudir/rodar violento)
    arm_vel: float = 0.0                  # freio de VELOCIDADE das juntas do braço (0=off;
                                          # joint_vel_l2 em shoulder/elbow/wrist). Ataca o
                                          # "correr" pra pegar/levar. TODO: migrar p/ accel/torque.
    # fundação escopada por skill:
    upright: float = 1.0   # ⚠ NUNCA baixar sem motivo forte — tentado e revertido 07-15
                            #    (afrouxar upright/posture pra "liberar reach" degradou o
                            #    treino inteiro; fundação de pé não é knob de reach).
    posture: float = 0.5


@dataclass
class Foundation:
    action_rate_l2: float = -0.1
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02


@dataclass
class Push:
    x: tuple[float, float] = (-0.5, 0.5)
    y: tuple[float, float] = (-0.5, 0.5)
    # FORÇA SUSTENTADA (apply_body_impulse na pelvis) — a perturbação contínua que
    # imita o peso da caixa deslocando o CoM; robustez de equilíbrio sob carga. Só treino.
    force_enabled: bool = False
    force_range: tuple[float, float] = (-50.0, 50.0)
    force_duration_s: tuple[float, float] = (0.3, 3.0)
    force_cooldown_s: tuple[float, float] = (1.5, 3.0)


@dataclass
class Plr:
    """Prioritized Level Replay sobre ALTURAS de prateleira (currículo de altura).

    Cada "nível" = uma altura de repouso da caixa. Lista VAZIA = PLR OFF (usa o
    `shelf_top` único da Scene, comportamento antigo). Adicionar um degrau do currículo
    = fazer APPEND de uma altura nova aqui (as anteriores continuam no sorteio →
    anti-esquecimento). O sampler rank-based dá um piso a todas e foca sozinho na mais
    difícil. Ver common/curriculums.py pro mecanismo."""
    shelf_levels: tuple[float, ...] = ()   # () = OFF; ex: (0.55, 0.45, 0.35) do alto p/ baixo
    floor_rho: float = 0.30                # piso uniforme: massa mínima em TODA altura
    focus_beta: float = 1.0               # agressividade do foco na altura difícil (rank)
    ema_alpha: float = 0.1                 # inércia do score de dificuldade (EMA)
    level_jitter_z: float = 0.02           # ±jitter em z dentro de cada nível (generaliza)
    seed_newest_high: bool = True          # nível mais novo (menor altura) começa prioritário
    sustain_term: str = "sustain_precise"  # termo cuja soma do episódio vira performance
    sustain_weight: float = 1.0            # peso do termo (normaliza performance p/ [0,1])


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
    plr: Plr = field(default_factory=Plr)
    train: Train = field(default_factory=Train)
