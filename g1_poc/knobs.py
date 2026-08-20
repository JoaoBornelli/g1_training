"""Todos os números do g1_poc, num lugar só.

Regra do pacote: nenhum número solto no meio do código. Um treino tem de ser
reproduzível por `git diff` deste arquivo.

Ver ESPECIFICACAO-g1_poc.md, §3 (cena), §7 (elos), §8 (recompensas), §9 (postura),
§10 (currículo).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Cena:
    # --- caixa ---
    caixa_meia_aresta: tuple[float, float, float] = (0.10, 0.10, 0.10)
    caixa_massa: float = 1.0
    caixa_xy: tuple[float, float] = (0.32, 0.00)   # borda perto do robô (lição 16/07)
    caixa_jitter_x: tuple[float, float] = (0.00, 0.20)
    caixa_jitter_y: tuple[float, float] = (-0.18, 0.18)
    caixa_jitter_yaw_deg: float = 15.0

    # --- prateleira (mocap) ---
    prateleira_meia_xy: float = 0.30
    prateleira_meia_z: float = 0.02
    prateleira_xy: tuple[float, float] = (0.50, 0.00)
    # o piso é 0.04 porque a laje tem 4 cm: com o topo em 0.04 ela APOIA no chão
    # em vez de atravessá-lo (dois corpos estáticos em contato gastam slots).
    prateleira_topo_piso: float = 0.04
    prateleira_topo_teto: float = 0.55
    prateleira_jitter_z: float = 0.02

    # a mobília sai do grupo 0, senão o `foot_height_scan` do fabricante
    # (include_geom_groups=(0,)) lê a prateleira COMO CHÃO.
    grupo_mobilia: int = 2

    # afastamento na forma de locomoção
    afasta_z: float = 5.0

    # --- ação ---
    escala_acao_mult: float = 0.8

    # --- reset da base ---
    reset_base_pose: dict = field(default_factory=lambda: {
        "x": (-0.10, 0.00), "y": (-0.10, 0.10), "z": (0.01, 0.05), "yaw": (-0.2, 0.2),
    })
    # §11.1 — a entrega do navegador: o robô chega andando, não parado.
    reset_base_vel_manipulacao: dict = field(default_factory=lambda: {
        "x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-0.4, 0.4),
    })


@dataclass
class Alvo:
    """O alvo do elo `pegar`. Altura ABSOLUTA do mundo (§7.1).

    Agachar não move este alvo. Portanto o robô tem de ficar de pé.
    Os valores vêm da skill Lift, que fechou a tarefa com eles.
    """
    pegar_x: tuple[float, float] = (0.20, 0.30)
    pegar_y: tuple[float, float] = (-0.05, 0.05)
    pegar_z: tuple[float, float] = (0.78, 0.85)

    # elo `carregar`: alvo no PEITO, ancorado no frame da BASE (§7.1)
    peito_b: tuple[float, float, float] = (0.25, 0.00, 0.15)

    # elo `botar`: ponto na prateleira, deslocamento LATERAL (o frontal exigiria
    # alcançar por cima de 20 cm de tampo — defeito medido em 16/07)
    botar_x: tuple[float, float] = (0.30, 0.40)
    botar_y: tuple[float, float] = (-0.12, 0.12)
    botar_topo_piso: float = 0.30    # uma mesa real tem 0.70 a 0.80 m
    botar_topo_teto: float = 0.80


@dataclass
class Tolerancia:
    raio_sucesso: float = 0.05           # esfera de 5 cm
    angulo_sucesso_rad: float = math.radians(20.0)
    sustenta_pegar_s: float = 1.0        # o `pegar` exige 1,0 s (estabilidade)
    sustenta_outros_s: float = 0.5
    carregar_s: float = 6.0              # o elo `carregar` fecha por tempo
    # `de_pe`
    pelve_min: float = 0.65
    inclinacao_max_rad: float = math.radians(20.0)
    # `botar` só fecha quando a prateleira carrega o peso
    fracao_apoio_botar: float = 0.80


@dataclass
class Recompensa:
    # --- fundação: pesos do `velocity` do G1, sem mudança ---
    track_lin: float = 2.0
    track_lin_std: float = math.sqrt(0.25)
    track_ang: float = 2.0
    track_ang_std: float = math.sqrt(0.50)
    upright: float = 1.0
    upright_std: float = math.sqrt(0.20)
    postura: float = 1.0
    foot_clearance: float = -2.0
    foot_swing_height: float = -0.25
    foot_slip: float = -0.1
    soft_landing: float = -1e-5
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    action_rate: float = -0.10           # cronograma o leva a -0.25
    dof_pos_limits: float = -10.0        # valor de `manipulation` e `tracking`
    self_collisions: float = -1.0

    # --- tarefa ---
    staged: float = 3.0
    reaching_std: float = 0.20
    bringing_std_piso: float = 0.10      # σ variável: max(piso, distância comandada)
    precise_pos: float = 2.0
    precise_pos_std: float = 0.05
    precise_ori: float = 1.0
    precise_ori_std: float = 0.40
    squeeze: float = 1.0
    squeeze_mu: float = 0.8              # μ pessimista da faixa de DR
    # a PONTE do platô do grasp (19/08). O `squeeze` satura em 6× F_ref e para de
    # guiar; o apoio da prateleira é contínuo e cai antes de a caixa se mover.
    unload: float = 2.0                  # igual ao `precise_pos`: é sinal de TAREFA
    unload_tol_queda: float = 0.03       # 3 cm abaixo do repouso já é queda
    joint_vel_hinge: float = -0.01       # cronograma o leva a -1.0
    joint_vel_max: float = 0.5

    # o alvo de cada palma é a face lateral da caixa (não o centro)
    lateral_offset: float | None = None   # None = caixa_meia_aresta[1]


@dataclass
class Postura:
    """§9 — o quarto regime do `variable_posture`.

    Os três dicionários do G1 ficam intocados. Este é o quarto, e ele responde à
    DEMANDA DA CAIXA, não à velocidade comandada.

    Regra: as juntas do plano sagital abrem, e as laterais ficam apertadas.
    """
    peso_dist: float = 10.0     # demanda += 10 * ‖caixa − alvo‖
    peso_ang: float = 6.0       # demanda += 6 * Δθ (rad)
    limiar: float = 1.5         # = running_threshold do mjlab

    std_manipulando: dict = field(default_factory=lambda: {
        r".*knee.*": 1.20,
        r".*hip_pitch.*": 1.00,
        r".*ankle_pitch.*": 0.50,
        r".*hip_roll.*": 0.20,
        r".*hip_yaw.*": 0.20,
        r".*ankle_roll.*": 0.15,
        r".*waist_pitch.*": 0.40,
        r".*waist_yaw.*": 0.60,
        r".*waist_roll.*": 0.15,
        r".*shoulder_pitch.*": 1.00,
        r".*elbow.*": 1.00,
        r".*shoulder_roll.*": 0.60,
        r".*shoulder_yaw.*": 0.40,
        r".*wrist.*": 0.40,
    })


@dataclass
class Terminacao:
    fell_over_rad: float = math.radians(70.0)
    caixa_largada_z: float = 0.20
    caixa_largada_dist: float = 0.40
    contato_ilegal_N: float = 50.0
    auto_colisao_N: float = 10.0


@dataclass
class Comando:
    """O `twist`. Parâmetros do mjlab, mais o giro no lugar (acréscimo do repo)."""
    resample_s: tuple[float, float] = (3.0, 8.0)
    rel_standing: float = 0.10
    rel_heading: float = 0.30
    rel_forward: float = 0.20
    heading_stiffness: float = 0.5
    lin_vel_x: tuple[float, float] = (-0.5, 1.0)
    lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)
    # metade dos envs PARADOS recebe giro no lugar; sem isto eles nunca giram
    frac_giro_no_standing: float = 0.5
    piso_giro_rad_s: float = 0.15


@dataclass
class Episodio:
    duracao_s: float = 20.0
    frac_locomocao: float = 0.30    # 30% locomoção / 70% manipulação


@dataclass
class DR:
    foot_friction: tuple[float, float] = (0.3, 1.2)
    encoder_bias: tuple[float, float] = (-0.015, 0.015)
    # atrito da CAIXA, por episódio (mode="reset"), compartilhado entre as palmas
    caixa_friction: tuple[float, float] = (0.8, 1.2)
    # ⚠ Era (1.0, 3.0) e foi para (10.0, 20.0) em 19/08, medido pela sonda.
    #
    # O fecho do `pegar` exige `sustenta_pegar_s = 1,0 s` = 50 passos ININTERRUPTOS.
    # Com o intervalo em 1-3 s o push chegava a cada 50-150 passos, portanto o sucesso
    # dependia de sortear o intervalo longo. Medido no `model_5100`, o push sozinho —
    # sem ruído nem jitter — derrubava as quatro condições juntas de 63,4% para 29,3%
    # e a sustentação de 11,18 s para 4,28 s, e era o ÚNICO dos três fatores do treino
    # que degradava (ruído e jitter ficaram dentro da variação amostral).
    #
    # O cronômetro é POR ENV (`is_global_time = False`). Com o episódio de 20 s e o
    # intervalo em (10, 20), o número esperado de empurrões por episódio é 20/15 = 1,3:
    # tipicamente UM, no máximo dois. O robô continua tendo de se recuperar de push —
    # o que ele não precisa mais é sobreviver a um a cada meio fecho.
    push_intervalo_s: tuple[float, float] = (10.0, 20.0)
    # ⚠ knob MORTO: nada em g1_poc o lê. Fica declarado porque a janela livre depois do
    # push é a mitigação natural se (10, 20) ainda colidir com o `sustenta`.
    push_janela_livre_s: float = 0.5
    # ⚠ base_com fica DESLIGADO: `dr.body_com_offset` corrompe a heap (medido)
    base_com: bool = False
    carga_kg: tuple[float, float] = (1.0, 1.0)   # o nível a alarga


@dataclass
class Cronograma:
    """§10.2 e §10.3 — os dois cronogramas por passo global.

    Os passos estão em `common_step_counter`, que conta 1 por passo de env.
    24 é o `num_steps_per_env`, portanto `1000 * 24` = 1000 iterações.
    """
    it = 24
    # ⚠ Os degraus estavam em 1000 e 2500 e foram para 8000 e 12000 em 19/08.
    #
    # A §10.2 justificava o passo 0 nas faixas de hoje com "a locomoção do treino atual
    # já funciona" — premissa de WARM-START de uma política que andava. O bloco 1 rodou
    # do ZERO (`resume=False`), então ela nunca valeu, e o cronograma avançou dois
    # estágios por passo global sem o robô ter dado um passo: na it 5099 ele recebia
    # comando de 2,0 m/s e 1,5 rad/s com `peak_height_mean = 2,7 mm`.
    #
    # Cair em meio segundo com esse comando não é falha de aprendizado — é o comando ser
    # impossível. E o tempo de vida do episódio governa a fatia de dados: com o episódio
    # de andar em 24 passos contra 961 da manipulação, andar fica com 1,1% das
    # transições, porque o sorteio 70/30 é por EPISÓDIO e o PPO aprende de PASSO.
    #
    # Mesmo defeito de fase do `hinge`: cronograma por passo global contra um progresso
    # que não aconteceu. Dois dos três cronogramas já saíram de fase — ver a dívida do
    # gate por competência na §10.3.
    locomocao: list = field(default_factory=lambda: [
        {"step": 0,           "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.3, 0.3), "ang_vel_z": (-0.5, 0.5)},
        {"step": 8000 * 24,   "lin_vel_x": (-0.8, 1.5), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
        {"step": 12000 * 24,  "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-0.6, 0.6), "ang_vel_z": (-1.5, 1.5)},
    ])
    # ⚠ O degrau de −1,00 estava em 3000 e foi para 10000 em 19/08, medido.
    # Na iteração 3080 ele bateu e o `joint_vel_hinge` + `action_rate_l2` passaram a
    # consumir 99,1% de TODAS as penalidades e 100% de todo o sinal positivo: a
    # recompensa líquida virou −0,03 (Mean reward −0,37). No mesmo passo o
    # `contato_ilegal` foi de 6,4% para 17,5% das terminações — com movimento caro,
    # escorar o tronco na prateleira economiza esforço.
    #
    # A causa é de FASE, não de valor: o cronograma é por passo global e pressupõe a
    # tarefa resolvida em 3000 iterações. Ela só saiu de zero em 3080
    # (`episode_success` 0,0060, o primeiro do projeto). A §17 põe "refino de pose" no
    # passo 6, o ÚLTIMO, e o freio chegou cinco passos adiantado.
    hinge: list = field(default_factory=lambda: [
        {"step": 0,          "weight": -0.01},
        {"step": 1500 * 24,  "weight": -0.10},
        {"step": 10000 * 24, "weight": -1.00},
    ])
    action_rate: list = field(default_factory=lambda: [
        {"step": 0,         "weight": -0.10},
        {"step": 3000 * 24, "weight": -0.25},
    ])


@dataclass
class Treino:
    """Números que o CLI sobrescreve, mas que ficam registrados aqui."""
    # ⚠ o default de `num_envs` no mjlab é 1. Esquecer a flag roda 1 env em silêncio.
    num_envs: int = 4096
    # warm-start SEMPRE com 5e-4 (lição do repo). O ADR-0001 declarou e nunca aplicou.
    lr_warm_start: float = 5e-4


@dataclass
class Knobs:
    cena: Cena = field(default_factory=Cena)
    alvo: Alvo = field(default_factory=Alvo)
    tol: Tolerancia = field(default_factory=Tolerancia)
    reward: Recompensa = field(default_factory=Recompensa)
    postura: Postura = field(default_factory=Postura)
    term: Terminacao = field(default_factory=Terminacao)
    comando: Comando = field(default_factory=Comando)
    episodio: Episodio = field(default_factory=Episodio)
    dr: DR = field(default_factory=DR)
    cronograma: Cronograma = field(default_factory=Cronograma)
    treino: Treino = field(default_factory=Treino)


ATIVO = Knobs()
