"""Todos os números da v3, num lugar só (A2 do plano; contrato em §6.2).

Regra do pacote: nenhum número solto no meio do código. Um treino tem de ser
reproduzível por `git diff` deste arquivo.

⚠ MÓDULO PURO: importa só `dataclasses`, `math`, `pathlib` e `numpy`. Nunca torch, mjlab nem
mujoco (plano §1.5). O sim do real e o robô carregam este arquivo por caminho, sem o
pacote (PD-43); por isso ele não importa outros módulos do g1_v3 nem do g1_limpo.

Os valores foram TRANSCRITOS da §6.2 do plano, da spec (`specs/g1-v3.md`), da interface
com o robô real (`specs/g1-v3-interface-real.md`) e dos `knobs.py` do g1_limpo. Cada campo
traz a fonte na linha. Campo `None` quer dizer "sai da Parte 0" (item indicado na fonte);
o `env_cfg.make_env_cfg` faz `assert` de que nenhum `None` chega ao cfg (§11, D10).

Pares `(início, refino)`: o valor do degrau d do refino é início + (fim − início)·d/5
(plano §6.9). Quem lê o par escolhe o índice pelo degrau.

`DR` (plano §6.8) e `Refino` (§6.9) estão no fim das classes; o `Refino` aponta para os
pares das outras seções pelo nome do campo do `Knobs` (em minúsculo).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ordem dos motores do SDK = ordem do MJCF (I §2.2)
JUNTAS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
TAREFAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR", "REORIENTAR", "LEVANTAR")  # one-hot (I §2.2)
FATIAS = ("ANDAR", "B", "C")  # fatias do currículo (§6.2)
ALVOS = ("P0", "P1", "P2", "P3", "P4", "B1", "B2", "B3", "B4")  # índice 0..8; −1 = nenhum
# Sentinela "nunca reamostra" (plano §1.5, D14 R6): passos de atraso sem reamostra. Uma só, para o
# lag sorteado no reset ficar no episódio inteiro. Não é número de treino, é o valor "infinito" do mjlab.
NUNCA = 10**9


@dataclass(frozen=True)
class Contrato:  # I §2, §3; S §6.3, §19.1
    escala_acao: float = 0.25  # I §3: q* = q_de_pé + 0,25·a; S §6.3
    dt_fisica: float = 0.005  # I §3: passo de física 5 ms; S §19
    decimacao: int = 4  # I §3: passo da política 0,02 s
    historico: int = 5  # I §2.1: histórico por bloco
    # (nome, canais por passo, escala, histórico) na ordem do vetor do ator
    blocos_ator: tuple = (  # I §2.1 (1 447 números)
        ("giro_base", 3, 0.2, 5), ("gravidade", 3, 1.0, 5),
        ("comando", 3, 1.0, 5), ("juntas", 29, 1.0, 5),
        ("vel_juntas", 29, 0.05, 5), ("acao", 29, 1.0, 5),
        ("tarefa", 6, 1.0, 5), ("caixa", 11, 1.0, 5), ("mapa", 882, 1.0, 0),
    )
    # (regex das juntas, kp, kd): I §3; UNITREE_G1_29DOF_CFG, commit 4960b84
    pd: tuple = (
        (r".*_hip_pitch_joint", 100.0, 2.0), (r".*_hip_yaw_joint", 100.0, 2.0),
        (r".*_hip_roll_joint", 100.0, 2.0), (r".*_knee_joint", 150.0, 4.0),
        (r".*_ankle_(pitch|roll)_joint", 40.0, 2.0), (r"waist_yaw_joint", 200.0, 5.0),
        (r"waist_(roll|pitch)_joint", 40.0, 5.0),
        (r".*_(shoulder_.*|elbow|wrist_roll)_joint", 40.0, 1.0),
        (r".*_wrist_(pitch|yaw)_joint", 40.0, 1.0),
    )
    referencias: str = "g1_v3/dados/referencias.npz"  # A3; relativo à raiz do repo


@dataclass(frozen=True)
class Cena:  # S §11.2, §12; PD-9, PD-46, PD-47; g1_limpo/knobs.py Cena e Nivel (cópia)
    caixa_meia_faixa: tuple = (0.07, 0.13)  # S §12: meia aresta da caixa (DR de tamanho)
    caixa_n_variantes: int = 8  # g1_limpo/knobs.py:31; S §12
    caixa_massa_base: float = 1.0  # S §11.2
    pad_atrito: tuple = (1.0, 0.02, 0.001)  # g1_limpo/cena.py:245 (atrito dos pads de palma)
    caixa_priority: int = 0  # PD-46
    caixa_atrito: tuple = (0.3, 0.02, 0.001)  # PD-46
    robo_atrito: tuple = (0.3, 0.02, 0.001)  # PD-46
    destino_borda: float = 0.15  # PD-47: borda da laje de destino (m)
    destino_jitter_y: float = 0.10  # PD-47: jitter em y do destino (m)
    destino_giro_max_graus: float = 10.0  # PD-47: giro máximo do destino
    grupo_mobilia: int = 1  # PD-9: grupo de geoms da mobília (1; g1_limpo usa 2)
    prateleira_meia_xy: float = 0.30  # g1_limpo/knobs.py Cena (cópia)
    prateleira_meia_z: float = 0.02  # g1_limpo/knobs.py Cena (cópia)
    prateleira_atrito: tuple = (1.0, 0.02, 0.001)  # g1_limpo/knobs.py Cena (cópia)
    prateleira_rgba: tuple = (0.5, 0.5, 0.55, 1.0)  # g1_limpo/knobs.py Cena (cópia)
    prateleira_condim: int = 3  # g1_limpo/knobs.py Cena (cópia)
    prateleira_xy: tuple = (0.50, 0.00)  # g1_limpo/knobs.py Cena (cópia)
    prateleira_topo_teto: float = 0.55  # g1_limpo/knobs.py Cena (cópia)
    caixa_xy: tuple = (0.32, 0.00)  # g1_limpo/knobs.py Cena (cópia)
    caixa_jitter_y: tuple = (-0.18, 0.18)  # g1_limpo/knobs.py Cena (cópia)
    # ⚠ por nível do PEGAR e do BOTAR (5 valores, índice = nível): o jitter em x aperta com o nível (g1_limpo/knobs.py Nivel:360)
    jitter_x_max: tuple = (0.20, 0.20, 0.20, 0.15, 0.08)
    # E11 S4: só z e yaw. O x e o y da base da pega vêm da `ref.recuo` ± `faixa_recuo` (eventos `_base_da_pega`)
    reset_base_manipula: tuple = (("z", (0.01, 0.05)), ("yaw", (-0.2, 0.2)))
    # D7 (29/10): reset da base no ANDAR, sobre a origem do terreno; rumo livre (plano §6.2)
    reset_base_andar: tuple = (("x", (-0.5, 0.5)), ("y", (-0.5, 0.5)), ("yaw", (-math.pi, math.pi)))
    # PD-46 (29/10: sai da cena.py): a laje tem priority 3, acima da caixa (0) e do robô (0; pés 1)
    prioridade_laje: int = 3
    # ⚠ cicatriz de 15/07 no g1_limpo: `elliptic` com `impratio=10` divergiu para NaN no reset
    njmax: int = 800  # g1_limpo/knobs.py:103; S §12 (L2)
    nconmax: int = 300  # g1_limpo/knobs.py:104; S §12 (L2)
    impratio: float = 1.0  # g1_limpo/knobs.py:105; S §12 (L2)
    cone: str = "pyramidal"  # g1_limpo/knobs.py:106; S §12 (L2)
    extent: float = 2.5  # g1_limpo/knobs.py:86
    # D14 M5: laje e caixa nascem `estaciona_z` acima do mundo (fase D); a fase E (`estaciona_cena`) usa o mesmo campo
    estaciona_z: float = 20.0
    caixa_condim: int = 3  # D14 R6: padrão do MuJoCo; g1_limpo/knobs.py Cena.caixa_condim


@dataclass(frozen=True)
class Comando:  # S §7, §11.2, §19
    frente_tetos: tuple = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)  # S §11.2: tetos para a frente (m/s)
    tras_lado_teto_max: float = 1.0  # S §11.2: teto para trás e de lado (m/s)
    giro_tetos: tuple = (0.25, 0.5, 0.75, 1.0)  # S §11.2: tetos de giro (rad/s)
    piso_velocidade: float = 0.25  # S §11.2: "entre 0,25 e o teto"
    baixo: tuple = (0.05, 0.25)  # S §19, conflito 5: subtarefa de comando baixo (m/s; rad/s)
    reamostra_s: tuple = (3.0, 8.0)  # S §7: intervalo de reamostra do comando (s)
    janela_media_s: float = 0.8  # S §19, conflito 5: média da velocidade (s)
    limiar_canal: float = 0.02  # S §19, conflito 5: canal "sem comando" abaixo disto
    deriva_pos_m: float = 0.10  # PD-15: deriva da âncora de posição (m)
    deriva_rumo_rad: float = 0.25  # PD-15: deriva da âncora de rumo (rad)
    ancora_s: float = 1.5  # PD-15: tempo da âncora (s)
    razao_portao: float = 0.50  # S §11.3: razão de deslocamento para o sucesso
    pedido_min_segmento: float = 0.5  # g1_limpo/comando.py:2507: duração mínima do segmento (s)
    terreno_v_max: float = 1.0  # PD-12: célula terreno usa o reto do nível (m/s)
    incl_max_terreno_graus: float = 15.0  # G3; S §11.3: a caixa a ≤ 15° em todo passo do CARREGAR no terreno


@dataclass(frozen=True)
class Cadeia:  # S §7
    espera_s: tuple = (0.5, 1.0)  # S §7: espera entre trechos (s)
    carregar_b_s: tuple = (4.0, 7.0)  # S §7: duração do CARREGAR de B (s)
    carregar_c_s: tuple = (0.5, 1.0)  # S §7: duração do CARREGAR de C (s)
    andar_final_s: float = 1.5  # S §7: ANDAR final (s)
    atraso_pegar_s: tuple = (0.0, 1.0)  # S §7: atraso antes do PEGAR (s)
    atraso_botar_s: tuple = (0.0, 1.5)  # S §7: atraso antes do BOTAR (s)
    episodio_s: float = 20.0  # S §7: duração do episódio (s)
    caixa_perdida_vira_andar: bool = True  # PD-27: caixa perdida vira ANDAR


@dataclass(frozen=True)
class Detector:  # I §4.2; PD-26
    tol_pelve: float = 0.03  # I §4.2: tolerância da pelve (m)
    tol_tronco_graus: float = 5.0  # I §4.2: tolerância do tronco (graus)
    tol_palma: float = 0.05  # I §4.2: tolerância das palmas (m)
    estavel_s: float = 0.5  # I §4.2: janela de estabilidade (s)
    juntas_paradas: float = 0.2  # I §4.2; PD-26: limiar de juntas paradas (|q̇| médio)


@dataclass(frozen=True)
class Alvos:  # S §8.2; PD-23
    p0_frente: float = 0.05  # S §8.2: P0, avanço à frente da caixa (m)
    palma_face: float = 0.035  # S §8.2: palma à face da caixa (m)
    p3_alto: float = 0.05  # S §8.2: P3, altura acima da pega (m)
    b1_alto: float = 0.05  # S §8.2: B1, altura acima da soltura (m)
    tol_palma: float = 0.03  # S §8.2: tolerância da palma (m)
    tol_caixa_p4: float = 0.05  # S §8.2: tolerância da caixa em P4 (m)
    tol_pelve_p4: float = 0.03  # S §8.2: tolerância da pelve em P4 (m)
    tempo_p4: float = 0.5  # S §8.2: tempo de permanência em P4 (s)
    tol_b1_horizontal: float = 0.03  # S §8.2: tolerância horizontal de B1 (m)
    raio_b2: float = 0.10  # S §8.2: raio de B2 (m)
    apoio_b2_mg: float = 0.8  # S §8.2: fração do peso no apoio de B2
    tempo_b2: float = 0.3  # S §8.2: tempo de permanência em B2 (s)
    b3_afasta: float = 0.05  # S §8.2: afastamento das palmas em B3 (m)
    tempo_b3: float = 0.5  # S §8.2: tempo de permanência em B3 (s)
    v_parada_b3: float = 0.05  # PD-39: caixa parada no B3, ‖v_caixa‖ abaixo disto (m/s)
    tol_pelve_b4: float = Detector.tol_pelve  # S §8.2; I §4.2: a mesma tolerância do detector (m)
    tol_tronco_b4_graus: float = Detector.tol_tronco_graus  # S §8.2; I §4.2: a mesma do detector (graus)
    tempo_b4: float = 0.5  # S §8.2: tempo de permanência em B4 (s)
    z_transporte: float = 0.90  # S §8.2: altura de transporte sobre o chão local (m); PD-21
    incl_inicio_graus: float = 30.0  # S §8.2, §9.1: inclinação inicial da caixa (graus)
    incl_fim_graus: float = 15.0  # S §8.2, §9.1: inclinação final da caixa (graus)
    delta_incl_graus: float = 15.0  # S §8.2, §9.1: tolerância de inclinação (graus)
    sigma_min: float = 0.01  # E3: piso do σ do alvo métrico (m); σ = 0 daria NaN no núcleo
    desloc_p0p2_frac: float = 1.0  # S §8.2; PD-23: deslocamento P0–P2, × meia aresta
    delta_p0p2_frac: float = 0.5  # S §8.2; PD-23: tolerância P0–P2, × meia aresta
    desloc_b2b4: float = 0.02  # S §8.2; PD-23: deslocamento B2–B4 (m)
    delta_b2b4: float = 0.02  # S §8.2; PD-23: tolerância B2–B4 (m)
    faixa_tronco_graus: float = 15.0  # S §8.2: faixa do tronco (graus)
    faixa_recuo: float = 0.08  # S §8.2: faixa do recuo (m)
    faixa_com: float = 0.03  # S §8.2: faixa do centro de massa (m)
    delta_tronco_graus: float = 15.0  # PD-23: tolerância do tronco (graus)
    delta_recuo: float = 0.08  # PD-23: tolerância do recuo (m)
    delta_com: float = 0.03  # PD-23: tolerância do centro de massa (m)


@dataclass(frozen=True)
class Recompensa:  # S §9.1–9.4, §10, §11.4, §19; PD-45, PD-49
    rastreio_linear: float = 1.0  # S §9.2, §19: receita da Unitree
    rastreio_angular: float = 0.5  # S §9.2, §19: receita da Unitree
    sigma_rastreio: float = 0.5  # S §19: σ do rastreio (receita da Unitree)
    piso_rastreio: float = -1.0  # S §19, conflito 5: piso do rastreio normalizado
    # ⚠ σ₀ do canal sem comando (dono, 15/10). Com σ 0,5, deslizar a 0,05 m/s custava 0,01.
    sigma_zero_lin: float = 0.10  # PD-49: σ₀ do linear sem comando (m/s)
    sigma_zero_ang: float = 0.15  # PD-49: σ₀ do giro sem comando (rad/s)
    passo: float = 0.5  # S §19, conflito 6: peso do comprimento do passo
    L_min: float = 0.20  # S §19, conflito 6: comprimento mínimo do passo (m); calibrar na Parte 0, item 8
    fator_T: tuple = (0.8, 1.4)  # S §19, conflito 6: faixa de L_alvo em × |v|
    vivo: float = 0.15  # S §9.2: peso do "vivo" (zero depois da caixa perdida)
    altura_pe: float = 1.0  # S §9.2: peso da altura do pé; foot_clearance_reward (unitree_rl_lab)
    alvo_pe: float = 0.10  # S §9.2: altura alvo do pé (m); foot_clearance_reward
    std_pe: float = 0.05  # S §9.2: σ da altura do pé (m); foot_clearance_reward
    tanh_pe: float = 2.0  # S §9.2: escala tanh da altura do pé; foot_clearance_reward
    escorrego: float = -0.2  # S §9.2: peso do escorregão do pé (receita da Unitree)
    vz: float = -2.0  # S §9.2: peso da velocidade vertical da base
    giro_xy: float = -0.05  # S §9.2: peso do giro em x e y
    vel_junta: float = -0.001  # S §9.2: peso da velocidade de junta
    acel_junta: float = -2.5e-7  # S §9.2: peso da aceleração de junta
    taxa_acao: float = -0.05  # S §9.2: peso do action_rate (receita da Unitree)
    energia: float = -2e-5  # S §9.2: peso da energia
    # ⚠ limite mole 95% pelo dono (08/10); a receita da Unitree usa 90%; a IK fica a 90%
    limite_junta: float = -5.0  # S §9.2: peso do limite de junta
    limite_mole: float = 0.95  # S §9.2, §6.5: fração do curso onde a rampa começa
    desvio_bracos: float = -0.1  # S §9.2: peso do desvio dos braços
    desvio_cintura: float = -1.0  # S §9.2: peso do desvio da cintura
    desvio_quadril: float = -1.0  # S §9.2: peso do desvio de hip_roll e hip_yaw
    tronco_plano: float = -5.0  # S §9.2: peso do tronco plano
    altura_base: float = -10.0  # S §9.2: peso da altura da base
    alvo_base: float = 0.78  # S §9.2: altura alvo da base (m)
    cerca_sola: float = -0.00131  # S §9.3; PD-30; Parte 0, item 3 (report §6); D4
    cerca_giro: float = -0.00095  # S §9.3; PD-30; Parte 0, item 3 (report §6); D4
    cerca_tronco: float = -0.00110  # S §9.3; PD-30; Parte 0, item 3 (report §6); D4
    cerca_com: float = -0.52945  # S §9.3; PD-30; Parte 0, item 3 (report §6); D4
    cerca_torque: float = -0.00710  # S §9.3; PD-30; Parte 0, item 3 (report §6); D4
    cerca_pelve: float = -0.10672  # S §9.3; PD-30; Parte 0, item 3 (report §6); D4
    cerca_recuo: float = -0.24920  # S §9.3; PD-30; Parte 0, item 3 (report §6); D4
    lim_sola_graus: float = 5.0  # S §9.3: limite da sola (graus)
    lim_giro_graus: float = 30.0  # S §9.3: limite do giro (graus)
    lim_tronco_graus: float = 15.0  # S §9.3: limite do tronco (graus)
    lim_com: float = 0.03  # S §9.3: limite do centro de massa (m)
    lim_torque: float = 0.8  # S §9.3: limite de torque (fração do τ_max)
    banda_torque: float = 0.2  # S §9.3: banda do torque (fração do τ_max)
    lim_pelve: float = 0.08  # S §9.3: limite da pelve (m)
    lim_recuo: float = 0.08  # S §9.3: limite do recuo (m)
    # Tabela por regex (cada junta casa UMA regex; ver test_knobs). Valores em N·m.
    tau_max: tuple = (  # S §9.3: joelho e hip_roll 139; hip_pitch, hip_yaw, waist_yaw 88; etc.
        (r".*knee_joint", 139.0), (r".*hip_roll_joint", 139.0),
        (r".*hip_(pitch|yaw)_joint", 88.0), (r"waist_yaw_joint", 88.0),
        (r"waist_(roll|pitch)_joint", 50.0), (r".*ankle_(pitch|roll)_joint", 50.0),
        (r".*_(shoulder_.*|elbow|wrist_roll)_joint", 25.0),
        (r".*_wrist_(pitch|yaw)_joint", 5.0),
    )
    pares_torque: tuple = (  # PD-45: pares de juntas com torque somado
        ("waist_pitch_joint", "waist_roll_joint"),
        ("left_ankle_pitch_joint", "left_ankle_roll_joint"),
        ("right_ankle_pitch_joint", "right_ankle_roll_joint"),
    )
    tau_par: float = 50.0  # PD-45: limite do torque somado dos pares
    contato_fora: float = -1.0  # S §9.2: peso do contato fora dos pés
    contato_fora_N: float = 1.0  # S §9.2: força do contato fora dos pés (N)
    contato_laje: float = -2.0  # S §9.2: peso do contato com a laje; g1_limpo/knobs.py Contato
    laje_joelho_N: tuple = (50.0, 10.0)  # S §9.2, §6.9: força (N) abaixo da qual a multa é zero; (início; refino); g1_limpo/knobs.py Contato
    laje_saturacao_N: float = 100.0  # S §9.2: força em que a multa satura; g1_limpo/knobs.py Contato
    freio: float = -6 / 29  # S §9.2, §11.4: peso do freio por degrau (−6/29 ≈ −0,207)
    freio_fator: float = 1.5  # S §9.2, §11.4
    # Regime do freio pelo total do comando (|v| + |ω|): parado abaixo do 1º, correndo a partir do 2º (g1_limpo
    # `velocidade_por_regime`; 29/10: saíram do literal do `freio`)
    freio_limiar_andar: float = 0.05
    freio_limiar_correr: float = 1.5
    # Tetos de velocidade por família de junta: (regex, rad/s). Cópia de g1_limpo/knobs.py Tarefa.
    vel_max_parado: tuple = (  # g1_limpo/knobs.py Tarefa:762-767 (vel_max_standing)
        (r".*hip_pitch.*", 1.5), (r".*hip_roll.*", 1.5), (r".*hip_yaw.*", 1.5),
        (r".*knee.*", 1.5), (r".*ankle_pitch.*", 1.5), (r".*ankle_roll.*", 1.5),
        (r".*waist_yaw.*", 1.5), (r".*waist_roll.*", 1.5), (r".*waist_pitch.*", 1.5),
        (r".*shoulder_pitch.*", 1.5), (r".*shoulder_roll.*", 1.5),
        (r".*shoulder_yaw.*", 1.5), (r".*elbow.*", 1.5), (r".*wrist.*", 1.0),
    )
    vel_max_andando: tuple = (  # g1_limpo/knobs.py Tarefa:778-784 (vel_max_walking)
        (r".*hip_pitch.*", 4.0), (r".*hip_roll.*", 5.0), (r".*hip_yaw.*", 4.0),
        (r".*knee.*", 6.5), (r".*ankle_pitch.*", 6.0), (r".*ankle_roll.*", 4.0),
        (r".*waist_yaw.*", 3.0), (r".*waist_roll.*", 5.0), (r".*waist_pitch.*", 2.5),
        (r".*shoulder_pitch.*", 3.0), (r".*shoulder_roll.*", 4.0),
        (r".*shoulder_yaw.*", 2.0), (r".*elbow.*", 3.0), (r".*wrist.*", 2.5),
    )
    vel_max_correndo: tuple = (  # g1_limpo/knobs.py Tarefa:785-791 (vel_max_running)
        (r".*hip_pitch.*", 5.0), (r".*hip_roll.*", 5.0), (r".*hip_yaw.*", 4.0),
        (r".*knee.*", 10.0), (r".*ankle_pitch.*", 9.0), (r".*ankle_roll.*", 4.0),
        (r".*waist_yaw.*", 3.0), (r".*waist_roll.*", 5.0), (r".*waist_pitch.*", 2.5),
        (r".*shoulder_pitch.*", 3.5), (r".*shoulder_roll.*", 4.5),
        (r".*shoulder_yaw.*", 2.5), (r".*elbow.*", 3.0), (r".*wrist.*", 2.5),
    )
    progresso: float = 1.8167  # S §9.4; Parte 0, item 3 (report §6); D1
    # um valor por alvo, na ordem de ALVOS; Parte 0, item 3 (report §6); PD-31
    b_alvo: tuple = (1.6721, 0.8360, 0.0334, 0.8360, 1.6721, 1.8144, 0.9072, 0.9072, 1.8144)  # S §9.4
    gamma: float = 0.99  # S §9.1: desconto do potencial e do PPO
    sacudida: float = 0.05  # S §8.2, §11.5; Parte 0, item 15 (report §6); D7
    aperto: float = 0.02  # S §8.2, §11.5; Parte 0, item 15 (report §6); D6, D8
    F_ref: float = 49.0  # S §8.2, §11.5: força de referência do aperto (N)
    terminacao: float = -200.0  # S §10: peso da terminação
    perda_caixa: float = 28.0515  # S §10; Parte 0, item 3 (report §6); D6
    devolve_b_alvo: bool = False  # S §10; PD-33; Parte 0, item 3 (report §6)
    sigma_pouso: tuple = (1.0, 0.35)  # S §9.1: σ do pouso (m/s), (início; refino)
    altura_pouso: float = 0.05  # S §9.1: altura do pouso (m), fator pleno abaixo dela
    altura_pouso_inicio: float = 0.10  # S §9.1; D3 da Parte 0 (dono, 29/10): o fator entra aos 0,10 m e fica pleno aos 0,05 m


@dataclass(frozen=True)
class Terminacao:  # S §10; g1_limpo/knobs.py Terminacao (cópia)
    incl_marcha: float = 0.8  # S §10: inclinação de marcha (rad)
    base_min: float = 0.20  # S §10: altura mínima da base (m)
    incl_manip_graus: float = 70.0  # S §10: inclinação na manipulação (graus)
    joelho_min: float = 0.05  # S §10; g1_limpo/knobs.py Terminacao: joelho_z_min (raio do joelho, m)
    caixa_folga_chao: float = 0.02  # g1_limpo/knobs.py Terminacao: folga do fundo da caixa ao chão (m)
    v_solta: float = 1.2  # g1_limpo/knobs.py Terminacao: velocidade relativa de caixa solta (m/s)
    chegada: tuple = (1.0, 0.25)  # S §10: (início; refino) da velocidade de chegada (m/s)
    hip_yaw_max: float = 1.0  # S §10: limite do hip_yaw (rad)
    batente_frac: float = 1.0  # S §10: fração do batente da tabela de juntas


@dataclass(frozen=True)
class Balanceador:  # S §11, §11.5; PD-12, PD-13
    piso_tarefa: float = 0.20  # S §11.1; PD-13: piso de fração por tarefa
    piso_sub: float = 0.10  # S §11.2; PD-13: piso de fração por subtarefa
    eps: float = 0.02  # S §11; PD-13: ε da fração de déficit
    portao: float = 0.75  # S §11.3: portão de sucesso da célula
    alfa: float = 0.05  # S §11; PD-13: taxa da EMA do sucesso
    iters_min_nivel: int = 50  # PD-12: carência mínima entre subidas de nível (iterações)
    iters_refino: int = 100  # S §11.5: iterações seguidas no portão para avançar o refino
    degraus: int = 5  # S §11.5: degraus do refino (0..5)
    passos_por_iteracao: int = 24  # D10: num_steps_per_env do rl_cfg(); guarda do termo v3 (D4)


@dataclass(frozen=True)
class Niveis:  # S §11.2
    pegar: tuple = ((0.55, 1), (0.45, 2), (0.30, 3), (0.15, 4), (0.04, 5))  # S §11.2: (topo h_k; massa m_k em kg) por nível 0..4
    botar: tuple = (0.55, 0.45, 0.30, 0.15, 0.04)  # S §11.2: topo por nível do BOTAR (m)
    topo_min_c: float = 0.30  # S §7; fatia C: o topo sorteado nunca fica abaixo disto (m), com o PEGAR no nível 4


@dataclass(frozen=True)
class Terreno:  # S §11.2, §12; PD-20
    linhas: int = 13  # PD-20; terrains/config.py: linhas do gerador de terreno
    tamanho: tuple = (8.0, 8.0)  # PD-20; terrains/config.py: tamanho de cada célula (m)
    # (tipo, teto em m): degrau, rampa e obstáculo (S §11.2)
    tetos_andar: tuple = (("degrau", 0.20), ("rampa", 0.36), ("obstaculo", 0.15))
    # D1; ordem das colunas do gerador (o eventos lê daqui, sem importar a cena)
    colunas: tuple = ("plano", "escada", "escada_inv", "rampa", "rampa_inv", "escada_aleatoria",
                      "irregular", "obstaculos", "caixas")
    # (coluna, linha máxima de CARREGAR): PD-20; uma coluna tem plataforma de 2 m se e só se está aqui.
    # P3 (30/10): a escada aleatória sobe só até a linha 6 — o degrau dela é U(0; 0,20)·(0,5 + 0,5·t)
    # (primitive_terrains.py:1061), e na linha 9 passava de 0,15 m (até 0,175 m); na 6, 0,15 m (E §2).
    linha_max_carregar: tuple = (("escada", 9), ("escada_inv", 9), ("rampa", 9), ("rampa_inv", 9),
                                 ("escada_aleatoria", 6), ("obstaculos", 8), ("caixas", 7))
    plataforma: float = 2.0  # S §12: plataforma plana em volta da laje (m)
    laje_do_centro: float = 0.75  # G1; S §12: o centro da laje a 0,75 m do centro da plataforma (m)
    # D1 (29/10: sai da cena.py): border_width do gerador de terreno (m)
    borda: float = 20.0


@dataclass(frozen=True)
class Estimador:  # I §2.3; S §13; PD-11, PD-18
    taxa_hz: float = 30.0  # I §2.3: taxa da leitura da caixa (Hz)
    latencia_s: float = 0.033  # I §2.3; lidar_sim/sensores/tipos.py: atraso da leitura (s)
    idade_max: float = 2.0  # I §2.3: idade da leitura cortada (s)
    fov_h_graus: float = 69.0  # PD-11: campo de visão horizontal da D435i (graus)
    fov_v_graus: float = 42.0  # PD-11: campo de visão vertical da D435i (graus)
    alcance: tuple = (0.28, 3.0)  # PD-11: alcance útil do depth (m); mínimo 0,28
    # (início, refino). ⚠ sigma_pos: início em metros, refino em FRAÇÃO da distância (2%)
    vies_meia: tuple = (0.005, 0.02)  # S §13 (dono, 15/10): viés da meia aresta (m)
    vies_alvo_botar_xy: tuple = (0.01, 0.05)  # S §13 (dono, 15/10): viés do alvo do BOTAR no plano (m)
    vies_alvo_botar_z: tuple = (0.005, 0.03)  # S §13 (dono, 15/10): viés do alvo do BOTAR em z (m)
    ori_persistente: tuple = (True, True)  # S §13 (dono, 15/10): viés de orientação persiste
    pontos_visiveis_min: int = 3  # PD-11: pontos dos 9 necessários para ver a caixa
    sigma_pos: tuple = (0.005, 0.02)  # PD-18; S §13: σ da posição (m; refino: 2% da distância)
    sigma_ori_graus: tuple = (1.0, 5.3)  # PD-18; S §13: σ da orientação (graus; refino: mediana 5,3°)
    perda: tuple = (0.005, 0.022)  # PD-18; S §13: probabilidade de perda da leitura (refino: 2,2%)
    deriva_odo: tuple = (0.002, 0.0075)  # PD-18; S §13: deriva da odometria (m/√m)
    deriva_rumo: tuple = (0.002, 0.005)  # PD-18: deriva do rumo (rad/√m)
    deriva_pos_giro: tuple = (0.004, 0.008)  # PD-18 (dono, 28/10): deriva de posição no giro (m/√rad)
    deriva_rumo_giro: tuple = (0.0035, 0.014)  # PD-18 (dono, 28/10): deriva de rumo no giro (rad/√rad)
    d435_pos: tuple = (0.0576235, 0.01753, 0.42987)  # lidar_sim/sensores/montagem.py: posição da D435i (m)
    d435_pitch: float = 0.8307767  # lidar_sim/sensores/montagem.py: pitch da D435i (rad)


@dataclass(frozen=True)
class Percepcao:  # I §2.4; PERC §2; PD-5..PD-10, PD-42
    ligada: bool = True  # S §4: PERCEPCAO_LIDAR
    celula: float = 0.10  # I §2.4: tamanho da célula do mapa (m)
    n_recorte: int = 21  # I §2.4: células do recorte por lado (21 × 21 = 2 × 2 m)
    n_interno: int = 51  # I §2.4: células do mapa interno por lado (5 × 5 m, centrado)
    altura_clip: float = 1.0  # I §2.4: altura cortada em ±1 m
    raio_min: float = 0.85  # PD-10: raio do corpo a excluir (m); braço + 5 cm
    raios_por_passo: int = 205  # PD-5: raios do lidar simplificado por passo
    alcance: tuple = (0.1, 6.0)  # PD-5: alcance do lidar (m)
    ruido_alcance: float = 0.02  # PERC §2: ruído de alcance (m)
    ruido_angulo_graus: float = 0.15  # PERC §2: ruído de ângulo (graus)
    perda: float = 0.01  # PERC §2: fração de raios perdidos
    ruido_incl_graus: float = 0.5  # PERC §2: ruído de inclinação do IMU (graus)
    sigma_ref: float = 0.05  # PD-6: σ de referência da confiança (m); I §2.4
    q_desloc: float = 4e-4  # PD-7: ruído de processo por deslocamento
    q_giro: float = 4e-4  # PD-7: ruído de processo por giro
    q_tempo: float = 2.5e-5  # PD-7: ruído de processo por tempo
    sigma_fixo: float = 0.02  # PD-42: piso de σ da medida (m)
    margem_corpo: float = 0.03  # PD-10: margem do filtro do corpo (m)
    mid360_pos: tuple = (0.0002835, 0.00003, 0.428434)  # PERC §2; montagem.py: posição do MID-360 (m)
    mid360_quat: tuple = (0.0, 0.99967, 0.0, -0.02556)  # PERC §2; montagem.py: quaternion do MID-360
    limiar_visto: float = 0.5  # D8: célula do recorte conta como vista com confiança acima disto


@dataclass(frozen=True)
class DR:  # S §12, DR §5; PD-24; §6.8 (quando: startup, reset, intervalo, portão, refino)
    atrito_pe: tuple = (0.3, 1.0)  # §6.8 startup: atrito do pé, compartilhado entre as geoms do pé
    massa_tronco: tuple = (-1.0, 3.0)  # §6.8 startup: massa somada ao torso_link (kg)
    vies_encoder: float = 0.015  # §6.8 startup: viés de encoder (rad)
    atraso_inicio: tuple = (0, 1)  # §6.8 reset: atraso da ação, propriocepção e mapa (passos)
    atraso_portao: tuple = (0, 2)  # §6.8 PEGAR no portão do nível 0 (passos); PD-28
    atraso_refino: tuple = (0, 3)  # §6.8 refino: atraso 0–2 → 0–3 (passos); PD-25
    degrau_atraso_refino: int = 3  # §6.8; PD-41: o atraso global vai a atraso_refino quando o degrau global chega aqui
    # §6.5 (29/10: sai do observacoes.py): ruído do ator, ± antes da escala; a capacidade do atraso é atraso_refino[1]
    ruido_obs: tuple = (("giro_base", 0.2), ("gravidade", 0.05), ("juntas", 0.01), ("vel_juntas", 1.5))
    empurrao: float = 0.5  # §6.8 intervalo: empurrão em x e y (m/s, ±)
    periodo_empurrao_s: float = 5.0  # §6.8 intervalo: a cada 5 s
    vies_imu_giro: float = 0.05  # §6.8 ANDAR no portão: viés do giro da base (rad/s)
    vies_imu_incl_graus: float = 1.0  # §6.8 ANDAR no portão: inclinação fixa do IMU (graus)
    atrito_pad: tuple = (0.6, 1.0)  # §6.8 PEGAR no portão; PD-46: atrito dos pads de palma
    atrito_laje: tuple = (0.3, 1.0)  # §6.8 PEGAR no portão; PD-46: atrito da laje
    passo_mu_pad: float = 0.025  # §6.8: a borda de baixo do μ dos pads desce 0,025 por nível
    mu_pad_min: float = 0.5  # §6.8: piso do μ dos pads
    com_caixa: bool = False  # §6.8; PD-38: liga o dr.body_com_offset (nasce desligada)
    com_caixa_frac0: tuple = (0.2, 0.2)  # §6.8 PEGAR no portão: CoM da caixa (xy; z; × meia aresta)
    com_caixa_frac_max: tuple = (0.3, 0.2)  # §6.8 níveis seguintes: CoM da caixa (xy; z; × meia aresta)
    forca_caixa_N: tuple = (5.0, 20.0)  # §6.8 CARREGAR: impulso na caixa (N)
    forca_caixa_dur_s: tuple = (0.1, 0.3)  # §6.8 CARREGAR: duração do impulso (s)
    forca_caixa_periodo_s: tuple = (3.0, 8.0)  # §6.8 CARREGAR: intervalo entre impulsos (s)
    rfi: tuple = (0.05, 0.1)  # §6.8: RFI × limite de torque (início; refino)
    pd: tuple = (0.9, 1.1)  # §6.8 refino: faixa do ganho do PD no degrau 5 (degrau 0 = 1,0)
    forca_motor: tuple = (0.9, 1.1)  # §6.8 refino: faixa da força do motor no degrau 5
    forca_motor_bracos: tuple = (0.8, 1.1)  # §6.8 refino: faixa da força do motor dos braços
    solref: tuple = (0.015, 0.03)  # §6.8 refino: solref[0] do chão e da caixa (início; refino)
    solref_base: float = 0.02  # §6.8 refino: solref[0] no degrau 0; PD-41


@dataclass(frozen=True)
class Refino:  # S §11.5; §6.9
    # (regra, tarefas, seção do Knobs, campo): o campo é um par (início; refino) ou (início; fim)
    regras: tuple = (
        ("laje_joelho_N", ("PEGAR", "BOTAR"), "recompensa", "laje_joelho_N"),
        ("incl_lim", ("PEGAR", "CARREGAR", "BOTAR"), "alvos", ("incl_inicio_graus", "incl_fim_graus")),
        ("chegada", ("BOTAR",), "terminacao", "chegada"),
        ("sigma_pouso", ("BOTAR",), "recompensa", "sigma_pouso"),
    )
    fator_por_degrau: tuple = ("aperto", "sacudida")  # §6.9: peso × d/5; campos de Recompensa
    tarefas_fator: tuple = ("PEGAR", "CARREGAR", "BOTAR")  # §6.9: tarefas com o fator
    degrau_batente: int = 1  # §6.9: batente e hip_yaw > 1 rad ligados a partir deste degrau


@dataclass(frozen=True)
class Knobs:  # agregador; g1_limpo/knobs.py:1608-1628
    contrato: Contrato = Contrato()
    cena: Cena = Cena()
    comando: Comando = Comando()
    cadeia: Cadeia = Cadeia()
    detector: Detector = Detector()
    alvos: Alvos = Alvos()
    recompensa: Recompensa = Recompensa()
    terminacao: Terminacao = Terminacao()
    balanceador: Balanceador = Balanceador()
    niveis: Niveis = Niveis()
    terreno: Terreno = Terreno()
    estimador: Estimador = Estimador()
    percepcao: Percepcao = Percepcao()
    dr: DR = DR()
    refino: Refino = Refino()


ATIVO = Knobs()


def dim_ator(c: Contrato = Contrato()) -> int:  # I §2.1: 1 447 números por passo
    # canais × histórico; o bloco sem histórico (mapa, 0) entra uma vez
    return sum(canais * max(hist, 1) for _nome, canais, _escala, hist in c.blocos_ator)


def caminho(rel: str) -> Path:  # relativo à raiz do repo: g1_v3/knobs.py → parents[1]
    return Path(__file__).resolve().parents[1] / rel


def q_de_pe(c: Contrato = Contrato()) -> np.ndarray:  # I §2.2: (29,); A3 gera referencias.npz
    with np.load(caminho(c.referencias)) as npz:
        return np.array(npz["q_de_pe"])
