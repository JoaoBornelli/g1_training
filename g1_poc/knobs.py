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
    # ⚠ 21/08: VOLTOU a 1,0. O 0,8 cortava 20% da autoridade de junta contra o
    # `G1_ACTION_SCALE` do fabricante, e autoridade é exatamente o que uma fase de
    # balanço precisa. Nenhuma medida justificava o corte.
    escala_acao_mult: float = 1.0

    # --- reset da base ---
    # A pose de reset é POR FORMA desde 21/08, e o motivo é o defeito central do
    # bloco 2.
    #
    # O fabricante sorteia `yaw: (-3.14, 3.14)` — o CÍRCULO INTEIRO
    # (`velocity_env_cfg.py:209`). Com `heading_command=True` e
    # `rel_heading = 0,30`, o ωz comandado vem do erro de rumo. Com ±3,14 esse erro
    # cobre toda a faixa e o robô TEM de aprender a girar desde a iteração 0.
    #
    # Nós sorteávamos ±0,2 rad, porque a mobília tem pose ABSOLUTA e o robô precisa
    # nascer olhando para ela. Consequência: o erro de rumo era sempre minúsculo, o
    # `track_angular_velocity` era satisfeito sem fazer nada, e o canal de yaw
    # nunca foi exercitado. Quando a política derivou para o giro, ela não tinha
    # autoridade nenhuma para sair.
    #
    # Na LOCOMOÇÃO a mobília sobe 5 m (`afasta_cena`). Não existe nada com que
    # alinhar o rumo. Portanto ali o ±3,14 é de graça, e é a receita do fabricante.
    reset_base_pose_manipula: dict = field(default_factory=lambda: {
        "x": (-0.10, 0.00), "y": (-0.10, 0.10), "z": (0.01, 0.05), "yaw": (-0.2, 0.2),
    })
    reset_base_pose_loco: dict = field(default_factory=lambda: {
        "x": (-0.50, 0.50), "y": (-0.50, 0.50), "z": (0.01, 0.05),
        "yaw": (-3.14159, 3.14159),
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

    # folga entre o topo NOVO da prateleira e o fundo da caixa segurada, no
    # instante em que o `pegar` fecha na cadeia `pegar`->`botar` (§7.3).
    # ⚠ A §7.3 e a §10.1 se CONTRADIZEM: a §7.3 garante segurança dizendo "o topo
    # novo fica no máximo em 0,55 m", e a §10.1 manda sortear a colocação em
    # 0,30-0,80 m. Com a caixa segurada a 0,82 m o fundo dela está em 0,72 m, e uma
    # laje em 0,80 m nasceria DENTRO da caixa. O teto efetivo é
    # `fundo_da_caixa - botar_folga_laje`, resolvido por env. Consumido em MACRO 2.
    botar_folga_laje: float = 0.05


@dataclass
class Celulas:
    """§10.1 — a célula que cada nível seleciona. Sete níveis, de 0 a 6.

    Três regras da tabela, e elas explicam por que só o PISO varia:

    - o TETO do topo é 0,55 m em todo nível (`Cena.prateleira_topo_teto`). O robô
      continua treinando a altura que domina.
    - o PISO da carga é 1 kg em todo nível (`Cena.caixa_massa`). Mesmo motivo.
    - o nível ACRESCENTA cadeias. Ele não substitui cadeias.

    ⚠ O nível 6 da §10.1 pede rotação no eixo HORIZONTAL, que exige tombar a caixa.
    É o Risco 1 da §19, e o G1 não tem mão. A célula do 6 fica igual à do 5, e os
    critérios de aceite da §0 não pedem o 6.

    ⚠ `topo_min[4:] = 0,04` é a MESMA laje de `Cena.prateleira_topo_piso = 0,04` —
    dois knobs, um número físico. Quem mudar um tem de mudar o outro.
    """
    topo_min: tuple[float, ...] = (0.55, 0.45, 0.30, 0.15, 0.04, 0.04, 0.04)
    carga_max: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0)
    ang_max_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 45.0, 90.0, 180.0, 180.0)
    # ⚠ O jitter x da caixa também é da célula (auditoria de travas, 20/08): com o
    # topo a 0,04 m, poses de pega só existem até x relativo ≈ 0,45 — com o jitter
    # de 0,20 fixo, 60% dos episódios do nível 4 exigiriam um passo à frente, que o
    # twist zerado cobra (−0,44/s) e nenhum termo de marcha paga.
    jitter_x_max: tuple[float, ...] = (0.20, 0.20, 0.20, 0.15, 0.08, 0.08, 0.08)
    # Fração de cada cadeia, na ordem
    #   (`pegar`, `reorientar`->`pegar`, `pegar`->`carregar`, `pegar`->`botar`).
    # Somam 1,0 em cada nível. ⚠ Só a máquina de elo consome isto (MACRO 2); em
    # MACRO 1 o campo fica declarado e não lido — é a tabela da §10.1 inteira, num
    # lugar só.
    cadeias: tuple[tuple[float, float, float, float], ...] = (
        (1.00, 0.00, 0.00, 0.00),
        (1.00, 0.00, 0.00, 0.00),
        (1.00, 0.00, 0.00, 0.00),
        (0.50, 0.50, 0.00, 0.00),
        (0.40, 0.25, 0.35, 0.00),
        (0.30, 0.20, 0.25, 0.25),
        (0.30, 0.20, 0.25, 0.25),
    )


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
    # ⚠ FIXO no valor do fabricante. O cronograma que o levava a −0,25 saiu em
    # 21/08: a medida da it 488 mostrou que a −0,10 e σ 0,54 o termo já cobra
    # −1,49/s, que é o PISO de ruído (0,10 × 2σ² × 29 = 1,69/s). Multiplicar por
    # 2,5 taxaria EXPLORAÇÃO, e não tremor.
    action_rate: float = -0.10
    # ⚠ 21/08: VOLTOU ao valor do fabricante (−1,0). O −10,0 vinha de
    # `manipulation`/`tracking` e é dez vezes o `velocity_env_cfg.py:317`. Andar
    # precisa de amplitude no quadril e no joelho; uma penalidade forte de limite
    # empurra as juntas para o meio da faixa e ACHATA o balanço. O
    # `peak_height_mean` entre 0,007 e 0,023 é consistente com isso.
    dof_pos_limits: float = -1.0
    self_collisions: float = -1.0

    # --- tarefa ---
    staged: float = 3.0
    # ⚠ `reaching_std` virou o PISO do σ, não o σ (20/08). O σ efetivo é
    # `max(reaching_std, distância inicial palma→face)`, por env, recalculado no
    # começo do elo — a MESMA correção que a §8.2 fez no `bringing`. Medido: com σ
    # fixo de 0,20 o gradiente de aproximação cai 1391× entre a prateleira a 0,55
    # (2,64/m) e a 0,04 (0,0019/m) — os níveis 3+ viravam sorte.
    reaching_std: float = 0.20
    bringing_std_piso: float = 0.10      # σ variável: max(piso, distância comandada)
    precise_pos: float = 2.0
    precise_pos_std: float = 0.05
    precise_ori: float = 1.0
    # σ variável do `precise_ori` (mesmo idioma do bringing/reaching): piso 0,40 rad,
    # teto = Δθ inicial do elo. Com σ fixo, 90° dá 2,0e-7 — o `reorientar` dos
    # níveis 4+ era sorte.
    precise_ori_std: float = 0.40
    squeeze: float = 1.0
    squeeze_mu: float = 0.8              # μ pessimista da faixa de DR
    # a PONTE do platô do grasp (19/08). O `squeeze` satura em 6× F_ref e para de
    # guiar; o apoio da prateleira é contínuo e cai antes de a caixa se mover.
    unload: float = 2.0                  # igual ao `precise_pos`: é sinal de TAREFA
    unload_tol_queda: float = 0.03       # 3 cm abaixo do repouso já é queda

    # §8.2.3 — a rampa da pelve, ligada em 20/08.
    # A condição 3 do fecho do `pegar` exige pelve >= 0,65 m e NADA pagava por ela.
    # ⚠ A justificativa correta (auditoria 20/08): quem precifica a pelve é só a
    # `pose`, a ~0,73/m (o default é o KNEES_BENT_KEYFRAME, pelve 0,76). O
    # `precise_pos` é INDIFERENTE à pelve abaixo do alvo e CONTRÁRIO acima
    # (−16,2/m no ponto de fecho) — não "paga por agachar", como uma versão
    # anterior deste pacote afirmou.
    #
    # A rampa tem DUAS partes, e cada uma fecha um buraco medido:
    #   longa (0,20→0,65): sem ela o termo é MORTO em 33% das pegas do nível 4
    #     (pelve na pega chega a 0,267 m) e a zona 0,20-0,45 não tem gradiente;
    #   fina (0,57→0,65): sem ela a inclinação é 5/m contra os −16,2/m do
    #     `precise_pos` no fecho — o robô perderia recompensa ao subir os últimos
    #     centímetros com os braços rígidos.
    # Com peso 2,0: 2,2/m na zona longa e 14,7/m na fina.
    postura_ereta: float = 2.0
    postura_ereta_rampa: float = 0.45        # a parte longa: 0,20 -> 0,65
    postura_ereta_rampa_fina: float = 0.08   # a parte fina : 0,57 -> 0,65
    # ⚠ O gate de DESCARGA (F_apoio < frac·m·g) é anti-hack medido: sem ele,
    # encostar as palmas e ficar de pé com a caixa APOIADA paga a rampa inteira —
    # +2,0/s por ficar exatamente no platô que o bloco 1 mediu.
    postura_ereta_frac_descarga: float = 0.2
    # §8.2.4 — a rampa da sustentação, ligada em 20/08.
    # O fecho exige 1,0 s ininterrupto e NENHUM termo diferencia 0,98 s de 0,00 s.
    # Medido: o push era o único fator que degradava o sucesso, exatamente porque
    # quebra o cronômetro. Esta é a rampa na coordenada TEMPO-NA-CONDIÇÃO.
    sustentacao: float = 0.5

    # §8.2.5 — `load`, o espelho do `unload`, SÓ no elo `botar` (20/08).
    # O fecho do `botar` exige F_apoio >= 0,8·m·g, e os termos de segurar apontam
    # todos contra soltar: medido, satisfazer a 3ª condição custava −3,0/s e pagava
    # ZERO — o `botar` fecharia por sorte. `load = clamp(F_apoio/m·g)` é a mesma
    # grandeza contínua do `unload`, invertida, gateada por "perto do alvo".
    load: float = 2.0
    # o gate de posição do `load`: 2× o raio de sucesso. Sem ele, LARGAR a caixa em
    # qualquer lugar do tampo pagaria o máximo.
    load_raio_mult: float = 2.0

    joint_vel_hinge: float = -0.01       # cronograma o leva a -1.0
    # ⚠ O `joint_vel_max = 0,5` único, no corpo todo e nas duas formas, SAIU em
    # 21/08. Ele custava −2,77/s no bloco 1, e a assinatura era
    # `peak_height_mean = 0,0042` — o pé subia 4 mm, portanto não havia passo. A
    # tarefa `velocity` do mjlab não tem este termo, e a skill Lift o tinha só nos
    # braços (`arm_vel`, escopo `.*(shoulder|elbow|wrist).*`).
    #
    # Agora o teto é POR JUNTA e o termo vale só na MANIPULAÇÃO. Na locomoção ele é
    # zero — exatamente o que o fabricante faz.
    #
    # A regra dos números: o plano sagital da perna precisa de velocidade para
    # agachar e para levantar, portanto fica largo. O braço tem de se mover de forma
    # CONTROLADA, portanto fica apertado. As juntas laterais não têm por que se
    # mover rápido em nenhuma das duas coisas.
    #
    # ⚠ A cobertura tem de ser TOTAL e sem sobreposição: 29 juntas, e o
    # `resolve_matching_names` recusa padrão que não casa e nome casado duas vezes.
    #   perna 12 · cintura 3 · braço 14 = 29
    joint_vel_max_manipulando: dict = field(default_factory=lambda: {
        r".*knee.*": 6.0,          # agachar fundo e levantar
        r".*hip_pitch.*": 6.0,     # idem
        r".*ankle_pitch.*": 6.0,   # acompanha o agachamento
        r".*hip_roll.*": 4.0,      # equilíbrio lateral
        r".*hip_yaw.*": 4.0,
        r".*ankle_roll.*": 4.0,
        r".*waist.*": 3.0,         # o tronco inclina e gira, mas sem safanão
        r".*shoulder.*": 2.0,      # o braço é a tarefa, e tem de ser CONTROLADO
        r".*elbow.*": 2.0,
        r".*wrist.*": 2.0,
    })

    # o alvo de cada palma é a face lateral da caixa (não o centro)
    lateral_offset: float | None = None   # None = caixa_meia_aresta[1]


@dataclass
class Postura:
    """§9 — o quarto regime do `variable_posture`.

    Os três dicionários do G1 ficam intocados. Este é o quarto, e ele responde à
    FORMA do episódio: `caixa_valida = 1` -> `std_manipulando`.

    Regra: as juntas do plano sagital abrem, e as laterais ficam apertadas.

    ⚠ Ele já respondeu à DEMANDA da caixa (`peso_dist`, `peso_ang`, `limiar`), e
    aquilo era um PENHASCO, não um gradiente. Ver o docstring de
    `postura.postura_manipulacao`. Quem levanta o robô agora é o termo
    `postura_ereta` (§8.2.3), que é rampa.
    """
    running_threshold: float = 1.5

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
    # ±1,0 é o valor do fabricante (`velocity_env_cfg.py:188`)
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    # ±1,0 é o valor do fabricante (`velocity_env_cfg.py:189`). O ±0,3 daqui saiu
    # em 21/08: ele estreitava o comando lateral sem nenhum motivo medido, e a
    # marcha lateral é parte do que o `velocity` treina.
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)
    # metade dos envs PARADOS recebe giro no lugar; sem isto eles nunca giram
    frac_giro_no_standing: float = 0.5
    piso_giro_rad_s: float = 0.15
    # §11.2 — A JANELA DE ESPERA (21/08). Todo episódio começa PARADO, e o
    # objetivo chega depois.
    #
    # Motivo medido: o `rel_standing_envs = 0,10` do fabricante escolhe 10% dos
    # envs e dá comando zero pelo EPISÓDIO INTEIRO. Portanto 10% das amostras
    # treinam "parado para sempre", 90% treinam "mover desde o passo 0", e a
    # TRANSIÇÃO parado→mover não existe no dado. No bloco 2 o robô girava a
    # ~1,5 rad/s COM COMANDO ZERO — ele nunca foi obrigado a ficar parado.
    #
    # A janela põe "parado" em 100% dos episódios, e sempre no começo. Ficar de
    # pé deixa de ser caso especial e passa a ser a base.
    #
    # A duração é SORTEADA de propósito: uma janela fixa é aprendível como "conte
    # 25 passos e depois mova", o que funciona no treino e falha no deploy, onde a
    # ordem chega em tempo arbitrário. Sorteada, a política tem de LER o canal de
    # comando.
    espera_s: tuple[float, float] = (0.3, 1.0)
    # o limiar de "comando ativo" da razão de marcha (§10.4). É o `command_threshold`
    # que os cinco termos de marcha do fabricante já usam. Ele gateia por PASSO, e não
    # por episódio: "parado" é propriedade do resample (3 a 8 s), não do episódio.
    marcha_limiar_cmd: float = 0.05


@dataclass
class Episodio:
    duracao_s: float = 20.0
    # ⚠ Desde 20/08 isto é a FATIA DE TRANSIÇÕES alvo, e não o sorteio. O sorteio é
    # resolvido pelo controlador em `curriculo.sorteia_forma`, a partir das durações
    # MEDIDAS: f = alvo·T_manip / (T_loco·(1−alvo) + alvo·T_manip). Com o episódio
    # de andar morrendo em 24 passos, 30% de sorteio davam 1,06% dos dados — e o
    # bloco manual de "frac 0,85" que consertava isso era exatamente a configuração
    # manual que o usuário vetou. O controlador despeja episódios de andar enquanto
    # eles são curtos e relaxa sozinho para ~0,30 quando a marcha amadurece.
    # ⚠ 21/08: isto virou o PISO INICIAL do balanço automático, e não mais a fatia
    # fixa. Com `auto_balanco = True` ele começa em 1,0 (locomoção pura, a caixa
    # não existe) e DESCE por competência até `alvo_loco_min`. Com
    # `auto_balanco = False` ele é a fatia fixa de antes — é o modo que o `play`
    # usa para pinar 0,0 (`--pegar`) ou 1,0 (`--andar`).
    frac_locomocao: float = 1.00
    # clamps do sorteio: nunca menos de 10% nem mais de 95% de locomoção
    frac_loco_min: float = 0.10
    frac_loco_max: float = 0.95
    forma_ema: float = 0.99

    # --- §10.4 — O BALANÇO AUTOMÁTICO DE FORMA (21/08) ---
    #
    # O problema medido no bloco 2: com 0,30 fixo desde a iteração 0, a locomoção
    # e a manipulação competiram por uma MLP só antes de existir marcha. A
    # `dur_loco_ema` subiu a 65 na it 260 e caiu a 11 na it 700, e NADA reagiu. O
    # `peak_height_mean` foi de 0,024 a 0,0069 no mesmo intervalo.
    #
    # O balanço não serve para CHEGAR na manipulação. Serve para NÃO PERDER a
    # marcha quando ela chegar. Por isso ele é assimétrico: lento para avançar,
    # rápido para defender.
    auto_balanco: bool = True
    alvo_loco_min: float = 0.30          # o destino: 30% de locomoção
    alvo_loco_max: float = 1.00          # o teto: locomoção pura
    alvo_passo: float = 0.02             # 1,00 -> 0,30 são 35 degraus
    alvo_iters_entre_degraus: int = 12   # >= 420 iterações na rampa inteira
    # ⚠ Carência antes do PRIMEIRO degrau. Sem ela o balanço desceria na iteração
    # 12: a `dur_loco_ema` nasce NEUTRA em 1000 passos (ver `sorteia_forma`) e o
    # limiar é 600, portanto o portão abre com dado que ainda não existe.
    alvo_iters_min: int = 200
    # ⚠ 24/08: o sinal do portão era `dur_loco_alvo = 600` — a DURAÇÃO do episódio de
    # locomoção — e ele media a coisa errada. Duração é sobrevivência, e ficar de pé
    # sobrevive: um robô imóvel marca 1000 passos, passava o limiar de 600 sem nunca
    # ter andado, e o balanço entregava a fatia até 0,30. O caminho de volta exigia
    # `dur < 480`, ou seja voltar a CAIR — "parou de andar" era invisível.
    #
    # O sinal agora é a RAZÃO DE MARCHA, produzida por `TwistPoc._update_metrics`:
    #
    #     razao = 1 − Σ‖v_cmd − v‖ / Σ‖v_cmd‖      (só nos passos de comando ativo)
    #
    # 0 é parado, 1 é rastreio perfeito. Ela é adimensional, portanto o limiar NÃO se
    # move quando o `commands_vel` alarga as faixas na iteração 5000.
    #
    # O 0,50 é do fabricante: o `terrain_levels_vel` rebaixa quem anda menos de
    # METADE da distância comandada (`velocity/mdp/curriculums.py:52-54`). Mesma
    # fração, ordem de integração trocada — metade da velocidade em vez de metade da
    # distância. Não é número chutado, que foi o defeito do `erro_giro_ema` de 21/08.
    razao_marcha_alvo: float = 0.50
    # histerese: desce com sinal >= limiar; sobe com sinal < 0,8·limiar
    alvo_desce_frac: float = 0.80
    # ⚠ O "segure e ande" antecipado (`frac_twist_livre`) SAIU em 21/08. Ele liberava
    # o twist em 30% dos envs de manipulação a partir do nível 3, para fechar um vão
    # de distribuição: antes do nível 4 nenhuma transição tem twist ≠ 0 com caixa
    # válida.
    #
    # O vão era real. A solução estava errada, e a geometria mostra por que:
    #
    #     borda perto da prateleira   x = 0,20 m
    #     robô no reset da manipulação x ≈ −0,05 m
    #     vão                          0,25 m
    #     vx comandado no nível 3      até 1,0 m/s
    #     tempo até o contato          ~0,25 s
    #
    # E o `contato_ilegal` dispara com 50 N na pelve, no tronco ou na coxa. Portanto
    # esses envs morriam de cara, contra a própria mesa.
    #
    # Duas falhas somadas: o gate não exigia `pegou`, portanto o twist liberava ANTES
    # de haver caixa na mão; e a mobília não saía. Isso é o oposto do `carregar` de
    # verdade, que só fica ativo DEPOIS do fecho do `pegar` e cuja transição sobe a
    # prateleira 5 m (§7.3). O atalho passava por cima do único mecanismo que torna
    # andar-carregando possível.
    #
    # Andar com a caixa aparece na cadeia `pegar` -> `carregar`, e em nenhum outro
    # lugar. O robô pega, a mesa sai, e ele anda livre.


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


@dataclass
class Cronograma:
    """⚠ 21/08: este bloco encolheu. Saíram `action_rate` (o fabricante roda −0,10
    fixo), os quatro knobs do gate do twist (voltou o `commands_vel` por passo
    global) e o par `freio_ar_*` (o sinal `duracao_loco` deixou de ter produtor).

    Um knob sem consumidor é pior que nenhum: alguém religa `freio_ar_sinal` e lê
    zero para sempre. Sobraram `locomocao`, `hinge` e os quatro `freio_hinge_*`.
    """
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
        # ⚠ 21/08: esta é a tabela do fabricante, copiada de
        # `velocity_env_cfg.py:403-405`, byte por byte. Os degraus em 8000 e 12000
        # eram meus e saíram. O mecanismo também voltou: `mdp.commands_vel`, por
        # PASSO GLOBAL, sem gate por competência.
        #
        # ⚠ Consequência a declarar: com `max_iterations = 3000` o degrau 1 (5000)
        # nunca dispara. Isso é o fabricante, e não um defeito — ele treina muito
        # mais que 3000 iterações. Para ver o currículo de comando andar, o bloco
        # precisa de >= 5000 iterações.
        {"step": 0,           "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
        {"step": 5000 * 24,   "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
        {"step": 10000 * 24,  "lin_vel_x": (-2.0, 3.0)},
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


    freio_hinge_sinal: str = "nivel_medio"
    freio_hinge_alvo: float = 3.0        # metade da tabela de células dominada
    freio_desce_frac: float = 0.8        # histerese, igual ao twist
    freio_iters_entre_degraus: int = 12
    # ⚠ `poc_estagio_twist` e a EMA NÃO vão para o checkpoint (o runner só salva
    # `common_step_counter`). Depois de um resume o gate recomeça pessimista
    # (estágio 0, EMA 0) e se recalibra em ~3τ ≈ 12 iterações. Declarado: é o
    # comportamento seguro, não um bug.


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
    celulas: Celulas = field(default_factory=Celulas)
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
