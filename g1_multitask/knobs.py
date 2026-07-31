"""Todos os números tunáveis do multi-tarefa, num lugar só.

Regra deste arquivo: **número sem justificativa é número copiado.** Cada campo
carrega a origem ao lado — `§14` quando vem da tabela de constantes do doc,
`medido` quando saiu de medição, `fabricante` quando é o valor testado do mjlab.

**Estado da calibração (Tarefa 12, rodada em 30/07):** `python g1_multitask/calibra.py`
passou por todos os kernels e por todas as 7 tarefas. Resultado: **um** número mudou
(`angulo_std_grosso_deg`, escala nova) e o resto ficou como estava, com a
justificativa numérica escrita ao lado. Os campos com `✅ CONFERIDO` foram
investigados e mantidos — "conferido, mantido" com o número na mão é resposta válida.

Três achados ficaram em **observação para o bloco 1**, não em mudança preventiva:
`box_shake`, `table_contact` e `arm_vel` superam o sinal de tarefa nas tarefas que
carregam a caixa — mas a medição é com ação nula, que é o pior caso possível, então
mudar peso agora seria calibrar contra um regime que não é o do treino.

Padrão igual ao `g1_training/skills/lift/knobs.py`: grupos de dataclass compostos
num `MultitaskKnobs`, e cada treino salvo é uma instância congelada em `configs/`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Scene:
    box_half: tuple[float, float, float] = (0.10, 0.10, 0.10)   # §14 — cubo de 20 cm
    box_mass: float = 1.0                                        # nível 1 do eixo de peso
    shelf_half_xy: float = 0.30                                  # §14
    shelf_half_z: float = 0.02                                   # §14 — fina = prateleira, não paredão
    shelf_top: float = 0.55                                      # nível 0 do eixo de altura
    box_xy: tuple[float, float] = (0.50, 0.0)
    table_xy: tuple[float, float] = (0.50, 0.0)

    afasta_distancia: float = 5.0
    """Quanto a prateleira (e a caixa) sobem nas tarefas que não as usam.

    ⚠️ **PARA CIMA, não para o lado.** Os envs ficam lado a lado em x e y com
    espaçamento menor que 5 m: deslocar em x punha a prateleira de cada env DENTRO do
    robô de um env vizinho e derrubava ele. Em z não há vizinho.

    Achado no `play` em 30/07: a prateleira ocupa **x de 0.20 a 0.80** com topo em
    0.55 m, que é altura de joelho. O destino do `andar` é
    `pos_robô + (d·cos(head), d·sin(head), 0)` com `d` até 2.0 m ⇒ com heading perto de
    zero **o robô anda direto contra ela**. Ela não faz parte da tarefa `andar`.

    **Por que 5 m e não 30.** O `object_pos_b` entra na obs e o normalizador dele
    APRENDE — 30 m envenena a estatística, que é a mesma classe de estrago do bug de
    índice do normalizador (§9b). O maior valor que a obs já vê é o destino do `andar`
    a 2.0 m, então 5 m é 2,5× isso: fora do alcance de um episódio e dentro da faixa
    que o normalizador aguenta."""

    box_jitter_x: tuple[float, float] = (0.0, 0.20)              # herdado do config ativo da Lift
    box_jitter_y: tuple[float, float] = (-0.18, 0.18)            # herdado do config ativo da Lift
    box_jitter_yaw_deg: float = 15.0
    """§14 — jitter de spawn de ±15°, GERAL, não filtro de tarefa.

    Não confundir com o eixo `giro` do `reorientar`: aquele é a rotação COMANDADA
    (alvo a N graus da orientação atual). Este é só a caixa nascer torta, e vale
    pra todas as tarefas. Consequência útil: com ±15° de jitter, trazer uma face
    pra frente nunca pede exatamente 90° — pede 75° a 105° — então os níveis
    pequenos do eixo de giro não são só andaime."""

    grupo_mobilia: int = 2
    """Grupo de geom da caixa e da prateleira.

    `common/box.py` não passa `group=` no `add_geom`, então as duas caem no grupo 0
    junto com o chão real. O `foot_height_scan` do fabricante tem
    `include_geom_groups=(0,)` -> hoje ele lê a prateleira COMO CHÃO. Tirar a
    mobília do grupo 0 conserta isso sem tocar em `box.py`."""

    level_jitter_z: float = 0.02        # §14 — ±0.02 m em cima da altura sorteada


@dataclass
class Command:
    v_max: float = 1.0
    """§14 / fabricante — borda do range stage-0 (`lin_vel_x/y = ±1.0`)."""
    w_max: float = 0.5
    """§14 / fabricante — `ang_vel_z = ±0.5` stage-0."""
    heading_gain: float = 0.5
    """§14 / fabricante — `heading_control_stiffness`."""

    d_morto_andar: float = 0.25          # §14 — é o R de chegada do `andar`
    d_morto_manipula: float = 0.30       # §14 — `alvo_peito_b[0]` + 0.10 (derivado)
    d_freio_extra: float = 0.50
    """§14 — `d_freio − d_morto`. ~2 passos pra desacelerar de 1.0 m/s (derivado)."""

    alvo_peito_b: tuple[float, float, float] = (0.20, 0.00, 0.15)
    """§14 — alvo da caixa em frame da BASE. Constante: é por isso que o `pegar`
    tem `alvo_pos = 0` no vetor de comando — o peito não precisa ser transmitido."""

    atraso_gatilho_s: tuple[float, float] = (0.0, 2.0)   # §14 — U(0, 2 s)


@dataclass
class Reward:
    # --- bloco 1: invariantes, sempre ligados (§6b/B e §6b/C) ---
    track_linear_velocity: float = 2.0
    track_angular_velocity: float = 2.0
    upright: float = 1.0
    postura: float = 0.5
    foot_clearance: float = -2.0
    foot_swing_height: float = -0.25
    soft_landing_feet: float = -1e-5
    self_collisions: float = -1.0
    dof_pos_limits: float = -1.0
    action_rate_l2: float = -0.25
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    arm_vel: float = -0.002
    """§14. ✅ **CONFERIDO E MANTIDO** (T12, 30/07), com uma ressalva de projeto.

    A calibração o flagou superando o sinal de tarefa em `andar c/ caixa` (−0.064
    contra +0.027). Artefato de ação nula outra vez: sem ação os braços chicoteiam
    enquanto o robô tomba, e `joint_vel_l2` é quadrático.

    A ressalva que fica registrada: nas 3 tarefas que **carregam**, o braço é
    ESTRUTURA, não gesto — ele precisa de velocidade pra corrigir a preensão quando a
    caixa escorrega. Punir velocidade de braço ali compete com a tarefa de um jeito
    que não compete no `andar` de mãos vazias, onde o alvo do termo (o "correr pra
    pegar") é justamente o que se quer punir.

    Não gateio agora: −0.002 é peso muito baixo, e gate a mais é complexidade que a
    medição ainda não justifica. Fica no radar do bloco 1 junto com o `box_shake`."""
    joint_acc: float = -2.5e-7
    feet_slip: float = -0.1

    # --- bloco 2: termos de tarefa, gateados pelo one-hot ---
    lift: float = 2.0                   # só no `pegar` — progresso de altura
    reaching: float = 1.0               # shaping; ANELA com catraca
    box_at_peito: float = 1.0           # grasp × kernel(caixa → peito)
    box_at_prateleira: float = 1.0      # kernel(caixa → prateleira), SEM grasp
    kernel_angulo: float = 1.0          # só no `reorientar`
    grasp: float = 0.5                  # bônus de toque, só no `pegar`
    hold_still: float = 0.5

    # --- bloco 3: anti-hacks ---
    com_balance: float = -3.0           # OFF no `andar` (gate explícito)
    table_contact: float = -1.5
    """§14. ✅ **CONFERIDO E MANTIDO** (T12, 30/07).

    A calibração o flagou superando o sinal de tarefa em `parado c/ caixa` (−0.20
    contra +0.023) e `andar c/ caixa` (−0.30 contra +0.027). É **artefato da medição
    com ação nula**: sem ação o robô tomba pra frente e encosta o corpo na mesa, o
    que é exatamente o que este termo existe pra punir. O numerador está certo; o
    denominador (sinal de tarefa) é que colapsou porque a caixa já escorregou.

    Confirma que o termo funciona, não que o peso está errado. O log dele no bloco 1
    diz se o robô treinado ainda encosta."""
    back_penalty: float = -0.5
    box_shake: float = -0.15
    """§14. OFF no `reorientar` por gate explícito. ⏸️ **EM OBSERVAÇÃO no bloco 1**
    — é o achado nº1 da calibração.

    A calibração mediu, nas TRÊS tarefas que nascem segurando:

    | tarefa          | box_shake | sinal de tarefa | razão |
    |-----------------|----------:|----------------:|------:|
    | `botar`         |    −1.25  |          +0.637 |    2× |
    | `parado c/ caixa`|   −1.43  |          +0.023 |   62× |
    | `andar c/ caixa`|    −1.34  |          +0.027 |   50× |

    Três coisas se somam, e é a combinação que incomoda:
    1. o termo é **quadrático** em `‖ω_caixa‖`, então caixa tombando custa caríssimo;
    2. o ator **não observa** velocidade angular da caixa (ele vê `box_rot_b`, e não
       há histórico pra derivar) — a §6b já registra isso como aceito;
    3. nas tarefas que carregam, deixar a caixa cair **já** custa o termo de tarefa
       inteiro; o `box_shake` cobra de novo pela mesma falha.

    ⚠️ **Ressalva de método:** a medição é com ação NULA, e aí os braços vão moles e
    a caixa tomba — é o pior caso possível de ω, não o regime treinado. As razões de
    50-62× são infladas porque o sinal de tarefa também colapsa (a caixa já saiu do
    peito). Limite superior, não previsão.

    **Não mexer preventivamente.** Olhar `Contrib/<tarefa>/box_shake` no relatório do
    bloco 1. Se com política treinada ainda superar o sinal, o conserto é peso —
    Categoria A, grátis — ou o mesmo gate que o `reorientar` já tem."""
    soft_landing_table: float = -1e-4
    hip_deviation: float = 0.0          # §14 — OFF
    joint_torque_pen: float = 0.0       # §14 — OFF (briga com o payload)

    # --- kernels ---
    sustain_std: float = 0.05
    """§14, herdado da Lift. ✅ **CONFERIDO E MANTIDO** (T12, 30/07).

    A curva: na tolerância de sucesso de 0.10 m este kernel vale **0.0183** — 2% do
    termo. E é escala ÚNICA, sem uma grossa cobrindo a cauda como o `orienta_face` e
    o `reaching` têm. Isso parecia problema, e não é.

    Por que fica: **nenhuma tarefa precisa fechar o vão de 0.15 m → 0.10 m usando só
    este termo.**
      - `pegar` tem que aproximar de longe, mas ali existem `lift` (+2.0, progresso
        de altura) e `reaching` (+1.0, duas escalas) cobrindo toda a aproximação;
      - `parado c/ caixa` e `andar c/ caixa` **começam NO alvo** (a caixa nasce no
        peito, medido: kernel de 0.60 a 0.78 no passo 1), então o gradiente forte
        está exatamente onde eles vivem — segurar. Eles nunca precisam subir de 0.15.

    E o std apertado é o que mata o hack de "segurar a caixa vagamente perto do peito
    sem precisão", que é o motivo pelo qual a Lift o escolheu apertado.

    ⚠️ Se algum dia uma tarefa tiver que ATINGIR o alvo do peito partindo de longe
    **sem** `lift`/`reaching` ligados, este número volta pra mesa."""

    std_coarse: float = 1.0             # §14, herdado da Lift
    std_fine: float = 0.25              # §14, herdado da Lift
    upright_std: float = 0.1
    """§14, herdado da Lift — o único std do desenho com justificativa numérica
    escrita: fator 0.86 a 10°, 0.55 a 20°, 0.26 a 30°, 0.06 a 45°. Demandante mas
    graduado. É o molde do que a Tarefa 12 tem que produzir pros outros."""

    com_margin: float = 0.08            # §14

    angulo_std_grosso_deg: float = 30.0
    """Escala GROSSA do kernel do `reorientar`. ✅ **ESCOLHIDO E VALIDADO** (T12, 30/07).

    É a única mudança de número que a calibração produziu — o resto foi "conferido,
    mantido". Não está no doc: é acréscimo meu, e o motivo é medido.

    Com escala ÚNICA de 5°, o nível 15° do eixo de giro (o PRIMEIRO nível, onde a
    tarefa começa) daria `exp(−(15/5)²) = 1.2e−4`. Gradiente nenhum até o robô já
    estar dentro de ~10° — ou seja o eixo de giro nasceria sem sinal de aprendizado.

    Com a grossa em 30° somada meio a meio com a fina:
      - no reset do nível 15°:            **0.389** (medido no env)
      - na tolerância de sucesso de 10°:   **0.457** (analítico)
      - a 45°:                              0.053
      - a 90°:                              ~0

    Mesmo desenho monotônico anti-vale do `reaching_reward` da Lift, pelo mesmo
    motivo: escala grossa mantém sinal de longe, fina premia a precisão final."""

    angulo_std_fino_deg: float = 5.0
    """Escala FINA — premia a precisão final. Dá ~0.14 na tolerância de 10°."""

    reorienta_xy_std: float = 0.05
    """§14 — casa a tolerância de 5 cm em xy do `reorientar`.

    Entra como FATOR multiplicativo do kernel de ângulo, nunca como penalidade por
    passo: 10 cm de deriva derrubam o termo a `exp(−0.10²/0.05²) = 0.018`, o que mata
    a pontuação sem castigar a manobra enquanto ela acontece."""

    botar_fracao_solta: float = 0.0
    """Quanto do termo do `botar` exige ter SOLTADO a caixa. **0.0 = kernel puro.**

    0.0 é o que a §4 do doc especifica, e o raciocínio dela fecha: "transportando, a
    caixa está longe da prateleira → reward baixa; aproximando e baixando → sobe;
    soltando → **continua alta**, porque não há fator de preensão". Não há vale
    porque a caixa **começa na mão, longe do alvo** — a condição de spawn
    "segurando" é que faz isso valer.

    Eu tinha posto 0.5 achando que o kernel puro criaria descasamento entre reward e
    sucesso (segurar no alvo pontuaria igual a soltar). O descasamento existe, mas é
    INDIFERENÇA, não incentivo contrário: segurar não paga mais que soltar. Deixo o
    knob pra a Tarefa 12 medir se a indiferença basta; o default segue o doc."""

    postura_std_parado: float = 0.05    # §6b — coluna `parado`
    postura_std_manipula: float = 0.5   # §6b — colunas `orientar`/`pegar`/`botar`

    postura_joints: tuple[str, ...] = (r".*(hip|knee|ankle|waist).*",)
    """Escopo da postura: perna + cintura, BRAÇOS LIVRES.

    Diferente da skill Stand, que usa `.*` (corpo todo). Aqui a mesma política
    manipula, então uma postura de corpo inteiro brigaria com a tarefa. O doc
    especifica o `std` por etapa mas não o escopo — esta é escolha do
    implementador, e o `posture` da Lift já usava exatamente este escopo."""


@dataclass
class Foundation:
    action_scale_mult: float = 0.8       # §14 — config ativo da Lift, movimento mais gentil


@dataclass
class DR:
    """Quais eventos de randomização de startup ligar (item 3e / §11b).

    Todos os três têm que estar ligados desde a 1ª iteração, não depois: eles mudam
    a DISTRIBUIÇÃO de treino, e a catraca do anelamento guarda o pico de competência.
    Ligar depois faria a catraca retirar a muleta num nível que o robô já não
    sustenta sob a distribuição nova."""

    foot_friction: bool = True           # dr.geom_friction — testado, OK em CPU e GPU
    encoder_bias: bool = True            # dr.encoder_bias  — testado, OK em CPU e GPU

    base_com: bool = False
    """❌ **DESLIGADO. `dr.body_com_offset` corrompe memória em CPU E EM GPU.**

    Item 0 do checklist, **resolvido em 30/07 — e a resposta foi desligar.**

    | backend | sintoma |
    |---|---|
    | CPU (warp) | core dump. 32 envs cai na construção; 4 envs roda 5 steps e dumpa no teardown |
    | GPU (T4, Kaggle) | `CUDA error: illegal memory access`, e `Warp CUDA error 700` em `wp_free_device_async` |

    Provado por A/B no mesmo processo-filho, 256 envs, `CUDA_LAUNCH_BLOCKING=1`: com o
    evento **cai**, sem o evento **sobrevive**. E derruba a task do PRÓPRIO FABRICANTE
    (`Mjlab-Velocity-Flat-Unitree-G1`) em CPU do mesmo jeito — não é interação com a
    nossa cena de 3 entidades.

    ⚠️ **O traceback mente.** O erro aparece em `curriculum.py::_medir`, no
    `valido.any()`, porque erro de CUDA é reportado de forma ASSÍNCRONA e aquela é a
    primeira chamada que força host-device sync. O evento roda em `mode="startup"`,
    ou seja antes do reset. Quem for reabrir isto: não caça o bug onde o traceback
    aponta.

    Terceira API de DR desta família a corromper heap neste projeto — depois do
    `dr.body_mass` (commit b46d730). Corolário que fica: **"o fabricante usa" não
    implica "funciona neste backend".**

    **O que se perde:** ±2.5 cm de randomização de CoM no `torso_link` (x/y ±0.025,
    z ±0.03). É gap real de sim-to-real, e fica em aberto — o truque do
    `write_external_wrench_to_sim` que resolveu o `dr.body_mass` não serve aqui,
    porque força constante não desloca centro de massa. `dr.pseudo_inertia` é
    candidato, e é igualmente não-testado.

    Religar só depois de o A/B acima passar numa versão nova do mjlab."""


@dataclass
class Push:
    """Escada de push — eixo GLOBAL, sem piso, o único com recuo de nível (§14).

    Um fator único multiplica os 6 componentes. Nível 3 é o `fator 1.00` do config
    ativo: `x/y ±0.6 · z ±0.4 · roll/pitch ±0.52 · yaw ±0.78 · 50 N por componente`.
    Do 3 pro 4 o fator não muda — só a DURAÇÃO da força alonga."""

    vel_x_full: tuple[float, float] = (-0.6, 0.6)       # fator 1.00
    vel_y_full: tuple[float, float] = (-0.6, 0.6)
    vel_z_full: tuple[float, float] = (-0.4, 0.4)
    roll_full: tuple[float, float] = (-0.52, 0.52)
    pitch_full: tuple[float, float] = (-0.52, 0.52)
    yaw_full: tuple[float, float] = (-0.78, 0.78)
    force_full: float = 50.0

    duracao_curta_s: tuple[float, float] = (0.3, 1.0)   # níveis 0-3
    duracao_longa_s: tuple[float, float] = (0.3, 3.0)   # nível 4
    cooldown_s: tuple[float, float] = (1.5, 3.0)        # §14 — medido
    intervalo_impulso_s: tuple[float, float] = (1.0, 3.0)


@dataclass
class Tolerancia:
    """Limiares de terminação e de sucesso (§14). Mudar um de SUCESSO é Categoria C —
    recomeçar do zero — porque a régua do currículo se move junto."""

    termina_ao_cair: bool = False
    """Se a queda encerra o episódio. **Desligado**, e isso é mudança de desenho.

    Ligado (o default do fabricante, e o que a §6b assumia) a queda encerra. Duas
    consequências ruins, as duas medidas:

    1. **A política nunca vê o estado caído**, então não pode aprender a levantar.
    2. **É a origem do vale de retorno negativo.** Terminação zera o valor futuro, então
       com reward por passo negativa MORRER CEDO RENDE MAIS. Visto nas duas runs de
       30/07: monolítica 57 -> 11 -> 164 -> 851 passos; residual 300 -> 150 -> 82 -> 39
       -> 935. As duas atravessaram, mas gastaram centenas de iterações no fundo.

    Desligado, o vale morre por construção: sem saída antecipada, "mais curto é melhor"
    deixa de existir. E ficar no chão custa o orçamento INTEIRO de positivos (~+3,5 de
    contrib de pé contra ~0 caído), então o retorno pune deitar com força.

    **O que autoriza desligar é medição, não teoria.** Com o BFM como base, levantar é
    alcançável: o `fumaca.py` nasce o robô caído (pitch 80°, pelve 0,32 m) e mede, com
    residual em ZERO —

        de-pe projetado no span    100% levantou    fim 0,780 m
        anti-sitonground           100%             fim 0,782 m
        move-ego-0-0               100%             fim 0,771 m

    O primeiro é o que a política de fato alcança pelos 20 coeficientes da `base_z`. Ela
    tem como descobrir. Numa política monolítica, sem o BFM, desligar isto seria caro —
    é a arquitetura residual que torna a mudança barata.

    Custo aceito: um env caído que não recupera roda até ~900 passos rendendo quase
    nada. O sinal de que isso está pesando é `upright` médio baixo com episódio cheio;
    se aparecer, a saída é terminação ATRASADA (encerra após ~3 s caído) em vez de
    nenhuma."""

    limite_queda_rad: float = math.radians(70.0)
    """Inclinação do torso que conta como queda. 70° = o `limit_angle` do fabricante.

    Com `termina_ao_cair = False` este número deixa de ser terminação e passa a ser SÓ o
    critério da flag `_nunca_caiu`, que o sucesso do `parado` lê. Mantido idêntico ao do
    fabricante para o sucesso significar a mesma coisa que antes."""

    # --- terminação ---
    largou_z: float = 0.30              # §14 — só `parado c/ caixa` e `andar c/ caixa`
    area_raio: float = 5.0              # §14 — 5 m do spawn

    # --- "de pé" (item 19) ---
    de_pe_z: float = 0.65
    """§14. Compara com 0.76 m, a pelve no keyframe `KNEES_BENT` (medido): permite
    agachar 11 cm, não permite ficar dobrado. Não reusa o `fell_over` (70°) porque
    aquele é folgado demais — daria "de pé" pra um robô dobrado a 65°."""
    de_pe_tilt_rad: float = 0.349       # 20°

    # --- sucesso (§6b/E) ---
    caixa_no_alvo: float = 0.10         # §14 — 3D, `pegar` e `botar`
    caixa_quieta_v: float = 0.05        # §14 — ‖v‖ < 0.05 m/s
    reorienta_angulo_deg: float = 10.0  # §14
    reorienta_xy: float = 0.05          # §14
    andar_raio: float = 0.25            # §14 — R de chegada
    sustenta_pegar_s: float = 5.0       # §14
    sustenta_andar_s: float = 5.0       # §14 — parado de pé 5 s após chegar
    sustenta_botar_s: float = 2.0       # §14 — quieta + de pé, simultâneos
    sustenta_reorienta_s: float = 2.0   # §14 — ângulo + xy + apoiada, simultâneos
    deriva_parado_log: float = 0.20
    """§14 — 🔧 F3: **logada, NÃO é portão.** O critério antigo era deriva < 0.20 m no
    episódio, e ele comprime a taxa de sucesso pra perto de zero sob push nível 4. O
    sucesso do `parado` é sobreviver os 20 s sem `fell_over`."""


@dataclass
class Curriculum:
    rho: float = 0.30                   # §14 — piso uniforme, o anti-esquecimento
    focus_beta: float = 1.0             # §14
    ema_alpha: float = 0.03             # §14 — faixa 0.02-0.05

    limiar_competencia: float = 0.90
    """§14 — **absoluto, todas as tarefas, sem escape hatch.** O portão é `min`
    sobre os níveis destravados da célula, não média: dominar o nível fácil e
    ignorar o difícil não passa."""

    congela_queda: float = 0.10         # §14 — queda > 0.10 congela a célula
    descongela_dist_pico: float = 0.05  # §14 — volta a < 0.05 do pico
    platô_amostras: int = 2000          # §14 — diagnóstico, NÃO portão
    platô_iters: int = 150
    platô_delta: float = 0.01
    alarme_transicoes: float = 2e8      # §14 — 4× a média de 4.6e7

    seed_newest_high: bool = True
    """Semeia o nível mais NOVO com dificuldade alta, pra o sorteio já focar nele.
    Herdado do `PlrHeights`."""


@dataclass
class Train:
    """⚠️ **DOCUMENTAÇÃO, não configuração. Nada aqui é lido pelo código.**

    Estes números moram em outro lugar, e mudá-los AQUI não tem efeito nenhum —
    ficaria um knob pendurado, do tipo que engana quem for tunar depois. Onde cada um
    vive de verdade:

    | número | onde se muda de verdade |
    |---|---|
    | `num_envs` | `cfg.env.scene.num_envs` no lançamento |
    | `num_steps_per_env` | `rl_cfg` do fabricante (já é 24) |
    | `entropy_coef` | `cfg.agent.algorithm.entropy_coef` (já é 0.01) |
    | `iters_por_bloco` | `cfg.agent.max_iterations` no lançamento |
    | `save_interval` | `rl_cfg` do fabricante (já é 50) |

    Ficam registrados porque são o número da §14 e o valor conferido do fabricante —
    serve de referência ao montar a célula de lançamento. Ver
    `g1_multitask/kaggle/`."""

    num_envs: int = 4096
    """§14 — e com DDP este número é POR RANK, não total."""
    num_steps_per_env: int = 24         # §14 — medido; = default do fabricante
    entropy_coef: float = 0.01          # = default do fabricante, conferido 30/07
    save_interval: int = 50             # = default do fabricante -> 20 ckpts em 1000 iters
    iters_por_bloco: int = 2500
    """A run é fatiada em blocos de 2k-3k, com inspeção entre um e o outro — não é
    uma sessão de 30 000. Consequência: `save`/`resume` dispara 10-15 vezes."""


@dataclass
class MultitaskKnobs:
    scene: Scene = field(default_factory=Scene)
    command: Command = field(default_factory=Command)
    reward: Reward = field(default_factory=Reward)
    foundation: Foundation = field(default_factory=Foundation)
    dr: DR = field(default_factory=DR)
    tolerancia: Tolerancia = field(default_factory=Tolerancia)
    push: Push = field(default_factory=Push)
    curriculum: Curriculum = field(default_factory=Curriculum)
    train: Train = field(default_factory=Train)
