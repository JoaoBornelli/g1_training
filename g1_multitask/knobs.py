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

    table_jitter_xy: float = 0.15
    """S12 — ±0.15 m no xy da prateleira, por episódio.

    Sem ele a mesa fica sempre no mesmo lugar e o `botar` decora um ponto em vez de
    aprender a pousar onde a prateleira está.

    ⚠️ **A caixa acompanha o mesmo delta.** Deslocar só a mesa deixaria a caixa
    pendurada fora dela — e o `reset_scene_plr` posiciona a caixa relativa ao xy
    NOMINAL, não ao da mesa.

    ⚠️ Tem de caber no alcance dos braços. Mais que isso obriga o `botar` a caminhar
    até o alvo, e ele não tem eixo de distância para graduar isso."""
    table_jitter_yaw_deg: float = 15.0
    """S12 — yaw da prateleira. A spec pede "um pouco de yaw" e não dá o número.

    ⚠️ **DERIVADO, não escolhido:** é o mesmo `box_jitter_yaw_deg` de ±15° que a caixa
    já usa. Reusar a amplitude que o repo já calibrou evita inventar uma segunda
    escala de jitter angular para a mesma cena.

    Girar a mesa em torno do centro dela NÃO move a caixa, que repousa no centro —
    por isso o yaw não entra no delta compartilhado do xy.

    Confirmar na rodada: se o `botar` não sofrer com o yaw, ele pode subir."""


@dataclass
class Command:
    v_max: float = 0.5
    """S5. Era 1.0, a borda do range stage-0 do fabricante. Decisão do desenho: o ULC
    usa 0.55 no mesmo robô, e 0.5 é a velocidade em que o perfil de duas fases fecha o
    orçamento de 20 s com folga (ver a conta da S14)."""
    a_max: float = 1.0
    """S5. Aceleração linear máxima, em m/s². São ~0.1 g; um bípede em marcha faz isso.
    Ela é o teto do limitador de taxa e a origem da banda de frenagem."""
    w_max: float = 1.2
    """S5. Era 0.5, o `ang_vel_z` stage-0 do fabricante. O ULC usa 1.2 no mesmo robô, e
    a fase 1 do perfil precisa girar até 180° dentro do orçamento do episódio."""
    alpha_max: float = 3.0
    """S5. Aceleração angular máxima, em rad/s².

    ⚠️ **PALPITE A VALIDAR.** A S5 manda medir: rodar o `model_stand_step_2000` no
    `play`, mandar um degrau de `wz` e ler a aceleração que a política entrega. A
    medição não é viável nesta rodada por dois motivos — o `play` não roda nesta
    máquina, e o env da Stand não tem comando de `wz` para receber o degrau. A própria
    S5 autoriza o fallback: "use 3,0 e registre no docstring que é um palpite".

    Consequência de errar: `alpha_max` alto demais deixa passar degrau angular que o
    robô não segue; baixo demais faz o giro da fase 1 arrastar. A banda angular é
    derivada dele, portanto o erro se propaga para ela."""
    heading_gain: float = 0.5
    """§14 / fabricante — `heading_control_stiffness`.

    ⚠️ **SEM USO desde a S5.** O perfil angular passou a ser trapezoidal, igual ao
    linear: `w_max` fora da banda, rampa dentro dela, zero dentro do morto. Um ganho
    proporcional não descreve mais o comportamento. O campo fica porque o `env.py` o
    passa ao construir o termo, e removê-lo mexeria em arquivo fora do escopo da S5."""

    d_morto_andar: float = 0.05
    """S5. Era 0.25, que era o R de chegada do `andar`. Os dois se separaram: o morto
    tem de ser MENOR que o raio de sucesso, senão o comando zera antes de o robô
    entrar no círculo que a régua exige. A S6 põe o raio de chegada em 0.10."""
    d_morto_manipula: float = 0.30       # §14 — `alvo_peito_b[0]` + 0.10 (derivado)
    d_freio_extra: float = 0.125
    """S5 — `d_freio − d_morto`, a banda em que a velocidade desce de `v_max` a zero.

    DERIVADO de `v_max` e `a_max`: `v²/2a = 0.5²/2 = 0.125 m`. O `commands.py` afirma
    essa igualdade num assert, para o número não voltar a ficar órfão.

    ⚠️ Era 0.50, dimensionado para `v_max = 1.0` com a justificativa "~2 passos para
    desacelerar" = 1.0 m/s². A 0.5 m/s os mesmos 0.50 m dariam 0.25 m/s², e o robô
    rastejaria o último meio metro."""
    v_max_carga_cheia: float = 0.25
    """S14 — `v_max` quando a caixa está no peso máximo do eixo (5 kg).

    Entre 1 kg e 5 kg o `v_max` cai LINEARMENTE de `v_max` para este valor. A
    inclinação sai da conta, não de um número digitado:
    `(0.50 − 0.25) / (5 − 1) = 0.0625` m/s por kg.

    ⚠️ **Ligado à massa SORTEADA, não ao nível.** A S1 sorteia a massa em
    `U(1, peso(nível))`; ligar ao nível daria 0.25 m/s a um env que tirou 1.2 kg.

    Orçamento que este número tem de fechar — `andar c/ caixa`, 2.0 m, 5 kg, rumo no
    nível fácil: giro 0.4 s + caminhada 8.0 s + sustentação 3.0 s = **11.4 s de 20 s**.
    ⚠️ Se `rumo` for acrescentado ao `andar c/ caixa` mais tarde, refazer a conta: com
    ±180° a 1.2 rad/s são +2.6 s."""
    morto_angular_rad: float = 0.087
    """S5 — 5°. O equivalente angular do `d_morto_andar`: dentro dele `w_cmd` é zero.

    Cabe dentro da tolerância de `alinhado` da S6, que dispara em 10°. Se o morto fosse
    maior que ela, o comando pararia de girar antes de o critério considerar alinhado."""

    alvo_peito_b: tuple[float, float, float] = (0.20, 0.00, 0.15)
    """§14 — alvo da caixa em frame da BASE. Constante: é por isso que o `pegar`
    tem `alvo_pos = 0` no vetor de comando — o peito não precisa ser transmitido."""



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
    dof_pos_limits: float = -0.5
    """§14 dizia −1.0. **Reduzido para −0.5 em 03/08/2026.**

    Com o clamp de pitch em 1,05 rad (`acao.py`), o joelho e o quadril encostam no
    batente **de propósito** durante o agachamento — é o que descer exige. O termo
    então cobra o comportamento certo. Na run antiga ele já chegava a −1,19 logado,
    que é mais que o peso inteiro do `reaching`.

    Não vai a zero: sem ele a política estaciona a junta no limite, e aí o alvo de
    junta perde autoridade (o PD satura). Metade é o começo; o `contrib` por
    tarefa × termo decide o valor final."""
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

    orcamento_tarefa: float = 4.0
    """Teto de sinal de tarefa POR PASSO, igual nas 7 tarefas. `<= 0` desliga (06/08).

    **DERIVADO, não escolhido.** É `track_linear_velocity + track_angular_velocity`
    do `unitree_g1_flat_env_cfg` — `2.0 + 2.0` — que é o orçamento de tarefa que o
    fabricante dá ao G1 na task de locomoção. Fora dele, o cfg do fabricante só tem
    `upright 1.0` e `pose 1.0`, que são piso postural e não sinal de tarefa.

    É o único orçamento do desenho que NÃO foi digitado por nós: os pesos do bloco 2
    saíram todos da §14. Ancorar nele é ancorar na referência testada.

    **O problema que ele resolve.** Os tetos medidos em 06/08, depois do bloco 1:

        parado 5.50 · andar 5.50 · pegar 6.00 · botar 2.50 · reorientar 3.50
        parado c/ caixa 7.00 · andar c/ caixa 7.00

    Só de sinal de tarefa: `botar` tinha **1.0** e as duas com caixa **5.5** — 5,5×.
    E as PENALIDADES não são gateadas: `table_contact −1.5`, `com_balance −2.0`,
    `back_penalty −0.5`, `action_rate_l2 −0.25` cobram o mesmo de todas. Ou seja o
    `botar` enfrentava a mesma pilha de custo com um quinto do sinal.

    O segundo motivo é o normalizador: o rsl_rl normaliza a vantagem **uma vez sobre
    o rollout inteiro** (`ppo.py:186-188`), com as 7 tarefas no mesmo pote. Tarefa com
    escala de recompensa menor sai com vantagem menor e é comprimida contra zero no
    gradiente. A escala relativa entre tarefas é, portanto, taxa de aprendizado
    relativa entre tarefas.

    Fatores que isto produz (calculados pelo `_equaliza_orcamento`, não digitados):

        parado 1.000 · andar 1.000 · pegar 0.800 · botar 4.000 · reorientar 2.000
        parado c/ caixa 0.727 · andar c/ caixa 0.727

    **Aceitação:** o `_std_vantagem_por_tarefa` do `runner.py` já loga desvio de
    vantagem por tarefa (medido antes: 0.79 / 1.11 / 1.15). Se a dispersão dessa
    série não encolher, o teto nominal não era o proxy certo para escala efetiva —
    e aí o lever passa a ser a vantagem medida, não o orçamento."""

    # --- bloco 3: anti-hacks ---
    com_balance: float = -2.0           # OFF no `andar` (gate explícito)
    """§14 dizia −3.0. **Reduzido para −2.0 em 03/08/2026.**

    A fórmula é `peso × clamp(fwd)²`, com `com_margin = 0.08`. Alcançar uma caixa no
    chão joga o centro de massa para frente por definição, então este termo se opõe
    diretamente à tarefa. Com −3.0:

        excursão   custo      contra o orçamento de tarefa (+4.0 no `pegar`)
        0,1 m      −0,03      irrelevante
        0,5 m      −0,75      19%, incômodo
        1,0 m      −3,00      cancela a tarefa inteira

    Não sei qual excursão um alcance ao chão produz — é o número que falta. Com −2.0 o
    caso de 1,0 m cai para −2,0, que ainda dói mas não zera o sinal.

    Não vai a zero: centro de massa sobre os pés é o que impede tombar para frente, e
    é justamente no agachamento que isso fica difícil. O `contrib` por tarefa × termo
    decide o valor final."""
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
    terminacao: float = -200.0
    """S9 — penalidade por terminação que NÃO é `time_out`. Valor do ULC, com o mesmo
    comprimento de episódio (20 s).

    Ela existe porque a S9 tirou o piso de sobrevivência: `track_linear_velocity` e
    `track_angular_velocity` deixaram de valer nas três tarefas de manipulação, e sem
    eles o retorno de um episódio curto encosta no de um longo.

    ⚠️ **`is_terminated`, não `time_out`.** Punir o fim natural do episódio ensina o
    robô a morrer.

    ⚠️ **O peso passa pelo `scale_by_dt`** (`reward_manager.py:127`), então o custo
    REAL de uma queda é `−200 × 0.02 = −4,0`, e não −200. Um episódio inteiro de
    `parado` rende cerca de 80 com o piso ligado, portanto a queda custa ~5% disso. Se
    o ULC media −200 já com o dt, o número está certo; se media −200 efetivos, falta
    um fator 50. A aceitação da S9 decide: se o comprimento médio do episódio não
    subir no bloco de 300 iterações, o número está fraco demais."""
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

    botar_std_grosso: float = 0.30
    """S/C2 (06/08) — escala GROSSA do `box_at_prateleira`, em metros.

    DERIVADA da razão que o repo já usa entre grosso e fino: o `orienta_face` usa
    30°/5° e o `reaching` usa 1.0/0.25 — razão 6:1 e 4:1. Com o fino em
    `sustain_std = 0.05`, a razão 6:1 dá 0.30.

    Ela cobre a distância real de transporte: a caixa nasce na mão a ~0.40 m da
    prateleira, e com o `std` único de 0.05 o termo valia 4e-28 ali — zero em float32.
    Com 0.30 ele vale 0.17 no spawn e cresce monotonicamente."""
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

    box_friction: bool = True
    """S13 — randomiza o atrito da superfície da caixa por episódio.

    A pega por abraço depende INTEIRAMENTE do atrito da superfície: sem preensão de
    dedos, o que segura a caixa é a força normal das palmas vezes μ. Sem randomizar,
    a política fica calibrada num único valor, e papelão contra plástico quebra a
    preensão no robô real.

    Mesma primitiva do `foot_friction` (`dr.geom_friction`), que o `knobs.DR` já marca
    como testada em CPU e GPU."""
    box_friction_range: tuple[float, float] = (0.8, 1.2)
    """S13 — faixa de μ tangencial da caixa. O nominal é 1.0 (`common/box.py:26`).

    ⚠️ **A spec pede "faixa estreita" e não dá o número.** ±20% em torno do nominal é
    a escolha, e ela é conservadora de propósito: randomização por episódio acrescenta
    variância ao `success_buf`, a EMA com α = 0.03 tem de mediá-la, e mais variância
    significa mais congelamento espúrio — que é justamente o que a S3 acabou de tratar.

    Alargar depois é Categoria A. Começar largo e descobrir que o currículo não avança
    custaria a rodada inteira. Confirmar com `Contrib/*/grasp` e com a taxa de sucesso
    do `pegar` antes de mexer."""

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
    andar_raio: float = 0.25            # §14 — R de chegada (usado pelo `andar c/ caixa`)
    andar_raio_chega: float = 0.10
    """S6 — o raio que DISPARA a chegada do `andar`. Mais apertado que o antigo 0.25.

    Tem de ser maior que o `d_morto_andar` de 0.05, senão o comando zera antes de o
    robô entrar no círculo que a régua exige."""
    andar_raio_mantem: float = 0.25
    """S6 — o raio que a SUSTENTAÇÃO exige, depois de a chegada disparar.

    ⚠️ Histerese, e ela não é conforto. O `cond` sustentado zera o contador quando a
    condição quebra. Sob push nível 4 — 50 N segurados por até 3 s — o robô sai de um
    círculo de 0.10 m e o contador reinicia sem parar. O `andar` nunca chegaria a 0.90,
    e ele é pai de `pegar` e de `reorientar`: a árvore inteira travaria atrás dele."""
    alinhado_chega_deg: float = 10.0
    """S6 — erro de rumo que DISPARA o alinhamento. O `morto_angular_rad` da S5 é 5°,
    portanto o comando ainda gira até bem dentro desta tolerância."""
    alinhado_mantem_deg: float = 25.0
    """S6 — erro de rumo que a sustentação exige. Mesma histerese do raio, mesmo motivo.

    ⚠️ O `alinhado` é GUARDA, não exigência independente. Quem chega andando para
    frente chega apontado, porque o alvo de orientação é o próprio rumo (S5). Não
    espere que ele morda."""
    sustenta_pegar_s: float = 5.0       # §14
    sustenta_andar_s: float = 3.0
    """S6 — era 5.0. Segundos de pé e dentro do raio depois de chegar.

    ⚠️ É um knob SÓ, e o `_exigencia_s` o usa para `ANDAR` e para `ANDAR_CAIXA`.
    Baixar para 3 s muda os dois, e isso é desejado: o orçamento de tempo do
    `andar c/ caixa` é o mais apertado das sete tarefas (ver a conta da S14)."""
    sustenta_botar_s: float = 2.0       # §14 — quieta + de pé, simultâneos
    sustenta_reorienta_s: float = 2.0   # §14 — ângulo + xy + apoiada, simultâneos
    deriva_parado_log: float = 0.20
    """§14 — 🔧 F3: **logada, NÃO é portão.** O critério antigo era deriva < 0.20 m no
    episódio, e ele comprime a taxa de sucesso pra perto de zero sob push nível 4. O
    sucesso do `parado` mede agora VELOCIDADE, não posição — ver `parado_v_max`."""

    parado_v_max: float = 0.30
    """Velocidade horizontal máxima que ainda conta como "parado", em m/s.

    **Novo em 03/08/2026.** Pedido do user: *"Ele não precisa ficar no mesmo lugar,
    mas sim não se mover."* Estar deslocado é grátis; **se mover** é que reprova.

    Por que 0,30 separa bem:

        andar          `v_max` = 1,0 m/s          persistente  -> reprova
        empurrão n4    ±50 N em ~35 kg = 1,43 m/s²
                       pico ~0,4 m/s em 0,3 s     transitório  -> cabe na folga
        balanço em pé  bem abaixo de 0,1 m/s      -> passa

    O empurrão fura o limiar, mas dura 0,3 a 3,0 s de um episódio de 20 s, ou seja
    1,5% a 15% — dentro dos 20% de folga da `parado_fracao`."""

    limite_fora_de_pe_s: float = 2.5
    """S7 — segundos ACUMULADOS fora de `de pé` que o `parado` tolera no episódio.

    A régua era `time_outs & nunca_caiu`, e um robô permanentemente agachado passava:
    o `fell_over` mede só INCLINAÇÃO (70°), e agachado com o tronco vertical ela é ~0°.

    2,5 s é o vão entre os dois comportamentos que a régua precisa separar. Um passo
    protetivo sob push custa de 1 a 2 s fora de `de pé` e passa. Agachar e ficar
    agachado consome os 20 s do episódio e não passa.

    ⚠️ ACUMULADO, e não consecutivo. Sob push nível 4 o robô sai e volta de `de pé`
    várias vezes; um contador consecutivo perdoaria dez quedas curtas seguidas.

    ⚠️ Usa o `de_pe` COMPARTILHADO, sem limiar próprio. Se o log mostrar o robô parado
    em 0,66 m o episódio inteiro, o conserto é um knob de `z` só para o `parado` — não
    mexer no `de_pe`, que `pegar`, `botar`, `parado c/ caixa` e `andar c/ caixa` usam.

    ⚠️ Não usar limiar em espaço de juntas. A medição do `model_stand_step` mostra que
    as juntas que se mexem são as que EQUILIBRAM (hip_yaw ±7,9°, hip_roll ±3,9°, o
    resto abaixo de 2,1°). Um limiar ali mira o mecanismo de controle e briga com o
    push. E os braços estão congelados na Stand; no multi-tarefa eles manipulam,
    portanto aqueles números não transferem."""

    parado_fracao: float = 0.80
    """Fração dos passos do episódio em que `de pé E devagar` tem que valer.

    **É a fração que implementa a intenção, não o instante.** A exigência de
    sustentação do `parado` é 0 s, então o critério é testado NUM passo só — o do
    `time_out`. Num passo só, dois furos medidos:

    - anda os 20 s e para no último passo -> aprovaria
    - fica sentado 19 s e levanta no último -> aprovaria (o `de_pe` entra na mesma
      fração, então isto fecha de graça)

    "Não se mover" é propriedade do episódio, não de um instante. Daí a fração.

    ⚠️ Mudar isto é **Categoria C** — a régua do currículo se move, e as EMA de
    `perf` acumuladas sob a régua antiga passam a significar outra coisa."""


@dataclass
class Curriculum:
    rho: float = 0.30                   # §14 — piso uniforme, o anti-esquecimento
    focus_beta: float = 1.0             # §14
    ema_alpha: float = 0.03             # §14 — faixa 0.02-0.05

    limiar_competencia: float = 0.90
    """§14 — **absoluto, todas as tarefas, sem escape hatch.** O portão é `min`
    sobre os níveis destravados da célula, não média: dominar o nível fácil e
    ignorar o difícil não passa."""

    ema_alpha_lenta: float = 0.003
    """`ema_alpha / 10`. É a taxa da REFERÊNCIA contra a qual a queda é medida (S3).

    A referência era o máximo corrido. O máximo de um sinal ruidoso não estima a
    média: ele estima a média mais 2,5σ a 3σ. A EMA de Bernoulli tem σ estacionário
    `sqrt(α·p(1−p)/(2−α))`; com α = 0.03 e uma amostra por atualização isso dá 0.062
    em p = 0.50 e 0.037 em p = 0.90. O desvio do máximo SOZINHO passa do
    `congela_queda` de 0.10, e a célula congelava sem nenhuma regressão real.

    Dez vezes mais lenta que a EMA de medição, de propósito: a referência tem que se
    mover devagar o bastante para que uma queda genuína apareça como diferença, e
    rápido o bastante para acompanhar melhora sustentada."""

    congela_queda: float = 0.10         # §14 — queda > 0.10 congela a célula
    descongela_dist_pico: float = 0.05
    """§14 — volta a < 0.05 da referência descongela.

    ⚠️ O nome cita `pico` e a referência deixou de ser o pico na S3. O nome fica como
    está porque a S3 não o cita, e a regra da spec proíbe mexer no que ela não cita."""
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
