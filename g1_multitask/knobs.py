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
    box_xy: tuple[float, float] = (0.30, 0.0)
    """Borda da FRENTE da mesa: `(table_x − shelf_half_xy) + box_half[0]` = 0.30.

    ⚠️ **Era 0.50 (centro da mesa) até 10/08** — o default que a Lift ABANDONOU em
    16/07 (`c2026_07_16_box_edge.py`): no centro, a pose em pé sem escorar
    fisicamente não alcança a caixa. O `box_jitter_x` assimétrico abaixo foi
    calibrado EM CIMA do 0.30 ("+0.20 é o limite prático de alcance; 0.50 era o
    'longe demais' original" — `c2026_07_16_generalize.py`). Recombinado com o
    nominal 0.50, ele punha a caixa sempre na metade OPOSTA da mesa: alcançável em
    ≈19% dos episódios, e o `cond_fisica` do `reorientar` no bloco 1 travou em
    0.17 — o teto era a fração alcançável. O `pegar` deu 0.0000."""
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

    table_jitter_xy: float = 0.05
    """S12 — ±0.05 m no xy da prateleira, por episódio.

    ⚠️ **Era 0.15 até 10/08.** Como o delta é compartilhado com a caixa, ele soma ao
    alcance exigido: com 0.15, ~19% dos episódios caíam além do limite prático de
    0.50 — e o portão de competência exige 0.90, então o teto estrutural (~0.81)
    nunca fecharia o eixo. Com 0.05 sobram ~6% além do limite (0.90 atingível).

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
    """O comando de velocidade. **Todos estes números vêm do cfg de velocity do
    fabricante** (`tasks/velocity/velocity_env_cfg.py:177-193`). Nenhum é escolhido
    aqui, e é essa a intenção: restaurar a faixa de comando testada foi UMA mudança,
    contra as três que afinar os termos de recompensa seria.

    ⚠️ O bloco antigo — `v_max`, `a_max`, `w_max`, `alpha_max`, `d_morto_*`,
    `d_freio_extra`, `heading_gain`, `morto_angular_rad`, `v_max_carga_cheia` — SAIU
    inteiro com o `DesiredTwistCommand` (§10b). Ele parametrizava um controlador de
    posição que não existe mais."""

    lin_vel_x: tuple[float, float] = (-1.0, 1.0)     # fabricante, stage 0
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)     # fabricante, stage 0
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)     # fabricante, stage 0
    resampling_s: tuple[float, float] = (3.0, 8.0)   # fabricante

    rel_standing_envs: float = 0.1
    """Fração de envs com comando zero. **É dividida** com o giro parado, pelo
    `frac_giro_no_standing` — 0,05 parado e 0,05 giro."""

    rel_heading_envs: float = 0.3
    """Fração de envs em que o `ωz` NÃO é sorteado: ele vira
    `clip(stiffness × erro_de_rumo, ±ang_vel_z)`, recalculado a cada passo.

    ⚠️ Só tem efeito com `heading_command=True` e `ranges.heading` preenchidos. Sem os
    dois, o `is_heading_env` nunca é escrito (`velocity_command.py:80-84`) e este
    campo fica inerte **em silêncio**."""

    heading_control_stiffness: float = 0.5
    """Ganho da lei de realimentação do heading. Fabricante.

    ⚠️ Este é o número que o antigo `heading_gain` copiava. Lá ele estava marcado
    "SEM USO desde a S5"; aqui ele volta a ter efeito, porque o modo heading do
    fabricante entra."""

    rel_forward_envs: float = 0.2
    """Fração de envs com comando retilíneo: `lin_vel_x = |x|.clamp(min=0,3)`,
    `lin_vel_y = 0`, `ωz = 0`. Fabricante.

    Ele é também o PRECEDENTE de que o giro parado é uma fração de envs com comando de
    forma restrita, e de onde sai a proporção do piso."""

    frac_giro_no_standing: float = 0.5
    """Que fatia do `rel_standing_envs` vira giro parado (`vx = vy = 0`, `ωz ≠ 0`).

    A fração sai de DENTRO do standing, e não de uma fração nova: o regime parado é o
    mais simples dos três, e reduzi-lo custa menos que reduzir dado de marcha.

    Sem uma fração dedicada o giro parado quase não é sorteado:
    `P(|vx| < 0,05 e |vy| < 0,05) = 0,05 × 0,05 = 0,25%`."""

    piso_giro_rad_s: float = 0.15
    """Piso de `|ωz|` no giro parado, em rad/s.

    **Obrigatório.** O gate dos quatro termos de marcha é `‖cmd_xy‖ + |ωz| > 0,05`, e
    10% dos sorteios de `ωz ~ U(−0,5, 0,5)` ficam abaixo disso. Sem piso, esses envs
    ficam sem sinal de tarefa nenhum.

    **Derivado, não escolhido:** o `rel_forward_envs` trava `lin_vel_x ≥ 0,3` num teto
    de 1,0, ou seja 30% do teto. Trinta por cento do teto de `ωz` (0,5) dá 0,15."""

    alvo_peito_b: tuple[float, float, float] = (0.20, 0.00, 0.15)
    """§14 — alvo da caixa relativo à base. Constante: é por isso que o `pegar` não
    precisa transmitir o alvo do peito no vetor de comando.

    ⚠️ Desde 10/08 o `alvo_peito_w` só usa xy na base; o z é ANCORADO NO MUNDO em
    `pelve_de_pé + 0.15 = 0.91 m` (ver `rewards.alvo_peito_w`). Alvo 100% na base
    fazia segurar agachado valer nota cheia — era o argmax do bloco 2."""


@dataclass
class Reward:
    # --- bloco 1: invariantes, sempre ligados (§6b/B e §6b/C) ---
    track_linear_velocity: float = 2.0
    track_angular_velocity: float = 2.0
    upright: float = 1.0
    postura: float = 0.5
    """Peso do ÚNICO termo de postura. Ele é BONIFICAÇÃO: a função devolve
    `exp(−média(err²/std²))`, que vive em `(0, 1]` e nunca é negativa. Sair da pose
    padrão PARA DE PAGAR; isso não cria dívida.

    ⚠️ **O escopo por tarefa saiu (§6).** Eram quatro termos gateados, com o `std`
    respondendo ao regime de velocidade e o escopo respondendo a se a mão estava
    ocupada. Sobra UM `variable_posture`, corpo todo, sem gate — o desenho do
    fabricante (`joint_names=(".*",)`).

    O escopo era desnecessário porque o termo **se auto-desliga onde o braço é a
    tarefa**. Com comando zero o regime é `standing`, e o cfg do g1 põe
    `std_standing = 0,05` para todas as juntas. Um ombro deslocado 0,5 rad dá
    `0,25/0,0025 = 100`; com 8 juntas de braço de 29 a média fica em ~27,6, e o termo
    vale `exp(−27,6) ≈ 1e−12`. Zero em float32, gradiente zero.

    Os `std` por regime vêm COLHIDOS do cfg do fabricante, não redigitados — ver o
    `env.py`. Os knobs `postura_std_parado`, `postura_std_manipula` e
    `postura_joints` saíram junto com o escopo."""


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
    action_rate_l2: float = -0.10
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    joint_acc: float = -2.5e-7
    feet_slip: float = -0.1

    # --- bloco 2: termos de tarefa, gateados pelo one-hot ---
    lift: float = 2.0                   # só no `pegar` — progresso de altura
    reaching: float = 1.0
    """Aproximação das palmas à caixa, no `pegar` e no `reorientar`. **PERMANENTE.**

    O item 22 do doc previa anelamento com catraca até 0.01. **Nunca foi implementado**
    (nada em `curriculum.py` muta peso de reward) e a decisão de 06/08 é não
    implementar: ele é o único termo que diz "ponha as mãos na caixa", e isso é
    pré-requisito físico das duas tarefas, não muleta de início de treino.

    Ele mantém 20% do orçamento no `pegar` e 50% no `reorientar` depois da
    equalização — a escala é uniforme dentro da tarefa, então a fatia não mudou."""
    box_at_peito: float = 1.0
    """grasp × kernel(caixa → peito). **Só no `locomover_carregando` desde 11/08.**

    Ele saiu do `pegar` por decisão de desenho: lá o alvo é altura de mundo, e ponto.
    Com isso a âncora de mundo do `alvo_peito_w` perdeu a razão de existir e voltou
    para o frame da BASE — que é o único frame observável durante a marcha (a pelve
    oscila, e não há canal de altura de mundo na obs). Ver `rewards.alvo_peito_w`."""
    box_at_prateleira: float = 1.0      # kernel(caixa → prateleira), SEM grasp
    kernel_angulo: float = 1.0          # só no `reorientar`
    grasp: float = 0.5                  # bônus de toque, só no `pegar`

    unload: float = 1.0
    """**A ponte contínua entre tocar e erguer** (11/08). Só no `pegar`.

        unload = preensão × clamp(1 − F_apoio / (m·g)) × [caixa acima do repouso]

    O buraco que ele preenche: `_grasp` é BOOLEANO, então tocar paga e apertar paga
    **zero** até a caixa se mover. Medido no bloco 3: preensão em 0,851 com a caixa
    subindo 4 mm. O robô fica no platô pago.

    A força de apoio da prateleira cai de `m·g` para 0 **antes de a caixa sair do
    lugar** — é essa a única grandeza que responde ao aperto de forma contínua.

    ⚠️ **Os dois fatores de gate são obrigatórios.** Sem `preensão`, derrubar a caixa
    da borda zera o apoio da mesa e paga o termo inteiro. Sem `caixa acima do
    repouso`, a caixa no chão também paga.

    ⚠️ O peso é `env.peso_amostrado × 9,81`, não `box_mass`: a DR de carga aplica
    força externa e o buffer já existe.

    O peso 1,0 herda o slot que o `box_at_peito` deixou no `pegar` — o orçamento da
    tarefa não muda. Subir é Categoria A."""

    sucesso_denso: float = 5.0
    """Bônus por a condição FÍSICA da tarefa valer AGORA, por segundo. (11/08)

    Ele existe porque o alvo não pagava nada: `cond_fisica` era só diagnóstico, e o
    log inteiro do bloco 3 mostrou 0,0000 sem nenhum termo de recompensa olhando para
    ele.

    ⚠️ **Fora de `TERMOS_DE_TAREFA`, de propósito.** Dentro, o orçamento do `pegar`
    iria a 9,5, o fator cairia para 0,42 e o próprio bônus se diluiria para 2,1 — ele
    se anularia. Ele é bônus de OBJETIVO, não sinal de aproximação.

    Vale nas QUATRO tarefas com caixa, então a paridade entre elas fica preservada. O
    `locomover` fica fora: a condição dele é `ones_like` (o critério real dele é a
    média de erro de velocidade), e ele coletaria 5,0/s de graça.

    ⚠️ O peso passa pelo `scale_by_dt`, então 5,0 aqui é 5,0 **por segundo** — e o
    `contrib` lê 5,0, porque o `_step_reward` divide o `dt` de volta
    (`reward_manager.py:132`).

    Por que ele importa tanto: `gamma = 0,99` a 50 Hz dá horizonte de 2,0 s. Prêmio a
    3 s de distância vale 0,22 do valor de face. Um bônus que paga CONTINUAMENTE
    enquanto a condição vale atravessa o desconto; um bônus terminal não."""

    action_rate_bracos: float = 0.25
    """Fator do `action_rate_l2` nos 14 canais de braço (11/08). `1.0` = desliga.

    Medido no bloco 3: `action_rate` custava −0,88 contra 1,07 de todo o sinal de
    tarefa coletado no `pegar` — **82%**. E a tarefa do `pegar` É mover os braços.

    Só os braços, e não o peso global: o termo precisa continuar cobrando jitter de
    perna, que é o que ele existe para conter na locomoção. Os índices saem de
    `find_joints`, e a classe confere que a ordem da ação bate com a ordem das
    juntas."""

    shake_gate_std: float = 0.10
    """Escala do gate do `box_shake` no `pegar`, em metros (11/08).

    O `box_shake` medido subia junto com o `lift` e cancelava o ganho dele. Agora ele
    só cobra **depois** de a caixa chegar perto do alvo: o fator é
    `preensão × exp(−(alvo_z − box_z)²/std²)`. Erguer sai de graça; sacudir a caixa
    já erguida custa.

    Mesmo desenho do `hold_still_bonus` da Lift, que gateia no hold pelo mesmo
    motivo — não taxar a manobra que a tarefa exige."""

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
    **sem** `lift`/`reaching` ligados, este número volta pra mesa.

    ⚠️ **Materializou-se em 11/08**, pela outra ponta: a âncora do peito foi pro
    MUNDO (0.91 m) e o caminho até ela ficou longo demais pro std único — ver
    `peito_std_grosso`."""

    peito_std_grosso: float = 0.30
    """Escala GROSSA do `box_at_peito`, em metros (11/08). Mesmo conserto e mesmo
    modo de falha do `botar_std_grosso`: com a âncora do peito em mundo, o std
    único de 0.05 valia `e⁻²⁵ = zero exato` a 25 cm do alvo — o caminho vertical
    ficou sem o segundo pagador e o `pegar` se acomodou em "mãos na caixa, sem
    erguer" (bloco 3: `lift = 0.02`, `std_vantagem/pegar` 0.18 → 0.075).

    DERIVADA da mesma razão 6:1 que o repo já usa (`botar_std_grosso` = 0.30 sobre
    fino 0.05). O `locomover_carregando` nasce NO alvo e não sente a grossa; o
    fino continua mandando perto do alvo, então a precisão final não afrouxa."""

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
    botar_fracao_solta: float = 0.5
    """Quanto do termo do `botar` exige ter SOLTADO a caixa.

        termo = (1 − f) × kernel  +  f × kernel × (1 − preensão)

    **0.5 desde 06/08.** A §4 do doc especifica 0.0 (kernel puro) e o raciocínio dela
    fecha para o TRANSPORTE: longe → baixa, aproximando → sobe, soltando → continua
    alta. O que ele não cobre é o fim da tarefa. Medido com f = 0.0:

        estado                          kernel   termo (peso efetivo 4.0)
        transportando (0.46 m, na mão)   0.046      0.184
        chegando      (0.20 m, na mão)   0.321      1.282
        NO DESTINO, ainda segurando      0.924      3.695
        NO DESTINO, soltou               0.924      3.695   <- idêntico

    O argmax é "segurar a caixa parada em cima da prateleira para sempre", e o
    critério de sucesso exige `~preensao`. Recompensa máxima num estado que o critério
    reprova — mesma classe de defeito que o C1 e o C2 do bloco 1. Eu tinha registrado
    isto como "indiferença, não incentivo contrário", e a indiferença basta para o
    argmax não ser o sucesso, que é o que importa.

    **Por que 0.5 e não outro valor.** É o mesmo 50/50 que o repo já usa para combinar
    duas parcelas que ambas têm de contar sem nenhuma dominar — `0.5 × grosso +
    0.5 × fino` no `reaching_reward`, no `orienta_face` e neste próprio termo.

    **Por que não 1.0**, que seria o incentivo mais forte: com f = 1 o termo é
    `kernel × (1 − preensao)`, ou seja **zero durante todo o transporte**, porque o
    robô está segurando. Some o gradiente que traz a caixa até a prateleira, que é a
    metade da tarefa. f tem de ficar em (0, 1).

    Com 0.5: transporte de 0.092 a 1.848 (20×) e soltar no destino dobra para 3.695.
    O teto de 4.0 do orçamento passa a ser atingível SÓ com a caixa solta no alvo."""

@dataclass
class Episodio:
    """Comprimento do episódio, POR REGIME. (11/08)

    Era global em 20 s. O `pegar` gasta ~3 s aproximando e o resto do episódio
    repetindo o mesmo estado — 17 s de amostra quase idêntica por tentativa.

    ⚠️ O `locomocao_s` continua sendo o `episode_length_s` do cfg, ou seja o TETO. O
    `max_episode_length` do mjlab é escalar (`manager_based_rl_env.py:281`), então o
    corte por tarefa entra como terminação própria — ver `terminations.time_out_por_tarefa`.

    Por que 10 s e não 5 s: a preensão se estabelece por volta de 3 s (derivado de
    `grasp = 0,851` num episódio de 20 s). Com 5 s sobrariam 2 s para apertar, erguer e
    sustentar, e o `sustenta_pegar_s` não caberia."""

    locomocao_s: float = 20.0
    manipulacao_s: float = 10.0


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
    """Empurrão FIXO, do fabricante. O eixo de currículo SAIU (§7).

    Os seis componentes e o intervalo são os do `push_robot` do
    `velocity_env_cfg.py:223-237`, sem escala e sem nível. Saíram junto o
    `push_fator` por env, a força sustentada (`empurrao_sustentado`) e a célula
    `(PARADO, PUSH)` do orquestrador.

    O único acréscimo nosso é a `JANELA_LIVRE_S` do `push.py`."""

    velocity_range: dict = field(default_factory=lambda: {
        "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
    })
    intervalo_s: tuple[float, float] = (1.0, 3.0)   # fabricante


@dataclass
class Tolerancia:
    """Limiares de terminação e de sucesso. Mudar um de SUCESSO é Categoria C —
    recomeçar do zero — porque a régua do currículo se move junto."""

    # --- terminação ---
    largou_z: float = 0.30              # só o `locomover_carregando`

    # --- "de pé" ---
    de_pe_z: float = 0.65
    """Compara com 0.76 m, a pelve no keyframe `KNEES_BENT` (medido): permite agachar
    11 cm, não permite ficar dobrado. Não reusa o `fell_over` (70°) porque aquele é
    folgado demais — daria "de pé" pra um robô dobrado a 65°."""
    de_pe_tilt_rad: float = 0.349       # 20°

    # --- critério BASE: erro médio de velocidade (§8) ---
    tol_v: float = 0.25
    """Erro linear médio máximo, em m/s. `erro_lin = (1/T) ∫ ‖v_cmd_xy − v_xy‖ dt`.

    O critério é FÍSICO. Ele mede metros por segundo, enquanto a recompensa calcula
    `exp(−erro²/σ²)` — as duas funções são diferentes, então o `σ` da recompensa não
    move a régua.

    **Derivado:** metade do `std` do `track_linear_velocity` do fabricante
    (`sqrt(0.25) = 0.5`). Com o comando em `U(−1, 1)` nos dois eixos, `‖v_cmd_xy‖`
    vale ~0,765 m/s em média, então 0,25 é ~33% de erro relativo.

    ⚠️ **Risco aceito, e declarado no §8:** o número é fixado SEM medição do
    deslocamento de pelve durante o agachamento. Com comando zero na manipulação, este
    limiar passa a medir a deriva. Um agachamento move a pelve ~0,3 m em ~1 s, o que
    dá ~0,015 m/s de média num episódio de 20 s — cabe com folga. Confirmar no log."""

    tol_w: float = 0.70
    """Erro angular médio máximo, em rad/s. `erro_ang = (1/T) ∫ |ωz_cmd − ωz| dt`.

    Com comando zero, ele mede rebolado de pelve — que é exatamente o buraco que o
    `hold_still` deixou ao sair (§10). Aqui ele volta como critério, não como
    recompensa.

    ⚠️ **Era 0.35 (metade do std do kernel do fabricante) até 11/08 — e 0.35 zerou o
    sucesso das 3 tarefas por 8000+ iterações.** O piso REAL, medido por tarefa em
    duas janelas independentes do bloco 3 (`contrib/<tarefa>/erro_vel_ang`):
    locomover 0.62-0.66 · pegar 0.65-0.67 · reorientar 0.72-0.73. Uniforme com e sem
    comando ⇒ o piso é sistêmico: a oscilação de guinada da própria marcha e o
    `push_robot` do fabricante (até ±0.78 rad/s a cada 1-3 s) moram DENTRO da
    integral. 0.35 selecionava andar arrastado — o gait raso do início do bloco 1
    passava, o gait bom não.

    O 0.70 é a MESMA derivação com o fator corrigido pela medição: 1× o std do
    kernel (`sqrt(0.5) ≈ 0.707`) em vez de metade. Mudança de régua feita na janela
    em que `perf ≡ 0` em tudo (nada a preservar). **Aceitação:** o `perf_n0` do
    `locomover` descola do zero em ~100-200 iterações; se não, o degrau é 0.80."""

    # --- condições adicionais por tarefa (§8) ---
    caixa_no_alvo: float = 0.10         # 3D, `pegar` e `botar`
    caixa_quieta_v: float = 0.05        # ‖v‖ < 0.05 m/s, só o `botar`
    reorienta_angulo_deg: float = 10.0
    reorienta_xy: float = 0.05

    alvo_tol_z: float = 0.02
    """Folga do PISO de altura do `pegar`, em metros: `box_z >= alvo_z − 0,02`. (11/08)

    ⚠️ **Piso, e não esfera.** O critério era `‖caixa − alvo‖ < 0,10` em 3D. Com o eixo
    `alvo` graduado isso reprovava o robô por fazer MAIS: alvo em +5 cm e caixa a
    +26 cm dá distância de 21 cm. E ele não observa o nível, então não teria como
    parar na altura certa.

    Com piso, o critério é monotônico em z: erguer mais nunca reprova. É o que torna a
    graduação por fração segura."""

    sustenta_pegar_s: float = 2.0
    """Segundos com a condição do `pegar` verdadeira. **Era 5,0 até 11/08.**

    ⚠️ Ele TEM de ser menor que o episódio da manipulação. Com episódio de 10 s e
    preensão estabelecida por volta de 3 s (derivado: `grasp` valia 0,851 de média num
    episódio de 20 s), 5,0 s exigiria a condição valendo de 5 s a 10 s sem uma falha —
    e num episódio de 5 s ela seria matematicamente impossível."""
    sustenta_carregar_s: float = 3.0
    sustenta_botar_s: float = 2.0       # quieta + de pé, simultâneos
    sustenta_reorienta_s: float = 2.0   # ângulo + xy + apoiada, simultâneos


@dataclass
class Curriculum:
    rho: float = 0.30                   # §14 — piso uniforme, o anti-esquecimento
    focus_beta: float = 1.0             # §14
    ema_alpha: float = 0.03             # §14 — faixa 0.02-0.05

    limiar_competencia: float = 0.90
    """**Absoluto, todas as tarefas, sem escape hatch.**

    ⚠️ O portão lê o nível CORRENTE (`perf[T][topo]`), e não mais o `min` sobre todos
    os níveis abertos. Consequência: o eixo avança mesmo se um nível anterior
    regrediu. O piso `ρ/L` continua sorteando os níveis antigos, então a regressão
    aparece no log — ela vira observabilidade, não portão.

    ⚠️ O congelamento SAIU (§9), e com ele o `ema_alpha_lenta`, o `congela_queda` e o
    `descongela_dist_pico`. Ele era o único mecanismo capaz de bloquear a abertura de
    um filho, e a referência já era EMA lenta, que suaviza sozinha."""

    platô_amostras: int = 2000          # §14 — diagnóstico, NÃO portão
    platô_iters: int = 150
    platô_delta: float = 0.01
    alarme_transicoes: float = 2e8      # §14 — 4× a média de 4.6e7

    seed_newest_high: bool = True
    """Semeia o nível mais NOVO com dificuldade alta, pra o sorteio já focar nele.
    Herdado do `PlrHeights`."""

    piso_amostragem: float = 0.15
    """Fração MÍNIMA de envs por tarefa aberta. (11/08)

    O sorteio de tarefa era uniforme (`randint`), então cada tarefa aberta levava
    `1/K` — e uma tarefa já resolvida consumia a mesma amostra que a travada. Agora a
    massa é inversa à competência:

        P(T) = piso + (1 − piso·K) · (1 − perf[T][topo]) / Σ(1 − perf)

    Com K = 5 o piso ocupa 0,75 e sobram 0,25 para distribuir. Com K = 3 sobram 0,55.

    ⚠️ O piso é anti-esquecimento, o mesmo papel do `rho` nos NÍVEIS. Sem ele, uma
    tarefa que chega a `perf = 1,0` sairia do sorteio e a política a esqueceria.

    ⚠️ `piso × K` tem de ficar abaixo de 1. Com 5 tarefas o teto do piso é 0,20."""


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
    episodio: Episodio = field(default_factory=Episodio)
    foundation: Foundation = field(default_factory=Foundation)
    dr: DR = field(default_factory=DR)
    tolerancia: Tolerancia = field(default_factory=Tolerancia)
    push: Push = field(default_factory=Push)
    curriculum: Curriculum = field(default_factory=Curriculum)
    train: Train = field(default_factory=Train)
