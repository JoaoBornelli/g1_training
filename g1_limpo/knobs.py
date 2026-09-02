"""Todos os números do g1_limpo, num lugar só.

Regra do pacote: nenhum número solto no meio do código. Um treino tem de ser
reproduzível por `git diff` deste arquivo.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Este pacote não importa `g1_training`, nem
`g1_poc`, nem `g1_multitask`. Os valores abaixo foram TRANSCRITOS à mão dessas
referências, e o `paridade.py` existe para provar que a transcrição está correta.

Ver `specs/g1-limpo.md` e `docs/planos/2026-08-25-g1-limpo.md`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Cena:
    """Geometria da cena. Transcrito de `g1_training/common/box.py` e
    `g1_poc/knobs.py::Cena`."""

    # --- caixa (corpo LIVRE, erguível) ---
    caixa_meia_aresta: tuple[float, float, float] = (0.10, 0.10, 0.10)
    caixa_massa: float = 1.0
    caixa_atrito: tuple[float, float, float] = (1.0, 0.02, 0.001)
    caixa_rgba: tuple[float, float, float, float] = (0.8, 0.5, 0.2, 1.0)
    caixa_condim: int = 3

    # --- a FACE ALVO, marcada visualmente ---
    # ⚠ O `reorientar` pede que UMA face específica fique normal ao robô. Ela é
    # CONSTANTE, e não sorteada: a dificuldade está na ORIENTAÇÃO DE NASCIMENTO da
    # caixa, e não em qual face se pede.
    #
    # `-X` no frame da caixa: com a caixa nascendo à frente do robô e quatérnion
    # identidade, o `-X` dela aponta de volta para ele. Portanto "zero voltas" é a
    # orientação de nascimento neutra.
    #
    # ⚠ O marcador é uma placa fina, SÓ VISUAL: `contype = 0`, `conaffinity = 0`,
    # `density = 0`. MEDIDO: massa e inércia ficam bit-idênticas à caixa sem ele, e o
    # `paridade.py` afirma isso. Sem o marcador a inspeção do `reorientar` seria cega
    # — um cubo uniforme girado 90° é visualmente idêntico.
    face_alvo_b: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    marcador_rgba: tuple[float, float, float, float] = (0.10, 0.90, 0.30, 1.0)
    marcador_espessura: float = 0.002
    marcador_folga: float = 0.005      # quanto a placa é menor que a face
    # borda perto do robô (lição de 16/07: mais longe e a pega vira sorte)
    caixa_xy: tuple[float, float] = (0.32, 0.00)
    caixa_jitter_y: tuple[float, float] = (-0.18, 0.18)
    caixa_jitter_yaw_deg: float = 15.0

    # --- prateleira (laje FINA, sem free joint -> o mjlab auto-envolve em MOCAP) ---
    prateleira_meia_xy: float = 0.30
    prateleira_meia_z: float = 0.02
    prateleira_atrito: tuple[float, float, float] = (1.0, 0.02, 0.001)
    prateleira_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.55, 1.0)
    prateleira_condim: int = 3
    prateleira_xy: tuple[float, float] = (0.50, 0.00)
    # ⚠ o piso é 0,04 porque a laje tem 4 cm de espessura total: com o TOPO em 0,04
    # ela APOIA no chão em vez de atravessá-lo. Dois corpos estáticos em contato
    # gastam slots de contato.
    prateleira_topo_piso: float = 0.04
    prateleira_topo_teto: float = 0.55
    prateleira_jitter_z: float = 0.02

    # ⚠ a mobília SAI do grupo 0. O `foot_height_scan` do fabricante usa
    # `include_geom_groups=(0,)` e leria a prateleira COMO CHÃO.
    grupo_mobilia: int = 2

    # afastamento da mobília na forma de locomoção pura
    afasta_z: float = 5.0

    # extensão da cena, para o viewer
    extent: float = 2.5

    # --- ação ---
    # ⚠ DECIDIDO: 1,0, o valor do fabricante. O 0,8 é o valor do `g1_multitask`, que
    # andou — mas ele está CONFUNDIDO com o resto daquela config (termo postural,
    # `terminacao`, `dof_pos_limits`), portanto não é evidência a favor de 0,8
    # especificamente. O commit `3fa588a` do g1_poc diz: "o 0,8 cortava 20% da
    # autoridade de junta, e autoridade é exatamente o que uma fase de balanço
    # precisa. Nenhuma medida justificava o corte."
    #
    # ⚠ PRÉ-REGISTRADO: se o portão da F1 falhar, este é o PRIMEIRO e ÚNICO número a
    # mover, para 0,8. Não mexer em mais nada no mesmo bloco.
    escala_acao_mult: float = 1.0

    # --- física de manipulação ---
    # cicatriz de 15/07: `elliptic` com `impratio=10` divergiu para NaN no reset
    # parcial. `pyramidal` com 1,0 é o par que roda.
    njmax: int = 800
    nconmax: int = 300
    impratio: float = 1.0
    cone: str = "pyramidal"

    # --- reset da base, POR FORMA ---
    # O fabricante sorteia yaw no CÍRCULO INTEIRO. Com a mobília de pose ABSOLUTA o
    # robô precisa nascer olhando para ela, portanto na manipulação o yaw é apertado.
    # Na locomoção a mobília sobe 5 m: não existe nada com que alinhar o rumo, e ali
    # o ±3,14 é de graça e é a receita do fabricante.
    #
    # ⚠ O ±0,2 GLOBAL foi o defeito central de um bloco medido: o erro de rumo era
    # sempre minúsculo, o `track_angular_velocity` era satisfeito sem fazer nada, e o
    # canal de yaw nunca foi exercitado. Quando a política derivou para o giro, ela
    # não tinha autoridade nenhuma para sair.
    reset_base_manipula: dict = field(default_factory=lambda: {
        "x": (-0.10, 0.00), "y": (-0.10, 0.10), "z": (0.01, 0.05), "yaw": (-0.2, 0.2),
    })
    reset_base_loco: dict = field(default_factory=lambda: {
        "x": (-0.50, 0.50), "y": (-0.50, 0.50), "z": (0.01, 0.05),
        "yaw": (-3.14159, 3.14159),
    })
    # ⚠ REMOVIDO em 2026-08-26 (achado por code review): `reset_base_vel_manipula`
    # existia com o comentário "o robô chega andando, não parado" e NENHUM consumidor —
    # o `env_cfg` passa `velocidade: {}` ao `reset_base_por_elo`, logo a base sempre
    # resetou em repouso. Um knob morto num arquivo cuja premissa é "um treino tem de
    # ser reproduzível por `git diff` deste arquivo" é pior que ausente: ele descreve um
    # comportamento que não existe.
    #
    # A decisão de FATO é: a base reseta EM REPOUSO, nos dois modos. Se um dia se quiser
    # o robô chegando andando, o caminho é passar a faixa ao evento — e aí o knob volta,
    # com um consumidor.


@dataclass
class Alvo:
    """O alvo do objetivo. Altura ABSOLUTA de mundo: agachar não move o alvo,
    portanto o robô tem de ficar de pé.

    ⚠ Transcrito de `g1_poc/knobs.py::Alvo`. Os valores vêm da skill Lift, que
    fechou a tarefa com eles.
    """

    # A ÂNCORA DO PEITO, no frame da BASE. Ela é o alvo dos DOIS elos que seguram a
    # caixa, e a diferença entre eles é só o REFERENCIAL:
    #
    #     `pegar` e `carregar`  x,y RELATIVOS ao robô · z ABSOLUTO
    #
    # ⚠ Os dois alvos são EXATAMENTE IGUAIS. O que impede o robô de ANDAR com a caixa
    # durante o `pegar` NÃO é o alvo — é o **comando de velocidade em ZERO**. Decisão
    # do dono em 25/08, e é o que o `g1_poc` faz (`comando.py:826`), cuja manipulação
    # funcionou.
    #
    # ⚠ E o z é ABSOLUTO justamente para agachar não valer: um alvo relativo em z
    # desceria com a pelve, e o robô satisfaria agachando até a caixa em vez de erguer
    # a caixa até o peito.
    peito_b: tuple[float, float, float] = (0.25, 0.00, 0.15)

    # ⚠ A ALTURA DE TRABALHO, ABSOLUTA EM MUNDO. Ela é o z do alvo nos DOIS elos que
    # seguram a caixa, e o referencial é dividido POR EIXO:
    #
    #     x, y   RELATIVOS ao robô  — a caixa está nas mãos, acompanha horizontalmente
    #     z      ABSOLUTO           — agachar NÃO pode baixar o alvo
    #
    # Sem o z absoluto no `carregar`, o robô satisfaz o alvo ANDANDO AGACHADO: o alvo
    # desce junto com a pelve e a caixa nunca precisa subir. Foi a inconsistência que
    # o dono apontou em 25/08, e ela é a mesma classe do defeito do `pegar`.
    #
    # DERIVAÇÃO do 0,95: a pelve do keyframe joelhos-flexionados fica em z = 0,798
    # (MEDIDO), e `peito_b.z = 0,15`. Logo `0,798 + 0,15 = 0,948`. O `smoke` confere
    # esta soma contra a pose default do robô, para o número não derivar em silêncio.
    altura_carregar: float = 0.95

    # ⚠ NÃO EXISTE JITTER NO ALVO, e é decisão do dono em 25/08: o alvo do `pegar` e o
    # do `carregar` são **exatamente iguais**. Um jitter em y de ±0,05 sobre x = 0,25
    # deslocava o alvo até 11° fora do eixo do robô, e ele aparecia "de lado" no
    # viewer. A variedade do episódio vem da caixa (jitter em x, y e yaw) e do nível,
    # e não do alvo.

    # ⚠ HISTÓRICO, e o motivo de este bloco ter mudado em 25/08. O alvo do `pegar` era
    # ABSOLUTO E FIXO em `z = (0.78, 0.85)`, transcrito de
    # `g1_training/skills/lift/knobs.py:51`, cujo comentário é "acima do topo da mesa
    # => erguer". Com o `shelf_top = 0.55` FIXO da Lift, aquilo significava "erguer 13
    # a 20 cm da mesa" e estava certo.
    #
    # Mas aqui a laje varia de 0,55 a 0,04 por nível, e o alvo continuou absoluto:
    #
    #     nível 0  laje 0,55  caixa 0,65  ->  erguer 0,13–0,20 m
    #     nível 6  laje 0,04  caixa 0,14  ->  erguer 0,64–0,71 m
    #
    # O mesmo termo passou a significar 5× trabalhos diferentes, e o eixo `topo_min`
    # graduava DUAS coisas: de onde pegar e quanto erguer. Isso vazou, não foi
    # decidido. Medido na pose de pé: pelve 0,798, torso 0,842, cotovelo 0,909 — o
    # alvo de 0,78 ficava 6 cm acima das palmas em repouso (0,717), e ABAIXO da âncora
    # do `carregar` (0,95): o robô teria de erguer de novo depois de já ter "pego".

    # elo `botar`: deslocamento LATERAL. O frontal exigiria alcançar por cima de
    # 20 cm de tampo — defeito medido em 16/07.
    botar_x: tuple[float, float] = (0.30, 0.40)
    botar_y: tuple[float, float] = (-0.12, 0.12)
    botar_topo_piso: float = 0.30
    botar_topo_teto: float = 0.80
    # folga entre o topo NOVO da prateleira e o fundo da caixa segurada, no instante
    # em que o `pegar` fecha na cadeia `pegar` -> `botar`. Sem este teto efetivo a
    # laje nasceria DENTRO da caixa.
    botar_folga_laje: float = 0.05

    # ⚠ A JANELA DE ESPERA, em segundos, SORTEADA por episódio. Portada do `g1_poc`
    # (`knobs.py:366`) em 02/09 — a manipulação do g1_limpo foi inspirada nele e esta
    # peça não tinha vindo na reescrita.
    #
    # O QUE ELA FAZ: enquanto ela corre, o bit `VALIDA` fica em ZERO num elo de
    # manipulação. Os sete incentivos pagam nada, e o elo NÃO pode fechar. Na borda ela
    # vai 0->1 com a caixa já assentada, e essa DESCONTINUIDADE é o sinal de "o objetivo
    # chegou" (`g1_poc/comando.py:374-381`).
    #
    # ⚠ SORTEADA, e não fixa. Fixa é aprendível como "conte N passos e depois mova";
    # sorteada, a política TEM de ler o canal de comando — que é o que o deploy exige.
    #
    # ⚠ E NÃO existe canal de tempo restante na observação, de propósito: o `g1_poc`
    # publica só o bit. A política reage à descontinuidade em vez de contar.
    #
    # ⚠ SÓ NA MANIPULAÇÃO. No `ANDAR` ela é ZERO. O `g1_poc` a tirou da locomoção em
    # 24/08 porque atrasava o aprendizado da marcha; na manipulação, com episódio de
    # 800+ passos, ela custa ~4%.
    espera_s: tuple[float, float] = (0.3, 1.0)


@dataclass
class Piso:
    """Os pisos anti-esquecimento. Três, e um deles não custa knob.

    ⚠ O piso de NÍVEL não é o `rho = 0,30` do `g1_multitask`. Aquele era piso sobre
    TAREFAS, e com 5 tarefas ele ocupava 0,75 do sorteio — o teto da locomoção ficava em
    0,55 contra os 0,945 que a fatia de 30% exigia, e a fatia alvo virava inalcançável.
    Piso de NÍVEL e piso de FATIA são eixos ORTOGONAIS: o de nível não toca a divisão
    locomoção × cadeia.

    ⚠ O piso de ELO é ESTRUTURAL e não tem knob: toda cadeia de 2 elos passa pelo 1º,
    portanto não se esquece o `pegar` enquanto se treina o `botar`. Isso vale mais que
    qualquer piso de amostragem.
    """

    # fração dos envs sorteada UNIFORMEMENTE sobre os níveis abertos
    frac_nivel_uniforme: float = 0.20


@dataclass
class Nivel:
    """A tabela de células. Só o PISO desce; o TETO é fixo, portanto cada nível
    CONTÉM o anterior.

    ⚠ A tabela DISCRETA do `g1_multitask` (0,55 … 0,00) tem dois defeitos: no nível
    6 a laje fica ENTERRADA (centro em −0,02 m), e a altura fácil DESAPARECE do
    treino no instante da promoção.
    """
    n_niveis: int = 7
    topo_min: tuple[float, ...] = (0.55, 0.45, 0.30, 0.15, 0.04, 0.04, 0.04)
    carga_max: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0)
    # ⚠ O EIXO DO `reorientar` É EM QUARTOS DE VOLTA, e não em graus (decidido
    # 2026-08-26). Qualquer uma das 6 faces chega à frente por composição de quartos
    # de volta, portanto o robô precisa aprender 6 primitivas: ±90° em X, Y ou Z.
    #
    # A dificuldade é QUANTOS quartos de volta faltam:
    #   0 voltas  a face marcada já está à frente, só torta pelo desalinho
    #   1 volta   uma das 4 faces ADJACENTES
    #
    # ⚠ TETO DE UMA VOLTA (decidido 2026-08-26). A face marcada NUNCA nasce do lado
    # OPOSTO: o robô só precisa aprender a girar no máximo 90°. A primitiva atômica É
    # o quarto de volta; compor voltas não é alvo de treino, e sai de graça aplicando
    # a primitiva outra vez.
    #
    # E cada nível CONTÉM o anterior: o sorteio é uniforme em `0..voltas_max`.
    #
    # ⚠ O eixo VERTICAL (Y) entra depois do horizontal (Z), e a razão é física:
    # girar em Z é PIVOTAR sobre a laje, e dá para empurrar com uma mão; girar em Y é
    # TOMBAR, e exige erguer uma aresta de um cubo de 20 cm.
    voltas_max: tuple[int, ...] = (0, 0, 1, 1, 1, 1, 1)
    eixo_vertical: tuple[bool, ...] = (False, False, False, False, True, True, True)

    # ⚠ O eixo do `reorientar` SATURA no nível 4, e está declarado: acima dele o que
    # gradua é a altura da laje e a carga, não a orientação.

    # o desalinho residual, em graus. É a tarefa do nível 0: endireitar a caixa.
    # ⚠ Antes de 26/08 este eixo era `ang_max_deg = (0, 0, 0, 45, 90, 180, 180)`, e com
    # zero nos três primeiros níveis o `reorientar` ficava SATISFEITO em t = 0 — ele
    # não fazia nada em 3 dos 7 níveis.
    desalinho_max_deg: tuple[float, ...] = (15.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0)
    # ⚠ o jitter em x APERTA com o nível, e não é estética: com o topo a 0,04 m o
    # alcance de pose de pega acaba em x relativo ~0,45 m. Sem apertar, os níveis
    # altos sorteariam poses fora de alcance e a competência viraria sorte.
    jitter_x_max: tuple[float, ...] = (0.20, 0.20, 0.20, 0.15, 0.08, 0.08, 0.08)

    # nível FORÇADO. `None` no treino; o `inspeciona.py` e o `play.py` o usam.
    # ⚠ Forçar o buffer de fora NÃO funciona: o termo de currículo roda no reset e
    # aplicaria o passeio por cima. Tem de ser knob.
    forcado: int | None = None


@dataclass
class Recompensa:
    """Os pesos da LOCOMOÇÃO (F1). Um treino tem de ser reproduzível por `git diff`
    deste bloco, portanto a tabela é escrita INTEIRA — inclusive os pesos que hoje
    são idênticos ao default do `mjlab`.

    ⚠ Escrever o valor idêntico NÃO é redundância: é a trava contra deriva silenciosa.
    Um upgrade de `mjlab` que mude um default do molde passaria sem erro e sem log, e
    o `escala_acao_mult` já é o único número pré-registrado para mexer se o portão
    falhar — não pode haver um segundo suspeito entrando pela porta de trás.

    ⚠ `scale_rewards_by_dt = True` e `dt = 0,02 s`, portanto **o peso É o valor por
    segundo**. Duas consequências que já custaram sessões:
      · `terminacao = −200` NÃO é −200. É `−200 × 0,02 = −4,0` de uma vez, no passo
        que termina. Contra os ~5,0/s do teto positivo, cair custa ~0,8 s de tudo.
      · `joint_acc = −2,5e−7` é desprezível de propósito: MEDIDO em 0,0006/s. Ele
        entra por paridade com o módulo que andou, e não por efeito.
    """

    # --- os positivos ---
    track_linear_velocity: float = 2.0
    track_angular_velocity: float = 2.0
    upright: float = 1.0
    pose: float = 1.0                     # `variable_posture`, σ COLHIDOS do molde

    # ⚠ O ÚNICO termo POSITIVO de marcha, e o fabricante o entrega em ZERO. Fica em
    # zero na F1, e é decisão declarada, não descuido: mexer nele no mesmo bloco que
    # o `escala_acao_mult` confundiria os dois. MEDIDO nos dois módulos: ausente no
    # que andou, 0,0 no que não andou — portanto ele NÃO é a explicação do andar.
    #
    # ⚠ E peso 0 apaga o `Metrics/air_time_mean` em silêncio (`reward_manager.py:122`
    # pula termo com peso zero). É por isso que a métrica migra para o
    # `metricas.py` — ver lá.
    air_time: float = 0.0

    # --- os freios do molde ---
    action_rate_l2: float = -0.1          # −0,10 destravou o andar num bloco medido
    dof_pos_limits: float = -1.0
    foot_clearance: float = -2.0
    foot_swing_height: float = -0.25
    foot_slip: float = -0.1
    soft_landing: float = -1e-5
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    self_collisions: float = -1.0

    # --- os dois termos NOVOS, do `g1_multitask` (o módulo que andou) ---
    terminacao: float = -200.0
    joint_acc: float = -2.5e-7

    # --- a MULTA DE CONTATO COM A MESA (01/09), uma por parte do corpo ---
    # ⚠ O PESO É DERIVADO, e não escolhido. O que escorar COMPRA é alcançar sem pagar
    # postura, e quem paga postura é o `postura_ereta`, com peso 2,0. Com −2,0 aqui:
    #
    #     pega ERETA      +2,0 de postura_ereta,   0,0 de multa   ->  +2,0
    #     pega ESCORADA    0,0 de postura_ereta,  −2,0 de multa   ->  −2,0
    #
    # Quatro pontos de diferença, e continua PAGÁVEL: 2,0/s contra um teto de tarefa de
    # 11,5/s são 17%, não um penhasco. A terminação que isto substitui custava −4,0 no
    # passo MAIS o retorno restante do episódio, da ordem de 90 — ~45x mais pesada.
    #
    # ⚠ TRÊS PESOS IGUAIS, e não um termo só. A partição por parte é MEDIÇÃO: ela é a
    # única coisa que separa "a coxa bateu na quina em pé" de "o tronco mergulhou", e as
    # duas pedem consertos opostos.
    contato_tronco: float = -2.0
    contato_palma: float = -2.0
    contato_dorso: float = -2.0

    # ⚠ O `contato_prateleira` SAIU em 27/08, e virou a terminação `contato_ilegal`
    # (`Terminacao.contato_ilegal_N`). O bloco 2 rodou 405 iterações com ele como
    # penalidade de −1,5 e o resultado decidiu: o contato do tronco caiu monotonicamente
    # (7,5% -> 3,8% -> 2,0% dos passos) e a manipulação caiu junto (`staged` 0,36 ->
    # 0,17). Uma multa que o robô pode pagar é uma multa que ele ORÇA — e com o
    # `action_rate_l2` em −2,04 escorar sai mais barato que se mover. O princípio do
    # g1_poc é o certo: terminar em vez de penalizar.
    #
    # ⚠ O `com_over_feet` do lift também NÃO entra, e a decisão é medida: ele é
    # `clamp(deriva − 0,05, min=0)²`, logo um mergulho de 30 cm custa 0,25² × 2,0 =
    # 0,125/s contra 11,5/s de tarefa — 1,1%. Com o peso padrão ele já é quase inerte;
    # mais fraco seria inerte. O g1_poc passou sem ele por isso, não por sorte.

    # o alvo do `foot_swing_height`, em metros. Do molde.
    altura_de_balanco: float = 0.10


@dataclass
class Marcha:
    """As duas réguas da locomoção. Adimensionais de propósito.

    **JUIZ (desde 27/08) — `eficiencia_min`.** Por SEGMENTO de comando:

        e_s = ⟨v_real · v̂_cmd⟩_s / ⟨‖v_cmd‖_s⟩ ,     portão = min(e_s)

    "Em média, que fração da velocidade comandada você entregou na direção comandada,
    no pior segmento?" 1,0 = entregou tudo. Negativo = foi para o lado errado.

    **DIAGNÓSTICO — `razao_marcha`.** `1 − Σ‖v_cmd_xy − v_xy‖ / Σ‖v_cmd_xy‖`.

    ⚠ POR QUE A TROCA, e é medição e não gosto. A `razao_marcha` é soma de NORMAS, e
    norma nunca cancela: com `v_real = v_politica + ruído`,
    `Σ‖(v_cmd − v_politica) − ruído‖ > Σ‖v_cmd − v_politica‖` para ruído de média zero.
    Logo ruído SEMPRE a infla, monotonicamente. No bloco 1 o `std` subiu de 0,43 (it
    1525) para 0,61 (it 4999) — porque a manipulação entrou e exploração voltou a valer
    — e a razão caiu de 0,514 para 0,426, enquanto DURAÇÃO (984 -> 988) e QUEDA (0,000 ->
    0,167) NÃO se moveram e o `play` determinístico andava bem. O portão congelou na
    banda morta e a rampa deu UM degrau em 1341 iterações.

    A projeção conserta pela forma: `Σ(v_real · v̂_cmd) = Σ(v_politica · v̂_cmd) +
    Σ(ruído · v̂_cmd)`, e o segundo termo tem média zero e encolhe com 1/√N.

    ⚠ E o segmento, em vez do episódio, fecha o outro buraco: média sobre o episódio
    inteiro deixaria o robô TROCAR TEMPO (parar 10 s e compensar depois). Cada segmento
    é pontuado sozinho.

    ⚠ AS DUAS nascem PESSIMISTAS em 0,0: robô imóvel projeta zero e tem erro igual ao
    comando. É o oposto do portão que media sobrevivência, que dava nota máxima à
    estátua.
    """

    # abaixo disto o comando conta como "parado" e o passo NÃO entra em soma nenhuma —
    # nem das normas, nem da projeção. Sem o gate, um comando de 0,001 m/s inflaria a
    # razão de graça, e o fabricante põe 10% dos envs em `is_standing_env`.
    limiar_comando: float = 0.05

    # piso de VALIDADE do segmento (não é alvo). Ver o docstring do campo homônimo em
    # `comando.TwistComRazaoDeMarchaCfg` para a derivação.
    pedido_min_segmento: float = 0.5


@dataclass
class Forma:
    """A fatia entre locomoção e manipulação. F5 põe o controlador.

    ⚠ `fatia_loco` NUNCA é 1,00, e o motivo não é o treino — é o NORMALIZADOR. Com
    1,00 os slots de manipulação do one-hot são constantes em zero, e
    `rsl_rl/modules/normalization.py:48` calcula `(x − _mean) / (_std + 1e−2)`, sem
    clamp. Com o canal constante, `_std -> 0` e `_mean -> 0`: ao acender, 1,0 entra na
    rede como **100,0**. Com 0,95, 5% dos episódios são de manipulação desde o passo 0
    e os slots sorteáveis nunca são constantes.

    E 0,95 não contraria "locomoção primeiro": a hipótese validada é sobre não
    entregar 70% das transições à manipulação com o robô imóvel, e não sobre a
    diferença entre 95% e 100%.
    """

    fatia_loco: float = 0.95

    # =========================================================================
    # O CONTROLADOR (F5)
    # =========================================================================
    # ⚠ `alvo_loco` NÃO é probabilidade de sorteio. É a fatia de TRANSIÇÕES alvo, e o
    # sorteio é RESOLVIDO das durações medidas:
    #
    #     f = alvo·Tm / (Tl·(1−alvo) + alvo·Tm)
    #
    # O sorteio é por EPISÓDIO e o PPO aprende por TRANSIÇÃO. Confundir os dois é a
    # armadilha medida deste projeto: com `Tl = 24` e `Tm = 961`, um sorteio de 0,30
    # entrega **1,06%** do gradiente à locomoção. Para entregar 0,30 de verdade é
    # preciso sortear 0,9449.
    #
    #   Tl    Tm    sorteio 0,30 entrega    para entregar 0,30, sortear
    #   24   961          1,06%                      0,9449
    #  150   500         11,4%                       0,5882
    #  400   500         25,5%                       0,3488
    # 1000   500         46,2%                       0,1765
    controla: bool = True
    alvo_loco_max: float = 0.95     # o piso inicial. Ver `fatia_loco`.
    alvo_loco_min: float = 0.30     # o destino
    alvo_passo: float = 0.02        # 33 degraus de 0,95 a 0,30
    iters_entre_degraus: int = 12   # => >= 396 iterações de rampa
    # clamps do SORTEIO resolvido (não do alvo)
    sorteio_min: float = 0.10
    sorteio_max: float = 0.95

    # ⚠ UM SINAL SÓ NO PORTÃO, e adimensional. Dois sinais conjuntivos já travaram uma
    # rampa para sempre: o `erro_giro_ema <= 0,30` ficou plano em 0,587 por 390
    # iterações enquanto a `razao_giro` marcava 0,373. O sinal é o do fabricante: o
    # `terrain_levels_vel` rebaixa quem anda menos de METADE da velocidade comandada.
    #
    # ⚠⚠ O SINAL MUDOU EM 27/08: era a `razao_marcha`, agora é a `eficiencia_min`. A
    # `razao_marcha` continua logada como DIAGNÓSTICO, e saiu do portão porque a forma
    # dela não sobrevive a ruído de ação: `Σ‖v_cmd − v_real‖` é soma de NORMAS, e norma
    # nunca cancela, portanto ruído de média zero SEMPRE a infla. MEDIDO no bloco 1: o
    # `std` subiu de 0,43 para 0,61 e a razão caiu de 0,514 para 0,426, enquanto DURAÇÃO
    # (984 -> 988) e QUEDA (0,000 -> 0,167) não se moveram e o `play` determinístico
    # andava bem. O portão congelou em 0,426 na banda morta e a rampa deu UM degrau em
    # 1341 iterações — ele leu ruído como incompetência.
    #
    # A `eficiencia_min` é a projeção `Σ(v_real · v̂_cmd)` por SEGMENTO de comando: o
    # ruído entra com média zero e encolhe com 1/√N, e cada segmento é pontuado sozinho,
    # o que impede o robô de compensar um segmento ruim com outro bom.
    limiar_portao: float = 0.50
    """⚠ PROVISÓRIO E NÃO MEDIDO na escala nova. O 0,50 vinha da `razao_marcha`, e
    coincide de escala: "entregar metade da velocidade comandada no PIOR segmento" é uma
    barra defensável. Mas ninguém mediu quanto a política que o `play` mostrou andando
    bem marca aqui — só o bloco 2 dirá, porque as duas curvas ficam no log lado a lado.
    Calibrar contra o log, não contra opinião."""
    # ⚠ ASSIMÉTRICO de propósito: lento para avançar, rápido para defender.
    histerese: float = 0.80         # devolve fatia se o sinal cai abaixo de 0,80×limiar
    # ⚠ Contada de quando o BALANÇO COMEÇOU, nunca de passo global absoluto.
    carencia_iters: int = 200

    # as EMAs. ⚠ O estado INICIAL é deliberadamente ASSIMÉTRICO:
    #   as DURAÇÕES nascem NEUTRAS (episódio cheio) — elas governam a FATIA, e um erro
    #   ali só desafina o sorteio por ~tau;
    #   o SINAL DO PORTÃO nasce PESSIMISTA em 0,0 — ele governa a entrega da fatia, e um
    #   portão que nasce aprovando entrega a locomoção ANTES de existir marcha. Foi
    #   exatamente o que a `dur_loco_ema` neutra em 1000 passos fez. Vale igual para a
    #   `eficiencia_min`: robô imóvel projeta 0, logo ela também nasce em 0,0.
    ema: float = 0.99
    dur_inicial_passos: float = 1000.0

    # ⚠⚠ PASSOS POR ITERAÇÃO DE PPO. Sem isto a carência e a rampa contam a coisa
    # errada, e foi um defeito MEDIDO em 2026-08-26 (achado por code review).
    #
    # O termo de currículo roda em `curriculum_manager.compute`, que o `_reset_idx`
    # chama — e o `_reset_idx` roda a CADA PASSO em que ALGUM env reseta. Com os
    # episódios dessincronizados isso é quase todo passo: medido com 128 envs, o
    # contador subiu para 48,8% dos passos em 400 passos, e com 4096 envs tenderia a
    # 100%.
    #
    # Portanto um `contador += 1` conta PASSOS, não iterações. Consequência medida: a
    # carência de "200 iterações" era atingida em ~17 iterações de PPO, e a rampa de
    # "396 iterações" em ~34. A fatia colapsaria de 0,95 para 0,30 em algumas dezenas
    # de iterações — exatamente a falha que este módulo existe para evitar.
    #
    # O conserto: derivar a iteração de `env.common_step_counter`, que conta PASSOS
    # (`manager_based_rl_env.py:431`) e que o mjlab JÁ PERSISTE no checkpoint
    # (`mjlab/rl/runner.py:73`) — o que torna a rampa resume-safe de graça.
    #
    # 24 é o `num_steps_per_env` do PPO do fabricante. O `smoke` confere contra o cfg.
    passos_por_iteracao: int = 24

@dataclass
class Tarefa:
    """Os sete incentivos da manipulação. TODOS positivos e contínuos (R3).

    ⚠ Nenhuma penalidade aqui. Penalidade limita COMO fazer o que já existe; ela não
    ensina a fazer. E booleano é platô — o `pegar` do `g1_poc` travou 22k iterações num
    `squeeze` booleano.

    Soma dos pesos = 11,5/s. É o teto da tarefa, e ele se compara com o PISO DA ESTÁTUA
    de **5,81/s** (medido 2026-08-26, robô travado num elo parado). Razão ~2:1 no
    fecho completo, e é a resposta à pergunta "ficar parado paga mais que agir?".
    """

    # --- os sete pesos ---
    staged: float = 3.0            # alcançar × (1 + trazer). O motor da fase inicial
    precise_pos: float = 2.0       # caixa NO alvo
    precise_ori: float = 1.0       # face pedida apontando ao robô
    squeeze: float = 1.0           # força nas DUAS palmas
    unload: float = 2.0            # a caixa deixou de pesar na laje
    postura_ereta: float = 2.0     # ergueu SEM agachar
    sustentacao: float = 0.5       # ficou lá

    # --- σ: NÃO SÃO NÚMEROS, SÃO A DISTÂNCIA INICIAL ---
    #
    # ⚠ ESTE É O ITEM DE MAIOR RISCO DA F3, e ele é medido. A palma nasce a 0,339 m da
    # caixa (mín 0,211, máx 0,481). Com σ FIXO de 0,10 o kernel `exp(−d²/σ²)` vale
    # 1e−05 ali, **e a derivada é ZERO**: o robô move a mão 1 cm para perto e nada
    # muda, 1 cm para longe e nada muda. Não existe pista. Foi isto que travou o
    # `g1_poc` — não uma preferência por ficar parado.
    #
    # Com `σ = d₀`, todo env nasce em `exp(−1) = 0,368` com derivada `2/d₀ × 0,368`:
    # 3,49 no env mais perto e 1,53 no mais longe. Vivo nos dois extremos, e sem
    # número mágico.
    #
    # O fator multiplica a distância medida. 1,0 é a derivação limpa; >1 alarga.
    #
    # ⚠ PRÉ-REGISTRADO: se o alcance não aparecer, este fator é o PRIMEIRO e ÚNICO
    # número a mover (para 1,5). NUNCA o peso — tornar o 1º centímetro positivo
    # exigiria peso > 12, quatro vezes o da locomoção, e o robô pararia de andar.
    sigma_fator: float = 1.0
    # piso do σ, em metros. Sem ele um env que nasce com a palma colada na caixa teria
    # σ ~ 0 e o kernel viraria um pico impossível de manter.
    sigma_min: float = 0.08

    # ⚠ O `precise_pos` é o ÚNICO com σ FIXO, e de propósito: ele mede "a caixa está NO
    # alvo", que é uma tolerância de aceite e não uma rampa de aproximação. Quem faz a
    # aproximação é o `staged`, com σ por env.
    precise_pos_sigma: float = 0.05

    # --- squeeze: o μ que DERIVA a força de referência das palmas ---
    # ⚠ `F_ref = m·g/(2μ)` — a força de aperto que o atrito precisa para segurar a
    # caixa. Com m = 1,0 kg e μ = 0,8 dá **6,13 N**.
    #
    # ⚠ ERA UM KNOB FIXO DE 12,0 N até 28/08, e o número não tinha derivação. Ele pedia
    # o DOBRO do que a física precisa e pagava METADE no primeiro newton — justo a faixa
    # em que a preensão tem de nascer. O `g1_poc` usa a conta, e não o número.
    #
    # ⚠ `tanh(min(F_n_E, F_n_D)/F_ref)` e não booleano. O `min` exige as DUAS mãos: uma
    # palma sozinha empurra a caixa, não a segura. E o `tanh` é contínuo desde a
    # primeira décima de newton — um limiar booleano é platô, e o platô travou o
    # `pegar` do g1_poc por 22k iterações.
    squeeze_mu: float = 0.8              # μ pessimista da faixa de atrito da caixa

    # --- postura ereta: a rampa dupla na pelve ---
    # ⚠ Ela paga por erguer SEM agachar, e é o que impede o robô de satisfazer o alvo
    # descendo até a caixa. Medido na pose de pé: pelve em 0,798 m.
    pelve_alvo: float = 0.75       # acima disto a rampa paga cheio
    pelve_piso: float = 0.45       # abaixo disto ela paga zero

    # --- sustentação ---
    # ⚠ O cronômetro lê SÓ a condição da tarefa. No `g1_multitask` ele lia também o
    # erro angular, e um push de ±0,78 rad/s a cada 1-3 s ZERAVA o contador: o
    # `perf` ficou 0 nas iterações 13.700 e 17.297 com o robô já andando. Push e régua
    # em compartimentos separados.
    sustenta_s: float = 1.0
    # a tolerância que conta como "na condição", em metros e radianos
    tol_pos: float = 0.10
    tol_ang_deg: float = 25.0


@dataclass
class Cadeia:
    """A tabela de cadeias de elo, fase F4.

    ⚠ TETO DE 2 ELOS. As cadeias são:
      índice 0: (PEGAR,)                 -> 1 elo (cadeia curta da F3)
      índice 1: (REORIENTAR, PEGAR)
      índice 2: (PEGAR, CARREGAR)
      índice 3: (PEGAR, BOTAR)

    O `pegar` aparece em TODAS: ele é o eixo de que não se esquece.
    """

    # [7 níveis × 4 cadeias] de probabilidades. Cada linha soma 1,0.
    # Nível baixo concentra na cadeia de 1 elo (índice 0);
    # nível alto abre as de 2 elos.
    prob_por_nivel: tuple[tuple[float, ...], ...] = (
        # Nível 0: cadeia curta domina (1 elo).
        # Racional: robustecer a pega antes de transições.
        (0.80, 0.10, 0.05, 0.05),
        # Nível 1: ainda principalmente cadeia curta.
        (0.75, 0.10, 0.10, 0.05),
        # Nível 2: distribui mais para 2 elos; reorientar entra.
        (0.60, 0.20, 0.10, 0.10),
        # Nível 3: equilibrado entre 1 e 2 elos; todas as cadeias.
        (0.40, 0.25, 0.20, 0.15),
        # Nível 4 e acima: favorece cadeias de 2 elos (transições).
        # Racional: com dificuldade física alta, a transição é o aprendizado.
        (0.20, 0.25, 0.30, 0.25),
        # Nível 5: mais 2 elos ainda.
        (0.15, 0.25, 0.35, 0.25),
        # Nível 6: máxima diversidade, 2 elos dominam.
        (0.10, 0.25, 0.35, 0.30),
    )

    # Tempos de sustentação (em segundos) quando o elo fecha.
    # PEGAR exige menor sustain (mais rápido em fechar e transicionar).
    # ⚠ O `pegar` exige o sustain MAIOR (0,5 s contra 0,3 s), e é decisão: ele é o elo
    # do qual todas as cadeias dependem, e um fecho por acidente de um frame
    # propagaria para os outros três. Um comentário anterior dizia o contrário do que
    # os valores fazem.
    sustenta_pegar_s: float = 0.5
    # ⚠ O comentário anterior aqui dizia "outros elos têm sustain MAIOR" com 0,3 contra
    # os 0,5 do `pegar` — dizia o contrário do que os valores fazem. O `pegar` tem o
    # sustain MAIOR de propósito: ele é o elo do qual TODAS as cadeias dependem, e um
    # fecho por acidente de um frame propagaria para os outros três.
    sustenta_outros_s: float = 0.3

    # ⚠⚠ O `CARREGAR` EXIGE DESLOCAMENTO, e não só tempo. Defeito MEDIDO em 2026-08-26
    # (achado por code review): o `pegar` e o `carregar` publicam EXATAMENTE o mesmo
    # alvo (é decisão, §4.2), e as condições de fechamento eram
    #
    #     PEGAR     perto & alinhado & de_pé
    #     CARREGAR  perto
    #
    # `perto` é SUBCONJUNTO de `perto & alinhado & de_pé` sobre um alvo que não muda.
    # Portanto no instante em que o `pegar` fechava, o `carregar` JÁ estava satisfeito:
    # o robô ficava parado 1,5 s e a cadeia era marcada como SUCESSO — o que move o
    # currículo de nível. A cadeia `pegar -> carregar` treinava "não andar".
    #
    # DERIVAÇÃO do 0,50 m: a faixa de comando do fabricante vai a 1,0 m/s e o portão da
    # F1 exige rastrear METADE dela, logo um robô aprovado cobre ~0,5 m em 1,0 s. Com
    # `carregar_s = 1,5 s` o pedido é conservador — não exige mais do que a locomoção já
    # provou fazer.
    carregar_dist_m: float = 0.50
    # o PISO de tempo do carregar (tempo mínimo andando), não um teto
    carregar_s: float = 1.5


@dataclass
class Contato:
    """A RAMPA da multa de contato com a mesa. Os PESOS vivem em `Recompensa`.

    ⚠⚠ ISTO ERA TERMINAÇÃO ATÉ 01/09, e a troca é decisão do dono apoiada em medição.
    Com a terminação, 76% dos episódios de manipulação morriam na mesa e ficar parado
    rendia 90 de retorno contra 66 de tentar. Pior: o `play` mostrou que a ação MÉDIA
    nem se aproximava da mesa — aqueles 76% eram RUÍDO de exploração encostando, e a
    terminação matava a exploração antes de ela refinar a pega.

    MEDIDO depois da troca, no bloco 7: `descarga` (caixa fora da laje) foi de 0,0 a
    0,994, `palmas_em_contato` de 0,09 a 0,63, e o `postura_ereta` — que exige pelve
    alta E preensão E descarga ao mesmo tempo — saiu de zero pela primeira vez.

    ⚠ E o precedente contrário NÃO se aplicava. O `contato_prateleira = −1,5` do bloco 2
    caiu por medição, mas rodou num sistema com quatro defeitos desde então consertados:
    piso da estátua em 8,265/s, alcance `min` sobre esfera, `unload` sem porteiro, e a
    lista da mesa invertida. A conta dele nem fecha: −1,5 sobre 7,5% dos passos são
    −0,11/s contra um teto de 11,5/s.

    ⚠ A FORMA É RAMPA, e não booleano. Booleano é platô — o defeito que travou o
    `squeeze` por 22 mil iterações — e apaga a distinção que o limiar existe para fazer:
    roçar não é escorar.
    """

    joelho_N: float = 50.0
    """Abaixo desta força a multa é ZERO. Roçar o tampo ao alcançar sai de graça.

    ⚠ MEDIDO no `g1_poc` (`knobs.py:328`), onde a mesma força governava a terminação.
    Não é número novo: é o mesmo joelho, com rampa no lugar do penhasco."""

    saturacao_N: float = 100.0
    """Acima desta força a multa satura no peso cheio.

    ⚠ O DOBRO do joelho, e a razão é a semântica: de 50 a 100 N existe gradiente para
    TIRAR o peso da mesa. Sem a faixa, o robô não teria pista de que aliviar ajuda."""


@dataclass
class Terminacao:
    """As terminações próprias. `time_out` e `fell_over` vêm do molde.

    ⚠ Este bloco NASCEU em 27/08. Até então o g1_limpo não tinha terminação própria
    nenhuma, e o princípio "terminar em vez de penalizar" do g1_poc tinha sido perdido na
    reescrita junto com os termos.

    ⚠ O CONTATO COM A MESA SAIU daqui em 01/09 e virou multa — ver `Contato`. O que
    sobra é a caixa largada, que continua terminação porque não há como "pagar" por ter
    perdido a caixa: com ela no chão, a tarefa acabou.
    """

    caixa_z_min: float = 0.10
    """Altura da caixa, relativa à origem do env, abaixo da qual ela CAIU.

    ⚠ 0,10 m é a meia-aresta: com o centro nessa altura a caixa está apoiada no CHÃO.
    Ela é o piso físico, e não uma tolerância escolhida."""

    caixa_dist_max: float = 0.45
    """Distância, em metros, de AMBAS as palmas ao centro da caixa para ela ter escapado.

    ⚠ A palma nasce a 0,339 m da caixa (mín 0,211, máx 0,481). Portanto este limiar é
    MAIOR que a distância de nascimento típica — e mesmo assim a terminação não dispara
    no reset, porque ela é armada pela primeira preensão. Os dois freios são
    independentes de propósito."""


@dataclass
class Knobs:
    cena: Cena = field(default_factory=Cena)
    alvo: Alvo = field(default_factory=Alvo)
    nivel: Nivel = field(default_factory=Nivel)
    recompensa: Recompensa = field(default_factory=Recompensa)
    marcha: Marcha = field(default_factory=Marcha)
    forma: Forma = field(default_factory=Forma)
    piso: Piso = field(default_factory=Piso)
    tarefa: Tarefa = field(default_factory=Tarefa)
    cadeia: Cadeia = field(default_factory=Cadeia)
    terminacao: Terminacao = field(default_factory=Terminacao)
    contato: Contato = field(default_factory=Contato)


ATIVO = Knobs()
