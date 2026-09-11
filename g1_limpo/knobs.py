"""Todos os números do g1_limpo, num lugar só.

Regra do pacote: nenhum número solto no meio do código. Um treino tem de ser
reproduzível por `git diff` deste arquivo.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Este pacote não importa `g1_training`, nem
`g1_poc`, nem `g1_multitask`. Os valores abaixo foram TRANSCRITOS à mão dessas
referências, e o `paridade.py` existe para provar que a transcrição está correta.

Ver `specs/g1-limpo.md` e `docs/planos/2026-08-25-g1-limpo.md`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Cena:
    """Geometria da cena. Transcrito de `g1_training/common/box.py` e
    `g1_poc/knobs.py::Cena`."""

    # --- caixa (corpo LIVRE, erguível) ---
    caixa_meia_aresta: tuple[float, float, float] = (0.10, 0.10, 0.10)
    # ⚠ DR DE TAMANHO desde a FASE 1 (spec §6.7, decisão do dono 02/09). K meio-lados
    # discretos, cubo, sorteados por env UMA vez no startup e escritos por mundo em
    # `geom_size` + `geom_rbound` + `geom_aabb` pelo caminho de DR do mjlab. O
    # `caixa_meia_aresta` acima segue sendo o spec de REFERÊNCIA (paridade, inspetor) e
    # a variante do meio da faixa.
    caixa_meia_aresta_faixa: tuple[float, float] = (0.07, 0.13)
    caixa_n_variantes: int = 8
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
    # posição da laje no RESET (e no CARREGAR/ANDAR, afastada em z por `afasta_z`).
    # ⚠ o avanço do BOTAR NÃO lê este knob (spec dois-bits §1.4, revisão do
    # coordenador): usa a constante de módulo `comando._AVANCO_LAJE_BOTAR`, medida
    # à parte — ver o comentário lá. Subir ESTE knob afastaria a caixa do PEGAR em
    # todo nível, e não é o que a spec pediu.
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
    #
    # MEDIDO 2026-09-08 (spec dois-bits §1.2, sonda `sonda_peito_bx.py`,
    # `model_6999`, cadeia B, 64 envs, 1100 passos, CPU): `caixa_b.x` no HOLD
    # (`pegou = True`), por faixa de `meia_aresta` (0,070 a 0,130 m):
    #
    #     meia_aresta   0,070  0,079  0,087  0,096  0,104  0,113  0,121  0,130
    #     p50 caixa_b.x 0,228  0,237  0,238  0,248  0,257  0,259  0,270  0,272
    #
    # GERAL p50 = 0,254 (n=45521). O candidato `0,10 + meia_x` da spec SUBESTIMA em
    # ~0,05 m em toda faixa (a caixa segura fica mais à frente que isso). O valor
    # atual (0,25) já bate com o medido — mantido, sem ajuste.
    #
    # ⚠ ADENDO — `caixa_b.z` MUNDIAL (revisão do coordenador): mesma sonda, geral
    # (n=44988), `root_link_pos_w[:,2] − env_origin_z` no HOLD: p10 0,913, p50
    # 1,025, p90 1,076. `caixa_b.y`: p10 −0,046, p50 −0,020, p90 0,054 — centrado,
    # sem desvio. O desvio de ~10 cm medido antes (relógio) era em Z, não em x/y:
    # o alvo a `altura_carregar = 0,95` ficava 7,5 cm ABAIXO de onde a caixa
    # repousa de verdade no peito. `peito_b.z` sobe de 0,15 para 0,222 (ver
    # `altura_carregar`, abaixo, para a derivação).
    peito_b: tuple[float, float, float] = (0.25, 0.00, 0.222)

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
    # DERIVAÇÃO (revisão do coordenador, MEDIDO 2026-09-08): a pelve do keyframe
    # joelhos-flexionados fica em z = 0,798, e `peito_b.z = 0,222` (medido, ver
    # acima). Logo `0,798 + 0,222 = 1,020`. O `smoke` confere esta soma contra a
    # pose default do robô, para o número não derivar em silêncio.
    #
    # ⚠ POR QUE SUBIU de 0,95: `caixa_b.z` medido no HOLD (sonda `sonda_peito_bx.
    # py`, n=44988) dá p10 0,913, p50 1,025, p90 1,076 — o alvo antigo (0,95)
    # ficava 7,5 cm ABAIXO de onde a caixa de fato repousa no peito. Esse desvio
    # em z (não em x/y, que estão centrados) era o que fazia `perto` oscilar no
    # limiar do fecho.
    altura_carregar: float = 1.02

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

    # ⚠ elo `botar` (spec g1-limpo-dois-bits.md §1.4): o topo NOVO nasce PERTO do topo
    # ATUAL (`limpo_topo`), não sorteado numa faixa absoluta — `botar_delta_topo` é a
    # excursão máxima, pra cima ou pra baixo. O alvo é LATERAL, na BORDA perto do
    # robô: `botar_delta_xy` espalha o ponto de pouso sobre a laje, e
    # `botar_recuo_borda` recua o alvo do centro para a borda — alcançar por cima do
    # tampo inteiro era o defeito medido em 16/07.
    botar_delta_topo: float = 0.10
    botar_delta_xy: float = 0.10
    botar_recuo_borda: float = 0.15
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
    # ⚠ TODA espera é a MESMA faixa (spec §6.3, §6.5): a espera inicial em ANDAR
    # publicado antes do PEGAR, e o "segurar parado" do CARREGAR da cadeia 3. Sorteada
    # para a política não contar passos. Decisão do dono, 02/09 (quarta rodada).
    espera_s: tuple[float, float] = (0.5, 1.5)


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
    # ⚠ O REORIENTAR ESTÁ INERTE NA RUN DA v2 (spec §8.3, decisão do dono 03/09): a
    # caixa nasce sempre dentro da tolerância de fechamento e o elo fecha em 0,3 s sem
    # trabalho. O slot continua sorteado (senão o normalizador o vê constante). Para o
    # cubo isto não muda nada em PEGAR, CARREGAR ou BOTAR. Quando a reorientação virar
    # foco, a tabela sai do nível e vai para um bloco `Reorientar` próprio (spec §8.3):
    #     voltas_max     = (0, 0, 1, 1, 1, 1, 1)       <- a de antes, por nível
    #     eixo_vertical  = (False, False, False, False, True, True, True)
    voltas_max: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0)
    eixo_vertical: tuple[bool, ...] = (False,) * 7

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
    pose: float = 1.0                     # `PosturaPorElo` (spec dois-bits §3.1)

    # ⚠ `std_standing` NÃO é mais COLHIDO do fabricante (spec dois-bits §3.1). O do
    # molde é `{".*": 0,05}`, uma entrada só apertada demais: a 10% da faixa de
    # junta o termo já vale 0,000, com GRADIENTE ZERO — canal morto, não penalidade
    # forte (medido em `recompensas.PosturaPorElo`). Este dict é calibrado para o
    # divisor REAL — as 21 juntas que sobram quando o braço sai da conta
    # (`pegou ∧ ¬soltou`): as 15 de perna+cintura mais os 6 punhos, que desde a v3.5
    # FICAM na média (spec `g1-limpo-cauda-parada-de-pe.md` §2.4).
    #
    # ⚠ MEDIDO 2026-09-08 (revisão independente, item A13), sem env, com o divisor
    # de 15 juntas: `exp(−média) = 0,92` a 0,1 rad de excursão uniforme (limiar era
    # `>= 0,8`); `exp(−média) = 0,051` a 0,6 rad (limiar era `<= 0,3`). Os dois
    # folgam a barra — a tabela abaixo não é mais ponto de partida.
    std_standing: dict = field(default_factory=lambda: {
        r".*hip_yaw.*": 0.30, r".*hip_roll.*": 0.30,
        r".*hip_pitch.*": 0.50, r".*knee.*": 0.50, r".*ankle_pitch.*": 0.50,
        r".*ankle_roll.*": 0.30,
        r".*waist.*": 0.30,
        r".*shoulder.*": 1.00, r".*elbow.*": 1.00,
        # ⚠ 1,00, o MESMO de `shoulder` e `elbow`, e é REVERSÃO MEDIDA do 0,30 da v3.5.
        # O 0,30 se justificava assim: "com 1,00 um punho a 57° custa 3,4% do `pose`".
        # A aritmética estava certa e a conclusão errada — o robô não opera a 57° com
        # UM punho, opera a ~1,6 rad (92°) com SEIS. Com 0,30 os seis sozinhos somam
        # `6 × (1,6/0,30)² = 170,7` ao expoente; dividido pelo divisor de 21 dá 8,13, e
        # `exp(−8,13) = 0,0003`.
        #
        # ⚠ MEDIDO em `model_10200`, na janela `pegou ∧ ¬soltou`: `pose` vale **0,0272**
        # de um teto de 1,0 — 2,7% — com `|Δvalor|` por passo p50 de `3e−05`. CANAL
        # MORTO, derivada zero: o termo que moldaria o gesto da pega não existia ali. E
        # o `pose` é a média de TODAS as 21 juntas ativas — matá-lo soltava o punho E a
        # perna. No mesmo checkpoint, `right_wrist_roll` a 2,034 rad contra o limite
        # mole de 1,972, e os dois `wrist_yaw` a 1,621 contra 1,614: o punho ia ao
        # batente mecânico, que é o oposto do que o 0,30 pretendia.
        #
        # ⚠ Com 1,00 e o MESMO erro de 1,6 rad, os seis somam 15,4; /21 = 0,73, e
        # `pose = exp(−0,73) = 0,48`. VIVO, com derivada.
        #
        # ⚠ O que NÃO muda, de propósito: o punho SEGUE dentro da média do `pose` (a
        # máscara do `PosturaPorElo` está certa — era o σ que estava errado), e o
        # pedido do dono, "a mão sempre alinhada com o antebraço", segue atendido, e
        # agora com gradiente para atendê-lo. `std_walking` também não muda: o regime
        # da pega é o `standing` — o twist é forçado a zero no `PEGAR` —, então
        # `std_standing` é o alvo exato.
        r".*wrist.*": 1.00,
    })

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

    # ⚠ GIRAR NO LUGAR (spec §9, decisão do dono 02/09). O sorteador do fabricante quase
    # nunca produz `lin ≈ 0 ∧ wz ≠ 0`: standing não gira, forward não pode, heading gira
    # ANDANDO. Um ramo `turning`: lin = 0 todo passo, |wz| ≥ turning_wz_min. A precedência
    # é `standing > turning > forward > heading`, portanto a fração REALIZADA é
    # rel_turning_envs × (1 − rel_standing_envs) ≈ 0,09.
    rel_turning_envs: float = 0.10
    turning_wz_min: float = 0.2


@dataclass
class Giro:
    """O envelope de guinada e o σ que o mede.

    ⚠ O TETO É UM PEDIDO DO DONO (10/09): "uma rotação completa precisa levar no
    máximo 4 s". `2π / 4 s = 1,5708 rad/s`, e o teto adotado é 1,6 — volta em
    3,93 s. O molde para em ±0,7 rad/s no segundo estágio do currículo de comando,
    o que dá 8,98 s por volta.

    ⚠⚠ O σ NÃO PODE FICAR CONSTANTE COM ESTE TETO, e não é preferência. Com σ fixo
    em `sqrt(0.5)` (o valor do molde) e comando em 1,6 rad/s, um robô que NÃO gira
    recebe `exp(−1,6²/0,5) = 0,006`, com derivada de 0,038 por rad/s: kernel morto,
    e o topo do envelope seria inaprendível. É o mesmo defeito medido nos sete
    incentivos da manipulação — σ fixo pequeno = derivada zero.

    O σ vira PROPORCIONAL AO COMANDO, com piso, igual ao `sigma_alcance` do termo de
    comando (`comando.AlvoCaixaCmd`):

        σ = (|cmd_wz| · sigma_fator).clamp(min=sigma_min)

    Com `sigma_fator = 1,0` e `sigma_min = sqrt(0.5)` o kernel fica IDÊNTICO ao de
    hoje para todo comando `|cmd| <= 0,707` (o piso morde) e vira um esticamento
    exato acima disso: a resposta ZERO no topo paga `exp(−1) = 0,37`, o mesmo que
    hoje se paga no topo de 0,7, e a derivada no topo sobe de 0,038 para 0,46 por
    rad/s — 12×.
    """

    # o teto do envelope, em rad/s. Vale para a faixa BASE do twist e para o segundo
    # estágio do currículo de comando. O estágio 0 (±0,5) fica intocado: ele é a
    # rampa de uma run do zero.
    wz_teto: float = 1.6

    # σ do `recompensas.giro_sem_gingado`: proporção do comando, e o piso.
    sigma_fator: float = 1.0
    sigma_min: float = math.sqrt(0.5)


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
    """Os incentivos da manipulação. TODOS positivos e contínuos (R3).

    ⚠ Nenhuma penalidade aqui. Penalidade limita COMO fazer o que já existe; ela não
    ensina a fazer. E booleano é platô — o `pegar` do `g1_poc` travou 22k iterações num
    `squeeze` booleano.

    ⚠ `sustentacao` SAIU (spec dois-bits §2.7): redundante com o fecho, e na cauda
    ficaria travado em 1,0 para sempre. `largou`/`sigma_solta` e `pose_de_braco`/
    `pose_de_braco_sigma` também saíram — a cauda é ANDAR com twist, e sair andando já
    tira as mãos; os braços na espera são assunto do `PosturaPorElo` (§3.1).

    `load` VOLTA (spec §2.7, mudança v3→v3.1): nada pagava por `apoiada` no BOTAR —
    pairar a 1 cm do alvo valia o mesmo que apoiar. `renda_congelada` (1,0) fecha todo
    elo.
    """

    # --- os incentivos ---
    staged: float = 3.0            # alcançar × (1 + trazer). O motor da fase inicial
    precise_pos: float = 3.0       # caixa NO alvo
    precise_ori: float = 1.0       # face pedida apontando ao robô
    squeeze: float = 1.0           # força nas DUAS palmas
    unload: float = 2.0            # a caixa deixou de pesar na laje
    postura_ereta: float = 2.0     # ergueu SEM agachar

    # ⚠ VOLTA (spec §2.7): a caixa apoiada no alvo, só no BOTAR, mesmo gate
    # `~_fora_do_botar` que `unload`/`postura_ereta` já usam para zerar lá.
    load: float = 2.0

    # ⚠ CONGELA a soma dos termos dependentes de elo no passo de todo fecho, e paga
    # esse número como nível constante daí em diante (v2.1, spec P3). Peso 1,0: o
    # valor já É a renda medida do elo que fechou, e não precisa de escala própria.
    renda_congelada: float = 1.0

    # ⚠ `velocidade_por_regime` (G2, spec `g1-limpo-lento-e-estavel.md` §3): penaliza
    # velocidade de junta ACIMA do limite por regime de comando (standing/walking/
    # running) — o espelho do `variable_posture` do fabricante, com VELOCIDADE em vez
    # de POSIÇÃO. Desde a dobradiça (spec tabela-por-estado §4) o limite é FRONTEIRA,
    # não escala: abaixo dele o custo é ZERO.
    #
    # ⚠⚠ O LIMITE DO `standing` É POR FAMÍLIA DE JUNTA, com os MESMOS 14 padrões de
    # `vel_max_walking` (padrões que não se sobrepõem — `resolve_matching_names_values`
    # não aceita ambiguidade). Era uma entrada só, `{".*": 2,0}`, o p99 da locomoção
    # parada: com o limite como ESCALA aquilo servia; como FRONTEIRA, a família manda.
    #
    # ⚠ FONTE: `|qd|` na pega com a caixa, `model_10200`, n = 16 919 — p50 por família
    # de 0,44 a 0,97, p90 de 2,0 a 4,1. O 1,5 deixa a MEDIANA livre e cobra do p90 para
    # cima, onde a pressa mora. Punho a 1,0: é a família mais rápida (p90 4,10, p99
    # 10,12), acima de qualquer perna, e não precisa girar para segurar uma caixa.
    #
    # ⚠ 1,0 nas pernas foi MEDIDO apertado demais com o limite como fronteira: custo
    # 1,74 -> 3,13/s e 13,6% dos passos no antigo clamp. 1,5 é o valor que mantém a
    # mediana livre.
    vel_max_standing: dict = field(default_factory=lambda: {
        r".*hip_pitch.*": 1.5,  r".*hip_roll.*": 1.5,   r".*hip_yaw.*": 1.5,
        r".*knee.*": 1.5,       r".*ankle_pitch.*": 1.5, r".*ankle_roll.*": 1.5,
        r".*waist_yaw.*": 1.5,  r".*waist_roll.*": 1.5, r".*waist_pitch.*": 1.5,
        r".*shoulder_pitch.*": 1.5, r".*shoulder_roll.*": 1.5,
        r".*shoulder_yaw.*": 1.5, r".*elbow.*": 1.5, r".*wrist.*": 1.0,
    })

    # ⚠ Os dois de baixo são o p99 MEDIDO de cada padrão naquele regime (spec §0), e
    # NÃO mudam com a dobradiça. Com ela a marcha normal custa ZERO (era `(v/vmax)²`,
    # ~0,96 no p99): ele não taxa a locomoção, ele morde o excesso.
    # ⚠ E o `running` é `max(p99_running, p99_walking)` por padrão, e isso é decisão: o
    # p99 de `running` saiu MENOR que o de `walking` em cinco padrões, porque a
    # amostra veio do transiente do `velocity_stages` novo (lin_vel_x foi a 2,0 m/s na
    # última iteração da bloco9). Limite de correr mais apertado que o de andar
    # puniria correr, ao contrário.
    vel_max_walking: dict = field(default_factory=lambda: {
        r".*hip_pitch.*": 4.0,  r".*hip_roll.*": 5.0,   r".*hip_yaw.*": 4.0,
        r".*knee.*": 6.5,       r".*ankle_pitch.*": 6.0, r".*ankle_roll.*": 4.0,
        r".*waist_yaw.*": 3.0,  r".*waist_roll.*": 5.0, r".*waist_pitch.*": 2.5,
        r".*shoulder_pitch.*": 3.0, r".*shoulder_roll.*": 4.0,
        r".*shoulder_yaw.*": 2.0, r".*elbow.*": 3.0, r".*wrist.*": 2.5,
    })
    vel_max_running: dict = field(default_factory=lambda: {
        r".*hip_pitch.*": 5.0,  r".*hip_roll.*": 5.0,   r".*hip_yaw.*": 4.0,
        r".*knee.*": 10.0,      r".*ankle_pitch.*": 9.0, r".*ankle_roll.*": 4.0,
        r".*waist_yaw.*": 3.0,  r".*waist_roll.*": 5.0, r".*waist_pitch.*": 2.5,
        r".*shoulder_pitch.*": 3.5, r".*shoulder_roll.*": 4.5,
        r".*shoulder_yaw.*": 2.5, r".*elbow.*": 3.0, r".*wrist.*": 2.5,
    })
    # ⚠⚠ PESO NEGATIVO, e é correção medida na revisão de 2026-09-08. A forma
    # positiva `exp(−média(v²/vmax²))` pagaria 2,0/s a um robô PARADO, em TODO env —
    # renda grátis que entra direto no piso da estátua (medido 8,265/s no PEGAR contra
    # 3,863/s no ANDAR, parado ganhando por 43%). A forma que roda paga ZERO parado:
    # ela cobra o excesso de velocidade, não premia a ausência dele — o mesmo idioma
    # do `contato_mesa` deste módulo (valor positivo; o peso negativo é quem faz dele
    # penalidade).
    #
    # ⚠⚠ A FORMA REAL É A DOBRADIÇA `média(relu(|v|/vmax − 1)²)`, SEM clamp
    # (`recompensas.velocidade_por_regime`, spec tabela-por-estado §4), e o `smoke.py`
    # (seção G2, item 3) TRAVA essa forma de propósito. Ela substitui o
    # `clamp(média((v/vmax)²), max=4)` da v3.1 — que por sua vez substituíra a
    # `1 − exp(...)` de derivada ZERO no teto. Duas mudanças de uma vez, as duas
    # medidas: (a) ABAIXO do limite o custo passa a ser ZERO — a `(v/vmax)²` cobrava
    # de leve o tempo todo, e movimento lento agora é livre; (b) ACIMA, o quadrado do
    # EXCESSO cresce sem teto — 2× o limite custa 1,0 (era 4,0), 5× custa 16,0 (era
    # 4,0, no clamp).
    #
    # ⚠ Confira à mão, com o peso −2,0: parado -> 0, custo ZERO. Tudo no limite -> 0,
    # custo ZERO. Tudo no DOBRO -> 1,0, custo **2,0/s**. No TRIPLO -> 4,0, custo
    # **8,0/s**. A 5× -> 16,0, custo **32,0/s**.
    #
    # ⚠ QUADRADO, e NÃO exponencial: o p99 da pega é 10 rad/s contra ~1,5 de limite, e
    # `e^{5,7}` num único passo dominaria o lote. O quadrado já cresce sem teto.
    #
    # ⚠ O PESO −2,0 FICA: a pressa medida mostra que nem 8,0/s segurava o robô contra
    # um degrau de fecho de 18,55/s — o que faltava era derivada ACIMA do clamp, não
    # peso.
    #
    # ⚠ O DÉBITO "clamp é licença acima do teto" FECHA AQUI. 2,26% dos passos da pega
    # com a caixa estavam em `valor = 4,0` com derivada ZERO — mover mais rápido era
    # grátis. Com a dobradiça a derivada em 3× é `4/vmax` por rad/s e sobe com o
    # excesso; e o `vel_max_standing` por FAMÍLIA (acima) é a fronteira que deixa a
    # mediana livre e cobra a pressa.
    velocidade_por_regime: float = -2.0

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

    # ⚠ O `precise_pos` é o ÚNICO com σ FIXO, e de propósito: ele é uma RAMPA DE ACEITE
    # — "a caixa está NO alvo?" — e não uma rampa de aproximação (essa é o `staged`, com
    # σ por env). O raio 0,18 é MAIOR que `tol_pos` (0,10): no limiar do fecho o termo
    # já paga `exp(−(0,10/0,18)²) = 0,73`, contra 0,018 do σ velho de 0,05 — o raio de
    # aceite cobre o raio de fechamento, em vez de morrer antes dele.
    precise_pos_sigma: float = 0.18

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
    #
    # ⚠ DÉBITO REGISTRADO, e NÃO consertado: o `squeeze` é CEGO AO ESMAGAMENTO. O
    # `tanh` satura e nada cobra a força ACIMA do `F_ref` — medidos 68,9 N de aperto
    # contra os 8,45 N de referência da medição (o `F_ref` deste módulo, com m = 1,0 kg
    # e μ = 0,8, dá 6,13 N; a divergência está no número da medição, não na fórmula).
    # Adiado DE PROPÓSITO: o `squeeze` é termo CONGELÁVEL, e mudar a forma dele mexe no
    # valor do fecho às vésperas de um resume. E o simulador não pune esmagar — isto é
    # dívida de sim-to-real, não defeito do treino de hoje.
    squeeze_mu: float = 0.8              # μ pessimista da faixa de atrito da caixa

    # --- postura ereta: a rampa dupla na pelve ---
    # ⚠ Ela paga por erguer SEM agachar, e é o que impede o robô de satisfazer o alvo
    # descendo até a caixa. Medido na pose de pé: pelve em 0,798 m.
    pelve_alvo: float = 0.75       # acima disto a rampa paga cheio
    pelve_piso: float = 0.45       # abaixo disto ela paga zero
    # ⚠ v3.5: o piso da rampa DEPOIS do fecho do BOTAR. MEDIDO no `model_9499`, estado
    # do fecho (`perto ∧ apoiada`), 8762 amostras: pelve p10 0,379 / p50 0,443 / p90
    # 0,571 / mín 0,334. O `pelve_piso` de 0,45 fica acima da MEDIANA — metade dos
    # fechos teria rampa 0 com derivada 0, onde a subida tem de nascer. 0,32 fica
    # 1,4 cm abaixo do mínimo observado; no p50 a rampa vale 0,29 e sobe até 1,0.
    pelve_piso_cauda: float = 0.32
    # ⚠ a rampa do `postura_ereta` satura ACIMA do próprio `pelve_alvo` (soma com
    # `pelve_margem`) para a política não parar exatamente na borda de onde a rampa já
    # satura, onde a derivada seria zero. `de_pe` do fecho NÃO lê mais `pelve_alvo`
    # (spec dois-bits §2.4): ele virou pose de junta — ver `de_pe_tol_rad`, abaixo.
    pelve_margem: float = 0.03

    # --- tolerâncias de fechamento ---
    # a tolerância que conta como "na condição", em metros e radianos
    tol_pos: float = 0.10
    tol_ang_deg: float = 25.0
    # ⚠ `de_pe` (spec dois-bits §2.4): a maior excursão de junta das PERNAS e da
    # CINTURA em relação ao default, em radianos — não mais a altura da pelve.
    # MEDIDO no PEGAR dos níveis 4–6, no instante em que o robô está DE PÉ com a
    # caixa erguida (model_6999, 64 envs, 1100 passos): p50 0,632 rad, p90 0,694
    # rad, máx 0,829 rad — idêntico nos três níveis.
    #
    # ⚠ MEDIÇÃO INVALIDADA (revisão do coordenador): "idêntico nos três níveis"
    # NÃO é porque os três forçam a mesma laje a 0,04 m — é porque `limpo_topo`
    # saiu igual (p50 0,43) nos três, ou seja, o FORÇAR DE NÍVEL NÃO AGIU. A laje
    # real da medição ficou em ~0,43 m, não 0,04 m. DOMINADO pelo quadril esquerdo
    # (pitch, p90 0,73 rad) e pelo joelho esquerdo (p90 0,61 rad) — postura
    # ASSIMÉTRICA, perna esquerda mais dobrada que a direita para equilíbrio.
    #
    # O valor 0,69 FICA — o fallback 0,35 travaria o fecho até de pé de qualquer
    # jeito — mas precisa ser RE-MEDIDO no nível 6 de verdade (laje a 0,04 m),
    # depois de consertar o forçar de nível na sonda.
    de_pe_tol_rad: float = 0.69


@dataclass
class Cadeia:
    """A tabela de cadeias de elo (spec `g1-limpo-dois-bits.md` §2).

    ⚠ O TETO É DERIVADO de `CADEIAS` (hoje 3) — B, R, C:
      índice 0 (B): (PEGAR,)              -> pegar, depois CAUDA carregar
      índice 1 (R): (REORIENTAR, PEGAR)   -> reorientar, pegar, depois CAUDA carregar
      índice 2 (C): (PEGAR, BOTAR)        -> pegar, botar, depois CAUDA botar

    O `pegar` aparece em TODAS: ele é o eixo de que não se esquece. O `CARREGAR` SAIU
    das tuplas — ele é o estado de CAUDA de quem fechou o PEGAR e não vai botar, e
    não fecha mais.
    """

    # ⚠ O INTERRUPTOR DA MÁQUINA DE ELO (§2.1). `prob_por_nivel = ()` — a tabela [7
    # níveis × 4 cadeias] — era o desliga; virou este bool. Com `False`,
    # `smoke.py` e `inspeciona.py` montam cfgs sem cadeia nenhuma, como faziam
    # passando a tabela vazia.
    ativa: bool = True

    # Tempos de sustentação (em segundos) quando o elo fecha.
    # ⚠ OS DOIS VALEM 0,5 s desde a v3.4. O `pegar` pede 0,5 s porque ele é o elo do
    # qual TODAS as cadeias dependem, e um fecho por acidente de um frame propagaria
    # para os outros três.
    sustenta_pegar_s: float = 0.5
    # ⚠ 0,3 → 0,5 s na v3.4 (spec `g1-limpo-botar-fecha-e-para.md` §2.3). Pedido do
    # dono: a caixa tem de ficar estável no alvo por mais de 0,5 s antes do fecho do
    # BOTAR. Ele aceita que o robô decore o tempo. Efeito colateral aceito: o
    # REORIENTAR inerte também lê este knob, e o atraso dele passa de 0,3 para 0,5 s.
    # ⚠⚠ ESTE NÚMERO VIVE EM DOIS ARQUIVOS: aqui e em `comando.AlvoCaixaCmdCfg`.
    # Mude os dois, ou eles derivam em silêncio.
    sustenta_outros_s: float = 0.5
    # ⚠ O REORIENTAR ESTÁ INERTE NA v2 (spec §8.3): o fecho dele ignora `alinhado` e o
    # elo vira um atraso de `sustenta_outros_s` antes do PEGAR. MEDIDO em 03/09 que
    # `voltas_max = 0` sozinho não bastava (o jitter lateral da caixa tira a direção
    # pedida da tolerância em ~1 de 6 envs). Quando a reorientação virar foco: False.
    reorientar_inerte: bool = True
    # ⚠ O REORIENTAR inerte CONTINUA no sorteio, a 5% (decisão do dono, 2026-09-04). O
    # one-hot do elo tem um canal para ele; o normalizador do rsl_rl é `(x − μ)/(σ + 0,01)`
    # sem clamp, e um canal constante tem σ = 0 — ao acender, `1,0` entra como ×100. É a
    # mesma regra que fixa `fatia_loco = 0,95`.
    # ⚠ O custo é 0,3 s em 5% dos envs de manipulação (0,08% do tempo). As cadeias 2 e 3
    # vão de 2,8% para 5,3% dos envs. Com `reorientar_inerte = False` o sorteio volta a 50%.
    prob_reorientar_inerte: float = 0.05

    # ⚠ `carregar_dist_m`/`carregar_s` SAÍRAM (spec dois-bits §2.4): o CARREGAR não
    # fecha mais, e não há mais "andou" para exigir — ele é a CAUDA de quem fechou o
    # PEGAR, e dura o resto do episódio.

    # ⚠⚠ O BALANCEADOR B/C (spec §2.5), o mecanismo que substitui `prob_por_nivel` (28
    # números) por 2. Decide, para quem começa no PEGAR, entre a cadeia B (segurar e
    # carregar) e a C (botar):
    #
    #     p_C = clamp((1 − s_C) / ((1 − s_B) + (1 − s_C) + 1e−6), piso, 1 − piso)
    #
    # `s_B`, `s_C` são EMAs de `concluiu` por cadeia, atualizadas UMA VEZ por
    # ITERAÇÃO de PPO — não por reset. Quanto menos uma cadeia conclui, mais ela é
    # sorteada.
    #
    # ⚠ `piso` trava `p_C` numa faixa: um slot do one-hot constante em 0 ou 1 entraria
    # como ×100 no normalizador do dia em que acendesse — a mesma regra de
    # `fatia_loco = 0,95`. A semente (`s_C = 1, s_B = 0`) nasce em `p_C = piso`: a
    # cadeia C, mais difícil, começa RARA e abre conforme deixa de concluir.
    balanceador_piso: float = 0.20
    balanceador_alpha: float = 0.05


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

    caixa_folga_chao: float = 0.02
    """Folga, em metros, do FUNDO da caixa ao chão abaixo da qual ela CAIU.
    ⚠ Substitui `caixa_z_min = 0,10`, que era a meia-aresta de UMA caixa. Com o tamanho
    variando (spec §6.7), o limiar fixo não acusava a queda da caixa de 0,13 m (centro
    em 0,13) e ficava a 1 cm de acusar a de 0,07 m na laje a 0,04 m. Agora
    `caiu = z_centro − meia_aresta_env < folga`: "o fundo está a menos de 2 cm do chão".
    Menor que `prateleira_topo_piso` (0,04), senão a laje mais baixa dispararia."""

    v_solta: float = 1.2
    """Velocidade relativa (caixa − base), em m/s, acima da qual a caixa foi solta ou
    atirada fora do alvo (spec `g1-limpo-soltar-termina.md` §2, `terminacoes.caixa_largada`).
    Substitui `caixa_dist_max` (v3.2): distância não pegava o arremesso curto.

    ⚠ MEDIDO 2026-09-09 (`scratchpad/sonda_v_solta.py`, model_6999, 32 envs, cadeia B,
    CPU): (a) caixa na mão, p90 0,85 m/s (p99/máx brutos, 2,1/3,5, são caixa JÁ perdida
    com `limpo_pegou` preso em 1 — não hold); (c) push na pega, longe da mão, p90
    1,08 m/s (p99/máx brutos, 2,1/4,7, são transiente de contato no RESET de um único
    env, não push — `push_robot` é ~0,5 m/s); (b) queda livre do peito (+0,45 m):
    0,71 / 0,85 / 1,31 m/s nos passos 1/3/5. `v_solta` fica acima do p90 LIMPO de
    (a)/(c) e abaixo de b.passo5 — não do máx/p99 brutos, contaminados (ver
    `sonda_v_solta.out`)."""

    joelho_z_min: float = 0.05
    """Altura mínima do joelho, em metros, acima da origem do env (spec dois-bits
    §3.2). Abaixo disto, `terminacoes.caiu` acusa — mesmo sem `bad_orientation`
    disparar: o robô agachou de LADO, sem tombar.

    ⚠ MEDIÇÃO INVALIDADA (revisão do coordenador): a sonda por nível forçado deu
    `limpo_topo` IDÊNTICO nos níveis 4, 5 e 6 (p50 0,43) — o forçar de nível NÃO
    variou a laje, e o mínimo de joelho 0,089 m foi medido numa laje a ~0,43 m,
    não a 0,04 m como o comentário anterior afirmava. Na laje baixa de verdade o
    agachamento é mais fundo, e 0,089 m não é um limite confiável.

    ⚠ VALOR CONSERVADOR: 0,05 m é o RAIO do joelho — a terminação só dispara com o
    joelho de fato tocando o chão, e não mata agachamento legítimo nenhum. Fica
    até uma medição na laje a 0,04 m de verdade (nível 6 real) resolver o
    forçar de nível."""


@dataclass
class PesoPorEstado:
    """A TABELA POR ESTADO (spec `g1-limpo-tabela-por-estado.md` §2): o peso de cada
    um dos dez termos em cada um dos dez estados de recompensa que o comando publica
    em `env.limpo_estado`. `recompensas.PesoPorEstado` multiplica o termo pela coluna.

    ⚠⚠ O PRINCÍPIO: o teto do que o robô ainda tem de fazer >= o piso do que ele já
    fez (a `renda_congelada`). Medido em `model_10200` era o contrário — BOTAR 2:1,
    cauda C 3:1, CARREGAR 7:1 — e as consequências: ele não anda com a caixa (97% dos
    passos com os dois pés no chão), não levanta depois de botar (pelve p10 em
    0,469 m) e corre na pega (fechar 1 s antes compra ~14 de renda). A
    `renda_congelada` FICA, cumulativa: ela garante o arranque. A tabela é a outra
    metade.

    ⚠ A ORDEM DAS COLUNAS é `comando.ESTADOS`. Ele NÃO é importado aqui (`cena.py`
    importa este módulo e `comando.py` importa `cena` — o ciclo fecharia); o `smoke`
    amarra os dois pelo comprimento das tuplas. As colunas:

        0 ANDAR   1 ESPERA_SEM   2 ESPERA_COM   3 REORIENTAR_SEM   4 REORIENTAR_COM
        5 PEGAR_SEM   6 PEGAR_COM   7 CARREGAR   8 BOTAR   9 CAUDA

    POR QUE CADA NÚMERO:

    · Os sete de manipulação, colunas 0-6 e 8-9: são o `VALIDA` de ontem escrito por
      extenso — 0 no ANDAR, nas duas esperas e na CAUDA; 1 no REORIENTAR e no PEGAR.
    · BOTAR = 2. Ele fecha hoje a 7,92/s (p50, 135 fechos) contra um piso de 13,79.
      ×2 leva o fecho a ~15,8 >= 13,79. Presente >= passado.
    · CARREGAR: só `precise_pos` fica (= 1); os outros seis vão a 0. Eles pagam
      13,20/s pela caixa estar no peito — atingido no instante da pega e satisfeito
      PARADO. É o piso da estátua com a caixa na mão, e todo termo que paga por estar
      parado tem de ser gateado na tarefa. `precise_pos` fica para a caixa não descer
      do peito (18 cm de raio; teto 3,0). Medido: `unload ≡ 1`, `load ≡ 0` e
      `postura_ereta` saturada ali — três dos seis são constantes, apagá-los custa
      zero gradiente.
    · Rastreio no CARREGAR = 3,5. Teto dos dois: 2,0 + 2,0 = 4,0; ×3,5 = 14 ≈ o piso
      de 13,79. Parado = 13,79 + 3 = 16,8; andando bem = 30,8; ganho de andar +14 e
      break-even de risco 45% (hoje +1,5 e 5%).
    · Rastreio nas outras colunas reproduz o antigo `rastreio_por_elo` (`fator = 1 −
      zerado × (1 − pegou)`), estado a estado: ANDAR 1 (twist vivo); ESPERA_SEM 0;
      ESPERA_COM 1; REORIENTAR_SEM e PEGAR_SEM 0 (twist zerado, nunca tocou —
      estátua); REORIENTAR_COM, PEGAR_COM, BOTAR e CAUDA 1 (já tocou — segurar/parar
      É a tarefa).
    · `postura_ereta` e `pose` na CAUDA = 8. Piso depois do BOTAR ×2: 13,79 + 15,8 =
      29,6. Teto da cauda `2k + k + 4 + 1 = 3k + 5` -> k = 8. Só estes dois separam
      "agachado com as mãos na caixa" de "de pé na pose default"; rastreio e `upright`
      são satisfeitos agachado. Break-even de risco 24/(29,6 + 24) = 45%.
    · `pose` em PEGAR_COM e ESPERA_COM = 4. Enquanto segura, ombro e cotovelo estão
      FORA do `pose` (máscara do `PosturaPorElo`), portanto ×4 atinge punho, perna e
      cintura — o punho torcido e a perna solta. Com std 1,00 o punho a 1,6 rad custa
      0,52/s de um teto de 1,0, 4,5% da renda de 11,38 — ele não se importa; ×4 leva
      a 2,1/s. NÃO em PEGAR_SEM (o braço ainda está na média e ×4 brigaria com o
      alcance), NÃO em BOTAR (as pernas agacham para a laje a 0,30 m), NÃO em
      CARREGAR (é marcha; `std_walking`).

    ⚠ `squeeze` e `unload` (e a `rampa × descarga` do `postura_ereta`) continuam
    zerados DENTRO do BOTAR por `_fora_do_botar`, que fica: o 2 dessas linhas em BOTAR
    é o `VALIDA` por extenso, e ali multiplica zero.

    ⚠ FORA DA TABELA, de propósito: `upright`, `terminacao`, `contato_*`, `joint_acc`,
    `action_rate_l2`, `velocidade_por_regime` e `renda_congelada` (que lê os sete pelo
    NOME, já multiplicados — é isso que faz o piso do BOTAR ×2 valer ~15,8).
    """

    #                            ANDAR ESP_SEM ESP_COM REOR_SEM REOR_COM PEG_SEM PEG_COM CARREGAR BOTAR CAUDA
    staged: tuple[float, ...] = (0.0,  0.0,    0.0,    1.0,     1.0,     1.0,    1.0,    0.0,     2.0,  0.0)
    precise_pos: tuple[float, ...] = (0.0, 0.0, 0.0,   1.0,     1.0,     1.0,    1.0,    1.0,     2.0,  0.0)
    precise_ori: tuple[float, ...] = (0.0, 0.0, 0.0,   1.0,     1.0,     1.0,    1.0,    0.0,     2.0,  0.0)
    squeeze: tuple[float, ...] = (0.0, 0.0,    0.0,    1.0,     1.0,     1.0,    1.0,    0.0,     2.0,  0.0)
    unload: tuple[float, ...] = (0.0,  0.0,    0.0,    1.0,     1.0,     1.0,    1.0,    0.0,     2.0,  0.0)
    postura_ereta: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0,     1.0,     1.0,    1.0,    0.0,     2.0,  8.0)
    load: tuple[float, ...] = (0.0,    0.0,    0.0,    1.0,     1.0,     1.0,    1.0,    0.0,     2.0,  0.0)
    track_linear_velocity: tuple[float, ...] = (1.0, 0.0, 1.0, 0.0, 1.0,  0.0,    1.0,    3.5,     1.0,  1.0)
    track_angular_velocity: tuple[float, ...] = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0,    1.0,    3.5,     1.0,  1.0)
    pose: tuple[float, ...] = (1.0,    1.0,    4.0,    1.0,     1.0,     1.0,    4.0,    1.0,     1.0,  8.0)


@dataclass
class Knobs:
    cena: Cena = field(default_factory=Cena)
    alvo: Alvo = field(default_factory=Alvo)
    nivel: Nivel = field(default_factory=Nivel)
    recompensa: Recompensa = field(default_factory=Recompensa)
    marcha: Marcha = field(default_factory=Marcha)
    giro: Giro = field(default_factory=Giro)
    forma: Forma = field(default_factory=Forma)
    piso: Piso = field(default_factory=Piso)
    tarefa: Tarefa = field(default_factory=Tarefa)
    peso_por_estado: PesoPorEstado = field(default_factory=PesoPorEstado)
    cadeia: Cadeia = field(default_factory=Cadeia)
    terminacao: Terminacao = field(default_factory=Terminacao)
    contato: Contato = field(default_factory=Contato)


ATIVO = Knobs()
