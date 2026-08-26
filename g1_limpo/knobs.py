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
    # o robô chega andando, não parado
    reset_base_vel_manipula: dict = field(default_factory=lambda: {
        "x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-0.4, 0.4),
    })


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
class Knobs:
    cena: Cena = field(default_factory=Cena)
    alvo: Alvo = field(default_factory=Alvo)
    nivel: Nivel = field(default_factory=Nivel)


ATIVO = Knobs()
