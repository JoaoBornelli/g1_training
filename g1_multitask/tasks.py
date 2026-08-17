"""As 5 tarefas: índice no one-hot, eixo de currículo, e a conta dos 24 eventos.

Este módulo é a ESPINHA do desenho. Uma política só, condicionada a comando:
`ação = f(estado, comando)`. A tarefa NÃO é um env diferente — é um one-hot dentro
do vetor de comando. É isso que faz o algoritmo de visão, em campo, entrar como
mais um escritor do mesmo vetor ("gire a caixa pra esquerda") sem tocar na política.

**Um eixo por tarefa.** O desenho anterior dava dois eixos a quatro tarefas, e o
segundo era o `peso`. O peso deixou de ser eixo — ver `LEVELS["peso"]`.

Ver `EXPERIMENTO.md`, §3 e §9.
"""
from __future__ import annotations

# ---------------------------------------------------------------------- one-hot
# Índices FIXADOS. A ordem entra no vetor de comando E no checkpoint, e a fatia
# [9:17] da obs é posicional -> trocar um número aqui invalida todo checkpoint
# anterior (Categoria C). Não reordenar.
#
# ⚠️ A numeração é COMPACTA (0..4), e não herdada das 7 tarefas antigas. Preservar
# os índices de `pegar`/`botar`/`reorientar` daria warm start um pouco melhor, mas
# deixaria buracos em `range(NUM_TASKS)` — e o `_equaliza_orcamento` assertaria
# contra a tarefa vazia. A reforma muda tanta recompensa que o ganho semântico do
# one-hot herdado seria especulativo.
LOCOMOVER = 0
PEGAR = 1
REORIENTAR = 2
LOCOMOVER_CARREGANDO = 3
BOTAR = 4

NUM_TASKS = 5
ONEHOT_DIM = 8
"""8 slots com 5 em uso.

⚠️ **Continua 8, e isso é de propósito.** Encolher para 5 mudaria a largura da
observação de 154 para 151, e largura é contrato com o checkpoint (§13). Os 3 slots
livres ficam constantes em zero e recebem o mesmo tratamento do 8º slot original:
coluna zerada na 1ª camada e stats do normalizador congeladas. Sem isso, um canal
constante ganha `_std = 0` no 1º update e, ao acender, entra 100× amplificado."""

NAMES = {
    LOCOMOVER: "locomover",
    PEGAR: "pegar",
    REORIENTAR: "reorientar",
    LOCOMOVER_CARREGANDO: "locomover_carregando",
    BOTAR: "botar",
}

# --------------------------------------------------------- conjuntos semânticos
# Usados por gates de reward, terminações e pela subclasse de comando. Ficam aqui,
# num lugar só, pra a máscara do gate e a máscara do log saírem da MESMA fonte.

MANIPULA = (PEGAR, BOTAR, REORIENTAR)
"""Manipulação. O comando de velocidade é ZERADO nestas três (§4)."""

CMD_ZERO = MANIPULA
"""Alias semântico: onde a subclasse de comando força `[0, 0, 0]`.

Nome próprio porque o conceito é do COMANDO, não da tarefa. Se um dia uma tarefa de
manipulação passar a andar, ela sai daqui sem sair de `MANIPULA`."""

ANDA = (LOCOMOVER, LOCOMOVER_CARREGANDO)
"""Tarefas com comando de velocidade sorteado."""

COM_CAIXA = (LOCOMOVER_CARREGANDO,)
"""Tarefas em que CARREGAR é o estado exigido -> largar é falha (terminação).

No `pegar`, largar no meio deve permitir nova tentativa no mesmo episódio; no
`botar`, soltar É o objetivo. Só aqui largar termina o episódio (§6b/D)."""

SPAWN_SEGURANDO = (LOCOMOVER_CARREGANDO, BOTAR)
"""Tarefas que nascem com as PALMAS TOCANDO a caixa (§3, §4).

Não é o mesmo conjunto que `COM_CAIXA`: o `botar` também nasce segurando (o estado
final do `pegar` é o estado inicial canônico dele), mas largar nele é o OBJETIVO.

⚠️ Nascer segurando é nascer TOCANDO, com força normal zero. Segurar é o que a
tarefa ensina — ver `pregrasp.py`. Medido em 30/07: sem esta condição de spawn as
tarefas não têm caminho de aquisição, porque todos os termos de tarefa dão 0.0 no
reset."""

COM_DR_PESO = (PEGAR, REORIENTAR, LOCOMOVER_CARREGANDO, BOTAR)
"""As quatro tarefas que envolvem a caixa. Só elas têm a DR de peso (§9)."""

TERMOS_DE_TAREFA = (
    "track_linear_velocity", "track_angular_velocity",
    "lift", "reaching", "grasp", "unload", "box_at_peito", "box_at_prateleira",
    "orienta_face",
)
"""Os termos POSITIVOS que dizem qual tarefa é. Fonte única do orçamento.

É sobre esta lista que o `_equaliza_orcamento` do `env.py` calcula quanto cada tarefa
pode ganhar por passo.

**Ficam de fora** o `upright` (sem gate: vale 1.0 em toda tarefa, logo não muda a
relação entre elas) e a postura (piso postural, o análogo do `pose` do fabricante).
Ficam de fora também todos os negativos: o objetivo é igualar a razão entre sinal e
penalidade, e para isso o denominador tem que ficar parado.

O `hold_still` saiu (§10): ele é redundante com o `track_angular_velocity`, que lê o
mesmo `root_link_ang_vel` e pune 3,5× mais forte a ω = 1.

O `sucesso_denso` fica de fora **de propósito** (17/08). Ele é bônus de OBJETIVO, não
sinal de aproximação: pôr os 5,0 dele no orçamento faria o fator do `pegar` cair para
0,42 e o próprio bônus se diluir para 2,1 — ele se anularia. Como ele vale nas quatro
tarefas com caixa, a paridade entre elas fica preservada de qualquer forma.

⚠️ Todos os pesos aqui são CONSTANTES em tempo de execução, e o orçamento depende
disso. Quem for mutar peso de reward em runtime tem que recalcular a escala no mesmo
passo, senão a tarefa afunda em silêncio."""

# ------------------------------------------------------------- eixos e níveis
LEVELS: dict[str, tuple[float, ...]] = {
    "velocidade": (1.0, 1.5, 2.0),
    "altura": (0.55, 0.45, 0.35, 0.25, 0.15, 0.05, 0.00),
    "giro": (15.0, 45.0, 90.0, 180.0, 360.0),
    "alvo": (0.2, 0.4, 0.6, 0.8, 1.0),
    "peso": (1.0, 5.0),
}
"""Níveis de cada eixo, do fácil pro difícil (§9).

- `velocidade` é o TETO de `lin_vel_x` do comando sorteado, por env. O `lin_vel_y` NÃO
  acompanha: a progressão do fabricante para o G1 alarga só o `x`.
- `giro` é a rotação COMANDADA (alvo posto a N graus da orientação ATUAL da caixa),
  não o quanto ela nasce torta. O último nível (360) é o salto qualitativo "a face
  alvo pode ser o topo ou o fundo" — exige erguer e rolar a caixa entre as palmas.
- `alvo` é o eixo do `pegar`, em **fração** da distância entre o repouso da caixa e uma
  altitude FIXA de 0,91 m no mundo. Ele gradua **quanto erguer** (17/08).

  ⚠️ **As duas pontas são do MUNDO; nenhuma segue o robô.** O repouso é propriedade da
  prateleira; o topo é constante. Um alvo que acompanha a pelve faz agachar encurtar o
  percurso, e o argmax vira levar o peito até a caixa em vez de erguer a caixa — foi o
  que o robô aprendeu no bloco 2. Ver `rewards._TOPO_RAMPA_Z`.

  ⚠️ **Fração, e não centímetro.** O eixo `altura` desce a prateleira num bloco
  futuro, e o repouso desce com ela. Valor absoluto em metros ficaria descolado: o
  nível 1,0 pararia longe do peito. Com fração, `alvo_z = repouso + f × (peito −
  repouso)` e o nível 1,0 é sempre o peito, em qualquer altura de prateleira.

  Com a prateleira em 0,55 m (repouso 0,65 m) e o peito em 0,91 m, os cinco níveis
  pedem 5,2 · 10,4 · 15,6 · 20,8 · 26,0 cm. O nível 0 fecha erguendo 5 cm.

⚠️ **`peso` NÃO É EIXO.** Ele está aqui porque é uma tabela de níveis, mas não
aparece em `AXES`, não tem célula no orquestrador, não tem EMA e não tem portão. Ele
é a DR de carga, em 2 níveis: `1 kg` fixo e `U(1, 5)` kg. O sucesso não é atrelado ao
peso — o critério é "fez o que tinha que fazer", qualquer que seja a massa.

O nível 1 CONTÉM o nível 0, porque o sorteio é `U(piso, teto)`. Sem isso a carga leve
sumiria do treino no momento em que a DR alargasse.

Os eixos `rumo`, `distancia_andar` e `push` deixaram de existir. Os dois primeiros
serviam ao comando derivado de um destino, que saiu; o `push` virou evento fixo."""

AXES: dict[int, dict[str, int]] = {
    LOCOMOVER: {"velocidade": 0},
    PEGAR: {"alvo": 0},
    REORIENTAR: {"giro": 0},
    LOCOMOVER_CARREGANDO: {"velocidade": 0},
    BOTAR: {"altura": 0},
}
"""O eixo de cada tarefa e o índice INICIAL dele.

Um eixo por tarefa, e isso não é economia: é o que faz o condicionamento da medição
ser desnecessário. Com dois eixos, a célula de um mede marginalizada sobre o outro, e
o portão trava — foi o bug de 06/08.

O `locomover_carregando` fica com `velocidade`, e não com `peso`, porque o peso não é
eixo.

⚠️ **O `pegar` trocou `altura` por `alvo` em 17/08.** O eixo `altura` move a
prateleira: ele gradua DE ONDE pegar, e baixá-la só aumenta a distância a erguer.
Nenhum eixo graduava QUANTO erguer, e é isso que travava a tarefa. A altura volta a
ser eixo do `pegar` num bloco futuro, quando o `alvo` esgotar — o `botar` continua com
ela. Serializar assim mantém o invariante de um eixo vivo por tarefa."""

NIVEIS_ATIVOS: dict[str, int] = {
    "velocidade": 1,
    "altura": 1,
    "giro": 1,
    "alvo": 5,
}
"""Quantos níveis de cada eixo o currículo pode abrir AGORA.

⚠️ **Isto é um congelamento deliberado (17/08), não a tabela física.** A decisão é
"ver se o robô consegue fazer TUDO antes de endurecer qualquer coisa": só o `alvo`
progride, porque ele é o eixo que destrava o `pegar`. Os outros três ficam no nível
mais fácil, e o currículo passa a ter um trabalho só — abrir as cinco tarefas.

`LEVELS` fica intacto de propósito: ele é a tabela física, e o `play.py` valida
`--velocidade` contra ela. Descongelar um eixo = mudar um número aqui.

⚠️ **Congelar exigiu um conserto no orquestrador.** Com um nível só, o `locomover`
não tem eixo a avançar nem DR de carga, então a regra do evento caía no `continue` e
ele NUNCA tinha evento — e sem o primeiro evento dele o `pegar` e o `reorientar` nunca
abriam. Ver a condição `eventos_tarefa[t] == 0` em `curriculum.py`."""


def exceto(*excluidas: int) -> tuple[int, ...]:
    """Todas as tarefas menos as passadas. Para gate "ligado em tudo, MENOS em X"."""
    fora = set(excluidas)
    return tuple(t for t in range(NUM_TASKS) if t not in fora)


def eixo_de(task: int) -> str:
    """O nome do eixo da tarefa. Um por tarefa, por construção."""
    (nome,) = AXES[task]
    return nome


def axis_levels(task: int, axis: str) -> tuple[float, ...]:
    """Os níveis que a tarefa de fato usa nesse eixo.

    Corta o início (`AXES`) e o teto (`NIVEIS_ATIVOS`). É a ÚNICA fonte do
    comprimento de célula: o orquestrador testa `abertos < len(axis_levels(...))`,
    então congelar um eixo aqui congela o portão dele."""
    inicio = AXES[task][axis]
    return LEVELS[axis][inicio: inicio + NIVEIS_ATIVOS[axis]]


def unlock_count() -> dict[str, int]:
    """Quantos destravamentos cada fonte contribui.

    Serve de teste: a conta é DERIVADA dos níveis ativos e dos índices iniciais, então
    se alguém mexer num nível sem querer o total denuncia.

    Com o congelamento de 17/08 são **12**: 4 aberturas de tarefa, 4 alargamentos de DR
    e 4 níveis de `pegar_alvo`.

    ⚠️ A DR de peso CONTA aqui desde 17/08. Ela não avança eixo, mas consome um evento
    (a regra abre a DR antes de mexer no eixo), e com os eixos congelados ela passou a
    ser a única fonte de evento de três tarefas. Deixá-la fora faria o total mentir."""
    out = {NAMES[t]: 0 for t in AXES}
    for task, axes in AXES.items():
        for axis in axes:
            out[NAMES[task]] += len(axis_levels(task, axis)) - 1
    for t in COM_DR_PESO:
        out[NAMES[t]] += 1
    out["aberturas"] = NUM_TASKS - 1  # a 1ª tarefa já nasce aberta
    return out


def total_unlocks() -> int:
    return sum(unlock_count().values())
