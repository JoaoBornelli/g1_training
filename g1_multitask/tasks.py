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
    "lift", "reaching", "grasp", "box_at_peito", "box_at_prateleira",
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

⚠️ Todos os pesos aqui são CONSTANTES em tempo de execução, e o orçamento depende
disso. Quem for mutar peso de reward em runtime tem que recalcular a escala no mesmo
passo, senão a tarefa afunda em silêncio."""

# ------------------------------------------------------------- eixos e níveis
LEVELS: dict[str, tuple[float, ...]] = {
    "velocidade": (1.0, 1.5, 2.0),
    "altura": (0.55, 0.45, 0.35, 0.25, 0.15, 0.05, 0.00),
    "giro": (15.0, 45.0, 90.0, 180.0, 360.0),
    "peso": (1.0, 5.0),
}
"""Níveis de cada eixo, do fácil pro difícil (§9).

- `velocidade` é o TETO de `lin_vel_x` e `lin_vel_y` do comando sorteado, por env. O
  `ang_vel_z` acompanha na mesma proporção do cfg do fabricante.
- `giro` é a rotação COMANDADA (alvo posto a N graus da orientação ATUAL da caixa),
  não o quanto ela nasce torta. O último nível (360) é o salto qualitativo "a face
  alvo pode ser o topo ou o fundo" — exige erguer e rolar a caixa entre as palmas.

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
    PEGAR: {"altura": 0},
    REORIENTAR: {"giro": 0},
    LOCOMOVER_CARREGANDO: {"velocidade": 0},
    BOTAR: {"altura": 0},
}
"""O eixo de cada tarefa e o índice INICIAL dele.

Um eixo por tarefa, e isso não é economia: é o que faz o condicionamento da medição
ser desnecessário. Com dois eixos, a célula de um mede marginalizada sobre o outro, e
o portão trava — foi o bug de 06/08.

O `locomover_carregando` fica com `velocidade`, e não com `peso`, porque o peso não é
eixo. O eixo `altura` do `pegar` e do `botar` continua sendo a posição da prateleira.
"""


def exceto(*excluidas: int) -> tuple[int, ...]:
    """Todas as tarefas menos as passadas. Para gate "ligado em tudo, MENOS em X"."""
    fora = set(excluidas)
    return tuple(t for t in range(NUM_TASKS) if t not in fora)


def eixo_de(task: int) -> str:
    """O nome do eixo da tarefa. Um por tarefa, por construção."""
    (nome,) = AXES[task]
    return nome


def axis_levels(task: int, axis: str) -> tuple[float, ...]:
    """Os níveis que a tarefa de fato usa nesse eixo (já cortando o início)."""
    return LEVELS[axis][AXES[task][axis]:]


def unlock_count() -> dict[str, int]:
    """Quantos destravamentos cada fonte contribui. Tem que somar 24 (§3).

    Serve de teste: a conta é DERIVADA dos níveis e dos índices iniciais, então se
    alguém mexer num nível sem querer o total denuncia.

    ⚠️ O alargamento da DR de peso NÃO conta aqui. Ele não é destravamento — ele pega
    carona no primeiro evento de cada tarefa com caixa."""
    out = {NAMES[t]: 0 for t in AXES}
    for task, axes in AXES.items():
        for axis, start in axes.items():
            out[NAMES[task]] += len(LEVELS[axis]) - 1 - start
    out["aberturas"] = NUM_TASKS - 1  # a 1ª tarefa já nasce aberta
    return out


def total_unlocks() -> int:
    return sum(unlock_count().values())
