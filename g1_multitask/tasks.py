"""As 7 tarefas: índice no one-hot, eixos de currículo, e a conta dos 60 eventos.

Este módulo é a ESPINHA do desenho. Uma política só, condicionada a comando:
`ação = f(estado, comando)`. A tarefa NÃO é um env diferente — é um one-hot
dentro do vetor de comando. É isso que faz o algoritmo de visão, em campo,
entrar como mais um escritor do mesmo vetor ("gire a caixa pra esquerda") sem
tocar na política.

Os eixos são INDEPENDENTES, não uma grade cartesiana: `7+5+4+3+3` células em vez
do produto delas. Cada eixo sobe sozinho quando a tarefa dele mostra competência,
e o piso `ρ/L` do PLR mantém os níveis antigos no sorteio (anti-esquecimento).

Ver `Obsidian-documents/Robotics/G1-Curriculum-Design.md` §7, §7b, §9 e §12.
"""
from __future__ import annotations

# ---------------------------------------------------------------------- one-hot
# Índices FIXADOS pela §9. A ordem entra no vetor de comando E no checkpoint, e a
# fatia [9:17] da obs é posicional -> trocar um número aqui invalida todo
# checkpoint anterior (Categoria C do §15, "recomeçar do zero"). Não reordenar.
PARADO = 0
ANDAR = 1
PEGAR = 2
BOTAR = 3
REORIENTAR = 4
PARADO_CAIXA = 5
ANDAR_CAIXA = 6
RESERVADO = 7

NUM_TASKS = 7
ONEHOT_DIM = 8
"""8 slots com 7 em uso. É a ÚNICA reserva que sobrou do desenho (§9b): a reserva
de largura de observação saiu, porque crescer a obs depois é enxerto de 6 tensores
com saída bit a bit idêntica. 1 slot custa desprezível e cobre "acrescentei uma
tarefa" sem migrar nada. O 8º leva o tratamento completo — coluna zerada na 1ª
camada e stats do normalizador congeladas — porque é o mesmo caso do target_pos_b:
canal constante ganha `_std=0` no 1º update e, ao acender, entra 100x amplificado."""

NAMES = {
    PARADO: "parado",
    ANDAR: "andar",
    PEGAR: "pegar",
    BOTAR: "botar",
    REORIENTAR: "reorientar",
    PARADO_CAIXA: "parado_caixa",
    ANDAR_CAIXA: "andar_caixa",
}

# --------------------------------------------------------- conjuntos semânticos
# Usados por gates de reward, terminações e escopo de postura. Ficam aqui, num
# lugar só, pra a máscara do gate e a máscara do log saírem da MESMA fonte.

COM_CAIXA = (PARADO_CAIXA, ANDAR_CAIXA)
"""Tarefas em que CARREGAR é o estado exigido -> largar é falha (terminação).
No `pegar`, largar no meio deve permitir nova tentativa no mesmo episódio; no
`botar`, soltar É o objetivo. Só aqui largar termina o episódio (§6b/D)."""

PARADAS = (PARADO, PARADO_CAIXA)
MANIPULA = (PEGAR, BOTAR, REORIENTAR)
ANDA = (ANDAR, ANDAR_CAIXA)

SPAWN_SEGURANDO = (PARADO_CAIXA, ANDAR_CAIXA, BOTAR)
"""Tarefas que nascem com as PALMAS TOCANDO a caixa (§3, §4).

Não é o mesmo conjunto que `COM_CAIXA`: o `botar` também nasce segurando (o estado
final do `pegar` é o estado inicial canônico dele, §4), mas largar nele é o
OBJETIVO, não falha — por isso ele fica fora do gate da terminação `largou`.

Medido em 30/07: sem esta condição de spawn as 3 não têm caminho de aquisição.
Todos os termos de tarefa dão 0.0 no reset, porque a caixa nasce na prateleira e
`reaching`/`grasp`/`lift` são gateados só no `pegar`.

⚠️ Nascer segurando é nascer TOCANDO, com força normal zero. Segurar é o que a
tarefa ensina — ver `pregrasp.py`. Consequência: para estes envs o comando
pré-gatilho é `parado c/ caixa`, não `parado`, senão a caixa escorrega antes de a
política receber objetivo."""

# ------------------------------------------------------------- eixos e níveis
LEVELS: dict[str, tuple[float, ...]] = {
    "altura": (0.55, 0.45, 0.35, 0.25, 0.15, 0.05, 0.00),
    "peso": (1.0, 2.0, 3.0, 4.0, 5.0),
    "distancia": (0.0, 0.3, 0.8, 2.0),
    "heading": (60.0, 120.0, 360.0),
    "giro": (15.0, 45.0, 90.0, 180.0, 360.0),
    "push": (0.0, 0.35, 0.70, 1.00, 1.00),
}
"""Níveis de cada eixo, do fácil pro difícil (§14).

- `giro` é a rotação COMANDADA (alvo posto a N graus da orientação ATUAL da
  caixa), não o quanto ela nasce torta. O último nível (360) é o salto
  qualitativo "a face alvo pode ser o topo ou o fundo" — exige erguer e rolar
  a caixa entre as palmas, e só ele precisa da mão.
- `push` é o único eixo GLOBAL (vale pra todas as tarefas ao mesmo tempo), o
  único aninhado por construção (sem piso `ρ/L`) e o único com RECUO de nível.
  Os valores são o fator sobre os 6 componentes de perturbação; força fica em
  (0,0) nos níveis 0-2 e vai a ±50 N nos 3-4, com a duração alongando.
"""

AXES: dict[int, dict[str, int]] = {
    PARADO: {},
    ANDAR: {"distancia": 1, "heading": 0},
    PEGAR: {"altura": 0, "peso": 0, "distancia": 0},
    BOTAR: {"altura": 0, "peso": 0},
    REORIENTAR: {"giro": 0, "altura": 0, "distancia": 0},
    PARADO_CAIXA: {"peso": 0},
    ANDAR_CAIXA: {"peso": 0, "distancia": 1},
}
"""Eixos de cada tarefa e o índice INICIAL de cada um.

O índice inicial da distância vem da NATUREZA da tarefa (fechado 29/07): quem
ANDA começa em 0.3, porque distância 0.0 não faz sentido pra uma tarefa de
locomoção; quem MANIPULA começa em 0.0, porque a caixa ao alcance é o caso
fácil. `parado`, `parado c/ caixa` e `botar` não têm o eixo — no `parado` o
destino é a própria posição, e no `botar` o robô já está com a caixa na mão.

O `parado` não tem eixo nenhum: a robustez dele vem do `push`, que é global.
"""

# Ordem do desempate round-robin dentro de cada tarefa. Necessária porque, com um
# nível por eixo, toda célula recebe fluxo de episódios IDÊNTICO -> EMAs idênticas
# -> empate triplo por construção. A medição segue primária; isto só desempata.
AXIS_ORDER = ("giro", "altura", "peso", "distancia", "heading")


def exceto(*excluidas: int) -> tuple[int, ...]:
    """Todas as tarefas menos as passadas. Para gate "ligado em tudo, MENOS em X"."""
    fora = set(excluidas)
    return tuple(t for t in range(NUM_TASKS) if t not in fora)


def axis_levels(task: int, axis: str) -> tuple[float, ...]:
    """Os níveis que a tarefa de fato usa nesse eixo (já cortando o início)."""
    return LEVELS[axis][AXES[task][axis]:]


def unlock_count() -> dict[str, int]:
    """Quantos destravamentos cada fonte contribui. Tem que somar 60 (§14).

    Serve de teste: a conta é DERIVADA dos níveis e dos índices iniciais, então
    se alguém mexer num nível sem querer o total denuncia. O bug que isto pega
    já aconteceu no próprio doc — as duas linhas de distância contavam 3
    destravamentos ignorando que tarefas que andam começam em 0.3, e a tabela
    somava 62 com o total escrito 60."""
    out = {NAMES[t]: 0 for t in AXES}
    for task, axes in AXES.items():
        for axis, start in axes.items():
            out[NAMES[task]] += len(LEVELS[axis]) - 1 - start
    out["push"] = len(LEVELS["push"]) - 1
    out["aberturas"] = NUM_TASKS - 1  # a 1ª tarefa já nasce aberta
    return out


def total_unlocks() -> int:
    return sum(unlock_count().values())
