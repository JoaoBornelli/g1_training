---
tipo: decisao
pai: "[[C1 multitask]]"
estado: superada
superada_por: C2 g1_poc
verificado: 2026-10-06
periodo: 30/07–18/08/26
---
## Decisão
Uma política, um orquestrador que é termo de currículo e roda no reset antes de eventos e comando. Estado em grade de células `(tarefa, eixo)`, cada uma com níveis, EMA de sucesso (`alpha` 0,03) e contagem de amostras.

## Regra do evento
Com `episódios(T) ≥ 200` e `perf[T][topo] ≥ 0,90`: abre a DR de peso se fechada, senão `abertos[T] += 1`; no mesmo evento abre todo filho cujos pais já tiveram o 1º evento. Amostragem de tarefa inversa à competência com piso 0,15 (medido com 3 abertas: 0,517 / 0,333 / 0,150).

## Tarefas e eixos (18/08)
| tarefa | eixo | episódio | sustentação |
|---|---|---|---|
| `locomover` | velocidade (1,0) | 20 s | fecha no `time_out` |
| `pegar` | alvo (0,2–1,0 da rampa) | 10 s | 2,0 s |
| `reorientar` | giro (15°) | 10 s | 2,0 s |
| `locomover_carregando` | velocidade (1,0) | 20 s | 3,0 s |
| `botar` | altura (0,55) | 10 s | 2,0 s |

Níveis disponíveis: velocidade 1,0–2,0 m/s; altura de prateleira 0,55 → 0,0 m; giro 15–360°; peso 1 ou U(1,5) kg (DR, não eixo). Congelados em 1 nível, exceto o `alvo` do `pegar`: os 12 destravamentos concentrados no gargalo.

## Evolução
7 tarefas (30/07: parado, andar, pegar, reorientar, parado c/ caixa, andar c/ caixa, botar) → 5 tarefas (07/08: parado+andar viram `locomover`). Destravamentos: 60 → 54 → 42 → 24 → 12. Congelamento por célula criado e depois removido (era o único mecanismo capaz de bloquear a abertura de um filho).

## Por que foi superada
Complexidade de currículo sem resultado proporcional. O C2 colapsou tudo num objetivo só.
