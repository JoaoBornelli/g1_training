---
tipo: frente
pai: "[[PPO]]"
estado: ativa
verificado: 2026-10-06
resumo: "Um modelo só para várias tarefas: C1 multitask, C2 g1_poc (superados), C3 g1_limpo (ativo), C4 v3 (em desenho)."
cluster: ppo
---
Um modelo só para várias tarefas, com currículo que abre ou dosa as tarefas. Quatro tentativas:

| | período | ideia central | resultado | estado |
|---|---|---|---|---|
| C1 multitask | 30/07–18/08 | one-hot + orquestrador de células, orçamento igual por tarefa | anda e pega separado; std fugiu | superada |
| C2 g1_poc | 18–25/08 | uma tarefa, o alvo da caixa define o comportamento | pega e reorienta sem andar | superada |
| C3 g1_limpo | 25/08–05/10 | locomoção primeiro, máquina de elo, currículo de cadeia | `model_23250` faz o ciclo em 0,55–0,15 m | ativa (referência) |
| C4 v3 | desde 01/10 | alvos macro de IK, potencial, refino em degraus | em desenho | em-desenho |

O que passou de um para o outro: o `unload` como ponte do aperto (C1 → C2 → C3), terminar em vez de penalizar (C2), a máquina de elo (C2 → C3), locomoção pela receita do fabricante (C2 → C3 → C4), e a lição do C3 de que a recompensa refina uma política competente mas não a produz do zero (→ C4 treina em degraus com resume).

## Filhos
<!-- filhos:inicio -->
- **C1 multitask** (abordagem · superada · 30/07–18/08/26) — Uma política com one-hot e orquestrador por célula. Andou e pegou SEPARADO; o pegar parou em 3 cm e o std fugiu. Superado pelo C2.
<!-- filhos:fim -->
