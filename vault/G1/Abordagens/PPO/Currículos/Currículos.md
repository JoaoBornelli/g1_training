---
tipo: frente
pai: "[[PPO]]"
estado: ativa
verificado: 2026-10-06
resumo: "Um modelo só para várias tarefas: C1 multitask, C2 g1_poc (superados), C3 g1_limpo (ativo), C4 v3 (em desenho)."
---
Um modelo para várias tarefas, com currículo.

## Filhos
<!-- filhos:inicio -->
- **C1 multitask** (abordagem · superada · 30/07–18/08/26) — Uma política com one-hot e orquestrador por célula. Andou e pegou SEPARADO; o pegar parou em 3 cm e o std fugiu. Superado pelo C2.
- **C2 g1_poc** (abordagem · superada · 18/08–25/08/26) — Uma tarefa, dois comandos; o alvo da caixa define o comportamento. Pegou e reorientou SEM ANDAR; portão do balanço media sobrevivência. Superado pelo C3.
- **C3 g1_limpo** (abordagem · ativa · 25/08–05/10/26) — Máquina de elo com locomoção primeiro, em duas linhas: a linha bloco (resume encadeado) gerou o 21700, e o refino v2 dele deu o model_23250, MELHOR modelo; a linha zero (12 recomeços) não produziu política competente.
- **C4 v3** (abordagem · em-desenho) — Quarta abordagem, em desenho: pacote g1_v3 treinado do zero só com as peças que funcionaram — potencial de progresso em alvos macro de IK, dois balanceadores, refino em degraus, locomoção da Unitree; mjlab primeiro, Isaac Lab depois.
<!-- filhos:fim -->
