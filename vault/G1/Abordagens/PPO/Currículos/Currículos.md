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
- **C3 g1_limpo** (abordagem · ativa) — Melhor checkpoint: model_23250.
- **C4 v3** (abordagem · em-desenho) — Quarta abordagem de currículo, em desenho: refazer simples só com o que funcionou, IK como alvo macro, locomoção da Unitree, Isaac Lab.
<!-- filhos:fim -->
