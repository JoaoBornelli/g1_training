---
tipo: frente
pai: "[[Abordagens]]"
estado: ativa
verificado: 2026-10-06
resumo: "Treino por PPO no mjlab: Tarefas isoladas (superada), Currículos C1–C4 e Estudos. Ativo: C3 g1_limpo (model_23250) e C4 v3 em desenho."
cluster: ppo
---
Treino por PPO (rsl_rl) no mjlab, com MuJoCo Warp, na Kaggle (2× T4, 4096 envs) e no Colab (8192 envs). Orçamento real de ~30 mil iterações por tarefa completa.

Linha do tempo:
- **Tarefas isoladas** (15–24/07): uma política por skill. O Lift Box pegou e ergueu, mas não encadeia com andar e esqueceu o Stand.
- **C1 multitask** (30/07–18/08): uma política com one-hot e orquestrador por célula; andou e pegou separado.
- **C2 g1_poc** (18–25/08): uma tarefa, dois comandos; pegou e reorientou sem andar.
- **C3 g1_limpo** (25/08–05/10): locomoção primeiro e máquina de elo; a linha bloco gerou o `model_21700`, refinado no `model_23250` (MELHOR).
- **C4 v3** (desde 01/10): refazer simples com alvos macro de IK e potencial de progresso; ainda sem código.
- **Estudos:** literatura e métodos estudados em paralelo, sem treino próprio.

Regras de desenho que o PPO ensinou estão em Princípios de recompensa; armadilhas de ferramenta em Infra.

## Filhos
<!-- filhos:inicio -->
- **Currículos** (frente · ativa) — Um modelo só para várias tarefas: C1 multitask, C2 g1_poc (superados), C3 g1_limpo (ativo), C4 v3 (em desenho).
- **Estudos** (frente · ativa) — Estudos em paralelo ao treino.
- **Tarefas isoladas** (frente · superada · 15/07–24/07/26) — Uma política por skill (Stand, Lift Box). Funcionou isolado, mas não encadeia: esqueceu o Stand e não anda. Superada pelos currículos.
<!-- filhos:fim -->
