---
tipo: frente
pai: "[[G1]]"
estado: ativa
verificado: 2026-10-06
resumo: "Macro das formas de controlar o G1: BFM-Zero (rejeitada), PPO (ativa, C3 g1_limpo e C4 v3) e Controle determinístico (IK + PD)."
---
Macro das abordagens de controle do G1: BFM-Zero, PPO, controle determinístico.

## Filhos
<!-- filhos:inicio -->
- **BFM-Zero** (frente · rejeitada · 22/07–06/08/26) — Usar o BFM-Zero pré-treinado como controlador. 6 abordagens, todas rejeitadas: o LaFAN não tem pose de agachamento parado; a mão para em ~0,54 m ou o robô deita.
- **Controle determinístico** (frente · ativa) — Controle sem RL no MuJoCo clássico (unitree-g1-sim, lidar_sim): IK + PD segura a pega; o autônomo fase 1 executa a missão A com oráculo e a política 24350.
- **PPO** (frente · ativa) — Treino por PPO no mjlab: Tarefas isoladas (superada), Currículos C1–C4 e Estudos. Ativo: C3 g1_limpo (model_23250) e C4 v3 em desenho.
<!-- filhos:fim -->
