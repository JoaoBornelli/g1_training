---
tipo: frente
pai: "[[G1]]"
estado: ativa
verificado: 2026-10-06
resumo: "Macro das formas de controlar o G1: BFM-Zero (rejeitada), PPO (ativa, C3 g1_limpo e C4 v3) e Controle determinístico (IK + PD)."
cluster: abordagens
---
As formas de controlar o G1 que o projeto testou, em ordem de tempo:

1. **BFM-Zero** (22/07–06/08): usar um modelo de comportamento pré-treinado como controlador. Seis abordagens, todas rejeitadas, pela mesma causa: o dataset LaFAN não tem agachamento parado com as mãos perto do chão.
2. **PPO** (15/07 em diante): treino próprio no mjlab. Começou por tarefas isoladas (Stand, Lift Box), passou por três currículos de modelo único (C1 multitask, C2 g1_poc, C3 g1_limpo) e está no desenho do C4 v3. O melhor modelo é o `model_23250` do C3.
3. **Controle determinístico** (julho e outubro): IK + PD em malha, sem aprendizado. Segura a pega no MuJoCo clássico e é a base do controlador autônomo do depósito, que chama a política treinada como executor.

Lição que atravessa as três: o que não está nos dados ou na recompensa não emerge — nem a pose do agachamento no BFM, nem a marcha num currículo que não a paga, nem a pega sem gradiente na coordenada do aperto.

## Filhos
<!-- filhos:inicio -->
- **BFM-Zero** (frente · rejeitada · 22/07–06/08/26) — Usar o BFM-Zero pré-treinado como controlador. 6 abordagens, todas rejeitadas: o LaFAN não tem pose de agachamento parado; a mão para em ~0,54 m ou o robô deita.
- **Controle determinístico** (frente · ativa) — Controle sem RL no MuJoCo clássico (unitree-g1-sim, lidar_sim): IK + PD segura a pega; o autônomo fase 1 executa a missão A com oráculo e a política 24350.
- **PPO** (frente · ativa) — Treino por PPO no mjlab: Tarefas isoladas (superada), Currículos C1–C4 e Estudos. Ativo: C3 g1_limpo (model_23250) e C4 v3 em desenho.
<!-- filhos:fim -->
