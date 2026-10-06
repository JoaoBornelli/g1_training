---
tipo: frente
pai: "[[Abordagens]]"
estado: ativa
verificado: 2026-10-06
resumo: "Controle sem RL no MuJoCo clássico (unitree-g1-sim, lidar_sim): IK + PD segura a pega; o autônomo fase 1 executa a missão A com oráculo e a política 24350."
---
Controle sem aprendizado por reforço, no MuJoCo clássico. Nasceu do BFM-Zero (o `ik_control.py` segurava a pega onde o BFM não segurava) e do relato do dono em 25/08: o treino entregava andar e pegar SEPARADOS, e a frente nova eram abordagens determinísticas de controle.

- **IK + PD do braço:** IK numérica (mink) em malha + PD com os ganhos oficiais; travar a mão foi ganho, não método.
- **Autônomo fase 1:** controlador do depósito (`lidar_sim/autonomo/`), missão A com oráculo, MuJoCo puro; o `model_24350` pega e pousa; zona morta pede rastreio e rampa.
- O G1 da UDESC é simulado local em MuJoCo porque a máquina não tem NVIDIA; no robô real, equilíbrio e marcha vêm do controlador da Unitree.

## Filhos
<!-- filhos:inicio -->
- **Autônomo fase 1** (abordagem · ativa) — O 24350 pega e pousa.
- **IK + PD do braço** (abordagem · validada) — Travar a mão foi ganho, não método.
- **g1-anda-e-pega-separado-controle-deterministico** (achado · ativa) — G1 em agosto/26 (relato do usuário 2026-08-25): treino preliminar dá andar E pegar a caixa SEPARADAMENTE, não encadeado; frente nova = abordagens determinísticas de controle; orçamento de GPU levantado no RunPods
- **unitree-g1-sim-mujoco-not-isaac** (achado · ativa) — G1 da UDESC simulado local — MuJoCo porque a máquina não tem NVIDIA; no robô real equilíbrio e marcha são serviço embutido, e os atuadores são torque puro
<!-- filhos:fim -->
