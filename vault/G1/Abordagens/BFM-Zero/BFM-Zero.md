---
tipo: frente
pai: "[[Abordagens]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 22/07–06/08/26
---
## Objetivo

Usar o BFM-Zero pré-treinado (LeCAR-Lab, Forward-Backward com CPR) como controlador do G1 sem treinar do zero. A meta concreta: levar as mãos perto do chão, numa pose parecida com a pega de uma caixa, e segurar essa pose.

## Ambiente

- Código e scripts: `~/Documents/BFM-Zero` (sem git; branch de origem `minimal_inference`). Treino residual: `g1_training/g1_residual/`, branch `exp/residual-bfm`.
- A inferência roda em CPU. O treino do BFM exige Isaac Lab e GPU NVIDIA.
- O BFM é comandado por um latente `z` de 256 números com norma fixa 16. `z` vem de `B(estado)`, o backward map.
- Três modos de obter `z`: reward inference (Reward Optimization), goal inference (Goal Reaching) e tracking inference (Tracking). O paper também descreve few-shot adaptation (Few-shot).

## Veredito da frente

**Nenhuma abordagem funcionou.** Nenhuma leva as mãos à pose de pega no chão e segura.

A causa é única: o dataset de treino (LaFAN, 40 movimentos, ~264k frames) não tem nenhuma pose de agachamento parado nem algo parecido. A política passa pela pose profunda e não para nela. O mapa comandado → alcançado (ver Mapa comandado-alcançado) mostra os dois atratores: a mão para num piso de ~0,54 m (meio abaixado) ou o robô cai e termina deitado (pelve ~0,38 m).

A causa está nos dados, não no ajuste. Uma abordagem nova sobre o BFM-Zero só vale se trouxer dados com essa pose, ou um fine-tune.

## O que a frente deixou para as outras

- Ferramentas de viabilidade de pose (`pose_check.py`, `find_pose.py`): separam "o controlador falhou" de "nenhum controlador conseguiria".
- O controle por IK em malha (`ik_control.py`) segura a pega sem o BFM. Ele seguiu no cluster Controle determinístico.
- O desenho do residual (ação somada a um controlador congelado) e as lições do portão de referência.

## Filhos

```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT file.name
```
