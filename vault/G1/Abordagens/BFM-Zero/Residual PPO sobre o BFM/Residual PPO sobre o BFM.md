---
tipo: abordagem
pai: "[[BFM-Zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 30/07–03/08/26
---
## Hipótese
Um PPO residual sobre o BFM congelado aprende a habilidade e o BFM dá o equilíbrio: `alvo = a_BFM(obs, z)·escala + pose_BFM + clamp(Δ)`. A RL também procura o `z` numa base de 20 direções.

## Método
Pacote `g1_residual/`, branch `exp/residual-bfm`, sobre o env do `g1_multitask` (7 tarefas, currículo em cascata). Treino no Kaggle. Três fases: Montagem e portão, Treino com busca de z, BFM só equilíbrio.

## Resultado
O `parado` funcionou e o robô andou. `pegar` nunca agachou; `grasp`, `lift`, `box_at_peito`, `box_at_prateleira`, `orienta_face` e `hold_still` ficaram em zero.

## Por que parou
O BFM não estabiliza o agachamento, então o residual teria de fazer o equilíbrio e não só o alcance. Com o BFM reduzido a "ficar de pé", o residual vira uma política monolítica mais cara. A linha parou em 03/08; o trabalho seguiu no `g1_multitask` e depois no `g1_limpo`.

## O que herda
Ver os filhos: portão de referência, terminação × hack de sentar, canal inerte como poço de entropia, clamp que escala a exploração.

## Filhos
```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT file.name
```
