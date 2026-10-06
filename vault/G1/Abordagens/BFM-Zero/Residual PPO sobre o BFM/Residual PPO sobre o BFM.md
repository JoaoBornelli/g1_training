---
tipo: abordagem
pai: "[[BFM-Zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 30/07–03/08/26
resumo: "PPO residual sobre o BFM congelado, com busca de z. Rejeitado: o BFM não estabiliza o agachamento e o residual teria de fazer todo o equilíbrio."
cluster: bfm
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
<!-- filhos:inicio -->
- **BFM só equilíbrio** (abordagem · rejeitada · 03/08/26) — BFM fixo em ficar de pé, todo o resto no residual; autoridade medida e pose → z rejeitado; linha parada em 03/08.
- **Montagem e portão** (abordagem · validada · 30/07/26) — Ator do BFM em lote no mjlab, validado contra o fabricante (erro 1e-6); registra 7 armadilhas silenciosas da obs.
- **Treino com busca de z** (abordagem · rejeitada · 30–31/07/26) — RL emite 29 residuais + 20 coeficientes de z; parado funcionou, pegar nunca agachou; escala_c engolia o gradiente.
<!-- filhos:fim -->
