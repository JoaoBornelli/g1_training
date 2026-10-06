---
tipo: abordagem
pai: "[[BFM-Zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 23/07/26
resumo: "BFM nas pernas e cintura, IK nos braços. Rejeitado: o alcance baixo fica preso ao envelope do BFM em pé; cintura no IK deixa o tronco mole."
---
## Hipótese
Dividir o corpo: o BFM controla pernas e cintura (equilíbrio), o IK controla só os braços (alcance), os dois todo passo.

## Método
`ik_bfm_coupled.py`: BFM nas juntas 0–14 com `z` = "ficar em pé" mais inclinação opcional (`LEAN`) e agachamento (`SQUAT`); mink nas juntas 15–28 em malha fechada, com base, pernas e cintura congeladas no IK. Um vetor de ação só, aplicado pelo PD do env.

## Resultado
Dar a cintura ao IK deixava o tronco MOLE (sai do laço de equilíbrio do BFM). Com a cintura de volta ao BFM o tronco fica firme, e o alcance baixo depende só da inclinação que o BFM aceita segurar.

## Por que parou
O alcance baixo fica preso ao envelope do BFM em pé (dentro do manifold). Não chega à caixa no chão.

## O que herda
A regra: a cintura pertence ao laço de equilíbrio.
