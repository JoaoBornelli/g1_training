---
tipo: abordagem
pai: "[[BFM-Zero]]"
estado: rejeitada
verificado: 2026-10-06
---
## Hipótese
Um PPO residual sobre o BFM congelado faz a tarefa.

## Método
g1_residual/: ação residual somada ao ator do BFM.

## Resultado
O BFM não estabiliza o agachamento; o residual faz o equilíbrio.

## Por que parou
pose → z rejeitado.

## O que herda


## Filhos
```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT tipo
```
