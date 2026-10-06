---
tipo: abordagem
pai: "[[BFM-Zero]]"
estado: parada
verificado: 2026-10-06
---
## Hipótese
Uma recompensa inferida dá o z da tarefa.

## Método
Reward inference do BFM-Zero no G1.

## Resultado
z é direção sem magnitude.

## Por que parou
Envelope de capacidade insuficiente para a pose profunda.

## O que herda


## Filhos
```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT tipo
```
