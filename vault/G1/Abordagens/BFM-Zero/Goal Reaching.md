---
tipo: abordagem
pai: "[[BFM-Zero]]"
estado: rejeitada
verificado: 2026-10-06
---
## Hipótese
Uma pose alvo leva o BFM à postura de pega.

## Método
goal_track.py: pose por slider, erro por bloco.

## Resultado
O BFM passa pela pose e não a segura.

## Por que parou
Overdrive por alvo rejeitado por medição.

## O que herda
Ferramenta goal_track.

## Filhos
```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT tipo
```
