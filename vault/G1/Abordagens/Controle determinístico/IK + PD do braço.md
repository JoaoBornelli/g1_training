---
tipo: abordagem
pai: "[[Controle determinístico]]"
estado: validada
verificado: 2026-10-06
---
## Hipótese
IK numérica + PD controla o braço.

## Método
unitree-g1-sim.

## Resultado
Travar a mão foi ganho, não método.

## Por que parou


## O que herda


## Filhos
```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT tipo
```
