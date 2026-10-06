---
tipo: abordagem
pai: "[[BFM-Zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 22/07/26
---
## Hipótese
Uma recompensa de alcance (mãos num alvo) inferida pelo BFM dá o `z` da tarefa, e o robô executa zero-shot.

## Método
Reward inference do BFM-Zero: o `z` é a média de `B(s)·r(s)` sobre um buffer de estados rotulados pela recompensa. Alvos de mão egocêntricos (peito, baixo à frente, alto aberto). O buffer padrão tinha só 1000 frames, e por isso foi construído um buffer LaFAN grande (ver Buffer LaFAN).

## Resultado
- O robô alcança alvos dentro do que o LaFAN cobre, mas não alcança o alvo baixo com precisão.
- `z` não é setpoint: `project_z` normaliza para norma 16 e descarta a magnitude. Pedir o alvo mais longe GIRA o `z` e troca o comportamento, em vez de intensificar o pedido.
- O punho não desce abaixo de ~0,54 m.

## Por que parou
Limite de cobertura do buffer: a recompensa só vira `z` útil onde existem estados parecidos no dataset. Fechou por medição, não por ajuste.

## O que herda
A noção de que a magnitude do pedido não existe no BFM. Motivou o Goal Reaching (entregar a pose em vez da recompensa) e o Few-shot (otimizar o `z` direto).

## Filhos
```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT file.name
```
