---
tipo: abordagem
pai: "[[Lift Box]]"
estado: validada
verificado: 2026-10-06
periodo: 16/07/26
resumo: "Variar a caixa e empurrar sob carga; rehearsal de 20% contra o esquecimento do Stand."
---
## Hipótese
Com a pega dominada, variar a posição da caixa e empurrar sob carga endurece a skill; misturar ficar de pé impede o esquecimento.

## Método
- `generalize`: caixa x +0,0…+0,20 (assimétrico; −x cairia da mesa), y ±0,18; `push_robot` ±0,8 + força ±50 N.
- `rehearsal`: o checkpoint do Lift no Stand-Step tinha DESAPRENDIDO a ficar de pé. `rehearsal_fraction` 0,2: 20% dos envs com caixa e mesa a 5 m, só a fundação e o push ativos.

## Resultado
O equilíbrio volta sem perder a pega.

## Por que parou


## O que herda
Esquecimento catastrófico entre skills é real; rehearsal é o remédio dentro de uma skill, e motivou o modelo único.
