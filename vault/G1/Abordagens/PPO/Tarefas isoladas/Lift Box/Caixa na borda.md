---
tipo: abordagem
pai: "[[Lift Box]]"
estado: validada
verificado: 2026-10-06
periodo: 16/07/26
resumo: "Caixa no centro da mesa era inalcançável em pé; mover para a borda resolveu sem penalidade nova."
cluster: tarefas-isoladas
---
## Hipótese
O robô se escora e abre um espacate porque a caixa no centro da mesa é fisicamente inalcançável em pé.

## Método
`box_edge`: a caixa vai do centro (x 0,50, sobre 30 cm de tampo) para a borda da mesa — ~0,20 m mais perto, nada mais muda. O alvo de erguer volta para dentro do alcance (`target_x` 0,40–0,50 → 0,20–0,30). `box_xy_jitter` ±10 cm.

## Resultado
O robô pega e ergue reto (checkpoint derivado do 4700).

## Por que parou


## O que herda
Atacar a geometria, não empilhar penalidade sobre o sintoma.
