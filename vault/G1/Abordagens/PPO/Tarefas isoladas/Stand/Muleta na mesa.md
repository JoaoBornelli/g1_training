---
tipo: achado
pai: "[[Stand]]"
estado: ativa
verificado: 2026-10-06
periodo: 15/07/26
---
## Fato
Com a mesa ao alcance, o robô aprende a se ESCORAR nela (mãos e coxas) para satisfazer o `upright` sem equilibrar.

## Evidência
`upright` não distingue equilibrar de escorar. Conserto em `skills/stand/env.py`: caixa e mesa FORA de alcance útil.

## Implicação
Um Stand escorado é prior prejudicial para o Lift, que precisa de um ficar de pé que recupera sozinho. Também: penalizar contato de qualquer parte que não seja a palma.
