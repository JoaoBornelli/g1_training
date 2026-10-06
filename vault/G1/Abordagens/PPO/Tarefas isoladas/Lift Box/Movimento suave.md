---
tipo: abordagem
pai: "[[Lift Box]]"
estado: validada
verificado: 2026-10-06
periodo: 17–24/07/26
resumo: "Corre, bate, abre espacate e rebola; amortecer por escala de ação e gate, nunca penalidade sempre ligada."
---
## Hipótese
O robô corre, bate na mesa, abre espacate e rebola; amortecimento estrutural e gateado resolve sem brigar com a tarefa.

## Método
- `joint_acc` −2,5e-7 (corpo todo) e `joint_torque_pen` (briga com o payload; só como capacidade).
- `smooth_impact` (18/07): `action_scale_mult` 0,8 (estrutural, fora da soma de recompensa) + sensor de impacto tronco × mesa reusando o `soft_landing` do mjlab (só a pancada custa).
- `hip_deviation` (23/07): desvio L1 de `hip_roll`/`yaw` contra a perna esticada ao lado (`hip_roll` ~53° no `model_15700`); a postura gaussiana saturava.
- `hold_still` (24/07): quietude da pelve gateada no hold (anti-rebolado); os 5 termos existentes eram cegos.

## Resultado
Movimento mais calmo, sem mudar a forma de obs e ação (warm-start).

## Por que parou


## O que herda
Princípio anti-blanket: penalidade aditiva sempre ligada compete com a tarefa e é vencida. Hacks medidos: o robô CORRE para passar menos tempo no estado penalizado (o freio que não se burla é a aceleração de junta), e TRAVA uma perna em tarefa parada. Amortecer por escala de ação, por gate de fase ou com escopo estreito.
