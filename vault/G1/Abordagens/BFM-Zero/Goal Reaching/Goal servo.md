---
tipo: abordagem
pai: "[[Goal Reaching]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 24/07/26
resumo: "Goal em malha fechada com overdrive integral. Rejeitado: mover o alvo gira o z em vez de empurrar mais."
---
## Hipótese
Como não dá para pedir "mais forte", pedir "mais longe": um goal que anda em malha fechada, mais uma ação integral (overdrive), destrava o alvo baixo.

## Método
`goal_servo.py`: o goal é reconstruído do estado real a ~10 Hz com os braços empurrados rumo ao alvo pelo Jacobiano amortecido (o goal fica perto do estado, em distribuição). Overdrive: `bias += Ki·erro`, e `squat += Ki·erro_vertical` desce o corpo quando os braços saturam.

## Resultado
O overdrive não converge: mover o alvo gira o `z` e troca a pose, em vez de intensificar.

## Por que parou
Overdrive por alvo REJEITADO (06/08): não existe integral possível sobre um `z` sem magnitude.

## O que herda
