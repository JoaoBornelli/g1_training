---
tipo: abordagem
pai: "[[Lift Box]]"
estado: validada
verificado: 2026-10-06
periodo: 15–16/07/26
resumo: "Deitar a caixa pagava altura de graça; gate de 10° e depois kernel suave de orientação."
cluster: tarefas-isoladas
---
## Hipótese
Medir a altura do centro de uma caixa que pode tombar é gameável; um fator de orientação fecha o hack.

## Método
- `gate_and_com` (15/07): gate duro de orientação em ~10° + recompensa de CoM sobre os pés.
- 16/07: o fator vira kernel suave `exp(−(1−cos θ)/upright_std)`, `upright_std` 0,1; `action_rate_l2` −0,1 → −0,25.
- `box_shake` −0,15: Σ(vel. angular da caixa)², denso, só angular.

## Resultado
Deitar a caixa na diagonal subia o centro 4 cm e pagava `lift` de graça; o gate fechou. O kernel suave dá gradiente em toda a faixa e conserta "a caixa roda na mão".

## Por que parou


## O que herda
Gate duro tem derivada zero fora do corte; o kernel suave puxa de qualquer ângulo.
