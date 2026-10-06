---
tipo: achado
pai: "[[Lift Box]]"
estado: ativa
verificado: 2026-10-06
periodo: 06/08/26
---
## Fato
Quando o alvo do comando é a própria caixa, o `progress` do `lift` vale sempre 1: o `lift` (peso 2,0) paga encostar as palmas com a caixa nivelada, não erguer.

## Evidência
`progress = clamp((box_z − rz)/(target_z − rz))` com `target_z = command[:,2] = box_z`. Movendo a caixa +0,10 e +0,60 m: `progress` 1,0000; em +0,00: NaN. Com o `rest_z` por env do PLR, `box_z == rest_z` no 1º passo de todo episódio, e `0 × NaN = NaN` chega ao PPO. Tell: `erro_posicao` em 0,0000 no TensorBoard.

## Implicação
Separar alvo de NAVEGAÇÃO de alvo de TAREFA: o `lift` recebe a altura do peito. Achado já no `g1_multitask` (auditoria de 06/08), defeito herdado do `rewards.py` do Lift.
