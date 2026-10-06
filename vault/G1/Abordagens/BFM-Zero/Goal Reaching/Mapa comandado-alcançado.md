---
tipo: achado
pai: "[[Goal Reaching]]"
estado: ativa
verificado: 2026-10-06
periodo: 24/07/26
---
## Fato
O BFM tem um piso de mão em ~0,54 m. Abaixo de 0,35 m comandado ele cai ou termina deitado.

## Evidência
`sweep_map.py` → `sweep_map.npz`. O mink gera a pose para cada altura de mão; mede-se o erro do IK e o erro do BFM separados.

| mão comandada (m) | 0,70 | 0,60 | 0,50 | 0,40 | 0,35 | 0,30 | 0,25 | 0,20 | 0,15 | 0,10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mão alcançada (m) | 0,69 | 0,58 | 0,55 | 0,54 | 0,57 | 0,15 | 0,15 | 0,16 | 0,17 | 0,18 |
| pelve (m) | 0,74 | 0,69 | 0,69 | 0,70 | 0,70 | 0,38 | 0,38 | 0,39 | 0,42 | 0,42 |
| caiu | não | não | não | não | não | sim | sim | sim | não | não |

Erro do IK ≤ 7 mm em todas as alturas. Erro do BFM de 0,05 a 0,14 m até 0,40 m; acima de 0,5 m dali para baixo.

## Implicação
Dois atratores, sem nada entre eles: meio abaixado (pelve ~0,70, mão ~0,54) ou deitado (pelve ~0,38). O mapa não é monotônico, então o overdrive (mandar mais baixo para chegar ao alvo) não passa do piso. Só fine-tune ou dados novos passariam.
