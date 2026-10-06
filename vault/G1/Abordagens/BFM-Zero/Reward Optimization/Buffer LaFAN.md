---
tipo: achado
pai: "[[Reward Optimization]]"
estado: ativa
verificado: 2026-10-06
periodo: 22/07/26
---
## Fato
O buffer de reward inference precisa de ~100k+ frames diversos; o `sample_data.npz` tem só 1000.

## Evidência
`build_lafan_buffer.py` varre o `lafan_29dof.pkl` (40 movimentos, ~264k frames) com as funções do próprio env (`set_state` + `_create_observation_backward`). Saídas: `lafan_buffer.npz` (662 MB) e `lafan_buffer_small.npz` (stride 8, 125 MB).

## Implicação
Com o buffer completo a inferência tem suporte máximo, e o alvo baixo continua fora. Isso isola a causa: não é falta de amostra, é falta da pose no dataset.
