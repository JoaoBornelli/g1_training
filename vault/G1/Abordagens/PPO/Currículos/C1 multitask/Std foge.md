---
tipo: achado
pai: "[[C1 multitask]]"
estado: ativa
verificado: 2026-10-06
periodo: 17–18/08/26
---
## Fato
O `std` da política sobe sozinho (0,43 → 0,98; 0,43 → 0,70 em 300 iterações) e, acima de ~0,50, nenhuma tarefa pontua.

## Evidência
`erro_vel_ang ≈ 0,591 + 1,606·(std − 0,43)` cruza `tol_w` 0,70 em `std` ≈ 0,498. Em it 23548: `sucesso` 0,0000 e `ALARME_estagnacao` com 7,9e8 transições. Causas: a força da entropia é ∝ 1/σ; `std_param` é vetor de 29; `action_rate_bracos` 0,25 cortou ¾ do freio nos braços. Realimentação: `track_angular_velocity` desabou 0,467 → 0,096.

## Implicação
Conserto proposto e não aplicado: `param.data.clamp_(max=0.45)` no mesmo gancho do runner. Reaquecer o `std` só serve para gradiente ESPARSO. Ver também a nota sobre o std da ação no C3.
