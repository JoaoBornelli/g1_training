---
tipo: abordagem
pai: "[[BFM-Zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 22/07/26
resumo: "Few-shot do paper: CEM sobre z em 3 alvos. Rejeitado: a mão se aproxima do chão mas não o bastante; fica meio abaixado ou deita."
---
## Hipótese
A few-shot adaptation do paper (single-pose) atinge um alvo de mão preciso que o reward inference zero-shot não atinge: otimizar só o `z`, com F, B e o ator congelados.

## Método
`few_shot_adapt.py`: CEM sobre o `z`, com warm-start do `z` zero-shot. Objetivo = distância real punho → alvo medida rodando a política no sim. População 12, elite 4, 20 iterações, σ inicial 4,0; σ do alvo 0,15.

Alvos (x frente, y lado, z altura, egocêntricos):
- alvo 1, peito: (0,25, ±0,20, 1,00)
- alvo 2, baixo à frente: (0,25, ±0,25, 0,20) — o da pega
- alvo 3, alto aberto: (0,10, ±0,10, 1,50)

Saídas: `z_fewshot_alvo{1,2,3}.npy`.

## Resultado
Similar ao Goal Reaching. No alvo 2 a mão se aproxima do chão, mas não o bastante para pegar uma caixa no chão. A política converge para meio abaixado ou deita no chão.

## Por que parou
O CEM só procura dentro do que o ator sabe fazer. Nenhum `z` leva a uma pose que não está no dataset de treino.

## O que herda
