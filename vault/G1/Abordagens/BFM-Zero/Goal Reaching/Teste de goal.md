---
tipo: abordagem
pai: "[[Goal Reaching]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 23–24/07/26
resumo: "Poses montadas à mão viram z; braços e frames do dataset funcionam, agachar não (a pose 6 era impossível)."
---
## Hipótese
Poses montadas à mão (braços, agachar) viram `z` e o BFM as executa.

## Método
`teste_goal.py`: teclas 1–3 poses de braço, 5–6 agachar, 4 um frame real do dataset (sanidade). Inclinação do tronco ao vivo (`.`/`,`) recalcula o `z` com o quaternion da base inclinado.

## Resultado
Frame do dataset e poses de braço: o BFM reproduz. Poses de agachar (5 e 6): o BFM não chega ou cai.

## Por que parou
A pose 6 era estaticamente impossível (ver Viabilidade da pose). A falha não era só do BFM.

## O que herda
As poses de referência em `~/Documents/BFM-Zero/poses/` (`default_em_pe`, `pegar`, `pose_10cm`).
