---
tipo: abordagem
pai: "[[Residual PPO sobre o BFM]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 03/08/26
resumo: "BFM fixo em ficar de pé, todo o resto no residual; autoridade medida e pose → z rejeitado; linha parada em 03/08."
---
## Hipótese
Com o BFM fixo em "ficar de pé" (`move-ego-0-0`, semente 0), andar, agachar, alcançar e segurar saem inteiros do residual de junta.

## Método
- `autoridade.py`: o residual vence o BFM? Malha aberta, sem PPO, direção de descida tirada da pose do `crouch-0.25`.
- `poses.py`: visor dos 41 comportamentos × 10 sementes, para escolher o prior olhando.
- `escala_c` → 0; clamp sagital (`hip_pitch`, `knee`, `ankle_pitch`, `waist_pitch`) de 0,35 → 1,05 rad; ação de 49 → 29.
- `pose → z` por `goal_inference` com o backward map extraído.

## Resultado
Autoridade (300 passos):

| fração pedida | pelve mín | palma mín | ereto |
|---|---|---|---|
| 0,00 | 0,770 | 0,647 | 100% |
| 0,25 | 0,567 | 0,470 | 96% |
| 0,50 | 0,068 | 0,037 | 19% (caiu) |

O BFM não resiste ao residual (ele consegue 2,5–3× o deslocamento pedido); o que limita é o clamp e a estabilidade da descida. `pose → z` REJEITADO: o `privileged_state` (463 dos 527 números do B) é zero no rastro do fabricante, o bloco fixo domina a pré-ativação em 15,2× e o `z` derivado sai constante (cos +0,079 contra +0,076 de acaso).

## Por que parou
O BFM não estabiliza o agachamento: o residual teria de fazer equilíbrio e alcance, e o BFM só custa throughput. Linha parada em 03/08.

## O que herda
- Canal inerte não é neutro: é POÇO DE ENTROPIA. 20 canais sem efeito deixavam o PPO inflar o `std` deles e colher bônus de graça (41% do orçamento de entropia).
- O clamp escala também a exploração inicial: limite 0,35 → 2,9°; 1,05 → 8,6°; 2,10 → 17,1°. Âncora: a monolítica aprendeu a ficar de pé explorando ±18°.
