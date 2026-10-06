---
tipo: abordagem
pai: "[[C2 g1_poc]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 20–24/08/26
resumo: "O controlador de forma passava a fatia de andar à manipulação ao medir sobrevivência; o robô nunca andou."
---
## Hipótese
Um controlador automático de forma passa a fatia de locomoção para a manipulação quando a locomoção estiver apta.

## Método
`_alvo_locomocao`: `apto = dur_loco_ema ≥ 600 passos` (12 s de 20), volta com `dur < 480`. Fatia nasce em 1,00 e desce 0,02 a cada 12 iterações. Mais a janela de espera: comando zero no começo de todo episódio.

## Resultado
O robô pegava a caixa no nível 1 e NUNCA aprendia a andar (`peak_height_mean` raramente acima de 0,03 contra 0,10).

## Por que parou
Duração mede SOBREVIVÊNCIA: um robô imóvel marca 1000 passos e passa no portão; em 420 iterações a fatia chegava a 0,30 sem marcha nenhuma, e voltar exigia o robô CAIR. Causa raiz confirmada com o dono em 24/08; o C3 começou com locomoção primeiro e fatia 100%.

## O que herda
- Um portão de habilidade precisa medir a habilidade (marcha), não sobreviver.
- A janela de espera entregava 100% de parado no início (o fabricante entrega 10% via `rel_standing_envs`) e colava o "fique parado" no estado de RESET; com o bootstrap V=0 da queda, o viés fechava. A janela foi tirada da locomoção (teoria do dono, 24/08).
