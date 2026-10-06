---
tipo: abordagem
pai: "[[Goal Reaching]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 06/08/26
resumo: "Ferramenta de pose por slider com erro por bloco agora × melhor; mostrou que o robô passa pela pose e vai embora."
---
## Hipótese
Com uma ferramenta que mede o erro por bloco, dá para achar uma pose de pega que o BFM reproduz.

## Método
`goal_track.py` (irmãos `pose_io.py`, `pose_check.py`). Modo MONTAR: física e BFM desligados, o robô do viewer é a pose autorada por slider, aterrada pelos pés. Modo TRACKING: o BFM tenta chegar; mexer um slider recalcula o `z` ao vivo. Erro por bloco com o par `agora` (este passo) e `melhor` (mínimo dos últimos 2 s).

## Resultado
Linha de base: a pose padrão em pé dá 2,6° de erro médio. Nas poses de pega, `melhor` baixo com `agora` alto: o robô PASSA pela pose e vai embora — o modo de falha do FB para pose estática (`z` é direção de ocupação, não setpoint). Assinatura recorrente: braço dentro do manifold, perna fora.

## Por que parou
Confirmou o diagnóstico; nenhuma pose de pega foi reproduzida.

## O que herda
- O par `agora` × `melhor` como veredito de "passou pela pose" contra "não há rota".
- `torch.set_num_threads(1)` vale 3× com o viewer do MuJoCo aberto (`act` de 24 ms para 8,4 ms).
