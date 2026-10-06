---
tipo: achado
pai: "[[C2 g1_poc]]"
estado: ativa
verificado: 2026-10-06
periodo: 20/08/26
resumo: "O comando resampleava um passo antes do time_out e zerava o sucesso; explica a diferença treino × sonda."
---
## Fato
Com `resampling_time_range = (20, 20)`, igual ao episódio, o comando resampleava no passo 999 e o `time_out` só no 1000: o sucesso era zerado UM PASSO antes do fim.

## Evidência
Log da it 5306: `pegou` 0,97 com `sucesso` 0,00; `nivel_medio` caiu de 0,0172 para 0,0085. O "sucesso 0,006 no treino contra 0,75 na sonda" do bloco 2, atribuído a push e ruído, era em grande parte esse wipe. O metric era ALIAS do buffer, não snapshot.

## Implicação
`resampling_time_range` = 10× o episódio (quem resampleia é o reset); metric vira clone; o smoke trava a regressão.
