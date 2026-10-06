---
tipo: abordagem
pai: "[[Goal Reaching]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 23/07/26
resumo: "IK de corpo inteiro gera a pose de pega e o BFM executa; o BFM caía."
---
## Hipótese
Um IK de corpo inteiro (mink) gera uma pose de pega balanceada (pés plantados, CoM sobre os pés) e o BFM a executa por `goal_inference`.

## Método
`ik_grab.py`: alvo das mãos na caixa → mink resolve a pose → `_create_observation_backward` → `goal_inference` → BFM executa. O IK não controla o robô, só gera o alvo.

## Resultado
O BFM caía ao executar a pose gerada.

## Por que parou
O BFM não segura a pose profunda. Levou a duas linhas: IK como controlador direto (`ik_control.py`, no cluster Controle determinístico) e o Híbrido BFM + IK.

## O que herda
O mink como gerador de pose estática, validado em 4 mm no Mapa comandado-alcançado.
