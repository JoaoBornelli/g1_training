---
tipo: abordagem
pai: "[[C2 g1_poc]]"
estado: validada
verificado: 2026-10-06
periodo: 19–20/08/26
resumo: "Mede força em vez de recompensa no play; três erros de agregação em que um extremo mascarava a distribuição."
---
## Hipótese
Recompensa não distingue "não aperta o bastante" de "aperta e não sobe"; força distingue.

## Método
`python -m g1_poc.sonda --checkpoint <ckpt>`: política carregada, CPU, poucos envs. Mede F_n por palma contra `F_ref = m·g/(2μ)`, apoio contra m·g, subida, saturação do `tanh`; e cada condição do fecho como FRAÇÃO, a conjunção, e QUAL condição caiu em cada quebra. Flags `--com-push`, `--com-ruido`, `--com-jitter` devolvem o que o `play` remove.

## Resultado
Achou a prensa (bloco 1) e mostrou o `pegar` resolvido no caso nominal (`model_5100`).

## Por que parou


## O que herda
Três erros de agregação no mesmo arquivo, mesmo padrão — um extremo mascarando a distribuição: (1) mínimo global do apoio pegava o passo de reset; (2) o filtro `apoio > 0` descartava justamente os passos com a caixa NO AR; (3) "sustentação máxima" de UM env sortudo. Reporte a média comparável ao treino (`episode_success`) e a fração de envs.
