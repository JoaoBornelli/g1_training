---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 22/09/26
resumo: "Rampa da cintura a 0,90: morreu na it 650; terminação por junta num treino do zero vira a saída barata."
---
- **O que mudou:** rampa da cintura no zero de 0,95 para **0,90** (3,0° de largura, cai na
  referência de IK, frac 0,899, que segue de graça); pico igual, `expm1(40 × 0,10) =
  expm1(80 × 0,05) = 53,6`; `NoBatente` mantida em `frac_max` 1,00; `SEG_POR_ITER` 12,7
  `[1643fb3]`.
- **Medido:** morreu na it 650 com `batente_da_cintura` dominante; todo episódio de
  manipulação acabava em ~7 passos; o PEGAR nunca treinou; o `model_650` cai na primeira
  tarefa `[9b7dfba]`. Com o comando parado e ator determinístico o `waist_pitch` vai a 1,1 do
  curso em 7 passos; o PEGAR derruba o robô em 0,8 s (memória
  `g1-limpo-rampa-estreita-nao-ensina`; notebook `g1_limpo_zero_kaggle.ipynb` em `6ab0258`).
  `model_650.pt` em `~/Downloads`, 22/09 15:22.
- **Mecanismo (hipótese do dono, na memória):** a terminação limita o custo do batente a −4,
  uma vez; ficar no batente custa 53,6/s pela rampa. Morrer sai mais barato.
- **Por que terminou:** a `NoBatente` sai do notebook do zero; a rampa fica `[9b7dfba]`.
- **Lição:** terminação por junta serve num resume com política treinada; num treino do zero
  ela vira a saída barata (avaliação de 02/10, §2; `g1-v3-decisoes` item 19).

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
