---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 14/09/26
resumo: "Freio por amax em vez de mean: colapso em 472 its; o limiar carrega a redução para a qual foi calibrado."
---
- **O que mudou antes:** `7d481d8` (14/09): `velocidade_por_regime` reduz por `amax` em vez de `mean`
  (troca multiplicava o custo da manipulação 15,5× na gravação, 0,11 → 1,78/s).
- **Medido (`25c9877`; handoff 14/09):** colapso em 472 its — episódio médio 136 passos contra 1000,
  `fell_over` 37 contra `time_out` 2,1, `razao_marcha` 0,055 contra portão 0,50, `sucesso` zerado.
  Viver rendia −0,5/s e morrer custava −3,78. Causa: `vel_max_walking` é o p99 por junta;
  `1 − 0,99²⁹ = 25%` dos passos de marcha normal passam a pagar.
- **Lição:** o limiar carrega a redução para a qual foi calibrado (memória
  `g1-limpo-reducao-e-limiar-andam-juntos`). O `amax` foi depois REMOVIDO (verificado 15/09, memória).

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
