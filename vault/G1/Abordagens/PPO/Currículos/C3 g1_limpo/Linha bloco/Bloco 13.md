---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 09–10/09/26
resumo: "Mão no alcançar: o robô apoia (load 0 → 0,44/s), mas o BOTAR não fecha porque o de_pe reprova pelo joelho em 99,975%."
---
- **Datas:** 09/09 → 10/09 (notebook `1dbfa06`, META 9500).
- **O que mudou antes:** v3.3 (`206070b`, spec `mao-no-alcancar`): `_alcancar` perde os ramos que
  devolviam 1 constante no BOTAR. Achado estrutural: no BOTAR o σ trava no piso `sigma_min = 0,08`,
  logo o kernel é `exp(−(d/0,08)²)` e nasce em 0,570, não 0,368. Medição na cauda CARREGAR da cadeia B
  (`model_6999`, 4284 amostras em 28 800 env-passos): `dist_palma_caixa` p50 0,0576 m, kernel p50
  0,595 → veredito "sobe". Com a cadeia C só 16 amostras em 38 400 env-passos (abaixo do piso de 200).
- **Medido (spec `g1-limpo-botar-fecha-e-para.md` §0):** `load` 0 → 0,441/s, `precise_pos` 0,82 →
  1,66/s, `s_C = 0,0000` em 2501 its. `time_out` 78,0% (spec `cauda-parada-de-pe` §0.3). Sonda
  `model_9499` (32 envs, CPU, 62 109 passos em BOTAR ∧ VALIDA): `alinhado` 0,9480, `perto` 0,8475,
  `apoiada` 0,7712, **`de_pe` 0,0958**. Dos 44 336 passos com três portões verdadeiros, o `de_pe` falha
  em **99,975%**, sempre pelo JOELHO (E 55,9%, D 44,1%); `amax|q − q_default|` p10 0,803, p50 1,457,
  p90 1,977 contra limiar 0,69. Os quatro fecharam juntos em 2 passos de 62 109. Hipótese do tornozelo
  REFUTADA (pior junta em 0,89% dos passos, zero reprovações). Causa geométrica: laje 0,30–0,55 m exige
  agachar; o PEGAR fecha (`s_B` 0,89) porque o alvo é o peito.
- **Play (dono):** "a mão virada" na pega (spec `cauda-parada-de-pe` §0.4).
- **Por que terminou:** fim da META (9499).
- **Lição:** portão de fecho tem de ser compatível com a geometria do elo; `de_pe` fica no PEGAR.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
