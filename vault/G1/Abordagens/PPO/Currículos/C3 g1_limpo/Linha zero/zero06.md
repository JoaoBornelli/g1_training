---
tipo: abordagem
pai: "[[Linha zero]]"
estado: parada
verificado: 2026-10-06
periodo: 23/09/26
resumo: "Freio −6 só no zero: a marcha avança, mas na pega a pressa volta a compensar; nenhum peso fixo serve."
---
- **O que mudou:** `velocidade_por_regime` a −6 nos dois notebooks do zero; o knob segue em
  −15 e as retomadas da linha bloco também `[65072eb] [8e405db]`. "O −2 da zero01 deixava o
  robô se mexer rápido demais" `[65072eb]`.
- **Medido:** "a marcha avança bem; na pega, pela conta de 21/09, a pressa volta a compensar
  (empate em ~−12)" `[plano 09-23 §1]`. O `nivel` da zero06 estava em 0 em todos os envs
  `[plano 09-23 §5]`.
- **Por que terminou:** nenhum peso fixo serve; o plano do freio em currículo propunha
  aplicar a mudança NA zero06 (no degrau 0 a recompensa é idêntica), mas o dono decidiu que a
  zero07 começa do zero `[plano 09-23 §10]`.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
