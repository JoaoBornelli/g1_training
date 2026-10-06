---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: parada
verificado: 2026-10-06
periodo: 11/09/26
resumo: "Tabela por estado, dobradiça de velocidade e punho σ 1,0; braço D no batente; morreu na ~11322 sem causa registrada."
---
- **Datas:** 11/09 (notebook `dbcc299`).
- **O que mudou antes:** `limpo_estado` com 10 estados (`ec03b44`); `PesoPorEstado` substitui `VALIDA`
  nos termos e o `rastreio_por_elo` (`c64ee98`): BOTAR ×2, cauda `postura_ereta`/`pose` ×8, CARREGAR
  rastreio ×3,5; `velocidade_por_regime` vira dobradiça sem clamp `média(relu(|v|/vmax − 1)²)` com
  `vel_max_standing` por família (1,5; punho 1,0) (`954ed94`); punho σ volta a 1,00 (`bcd0e84`).
  Smoke 627 ok / 0 (`1e33711`). Primeira run no Colab, 8192 envs, META 12500, LR 5e-4.
- **Medido:** "VALIDADO no play do `model_10500`: a cadeia C inteira, no nível ~3" (memória
  `g1-limpo-tabela-por-estado`; o run de origem do 10500 não está escrito). `model_11322` (memória
  `g1-limpo-faixa-de-pose`, `registra_juntas --topo 0.15`): `right_shoulder_yaw` 2,63 (curso 2,62),
  `right_wrist_roll` 2,11 (curso 1,97), `wrist_pitch`/`wrist_yaw` no batente em toda fase; braço E com
  `wrist_roll` 0,82 e `shoulder_yaw` 1,25. Com a laje a 0,15 m agachar custa 2,03 rad de `hip_pitch`
  (`c9578ae`).
- **Por que terminou:** "a bloco16 morreu na ~11322" (`d74b93f`). Causa da morte não registrada.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
