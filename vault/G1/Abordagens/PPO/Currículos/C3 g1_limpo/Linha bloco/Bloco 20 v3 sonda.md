---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 14/09/26
resumo: "Correções M2–M6 sobre o 13999: s_B 0,62 → 0,86 mas s_C 0,484 → 0,002; reaprender por warm-start não aconteceu."
cluster: c3-bloco
---
- **O que mudou antes:** correções M2–M6 sobre o `model_13999` (`f1e4e22`, `f7ae0f6`): `upright` na
  tabela (BOTAR 4, CAUDA 1); `faixa_de_pose` na tabela (BOTAR 2); `FaixaDePose.perna[BOTAR]` 1,3 → 0;
  `apoiada` vira faixa (teto `peso + folga_apoiada_N` = 30 N); PEGAR exige `~apoiada`. Alvo pinado em
  1,02 (`PINA_ALVO = True`) para isolar as correções. Notebook `g1_limpo_sonda.ipynb` (`51cc0ac`).
- **Medido (handoff 14/09):** na it 14391, `s_B` 0,62 → **0,86**, `s_C` 0,484 → **0,002**.
- **Por que terminou:** 2000 its planejadas; resultado inconclusivo. Hipótese do dono (preferida no
  handoff): as quatro correções juntas tornaram inviável a única estratégia de pousar que a política
  conhecia (escorar), e reaprender em 2000 its de warm-start não acontece. Teste proposto e não
  rodado: `folga_apoiada_N = 1e9`. A memória da época (citada em `docs/relatorios/2026-09-30-plato-do-botar.md`)
  registra que `upright` ×4 no BOTAR matou o `s_C`.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
