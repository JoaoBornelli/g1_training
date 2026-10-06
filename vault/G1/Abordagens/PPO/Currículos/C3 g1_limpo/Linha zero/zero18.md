---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 30/09/26
resumo: "Plano de resume do 4650 (sorteio de twist, pose, contato com a laje, teto do p_C); nunca implementado, trocado pelo lote K1b + R1 + K12."
cluster: c3-zero
---
- **Plano:** resume do `model_4650` com o sorteio do twist (30% parado, 30% girando), `pose` ×4 no ANDAR e ×8 na ESP_SEM, contato com a laje de 5 a 30 N, teto 0,30 do `p_C`, postura de pé com IK e métricas (PLANO-Z18).
- **Decisões do dono:** D-A2a, D-A2b (pose do molde), D-B1, D-E (um resume só).
- **Por que não rodou:**
  - REL-PLATO §0: o zero18 "NÃO resolve o platô do BOTAR".
  - REL-MONO: os itens 1.1 e 1.3 violam a monotonia no estágio 0 do `command_vel` (V2: o giro no lugar empata com a estátua).
  - Nenhum knob dele existe no código: `balanceador_teto` e `rel_standing_envs` como knob não aparecem em `g1_limpo/` (verificado por grep em 06/10).
  - O lote de 01/10 o substituiu.
- **SPEC-TETO:** decisão D1b do dono, deixar a zero17 seguir na fase 4 antes de aplicar.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
