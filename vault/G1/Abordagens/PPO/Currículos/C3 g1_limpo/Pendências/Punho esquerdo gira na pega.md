---
tipo: pendencia
pai: "[[Pendências do C3]]"
estado: aberta
verificado: 2026-10-06
periodo: 05/10/26
resumo: "O wrist_yaw E vai a 88% do curso na pega: COMANDO no elo PEGAR (alvo +0,6 rad) mais contato da caixa."
cluster: c3
---
O wrist_yaw E vai a 88% do curso na pega: COMANDO no elo PEGAR (alvo +0,6 rad) mais contato da caixa.

Testes do model_24350 e 24800; ver g1-limpo-punho-esquerdo-empurrado-pela-caixa.

**Medido:** no `model_24350`, o `wrist_yaw` E vai de +0,04 a +1,46 rad (88% do curso) com alvo do PD em +0,30 a +0,40; a mão E gira 44° e desliza 7,2 cm sobre a face em 1,5 s (a D, 11° e 1,2 cm). No `24800` a ação sobe a +8,5 (alvo +0,60 rad) e a junta vai a +1,18: os 0,58 rad além do alvo pedem ~9,7 N·m contra ±5 do motor, então é contato. No CARREGAR a ação cai a −1 e a mão volta.

**Causa:** COMANDO no elo PEGAR mais CONTATO da caixa; a mão gira para cima.

**Próximo passo:** na v3 a mão fica fixa em relação à caixa nas tabelas `theta-30`; medir o `wrist_yaw` E na pega.
