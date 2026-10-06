---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "MEDIDO 30/09: 28 de 28 fechos forçados do PEGAR terminam em ≤1,6 s (24 caixa_largada, 4 queda); o one-hot CARREGAR faz a política largar a caixa, mesmo com twist 0 — por isso o PEGAR nunca fecha; o currículo de cadeia (zero17) resolve"
---
MEDIDO 30/09 (CPU, `model_3450`, zero16): forçar o fecho do PEGAR com a caixa erguida a < 12 cm do alvo leva 28 de 28 envs à terminação em 0,2 a 1,6 s (24 `caixa_largada`, 4 `fell_over`). Eles falham logo que o one-hot vira CARREGAR, inclusive com twist 0. Pairar sem fechar chega ao `time_out` em 45 de 48 episódios. A política nunca treinou o CARREGAR com a caixa.

**Why:** fechar valia morrer em ~1 s, e pairar valia renda até o fim. O PPO aprendeu a não fechar (`s_B` 0,006 → 0).
**How to apply:** o currículo de cadeia (zero17) torna o fecho sobrevivível. Na fase 1 o fecho SEGURA no PEGAR, e a sonda deu 13 de 19 vivos em 4 s. O `model_21700` segue como a referência que "pega a caixa muito bem".

Relacionado: g1-limpo-curriculo-de-cadeia g1-limpo-pega-parada-na-mesa g1-limpo-fechar-paga-menos-que-pairar
