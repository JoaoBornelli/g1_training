---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 27–28/09/26
resumo: "Retomada da zero08 com precise_ori híbrido: tombo real p50 93° contra ANG publicado 62°; a caixa gira com os antebraços."
cluster: c3-zero
---
- **zero08** (RUN aberta em `git:6ab0258`, 24/09): o freio `velocidade_por_regime` passou a valer zero no estado ANDAR, porque todas as runs do zero comprimiam as pernas (0,32–0,43 contra 0,51 do bloco 1) e a zero07 travou com a pelve a 59° na pega.
- **Medidas:** fechos até `s_B` 0,115 (it ~1430); depois `caixa_na_pega` 78° e `s_B` 0,007 (REL-TOMBO §1). No `model_1750` a caixa chegava a 66° na pega, kernel 0,001, derivada 0,011/rad, sem fecho (`git:ff5ac85`). σ de 25° sozinho reprovado (MEM:g1-limpo-sigma-do-alvo-congelado-degenera).
- **zero09** (retomada da zero08, `precise_ori` já híbrido): no `model_2000`, tombo real p50 93°, `ANG` publicado 62°, guinada da face lateral 46°, razão `precise_ori/staged` 0,057. PEGAR forçado (59 643 amostras): tombo no mundo p10/p50/p90 68°/81°/94°. O giro da caixa dentro da mão foi só 4°/12°/37°: a caixa gira com os antebraços (V-ANG; `git:04e99ca`).

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
