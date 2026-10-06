---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 24–28/09/26
resumo: "Freio zero no ANDAR: fechos até s_B 0,115 e depois a caixa tomba a 78° na pega."
---
- **O que mudou:** `velocidade_por_regime` vale ZERO no estado ANDAR; fora dele nada muda
  (piso −6/29 por junta e fator ** degrau) `[6ab0258]`. O notebook Colab mantém 8192 envs e
  `SEG_POR_ITER` 12,7; o Kaggle 4096 e 6,9.
- **Medido (fora da janela, 29/09):** fechos até `s_B` 0,115 na it ~1430; depois
  `caixa_na_pega` 78° e `s_B` 0,007 (`docs/relatorios/2026-09-29-tombo-da-caixa-achados-e-correcoes.md`
  §1). σ 25° sozinho REPROVADO em 28/09: `ANG` 66° na zero08, kernel 0,001 (memória
  `g1-limpo-sigma-do-alvo-congelado-degenera`). Mesmo padrão "crítico foge do fecho" na zero08
  e na zero11 (memória `g1-limpo-fechar-paga-menos-que-pairar`). Checkpoints `model_550(3).pt`
  e `model_1050(1).pt` baixados em 25/09 (atribuição à zero08 PROVÁVEL, pela data).
- **Por que terminou:** retomada como zero09 em 28/09 com `precise_ori` híbrido
  (`vault/.../g1-limpo-ang-cego-ao-tombo.md`). A zero10 nasce em 28/09 com o `ANG` pelo eixo de
  cima `[04e99ca]`. Fora da janela.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
