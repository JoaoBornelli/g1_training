---
tipo: abordagem
pai: "[[Linha zero]]"
estado: parada
verificado: 2026-10-06
periodo: 30/09/26
resumo: "Reverte a âncora; play do 3300 mostra pelve 0,717 e tronco 22° pela forma presa na laje 0,55; substituída 2 h depois."
cluster: c3-zero
---
- **Commit:** `b2909e9` (08:27).
- **Mudanças:** a âncora volta a 0,85–0,95; `altura_carregar` 0,90; `peito_b.z` 0,102; o fecho do PEGAR perde o `~apoiada`, que só agia com o piso abaixo de 0,80. A reta que zera a 90° FICA.
- **Play do `model_3300`** (30/09, SPEC-POSTURA §1), pairando com a caixa:
  - pelve 0,717 m, pelve inclinada +22°, cintura −0,33.
  - Causa: a chave da `forma_postural` passa do fim da tabela de IK, e o `clamp` devolve a linha da laje de 0,55 (pelve 0,720 m, tronco 22°). O `upright` pede o tronco a 0°.
  - IK de pé com a caixa a 0,90 m: pelve 0,78–0,79, nivelada, joelho 0,20–0,42.
  - SPEC-POSTURA fase 1 ficou "aguarda aprovação" e nunca foi implementada (MEM:g1-limpo-postura-de-pe).
  - A run do `model_3300` não está identificada em nenhuma fonte.
- **Por que terminou:** a zero17 foi commitada 2 h depois (`c0352b0`, 10:16). Nenhuma fonte registra treino próprio da zero16.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
