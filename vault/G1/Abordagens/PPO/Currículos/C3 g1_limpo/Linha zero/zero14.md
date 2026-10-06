---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 29/09/26
resumo: "unload × _trazer e pose pelo estado; nasce o enunciado do comportamento; ergue, mas a caixa deita 90°."
cluster: c3-zero
---
- **Commit:** `2b35d61`.
- **Mudanças:**
  - `_trazer` = `exp(−(d/σ_trazer)²) × _alinha(aproximacao=True)` vira fonte única. O `unload` multiplica por ele, e a `postura_ereta` herda.
  - Contas sobre o `model_1750`: pairar 15,2 → 13,5/s; erguer reto +4,9 → +6,6/s sobre pairar; gradiente no início da subida 0,08 → 0,14/s por cm.
  - O `PosturaPorElo` lê o regime parado por `limpo_twist_zerado`. Antes, o laço de rumo trocava o σ do punho de 1,0 para 0,3 e o da cintura de 0,3 para 0,1 (achados 2 e 4 de REL-AUD, corrigidos).
  - Nasce o enunciado `specs/g1-limpo-comportamento.md` (`git:2b35d61`).
- **Resultado:** a zero14 "fez erguer, mas a caixa DEITAVA 90°" (MEM:g1-limpo-pega-parada-na-mesa). A medida direta do play que sustenta a zero15 é da zero12, it ~3100 (`git:ce3d460`).
- **Auditoria de 29/09** (REL-AUD, HEAD `7a23ea9`): 19+ achados classificados de A a G. Os relevantes para as runs seguintes:
  - 1: pairar contra a cauda.
  - 3: a cadeia C pula o CARREGAR.
  - 5: a estátua com caixa colhe ~77% da renda de andar.
  - 9: o fecho do BOTAR não exige mãos livres.
  - 10: impacto sem preço.
  - 11: `squeeze` paga apertar além do necessário.
  - 15: contato com a laje grátis abaixo de 50 N.
  - 32: `precise_ori` = 0 no CARREGAR.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
