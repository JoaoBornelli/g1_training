---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "Desenho da recompensa sequencial do g1_limpo: teto do presente >= piso do passado; tabela peso[estado][termo] em 10 estados; renda_congelada fica como piso; a orientação paga em toda tarefa de manipulação"
cluster: c3
---
Spec `specs/g1-limpo-tabela-por-estado.md` (2026-09-11).

**Princípio:** o teto do que o robô ainda tem de fazer ≥ o piso do que já fez. Medido ao contrário em `model_10200`: BOTAR 2:1, cauda 3:1, CARREGAR 7:1.

**Duas metades.** `renda_congelada` FICA cumulativa: é o arranque. A tabela `peso[estado][termo]` (10 estados) é o marginal: BOTAR ×2, cauda `postura_ereta`/`pose` ×8, CARREGAR rastreio ×3,5 e `precise_ori` ×4. Regra do dono (30/09): a orientação paga em toda tarefa de manipulação.

**VALIDADO** no `play` do `model_10500`: a cadeia C inteira, no nível ~3.

**Aberto:** a pose das mãos. Ombro e cotovelo saem da máscara do `pose` enquanto seguram; multiplicar o `pose` não alcança o braço.

**REJEITADOS:** catraca `max(0, M − viva)` e déficit `max(0, fecho − viva)`, que pinam o total; multiplicador por posição na cadeia, inobservável.

Relacionado: g1-limpo-renda-congelada-paga-10x-o-deficit, g1-limpo-nao-anda-com-a-caixa, g1-limpo-piso-da-estatua-travava-a-pega, g1-limpo-curriculo-de-cadeia
