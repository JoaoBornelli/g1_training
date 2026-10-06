---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o robo pega de pe e anda agachado com a caixa porque o postura_ereta esta zerado na coluna CARREGAR e a justificativa do zero caducou"
cluster: c3
---
Medido no `model_15200` (`carregar_15200_055.csv`): a pelve cai de **0,748** na pega
para **0,564** ao carregar, com joelho +1,718. Andar VAZIO fica em 0,726 com o mesmo
desvio de 1,3 cm — andar em pé já está no repertório.

A causa é `knobs.py:1143`: `postura_ereta` vale **0,0** na coluna CARREGAR. Ele foi
zerado por estar "saturado ali", e isso caducou — a rampa (`pelve_piso` 0,45,
`pelve_alvo` 0,75) vale 0,379 hoje, no meio, com derivada cheia.

Sobram só `precise_pos` e rastreio, e os dois são satisfeitos agachado: a caixa fica em
0,871 contra alvo 0,85, e andar custa 0,8% do `precise_pos`.

**Why:** calibração de tabela envelhece com o comportamento que a gerou.
**How to apply:** ao zerar termo por estar constante, datar e reconferir depois.
**Detalhe:** `g1_training/docs/planos/2026-09-15-limite-de-pelve-no-carregar.md`

Relacionado: g1-limpo-nao-anda-com-a-caixa, g1-limpo-tabela-por-estado
