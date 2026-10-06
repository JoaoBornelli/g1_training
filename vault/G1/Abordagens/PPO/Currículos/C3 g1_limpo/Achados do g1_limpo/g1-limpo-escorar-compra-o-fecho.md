---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o robô se apoia na caixa porque isso dispara o `apoiada` e abre a CAUDA, que paga 58/s contra 35/s do BOTAR"
cluster: c3
---
`apoiada = forca >= 0,5 × m·g` (`comando.py:1303`) tem limiar de **4,9 N** com a
caixa de 1 kg. O robô pesa ~343 N: apoiar 1,5% do peso do corpo já dispara.

Esse fecho é a porta da CAUDA. Teto ao vivo por estado:

- BOTAR ~35/s (13,8 congelado + 15,8 dos sete ×2 + 4 rastreio)
- CAUDA ~58/s (29,6 congelado + 16 `postura_ereta`×8 + 8 `pose`×8 + 4)

**Fechar 1 s antes vale ~23.** Escorar custa ~1/s de `upright`.

`descarga` tem `clamp(0,1)`: acima de `f = m·g` a derivada é zero, e empurrar para
baixo fica invisível ao `unload` e ao `load`.

**Why:** a multa de postura nunca ia ganhar de um fecho anuitizado.
**How to apply:** o conserto é `apoiada` virar FAIXA (`<= peso + folga_N`, newton
absoluto), não multa. Escorar deixa de fechar.
**Detalhe:** `g1_training/docs/planos/2026-09-14-correcoes-para-o-treino-do-zero.md`

Relacionado: g1-limpo-renda-congelada-paga-10x-o-deficit, g1-limpo-tabela-por-estado
