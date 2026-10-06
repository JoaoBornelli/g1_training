---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "O robo aperta a caixa com 8x a forca necessaria e nenhum termo cobra por isso: squeeze satura em 1/8 da forca aplicada"
---
MEDIDO 2026-09-10, `model_10200`, cauda `CARREGAR`, n = 25 852.

Forca nas palmas: p10/p50/p90 = **49,1 / 68,9 / 92,1 N**. A referencia do `squeeze` e `F_ref = 8,45 N` (caixa de 1,378 kg, μ = 0,8). Portanto `F/F_ref = 7,29`, o `tanh` esta saturado, e a derivada e **0,00000/s por N**.

Nenhum termo do conjunto cobra aperto excessivo. `contato_palma` mede a MESA, nao a caixa (0,0000/s medido na cauda).

**Why:** um `tanh` que satura em 1/8 do ponto de operacao nao e incentivo nem freio — e uma constante. O robo paga zero por esmagar, e num robo real isso destroi a carga e a mao.

**How to apply:** ao usar `tanh(F/F_ref)`, confira o F medido contra o F_ref. Se a razao passa de ~3, o canal esta morto e o excesso nao tem preco.
**Detalhe:** `g1_training/docs/planos/2026-09-21-cuidado-com-a-caixa-impacto-e-aperto.md` — anotado como correcao futura em 21/09, a pedido do dono.

Relacionado: g1-limpo-impacto-da-caixa-sem-preco, g1-limpo-cadeia-da-pega
