---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "MEDIDO 30/09: o registra_juntas.py monta a observação com giro_w = 0, e o ator fica cego ao tombo da caixa; todo tombo medido em play está confundido; o 5000_c.csv é do model_5200"
cluster: c3
---
MEDIDO 30/09: o `registra_juntas.py` chama `monta_observacao` sem `giro_w` (registra_juntas.py:370 → pilota.py:211-214, default zero). No treino o comando publica o giro de correção do tombo em todo regime (comando.py:2097-2106), e a sonda mostra o ator corrigindo +45 a +56° nas palmas no CARREGAR com o giro real. O registrador também publica o elo seguinte na espera e recalcula os alvos a cada passo. Pela dinâmica, o `5000_c.csv` bate com o `model_5200`.

**Why:** plays cegos ao giro sustentaram conclusões sobre o tombo da caixa no CARREGAR.
**How to apply:** tombo de caixa só se lê em play com `giro_w`; até lá, só no log do treino.

Relacionado: g1-limpo-escala-da-acao-do-punho g1-limpo-botar-esquecido-na-fase-4
