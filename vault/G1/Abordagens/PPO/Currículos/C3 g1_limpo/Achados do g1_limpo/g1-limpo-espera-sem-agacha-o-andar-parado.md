---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "MEDIDO 01/10: o agachamento nasce nos 0,5 s após o reset, dentro do de_pe (Δ 0,67 vs 0,69 rad); custa 0,3/s de pose e adianta o fecho ~0,5 s (~7 de renda); a espera inicial é indistinguível da locomoção parada para o ator"
cluster: c3
---
MEDIDO 01/10 (plays): o agachamento nasce nos 0,5 s após o reset (joelho 0,67 → 1,34; pelve 0,79 → 0,65), dentro do `de_pe` do fecho (Δ 0,67 contra 0,69 rad). A espera publica ANDAR, twist zero e caixa zerada: o ator não a distingue da locomoção parada; o crítico sim. Custo: `pose` ×1 cai de 1,0 a 0,69 (−0,3/s; derivada −0,13/rad). Ganho: a pega ergue com as PERNAS (joelho 1,34 → 1,60 → 1,26); começar agachado adianta o fecho ~0,5 s, e fechar 1 s antes compra ~14 de renda.

**Why:** o ganho vem da pressa do desenho, não de termo que pague agachar; a pista é "parado logo após o reset".
**How to apply:** matar a pista não basta (agachar é barato); tirar o GANHO: espera inicial zero, ou porta só de pé com limiar próprio. Referência IK no `pose` REJEITADA pelo dono (01/10).

Relacionado: g1-limpo-postura-de-pe g1-limpo-janela-de-espera g1-limpo-fechar-paga-menos-que-pairar
