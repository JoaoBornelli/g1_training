---
tipo: decisao
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "MEDIDO 30/09: a abertura do BOTAR teleporta a laje para dentro da perna em 31–72% das aberturas; DECISÃO 01/10: fica a DR ±0,10 em xyz, com o sorteio em x só afastando a laje (toque 0,6% de pé)"
---
MEDIDO 30/09 (replay cinemático, sem dinâmica): a abertura do BOTAR põe a laje em base + R_yaw·(0,50 ± 0,10; ± 0,10), topo0 ± 0,10, quatérnion identidade, sem checar o corpo (comando.py:1799-1887). Ela nasce dentro da perna em 31–72% das aberturas, com força de kN; a cláusula de arremesso do `caixa_largada` liga em 36–48% dos toques.

DECISÃO DO DONO (01/10): fica a DR atual (laje ±0,10 em xyz, alvo ±0,05); o sorteio em x só afasta a laje. Medido: toque 39% → 0,6% de pé, 33% → 4,7% na pose da zero17. Laje girada com o rumo e "mesma mesa com altura nova" REJEITADOS.

**Why:** fere o §3 (nada toca a laje) e pune o robô por um evento que ele não vê.
**How to apply:** o toque que sobra vem da postura; o robô tem de chegar de pé ao BOTAR.
**Detalhe:** docs/relatorios/2026-09-30-plato-do-botar.md

Relacionado: g1-limpo-botar-esquecido-na-fase-4
