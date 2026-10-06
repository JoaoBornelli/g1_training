---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o std cai rápido no começo em toda run desta pilha (fábrica 1,0→0,61 na it 200); na g1_limpo o freio −15 aprofunda a queda até ~0,45, onde a dobradiça desliga; o LR da config não importa"
---
- A queda é mecânica do PPO: `std_param` sem piso, entropia 0,01. O std de uma junta cai quando ruído²×vantagem fica abaixo de −0,7%. LR da config e `PPOPorElo` não pesam.
- As runs do zero comprimem as PERNAS: 0,32–0,43, contra 0,51 no bloco 1 e 0,65 nos blocos. A zero01, com freio −2, também.
- MEDIDO 24/09 (`model_3150`, ANDAR): ruído de perna 0,3→0,6 custa −2,35/s. O maior termo que o bloco 1 não tinha é o freio (−0,44/s, quadrático no ruído; −14,8/s com σ 0,96 e peso −15). Freio ZERADO no ANDAR para a `zero08`; a `zero07` travou (tronco 67° na pega, escoro).

**Why:** perna sem exploração não acha a marcha.
**How to apply:** compare o std das PERNAS, não a média das 29.

Relacionado: g1-limpo-std-da-acao-por-junta, g1-limpo-freio-nao-taxa-a-marcha
