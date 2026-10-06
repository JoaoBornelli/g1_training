---
tipo: decisao
pai: "[[C4 v3]]"
estado: ativa
verificado: 2026-10-06
resumo: "DECISÃO DO DONO 05/10: na v3 o gate B2 (caixa apoiada, palmas ainda nas faces) exige força normal TOTAL robô→caixa < F_solta; a força vai ao gate, à recompensa e ao crítico, nunca ao ator, que a infere pelo torque do PD (q, q̇, última ação)"
---
DECISÃO DO DONO (05/10): o alvo B2 da v3 só avança com a caixa apoiada, as palmas ainda nas faces e a força normal total do robô sobre a caixa abaixo de `F_solta` (proposta 0,25·m·g, a calibrar), somando todos os contatos e não só os pads. O ator não recebe a força; ele a infere pela propriocepção, porque o torque do PD é função de posição, velocidade e última ação. O crítico recebe a força como canal privilegiado. Registrado em `specs/g1-v3.md` §8.2 e §6.2.

**Why:** sem o gate, o robô afasta as palmas com o pulso ainda prensando e arrasta a caixa; no real não há sensor de força.
**How to apply:** gate no B2, não no B3 (palmas a 5 cm já zeram a força; REJEITADO 05/10); "força na faixa" vale de P2 a B1.

Relacionado: g1-limpo-caixa-presa-pelo-pulso g1-v3-palma-de-plastico-5kg g1-limpo-v3-alvos-macro
