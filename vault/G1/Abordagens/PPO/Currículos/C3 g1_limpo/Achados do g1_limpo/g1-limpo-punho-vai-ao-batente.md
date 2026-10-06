---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "no g1_limpo apertar o std do punho no `pose` matou o termo em vez de discipliná-lo; revertido para 1,00 em 2026-09-11"
cluster: c3
---
A v3.5 pôs os 6 punhos na média do `pose` e apertou `std_standing[".*wrist.*"]` de 1,00 para 0,30. **Não disciplinou: matou o termo.** REVERTIDO para 1,00 em 2026-09-11 (`exp/g1-limpo-v2`, commit `bcd0e84`). O punho FICA na média; a máscara do `PosturaPorElo` não muda — era o σ que estava errado.

MEDIDO (`model_10200`): `pose` na pega vale 0,0272 de um teto de 1,0, com |Δvalor| p50 de 3e−05 — derivada zero. `right_wrist_roll` p50 2,034 rad, ALÉM do batente de 1,972; os dois `wrist_yaw` a 1,621 contra 1,614.

**Mecanismo:** com std 0,30 e erro de 1,6 rad os seis somam 170,7 ao expoente (/21 = 8,1; `pose` 0,0003). Com 1,00 somam 15,4 (/21 = 0,73; `pose` 0,48). O "139" que circulava é o somatório medido, equivalente a 1,44 rad rms, não a 1,6.

**Why:** std apertado numa junta longe do default não aumenta a pressão; satura o termo e apaga o gradiente de TODAS as juntas.
**How to apply:** antes de apertar um std, calcule `err²/std²` no erro MEDIDO. Expoente acima de ~3 mata o termo.

Relacionado: g1-postura-penhasco-std-standing, g1-sigma-e-a-distancia-inicial
