---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "No g1_limpo, ficar parado no elo de manipulacao pagava mais que tentar (31/08, consertado); em 23/09 o forma_postural paga 0,81/s a estatua"
---
**VALIDADO em 2026-08-31.** No elo `PEGAR` o twist e zero, e os dois `track_*` pagavam
4,0/s por rastrear nada. Piso da estatua: PEGAR 8,265/s contra ANDAR 3,863/s; parado
rendia 145 contra 102 de explorar. Conserto: `rastreio_por_elo` zera os dois nos elos
que nao andam; o piso caiu para ~4,5/s.

Em 23/09 o piso mede 4,86/s contra 4,24 a 4,42 do ANDAR: o `forma_postural` (16/09)
paga 0,81/s a estatua. Margem do smoke 0,5 -> 1,0. Pagar so o GANHO sobre a pose da
abertura ficou em aberto.

**Why:** gradiente vivo nao basta. Se ficar parado rende mais que o caminho, a politica
fica parada, e ela esta certa.
**How to apply:** tarefa que nao nasce: meca o PISO do estado antes de mexer em peso ou sigma.
**Detalhe:** `g1_training/vault/G1/Abordagens/PPO/Currículos/C3 g1_limpo/g1-limpo-piso-da-estatua.md`

Relacionado: g1-limpo-portao-diluido-pela-fatia, g1-limpo-cadeia-da-pega, g1-limpo-escoro-e-o-tronco
