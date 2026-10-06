---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "no comando PARADO (v=0) a ação MÉDIA leva o waist_pitch ao batente em 7 passos; com NoBatente todo episódio de manipulação morre ali e PEGAR nunca treina; a terminação (−4) esconde a rampa (−53,6/s)"
---
MEDIDO (`zero04`, 22/09): parado, o `waist_pitch` vai a 1,1 do curso em 7 passos; o PEGAR derruba o robô em 0,8 s, e o treino nunca passou do passo 7 em manipulação.

MECANISMO (hipótese do dono, coerente com a medida): o `NoBatente` limita o custo de ir ao batente em −4, uma vez; sem ele, ficar no batente custa 53,6/s pela rampa, e o robô fica vivo para aprender. A `zero01`, sem `NoBatente`, andou e pegou.

MEDIDO 02/10 (conta, `cena.npz`, std 1,0): terminar pelo alvo comandado no batente mata 92,7% dos passos; pelo ângulo medido, 4 de 7 plays do 23250 morrem em 1,6 s (tornozelo).

**Why:** a espera parada precede toda manipulação; episódios curtos dominam a amostra.
**How to apply:** a regra "nunca chegar ao limite" é cobrada pela rampa, não por terminação.
**Detalhe:** g1_training/docs/relatorios/2026-10-02-avaliacao-v3.md §2

Relacionado: g1-limpo-batente-e-restricao-mole, g1-limpo-janela-de-espera
