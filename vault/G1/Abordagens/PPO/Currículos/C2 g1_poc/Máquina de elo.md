---
tipo: abordagem
pai: "[[C2 g1_poc]]"
estado: validada
verificado: 2026-10-06
periodo: 20/08/26
resumo: "Cadeias de elos sem reset entre eles, σ por elo e load no botar; pegar e reorientar fecharam."
---
## Hipótese
Cadeias de elos (pegar → carregar → botar, pegar → reorientar) treinam a sequência sem reset entre elos.

## Método
Cadeia sorteada pela célula do nível (reorientar no 3, carregar no 4, botar no 5); o elo avança SEM reset; o sucesso trava no fecho do último elo e o episódio continua. A prateleira se move no fecho do `pegar` (+5 m no carregar; topo novo no botar). `carregar` fecha SUSTENTADO (6 s e perto por 0,5 s). `load` (espelho do `unload`) só no `botar`. σ por elo no `reaching` e no `precise_ori` (`max(σ_min, distância inicial)`). Obs `face_normal_b` (ator 112 → 115) com cirurgia de checkpoint (`expande_checkpoint.py`).

## Resultado
Pegar e reorientar fecharam; carregar e botar dependiam da marcha, que não veio.

## Por que parou


## O que herda
- σ fixo derruba o gradiente: no `reaching` caía 1391× entre prateleira 0,55 e 0,04; no `precise_ori` 90° dava 2,0e-7. σ = distância inicial.
- Sem `load`, soltar a caixa custava −3,0/s e pagava zero.
- A cirurgia de checkpoint tem de falhar alto (zero mudanças = erro) e o auto-teste tem de exercitar a poda real.
- A máquina de elo seguiu para o C3.
