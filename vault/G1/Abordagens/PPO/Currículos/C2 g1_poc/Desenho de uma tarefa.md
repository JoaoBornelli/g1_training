---
tipo: decisao
pai: "[[C2 g1_poc]]"
estado: superada
superada_por: C3 g1_limpo
verificado: 2026-10-06
periodo: 18/08/26
resumo: "twist + caixa_alvo de 10 números com bit caixa_valida; terminar em vez de penalizar; currículo por env de 90 linhas."
---
## Decisão
Uma tarefa, dois comandos. `caixa_alvo` tem 10 números: posição, (face, direção da normal) — a caixa é cubo, quaternião daria ambiguidade de 90° — e o bit `caixa_valida`. Sem o bit, "não existe caixa" e "a caixa está no alvo" dão a mesma obs (erro zero).

## Como o alvo vira comportamento
| alvo | o robô faz |
|---|---|
| `caixa_valida = 0` | fica de pé, anda, gira |
| altura de carregar, 0,78–0,85 m em MUNDO | pega e levanta |
| peito, frame da base | carrega |
| prateleira | põe |
| mesma posição, girado | reorienta |

## Peças
- Obs: ator 112 canais (→ 115 → 118), crítico 125 (→ 128). `base_lin_vel` só no crítico (revertido depois).
- Alvo de cada palma na FACE lateral da caixa, não no centro: com o centro o `reaching` estagnava com uma mão só.
- 5 termos de tarefa multiplicados por `caixa_valida` (vetor zerado dá `exp(0) = 1`: sem o bit, não existir caixa pagaria o máximo).
- Quarto regime da postura pela DEMANDA da caixa (`10·dist + 6·ângulo`): o `std_standing` 0,05 do G1 cobrava 100 por um ombro a 0,5 rad.
- Terminações próprias em vez de multas: `table_contact` → `contato_ilegal` (pelve, tronco, coxa); `box_shake` → `precise_ori`; `com_balance` → `upright` + `fell_over`.
- Currículo por env no molde do `terrain_levels_vel`: sobe com sucesso, desce sem. O nível equilibra em p(sucesso) ≈ 0,5, e o rebaixamento é o anti-esquecimento.
- Prateleira com piso de topo em 0,04 m (a laje de 4 cm apoia no chão).

## Por que foi superada
O desenho funcionou para manipular, mas a locomoção não formou dentro dele. O C3 mantém a máquina de elo e põe a locomoção primeiro.
