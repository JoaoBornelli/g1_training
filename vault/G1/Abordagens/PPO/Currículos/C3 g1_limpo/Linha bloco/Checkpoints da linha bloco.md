---
tipo: achado
pai: "[[Linha bloco]]"
estado: ativa
verificado: 2026-10-06
periodo: 26/08–02/10/26
resumo: "Tabela dos checkpoints da linha bloco (4999 a 23250): origem, o que cada um faz no play e a fonte."
cluster: c3-bloco
---
| checkpoint | origem | o que faz | fonte |
|---|---|---|---|
| `model_4999` | fim da bloco9 | sonda de velocidade de junta (manipulação parada p90 3,13 rad/s) | spec `lento-e-estavel` §0 |
| `model_5950` | bloco10 | sondas da v3.1: BOTAR alcançado 46,3%, 0/190 fecham | spec `dois-bits` §0.1 |
| `model_6999` | fim da bloco10 | ponto de retomada das blocos 11, 12 e 13; caixa no hold a 1,025 m | `c1801ea`, `34a356f`, spec `mao-no-alcancar` |
| `model_9499` | fim da bloco13 | sonda do `de_pe` (99,975% de reprovação pelo joelho) | spec `botar-fecha-e-para` §0 |
| `ckp9100` | PROVÁVEL bloco13 (`1dbfa06`), ~09/09; arquivo fora de `~/Downloads` | completa o ciclo no play em laje 0,35 e 0,55; falha em 0,15 (caixa a 90°). 1º ciclo completo da linha bloco no play | vault `g1-limpo-registro-de-checkpoints`; relatório 01/10 §5b |
| `model_10200` | bloco14 ou 15 | base da tabela por estado e de várias memórias de 10/09 | spec `tabela-por-estado` §0 |
| `model_10500` | bloco15/16 (não escrito) | play valida a cadeia C inteira no nível ~3 | memória `g1-limpo-tabela-por-estado` |
| `model_11322` | fim da bloco16 | braço D no batente (`shoulder_yaw` 2,63) | memória `g1-limpo-faixa-de-pose` |
| `model_13999` / `model_14000` | fim da bloco17, tag `estavel-bloco17` | cadeia inteira com três defeitos; `s_B` 0,62, `s_C` 0,484; base da sonda e do bloco18 | handoff 14/09; `9b6cc8f` |
| `model_15200` | bloco18/"bloco19" | anda agachado com a caixa (pelve 0,564) | plano 15/09 |
| `model_15999` | fim da bloco18/"bloco19" | ponto de retomada da bloco21 | `d117342`; vault registro |
| `model_17999` | bloco21 (PROVÁVEL `a4fbd43`), 17/09 | completa em 0,25 e 0,55 (larga a −1,49 m/s na 0,55); cai na 0,05 | vault registro |
| `model_20500` | bloco22 (PROVÁVEL `9495e32`), 17/09 | completa em 0,25 (−0,06 m/s); dono: "bota com a perna toda errada" | vault registro |
| `model_21700` | bloco22, resume do `model_20900` (QUASE CERTO `7b1b5ac`), 18/09 | completa em 0,15; dono: "todas as tarefas satisfatórias, menos o BOTAR": tronco 65°, braço a 55% do alcance, `hip_yaw` E −34° | vault registro e `g1-limpo-botar-veredito-do-dono-21700` |
| `model_23250` | `21700` refinado na v2 (`39b8b82`), 02/10 | MELHOR MODELO até 02/10 (dono); bota em 0,55, 0,35, 0,15 com 1 e 3 kg | vault registro |

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
