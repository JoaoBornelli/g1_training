---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "medição de velocidade de junta no bloco9: a manipulação parada é mais rápida que correr, e a causa é o rastreio desligado (P4)"
---
**MEDIDO 2026-09-10 (`model_10200`, bloco15), `|qd|` p90 em rad/s:**

| regime | 08-09 | agora |
|---|---|---|
| manipulação parada (pool) | 3,13 | **0,72** |
| ↳ **só o PEGAR** | 3,13 | **2,32** |
| locomoção parada | 0,43 | 0,10 |
| andando | 2,00 | 0,95 |

O `velocidade_por_regime` MORDEU em todo regime, menos no PEGAR. O pool de manipulação parada é diluído pelo BOTAR e pela cauda, que estão quietos: isolado, o PEGAR ainda corre a 2,32. Piores juntas ali: `hip_pitch` 3,52, `knee` 3,17, `shoulder_pitch` 3,01, `wrist` 2,73.

**Why:** um agregado sobre regimes esconde o regime que não melhorou.
**How to apply:** meça `velocidade_de_junta` por regime E por elo; o pool sozinho dá falso positivo.

Medido em 2026-09-08, `model_4999` do `bloco9`, 32 envs em CPU, 600 passos.

O robô se move mais rápido parado com a caixa do que correndo.

| fase | p90 | p99 |
|---|---|---|
| manipulação parada | 3,13 | 8,19 |
| locomoção parada | 0,43 | 2,24 |
| andando | 2,00 | 4,49 |
| correndo | 2,50 | 7,40 |

A locomoção parada recebe a mesma ordem e produz 7× menos. A diferença é o `rastreio_por_elo`: ele paga 4,0/s por comando zero na locomoção e zero na manipulação. Foi o P4 da v2.1.

As pernas correm, não os braços: joelho 6,22 e tornozelo pitch 7,05 contra ombro 2,68.

CONSERTADO em 2026-09-08, commits 8332caf e c3c9d08, sem push. O rastreio volta nos elos parados com um terceiro fator, `limpo_pegou`, senão quem nunca toca a caixa colhe 4,0/s. O `velocidade_por_regime` espelha o `variable_posture` com três dicts de limite, e é PENALIDADE: a forma positiva pagaria 2,0/s a um robô parado, em todo env.

**Why:** Sem esses números o peso de qualquer freio seria chute.
**How to apply:** O limite do `standing` é 2,0 rad/s, que é o p99 da locomoção parada. Sonda: `scratchpad/wt-medicao/mede_vel_junta.py`.

Relacionado: g1-limpo-auditoria-gradientes, g1-postura-penhasco-std-standing
