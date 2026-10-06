---
tipo: abordagem
pai: "[[C1 multitask]]"
estado: validada
verificado: 2026-10-06
periodo: 06/08/26
resumo: "16 specs e auditoria por agentes: orçamento igual de 4,0 e 5 defeitos (lift sem erguer, botar pagando segurar)."
cluster: c1
---
## Hipótese
16 specs (S0–S15) fecham os buracos do desenho: o currículo precisa governar cena, carga e perturbação, e o sinal de tarefa precisa ser igual entre tarefas.

## Método
- S1/S2: `env.nivel` passa a mover prateleira, peso e push (antes 24 dos 54 destravamentos mediam um eixo CONSTANTE).
- S5–S8, S14: perfil de duas fases e réguas de sucesso; S11: eixo `distancia_andar` e rumo (60 → 54 destravamentos).
- Piso de sobrevivência, twist na obs, DR de cena.
- Orçamento igual: gate vira vetor de peso por (termo × tarefa); sinal de tarefa em 4,0/passo, derivado de `track_linear + track_angular` do `unitree_g1_flat_env_cfg`.
- Auditoria com 5 agentes read-only contra as referências oficiais do mjlab (Go1, tracking, YAM) + passada adversarial; todo achado reproduzido antes do conserto.

## Resultado
Cinco defeitos reais corrigidos: (C1) o `lift` não media erguer (`progress ≡ 1`, NaN no repouso) → `lift_ao_peito`; (C2) o `botar` pagava mais para SEGURAR que para soltar (o `reaching` valia ~0,99 no spawn) → `botar` sai do gate do `reaching`; (A1) `box_at_prateleira` com σ 0,05 valia 4e-28 no spawn → escala dupla 0,30 + 0,05; (C3) o currículo travava para sempre; e o `table_contact` cobrava o PÉ encostar na prateleira.

## Por que parou


## O que herda
- Antes do bloco 1 os tetos de sinal iam de 1,0 (botar) a 5,5 (parado c/ caixa): o rsl_rl normaliza a vantagem sobre o rollout inteiro, então escala menor vira taxa de aprendizado menor.
- Fatores derivados de pesos e gates, nunca digitados.
- Kernel com σ de precisão final não dá gradiente de transporte: use duas escalas.
