---
tipo: abordagem
pai: "[[Residual PPO sobre o BFM]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 30–31/07/26
resumo: "RL emite 29 residuais + 20 coeficientes de z; parado funcionou, pegar nunca agachou; escala_c engolia o gradiente."
---
## Hipótese
A RL emite 29 residuais de junta + 20 coeficientes `c` numa base de comportamento: `z = project_16(prior[tarefa] + escala_c·(c @ B))`.

## Método
- `base_z.py`: os 41 comportamentos ocupam ~14 dimensões efetivas; uma base de 20 direções reconstrói todos com cos ≥ 0,93 (k=12 0,76; k=16 0,81; k=32 0,98).
- Ação de 49 números; clamp 0,35 rad na cadeia de equilíbrio, 2,0 no braço.
- `escala_delta` 0,15: sem ela a política aleatória saturava o clamp e jogava o braço ±115° a cada 20 ms (episódio de 30 passos contra 150).
- Throughput: 13 056 steps/s com 2048 envs, contra 22 171 da monolítica com 4096.

## Resultado
`parado` funcionou (o prior já é o comportamento). `andar` andou mas não parou no alvo (preso em `move-ego-0-0.3`; o BFM não vê o nosso twist, só o `z`). `pegar` nunca agachou.

Causa medida em 31/07: com `escala_c` 0,3 a política emitia |c| ~ 10, o que movia o `z` só 24,6°, e trocar de comportamento exige 75–90°. Gradiente que desaparece: mexer `c` de 1 movia o `z` 0,8°, abaixo do ruído. Subir para 1,0 desmontou a run (episódio 765 → 17,9 passos).

## Por que parou
Com a busca ligada, ou o ruído sacode o `z` a cada 20 ms num BFM com histórico, ou a busca acha locomoção — o robô anda tentando pegar. Os 41 vêm de uma biblioteca de LOCOMOÇÃO.

## O que herda
- As 10 sementes de um comportamento não são cópias: `move-ego-0-0` tem 60° entre elas, e a média delas deriva 0,99 m em 300 passos contra 0,19 m da semente 0.
- Tirar a terminação por queda abre o hack de SENTAR: o `fell_over` mede só a inclinação do tronco (70°), e sentado o tronco fica vertical — sucesso 0,955 com o robô no chão. Revertido.
- O BFM SABE levantar: nascido caído, 3 priors levantam 100%. Os 41 são amostra do manifold, não o manifold.
