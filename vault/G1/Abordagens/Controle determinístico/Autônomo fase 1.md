---
tipo: abordagem
pai: "[[Controle determinístico]]"
estado: ativa
verificado: 2026-10-06
resumo: "Controlador do depósito sem RL que chama a política como executor; missão A com oráculo; o 24350 pega em 1,8 s, carrega e pousa; comando baixo exige camada de rastreio."
cluster: controle
---
## Hipótese
Um controlador sem aprendizado executa a missão do depósito chamando a política treinada como executor de tarefa (uma tarefa por vez pelo one-hot), com um oráculo de pose no lugar da percepção.

## Método
Pacote `autonomo/` no worktree `lidar_sim`; spec `specs/2026-10-05-autonomo-fase1.md`, plano `docs/planos/2026-10-05-autonomo-do-sim-ao-real.md`. Missão A, cenário fixo (semente 98), oráculo de pose, controlador sem `import mujoco` (só sinais do robô, como no real). Contrato da cena (`cena.npz`) no `~/g1_pilota_v2`.

## Resultado
Rodado com o `model_24350`: pega em 1,8 s, carrega e pousa. Calibração medida: a política ignora vx < ~0,45 m/s e wz < ~0,6 rad/s, por isso o controlador ganhou uma camada de rastreio (curva + PI + zona morta), rampa e distância de parada; a curva vai em `cenarios/curva_*.json` por checkpoint.

## Por que parou
Ativa (fase 1 concluída em 05/10).

## O que herda
- `apoiada` por queda de torque REMOVIDO pelo dono: o aperto lateral domina o torque; agora a caixa conta como apoiada pelo ponto mais baixo assentado. O dono pensa em levar esse `apoiada` geométrico ao treino.
- O contrato `cena.npz` tem de vir do branch do checkpoint (o de 30/09 tinha `altura_carregar` 0,80 contra 0,90).
- A zona morta do comando é a mesma da pendência Giro no lugar estranho (comando baixo não é respeitado).

## Filhos
<!-- filhos:inicio -->
- **g1-autonomo-fase1** (achado · ativa) — Controlador autônomo no depósito (lidar_sim/autonomo/), fase 1 = missão A com oráculo, MuJoCo puro, base do real; 05/10 o 24350 pega, carrega e pousa; zona morta da política (vx<0,45, wz<0,6) pede rastreio+rampa; apoiada por torque REMOVIDO
<!-- filhos:fim -->
