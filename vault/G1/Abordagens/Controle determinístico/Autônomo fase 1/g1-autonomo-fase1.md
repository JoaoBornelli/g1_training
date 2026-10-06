---
tipo: achado
pai: "[[Autônomo fase 1]]"
estado: ativa
verificado: 2026-10-06
resumo: "Controlador autônomo no depósito (lidar_sim/autonomo/), fase 1 = missão A com oráculo, MuJoCo puro, base do real; 05/10 o 24350 pega, carrega e pousa; zona morta da política (vx<0,45, wz<0,6) pede rastreio+rampa; apoiada por torque REMOVIDO"
cluster: controle
---
DECISÃO DO DONO (05/10): pacote `autonomo/` (`lidar_sim`); missão A, cenário fixo (semente 98), oráculo de pose; controlador sem `mujoco`; fecho do treino, só com sinais do robô. Rodado com o `model_24350`: pega em 1,8 s, carrega e pousa. MEDIDO (calibração): a política ignora vx < ~0,45 m/s e wz < ~0,6 rad/s → camada de rastreio (curva + PI + zona morta), rampa e distância de parada. `apoiada` por queda de torque REMOVIDO (dono): o aperto lateral domina; agora é a caixa assentada pelo ponto mais baixo. O contrato `cena.npz` tem de vir do branch do checkpoint (o de 30/09 tinha `altura_carregar` 0,80 contra 0,90).

**Why:** a fase 1 é a base do controlador real.
**How to apply:** contrato em `~/g1_pilota_v2`; a curva por checkpoint em `cenarios/curva_*.json`; o dono pensa em levar o `apoiada` geométrico ao treino.
**Detalhe:** lidar_sim `specs/2026-10-05-autonomo-fase1.md`; `docs/planos/2026-10-05-autonomo-do-sim-ao-real.md`

Relacionado: g1-cena-deposito-sim g1-limpo-aperto-fixo-nao-acompanha-massa g1-track-lin-parado-colhe-27pct
