---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "O robo nao girava por dois motivos somados: o termo de guinada media o gingado nao comandado, e o envelope era lento demais; o sigma tem de acompanhar o envelope"
---
O `g1_limpo` nao girava. Duas causas, ambas medidas em 2026-09-10.

**1. O termo media o que ninguem pede.** `track_angular_velocity` do molde soma `wx² + wy²` ao erro de guinada. Roll e pitch sao o gingado da marcha. O canal valia 6,5% do maximo. `recompensas.giro_sem_gingado` tira os dois: canal foi a 38%, erro de guinada caiu 25%.

**2. O envelope era lento.** O topo do molde e ±0,7 rad/s — uma volta em 9 s. Pedido do dono: volta em 4 s. Adotado **±1,6 rad/s** (`knobs.Giro.wz_teto`).

**A REGRA:** ao alargar um envelope, o σ do kernel alarga junto. Com σ fixo em `sqrt(0.5)`, o topo de 1,6 pagaria 0,006 — kernel morto. O σ virou `max(|wz pedido| × 1,0 , 0,707)`: identico ate 0,707, esticado acima. Derivada no topo sobe 12×.

Relacionado: g1-sigma-e-a-distancia-inicial, g1-andar-como-rastreio-de-velocidade, g1-limpo-plano-tres-tarefas
