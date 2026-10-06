---
tipo: abordagem
pai: "[[C2 g1_poc]]"
estado: validada
verificado: 2026-10-06
periodo: 21/08/26
---
## Hipótese
A locomoção tem de ser EXATAMENTE o treino `velocity` do fabricante, com uma mudança só: comando zero nos primeiros segundos (diretriz do dono, 21/08).

## Método
Commit −318/+247: sai `giro_indevido`, sai o cronograma próprio do twist, sai o `action_rate` por competência (o fabricante roda −0,10 fixo); reset de yaw ±3,14. `joint_vel_hinge` vira por forma e por grupo de junta: zero na locomoção (o termo veio do `lift_cube`, um braço YAM sem marcha; 0,5 rad/s é uma ordem abaixo do balanço). `base_lin_vel` volta ao ator com ruído ±0,5 — sem histórico de obs ela é irrecuperável. `max_iterations` 3000 → 5000 para o 1º degrau do `commands_vel` disparar.

## Resultado
`razao_giro` 0,0199 → 0,373 (bloco 3). No bloco 3 o robô sobrevivia 13,5 s e andava, mas com passo de 3 mm: nada pedia passo maior.

## Por que parou


## O que herda
- Cada desvio do fabricante precisa de medida que o justifique; a maioria dos meus estava errada.
- "No-State-Estimation" da Unitree funciona COM histórico de obs; tirar o canal sem pôr o histórico tirou 50% do sinal de locomoção.
- Escrita no lugar destrói o comando: `vel_command_b[zero] = 0` deixava o buffer em zero quando o portão abria, e o `carregar` rodava parado sendo PAGO por não andar.
