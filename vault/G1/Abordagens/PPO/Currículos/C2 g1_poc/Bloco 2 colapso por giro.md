---
tipo: abordagem
pai: "[[C2 g1_poc]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 21/08/26
resumo: "O robô aprendeu a girar e caía; o termo giro_indevido foi criado e apagado no mesmo dia."
cluster: c2
---
## Hipótese
Fechar o yaw (único eixo sem cerca) e consertar a entrega do navegador estabilizam a locomoção.

## Método
Diagnóstico na it 1216 do bloco 2: `fell_over` 95,2% das terminações, episódio de locomoção 0,21 s, `track_ang/track_lin` 0,0199 com pesos iguais, wz ~1,5 rad/s com comando zero. Quatro consertos, entre eles o termo `giro_indevido` e a velocidade residual de base só na manipulação (era global: todo episódio nascia empurrado, inclusive os de andar).

## Resultado
O robô aprendeu a GIRAR, e cair era consequência: ótimo local (200 passos a 0,085/passo = 7,35 contra 41 a 0,138 = 4,65).

## Por que parou
O `giro_indevido` foi apagado no mesmo dia (ver Paridade com o fabricante): ele custava ~0,06/s contra 1,49/s do `action_rate`, e punia a contra-rotação de tronco que toda passada precisa.

## O que herda
- O fabricante exclui o z do `body_angular_velocity_penalty` DE PROPÓSITO (`# Don't penalize z-angular velocity.`). Ler isso como vão foi o erro.
- O controlador de forma sorteava 91% de locomoção (correto) e o empurrão global caía 91% das vezes na habilidade que não funcionava.
- Fatia por EPISÓDIO não é fatia por PASSO: com episódios de andar de 24 passos, 30% dos episódios davam 1,1% do gradiente.
