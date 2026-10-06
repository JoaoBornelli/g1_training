---
tipo: abordagem
pai: "[[Controle determinístico]]"
estado: validada
verificado: 2026-10-06
resumo: "IK numérica (DLS) + servo de posição com os ganhos oficiais do G1 e integrador implicitfast: mão no alvo sem oscilação; base dos alvos de IK do C3 e do C4."
cluster: controle
---
## Hipótese
Mover a mão do G1 até um ponto XYZ com IK numérica e PD, como a Unitree faz no robô real (ângulo-alvo + ganhos no motor).

## Método
Projeto `unitree-g1-sim` (MuJoCo), `src/sim.py`. IK por Jacobiano de diferença finita com `mj_forward` como FK, damped least squares (λ = 0,05), resolve uma vez → ângulos alvo → atuadores do braço como servos de posição com os ganhos oficiais do G1 (kp = [100,100,50,50,20,20,20], kd = [2,2,2,2,1,1,1]) e integrador `implicitfast`.

## Resultado
Oscilação 0,0 em todas as juntas, mão no alvo com ~20 mm de queda por gravidade (some com feedforward de gravidade). Euler explícito não estabiliza (o `wrist_roll` zunia ±6°). Movimento suave por interpolação em espaço de junta entre poses-chave, como o `rt/arm_sdk` da Unitree; linha reta cartesiana exige IK por quadro e é sensível.

## Por que parou
Validada em 10–11/07; serviu de base ao `ik_control.py` do BFM-Zero e aos alvos de IK do C3 e do C4.

## O que herda
- O braço chicoteava por ganhos PD absurdos (kp 1500); os ganhos oficiais resolvem.
- Travar a mão foi ganho, não método. Depurar a IK headless antes de editar o sim.

## Filhos
<!-- filhos:inicio -->
- **g1-arm-ik-pd-control** (achado · ativa) — Controle do braço do G1 no unitree-g1-sim: IK numérica + PD; travar mão foi ganho, não método
<!-- filhos:fim -->
