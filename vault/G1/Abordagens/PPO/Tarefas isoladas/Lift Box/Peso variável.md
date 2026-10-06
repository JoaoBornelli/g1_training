---
tipo: abordagem
pai: "[[Lift Box]]"
estado: validada
verificado: 2026-10-06
periodo: 17/07/26
resumo: "DR de massa corrompe a heap; peso por força externa (payload) de 0,5 a 5 kg."
---
## Hipótese
A pega fica robusta a caixas de 0,5 a 5 kg.

## Método
- Tentativa 1: DR de massa por `dr.body_mass`/`dr.pseudo_inertia` → REVERTIDA: código órfão do mjlab, corrompe a heap (`corrupted size vs prev_size` na CPU, CUDA illegal memory access na GPU).
- Tentativa 2: `apply_box_payload` com `write_external_wrench_to_sim` (primitivo testado), força −z = (m − massa)·g no CoM da caixa, peso sorteado por episódio. O peso não está na obs; o robô infere pelo torque de junta.

## Resultado
Payload validado na CPU, sem corrupção, nas duas alturas do PLR.

## Por que parou


## O que herda
Use o mecanismo que o fabricante testa. Força externa simula só o peso, não a inércia.
