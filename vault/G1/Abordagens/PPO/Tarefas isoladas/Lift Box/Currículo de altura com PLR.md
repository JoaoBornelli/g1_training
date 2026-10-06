---
tipo: abordagem
pai: "[[Lift Box]]"
estado: validada
verificado: 2026-10-06
periodo: 16–17/07/26
resumo: "Prateleira mocap e PLR de altura 0,55 → 0,25; nível inalcançável colapsa o currículo."
cluster: tarefas-isoladas
---
## Hipótese
Baixar a prateleira por degraus, mantendo as alturas já dominadas no sorteio, leva a pega até perto do chão.

## Método
- Mesa free-body vira PRATELEIRA fina MOCAP: altura por env em runtime (`shelf_top`), sem recompilar.
- `target_z` fica ABSOLUTO em 0,78–0,85 (decisão do dono): prateleira mais baixa = erguer mais.
- PLR (Prioritized Level Replay) em `common/curriculums.py`: EMA de dificuldade por nível a partir do `sustain_precise`, sorteio por rank `P = ρ/N + (1−ρ)/rank^(1/β)`; `rest_z` por env.
- `plr_height1`: [0,55; 0,45], warm-start `lift_6550`; `table_contact` −0,8 → −1,5, `arm_vel` −0,002. `plr_h2_payload`: [0,55; 0,45; 0,35; 0,25].

## Resultado
Domina 0,55 e 0,45 (`model_7300`) e segue para 0,35 e 0,25. `posture` 0,8 testado e descartado: brigava com o agachar.

## Por que parou


## O que herda
Nunca acrescente um nível que o robô não alcança: o PLR foca no mais difícil, um nível impossível recebe ~70% da amostragem, e o colapso atropela até as alturas dominadas. Um degrau alcançável por vez.
