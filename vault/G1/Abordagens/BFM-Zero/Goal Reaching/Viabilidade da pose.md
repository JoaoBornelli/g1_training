---
tipo: achado
pai: "[[Goal Reaching]]"
estado: ativa
verificado: 2026-10-06
periodo: 24/07–06/08/26
resumo: "A pose 6 tinha o CoM 5,4 cm fora do apoio; ferramentas find_pose, pose_check, pose6_reach, reach10 separam falha de controle de pose impossível."
---
## Fato
A pose 6 do Teste de goal tinha o CoM 5,4 cm FORA do polígono de apoio: era estaticamente impossível para qualquer controlador.

## Evidência
- `find_pose.py`: busca em grade uma pose com as mãos a ~10 cm do chão que passe em 5 critérios — mãos na altura, juntas no limite do XML, CoM no polígono real das 8 esferas dos pés com margem, pés planos, torque estático dentro do `ctrlrange`.
- `pose6_reach.py`: rampa até a pose 6 com torque + compensação de gravidade, sem BFM e sem IK (PD puro por posição desaba nessa profundidade).
- `reach10.py`: mãos a 10 cm por TRAJETÓRIA, com o CoM travado em cada waypoint. Mostrou que a "pose 7" (só ângulos) tinha a pelve girada ~115°, e que interpolar ângulos linearmente tomba no caminho mesmo com destino válido.
- `pose_check.py`: a viabilidade isolada, sem mink.

## Implicação
- O pé ESQUERDO tem as esferas de contato nos bodies `dummy_lf_1..4`, não em `left_ankle_roll_link`. Filtrar por "ankle_roll" mede o polígono errado.
- Aterrar a pose pelo corpo mais baixo está errado: com o braço estendido a MÃO vira o mínimo. Aterra-se pelas esferas do pé.
