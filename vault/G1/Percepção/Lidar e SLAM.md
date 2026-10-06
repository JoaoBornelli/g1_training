---
tipo: abordagem
pai: "[[Percepção]]"
estado: ativa
resumo: "MID-360 simulado com o padrão real da Livox e SLAM por KISS-ICP + ESKF com a IMU: σ de 1 cm com deskew; o 24999 não gira no lugar, roteiro usa curva."
verificado: 2026-10-06
cluster: percepcao
---
## Hipótese
Simular o MID-360 de forma simples, com o mesmo código no sim e no real, e estudar SLAM bloco a bloco até ter LOCALIZAÇÃO medida contra a pose verdadeira.

## Método
- **Lidar** (23/09): `estudo_lidar/lidar_parado.py` no worktree `lidar_sim`; robô parado, `mj_multiRay` a 10 Hz (35–55 ms por leitura de 20 mil raios), padrão real do `Livox-SDK/livox_laser_simulation`. Desenho do sensor no treino: chão nos pés + frente.
- **SLAM** (30/09–01/10): KISS-ICP + ESKF à mão em numpy com a IMU do MID-360 (3 níveis de ruído), ao vivo no processo do sim; mapa Open3D a 5 cm; roteiro cego de ida e volta de 4 m; AprilTags de posição desconhecida até 1,5 m (chão, mesa, caixa). Spec `specs/2026-09-30-slam-percepcao.md`, 47 testes verdes.

## Resultado
- A borda do FOV do padrão real chega a 52,2° nos lados e 48,3° na frente e atrás; 20 mil raios deixam falhas na borda.
- σ do incremento por fase: 5 cm sem deskew pela IMU, 1 cm com.
- O `model_24999` não gira no lugar: o giro de 180° parado deriva 1,2 m e bate na estante; o roteiro usa curva andando ou reta.

## Por que parou
Ativa. Retângulo REJEITADO (0 de 300 sementes livres); parede REJEITADA como marco (a câmera só vê até 2,4 m). Alternativa sem ROS: RKO-LIO.

## O que herda
- Exclua o `torso_link` do raycast (o sensor fica dentro da esfera da cabeça); ombros e mãos recebem ~3% dos raios.
- O detector de tag põe a tag +1,3% mais longe a 2,2 m, só na linha de visada.
- Venv próprio `~/.venvs/percepcao` (o de Documents tem o mjlab).

## Filhos
<!-- filhos:inicio -->
- **g1-lidar-sim-visualizacao-slam** (achado · ativa) — estudo do lidar: script linear que simula uma leitura do MID-360 no G1 parado; a borda do FOV do padrão da Livox não é círculo
- **g1-limpo-lidar-sim-desenho** (achado · ativa) — desenho do lidar simulado para o g1_limpo — requisitos e decisões; brainstorming iniciado 2026-09-22
- **g1-slam-percepcao-desenho** (achado · ativa) — SLAM do G1 no depósito simulado: objetivo = localização medida contra a pose verdadeira; KISS-ICP + Kalman com a IMU do MID-360; em brainstorming 2026-09-30
<!-- filhos:fim -->
