---
tipo: achado
pai: "[[Lidar e SLAM]]"
estado: ativa
verificado: 2026-10-06
resumo: "estudo do lidar: script linear que simula uma leitura do MID-360 no G1 parado; a borda do FOV do padrão da Livox não é círculo"
cluster: percepcao
---
Objetivo do dono (2026-09-23): simular o MID-360 de forma simples para aprender e depois estudar SLAM. Feito: `estudo_lidar/lidar_parado.py` no worktree `lidar_sim` (não commitado). Robô parado, sala com caixas e coluna mocap que o mouse arrasta, `mj_multiRay` a 10 Hz no viewer passivo (35–55 ms por leitura de 20 mil), grava `nuvem.npz` ao fechar.

**Why:** o padrão real é o `scan_mode/mid360.csv` do `Livox-SDK/livox_laser_simulation` (MIT; 800 mil disparos azimute/zênite; a coluna `Time/s` é índice). Nele a borda do FOV chega a 52,2° só nos lados do sensor e a 48,3° na frente e atrás, mesmo com 4 s. Uma leitura de 20 mil raios deixa a borda com falhas.
**How to apply:** o sensor fica dentro da esfera da cabeça: exclua o corpo `torso_link` do raycast. Ombros e mãos recebem ~3% dos raios.

Relacionado: g1-limpo-lidar-sim-desenho g1-mjcf-vem-do-urdf-rev-1-0
