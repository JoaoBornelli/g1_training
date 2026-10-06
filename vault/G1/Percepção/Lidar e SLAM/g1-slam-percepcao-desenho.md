---
tipo: achado
pai: "[[Lidar e SLAM]]"
estado: ativa
verificado: 2026-10-06
resumo: "SLAM do G1 no depósito simulado: objetivo = localização medida contra a pose verdadeira; KISS-ICP + Kalman com a IMU do MID-360; em brainstorming 2026-09-30"
---
Decisões do dono (2026-09-30): objetivo é LOCALIZAÇÃO, em blocos de estudo, medida contra a pose verdadeira. KISS-ICP (pip) + ESKF à mão em numpy com a IMU do MID-360; IMU em 3 níveis de ruído. Roda AO VIVO no processo do sim, sem gravar a volta (disco: só trajetórias TUM); as 5 fases rodam juntas na mesma volta. Mapa Open3D (.ply, 5 cm): percepção + referência; janela por flag. Comando cego, roteiro de IDA E VOLTA de 4 m (retângulo REJEITADO 2026-10-01: 0 de 300 sementes livres). AprilTag de posição desconhecida, só até 1,5 m (serve ao posicionamento na estação de pegar/botar; o detector põe a tag +1,3% mais longe a 2,2 m, só na linha de visada), ID = classe: CHÃO marco fixo, MESA marco da sessão (placa baixa), CAIXA só objeto. PAREDE REJEITADA: câmera fixa a 47,6° vê só até 2,4 m. RKO-LIO é a alternativa sem ROS.

**Why:** preparar a percepção que vai para o robô real, entendendo cada bloco.
**How to apply:** spec APROVADA 2026-10-01; implementado por agentes Sonnet, 47 testes verdes; σ do incremento POR FASE (5 cm sem deskew IMU, 1 cm com); semente 48; o `model_24999` NÃO gira no lugar (dono, 2026-10-01): o giro de 180° parado deriva 1,2 m e bate na estante aos 21,4 s; roteiro tem de usar curva andando ou só reta. Venv próprio `~/.venvs/percepcao` (o de Documents tem o mjlab).
**Detalhe:** `specs/2026-09-30-slam-percepcao.md` e `docs/planos/2026-10-01-slam-percepcao.md` (worktree `lidar_sim`)

Relacionado: g1-cena-sensores-parte2 g1-camera-profundidade-d435i
