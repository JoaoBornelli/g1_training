---
tipo: achado
pai: "[[Lidar e SLAM]]"
estado: ativa
verificado: 2026-10-06
resumo: "desenho do lidar simulado para o g1_limpo — requisitos e decisões; brainstorming iniciado 2026-09-22"
cluster: percepcao
---
O lidar (MID-360 na cabeça) entra na observação do G1 físico. Dono (22/09): a política vê o chão nos pés e as superfícies à frente (1 a 3 m). Dono (02/10): a grade fica na v3 por chave, DESLIGADA na primeira v3 (com só a laje, a cena vira atalho para a tarefa); entra por append quando o suporte variar. Exigência central: a mesma grade no sim do real e no robô real.

**Why:** o sensor fica invertido (URDF `mid360_joint`: roll π, pitch 2,9°, ~1,25 m do chão). A zona cega no chão chega a ~1,0–1,3 m. O chão nos pés exige mapa com memória.
**How to apply:** DECIDIDO (dono, 06/10): para rampas e degraus, a entrada é um mapa de elevação 2,5D (2 × 2 m, 10 cm, altura e confiança), montado pelo MESMO filtro (Kalman por célula + odometria) no treino e no real; no treino, raios de um lidar simplificado na pose do MID-360; a zona cega aparece sozinha. Substitui a grade 3D na entrada da política (plano §0). O `height_scanner` da Unitree está desligado na receita.
**Detalhe:** `g1_training/docs/planos/2026-10-02-percepcao-lidar-na-v3.md`

Relacionado: g1-limpo-caminho-para-o-robo-real g1-camera-profundidade-d435i g1-v3-prateleira
