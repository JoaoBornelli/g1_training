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
**How to apply:** o `terrain_scan` do fabricante não se reproduz no real. Uma função única, sem import de simulador, faz pontos → grade no mjlab, no sim do real e no robô.
**Detalhe:** `g1_training/docs/planos/2026-10-02-percepcao-lidar-na-v3.md`

Relacionado: g1-limpo-caminho-para-o-robo-real g1-camera-profundidade-d435i g1-v3-prateleira
