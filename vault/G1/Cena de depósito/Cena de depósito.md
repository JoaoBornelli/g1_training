---
tipo: frente
pai: "[[G1]]"
estado: ativa
verificado: 2026-10-06
resumo: "Cena realista do G1 no MuJoCo clássico, em 4 subprojetos: depósito, sensores (lidar, IMU, D435i com a política carregada), políticas e controlador autônomo."
---
Projeto de cena realista para testar o G1 fora do treino: um depósito no MuJoCo clássico com prateleiras, caixas e sensores.

Quatro subprojetos: (1) a cena do depósito; (2) os sensores — lidar MID-360, IMU e câmera D435i simulados com a política carregada, sem ROS 2 (API Python, arquivo depois); (3) as políticas treinadas rodando na cena; (4) o controlador autônomo (ver Autônomo fase 1, em Controle determinístico).

## Filhos
<!-- filhos:inicio -->
- **g1-cena-deposito-sim** (achado · ativa) — projeto de cena realista para o G1 no MuJoCo clássico: depósito, sensores, políticas e controlador autônomo; spec e plano da parte 1 escritos 2026-09-24
- **g1-cena-sensores-parte2** (achado · ativa) — parte 2 do projeto do depósito: lidar MID-360, IMU e câmera D435i simulados com a política carregada; FEITA 2026-09-24
<!-- filhos:fim -->
