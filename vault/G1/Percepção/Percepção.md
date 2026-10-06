---
tipo: frente
pai: "[[G1]]"
estado: ativa
verificado: 2026-10-06
resumo: "Ver a caixa e o ambiente: visão da caixa por RGB/depth (RANSAC dispensa a rede), câmera D435i fixa, lidar MID-360 simulado e SLAM por KISS-ICP + Kalman."
cluster: percepcao
---
Sensores e estimação do que o robô precisa saber sem oráculo: onde está a caixa, como ela está orientada, onde está o robô.

- **Visão da caixa:** de CNN em RGB (orientação do cubo, face da tag, memória das faces, continuidade por HMM) até o achado de que o RANSAC de planos no depth dá o frame da caixa sem a rede.
- **Câmera D435i:** fixa na cabeça a 47,6° para baixo; vê o chão até 2,4 m; FOV do RGB e do depth diferem.
- **Lidar e SLAM:** MID-360 simulado com o mesmo código no sim e no real; SLAM por KISS-ICP + Kalman com a IMU do MID-360.
- **Modelo:** o `g1.xml` do mjlab é conversão exata do URDF rev 1.0; os frames do MID-360 e da D435 vêm de lá.
- O oráculo da caixa no real: a v3 treina com estimador simulado; câmera sem marcador + depth.

## Filhos
<!-- filhos:inicio -->
- **Lidar e SLAM** (abordagem · ativa) — Lidar simulado com o mesmo código no sim e no real; SLAM por KISS-ICP + Kalman com a IMU do MID-360.
- **Visão da caixa** (abordagem · ativa) — RANSAC dispensa a rede.
- **Câmera D435i** (achado · ativa) — G1 usa RealSense D435i na cabeça; FOV do RGB e do depth diferem; câmera fixa a 47,6° para baixo, sem pescoço
- **g1-limpo-oraculo-da-caixa-no-real** (decisao · ativa) — o ator recebe a caixa exata (oráculo); DECISÃO 02/10: a v3 treina com estimador simulado (forma desde o começo, erro em quase tudo crescendo no refino); caixa no real por câmera RGB sem marcador + depth
- **g1-mjcf-vem-do-urdf-rev-1-0** (achado · ativa) — o g1.xml do mjlab é conversão exata do URDF g1_29dof_rev_1_0 da Unitree; os frames do MID-360 e da D435 foram descartados
<!-- filhos:fim -->
