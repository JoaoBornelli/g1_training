---
tipo: frente
pai: "[[G1]]"
estado: ativa
verificado: 2026-10-06
resumo: "Ver a caixa e o ambiente."
---
Ver a caixa e o ambiente.

## Filhos
<!-- filhos:inicio -->
- **Lidar e SLAM** (abordagem · ativa) — Lidar simulado com o mesmo código no sim e no real; SLAM por KISS-ICP + Kalman com a IMU do MID-360.
- **Visão da caixa** (abordagem · ativa) — RANSAC dispensa a rede.
- **Câmera D435i** (achado · ativa) — Fixa, 47,6° para baixo, vê o chão até 2,4 m.
- **g1-limpo-oraculo-da-caixa-no-real** (decisao · ativa) — o ator recebe a caixa exata (oráculo); DECISÃO 02/10: a v3 treina com estimador simulado (forma desde o começo, erro em quase tudo crescendo no refino); caixa no real por câmera RGB sem marcador + depth
<!-- filhos:fim -->
