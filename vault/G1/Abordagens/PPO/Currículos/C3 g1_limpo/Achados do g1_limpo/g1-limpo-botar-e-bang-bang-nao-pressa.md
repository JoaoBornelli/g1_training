---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o BOTAR do model_13999 e bang-bang; o alvo e alcancavel ate a laje no chao, e agachar troca inclinacao de tronco por 0,05 de margem"
---
Medido em `carrega085.csv` (`model_13999`): BOTAR tem p50 de 0,02 rad/s e pico de
11,58 — o pico não passa o do ANDAR (10,40). Difere o ciclo: 84% dos passos
congelados, 6 juntas além do limite rígido.

**ALCANCE REFUTADO (2026-09-14).** IK com pés planos, CoM do conjunto com caixa de
5 kg, autocolisão a 4 cm, duas mãos, largura ≤ 0,537 m, pelve ≤ 60° e joelho ≥ 0,05
alcança com erro zero: margem de **0,35** na laje a 0,55 e de **0,25** na laje a 0,05.

**Dois ramos na laje baixa.** Curvar (tronco 82°, pelve 60°, pelve_z 0,58) e agachar
(tronco 30°, pelve 6°, pelve_z 0,30) alcançam o mesmo ponto; agachar custa 0,05 de
margem. O `fell_over` de 70° mede a PELVE, não o tronco (`terminations.py:31`).

**Why:** a pose extrema é escolha, não falta de alcance.
**How to apply:** referência de pose no BOTAR é viável; termo de velocidade não.

Relacionado: g1-limpo-velocidade-de-junta-medida, g1-limpo-alvo-da-pega-era-circular
