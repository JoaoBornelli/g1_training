---
tipo: achado
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o std da ação do g1_limpo é UM vetor de 29, sem estado; parou em 0,78 por 2800 iterações e caiu para 0,67 quando o ruído passou a custar renda proporcional"
---
# Desvio da ação por junta — lido nos checkpoints

Fonte: `actor_state_dict["distribution.std_param"]` dos `.pt` em `~/Downloads`. Ordem das
juntas = ordem do `g1.xml` da mjlab (pernas L/R, cintura, braços L/R). Lido em 2026-09-17.

| junta | 1600 | 15200 | 15999 | 17999 |
|---|---|---|---|---|
| L/R hip_pitch | 0,22 | 0,60 | 0,62 | 0,64 |
| L/R hip_roll | 0,60 | 0,65 | 0,71 | 0,71 |
| L/R hip_yaw | 0,54 | 0,57 | 0,60 | 0,60 |
| L/R knee | 0,32 | 0,64 | 0,67 | 0,67 |
| L/R ankle_pitch | 0,60 | 0,75 | 0,79 | 0,77 |
| L/R ankle_roll | 0,75 | 0,61 | 0,63 | 0,57 |
| waist_yaw / roll / pitch | 0,58 / 0,72 / 0,23 | 0,47 / 0,73 / 0,61 | 0,51 / 0,80 / 0,66 | 0,51 / 0,79 / 0,65 |
| L/R shoulder_pitch | 0,50 | 0,86 | 0,93 | 0,94 |
| L/R shoulder_roll | 0,59 | 0,85 | 0,89 | 0,90 |
| L/R shoulder_yaw | 0,55 | 0,66 | 0,70 | 0,71 |
| L/R elbow | 0,57 | 0,71 | 0,75 | 0,76 |
| L/R wrist_roll | 0,57 | 0,52 | 0,54 | 0,55 |
| L/R wrist_pitch | 0,55 | 1,22 | 1,28 | 1,28 |
| L/R wrist_yaw | 0,53 | 1,24 | 1,31 | 1,32 |
| **média (29)** | **0,53** | **0,74** | **0,78** | **0,78** |

Mecânica (rsl_rl `modules/distribution.py`): `std_type="scalar"` é UM vetor de 29 valores,
independente do estado, `nn.Parameter`, sem clamp útil (`std_range` 1e-6..1e6). A entropia
(`entropy_coef=0.01`, herdado do G1 velocity) empurra cada σ para cima com gradiente
`0,01/σ`. O surrogate só puxa para baixo onde o ruído CUSTA vantagem. No punho o batente
absorve o ruído (`limite_de_junta` cobre 0,85–1,00), então o custo é quase nulo e σ sobe
até 1,3. Entre 15999 e 17999 (bloco 21, com `forma_postural` e o ângulo do pad) o punho
NÃO afinou: 1,28 → 1,28.
