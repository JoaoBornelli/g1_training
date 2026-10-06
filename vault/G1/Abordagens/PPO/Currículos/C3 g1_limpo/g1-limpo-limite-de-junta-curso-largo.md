---
tipo: achado
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "Medido em 2026-09-17 nos CSVs do registra_juntas (20500_impratio_10_tempo_075_{botar,carregar}.csv) e na referência ik/ref_botar.npz."
---
# `limite_de_junta`: onde os 85 % do curso mordem e onde não mordem

Medido em 2026-09-17 nos CSVs do `registra_juntas` (`20500_impratio_10_tempo_075_{botar,carregar}.csv`) e na referência `ik/ref_botar.npz`. `frac = |q − centro do curso| / meio-curso`,
o mesmo do termo. `folga = 0,85 / maior frac fora do BOTAR`.

| junta | curso | rampa 85 % | uso máx fora do BOTAR | BOTAR 20500 | folga |
|---|---|---|---|---|---|
| wrist_roll D | 226° | ±96° | 0,12 | 0,49 | 7,3× |
| shoulder_roll E | 220° | ±93° | 0,20 (IK) | 0,21 | 4,3× |
| **hip_yaw E** | 316° | ±134° | 0,20 (pega) | **0,96 = 151°** | 4,3× |
| shoulder_yaw D/E | 300° | ±128° | 0,23 / 0,25 | 0,25 / 0,34 | 3,7× / 3,4× |
| waist_yaw | 300° | ±128° | 0,24 | 0,33 | 3,5× |
| shoulder_roll D | 220° | ±93° | 0,26 | 0,04 | 3,3× |
| **hip_yaw D** | 316° | ±134° | 0,32 (pega) | 0,42 = 63° | 2,7× |
| hip_roll E/D | 200° assimétrico | −15°..155° | 0,73 (marcha) | 0,32 / 0,77 | 1,2× / 1,0× |
| hip_pitch E/D | 310° | ±132° | **0,88 / 0,89 (IK)** | 0,51 / 0,69 | 1,0× |
| waist_pitch | 60° | ±25° | **0,90 (IK)** | 0,80 | 0,9× |
| ankle_pitch D | 80° | ±34° | 0,59 (pega, marcha) | **1,03** | 1,4× |

Três achados:

1. **Curso largo**: hip_yaw, shoulder_yaw, waist_yaw, shoulder_roll, wrist_roll têm curso de
   220° a 316°, e o robô usa até 0,32 dele fora do BOTAR. A rampa em 0,85 começa a 3 a 7
   vezes o uso normal. Para essas, um limiar em fração do curso mecânico não é limite.
2. **hip_roll é assimétrico** (−30°..170°): o centro do curso fica a 70° e a pose de repouso
   a 0,70 do centro. A marcha vive a 0,73. A rampa em 0,85 fica a 15° do repouso.
3. **A referência da IK está ALÉM da rampa** no hip_pitch (0,88) e no waist_pitch (0,90)
   nas lajes baixas. O `limite_de_junta` multa a pose que o `forma_postural` pede.

hip_yaw: marcha ±6°, carregar ±10°, IK ±14°, transiente da pega 20°/36°.
Candidatos de limiar: ±29° (frac 0,18, teto em ±52°), ±40° (0,25, teto ±64°), ±52° (0,33, teto ±75°).
