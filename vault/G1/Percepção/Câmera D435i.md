---
tipo: achado
pai: "[[Percepção]]"
estado: ativa
verificado: 2026-10-06
resumo: "G1 usa RealSense D435i na cabeça; FOV do RGB e do depth diferem; câmera fixa a 47,6° para baixo, sem pescoço"
cluster: percepcao
---
Fixa, 47,6° para baixo, vê o chão até 2,4 m.

O Unitree G1 usa a Intel RealSense D435i na cabeça, junto com LiDAR Livox Mid-360
(pesquisado 2026-09-03).

FOV vertical, para `fovy` no MuJoCo:
- Depth 16:9 (848×480, 1280×720): 58°, horizontal 87°
- Depth 4:3 (640×480): 62°, horizontal 75°
- RGB (1920×1080): 42°, horizontal 69°

Os dois FOV diferem. Uma câmera no MJCF não reproduz os dois streams.

Min-Z por resolução: 175 mm em 640×480, 195 mm em 848×480, 280 mm em 1280×720.
Alcance útil 0,3–3 m. Erro ±2% até 2 m. Baseline estéreo 50 mm.

Pose de montagem (verificado 2026-09-30 no `g1_29dof_rev_1_0.urdf` do `unitree_ros`): `d435_joint` FIXO no
`torso_link`, xyz (0,0576; 0,0175; 0,4299), pitch 0,8308 rad = 47,6° para baixo. `head_joint` também é fixo: o G1
não tem pescoço. Só a `waist_pitch` (±0,52 rad) inclina a câmera. No sim, de pé: câmera a 1,23 m, eixo 48–53° abaixo
do horizonte, o RGB vê o chão de 0,35 a 2,4 m. O MuJoCo Menagerie e o `unitree_mujoco` não definem `<camera>`.

**Why:** Os números vêm do datasheet Intel D400. A Unitree não publica valores
próprios. Repetir a busca custa caro.

Relacionado: g1-box-vision-redesenho unitree-g1-sim-mujoco-not-isaac
