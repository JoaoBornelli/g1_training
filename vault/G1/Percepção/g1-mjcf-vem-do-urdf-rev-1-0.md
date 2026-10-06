---
tipo: achado
pai: "[[Percepção]]"
estado: ativa
verificado: 2026-10-06
resumo: "o g1.xml do mjlab é conversão exata do URDF g1_29dof_rev_1_0 da Unitree; os frames do MID-360 e da D435 foram descartados"
cluster: percepcao
---
O `mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml` (o robô do `g1_limpo`) é conversão exata do `g1_29dof_rev_1_0.urdf` do `unitree_ros` (verificado 2026-09-23). Os 29 corpos móveis batem com Δpos 0 mm, Δângulo < 0,0001°, massas iguais e massa total 33,341 kg. Os links fixos com massa (cabeça, mão de borracha) foram fundidos no pai. Os frames `mid360_link` e `d435_link` foram descartados; sobraram só os sites de IMU.

**Why:** a pose do lidar do URDF vale no sim sem correção: `torso_link` + (0,0003; 0,00003; 0,4284), rpy (π; 0,0511; 0). O URDF com Dex3 tem a mesma pose.

Relacionado: g1-limpo-lidar-sim-desenho g1-camera-profundidade-d435i
