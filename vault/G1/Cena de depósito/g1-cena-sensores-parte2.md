---
tipo: achado
pai: "[[Cena de depósito]]"
estado: ativa
verificado: 2026-10-06
resumo: "parte 2 do projeto do depósito: lidar MID-360, IMU e câmera D435i simulados com a política carregada; FEITA 2026-09-24"
cluster: cena
---
Parte 2 FEITA (2026-09-24, não commitada): `sensores/` roda um checkpoint do `g1_limpo` no depósito e entrega lidar MID-360, IMU do MID-360 e D435i (RGB + profundidade) por API Python. `confere` verde com o `model_24999.pt` (de pé 10 s). Câmeras desligadas por padrão (`rgb_ligado`, `prof_ligado`): 8 ms por passo de controle sem câmera, 55 só com profundidade, 153 com as duas (tempo real 20 ms).

Decisões do dono: sem ROS 2; arquivo depois; sem solda (a solda testada tremia 15 mm); o mais real possível, com modo ideal; raio sem retorno é leitura normal num galpão grande.

**Why:** desenvolver a locomoção com os sensores como entrada e testar a política dos motores.
**How to apply:** o `pilota.py` roda sem edição, pela ponte de índices por nome (`sensores/ponte.py`).
**Detalhe:** `specs/2026-09-24-sensores.md` e `docs/planos/2026-09-24-sensores.md` (worktree `lidar_sim`)

Relacionado: g1-cena-deposito-sim g1-lidar-sim-visualizacao-slam
