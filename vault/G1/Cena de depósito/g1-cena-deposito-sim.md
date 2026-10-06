---
tipo: achado
pai: "[[Cena de depósito]]"
estado: ativa
verificado: 2026-10-06
resumo: "projeto de cena realista para o G1 no MuJoCo clássico: depósito, sensores, políticas e controlador autônomo; spec e plano da parte 1 escritos 2026-09-24"
---
O dono quer (2026-09-24), em ordem: (1) cena de DEPÓSITO genérico (estantes, paletes, mesas, obstáculos, caixas, AprilTags); (2) lidar MID-360 + IMU no robô da cena; (3) os checkpoints de loco-manipulação na cena, com física; (4) o controlador autônomo.

Decisões: planta por semente; caixas de 1 a 5 kg (treino: 1,0 kg); suporte = palete, mesa ou estante (treino: topo ≤ 0,55 m); tags nas caixas e diante das estruturas.

**Why:** estudar SLAM e controlador com a pose verdadeira.
**How to apply:** a cena parte do robô do treino por uma CÓPIA sem edição do `g1_limpo` (sem rebase, decisão do dono), não do `g1.xml`; o `mjlab` aplica as opções de física no modelo compilado, e o XML precisa recebê-las. O `g1-box-vision` ainda NÃO tem AprilTag.
**Detalhe:** `specs/2026-09-24-deposito-cena.md` e `docs/planos/2026-09-24-deposito-cena.md` no worktree `lidar_sim`; parte 1 FEITA (2026-09-24): `deposito/` com `confere` verde, galpão 20 × 14 m, não commitada.

Relacionado: g1-cena-sensores-parte2
