---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "braço direito no batente em TODA fase, inclusive antes da pega; o pose não corrige porque é exp(−média) de 21 juntas"
---
Medido em `model_11322` (`registra_juntas --topo 0.15`): `right_shoulder_yaw`
2,63 (curso 2,62), `right_wrist_roll` 2,11 (curso 1,97), `wrist_pitch` e
`wrist_yaw` no batente — **em toda fase, inclusive antes de tocar a caixa**.
O braço esquerdo faz o mesmo trabalho com `wrist_roll` 0,82 e `shoulder_yaw`
1,25: é ele a régua.

O `pose` não conserta porque é `exp(−média)` de 21 juntas: uma junta ruim
derruba o termo e o gradiente de todas as outras cai junto.

Entrou `faixa_de_pose`: soma por junta de `exp(relu(|q−q_def|−tol)/1,5)−1`,
tolerância por família × estado, peso −0,5. Custo medido 2,4 a 3,7/s.

`dof_pos_limits` FICA: na perna ele é mais apertado (`ankle_roll` 0,236).

**Why:** aumentar o peso do `pose` multiplicaria força sobre as vítimas, não sobre a causa.
**How to apply:** tolerância 0 desliga a família no estado; perna sai no ANDAR, CARREGAR e PEGAR.

Relacionado: g1-limpo-punho-vai-ao-batente, g1-limpo-tabela-por-estado
