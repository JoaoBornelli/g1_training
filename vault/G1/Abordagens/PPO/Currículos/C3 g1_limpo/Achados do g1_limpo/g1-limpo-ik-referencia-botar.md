---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "IK de referência da postura; DONO 01/10 aprovou as tabelas da v3 (ref_v3_manipulacao.npz e ref_v3_de_pe.npz) com folga de 10 mm na laje — a ref_botar.npz antiga punha a perna dentro da laje em 12 de 15 poses"
---
APROVADAS pelo dono (01/10), vistas de frente e de lado: `g1_limpo/ik/ref_v3_manipulacao.npz` (15 poses, 5 lajes × 3 caixas) e `ref_v3_de_pe.npz` (de pé com a caixa no alvo, 1–5 kg). Regras: as de `gera_botar.py` (pelve ≤ 60°, folgas, sem rotação interna do hip_yaw, pernas da pose 3) + folga robô × laje de 10 mm + cintura ≤ 25°. A `ref_botar.npz` (16/09) deixava a laje fora da folga: perna dentro dela até 54 mm. Tronco curvado na laje baixa com braço estendido é o desejado.

**Why:** é a referência de postura da v3.
**How to apply:** o nível mais baixo põe a caixa a 4 cm do chão; ali o robô fica 9–11 cm atrás, e a caixa de 14 cm rente à borda. `ve_botar.py --foto` renderiza.
**Detalhe:** g1_training/vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md (item 15)

Relacionado: g1-limpo-botar-deita-o-tronco g1-limpo-v3-alvos-macro
