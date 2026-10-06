---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "fração do curso não é limite em junta de curso largo (hip_yaw 316°): o hip_yaw foi a 151° no BOTAR; a rampa em 40° virou preço pago; do zero só o clip da ação impede"
cluster: c3
---
MEDIDO 2026-09-17. `hip_yaw` no BOTAR: −144° (laje 0,25) e −96° (laje 0,53), onde a IK quer ±9°.

Rampa do `limite_de_junta` a partir de 40°: paga por 900 iterações sem mover a pose.

CLIP DA AÇÃO REJEITADO (18/09): "o sinal só está sendo cortado após a decisão ser tomada".

TERMINAÇÃO `|hip_yaw| > 1,0 rad` (só no notebook de resume, fora do git, 18/09): `s_C` caiu de 0,73 a 0,02 e voltou a 0,6 em ~400 iterações. Ela segue disparando em ~1% dos fins de episódio. Falta o CSV do checkpoint final confirmar o quadril no BOTAR.

**Why:** o dono quer que a POLÍTICA aprenda a pose; um corte depois da decisão esconde o hábito e não vale no robô real.
**How to apply:** terminação impede e a política reaprende em ~400 iterações; preço só encarece. Não propor clip de ação de novo.
**Detalhe:** `g1_training/vault/G1/Abordagens/PPO/Currículos/C3 g1_limpo/g1-limpo-limite-de-junta-curso-largo.md`

Relacionado: g1-limpo-forma-postural, g1-limpo-botar-deita-o-tronco
