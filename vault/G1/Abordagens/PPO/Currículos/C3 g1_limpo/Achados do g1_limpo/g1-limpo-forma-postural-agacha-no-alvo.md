---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: rejeitada
verificado: 2026-10-06
resumo: "REFUTADO como causa (30/09): o clamp da forma_postural em h 0,68 puxa só 0,12/s para o agachamento; pose+postura_ereta puxam ~0,7/s para ficar de pé"
---
MEDIDO 29/09 (`model_3450`, zero15, `juntas.csv`): no PEGAR a caixa chega a 4,5 cm do alvo e sai do apoio, mas o fecho nunca passa. Reprovam dois portões: tombo 37° a 47° (limite 25°) e `de_pe`, com `hip_pitch` a 0,81–0,85 rad do default (limite 0,69). A `forma_postural` (peso 3) tem chave na altura da caixa, e a tabela `ik/ref_botar.npz` acaba em h = 0,68 (pelve 0,72, tronco 22°, pés 0,40 m). Acima disso o clamp segue pagando o agachamento da pega, mas REFUTADO como causa (30/09): na pose medida ele puxa só 0,12/s para o agachamento, contra ~0,7/s de `pose` + `postura_ereta` para ficar de pé. O tombo vem da mão, e não de escorregar: só o `wrist_pitch` esquerdo deriva de −0,71 a −1,11 rad.

**Why:** hipótese restante, não medida: esticar as pernas sem mexer os braços sobe a caixa 10 cm e a tira da tolerância.
**How to apply:** julgar o play com cena reexportada; a de `~/g1_pilota` (18/09) pede alvo 0,90 e `peito_b.z` 0,102.

Relacionado: g1-limpo-pega-parada-na-mesa g1-limpo-forma-postural
