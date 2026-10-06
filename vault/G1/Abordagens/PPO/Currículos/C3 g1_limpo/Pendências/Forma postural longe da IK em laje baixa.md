---
tipo: pendencia
pai: "[[Pendências do C3]]"
estado: aberta
verificado: 2026-10-06
periodo: 05/10/26
resumo: "Na laje 0,15 com 5 kg: tronco 109° contra 66° da IK, pés 0,10 m mais estreitos; a cabeça bate na caixa."
cluster: c3
---
Na laje 0,15 com 5 kg: tronco 109° contra 66° da IK, pés 0,10 m mais estreitos; a cabeça bate na caixa.

forma_postural ponderada (af97a5a) não resolveu em ~300 its.

**Medido (24800, laje 0,15, 5 kg, PEGAR em t = 1,92 s):** pelve 0,543 m (IK 0,549), pelve inclinada 76° (IK 39°), tronco 109° (IK 66°), cabeça a 7 mm da caixa. Na janela da pega, pés 0,10–0,12 m mais estreitos que a IK e tronco 20–28° fora; sola e quadril já seguem a IK (r ≥ 0,91).

**Tentado:** `forma_postural` ponderada (pelve 0,15, tronco 0,35, pés 0,25, pad 0,10, sola 0,075, quadril 0,075; derivada do tronco 2,1×) no resume do 24350: sem efeito em ~300 its.

**Próximo passo:** na v3 a pose vem dos alvos macro de IK com potencial de progresso, não de um termo de forma somado.
