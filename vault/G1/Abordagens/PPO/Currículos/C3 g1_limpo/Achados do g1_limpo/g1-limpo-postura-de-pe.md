---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "MEDIDO 30/09: a 'bunda empinada' (pelve +22°, cintura −0,33, pelve 0,72 m) vem da forma_postural presa na linha da laje de 0,55 com a caixa no peito; IK dá a pose de pé (pelve 0,79, nivelada, joelho 0,15–0,42); spec postura-de-pe"
---
MEDIDO 30/09 (play do `model_3300` + IK): com a caixa acima da laje, a chave da `forma_postural` passa do fim da tabela (h 0,68) e o `clamp` pede pelve 0,720 m e tronco 22°; o play fica a 0,717 m. O `upright` (fora da tabela por estado) põe o ótimo do tronco em ~3°, contra 24° da IK no BOTAR (medido 30/09), e nada mede a pelve: o robô dobra no quadril e a cintura compensa. A IK de pé dá pelve 0,79 m nivelada, tronco vertical, joelho 0,15 sem caixa e 0,20–0,42 com 1–5 kg.

**Why:** o enunciado §3 agora pede a pose de pé humana; a referência presa contradiz.
**How to apply:** fase 1 = linhas de pé na tabela, pelve como 7ª grandeza, forma no CARREGAR; aguarda aprovação do dono. Fase 2 (sem caixa): a opção (b), referência IK no `pose` parado, foi REJEITADA pelo dono em 01/10; ver g1-limpo-espera-sem-agacha-o-andar-parado.
**Detalhe:** specs/g1-limpo-postura-de-pe.md

Relacionado: g1-limpo-forma-postural-agacha-no-alvo g1-limpo-curriculo-de-cadeia g1-limpo-ik-referencia-botar
