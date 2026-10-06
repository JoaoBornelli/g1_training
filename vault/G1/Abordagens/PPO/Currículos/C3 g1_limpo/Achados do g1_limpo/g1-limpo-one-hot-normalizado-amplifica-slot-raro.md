---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "MEDIDO 30/09: a EmpiricalNormalization do rsl_rl põe o slot raro do one-hot a ~95× (CARREGAR) e 25× (REORIENTAR); desde a zero17 o PPOPorElo fixa média 0 e desvio 1, e o load dobra a normalização antiga na 1ª camada"
cluster: c3
---
MEDIDO 30/09 no `model_3450`: o rsl_rl normaliza a observação por `(x − μ)/(σ + 0,01)`, com estatística do treino todo. O slot do one-hot que quase nunca acende tem σ perto de 0: aceso, o CARREGAR entra a 94,8, o BOTAR a 89 e o REORIENTAR a 24,9, no lugar de 1. Desde a zero17, o `PPOPorElo` fixa média 0 e desvio 1 nos três blocos de one-hot. O `load` dobra a normalização antiga na 1ª camada nos slots com σ ≥ 0,01, e a rede fica idêntica. Os slots nunca acesos voltam à escala 1.

**Why:** cada fase nova acende um slot raro. Sem a dobra, o resume de um checkpoint antigo deslocaria a 1ª camada em 1,5 no ANDAR e em 38 no REORIENTAR.
**How to apply:** no simulador novo, manter o one-hot fora da normalização desde a iteração 0.

Relacionado: g1-limpo-curriculo-de-cadeia g1-mjlab-warmstart-obs-armadilhas
