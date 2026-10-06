---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "altura_carregar 1,02 vinha de medir onde a caixa JÁ estava; o piso real do alvo é 0,80 e vem do tol_pos, não da anatomia"
---
`altura_carregar = 1,02` foi derivado de uma sonda no `model_6999` — `caixa_b.z`
no HOLD, p50 1,025. Medir onde a caixa repousa e mover o alvo para lá é circular:
o robô segurava na altura do ombro (1,074) e o knob subiu atrás dele.

Baixado para **0,85** em 14/09. Custo de braço medido por cinemática direta,
Σ|q−default| de ombro e cotovelo: 1,30 a 1,02; **0,80 a 0,85**; 0,40 a 0,70.

⚠ O piso é **0,80**, e vem do FECHO: a laje sobe a 0,57 e a caixa maior em cima
tem centro em 0,70; com `tol_pos = 0,10` um alvo em 0,70 é satisfeito com a caixa
AINDA NA LAJE. O alvo e o `_perto` falam de CENTRO da caixa, os dois.

**Why:** descer abaixo de 0,80 exige `& ~apoiada` no fecho do PEGAR.
**How to apply:** `peito_b.z` deriva do alvo (`alvo − 0,798`), nunca o contrário.

Relacionado: g1-limpo-faixa-de-pose, g1-limpo-escorar-compra-o-fecho
