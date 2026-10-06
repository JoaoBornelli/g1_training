---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o amax no velocidade_por_regime matou a movimentacao do G1 e foi REMOVIDO; o codigo roda mean nos tres regimes e quem protege o batente e o limite_de_junta"
cluster: c3
---
`torch.amax` no `velocidade_por_regime` colapsou duas runs. A `bloco19` morreu em 472
iterações (episódio de 136 passos contra 1000, `razao_marcha` 0,055). A `zero01` morreu
em 113 (episódio de 49 para 8,57 passos): morrer virou a jogada ótima.

A causa é aritmética: `vel_max_walking` é o p99 **por junta**, e com 29 juntas a chance
de pelo menos uma passar do próprio p99 é `1 − 0,99²⁹ = 25%`. Um quarto dos passos de
marcha NORMAL passou a pagar.

**AMAX REMOVIDO (verificado 2026-09-15).** O gate por regime também saiu. O código roda
`torch.mean(relu(|v|/vmax − 1)², dim=1)` nos TRÊS regimes (`recompensas.py:963`). Quem
protege o curso mecânico agora é o `limite_de_junta`, que substituiu o `dof_pos_limits`
do fabricante em 14/09.

**Why:** o limiar carrega a redução para a qual foi calibrado.
**How to apply:** ao trocar uma redução, recalibre o limiar — ou não troque.

Relacionado: g1-limpo-velocidade-de-junta-medida, g1-limpo-batente-e-restricao-mole
