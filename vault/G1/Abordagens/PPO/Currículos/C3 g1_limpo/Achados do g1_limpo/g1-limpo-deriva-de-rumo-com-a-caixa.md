---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "deriva de rumo sem comando; 17/09 laço de rumo aplicado; MEDIDO 02/10: com k = 0,5 a marcha reta assenta 6–13° fora do rumo (erro = viés/k); kernel de rumo efetivo 81°; proposta k = 2"
---
MEDIDO 17/09 (19300): marcha reta +4,4 °/s; a política (MLP sem memória) não observa o rumo.

MEDIDO 02/10 (plays 21850–23250, marcha reta, laço k = 0,5): o rumo assenta até 13° à direita (mediana ~6°). Erro de equilíbrio = b/k, viés b ≈ −0,06 rad/s; kernel de rumo efetivo σ/k = 81°. TESTE 02/10 com k = 2 (23250): rumo −2,1° (dp 0,3°) contra −6,3° com k = 0,5; desvio da linha −0,20 m contra −0,40 m.

**Why:** "não girar" é uma integral, o rumo; só um laço de rumo a fecha (`heading_command`).
**How to apply:** desde 17/09: laço em 80% dos envs (`rel_heading_envs` 0,8), wz = 0,5 × erro nos parados. Desde 02/10 o `registra_juntas --rumo-k` tem padrão 2,0; o treino segue em 0,5 até a v3. Preço somado e σ menor REJEITADOS.
**Detalhe:** g1_training/vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md (propostas, deriva de rumo)

Relacionado: g1-limpo-giro-canal-e-envelope, g1-limpo-forma-postural, g1-limpo-caminho-para-o-robo-real
