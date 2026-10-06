---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
resumo: "Cinco armadilhas medidas de warm-start e observação no mjlab/rsl_rl — enxerto de entrada, normalizador de canal constante, register_buffer, seed e índice de obs"
---
Cinco armadilhas medidas em 07/2026 contra as classes reais do rsl_rl. Todas falham em silêncio.

- **Alargar a entrada de uma rede treinada custa 6 tensores.** Conflitam `mlp.0.weight` (completar com zeros), `obs_normalizer._mean` (0), `._var` (1), `._std` (1), mais os dois momentos do Adam. Canal novo entra NO FIM; inserir no meio exige reordenar colunas.
- **`optimizer.load_state_dict()` não confere shape.** Ele carrega e estoura no primeiro `step()`.
- **Canal constante envenena o normalizador.** Um canal fixo em zero termina com `_std = 0`, e depois o valor 1,0 entra na rede como 100,0. Congele o normalizador do canal, ou escale na mão.
- **Índice congelado é atributo Python, nunca `register_buffer`.** Como buffer ele vira chave no `state_dict` e o load com `strict=True` falha em todo checkpoint antigo.
- **Nunca hardcode índice de observação.** O layout é posicional; remover um termo desloca todos os seguintes e a rede lê o canal errado. Derive a fatia de `group_obs_term_names`.

**How to apply:** fixe o seed entre sessões (default 42). Trocar o seed troca a população de robôs no meio do treino, e as células medidas contra a distribuição antiga desabam.

Relacionado: g1-mjlab-mutacao-em-treino, warmstart-lower-lr, g1-de-pe-e-gratis-no-mjlab
