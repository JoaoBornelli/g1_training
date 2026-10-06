---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: parada
verificado: 2026-10-06
periodo: 28–30/08/26
resumo: "Cadeia da pega consertada; ~46% da manipulação ainda termina na mesa; o run se perdeu com a sessão da Kaggle."
cluster: c3-bloco
---
- **Datas:** depois de [917bf38] (28/08 16:37) e antes de [ac93fce] (31/08 07:46), que cita seu número.
- **O que mudou:** os consertos de [917bf38].
- **Medido:** `contato_ilegal` 14,8% das terminações sobre uma fatia de manipulação de 32,1% → ~46% dos episódios de manipulação terminando na mesa; com um sensor único não havia como separar tronco, coxa ou pad [ac93fce; `g1_limpo/cena.py:125`].
- **Por que terminou:** o run se PERDEU: "Sessão interativa que expira leva `/kaggle/working` embora — foi assim que o bloco 4 se perdeu" [`g1_limpo/kaggle/g1_limpo_kaggle.ipynb`, célula de persistência; também "uma queda de internet levou 1500 iterações", sem dizer de qual bloco].
- **Mudança seguinte:** a terminação da mesa partida em três sensores, puramente medição: `contato_tronco` (pelve, tronco, quadril, coxa), `contato_palma` (`.*_palm_pad`), `contato_dorso` (`.*_hand_back_pad`); mesma união, mesmo limiar, smoke trava união e disjunção. Smoke 352 ok [ac93fce].
- **Lição:** `reduce="netforce"` dá um número por sensor; para diagnosticar contato é preciso um sensor por parte do corpo [ac93fce]. Persistir a saída com commit da Kaggle [notebook].

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
