---
tipo: abordagem
pai: "[[Refino v2]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 05/10/26
resumo: "aperto_excessivo −0,15 e forma_postural ponderada: a força não saiu de ~200 N em ~300 its; parado pelo dono, tema vai para a v3."
---
- **Commits:**
  - `79b6afc`: `aperto_excessivo` −0,05 → −0,15, preço inicial com 1 kg ~−2,3/s.
  - `af97a5a`: `forma_postural` com pesos (pelve, tronco, pés, pad, sola, quadril) = (0,15; 0,35; 0,25; 0,10; 0,075; 0,075). A derivada do tronco sobe 2,1× e a dos pés 1,5×.
- **Partida:** o dono quis o 24350 para não herdar a queda de nível do 24,6k (SESS-05/10).
- **It 24669** (~320 its): `s_cauda` 0,94; `s_C` ~0,42 estável; `s_B` 0,82 → ~0,72; `caixa_largada` 0,35–0,6; força 210 → ~200 N, igual ao run de −0,05. "O peso 3× não acelerou a queda."
- **Hipótese do assistente, não medida:** a força vem da geometria da pega (cápsula do pulso e tronco, ombro no limite).
- **Fim:** o dono escreveu "parei o resume, não tava aprendendo nada. vou deixar como está e tentar fazer aprender no v3" (SESS-05/10). Registrado como REJEITADO: "−0,05 e −0,15 não tiraram a força de ~200 N em ~300 it" (MEM:g1-limpo-aperto-fixo-nao-acompanha-massa; MEM:g1-limpo-pendencias item 8).
- **Cuidado de branch:** para retomar o 23250 ou o 24350 com a física em que eles treinaram, o clone tem de usar o `39b8b82` (MEM:g1-limpo-pendencias).

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
