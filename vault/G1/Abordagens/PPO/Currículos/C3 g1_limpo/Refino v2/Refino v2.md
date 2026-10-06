---
tipo: abordagem
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
periodo: 01–05/10/26
resumo: "Resume do model_21700 com o código da zero17: model_23250 é o MELHOR modelo; as continuações e os resumes do 24350 (freio em catraca, preço do aperto) não melhoraram."
---
## Configuração
Fontes: V-CKPT; REL-APR §5; SESS-01/10.

- **Partida:** `model_21700` da linha bloco (sha `8cf48038cbf2`, resume `bloco22` de 18/09, commit `7b1b5ac` QUASE CERTO, terminação `|hip_yaw| > 1,0 rad` só no notebook).
  - Veredito do dono em 18/09: "todas as tarefas satisfatórias, menos o BOTAR: consegue, inclina o tronco demais sem esticar os braços, gira a perna" (V-21700).
- **Código:** HEAD `39b8b82` da `exp/g1-limpo-v2`, config da zero17, LR 5e-4, freio por env ligado. 21700 → 21950 com 4096 envs; 21950 → 23250 com 8192.
- **Compatibilidade:** ator 114 e crítico 131 nos dois. `observacoes.py` e `cena.py` sem mudança desde `7b1b5ac`. O resume entra na fase 1, porque o checkpoint não tem `fase_cadeia`. Os 3 canais de giro mudaram de significado em `04e99ca` (REL-APR §5). O resume atravessou 37 commits (REL-APR §5b).

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.

## Filhos
<!-- filhos:inicio -->
- **Continuação 23250 a 24393** (abordagem · rejeitada · 03/10/26) — Mesma config: o BOTAR fecha 94% das C que chegam, mas o freio sobe em catraca (1,75 → 3,0) e derruba nível e s_C; o freio precisa de regra de descida.
- **Resume 24350 com freio de aperto** (abordagem · rejeitada · 05/10/26) — priority, DR de μ [0,6; 1,0] e aperto_excessivo −0,05: a força fica em ~200 N e nível e s_C caem; o 24800 anda mais com a caixa.
- **Resume 24350 com preço 3x e forma ponderada** (abordagem · rejeitada · 05/10/26) — aperto_excessivo −0,15 e forma_postural ponderada: a força não saiu de ~200 N em ~300 its; parado pelo dono, tema vai para a v3.
- **model_23250** (abordagem · validada · 01–02/10/26) — 21700 refinado: bota em 0,55, 0,35 e 0,15 m com 1 e 3 kg e se levanta; funil da C 44–65%, quedas 3–5× menores que a zero17.
- **Testes do model_24350** (achado · ativa · 05/10/26) — Sem treino: punho E gira 44° na pega, o aperto é ~240 N fixo (limite do motor, não acompanha a massa), contato pelas cápsulas do pulso e tronco.
<!-- filhos:fim -->
