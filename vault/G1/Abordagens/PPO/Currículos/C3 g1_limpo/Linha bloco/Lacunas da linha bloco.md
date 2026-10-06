---
tipo: pendencia
pai: "[[Linha bloco]]"
estado: aberta
verificado: 2026-10-06
periodo: 06/10/26
resumo: "O que as fontes não confirmam: bloco 5, limites dos blocos 2, 6, 7, 10, 15, 16, nome bloco18 × bloco19, origem do ckp9100."
---
### Blocos 1–9
1. **Bloco 5 não tem registro** em nenhuma fonte lida (git, specs, docs, vault, memória, comentários de código). Não há transcrição de sessão do Claude nesta janela em `~/.claude/projects` (a mais antiga é de 11/09). Pode ser a sessão perdida do bloco 4, a "queda de internet que levou 1500 iterações" do notebook, ou um run sem diagnóstico escrito.
2. **Pontos de partida e limites dos runs 2, 4, 6 e 7 não estão confirmados.** O bloco 2 como resume do fim do bloco 1 é inferência. Não se sabe se o bloco 6 foi do zero, nem se o "resultado 577 iterações depois do resume" (forma 0,7277 → 0,1773) é a continuação do bloco 6 ou o começo do bloco 7. Não se sabe de onde partiu o bloco 7: as its 1600–2514 dele são compatíveis com um resume do bloco 6 (~785 + 577 ≈ 1362). O relatório de 01/10 diz "1 recomeço do zero (bloco1); depois 21 blocos em resume encadeado", o que contradiz o bloco 3 "do ZERO" e o bloco 8 "treino do zero" das fontes desta janela.
3. **A comparação "sem patch × com patch" do bloco 7** (`fell_over` 51,9% → 0,9%, marcha 0,484 → 0,762) diz "a MESMA iteração", mas não diz se compara dois runs ou dois trechos do mesmo run. Também não se sabe se a janela de espera de 02/09 entrou no bloco 7.
4. **O bloco 8 rodou em duas plataformas.** Ele começou no Colab (8192 envs, 8,54 s/iter) e seguiu na Kaggle a partir da it 1050 (4096 envs, 4,8 s/iter). Os níveis de iteração podem misturar as duas escalas. O checkpoint do warm-start do bloco 9 aparece como `model_1800` (memória) e como `model_2562` (proposta); confirmei só o primeiro como o usado, e só pela memória.
5. **A data exata de fim** dos blocos 1, 3, 4 e 6 vem dos timestamps dos commits que citam seus números, não de logs. Nenhum log tfevents foi lido.
6. **Divergência documental sobre o currículo no checkpoint:** a spec §8.1 diz que nada vai para o checkpoint; o plano e o commit 237881b o persistem. A memória `g1-limpo-rampa-queima-janelas` propõe um conserto da rampa na banda morta que não aparece em nenhum commit desta janela. A troca do portão para `eficiencia_min` [7607d8b] foi a resposta lateral.

### Blocos 9–22
1. **Nome e limites dos runs depois da bloco17.** O código do `9b6cc8f` foi publicado como `RUN = bloco18`
   (META 16000), mas o plano de 15/09 chama o `model_15200` de "do bloco19" e o `7ede785` diz que "o
   bloco 19 fecha em 16000"; o nome `bloco19` também é o da run colapsada da v3 (14/09). Não achei qual
   run produziu os modelos 14000–16000 nem onde o bloco18 acaba.
2. **Iterações de início e fim das blocos 10, 15 e 16, e o run de origem dos `model_10200` e `model_10500`.**
   O git não registra as fronteiras; a memória atribui o 10200 à bloco15, mas a bloco14 parou na ~10300.
   Não há métricas de fim da bloco15 nem a causa da "morte" da bloco16 na ~11322. O código da bloco10
   (v2.1 sem mudanças) é inferido, não confirmado.
3. **Métricas de log das blocos 16, 17 e 18.** Não há `s_B`/`s_C`/`sucesso` por iteração na janela,
   além da linha de base 0,62/0,484 do `model_13999`. O "`s_C` 0 → 0,34" da memória `plano-tres-tarefas`
   não tem fonte nem iteração. O `ckp9100` é "PROVÁVEL bloco13", e outra nota do vault cita um
   `model_9100` da `zero02` (21/09); pode haver confusão entre os dois arquivos.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
