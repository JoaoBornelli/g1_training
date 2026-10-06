---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: validada
verificado: 2026-10-06
periodo: 16–18/09/26
resumo: "Limite de pelve, forma_postural contra a IK, sola e quadril; chegam ao model_21700, que faz tudo menos um BOTAR bom (tronco 65°, perna virada)."
cluster: c3-bloco
---
- **bloco20:** `limite_de_pelve` no CARREGAR (`c1a5266`, 16/09, META 18000).
- **bloco21:** `forma_postural` contra a referência de IK, retomada do 15999 (`a4fbd43`/`46dea30`) → `model_17999`.
- **bloco22:** sola e rotação do quadril (`9495e32`) e terminação `|hip_yaw| > 1,0` fora do git → `model_20500`, `model_21700`.

Veredito do dono no `model_21700` (18/09): "todas as tarefas satisfatórias, menos o BOTAR": tronco 65°, braço a 55% do alcance, `hip_yaw` E −34°. Hipótese dele: esses hábitos vêm do resume. Isso motivou o teste da linha zero. Detalhe nas notas originais g1-limpo-botar-veredito-do-dono-21700 e g1-limpo-registro-de-checkpoints, e na tabela de Checkpoints da linha bloco.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
