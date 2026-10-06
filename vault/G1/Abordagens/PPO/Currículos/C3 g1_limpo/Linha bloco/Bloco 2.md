---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 27–28/08/26
resumo: "Multa −1,5 por tocar a mesa: o contato caiu e a manipulação caiu junto; multa que o robô pode pagar ele orça."
cluster: c3-bloco
---
- **Datas:** 27/08 → 28/08 manhã (o commit seguinte, 08:39 de 28/08, já cita o resultado) [6c6a5f8].
- **O que mudou antes:** `contato_prateleira = −1,5`, booleano, sobre o sensor `corpo_prateleira` (6 geoms: pelve, tronco, quadril, coxa) [f4ccb6c; 6c6a5f8]; portão por `eficiencia_min` [7607d8b]. Ponto de partida: resume do fim do bloco 1, inferido de "Retomar do `model_5000.pt` (fim do bloco 1), não do checkpoint do bloco 2" [vault:g1-limpo-mesa-contato-zero] (ver Lacunas).
- **Hipótese:** a multa leve tira o tronco da mesa sem matar a manipulação; peso e forma "do módulo que os treinou" [f4ccb6c].
- **Medido:** 405 iterações. Contato do tronco caiu monotonicamente 7,5% → 3,8% → 2,0% dos passos E a manipulação caiu junto: `staged` 0,36 → 0,17 [6c6a5f8; vault:g1-limpo-mesa-contato-zero]. O dono parou ao ver o movimento no play [vault:g1-limpo-mesa-contato-zero].
- **Por que terminou:** "multa que o robô pode pagar é multa que ele orça" [6c6a5f8].
- **Mudança seguinte:** tocar a mesa vira TERMINAÇÃO `contato_ilegal`, corpo inteiro (`.*_collision`, 33 geoms contra 6), limiar de força 50 N medido no g1_poc, sem penalidade em cima (o `−200` já cobra −4,0 mais o retorno perdido) [6c6a5f8]. Risco latente declarado: nos níveis 4–6 a laje a 0,04 m é degrau na frente dos pés; não implementado porque os blocos 1 e 2 nunca saíram do nível 0 (topo ≥ 0,55 m) [6c6a5f8]. Smoke 327 ok.
- **Lição:** (na época) terminar em vez de penalizar. Esta lição foi REVISADA em 02/09 (bloco 7): termina só o que não pode ser pago [8c2c929].

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
