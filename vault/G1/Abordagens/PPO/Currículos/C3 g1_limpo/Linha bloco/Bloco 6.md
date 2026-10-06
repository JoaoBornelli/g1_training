---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: validada
verificado: 2026-10-06
periodo: 30–31/08/26
resumo: "Partição da mesa (94,7% é o tronco), piso da estátua (parado rendia 145 contra 102) e portão diluído; os consertos destravaram a pega."
---
- **Datas:** medido na it 493 e na it 785 antes de [ac93fce] (31/08 07:46, partição já ativa no run) e de [369af74]/[0b7d925] (31/08 10:30 e 11:28) [mem: g1-limpo-escoro-e-o-tronco; 369af74; 0b7d925].
- **Escala:** 8192 envs; it 493 = 988 equivalentes de 4096; it 785 = 1571 equivalentes [mem: g1-limpo-escoro-e-o-tronco; 369af74].
- **O que mudou antes:** a partição da terminação em três (no run, embora o commit seja de 31/08 cedo) [mem: g1-limpo-escoro-e-o-tronco]. Do zero ou resume: não registrado.
- **Medido:**
  - it 493: `fell_over` 1% das terminações; dos contatos com a mesa, `contato_tronco` 94,7%, `contato_palma` 5,3%, `contato_dorso` 0,0% exato. O tronco na mesa é ~50% dos episódios de manipulação. Pôr o pad da palma na lista não estreitou a aproximação; o freio geométrico do dorso funciona [mem: g1-limpo-escoro-e-o-tronco].
  - Piso da estátua medido no smoke (robô travado): ANDAR 3,863/s, PEGAR 8,265/s (2,1×); no PEGAR os dois `track_*` pagavam 4,0/s por rastrear comando NULO [369af74].
  - it 785: episódio de 17,6 s, 60% de morte na mesa. Conta: parado 8,265 × 17,6 = 145; explorar 0,6 × (8,265 × 8,8) + 0,4 × 145 = 102. Ficar parado ganhava por 43% [369af74]. O play confirmou: na ação MÉDIA o robô fica imóvel na pose default e não tenta pegar [369af74; `env_cfg.py:334`].
  - Portão diluído: `eficiencia_min` é zero exato nos envs com twist zero; ponto fixo `efic × (1 − fatia) = limiar` → fatia ≤ 1 − 0,50/0,80 = 0,375; `forma` não descia de 0,63. Medido na it 785: eficiência de quem anda ~0,80, fatia 0,272, média diluída prevista 0,582 contra 0,5844 medida; `alvo` ~0,79 (desceu de 0,95 e SUBIU de volta pela histerese); des-diluída 0,5844 / 0,7277 = 0,803 [0b7d925; `leitura.py:57`].
- **Por que terminou/foi trocado:** dois defeitos de renda, não de gradiente [vault:g1-limpo-piso-da-estatua]:
  - `rastreio_por_elo`: os dois `track_*` zeram nos elos que não andam (ANDAR, CARREGAR andam); piso do PEGAR cai de 8,265 para ~4,5/s; a conta vira 75 contra 88 e explorar ganha. Inverte a decisão da spec §4.2 [369af74]. Smoke 357 ok.
  - Portão por máscara `segmentos > 0`; escada lê `eficiencia_min / forma` [0b7d925]. Smoke 368 ok.
- **Resultado, 577 iterações depois do resume com os dois consertos** (run não nomeado na fonte; ver Lacunas): `forma` 0,7277 → 0,1773 (a rampa chegou ao piso); por env de manipulação `palmas_em_contato` 3,7×, `staged` 3,2×, `squeeze` saiu de zero; `caixa_largada` subiu 24× (pega e perde); locomoção segurou: eficiência des-diluída 0,803 → 0,758, quedas em 0,28%. Gargalo novo: `contato_tronco` = 56% das terminações e `upright` 0,83 → 0,56 [vault:g1-limpo-piso-da-estatua].
- **Lições:** se ficar parado rende mais que o caminho, a política fica parada e está certa — medir o PISO do estado antes de mexer em peso ou σ [mem: g1-limpo-piso-da-estatua-travava-a-pega]; métrica gateada por comando ativo vale zero em quem não recebeu o comando, e a média sobre a frota vira sinal de incompetência; para ler marcha, `seg_proj/seg_pedido` (densa, sem diluição), não `eficiencia_min/forma` (oscilou 0,758 → 0,448 → 0,738 em 300 its sem nada acontecer) [mem: g1-limpo-portao-diluido-pela-fatia].

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
