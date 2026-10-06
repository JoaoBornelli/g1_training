---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: validada
verificado: 2026-10-06
periodo: 27/08/26
resumo: "Locomoção primeiro com fatia 0,95: razão de marcha 0,553 e portão aberto na it ~963; o PEGAR não fecha e o robô escora na mesa no play."
---
- **Datas:** treino sobre o código de 26/08 (todas as fases F0–F6 ativas, fatia 0,95); log lido em 27/08 no Kaggle [vault:g1-limpo-bloco1-marcha-formou]. Do zero. 5000 iterações (até a it 4999) [vault:g1-limpo-bloco1-marcha-formou].
- **Hipótese:** com a manipulação limitada a 5% e um portão que mede MARCHA (`razao_marcha`, adimensional, nasce 0 para a estátua) a locomoção forma como no multitask [24121d7; f84f9d5; plano].
- **Medido** [vault:g1-limpo-bloco1-marcha-formou, salvo indicado]:
  - Duração antes da razão: na abertura razão −0,08 com episódio de 0,43 s [vault:g1-leitura-log-e-kaggle].
  - it 254: `razao_marcha` −0,136.
  - it 380: razão +0,004, pico do pé 0,0097 m (caiu de 0,0165), escorrego 0,240 m/s, `tempo_de_voo` plano — "arrasto", não passo [vault:g1-leitura-log-e-kaggle].
  - it 500: `tempo_de_voo` 0,0227 s (= 1,1 passo de controle; degrau útil é ≥ 0,10 s) [vault:g1-leitura-log-e-kaggle].
  - it 800: `razao_marcha` **0,468**; erro 91% → 50,3% do comando = 0,380 m/s úteis; `fell_over` 0,083; duração 984/1000; voo, pico, força de pouso e momento angular sobem juntos. Marcha de cadência alta: passo ~3 cm, folga ~18 mm (18% do alvo de 0,10 m; lida no `Metrics/peak_height_mean` do fabricante — o `pico_de_altura` próprio lia 5 mm por diluição de 3,6×), escorrego 0,258 m/s.
  - **Portão abriu na it ~963.** it 1000: razão 0,553 (erro 38,8%, 0,468 m/s), duração 1000, `fell_over` 0,0000, `Curriculum/forma` 0,9500 → 0,8886; voo dobrou para 154 ms, folga 23,8 mm, e escorrego, força de pouso, momento angular e aceleração de ação CAÍRAM.
  - it 1000 → 1525: `forma` 0,8886 → 0,8483, 2 degraus em 525 its contra 44 no desenho: a rampa "queima janelas" na banda morta (0,40 a 0,50) porque `ultimo_degrau = janela` perde as janelas puladas [mem: g1-limpo-rampa-queima-janelas].
  - it 3658: `std` 0,43 → 0,60, entropia 14,7 → 25,5; manipulação acorda (`squeeze` 0,0002 → 0,0664, `postura_ereta` 0 → 0,0256, `unload` 0,14 → 0,33); marcha paga (balanço 398 → 146 ms, escorrego +31%, força de pouso +21%, rastreio angular −65%, `action_rate` +68%, recompensa −30%, razão 0,514 → 0,462). Fatia congelada em 0,787 (79/21 em vez de 70/30). `sucesso` 0,0000 com `avancos` 0,117 ≈ `fatia_cadeia` 0,150.
  - it 4999 (fechamento): duração 984 → 988, `fell_over` 0,000 → 0,167; razão 0,514 → 0,426 com `std` 0,43 (it 1525) → 0,61 (it 4999). `forma` 0,7868 → 0,7850 em 1341 its (UM degrau) [também 7607d8b]. Manipulação: `staged` 0,73, `unload` 0,49, `squeeze` 0,14, `postura_ereta` 0,099, `sustentacao` 0,0002, `sucesso` 0,0000, `avancos` 0,138: o REORIENTAR fecha, o PEGAR não. `tempo_de_voo` subiu 5,9× para 0,860 com altura 29 mm — lido depois como ruído de ação, não claudicação.
  - Play determinístico do dono no `model_3800`: andou bem (régua de verdade). No play, nos elos de manipulação "o robô se joga na mesa" e põe a MÃO NA MESA [f4ccb6c; 6c6a5f8].
- **Por que terminou:** fim do orçamento de 5000 its; o portão congelou na banda morta (a queda de `razao_marcha` era inflação por ruído de ação, não perda de marcha) e o escoro na mesa apareceu no play [7607d8b; f4ccb6c].
- **Mudanças que ele causou:** `contato_prateleira −1,5` booleano (o sensor `corpo_prateleira` não tinha consumidor) [f4ccb6c/ee13add]; o portão passa a ler `eficiencia_min` (projeção por segmento de comando, `min` entre segmentos) no lugar de `razao_marcha`, e o degrau do `std` na escada vai de 0,85 para 0,60 [7607d8b]; smoke 325 ok.
- **Lições que sobreviveram:** locomoção primeiro com fatia 0,95 reproduz a marcha num terceiro módulo (VALIDADO, citado em `docs/relatorios/2026-10-01-inventario-v3.md` C1); ler duração antes da razão; métricas de ação amostrada não se leem com `std` variando, o play determinístico é a régua; `razao_marcha` (soma de normas) é inflada por ruído; a 21% de manipulação a locomoção não paga preço mensurável, o ponto de ruptura entre 21% e 70% ninguém mediu [vault:g1-limpo-bloco1-marcha-formou]; os degraus da escada escolhidos "por múltiplo" (`pico_de_altura ≥ 0,010`, `tempo_de_voo ≥ 0,10` na it 800) falharam enquanto a marcha nascia.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
