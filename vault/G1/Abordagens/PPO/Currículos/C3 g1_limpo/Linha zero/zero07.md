---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 23–24/09/26
resumo: "Freio em currículo por degrau de competência: pernas comprimidas e travou na pega com a pelve a 59°."
---
- **O que mudou (hipótese: o freio sobe com a competência, como no YAM do mjlab):**
  `velocidade_por_regime` troca a MÉDIA pela SOMA das dobradiças (uma junta sozinha paga o
  preço cheio); piso −6/29 = 0,207 por junta (o −6 da zero06 na média); fora do ANDAR o preço
  sobe por env, `freio_fator ** limpo_freio`, ×1,5 por degrau; o degrau persegue o `nivel` do
  env, um degrau a cada 10 episódios de cadeia, só com sucesso, fora de nível sorteado, nunca
  desce, vai no checkpoint. Métricas novas `freio_degrau` e `mao_na_pega`. Kaggle com
  `RETOMA = False`. Linhagem dos blocos fica com −15/29 e teto 0 `[3191e5f] [plano 09-23]`.
- **Medido:** no `model_3150` (elo ANDAR, 64 envs): ruído de perna 0,3 → 0,6 tira 2,35/s da
  recompensa; o maior termo que o bloco 1 não tinha é o freio, −0,44/s, quadrático no ruído.
  "Todas as runs do zero comprimiam as pernas (0,32 a 0,43 contra 0,51 do bloco 1), e a zero07
  travou com a pelve a 59 graus na pega" `[6ab0258]`. A memória diz "tronco 67° na pega,
  escoro" (`g1-limpo-std-cai-cedo`). `model_3150.pt` em `~/Downloads`, 24/09 15:50.
  Também em 24/09 `[9ad847b]`: a face congelava antes de a tarefa ligar; 8 de 256 envs abriam o
  PEGAR com 1,4° a 4,8° de erro sem tocar na caixa; giro máximo na abertura 0,084 → 0,0011 rad.
  O piso da estátua mede 4,86/s contra 4,24 a 4,42 do ANDAR (o `forma_postural` paga 0,81/s à
  estátua desde 16/09) `[9ad847b]`.
- **Por que terminou:** travou; o freio sai do ANDAR `[6ab0258]`. O plano do lidar de 24/09
  guarda a zero07 "como referência sem percepção" (`docs/planos/2026-09-24-grade-de-ocupacao-lidar.md`).
- **Lição:** compare o std das PERNAS, não a média das 29; perna sem exploração não acha a
  marcha.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
