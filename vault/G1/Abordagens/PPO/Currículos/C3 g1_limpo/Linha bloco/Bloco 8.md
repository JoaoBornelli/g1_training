---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 03–04/09/26
resumo: "Contrato de troca v2 do zero: marcha madura na it 1503, mas a manipulação regride; a auditoria achou 10 buracos de gradiente."
cluster: c3-bloco
---
- **Datas:** desde 03/09 [mem: g1-limpo-contrato-de-troca-de-tarefa; `specs/g1-limpo-curriculo-de-cadeia.md` §2 "03/09 bloco8 treina do zero com a observação v2"]. Auditoria na it 1503 em 04/09 [auditoria].
- **Plataforma:** Colab com 8192 envs a 8,54 s/iter (lote 196 608 transições, minilote 49 152) [specs/g1-limpo-notebook-colab.md]; seguiu na Kaggle a partir da it 1050 com 4096 envs e 4,8 s/iter [mem: g1-limpo-contrato-de-troca-de-tarefa].
- **O que mudou antes:** todo o contrato v2 (gate da caixa, VALIDA fora, esperas publicando ANDAR, cadeia 3 de três elos, espera final, tamanho da caixa 0,07–0,13 m, `giro_b`, crítico com elo interno, renda monótona do BOTAR, giro no lugar, REORIENTAR inerte, apoio vertical) [d3ea735 … f58b6fb].
- **Medido** [mem: g1-limpo-contrato-de-troca-de-tarefa; auditoria §3, §7]:
  - Locomoção formou na it 324; portão da forma abriu na 502; marcha madura na 1503: eficiência mascarada por env de locomoção 0,745 contra limiar 0,50, `razao_marcha` 0,55, voo 0,147 s, queda 1,9% das terminações. Portão saturado no piso: `alvo_loco_min = 0,30` virou a trava do orçamento de manipulação.
  - Manipulação nunca fecha o PEGAR e REGREDIU entre 1213 e 1503: `squeeze` 92% → 84%, `staged` 1,48 → 1,33, palmas em contato 0,97 → 0,84, `caixa_largada` dobrou para 16,7% das terminações.
  - Decomposição na 1503 (por env de manipulação): preensão 0,843, descarga 0,933, rampa da pelve 0,718 (pelve 0,665 m contra `pelve_alvo` 0,75), `trazer` ≈ 0,48 (d/d₀ ≈ 0,86: caixa a 86% da distância inicial ao alvo), alinha ≈ 0,60; Δθ 8,2° contra 25°. `staged` 1,33 de 2,0; `precise_pos` 0,7%; `sustentacao` 0,06%; `sucesso`, `load`, `largou` 0. `action_rate_l2` −2,16/s.
  - Orçamento: `ELOS_SORTEAVEIS = (REORIENTAR, PEGAR)` uniforme → metade da manipulação é o REORIENTAR inerte (temporizador de 0,3 s); cadeias com CARREGAR ou BOTAR = 4% dos envs [mem: g1-limpo-contrato-de-troca-de-tarefa]; a cadeia 3 é ~6 envs de 4096 no nível 0 [2d07044].
- **Auditoria (04/09), 10 buracos** [auditoria §4]: H1 `precise_pos_sigma` 0,05 < `tol_pos` 0,10 (paga 0,018 no limiar, 1e−26 a 0,385 m); H2 `sustentacao` morta após o 1º avanço (`avancou` pegajoso, `comando.py:1007`); H3 fechar o PEGAR custa ≈ −1,5 a −2,5/s; H4 fechar o CARREGAR custa −2 a −6/s; H5 sem gradiente de volta à pose depois de largar; H6 `pose` morto no CARREGAR; H7 rastreio paga 4,0/s no segurar-parado; H8 `andou ≥ 0,50 m` sem termo de deslocamento; H9 `postura_ereta` satura no limiar de `de_pe`; H10 ANDAR sem kernel macro. Peso do par caixa→alvo invertido (macro 3,0, preciso 2,0).
- **Por que foi trocado:** a auditoria levou à v2.1 de gradientes [53f81e9]; o bloco 9 retoma de um checkpoint do bloco 8 [mem: g1-limpo-auditoria-gradientes]. Checkpoints citados: `model_1800`, `model_2562` [docs/planos/2026-09-04-proposta-gradientes-g1-limpo.md:224; mem: g1-limpo-auditoria-gradientes].
- **Lições:** a gangorra marcha × manipulação (manipulação regride enquanto a marcha amadurece); um termo de aceite com σ menor que a tolerância do fecho é canal morto; condição de fecho e renda pós-fecho têm de ser auditadas juntas (regra do dono "do fecho de um elo ao seguinte a renda não cai").

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
