---
tipo: abordagem
pai: "[[Linha zero]]"
estado: parada
verificado: 2026-10-06
periodo: 20–22/09/26
resumo: "Cintura em família própria e NoBatente, na Kaggle: chegou à it ~11191 com s_B 0,32; a caixa tomba na descida; trocada quando a recompensa mudou."
cluster: c3-zero
---
- **Datas e lugar:** nasce em 20/09 na Kaggle, 4096 envs, notebook `g1_limpo_zero.ipynb`;
  "linhagem nova", continuação pelo notebook de resume `g1_limpo_kaggle.ipynb` `[d4e5e4e]`.
  O 1º lançamento morreu na célula 11 com `ModuleNotFoundError` (import de
  `TerminationTermCfg` de módulo inexistente), corrigido em `[713a855]`.
- **O que mudou antes (hipótese):**
  - A cintura vira família própria. A rampa de 0,85 cobrava a própria referência de IK:
    `waist_pitch` (curso ±29,8°) a 26,8° em 14 das 15 poses = frac 0,900, custando 1,72 por
    passo. No zero a cintura vai a (k 80; teto 0,05; limiar 0,95): referência paga zero,
    batente custa 53,6 em vez de 19,1 `[d4e5e4e]`.
  - Terminação `NoBatente` nas três juntas da cintura (`VIGIA_BATENTE = True`,
    `FRAC_MAX = 1.00`): rampa que acaba no batente deixa o além sem derivada; o
    `waist_pitch` medido chegava a 1,061. Tornozelo fora (`ankle_roll` frac 1,49,
    `ankle_pitch` 1,23 pelo contato). `hip_yaw` fora de propósito: teste do preço sozinho
    `[d4e5e4e]`.
  - A chave da referência de IK no PEGAR passa a ser a caixa `[8cb68cc]`.
  - Freio ainda a −2 na média, regime pelo comando (o conserto de 21/09 é posterior; status
    20/09: "Nenhum checkpoint até o `zero02` na 5477 viu este código").
- **Medido:**
  - `impacto_da_caixa` 8,66 na it 3 451 e 6,79 na 5 477 (contra 6,27 no bloco22 it 21 510):
    plano em toda a linhagem, sem peso `[plano 09-21 cuidado]`.
  - `model_5477`, carregar: caixa |v| p95 0,92 m/s, máx 3,30, aceleração máx 6,5 g (o
    `model_24999` da linha bloco: 0,48 / 0,51 / 1,0 g) `[plano 09-21 cuidado]`.
  - `velocidade_de_junta` 1,92 na 5477 (status Obsidian, "o que vigiar").
  - `model_9100` (21/09, `registra_juntas`, CSVs `ckp9100_m2_t0{55,35,15}_i2*.csv` na raiz,
    datados 21/09 15:30–16:38): caixa chega ao BOTAR tombada 29,7° (laje 0,55) e 52,2° (laje
    0,35) e termina a 0,0°; giro total da face 33,2° e 53,2°; altura exata (0,650 e 0,450);
    erro xy 0,06–0,17 m. Na laje 0,15 o robô cai (rumo gira 102°, pelve 0,334, tronco 129°).
    `sigma_ori` preso no piso 0,20 rad: `precise_ori` 0,0002 a 33,2°, derivada 1e−4 por grau.
    `s_B` = 0,32 (`vault/.../2026-09-21-botar-alinhado-pune-endireitar.md`,
    `[5698f08]`, `[plano 09-21 botar]`).
  - Its 10 751 a 11 191: `caixa_na_pega` entre 53° e 59°, contra 25° do fecho; o dono vê no
    viewer o tombo na DESCIDA, o punho não compensa o arco do cotovelo `[ab8ea24]`.
- **Por que terminou:** o conjunto de recompensa mudou em 21–22/09 (regime pelo estado e
  freio −15 `[fe8f3e6]`, `FACE_DE_PE` `[5698f08]`, σ pela tolerância `[ab8ea24]`), e a regra
  dos notebooks é RUN novo a cada mudança de conjunto. Em 22/09 nasce o notebook só-Colab com
  `RUN = "zero03"` `[0682e6e]`. Nenhum documento registra um veredito da zero02.
- **Lições:** (1) `σ = erro inicial` só funciona com alvo independente do robô; alvo
  congelado da pose atual nasce com erro zero e o kernel morre a partir de ~20°, então tombar
  a caixa ao erguê-la era grátis `[ab8ea24]`; (2) no BOTAR o `alinhado` congelado punia o
  ENDIREITAMENTO `[5698f08]`; (3) a cinemática da caixa no carregar melhora com maturidade
  (6,5 g → 1,0 g), a da pega não `[plano 09-21 cuidado]`.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
