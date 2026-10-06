---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 22/09/26
resumo: "Freio −15 e regime pelo estado: o batente da cintura domina (46,6 das terminações); rampa mais estreita que o ruído não dá gradiente."
cluster: c3-zero
---
- **Lugar:** Colab, notebook enxuto `g1_limpo_zero_colab.ipynb` (9 células, 15 KB),
  `NUM_ENVS = 8192`, `SEG_POR_ITER = 9,0` `[0682e6e]`.
- **O que mudou antes (hipótese):**
  - Regime do freio pelo ESTADO (`env.limpo_twist_zerado`): o laço de rumo tirava a pega do
    `standing` em 31,3% dos passos, 58% do freio perdido. Peso −2 → **−15**, por SEGURANÇA DE
    PESSOAS: mão a 2,58 m/s na pega contra 0,25 m/s da ISO/TS 15066; rajada de 0,62 s passa a
    custar 8,64 contra ~6,9 que a pressa compra `[fe8f3e6]`.
  - `FACE_DE_PE` no BOTAR: eixo Z da caixa contra a vertical; o `precise_ori` vira de multa
    2,00 a prêmio 1,26 por endireitar `[5698f08]`.
  - σ do `FACE_CONGELADA` = `tol_ang_deg` 25° (kernel 0,852/0,368/0,011 a 10°/25°/53°)
    `[ab8ea24]`.
  - Mantém `NoBatente` e rampa da cintura (80; 0,05; 0,95), pirâmide com impratio 2 `[0682e6e]`.
- **Medido:** contagem de recompensas real 31, não 29 (o assert caiu na 1ª run, `[8a457e4]`).
  `Iteration time` 12,67 s com 8192 envs (com 9,0 a sessão pedia 14,8 h contra 10,5)
  `[1643fb3]`. Na it 451: `batente_da_cintura` **46,63** contra `time_out` 6,33 e `fell_over`
  0,33; episódios de manipulação em ~10 passos contra ~800 da locomoção; `Curriculum/elo`
  0,9499, previsão 0,949 `[1643fb3]`.
- **Causa:** a rampa de 0,95 tinha 1,5° de largura num meio-curso de 30°; com `Mean action
  std` 0,51 o ruído atravessa 1,5° num passo: nenhum aviso antes do batente `[1643fb3]`.
- **Por que terminou:** parada na it 451.
- **Lição:** rampa mais estreita que o passo do ruído não dá gradiente; a terminação vira o
  único sinal.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
