---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 14–17/09/26
resumo: "Três lançamentos com o mesmo nome: colapso pelo amax (it 113), NaN do cone elíptico (it ~755) e pirâmide com impratio 2 sem fim registrado."
cluster: c3-zero
---
São três lançamentos com o mesmo nome. Os documentos os misturam.

#### zero01, 1º lançamento (14/09) — colapso pelo `amax`

- **Datas e lugar:** 14/09, Colab, notebook `g1_limpo_zero_colab.ipynb`, clona
  `exp/g1-limpo-v3` (`[96be97a]`, `[handoff 14/09]`).
- **O que mudou antes (hipótese):** as correções M1–M7 de 14/09. O freio
  `velocidade_por_regime` com `amax` só no regime `standing` (2ª tentativa, `[25c9877]`),
  depois de a 1ª tentativa (`amax` nos três regimes, `[7d481d8]`) matar a `bloco19` em 472
  its (episódio 136 contra 1000, `fell_over` 37 contra `time_out` 2,1, `razao_marcha` 0,055,
  sucesso zero; viver rendia −0,5/s, morrer custava −3,78) `[25c9877]`.
- **Medido:** episódio de 49 passos na it 29 → **8,57** na it 113; `fell_over` 164 → 919;
  `action std` 1,17 → 0,91. Morrer no passo 8 custa −6,6; viver 49 passos custa −30,1.
  Manipulação em 2,7% dos envs, `avancos` 0: o colapso é todo na locomoção `[handoff 14/09]`.
- **Causa:** o molde sorteia `rel_standing_envs = 0.1`, então 10% dos envs de LOCOMOÇÃO
  recebem comando zero e caem no regime `standing`. Robô caindo tem juntas a 15 rad/s:
  `(15/1,5 − 1)² × 2 ≈ 160/s` `[handoff 14/09]`.
- **Por que terminou:** "o dado da it 113 já mostra que não vai" convergir; o dono optou por
  deixar rodar e recomeçar `[handoff 14/09]`. O handoff propõe uma 3ª tentativa (amax só se
  `standing` E estado ≠ ANDAR) e uma `zero02` com `SEG_POR_ITER = 7,4` (T4 mediu 7,38 s/it).
  Nada disso aparece em commit: o código passou a rodar `mean` nos três regimes a partir do
  `[9b6cc8f]` (ver topologia).
- **Lição:** o limiar carrega a redução para a qual foi calibrado; trocar uma sem a outra
  muda o significado do número (`1 − 0,99²⁹ = 25%` dos passos de marcha normal passavam do
  p99 por junta) `[25c9877] [handoff 14/09]`.

#### zero01, 2º lançamento (17/09, manhã) — NaN do cone elíptico

- **O que mudou:** notebook do zero refeito em 16/09 (`[618373f]`, "o dono autorizou
  SOBRESCREVER a `zero01` antiga"), clona `exp/g1-limpo-v2` desde `[a4fbd43]`. Em 17/09
  09:28 ganha `FISICA_ELIPTICA = True` (`[f2ab5ba]`). Traz `limite_de_junta` do zero,
  `limite_de_pelve`, `forma_postural`.
- **Medido:** o elíptico com impratio 10 divergiu para NaN "pela segunda vez (15/07 e 17/09,
  it ~755 no Colab, check_nan no grupo actor)" `[c8bfaa0]`. A lista do log no notebook
  baixado mostra a pasta `2026-09-17_12-41-05_bloco20` com `model_0` a `model_750`
  (`~/Downloads/g1_limpo_zero.ipynb`, célula 7). A pasta saiu com nome `bloco20` porque a
  célula do pacote sobrescrevia o `RUN` `[c8bfaa0]`.
- **Causa (achada em 05/10):** Hessiano do cone indefinido perto do ápice (mujoco_warp #1657,
  fix ausente no 3.10.0.3), busca de linha esgotada com impratio 10 e 20 iterações (mujoco
  #3628), warm-start do Warp que o reset não limpa (`vault/.../g1-limpo-cone-eliptico-nan.md`).
- **Por que terminou:** NaN. `[c8bfaa0]` volta à pirâmide com impratio 2,0 "a pedido do
  dono", para reduzir o creep da caixa nas palmas.

#### zero01, 3º lançamento (17/09, tarde) — "zero01-piramidal_impratio_2"

- **Fonte única:** o notebook baixado `~/Downloads/g1_limpo_zero.ipynb` (22/09 10:41). Célula
  do repo: `RUN = "zero01-piramidal_impratio_2"`, clone `8092387`, `FISICA_ELIPTICA = False`,
  `IMPRATIO = 2.0`, `NUM_ENVS = 8192`, limiar do `limite_de_junta` 0,85 nas famílias (clone
  anterior ao `a37300c`, portanto sem a rampa de 40° no `hip_yaw`). A pasta de log ainda saiu
  como `bloco20`.
- **Medido (saída truncada às últimas 5000 linhas, its 344–392 de 4200; 12,1 s/it; 16 265
  passos/s):**

  | grandeza | it 350 | it 392 |
  |---|---|---|
  | Mean episode length | 985,75 | 978,74 |
  | Mean action std | 0,44 | 0,44 |
  | `razao_marcha` | 0,124 | 0,256 |
  | `fell_over` / `time_out` | 0,29 / 11,0 | 0,29 / 11,0 |
  | `avancos` | 0,000 | 0,014 |
  | `Curriculum/elo` | 0,949 | 0,949 |
  | `velocidade_por_regime` | −0,136 | −0,136 |
  | `fracao_do_curso` | 0,886 | 0,902 |
  | `s_B`, `s_C`, `sucesso` | 0 | 0 |

- **Por que terminou:** não registrado. O `[d4e5e4e]` (20/09) diz só "A `zero01` do Colab
  morreu por NaN; a `zero02` nasce limpa na Kaggle".
- **O que a memória afirma sem fonte primária encontrada:** "a `zero01`, só com a rampa,
  andou e pegou" (memória `g1-limpo-rampa-estreita-nao-ensina`; repetido em
  `docs/relatorios/2026-10-02-avaliacao-v3.md` §2 e em `vault/.../C4 v3/g1-v3-decisoes.md`
  item 19) e "a zero01, com freio −2, também [comprime as pernas]" (memória
  `g1-limpo-std-cai-cedo`). Nenhum número de pega da zero01 foi achado.
- **Ligação com o veredito de 18/09:** a nota do dono cita o treino do zero "`RUN = "zero01"`,
  com o limiar de 40° e o rumo desde a primeira iteração" como o teste dos hábitos. O clone
  `8092387` tem o rumo (`3a4701c`) mas não o limiar de 40° (`a37300c`, 16:10). Ou houve um
  4º lançamento, ou a nota descreve a intenção.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
