---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: validada
verificado: 2026-10-06
periodo: 15–16/09/26
resumo: "v2 refeita da tag: alvo 0,85–0,95, inércia real e limite_de_junta exponencial; model_15200 anda agachado com a caixa (pelve 0,564)."
---
- **O que mudou antes (`9b6cc8f`, sobre a tag `estavel-bloco17`, retomada do `model_13999`/`14000`):**
  1. Alvo do PEGAR sorteado em 0,85–0,95 por episódio (o 1,02 era circular: altura do ombro 1,074;
     custo de braço Σ|q−default| 1,30 a 1,02 contra 0,80 a 0,85; piso 0,80 pelo fecho).
  2. Inércia real da caixa: `carga_caixa` escreve `body_mass` e `body_inertia` (`I = (2/3)·m·a²`) em vez
     de força externa; `dr.pseudo_inertia` não corrompe mais a heap no mjlab 1.5.3 (A/B CPU, 8 envs,
     massa 1,04–4,18 kg). GPU não verificada.
  3. `limite_de_junta` exponencial substitui `dof_pos_limits` (a reta cobrava 0,055/s com o
     `waist_pitch` no batente o BOTAR inteiro). Rampa a partir de 85% do curso, teto por família.
     Reprovadas no portão pré-GPU: `k 20 / teto 0,15` (`left_ankle_roll` 95,7% no teto) e
     `k 12 / teto 0,25` (`right_wrist_roll` 59,1%). Custo medido em `carrega085.csv`: andar 3,78/s,
     pegar 22,31/s, botar 47,10/s, zero na pose default. Métrica `fracao_do_curso`. Limitação 11: o
     `ankle_roll` chega a 150% do curso (restrição mole do MJCF).
  - Notebook: `RUN = bloco18`, META 16000 (`611a5f1`, `af34cab`).
- **Medido:** `model_15200` (`docs/planos/2026-09-15-limite-de-pelve-no-carregar.md`, "do `bloco19`"):
  pelve 0,748 na pega → **0,564 ao carregar** (joelho +1,718, `hip_pitch` −1,016, desvio 1,2 cm), andar
  vazio 0,726; palma z 0,871 contra alvo 0,85; perda do `precise_pos` no p90 0,8%. Causa:
  `postura_ereta` = 0 na coluna CARREGAR (`knobs.py:1143`). `std` da ação (vault
  `g1-limpo-std-da-acao-por-junta`): média 0,74 no 15200, 0,78 no 15999; punho pitch/yaw 1,22–1,31.
  Checkpoint corrente em 15/09: `model_15999.pt` (`d117342`).
- **Por que terminou:** META 16000. ⚠ `7ede785` (16/09) diz "o bloco 19 fecha em 16000" e `618373f`
  mede "no log do bloco19 (4096 envs × 24 passos a 14265 passos/s), 6,9 s/iter". O código do bloco18
  parece ter rodado (todo ou em parte) sob o nome `bloco19`, que já tinha sido usado pela run colapsada
  da v3. Não confirmado.
- **Sucessores (fora da janela):** `bloco20` = limite de pelve (`c1a5266`, 16/09, META 18000);
  `bloco21` = `forma_postural` contra a IK, retomada do 15999 (`a4fbd43`/`46dea30`, registro de
  checkpoints); `bloco22` = sola e rotação do quadril (`9495e32`) e terminação `|hip_yaw| > 1,0` fora
  do git → `model_21700`.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
