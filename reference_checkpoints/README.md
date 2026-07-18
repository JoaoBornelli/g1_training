# Checkpoints de referência

Milestones bons, versionados de propósito (exceção ao `*.pt` do `.gitignore`).
São `.pt` do rsl_rl (~6MB). O treino normal segue salvando no Google Drive; aqui
ficam só os pontos de partida confiáveis pra warm-start / play / comparação.

| arquivo | skill | descrição | como rodar no play |
|---|---|---|---|
| `model_stand_step_2000_ficar_de_pe.pt` | **stand** | ficar de pé + recuperar empurrão (Stand-Step, it 2000) | `python play.py reference_checkpoints/model_stand_step_2000_ficar_de_pe.pt --task Stand-Step` |
| `model_lift_6550_altura_normal.pt` | **lift** | pegar e erguer a caixa na altura NORMAL (mesa ~0.65, it 6550) | `python play.py reference_checkpoints/model_lift_6550_altura_normal.pt --task Lift` |
| `model_7300_0-55_0-45_height.pt` | **lift** | currículo de altura via PLR: alturas 0.55 E 0.45 dominadas (it 7300) | `python play.py reference_checkpoints/model_7300_0-55_0-45_height.pt --task Lift --shelf-top 0.45` |
| `model_8150_dual_height_stable.pt` | **lift** | 2 alturas [0.55, 0.45] + **payload 0.5–5kg** (it 8150) | `python play.py reference_checkpoints/model_8150_dual_height_stable.pt --task Lift --shelf-top 0.45 --weight 3.0` |
| `model_8650_triple_height_payload.pt` | **lift** | **3 alturas [0.55, 0.45, 0.35] + payload 0.5–5kg** (it 8650) | `python play.py reference_checkpoints/model_8650_triple_height_payload.pt --task Lift --shelf-top 0.35 --weight 3.0` |
| `model_10000_smooth_impact.pt` | **lift** | 4 alturas [0.55,0.45,0.35,0.25] + payload + **action_scale 0.8 + impact** (it 10000). RECUPERADO do dip profundo do warm-start, MAS exibe o hack de "correr" pra minimizar tempo sob penalidade (por-step) → salvo "por enquanto" como referência/comparação, NÃO milestone limpo | `python play.py reference_checkpoints/model_10000_smooth_impact.pt --task Lift --shelf-top 0.25 --weight 3.0` |

Notas:
- `lift_6550` é a **base do currículo de altura**: warm-start daqui pra baixar a
  caixa (prateleira mocap, `shelf_top`) com rehearsal multi-altura, sem esquecer o
  lift alto.
- `7300_0-55_0-45` é o **1º degrau do PLR concluído** (0.55+0.45 via Prioritized
  Level Replay). Generalizou no play até ~0.25 sem retreinar. Base pra: descer o
  currículo (append de altura) E o DR de massa (0.5–5kg) nessas 2 alturas.
- `8150_dual_height_stable` = 2 alturas + payload (o warm-start do currículo de altura).
- `8650_triple_height_payload` = **3 alturas [0.55,0.45,0.35] + payload** — a task
  central madura. **Warm-star SEMPRE com `learning_rate=5e-4`** ([[warmstart-lower-lr]]),
  senão os 1ºs updates destroem o equilíbrio (fell_over dispara). Refino de movimento
  (rebolar/se-jogar) por knob NÃO funcionou — deixar pra depois, gated no hold.
- `stand_step_2000` é a **skill de ficar-de-pé separada** — no orquestrador é ela
  que roda quando não há caixa pra pegar (a política de lift NÃO precisa ficar de
  pé sozinha; são políticas congeladas distintas).
