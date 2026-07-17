# Checkpoints de referência

Milestones bons, versionados de propósito (exceção ao `*.pt` do `.gitignore`).
São `.pt` do rsl_rl (~6MB). O treino normal segue salvando no Google Drive; aqui
ficam só os pontos de partida confiáveis pra warm-start / play / comparação.

| arquivo | skill | descrição | como rodar no play |
|---|---|---|---|
| `model_stand_step_2000_ficar_de_pe.pt` | **stand** | ficar de pé + recuperar empurrão (Stand-Step, it 2000) | `python play.py reference_checkpoints/model_stand_step_2000_ficar_de_pe.pt --task Stand-Step` |
| `model_lift_6550_altura_normal.pt` | **lift** | pegar e erguer a caixa na altura NORMAL (mesa ~0.65, it 6550) | `python play.py reference_checkpoints/model_lift_6550_altura_normal.pt --task Lift` |

Notas:
- `lift_6550` é a **base do currículo de altura**: warm-start daqui pra baixar a
  caixa (prateleira mocap, `shelf_top`) com rehearsal multi-altura, sem esquecer o
  lift alto.
- `stand_step_2000` é a **skill de ficar-de-pé separada** — no orquestrador é ela
  que roda quando não há caixa pra pegar (a política de lift NÃO precisa ficar de
  pé sozinha; são políticas congeladas distintas).
