---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 28/09/26
resumo: "precise_ori ×4 no PEGAR_COM: fechos aparecem e somem; fechar paga menos que pairar e o crítico foge do fecho."
cluster: c3-zero
---
- **Commit:** `829ee03`: `precise_ori` vale 4 no PEGAR_COM. A 60° a derivada vai de 0,18 a 0,72/rad. "Nenhum outro termo vê o tombo."
- **Retomada da zero10 na it ~1300** (REL-TOMBO §1).
- **Medidas** (REL-TOMBO §1–§2):
  - Fechos com `s_B` 0,07 nas its 1446–1568, caixa a ~22°. Depois a caixa vai a 60° e o `s_B` cai a 0,002.
  - Viewer do dono: sobe até o alvo, tomba, desce à mesa, endireita, e repete a cada ~1,6 s.
  - PEGAR forçado, `model_1600` (pico) contra `model_1950` (colapso): fecho/episódio 0,35 → 0,03 no nível 0, tombo p50 25° → 60°. O colapso é igual em todos os níveis, então o piso do currículo NÃO é a causa (REFUTADO).
  - Tombo por distância ao alvo (`model_1600`, 90 278 amostras): 7° a > 0,30 m, 39° a < 0,05 m. A recompensa cresce com a aproximação (15,7 → 25,0/s). Em 781 aproximações, o cronômetro do fecho fica em 0 em 90% delas.
  - Erguer tomba a caixa 1,6 rad por metro de subida. Ombro, cotovelo e `wrist_pitch` giram no mesmo eixo, e só o punho desacopla. A escala de ação do punho é 0,075 rad/unidade, contra 0,439 do ombro e do cotovelo: 6× menor, que é o padrão do mjlab (REL-TOMBO §0, §2.4).
  - Sonda do punho: `wrist_pitch` +0,45 rad → −24° de tombo; −0,45 rad → +42°. A política manda o `wrist_pitch` para o lado negativo, o que tomba (REL-TOMBO §2.4).
- **Renda por estado** (`model_1600`, REL-TOMBO §2.3; MEM:g1-limpo-fechar-paga-menos-que-pairar):

  | Estado | Renda/s |
  |---|---|
  | PEGAR_COM, pairando no alvo | 25,0 |
  | PEGAR_COM + fechou, sem avançar | 32,3 |
  | ESPERA_COM + fechou | 23,2 |
  | CARREGAR + fechou | 16,5 |
  | BOTAR | 20,7 |

- **Diagnóstico:** fechar paga menos que pairar, e o crítico aprende a fugir do fecho. Tombar a caixa é um jeito de não fechar. O padrão é o mesmo nas zero08 e zero11 (REL-TOMBO achados A e B).
- **Por que terminou:** a economia do PEGAR premiava pairar. Isso levou às correções S1–S5.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
