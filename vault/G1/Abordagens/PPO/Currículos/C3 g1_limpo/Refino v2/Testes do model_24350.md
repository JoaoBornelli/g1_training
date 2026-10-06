---
tipo: achado
pai: "[[Refino v2]]"
estado: ativa
verificado: 2026-10-06
periodo: 05/10/26
resumo: "Sem treino: punho E gira 44° na pega, o aperto é ~240 N fixo (limite do motor, não acompanha a massa), contato pelas cápsulas do pulso e tronco."
---
Fontes: REL-APR §3b; REL-DR §1–§2; MEM:g1-limpo-aperto-fixo-nao-acompanha-massa; MEM:g1-limpo-caixa-presa-pelo-pulso.

- **Punho E** (`24350_B_45_acao.csv`): o `wrist_yaw` E vai de +0,04 a +1,46 rad (88% do curso), com alvo do PD em +0,30 a +0,40 rad (ação +4 a +5).
  - A mão E gira 44° e desliza 7,2 cm sobre a face em 1,5 s; a D, 11° e 1,2 cm. Com impratio 10, a E gira 24° e desliza 4,2 cm.
  - Motor do punho: kp 16,8 e ±5 N·m.
- **Aperto por massa** (laje 0,45, μ 1,0):

  | Massa | `F_ref` | Palma E / D | min / `F_ref` |
  |---|---|---|---|
  | 1 kg | 6,1 N | 40,3 / 36,6 N | 6,0 |
  | 3 kg | 18,4 N | 39,5 / 36,5 N | 2,0 |
  | 5 kg | 30,7 N | 29,4 / 38,2 N | 1,0 |

  Com 5 kg a caixa desce 17 cm na mão.
- **Contatos na caixa** (roteiro `carregar`, mediana; REL-DR §1):

  | Contato | Força lateral |
  |---|---|
  | cápsula do pulso D | 73 N |
  | cápsula do pulso E | 64 N |
  | tronco | 31 N |
  | pad D | 36 N |
  | pad E | 46 N |
  | **total robô → caixa** | **~240 N** |

  - Torques: `shoulder_yaw` em ±24–27 N·m (limite 25) e `wrist_yaw` em ±5 (limite). O aperto é o limite do motor.
  - O `squeeze` só vê os pads e satura (derivada zero). Não existe termo de torque.
- **DR com `priority`** (caixa 2, mesa 3; impratio 2): μ 1,0 e 0,6 seguram 1–5 kg. μ 0,45 solta a caixa de 5 kg em 11,5 s (6,7 s com impratio 1). μ 0,3 solta a de 3 kg em 7,0 s e a de 5 kg em 2,8 s. A força total fica em 230–250 N em todos os casos.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
