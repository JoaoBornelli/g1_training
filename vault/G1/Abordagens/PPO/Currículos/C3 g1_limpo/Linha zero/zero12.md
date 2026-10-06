---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 29/09/26
resumo: "S1–S5 (renda pós-fecho, _alinha, σ fixo), do zero: a pega parou na mesa e depois deitou a caixa a 90°."
cluster: c3-zero
---
- **Commit:** `7d4be42`. Plano: PLANO-S ("RUN nova: zero12, do zero"). Regra do plano: alterar código existente, nenhum termo novo, nenhum peso subido.
- **Mudanças:**

  | Item | Mudança | Efeito previsto |
  |---|---|---|
  | S1 | coluna PEG_COM: rastreio 1 → 0, `pose` 4 → 1 | pairar cai de 25,0 para 18,6/s |
  | S2 | quem falha o `perto` no fim da espera vai à cauda CARREGAR; `tem_prox` exclui o CARREGAR | o estado de 32/s deixa de existir |
  | S3 | `_alinha` (o híbrido) multiplica o `trazer` do `staged` e o `precise_pos`, só no regime de pé | erguer tombando 17° → 60° passa de +0,10 a −2,43/s |
  | S4 | σ de orientação fixo na tolerância no regime de pé | fecha o exploit de alargar o σ tombando na espera |
  | S5 | RUN zero12, assert de clone, as três tabelas na impressão digital | — |

  Rejeitados (REL-TOMBO): `renda_congelada` ×1,7, portão de 40°, janela gaussiana σ 25°, escala do punho (dono), `condim 4`.
- **Medidas** (REL-MESA §1, `model_1750`, nível 0, 32 envs, 18 s, 26 208 passos):

  | Grandeza | p10 / p50 / p90 |
  |---|---|
  | força da palma mais fraca / `F_ref` | 3,8 / 4,9 / 6,7 |
  | subida máxima por env | 2 / 8 / 13 mm |
  | `ANG` | 0,1° / 0,25° / 1,2° |
  | caixa ao alvo | 0,23 / 0,29 / 0,34 m |

  - 82% dos passos com a caixa na mesa. Renda 12,7/s, dos quais ~11/s não dependem de progresso.
  - Platô da it ~1900 à 2135 (REL-MESA). Estagnou na it 2629 com a caixa fora da mesa e longe do peito. No log, `unload/(2·squeeze)` ~0,8 (era 0,18 no model_1750), `renda_congelada` e `sucesso` em 0 (`git:2b35d61`).
- **Cinemática** (REL-MESA §2): levar a caixa da mesa à âncora do peito gira a mão de 20° a 35° por construção (eixos `0 1 0` coaxiais no `g1.xml`). Até 28/09 erguer tombado pagava +5,5/s sobre ficar na mesa. Depois da S3: +2,9/s a 30° e +1,5/s a 40°. Erguer reto pagaria +7,4/s.
- **Play da it ~3100** (`juntas.csv`): o robô DEITAVA a caixa com a face de cima para o peito, 43° aos 2 s e 89° aos 7,5 s. O `caixa_na_pega` do log marcava ~8°, diluído entre os envs (`git:ce3d460`; REL-AUD §0).
- **Por que terminou:** a pega parou na mesa, porque cada correção do tombo encareceu o único erguer que a política sabia (MEM:g1-limpo-pega-parada-na-mesa).

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
