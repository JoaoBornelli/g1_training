---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 29/09/26
resumo: "Reta zera a 90° e âncora baixa 0,75–0,85: fecho forçado do PEGAR mata 28 de 28 episódios; o CARREGAR nunca foi praticado."
cluster: c3-zero
---
- **Commit:** `ce3d460`.
- **Mudanças:**
  - A reta vira `max(0, 1 − Δθ/(π/2))`: 0,67 a 30°, 0 a 90°, 0,64/rad.
  - **Caminho A (decisão do dono):** `altura_carregar_faixa` (0,85; 0,95) → (0,75; 0,85), `peito_b.z` 0,102 → 0,002. A subida natural chega com −15° a +15°.
  - O fecho do PEGAR exige `~apoiada`.
- **Medidas** (SPEC-CADEIA §1; `git:b2909e9`):
  - Play do `model_3450`: o PEGAR reprova no tombo (25–29°) e no de pé (`hip_pitch` a 0,96 rad do default, limite 0,69). Perto e fora do apoio passam em 100%.
  - Rollout sem update (48 envs): 34 de 36 episódios de manipulação no PEGAR com a caixa erguida > 5 cm; 0 de 36 com algum elo fechado; 45 de 48 chegam ao `time_out`.
  - `s_B`: 0,006 (it 3147) → 0,0012 (3347) → 0 (3484).
  - Fecho forçado do PEGAR com a caixa a < 12 cm do alvo: 28 de 28 terminam em ≤ 1,6 s (24 `caixa_largada`, 4 `fell_over`). Com o one-hot fora da normalização, 11 de 11 terminam (0,6–1,9 s). Com a laje mantida, 7 de 8 terminam em ≤ 1,5 s.
- **Causa:** a política nunca praticou o CARREGAR com a caixa. Quando o one-hot vira CARREGAR, ela larga a caixa, mesmo com twist zero (SPEC-CADEIA §1; MEM:g1-limpo-fechar-o-pegar-mata-o-episodio).
- **Normalização do one-hot no `model_3450`** (SPEC-CADEIA §1):

  | Slot | σ | Entrada da rede com o slot aceso |
  |---|---|---|
  | ANDAR | 0,4986 | ~1 |
  | PEGAR | 0,4985 | ~1 |
  | REORIENTAR | 0,0302 | ~25 |
  | BOTAR | 0,0012 | ~89 |
  | CARREGAR | 0,00055 | ~94 |

- **Divergência de fonte:** SPEC-CADEIA chama o `model_3450` de "da zero15". MEM:g1-limpo-fechar-o-pegar-mata-o-episodio diz "model_3450, zero16". Pela transcrição de 30/09, o `model_3450.pt` foi baixado em 29/09 às 20:41. Isso é antes do commit da zero16 (30/09 08:27), então "zero15" é o mais provável.
- **Por que terminou:** o dono reverteu a âncora baixa: "uma correção sem relação com o problema" (`git:b2909e9`).

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
