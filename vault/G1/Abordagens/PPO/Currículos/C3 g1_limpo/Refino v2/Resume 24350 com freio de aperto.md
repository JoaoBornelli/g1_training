---
tipo: abordagem
pai: "[[Refino v2]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 05/10/26
resumo: "priority, DR de μ [0,6; 1,0] e aperto_excessivo −0,05: a força fica em ~200 N e nível e s_C caem; o 24800 anda mais com a caixa."
---
- **Commits:** `0912915` (código; spec SPEC-APERTO), clone `025a9d6`.
- **Mudanças:**
  - `priority` 2 na caixa e 3 na laje.
  - Evento `atrito_caixa` com μ em [0,6; 1,0]; `squeeze_mu` 0,8 → 0,6.
  - Sensor `caixa_robo` (12 fatias, `maxforce`).
  - Termo `aperto_excessivo` = relu(F_tot/(2k·F_ref) − 1)², k = 3, peso −0,05, fora do BOTAR.
  - Métrica `forca_total_na_caixa`.
- **Log** (SESS-05/10, leituras dos prints):

  | Faixa de its | Leitura |
  |---|---|
  | 24350 → 24422 | força total em platô de ~211 N; `s_C` caiu de 0,37 a 0,07 e voltou a 0,33 (choque do resume: a `priority` passou condim e solref da caixa a todos os contatos); `caixa_largada` 0,6–0,7 |
  | 24422 → 24521 | `s_C` 0,42; força 211 → 203 N |
  | 24521 → ~24700 | força 203 → ~200 N; `s_B`/`s_cauda`/`s_C` 0,75 / 0,93 / 0,43 |
  | 24700 → 24800 | força 200 → 195 N; `caixa_largada` ~0,52 → 0,39; `s_B` 0,78 → 0,695; `s_C` 0,45 → 0,386; nível ~3,03 (24,6k) → 2,886 (24807) |

  - Derivada do preço a 200 N com 1 kg: ~0,006/s por N (`git:79b6afc`).
  - O assistente recomendou parar e guardar o pico (~24600–24650).
- **Plays por massa** (12 casos, `carregar`, laje 0,45, impratio 2; SESS-05/10):

  | Checkpoint | μ | Força no PEGAR (1 / 3 / 5 kg) | Força no CARREGAR (1 / 3 / 5 kg) | Alvo do punho E | Andou em 10 s |
  |---|---|---|---|---|---|
  | 24600 | 1,0 | 235 / 250 / 259 N | 232 / 233 / 243 N | +0,53 rad | 2,3 / 1,8 / 1,8 m |
  | 24600 | 0,6 | 236 / 249 / 264 N | 234 / 240 / 225 N | +0,54 rad | 2,3 / 2,1 / 1,9 m |
  | 24800 | 1,0 | 213 / 218 / 226 N | 216 / 239 / 258 N | +0,67 rad | 3,5 / 2,9 / 2,9 m |
  | 24800 | 0,6 | 222 / 225 / 237 N | 226 / 234 / 238 N | +0,65 rad | 3,4 / 3,0 / 3,0 m |

  - A caixa não cai em nenhum caso. A política não modula o aperto pela massa.
  - O 24800 anda mais com a caixa (~3 m contra ~2 m, para 5 m comandados).
  - O comando de giro do punho cresceu (ação de 7 para 9).
  - BOTAR com 3 kg e μ 0,6: a caixa pousa de pé nas lajes 0,15, 0,35 e 0,55, e o robô sai andando ~4,4 m. Os CSVs `24800_*.csv` estão na raiz do repo.
- **Punho no 24800** (`24800_mu06_m3.csv`; MEM:g1-limpo-punho-esquerdo-empurrado-pela-caixa):
  - No PEGAR, a ação do `wrist_yaw` E sobe a +8,5 (alvo +0,60 rad) e a junta vai a +1,18 rad. Os 0,58 rad além do alvo pedem ~9,7 N·m, contra ±5 do motor: é o contato.
  - No CARREGAR, a ação cai a −1 e a mão volta a +0,2 rad.
  - O giro é COMANDO no elo PEGAR mais CONTATO. A versão "só contato" foi corrigida. A mão gira para cima, não para baixo (dono).
- **Postura no 24800** (SESS-05/10; `git:af97a5a`):
  - Na laje 0,15 com 5 kg, PEGAR em t = 1,92 s: pelve 0,543 m (IK 0,549), pelve inclinada 76° (IK 39°), tronco 109° (IK 66°), cabeça a 7 mm da caixa.
  - Na janela da pega, os pés ficam 0,10–0,12 m mais estreitos que a IK (r 0,39–0,49), e o tronco 20–28° fora (r 0,69–0,78). Sola e quadril já em r ≥ 0,91.
  - Com 5 kg na laje 0,15, o dono viu: "não parece estar agachando o necessário e na pega/botar ele parece bater a cabeça na caixa". "Com mu de 0.6 ele ainda consegue pegar, abaixo disso deixa cair."

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
