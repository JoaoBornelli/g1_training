---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: validada
verificado: 2026-10-06
periodo: 11–14/09/26
resumo: "faixa_de_pose: model_14000, tag estavel-bloco17, roda a cadeia inteira com três defeitos (juntas rápidas, escora na caixa, mãos tortas)."
cluster: c3-bloco
---
- **Datas:** 11/09 → ~14/09 (notebook `d74b93f`, META 14000). Tag `estavel-bloco17` = `d74b93f`.
- **O que mudou antes:** `faixa_de_pose` (`c9578ae`): soma por junta de `exp(relu(|q−q_def|−tol)/1,5)−1`,
  tolerância por família × estado, peso −0,5; perna e cintura zeradas no ANDAR e CARREGAR, perna no
  PEGAR. Custo medido 2,4 a 3,7/s. 29 termos (o smoke da tag nasce 636 ok / 2 falhas: contadores,
  memória `g1-limpo-tag-estavel-smoke-vermelho`).
- **Medido:**
  - `model_14000` (handoff 14/09): roda a cadeia inteira (anda, pega, segura, pousa na laje, sai
    andando) com três defeitos — juntas rápidas na manipulação; apoia cabeça/tronco na caixa ao pousar;
    mãos tortas no BOTAR. Na cadeia do carregar `right_shoulder_yaw` 0,28 contra faixa 0,8; na cadeia do
    botar TRAVA em 2,63 com curso 2,62 (`f1e4e22`).
  - `model_13999`: `s_B` 0,62 e `s_C` 0,484 (linha de base da sonda, handoff 14/09).
    `carrega085.csv`: BOTAR p50 0,02 rad/s, pico 11,58 (ANDAR 10,40), 84% dos passos congelados, 6
    juntas além do limite (memória `g1-limpo-botar-e-bang-bang-nao-pressa`). Velocidade de junta: p50
    de UMA junta acima do vmax por passo, p90 de três (`juntas14k.csv`, `7d481d8`).
- **Por que "terminou":** fim da META 14000. O dono passou a fazer correções; o pipeline oficial foi
  congelado na tag (`96be97a`).

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
