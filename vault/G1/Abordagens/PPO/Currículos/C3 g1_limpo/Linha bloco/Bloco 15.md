---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: parada
verificado: 2026-10-06
periodo: 10–11/09/26
resumo: "Giro sem gingado e envelope ±1,6 rad/s; o model_10200 gera a tabela por estado (teto do que falta ≥ piso do feito)."
---
- **Datas:** 10/09 → 11/09 (notebook em `d2ac2ac`: `RUN=bloco15`, META 12500).
- **O que mudou antes:** `giro_sem_gingado` (`d2ac2ac`) tira `wx² + wy²` do erro de guinada; o 3º
  estágio do envelope sai (filtro por `step < 10000×24`); envelope de guinada ±0,7 → ±1,6 rad/s (volta
  em 3,93 s, pedido do dono "uma volta em no máximo 4 s"; `80244b2`, spec `envelope-de-giro`), com σ
  proporcional ao comando (`σ = max(|cmd_wz|·1,0, 0,707)`; derivada no topo 12×). Retomada do
  checkpoint mais alto (spec `giro-e-teclas` §0b.3).
- **Medido:**
  - Giro (memória `g1-limpo-giro-canal-e-envelope`): canal angular foi a 38%, erro de guinada caiu
    25%. Iteração não registrada.
  - `model_10200` (spec `tabela-por-estado` §0; memórias de 10/09): piso × teto — BOTAR 13,79 contra
    ~7 (2:1); cauda C 21,76 contra ~8 (3:1); CARREGAR 13,79 + 13,20 contra 4 (7:1). Não anda com a
    caixa: `|qd|` de perna p50 0,039 contra 0,578 (15×), 97,2% dos passos com dois pés no chão, n =
    16 663; andar a 1 m/s acrescenta +1,49/s = +5,2% da renda parada de 28,64/s. Não levanta depois de
    botar: pelve p10 0,469 m, mediana leva 8 s. Corre na pega: `|qd|` p90 do PEGAR 2,32 rad/s
    (pool de manipulação parada 0,72; locomoção parada 0,10; andando 0,95). Squeeze cego: força nas
    palmas p50 68,9 N contra `F_ref` 8,45 N (n = 25 852). `renda_congelada` sobrepaga 10,3× na
    cadeia B e 1,22× na C. `pose` na pega 0,0272 com |Δ| p50 3e−05 (punho σ 0,30 matou o canal);
    `right_wrist_roll` 2,034 rad além do batente 1,972 (`bcd0e84`). 2,26–2,71% dos passos do PEGAR no
    clamp do G2 (`a6259fe`, `954ed94`).
  - ⚠ A memória `g1-limpo-velocidade-de-junta-medida` atribui o `model_10200` à bloco15, mas a bloco14
    foi parada na ~10300; o 10200 pode ser da bloco14. Não confirmado.
- **Por que terminou:** o lote da tabela por estado mudou a forma da recompensa → nome novo bloco16
  (`dbcc299`). Iteração final não registrada.
- **Lição:** "o teto do que falta fazer tem de ser ≥ o piso do que já foi feito".

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
