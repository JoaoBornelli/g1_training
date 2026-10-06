---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 09/09/26
resumo: "Dois bits v3.1 (cadeias B, R, C, alvo congelado): o robô ATIRA a caixa no BOTAR; termo que paga estado da caixa tem de exigir a mão."
---
- **Datas:** 09/09 (notebook `c1801ea` com `RUN=bloco11`).
- **O que mudou antes (08/09):**
  - Lote F1/F2/G1/G2 (`8332caf`): σ de alcance medido no fim da espera; `pose_de_braco`
    `exp(-(rms(dq_bracos)/1,0)²)` peso 1,0; rastreio volta nos elos parados quando a tarefa está ativa
    E `limpo_pegou`; `velocidade_por_regime` peso −2,0 com `standing .* = 2,0 rad/s` (p99 da
    locomoção parada). Smoke 524 → 588 ok (`c3c9d08`).
  - Dois bits v3.1 (spec `g1-limpo-dois-bits.md`; `47bf36c`, `1cab2a0`, `02179e9`, `b2cba3e`, `60b8fdf`,
    `34a356f`): alvo congelado com twist zero; BOTAR abre com a pose corrente da base, alvo na borda
    perto do robô; `CADEIAS` 4 → 3 (B, R, C), CARREGAR vira estado de cauda; fecho só ARMA a espera,
    o avanço é em `_aplica_espera`; `prob_por_nivel` (28 números) sai e entra o balanceador B/C (`p_C`
    por EMAs de `concluiu`); `sustentacao`, `largou` e `pose_de_braco` saem, `load` volta (28 termos);
    `PosturaPorElo` age em todo elo; `fell_over` ganha cláusula de joelho (`joelho_z_min` 0,10 → 0,07
    → 0,05); G2 vira `clamp(média(v²/vmax²), max=4,0)`; `PPOPorElo` em 5 grupos. Smoke 596 ok / 0.
  - Hipótese: o BOTAR nunca fechava por abertura mortal, alvo errado e deriva.
- **Medido (spec `g1-limpo-soltar-termina.md` §0; commit `88fe9b5`):** its 7044 → 7299: `s_C =
  0,0000`, `load = 0,0014`, `impacto_da_caixa` 4,97 → 7,70, `caixa_largada` 10,5% → 38,5% das
  terminações, `avancos` 0,59 ("o maior já visto"). O robô chega ao BOTAR e **atira a caixa** no
  alvo. Sonda de 8 estados: atirar 10,39/s contra apoiar 12,19/s, com as mãos paradas em todos.
  Causa: `staged` (até 6/s, `_alcancar ≡ 1` no BOTAR), `precise_pos` (até 3/s, sem gate de mão) e
  `precise_ori` pagam posição da caixa sem exigir a mão; `squeeze`/`unload`/`postura_ereta` valem
  zero no BOTAR. Segundo buraco dormente: na cauda `staged = 1` → 3,0/s com a caixa a 5 m.
- **Por que terminou:** interrompida (~7300) para o lote v3.2.
- **Lição:** todo termo que paga estado da caixa tem de exigir a contribuição do robô (memória
  `g1-limpo-botar-paga-nao-botar`).

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
