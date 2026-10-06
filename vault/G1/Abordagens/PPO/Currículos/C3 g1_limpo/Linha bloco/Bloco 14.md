---
tipo: abordagem
pai: "[[Linha bloco]]"
estado: validada
verificado: 2026-10-06
periodo: 10/09/26
resumo: "BOTAR sem de_pe e cauda parada de pé: o BOTAR fecha pela primeira vez (s_C 0 → 0,195); parada pelo 3º estágio de velocidade."
cluster: c3-bloco
---
- **Datas:** 10/09 (notebook `e63e92e`, META 11500).
- **O que mudou antes:**
  - v3.4 (`0234e7b`, spec `botar-fecha-e-para`; decisão do dono de 10/09): `de_pe` sai do fecho do
    BOTAR; a cauda pós-BOTAR fica parada de pé (twist zero até o fim) com a cena intacta (o robô se
    apoia na caixa, `forca_de_apoio` p50 = 1,24·m·g); `sustenta_outros_s` 0,3 → 0,5 s.
  - v3.5 (spec `cauda-parada-de-pe`; `aa02fd1`, `41002df`, `78d859e`, `42c4453`): `VALIDA` lê o elo
    publicado (a cauda não paga `precise_pos`/`load` duas vezes); depois do fecho, tirar a caixa do
    alvo termina; `postura_ereta` paga a pelve sozinha depois do fecho; punho na média do `pose` com
    `std_standing[".*wrist.*"] = 0,30`.
- **Medido:**
  - it 9847 (memória `g1-limpo-botar-paga-nao-botar`): **`s_C` 0,0000 → 0,1952**, `sucesso` 0,167 →
    0,306. A memória `g1-limpo-plano-tres-tarefas` registra "`s_C` 0 → 0,34" para a v3.4 (fonte e
    iteração não identificadas).
  - it 9847, giro (spec `giro-e-teclas` §0.2): `track_linear_velocity` 59% do máximo (painel 0,9951),
    `track_angular_velocity` **6,5%** (painel 0,1095); erro² angular 1,37 contra linear 0,132. O robô
    não gira (visto no `pilota`).
  - Cruzamento da it 10000 (spec `giro-e-teclas` §0b; commit `d2ac2ac`), 9847 → 10299: `s_C` 0,1952 →
    0,1418; `sucesso` 0,3062 → 0,2230; Mean reward 156,3 → 126,6; `caixa_largada` 10,7% → 16,3%;
    action std 0,60 → 0,67; entropy 25,2 → 28,5. O 3º estágio do molde subiu `lin_vel_x` a (−2,0; 3,0)
    por passo fixo.
- **Por que terminou:** o dono parou na ~10300 ("corta o estágio, o robô não precisa aprender a
  correr", 10/09).
- **Lição:** estágio de currículo por passo fixo atravessa o resume e pode derrubar a tarefa; o
  primeiro BOTAR fechado da linha.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
