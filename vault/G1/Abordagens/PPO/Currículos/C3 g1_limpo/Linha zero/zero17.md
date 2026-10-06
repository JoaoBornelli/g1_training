---
tipo: abordagem
pai: "[[Linha zero]]"
estado: validada
verificado: 2026-10-06
periodo: 30/09–01/10/26
resumo: "Currículo de cadeia por fases, one-hot fora da normalização e R4/R4b: s_C 0,004 → 0,20 e BOTAR resolvido, mas agachado, CARREGAR ruim e quedas; o 21700 no mesmo código superou."
---
### Desenho
Fontes: `git:c0352b0`, `git:a96039a`, `git:b919e32`, SPEC-CADEIA.

- **Fases** (só avançam):

  | Fase | Fecho do PEGAR em B e R | Cauda CARREGAR | `p_C` |
  |---|---|---|---|
  | 1 | 90% SEGURAM no PEGAR (sem reabrir σ, alvo e eixo); 10% vão à cauda | parada | 0,05 |
  | 2 | 100% vão à cauda | 90% parada | 0,05 |
  | 3 | 100% vão à cauda | anda | 0,05 |
  | 4 | comportamento anterior | anda | balanceador, piso 0,20 |

- **Trocas:** 1 → 2 com `s_B ≥ 0,50` e ≥ 300 its; 2 → 3 com `s_cauda ≥ 0,60` e ≥ 200 its; 3 → 4 com `s_cauda ≥ 0,60` e ≥ 300 its.
- **One-hot fora da normalização:** média 0 e desvio 1 fixos no ator e no crítico. O `load` dobra a normalização antiga na 1ª camada.
- **Orientação no CARREGAR:** `precise_ori` vai de 0 a 4 na coluna CARREGAR. Fecha o achado 32: com a caixa nivelada, +2,4/s, parada ou andando (`git:a96039a`).
- **C = (PEGAR, CARREGAR parado, BOTAR):** o CARREGAR da C fecha pela régua do PEGAR em 0,5 s, e a laje fica até o BOTAR. Fecha o achado 3 (`git:b919e32`).
- **Sucesso:** `concluiu_ate_o_fim` = `concluiu ∧ time_out` move o nível e as EMAs `s_B`/`s_C`. Decisão do dono, contra a spec dois-bits §2.5 (`git:b919e32`).
- **Sondas na CPU** (`model_3450`, 40 envs):
  - A SEGURA sobrevive 4 s em 13 de 19 fechos forçados (antes, 0 de 28).
  - Ela paga 23,25/s contra 11,67/s de pairar.
  - Com a dobra no `load`, o ator fica idêntico no ANDAR, no REORIENTAR e no PEGAR (erro relativo 1e−5).
  - Depois do `b919e32`: a C inteira fecha por sustentação em 4 de 4 envs, e a SEGURA paga 23,95/s contra 11,63/s, com 20 de 26 vivos em 4 s.
- **Ponto de partida:** a spec diz "RUN zero17, do zero", mas o dono retomou do `model_3100` (MEM:g1-limpo-curriculo-de-cadeia; SESS-30/09: "Como o resume será do `model_3100` e não um treino do zero"). O `model_3100(1).pt` foi baixado em 29/09 às 19:27, antes do commit da zero15 (19:41). A run de origem não está registrada. O log lido vai da it 3100 à 4867 (REL-MONO §2).

### Sessão 1
- **Fim da fase 3** (it ~4650): `p_C` 0,05, `s_C` 0,001, nível 2,9, `fell_over` 0,8, `s_B` 0,55, `s_cauda` 0,85 (SPEC-TETO §1, §4).
- **Fase 4** (it 4990): `p_C` 0,70, `s_C` 0,000 (325 its em zero), nível 1,34, `fell_over` 1,3 (pico 2,8), `s_B` 0,56 (SPEC-TETO §1).
  - O balanceador manda ~70% das cadeias para a C. Cada C que falha desce o nível do env.
  - Nas fases 1–3 o `s_C` médio foi de 4–7% (REL-PLATO §1 A5).
- **Platô do BOTAR** (REL-PLATO, 30/09), por sondas de rede:
  - V(alvo) − V(congelado) no BOTAR: +15,1 (4000), +11,4 (4150), +0,1 (4500), +4,7 (4650), −8,7 (5000), −8,9 (5200).
  - V_CAUDA − V_BOTAR caiu de +11 a +15 para +0,9.
  - Altura comandada das palmas na abertura do BOTAR: −10,0 cm (4000), +4,2 cm (5200).
  - O crítico e o ator "esqueceram" o BOTAR entre o 4650 e o 5000.
- **Outras travas** (REL-PLATO):
  - A3: a laje teleporta para dentro da perna em 31–72% das aberturas.
  - A4: o CARREGAR da C é estátua. O rastreio ×3,5 paga 14/s parado, e fechar vale −0,8 a +5,4 contra ficar.
  - A8: o `upright` põe o ótimo do tronco em ~3°, contra 24° da IK do BOTAR.
  - A9: o registrador era cego ao giro (`giro_w` = 0).
  - O fecho do BOTAR não exige ficar de pé.
- **`model_5000` original:** play `5000_c.csv`. A caixa desce 18 cm em 1 s e para 17 cm acima do alvo e 23 cm fora em xy. Nenhuma junta se move por 3,5 s (LOTE-Z18 §C). Pela dinâmica, esse CSV bate com o `model_5200` (REL-PLATO A10).

### Sessão 2
- **R2** (`git:b52215a`): rastreio ×1 só no elo CARREGAR aberto da C. Com ×1, fechar passa a valer +15 a +21.
- **REL-MONO** (workflow de 76 min, 01/10) achou dois defeitos no R2:
  - V1: o R2 paga o desvio da C. A cauda desviada fica em ×3,5 contra ×1 do elo, 10/s a mais. Desvio − avançar: até +22,4.
  - D5: o R2 é um multiplicador inobservável, que o enunciado rejeita no §6.
  - Achado novo (§2.3): a cauda andando herda twist zero por ~2,9 s (17–27% do tempo de cauda).
- **K1b + R1 + K12** (`git:2f83bfd`; SPEC-LOTE; SPEC-K1b):
  - K1b: rastreio ×1 no CARREGAR em regime parado, em toda cadeia; a C desviada fica sempre parada.
  - R1: o dx da laje do BOTAR fica em [0; 0,10]. Toque 39% → 0,6% de pé; 33% → 4,7% na pose da zero17.
  - K12: seis métricas sem peso da cadeia C.
  - Resume do `model_4650` com LR 5e-4 (MEM:g1-limpo-carregar-parado-e-estatua; V-CKPT).
- **Resultado** (its 4650–5180; SPEC-RASO §0):
  - 40–48% das C chegam ao BOTAR, e o fecho é ~0,4% por visita (`s_C` 0,0015).
  - Sondas no `model_5000(1)`: V(alvo) − V(congelado) = −3,1; V_CAUDA − V_BOTAR = −3,6. O ator comanda subir as palmas em todo ponto da descida (+20 cm no alvo).
  - Antes do R4, `s_C` = 0,0038 na it 5520 (REL-APR §2).
- **Checkpoints** (V-CKPT; REL-APR §2):

  | Checkpoint | sha256 (12) | Commit | Play | Veredito do dono |
  |---|---|---|---|---|
  | `model_5000(1)` | `b99947bca7a3` | `2f83bfd` CERTO | menor distância ao alvo 0,20 m, tombo 15–19°; FALHA, caixa cai a 90° | "pega, segura e larga no botar" |
  | `model_5350` | `6d89b4fc2e5c` | `2f83bfd` CERTO | completa na laje 0,55; solta ao trocar para ANDAR (−0,32 m/s); menor distância 0,12 m | — |

### Sessão 3
- **Mudança:** com `s_C` = 0, 50% das lajes do BOTAR nascem em [teto_raso − 0,10; teto_raso], com teto_raso = min(teto físico, 0,64 m). A fração cai a 0 com `s_C` ≥ 0,30.
- **Origem do 0,64 m:** por replay cinemático, a laje de 0,68 m toca o robô de pé em 30% das aberturas (`git:c4835ca`; SPEC-RASO).
- **Resultado:** `s_C` 0,053 na it 5800 (REL-APR §2).
- **`model_5800`** (sha `66bee41a688e`, `c4835ca` CERTO):
  - Completa na laje 0,55: pouso a −0,02 m/s, caixa a 4 cm do alvo, tombo 4° (3,5–7°).
  - Veredito do dono: "pega e bota, mas fica meio agachado, as pernas tocam a laje por baixo, e não bota nas lajes mais baixas".
  - Medido: coxa e canela E na laje em 137 de 150 passos do BOTAR, a ~1 mm (V-CKPT; REL-APR §2).

### Sessão 4
- **Motivo:** com o nível médio em ~1,8, a laje do BOTAR herdava o topo do PEGAR (0,30–0,55 m) e quase nunca ficava abaixo de 0,15 m (veredito do dono, `git:39b8b82`).
- **Mudança:** topo = teto_raso − prof·U, com prof de 0,10 m (`s_C` = 0) até o chão (`s_C` ≥ 0,50). Resolve o D17 (enunciado §2: alturas independentes).
- **Resultado** (REL-APR §2): `s_C` ~0,20 nas its 6318–6320.
  - Funil da C na it 6320 (÷ `c_episodio`): 55% chegam ao CARREGAR, 35% ao BOTAR, 17% terminam com sucesso.
  - Topo médio 0,48 m na chegada e 0,53 m no sucesso.
  - `caixa_largada` subiu de 0,13 (it 5800) para 0,54–0,67, e o log não diz em que elo.
- **`model_6050`:** menor distância 0,08 m, tombo 15–18°. A caixa fica apoiada numa aresta, ou seja, pousa tombada, o que o §4.5 proíbe (REL-APR §2).
- **CARREGAR do 6050** (`6050_carregar.csv`, vx 0,5; REL-APR §4):
  - ~0,35 m/s no rumo do corpo.
  - Gira sem comando: rumo de −22° a −68° em 9 s.
  - Agachado: pelve 0,69–0,70 m, joelho 1,05–1,25.
  - Passo curto: oscilação do joelho ~0,15 rad, contra ~0,5 no ANDAR.
  - Caixa a 1,02–1,03 m, 13 cm acima do alvo.
  - Sem caixa, o mesmo checkpoint anda a 0,74 m/s para 0,8.
- **Agachado antes de pegar** (REL-APR §1, plays `parado:ANDAR:6.0`):

  | Checkpoint | Joelho (rad) | Pelve (m) |
  |---|---|---|
  | keyframe | 0,67 | 0,79 |
  | `model_5000(1)` | 0,85 | 0,73 |
  | `model_5350` | 1,39 | 0,64 |
  | `model_5800` | 1,44–1,55 | 0,61 |

  O enunciado §3 pede joelho ~0,15 e pelve 0,78–0,79. Causa: a espera publica ANDAR com twist zero e caixa zerada, o que o ator não distingue do ANDAR parado. Agachar adianta o fecho ~0,5 s, e fechar 1 s antes compra ~14 de renda. O `pose` paga mais o semi-agachado (1,0) que o de pé (0,893).

### Por que a zero17 terminou
- **Pendências do dono em 01/10** (MEM:g1-limpo-pendencias; SESS-01/10):
  - PEGAR tomba a caixa (38–47° no play do 6050, confundido pelo registrador).
  - CARREGAR "todo estranho".
  - BOTAR em mais alturas.
  - Agachado parado.
  - Funil da C.
- **Teste do 21700:** o dono perguntou "eu não consigo partir de um outro treino só pra ver? tipo o 21700?". O resume do 21700 no mesmo código superou a zero17 em tudo (próxima seção), e a zero17 não foi mais retomada.
- **Leitura do REL-APR §5b:**
  - A linha zero recomeçou ~12 vezes (zero01 a zero17). Ela tem ~6 300 its na zero17, e as outras runs, de 1 a 3 mil cada, foram descartadas.
  - A linha bloco teve 1 recomeço e 21 blocos em resume encadeado (~21 700 its).
  - "A recompensa atual sustenta e refina uma política competente, mas não a produz a partir do zero."

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
