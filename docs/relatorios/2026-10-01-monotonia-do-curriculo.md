# Monotonia das recompensas do currículo — zero17 com o R2

Data: 01/10. Pedido do dono: "a recompensa da tarefa posterior deve sempre ser maior que a
anterior", julgada contra `specs/g1-limpo-comportamento.md`.

Método: workflow só-leitura de 76 min. Três leitores Haiku extraíram a máquina de elo, a
tabela de recompensas e o currículo. Um juiz Opus montou a conta de ficar contra avançar em
cada transição. Um verificador Opus refez as contas para refutar o juiz. Nenhum agente rodou
env, smoke ou treino. As contas são numpy sobre as fórmulas do código, os CSVs dos plays e o
log do TensorBoard da zero17. Scripts, saídas e o resultado bruto em JSON (temporários):
`/tmp/claude-1002/-home-joaobornelli-Documents-g1-training/7edc1d25-9383-4df4-a40a-9e392434fa57/scratchpad/wf_revisao/`
(`juiz/`, `verif/`, `juiz_result.json`, `verif_result.json`).

Regra testada: §3 "Sem estátua" e §3 "Progressão na cadeia". A conta compara o valor de
ficar com o de avançar no horizonte do PPO: γ = 0,99 por passo de 0,02 s, então uma taxa
constante vale 2 s. ΔV(2 s) > 0 quer dizer que avançar paga mais.

## 0. Resposta

1. Com o R2, a progressão principal sobe em cada degrau:

   | Degrau | ΔV(2 s) ou Δ da taxa |
   |---|---|
   | Mão na caixa (PEGAR_SEM → PEGAR_COM) | +15,7 a +18,2 |
   | Fecho do PEGAR, B e R, fase 1 / 2 / 3–4 | +8,0..+25,3 / +11,9..+24,9 / +6,8..+21,0 |
   | Fecho do PEGAR, cadeia C | +4,6 a +12,8 |
   | Fecho do CARREGAR da C | +10,5 a +21,8 (sem o R2: −5,1 a +6,2) |
   | BOTAR: pairar → apoiar | +4,0/s |
   | BOTAR: apoiada → fecho → CAUDA | +11,3/s |

2. A regra falha em dois pontos. O verificador confirmou os dois.
   - **V1.** O R2 paga o desvio da cadeia C. A C que falha o `perto` no fim de uma espera
     vai à cauda CARREGAR, e a cauda paga o rastreio ×3,5. O elo aberto paga ×1. Falhar
     rende mais que avançar.
   - **V2.** No estágio 0 do `command_vel` (|wz| ≤ 0,5, até a it 5000), o giro no lugar
     empata com a estátua. Com o pose ×4 do zero18 1.3 no ANDAR, a estátua vence.
3. Achado novo de código: a cauda CARREGAR que anda herda vx = vy = 0 por ~2,9 s
   (`comando.py:1501-1502`). Nesse trecho, o rastreio ×3,5 paga a estátua.
4. Correção minha: eu propus o R2 sem conferir o caminho de desvio. Também não conferi o
   §6, que rejeita o "multiplicador por posição na cadeia: inobservável pelo crítico". O R2
   está commitado. Ele não deve entrar num resume sem o K1 ou o K1b (§5).

## 1. Transições

Veredito final: o do juiz, corrigido pelo verificador quando ele refez a conta. "—" quer
dizer não reverificado.

| Id | Fase | Cadeia | Ficar → avançar | ΔV(2 s) | Juiz | Verificador |
|---|---|---|---|---|---|---|
| A1 | 1–4 | ANDAR | estátua 3,9–4,4/s → marcha 4,8–5,3/s | +0,8 a +2,8 | OK | — |
| A1z | zero18 1.3 | ANDAR | estátua → marcha, pose ×4 | +0,8 a +3,4 com o braço no default | VIOLA | parcial: RISCO de passo miúdo |
| A2 | 1–4 | ANDAR, giro no lugar | estátua 5,55/s → girador 5,55/s (estágio 0) | −0,5 a +0,4 no estágio 0; +0,5 a +1,1 no estágio 1 | RISCO | confirmado, pior: empate no estágio 0 |
| A2z | zero18 1.1 + 1.3 | ANDAR, giro no lugar | estátua 8,55/s → girador 8,1–8,46/s (estágio 0) | −0,9 a −0,2 | VIOLA | confirmado |
| B1 | 1–4 | B, R, C | ESPERA_SEM 1,8–2,0/s → PEGAR_SEM 4,70–4,86/s | degrau +2,7 a +3,1/s, por temporizador | OK | — |
| B1z | zero18 1.3 | B, R, C | ESPERA_SEM ×8 7,2–9,0/s → PEGAR_SEM 4,70–4,86/s | degrau −2,4 a −4,3/s, por temporizador | RISCO | confirmado com ressalvas |
| B2 | 1–4 | B, R, C | PEGAR_SEM estátua 4,6–4,86/s → mãos na caixa 9,7–13,7/s | +15,7 a +18,2 | OK | — |
| B3 | 1–4 | B, R, C | apertar sem erguer 12,7–13,7/s → caixa no alvo 17,7–22,4/s | +8,0 a +19,4; gradiente 0,2/s por cm | RISCO | confirmado |
| B4-F1 | 1 | B, R | pairar 11,7–18,6/s → fecho → SEGURA ou cauda parada | +8,0 a +25,3 | OK | — |
| B4-F2 | 2 | B, R | pairar → fecho → cauda | +11,9 a +24,9 | OK | — |
| B4-F34 | 3–4 | B, R | pairar → fecho → cauda andando | +6,8 a +21,0 | OK | — |
| B5-F34 | 3–4 | B, R, C desviada | estátua na cauda andando → andar com a caixa | ganho +4,8 a +6,7/s; empata com 14–23% de risco extra de terminar | RISCO | confirmado; o ganho é 2× o do juiz |
| R1-inerte | 1–4 | R | REORIENTAR inerte 10,7/s → espera ~2/s | degrau −8,7/s, por temporizador | OK | — |
| C3 | 1–4 | C | pairar → fecho → elo CARREGAR ×1 | +4,6 a +12,8 | OK | — |
| C3b | 1–4 | C | fim da espera do PEGAR: elo ou desvio | desvio − avançar: fases 1–2 +1,3 a +22,4; fases 3–4 +11,6 a +21,4 (q = 0,1) | VIOLA | confirmado, mais forte |
| C4 | 1–4 | C | elo CARREGAR S_P + 7,0–8,2/s → fecho → BOTAR | +10,5 a +21,8 | OK | — |
| C4b | 1–4 | C | fim da espera do CARREGAR: BOTAR ou desvio | desvio − BOTAR: fases 1–2 −5,6 a +4,0; fases 3–4 −9,9 a +4,2 sem risco na abertura | VIOLA | confirmado; empate também nas fases 3–4 |
| C4c | 1–4 | C | espera → abertura do BOTAR | p de terminar 0,11–0,35 contra p* 0,44; com o R1, p ≤ 0,02 | RISCO | confirmado |
| C5 | 4 | C | BOTAR congelado 36,96/s → caixa no alvo | p* 0,25–0,33; 84% do ganho vem depois do fecho | RISCO | confirmado |
| C5b | 4 | C | pairar 42,17/s → apoiar 46,17/s | +4,0/s | OK | — |
| C6 | 4 | C | apoiada 46,17/s → CAUDA 57,43/s | +11,3/s | OK | — |
| C7 | 4 | C | CAUDA agachado 57,43/s → de pé 62,4–63,2/s | +0,3 a +3,4; vale de −3,7/s ao soltar | RISCO | confirmado com ressalvas |

S_P, S_C e S_B são as rendas congeladas no fecho do PEGAR, do CARREGAR e do BOTAR. q é a
fração das C no elo CARREGAR que chegam ao BOTAR. p* é o risco de terminar que empata a
conta.

## 2. Violações

### V1. Desvio pago da cadeia C (C3b, C4b; D1, D5)

Mecanismo, no código:
1. No fim da espera, o `perto` é reconferido contra o alvo congelado no mundo
   (`comando.py:764-772`). Na espera com a caixa, os sete termos valem 0
   (`knobs.py:1310-1320`): baixar ou afastar a caixa é grátis.
2. Quem falha o `perto` vai à cauda CARREGAR com `fechou = True` (`comando.py:804-839`).
   A cadeia C desviada vai sempre à cauda.
3. O flag do R2 exige `~fechou` (`comando.py:167`). A cauda fica fora do flag e paga o
   rastreio ×3,5 (`knobs.py:1321-1322`). O elo aberto paga ×1.
4. A diferença é 2,5 × 4 = 10/s, até o fim do episódio.
5. Nas fases 3–4 a cauda anda, mas herda twist zero por ~2,9 s (seção 2.3). Nesse trecho
   o desvio paga S_P + 16–19/s.
6. Na F4 da zero17, p_C = 0,67 e s_C = 0,000 (log). A C é a maioria dos episódios de PEGAR.

Números refeitos pelo verificador (`verif/v7_desvio_fase34.py`):

| Ponto de escolha | Desvio − avançar, ΔV(2 s) |
|---|---|
| Fim da espera do PEGAR, fases 1–2 | +1,3 a +22,4 (q de 1 a 0) |
| Fim da espera do PEGAR, fases 3–4 | +11,6 a +21,4 com q = 0,1; −3,1 a +10,9 com q = 1 |
| Fim da espera do CARREGAR, fases 1–2 | −5,6 a +4,0 |
| Fim da espera do CARREGAR, fases 3–4 | −9,9 a +4,2 sem risco na abertura; −2,1 a +30,0 com o teleporte de hoje |

O log não mede q. O juiz estima q entre 2% e 54% na F4.

Enunciado: §3 "Progressão na cadeia"; §3 "Sem estátua"; §4.4 (a caixa fica no alvo de
transporte); §6 (multiplicador por posição na cadeia, inobservável pelo crítico:
rejeitado). Nas fases 1–2, o ator e o crítico veem a mesma observação no elo da C e na
cauda parada de B (`observacoes.py:86-141`). O `PPOPorElo` normaliza a vantagem num grupo
CARREGAR que mistura ×1 e ×3,5 (`algoritmo.py:198-221`).

Conserto: K1 ou K1b (seção 5).

### V2. Giro no lugar no estágio 0 (A2, A2z; D6)

Mecanismo:
1. A zero17 rodou no estágio 0 do `command_vel` em todo o log lido (it 3100–4867):
   vx ≤ 1,0 e |wz| ≤ 0,5.
2. No estágio 0, o giro no lugar sorteia |wz| em U(0,2; 0,5) (`comando.py:2319-2331`).
3. O σ do rastreio angular é max(|cmd_wz|, 0,707) (`recompensas.py:461`). Com |wz| ≤ 0,5,
   o piso morde sempre, e a estátua colhe exp(−wz²/0,5) = 0,61 a 0,92 do termo.
4. A estátua soma 5,55/s (angular 1,55 de 2,0). Um girador realista também soma 5,55/s.
   O ganho bruto de girar é +0,45/s, e o custo de mover o corpo come esse ganho.
5. Com o pose ×4 no ANDAR (zero18 1.3), a estátua leva +3,0/s, e o girador leva
   +3 × (0,85–0,94). A estátua vence por 0,1 a 0,45/s.
6. No estágio 1 (it ≥ 5000, |wz| até 1,6), o σ cresce com o comando, a estátua colhe 0,37
   no topo, e girar vence por +0,5 a +1,1 em 2 s.

Efeito sobre a spec do giro (`specs/g1-limpo-giro-no-lugar.md`, item 1, igual ao zero18
1.1): no estágio 0, subir o giro no lugar para 0,30 dá 21% dos sorteios a um estado onde
ficar parado paga o mesmo que girar.

Enunciado: §3 "Locomoção" (o giro acontece parado, sem deslocamento); §3 "Sem estátua";
§4.1.

Conserto: nenhum K do workflow cobre o estágio 0. Duas opções a medir antes:
- O item 1 da spec do giro só vale depois da it 5000. Um resume de checkpoint ≥ 5000 já
  cumpre isso, porque o `common_step_counter` vai no checkpoint. O R5 (resume do
  `model_4650`) roda 350 its no estágio 0.
- O piso `sigma_min` do angular desce para perto de 0,2. Isso muda todo rastreio
  angular, inclusive a deriva de rumo parado. Medir a derivada antes.

E o K7 (seção 5) para o zero18 1.3.

### 2.3 Achado novo: a cauda andando herda twist zero

- `_zera_twist_nos_parados` escreve vx = vy = 0 nos envs parados (`comando.py:1501-1502`).
  Quando o env passa de parado a andando, nada re-sorteia o twist. Isso acontece na
  entrada da cauda andando: fase ≥ 3, e 10% na fase 2.
- O fabricante só muda vx e vy no próximo re-sorteio, com segmentos de U(3; 8) s (mjlab
  `command_manager.py:109-115`, `velocity_env_cfg.py:183`). O verificador mediu ~2,9 s em
  média, ou 17–27% do tempo de cauda (`verif/v6_twist_herdado.py`).
- Nesse trecho, os envs com laço de rumo (80%, `knobs.py:552`) recebem só o giro de
  correção, e o rastreio ×3,5 paga a estátua.
- Efeitos: o V1 piora nas fases 3–4; a fase 3 pratica menos andar com a caixa. O smoke
  não detecta, porque confere o máximo sobre todos os envs (`smoke.py:1944-1945`).
- Conserto candidato do verificador, não avaliado: re-sortear o twist na entrada da cauda
  andando (K13).

## 3. Riscos

| Id | Risco | Enunciado |
|---|---|---|
| A1z | zero18 1.3 no ANDAR andando: o ×4 cobra primeiro braço e punho, que dão 46–81% do déficit do pose andando. A resposta barata é andar com o braço no default (margem +0,4 a +1,7/s). Sobra risco de passo miúdo | §3 Pés; §4.1 |
| B1z | zero18 1.3 na ESP_SEM ×8: degrau de −2,4 a −4,3/s na abertura do PEGAR, sem escolha; puxa ao keyframe (D7) | §3 De pé; §4.2 |
| B3 | A subida do PEGAR é rasa: 0,2/s por cm. Apertar sem erguer, proibido no §4.3, paga 57–61% do teto | §4.3 |
| B5-F34 | Andar com a caixa: a renda congelada funciona como seguro contra a queda. Andar empata com 14–23% de risco extra de terminar | §4.4; §3 Sem estátua |
| C4c | Abertura do BOTAR: a laje nasce no corpo. O R1 resolve a parte em x. A laje tem quatérnion identidade, então a quina se aproxima com o rumo do robô: borda a 0,09 m com rumo de 30°. O R1 não trata isso | §3 Contato; §4.5 |
| C5 | Descida do BOTAR: vales de 0,47 a 2,5/s nos caminhos usados. 84% do ganho vem depois do fecho, e o crítico inverteu o valor de descer | §4.5 |
| C7 | CAUDA: soltar agachado tem vale de −3,7/s com a multa cheia, e 0 com o limiar de 50 N de hoje. Na CAUDA a caixa some da observação (`comando.py:857`, `observacoes.py:140`), e a terminação dispara com a caixa a mais de 0,18 m do alvo (`terminacoes.py:117-120`). Mexer as mãos é risco às cegas | §4.5; §3 Contato |

## 4. Desalinhamentos com o enunciado

| Id | Estado | Enunciado | O que a recompensa paga | Grav. | Verificador |
|---|---|---|---|---|---|
| D1 | fim da ESPERA_COM; cauda da C desviada | §3 Progressão; §4.4 | o desvio paga mais (V1) | alta | confirmado |
| D2 | abertura do BOTAR | §3 Contato; §4.5 entrada | laje no corpo em 31–72% das aberturas | alta | confirmado; mais a quina com o rumo |
| D3 | BOTAR | §4.5 | descer paga +0,32/s por cm; a palma fora custa ~0,9/s por cm (uma) a 1,6–1,8 (duas) | alta | com ressalvas |
| D4 | BOTAR; PEGAR_SEM com laje baixa | §3 Postura (inclinar quando o alvo exige) | `upright` fora da tabela: ótimo do tronco em ~3° | média | confirmado |
| D5 | elo CARREGAR da C; cauda parada | §6 (multiplicador inobservável rejeitado) | R2: ×1 e ×3,5 com a mesma observação | média | confirmado |
| D6 | ANDAR | §4.1; §5 | `track_lin` com σ fixo 0,5: canal morto acima de ~1,2 m/s | média | parcial: só no estágio 1 |
| D7 | ANDAR parado; ESP_SEM; CAUDA | §3 De pé (joelho ~0,15 rad) | `pose` no keyframe: joelho 0,669, pelve 0,756 m | média | com ressalvas: na CAUDA o `postura_ereta` ×8 puxa a pelve a 0,78 m |
| D8 | todo estado depois de um fecho | §3 Sem estátua; §5 última linha | a renda congelada some na queda, e o crítico não a observa | média | com ressalvas |
| D9 | ESPERA_COM | §4.4; §3 Empurrão | sete termos em 0; o push reprova o `perto` | média | com ressalvas: o push não foi medido |
| D10 | PEGAR e o congelado | §2; §4.3 | `squeeze` satura: aperto de 68,9 N sem preço | média | confirmado |
| D11 | BOTAR | §2; §4.5 | impacto sem preço | média | confirmado |
| D12 | REORIENTAR inerte | §4.6 | `precise_pos` paga 3/s sem trabalho | baixa | — |
| D13 | fecho do BOTAR; sucesso do episódio | §4.5 Fecho; §1 | fecho com as mãos na caixa; o sucesso não exige de pé | média | confirmado |
| D14 | CAUDA | §4.5; §3 Contato | vale ao soltar agachado | média | com ressalvas |
| D15 | freio | §3 Movimento controlado | catraca: freio 2,3 com nível 1,4; −2,46/s, o maior termo negativo | média | confirmado |
| D16 | troca 1 → 2 | §1 | o portão lê só o `s_B` | média | não mordeu na zero17 |
| D17 | abertura do BOTAR | §2 Lajes | a laje herda o topo do PEGAR ±0,10, até ~0,65 m | baixa | — |

## 5. Consertos

Todos precisam de spec e de "implementa". As linhas são estimativa do juiz, no formato
arquivo antes → depois.

| Id | Mudança | Ataca | Linhas | Verificador | Efeito colateral |
|---|---|---|---|---|---|
| K1 | `carregar_elo_aberto` sem o `~fechou`: a C desviada também fica em ×1 | V1 | comando 2479 → 2479; smoke 6243 → 6243 | resolve o prêmio: o BOTAR vence o desvio por 16 a 26 em 2 s | nas fases 3–4 a C desviada anda com ×1, e a cauda de B anda com ×3,5, com a mesma observação: o D5 cresce |
| K1b | rastreio do CARREGAR em ×1 no regime parado (o mesmo `standing` do `PosturaPorElo`) e ×3,5 andando, em toda cadeia; a C desviada fica sempre parada; saem `carregar_elo_aberto` e `env.limpo_carregar_elo` | V1, D5 | saldo ≈ −40 (comando 2479 → ~2460; recompensas 1354 → ~1356; smoke ≈ −20) | ataca a raiz e é observável; só funciona com a C desviada sempre parada | a cauda parada de B perde 10/s nas fases 1–2: o fecho do PEGAR na F2 cai para +4,6..+12,8, ainda positivo; degrau de 10/s em ‖cmd‖ = 0,05, observável |
| K2 | reconferir o `perto` contra o alvo reancorado na base | D9 | comando 2479 → ~2482 | correto; complemento do K1, não substituto | pequeno |
| K3 (R3) | `upright` na tabela por estado, 0 no BOTAR e no PEGAR_SEM | D4, C5 | knobs 1600 → ~1603; smoke +2 | ataca o D4 | o BOTAR perde ~1/s, e o C4b pende ~2 para o desvio; risco de mergulho do tronco |
| K4 | piso do σ_alcance no BOTAR de 0,08 para 0,12–0,15 m | C5, D3 | comando 2479 → ~2482; knobs +2; smoke +1 | ajuda o BOTAR; o "5,7×" da motivação é o caso de duas palmas (uma: 1,5–2,8×) | a mão afrouxa (lição da v3.3); o S_B sobe |
| K5 (R4) | BOTAR raso primeiro | C5, D3 | comando +6; knobs +3; env_cfg +1; smoke +2 | a ideia é certa, mas colide com o §2 | laje a 0,70–0,80 m; a profundidade cresce pelo `s_C`, que está em 0 |
| K6 | portão 1 → 2 também pelo `s_cauda` | D16 | uma condição; smoke +1 | inócuo agora | — |
| K7 | zero18 1.3: o ×8 da ESP_SEM fica; o ANDAR fica em ×1 | V2, A1z | 0 | aceitável | sem o K7, o giro no lugar com ×4 é VIOLA no estágio 0 |
| K8 | σ do `track_lin` proporcional ao comando, com piso 0,5 | D6 | recompensas 1354 → ~1368; env_cfg +2 | só morde no estágio 1 | obriga a decidir o termo vz² |
| K9 | reconciliar o §3 (joelho 0,15) com a decisão D-A2b (pose do molde) | D7 | 0 (decisão do dono) | decisão do dono | o conflito real está no ANDAR parado e na ESP_SEM |
| K10 | freio: medir antes; se pesar, o degrau desce com o nível | D15 | curriculo 537 → 538 | medir antes | afrouxa o preço de segurança |
| K11 | canal da renda congelada no crítico | D8 | observacoes 141 → ~150; env_cfg +4 | atende ao §5 | um append no fim quebra a fatia do one-hot e o assert do `PPOPorElo`: +1–2 linhas |
| K12 (R6) | métricas sem peso por episódio C: entrou no CARREGAR, chegou ao BOTAR, falhou o `perto` em cada espera, terminou nos 10 passos depois da abertura, `fell_over` por elo; mais a fração de twist herdado, o rumo na abertura do BOTAR e o desvio por fase | mede o que decide K1, K2, R1 e K5 | metricas 574 → ~600; comando +4 | correto, sem risco | — |
| R1 | a laje do BOTAR só se afasta: dx ∈ [0; 0,10] (decidido) | C4c, D2 | comando 2479 → 2479 | positivo; também ajuda o C4b (p de 0,11 para 0,02) | a quina com o rumo continua |
| K13 | re-sortear o twist na entrada da cauda andando | seção 2.3; V1 nas fases 3–4; B5-F34 | não estimado | proposto pelo verificador, não avaliado | — |

Itens do plano zero18 (`docs/planos/2026-09-30-plano-zero18.md`):

| Item | Veredito do verificador |
|---|---|
| 1.1 sorteio do twist (standing 0,30, turning 0,30) | RISCO maior que o avaliado: no estágio 0, 21% dos sorteios ensinam que ficar parado basta |
| 1.2 métrica de giro | neutro |
| 1.3 pose ×4 no ANDAR, ×8 na ESP_SEM | ANDAR: RISCO de passo miúdo, e VIOLA no giro no lugar do estágio 0; ESP_SEM: degrau sem escolha. Usar o K7 |
| 1.4 contato de 5 a 30 N | RISCO: o vale da CAUDA morde a partir de 5 N; só o elo da C paga o toque, porque o desvio manda a laje a +5 m, e isso piora o V1; punho e antebraço não têm sensor e ficam mais baratos que a palma (`cena.py:133-143`) |
| 1.5 teto 0,30 do `p_C` | neutro na monotonia; corta a prática do BOTAR a 0,45×; reduz a fração de C, onde vive o V1 |
| 1.6 postura de pé com caixa (IK) | OK no PEGAR e na cauda de B; o desvio ganha a forma (+1,2 a +1,8/s), e o C4b pende mais 2,4 a 3,6 para o desvio |
| 1.7 métrica `caixa_no_carregar` | neutro; filtrar por cadeia 2 ∧ CARREGAR ∧ ¬fechou |
| 1.8 registrador | neutro; ainda grava com `giro_w` = 0 (A9 do relatório do platô) |

## 6. Ordem sugerida pelo verificador

1. K12, com a fração de twist herdado.
2. K1b, ou o K1 junto com o R2. Antes de qualquer resume com o R2.
3. R1.
4. K13, depois de avaliar.
5. K2.
6. K7 no zero18.
7. K4, K3 e K5 para o BOTAR.

## 7. Limites

- Nada rodou dinâmica. As taxas vêm de fórmula, de replay cinemático e do log. O JSON do
  juiz marca cada número como MEDIDO, INFERIDO ou CÓDIGO.
- O log não mede q. O K12 mede.
- O script do juiz não reproduz dois números do texto dele (A1z, A2z). Este relatório usa
  os números refeitos pelo verificador.
- O vale do C7 depende de uma força de contato que ninguém mediu.
- O efeito do push sobre o `perto` (D9) não foi medido.
