---
tipo: abordagem
pai: "[[C3 g1_limpo]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 14/09–01/10/26
resumo: "~12 recomeços do zero (zero01–zero17), cada um com todas as restrições desde a it 0; nenhum produziu política competente; a zero17 trouxe o currículo de cadeia."
---
## Resumo em tabela (zero01–zero08)
| run | datas | hospedeiro | mudança-chave antes dele | fim | causa do fim |
|---|---|---|---|---|---|
| zero01 (1ª) | 14/09 | Colab, branch `exp/g1-limpo-v3` | `amax` no regime `standing` do freio | it 113 | colapso: morrer virou ótimo |
| zero01 (2ª, elíptica) | 17/09 manhã | Colab, 8192 envs | cone elíptico + impratio 10 | it ~755 | NaN no `check_nan` do ator |
| zero01 (3ª, "piramidal_impratio_2") | 17/09 tarde → ? | Colab, 8192 envs | pirâmide + impratio 2 | ≥ it 392 (desconhecido) | desconhecido |
| zero02 | 20/09 → 22/09 | Kaggle, 4096 envs | cintura em família própria + `NoBatente` | ~it 11 191 | troca do conjunto de recompensa (21–22/09) |
| zero03 | 22/09 | Colab, 8192 envs | freio −15 + regime pelo estado + `FACE_DE_PE` + σ pela tolerância | it 451 | `batente_da_cintura` dominante |
| zero04 | 22/09 | Colab | rampa da cintura de 0,95 para 0,90 | it 650 | `batente_da_cintura` dominante |
| zero05 | 22/09 → 23/09 | Colab e Kaggle | `NoBatente` sai | ? | freio −15 comprime o std |
| zero06 | 23/09 | Colab e Kaggle | freio −6 só no notebook | ? | substituída pelo freio em currículo |
| zero07 | 23/09 → 24/09 | Kaggle (`RETOMA = False`) | freio = soma das dobradiças + degrau por env | ~it 3150 | pernas comprimidas, travou na pega |
| zero08 | 24/09 → ~28/09 | Colab e Kaggle | freio zero no estado ANDAR | retomada como zero09 (28/09) | fechos sobem e somem (tombo da caixa) |

## Linha do tempo (zero08–refino)
| Data | Run / evento | Commit | Mudança central | Por que terminou |
|---|---|---|---|---|
| 24–27/09 | zero08, zero09 (antecedentes) | `6ab0258` | freio zero no ANDAR | caixa a 66° na pega (zero08), ANG cego ao tombo (zero09) |
| 28/09 | zero10 | `ff5ac85`, `04e99ca` | `precise_ori` híbrido; ANG pelo eixo de cima | erguia tombando com o cotovelo (17° → 60°) |
| 28/09 | zero11 (retomada da zero10, ~it 1300) | `829ee03` | `precise_ori` ×4 no PEGAR_COM | fechos aparecem e somem; pairar paga mais que fechar |
| 29/09 | zero12 (do zero) | `7d4be42` | S1–S5: renda pós-fecho, `_alinha`, σ fixo | pega parada na mesa (model_1750); estagnou na it 2629; caixa deitada a 90° no play da it ~3100 |
| 29/09 | zero13 | `7a23ea9` | `_alinha(aproximacao=True)` só a reta (caminho B) | sem registro de treino (lacuna) |
| 29/09 | zero14 | `2b35d61` | `unload × _trazer`; `pose` pelo estado | erguia, mas a caixa deitava 90° |
| 29/09 | zero15 | `ce3d460` | reta zera a 90°; âncora 0,75–0,85; `~apoiada` | PEGAR reprova no tombo (25–29°) e no de pé; âncora revertida |
| 30/09 | zero16 | `b2909e9` | reverte a âncora; a reta a 90° fica | substituída pela zero17 no mesmo dia (2 h depois) |
| 30/09–01/10 | zero17 (resume do `model_3100`) | `c0352b0`, `a96039a`, `b919e32`, depois `b52215a`, `2f83bfd`, `c4835ca`, `39b8b82` | currículo de cadeia por fases; one-hot fora da normalização; C com CARREGAR; sucesso = fecho ∧ time_out; R2, K1b, R1, K12, R4, R4b | BOTAR resolvido por R4/R4b (s_C 0,20), mas agachado, CARREGAR ruim, quedas; o dono testou o 21700 no mesmo código |
| 30/09 | zero18 | — | plano de resume do 4650 | nunca implementado; trocado pelo lote K1b + R1 + K12 |
| 01–02/10 | Refino v2: `21700_zero_teste` | `39b8b82` | resume do `model_21700` (linha bloco) com a config da zero17 | Colab encerrou por recursos na it 23253; `model_23250` = MELHOR |
| 03/10 | continuação 23250 → ~24393 | `39b8b82` | mesma config | freio em catraca (1,75 → 3,0) derruba a cadeia |
| 05/10 | resume 24350 + freio de aperto −0,05 + DR μ | `0912915` (clone `025a9d6`) | `priority`, DR μ [0,6; 1,0], `aperto_excessivo` | força total parada em ~195–200 N; nível e s_C caem desde 24,6k |
| 05/10 | resume 24350 + −0,15 + forma ponderada | `79b6afc`, `af97a5a` | preço 3×; `forma_postural` ponderada | parado pelo dono: "não tava aprendendo nada"; tema vai para a v3 |

## Por que a linha zero começou
**1. As correções do `model_14000` foram desenhadas "para o treino do zero".** O plano de
14/09 (`docs/planos/2026-09-14-correcoes-para-o-treino-do-zero.md`, em `[51cc0ac]`; o arquivo
só existe no ramo `exp/g1-limpo-v3`) lista três defeitos medidos no `model_14000` da linha
bloco: juntas rápidas demais na manipulação, o robô se apoia na caixa para pousar, mãos
tortas no BOTAR. Sete mudanças (M1–M7): `amax` no freio, `faixa_de_pose.perna` do BOTAR
1,3→0, `apoiada` como faixa (teto `peso + 30 N`), `& ~apoiada` no fecho do PEGAR, `upright`
×4 e `faixa_de_pose` ×2 na coluna BOTAR, alvo da pega sorteado em 0,75–1,0
`[7d481d8] [f1e4e22] [f7ae0f6]`.

**2. A sonda em warm-start não reaprendeu.** A sonda `bloco20` (2000 its sobre o
`model_13999`, alvo pinado em 1,02) deu `s_B` 0,62→0,86, mas `s_C` 0,484→**0,002** na it
14391. Hipótese do dono, registrada no handoff: as correções juntas tornaram inviável a única
estratégia de pouso que a política conhecia (escorar), e o reaprendizado em 2000 its de
warm-start não acontece `[handoff 14/09 §Task 3]`.

**3. Hipótese dos hábitos herdados do resume.** Em 18/09 o dono reprova só o BOTAR do
`model_21700` (tronco 65°, braço em 55% do alcance, `hip_yaw` −34°) e escreve que esses
comportamentos "podem ser derivados do resume e não ocorram no treinamento do zero"
(`vault/.../g1-limpo-botar-veredito-do-dono-21700.md`). Em 20/09 o commit da cintura diz que
"o run do zero é o teste de se o preço sozinho impede o hábito da perna" `[d4e5e4e]`; o status
de 20/09 põe como pergunta da zero02 "a torção da perna nasce quando o preço existe desde a
primeira iteração?".

**4. A diferença estrutural em relação à linha bloco.** A linha bloco é UM treino do zero
(bloco1) e depois resumes encadeados, cada um mudando a recompensa sobre uma política que já
andava e pegava. A linha zero recomeça com pesos aleatórios a cada mudança do conjunto de
recompensa, e enfrenta todas as restrições desde a it 0. O relatório de 01/10 conta ~12
recomeços (zero01 a zero17) entre 14/09 e 01/10, de 1 a 3 mil its cada, descartados
(`docs/relatorios/2026-10-01-aprendizados-zero17-para-v3.md` §5b).

**5. Separação de pipeline (14/09).** Notebook oficial clona a TAG `estavel-bloco17`; sonda e
zero clonam `exp/g1-limpo-v3`; o zero do Colab grava em `MyDrive/g1_limpo_zero` e o da Kaggle
no dataset `g1-limpo-zero`, porque o resume pega o `model_*.pt` de maior número por `rglob`
e um zero acima da it 14000 seria escolhido no lugar do checkpoint bom `[96be97a]`.

#### Fato de topologia que muda a leitura (verificado no git)

Os commits de 14/09 `7d481d8 → 96be97a` (amax, `apoiada` em faixa, `~apoiada` no PEGAR,
`upright` ×4, `faixa_de_pose` ×2, alvo 0,75–1,0) vivem SÓ no ramo `exp/g1-limpo-v3`
(`git branch --contains 96be97a`). O `[9b6cc8f]` (bloco 18, 15/09) tem como pai `d74b93f`
(a tag `estavel-bloco17`), e não `96be97a`. No `exp/g1-limpo-v2`, de onde saem a zero01 de
17/09 em diante, NÃO existem `folga_apoiada_N`, `~apoiada` no PEGAR nem `excesso.amax`;
o alvo fica em 0,85–0,95 (centro 0,90). Conferido com `git show` e
`git merge-base --is-ancestor 96be97a 6ab0258` (falso). A memória
`g1-limpo-reducao-e-limiar-andam-juntos` registra "AMAX REMOVIDO (verificado 2026-09-15)",
o que bate com isso.

## O que entrou no código entre as runs
| data | commit | mudança | números |
|---|---|---|---|
| 15/09 | `9b6cc8f` | alvo do PEGAR sorteado 0,85–0,95; inércia real da caixa (massa 1,04–4,18 kg, `I = (2/3)·m·a²`); `limite_de_junta` exponencial no lugar do `dof_pos_limits` | a reta do fabricante cobrava 0,055/s com o `waist_pitch` no batente; tabelas `k20/teto0,15` e `k12/teto0,25` reprovaram (95,7% e 59,1% dos passos no teto); `carrega085.csv`: andar 3,78/s, pegar 22,31/s, botar 47,10/s; NO ZERO a rampa é 0,85–1,00 em todas as famílias |
| 16/09 | `c1a5266` | `limite_de_pelve` no CARREGAR, `(relu(0,74 − z)/0,10)²`, peso −1 | custa 3,10/s, 18% da renda parada de 16,8/s; pelve 0,564 andando com caixa contra 0,726 vazia |
| 16/09 | `618373f` | notebook do zero (Kaggle): sem resume; rampa do limite forçada a (20; 0,15) nas 14 famílias | 5478 its em 10,5 h a 6,9 s/it, medido no bloco19 (4096 envs × 24 passos a 14 265 passos/s); estágio 1 do currículo de comando na it 5000 |
| 16/09 | `46dea30`, `cc3b511` | `forma_postural` contra a referência de IK (`ik/ref_botar.npz`); rampa do tronco a 90° (erro chega a 77) | — |
| 17/09 | `3a4701c` | laço de rumo: `wz = k·(rumo_ref − rumo)` nos parados; `rel_heading_envs` 0,8 | `model_19300` girava −0,38 rad/s na pega (89° em 7 s) |
| 17/09 | `a37300c` | limiar por família; `hip_yaw` com rampa a partir de 40° | no zero: `hip_yaw` (20; 0,15; 0,25), rampa de 40° a 64° |
| 18/09 | `b226853` → `7b1b5ac` | clip da ação do `hip_yaw` em ±0,7 rad, REVERTIDO no mesmo dia | — |
| 20/09 | `8cb68cc` | chave da referência de IK no PEGAR = altura da CAIXA, e não o alvo | antes o clamp devolvia sempre a pose da laje 0,55; piso da estátua 4,894→4,882/s |

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.

## Filhos
<!-- filhos:inicio -->
- **zero01** (abordagem · rejeitada · 14–17/09/26) — Três lançamentos com o mesmo nome: colapso pelo amax (it 113), NaN do cone elíptico (it ~755) e pirâmide com impratio 2 sem fim registrado.
- **zero02** (abordagem · parada · 20–22/09/26) — Cintura em família própria e NoBatente, na Kaggle: chegou à it ~11191 com s_B 0,32; a caixa tomba na descida; trocada quando a recompensa mudou.
- **zero03** (abordagem · rejeitada · 22/09/26) — Freio −15 e regime pelo estado: o batente da cintura domina (46,6 das terminações); rampa mais estreita que o ruído não dá gradiente.
- **zero04** (abordagem · rejeitada · 22/09/26) — Rampa da cintura a 0,90: morreu na it 650; terminação por junta num treino do zero vira a saída barata.
- **zero05** (abordagem · rejeitada · 22–23/09/26) — Sem NoBatente: o freio −15 custava −14,8/s com o std perto de 1 e comprimiu o std a 0,45; freio forte corta a exploração.
- **zero06** (abordagem · parada · 23/09/26) — Freio −6 só no zero: a marcha avança, mas na pega a pressa volta a compensar; nenhum peso fixo serve.
- **zero07** (abordagem · rejeitada · 23–24/09/26) — Freio em currículo por degrau de competência: pernas comprimidas e travou na pega com a pelve a 59°.
- **zero08** (abordagem · rejeitada · 24–28/09/26) — Freio zero no ANDAR: fechos até s_B 0,115 e depois a caixa tomba a 78° na pega.
- **zero09** (abordagem · rejeitada · 27–28/09/26) — Retomada da zero08 com precise_ori híbrido: tombo real p50 93° contra ANG publicado 62°; a caixa gira com os antebraços.
- **zero10** (abordagem · rejeitada · 28/09/26) — precise_ori híbrido e ANG pelo eixo de cima: erguia a caixa tombando com o cotovelo (17° → 60°).
- **zero11** (abordagem · rejeitada · 28/09/26) — precise_ori ×4 no PEGAR_COM: fechos aparecem e somem; fechar paga menos que pairar e o crítico foge do fecho.
- **zero12** (abordagem · rejeitada · 29/09/26) — S1–S5 (renda pós-fecho, _alinha, σ fixo), do zero: a pega parou na mesa e depois deitou a caixa a 90°.
- **zero13** (abordagem · parada · 29/09/26) — Caminho B, staged e precise_pos veem o tombo só pela reta; sem registro de treino.
- **zero14** (abordagem · rejeitada · 29/09/26) — unload × _trazer e pose pelo estado; nasce o enunciado do comportamento; ergue, mas a caixa deita 90°.
- **zero15** (abordagem · rejeitada · 29/09/26) — Reta zera a 90° e âncora baixa 0,75–0,85: fecho forçado do PEGAR mata 28 de 28 episódios; o CARREGAR nunca foi praticado.
- **zero16** (abordagem · parada · 30/09/26) — Reverte a âncora; play do 3300 mostra pelve 0,717 e tronco 22° pela forma presa na laje 0,55; substituída 2 h depois.
- **zero17** (abordagem · validada · 30/09–01/10/26) — Currículo de cadeia por fases, one-hot fora da normalização e R4/R4b: s_C 0,004 → 0,20 e BOTAR resolvido, mas agachado, CARREGAR ruim e quedas; o 21700 no mesmo código superou.
- **zero18** (abordagem · rejeitada · 30/09/26) — Plano de resume do 4650 (sorteio de twist, pose, contato com a laje, teto do p_C); nunca implementado, trocado pelo lote K1b + R1 + K12.
- **Achados soltos da linha zero** (achado · ativa · 14/09–06/10/26) — Achados datados: escorar compra o fecho, alvo 1,02 circular, curso largo do hip_yaw, deriva de rumo, caixa_na_pega diluída, maior μ do par, NaN do cone.
- **Lacunas da linha zero** (pendencia · aberta · 06/10/26) — O que as fontes não confirmam: fim da zero01 piramidal, zero02 × NoBatente, model_9100, linhagem do model_3100 e do 3450.
<!-- filhos:fim -->
