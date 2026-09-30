# Currículo de cadeia — spec v2

Data: 2026-09-30. Estado: APROVADA pelo dono no conceito ("tarefas novas só abrem quando a
anterior está dominada, da mesma maneira que no andar"). Implementação: RUN zero17, do zero.
Referência de comportamento: `specs/g1-limpo-comportamento.md`.

## 0. Resumo

O treino do zero não aprende a fechar o PEGAR, porque fechar mata o episódio: depois do
fecho vem o CARREGAR, a política nunca o praticou, e ela solta a caixa em menos de 2 s.
Esta spec faz a cadeia abrir por fases, como a locomoção já abre antes da manipulação:
aprende a pegar, depois a carregar parado, depois a carregar andando, depois a botar. Cada
fase muda FRAÇÕES, e não liga nem desliga tarefa: a tarefa ainda não aberta fica com uma
fração baixa ativa, como o REORIENTAR, para ter prática desde cedo e para o seu slot do
one-hot acender. A troca de fase é automática, pela taxa de sucesso, e o mesmo pipeline
roda igual em outro simulador.

A spec também tira o one-hot da normalização empírica da observação. Um slot que nunca
acendeu tem desvio ~0,0006, e ao acender a rede recebe ~94 no lugar de 1.

## 1. O problema, medido

Ambiente de treino real (mjlab, CPU), `model_3450` da zero15, 29 e 30/09.

| Medida | Valor |
|---|---|
| Rollout sem update, 48 envs: episódios de manipulação no PEGAR com a caixa | 34 de 36 |
| Idem: caixa erguida mais de 5 cm | 34 de 36 |
| Idem: algum elo fechou | 0 de 36 |
| Idem: episódios que chegam ao `time_out` | 45 de 48 |
| Fecho forçado do PEGAR, caixa erguida a < 12 cm do alvo: terminou em ≤ 1,6 s | 28 de 28 (24 `caixa_largada`, 4 `fell_over`) |
| Fecho forçado com o one-hot fora da normalização | 11 de 11 terminaram, em 0,6 a 1,9 s |
| Fecho forçado com o one-hot fora da normalização e a laje mantida | dos 8 que entraram no CARREGAR, 7 terminaram em ≤ 1,5 s |
| `s_B` na zero15 | 0,006 na it 3147 → 0,0012 na 3347 → 0 na 3484 |

**Causa da queda.** Nem a escala do one-hot nem a saída da laje explicam a queda: com as
duas neutralizadas, a política ainda solta a caixa logo depois que o one-hot vira
CARREGAR, inclusive com o comando de andar em zero. Na troca, o slot do PEGAR apaga, e a
política só conhece esse estado pelo ANDAR, em que os braços ficam ao lado do corpo. Ela
nunca praticou o CARREGAR com a caixa, porque nunca fechou o PEGAR. O único conserto é
prática.

**Por que o robô não fecha.** Para o crítico, fechar rende ~1 s de renda e depois zero;
pairar sem fechar rende até o fim. O PPO escolhe pairar. O `s_B` subiu logo depois da
zero15 e voltou a zero: os poucos fechos que aconteceram foram punidos pela queda.

**Risco latente da normalização.** O `rsl_rl` normaliza cada canal por
`(x − média) / (desvio + 0,01)`, com estatística acumulada o treino todo:

| Slot do one-hot do ator | Desvio no `model_3450` | Entrada da rede com o slot aceso |
|---|---|---|
| ANDAR | 0,4986 | ~1 |
| REORIENTAR | 0,0302 | ~25 |
| PEGAR | 0,4985 | ~1 |
| CARREGAR | 0,00055 | ~94 |
| BOTAR | 0,0012 | ~89 |

O crítico tem o mesmo defeito no one-hot publicado e no `elo_interno`. No `model_21700`,
da linhagem dos blocos, os slots CARREGAR e BOTAR têm desvio saudável. Cada troca de
fase acende um slot pouco visto, então a normalização do one-hot tem de sair antes.

**Portões do fecho onde o robô paira** (play da zero15, alvo a 0,80 m): perto 100%, fora
do apoio 100%, tombo ≤ 25° 0% (medido 25° a 29°), de pé ≤ 0,69 rad 0% (`hip_pitch` a
0,96 rad). Os dois que reprovam estão perto do limite, e fechos acontecem por acaso. O que
falta é o fecho valer a pena.

## 2. Por que a linhagem dos blocos aprendeu

| Data | Evento |
|---|---|
| 03/09 | bloco8 treina do zero com a observação v2 |
| 04/09 | bloco9 fecha o PEGAR pela 1ª vez na it 2616. O fecho era TERMINAL: o robô seguia segurando no mesmo alvo, e fechar pagava ~25/s contra ~12,5/s |
| 08/09 | A cauda do CARREGAR entra (`1cab2a0`) numa política que já fechava |
| 29/09 | O estado "fechou e ficou no PEGAR" pagava 32,3/s contra 20,7/s do BOTAR (`model_1600`), e saiu (S2) |

A cadeia de hoje é a dos blocos 12 a 18. O que mudou é o ponto de partida: aqueles blocos
herdaram uma política que já fechava. O treino do zero encontra a cauda desde a it 0.

## 3. Objetivo e fora de escopo

**Objetivo.** Um pipeline do zero que aprende, em ordem, PEGAR, CARREGAR parado, CARREGAR
andando e BOTAR, sem intervenção manual, e que roda igual em outro simulador.

**Fora de escopo:** os portões do fecho (§8, D2 e D3); o REORIENTAR, que segue inerte;
qualquer termo de recompensa novo. A cadeia C ganhou o CARREGAR em 30/09 (§4).

## 4. As fases

A fase é UMA por run. Ela só avança. A fase 4 é o comportamento de hoje, e é o default de
`inspecao` e `play`. O `fase_inicial` é piso: o `play` de um checkpoint salvo nas fases 1
a 3 roda a fase 4.

| | Fase 1 — PEGAR | Fase 2 — CARREGAR parado | Fase 3 — CARREGAR andando | Fase 4 — BOTAR |
|---|---|---|---|---|
| Fecho do PEGAR em B e R, depois da espera | 90% SEGURA: o elo fica PEGAR, fechado, até o fim; 10% vão à cauda CARREGAR | 100% cauda CARREGAR | 100% cauda CARREGAR | como hoje |
| Comando de andar na cauda CARREGAR | zero | zero em 90%, o do fabricante em 10% | o do fabricante | como hoje |
| `p_C`, cadeia C | 0,05 | 0,05 | 0,05 | o balanceador de hoje, piso 0,20 |

**Quem SEGURA na fase 1.** O elo fica PEGAR com `fechou`. O one-hot publicado é PEGAR, o
estado de recompensa é PEGAR_COM e os termos ao vivo seguem pagando, mais a renda
congelada: é o estado medido de ~32/s, contra ~17 a 25/s de pairar. O σ, o alvo e o eixo
de cima NÃO são recapturados: a tarefa não reabre, ela continua. A laje sai, como hoje, e
soltar a caixa continua terminando o episódio. A fase 1 ensina a fechar e a segurar parado.
A cadeia C desviada, que falhou o `perto` no fim da espera, vai sempre à cauda, como hoje.

**Os 10% da cauda na fase 1**, e os 10% andando na fase 2, são a fração baixa ativa. Eles
dão prática da tarefa seguinte desde cedo e mantêm o slot do CARREGAR aceso. A escolha é
por env, uma vez, na entrada da cauda.

**A cauda parada** usa o laço de rumo dos elos parados: o CARREGAR com comando zero conta
como parado para o twist, para o freio e para o `pose`.

**A cadeia C tem o CARREGAR** (30/09, achado 3 da auditoria, enunciado §1): C = PEGAR →
CARREGAR parado → BOTAR. O CARREGAR da C é um elo: abre no fim da espera do PEGAR, com a
caixa no alvo de transporte e comando de andar zero, e fecha pela régua do PEGAR (caixa
no alvo, nivelada, robô de pé, 0,5 s). A laje fica; o BOTAR a reposiciona ao abrir. Quem
falha o `perto` no fim da espera do PEGAR ou do CARREGAR vai à cauda, como antes. O
`p_C` continua em 0,05 até a fase 4, e o CARREGAR da C é parado em todas as fases.

## 5. Métricas de troca e persistência

| Métrica | Definição |
|---|---|
| `s_B` | EMA, por iteração, da fração de episódios B que concluíram E chegaram ao fim pelo `time_out` (30/09: fecho e sobrevivência). Já existia, só com o fecho |
| `s_cauda` | NOVA. EMA, por iteração, da fração de episódios que ACABARAM no CARREGAR e acabaram por `time_out`, e não por terminação. Zera em toda troca de fase |
| `fase_cadeia` | NOVA. 1, 2, 3 ou 4 |
| `iter_fase` | NOVA. O `iters_balanco` do instante da última troca |

| Troca | Condição |
|---|---|
| 1 → 2 | `s_B ≥ 0,50` e ≥ 300 iterações na fase |
| 2 → 3 | `s_cauda ≥ 0,60` e ≥ 200 iterações na fase |
| 3 → 4 | `s_cauda ≥ 0,60` e ≥ 300 iterações na fase |

**Por que esses números.** Com `s_B = 0,5`, metade dos episódios B já fecha e pratica o
pós-fecho quando a cauda vira regra; a bloco9 tinha sucesso 0,36 a 0,57 quando a cauda
entrou. `s_cauda ≥ 0,6` exige que a maioria das caudas sobreviva antes de somar o próximo
desafio. A EMA tem α = 0,05, com memória de ~20 iterações; o mínimo de iterações impede
trocar numa oscilação. A fração de 10% mantém o fecho lucrativo na fase 1:
0,9 × ~32/s contra ~17 a 25/s de pairar. O `p_C` de 0,05 dá prática rara de BOTAR.

**Fim de episódio.** O `mjlab` grava `env.reset_time_outs` antes do reset, e o comando
reinicia antes das terminações (`manager_based_rl_env.py:438, 581, 587`). O
`_atualiza_balanceador` e o `nivel` leem o `time_out` do episódio que acabou.

**O sucesso é fecho e sobrevivência** (30/09, decisão do dono): `concluiu_ate_o_fim` =
`concluiu ∧ time_out` move o nível e as EMAs `s_B`/`s_C`. Um fecho seguido da queda da
caixa ou do robô não conta. Isso substitui a "limitação declarada" da spec dois-bits §2.5.
Consequência: na fase 1 o `s_B` só sobe quando o robô fecha E segura até o fim do
episódio, e o nível só sobe com o episódio inteiro. `metrics["sucesso"]` segue no fecho.

**`s_B ≥ 0,50` e o passeio de nível.** O passeio ±1 equilibra a taxa de sucesso perto de
0,5 por construção. O `s_B` fica abaixo enquanto envs falham no nível 0 e perto de 0,5
depois; o instante da troca 1 → 2 tem ruído. O dono manteve o portão (30/09); o nível
médio (`Curriculum/nivel`) é o medidor de competência a acompanhar.

**Persistência.** `fase_cadeia`, `iter_fase` e `s_cauda` entram em
`runner.CHAVES_ESCALARES`. Um resume retoma na fase certa.

**Log.** `Curriculum/forma/fase_cadeia` e `Curriculum/forma/s_cauda`, ao lado do `s_B`.

## 6. O one-hot fora da normalização

No `PPOPorElo`, os canais do one-hot ficam com média 0 e desvio 1, fixos, no ator e no
crítico: o publicado do ator, o publicado do crítico e o `elo_interno` do crítico. A
fixação roda antes de cada `act` e depois de cada atualização da normalização. Um one-hot
é categórico: padronizá-lo só amplifica o slot raro.

**Resume de checkpoint anterior.** Fixar a normalização muda a entrada de uma rede que
treinou com o one-hot normalizado. No `model_3450`, a 1ª camada do ator desloca 1,5 a 1,8
no ANDAR e no PEGAR, e 38 no REORIENTAR. O `load` do `PPOPorElo` DOBRA a normalização
antiga na 1ª camada, e a rede fica idêntica nos slots com desvio ≥ 0,01. Nos slots que
nunca acenderam (o CARREGAR e o BOTAR da linhagem zero), a escala volta a 1. Num
checkpoint já fixado, a dobra é a identidade. O `runner.load` de um checkpoint sem
`iter_fase` começa a fase 1 no resume.

## 7. Riscos

1. **Correria.** A renda depois do fecho é paga por segundo, então fechar mais cedo compra
   mais renda. O freio (`velocidade_por_regime`) limita a pressa e fica ligado desde a
   fase 1. Sinais: `mao_na_pega`, contra a meta de 0,25 m/s, e `velocidade_de_junta`.
2. **Queda na troca 1 → 2.** O estado de ~32/s some. Espere queda de retorno e pico de
   value loss na troca.
3. **Portões inalcançáveis.** Se `s_B < 0,05` depois de 1500 iterações na fase 1, os
   portões do fecho são a trava. Aí vale a decisão D2.
4. **Canal inobservável.** O crítico não vê se o PEGAR já fechou; o estado fechado e o
   não fechado leem igual. A bloco9 aprendeu nessa mesma condição.
5. **Salto do `p_C` na troca 3 → 4.** O `p_C` sai de 0,05 para o do balanceador, que chega
   a 0,80 com `s_C` baixo, e a prática de B cai ao piso de 0,20. Se `s_B` cair, o
   balanceador devolve prática a B.
6. **A laje sai no fim da espera.** Na sonda, 6 de 19 SEGURAS largam a caixa nos
   primeiros 4 s. O robô escora na laje, e a laje sai. Fechar ainda vale mais que pairar.
7. **A orientação da caixa no CARREGAR.** RESOLVIDO em 30/09: o `precise_ori` vale 4 na
   coluna CARREGAR, o peso do PEGAR_COM. Antes, o único canal contra o tombo ali era a
   reta do `precise_pos`, e a orientação passa a pagar em toda tarefa de manipulação.

## 8. Decisões do dono

- **D1.** Limiares, mínimos e frações das §4 e §5: aprovados como proposta; ajustáveis
  em `knobs.Cadeia`.
- **D2.** O portão "de pé" no fecho do PEGAR: fica. O enunciado não o pede; rever se o
  risco 3 acontecer.
- **D3.** O portão de tombo de 25°: fica. O enunciado pede a caixa nivelada.
- **D4.** Fase 2 parada antes da 3: fica.
- **D5.** Fases só avançam.

## 9. Implementação

Sem termo de recompensa novo. Tudo reaproveita peças que já existem.

| Arquivo | Onde | O que muda |
|---|---|---|
| `knobs.py` | `Cadeia` | `fase_inicial`, frações, `p_c_antes_do_botar`, limiares e mínimos, com a justificativa |
| `comando.py` | cfg do `AlvoCaixaCmd` | os mesmos campos, repassados do knob |
| `comando.py` | `_aplica_espera`, ramo da cauda | na fase 1, 90% SEGURA (sem virar CARREGAR, sem reabrir σ, alvo e eixo); a cauda marca se é parada |
| `comando.py` | `_zera_twist_nos_parados` | a cauda parada conta como parado |
| `comando.py` | `_resolve_p_c` | `p_C = p_c_antes_do_botar` antes da fase 4 |
| `comando.py` | `_atualiza_balanceador` | `s_cauda` e a troca de fase, na mesma borda de iteração do `s_B` |
| `comando.py` | `CADEIAS`, `_fecha_elo_corrente`, `_sustain_alvo_de`, `_aplica_espera` | C = (PEGAR, CARREGAR, BOTAR); o CARREGAR da C fecha pela régua do PEGAR e é parado; o guarda do avanço é `_sigma_pendente` |
| `comando.py`, `curriculo.py` | `concluiu_ate_o_fim`, `nivel` | o sucesso é fecho e `time_out` |
| `curriculo.py` | `garante_forma` e o log da forma | as chaves novas e seus valores no log |
| `runner.py` | `CHAVES_ESCALARES` e `load` | as três chaves novas; `iter_fase` no resume de checkpoint antigo |
| `algoritmo.py` | `PPOPorElo` | o one-hot fora da normalização, e a dobra no `load` (§6) |
| `env_cfg.py` | `make_env_cfg` | repasse dos knobs; `inspecao` e `play` usam a fase 4 |
| `smoke.py` | checagens novas | §10 |
| notebooks | RUN zero17, do zero | assert novo |

**Justificativa de ser aditiva** (regra do `CLAUDE.md`): a mudança troca uma intervenção
manual que aconteceu uma vez na história por uma regra automática, e conserta um defeito
de escala da normalização. Nenhum termo existente sai, porque o problema não é de
recompensa: é a ordem em que os elos ficam disponíveis.

## 10. Verificação

**Sondas na CPU, sem update** (roda o orquestrador):
1. Fase 1: fecho forçado com a caixa erguida. Critério: a maioria dos que SEGURAM
   sobrevive 4 s, contra 0 de 28 hoje.
2. Troca de fase com EMA sintética: 1 → 2 → 3 → 4 nos limiares, nunca volta, e
   `s_cauda` zera na troca.
3. `save` e `load` preservam `fase_cadeia`, `iter_fase` e `s_cauda`.
4. A normalização do one-hot fica em média 0 e desvio 1 depois de passos de rollout.

**Resultado da sonda (30/09, `model_3450`, 40 envs, CPU):**

| Medida | Resultado |
|---|---|
| Fecho forçado, rota SEGURA | 13 de 19 sobrevivem 4 s (antes, 0 de 28) |
| Recompensa por segundo | SEGURA 23,25 (renda 11,61) contra pairar 11,67 |
| Dobra no `load` | ator idêntico no ANDAR, REORIENTAR e PEGAR (relativo 1e-5) |
| Resume de checkpoint antigo | fase 1, `iter_fase` = `iters_balanco` |
| `p_C`, one-hot fixo, save/load, troca de fase | corretos |

**Smoke** (roda o dono): as mesmas quatro, mais: `inspecao` usa a fase 4; antes da fase 4
o `p_C` é 0,05; a cauda parada tem twist zero; na cadeia C o 1º avanço abre o CARREGAR
parado e o 2º o BOTAR; `concluiu_ate_o_fim` é falso com terminação e verdadeiro no
`time_out`.

**Sinais no log da Kaggle:** na fase 1, `s_B` sai de zero e sobe, `renda_congelada` acima
de zero e `time_out` domina; na troca 1 → 2, queda de retorno e pico de value loss,
seguidos de recuperação; em toda fase, `mao_na_pega` e `velocidade_de_junta`.

## 11. Portabilidade para outro simulador

O currículo depende só de: a definição das cadeias e do fecho; o resultado de cada
episódio (concluiu, acabou por `time_out` ou por terminação); EMAs por iteração de PPO. A
normalização do one-hot é do algoritmo de RL, e não do simulador. O que precisa ser
conferido no simulador novo é a ordem do reset: o resultado do episódio que acabou tem de
ser lido antes do sorteio do episódio seguinte.
