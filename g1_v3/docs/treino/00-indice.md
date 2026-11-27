# Índice da documentação do treino da v3

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:
- `env_cfg.py`, `knobs.py`, `balanceador.py`, `observacoes.py`, `metricas.py` (inventário de termos, eventos, knobs, células, fatias).
- `comando.py`, `recompensas.py`, `terminacoes.py`, `eventos.py`, `cena.py`, `curriculo.py`, `alvos.py`, `potencial.py`, `tarefa.py`, `percepcao_lidar.py` (definição dos termos do glossário e conferência de leitores).
- `play.py`, `train.py`, `coleta.py`, `mede_deriva.py`, `exporta_cena.py`, `smoke.py`, `parte0/*.py`, `dados/gera_*.py`, `container/entrada.sh`, `container/*.ipynb` (cabeçalhos e importações, para separar o que não faz parte do treino).
- `mjlab/managers/event_manager.py` (reset dos eventos de intervalo).
- Os onze documentos desta pasta (conferência de cobertura e de contradições).

## 1. Propósito

Esta pasta descreve como o treino da v3 funciona rodando, descrito só pelo código. Ela não descreve intenção, plano nem história. Cada afirmação leva uma referência `arquivo.py:linha`. Cada número leva o nome do knob e o valor lido em `knobs.py`. Um fato que o código lido não mostra leva a marca "Não determinado pelo código lido".

Convenção de referência: `arquivo.py:linha` aponta para `g1_v3/`. `mjlab/...` e `rsl_rl/...` apontam para as bibliotecas do fabricante.

## 2. Regra de manutenção

Toda mudança de código em `g1_v3` atualiza, no mesmo commit:
1. as seções dos documentos desta pasta que a mudança toca (número, condição, ordem, faixa, referência de linha);
2. a linha "Código descrito: `g1_v3` no commit `...`" de cada documento tocado e deste índice.

Uma mudança só em `knobs.py` atualiza todo valor citado do knob (procure `Classe.campo` com grep na pasta). Uma linha de referência que se move também conta como mudança.

## 3. Mapa dos documentos

Não existe documento 09.

| Arquivo | O que cobre |
|---|---|
| `01-ciclo-do-treino.md` | `entrada.sh`, `train.py`, tempos, construção do env, ordem de um passo, ordem do reset, soma da recompensa, fecho do currículo no ciclo, uma iteração do PPO, redes e hiperparâmetros, `algoritmo.py`, `runner.py`, checkpoint, play, logs e métricas. |
| `02-curriculo-e-balanceador.md` | Estado do `Curriculo`: tarefas, fatias, células, níveis, subníveis, linha de terreno, degrau de refino, zona, travas de DR, `restricao`, fecho da iteração, termo `v3`, checkpoint do currículo, knobs do balanceador. |
| `03-reset-terreno-e-perturbacoes.md` | Modos de evento, tabela dos 20 eventos, sorteio da fatia e do terreno, cena no reset, gerador de terreno, entidades e sensores, empurrão, força na caixa, `rfi`, randomização de domínio, reset no play. |
| `04-comando-de-velocidade.md` | Termo `twist`: quando e como o comando é sorteado, tetos por nível, segmento, âncora, nota do segmento, filtros, relógio de marcha, métricas do comando. |
| `05-cadeia-de-manipulacao.md` | Termo `cadeia`: trechos, etapas P0 a P3 e B1 a B3, fecho, invariante, zona, Φ e σ, caixa perdida, estimador da caixa, odometria, registro de sucesso, buffers `v3_*`. |
| `06-recompensas-de-locomocao-e-estilo.md` | Gerente de recompensa, máscara por tarefa, 18 termos de locomoção e estilo, rastreio, relógio de marcha. |
| `07-recompensas-de-manipulacao-e-cercas.md` | 15 termos de manipulação, cercas, contato, freio, `progresso` (renda, Φ, pressa, conquistas), `perda_caixa`, `terminacao`. |
| `08-terminacoes-observacoes-e-acao.md` | Ação e PD, atraso, observação do ator e do crítico, ruído, normalizador, estimador e lidar, as cinco terminações, play. |
| `10-tarefa-andar.md` | Visão por tarefa: reset, comando, progressão, perturbações, recompensas, terminações, observações e métricas do ANDAR. |
| `11-tarefa-pegar.md` | O mesmo recorte para o PEGAR. |
| `12-tarefa-carregar.md` | O mesmo recorte para o CARREGAR. |
| `13-tarefa-botar.md` | O mesmo recorte para o BOTAR. |

## 4. Pergunta, onde está

Notação: `02 §5.8` = documento 02, seção 5.8.

| # | Pergunta | Onde |
|---|---|---|
| 1 | O que é sorteado no reset de um env do ANDAR? | `10 §1.2` a `§1.4` e `§2.2`; terreno e fatia em `03 §4`; base em `03 §5.2` |
| 2 | Quando o empurrão acontece? | `03 §9` (intervalo por env, sorteio e velocidade); efeito na âncora em `04 §7`; o reset não reinicia o relógio dele em `01 §6.4` |
| 3 | Que condição sobe a linha de terreno? | `02 §5.4` e `§5.8`; ANDAR (uma linha por coluna, movida pela taxa de travessia) em `10 §5.6`; CARREGAR em `12 §5.4` |
| 4 | Que condição sobe o nível de velocidade? | `02 §5.8`; ANDAR em `10 §5.4`; CARREGAR em `12 §5.4`; PEGAR em `11 §5.4`; BOTAR em `13 §5.4` |
| 5 | Que condição abre uma célula ou a fatia C? | `02 §5.6` e `§5.7`; ANDAR em `10 §5.5`; BOTAR em `13 §5.8` |
| 6 | Que condição sobe o degrau de refino? | `02 §5.5`; `10 §5.8`; `11 §5.5`; `12 §5.5`; `13 §5.5` |
| 7 | Quais termos pagam com o robô parado? | `06 §6.3` (rastreio sem comando), `06 §7.2` (`passo`, × `_avanco` em `06 §7.4`), `06 §5.4` (`vivo`); `altura_pe` vale 0 com comando de norma 0 (`06 §3`; × `_avanco` em `06 §7.4`); `progresso` em `07 §6.4` |
| 8 | Quais termos de recompensa valem em cada tarefa? | `10 §7.1` (27), `11 §7.1` (23), `12 §7.2` (31), `13 §7.2` (23); fonte `env_cfg.py:42-54` |
| 9 | Qual o peso de cada recompensa e como ele vira valor por passo? | `06 §1` e `§3`; `07 §1` e `§2`; fonte `env_cfg.py:131-134` |
| 10 | Quando a fatia é sorteada e com que probabilidade? | `03 §4.1` e `§4.2`; `02 §7.1` |
| 11 | Quando o fecho da iteração do balanceador roda e o que ele faz, passo a passo? | `01 §6.2` e `§9`; `02 §5` |
| 12 | Quantos passos tem uma iteração do PPO e um episódio? | `01 §3` |
| 13 | Quais os hiperparâmetros do PPO e as redes? | `01 §11` |
| 14 | O que a v3 muda no PPO do `rsl_rl`? | `01 §12` e `§13` |
| 15 | Como o comando de velocidade é sorteado no ANDAR? | `04 §4.3`; no terreno (só frente e rumo) `04 §4.5`; `10 §3.1` |
| 16 | Quais os tetos de velocidade por nível? | `04 §4.2` a `§4.4`; `10 §3.1`; CARREGAR em `12 §3.4` |
| 17 | Quando o comando muda dentro do episódio? | `04 §3`; `10 §3.2`; `12 §3.2` |
| 18 | Como um segmento de comando recebe nota? | `04 §8`; ANDAR em `10 §5.2` |
| 19 | Quando uma etapa (P0 a P3, B1 a B3) fecha? | `05 §6.4`; `11 §4.2`; `13 §4.5` |
| 20 | O que a zona muda e como ela sobe e desce? | `05 §7`; `02 §5.4` |
| 21 | Quando a caixa conta como perdida e o que acontece? | `05 §11`; `07 §7.1`; `13 §8.2` |
| 22 | Que condições terminam o episódio? | `08 §6`; `01 §8`; por tarefa em `10 §8`, `11 §8`, `12 §8`, `13 §8` |
| 23 | Que perturbações e randomizações existem e quando cada uma roda? | `03 §2` e `§9` a `§11`; `02 §10`; por tarefa em `10 §6`, `11 §6`, `12 §6`, `13 §6` |
| 24 | O que o ator observa e o que só o crítico observa? | `08 §4.2`, `§4.5` e `§4.6` |
| 25 | Como a ação da rede vira alvo de junta e torque? | `08 §3` |
| 26 | Como o terreno é gerado e como o env escolhe coluna e linha? | `03 §7` e `§4.4` |
| 27 | O que é a `restricao` e o que ela multiplica? | `02 §9`; `06 §2.2`; `07 §3` |
| 28 | Como o `progresso` é calculado (renda de etapa, Φ, pressa, conquistas)? | `07 §6`; `05 §9` |
| 29 | Qual a ordem exata de um passo do env? | `01 §5`; `08 §2`; `05 §2.1` |
| 30 | Qual a ordem exata do reset de um env? | `01 §6`; eventos em `03 §2`; `05 §2.3` |
| 31 | O que muda entre treino e play? | `01 §15`; `08 §7`; `03 §12` |
| 32 | Que métricas e chaves de log o treino registra? | `01 §16`; por tarefa em `10 §10`, `11 §10`, `12 §10`, `13 §10` |
| 33 | Como a retomada funciona e o que o checkpoint guarda? | `01 §2.4` e `§13`; `02 §12` |
| 34 | Como o estimador da caixa e a odometria funcionam? | `05 §12`; `08 §5` |
| 35 | Quando o relógio de marcha anda? | `04 §11`; `06 §7.1` |
| 36 | Que cercas de postura existem e quando cada uma custa? | `07 §3` e `§4.1` a `§4.7` |
| 37 | Como o freio escolhe o teto de velocidade de junta? | `07 §4.10` |
| 38 | Que variáveis de ambiente controlam a corrida? | `01 §2.1` |
| 39 | Que sensores e entidades a cena tem? | `03 §8` |
| 40 | Onde o comentário do código diz uma coisa e o código faz outra? | Linhas "⚠ Divergência" (use `grep -n "Divergência" *.md`); resumo em `12` ("Divergências entre comentário e código") e `05 §16` |
| 41 | Que knobs nenhum código de treino lê? | Seção 7 deste índice |
| 42 | Como a linha de terreno do ANDAR por coluna sobe e desce? | `02 §5.4` e `§5.8`; `10 §5.6`; chaves `linha_<coluna>` em `01 §6.2` e `10 §10.3` |
| 43 | Quando o episódio de terreno do ANDAR acaba na travessia? | `08 §6`; `01 §8`; `10 §8`; o registro da visita e da travessia em `03 §4.6` |
| 44 | Como o reset sorteia a coluna e a linha do terreno do ANDAR? | `03 §4.4`; `10 §1.4`; pesos por coluna em `10`, seção "Como o treino roda: dinâmica dos sorteios" |
| 45 | Como o desconto da altura do pé age? | `06 §5.5` e `§7.3`; `06 §7.4` (fator `_avanco`) |
| 46 | Como o treino roda com os sorteios: mistura do comando por nível, distância pedida, empurrão contra os degraus, ordem da visita? | `10`, seção "Como o treino roda: dinâmica dos sorteios"; `04 §4.6` |

## 5. Glossário

| Termo | O que é | Referência no código |
|---|---|---|
| tarefa | Um dos quatro comportamentos do currículo: ANDAR, PEGAR, CARREGAR, BOTAR. O ator recebe a tarefa como vetor de 6 canais; REORIENTAR e LEVANTAR não existem no env. | `balanceador.py:20`; `knobs.py:42`; `comando.py:309-310` |
| task do mjlab | O nome registrado do env (`Mjlab-G1-V3` e `Mjlab-G1-V3-SemPercepcao`). Não é a tarefa do currículo. | `tarefa.py:19-26` |
| fatia | O grupo de tarefas que um env recebe no reset: ANDAR, B ou C. | `knobs.py:43`; `balanceador.py:34`; `eventos.py:133-175` |
| trecho | Uma parte da sequência de tarefas de uma fatia. Cada trecho publica uma tarefa e acaba por tempo ou por aviso. | `comando.py:164-197`; `comando.py:201-218` |
| célula | Uma unidade de medida do currículo: um par (tarefa, tipo) com nota `s`. São 8, na ordem de `LAYOUT`. | `balanceador.py:22-25`; `balanceador.py:55-61` |
| nota `s` | A média móvel do sucesso da célula, atualizada no fecho da iteração. | `balanceador.py:55-61`; `balanceador.py:153-195` |
| nível | Um inteiro por tarefa. Ele escolhe a dificuldade dos sorteios (tetos de velocidade, altura da laje, massa). | `balanceador.py:83`; `balanceador.py:322` |
| subnível | Um degrau do nível 0 do PEGAR e do BOTAR. | `balanceador.py:91`; `balanceador.py:325` |
| linha de terreno | Um inteiro por tarefa de cadeia (ANDAR, CARREGAR). Ele escolhe a linha da grade de terreno. No ANDAR, o inteiro é o piso da média das linhas por coluna (`nivel_terreno`). | `balanceador.py:84`; `balanceador.py:329`; `eventos.py:43-74` |
| portão | O limiar que a nota `s` de uma célula precisa alcançar. É um por célula. | `balanceador.py:96-98`; `balanceador.py:122-123` |
| zona | Um número em [0, 1] por etapa. Com z = 1 o critério de fecho é largo; com z = 0 é o final. | `knobs.py:211-220`; `balanceador.py:105` |
| degrau de refino | Um inteiro por tarefa, de 0 a `Balanceador.degraus`. Ele leva as perturbações e certas regras do valor inicial ao final. | `balanceador.py:48-51`; `balanceador.py:338`; `knobs.py:401` |
| degrau global | O menor degrau de refino entre as quatro tarefas. | `balanceador.py:341-343` |
| `restricao` | O menor valor de `nível / nível máximo` entre as quatro tarefas. Ela escala o estilo (os desvios, só no CARREGAR) e as cercas (só no ANDAR). | `balanceador.py:345-348`; `recompensas.py:453-459` |
| trava (de DR) | Um bit que liga uma randomização e nunca desliga: `portao0`, `vies`, `subida`. | `balanceador.py:197-202` |
| iteração | O bloco de `Balanceador.passos_por_iteracao` passos de ambiente. Também é a unidade do fecho do currículo. | `knobs.py:402`; `curriculo.py:119-122`; `tarefa.py` (`num_steps_per_env`) |
| fecho da iteração | O método que atualiza todo o estado do currículo uma vez por iteração. | `balanceador.py:153-195` |
| etapa (ou alvo) | Um sub-objetivo geométrico: P0, P1, P2, P3 (PEGAR) e B1, B2, B3 (BOTAR). O índice vai de 0 a 6; −1 é "sem alvo". | `knobs.py:44`; `balanceador.py:33` |
| fecho da etapa | A condição mantida pelo tempo da etapa que faz o índice do alvo avançar. | `alvos.py:195-210`; `comando.py:41-91` |
| invariante | O produto de três portões (deslocamento, inclinação, mãos) que tem de valer 1 para a etapa fechar. | `comando.py:462-466` |
| aviso | Latch ligado pelo fecho do P3 (PEGAR) ou do B3 (BOTAR). Ele deixa o trecho avançar. | `comando.py:604-607`; `comando.py:201-218` |
| visita (de etapa) | Uma abertura de etapa registrada para a taxa de fecho da zona. | `balanceador.py:106-107`; `comando.py:646-650` |
| linha da coluna | Um número real por coluna de terreno do ANDAR (a coluna 0, o plano, fica fora). Ele sobe ou desce no fecho da iteração pela taxa de travessia da coluna (travessias por visita). | `balanceador.py:84-89`; `balanceador.py:177-182` |
| visita (de coluna) | Um episódio que cumpre as três condições: era ANDAR em terreno (coluna ≠ 0); tinha passos (`episode_length_buf` > 0); e o comando pediu ≥ `Terreno.dist_travessia`, ou o episódio acabou em queda, ou a base passou dessa distância. A coluna conta uma visita. | `eventos.py:155-156` |
| travessia | Uma visita em que a base passou de `Terreno.dist_travessia` (3,5 m, `knobs.py:442`) da origem da célula. O episódio acaba nela, como `time_out`. O episódio acaba nela, como `time_out`. | `eventos.py:143-160`; `terminacoes.py:58-72` |
| rumo | O alvo de giro do ANDAR de terreno: o múltiplo de π/2 mais próximo do rumo de partida. O fabricante calcula o wz por ele. | `comando.py:932-937`; `comando.py:943-947` |
| Φ | O potencial de proximidade ao alvo corrente. Ele paga renda no `progresso`. | `potencial.py:15-37`; `recompensas.py:286-291` |
| σ | A escala de distância do Φ, fixada na abertura de cada etapa. | `comando.py:359-362`; `knobs.py:208` |
| pressa (`h`) | O fator em [0, 1] que decai com o tempo e recarrega com melhora do Φ; multiplica a parcela da etapa. | `recompensas.py:294-296` |
| conquista | O prêmio de uma etapa fechada no `progresso`. | `recompensas.py:286-291`; `knobs.py:343` |
| termo `cadeia` | O comando que mede a cena, abre etapas, avança o trecho e publica a tarefa. | `comando.py:282-669` |
| termo `twist` | O comando que sorteia e guarda a velocidade pedida e mede a nota do segmento. | `comando.py:689-943` |
| segmento | O intervalo entre duas mudanças do comando do `twist`. Ele recebe uma nota ao fechar. | `comando.py:735-751`; `comando.py:759-794` |
| âncora | O instante e a pose de referência da deriva do segmento: `ancora_s` depois da abertura e depois de cada empurrão. | `comando.py:221-224`; `knobs.py:145` |
| canal (com ou sem comando) | O rastreio trata linear e giro em canais. Abaixo de `Comando.limiar_canal` o canal é "sem comando". | `knobs.py:138`; `recompensas.py:374-409` |
| fração entregue | A velocidade filtrada da base sobre o comando, com constante `Comando.tau_fracao`. | `comando.py:854-858`; `knobs.py:136` |
| relógio de marcha | A fase de 0,8 s que só anda com comando acima de `Comando.limiar_marcha`. | `comando.py:869-873`; `knobs.py:139-141` |
| `g_marcha` | O acerto filtrado do relógio de marcha, em [0, 1]. Ele multiplica o rastreio. | `comando.py:869-871` |
| regime (do freio) | Parado, andando ou correndo, pelo total do comando. Cada regime tem um teto de velocidade de junta. | `recompensas.py:900-912`; `knobs.py:305-325` |
| cerca | Um termo de custo que vale 0 dentro de um limite de postura e cresce fora dele. | `recompensas.py:156-198` |
| máscara | A seleção das tarefas em que um termo vale; fora delas o valor é 0. | `curriculo.py:93-100`; `recompensas.py:361-365` |
| buffer `v3_*` | Um tensor por env que o env guarda (tarefa, fatia, alvo, pulsos, relógios). | `curriculo.py:39-91` |
| pulso | Um buffer que vale True por um passo (`v3_perdeu_agora`). | `comando.py:411`; `terminacoes.py:75-84` |
| episódio de terreno | Um episódio que o `fatia` sorteia com terreno (probabilidade = fração da célula `terreno`). | `eventos.py:36-40`; `eventos.py:133-175` |
| ator e crítico | Os dois grupos de observação. O crítico não tem ruído nem atraso e vê canais privilegiados. | `observacoes.py:361-422` |
| estimador da caixa | A leitura simulada da caixa (câmera RGB e profundidade). O ator só vê esta leitura. | `estimador.py`; `observacoes.py:140-151` |
| odometria | A pose da base estimada com deriva, que substitui a pose verdadeira nas observações. | `estimador.py`; `eventos.py:228-241` |
| mapa | O vetor de 882 canais do lidar (altura e confiança). Existe com `Percepcao.ligada` = True. | `observacoes.py:361-422`; `percepcao_lidar.py` |
| play | O modo de inspeção: ator sem ruído, sem empurrão, sem DR de refino, sem currículo. | `env_cfg.py:80-88`; `env_cfg.py:200`; `env_cfg.py:216` |
| `ATIVO` | A instância única de `Knobs` que o treino lê. | `knobs.py:569` |

## 6. Arquivos de `g1_v3` que não fazem parte do treino rodando

O treino rodando é `container/entrada.sh`, que chama `python -m g1_v3.train`. Ele carrega `tarefa.py`, e `tarefa.py` carrega o resto. Os arquivos abaixo não entram nesse caminho. Conferência: nenhum módulo de treino os importa (grep de `import` e `from g1_v3`).

| Arquivo | Por que não faz parte do treino | Conferência |
|---|---|---|
| `play.py` | Entrada de inspeção com política treinada. Chama `mjlab.scripts.play`. | `play.py:1-11` |
| `smoke.py` | Checks da v3. Ele constrói envs e roda física, e imprime `N ok / M falhas`. Ninguém o importa. | `smoke.py:1-12` |
| `coleta.py` | Coleta dados passo a passo de uma política treinada e grava `coleta.npz`. | `coleta.py:1-14` |
| `mede_deriva.py` | Mede a deriva de posição e rumo por segmento de uma política treinada. | `mede_deriva.py:1-8` |
| `exporta_cena.py` | Exporta a cena para o MuJoCo clássico e prova a paridade da observação. O consumidor é `pilota_v3.py`, fora de `g1_v3`. | `exporta_cena.py:1-12` |
| `parte0/` (`comum.py`, `custo_filtro.py`, `item1_phi.py`, `item2_derivadas.py`, `item4_balanceador.py`, `item6_comando_baixo.py`, `item8_passo.py`, `item15_aperto.py`, `saida/*.txt`) | Estudos de mesa. Só `parte0` importa `parte0`. Os `.txt` são saídas gravadas dos estudos. | `parte0/comum.py:1-8`; `parte0/item8_passo.py:81`; `parte0/item4_balanceador.py:64` |
| `dados/gera_capsulas.py`, `dados/gera_mid360.py`, `dados/gera_referencias.py` | Geradores dos arquivos `.npz`. O treino lê os `.npz`, e não os geradores. | `dados/gera_capsulas.py:1`; `observacoes.py:66`; `curriculo.py:20`; `knobs.py:73` |
| `container/treino_runpod.ipynb`, `container/treino_kaggle.ipynb` | Notebooks de lançamento. Eles escolhem as variáveis e chamam `entrada.sh`. | `container/treino_runpod.ipynb` (célula que roda `entrada.sh`) |

Os `.npz` de `dados/` fazem parte do treino: `referencias.npz` (`knobs.py:73`, `knobs.py:584`), `capsulas.npz` (`observacoes.py:66`) e `mid360_passos.npz` (`cena.py:425`, `curriculo.py:20`).

### Funções de módulos de treino sem chamador no treino

| Função | O que faz | Quem a chama |
|---|---|---|
| `alvos.detecta_fim` (`alvos.py:213-228`) | Detector de pose de fim, com `Detector.estavel_s` e `Detector.juntas_paradas`. | Nenhum módulo de treino. O comentário em `alvos.py:216` cita `pilota_v3.py`, fora do pacote. |
| `potencial.dobradica` (`potencial.py:65-67`) | `relu(x − limite)`. | Nenhum módulo. |
| `potencial.L_alvo` (`potencial.py:109-119`) | Comprimento de passo alvo por `Recompensa.L_min` e `Recompensa.fator_T`. | Só `parte0/item8_passo.py:81`. |
| `knobs.dim_ator` (`knobs.py:575-578`) | Soma os canais do ator (1 457 números). | Só `exporta_cena.py`. |

## 7. Knobs sem leitor no treino

Conferência por grep em `g1_v3` (todos os módulos de treino, e depois `parte0/`, `smoke.py` e os scripts de fora). Um campo entra aqui se nenhuma instrução executada no treino o lê. O `env_cfg._sem_none` percorre todos os campos (`env_cfg.py:66-77`), mas só confere que nenhum vale `None`: isso não conta como leitura.

| Knob | Valor | Linha | Quem o lê fora do treino |
|---|---|---|---|
| `Contrato.historico` | 5 | `knobs.py:56` | Ninguém. O histórico de cada termo vem de `Contrato.blocos_ator` (`observacoes.py:369`, `observacoes.py:379`). |
| `Recompensa.L_min` | 0,20 m | `knobs.py:246` | `parte0/item8_passo.py:81`, por `potencial.L_alvo`. |
| `Recompensa.fator_T` | (0,8; 1,4) | `knobs.py:247` | `parte0/item8_passo.py:81`, por `potencial.L_alvo`. |
| `Balanceador.piso_tarefa` | 0,20 | `knobs.py:388` | `parte0/item4_balanceador.py:64`. O balanceador de células usa `Balanceador.piso_sub` (`balanceador.py:299`). |
| `Percepcao.raios_por_passo` | 205 | `knobs.py:489` | Ninguém. O número de raios vem do arquivo `mid360_passos.npz` (`percepcao_lidar.py:19-29`). |
| `Detector.estavel_s` | 0,5 s | `knobs.py:169` | Só `alvos.detecta_fim` (`alvos.py:228`), sem chamador no treino. |
| `Detector.juntas_paradas` | 0,2 | `knobs.py:170` | Só `alvos.detecta_fim` (`alvos.py:225`), sem chamador no treino. |

Os campos `Detector.tol_pelve`, `Detector.tol_tronco_graus` e `Detector.tol_palma` têm um leitor indireto: `Alvos.tol_pelve_b3`, `Alvos.tol_tronco_b3_graus`, `Alvos.tol_palmas_b3` e a tabela `Alvos.zona` os usam como valor na definição da classe (`knobs.py:198-200`, `knobs.py:218-219`). O treino lê então os campos de `Alvos`.

Os pesos de `Recompensa` não aparecem no texto do código como `k.recompensa.<nome>`. O `make_env_cfg` os lê por `getattr(k.recompensa, nome)` (`env_cfg.py:132`). Por isso eles têm leitor.
