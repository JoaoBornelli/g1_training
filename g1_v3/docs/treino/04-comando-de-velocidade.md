# Comando de velocidade (twist), segmentos e nota

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:
- `comando.py`, `knobs.py`, `env_cfg.py`
- `balanceador.py`, `curriculo.py`, `eventos.py`, `terminacoes.py`, `tarefa.py`, `alvos.py` (partes que o `twist` chama ou lê)
- `recompensas.py`, `observacoes.py`, `metricas.py` (só os trechos que leem o `twist`)
- `mjlab/tasks/velocity/mdp/velocity_command.py` (o `_update_command` do fabricante), `mjlab/managers/command_manager.py`, `mjlab/managers/event_manager.py`, `mjlab/entity/data.py` (`heading_w`)
- `mjlab/envs/manager_based_rl_env.py`, `mjlab/envs/mdp/events.py`

Convenção: os números vêm de `knobs.py`. Escrevo `Classe.campo` = valor (`knobs.py:linha`). O passo da política dura `step_dt` = 0,02 s (`Contrato.dt_fisica` = 0,005 s, `knobs.py:54`; `Contrato.decimacao` = 4, `knobs.py:55`; produto em `env_cfg.py:103-104`).

## 1. Vocabulário

| Termo | Significado |
|---|---|
| comando | Vetor de 3 números (vx, vy, wz) que diz ao robô a velocidade desejada. |
| termo `twist` | Objeto `TwistV3` que guarda e sorteia o comando (`comando.py:689`). |
| termo `cadeia` | Objeto `CadeiaCmd` que publica a tarefa corrente e troca de trecho (`comando.py:282`). Roda antes do `twist` (`env_cfg.py:118-128`). |
| tarefa | ANDAR, PEGAR, CARREGAR ou BOTAR (`comando.py:30`; `knobs.py:42`). |
| fatia | Grupo de tarefas que um env recebe no reset: ANDAR, B ou C (`comando.py:32`; `knobs.py:43`). |
| trecho | Parte da cadeia de tarefas da fatia (por exemplo, ANDAR de espera, depois PEGAR) (`comando.py:175-180`). |
| célula | Classe de sorteio do currículo. O `twist` usa as de ANDAR (`andar`, `terreno`) e as de CARREGAR (`reto`, `giro_lugar`, `arco`, `terreno`) (`balanceador.py:22-24`). |
| nível | Inteiro do currículo por tarefa. Ele escolhe os tetos de velocidade (`balanceador.py:322`). |
| segmento | Intervalo entre duas mudanças do comando (seção 6). |
| âncora | Instante e pose que servem de referência para medir a deriva do segmento (seção 7). |
| nota | Número que o segmento entrega ao currículo (seção 8). |
| rumo | Ângulo do eixo frontal do robô no mundo (`heading_w`). O ANDAR de terreno segue um rumo alvo e o `wz` dele sai do erro de rumo (seção 4.5). |
| visita, travessia | Visita: episódio de ANDAR em terreno, com pelo menos um passo (`episode_length_buf` > 0), em que o comando pediu ≥ `dist_travessia`, ou que acabou em queda, ou em que a base passou de `dist_travessia`. Travessia: visita em que a base passou de `dist_travessia` da origem da célula (seção 4.6; `eventos.py:143-160`). |

## 2. Quem recebe comando e em que referencial

O `twist` guarda o comando em `vel_command_b`, no referencial da base do robô (b). O comando é o vetor (vx, vy, wz): vx para a frente, vy para o lado, wz o giro em torno do eixo vertical (`velocity_command.py:38`, `comando.py:925`). A propriedade `command` devolve `vel_command_b` (`velocity_command.py:58-59`). O ator lê esse vetor como o termo `comando` da observação (`observacoes.py:392`). A recompensa de rastreio compara o comando com `root_link_lin_vel_b` e `root_link_ang_vel_b`, também no referencial da base (`recompensas.py:385-387`). O rastreio do canal com comando usa a velocidade filtrada `v_filt` na projeção (seção 10) e, no linear, a velocidade instantânea na parte perpendicular; o canal sem comando usa a velocidade instantânea (`recompensas.py:398-409`).

`vel_command_w` recebe uma cópia de `vel_command_b` no sorteio (`comando.py:938`). Nenhum código lido rotaciona essa cópia: o `_update_command` roda (rumo, seção 4.5), mas `is_world_env` nunca liga (seção 5). No ANDAR de terreno, a cópia guarda wz = 0, porque o `wz` de rumo sai depois, a cada passo (`comando.py:943-948`).

Quem recebe comando diferente de zero (condição em `comando.py:905`):

| Fatia | Trecho (duração) | Comando |
|---|---|---|
| ANDAR | ANDAR, `Cadeia.episodio_s` = 20,0 s (`knobs.py:160`; `comando.py:176`) | Sorteado (seção 4). No terreno: só frente, e o `wz` vem do rumo (seção 4.5). |
| B | ANDAR de espera, `Cadeia.espera_s` = (0,5; 1,0) s (`knobs.py:154`) | Zero. |
| B | PEGAR (por aviso) | Zero. |
| B | CARREGAR, `Cadeia.carregar_b_s` = (4,0; 7,0) s (`knobs.py:155`) | Sorteado. Fica fixo no trecho. |
| C | ANDAR de espera, PEGAR, CARREGAR (`Cadeia.carregar_c_s` = (0,5; 1,0) s, `knobs.py:156`), BOTAR, ANDAR final (`Cadeia.andar_final_s` = 1,5 s, `knobs.py:157`) | Zero em todos. |

A condição exata é `(tarefa == ANDAR) & (fatia == ANDAR)` ou `(tarefa == CARREGAR) & (fatia == B)` (`comando.py:905`). Fora dela, `vel_command_b` recebe 0 e a célula do env vale −1 (`comando.py:901-902`). Quando a caixa se perde e a tarefa vira ANDAR dentro da fatia B ou C, a condição falha e o comando vai a zero (`comando.py:420`, `comando.py:905`).

## 3. Quando o comando é sorteado

O sorteio roda em `TwistV3._resample_command` (`comando.py:897`). Quem o chama depende do momento.

| Momento | O que acontece | Referência |
|---|---|---|
| Criação do env (startup) | Nenhum sorteio de comando. O `__init__` só zera os buffers e abre o segmento de todos os envs com comando zero. | `comando.py:699-733` |
| Reset do episódio | O reset da `cadeia` abre o primeiro trecho e chama `_publica`, que chama `twist._resample`. Depois o reset do `twist` sorteia outra vez. O segundo sorteio sobrescreve o primeiro (comando, célula e `time_left`). O rumo alvo só nasce no segundo, porque só nele `_agora` vale 0 (seção 4.5). | `comando.py:660`, `comando.py:568-584`, `comando.py:519`, `comando.py:893` |
| Troca de trecho (a cada passo em que a cadeia avança) | `_abre_trecho` chama `_publica`, que reamostra o comando do `twist`. | `comando.py:619`, `comando.py:519` |
| Caixa perdida (a cada passo) | `_perda` chama `_publica(ids, ANDAR)`. O `twist` reamostra e o comando vai a zero (a condição da seção 2 falha). | `comando.py:420` |
| Temporizador do `twist` (a cada passo) | `time_left` cai `dt` por passo. Quando chega a ≤ 0, o env reamostra. | `command_manager.py:109-114` |

Ordem de chamada dentro de um passo do env: terminações, recompensas, métricas, reset dos envs que terminaram, e só depois `command_manager.compute` (`manager_based_rl_env.py:436-456`). Dentro do `compute`, a `cadeia` roda antes do `twist` (`command_manager.py:249-251`, `env_cfg.py:118-128`). Os eventos de passo e de intervalo (entre eles o `empurrao`) rodam depois do `compute` (`manager_based_rl_env.py:458-461`). Dentro do `compute` do `twist`, o `_update_command` roda depois da reamostragem: o `wz` de rumo de um comando novo vale já no mesmo passo (`command_manager.py:109-115`; `comando.py:943-948`). A `cadeia` sobrescreve `compute` sem chamar o da classe mãe, então o temporizador `time_left` dela não roda (`comando.py:596`); ela decide a troca de trecho pelo próprio relógio (`comando.py:608-619`).

### 3.1 Período de reamostragem

`CommandTerm._resample` sorteia `time_left` uniforme em `resampling_time_range` e incrementa `command_counter` (`command_manager.py:117-123`). Para o `twist`, esse intervalo é `Comando.reamostra_s` = (3,0; 8,0) s (`knobs.py:137`; `env_cfg.py:123`). O sorteio de `time_left` usa o gerador global do torch, não o `v3_gerador` (`command_manager.py:119-121`; contraste com `comando.py:899`).

Quem reamostra e quem fica fixo:

| Caso | Comportamento | Referência |
|---|---|---|
| ANDAR na fatia ANDAR | Reamostra por temporizador, com período uniforme em (3,0; 8,0) s. | `comando.py:905`, `knobs.py:137` |
| CARREGAR na fatia B | O comando é sorteado na abertura do trecho. `time_left` recebe infinito logo depois do sorteio, então o temporizador nunca dispara. O comando fica fixo até o trecho acabar. | `comando.py:940-941` |
| Todos os outros (comando zero) | O temporizador roda e reamostra a cada (3,0; 8,0) s. O resultado é comando zero de novo, mas o contador muda e o segmento fecha e reabre. | `comando.py:901-902`, `command_manager.py:117-123` |

Cada reamostra, por qualquer caminho, incrementa `command_counter`. Esse contador é o sinal que o `twist` usa para detectar a fronteira de segmento (seção 6).

## 4. O sorteio do comando

### 4.1 Escolha da célula

A célula define o tipo de movimento. Ela é escolhida na reamostra, por tarefa (`comando.py:906-925`).

- Episódio de terreno: `terrain_types != 0` (a coluna 0 é plano). Nele, a célula é `terreno` da tarefa (`comando.py:914-916`). O `sorteia_fatia` escreve `terrain_types` no reset, antes do `twist` (`eventos.py:161-175`). No reset, o env vira episódio de terreno com probabilidade igual à fração da célula `terreno` no sorteio de células da tarefa da fatia: ANDAR para a fatia ANDAR, CARREGAR para as fatias B e C (`eventos.py:36-40`, `eventos.py:165-167`). Só as fatias ANDAR e B recebem coluna de terreno; a fatia C fica com `terrain_types` = 0 (`eventos.py:70-74`). No ANDAR, a coluna sai por pesos que crescem com o atraso da linha da coluna, e a linha sai por arredondamento estocástico da linha da coluna (`eventos.py:61-66`).
- Fora do episódio de terreno: `sorteia_celulas(tarefa, n, gerador, sem_tipo="terreno")` sorteia entre as células abertas, com o peso de `terreno` zerado (`comando.py:917-918`; `balanceador.py:306-316`).
- O peso de cada célula aberta é `F_i = piso_sub + (1 − n·piso_sub)·(D_i + eps) / Σ(D + eps)`, com `D_i = max(portão_i − s_i, 0)` (`balanceador.py:37-45`, `balanceador.py:295-299`). `s_i` é a nota média (EMA) da célula. `Balanceador.piso_sub` = 0,10 (`knobs.py:389`). `Balanceador.eps` = 0,02 (`knobs.py:390`). `n` é o número de células abertas, com `terreno` incluída quando aberta. Na célula `terreno` do ANDAR, `D` não é a nota: é a distância média das linhas de terreno ao topo, `1 − (média das linhas das colunas de terreno) / linha máxima` (`balanceador.py:139-143`). O peso zerado de `terreno` entra depois da conta de `F`, direto em `torch.multinomial` (`balanceador.py:313-316`).
- Portões por célula: `Balanceador.portao` = 0,75 para as de CARREGAR (`knobs.py:391`); `Balanceador.portao_andar` = 0,75 para `andar` (`knobs.py:392`); `Balanceador.portao_andar_terreno` = 0,50 para `terreno` do ANDAR (`knobs.py:393`); a escolha por célula está em `balanceador.py:97-98`.
- Abertura das células: no começo só `ANDAR/andar` está aberta (`balanceador.py:27`, `balanceador.py:79`). No ANDAR, com a célula `terreno` fechada, o sorteio sempre cai em `andar`. As outras abrem no fecho de iteração (`balanceador.py:204-232`):
  - `CARREGAR/reto` e `PEGAR/pegar` abrem quando o nível do ANDAR é ≥ `Balanceador.nivel_abre_b` = 5 (`knobs.py:396`; `balanceador.py:209-211`);
  - `giro_lugar` e depois `arco` (ordem de `VELOCIDADE`, `balanceador.py:30`) abrem, uma por fecho, quando a média das notas das células abertas da tarefa é ≥ a média dos portões (`balanceador.py:212-217`);
  - `terreno` do ANDAR abre quando o nível do ANDAR é ≥ `Balanceador.nivel_abre_terreno` = 2 e a média das notas das células abertas passa. A nota da célula `terreno` não entra, porque o robô parado na plataforma a mantém perto de 0,5 (`knobs.py:412`; `balanceador.py:221-228`). O robô aprende a andar no plano antes da escada;
  - `terreno` do CARREGAR abre quando o nível do CARREGAR é o máximo e a média das abertas passa; exige também `nivel_terreno("ANDAR")` ≥ 1, o piso da média das linhas do ANDAR (`balanceador.py:222-228`; `balanceador.py:329-333`).

O nível lido é `env.v3_cur.nivel(tarefa)` no instante do sorteio (`comando.py:920`). Mesmo na célula `terreno`, o nível é o de velocidade da tarefa, e não a linha de terreno (`comando.py:920`). No CARREGAR, `terreno` só abre no nível máximo, e os tetos dela são os do último nível. No ANDAR, `terreno` abre no nível 2 ou acima, então os tetos dela são os do nível corrente do ANDAR (2 a 7); os níveis 0 e 1 não sorteiam terreno.

### 4.2 De onde vem o nível

O nível muda só quando o currículo fecha uma iteração do PPO. O termo de currículo `v3` roda a cada reset e chama `fecha_iteracao` quando `common_step_counter // passos_por_iteracao` muda (`curriculo.py:103-122`). `Balanceador.passos_por_iteracao` = 24 (`knobs.py:402`); o mesmo knob define `num_steps_per_env` do PPO (`tarefa.py:49`). O termo de currículo roda no início do `_reset_idx`, antes do reset dos comandos (`manager_based_rl_env.py:553-554`, `manager_based_rl_env.py:581`). A primeira chamada só grava a iteração corrente em `v3_iter` (inicial −1) e não fecha nada (`curriculo.py:58-59`, `curriculo.py:120-122`).

A tarefa sobe um nível quando `ok` vale por `Balanceador.iters_nivel` = 50 fechos seguidos; um fecho com `ok` falso zera a contagem (`knobs.py:395`; `balanceador.py:250-277`). Abaixo do nível máximo, `ok` exige: todas as células de velocidade da tarefa abertas, e média das notas ≥ média dos portões (`balanceador.py:125-128`, `balanceador.py:259-260`). No nível máximo do CARREGAR, a mesma contagem sobe a linha de terreno (`_nivel_ter`), e `ok` exige a célula `terreno` aberta e no portão (`balanceador.py:261-262`, `balanceador.py:269-272`). No nível máximo do ANDAR, `ok` é falso e o nível fica em 7 (`balanceador.py:263-264`). A linha de terreno não muda os tetos de velocidade. O nível máximo do ANDAR é `round((frente_tetos[-1] − andar_inicio) / andar_passo)` = `round((1,0 − 0,3) / 0,1)` = 7. O do CARREGAR é `len(frente_tetos) − 1` = 4 (`balanceador.py:72-73`).

A linha de terreno do ANDAR é um vetor por coluna (`Curriculo.linha`, 9 posições; a 0, `plano`, não conta). O `fecha_iteracao` move cada linha pela taxa de travessia da coluna, que é por visita (`trav_ter / vis_ter`): com pelo menos `Balanceador.linha_n_min` = 50 visitas na iteração, `linha ← clip(linha + linha_ganho · (taxa − zona_t_alvo); 0; 12)`, com `linha_ganho` = 2,5 e `zona_t_alvo` = 0,6 (`knobs.py:404`, `knobs.py:410-411`; `balanceador.py:178-182`). Taxa 1,0 sobe 1,0 linha; taxa 0 desce 1,5; taxa 0,6 mantém. `nivel_terreno("ANDAR")` devolve o piso da média das 8 linhas (`balanceador.py:329-333`). A linha máxima do ANDAR é `Terreno.linhas − 1` = 12 (`knobs.py:429`; `balanceador.py:77`). Visita e travessia vêm do `sorteia_fatia` (seção 4.6).

### 4.3 Sorteio no ANDAR (tipos `andar` e `terreno`)

A função é `sorteia_velocidade` (`comando.py:118`). No ANDAR (`comando.py:125-147`), o tipo `andar` (plano) sorteia frente, trás, lado e giro. O tipo `terreno` recebe só frente e rumo (item 3 e item 4):

1. Os tetos do nível `n` (`comando.py:127-130`):
   - lado = `min(andar_lado_inicio + andar_passo·n, lado_teto_max)`
   - frente = `min(andar_inicio + andar_passo·n, frente_tetos[-1])`
   - trás = `min(andar_inicio + andar_passo·n, tras_teto_max)`
   - giro = `min(andar_giro_inicio + andar_passo·n, giro_tetos[-1])`
2. No tipo `terreno`, frente e trás passam por `min(·, terreno_v_max)` (`comando.py:131-132`). Com os knobs atuais isso não muda nada, porque frente ≤ `frente_tetos[-1]` = 1,0 e `terreno_v_max` = 1,0 (`knobs.py:122`, `knobs.py:148`), e trás ≤ `tras_teto_max` = 0,5 (`knobs.py:123`).
3. Linear. O raio é `r = √U1` e o ângulo é `ang = π·(U2 − 0,5)`, em [−π/2; π/2) (`comando.py:136`).
   - Tipo `andar`: ponto uniforme na meia elipse (`comando.py:140-142`). O semieixo em x vale `frente` com probabilidade `frente / (frente + tras)`, senão `−tras` (`U3·(frente + tras) < frente`). `vx = semi_x · r · cos(ang)` e `vy = lado · r · sin(ang)`.
   - Tipo `terreno` (T19, 27/11): `vx = frente · r` e `vy = 0` (`comando.py:137-138`). Não há `trás`, nem `lado`, nem `cos`. O `ang` é sorteado e não é usado. Então `vx` vai de 0 a `frente`, com densidade crescente: `P(vx ≤ x) = (x / frente)²`.
4. Giro. Tipo `andar`: `wz = giro · (2·U − 1)`, uniforme em (−giro; giro) (`comando.py:143`). Tipo `terreno`: `wz` fica 0 no sorteio. O `wz` vem do rumo, a cada passo (seção 4.5).
5. Giro mínimo: se `|wz| < giro_min`, `wz` vira 0 (`comando.py:144`). No terreno, `wz` já é 0 e o passo não muda nada.
6. Linear mínimo: se `‖(vx, vy)‖ < linear_min`, os dois vão a 0 (`comando.py:145`).
7. Comando parado: com probabilidade `andar_parado`, o vetor inteiro (vx, vy, wz) vai a 0. O sorteio é por env, uniforme: `U < andar_parado` (`comando.py:146`).

Cada `U` é uniforme em [0; 1), tirada do `v3_gerador` (`comando.py:98-99`, `comando.py:136`, `comando.py:140`; gerador criado em `curriculo.py:62-64`).

Knobs usados:

| Knob | Valor | Linha |
|---|---|---|
| `Comando.andar_inicio` | 0,3 m/s | `knobs.py:129` |
| `Comando.andar_lado_inicio` | 0,2 m/s | `knobs.py:130` |
| `Comando.andar_giro_inicio` | 0,3 rad/s | `knobs.py:131` |
| `Comando.andar_passo` | 0,1 por nível | `knobs.py:134` |
| `Comando.frente_tetos[-1]` | 1,0 m/s | `knobs.py:122` |
| `Comando.tras_teto_max` | 0,5 m/s | `knobs.py:123` |
| `Comando.lado_teto_max` | 0,3 m/s | `knobs.py:124` |
| `Comando.giro_tetos[-1]` | 0,5 rad/s | `knobs.py:125` |
| `Comando.giro_min` | 0,1 rad/s | `knobs.py:132` |
| `Comando.linear_min` | 0,1 m/s | `knobs.py:133` |
| `Comando.andar_parado` | 0,10 | `knobs.py:135` |

Tetos do ANDAR por nível (calculados com as fórmulas do item 1). A tabela vale para `andar`. Para `terreno`, só a coluna Frente vale:

| Nível | Frente (m/s) | Trás (m/s) | Lado (m/s) | Giro (rad/s) |
|---|---|---|---|---|
| 0 | 0,3 | 0,3 | 0,2 | 0,3 |
| 1 | 0,4 | 0,4 | 0,3 | 0,4 |
| 2 | 0,5 | 0,5 | 0,3 | 0,5 |
| 3 | 0,6 | 0,5 | 0,3 | 0,5 |
| 4 | 0,7 | 0,5 | 0,3 | 0,5 |
| 5 | 0,8 | 0,5 | 0,3 | 0,5 |
| 6 | 0,9 | 0,5 | 0,3 | 0,5 |
| 7 | 1,0 | 0,5 | 0,3 | 0,5 |

Probabilidades derivadas das fórmulas acima (tipo `andar`; para `terreno`, a probabilidade de o linear cair abaixo de `linear_min` vale `(linear_min / frente)²`, porque `P(vx < x) = (x / frente)²`):
- Probabilidade de a meia elipse da frente ser sorteada: `frente / (frente + tras)`. No nível 0 vale 0,5. No nível 7 vale 1,0 / 1,5.
- Probabilidade de `wz` ser zerado pelo giro mínimo: `giro_min / giro` (para giro ≥ 0,1). No nível 0 vale 0,1 / 0,3.
- Probabilidade de o linear cair abaixo de `linear_min`: o círculo de raio `linear_min` = 0,1 cabe na elipse de todos os níveis (`lado` ≥ 0,2 e `frente`, `tras` ≥ 0,3). A probabilidade vale então `2 · linear_min² / (lado · (frente + tras))`, ou seja, `0,02 / (lado · (frente + tras))` com `linear_min` = 0,1. É uma conta sobre a fórmula do item 3, e o código não a calcula.
- Os três zeros são aplicados em sequência sobre o mesmo vetor: o giro mínimo zera só `wz` (`comando.py:144`), o linear mínimo zera só (vx, vy) (`comando.py:145`), e o comando parado zera os três (`comando.py:146`). Os sorteios de `wz` e do linear são independentes (`comando.py:136-143`).

Probabilidades por nível no ANDAR (derivadas das fórmulas acima; `P_giro = giro_min / giro`; `P_tudo = andar_parado + (1 − andar_parado) · P_lin · P_giro`):

| Nível | P(linear zerado pelo mínimo) | P(giro zerado pelo mínimo) | P(comando todo zero) |
|---|---|---|---|
| 0 | 0,167 | 0,333 | 0,150 |
| 1 | 0,083 | 0,250 | 0,119 |
| 2 | 0,067 | 0,200 | 0,112 |
| 3 | 0,061 | 0,200 | 0,111 |
| 4 | 0,056 | 0,200 | 0,110 |
| 5 | 0,051 | 0,200 | 0,109 |
| 6 | 0,048 | 0,200 | 0,109 |
| 7 | 0,044 | 0,200 | 0,108 |

⚠ Divergência: o comentário da linha do `linear_min` em `comando.py:145` diz "(= `limiar_marcha`)". O código tem dois knobs separados, `linear_min` (`knobs.py:133`) e `limiar_marcha` (`knobs.py:141`). O zero do sorteio usa `<` (`comando.py:145`), e o relógio da marcha liga com `>` sobre a norma dos três canais (`comando.py:863`). Os dois valem 0,1, mas nada no código os mantém iguais.

### 4.4 Sorteio no CARREGAR (fatia B)

Esse ramo vale para qualquer tarefa diferente de ANDAR; o `twist` só o chama para CARREGAR (`comando.py:125`, `comando.py:906`). O tipo da célula define o que anda (`comando.py:148-158`). Nenhuma célula do CARREGAR tem sorteio de parado.

1. Tetos do nível `n` (`comando.py:148-150`):
   - frente = `frente_tetos[n]`
   - trás = `min(frente, tras_teto_max)`
   - lado = `min(frente, lado_teto_max)`
   - giro = `giro_tetos[min(n, 3)]`
2. No tipo `terreno`, frente e trás passam por `min(·, terreno_v_max)` (`comando.py:151-152`). Com os knobs atuais não muda nada, porque `frente_tetos[n]` ≤ 1,0.
3. Linear (tipos `reto`, `terreno`, `arco`): função `_anda` (`comando.py:154-155`, `comando.py:102-110`).
   - Eixo: `randint(0, 4)`, uniforme: 0 é +x, 1 é −x, 2 é +y, 3 é −y.
   - Teto: frente (+x), trás (−x), lado (±y).
   - Piso: `piso_velocidade` para ±x, `piso_lado` para ±y.
   - Módulo = `piso + (teto − piso) · U`, uniforme entre piso e teto.
   - O outro eixo recebe 0. Não há elipse.
4. Giro (tipos `arco`, `giro_lugar`): função `_giro` (`comando.py:156-157`, `comando.py:113-115`). O sinal é + se `U ≥ 0,5` e − se não. O módulo é `piso_velocidade + (giro − piso_velocidade) · U`.
5. Sem linear no `giro_lugar`: o linear fica 0. Sem giro nos tipos `reto` e `terreno`: o giro fica 0.
6. Não existe sorteio de comando parado nesta tarefa. O módulo mínimo de cada canal sorteado é o piso (> 0), então o comando nunca é zero quando a fatia é B e a tarefa é CARREGAR.

Knobs usados:

| Knob | Valor | Linha |
|---|---|---|
| `Comando.frente_tetos` | (0,4; 0,55; 0,7; 0,85; 1,0) m/s | `knobs.py:122` |
| `Comando.tras_teto_max` | 0,5 m/s | `knobs.py:123` |
| `Comando.lado_teto_max` | 0,3 m/s | `knobs.py:124` |
| `Comando.giro_tetos` | (0,3; 0,4; 0,5; 0,5) rad/s | `knobs.py:125` |
| `Comando.piso_velocidade` | 0,25 | `knobs.py:126` |
| `Comando.piso_lado` | 0,2 | `knobs.py:127` |

Tetos do CARREGAR por nível (calculados com as fórmulas do item 1):

| Nível | Frente (m/s) | Trás (m/s) | Lado (m/s) | Giro (rad/s) |
|---|---|---|---|---|
| 0 | 0,4 | 0,4 | 0,3 | 0,3 |
| 1 | 0,55 | 0,5 | 0,3 | 0,4 |
| 2 | 0,7 | 0,5 | 0,3 | 0,5 |
| 3 | 0,85 | 0,5 | 0,3 | 0,5 |
| 4 | 1,0 | 0,5 | 0,3 | 0,5 |

Faixa do módulo por direção no CARREGAR: frente e trás de 0,25 até o teto; lado de 0,2 até o teto; giro de 0,25 até o teto.

Por tipo de célula do CARREGAR:

| Tipo | Linear | Giro |
|---|---|---|
| `reto` | sim | não |
| `terreno` | sim | não |
| `arco` | sim | sim |
| `giro_lugar` | não | sim |

O `terreno` do CARREGAR só ocorre em episódio de terreno (`comando.py:914-916`).

### 4.5 O rumo do ANDAR de terreno

No ANDAR de terreno, o robô anda para a frente e segue um rumo alvo (T19, 27/11). O comando de terreno tem vx > 0, vy = 0 e `wz` do rumo (item 3 e item 4 da seção 4.3). O mjlab calcula o `wz` com o `_update_command` dele. A `TwistV3` liga o mecanismo e põe o alvo.

| Etapa | O que acontece | Referência |
|---|---|---|
| Quem liga | `is_heading_env` vale verdadeiro no env de terreno com comando linear de norma ≥ `Comando.limiar_canal` (0,02). Como o linear sorteado é 0 ou ≥ `linear_min` (0,1), a regra é "terreno e linear diferente de zero". O env de terreno parado (`andar_parado` ou `linear_min`) fica sem rumo e com `wz` = 0. Todo env fora do terreno tem `is_heading_env` falso. O `_resample_command` recalcula o flag a cada reamostragem. | `comando.py:903`, `comando.py:932` |
| O alvo | `heading_target` nasce quando `_agora` vale 0 (o segundo sorteio do reset, seção 3). É o múltiplo de π/2 mais próximo do rumo de partida: `round(atan2(f_y, f_x) / (π/2)) · π/2`, com `f` o vetor frontal do robô, lido do `qpos` (o `xquat` ainda é o do episódio anterior). O alvo vale para todo env de terreno e fica fixo no episódio. As reamostragens seguintes não o mudam. | `comando.py:933-937` |
| O erro | `erro = wrap_to_pi(alvo − heading_w)` | `velocity_command.py:118` |
| O `wz` do fabricante | `wz = clip(heading_control_stiffness · erro; −0,5; +0,5)`, com `heading_control_stiffness` = 1,0 e o teto igual à faixa de giro `ranges.ang_vel_z` = ±`giro_tetos[-1]` = ±0,5 rad/s. Só nos envs com `is_heading_env`. | `velocity_command.py:119-124`, `velocity_command.py:284`, `env_cfg.py:117`, `env_cfg.py:126` |
| O corte | Se `|wz| < giro_min` (0,1), `wz` vira 0: zero ou ≥ `giro_min`, a regra da T15. Com ganho 1, o corte vale para `|erro| < 0,1 rad` = 5,7° (derivado: 0,1 · 180/π). | `comando.py:943-948` |

Com ganho 1, `wz` é o próprio erro entre 0,1 e 0,5 rad. Ele satura em 0,5 rad/s a partir de `|erro|` = 0,5 rad (28,6°). O `wz` salta de 0 para ±0,1 rad/s em `|erro|` = 0,1 rad.

O rumo de partida do ANDAR é uniforme em (−π; π): `Cena.reset_base_andar` tem `yaw` em ±π e `x`, `y` em ±0,5 m (`knobs.py:103`; `eventos.py:211-225`). O alvo é o múltiplo de π/2 mais próximo. O erro inicial é então uniforme em (−π/4; π/4) = ±0,785 rad (derivado: a distância ao múltiplo mais próximo). Daí saem estes números (derivados, conta sobre a distribuição uniforme):
- `|erro|` < 0,1 rad (sem giro de início): 0,2 / (π/2) = 0,127.
- `|erro|` ≥ 0,5 rad (`wz` saturado de início): 2 · (π/4 − 0,5) / (π/2) = 0,363.

O ator lê o `wz` de rumo como o termo `comando` da observação (`vel_command_b`, seção 2). A cópia `vel_command_w` guarda o `wz` = 0 do sorteio e não acompanha o rumo (`comando.py:938`). A seção 6.2 descreve o efeito sobre a nota.

### 4.6 Como o treino roda: o comando no plano e no terreno

Esta seção traz números derivados (conta sobre as fórmulas ou simulação com as funções do código). A simulação roda `sorteia_velocidade` do snapshot do commit `69e8b6b`, com `torch.Generator().manual_seed(0)` novo para cada nível e 200 000 sorteios.

**O comando do plano (tipo `andar`) por nível do ANDAR.** Cada linha vem de 200 000 chamadas de `sorteia_velocidade("andar", n, "ANDAR", ...)`. As colunas são frações dos sorteios (derivado: simulação).

| Nível | Parado (vx, vy, wz = 0) | `|vy| > |vx|` | `|wz| ≥ giro_min` | `vx < 0` | Linear zero (vx, vy = 0) |
|---|---|---|---|---|---|
| 0 | 0,150 | 0,263 | 0,601 | 0,376 | 0,250 |
| 1 | 0,119 | 0,332 | 0,675 | 0,413 | 0,175 |
| 2 | 0,112 | 0,279 | 0,719 | 0,420 | 0,160 |
| 4 | 0,110 | 0,239 | 0,719 | 0,350 | 0,150 |
| 7 | 0,108 | 0,194 | 0,719 | 0,281 | 0,141 |

A coluna "Parado" bate com `P(comando todo zero)` da seção 4.3. A coluna `|wz| ≥ giro_min` inclui os parados como não girando: ela vale `(1 − andar_parado) · (1 − giro_min / giro)`. A fração `vx < 0` sobe de 0,376 no nível 0 até 0,420 no nível 2 e cai a 0,281 no nível 7, porque `tras` fica em 0,5 enquanto `frente` sobe até 1,0. A fração `|vy| > |vx|` cai com o nível pelo mesmo motivo: `lado` fica em 0,3 e `frente` sobe.

**O comando do terreno (tipo `terreno`) por nível do ANDAR.** O ANDAR de terreno só abre no nível 2 (`nivel_abre_terreno`). Os níveis 0 e 1 não sorteiam terreno; as linhas deles servem de comparação. Cada episódio de terreno dura 20 s no máximo. O `twist` reamostra o comando a cada U(3; 8) s (seção 3.1), então um episódio tem de 3 a 7 comandos.

Simulação (derivado): 200 000 episódios por nível. A duração de cada um de 8 segmentos é U(3; 8) s, sorteada antes dos comandos, com o mesmo gerador. O comando de cada segmento vem de `sorteia_velocidade("terreno", n, "ANDAR", ...)`. Os 20 s do episódio cortam o último segmento. Hipóteses: o robô entrega a fração `f` do `vx` pedido, sem demora e em linha reta, e não cai. A base parte da origem da célula. `Terreno.dist_travessia` = 3,5 m (`knobs.py:442`).

Cada fração diz a base. "Por episódio" divide pelos episódios de terreno do nível. "Por visita" divide pelas visitas, a base da taxa que move a linha (seção 4.2). Sem queda e com a partida na origem, a base só passa de 3,5 m se o comando pediu ≥ 3,5 m. Então as visitas são os episódios com pedido ≥ 3,5 m.

| Nível | `vx` máx. (m/s) | Primeiro comando parado | Distância pedida em 20 s (m) | Pedido ≥ 3,5 m, por episódio (visitas sem queda) | Passa de 3,5 m com `f` = 0,7, por episódio | Passa de 3,5 m com `f` = 0,7, por visita | Tempo até 3,5 m com `f` = 1,0 (mediana, entre os que chegam) |
|---|---|---|---|---|---|---|---|
| 0 | 0,3 | 0,201 | 3,47 | 0,514 | 0,055 | 0,107 | 16,1 s |
| 1 | 0,4 | 0,157 | 4,73 | 0,817 | 0,451 | 0,551 | 13,2 s |
| 2 | 0,5 | 0,137 | 5,95 | 0,922 | 0,728 | 0,790 | 10,8 s |
| 4 | 0,7 | 0,119 | 8,38 | 0,978 | 0,922 | 0,943 | 7,7 s |
| 7 | 1,0 | 0,110 | 11,99 | 0,993 | 0,979 | 0,986 | 5,3 s |

Como ler a tabela:
- "Primeiro comando parado" é a fração dos primeiros comandos com vx = 0 (`andar_parado` mais `linear_min`). Ela bate com `0,10 + 0,90 · (linear_min / frente)²` (derivado: conta), por exemplo 0,10 + 0,90 · 0,111 = 0,200 no nível 0. Ela e a distância pedida não dependem de `dist_travessia`.
- "Distância pedida" é `Σ vx · duração` sobre os segmentos do episódio. O código soma `‖cmd_xy‖` por passo em `soma_cmd_marcha` e multiplica por `step_dt`; é a mesma conta (`comando.py:847`, `eventos.py:155`).
- Com `f` = 1,0, a fração dos episódios que passam de 3,5 m em 20 s é igual à fração com pedido ≥ 3,5 m. O episódio acaba na travessia (`terminacoes.py:69-72`), então a duração do episódio de terreno é o tempo da última coluna para os que chegam e 20 s para os outros.
- Com `f` = 0,7, a taxa de travessia por visita no nível 2 é 0,790. Essa taxa passa de `zona_t_alvo` (0,6) e faz a linha do terreno subir (seção 4.2). Por episódio, a fração é 0,728. Nos níveis 0 e 1, a taxa por visita (0,107 e 0,551) fica abaixo de 0,6.
- O comentário de `nivel_abre_terreno` dá os mesmos números por visita: ~79 % no nível 2, ~55 % no nível 1 e ~11 % no nível 0 (`knobs.py:412`).
- Com o deslocamento de partida (x e y em ±0,5 m) e o rumo alvo uniforme nas 4 faces, as frações por episódio mudam até 0,004 nos níveis 1 a 7. No nível 0, com `f` = 0,7, a fração sobe de 0,055 para 0,081 (simulação extra: os mesmos comandos, a partida e a face sorteadas com `torch.Generator().manual_seed(1)`, e a distância lida no fim do episódio).

A visita e a travessia contam no reset, em `eventos.sorteia_fatia` (`eventos.py:143-160`). O arquivo `08-terminacoes-observacoes-e-acao.md`, seção 6.2, descreve a regra e a ordem de leitura da posição.

## 5. O que a `TwistV3` herda da classe do mjlab e o que troca

`TwistV3` estende `UniformVelocityCommand` (`comando.py:689`), que estende `CommandTerm` (`velocity_command.py:25`). A cfg `TwistV3Cfg` estende `UniformVelocityCommandCfg`, liga `heading_command` e zera `rel_heading_envs` e `rel_standing_envs` (`comando.py:676-680`). A cfg passa `ranges` com vx em ±1,0, vy em ±0,3, wz em ±0,5 e `heading` em (−π; π) (`env_cfg.py:117`, `env_cfg.py:124-128`). O fabricante exige `ranges.heading` quando `heading_command` é verdadeiro, e o valor não é usado: o alvo de rumo vem do `TwistV3` (`velocity_command.py:31-34`). O sorteio da `TwistV3` não lê `ranges`. No mjlab, `ranges` é lido na validação do `__init__` (`velocity_command.py:31-34`), no método que a `TwistV3` troca (`velocity_command.py:75-114`), no teto do `wz` de rumo (`velocity_command.py:119-124`) e na GUI do visualizador (`velocity_command.py:153-158`). O projeto lê `ranges` fora do treino, para o envelope de velocidade que o exportador grava (`exporta_cena.py:259-263`). Os outros campos da cfg do mjlab ficam no padrão, porque a `TwistV3Cfg` só troca os três acima: `heading_control_stiffness` 1,0, `rel_world_envs` 0, `rel_forward_envs` 0, `init_velocity_prob` 0 (`velocity_command.py:281-295`). O `time_left`, o `command_counter` e o dicionário `metrics` vêm da classe `CommandTerm` (`command_manager.py:50-58`).

| Peça | Origem | Fica ou troca |
|---|---|---|
| `__init__` | mjlab cria `vel_command_b`, `vel_command_w`, `heading_*`, `is_*_env` e as métricas `error_vel_xy` e `error_vel_yaw` (`velocity_command.py:28-56`) | Fica. A `TwistV3` chama `super().__init__` e acrescenta buffers próprios (`comando.py:700`). |
| `command` | mjlab devolve `vel_command_b` (`velocity_command.py:58-59`) | Fica. |
| `compute` | mjlab: `_update_metrics`, `time_left -= dt`, reamostra os vencidos, `_update_command` (`command_manager.py:109-115`). A subclasse do mjlab acrescenta o joystick da GUI: com ele ligado, sobrescreve `vel_command_b` de um env com os sliders (`velocity_command.py:200-207`). | Fica. |
| `_resample` | mjlab sorteia `time_left`, chama `_resample_command` e incrementa o contador (`command_manager.py:117-123`) | Fica. |
| `reset` | mjlab zera as métricas e o contador e chama `_resample` (`command_manager.py:99-107`) | A `TwistV3` envolve: fecha o segmento, zera os estados, chama `super().reset` e reabre (`comando.py:879-895`). |
| `_update_metrics` | mjlab acumula `error_vel_xy` e `error_vel_yaw` (`velocity_command.py:61-73`) | A `TwistV3` reescreve e chama `super()._update_metrics()` no meio (`comando.py:810-877`, chamada em `comando.py:844`). |
| `_resample_command` | mjlab sorteia uniforme em `ranges`, com rumo, parado, mundo, só-frente e velocidade inicial (`velocity_command.py:75-114`) | Troca por inteiro. A `TwistV3` não chama o `super` (`comando.py:897-941`). |
| `_update_command` | mjlab aplica rumo, referencial de mundo e envs parados (`velocity_command.py:116-140`) | A `TwistV3` chama o `super` e depois zera o `wz` de rumo menor que `giro_min` (`comando.py:943-948`). Só o rumo age: `is_standing_env` e `is_world_env` ficam falsos. |

Consequência: o código da `TwistV3` usa só o rumo do mjlab (seção 4.5) e nenhum dos mecanismos abaixo.
- Envs parados do mjlab: `is_standing_env` nunca é sorteado, porque `_resample_command` do mjlab não roda (`velocity_command.py:84`). O comando parado da `TwistV3` é sorteado em `sorteia_velocidade` (`comando.py:146`) ou vem da tarefa.
- Referencial de mundo: `is_world_env` nunca é sorteado (`velocity_command.py:87`).
- Só-frente: `is_forward_env` nunca é sorteado (`velocity_command.py:92`).
- Velocidade inicial no reset (`init_velocity_prob`, `velocity_command.py:101-114`): não roda.

## 6. O segmento

Um segmento é o intervalo em que o comando fica igual. A fronteira é a mudança de `command_counter` desde o último passo (`comando.py:818`).

### 6.1 Quando abre e quando fecha

| Evento | Fecha o segmento antigo? | Abre o novo? | Referência |
|---|---|---|---|
| Criação do env | Não | Sim, para todos os envs | `comando.py:733` |
| Reset do episódio | Sim, com `falha = ~time_out` | Sim, depois do sorteio do reset | `comando.py:882-894` |
| `command_counter` mudou em um passo | Sim, com `falha = v3_perdeu & seg_carrega` | Sim | `comando.py:818-821` |

Em um reset, `time_out = termination_manager.time_outs[env_ids]` (`comando.py:882`). Qualquer terminação que não seja `time_out` vale como falha (queda, caixa perdida com a chave desligada, chegada, batente) (`env_cfg.py:138-146`).

No passo, o `twist` detecta a mudança do contador no começo de `_update_metrics` (`comando.py:818-821`). Quando a `cadeia` ou a caixa perdida reamostram, isso acontece antes do `twist` no mesmo `compute`, e o `twist` vê a mudança no mesmo passo. Quando o temporizador do próprio `twist` reamostra, isso acontece depois de `_update_metrics` (`command_manager.py:110-114`), e o `twist` vê a mudança no passo seguinte.

### 6.2 O que `_abre` grava

`_abre(ids)` roda na abertura (`comando.py:735-751`):
- `t_ab` = `_agora` (instante da abertura).
- `seg_cmd` = `vel_command_b` (cópia do comando). Nos envs de rumo (`is_heading_env`), o `wz` de `seg_cmd` vira 0 (`comando.py:742`). Motivo: o `wz` de rumo muda a cada passo, e a nota cobraria um giro que o robô corrige em menos de 1 s. O canal de giro do segmento fica "sem comando" e a nota cobra a deriva dele (seção 8.1). O `_resample_command` liga o `is_heading_env` antes do `_abre`, no reset e na troca de segmento (`comando.py:932`). O `_abre` da troca de segmento por temporizador roda no passo seguinte, depois do `_update_command`; sem o zero, a cópia levaria o `wz` do passo (`comando.py:741-742`).
- `seg_cel` = `v3_celula` (célula do env, −1 sem célula).
- `seg_carrega` = `v3_tarefa != ANDAR` (verdadeiro nas tarefas com caixa).
- `seg_visto` = `command_counter`.
- Zeram: `seg_proj`, `seg_int`, `seg_nota`, `seg_dur` e `ancora_feita`.
- `v3_incl_max_seg` volta a 0.

`_agora` é o relógio do episódio do próprio `twist`: soma `step_dt` por chamada de `_update_metrics` (`comando.py:812-813`) e zera no reset (`comando.py:884`). A soma usa `env.step_dt`, e não o argumento `dt` do `compute`.

### 6.3 O que se acumula a cada passo

Em cada `_update_metrics`, o `twist` lê `vel` = `root_link_lin_vel_b[:, :2]`, `wz` = `root_link_ang_vel_b[:, 2]` e `cmd` = `vel_command_b` (`comando.py:823-825`). Todas as somas abaixo usam `dt = env.step_dt`.

| Acumulador | Fórmula por passo | Desde | Referência |
|---|---|---|---|
| `seg_proj[0]` | `Σ (vel · cmd_xy / max(‖cmd_xy‖; 1e-6)) · dt` | abertura | `comando.py:826-827` |
| `seg_int[0]` | `Σ ‖cmd_xy‖ · dt` | abertura | `comando.py:828` |
| `seg_proj[1]` | `Σ wz · sinal · dt`, com `sinal` = +1 se `cmd_z ≥ 0`, senão −1 | abertura | `comando.py:829-830` |
| `seg_int[1]` | `Σ |cmd_z| · dt` | abertura | `comando.py:831` |
| `seg_nota` | `Σ (vx, vy, wz) · dt · apos` | âncora | `comando.py:854-856` |
| `seg_dur` | `Σ dt · apos` | âncora | `comando.py:857` |
| `v3_incl_max_seg` | máximo de `rad2deg(inclinacao_da_caixa(quat da caixa))` | abertura | `comando.py:753-757`, `comando.py:832` |

`apos` vale 1 quando `_agora ≥ v3_ancora[:, 3]` e 0 caso contrário (`comando.py:855`).

## 7. A âncora e o efeito do empurrão

A âncora é o ponto de referência da deriva. Ela guarda a pose (x, y, rumo no mundo) e o instante, em `v3_ancora` de 4 colunas (`curriculo.py:29`; `comando.py:836-842`).

Instante da âncora: `max(t_ab, v3_t_empurrao) + ancora_s` (`tempo_da_ancora`, `comando.py:221-224`; chamada em `comando.py:834`). `Comando.ancora_s` = 1,5 s (`knobs.py:145`).

A cada passo, o `twist` (`comando.py:833-842`):
1. Recalcula o instante `novo`.
2. Mantém `ancora_feita` só nos envs em que `novo` é igual ao instante guardado. Um instante novo regrava a pose (`comando.py:835`).
3. Grava o novo instante em `v3_ancora[:, 3]`.
4. Para os envs com `ancora_feita` falso e `_agora ≥ novo`, grava `root_link_pos_w[:, :2]` e `heading_w` em `v3_ancora[:, :3]` e põe `ancora_feita` em verdadeiro.

O empurrão afeta a âncora (o código tem esse mecanismo):
- O evento `empurrao` é de intervalo (`env_cfg.py:200-205`). Ele chama `push_by_setting_velocity` e depois grava `v3_t_empurrao = episode_length_buf · step_dt` nos envs empurrados (`eventos.py:261-269`).
- O mjlab soma à velocidade da base, no referencial do mundo, um valor uniforme em ±`DR.empurrao` = ±0,5 m/s em x e em y (`env_cfg.py:204-205`; `knobs.py:517`; `events.py:316-337`). Não há empurrão no modo `play` (`env_cfg.py:200`).
- O intervalo do empurrão é por env, uniforme em `DR.periodo_empurrao_s` = (3,0; 7,0) s (`knobs.py:520`; `env_cfg.py:203`). O primeiro instante de cada env é sorteado uma vez, na criação do gerenciador de eventos (startup), na mesma faixa (`event_manager.py:371-375`). Em cada passo, depois do `compute` dos comandos, o contador cai `step_dt`, e o evento dispara nos envs com contador < 1e-6, que sorteiam o próximo intervalo na hora (`event_manager.py:265-268`, `event_manager.py:278-287`). O reset do episódio não reinicia esse contador, porque `eventos.empurrao` é função e não classe (`event_manager.py:225-233`, `event_manager.py:359-360`). O primeiro empurrão de um episódio cai, portanto, em qualquer ponto entre 0 e 7 s do começo, conforme o contador herdado.
- No reset, `v3_t_empurrao` volta a −∞ (`eventos.py:163`). Sem empurrão, vale a abertura.
- Depois de um empurrão, `novo = t_empurrao + ancora_s` passa a ser maior que o instante guardado. O `twist` marca a âncora como não feita e regrava a pose quando `_agora` o alcança (`comando.py:834-842`).
- Enquanto `_agora` está antes da nova âncora, `apos` vale 0, e `seg_nota` e `seg_dur` não acumulam (`comando.py:855-857`). O que já foi acumulado antes do empurrão fica. `seg_proj` e `seg_int` seguem acumulando desde a abertura (`comando.py:826-831`).
- O empurrão não fecha o segmento nem muda o comando. O segmento continua o mesmo (nenhum código do empurrão toca `command_counter`, `seg_*` ou `vel_command_b`).
- O instante da âncora entra também na regra de registro dos segmentos com caixa: `registra = t_fecho ≥ t_ancora` (`comando.py:241`). Um empurrão perto do fecho adia a âncora e pode deixar o segmento fora do currículo, salvo falha. O ANDAR não usa essa regra (seção 8.1).
- O empurrão que cai antes da abertura do segmento não conta: `max(t_ab, t_empurrao)` escolhe a abertura.

Escala de tempo: `t_empurrao` usa `episode_length_buf · step_dt` (`eventos.py:269`), e a âncora é comparada com `_agora`. No passo do reset, o `compute` roda com `episode_length_buf` = 0 e já soma um `step_dt` em `_agora` (`manager_based_rl_env.py:590`, `manager_based_rl_env.py:456`, `comando.py:813`). Assim, `_agora` vale um `step_dt` a mais que `episode_length_buf · step_dt` no mesmo passo. O evento `empurrao` roda depois do `compute` (`manager_based_rl_env.py:456-461`) e grava `t_empurrao` = `episode_length_buf · step_dt` desse passo. Com isso, a pose da âncora do empurrão é gravada quando `_agora ≥ t_empurrao + ancora_s`, o que ocorre no passo `n + 74` e não em `n + 75`, para um empurrão no passo `n` (`ancora_s` / `step_dt` = 1,5 / 0,02 = 75). Esse número depende do arredondamento do float32 em `_agora`.

## 8. A nota do segmento

O fecho calcula uma nota `x` e um indicador de registro `reg` (`comando.py:759-808`). O cálculo muda conforme `seg_carrega`.

Primeiro, o fecho mede a deriva (`comando.py:763-766`):
- `dpos` = `‖pose_ult[:2] − ancora[:2]‖`
- `drum` = `|wrap_to_pi(pose_ult[2] − ancora[2])|`
- `pose_ult` é a pose (x, y, rumo no mundo) do fim do último passo (`comando.py:814-815`). Ela é gravada no começo de `_update_metrics`. No reset, serve de pose do fim do episódio, porque o reset do mjlab já teletransportou o robô (`comando.py:730-732`).

### 8.1 Segmentos de ANDAR (`seg_carrega` falso)

A nota é a média de dois canais (linear e giro) (`comando.py:780-789`):

- `c` = `seg_cmd`; `v̄` = `seg_nota / max(seg_dur; 1e-6)` (velocidade média desde a âncora, vetor (v̄x, v̄y, v̄z)).
- `den = (‖c_xy‖; |c_z|)`. Canal ativo: `den ≥ limiar_canal`. `Comando.limiar_canal` = 0,02 (`knobs.py:138`). Depois, `den` recebe `clamp(min = limiar_canal)`.
- Projeção: linear `p_lin = (v̄_xy · c_xy) / den_lin`; giro `p_giro = v̄_z · sinal(c_z)`.
- Razão: `razão = clamp(p / den; 0; 1)` por canal. É `(v̄·ĉ) / ‖c‖` no linear e `ω̄ / c_z` no giro. A razão negativa vale 0 e o excesso vale 1.
- Deriva, para canal sem comando: `exp(−‖v̄_xy‖² / σ²)` no linear e `exp(−v̄_z² / σ²)` no giro, com `σ` = `Recompensa.sigma_rastreio` = 0,5 (`knobs.py:238`; `comando.py:787`).
- `nota = média(onde(ativo; razão; deriva))` sobre os dois canais.
- `x = nota · (1 − falha)`. A queda faz `x = 0`.
- `reg = (seg_dur > 0) | falha`: o segmento registra se teve tempo depois da âncora ou se terminou em falha (`comando.py:790`). O ANDAR não usa `t_ancora` para decidir o registro.

Um segmento de ANDAR com comando zero tem os dois canais inativos. A nota é a média das duas derivas. Um segmento de ANDAR de terreno com comando linear tem o canal linear ativo e o canal de giro inativo, porque `seg_cmd` tem `wz` = 0 (seção 6.2). A nota é a média da razão linear e da deriva de giro, `exp(−v̄_z² / σ²)`. O giro de alinhamento ao rumo entra em `v̄_z`. Com `v̄_z` = 0,1 rad/s a deriva vale 0,96, e com 0,25 rad/s vale 0,78 (derivado: conta sobre a fórmula, com `σ` = 0,5). Os acumuladores `seg_proj[1]` e `seg_int[1]` leem o `vel_command_b` do passo, e por isso incluem o `wz` de rumo. Só os segmentos com caixa usam esses dois (seção 8.2), e o ANDAR não os usa.

### 8.2 Segmentos com caixa (`seg_carrega` verdadeiro)

A nota é 0 ou 1, vinda de `sucesso_do_segmento` (`comando.py:227-242`):

- Canal linear ativo: `‖c_xy‖ ≥ limiar_canal`. Canal de giro ativo: `|c_z| ≥ limiar_canal` (`comando.py:233-234`).
- Razão por canal: `seg_proj / max(seg_int; 1e-6)` (`comando.py:235`). A razão do linear é `Σ(v·ĉ)dt / Σ‖c‖dt`. A razão do giro é `Σ wz·sinal·dt / Σ|c_z|dt`.
- `ok_lin` = canal ativo ? `razão_lin ≥ razao_portao` : `dpos < lim_pos` (`comando.py:237`).
- `ok_giro` = canal ativo ? `razão_giro ≥ razao_portao` : `drum < deriva_rumo_rad` (`comando.py:238`).
- `sucesso = ok_lin & ok_giro & ~(carregar & perdeu)` (`comando.py:239`). `perdeu` é `v3_perdeu` do env no fecho. No fecho do reset, a `cadeia` já zerou `v3_perdeu` (`comando.py:651-653`), então `~(carregar & perdeu)` não pesa ali; só a `falha` (`~time_out`) zera a nota (`comando.py:789`, `comando.py:883`).
- Terreno: se a célula do segmento é `CARREGAR/terreno` e `v3_incl_max_seg > incl_max_terreno_graus`, o sucesso vira falso (`comando.py:240`; terreno definido em `comando.py:767-768`).
- `reg = t_fecho ≥ t_ancora | falha`, com `t_fecho = _agora` e `t_ancora = v3_ancora[:, 3]` (`comando.py:241`, `comando.py:771`, `comando.py:790`). Um segmento que fecha antes da âncora não entra no currículo, a não ser que termine em falha.
- `x = sucesso · (1 − falha)` (`comando.py:789`).

Knobs usados:

| Knob | Valor | Linha |
|---|---|---|
| `Comando.razao_portao` | 0,50 | `knobs.py:146` |
| `Comando.deriva_pos_inicio` | 0,25 m | `knobs.py:143` |
| `Comando.deriva_pos_m` | 0,10 m | `knobs.py:142` |
| `Comando.deriva_rumo_rad` | 0,25 rad | `knobs.py:144` |
| `Comando.incl_max_terreno_graus` | 15,0 graus | `knobs.py:149` |
| `Comando.limiar_canal` | 0,02 | `knobs.py:138` |

O limite de deriva de posição `lim_pos` muda com o progresso. Ele vale `no_degrau(deriva_pos_inicio, deriva_pos_m, restricao, 1)` = `0,25 + (0,10 − 0,25) · restricao` (`comando.py:773`; `balanceador.py:48-51`). `restricao` é o menor entre os quocientes `nível / nível_máximo` das 4 tarefas (`balanceador.py:345-348`). Com `restricao` = 0, o limite é 0,25 m. Com `restricao` = 1, é 0,10 m.

## 9. Registro no currículo e tratamento da queda

### 9.1 O que se registra e quando

A cada fecho, o `twist` registra no currículo, por célula, a contagem e a soma das notas dos segmentos com `reg` verdadeiro e `seg_cel ≥ 0` (`comando.py:791-794`). A chamada é `v3_cur.registra(célula, n, soma)` (`balanceador.py:148-151`).

- Só as células de ANDAR (`andar`, `terreno`) e de CARREGAR (`reto`, `giro_lugar`, `arco`, `terreno`) recebem registro do `twist`. Segmentos sem célula (`seg_cel = −1`: comando zero por tarefa, PEGAR, BOTAR, ANDAR de espera) não entram (`comando.py:901`, `comando.py:792`).
- O registro só soma em acumuladores. A nota da célula muda em `fecha_iteracao`, uma vez por iteração do PPO: com `x = soma / n` das medidas da iteração, `s = x` se a célula não tem medida anterior, senão `s += alfa · (x − s)` (`balanceador.py:153-168`). `Balanceador.alfa` = 0,05 (`knobs.py:394`). Uma iteração sem registro (`n = 0`) não muda `s` (`balanceador.py:164-167`). Quando a tarefa sobe de nível ou de linha de terreno, `n_medidas` das células dela volta a 0, e a primeira medida seguinte vira `s` (`balanceador.py:269-278`).

### 9.2 Queda e perda da caixa

| Situação | Tratamento | Referência |
|---|---|---|
| Terminação que não é `time_out` (queda e outras) | No reset, `_fecha(env_ids, falha = ~time_out)`. Para ANDAR, `x = 0` e registra mesmo sem tempo depois da âncora. Para segmento com caixa, `x = 0` e registra mesmo antes da âncora. | `comando.py:882-883`, `comando.py:789-790`, `comando.py:241` |
| `time_out` | `falha` falso. O segmento fecha pela regra normal (`seg_dur > 0` no ANDAR; `t_fecho ≥ t_ancora` com caixa). | `comando.py:882-883` |
| Caixa perdida no meio do episódio (a chave `caixa_perdida_vira_andar` = verdadeiro, `knobs.py:161`) | A `cadeia` marca `v3_perdeu` e publica ANDAR (`comando.py:412-420`). O `twist` reamostra. O contador muda e o segmento antigo fecha com `falha = v3_perdeu & seg_carrega`: nota 0. O currículo recebe esse zero só se o segmento tinha célula (`seg_cel ≥ 0`), o que vale para o CARREGAR da fatia B. Segmentos de PEGAR e BOTAR têm `seg_cel` = −1 e não entram. | `comando.py:818-820`, `comando.py:791-792` |
| Segmento aberto depois da perda | `seg_carrega` falso (tarefa ANDAR) e comando zero. A nota segue a regra do ANDAR, mas a célula é −1 e o currículo não a recebe. | `comando.py:744`, `comando.py:901`, `comando.py:905` |

A terminação `time_out` é a de fim da cadeia: `episode_length_buf ≥ max_episode_length`, ou `v3_fim`, ou a travessia do terreno no ANDAR (`env_cfg.py:138`; `terminacoes.py:53-72`). A travessia é um `time_out`: o segmento fecha pela regra normal, sem `falha` (`comando.py:882-883`). A seção 6.2 do arquivo `08-terminacoes-observacoes-e-acao.md` descreve a condição. O tamanho do episódio é `Cadeia.episodio_s` = 20,0 s (`env_cfg.py:105`). A terminação `caixa_perdida` só termina o episódio com a chave desligada (`terminacoes.py:75-78`, `env_cfg.py:141-143`).

## 10. Filtros que a classe mantém

Os três estados abaixo moram na `TwistV3` e zeram no reset do episódio (`comando.py:885-888`). O passo de tempo é `dt = env.step_dt` (`comando.py:812`). A constante de tempo é `Comando.tau_fracao` = 0,4 s (`knobs.py:136`). O fator por passo é `α = 1 − exp(−dt / tau_fracao)` (`−math.expm1(−dt / k.tau_fracao)`, `comando.py:858`, `comando.py:870`). Com `dt` = 0,02 s e `tau_fracao` = 0,4 s, `α` = 0,0488.

| Filtro | Fórmula por passo | Reset | Referência |
|---|---|---|---|
| `v_filt` (velocidade filtrada) | `v_filt += α · (v3 − v_filt)`, com `v3 = (vel_x, vel_y, wz)` da base | `v_filt = 0` | `comando.py:854`, `comando.py:858`, `comando.py:886` |
| `q_filt` (acerto do relógio da marcha filtrado) | `q_filt += α · (acerto − q_filt)` | `q_filt = 0,5` | `comando.py:870`, `comando.py:887` |
| `g_marcha` (fator g) | `g = clamp(2 · q_filt − 1; 0; 1)` | `g = 0` | `comando.py:871`, `comando.py:888` |

Todos rodam em todo passo e em todo env, com ou sem comando.

`v_filt` e `g_marcha` são lidos pela recompensa de rastreio: `v̄ = v_filt` na projeção da tenda dos canais com comando, e a tenda é multiplicada por `g_marcha` (`recompensas.py:399`, `recompensas.py:407-409`). Desde a T20 (27/11), a velocidade perpendicular ao comando linear sai da velocidade instantânea, e não da filtrada (`recompensas.py:403-404`). O canal de giro com comando usa só a projeção filtrada (`recompensas.py:406`). `g_marcha` também multiplica o sinal comandado de uma renda do progresso (`recompensas.py:863-865`). O segmento não usa `v_filt`: ele usa `seg_nota` (seção 8).

Acerto do relógio (`comando.py:863-871`):
- `contato = scene["pes_chao"].data.found > 0`, um valor por pé (`comando.py:867`).
- `acerto = n_passo(contato, v3_fase_marcha, fase_apoio, on).mean(−1)`, a média dos dois pés, gravada em `v3_acerto_marcha` (`comando.py:868`).
- `n_passo` põe o pé esquerdo na fase `f` e o direito em `(f + 0,5) mod 1`. O alvo de apoio de cada pé é `fase < fase_apoio` ou `~on`. O acerto de um pé vale 1 quando `contato == alvo` e 0 caso contrário (`recompensas.py:75-83`). `Comando.fase_apoio` = 0,55 (`knobs.py:139`).
- O acerto usa a fase do passo anterior (`v3_fase_marcha` ainda não foi atualizada neste passo) e o contato e o `on` de agora.
- Reinício de `q_filt`: nos envs com `on` agora e `v3_marcha_on` falso no passo anterior (transição de parado para comando), `q_filt` volta a 0,5 antes do filtro (`comando.py:869`).

O reset do `twist` não zera `v3_fase_marcha`, `v3_marcha_on` nem `v3_acerto_marcha`. Os três são sobrescritos em `_update_metrics` (`comando.py:868`, `comando.py:873-874`). No primeiro `_update_metrics` de um episódio, o acerto usa a fase e o `on` que o env tinha no fim do episódio anterior (`comando.py:868`). Nos arquivos lidos, nenhum outro código zera esses buffers.

## 11. O relógio da marcha

O relógio da marcha é uma fase em [0; 1) que avança só quando há comando (`comando.py:859-874`).

- `on = ‖vel_command_b‖ > limiar_marcha`, a norma dos três canais, com comparação estrita (`comando.py:863`). `Comando.limiar_marcha` = 0,1 (`knobs.py:141`).
- `_fase_t += dt · on`: o tempo com comando (`comando.py:872`). Com comando zero, `_fase_t` para onde está e volta dali quando o comando volta.
- `v3_fase_marcha = (_fase_t / periodo_marcha) mod 1` (`comando.py:873`). `Comando.periodo_marcha` = 0,8 s (`knobs.py:140`). O período é fixo.
- `v3_marcha_on = on` (`comando.py:874`).
- `_fase_t` zera no reset (`comando.py:885`).
- O comando lido é `vel_command_b` no começo de `_update_metrics`. A reamostra pelo temporizador do próprio `twist` acontece depois dessa leitura (`command_manager.py:110-114`), e o `on` correspondente aparece no passo seguinte.
- O relógio roda depois das recompensas do passo. A recompensa `passo` do passo `t` lê a fase do passo `t − 1` (`recompensas.py:631-633`; `manager_based_rl_env.py:439-456`).
- A fase chega ao ator como seno e cosseno de `2π · v3_fase_marcha` (`observacoes.py:313-315`).
- A recompensa `passo` recalcula `on` com o comando do passo corrente e chama `n_passo` com `v3_fase_marcha`. O valor sai multiplicado por `_avanco` (T19). O `on` usa a norma dos três canais, então o `wz` de rumo conta (`recompensas.py:631-633`).

## 12. Métricas que a classe publica

As métricas ficam em `self.metrics`. No reset, `CommandTerm.reset` publica a média sobre os envs que terminaram e zera os valores (`command_manager.py:99-104`). O `CommandManager` as nomeia `Metrics/twist/<nome>` (`command_manager.py:241-247`), e elas entram em `extras["log"]` (`manager_based_rl_env.py:581-582`). Um segmento que fecha no reset atualiza as métricas de eficiência antes da leitura da média (`comando.py:883`, `comando.py:893`).

| Métrica | Fórmula | Referência |
|---|---|---|
| `error_vel_xy` | `Σ ‖cmd_xy − vel_xy‖ / max_command_step` por passo; `max_command_step = reamostra_s[1] / step_dt` = 8,0 / 0,02 = 400 | `velocity_command.py:61-73`, `knobs.py:137` |
| `error_vel_yaw` | `Σ |cmd_z − wz| / max_command_step` por passo | `velocity_command.py:61-73` |
| `razao_marcha` | `1 − Σ(‖cmd_xy − vel_xy‖ · ativo) / Σ(‖cmd_xy‖ · ativo)`, com `ativo = ‖cmd_xy‖ > limiar_canal`; 0 se a soma do denominador é 0. Escrita a cada passo | `comando.py:845-850` |
| `acerto_relogio` | `Σ(acerto · on) / max(Σ on; 1)`, com `acerto = v3_acerto_marcha` | `comando.py:875-877` |
| `segmentos` | Número de segmentos do episódio com `Σ‖c_xy‖dt ≥ pedido_min_segmento` (`vale`) | `comando.py:797-804` |
| `eficiencia_media` | Média, sobre os segmentos `vale`, de `e = seg_proj[0] / max(seg_int[0]; 1e-6)` | `comando.py:797-803` |
| `eficiencia_min` | Mínimo de `e` sobre os segmentos `vale`; o primeiro `vale` define o valor inicial | `comando.py:797-802` |

`Comando.pedido_min_segmento` = 0,5 (`knobs.py:147`). A eficiência é diagnóstico: não entra na nota nem na recompensa (`comando.py:795-796`).

⚠ Divergência: o comentário de `pedido_min_segmento` diz "duração mínima do segmento (s)" (`knobs.py:147`). O código compara `Σ‖c_xy‖·dt` (unidade m, a integral do módulo do comando) com o knob (`comando.py:797-798`). Um segmento de 1 s com comando de 0,5 m/s dá 0,5 e passa; um de 5 s com comando de 0,1 m/s também dá 0,5.

Os acumuladores `soma_erro_marcha`, `soma_cmd_marcha`, `soma_acerto` e `passos_on` são atributos da classe, fora de `metrics`, e zeram no reset (`comando.py:705-708`, `comando.py:889-892`).

## 13. Ordem das operações em `_update_metrics`

Cada chamada (um por passo, para todos os envs) executa, nesta ordem (`comando.py:810-877`):

| Bloco | O que faz | Linhas |
|---|---|---|
| 0 | Soma `step_dt` em `_agora`; grava `pose_ult` | `comando.py:812-815` |
| 1 | Detecta fronteira (`command_counter` mudou); fecha e abre o segmento | `comando.py:816-821` |
| 2 | Acumula `seg_proj`, `seg_int` e `v3_incl_max_seg` | `comando.py:822-832` |
| 3 | Atualiza o instante e a pose da âncora | `comando.py:833-842` |
| 4 | Métricas do mjlab, `razao_marcha`; soma `seg_nota` e `seg_dur`; filtra `v_filt` | `comando.py:843-858` |
| 5 | Acerto do relógio, `q_filt`, `g_marcha`, fase e `on` | `comando.py:859-877` |

O `reset` do `twist` roda em outra ordem (`comando.py:879-895`): fecha o segmento, zera `_agora`, `_fase_t`, `v_filt`, `q_filt`, `g_marcha` e as somas, chama `super().reset` (que publica e zera as métricas, zera o contador, e sorteia o comando) e abre o segmento novo.

## 14. Outros comportamentos do código

- O reset da `cadeia` roda antes do reset do `twist`, porque o dicionário de comandos põe a `cadeia` primeiro (`env_cfg.py:118-128`; `command_manager.py:241-247`). O reset da `cadeia` zera `v3_perdeu` e os outros flags de caixa (`comando.py:651-653`).
- O `compute(dt=0.0)` do `reset` público do env chama `_update_metrics`, que soma `step_dt` em `_agora` mesmo assim (`manager_based_rl_env.py:372`; `comando.py:812-813`).
- Quando o viewer liga o joystick, o `compute` herdado sobrescreve `vel_command_b` do env escolhido com os sliders, depois de `_update_command` (`velocity_command.py:200-207`).
- No trecho CARREGAR da fatia B, se o comando é diferente de zero, `_abre_trecho` chama `eventos.laje_fora` para tirar a laje do caminho (`comando.py:592-594`).
- A recompensa `altura_pe` e outras leem o comando do `twist` com limiar próprio. Elas ficam fora deste documento.

## 15. Não determinado pelo código lido

- Semente do `v3_gerador`: `curriculo.py:62-64` usa `cfg.seed` ou 0 quando `cfg.seed` é `None`. O mjlab define `seed` como `None` por padrão (`manager_based_rl_env.py:94`). Não determinado pelo código lido: se o `cfg.seed` já tem valor quando `garante_v3` cria o gerador, e qual valor o script de treino da v3 passa.
- Quando duas terminações (um `time_out` e uma queda) valem no mesmo passo, o `time_outs[env_ids]` que define a `falha` do segmento depende da conta do `termination_manager` (`comando.py:882`). Não determinado pelo código lido: essa conta fica no mjlab e não foi lida.
