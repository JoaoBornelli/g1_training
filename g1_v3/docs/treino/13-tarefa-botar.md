# Tarefa BOTAR

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:

- `g1_v3/`: `env_cfg.py`, `tarefa.py`, `knobs.py`, `comando.py`, `alvos.py`, `potencial.py`, `eventos.py`, `balanceador.py`, `curriculo.py`, `recompensas.py`, `terminacoes.py`, `observacoes.py`, `estimador.py`, `metricas.py`, `cena.py`, `algoritmo.py`, `runner.py`, `tarefa.py`, `percepcao_lidar.py`.
- `mjlab/`: `envs/manager_based_rl_env.py`, `envs/mdp/events.py`, `envs/mdp/observations.py`, `envs/mdp/dr/geom.py`, `envs/mdp/dr/joint.py`, `envs/mdp/dr/actuator.py`, `actuator/actuator.py`, `actuator/builtin_group.py`, `managers/event_manager.py`, `managers/reward_manager.py`, `managers/termination_manager.py`, `managers/command_manager.py`, `managers/observation_manager.py`, `managers/metrics_manager.py`, `managers/curriculum_manager.py`, `utils/buffers/delay_buffer.py`, `tasks/velocity/mdp/velocity_command.py`, `tasks/velocity/config/g1/rl_cfg.py`.
- `rsl_rl/`: `utils/logger.py` (trecho do log de extras).

BOTAR é a tarefa em que o robô leva a caixa a um ponto de soltura sobre uma laje de destino, apoia a caixa na laje, afasta as mãos e volta à pose de pé (`comando.py:73-88`). Este documento descreve o que o código faz com um env nessa tarefa, do reset ao fim do episódio, e como a tarefa progride no treino.

Duas unidades de tempo valem em todo o documento.

- Um passo da política dura `Contrato.dt_fisica` (0,005 s, `knobs.py:54`) vezes `Contrato.decimacao` (4, `knobs.py:55`), ou seja 0,02 s (`mjlab/envs/manager_based_rl_env.py:266-269`).
- Uma iteração do PPO tem `Balanceador.passos_por_iteracao` (24, `knobs.py:402`) passos por env (`tarefa.py:49`).

---

## 1. O que é a tarefa e quando um env está nela

### 1.1 Peças usadas no documento

| Peça | O que é | Onde |
|---|---|---|
| env | Uma cópia paralela do simulador: um robô, uma caixa, uma laje. | `mjlab/envs/manager_based_rl_env.py:166` |
| tarefa | O rótulo que o env publica em `v3_tarefa`: ANDAR, PEGAR, CARREGAR ou BOTAR. Ele vira um one-hot de 6 canais e escolhe quais recompensas e terminações valem. | `knobs.py:42`; `observacoes.py:135-137`; `curriculo.py:93-100` |
| fatia | O grupo de tarefas que um env recebe no reset. Há três: ANDAR, B e C. | `knobs.py:43` |
| trecho | Um degrau da cadeia da fatia. Cada trecho publica uma tarefa e acaba por tempo ou por aviso. | `comando.py:163-194` |
| etapa (alvo) | O alvo corrente dentro do PEGAR e do BOTAR: P0, P1, P2, P3, B1, B2, B3. O índice vai de 0 a 6; −1 quer dizer "sem alvo". | `knobs.py:44`; `alvos.py:16` |
| aviso | Um latch (valor que fica ligado até o trecho fechar) que marca o fecho da etapa final do trecho: P3 no PEGAR, B3 no BOTAR. | `comando.py:604-607` |
| célula | A unidade do balanceador que acumula a nota de um tipo de episódio. O BOTAR tem uma célula só, `botar`. | `balanceador.py:22-24` |
| zona z | Escala em [0, 1] do critério de fecho de cada etapa. Com z = 1 o critério é largo; com z = 0 é o final. | `balanceador.py:105` |
| nível, subnível | Posição da tarefa na tabela de dificuldade. O BOTAR tem 5 níveis; o nível 0 tem 2 subníveis. | `balanceador.py:72-74,90-92` |
| degrau | Posição da tarefa no refino (0 a 5). Cada tarefa tem o seu; o degrau global é o menor dos quatro. | `balanceador.py:99,341-343` |

### 1.2 Quando o env está na tarefa BOTAR

O env está na tarefa BOTAR quando `v3_tarefa` vale BOTAR. Isso só acontece em episódios da fatia C. A tabela abaixo mostra a cadeia de trechos da fatia C (`comando.py:178-179`).

| # | Tarefa publicada | Como o trecho acaba | Duração ou atraso (knob) |
|---|---|---|---|
| 1 | ANDAR (espera) | Por tempo | U(0,5; 1,0) s: `Cadeia.espera_s` (`knobs.py:154`) |
| 2 | PEGAR | Por aviso: o fecho do P3 e depois o atraso | Atraso U(0,0; 1,0) s: `Cadeia.atraso_pegar_s` (`knobs.py:158`) |
| 3 | CARREGAR | Por tempo | U(0,5; 1,0) s: `Cadeia.carregar_c_s` (`knobs.py:156`) |
| 4 | **BOTAR** | Por aviso: o fecho do B3 e depois o atraso | Atraso U(0,0; 1,5) s: `Cadeia.atraso_botar_s` (`knobs.py:159`) |
| 5 | ANDAR (final) | Por tempo; ao acabar, fim da cadeia | 1,5 s: `Cadeia.andar_final_s` (`knobs.py:157`) |

O sorteio da duração ou do atraso de cada trecho é uniforme: `lo + (hi − lo)·u`, com u em [0, 1), uma vez na abertura do trecho (`comando.py:578-579`).

O env **entra** na tarefa BOTAR quando o trecho 3 acaba. O trecho 3 acaba quando o relógio dele chega à duração sorteada (`avanca_trecho`, `comando.py:209-210`). Nessa abertura `_abre_trecho` publica BOTAR (`comando.py:584`), registra o PEGAR pendente (`comando.py:588-589`) e abre o B1 (`comando.py:590-591`).

O env **sai** da tarefa BOTAR por um destes caminhos.

| Saída | Condição exata | Efeito sobre a tarefa |
|---|---|---|
| Caixa perdida | `perda` em `_perda` (seção 8.2) com `Cadeia.caixa_perdida_vira_andar` = True (`knobs.py:161`) | A tarefa publicada vira ANDAR; o trecho congela (`comando.py:412-420,610-613`) |
| Fim do trecho BOTAR | `v3_aviso` ligado e relógio ≥ atraso sorteado (`comando.py:210-211`) | O trecho 5 abre e a tarefa vira ANDAR |
| Fim do episódio | Terminação ou `time_out` (seção 8) | O env faz reset; a tarefa recomeça pelo trecho 1 |

Entre o fecho do B3 e o fim do atraso a tarefa continua BOTAR e o índice do alvo vale −1 (`alvos.py:206-208`).

### 1.3 Sorteio da fatia e da tarefa

A tarefa do env **não é sorteada**. A ordem dos trechos é fixa por fatia (`comando.py:175-180`). O que se sorteia é a fatia, no reset, e as durações dos trechos.

A fatia sai de `torch.multinomial` sobre o vetor de frações, com reposição, uma amostra por env que faz reset (`balanceador.py:302-304`; `eventos.py:161`). O vetor vem de `fracoes_fatias` (`balanceador.py:282-293`).

1. Uma fatia está aberta quando tem ao menos uma célula aberta. A fatia ANDAR abre no começo (`balanceador.py:26-27,110`). A fatia B tem as células `CARREGAR/reto` e `PEGAR/pegar`. A fatia C tem a célula `BOTAR/botar` (`balanceador.py:34`).
2. O teto de cada fatia aberta: 1,0 para o ANDAR; `Balanceador.rampa_fatia_inicio` (0,05, `knobs.py:397`) + `Balanceador.rampa_fatia_passo` (0,02, `knobs.py:398`) × `_rampa[f]` para B e C (`balanceador.py:288`).
3. As fatias abertas entram em ordem crescente de teto. A fatia da posição j recebe `min(teto, resto/(n − j))`, e o resto cai desse valor (`balanceador.py:290-292`). O que o teto corta volta em partes iguais às fatias de teto maior.
4. Fatia fechada recebe 0.

A fração da fatia C depende só de quantas fatias estão abertas e dos contadores `_rampa[1]` e `_rampa[2]`, porque o teto da fatia B também entra na divisão do resto (`balanceador.py:288-292`). Ela **não depende da nota** de nenhuma célula (o código não lê `s` em `fracoes_fatias`). `_rampa[f]` sobe 1 a cada `Balanceador.rampa_fatia_its` (12, `knobs.py:399`) fechos de iteração depois da abertura da fatia, e nunca desce (`balanceador.py:189-193`). Com `rampa_fatia_inicio` 0,05 e `rampa_fatia_passo` 0,02, o teto da fatia C chega a 0,35 com `_rampa[2]` = 15 (aritmética dos dois knobs).

A fração de episódios que de fato passa pelo BOTAR é menor que a fração da fatia C. O env só chega ao trecho 4 se fechar o P3 no PEGAR (trecho 2, por aviso) e cumprir o CARREGAR do trecho 3.

A fatia C é sempre sobre terreno plano: o sorteio de terreno só vale para as fatias ANDAR e B (`eventos.py:71`), e a fatia C recebe coluna 0 e linha 0 (`eventos.py:74`).

---

## 2. Reset

O reset de um env da fatia C acontece na criação do episódio. O BOTAR **não abre** no reset: o trecho inicial é o ANDAR de espera (`comando.py:660`). Este reset põe tudo o que o BOTAR usa depois: a caixa, a laje de origem, a base, as juntas, a carga.

### 2.1 Ordem dos eventos

A ordem do `_reset_idx` vem do mjlab (`mjlab/envs/manager_based_rl_env.py:553-591`): termo de currículo, `sim.reset`, `scene.reset`, eventos de modo `reset` na ordem do dict, e depois os resets dos managers (observação, ação, recompensa, métricas, currículo, comando, evento, terminação). Os eventos de reset da v3 seguem a ordem do dict de `env_cfg.py:153-184`.

| # | Passo | Função | O que escreve | Ref. |
|---|---|---|---|---|
| 0 | Termo de currículo `v3` | `curriculo.v3` | Fecha a iteração do balanceador se ela virou (seção 5.1) e ajusta o teto do atraso | `curriculo.py:103-143` |
| 1 | `fatia` | `eventos.sorteia_fatia` | `v3_fatia`; `v3_t_empurrao` = −∞; coluna 0 e linha 0 do terreno na fatia C; origem do env | `env_cfg.py:154`; `eventos.py:133-175` |
| 2 | `cena` | `eventos.posiciona_cena` | `v3_topo`, `v3_massa`, `v3_caixa_xy`, `v3_rumo_cena`; pose da laje e da caixa | `env_cfg.py:156`; `eventos.py:332-367` |
| 3 | `base` | `eventos.reset_base_por_fatia` | Pose da base atrás da caixa | `env_cfg.py:157-158`; `eventos.py:178-225` |
| 4 | `juntas` | `reset_joints_by_offset` | Pose de pé, velocidade zero | `env_cfg.py:159-160` |
| 5 | `carga` | `eventos.carga` | Massa e inércia da caixa no modelo | `env_cfg.py:161`; `eventos.py:443-461` |
| 6 | `atrito_pad` | `eventos.atrito_pad` | μ dos pads de palma | `env_cfg.py:162-163`; `eventos.py:464-475` |
| 7 | `atrito_laje` | `eventos.atrito_laje` | μ da laje | `env_cfg.py:164-165`; `eventos.py:478-489` |
| 8 | `vies_imu` | `eventos.vies_imu` | `v3_vies_giro`, `v3_incl_imu` | `env_cfg.py:166`; `eventos.py:492-509` |
| 9 | `pd` | `eventos.pd` | Ganhos kp e kd (só fora do play) | `env_cfg.py:170`; `eventos.py:523-531` |
| 10 | `forca_motor` | `eventos.forca_motor` | Limite de esforço dos motores | `env_cfg.py:173-175`; `eventos.py:534-545` |
| 11 | `solref` | `eventos.solref` | `solref[0]` das geoms de contato | `env_cfg.py:177-180`; `eventos.py:548-565` |
| 12 | `filtros` | `eventos.reinicia_filtros` | Filtro de elevação, odometria e estimador da caixa | `env_cfg.py:184`; `eventos.py:228-241` |

O evento `com_caixa` só entra no cfg se `DR.com_caixa` for True (`env_cfg.py:182-183`). O valor é False (`knobs.py:527`), então o evento não existe.

Depois dos eventos, o reset do comando roda a cadeia (`CadeiaCmd._resample`, `comando.py:639-660`) e em seguida o twist (`TwistV3.reset`, `comando.py:879-895`). A ordem é a do dict de comandos (`env_cfg.py:118-128`; `mjlab/managers/command_manager.py:241-247`).

### 2.2 Fatia, terreno e origem

- `v3_fatia` recebe a fatia sorteada (`eventos.py:162`).
- `v3_t_empurrao` volta a −∞ (`eventos.py:163`). Esse valor serve de âncora ao ANDAR e não afeta o BOTAR.
- Na fatia C, `terreno_ep` é descartado por `ter = terreno_ep & (andar | fatia == B)`. O env recebe `terrain_levels` = 0, `terrain_types` = 0 e a origem do sub-terreno (0, 0) (`eventos.py:71-74,172-175`). A coluna 0 do gerador é `flat()` (`cena.py:263`, `Terreno.colunas`, `knobs.py:434`).

### 2.3 Caixa e laje de origem (`posiciona_cena`)

A laje de origem é a laje onde a caixa começa. A mesma entidade (`table`) vira depois a laje de destino (seção 3). A laje é um corpo mocap (cinemático): a pose é o centro do corpo, `Cena.prateleira_meia_z` (0,02 m, `knobs.py:90`) abaixo do topo pedido (`eventos.py:276-294`).

| Item | Valor sorteado ou posto | Distribuição e momento | Knob e ref. |
|---|---|---|---|
| Topo da laje de origem (sobre o chão local) | `topo = lo + (h₀ + extra − lo)·u₀`; `lo = max(h_k, topo_min_c) + extra`; `u₀` uniforme em [0, 1) | Reset | `Niveis.pegar[nível do PEGAR][0]` = h_k (`knobs.py:417`); h₀ = `Niveis.pegar[0][0]` = 0,55 m; `Niveis.topo_min_c` 0,30 m (`knobs.py:419`); `extra` = `Niveis.subnivel_extra[subnível do PEGAR]` (0,10; 0,0) m (`knobs.py:422`). `eventos.py:82-109,344-346` |
| Massa da caixa | `massa = m_min + (m_j − m_min)·u₁`; `u₁` uniforme | Reset | `Cena.caixa_massa_base` = m_min = 1,0 kg (`knobs.py:80`); m_j = `Niveis.pegar[j][1]` com j = `nivel_passou("PEGAR")`; j = −1 vale m_min (`eventos.py:102,107-108`) |
| Meia aresta da caixa | Uma de 8 variantes igualmente espaçadas | Startup, uma vez por env | `Cena.caixa_meia_faixa` (0,07; 0,13) m (`knobs.py:78`); `Cena.caixa_n_variantes` 8 (`knobs.py:79`). `eventos.py:416-440` |
| x da caixa | `caixa_xy[0] + dx`; `dx = jitter_x_max[nível do PEGAR]·u₂` (só para um lado) | Reset | `Cena.caixa_xy` (0,32; 0,00) m (`knobs.py:96`); `Cena.jitter_x_max` (0,20; 0,20; 0,20; 0,15; 0,08) m (`knobs.py:99`). `eventos.py:358,361` |
| y da caixa | `caixa_xy[1] + U(−0,18; 0,18)` | Reset | `Cena.caixa_jitter_y` (−0,18; 0,18) m (`knobs.py:97`). `eventos.py:359` |
| Laje de origem, xy | `prateleira_xy` | Reset | `Cena.prateleira_xy` (0,50; 0,00) m (`knobs.py:94`). `eventos.py:362,366` |
| Rumo e deslocamento da cena | 0 e 0 | Reset | Só a fatia B sobre terreno usa rumo e deslocamento (`eventos.py:355-356`) |
| Altura da caixa | O centro fica a `v3_meia` acima do topo | Reset | `eventos.py:297-307` |
| Velocidade da caixa | Zero | Reset | `eventos.py:307` |

O núcleo `sorteia_topo_e_massa` usa o nível do PEGAR, o nível que o PEGAR já passou no portão (`nivel_passou`) e o subnível do PEGAR (`eventos.py:344-346`). Por isso a caixa que o BOTAR carrega depende do **PEGAR** e não do BOTAR. A subida desses níveis está na seção 5.

A massa e a inércia chegam ao modelo no evento `carga`: massa = `v3_massa`; inércia = (2/3)·m·a² por eixo, com a = `v3_meia` (`eventos.py:455-461`).

### 2.4 Base e juntas

A base nasce atrás da caixa, no mesmo y (`_base_da_pega`, `eventos.py:178-208`).

| Item | Valor | Distribuição e momento | Knob e ref. |
|---|---|---|---|
| Distância da base à caixa (x) | `d = recuo(topo, a) + (2·u₀ − 1)·faixa_recuo·jitter` | Reset, u₀ uniforme | `Alvos.faixa_recuo` 0,08 m (`knobs.py:227`); `jitter` = `Niveis.subnivel_jitter[subnível do PEGAR]` (0,0; 1,0) (`knobs.py:424`); `recuo(topo, a)` vem de `Referencias` (`eventos.py:192-194`) |
| Posição xy | caixa − R(rumo)·(d, 0): x = x da caixa − d; y = y da caixa | Reset | `eventos.py:201-202` |
| Yaw | `rumo_cena + U(−0,2; 0,2)` rad | Reset | `Cena.reset_base_manipula` yaw (−0,2; 0,2) (`knobs.py:101`). `eventos.py:195,198` |
| z | origem z + z padrão do robô + `U(0,01; 0,05)` m | Reset | `Cena.reset_base_manipula` z (0,01; 0,05) (`knobs.py:101`). `eventos.py:196,199,204` |
| Velocidade da base | A do estado padrão | Reset | `eventos.py:208` |
| Juntas | Posição padrão (`q_de_pe`), velocidade zero | Reset | faixa (0,0; 0,0) em `env_cfg.py:159-160`; `cena.py:225` |

Não determinado pelo código lido: o valor numérico de `recuo(topo, a)` e de `q_de_pe`. Eles vêm de `dados/referencias.npz`, que não está entre as fontes.

### 2.5 Estado de episódio zerado pelo reset do comando

`CadeiaCmd._resample` limpa o que o BOTAR usa (`comando.py:639-660`).

- Registra o PEGAR e o BOTAR que estavam abertos (seção 5.2) (`comando.py:643-645`).
- Conta como falha na zona a visita de etapa que estava aberta (`comando.py:646-650`).
- Zera `v3_pegou`, `v3_perdeu`, `v3_pega_fechada`, `v3_bota_fechada`, `v3_perdeu_agora`, `v3_avancou`, `v3_aviso`, `v3_fim` (`comando.py:651-653`).
- Põe `v3_relogio_alvo` = 0, `v3_sigma` = 1,0, `v3_phi` = −1,0, `v3_h` = 1,0, `v3_phi_melhor` = 0,0, `v3_v_chegada` = −1,0 (`comando.py:654-658`).
- Abre o trecho inicial da fatia (`comando.py:660`): na fatia C, o ANDAR de espera, com `v3_alvo_idx` = −1 (`comando.py:585-587`).

O reset do twist zera o relógio de marcha (`_fase_t`), os filtros de velocidade e do acerto (`v_filt`, `q_filt`, `g_marcha`) e as somas (`comando.py:884-892`).

O reset do estimador (evento `filtros`) põe a odometria na pose verdadeira, zera o estimador da caixa e sorteia os vieses do episódio (seção 6) (`eventos.py:237-241`; `estimador.py:262-280`).

---

## 3. Comando e alvo

### 3.1 O comando de velocidade no BOTAR é zero

O comando de velocidade (o twist) é (vx, vy, ωz). Ele só é diferente de zero em dois casos: tarefa ANDAR na fatia ANDAR, e tarefa CARREGAR na fatia B (`comando.py:905`). A atribuição começa por `vel_command_b[env_ids] = 0.0` (`comando.py:902`) e `v3_celula` = −1 (`comando.py:901`).

Num env da fatia C, o comando vale (0, 0, 0) o episódio inteiro, incluindo o BOTAR. Consequências no código:

- A observação `comando` do ator é zero (`observacoes.py:392`).
- O relógio de marcha não corre: `on = ‖comando‖ > Comando.limiar_marcha` (0,1, `knobs.py:141`) dá False, e a fase só avança com `on` (`comando.py:863,872-873`). A fase fica em 0 depois do reset, e a observação `fase` vale (sen 0, cos 0) = (0, 1) (`observacoes.py:312-314`).
- O regime do freio é "parado" (seção 7).
- O segmento de comando do twist não registra célula (`v3_celula` = −1; `comando.py:792`).

O twist faz uma nova amostragem a cada U(3,0; 8,0) s, com `Comando.reamostra_s` (3,0; 8,0) s (`knobs.py:137`; `env_cfg.py:122-123`; `mjlab/managers/command_manager.py:109-123`). Na abertura do BOTAR a cadeia chama `twist._resample(ids)` (`comando.py:519`). Essas reamostragens só gravam comando zero.

### 3.2 O alvo do BOTAR: o ponto de soltura

O alvo do BOTAR é o ponto de soltura, a posição do centro da caixa sobre a laje de destino. Ele nasce na abertura do BOTAR (`_abre_botar`, `comando.py:542-566`) e **não muda** depois. `v3_alvo_w` só é escrito em `_abre_p0` e em `_abre_botar` (`comando.py:391,558`). O B2 e o B3 não têm alvo de posição próprio; o ponto de soltura continua em `v3_alvo_w`.

A laje de destino é a mesma laje mocap da cena, teletransportada na abertura do BOTAR.

**Sorteio** (`sorteia_destino`, `eventos.py:373-390`; momento: abertura do BOTAR, um sorteio por env, com quatro uniformes em [0, 1)):

| Variável | Fórmula | Knob |
|---|---|---|
| θ (giro da laje, rad) | `(2·u₀ − 1)·radianos(destino_giro_max_graus)` | `Cena.destino_giro_max_graus` 10,0° (`knobs.py:87`) |
| y (m) | `(2·u₁ − 1)·destino_jitter_y` | `Cena.destino_jitter_y` 0,10 m (`knobs.py:86`) |
| longe (m) | subnível abaixo do último: `u₂·subnivel_longe[subnível]`; subnível último: `u₂·jitter_x_max[nível do BOTAR]` | `Niveis.subnivel_longe` (0,10; 0,20) m (`knobs.py:423`); `Cena.jitter_x_max` (`knobs.py:99`) |
| topo (m, sobre o chão local) | subnível abaixo do último: `botar[0] + subnivel_extra[subnível]`; subnível último: `botar[n] + (botar[0] − botar[n])·u₃` | `Niveis.botar` (0,55; 0,45; 0,30; 0,15; 0,04) m (`knobs.py:418`); `Niveis.subnivel_extra` (0,10; 0,0) m (`knobs.py:422`) |

O subnível "abaixo do último" é o subnível 0 (`eventos.py:387`; `_sub_max` = 1, `balanceador.py:92`). Faixas por nível, derivadas da tabela acima:

| Nível do BOTAR | Subnível | Topo da laje (m) | `longe` máximo (m) |
|---|---|---|---|
| 0 | 0 | fixo 0,65 (0,55 + 0,10) | 0,10 |
| 0 | 1 | fixo 0,55 | 0,20 (`jitter_x_max[0]`) |
| 1 | 1 | U(0,45; 0,55) | 0,20 (`jitter_x_max[1]`) |
| 2 | 1 | U(0,30; 0,55) | 0,20 (`jitter_x_max[2]`) |
| 3 | 1 | U(0,15; 0,55) | 0,15 (`jitter_x_max[3]`) |
| 4 | 1 | U(0,04; 0,55) | 0,08 (`jitter_x_max[4]`) |

**Construção** (`posiciona_destino`, `eventos.py:393-413`; `alvos.laje_destino`, `alvos.py:182-192`):

1. `recuo = ref.recuo(topo, meia aresta) + longe` (`eventos.py:405`).
2. `e(θ) = destino_borda·cos θ + prateleira_meia_xy·sen|θ| − destino_borda`, com `Cena.destino_borda` 0,15 m (`knobs.py:85`) e `Cena.prateleira_meia_xy` 0,30 m (`knobs.py:89`) (`alvos.py:188`).
3. Alvo no frame da base: `(recuo + e(θ), y)`. Centro da laje no frame da base: alvo + 0,15·(cos θ, sen θ), onde 0,15 = `prateleira_meia_xy − destino_borda` (`alvos.py:189-191`).
4. O frame da base é o da posição e do yaw da base **no instante da abertura** (`eventos.py:407-412`).
5. A laje vai para o centro, com yaw = yaw da base + θ, e z do centro = topo no mundo − `prateleira_meia_z` (`eventos.py:410`; `_laje`, `eventos.py:276-294`).
6. `v3_topo_destino` recebe o topo (`eventos.py:411`).
7. O ponto de soltura é (alvo xy no mundo, topo no mundo); a cadeia soma `v3_meia` ao z (`comando.py:549`).

**Viés do alvo.** Na abertura, se o gerador `env.v3_gerador` existe (`comando.py:550`), o ponto de soltura recebe um erro uniforme por eixo: `(2u − 1)·(vxy, vxy, vz)`, com u em [0, 1) por eixo (`comando.py:550-557`). O `garante_v3` cria o gerador em todo env (`curriculo.py:62-64`), então o viés entra no treino. `vxy` e `vz` seguem `no_degrau` do degrau global g (seção 5.6):

- `vxy` vai de `Estimador.vies_alvo_botar_xy[0]` = 0,01 m a `[1]` = 0,05 m (`knobs.py:461`).
- `vz` vai de `Estimador.vies_alvo_botar_z[0]` = 0,005 m a `[1]` = 0,03 m (`knobs.py:462`).

O ponto com o viés é o `v3_alvo_w`. Ele vale para a observação, para o Φ do B1, para o fecho do B1 (`comando.py:476`) e para o raio da perda da caixa (`comando.py:407`).

### 3.3 Quando algo muda dentro do episódio

| O que | Quando muda |
|---|---|
| Comando de velocidade | Nunca sai de zero (seção 3.1) |
| Alvo (`v3_alvo_w`) | Só na abertura do BOTAR |
| Índice da etapa (`v3_alvo_idx`) | No fecho de cada etapa (seção 4), na perda (vira B3) e na abertura do BOTAR (vira B1) |
| Laje de destino | Só na abertura do BOTAR |
| Zona efetiva dos critérios (`v3_alvos`) | Uma vez por iteração do PPO, no primeiro `compute` com a iteração nova (`comando.py:428-443`) |

---

## 4. Etapas dentro do episódio e critério de avanço

### 4.1 Ordem dentro de um passo

O avanço das etapas acontece no `compute` da cadeia. A ordem do passo vem de `mjlab/envs/manager_based_rl_env.py:418-464`:

1. A ação é processada e a física roda 4 subpassos (`decimation`).
2. `episode_length_buf` e `common_step_counter` sobem 1.
3. As terminações são calculadas (`:436`).
4. As recompensas são calculadas (`:440`) e as métricas (`:441`).
5. Os envs terminados fazem reset (`:444-448`).
6. `sim.forward()` (`:454`).
7. `command_manager.compute`: primeiro a cadeia, depois o twist (`:456`).
8. Os eventos de modo `step` e depois os de modo `interval` (`:458-461`).
9. `sim.sense()` e as observações (`:463-464`).

Consequência: o fecho de uma etapa, escrito no passo t, aparece nas recompensas e nas terminações do passo t + 1. Isso vale para `v3_alvo_idx`, `v3_perdeu`, `v3_perdeu_agora` e `v3_bota_fechada`.

### 4.2 Sequência do `compute` da cadeia

`CadeiaCmd.compute` (`comando.py:596-629`) executa nesta ordem:

1. `_mede`: as medidas do passo (`comando.py:598`).
2. `_perda`: detecta a perda da caixa (`comando.py:599`). Em seguida o `compute` publica o pulso em `env.extras["v3_perdeu_agora"]`, que o algoritmo de PPO lê (`comando.py:601`; seção 8.3).
3. `_abre_p0`: só age em PEGAR sem alvo (`comando.py:602`).
4. `_alvos`: o fecho e o avanço das etapas (`comando.py:603`).
5. Atualiza o aviso: `v3_aviso |= fecho & por_detector[trecho]`, com `fecho` = `v3_bota_fechada` se a tarefa é BOTAR, `v3_pega_fechada` nos outros casos (`comando.py:606-607`).
6. `avanca_trecho` (`comando.py:608-609`). Se a caixa foi perdida, o trecho não avança e o fim da cadeia não dispara (`comando.py:610-613`).
7. Se o trecho avançou: abre o trecho seguinte (`comando.py:617-619`). Se acabou a cadeia: `v3_fim` = True e registra o BOTAR (`comando.py:614-616`).
8. Passo da odometria (`comando.py:623-625`), `_grava_chegada` (`comando.py:626`) e `v3_vz_caixa_ant` (`comando.py:629`).

### 4.3 Abertura do B1

A abertura do BOTAR chama `_abre_botar` (`comando.py:542-566`): põe o ponto de soltura em `v3_alvo_w`, `v3_alvo_idx` = B1, `v3_phi` = −1, `v3_h` = 1, `v3_phi_melhor` = 0, `v3_relogio_alvo` = 0, `v3_ponto_apoio` = 0, `v3_v_chegada` = −1, `v3_sigma` = σ do B1, e `self.bota[ids]` = True.

### 4.4 O σ de cada etapa

O σ é a escala de distância do Φ (seção 7.3). Ele é fixado na abertura da etapa como `max(d, sigma_min_etapa[etapa])`, onde `d` é a distância da etapa naquele instante (`comando.py:348-362`). `Alvos.sigma_min_etapa` = (0,05; 0,05; 0,08; 0,20; 0,20; 0,05; 0,0) m para (P0, P1, P2, P3, B1, B2, B3) (`knobs.py:208`). No BOTAR: B1 = 0,20 m, B2 = 0,05 m, B3 = 0,0. O σ de B2 e B3 é gravado no fecho da etapa anterior (`comando.py:502-503`); o do B1, na abertura do BOTAR (`comando.py:565`).

### 4.5 Critério de fecho de cada etapa

O avanço de uma etapa exige a condição `ok_agora` verdadeira **sem interrupção** durante o tempo de permanência. O relógio soma o passo (0,02 s) a cada passo com `ok_agora` verdadeiro e volta a 0 quando ela falha (`alvos.avanca`, `alvos.py:195-210`). A etapa avança quando `ok_agora & relógio ≥ tempo − 1e-9`. Depois do B3, o índice vira −1 (`alvos.py:206-208`).

`ok_agora_botar` (`comando.py:73-88`) é a condição de cada etapa, com `& (invariantes == 1,0)`. Os valores efetivos dos limites vêm de `env.v3_alvos` (a zona, seção 5.3), refeito a cada iteração (`comando.py:428-443`). A tabela abaixo mostra o valor final (z = 0) e o largo (z = 1), de `Alvos.zona` (`knobs.py:211-220`); o valor efetivo é `final + z·(largo − final)` (`comando.py:436`; `balanceador.py:48-51`).

| Etapa | Condição `ok_agora` (além do invariante) | Tempo de permanência | Valores final → largo |
|---|---|---|---|
| **B1** (levar ao destino e apoiar) | ‖caixa − alvo B1‖ ≤ `raio_b1` **e** força vertical de apoio ≥ `apoio_b1_mg`·peso **e** ‖v da caixa‖ < `v_parada` (`comando.py:82-83`) | `tempo_b1` | `raio_b1` 0,10 → 0,20 m (`knobs.py:191`); `tempo_b1` 0,3 → 0,1 s (`knobs.py:194`). `apoio_b1_mg` 0,8 (`knobs.py:192`) e `v_parada` 0,05 m/s (`knobs.py:195`) **não** afrouxam |
| **B2** (afastar as mãos) | As duas palmas com afastamento ≥ `b2_afasta` **e** a caixa em contato com a laje (`apoio_caixa` com `found` > 0) **e** ‖v da caixa‖ < `v_parada` (`comando.py:84`) | `tempo_b2` | `b2_afasta` 0,05 → 0,02 m (`knobs.py:196`); `tempo_b2` 0,5 → 0,2 s (`knobs.py:197`) |
| **B3** (voltar à pose de pé) | \|pelve − pelve de pé\| ≤ `tol_pelve_b3` **e** \|inclinação do tronco\| ≤ `tol_tronco_b3_graus` **e** maior distância palma–pose de pé ≤ `tol_palmas_b3` (`comando.py:86`) | `tempo_b3` | `tol_pelve_b3` 0,03 → 0,06 m (`knobs.py:198`); `tol_tronco_b3_graus` 5,0 → 10,0° (`knobs.py:199`); `tol_palmas_b3` 0,05 → 0,10 m (`knobs.py:200`); `tempo_b3` 0,5 → 0,2 s (`knobs.py:201`) |

Medidas usadas (todas em `_mede`, `comando.py:317-346`):

- Caixa e alvo: `caixa_p` é a posição verdadeira da caixa; o alvo B1 é `v3_alvo_w` (`comando.py:476`; `alvos.caixa_alvo_w`, `alvos.py:163-179`).
- Força de apoio: `forca_de_apoio`, a soma do módulo da componente z da força do sensor `apoio_caixa` (caixa × laje) (`comando.py:255-266,340`). Peso: `v3_massa` × |g_z| do modelo (`comando.py:341`).
- Afastamento da palma: `max_eixo(|palma no frame da caixa|) − (a + Alvos.palma_face)`, por palma, com `Alvos.palma_face` 0,045 m (`knobs.py:178`) (`comando.py:478`).
- Pelve: z da base − z da sola mais baixa (`z_pe_min`, `observacoes.py:236-243`; `comando.py:328`). A referência `pelve_de_pe` vem de `referencias.npz` (não lido; não determinado).
- Inclinação do tronco: `recompensas.incl_torso` (`recompensas.py:467-470`).
- Distância das palmas à pose de pé: no frame da base, a maior das duas (`comando.py:343-344`).

**Invariante.** O invariante é o produto de três portões, calculado com os valores finais `A` do `Alvos` (`comando.py:463-466`):

- `desloc`: vale `rampa(desloc_apoio; desloc_b2b4, delta_b2b4)` no B2 e no B3 e vale 1 no B1; o `desloc_apoio` é a distância xy da caixa ao `v3_ponto_apoio`, e vale 0 com a caixa perdida (`recompensas.py:262-263`; `comando.py:457,461`). `Alvos.desloc_b2b4` 0,02 m (`knobs.py:224`), `Alvos.delta_b2b4` 0,06 m (`knobs.py:225`).
- `incl`: `rampa(inclinação da caixa; incl_lim; delta_incl_graus)` no B1 e no B2; 1 no B3 (`recompensas.py:56,266`). `Alvos.delta_incl_graus` 15,0° (`knobs.py:205`).
- `maos`: no B1 vale (palmas em contato)/2; nos outros vale 1 (`recompensas.py:56,267`; `comando.py:464`).

A rampa é `clip(1 − (x − x_lim)/Δ, 0, 1)` (`potencial.py:23-29`) e vale exatamente 1,0 abaixo do limite. O invariante `== 1,0` exige então: no B1, **as duas palmas** em contato com a caixa e inclinação ≤ `incl_lim`; no B2, deslocamento ≤ 0,02 m e inclinação ≤ `incl_lim`; no B3, deslocamento ≤ 0,02 m. O `incl_lim` é o limite de inclinação por degrau (seção 5.5).

### 4.6 O que acontece no fecho de cada etapa

Num passo com avanço (`comando.py:482-504`):

- Fecho do B1: `v3_ponto_apoio` recebe a posição da caixa nesse passo (`comando.py:494-495`).
- Fecho do B3: `v3_bota_fechada` fica True (`comando.py:498`). O índice vira −1 (`alvos.py:206-208`).
- `v3_avancou` recebe o avanço (`comando.py:499`), `v3_relogio_alvo` o relógio (`comando.py:500`) e `v3_alvo_idx` o novo índice (`comando.py:501`).
- Troca de etapa (`troca = av & ~fecha_p3 & novo ≥ 0`): `v3_sigma` recebe o σ da etapa nova e `v3_h` volta a 1 e `v3_phi_melhor` a 0 (`comando.py:502-504`).

### 4.7 Do fecho do B3 ao fim do trecho

O aviso do trecho BOTAR liga no mesmo `compute` do fecho do B3 (`comando.py:606-607`). O relógio do trecho só corre com o aviso (`comando.py:209`). O trecho avança quando `aviso` e `relógio ≥ atraso sorteado − 1e-6` (`comando.py:210-211`; `_EPS` 1e-6 em `comando.py:35`). Depois, o trecho 5 (ANDAR final) dura 1,5 s e acaba a cadeia: `v3_fim` = True, e a terminação `time_out` dispara no passo seguinte (`terminacoes.py:53-72`).

### 4.8 Visitas da zona

A zona z de cada etapa depende de contagens de **visitas** e de **fechos**. Uma visita é o período em que uma etapa está aberta. A cadeia conta as visitas a cada passo (`comando.py:444-452,483-491`).

- `etapa` = o índice do alvo, ou −1 se o índice é < 0, ou se o índice é o P3 com a pega fechada, ou se a caixa foi perdida (`comando.py:447`).
- Uma visita começa quando a `etapa` muda para um valor ≥ 0: `t_vis` = 0. Depois `t_vis` soma o passo a cada passo (`comando.py:448-452`).
- A visita acaba **com fecho** quando `av` é True e `t_vis ≥ 0`: conta 1 visita e 1 fecho na etapa (`comando.py:483,486-490`).
- A visita acaba **sem fecho** quando `av` é False e `t_vis ≥ Balanceador.zona_t_visita` (8,0 s, `knobs.py:407`): conta 1 visita (`comando.py:484,486-487`).
- A visita acaba **sem fecho** quando a `etapa` muda sem fecho (troca por fora, por exemplo a perda): conta 1 visita (`comando.py:449,486`).
- No reset, a visita aberta conta 1 visita sem fecho (`comando.py:646-650`).
- Depois que a visita acaba, `t_vis` = −1 e ela não reabre até a `etapa` mudar (`comando.py:491`).

As contagens vão para `vis_acum[e]` e `fech_acum[e]` do currículo (`comando.py:488-490`).

---

## 5. Progressão

A progressão do BOTAR roda no `fecha_iteracao`, em sete passos de ordem fixa (`balanceador.py:153-195`). Antes do primeiro passo, o contador `it` do currículo soma 1 (`balanceador.py:162`).

1. EMA da nota das células.
2. Zona z das etapas.
3. Contador do refino de cada tarefa.
4. Abertura de células e fatias (`_abre`).
5. Rampa das fatias.
6. Subidas de nível (`_sobe`).
7. Travas da DR (`_trava_dr`).

### 5.1 Quando a iteração fecha

O termo `curriculo.v3` roda no início de cada `_reset_idx`. A iteração é `common_step_counter // passos_por_iteracao` (24, `knobs.py:402`). Se a iteração é diferente da última guardada em `v3_iter` e `v3_iter ≥ 0`, o termo chama `cur.fecha_iteracao()` (`curriculo.py:119-122`). A primeira chamada só guarda a iteração (`curriculo.py:120-122`). Se nenhum env fizer reset durante uma iteração, o fecho espera o primeiro reset seguinte.

### 5.2 A célula `botar` e quem mede a nota

A nota `s` da célula `botar` é uma EMA (média móvel exponencial) do sucesso dos episódios que passaram pelo BOTAR.

**Quem mede.** A cadeia registra um BOTAR por episódio com `CadeiaCmd._registra(..., botar=True)` (`comando.py:521-540`). O registro chama `Curriculo.registra(célula, n, soma)` com `n` = número de envs registrados e `soma` = número de sucessos (`comando.py:539`; `balanceador.py:148-151`). O twist **não** registra a célula `botar` (o `seg_cel` é −1 na fatia C, `comando.py:792`).

**Quando registra.**

- O BOTAR registra no **fim da cadeia** (`comando.py:614-616`) e no **reset** (`comando.py:645`).
- Só registra env com `self.bota` = True, ou seja, env que abriu o BOTAR (`comando.py:526-528`). Um episódio que acabou sem abrir o BOTAR não conta na célula `botar`.
- No reset, `falhou = ~time_outs[env_ids]` (`comando.py:643`). Terminação que não é `time_out` (queda, chegada, batente) conta como falha. Um `time_out` com B3 fechado conta como sucesso.

**Condição do sucesso** (`comando.py:533-535`; `sucesso_do_botar`, `comando.py:91-93`):

`sucesso = v3_bota_fechada & ~falhou & ~v3_perdeu & (max(z[4:]) == 0,0)`

O último termo exige as zonas z do B1, do B2 e do B3 **todas em 0,0**. Enquanto alguma zona for maior que 0, todo episódio entra na célula como falha.

**EMA** (`balanceador.py:162-168`): uma atualização por fecho de iteração para a célula que teve `n > 0` registros: `x = Σ sucessos / n`; se `n_medidas == 0`, `s = x`; senão `s ← s + alfa·(x − s)`, com `Balanceador.alfa` 0,05 (`knobs.py:394`). `n_medidas` sobe 1. Com `n = 0`, `s` não muda.

### 5.3 Zona adaptativa z (B1, B2, B3)

O que muda no env: o valor efetivo dos critérios de fecho (seção 4.5): `final + z·(largo − final)`. Só o fecho (`ok_agora_*`), o tempo de permanência e a qualidade do CARREGAR leem o efetivo; o Φ, a distância `d`, os portões e o ponto-alvo seguem no valor final (`comando.py:428-431`).

**Estado inicial:** z = 1,0 em todas as etapas (`balanceador.py:105`).

**Quem mede:** a cadeia conta visitas e fechos (seção 4.8). O `fecha_iteracao` lê os acumuladores.

**Condição de atualização** (`balanceador.py:169-174`), para cada etapa `e` no fecho de uma iteração:

- Se `vis_acum[e] ≥ Balanceador.zona_n_min` (200, `knobs.py:406`): `taxa[e] = fech_acum[e]/vis_acum[e]`; `z[e] = clip(z[e] − zona_ganho·(taxa[e] − zona_t_alvo), 0, 1)`; os acumuladores zeram.
- `Balanceador.zona_ganho` 0,05 (`knobs.py:405`); `Balanceador.zona_t_alvo` 0,6 (`knobs.py:404`).
- Taxa acima de 0,6: z desce (o critério aperta). Taxa abaixo de 0,6: z sobe (o critério afrouxa).

**Trava de cadeia** (`balanceador.py:175-176`): depois da atualização, `z[e] = max(z[e], z[e−1] − Balanceador.zona_folga_cadeia)` para e = B2 (5) e B3 (6), com `zona_folga_cadeia` 0,2 (`knobs.py:408`). O B1 (índice 4) não sofre trava do P3: o B1 abre a cadeia do BOTAR.

**Momento:** os valores efetivos (`env.v3_alvos` e os tempos) são refeitos uma vez por iteração, no primeiro `_alvos` com `cur.it` diferente de `it_zona` (`comando.py:428-443`).

### 5.4 Níveis e subníveis do BOTAR

O BOTAR tem 5 níveis (0 a 4), pois `max_nivel["BOTAR"] = len(Niveis.botar) − 1` = 4 (`balanceador.py:74`; `knobs.py:418`). O nível 0 tem 2 subníveis (0 e 1), pois `_sub_max = len(Niveis.subnivel_extra) − 1` = 1 (`balanceador.py:92`; `knobs.py:422`).

**O que muda quando sobe** (efeitos do BOTAR):

| O que sobe | Efeito no env |
|---|---|
| Subnível 0 → 1 | A laje de destino vai de topo fixo 0,65 m e `longe` ≤ 0,10 m para topo 0,55 m e `longe` ≤ 0,20 m (seção 3.2) |
| Nível n → n + 1 | O topo da laje de destino passa a U(`botar[n+1]`; 0,55) m e o `longe` máximo segue `jitter_x_max[n+1]` (seção 3.2); o `v3_freio` do BOTAR sobe 1 (seção 7); o canal `niveis` do crítico muda (seção 9) |
| Nível n → n + 1 | O menor nível relativo entre as quatro tarefas (`restricao`) pode subir: afeta o fator de estilo das recompensas e o teto do atraso (seção 5.6) |

O `v3_freio` é `niveis_abertos("BOTAR")` = nível + 1, escrito na publicação da tarefa (`balanceador.py:335-336`; `comando.py:515-517`).

**Condição exata de subida** (`_sobe`, `balanceador.py:250-278`), para t = BOTAR:

1. Se `nível < max_nivel` (4): `ok = célula botar aberta & s_botar ≥ Balanceador.portao` (0,75, `knobs.py:391`). A condição `_passa_media` com uma célula equivale a `s ≥ portão` (`balanceador.py:125-128`). Com nível = 4, `ok` = False para o BOTAR, pois ele não é tarefa de cadeia (`balanceador.py:261-264`).
2. `_seguidas[BOTAR]` soma 1 se `ok`, senão vai a 0 (`balanceador.py:265`).
3. Se `_seguidas ≥ Balanceador.iters_nivel` (50, `knobs.py:395`): `_seguidas` = 0 e sobe (`balanceador.py:266-268`). Se `subnível < _sub_max`, sobe o subnível; senão sobe o nível (`balanceador.py:273-276`).
4. Depois da subida, `n_medidas` de todas as células da tarefa vai a 0 (`balanceador.py:277-278`). A EMA recomeça na primeira medida nova. O `_sobe` **não** exige `n_medidas > 0`: `s` guarda o valor antigo até essa primeira medida (`balanceador.py:165-167,265`).

**Quem mede:** a nota `s_botar` (seção 5.2).

### 5.5 Degrau de refino do BOTAR

O degrau é um contador de 0 a 5 que aperta as regras do BOTAR e, pelo degrau global, a randomização (seção 5.6).

**Condição de subida** (`_conta_refino`, `balanceador.py:234-248`), por fecho de iteração, enquanto `degrau[BOTAR] < Balanceador.degraus` (5, `knobs.py:401`):

- `ultimo = nível do BOTAR == max_nivel (4)` (o BOTAR não é de cadeia, então não exige terreno) (`balanceador.py:239-240`).
- Se `ultimo` e a média das células abertas do BOTAR (só `botar`) ≥ `portao`: `_cont` soma 1; senão `_cont` = 0 (`balanceador.py:241-245`).
- Se `_cont ≥ Balanceador.iters_refino` (100, `knobs.py:400`): `degrau` soma 1 e `_cont` = 0 (`balanceador.py:246-248`). O degrau nunca desce.

**Quem mede:** a nota `s_botar` (seção 5.2).

**O que muda no env.** O env lê o degrau da tarefa em `v3_degrau`, gravado na publicação da tarefa (`comando.py:516-518`). Ele **não** se atualiza dentro do trecho. Com d = `v3_degrau`, a função `por_degrau(ini, fim)` vale `ini + (fim − ini)·d/5` (`recompensas.py:447-450`; `balanceador.py:48-51`).

| Grandeza | Início (d = 0) | Fim (d = 5) | Onde atua | Ref. |
|---|---|---|---|---|
| `incl_lim`: limite de inclinação da caixa no invariante e no Φ | 30,0° | 15,0° | Portão `incl` (B1, B2); fecho do B1 e do B2 | `Alvos.incl_inicio_graus`, `incl_fim_graus` (`knobs.py:203-204`); `recompensas.py:462-464` |
| σ do pouso | 1,0 m/s | 0,35 m/s | Fator de pouso do Φ do B1 | `Recompensa.sigma_pouso` (`knobs.py:368`); `recompensas.py:549` |
| Força do joelho da laje | 50,0 N | 10,0 N | `contato_laje` | `Recompensa.laje_joelho_N` (`knobs.py:302`); `recompensas.py:890` |
| Fator de `sacudida` e `aperto` | 0 | 1 | Custos no B1 | `recompensas.py:944,955` |
| Limite da terminação `chegada` | 1,0 m/s | 0,25 m/s | Terminação | `Terminacao.chegada` (`knobs.py:381`); `terminacoes.py:149` |
| Terminação `batente` | desligada com d < 1 | ligada com d ≥ 1 | Terminação | `Refino.degrau_batente` 1 (`knobs.py:547`); `terminacoes.py:164` |

### 5.6 Degrau global e restrição (efeitos compartilhados)

Dois valores resumem as quatro tarefas.

- **Degrau global** g = menor degrau entre ANDAR, PEGAR, CARREGAR e BOTAR (`balanceador.py:341-343`). Ele só sobe quando as quatro tarefas chegam lá. Por isso o BOTAR segura o g.
- **Restrição** = menor valor de `nível/max_nivel` entre as quatro tarefas; começa em 0 e nunca desce (`balanceador.py:345-348`).

Efeitos que o BOTAR sente:

| Grandeza | Fórmula | Knob |
|---|---|---|
| Fator de estilo das recompensas `giro_xy`, `vel_junta`, `acel_junta`, `energia` | `0,25 + 0,75·restrição` | `Recompensa.estilo_inicio` 0,25 (`knobs.py:260`); `recompensas.py:453-459` |
| Teto do atraso (`atraso_max`), em passos | g < 3: `ceil(2·restrição − 1e-9)`; g ≥ 3: 3 | `DR.atraso_portao[1]` 2 (`knobs.py:512`); `DR.atraso_refino[1]` 3 (`knobs.py:513`); `DR.degrau_atraso_refino` 3 (`knobs.py:514`). `balanceador.py:381-382` |
| Faixas da DR (`pd`, `forca_motor`, `solref`, `rfi`) | `no_degrau(ini, fim, g, 5)` | seção 6 |
| Magnitudes do estimador da caixa | `no_degrau(início, refino, g, 5)` | seção 6 |
| Viés do alvo do B1 | `no_degrau`, seção 3.2 | `Estimador.vies_alvo_botar_*` |
| Deriva da odometria | `no_degrau`, `comando.deriva_do_degrau` | `Estimador.deriva_*` (`knobs.py:473-476`); `comando.py:247-250` |

### 5.7 Travas da DR (ligam e nunca desligam)

Três travas ligam recursos de randomização (`balanceador.py:101-103,197-202`).

| Trava | Condição | Liga |
|---|---|---|
| `portao0` | `nivel_passou("PEGAR") ≥ 0` | `atrito_pad`, `atrito_laje` |
| `vies` | A célula `ANDAR/andar` passa no portão (`s ≥ portao_andar`) | `vies_imu` |
| `subida` | `nível PEGAR + nível CARREGAR > 0` | `forca_caixa` (não vale no BOTAR) e `rfi` |

`nivel_passou(t)` devolve o maior nível cuja célula já passou no portão (−1 = nenhum). Com o subnível abaixo do último o nível 0 ainda não passou; no nível 0 exige `n_medidas > 0` (`balanceador.py:350-356`). `Balanceador.portao_andar` vale 0,75 (`knobs.py:392`).

### 5.8 Abertura da célula BOTAR e da fatia C

`_abre` (`balanceador.py:204-232`) abre a célula `BOTAR/botar` quando ela está fechada e `s` da célula `PEGAR/pegar` ≥ `portao` (0,75). A fatia C passa a ter célula aberta no mesmo fecho, e o primeiro `_it_abertura[2]` é gravado (`balanceador.py:186-188,231-232`). A célula nova entra com `s` = 0 e não abre mais nada no mesmo fecho.

A fatia B abre primeiro (e com ela a célula `PEGAR/pegar`) quando o nível do ANDAR ≥ `Balanceador.nivel_abre_b` (5, `knobs.py:396`) (`balanceador.py:209-211`).

### 5.9 Linha de terreno

O BOTAR **não tem** linha de terreno. Só o ANDAR e o CARREGAR têm célula `terreno` (`balanceador.py:31,77-78`). O `_nivel_ter` só sobe no CARREGAR; o ANDAR tem uma linha por coluna (`balanceador.py:84-89`). A fatia C sempre usa o plano (seção 1.3).

---

## 6. Perturbações e randomizações

Todas as DR abaixo valem para um env no BOTAR, com a exceção marcada. No modo play (`play=True`), `pd`, `forca_motor`, `solref`, o empurrão, a força na caixa e o RFI não existem (`env_cfg.py:169-181,200-210`). O treino usa `play=False`.

### 6.1 Tabela

| Perturbação | Momento | Faixa e distribuição | Condição | Vale no BOTAR | Ref. |
|---|---|---|---|---|---|
| Atrito do pé | Startup, 1 vez por env, igual nas 14 geoms do pé | U(0,3; 1,0), valor absoluto | Sempre | Sim | `DR.atrito_pe` (`knobs.py:508`); `env_cfg.py:60,185-187`; `mjlab/envs/mdp/dr/geom.py:117-147` |
| Massa do tronco | Startup, 1 vez por env | Soma U(−1,0; 3,0) kg ao `torso_link` | Sempre | Sim | `DR.massa_tronco` (`knobs.py:509`); `env_cfg.py:188-190` |
| Viés de encoder | Startup, 1 vez por env, por junta | U(−0,015; 0,015) rad | Sempre | Sim. O ator lê as juntas com `biased=True`; o crítico sem | `DR.vies_encoder` (`knobs.py:510`); `env_cfg.py:191-192`; `mjlab/envs/mdp/dr/joint.py:209-241`; `observacoes.py:394, 404`; `mjlab/envs/mdp/observations.py:60` |
| Tamanho da caixa | Startup | Uma de 8 meias arestas em [0,07; 0,13] m | Sempre | Sim | seção 2.3 |
| Massa da caixa | Reset | U(1,0; m_j) kg | Sempre na fatia C | Sim | seção 2.3 |
| Atrito do pad | Reset | Sem a trava `portao0`: 1,0 fixo. Com a trava: U(`mu_min`; 1,0), `mu_min = max(0,5; 0,6 − 0,025·(nível PEGAR + nível CARREGAR))` | Trava `portao0` | Sim | `Cena.pad_atrito[0]` (`knobs.py:81`); `DR.atrito_pad` (0,6; 1,0) (`knobs.py:523`); `DR.passo_mu_pad` 0,025 (`knobs.py:525`); `DR.mu_pad_min` 0,5 (`knobs.py:526`); `eventos.py:464-475`; `balanceador.py:371-373` |
| Atrito da laje | Reset | Sem a trava: 1,0 fixo. Com a trava: U(0,3; 1,0) | Trava `portao0` | Sim | `Cena.prateleira_atrito[0]` (`knobs.py:91`); `DR.atrito_laje` (0,3; 1,0) (`knobs.py:524`); `eventos.py:478-489` |
| Atrito da caixa e do dorso da mão | Constante (sem sorteio) | Caixa: (0,3; 0,02; 0,001); pad de dorso e geoms de colisão do robô: `Cena.robo_atrito` (0,3; 0,02; 0,001) | — | Sim | `Cena.caixa_atrito` (`knobs.py:83`); `Cena.robo_atrito` (`knobs.py:84`); `Cena.caixa_priority` 0 (`knobs.py:82`); `Cena.prioridade_laje` 3 (`knobs.py:105`); `Cena.caixa_condim` e `Cena.prateleira_condim` 3 (`knobs.py:115,93`); `cena.py:109-113, 155, 219` |
| Viés do IMU | Reset | Giro: U(−0,05; 0,05) rad/s por eixo. Inclinação: U(−1,0; 1,0)° em x e y; z zero | Trava `vies` | Sim | `DR.vies_imu_giro` (`knobs.py:521`); `DR.vies_imu_incl_graus` (`knobs.py:522`); `eventos.py:492-509` |
| Ganhos kp e kd do PD | Reset | Escala U(`no_degrau(1; 0,9; g)`; `no_degrau(1; 1,1; g)`), duas amostras (kp e kd) por alvo | Sempre; faixa pelo degrau global g | Sim | `DR.pd` (0,9; 1,1) (`knobs.py:535`); `eventos.py:523-531`; `mjlab/envs/mdp/dr/actuator.py:72-90` |
| Força do motor | Reset | Escala U em `no_degrau(1; 0,9; g)` a `no_degrau(1; 1,1; g)` (pernas e cintura); `no_degrau(1; 0,8; g)` a `no_degrau(1; 1,1; g)` (braços) | Pelo degrau global g | Sim | `DR.forca_motor` (0,9; 1,1) (`knobs.py:536`); `DR.forca_motor_bracos` (0,8; 1,1) (`knobs.py:537`); `eventos.py:534-545` |
| `solref[0]` das geoms de contato | Reset, 1 valor por env nas geoms do pé, dos pads, da laje e da caixa | U(`no_degrau(0,02; 0,015; g)`; `no_degrau(0,02; 0,03; g)`) | Pelo degrau global g | Sim | `DR.solref_base` 0,02 (`knobs.py:540`); `DR.solref` (0,015; 0,03) (`knobs.py:539`); `env_cfg.py:177-180`; `eventos.py:548-565` |
| Torque aleatório (RFI) | A cada passo, vale nos 4 subpassos | Por junta: U(−1; 1)·s·τ_max, com `s = no_degrau(0,05; 0,1; g)` | Trava `subida`; sem ela s = 0 | Sim | `DR.rfi` (0,05; 0,1) (`knobs.py:534`); `eventos.py:605-615`; `balanceador.py:384`; `TAU_MAX` (`knobs.py:572`) |
| Empurrão na base | Intervalo, por env | Soma U(−0,5; 0,5) m/s à velocidade da base em x e em y | Timer por env: o primeiro valor U(3,0; 7,0) s; depois de cada disparo, novo U(3,0; 7,0) s. O reset do episódio **não** reinicia o timer | Sim, sem máscara de tarefa | `DR.empurrao` (`knobs.py:517`); `DR.periodo_empurrao_s` (3,0; 7,0) s (`knobs.py:520`); `env_cfg.py:201-205`; `eventos.py:261-269`; `mjlab/managers/event_manager.py:225-233, 277-288, 361-375`; `mjlab/envs/mdp/events.py:316-337` |
| Força na caixa | Passo | Impulso por eixo U(−11,5; 11,5) N, duração U(0,1; 0,3) s, pausa U(3,0; 8,0) s | Trava `subida`; só com (tarefa CARREGAR & `v3_pegou` & ~`v3_perdeu`) | **Não**: o evento zera o torque nas outras tarefas | `DR.forca_caixa_N` (`knobs.py:531`); `DR.forca_caixa_dur_s` (`knobs.py:532`); `DR.forca_caixa_periodo_s` (`knobs.py:533`); `eventos.py:584-602` |
| Ruído do ator | A cada passo | Uniforme ±: `giro_base` 0,2; `gravidade` 0,05; `juntas` 0,01; `vel_juntas` 1,5. O ruído entra na ordem ruído, depois escala | Só no ator | Sim | `DR.ruido_obs` (`knobs.py:516`); `observacoes.py:373-387`; `mjlab/managers/observation_manager.py:331-340` |
| Atraso da observação | Reset (lag sorteado no primeiro passo depois do reset) | Inteiro uniforme em [0; `max_lag`] por env, nos termos `giro_base`, `gravidade`, `juntas`, `vel_juntas`, `mapa` do ator | `max_lag` = `DR.atraso_inicio[1]` = 0 na criação (`knobs.py:511`); depois `atraso_max` (seção 5.6) | Sim | `observacoes.py:55, 381-386`; `mjlab/utils/buffers/delay_buffer.py:247-287`; `eventos.py:244-258`; `curriculo.py:124-126` |
| Atraso da ação | Reset (lag sorteado no primeiro subpasso de física depois do reset e mantido até o fim do episódio) | Inteiro uniforme em [0; `max_lag`] passos de física por env; `max_lag` de criação = `DR.atraso_refino[1]·Contrato.decimacao` = 3 × 4 = 12 (`knobs.py:513`, `knobs.py:55`); teto efetivo = `atraso_max·decimacao` | Mesmo `atraso_max` | Sim | `cena.py:188-198`; `eventos.py:253-255`; `mjlab/actuator/actuator.py:310-322, 424-425`; `mjlab/actuator/builtin_group.py:158-173`; `mjlab/utils/buffers/delay_buffer.py:247-287` |
| Estimador da caixa | A cada leitura e a cada episódio | Seção 6.2 | Sempre | Sim | `estimador.py:204-387` |
| Viés do alvo do B1 | Abertura do BOTAR | Seção 3.2 | Sempre | Sim | `comando.py:550-557` |
| Deriva da odometria | A cada passo | Ruído gaussiano por eixo, variância `deriva²·|Δp| + deriva_giro²·|Δψ|` | Sempre | Sim | `estimador.py:158-184`; `comando.py:623-625` |
| Mapa do lidar (termo `mapa` do ator) | A cada passo da política, por raio | Alcance: ruído gaussiano com desvio `Percepcao.ruido_alcance` = 0,02 m. Direção: ruído gaussiano por componente com desvio `Percepcao.ruido_angulo_graus` = 0,15°. Perda: cada raio cai com probabilidade `Percepcao.perda` = 0,01. Giro de inclinação por leitura: gaussiano por eixo com desvio `Percepcao.ruido_incl_graus` = 0,5°, composto com o giro fixo do IMU do episódio (`v3_incl_imu`). Só valem raios com alcance em `Percepcao.alcance` = (0,1; 6,0) m | `Percepcao.ligada` = True | Sim | `knobs.py:483-494`; `percepcao_lidar.py:58-65`; `observacoes.py:140-157` |

O evento `com_caixa` não está registrado (seção 2.1).

⚠ Divergência: o comentário de `observacoes.py:54` diz "Atraso 0–1 passo"; o código dá ao buffer da observação a capacidade `DR.atraso_refino[1]` = 3 passos (`observacoes.py:371, 384`; `knobs.py:513`) e o teto efetivo `atraso_max` de 0 a 3 (seção 5.6).

O lag do atuador sai da mesma classe `DelayBuffer` da observação, em outra instância, com `delay_update_period` = `NUNCA` (10⁹) e `delay_per_env_phase` False (`cena.py:195-197`): o buffer sorteia o lag quando o contador de passos do env vale 0, ou seja, na primeira chamada depois do reset (`delay_buffer.py:208-209, 247-258`). O buffer do atuador recebe um comando por subpasso de física (`actuator.py:174-176`). Os grupos de atuador que o mjlab funde compartilham um buffer (`eventos.py:253-254`).

### 6.2 Estimador da caixa (o que o ator vê da caixa)

O ator não lê a posição verdadeira da caixa. Ele lê a saída do `EstimadorCaixa`, que simula uma câmera (`estimador.py:204-387`). As leituras têm estas regras.

- **Grade de amostragem.** A leitura cai numa grade de `Estimador.taxa_hz` = 30 Hz (`knobs.py:451`) desde o primeiro passo depois do reset (`estimador.py:289-291, 320-338`).
- **Latência.** A amostra entra na saída `Estimador.latencia_s` = 0,033 s depois do instante dela (`knobs.py:452`; `estimador.py:343`).
- **Visibilidade.** A amostra só vale se o centro da caixa está no campo de visão da câmera D435i e na faixa do depth, e se `Estimador.pontos_visiveis_min` = 3 dos 9 pontos da caixa estão sem oclusão pelas cápsulas dos braços. Campo: horizontal 69°, vertical 42° (`Estimador.fov_h_graus`, `fov_v_graus`, `knobs.py:455-456`); faixa (0,28; 3,0) m (`Estimador.alcance`, `knobs.py:457`) (`estimador.py:111-126`).
- **Perda.** Cada amostra visível é descartada com probabilidade `perda` (uniforme); há rajadas de perda de duração U(0,3; 1,0) s (`Estimador.rajada_s`, `knobs.py:468`) abertas com taxa `rajada_taxa` por segundo (`estimador.py:310-317, 331`).
- **Ruído da posição:** por eixo, U(−1; 1)·(σ_pos + σ_rel·dist), mais o viés de posição do episódio (`estimador.py:326-327`).
- **Vieses do episódio:** meia aresta (`vies_meia`), posição (`vies_pos`) e orientação (ângulo lognormal de mediana e p90 dados, eixo uniforme na esfera), sorteados em `reinicia` (`estimador.py:275-279, 302-309`).
- **Ruído da orientação por leitura:** gaussiano por eixo, com desvio `ori_leitura_graus`, sorteado uma vez por passo da política e igual para todas as amostras da grade desse passo (`estimador.py:307`; `_sorteia(..., True)` com `normal=True`, `estimador.py:250-260`).
- **Idade:** `t − instante` da última leitura, cortada em `Estimador.idade_max` = 2,0 s (`knobs.py:453`; `estimador.py:378-385`). Sem leitura, os 7 primeiros canais valem zero (`estimador.py:386`).
- **Caixa presa às palmas:** sem leitura nova, se na última leitura `|caixa − meio das palmas| < meia aresta + Estimador.presa_margem` (0,05 m, `knobs.py:454`), a saída anda com o meio das palmas e gira com o eixo delas (`estimador.py:353-371`).

Magnitudes pelo degrau global g (`observacoes.py:76-90`; `no_degrau(início, refino, g, 5)`):

| Magnitude | Início | Refino | Knob |
|---|---|---|---|
| σ da posição absoluto | 0,005 m | 0 | `Estimador.sigma_pos[0]` (`knobs.py:471`) |
| σ da posição relativa à distância | 0 | 0,02 | `Estimador.sigma_pos[1]` (`knobs.py:471`) |
| Probabilidade de perda | 0,005 | 0,022 | `Estimador.perda` (`knobs.py:472`) |
| Viés da meia aresta | 0,005 m | 0,02 m | `Estimador.vies_meia` (`knobs.py:459`) |
| Viés de posição | 0,0 m | 0,02 m | `Estimador.vies_pos` (`knobs.py:460`) |
| Orientação, mediana | 1,5° | 5,3° | `Estimador.ori_graus` (`knobs.py:465`) |
| Orientação, p90 | 2,5° | 15,3° | `Estimador.ori_graus` (`knobs.py:465`) |
| Ruído da orientação por leitura | 0,0° | 1,0° | `Estimador.ori_leitura_graus` (`knobs.py:466`) |
| Taxa de rajada | 0,0/s | 0,035/s | `Estimador.rajada_taxa` (`knobs.py:469`) |

---

## 7. Recompensas e penalidades

### 7.1 Regras comuns

- Cada termo é uma função `f(env, tarefas)`. Ela devolve um tensor por env; o valor vale 0 onde `v3_tarefa` não está em `tarefas` (`recompensas.py:361-365`). A máscara usa `torch.where` (`curriculo.py:93-100`).
- O mjlab calcula `valor × peso × dt` e troca NaN e infinito por 0 (`mjlab/managers/reward_manager.py:125-129`). O `dt` é o passo da política, 0,02 s (`knobs.py:54-55`), pois `scale_rewards_by_dt` é True (`mjlab/envs/manager_based_rl_env.py:157`). Um termo de peso 0,0 é pulado (`reward_manager.py:122-124`).
- O env_cfg registra 33 termos (`env_cfg.py:42-54`). **23** têm "BOTAR" na tupla de tarefas e valem no BOTAR. Os 10 que não valem estão em ANDAR e CARREGAR apenas: `rastreio_linear`, `rastreio_angular`, `passo`, `vivo`, `altura_pe`, `desvio_bracos`, `desvio_cintura`, `desvio_quadril`, `tronco_plano`, `altura_base`.
- O peso vem de `Recompensa.<nome>` (`env_cfg.py:131-134`), com duas exceções fixas: `progresso` = +1,0 e `perda_caixa` = −1,0 (`env_cfg.py:57`).
- Os termos que pagam valor positivo e zeram com a caixa perdida usam `paga=True` (`recompensas.py:363-364`). Nenhum dos 23 termos do BOTAR usa `paga=True`: o `progresso` trata a perda por dentro.
- Depois da perda da caixa a tarefa publicada vira ANDAR. Os termos de tupla `TODAS` seguem valendo, e os de tupla `PEGAR_BOTAR`, `PEGAR_CARREGAR_BOTAR` deixam de valer (seção 1.2).

Tuplas de tarefa (`env_cfg.py:33-36`): `TODAS` = ANDAR, PEGAR, CARREGAR, BOTAR; `PEGAR_BOTAR`; `PEGAR_CARREGAR_BOTAR`.

### 7.2 Tabela dos 23 termos

Legenda dos fatores: **E** = fator de estilo `0,25 + 0,75·restrição` (`recompensas.py:453-459`; `Recompensa.estilo_inicio` 0,25, `knobs.py:260`). **S1** = `por_restricao(0,0; so="ANDAR")`: vale a restrição só no ANDAR e **1** nas outras tarefas, portanto 1 no BOTAR (`recompensas.py:453-459`). **D** = fator do degrau `d/5`, com d = `v3_degrau` (`recompensas.py:944,955`). O peso, o valor numérico e o dt: valor por passo = peso × termo × 0,02.

| # | Termo | Tuplas | Peso (knob; valor) | Fórmula resumida | Fatores | Vale zero quando | Ref. |
|---|---|---|---|---|---|---|---|
| 1 | `escorrego` | TODAS | `Recompensa.escorrego` −0,2 (`knobs.py:253`) | Σ pés em contato ‖v_xy do tornozelo‖ | nenhum | nenhum pé em contato | `recompensas.py:100-103,659-664` |
| 2 | `vz` | TODAS | `Recompensa.vz` −2,0 (`knobs.py:254`) | v_z² da base no mundo | nenhum | — | `recompensas.py:106-108,667-669` |
| 3 | `giro_xy` | TODAS | `Recompensa.giro_xy` −0,05 (`knobs.py:255`) | ω_x² + ω_y² da base, no frame da base | E | — | `recompensas.py:111-113,672-674` |
| 4 | `vel_junta` | TODAS | `Recompensa.vel_junta` −0,001 (`knobs.py:256`) | Σ q̇² das 29 juntas | E | — | `recompensas.py:116-118,677-679` |
| 5 | `acel_junta` | TODAS | `Recompensa.acel_junta` −5e-8 (`knobs.py:257`) | Σ q̈² das 29 juntas | E | — | `recompensas.py:121-123,682-684` |
| 6 | `taxa_acao` | TODAS | `Recompensa.taxa_acao` −0,01 (`knobs.py:258`) | Σ (a − a_anterior)² da ação crua | nenhum | — | `recompensas.py:126-128,687-690` |
| 7 | `energia` | TODAS | `Recompensa.energia` −2e-5 (`knobs.py:259`) | Σ \|τ·q̇\|, com τ = `qfrc_actuator` | E | — | `recompensas.py:131-133,693-697` |
| 8 | `limite_junta` | TODAS | `Recompensa.limite_junta` −5,0 (`knobs.py:262`) | Σ_j relu(\|q−c\|/m − `limite_mole`)/(1 − `limite_mole`), com `limite_mole` 0,95 (`knobs.py:263`) | nenhum | todas as juntas abaixo de 95 % do curso | `recompensas.py:136-138,700-704`; `potencial.py:78-85` |
| 9 | `cerca_sola` | PEGAR_BOTAR | `Recompensa.cerca_sola` −0,05475 (`knobs.py:270`) | Σ pés em contato relu(inclinação da sola° − `lim_sola_graus`), com limite 5,0° (`knobs.py:277`) | nenhum | sola a ≤ 5° ou pé fora do chão | `recompensas.py:156-158,736-741` |
| 10 | `cerca_giro` | TODAS | `Recompensa.cerca_giro` −0,03965 (`knobs.py:271`) | relu(máx(−q_yaw_E, 0)° − 30°) + relu(máx(q_yaw_D, 0)° − 30°), `lim_giro_graus` 30,0° (`knobs.py:278`) | S1 | hip_yaw para dentro ≤ 30° | `recompensas.py:161-168,744-747` |
| 11 | `cerca_tronco` | TODAS | `Recompensa.cerca_tronco` −0,04596 (`knobs.py:272`) | relu(inclinação do torso° − (tronco_IK° + 15,0°)); `lim_tronco_graus` 15,0° (`knobs.py:279`) | S1 | inclinação dentro da faixa | `recompensas.py:171-173,750-758` |
| 12 | `cerca_com` | TODAS | `Recompensa.cerca_com` −22,09035 (`knobs.py:273`) | relu(`lim_com` − margem do CoM) com os dois pés em contato; `lim_com` 0,03 m (`knobs.py:280`) | S1 | um pé fora do chão, ou margem ≥ 0,03 m | `recompensas.py:176-179,761-764` |
| 13 | `cerca_torque` | TODAS | `Recompensa.cerca_torque` −0,29615 (`knobs.py:274`) | Σ juntas soltas relu(\|τ\|/τ_max − 0,8)/0,2 + Σ pares relu((\|τ_p\| + \|τ_r\|)/50 − 0,8)/0,2 | S1 | torque abaixo de 80 % | `recompensas.py:182-193,767-772` |
| 14 | `cerca_pelve` | TODAS | `Recompensa.cerca_pelve` −4,45279 (`knobs.py:275`) | relu(\|h_pelve − pelve_IK\| − `lim_pelve`), `lim_pelve` 0,08 m (`knobs.py:283`) | S1 | desvio ≤ 0,08 m | `recompensas.py:196-198,775-788` |
| 15 | `cerca_recuo` | PEGAR_CARREGAR_BOTAR | `Recompensa.cerca_recuo` −10,39719 (`knobs.py:276`) | relu(\|recuo − recuo_IK\| − `lim_recuo`), `lim_recuo` 0,08 m (`knobs.py:284`) | nenhum | desvio ≤ 0,08 m; ou BOTAR com alvo B3 ou −1 | `recompensas.py:201-203,791-802` |
| 16 | `contato_fora` | TODAS | `Recompensa.contato_fora` −1,0 (`knobs.py:299`) | Contatos de corpos fora dos pés com o terreno (> `Recompensa.contato_fora_N` = 1,0 N, `knobs.py:300`) + autocolisões + contatos robô–caixa (> 1,0 N por fatia, 12 fatias) fora da janela | nenhum | sem contato; o contato com a caixa vale 0 com o alvo em P1, P2, P3, B1 ou B2 e com a tarefa CARREGAR | `recompensas.py:58-59,206-211,805-813` |
| 17 | `contato_laje` | PEGAR_BOTAR | `Recompensa.contato_laje` −2,0 (`knobs.py:301`) | Σ de 3 sensores de clamp((F − joelho)/saturação, 0, 1) | nenhum | força da laje ≤ joelho em todos os sensores | `recompensas.py:887-896` |
| 18 | `freio` | PEGAR_CARREGAR_BOTAR | `Recompensa.freio` −6/29 ≈ −0,2069 (`knobs.py:304`) | Σ_j relu(\|q̇\|/v_max − 1)² × 1,5^`v3_freio` × min(s̄/portão, 1) | `freio_fator`^freio; rampa | todas as juntas abaixo do teto; ou rampa 0 | `recompensas.py:899-919` |
| 19 | `progresso` | TODAS | +1,0 fixo (`env_cfg.py:57`) | Renda por segundo, seção 7.3 | h, etapa | `v3_perdeu`, ou tarefa sem caixa sem a pega fechada | `recompensas.py:286-291,816-884` |
| 20 | `sacudida` | PEGAR_CARREGAR_BOTAR | `Recompensa.sacudida` −0,05 (`knobs.py:353`) | Σ(Δv/(dt·5,04))² + Σ(Δω/(dt·31,56))² da caixa | D | alvo fora de P3 e B1 | `recompensas.py:339-342,922-945` |
| 21 | `aperto` | PEGAR_CARREGAR_BOTAR | `Recompensa.aperto` −0,05 (`knobs.py:355`) | min(F_esq, F_dir)/`F_ref` | D | alvo fora de P3 e B1 | `recompensas.py:948-957`; `potencial.py:55-62` |
| 22 | `perda_caixa` | TODAS | −1,0 fixo (`env_cfg.py:57`); `Recompensa.perda_caixa` 240,0 (`knobs.py:367`) | `v3_perdeu_agora × 240,0` | nenhum | `v3_perdeu_agora` False | `recompensas.py:311-313,960-962` |
| 23 | `terminacao` | TODAS | `Recompensa.terminacao` −300,0 (`knobs.py:366`) | 1 em terminação que não é `time_out` | nenhum | sem terminação ou só `time_out` | `recompensas.py:214-216,965-968` |

Notas por termo:

- O termo 22 vale `−1,0 × 240,0 × 0,02 = −4,8` por pulso de perda; o termo 23, `−300,0 × 0,02 = −6,0` por terminação.
- `sacudida` e `aperto`: o fator `d/5` vale 0 no degrau 0 e 1 no degrau 5. `Recompensa.sacudida_norma` = (5,04; 31,56) (`knobs.py:358`); `Recompensa.F_ref` = 49,0 N (`knobs.py:359`).
- `cerca_com`, `cerca_torque`, `cerca_pelve`, `cerca_giro` e `cerca_tronco` têm o fator `S1`, que vale 1 no BOTAR (`recompensas.py:453-459`).
- A tabela `TAU_MAX` por junta vem de `Recompensa.tau_max` (`knobs.py:286-292`); os pares de torque (cintura e dois tornozelos), de `Recompensa.pares_torque` (`knobs.py:293-297`); `tau_par` 50,0 (`knobs.py:298`); `lim_torque` 0,8 (`knobs.py:281`); `banda_torque` 0,2 (`knobs.py:282`).
- `contato_fora`, `cerca_recuo`, `sacudida` e `aperto` dependem do índice do alvo (coluna "Vale zero quando").

⚠ Divergência: o docstring de `n_terminacao` diz "Peso −200 no knob"; o knob `Recompensa.terminacao` vale −300,0 (`recompensas.py:215`; `knobs.py:366`).

⚠ Divergência: o docstring de `cerca_sola` diz "Todas as tarefas"; o `env_cfg` a registra só com as tarefas PEGAR e BOTAR (`recompensas.py:737`; `env_cfg.py:48`).

### 7.3 Fórmulas dos termos principais

#### 7.3.1 `progresso`

`progresso` é a renda por segundo do BOTAR. Ele soma três parcelas (`n_progresso`, `recompensas.py:286-291`):

`r = renda_etapa · etapa · h_ef + renda_phi · Φ_pago + renda_conquista · conquistas`

Knobs: `Recompensa.renda_etapa` 6,0 (`knobs.py:339`); `Recompensa.renda_phi` 3,0 (`knobs.py:338`); `Recompensa.renda_conquista` 2,0 (`knobs.py:343`).

A tabela de estados do env no BOTAR (`recompensas.py:845-884`). Ela usa as variáveis do código: `sem_caixa`, `bota_fechada`, `fecho_pega`.

| Estado do env | `etapa` | `Φ_pago` | `conquistas` | `h_ef` |
|---|---|---|---|---|
| B1, B2 ou B3 aberto, sem perda e sem `v3_bota_fechada` | 4 (B1), 5 (B2), 6 (B3) | Φ da etapa (seção 7.3.2) | B1: `c₀ + c₁`; B2 e B3: 0 | `v3_h` |
| `v3_bota_fechada` ligado e sem perda: do fecho do B3 até o fim do episódio, na tarefa BOTAR e no ANDAR final | 7 | 0 | 0 | 1 |
| Caixa perdida (`v3_perdeu`), com precedência sobre as outras linhas | 0 | 0 | 0 | 1 |

- `sem_caixa = v3_perdeu | (~v3_pega_fechada & tarefa ≠ PEGAR)` (`recompensas.py:851`). No BOTAR de um episódio normal `v3_pega_fechada` vale True (ligado no fecho do P3, `comando.py:492-496`; zerado só no reset, `comando.py:651`), então `sem_caixa` só liga com a perda.
- `etapa = 0` com `sem_caixa`; senão 7 com `bota_fechada`; senão 4 com `fecho_pega` (`pega_fechada` e índice < B1); senão o índice do alvo (`recompensas.py:853`). No BOTAR, `fecho_pega` vale False, pois o índice é ≥ B1.
- `paga_phi = ~(sem_caixa | bota_fechada | fecho_pega)`; `com_alvo = paga_phi & índice ≥ 0` (`recompensas.py:854-856`).
- Valores da parcela de etapa por segundo: B1 = 6,0 × 4 × h = 24·h; B2 = 30·h; B3 = 36·h; depois do B3 = 6,0 × 7 × 1 = 42.
- **Conquistas** (`mantem`): só valem sem `sem_caixa` e sem `bota_fechada`, e no alvo B1 (da tupla `_CAIXA_ALVO` = P3, B1; `recompensas.py:58,875`). No B1: `conquistas[0] + conquistas[1]·almofadas + conquistas[2]·com_peso`, com `almofadas` = 1 fora do P2 e `com_peso` = 0 fora do P3 e depois do fecho do P3 (`recompensas.py:874-882`). Logo, no B1 valem `c₀ + c₁`.
  - `c₀ = 1 − tanh(d_faces/(largura_tanh·sigma_palmas))`, `d_faces` = distância média das palmas à pose do P2 no frame da caixa (`recompensas.py:560,563`); `Alvos.largura_tanh` 2,0 (`knobs.py:221`); `Recompensa.sigma_palmas` 0,08 m (`knobs.py:344`).
  - `c₁ = tanh(min(F_esq, F_dir)/F_pega)`, com `F_pega = margem_pega·m·g/(2·μ)`; `Recompensa.margem_pega` 2,5 (`knobs.py:365`); μ = `DR.atrito_pad[0]` = 0,6 (`knobs.py:523`) (`recompensas.py:557-559`). F_esq e F_dir vêm das 12 fatias do sensor `caixa_faces` (`recompensas.py:345-356`).
- **Fator h** (renda da etapa): em cada etapa com alvo, `h' = min(1, h·exp(−dt/τ) + recarga·ganho)` (`n_pressa`, `recompensas.py:294-296`), com `ganho = relu(Φ − melhor Φ da etapa)`; `Recompensa.tau_renda` 3,0 s (`knobs.py:347`); `Recompensa.recarga_renda` 5,0 (`knobs.py:348`); `dt` 0,02 s. O `v3_phi_melhor` vira `max(melhor, Φ)` (`recompensas.py:858-869`). O h volta a 1 e o melhor Φ a 0 na abertura de cada etapa (`comando.py:504,561`). O h multiplica **só** a parcela da etapa (`h_ef`, `recompensas.py:870`).
- `v3_phi` guarda o Φ do alvo corrente (−1 sem alvo) para as métricas (`recompensas.py:883`).

#### 7.3.2 Φ de cada etapa

Φ é a proximidade da etapa corrente, em [0, 1] (`phi_do_alvo`, `recompensas.py:274-283`):

`Φ = (1 − tanh(d / (largura_tanh · σ))) · fator · Π portões`

com `Alvos.largura_tanh` 2,0 (`knobs.py:221`) e σ = `v3_sigma` com piso 1e-6 (`recompensas.py:34,281`). `Φ = 0` com índice < 0.

Distância `d` por etapa (`distancia_do_alvo`, `recompensas.py:224-244`, com os valores finais de `Alvos`):

| Etapa | `d` |
|---|---|
| B1 | hypot(‖caixa − alvo B1‖, `escala_apoio_b1`·déficit), com déficit = relu(1 − F_apoio/(`apoio_b1_mg`·peso)); `Alvos.escala_apoio_b1` 0,10 m (`knobs.py:193`); `Alvos.apoio_b1_mg` 0,8 (`knobs.py:192`) |
| B2 | Média das duas palmas de ‖palma no frame da caixa − palma alvo‖; a palma alvo do B2 tem x = 0, y = ±(a + `palma_face` + `b2_afasta`), z = 0; `palma_face` 0,045 m (`knobs.py:178`), `b2_afasta` 0,05 m (`knobs.py:196`) (`alvos.py:138-154`) |
| B3 | hypot(hypot(Δpelve/`tol_pelve_b3`, Δtronco°/`tol_tronco_b3_graus`), Δpalmas/`tol_palmas_b3`), adimensional; tolerâncias finais `Alvos.tol_pelve_b3` 0,03 m, `tol_tronco_b3_graus` 5,0° e `tol_palmas_b3` 0,05 m (`knobs.py:198-200`, valores de `Detector.tol_pelve`, `tol_tronco_graus` e `tol_palma`, `knobs.py:166-168`). A pelve do Δpelve é a altura do sensor `altura_pelve` menos a pelve de referência do alvo (`recompensas.py:536`), e não a medida do fecho (seção 4.5) |

Fator e portões (`recompensas.py:548-568`; `portoes_do_alvo`, `recompensas.py:247-271`; `potencial.rampa`, `potencial.py:23-29`):

| Portão | B1 | B2 | B3 |
|---|---|---|---|
| `fator` (pouso) | `fator_pouso(v_z da caixa, fundo sobre a laje, σ_pouso, 0,05, 0,10)` | 1 | 1 |
| `desloc` | 1 | `rampa(desloc_apoio; 0,02; 0,06)` | `rampa(desloc_apoio; 0,02; 0,06)` |
| `incl` | `rampa(inclinação da caixa; incl_lim; 15,0°)` | idem | 1 |
| `maos` | 1 | 1 | 1 |
| `tronco` | `rampa(relu(incl_torso° − tronco_IK°); 15,0°; 15,0°)` | idem | 1 |
| `recuo` | `rampa(\|recuo − recuo_IK\|; 0,08; 0,30)` | idem | 1 |
| `com` | `rampa(relu(0,03 − margem CoM); 0; 0,03)` com os dois pés em contato; senão 1 | idem | idem |

Knobs: `Alvos.faixa_tronco_graus` 15,0° (`knobs.py:226`); `Alvos.delta_tronco_graus` 15,0° (`knobs.py:229`); `Alvos.faixa_recuo` 0,08 m (`knobs.py:227`); `Alvos.delta_recuo` 0,30 m (`knobs.py:230`); `Alvos.faixa_com` 0,03 m (`knobs.py:228`); `Alvos.delta_com` 0,03 m (`knobs.py:231`); `Recompensa.altura_pouso` 0,05 m e `Recompensa.altura_pouso_inicio` 0,10 m (`knobs.py:369-370`); σ_pouso por degrau (seção 5.5). O `maos` do Φ vale 1 (a firmeza da pega entra como conquista `c₁`, `recompensas.py:566`).

`fator_pouso`: `1 − s·(1 − exp(−(v/σ_pouso)²))`, com `s = clip((altura_inicio − fundo)/(altura_inicio − altura), 0, 1)` e `fundo` = z da caixa − meia aresta − topo da laje de destino (`potencial.py:40-52`; `recompensas.py:548-551`). O fator vale 1 com o fundo acima de 0,10 m.

⚠ Divergência: o docstring de `fator_pouso` diz que o fator multiplica o Φ "no B1 e no B2" (`potencial.py:48`). O código aplica o fator só ao B1: `_POUSO = _ix("B1")` (`recompensas.py:57`) e `torch.where(_em(indice, _POUSO), pouso, 1)` (`recompensas.py:551`).

⚠ Divergência: o docstring de `por_degrau` lista "o fator do B2 no `progresso`" entre os chamadores (`recompensas.py:448-449`). Em `progresso` não existe fator de degrau no B2; os chamadores reais são `incl_lim`, o pouso de `_phi_env`, `contato_laje`, `sacudida`, `aperto` e `terminacoes.chegada`.

`recuo` medido: `‖xy da caixa − xy da pelve‖` (`recompensas.py:531`). `recuo_IK`: `_recuo_ref`, que no B1 e no B2 usa `ref.recuo(topo_destino, a)` (`recompensas.py:509-516`; `_topo_do`, `recompensas.py:496-498`). A margem do CoM vem de `margem_com` (`recompensas.py:316-336`).

#### 7.3.3 `cerca_recuo`, `cerca_pelve`, `cerca_tronco`, `cerca_com`

- `cerca_recuo = relu(|recuo − recuo_IK| − 0,08)` (`recompensas.py:201-203`). Vale 0 no BOTAR com o alvo B3 ou −1 (`recompensas.py:801`).
- `cerca_pelve = relu(|h_pelve − pelve_IK| − 0,08)`. `pelve_IK` = `ref_do_alvo(índice)[pelve]` quando a tarefa é PEGAR ou BOTAR e o índice ≥ 0; senão `pelve_de_pe` (`recompensas.py:775-788`). Para B1 e B2 a pelve de referência é `ref.pelve_z(topo_destino, a)`; para B3, `pelve_de_pe` (`observacoes.py:127-128`).
- `cerca_tronco = relu(inclinação do torso° − (tronco_IK° + 15,0°))`. `tronco_IK` é `ref.tronco(topo_destino, a)` em B1 e B2, e 0 em B3 e sem alvo (`recompensas.py:753-758`; `observacoes.py:129`).
- `cerca_com = relu(0,03 − margem)` com as duas solas em contato. O CoM combina o CoM do robô e o da caixa quando `v3_pegou & ~v3_perdeu` e o índice está em P3, B1 ou B2; com o B3 a caixa não entra (`recompensas.py:419-435,431`). A margem é a distância com sinal do CoM às arestas do fecho convexo dos 8 cantos das solas (`recompensas.py:316-336,473-479`).

#### 7.3.4 `contato_laje`

`contato_laje = Σ_{s ∈ {corpo, palma, dorso}} clip((F_s − joelho)/saturação, 0, 1)` (`recompensas.py:887-896`):

- `F_s` é o módulo da força líquida (`reduce="netforce"`, 1 slot) do sensor `s` contra a laje (`amax` sobre o slot único, `recompensas.py:894`): `corpo_prateleira` (geoms `pelvis_collision`, `torso_collision`, `*_hip_collision`, `*_thigh_collision`), `palma_prateleira` (`*_palm_pad`), `dorso_prateleira` (`*_hand_back_pad`) (`cena.py:45-50, 368-373`).
- `joelho = por_degrau(50,0; 10,0)` N (`Recompensa.laje_joelho_N`, `knobs.py:302`).
- `saturação = max(100,0 − joelho, 1e-6)`, com `Recompensa.laje_saturacao_N` 100,0 N (`knobs.py:303`).

#### 7.3.5 `freio`

`freio = Σ_j relu(|q̇_j|/v_max,j − 1)² · freio_fator^v3_freio · min(s̄/portão, 1)` (`recompensas.py:899-919`).

- Regime pelo total do comando `‖cmd_xy‖ + |cmd_z|`: parado se `< Recompensa.freio_limiar_andar` 0,05 (`knobs.py:308`); andando se `< freio_limiar_correr` 1,5 (`knobs.py:309`); correndo no resto (`recompensas.py:906-909`). No BOTAR o total é 0 e o regime é **parado**, com a tabela `Recompensa.vel_max_parado`: 1,5 rad/s em todas as juntas, exceto os punhos (`.*wrist.*`), com 1,0 rad/s (`knobs.py:311-317`; `recompensas.py:42-44`).
- `freio_fator` = 1,5 (`knobs.py:305`); `v3_freio` = nível do BOTAR + 1 (seção 5.4).
- `s̄` = média de `s` das células abertas da tarefa do env (0 se nenhuma está aberta); `portão` = `Balanceador.portao` 0,75 (`knobs.py:391`) (`recompensas.py:916-918`).

#### 7.3.6 `sacudida` e `aperto`

- `sacudida = [Σ(Δv_lin/(dt·5,04))² + Σ(Δω/(dt·31,56))²] · (d/5)`, com Δ = velocidade da caixa no mundo (6 componentes) menos a da chamada anterior (`recompensas.py:339-342,934-945`). O estado `v_ant` anda mesmo fora da tarefa (`recompensas.py:942`). Só vale com o alvo em P3 ou B1 (`recompensas.py:943`).
- `aperto = min(F_esq, F_dir)/F_ref · (d/5)`, só com o alvo em P3 ou B1 (`recompensas.py:948-957`).

#### 7.3.7 `perda_caixa` e `terminacao`

- `perda_caixa = v3_perdeu_agora × 240,0` (`recompensas.py:311-313`); `v3_perdeu_agora` é um pulso escrito no `compute` da cadeia no passo da perda e lido pela recompensa no passo seguinte (`comando.py:411`; seção 8.2).
- `terminacao = terminated & ~time_out` (`recompensas.py:214-216,965-968`).

---

## 8. Terminações

### 8.1 Tabela

O env_cfg registra cinco terminações (`env_cfg.py:137-147`). A terminação marcada `time_out=True` conta como truncamento; as outras contam como `terminated` (`mjlab/managers/termination_manager.py:102-113`). Um env terminado ou truncado faz reset no mesmo passo (`mjlab/envs/manager_based_rl_env.py:444-448`; `auto_reset` True, `:144`).

| Termo | Tarefas | Tipo | Condição exata | Ref. |
|---|---|---|---|---|
| `time_out` (`fim_da_cadeia`) | Nenhuma máscara | Truncamento | `episode_length_buf ≥ max_episode_length` **ou** `v3_fim` **ou** a travessia do terreno do ANDAR (`pose_ult` a mais de `Terreno.dist_travessia` = 3,5 m da origem da célula; exige `v3_fatia` = ANDAR, então vale sempre False no BOTAR). `max_episode_length` = ceil(`Cadeia.episodio_s`/0,02) = 1000 passos, com `Cadeia.episodio_s` 20,0 s (`knobs.py:160`) | `terminacoes.py:53-72`; `mjlab/envs/manager_based_rl_env.py:280-283` |
| `queda` | TODAS | Terminação | No BOTAR (ramo de manipulação): inclinação da base do robô > 70° **ou** folga do joelho mais baixo sobre o terreno < 0,05 m | `terminacoes.py:35-50,86-112`; `Terminacao.incl_manip_graus` 70,0° (`knobs.py:377`); `Terminacao.joelho_min` 0,05 m (`knobs.py:378`) |
| `caixa_perdida` | PEGAR_CARREGAR_BOTAR | Terminação **só se** `caixa_perdida_vira_andar` for False | `v3_perdeu_agora & (não vira_andar)`; com a chave True (`knobs.py:161`) o termo nunca dispara | `terminacoes.py:75-83,115-125`; `env_cfg.py:141-142` |
| `chegada` | BOTAR | Terminação | Primeiro contato caixa–laje (`apoio_caixa.compute_first_contact`) **e** `−v3_vz_caixa_ant > limite`, com `limite = por_degrau(1,0; 0,25)` m/s | `terminacoes.py:128-151`; `Terminacao.chegada` (`knobs.py:381`) |
| `batente` | TODAS | Terminação | Alguma das 29 juntas com `\|q − centro\|/meio > batente_frac` **ou** `\|hip_yaw\| > hip_yaw_max` em alguma perna, **e** `v3_degrau ≥ degrau_batente` | `terminacoes.py:154-183`; `Terminacao.batente_frac` 1,0 (`knobs.py:383`); `Terminacao.hip_yaw_max` 1,0 rad (`knobs.py:382`); `Refino.degrau_batente` 1 (`knobs.py:547`) |

Detalhes das condições.

- **`queda`.** A inclinação é o ângulo entre o eixo z da base e a vertical, `acos(−projected_gravity_b[2])`; a comparação é `incl_rad > radianos(70°)` (`terminacoes.py:97,47`). A folga do joelho é a altura medida pelo sensor `altura_joelhos`, no joelho de menor folga (`terminacoes.py:103-107,47`). Se a caixa foi perdida e a tarefa virou ANDAR, o ramo muda: inclinação > `Terminacao.incl_marcha` 0,8 rad (`knobs.py:375`) **ou** (z da pelve − z do chão) < `Terminacao.base_min` 0,20 m (`knobs.py:376`) (`terminacoes.py:46`).
- **`chegada`.** `v3_vz_caixa_ant` guarda a velocidade vertical da caixa do passo anterior (escrita no fim de `compute`, `comando.py:629`). O limite vem do degrau do BOTAR (seção 5.5). O sensor `apoio_caixa` é o contato entre a geom da caixa e a geom da laje (`cena.py:361-367`).
- **`batente`.** O centro e o meio do curso vêm dos limites duros do modelo (`terminacoes.py:176-181`). `>` estrito: 1,0 exato não termina (`terminacoes.py:162-164`). Os dois gatilhos são alternativos.
- **Tarefa usada.** A máscara das terminações lê `v3_tarefa` (`terminacoes.py:112,125,151,183`), que muda para ANDAR com a perda.

### 8.2 A perda da caixa

A perda não termina o episódio com a chave `Cadeia.caixa_perdida_vira_andar` = True. A condição é calculada em `CadeiaCmd._perda` (`comando.py:394-420`), no `compute` da cadeia (depois das terminações e das recompensas do mesmo passo).

`perda = (caiu | arremesso | fora_da_laje) & (tarefa ≠ ANDAR) & (episode_length_buf > 0)` (`comando.py:409-410`), com (`comando.py:59-65`):

- **caiu**: `h_caixa − v3_meia < Terminacao.caixa_folga_chao` (0,02 m, `knobs.py:379`). `h_caixa` é a altura do centro da caixa sobre o terreno, do sensor `altura_caixa`.
- **arremesso**: `v3_pegou & (‖v_caixa − v_base‖ > Terminacao.v_solta) & fora_do_alvo`. `v_solta` 1,2 m/s (`knobs.py:380`). No BOTAR, `fora_do_alvo` = `‖caixa − v3_alvo_w‖ > Alvos.raio_b1` (0,10 m, `knobs.py:191`, o valor final, não o da zona) (`comando.py:407-408`).
- **fora_da_laje**: vale só com o índice em P0, P1 ou P2; não vale no BOTAR.

**Efeito com a chave ligada** (`comando.py:412-420`):

- `v3_perdeu_agora` recebe o pulso (`comando.py:411`).
- Nos envs com perda: `v3_perdeu` = True, `v3_alvo_idx` = B3, `v3_phi` = −1, `v3_h` = 1, `v3_phi_melhor` = 0, `v3_relogio_alvo` = 0, `v3_sigma` = σ do B3, e a tarefa publicada vira ANDAR (`_publica`).
- O trecho congela até o fim do episódio (`comando.py:610-613`). O episódio termina por `time_out` aos 20 s, ou por outra terminação.
- O B3 desse env fecha pela pose de pé, com o invariante forçado a 1: `inv` do B3 da perda vale 1 porque `desl_apoio` é zero (`comando.py:460-461,475-481`).

A terminação `caixa_perdida` e a recompensa `perda_caixa` leem `v3_perdeu_agora` só no passo seguinte ao da detecção (0,02 s de atraso), porque o comando roda depois das terminações e das recompensas (seção 4.1). A medida `h_caixa` do ramo `caiu` vem do sensor `altura_caixa` (`comando.py:336`), e `sim.sense()` roda depois do `command_manager.compute` (`mjlab/envs/manager_based_rl_env.py:456, 463`). Não determinado pelo código lido: quantos passos depois da física o ramo `caiu` cobra; o comentário de `comando.py:398-399` diz dois passos, e este documento não o confirma pelo código do sensor.

### 8.3 O que acontece depois de uma terminação

1. A recompensa `terminacao` (−6,0 por passo da terminação: −300,0 × 0,02) soma no passo da terminação (seção 7).
2. O env faz reset no mesmo passo (seção 2). O reset registra o BOTAR aberto com `falhou = ~time_outs` (seção 5.2).
3. O algoritmo de PPO soma ao reward do passo `γ·min(V̄_g, 0)` quando o env terminou por terminação (não por `time_out`) ou teve o pulso `v3_perdeu_agora`; `V̄_g` é a média do valor no rollout anterior do grupo `g` do crítico (um grupo por tarefa e etapa); com `V̄_g ≥ 0` nada muda. No primeiro rollout `v_grupo` ainda é `None` e a soma não acontece (`algoritmo.py:98-135, 160-169`). γ = 0,99 (`mjlab/tasks/velocity/config/g1/rl_cfg.py:37`).
4. O mjlab loga `Episode_Termination/<termo>` com a contagem por termo (`mjlab/managers/termination_manager.py:88-100`).

---

## 9. Observações próprias da tarefa

O ator e o crítico recebem os mesmos termos de base e histórico. O crítico recebe também canais privilegiados (`observacoes.py:361-422`). Esta seção lista o que a tarefa BOTAR muda ou usa. Os termos gerais (giro da base, gravidade, juntas, velocidade das juntas, ação, mapa) valem em todas as tarefas.

### 9.1 O ator

| Termo | Canais | O que vale no BOTAR | Ref. |
|---|---|---|---|
| `comando` | 3 | Zero (seção 3.1) | `observacoes.py:392`; `comando.py:900-902` |
| `fase` | 2 | (0, 1) fixo: a fase não anda sem comando | `observacoes.py:312-314`; `comando.py:872-873` |
| `tarefa` | 6 | One-hot de `v3_tarefa`; canal 3 = BOTAR; sem ruído | `observacoes.py:135-137` |
| `caixa` | 11 | Caixa estimada (8) + alvo (3), seção 9.2 | `observacoes.py:160-199` |

O ator **não** vê qual etapa (B1, B2, B3) está aberta, nem a zona z, nem o degrau. Ele infere a etapa pelo estado da caixa e do robô.

Cada termo do ator tem histórico de 5 passos, exceto o `mapa` (0) (`Contrato.blocos_ator`, `knobs.py:58-63`). Escala e ruído estão na seção 6.

### 9.2 O termo `caixa`

`caixa` tem 11 canais: 8 da `saida_b` do estimador e 3 do alvo (`observacoes.py:172-199`).

| Canais | Conteúdo | Ref. |
|---|---|---|
| 0–2 | Posição estimada da caixa no frame da base estimada | `estimador.py:378-385` |
| 3–5 | Rotvec da orientação relativa à base, reduzido pelas 24 rotações do cubo | `estimador.py:82-91, 384` |
| 6 | Meia aresta estimada (com viés do episódio) | `estimador.py:309, 385` |
| 7 | Idade da leitura, cortada em `Estimador.idade_max` (2,0 s) | `estimador.py:380, 385` |
| 8–10 | Alvo `v3_alvo_w` no frame da base estimada (a odometria) | `observacoes.py:195`; `estimador.py:191-194` |

Regras de zeragem (`observacoes.py:196-199`):

- Todos os 11 canais valem zero quando a tarefa é ANDAR (por exemplo, depois da perda e no ANDAR final).
- Os canais do alvo (8–10) valem zero quando `(tarefa == CARREGAR) | where(tarefa == BOTAR, v3_aviso, v3_alvo_idx < 0)`. No BOTAR, o alvo zera com o latch do aviso, e não com o índice (o índice cai no fecho do B3 e o aviso liga no mesmo `compute`, `comando.py:501,606-607`).
- O alvo no BOTAR é sempre o ponto de soltura do B1 (seção 3.2), nas três etapas.

### 9.3 O crítico (canais privilegiados)

Além dos termos do ator sem ruído e sem atraso (`observacoes.py:404`), o crítico tem estes termos próprios (`observacoes.py:405-418`).

| Termo | Canais | Conteúdo | Ref. |
|---|---|---|---|
| `vel_lin_base` | 3 (histórico 5) | Sensor `robot/imu_lin_vel` | `observacoes.py:406-407` |
| `caixa_verdade` | 7 | Posição (3), rotvec (3) e meia aresta (1) verdadeiras no frame da base; zero no ANDAR | `observacoes.py:267-273` |
| `alvo_verdade` | 3 | `v3_alvo_w` no frame da base verdadeira; zero com índice < 0 | `observacoes.py:276-279` |
| `alvo_macro` | 11 | One-hot do estado da cadeia (seção 9.4) | `observacoes.py:282-295` |
| `referencia` | 8 | Palmas (6), pelve (1) e tronco (1) do alvo corrente; B1 e B2 usam `v3_topo_destino` | `observacoes.py:116-130, 321-324`; `recompensas.py:496-506` |
| `niveis` | 4 | `nível/max_nivel` de ANDAR, PEGAR, CARREGAR, BOTAR; igual em todo env | `observacoes.py:327-335` |
| `degrau` | 1 | `v3_degrau/5` (o do BOTAR) | `observacoes.py:338-340` |
| `massa` | 1 | `v3_massa/5` kg | `observacoes.py:343-345` |
| `forca_caixa` | 1 | Σ do módulo das 12 fatias de força robô→caixa, dividida por 100 N | `observacoes.py:348-356`; `metricas.py:338-345` |
| `privilegiados` | 4 | (`v3_marcha_on`, `v3_h`, relu(`v3_phi_melhor` − max(`v3_phi`, 0)), `z` da etapa corrente; 0 sem etapa) | `observacoes.py:298-318` |
| `tarefa_critico` | 6 | One-hot da tarefa, por último | `observacoes.py:417` |

Na referência do alvo: B1 e B2 usam a pelve e o tronco de `ref.pelve_z(topo_destino, a)` e `ref.tronco(topo_destino, a)`; só o B2 tem palmas-alvo (as do afastamento); no B3 a pelve é `pelve_de_pe` e o tronco, 0 (`observacoes.py:116-130`; `alvos.py:138-154`).

### 9.4 O one-hot `alvo_macro`

O `alvo_macro` tem `len(ALVOS) + 4` = 11 canais (`observacoes.py:46`). A posição que vale 1 é escolhida em ordem de prioridade crescente (`observacoes.py:289-295`):

1. O índice do alvo (0 a 6; B1 = 4, B2 = 5, B3 = 6), com −1 indo ao canal 10 ("nenhum").
2. Canal 7: pega fechada com o índice no P3.
3. Canal 8: `v3_bota_fechada`.
4. Canal 9: `v3_perdeu` (prioridade máxima).

O algoritmo usa essa fatia como grupo da normalização da vantagem (um grupo por tarefa e etapa) (`algoritmo.py:98-112`; `runner.py:61-62`).

### 9.5 Normalização

Treze termos ficam fora do normalizador empírico, com média 0 e desvio fixo (`runner.py:32-34`; `algoritmo.py:69-83`): `tarefa`, `mapa`, `alvo_macro`, `niveis`, `degrau`, `massa`, `tarefa_critico` e `privilegiados` têm desvio 1,0; `caixa`, `caixa_verdade`, `alvo_verdade` e `referencia` têm 0,1; `forca_caixa` tem 0,3. Os demais termos passam pelo normalizador empírico do mjlab, com piso `PISO_DESVIO` = 0,05 no desvio (`algoritmo.py:66, 79`).

---

## 10. Métricas

O mjlab registra as métricas por episódio e loga a média (`mjlab/managers/metrics_manager.py:107-133`; `Episode_Metrics/<nome>`). Métricas com `reduce="last"` devolvem uma flag ("teve o evento") e logam o valor do episódio no reset, com o prefixo `Episodio/`, que o rsl_rl média sobre os elementos de todos os passos da iteração (`metricas.py:81-90`; `rsl_rl/utils/logger.py:180-196`). Os termos estão montados em `metricas.termos` (`metricas.py:522-571`).

### 10.1 Métricas que dependem do BOTAR

| Chave | O que mede | Quando conta | Ref. |
|---|---|---|---|
| `Episodio/phi_B1`, `phi_B2`, `phi_B3` (e `Episode_Metrics/phi_<alvo>`: flag) | Média do Φ (`v3_phi`) nos passos com o alvo corrente igual à etapa e tarefa com caixa | Passos com `v3_alvo_idx == etapa` e tarefa PEGAR, CARREGAR ou BOTAR | `metricas.py:245-270, 558-559` |
| `Episodio/tempo_B1`, `tempo_B2`, `tempo_B3` | Segundos com a etapa corrente | Idem | `metricas.py:273-294, 560` |
| `Episodio/velocidade_de_chegada` | Velocidade de descida da caixa no primeiro contato caixa–laje (m/s, positiva); a mesma que a terminação `chegada` julga | Episódios com toque (`v3_v_chegada ≥ 0`) | `metricas.py:376-395`; `comando.py:631-637` |
| `Episodio/deslocamento_depois_do_apoio` | Maior deslocamento xy da caixa desde `v3_ponto_apoio` | Passos em B2 ou B3 sem `v3_perdeu` | `metricas.py:398-424` |
| `Episodio/aceleracao_da_caixa` | Média do custo da sacudida da caixa sem o fator de degrau | Passos com alvo P3 ou B1 | `metricas.py:459-491` |
| `Episode_Metrics/fracao_avisos` | Fração dos passos do episódio com o latch `v3_aviso` ligado | Todos os passos | `metricas.py:510-519, 567` |
| `Episode_Metrics/forca_robo_caixa` | Média da força total robô→caixa (soma de módulos), N | Todos os passos | `metricas.py:338-345, 562` |
| `Episode_Metrics/idade_leituras` | Idade média da leitura da caixa (s); 0 em ANDAR | Todos os passos | `metricas.py:494-507, 566` |

### 10.2 Métricas gerais registradas (valem no BOTAR)

`tempo_de_voo`, `pico_de_altura`, `velocidade_de_escorrego`, `velocidade_de_junta`, `fracao_do_curso` (máximo do episódio), `forca_de_pouso`, `passo_medio` (conta toques só com comando acima de `Comando.limiar_canal` 0,02, `knobs.py:138`, então fica sem leitura no BOTAR), `fracao_mapa_visto` (limiar `Percepcao.limiar_visto` 0,5, `knobs.py:503`) (`metricas.py:532-556`).

Métricas de outras etapas, registradas mas que não contam no BOTAR: `phi_P0..P3` e `tempo_P0..P3`, `inclinacao_caixa_p3`, `erro_p3_*`, `deslocamento_caixa_laje`, `fracao_fora_da_plataforma` (`metricas.py:558-570`).

### 10.3 Log do currículo (`Curriculum/v3/<chave>`)

O termo `curriculo.v3` devolve um dict a cada reset, e o mjlab o publica como `Curriculum/v3/<chave>` (`curriculo.py:103-143`; `mjlab/managers/curriculum_manager.py:95-110`). As chaves do BOTAR:

| Chave | Valor |
|---|---|
| `s_BOTAR_botar` | EMA da nota da célula (seção 5.2) |
| `nivel_BOTAR`, `subnivel_BOTAR`, `degrau_BOTAR` | Nível, subnível e degrau do BOTAR |
| `z_B1`, `z_B2`, `z_B3` | Zona atual da etapa |
| `taxa_B1`, `taxa_B2`, `taxa_B3` | Última taxa de fecho por visita (−1 se sem medida) |
| `frac_C` (e `frac_ANDAR`, `frac_B`) | Fração de sorteio da fatia C, a de `fracoes_fatias` (seção 1.3) |
| `degrau_global`, `restricao`, `atraso_max`, `mu_min_pad` | Valores compartilhados (seção 5.6) |

### 10.4 Logs do mjlab e do PPO

- `Episode_Reward/<termo>`: média, sobre os envs que fazem reset, da soma do episódio de cada termo de recompensa, dividida por `episode_length_s` = `Cadeia.episodio_s` 20,0 s (`knobs.py:160`; `env_cfg.py:105`; `mjlab/managers/reward_manager.py:100-114`). Vale para os 23 termos da seção 7.
- `Episode_Termination/<termo>`: contagem por terminação (`mjlab/managers/termination_manager.py:88-100`).
- `Policy/std/<junta>`: desvio da ação por junta (`runner.py:68-86`).
