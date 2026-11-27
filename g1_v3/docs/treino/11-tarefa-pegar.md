# Tarefa PEGAR

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:
- Código da v3 (`g1_v3/`): `env_cfg.py`, `tarefa.py`, `comando.py`, `alvos.py`, `potencial.py`, `eventos.py`, `balanceador.py`, `curriculo.py`, `recompensas.py`, `terminacoes.py`, `observacoes.py`, `estimador.py`, `metricas.py`, `knobs.py`, `cena.py`, `algoritmo.py`, `runner.py`.
- Biblioteca `mjlab`: `envs/manager_based_rl_env.py`, `managers/command_manager.py`, `managers/event_manager.py`, `managers/reward_manager.py`, `managers/termination_manager.py`, `managers/curriculum_manager.py`, `managers/observation_manager.py`, `managers/metrics_manager.py`, `envs/mdp/events.py`, `envs/mdp/dr/_core.py`, `envs/mdp/dr/_types.py`, `envs/mdp/dr/geom.py`, `envs/mdp/dr/joint.py`, `envs/mdp/dr/body.py`, `envs/mdp/dr/actuator.py`, `utils/buffers/delay_buffer.py`, `utils/noise/noise_cfg.py`, `actuator/actuator.py`, `entity/entity.py`, `tasks/velocity/config/g1/env_cfgs.py`, `tasks/velocity/config/g1/rl_cfg.py`.
- Biblioteca `rsl_rl`: `algorithms/ppo.py`.

Todo número abaixo traz o nome do knob e o valor lido em `knobs.py`, com a linha. Uma referência `mjlab/...` ou `rsl_rl/...` aponta para a biblioteca. Um valor que é literal no código, e não knob, recebe a palavra "literal".

---

## 1. O que é a tarefa e quando um env está nela

A tarefa PEGAR é o trecho do episódio em que o robô, de pé e parado, leva as duas palmas até uma caixa apoiada numa laje e a ergue até o ponto de transporte. O robô não anda nesse trecho. O comando de velocidade vale zero (seção 3).

### 1.1 Vocabulário

| Termo | O que é | Onde vive |
|---|---|---|
| Passo | Um passo da política. Dura `Contrato.dt_fisica` = 0,005 s vezes `Contrato.decimacao` = 4, ou seja 0,02 s. | `knobs.py:54-55`; `env_cfg.py:103-104`; `mjlab/envs/manager_based_rl_env.py:266-267` |
| Iteração | Janela de `Balanceador.passos_por_iteracao` = 24 passos (`knobs.py:402`), igual a `num_steps_per_env` do PPO (`tarefa.py:49`). O currículo fecha uma iteração quando `common_step_counter // 24` muda. | `curriculo.py:119-122` |
| Fatia | Grupo de tarefas que um env recebe no reset. Há três: ANDAR, B e C. | `knobs.py:43`; `eventos.py:31-32` |
| Trecho | Um bloco da cadeia de uma fatia, com uma tarefa publicada. | `comando.py:163-194` |
| Tarefa publicada | O buffer `v3_tarefa` (um inteiro por env). O índice da PEGAR é 1 em `TAREFAS`. | `knobs.py:42`; `comando.py:30`, `comando.py:511` |
| Alvo ou etapa | Subdivisão do PEGAR e do BOTAR. O PEGAR usa P0, P1, P2 e P3 (índices 0 a 3). | `knobs.py:44`; `alvos.py:16` |
| Célula | Unidade que o balanceador mede. O PEGAR tem uma célula, `("PEGAR", "pegar")`. | `balanceador.py:22-24` |
| Nível | Posição na tabela `Niveis.pegar`. Vai de 0 a 4. | `balanceador.py:72-74`; `knobs.py:417` |
| Subnível | Divisão do nível 0 do PEGAR em dois degraus (índices 0 e 1). | `balanceador.py:90-91`; `knobs.py:422` |
| Degrau | Passo do refino. Vai de 0 a `Balanceador.degraus` = 5 (`knobs.py:401`). Cada tarefa tem o seu. | `balanceador.py:237-248` |
| Degrau global | O menor degrau entre as quatro tarefas. | `balanceador.py:341-343` |
| Restrição | O menor valor de nível/nível máximo entre as quatro tarefas, entre 0 e 1. | `balanceador.py:345-348` |
| Zona | Escala `z` entre 0 e 1 por etapa. Com `z` = 1 o critério de fecho é largo. Com `z` = 0 vale o critério final. | `balanceador.py:105`, `comando.py:428-443` |

### 1.2 Quem está na PEGAR

Um env está na tarefa PEGAR quando `v3_tarefa == 1`. A máscara de todo termo de recompensa e das terminações `queda`, `caixa_perdida`, `chegada` e `batente` testa isso (`curriculo.py:93-100`). A terminação `time_out` não usa máscara (`terminacoes.py:58-72`). Ela inclui também a travessia do terreno do ANDAR, que exige `v3_fatia` = ANDAR e por isso nunca vale num env da PEGAR. Só os envs das fatias B e C passam pela PEGAR. A fatia ANDAR não tem trecho PEGAR (`comando.py:175-180`).

A cadeia de trechos de cada fatia vem de `comando.py:171-194`. Cada trecho sorteia uma duração no momento em que abre (`comando.py:578-579`).

| Fatia | Trechos, em ordem | Duração de cada trecho |
|---|---|---|
| ANDAR | ANDAR | `Cadeia.episodio_s` = 20,0 s (`knobs.py:160`) |
| B | ANDAR, PEGAR, CARREGAR | ANDAR: U(`Cadeia.espera_s`) = U(0,5; 1,0) s (`knobs.py:154`). PEGAR: atraso U(`Cadeia.atraso_pegar_s`) = U(0,0; 1,0) s depois do fecho do P3 (`knobs.py:158`). CARREGAR: U(`Cadeia.carregar_b_s`) = U(4,0; 7,0) s (`knobs.py:155`). |
| C | ANDAR, PEGAR, CARREGAR, BOTAR, ANDAR | ANDAR: U(0,5; 1,0) s. PEGAR: como na B. CARREGAR: U(`Cadeia.carregar_c_s`) = U(0,5; 1,0) s (`knobs.py:156`). BOTAR: atraso U(`Cadeia.atraso_botar_s`) = U(0,0; 1,5) s (`knobs.py:159`). ANDAR final: `Cadeia.andar_final_s` = 1,5 s (`knobs.py:157`). |

⚠ Divergência: o comentário de `knobs.py:158` diz que `atraso_pegar_s` é o "atraso antes do PEGAR". O código usa o valor como o tempo que o trecho PEGAR espera depois do aviso de fecho do P3 (`comando.py:201-216`, `comando.py:604-609`).

Todo episódio B ou C começa em ANDAR, não em PEGAR (`comando.py:659-660`). Nesse trecho de espera o env está na tarefa ANDAR, parado, e valem as recompensas da tarefa ANDAR (não descritas aqui). O comando do trecho é zero porque a fatia não é ANDAR (`comando.py:905`).

### 1.3 Quando o env entra e sai da PEGAR

| Evento | Condição exata | Efeito | Referência |
|---|---|---|---|
| Entrada | O relógio do trecho ANDAR de espera chega à duração sorteada: `relogio + dt >= duracao − 1e-6` (literal). | `v3_tarefa` passa a 1. `v3_alvo_idx` = −1 (sem alvo). O env espera, parado, sem alvo. | `comando.py:201-216`, `comando.py:568-594` |
| Abertura do P0 | `v3_alvo_idx < 0`, tarefa PEGAR e `v3_estimador.tem_leitura()` verdadeiro. | `v3_alvo_idx` = P0. O PEGAR fica pendente de registro (`self.pega` = True). | `comando.py:364-392` |
| Saída por avanço | `v3_aviso` verdadeiro (o P3 fechou) e o relógio do trecho, que só corre com o aviso, chega a U(0,0; 1,0) s. | O trecho seguinte abre (CARREGAR). | `comando.py:604-619`, `comando.py:201-216` |
| Saída por perda da caixa | `caixa_perdida_agora` verdadeiro, `v3_tarefa` ≠ ANDAR, `episode_length_buf` > 0, e `caixa_perdida_vira_andar` = True (`knobs.py:161`) para o efeito abaixo. | `v3_perdeu_agora` = True por um passo. `v3_perdeu` = True. `v3_alvo_idx` = B3. `v3_tarefa` passa a ANDAR. O trecho congela até o fim do episódio. | `comando.py:394-420`, `comando.py:610-613` |
| Saída por reset | Qualquer terminação ou o limite de tempo. | O episódio acaba. Um episódio novo sorteia outra fatia. | `manager_based_rl_env.py:436-447` |

Uma terminação por perda da caixa não existe com `caixa_perdida_vira_andar` = True: `n_caixa_perdida` devolve `perdeu_agora & (not vira_andar)` (`terminacoes.py:75-83`).

A função `caixa_perdida_agora` é verdadeira quando vale uma das três condições abaixo (`comando.py:59-65`, chamada em `comando.py:407-410`). No PEGAR, `fora_do_alvo` vale sempre True, porque só o BOTAR mede a distância ao alvo de soltura (`comando.py:407-408`).

| Condição | Expressão exata |
|---|---|
| Caiu | `h_caixa − v3_meia` < `Terminacao.caixa_folga_chao` = 0,02 m (`knobs.py:379`). Comparação estrita. `h_caixa` é a leitura do sensor `altura_caixa`. |
| Arremesso | `v3_pegou` e ‖v_caixa − v_base‖ > `Terminacao.v_solta` = 1,2 m/s (`knobs.py:380`) e `fora_do_alvo`. |
| Fora da laje | `v3_alvo_idx` entre P0 e P2 e o centro da caixa fora do topo da laje: algum eixo xy do centro da caixa, no frame da laje, passa de `Cena.prateleira_meia_xy` = 0,30 m (`knobs.py:89`; `comando.py:334-335`). |

O pulso `v3_perdeu_agora` é escrito no `compute` da cadeia, depois das terminações e das recompensas do mesmo passo. A recompensa `perda_caixa` o lê só no passo seguinte (`comando.py:395-399`).

Depois da perda a tarefa publicada é ANDAR. Valem então os termos de recompensa da tupla de ANDAR, e a terminação `queda` usa o critério de marcha: `Terminacao.incl_marcha` = 0,8 rad ou altura da pelve sobre o terreno menor que `Terminacao.base_min` = 0,20 m (`knobs.py:375-376`; `terminacoes.py:46`).

O trecho PEGAR não tem duração máxima. Sem o fecho do P3, ele dura até o fim do episódio (`comando.py:201-216`).

O limite de tempo do episódio é `Cadeia.episodio_s` = 20,0 s (`knobs.py:160`; `env_cfg.py:105`). Em passos, `max_episode_length = ceil(20,0 / 0,02)` = 1000 (`mjlab/envs/manager_based_rl_env.py:281-283`).

### 1.4 Sorteio da fatia

O sorteio da fatia acontece no reset do episódio, para cada env que reseta. É o primeiro evento de reset (`env_cfg.py:154`; `eventos.py:133-175`). Não há sorteio da tarefa: a fatia fixa a cadeia de trechos e os trechos fixam a tarefa.

Distribuição: escolha categórica com reposição, pelas frações de `fracoes_fatias()` (`eventos.py:161`; `balanceador.py:302-304`). O gerador é `v3_gerador`.

As frações dependem do estado do currículo (`balanceador.py:282-293`):

- Uma fatia está aberta se alguma célula dela está aberta. A fatia B tem as células PEGAR e CARREGAR. A fatia C tem a célula BOTAR (`balanceador.py:34`, `balanceador.py:287`).
- O teto da fatia ANDAR é 1,0. O teto de B e de C é `Balanceador.rampa_fatia_inicio` + `Balanceador.rampa_fatia_passo` × `_rampa[f]` = 0,05 + 0,02 × `_rampa[f]` (`knobs.py:397-398`; `balanceador.py:288`).
- As fatias abertas entram em ordem crescente de teto. A fração de cada uma é `min(teto, resto / (n_abertas − j))`, e o resto desce dessa fração (`balanceador.py:289-292`).
- Fatia fechada vale 0.
- O contador `_rampa[f]` sobe 1 a cada `Balanceador.rampa_fatia_its` = 12 fechos de iteração, contados desde a abertura da fatia. Nunca desce (`knobs.py:399`; `balanceador.py:189-193`).

A fatia B abre quando o nível do ANDAR chega a `Balanceador.nivel_abre_b` = 5 (`knobs.py:396`; `balanceador.py:209-211`). A abertura da B abre as células CARREGAR `reto` e PEGAR juntas. A fatia C abre quando a nota da célula PEGAR atinge o portão: `s >= Balanceador.portao` = 0,75 (`knobs.py:391`; `balanceador.py:231-232`). A abertura da C não exige o último subnível do PEGAR.

A fração do PEGAR no lote é `frac_B + frac_C`.

### 1.5 Ordem de execução de um passo e de um reset

Ordem de um passo (`mjlab/envs/manager_based_rl_env.py:378-470`):

1. Processa a ação. Roda `decimacao` = 4 subpassos de física (`manager_based_rl_env.py:421-427`).
2. Soma 1 em `episode_length_buf` e em `common_step_counter` (`manager_based_rl_env.py:430-431`).
3. Calcula as terminações (`manager_based_rl_env.py:436`).
4. Calcula as recompensas, cada uma × peso × `step_dt` (`manager_based_rl_env.py:440`; `mjlab/managers/reward_manager.py:118-127`). `scale_rewards_by_dt` vale True (`manager_based_rl_env.py:157`).
5. Calcula as métricas (`manager_based_rl_env.py:441`).
6. Reseta os envs terminados (`manager_based_rl_env.py:443-448`).
7. Roda `forward` (`manager_based_rl_env.py:454`).
8. Calcula os comandos, `CadeiaCmd` primeiro e `TwistV3` depois (`manager_based_rl_env.py:456`; `env_cfg.py:118-128`).
9. Roda os eventos de modo `step` e `interval` (`manager_based_rl_env.py:458-461`).
10. Roda `sim.sense`, que atualiza os sensores de altura e de raios (`manager_based_rl_env.py:463`).
11. Calcula as observações (`manager_based_rl_env.py:464`).

Consequência: as recompensas e as terminações de um passo leem a tarefa e o alvo que a cadeia publicou no passo anterior. As observações do passo leem o que a cadeia acabou de publicar.

Ordem de um reset (`manager_based_rl_env.py:553-590`): currículo `v3`, `sim.reset`, `scene.reset`, eventos de modo `reset`, reset do gerenciador de observação, de ação, de recompensa, de métricas, de currículo, de comando, de eventos e de terminação, e `episode_length_buf` = 0.

O env herda de `unitree_g1_rough_env_cfg` do `mjlab` (`env_cfg.py:91`). `env_cfg.py` troca por inteiro os campos de cena, sensores, terreno, ação, observações, comandos, recompensas, terminações, eventos, currículo e métricas (`env_cfg.py:93-218`).

---

## 2. Reset

### 2.1 Ordem

Para um env que reseta e vai para uma fatia B ou C, a ordem é esta.

1. Currículo `v3` (`curriculo.py:103-143`). Se `common_step_counter // 24` mudou desde a chamada anterior, fecha a iteração do balanceador (`curriculo.py:119-122`). O fecho pode mudar níveis, zona, degrau e frações antes dos sorteios abaixo. Se o teto de atraso do currículo difere de `DR.atraso_inicio[1]` = 0 (`knobs.py:511`), reaplica o teto de atraso (`curriculo.py:123-126`).
2. `sim.reset` e `scene.reset`.
3. Eventos de modo `reset`, na ordem do dicionário (`env_cfg.py:153-184`):

| # | Evento | Função |
|---|---|---|
| 1 | `fatia` | `eventos.sorteia_fatia` |
| 2 | `cena` | `eventos.posiciona_cena` |
| 3 | `base` | `eventos.reset_base_por_fatia` |
| 4 | `juntas` | `reset_joints_by_offset` do `mjlab` |
| 5 | `carga` | `eventos.carga` |
| 6 | `atrito_pad` | `eventos.atrito_pad` |
| 7 | `atrito_laje` | `eventos.atrito_laje` |
| 8 | `vies_imu` | `eventos.vies_imu` |
| 9 | `pd` (só no treino) | `eventos.pd` |
| 10 | `forca_motor` (só no treino) | `eventos.forca_motor` |
| 11 | `solref` (só no treino) | `eventos.solref` |
| 12 | `filtros` | `eventos.reinicia_filtros` |

O evento `com_caixa` só entra no cfg com `DR.com_caixa` verdadeiro (`env_cfg.py:182-183`). O knob vale False (`knobs.py:527`), então o evento não existe.

4. Reset do gerenciador de observação (zera os buffers de atraso e de janela, `mjlab/managers/observation_manager.py:248-253`).
5. Reset dos comandos. `CadeiaCmd.reset` roda primeiro: registra o PEGAR pendente, limpa os buffers do episódio e abre o primeiro trecho da fatia (`comando.py:639-660`). `TwistV3.reset` roda depois (`comando.py:879-895`).

### 2.2 O que cada evento põe ou sorteia

| Evento | Quando | O que faz | Distribuição e faixa | Referência |
|---|---|---|---|---|
| `fatia` | Reset | Sorteia a fatia (seção 1.4). Põe `v3_t_empurrao` = −∞. Sorteia se o episódio é de terreno e a coluna e a linha do terreno. Põe `terrain_levels`, `terrain_types` e `env_origins`. | Ver 1.4 e 2.3. | `eventos.py:133-175` |
| `cena` | Reset | Sorteia topo e massa da caixa. Põe laje e caixa. Grava `v3_topo`, `v3_massa`, `v3_caixa_xy`, `v3_rumo_cena`. | Ver 2.4. | `eventos.py:332-367` |
| `base` | Reset | Põe a base do robô atrás da caixa. | Ver 2.5. | `eventos.py:178-225` |
| `juntas` | Reset | Põe as juntas em `q_de_pe` (a pose de pé de `init_state`), com velocidade 0. Prende o resultado aos limites suaves das juntas. | Deslocamento de posição `position_range` = (0,0; 0,0) e de velocidade `velocity_range` = (0,0; 0,0), literais em `env_cfg.py:159-160`. | `env_cfg.py:159-160`; `mjlab/envs/mdp/events.py:279-313`; `cena.py:222-226` |
| `carga` | Reset | Escreve a massa de `v3_massa` e a inércia do cubo no modelo: `I = (2/3)·m·a²`, com `a` = `v3_meia`. | Determinístico (usa a massa já sorteada). | `eventos.py:443-461` |
| `atrito_pad` | Reset | Sorteia o atrito (coluna 0) das duas almofadas de palma, um sorteio por geom. | Sem a trava `portao0`: fixo em `Cena.pad_atrito[0]` = 1,0 (`knobs.py:81`). Com a trava: U(`mu_min_pad`; `DR.atrito_pad[1]` = 1,0) (`knobs.py:523`). Ver 5.7. | `eventos.py:464-475` |
| `atrito_laje` | Reset | Sorteia o atrito (coluna 0) da laje, um sorteio por env. | Sem a trava `portao0`: fixo em `Cena.prateleira_atrito[0]` = 1,0 (`knobs.py:91`). Com a trava: U(`DR.atrito_laje`) = U(0,3; 1,0) (`knobs.py:524`). | `eventos.py:478-489` |
| `vies_imu` | Reset | Sorteia o viés do giro da base e a inclinação fixa do IMU. | Com a trava `vies`: giro U(−`DR.vies_imu_giro`; +`DR.vies_imu_giro`) = ±0,05 rad/s por eixo (`knobs.py:521`). Inclinação em x e y U(±`DR.vies_imu_incl_graus`) = ±1,0° (`knobs.py:522`), z = 0. Sem a trava: giro 0 e inclinação = identidade. | `eventos.py:492-509` |
| `pd` | Reset, só no treino | Escala kp e kd de cada junta, um sorteio de kp e um de kd por junta e por env. | U(1 + (`DR.pd`[0] − 1)·g/5; 1 + (`DR.pd`[1] − 1)·g/5), com g = degrau global. `DR.pd` = (0,9; 1,1) (`knobs.py:535`). Com g = 0 a faixa é (1,0; 1,0). | `eventos.py:524-531`; `balanceador.py:368-369`, `balanceador.py:385`; `mjlab/envs/mdp/dr/actuator.py:76-90` |
| `forca_motor` | Reset, só no treino | Escala o limite de esforço, um sorteio por junta e env. Pernas e cintura usam `DR.forca_motor` = (0,9; 1,1) (`knobs.py:536`). Braços usam `DR.forca_motor_bracos` = (0,8; 1,1) (`knobs.py:537`). | Mesma interpolação por g. | `eventos.py:535-545`; `balanceador.py:386-387` |
| `solref` | Reset, só no treino | Põe `solref[0]` nas geoms dos pés, das palmas e dorsos, da laje e da caixa. Um valor por env. | U(`DR.solref_base` + (`DR.solref`[0] − `DR.solref_base`)·g/5; `DR.solref_base` + (`DR.solref`[1] − `DR.solref_base`)·g/5). `DR.solref_base` = 0,02 (`knobs.py:540`). `DR.solref` = (0,015; 0,03) (`knobs.py:539`). | `eventos.py:549-565`; `balanceador.py:388` |
| `filtros` | Reset, depois da `base` | Zera o filtro de elevação do mapa. Reinicia a odometria com a pose verdadeira. Reinicia o estimador da caixa e sorteia os vieses do episódio. | Ver 6.5. | `eventos.py:228-241`; `estimador.py:148-156`, `estimador.py:262-280` |

### 2.3 Terreno

Dentro das fatias que têm PEGAR, só a fatia B sorteia terreno. A fatia C nunca usa terreno (`eventos.py:70-74`). A fatia ANDAR também sorteia terreno, com regra própria, que este documento não descreve.

- Para uma fatia B, o episódio é de terreno com probabilidade `p` = fração da célula `("CARREGAR", "terreno")` entre as células abertas da tarefa CARREGAR. `p` vale 0 enquanto essa célula está fechada (`eventos.py:36-40`, `eventos.py:165-167`). A fração de uma célula é `piso + (1 − n·piso)·(D + ε)/Σ(D + ε)`, com `D = max(portao − s, 0)`, `piso` = `Balanceador.piso_sub` = 0,10 (`knobs.py:389`), `ε` = `Balanceador.eps` = 0,02 (`knobs.py:390`) (`balanceador.py:37-45`, `balanceador.py:295-299`).
- Num episódio de terreno B, a coluna é uniforme entre as colunas de `Terreno.linha_max_carregar` (`knobs.py:439`): escada, escada_inv, rampa, rampa_inv, escada_aleatoria, obstaculos, caixas. A linha é uniforme em 0..`min(nivel_terreno("CARREGAR"), teto da coluna)`. Os tetos são 9, 9, 9, 9, 6, 8, 7 (`eventos.py:43-74`).
- Fora do episódio de terreno, a coluna e a linha valem 0 (plano) (`eventos.py:74`).
- O episódio de terreno coloca a laje e a caixa na plataforma da coluna, com rumo `k·π/2` (k uniforme em {0, 1, 2, 3}) e deslocamento de `Terreno.laje_do_centro` − `Cena.prateleira_xy[0]` = 0,75 − 0,50 m ao longo do rumo (`knobs.py:444`, `knobs.py:94`; `eventos.py:112-130`, `eventos.py:355-367`). Sem terreno, o rumo e o deslocamento valem 0.

O sensor `altura_pelve`, `altura_joelhos` e `altura_caixa` medem a altura sobre o terreno local (`cena.py:392-397`).

### 2.4 Caixa e laje (`eventos.py:332-367`, `eventos.py:82-109`)

A caixa tem meia aresta `a` = `v3_meia`. Essa meia aresta é sorteada uma vez, no startup, entre `Cena.caixa_n_variantes` = 8 valores igualmente espaçados em `Cena.caixa_meia_faixa` = (0,07; 0,13) m (`knobs.py:78-79`; `eventos.py:416-440`). Depois, ela é fixa durante toda a vida do env.

No reset, para cada env B ou C:

| Grandeza | Fórmula | Distribuição | Referência |
|---|---|---|---|
| Topo da laje, sobre o chão local | Fatia B: `lo + (h_0 + extra − lo)·u₀`, com `lo = h_k + extra`. Fatia C: `lo = max(h_k, Niveis.topo_min_c) + extra`. | Uniforme. `h_k` e `h_0` vêm de `Niveis.pegar` (`knobs.py:417`): `h_k` = topo do nível atual, `h_0` = 0,55 m. `Niveis.topo_min_c` = 0,30 m (`knobs.py:419`). `extra` = `Niveis.subnivel_extra[subnível]` = (0,10; 0,0) m (`knobs.py:422`). | `eventos.py:82-109`, `eventos.py:344-346` |
| Massa da caixa | `1,0 + (teto − 1,0)·u₁`. Fatia B: teto = `m_k` do nível atual. Fatia C: teto = `m_j` do maior nível do PEGAR que já passou no portão (`nivel_passou`). Se nenhum passou, teto = 1,0. | Uniforme. Massa mínima = `Cena.caixa_massa_base` = 1,0 kg (`knobs.py:80`). `m_k` vem de `Niveis.pegar` (kg). | `eventos.py:99-109`; `balanceador.py:350-356` |
| Jitter em x da caixa | `dx = Cena.jitter_x_max[nível PEGAR]·u` | Uniforme em [0; `jitter_x_max`], só para o lado positivo. `Cena.jitter_x_max` = (0,20; 0,20; 0,20; 0,15; 0,08) m (`knobs.py:99`). | `eventos.py:358` |
| Jitter em y da caixa | `dy` | Uniforme em `Cena.caixa_jitter_y` = (−0,18; 0,18) m (`knobs.py:97`). | `eventos.py:359` |
| Posição da caixa no frame da origem | `desloc + R(rumo)·(Cena.caixa_xy[0] + dx; Cena.caixa_xy[1] + dy)` | `Cena.caixa_xy` = (0,32; 0,00) m (`knobs.py:96`). | `eventos.py:361` |
| Posição da laje | `desloc + R(rumo)·Cena.prateleira_xy`, rumo da laje = rumo da cena | `Cena.prateleira_xy` = (0,50; 0,00) m (`knobs.py:94`). Centro da laje a `Cena.prateleira_meia_z` = 0,02 m abaixo do topo (`knobs.py:90`). | `eventos.py:276-294`, `eventos.py:362-366` |
| Pose da caixa | Centro a `a` acima do topo. Quaternion identidade. Velocidade 0. | Determinístico. | `eventos.py:297-307` |

O nível 0 tem `h_k = h_0`. Logo o topo de uma fatia B no nível 0 vale `h_0 + extra` (um valor só): 0,55 + 0,10 = 0,65 m no subnível 0, e 0,55 m no subnível 1 (`knobs.py:417`, `knobs.py:422`). A massa no nível 0 vale 1,0 kg, porque `m_0` = 1 kg.

O sorteio usa dois uniformes `u₀` e `u₁` do `v3_gerador` (`eventos.py:104`). A massa sorteada é a massa simulada: o evento `carga` só a escreve no modelo (`eventos.py:443-461`).

Para fatia ANDAR, a laje e a caixa ficam estacionadas `Cena.estaciona_z` = 20,0 m acima da origem (`knobs.py:114`; `eventos.py:310-319`). Isso não vale para o PEGAR.

### 2.5 Base do robô (`eventos.py:178-225`)

A base é posta depois da caixa, em relação a ela.

| Grandeza | Fórmula | Distribuição | Referência |
|---|---|---|---|
| Distância atrás da caixa `d` | `d = ref.recuo(topo, a) + (2·u₀ − 1)·Alvos.faixa_recuo·jitter` | `u₀` uniforme. `Alvos.faixa_recuo` = 0,08 m (`knobs.py:227`). `jitter` = `Niveis.subnivel_jitter[subnível PEGAR]` = (0,0; 1,0) (`knobs.py:424`). No subnível 0 não há jitter. `ref.recuo` é a grade de referência (seção "Não determinado"). | `eventos.py:192-194` |
| Posição xy | `caixa_xy_mundo − R(rumo)·(d; 0)` | Determinístico. | `eventos.py:201-202` |
| Rumo (yaw) | `rumo_da_cena + yaw` | `yaw` uniforme em `Cena.reset_base_manipula` yaw = (−0,2; 0,2) rad (`knobs.py:101`). | `eventos.py:195-198` |
| Altura z | `z_origem + z_padrão + dz` | `dz` uniforme em `Cena.reset_base_manipula` z = (0,01; 0,05) m (`knobs.py:101`). `z_padrão` = `pelve_de_pe + 0,01` literal (`cena.py:223`). | `eventos.py:196-200`, `eventos.py:204` |
| Roll e pitch | 0 (o quaternion é o padrão vezes um giro só em z) | Determinístico. | `eventos.py:205` |
| Velocidade da base | A do estado padrão da entidade: zero, porque `init_state` não define velocidade e o padrão de `InitialStateCfg` é zero. | Determinístico. | `eventos.py:208`; `cena.py:222-226`; `mjlab/entity/entity.py:77-78` |

### 2.6 Estado do episódio escrito pela cadeia no reset (`comando.py:639-660`)

`CadeiaCmd._resample` roda no reset. Na ordem:

1. Registra o PEGAR e o BOTAR pendentes, com `falhou = ~time_outs` (seção 5.2).
2. Se havia uma visita de zona aberta, conta uma visita sem fecho para aquela etapa (`comando.py:646-650`).
3. Limpa `v3_pegou`, `v3_perdeu`, `v3_pega_fechada`, `v3_bota_fechada`, `v3_perdeu_agora`, `v3_avancou`, `v3_aviso` e `v3_fim` (`comando.py:651-653`).
4. Põe `v3_relogio_alvo` = 0, `v3_sigma` = 1,0 (literal), `v3_phi` = −1,0 (sentinela), `v3_h` = 1,0, `v3_phi_melhor` = 0,0 e `v3_v_chegada` = −1,0 (`comando.py:654-658`).
5. Abre o primeiro trecho da fatia (`comando.py:659-660`). O trecho ANDAR de espera sorteia sua duração, zera o relógio do trecho e o latch `v3_aviso`, repõe `v3_h` = 1,0 e `v3_phi_melhor` = 0,0, publica a tarefa ANDAR com `v3_freio` = nível do ANDAR + 1 e `v3_degrau` = degrau do ANDAR, e põe `v3_alvo_idx` = −1 e `v3_phi` = −1,0 (`comando.py:568-594`, `comando.py:508-519`).

Os buffers `v3_ponto_apoio`, `v3_caixa_pega_w` e `v3_alvo_w` não são limpos no reset. O PEGAR só lê `v3_caixa_pega_w` e `v3_alvo_w` depois que `_abre_p0` os reescreve (`comando.py:384-392`).

### 2.7 Abertura do P0

Depois da espera, a tarefa passa a PEGAR sem alvo. O P0 abre no primeiro passo em que `v3_estimador.tem_leitura()` vale True (`comando.py:370-392`). `tem_leitura` vale True quando há uma leitura entregue e a idade dela é menor que `Estimador.idade_max` = 2,0 s (`knobs.py:453`; `estimador.py:373-376`).

Na abertura do P0 (`comando.py:383-392`):

- `v3_alvo_idx` = P0. `v3_phi` = −1,0. `v3_h` = 1,0. `v3_phi_melhor` = 0,0. `v3_relogio_alvo` = 0,0.
- `v3_caixa_pega_w` e `v3_alvo_w` recebem a posição ESTIMADA da caixa no mundo: `do_frame_da_base(saida_b[:3], odometria)` (`estimador.py:197-201`).
- `v3_sigma` = `max(d, Alvos.sigma_min_etapa[P0])`, com `d` a distância média das palmas ao alvo P0 e `Alvos.sigma_min_etapa[P0]` = 0,05 m (`knobs.py:208`; `comando.py:359-362`).

Um env que cai antes de o P0 abrir não conta no registro do PEGAR: só o P0 aberto torna o PEGAR pendente (`comando.py:384`, `comando.py:521-530`).

### 2.8 Propriedades físicas da cena

| Peça | Propriedade | Valor | Referência |
|---|---|---|---|
| Caixa | Corpo livre, cubo, massa inicial do modelo | `Cena.caixa_massa_base` = 1,0 kg. O evento `carga` troca a massa por `v3_massa` a cada reset. | `knobs.py:80`; `cena.py:106-115`; `eventos.py:443-461` |
| Caixa | Atrito, `priority`, `condim` | `Cena.caixa_atrito` = (0,3; 0,02; 0,001), `Cena.caixa_priority` = 0, `Cena.caixa_condim` = 3. | `knobs.py:83`, `knobs.py:82`, `knobs.py:115` |
| Laje | Corpo cinemático (mocap), fino, meia aresta xy e z | `Cena.prateleira_meia_xy` = 0,30 m, `Cena.prateleira_meia_z` = 0,02 m. | `knobs.py:89-90`; `cena.py:118-128` |
| Laje | Atrito de base, `priority`, `condim` | `Cena.prateleira_atrito` = (1,0; 0,02; 0,001), `Cena.prioridade_laje` = 3, `Cena.prateleira_condim` = 3. O atrito de base é substituído pelo evento `atrito_laje` (seção 2.2). | `knobs.py:91`, `knobs.py:105`, `knobs.py:93` |
| Pad de palma | Atrito, `condim` | `Cena.pad_atrito` = (1,0; 0,02; 0,001), `condim` = 3. O primeiro valor é substituído pelo evento `atrito_pad`. | `knobs.py:81`; `cena.py:154` |
| Pad de dorso | Atrito, `condim` | `Cena.robo_atrito` = (0,3; 0,02; 0,001), `condim` = 1. | `knobs.py:84`; `cena.py:155` |
| Solver | Cone, `impratio`, `ccd_iterations`, `nconmax`, `njmax` | `Cena.cone` = "pyramidal", `Cena.impratio` = 1,0, `Cena.ccd_iterations` = 50, `Cena.nconmax` = 300, `Cena.njmax` = 800. | `knobs.py:107-111`; `env_cfg.py:98-102` |

Não determinado pelo código lido: o atrito efetivo do contato entre o pad de palma e a caixa, porque o valor depende da regra de mistura do MuJoCo para duas geoms de mesma `priority` (0), e o código lido só declara os valores acima.

---

## 3. Comando ou alvo

### 3.1 Comando de velocidade

O comando de velocidade `twist` vale zero em todo o trecho PEGAR.

- O `TwistV3._resample_command` zera `vel_command_b` do env e põe `v3_celula` = −1. Só os envs que satisfazem `(tarefa == ANDAR & fatia == ANDAR) | (tarefa == CARREGAR & fatia == B)` recebem velocidade sorteada. Um env em PEGAR não satisfaz nenhuma das duas (`comando.py:897-938`).
- O `twist` reamostra a cada `time_left`, sorteado em U(`Comando.reamostra_s`) = U(3,0; 8,0) s (`knobs.py:137`; `env_cfg.py:122-123`; `mjlab/managers/command_manager.py:117-123`). Também reamostra na abertura de cada trecho (`comando.py:519`) e no reset (`comando.py:893`). Em PEGAR a reamostra mantém o comando em zero, mantém `v3_celula` = −1 e soma 1 no contador de segmentos.
- Com `v3_celula` = −1, nenhum segmento de velocidade se registra na célula de um balanceador (`comando.py:792`).
- O relógio de marcha não corre sem comando: `on = ‖comando‖ > Comando.limiar_marcha` = 0,1 (`knobs.py:141`; `comando.py:863`). Na PEGAR `on` é falso. `_fase_t` só soma com `on` (`comando.py:872`), e `v3_fase_marcha` = `_fase_t / Comando.periodo_marcha` (resto 1), com `periodo_marcha` = 0,8 s (`knobs.py:140`; `comando.py:873`). Num episódio B ou C o comando é zero desde o reset. Logo `v3_fase_marcha` fica em 0 em todo o PEGAR.

### 3.2 Alvo

O robô recebe o alvo pela observação `caixa` (seção 9). O alvo tem estas partes.

| Parte | Como é produzida | Quando muda | Referência |
|---|---|---|---|
| Índice do alvo `v3_alvo_idx` | P0 na abertura. Sobe um a cada fecho de etapa. Fica em P3 depois do fecho do P3. | A cada fecho de etapa (seção 4). | `comando.py:383`, `comando.py:492-501`; `alvos.py:195-210` |
| Posição do alvo `v3_alvo_w` | Posição estimada da caixa no instante em que o P0 abre. | Só na abertura do P0. As etapas P1 a P3 não a reescrevem. | `comando.py:390-391` |
| Palmas-alvo no frame da caixa | `palmas_alvo_caixa(índice, a, Alvos)` (seção 4.2). | A cada etapa. | `alvos.py:138-154` |
| Posição-alvo da caixa no P3 | `caixa_alvo_w(P3, ...)`: base + R(yaw da base)·`caixa_transporte_b(a)[:2]` no xy, e `chão local + Alvos.z_transporte` no z. | A cada passo, porque o xy segue a base. | `alvos.py:163-179`; `recompensas.py:487-493` |

`Alvos.z_transporte` = 0,90 m (`knobs.py:202`). O chão local é a altura z da pelve menos a leitura do sensor `altura_pelve` (`recompensas.py:489-491`).

---

## 4. Etapas dentro do episódio e critério de avanço

O PEGAR tem quatro etapas, P0, P1, P2 e P3. A função `alvos.avanca` avança a etapa quando `ok_agora` vale por um tempo mínimo (`alvos.py:195-210`):

- `rel = relogio + dt` se `ok_agora`, senão `rel = 0`.
- `avancou = ok_agora & (indice >= 0) & (rel >= tempo − 1e-9)` (o `1e-9` é literal).
- O índice avança 1. Depois do B3 vira −1. O relógio volta a 0 no avanço.

O `ok_agora` do PEGAR é `ok_agora_pegar(...) & (tarefa == PEGAR) & ~v3_pega_fechada` (`comando.py:471-474`). O tempo de permanência vem de `self.tempos`, refeito a cada iteração (`comando.py:428-443`):

`tempos = [0,0; 0,0; tempo_p2; tempo_p3; tempo_b1; tempo_b2; tempo_b3]` (os dois zeros são literais em `comando.py:442`).

⚠ Divergência: o texto de `alvos.py:200` diz que o alvo sem tempo é o P2, com tempo 0. O código dá tempo 0 ao P0 e ao P1, e dá ao P2 o `tempo_p2` efetivo (`comando.py:442`).

### 4.1 Valores efetivos da zona

A cada iteração a cadeia refaz o `Alvos` efetivo. Para cada linha `(etapa, campo, largo)` de `Alvos.zona` (`knobs.py:211-220`), o valor efetivo é `final + z[etapa]·(largo − final)` (`comando.py:433-441`; `balanceador.py:48-51`). O fecho (`ok_agora_pegar`), o tempo de permanência e a qualidade da caixa leem o valor efetivo. O Φ, a distância `d`, os portões e o ponto-alvo leem o valor final (`comando.py:429-432`).

| Etapa | Campo | Final (`knobs.py`) | Largo (`knobs.py`) |
|---|---|---|---|
| P0 | `tol_palma` | `Alvos.tol_palma` = 0,03 m (`knobs.py:183`) | 0,05 m (`knobs.py:212`) |
| P1 | `tol_palma` | 0,03 m | 0,05 m (`knobs.py:212`) |
| P2 | `tempo_p2` | `Alvos.tempo_p2` = 0,3 s (`knobs.py:182`) | 0,2 s (`knobs.py:213`) |
| P3 | `subida_min_p3` | `Alvos.subida_min_p3` = 0,03 m (`knobs.py:186`) | 0,01 m (`knobs.py:214`) |
| P3 | `tol_transporte_p3` | `Alvos.tol_transporte_p3` = 0,05 m (`knobs.py:187`) | 0,20 m (`knobs.py:214`) |
| P3 | `tol_pelve_p3` | `Alvos.tol_pelve_p3` = 0,03 m (`knobs.py:188`) | 0,10 m (`knobs.py:214`) |
| P3 | `tempo_p3` | `Alvos.tempo_p3` = 0,5 s (`knobs.py:189`) | 0,2 s (`knobs.py:215`) |

P0 e P1 repartem o campo `tol_palma`, cada uma com o `z` dela (`comando.py:437-438`).

### 4.2 Critério de cada etapa

`ok_agora_pegar` (`comando.py:41-56`) devolve:

`ok = ((idx == P0 | idx == P1) & p01) | ((idx == P2) & p2) | ((idx == P3) & p3)` e depois `& (invariantes == 1,0)`.

Em todas as etapas, `invariantes = desloc · incl · maos`, três portões de `portoes_do_alvo` (`comando.py:463-466`). Cada portão é uma `rampa`: `clip(1 − (x − x_lim)/Δ, 0, 1)` (`potencial.py:23-29`). O invariante vale exatamente 1,0 só quando todos os três valem 1,0, isto é, quando `x ≤ x_lim` em cada rampa que vale na etapa.

| Etapa | Palmas-alvo no frame da caixa (`alvos.palmas_alvo_caixa`) | Condição de `ok_agora` (valores efetivos) | Permanência | Referência |
|---|---|---|---|---|
| P0 | Palma 0 em +y, palma 1 em −y. `x = −(a + Alvos.p0_frente)`. `y = ±(a + Alvos.palma_face + Alvos.folga_p1)`. `z = 0`. | Distância 3D de cada palma ao alvo ≤ `tol_palma` efetiva de P0. Invariante = 1,0 (só `desloc`). | 0,0 s (literal) | `alvos.py:138-154`; `comando.py:41-56`, `comando.py:468` |
| P1 | `x = 0`. `y = ±(a + Alvos.palma_face + Alvos.folga_p1)`. | Distância de cada palma ≤ `tol_palma` efetiva de P1. Invariante = 1,0 (só `desloc`). | 0,0 s (literal) | idem |
| P2 | `x = 0`. `y = ±(a + Alvos.palma_face)`. Alvo só para a distância do Φ e da conquista. | As duas palmas em contato com a caixa (`contatos.all(-1)`). Invariante = 1,0 (`desloc` e `incl`). | `tempo_p2` efetivo | `comando.py:52`, `comando.py:55` |
| P3 | Nenhuma palma-alvo. | `subida ≥ subida_min_p3`, `d_transporte ≤ tol_transporte_p3`, `d_pelve ≤ tol_pelve_p3`, caixa sem contato com a laje (`~apoio`). Invariante = 1,0 (`incl` e `maos`). | `tempo_p3` efetivo | `comando.py:54-55`, `comando.py:467-474` |

Constantes: `Alvos.p0_frente` = 0,05 m (`knobs.py:175`). `Alvos.palma_face` = 0,045 m (`knobs.py:178`). `Alvos.folga_p1` = 0,02 m (`knobs.py:179`).

Definição das medidas, todas em `comando.py:317-346` e `comando.py:467-473`:

- `distância da palma` = `‖palmas_c − palma-alvo‖` por palma. `palmas_c` é a posição de cada sítio de palma no frame da caixa.
- Frame da caixa: `z` = normal de face mais vertical, apontando para cima. `x` = normal horizontal de face mais alinhada ao rumo da base. `y` = `z × x` (`alvos.py:109-128`).
- `contatos` = os sensores `palma_E` e `palma_D` acusam contato com a geom da caixa (`cena.py:349-354`).
- `subida` = `h_caixa − a − v3_topo`. É o fundo da caixa sobre o topo da laje (`h_caixa` vem do sensor `altura_caixa`).
- `d_transporte` = `‖posição da caixa − caixa_alvo_w(P3)‖`.
- `d_pelve` = `|altura_pelve − ref.pelve_com_caixa(a)|`.
- `apoio` = a caixa toca a laje: `apoio_caixa.found.amax(-1) > 0` (`comando.py:333`).
- `desl` = `‖xy da caixa − v3_caixa_pega_w xy‖`. A posição de referência é a estimada na abertura do P0 (`comando.py:455`).
- `incl` = inclinação em graus da face mais vertical da caixa (`alvos.py:131-135`).
- `maos` = número de palmas em contato dividido por 2 (`comando.py:464`).

Os portões do invariante (`recompensas.py:247-271`, chamados em `comando.py:463-466`):

| Portão | Vale em | Fórmula | Constantes |
|---|---|---|---|
| `desloc` | P0, P1, P2 | `rampa(desl, Alvos.desloc_p0p2_frac·a, Alvos.delta_p0p2_frac·a)` | `Alvos.desloc_p0p2_frac` = 1,0 (`knobs.py:222`). `Alvos.delta_p0p2_frac` = 0,5 (`knobs.py:223`). |
| `incl` | P2, P3 | `rampa(incl, incl_lim, Alvos.delta_incl_graus)` | `Alvos.delta_incl_graus` = 15,0° (`knobs.py:205`). `incl_lim` = `Alvos.incl_inicio_graus` + (`Alvos.incl_fim_graus` − `Alvos.incl_inicio_graus`)·degrau/5, com `incl_inicio_graus` = 30,0° e `incl_fim_graus` = 15,0° (`knobs.py:203-204`). O degrau é o `v3_degrau` do env (`recompensas.py:447-464`). |
| `maos` | P3 | valor passado = `contatos/2` | Vale 1,0 só com as duas palmas em contato. |

Os demais portões (`tronco`, `recuo`, `com`) entram com valor que dá 1,0 no invariante (`comando.py:465`).

Fora de seus alvos, cada portão vale 1,0 (`recompensas.py:262-270`).

### 4.3 Efeitos de um fecho

Depois de `avanca` (`comando.py:482-503`):

- Um avanço em P2 põe `v3_pegou` = True (`comando.py:497`).
- Um avanço em P3 ("fecha_p3") põe `v3_pega_fechada` = True e mantém o índice em P3 (`comando.py:492-496`).
- Um avanço em P0, P1 ou P2 reescreve `v3_sigma` com `max(d, Alvos.sigma_min_etapa[novo])`, põe `v3_h` = 1,0 e `v3_phi_melhor` = 0,0 (`comando.py:502-504`).
- `Alvos.sigma_min_etapa` = (0,05; 0,05; 0,08; 0,20; 0,20; 0,05; 0,0) m (`knobs.py:208`). Para o PEGAR valem os quatro primeiros.

O aviso do trecho é o fecho do P3. `v3_aviso |= (v3_pega_fechada & trecho_por_detector)` (`comando.py:604-607`). O latch fica até o fim do trecho. Depois do aviso, o relógio do trecho corre até o atraso sorteado U(0,0; 1,0) s (`Cadeia.atraso_pegar_s`, `knobs.py:158`), e o trecho PEGAR acaba (`comando.py:201-216`).

### 4.4 Visitas e fechos (zona)

Uma visita é o tempo em que uma etapa fica aberta. A cadeia conta visitas e fechos por etapa para a zona (`comando.py:444-452`, `comando.py:483-491`):

- Uma visita abre quando `etapa = v3_alvo_idx` (com `v3_alvo_idx ≥ 0`, sem `v3_perdeu` e fora do estado "P3 com pega fechada") difere de `vis_e`. A etapa vira `vis_e` e `t_vis` = 0.
- `t_vis` soma `dt` a cada passo.
- Fecho: a visita encerra com `av & (t_vis ≥ 0)`. Conta 1 visita e 1 fecho.
- Sem fecho por tempo: `~av & (t_vis ≥ Balanceador.zona_t_visita)` com `zona_t_visita` = 8,0 s (`knobs.py:407`). Conta 1 visita, sem fecho. A etapa continua aberta. A visita não reabre até a etapa mudar.
- Sem fecho por troca: a etapa muda por fora (perda da caixa ou troca de tarefa) com `t_vis ≥ 0`. Conta 1 visita, sem fecho.
- Sem fecho por reset: uma visita aberta no reset soma 1 visita à etapa (`comando.py:646-650`).
- Os contadores vão para `cur.vis_acum` e `cur.fech_acum` (`comando.py:488-490`).

---

## 5. Progressão

### 5.1 Quem mede a nota

| Grandeza | Quem mede | Referência |
|---|---|---|
| Nota `s` da célula PEGAR | `CadeiaCmd._registra`, um sucesso 0 ou 1 por episódio com P0 aberto. | `comando.py:521-540` |
| Taxa de fecho por visita (zona) | `CadeiaCmd._alvos`, por contagem de visitas e fechos. | `comando.py:444-491` |
| Todas as subidas | `Curriculo.fecha_iteracao`, uma vez por iteração. | `balanceador.py:153-195` |

`_registra` (`comando.py:521-540`) conta o PEGAR em dois momentos:

- Na abertura do BOTAR (fatia C), com `falhou` = False (`comando.py:589`).
- No reset do episódio, com `falhou = ~time_outs` (`comando.py:643-645`). Na fatia B é o único momento.

O sucesso do episódio é (`comando.py:537`; `comando.py:68-70`):

`sucesso = v3_pega_fechada & ~falhou & ~v3_perdeu & (max(z[P0..P3]) == 0,0)`

`falhou` é verdadeiro quando o episódio acabou por uma terminação que não é `time_out`. Um sucesso exige que `z` valha exatamente 0,0 nas quatro etapas. Um fecho com `z` maior que 0 conta como falha para o nível, o subnível, o refino e as travas de DR.

O registro soma `n` (episódios) e `suc` (sucessos) em `_acum_n` e `_acum_suc` da célula (`balanceador.py:148-151`).

### 5.2 Fecho da iteração: ordem

`fecha_iteracao` roda na ordem abaixo (`balanceador.py:153-195`):

1. `it += 1`. Atualiza a nota `s` de cada célula (5.3).
2. Atualiza a zona de cada etapa com visitas suficientes (5.6).
3. Conta o refino de cada tarefa (5.5).
4. Abre células (`_abre`, 5.4).
5. Marca a iteração de abertura das fatias B e C, na primeira vez que têm célula aberta.
6. Sobe a rampa das fatias B e C.
7. Sobe nível, subnível ou linha de terreno (`_sobe`, 5.4).
8. Liga as travas do DR (5.7).

### 5.3 Nota da célula

Para cada célula com `n > 0` na iteração (`balanceador.py:163-168`):

- `x = soma/n`.
- Se `n_medidas == 0`: `s = x`. Senão: `s = s + alfa·(x − s)`, com `alfa` = `Balanceador.alfa` = 0,05 (`knobs.py:394`).
- `n_medidas += 1`. Os acumuladores zeram.
- Com `n = 0`, `s` não muda. `s` começa em 0,0 (`balanceador.py:79`).

O portão da célula PEGAR é `Balanceador.portao` = 0,75 (`knobs.py:391`; `balanceador.py:97-98`). A célula passa quando `s ≥ 0,75` (`balanceador.py:122-123`).

### 5.4 Nível e subnível

A tabela de níveis do PEGAR é `Niveis.pegar` (`knobs.py:417`).

| Nível | Topo `h_k` (m) | Massa `m_k` (kg) | `Cena.jitter_x_max` (m) (`knobs.py:99`) | Expoente do freio (nível + 1) |
|---|---|---|---|---|
| 0 | 0,55 | 1 | 0,20 | 1 |
| 1 | 0,45 | 2 | 0,20 | 2 |
| 2 | 0,30 | 3 | 0,20 | 3 |
| 3 | 0,15 | 4 | 0,15 | 4 |
| 4 | 0,04 | 5 | 0,08 | 5 |

O nível máximo do PEGAR é `len(Niveis.pegar) − 1` = 4 (`balanceador.py:74`). O expoente do freio é `niveis_abertos` = nível + 1, escrito em `v3_freio` na abertura do trecho (`balanceador.py:335-336`; `comando.py:515`).

Condição de subida (`balanceador.py:250-278`), para a tarefa PEGAR:

1. Abaixo do nível máximo: `ok = célula aberta & s ≥ portão`.
2. `_seguidas` soma 1 se `ok`, senão volta a 0.
3. Quando `_seguidas ≥ Balanceador.iters_nivel` = 50 (`knobs.py:395`), `_seguidas` volta a 0 e sobe um degrau de progresso:
   - Se `subnível < _sub_max` (= `len(Niveis.subnivel_extra) − 1` = 1): `subnível += 1`.
   - Senão: `nível += 1`.
4. Na subida, as células da tarefa zeram `n_medidas`. A nota `s` antiga fica até a primeira medida nova.
5. No nível máximo, o PEGAR não sobe mais: `ok` vale False, porque o PEGAR não é uma tarefa de cadeia (`balanceador.py:261-264`, `balanceador.py:31`).

Subníveis do nível 0 (`knobs.py:422-424`):

| Subnível | `subnivel_extra` (m) | Topo no nível 0 | `subnivel_longe` (BOTAR) | `subnivel_jitter` da base |
|---|---|---|---|---|
| 0 | 0,10 | 0,55 + 0,10 = 0,65 m | 0,10 m | 0,0 |
| 1 | 0,0 | 0,55 m | 0,20 m | 1,0 |

`subnivel_longe` só vale para a laje de destino do BOTAR. Está na tabela porque o subnível é compartilhado.

### 5.5 Refino (degrau)

`_conta_refino` roda por tarefa (`balanceador.py:234-248`):

- Se `degrau ≥ Balanceador.degraus` = 5, não faz nada.
- `ultimo` = `nível == nível máximo`. Para o PEGAR não há linha de terreno a exigir (`balanceador.py:239-240`).
- Se `ultimo` e a média das notas das células abertas ≥ a média dos portões, `_cont += 1`. Senão `_cont = 0`.
- Se `_cont ≥ Balanceador.iters_refino` = 100 (`knobs.py:400`), `degrau += 1` e `_cont = 0`. O degrau nunca desce.

O que o degrau do PEGAR muda, quando o PEGAR é a tarefa publicada (lido de `v3_degrau`, escrito na abertura do trecho, `comando.py:516`):

| Efeito | Fórmula | Referência |
|---|---|---|
| Inclinação limite dos portões `incl` | 30,0° → 15,0° (`knobs.py:203-204`) | `recompensas.py:462-464` |
| Força mínima do `contato_laje` | `Recompensa.laje_joelho_N` = (50,0; 10,0) N (`knobs.py:302`) | `recompensas.py:887-896` |
| Fator da `sacudida` e do `aperto` | `d/5` (de 0 a 1) | `recompensas.py:944`, `recompensas.py:955` |
| Liga a terminação `batente` | A partir de `Refino.degrau_batente` = 1 (`knobs.py:547`) | `terminacoes.py:164`, `terminacoes.py:180-181` |
| Observação do crítico `degrau` | `v3_degrau/5` | `observacoes.py:338-340` |

O degrau global (o menor entre as quatro tarefas) controla a DR do refino, os erros do estimador e a deriva da odometria (seção 6).

### 5.6 Zona adaptativa

A zona de cada etapa é `z` entre 0 e 1, inicial 1,0 (`balanceador.py:105`). Em cada fecho de iteração (`balanceador.py:169-176`):

1. Para cada etapa `e` com `vis_acum[e] ≥ Balanceador.zona_n_min` = 200 (`knobs.py:406`):
   - `taxa[e] = fech_acum[e] / vis_acum[e]`.
   - `z[e] = clip(z[e] − Balanceador.zona_ganho·(taxa[e] − Balanceador.zona_t_alvo), 0, 1)`.
   - `vis_acum[e]` e `fech_acum[e]` voltam a 0.
   - `Balanceador.zona_ganho` = 0,05 (`knobs.py:405`). `Balanceador.zona_t_alvo` = 0,6 (`knobs.py:404`).
2. Para as etapas P1, P2 e P3: `z[e] = max(z[e], z[e−1] − Balanceador.zona_folga_cadeia)`, com `zona_folga_cadeia` = 0,2 (`knobs.py:408`). Uma etapa não fica mais apertada que a anterior por mais que isso.

Uma taxa acima de 0,6 desce `z` (aperta o critério). Uma taxa abaixo de 0,6 sobe `z` (afrouxa).

### 5.7 Linha de terreno, restrição, travas do DR e rampa

- Linha de terreno: o PEGAR não tem linha própria. `_nivel_ter["PEGAR"]` fica em 0 (`balanceador.py:84`). Um env B pode ser sorteado em terreno quando a célula CARREGAR `terreno` está aberta (seção 2.3).
- Restrição: `min(nível[t]/nível_máximo[t])` sobre ANDAR, PEGAR, CARREGAR e BOTAR (`balanceador.py:345-348`). Os níveis máximos são: ANDAR = `round((Comando.frente_tetos[-1] − Comando.andar_inicio)/Comando.andar_passo)` = round((1,0 − 0,3)/0,1) = 7 (`knobs.py:122`, `knobs.py:129`, `knobs.py:134`); CARREGAR = `len(Comando.frente_tetos) − 1` = 4; PEGAR = 4; BOTAR = 4 (`balanceador.py:72-74`). O PEGAR influencia a restrição por seu nível.
- Travas do DR (só ligam, nunca desligam; `balanceador.py:197-202`):
  - `portao0` liga quando `nivel_passou("PEGAR") ≥ 0`. A função devolve o nível do PEGAR quando `s ≥ 0,75`, `subnível == _sub_max` e (`nível > 0` ou `n_medidas > 0`), e devolve nível − 1 senão. Logo a trava liga com o nível do PEGAR ≥ 1, ou com o nível 0, `s ≥ 0,75`, subnível 1 e `n_medidas > 0` (`balanceador.py:350-356`).
  - `vies` liga quando a célula ANDAR `andar` passa no portão `Balanceador.portao_andar` = 0,75 (`knobs.py:392`).
  - `subida` liga quando `nível PEGAR + nível CARREGAR > 0`.
- `mu_min_pad` = `max(DR.mu_pad_min, DR.atrito_pad[0] − DR.passo_mu_pad·(nível PEGAR + nível CARREGAR))` quando `portao0` está ligada. `DR.mu_pad_min` = 0,5. `DR.atrito_pad[0]` = 0,6. `DR.passo_mu_pad` = 0,025 (`knobs.py:523-526`; `balanceador.py:371-373`).
- Atraso máximo: `DR.atraso_refino[1]` = 3 se `degrau_global ≥ DR.degrau_atraso_refino` = 3 (`knobs.py:513-514`). Senão `ceil(DR.atraso_portao[1]·restrição)`, com `DR.atraso_portao[1]` = 2 (`knobs.py:512`; `balanceador.py:381-382`).

### 5.8 Resumo do que muda no env a cada subida

| Subida | Condição | O que muda no env |
|---|---|---|
| Fatia B abre | Nível ANDAR ≥ 5 | O PEGAR passa a ser sorteado (via fatia B). Teto da fatia B começa em 0,05. |
| Fatia C abre | `s_PEGAR ≥ 0,75` | A fatia C passa a ser sorteada. O BOTAR entra na cadeia. |
| Rampa da fatia | A cada 12 fechos desde a abertura | O teto da fatia sobe 0,02. |
| Subnível 0 → 1 | `s ≥ 0,75` por 50 fechos seguidos | Topo da laje de 0,65 m para 0,55 m. Jitter da base de 0,0 para 1,0 da faixa. |
| Nível n → n+1 | Idem, no subnível 1 | Topo, massa, jitter em x, expoente do freio e `mu_min_pad` (tabela 5.4). |
| Degrau d → d+1 | Nível 4 e `s ≥ 0,75` por 100 fechos seguidos | Itens da tabela 5.5. |
| Zona | Taxa de fecho por 200 visitas | Critério de fecho da etapa (4.1). |

---

## 6. Perturbações e randomizações

### 6.1 No startup (uma vez, na criação do env)

Eventos de modo `startup` (`env_cfg.py:185-198`). O sorteio é por env, fixo durante toda a vida do env.

| Evento | O que sorteia | Faixa | Referência |
|---|---|---|---|
| `atrito_pe` | Atrito dos geoms dos pés. Um valor por env, compartilhado pelas 14 geoms. Operação `abs`. | U(`DR.atrito_pe`) = U(0,3; 1,0) (`knobs.py:508`). | `env_cfg.py:185-187`; `mjlab/envs/mdp/dr/_core.py:107-148` |
| `massa_tronco` | Soma à massa do `torso_link`. | U(`DR.massa_tronco`) = U(−1,0; 3,0) kg (`knobs.py:509`). | `env_cfg.py:188-190` |
| `vies_encoder` | Viés de encoder por junta e env. | U(−`DR.vies_encoder`; +`DR.vies_encoder`) = ±0,015 rad (`knobs.py:510`). O viés entra só na observação `juntas` do ator (`biased=True`); o crítico lê `juntas` sem viés (`observacoes.py:394`, `observacoes.py:404`). | `env_cfg.py:191-192`; `mjlab/envs/mdp/dr/joint.py:209-239` |
| `tamanho_caixa` | A meia aresta da caixa (seção 2.4). | Escolha uniforme entre 8 valores de (0,07; 0,13) m. | `eventos.py:416-440` |
| `atraso` | Põe o teto do atraso nos buffers do atuador e da observação. | `DR.atraso_inicio[1]` = 0 passos (`knobs.py:511`). | `env_cfg.py:197-198`; `eventos.py:244-258` |

Os eventos `atrito_pe` e `massa_tronco` sorteiam pelo gerador global do torch. `tamanho_caixa` sorteia pelo `v3_gerador` (`eventos.py:430`).

### 6.2 No reset do episódio

Ver a tabela 2.2: `atrito_pad`, `atrito_laje`, `vies_imu`, `pd`, `forca_motor`, `solref`. Todas valem na PEGAR, nas condições de cada linha.

### 6.3 Empurrão (intervalo)

| Item | Valor |
|---|---|
| Quando | Evento de modo `interval`, por env (`env_cfg.py:200-205`). Não existe no modo `play`. |
| Intervalo | `DR.periodo_empurrao_s` = (3,0; 7,0) s (`knobs.py:520`). O tempo restante inicial de cada env é U(3,0; 7,0) s. Depois de cada disparo o env sorteia outro (`mjlab/managers/event_manager.py:278-286`, `event_manager.py:362-376`). O reset do episódio não reinicia esse relógio, porque o termo é uma função (`knobs.py:518-519`; `event_manager.py:225-233`). |
| O que faz | Soma à velocidade linear da base, no mundo, um valor U(−`DR.empurrao`; +`DR.empurrao`) em x e em y. `DR.empurrao` = 0,5 m/s (`knobs.py:517`). Os outros eixos não recebem soma. |
| Condição | Vale em todas as tarefas, inclusive PEGAR. |
| Efeito colateral | Grava `v3_t_empurrao = episode_length_buf·step_dt`. O ANDAR usa esse instante na âncora (seção do ANDAR). |

Referências: `eventos.py:261-269`; `mjlab/envs/mdp/events.py:316-336`.

### 6.4 Força na caixa e ruído de torque

- Força na caixa (`forca_caixa`, modo `step`): não atua no PEGAR. A função tem dois filtros. Primeiro, só dispara com a trava `dr()["forca_caixa"]` ligada (a trava `subida`, 5.7). Segundo, depois que o `mjlab` escreve o impulso, a função zera o torque e a força de todo env em que `~(tarefa == CARREGAR & v3_pegou & ~v3_perdeu)`. Um env em PEGAR cai sempre nesse conjunto (`eventos.py:584-602`; `env_cfg.py:207-209`).
- Ruído de torque (`rfi`, modo `step`): atua no PEGAR. A cada passo da política, depois dos comandos (passo 9 da seção 1.5), o evento escreve em `qfrc_applied`, para cada junta e env, o valor `U(−1; 1)·s·TAU_MAX`. A escrita substitui o valor anterior, e vale nos 4 subpassos de física do passo seguinte. `s` = `DR.rfi` interpolado: `s = DR.rfi[0] + (DR.rfi[1] − DR.rfi[0])·g/5` se a trava `subida` está ligada, senão `s = 0`. `DR.rfi` = (0,05; 0,1) (`knobs.py:534`). `TAU_MAX` é o torque máximo por junta de `Recompensa.tau_max` (`knobs.py:286-292`, `knobs.py:572`) (`eventos.py:605-615`; `balanceador.py:384`).

### 6.5 Estimador da caixa e odometria

O ator vê a caixa pelo estimador simulado, não pela posição verdadeira. O estimador vale no PEGAR. As faixas seguem o degrau global g: valor = `inicio + (fim − inicio)·g/5` (`observacoes.py:76-90`; `balanceador.py:48-51`).

| Grandeza | Par (início; refino) | Significado | Referência |
|---|---|---|---|
| `Estimador.sigma_pos` | (0,005; 0,02) | Ruído gaussiano da posição por eixo: σ = `sigma_pos` (m) que vai de 0,005 a 0, mais `sigma_pos_rel`·distância câmera-caixa, que vai de 0 a 0,02. | `knobs.py:471`; `observacoes.py:84-85`; `estimador.py:294-298` |
| `Estimador.vies_pos` | (0,0; 0,02) m | Viés U(−1; 1) por eixo, sorteado por episódio, vezes a magnitude. | `knobs.py:460`; `estimador.py:241-244`, `estimador.py:326-327` |
| `Estimador.vies_meia` | (0,005; 0,02) m | Viés da meia aresta, U(−1; 1) por episódio vezes a magnitude. | `knobs.py:459`; `estimador.py:309` |
| `Estimador.ori_graus` | ((1,5; 2,5); (5,3; 15,3))° | (mediana; p90) do ângulo do viés de orientação do episódio, lognormal. Eixo uniforme na esfera, sorteado por episódio. | `knobs.py:465`; `estimador.py:299-306` |
| `Estimador.ori_leitura_graus` | (0,0; 1,0)° | Giro aleatório por leitura em cada eixo. | `knobs.py:466`; `estimador.py:307` |
| `Estimador.perda` | (0,005; 0,022) | Probabilidade de uma leitura se perder. | `knobs.py:472`; `estimador.py:331` |
| `Estimador.rajada_taxa` | (0,0; 0,035) 1/s | Taxa de abertura de rajadas de perda. A duração é U(`Estimador.rajada_s`) = U(0,3; 1,0) s (`knobs.py:468`). | `knobs.py:469`; `estimador.py:311-317` |

Parâmetros fixos do estimador: taxa de leitura `Estimador.taxa_hz` = 30,0 Hz, latência `Estimador.latencia_s` = 0,033 s, idade máxima `Estimador.idade_max` = 2,0 s (`knobs.py:451-453`). A visibilidade exige o centro da caixa no campo de visão da câmera, com `fov_h_graus` = 69,0° e `fov_v_graus` = 42,0° (`knobs.py:455-456`), no alcance `Estimador.alcance` = (0,28; 3,0) m (`knobs.py:457`), e pelo menos `Estimador.pontos_visiveis_min` = 3 dos 9 pontos livres de oclusão pelos braços (`knobs.py:470`; `estimador.py:111-126`).

Os sorteios por episódio (viés de meia, de posição e de orientação, e o fim da rajada de perda) acontecem em `EstimadorCaixa.reinicia`, no reset (`estimador.py:262-280`).

Como o estimador produz a leitura (`estimador.py:204-209`, `estimador.py:282-376`):

- O estimador dá um passo por passo da política, na hora de calcular as observações (passo 11 da seção 1.5), com o tempo `common_step_counter·step_dt` (`observacoes.py:140-151`, `observacoes.py:191-192`).
- A amostra cai numa grade de tempo de período 1/`Estimador.taxa_hz` (`knobs.py:451`). A posição e a rotação da amostra são interpoladas entre o passo anterior e o atual. A amostra entra na fila se a caixa é visível, se um sorteio U(0; 1) é maior ou igual a `perda`, e se o instante da grade não está dentro de uma rajada de perda (`estimador.py:319-337`).
- A fila entrega uma amostra quando o instante dela é menor ou igual a t − `Estimador.latencia_s` (`knobs.py:452`). O registro guarda a entregue mais recente (`estimador.py:343-352`).
- A idade da leitura é t menos o instante da amostra registrada, cortada em `Estimador.idade_max` (`estimador.py:379-380`).
- Sem leitura nova, a saída é a última leitura. Se a última leitura estava presa às palmas, a saída anda com o ponto médio das palmas e gira com o eixo delas. A leitura está presa às palmas quando ‖caixa − meio das palmas‖ < meia aresta lida + `Estimador.presa_margem` = 0,05 m (`knobs.py:454`; `estimador.py:359-371`).
- A saída vai ao frame da base estimada pela odometria atual (`estimador.py:381-385`).

A odometria deriva com o degrau global. Cada passo soma ruído com variância `deriva_odo²·|Δp| + deriva_pos_giro²·|Δψ|` (posição) e `deriva_rumo²·|Δp| + deriva_rumo_giro²·|Δψ|` (rumo) (`estimador.py:166-174`). Os pares (início; refino) são `Estimador.deriva_odo` = (0,002; 0,0075), `Estimador.deriva_rumo` = (0,002; 0,005), `Estimador.deriva_pos_giro` = (0,004; 0,008), `Estimador.deriva_rumo_giro` = (0,0035; 0,014) (`knobs.py:473-476`; `comando.py:247-250`).

### 6.6 Ruído e atraso de observação

- Ruído do ator: uniforme aditivo, antes da escala, só no grupo `actor` (com `enable_corruption` = True no treino). `DR.ruido_obs` = giro_base ±0,2, gravidade ±0,05, juntas ±0,01, vel_juntas ±1,5 (`knobs.py:516`; `observacoes.py:373-381`, `observacoes.py:420`). O crítico não tem ruído (`observacoes.py:404`).
- Atraso do ator: nos termos `giro_base`, `gravidade`, `juntas`, `vel_juntas` e `mapa` (`observacoes.py:55`). O lag é inteiro, uniforme em [0; teto], sorteado por env no primeiro passo depois do reset e fixo no episódio (`observacoes.py:382-386`; `mjlab/utils/buffers/delay_buffer.py:194-216`, `delay_buffer.py:247-281`). O teto é o atraso máximo do currículo (5.7); no startup vale `DR.atraso_inicio[1]` = 0.
- Atraso de comando no atuador: o mesmo mecanismo, com teto `atraso·Contrato.decimacao` passos de física (`eventos.py:253-255`; `cena.py:188-195`).
- Viés do IMU no ator: o termo `giro_base` soma o sensor de giro e `v3_vies_giro`. O termo `gravidade` gira a gravidade projetada pela matriz `v3_incl_imu`. Os dois buffers vêm do evento `vies_imu` (seção 2.2) (`observacoes.py:202-209`).

### 6.7 No modo play

O modo `play` remove a corrupção do ator, o empurrão, a força na caixa, o RFI, as DR `pd`, `forca_motor` e `solref`, e o currículo `v3`. O episódio vale `int(1e9)` s (`env_cfg.py:62-63`, `env_cfg.py:105`, `env_cfg.py:113-114`, `env_cfg.py:169-181`, `env_cfg.py:200-216`).

---

## 7. Recompensas e penalidades

Cada termo devolve um valor `v` por env e por passo. A recompensa do passo é `peso × v × step_dt`, onde `step_dt` = 0,02 s (`reward_manager.py:118-127`; seção 1.1). Um termo com peso 0 não roda. O manager troca NaN e infinito por 0 (`reward_manager.py:129`).

Todo termo recebe `tarefas` e vale 0 fora delas, com `torch.where` (`recompensas.py:361-365`). Nos termos abaixo, `tarefas` contém "PEGAR". A tupla vem de `env_cfg.py:32-54`.

Os termos que NÃO valem no PEGAR (tupla `ANDAR_CARREGAR`): `rastreio_linear`, `rastreio_angular`, `passo`, `vivo`, `altura_pe`, `desvio_bracos`, `desvio_cintura`, `desvio_quadril`, `tronco_plano`, `altura_base` (`env_cfg.py:43-48`).

O fator `f_est` que aparece abaixo é `por_restricao(env, Recompensa.estilo_inicio)` = `estilo_inicio + (1 − estilo_inicio)·restrição`, com `Recompensa.estilo_inicio` = 0,25 (`knobs.py:260`) e a restrição de 5.7 (`recompensas.py:453-459`). Ele vale para todos os envs, inclusive os da PEGAR. O fator `por_restricao(env, 0,0, so="ANDAR")` vale 1 nos envs fora do ANDAR, então não reduz nada no PEGAR (`recompensas.py:456-459`).

### 7.1 Tabela dos 23 termos

| Termo | Peso (knob = valor) | Fórmula resumida de `v` | Fatores que multiplicam | Vale zero quando |
|---|---|---|---|---|
| `escorrego` | `Recompensa.escorrego` = −0,2 (`knobs.py:253`) | Σ_pés em contato de ‖v_xy do tornozelo‖ (`recompensas.py:100-103`, `639-644`) | — | Nenhum pé em contato |
| `vz` | `Recompensa.vz` = −2,0 (`knobs.py:254`) | v_z² da base no mundo (`recompensas.py:106-108`, `647-649`) | — | — |
| `giro_xy` | `Recompensa.giro_xy` = −0,05 (`knobs.py:255`) | ω_x² + ω_y² da base no frame da base (`recompensas.py:111-113`, `652-654`) | `f_est` | — |
| `vel_junta` | `Recompensa.vel_junta` = −0,001 (`knobs.py:256`) | Σ q̇² nas 29 juntas (`recompensas.py:116-118`, `657-659`) | `f_est` | — |
| `acel_junta` | `Recompensa.acel_junta` = −5e-8 (`knobs.py:257`) | Σ q̈² nas 29 juntas (`recompensas.py:121-123`, `662-664`) | `f_est` | — |
| `taxa_acao` | `Recompensa.taxa_acao` = −0,01 (`knobs.py:258`) | Σ (a − a_anterior)² da saída crua (`recompensas.py:126-128`, `667-670`) | — | — |
| `energia` | `Recompensa.energia` = −2e-5 (`knobs.py:259`) | Σ \|τ·q̇\|, com τ = `qfrc_actuator` (`recompensas.py:131-133`, `673-677`) | `f_est` | — |
| `limite_junta` | `Recompensa.limite_junta` = −5,0 (`knobs.py:262`) | Σ_j relu(\|q − c\|/m − 0,95)/(1 − 0,95), com `Recompensa.limite_mole` = 0,95 (`knobs.py:263`) (`potencial.py:70-85`; `recompensas.py:136-138`, `680-684`) | — | Todas as juntas abaixo de 95% do curso |
| `cerca_sola` | `Recompensa.cerca_sola` = −0,05475 (`knobs.py:270`) | Σ_pés em contato relu(inclinação da sola° − `lim_sola_graus`), com `Recompensa.lim_sola_graus` = 5,0 (`knobs.py:277`) (`recompensas.py:156-158`, `716-721`) | — | Pé fora do chão ou sola dentro de 5° |
| `cerca_giro` | `Recompensa.cerca_giro` = −0,03965 (`knobs.py:271`) | relu(max(−q_yaw_E, 0)° − 30) + relu(max(q_yaw_D, 0)° − 30), com `Recompensa.lim_giro_graus` = 30,0 (`knobs.py:278`) (`recompensas.py:161-168`, `724-727`) | `por_restricao(0,0, so="ANDAR")` = 1 | Os dois hip_yaw não giram para dentro além de 30° |
| `cerca_tronco` | `Recompensa.cerca_tronco` = −0,04596 (`knobs.py:272`) | relu(incl_torso° − (tronco_ref° + `lim_tronco_graus`)), com `Recompensa.lim_tronco_graus` = 15,0 (`knobs.py:279`). `tronco_ref` = tronco do alvo corrente (rad → graus; valor em P0 a P2, 0 em P3) (`recompensas.py:171-173`, `730-738`) | idem | Torso dentro da faixa |
| `cerca_com` | `Recompensa.cerca_com` = −22,09035 (`knobs.py:273`) | Com os dois pés em contato: relu(`lim_com` − margem do CoM). `Recompensa.lim_com` = 0,03 m (`knobs.py:280`) (`recompensas.py:176-179`, `741-744`) | idem | Menos de dois pés em contato, ou margem ≥ 0,03 m |
| `cerca_torque` | `Recompensa.cerca_torque` = −0,29615 (`knobs.py:274`) | (Σ_soltas relu(\|τ\|/τ_max − 0,8) + Σ_pares relu((\|τ_a\| + \|τ_b\|)/50 − 0,8)) / 0,2. `lim_torque` = 0,8 (`knobs.py:281`), `banda_torque` = 0,2 (`knobs.py:282`), `tau_par` = 50,0 N·m (`knobs.py:298`) (`recompensas.py:182-193`, `747-752`) | idem | Todos os torques abaixo de 80% |
| `cerca_pelve` | `Recompensa.cerca_pelve` = −4,45279 (`knobs.py:275`) | relu(\|altura_pelve − pelve_ref\| − `lim_pelve`), com `Recompensa.lim_pelve` = 0,08 m (`knobs.py:283`) (`recompensas.py:196-198`, `755-768`) | idem | Pelve a menos de 0,08 m da referência |
| `cerca_recuo` | `Recompensa.cerca_recuo` = −10,39719 (`knobs.py:276`) | relu(\|recuo − recuo_ref\| − `lim_recuo`), com `Recompensa.lim_recuo` = 0,08 m (`knobs.py:284`) (`recompensas.py:201-203`, `771-782`) | — | Recuo a menos de 0,08 m da referência |
| `contato_fora` | `Recompensa.contato_fora` = −1,0 (`knobs.py:299`) | Nº de corpos fora dos pés em contato com o terreno acima de `contato_fora_N` + nº de corpos em autocolisão + nº de fatias de contato robô–caixa acima de `contato_fora_N` (esta parcela só fora da janela da pega). `Recompensa.contato_fora_N` = 1,0 N (`knobs.py:300`) (`recompensas.py:206-211`, `785-793`) | — | Sem contato indevido; a parcela da caixa vale 0 com o alvo em P1, P2 ou P3 |
| `contato_laje` | `Recompensa.contato_laje` = −2,0 (`knobs.py:301`) | Σ nos 3 sensores de laje de clip((F − joelho)/sat, 0, 1) (`recompensas.py:887-896`) | — | Força abaixo de `joelho` em todos os sensores |
| `freio` | `Recompensa.freio` = −6/29 (`knobs.py:304`) | Σ_j relu(\|q̇_j\|/vmax_j − 1)² (`recompensas.py:899-919`) | `Recompensa.freio_fator` ^ `v3_freio` (base 1,5, `knobs.py:305`) × rampa `min(s/0,75; 1)` | Velocidade de todas as juntas abaixo do teto |
| `progresso` | 1,0 fixo (`env_cfg.py:57`) | `renda_etapa·etapa·h + renda_phi·Φ + renda_conquista·C` (7.2) | — | Espera sem alvo; caixa perdida |
| `sacudida` | `Recompensa.sacudida` = −0,05 (`knobs.py:353`) | Σ(Δv/(dt·5,04))² + Σ(Δω/(dt·31,56))² da caixa (`recompensas.py:339-342`, `902-925`) | `degrau/5` | Alvo ≠ P3; degrau 0 |
| `aperto` | `Recompensa.aperto` = −0,05 (`knobs.py:355`) | min(F_esq, F_dir)/`F_ref`, com `Recompensa.F_ref` = 49,0 N (`knobs.py:359`) (`potencial.py:55-62`; `recompensas.py:948-957`) | `degrau/5` | Alvo ≠ P3; degrau 0 |
| `perda_caixa` | −1,0 fixo (`env_cfg.py:57`) | `perdeu_agora × Recompensa.perda_caixa`, com `perda_caixa` = 240,0 (`knobs.py:367`) (`recompensas.py:311-313`, `940-942`) | — | Passo sem pulso de perda |
| `terminacao` | `Recompensa.terminacao` = −300,0 (`knobs.py:366`) | 1 se o env terminou por algo que não é `time_out` (`recompensas.py:214-216`, `945-948`) | — | Passo sem terminação |

Os pesos × `step_dt` dão o custo por passo. Exemplo: `terminacao` custa −300,0 × 0,02 = −6,0 por queda. `perda_caixa` custa −1,0 × 240,0 × 0,02 = −4,8 no passo seguinte à perda (`comando.py:394-399`).

### 7.2 Fórmula completa do `progresso`

O `progresso` é uma renda por segundo, não uma diferença de potencial (`recompensas.py:286-291`, `796-864`).

`v = renda_etapa·etapa·h_ef + renda_phi·Φ_pago + renda_conquista·C`

com `Recompensa.renda_etapa` = 6,0 (`knobs.py:339`), `Recompensa.renda_phi` = 3,0 (`knobs.py:338`) e `Recompensa.renda_conquista` = 2,0 (`knobs.py:343`).

**Etapa e Φ pago** (`recompensas.py:851-855`), com a precedência abaixo. `sem_caixa = v3_perdeu | (~v3_pega_fechada & tarefa ≠ PEGAR)`. No PEGAR, `sem_caixa` vale `v3_perdeu`. `fecho_pega = v3_pega_fechada & (v3_alvo_idx < B1)`.

| Condição | `etapa` | `Φ_pago` |
|---|---|---|
| `sem_caixa` | 0 | 0 |
| `v3_bota_fechada` | 7 | 0 |
| `fecho_pega` (P3 fechado; vale no PEGAR e no CARREGAR, porque o índice fica em P3) | 4 | 0 |
| `v3_alvo_idx ≥ 0` | `v3_alvo_idx` | `Φ_c` (abaixo) |
| resto (espera, `v3_alvo_idx = −1`) | 0 | 0 |

Com a pega fechada e a tarefa PEGAR, o termo paga `renda_etapa·4` = 6,0 × 4 = 24 por segundo, mais as conquistas (h_ef = 1).

**Φ do alvo** (`recompensas.py:274-283`): `Φ = (1 − tanh(d/(Alvos.largura_tanh·σ)))·fator·Π portões`, com `Alvos.largura_tanh` = 2,0 (`knobs.py:221`). O `fator` vale 1 no PEGAR (só o B1 usa o fator de pouso, `recompensas.py:551`). `σ` é `v3_sigma` (abertura da etapa, 4.3). O `Φ` vale 0 com alvo −1.

A distância `d` (`recompensas.py:224-244`):

| Etapa | `d` |
|---|---|
| P0, P1, P2 | média nas duas palmas de ‖palma no frame da caixa − palma-alvo‖ |
| P3 | `hypot(‖caixa − caixa_alvo_w‖, Δpelve·Alvos.tol_transporte_p3/Alvos.tol_pelve_p3)`, com os valores finais 0,05 e 0,03 (`knobs.py:187-188`) e `Δpelve = altura_pelve − pelve_ref` |

Os portões do Φ para P0..P3 (`recompensas.py:247-271`, `562-567`):

| Portão | Vale em | Fórmula | Constantes |
|---|---|---|---|
| `desloc` | P0, P1, P2 | `rampa(‖xy_caixa − v3_caixa_pega_w xy‖, 1,0·a, 0,5·a)` | `knobs.py:222-223` |
| `incl` | P2, P3 | `rampa(incl_caixa°, incl_lim, 15,0)` | `incl_lim` de 5.4; `knobs.py:205` |
| `maos` | — | O `_phi_env` passa 1,0 (`torch.ones_like`), então o portão não atua no Φ | `recompensas.py:566` |
| `tronco` | P0..P3 | `rampa(relu(incl_torso° − tronco_ref°), Alvos.faixa_tronco_graus, Alvos.delta_tronco_graus)` | `faixa_tronco_graus` = 15,0° (`knobs.py:226`); `delta_tronco_graus` = 15,0° (`knobs.py:229`) |
| `recuo` | P0..P3 | `rampa(\|recuo − recuo_ref\|, Alvos.faixa_recuo, Alvos.delta_recuo)` | `faixa_recuo` = 0,08 m (`knobs.py:227`); `delta_recuo` = 0,30 m (`knobs.py:230`) |
| `com` | P0..P3, com dois pés em contato | `rampa(relu(Alvos.faixa_com − margem), 0, Alvos.delta_com)` | `faixa_com` = 0,03 m (`knobs.py:228`); `delta_com` = 0,03 m (`knobs.py:231`) |

⚠ Divergência: a docstring de `portoes_do_alvo` (`recompensas.py:254-255`) diz que o `_phi_env` passa a firmeza da pega como `maos`. O código passa `torch.ones_like(maos)` (`recompensas.py:566`).

`recuo` = ‖xy da caixa − xy da pelve‖ (`recompensas.py:531`). `recuo_ref` é a norma xy de `caixa_transporte_b(a)` no P3, e `ref.recuo(topo, a)` nas outras etapas do PEGAR (`recompensas.py:509-516`). `tronco_ref` = `ref_do_alvo(idx)[7]` em rad. Vale `ref.tronco(topo, a)` em P0, P1 e P2, e vale 0 em P3 e com índice < 0 (`observacoes.py:93`, `observacoes.py:123-130`).

**Φ_c** (`recompensas.py:850`): `Φ_c = Φ` em P0, P1 e P2. `Φ_c = Φ · c₂` em P3, com `c₂` = `conquistas[2]` (o peso nas mãos presas, abaixo).

**Conquistas** (`recompensas.py:557-563`): três números em [0, 1].

| `c` | Fórmula | Constantes |
|---|---|---|
| `c₀` | `1 − tanh(d_faces/(Alvos.largura_tanh·Recompensa.sigma_palmas))`, com `d_faces` = distância média das palmas às palmas-alvo do P2 | `sigma_palmas` = 0,08 m (`knobs.py:344`) |
| `c₁` (pega firme) | `tanh(min(F_esq, F_dir)/F_pega)`, com `F_pega = margem_pega·m·g/(2·μ)`, `m` = `v3_massa`, `g` = gravidade do modelo, `μ = DR.atrito_pad[0]` | `Recompensa.margem_pega` = 2,5 (`knobs.py:365`); `DR.atrito_pad[0]` = 0,6 (`knobs.py:523`) |
| `c₂` (peso nas mãos) | `clip(1 − apoio_N/peso_N, 0, 1)·c₁`, com `apoio_N` = força vertical |Fz| somada no sensor `apoio_caixa`, `peso_N = m·g` | — |

`F_esq` e `F_dir` somam, nas 12 fatias do sensor `caixa_faces`, a componente normal do contato das faces ±y da caixa. A face +y é a da palma esquerda. As faces ±x e ±z não entram (`recompensas.py:345-356`).

Combinação em `progresso` (`recompensas.py:874-882`):

`C = mantem · (c₀ + c₁·almofadas + c₂·com_peso)`

- `mantem = ~sem_caixa & ~v3_bota_fechada & (fecho_pega | idx ∈ {P3, B1} | idx == P2)`. Em P0 e P1 `mantem` vale False: sem conquista.
- `almofadas` = fração das duas palmas em contato (`_contato_palmas`) em P2, e 1,0 nos outros casos.
- `com_peso` = 1 em P3 e depois do fecho do P3, senão 0.

**O fator `h`** (`recompensas.py:858-870`; `n_pressa`, `recompensas.py:294-296`):

- `h_ef = h` quando a etapa está aberta com Φ pago (`com_alvo`), senão 1.
- Com etapa aberta: `h ← min(1, h·exp(−dt/τ) + recarga·ganho)`, com `τ = Recompensa.tau_renda` = 3,0 s (`knobs.py:347`), `recarga = Recompensa.recarga_renda` = 5,0 (`knobs.py:348`), `ganho = relu(Φ_c − v3_phi_melhor)`.
- `v3_phi_melhor ← max(v3_phi_melhor, Φ_c)`.
- `h` volta a 1,0 e `v3_phi_melhor` a 0,0 em: abertura do P0, troca de etapa, abertura de trecho, perda da caixa, reset (`comando.py:388`, `413`, `500`, `579`, `653`).

O termo grava `v3_phi` = Φ cru do alvo corrente (−1 sem alvo) (`recompensas.py:883`).

### 7.3 Fórmulas dos demais termos

**Frame comum**: `incl_torso` é a inclinação do `torso_link` em graus (`recompensas.py:467-470`). `altura_pelve` é a leitura do sensor `altura_pelve` (`cena.py:394`).

**`cerca_com`** (`recompensas.py:316-336`, `471-477`, `417-433`): `margem` é a distância com sinal do CoM às arestas do fecho convexo dos 8 cantos das solas, positiva dentro. O CoM é o combinado do robô inteiro e da caixa. A caixa entra só com `v3_pegou & ~v3_perdeu & idx ∈ {P3, B1, B2}`. Em PEGAR, vale depois do fecho do P2 (a tarefa mantém `idx = P3`).

**`cerca_torque`** (`recompensas.py:182-193`): os pares acoplados são `Recompensa.pares_torque` (`knobs.py:293-297`): (waist_pitch, waist_roll), (left_ankle_pitch, left_ankle_roll), (right_ankle_pitch, right_ankle_roll). Os `τ_max` por junta estão em `Recompensa.tau_max` (`knobs.py:286-292`). A junta de um par não entra como solta.

**`cerca_pelve`** (`recompensas.py:775-788`): `pelve_ref` = `ref_do_alvo(idx)[6]` com alvo ≥ 0 (P0, P1, P2 usam a grade `pelve_z(topo, a)`; P3 usa `pelve_com_caixa(a)`). Com `idx < 0`, `pelve_ref = pelve_de_pe`.

**`cerca_tronco`** (`recompensas.py:750-758`): `tronco_ref` = `ref_do_alvo(idx)[7]` convertido a graus. Vale `ref.tronco(topo, a)` em P0, P1 e P2, e 0 em P3 e com `idx < 0` (`observacoes.py:123-130`).

**`cerca_recuo`** (`recompensas.py:791-802`): sem exceção no PEGAR. A exceção `solta` só vale no BOTAR.

**`contato_fora`** (`recompensas.py:805-813`): o sensor `nao_pes_chao` é o contato de todo corpo, menos os links do tornozelo, com o terreno (`cena.py:402-408`). A parcela da caixa conta as fatias do sensor `caixa_robo` com ‖F‖ > 1 N (12 fatias, `cena.py:375-378`). A `janela` é `idx ∈ {P1, P2, P3, B1, B2}` ou tarefa CARREGAR. No PEGAR, a parcela da caixa só vale com `idx` em P0 ou −1 (`recompensas.py:59`, `792`).

**`contato_laje`** (`recompensas.py:887-896`): `joelho = por_degrau(50,0; 10,0)`. `sat = max(Recompensa.laje_saturacao_N − joelho, 1e-6)`, com `laje_saturacao_N` = 100,0 N (`knobs.py:303`). Para cada um dos três sensores `corpo_prateleira`, `palma_prateleira`, `dorso_prateleira` (`cena.py:49-50`), soma `clip((F − joelho)/sat, 0, 1)`, com `F` = máximo, sobre as fatias, da norma da força.

**`freio`** (`recompensas.py:899-919`): regime pelo comando. `total = ‖cmd_xy‖ + |ω|`. Parado: `total < Recompensa.freio_limiar_andar` = 0,05. Andando: `total < Recompensa.freio_limiar_correr` = 1,5. Correndo: o resto (`knobs.py:308-309`). No PEGAR o comando vale zero, então o regime é "parado" e o teto de cada junta vem de `Recompensa.vel_max_parado` (`knobs.py:311-317`): 1,5 rad/s nas juntas de perna, cintura, ombro e cotovelo, e 1,0 rad/s no punho. O custo é `Σ relu(|q̇|/vmax − 1)²`. A rampa é `min(s_médio_das_células_abertas_da_tarefa/Balanceador.portao, 1)`, com `portao` = 0,75 (`knobs.py:391`). O expoente do `freio_fator` é `v3_freio` (nível + 1).

**`sacudida`** (`recompensas.py:922-945`): Δv e Δω são as diferenças da velocidade da caixa entre duas chamadas seguidas do termo, em `root_link_vel_w` (`recompensas.py:936-939`). `Recompensa.sacudida_norma` = (5,04; 31,56) (`knobs.py:358`). A `v_ant` é da classe e anda em todo passo, mesmo fora do P3. O valor passa só com `idx ∈ {P3, B1}`, o que no PEGAR é o P3 e o período depois do fecho do P3 (o índice fica em P3).

**`aperto`** (`recompensas.py:948-957`): mesma máscara. `valor = min(F_esq, F_dir)/F_ref · degrau/5`.

**`terminacao`** (`recompensas.py:965-968`): `terminated & ~time_outs`.

### 7.4 Efeitos sobre o PPO ligados à PEGAR

- Normalização da vantagem por grupo, em cada coleta de uma iteração (24 passos por env, `tarefa.py:49`) (`algoritmo.py:34-60`, `algoritmo.py:98-112`, `algoritmo.py:137-169`): a vantagem crua `returns − values` é normalizada (média 0, desvio 1) dentro de cada grupo de pelo menos 2 amostras. O grupo é 0 para a tarefa ANDAR com o slot "nenhum" (o último, canal 10), e `1 + tarefa·11 + slot` nos demais casos, com `tarefa` = argmax das 6 últimas colunas do crítico, e `slot` = argmax da fatia `alvo_macro` (11 canais). Os grupos da PEGAR são: P0 = 12, P1 = 13, P2 = 14, P3 = 15, P3 com pega fechada = 19, sem alvo = 22. Um grupo ralo (menos de 2 amostras) diferente do grupo 0 usa a média e o desvio de todas as amostras de grupo diferente de 0, se elas somam pelo menos 2. O grupo 0 ralo fica com a vantagem crua.
- Bônus de futuro negativo em queda e perda (`algoritmo.py:127-131`): a partir do segundo rollout (quando `v_grupo` existe), numa terminação sem `time_out` e no passo com `v3_perdeu_agora`, a recompensa soma `γ·min(V̄_g, 0)`, com `V̄_g` a média do valor do grupo no rollout anterior. O grupo vem das observações do crítico gravadas no passo. `γ` = 0,99 (`mjlab/tasks/velocity/config/g1/rl_cfg.py:37`).
- Normalizador da observação: o ator e o crítico têm termos com média 0 e desvio fixo, fora da estatística empírica: `tarefa`, `mapa` (desvio 1,0) e `caixa` (desvio 0,1) no ator; no crítico, além desses, `alvo_macro`, `niveis`, `degrau`, `massa`, `tarefa_critico` e `privilegiados` (desvio 1,0), `caixa_verdade`, `alvo_verdade` e `referencia` (desvio 0,1) e `forca_caixa` (desvio 0,3) (`runner.py:32-34`, `runner.py:46-66`). O piso do desvio dos demais canais é `PISO_DESVIO` = 0,05, literal em `algoritmo.py:66` (aplicado em `algoritmo.py:69-83`).
- Um `time_out` vira bootstrap: `rewards += γ·V·time_outs` (`rsl_rl/algorithms/ppo.py:150-155`).

---

## 8. Terminações

Os cinco termos de `env_cfg.py:137-147`. Cada um vale no PEGAR conforme a tabela.

| Termo | Tipo | Condição exata no PEGAR | Referência |
|---|---|---|---|
| `time_out` | Truncamento (`time_out=True`) | `episode_length_buf ≥ max_episode_length` (1000 passos) ou `v3_fim`. A travessia do terreno do ANDAR (`pose_ult` a mais de `Terreno.dist_travessia` = 3,5 m da origem da célula) também dispara o `time_out`, mas só na fatia ANDAR: no PEGAR ela vale sempre False. | `terminacoes.py:53-72`; `env_cfg.py:138` |
| `queda` | Terminação | `incl_rad > radians(Terminacao.incl_manip_graus)` ou `(z_joelho − z_chao_joelho) < Terminacao.joelho_min`. O joelho é o de menor altura sobre o terreno. `incl_manip_graus` = 70,0° (`knobs.py:377`). `joelho_min` = 0,05 m (`knobs.py:378`). | `terminacoes.py:35-50`, `74-100` |
| `caixa_perdida` | Terminação | Nunca dispara com `caixa_perdida_vira_andar` = True. | `terminacoes.py:75-83`, `103-113`; `knobs.py:161` |
| `chegada` | Terminação | Só vale no BOTAR (`tarefas=("BOTAR",)`). Nunca dispara no PEGAR. | `env_cfg.py:144` |
| `batente` | Terminação | `(any(\|q − c\|/m > Terminacao.batente_frac) \| any(\|q_hip_yaw\| > Terminacao.hip_yaw_max)) & (v3_degrau ≥ Refino.degrau_batente)`. `batente_frac` = 1,0 (`knobs.py:383`). `hip_yaw_max` = 1,0 rad (`knobs.py:382`). `degrau_batente` = 1 (`knobs.py:547`). A comparação é estrita. `c` e `m` são o centro e o meio curso dos limites duros do modelo, nas 29 juntas. | `terminacoes.py:154-183` |

A inclinação `incl_rad` é `acos(−projected_gravity_b[z])` (`terminacoes.py:97`). A altura do joelho e do chão do joelho vêm dos sensores `altura_joelhos` (`terminacoes.py:103-107`).

O `v3_degrau` que o `batente` lê é o degrau da tarefa publicada na última abertura de trecho (`comando.py:516-518`). Ele não muda durante o trecho PEGAR.

`time_out` não conta como queda. Se um `time_out` e uma terminação ocorrem no mesmo passo, `n_terminacao` e o `falhou` do registro tratam o passo como `time_out` (`recompensas.py:214-216`; `comando.py:643`).

O que acontece depois de uma terminação:

1. O passo calcula as recompensas, com `terminacao` = −300,0 × 0,02 = −6,0 (`knobs.py:366`).
2. O env reseta no mesmo passo (`manager_based_rl_env.py:443-448`).
3. `CadeiaCmd._resample` registra o PEGAR pendente com `falhou = ~time_outs`. Uma queda depois do P0 aberto conta como falha da célula (seção 5.1).
4. O PPO soma o bônus `γ·min(V̄_g, 0)` (seção 7.4).

---

## 9. Observações

### 9.1 Estrutura (`observacoes.py:361-422`)

O ator e o crítico leem os mesmos termos de base, na ordem de `Contrato.blocos_ator` (`knobs.py:58-63`): `giro_base`, `gravidade`, `comando`, `fase`, `juntas`, `vel_juntas`, `acao`, `tarefa`, `caixa`, `mapa`. Cada um (menos o `mapa`) tem janela de `Contrato.historico` = 5 passos. O ator tem ruído e atraso (seção 6.6). O crítico não tem.

A ação do robô é a posição-alvo de cada junta: `q* = q_de_pe + escala·a`, com `Contrato.escala_acao` = 0,25 e `Contrato.escala_acao_punho` = 0,075 (`knobs.py:52-53`; `env_cfg.py:108-109`).

### 9.2 Termos do ator próprios da PEGAR

| Termo | Canais | Valor no PEGAR | Referência |
|---|---|---|---|
| `tarefa` | 6 | One-hot de `v3_tarefa` na posição 1. Sem ruído. | `observacoes.py:135-137` |
| `caixa` | 11 | 8 canais da caixa estimada (posição no frame da base estimada, rotvec, meia aresta, idade da leitura) mais 3 do alvo (`v3_alvo_w` no frame da base estimada). Só os 3 canais do alvo valem zero quando `v3_alvo_idx < 0` (espera do PEGAR). Os 8 canais da caixa seguem a leitura do estimador: os 7 primeiros valem zero enquanto nunca houve leitura válida no episódio. | `observacoes.py:172-199`; `estimador.py:378-387` |
| `comando` | 3 | Zero no PEGAR. | seção 3.1 |
| `fase` | 2 | Seno e cosseno de `2π·v3_fase_marcha`. Constante no PEGAR (seção 3.1). | `observacoes.py:298-314` |
| `mapa` | 882 | O mapa do lidar. É o mesmo em todas as tarefas. | `observacoes.py:154-157` |

O termo `caixa` recebe, na `saida_b`, a leitura entregue pelo estimador: a posição, a rotação reduzida pela simetria do cubo (24 rotações), a meia aresta e a idade, e `out[~seen, :7] = 0` se nunca houve leitura (`estimador.py:378-387`). O alvo vai ao frame da base estimada (`estimador.py:191-194`).

O normalizador do ator não normaliza os termos `tarefa` e `mapa` (desvio fixo 1,0) e fixa `caixa` em desvio 0,1 e média 0 (`runner.py:32-34`).

### 9.3 Termos do crítico próprios da PEGAR

| Termo | Canais | Valor no PEGAR | Referência |
|---|---|---|---|
| `vel_lin_base` | 3 (janela de 5) | Velocidade linear verdadeira da base (sensor `imu_lin_vel`). | `observacoes.py:405-407` |
| `caixa_verdade` | 7 | Posição (3), rotvec (3) e meia aresta (1) verdadeiros da caixa no frame da base. | `observacoes.py:267-273` |
| `alvo_verdade` | 3 | `v3_alvo_w` no frame da base verdadeira. Zero com alvo < 0. | `observacoes.py:276-279` |
| `alvo_macro` | 11 | One-hot: P0..P3 = canais 0 a 3. Pega fechada (índice P3 e `v3_pega_fechada`) = canal 7. Sem alvo = canal 10. A perda (canal 9) tem prioridade sobre os outros, mas muda a tarefa para ANDAR. | `observacoes.py:282-295` |
| `referencia` | 8 | Palmas-alvo (6), pelve de referência (1), tronco de referência (1) do alvo corrente. Zero com alvo < 0. No P3 as palmas-alvo e o tronco valem 0, e a pelve vale `pelve_com_caixa(a)` (`observacoes.py:122-130`). | `observacoes.py:116-130`, `321-324`; `recompensas.py:501-506` |
| `niveis` | 4 | `nível/nível máximo` de ANDAR, PEGAR, CARREGAR, BOTAR. | `observacoes.py:327-335` |
| `degrau` | 1 | `v3_degrau/5`. | `observacoes.py:338-340` |
| `massa` | 1 | `v3_massa/5` (kg). | `observacoes.py:343-345` |
| `forca_caixa` | 1 | Soma das normas das 12 fatias de `caixa_robo`, dividida por 100 N. | `observacoes.py:348-356`; `metricas.py:338-345` |
| `privilegiados` | 4 | `[v3_marcha_on, v3_h, relu(v3_phi_melhor − max(v3_phi, 0)), z da etapa corrente (0 com alvo < 0)]`. | `observacoes.py:298-318` |
| `tarefa_critico` | 6 | One-hot de `v3_tarefa`. É o último termo (o PPO lê as 6 últimas colunas). | `observacoes.py:417`; `algoritmo.py:106` |

---

## 10. Métricas

O treino registra estas métricas durante o PEGAR. Os registros de métrica `Episode_Metrics/<nome>` saem da média sobre os envs que resetam. A redução é `mean` (padrão), `max` ou `last`, conforme cada termo (`mjlab/managers/metrics_manager.py:107-128`). As chaves `Episodio/<nome>` saem de `_loga_episodios`: um valor por episódio fechado com o evento (`metricas.py:81-90`).

### 10.1 Métricas da cena e do PEGAR (`metricas.py:522-571`)

| Chave | Redução | O que mede | Referência |
|---|---|---|---|
| `phi_P0`, `phi_P1`, `phi_P2`, `phi_P3` | `last` | `Episode_Metrics/phi_<alvo>`: fração dos episódios que tiveram o alvo. `Episodio/phi_<alvo>`: média do Φ (`v3_phi`) nos passos com o alvo e tarefa com caixa. | `metricas.py:245-270`, `558-560` |
| `tempo_P0`..`tempo_P3` | `last` | `Episodio/tempo_<alvo>`: segundos com o alvo corrente. | `metricas.py:273-294` |
| `erro_p3_vertical`, `erro_p3_lateral`, `erro_p3_pelve` | `last` | `Episodio/erro_p3_*`: erro médio no P3 (m): vertical da caixa ao alvo, lateral da caixa ao alvo, e `|altura_pelve − pelve_com_caixa|`. | `metricas.py:297-326` |
| `inclinacao_caixa_p3` | `max` | Pico da inclinação da caixa (rad) nos passos com alvo P3. | `metricas.py:329-335` |
| `forca_robo_caixa` | `mean` | Soma das normas das fatias de força robô–caixa (N). | `metricas.py:338-345` |
| `deslocamento_caixa_laje` | `max` | Maior deslocamento xy da caixa desde o primeiro passo do P0, nos passos de P0 a P2. | `metricas.py:348-373` |
| `aceleracao_da_caixa` | `last` | `Episodio/aceleracao_da_caixa`: média do custo da sacudida nos passos com alvo P3 ou B1, sem o fator do degrau. | `metricas.py:459-491` |
| `idade_leituras` | `mean` | Idade da leitura do estimador, zero onde a tarefa é ANDAR. | `metricas.py:494-507` |
| `fracao_avisos` | `mean` | Fração dos passos depois do aviso no trecho (o fecho do P3 no PEGAR). | `metricas.py:510-519` |
| `fracao_mapa_visto` | `mean` | Fração das células do recorte com confiança acima de `Percepcao.limiar_visto` = 0,5 (`knobs.py:503`). | `metricas.py:222-232` |

Métricas gerais, ativas em todas as tarefas: `tempo_de_voo`, `pico_de_altura`, `velocidade_de_escorrego`, `velocidade_de_junta`, `fracao_do_curso` (`max`), `forca_de_pouso`, `passo_medio` (`metricas.py:533-556`).

O código registra ainda `velocidade_de_chegada`, `deslocamento_depois_do_apoio` e `fracao_fora_da_plataforma` (`metricas.py:564-568`). Este documento não as descreve. A primeira e a segunda medem o BOTAR (`metricas.py:376-424`), e a terceira mede o CARREGAR da fatia B em terreno (`metricas.py:427-448`).

Sinal do `erro_p3_vertical`: é o alvo menos a caixa em z (`−e[z]`), com sinal. O `erro_p3_lateral` e o `erro_p3_pelve` são valores absolutos (`metricas.py:309-314`).

### 10.2 Termos do `Episode_Reward`

O gerenciador de recompensas registra `Episode_Reward/<termo>` = soma por episódio de cada termo (já × peso × `step_dt`) dividida por `max_episode_length_s` (20,0 s, `Cadeia.episodio_s`) (`mjlab/managers/reward_manager.py:106-110`).

### 10.3 Chaves do currículo para o PEGAR (`curriculo.py:103-143`)

Publicadas como `Curriculum/v3/<chave>`, a cada reset:

| Chave | Valor |
|---|---|
| `frac_ANDAR`, `frac_B`, `frac_C` | Frações das fatias (1.4). |
| `s_PEGAR_pegar` | Nota `s` da célula PEGAR. |
| `nivel_PEGAR` | Nível do PEGAR. |
| `subnivel_PEGAR` | Subnível do PEGAR. |
| `degrau_PEGAR` | Degrau do PEGAR. |
| `z_P0`..`z_P3` e `taxa_P0`..`taxa_P3` | Zona de cada etapa e a última taxa de fecho por visita (−1 sem medida). |
| `degrau_global`, `restricao` | Degrau global e restrição. |
| `atraso_max`, `mu_min_pad` | Teto de atraso e borda de baixo do atrito da almofada. |

### 10.4 Métricas do PPO

O `RunnerV3` grava `Policy/std/<junta>` = desvio da ação por junta (`runner.py:68-86`).

---

## Não determinado pelo código lido

- Não determinado pelo código lido: os valores numéricos das grades e poses de referência `ref.recuo`, `ref.pelve_z`, `ref.tronco`, `ref.caixa_transporte_b`, `ref.pelve_com_caixa`, `ref.palmas_de_pe_b`, `ref.cantos_sola`, `ref.q_de_pe` e `ref.pelve_de_pe`. O código as lê de `g1_v3/dados/referencias.npz` (`alvos.py:47-67`; `knobs.py:73`). O arquivo de dados não foi lido.
- Não determinado pelo código lido: o sinal e a escala exatos da leitura do sensor `altura_caixa` (o ponto do corpo da caixa que o anel de raios mede). `cena.py:392-397` só dá o raio do anel, o número de amostras e a regra de alinhamento. O cálculo do valor está em `mjlab/sensor`, não lido.
- Não determinado pelo código lido: a distribuição interna de `dr.geom_friction` por geom quando `shared_random` é falso foi lida em `_core.py`, mas o valor efetivo de `solref` e do atrito no contato caixa–laje depende da prioridade das geoms no MuJoCo, que o código da v3 só declara (`cena.py:106-127`).
