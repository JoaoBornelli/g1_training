# Terminações, observações e ação

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:

- Código da v3: `terminacoes.py`, `observacoes.py`, `estimador.py`, `percepcao_lidar.py`, `env_cfg.py`, `knobs.py`, `cena.py`, `curriculo.py`, `tarefa.py`, `runner.py`, `algoritmo.py`; trechos de `eventos.py`, `comando.py`, `balanceador.py`, `recompensas.py`, `alvos.py`, `metricas.py`, `potencial.py`, `exporta_cena.py`; dados `dados/referencias.npz`, `dados/mid360_passos.npz`, `dados/capsulas.npz`.
- `mjlab`: `envs/manager_based_rl_env.py`, `envs/mdp/observations.py`, `envs/mdp/actions/actions.py`, `envs/mdp/dr/actuator.py`, `envs/mdp/dr/joint.py`, `managers/action_manager.py`, `managers/observation_manager.py`, `managers/termination_manager.py`, `managers/reward_manager.py`, `utils/buffers/delay_buffer.py`, `utils/buffers/circular_buffer.py`, `utils/noise/noise_cfg.py`, `actuator/actuator.py`, `actuator/builtin_actuator.py`, `actuator/builtin_group.py`, `utils/spec.py`, `entity/entity.py`, `entity/data.py`, `asset_zoo/robots/unitree_g1/g1_constants.py`, `asset_zoo/robots/unitree_g1/xmls/g1.xml`, `sensor/terrain_height_sensor.py`, `sensor/raycast_sensor.py`, `sensor/sensor.py`, `sim/sim.py`, `rl/vecenv_wrapper.py`, `rl/config.py`, `tasks/velocity/mdp/velocity_command.py`, `tasks/velocity/config/g1/rl_cfg.py`, `managers/command_manager.py`, `envs/mdp/events.py`, `utils/actuator.py`, `terrains/terrain_entity.py`, `tasks/velocity/mdp/terminations.py`.
- `rsl_rl`: `algorithms/ppo.py`, `modules/normalization.py`.

## 1. Termos usados

- **env**: uma cópia paralela do robô e da cena. O treino roda muitos envs ao mesmo tempo.
- **passo da política**: um intervalo em que a rede emite uma ação. Dura `Contrato.dt_fisica` (0,005 s, `knobs.py:54`) × `Contrato.decimacao` (4, `knobs.py:55`) = 0,02 s (`mjlab/envs/manager_based_rl_env.py:266-268`). Em cada passo a física avança 4 subpassos.
- **tarefa**: o modo corrente do env. Os trechos da cadeia só publicam ANDAR, PEGAR, CARREGAR e BOTAR (`comando.py:175-179`). O vetor `TAREFAS` tem 6 nomes, mas REORIENTAR e LEVANTAR nunca aparecem num trecho (`knobs.py:42`).
- **cadeia** e **trecho**: a cadeia é a sequência de tarefas de um episódio. Cada trecho é uma tarefa com duração sorteada (`comando.py:175-179`).
- **etapa** ou **alvo**: o ponto de controle dentro de PEGAR e BOTAR. O vetor `ALVOS` lista P0, P1, P2, P3, B1, B2, B3 nos índices 0 a 6 (`knobs.py:44`). O índice −1 significa "nenhum".
- **degrau do refino**: um inteiro de 0 a `Balanceador.degraus` = 5 (`knobs.py:401`) por tarefa. `v3_cur.degrau(tarefa)` o devolve (`balanceador.py:338-339`). O degrau sobe 1 e nunca desce. A cada fecho de iteração do balanceador, a contagem da tarefa cresce 1 quando a tarefa está no último nível (nas tarefas de cadeia, também com a célula `terreno` aberta e a linha de terreno no máximo: no ANDAR, o piso da média das linhas por coluna igual a 12) e a média das notas das células abertas é maior ou igual à média dos portões. Senão a contagem volta a 0. O degrau sobe quando a contagem chega a `Balanceador.iters_refino` = 100 (`balanceador.py:234-248`; `knobs.py:400`). O fecho de iteração ocorre quando `common_step_counter // Balanceador.passos_por_iteracao` muda, com `passos_por_iteracao` = 24 (`curriculo.py:119-122`; `knobs.py:402`).
- **degrau global**: o menor degrau entre as 4 tarefas (`balanceador.py:341-343`).
- **restrição**: o menor valor de `nível / nível máximo` entre as 4 tarefas, um número em [0, 1] (`balanceador.py:345-348`).
- **trava do DR** (randomização de domínio): um booleano do balanceador que só liga e nunca desliga (`balanceador.py:101-103, 198-202`).
- **`v3_*`**: buffers por env que o código da v3 pendura no env (criados em `curriculo.py:39-90`).
- **`ATIVO`**: a instância `Knobs()` de `knobs.py:569`. Os módulos de observação, estimador e terminação leem o `ATIVO` do módulo, e não o `k` que `make_env_cfg` recebe (`observacoes.py:42`, `terminacoes.py:109-110`, `env_cfg.py:81-88`).

## 2. Ordem de um passo da política

| # | Etapa | Referência |
|---|---|---|
| 1 | A ação crua da rede entra no gerente de ação. O termo de ação calcula `crua × escala + offset`. | `manager_based_rl_env.py:419`, `actions.py:154-157` |
| 2 | 4 subpassos. Em cada um: aplica o alvo ao atuador (com atraso), escreve no sim e avança a física. | `manager_based_rl_env.py:421-427` |
| 3 | Os contadores `episode_length_buf` e `common_step_counter` sobem 1. | `manager_based_rl_env.py:429-431` |
| 4 | As terminações rodam. | `manager_based_rl_env.py:436` |
| 5 | As recompensas rodam. | `manager_based_rl_env.py:440` |
| 6 | Os envs terminados sofrem reset (`_reset_idx`). | `manager_based_rl_env.py:444-448, 553-591` |
| 7 | `sim.forward()` recalcula as grandezas derivadas de todos os envs. | `manager_based_rl_env.py:454` |
| 8 | O gerente de comandos roda: a cadeia (perda de caixa, alvos, odometria, vz da caixa) e o `twist` (relógio de marcha). | `manager_based_rl_env.py:456`, `comando.py:596-629, 872-874` |
| 9 | Os eventos de modo `step` (`forca_caixa`, `rfi`) e `interval` (`empurrao`) rodam. | `manager_based_rl_env.py:458-461`, `env_cfg.py:200-210` |
| 10 | `sim.sense()` roda os raios (sensores de altura e lidar). | `manager_based_rl_env.py:463`, `sim/sim.py:522-541` |
| 11 | O gerente de observação calcula os grupos `actor` e `critic`. | `manager_based_rl_env.py:464` |

Três consequências desta ordem:

- Os sensores de altura (`altura_pelve`, `altura_joelhos`, `altura_caixa`) e o lidar só atualizam os dados no `sim.sense()` (`sensor/raycast_sensor.py:568-583` apenas empacota buffers que o `sense` preencheu). As terminações (etapa 4) leem, portanto, a altura do `sense` do passo anterior.
- A cadeia roda depois das terminações. O pulso `v3_perdeu_agora` e o buffer `v3_vz_caixa_ant` valem, nas terminações, pelo estado do passo anterior (`terminacoes.py:75-84`, `comando.py:409-411, 629`).
- O `twist` grava `pose_ult` na etapa 8. A terminação por travessia (etapa 4) e o registro da visita (etapa 6) leem, por isso, a pose do passo anterior (seção 6.2).

## 3. Ação

### 3.1 O que a rede comanda

A rede emite 29 números por passo, um por junta. O termo de ação é `JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=<dict por junta>, use_default_offset=True)` (`env_cfg.py:108-109`). O alvo de posição de cada junta é:

```
alvo_PD = acao_crua × escala_da_junta + q_de_pe
```

(`actions.py:154-157`; o offset vem de `default_joint_pos`, `actions.py:218-219`).

Antes de ir ao atuador, o termo subtrai o viés do encoder da junta: `alvo_enviado = alvo_PD − encoder_bias` (`actions.py:221-224`). O viés é sorteado no início do env (seção 3.6).

Não há limite sobre a ação. O termo de ação não recebe `clip` (`env_cfg.py:108-109`; padrão `None` em `action_manager.py:29`). O `rl_cfg()` não define `clip_actions` (`tarefa.py:45-50`; padrão `None` em `mjlab/rl/config.py:121`), e o invólucro só clampa se ele existir (`rl/vecenv_wrapper.py:75-76`). O alvo também não é limitado ao curso da junta: o atuador de posição tem `ctrllimited = False` (`utils/spec.py:315-322`).

O gerente de ação guarda a ação crua do passo em `action` e as duas anteriores (`action_manager.py:166-169`). No reset do env, as três voltam a zero (`action_manager.py:140-150`).

### 3.2 Escala por junta

`escala_por_junta` devolve uma escala por junta (`knobs.py:589-592`):

| Juntas | Escala (rad por unidade de ação) | Knob |
|---|---|---|
| `left_wrist_pitch_joint`, `left_wrist_yaw_joint`, `right_wrist_pitch_joint`, `right_wrist_yaw_joint` (4) | 0,075 | `Contrato.escala_acao_punho` (`knobs.py:53`) |
| As outras 25 | 0,25 | `Contrato.escala_acao` (`knobs.py:52`) |

⚠ Divergência: o comentário de `env_cfg.py:107` diz "escala 0,25"; o código dá 0,075 aos quatro punhos (pitch e yaw) por `escala_por_junta`.

O `wrist_roll` usa 0,25: a regex do punho casa só `wrist_pitch` e `wrist_yaw` (`knobs.py:591`).

### 3.3 Pose de offset

O offset da ação é a pose de pé `q_de_pe`, 29 valores lidos de `dados/referencias.npz`, chave `q_de_pe` (`knobs.py:584-587`). `cena.py:225` a grava como `init_state.joint_pos`. O `JointPositionAction` copia `default_joint_pos` para o offset (`actions.py:218-219`).

Essa mesma pose é a posição de reset das juntas: o evento `juntas` usa `reset_joints_by_offset` com `position_range` (0,0; 0,0) e `velocity_range` (0,0; 0,0), sem sorteio (`env_cfg.py:159-160`). Ela também é a referência dos termos `juntas` e `vel_juntas` da observação (`mjlab/envs/mdp/observations.py:51-72`; a velocidade padrão é 0 em `cena.py:226`).

A altura inicial da base é `pelve_de_pe` + 0,01 m. O valor de `pelve_de_pe` no arquivo é 0,78905 m (`cena.py:170-173, 223`).

| Junta | Offset (rad) | Junta | Offset (rad) |
|---|---|---|---|
| `left_hip_pitch` | −0,0289 | `right_hip_pitch` | −0,0289 |
| `left_hip_roll` | −0,0048 | `right_hip_roll` | 0,0048 |
| `left_hip_yaw` | 0,0010 | `right_hip_yaw` | −0,0010 |
| `left_knee` | 0,1624 | `right_knee` | 0,1624 |
| `left_ankle_pitch` | −0,1335 | `right_ankle_pitch` | −0,1335 |
| `left_ankle_roll` | 0,0049 | `right_ankle_roll` | −0,0049 |
| `waist_yaw` | −0,0050 | `left_shoulder_pitch` | 0,2000 |
| `waist_roll` | ≈ 0 (4,7·10⁻⁶) | `left_shoulder_roll` | 0,2000 |
| `waist_pitch` | ≈ 0 (2,4·10⁻⁸) | `left_shoulder_yaw` | ≈ 0 |
| `left_elbow` | 0,6000 | `right_shoulder_pitch` | 0,2000 |
| `left_wrist_roll`, `left_wrist_pitch`, `left_wrist_yaw` | ≈ 0 | `right_shoulder_roll` | −0,2000 |
| `right_elbow` | 0,6000 | `right_shoulder_yaw` | ≈ 0 |
| `right_wrist_roll`, `right_wrist_pitch`, `right_wrist_yaw` | ≈ 0 | | |

A ordem das 29 juntas é a de `JUNTAS` (`knobs.py:31-41`). O `g1.xml` declara as juntas na mesma ordem (`asset_zoo/robots/unitree_g1/xmls/g1.xml`, elementos `<joint>`). O termo de ação resolve as juntas pela ordem natural do modelo (`entity/entity.py:527-549`), e o vetor de 29 ações segue essa ordem.

### 3.4 Ganhos PD e limite de torque por grupo

O atuador de cada grupo é um `BuiltinPositionActuatorCfg`. Há um por linha de `Contrato.pd` (`cena.py:176-199`). Os campos `kp` e `kd` vêm da linha; `armature` e `effort_limit` vêm do grupo de atuador do mjlab que contém a primeira junta da regex (`cena.py:184-186, 190-194`).

O atuador usa `gainprm[0] = kp`, `biasprm[1] = −kp`, `biasprm[2] = −kd` e `forcerange = ±effort_limit` (`utils/spec.py:311-313, 323-325`). Pela convenção de atuador afim do MuJoCo, a força vale `kp·(alvo − q) − kd·q̇`, limitada a ±`effort_limit`.

| Grupo (regex em `Contrato.pd`) | kp | kd | Limite de torque (N·m) | Origem do limite |
|---|---|---|---|---|
| `.*_hip_pitch_joint` | 100,0 | 2,0 | 88 | `g1_constants.py:96-99` |
| `.*_hip_yaw_joint` | 100,0 | 2,0 | 88 | `g1_constants.py:96-99` |
| `.*_hip_roll_joint` | 100,0 | 2,0 | 139 | `g1_constants.py:101-104` |
| `.*_knee_joint` | 150,0 | 4,0 | 139 | `g1_constants.py:101-104` |
| `.*_ankle_(pitch|roll)_joint` | 40,0 | 2,0 | 50 | `g1_constants.py:172-176` (2 × 25) |
| `waist_yaw_joint` | 200,0 | 5,0 | 88 | `g1_constants.py:96-99` |
| `waist_(roll|pitch)_joint` | 40,0 | 5,0 | 50 | `g1_constants.py:165-169` (2 × 25) |
| `.*_(shoulder_.*|elbow|wrist_roll)_joint` | 40,0 | 1,0 | 25 | `g1_constants.py:91-94` |
| `.*_wrist_(pitch|yaw)_joint` | 40,0 | 1,0 | 5 | `g1_constants.py:106-109` |

Os valores de kp e kd são os de `Contrato.pd` (`knobs.py:65-72`; linhas 66 a 71). O valor numérico de `armature` não é lido em nenhum arquivo da v3; ele vem das constantes do mjlab. Cada `armature` sai de `reflected_inertia_from_two_stage_planetary` (`mjlab/utils/actuator.py:25-34`): `I₁·(g₂·g₃)² + I₂·g₃² + I₃`, com os dados de rotor e engrenagem de `g1_constants.py:35-89`. Os valores calculados por essa fórmula, em kg·m²:

| Motor (grupos de `Contrato.pd`) | `armature` |
|---|---|
| 5020 (`shoulder`, `elbow`, `wrist_roll`) | 0,003610 |
| 7520_14 (`hip_pitch`, `hip_yaw`, `waist_yaw`) | 0,010178 |
| 7520_22 (`hip_roll`, `knee`) | 0,025102 |
| 4010 (`wrist_pitch`, `wrist_yaw`) | 0,004250 |
| 2 × 5020 (`ankle_pitch`, `ankle_roll`, `waist_roll`, `waist_pitch`) | 0,007219 |

A associação dos grupos de motor às juntas está em `g1_constants.py:125-178`.

Os mesmos limites aparecem na tabela `Recompensa.tau_max` (`knobs.py:286-292`) e em `TAU_MAX` (`knobs.py:572`). Essa tabela serve às recompensas e ao ruído de torque (seção 3.7), e não ao atuador.

### 3.5 Atraso da ação

O atuador atrasa o comando em passos de física (5 ms cada). Todos os 9 grupos usam os mesmos parâmetros de atraso (`cena.py:188-198`):

| Parâmetro | Valor | Origem |
|---|---|---|
| `delay_min_lag` | 0 | `cena.py:195` |
| `delay_max_lag` (capacidade) | `DR.atraso_refino[1]` × `Contrato.decimacao` = 3 × 4 = 12 passos de física | `knobs.py:513, 55`; `cena.py:188, 195` |
| `delay_update_period` | `NUNCA` = 10⁹ (nunca reamostra) | `knobs.py:47`; `cena.py:197` |
| `delay_per_env_phase` | `False` | `cena.py:197` |

Como os 9 grupos têm parâmetros iguais, o mjlab os funde num buffer de atraso só (`actuator/builtin_group.py:98-121, 158-173`). O buffer tem um lag por env, e esse lag vale para as 29 juntas ao mesmo tempo. A cada subpasso de física, o buffer recebe o alvo e devolve o alvo de `lag` subpassos atrás (`builtin_group.py:187-192`, `utils/buffers/delay_buffer.py:229-245`). A realimentação (posição e velocidade) nunca é atrasada.

O lag de cada env é sorteado na primeira chamada de `compute` depois do reset, porque `_step_count` volta a 0 no reset e `update_period` vale 10⁹ (`delay_buffer.py:194-209, 247-258`). O sorteio é um inteiro uniforme em [`min_lag`, `max_lag` corrente] (`delay_buffer.py:269-275`). O sorteio usa o gerador global do torch, porque o buffer não recebe gerador (`delay_buffer.py:269-277`; `builtin_group.py:161-169`), e não o `v3_gerador`. O lag fica fixo até o próximo reset. Nos primeiros subpassos, enquanto o histórico é curto, o buffer devolve o alvo mais antigo disponível (`delay_buffer.py:240-245`).

O `max_lag` corrente (o teto) não é a capacidade. Ele vale:

| Momento | Teto do atuador (subpassos) | Teto da observação (passos) | Referência |
|---|---|---|---|
| Início do env (evento `atraso`, modo `startup`) | `DR.atraso_inicio[1]` × 4 = 0 | `DR.atraso_inicio[1]` = 0 | `knobs.py:511`; `env_cfg.py:197-198`; `eventos.py:244-258` |
| A cada lote de resets, se o teto calculado difere de `DR.atraso_inicio[1]` | teto × 4 | teto | `curriculo.py:123-126` |

O teto calculado é `cur.dr()["atraso_max"]` (`balanceador.py:381-382`):

- se o degrau global ≥ `DR.degrau_atraso_refino` (3, `knobs.py:514`): `DR.atraso_refino[1]` = 3;
- senão: `ceil(DR.atraso_portao[1] × restrição − 10⁻⁹)`, com `DR.atraso_portao[1]` = 2 (`knobs.py:512`). Restrição 0 dá 0; restrição em (0; 0,5] dá 1; restrição em (0,5; 1] dá 2.

O novo teto vale para os buffers de todos os envs (`eventos.py:254-258`), mas só o env que reseta sorteia um lag novo com ele.

Logo o atuador sorteia o lag em {0, …, 4 × teto} subpassos, de 0 a 60 ms no teto máximo. O mesmo inteiro de teto vale, em passos de 0,02 s, para a observação (seção 4.4).

Sem o evento `atraso`, a capacidade do buffer fica em 12 (atuador) e 3 (observação); o evento só muda `max_lag` (`eventos.py:251-258`).

### 3.6 Viés do encoder

O evento `vies_encoder` (modo `startup`) roda `dr.encoder_bias` com `bias_range` = (−`DR.vies_encoder`; +`DR.vies_encoder`) = (−0,015; +0,015) rad (`knobs.py:510`; `env_cfg.py:191-192`). Sorteia uma vez por env e por junta, uniforme (`mjlab/envs/mdp/dr/joint.py:209-241`), e o valor fica fixo na vida do env. Ele entra em dois lugares: no alvo enviado (seção 3.1) e no termo `juntas` do ator (seção 4.2).

### 3.7 Perturbações do atuador (só no treino)

Estes eventos não existem no play (`env_cfg.py:169-181, 200-210`). Todos sorteiam uniforme com o gerador `v3_gerador` (`eventos.py:77-79, 516-521`).

| Evento | Quando | O que sorteia | Faixa | Referência |
|---|---|---|---|---|
| `pd` | reset do episódio | Fator de escala de kp e de kd, independentes, por env e por junta. | [`no_degrau(1; 0,9; g; 5)`; `no_degrau(1; 1,1; g; 5)`], com `DR.pd` = (0,9; 1,1) (`knobs.py:535`) e g = degrau global. Em g = 0 vale [1; 1]; em g = 5, [0,9; 1,1]. | `eventos.py:523-532`; `balanceador.py:368-369, 385`; `dr/actuator.py:70-108` |
| `forca_motor` (pernas e cintura) | reset do episódio | Fator de escala do limite de torque, por env e por junta. | [`no_degrau(1; 0,9; g; 5)`; `no_degrau(1; 1,1; g; 5)`] com `DR.forca_motor` = (0,9; 1,1) (`knobs.py:536`). | `eventos.py:534-546`; `balanceador.py:368-369, 386`; `dr/actuator.py:230-257` |
| `forca_motor` (braços: shoulder, elbow, wrist) | reset do episódio | Idem. | [`no_degrau(1; 0,8; g; 5)`; `no_degrau(1; 1,1; g; 5)`] com `DR.forca_motor_bracos` = (0,8; 1,1) (`knobs.py:537`). | `eventos.py:534-546`; `balanceador.py:368-369, 387`; `env_cfg.py:151-152` |
| `rfi` | a cada passo da política | Torque externo por junta e por env: `(2u − 1) · s · TAU_MAX[j]`, com u uniforme em [0; 1). Vale nos 4 subpassos. | `s = no_degrau(DR.rfi[0]; DR.rfi[1]; g; 5)` = de 0,05 a 0,10 (`knobs.py:534`) quando a trava `subida` está ligada (nível de PEGAR + nível de CARREGAR > 0); senão `s = 0`. | `eventos.py:605-615`; `balanceador.py:202, 384` |

A função `no_degrau(ini, fim, d, degraus)` vale `ini + (fim − ini) · d / degraus` (`balanceador.py:48-51`). Aqui `degraus` = `Balanceador.degraus` = 5 (`knobs.py:401`).

O `rfi` roda depois da física (etapa 9 da seção 2) e escreve o torque em `qfrc_applied` (`eventos.py:615`). Nenhum código lido zera esse campo entre subpassos, então o valor escrito vale nos 4 subpassos do passo seguinte. Com `s = 0`, o evento escreve zero.

## 4. Observações

### 4.1 Como o gerente de observação processa um termo

Cada termo passa por esta cadeia, nesta ordem (`mjlab/managers/observation_manager.py:17-21`, `330-362`):

1. a função do termo calcula o valor (`:330`);
2. o ruído soma um valor (`:331-332`), se o grupo tem corrupção ligada;
3. o clip limita (`:335-336`); a v3 não usa clip;
4. a escala multiplica (`:337-340`);
5. o atraso devolve o valor de `lag` passos atrás (`:348-351`);
6. o histórico guarda os últimos `history_length` valores e os achata (`:352-358`).

Os termos de um grupo são concatenados na ordem em que aparecem no dicionário (`:377-380`). O histórico achatado fica na ordem "termo a termo": primeiro os `H` quadros do termo 1, do mais antigo ao mais novo, depois os do termo 2, e assim por diante (`utils/buffers/circular_buffer.py:161-175`; `observation_manager.py:60-65`). Dentro de um termo, cada quadro ocupa uma fatia contígua das suas dimensões.

No reset de um env, os buffers de atraso e de histórico dele são zerados (`observation_manager.py:239-259`). No primeiro `append` depois do reset, o histórico inteiro é preenchido com o primeiro valor (`circular_buffer.py:210-215`). O valor lido do atraso nos primeiros passos é limitado ao que existe no buffer (`delay_buffer.py:240-243`).

A observação é calculada com `update_history=True` ao fim de cada passo e também no `reset` do env (`manager_based_rl_env.py:374, 464`).

O gerente descarta o ruído de um grupo que tem `enable_corruption=False` (`observation_manager.py:436-437`). O grupo `actor` usa `enable_corruption=True`, e o grupo `critic` usa `False` (`observacoes.py:419-422`).

### 4.2 Observação do ator

O ator recebe um vetor de 1 457 números quando o lidar está ligado. A conta é (3 + 3 + 3 + 2 + 29 + 29 + 29 + 6 + 11) × 5 + 882 = 575 + 882. `blocos_ator` (nome, canais por passo, escala, histórico) fixa os valores (`knobs.py:58-63`); `grupos` monta os termos (`observacoes.py:361-422`).

| # | Termo | Posição no vetor | Dim por quadro | Histórico | Escala | Ruído antes da escala | Atraso | O que mede |
|---|---|---|---|---|---|---|---|---|
| 1 | `giro_base` | 0–14 | 3 | 5 | 0,2 (`knobs.py:59`) | ±0,2 (`DR.ruido_obs`, `knobs.py:516`) | sim | Velocidade angular da pelve (rad/s) mais o viés fixo do episódio (seção 5.5). |
| 2 | `gravidade` | 15–29 | 3 | 5 | 1,0 (`knobs.py:59`) | ±0,05 (`knobs.py:516`) | sim | Vetor da gravidade no frame da base, girado pela inclinação fixa do IMU (seção 5.5). |
| 3 | `comando` | 30–44 | 3 | 5 | 1,0 (`knobs.py:60`) | nenhum | não | Comando de velocidade da base (vx, vy, wz) no frame da base. |
| 4 | `fase` | 45–54 | 2 | 5 | 1,0 (`knobs.py:60`) | nenhum | não | Seno e cosseno da fase do relógio de marcha. |
| 5 | `juntas` | 55–199 | 29 | 5 | 1,0 (`knobs.py:60`) | ±0,01 (`knobs.py:516`) | sim | Posição de cada junta mais o viés do encoder, menos `q_de_pe` (rad). |
| 6 | `vel_juntas` | 200–344 | 29 | 5 | 0,05 (`knobs.py:61`) | ±1,5 (`knobs.py:516`) | sim | Velocidade de cada junta (rad/s). |
| 7 | `acao` | 345–489 | 29 | 5 | 1,0 (`knobs.py:61`) | nenhum | não | A ação crua do passo que acabou de rodar. |
| 8 | `tarefa` | 490–519 | 6 | 5 | 1,0 (`knobs.py:62`) | nenhum | não | One-hot de `v3_tarefa` nos 6 nomes de `TAREFAS`. |
| 9 | `caixa` | 520–574 | 11 | 5 | 1,0 (`knobs.py:62`) | nenhum | não | A caixa estimada (8) e o alvo corrente (3), no frame de rumo (só o yaw) da base estimada (`estimador.py:191-194, 381-383`). |
| 10 | `mapa` | 575–1456 | 882 | 0 (sem histórico) | 1,0 (`knobs.py:62`) | nenhum | sim | Mapa de elevação do lidar (441 alturas e 441 confianças). |

A lista de termos e a ordem estão em `observacoes.py:389-400`. A ordem de concatenação é a ordem desta lista (`observacoes.py:403`).

Fonte de cada termo:

- `giro_base`: `builtin_sensor("robot/imu_ang_vel") + v3_vies_giro` (`observacoes.py:202-204`). O sensor é o giroscópio `imu_ang_vel` no site `imu_in_pelvis` (`g1.xml`, elemento `<gyro>`).
- `gravidade`: `v3_incl_imu · projected_gravity` (`observacoes.py:207-209`). O `projected_gravity_b` é a gravidade no frame do corpo raiz (`mjlab/envs/mdp/observations.py:38-43`).
- `comando`: `generated_commands(command_name="twist")` (`observacoes.py:392`). O mjlab devolve `vel_command_b`: (vx, vy, wz) no frame da base (`observations.py:91-94`, `tasks/velocity/mdp/velocity_command.py:58-59`). Cada reamostragem zera o comando (`comando.py:902`). Só dois casos recebem velocidade: a tarefa ANDAR na fatia ANDAR e a tarefa CARREGAR na fatia B (`comando.py:905`). As outras tarefas ficam com comando zero. A fatia é o grupo de tarefas que o env recebe no reset.
  - Quando o comando é sorteado: no reset do episódio, na abertura de cada trecho da cadeia (`comando.py:519`) e quando `time_left` chega a 0 (`command_manager.py:109-123`). `time_left` é sorteado em U(`Comando.reamostra_s`) = (3,0; 8,0) s (`knobs.py:137`; `env_cfg.py:123`). No CARREGAR da fatia B, `time_left` vai a infinito: o comando sorteado na abertura vale até o fim do trecho (`comando.py:940-941`).
  - ANDAR (`sorteia_velocidade`, `comando.py:125-147`), com `n` = nível do ANDAR (0 a 7, `balanceador.py:72`): `frente` = min(`Comando.andar_inicio` + `Comando.andar_passo`·n; `Comando.frente_tetos[-1]`) = min(0,3 + 0,1·n; 1,0) m/s. `tras` = min(0,3 + 0,1·n; `Comando.tras_teto_max`) = min(0,3 + 0,1·n; 0,5) m/s. `lado` = min(`Comando.andar_lado_inicio` + 0,1·n; `Comando.lado_teto_max`) = min(0,2 + 0,1·n; 0,3) m/s. `giro` = min(`Comando.andar_giro_inicio` + 0,1·n; `Comando.giro_tetos[-1]`) = min(0,3 + 0,1·n; 0,5) rad/s (`knobs.py:122-134`).
  - ANDAR no plano (tipo `andar`), forma do sorteio (`comando.py:136-146`): (vx, vy) é um ponto uniforme numa meia elipse, com raio √U e ângulo U(−π/2; +π/2). O semieixo x vale `frente` com probabilidade `frente`/(`frente`+`tras`) e −`tras` no resto. O semieixo y vale `lado`. wz = `giro`·U(−1; 1). Depois, |wz| < `Comando.giro_min` (0,1) vira 0, e ‖(vx, vy)‖ < `Comando.linear_min` (0,1) vira (0, 0) (`knobs.py:132-133`). Com probabilidade `Comando.andar_parado` (0,10), os três valores viram 0 (`knobs.py:135`). Na célula `terreno`, `frente` e `tras` ficam limitados a `Comando.terreno_v_max` = 1,0 (`comando.py:131-132`; `knobs.py:148`). T19 (27/11): no ANDAR de terreno o sorteio é só `vx` = `frente` · √U, com `vy` = 0 e `wz` = 0 (`comando.py:137-138`). O `wz` vem do rumo: o mjlab põe `wz` = erro de rumo (ganho 1,0), com teto de ±0,5 rad/s, e a v3 zera `|wz|` < `giro_min` (`comando.py:943-948`). O arquivo `04-comando-de-velocidade.md`, seção 4.5, descreve o rumo.
  - CARREGAR na fatia B, com `n` = nível do CARREGAR (0 a 4) (`comando.py:148-157`): `frente` = `Comando.frente_tetos[n]` = (0,4; 0,55; 0,7; 0,85; 1,0)[n] m/s; `tras` = min(`frente`; `Comando.tras_teto_max` 0,5); `lado` = min(`frente`; `Comando.lado_teto_max` 0,3); `giro` = `Comando.giro_tetos[min(n, 3)]` = (0,3; 0,4; 0,5; 0,5) rad/s (`knobs.py:122-125`). Os tipos de célula `reto`, `terreno` e `arco` sorteiam (vx, vy) por `_anda`: uma de 4 direções (+x, −x, +y, −y) com probabilidade 1/4, e módulo U(piso; teto). O piso vale `Comando.piso_velocidade` = 0,25 nas direções x e `Comando.piso_lado` = 0,2 nas direções y (`comando.py:102-110`; `knobs.py:126-127`). O tipo `terreno` limita `frente` e `tras` a 1,0. Os tipos `arco` e `giro_lugar` sorteiam wz = ±U(0,25; `giro`), com sinal de probabilidade 1/2 (`comando.py:113-115, 156-157`). O tipo `giro_lugar` deixa o linear em zero.
  - O tipo de célula é sorteado entre as células abertas (`comando.py:915-918`; `balanceador.py:306-316`). A regra das probabilidades dessas células fica fora deste documento.
- `fase`: `privilegiados(fase=True)` devolve `[sin(2π·f), cos(2π·f)]`, com `f = v3_fase_marcha` (`observacoes.py:312-314`). O relógio soma `step_dt` a `_fase_t` só quando `‖comando‖ > Comando.limiar_marcha` (0,1, `knobs.py:141`), e `f = (_fase_t / Comando.periodo_marcha) mod 1`, com `Comando.periodo_marcha` = 0,8 s (`knobs.py:140`; `comando.py:863, 872-873`). O relógio roda no `_update_metrics` do `twist`, no início do `compute` do gerente de comando, antes de o gerente reamostrar o comando (`command_manager.py:109-115`). O reset do env zera `_fase_t` (`comando.py:885`).
- `juntas`: `joint_pos_rel(biased=True)` = `joint_pos_biased − default_joint_pos` (`observacoes.py:394`; `observations.py:51-61`; `entity/data.py:369-371`).
- `vel_juntas`: `joint_vel_rel` = `joint_vel − default_joint_vel` (`observacoes.py:395`; `observations.py:64-72`).
- `acao`: `last_action` devolve `action_manager.action`, a ação crua do passo corrente, antes da escala (`observacoes.py:396`; `observations.py:80-83`; `action_manager.py:114-117, 169`).
- `tarefa`: `um_de_seis`, `one_hot(v3_tarefa, 6)` sem ruído (`observacoes.py:135-137`). Os canais 4 e 5 (REORIENTAR, LEVANTAR) nunca valem 1, porque nenhum trecho publica essas tarefas (`comando.py:175-179`).
- `caixa`: seção 5.3.
- `mapa`: seção 5.4.

### 4.3 Ruído do ator

O ruído do ator é uniforme e aditivo. `Unoise(n_min=−r, n_max=+r)` soma, a cada elemento, um sorteio de U(−r; +r) (`observacoes.py:380`; `utils/noise/noise_cfg.py:62-84`; operação padrão "add" em `noise_cfg.py:20`).

- Quando: a cada passo da política, para todos os 29 (ou 3) elementos, de forma independente.
- Onde: antes da escala, do atraso e do histórico (`observation_manager.py:331-340`). Cada quadro do histórico leva um sorteio próprio, porque o ruído entra antes do histórico. Os quadros preenchidos com o primeiro valor depois do reset repetem o mesmo sorteio (`circular_buffer.py:210-215`).
- Quais termos: só `giro_base`, `gravidade`, `juntas` e `vel_juntas`, com os raios de `DR.ruido_obs` = (("giro_base", 0,2), ("gravidade", 0,05), ("juntas", 0,01), ("vel_juntas", 1,5)) (`knobs.py:516`).
- Ruído efetivo depois da escala: `giro_base` ±0,04 (0,2 × 0,2), `gravidade` ±0,05, `juntas` ±0,01 rad, `vel_juntas` ±0,075 (1,5 × 0,05).

### 4.4 Atraso da observação

Cinco termos do ator têm atraso: `giro_base`, `gravidade`, `juntas`, `vel_juntas` e `mapa` (`_ATRASO`, `observacoes.py:55`). Os outros cinco não têm. O crítico não tem atraso em nenhum termo (`observacoes.py:373-387`).

Cada termo atrasado tem o seu `DelayBuffer`, com estes parâmetros (`observacoes.py:371, 382-386`):

| Parâmetro | Valor | Origem |
|---|---|---|
| `delay_min_lag` | 0 | `observacoes.py:383` |
| `delay_max_lag` (capacidade) | `DR.atraso_refino[1]` = 3 | `knobs.py:513`; `observacoes.py:371, 384` |
| `delay_update_period` | `NUNCA` = 10⁹ | `knobs.py:47`; `observacoes.py:385` |
| `delay_per_env_phase` | `False` | `observacoes.py:386` |
| `delay_per_env` | `True` (padrão do mjlab) | `observation_manager.py:42, 471` |

O lag é sorteado por termo e por env, na primeira chamada depois do reset, como inteiro uniforme em [0, teto corrente]. O lag fica fixo até o próximo reset (`delay_buffer.py:194-209, 247-258, 269-275`). Os cinco termos de um mesmo env sorteiam lags independentes. O teto corrente é o mesmo da seção 3.5 (0 no início do treino, 1 ou 2 conforme a restrição, 3 com degrau global ≥ 3). A unidade é o passo da política (0,02 s). O atraso entra antes do histórico: o histórico guarda os valores já atrasados (`observation_manager.py:348-358`).

### 4.5 Observação do crítico

O crítico recebe os mesmos 10 termos do ator, na mesma ordem e com o mesmo histórico e a mesma escala, mas sem ruído e sem atraso. Depois deles vêm 11 termos que só o crítico tem (`observacoes.py:389-418`).

Diferenças nos termos compartilhados:

- O termo `juntas` usa `joint_pos_rel` sem `biased`: o crítico vê a posição verdadeira, sem o viés do encoder (`observacoes.py:404`).
- O crítico recebe o mesmo `giro_base` e o mesmo `gravidade`. Eles incluem o viés fixo e a inclinação fixa do IMU, porque a função é a mesma.
- O crítico recebe o mesmo `mapa` e a mesma `caixa` do ator: o mapa do lidar ruidoso e a caixa estimada. O código não dá ao crítico o terreno verdadeiro.

Termos exclusivos do crítico, em ordem (`observacoes.py:405-418`):

| # | Termo | Posição | Dim | Histórico | Escala | Função e conteúdo |
|---|---|---|---|---|---|---|
| 11 | `vel_lin_base` | 1457–1471 | 3 | 5 (`hist_padrao`, `observacoes.py:407`) | 1,0 | `builtin_sensor("robot/imu_lin_vel")`, velocímetro `imu_lin_vel` no site `imu_in_pelvis` (`g1.xml`). |
| 12 | `caixa_verdade` | 1472–1478 | 7 | 0 | 1,0 | Posição (3) da caixa real no frame completo da base real (com inclinação), rotvec (3) de `conj(quat da base) · quat da caixa`, sem a redução pela simetria do cubo, e meia aresta real (1) (`observacoes.py:261-273`). Zero no ANDAR. |
| 13 | `alvo_verdade` | 1479–1481 | 3 | 0 | 1,0 | O alvo corrente `v3_alvo_w` no frame completo da base real. Zero se `v3_alvo_idx < 0` (`observacoes.py:261-264, 276-279`). |
| 14 | `alvo_macro` | 1482–1492 | 11 | 0 | 1,0 | One-hot do estado da cadeia (ver abaixo). |
| 15 | `referencia` | 1493–1500 | 8 | 0 | 1,0 | Referência do alvo: palmas (6), pelve (1), tronco (1) (`observacoes.py:116-130, 321-324`). |
| 16 | `niveis` | 1501–1504 | 4 | 0 | 1,0 | `nível / nível máximo` de ANDAR, PEGAR, CARREGAR e BOTAR; o mesmo valor em todos os envs (`observacoes.py:327-335`). |
| 17 | `degrau` | 1505 | 1 | 0 | 1,0 | `v3_degrau / 5` (`observacoes.py:338-340`). |
| 18 | `massa` | 1506 | 1 | 0 | 1,0 | `v3_massa / 5` (kg ÷ 5) (`observacoes.py:343-345`). |
| 19 | `forca_caixa` | 1507 | 1 | 0 | 1,0 | Soma dos módulos das 12 fatias de força de `caixa_robo`, ÷ 100 N (`observacoes.py:348-356`; `metricas.py:338-345`). |
| 20 | `privilegiados` | 1508–1511 | 4 | 0 | 1,0 | `[v3_marcha_on, v3_h, lacuna, z]` (`observacoes.py:315-318`). |
| 21 | `tarefa_critico` | 1512–1517 | 6 | 0 | 1,0 | One-hot de `v3_tarefa` (`um_de_seis`). Fica por último. |

O vetor do crítico tem 1 457 + 61 = 1 518 números com o lidar ligado. As posições são a soma das dimensões dos termos; o código só confere o tamanho da entrada da rede contra o normalizador (`runner.py:60-65`).

Os divisores `_ESC_DEGRAU` = 5,0, `_ESC_MASSA_KG` = 5,0 e `_ESC_FORCA_N` = 100,0 são constantes de `observacoes.py:60-62`. Elas não são knobs.

Conteúdo dos termos do crítico:

- `alvo_macro` (`observacoes.py:282-295`): 11 canais. Os canais 0 a 6 são o índice de `ALVOS` (P0 a B3). O canal 7 é "pega fechada", o 8 é "bota fechada", o 9 é "perda" e o 10 é "nenhum". O slot sai desta prioridade, da menor para a maior: o índice do alvo; −1 vira "nenhum" (10); o índice P3 com `v3_pega_fechada` vira 7; `v3_bota_fechada` vira 8; `v3_perdeu` vira 9. O último vencedor fica.
- `privilegiados` (`observacoes.py:315-318`): o canal 0 é `v3_marcha_on` (1 se `‖comando‖ > limiar_marcha`, avaliado no `_update_metrics` do `twist` antes da reamostragem do comando no mesmo passo, `comando.py:863, 874`). O canal 1 é `v3_h`. O canal 2 é a lacuna `relu(v3_phi_melhor − max(v3_phi, 0))`. O canal 3 é o `z` da etapa corrente em `v3_cur.z`, e 0 se não há alvo. Não determinado pelo código lido: como `Curriculo.z`, `v3_h` e `v3_phi_melhor` evoluem (código de `balanceador.py` e `recompensas.py` fora do que foi lido para este assunto).
- `referencia` (`observacoes.py:116-130, 321-324`): o termo chama `recompensas.ref_do_alvo` com o índice do alvo corrente. A altura `h` do topo da laje é `v3_topo_destino` nos alvos B1 e B2 e `v3_topo` nos outros (`recompensas.py:496-506`). As palmas-alvo vêm de `alvos.palmas_alvo_caixa`, no frame da caixa, com 0 nos alvos P3, B1 e B3 e fora de alvo (`alvos.py:138-154`). A pelve de referência vale `ref.pelve_z(h, a)` nos alvos P0, P1, P2, B1 e B2, `ref.pelve_com_caixa(a)` no P3, `ref.pelve_de_pe` no B3 e 0 nos demais. O tronco vale `ref.tronco(h, a)` nos mesmos alvos de IK e 0 nos demais. Não determinado pelo código lido: os valores numéricos de `Referencias.pelve_z`, `pelve_com_caixa` e `tronco` (classe `alvos.Referencias`, calculada a partir de `referencias.npz`).

### 4.6 O que o ator não vê e o crítico vê

| O que o crítico recebe | Onde |
|---|---|
| Velocidade linear da base (`vel_lin_base`). | seção 4.5, termo 11 |
| A caixa verdadeira, sem o erro do estimador e com a meia aresta real. | termo 12 |
| O alvo verdadeiro no frame da base real, sem a deriva da odometria. | termo 13 |
| O estado da cadeia (`alvo_macro`): etapa, pega fechada, bota fechada, perda. | termo 14 |
| A referência do alvo (palmas, pelve, tronco). | termo 15 |
| O nível de cada tarefa, o degrau do refino da tarefa corrente e a massa da caixa. | termos 16, 17, 18 |
| A força total de aperto na caixa. | termo 19 |
| O relógio de marcha ligado, o fator `h` da renda, a lacuna do Φ e o `z` da etapa. | termo 20 |
| A posição das juntas sem o viés do encoder. | termo 5 do crítico |
| Todos os termos compartilhados sem ruído e sem atraso. | seção 4.4 |

O ator e o crítico leem as mesmas funções para o mapa do lidar, a caixa estimada, o giro e a gravidade com o viés do IMU. O ator ainda soma ruído (giro, gravidade) e atraso (giro, gravidade, mapa) a esses termos, e o crítico não. Nenhum dos dois grupos contém o tempo restante do episódio, o valor do lag, o valor do viés do encoder ou o viés do IMU.

### 4.7 O normalizador da rede

Os dois modelos têm `obs_normalization=True` (`mjlab/tasks/velocity/config/g1/rl_cfg.py:16, 26`; `tarefa.py:45`). O normalizador empírico calcula `(x − média) / (desvio + 0,01)` por canal (`rsl_rl/modules/normalization.py:18, 48`). O 0,01 é o `eps` do `rsl_rl`.

Depois de cada `process_env_step`, o algoritmo da v3 corrige o normalizador (`algoritmo.py:114-134`, `fixa_normalizador` em `algoritmo.py:69-83`):

- impõe um piso no desvio: `PISO_DESVIO` = 0,05 (`algoritmo.py:66, 79`);
- fixa os termos listados abaixo em média 0 e desvio fixo (`algoritmo.py:80-83`; `runner.py:32-34, 46-66`).

| Termo | Desvio fixo |
|---|---|
| `tarefa`, `mapa`, `alvo_macro`, `niveis`, `degrau`, `massa`, `tarefa_critico`, `privilegiados` | 1,0 |
| `caixa`, `caixa_verdade`, `alvo_verdade`, `referencia` | 0,1 |
| `forca_caixa` | 0,3 |

Esses valores são constantes de `runner.py:32-34` (`_FIXOS`) e do piso em `algoritmo.py:66`. Eles não são knobs.

## 5. Estimador e percepção

O ator não recebe o estado verdadeiro da caixa nem do terreno. Ele recebe duas estimativas simuladas: a caixa estimada (termo `caixa`) e o mapa de elevação do lidar (termo `mapa`). Ambas dependem de uma odometria com deriva.

### 5.1 Quando o estimador roda

A guarda `atualiza_percepcao` roda o mapa e a caixa estimada uma vez por passo para todos os envs, mesmo que dois termos leiam cada um (`observacoes.py:140-151`). Ela compara `v3_percepcao_passo` com `common_step_counter`. O mapa só roda se o env tem o sensor `lidar` na cena (`observacoes.py:148-149`). A caixa estimada roda sempre (`observacoes.py:150`).

Os objetos de estado são criados uma vez em `garante_v3`: `v3_filtro` (mapa), `v3_padrao` (padrão dos raios), `v3_gerador` (gerador de números aleatórios, semeado com a seed do cfg ou 0), `v3_odo` (odometria) e `v3_estimador` (caixa) (`curriculo.py:52-68`). Todos os sorteios do estimador, da odometria e do lidar usam esse gerador.

O reset de cada env zera o filtro, a odometria e o estimador (evento `filtros`, o último evento de reset, `env_cfg.py:184`; `eventos.py:228-241`). Ele roda depois do evento `base`, e a odometria começa com a pose verdadeira lida do `qpos` (`eventos.py:228-241`).

### 5.2 Odometria

A odometria (`Odometria`, `estimador.py:129-188`) mantém uma pose estimada (`pos_w`, `quat_w`) e a verdadeira do passo anterior. A cada passo, no `compute` da cadeia (`comando.py:624-626`):

1. calcula o deslocamento verdadeiro no frame da base do passo anterior: `dxy`, `dz`, `dψ` (`estimador.py:159-165`);
2. soma ruído gaussiano, `z ~ N(0; 1)` com 4 componentes (`estimador.py:166-174`):
   - `s_xyz = sqrt(deriva_odo² · ‖Δp‖ + deriva_pos_giro² · |Δψ|)`, aplicado a `dxy` (2 eixos) e a `dz`;
   - `s_psi = sqrt(deriva_rumo² · ‖Δp‖ + deriva_rumo_giro² · |Δψ|)`, aplicado a `dψ`;
3. integra o deslocamento com a pose estimada (`estimador.py:175-181`). Roll e pitch da pose estimada são os verdadeiros, e só o yaw é o estimado (`estimador.py:180-181`).

As quatro derivas valem `no_degrau(ini, fim, g, 5)` no degrau global g (`comando.py:247-250`, chamado em `comando.py:623-625`):

| Deriva | Início (g = 0) | Refino (g = 5) | Knob |
|---|---|---|---|
| `deriva_odo` (m/√m) | 0,002 | 0,0075 | `Estimador.deriva_odo` (`knobs.py:473`) |
| `deriva_rumo` (rad/√m) | 0,002 | 0,005 | `Estimador.deriva_rumo` (`knobs.py:474`) |
| `deriva_pos_giro` (m/√rad) | 0,004 | 0,008 | `Estimador.deriva_pos_giro` (`knobs.py:475`) |
| `deriva_rumo_giro` (rad/√rad) | 0,0035 | 0,014 | `Estimador.deriva_rumo_giro` (`knobs.py:476`) |

### 5.3 A caixa estimada (termo `caixa`)

`_calcula_caixa` produz 11 números por env (`observacoes.py:172-199`): `saida_b` (8) e o alvo corrente no frame da base estimada (3).

**Visibilidade** (`visivel`, `estimador.py:111-126`; chamada em `observacoes.py:187-188`). A caixa tem 9 pontos: o centro e os 8 cantos, em fração da meia aresta real (`observacoes.py:49-50, 166-169`). A câmera é o site `d435` no torso (`observacoes.py:47`, `cena.py:159-167`). A caixa é visível se:

- o ponto do centro está dentro do campo de visão e da faixa de distância; e
- pelo menos `Estimador.pontos_visiveis_min` = 3 (`knobs.py:470`) dos 9 pontos estão no campo de visão, na faixa e sem oclusão.

Os limites são: metade do campo horizontal = `Estimador.fov_h_graus` / 2 = 69° / 2 (`knobs.py:455`), metade do vertical = `Estimador.fov_v_graus` / 2 = 42° / 2 (`knobs.py:456`) e distância em `Estimador.alcance` = (0,28; 3,0) m (`knobs.py:457`). A oclusão testa a distância do segmento câmera–ponto às 8 cápsulas dos braços; há oclusão se a distância fica menor que o raio da cápsula (`estimador.py:120-126`; `_BRACO`, `observacoes.py:71`). A câmera fica em `Estimador.d435_pos` (`knobs.py:477`) com pitch `Estimador.d435_pitch` (`knobs.py:478`).

**Leituras e entrega** (`EstimadorCaixa.passo`, `estimador.py:282-371`). O tempo é `t = common_step_counter × step_dt` (`observacoes.py:191`).

- Grade: o estimador gera uma amostra em cada instante `t0 + k / taxa_hz`, com `Estimador.taxa_hz` = 30 Hz (`knobs.py:451`) e `t0` o instante do primeiro passo depois do reset (`estimador.py:289-291, 320-338`). Cada amostra interpola linearmente a pose real entre o passo anterior e o atual, e a orientação por interpolação normalizada (`estimador.py:325-330`).
- Latência: a amostra só chega em `t_amostra + Estimador.latencia_s` = 0,033 s (`knobs.py:452`; `estimador.py:343`). O registro guarda a entrega mais recente (`estimador.py:344-352`). A fila tem `ceil(0,033 × 30) + 3` = 4 espaços (`estimador.py:217-218`).
- Validade de uma amostra: é válida se a caixa é visível no passo atual, se um sorteio U[0; 1) ≥ `perda` e se o instante da amostra ≥ `_rajada_ate` do env (`estimador.py:331`).
- Saída: é a última amostra entregue. Sem leitura nova, se a caixa está presa às palmas, ela anda com o ponto médio das palmas e gira com o rumo do eixo das palmas; solta, fica parada no mundo (`estimador.py:353-371`). A caixa está presa quando `‖caixa − meio das palmas‖ < meia aresta + Estimador.presa_margem` (0,05 m, `knobs.py:454`), avaliada a cada leitura nova (`estimador.py:364`).
- Conversão para a base: `saida_b` põe a pose no frame de rumo da base estimada e reduz a orientação pela simetria do cubo (`estimador.py:378-387`).

**Os 8 canais de `saida_b`** (`estimador.py:378-387`): posição no frame da base (3), rotvec (3), meia aresta estimada (1) e idade (1).

- O rotvec é o de menor ângulo entre as 24 rotações do cubo (`reduz_simetria_cubo`, `estimador.py:59-91`), do quaternion `conj(quat da base real) · quat estimada`.
- A idade é `t − instante da última leitura`, limitada a `Estimador.idade_max` = 2,0 s (`knobs.py:453`).
- Sem nenhuma leitura no episódio, os 7 primeiros canais valem 0 e a idade vale 2,0 (`estimador.py:380, 386`).

**Os 3 canais do alvo** (`observacoes.py:195-199`): `alvo_no_frame_da_base(v3_alvo_w, odo)` (`estimador.py:191-194`). O alvo vale 0 em três casos: tarefa CARREGAR; tarefa BOTAR com `v3_aviso` ligado; outra tarefa com `v3_alvo_idx < 0` (`observacoes.py:196-197`). Nos 11 canais, tudo vale 0 quando a tarefa é ANDAR (`observacoes.py:199`).

No BOTAR, o alvo `v3_alvo_w` nasce com um viés uniforme, sorteado na abertura do BOTAR: U(−v; +v) por eixo, com `v` em x e y = `no_degrau(0,01; 0,05; g; 5)` (`Estimador.vies_alvo_botar_xy`, `knobs.py:461`) e `v` em z = `no_degrau(0,005; 0,03; g; 5)` (`Estimador.vies_alvo_botar_z`, `knobs.py:462`) (`comando.py:551-557`). O PEGAR abre o P0 na primeira leitura válida da caixa: o env está em PEGAR sem alvo e `tem_leitura()` vale verdadeiro, isto é, existe uma entrega e `t − instante da entrega < Estimador.idade_max` = 2,0 s (`comando.py:364-392`; `estimador.py:373-376`; `knobs.py:453`).

**Sorteios por episódio** (no reset do env, `EstimadorCaixa.reinicia`, `estimador.py:262-280`):

| Sorteio | Distribuição | Escala |
|---|---|---|
| `_z_meia` | U[−1; 1], um por env | × `vies_meia`(g) (m), somado à meia aresta (`estimador.py:309`) |
| `_z_pos` | U[−1; 1], 3 eixos por env | × `vies_pos`(g) (m), somado à posição em cada leitura (`estimador.py:326-327`) |
| `_eixo_ori` | normal 3D normalizada: direção uniforme na esfera | eixo do viés de orientação (`estimador.py:277-278`) |
| `_z_ang` | N(0; 1) | ângulo do viés `θ = mediana · exp(σ_ln · z)`, com `σ_ln = ln(p90 / mediana) / Φ⁻¹(0,9)` (`estimador.py:14, 302-306`) |
| `_rajada_ate` | −∞ | nenhuma rajada aberta (`estimador.py:280`) |

**Sorteios por passo** (`estimador.py:282-338`):

| Sorteio | Distribuição | Escala |
|---|---|---|
| Ruído de posição da amostra | U[−1; 1] por eixo, por amostra | × (`sigma_pos`(g) + `sigma_pos_rel`(g) × distância câmera–caixa) (`estimador.py:297-298, 326`) |
| Ruído de orientação da leitura | rotvec U[−1; 1] por eixo, um por passo | × `ori_leitura_graus`(g) (`estimador.py:307`) |
| Perda da amostra | U[0; 1) por amostra | a amostra cai se o sorteio < `perda`(g) (`estimador.py:331`) |
| Abertura de rajada | U[0; 1) por env | abre se `_rajada_ate ≤ t` e o sorteio < `rajada_taxa`(g) × dt (`estimador.py:312-317`) |
| Duração da rajada | U[0; 1) por env | `rajada_s[0] + (rajada_s[1] − rajada_s[0]) × u` (`estimador.py:315-317`) |

A rajada só abre se `rajada_taxa`(g) > 0 e o gerador existe (`estimador.py:313`). Durante a rajada (`g < _rajada_ate`), toda amostra é inválida.

**Magnitudes por degrau global** (`_magnitude`, `observacoes.py:76-90`). Cada par (início, refino) vale `no_degrau(início; refino; g; 5)`, com g = `v3_cur.degrau_global()` (`observacoes.py:189`):

| Grandeza | Início (g = 0) | Refino (g = 5) | Knob |
|---|---|---|---|
| `sigma_pos` (m, absoluto) | 0,005 | 0 | `Estimador.sigma_pos[0]` (`knobs.py:471`) |
| `sigma_pos_rel` (fração da distância) | 0 | 0,02 | `Estimador.sigma_pos[1]` (`knobs.py:471`) |
| `vies_meia` (m) | 0,005 | 0,02 | `Estimador.vies_meia` (`knobs.py:459`) |
| `vies_pos` (m) | 0,0 | 0,02 | `Estimador.vies_pos` (`knobs.py:460`) |
| Mediana do ângulo do viés (graus) | 1,5 | 5,3 | `Estimador.ori_graus` (`knobs.py:465`) |
| p90 do ângulo do viés (graus) | 2,5 | 15,3 | `Estimador.ori_graus` (`knobs.py:465`) |
| `ori_leitura_graus` | 0,0 | 1,0 | `Estimador.ori_leitura_graus` (`knobs.py:466`) |
| `perda` (probabilidade por amostra) | 0,005 | 0,022 | `Estimador.perda` (`knobs.py:472`) |
| `rajada_taxa` (1/s) | 0,0 | 0,035 | `Estimador.rajada_taxa` (`knobs.py:469`) |

A duração da rajada, `Estimador.rajada_s` = (0,3; 1,0) s (`knobs.py:468`), não depende do degrau (`estimador.py:314-316`).

### 5.4 O lidar e o mapa (termo `mapa`)

O lidar está ligado. `Percepcao.ligada` = `True` (`knobs.py:483`). A task `Mjlab-G1-V3` usa `make_env_cfg()` com `ATIVO` (`tarefa.py:23, 58-64`). Com `ligada` verdadeiro, a cena ganha o sensor `lidar` (`cena.py:419-429`) e a observação inclui o `mapa` nos dois grupos (`observacoes.py:401-402`).

A task `Mjlab-G1-V3-SemPercepcao` usa `ligada=False` (`tarefa.py:26, 53-54, 66-72`). Nela o sensor sai da cena (`cena.py:419-420`), o `mapa` sai dos dois grupos, o vetor do ator tem 575 números e o do crítico, 636.

**O sensor** (`cena.py:423-428`): `RayCastSensorCfg` ancorado no site `mid360` do `torso_link` (posição `Percepcao.mid360_pos`, `knobs.py:501`; quaternion `Percepcao.mid360_quat`, `knobs.py:502`; site criado em `cena.py:165`). Os raios têm `ray_alignment="base"`, `max_distance` = `Percepcao.alcance[1]` = 6,0 m (`knobs.py:490`), grupos de geom (0, 1, 3) (terreno, mobília, almofadas) e `exclude_parent_body=True`.

**O padrão de raios** (`PadraoMid360`, `percepcao_lidar.py:18-29`): `dados/mid360_passos.npz` tem direções de forma (200, 205, 3). A cada passo, `direcoes(passo)` devolve o trecho `passo mod 200`, com `passo = common_step_counter`, e a observação copia `direcoes(passo + 1)` para o sensor, para o `sense` seguinte (`observacoes.py:222-223`). Há, portanto, 205 raios por passo. O knob `Percepcao.raios_por_passo` (205, `knobs.py:489`) não é lido por nenhuma instrução; o número de raios sai do arquivo.

**Pipeline de uma leitura** (`passo_de_percepcao`, `observacoes.py:100-113`; chamada em `observacoes.py:212-233`), uma vez por passo:

1. **Pontos** (`pontos_no_frame_do_mapa`, `percepcao_lidar.py:43-73`). Com o gerador ligado, cada raio recebe:
   - ruído de alcance N(0; `Percepcao.ruido_alcance`) = N(0; 0,02 m) (`knobs.py:491`);
   - ruído de direção N(0; `Percepcao.ruido_angulo_graus`) = N(0; 0,15°) por componente, com a direção renormalizada (`knobs.py:492`);
   - perda: o raio cai se U[0; 1) < `Percepcao.perda` = 0,01 (`knobs.py:493`);
   - uma inclinação por env e por passo: rotvec N(0; `Percepcao.ruido_incl_graus`) = N(0; 0,5°) por eixo, composta com a inclinação fixa do IMU `v3_incl_imu` (`knobs.py:494`; `percepcao_lidar.py:62-64`).

   Sem o gerador, nenhum ruído entra. Um raio vale se a distância `d` está em `Percepcao.alcance` = (0,1; 6,0) m (`knobs.py:490`; `percepcao_lidar.py:65`). Os pontos vão ao mundo, depois ao frame de rumo da base (`percepcao_lidar.py:66-72`). A altura `z` do ponto é medida a partir da sola do pé mais baixo (`z_pe_min`, `observacoes.py:236-243`).
2. **Filtro do corpo** (`filtra_corpo`, `percepcao_lidar.py:76-89`). Remove o ponto a menos de `raio + Percepcao.margem_corpo` de uma das 33 cápsulas do corpo, com `margem_corpo` = 0,03 m (`knobs.py:500`), e o ponto a menos de `Percepcao.raio_min` = 0,85 m do sensor (`knobs.py:488`). As cápsulas vêm de `dados/capsulas.npz` (`observacoes.py:66-73`).
3. **Transporte** (`FiltroDeElevacao.prediz`, `percepcao_lidar.py:141-161`). Move o mapa interno pelo deslocamento da odometria, só em células inteiras; o resto fica em `o`. A altura `h` das células vistas desce pela variação de `z_ref`. A variância `P` cresce em `Percepcao.q_desloc` × deslocamento + `Percepcao.q_giro` × |giro| + `Percepcao.q_tempo` × dt, com `q_desloc` = 4·10⁻⁴, `q_giro` = 4·10⁻⁴ e `q_tempo` = 2,5·10⁻⁵ (`knobs.py:496-498`). A célula que sai do mapa volta a "nunca vista".
4. **Atualização de Kalman por célula** (`atualiza`, `percepcao_lidar.py:163-190`). Cada célula usa a média dos `z` dos pontos que caem nela. Com `n` pontos e distância média `d̄`: `R = (rad(ruido_incl_graus)·d̄)² + (ruido_alcance² + (d̄·rad(ruido_angulo_graus))²) / n`. A célula nunca vista recebe `h = z` e `P = R`. A célula já vista recebe o ganho de Kalman `P / (P + R)`. Depois `P ≥ Percepcao.sigma_fixo²`, com `sigma_fixo` = 0,02 m (`knobs.py:499`).
5. **Recorte** (`recorte`, `percepcao_lidar.py:192-200`). Devolve 882 números: 441 alturas e depois 441 confianças.

**Geometria do mapa**: o mapa interno tem `Percepcao.n_interno` × `Percepcao.n_interno` = 51 × 51 células (`knobs.py:486`), cada uma com `Percepcao.celula` = 0,10 m (`knobs.py:484`), centrado na base (`percepcao_lidar.py:99-117`). O recorte tem `Percepcao.n_recorte` × `Percepcao.n_recorte` = 21 × 21 células (`knobs.py:485`), de −1,0 m a +1,0 m em x e em y.

**Ordem dentro do recorte** (`percepcao_lidar.py:114-117, 192-200`): a linha vai de x = +1,0 m a x = −1,0 m (à frente primeiro); dentro da linha, y vai de +1,0 m a −1,0 m. O vetor de 882 números tem as 441 alturas nos índices 0 a 440 e as 441 confianças nos índices 441 a 881.

**Valores**: a altura é o `h` da célula, limitado a ±`Percepcao.altura_clip` = ±1,0 m (`knobs.py:487`). A confiança é `σ_ref² / (σ_ref² + P)`, com `Percepcao.sigma_ref` = 0,05 m (`knobs.py:495`). A célula nunca vista vale 0 na altura e 0 na confiança (`percepcao_lidar.py:197-199`).

O ator recebe o mapa com atraso (`mapa` está em `_ATRASO`, seção 4.4) e sem ruído adicional do gerente de observação. O crítico recebe o mesmo mapa, sem atraso. O knob `Percepcao.limiar_visto` (0,5, `knobs.py:503`) não entra na observação. Ele alimenta a função de métrica `fracao_mapa_visto` (`metricas.py:555`) e o exportador de cena (`exporta_cena.py:582`).

O mapa interno não gira. O recorte lê o mapa nos eixos do rumo da base: `_no_mapa` gira os centros das 441 células do recorte pelo rumo `psi` acumulado e soma o resto `o` (`percepcao_lidar.py:127-130, 194`).

Um raio sem acerto, ou com distância acima de `max_distance`, volta do sensor com `distances` = −1,0 (`mjlab/sensor/raycast_sensor.py:784`). Como `d` fica abaixo de `Percepcao.alcance[0]` = 0,1 m, a condição de validade `d ∈ [0,1; 6,0]` descarta o raio (`percepcao_lidar.py:65`). O ruído de alcance é somado antes dessa condição.

### 5.5 Viés do IMU

O evento `vies_imu` (modo `reset`) sorteia, por env e a cada reset (`env_cfg.py:166`; `eventos.py:492-509`):

- se a trava `vies` do balanceador está ligada (`cur.dr()["vies_imu"]`): o giro fixo `v3_vies_giro`, U(−`DR.vies_imu_giro`; +`DR.vies_imu_giro`) = U(−0,05; +0,05) rad/s por eixo (`knobs.py:521`); e a inclinação fixa `v3_incl_imu`, uma rotação com ângulos U(−`DR.vies_imu_incl_graus`; +`DR.vies_imu_incl_graus`) = U(−1°; +1°) em torno de x e de y, e 0 em z (`knobs.py:522`; `eventos.py:500-505`);
- se a trava está desligada: giro zero e inclinação identidade (`eventos.py:506-507`).

A trava `vies` liga quando a nota da célula `andar` do ANDAR chega ao portão da célula (`balanceador.py:122-123, 201`). O portão da célula `andar` é `Balanceador.portao_andar` = 0,75 (`knobs.py:392`; `balanceador.py:97-98`).

O giro entra em `giro_base` (somado ao sensor). A inclinação entra em `gravidade` (multiplica o vetor) e na leitura do lidar (`percepcao_lidar.py:56, 62-64`).

### 5.6 Resumo do que cada parte põe na observação

| Parte | Entra em | Ligada? |
|---|---|---|
| Viés do encoder (startup, seção 3.6) | `juntas` do ator | sempre |
| Viés e inclinação do IMU (reset, seção 5.5) | `giro_base`, `gravidade` dos dois grupos; lidar | depende da trava `vies` |
| Estimador da caixa (seção 5.3) | `caixa` dos dois grupos | sempre |
| Odometria (seção 5.2) | frame do alvo e da caixa; transporte do mapa | sempre |
| Lidar e mapa (seção 5.4) | `mapa` dos dois grupos | `Percepcao.ligada` = `True` |

## 6. Terminações

### 6.1 Como funcionam

Cada terminação é uma função que devolve um booleano por env. O gerente de terminação avalia as cinco a cada passo (`termination_manager.py:102-113`). Se a terminação tem `time_out=True`, o resultado entra em `time_outs` (truncamento). Se `time_out=False`, entra em `terminated` (falha) (`termination_manager.py:108-111`). O env faz reset no mesmo passo se qualquer terminação vale (`manager_based_rl_env.py:436, 444-448`).

O `cfg.terminations` tem estas cinco, nesta ordem (`env_cfg.py:137-147`):

| Nome | Função | `time_out` | Tarefas (`mascara`) | Referência |
|---|---|---|---|---|
| `time_out` | `terminacoes.fim_da_cadeia` | `True` | todos os envs | `env_cfg.py:138` |
| `queda` | `terminacoes.queda` | `False` | ANDAR, PEGAR, CARREGAR, BOTAR | `env_cfg.py:140` |
| `caixa_perdida` | `terminacoes.caixa_perdida` | `False` | PEGAR, CARREGAR, BOTAR | `env_cfg.py:141-143` |
| `chegada` | `terminacoes.chegada` | `False` | BOTAR | `env_cfg.py:144` |
| `batente` | `terminacoes.batente` | `False` | ANDAR, PEGAR, CARREGAR, BOTAR | `env_cfg.py:146` |

A máscara por tarefa usa `torch.where(mascara(env, tarefas), valor, falso)` e não multiplicação, para um NaN de um env fora da tarefa não vazar (`terminacoes.py:111-112, 124-125, 150-151, 182-183`). `mascara` devolve `True` onde `v3_tarefa` está em `tarefas` (`curriculo.py:93-100`).

O gerente registra, no reset, quantos envs terminaram por cada termo (`Episode_Termination/<nome>`, `termination_manager.py:93-97`).

### 6.2 `time_out`: fim da cadeia

Condição, por env (`terminacoes.py:53-72`):

```
episode_length_buf >= max_episode_length   OU   v3_fim   OU   atravessou
atravessou = (fatia == ANDAR)  E  (terrain_types != 0)  E  (d > Terreno.dist_travessia)  E  (episode_length_buf > 2)
d          = ‖ twist.pose_ult[:2] − scene.env_origins[:2] ‖
```

- `max_episode_length` = `ceil(episode_length_s / step_dt)` (`manager_based_rl_env.py:281-283`). Com `Cadeia.episodio_s` = 20,0 s (`knobs.py:160`; `env_cfg.py:105`) e `step_dt` = 0,02 s, o valor é 1 000 passos.
- `v3_fim` liga quando o último trecho da cadeia acaba (`comando.py:608-614`; `avanca_trecho`, `comando.py:201-216`). O último trecho da fatia ANDAR dura `Cadeia.episodio_s` = 20,0 s. O último da fatia B é o CARREGAR, com duração U(`Cadeia.carregar_b_s`) = (4,0; 7,0) s (`knobs.py:155`). O último da fatia C é o ANDAR final, com `Cadeia.andar_final_s` = 1,5 s (`knobs.py:157`; `comando.py:175-179`). A fatia é o grupo de tarefas que o env recebe no reset.
- A cadeia roda depois das terminações (seção 2). O `v3_fim` ligado no passo N faz o `time_out` valer no passo N+1.
- Depois de uma perda de caixa com `Cadeia.caixa_perdida_vira_andar` = `True`, o trecho fica congelado e o `v3_fim` não liga (`comando.py:610-614`). O episódio acaba só pelo tempo máximo.
- O reset do env limpa `v3_fim` (`comando.py:651-653`).
- Não depende do degrau do refino e não é mascarada por tarefa.
- É fim de tempo: entra em `time_outs`.

**A travessia (`atravessou`, T19, 27/11).** O episódio de ANDAR em terreno acaba quando a base passa de `Terreno.dist_travessia` = 3,5 m da origem da célula (`terminacoes.py:69-72`; `knobs.py:442`). A origem e a coluna do terreno vêm do `sorteia_fatia` (`eventos.py:161-175`).
- Quando vale: só na fatia ANDAR (`FATIAS.index("ANDAR")` = 0, `knobs.py:43`) com `terrain_types` ≠ 0, isto é, nas colunas 1 a 8. O plano (coluna 0) e as fatias B e C nunca terminam por travessia. Não depende do degrau do refino.
- Distância: norma da diferença (x, y) entre a pose da base e a origem da célula. A região é um círculo de raio 3,5 m, e não o retângulo do mjlab. A docstring de `fim_da_cadeia` faz a mesma comparação (`terminacoes.py:61-64`): a função `terrain_edge_reached` do mjlab usa `|dx| > 3,8 m` ou `|dy| > 3,8 m`, com 8,0 / 2 × 0,95 = 3,8 (`mjlab/tasks/velocity/mdp/terminations.py:94-100`). Aqui a distância é a norma, e o limite é `dist_travessia`.
- Guarda de passos: `episode_length_buf > 2`. O término não dispara nos 2 primeiros passos do episódio. É a guarda do `terrain_edge_reached` (`mjlab/tasks/velocity/mdp/terminations.py:100`), contra posição velha logo depois do reset.
- É `time_out`: entra em `time_outs`, não em `terminated`. A travessia não paga o termo `terminacao` (seção 6.7), e o PPO soma o valor do futuro (seção 6.8). O segmento de comando fecha sem `falha` (`comando.py:882-883`).

**Qual posição a travessia lê.** A condição lê `twist.pose_ult`, e não `root_link_pos_w`. O `pose_ult` é a pose (x, y, rumo) que o `twist` grava no começo de `_update_metrics` (`comando.py:732`, `comando.py:814-815`). Esse método roda no `compute` do gerente de comandos, etapa 8 da seção 2. Na etapa 4 do passo `N`, o `pose_ult` ainda guarda a pose do fim do passo `N − 1`. O término lê, portanto, uma posição um passo (0,02 s) mais velha que a física do passo `N` (derivado: a ordem da seção 2).

**A ordem de leitura da posição, no passo e no reset.** O término e o registro leem o mesmo buffer, antes de alguém o reescrever:

| # | Etapa do passo `N` (seção 2) | O que lê ou escreve | Referência |
|---|---|---|---|
| 4 | Terminações | `fim_da_cadeia` lê `pose_ult` (pose do fim do passo `N − 1`) e `env_origins` (a origem do episódio que acaba). Se `d` > 3,5 m, o `time_out` vale. | `terminacoes.py:69-72` |
| 6 | Reset dos envs terminados: o evento `fatia`, o primeiro dos eventos de reset | `sorteia_fatia` lê o mesmo `pose_ult`, a mesma origem, `soma_cmd_marcha` e `terminated`, e conta a visita e a travessia. Depois sobrescreve a fatia, o terreno e `env_origins` (`eventos.py:143-175`). | `eventos.py:143-160`, `env_cfg.py:154` |
| 6 | Reset do `twist` | Zera `soma_cmd_marcha`, depois do `sorteia_fatia` (seção 6.7, itens 3 e 6). | `comando.py:889-892` |
| 8 | `compute` dos comandos | `_update_metrics` grava `pose_ult` com a pose de agora. Nos envs que acabaram de resetar, é a pose nova, depois do evento `base`. | `comando.py:814-815` |

Consequência: a travessia que termina o episódio no passo `N` conta também no registro do mesmo reset, porque os dois leem o `pose_ult` do passo `N − 1`. O comentário do código dá a razão do desenho: com a leitura direta de `root_link_pos_w`, o término dispararia no passo em que o robô passa de 3,5 m, mas o `pose_ult` do registro ainda estaria a ≤ 3,5 m, e a travessia não contaria (`terminacoes.py:65-67`). Com o `pose_ult`, o término sai um passo depois.

**O que conta visita e travessia.** O `sorteia_fatia` registra o episódio de ANDAR em terreno que acabou, antes de escrever a fatia nova (`eventos.py:143-160`). Para cada env que reseta:
- `pedido` = `soma_cmd_marcha × step_dt`. A soma acumula `‖cmd_xy‖` nos passos com `‖cmd_xy‖` > `limiar_canal`, e é a distância que o comando pediu no episódio (`comando.py:847`; `eventos.py:155`).
- Visita: fatia anterior ANDAR, coluna anterior ≠ 0, `episode_length_buf` > 0 (o episódio teve passos) e (`pedido` ≥ `dist_travessia`, ou o episódio acabou em falha, ou `d` > `dist_travessia`) (`eventos.py:156-157`). "Falha" é `terminated`, que não inclui o `time_out`. No ANDAR, só `queda` e `batente` entram.
- O primeiro reset (na construção do env) não conta visita, por causa da guarda de passos. O `terrain_types` não exclui esse reset: o mjlab dá a cada env uma coluna e uma origem na construção, e as colunas cobrem de 0 a 8 (`mjlab/terrains/terrain_entity.py:346-364`). O `v3_fatia` nasce 0 (ANDAR) e o `pose_ult` nasce 0 (`comando.py:732`), então `d` seria a distância da origem da célula à origem do mundo. O `episode_length_buf` nasce 0 e só volta a 0 no fim do `_reset_idx`, depois dos eventos (`manager_based_rl_env.py:232`, `manager_based_rl_env.py:590`). Nesse reset ele vale 0, e a guarda exclui todos os envs (`eventos.py:147-149`).
- Travessia: visita com `d` > `dist_travessia` (`eventos.py:158`).
- Um episódio de terreno que acaba no tempo máximo, com `pedido` < `dist_travessia` e `d` ≤ `dist_travessia`, não é visita. Um episódio que acaba em queda com `d` ≤ `dist_travessia` é visita sem travessia.
- O registro soma em `vis_ter` e `trav_ter` do currículo, por coluna (`eventos.py:158-160`). O `fecha_iteracao` move a linha da coluna com a taxa `trav_ter / vis_ter` (`balanceador.py:178-182`; arquivo `04-comando-de-velocidade.md`, seção 4.2).

### 6.3 `queda`

Condição, por env, com a tarefa do env (`terminacoes.py:35-50`, `86-112`):

- ANDAR e CARREGAR: `inclinação > Terminacao.incl_marcha` OU `(z_pelve − z_chão_pelve) < Terminacao.base_min`.
- PEGAR e BOTAR: `inclinação > radianos(Terminacao.incl_manip_graus)` OU `(z_joelho − z_chão_joelho) < Terminacao.joelho_min`.

| Limiar | Valor | Knob |
|---|---|---|
| `incl_marcha` | 0,8 rad | `Terminacao.incl_marcha` (`knobs.py:375`) |
| `base_min` | 0,20 m | `Terminacao.base_min` (`knobs.py:376`) |
| `incl_manip_graus` | 70° | `Terminacao.incl_manip_graus` (`knobs.py:377`) |
| `joelho_min` | 0,05 m | `Terminacao.joelho_min` (`knobs.py:378`) |

Como cada variável é medida (`terminacoes.py:96-107`):

- `inclinação` = `acos(clamp(−projected_gravity_b[z], −1; 1))`, o ângulo do eixo z da base com a vertical, em rad (`terminacoes.py:97`). A leitura é a do estado corrente.
- `z_pelve` é o z do frame da pelve (`frame_pos_w`) do sensor `altura_pelve` (`terminacoes.py:99-100`). O sensor lança 4 raios num anel de raio 0,05 m para baixo, só contra o terreno (grupo 0), e devolve a menor folga entre o frame e o terreno (`cena.py:394`; `sensor/terrain_height_sensor.py:34-37, 64, 83-84`). `z_chão_pelve` = `z_pelve − folga` (`terminacoes.py:101`), ou seja, o ponto mais alto do terreno sob o anel. A altura é, assim, medida contra o terreno e não contra a origem do mundo. Um raio sem acerto vale `max_distance` = 10,0 m (`mjlab/sensor/raycast_sensor.py:404`). Se todos os raios do frame erram, a folga vale o z do frame limitado a [0; 10] m. Um acerto pela face de baixo da geometria dá folga 0 (`terrain_height_sensor.py:69-80`).
- O joelho: o sensor `altura_joelhos` tem dois frames (os joelhos) com 4 raios num anel de raio 0,03 m cada (`cena.py:395-396`). O código escolhe o joelho de menor folga (`argmin`) e usa o z dele e o z de chão dele (`terminacoes.py:103-107`).

Os 4 raios, a seleção do joelho e o uso do terreno só valem para os sensores de altura, que são lidos do `sense` do passo anterior (seção 2).

REORIENTAR e LEVANTAR não têm ramo e dão `False` (`terminacoes.py:44`), mas essas tarefas não aparecem na cadeia. Não depende do degrau do refino. É falha: entra em `terminated`.

### 6.4 `caixa_perdida`

Condição (`terminacoes.py:75-83, 115-125`):

```
v3_perdeu_agora  E  (não Cadeia.caixa_perdida_vira_andar)
```

Com `Cadeia.caixa_perdida_vira_andar` = `True` (`knobs.py:161`; passado por `params`, `env_cfg.py:141-143`), a terminação vale sempre `False`. Nesse caso a perda não termina o episódio. A cadeia reage assim, no passo em que detecta a perda (`comando.py:412-420`):

- marca `v3_perdeu`;
- muda o alvo para B3;
- reinicia `v3_phi` (−1,0), `v3_h` (1,0), `v3_phi_melhor` (0,0), `v3_relogio_alvo` (0,0) e `v3_sigma` (o σ do alvo B3);
- publica a tarefa ANDAR para o env.

Com a chave em `False`, a terminação vale `True` no passo seguinte ao da detecção, porque a cadeia roda depois das terminações (`terminacoes.py:75-84`; seção 2).

O pulso `v3_perdeu_agora` é escrito pela cadeia como (`comando.py:59-65, 406-411`):

```
(caiu | arremesso | fora_da_laje)  E  (tarefa != ANDAR)  E  (episode_length_buf > 0)
```

| Parte | Condição | Limiar e knob |
|---|---|---|
| `caiu` | `altura_caixa − v3_meia < Terminacao.caixa_folga_chao` | 0,02 m (`knobs.py:379`) |
| `arremesso` | `v3_pegou` E `‖v_caixa − v_base‖ > Terminacao.v_solta` E fora do alvo de soltura | 1,2 m/s (`knobs.py:380`) |
| `fora_da_laje` | alvo entre P0 e P2 E o centro da caixa fora da laje | fora = `|x|` ou `|y|` no frame da laje > `Cena.prateleira_meia_xy` = 0,30 m (`knobs.py:89`; `comando.py:334-335`) |

- `altura_caixa` é a folga mínima do frame do corpo `box` contra o terreno, 4 raios num anel de 0,05 m (`cena.py:397`; `comando.py:336`).
- "Fora do alvo de soltura" é: `‖caixa − v3_alvo_w‖ > Alvos.raio_b1` = 0,10 m (`knobs.py:191`) no BOTAR; verdadeiro nas outras tarefas (`comando.py:407-408`).
- A condição `episode_length_buf > 0` (`comando.py:410`) evita uma perda falsa no primeiro passo do episódio, quando `altura_caixa` ainda tem o valor do fim do episódio anterior.
- O sensor `altura_caixa` mede só contra o terreno (grupo de geoms 0, `cena.py:330-340`). Ele não mede contra a laje, que está no grupo de geoms 1 (`Cena.grupo_mobilia` = 1, `knobs.py:88`; `cena.py:118-128`).

Os sensores de altura da caixa só atualizam no `sense`. Com o ramo `caiu`, o pulso sai um passo depois do que a física indica (`comando.py:394-400`).

Com a chave em `True` ou `False`, o passo seguinte à detecção paga o termo de recompensa `perda_caixa`, em todas as tarefas (`env_cfg.py:53`). Com a chave em `False`, o passo paga também o termo `terminacao` (seção 6.7). O valor é `v3_perdeu_agora × Recompensa.perda_caixa` = 240,0 (`knobs.py:367`), com o peso fixo −1,0 (`env_cfg.py:57`) e o fator `step_dt`: −240,0 × 0,02 = −4,8 (`recompensas.py:311-313, 960-962`). A perda alimenta ainda uma correção da recompensa no algoritmo (seção 6.8). Não depende do degrau do refino. Não é fim de tempo.

### 6.5 `chegada`

Condição, só no BOTAR (`terminacoes.py:128-151`):

```
primeiro_contato(apoio_caixa)  E  ( −v3_vz_caixa_ant  >  limite_do_degrau )
```

- `apoio_caixa` é o sensor de contato entre o geom da caixa e o geom da laje, com `track_air_time=True` (`cena.py:361-367`). `compute_first_contact(dt=step_dt)` devolve verdadeiro no passo em que o contato nasce, e o código lê a coluna 0 (`terminacoes.py:148`).
- `v3_vz_caixa_ant` é a velocidade vertical da caixa no fim do passo anterior, escrita pela cadeia (`comando.py:629`). É negativa na descida; `−vz > limite` mede uma descida rápida.
- `limite_do_degrau` = `por_degrau(env, 1,0; 0,25)` = `no_degrau(Terminacao.chegada[0]; Terminacao.chegada[1]; v3_degrau; 5)` (`terminacoes.py:149`; `recompensas.py:447-450`; `Terminacao.chegada` = (1,0; 0,25) m/s, `knobs.py:381`).

O `v3_degrau` é o degrau da tarefa do env, gravado quando o trecho abre (`comando.py:508-518`). Valores do limite:

| Degrau | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Limite (m/s) | 1,00 | 0,85 | 0,70 | 0,55 | 0,40 | 0,25 |

Depende do degrau do refino. Não é fim de tempo.

### 6.6 `batente`

Condição (`terminacoes.py:154-183`):

```
( alguma das 29 juntas passa do batente   OU   |hip_yaw| passa do limite )   E   ( v3_degrau >= Refino.degrau_batente )
```

- `fora`: `|q − centro| / meio > Terminacao.batente_frac` para qualquer das 29 juntas. `q` é `joint_pos` (sem viés do encoder). `centro` = (limite inferior + limite superior) / 2 e `meio` = (superior − inferior) / 2, dos limites duros do modelo, `joint_pos_limits` (`terminacoes.py:177-180`; `entity/entity.py:706-710`). `batente_frac` = 1,0 (`knobs.py:383`). A comparação é estrita (`>`).
- `yaw`: `|q| > Terminacao.hip_yaw_max` em `left_hip_yaw_joint` ou `right_hip_yaw_joint`, com `hip_yaw_max` = 1,0 rad (`knobs.py:382`; `terminacoes.py:32, 163`).
- Os dois gatilhos são alternativos (OU) (`terminacoes.py:164`).
- `Refino.degrau_batente` = 1 (`knobs.py:547`). Com o degrau 0, a terminação vale `False`. `v3_degrau` é o degrau da tarefa do env (`terminacoes.py:181`).

Depende do degrau do refino. É falha: entra em `terminated`.

### 6.7 O que acontece depois de uma terminação

Se qualquer terminação vale no passo, `reset_buf` vale verdadeiro para o env e o env sofre reset no mesmo passo (`manager_based_rl_env.py:436-448`). O reset roda nesta ordem (`manager_based_rl_env.py:553-591`):

1. o termo de currículo `v3` (inclui o teto do atraso);
2. o reset do sim e da cena (zera os buffers de atraso do atuador, `entity.py:832-841`);
3. os eventos de reset, na ordem do `cfg.events` (`env_cfg.py:153-211`): `fatia`, `cena`, `base`, `juntas`, `carga`, `atrito_pad`, `atrito_laje`, `vies_imu`, depois `pd`, `forca_motor`, `solref` (só no treino), `com_caixa` (só com a chave ligada), `filtros`;
4. o gerente de observação (zera atraso e histórico);
5. o gerente de ação (zera as ações);
6. as recompensas, as métricas, o currículo, os comandos, os eventos e as terminações;
7. `episode_length_buf` volta a 0.

**O termo de recompensa `terminacao`** vale em todas as tarefas (`env_cfg.py:53`). A função `n_terminacao` devolve `(terminado & ~tempo_esgotado).float()` (`recompensas.py:214-216`, `965-968`). O peso é `Recompensa.terminacao` = −300,0 (`knobs.py:366`). O mjlab multiplica a recompensa por `step_dt` (`scale_rewards_by_dt` = `True`, `manager_based_rl_env.py:157`; `reward_manager.py:118, 127`). A terminação paga, então, −300,0 × 0,02 = −6,0 no passo em que ela dispara.

Nenhuma terminação por `time_out` recebe essa cobrança. Se um env cai no mesmo passo em que o tempo acaba, `time_outs` vale verdadeiro e `terminado & ~tempo_esgotado` vale falso, e o passo não paga a queda.

### 6.8 O que o algoritmo faz com cada tipo de fim

- Fim por `time_out`: o env tem `extras["time_outs"]` verdadeiro, porque `is_finite_horizon` vale `False` (`manager_based_rl_env.py:134`; `rl/vecenv_wrapper.py:82-83`). O PPO soma `gamma × V(s)` à recompensa desse passo, o que mantém o valor do futuro (`rsl_rl/algorithms/ppo.py:150-155`). Com `gamma` = 0,99 (`mjlab/tasks/velocity/config/g1/rl_cfg.py:37`). O fim da cadeia (`v3_fim`) também é um `time_out` e recebe esse tratamento.
- Fim por falha (`queda`, `caixa_perdida`, `chegada`, `batente`): o PPO não soma nenhum valor futuro. A v3 soma ainda `gamma × min(V̄_grupo; 0)` à recompensa, quando o passo termina em falha ou quando `v3_perdeu_agora` vale verdadeiro (`algoritmo.py:114-131`). `V̄_grupo` é a média do valor do grupo (tarefa e alvo) no rollout anterior.

## 7. Play

`make_env_cfg(play=True)` muda estes pontos (`env_cfg.py:80-83, 105, 113-114, 169-181, 200-210, 216`):

- o ator fica sem ruído: `enable_corruption=False` (`env_cfg.py:113-114`);
- `episode_length_s` = 10⁹ s (`_EPISODIO_PLAY_S`, `env_cfg.py:63, 105`); o `time_out` dispara por `v3_fim`;
- saem os eventos `pd`, `forca_motor`, `solref`, `empurrao`, `forca_caixa` e `rfi`;
- sai o termo de currículo `v3`: o teto do atraso fica no valor do startup (0) e as travas do DR ficam como o checkpoint as carregou (`env_cfg.py:213-216`).

Os eventos de startup (`atrito_pe`, `massa_tronco`, `vies_encoder`, `tamanho_caixa`, `atraso`) e a lista de terminações ficam iguais (`env_cfg.py:185-198, 137-147`).

## 8. Lacunas

Não determinado pelo código lido:

- a evolução de `Curriculo.z`, `v3_h` e `v3_phi_melhor`, usados no termo `privilegiados` (seção 4.5);
- os valores numéricos de `Referencias.pelve_z`, `pelve_com_caixa` e `tronco`, usados no termo `referencia` (seção 4.5);
- a regra de probabilidade que escolhe a célula de velocidade do CARREGAR e a fatia de cada episódio (seção 4.2).

Knobs que nenhuma instrução lida usa: `Contrato.historico` (`knobs.py:56`; o histórico real vem de `Contrato.blocos_ator`) e `Percepcao.raios_por_passo` (`knobs.py:489`).
