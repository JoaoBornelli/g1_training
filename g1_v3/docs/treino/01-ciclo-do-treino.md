# Ciclo do treino, PPO e métricas

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:

- `g1_v3/container/entrada.sh`, `train.py`, `play.py`, `tarefa.py`, `runner.py`, `algoritmo.py`, `env_cfg.py`, `metricas.py`, `knobs.py`.
- `g1_v3/curriculo.py`, `eventos.py`, `terminacoes.py`, `observacoes.py` (função `grupos` e termos `alvo_macro`, `privilegiados`), `comando.py` (classes `CadeiaCmd` e `TwistV3`, funções `_monta`, `avanca_trecho`, `sucesso_do_pegar`, `sucesso_do_botar`, `sucesso_do_segmento`), `balanceador.py` (`Curriculo`), `recompensas.py` (`por_degrau`, `terminacao`, `perda_caixa`, `_fim`), `potencial.py` (`curso`), `cena.py` (`_atuadores`, `robot_cfg`).
- `mjlab/envs/manager_based_rl_env.py`, `mjlab/envs/mdp/events.py`, `mjlab/envs/mdp/actions/actions.py`, `mjlab/envs/mdp/dr/` (`_core.py`, `_types.py`, `actuator.py`, `geom.py`, `body.py`, `joint.py`).
- `mjlab/managers/` (`reward_manager`, `termination_manager`, `event_manager`, `command_manager`, `curriculum_manager`, `metrics_manager`, `action_manager`, `observation_manager`, `manager_base`).
- `mjlab/rl/` (`runner`, `vecenv_wrapper`, `config`), `mjlab/scripts/train.py`, `mjlab/scripts/play.py`, `mjlab/tasks/registry.py`, `mjlab/tasks/velocity/config/g1/env_cfgs.py`, `mjlab/tasks/velocity/config/g1/rl_cfg.py`, `mjlab/tasks/velocity/velocity_env_cfg.py`, `mjlab/tasks/velocity/mdp/velocity_command.py`, `mjlab/utils/gpu.py`, `mjlab/utils/os.py`, `mjlab/utils/torch.py`, `mjlab/utils/buffers/delay_buffer.py`, `mjlab/utils/noise/noise_cfg.py`, `mjlab/utils/wrappers/video_recorder.py`, `mjlab/sim/sim.py`, `mjlab/scene/scene.py`, `mjlab/entity/entity.py`, `mjlab/entity/data.py`, `mjlab/sensor/sensor.py`, `mjlab/viewer/base.py`, `mjlab/actuator/actuator.py` e `mjlab/actuator/builtin_group.py` (só a configuração, a criação e o uso do atraso).
- `rsl_rl/runners/on_policy_runner.py`, `rsl_rl/algorithms/ppo.py`, `rsl_rl/models/mlp_model.py`, `rsl_rl/modules/distribution.py`, `rsl_rl/modules/mlp.py`, `rsl_rl/modules/normalization.py`, `rsl_rl/storage/rollout_storage.py`, `rsl_rl/utils/logger.py`, `rsl_rl/utils/utils.py` (`check_nan`).

Convenções deste documento. Uma referência `arquivo.py:linha` sem prefixo aponta para `g1_v3/`. `entrada.sh` é `container/entrada.sh`. Uma referência `mjlab/...` ou `rsl_rl/...` aponta para a biblioteca. Quando uma seção já citou o caminho completo de um arquivo da biblioteca, o texto seguinte usa só o nome do arquivo: `rl_cfg.py` é `mjlab/tasks/velocity/config/g1/rl_cfg.py`; `config.py` é `mjlab/rl/config.py`; `ppo.py` é `rsl_rl/algorithms/ppo.py`; `on_policy_runner.py`, `logger.py`, `normalization.py`, `distribution.py`, `mlp_model.py` e `rollout_storage.py` são arquivos de `rsl_rl/`; `reward_manager.py`, `termination_manager.py`, `event_manager.py`, `command_manager.py`, `curriculum_manager.py`, `metrics_manager.py`, `action_manager.py` e `observation_manager.py` são arquivos de `mjlab/managers/`; `manager_based_rl_env.py` é `mjlab/envs/manager_based_rl_env.py`; `actions.py` é `mjlab/envs/mdp/actions/actions.py`; `velocity_command.py` é `mjlab/tasks/velocity/mdp/velocity_command.py`. "Env" é um dos mundos simulados em paralelo. "Passo de controle" é um passo da política (uma chamada de `step`). "Iteração" é uma volta do laço do PPO.

## 1. Visão geral

Uma iteração do PPO coleta `Balanceador.passos_por_iteracao` = 24 passos de controle em todos os envs (`knobs.py:402`, `tarefa.py:49`). Depois ela calcula retornos e vantagens e atualiza as redes (`rsl_rl/runners/on_policy_runner.py:79-108`).

O currículo fecha uma iteração dele por iteração do PPO. Quem faz o fecho é o termo de currículo `v3`, dentro do reset de um env, quando o contador de passos cruza um múltiplo de 24 (`curriculo.py:119-121`). O PPO não chama o fecho.

## 2. Como o treino parte

### 2.1 O script `entrada.sh`

O script `container/entrada.sh` roda o treino no pod. Ele lê variáveis de ambiente e monta uma linha de comando.

| Variável | Valor padrão | Efeito | Referência |
|---|---|---|---|
| `RUN` | vazio (obrigatória) | Nome da corrida. Só aceita letras, números, `.`, `-` e `_`. | `entrada.sh:8,19-20` |
| `RETOMA` | vazio (obrigatória) | `0` abre corrida nova; `1` retoma a última corrida do mesmo `RUN` que tem checkpoint. Qualquer outro valor encerra com erro. | `entrada.sh:9,21-24` |
| `ITERS` | 30000 | Vira `--agent.max-iterations`. | `entrada.sh:10,86` |
| `NUM_ENVS` | 4096 | Vira `--env.scene.num-envs`. | `entrada.sh:11,85` |
| `PERCEPCAO` | 1 | `1` escolhe a task `Mjlab-G1-V3`; `0` escolhe `Mjlab-G1-V3-SemPercepcao`. | `entrada.sh:12,80-84` |
| `SAIDA` | `/workspace/g1_v3` | Vira `--log-root`. | `entrada.sh:13,86` |
| `TENSORBOARD` | 1 | Com `1`, sobe o TensorBoard em segundo plano na porta 6006 (`--bind_all`), lendo `SAIDA`. Só roda depois do ramo `SECO`. | `entrada.sh:14,105-107` |
| `SECO` | 0 | Com `1`, pula as checagens de GPU, imprime o comando e sai sem treinar. | `entrada.sh:15,36,99-102` |
| `VIDEO` | 0 | Com `1`, exporta `MUJOCO_GL=egl` e acrescenta `--video True --video-interval 12000 --video-length 1000`. Com `0`, exporta `MUJOCO_GL=disable`. | `entrada.sh:16,28-32,93-97` |

Com `SECO` diferente de 1, o script exige que `nvidia-smi` veja uma GPU, que o torch tenha CUDA, que `mjlab` seja a versão 1.5.3 e que `rsl-rl-lib` seja a versão 5.4.0 (`entrada.sh:36-48`). Depois imprime o último commit do repositório (`entrada.sh:49`).

A pasta do experimento é `$SAIDA/g1_v3` (`entrada.sh:54`). O script procura nela as pastas com o formato `AAAA-MM-DD_HH-MM-SS_$RUN` (`entrada.sh:56`).

- Com `RETOMA=0`, a existência de uma pasta do mesmo `RUN` encerra com erro (`entrada.sh:58-63`).
- Com `RETOMA=1`, o script percorre as pastas da última para a primeira, na ordem do nome, e escolhe a primeira que tem um arquivo `model_*.pt` (`entrada.sh:68-75`). Sem nenhuma, encerra com erro (`entrada.sh:75`).

O comando final é `python -m g1_v3.train <TASK> --env.scene.num-envs <NUM_ENVS> --agent.run-name <RUN> --agent.max-iterations <ITERS> --agent.logger tensorboard --log-root <SAIDA>` (`entrada.sh:85-86`). Com `RETOMA=1`, o script acrescenta `--agent.resume True --agent.load-run "^<pasta escolhida>$"` (`entrada.sh:87-91`). O comando não passa `--agent.experiment-name`, nem `--gpu-ids`, nem `--agent.seed` (`entrada.sh:85-97`). O script substitui o próprio processo pelo comando com `exec`, depois de entrar na raiz do repositório (`entrada.sh:110-111`).

### 2.2 `train.py` e o registro da tarefa

`train.py` importa `g1_v3.tarefa`. O import registra as duas tasks no mjlab. Depois `train.py` chama `mjlab.scripts.train.main` (`train.py:10-14`).

O módulo `tarefa.py` registra, no momento do import, duas tasks com `register_mjlab_task` (`tarefa.py:58-72`):

| Task | `env_cfg` de treino | `play_env_cfg` | `rl_cfg` | `runner_cls` |
|---|---|---|---|---|
| `Mjlab-G1-V3` (`tarefa.py:23`) | `make_env_cfg()` | `make_env_cfg(play=True)` | `rl_cfg()` | `RunnerV3` |
| `Mjlab-G1-V3-SemPercepcao` (`tarefa.py:26`) | `make_env_cfg(_sem_percepcao())` | `make_env_cfg(_sem_percepcao(), play=True)` | `rl_cfg()` | `RunnerV3` |

`_sem_percepcao` copia os knobs com `Percepcao.ligada` = False (`tarefa.py:53-54`). Com `ligada` = False, o termo `mapa` sai dos dois grupos de observação (`observacoes.py:401-402`).

Os knobs ativos são `ATIVO = Knobs()` (`knobs.py:569`). A função `rl_cfg()` parte de `unitree_g1_ppo_runner_cfg()` do mjlab e troca quatro campos (`tarefa.py:45-50`): `experiment_name` = `"g1_v3"` (`tarefa.py:27,46`), `algorithm.class_name` = `"g1_v3.algoritmo:PPOPorTarefa"` (`algoritmo.py:27`), `algorithm.entropy_coef` = 0,001 e `num_steps_per_env` = `Balanceador.passos_por_iteracao` = 24 (`knobs.py:402`). O resto vem do mjlab (seção 11).

### 2.3 O `main` do mjlab

`mjlab.scripts.train.main` monta um `TrainConfig` com o `tyro`, a partir dos cfgs da task escolhida (`mjlab/scripts/train.py:43-46,230-254`). O `launch_training` faz o resto (`mjlab/scripts/train.py:182-227`).

1. Calcula a pasta da corrida: `<log-root>/<experiment_name>/<AAAA-MM-DD_HH-MM-SS>_<run_name>`. Na retomada, a pasta nova também leva a data e hora do momento (`mjlab/scripts/train.py:186-190`). Com os padrões do `entrada.sh`, a pasta é `/workspace/g1_v3/g1_v3/<data>_<RUN>`.
2. Seleciona GPUs com `select_gpus(gpu_ids)`. O padrão de `gpu_ids` é `[0]` (`mjlab/scripts/train.py:40`). Sem a variável `CUDA_VISIBLE_DEVICES`, a função usa todas as GPUs que o torch vê e escolhe o índice 0 (`mjlab/utils/gpu.py:52-77`). Com `CUDA_VISIBLE_DEVICES` vazia, a função devolve CPU (`mjlab/utils/gpu.py:62-64`). O `launch_training` grava em `CUDA_VISIBLE_DEVICES` a lista escolhida, ou vazia em modo CPU (`mjlab/scripts/train.py:195-199`).
3. Com uma GPU ou nenhuma, chama `run_train` direto. Com mais de uma GPU, usa `torchrunx` (`mjlab/scripts/train.py:201-227`). O `entrada.sh` não passa `--gpu-ids`, então o treino usa uma GPU.

O `run_train` executa, em ordem (`mjlab/scripts/train.py:49-179`):

1. Escolhe o dispositivo. Com `CUDA_VISIBLE_DEVICES` vazia, é `cpu`, a semente vale `cfg.agent.seed` e o `rank` vale 0. Com a variável preenchida, o dispositivo é `cuda:<LOCAL_RANK>` (`LOCAL_RANK` vale 0 se não existir) e a semente vale `cfg.agent.seed` + `RANK` (`mjlab/scripts/train.py:50-62`). O padrão de `seed` é 42 (`mjlab/rl/config.py:86`).
2. Chama `configure_torch_backends()` com os padrões `allow_tf32=True` e `deterministic=False` (`mjlab/scripts/train.py:64`, `mjlab/utils/torch.py:5`).
3. Grava a semente em `cfg.agent.seed` e em `cfg.env.seed` (`mjlab/scripts/train.py:66-67`). O construtor do env semeia o gerador global com `cfg.env.seed` (`manager_based_rl_env.py:186-187`) e o `v3_gerador` do env nasce com a mesma semente (`curriculo.py:62-64`).
4. Cria o env: `ManagerBasedRlEnv(cfg.env, device)` (`mjlab/scripts/train.py:110-112`).
5. Com `--agent.resume True`, resolve o caminho do checkpoint com `get_checkpoint_path(log_root_path, load_run, load_checkpoint)`, onde `log_root_path` é a pasta do experimento (`mjlab/scripts/train.py:114-135`). A resolução acontece antes de o runner existir. O carregamento vem no item 9.
6. Com `--video`, embrulha o env em `VideoRecorder` (só no `rank` 0) (`mjlab/scripts/train.py:138-146`).
7. Embrulha o env em `RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)` (`mjlab/scripts/train.py:148`). O padrão de `clip_actions` é `None`: a ação não é limitada (`mjlab/rl/config.py:121`, `mjlab/rl/vecenv_wrapper.py:75-76`). O construtor do wrapper chama `env.reset()` (`mjlab/rl/vecenv_wrapper.py:24-25`).
8. Grava `params/env.yaml` e `params/agent.yaml` na pasta da corrida (`mjlab/scripts/train.py:163-165`) e cria o runner com a classe registrada, `RunnerV3` (`mjlab/scripts/train.py:153-155,167`). O runner chama `add_git_repo_to_log(__file__)` (`mjlab/scripts/train.py:170`).
9. Com o caminho do item 5, imprime o caminho e chama `runner.load(caminho)` (`mjlab/scripts/train.py:171-173`).
10. Chama `runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)` (`mjlab/scripts/train.py:175-177`) e, no fim, `env.close()` (`mjlab/scripts/train.py:179`).

### 2.4 A retomada

O `--agent.load-run` é uma expressão regular. O `get_checkpoint_path` filtra as pastas do experimento com `re.match` (a âncora vale só no início do nome), ordena por nome e usa a última (`mjlab/utils/os.py:40-51`). O `--agent.load-checkpoint` tem o padrão `model_.*.pt` (`mjlab/rl/config.py:117`). O `get_checkpoint_path` ordena os nomes de arquivo preenchendo com zeros à esquerda até 15 caracteres e usa o último (`mjlab/utils/os.py:58-60`).

O laço do `learn` percorre as iterações de `start_it` = `current_learning_iteration` até `start_it + max_iterations − 1` (`rsl_rl/runners/on_policy_runner.py:77-79`). O `current_learning_iteration` nasce 0 (`rsl_rl/runners/on_policy_runner.py:54`). Na retomada, o `load` o troca pelo campo `iter` do checkpoint (`mjlab/rl/runner.py:134-136`). O campo `iter` guarda o valor do `current_learning_iteration` no instante do `save` (`mjlab/rl/runner.py:77`). O `learn` atribui `current_learning_iteration = it` depois de terminar a iteração `it` (`rsl_rl/runners/on_policy_runner.py:112`). Por isso, o primeiro laço da retomada tem `it` igual ao `iter` do checkpoint: a iteração com esse índice roda uma segunda vez.

## 3. Os tempos

| Grandeza | Valor | Origem |
|---|---|---|
| Passo da física (`dt_fisica`) | 0,005 s | `Contrato.dt_fisica` (`knobs.py:54`), aplicado em `cfg.sim.mujoco.timestep` (`env_cfg.py:103`) |
| Decimação | 4 subpassos de física por passo de controle | `Contrato.decimacao` (`knobs.py:55`), aplicado em `cfg.decimation` (`env_cfg.py:104`) |
| Duração do passo de controle | 0,02 s (50 Hz) | `timestep × decimation` (`mjlab/envs/manager_based_rl_env.py:265-268`) |
| Duração do episódio no treino | 20,0 s | `Cadeia.episodio_s` (`knobs.py:160`), aplicado em `cfg.episode_length_s` (`env_cfg.py:105`) |
| Subpassos de física por episódio | 4 000 | 1000 passos × `Contrato.decimacao` = 4 (`knobs.py:55`) |
| Duração do episódio em passos | 1000 | `ceil(episode_length_s / step_dt)` = `ceil(20,0 / 0,02)` (`mjlab/envs/manager_based_rl_env.py:281-283`) |
| Passos por env por iteração | 24 | `Balanceador.passos_por_iteracao` (`knobs.py:402`) |
| Tempo simulado por iteração | 0,48 s por env | 24 passos × 0,02 s |
| Iterações por episódio | cerca de 41,7 | 1000 passos ÷ 24 passos por iteração |
| Amostras por iteração | 98 304 com 4096 envs | 24 × `NUM_ENVS` (`entrada.sh:11`) |
| Tamanho do minibatch | 24 576 com 4096 envs | amostras ÷ `num_mini_batches` = 4 (`rsl_rl/storage/rollout_storage.py:229-230`, `mjlab/tasks/velocity/config/g1/rl_cfg.py:34`) |
| Passos do otimizador por iteração | 20 | `num_learning_epochs` = 5 × `num_mini_batches` = 4 (`rl_cfg.py:33-34`, `rsl_rl/algorithms/ppo.py:321`) |
| Duração do episódio no play | 10⁹ s (cerca de 5 × 10¹⁰ passos) | `_EPISODIO_PLAY_S` (`env_cfg.py:63,105`) |

Os outros parâmetros da física que o `make_env_cfg` troca: `njmax` = `Cena.njmax` = 800 (`knobs.py:107`), `nconmax` = `Cena.nconmax` = 300 (`knobs.py:108`), `ccd_iterations` = `Cena.ccd_iterations` = 50 (`knobs.py:109`), `impratio` = `Cena.impratio` = 1,0 (`knobs.py:110`) e `cone` = `Cena.cone` = `"pyramidal"` (`knobs.py:111`), todos em `env_cfg.py:98-102`. O molde do fabricante fixa `iterations` = 10 e `ls_iterations` = 20 do solver (`mjlab/tasks/velocity/velocity_env_cfg.py:450-451`). O `make_env_cfg` não os troca.

Ao iniciar o `learn`, todos os envs recebem um `episode_length_buf` sorteado: inteiro uniforme em [0, 1000) (`rsl_rl/runners/on_policy_runner.py:59-62`). O primeiro episódio de cada env dura, no máximo, `1000 − episode_length_buf` passos, porque a terminação `time_out` compara o contador com 1000 (`terminacoes.py:53-72`). Os episódios seguintes têm o tamanho completo.

## 4. A construção do env

A criação do env roda uma vez e precede o primeiro passo. A ordem é (`mjlab/envs/manager_based_rl_env.py:177-251,299-354`):

1. Semeia o gerador global com `cfg.seed` (`manager_based_rl_env.py:186-187`).
2. Cria a cena e a simulação (`manager_based_rl_env.py:196-209`).
3. Zera `common_step_counter` e `episode_length_buf` (`manager_based_rl_env.py:231-234`).
4. Cria os managers na ordem: eventos, comandos, ação, observações, terminações, recompensas, currículo, métricas, gravador (`manager_based_rl_env.py:306-346`). O manager de eventos precede os demais, para o modelo expandir os campos de randomização de domínio (`manager_based_rl_env.py:306-309`).
5. O construtor do termo de comando `cadeia` chama `garante_v3`, que cria o objeto de currículo `v3_cur` e os buffers `v3_*` (`comando.py:293`, `curriculo.py:39-90`).
6. O construtor do `ObservationManager` chama cada função de termo uma vez, para medir a dimensão do termo (`observation_manager.py:446`). Essas chamadas incluem os termos `caixa` e `mapa`, que rodam o estimador (seção 5.6).
7. Roda os eventos de modo `startup`, na ordem do dicionário (seção 6.5), com `apply(mode="startup")` (`manager_based_rl_env.py:352-354`).
8. O wrapper do rsl_rl chama `env.reset()` (`mjlab/rl/vecenv_wrapper.py:25`). Esse reset roda `_reset_idx` em todos os envs (`manager_based_rl_env.py:364-376`). É a primeira chamada do termo de currículo `v3`.

O `reset()` do env roda `_reset_idx`, `scene.write_data_to_sim()`, `sim.forward()`, `command_manager.compute(dt=0.0)`, `sim.sense()` e `observation_manager.compute(update_history=True)`, nesta ordem (`manager_based_rl_env.py:369-374`). Ele também zera `extras["log"]` e chama `recorder_manager.record_post_reset` (`manager_based_rl_env.py:368,375`).

No primeiro `reset()`, `common_step_counter` vale 0, e a chamada do item 6 acima já gravou `v3_percepcao_passo` = 0. A guarda `atualiza_percepcao` (seção 5.6) então não roda de novo: as observações `caixa` e `mapa` desse reset saem do cálculo feito na criação do env (`observacoes.py:145-151`, `curriculo.py:60-61`).

## 5. A ordem exata dentro de um passo do env

A função `step(action)` executa os itens abaixo, nesta ordem (`mjlab/envs/manager_based_rl_env.py:378-479`). O reset só roda para os envs em que alguma terminação disparou, porque `auto_reset` vale True por padrão (`manager_based_rl_env.py:144,445`).

| N | Etapa | O que faz | Referência |
|---|---|---|---|
| 1 | Zera o log | `extras["log"]` vira um dicionário vazio. As outras chaves de `extras` persistem entre passos. | `manager_based_rl_env.py:418` |
| 2 | Processa a ação | Guarda a ação no histórico do `ActionManager` e calcula o alvo das juntas. | `manager_based_rl_env.py:419`, `mjlab/managers/action_manager.py:152-175` |
| 3 | Laço de decimação | Repete `cfg.decimation` = `Contrato.decimacao` = 4 vezes (`knobs.py:55`) o bloco 3a a 3e. Cada repetição soma 1 a `_sim_step_counter` antes do bloco. | `manager_based_rl_env.py:421-427` |
| 3a | Aplica a ação | Escreve o alvo de posição de cada junta. | `manager_based_rl_env.py:423`, `mjlab/envs/mdp/actions/actions.py:221-224` |
| 3b | Escreve a cena | `scene.write_data_to_sim()`. | `manager_based_rl_env.py:424` |
| 3c | Passo da física | `sim.step()`, com o passo de `Contrato.dt_fisica` = 0,005 s (`knobs.py:54`). | `manager_based_rl_env.py:425`, `mjlab/sim/sim.py:492-498` |
| 3d | Atualiza a cena | `scene.update(dt=physics_dt)`. | `manager_based_rl_env.py:426` |
| 3e | Métricas por subpasso | `metrics_manager.compute_substep()`. Nenhum termo de `metricas.termos` usa `per_substep`, então a chamada não faz nada. | `manager_based_rl_env.py:427`, `mjlab/managers/metrics_manager.py:135-145`, `metricas.py:532-570` |
| 4 | Contadores | `episode_length_buf += 1` e `common_step_counter += 1`. | `manager_based_rl_env.py:430-431` |
| 5 | Terminações | `termination_manager.compute()`. Calcula `reset_buf`, `reset_terminated` e `reset_time_outs`. | `manager_based_rl_env.py:436-438` |
| 6 | Recompensas | `reward_manager.compute(dt=step_dt)`. | `manager_based_rl_env.py:440` |
| 7 | Métricas | `metrics_manager.compute()`. | `manager_based_rl_env.py:441` |
| 8 | Reset dos envs que acabaram | Se `reset_buf` tem algum env, chama `recorder_manager.record_pre_reset`, `_reset_idx(reset_env_ids)` e `scene.write_data_to_sim()`. | `manager_based_rl_env.py:444-448` |
| 9 | Cinemática | `sim.forward()` uma vez, para todos os envs. | `manager_based_rl_env.py:454` |
| 10 | Comandos | `command_manager.compute(dt=step_dt)`. | `manager_based_rl_env.py:456` |
| 11 | Eventos de passo | `event_manager.apply(mode="step")`, se existir algum termo desse modo. | `manager_based_rl_env.py:458-459` |
| 12 | Eventos de intervalo | `event_manager.apply(mode="interval")`, se existir algum termo desse modo. | `manager_based_rl_env.py:460-461` |
| 13 | Sensores | `sim.sense()`. | `manager_based_rl_env.py:463` |
| 14 | Observações | `observation_manager.compute(update_history=True)`. | `manager_based_rl_env.py:464` |
| 15 | Fecho | Para os envs resetados, `recorder_manager.record_post_reset`. Depois `record_post_step`. Devolve observações, recompensa, `reset_terminated`, `reset_time_outs` e `extras`. | `manager_based_rl_env.py:466-479` |

O wrapper do rsl_rl completa o passo. Ele junta `terminated | truncated` em `dones` (tipo `long`) e põe `extras["time_outs"] = truncated` quando `is_finite_horizon` é False, que é o padrão (`mjlab/rl/vecenv_wrapper.py:72-89`, `manager_based_rl_env.py:134`).

Consequências da ordem:

- O `sim.forward()` só roda no item 9. As terminações e as recompensas rodam sem ele (`manager_based_rl_env.py:436,440,454`). Elas leem as grandezas derivadas (posição de corpos, sensores) como a física as deixou. Para os envs resetados no item 8, essas grandezas só se atualizam no item 9.
- O reset (item 8) roda depois das recompensas e das métricas do mesmo passo. A recompensa de um passo terminal pertence ao episódio que acabou.
- Os comandos (item 10) rodam depois do reset (item 8) e depois das terminações e das recompensas. Um valor que o comando escreve no passo `t` só é lido por terminações e recompensas no passo `t+1`. É o caso do pulso `v3_perdeu_agora` (seção 5.4).
- Os eventos de passo e de intervalo rodam depois dos comandos e depois do reset. Nenhum `sim.step()` roda entre esses eventos e o fim do passo. O empurrão, a força e o torque que eles escrevem agem a partir dos subpassos de física do passo seguinte.

### 5.1 A ação (itens 2 e 3a)

O `process_action` desloca o histórico (`prev_prev ← prev ← atual ← nova`) e entrega a fatia da ação ao termo (`mjlab/managers/action_manager.py:167-174`). O termo `joint_pos` é um `JointPositionAction` com `actuator_names=(".*",)`, que cobre as 29 juntas (`env_cfg.py:108-109`). A ordem do vetor de ação é a de `target_names`, que o mjlab resolve (`mjlab/envs/mdp/actions/actions.py:100-110`); a escala vem de um dicionário nome → valor montado de `JUNTAS` (`knobs.py:31-41,589-592`). O termo calcula `alvo = ação × escala + offset` (`mjlab/envs/mdp/actions/actions.py:154-162`).

- A escala é `Contrato.escala_acao` = 0,25 rad por unidade para todas as juntas, exceto `wrist_pitch` e `wrist_yaw`, que usam `Contrato.escala_acao_punho` = 0,075 (`knobs.py:52-53,589-592`, `env_cfg.py:109`).
- O offset é a posição padrão das juntas (`use_default_offset=True`), que vem de `q_de_pe` (`env_cfg.py:109`, `cena.py:225`, `mjlab/envs/mdp/actions/actions.py:216-219`).
- Não há `clip` no termo (`env_cfg.py:108-109`; o padrão de `clip` é `None`, `mjlab/managers/action_manager.py:29`).

Em cada subpasso de física, o termo escreve `alvo − encoder_bias` como alvo de posição (`mjlab/envs/mdp/actions/actions.py:221-224`). O `encoder_bias` é o viés de encoder sorteado no startup (seção 6.5).

Cada um dos 9 grupos de atuadores (uma linha de `Contrato.pd`) tem `delay_max_lag` = `DR.atraso_refino[1] × decimação` = 3 × 4 = 12 passos de física, `delay_min_lag` = 0, `delay_update_period` = `NUNCA` (10⁹) e `delay_per_env_phase` = False (`cena.py:176-198`, `knobs.py:47,55,513`). Os grupos com a mesma configuração de atraso compartilham um só `DelayBuffer` (`mjlab/actuator/builtin_group.py:111-121,135-154,158-173`). O buffer recebe o alvo e escreve o valor atrasado em cada subpasso de física, dentro de `scene.write_data_to_sim()` (item 3b) (`mjlab/entity/entity.py:852-858,1268-1270`, `mjlab/actuator/builtin_group.py:175-192`). O atraso máximo efetivo é o atributo `max_lag` do buffer, que o evento `atraso` e o termo `v3` ajustam (seções 6.2 e 6.5). Como `delay_update_period` = 10⁹ e a fase é 0, o `DelayBuffer` sorteia o atraso de cada env uma vez, no primeiro subpasso depois do reset do env, com `randint(0, max_lag + 1)` (`mjlab/utils/buffers/delay_buffer.py:247-258,260-277`). O reset do env zera o buffer (item 3 da tabela da seção 6; `mjlab/scene/scene.py:190-194`, `mjlab/entity/entity.py:832-841`, `mjlab/actuator/actuator.py:415-425`). Cada `DelayBuffer` tem um atraso por env, que vale para todos os alvos que o buffer cobre. Os 9 grupos têm a mesma configuração de atraso (`cena.py:192-198`); a regra de fusão e a dúvida sobre `has_delay` ficam na seção 17.

### 5.2 As terminações (item 5)

O `compute` percorre os termos na ordem do dicionário e acumula (`mjlab/managers/termination_manager.py:102-113`):

- um termo com `time_out=True` soma em `time_outs` (truncamento);
- os demais somam em `terminated`.

A ordem dos termos é `time_out`, `queda`, `caixa_perdida`, `chegada`, `batente` (`env_cfg.py:137-147`). A seção 8 traz as condições exatas.

### 5.3 As recompensas (item 6)

A seção 7 explica a soma e a multiplicação por `dt`.

### 5.4 Os comandos (item 10)

O `CommandManager.compute` chama `compute(dt)` de cada termo, na ordem do dicionário: `cadeia`, depois `twist` (`mjlab/managers/command_manager.py:249-251`, `env_cfg.py:118-128`).

**Termo `cadeia`** (`CadeiaCmd`). A classe substitui o `compute` do mjlab (`comando.py:596-629`). A ordem dentro dele:

1. `_mede`: lê base, caixa, laje, palmas, contatos e tronco (`comando.py:598`).
2. `_perda`: escreve o pulso `v3_perdeu_agora` (`comando.py:599`, `comando.py:394-420`). O pulso vale True nos envs em que a caixa foi perdida neste passo, em tarefa diferente de ANDAR e com `episode_length_buf` maior que 0 (`comando.py:409-411`). Com `Cadeia.caixa_perdida_vira_andar` = True (`knobs.py:161`), a perda marca `v3_perdeu`, põe o alvo em B3 e publica a tarefa ANDAR (`comando.py:412-420`).
3. Publica o pulso em `env.extras["v3_perdeu_agora"]` (`comando.py:601`). É o mesmo tensor do buffer, não uma cópia.
4. `_abre_p0`: abre o alvo P0 do PEGAR na primeira leitura válida da caixa (`comando.py:602`, `comando.py:370-392`).
5. `_alvos`: avalia o fecho do alvo corrente e avança o alvo (`comando.py:603`).
6. Atualiza o latch `v3_aviso` com o fecho do alvo do trecho (`comando.py:606-607`).
7. `avanca_trecho`: avança o relógio do trecho e decide o fim (`comando.py:608-609`). Uma caixa perdida congela o trecho (`comando.py:610-613`).
8. Marca `v3_fim` no fim da cadeia e registra o BOTAR pendente no balanceador (`comando.py:614-616`).
9. `_abre_trecho` para os envs que avançaram de trecho (`comando.py:617-619`).
10. Dá um passo na odometria `v3_odo` (`comando.py:623-625`).
11. `_grava_chegada` e `v3_vz_caixa_ant[:] = vz da caixa` (`comando.py:626-629`).

**Termo `twist`** (`TwistV3`, que herda `UniformVelocityCommand`). O `compute` vem do `CommandTerm` (`mjlab/managers/command_manager.py:109-115`, `mjlab/tasks/velocity/mdp/velocity_command.py:200-201`). A ordem:

1. `_update_metrics` (`comando.py:810-877`). Esse método, nesta ordem: soma `dt` no relógio do episódio `_agora` e guarda a pose do fim do passo em `pose_ult` (813-815); fecha e abre segmentos quando `command_counter` difere de `seg_visto` (818-821); acumula a velocidade e o comando no segmento e a inclinação máxima da caixa (823-832); trata a âncora (834-842); chama o `_update_metrics` do fabricante (`error_vel_xy`, `error_vel_yaw`) e atualiza `razao_marcha` (844-850); acumula a velocidade desde a âncora e atualiza o filtro `v_filt` com a constante `Comando.tau_fracao` = 0,4 s (`knobs.py:136`) (854-858); atualiza o acerto do relógio de marcha, o filtro `q_filt`, o fator `g_marcha`, a fase `v3_fase_marcha` e `acerto_relogio` (863-877).
2. `time_left -= dt` (`mjlab/managers/command_manager.py:111`).
3. Os envs com `time_left <= 0` reamostram: `time_left` ← sorteio uniforme em `Comando.reamostra_s` = (3,0; 8,0) s (`knobs.py:137`, `env_cfg.py:122-123`), `_resample_command` e `command_counter += 1` (`mjlab/managers/command_manager.py:112-114,117-129`). O `_resample_command` do `TwistV3` sorteia por célula do currículo (`comando.py:897-941`). No ANDAR, ele liga `is_heading_env` nos envs de terreno com comando linear de norma ≥ `Comando.limiar_canal` e, no começo do episódio, grava `heading_target` (`comando.py:932-937`). Num CARREGAR da fatia B, ele fixa `time_left` em infinito: o comando fica constante no trecho (`comando.py:939-941`).
4. `_update_command` chama o do fabricante, que põe o wz de rumo (`heading_control_stiffness` × erro de rumo, com teto na faixa de giro) nos envs com `is_heading_env`. Depois zera o wz de módulo menor que `Comando.giro_min` nesses envs (`comando.py:943-947`). Fora do terreno do ANDAR nenhum env liga `is_heading_env`, então o método não muda nada.

O sorteio da velocidade pela função `sorteia_velocidade` está em `comando.py:118-158`. O `_resample_command` roda dentro do `_resample` do mjlab, que sorteia `time_left` antes (`mjlab/managers/command_manager.py:117-123`).

**A cadeia de trechos.** O trecho é um pedaço do episódio com uma tarefa publicada. `_monta` define a tabela de trechos de cada fatia (`comando.py:171-194`). A duração, ou o atraso, de cada trecho é sorteada quando o trecho abre (no reset, ou quando o trecho anterior acaba), uniforme em [lo; hi), com `lo + (hi − lo) × u` (`comando.py:578-579`). Um trecho "por aviso" só avança depois que o latch `v3_aviso` liga, e conta o atraso a partir do aviso (`comando.py:201-216`).

| Fatia | Trecho | Tarefa | Por aviso | Faixa (s) | Knob |
|---|---|---|---|---|---|
| ANDAR | 1 | ANDAR | não | (20,0; 20,0) | `Cadeia.episodio_s` (`knobs.py:160`) |
| B | 1 | ANDAR | não | (0,5; 1,0) | `Cadeia.espera_s` (`knobs.py:154`) |
| B | 2 | PEGAR | sim | (0,0; 1,0) | `Cadeia.atraso_pegar_s` (`knobs.py:158`) |
| B | 3 | CARREGAR | não | (4,0; 7,0) | `Cadeia.carregar_b_s` (`knobs.py:155`) |
| C | 1 | ANDAR | não | (0,5; 1,0) | `Cadeia.espera_s` |
| C | 2 | PEGAR | sim | (0,0; 1,0) | `Cadeia.atraso_pegar_s` |
| C | 3 | CARREGAR | não | (0,5; 1,0) | `Cadeia.carregar_c_s` (`knobs.py:156`) |
| C | 4 | BOTAR | sim | (0,0; 1,5) | `Cadeia.atraso_botar_s` (`knobs.py:159`) |
| C | 5 | ANDAR | não | (1,5; 1,5) | `Cadeia.andar_final_s` (`knobs.py:157`) |

Ao acabar o último trecho da fatia, `avanca_trecho` devolve `fim`, `v3_fim` liga, e a terminação `time_out` dispara no passo seguinte (seção 8). No terreno do ANDAR, o `time_out` dispara também na travessia (seção 8). O aviso do trecho PEGAR é o fecho do P3, e o do trecho BOTAR é o fecho do B3 (`comando.py:606-607`).

### 5.5 Os eventos de passo e de intervalo (itens 11 e 12)

O item 11 roda os termos de modo `step`, na ordem do dicionário: `forca_caixa` e `rfi` (`env_cfg.py:207-210`). O item 12 roda os de modo `interval`: `empurrao` (`env_cfg.py:201-205`). Esses três só existem no treino (seção 15). A tabela 6.6 detalha cada um.

### 5.6 As observações (item 14)

O `compute(update_history=True)` calcula, para cada grupo (`actor` e `critic`), os termos na ordem do dicionário (`mjlab/managers/observation_manager.py:305-318`). Para cada termo, a ordem é (`observation_manager.py:329-362`):

1. chama a função do termo e clona o resultado (linha 330);
2. aplica ruído, se o termo tem (linhas 331-334);
3. aplica `clip`, se houver (linhas 335-336);
4. multiplica pela `scale` (linhas 337-340);
5. confere NaN e infinito só se `nan_policy` do grupo não é `disabled`; o padrão é `disabled` e `grupos` não o muda, então a conferência não roda (linhas 342-346, `observation_manager.py:100`, `observacoes.py:419-422`);
6. passa pelo buffer de atraso, se `delay_max_lag > 0` (linhas 348-351);
7. passa pelo buffer de histórico, se `history_length > 0`, e entrega o buffer achatado (linhas 352-362). O buffer só recebe um valor novo se `update_history` é True ou se ele ainda não foi inicializado (linha 354).

Depois, concatena os termos do grupo (`observation_manager.py:377-380`). Os dois grupos têm `concatenate_terms=True`. O grupo `actor` tem `enable_corruption=True`, e o `critic` tem `enable_corruption=False` (`observacoes.py:419-422`). Com `enable_corruption` False, o manager zera o ruído de cada termo (`observation_manager.py:436-437`).

| Elemento | Valor | Referência |
|---|---|---|
| Termos do ator, em ordem | `giro_base`, `gravidade`, `comando`, `fase`, `juntas`, `vel_juntas`, `acao`, `tarefa`, `caixa`, `mapa` | `observacoes.py:389-400` |
| Termos do crítico, em ordem | os mesmos do ator (com `juntas` sem viés) e depois `vel_lin_base`, `caixa_verdade`, `alvo_verdade`, `alvo_macro`, `referencia`, `niveis`, `degrau`, `massa`, `forca_caixa`, `privilegiados`, `tarefa_critico` | `observacoes.py:403-418` |
| Escala e histórico | `Contrato.blocos_ator`: (nome, canais, escala, histórico). Escalas: `giro_base` 0,2, `vel_juntas` 0,05 e 1,0 nos demais (`knobs.py:58-63`). O histórico vem do quarto elemento de cada tupla: 5 passos em todos os blocos, exceto `mapa` (0). O campo `Contrato.historico` = 5 (`knobs.py:56`) não é lido por nenhum código lido. No crítico, `vel_lin_base` tem histórico 5 pelo literal `hist_padrao=5` (`observacoes.py:407`), e os termos fora de `blocos_ator` têm histórico 0. | `knobs.py:58-63`, `observacoes.py:368-369,373-381,405-408` |
| Ruído do ator | uniforme em [−r, +r), somado ao valor bruto; a escala multiplica o resultado depois: `giro_base` ±0,2; `gravidade` ±0,05; `juntas` ±0,01; `vel_juntas` ±1,5 (`DR.ruido_obs`) | `knobs.py:516`, `observacoes.py:380`, `mjlab/utils/noise/noise_cfg.py:20,76-80` |
| Termos do ator com atraso | `giro_base`, `gravidade`, `juntas`, `vel_juntas`, `mapa` | `observacoes.py:55,376-386` |
| Atraso | `delay_min_lag` = 0, `delay_max_lag` = `DR.atraso_refino[1]` = 3, `delay_update_period` = `NUNCA` (10⁹), `delay_per_env_phase` = False | `observacoes.py:382-386`, `knobs.py:47,513` |
| Entrada do ator com percepção ligada | 1 457 números (soma de canais × histórico, `mapa` uma vez) | `knobs.py:58-63,575-577` |
| Entrada do crítico | 1 457 (os mesmos termos do ator) + 46 (`caixa_verdade` 7, `alvo_verdade` 3, `alvo_macro` 11, `referencia` 8, `niveis` 4, `degrau` 1, `massa` 1, `forca_caixa` 1, `privilegiados` 4, `tarefa_critico` 6) + 5 × (canais de `robot/imu_lin_vel`) = 1 503 + 5 × (canais). Os canais de `imu_lin_vel` ficam na seção 17. | `observacoes.py:46,267-336,403-418` |
| Último termo do crítico | `tarefa_critico`: one-hot de 6 canais da tarefa (`TAREFAS`) | `observacoes.py:417`, `knobs.py:42` |

O sorteio do atraso. O `DelayBuffer` sorteia o atraso por env, com `randint(min_lag, max_lag + 1)`, quando `(step_count + fase) % update_period == 0` (`mjlab/utils/buffers/delay_buffer.py:247-258`, `_sample_lags` em `delay_buffer.py:260`). Com `update_period` = 10⁹ e fase 0, isso vale só quando `step_count` é 0. O reset do buffer zera `step_count` (`mjlab/utils/buffers/delay_buffer.py:194-209`). Então o sorteio acontece uma vez por episódio, na primeira observação calculada depois do reset do env, e o valor vale até o fim do episódio. O teto do sorteio é o atributo `max_lag` do buffer, que o evento `atraso` e o termo `v3` ajustam (seção 6.5 e seção 9).

⚠ Divergência: o comentário de `observacoes.py:54` diz atraso de 0 a 1 passo; o código sorteia em [0; `max_lag`], com `max_lag` = `DR.atraso_inicio[1]` = 0 no startup e até `DR.atraso_refino[1]` = 3 depois (`observacoes.py:382-386`, `knobs.py:511-514`, `eventos.py:244-258`).

O termo `mapa` e o termo `caixa` chamam `atualiza_percepcao`. A função roda o estimador uma vez por valor de `common_step_counter`, mesmo com dois grupos lendo (`observacoes.py:140-163`).

## 6. A ordem exata dentro do reset de um env (`_reset_idx`)

O `_reset_idx(env_ids)` recebe os envs que acabaram. Ele roda no item 8 de um passo (seção 5) e no `reset()` inicial (seção 4). A ordem (`mjlab/envs/manager_based_rl_env.py:553-591`):

| N | Etapa | Referência |
|---|---|---|
| 1 | `curriculum_manager.compute(env_ids)`: chama o termo `v3` (seção 6.2). | `manager_based_rl_env.py:554`, `mjlab/managers/curriculum_manager.py:112-117` |
| 2 | `sim.reset(env_ids)`: reinicia os dados da simulação dos envs. | `manager_based_rl_env.py:555`, `mjlab/sim/sim.py:500-511` |
| 3 | `scene.reset(env_ids)`: para cada entidade, zera os alvos de atuador (posição, velocidade e esforço) e reinicia o estado de cada atuador, o que zera o buffer de atraso do atuador (o atraso novo sai no primeiro subpasso depois); para cada sensor, invalida o cache de dados. | `manager_based_rl_env.py:556`, `mjlab/scene/scene.py:190-194`, `mjlab/entity/entity.py:832-841`, `mjlab/entity/data.py:216-225`, `mjlab/actuator/actuator.py:415-425`, `mjlab/sensor/sensor.py:129-139` |
| 4 | Eventos de modo `reset`, na ordem do dicionário (seção 6.3). | `manager_based_rl_env.py:558-562` |
| 5 | `observation_manager.reset(env_ids)` (seção 6.4). | `manager_based_rl_env.py:566` |
| 6 | `action_manager.reset(env_ids)`. | `manager_based_rl_env.py:569` |
| 7 | `reward_manager.reset(env_ids)`. | `manager_based_rl_env.py:572` |
| 8 | `metrics_manager.reset(env_ids)`. | `manager_based_rl_env.py:575` |
| 9 | `curriculum_manager.reset(env_ids)`. | `manager_based_rl_env.py:578` |
| 10 | `command_manager.reset(env_ids)`: `cadeia` e depois `twist`. | `manager_based_rl_env.py:581` |
| 11 | `event_manager.reset(env_ids)`. | `manager_based_rl_env.py:584` |
| 12 | `termination_manager.reset(env_ids)`. | `manager_based_rl_env.py:587` |
| 13 | `episode_length_buf[env_ids] = 0`. | `manager_based_rl_env.py:590` |

Cada manager devolve um dicionário. O `_reset_idx` junta todos em `extras["log"]` (`manager_based_rl_env.py:567-588`). O que o rsl_rl faz com esse log está na seção 16.

### 6.1 O primeiro item decide a iteração do currículo

O item 1 é a primeira etapa do reset. Ele roda o fecho da iteração do balanceador, se for o caso, e os sorteios do reset vêm depois. Os sorteios dos itens 4 e 10 já usam a tabela de frações e os níveis atualizados. As notas que o item 10 registra (seção 9.2) entram na iteração seguinte, porque o fecho já rodou.

O evento `fatia` (item 4) registra a visita e a travessia do terreno do ANDAR do episódio que acabou. Ele lê `pose_ult` e `soma_cmd_marcha` do `twist` antes de o `twist` reiniciar no item 10 (`eventos.py:143-160`). O registro cai depois do fecho do item 1, então entra na iteração seguinte.

### 6.2 O termo de currículo `v3`

O `curriculo.v3(env, env_ids)` roda em todo reset, com os envs que estão sendo resetados (`curriculo.py:103-143`). A ordem:

1. `garante_v3(env)`: cria o que falta, sem zerar estado (`curriculo.py:117`).
2. Calcula `it = common_step_counter // Balanceador.passos_por_iteracao` (24, `knobs.py:402`) (`curriculo.py:119`).
3. Se `v3_iter >= 0` e `it != v3_iter`, chama `cur.fecha_iteracao()` (`curriculo.py:120-121`). Na primeira chamada do env, `v3_iter` vale −1, então o fecho não roda (`curriculo.py:58-59`). Depois, guarda `v3_iter = it` (`curriculo.py:122`).
4. Lê `cur.dr()`. Se o teto de atraso `atraso_max` difere de `DR.atraso_inicio[1]` = 0 (`knobs.py:511`), chama `eventos.atraso(env, env_ids, teto, decimação)` (`curriculo.py:123-126`). A função ignora `env_ids`: ela troca o `max_lag` dos buffers de todos os envs, para `teto × decimação` nos atuadores e `teto` nas observações (`eventos.py:244-258`).
5. Monta e devolve o dicionário de log (`curriculo.py:127-143`). As chaves são `frac_ANDAR`, `frac_B`, `frac_C`; `s_<tarefa>_<tipo>` das 8 células; `nivel_<tarefa>` e `degrau_<tarefa>` das 4 tarefas; `subnivel_PEGAR` e `subnivel_BOTAR`; `z_<etapa>` e `taxa_<etapa>` das 7 etapas; `nivel_terreno_ANDAR` (o piso da média das linhas por coluna) e `nivel_terreno_CARREGAR`; `linha_<coluna>` para cada uma das 8 colunas de terreno do ANDAR, com a linha real (decimal) da coluna (`curriculo.py:139-140`); `atraso_max`, `mu_min_pad`, `degrau_global` e `restricao`. O mjlab guarda o dicionário em `compute` e o publica como `Curriculum/v3/<chave>` no `reset` do manager de currículo (`mjlab/managers/curriculum_manager.py:95-117`).

Consequências do item 3:

- O fecho acontece no primeiro reset depois que `common_step_counter` cruza um múltiplo de 24. Se nenhum env reseta nesse passo, o fecho espera o primeiro reset seguinte.
- Se o contador avança mais de uma iteração entre dois resets, o fecho roda uma vez só.
- Depois de uma retomada, o `common_step_counter` volta ao valor do checkpoint (`mjlab/rl/runner.py:138-140`), enquanto `v3_iter` guarda o valor da primeira chamada, que é 0 (`curriculo.py:58-59,122`). O primeiro reset depois da retomada, se o `it` calculado for diferente de 0, roda um fecho.

### 6.3 Os eventos de reset, na ordem do dicionário

O `EventManager.apply(mode="reset")` percorre os termos de modo `reset` na ordem em que o `make_env_cfg` os insere no dicionário (`mjlab/managers/event_manager.py:263,292-306`, `env_cfg.py:153-199`). Todos os termos têm `min_step_count_between_reset` = 0 (o padrão, `mjlab/managers/event_manager.py:138`), então todos disparam em todo reset de qualquer env (`event_manager.py:300-306`). Ao fim do laço, se algum termo disparado exige recálculo de constantes do modelo, `recompute_constants` roda uma vez com o nível mais forte (`event_manager.py:261,328-333`). O termo `carga` exige o nível `set_const`, então o recálculo roda em todo reset (`eventos.py:443`).

A tabela lista os 12 termos do treino. No play, saem `pd`, `forca_motor` e `solref` (seção 15). O termo `com_caixa` entra entre `solref` e `filtros` só se `DR.com_caixa` for True; o valor é False (`knobs.py:527`), então ele não existe no cfg (`env_cfg.py:182-183`).

O estado de DR que os termos leem vem de `cur.dr()` (`balanceador.py:358-389`). Três travas ligam e nunca desligam (`balanceador.py:197-202`):

- `portao0`: `nivel_passou("PEGAR") >= 0`;
- `vies`: a célula (ANDAR, `andar`) está no portão (`s >= portão`);
- `subida`: `nível PEGAR + nível CARREGAR > 0`.

O símbolo `g` abaixo é `degrau_global()`, o menor degrau entre as quatro tarefas (`balanceador.py:341-343`). `no_degrau(ini, fim, d, degraus)` vale `ini + (fim − ini) × d / degraus`, com `Balanceador.degraus` = 5 (`balanceador.py:48-51`, `knobs.py:401`).

| N | Termo | Função | O que faz e quando | Referência |
|---|---|---|---|---|
| 1 | `fatia` | `eventos.sorteia_fatia` | Sorteia a fatia do episódio (ANDAR, B ou C) por escolha multinomial com reposição, com as probabilidades de `cur.fracoes_fatias()`. Põe `v3_t_empurrao` em −∞. Sorteia se o episódio é de terreno: uniforme em [0, 1) menor que a fração da célula `terreno` (do ANDAR para a fatia ANDAR, do CARREGAR para as fatias B e C). O terreno só vale nas fatias ANDAR e B. Na fatia ANDAR, a coluna do terreno é uma das colunas 1 a 8 (`len(Terreno.colunas)` = 9, `knobs.py:434-435`), sorteada por pesos de `balanceador.fracoes` sobre o atraso da linha de cada coluna, `1 − linha/(Terreno.linhas − 1)`, com piso `Terreno.piso_coluna` = 0,05 por coluna (`knobs.py:443`); a linha é `floor(linha[coluna] + u)`, com u uniforme em [0, 1), no máximo `Terreno.linhas − 1` = 12 (`eventos.py:60-67`). Antes de sortear, o evento registra a visita e a travessia do episódio que acabou (seção 6.1). Na fatia B, a coluna é uniforme entre as 7 colunas de `Terreno.linha_max_carregar` (`knobs.py:439-440`) e a linha é uniforme de 0 até `min(nível de terreno do CARREGAR, teto da coluna)`. Escreve nível, tipo e origem do terreno do env. | `env_cfg.py:154`, `eventos.py:43-74,133-175`, `balanceador.py:282-293,302-304` |
| 2 | `cena` | `eventos.posiciona_cena` | Sorteia topo e massa da caixa e põe laje e caixa. Na fatia ANDAR, estaciona laje e caixa a `Cena.estaciona_z` = 20 m acima da origem (`knobs.py:114`). Nas fatias B e C, com `(h_k, m_k)` = `Niveis.pegar[nível do PEGAR]` e `h_0` = `Niveis.pegar[0][0]` (`knobs.py:417`), `extra` = `Niveis.subnivel_extra[subnível do PEGAR]` (`knobs.py:422`) e `m_min` = `Cena.caixa_massa_base` = 1,0 kg (`knobs.py:80`): na fatia B, topo uniforme em [`h_k + extra`; `h_0 + extra`] e massa uniforme em [`m_min`; `m_k`]; na fatia C, topo uniforme em [`max(h_k; Niveis.topo_min_c) + extra`; `h_0 + extra`] (`topo_min_c` = 0,30 m, `knobs.py:419`) e massa uniforme em [`m_min`; `m_j`], com `m_j` = `Niveis.pegar[j][1]` e `j` = `nivel_passou("PEGAR")`: o nível atual do PEGAR, se a célula `PEGAR/pegar` tem `s >= portão`, o subnível é o último e (o nível é maior que 0 ou `n_medidas > 0`); senão o nível anterior; `j` = −1 dá `m_min` (`balanceador.py:350-356`, `eventos.py:102`). Posição da caixa = `Cena.caixa_xy` (`knobs.py:96`) com deslocamento em x uniforme em [0, `Cena.jitter_x_max[nível PEGAR]`] (`knobs.py:99`) e em y uniforme em `Cena.caixa_jitter_y` (`knobs.py:97`); laje em `Cena.prateleira_xy` (`knobs.py:94`). Em episódio de terreno da fatia B, o rumo da cena é `k·π/2`, com `k` uniforme em {0, 1, 2, 3}. Publica `v3_topo`, `v3_massa`, `v3_caixa_xy` e `v3_rumo_cena`. | `env_cfg.py:156`, `eventos.py:82-109,112-130,332-367` |
| 3 | `base` | `eventos.reset_base_por_fatia` | ANDAR: chama `reset_root_state_uniform` com x e y uniformes em [−0,5; +0,5] m e yaw uniforme em [−π, π] (`Cena.reset_base_andar`, `knobs.py:103`), sobre a pose padrão e a origem do env; velocidade vale a padrão. B e C: base atrás da caixa a `d` = recuo de referência (função do topo e da meia aresta) + uniforme em [−1, 1] × `Alvos.faixa_recuo` (0,08 m, `knobs.py:227`) × `Niveis.subnivel_jitter[subnível do PEGAR]` (`knobs.py:424`); yaw = rumo da cena + uniforme em [−0,2; 0,2] rad e z somado uniforme em [0,01; 0,05] m (`Cena.reset_base_manipula`, `knobs.py:101`). Roda depois de `cena`, porque usa a posição da caixa. | `env_cfg.py:157-158`, `eventos.py:178-225`, `mjlab/envs/mdp/events.py:111-190` |
| 4 | `juntas` | `reset_joints_by_offset` (mjlab) | Põe as juntas na posição padrão (`q_de_pe`) e velocidade 0, limitando ao curso suave. As faixas de posição e velocidade são (0,0; 0,0). | `env_cfg.py:159-160`, `mjlab/envs/mdp/events.py:279-313` |
| 5 | `carga` | `eventos.carga` | Escreve a massa de `v3_massa` e a inércia `(2/3)·m·a²` (com `a` = `v3_meia`) no modelo da caixa. Não sorteia nada. | `env_cfg.py:161`, `eventos.py:443-461` |
| 6 | `atrito_pad` | `eventos.atrito_pad` | Atrito dos pads de palma por `dr.geom_friction` (eixo 0, o atrito tangencial; operação `abs`; sem `shared_random`: um sorteio uniforme por env e por geom, pelo gerador global do torch) (`mjlab/envs/mdp/dr/geom.py:118-147`, `mjlab/envs/mdp/dr/_core.py:110-134`). Com `dr()["atrito_pad"]` = `portao0` True: faixa (`mu_min_pad`, `DR.atrito_pad[1]` = 1,0), com `mu_min_pad` = `max(DR.mu_pad_min, DR.atrito_pad[0] − DR.passo_mu_pad × (nível PEGAR + nível CARREGAR))` = `max(0,5; 0,6 − 0,025 × soma)` (`knobs.py:523,525,526`). Com `portao0` False: faixa fixa (`Cena.pad_atrito[0]`, o mesmo valor) = 1,0 (`knobs.py:81`). | `env_cfg.py:162-163`, `eventos.py:464-475`, `balanceador.py:371-373` |
| 7 | `atrito_laje` | `eventos.atrito_laje` | Atrito da laje por `dr.geom_friction`, com o mesmo eixo, a mesma operação e o mesmo gerador do `atrito_pad`. Com `portao0` True: faixa `DR.atrito_laje` = (0,3; 1,0) (`knobs.py:524`). Com `portao0` False: faixa fixa `Cena.prateleira_atrito[0]` = 1,0 (`knobs.py:91`). | `env_cfg.py:164-165`, `eventos.py:478-489` |
| 8 | `vies_imu` | `eventos.vies_imu` | Com `dr()["vies_imu"]` = `vies` True: viés do giro da base uniforme em ±`DR.vies_imu_giro` = ±0,05 rad/s por eixo (`knobs.py:521`) e inclinação do IMU em x e y uniforme em ±`DR.vies_imu_incl_graus` = ±1° (`knobs.py:522`); o eixo z fica 0. Com `vies` False: giro 0 e inclinação identidade. | `env_cfg.py:166`, `eventos.py:492-509` |
| 9 | `pd` | `eventos.pd` | Fator multiplicativo (`operation="scale"`) sobre kp e kd. Sorteio uniforme pelo `v3_gerador` na faixa `dr()["pd"]` = [`no_degrau(1, 0,9, g, 5)`, `no_degrau(1, 1,1, g, 5)`] (`DR.pd`, `knobs.py:535`). O `dr.pd_gains` faz um sorteio de kp e outro, independente, de kd, cada um com forma (envs resetados, alvos do grupo de atuador) (`mjlab/envs/mdp/dr/actuator.py:30-90`). Só no treino. | `env_cfg.py:170`, `eventos.py:523-531`, `balanceador.py:385` |
| 10 | `forca_motor` | `eventos.forca_motor` | Fator (`operation="scale"`) sobre o limite de esforço. Pernas e cintura: faixa `no_degrau(1, DR.forca_motor, g, 5)` com `DR.forca_motor` = (0,9; 1,1) (`knobs.py:536`). Braços: faixa `no_degrau(1, DR.forca_motor_bracos, g, 5)` com (0,8; 1,1) (`knobs.py:537`). Sorteio uniforme pelo `v3_gerador`, um por env e por alvo do grupo de atuador (`mjlab/envs/mdp/dr/actuator.py:192-242`). Só no treino. | `env_cfg.py:173-175`, `eventos.py:534-545`, `balanceador.py:386-387` |
| 11 | `solref` | `eventos.solref` | Um valor por env, uniforme em `dr()["solref"]` = [`no_degrau(0,02, 0,015, g, 5)`, `no_degrau(0,02, 0,03, g, 5)`] (`DR.solref_base` = 0,02, `DR.solref` = (0,015; 0,03), `knobs.py:539-540`), escrito em `solref[0]` das geoms dos pés, dos pads, da laje e da caixa. Só no treino. | `env_cfg.py:177-180`, `eventos.py:548-565`, `balanceador.py:388` |
| 12 | `filtros` | `eventos.reinicia_filtros` | Reinicia o filtro de elevação, a odometria (com a pose verdadeira da base) e o estimador da caixa. Roda depois de `base`. | `env_cfg.py:184`, `eventos.py:228-241` |

Nos termos 9, 10 e 11, as faixas valem `ini` no degrau global 0 e chegam ao valor final no degrau 5.

### 6.4 O reset de cada manager

| Manager | O que o reset faz | Referência |
|---|---|---|
| Observações | Invalida o cache de observações. Chama o `reset` dos termos que são classe. Zera os buffers de atraso e de histórico dos envs resetados. | `mjlab/managers/observation_manager.py:239-259` |
| Ação | Zera `prev_action`, `prev_prev_action` e `action` dos envs. Zera as ações brutas de cada termo. | `mjlab/managers/action_manager.py:140-150`, `mjlab/envs/mdp/actions/actions.py:165-167` |
| Recompensas | Loga `Episode_Reward/<termo>` = média, sobre os envs resetados, da soma do episódio dividida por `max_episode_length_s` (20,0 s, `knobs.py:160`). Zera as somas. Chama o `reset` dos termos que são classe. | `mjlab/managers/reward_manager.py:100-114` |
| Métricas | Loga `Episode_Metrics/<termo>` conforme `reduce` (seção 16). Zera somas, máximos e contador de passos. Chama o `reset` dos termos que são classe (que logam as chaves `Episodio/...`). | `mjlab/managers/metrics_manager.py:107-133` |
| Currículo | Loga as chaves que o termo `v3` devolveu, como `Curriculum/v3/<chave>`. | `mjlab/managers/curriculum_manager.py:95-110` |
| Comandos | Para cada termo, na ordem `cadeia`, `twist`: loga as métricas do termo como `Metrics/<termo>/<métrica>` e as zera, zera `command_counter` e reamostra (`_resample`). Detalhe abaixo. | `mjlab/managers/command_manager.py:99-107,241-247`, `env_cfg.py:118-128` |
| Eventos | Chama o `reset` dos termos de evento que são classe (`forca_caixa`). Esse `reset` zera a força aplicada e sorteia de novo o tempo de espera até o próximo impulso. Não reinicia o relógio de `empurrao`, que é uma função. | `mjlab/managers/event_manager.py:217-234`, `mjlab/envs/mdp/events.py:584-605` |
| Terminações | Loga `Episode_Termination/<termo>` = número de envs resetados em que o termo disparou. Chama o `reset` dos termos que são classe. | `mjlab/managers/termination_manager.py:88-100` |

**Reset do termo `cadeia`.** A classe usa o `reset` do mjlab, que chama `_resample` (`mjlab/managers/command_manager.py:99-107`). O `_resample` da `CadeiaCmd` roda assim (`comando.py:639-660`):

0. O `reset` herdado do `CommandTerm` não loga métrica (o `cadeia` não tem métrica), zera `command_counter` e chama `_resample` (`mjlab/managers/command_manager.py:99-107`).
1. Registra no balanceador o PEGAR e o BOTAR que estão abertos nos envs resetados. O resultado conta como falha quando o episódio acabou por terminação que não é `time_out` (`falhou = ~time_outs`) (`comando.py:641-645`, `comando.py:521-540`).
2. Para cada visita de zona aberta, soma 1 às visitas da etapa, sem somar fecho (`comando.py:646-650`).
3. Zera os buffers `v3_pegou`, `v3_perdeu`, `v3_pega_fechada`, `v3_bota_fechada`, `v3_perdeu_agora`, `v3_avancou`, `v3_aviso` e `v3_fim` (`comando.py:651-653`).
4. Põe `v3_relogio_alvo` em 0, `v3_sigma` em 1, `v3_phi` em −1, `v3_h` em 1, `v3_phi_melhor` em 0 e `v3_v_chegada` em −1 (`comando.py:654-658`).
5. Abre o trecho inicial da fatia do env com `_abre_trecho` (`comando.py:659-660`). Esse método sorteia a duração do trecho, publica a tarefa e, no PEGAR e no ANDAR, deixa o alvo em −1 (`comando.py:568-594`). A publicação chama `_publica`, que escreve `v3_tarefa`, `v3_freio` e `v3_degrau` e chama o `_resample` do termo `twist` nos mesmos envs (`comando.py:508-519`). Esse `_resample` sorteia `time_left`, o comando de velocidade e soma 1 em `command_counter` do `twist`.

**Reset do termo `twist`.** O `reset` do `TwistV3` (`comando.py:879-895`):

1. Fecha o segmento aberto de cada env resetado, com falha quando o episódio acabou por terminação que não é `time_out` (`comando.py:882-883`). O fecho registra a nota do segmento na célula do currículo (`comando.py:759-794`).
2. Zera o relógio `_agora`, a fase `_fase_t`, o filtro `v_filt`, o acerto de marcha e suas somas. Põe `q_filt` em 0,5 e `g_marcha` em 0 (`comando.py:884-892`).
3. Chama o `reset` do mjlab (zera as métricas e o contador e chama `_resample`, que sorteia o comando e o `time_left`) (`comando.py:893`, `mjlab/managers/command_manager.py:99-107,117-129`).
4. Abre o segmento novo (`comando.py:894`, `comando.py:735-751`).

O comando de velocidade do episódio novo é sorteado duas vezes, as duas depois de `v3_tarefa`, `v3_fatia` e a célula do trecho já estarem escritos: a primeira dentro do reset da `cadeia` (item 5 acima) e a segunda dentro do `super().reset` do `twist` (item 3). O segundo sorteio sobrescreve o primeiro e vale para o episódio. O `super().reset` do `twist` zera `command_counter` antes do segundo sorteio, que o deixa em 1, e o `_abre` guarda esse valor em `seg_visto`.

A pose usada no fecho do segmento é `pose_ult`, gravada no `compute` do passo anterior. Ela existe porque o `_reset_idx` teletransporta o robô (item 4 da tabela 6, evento `base`), e o reset dos comandos vem depois (`comando.py:730-732,814-815`).

### 6.5 Os eventos de startup (uma vez, na criação do env)

Rodam no fim da criação do env, na ordem do dicionário (`manager_based_rl_env.py:352-354`, `env_cfg.py:185-198`).

| Termo | O que faz | Referência |
|---|---|---|
| `atrito_pe` | Atrito dos 14 geoms dos pés (7 por pé), na faixa `DR.atrito_pe` = (0,3; 1,0) (`knobs.py:508`), com `operation="abs"` e `shared_random=True`. O `dr.geom_friction` sorteia só o eixo 0 (atrito tangencial): um valor uniforme na faixa por env, igual para os 14 geoms, pelo gerador global do torch (`mjlab/envs/mdp/dr/geom.py:118-147`, `mjlab/envs/mdp/dr/_core.py:115-125`). | `env_cfg.py:60,185-187` |
| `massa_tronco` | Soma uma massa ao corpo `torso_link`, na faixa `DR.massa_tronco` = (−1,0; 3,0) kg, operação `add` (`knobs.py:509`). O `dr.body_mass` sorteia um valor uniforme por env, pelo gerador global do torch, e muda só a massa, sem mudar a inércia (`mjlab/envs/mdp/dr/body.py:285-325`). | `env_cfg.py:188-190` |
| `vies_encoder` | Viés de encoder, na faixa ±`DR.vies_encoder` = ±0,015 rad (`knobs.py:510`), pelo `dr.encoder_bias`, que sorteia um valor uniforme por env e por junta (29) e o guarda em `encoder_bias` do robô (`mjlab/envs/mdp/dr/joint.py:209-241`). O termo de ação subtrai esse viés do alvo (seção 5.1). | `env_cfg.py:191-192` |
| `tamanho_caixa` | Cada env recebe uma meia aresta da caixa entre 8 valores igualmente espaçados (`Cena.caixa_n_variantes`, `knobs.py:79`) de `Cena.caixa_meia_faixa` = (0,07; 0,13) m (`knobs.py:78`). A escolha é uniforme entre os 8 valores, pelo `v3_gerador`. Grava `v3_meia`. O valor fica fixo até o fim do treino. | `env_cfg.py:193-195`, `eventos.py:417-440` |
| `atraso` | Põe `max_lag` nos buffers de atraso: passos de física no atuador = `DR.atraso_inicio[1]` × decimação = 0 × 4 e passos de política na observação = `DR.atraso_inicio[1]` = 0 (`knobs.py:511`). A capacidade dos buffers não muda. | `env_cfg.py:197-198`, `eventos.py:244-258` |

O atraso real cresce depois. O termo `v3` chama `eventos.atraso` com o teto de `dr()["atraso_max"]` (seção 6.2). Esse teto vale `DR.atraso_refino[1]` = 3 quando `degrau_global() >= DR.degrau_atraso_refino` = 3 (`knobs.py:513-514`). Senão, vale `ceil(DR.atraso_portao[1] × restricao() − 10⁻⁹)` com `DR.atraso_portao[1]` = 2 (`knobs.py:512`, `balanceador.py:381-382`). A função `restricao()` é o menor nível relativo entre as quatro tarefas (`balanceador.py:345-348`).

### 6.6 Os eventos de passo e de intervalo (só no treino)

| Termo | Modo | O que faz | Referência |
|---|---|---|---|
| `forca_caixa` | `step` | Impulso de força na caixa. Só age quando `dr()["forca_caixa"]` = `subida` é True; com a trava False, o termo retorna antes de chamar o mjlab, e o relógio dos impulsos não corre (`eventos.py:594-597`). A espera inicial de cada env é sorteada no construtor do termo (`mjlab/envs/mdp/events.py:456-461`). A cada env: espera sorteada uniforme em `DR.forca_caixa_periodo_s` = (3,0; 8,0) s, depois força com cada componente uniforme em `DR.forca_caixa_N` = (−11,5; 11,5) N, que dura um tempo uniforme em `DR.forca_caixa_dur_s` = (0,1; 0,3) s (`knobs.py:531-533`). Depois do impulso do mjlab, o termo zera a força nos envs que não estão em CARREGAR com `v3_pegou` True e `v3_perdeu` False. | `env_cfg.py:207-209`, `eventos.py:584-602`, `mjlab/envs/mdp/events.py:386-552` |
| `rfi` | `step` | Escreve um torque de ruído em cada junta de cada env, uniforme em ±`s` × `TAU_MAX` da junta (o torque máximo por junta, montado de `Recompensa.tau_max`, `knobs.py:286-292,572`). `s` = `dr()["rfi"]` = `no_degrau(DR.rfi[0], DR.rfi[1], g, 5)` com `DR.rfi` = (0,05; 0,1) (`knobs.py:534`) se a trava `subida` é True, senão 0. Roda em todo passo de controle, e o valor vale nos 4 subpassos. | `env_cfg.py:210`, `eventos.py:605-615`, `balanceador.py:384` |
| `empurrao` | `interval` | Soma à velocidade da base uma amostra uniforme em x e em y na faixa ±`DR.empurrao` = ±0,5 m/s (`knobs.py:517`) e grava o instante do empurrão em `v3_t_empurrao`. O intervalo do env é uniforme em `DR.periodo_empurrao_s` = (3,0; 7,0) s (`knobs.py:520`). O relógio nasce sorteado na criação do manager, é sorteado de novo quando o termo dispara e não reinicia no reset do episódio. ⚠ Divergência: o docstring de `EventTermCfg.is_global_time` diz que o relógio de cada env reinicia no reset do episódio; o código só reinicia os termos que são classe (`event_manager.py:133-136,217-234`). | `env_cfg.py:201-205`, `eventos.py:261-269`, `mjlab/managers/event_manager.py:217-234,265-288,361-375`, `mjlab/envs/mdp/events.py:316-337` |

## 7. Como a recompensa é somada e multiplicada pelo `dt`

O `reward_manager.compute(dt)` roda no item 6 de um passo, com `dt` = `step_dt` = 0,02 s (`manager_based_rl_env.py:440`). A conta, termo a termo, na ordem do dicionário (`mjlab/managers/reward_manager.py:116-133`):

1. Zera o buffer de recompensa (`reward_manager.py:117`).
2. Define `scale = dt` porque `scale_rewards_by_dt` vale True (`reward_manager.py:118`, `manager_based_rl_env.py:157,329-331`).
3. Um termo com `weight` igual a 0 é pulado (`reward_manager.py:122-124`).
4. `valor = func(env, **params)` e a checagem de forma `(num_envs,)` (`reward_manager.py:125-126`).
5. `valor = valor × weight × dt` (`reward_manager.py:127`).
6. `nan_to_num` troca NaN e infinitos por 0 (`reward_manager.py:129`).
7. Soma `valor` no buffer da recompensa e na soma do episódio do termo (`reward_manager.py:130-131`).

A recompensa do passo é a soma dos termos. O PPO recebe esse total (`manager_based_rl_env.py:440`, `mjlab/rl/vecenv_wrapper.py:77-88`). Nenhum código lido normaliza ou limita a recompensa.

O `make_env_cfg` monta um `RewardTermCfg` por termo, com `func = recompensas.<nome>`, `weight` = campo de mesmo nome em `Recompensa` e `params = {"tarefas": ...}` (`env_cfg.py:131-134`). Duas exceções: `progresso` tem peso 1,0 e `perda_caixa` tem peso −1,0, fixos no cfg (`env_cfg.py:57`).

Os termos que usam `_fim` valem 0 fora das suas tarefas. `_fim` aplica `torch.where(mascara, valor, 0)` (`recompensas.py:361-365`), com a máscara de `curriculo.mascara(env, tarefas)`: True onde `v3_tarefa` está em `tarefas` (`curriculo.py:93-100`). Com `paga=True`, `_fim` também zera o valor nos envs com `v3_perdeu` True (`recompensas.py:363-364`). Os termos `perda_caixa` e `terminacao` chamam `_fim` sem `paga` (`recompensas.py:962,968`).

⚠ Divergência: o docstring de `n_terminacao` diz peso −200 no knob; o código lê `Recompensa.terminacao` = −300,0 (`recompensas.py:215`, `knobs.py:366`).

Os grupos de tarefas são `TODAS` = (ANDAR, PEGAR, CARREGAR, BOTAR), `ANDAR_CARREGAR`, `PEGAR_BOTAR` e `PEGAR_CARREGAR_BOTAR` (`env_cfg.py:33-36`).

Exemplos de escala. Qualquer terminação que não é `time_out` (`queda`, `caixa_perdida`, `chegada` ou `batente`) paga `1 × (−300) × 0,02` = −6,0 no passo em que dispara (`terminacao` = −300,0, `knobs.py:366`; valor 1 quando `terminated` é True e `time_outs` é False; se as duas valem True no mesmo passo, o valor é 0, `recompensas.py:214-216,965-968`). A perda da caixa paga `240 × (−1,0) × 0,02` = −4,8, no passo em que o pulso `v3_perdeu_agora` é lido (`perda_caixa` = 240,0, `knobs.py:367`, `recompensas.py:311-313,960-962`).

Os 33 termos, na ordem do dicionário (`env_cfg.py:42-54`):

| Termo | Tarefas | Peso no cfg | Referência do peso |
|---|---|---|---|
| `rastreio_linear` | ANDAR, CARREGAR | `Recompensa.rastreio_linear` = 2,0 | `knobs.py:236` |
| `rastreio_angular` | ANDAR, CARREGAR | `rastreio_angular` = 1,0 | `knobs.py:237` |
| `passo` | ANDAR, CARREGAR | `passo` = 0,5 | `knobs.py:244` |
| `vivo` | ANDAR, CARREGAR | `vivo` = 1,0 | `knobs.py:248` |
| `altura_pe` | ANDAR, CARREGAR | `altura_pe` = 1,0 | `knobs.py:249` |
| `escorrego` | todas | `escorrego` = −0,2 | `knobs.py:253` |
| `vz` | todas | `vz` = −2,0 | `knobs.py:254` |
| `giro_xy` | todas | `giro_xy` = −0,05 | `knobs.py:255` |
| `vel_junta` | todas | `vel_junta` = −0,001 | `knobs.py:256` |
| `acel_junta` | todas | `acel_junta` = −5e-8 | `knobs.py:257` |
| `taxa_acao` | todas | `taxa_acao` = −0,01 | `knobs.py:258` |
| `energia` | todas | `energia` = −2e-5 | `knobs.py:259` |
| `limite_junta` | todas | `limite_junta` = −5,0 | `knobs.py:262` |
| `desvio_bracos` | ANDAR, CARREGAR | `desvio_bracos` = −0,1 | `knobs.py:264` |
| `desvio_cintura` | ANDAR, CARREGAR | `desvio_cintura` = −1,0 | `knobs.py:265` |
| `desvio_quadril` | ANDAR, CARREGAR | `desvio_quadril` = −1,0 | `knobs.py:266` |
| `tronco_plano` | ANDAR, CARREGAR | `tronco_plano` = −20,0 | `knobs.py:267` |
| `altura_base` | ANDAR, CARREGAR | `altura_base` = −10,0 | `knobs.py:268` |
| `cerca_sola` | PEGAR, BOTAR | `cerca_sola` = −0,05475 | `knobs.py:270` |
| `cerca_giro` | todas | `cerca_giro` = −0,03965 | `knobs.py:271` |
| `cerca_tronco` | todas | `cerca_tronco` = −0,04596 | `knobs.py:272` |
| `cerca_com` | todas | `cerca_com` = −22,09035 | `knobs.py:273` |
| `cerca_torque` | todas | `cerca_torque` = −0,29615 | `knobs.py:274` |
| `cerca_pelve` | todas | `cerca_pelve` = −4,45279 | `knobs.py:275` |
| `cerca_recuo` | PEGAR, CARREGAR, BOTAR | `cerca_recuo` = −10,39719 | `knobs.py:276` |
| `contato_fora` | todas | `contato_fora` = −1,0 | `knobs.py:299` |
| `contato_laje` | PEGAR, BOTAR | `contato_laje` = −2,0 | `knobs.py:301` |
| `freio` | PEGAR, CARREGAR, BOTAR | `freio` = −6/29 | `knobs.py:304` |
| `progresso` | todas | 1,0 fixo no cfg | `env_cfg.py:57` |
| `sacudida` | PEGAR, CARREGAR, BOTAR | `sacudida` = −0,05 | `knobs.py:353` |
| `aperto` | PEGAR, CARREGAR, BOTAR | `aperto` = −0,05 | `knobs.py:355` |
| `perda_caixa` | todas | −1,0 fixo no cfg (o custo 240,0 está dentro do termo) | `env_cfg.py:57`, `knobs.py:367` |
| `terminacao` | todas | `terminacao` = −300,0 | `knobs.py:366` |

A fórmula interna de cada termo não pertence a este documento.

## 8. As terminações

O episódio de um env acaba quando `reset_buf` é True, ou seja, quando algum termo dispara (`mjlab/managers/termination_manager.py:102-113`). Os termos, na ordem do dicionário (`env_cfg.py:137-147`):

| Termo | Conta como | Tarefas | Condição exata | Referência |
|---|---|---|---|---|
| `time_out` | truncamento (`time_out=True`) | todas, sem máscara | `episode_length_buf >= max_episode_length` ou `v3_fim` True ou travessia. A travessia vale só no ANDAR em terreno (`v3_fatia` = ANDAR e coluna ≠ 0): a distância no plano xy entre o `pose_ult` do `twist` e a origem da célula passa de `Terreno.dist_travessia` = 3,5 m (`knobs.py:442`), com `episode_length_buf > 2`. O término sai um passo depois da posição que passou, porque o `pose_ult` é o do passo anterior. `max_episode_length` vale 1000 no treino. | `terminacoes.py:53-72`, `env_cfg.py:138` |
| `queda` | terminação | todas | Por tarefa (abaixo). | `terminacoes.py:35-50,86-112`, `env_cfg.py:140` |
| `caixa_perdida` | terminação | PEGAR, CARREGAR, BOTAR | `v3_perdeu_agora` True e `Cadeia.caixa_perdida_vira_andar` False. O valor é True (`knobs.py:161`), então o termo vale sempre False no cfg atual. | `terminacoes.py:75-83,115-125`, `env_cfg.py:141-142` |
| `chegada` | terminação | BOTAR | Primeiro contato do sensor `apoio_caixa` e `−v3_vz_caixa_ant > limite`. O limite vale `1,0 + (0,25 − 1,0) × v3_degrau / 5` m/s (`Terminacao.chegada` = (1,0; 0,25), `knobs.py:381`; `Balanceador.degraus` = 5, `knobs.py:401`). | `terminacoes.py:128-151`, `recompensas.py:447-450`, `balanceador.py:48-51`, `env_cfg.py:144` |
| `batente` | terminação | todas | Alguma das 29 juntas com `\|q − centro\| / meio > Terminacao.batente_frac` (1,0, `knobs.py:383`; `centro` e `meio` são a média e a meia diferença de `joint_pos_limits`, `terminacoes.py:177-180`), ou `\|hip_yaw\|` de qualquer perna maior que `Terminacao.hip_yaw_max` (1,0 rad, `knobs.py:382`), e `v3_degrau >= Refino.degrau_batente` (1, `knobs.py:547`; `v3_degrau` é o degrau da tarefa publicada, `comando.py:512-518`). Os comparadores dos limites são estritos (`>`); o do degrau é `>=`. | `terminacoes.py:154-183`, `potencial.py:70-75`, `env_cfg.py:146` |

A condição de `queda`, por tarefa (`terminacoes.py:35-50`). A inclinação é `incl = acos(clamp(−projected_gravity_b[z], −1, 1))` (`terminacoes.py:97`).

| Tarefa | Condição de queda | Knobs |
|---|---|---|
| ANDAR e CARREGAR | `incl > Terminacao.incl_marcha` ou `z_pelve − z_chão_da_pelve < Terminacao.base_min` | `incl_marcha` = 0,8 rad (`knobs.py:375`); `base_min` = 0,20 m (`knobs.py:376`) |
| PEGAR e BOTAR | `incl > radianos(Terminacao.incl_manip_graus)` ou `z_joelho − z_chão_do_joelho < Terminacao.joelho_min`, com o joelho mais baixo em relação ao terreno | `incl_manip_graus` = 70° (`knobs.py:377`); `joelho_min` = 0,05 m (`knobs.py:378`) |
| REORIENTAR e LEVANTAR | sempre False | `terminacoes.py:44-50` |

O `z` de chão vem do terreno sob a pelve e sob os joelhos, pelos sensores `altura_pelve` e `altura_joelhos` (`terminacoes.py:99-107`).

Cada termo aplica a máscara de tarefas com `torch.where` (`terminacoes.py:111-112,124-125,150-151,182-183`).

O PPO trata `time_out` e as outras terminações de forma diferente (seção 10.2). A recompensa `terminacao` paga só nas terminações que não são `time_out` (`recompensas.py:214-216`).

## 9. O currículo dentro do ciclo

O objeto de currículo `v3_cur` (classe `balanceador.Curriculo`) guarda o estado do currículo: células, níveis, fatias, degraus de refino e travas de DR. O env cria o objeto uma vez, a partir dos knobs `ATIVO` do módulo (`curriculo.py:46-47`).

### 9.1 Quem chama o fecho da iteração e quando

Só o termo `v3` chama `cur.fecha_iteracao()` (`curriculo.py:120-121`). O manager de currículo chama o termo no item 1 de todo `_reset_idx` (seção 6). A condição exata: `v3_iter >= 0` e `common_step_counter // 24 != v3_iter`. O PPO e o runner não chamam o fecho. O divisor 24 é `Balanceador.passos_por_iteracao` (`knobs.py:402`), o mesmo valor de `num_steps_per_env` do PPO (`tarefa.py:49`). Assim, o fecho acompanha as iterações do PPO enquanto o contador de passos do env avança 24 por iteração.

O contador `common_step_counter` conta passos de controle do env inteiro, não de um env individual (`manager_based_rl_env.py:431`). O fecho roda no primeiro reset de qualquer env depois que o contador cruza um múltiplo de 24 (seção 6.2).

No play, o termo `v3` não existe (seção 15). Nenhum fecho roda.

### 9.2 O que alimenta o fecho

O fecho consome somas que o env registra durante a iteração, com `cur.registra(célula, n, soma)`. A função só acumula `n` medidas e a `soma` das notas na célula (`balanceador.py:148-151`). Duas rotinas do env registram:

- **`TwistV3._fecha`** registra os segmentos de comando do ANDAR e do CARREGAR. Ela roda na fronteira de segmento, quando o contador de comandos muda (`comando.py:818-821`), e no reset do env (`comando.py:883`). Num segmento de tarefa com caixa (`seg_carrega`, tarefa diferente de ANDAR na abertura), a nota é o sucesso 0/1 de `sucesso_do_segmento`, e o segmento registra se fecha depois da âncora ou se a falha vale (`comando.py:227-244,769-790`). Na fronteira de segmento, a falha vale quando `v3_perdeu` e `seg_carrega` são True; no reset do env, quando o episódio acabou por terminação que não é `time_out` (`comando.py:820,882-883`). Num segmento do ANDAR, a nota é a média de dois canais, linear e giro. O canal comandado (norma do comando maior ou igual a `Comando.limiar_canal` = 0,02, `knobs.py:138`) vale `clamp(razão, 0, 1)`, com a razão entre a velocidade média projetada no comando e a norma do comando. O canal sem comando vale `exp(−‖v̄‖² / σ²)`, com σ = `Recompensa.sigma_rastreio` = 0,5 (`knobs.py:238`). Uma terminação que não é `time_out` zera a nota. O segmento do ANDAR registra se tem duração depois da âncora (`seg_dur > 0`) ou se a falha vale (`comando.py:774-790`).
- **`CadeiaCmd._registra`** registra o PEGAR e o BOTAR da cadeia. O PEGAR registra na abertura do BOTAR e no reset. O BOTAR registra no fim da cadeia e no reset (`comando.py:521-540,589,615-616,641-645`). O sucesso do PEGAR é `v3_pega_fechada & ~falhou & ~v3_perdeu`, e o do BOTAR é `v3_bota_fechada & ~falhou & ~v3_perdeu` (`comando.py:68-70,91-93`). Os dois exigem também que todas as zonas da tarefa valham 0: `max(z[:4]) == 0` para o PEGAR e `max(z[4:]) == 0` para o BOTAR (`comando.py:533-537`). Sem isso, o episódio registra como falha.

### 9.3 O que o fecho faz

`fecha_iteracao` executa, nesta ordem (`balanceador.py:153-195`):

1. `it += 1` (`balanceador.py:162`).
2. Atualiza a nota `s` de cada célula com medidas na iteração. A nota da célula `ANDAR/terreno` segue no refino, mas o déficit dela, que pesa o sorteio das células, é `1 − média das linhas / teto` (`balanceador.py:136-144`). Com `n = _acum_n[i] > 0`, calcula `x = _acum_suc / n`. Se a célula ainda não tem medida (`n_medidas == 0`), `s = x`. Senão, `s = s + alfa × (x − s)`, com `Balanceador.alfa` = 0,05 (`knobs.py:394`). Soma 1 em `n_medidas` e zera os acumuladores. Uma célula sem medidas na iteração não muda (`balanceador.py:163-168`).
3. Atualiza as zonas. Para cada etapa `e` com `vis_acum[e] >= Balanceador.zona_n_min` = 200 (`knobs.py:406`): `taxa[e] = fech_acum[e] / vis_acum[e]`, `z[e] = clip(z[e] − zona_ganho × (taxa[e] − zona_t_alvo), 0, 1)` com `zona_ganho` = 0,05 e `zona_t_alvo` = 0,6 (`knobs.py:404-405`), e zera os dois acumuladores (`balanceador.py:170-174`). Depois, para as etapas de índice 1, 2, 3, 5 e 6, `z[e] = max(z[e], z[e−1] − zona_folga_cadeia)` com `zona_folga_cadeia` = 0,2 (`knobs.py:408`, `balanceador.py:175-176`).
4. Move a linha de terreno de cada coluna do ANDAR (as colunas 1 a 8). Para cada coluna `c` com `vis_ter[c] >= Balanceador.linha_n_min` = 50 (`knobs.py:410`): `taxa = trav_ter[c] / vis_ter[c]` e `linha[c] = clip(linha[c] + linha_ganho × (taxa − zona_t_alvo), 0, max_ter)`, com `linha_ganho` = 2,5 (`knobs.py:411`), `zona_t_alvo` = 0,6 e `max_ter` o teto de linhas do ANDAR. Depois zera `vis_ter[c]` e `trav_ter[c]`. Uma coluna com menos visitas não muda e guarda as visitas para a iteração seguinte (`balanceador.py:177-182`). O `eventos.sorteia_fatia` preenche `vis_ter` e `trav_ter` (seção 6.1).
5. Conta o refino de cada tarefa (`_conta_refino`, `balanceador.py:183-184,234-248`). Enquanto `degrau < Balanceador.degraus` (5, `knobs.py:401`), se a tarefa está no último nível (e, nas tarefas de cadeia, com a célula `terreno` aberta no último nível de terreno) e a média das notas das células abertas é maior ou igual à média dos portões, `cont += 1`. Senão, `cont = 0`. Com `cont >= Balanceador.iters_refino` (100, `knobs.py:400`), `degrau += 1` e `cont = 0`.
6. Abre células (`_abre`, `balanceador.py:185,204-232`). A fatia B abre (células `CARREGAR/reto` e `PEGAR/pegar`) quando o nível do ANDAR é maior ou igual a `Balanceador.nivel_abre_b` = 5 (`knobs.py:396`). Cada tarefa de cadeia abre a próxima célula de velocidade fechada quando já tem alguma aberta e a média das notas das abertas passa no portão. A célula `terreno` de uma tarefa de cadeia abre quando a célula não está aberta, a média das notas das abertas passa no portão e o nível da tarefa vale o mínimo: no ANDAR, `nivel_abre_terreno` = 2 (`knobs.py:412`); no CARREGAR, o nível máximo. A nota da célula `terreno` do ANDAR não entra nessa regra. O CARREGAR exige também `nivel_terreno("ANDAR")` ≥ 1, o piso da média das linhas por coluna (`balanceador.py:219-229`). A célula `BOTAR/botar` abre quando a célula `PEGAR/pegar` passa no portão (`s >= portão`).
7. Marca o fecho em que a fatia B e a fatia C abriram (`_it_abertura`), e sobe a rampa da fatia: com `d = it − _it_abertura[f]`, se `d > 0` e `d % Balanceador.rampa_fatia_its == 0` (12, `knobs.py:399`), `_rampa[f] += 1` (`balanceador.py:186-193`).
8. Sobe níveis (`_sobe`, `balanceador.py:194,250-280`). Para cada tarefa, `ok` indica que as células de velocidade estão abertas e no portão (média), ou, no nível máximo do CARREGAR com linha de terreno abaixo do teto, que a célula `terreno` está aberta e no portão. O ANDAR não sobe linha aqui: a linha dele sobe por coluna no item 4. Com `ok`, `seguidas += 1`; senão, `seguidas = 0`. Com `seguidas >= Balanceador.iters_nivel` (50, `knobs.py:395`), a tarefa sobe um subnível, um nível ou uma linha de terreno. A subida de subnível ou de nível zera `n_medidas` das células da tarefa. A subida de linha de terreno (só o CARREGAR) zera só a da célula `terreno`.
9. Liga as travas de DR (`_trava_dr`, `balanceador.py:195,197-202`), descritas na seção 6.3.

⚠ Divergência: o comentário de `balanceador.py:96` diz portão de 0,80 para o ANDAR no plano e o de `balanceador.py:71` diz 10 níveis para o ANDAR; o código lê `Balanceador.portao_andar` = 0,75 (`knobs.py:392`) e calcula 7 níveis (`balanceador.py:72`).

O portão de cada célula é `Balanceador.portao_andar_terreno` = 0,50 para `ANDAR/terreno`, `Balanceador.portao_andar` = 0,75 para `ANDAR/andar` e `Balanceador.portao` = 0,75 para as demais (`knobs.py:391-393`, `balanceador.py:97-98`). O nível máximo do ANDAR é `round((Comando.frente_tetos[-1] − Comando.andar_inicio) / Comando.andar_passo)` = `round((1,0 − 0,3) / 0,1)` = 7 (`knobs.py:122,129,134`, `balanceador.py:72`). O nível máximo do CARREGAR é `len(Comando.frente_tetos) − 1` = 4. O do PEGAR e o do BOTAR é `len(Niveis.pegar) − 1` e `len(Niveis.botar) − 1`, ambos 4 (`knobs.py:417-418,122`, `balanceador.py:73-74`).

## 10. Uma iteração do PPO

A classe `OnPolicyRunner.learn` executa o laço (`rsl_rl/runners/on_policy_runner.py:56-134`). O runner da v3 (`RunnerV3`) herda esse laço sem mudá-lo.

### 10.1 A coleta (rollout)

Para cada iteração `it` de `start_it` até `start_it + max_iterations − 1` (`on_policy_runner.py:79`), sob `torch.inference_mode()` (`on_policy_runner.py:82`), repete `num_steps_per_env` = 24 vezes (`on_policy_runner.py:83`):

1. `actions = alg.act(obs)` (`on_policy_runner.py:85`). O ator amostra uma ação `Normal(média, std)`, e o crítico calcula o valor. O `act` guarda ação, valor, log-probabilidade, parâmetros da distribuição e as observações anteriores ao passo (`rsl_rl/algorithms/ppo.py:115-126`).
2. `env.step(actions)` (`on_policy_runner.py:87`). A seção 5 detalha o passo. As observações devolvidas já são as de depois do reset para os envs que acabaram.
3. `check_nan(obs, rewards, dones)` (`on_policy_runner.py:89-90`; a chave `check_for_nan` não existe no cfg, então vale o padrão True). A função levanta `ValueError` se algum grupo de observação ou a recompensa tem NaN (`rsl_rl/utils/utils.py:275-287`).
4. `alg.process_env_step(obs, rewards, dones, extras)` (`on_policy_runner.py:94`). A seção 10.2 detalha.
5. `logger.process_env_step(rewards, dones, extras, ...)` (`on_policy_runner.py:98`). Guarda o `extras["log"]` do passo e acumula recompensa e comprimento dos episódios (`rsl_rl/utils/logger.py:121-149`).

Depois dos 24 passos, ainda sob `inference_mode`, chama `alg.compute_returns(obs)` (`on_policy_runner.py:105`), com as últimas observações.

### 10.2 `process_env_step` (versão da v3)

O `PPOPorTarefa.process_env_step` (`algoritmo.py:114-135`) faz, em ordem:

1. Se `fatia_alvo` já foi definida pelo runner e `v_grupo` já existe (isto é, a partir da segunda iteração, porque `v_grupo` só nasce em `compute_returns`), calcula o grupo `g` de cada env a partir das observações do crítico guardadas no `act` (`algoritmo.py:127-128`).
2. Calcula `ruim = (dones > 0 e não time_out) ou extras["v3_perdeu_agora"]`, com `time_outs = extras.get("time_outs", zeros)` (`algoritmo.py:129-130`).
3. Soma à recompensa `gamma × min(v_grupo[g], 0) × ruim`, com `gamma` = 0,99 (`algoritmo.py:131`). `v_grupo[g]` é a média do valor do grupo no rollout anterior (seção 10.3). Com `v_grupo[g] >= 0`, a soma é 0.
4. Chama `PPO.process_env_step` (`algoritmo.py:132`). Esse método (`ppo.py:128-161`):
   - atualiza os normalizadores do ator e do crítico com `obs` (as observações pós-passo) (`ppo.py:133-134`);
   - guarda `rewards` e `dones` na transição (`ppo.py:140-141`);
   - soma `gamma × valor × time_outs` à recompensa guardada, quando `extras` tem `time_outs` (`ppo.py:150-155`);
   - grava a transição no armazenamento, limpa a transição e chama `reset(dones)` no ator e no crítico, que não fazem nada nas redes sem memória (`ppo.py:158-161`, `mlp_model.py:121-123`).
5. Chama `fixa_normalizador` no normalizador do ator e no do crítico, com as fatias fixas de cada rede (`algoritmo.py:133-134`).

O pulso `extras["v3_perdeu_agora"]` que o PPO lê é o escrito pelo comando do passo corrente (seção 5, item 10). A recompensa `perda_caixa` e a terminação `caixa_perdida` leem o pulso do passo anterior (seção 5.4).

### 10.3 `compute_returns` (versão da v3)

O `PPOPorTarefa.compute_returns` (`algoritmo.py:137-169`) executa:

1. Chama `PPO.compute_returns` (`algoritmo.py:144`). Esse método (`ppo.py:163-188`) calcula `last_values = critic(obs)` e, de trás para frente nos 24 passos, `delta = r + (1 − done) × gamma × V(próximo) − V`, `vantagem = delta + (1 − done) × gamma × lambda × vantagem` e `retorno = vantagem + V`, com `gamma` = 0,99 e `lambda` = 0,95 (`ppo.py:173-183`). Um `done` (terminação ou `time_out`) corta a soma. O `time_out` entra no valor pela recompensa de bootstrap (seção 10.2). Depois normaliza a vantagem pela média e pelo desvio do lote inteiro (`ppo.py:185-188`).
2. Confere que as 6 últimas colunas da observação do crítico somam 1 em cada amostra, com tolerância de 1e-3 (`algoritmo.py:149-153`).
3. Recalcula a vantagem crua `retornos − valores` (`algoritmo.py:157`) e o grupo de cada amostra (`algoritmo.py:158`).
4. Substitui `st.advantages` pela vantagem normalizada por grupo (`algoritmo.py:159`).
5. Se `fatia_alvo` existe, atualiza `v_grupo`: soma e conta os valores do rollout por grupo, e `v_grupo` vira `soma / contagem` nos grupos com amostra e mantém o valor anterior nos demais (`algoritmo.py:160-169`). O vetor `v_grupo` tem `1 + 6 × n` posições, com `n` = largura da fatia `alvo_macro` do crítico = 11 (`algoritmo.py:162-163`, `observacoes.py:46`).

### 10.4 `update`

O `PPO.update` (`ppo.py:190-344`) roda 5 épocas de 4 minibatches. Os dados são achatados em `(passos × envs)`. Um `randperm` sorteia os índices uma vez por chamada, e as 5 épocas reusam a mesma divisão em 4 minibatches de `passos × envs / 4` amostras (`rsl_rl/storage/rollout_storage.py:225-262`). Para cada minibatch:

1. Recalcula as médias da distribuição, o log-prob das ações guardadas, a entropia e os valores (`ppo.py:221-231`).
2. Ajusta a taxa de aprendizado (agenda `adaptive`, seção 11.2) (`ppo.py:233-259`).
3. `razão = exp(log-prob − log-prob antigo)`. A perda do ator é `mean(max(−A × razão, −A × clamp(razão, 1 − clip, 1 + clip)))` com `clip` = 0,2 (`ppo.py:262-267`).
4. A perda do crítico, com `use_clipped_value_loss` True, é `mean(max((V − R)², (V_antigo + clamp(V − V_antigo, −clip, +clip) − R)²))` (`ppo.py:270-274`).
5. `perda = perda_ator + value_loss_coef × perda_crítico − entropy_coef × média(entropia)`, com `value_loss_coef` = 1,0 e `entropy_coef` = 0,001 (`ppo.py:278`).
6. Zera os gradientes, `backward`, corta a norma do gradiente dos parâmetros do ator e, separadamente, dos do crítico em `max_grad_norm` = 1,0 e dá o passo do Adam (`ppo.py:290-304`).

Ao fim, devolve as médias de `value`, `surrogate` e `entropy` sobre `num_learning_epochs × num_mini_batches` = 20 passos (`ppo.py:310-335`) e limpa o armazenamento (`ppo.py:342`).

### 10.5 Registro e salvamento

Depois do `update`, o runner chama `logger.log(...)` (`on_policy_runner.py:115-125`) e, se o escritor existe e `it % save_interval == 0`, chama `save(<pasta>/model_<it>.pt)` (`on_policy_runner.py:128-129`). Ao fim das iterações, salva `model_<última iteração>.pt` e para o escritor (`on_policy_runner.py:132-134`).

## 11. O PPO do treino: redes e hiperparâmetros

### 11.1 As redes

O ator e o crítico são duas redes separadas, `MLPModel`, criadas por `PPO.construct_algorithm` (`ppo.py:432-436`). Os dois leem o grupo de observação com o próprio nome (`mjlab/rl/config.py:92-94`).

| Rede | Camadas ocultas | Ativação | Normaliza a observação | Saída | Referência |
|---|---|---|---|---|---|
| Ator | (512, 256, 128) | `elu` | Sim (`EmpiricalNormalization`) | 29 médias de ação, uma por junta | `rl_cfg.py:13-22`, `ppo.py:432`, `mjlab/rl/vecenv_wrapper.py:21` |
| Crítico | (512, 256, 128) | `elu` | Sim | 1 valor | `rl_cfg.py:23-27`, `ppo.py:436` |

O ator tem distribuição Gaussiana, com `init_std` = 1,0 e `std_type` = `"scalar"` (`rl_cfg.py:17-21`). O desvio é um vetor de 29 parâmetros aprendidos, `std_param`, independente da observação (`rsl_rl/modules/distribution.py:164-166`, `learn_std` True por padrão na linha 150). O desvio é cortado em [1e-6; 1e6] ao usar (`distribution.py:148,187`). A ação é amostrada de `Normal(média da rede, std)` (`distribution.py:191-195`). O crítico não tem distribuição (`rl_cfg.py:23-27`). Cada rede é uma sequência de camadas `nn.Linear` e ativação `elu`, com a última camada linear (`rsl_rl/modules/mlp.py:54-64`). Nenhum código lido troca a inicialização dos pesos: `GaussianDistribution` herda um `init_mlp_weights` que não faz nada (`distribution.py:120-129,132`, `mlp_model.py:79-80`), e nenhum código lido chama `MLP.init_weights` (`rsl_rl/modules/mlp.py:80`). Os pesos ficam com a inicialização padrão de `nn.Linear` do torch.

O otimizador é o Adam, sobre os parâmetros do ator e do crítico juntos, com taxa inicial `learning_rate` = 1e-3 (`config.py:76`, `ppo.py:92-94`, `rl_cfg.py:35`).

### 11.2 Os hiperparâmetros

| Parâmetro | Valor | Referência |
|---|---|---|
| `num_steps_per_env` | 24 | `knobs.py:402`, `tarefa.py:49` |
| `num_learning_epochs` | 5 | `rl_cfg.py:33` |
| `num_mini_batches` | 4 | `rl_cfg.py:34` |
| `learning_rate` inicial | 1e-3 | `rl_cfg.py:35` |
| `schedule` | `adaptive` | `rl_cfg.py:36` |
| `desired_kl` | 0,01 | `rl_cfg.py:39` |
| `gamma` | 0,99 | `rl_cfg.py:37` |
| `lam` (lambda do GAE) | 0,95 | `rl_cfg.py:38` |
| `entropy_coef` | 0,001 | `tarefa.py:48` |
| `clip_param` | 0,2 | `rl_cfg.py:31` |
| `value_loss_coef` | 1,0 | `rl_cfg.py:29` |
| `use_clipped_value_loss` | True | `rl_cfg.py:30` |
| `max_grad_norm` | 1,0 | `rl_cfg.py:40` |
| `normalize_advantage_per_mini_batch` | False | `config.py:71` |
| `optimizer` | `adam` | `config.py:76` |
| `save_interval` | 50 iterações | `rl_cfg.py:43` |
| `max_iterations` no cfg | 30 000 (o `entrada.sh` sobrescreve com `ITERS`) | `rl_cfg.py:45`, `entrada.sh:86` |
| `obs_normalization` (ator e crítico) | True | `rl_cfg.py:16,25` |
| `seed` | 42 | `mjlab/rl/config.py:86` |

A agenda `adaptive` age a cada minibatch, dentro do laço de `update`, 20 vezes por iteração (`ppo.py:234-259`). Ela calcula a divergência KL média entre a distribuição antiga e a nova no minibatch. Se `KL > desired_kl × 2` (0,02), a taxa vira `max(1e-5; taxa / 1,5)` (`ppo.py:246-247`). Se `KL < desired_kl / 2` (0,005) e `KL > 0`, a taxa vira `min(1e-2; taxa × 1,5)` (`ppo.py:248-249`). Depois, a taxa vale para todos os grupos de parâmetros do otimizador (`ppo.py:258-259`).

### 11.3 A normalização das observações

Cada rede tem um `EmpiricalNormalization` sobre o vetor concatenado do seu grupo (`rsl_rl/models/mlp_model.py:60-64,118`). Ele guarda `_mean`, `_var`, `_std` e `count` como buffers (`rsl_rl/modules/normalization.py:31-34`).

- A atualização roda a cada passo do env, em `process_env_step`, com as observações pós-passo de todos os envs (`ppo.py:133-134`, `mlp_model.py:171-178`). A conta é uma média móvel por lote: `taxa = n_lote / contagem_total`, com `until` = `None` (nunca para) (`normalization.py:51-66`). A atualização só roda com a rede em modo de treino (`normalization.py:53`).
- O `forward` calcula `(x − média) / (std + eps)`, com `eps` = 0,01 (`normalization.py:18,46-48`).
- Em cada passo do env, depois da atualização, o `fixa_normalizador` põe um piso de `PISO_DESVIO` = 0,05 em `_std` e fixa as fatias de `_FIXOS` em média 0 e no desvio do contrato (seções 12 e 13) (`algoritmo.py:66,69-83,133-134`).

## 12. O que `algoritmo.py` muda no PPO do `rsl_rl`

O módulo define `PPOPorTarefa(PPO)` e o caminho `"g1_v3.algoritmo:PPOPorTarefa"` (`algoritmo.py:27,86`). Ele muda quatro coisas.

| Mudança | Onde | O que faz |
|---|---|---|
| Vantagem normalizada por grupo | `compute_returns`, `normaliza_por_grupo` (`algoritmo.py:34-60,137-159`) | O PPO base normaliza a vantagem pela média e pelo desvio do lote todo (`ppo.py:185-188`). A v3 normaliza dentro de cada grupo. Um grupo com 2 ou mais amostras sai com `(a − média) / (desvio + 1e-8)`, com o desvio do torch (não viesado). Um grupo com menos de 2 amostras: o grupo 0 (ANDAR) fica com a vantagem crua. Os outros usam a média e o desvio de todas as amostras de grupo diferente de 0, se houver 2 ou mais dessas, senão ficam crus. |
| Definição do grupo | `_grupo` (`algoritmo.py:98-112`) | A tarefa é o `argmax` das 6 últimas colunas da observação do crítico (`algoritmo.py:106`). Sem `fatia_alvo`, o grupo é a tarefa. Com `fatia_alvo` (o runner a define), o grupo é 0 quando a tarefa é ANDAR (índice 0) e o slot da fatia `alvo_macro` é o último (`n − 1`, "nenhum"), e `1 + tarefa × n + slot` nos demais casos (`algoritmo.py:107-112`). |
| Penalidade de futuro negativo | `process_env_step` (`algoritmo.py:114-135`) | Seção 10.2. Soma `gamma × min(V̄_grupo, 0)` à recompensa nas terminações que não são `time_out` e no passo do pulso `v3_perdeu_agora`. `V̄_grupo` é `v_grupo`, calculado no `compute_returns` do rollout anterior (`algoritmo.py:160-169`). |
| Normalizador com piso e canais fixos | `fixa_normalizador`, `PISO_DESVIO` (`algoritmo.py:66-83`) | Põe piso de 0,05 em `_std` e fixa cada fatia de `fatias_fixas` em média 0 e no desvio dado. Sem normalizador (`Identity`), não faz nada. A fixação roda a cada passo do env, depois da atualização do PPO base. |

O restante do PPO (`act`, `update`, perda, agenda de taxa, salvamento) vem do `rsl_rl` sem mudança.

## 13. O que `runner.py` muda e o que o checkpoint guarda

A classe `RunnerV3(MjlabOnPolicyRunner)` acrescenta o construtor e quatro métodos (`runner.py:37-112`): `_fixa_obs`, `_embrulha_log`, `save` e `load`.

**Construtor.** Depois do construtor do mjlab, chama `_embrulha_log()` e `_fixa_obs()` (`runner.py:41-44`).

**`_fixa_obs` (`runner.py:46-66`).** Para cada grupo `actor` e `critic`, percorre os termos na ordem de `active_terms` e acumula a largura de cada um (`prod(dim)`). Se o nome do termo está em `_FIXOS`, guarda `(slice, desvio)` (`runner.py:55-60`). No crítico, guarda também a fatia do termo `alvo_macro` em `alg.fatia_alvo` (`runner.py:61-62`). Confere que a soma das larguras é igual à entrada do normalizador (`runner.py:64-65`). Entrega o dicionário `{"actor": [...], "critic": [...]}` em `alg.fatias_fixas` (`runner.py:66`).

O dicionário `_FIXOS` (`runner.py:32-34`):

| Termo | Desvio fixo | Termo | Desvio fixo |
|---|---|---|---|
| `tarefa` | 1,0 | `caixa` | 0,1 |
| `mapa` | 1,0 | `caixa_verdade` | 0,1 |
| `alvo_macro` | 1,0 | `alvo_verdade` | 0,1 |
| `niveis` | 1,0 | `referencia` | 0,1 |
| `degrau` | 1,0 | `forca_caixa` | 0,3 |
| `massa` | 1,0 | | |
| `tarefa_critico` | 1,0 | | |
| `privilegiados` | 1,0 | | |

Os outros termos (por exemplo `giro_base`, `gravidade`, `comando`, `fase`, `juntas`, `vel_juntas`, `acao`, `vel_lin_base`) passam pelo normalizador empírico. Como o `forward` soma `eps` = 0,01 ao desvio, o divisor efetivo de uma fatia fixa é o desvio mais 0,01 (`normalization.py:46-48`).

**`_embrulha_log` (`runner.py:68-86`).** Troca `logger.log` por uma função que chama o original e, se o escritor existe, grava `Policy/std/<junta>` para cada uma das 29 juntas, com o desvio médio da ação sobre a dimensão 0 (`runner.py:76-84`). Os nomes vêm de `target_names` do termo `joint_pos`, na ordem do vetor de ação. O `Policy/mean_std` do rsl_rl continua sendo gravado (`rsl_rl/utils/logger.py:211`).

**`save` (`runner.py:88-90`).** Junta `"v3_curriculo": env.v3_cur.estado()` ao dicionário `infos` e chama o `save` do mjlab. O `save` do mjlab acrescenta `infos["env_state"] = {"common_step_counter": ...}` e grava `actor_state_dict`, `critic_state_dict` e `optimizer_state_dict` (de `alg.save()`), `iter` e `infos` (`mjlab/rl/runner.py:67-81`, `ppo.py:360-370`). Os `state_dict` do ator e do crítico incluem os buffers do normalizador (`_mean`, `_var`, `_std`, `count`).

O `estado()` do currículo devolve `it`, as células (`aberta`, `s`, `n`), os acumuladores `acum`, por tarefa (`nivel`, `nivel_ter`, `seguidas`, `degrau`, `cont`, `sub`), `dr`, `it_abertura`, `rampa` e a `zona` (`z`, `visitas`, `fechos`) e a chave `terreno` (`linha`, `visitas`, `travessias`: as linhas por coluna do ANDAR e as contagens da iteração) (`balanceador.py:393-405`). O `nivel_ter` do ANDAR guardado é o piso da média das linhas. O checkpoint não guarda o estado de episódio (cadeia, alvos, buffers de comando).

**`load` (`runner.py:92-112`).** Roda assim:

1. Lê o checkpoint em CPU e pega o `critic_state_dict` (`runner.py:96`).
2. Se `load_cfg` é `None` e alguma chave do crítico do checkpoint não existe no crítico atual ou tem forma diferente, define `load_cfg = {"actor": True, "critic": False, "optimizer": False, "iteration": True}` e imprime um aviso (`runner.py:97-101`). Nesse caso, o crítico e o otimizador nascem novos. A conferência olha só as chaves do checkpoint. Uma chave que existe só no crítico atual não dispara a troca.
3. Chama o `load` do mjlab (`runner.py:102`). Ele migra formatos antigos de checkpoint, chama `alg.load`, troca `current_learning_iteration` por `iter` se `load_cfg["iteration"]` é verdadeiro e restaura `common_step_counter` de `infos["env_state"]` (`mjlab/rl/runner.py:97-141`, `ppo.py:372-394`).
4. Se `infos` não tem `v3_curriculo` ou o valor é vazio, imprime um aviso e devolve. O currículo começa do zero (`runner.py:103-109`).
5. Senão, chama `v3_cur.carrega(estado)` e imprime `currículo restaurado: it=<it>` (`runner.py:110-111`). O `carrega` restaura células, acumuladores, níveis, degraus, travas de DR, `it_abertura`, `rampa`, zonas e, na chave `terreno`, as linhas por coluna, as visitas e as travessias. Num checkpoint sem a chave, todas as colunas voltam à linha 0 sem visitas. O `nivel_ter` do ANDAR no checkpoint não volta: o nível de terreno dele é a linha por coluna. Limita o nível de cada uma das quatro tarefas ao máximo atual dela (`balanceador.py:407-443`). Não restaura `taxa`: o valor volta a −1 só quando o estado da zona tem tamanho diferente do atual (`balanceador.py:436-443`).

## 14. A pasta de log e os checkpoints

| Item | Valor | Referência |
|---|---|---|
| Pasta da corrida | `<log-root>/g1_v3/<AAAA-MM-DD_HH-MM-SS>_<RUN>` | `mjlab/scripts/train.py:186-190`, `tarefa.py:27,46` |
| Arquivos de checkpoint | `model_<it>.pt`, a cada `save_interval` = 50 iterações com `it % 50 == 0`, e `model_<última iteração>.pt` no fim | `rl_cfg.py:43`, `on_policy_runner.py:128-133` |
| Configuração gravada | `params/env.yaml` e `params/agent.yaml` | `mjlab/scripts/train.py:163-165` |
| Eventos do TensorBoard | na própria pasta da corrida; `SummaryWriter` com `flush_secs=10` | `rsl_rl/utils/logger.py:103-105`, `entrada.sh:86` |
| Estado do git | pasta `git/` com um arquivo `<repositório>.diff` por repositório git que o logger acha a partir de `rsl_rl.__file__` e de `mjlab/scripts/train.py`. Sem repositório, imprime um aviso e segue. | `rsl_rl/utils/logger.py:44,300-317`, `mjlab/scripts/train.py:170` |
| Vídeos | `<run>/videos/train/`, só com `--video True` e no `rank` 0. O `VideoRecorder` conta as chamadas de `step` em `step_count`. Quando `step_count % video_interval == 0`, ele começa a gravar o env 0 (`step_trigger`) e grava um quadro por passo até juntar `video_length` quadros. Com `entrada.sh` e `VIDEO=1`: `video_interval` = 12000 e `video_length` = 1000, o que dá a primeira gravação no passo 0 e as seguintes a cada 12000 passos. | `mjlab/scripts/train.py:138-146`, `mjlab/utils/wrappers/video_recorder.py:85-135`, `entrada.sh:93-97` |

O `save` roda só quando o escritor existe (`on_policy_runner.py:128`). O envio do modelo a um serviço externo (`upload_model` True) não faz nada com o escritor do TensorBoard, porque `Logger.save_model` só age se o escritor é um `LogWriter` (`mjlab/rl/runner.py:80-81`, `rsl_rl/utils/logger.py:290-293`). O `SummaryWriter` do TensorBoard não é um `LogWriter` (`rsl_rl/utils/logger.py:103-105`).

## 15. O que muda entre treino e play

O `make_env_cfg(play=True)` gera o `play_env_cfg`. O `mjlab.scripts.play` o carrega com `load_env_cfg(task, play=True)` (`mjlab/scripts/play.py:64`, `mjlab/tasks/registry.py:48-55`).

| Aspecto | Treino | Play | Referência |
|---|---|---|---|
| Duração do episódio | `Cadeia.episodio_s` = 20,0 s | `_EPISODIO_PLAY_S` = 10⁹ s. Quem termina é `v3_fim`. | `env_cfg.py:63,105` |
| Corrupção da observação do ator | ruído ligado (`enable_corruption=True`) | desligada (`enable_corruption=False`), então o ruído de cada termo vira `None` | `env_cfg.py:113-114`, `observation_manager.py:436-437` |
| Eventos de reset `pd`, `forca_motor`, `solref` | presentes | ausentes | `env_cfg.py:169-181` |
| Evento de intervalo `empurrao` | presente | ausente | `env_cfg.py:200-205` |
| Eventos de passo `forca_caixa` e `rfi` | presentes | ausentes | `env_cfg.py:200,207-210` |
| Eventos mantidos | tudo o mais | `fatia`, `cena`, `base`, `juntas`, `carga`, `atrito_pad`, `atrito_laje`, `vies_imu`, `filtros` (reset) e `atrito_pe`, `massa_tronco`, `vies_encoder`, `tamanho_caixa`, `atraso` (startup) | `env_cfg.py:153-199` |
| Currículo | termo `v3` presente | `cfg.curriculum = {}`: sem termo `v3` | `env_cfg.py:216` |
| Terminações, recompensas e métricas | as mesmas | as mesmas | `env_cfg.py:131-147,217` |

Consequências do `cfg.curriculum` vazio:

- O fecho da iteração nunca roda (seção 9.1).
- `v3_iter` fica em −1 (`curriculo.py:58-59`).
- `eventos.atraso` só roda no startup, com `DR.atraso_inicio[1]` = 0. O teto do atraso nunca sobe, então o sorteio do atraso dá 0 (seção 5.6).
- O objeto `v3_cur` existe, porque o construtor do `CadeiaCmd` o cria (`comando.py:293`). O estado dele vem do checkpoint, carregado pelo `RunnerV3.load` (`runner.py:103-111`).
- Os sorteios que leem `cur.dr()` e `cur.nivel(...)` (atritos, `vies_imu`, `cena`, `fatia`) usam o estado restaurado.

O `g1_v3/play.py` importa `g1_v3.tarefa` e chama `mjlab.scripts.play.main` (`play.py:7-11`). O restante do play vem do script do mjlab (`mjlab/scripts/play.py:59-209`):

- O dispositivo é `cuda:0` se o torch vê CUDA, senão `cpu` (`mjlab/scripts/play.py:62`).
- `num_envs` vem de `--num-envs`. Sem ele, vale o padrão do molde do fabricante, 1 (`mjlab/scripts/play.py:158-159`, `mjlab/tasks/velocity/velocity_env_cfg.py:426`).
- O runner é `RunnerV3(env, asdict(agent_cfg), device=device)`, sem pasta de log (`mjlab/scripts/play.py:203-204`).
- O `load` usa `load_cfg={"actor": True}` e `strict=True` (`mjlab/scripts/play.py:205-207`). Só o ator é carregado, a iteração não é trocada, e o `RunnerV3.load` restaura o currículo e o `common_step_counter`.
- A política vem de `get_inference_policy` (`mjlab/scripts/play.py:208`), que põe a rede em modo de avaliação e devolve o ator (`rsl_rl/runners/on_policy_runner.py:163-166`). O `forward` do ator, sem `stochastic_output`, devolve a média da distribuição (`rsl_rl/models/mlp_model.py:82-107`).
- O visualizador é `NativeMujocoViewer` (com tela) ou `ViserPlayViewer` (sem tela), escolhido por `--viewer` (`mjlab/scripts/play.py:283-295`). A cada passo, sob `torch.no_grad()`, o visualizador chama `env.get_observations()` (que devolve o cache do `ObservationManager`), `policy(obs)` e `env.step(actions)` (`mjlab/viewer/base.py:282-294`). No play, o `step` roda a mesma ordem da seção 5, com os eventos da coluna Play da tabela acima.

O bloco `if play:` do molde `unitree_g1_rough_env_cfg` do fabricante mexe só em episódio, observações, eventos, terminações, currículo e terreno (`mjlab/tasks/velocity/config/g1/env_cfgs.py:164-182`). O `make_env_cfg` substitui todos esses campos depois (`env_cfg.py:94-96,105,112,137,153-216`). O bloco não deixa efeito.

## 16. Logs e métricas

### 16.1 Como o logger agrega

O `Logger` do rsl_rl acumula e agrega em quatro níveis.

1. A cada passo de coleta, se o escritor existe, o logger guarda o dicionário `extras["log"]` do passo em uma lista (`rsl_rl/utils/logger.py:129-133`). O dicionário só tem conteúdo nos passos em que algum env resetou (seção 6, parágrafo depois da tabela).
2. Em `log(...)`, para cada chave presente, o logger concatena os valores de todos os passos da iteração e grava a média (`rsl_rl/utils/logger.py:183-205`). Uma chave com `/` é gravada como está. Uma chave sem `/` ganha o prefixo `Episode/` (`rsl_rl/utils/logger.py:198-203`).
3. Cada manager entrega, por reset, um escalar que já é a média sobre os envs resetados naquele passo (por exemplo `torch.mean(...)` em `reward_manager.py:107`). A média final da iteração é, então, uma média sem peso dos passos com reset, e não uma média por episódio. As chaves `Episodio/...` de `metricas.py` entregam um vetor com um elemento por episódio, e a média delas é por episódio (`metricas.py:81-90`).
4. O logger mantém duas filas de 100 elementos (`deque(maxlen=100)`) com a soma de recompensa e o comprimento de cada episódio que acabou (`dones > 0`, inclusive `time_out`) (`rsl_rl/utils/logger.py:53-54,141-149`). A soma de recompensa vem do `rewards` já multiplicado por `dt` (seção 7). O logger recebe o `rewards` que o env devolveu: a soma não inclui o termo `γ × min(V̄_grupo, 0)` da seção 10.2 nem o bootstrap de `time_out` (`on_policy_runner.py:87,94,98`).

### 16.2 As famílias de chaves

| Chave | O que é | Referência |
|---|---|---|
| `Loss/value`, `Loss/surrogate`, `Loss/entropy` | médias das perdas sobre os 20 passos do `update` | `ppo.py:310-335`, `logger.py:206-207` |
| `Loss/learning_rate` | taxa de aprendizado ao fim do `update` | `logger.py:208` |
| `Policy/mean_std` | média do desvio da ação sobre as 29 juntas | `logger.py:211` |
| `Policy/std/<junta>` | desvio da ação de cada junta | `runner.py:76-84` |
| `Perf/total_fps`, `Perf/collection_time`, `Perf/learning_time` | velocidade e tempos de relógio | `logger.py:214-217` |
| `Train/mean_reward`, `Train/mean_episode_length` | média dos últimos 100 episódios terminados | `logger.py:225-226` |
| `Train/mean_reward/time`, `Train/mean_episode_length/time` | os mesmos valores, com o tempo acumulado em segundos como eixo | `logger.py:227-233` |
| `Episode_Reward/<termo>` | média, sobre os envs resetados, da soma do episódio do termo (já com `dt`) dividida por `max_episode_length_s` = 20,0 s | `reward_manager.py:100-111` |
| `Episode_Termination/<termo>` | número de envs resetados em que o termo disparou | `termination_manager.py:88-97` |
| `Episode_Metrics/<termo>` | métrica de `metricas.py` conforme o `reduce` (seção 16.3) | `metrics_manager.py:107-133` |
| `Metrics/<comando>/<métrica>` | métricas do termo de comando. O `cadeia` não tem métrica. O `twist` tem `error_vel_xy`, `error_vel_yaw` (fabricante) e `razao_marcha`, `acerto_relogio`, `eficiencia_min`, `eficiencia_media`, `segmentos`. | `command_manager.py:99-107,241-247`, `velocity_command.py:50-51`, `comando.py:686,703-704` |
| `Curriculum/v3/<chave>` | o dicionário que o termo `v3` devolve (seção 6.2) | `curriculum_manager.py:95-110`, `curriculo.py:127-143` |
| `Episodio/<nome>` | valor por episódio dos termos de métrica que são classe | `metricas.py:81-90` |

### 16.3 Os termos de `metricas.py`

O `metricas.termos(k)` monta 32 termos de métrica (`metricas.py:522-571`). Uma métrica não tem peso e não é multiplicada por `dt`. O `MetricsManager` não escreve no buffer de recompensa (`mjlab/managers/metrics_manager.py:147-173`). Ela é calculada uma vez por passo de controle, depois das recompensas, e o reset vem em seguida (seção 5, itens 7 e 8). Nenhum termo usa `per_substep`.

O `reduce` define o que vira `Episode_Metrics/<termo>` no reset do env (`metrics_manager.py:107-133`):

| `reduce` | Valor registrado |
|---|---|
| `mean` (padrão) | soma dos valores do episódio dividida pelo número de passos do episódio (`max(passos, 1)`) |
| `max` | maior valor do episódio |
| `last` | valor do último passo do episódio |

Os termos de classe têm `reset` próprio. Esse `reset` zera o estado do termo, e alguns também registram `Episodio/<nome>`.

Nos termos com `reduce="last"` e a marca "flag", o `__call__` devolve 1 se o episódio teve o evento e 0 se não teve. Então `Episode_Metrics/<termo>` é a fração dos envs resetados que tiveram o evento. O valor por episódio sai em `Episodio/<nome>`, só para os envs com o evento.

| Termo | `reduce` | O que mede, por passo | Referência |
|---|---|---|---|
| `tempo_de_voo` | `mean` | Média, sobre os pés com `current_air_time > 0`, do tempo no ar, em s. 0 sem pé no ar. | `metricas.py:93-101` |
| `pico_de_altura` | `mean` | Classe. Guarda o maior valor de `altura_pes` de cada pé enquanto o pé está sem contato. No passo do primeiro contato, devolve a média dos picos dos pés que pousaram e zera o pico. Nos outros passos devolve 0. | `metricas.py:104-133` |
| `velocidade_de_escorrego` | `mean` | Média, sobre os pés com contato, da norma xy da velocidade do sítio do pé (`left_foot`, `right_foot`), em m/s. | `metricas.py:136-142` |
| `velocidade_de_junta` | `mean` | Raiz do valor quadrático médio de `joint_vel` das 29 juntas. | `metricas.py:145-149` |
| `fracao_do_curso` | `max` | Maior `\|q − centro\| / meio` sobre as 29 juntas, com os limites duros do modelo (1,0 é o batente). | `metricas.py:152-165`, `potencial.py:70-75` |
| `forca_de_pouso` | `mean` | Média, sobre os pés no passo do primeiro contato, da norma da força do contato, em N. 0 nos outros passos. | `metricas.py:168-173` |
| `passo_medio` | `last` | Classe. Média corrente, desde o reset, do comprimento do passo no toque, em m. Só conta toque com a norma xy do comando maior ou igual a `Comando.limiar_canal` = 0,02 (`knobs.py:138`). O comprimento vem de `recompensas.comprimento_do_passo`. | `metricas.py:176-219`, `metricas.py:549-553` |
| `fracao_mapa_visto` | `mean` | Fração das confianças do mapa (a segunda metade do vetor `v3_mapa_obs`, de 882 canais, `knobs.py:62`; são 441) acima de `Percepcao.limiar_visto` = 0,5 (`knobs.py:503`). 0 sem mapa. | `metricas.py:222-232`, `metricas.py:554-555` |
| `phi_P0`, `phi_P1`, `phi_P2`, `phi_P3`, `phi_B1`, `phi_B2`, `phi_B3` | `last` | Classe, flag "teve o alvo". Soma o potencial `v3_phi` nos passos em que `v3_alvo_idx` é o alvo e a tarefa tem caixa (PEGAR, CARREGAR ou BOTAR). `Episodio/phi_<alvo>` = média desse potencial no episódio. | `metricas.py:245-270,558-559` |
| `tempo_P0` ... `tempo_B3` (7 termos) | `last` | Classe, flag "teve o alvo". Conta os passos em que `v3_alvo_idx` é o alvo e a tarefa tem caixa. `Episodio/tempo_<alvo>` = passos × `step_dt`, em s. | `metricas.py:273-294,558-560` |
| `inclinacao_caixa_p3` | `max` | Inclinação da caixa, em rad, no passo em que o alvo é P3. 0 nos outros passos. | `metricas.py:329-335,561` |
| `forca_robo_caixa` | `mean` | Soma dos módulos das forças do sensor `caixa_robo`, em N. | `metricas.py:338-345,562` |
| `deslocamento_caixa_laje` | `max` | Classe. Deslocamento xy da caixa desde a posição do primeiro passo com alvo P0, nos passos com alvo P0, P1 ou P2, em m. 0 fora deles. | `metricas.py:348-373,563` |
| `velocidade_de_chegada` | `last` | Classe, flag "teve toque" (`v3_v_chegada >= 0`). `Episodio/velocidade_de_chegada` = `v3_v_chegada`, a velocidade de descida da caixa no primeiro contato com a laje no BOTAR, em m/s. | `metricas.py:376-395,564` |
| `deslocamento_depois_do_apoio` | `last` | Classe, flag "teve apoio". Maior deslocamento xy da caixa em relação a `v3_ponto_apoio` nos passos com alvo B2 ou B3 e sem `v3_perdeu`. `Episodio/deslocamento_depois_do_apoio` = esse maior valor, em m. | `metricas.py:398-424,565` |
| `idade_leituras` | `mean` | Idade, em s, da leitura do estimador da caixa (coluna 7 de `saida_b`). 0 onde a tarefa não tem caixa. 0 se o estimador ou a odometria não existem. | `metricas.py:494-507,566` |
| `fracao_avisos` | `mean` | O latch `v3_aviso` como 0 ou 1. A média do episódio é a fração dos passos com o aviso ligado. | `metricas.py:510-519,567` |
| `fracao_fora_da_plataforma` | `last` | Classe, flag "teve passo de evento". Evento: tarefa CARREGAR, fatia B e episódio de terreno (`terrain_types > 0`). Conta os passos em que o chão sob a pelve difere do z da origem em mais de 1 cm. `Episodio/fracao_fora_da_plataforma` = passos fora ÷ passos do evento. | `metricas.py:427-456,568` |
| `aceleracao_da_caixa` | `last` | Classe, flag "teve passo preso". Média do custo de sacudida nos passos em que o alvo está no grupo de alvos com a caixa presa (`recompensas._CAIXA_ALVO`). `Episodio/aceleracao_da_caixa` = essa média. | `metricas.py:459-491,569` |
| `erro_p3` | `last` | Classe, flag "teve o P3". Média, nos passos com alvo P3 e tarefa com caixa, de três erros em m: vertical (−z do erro da caixa ao alvo), lateral (componente lateral absoluta, no frame do rumo) e da pelve (diferença absoluta entre a altura da pelve e `pelve_com_caixa`). Sai em `Episodio/erro_p3_vertical`, `Episodio/erro_p3_lateral` e `Episodio/erro_p3_pelve`. | `metricas.py:297-326,570` |

Dois comentários sobre a leitura dos termos de classe com flag. O termo roda no passo da terminação, e o reset vem em seguida. Então o último passo do episódio entra na conta (seção 5, itens 7 e 8). O `reset` do termo loga o valor do episódio e só depois zera o estado (`metricas.py:81-90`), e a métrica de classe roda dentro do `metrics_manager.reset`, que precede o reset dos comandos (`manager_based_rl_env.py:575,581`). Por isso `velocidade_de_chegada` lê `v3_v_chegada` do episódio que fecha, e não o valor reiniciado em −1 pelo reset do comando.

## 17. Não determinado pelo código lido

- Não determinado pelo código lido: o número de canais de `robot/imu_lin_vel`, que entra no termo `vel_lin_base` do crítico (`observacoes.py:406-407`). O `g1.xml` declara o sensor como `velocimeter`, mas o código lido não fixa o número de canais. Com 3 canais, a entrada do crítico seria 1 518. A dimensão de `caixa_verdade` (7), `alvo_verdade` (3), `referencia` (8), `niveis` (4) e `privilegiados` (4) vem das docstrings das funções, e não de uma instrução que o código lido execute como contagem.
- Não determinado pelo código lido: o que `Sensor.reset` faz em cada subclasse de sensor além de invalidar o cache (`mjlab/sensor/sensor.py:129-139`). As subclasses de sensor não foram lidas.
- Não determinado pelo código lido: se os 9 grupos de atuadores caem num só buffer de atraso fundido. A regra de fusão está em `mjlab/actuator/builtin_group.py:111-121` (mesma classe, mesmo tipo de transmissão e mesma configuração de atraso), mas o valor da propriedade `has_delay` não foi lido.
- Não determinado pelo código lido: o que `NativeMujocoViewer` e `ViserPlayViewer` fazem fora de `_execute_step` (`mjlab/viewer/base.py:282-294`), por exemplo como o viewer reage a um reset.
- Não determinado pelo código lido: o tempo de relógio por iteração e os valores de `Perf/*`. Dependem da máquina.
- Não determinado pelo código lido: a fórmula interna de cada termo de recompensa e de cada recompensa de progresso. Elas ficam fora do assunto deste documento, que cobre só como o gerenciador as soma.
