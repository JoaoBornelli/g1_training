---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
resumo: "mjlab: o que pode mudar em tempo de treino/episódio — deepcopy assimétrico, currículo só no reset, sem duração por env, sem bootstrap em falha"
cluster: transversais
---
# g1-mjlab-mutacao-em-treino (original)

Corpo original da memória `g1-mjlab-mutacao-em-treino`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Auditoria de código do mjlab 1.5.1 (subagente, 2026-08-20), com file:line conferidos. Base para a máquina de elos do `g1_poc` (§7).

## Deepcopy é ASSIMÉTRICO entre managers

- `CommandManager` e `ActionManager` **não** deepcopiam o cfg → `command_term.cfg` é o MESMO objeto de `env.cfg.commands[nome]`. Mutar `cfg.ranges.lin_vel_x` em runtime funciona — é o mecanismo do próprio `commands_vel` (`tasks/velocity/mdp/curriculums.py:90-96`), e `_resample_command` relê `self.cfg.ranges` a cada resample.
- Os outros cinco (reward, termination, event, observation, curriculum) **deepcopiam** no `__init__`. Pós-construção, o único caminho é `<manager>.get_term_cfg(nome)` — mutar `cfg.rewards[...]` etc. depois de o env existir é NO-OP silencioso.
- `mdp.reward_curriculum` muda `weight` por `setattr` e `params` por `.update()` no cfg vivo do RewardManager (`envs/mdp/curriculums.py:69-104`).

## Currículo e reset

- `curriculum_manager.compute(env_ids)` roda **só** no `_reset_idx` (linha 554), só para os envs que resetam. Não há gancho por passo para envs vivos — lógica por passo tem de viver em comando/recompensa/evento `mode="step"`.
- `episode_length_buf[env_ids] = 0` é a ÚLTIMA linha do `_reset_idx` (590) → dentro de um termo de currículo o buffer ainda vale a duração final do episódio que acabou.
- Ordem no reset: currículo.compute → sim/scene reset → eventos reset → … → comando.reset (resample) → … O currículo muda `cfg.ranges` ANTES do resample do mesmo reset.

## Mudar a cena no meio do episódio

- Precedente da casa: `tracking/mdp/commands.py:407-417` escreve estado do robô em `_update_command` e chama `sim.forward()` manual quando precisa ler depois de escrever.
- `command_manager.compute()` roda DEPOIS do único `sim.forward()` do step (`manager_based_rl_env.py:454,456`) → escrita ali só vale nas obs do passo seguinte. Para teleportar mobília isso basta; **não** chamar `forward()` extra (custo global por passo).
- `write_mocap_pose_to_sim` só em entidade fixed-base (o mjlab auto-envolve em mocap, `utils/spec.py:151-165`); robô/free-joint → `ValueError`. Todas as `write_*` aceitam `env_ids`.
- `write_external_wrench_to_sim` PERSISTE até sobrescrever ou resetar (`xfrc_applied`; `envs/mdp/events.py:340-370`) — a carga da caixa aplicada uma vez no reset está correta. `sim.reset(env_ids)` zera o wrench ANTES dos eventos de reset, então reaplicar em `mode="reset"` funciona.

## Limites que mudam desenho

- **Não existe duração de episódio por env**: `episode_length_s` é escalar global. Terminação custom por env é possível, mas `Episode_Reward/*` continua dividindo pelo `max_episode_length_s` GLOBAL (`reward_manager.py:109`) → log distorcido. Elimina "encurtar episódio de manipulação" como forma de equilibrar a fatia de dados.
- **rsl_rl só faz bootstrap em `time_outs`** (`ppo.py:150-155`: `rewards += γ·V·time_outs`). Terminação de falha = V 0 duro. Uma terminação que passa a disparar (ex.: `caixa_largada` consertada) vira punição forte de verdade.
- **Não há idioma de sub-metas sequenciais** em `tasks/`. O padrão da casa é o latch `episode_success = maximum(...)` do `LiftingCommand` (`manipulation/mdp/commands.py:47`), zerado em `_resample_command`. Índice de estágio por env estende esse padrão; não há utilitário pronto.
- Sensores `reduce="netforce"`: força SEMPRE em frame global, `num_slots` ignorado (1 slot por primary), `found > 0` = contato. `data.found` é `[B, P]`, `data.force` é `[B, P, 3]`.
- Recompensa zero para subconjunto de envs é seguro — nenhuma normalização assume não-zero; `nan_to_num` no compute.

Relacionado: g1-mjlab-velocity-anatomia, g1-poc-bloco1-treino, g1-postura-penhasco-std-standing.
