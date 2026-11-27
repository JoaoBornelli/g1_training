# Recompensas e penalidades de locomoção e de estilo

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:
- `recompensas.py` (inteiro)
- `potencial.py` (inteiro)
- `env_cfg.py` (inteiro)
- `knobs.py` (classes `Contrato`, `Comando`, `Cadeia`, `Alvos`, `Recompensa`, `Balanceador`, `Niveis`)
- `comando.py` (sorteio de velocidade, `TwistV3`, `_publica`, reset da cadeia)
- `balanceador.py` (`no_degrau`, `restricao`, `_sobe`, `max_nivel`)
- `curriculo.py` (`mascara`, `v3`, buffers `v3_*`)
- `observacoes.py` (linhas 301-318, relógio de marcha)
- `alvos.py` (`Referencias`)
- `cena.py` (sensores `pes_chao`, `altura_pes`, `altura_pelve`)
- `mjlab/managers/reward_manager.py`
- `mjlab/envs/manager_based_rl_env.py`
- `mjlab/managers/command_manager.py`
- `mjlab/managers/action_manager.py`
- `mjlab/entity/data.py`
- `mjlab/sensor/terrain_height_sensor.py`
- `mjlab/sensor/contact_sensor.py` (trecho de `found`)
- `mjlab/tasks/velocity/mdp/velocity_command.py`

Convenções deste documento:
- "Termo" é uma função de recompensa registrada no gerente de recompensa. Ela devolve um tensor `(E,)`, um valor por env.
- "Núcleo" é a função pura `n_*` que faz a conta sobre tensores. "Invólucro" é a função sem prefixo que lê o env, chama o núcleo e aplica a máscara.
- "Tarefa" é a etapa da cadeia em que o env está (`ANDAR`, `PEGAR`, `CARREGAR`, `BOTAR`). O buffer `env.v3_tarefa` guarda a tarefa de cada env (`curriculo.py:93-100`).
- "Fatia" é o tipo de episódio que um env recebe no reset, um de `ANDAR`, `B`, `C` (`knobs.py:43`; o `sorteia_fatia` a escreve antes do reset do comando, `comando.py:910`). Ela decide em quais tarefas o comando de velocidade é diferente de zero (`comando.py:905`). Quais tarefas cada fatia contém: Não determinado pelo código lido neste assunto.
- "Por segundo" é o valor do termo vezes o peso. O gerente já divide o dt de volta para esse número (`mjlab/managers/reward_manager.py:132`).
- Número sem knob é constante escrita no código. O texto diz isso e dá a linha.

## 1. Como o gerente de recompensa aplica peso e dt

### 1.1 Registro dos termos

| Fato | Referência |
|---|---|
| O `make_env_cfg` cria um `RewardTermCfg` por linha de `_RECOMPENSAS`. A função é `getattr(recompensas, nome)`. O peso é `getattr(k.recompensa, nome)`, o campo de `Recompensa` com o mesmo nome do termo. O parâmetro `tarefas` é a tupla da linha. | `env_cfg.py:131-134` |
| Só `progresso` (1,0) e `perda_caixa` (−1,0) têm peso fixo no cfg. Os 18 termos deste documento leem o peso do knob. | `env_cfg.py:57` |
| O cfg do fabricante é substituído por inteiro: nenhum termo de recompensa do fabricante sobrevive. | `env_cfg.py:6-7`, `env_cfg.py:91`, `env_cfg.py:131` |
| `TODAS` = (`ANDAR`, `PEGAR`, `CARREGAR`, `BOTAR`). `ANDAR_CARREGAR` = (`ANDAR`, `CARREGAR`). | `env_cfg.py:33-34` |

### 1.2 Conta do gerente a cada passo

`RewardManager.compute(dt)` roda uma vez por passo do env. Para cada termo, na ordem do dict:

1. `scale = dt`, porque `scale_rewards_by_dt` vale `True` (`mjlab/managers/reward_manager.py:118`, `mjlab/envs/manager_based_rl_env.py:157`). O `g1_v3` não muda esse campo.
2. Se `weight == 0.0`, o gerente pula a função e grava 0 na coluna do termo (`mjlab/managers/reward_manager.py:122-124`). Nenhum dos 18 termos tem peso zero em `knobs.py:236-268`.
3. `value = func(env, **params)`; o gerente confere a forma `(E,)` (`mjlab/managers/reward_manager.py:125-126`).
4. `value = value · weight · scale` (`mjlab/managers/reward_manager.py:127`).
5. `nan_to_num` troca NaN e infinito por 0 (`mjlab/managers/reward_manager.py:129`).
6. O gerente soma o valor na recompensa do passo e no acumulado do episódio do termo (`mjlab/managers/reward_manager.py:130-131`).

O dt do env é `timestep × decimation` (`mjlab/envs/manager_based_rl_env.py:266-268`). Os valores:
- `Contrato.dt_fisica` = 0,005 s (`knobs.py:54`).
- `Contrato.decimacao` = 4 (`knobs.py:55`).
- Logo `step_dt` = 0,02 s. O cfg copia os dois (`env_cfg.py:103-104`).

Consequência: a recompensa de um passo de um termo é `valor × peso × 0,02`. O valor "por segundo" das tabelas é `valor × peso`.

### 1.3 Ordem dentro de um passo do env

| Ordem | Ação | Referência |
|---|---|---|
| 1 | Processa a ação. O gerente de ação guarda `prev_action ← action` e `action ← nova ação`. | `mjlab/envs/manager_based_rl_env.py:419`, `mjlab/managers/action_manager.py:167-169` |
| 2 | 4 subpassos de física (`Contrato.decimacao`). | `mjlab/envs/manager_based_rl_env.py:421-427` |
| 3 | Terminações. | `mjlab/envs/manager_based_rl_env.py:436` |
| 4 | Recompensas. | `mjlab/envs/manager_based_rl_env.py:440` |
| 5 | Reset dos envs terminados (`_reset_idx`). Dentro dele: o currículo roda primeiro, depois os eventos, depois `reward_manager.reset`, depois `command_manager.reset`. | `mjlab/envs/manager_based_rl_env.py:447`, `mjlab/envs/manager_based_rl_env.py:554`, `mjlab/envs/manager_based_rl_env.py:572`, `mjlab/envs/manager_based_rl_env.py:581` |
| 6 | `sim.forward()`. | `mjlab/envs/manager_based_rl_env.py:454` |
| 7 | `command_manager.compute(dt)`: primeiro `_update_metrics` de cada comando, depois o contador `time_left` cai `dt`, depois a reamostra dos que chegaram a 0. | `mjlab/envs/manager_based_rl_env.py:456`, `mjlab/managers/command_manager.py:109-115` |

Efeito para os termos deste documento:
- O termo lê o comando `twist` do segmento corrente. A reamostra só acontece no passo 7, depois da recompensa.
- O termo lê `v_filt`, `g_marcha`, `v3_fase_marcha` e `v3_marcha_on`. O `TwistV3._update_metrics` escreve esses buffers no passo 7. Logo a recompensa do passo `t` lê o valor escrito no passo `t−1` (`comando.py:858`, `comando.py:871`, `comando.py:873-874`).
- O termo `taxa_acao` lê `action` e `prev_action` já deslocados no passo 1.
- Os sensores e as velocidades lidos no passo 4 vêm do estado depois dos 4 subpassos. O mjlab documenta que as grandezas derivadas ficam atrasadas um subpasso de física (`mjlab/envs/manager_based_rl_env.py:430-434`).

### 1.4 Log por episódio

No reset de um env, o gerente grava `Episode_Reward/<termo>` = média do acumulado do episódio dividida por `max_episode_length_s`, e zera o acumulado (`mjlab/managers/reward_manager.py:100-114`). O cfg de treino usa `Cadeia.episodio_s` = 20,0 s (`knobs.py:160`, `env_cfg.py:105`). No modo `play`, o episódio é `_EPISODIO_PLAY_S` = 10⁹ s (`env_cfg.py:63`, `env_cfg.py:105`).

## 2. Máscara por tarefa, zero depois da caixa perdida e fator de restrição

### 2.1 `_fim` e `mascara`

- `mascara(env, tarefas)` devolve `(E,)` bool: `True` onde `env.v3_tarefa` está em `tarefas`. Ela é uma tabela booleana indexada por `v3_tarefa` (`curriculo.py:93-100`).
- `_fim(env, tarefas, valor, paga=False)` faz duas trocas com `torch.where`, e nunca multiplica pela máscara (`recompensas.py:361-365`):
  1. Com `paga=True`: onde `env.v3_perdeu` vale `True`, o valor vira 0 (`recompensas.py:363-364`).
  2. Onde `mascara` vale `False`, o valor vira 0 (`recompensas.py:365`).
- `env.v3_perdeu` vira `True` quando a cadeia detecta a caixa perdida num env cuja tarefa não é `ANDAR`, com `episode_length_buf > 0` e `Cadeia.caixa_perdida_vira_andar` = `True` (`knobs.py:161`; `comando.py:409-414`). Na mesma chamada a cadeia publica a tarefa `ANDAR` para esse env (`comando.py:420`). O env fica então em `ANDAR` com `v3_perdeu` = `True`. O valor volta a `False` no reset do episódio (`comando.py:651-653`). Os 18 termos usam `paga=True` só em `rastreio_linear`, `rastreio_angular`, `passo` e `altura_pe`. O `vivo` zera pelo próprio núcleo (seção 4.4).

### 2.2 `por_restricao` e `restricao`

- `por_restricao(env, ini, so=None)` devolve um fator `(E,)` (`recompensas.py:453-459`):
  - `f = no_degrau(ini, 1,0, restricao, 1) = ini + (1,0 − ini) · restricao / 1` (`recompensas.py:456`, `balanceador.py:48-51`).
  - Com `so` (o nome de uma tarefa), o fator vale só nos envs dessa tarefa, por `TAREFAS.index(so)`; nos outros vale 1 (`recompensas.py:457-458`). Sem `so`, o fator é igual para todos os envs. Dos 18 termos, só os três `desvio_*` passam `so`, com `so="CARREGAR"` (`recompensas.py:710`, `recompensas.py:716`, `recompensas.py:722`). As cercas, do documento 07, passam `so="ANDAR"`.
- Os sete termos de estilo chamam `por_restricao` com `ini` = `Recompensa.estilo_inicio` = 0,25 (`knobs.py:260`). Quatro (`giro_xy`, `vel_junta`, `acel_junta`, `energia`) chamam sem `so`: o fator vai de 0,25 (`restricao` = 0) a 1,0 (`restricao` = 1) em todas as tarefas. Os três `desvio_*` (27/11, T20) chamam com `so="CARREGAR"`: no `CARREGAR` o fator é o mesmo, e no `ANDAR` vale 1, o peso cheio desde a iteração 0 (comentário em `knobs.py:260`).
- `restricao()` é um escalar Python: o menor valor de `nivel/max_nivel` entre as 4 tarefas (`balanceador.py:345-348`).
  - `max_nivel` por tarefa (`balanceador.py:72-74`):
    - `ANDAR` = `round((Comando.frente_tetos[-1] − Comando.andar_inicio) / Comando.andar_passo)` = `round((1,0 − 0,3) / 0,1)` = 7 (`knobs.py:122`, `knobs.py:129`, `knobs.py:134`).
    - `CARREGAR` = `len(Comando.frente_tetos) − 1` = 4 (`knobs.py:122`).
    - `PEGAR` = `len(Niveis.pegar) − 1` = 4 (`knobs.py:417`).
    - `BOTAR` = `len(Niveis.botar) − 1` = 4 (`knobs.py:418`).
  - O nível de uma tarefa só sobe de 1 em 1 (`balanceador.py:276`). Ele sobe quando a tarefa fica no portão por `Balanceador.iters_nivel` = 50 fechos de iteração seguidos (`knobs.py:395`, `balanceador.py:265-276`). `PEGAR` e `BOTAR` gastam antes um fecho de 50 seguidos no subnível: o subnível sobe de 0 a `len(Niveis.subnivel_extra) − 1` = 1 antes de o nível subir (`balanceador.py:273-276`, `balanceador.py:92`, `knobs.py:422`). `ANDAR` e `CARREGAR` não têm subnível (`balanceador.py:326-327`). Fora da restauração de estado salvo (`balanceador.py:416`), a única atribuição que muda o nível é a subida de `balanceador.py:276`.
  - `restricao` muda só quando `fecha_iteracao` roda. A função `curriculo.v3` roda no reset de algum env e chama o fecho uma vez quando `common_step_counter // Balanceador.passos_por_iteracao` difere do valor guardado. `passos_por_iteracao` = 24 (`knobs.py:402`, `curriculo.py:119-122`).
  - O fator `por_restricao` lido por um termo só muda, portanto, num passo em que algum env reseta e o contador de iteração mudou.

## 3. Tabela dos 18 termos

Legenda do fator: `f_r` = `por_restricao(R.estilo_inicio)` = 0,25 + 0,75 · `restricao`, em [0,25; 1] (`recompensas.py:453-459`, `knobs.py:260`). Nos três `desvio_*`, `f_r` vale só no `CARREGAR`; no `ANDAR` o fator vale 1.
`AC` = (`ANDAR`, `CARREGAR`) (`env_cfg.py:34`). `TODAS` = as quatro (`env_cfg.py:33`).

| # | Termo | Tarefas (`env_cfg.py`) | Peso (knob = valor; `knobs.py`) | Sinal | Fator que multiplica o valor | Máximo ou mínimo por segundo |
|---|---|---|---|---|---|---|
| 1 | `rastreio_linear` | AC (`:43`) | `Recompensa.rastreio_linear` = 2,0 (`:236`) | + | canal comandado: `g_marcha`, `exp(−v⊥²/σ²)` com a `v⊥` instantânea, no `CARREGAR` também a qualidade `q`; zera depois da caixa perdida | máx +2,0; mín 0 |
| 2 | `rastreio_angular` | AC (`:43`) | `Recompensa.rastreio_angular` = 1,0 (`:237`) | + | canal comandado: `g_marcha`; no `CARREGAR` também `q`; zera depois da caixa perdida | máx +1,0; mín 0 |
| 3 | `passo` | AC (`:43`) | `Recompensa.passo` = 0,5 (`:244`) | + | `_avanco`, de `piso_avanco` a 1 (seção 7.4); zera depois da caixa perdida | máx +1,0 (2 pés × 0,5); mín 0 |
| 4 | `vivo` | AC (`:44`) | `Recompensa.vivo` = 1,0 (`:248`) | + | nenhum; vale 0 depois da caixa perdida | máx +1,0; mín 0 |
| 5 | `altura_pe` | AC (`:44`) | `Recompensa.altura_pe` = 1,0 (`:249`) | + | `_avanco` (seção 7.4); zera com comando de norma 0 e depois da caixa perdida | máx +1,0; mín tende a 0 (o valor é `exp` e não chega a 0) |
| 6 | `escorrego` | TODAS (`:44`) | `Recompensa.escorrego` = −0,2 (`:253`) | − | nenhum | máx 0; sem piso no código (Σ ‖v_xy‖ dos pés em contato) |
| 7 | `vz` | TODAS (`:44`) | `Recompensa.vz` = −2,0 (`:254`) | − | nenhum | máx 0; sem piso no código (v_z²) |
| 8 | `giro_xy` | TODAS (`:45`) | `Recompensa.giro_xy` = −0,05 (`:255`) | − | `f_r` | máx 0; sem piso no código |
| 9 | `vel_junta` | TODAS (`:45`) | `Recompensa.vel_junta` = −0,001 (`:256`) | − | `f_r` | máx 0; sem piso no código |
| 10 | `acel_junta` | TODAS (`:45`) | `Recompensa.acel_junta` = −5e-8 (`:257`) | − | `f_r` | máx 0; sem piso no código |
| 11 | `taxa_acao` | TODAS (`:45`) | `Recompensa.taxa_acao` = −0,01 (`:258`) | − | nenhum | máx 0; sem piso no código |
| 12 | `energia` | TODAS (`:46`) | `Recompensa.energia` = −2e-5 (`:259`) | − | `f_r` | máx 0; sem piso no código |
| 13 | `limite_junta` | TODAS (`:46`) | `Recompensa.limite_junta` = −5,0 (`:262`) | − | nenhum | máx 0; cada junta custa até 5,0 no batente do modelo; o código não limita acima disso |
| 14 | `desvio_bracos` | AC (`:46`) | `Recompensa.desvio_bracos` = −0,1 (`:264`) | − | `f_r` só no `CARREGAR`; no `ANDAR` vale 1 | máx 0; sem piso no código |
| 15 | `desvio_cintura` | AC (`:47`) | `Recompensa.desvio_cintura` = −1,0 (`:265`) | − | `f_r` só no `CARREGAR`; no `ANDAR` vale 1 | máx 0; sem piso no código |
| 16 | `desvio_quadril` | AC (`:47`) | `Recompensa.desvio_quadril` = −1,0 (`:266`) | − | `f_r` só no `CARREGAR`; no `ANDAR` vale 1 | máx 0; sem piso no código |
| 17 | `tronco_plano` | AC (`:48`) | `Recompensa.tronco_plano` = −20,0 (`:267`) | − | nenhum | máx 0; mín −20,0 (‖g_xy‖² ≤ 1, porque a gravidade é unitária) |
| 18 | `altura_base` | AC (`:48`) | `Recompensa.altura_base` = −10,0 (`:268`) | − | nenhum | máx 0; sem piso no código |

Notas da tabela:
- Os números de linha de "Tarefas" são as de `env_cfg.py` (`_RECOMPENSAS`, `env_cfg.py:42-54`); os de "Peso" são as de `knobs.py`.
- A coluna "máximo ou mínimo por segundo" é `|peso| × extremo do valor`. Ela omite o fator `f_r`, que só reduz o custo de 1 a 0,25, e o fator `_avanco`, que só reduz o ganho de 1 a `piso_avanco` (0,25).
- A gravidade que o `tronco_plano` usa é o vetor `(0, 0, −1)` do mjlab girado pelo quaternion da base (`mjlab/entity/entity.py:809`, `mjlab/entity/data.py:584-586`). Por isso a norma é 1 e `‖g_xy‖² = 1 − g_z²`.

## 4. Entradas comuns

| Leitura | De onde vem | Referência |
|---|---|---|
| `cmd` `(E,3)` = (vx, vy, wz) do comando `twist`, no frame da base | `command_manager.get_command("twist")` devolve `vel_command_b` do termo | `recompensas.py:385`, `mjlab/tasks/velocity/mdp/velocity_command.py:58-59` |
| `v_xy`, `wz` instantâneos da base | `root_link_lin_vel_b[:, :2]` e `root_link_ang_vel_b[:, 2]`: velocidade do link raiz, girada para o frame da base | `recompensas.py:387`, `mjlab/entity/data.py:595-602` |
| `v_filt` `(E,3)`: velocidade filtrada [vx, vy, wz] | buffer do `TwistV3`, atualizado a cada passo no `_update_metrics` | `comando.py:722`, `comando.py:854`, `comando.py:858` |
| `g_marcha` `(E,)` | buffer do `TwistV3`, atualizado a cada passo | `comando.py:725`, `comando.py:871` |
| `v3_fase_marcha` `(E,)` em [0, 1) | buffer do env, escrito pelo `TwistV3` | `curriculo.py:81`, `comando.py:873` |
| `contato` `(E,2)` = `pes_chao.data.found > 0` | sensor de contato: primário = os subárvores dos links `left_ankle_roll_link` e `right_ankle_roll_link`; sem secundário (qualquer contato conta); `found` > 0 significa pelo menos um contato | `cena.py:384-389`, `mjlab/sensor/contact_sensor.py:110` |
| `h` `(E,2)` = `altura_pes.data.heights` | sensor de altura sobre o terreno (grupo de geoms 0) nos dois links de tornozelo: anel de 6 raios de raio 0,03 m, redução pelo mínimo dos raios | `cena.py:392-393`, `cena.py:330-340`, `mjlab/sensor/terrain_height_sensor.py:34`, `mjlab/sensor/terrain_height_sensor.py:83-84` |
| `h_pelve` `(E,)` = `altura_pelve.data.heights[:, 0]` | idem, no corpo `pelvis`: anel de 4 raios de raio 0,05 m | `cena.py:394` |
| velocidade dos pés `(E,2,2)` | `body_link_vel_w[:, pés, :2]`: velocidade linear (x, y) no mundo da origem do link de cada tornozelo | `recompensas.py:652`, `mjlab/entity/data.py:292-298` |
| `joint_vel`, `joint_acc` `(E,29)` | `qvel` e `qacc` das 29 juntas | `mjlab/entity/data.py:374-381` |
| `qfrc_actuator` `(E,29)` | força de atuador no espaço das juntas | `mjlab/entity/data.py:415-423` |
| `joint_pos`, `joint_pos_limits` | posição das juntas e limites `(29,2)` do modelo | `recompensas.py:702-703` |
| `projected_gravity_b` `(E,3)` | gravidade projetada no frame da base | `mjlab/entity/data.py:584-586` |
| `action`, `prev_action` `(E,29)` | saída crua da política no passo corrente e no passo anterior | `recompensas.py:689`, `mjlab/managers/action_manager.py:113-123` |

Os números `0,03` m, `6` raios, `0,05` m e `4` raios são literais de `cena.py:392-394`. Eles não são knobs.

### 4.1 Comando que os termos leem

O comando `twist` é a velocidade pedida à base. Ele muda de valor em três momentos:

| Quando | O que acontece | Referência |
|---|---|---|
| No reset do episódio | `TwistV3.reset` chama `CommandTerm.reset`, que reamostra | `comando.py:893`, `mjlab/managers/command_manager.py:99-107` |
| Na troca de tarefa da cadeia (início de cada trecho) | `_publica` chama `twist._resample(ids)` | `comando.py:519` |
| A cada passo, no ANDAR de terreno com comando linear | O `TwistV3._update_command` reescreve o wz do comando: `clip(heading_control_stiffness · erro de rumo, −0,5; 0,5)` (rigidez 1,0), e zera o wz de módulo menor que `Comando.giro_min`. Só os envs com `is_heading_env` (ANDAR de terreno com ‖(vx, vy)‖ ≥ `Comando.limiar_canal`) | `comando.py:943-948`, `comando.py:932`, `mjlab/tasks/velocity/mdp/velocity_command.py:116-122`, `mjlab/tasks/velocity/mdp/velocity_command.py:284` |
| Por intervalo, quando `time_left` chega a 0 ou menos | `time_left` recebe um sorteio uniforme em `Comando.reamostra_s` = (3,0; 8,0) s, nas reamostras. O valor chega ao termo por `resampling_time_range` | `knobs.py:137`, `env_cfg.py:123`, `mjlab/managers/command_manager.py:109-123` |

Regras do sorteio do vetor (vx, vy, wz), executadas a cada reamostra:
- O comando é zero (0, 0, 0) em todo env que não seja `ANDAR` na fatia `ANDAR` nem `CARREGAR` na fatia `B` (`comando.py:901-905`). Exemplos: `ANDAR` nas fatias `B` e `C`, `PEGAR`, `BOTAR`, `CARREGAR` na fatia `C`.
- No `CARREGAR` da fatia `B`, o comando é sorteado na abertura do trecho e fica fixo: `time_left` vira infinito (`comando.py:940-941`).
- `ANDAR` (`comando.py:125-147`), com `nivel` = nível do `ANDAR` no currículo (de 0 a 7, seção 2.2):
  - frente = `min(Comando.andar_inicio + Comando.andar_passo · nivel, Comando.frente_tetos[-1])`, com 0,3, 0,1 e 1,0 (`knobs.py:129`, `knobs.py:134`, `knobs.py:122`).
  - trás = `min(0,3 + 0,1 · nivel, Comando.tras_teto_max)`, com `tras_teto_max` = 0,5 (`knobs.py:123`).
  - lado = `min(Comando.andar_lado_inicio + 0,1 · nivel, Comando.lado_teto_max)`, com 0,2 e 0,3 (`knobs.py:130`, `knobs.py:124`).
  - giro = `min(Comando.andar_giro_inicio + 0,1 · nivel, Comando.giro_tetos[-1])`, com 0,3 e 0,5 (`knobs.py:131`, `knobs.py:125`).
  - Tipo `andar` (células de `andar`): (vx, vy) sai de uma meia elipse. O raio `r = √U` com `U` uniforme em [0, 1). O ângulo é `π · (U − 0,5)`, uniforme em (−π/2, π/2). O semieixo em x é igual a `frente` com probabilidade `frente/(frente + trás)` e igual a `−trás` no resto. Então `vx = semi_x · r · cos(ângulo)` e `vy = lado · r · sin(ângulo)` (`comando.py:136`, `comando.py:140-142`). `wz = giro · (2U − 1)`, uniforme em [−giro, giro) (`comando.py:143`).
  - Tipo `terreno` (27/11, T19): só `vx = frente · r`, com `r = √U`; `vy` = 0 e `wz` = 0 no sorteio (`comando.py:137-138`). O `wz` vem do rumo a cada passo (tabela acima). A semente do rumo: no começo do episódio, o alvo é o múltiplo de π/2 mais próximo do rumo de partida da base (`comando.py:933-937`).
  - `|wz| < Comando.giro_min` (0,1) vira 0 (`knobs.py:132`, `comando.py:144`).
  - `‖(vx, vy)‖ < Comando.linear_min` (0,1) zera o par (vx, vy) (`knobs.py:133`, `comando.py:145`).
  - Com probabilidade `Comando.andar_parado` = 0,10, os três valores viram 0 (`knobs.py:135`, `comando.py:146`).
  - Na célula `terreno`, `frente` tem teto `Comando.terreno_v_max` = 1,0 (`knobs.py:148`, `comando.py:131-132`). O `trás` tem o mesmo teto, mas o tipo `terreno` não o usa.
- `CARREGAR` (`comando.py:148-158`, `comando.py:102-116`):
  - frente = `Comando.frente_tetos[nivel]` = (0,4; 0,55; 0,7; 0,85; 1,0)[nivel] (`knobs.py:122`); trás = `min(frente, tras_teto_max)`; lado = `min(frente, lado_teto_max)`; giro = `Comando.giro_tetos[min(nivel, 3)]` = (0,3; 0,4; 0,5; 0,5) (`knobs.py:125`).
  - Tipos `reto`, `terreno` e `arco`: escolhe uma de 4 direções (+x, −x, +y, −y) com probabilidade igual; o módulo é uniforme entre o piso e o teto da direção. Piso em x = `Comando.piso_velocidade` = 0,25; piso em y = `Comando.piso_lado` = 0,2 (`knobs.py:126-127`, `comando.py:102-111`).
  - Tipos `arco` e `giro_lugar`: `wz` = ±U[`piso_velocidade`; giro], sinal sorteado com probabilidade 0,5 (`comando.py:113-115`, `comando.py:156-157`). No tipo `giro_lugar`, (vx, vy) fica 0 (`comando.py:153`).
  - Tipo `terreno`: frente e trás passam por `min(·, Comando.terreno_v_max)` = 1,0 (`comando.py:151-152`, `knobs.py:148`).

## 5. Seções por termo

Para cada termo o texto dá: fórmula, entradas, peso, tarefas, máscaras, fatores, quando vale zero e faixa.

### 5.1 `rastreio_linear`

Esta seção e a 5.2 usam a mesma função `_rastreio`. Ela está descrita em detalhe na seção 6.

- **Fórmula.** Invólucro: `_fim(env, tarefas, _rastreio_com_qualidade(env, True), paga=True)` (`recompensas.py:597-599`). Dentro dele: `r = _rastreio(env, True)`. Onde a tarefa é `CARREGAR`, o resultado é `aplica_qualidade(r, q)` = `q · relu(r) + min(r, 0)`. Nos outros envs é `r` (`recompensas.py:590-594`, `recompensas.py:306-308`).
- **Entradas.** `cmd[:, :2]`, `v_xy` (na tenda, só pela `v⊥`), `v_filt[:, :2]` (na projeção da tenda), `g_marcha`, e, no `CARREGAR`, a qualidade `q` (seção 6.4).
- **Peso.** `Recompensa.rastreio_linear` = 2,0 (`knobs.py:236`).
- **Tarefas.** `ANDAR` e `CARREGAR` (`env_cfg.py:43`).
- **Máscaras.** `mascara(AC)`; `paga=True`, então 0 depois de `v3_perdeu`.
- **Fatores.** `g_marcha`; `exp(−v⊥²/σ²)`, com a `v⊥` instantânea (27/11, T20); `q` no `CARREGAR`. Detalhe na seção 6.
- **Vale zero quando.**
  - a tarefa é `PEGAR` ou `BOTAR`;
  - `v3_perdeu` é `True`;
  - o canal é comandado (`‖cmd_xy‖ ≥ Comando.limiar_canal` = 0,02, `knobs.py:138`) e a fração entregue `p` é ≤ 0 ou ≥ 2;
  - o canal é comandado e `g_marcha` = 0;
  - no `CARREGAR`, `q` = 0.
- **Faixa.** Valor em [0, 1]; por segundo em [0, 2,0]. O valor nunca é negativo: o ramo comandado é ≥ 0 e o ramo sem comando é `exp(·)` > 0.

### 5.2 `rastreio_angular`

- **Fórmula.** Igual à do `rastreio_linear`, com `_rastreio(env, False)` (`recompensas.py:602-604`). O canal usa `c = |cmd_z|`, `erro2 = (cmd_z − wz)²`, `v2 = wz²` (`recompensas.py:392-393`).
- **Entradas.** `cmd[:, 2]`, `wz`, `v_filt[:, 2]`, `g_marcha`, `q` no `CARREGAR`.
- **Peso.** `Recompensa.rastreio_angular` = 1,0 (`knobs.py:237`).
- **Tarefas e máscaras.** Iguais às do `rastreio_linear` (`env_cfg.py:43`).
- **Fatores.** `g_marcha`; `q` no `CARREGAR`. O fator da velocidade perpendicular vale 1 (`perp2` = 0, `recompensas.py:406`).
- **Comando de giro no ANDAR de terreno.** Nos envs com `is_heading_env`, o `cmd_z` muda a cada passo: é o giro de alinhamento ao rumo (seção 4.1). O canal angular lê esse `cmd_z` do passo corrente, e o `v_filt` o segue com o filtro (`comando.py:943-948`).
- **Vale zero quando.** Tarefa `PEGAR` ou `BOTAR`; `v3_perdeu`; canal comandado (`|cmd_z| ≥ 0,02`) com `p` ≤ 0 ou ≥ 2; `g_marcha` = 0; `q` = 0 no `CARREGAR`.
- **Faixa.** Valor em [0, 1]; por segundo em [0, 1,0].

### 5.3 `passo`

- **Fórmula.** Núcleo `n_passo(contato, fase, fase_apoio, on)` = `(contato == apoio).float()`, de forma `(E,2)`, com `apoio = (fases < fase_apoio) | ~on[:, None]` e `fases = [fase, (fase + 0,5) mod 1]` (`recompensas.py:75-82`). Invólucro: `_fim(env, tarefas, n_passo(...).sum(-1) * _avanco(env), paga=True)` (`recompensas.py:633`).
- **Entradas.** `contato` do sensor `pes_chao`; `env.v3_fase_marcha`; `on = ‖cmd‖ > Comando.limiar_marcha` (`recompensas.py:631-632`).
- **Peso.** `Recompensa.passo` = 0,5 (`knobs.py:244`).
- **Tarefas.** `ANDAR` e `CARREGAR` (`env_cfg.py:43`).
- **Máscaras.** `mascara(AC)`; `paga=True`.
- **Fatores.** `_avanco` (27/11, T19; seção 7.4): o valor `n_passo(...).sum(-1)` vale × `_avanco(env)` (`recompensas.py:633`).
- **Vale zero quando.** Tarefa `PEGAR` ou `BOTAR`; `v3_perdeu`; nenhum dos dois pés bate com o alvo do relógio.
- **Faixa.** `n_passo` em {0, 1, 2}; `_avanco` em [`piso_avanco`; 1] (seção 7.4). Por segundo, no máximo 1,0 (com `_avanco` = 1).
- Detalhe do relógio e do alvo na seção 7.

### 5.4 `vivo`

- **Fórmula.** Núcleo `n_vivo(perdeu) = 1 − perdeu.float()` (`recompensas.py:85-87`). Invólucro: `_fim(env, tarefas, n_vivo(env.v3_perdeu))`, sem `paga` (`recompensas.py:636-638`). O zero depois da perda vem do próprio núcleo.
- **Entradas.** `env.v3_perdeu`.
- **Peso.** `Recompensa.vivo` = 1,0 (`knobs.py:248`).
- **Tarefas.** `ANDAR` e `CARREGAR` (`env_cfg.py:44`).
- **Máscaras.** `mascara(AC)`.
- **Fatores.** Nenhum.
- **Vale zero quando.** Tarefa `PEGAR` ou `BOTAR`; `v3_perdeu` é `True`.
- **Faixa.** Valor em {0, 1}; por segundo em {0; 1,0}. A detecção da perda exige tarefa diferente de `ANDAR` (`comando.py:409-410`), mas depois dela a cadeia publica a tarefa `ANDAR` com `v3_perdeu` = `True` até o reset (`comando.py:414`, `comando.py:420`). Nesse estado a máscara `AC` passa e o termo vale 0. Um env que nunca perdeu a caixa recebe 1 por passo nas tarefas `ANDAR` e `CARREGAR`.

### 5.5 `altura_pe`

- **Fórmula.** Núcleo `n_altura_pe(h, v_xy, alvo, std, tanh_k)` = `exp(−Σ_pés [(h − alvo)² · tanh(tanh_k · ‖v_xy‖) / std])`, com `h` `(E,2)` e `v_xy` `(E,2,2)` (`recompensas.py:90-97`). A soma corre sobre os dois pés dentro do `exp`: o resultado é um número por env, e não um por pé. Invólucro: `h` = `altura_pes.heights` menos o desconto da pelve (abaixo); `valor = n_altura_pe(h, v, R.alvo_pe, R.std_pe, R.tanh_pe) · _avanco(env)`; depois `where(anda, valor, 0)`; depois `_fim(..., paga=True)` (`recompensas.py:649-656`).
- **Desconto da pelve** (27/11, T20). O corpo que balança para erguer o pé não rende; a perna que encurta rende. Passos (`recompensas.py:649-651`):
  1. `q` = posição de `left_hip_pitch_link` e `right_hip_pitch_link` menos a posição da raiz, no mundo. `_QUADRIS_CORPOS` usa `preserve_order=True`, a mesma ordem esquerda/direita das colunas de `altura_pes` (comentário em `recompensas.py:37`; `recompensas.py:649`).
  2. `z_pelve` = o z de `q` no frame da pelve (`quat_apply_inverse` com o quaternion da raiz) (`recompensas.py:650`).
  3. `h = altura_pes.heights − (q_z − z_pelve)` (`recompensas.py:651`). O desconto vale 0 com a pelve nivelada e tem sinal.
- **Entradas.** `h` do sensor `altura_pes`, já descontado; velocidade horizontal dos tornozelos; `cmd` do `twist`; `anda = cmd.norm(dim=-1) > 0` (`recompensas.py:652-654`).
- **Peso.** `Recompensa.altura_pe` = 1,0 (`knobs.py:249`).
- **Parâmetros.** `alvo_pe` = 0,12 m (`knobs.py:250`; 27/11, T20: era 0,10). `std_pe` = 0,0025 m², que divide o erro ao quadrado (`knobs.py:251`). `tanh_pe` = 2,0 (`knobs.py:252`).
- **σ.** `std_pe` já é σ², pois divide `(h − alvo)²`. Logo σ = √0,0025 = 0,05 m (derivado de `knobs.py:251`).
- **Tarefas.** `ANDAR` e `CARREGAR` (`env_cfg.py:44`).
- **Máscaras.** `mascara(AC)`; `paga=True`; `anda`.
- **Fatores.** O `tanh(tanh_pe · ‖v_xy‖)` pesa o erro de cada pé pela velocidade horizontal dele: pé parado dá custo 0 e valor 1 para aquele pé. O `_avanco` multiplica o valor (27/11, T19; seção 7.4).
- **Vale zero quando.** Tarefa `PEGAR` ou `BOTAR`; `v3_perdeu`; `‖cmd‖` = 0 exato (os 3 canais do `twist` em zero).
- **Faixa.** `n_altura_pe` em (0, 1]; `_avanco` em [`piso_avanco`; 1]. Por segundo em (0, 1,0]. Com comando diferente de zero, dois pés parados dão `n_altura_pe` = 1. O código não tem portão de velocidade do pé.
- Detalhe do alvo e do comando zero na seção 7.

### 5.6 `escorrego`

- **Fórmula.** Núcleo `n_escorrego(v_xy, contato)` = `Σ_pés where(contato, ‖v_xy‖, 0)` (`recompensas.py:100-103`). Invólucro: `_fim(env, tarefas, n_escorrego(v, contato))` (`recompensas.py:659-664`).
- **Entradas.** Velocidade horizontal (x, y) no mundo da origem do link de cada tornozelo; `contato` do `pes_chao`.
- **Peso.** `Recompensa.escorrego` = −0,2 (`knobs.py:253`).
- **Tarefas.** `TODAS` (`env_cfg.py:44`).
- **Máscaras.** `mascara(TODAS)`. Sem `paga`, sem fator de restrição.
- **Vale zero quando.** Nenhum pé em contato; ou todo pé em contato tem velocidade horizontal 0.
- **Faixa.** Valor ≥ 0, em m/s, sem teto. Por segundo ≤ 0.

### 5.7 `vz`

- **Fórmula.** Núcleo `n_vz(v_z) = v_z²` (`recompensas.py:106-108`). Invólucro: `_fim(env, tarefas, n_vz(root_link_lin_vel_w[:, 2]))` (`recompensas.py:667-669`).
- **Entradas.** Componente z, no mundo, da velocidade linear do link raiz.
- **Peso.** `Recompensa.vz` = −2,0 (`knobs.py:254`).
- **Tarefas.** `TODAS` (`env_cfg.py:44`).
- **Máscaras e fatores.** `mascara(TODAS)`; nenhum fator.
- **Vale zero quando.** A velocidade vertical da base é 0.
- **Faixa.** Valor ≥ 0 (m²/s²), sem teto. Por segundo ≤ 0.

### 5.8 `giro_xy`

- **Fórmula.** Núcleo `n_giro_xy(g_xy) = ω_x² + ω_y²` (`recompensas.py:111-113`). Invólucro: `_fim(env, tarefas, n_giro_xy(root_link_ang_vel_b[:, :2])) · por_restricao(env, R.estilo_inicio)` (`recompensas.py:674`).
- **Entradas.** Componentes x e y, no frame da base, da velocidade angular do link raiz.
- **Peso.** `Recompensa.giro_xy` = −0,05 (`knobs.py:255`).
- **Tarefas.** `TODAS` (`env_cfg.py:45`).
- **Máscaras.** `mascara(TODAS)`.
- **Fatores.** `f_r` = 0,25 + 0,75 · `restricao` (seção 2.2).
- **Vale zero quando.** ω_x = ω_y = 0.
- **Faixa.** Valor ≥ 0 (rad²/s²), sem teto; depois do fator `f_r` em [0,25; 1]. Por segundo ≤ 0.

### 5.9 `vel_junta`

- **Fórmula.** Núcleo `n_vel_junta(qd) = Σ_j q̇_j²` sobre as 29 juntas (`recompensas.py:116-118`). Invólucro: `_fim(..., n_vel_junta(joint_vel)) · por_restricao(env, R.estilo_inicio)` (`recompensas.py:679`).
- **Entradas.** `joint_vel` `(E,29)`.
- **Peso.** `Recompensa.vel_junta` = −0,001 (`knobs.py:256`).
- **Tarefas.** `TODAS` (`env_cfg.py:45`).
- **Máscaras e fatores.** `mascara(TODAS)`; `f_r`.
- **Vale zero quando.** Todas as juntas estão paradas.
- **Faixa.** Valor ≥ 0, sem teto. Por segundo ≤ 0.

### 5.10 `acel_junta`

- **Fórmula.** Núcleo `n_acel_junta(qdd) = Σ_j q̈_j²` (`recompensas.py:121-123`). Invólucro: `_fim(..., n_acel_junta(joint_acc)) · por_restricao(env, R.estilo_inicio)` (`recompensas.py:684`).
- **Entradas.** `joint_acc` = `qacc` das 29 juntas (`mjlab/entity/data.py:379-381`).
- **Peso.** `Recompensa.acel_junta` = −5e-8 (`knobs.py:257`).
- **Tarefas.** `TODAS` (`env_cfg.py:45`).
- **Máscaras e fatores.** `mascara(TODAS)`; `f_r`.
- **Vale zero quando.** Todas as acelerações são 0.
- **Faixa.** Valor ≥ 0, sem teto. Por segundo ≤ 0.

### 5.11 `taxa_acao`

- **Fórmula.** Núcleo `n_taxa_acao(a, a_ant) = Σ (a − a_ant)²` sobre as 29 saídas (`recompensas.py:126-128`). Invólucro: `_fim(env, tarefas, n_taxa_acao(am.action, am.prev_action))` (`recompensas.py:687-690`).
- **Entradas.** `action` e `prev_action` do gerente de ação: a saída crua da política, antes de escala e offset.
- **Peso.** `Recompensa.taxa_acao` = −0,01 (`knobs.py:258`).
- **Tarefas.** `TODAS` (`env_cfg.py:45`).
- **Máscaras.** `mascara(TODAS)`. Sem fator de restrição.
- **Vale zero quando.** A ação é igual à do passo anterior.
- **Faixa.** Valor ≥ 0, sem teto. Por segundo ≤ 0.
- **Reset.** No reset do env o gerente de ação zera `action` e `prev_action` (`mjlab/managers/action_manager.py:144-146`). No primeiro passo depois do reset, `prev_action` = 0 e o custo é `Σ a²`.

### 5.12 `energia`

- **Fórmula.** Núcleo `n_energia(tau, qd) = Σ_j |τ_j · q̇_j|` (`recompensas.py:131-133`). Invólucro: `_fim(..., n_energia(qfrc_actuator, joint_vel)) · por_restricao(env, R.estilo_inicio)` (`recompensas.py:697`).
- **Entradas.** `qfrc_actuator` `(E,29)` e `joint_vel` `(E,29)`.
- **Peso.** `Recompensa.energia` = −2e-5 (`knobs.py:259`).
- **Tarefas.** `TODAS` (`env_cfg.py:46`).
- **Máscaras e fatores.** `mascara(TODAS)`; `f_r`.
- **Vale zero quando.** Todo produto τ·q̇ é 0.
- **Faixa.** Valor ≥ 0 (potência mecânica em valor absoluto), sem teto. Por segundo ≤ 0.

### 5.13 `limite_junta`

- **Fórmula.** Núcleo `n_limite_junta(q, lo, hi, mole)` = `potencial.excesso_de_junta(q, (lo + hi)/2, (hi − lo)/2, mole)` (`recompensas.py:136-138`). O `excesso_de_junta` é `Σ_j relu(|q_j − c_j|/m_j − mole) / (1 − mole)`, com `c` o centro e `m` a meia largura do curso (`potencial.py:70-76`, `potencial.py:78-86`). Invólucro: `_fim(env, tarefas, n_limite_junta(joint_pos, lim[..., 0], lim[..., 1], R.limite_mole))` (`recompensas.py:700-704`).
- **Entradas.** `joint_pos` e `joint_pos_limits` do modelo.
- **Peso.** `Recompensa.limite_junta` = −5,0 (`knobs.py:262`).
- **Parâmetro.** `Recompensa.limite_mole` = 0,95 (`knobs.py:263`). O denominador `1 − mole` vale 0,05.
- **Tarefas.** `TODAS` (`env_cfg.py:46`).
- **Máscaras e fatores.** `mascara(TODAS)`. Sem fator de restrição.
- **Vale zero quando.** Toda junta está a menos de 95% do curso a partir do centro.
- **Faixa.** Por junta, 0 até 0,95 do curso e 1 no limite do modelo. Cresce linear dentro dos últimos 5%. O código não limita acima de 1. Por segundo ≤ 0.

### 5.14 `desvio_bracos`

- **Fórmula.** Núcleo `n_desvio(q, alvo, idx) = Σ_{j∈idx} |q_j − alvo_j|` (`recompensas.py:141-143`). Invólucro: `_fim(..., n_desvio(q, _q_alvo(env, q), _BRACOS)) · por_restricao(env, R.estilo_inicio, so="CARREGAR")` (`recompensas.py:707-710`).
- **Juntas.** `_BRACOS` = índices das juntas cujo nome casa `shoulder|elbow|wrist`: 14 juntas, 7 por braço (`recompensas.py:38`, `knobs.py:31-41`).
- **Alvo `_q_alvo`.** No `CARREGAR` (`env.v3_tarefa == CARREGAR`) o alvo é `q_com_caixa(env.v3_meia)`. Nos outros envs é `q_de_pe` (`recompensas.py:412-416`).
  - `q_com_caixa(a)` interpola linearmente em `a` (meia aresta da caixa) entre 3 poses de referência do arquivo de referências (`alvos.py:84-85`, `alvos.py:90-95`).
  - `q_de_pe` é um vetor fixo `(29,)` do arquivo de referências (`alvos.py:65`).
  - O arquivo é `Contrato.referencias` = `g1_v3/dados/referencias.npz` (`knobs.py:73`). Os 29 números dele: Não determinado pelo código lido (o arquivo é binário).
- **Peso.** `Recompensa.desvio_bracos` = −0,1 (`knobs.py:264`).
- **Tarefas.** `ANDAR` e `CARREGAR` (`env_cfg.py:46`).
- **Máscaras e fatores.** `mascara(AC)`; `f_r` só no `CARREGAR`. No `ANDAR` o fator vale 1: o peso é cheio desde a iteração 0 (`so="CARREGAR"`, seção 2.2).
- **Vale zero quando.** Tarefa `PEGAR` ou `BOTAR`; os 14 braços estão no alvo.
- **Faixa.** Valor ≥ 0 (rad), sem teto. Por segundo ≤ 0.

### 5.15 `desvio_cintura`

- **Fórmula.** Igual à do `desvio_bracos`, com `_CINTURA` (`recompensas.py:713-716`).
- **Juntas.** `_CINTURA` = juntas cujo nome contém `waist`: 3 juntas (`waist_yaw`, `waist_roll`, `waist_pitch`) (`recompensas.py:39`, `knobs.py:36`).
- **Alvo, tarefas, máscaras e fatores.** Os mesmos do `desvio_bracos`, com `so="CARREGAR"` (`recompensas.py:716`).
- **Peso.** `Recompensa.desvio_cintura` = −1,0 (`knobs.py:265`).
- **Faixa.** Valor ≥ 0 (rad), sem teto. Por segundo ≤ 0.

### 5.16 `desvio_quadril`

- **Fórmula.** Igual à do `desvio_bracos`, com `_QUADRIL` (`recompensas.py:719-722`).
- **Juntas.** `_QUADRIL` = juntas cujo nome casa `hip_(roll|yaw)`: 4 juntas (roll e yaw, esquerda e direita) (`recompensas.py:40`).
- **Alvo, tarefas, máscaras e fatores.** Os mesmos do `desvio_bracos`, com `so="CARREGAR"` (`recompensas.py:722`).
- **Peso.** `Recompensa.desvio_quadril` = −1,0 (`knobs.py:266`).
- **Faixa.** Valor ≥ 0 (rad), sem teto. Por segundo ≤ 0.

### 5.17 `tronco_plano`

- **Fórmula.** Núcleo `n_tronco_plano(g_xy) = ‖g_xy‖²` (`recompensas.py:146-148`). Invólucro: `_fim(env, tarefas, n_tronco_plano(projected_gravity_b[:, :2]))` (`recompensas.py:725-727`).
- **Entradas.** Componentes x e y da gravidade projetada no frame da base.
- **Peso.** `Recompensa.tronco_plano` = −20,0 (`knobs.py:267`).
- **Tarefas.** `ANDAR` e `CARREGAR` (`env_cfg.py:48`).
- **Máscaras.** `mascara(AC)`. Sem `paga`, sem fator de restrição.
- **Vale zero quando.** Tarefa `PEGAR` ou `BOTAR`; a base está com o eixo z na vertical.
- **Faixa.** Valor em [0, 1]; `‖g_xy‖² = sen²(θ)`, com θ o ângulo entre o eixo z da base e a vertical (derivado de `mjlab/entity/entity.py:809`). Por segundo em [−20,0; 0].

### 5.18 `altura_base`

- **Fórmula.** Núcleo `n_altura_base(h_pelve, alvo) = (h_pelve − alvo)²` (`recompensas.py:151-153`). Invólucro: `_fim(env, tarefas, n_altura_base(h, R.alvo_base))` (`recompensas.py:730-733`).
- **Entradas.** `h` = `altura_pelve.data.heights[:, 0]`, a altura da pelve sobre o terreno (seção 4).
- **Peso.** `Recompensa.altura_base` = −10,0 (`knobs.py:268`).
- **Parâmetro.** `Recompensa.alvo_base` = 0,78 m (`knobs.py:269`).
- **Tarefas.** `ANDAR` e `CARREGAR` (`env_cfg.py:48`).
- **Máscaras.** `mascara(AC)`. Sem fator de restrição. O alvo é o mesmo nas duas tarefas.
- **Vale zero quando.** Tarefa `PEGAR` ou `BOTAR`; a pelve está a exatamente 0,78 m do terreno.
- **Faixa.** Valor ≥ 0 (m²), sem teto. Por segundo ≤ 0.

## 6. O rastreio

### 6.1 Dois canais por termo

O rastreio mede quanto a base cumpre o comando. A função `_rastreio(env, linear, normalizado=False)` devolve um valor `(E,)` e serve aos dois termos (`recompensas.py:374-409`).

| Item | Canal linear | Canal angular |
|---|---|---|
| `c` (tamanho do comando) | `‖cmd_xy‖` | `|cmd_z|` |
| `erro2` | `‖cmd_xy − v_xy‖²` (instantâneo) | `(cmd_z − wz)²` (instantâneo) |
| `v2` | `‖v_xy‖²` (instantâneo); entra também na `v⊥` instantânea (seção 6.2) | `wz²` (instantâneo) |
| `σ₀` (só no modo normalizado) | `Recompensa.sigma_zero_lin` = 0,10 (`knobs.py:241`) | `no_degrau(sigma_zero_ang_inicio, sigma_zero_ang, restricao, 1)` = 0,5 + (0,15 − 0,5) · `restricao` (`knobs.py:243`, `knobs.py:242`) |
| Referências | `recompensas.py:389-391` | `recompensas.py:392-394` |

### 6.2 Canal com comando

O canal é comandado quando `c ≥ Comando.limiar_canal` = 0,02 (`knobs.py:138`, `recompensas.py:409`).

Passos do cálculo (`recompensas.py:398-409`):
1. `v̄ = twist.v_filt`. É a velocidade da base filtrada, com zero no reset do episódio (`comando.py:722`, `comando.py:886`).
2. `den = max(c, limiar_canal)` (`recompensas.py:400`).
3. Linear: `proj = (v̄_xy · cmd_xy) / den`. É a componente de `v̄` ao longo do comando, em m/s. Como `c ≥ limiar_canal` neste ramo, `den = c` e `proj = v̄ · ĉ` (`recompensas.py:402`).
   `perp2 = max(‖v_xy‖² − (v_xy · cmd_xy / den)², 0)`: o quadrado da componente perpendicular ao comando, da velocidade **instantânea** `v_xy` (27/11, T20; a regra de 16/11, T3). A filtrada fazia a média e some com o vaivém lateral de uma passada. A projeção `proj` da tenda segue na filtrada (`recompensas.py:404`).
4. Angular: `proj = v̄_z · sinal(cmd_z)`; `perp2 = 0` (`recompensas.py:406`).
5. Fração entregue `p = proj / den`. No linear, `p = (v̄ · ĉ)/‖c‖`; no angular, `p = ω̄ / c_z` em módulo.
6. `tenda = clamp(1 − |1 − p|, 0, 1) · exp(−perp2 / σ²)`, com `σ = Recompensa.sigma_rastreio` = 0,5 (`knobs.py:238`, `recompensas.py:407`). No canal linear, `perp2` é o da instantânea do passo 3. O fator `1` e o `1` de `1 − |1 − p|` são constantes do código.
7. Valor final do canal: `tenda · g_marcha` (`recompensas.py:409`).

Valores da tenda (derivados da fórmula): `p = 0` vale 0; `p = 1` vale 1; `p = 2` vale 0; `p < 0` e `p > 2` valem 0. Vale em qualquer tamanho de comando.

A velocidade perpendicular instantânea multiplica a tenda por `exp(−perp2/σ²)` só no canal linear. Com `v⊥` instantânea = 0,5 m/s e σ = 0,5, o fator é `e⁻¹` ≈ 0,37 (derivado de `knobs.py:238`).

### 6.3 Canal sem comando

Quando `c < limiar_canal`, o valor é `exp(−erro2 / σ²)` com `σ = sigma_rastreio` = 0,5 (`recompensas.py:409`). Ele usa a velocidade instantânea, e não `v_filt`. Ele não usa `g_marcha`, nem a velocidade perpendicular. Parado, o valor é 1; com deriva `v`, é `exp(−v²/0,25)`.

Com os dois canais sem comando, o rastreio paga até 2,0 + 1,0 = 3,0 por segundo parado (derivado dos pesos `knobs.py:236-237`).

### 6.4 Qualidade no `CARREGAR`

A qualidade `q` entra só onde `env.v3_tarefa == CARREGAR` (`recompensas.py:593-594`). Ela escala só a parte positiva: `q · relu(r) + min(r, 0)`. Como `r ≥ 0`, o resultado é `q · r` (`recompensas.py:306-308`).

Fórmula (`recompensas.py:299-303`, `recompensas.py:572-587`):
`q = núcleo(d_transporte; tol) · rampa(incl; incl_lim, Alvos.delta_incl_graus) · (contatos das palmas) / 2`

| Parcela | Definição | Referência |
|---|---|---|
| `núcleo(d; σ)` | `exp(−(d/σ)²)` | `potencial.py:15-20` |
| `d_transporte` | `‖posição da caixa − alvo de transporte‖`. Na célula `terreno` do `CARREGAR`: alvo = pelve + `caixa_transporte_b` girado pelo rumo da pelve. Fora dela: o alvo da caixa do índice `P3` (`alvo_da_caixa_w`) | `recompensas.py:579-584` |
| `tol` | `tol_transporte_p3` efetivo da zona: `Alvos.tol_transporte_p3` = 0,05 m no valor final e 0,20 m no valor largo; `tol = 0,05 + z_P3 · (0,20 − 0,05)`, com `z_P3` em [0, 1], refeito uma vez por iteração | `knobs.py:187`, `knobs.py:214`, `comando.py:428-441`, `recompensas.py:587` |
| `rampa(x; x_lim, Δ)` | `clamp(1 − (x − x_lim)/Δ, 0, 1)` | `potencial.py:23-29` |
| `incl` | inclinação da caixa em graus (`inclinacao_da_caixa`: ângulo entre a face mais vertical e a vertical) | `recompensas.py:585`, `alvos.py:131-135` |
| `incl_lim` | `por_degrau(Alvos.incl_inicio_graus, Alvos.incl_fim_graus)` = 30° → 15° com o degrau do `CARREGAR`; `Alvos.delta_incl_graus` = 15° | `recompensas.py:462-464`, `knobs.py:203-205`, `balanceador.py:48-51` |
| contatos das palmas | número de sensores `palma_E` e `palma_D` com `found > 0` (0, 1 ou 2); `/2` leva a [0, 1] | `recompensas.py:482-484`, `recompensas.py:303` |

O degrau (`v3_degrau`) do `CARREGAR` vai de 0 a `Balanceador.degraus` = 5 (`knobs.py:401`). A cadeia o publica a cada troca de tarefa, em `_publica` (`comando.py:512-518`).

### 6.5 Fator `g_marcha`

`g_marcha` é a fração do relógio de marcha que os pés cumprem, remapeada de [0,5; 1] para [0; 1]. O cálculo vive no `TwistV3._update_metrics`, a cada passo, depois da recompensa (`comando.py:863-871`):

1. `on = ‖cmd‖ > Comando.limiar_marcha` = 0,1 (`knobs.py:141`, `comando.py:863`).
2. `acerto = n_passo(contato, v3_fase_marcha, fase_apoio, on).mean(-1)`: a média dos dois pés (`comando.py:868`).
3. Onde `on` vale `True` e `v3_marcha_on` (o `on` do passo anterior) vale `False`, `q̄ ← 0,5` (`comando.py:869`).
4. `q̄ ← q̄ + (1 − exp(−dt/τ)) · (acerto − q̄)`, com `τ = Comando.tau_fracao` = 0,4 s e `dt` = 0,02 s (`knobs.py:136`, `comando.py:870`). O ganho é 1 − e^(−0,05) ≈ 0,049 (derivado).
5. `g_marcha = clamp(2 · q̄ − 1, 0, 1)` (`comando.py:871`).

Os números `0,5`, `2` e `1` dos passos 3 e 5 são constantes do código.

No reset do episódio: `q̄ ← 0,5` e `g_marcha ← 0` (`comando.py:887-888`).

Efeito: `acerto` = 0,5 (acaso) dá `g` = 0; `acerto` = 1 dá `g` = 1. Com `on = False`, o `acerto` mede os pés no chão (alvo "ambos em contato"), e vale perto de 1 com os pés plantados.

O filtro de `v̄` usa o mesmo `τ`: `v_filt ← v_filt + (1 − exp(−dt/τ)) · (v3 − v_filt)`, com `v3 = [vx, vy, wz]` instantâneos (`comando.py:854`, `comando.py:858`).

### 6.6 Modo normalizado

`_rastreio(env, linear, normalizado=True)` chama `potencial.rastreio_normalizado(erro2, c, v2, σ, σ₀, limiar, piso)` (`recompensas.py:395-397`). Ele não usa `v_filt`, nem `g_marcha`, nem `q`.

Fórmula (`potencial.py:88-106`):
- `c_n = max(c, limiar)`.
- Com `c ≥ limiar`: `(exp(−erro2/σ²) − exp(−c_n²/σ²)) / (1 − exp(−c_n²/σ²))`. Seguir o comando vale 1; parado vale 0.
- Com `c < limiar`: `exp(−v2/σ₀²) − 1`. Só o deslize custa.
- O resultado passa por `clamp(·, min=piso)`.

Valores dos knobs:
- `σ` = `Recompensa.sigma_rastreio` = 0,5 (`knobs.py:238`).
- `σ₀` linear = `Recompensa.sigma_zero_lin` = 0,10 (`knobs.py:241`).
- `σ₀` angular = 0,5 + (0,15 − 0,5) · `restricao`, com `sigma_zero_ang_inicio` = 0,5 e `sigma_zero_ang` = 0,15 (`knobs.py:243`, `knobs.py:242`).
- `limiar` = `Comando.limiar_canal` = 0,02 (`knobs.py:138`).
- `piso` = `Recompensa.piso_rastreio` = −1,0 (`knobs.py:239`).

Quem usa o modo normalizado: o termo `progresso`, em `recompensas.py:861`. Ele monta o sinal `comandado` = `g_marcha · (com_lin · clamp(r_lin, 0, 1) + com_ang · clamp(r_ang, 0, 1)) / max(com_lin + com_ang, 1)`, a média dos canais comandados (`com_lin`: `‖cmd_xy‖ ≥ Comando.limiar_canal`; `com_ang`: `|cmd_z| ≥ Comando.limiar_canal`), ou o sinal de deriva `clamp((1 + r_lin)(1 + r_ang), 0, 1)` quando nenhum canal é comandado (`recompensas.py:860-865`). O sinal é multiplicado por `rampa(d_transporte; Recompensa.qualidade_larga)` = `rampa(d; 0,05; 0,25)` (`knobs.py:351`, `recompensas.py:866`). O sinal só altera `v3_h` onde `carrega` vale `True` (`recompensas.py:867-868`). Os termos `rastreio_linear` e `rastreio_angular` nunca usam o modo normalizado.

## 7. O relógio de marcha, `passo` e `altura_pe`

### 7.1 O relógio

O relógio de marcha é um número `fase` em [0, 1) que gira com o tempo de comando.
- `_fase_t` soma `dt` a cada passo em que `on` vale `True`. Com `on = False`, ele fica parado (`comando.py:872`).
- `v3_fase_marcha = (_fase_t / Comando.periodo_marcha) mod 1`, com `periodo_marcha` = 0,8 s (`knobs.py:140`, `comando.py:873`).
- No reset do episódio, `_fase_t ← 0` (`comando.py:885`).
- O relógio escreve depois da recompensa. O `passo` do passo `t` lê a fase do passo `t−1` (seção 1.3).

### 7.2 Alvo de contato do `passo`

`n_passo` compara o contato de cada pé com o alvo `apoio` (`recompensas.py:80-82`):
- Pé na posição 0: fase = `fase`. Pé na posição 1: fase = `(fase + 0,5) mod 1`. O `0,5` é constante do código (`recompensas.py:80`).
- Com `on = True`: o pé deve estar em contato quando `fase < Comando.fase_apoio` = 0,55 (`knobs.py:139`), e no ar no resto.
- Com `on = False` (`‖cmd‖ ≤ Comando.limiar_marcha` = 0,1): `apoio = True` para os dois pés. Cada pé em contato vale 1; cada pé no ar vale 0. O relógio está parado: o alvo dos dois pés é o chão. O código não tem outra regra para comando zero (`recompensas.py:81`).
- Por segundo parado: até 2 pés × 0,5 = 1,0 (derivado de `knobs.py:244`).

Fatos derivados do comparador (com `on = True`, `fase_apoio` = 0,55, `periodo_marcha` = 0,8):
- Pé 0 em apoio: `fase ∈ [0; 0,55)`. Pé 1 em apoio: `fase ∈ [0; 0,05) ∪ [0,5; 1)`.
- Os dois pés em apoio: `fase ∈ [0; 0,05) ∪ [0,5; 0,55)`, 10% do ciclo: duas janelas de 0,04 s, 0,08 s por ciclo de 0,8 s.
- Cada pé fica 0,45 do ciclo no ar: 0,36 s.
- Com `on = True`, o alvo não permite os dois pés no ar.

O `n_passo` não tem portão de velocidade. Desde 27/11 (T19), o termo `passo` vale × `_avanco` (seção 7.4): marchar no lugar com comando diferente de zero paga o piso, e atravessar paga o valor inteiro. No nível 0 do ANDAR o piso é 1,0 e o `passo` paga até 1,0 por segundo (peso 0,5 × 2 pés), como antes. Quem corta o ganho do rastreio nesse caso é o `g_marcha`.

### 7.3 Alvo de altura do `altura_pe`

- O alvo é `Recompensa.alvo_pe` = 0,12 m (27/11, T20: era 0,10) de altura do link do tornozelo sobre o terreno, depois do desconto da pelve da seção 5.5 (`knobs.py:250`).
- O erro de cada pé é `(h − 0,12)²`, dividido por `std_pe` = 0,0025 m² e multiplicado por `tanh(tanh_pe · ‖v_xy‖)` com `tanh_pe` = 2,0 (`knobs.py:251-252`, `recompensas.py:96`).
- Com velocidade horizontal igual a 0, o custo do pé é 0. Com `‖v_xy‖` = 0,5 m/s, `tanh(1,0)` ≈ 0,76 (derivado de `knobs.py:252`).
- O valor é `exp(−custo_pé0 − custo_pé1)` (`recompensas.py:97`), vezes `_avanco` (`recompensas.py:655`).
- O termo vale 0 em todo env cuja norma do comando seja exatamente 0 (`recompensas.py:653-656`). Esse limite (`> 0`) difere do limite do relógio (`> limiar_marcha`). Um comando de norma entre 0 e 0,1 paga `altura_pe` e mantém o relógio parado.
- Altura do `h` com o pé apoiado no chão: Não determinado pelo código lido (depende da geometria do link do tornozelo no modelo). O comentário do knob diz que o sensor mede o tornozelo, a 3,5 cm do chão no apoio (`knobs.py:250`). O código não mede esse número.

### 7.4 O avanço, fator do `passo` e do `altura_pe`

`_avanco(env)` devolve `(E,)`: `piso + (1 − piso) · p` (`recompensas.py:607-616`; 27/11, T19).
- `p` é a tenda da fração entregue do comando, já × `g_marcha` (`_rastreio`, seção 6): a do canal linear se `‖cmd_xy‖ ≥ Comando.limiar_canal`; senão a do canal angular se `|cmd_z| ≥ Comando.limiar_canal`; sem comando, `p` = 1 (`recompensas.py:613-614`). A qualidade `q` do `CARREGAR` não entra. O `_rastreio` lê `v_filt` e `g_marcha` do passo anterior (seção 1.3).
- `piso = no_degrau(1,0; piso_avanco; nivel("ANDAR"); max_nivel["ANDAR"])` = 1,0 + (`piso_avanco` − 1,0) · nível/7, com `Recompensa.piso_avanco` = 0,25 (`recompensas.py:615`, `knobs.py:245`). O piso segue o nível do `ANDAR`, também nos envs do `CARREGAR`.
- Piso por nível do ANDAR (derivado da fórmula): 0 → 1,00; 1 → 0,89; 2 → 0,79; 4 → 0,57; 7 → 0,25. No nível 0 o `_avanco` vale 1 e não corta nada (a partida do zero é a da T16, comentário em `recompensas.py:610`).
- Efeito: marchar no lugar com comando dá `p` = 0 e `_avanco` = piso. Entregar o comando dá `p` = 1 e `_avanco` = 1. No nível 7, o `passo` no lugar paga no máximo 1,0 × 0,25 = 0,25 por segundo (derivado).
- Motivo do piso (comentário do knob, `knobs.py:245`): com piso 0 uma corrida do zero nasce morta, porque `passo`, `g_marcha` e rastreio dependeriam um do outro.

## 8. Os knobs que esses termos leem

### 8.1 Classe `Recompensa`

| Knob | Valor | Usado em | Linha |
|---|---|---|---|
| `rastreio_linear` | 2,0 | peso do `rastreio_linear` | `knobs.py:236` |
| `rastreio_angular` | 1,0 | peso do `rastreio_angular` | `knobs.py:237` |
| `sigma_rastreio` | 0,5 | σ do rastreio (tenda, canal sem comando, modo normalizado) e da conta de deriva do segmento em `TwistV3` | `knobs.py:238`; `recompensas.py:396`, `recompensas.py:407`, `recompensas.py:409`; `comando.py:787` |
| `piso_rastreio` | −1,0 | piso do modo normalizado | `knobs.py:239` |
| `sigma_zero_lin` | 0,10 | σ₀ linear do modo normalizado | `knobs.py:241` |
| `sigma_zero_ang` | 0,15 | σ₀ angular final do modo normalizado | `knobs.py:242` |
| `sigma_zero_ang_inicio` | 0,5 | σ₀ angular com `restricao` = 0 | `knobs.py:243` |
| `passo` | 0,5 | peso do `passo` | `knobs.py:244` |
| `piso_avanco` | 0,25 | piso do `_avanco` no nível máximo do ANDAR | `knobs.py:245` |
| `vivo` | 1,0 | peso do `vivo` | `knobs.py:248` |
| `altura_pe` | 1,0 | peso do `altura_pe` | `knobs.py:249` |
| `alvo_pe` | 0,12 m | altura-alvo do pé (era 0,10 até 27/11, T20) | `knobs.py:250` |
| `std_pe` | 0,0025 m² | divisor do erro ao quadrado (σ = 0,05 m) | `knobs.py:251` |
| `tanh_pe` | 2,0 | escala do `tanh` da velocidade do pé | `knobs.py:252` |
| `escorrego` | −0,2 | peso | `knobs.py:253` |
| `vz` | −2,0 | peso | `knobs.py:254` |
| `giro_xy` | −0,05 | peso | `knobs.py:255` |
| `vel_junta` | −0,001 | peso | `knobs.py:256` |
| `acel_junta` | −5e-8 | peso | `knobs.py:257` |
| `taxa_acao` | −0,01 | peso | `knobs.py:258` |
| `energia` | −2e-5 | peso | `knobs.py:259` |
| `estilo_inicio` | 0,25 | `ini` do fator `por_restricao` dos 7 termos de estilo; os 3 `desvio_*` o usam só no `CARREGAR` | `knobs.py:260` |
| `limite_junta` | −5,0 | peso | `knobs.py:262` |
| `limite_mole` | 0,95 | fração do curso onde a rampa começa | `knobs.py:263` |
| `desvio_bracos` | −0,1 | peso | `knobs.py:264` |
| `desvio_cintura` | −1,0 | peso | `knobs.py:265` |
| `desvio_quadril` | −1,0 | peso | `knobs.py:266` |
| `tronco_plano` | −20,0 | peso | `knobs.py:267` |
| `altura_base` | −10,0 | peso | `knobs.py:268` |
| `alvo_base` | 0,78 m | altura-alvo da pelve | `knobs.py:269` |

Os campos `L_min` (0,20) e `fator_T` (0,8; 1,4) existem em `Recompensa` (`knobs.py:246-247`). Nenhum dos 18 termos os lê. A função `potencial.L_alvo` os usa e nenhum termo de `recompensas.py` a chama.

### 8.2 Classe `Comando` (o que os termos leem do `twist`)

| Knob | Valor | Usado em | Linha |
|---|---|---|---|
| `limiar_canal` | 0,02 | separa canal com comando de canal sem comando | `knobs.py:138` |
| `limiar_marcha` | 0,1 | liga o relógio (`on`) | `knobs.py:141` |
| `fase_apoio` | 0,55 | fração do ciclo com o pé no chão | `knobs.py:139` |
| `periodo_marcha` | 0,8 s | período do relógio | `knobs.py:140` |
| `tau_fracao` | 0,4 s | τ do filtro de `v_filt` e de `q̄` | `knobs.py:136` |
| `reamostra_s` | (3,0; 8,0) s | intervalo de reamostra do comando | `knobs.py:137` |
| `frente_tetos` | (0,4; 0,55; 0,7; 0,85; 1,0) m/s | tetos do comando; `max_nivel` | `knobs.py:122` |
| `tras_teto_max` | 0,5 m/s | teto de trás | `knobs.py:123` |
| `lado_teto_max` | 0,3 m/s | teto de lado | `knobs.py:124` |
| `giro_tetos` | (0,3; 0,4; 0,5; 0,5) rad/s | tetos de giro | `knobs.py:125` |
| `piso_velocidade`, `piso_lado` | 0,25; 0,2 m/s | pisos do `CARREGAR` | `knobs.py:126-127` |
| `andar_inicio`, `andar_lado_inicio`, `andar_giro_inicio`, `andar_passo` | 0,3; 0,2; 0,3; 0,1 | tetos do `ANDAR` por nível | `knobs.py:129-131`, `knobs.py:134` |
| `giro_min`, `linear_min` | 0,1; 0,1 | comando zero ou acima destes | `knobs.py:132-133` |
| `andar_parado` | 0,10 | fração dos sorteios do `ANDAR` com comando zero | `knobs.py:135` |
| `terreno_v_max` | 1,0 m/s | teto na célula `terreno` | `knobs.py:148` |

### 8.3 Outros knobs lidos

| Knob | Valor | Usado em | Linha |
|---|---|---|---|
| `Contrato.dt_fisica` | 0,005 s | `step_dt` | `knobs.py:54` |
| `Contrato.decimacao` | 4 | `step_dt` | `knobs.py:55` |
| `Cadeia.episodio_s` | 20,0 s | log por episódio do gerente | `knobs.py:160` |
| `Cadeia.caixa_perdida_vira_andar` | `True` | `v3_perdeu` vira `True` na perda | `knobs.py:161` |
| `Balanceador.iters_nivel` | 50 | subida de nível; muda `restricao` | `knobs.py:395` |
| `Balanceador.passos_por_iteracao` | 24 | fecho de iteração do currículo | `knobs.py:402` |
| `Balanceador.degraus` | 5 | `por_degrau` | `knobs.py:401` |
| `Alvos.tol_transporte_p3` | 0,05 m (final); 0,20 m (largo) | tolerância da qualidade `q` | `knobs.py:187`, `knobs.py:214` |
| `Alvos.incl_inicio_graus` | 30,0° | `incl_lim` no degrau 0 | `knobs.py:203` |
| `Alvos.incl_fim_graus` | 15,0° | `incl_lim` no degrau 5 | `knobs.py:204` |
| `Alvos.delta_incl_graus` | 15,0° | largura da rampa de inclinação | `knobs.py:205` |
| `Niveis.pegar`, `Niveis.botar` | 5 entradas cada | `max_nivel` de `PEGAR` e `BOTAR` = 4 | `knobs.py:417-418` |

## 9. Outros comportamentos do código dentro do assunto

1. **Zero por `torch.where`, e não por produto.** Os termos mascaram com `torch.where`. Isso impede que um NaN de um env fora da tarefa chegue à soma (`recompensas.py:361-365`). O gerente ainda troca NaN e infinito por 0 em cada termo (`mjlab/managers/reward_manager.py:129`).
2. **Ordem do fator de restrição.** O fator `por_restricao` multiplica o resultado de `_fim`, que já tem zero fora das tarefas (`recompensas.py:674`).
3. **Os 7 termos com `f_r`.** São `giro_xy`, `vel_junta`, `acel_junta`, `energia`, `desvio_bracos`, `desvio_cintura` e `desvio_quadril`. Todos usam `ini` = `estilo_inicio` = 0,25. Os três `desvio_*` o aplicam só no `CARREGAR` (`so="CARREGAR"`); no `ANDAR` o fator vale 1. Os 11 termos restantes não têm esse fator (`passo` e `altura_pe` têm o `_avanco`, seção 7.4): `rastreio_linear`, `rastreio_angular`, `passo`, `vivo`, `altura_pe`, `escorrego`, `vz`, `taxa_acao`, `limite_junta`, `tronco_plano`, `altura_base`.
4. **`passo` e `rastreio` compartilham o relógio.** O núcleo `n_passo` é chamado pelo termo `passo` e pelo `TwistV3._update_metrics`, com o mesmo contato, a mesma fase e o mesmo `on` (`recompensas.py:633`, `comando.py:868`). O `acerto` que gera `g_marcha` é, portanto, o mesmo número que o `passo` paga antes do `_avanco`, dividido por 2.
5. **Termo com peso zero não roda.** O gerente pula uma função cujo peso vale 0 (`mjlab/managers/reward_manager.py:122-124`). Os buffers `g_marcha`, `v_filt` e `v3_fase_marcha` são escritos em `TwistV3._update_metrics`, que roda a cada passo do env sem depender do peso de nenhum termo (`comando.py:810-877`).
6. **Dois alvos de postura.** `_q_alvo` troca `q_de_pe` por `q_com_caixa(a)` só na tarefa `CARREGAR`. `desvio_*` e `rastreio_*` leem a tarefa corrente do env, e não o índice do alvo (`recompensas.py:412-416`).
7. **Ordem de dois pés.** No `n_passo`, a posição 0 do tensor `(E,2)` recebe `fase` e a posição 1 recebe `fase + 0,5` (`recompensas.py:80`). Qual pé é a posição 0 é Não determinado pelo código lido (ver seção 10).
8. **O ator vê o relógio.** A observação `fase=True` entrega seno e cosseno de `2π · v3_fase_marcha` e `v3_marcha_on` (`observacoes.py:313`, `observacoes.py:318`). `TwistV3.reset` não zera `v3_fase_marcha` nem `v3_marcha_on`: o próximo `_update_metrics` os reescreve (`comando.py:879-895`, `comando.py:873-874`).
9. **Gravidade e `tronco_plano`.** O valor depende só da inclinação da base, e não do rumo (yaw), pois usa só as componentes x e y da gravidade no frame da base.
10. **`mede_deriva.py:113`** chama `_rastreio(env, True)` para medir a tenda sem peso e sem qualidade. Esse uso fica fora do treino.
11. **Funções de `potencial.py` sem chamador no treino.** `potencial.dobradica(x, limite)` devolve `relu(x − limite)` (`potencial.py:65-67`). Nenhum módulo de treino a chama: as cercas de `recompensas.py` fazem o `relu` direto. `potencial.L_alvo` também não tem chamador no treino (seção 8.1).

## 10. Não determinado pelo código lido

- Não determinado pelo código lido: a ordem esquerdo/direito dos dois pés nos tensores `(E,2)` do sensor `pes_chao`, do sensor `altura_pes` e de `body_link_vel_w`. O `find_bodies` sem `preserve_order` devolve a ordem dos corpos do modelo, e a ordem dos corpos do modelo não está no código lido (`mjlab/entity/entity.py:512-515`).
- Não determinado pelo código lido: os 29 valores de `q_de_pe` e de `q_com_caixa`. Eles vêm do arquivo binário `referencias.npz` (`alvos.py:65`, `alvos.py:63`).
- Não determinado pelo código lido: a altura `h` do link do tornozelo sobre o terreno com o pé apoiado.
- Não determinado pelo código lido: o valor de `restricao` ao longo do treino. Ele depende do desempenho medido pelo balanceador.
