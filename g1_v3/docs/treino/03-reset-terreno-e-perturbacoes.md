# Reset, terreno, perturbações e randomização de domínio

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:

- Código da v3: `eventos.py`, `cena.py`, `env_cfg.py`, `knobs.py`, `balanceador.py`, `curriculo.py`, `tarefa.py`.
- Código da v3, só trechos: `alvos.py` (`base_para_mundo`, `Referencias`, `laje_destino`), `comando.py` (construção, `_abre_botar`, `_abre_trecho`, `_resample_command`, reset, `_update_metrics`), `observacoes.py` (grupos, `giro_base`, `gravidade`), `estimador.py` (`reinicia`, `_sorteia`), `percepcao_lidar.py` (`reinicia`), `runner.py` (`save`, `load`), `terminacoes.py` (`fim_da_cadeia`), `train.py`, `play.py`.
- mjlab: `scripts/play.py`, `actuator/builtin_group.py`, `envs/mdp/events.py`, `managers/event_manager.py`, `managers/manager_base.py`, `managers/scene_entity_config.py`, `managers/observation_manager.py`, `envs/manager_based_rl_env.py`, `envs/mdp/dr/_core.py`, `_types.py`, `geom.py`, `body.py`, `joint.py`, `actuator.py`, `envs/mdp/actions/actions.py`, `entity/data.py`, `actuator/actuator.py`, `utils/buffers/delay_buffer.py`, `utils/lab_api/math.py`, `terrains/terrain_generator.py`, `terrain_entity.py`, `config.py`, `primitive_terrains.py`, `heightfield_terrains.py`, `scene/scene.py`, `tasks/velocity/config/g1/env_cfgs.py`, `tasks/velocity/velocity_env_cfg.py`.

Convenções do texto:

- Os caminhos de biblioteca são relativos a `site-packages`: `mjlab/...`.
- "Calculado" marca um número que não está escrito no código. Ele sai de uma conta sobre as linhas citadas.
- Decimal com vírgula. Intervalo `(a; b)` é uniforme entre `a` e `b`, salvo outra indicação.
- Um episódio acaba por terminação ou por tempo. Um passo é um passo da política: `Contrato.dt_fisica` × `Contrato.decimacao` = 0,005 s × 4 = 0,02 s (`knobs.py:54-55`; `mjlab/envs/manager_based_rl_env.py:266-268`).
- O sufixo "no play" quer dizer `make_env_cfg(play=True)` (`env_cfg.py:80`).

---

## 1. Quando cada modo de evento roda

Um evento é uma função que o gerente de eventos do mjlab chama num momento fixo. Cada evento tem um modo.

| Modo | Quando roda | Em quem | Referência |
|---|---|---|---|
| `startup` | Uma vez, no fim de `load_managers`, com todos os gerentes já criados e antes do primeiro reset. | Todos os envs (`env_ids` = None). | `mjlab/envs/manager_based_rl_env.py:353-354`; `mjlab/managers/event_manager.py:324-326` |
| `reset` | Dentro de `_reset_idx`, para os envs que terminaram o episódio, na ordem do dicionário do cfg. | Só os envs em reset. | `mjlab/envs/manager_based_rl_env.py:553-562`; `mjlab/managers/event_manager.py:292-306` |
| `step` | A cada passo, depois de `command_manager.compute` e antes dos eventos de `interval`. | Todos os envs (`env_ids` = None). | `mjlab/envs/manager_based_rl_env.py:456-459`; `mjlab/managers/event_manager.py:289-291` |
| `interval` | A cada passo o gerente subtrai `step_dt` do tempo restante de cada env. O evento dispara no env em que o tempo restante fica abaixo de 1e-6 s. | Só os envs que dispararam. | `mjlab/envs/manager_based_rl_env.py:460-461`; `mjlab/managers/event_manager.py:263-287` |

Ordem completa do `_reset_idx` (`mjlab/envs/manager_based_rl_env.py:553-591`):

1. `curriculum_manager.compute` (aqui roda o termo `v3`, `curriculo.py:103`).
2. `sim.reset` e `scene.reset`.
3. Eventos de `reset`, na ordem do cfg.
4. Reset dos gerentes de observação, ação, recompensa, métrica, currículo, **comando**, evento e terminação.

Consequência: a fatia, a cena e a base já estão escritas quando o gerente de comando abre o primeiro trecho do episódio.

Ordem dos eventos de `reset` no cfg (`env_cfg.py:153-184`): `fatia`, `cena`, `base`, `juntas`, `carga`, `atrito_pad`, `atrito_laje`, `vies_imu`, `pd`, `forca_motor`, `solref`, `com_caixa` (só com a chave ligada), `filtros`. O gerente guarda os termos na ordem em que o cfg os traz (`mjlab/managers/event_manager.py:347-358`).

Ordem por passo (`mjlab/envs/manager_based_rl_env.py:421-461`): física (4 subpassos) → terminações e recompensas → reset dos envs terminados → `forward` → comando → eventos `step` (`forca_caixa`, `rfi`, nessa ordem de cfg) → eventos `interval` (`empurrao`).

Quando o gerente chama um evento `reset`, ele registra o passo do disparo e chama a função com os `env_ids` do reset (`mjlab/managers/event_manager.py:300-306`). O campo `min_step_count_between_reset` vale 0 por padrão (`mjlab/managers/event_manager.py:138`) e nenhum termo de `env_cfg.py` o altera: todo evento de reset roda em todo reset.

Depois que os eventos de `reset` rodam, o gerente recalcula as constantes do modelo no nível mais forte entre os eventos que dispararam (`mjlab/managers/event_manager.py:328-333`). Os eventos `carga`, `com_caixa` e `massa_tronco` pedem o nível `set_const` (`eventos.py:443`, `eventos.py:568`; `mjlab/envs/mdp/dr/body.py:284`).

### 1.1 Qual gerador de números aleatórios cada sorteio usa

O env tem um gerador próprio, `v3_gerador`, criado na construção do `CadeiaCmd` por `garante_v3` (`comando.py:293`; `curriculo.py:62-64`). Ele nasce com a semente `cfg.seed`, ou 0 se `cfg.seed` é vazio ou 0 (`curriculo.py:64`). O outro gerador é o global do torch.

| Gerador | Quem sorteia |
|---|---|
| `v3_gerador` | `sorteia_fatia` (fatia, terreno, coluna, linha), `posiciona_cena` (topo, massa, rumo, jitter), `_base_da_pega`, `tamanho_caixa`, `vies_imu`, `pd`, `forca_motor`, `solref`, `com_caixa`, `rfi`, `sorteia_destino`, `reinicia_filtros` (estimador) |
| global do torch | `reset_root_state_uniform` (base do ANDAR), `atrito_pad`, `atrito_laje`, `atrito_pe`, `massa_tronco`, `vies_encoder`, `push_by_setting_velocity`, `apply_body_impulse` (`forca_caixa`), tempos do `interval`, lag dos buffers de atraso |

Referências: `eventos.py:60`, `eventos.py:79`, `eventos.py:191`, `eventos.py:430`, `eventos.py:498-502`, `eventos.py:516-520`, `eventos.py:560`, `eventos.py:613`, `eventos.py:381`; `mjlab/utils/lab_api/math.py:1361-1379` (`sample_uniform` usa `torch.rand` sem gerador); `mjlab/envs/mdp/events.py:174-190`, `:336`, `:546`; `mjlab/utils/buffers/delay_buffer.py:269-277` (`generator` = None, porque `cena.py` e `observacoes.py` não o passam).

---

## 2. Tabela de todos os eventos de `env_cfg.py`

São 20 eventos com a chave `com_caixa` desligada (valor lido: `DR.com_caixa` = False, `knobs.py:527`) e 21 com ela ligada. A coluna "Play" diz se o evento existe em `make_env_cfg(play=True)`.

| Nome | Modo | Função | Parâmetros do cfg | Play | Linha |
|---|---|---|---|---|---|
| `fatia` | reset | `eventos.sorteia_fatia` | `k_ter` = `k.terreno` (a classe `Terreno` de `knobs.py`) | sim | `env_cfg.py:154` |
| `cena` | reset | `eventos.posiciona_cena` | nenhum | sim | `env_cfg.py:156` |
| `base` | reset | `eventos.reset_base_por_fatia` | `faixa_andar` = `Cena.reset_base_andar`: x (−0,5; 0,5) m, y (−0,5; 0,5) m, yaw (−π; π) rad (`knobs.py:103`). `faixa_manipula` = `Cena.reset_base_manipula`: z (0,01; 0,05) m, yaw (−0,2; 0,2) rad (`knobs.py:101`). | sim | `env_cfg.py:157-158` |
| `juntas` | reset | `reset_joints_by_offset` (mjlab) | `position_range` (0,0; 0,0), `velocity_range` (0,0; 0,0), escritos no próprio `env_cfg.py` | sim | `env_cfg.py:159-160` |
| `carga` | reset | `eventos.carga` | nenhum | sim | `env_cfg.py:161` |
| `atrito_pad` | reset | `eventos.atrito_pad` | `asset_cfg` = geoms `cena.PALM_PAD_GEOMS` do robô (`left_palm_pad`, `right_palm_pad`) | sim | `env_cfg.py:162-163` |
| `atrito_laje` | reset | `eventos.atrito_laje` | `asset_cfg` = geom `table_geom` da entidade `table` | sim | `env_cfg.py:164-165` |
| `vies_imu` | reset | `eventos.vies_imu` | nenhum | sim | `env_cfg.py:166` |
| `pd` | reset | `eventos.pd` | `asset_cfg` = robô inteiro | **não** | `env_cfg.py:170` |
| `forca_motor` | reset | `eventos.forca_motor` | `asset_cfg_pernas` = grupos de atuador 0 a 6; `asset_cfg_bracos` = grupos 7 e 8 | **não** | `env_cfg.py:173-175` |
| `solref` | reset | `eventos.solref` | geoms do robô: os 14 `*_foot{1..7}_collision`, os 2 pads de palma e os 2 pads de dorso; geom da laje; geom da caixa | **não** | `env_cfg.py:177-180` |
| `com_caixa` | reset | `eventos.com_caixa` | `asset_cfg` = corpo `box`. Só entra no cfg se `DR.com_caixa` é True. | sim, se registrado | `env_cfg.py:182-183` |
| `filtros` | reset | `eventos.reinicia_filtros` | nenhum | sim | `env_cfg.py:184` |
| `atrito_pe` | startup | `dr.geom_friction` | `ranges` = `DR.atrito_pe` (0,3; 1,0) (`knobs.py:508`); `operation` "abs"; `shared_random` True; geoms: os 14 de pé | sim | `env_cfg.py:185-187` |
| `massa_tronco` | startup | `dr.body_mass` | `ranges` = `DR.massa_tronco` (−1,0; 3,0) kg (`knobs.py:509`); `operation` "add"; corpo `torso_link` | sim | `env_cfg.py:188-190` |
| `vies_encoder` | startup | `dr.encoder_bias` | `bias_range` = (−`DR.vies_encoder`; +`DR.vies_encoder`) = (−0,015; 0,015) rad (`knobs.py:510`); robô inteiro | sim | `env_cfg.py:191-192` |
| `tamanho_caixa` | startup | `eventos.tamanho_caixa` | `faixa` = `Cena.caixa_meia_faixa` (0,07; 0,13) m (`knobs.py:78`); `n_variantes` = `Cena.caixa_n_variantes` = 8 (`knobs.py:79`); geom `box_geom` | sim | `env_cfg.py:193-195` |
| `atraso` | startup | `eventos.atraso` | `passos_max` = `DR.atraso_inicio[1]` = 0 (`knobs.py:511`); `decimacao` = `Contrato.decimacao` = 4 (`knobs.py:55`) | sim | `env_cfg.py:197-198` |
| `empurrao` | interval | `eventos.empurrao` | `interval_range_s` = `DR.periodo_empurrao_s` (3,0; 7,0) s (`knobs.py:520`); `velocity_range` x e y = (−`DR.empurrao`; `DR.empurrao`) = (−0,5; 0,5) m/s (`knobs.py:517`) | **não** | `env_cfg.py:201-205` |
| `forca_caixa` | step | `eventos.forca_caixa` | `force_range` = `DR.forca_caixa_N` (−11,5; 11,5) N (`knobs.py:531`); `torque_range` (0,0; 0,0), escrito no `env_cfg.py`; `duration_s` = `DR.forca_caixa_dur_s` (0,1; 0,3) s (`knobs.py:532`); `cooldown_s` = `DR.forca_caixa_periodo_s` (3,0; 8,0) s (`knobs.py:533`); corpo `box` | **não** | `env_cfg.py:207-209` |
| `rfi` | step | `eventos.rfi` | nenhum | **não** | `env_cfg.py:210` |

O bloco que sai no play tem duas partes. Os eventos `pd`, `forca_motor` e `solref` ficam num `**({...} if not play else {})` (`env_cfg.py:169-181`). Os eventos `empurrao`, `forca_caixa` e `rfi` ficam num `if not play:` (`env_cfg.py:200-210`).

⚠ Divergência: o docstring de `make_env_cfg` diz que o `dr` do `k` vale só para quais eventos entram e que as faixas vêm do `Curriculo` (`env_cfg.py:84-87`). O código lê `k.dr` direto para as faixas de `atrito_pe`, `massa_tronco`, `vies_encoder`, `atraso`, `empurrao` e `forca_caixa` (`env_cfg.py:186-209`). Só os eventos `pd`, `forca_motor`, `solref`, `com_caixa` e `rfi` pedem a faixa ao `Curriculo.dr()` (`eventos.py:529`, `:543`, `:556`, `:575`, `:612`). Os eventos `atrito_pad`, `atrito_laje` e `vies_imu` pedem ao `Curriculo.dr()` só a chave que liga o sorteio e leem a faixa de `ATIVO.dr` ou de `ATIVO.cena` (`eventos.py:472-474`, `:486-488`, `:499-502`).

---

## 3. Peças do currículo que os eventos leem

Cada peça abaixo é usada pelo nome nas seções seguintes.

| Peça | O que é | Referência |
|---|---|---|
| `v3_cur` | O objeto `Curriculo` do env. Guarda as células, os níveis, as fatias, os degraus e as travas do currículo. Os eventos só o consultam. | `curriculo.py:46-47`; `balanceador.py:64-112` |
| Fatia | O grupo de tarefas que um env recebe no reset: ANDAR (0), B (1) ou C (2). A fatia B reúne PEGAR e CARREGAR; a fatia C é o BOTAR. | `knobs.py:43`; `balanceador.py:34`; `eventos.py:32` |
| Célula | Uma linha do currículo: um par (tarefa, tipo) com uma nota `s` e um portão. Há 8 células: ANDAR `andar` e `terreno`; CARREGAR `reto`, `giro_lugar`, `arco`, `terreno`; PEGAR `pegar`; BOTAR `botar`. | `balanceador.py:22-24`, `:55-61` |
| Nota `s` da célula | Média móvel exponencial do resultado dos segmentos da célula, com taxa `Balanceador.alfa` = 0,05 (`knobs.py:394`). Atualiza no fecho da iteração. | `balanceador.py:162-168` |
| Portão | O limiar de `s` para a célula contar como aprovada: `Balanceador.portao_andar` = 0,75 na célula ANDAR `andar`; `Balanceador.portao_andar_terreno` = 0,50 na célula ANDAR `terreno`; `Balanceador.portao` = 0,75 nas demais. | `knobs.py:391-393`; `balanceador.py:96-98`, `:122-123` |
| Nível | O índice de dificuldade de uma tarefa. Máximo: ANDAR 7 (`round((1,0 − 0,3) / 0,1)` com `Comando.frente_tetos[-1]`, `Comando.andar_inicio`, `Comando.andar_passo`); CARREGAR 4; PEGAR 4; BOTAR 4. | `balanceador.py:72-74`; `knobs.py:122`, `:129`, `:134`, `:417-418` |
| Subnível | Só PEGAR e BOTAR têm. Há dois (0 e 1). O subnível 0 vale só dentro do nível 0. Para ANDAR e CARREGAR, `subnivel()` devolve o último (1). | `balanceador.py:91-92`, `:325-327`; `knobs.py:422-424` |
| Linha de terreno | O índice de dificuldade do terreno. No CARREGAR é um inteiro por tarefa (`nivel_terreno`). No ANDAR é um número real por coluna de terreno (`Curriculo.linha`), e `nivel_terreno("ANDAR")` é o piso da média das 8 colunas. Teto: ANDAR `Terreno.linhas` − 1 = 12; CARREGAR o maior valor de `Terreno.linha_max_carregar` = 9. | `balanceador.py:77-78`, `:87`, `:329-333`; `knobs.py:429`, `:439-440` |
| Degrau | O índice de refino de uma tarefa (0 a `Balanceador.degraus` = 5). Sobe 1 quando a tarefa fica `Balanceador.iters_refino` = 100 iterações seguidas no "último nível" com a média das células abertas no portão (média das notas `s` ≥ média dos portões). "Último nível" quer dizer: nível = nível máximo e, no ANDAR e no CARREGAR, também a célula `terreno` aberta e a linha de terreno no teto (no ANDAR, `nivel_terreno` = 12: as 8 linhas em 12). A contagem zera quando uma iteração falha a condição. O degrau para de subir em `Balanceador.degraus` = 5. Nunca desce. | `knobs.py:400-401`; `balanceador.py:128`, `:234-248` |
| Degrau global `g` | O menor degrau entre as quatro tarefas. | `balanceador.py:341-343` |
| Restrição | O menor valor de `nível / nível máximo` entre as quatro tarefas. | `balanceador.py:345-348` |
| Trava | Um booleano do `dr()` que só liga e nunca desliga. Três travas: `portao0`, `vies`, `subida` (Seção 11.1). | `balanceador.py:101-103`, `:197-202` |
| `dr()` | O dicionário que o `Curriculo` devolve com o estado atual da randomização. Os eventos de reset e de passo o leem a cada chamada. | `balanceador.py:358-389` |
| Iteração | Uma iteração são `Balanceador.passos_por_iteracao` = 24 passos de env (`knobs.py:402`; `tarefa.py:49`). O termo `v3` fecha a iteração do balanceador quando `common_step_counter // 24` muda. | `curriculo.py:119-122`; `balanceador.py:153-195` |
| Fecho da iteração | Atualiza as notas, as zonas, o refino, as aberturas de célula e de fatia, a rampa da fatia, os níveis e as travas, nessa ordem. | `balanceador.py:153-195` |

O termo `v3` roda no começo de cada `_reset_idx`, portanto antes dos eventos de reset. Todo sorteio de reset lê o `dr()` do fecho mais recente.

---

## 4. Sorteio da fatia e da célula de terreno no reset

### 4.1 Evento `fatia` (`eventos.sorteia_fatia`, `eventos.py:133-175`)

Quando: no reset de cada env, e é o primeiro evento de reset (`env_cfg.py:154`).

Passo a passo:

| Ordem | O que escreve | Distribuição | Referência |
|---|---|---|---|
| 0 | `cur.vis_ter` e `cur.trav_ter` | Visita e travessia do episódio de ANDAR em terreno que acabou, somadas por coluna, antes de a fatia e o terreno novos sobrescreverem os do env (Seção 4.6). | `eventos.py:143-160` |
| 1 | `v3_fatia[env_ids]` | Escolha com reposição, pesos `Curriculo.fracoes_fatias()` (`torch.multinomial` no `v3_gerador`). | `eventos.py:161-162`; `balanceador.py:302-304` |
| 2 | `v3_t_empurrao[env_ids]` = −∞ | Constante. | `eventos.py:163` |
| 3 | `terreno_ep` (episódio de terreno, sim ou não) | Bernoulli: `u < p`, com `u` uniforme em [0; 1) e `p` descrito na Seção 4.3. | `eventos.py:165-167` |
| 4 | `terrain_levels`, `terrain_types`, `scene.env_origins` | Coluna e linha da Seção 4.4. | `eventos.py:168-175` |

O evento não escreve `v3_celula` (`eventos.py:136`). O escritor de `v3_celula` é o comando (`comando.py:897-925`, Seção 4.5).

O passo 2 zera a marca do último empurrão. O comando usa essa marca como âncora de tempo (`comando.py:221`, `:834`).

### 4.2 Pesos das fatias (`Curriculo.fracoes_fatias`, `balanceador.py:282-293`)

- Uma fatia está aberta se alguma célula dela está aberta (`balanceador.py:287`).
- Fatia fechada recebe 0.
- Cada fatia aberta tem um teto de fração: 1,0 para o ANDAR; para B e C, `Balanceador.rampa_fatia_inicio` + `Balanceador.rampa_fatia_passo` × `rampa[f]` = 0,05 + 0,02 × `rampa[f]` (`knobs.py:397-398`; `balanceador.py:288`).
- `rampa[f]` sobe 1 a cada `Balanceador.rampa_fatia_its` = 12 fechos de iteração depois da abertura da fatia (`knobs.py:399`; `balanceador.py:186-193`). Nunca desce.
- O algoritmo ordena as fatias abertas pelo teto, do menor ao maior. Cada uma recebe `min(teto, resto / fatias_que_faltam)`. O `resto` começa em 1,0 (`balanceador.py:289-293`). O que o teto de uma fatia corta volta em partes iguais às fatias de teto maior.

Quando cada fatia abre:

| Fatia | Condição de abertura | Referência |
|---|---|---|
| ANDAR | Aberta desde o começo (célula `andar`). | `balanceador.py:27`, `:110` |
| B | `nivel("ANDAR")` ≥ `Balanceador.nivel_abre_b` = 5. A abertura abre juntas as células CARREGAR `reto` e PEGAR `pegar`. | `knobs.py:396`; `balanceador.py:209-211` |
| C | A nota `s` da célula PEGAR `pegar` ≥ `Balanceador.portao` = 0,75. A abertura abre a célula BOTAR `botar`. | `knobs.py:391`; `balanceador.py:231-232` |

### 4.3 Probabilidade `p` de episódio de terreno

`p` depende da fatia sorteada (`eventos.py:165-166`):

- Fatia ANDAR: `p` = fração da célula `terreno` do ANDAR no sorteio de células do ANDAR. Enquanto só a fatia ANDAR está aberta (a B abre no nível 5), `p` é também a fração de episódios de terreno do treino.
- Fatia B ou C: `p` = fração da célula `terreno` do CARREGAR no sorteio de células do CARREGAR. Na fatia C, o `p` é calculado e depois ignorado, porque `escolhe_celula_terreno` zera a fatia C (Seção 4.4).

A fração vem de `Curriculo.fracoes_celulas(tarefa)` (`eventos.py:36-40`; `balanceador.py:295-299`):

- Célula `terreno` fechada: fração 0 (`eventos.py:37`, `:40`).
- Célula aberta: `F_i = piso + (1 − n · piso) · (D_i + eps) / Σ(D_j + eps)`, com `D_i = max(portao_i − s_i, 0)` (na célula `terreno` do ANDAR, `D_i = 1 − média(linha[1:]) / 12`, a distância das 8 linhas ao topo; a nota `s` dela não entra, `balanceador.py:136-144`), `n` = células abertas da tarefa, `piso` = `Balanceador.piso_sub` = 0,10 e `eps` = `Balanceador.eps` = 0,02 (`knobs.py:389-390`; `balanceador.py:37-45`). O arquivo `02-curriculo-e-balanceador.md`, seção 7.2, traz a tabela de `p` por média das linhas e nota de `andar`: 0,885 com a célula recém-aberta (linhas em 0) e a nota de `andar` no portão, 0,500 com as 8 linhas em 12 (derivado: a fórmula).

Quando uma célula de velocidade abre (`balanceador.py:212-217`): no CARREGAR, as células abrem em ordem (`reto` com a fatia B, depois `giro_lugar`, depois `arco`). A próxima célula fechada abre quando a média das notas `s` das células já abertas da tarefa chega à média dos portões delas. Abre uma célula por fecho e por tarefa. A célula nova entra com `s` = 0 (`balanceador.py:29-30`, `:125-128`, `:205-206`). O ANDAR só tem a célula de velocidade `andar`.

Quando a célula `terreno` abre (`balanceador.py:218-228`), para ANDAR e CARREGAR:

- O nível da tarefa: no CARREGAR, o máximo (`_nivel[t] == max_nivel[t]` = 4). No ANDAR, `_nivel["ANDAR"]` ≥ `Balanceador.nivel_abre_terreno` = 2 (`knobs.py:412`, `balanceador.py:222`). O ANDAR abre a célula com 2 de 7 níveis: o robô aprende a andar antes da escada (comentário do código).
- A média das células abertas da tarefa passa a média dos portões (`_passa_media`, `balanceador.py:125-128`). No ANDAR só a célula `andar` está aberta: a condição vira `s` de `andar` ≥ 0,75. No `CARREGAR`, as células `reto`, `giro_lugar` e `arco` entram na média.
- Só no CARREGAR: `nivel_terreno("ANDAR")` ≥ 1, isto é, o piso da média das 8 linhas do ANDAR ≥ 1 (`balanceador.py:225`). A nota da célula ANDAR `terreno` não entra mais.
- O menor fecho de abertura da célula `terreno` do ANDAR é o 101 (derivado: 2 subidas de nível de 50 fechos seguidos cada, e a abertura no fecho seguinte; Seção 5.6 do arquivo 02).

A linha de terreno do CARREGAR sobe 1 quando o nível é máximo, a linha está abaixo do teto, a célula `terreno` está aberta e passa no portão, e isso se mantém por `Balanceador.iters_nivel` = 50 fechos seguidos (`knobs.py:395`; `balanceador.py:261-271`). A linha de cada coluna do ANDAR segue outra regra: `linha[c] += linha_ganho × (taxa − zona_t_alvo)`, com a taxa de travessia da coluna (`balanceador.py:177-182`; Seção 5.4b do arquivo 02).

### 4.4 Coluna e linha (`eventos.escolhe_celula_terreno`, `eventos.py:43-74`)

Coluna é o tipo de terreno (índice em `Terreno.colunas`, `knobs.py:434-435`). Linha é a dificuldade (0 a 12). A função sorteia quatro uniformes por episódio no `v3_gerador`, em float64 (`eventos.py:60`).

| Fatia | Coluna | Linha |
|---|---|---|
| ANDAR de terreno (`terreno_ep` = sim) | De 1 a 8, pela inversa da soma acumulada de `fracoes(atraso, Terreno.piso_coluna, Balanceador.eps)` sobre `u₀`, com `atraso[c]` = 1 − `linha[c]` / 12, `piso_coluna` = 0,05 e `eps` = 0,02 (`eventos.py:63-65`; `knobs.py:443`). Inclui `irregular`. Com as linhas iguais, 1/8 cada. | `⌊linha[coluna] + u₁⌋`, com teto em 12 (`eventos.py:66`). Arredondamento estocástico: a média das linhas sorteadas da coluna é a `linha[coluna]` |
| B de terreno | Uniforme entre as 7 colunas de `Terreno.linha_max_carregar` (1/7 cada): `escada`, `escada_inv`, `rampa`, `rampa_inv`, `escada_aleatoria`, `obstaculos`, `caixas`. A coluna `irregular` nunca sai (`eventos.py:56-57`, `:67-68`). | Uniforme inteira em 0..`min(nivel_terreno("CARREGAR"), teto da coluna)`: `⌊u₃ · (min(nivel, teto) + 1)⌋` (`eventos.py:69`) |
| C | Coluna 0 (`plano`). | Linha 0. |
| ANDAR ou B sem terreno (`terreno_ep` = não) | Coluna 0. | Linha 0. |

Pesos das colunas do ANDAR (derivado: a fórmula da tabela; a coluna "freq." vem de `escolhe_celula_terreno` com `torch.Generator().manual_seed(0)` e 200 000 sorteios, todos de ANDAR com `terreno_ep` verdadeiro). Linhas do exemplo: `escada` 3, `escada_inv` 2, `rampa` 6, `rampa_inv` 5, `escada_aleatoria` 1, `irregular` 8, `obstaculos` 6, `caixas` 10.

| Coluna | Linha | Peso | Freq. (200 000) |
|---|---|---|---|
| `escada` | 3 | 0,147 | 0,147 |
| `escada_inv` | 2 | 0,158 | 0,158 |
| `rampa` | 6 | 0,116 | 0,116 |
| `rampa_inv` | 5 | 0,126 | 0,126 |
| `escada_aleatoria` | 1 | 0,168 | 0,169 |
| `irregular` | 8 | 0,095 | 0,095 |
| `obstaculos` | 6 | 0,116 | 0,116 |
| `caixas` | 10 | 0,074 | 0,074 |

- O mínimo por coluna é `piso_coluna` = 0,05 mais um resto. Com sete colunas na linha 12 e `caixas` na linha 0, cada coluna vencida pesa 0,060 (`0,05 + 0,60 × 0,02 / 1,16`) e `caixas` pesa 0,578 (simulação: 0,060 e 0,578). O peso nunca chega a 0,05.
- Com as 8 linhas iguais, os pesos valem 1/8, qualquer que seja a linha: o atraso é igual em todas as colunas.
- As três colunas de escada (`escada`, `escada_inv`, `escada_aleatoria`) somam 0,474 no exemplo e 0,375 com as linhas iguais.
- A linha 2,4 de uma coluna sai 2 com chance 0,6 e 3 com chance 0,4 (derivado: conta). Em simulação, a linha 0,4 em todas as colunas dá a média 0,398 por coluna.

Referências: máscara `ter = terreno_ep & (andar | (fatia == B))` (`eventos.py:70-71`); saída `torch.where(ter, coluna, 0)` e `torch.where(ter, linha, 0)` (`eventos.py:74`).

`Terreno.linha_max_carregar` (`knobs.py:439-440`): `escada` 9, `escada_inv` 9, `rampa` 9, `rampa_inv` 9, `escada_aleatoria` 6, `obstaculos` 8, `caixas` 7.

O `nivel_terreno("CARREGAR")` vale para todos os envs (`eventos.py:168`). A fatia ANDAR não o usa: lê `Curriculo.linha` (`eventos.py:169`).

Escrita no terreno (`eventos.py:173-175`):

- `terrain_levels[env_ids]` = linha e `terrain_types[env_ids]` = coluna.
- `scene.env_origins[env_ids]` = `terrain_origins[linha, coluna]`.
- `scene.env_origins` é o tensor `env_origins` da entidade de terreno (`mjlab/scene/scene.py:103-109`). Todo evento e todo termo que soma `env_origins` usa a origem do sub-terreno sorteado.

No começo da vida do env (construção), a entidade de terreno já atribui a cada env uma linha e uma coluna (linha uniforme sobre as 13 linhas, coluna proporcional ao número de colunas) (`mjlab/terrains/terrain_entity.py:325-365`). O primeiro `sorteia_fatia` sobrescreve essa atribuição. Ele não conta visita dela, porque no primeiro reset `episode_length_buf` vale 0 (Seção 4.6).

### 4.5 Como o comando usa o resultado

- O comando lê `terrain_types[ids] != 0` como "episódio de terreno" e põe a célula `terreno` da tarefa em `v3_celula` (`comando.py:910-915`).
- Fora do episódio de terreno, o comando sorteia a célula entre as abertas, sem a `terreno` (`sem_tipo="terreno"`, `comando.py:917-918`; `balanceador.py:306-316`).
- O comando só sorteia velocidade para ANDAR na fatia ANDAR e CARREGAR na fatia B (`comando.py:905`).

---

### 4.6 Visita e travessia do episódio de terreno do ANDAR

O `sorteia_fatia` registra o episódio de ANDAR em terreno que acabou. Os dois vetores (`vis_ter`, `trav_ter`) movem a linha de cada coluna no fecho (Seção 5.4b do arquivo 02). Os números "(derivado)" vêm de uma chamada de `sorteia_fatia` com um env simulado, e as fórmulas vêm do código.

**O que conta** (`eventos.py:143-160`). O registro roda antes de a fatia e o terreno novos sobrescreverem os do env. Para cada env que reseta:
- `d` = distância xy entre `twist.pose_ult` e `scene.env_origins`, a origem da célula do episódio que acabou (`eventos.py:153-154`).
- `pedido` = `soma_cmd_marcha` × `step_dt`. A soma acumula `‖cmd_xy‖` nos passos com comando linear ativo (`comando.py:847`; `eventos.py:155`). É a distância que o comando pediu no episódio.
- Visita: três condições juntas. A fatia anterior é ANDAR e a coluna anterior é ≠ 0 (não é o plano); o episódio teve passos (`episode_length_buf` > 0); e `pedido` ≥ `Terreno.dist_travessia` = 3,5 m, ou `terminated`, ou `d` > 3,5 m (`eventos.py:156-157`; `knobs.py:442`).
- Travessia: visita com `d` > 3,5 m (`eventos.py:158`).
- `terminated` não inclui o `time_out`. Um episódio que acaba por tempo, sem passar de 3,5 m e sem pedir 3,5 m, não é visita.
- `torch.bincount` soma as visitas e as travessias por coluna em `vis_ter` e `trav_ter` (`eventos.py:158-160`).

Casos de reset (derivado: o `sorteia_fatia` do snapshot com 9 envs simulados, coluna `escada`, `step_dt` = 0,02 s, `dist_travessia` = 3,5 m):

| Env | `d` (m) | `pedido` (m) | Caiu | `episode_length_buf` | Visita | Travessia |
|---|---|---|---|---|---|---|
| 0 | 3,7 | 1,0 | não | 400 | sim | sim |
| 1 | 3,7 | 4,0 | não | 600 | sim | sim |
| 2 | 1,0 | 1,0 | sim | 50 | sim | não |
| 3 | 2,0 | 3,7 | não | 900 | sim | não |
| 4 | 1,0 | 2,0 | não | 1000 | não | não |
| 5 (coluna `plano`) | 4,0 | 4,0 | não | 600 | não | não |
| 6 | 1,0 | 1,0 | não | 1000 | não | não |
| 7 | 3,4 | 3,4 | não | 1000 | não | não |
| 8 (primeiro reset) | 3,7 | 4,0 | não | 0 | não | não |

O env 0 mostra o caso do deslocamento de partida: o pedido fica abaixo de 3,5 m, mas a base passa de 3,5 m. O env 3 mostra o episódio que pediu 3,5 m e não chegou: a visita vale, e a travessia não. O env 8 tem os valores do env 1, mas `episode_length_buf` = 0: a guarda de passos o exclui. A simulação dá `vis_ter`[`escada`] = 4 e `trav_ter`[`escada`] = 2.

**A ordem de leitura da posição.** O término e o registro leem o mesmo buffer, o `twist.pose_ult`. Ordem dentro de um passo do env (`mjlab/envs/manager_based_rl_env.py:430-456`):
1. Física, depois `episode_length_buf` += 1.
2. Terminações: `fim_da_cadeia` lê `pose_ult` e `env_origins`. O `pose_ult` ainda é o gravado no `compute` do passo anterior, a pose do fim do passo N − 1 (`terminacoes.py:69-72`; `mjlab/envs/manager_based_rl_env.py:436`). A guarda `episode_length_buf` > 2 vale só para a travessia (`terminacoes.py:71`).
3. Reset dos envs terminados: o evento `fatia` lê o mesmo `pose_ult`, a mesma origem, `soma_cmd_marcha`, `terminated` e `episode_length_buf`, que vale ≥ 1 depois do passo 1. O reset do `twist` zera `soma_cmd_marcha` depois (`comando.py:890`; `mjlab/envs/manager_based_rl_env.py:447`).
4. `command_manager.compute`: `_update_metrics` grava `pose_ult` com a pose de agora. Nos envs que acabaram de resetar, é a pose nova, depois do evento `base` (`comando.py:813-815`; `mjlab/envs/manager_based_rl_env.py:456`).

Consequência: a travessia que acaba o episódio no passo N conta também no registro do mesmo reset. O término sai um passo depois de a base passar de 3,5 m, porque a leitura de `pose_ult` tem um passo de atraso (comentário de `fim_da_cadeia`, `terminacoes.py:61-67`).

**O primeiro reset.** O primeiro reset é o `reset` do env, que chama `_reset_idx` com todos os envs (`mjlab/envs/manager_based_rl_env.py:356-369`). Antes dele, a entidade de terreno já deu a cada env uma coluna e uma origem na construção (`mjlab/terrains/terrain_entity.py:203-205`, `:325-365`), o `v3_fatia` nasce 0, que é ANDAR (`curriculo.py:24`), e o `pose_ult` nasce 0 (`comando.py:706`, `comando.py:732`). Sem outra guarda, o `d` seria a distância da origem da célula ao ponto (0; 0) do mundo, maior que 3,5 m para quase toda célula.

A guarda `episode_length_buf` > 0 exclui esse reset (`eventos.py:148-150`, `:156`). O `episode_length_buf` nasce 0 na construção, e o `_reset_idx` só o zera no fim, depois dos eventos de reset (`mjlab/envs/manager_based_rl_env.py:232-234`, `:560-562`, `:590`). No primeiro reset ele vale 0, e nenhum env conta visita nem travessia. Num reset do passo, o `episode_length_buf` já recebeu += 1 antes do reset (passo 1 acima), e a condição `episode_length_buf` > 0 é verdadeira em todo env.

Simulação (derivado): o `sorteia_fatia` do snapshot com 4096 envs de ANDAR, colunas 1 a 8, origens a 20 m do mundo, `pose_ult` = 0 e `episode_length_buf` = 0. Resultado: `vis_ter` e `trav_ter` ficam em 0 nas 9 colunas, e o primeiro fecho deixa as 8 linhas em 0. As linhas só sobem com travessias de episódios reais, e o piso `nivel_terreno("ANDAR")` ≥ 1, que a abertura da célula `terreno` do CARREGAR exige, depende delas (Seção 5.6 do arquivo 02).

## 5. A cena no reset

### 5.1 Evento `cena` (`eventos.posiciona_cena`, `eventos.py:332-367`)

Quando: no reset de cada env, depois de `fatia` e antes de `base` (`env_cfg.py:154-157`). A ordem é obrigatória: `base` lê o `v3_caixa_xy` que `cena` escreve (`eventos.py:364`, `:201`).

Entradas lidas do currículo (`eventos.py:344-346`):

- `nivel("PEGAR")` — o nível corrente do PEGAR.
- `nivel_passou("PEGAR")` — o maior nível do PEGAR cuja célula já passou no portão (−1 = nenhum). Com subnível abaixo do último, o nível 0 ainda não passou (`balanceador.py:350-356`).
- `Niveis.subnivel_extra[subnivel("PEGAR")]` — o deslocamento extra de topo: (0,10; 0,0) m (`knobs.py:422`).

#### 5.1.1 Topo da laje e massa da caixa (`sorteia_topo_e_massa`, `eventos.py:82-109`)

`Niveis.pegar` = ((0,55; 1), (0,45; 2), (0,30; 3), (0,15; 4), (0,04; 5)): par (topo `h_k` em m; massa `m_k` em kg) por nível 0 a 4 (`knobs.py:417`). `h_0` = 0,55 m (`knobs.py:417`). `m_min` = `Cena.caixa_massa_base` = 1,0 kg (`knobs.py:80`). `Niveis.topo_min_c` = 0,30 m (`knobs.py:419`).

Dois uniformes por episódio, `u₀` para o topo e `u₁` para a massa (`eventos.py:104`).

| Fatia | Topo (m, sobre o chão local) | Massa (kg) |
|---|---|---|
| ANDAR | 0 (`eventos.py:106`) | `m_min` (teto da massa = `m_min`, `eventos.py:107-108`) |
| B | `lo + (h_0 + extra − lo) · u₀`, com `lo` = `h_k + extra` | `m_min + (m_k − m_min) · u₁` |
| C | o mesmo, com `lo` = `max(h_k, topo_min_c) + extra` | `m_min + (m_j − m_min) · u₁`, com `m_j` = `Niveis.pegar[nivel_passou][1]` se `nivel_passou ≥ 0`, senão `m_min` |

Referências: `eventos.py:102`, `:105-108`. A fatia C mede a massa pelo maior nível do PEGAR que já passou no portão, não pelo nível corrente (`eventos.py:102`).

A saída inclui `topo_w` = `env_origins[:, 2]` + topo, a altura absoluta (`eventos.py:109`). O evento grava `v3_topo` (altura sobre o chão local) e `v3_massa` (`eventos.py:347-348`).

Com o PEGAR no subnível 0 e no nível 0, `lo` e o teto coincidem: topo fixo = 0,55 + 0,10 = 0,65 m (calculado de `eventos.py:105-106` e `knobs.py:417`, `:422`).

#### 5.1.2 ANDAR: laje e caixa estacionadas (`estaciona_cena`, `eventos.py:310-319`)

Na fatia ANDAR, a laje e a caixa vão para `Cena.estaciona_z` = 20,0 m acima da origem do env (`knobs.py:114`; `eventos.py:317`). A laje fica em (`origem_xy` + `Cena.prateleira_xy`) = (0,50; 0,00) m e a caixa em (`origem_xy` + `Cena.caixa_xy`) = (0,32; 0,00) m (`knobs.py:94`, `:96`; `eventos.py:318-319`; `_laje`, `eventos.py:276-294`).

O evento não escreve `v3_caixa_xy` nem `v3_rumo_cena` para os envs da fatia ANDAR (`eventos.py:349-353`). Esses dois buffers guardam o valor do episódio anterior do mesmo env.

#### 5.1.3 B e C: rumo e deslocamento sobre o terreno (`cena_na_plataforma`, `eventos.py:112-130`)

O evento chama a função com `terreno_b` = (fatia == B) e `terrain_types` > 0 (`eventos.py:355-356`).

- Se nenhum env tem `terreno_b`, devolve rumo 0 e deslocamento 0, sem tirar nada do gerador (`eventos.py:123-124`).
- Senão, sorteia um uniforme por env (de todos os envs B e C do lote), `k = ⌊u · 4⌋` limitado a 3, e o rumo é `k · π/2` rad (0, 90, 180 ou 270 graus) nos envs `terreno_b`; 0 nos demais (`eventos.py:125-126`).
- O deslocamento é `R(rumo) · (d, 0)`, com `d` = `Terreno.laje_do_centro` − `Cena.prateleira_xy[0]` = 0,75 − 0,50 = 0,25 m, só nos envs `terreno_b` (`knobs.py:444`, `:94`; `eventos.py:127-129`). `R(rumo)` gira o vetor pelo ângulo `rumo` (`alvos.py:157-160`).
- Efeito: o centro da laje fica a 0,75 m do centro da plataforma, na direção do rumo.

#### 5.1.4 B e C: posição da caixa e da laje (`eventos.py:354-367`)

Dois uniformes por env, `u₀` e `u₁` (`eventos.py:357`):

| Grandeza | Fórmula | Faixa | Referência |
|---|---|---|---|
| `dx` | `Cena.jitter_x_max[nivel("PEGAR")] · u₀` | (0; `jitter_x_max[nível]`) m, um lado só. `Cena.jitter_x_max` = (0,20; 0,20; 0,20; 0,15; 0,08) m | `eventos.py:358`; `knobs.py:99` |
| `dy` | `caixa_jitter_y[0] + (caixa_jitter_y[1] − caixa_jitter_y[0]) · u₁` | (−0,18; 0,18) m (`Cena.caixa_jitter_y`) | `eventos.py:359`; `knobs.py:97` |
| `caixa_rel` | `desloc + R(rumo) · (0,32 + dx, 0 + dy)` | Posição da caixa, relativa à origem do env | `eventos.py:361`; `knobs.py:96` |
| `laje_rel` | `desloc + R(rumo) · (0,50; 0,00)` | A laje não leva o jitter | `eventos.py:362`; `knobs.py:94` |

Escritas (`eventos.py:363-367`):

- `v3_caixa_xy[ids]` = `caixa_rel`; `v3_rumo_cena[ids]` = `rumo`.
- Laje: `_laje(env, ids, topo_w, xy=origem + laje_rel, yaw=rumo)` — mocap, centro em z = `topo_w` − `Cena.prateleira_meia_z` (0,02 m, `knobs.py:90`), rotação só em z (`eventos.py:276-294`).
- Caixa: `_caixa(env, ids, topo_w, origem + caixa_rel)` — centro em z = `topo_w` + `v3_meia[ids]`; quaternion identidade; velocidade zero (`eventos.py:297-307`).

A caixa não gira com o rumo. Como o rumo é múltiplo de 90° e a caixa é um cubo, a rotação não mudaria a forma (consequência geométrica, não escrita no código).

`v3_meia` é a meia aresta da caixa de cada env, sorteada no startup (Seção 11.5).

### 5.2 Evento `base` (`eventos.reset_base_por_fatia`, `eventos.py:211-225`)

Quando: no reset, depois de `cena`. Separa os envs pela fatia e escreve a pose da base e a velocidade da base.

**Fatia ANDAR** (`eventos.py:221-224`): chama `reset_root_state_uniform` do mjlab com `pose_range` = `Cena.reset_base_andar` e `velocity_range` = {}.

- `x` ∈ (−0,5; 0,5) m, `y` ∈ (−0,5; 0,5) m, `yaw` ∈ (−π; π) rad, cada um uniforme e independente (`knobs.py:103`; `mjlab/envs/mdp/events.py:27-41`, `:143`).
- `z`, `roll` e `pitch`: 0 (chaves ausentes valem (0; 0), `mjlab/envs/mdp/events.py:38-39`).
- Posição final = estado padrão do robô + sorteio + `env_origins` (`mjlab/envs/mdp/events.py:174-175`). O z padrão é `pelve_de_pe` + 0,01 m (`cena.py:223`).
- Orientação = quaternion padrão × quaternion do yaw sorteado (`mjlab/envs/mdp/events.py:177-180`).
- Velocidade = velocidade padrão da raiz (`mjlab/envs/mdp/events.py:183-190`).

**Fatias B e C** (`_base_da_pega`, `eventos.py:178-208`). A função escreve a pose da base atrás da caixa. Entradas e sorteios:

| Grandeza | Fórmula | Faixa | Referência |
|---|---|---|---|
| `recuo` | `Referencias.recuo(v3_topo, v3_meia)`: interpolação bilinear numa grade (topo, meia aresta) lida de `referencias.npz` | Os valores da grade não estão no código lido | `eventos.py:192`; `alvos.py:30-37`, `:58`, `:75-76` |
| `jitter` | `Niveis.subnivel_jitter[subnivel("PEGAR")]` | (0,0; 1,0): subnível 0 sem jitter | `eventos.py:193`; `knobs.py:424` |
| `d` | `recuo + (2 · u₀ − 1) · Alvos.faixa_recuo · jitter` | `recuo` ± 0,08 m × jitter (`Alvos.faixa_recuo` = 0,08, `knobs.py:227`) | `eventos.py:194` |
| `yaw` | `v3_rumo_cena + lo_yaw + (hi_yaw − lo_yaw) · u₁` | rumo + (−0,2; 0,2) rad (`Cena.reset_base_manipula["yaw"]`, `knobs.py:101`) | `eventos.py:195-198` |
| `dz` | `lo_z + (hi_z − lo_z) · u₂` | (0,01; 0,05) m (`Cena.reset_base_manipula["z"]`, `knobs.py:101`) | `eventos.py:196`, `:199` |

Três uniformes por env no `v3_gerador` (`eventos.py:191`). A posição xy é `caixa_mundo − R(rumo) · (d, 0)`, com `caixa_mundo` = `env_origins_xy` + `v3_caixa_xy` (`eventos.py:201-202`). O z é `env_origins_z` + z padrão + `dz` (`eventos.py:200`, `:204`). A orientação é o quaternion padrão × yaw (`eventos.py:205`). A velocidade é a velocidade padrão da raiz (`eventos.py:208`).

A função reimplementa só a escrita de `reset_root_state_uniform`, porque a distância depende do topo e da caixa de cada episódio (`eventos.py:181-185` descreve a intenção; o código está em `eventos.py:203-208`).

`faixa_manipula` traz só as chaves `z` e `yaw` (`knobs.py:101`). O código não lê `x` nem `y` dela (`eventos.py:195-199`).

### 5.3 Evento `juntas` (`reset_joints_by_offset`, `mjlab/envs/mdp/events.py:279-313`)

Quando: no reset de todos os envs e fatias. Com `position_range` (0,0; 0,0) e `velocity_range` (0,0; 0,0) o evento:

- põe cada junta na posição padrão (`init_state.joint_pos` = `q_de_pe`, `cena.py:225`) e limita ao limite mole de posição (`mjlab/envs/mdp/events.py:296-299`);
- põe a velocidade de cada junta em 0 (a velocidade padrão é 0, `cena.py:226`; `mjlab/envs/mdp/events.py:301-302`).

### 5.4 Evento `carga` (`eventos.carga`, `eventos.py:443-461`)

Quando: no reset, depois de `cena`. Não sorteia nada. Escreve no modelo de cada env:

- `body_mass` da caixa = `v3_massa` (kg) (`eventos.py:455`, `:459`);
- `body_inertia` da caixa = `(2/3) · m · a²` nos três eixos, com `a` = `v3_meia` (`eventos.py:456`, `:460-461`).

### 5.5 Evento `filtros` (`eventos.reinicia_filtros`, `eventos.py:228-241`)

Quando: no reset, depois de `base`. Reinicia o estado de estimação do episódio:

- `v3_filtro.reinicia(env_ids)`: zera o mapa de elevação, a confiança, o mapa visto e a referência de altura (`percepcao_lidar.py:132-139`).
- `v3_odo.reinicia(env_ids, pos, quat)`: a odometria estimada começa na pose verdadeira, lida do `qpos` da junta livre da base (`eventos.py:239-240`; `estimador.py:148-156`).
- `v3_estimador.reinicia(env_ids)`: sorteia por episódio quatro variáveis unitárias no `v3_gerador`: `_z_meia` uniforme em (−1; 1) (um por env), `_z_pos` uniforme em (−1; 1) por eixo (3 por env), um eixo de orientação normal normalizado (3 por env) e `_z_ang` normal (um por env) (`estimador.py:250-260`, `:275-279`). As grandezas físicas (metros, graus) vêm da escala aplicada em `estimador.py:passo`, fora do escopo deste documento.

### 5.6 O que a cadeia faz com a laje durante o episódio

Dois eventos de cena não são termos do cfg. O comando os chama:

- `laje_fora` (`eventos.py:322-329`): tira a laje do caminho. Move só a laje para `estaciona_z` = 20,0 m acima da origem, em (`origem_xy` + `prateleira_xy`). O comando chama no `_abre_trecho` quando o trecho é CARREGAR, a fatia é B e o comando de velocidade tem norma L1 maior que 0 (`comando.py:592-594`).
- `posiciona_destino` (Seção 6), chamado em `_abre_botar` (`comando.py:542-548`).

---

## 6. A laje de destino do BOTAR (`posiciona_destino`, `eventos.py:393-413`)

Quando: na abertura do trecho BOTAR, não no reset (`comando.py:542-548`).

Sorteio (`sorteia_destino`, `eventos.py:373-390`): quatro uniformes por env no `v3_gerador` (`eventos.py:381`).

| Grandeza | Fórmula | Faixa | Referência |
|---|---|---|---|
| `theta` | `(2 · u₀ − 1) · radianos(destino_giro_max_graus)` | ± 10° (`Cena.destino_giro_max_graus` = 10,0, `knobs.py:87`) | `eventos.py:382-383` |
| `y` | `(2 · u₁ − 1) · destino_jitter_y` | ± 0,10 m (`Cena.destino_jitter_y`, `knobs.py:86`) | `eventos.py:384` |
| `longe` | `u₂ · Cena.jitter_x_max[nivel("BOTAR")]` | (0; `jitter_x_max[nível]`) m. Valores em `knobs.py:99` | `eventos.py:385` |
| `topo` | `botar[nível] + (botar[0] − botar[nível]) · u₃` | (`Niveis.botar[nível]`; 0,55) m. `Niveis.botar` = (0,55; 0,45; 0,30; 0,15; 0,04) m (`knobs.py:418`) | `eventos.py:386` |

Com o subnível do BOTAR abaixo do último (subnível 0): `longe` = `u₂ · Niveis.subnivel_longe[0]` = (0; 0,10) m e `topo` = `Niveis.botar[0] + Niveis.subnivel_extra[0]` = 0,55 + 0,10 = 0,65 m, fixo (`eventos.py:387-389`; `knobs.py:422-423`).

Posicionamento (`eventos.py:401-413`):

1. `recuo` = `Referencias.recuo(topo, v3_meia)` + `longe` (`eventos.py:405`).
2. `laje_destino(recuo, theta, y, Cena.prateleira_meia_xy, Cena.destino_borda)` devolve o alvo e o centro da laje no frame da base. `e(θ)` = `borda · cos θ + meia_xy · sen|θ| − borda`; alvo = (`recuo` + `e(θ)`, `y`); centro = alvo + (`meia_xy` − `borda`) · (cos θ, sen θ). `Cena.prateleira_meia_xy` = 0,30 m e `Cena.destino_borda` = 0,15 m (`alvos.py:182-192`; `knobs.py:89`, `:85`).
3. A laje (mocap) vai para a posição `base + R(rumo_da_base) · centro`, com yaw = `rumo_da_base + theta` e topo em `env_origins_z + topo` (`eventos.py:407-410`).
4. O evento grava `v3_topo_destino` = `topo` e devolve o ponto de soltura no mundo (alvo em xy, `topo_w` em z) (`eventos.py:411-413`).

O comando soma ao ponto de soltura `v3_meia` em z e um viés do alvo do BOTAR (`comando.py:549-557`).

---

## 7. O gerador de terreno (`cena.terreno_cfg`, `cena.py:251-277`)

O terreno é construído uma vez, na criação da entidade (`mjlab/terrains/terrain_entity.py:133-148`). Nenhum evento regenera o terreno. O que muda a cada reset é qual célula cada env usa (Seção 4.4).

`make_env_cfg` troca o terreno do molde pelo da v3 depois de criar o molde, e `terreno_cfg` não recebe `play` (`env_cfg.py:91`, `:96`; `cena.py:251`). Por isso o play usa o mesmo gerador do treino. Os ajustes de terreno do molde para o play (`num_rows` 5, `curriculum` False) não valem (`mjlab/tasks/velocity/config/g1/env_cfgs.py:178-183`).

### 7.1 Estrutura da grade

| Parâmetro | Valor | Referência |
|---|---|---|
| Linhas (`num_rows`) | `Terreno.linhas` = 13 | `knobs.py:429`; `cena.py:275` |
| Colunas | 9: uma por tipo, na ordem de `sub_terrains` (`curriculum` = True ignora `num_cols`) | `cena.py:262-276`; `mjlab/terrains/terrain_generator.py:157-158` |
| Tamanho da célula | `Terreno.tamanho` = (8,0; 8,0) m | `knobs.py:430`; `cena.py:275` |
| Borda da grade | `Terreno.borda` = 20,0 m de largura em volta de toda a grade. A altura da borda é `TerrainGeneratorCfg.border_height` = 1,0 m (padrão do mjlab, que `cena.py` não passa), com centro em z = −0,5 m. | `knobs.py:446`; `cena.py:276`; `mjlab/terrains/terrain_generator.py:104-107`, `:334-354` |
| Proporção de cada coluna | Não usada para sortear a coluna: o env escolhe a coluna pelo evento `fatia` (Seção 4.4). No modo `curriculum`, a proporção só distribui os envs entre as colunas na criação (cada tipo tem `proportion` = 1,0, o padrão). | `mjlab/terrains/terrain_generator.py:61-66`; `mjlab/terrains/terrain_entity.py:146-148`, `:350-355` |
| Dificuldade da linha `L` | `t = L / 12` (`t = L / max(num_rows − 1, 1)`), `difficulty_range` = (0,0; 1,0) | `mjlab/terrains/terrain_generator.py:122`, `:253-257` |
| Posição da célula | A grade fica centrada na origem do mundo. A linha avança em x, a coluna em y. | `mjlab/terrains/terrain_generator.py:269-283` |
| Origem de spawn da célula | A origem de spawn do sub-terreno, em coordenadas do mundo. | `mjlab/terrains/terrain_generator.py:259-267`, `:321` |
| Semente do terreno | `TerrainGeneratorCfg.seed` não é passado: vale `None` e o gerador usa `np.random.randint(0, 10000)`. | `cena.py:275-276`; `mjlab/terrains/terrain_generator.py:165-169` |

Tamanho calculado da grade de tipos: 13 × 8 = 104 m em x e 9 × 8 = 72 m em y, mais a borda de 20 m.

Os tetos de dificuldade vêm de `Terreno.tetos_andar` = (`degrau` 0,20; `rampa` 0,36; `obstaculo` 0,15) (`knobs.py:432`; `cena.py:259`, `:261`). `Terreno.plataforma` = 2,0 m é o `platform_width` passado a cada tipo que tem plataforma (`knobs.py:441`; `cena.py:260`).

### 7.2 Tipos de coluna

A ordem abaixo é a ordem das colunas (0 a 8), igual a `Terreno.colunas` (`knobs.py:434-435`; `cena.py:262-274`).

| Col. | Nome | Preset e parâmetros passados | Parâmetros do preset do mjlab (não sobrescritos) | O que cresce com `t` | Plataforma e origem do spawn |
|---|---|---|---|---|---|
| 0 | `plano` | `flat()` | — | nada | Plano 8 × 8 m em z = 0. Origem (4; 4; 0) na célula (`primitive_terrains.py:41-51`). |
| 1 | `escada` | `pyramid_stairs(step_height_range=(0; 0,20), platform_width=2,0)` | `step_width` 0,3 m; `border_width` 1,0 m (`config.py:46-54`) | Altura do degrau `h` = 0,20 · `t` m (`primitive_terrains.py:79-81`) | Anéis: `int((8 − 2·1,0 − 2,0) / (2·0,3))` = 6 (calculado de `primitive_terrains.py:84-92`). O centro sobe até (6 + 1) · `h`. Origem do spawn no centro, em z = 7 · `h` (`primitive_terrains.py:213-235`). O topo central mede 8 − 2·1,0 − 2·6·0,3 = 2,4 m de lado (calculado de `primitive_terrains.py:213-217`). |
| 2 | `escada_inv` | `pyramid_stairs_inv(step_height_range=(0; 0,20), platform_width=2,0)` | iguais às da `escada` | Profundidade do degrau `h` = 0,20 · `t` m | Poço: origem do spawn no fundo, em z = −7 · `h` (`primitive_terrains.py:410-411`). |
| 3 | `rampa` | `hf_pyramid_slope(slope_range=(0; 0,36), platform_width=2,0)` | `border_width` 0,25 m; resolução 0,1 m; escala vertical 0,005 m (`config.py:72-81`; `heightfield_terrains.py:226-228`) | Inclinação `slope` = 0,36 · `t` (`heightfield_terrains.py:243-245`) | Altura da pirâmide: `height_max · xx · yy`, limitada ao valor no canto da plataforma de 2,0 m (`heightfield_terrains.py:253-286`). Com `t` = 1: plataforma central a ≈ 0,74 m (calculado). Origem do spawn no centro, em z = altura física máxima (`heightfield_terrains.py:368-370`). |
| 4 | `rampa_inv` | `hf_pyramid_slope_inv(slope_range=(0; 0,36), platform_width=2,0)` | iguais às da `rampa`, `inverted` True | `slope` = −0,36 · `t` (`heightfield_terrains.py:238-241`) | Origem do spawn no fundo da depressão, em z = −altura física máxima (`heightfield_terrains.py:346-349`, `:365-366`). |
| 5 | `escada_aleatoria` | `random_stairs(step_height_range=(0; 0,20), platform_width=2,0)` | `step_width` 0,8 m; `border_width` 0,25 m (`config.py:198-208`) | A altura de cada degrau é `U(0; 0,20) · (0,5 + 0,5 · t)` m, sorteada por degrau no gerador do terreno (`primitive_terrains.py:1060-1061`) | Degraus: `int((8 − 2·0,25 − 2,0) / (2·0,8))` = 3 (calculado de `primitive_terrains.py:1025-1033`). A altura acumula degrau a degrau. Origem do spawn no centro, em z = altura acumulada (`primitive_terrains.py:1140`, `:1159`). Topo central 8 − 2·0,25 − 2·3·0,8 = 2,7 m de lado (calculado de `primitive_terrains.py:1143-1146`). |
| 6 | `irregular` | `random_rough(scale_with_difficulty=True)` | `noise_range` (0,02; 0,10) m; `noise_step` 0,02 m; `border_width` 0,25 m (`config.py:99-108`) | Amplitude do ruído × `t`: (0,02 · `t`; 0,10 · `t`) m (`heightfield_terrains.py:418-420`) | Sem parâmetro de plataforma. Uma grade de pontos a cada 0,1 m recebe alturas sorteadas com igual probabilidade entre os valores `noise_lo + k · 0,02 m` (k inteiro), em unidades de 0,005 m, de `noise_lo` até abaixo de `noise_hi + 0,02 m` (`np.arange(height_min, height_max + height_step, height_step)`); o último valor pode passar de `noise_hi`. Depois uma spline interpola a grade (`heightfield_terrains.py:455-475`). Origem do spawn em z = 0,06 · `t` m (`heightfield_terrains.py:535-536`; calculado como a média da faixa). |
| 7 | `obstaculos` | `discrete_obstacles(obstacle_height_range=(0; 0,15), platform_width=2,0)` | 40 obstáculos; largura (0,3; 1,0) m; `border_width` 0,25 m (`config.py:123-133`) | Altura do obstáculo `obs_h` = 0,15 · `t` m, truncada em múltiplos de 0,005 m (`int(obs_height / vertical_scale)`; `heightfield_terrains.py:725-727`, `:733`) | Cada obstáculo tem altura sorteada, com igual probabilidade, entre −`obs_h`, −`obs_h` // 2, +`obs_h` // 2 e +`obs_h`: buracos e blocos (`heightfield_terrains.py:752-753`). Largura e comprimento sorteados de modo independente entre {0,2; 0,6; 1,0} m: `int(0,3 / 0,1)` vale 2 em ponto flutuante (0,3 / 0,1 = 2,9999999999999996), e `np.arange(2, 11, 4)` dá os pixels 2, 6 e 10 de 0,1 m (calculado de `heightfield_terrains.py:734-735`, `:747`, `:757-758`). Posição sorteada em passos de 0,4 m (`heightfield_terrains.py:760-765`). Quadrado central de 2,0 m zerado (`heightfield_terrains.py:771-779`). Origem do spawn em z = 0 (`heightfield_terrains.py:840-841`). |
| 8 | `caixas` | `random_spread_boxes(box_height_range=(0; 0,15), box_yaw_range=(0; 0), platform_width=2,0)` | 80 caixas; largura (0,1; 1,0) m; comprimento (0,1; 2,0) m; `border_width` 0,25 m (`config.py:167-179`) | Número de caixas `int(80 · (0,5 + 0,5 · t))`; altura `U(0; 0,15) · (0,2 + 0,8 · t)` m (`primitive_terrains.py:743`, `:793-796`) | Piso plano com topo em z = 0 (`primitive_terrains.py:760-770`). Cada caixa tem largura `U(0,1; 1,0)` m, comprimento `U(0,1; 2,0)` m e posição uniforme na área interna (`primitive_terrains.py:791-804`). Uma caixa cuja posição cai no quadrado central de 2,0 m, alargado de meia largura e meio comprimento da caixa, é descartada, qualquer que seja o giro (`primitive_terrains.py:806-810`). O giro vale sempre 0, porque `box_yaw_range` = (0; 0) (`primitive_terrains.py:815`). Origem do spawn em (4; 4; 0) (`primitive_terrains.py:832`). |

Os números 0,20, 0,36 e 0,15 vêm de `Terreno.tetos_andar` (`knobs.py:432`). Os números 2,0 vêm de `Terreno.plataforma` (`knobs.py:441`).

### 7.3 Teto de cada coluna no CARREGAR

A fatia B usa a linha limitada por `Terreno.linha_max_carregar` (`knobs.py:439-440`). A tabela abaixo mostra, calculados das fórmulas da Seção 7.2, os parâmetros máximos na linha-teto.

| Coluna | Linha-teto | `t` | Parâmetro máximo (calculado) |
|---|---|---|---|
| `escada`, `escada_inv` | 9 | 0,75 | degrau 0,20 · 0,75 = 0,15 m |
| `rampa`, `rampa_inv` | 9 | 0,75 | `slope` 0,36 · 0,75 = 0,27 |
| `escada_aleatoria` | 6 | 0,50 | degrau máximo 0,20 · (0,5 + 0,5 · 0,50) = 0,15 m |
| `obstaculos` | 8 | 0,667 | `obs_h` 0,15 · 8/12 = 0,0999… m, truncado para 19 unidades de 0,005 m = 0,095 m (calculado) |
| `caixas` | 7 | 0,583 | altura máxima 0,15 · (0,2 + 0,8 · 0,583) = 0,10 m |

O ANDAR vai até a linha 12 (`t` = 1,0) em todas as colunas de 1 a 8.

### 7.4 Colunas e plataforma

`cena.colunas(t)` marca "plataforma" nas colunas que aparecem em `Terreno.linha_max_carregar` e guarda a linha-teto (`cena.py:239-245`). `COLUNAS` usa o `ATIVO` (`cena.py:248`). Os sub-terrenos recebem `platform_width` = 2,0 em `escada`, `escada_inv`, `rampa`, `rampa_inv`, `escada_aleatoria`, `obstaculos` e `caixas` (`cena.py:264-273`).

---

## 8. Entidades e sensores da cena

### 8.1 Parâmetros de física do env (`env_cfg.py:97-105`)

| Parâmetro | Valor | Referência |
|---|---|---|
| `scene.extent` | `Cena.extent` = 2,5 | `knobs.py:112` |
| `sim.njmax` | `Cena.njmax` = 800 | `knobs.py:107` |
| `sim.nconmax` | `Cena.nconmax` = 300 | `knobs.py:108` |
| `ccd_iterations` | `Cena.ccd_iterations` = 50 | `knobs.py:109` |
| `impratio` | `Cena.impratio` = 1,0 | `knobs.py:110` |
| `cone` | `Cena.cone` = "pyramidal" | `knobs.py:111` |
| `timestep` | `Contrato.dt_fisica` = 0,005 s | `knobs.py:54` |
| `decimation` | `Contrato.decimacao` = 4 | `knobs.py:55` |
| `episode_length_s` | `Cadeia.episodio_s` = 20,0 s no treino; 10⁹ s no play | `knobs.py:160`; `env_cfg.py:63`, `:105` |

### 8.2 Entidades (`cena.entidades`, `cena.py:293-310`; terreno em `cena.py:251-277`)

| Entidade | O que é | Dados principais | Referência |
|---|---|---|---|
| `robot` | G1 de 29 juntas com pads de palma e de dorso, sites do lidar e da câmera, e PD da Unitree | Ver 8.3 | `cena.py:202-234` |
| `box` | Caixa: corpo livre com um geom cubo `box_geom` | Meia aresta de referência = meio de `Cena.caixa_meia_faixa` = 0,10 m. Massa inicial `Cena.caixa_massa_base` = 1,0 kg. Atrito `Cena.caixa_atrito` = (0,3; 0,02; 0,001). `priority` `Cena.caixa_priority` = 0. `condim` `Cena.caixa_condim` = 3. Grupo de geom `Cena.grupo_mobilia` = 1. Posição inicial (0,32; 0,00; 0,65 + 20,0) m. | `cena.py:69-71`, `:106-115`, `:299-303`; `knobs.py:78`, `:80`, `:83`, `:82`, `:115`, `:88`, `:96`, `:114` |
| `table` | Laje (o nome do corpo é `table`, o do geom é `table_geom`): corpo mocap, sem junta livre | Meia dimensão (`Cena.prateleira_meia_xy`; `Cena.prateleira_meia_xy`; `Cena.prateleira_meia_z`) = (0,30; 0,30; 0,02) m, ou seja, 0,6 × 0,6 × 0,04 m. Atrito `Cena.prateleira_atrito` = (1,0; 0,02; 0,001). `priority` `Cena.prioridade_laje` = 3. `condim` `Cena.prateleira_condim` = 3. Cor (RGBA) `Cena.prateleira_rgba` = (0,5; 0,5; 0,55; 1,0) (`knobs.py:92`; `cena.py:89-90,127`): só aparência, passada ao geom na criação do modelo; nenhuma conta do treino a lê. Grupo 1. Posição inicial (0,50; 0,00; 0,53 + 20,0) m. | `cena.py:118-128`, `:305-309`; `knobs.py:89-93`, `:105`, `:94`, `:114` |
| `terrain` | Gerador de terreno por colunas (Seção 7) | `terrain_type` "generator" | `cena.py:277` |

As posições iniciais de `box` e `table` valem até o primeiro reset. Os z iniciais (0,65 e 0,53, mais 20,0 de `Cena.estaciona_z`) saem de `geometria_de_repouso`: topo = `Cena.prateleira_topo_teto` = 0,55 m; centro da laje = 0,55 − 0,02; caixa = 0,55 + 0,10 (`cena.py:281-290`; `knobs.py:95`).

O geom da caixa e o da laje caem no grupo de geom 1, que não é o chão (grupo 0). Os sensores de altura leem só o grupo 0 (`cena.py:95-103`, `:330-340`).

### 8.3 O robô (`cena.robot_cfg`, `cena.py:202-234`)

- Posição inicial: (0; 0; `pelve_de_pe` + 0,01) m. Juntas na pose de pé `q_de_pe`, velocidade 0 (`cena.py:222-227`).
- Colisões (`cena.py:209-220`): geoms `.*_collision` e `.*_pad`. `condim` 3 nos pés (`^(left|right)_foot[1-7]_collision$`) e nos pads de palma; 1 nos pads de dorso e nos demais. `priority` 1 nos pés. Atrito (primeiro padrão que casa vence): pés (0,6); demais `.*_collision` `Cena.robo_atrito` = (0,3; 0,02; 0,001) (`knobs.py:84`).
- Pads (`cena.py:132-156`): as cápsulas `*_hand_collision` saem e entram, em cada mão, um pad de palma e um pad de dorso (caixas de meia dimensão (0,035; 0,008; 0,045) m em (0,10; ±dz; 0) no corpo da mão). Pad de palma: dz = ±0,037 m (esquerda −, direita +), `condim` 3, atrito `Cena.pad_atrito` = (1,0; 0,02; 0,001) (`knobs.py:81`). Pad de dorso: dz = +0,015 m na esquerda e −0,015 m na direita (sinal oposto ao da palma), `condim` 1, atrito `Cena.robo_atrito`. Grupo de geom 3 (`cena.py:41`).
- Sites no `torso_link`: `mid360` em `Percepcao.mid360_pos` = (0,0002835; 0,00003; 0,428434) m com quaternion `Percepcao.mid360_quat` = (0,0; 0,99967; 0,0; −0,02556) (`knobs.py:501-502`); `d435` em `Estimador.d435_pos` = (0,0576235; 0,01753; 0,42987) m, girado `Estimador.d435_pitch` = 0,8307767 rad em torno de y (`knobs.py:477-478`) (`cena.py:159-167`).
- Atuadores (`cena.py:176-199`): um `BuiltinPositionActuatorCfg` por linha de `Contrato.pd` (9 grupos). `stiffness` e `damping` da linha; `effort_limit` e `armature` do motor da junta no mjlab; `delay_min_lag` 0; `delay_max_lag` = `DR.atraso_refino[1]` × `Contrato.decimacao` = 3 × 4 = 12 passos de física; `delay_update_period` = `NUNCA` = 10⁹; `delay_per_env_phase` False (`knobs.py:47`, `:513`).

`Contrato.pd` (kp; kd), por linha (`knobs.py:65-72`):

| Linha | Juntas (regex) | kp | kd |
|---|---|---|---|
| 0 | `.*_hip_pitch_joint` | 100,0 | 2,0 |
| 1 | `.*_hip_yaw_joint` | 100,0 | 2,0 |
| 2 | `.*_hip_roll_joint` | 100,0 | 2,0 |
| 3 | `.*_knee_joint` | 150,0 | 4,0 |
| 4 | `.*_ankle_(pitch|roll)_joint` | 40,0 | 2,0 |
| 5 | `waist_yaw_joint` | 200,0 | 5,0 |
| 6 | `waist_(roll|pitch)_joint` | 40,0 | 5,0 |
| 7 | `.*_(shoulder_.*|elbow|wrist_roll)_joint` | 40,0 | 1,0 |
| 8 | `.*_wrist_(pitch|yaw)_joint` | 40,0 | 1,0 |

### 8.4 Sensores (`cena.sensores`, `cena.py:343-429`)

`env_cfg.py:95` troca a lista de sensores do molde pela desta função. São 11 sensores fixos, 6 de altura e contato e, com `Percepcao.ligada` = True (`knobs.py:483`), mais o lidar: 18 no total (calculado de `cena.py:417-429`).

| Sensor | Tipo | Primário | Secundário | Campos e redução | Linha |
|---|---|---|---|---|---|
| `palma_E`, `palma_D` | contato | geom do pad de palma do robô | geom `box_geom` | `found`, `force`; `netforce`; 1 slot | `cena.py:348-353` |
| `dorso_E`, `dorso_D` | contato | geom do pad de dorso | geom `box_geom` | só `found`; sem redução; 1 slot | `cena.py:355-360` |
| `apoio_caixa` | contato | geom `box_geom` | geom `table_geom` | `found`, `force`; `netforce`; 1 slot; `track_air_time` True | `cena.py:361-367` |
| `corpo_prateleira` | contato | `pelvis_collision`, `torso_collision`, `.*_hip_collision`, `.*_thigh_collision` | `table_geom` | `found`, `force`; `netforce`; 1 slot | `cena.py:45`, `:368-373` |
| `palma_prateleira` | contato | `.*_palm_pad` | `table_geom` | idem | `cena.py:46`, `:368-373` |
| `dorso_prateleira` | contato | `.*_hand_back_pad` | `table_geom` | idem | `cena.py:47`, `:368-373` |
| `caixa_robo` | contato | geom `box_geom` | subárvore da pelve do robô | `force`, `found`; `maxforce`; 12 slots | `cena.py:375-378` |
| `auto_colisao` | contato | subárvore da pelve | subárvore da pelve | `found`, `force`; sem redução; 1 slot; histórico de 4 | `cena.py:379-382` |
| `pes_chao` | contato | subárvore de `left_ankle_roll_link` e de `right_ankle_roll_link` | nenhum (qualquer contato conta) | `found`, `force`; `netforce`; 1 slot; `track_air_time` True | `cena.py:383-389` |
| `altura_pes` | altura do terreno | corpos `left_ankle_roll_link`, `right_ankle_roll_link` | — | anel de 0,03 m de raio, 6 amostras; alinhado ao yaw; grupo de geom 0 | `cena.py:392-393`, `:330-340` |
| `altura_pelve` | altura do terreno | corpo `pelvis` | — | anel de 0,05 m, 4 amostras | `cena.py:394` |
| `altura_joelhos` | altura do terreno | corpos `left_knee_link`, `right_knee_link` | — | anel de 0,03 m, 4 amostras | `cena.py:395-396` |
| `altura_caixa` | altura do terreno | corpo `box` | — | anel de 0,05 m, 4 amostras | `cena.py:397` |
| `nao_pes_chao` | contato | cada corpo do robô, menos `.*_ankle_roll_link` | corpo `terrain` | `found`, `force`; `netforce`; 1 slot | `cena.py:402-408` |
| `caixa_faces` | contato | geom `box_geom` | subárvore da pelve | `found`, `force`, `normal`, `tangent`; `maxforce`; 12 slots; referencial global | `cena.py:410-415` |
| `lidar` | raios | site `mid360` do robô | — | padrão de raios dado pelas direções do primeiro passo de `mid360_passos.npz` (o número de raios é o do arquivo, que o código lido não mostra; `Percepcao.raios_por_passo` = 205, `knobs.py:489`); alinhado à base; alcance máximo `Percepcao.alcance[1]` = 6,0 m; grupos de geom 0, 1 e 3; exclui o corpo pai | `cena.py:313-327`, `:423-428`; `knobs.py:490` |

⚠ Divergência: o comentário de `env_cfg.py:95` fala em "os 7 de altura/contato". A tupla `alturas` tem 6 elementos (`cena.py:418`) e o lidar entra à parte (`cena.py:429`).

O molde do mjlab (sensores `terrain_scan`, `foot_height_scan`, `feet_ground_contact`, `self_collision`) não sobrevive: a atribuição `cfg.scene.sensors = ...` o substitui inteiro (`env_cfg.py:95`).

---

## 9. O empurrão

### 9.1 Evento `empurrao` (`eventos.empurrao`, `eventos.py:261-269`)

| Item | Valor | Referência |
|---|---|---|
| Modo | `interval`, com tempo por env (`is_global_time` = False, o padrão) | `env_cfg.py:201-203`; `mjlab/managers/event_manager.py:133-136` |
| Primeiro tempo | Cada env sorteia o tempo até o primeiro empurrão com `U(3,0; 7,0)` s (`DR.periodo_empurrao_s`), na criação do gerente | `knobs.py:520`; `mjlab/managers/event_manager.py:370-375` |
| Período | Depois de cada disparo, o env sorteia o próximo tempo com `U(3,0; 7,0)` s (torch global) | `mjlab/managers/event_manager.py:278-287` |
| Reinício no reset | Nenhum. O gerente só reinicia o tempo dos termos de classe. `empurrao` é função. O tempo restante segue entre episódios. | `mjlab/managers/event_manager.py:225-226`, `:359-360` |
| O que faz | Soma à velocidade linear da raiz do robô, no referencial do mundo, `U(−0,5; 0,5)` m/s em x e `U(−0,5; 0,5)` m/s em y. `DR.empurrao` = 0,5 m/s. As chaves z, roll, pitch e yaw valem (0; 0). | `knobs.py:517`; `mjlab/envs/mdp/events.py:316-337`, `:27-41` |
| Quem recebe | Todo env que dispara, em qualquer tarefa e em qualquer fatia. A função não tem máscara nem trava de currículo. | `eventos.py:261-269` |
| Escrita extra | `v3_t_empurrao[env_ids]` = `episode_length_buf · step_dt` (s desde o começo do episódio) | `eventos.py:269` |
| Zera `v3_t_empurrao` | O evento `fatia` põe −∞ no reset | `eventos.py:163` |
| Quem lê `v3_t_empurrao` | O comando, em `tempo_da_ancora` | `comando.py:221`, `:834` |
| Currículo | Nenhum. O evento não lê `Curriculo.dr()`: vale desde a iteração 0, em toda tarefa, com a faixa fixa de `DR.empurrao`. | `env_cfg.py:200-205` |
| Play | Não existe | `env_cfg.py:200-205` |

O empurrão altera a velocidade do robô, não a da caixa. Quem segura a caixa a recebe só pelo contato.

### 9.2 O empurrão contra um lance de degraus (derivado)

Esta seção compara o empurrão com o tempo que o robô leva para cruzar a escada que o código monta no episódio de terreno do ANDAR.

**A geometria da `escada`** (Seção 7.2; `primitive_terrains.py:84-92`, `:135-139`, `:233-234`). A célula tem 8 m de lado, borda de 1,0 m, plataforma central de 2,0 m (`Terreno.plataforma`) e degraus de 0,3 m de profundidade. O número de anéis é `int((8 − 2 · 1,0 − 2,0) / (2 · 0,3))` = 6. O topo central mede 8 − 2 · 1,0 − 2 · 6 · 0,3 = 2,4 m de lado. O topo do anel `k` (de fora para dentro, `k` = 0 a 5) é (`k` + 1) · `h`, o topo central é 7 · `h`, e `h` = 0,20 · `L` / 12 m na linha `L`.

A partir do centro, em linha reta ao longo de um eixo:
- 1,2 m de plataforma (a metade de 2,4 m);
- 1,8 m de degraus (6 anéis de 0,3 m, de 1,2 m a 3,0 m);
- de 3,0 m a 4,0 m, a borda plana de 1,0 m, em z = 0; a célula vizinha começa em 4,0 m (metade dos 8 m da célula).

São 7 degraus de altura `h`: o primeiro a 1,2 m do centro e o último a 3,0 m. O `dist_travessia` = 3,5 m fica no meio da borda plana: a travessia exige passar do último degrau, e o robô ainda não chega à célula vizinha (`knobs.py:442`; o comentário do knob dá a mesma geometria). A `escada` desce do centro para a borda, e a `escada_inv` sobe (o spawn fica no fundo do poço). Na linha 12, `h` = 0,20 m, e a escada inteira tem 7 · 0,20 = 1,4 m de desnível.

**O empurrão** (Seção 9.1). O intervalo vale U(3; 7) s por env, com média 5 s (`knobs.py:520`), e o tempo restante segue entre episódios. A faixa é U(−0,5; 0,5) m/s em x e em y, no referencial do mundo (`knobs.py:517`). Simulação (derivado): 200 000 sorteios com `torch.Generator().manual_seed(0)`; o módulo do empurrão tem média 0,382 m/s e máximo 0,707 m/s; 71,6 % dos empurrões passam de 0,3 m/s e 21,4 % passam de 0,5 m/s.

**A chance de um empurrão dentro de uma janela de `T` s** (derivado: 200 000 janelas, cada uma com instantes de empurrão de intervalos U(3; 7) s e começo uniforme depois de 30 s, que descarta a fase inicial; as medianas até 3,5 m são as da simulação do arquivo 04, seção 4.6):

| Janela | `T` (s) | P(≥ 1 empurrão) | Empurrões esperados |
|---|---|---|---|
| Lance de 1,8 m a 1,0 m/s | 1,8 | 0,359 | 0,36 |
| 3,5 m a 1,0 m/s | 3,5 | 0,695 | 0,70 |
| Lance de 1,8 m a 0,5 m/s | 3,6 | 0,712 | 0,72 |
| 3,5 m: mediana no nível 7 (`f` = 1,0, entre os que chegam, 04 seção 4.6) | 5,3 | 0,928 | 1,06 |
| Lance de 1,8 m a 0,3 m/s | 6,0 | 0,975 | 1,20 |
| 3,5 m a 0,5 m/s | 7,0 | 1,000 | 1,40 |
| 3,5 m: mediana no nível 2 (`f` = 1,0) | 10,8 | 1,000 | 2,16 |
| 3,5 m a 0,3 m/s | 11,7 | 1,000 | 2,33 |
| 3,5 m: mediana no nível 0 (`f` = 1,0) | 16,1 | 1,000 | 3,22 |

Para `T` ≤ 3 s há no máximo um empurrão (o intervalo mínimo é 3 s), e a chance vale `T` / 5 (derivado: 1,8 / 5 = 0,36). Num episódio de 20 s, o robô recebe em média 4 empurrões (derivado: 20 / 5).

**O que a comparação diz.** Um lance de degraus a 0,5 m/s leva 3,6 s, e o robô recebe um empurrão nele com chance 0,71. O empurrão de média 0,38 m/s é comparável à velocidade pedida no nível 2 (`frente` máxima 0,5 m/s, `vx` = `frente` · √`u`). Ele vale desde a iteração 0 e não depende do currículo, então o robô aprende a escada já com o empurrão ligado. O arquivo `04-comando-de-velocidade.md`, seção 4.6, traz os tempos até 3,5 m por nível.

---

## 10. Força na caixa e ruído de força nos atuadores

### 10.1 Evento `forca_caixa` (classe `eventos.forca_caixa`, `eventos.py:584-602`)

A classe herda `apply_body_impulse` do mjlab (`mjlab/envs/mdp/events.py:386-605`). Um impulso é uma força constante aplicada à caixa por um tempo curto.

| Item | Valor | Referência |
|---|---|---|
| Modo | `step`: roda em todos os envs, todo passo, com `env_ids` = None | `env_cfg.py:207`; `mjlab/managers/event_manager.py:289-291` |
| Trava | `Curriculo.dr()["forca_caixa"]` = trava `subida`: `nivel("PEGAR") + nivel("CARREGAR") > 0`. Com a trava desligada, `__call__` volta sem fazer nada, e os temporizadores do pai não avançam. | `eventos.py:595-596`; `balanceador.py:202`, `:383` |
| Ciclo de um env | Espera (cooldown) → dispara a força → mantém → expira → nova espera | `mjlab/envs/mdp/events.py:394-402`, `:467-552` |
| Cooldown | `U(3,0; 8,0)` s (`DR.forca_caixa_periodo_s`). Sorteado na criação (`DR.forca_caixa_periodo_s` do cfg), a cada reset do env, a cada disparo e a cada expiração. O sorteio da expiração substitui o do disparo, então a espera até o próximo impulso conta desde a expiração. O contador desce a cada passo, também com o impulso ativo, e um env só dispara se não tem impulso ativo e o contador chegou a 0 ou menos. | `knobs.py:533`; `mjlab/envs/mdp/events.py:456-461`, `:510`, `:513-516`, `:552`, `:602-605` |
| Duração | `U(0,1; 0,3)` s (`DR.forca_caixa_dur_s`), sorteada no disparo | `knobs.py:532`; `mjlab/envs/mdp/events.py:544-549` |
| Força | Cada componente (x, y, z) uniforme em `U(−11,5; 11,5)` N (`DR.forca_caixa_N`), independentes. Norma máxima calculada: 11,5 · √3 ≈ 19,9 N. | `knobs.py:531`; `mjlab/envs/mdp/events.py:520-526` |
| Torque | 0 (faixa (0,0; 0,0)) | `env_cfg.py:208` |
| Ponto de aplicação | Centro de massa da caixa (`body_point_offset` não é passado) | `env_cfg.py:207-209`; `mjlab/envs/mdp/events.py:528-538` |
| Corpo | `box` | `env_cfg.py:209` |
| Magnitude por nível | Constante. Não depende do nível, do degrau nem da massa. | `eventos.py:584-602` |
| Máscara | Depois do pai escrever, a classe zera a força dos envs fora de `(tarefa == CARREGAR) & v3_pegou & ~v3_perdeu` e escreve zero a cada passo nesses envs. `CARREGAR` = índice 2 de `TAREFAS`. | `eventos.py:33`, `:598-602`; `knobs.py:42` |
| Efeito da máscara | Os temporizadores seguem contando nos envs mascarados; o que a máscara corta é a força escrita. Um impulso que dispara com o env fora da máscara é zerado no mesmo passo. Um impulso disparado com o env dentro da máscara e depois zerado (por exemplo, quando `v3_perdeu` vira True) não volta: o pai só escreve a força no disparo. | `eventos.py:597-602`; `mjlab/envs/mdp/events.py:540-542` |
| Zeragem no reset | `reset(env_ids)` zera a força ativa dos envs em reset e sorteia novo cooldown. O gerente chama `reset` nos termos de classe no fim do reset. | `mjlab/envs/mdp/events.py:584-605`; `mjlab/managers/event_manager.py:217-220` |
| Play | Não existe | `env_cfg.py:200`, `:207-209` |

`v3_pegou` vira True quando o alvo avança a partir do P2 (`comando.py:497`) e volta a False no reset (`comando.py:651-653`). `v3_perdeu` vira True na perda da caixa (`comando.py:414`) e volta a False no reset (`comando.py:651-653`).

### 10.2 Evento `rfi` (`eventos.rfi`, `eventos.py:605-615`)

RFI é o ruído de força no atuador: um torque aleatório somado em cada junta.

| Item | Valor | Referência |
|---|---|---|
| Modo | `step`: todo passo, todos os envs, sem trava e sem máscara de tarefa | `env_cfg.py:210`; `eventos.py:605-615` |
| Escala `s` | `Curriculo.dr().get("rfi", 0.0)`. Com a trava `subida` desligada, `s` = 0. Com ela ligada, `s` = `no_degrau(DR.rfi[0], DR.rfi[1], g, Balanceador.degraus)` = 0,05 + (0,10 − 0,05) · `g` / 5. | `eventos.py:612`; `balanceador.py:51`, `:384`; `knobs.py:534`, `:401` |
| Sorteio | Um uniforme em [0; 1) por env e por junta (29), float64, no `v3_gerador`, todo passo | `eventos.py:613` |
| Torque escrito | `τ = (2u − 1) · s · TAU_MAX[junta]`, em N·m, uniforme em ± `s · TAU_MAX` | `eventos.py:614` |
| `TAU_MAX` | Tabela por regex de `Recompensa.tau_max`, na ordem de `JUNTAS`: joelho 139; hip_roll 139; hip_pitch e hip_yaw 88; waist_yaw 88; waist_roll e waist_pitch 50; tornozelo (pitch e roll) 50; ombro, cotovelo e wrist_roll 25; wrist_pitch e wrist_yaw 5 (N·m) | `knobs.py:286-292`, `:572` |
| Onde escreve | `sim.data.qfrc_applied[:, joint_v_adr]`, uma vez por passo da política. Os eventos `step` rodam depois da física do passo (`mjlab/envs/manager_based_rl_env.py:421-461`): o torque escrito num passo vale nos 4 subpassos do passo seguinte. | `eventos.py:615` |
| Com `s` = 0 | Escreve zero. O sorteio roda mesmo assim. | `eventos.py:612-615` |
| Play | Não existe | `env_cfg.py:200`, `:210` |

O código lido de `mjlab/sim` e `mjlab/entity` só lê `qfrc_applied`; não o zera (busca textual em `mjlab/sim/*.py` e `mjlab/entity/*.py`). O valor escrito persiste até a próxima escrita. Se o integrador do MuJoCo mantém esse campo entre os subpassos, ver a Seção 13.

---

## 11. Randomização de domínio

Randomização de domínio (DR) é o sorteio de parâmetros físicos do simulador para o robô treinar sob variação. A tabela resume quando cada uma ocorre e se a faixa depende do currículo.

| DR | Momento | Faixa | Depende do currículo? | Play |
|---|---|---|---|---|
| Atrito do pé | startup | `DR.atrito_pe` (0,3; 1,0) | Não | sim |
| Massa do tronco | startup | `DR.massa_tronco` (−1,0; 3,0) kg somados | Não | sim |
| Viés de encoder | startup | ± `DR.vies_encoder` = 0,015 rad | Não | sim |
| Tamanho da caixa | startup | 8 meias arestas entre 0,07 e 0,13 m | Não | sim |
| Atraso | startup (teto) + sorteio do lag a cada episódio | Teto sobe de 0 a 3 passos | Sim (restrição e degrau global) | sim (teto 0) |
| Atrito do pad de palma | reset | Liga com a trava `portao0`; borda de baixo desce com o nível | Sim | sim |
| Atrito da laje | reset | Liga com a trava `portao0`; (0,3; 1,0) | Liga com o currículo; a faixa é fixa | sim |
| Viés do IMU | reset | Liga com a trava `vies` | Liga com o currículo; a faixa é fixa | sim |
| Ganhos PD (kp e kd) | reset | Fator (1 → 0,9; 1 → 1,1) pelo degrau global | Sim | **não** |
| Força do motor | reset | Fator pernas (1 → 0,9; 1 → 1,1); braços (1 → 0,8; 1 → 1,1) | Sim | **não** |
| `solref` | reset | (0,02 → 0,015; 0,02 → 0,03) pelo degrau global | Sim | **não** |
| Centro de massa da caixa | reset | Só com `DR.com_caixa` = True (valor lido False) | Sim, se ligado | sim, se ligado |
| Massa e inércia da caixa | reset | Massa por fatia e nível (Seção 5.1.1) | Sim | sim |
| RFI | step | Escala 0,05 a 0,10 | Sim | **não** |

### 11.1 Travas e degrau que ligam as faixas (`Curriculo.dr()`, `balanceador.py:358-389`)

Os valores abaixo saem do fecho de iteração mais recente.

| Chave do `dr()` | Valor | Referência |
|---|---|---|
| `portao0` (trava) | Liga quando `nivel_passou("PEGAR") >= 0`: a célula PEGAR `pegar` passou no portão com o subnível no último e, no nível 0, com ao menos uma medida; ou o nível do PEGAR é ≥ 1. Só liga. | `balanceador.py:200`, `:350-356` |
| `vies` (trava) | Liga quando a nota `s` da célula ANDAR `andar` ≥ `Balanceador.portao_andar` = 0,75. Só liga. | `balanceador.py:201`; `knobs.py:392` |
| `subida` (trava) | Liga quando `nivel("PEGAR") + nivel("CARREGAR") > 0`. Só liga. | `balanceador.py:202` |
| `g` | Degrau global (0 a 5) | `balanceador.py:341-343`, `:366` |
| `atrito_pad` | `portao0` | `balanceador.py:377` |
| `atrito_laje` | `portao0` | `balanceador.py:378` |
| `vies_imu` | `vies` | `balanceador.py:376` |
| `mu_min_pad` | `DR.atrito_pad[0]` = 0,6 sem `portao0`; com `portao0`: `max(DR.mu_pad_min, DR.atrito_pad[0] − DR.passo_mu_pad · (nivel("PEGAR") + nivel("CARREGAR")))` = `max(0,5; 0,6 − 0,025 · soma)`. `DR.mu_pad_min` = 0,5 (`knobs.py:526`); `DR.passo_mu_pad` = 0,025 (`knobs.py:525`). | `balanceador.py:371-373`, `:379` |
| `atraso_max` | `DR.atraso_refino[1]` = 3 se `g ≥ DR.degrau_atraso_refino` = 3; senão `ceil(DR.atraso_portao[1] · restricao − 1e-9)`, com `DR.atraso_portao[1]` = 2 | `balanceador.py:381-382`; `knobs.py:512-514` |
| `forca_caixa` | `subida` | `balanceador.py:383` |
| `rfi` | `no_degrau(0,05; 0,10; g; 5)` com `subida`; senão 0,0 | `balanceador.py:384`; `knobs.py:534` |
| `pd` | `[no_degrau(1; 0,9; g; 5); no_degrau(1; 1,1; g; 5)]` = `[1 − 0,02 · g; 1 + 0,02 · g]` (calculado) | `balanceador.py:368-369`, `:385`; `knobs.py:535` |
| `forca_motor` | `[1 − 0,02 · g; 1 + 0,02 · g]` (pernas e cintura) | `balanceador.py:386`; `knobs.py:536` |
| `forca_motor_bracos` | `[1 − 0,04 · g; 1 + 0,02 · g]` (calculado a partir de (0,8; 1,1)) | `balanceador.py:387`; `knobs.py:537` |
| `solref` | `[0,02 − 0,001 · g; 0,02 + 0,002 · g]` (calculado a partir de `DR.solref_base` = 0,02 e `DR.solref` (0,015; 0,03)) | `balanceador.py:388`; `knobs.py:539-540` |
| `com_caixa` | `DR.com_caixa_frac_max` (0,3; 0,2) se `subida`; senão `DR.com_caixa_frac0` (0,2; 0,2) se `portao0`; senão (0; 0). Vale (0; 0) se `DR.com_caixa` é False. | `balanceador.py:374`, `:380`; `knobs.py:527-529` |

`no_degrau(ini, fim, d, degraus)` = `ini + (fim − ini) · d / degraus` (`balanceador.py:48-51`). `Balanceador.degraus` = 5 (`knobs.py:401`).

A classe `Refino` tem um só campo, `degrau_batente` = 1 (`knobs.py:547`). Nenhum evento e nenhuma chave do `dr()` o lê; a terminação `batente` o lê (`terminacoes.py:181`).

Todas as faixas do `dr()` dependem do currículo só por `g`, pelo nível e pelas travas. Nenhuma depende da fatia ou da tarefa do env: os eventos `pd`, `forca_motor`, `solref`, `atrito_pad`, `atrito_laje`, `vies_imu`, `com_caixa` e `rfi` sorteiam para todos os envs que resetam (ou todos, no passo), sem máscara. A única máscara por tarefa entre as perturbações é a da `forca_caixa` (Seção 10.1).

O degrau global `g` só sobe quando as quatro tarefas passam do degrau anterior, porque é o mínimo (`balanceador.py:341-343`). Cada degrau de tarefa exige o último nível da tarefa e, no ANDAR e no CARREGAR, a célula `terreno` aberta e a linha de terreno no teto (`balanceador.py:239-241`).

### 11.2 Atrito

**`atrito_pe`** (startup, `dr.geom_friction`; `env_cfg.py:185-187`):

- Sorteia um valor por env, uniforme em `DR.atrito_pe` = (0,3; 1,0), com o gerador global do torch.
- `shared_random` = True: o mesmo valor vale para os 14 geoms de pé (7 de cada lado) do env (`mjlab/envs/mdp/dr/_core.py:115-125`).
- `operation` = "abs": o valor sorteado substitui o atrito (`mjlab/envs/mdp/dr/_types.py:83-88`).
- Escreve só o eixo 0 (atrito tangencial) (`mjlab/envs/mdp/dr/geom.py:134-146`; `_core.py:138-148`).
- O valor fica fixo até o fim do treino.

**`atrito_pad`** (reset, `eventos.atrito_pad`, `eventos.py:464-475`):

- Com `portao0` desligada: faixa (`Cena.pad_atrito[0]`; `Cena.pad_atrito[0]`) = (1,0; 1,0). O atrito fica fixo em 1,0 (`knobs.py:81`).
- Com `portao0` ligada: faixa (`mu_min_pad`; `DR.atrito_pad[1]`) = (`mu_min_pad`; 1,0) (`knobs.py:523`). `mu_min_pad` vale 0,6 − 0,025 · (`nivel("PEGAR")` + `nivel("CARREGAR")`), com piso 0,5 (Seção 11.1).
- `dr.geom_friction` com `operation` "abs", sem `shared_random`: cada um dos 2 pads de palma recebe um valor independente por env, a cada reset (`mjlab/envs/mdp/dr/_core.py:126-134`).
- Escreve só o eixo 0. Vale em todas as fatias e tarefas.

**`atrito_laje`** (reset, `eventos.atrito_laje`, `eventos.py:478-489`):

- Com `portao0` desligada: faixa (`Cena.prateleira_atrito[0]`; idem) = (1,0; 1,0) (`knobs.py:91`).
- Com `portao0` ligada: faixa `DR.atrito_laje` = (0,3; 1,0), fixa (`knobs.py:524`).
- Um valor por env e por reset, no geom `table_geom`. Eixo 0.

A laje tem `priority` 3 (`Cena.prioridade_laje`, `knobs.py:105`). Os pés têm `priority` 1 (`cena.py:217`). A regra de prioridade de contato do MuJoCo não está no código lido (Seção 13).

### 11.3 Massa e viés

**`massa_tronco`** (startup, `dr.body_mass`; `env_cfg.py:188-190`):

- Soma `U(−1,0; 3,0)` kg (`DR.massa_tronco`) à massa padrão do `torso_link`, um sorteio por env, com o gerador global (`mjlab/envs/mdp/dr/_core.py:110-111`, `:127-136`; `_types.py:95-100`).
- `operation` "add" parte da massa padrão do modelo, não da massa corrente (`uses_defaults` True).
- A função escreve só `body_mass`. A inércia não muda (`mjlab/envs/mdp/dr/body.py:315-326`).
- O valor fica fixo até o fim do treino.

**`vies_encoder`** (startup, `dr.encoder_bias`; `env_cfg.py:191-192`):

- Sorteia um viés por junta (29) e por env, uniforme em (−0,015; 0,015) rad, com o gerador global (`mjlab/envs/mdp/dr/joint.py:209-241`).
- O alvo de posição enviado ao atuador vale `ação processada − viés` (`mjlab/envs/mdp/actions/actions.py:221-224`).
- A observação `juntas` do ator soma o viés à posição (`biased` True); o crítico lê `juntas` sem viés (`observacoes.py:394`, `:404`; `mjlab/entity/data.py:368-371`; `mjlab/envs/mdp/observations.py:51-61`).
- O valor fica fixo até o fim do treino.

**`vies_imu`** (reset, `eventos.vies_imu`, `eventos.py:492-509`):

- Trava `vies` desligada: `v3_vies_giro` = 0 e `v3_incl_imu` = identidade (`eventos.py:505-507`).
- Trava `vies` ligada: dois sorteios uniformes (2, n, 3) no `v3_gerador` (`eventos.py:500`):
  - `v3_vies_giro` = `(2 · u₀ − 1) · DR.vies_imu_giro` = ± 0,05 rad/s por eixo (`knobs.py:521`; `eventos.py:501`);
  - ângulos = `(2 · u₁ − 1) · radianos(DR.vies_imu_incl_graus)` = ± 1,0° por eixo, com o ângulo em z forçado a 0 (`knobs.py:522`; `eventos.py:502-503`); `v3_incl_imu` é a matriz de rotação desse ângulo (`eventos.py:504`).
- Um sorteio por episódio, todas as fatias e tarefas, sem máscara.
- O ator soma `v3_vies_giro` ao giro da base e gira a gravidade projetada por `v3_incl_imu` (`observacoes.py:203-209`).

### 11.4 Ganhos PD, força do motor e `solref`

**`pd`** (reset, `eventos.pd`, `eventos.py:523-531`):

- Chama `dr.pd_gains` com `kp_range` = `kd_range` = `dr()["pd"]` (Seção 11.1), `operation` "scale" e `distribution` uniforme sobre o `v3_gerador` (`eventos.py:516-520`, `:529-531`).
- Sorteia, por env e por junta, um fator de kp e um fator de kd independentes (`mjlab/envs/mdp/dr/actuator.py:76-92`). O ganho final é o ganho padrão × o fator (`mjlab/envs/mdp/dr/actuator.py:97-107`).
- Cobre os 9 grupos de atuador (`SceneEntityCfg("robot")` seleciona todos: `mjlab/managers/scene_entity_config.py:93`).
- Com `g` = 0, a faixa é (1; 1) e o fator vale 1. Com `g` = 5: (0,9; 1,1). A cada reset, sem trava de tarefa.

**`forca_motor`** (reset, `eventos.forca_motor`, `eventos.py:534-545`):

- Chama `dr.effort_limits` duas vezes, com `operation` "scale" e distribuição uniforme no `v3_gerador` (`eventos.py:543-545`).
- Pernas e cintura (grupos 0 a 6, `env_cfg.py:151-152`): faixa `dr()["forca_motor"]`. Com `g` = 5: (0,9; 1,1).
- Braços (grupos 7 e 8): faixa `dr()["forca_motor_bracos"]`. Com `g` = 5: (0,8; 1,1).
- O índice `bracos` reúne as linhas de `Contrato.pd` cuja regex casa `shoulder|elbow|wrist` (linhas 7 e 8); `pernas` reúne as outras sete (`env_cfg.py:151-152`; `knobs.py:65-72`).
- O fator multiplica os dois lados do limite de força do atuador (`mjlab/envs/mdp/dr/actuator.py:250-257`) e um fator por junta (`actuator.py:235-245`).

**`solref`** (reset, `eventos.solref`, `eventos.py:548-565`):

- Um uniforme por env, no `v3_gerador`: `u = lo + (hi − lo) · U`, com (`lo`; `hi`) = `dr()["solref"]` (`eventos.py:556-560`).
- Escreve `solref[0]` em todos os geoms do robô (pés, pads de palma, pads de dorso), da laje e da caixa, com o mesmo `u` dentro do env (`eventos.py:561-565`). A coluna 1 (amortecimento) não muda.
- Com `g` = 0: (0,02; 0,02). Com `g` = 5: (0,015; 0,03). O terreno fica de fora (`env_cfg.py:178`).

### 11.5 Tamanho, massa e centro de massa da caixa

**`tamanho_caixa`** (startup, `eventos.tamanho_caixa`, `eventos.py:416-440`):

- `valores` = 8 valores igualmente espaçados em `Cena.caixa_meia_faixa` = (0,07; 0,13) m: 0,070; 0,0786; 0,0871; 0,0957; 0,1043; 0,1129; 0,1214; 0,130 (calculado de `eventos.py:429`).
- Cada env escolhe um valor com igual probabilidade (`⌊u · 8⌋`, `u` no `v3_gerador`) (`eventos.py:430`).
- Escreve `geom_size`, `geom_rbound` (`a · √3`) e `geom_aabb` do geom da caixa e grava `v3_meia[env]` = a meia aresta (`eventos.py:433-440`).
- Fica fixo até o fim do treino.

**Massa da caixa:** sorteada no reset pelo evento `cena` (Seção 5.1.1) e escrita pelo evento `carga` (Seção 5.4). A faixa depende da fatia e do nível do PEGAR (B), ou do maior nível do PEGAR que passou no portão (C). O ANDAR usa 1,0 kg.

**`com_caixa`** (reset, `eventos.com_caixa`, `eventos.py:568-581`):

- Só existe no cfg se `DR.com_caixa` é True. O valor lido é False (`knobs.py:527`; `env_cfg.py:182-183`).
- Se ligado: `fx, fz` = `dr()["com_caixa"]` (Seção 11.1). Desloca o centro de massa da caixa por eixo: x e y uniformes em (−`fx`; `fx`) e z uniforme em (−`fz`; `fz`), multiplicados pela meia aresta `v3_meia` do env, somados à posição padrão (`eventos.py:575-581`).

### 11.6 Atraso

O atraso usa dois tipos de buffer, cada um com um lag por env: um buffer no atuador (comando de posição) e um buffer por termo de observação do ator. Os 9 grupos de atuador têm o mesmo tipo, a mesma transmissão e a mesma configuração de atraso (`cena.py:190-198`). O mjlab os funde num buffer único por env (`mjlab/actuator/builtin_group.py:112-121`, `:161-173`). Por isso as 29 juntas compartilham o mesmo lag de comando em cada env.

**Evento `atraso`** (startup, `eventos.atraso`, `eventos.py:244-258`): põe o teto corrente do atraso nos buffers.

- Buffer do atuador (um só, fundido; `eventos.py` o junta por `id`): `max_lag` = `passos_max · decimacao` (passos de física) (`eventos.py:253-255`).
- Buffers de observação: `max_lag` = `passos_max` (passos da política) (`eventos.py:256-258`).
- No startup, `passos_max` = `DR.atraso_inicio[1]` = 0. A capacidade do buffer (3 passos da política e 12 de física) não muda (`cena.py:188`; `observacoes.py:371`, `:384`).

**Quando o teto sobe:** o termo `v3` chama `eventos.atraso` em todo reset em que `dr()["atraso_max"]` difere de `DR.atraso_inicio[1]` (0). A chamada põe o novo teto em todos os buffers de todos os envs (`curriculo.py:123-126`). `atraso_max` segue a regra da Seção 11.1: `ceil(2 · restricao)` até `g` = 3 e 3 depois.

**Sorteio do lag:** cada buffer sorteia o lag no primeiro `compute` depois do reset do env, uniforme inteiro em {0, …, `max_lag`}, com o gerador global do torch. Nos primeiros passos depois do reset, o lag efetivo fica limitado ao número de quadros já guardados menos 1 (`delay_buffer.py:240-243`). O lag fica fixo no resto do episódio, porque `delay_update_period` = `NUNCA` = 10⁹ e `delay_per_env_phase` = False (`mjlab/utils/buffers/delay_buffer.py:194-209`, `:247-258`, `:269-277`; `cena.py:196-198`; `observacoes.py:383-386`).

⚠ Divergência: o comentário de `cena.py:196` diz que o lag sorteado no reset está em `eventos`. O código de `eventos.py` só ajusta o teto (`eventos.py:253-258`). O sorteio acontece dentro de `DelayBuffer._update_lags`.

Termos de observação com atraso: `giro_base`, `gravidade`, `juntas`, `vel_juntas` e `mapa`, só no ator (`observacoes.py:55`, `:375-386`). Cada termo tem o seu buffer e o seu lag por env (`mjlab/managers/observation_manager.py:465-474`).

O atraso de comando do atuador é em passos de física (0,005 s); o de observação, em passos da política (0,02 s).

Play: o evento `atraso` roda com teto 0. O termo `v3` não existe no play, e o teto não sobe (`env_cfg.py:197-198`, `:216`).

### 11.7 Ruído de observação

Ruído por passo no ator, uniforme em ± `DR.ruido_obs`: `giro_base` 0,2; `gravidade` 0,05; `juntas` 0,01; `vel_juntas` 1,5 (valor do ruído antes de qualquer escala do termo) (`knobs.py:516`; `observacoes.py:375-380`). O crítico não leva ruído (`observacoes.py:366`, `:375`). No play, `enable_corruption` do ator vale False (`env_cfg.py:113-114`).

---

## 12. Reset do robô e dos envs no play

`make_env_cfg(play=True)` muda, em relação ao treino (`env_cfg.py:80-82`, `:105`, `:113-114`, `:169-181`, `:200-210`, `:216`):

| Item | Treino | Play |
|---|---|---|
| Eventos `pd`, `forca_motor`, `solref` | Existem | Não existem |
| Eventos `empurrao`, `forca_caixa`, `rfi` | Existem | Não existem |
| Corrupção do ator (ruído) | Ligada | Desligada |
| Termo de currículo `v3` | Existe | Não existe |
| Duração do episódio | `Cadeia.episodio_s` = 20,0 s | 10⁹ s |
| Demais eventos de reset e de startup | Existem | Existem |
| Terreno | Gerador da Seção 7 | O mesmo |

No play, os eventos de reset `fatia`, `cena`, `base`, `atrito_pad`, `atrito_laje`, `vies_imu` e `com_caixa` (se registrado) leem o `v3_cur` como ele está. O `play` do mjlab cria o runner registrado na task (`RunnerV3`) e chama `runner.load` (`mjlab/scripts/play.py:203-205`; `tarefa.py:58-64`). O `RunnerV3.load` carrega o estado do currículo do checkpoint quando ele existe (`runner.py:92-110`). O termo `v3` não fecha iterações no play, então o estado fica congelado (`env_cfg.py:216`).

---

## 13. Não determinado pelo código lido

- Número de envs do treino: `train.py` repassa a linha de comando ao `mjlab.scripts.train` e não fixa `scene.num_envs` (`train.py:1-18`).
- Valor de `cfg.seed` do treino. O `v3_gerador` usa `cfg.seed` ou 0 (`curriculo.py:64`). O terreno usa uma semente sorteada na construção (`mjlab/terrains/terrain_generator.py:165-169`). O código lido não mostra como o `np.random` global é semeado.
- Valores da grade `grade_recuo` de `referencias.npz`, usada por `Referencias.recuo` (`alvos.py:58`, `:75-76`). O arquivo de dados não foi lido.
- A regra de prioridade de contato do MuJoCo, que justifica a escolha de geoms em `solref` e as prioridades 1 (pés) e 3 (laje). O código lido só define as prioridades (`cena.py:217`; `knobs.py:105`).
- Se o MuJoCo mantém `qfrc_applied` entre os 4 subpassos sem zerá-lo. O código do mjlab lido não o zera, mas o integrador do MuJoCo não foi lido.
- A orientação inicial padrão da raiz do robô (quaternion padrão) e a velocidade padrão da raiz: `default_root_state` vem de `EntityCfg.InitialStateCfg`, e o código lido só define a posição e as juntas (`cena.py:222-227`).
- O formato exato do mapeamento entre a ordem de `JUNTAS` (`knobs.py:31-41`) e `joint_v_adr` usado por `rfi` (`eventos.py:615`). O código lido assume a mesma ordem.
- Os valores físicos finais (m, graus) dos sorteios unitários do estimador na Seção 5.5: ficam em `estimador.py:passo`, fora do escopo deste documento.
