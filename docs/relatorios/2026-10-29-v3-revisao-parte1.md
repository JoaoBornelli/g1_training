# Revisão da Parte 1 da v3 (D14)

**Veredito: NÃO PRONTO para a fase I. Dois defeitos derrubam o env antes do primeiro passo; quatro defeitos silenciosos mudam o treino.**

| severidade | achados |
|---|---|
| bloqueia o treino | 2 |
| defeito silencioso | 4 |
| regra do plano | 6 |
| menor | 9 |

Escopo: `git diff a02f741..3524b6b -- g1_v3 pilota_v3.py registra_juntas_v3.py` (o último commit da fase D é o
3524b6b). A API foi conferida no mjlab 1.5.3 instalado em `.venv`. Nenhum env foi construído. Um só objeto foi
construído fora do env: cada `ContactSensorCfg` de `cena.sensores()` com `.build()`, para provar o B2.

Suíte (`unittest discover`): 228 testes, 37 erros, 1 skip. Os 37 erros estão todos nos testes da Parte 2
(`test_*_e.py`, commits 0a0968c e b56abfc, depois da fase D). Eles são o vermelho esperado do TDD da fase E.
Só os testes da fase D: 191 ok, 1 skip (o teste de CUDA de `test_balanceador.py:298`).

Conferido sem achado: zero import do `g1_limpo`, do `g1_poc`, do `g1_multitask` e do `g1_training`; módulos puros
sem mjlab e sem mujoco; a ordem dos termos da §6.5 (recompensas, ator, crítico, terminações, eventos, comandos);
máscara por `torch.where` em todo termo; os positivos zeram com `v3_perdeu`; nenhum termo paga positivo no PEGAR
nem no BOTAR (os rastreios dão ≤ 0 porque o `twist` é zero nessas tarefas, `comando.py:414`); alturas contra o
terreno (`altura_pelve`, `altura_joelhos`, `altura_pes`); unidades das cercas (graus, metros, fração); todo invólucro
chama o núcleo testado; todo campo do `knobs` lido nos módulos existe (checagem estática); campos do cfg
(`sim.njmax`, `mujoco.impratio`, `rl_cfg`) existem na dataclass; nenhum tensor de CPU contra CUDA quebra.

---

## Bloqueia o treino

| # | arquivo:linha | o que está errado | cenário de falha | correção | plano |
|---|---|---|---|---|---|
| B1 | `g1_v3/recompensas.py:373`, `:450` | `robo.data.qfrc_actuator[:, 6:]` tira 6 colunas a mais. No mjlab, `qfrc_actuator` já devolve só as 29 juntas (`entity/data.py:415-422`, `_joint_dof_field` com `joint_v_adr`; a junta livre fica fora, `entity/entity.py:1215-1229`). O fatiamento dá (E,23). | Primeiro `reward_manager.compute`, no primeiro `env.step`: `energia` faz (E,23)·(E,29) e `cerca_torque` faz (E,23)/(29,). `RuntimeError`. O treino, o smoke 9 e o 14 caem. Os testes não pegam: o núcleo recebe tensor sintético de 29. | `recompensas.py:373` e `:450`: tirar o `[:, 6:]`. Corrigir a docstring de `:371` ("o mjlab já devolve as 29 juntas"). Teste de invólucro com env falso na lane 6. | D5 (`energia`, `cerca_torque`); §6.3 (`tau (E,29)`) |
| B2 | `g1_v3/cena.py:395-400` | `caixa_faces` tem `global_frame=True` e `reduce="none"`, mas os `fields` não têm `"tangent"`. O mjlab exige `normal` e `tangent` nesse caso (`sensor/contact_sensor.py:232-237`). | `Scene._add_sensors` chama `sensor_cfg.build()` e levanta `ValueError` na construção da cena. Nenhum env nasce: treino, smoke 2 a 15, exportador. Provado: `cena.sensores()[...].build()` levanta o erro. | `cena.py:399`: `fields=("found", "force", "pos", "normal", "tangent")`. Corrigir o texto da D1 no plano (linha 1345-1347). Teste: todo sensor de `cena.sensores()` faz `.build()` (lane 6). | D1 item 7 (`caixa_faces`) |

## Defeito silencioso

| # | arquivo:linha | o que está errado | cenário de falha | correção | plano |
|---|---|---|---|---|---|
| S1 | `g1_v3/cena.py:387-393`; lido em `g1_v3/recompensas.py:484-488` | `nao_pes_chao` usa `mode="subtree", pattern="pelvis", exclude=(".*_ankle_roll_link",)`. O `exclude` filtra a lista de NOMES primários (`contact_sensor.py:536-548`), que aqui é só `["pelvis"]`. O sensor de subárvore do MuJoCo pega todos os descendentes, inclusive os dois pés. | Robô de pé: força líquida subárvore × terreno ≈ m·g ≈ 350 N > 1 N. O `contato_fora` vale 1 em todo passo com pé no chão: −1,0/s constante, o mesmo tamanho do `rastreio_linear`. O custo some só na fase de voo (os dois pés no ar): o termo paga para saltar. Joelho ou mão no chão não somam nada (a contagem fica 1): o canal fica morto para o que ele mede (CLAUDE.md). Não quebra; estraga a corrida. Corrigir antes do 1º run. | `cena.py:389-390`: `primary=ContactMatch(mode="body", pattern=r".*", entity=robo, exclude=(r".*_ankle_roll_link",))`, `fields=("force",)`. Um primário por corpo: `data.force` vira (E,P,3) e `recompensas.py:486` já conta corpos com `> 1 N`, sem mudança. Smoke: de pé, `contato_fora` = 0 (lane 4). | D1 item 7 ("primário os corpos do robô menos os `*_ankle_roll_link`"); D5 `contato_fora` ("nº de corpos fora dos pés") |
| S2 | `g1_v3/eventos.py:110-112`; `g1_v3/comando.py:238-240` | `v3_odo.reinicia` lê `root_link_pos_w`/`quat_w`. Eles vêm de `data.xpos`/`xquat` (`entity/data.py:245-249`). No `_reset_idx` o `reset_root_state_uniform` escreve só o `qpos`; o `sim.forward()` vem depois do reset inteiro. A pose lida é a do fim do episódio anterior. Dois donos fazem o mesmo `reinicia`. | A odometria nasce na pose velha (até ~30–100 m de distância, em outra célula). No passo seguinte o Δ verdadeiro é o teleporte, e o ruído do passo sai com σ = deriva·√‖teleporte‖: no início até ~2 cm e ~1° de rumo; no refino até ~7 cm e ~3°. A pose estimada já começa errada, e a D2b pede "pose estimada = a verdadeira; Δ = 0". Fase D: só o `z_ref` usa, e por diferença; nada muda no mapa. Fase E: `alvo_no_frame_da_base` usa `odo.pos_w`, e o alvo erra sem a spec pedir. | `eventos.py:112`: ler a pose do `qpos`: `q = env.sim.data.qpos[env_ids][:, robo.indexing.free_joint_q_adr]`; `pos, quat = q[:, :3], q[:, 3:7]`. Apagar o `reinicia` duplicado de `comando.py:238-240`: o dono do reset é o evento `filtros` (D7); o dono do `passo` é a cadeia (PD-44). | D2b (teste "depois do `reinicia`, `delta_b()` = 0"); D7 `filtros`; §6.4 `v3_odo` |
| S3 | `g1_v3/curriculo.py:24`, `:65-66` | `v3_celula` nasce com 0 (o `reto` do ANDAR). A §6.4 diz −1 = sem célula. O `TwistV3.__init__` copia o 0 para `seg_cel` (`comando.py:313`). | No 1º `env.reset()`, `TwistV3.reset` fecha um segmento fantasma: `falha = ~time_outs` = True e `_agora` 0 ≥ âncora 0, então `reg` = True. Registra E falhas (4 096) no `reto`. A 1ª `fecha_iteracao` põe `s_reto` = 0 pela regra "primeira medida, s = x" e `n_medidas` = 1. A 1ª medida real entra pela EMA (α 0,05) desde 0, e não como `s = x`. O `reto` leva ~20–40 its a mais para o portão; a fatia B e as células seguintes abrem depois. | `curriculo.py`: tirar `v3_celula` de `_LONGOS` e criar com `_cria(env, "v3_celula", (), torch.long, -1)`, como o `v3_alvo_idx`. | §6.4 `v3_celula`; PD-13 (EMA) |
| S4 | `g1_v3/comando.py:334-341` | A regra da âncora vale também para a queda: `reg = t_fecho ≥ t_ancora`, e `falha` só zera o sucesso. Um segmento que acaba em queda antes da âncora (1,5 s depois da abertura ou do último empurrão) não registra. | Queda típica: logo depois da troca de comando, ou logo depois do empurrão (a âncora vai para +1,5 s do empurrão). Essas quedas somem da taxa de sucesso. Uma célula passa no portão 0,75 com quedas não contadas; o nível sobe cedo. O empurrão que derruba nunca conta como falha. | `comando.py`, depois de `:337`: `reg = reg \| falha`. Depende do S3 (sem o −1, a mudança registraria o fantasma do 1º reset). O dono confirma a leitura: a D3 manda a queda registrar falha; a PD-15 tira só a frenagem. | D3 item 4 ("um segmento aberto numa terminação que não é `time_out` registra falha"); S §11.3 ("sem queda") |

## Regra do plano

| # | arquivo:linha | o que está errado | cenário de falha | correção | plano |
|---|---|---|---|---|---|
| R1 | `g1_v3/smoke.py:236-246` | A seção 6 teleporta a base 1,0 m de uma vez e fica 50 passos parada. A D12 pede a base movida 1,0 m "por escrita da pose em 50 passos". | O teleporte único não testa o transporte por célula com resto a 50 Hz, o defeito do mapa congelado de 29/10 (§20). O check passa mesmo com a regressão. | Laço de 50 passos: escreve +0,02 m na pose, `env.sim.forward()`, `env.step(zeros)`. | D12 item 6; §20 (29/10, transporte do mapa) |
| R2 | `g1_v3/smoke.py:333-349` | A seção 10 escreve a pose e dá um passo. A `queda` por altura lê o `altura_pelve` do `sim.sense()` anterior: o raycast só atualiza no `sense` (`raycast_sensor.py:691`), e o `step` calcula a terminação antes do `sense`. | O check "base 0,10 m" lê a altura do reset (0,8 m) e não cai pela altura. Ele falha, ou passa pela explosão das pernas enterradas 0,7 m no chão. O dono recebe falha falsa ou acerto pelo motivo errado. | Depois de escrever a pose: `env.sim.forward()`, `env.sim.sense()` e `terminacoes.queda(env, TODAS)` direto, sem `step`. | D12 item 10 |
| R3 | `g1_v3/observacoes.py:253-255`, `:221-231`, `:157-160` | Funções repetidas: `tarefa_critico` é igual a `um_de_seis` (`:109-111`); `niveis` refaz a tabela `Curriculo._max` (`balanceador.py:55-56`); `_rumo` é o `estimador._yaw`. | Duas cópias divergem: mudar os níveis no `balanceador` deixa o `niveis` do crítico errado em silêncio. | `critico["tarefa_critico"] = termo("tarefa_critico", um_de_seis)` e apagar a função (−4 linhas). `niveis`: ler o máximo do `Curriculo` (tornar `_max` público, sem função nova). `_rumo`: opcional. | CLAUDE.md (minimização; função repetida) |
| R4 | `pilota_v3.py:628` | O driver não mostra o `alvos.detecta_fim` como status. A D11 pede. O `alvos` já está carregado (`pilota_v3.py:53`). | Na fase D o PEGAR não treina, então nada quebra. Na fase E o dono não vê o aviso do detector no driver. | Pôr `detecta_fim` no laço (status na linha de `\r`), ou o plano registra que vai para a fase E. O estimador da caixa depende da fase E (sem `EstimadorCaixa`). | D11 (`pilota_v3.py`: "o detector de fim como status") |
| R5 | `g1_v3/comando.py:292`, `:279-280` | `self.metrics.clear()` tira as métricas do fabricante. A §6.7 pede "mais os do fabricante". Os acumuladores internos `soma_erro_marcha` e `soma_cmd_marcha` vão ao log como métrica. | Log com dois campos de soma corrida sem sentido por episódio; faltam `error_vel_xy` e `error_vel_yaw`. | Manter as métricas do fabricante (chamar o `_update_metrics` do pai ou recalcular os dois erros) e guardar as duas somas como atributo, fora de `metrics`. | §6.7 (Twist) |
| R6 | `g1_v3/cena.py:56-58`, `:108`, `:192`; `g1_v3/observacoes.py:50` | Números fora do `knobs`: geometria do pad (`_PAD_HALF`, `_PALM_DZ`, `_PALM_X`), `condim` 3 da caixa, e a sentinela 10⁹ repetida em dois módulos. | Nenhuma falha hoje. A §1.5 diz "todo número mora no `knobs.py`". | Decisão do dono: mover para `Cena` (`pad_meia`, `pad_dz`, `pad_x`, `caixa_condim`) ou registrar na §1.5 a exceção "geometria do modelo e sentinelas". Unificar o 10⁹ numa constante só. | §1.5 |

## Menor

| # | arquivo:linha | o que está errado | cenário de falha | correção | plano |
|---|---|---|---|---|---|
| M1 | `g1_v3/env_cfg.py:8` | A docstring diz que a única constante são as tuplas de tarefa. Há `_EPISODIO_PLAY_S` (`:52`) e `_GEOMS_DOS_PES` (`:49`). | Texto falso. | Corrigir a frase. Os dois valores são do plano (D10) e do modelo; ficam. | D10 |
| M2 | `g1_v3/container/entrada.sh:72`; `pilota_v3.py:142-144`, `:344`; `g1_v3/smoke.py:75-77`, `:143`, `:244`, `:286` | Marcas `⚠ DÚVIDA` já respondidas (ver a seção "DÚVIDAS no código"). `smoke._parado` não faz nada útil. | Leitor perde tempo; a docstring do `Percepcao` diz que o exportador não a constrói, e ele constrói (`exporta_cena.py:541`). | Tirar as marcas; apagar `_parado` (o robô fica parado pela ação zero). | D13 item 3; D11 (29/10) |
| M3 | `g1_v3/smoke.py:160-165` | A seção 4 reordena por nome e não confere `robo.joint_names == list(JUNTAS)`. Os índices fixos de `recompensas.py:33-36` (`_BRACOS`, `_YAW_E`...) supõem a ordem de `JUNTAS`. | Só o exportador confere a ordem (`exporta_cena.py:216-219`). O treino não. | Somar `check("ordem das juntas = JUNTAS", robo.joint_names == list(JUNTAS))`. | I §2.2; §6.2 `JUNTAS` |
| M4 | `g1_v3/testes/test_algoritmo.py:5` | A docstring diz que o `std()` do torch é "sem correção de Bessel". O padrão é COM correção (n−1). | Texto falso; o teste passa igual. | Corrigir a frase. | D9 |
| M5 | `g1_v3/cena.py:285-300` | Na fase D a caixa e a laje ficam no `init_state`, no mundo, sem a origem do env: em (0,32; 0) e (0,50; 0). Esse ponto é o centro da célula (6, 4), `rampa_inv` linha 6, cuja origem é (0; 0). | Ep. de terreno do ANDAR nessa célula (nível de terreno ≥ 6, ~1% dos ep. de terreno) nasce com a laje na altura do peito e a ±0,5 m: contato no nascimento; o lidar vê a laje. | Antecipar o `estaciona_cena` (E1) para a fase D, ou somar 20 m ao z do `init_state` das duas entidades até a fase E. | §2.2 (`afasta_cena` → `estaciona_cena`) |
| M6 | `g1_v3/comando.py:60` | `sorteia_velocidade` cria o tensor de saída na CPU; os sorteios saem no device do gerador (GPU). Funciona por cópia entre devices, com sincronização a cada reamostra. | Custo; foge da convenção da §6.3. | `v = torch.zeros(n, 3, device=None if gerador is None else gerador.device)`. | §6.3 (convenções, 29/10) |
| M7 | `g1_v3/recompensas.py:311`; `g1_v3/metricas.py:173` | `c_w = quat_apply(root_link_quat_w, c_b)` gira pelo quatérnion inteiro. O comentário diz "girado pelo rumo". Com a base inclinada, ĉ_w ganha componente z e o `L` soma o desnível do pé. | Base a 10° de pitch numa escada de 0,20 m: +3,5 cm no `L`. | `quat_apply_yaw` (`mjlab/utils/lab_api/math.py:673`) nos dois arquivos. | PD-16 |
| M8 | `g1_v3/recompensas.py:27-28`; `g1_v3/terminacoes.py:85`; `g1_v3/observacoes.py:39`; `g1_v3/curriculo.py:18` | `make_env_cfg(k)` honra só parte do `k`: pesos, `comando`, `cadeia`, `terreno`, `percepcao.ligada`. Os σ, limites, ruído do lidar, terminação e balanceador saem do `ATIVO` do módulo. | Hoje só a SemPercepcao passa um `k` diferente, e funciona. Um script da Parte 0 ou um teste que passe outro `Knobs` muda o peso e não o resto, em silêncio. | Registrar no docstring de `make_env_cfg` que só esses campos valem, ou passar as seções por `params`. | §6.3 (SemPercepcao por `make_env_cfg(k)`) |
| M9 | `g1_v3/eventos.py:54` | Todo ep. de plano nasce na célula (linha 0, coluna 0), no canto da grade: 24 m até a beira da borda em −x e em −y. | No nível 5 (1,5 m/s, 20 s), um ep. raro anda para fora do mundo e cai: queda falsa no `reto`. Só o treino mostra a taxa. | Opcional: sortear a linha do plano em 0..R−1 (a coluna 0 é plana em toda linha); medir `Episode_Termination/queda` no ANDAR plano. | D7; PD-20 |

---

## Pendências

Fonte: `scratchpad/d14_pendencias.md`.

| pendência | veredito |
|---|---|
| A5/D1: pads fora do `capsulas.npz`; o lidar vê o grupo 3 | **Resolvida.** O `capsulas.npz` guarda a cápsula da mão (`*_wrist_yaw_link`, 0,07→0,15 m, raio 0,035; a cena apaga o geom, a tabela não). O canto mais longe do pad fica a 5,1 cm do eixo, dentro de raio + margem (6,5 cm). O `raio_min` 0,85 m é a segunda guarda. |
| D1: `_PRIORIDADE_LAJE` e `_BORDA_TERRENO` no módulo | **Resolvida.** `Cena.prioridade_laje` (`cena.py:121`) e `Terreno.borda` (`cena.py:268`). |
| D1: `nao_pes_chao` com `exclude` no subtree | **Defeito: S1.** O `exclude` não filtra corpos dentro da subárvore. |
| D1: `caixa_meia_faixa` como meia aresta nominal; `condim` 3 literal; marcador da face | **Aceitável na fase D.** O `tamanho_caixa` da fase E sobrescreve a meia aresta; o marcador é do REORIENTAR, fora do escopo. O `condim` 3 literal fere a §1.5: R6. |
| D1: sensores de altura com `ray_alignment="yaw"` | **Resolvida.** Certo: os raios descem na vertical com o corpo inclinado; com `"base"` a pelve inclinada mediria a altura na oblíqua. |
| C1: custo do filtro 558 ms em CPU | **Só o I5 mostra.** Nota: `filtra_corpo` monta (E,205,33,3); a 4 096 envs são ~330 MB por tensor intermediário. |
| B5 item 15: custo do aperto | **Fora da Parte 1.** É o peso do `aperto` da fase H; nada a corrigir agora. |
| D9: docstring do `test_algoritmo` (Bessel) | **Defeito de texto: M4.** |
| D7: evento `atraso` por atributos privados | **Resolvida pela leitura.** `DelayBuffer._sample_lags` lê `self.max_lag` a cada sorteio; a capacidade fica no `CircularBuffer` do build. O buffer do atuador é fundido e apelidado em cada atuador (`builtin_group.py:158-172`); o laço por `id` está certo. O smoke 8 confirma. |
| D7: `reinicia_filtros` lê `root_link_pos_w` depois do reset da base | **Defeito: S2.** A pose lida é a do fim do episódio anterior. |
| D2: `sensor._local_directions` privado | **Resolvida pela leitura.** `prepare_rays` lê `self._local_directions` a cada `sense`, fora do grafo CUDA (`raycast_sensor.py:691-731`); o `copy_` mantém o storage. O índice do padrão bate: o passo N lê `direcoes(N)`, que o passo N−1 copiou. O smoke 5 confirma. |
| D2/D3/D5/D7: invólucros sem teste de runtime | **Lidos contra o mjlab 1.5.3.** Saíram B1, B2, S1 a S4. O resto só o smoke mostra. |
| D5: `root_body_id` global | **Resolvida.** `root_link_pose_w` indexa `data.xpos[:, root_body_id]` (`entity/data.py:247`): o id é global. |
| D2: `_SEM_SORTEIO`, `_ESC_DEGRAU`, `_ESC_MASSA_KG`, /100 N no módulo | **Aceitável.** São constantes de contrato da §6.5 (/5, /5 kg, /100 N) e a sentinela da D1/D2, e não números de treino. O 10⁹ repetido em `cena.py:192`: R6. |
| cena: `_PAD_HALF`, `_PALM_DZ`, `_PALM_X`, `_GRUPO_PAD` | **Decisão do dono: R6.** Recomendo a exceção "geometria do modelo" na §1.5. |
| D13: `entrada.sh` sem `--agent.experiment-name` | **Resolvida.** A D10 fechou com `EXPERIMENT = "g1_v3"` (`tarefa.py:27`). Tirar a marca: M2. |
| D11a: `constroi_env` zera o atraso da observação por atributo privado | **Resolvida pela leitura.** Mesmo mecanismo do evento `atraso`; com `max_lag` 0 o sorteio dá 0. A paridade confirma. |
| D11a fix: versão por git, kd por `biasprm[2]`, mapa impresso, teste do `monta_observacao` | **Resolvida.** Commit 9042b69: `exporta_cena.py:378-386` (versão), `:399-401` (kd), `:494-507` e `:564-566` (mapa); `test_pilota.py:120-180`. |
| D10: `_EPISODIO_PLAY_S` e `_GEOMS_DOS_PES` no `env_cfg` | **Aceitável.** O 10⁹ é da D10 e os nomes são do modelo. A docstring mente: M1. |
| D12 S6: "parado" | **Resolvida.** É a ação zero que deixa o robô parado; `TwistV3._update_command` é `pass` (`comando.py:440-441`). O `_parado` sai: M2. |
| D12 S4: alvo físico com o viés | **Resolvida.** O mjlab subtrai o `encoder_bias` do alvo (`envs/mdp/actions/actions.py:222-223`). O "q_de_pé" da D12 é o alvo nominal; a forma do smoke está certa. Ajustar o texto da D12. |
| D12 S6: teleporte e odometria | **Resolvida.** `v3_odo.passo` lê a pose verdadeira depois do `forward` (`comando.py:254`); o teleporte entra como um Δ de 1 m num passo. A seção ainda foge da D12: R1. |
| D12 S10: valor da `queda` depois do reset no mesmo passo | **Resolvida, com ressalva.** `get_term("queda")` devolve o valor calculado antes do reset (o `reset` não zera `_term_dones`, `termination_manager.py:88-100`). A altura lê o `sense` anterior: R2. |
| D12 S12: assinatura do `RslRlVecEnvWrapper` | **Resolvida.** `RslRlVecEnvWrapper(env, clip_actions=None)` (`rl/vecenv_wrapper.py:10-14`); o runner nasce como no `scripts/train.py:150-167`. |

## DÚVIDAS no código

| marca | veredito |
|---|---|
| `g1_v3/env_cfg.py:101` (`ranges` e `resampling_time_range`) | **O plano responde.** `resampling_time_range` = `Comando.reamostra_s` (3; 8), §6.2 linha 418, S §7. O `ranges` é o envelope do `knobs` (`frente_tetos[-1]`, `tras_lado_teto_max`, `giro_tetos[-1]`); o exportador o lê como `envelope_treino` (`exporta_cena.py:248-251`); o sorteio o ignora. Tirar a marca. |
| `g1_v3/env_cfg.py:118` (tarefas da `queda`) | **O plano responde.** D6 (linhas 1542-1545): a queda vale nas quatro tarefas, com regra diferente por par; §6.6: todo termo de terminação recebe `tarefas`. Tirar a marca. |
| `g1_v3/env_cfg.py:152` (currículo no play) | **Decisão do dono.** A D10 não fala do currículo no play. O motivo escrito está errado: os buffers nascem no `CadeiaCmd.__init__` (`comando.py:195`), e não no termo. No play com checkpoint (`RunnerV3.load`), o termo segue fechando iterações e abre células e níveis durante o play. Recomendo tirar o `v3` no play: o estado carregado fica congelado. |
| `g1_v3/container/entrada.sh:72` | **Resolvida.** `tarefa.py:27`; D13 item 3. Tirar. |
| `g1_v3/smoke.py:75` | **Resolvida.** `_update_command` é `pass`; o robô fica parado pela ação zero. |
| `g1_v3/smoke.py:143` (layout do histórico) | **Resolvida.** O mjlab documenta a ordem por termo, tempo-maior, mais antigo primeiro (`observation_manager.py:60-65`); o contrato do exportador diz o mesmo (`exporta_cena.py:99-101`). O `reshape(E, hist, canais)` está certo. |
| `g1_v3/smoke.py:162` (ordem das juntas) | **O plano responde.** §6.2 `JUNTAS`: ordem do SDK = ordem do MJCF (I §2.2). Somar o check explícito: M3. |
| `g1_v3/smoke.py:170` (alvo − viés) | **Resolvida.** Ver a pendência S4. |
| `g1_v3/smoke.py:244` (teleporte) | **Resolvida.** Ver a pendência S6; a seção ainda foge da D12 (R1). |
| `g1_v3/smoke.py:286` (atributos privados) | **Resolvida para o mjlab 1.5.3**, que o `entrada.sh:38` pina. Manter como nota `⚠`, sem "DÚVIDA". |
| `g1_v3/smoke.py:332` (tupla da `queda`) | **O plano responde.** Igual a `env_cfg.py:118`. |
| `registra_juntas_v3.py:104` (limpar laje e caixa) | **Decisão do dono.** Recomendo manter: espelha o `estaciona_cena` da fase E (no ANDAR a laje e a caixa saem; §2.2). |
| `registra_juntas_v3.py:303` (`alvo_x/y/z`) | **O plano responde em parte.** D11: "as colunas de hoje mais `tarefa`, ...". A coluna fica; NaN na fase D; a fase E preenche pelo `alvos.caixa_alvo_w`. |
| `registra_juntas_v3.py:307` (`aviso_fim`, `idade_caixa`) | **O plano pede as duas colunas (D11).** `idade_caixa` depende do estimador da fase E. `aviso_fim` pode entrar já pelo `alvos.detecta_fim` (fase B): mesma correção do R4. |
| `pilota_v3.py:79` (REORIENTAR e LEVANTAR sem tecla) | **O plano responde.** §6.3, linha 754: os dois são canais reservados do one-hot, sem env. Sem tecla está certo. Tirar a marca. |
| `pilota_v3.py:142` (assinatura `Percepcao(m, c)`) | **Resolvida.** §20 (29/10, D11) e `exporta_cena.py:541`, que já constrói `pilota.Percepcao(m, dados)`. A frase "o exportador ainda não constrói" está velha. Tirar. |
| `pilota_v3.py:344` (ELU) | **Resolvida.** `tarefa.rl_cfg` parte do `unitree_g1_ppo_runner_cfg`: `activation="elu"`, `hidden_dims=(512, 256, 128)` (`mjlab/tasks/velocity/config/g1/rl_cfg.py:13-16`); a última camada do MLP é linear. Tirar. |
| `pilota_v3.py:628` (estimador e detector fora do laço) | **O plano pede os dois (D11).** O estimador depende da fase E. O detector pode entrar já: R4. |

---

## Lanes de correção

Arquivos disjuntos entre as lanes. Contagem de linhas antes → depois.

| lane | arquivos | correções | contagem |
|---|---|---|---|
| 1 — bloqueios e contato | `g1_v3/cena.py`, `g1_v3/recompensas.py`, `g1_v3/metricas.py` | B1, B2, S1, M7; M5 se o dono escolher o z de 20 m; R6 se o dono escolher mover os números | cena 414 → 414; recompensas 494 → 494; metricas 236 → 236 |
| 2 — currículo, comando e odometria | `g1_v3/curriculo.py`, `g1_v3/comando.py`, `g1_v3/eventos.py` | S2 (pose do `qpos`; apagar o `reinicia` duplicado), S3, S4, R5, M6; M9 opcional | curriculo 113 → 114; comando 441 → ~440; eventos 139 → ~141 |
| 3 — observações | `g1_v3/observacoes.py`, `g1_v3/balanceador.py` | R3 (`tarefa_critico` → `um_de_seis`; `niveis` lê o máximo do `Curriculo`, com `_max` público) | observacoes 319 → ~310; balanceador ±0 |
| 4 — smoke | `g1_v3/smoke.py` | R1, R2, M3; check novo: de pé no plano, `contato_fora` = 0; tirar `_parado` e as marcas resolvidas | 474 → ~485 (fora da meta) |
| 5 — textos e marcas | `g1_v3/env_cfg.py`, `g1_v3/container/entrada.sh`, `pilota_v3.py`, `registra_juntas_v3.py`, `g1_v3/testes/test_algoritmo.py` | M1, M2, M4, M8 (docstring); marcas resolvidas; R4 e `aviso_fim` se o dono não adiar para a E; DÚVIDA `env_cfg.py:152` depois da decisão do dono | env_cfg 157 → ~152; pilota 677 → ~690 com o detector |
| 6 — testes | `g1_v3/testes/test_cfg.py`, `g1_v3/testes/test_recompensas.py`, `g1_v3/testes/test_curriculo.py` | todo sensor de `cena.sensores()` faz `.build()` (pega o B2); `nao_pes_chao` com primário por corpo (S1); `energia` e `cerca_torque` com env falso e `qfrc_actuator` (E,29) devolvem (E,) (pega o B1); `garante_v3` cria `v3_celula` = −1 (S3) | +~35 linhas de teste |
| 7 — plano (orquestrador) | `docs/planos/2026-10-14-implementacao-g1-v3.md` | D1 item 7: `caixa_faces` com `"tangent"`; `nao_pes_chao` com primário por corpo. D12 item 4: "alvo = q_de_pé − viés". D3 item 4: a queda registra falha antes da âncora (S4). §1.5: exceção da R6, se o dono aceitar. §20: linha desta revisão. | — |

Ordem: as lanes 1 a 6 rodam em paralelo. A lane 6 escreve os testes que falham antes das lanes 1 e 2 (TDD). O
smoke da fase I só vale depois das lanes 1, 2 e 4.
