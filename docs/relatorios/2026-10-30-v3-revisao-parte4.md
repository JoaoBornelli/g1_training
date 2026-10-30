# Revisão da Parte 4 da v3 (H8)

**Veredito: NADA DERRUBA O ENV E AS 24 DECISÕES DO CASO 7 ESTÃO NO CÓDIGO COMO O CONTRATO DIZ; A ODOMETRIA ANDA DE NOVO. MAS O APERTO É UM CANAL QUASE MORTO E A ALAVANCA DO REGISTRO (O `F_ref`) RENDE NO MÁXIMO 1,69×; A ESCADA (2) DA S §10 NÃO CONTA A SACUDIDA; A s37 FALHA SEMPRE E A s34 E A MÁSCARA DA s36 PASSAM SEM TESTAR.**

| severidade | achados |
|---|---|
| bloqueia o treino | 0 |
| defeito silencioso | 0 |
| regra do plano | 4 |
| menor (texto, smoke, limpeza) | 12 |

Escopo: o pacote inteiro, com foco em `git diff df32c29..HEAD` (fase H, nove commits de `0e480c5` a `ad107d0`).
Código: 14 arquivos, +533 −144 (sem testes; o smoke é +247 −1). Testes: 18 arquivos, +1 435 −25. Li inteiros os diffs
da fase H e, no código de hoje: `balanceador.dr()`, `recompensas` (`sacudida`, `aperto`, `forcas_por_face`,
`por_degrau`), `terminacoes` (`n_batente`, `batente`), `eventos` (DR do refino), `estimador.EstimadorCaixa` e
`Odometria`, `comando` (`compute`, `_abre_trecho`, `_abre_botar`, `_publica`, `_resample`), `observacoes`
(`_magnitude`, `forca_caixa`), `metricas` (`aceleracao_da_caixa`, `forca_robo_caixa`), `env_cfg`, `cena` (`_atuadores`,
`robot_cfg`, sensores), `runner`, `curriculo` e o smoke 34 a 39. API do mjlab 1.5.3 no `.venv`: `envs/mdp/dr/`
(`_types`, `_core`, `actuator`, `body`), `envs/mdp/events.py` (`apply_body_impulse`), `managers/event_manager.py`,
`reward_manager.py`, `metrics_manager.py`, `manager_base.py`, `scene_entity_config.py`, `entity/entity.py`, `data.py`,
`sensor/contact_sensor.py`, `sim/sim_data.py`, `envs/manager_based_rl_env.py`; `rsl_rl/runners/on_policy_runner.py` e
`utils/logger.py`; `mujoco_warp/_src/types.py` (`Contact`). Nenhum env foi construído; nada de smoke, simulador ou treino.
A varredura de remoções está em
`/tmp/claude-1002/-home-joaobornelli-Documents-g1-training/5d540910-028a-4291-8d9f-065788e279ae/scratchpad/h8/remocoes.txt`
(o script ao lado, `remocoes.py`).

Suíte (`unittest discover`): 504 testes, todos ok, 1 skip. Imports do topo de 15 módulos: nenhum sem uso (conferido
por AST; o `pyflakes` não está no `.venv`).

Conferido sem achado:
- Ordem do `env.step` (`manager_based_rl_env.py:419-464`) para a fase H: física (4 subpassos); terminações (`batente`);
  recompensas (`sacudida`, `aperto`, com o `cvel` um subpasso atrasado, igual em toda chamada); métricas
  (`aceleracao_da_caixa`); reset (`reward_manager.reset` e `metrics_manager.reset` zeram a `v_ant` das duas instâncias;
  `event_manager.reset` chama o `reset` da `forca_caixa`, que zera o wrench ativo); `sim.forward()`; `command_manager`
  (a cadeia dá o `v3_odo.passo` e grava o viés do alvo do BOTAR); eventos `step` (`forca_caixa`, `rfi`, escritos para a
  física do passo seguinte, os 4 subpassos); `sense`; observação (`_magnitude` pelo degrau global). Num env que reseta, o
  `reinicia_filtros` põe a `_pos_v` da odometria na pose do `qpos` e o `passo` do mesmo `compute` dá Δ = 0.
- `pd` e `forca_motor`: o `scale` do mjlab lê o default (`get_default_field`, `uses_defaults=True`): nada acumula entre
  resets. Os 9 grupos são `BuiltinPositionActuator` (o tipo que o `pd_gains` aceita); `sort_actuators` é False
  (`entity.py:89`, nenhum `g1_v3` o liga), e o `_actuators` sai na ordem da `Contrato.pd` (`cena.py:185`): grupos 7 e 8
  são os braços.
- `solref`: as `geom_ids` do mjlab são `torch.int` (`entity.py:1199`), o mesmo dtype do `env_ids`: o `meshgrid` não
  estoura (com dtypes diferentes ele recusa; conferido). Nenhuma geom da cena define `solref` (o `g1.xml`, os pads, a
  caixa, a laje e o terreno ficam no 0,02 do MuJoCo): no degrau 0 a escrita de 0,02 não muda nada.
- `com_caixa`: o `combine` recebe `base` (n, 1, 3) e o sorteio do `_dist`; `a[:, None, None]` casa. O `_dist` recebe
  `lo` e `hi` como tensores (`_core.py:297-298`): o `.to(lo.dtype)` vale.
- `forca_caixa`: o pai lê `asset_cfg.body_ids` depois do `resolve` (`manager_base.py:142-147`); com o corpo único
  resolvido para `slice(None)`, `_num_bodies` é o do asset. O `reset` do pai entra no `event_manager.reset` (classe de
  modo `step`).
- `rfi`: nada no mjlab escreve o `qfrc_applied` (só o viewer nativo, fora do treino); `joint_v_adr` exclui a junta livre.
- Sensor de contato: fatia vazia sai com força 0 (sensor do MuJoCo), não NaN; a normal vai do primário (caixa) ao robô
  (`contact_sensor.py:114-115`). O `custo_aperto` usa `min(F_esq, F_dir)`: trocar esquerda e direita não muda o custo.
- `runner._embrulha_log`: o rsl_rl chama `self.logger.log(...)` só por nome (`on_policy_runner.py:115-125`);
  `action_std` é o `output_std`; `target_names` existe no `JointPositionAction` (`actions.py:150`).
- NaN fora da máscara: os cinco termos novos devolvem 0 ou False por `torch.where`; nenhum multiplica pela máscara.
  Nenhum buffer `v3_*` novo no env (a `v_ant`, os vieses e a rajada moram nos objetos). Módulos puros: o `estimador` ganha
  só `statistics` (stdlib) e o `knobs` só `re`; nenhum importa outro do pacote.

---

## 1. As 24 decisões do caso 7 no código

| id | contrato | código | veredito |
|---|---|---|---|
| P1 | uma fórmula do degrau | `balanceador.py:39` `no_degrau`; `recompensas.py:411` `por_degrau` é o invólucro; os três campos mortos do `Refino` saíram | ok |
| P2 | custos com peso negativo | `knobs.py:293-294` (−0,05; −0,02); `env_cfg.py:126-129` passa o knob cru; os termos devolvem ≥ 0 | ok |
| P3, P4, P15 | subclasse do impulso, `step`, máscara (CARREGAR & pegou & ~perdeu), trava "subida" sem escala | `eventos.py:552-570`; `env_cfg.py:200-203`; `knobs.py` `forca_caixa_N` (−11,5; 11,5) | ok |
| P5 | degrau global = o menor | `balanceador.py:254-256` | ok |
| P7 | crítico vê a força total /100 | `observacoes.py:311-319` sobre `metricas.forca_robo_caixa` (soma dos módulos) | ok |
| P8, P10 | norma (5,04; 31,56), quadrático | `knobs.py:297`; `recompensas.py:325-328` | ok |
| P9 | `v_ant` da classe, zerada no reset | `recompensas.py:795-818` | ok |
| P11 | máscara pelo índice P3, P4, B1, B2, sem `~apoio` | `recompensas.py:815`, `:829` (`_CAIXA_ALVO`) | ok; a métrica repete o conjunto (R3) |
| P12 | `maxforce`; face pela normal | `cena.py:408`; `recompensas.py:331-342` | ok |
| P13, P14 | 29 juntas, `>` estrito, `potencial.curso` | `terminacoes.py:142-171`; `potencial.py:78-84` | ok |
| P16 | `dr.*` pelo `v3_gerador` | `eventos.py:484-488`; a `forca_caixa` é a exceção documentada | ok |
| P17 | `solref` nos 14 pés, 4 pads, laje e caixa, um u por env | `eventos.py:516-533`; `env_cfg.py:170-173` | ok |
| P18 | `actuator_ids` dos grupos: braços [7, 8], pernas [0..6] | `env_cfg.py:146-147`, `:166-168` | ok (o nome que o mjlab resolve é cosmético, M10) |
| P19 | `body_com_offset` × meia aresta, só com a chave | `eventos.py:536-549`; `env_cfg.py:175-176` | ok no treino; o `dr()` lê a chave do `ATIVO`, e não do `k` (R4) |
| P20 | `knobs.TAU_MAX`, sem ciclo | `knobs.py:480`; lido por `recompensas.py:690` e `eventos.py:582` | ok |
| P21 | lognormal (mediana, p90), 1° por leitura, vieses U[±], rajada | `estimador.py:269-317`; `observacoes.py:73-87` | ok |
| P22 | viés do alvo do BOTAR no `v3_alvo_w`, degrau global | `comando.py:490-497` | ok |
| P23 | métrica e as três chaves do log | `metricas.py:425-457`; `curriculo.py:129` | ok |
| P24 | o play sem as cinco DR | `env_cfg.py:164-174` e `:192-203` | ok |

O degrau global contra o da tarefa: `por_degrau` (sacudida, aperto, `incl_lim`, pouso, `contato_laje`, `chegada`) e o
`batente` leem `v3_degrau`, que o `_publica` grava com `cur.degrau(tarefa)` na abertura do trecho (`comando.py:450-461`);
a DR do refino (`dr()`), o estimador (`observacoes.py:185`), a deriva da odometria (`comando.py:555-558`) e o viés do
BOTAR (`comando.py:492`) leem `degrau_global()`. Confere com a §6.9.

## 2. A odometria e o fce6a42

O `v3_odo.passo` voltou ao `compute` (`comando.py:555-558`), depois do `_abre_trecho` e antes do `_grava_chegada`, com
`deriva_do_degrau` no degrau global. O resto do que o `fce6a42` tirou tem registro na F9: o `_espera` sem o `self.pega`
e o `self.pega` no `_abre_p0` (P6); o latch do aviso (P5); o ponto de apoio no fecho do B2 (P1, S2); o `afasta` com o
`palma_face` (S3); o `_registra` do BOTAR com `falhou` = não `time_out` (P7). A chamada da odometria é a única remoção
sem registro do commit (caso 8).

## 3. Varredura de remoções (`dacbd1e..HEAD`)

127 commits tocam `g1_v3/`, `pilota_v3.py` e `registra_juntas_v3.py`; 77 removem linhas que não são comentário nem
docstring: 976 linhas, 908 fora de `parte0/saida/` (os textos de saída dos itens). Cada remoção de comportamento (chamada,
escrita de buffer, termo, guarda, check) foi conferida contra a mensagem do commit, a §20 e o registro de problemas.
Todas têm justificativa, menos uma: a chamada `env.v3_odo.passo(...)` do `fce6a42` (já é o caso 8, resolvido em
`ce4dc30` e na H5). As que mais pareciam remoção e são troca:

| commit | o que saiu | para onde foi |
|---|---|---|
| `5300d26` | `cfg.delay_update_period = _SEM_SORTEIO` no `observacoes` | `observacoes.py:348` e `cena.py:192`, com `knobs.NUNCA` |
| `1301cb6` | `v3_odo.reinicia` no `comando` e no `eventos:112` | `eventos.reinicia_filtros` (`eventos.py:214`), depois da base (D14) |
| `95d5ed5`, `683ecfd` | o zero do alvo só no CARREGAR | `observacoes.py:191`, com o latch do BOTAR e o índice −1 (F9 R5) |
| `4c326ee` | checks da s28 ("0,5 s + atraso") e das palmas | checks pelo aviso e pelo `caixa_robo` (F9 M1) |
| `8f4409d` | `_media_dos_com_evento` e os retornos das métricas de evento | `_loga_episodios` no `reset` (caso 1) |
| `90f994b` | o check da terminação `caixa_perdida` na s32 | o `v3_perdeu` no passo (G7 M1) |
| `0e480c5`, `2eaf974`, `ecd2268`, `e16bfb2` | `rfi` e `atraso_max` da E, sinais, `Refino`, `_z_ori`, `ori_persistente`, `sigma_ori_graus`, `_MAGNITUDE_INICIO`, `reduce="none"` | as decisões P1, P2, P3, P12, P21 |

## 4. A derivada no ponto de operação (CLAUDE.md: canal morto)

Valor por segundo = peso · (d/5) · custo (o mjlab multiplica por dt). Sacudida no p50 do 23250 (custo 2: um de cada
norma); aperto com 117,5 N por face e `F_ref` 49 N.

| degrau d | sacudida /s | ∂/∂a (por m/s², por s) | ∂/∂α (por rad/s², por s) | aperto /s | ∂/∂F (por N, por s) | afrouxar 117,5 → 49 N rende /s |
|---|---|---|---|---|---|---|
| 1 | −0,020 | −3,97e−3 | −6,34e−4 | −0,0036 | −7,42e−6 | 0,0011 |
| 2 | −0,040 | −7,94e−3 | −1,27e−3 | −0,0073 | −1,48e−5 | 0,0022 |
| 3 | −0,060 | −1,19e−2 | −1,90e−3 | −0,0109 | −2,23e−5 | 0,0033 |
| 4 | −0,080 | −1,59e−2 | −2,54e−3 | −0,0146 | −2,97e−5 | 0,0044 |
| 5 | −0,100 | −1,98e−2 | −3,17e−3 | −0,0182 | −3,71e−5 | 0,0055 |

- Sacudida: canal vivo. 10% menos de aceleração linear no degrau 5 rende 0,0095/s (o quadrático). Um impacto de 1 m/s
  num passo vale custo 98 (0,25 m/s: 6,2): o termo cobra a chegada na laje no B1 e no B2, o que o E §3 pede ("sem
  impacto"). Na queda da caixa da mão a perda pela velocidade relativa (1,2 m/s, ~6 passos de queda livre) tira o
  índice do P4 antes do choque no chão: o choque não é cobrado.
- Aperto: o alerta do caso 7 confere (3,7e−5 por N por s no degrau 5). Afrouxar até o mínimo rende 0,0011–0,0055/s, 5 a
  30 vezes menos que 10% da sacudida, e ~1/300 do rastreio. Mas a alavanca não é o `F_ref`: a derivada em F = 117,5 N,
  e^(−F/F_ref)/F_ref, tem o máximo em F_ref = F, 1/(e·117,5) = 3,13e−3 por N, 1,69× a de hoje (1,86e−3). Nenhum `F_ref`
  dá mais que 1,69×. O que escala é o peso (limitado pela escada (2)) ou a forma do custo (R1, P1).

---

## Defeito silencioso

Nenhum.

## Regra do plano

| # | arquivo:linha | o que está errado | cenário | correção | plano |
|---|---|---|---|---|---|
| R1 | `g1_v3/knobs.py:294`, `:298`; `g1_v3/potencial.py:70`; registro, caso 7 ("a alavanca é o F_ref, não o peso") | O aperto é um canal quase morto no ponto de operação (seção 4), e o remédio que o registro e o plano deixam para a fase I não funciona: o `F_ref` sobe a derivada em 117,5 N no máximo 1,69× (em F_ref = 117,5 N); acima ou abaixo disso ela cai | Fase I: a força por face não cai; sobe-se o `F_ref` para 80–120 N; o ganho de afrouxar vai de 0,0055 a ≤ 0,0094/s no degrau 5, ainda ~1/100 do rastreio; o aperto segue sem efeito e o diagnóstico ("faltou F_ref") sai errado | Corrigir o texto do caso 7 e da §6.2 (`F_ref`): a alavanca é o peso, com a escada (2) refeita, ou a forma. Decisão: P1 | CLAUDE.md (gradiente; canal morto); S §8.2 ("a Parte 0 sobe `F_ref` se ela for rasa demais") |
| R2 | `g1_v3/parte0/item3_escada.py` (escada (2)); `g1_v3/parte0/saida/item3_escada.txt:65-69` | A escada (2) da S §10 soma o `r_aperto` que a perda evita, e não a sacudida, que a perda também evita (o índice sai de P3–B2). No degrau 5, no p50: 0,05·2·20 s = 2,0 de retorno evitado, 3,6× o custo da perda (0,561) | A escada (2) do texto (perda > B + Φ + aperto evitado) falha com a sacudida: 0,561 < 0,4675 + 2,0. Ela só vale porque a perda zera os termos positivos (`paga=True`: vivo e rastreios, ~1,6/s no CARREGAR), que a escada não conta. Uma mudança futura nesses termos quebra a escada sem aviso | Refazer o item 3 com a sacudida evitada e com os termos positivos que a perda zera (o lado direito completo), ou escrever na S §10 que a escada vale por eles. P2 | S §10 (escada); enunciado §3 ("Desistir nunca rende mais") |
| R3 | `g1_v3/metricas.py:445` | O conjunto "caixa presa" (P3, P4, B1, B2) é escrito de novo, e não lido de `recompensas._CAIXA_ALVO` com `_em` | Mudar a máscara do termo (P11) e esquecer a métrica: o `Episodio/aceleracao_da_caixa`, que recalibra a norma na fase I, mede outros passos que o termo | `presa = recompensas._em(env.v3_alvo_idx, recompensas._CAIXA_ALVO)` (os dois já são importáveis; −1 linha de import do `alvos`) | §1.5 (função repetida) |
| R4 | `g1_v3/curriculo.py:47`; `g1_v3/env_cfg.py:82-83`, `:175-176` | O `env_cfg` monta os eventos pelo `k.dr` (o `com_caixa` só entra com `k.dr.com_caixa`), mas o `Curriculo` lê o `ATIVO.dr`: as faixas do `dr()` (`com_caixa`, `pd`, `forca_motor`, `solref`, `rfi`, `atraso_max`) nunca seguem o `k`. A docstring diz "Do `k` só valem ... `dr`" | Treino com o `ATIVO`: igual. Um `k` variante (smoke s37, um teste de DR): o cfg tem o evento e o `dr()` dá a faixa do `ATIVO`; na s37, `com_caixa` = [0, 0], CoM zero (M1) | Mínimo: a s37 põe `env.v3_cur.k_dr = k.dr` antes do `_poe_curriculo` (+1); a docstring diz que o `dr` do `k` vale só para os eventos do cfg | §6.2 (um número, uma fonte); PD-38 |

## Menor

| # | arquivo:linha | o que está errado | cenário | correção |
|---|---|---|---|---|
| M1 | `g1_v3/smoke.py:1408` | "CoM não todo zero" falha sempre: o `dr()` lê o `ATIVO.dr` (R4) e dá `com_caixa` = [0, 0]; o CoM fica no default (0: a geom da caixa está na origem do corpo, `cena.py:79-86`) | A s37 marca falha e a PD-38 fica sem a medida certa (o heap roda com o deslocamento zero) | R4 (+1 linha) |
| M2 | `g1_v3/smoke.py:1252-1264` | A s34 confere a `sacudida` e o `aperto` "finitos e ≥ 0" com todos os envs no ANDAR: o currículo novo só abre o ANDAR (`balanceador.py:20`), a máscara zera os dois e o check passa sem medir | Um NaN no `aperto` com a caixa presa passa | `_abre_trecho_de(env, "PEGAR")` e `env.v3_alvo_idx[:] = P3` antes dos 5 passos (a caixa na laje, a máscara aberta) |
| M3 | `g1_v3/smoke.py:1376-1384` | A máscara da `forca_caixa` é conferida só onde ela é trivialmente zero. Forçar `v3_tarefa = ANDAR` não desalinha nada: com só o ANDAR aberto, toda fatia é ANDAR e a tarefa já é ANDAR (o `_publica` só a escreve na abertura do trecho): a linha não faz nada. Mas `v3_pegou` nunca é True com ação zero, e uma máscara que esquecesse o `pegou` ou o `perdeu` passaria | A s36 aprova uma máscara quebrada | Sem forçar a tarefa: env 0 com `v3_tarefa` = CARREGAR e `v3_pegou` = True, reescritos a cada passo; o check pede wrench ≠ 0 no env 0 em algum de 450 passos (o intervalo é 3–8 s, até 400 passos) e zero nos outros. Forçar o ANDAR pelo currículo já é o padrão (células B e C fechadas) |
| M4 | `g1_v3/smoke.py:1386-1388` | A DÚVIDA tem resposta: o `SimData` embrulha a estrutura aninhada (`sim_data.py:201-215`), e `env.sim.data.contact.solref` (naconmax, 2), `.geom` e `.worldid` existem (`mujoco_warp/_src/types.py:1963-1987`). O print não mede a P17 | A regra da prioridade fica sem prova na fase I | Check: nos contatos ativos com uma geom de pé, `contact.solref[:, 0]` = `geom_solref[worldid, pé, 0]` |
| M5 | `g1_v3/smoke.py:1291-1300` | A contagem do batente por junta usa 200 passos de ação aleatória no ANDAR: mede o curso de um robô que cai, e não o da política | A decisão da P13 (ajuste por família) sai de um dado que não diz nada sobre o tornozelo da política (até 1,07 nos plays) | Manter como diagnóstico; a medida da P13 vem de um play com checkpoint no degrau 1 (o `registra_juntas_v3` já grava as juntas) |
| M6 | `g1_v3/smoke.py:76-77` | A docstring do `_poe_curriculo` diz "o reset sorteia o degrau de cada env (comando.py:458)": o `_publica` grava `cur.degrau(tarefa)`, sem sorteio | Leitor espera degraus misturados no lote | Trocar por "o reset publica o degrau da tarefa (`comando.py:450-461`)" |
| M7 | `g1_v3/estimador.py:214`, `:273-274` | `_t_ant` só serve de bandeira: o dt sai de `self.t` | Dois nomes para o mesmo instante | `self._primeiro = True`, ou `self.t = None` no `__init__` e `dt = 0 if self.t is None else t − self.t` (−1 linha) |
| M8 | `g1_v3/cena.py:408` | O campo `pos` do `caixa_faces` não tem mais leitor (a face sai da normal, P12) | Memória e cópia por passo à toa | Tirar `"pos"` de `fields` |
| M9 | `g1_v3/terminacoes.py:32`, `:165-167`; `g1_v3/recompensas.py:40`, `:623`; `g1_v3/metricas.py:159-162` | Os índices dos dois `hip_yaw` e o centro e o meio dos limites duros estão em três módulos | Três fontes da mesma conta | Uma `potencial`-irmã não serve (módulo puro); aceitar com o comentário, ou um `recompensas.limites_duros(robo)` lido pelas outras duas |
| M10 | `g1_v3/env_cfg.py:166-168` | Com só `actuator_ids`, o `SceneEntityCfg.resolve` preenche `actuator_names` com os nomes dos atuadores 0..8 do spec (por junta), e não dos grupos (`scene_entity_config.py:190-191`) | Sem efeito: o `effort_limits` só lê o `actuator_ids`. Um leitor do nome erra | Comentário no `env_cfg` |
| M11 | `g1_v3/testes/test_estimador_e.py:45`; plano `:684`; `g1_v3/parte0/item15_aperto.py:197`, `:206` | Textos mortos: a chave `sigma_ori_graus` (o `passo` a ignora); o contrato da fase E ainda lista `sigma_ori_graus`; o item 15 cita o `fator_por_degrau` (saiu) e diz que a sacudida usa 2,9 e 17 (a norma é 5,04 e 31,56) | Um leitor copia a chave morta | Tirar a chave; nota "(fase H: chaves novas, §6.3 Fase H)" no plano; o texto do item 15 com a norma do knob |
| M12 | §16 | A fase H somou +389 linhas no pacote (5 666 → 6 055, sem o smoke e o exportador), 34% acima dos ~+290 da §12.4. Os maiores: `eventos` +108 (estimado +75), `terminacoes` +34 (+23), `runner` +24 (+16), `metricas` +38 (+28); o smoke +247 (+178). A sobra é docstring e comentário | A §16 (~6 550 no fim da H) segue folgada | Nenhuma; anotar a contagem real na §20 |

---

## Contagem (§16): antes (`df32c29`) → depois (HEAD)

| arquivo | antes | depois | estimado na §12.4 |
|---|---|---|---|
| `knobs.py` | 485 | 494 | ~485 |
| `balanceador.py` | 296 | 318 | ~312 |
| `potencial.py` | 120 | 127 | ~124 |
| `recompensas.py` | 783 | 841 | ~835 |
| `terminacoes.py` | 137 | 171 | ~160 |
| `eventos.py` | 475 | 583 | ~550 |
| `estimador.py` | 316 | 353 | ~345 |
| `comando.py` | 778 | 793 | ~790 |
| `curriculo.py` | 128 | 130 | ~131 |
| `runner.py` | 42 | 66 | ~58 |
| `env_cfg.py` | 185 | 212 | ~212 |
| `cena.py` | 423 | 423 | 423 |
| `observacoes.py` | 375 | 383 | ~378 |
| `metricas.py` | 497 | 535 | ~525 |
| pacote, sem smoke e exportador | 5 666 | 6 055 | ~5 956 |
| `smoke.py` | 1 272 | 1 519 | ~1 450 |
| testes (18 arquivos) | — | +1 435 −25 | — |

---

## Pendências

Veredito de cada item do pedido.

| item | veredito |
|---|---|
| P1–P24 no código | Todas como o contrato diz (seção 1). P19 vale no treino; com um `k` variante, R4. |
| Sinais dos pesos | Negativos no knob; termos ≥ 0; o `env_cfg` passa o knob cru. |
| Máscara da `forca_caixa` | Certa no código (`eventos.py:562-570`); o smoke não a prova (M3). |
| Modos `step` | `forca_caixa` e `rfi` em `step`, depois do `empurrao`; o mjlab chama com `env_ids` None. |
| Degrau global × da tarefa | Certo (seção 1, último parágrafo). |
| `>` estrito nas 29 juntas | Certo; `joint_pos_limits` são os limites duros (o `soft` 0,9 fica no `soft_joint_pos_limits`). |
| `solref` nas geoms certas | 14 pés, 4 pads, laje, caixa; o terreno fora. Prova no contato: M4. |
| `actuator_ids` dos grupos | [7, 8] e [0..6]; a ordem do `robo.actuators` é a da `Contrato.pd` (`sort_actuators` False). |
| `com_caixa` só com a chave | Certo no cfg; o `dr()` lê a chave do `ATIVO` (R4, M1). |
| `TAU_MAX` sem ciclo | Certo. |
| Lognormal, vieses uniformes, rajada | Certos; o início fica igual ao de antes ((1,5°; 2,5°) é a χ₃ de σ 1°; `vies_pos` 0; leitura 0). |
| Viés do alvo do BOTAR | No `v3_alvo_w`, antes do índice B1; sem gerador, sem viés. |
| Chaves do log | `atraso_max`, `mu_min_pad`, `degrau_global` no `v3`; `Policy/std/<junta>`; `Episodio/aceleracao_da_caixa`. |
| Play sem as cinco DR | Certo; o `com_caixa` fica (não é uma das cinco). |
| `v3_odo.passo` e o fce6a42 | De volta; o resto do fce6a42 tem registro (seção 2). |
| Varredura de remoções | Só o `v3_odo.passo` sem registro (seção 3). |
| NaN, máscara, buffer, números, imports, repetição | Sem achado, menos R3 e M9 (repetição). `_ESC_FORCA_N` = 100 é normalização de contrato (§1.5). |
| Derivada: sacudida no p50 | Viva (seção 4). |
| Derivada: aperto em 117,5 N | Quase morta; a alavanca do `F_ref` é ≤ 1,69× (R1, P1). |
| Contagem | +389 no pacote, 34% acima da estimativa (M12). |
| Smoke × cfg real (nomes) | Os nomes dos termos e dos params batem com o `env_cfg` (`sacudida`, `aperto`, `batente`, `solref` e os três `asset_cfg_*`, `com_caixa`, `forca_caixa`, `rfi`). |
| s36 força ANDAR | Não desalinha: com só o ANDAR aberto a linha não faz nada. Forçar o ANDAR pelo currículo já é o caso (células B e C fechadas no `Curriculo` novo). O defeito é o oposto: o check não exercita a máscara (M3). |
| `test_estimador_e.py:45` | Confirmada (M11). |

## Pendências do dono

| # | pergunta | opções | recomendação |
|---|---|---|---|
| P1 | O aperto quase não tem gradiente (R1). O que fazer se a força por face não cair na fase I? | A: medir primeiro (a força por face no log; hoje só a total `forca_robo_caixa`) e decidir depois. B: subir o peso; o teto sai da escada (2) refeita (P2). C: trocar a forma por uma de derivada constante, −w·min(F_esq, F_dir)/F_ref (0,02/49 = 4,1e−4 por N por s no degrau 5, 11× a de hoje), que perde a saturação | A, com o texto do caso 7 corrigido (a alavanca é o peso ou a forma, não o `F_ref`). Hoje o termo não atrapalha: o custo é ≤ 0,018/s. |
| P2 | A escada (2) não conta a sacudida (R2). | A: refazer o item 3 com a sacudida evitada e com os termos positivos que a perda zera. B: escrever na S §10 que a escada vale pelos termos zerados | A. O item 3 é um script da Parte 0, sem GPU. |

## DÚVIDAS no código

| marca | veredito |
|---|---|
| `g1_v3/smoke.py:1315` (acha o evento pela função) | As duas formas funcionam: o `env.cfg.events` é o mesmo dict que o `EventManager` resolve no lugar (`manager_based_rl_env.py:185`, `:306`; `manager_base.py:142-147`). Pelo nome, `env.cfg.events["solref"]`, é mais curto. Apagar a marca. |
| `g1_v3/smoke.py:1386` (API do `solref` dos contatos) | Existe: `env.sim.data.contact.solref` (M4). Trocar o print por um check. |

---

## Lanes de correção

Arquivos disjuntos dentro de cada onda. Contagem antes → depois (estimada).

**Onda 0 (orquestrador, Opus):** as decisões P1 e P2; a lane 5 corrige o texto.

**Onda 1 (código e testes, Haiku, em paralelo)**

| lane | arquivos | correções | contagem |
|---|---|---|---|
| 1 — métrica | `g1_v3/metricas.py`, `g1_v3/testes/test_metricas_h.py` | R3 (a máscara do `recompensas`) | 535 → ~534 |
| 2 — estimador e cena | `g1_v3/estimador.py`, `g1_v3/cena.py` | M7, M8 | 353 → ~352; 423 → 423 |
| 3 — testes velhos | `g1_v3/testes/test_estimador_e.py` | M11 (a chave morta) | 218 → 218 |
| 4 — cfg | `g1_v3/env_cfg.py` | R4 (docstring), M10 (comentário) | 212 → ~213 |

**Onda 2**

| lane | arquivos | correções | contagem |
|---|---|---|---|
| 6 — smoke | `g1_v3/smoke.py` | M1 (R4: `k_dr`), M2, M3, M4, M6; apagar a DÚVIDA da `:1315` | 1 519 → ~1 535 |
| 7 — Parte 0 | `g1_v3/parte0/item3_escada.py`, `g1_v3/parte0/item15_aperto.py` e as saídas | R2 (pela P2), M11 (textos do item 15) | +~10 |

**Lane 5 (orquestrador, Opus; antes da onda 1)**

| lane | arquivos | correções |
|---|---|---|
| 5 — texto | `docs/planos/2026-10-14-implementacao-g1-v3.md`, `docs/relatorios/2026-10-30-v3-problemas-e-solucoes.md` | R1 (caso 7 e §6.2 `F_ref`: a alavanca é ≤ 1,69×); M11 (`:684`); M12 (a contagem real na §20); a linha desta revisão na §20 |

Nenhum id bloqueia o treino. Antes do smoke da fase I: M1 (a s37 falha sempre), M2 e M3 (passam sem testar). Antes de
decidir a PD-38: M1. Antes de mexer no aperto na fase I: P1 e o texto do R1. M9 fica opcional.
