# Recompensas e penalidades de manipulação, cercas e eventos

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:
- `recompensas.py`, `env_cfg.py`, `potencial.py`, `alvos.py`, `knobs.py`
- `comando.py` (núcleos `ok_agora_*`, `caixa_perdida_agora`, `sorteia_velocidade`, `_monta`, `avanca_trecho`, `CadeiaCmd`, escritas do `TwistV3` que a recompensa lê)
- `curriculo.py`, `balanceador.py` (`no_degrau`, `Curriculo`, `restricao`, `degrau`, `niveis_abertos`, `fecha_iteracao`, `_sobe`)
- `terminacoes.py`, `cena.py` (sensores e grupos de geoms), `observacoes.py` (`referencia_do_alvo`), `eventos.py` (`sorteia_fatia`, `posiciona_cena`, `sorteia_topo_e_massa`, `posiciona_destino`, `tamanho_caixa`), `estimador.py` (`tem_leitura`)
- `mjlab/managers/reward_manager.py`, `mjlab/managers/termination_manager.py`, `mjlab/managers/command_manager.py`, `mjlab/envs/manager_based_rl_env.py`, `mjlab/sensor/contact_sensor.py`

Referências sem prefixo de pasta são arquivos de `g1_v3/`. Referências `mjlab/...` são da biblioteca do fabricante.

---

## 1. Tabela dos 15 termos

Convenção de sinal: o sinal da recompensa é o sinal do peso. Todos os pesos dos 15 termos são negativos, menos o `progresso`. A `restricao` (um escalar do currículo, igual para todos os envs) e o degrau de refino (um inteiro por tarefa) estão definidos na seção 2.

| termo | tarefas onde está registrado (`env_cfg.py`) | peso: knob e valor | sinal | fator que multiplica |
|---|---|---|---|---|
| `cerca_sola` | PEGAR, BOTAR (`env_cfg.py:48`) | `Recompensa.cerca_sola` = −0,05475 (`knobs.py:270`) | − | nenhum |
| `cerca_giro` | ANDAR, PEGAR, CARREGAR, BOTAR (`env_cfg.py:49`) | `Recompensa.cerca_giro` = −0,03965 (`knobs.py:271`) | − | `restricao` só no ANDAR, por `so="ANDAR"` (`recompensas.py:747`) |
| `cerca_tronco` | as quatro (`env_cfg.py:49`) | `Recompensa.cerca_tronco` = −0,04596 (`knobs.py:272`) | − | `restricao` só no ANDAR, por `so="ANDAR"` (`recompensas.py:758`) |
| `cerca_com` | as quatro (`env_cfg.py:49`) | `Recompensa.cerca_com` = −22,09035 (`knobs.py:273`) | − | `restricao` só no ANDAR, por `so="ANDAR"` (`recompensas.py:764`) |
| `cerca_torque` | as quatro (`env_cfg.py:49`) | `Recompensa.cerca_torque` = −0,29615 (`knobs.py:274`) | − | `restricao` só no ANDAR, por `so="ANDAR"` (`recompensas.py:772`) |
| `cerca_pelve` | as quatro (`env_cfg.py:50`) | `Recompensa.cerca_pelve` = −4,45279 (`knobs.py:275`) | − | `restricao` só no ANDAR, por `so="ANDAR"` (`recompensas.py:788`) |
| `cerca_recuo` | PEGAR, CARREGAR, BOTAR (`env_cfg.py:50`) | `Recompensa.cerca_recuo` = −10,39719 (`knobs.py:276`) | − | nenhum |
| `contato_fora` | as quatro (`env_cfg.py:50`) | `Recompensa.contato_fora` = −1,0 (`knobs.py:299`) | − | nenhum |
| `contato_laje` | PEGAR, BOTAR (`env_cfg.py:51`) | `Recompensa.contato_laje` = −2,0 (`knobs.py:301`) | − | o degrau muda o joelho da força (`laje_joelho_N`), e não multiplica |
| `freio` | PEGAR, CARREGAR, BOTAR (`env_cfg.py:51`) | `Recompensa.freio` = −6/29 ≈ −0,2069 (`knobs.py:304`) | − | `freio_fator ^ v3_freio` × rampa `min(s/portão, 1)` (`recompensas.py:919`) |
| `progresso` | as quatro (`env_cfg.py:51`) | peso fixo +1,0 (`env_cfg.py:57`); a escala está nas rendas `renda_etapa` = 6,0, `renda_phi` = 3,0, `renda_conquista` = 2,0 (`knobs.py:338-343`) | + | `h` (só na parcela da etapa); máscara de conquista |
| `sacudida` | PEGAR, CARREGAR, BOTAR (`env_cfg.py:52`) | `Recompensa.sacudida` = −0,05 (`knobs.py:353`) | − | `degrau/5` (`recompensas.py:944`); só com alvo P3 ou B1 |
| `aperto` | PEGAR, CARREGAR, BOTAR (`env_cfg.py:52`) | `Recompensa.aperto` = −0,05 (`knobs.py:355`) | − | `degrau/5` (`recompensas.py:955`); só com alvo P3 ou B1 |
| `perda_caixa` | as quatro (`env_cfg.py:53`) | peso fixo −1,0 (`env_cfg.py:57`); o custo é `Recompensa.perda_caixa` = 240,0 (`knobs.py:367`) | − | nenhum |
| `terminacao` | as quatro (`env_cfg.py:53`) | `Recompensa.terminacao` = −300,0 (`knobs.py:366`) | − | nenhum |

"As quatro" são ANDAR, PEGAR, CARREGAR e BOTAR (`env_cfg.py:33`). As tuplas de tarefa estão em `env_cfg.py:33-36`.

---

## 2. Como um valor vira recompensa

Esta seção descreve o que vale para os 15 termos. As seções seguintes só repetem o que muda por termo.

**Valor, peso e dt.** Cada termo devolve um tensor `(E,)`, uma grandeza por env, sem sinal e sem peso. O gerente de recompensa do mjlab calcula `valor × peso × dt` e soma os termos (`mjlab/managers/reward_manager.py:116-131`).
- O `peso` sai de `_PESO_FIXO` quando o nome está nele (`progresso` = +1,0 e `perda_caixa` = −1,0) e do campo de mesmo nome de `Recompensa` nos outros 13 (`env_cfg.py:57`, `env_cfg.py:131-134`).
- O `dt` é `env.step_dt` = `Contrato.dt_fisica` × `Contrato.decimacao` = 0,005 s × 4 = 0,02 s (`knobs.py:54-55`, `env_cfg.py:103-104`, `mjlab/envs/manager_based_rl_env.py:266-268`).
- O gerente troca NaN e infinito por zero depois da multiplicação (`mjlab/managers/reward_manager.py:129`). Um termo com peso 0 não é chamado (`mjlab/managers/reward_manager.py:122-124`). Nenhum dos 15 tem peso 0.
- A "taxa por segundo" deste documento é `valor × peso`. A recompensa do passo é a taxa × 0,02.

**Máscara de tarefa.** A tarefa (`v3_tarefa`) é o tipo do trecho corrente da cadeia: ANDAR, PEGAR, CARREGAR ou BOTAR. Todo termo termina em `_fim(env, tarefas, valor)`. Ela devolve `valor` onde a tarefa do env está em `tarefas` e zero nos outros (`recompensas.py:361-365`). Ela usa `torch.where`, e não multiplicação, porque NaN × 0 = NaN (`curriculo.py:93-100`). A lista `tarefas` vem de `_RECOMPENSAS` (`env_cfg.py:42-54`). O parâmetro `paga` de `_fim` zera o valor depois da caixa perdida. Nenhum dos 15 termos usa `paga=True`.

**Ordem dentro de um passo.** Em `step`, o mjlab calcula as terminações, depois as recompensas, depois o reset dos envs terminados, depois `sim.forward()`, depois os comandos (`cadeia` e `twist`) (`mjlab/envs/manager_based_rl_env.py:436`, `:440`, `:447`, `:454`, `:456`). Por isso a recompensa de um passo lê o que os comandos gravaram no passo anterior: `v3_alvo_idx`, `v3_sigma`, `v3_perdeu`, `v3_perdeu_agora`, `v3_pega_fechada`, `v3_bota_fechada`, `v3_tarefa`, `v3_freio`, `v3_degrau`.

**Dois fatores de currículo.**
- A **restrição** (`restricao`) é um número em [0, 1] do currículo: o menor valor de `nível/nível_máximo` entre as quatro tarefas (`balanceador.py:345-348`). O nível máximo é 7 no ANDAR, `round((Comando.frente_tetos[-1] − Comando.andar_inicio) / Comando.andar_passo)` = round((1,0 − 0,3) / 0,1) (`knobs.py:122`, `knobs.py:129`, `knobs.py:134`, `balanceador.py:72`), e 4 nas outras: `len(Comando.frente_tetos) − 1` = 4, `len(Niveis.pegar) − 1` = 4, `len(Niveis.botar) − 1` = 4 (`balanceador.py:73-74`, `knobs.py:417-418`). A restrição vale 0 enquanto uma tarefa está no nível 0. Ela vale 1 só com as quatro no nível máximo. O nível de uma tarefa sobe depois de `Balanceador.iters_nivel` = 50 fechos de iteração seguidos com a condição `ok` verdadeira. Abaixo do nível máximo, `ok` exige todas as células de velocidade da tarefa abertas (ANDAR: `andar`; CARREGAR: `reto`, `giro_lugar`, `arco`; PEGAR: `pegar`; BOTAR: `botar`) e a média das notas `s` dessas células maior ou igual à média dos portões delas (`balanceador.py:29-30`, `balanceador.py:125-128`). O portão de cada célula é `Balanceador.portao_andar` = 0,75 nas células `andar` do ANDAR, `Balanceador.portao_andar_terreno` = 0,50 na célula `terreno` do ANDAR e `Balanceador.portao` = 0,75 nas outras tarefas (`knobs.py:391-393`, `balanceador.py:96-98`). A nota `s` de uma célula é uma média móvel exponencial atualizada a cada fecho de iteração, com taxa `Balanceador.alfa` = 0,05 (`knobs.py:394`, `balanceador.py:159-168`). Um fecho com `ok` falso zera a contagem (`knobs.py:395`, `balanceador.py:250-278`). No PEGAR e no BOTAR, o nível 0 tem dois subníveis (`len(Niveis.subnivel_extra)` = 2, `knobs.py:422`). Os primeiros 50 fechos sobem o subnível, e só os 50 seguintes sobem o nível. Por isso `restricao` só sai de 0 depois de pelo menos 100 fechos nessas duas tarefas (`balanceador.py:92`, `balanceador.py:273-276`). No nível máximo a tarefa não sobe mais o nível. No CARREGAR, com a linha de terreno abaixo do teto, `ok` passa a exigir a célula `terreno` aberta e com nota no portão dela, e a subida eleva a linha de terreno; `restricao` não muda (`balanceador.py:261-262`, `balanceador.py:269-271`). No ANDAR a linha de terreno não sobe por esse caminho (27/11, T19): é uma linha por coluna, movida no fecho de iteração por `linha ← clip(linha + Balanceador.linha_ganho · (taxa de travessia − Balanceador.zona_t_alvo), 0, teto)`, só para a coluna com pelo menos `Balanceador.linha_n_min` visitas (`balanceador.py:177-182`, `knobs.py:410-411`). No ANDAR, no nível máximo, `ok` vale falso, como no PEGAR e no BOTAR (`balanceador.py:263-264`). A célula `terreno` do ANDAR abre no nível `Balanceador.nivel_abre_terreno` = 2, com as células abertas no portão (`balanceador.py:222-226`, `knobs.py:412`).
- Um fecho de iteração roda dentro do termo de currículo `v3`, que o mjlab chama na primeira linha do reset dos envs, antes de `sim.reset` e antes dos eventos de reset (`mjlab/envs/manager_based_rl_env.py:553-554`). O termo calcula `iteração = common_step_counter // Balanceador.passos_por_iteracao` e fecha uma iteração quando esse número muda desde a chamada anterior (`knobs.py:402`, `curriculo.py:103-121`). O fecho acontece, portanto, no primeiro reset depois que o contador de iteração muda, e no máximo uma vez por chamada.
- O degrau de uma tarefa sobe em `balanceador._conta_refino`, a cada fecho de iteração. A condição: a tarefa está no nível máximo (e, nas tarefas da cadeia, com a célula de terreno aberta e o nível de terreno no máximo) e a média das células abertas passa no portão. Depois de `Balanceador.iters_refino` = 100 fechos seguidos nessa condição o degrau sobe 1, até `Balanceador.degraus` = 5. No ANDAR, o nível de terreno é o piso da média das linhas por coluna (`balanceador.py:329-330`). O degrau nunca desce (`balanceador.py:234-248`, `knobs.py:400-401`).
- O **degrau** (`v3_degrau`) é o degrau de refino da tarefa, de 0 a `Balanceador.degraus` = 5 (`knobs.py:401`). O `v3_degrau` e o `v3_freio` (= nível da tarefa + 1, `balanceador.py:335-336`) são copiados no começo de cada trecho, em `_publica` (`comando.py:508-519`). Uma subida de degrau só chega ao termo no trecho seguinte.

**Funções de fator.**
- `balanceador.no_degrau(ini, fim, d, degraus)` = `ini + (fim − ini)·d/degraus` (`balanceador.py:48-51`).
- `por_degrau(env, ini, fim)` = `no_degrau(ini, fim, v3_degrau, 5)` (`recompensas.py:447-450`). Com `(0, 1)` vale `degrau/5`.
- `por_restricao(env, ini, so=None)` = `no_degrau(ini, 1.0, restricao, 1)` = `ini + (1 − ini)·restricao`. Com `so` (o nome de uma tarefa), o fator só vale nos envs dessa tarefa, e vale 1 nos outros (`recompensas.py:453-459`). As cercas chamam `por_restricao(env, 0,0, so="ANDAR")`: o fator é a própria `restricao` no ANDAR e 1 fora dele. Os três `desvio_*` do documento 06 chamam com `so="CARREGAR"`.

**Alvos e etapas.** "Alvo corrente" é o índice `v3_alvo_idx` em `ALVOS` = (P0, P1, P2, P3, B1, B2, B3), valores 0 a 6; −1 quer dizer nenhum (`knobs.py:44`, `alvos.py:16`). Os alvos P0 a P3 são as etapas da pega no PEGAR. Os alvos B1 a B3 são as etapas do BOTAR. No CARREGAR o índice fica em P3 (seção 5.2).

Grupos de índices que os termos usam (`recompensas.py:54-59`):

| grupo | alvos | usado por |
|---|---|---|
| `_CAIXA_ALVO` | P3, B1 | máscara `presa` de `sacudida` e `aperto`; conquistas do `progresso` |
| `_P3_B2` | P3, B1, B2 | caixa entra no CoM de `cerca_com` |
| `_JANELA_PEGA` | P1, P2, P3, B1, B2 | janela em que `contato_fora` ignora a caixa |
| `_INCL` | P2, P3, B1, B2 | portão `incl` |
| `_MAOS` | P3, B1 | portão `maos` |
| `_P0_P2` | P0, P1, P2 | portão `desloc` da pega |
| `_B2_B3` | B2, B3 | portão `desloc` do apoio |
| `_PALMA_ALVO` | P0, P1, P2, B2 | distância `d` por palmas |
| `_B1_B2` | B1, B2 | topo de destino em `_topo_do` |
| `_POUSO` | B1 | fator de pouso |

---

## 3. Estado de cada cerca, por tarefa e por `restricao`

Uma cerca é um custo em forma de dobradiça: vale zero dentro do limite e cresce linearmente fora dele. A cerca não tem banda, salvo a `cerca_torque` (seção 4.5).

Estados usados na tabela:
- **ativa**: o termo calcula o custo, com fator 1.
- **desligada**: o termo está registrado, mas o fator `restricao` vale 0, e o custo vale 0.
- **fator r**: o fator vale `restricao` (de 0 a 1). Ele multiplica o custo.
- **não registrada**: o termo vale zero nesta tarefa por causa de `env_cfg.py`.

| cerca | ANDAR, `restricao` 0 | ANDAR, `restricao` 1 | PEGAR (0 e 1) | CARREGAR (0 e 1) | BOTAR (0 e 1) |
|---|---|---|---|---|---|
| `cerca_sola` | não registrada | não registrada | ativa | não registrada | ativa |
| `cerca_giro` | desligada | ativa | ativa | ativa | ativa |
| `cerca_tronco` | desligada | ativa | ativa | ativa | ativa |
| `cerca_com` | desligada | ativa | ativa | ativa | ativa |
| `cerca_torque` | desligada | ativa | ativa | ativa | ativa |
| `cerca_pelve` | desligada | ativa | ativa | ativa | ativa |
| `cerca_recuo` | não registrada | não registrada | ativa | ativa | ativa (zero com alvo B3 ou −1) |

Leitura da tabela:
- No ANDAR, entre `restricao` 0 e 1, as cinco cercas com fator valem `restricao × custo` (fator r). A prova é `por_restricao(env, 0,0, so="ANDAR")` (`recompensas.py:453-459`).
- Nas tarefas PEGAR, CARREGAR e BOTAR o fator vale 1 com qualquer `restricao`: o ramo `so` devolve `torch.ones_like` fora do ANDAR (`recompensas.py:457-458`).
- `cerca_sola` e `cerca_recuo` não chamam `por_restricao`.
- A `cerca_recuo` tem uma exceção própria: ela vale 0 no BOTAR com alvo B3 ou sem alvo (`recompensas.py:801-802`).

---

## 4. Seções por termo

### 4.1 `cerca_sola`: a sola do pé em contato fica plana

**Fórmula.** `Σ_pés [pé em contato] · relu(incl_pé − lim_sola_graus)` (`recompensas.py:156-158`). O custo é em graus.

**Entradas** (`recompensas.py:736-741`):
- `incl_pé`: inclinação do eixo z do corpo `left_ankle_roll_link` e `right_ankle_roll_link` em relação à vertical. `_incl_graus` = `rad2deg(acos(clamp(1 − 2(x² + y²), −1, 1)))`, com x e y as componentes do quaternion wxyz (`recompensas.py:368-371`, `recompensas.py:36`).
- `contato`: `pes_chao.found > 0`, um booleano por pé. O sensor `pes_chao` tem como primários os dois corpos de tornozelo e `secondary=None`, então qualquer contato do pé conta, inclusive com a laje (`cena.py:384-389`).
- Limite: `Recompensa.lim_sola_graus` = 5,0 graus (`knobs.py:277`).

**Tarefas.** PEGAR e BOTAR (`env_cfg.py:48`).
⚠ Divergência: o docstring diz "Todas as tarefas" (`recompensas.py:737`); o registro em `env_cfg.py:48` usa `PEGAR_BOTAR`. Vale o registro.

**Máscaras e fatores.** Nenhum fator de `restricao` nem de degrau. Não usa `paga`.

**Vale zero quando:** a tarefa é ANDAR ou CARREGAR; o pé não tem contato; a inclinação do pé é ≤ 5,0 graus.

### 4.2 `cerca_giro`: o `hip_yaw` das pernas não gira para dentro além do limite

**Fórmula.** `relu(max(−q_E, 0)° − lim) + relu(max(q_D, 0)° − lim)` com `q_E` e `q_D` em rad, convertidos em graus por `rad2deg` (`recompensas.py:161-168`). Só o lado que gira para dentro conta: o sinal da perna esquerda é invertido (`−q_E`) e o da direita não.

**Entradas** (`recompensas.py:744-747`): `joint_pos` das juntas `left_hip_yaw_joint` (índice 2 de `JUNTAS`) e `right_hip_yaw_joint` (índice 8) (`recompensas.py:41`, `knobs.py:31-41`). Limite: `Recompensa.lim_giro_graus` = 30,0 graus (`knobs.py:278`).

**Tarefas.** As quatro (`env_cfg.py:49`).

**Máscaras.** Só a máscara de tarefa `_fim` (`recompensas.py:361-365`): o valor vale zero fora das tarefas registradas. Nenhuma outra máscara.

**Fatores.** `por_restricao(env, 0,0, so="ANDAR")` (`recompensas.py:747`): no ANDAR o custo é multiplicado por `restricao`; nas outras tarefas por 1. Não segue o degrau. Não usa `paga`.

**Vale zero quando:** o giro para dentro é ≤ 30,0 graus; ANDAR com `restricao` 0.

### 4.3 `cerca_tronco`: o torso não se inclina além da referência mais o limite

**Fórmula.** `relu(incl_torso − (tronco_ik + lim_tronco_graus))`, tudo em graus (`recompensas.py:171-173`).

**Entradas** (`recompensas.py:750-758`):
- `incl_torso`: inclinação do eixo z de `torso_link` em relação à vertical, com a mesma conta de `_incl_graus` (`recompensas.py:467-470`).
- `tronco_ik`: nas tarefas PEGAR e BOTAR é `rad2deg(ref_do_alvo(env, v3_alvo_idx)[:, 7])`, o tronco de referência do alvo corrente. Nas outras tarefas vale 0 (`recompensas.py:756-757`).
- `ref_do_alvo` devolve o canal 7 (tronco) só para os alvos P0, P1, P2, B1 e B2, por `ref.tronco(topo, meia aresta)`, uma tabela bilinear. Para P3, B3 e −1 devolve 0 (`observacoes.py:93`, `observacoes.py:123-130`, `recompensas.py:501-506`).
- O topo usado é `v3_topo_destino` nos alvos B1 e B2 e `v3_topo` nos outros (`recompensas.py:496-498`).
- Limite: `Recompensa.lim_tronco_graus` = 15,0 graus (`knobs.py:279`).

**Tarefas.** As quatro (`env_cfg.py:49`).

**Máscaras.** Só a máscara de tarefa `_fim` (`recompensas.py:361-365`): o valor vale zero fora das tarefas registradas. Nenhuma outra máscara.

**Fatores.** `por_restricao(env, 0,0, so="ANDAR")` (`recompensas.py:758`).

**Vale zero quando:** `incl_torso` ≤ `tronco_ik` + 15,0; ANDAR com `restricao` 0.

### 4.4 `cerca_com`: o centro de massa fica dentro do polígono dos pés

**Fórmula.** `relu(lim_com − margem) · [dois pés em contato]` (`recompensas.py:176-179`). A `margem` é a distância com sinal do CoM às arestas do fecho convexo dos cantos das solas, positiva dentro, em metros (`recompensas.py:316-336`).

**Como a margem é calculada** (`recompensas.py:316-336`):
- Os cantos são 8 pontos: `ref.cantos_sola` tem 4 cantos por pé, no frame do tornozelo, levados ao mundo pela pose dos dois tornozelos (`recompensas.py:438-444`, `alvos.py:64`).
- Cada par de cantos define uma reta. A reta é aresta do fecho quando todos os cantos ficam do mesmo lado dela (tolerância `_EPS` = 1e-6 m, `recompensas.py:34`) e o comprimento do par é > `_EPS`.
- A margem é o mínimo, sobre as arestas válidas, da distância com sinal do CoM. Sem aresta válida, a margem vale 1e3.

**Entradas** (`recompensas.py:473-479`, `recompensas.py:761-764`):
- CoM combinado (`_com_combinado`, `recompensas.py:419-435`): CoM do robô (`subtree_com` e `body_subtreemass` do corpo raiz) ponderado pela massa. A caixa entra com `v3_massa` e `root_com_pos_w` só quando `v3_pegou & ~v3_perdeu` e o alvo está em {P3, B1, B2} (`recompensas.py:57`, `recompensas.py:431`). Só o xy é usado.
- `dois_pes`: `pes_chao.found > 0` nos dois pés (`recompensas.py:478`).
- Limite: `Recompensa.lim_com` = 0,03 m (`knobs.py:280`).

**Tarefas.** As quatro (`env_cfg.py:49`).

**Máscaras.** Só a máscara de tarefa `_fim` (`recompensas.py:361-365`): o valor vale zero fora das tarefas registradas. Nenhuma outra máscara.

**Fatores.** `por_restricao(env, 0,0, so="ANDAR")` (`recompensas.py:764`).

**Vale zero quando:** menos de dois pés em contato; `margem ≥ 0,03` m; ANDAR com `restricao` 0.

### 4.5 `cerca_torque`: o torque de cada junta e de cada par fica abaixo do limite

**Fórmula** (`recompensas.py:182-193`):
`( Σ_juntas soltas relu(|τ|/τ_max − lim) + Σ_pares relu((|τ_p| + |τ_r|)/tau_par − lim) ) / banda`

**Entradas** (`recompensas.py:767-772`):
- `τ` = `qfrc_actuator` das 29 juntas.
- `τ_max` por junta, `TAU_MAX`, montado da tabela por regex `Recompensa.tau_max` (`knobs.py:286-292`, `knobs.py:572`):

| juntas | `τ_max` (N·m) |
|---|---|
| `.*knee_joint`, `.*hip_roll_joint` | 139,0 |
| `.*hip_(pitch\|yaw)_joint`, `waist_yaw_joint` | 88,0 |
| `waist_(roll\|pitch)_joint`, `.*ankle_(pitch\|roll)_joint` | 50,0 |
| `.*_(shoulder_.*\|elbow\|wrist_roll)_joint` | 25,0 |
| `.*_wrist_(pitch\|yaw)_joint` | 5,0 |

- Os pares são `Recompensa.pares_torque`: (`waist_pitch_joint`, `waist_roll_joint`), (`left_ankle_pitch_joint`, `left_ankle_roll_joint`) e (`right_ankle_pitch_joint`, `right_ankle_roll_joint`) (`knobs.py:293-297`). As 6 juntas dos pares não entram como soltas. As outras 23 entram.
- `tau_par` = `Recompensa.tau_par` = 50,0 N·m (`knobs.py:298`).
- Limite: `Recompensa.lim_torque` = 0,8 (fração de `τ_max` ou de `tau_par`) (`knobs.py:281`).
- Banda: `Recompensa.banda_torque` = 0,2 (`knobs.py:282`). A banda divide o excesso. Ela não é uma largura de rampa com teto: o custo continua a crescer linearmente além de `lim + banda`.

**Tarefas.** As quatro (`env_cfg.py:49`).

**Máscaras.** Só a máscara de tarefa `_fim` (`recompensas.py:361-365`): o valor vale zero fora das tarefas registradas. Nenhuma outra máscara.

**Fatores.** `por_restricao(env, 0,0, so="ANDAR")` (`recompensas.py:772`).

**Vale zero quando:** todas as juntas soltas têm `|τ|/τ_max ≤ 0,8` e todos os pares têm `(|τ_p| + |τ_r|)/50,0 ≤ 0,8`; ANDAR com `restricao` 0.

### 4.6 `cerca_pelve`: a altura da pelve segue a referência da tarefa

**Fórmula.** `relu(|h − ik| − lim_pelve)`, em metros (`recompensas.py:196-198`).

**Entradas** (`recompensas.py:775-788`):
- `h`: altura da pelve sobre o terreno, `altura_pelve.heights[:, 0]` (`recompensas.py:781`).
- `ik`, por regra, nesta ordem de precedência (a última que vale vence):
  1. Base: `ref.pelve_de_pe` (um escalar do arquivo de referências) em todo env.
  2. Tarefa CARREGAR: `ref.pelve_com_caixa(v3_meia)`, interpolada na meia aresta da caixa.
  3. Tarefa PEGAR ou BOTAR **com** `v3_alvo_idx ≥ 0`: `ref_do_alvo(env, v3_alvo_idx)[:, 6]` (`recompensas.py:785-787`). Esse canal vale: `ref.pelve_z(topo, meia)` nos alvos P0, P1, P2, B1, B2; `ref.pelve_com_caixa(meia)` no P3; `ref.pelve_de_pe` no B3 (`observacoes.py:127-128`).
  - Em PEGAR ou BOTAR com alvo −1 vale a regra 1 (pose de pé).
- Limite: `Recompensa.lim_pelve` = 0,08 m (`knobs.py:283`).

**Tarefas.** As quatro (`env_cfg.py:50`).

**Máscaras.** Só a máscara de tarefa `_fim` (`recompensas.py:361-365`): o valor vale zero fora das tarefas registradas. Nenhuma outra máscara.

**Fatores.** `por_restricao(env, 0,0, so="ANDAR")` (`recompensas.py:788`).

**Vale zero quando:** `|h − ik| ≤ 0,08` m; ANDAR com `restricao` 0.

### 4.7 `cerca_recuo`: a caixa fica à distância de referência da pelve

**Fórmula.** `relu(|recuo − ik| − lim_recuo)`, em metros (`recompensas.py:201-203`).

**Entradas** (`recompensas.py:791-802`):
- `recuo`: norma, no plano xy do mundo, de (posição da caixa − posição da pelve) (`root_link_pos_w` da caixa e do robô).
- `ik` = `_recuo_ref(env, v3_alvo_idx)` (`recompensas.py:509-516`): `‖ref.caixa_transporte_b(meia)[:2]‖` quando o alvo é P3 ou a tarefa é CARREGAR; `ref.recuo(topo, meia)` nos outros casos. O topo é o de `_topo_do` (seção 4.3).
- Limite: `Recompensa.lim_recuo` = 0,08 m (`knobs.py:284`).
- Zerado quando `tarefa == BOTAR` e (`v3_alvo_idx == B3` ou `v3_alvo_idx < 0`) (`recompensas.py:801-802`).

**Tarefas.** PEGAR, CARREGAR, BOTAR (`env_cfg.py:50`).

**Máscaras.** A máscara de tarefa `_fim` e um zero próprio: a cerca vale 0 no BOTAR com alvo B3 ou sem alvo, por `torch.where` (`recompensas.py:801-802`).

**Fatores.** Nenhum. A `restricao` e o degrau não entram. Não usa `paga`.

**Vale zero quando:** `|recuo − ik| ≤ 0,08` m; BOTAR com alvo B3 ou sem alvo; tarefa ANDAR (não registrada).
Nota: no PEGAR com alvo −1 (espera pela primeira leitura da caixa, seção 6.3) o termo calcula `ref.recuo` com o `v3_topo` e vale a conta normal (`recompensas.py:801-802` só zera o BOTAR).

### 4.8 `contato_fora`: contatos que não são os pés no chão

**Fórmula.** `(nº de corpos com força no chão > contato_fora_N) + (autocolisão) + (nº de slots de contato com a caixa com força > contato_fora_N, fora da janela da pega)` (`recompensas.py:206-211`, `recompensas.py:805-813`).

**Entradas:**
- Chão: sensor `nao_pes_chao`. Seus primários são os corpos do robô que casam `.*`, menos os `.*_ankle_roll_link`; o secundário é o corpo `terrain` (`cena.py:402-408`). A força é um vetor por corpo (`mjlab/sensor/contact_sensor.py:188-196`). O termo usa `force.norm(-1) > 1,0` e conta os corpos.
  ⚠ Divergência: o comentário em `recompensas.py:808` diz `(E,1)`; o sensor devolve um valor por corpo primário.
- Autocolisão: `(auto_colisao.found > 0).sum(-1)`. O sensor tem um primário (`subtree` da `pelvis`) contra o mesmo subtree, então o termo soma 0 ou 1 (`cena.py:379-382`, `mjlab/sensor/contact_sensor.py:187-190`).
- Caixa: sensor `caixa_robo`, primário a geom da caixa e secundário o subtree da pelve, 12 slots (`cena.py:375-378`). Conta os slots com `force.norm(-1) > 1,0`.
- Janela da pega: `v3_alvo_idx` em {P1, P2, P3, B1, B2} (`recompensas.py:59`) ou tarefa CARREGAR. Dentro da janela a parcela da caixa vale 0.
- Limiar: `Recompensa.contato_fora_N` = 1,0 N (`knobs.py:300`).

**Tarefas.** As quatro (`env_cfg.py:50`).

**Máscaras.** A máscara de tarefa `_fim` (`recompensas.py:361-365`) e a janela da pega, que zera só a parcela da caixa (`torch.where`, `recompensas.py:812-813`). A parcela do chão e a da autocolisão não têm janela.

**Fatores.** Nenhum. Não usa `paga`. Não segue a `restricao` nem o degrau. Não tem fator só do ANDAR.

**Vale zero quando:** nenhum corpo não-pé passa de 1,0 N contra o chão; sem autocolisão; sem contato com a caixa acima de 1,0 N, ou com a caixa dentro da janela da pega (alvos P1 a B2 ou CARREGAR). Nos alvos P0 e B3, no alvo −1 e no ANDAR, o contato da caixa com o robô conta.

### 4.9 `contato_laje`: o corpo, a palma e o dorso da mão na laje

**Fórmula.** `Σ_{3 sensores} clamp((F − joelho) / sat, 0, 1)`, com F = maior norma de força entre os primários do sensor (`recompensas.py:887-896`).

**Entradas:**
- Sensores `corpo_prateleira`, `palma_prateleira` e `dorso_prateleira` (`MESA_POR_GRUPO`, `cena.py:49-50`). O primário de cada um é um grupo de geoms do robô contra a geom da laje (`TABLE_GEOM`): tronco (`pelvis_collision`, `torso_collision`, `.*_hip_collision`, `.*_thigh_collision`), almofada da palma (`.*_palm_pad`), almofada do dorso da mão (`.*_hand_back_pad`) (`cena.py:45-47`, `cena.py:368-373`).
- `joelho` = `por_degrau(env, laje_joelho_N[0], laje_joelho_N[1])` = 50,0 + (10,0 − 50,0)·degrau/5 N, isto é, 50,0 no degrau 0 e 10,0 no degrau 5 (`Recompensa.laje_joelho_N` = (50,0; 10,0), `knobs.py:302`).
- `sat` = `max(laje_saturacao_N − joelho, 1e-6)`, com `Recompensa.laje_saturacao_N` = 100,0 N (`knobs.py:303`, `recompensas.py:891`). A parcela de cada sensor chega a 1 quando F = 100,0 N.

**Tarefas.** PEGAR e BOTAR (`env_cfg.py:51`).

**Máscaras.** Só a máscara de tarefa `_fim` (`recompensas.py:361-365`). Nenhuma outra.

**Fatores.** O degrau muda o `joelho` (`por_degrau`, `recompensas.py:890`) e não multiplica o valor. Não segue a `restricao`. Não tem fator só do ANDAR. Não usa `paga`. O valor total fica em [0, 3].

**Vale zero quando:** a força de cada um dos três sensores é ≤ `joelho` (50,0 N no degrau 0; 10,0 N no degrau 5); tarefa ANDAR ou CARREGAR.

### 4.10 `freio`: excesso de velocidade de junta sobre o teto do regime

**Fórmula.** `custo × freio_fator ^ v3_freio × rampa`, com `custo = Σ_29 juntas relu(|q̇|/vmax − 1)²` (`recompensas.py:913`, `recompensas.py:919`).

**Entradas** (`recompensas.py:899-919`):
- Comando do `twist` (vx, vy, wz). `total = ‖cmd_xy‖ + |cmd_z|`.
- Regime, por `total`:
  - parado se `total < Recompensa.freio_limiar_andar` = 0,05;
  - andando se não parado e `total < Recompensa.freio_limiar_correr` = 1,5;
  - correndo nos outros casos (`knobs.py:308-309`).
- `vmax` por junta e regime (rad/s), das tabelas por regex `vel_max_parado`, `vel_max_andando`, `vel_max_correndo` (`knobs.py:311-331`, `recompensas.py:42-44`):

| junta | parado | andando | correndo |
|---|---|---|---|
| `hip_pitch` | 1,5 | 4,0 | 5,0 |
| `hip_roll` | 1,5 | 5,0 | 5,0 |
| `hip_yaw` | 1,5 | 4,0 | 4,0 |
| `knee` | 1,5 | 6,5 | 10,0 |
| `ankle_pitch` | 1,5 | 6,0 | 9,0 |
| `ankle_roll` | 1,5 | 4,0 | 4,0 |
| `waist_yaw` | 1,5 | 3,0 | 3,0 |
| `waist_roll` | 1,5 | 5,0 | 5,0 |
| `waist_pitch` | 1,5 | 2,5 | 2,5 |
| `shoulder_pitch` | 1,5 | 3,0 | 3,5 |
| `shoulder_roll` | 1,5 | 4,0 | 4,5 |
| `shoulder_yaw` | 1,5 | 2,0 | 2,5 |
| `elbow` | 1,5 | 3,0 | 3,0 |
| `wrist` (as 3 juntas) | 1,0 | 2,5 | 2,5 |

- `freio_fator` = `Recompensa.freio_fator` = 1,5 (`knobs.py:305`). `v3_freio` = nível da tarefa + 1, copiado no começo do trecho (`balanceador.py:335-336`, `comando.py:515-517`). O preço sobe 1,5 vezes a cada nível.
- `rampa` = `clamp(s_t / Balanceador.portao, max=1)` indexada pela tarefa do env. `s_t` é a média das notas `s` das células abertas da tarefa; sem célula aberta, 0. `Balanceador.portao` = 0,75 (`knobs.py:391`, `recompensas.py:916-918`).
- O comando do `twist` vale zero, e o regime do `freio` é "parado", nestes trechos: PEGAR, BOTAR, ANDAR das fatias B e C, e CARREGAR da fatia C. O sorteio de velocidade roda só para ANDAR da fatia ANDAR e CARREGAR da fatia B (`comando.py:901-905`). Ele roda no reset do episódio e na abertura de cada trecho (`_publica` chama `_resample`, `comando.py:519`), e a cada reamostra o mjlab sorteia `time_left` uniforme em `Comando.reamostra_s` = (3,0; 8,0) s; quando `time_left` chega a 0, o comando é reamostrado de novo (`knobs.py:137`, `env_cfg.py:122-123`, `mjlab/managers/command_manager.py:109-121`). O comando do CARREGAR da fatia B fica fixo até o fim do trecho (`comando.py:940-941`). Para o `freio`, que não está registrado no ANDAR, só o CARREGAR da fatia B tem comando diferente de zero. Nele o sorteio depende do tipo de célula: `reto`, `terreno` e `arco` sorteiam o linear; `arco` e `giro_lugar` sorteiam o giro (`comando.py:148-158`).
- Faixas do comando do CARREGAR, com `nível` = nível do CARREGAR (0 a 4) (`comando.py:148-157`): teto de frente = `Comando.frente_tetos[nível]` com `frente_tetos` = (0,4; 0,55; 0,7; 0,85; 1,0) m/s (`knobs.py:122`); teto de trás = `min(frente, Comando.tras_teto_max)` com `tras_teto_max` = 0,5 m/s (`knobs.py:123`); teto de lado = `min(frente, Comando.lado_teto_max)` com `lado_teto_max` = 0,3 m/s (`knobs.py:124`); teto de giro = `Comando.giro_tetos[min(nível, 3)]` com `giro_tetos` = (0,3; 0,4; 0,5; 0,5) rad/s (`knobs.py:125`). Na célula `terreno`, frente e trás ficam em no máximo `Comando.terreno_v_max` = 1,0 m/s (`knobs.py:148`). O linear (`reto`, `terreno`, `arco`) escolhe um de 4 eixos (+x, −x, +y, −y) com probabilidade 1/4 cada, e o módulo é uniforme em [piso; teto do eixo], com piso `Comando.piso_velocidade` = 0,25 m/s para ±x e `Comando.piso_lado` = 0,2 m/s para ±y (`comando.py:102-110`, `knobs.py:126-127`). O giro (`arco`, `giro_lugar`) tem sinal sorteado com probabilidade 1/2 e módulo uniforme em [0,25; teto de giro] rad/s (`comando.py:113-115`). Cada sorteio é feito na criação do comando do trecho (reset do episódio ou abertura do trecho).

**Tarefas.** PEGAR, CARREGAR, BOTAR (`env_cfg.py:51`).

**Máscaras.** Só a máscara de tarefa `_fim` (`recompensas.py:361-365`). A rampa e o regime não mascaram: eles escalam o valor.

**Fatores.** `Recompensa.freio_fator ^ v3_freio` e `rampa` (`recompensas.py:919`). Com `v3_freio` = nível da tarefa + 1, o fator vai de 1,5 (nível 0) a 1,5⁵ ≈ 7,59 (nível 4); com o peso −6/29, o preço por unidade de custo vai de ≈ −0,31 a ≈ −1,57. Não depende da `restricao` nem do degrau de refino. Não tem fator só do ANDAR. Não usa `paga`.

**Vale zero quando:** toda junta com `|q̇| ≤ vmax` do regime; a rampa é 0 (tarefa sem célula aberta ou com `s = 0`); tarefa ANDAR (não registrada).

### 4.11 `sacudida`: variação brusca da velocidade da caixa

**Fórmula.** `Σ(Δv/(dt·norma_lin))² + Σ(Δω/(dt·norma_ang))²`, sem raiz, com `Δ` = velocidade da caixa na chamada atual menos a da chamada anterior (`recompensas.py:339-342`). O valor final é `custo × degrau/5`, só onde o alvo é P3 ou B1 (`recompensas.py:941-945`).

**Entradas:**
- `box.root_link_vel_w`, seis componentes (lineares e angulares, no mundo) (`recompensas.py:936`).
- `norma_lin`, `norma_ang` = `Recompensa.sacudida_norma` = (5,04; 31,56) (`knobs.py:358`).
- `dt` = `env.step_dt` = 0,02 s.
- A classe guarda `v_ant`, seis números por env. Ela se atualiza em **toda** chamada, mesmo fora da tarefa (`recompensas.py:928-939`). O gerente zera `v_ant` nos envs resetados: `reset` é chamado porque o termo é uma classe (`recompensas.py:931-932`, `mjlab/managers/reward_manager.py:112-113`, `:174-175`).
- Máscara `presa`: `v3_alvo_idx` em {P3, B1} (`_CAIXA_ALVO`, `recompensas.py:58`).
- Fator: `por_degrau(env, 0,0, 1,0)` = `degrau/5`.

**Tarefas.** PEGAR, CARREGAR, BOTAR (`env_cfg.py:52`).

**Máscaras.** A máscara de tarefa `_fim` e a máscara `presa` (alvo em P3 ou B1), as duas por `torch.where` (`recompensas.py:943-945`).

**Fatores.** `degrau/5` (`por_degrau(env, 0,0, 1,0)`, `recompensas.py:944`): o degrau de refino da tarefa do env. Não segue a `restricao`. Não tem fator só do ANDAR. Não usa `paga`.

**Vale zero quando:** o alvo não é P3 nem B1; o degrau da tarefa é 0; a velocidade da caixa não muda entre duas chamadas. No primeiro passo depois de um reset `v_ant` vale 0, e o Δ é a velocidade inteira.

### 4.12 `aperto`: força de aperto das palmas sobre a caixa

**Fórmula.** `min(F_esq, F_dir) / F_ref × degrau/5`, só onde o alvo é P3 ou B1 (`potencial.py:55-62`, `recompensas.py:948-957`).

**Entradas:**
- Sensor `caixa_faces`: primário a geom da caixa, secundário o subtree da pelve, 12 slots `maxforce`, quadro global (`cena.py:410-415`).
- `forcas_por_face(normal, força, achado, R_caixa)` (`recompensas.py:345-356`): para cada fatia, `n_c = R_caixaᵀ · normal` no frame da caixa; a face é o eixo de maior `|n_c|`. `F_n = |⟨força, normal⟩|`. `F_esq` soma as fatias com `achado`, face ±y e `n_c[y] > 0`; `F_dir` soma as com `n_c[y] < 0`. As faces ±x e ±z não entram. Uma fatia sem achado vale 0.
- `R_caixa` = `alvos.frame_da_caixa(quat da caixa, rumo da base)` (`alvos.py:109-128`).
- `F_ref` = `Recompensa.F_ref` = 49,0 N (`knobs.py:359`).
- Fator: `por_degrau(env, 0,0, 1,0)`. Máscara `presa`: alvo em {P3, B1}.

**Tarefas.** PEGAR, CARREGAR, BOTAR (`env_cfg.py:52`).

**Máscaras.** A máscara de tarefa `_fim` e a máscara `presa` (alvo em P3 ou B1), as duas por `torch.where` (`recompensas.py:956-957`).

**Fatores.** `degrau/5` (`por_degrau(env, 0,0, 1,0)`, `recompensas.py:955`). Não segue a `restricao`. Não tem fator só do ANDAR. Não usa `paga`.

**Vale zero quando:** o alvo não é P3 nem B1; o degrau é 0; uma das duas palmas não aperta nenhuma face ±y (o mínimo é 0). O custo é linear em `min(F_esq, F_dir)`, sem saturação.

### 4.13 `perda_caixa`

Ver seção 7.

### 4.14 `terminacao`

Ver seção 7.

### 4.15 `progresso`

Ver seção 6.

---

## 5. Contexto do `progresso`: a cadeia e os alvos

Esta seção descreve o que o `progresso` lê da cadeia de trechos e do alvo corrente. A seção 6 usa estes fatos.

### 5.1 A cadeia de trechos

Cada env roda uma fatia da cadeia (`_monta`, `comando.py:171-194`):

| fatia | trechos, em ordem |
|---|---|
| ANDAR | ANDAR por `Cadeia.episodio_s` = 20,0 s |
| B | ANDAR (espera), PEGAR (por aviso), CARREGAR (por tempo) |
| C | ANDAR (espera), PEGAR (por aviso), CARREGAR (por tempo), BOTAR (por aviso), ANDAR (final) |

A fatia de cada env é sorteada no reset do episódio, por `torch.multinomial` com as frações `Curriculo.fracoes_fatias` (`eventos.py:133-162`, `balanceador.py:282-292`, `balanceador.py:302-304`). A duração de um trecho por tempo, ou o atraso de um trecho por aviso, é sorteada uniforme na faixa `lo + (hi − lo)·U[0,1)` na abertura de cada trecho (`comando.py:578-579`). Valores: `Cadeia.espera_s` = (0,5; 1,0) s, `Cadeia.atraso_pegar_s` = (0,0; 1,0) s, `Cadeia.carregar_b_s` = (4,0; 7,0) s, `Cadeia.carregar_c_s` = (0,5; 1,0) s, `Cadeia.atraso_botar_s` = (0,0; 1,5) s, `Cadeia.andar_final_s` = 1,5 s (`knobs.py:154-160`). Um trecho por aviso só avança com o aviso e com o relógio do atraso (`comando.py:201-216`). O aviso do PEGAR é o fecho do P3 (`v3_pega_fechada`). O aviso do BOTAR é o fecho do B3 (`v3_bota_fechada`) (`comando.py:604-607`).

### 5.2 Como o alvo corrente avança

`CadeiaCmd._alvos` calcula, a cada passo, a condição `ok` do alvo corrente. `alvos.avanca` soma `dt` ao relógio do alvo enquanto `ok` vale e zera o relógio quando `ok` falha. O alvo avança quando o relógio chega ao tempo do alvo (`alvos.py:195-210`, `comando.py:482`). O índice passa a `índice + 1`. Depois do B3 vira −1.

O tempo de cada alvo é `[0, 0, tempo_p2, tempo_p3, tempo_b1, tempo_b2, tempo_b3]` em P0..B3 (`comando.py:442-443`). Com `z` = 0 os tempos são `Alvos.tempo_p2` = 0,3 s, `tempo_p3` = 0,5 s, `tempo_b1` = 0,3 s, `tempo_b2` = 0,5 s, `tempo_b3` = 0,5 s (`knobs.py:182`, `:189`, `:194`, `:197`, `:201`). A zona adaptativa afrouxa o fecho de cada etapa. Cada etapa tem um `z` em [0, 1] (`Curriculo.z`). O valor efetivo de um campo é `final + z·(largo − final)`, onde `final` é o campo de `Alvos` e `largo` é o terceiro item da tupla de `Alvos.zona` (`knobs.py:211-220`, `comando.py:433-441`). A cadeia refaz o `Alvos` efetivo (`env.v3_alvos`) e os tempos de permanência uma vez por iteração de currículo, quando `cur.it` muda (`comando.py:428-443`). Os fechos `ok_agora_pegar` e `ok_agora_botar` leem o efetivo (`comando.py:472-481`). O `progresso`, o Φ, a distância `d`, os portões e o disparo da perda leem os valores finais `K = ATIVO.alvos` (`recompensas.py:33`, `recompensas.py:545-546`, `recompensas.py:564-568`, `comando.py:407`).

Valores "largo" da zona (`knobs.py:211-220`): P0 e P1 `tol_palma` = 0,05; P2 `tempo_p2` = 0,2; P3 `subida_min_p3` = 0,01, `tol_transporte_p3` = 0,20, `tol_pelve_p3` = 0,10, `tempo_p3` = 0,2; B1 `raio_b1` = 0,20, `tempo_b1` = 0,1; B2 `b2_afasta` = 0,02, `tempo_b2` = 0,2; B3 `tol_pelve_b3` = 0,06, `tol_tronco_b3_graus` = 10,0, `tol_palmas_b3` = 0,10, `tempo_b3` = 0,2. O `apoio_b1_mg` não tem zona.

O `z` de uma etapa muda no fecho de iteração, quando a etapa acumulou `Balanceador.zona_n_min` = 200 visitas encerradas: `z ← clip(z − Balanceador.zona_ganho·(taxa − Balanceador.zona_t_alvo), 0, 1)`, com `zona_ganho` = 0,05 e `zona_t_alvo` = 0,6. A `taxa` é fechos por visita. Depois, `z[e] ← max(z[e], z[e−1] − Balanceador.zona_folga_cadeia)` para P1 a P3 e B2 a B3, com `zona_folga_cadeia` = 0,2. Uma visita sem fecho encerra depois de `Balanceador.zona_t_visita` = 8,0 s da abertura da etapa (`balanceador.py:169-184`, `comando.py:444-491`, `knobs.py:404-408`).
⚠ Divergência: o docstring de `alvos.avanca` diz "Alvo sem tempo (P2, tempo 0)" (`alvos.py:200`). O código dá tempo 0 ao P0 e ao P1, e `tempo_p2` ao P2 (`comando.py:442-443`).

Condições de fecho (`ok`), todas com `& (invariantes == 1,0)` (`comando.py:41-56`, `comando.py:73-88`). O invariante é o produto dos portões `desloc · incl · maos` com `maos = contatos/2` (`comando.py:463-466`).

| alvo | condição de fecho |
|---|---|
| P0, P1 | as duas palmas a `d ≤ tol_palma` da pose-alvo no frame da caixa |
| P2 | as duas palmas em contato com a caixa (sensores `palma_E` e `palma_D`) |
| P3 | `subida ≥ subida_min_p3` E `d_transporte ≤ tol_transporte_p3` E `d_pelve ≤ tol_pelve_p3` E a caixa sem contato com a laje |
| B1 | `‖caixa − alvo‖ ≤ raio_b1` E `apoio_N ≥ apoio_b1_mg · peso_N` E `‖v_caixa‖ < v_parada` |
| B2 | as duas palmas afastadas `≥ b2_afasta` da face E a caixa apoiada E `‖v_caixa‖ < v_parada` |
| B3 | `|Δpelve| ≤ tol_pelve_b3` E `|Δtronco| ≤ tol_tronco_b3_graus` E `dpalmas ≤ tol_palmas_b3` |

Valores finais (z = 0): `tol_palma` = 0,03 m, `subida_min_p3` = 0,03 m, `tol_transporte_p3` = 0,05 m, `tol_pelve_p3` = 0,03 m, `raio_b1` = 0,10 m, `apoio_b1_mg` = 0,8, `v_parada` = 0,05 m/s, `b2_afasta` = 0,05 m, `tol_pelve_b3` = 0,03 m, `tol_tronco_b3_graus` = 5,0 graus, `tol_palmas_b3` = 0,05 m (`knobs.py:183`, `:186-188`, `:191-192`, `:195-196`, `:198-200`).

Efeitos no fecho:
- Fecho do P2: `v3_pegou` passa a True (`comando.py:497`).
- Fecho do P3: `v3_pega_fechada` passa a True e o índice **não** avança: continua em P3 (`comando.py:492-493`, `comando.py:496`). O PEGAR acaba pelo aviso.
- Fecho do B1: a posição da caixa é gravada em `v3_ponto_apoio` (`comando.py:494-495`).
- Fecho do B3: `v3_bota_fechada` passa a True (`comando.py:498`).
- Troca de alvo (fecho de um alvo que não é o P3 e que tem próximo alvo, `novo ≥ 0`; o fecho do B3 deixa o índice em −1 e não recalcula nada): `v3_sigma` é recalculado, `v3_h` volta a 1,0 e `v3_phi_melhor` a 0,0 (`comando.py:502-504`).

### 5.3 Abertura do alvo e σ

O σ do alvo (`v3_sigma`) é a distância `d` do alvo no instante de abertura, com piso por etapa: `σ = max(d, sigma_min_etapa[alvo])` (`comando.py:359-362`). `Alvos.sigma_min_etapa` = (0,05; 0,05; 0,08; 0,20; 0,20; 0,05; 0,0) para P0, P1, P2, P3, B1, B2, B3 (`knobs.py:208`).

Quando `v3_sigma`, `v3_h` e `v3_phi_melhor` são reinicializados:

| evento | código | `v3_alvo_idx` | `v3_sigma` | `v3_h`, `v3_phi_melhor` |
|---|---|---|---|---|
| reset do episódio | `comando.py:651-657` | (cada trecho escreve) | 1,0 | 1,0 e 0,0 |
| abertura de trecho | `comando.py:568-594` | −1 em ANDAR e PEGAR | (não muda) | 1,0 e 0,0 |
| P0 abre (primeira leitura válida da caixa) | `comando.py:370-392` | P0 | `max(d(P0), 0,05)` | 1,0 e 0,0 |
| troca de alvo | `comando.py:502-504` | índice + 1 | `max(d(novo), piso)` | 1,0 e 0,0 |
| abertura do BOTAR | `comando.py:542-566` | B1 | `max(d(B1), 0,20)` | 1,0 e 0,0 |
| caixa perdida | `comando.py:412-420` | B3 | `max(d(B3), 0,0)` | 1,0 e 0,0 |

O P0 só abre quando existe a primeira leitura válida da caixa (idade menor que `Estimador.idade_max`). Até lá o PEGAR fica com alvo −1 (`comando.py:364-392`).

---

## 6. O `progresso`

### 6.1 Fórmula

`progresso = renda_etapa · etapa · h + renda_phi · Φ_pago + renda_conquista · conquistas` (`recompensas.py:286-291`, chamada em `recompensas.py:879-882`).

| símbolo | significado | knob e valor |
|---|---|---|
| `renda_etapa` | taxa por unidade de etapa | `Recompensa.renda_etapa` = 6,0 (`knobs.py:339`) |
| `renda_phi` | taxa do potencial | `Recompensa.renda_phi` = 3,0 (`knobs.py:338`) |
| `renda_conquista` | taxa das conquistas | `Recompensa.renda_conquista` = 2,0 (`knobs.py:343`) |
| `h` | fator de pressa em [0, 1], só na parcela da etapa | seção 6.5 |
| `Φ_pago` | potencial do alvo corrente, ou 0 | seções 6.3 e 6.4 |
| `conquistas` | soma das três conquistas mantidas | seção 6.6 |

O valor é uma taxa por segundo. O peso do cfg é +1,0 e o mjlab multiplica por `dt` = 0,02 s (seção 2). O termo também escreve três buffers como efeito colateral, em todo env e a cada chamada, sem máscara de tarefa: `v3_h`, `v3_phi_melhor` e `v3_phi` (`recompensas.py:868-869`, `recompensas.py:883`). O `v3_phi` guarda o Φ cru do alvo corrente (−1 sem alvo).

### 6.2 A etapa

A etapa é um inteiro que escolhe a parcela `renda_etapa · etapa · h`. Precedência (a primeira condição que vale vence) (`recompensas.py:850-854`):

| ordem | condição | etapa | Φ pago |
|---|---|---|---|
| 1 | `v3_perdeu` OU (`~v3_pega_fechada` E tarefa ≠ PEGAR) (variável `sem_caixa`) | 0 | 0 |
| 2 | `v3_bota_fechada` | 7 | 0 |
| 3 | `v3_pega_fechada` E `v3_alvo_idx < B1` (variável `fecho_pega`) | 4 | 0 |
| 4 | `v3_alvo_idx ≥ 0` (P0 a B3: etapa = índice) | 0 a 6 | Φ do alvo |
| 5 | `v3_alvo_idx = −1` | 0 | 0 |

(`etapa = where(sem_caixa, 0, where(bota_fechada, 7, where(fecho_pega, 4, idx.clamp(min=0))))` e `paga_phi = ~(sem_caixa | bota_fechada | fecho_pega)`, `recompensas.py:851-854`.)

Pontos que decorrem da condição:
- `v3_pega_fechada` e `v3_bota_fechada` só voltam a False no reset do episódio (`comando.py:651-653`). Depois do fecho do P3 o `fecho_pega` vale no PEGAR (índice P3) e no CARREGAR (índice P3). No BOTAR o índice é B1 ou maior, e a condição `alvo < B1` falha: `fecho_pega` vale False.
- A tabela da etapa por alvo: P0 = 0, P1 = 1, P2 = 2, P3 = 3, B1 = 4, B2 = 5, B3 = 6. Depois do fecho do P3: 4. Depois do fecho do B3: 7.
- A perda da caixa leva o env à tarefa ANDAR com alvo B3 e `v3_perdeu` True. Pela linha 1, a etapa e o Φ valem 0 até o reset (`comando.py:412-420`).

### 6.3 O potencial Φ

O Φ de um alvo mede a proximidade ao alvo corrente, em [0, 1]. O valor vem de `_phi_env` (`recompensas.py:519-569`), `phi_do_alvo` (`recompensas.py:274-283`) e `potencial.phi` (`potencial.py:32-37`).

**Fórmula:** `Φ = (1 − tanh(d / (largura_tanh · σ))) · fator · Π portões`. Vale 0 com `v3_alvo_idx < 0`.
- `largura_tanh` = `Alvos.largura_tanh` = 2,0 (`knobs.py:221`).
- `σ` = `v3_sigma` (seção 5.3), com piso de `_EPS` = 1e-6 só no denominador (`recompensas.py:281`).
- O `progresso` lê os valores finais de `Alvos` (`K`), e não os valores afrouxados pela zona.

**Distância `d` por alvo** (`distancia_do_alvo`, `recompensas.py:224-244`):

| alvo | `d` |
|---|---|
| P0, P1, P2, B2 | média, sobre as duas palmas, de `‖palma − palma_alvo‖` no frame da caixa |
| P3 | `hypot(‖caixa − alvo_P3‖, Δpelve · tol_transporte_p3/tol_pelve_p3)` |
| B1 | `hypot(‖caixa − alvo_B1‖, escala_apoio_b1 · déficit_apoio)` |
| B3 | `√((Δpelve/tol_pelve_b3)² + (Δtronco/tol_tronco_b3_graus)² + (dpalmas/tol_palmas_b3)²)` |
| −1 | 0 |

Definições:
- `palma_alvo` (`alvos.py:138-154`): palma 0 em +y, palma 1 em −y, `z = 0`. P0: `x = −(a + p0_frente)`; P1 e P2: `x = 0`; B2: `x = 0`. `y = ±(a + palma_face + folga)`, com `folga` = `folga_p1` no P0 e no P1, 0 no P2 e `b2_afasta` no B2. `a` é a meia aresta da caixa (`v3_meia`).
- `alvo_P3` (`alvos.py:163-179`): `pelve_xy + R(rumo) · caixa_transporte_b(a)[:2]`, com `z = chão + z_transporte`. O chão é `z da pelve − altura_pelve.heights` (`recompensas.py:487-493`).
- `alvo_B1` = `v3_alvo_w`, o ponto de soltura (`alvos.py:173-179`).
- `Δpelve` = `altura_pelve − ref_do_alvo[:, 6]` (m). `Δtronco` = `incl_torso − rad2deg(ref_do_alvo[:, 7])` (graus) (`recompensas.py:536`, `recompensas.py:546`).
- `déficit_apoio` = `relu(1 − apoio_N / (apoio_b1_mg · peso_N))`. `apoio_N` é a soma de `|Fz|` do sensor `apoio_caixa`. `peso_N` = `v3_massa · g`, com `g` lido de `env.sim.mj_model.opt.gravity[2]` (`recompensas.py:542-544`).
- `dpalmas` = a maior das duas distâncias palma–`palmas_de_pe_b`, no frame da base (`recompensas.py:539-541`).
- Valores: `p0_frente` = 0,05 m, `palma_face` = 0,045 m, `folga_p1` = 0,02 m, `b2_afasta` = 0,05 m, `z_transporte` = 0,90 m, `tol_transporte_p3` = 0,05 m, `tol_pelve_p3` = 0,03 m, `escala_apoio_b1` = 0,10 m, `apoio_b1_mg` = 0,8, `tol_pelve_b3` = 0,03 m, `tol_tronco_b3_graus` = 5,0 graus, `tol_palmas_b3` = 0,05 m (`knobs.py:175`, `:178-179`, `:193`, `:196`, `:187-188`, `:192`, `:198-202`).

**Portões** (`portoes_do_alvo`, `recompensas.py:247-271`). A rampa é `rampa(x, lim, Δ) = clamp(1 − (x − lim)/Δ, 0, 1)` (`potencial.py:23-29`): vale 1 abaixo de `lim` e cai linear a 0 em `lim + Δ`. Cada portão está em [0, 1] e vale 1 fora do conjunto de alvos em que age.

| portão | alvos | argumento `x`, `lim`, `Δ` |
|---|---|---|
| `desloc` | P0, P1, P2 | `x = ‖xy da caixa − v3_caixa_pega_w‖`, `lim = desloc_p0p2_frac · a`, `Δ = delta_p0p2_frac · a` |
| `desloc` | B2, B3 | `x = ‖xy da caixa − v3_ponto_apoio‖` (0 com `v3_perdeu`), `lim = desloc_b2b4`, `Δ = delta_b2b4` |
| `incl` | P2, P3, B1, B2 | `x` = inclinação da caixa (graus), `lim = incl_lim`, `Δ = delta_incl_graus` |
| `maos` | P3, B1 | vale 1 (ver abaixo) |
| `tronco` | todos menos B3 | `x = relu(incl_torso − tronco_ref)`, `lim = faixa_tronco_graus`, `Δ = delta_tronco_graus` |
| `recuo` | todos menos B3 | `x = ‖recuo − recuo_ref‖`, `lim = faixa_recuo`, `Δ = delta_recuo` |
| `com` | com dois pés em contato | `x = relu(faixa_com − margem_com)`, `lim = 0`, `Δ = delta_com` |

Valores: `desloc_p0p2_frac` = 1,0, `delta_p0p2_frac` = 0,5, `desloc_b2b4` = 0,02 m, `delta_b2b4` = 0,06 m, `delta_incl_graus` = 15,0 graus, `faixa_tronco_graus` = 15,0 graus, `delta_tronco_graus` = 15,0 graus, `faixa_recuo` = 0,08 m, `delta_recuo` = 0,30 m, `faixa_com` = 0,03 m, `delta_com` = 0,03 m (`knobs.py:222-231`, `:205`).
- `incl_lim` = `por_degrau(env, incl_inicio_graus, incl_fim_graus)` = 30,0 + (15,0 − 30,0)·degrau/5 graus, de 30,0 (degrau 0) a 15,0 (degrau 5) (`Alvos.incl_inicio_graus` = 30,0 e `incl_fim_graus` = 15,0, `knobs.py:203-204`, `recompensas.py:462-464`).
- `recuo` mede `rec = ‖xy caixa − xy pelve‖` contra `_recuo_ref` (seção 4.7) (`recompensas.py:531`, `recompensas.py:567`).
- `com` usa `margem_com` da seção 4.4 e vale 1 sem dois pés (`recompensas.py:270`).
- ⚠ Divergência: o docstring de `portoes_do_alvo` e o comentário de `_phi_env` descrevem o portão `maos` como a firmeza da pega (`recompensas.py:254-256`, `recompensas.py:555-556`). A chamada passa `torch.ones_like(maos)` (`recompensas.py:566`), então o portão `maos` vale 1 no `progresso`. A firmeza da pega entra como conquista (seção 6.6).

**Fator de pouso.** O fator vale 1 em todos os alvos, menos no B1 (`recompensas.py:551`).
⚠ Divergência: o docstring de `fator_pouso` diz que ele multiplica o B1 e o B2 (`potencial.py:48`); o código o aplica só ao B1 (`_POUSO`, `recompensas.py:57`, `recompensas.py:551`).
No B1 é `fator_pouso(vz_caixa, fundo, σ_pouso, altura_pouso, altura_pouso_inicio)` (`potencial.py:40-52`):
`1 − s · (1 − exp(−(vz/σ_pouso)²))`, com `s = clip((altura_pouso_inicio − fundo)/(altura_pouso_inicio − altura_pouso), 0, 1)`.
- `fundo` = `z da caixa − a − (origem z do env + v3_topo_destino)` (`recompensas.py:548`).
- `σ_pouso` = `por_degrau(env, sigma_pouso[0], sigma_pouso[1])` = 1,0 + (0,35 − 1,0)·degrau/5 m/s (`Recompensa.sigma_pouso` = (1,0; 0,35), `knobs.py:368`).
- `altura_pouso` = 0,05 m e `altura_pouso_inicio` = 0,10 m (`knobs.py:369-370`). O fator vale 1 com `fundo ≥ 0,10` m e entra pleno com `fundo ≤ 0,05` m.

**Φ do P3 pago.** No alvo P3 o `progresso` usa `Φ_c = Φ · c2`, com `c2` o peso nas mãos vezes a pega firme (seção 6.6). O `v3_phi` grava o Φ cru (`recompensas.py:850`, `recompensas.py:883`).

### 6.4 Valor por segundo em cada etapa

Taxa = `6,0 · etapa · h + 3,0 · Φ_pago + 2,0 · conquistas` (todas as taxas em 1/s, antes de × 0,02 s). As colunas máximas supõem `h = 1`, `Φ = 1` e cada conquista = 1 (`recompensas.py:875-882`).

| situação | etapa | `6,0 · etapa · h` | `3,0 · Φ_pago` | `2,0 · conquistas` | máximo |
|---|---|---|---|---|---|
| alvo P0 | 0 | 0 | `3,0 · Φ` | 0 | 3,0 |
| alvo P1 | 1 | `6,0 · h` | `3,0 · Φ` | 0 | 9,0 |
| alvo P2 | 2 | `12,0 · h` | `3,0 · Φ` | `2,0 · (c0 + c1 · almofadas)` | 19,0 |
| alvo P3 (antes do fecho) | 3 | `18,0 · h` | `3,0 · Φ · c2` | `2,0 · (c0 + c1 + c2)` | 27,0 |
| P3 fechado (PEGAR depois do fecho, CARREGAR) | 4 | `24,0 · h_ef` | 0 | `2,0 · (c0 + c1 + c2)` | 30,0 |
| alvo B1 | 4 | `24,0 · h` | `3,0 · Φ` | `2,0 · (c0 + c1)` | 31,0 |
| alvo B2 | 5 | `30,0 · h` | `3,0 · Φ` | 0 | 33,0 |
| alvo B3 | 6 | `36,0 · h` | `3,0 · Φ` | 0 | 39,0 |
| B3 fechado (`v3_bota_fechada`) | 7 | 42,0 | 0 | 0 | 42,0 |
| `v3_perdeu`, ou sem pega fechada fora do PEGAR, ou PEGAR sem alvo | 0 | 0 | 0 | 0 | 0 |

`h_ef` = `v3_h` quando o env está pagando Φ (`com_alvo`) ou no CARREGAR com pega fechada (`carrega`); vale 1 nos outros casos (`recompensas.py:870`). No PEGAR depois do fecho do P3, `h_ef` = 1. As conquistas c0, c1 e c2 estão na seção 6.6. A conta `etapa` usa a variável `etapa.float()` (`recompensas.py:879`).

### 6.5 O fator `h` (pressa)

`h` está em [0, 1] e multiplica só a parcela da etapa. Ele cai quando o Φ não melhora e volta a subir quando o Φ passa do melhor valor da etapa (`recompensas.py:856-870`).

- **Com alvo e Φ pago** (`com_alvo = paga_phi & (idx ≥ 0)`): `h ← min(1, h · exp(−dt/τ) + recarga · relu(Φ_c − v3_phi_melhor))` (`n_pressa`, `recompensas.py:294-296`, `recompensas.py:858`). `τ` = `Recompensa.tau_renda` = 3,0 s; `recarga` = `Recompensa.recarga_renda` = 5,0 (`knobs.py:347-348`). Por passo, o `h` é multiplicado por `exp(−0,02/3,0)`.
  - `v3_phi_melhor ← max(v3_phi_melhor, Φ_c)` (`recompensas.py:869`).
  - `Φ_c` é o Φ do alvo, e é `Φ · c2` no alvo P3 (`recompensas.py:850`).
- **No CARREGAR com pega fechada** (`carrega = fecho_pega & ~sem_caixa & (tarefa == CARREGAR)`): `h ← h + (dt/τ) · (sinal − h)`, uma média móvel de `sinal` com `τ` = 3,0 s (`recompensas.py:857`, `recompensas.py:867`).
  - `sinal`, com `com_lin = ‖cmd_xy‖ ≥ limiar_canal` e `com_ang = |cmd_z| ≥ limiar_canal`:
    - com comando: `g · (com_lin · clamp(r_lin, 0, 1) + com_ang · clamp(r_ang, 0, 1)) / max(com_lin + com_ang, 1)`;
    - sem comando nos dois canais: `clamp((1 + r_lin)(1 + r_ang), 0, 1)`;
    - em ambos os casos, multiplicado por `rampa(d_transporte, 0,05; 0,25)`, a "qualidade larga" (`recompensas.py:859-866`).
  - `g` = `twist.g_marcha` em [0, 1], o acerto filtrado do relógio de marcha (`comando.py:871`). Conta: a cada passo, depois das recompensas, o acerto do passo é a média sobre os dois pés de `contato == apoio` (`recompensas.py:75-82`, `comando.py:868`); o filtro `q̄` recebe `q̄ += (1 − exp(−dt/τ_f))·(acerto − q̄)` com `τ_f` = `Comando.tau_fracao` = 0,4 s (`knobs.py:136`, `comando.py:870`); `g = clamp(2·q̄ − 1, 0, 1)` (`comando.py:871`). `q̄` vale 0,5 (g = 0) no reset do episódio e no passo em que o comando passa de parado a ligado (`comando.py:869`, `comando.py:887`). O comando está "ligado" (`on`) quando `‖comando‖ > Comando.limiar_marcha` = 0,1 (`knobs.py:141`, `comando.py:863`); o relógio de fase avança só com `on`, com período `Comando.periodo_marcha` = 0,8 s e fração de apoio `Comando.fase_apoio` = 0,55 (`knobs.py:139-140`, `comando.py:872-873`). Como o `progresso` roda antes, ele lê o `g` do passo anterior.
  - `r_lin` e `r_ang` são o rastreio normalizado (6.5.1).
  - `d_transporte` é a distância da caixa ao alvo de transporte (6.5.2).
  - `limiar_canal` = `Comando.limiar_canal` = 0,02 (`knobs.py:138`). `qualidade_larga` = `Recompensa.qualidade_larga` = (0,05; 0,25): vale 1 até 0,05 m e 0 a 0,30 m do alvo de transporte (`knobs.py:351`).
- **Nos outros casos** `h` não muda (`recompensas.py:868`).
- `h` volta a 1,0 e `v3_phi_melhor` a 0,0 nos eventos da tabela da seção 5.3.

#### 6.5.1 Rastreio normalizado

`_rastreio(env, linear, normalizado=True)` chama `potencial.rastreio_normalizado(erro2, c, v2, σ, σ₀, limiar, piso)` (`recompensas.py:374-397`, `potencial.py:88-106`).
- Linear: `erro2 = ‖cmd_xy − v_xy‖²`, `c = ‖cmd_xy‖`, `v2 = ‖v_xy‖²`, `σ₀` = `Recompensa.sigma_zero_lin` = 0,10 m/s (`knobs.py:241`). As velocidades são da base, no frame da base (`root_link_lin_vel_b`).
- Angular: `erro2 = (cmd_z − ω_z)²`, `c = |cmd_z|`, `v2 = ω_z²`, `σ₀ = no_degrau(sigma_zero_ang_inicio, sigma_zero_ang, restricao, 1)` = 0,5 + (0,15 − 0,5)·`restricao` (`knobs.py:242-243`). `Recompensa.sigma_zero_ang_inicio` = 0,5 e `sigma_zero_ang` = 0,15.
- `σ` = `Recompensa.sigma_rastreio` = 0,5 (`knobs.py:238`).
- Com `c ≥ limiar`: `(expm1(−erro2/σ²) − expm1(−c_n²/σ²)) / (−expm1(−c_n²/σ²))`, com `c_n` = `max(c, limiar)` (`potencial.py:101-104`). Com `c ≥ limiar`, `c_n` = `c`.
- Com `c < limiar`: `exp(−v2/σ₀²) − 1`.
- O resultado é limitado por baixo em `Recompensa.piso_rastreio` = −1,0 (`knobs.py:239`).

#### 6.5.2 Distância de transporte

`_qualidade(env)[1]` (`recompensas.py:572-587`) é a distância `‖posição da caixa − alvo‖`. O alvo é `alvo_da_caixa_w(env, P3)`. Na célula `terreno` do CARREGAR (`v3_celula == id_celula("CARREGAR", "terreno")`), o alvo é a pelve mais `caixa_transporte_b`, com o xy girado pelo yaw da pelve (`alvos.base_para_mundo`) e `z = z_pelve + tb_z` (`recompensas.py:577-583`).

### 6.6 As conquistas

Há três conquistas, cada uma em [0, 1], calculadas por `_phi_env` em todo env (`recompensas.py:557-563`):

| conquista | fórmula | knobs |
|---|---|---|
| `c0` palmas nas faces | `1 − tanh(d_faces / (largura_tanh · sigma_palmas))`, com `d_faces` a média das duas distâncias palma–`palma_alvo(P2)` no frame da caixa | `Recompensa.sigma_palmas` = 0,08 m (`knobs.py:344`); `Alvos.largura_tanh` = 2,0 |
| `c1` pega firme (`maos`) | `tanh(min(F_esq, F_dir) / F_pega)`, com `F_pega = margem_pega · v3_massa · g / (2 · atrito_pad[0])` | `Recompensa.margem_pega` = 2,5 (`knobs.py:365`); `DR.atrito_pad[0]` = 0,6 (`knobs.py:523`) |
| `c2` peso nas mãos × pega firme | `clamp(1 − apoio_N/peso_N, 0, 1) · c1` | `apoio_N` e `peso_N` como em 6.3 |

`F_esq` e `F_dir` vêm de `forcas_por_face` (seção 4.12).

O `progresso` paga `2,0 · (c0 · [mantém] + c1 · almofadas · [mantém] + c2 · com_peso · [mantém])` (`recompensas.py:874-881`):
- `mantém` = `~sem_caixa & ~bota_fechada & (fecho_pega | alvo em {P3, B1} | alvo == P2)`. Fora disso as conquistas valem 0.
- `almofadas` = a fração das duas almofadas de palma (`palma_E`, `palma_D`) em contato com a caixa (`found > 0`), média das duas, no alvo P2; 1 nos outros alvos (`recompensas.py:482-484`, `recompensas.py:876`).
- `com_peso` = 1 no alvo P3 e depois do fecho do P3 (`no_p3 | fecho_pega`); 0 nos outros (`recompensas.py:877`).

### 6.7 Soma por tarefa

O valor total do `progresso` em cada tarefa é a linha da tabela 6.4 que corresponde ao estado do env. A tabela abaixo diz quais linhas cada tarefa visita.

| tarefa | linhas visitadas | valor |
|---|---|---|
| ANDAR (fatia ANDAR, e o ANDAR de espera das fatias B e C) | sem pega fechada | 0 |
| ANDAR (final da fatia C, depois do B3 fechado) | B3 fechado | 42,0 |
| ANDAR com `v3_perdeu` (caixa perdida) | perdeu | 0 |
| PEGAR | sem alvo (espera), P0, P1, P2, P3, P3 fechado | 0; 0 a 3; até 9; até 19; até 27; até 30 |
| CARREGAR | P3 fechado (alvo fica em P3); perdeu | até 30 (com `h` pela média móvel); 0 |
| BOTAR | B1, B2, B3, B3 fechado; perdeu | até 31; até 33; até 39; 42,0; 0 |

O ANDAR final só abre depois do fecho do B3, porque o trecho BOTAR avança por aviso (`comando.py:604-607`, `comando.py:201-216`).

---

## 7. A perda da caixa e a terminação

### 7.1 `perda_caixa`: valor por evento e quando dispara

**Fórmula.** `v3_perdeu_agora · Recompensa.perda_caixa` = `perdeu_agora · 240,0` (`recompensas.py:311-313`, `recompensas.py:960-962`). O peso do cfg é −1,0 (`env_cfg.py:57`). Por evento, a recompensa é `240,0 × (−1,0) × 0,02` = −4,8 em um único passo. A taxa por segundo é −240,0.

**Tarefas.** As quatro (`env_cfg.py:53`). A máscara é lida no passo em que o termo cobra, e a tarefa nesse passo já é ANDAR (a perda converte o trecho em ANDAR). Por isso o registro precisa incluir o ANDAR.

**Quando `v3_perdeu_agora` vale True.** `CadeiaCmd._perda`, a cada passo, no `compute` da cadeia (depois das recompensas do passo) (`comando.py:394-420`, `comando.py:596-599`):

`perda = (caiu | arremesso | fora_da_laje) & (tarefa ≠ ANDAR) & (episode_length_buf > 0)` (`comando.py:409-410`), com (`comando.py:59-65`):
- `caiu`: `(altura_caixa.heights − v3_meia) < Terminacao.caixa_folga_chao` = 0,02 m (`knobs.py:379`, `comando.py:409`). A altura vem do sensor `altura_caixa`, que atualiza no `sim.sense()` e, por isso, está um passo atrasada.
- `arremesso`: `v3_pegou & (v_rel > Terminacao.v_solta) & fora_do_alvo`. `v_rel` = `‖v_caixa − v_base‖` (m/s). `Terminacao.v_solta` = 1,2 m/s (`knobs.py:380`, `comando.py:337`). `fora_do_alvo` é, no BOTAR, `‖caixa − v3_alvo_w‖ > Alvos.raio_b1` = 0,10 m, e é True nas outras tarefas (`comando.py:407-408`).
- `fora_da_laje`: `v3_alvo_idx` em [P0, P2] e a caixa fora do topo da laje. "Sobre a laje" é `|x| ≤ Cena.prateleira_meia_xy` e `|y| ≤ Cena.prateleira_meia_xy` da caixa no frame da laje, com `prateleira_meia_xy` = 0,30 m (`knobs.py:89`, `comando.py:334-335`).

**Efeito.** Com `Cadeia.caixa_perdida_vira_andar` = True (`knobs.py:161`), a perda marca `v3_perdeu`, leva o alvo a B3, reinicia `v3_sigma`, `v3_h` e `v3_phi_melhor` e publica a tarefa ANDAR (`comando.py:412-420`). O `v3_perdeu_agora` é um pulso: o `compute` o reescreve a cada passo (`comando.py:411`). O reset do episódio também o zera (`comando.py:651-653`).

**Vale zero quando:** não há perda neste passo; no primeiro passo do episódio (`episode_length_buf == 0`); a tarefa era ANDAR no passo da detecção.

### 7.2 `terminacao`: valor por evento e quando dispara

**Fórmula.** `terminated & ~time_outs`, convertido a 1,0 (`recompensas.py:214-216`, `recompensas.py:965-968`). O peso é `Recompensa.terminacao` = −300,0 (`knobs.py:366`). Por evento a recompensa é `1,0 × (−300,0) × 0,02` = −6,0 em um passo. A taxa por segundo é −300,0.
⚠ Divergência: o docstring de `n_terminacao` diz "Peso −200 no knob" (`recompensas.py:215`). O knob vale −300,0.

**Tarefas.** As quatro (`env_cfg.py:53`).

**Fonte de `terminated` e `time_outs`.** O gerente de terminação do mjlab roda `term_cfg.func` de cada termo. Um termo `time_out=True` soma em `time_outs`; os outros somam em `terminated` (`mjlab/managers/termination_manager.py:102-113`). O termo `terminacao` cobra quando um termo que não é `time_out` vale True e nenhum `time_out` vale True no mesmo passo (`recompensas.py:216`). As terminações do cfg (`env_cfg.py:137-147`):

| termo | tipo | tarefas | condição exata |
|---|---|---|---|
| `time_out` (`fim_da_cadeia`) | `time_out` | (sem máscara) | `episode_length_buf ≥ max_episode_length` OU `v3_fim` OU `atravessou` (`terminacoes.py:58-72`). `atravessou` (27/11, T19): `v3_fatia == ANDAR` E coluna de terreno ≠ 0 E a distância xy de `twist.pose_ult` à origem da célula `> Terreno.dist_travessia` (3,5 m) E `episode_length_buf > 2`. O término não leva penalidade (`time_out`) e sai um passo depois da travessia, porque o `pose_ult` é o do passo anterior (`terminacoes.py:69-72`, `knobs.py:442`). `max_episode_length` = `ceil(Cadeia.episodio_s / step_dt)` = ceil(20,0 / 0,02) = 1000 passos (`knobs.py:160`, `env_cfg.py:105`, `mjlab/envs/manager_based_rl_env.py:281-283`). `v3_fim` vira True quando o último trecho da cadeia termina (`comando.py:213-214`, `:614`) |
| `queda` | terminação | as quatro | ANDAR ou CARREGAR: `incl > Terminacao.incl_marcha` OU `(z_pelve − z_chão_pelve) < Terminacao.base_min`. PEGAR ou BOTAR: `incl > rad(Terminacao.incl_manip_graus)` OU `(z_joelho_mais_baixo − z_chão_joelho) < Terminacao.joelho_min` (`terminacoes.py:35-50`, `terminacoes.py:86-112`) |
| `caixa_perdida` | terminação | PEGAR, CARREGAR, BOTAR | `v3_perdeu_agora & (não vira_andar)` (`terminacoes.py:75-83`) |
| `chegada` | terminação | BOTAR | primeiro contato da caixa com a laje E `−v3_vz_caixa_ant > por_degrau(env, 1,0, 0,25)` (`terminacoes.py:128-151`) |
| `batente` | terminação | as quatro | `(alguma junta com curso > batente_frac OU algum \|hip_yaw\| > hip_yaw_max) E v3_degrau ≥ Refino.degrau_batente` (`terminacoes.py:154-182`) |

Valores: `Terminacao.incl_marcha` = 0,8 rad, `base_min` = 0,20 m, `incl_manip_graus` = 70,0 graus, `joelho_min` = 0,05 m, `chegada` = (1,0; 0,25) m/s, `hip_yaw_max` = 1,0 rad, `batente_frac` = 1,0 (`knobs.py:375-383`); `Refino.degrau_batente` = 1 (`knobs.py:547`).
- Com `Cadeia.caixa_perdida_vira_andar` = True a terminação `caixa_perdida` nunca dispara.
- A inclinação `incl` é `acos(−projected_gravity_b[:, 2])`. O `curso` de uma junta é `|q − centro|/meio` dos limites do modelo (`terminacoes.py:97`, `terminacoes.py:162`, `potencial.py:70-75`).
- As alturas são medidas contra o terreno, pelos sensores `altura_pelve` e `altura_joelhos` (`terminacoes.py:99-107`).
- A `chegada` limita a velocidade de descida da caixa: o limite vai de 1,0 m/s (degrau 0) a 0,25 m/s (degrau 5).

**Vale zero quando:** o episódio não terminou neste passo; o episódio terminou só por `time_out`; terminou por `time_out` no mesmo passo de uma terminação (o termo `& ~time_outs` zera).

---

## 8. Knobs de `Recompensa` que estes termos leem

| knob | valor | lido por |
|---|---|---|
| `cerca_sola` | −0,05475 (`knobs.py:270`) | peso de `cerca_sola` |
| `cerca_giro` | −0,03965 (`knobs.py:271`) | peso de `cerca_giro` |
| `cerca_tronco` | −0,04596 (`knobs.py:272`) | peso de `cerca_tronco` |
| `cerca_com` | −22,09035 (`knobs.py:273`) | peso de `cerca_com` |
| `cerca_torque` | −0,29615 (`knobs.py:274`) | peso de `cerca_torque` |
| `cerca_pelve` | −4,45279 (`knobs.py:275`) | peso de `cerca_pelve` |
| `cerca_recuo` | −10,39719 (`knobs.py:276`) | peso de `cerca_recuo` |
| `lim_sola_graus` | 5,0 (`knobs.py:277`) | `cerca_sola` |
| `lim_giro_graus` | 30,0 (`knobs.py:278`) | `cerca_giro` |
| `lim_tronco_graus` | 15,0 (`knobs.py:279`) | `cerca_tronco` |
| `lim_com` | 0,03 (`knobs.py:280`) | `cerca_com` |
| `lim_torque` | 0,8 (`knobs.py:281`) | `cerca_torque` |
| `banda_torque` | 0,2 (`knobs.py:282`) | `cerca_torque` |
| `lim_pelve` | 0,08 (`knobs.py:283`) | `cerca_pelve` |
| `lim_recuo` | 0,08 (`knobs.py:284`) | `cerca_recuo` |
| `tau_max` | tabela de 8 regex (`knobs.py:286-292`) | `cerca_torque`, via `TAU_MAX` (`knobs.py:572`) |
| `pares_torque` | 3 pares (`knobs.py:293-297`) | `cerca_torque` |
| `tau_par` | 50,0 (`knobs.py:298`) | `cerca_torque` |
| `contato_fora` | −1,0 (`knobs.py:299`) | peso de `contato_fora` |
| `contato_fora_N` | 1,0 (`knobs.py:300`) | `contato_fora` |
| `contato_laje` | −2,0 (`knobs.py:301`) | peso de `contato_laje` |
| `laje_joelho_N` | (50,0; 10,0) (`knobs.py:302`) | `contato_laje` |
| `laje_saturacao_N` | 100,0 (`knobs.py:303`) | `contato_laje` |
| `freio` | −6/29 (`knobs.py:304`) | peso de `freio` |
| `freio_fator` | 1,5 (`knobs.py:305`) | `freio` |
| `freio_limiar_andar` | 0,05 (`knobs.py:308`) | `freio` |
| `freio_limiar_correr` | 1,5 (`knobs.py:309`) | `freio` |
| `vel_max_parado`, `vel_max_andando`, `vel_max_correndo` | tabelas (`knobs.py:311-331`) | `freio` |
| `renda_phi` | 3,0 (`knobs.py:338`) | `progresso` |
| `renda_etapa` | 6,0 (`knobs.py:339`) | `progresso` |
| `renda_conquista` | 2,0 (`knobs.py:343`) | `progresso` |
| `sigma_palmas` | 0,08 (`knobs.py:344`) | `progresso` (conquista `c0`) |
| `tau_renda` | 3,0 (`knobs.py:347`) | `progresso` (`h`) |
| `recarga_renda` | 5,0 (`knobs.py:348`) | `progresso` (`h`) |
| `qualidade_larga` | (0,05; 0,25) (`knobs.py:351`) | `progresso` (`h` no CARREGAR) |
| `sacudida` | −0,05 (`knobs.py:353`) | peso de `sacudida` |
| `aperto` | −0,05 (`knobs.py:355`) | peso de `aperto` |
| `sacudida_norma` | (5,04; 31,56) (`knobs.py:358`) | `sacudida` |
| `F_ref` | 49,0 (`knobs.py:359`) | `aperto` |
| `margem_pega` | 2,5 (`knobs.py:365`) | `progresso` (conquista `c1`) |
| `terminacao` | −300,0 (`knobs.py:366`) | peso de `terminacao` |
| `perda_caixa` | 240,0 (`knobs.py:367`) | custo de `perda_caixa` |
| `sigma_pouso` | (1,0; 0,35) (`knobs.py:368`) | `progresso` (fator de pouso) |
| `altura_pouso` | 0,05 (`knobs.py:369`) | `progresso` (fator de pouso) |
| `altura_pouso_inicio` | 0,10 (`knobs.py:370`) | `progresso` (fator de pouso) |
| `sigma_rastreio` | 0,5 (`knobs.py:238`) | `progresso` (rastreio normalizado em `h`) |
| `piso_rastreio` | −1,0 (`knobs.py:239`) | `progresso` (rastreio normalizado em `h`) |
| `sigma_zero_lin` | 0,10 (`knobs.py:241`) | `progresso` (rastreio normalizado em `h`) |
| `sigma_zero_ang` | 0,15 (`knobs.py:242`) | `progresso` (rastreio normalizado em `h`) |
| `sigma_zero_ang_inicio` | 0,5 (`knobs.py:243`) | `progresso` (rastreio normalizado em `h`) |

### Knobs de outras classes que estes termos leem

| knob | valor | lido por |
|---|---|---|
| `Alvos.*` (lista na seção 6.3 e 5.2) | `knobs.py:175-231` | `progresso`, fechos, `sigma` |
| `Alvos.incl_inicio_graus`, `incl_fim_graus`, `delta_incl_graus` | 30,0; 15,0; 15,0 (`knobs.py:203-205`) | portão `incl` do `progresso` |
| `Alvos.sigma_min_etapa` | (0,05; 0,05; 0,08; 0,20; 0,20; 0,05; 0,0) (`knobs.py:208`) | `v3_sigma` |
| `Alvos.largura_tanh` | 2,0 (`knobs.py:221`) | Φ e conquista `c0` |
| `Comando.limiar_canal` | 0,02 (`knobs.py:138`) | `progresso` (`h`) |
| `Contrato.dt_fisica`, `Contrato.decimacao` | 0,005 s; 4 (`knobs.py:54-55`) | `dt` de passo |
| `Balanceador.portao` | 0,75 (`knobs.py:391`) | rampa do `freio` |
| `Balanceador.degraus` | 5 (`knobs.py:401`) | `por_degrau` |
| `Balanceador.iters_nivel` | 50 (`knobs.py:395`) | subida de nível, logo `restricao` |
| `Balanceador.passos_por_iteracao` | 24 (`knobs.py:402`) | fecho de iteração |
| `DR.atrito_pad` | (0,6; 1,0) (`knobs.py:523`) | `F_pega` (conquista `c1`) |
| `Cadeia.caixa_perdida_vira_andar` | True (`knobs.py:161`) | `perda_caixa` e `caixa_perdida` |
| `Terminacao.caixa_folga_chao`, `v_solta` | 0,02 m; 1,2 m/s (`knobs.py:379-380`) | disparo da perda |
| `Terminacao.*` | `knobs.py:375-383` | terminações da seção 7.2 |
| `Cena.prateleira_meia_xy` | 0,30 m (`knobs.py:89`) | disparo da perda |
| `Cena.caixa_meia_faixa`, `caixa_n_variantes`, `caixa_massa_base` | (0,07; 0,13); 8; 1,0 (`knobs.py:78-80`) | `v3_meia`, `v3_massa` |
| `Niveis.pegar`, `Niveis.botar`, `Niveis.topo_min_c`, `Niveis.subnivel_extra` | `knobs.py:417-419`, `knobs.py:422` | `v3_topo`, `v3_massa`, `v3_topo_destino`, subníveis |
| `Estimador.idade_max` | 2,0 s (`knobs.py:453`) | abertura do P0 |
| `Balanceador.iters_refino` | 100 (`knobs.py:400`) | subida do degrau |
| `Balanceador.zona_t_alvo`, `zona_ganho`, `zona_n_min`, `zona_t_visita`, `zona_folga_cadeia` | 0,6; 0,05; 200; 8,0; 0,2 (`knobs.py:404-408`) | zona do fecho |
| `Alvos.zona` | tupla de 15 itens (`knobs.py:211-220`) | fecho dos alvos |
| `Comando.reamostra_s` | (3,0; 8,0) s (`knobs.py:137`) | reamostra do comando do `freio` |
| `Refino.degrau_batente` | 1 (`knobs.py:547`) | terminação `batente` |
| `Balanceador.portao_andar`, `portao_andar_terreno` | 0,75; 0,50 (`knobs.py:392-393`) | portão das células do ANDAR, na subida de nível e logo na `restricao`; a célula `terreno` do ANDAR não entra mais na subida de nível (27/11, T19) |
| `Balanceador.linha_n_min`, `linha_ganho`, `nivel_abre_terreno` | 50; 2,5; 2 (`knobs.py:410-412`) | linha de terreno do ANDAR por coluna e abertura da célula `terreno` dele; o nível de terreno do ANDAR entra no degrau de refino |
| `Terreno.dist_travessia` | 3,5 m (`knobs.py:442`) | terminação `time_out` (`atravessou`) |
| `Balanceador.alfa` | 0,05 (`knobs.py:394`) | média móvel da nota `s` das células, lida pela rampa do `freio` e pelos portões |
| `Balanceador.rampa_fatia_inicio`, `rampa_fatia_passo`, `rampa_fatia_its` | 0,05; 0,02; 12 (`knobs.py:397-399`) | fração de envs por fatia, logo quais trechos o `progresso` visita |
| `Balanceador.nivel_abre_b` | 5 (`knobs.py:396`) | nível do ANDAR que abre as fatias B e C |
| `Comando.tau_fracao`, `limiar_marcha`, `fase_apoio`, `periodo_marcha` | 0,4 s; 0,1; 0,55; 0,8 s (`knobs.py:136`, `:141`, `:139`, `:140`) | fator `g` do sinal de `h` no CARREGAR |
| `Comando.frente_tetos`, `tras_teto_max`, `lado_teto_max`, `giro_tetos`, `piso_velocidade`, `piso_lado`, `terreno_v_max` | (0,4; 0,55; 0,7; 0,85; 1,0); 0,5; 0,3; (0,3; 0,4; 0,5; 0,5); 0,25; 0,2; 1,0 (`knobs.py:122-127`, `:148`) | comando do CARREGAR, lido pelo regime do `freio` |
| `Comando.andar_inicio`, `andar_passo` | 0,3; 0,1 (`knobs.py:129`, `:134`) | nível máximo do ANDAR (7), logo a `restricao` |
| `Cadeia.espera_s`, `atraso_pegar_s`, `carregar_b_s`, `carregar_c_s`, `atraso_botar_s`, `andar_final_s`, `episodio_s` | (0,5; 1,0); (0,0; 1,0); (4,0; 7,0); (0,5; 1,0); (0,0; 1,5); 1,5; 20,0 (`knobs.py:154-160`) | duração dos trechos |

---

## 9. Outros comportamentos do código dentro do assunto

1. **O `progresso` muda estado.** Ele grava `v3_h`, `v3_phi_melhor` e `v3_phi` em todo env, a cada chamada, sem a máscara de tarefa (`recompensas.py:868-869`, `recompensas.py:883`).
2. **O `progresso` roda o rastreio normalizado a cada passo**, também fora do CARREGAR, mas só usa o resultado no ramo `carrega` (`recompensas.py:861-867`).
3. **O Φ lê σ e índice do passo anterior.** `_phi_env` usa `env.v3_alvo_idx` e `env.v3_sigma` como estão na chamada. O comando de cadeia os reescreve depois (seção 2).
4. **O alvo fica em P3 durante o CARREGAR.** O índice não avança no fecho do P3 (`comando.py:493`). Por isso as cercas e os termos `sacudida` e `aperto` tratam o CARREGAR como "alvo P3" (`_CAIXA_ALVO`, `recompensas.py:58`) e a `cerca_recuo` e a `cerca_pelve` usam as referências de transporte no CARREGAR (`recompensas.py:513-516`, `recompensas.py:783-784`).
5. **Distância do B1 com déficit de apoio.** O B1 só chega a Φ alto com a laje carregando o peso: o déficit de apoio entra na distância `d` (seção 6.3).
6. **Cerca de torque sem teto.** O excesso além do limite cresce linearmente: não há saturação em `lim + banda` (`recompensas.py:188-193`).
7. **A `cerca_com` usa a caixa só presa.** A massa da caixa entra no CoM combinado só com `v3_pegou & ~v3_perdeu` e alvo em {P3, B1, B2}. Com o alvo B3 o CoM é o do robô (`recompensas.py:431`).
8. **O `contato_fora` conta o contato com a caixa fora da janela.** No P0, no B3 e no ANDAR, qualquer contato do robô com a caixa acima de 1,0 N soma 1 por slot (`recompensas.py:811-813`).
9. **O `freio` depende do estado de currículo de todas as tarefas.** A rampa lê a média de `s` das células abertas da tarefa, e o expoente lê o nível da tarefa. Ambos mudam entre iterações (`recompensas.py:916-919`).
10. **O módulo diz "seis cercas" e lista sete nomes** (`recompensas.py:6-7`): sola, giro, tronco, CoM, torque, pelve e recuo. O código registra as sete.
11. **A meia aresta `v3_meia` é sorteada na criação do env (startup).** O evento `tamanho_caixa` monta `Cena.caixa_n_variantes` = 8 valores igualmente espaçados em `Cena.caixa_meia_faixa` = (0,07; 0,13) m. Cada env recebe um dos 8, escolhido com probabilidade uniforme (`env_cfg.py:193-195`, `eventos.py:429-440`, `knobs.py:78-79`).
12. **`v3_topo` e `v3_massa` são sorteados no reset do episódio**, pelo evento `cena`, depois do termo de currículo (`eventos.py:332-348`, `eventos.py:82-109`). Com `(h_k, m_k)` = `Niveis.pegar[nível do PEGAR]`, `h_0` = `Niveis.pegar[0][0]` = 0,55 m e `extra` = `Niveis.subnivel_extra[subnível do PEGAR]` (`knobs.py:417`, `knobs.py:422`):
    - fatia B: topo uniforme em [`h_k` + `extra`; `h_0` + `extra`] m; massa uniforme em [`Cena.caixa_massa_base`; `m_k`] kg, com `caixa_massa_base` = 1,0 (`knobs.py:80`);
    - fatia C: topo uniforme em [`max(h_k, Niveis.topo_min_c)` + `extra`; `h_0` + `extra`] m, com `topo_min_c` = 0,30 (`knobs.py:419`); massa uniforme em [1,0; `m_j`] kg, com `m_j` = `Niveis.pegar[nível_passou][1]`, ou 1,0 se nenhum nível passou;
    - fatia ANDAR: topo 0 e massa 1,0 kg.
13. **`v3_topo_destino` é sorteado na abertura do BOTAR**, em `posiciona_destino`: topo uniforme em [`Niveis.botar[nível do BOTAR]`; `Niveis.botar[0]`] m. Com o subnível do BOTAR abaixo do último, o topo vale `Niveis.botar[0]` + `Niveis.subnivel_extra[subnível]` (`eventos.py:373-390`, `eventos.py:393-413`, `knobs.py:418`, `knobs.py:422`).
14. **O `progresso` não usa `n_qualidade` nem `aplica_qualidade`.** Essas duas funções escalam o rastreio positivo do CARREGAR (`recompensas.py:299-308`, `recompensas.py:590-594`). O `progresso` lê só a distância de transporte de `_qualidade(env)[1]` (`recompensas.py:866`).
15. **A leitura da caixa abre o P0.** O P0 abre quando `v3_estimador.tem_leitura()` é verdadeiro: a leitura existe e tem idade menor que `Estimador.idade_max` = 2,0 s (`estimador.py:373-376`, `knobs.py:453`).

---

## 10. Não determinado pelo código lido

- Não determinado pelo código lido: os valores numéricos das tabelas de `referencias.npz` (`pelve_de_pe`, `grade_pelve`, `grade_tronco`, `grade_recuo`, `caixa_transporte_b`, `pelve_com_caixa`, `cantos_sola`). O código só as carrega (`alvos.py:47-67`).
- Não determinado pelo código lido: o sinal de `ref.tronco` (a grade de tronco em rad) e portanto se `tronco_ik` desloca o limite da `cerca_tronco` para cima ou para baixo em relação a `incl_torso`, que é sempre ≥ 0 (`recompensas.py:171-173`, `alvos.py:72-73`).
- Não determinado pelo código lido: o valor numérico de `g` em `peso_N` e em `F_pega`. O código o lê de `env.sim.mj_model.opt.gravity[2]` do modelo MuJoCo (`recompensas.py:543`, `recompensas.py:557`).
- Não determinado pelo código lido: o valor de `v3_degrau` e de `restricao` em um instante do treino. Eles dependem do estado do currículo, que evolui pelas notas medidas (`balanceador.py:153-278`).
- Não determinado pelo código lido: a geometria exata dos geoms `*_palm_pad` e `*_hand_back_pad` e dos grupos de colisão do tronco. O código só lê os nomes (`cena.py:45-47`).

---

## 11. Estado do env que os termos leem: quem escreve e quando

Cada linha diz qual buffer do env um dos 15 termos lê e quando ele muda. "Passo" quer dizer o `compute` da cadeia (`comando.py:596-629`), que roda depois das recompensas e das terminações do mesmo passo (seção 2).

| buffer | lido por | escrito por, e quando |
|---|---|---|
| `v3_tarefa` | máscara `_fim` de todo termo; fator `por_restricao` com `so` (cercas: `so="ANDAR"`; `desvio_*`: `so="CARREGAR"`) | `_publica`, na abertura de um trecho e na perda da caixa (`comando.py:511`) |
| `v3_alvo_idx` | `cerca_tronco`, `cerca_pelve`, `cerca_recuo`, `contato_fora`, `sacudida`, `aperto`, `progresso`, `cerca_com` | −1 na abertura de um trecho ANDAR ou PEGAR (`comando.py:585-586`); P0 na primeira leitura válida da caixa (`comando.py:386`); B1 na abertura do BOTAR (`comando.py:559`); B3 na perda (`comando.py:415`); `índice + 1` no avanço, com o P3 parado (`comando.py:492-501`) |
| `v3_sigma` | `progresso` (Φ) | 1,0 no reset do episódio (`comando.py:655`); recalculado na abertura do P0, na abertura do B1, na perda e na troca de alvo (`comando.py:392`, `:565`, `:419`, `:503`) |
| `v3_h`, `v3_phi_melhor` | `progresso` | 1,0 e 0,0 no reset, na abertura de trecho, do P0, do BOTAR, na perda e na troca de alvo (`comando.py:657`, `:583`, `:388`, `:561`, `:417`, `:504`); a cada passo, o próprio `progresso` os atualiza (`recompensas.py:868-869`) |
| `v3_pegou` | `cerca_com` (entrada do CoM), `progresso` (CoM nos portões) | False no reset; True no fecho do P2 (`comando.py:651-653`, `:497`) |
| `v3_perdeu` | `progresso`, `cerca_com` | False no reset; True na perda (`comando.py:651-653`, `:414`) |
| `v3_perdeu_agora` | `perda_caixa`, terminação `caixa_perdida` | reescrito a cada passo (`comando.py:411`); False no reset (`comando.py:651-653`) |
| `v3_pega_fechada` | `progresso` | False no reset; True no fecho do P3 (`comando.py:496`) |
| `v3_bota_fechada` | `progresso` | False no reset; True no fecho do B3 (`comando.py:498`) |
| `v3_freio`, `v3_degrau` | `freio`; `contato_laje`, `sacudida`, `aperto`, `progresso` (por `por_degrau`), terminações `chegada` e `batente` | copiados da tarefa do trecho em `_publica`, na abertura de cada trecho (`comando.py:515-518`) |
| `v3_ponto_apoio` | `progresso` (portão `desloc` do B2 e do B3) | 0 na abertura do BOTAR; posição da caixa no fecho do B1 (`comando.py:563`, `:495`) |
| `v3_caixa_pega_w` | `progresso` (portão `desloc` do P0 ao P2) | posição estimada da caixa na abertura do P0 (`comando.py:390`) |
| `v3_alvo_w` | `progresso` (alvo do B1), perda (`fora_do_alvo`) | posição estimada da caixa no P0 (`comando.py:391`); ponto de soltura, com viés, na abertura do BOTAR (`comando.py:558`) |
| `v3_meia` | quase todos os termos de manipulação | sorteada uma vez por env na criação do env, evento `tamanho_caixa` (`eventos.py:440`) |
| `v3_topo`, `v3_massa` | `cerca_tronco`, `cerca_pelve`, `cerca_recuo`, `progresso`, `cerca_com` | sorteados no reset do episódio, evento `cena` (`eventos.py:347-348`) |
| `v3_topo_destino` | `cerca_tronco`, `cerca_pelve`, `cerca_recuo` e `progresso` nos alvos B1 e B2; pouso | sorteado na abertura do BOTAR (`eventos.py:411`) |
| `v3_celula` | `progresso` (alvo de transporte do CARREGAR) | `twist`, no reset e na reamostra do comando (`comando.py:901-919`) |
| `twist.vel_command_b` | `freio`, `progresso` | `twist`, no reset, na abertura de trecho e a cada reamostra de `Comando.reamostra_s` (`comando.py:902-925`). No ANDAR de terreno com comando linear, o `wz` é reescrito a cada passo pelo rumo (`comando.py:943-948`; 27/11, T19). O `freio` não está registrado no ANDAR, e o `progresso` só usa o comando no CARREGAR com pega fechada (seção 6.5): a reescrita no ANDAR não muda o valor dos 15 termos |
| `twist.g_marcha` | `progresso` (sinal de `h` no CARREGAR) | `twist`, a cada passo, depois das recompensas (`comando.py:871`) |
| `v3_cur` (`restricao`, notas `s`, degraus, níveis) | cercas com `so="ANDAR"`, `freio`, `progresso`, `_rastreio` | `fecha_iteracao`, no primeiro reset depois que `common_step_counter // Balanceador.passos_por_iteracao` muda (`curriculo.py:119-121`) |
| `v3_alvos` (zona) | só o fecho dos alvos e a qualidade do CARREGAR; os 15 termos usam `K = ATIVO.alvos` | refeito quando `cur.it` muda (`comando.py:428-441`) |

---

## 12. Índice das funções do assunto

Cada núcleo `n_*` recebe tensores e devolve o valor sem sinal e sem peso. Cada invólucro lê o env e aplica `_fim`.

| termo | núcleo | invólucro | invólucro lê |
|---|---|---|---|
| `cerca_sola` | `n_cerca_sola` (`recompensas.py:156`) | `cerca_sola` (`:736`) | `_incl_graus` (`:368`), sensor `pes_chao` |
| `cerca_giro` | `n_cerca_giro` (`:161`) | `cerca_giro` (`:744`) | `joint_pos`, `por_restricao` (`:453`) |
| `cerca_tronco` | `n_cerca_tronco` (`:171`) | `cerca_tronco` (`:750`) | `incl_torso` (`:467`), `ref_do_alvo` (`:501`) |
| `cerca_com` | `n_cerca_com` (`:176`), `margem_com` (`:316`) | `cerca_com` (`:761`) | `_pes_e_com` (`:473`), `_com_combinado` (`:419`), `_cantos_mundo` (`:438`) |
| `cerca_torque` | `n_cerca_torque` (`:182`) | `cerca_torque` (`:767`) | `qfrc_actuator`, `TAU_MAX`, `_PARES` (`:60`) |
| `cerca_pelve` | `n_cerca_pelve` (`:196`) | `cerca_pelve` (`:775`) | `altura_pelve`, `ref_do_alvo` |
| `cerca_recuo` | `n_cerca_recuo` (`:201`) | `cerca_recuo` (`:791`) | `_recuo_ref` (`:509`), `_topo_do` (`:496`) |
| `contato_fora` | `n_contato_fora` (`:206`) | `contato_fora` (`:805`) | sensores `nao_pes_chao`, `auto_colisao`, `caixa_robo` |
| `contato_laje` | (conta no invólucro) | `contato_laje` (`:887`) | `MESA_POR_GRUPO` (`cena.py:49`), `por_degrau` (`:447`) |
| `freio` | (conta no invólucro) | `freio` (`:899`) | `_VMAX` (`:42`), `Curriculo.celulas` |
| `progresso` | `n_progresso` (`:286`), `n_pressa` (`:294`), `phi_do_alvo` (`:274`), `distancia_do_alvo` (`:224`), `portoes_do_alvo` (`:247`), `potencial.phi`, `potencial.rampa`, `potencial.fator_pouso`, `potencial.rastreio_normalizado` | `progresso` (`:816`), `_phi_env` (`:519`) | `_rastreio` (`:374`), `_qualidade` (`:572`), `alvo_da_caixa_w` (`:487`), `_contato_palmas` (`:482`), `incl_lim` (`:462`), `forcas_por_face` (`:345`) |
| `sacudida` | `n_sacudida` (`:339`) | classe `sacudida` (`:922`) | `box.root_link_vel_w` |
| `aperto` | `potencial.custo_aperto` (`potencial.py:55`), `forcas_por_face` (`:345`) | `aperto` (`:948`) | sensor `caixa_faces`, `alvos.frame_da_caixa` (`alvos.py:109`) |
| `perda_caixa` | `n_perda_caixa` (`:311`) | `perda_caixa` (`:960`) | `v3_perdeu_agora`; a detecção é `caixa_perdida_agora` (`comando.py:59`) |
| `terminacao` | `n_terminacao` (`:214`) | `terminacao` (`:965`) | `termination_manager.terminated`, `.time_outs` |
