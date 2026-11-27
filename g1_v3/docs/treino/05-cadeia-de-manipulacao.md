# Cadeia de manipulação (PEGAR, CARREGAR, BOTAR)

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:
- `comando.py` (inteiro), `alvos.py` (inteiro), `potencial.py` (inteiro), `estimador.py` (inteiro), `knobs.py` (inteiro).
- `dados/gera_referencias.py` (inteiro) e as chaves e valores de `dados/referencias.npz` (`topos`, `meias`, `pelve_de_pe`, `pelve_com_caixa`, `caixa_transporte_b`).
- `curriculo.py`, `balanceador.py`, `eventos.py`, `terminacoes.py`, `env_cfg.py` (inteiros).
- `recompensas.py` (inteiro), `observacoes.py` (linhas 1 a 340), `algoritmo.py` (linhas 60 a 169).
- `metricas.py` (trechos: `_com_caixa`, `phi_medio_do_alvo`, `tempo_no_alvo`, `erro_p3`, `deslocamento_da_caixa`, `velocidade_de_chegada`, `deslocamento_depois_do_apoio`, `aceleracao_da_caixa`, `idade_das_leituras`, `fracao_avisos`).
- `cena.py` (linhas 320 a 425: sensores).
- `mjlab/envs/manager_based_rl_env.py` (`step`, `_reset_idx`, `scale_rewards_by_dt`, `step_dt`, `max_episode_length`), `mjlab/managers/command_manager.py` (`reset`, `compute`, `_resample`), `mjlab/managers/reward_manager.py` (`compute`), `mjlab/envs/mdp/events.py` (`apply_body_impulse`).

## 1. Conceitos usados neste documento

| Termo | O que é |
|---|---|
| Tarefa | O rótulo que o robô recebe a cada passo: ANDAR, PEGAR, CARREGAR ou BOTAR. Vai ao ator como vetor one-hot de 6 canais (`comando.py:308-310`, `knobs.py:42`). |
| Fatia | O grupo de tarefas que um env recebe no reset: ANDAR, B ou C (`knobs.py:43`). A fatia ANDAR tem um trecho só. A fatia B tem os trechos ANDAR parado, PEGAR e CARREGAR. A fatia C tem os trechos ANDAR parado, PEGAR, CARREGAR parado, BOTAR e ANDAR parado (`comando.py:175-180`). |
| Trecho | Um pedaço da sequência de tarefas de uma fatia. Cada trecho publica uma tarefa e acaba por tempo ou por aviso (`comando.py:163-168`). |
| Etapa ou alvo | Um sub-objetivo geométrico dentro do PEGAR (P0, P1, P2, P3) ou do BOTAR (B1, B2, B3). O índice vai de 0 a 6 na ordem `P0, P1, P2, P3, B1, B2, B3`; −1 quer dizer "sem alvo" (`knobs.py:44`, `alvos.py:16`). |
| Fecho de uma etapa | A condição que, mantida sem interrupção pelo tempo da etapa, faz o índice do alvo avançar (`alvos.py:195-210`). |
| Invariante | O produto de três portões (deslocamento, inclinação, mãos) que tem de valer 1 para uma etapa fechar (`comando.py:466`). |
| Aviso | O latch (valor que, ligado, só desliga na abertura do próximo trecho) que marca o fecho do P3 no PEGAR e o do B3 no BOTAR (`comando.py:604-607`). |
| Zona | A escala z de 0 a 1, uma por etapa, que vai do critério largo (z = 1) ao critério final (z = 0) (`balanceador.py:105`). |
| Φ | O potencial de proximidade ao alvo corrente. Paga renda no termo `progresso` (seção 9). |
| σ | A escala de distância de Φ, fixada na abertura de cada etapa (seção 9). |
| Degrau de refino | O inteiro de 0 a 5 que aperta a dificuldade de uma tarefa. O degrau global é o menor degrau das 4 tarefas (`balanceador.py:338-343`). |
| Nível e subnível | O nível 0 a 4 do PEGAR escolhe a altura da laje de pega e a massa máxima da caixa. O nível 0 a 4 do BOTAR escolhe a altura da laje de destino. O nível 0 do PEGAR e o do BOTAR têm dois subníveis cada (`knobs.py:417-424`). |
| Caixa perdida | O estado `v3_perdeu` (seção 11). |
| Laje | A superfície em que a caixa repousa: a laje da pega e a laje de destino do BOTAR. É uma só entidade (`env.scene["table"]`) movida por teleporte (`eventos.py:276-294`). |
| `a` | A meia aresta da caixa do env, `v3_meia`. A caixa é um cubo. |

Frame da caixa (`alvos.py:109-128`): o eixo z é a normal de face mais vertical, com sinal para cima. O eixo x é a normal de face horizontal mais alinhada ao rumo da base; ela aponta do robô para o lado distante da caixa. O eixo y é z × x. A palma 0 fica em +y e a palma 1 em −y (`alvos.py:139`).

## 2. Quando cada parte roda

### 2.1 Ordem dentro de um passo do env

Um passo da política vale `Contrato.dt_fisica` = 0,005 s vezes `Contrato.decimacao` = 4, isto é 0,02 s (`knobs.py:54-55`). A ordem do mjlab (`mjlab/envs/manager_based_rl_env.py:421-464`) é:

1. Física, 4 subpassos.
2. `episode_length_buf += 1` e `common_step_counter += 1` (`manager_based_rl_env.py:430-431`).
3. Terminações.
4. Recompensas.
5. Métricas.
6. Reset dos envs que terminaram (seção 2.3).
7. `sim.forward()`.
8. `command_manager.compute`: primeiro o termo `cadeia`, depois o termo `twist` (a ordem do dicionário em `env_cfg.py:118-128`). A cadeia mede, detecta perda, abre etapas, avança o trecho e dá o passo da odometria. O twist, nesse mesmo `compute`, soma `dt` ao seu relógio e reamostra o comando dos envs com `time_left <= 0` (`mjlab/managers/command_manager.py:109-123`).
9. Eventos de modo `step` e `interval`.
10. `sim.sense()` (atualiza os sensores de altura e contato).
11. Observações. O estimador da caixa dá o seu passo aqui (`observacoes.py:140-151`).

Consequência: as terminações e as recompensas de um passo leem os buffers que a cadeia escreveu no passo anterior. O estimador, ao contrário, é atualizado depois da cadeia; a cadeia lê o estado que ele deixou no passo de observação anterior.

### 2.2 Ordem dentro de `CadeiaCmd.compute`

`comando.py:596-629`:

1. `_mede`: mede o estado verdadeiro (`comando.py:598`).
2. `_perda`: detecta a perda da caixa (`comando.py:599`).
3. Copia `v3_perdeu_agora` para `env.extras["v3_perdeu_agora"]` (`comando.py:601`).
4. `_abre_p0`: abre o P0 se há leitura válida (`comando.py:602`).
5. `_alvos`: zona, visitas, fecho das etapas e avanço do alvo (`comando.py:603`).
6. Latch do aviso (`comando.py:606-607`).
7. `avanca_trecho` e congelamento por perda (`comando.py:608-613`).
8. Fim da cadeia e registro do BOTAR (`comando.py:614-616`).
9. Abertura do trecho novo (`comando.py:617-619`).
10. Passo da odometria (`comando.py:623-625`).
11. `_grava_chegada` (`comando.py:626`).
12. `v3_vz_caixa_ant` recebe a velocidade vertical da caixa (`comando.py:629`).

O comando é chamado uma vez por passo para o env inteiro. Num env recém-resetado o `compute` roda no mesmo passo do reset (seção 2.1, item 8), então o relógio do primeiro trecho já soma um `dt` nesse passo.

### 2.3 Ordem do reset

`mjlab/envs/manager_based_rl_env.py:553-591`: currículo (`curriculo.v3`), `sim.reset`, `scene.reset`, eventos de modo `reset`, observações, ação, recompensas, métricas, currículo, `command_manager.reset` (cadeia e depois twist), eventos, terminações; por fim `episode_length_buf = 0`.

Os eventos de reset rodam na ordem do dicionário `events` (`env_cfg.py:153-184`): `fatia`, `cena`, `base`, `juntas`, `carga`, `atrito_pad`, `atrito_laje`, `vies_imu`, depois `pd`, `forca_motor` e `solref` (só fora do play), `com_caixa` (só com `DR.com_caixa`; `env_cfg.py:182`) e `filtros`.

### 2.4 Sorteios e mudanças de estado, por momento

| Momento | O que muda | Distribuição e faixa | Ref. |
|---|---|---|---|
| Criação do env (startup) | Tamanho da caixa `v3_meia`. Um valor por env, fixo até o fim do treino. | Escolha uniforme entre `linspace(0,07; 0,13; 8)`: `Cena.caixa_meia_faixa` = (0,07; 0,13) m (`knobs.py:78`), `Cena.caixa_n_variantes` = 8 (`knobs.py:79`). | `eventos.py:416-440`, `env_cfg.py:193-195` |
| Criação do env | Gerador `v3_gerador`, semente do cfg (0 se ausente). Todo sorteio da cadeia, do estimador e da odometria usa este gerador. | Semente. | `curriculo.py:62-64` |
| Reset do episódio | Fatia. | `torch.multinomial` sobre `fracoes_fatias()`. | `eventos.py:161`, `balanceador.py:282-304` |
| Reset | Episódio de terreno (só ANDAR e B). | Bernoulli com p = fração da célula `terreno` da tarefa. | `eventos.py:165-167` |
| Reset | Topo da laje de pega e massa da caixa (B e C). | Uniforme (seção 4.2). | `eventos.py:82-109,344-348` |
| Reset | Posição da caixa (B e C): dx, dy. | dx = `jitter_x_max[nível PEGAR]`·U[0; 1) (`Cena.jitter_x_max` = (0,20; 0,20; 0,20; 0,15; 0,08) m, `knobs.py:99`); dy = U[`caixa_jitter_y`] = U[−0,18; 0,18] m (`knobs.py:97`). | `eventos.py:357-359` |
| Reset | Rumo da cena na plataforma (só B em terreno). | k uniforme em {0, 1, 2, 3}, rumo = k·π/2. | `eventos.py:112-130,355-356` |
| Reset | Base do robô (B e C). | Seção 4.3. | `eventos.py:178-208` |
| Reset | Estado do estimador: viés de meia aresta, viés de posição, eixo e ângulo do viés de orientação. | Seção 12. | `estimador.py:262-280` |
| Abertura de cada trecho | Duração ou atraso do trecho, `alvo_s`. | Uniforme entre os dois extremos da faixa do trecho (seção 5). | `comando.py:578-579` |
| Abertura do BOTAR | Laje de destino e viés do alvo de soltura. | Seção 6.4 (laje de destino) e 5.4 (viés). | `eventos.py:373-413`, `comando.py:542-566` |
| Abertura do trecho CARREGAR (fatia B) | Célula e velocidade do CARREGAR. O comando fica fixo até o fim do trecho. | Seção 5.5. | `comando.py:519,897-941` |
| A cada `Comando.reamostra_s` = (3,0; 8,0) s (`knobs.py:137`) | Comando de velocidade do ANDAR na fatia ANDAR. Nas outras tarefas o twist reamostra e o comando continua zero. | Intervalo ~ U[3,0; 8,0] s por env; velocidade por `sorteia_velocidade` (`comando.py:118-147`). No ANDAR de terreno o sorteio dá só vx = frente·r, com vy = 0 e wz = 0; o wz vem do rumo a cada passo (`comando.py:137-142`, `comando.py:943-947`). | `env_cfg.py:122-123`, `mjlab/managers/command_manager.py:109-123`, `comando.py:897-938` |
| A cada passo | Passo do estimador da caixa e da odometria. | Gaussiano e uniforme (seção 12). | `observacoes.py:191`, `comando.py:623-625` |
| A cada iteração do PPO (24 passos por env) | Zona z de cada etapa; `Alvos` efetivo; tempos de permanência. | Determinístico (seção 7). | `balanceador.py:169-176`, `comando.py:428-443` |

`Balanceador.passos_por_iteracao` = 24 (`knobs.py:402`). A iteração do currículo fecha em `curriculo.v3` quando `common_step_counter // 24` muda; o fecho roda dentro do reset de qualquer env (`curriculo.py:119-122`).

## 3. Buffers `v3_*` do env

`garante_v3` cria os buffers e não recria um que existe (`curriculo.py:33-90`). O `CadeiaCmd.__init__` o chama na construção (`comando.py:293`). Valor inicial: `v3_alvo_idx` e `v3_celula` = −1; `v3_t_empurrao` = −∞; `v3_phi` = −1; `v3_h` = 1; `v3_v_chegada` = −1; o resto 0 ou falso (`curriculo.py:70-86`).

### 3.1 Estado da cadeia

| Buffer | Forma e tipo | Significado | Quem escreve | Quem lê |
|---|---|---|---|---|
| `v3_fatia` | (E,) long | Fatia do episódio: 0 ANDAR, 1 B, 2 C. | `eventos.sorteia_fatia`, no reset (`eventos.py:162`). | `comando.py:592,660,905`; `eventos.py:165-169`. |
| `v3_trecho` | (E,) long | Índice do trecho corrente na tabela de trechos. | `_abre_trecho` (`comando.py:581`). | `compute` (`comando.py:607-609`). |
| `v3_tarefa` | (E,) long | Tarefa publicada (índice de `TAREFAS`). | `_publica` (`comando.py:511`). | Recompensas, terminações, observações, métricas. |
| `v3_freio` | (E,) long | Nível aberto da tarefa publicada + 1 (`niveis_abertos`). | `_publica` (`comando.py:515,518`). | `recompensas.freio` (`recompensas.py:919`). |
| `v3_degrau` | (E,) long | Degrau de refino da tarefa publicada, lido na abertura do trecho. | `_publica` (`comando.py:516,518`). | `recompensas.por_degrau` (`recompensas.py:450`); `terminacoes.batente` (`terminacoes.py:181`); `observacoes.degrau`. |
| `v3_alvo_idx` | (E,) long | Etapa corrente; −1 sem alvo. | `_abre_p0` (`comando.py:386`), `_perda` (`comando.py:415`), `_alvos` (`comando.py:501`), `_abre_botar` (`comando.py:559`), `_abre_trecho` (`comando.py:586`). | Recompensas, observações, métricas, `_perda`. |
| `v3_celula` | (E,) long | Célula do currículo do segmento de comando; −1 sem registro. | `TwistV3._resample_command` (`comando.py:901,919`). | `recompensas._qualidade` (`recompensas.py:582`), twist. |
| `v3_pegou` | (E,) bool | A pega foi feita: o P2 fechou. | `_alvos` (`comando.py:497`). | `_perda` (`comando.py:409`), `_com_combinado` (`recompensas.py:431`), `forca_caixa` (`eventos.py:598`). |
| `v3_perdeu` | (E,) bool | A caixa foi perdida (latch até o reset). | `_perda` (`comando.py:414`). | Recompensas, observações, métricas, twist, `_alvos`. |
| `v3_perdeu_agora` | (E,) bool | Pulso do passo em que a perda foi detectada. | `_perda` (`comando.py:411`). | `terminacoes.caixa_perdida` (`terminacoes.py:123`), `recompensas.perda_caixa` (`recompensas.py:962`), `PPOPorTarefa.process_env_step` (`algoritmo.py:130`). |
| `v3_fim` | (E,) bool | O último trecho da fatia acabou. | `compute` (`comando.py:614`). | `terminacoes.fim_da_cadeia` (`terminacoes.py:72`). |
| `v3_aviso` | (E,) bool | Latch do aviso do trecho. Zera na abertura de cada trecho. | `_abre_trecho` (`comando.py:582`), `compute` (`comando.py:607`). | `avanca_trecho`; `observacoes._calcula_caixa` (`observacoes.py:196`); `metricas.fracao_avisos`. |
| `v3_avancou` | (E,) bool | A etapa avançou neste passo. | `_alvos` (`comando.py:499`). | Não determinado pelo código lido: nenhum leitor aparece nos arquivos lidos. |
| `v3_pega_fechada` | (E,) bool | O P3 fechou. Fica ligado até o reset. | `_alvos` (`comando.py:496`). | `ok_agora_pegar` (via `pega`), `recompensas.progresso`, `observacoes.alvo_macro`. |
| `v3_bota_fechada` | (E,) bool | O B3 fechou. Fica ligado até o reset. | `_alvos` (`comando.py:498`). | `compute` (`comando.py:606`), `recompensas.progresso`, `observacoes.alvo_macro`. |
| `v3_relogio_alvo` | (E,) float | Tempo seguido em que o fecho da etapa corrente vale. | `_alvos` (`comando.py:500`); zera em `_abre_p0`, `_abre_botar`, `_perda`, `_resample`. | `alvos.avanca`. |
| `v3_sigma` | (E,) float | σ da etapa corrente (m, ou adimensional no B3). | `_abre_p0` (`comando.py:392`), `_alvos` (`comando.py:503`), `_abre_botar` (`comando.py:565`), `_perda` (`comando.py:419`), `_resample` (`comando.py:655`). | `recompensas.progresso` (`recompensas.py:846`). |
| `v3_caixa_pega_w` | (E,3) float | Posição estimada da caixa no mundo, gravada na abertura do P0. | `_abre_p0` (`comando.py:390`). | `alvo_da_caixa_w`; portão de deslocamento do Φ (`recompensas.py:565`); `_alvos` (`comando.py:455`). |
| `v3_alvo_w` | (E,3) float | Na abertura do P0: a posição estimada da caixa. Na abertura do BOTAR: o ponto de soltura do B1 com viés. | `_abre_p0` (`comando.py:391`), `_abre_botar` (`comando.py:558`). | `alvos.caixa_alvo_w` (B1), observações, `_perda` (`comando.py:407`). Não zera no reset. |
| `v3_ponto_apoio` | (E,3) float | Posição da caixa no fecho do B1. Zera na abertura do BOTAR. | `_alvos` (`comando.py:495`), `_abre_botar` (`comando.py:563`). | `_alvos` (`comando.py:457`), `_phi_env` (`recompensas.py:554`), métrica `deslocamento_depois_do_apoio`. |
| `v3_v_chegada` | (E,) float | Velocidade de descida da caixa no primeiro toque da laje no BOTAR; −1 sem toque. | `_grava_chegada` (`comando.py:637`), `_abre_botar` (`comando.py:564`), `_resample` (`comando.py:658`). | Métrica `velocidade_de_chegada`. |
| `v3_vz_caixa_ant` | (E,) float | Velocidade vertical da caixa no fim do passo anterior. | `compute` (`comando.py:629`), todo env, todo passo. | `terminacoes.chegada` (`terminacoes.py:150`), `_grava_chegada`. |
| `v3_phi` | (E,) float | Φ do alvo corrente; −1 sem alvo. | `recompensas.progresso` (`recompensas.py:883`); sentinela −1 em `comando.py:387,416,587,656`. | Métricas `phi_<alvo>`, `observacoes.privilegiados`. |
| `v3_h` | (E,) float | Fator da renda da etapa, em [0; 1]. | `recompensas.progresso` (`recompensas.py:868`); recomeça em 1 em `comando.py:388,417,504,561,583,657`. | `recompensas.progresso`, `observacoes.privilegiados`. |
| `v3_phi_melhor` | (E,) float | Melhor Φ da etapa. | `recompensas.progresso` (`recompensas.py:869`); recomeça em 0 junto com `v3_h`. | `recompensas.progresso`, `observacoes.privilegiados`. |
| `v3_alvos` | objeto `Alvos` | `Alvos` efetivo da zona. Nasce igual ao `Alvos` final. | `_alvos` (`comando.py:441`); criação em `curriculo.py:49`. | `ok_agora_pegar`, `ok_agora_botar`, `recompensas._qualidade` (`recompensas.py:587`). |

### 3.2 Estado da cena

| Buffer | Forma e tipo | Significado | Quem escreve | Quem lê |
|---|---|---|---|---|
| `v3_meia` | (E,) float | Meia aresta da caixa. | `eventos.tamanho_caixa`, startup (`eventos.py:440`). | Quase todo módulo. |
| `v3_massa` | (E,) float | Massa da caixa (kg). | `posiciona_cena` (`eventos.py:348`). A massa simulada é escrita por `carga` (`eventos.py:443-461`). | `recompensas`, observações, `carga`. |
| `v3_topo` | (E,) float | Topo da laje de pega sobre o chão local. | `posiciona_cena` (`eventos.py:347`). | `_alvos` (`comando.py:473`), `_base_da_pega`, `recompensas._topo_do`. |
| `v3_topo_destino` | (E,) float | Topo da laje de destino sobre o chão local. | `posiciona_destino` (`eventos.py:411`). | `recompensas._topo_do`, `_phi_env` (`recompensas.py:548`). |
| `v3_caixa_xy` | (E,2) float | xy da caixa de B e C relativo à origem do env. | `posiciona_cena` (`eventos.py:364`). | `_base_da_pega` (`eventos.py:201`). |
| `v3_rumo_cena` | (E,) float | Rumo da cena (rad). | `posiciona_cena` (`eventos.py:365`). | `_base_da_pega` (`eventos.py:197`). |
| `v3_t_empurrao` | (E,) float | Instante do último empurrão (s desde o início do episódio). | `eventos.empurrao` (`eventos.py:269`); volta a −∞ em `sorteia_fatia` (`eventos.py:163`). | `TwistV3._update_metrics` (âncora). |

### 3.3 Estimador, percepção e marcha

| Buffer | Forma e tipo | Significado | Quem escreve | Quem lê |
|---|---|---|---|---|
| `v3_gerador` | `torch.Generator` | Gerador de todo sorteio da v3. | `garante_v3` (`curriculo.py:62-64`). | Toda função que sorteia. |
| `v3_odo` | `Odometria` | Pose estimada da base, com deriva. | `reinicia_filtros` (`eventos.py:240`), `compute` (`comando.py:624`). | `observacoes`, `_abre_p0`. |
| `v3_estimador` | `EstimadorCaixa` | Leitura simulada da caixa. | `reinicia_filtros` (`eventos.py:241`), `_calcula_caixa` (`observacoes.py:191`). | `_abre_p0` (`comando.py:380-381`), `_calcula_caixa`, `metricas.idade_das_leituras`. |
| `v3_caixa_obs` | (E,11) float | Caixa estimada (8) e alvo corrente (3), no frame da base. | `atualiza_percepcao` (`observacoes.py:150`). | Termo `caixa` do ator e do crítico (`observacoes.py:160-163`). |
| `v3_mapa_obs` | (E,882) float | Mapa do ator, do lidar. | `atualiza_percepcao` (`observacoes.py:149`). | Termo `mapa`. |
| `v3_filtro`, `v3_padrao`, `v3_percepcao_passo` | objetos e int | Filtro de elevação do mapa, padrão do MID-360, passo da última percepção. | `curriculo.py:52-60`. | `observacoes.atualiza_percepcao`. |
| `v3_vies_giro` (E,3), `v3_incl_imu` (E,3,3) | float | Viés do giro e inclinação fixa do IMU, por episódio. | `eventos.vies_imu` (`eventos.py:508-509`). | `observacoes.giro_base`, `observacoes.gravidade`. |
| `v3_ancora` | (E,4) float | Pose e instante da âncora do segmento (x, y, rumo, t). | `TwistV3._update_metrics` (`comando.py:836-841`). | `TwistV3._fecha` (`comando.py:763`). |
| `v3_incl_max_seg` | (E,) float | Inclinação máxima da caixa no segmento (graus). | `TwistV3._abre`, `_mede_incl` (`comando.py:751,757`). | `TwistV3._fecha`. |
| `v3_fase_marcha`, `v3_marcha_on`, `v3_acerto_marcha` | float, bool, float | Fase do relógio de marcha, comando acima do limiar, acerto dos pés. | `TwistV3._update_metrics` (`comando.py:868-874`). | `recompensas.passo`, `observacoes.privilegiados`. |
| `v3_cur` | `Curriculo` | Estado do currículo (células, níveis, degraus, zona). | `curriculo.py:46-47`. | Toda a cadeia. |
| `v3_ref` | `Referencias` | Tabelas de referência da IK (seção 6.1). | `curriculo.py:50-51`. | Toda a cadeia. |
| `v3_iter` | int | Iteração do currículo da última chamada. | `curriculo.v3` (`curriculo.py:122`). | `curriculo.v3`. |

### 3.4 Atributos internos de `CadeiaCmd`

`comando.py:291-306`: `tab` e `faixas` (tabela de trechos), `relogio` (tempo do trecho), `alvo_s` (duração ou atraso sorteado), `pega` (PEGAR aberto sem registro), `bota` (BOTAR aberto sem registro), `sit_palmas`, `sigma_min`, `it_zona`, `tempos` (permanência por etapa), `vis_e` (etapa da visita), `t_vis` (tempo da visita).

## 4. Fatias com caixa e cena do episódio

### 4.1 Quando cada fatia abre e quanto recebe

Fatia ANDAR: abre na iteração 0 (`balanceador.py:110`). Fatia B: abre quando o nível do ANDAR chega a `Balanceador.nivel_abre_b` = 5 (`knobs.py:396`); a abertura abre juntas as células `CARREGAR reto` e `PEGAR pegar` (`balanceador.py:209-211`). Fatia C: abre quando a célula do PEGAR passa o portão, isto é `s >= Balanceador.portao` = 0,75 (`knobs.py:391`; `balanceador.py:122-123,231-232`).

Fração de cada fatia aberta (`balanceador.py:282-293`): o ANDAR tem teto 1. B e C têm teto `rampa_fatia_inicio + rampa_fatia_passo · degraus_da_rampa`, com `Balanceador.rampa_fatia_inicio` = 0,05 (`knobs.py:397`) e `rampa_fatia_passo` = 0,02 (`knobs.py:398`). As fatias abertas, em ordem crescente de teto, recebem `min(teto, resto / fatias_restantes)`. O degrau da rampa sobe 1 a cada `Balanceador.rampa_fatia_its` = 12 fechos de iteração desde a abertura da fatia (`knobs.py:399`; `balanceador.py:189-193`). Fatia fechada recebe 0.

### 4.2 Topo da laje, massa e posição da caixa (fatias B e C)

`posiciona_cena` (`eventos.py:332-367`) e `sorteia_topo_e_massa` (`eventos.py:82-109`), no reset:

- `Niveis.pegar` = ((0,55; 1), (0,45; 2), (0,30; 3), (0,15; 4), (0,04; 5)) (`knobs.py:417`): par (topo h_k em m; massa m_k em kg) por nível. `h_0` = 0,55 m.
- `extra` = `Niveis.subnivel_extra[subnível do PEGAR]`, com `subnivel_extra` = (0,10; 0,0) m (`knobs.py:422`). O subnível começa em 0 e sobe uma vez para 1 (`balanceador.py:273-276`).
- Fatia B: topo ~ U[h_k + extra; h_0 + extra]; massa ~ U[`caixa_massa_base`; m_k]. `Cena.caixa_massa_base` = 1,0 kg (`knobs.py:80`).
- Fatia C: topo ~ U[max(h_k; `Niveis.topo_min_c`) + extra; h_0 + extra], com `topo_min_c` = 0,30 m (`knobs.py:419`). Massa ~ U[1,0; m_j], com j = o maior nível do PEGAR que já passou no portão (`nivel_passou`; `balanceador.py:350-356`), e m_j = 1,0 kg se nenhum passou.
- Fatia ANDAR: topo 0 e laje e caixa estacionadas a `Cena.estaciona_z` = 20,0 m acima da origem (`knobs.py:114`; `eventos.py:310-319`).
- Posição: a laje fica em `Cena.prateleira_xy` = (0,50; 0,00) m. A caixa fica em `Cena.caixa_xy` = (0,32; 0,00) m mais dx e dy (tabela da seção 2.4), com o centro a `a` acima do topo (`knobs.py:94,96`; `eventos.py:297-307`). A laje tem meia largura `Cena.prateleira_meia_xy` = 0,30 m e meia espessura `prateleira_meia_z` = 0,02 m (`knobs.py:89-90`).
- Terreno da fatia B (`eventos.py:112-130`): em episódio de terreno com coluna diferente de 0, a cena gira por k·π/2 e o centro da laje fica a `Terreno.laje_do_centro` do centro da plataforma. Fatia C usa sempre o plano (`eventos.py:70-74`).
- Massa e inércia: `carga` escreve a massa de `v3_massa` e a inércia (2/3)·m·a² do cubo (`eventos.py:443-461`).

### 4.3 Base do robô

- Fatia ANDAR: `reset_root_state_uniform` com x ~ U[−0,5; 0,5] m, y ~ U[−0,5; 0,5] m e yaw ~ U[−π; π] (`Cena.reset_base_andar`, `knobs.py:103`; `eventos.py:224`).
- Fatias B e C (`eventos.py:178-208`): a base fica atrás da caixa, no frame do rumo da cena, à distância d = `ref.recuo(topo, a)` + U[−1; 1]·`Alvos.faixa_recuo`·`jitter`. `Alvos.faixa_recuo` = 0,08 m (`knobs.py:227`). `jitter` = `Niveis.subnivel_jitter[subnível do PEGAR]`, com `subnivel_jitter` = (0,0; 1,0) (`knobs.py:424`): no subnível 0 não há jitter. O yaw é rumo + U[−0,2; 0,2] rad e o z soma U[0,01; 0,05] m (`Cena.reset_base_manipula`, `knobs.py:101`).

## 5. Sequência de tarefas dentro de um episódio

### 5.1 Tabela de trechos

`_monta` (`comando.py:171-194`) monta a tabela. Cada trecho tem tarefa, modo de término e faixa de sorteio `(lo, hi)` em segundos. Um trecho "por tempo" dura `alvo_s`. Um trecho "por aviso" espera o aviso e depois `alvo_s` de atraso.

| Fatia | Trecho | Tarefa | Término | Faixa (knob, valor, linha) |
|---|---|---|---|---|
| ANDAR | 1 | ANDAR | tempo | `Cadeia.episodio_s` = 20,0 s (`knobs.py:160`) |
| B | 1 | ANDAR (parado) | tempo | `Cadeia.espera_s` = (0,5; 1,0) s (`knobs.py:154`) |
| B | 2 | PEGAR | aviso, depois atraso | `Cadeia.atraso_pegar_s` = (0,0; 1,0) s (`knobs.py:158`) |
| B | 3 | CARREGAR | tempo | `Cadeia.carregar_b_s` = (4,0; 7,0) s (`knobs.py:155`) |
| C | 1 | ANDAR (parado) | tempo | `Cadeia.espera_s` = (0,5; 1,0) s |
| C | 2 | PEGAR | aviso, depois atraso | `Cadeia.atraso_pegar_s` = (0,0; 1,0) s |
| C | 3 | CARREGAR (parado) | tempo | `Cadeia.carregar_c_s` = (0,5; 1,0) s (`knobs.py:156`) |
| C | 4 | BOTAR | aviso, depois atraso | `Cadeia.atraso_botar_s` = (0,0; 1,5) s (`knobs.py:159`) |
| C | 5 | ANDAR (parado) | tempo | `Cadeia.andar_final_s` = 1,5 s (`knobs.py:157`); faixa (1,5; 1,5) |

O ANDAR das fatias B e C tem comando zero: o twist só sorteia velocidade para ANDAR na fatia ANDAR e para CARREGAR na fatia B (`comando.py:905`). O CARREGAR da fatia C fica parado.

O trecho inicial da fatia é `tab.inicio[fatia]`, aberto no reset (`comando.py:659-660`). O último trecho de cada fatia tem `proximo` = −1 (`comando.py:185`).

### 5.2 Avanço do trecho

`avanca_trecho` (`comando.py:201-216`), chamada a cada passo com `duracao = atraso = alvo_s`:

- Trecho por tempo: o relógio soma `dt` a cada passo. Avança quando `relogio >= alvo_s − 1e-6` (`_EPS`, `comando.py:35`).
- Trecho por aviso: o relógio só soma `dt` com `v3_aviso` ligado. Avança quando `v3_aviso` está ligado e `relogio >= alvo_s − 1e-6`. Com `alvo_s` = 0 avança no passo em que o aviso liga.
- Ao avançar o relógio volta a 0. No último trecho (`proximo` < 0) o avanço liga `fim` e o índice do trecho não muda (`comando.py:213-215`).

O aviso (`comando.py:606-607`): `fecho = v3_bota_fechada` se a tarefa é BOTAR, senão `v3_pega_fechada`. O latch recebe `fecho` apenas se o trecho é "por aviso". Por isso o aviso do PEGAR liga no fecho do P3, e o do BOTAR liga no fecho do B3. O trecho PEGAR não tem limite de tempo: ele dura até o fecho do P3 e o atraso, ou até o fim do episódio.

### 5.3 PEGAR vira CARREGAR

1. O trecho PEGAR abre sem alvo (`v3_alvo_idx` = −1; `comando.py:585-587`). A tarefa publicada é PEGAR e o comando de velocidade é zero (`comando.py:905`).
2. `_abre_p0` (`comando.py:370-392`) abre o P0 na primeira leitura válida da caixa. A condição é `v3_alvo_idx < 0` e `v3_tarefa == PEGAR` e `v3_estimador.tem_leitura()`. `tem_leitura` vale quando já houve uma leitura entregue e `t − instante_da_leitura < Estimador.idade_max` = 2,0 s (`knobs.py:453`; `estimador.py:373-376`). Na abertura: `self.pega` liga (o PEGAR passa a ter registro pendente), `v3_alvo_idx` = P0, `v3_phi` = −1, `v3_h` = 1, `v3_phi_melhor` = 0, `v3_relogio_alvo` = 0, `v3_caixa_pega_w` e `v3_alvo_w` recebem a posição estimada da caixa no mundo (`do_frame_da_base` da saída do estimador) e `v3_sigma` recebe o σ do P0.
3. As etapas P0, P1, P2 e P3 fecham em ordem (seção 6).
4. No fecho do P3 `v3_pega_fechada` liga e o índice fica em P3 (`comando.py:492-493,496`). O aviso liga no mesmo passo.
5. Depois de `alvo_s` segundos de atraso (uniforme em `atraso_pegar_s`) o trecho avança e abre o CARREGAR. Durante o atraso a tarefa publicada continua PEGAR.
6. O trecho CARREGAR não zera o índice do alvo: o índice fica em P3 (só ANDAR e PEGAR zeram; `comando.py:585`).
7. Se o CARREGAR anda (fatia B com a soma dos módulos do comando maior que 0), `eventos.laje_fora` move só a laje para `Cena.estaciona_z` = 20,0 m acima da origem (`knobs.py:114`). A caixa não é movida por esse evento (`comando.py:592-594`; `eventos.py:322-329`).

### 5.4 CARREGAR vira BOTAR (fatia C)

1. O CARREGAR da fatia C dura `alvo_s` ~ U[0,5; 1,0] s, parado.
2. Ao avançar para o trecho BOTAR, `_abre_trecho` roda (`comando.py:588-591`): registra o PEGAR pendente com `falhou = falso` (seção 13) e chama `_abre_botar`.
3. `_abre_botar` (`comando.py:542-566`):
   - `posiciona_destino` sorteia e move a laje de destino e devolve o ponto de soltura (seção 6.4, "Laje de destino"). O ponto sobe `a` em z (`comando.py:549`). O viés só entra se o `v3_gerador` existe (`comando.py:550`).
   - Soma um viés uniforme: `(2u − 1) · [vxy, vxy, vz]`, com u ~ U[0; 1)³. `vxy` e `vz` vêm de `no_degrau(par[0], par[1], degrau_global, 5)`: `Estimador.vies_alvo_botar_xy` = (0,01; 0,05) m (`knobs.py:461`) e `Estimador.vies_alvo_botar_z` = (0,005; 0,03) m (`knobs.py:462`). A norma máxima no degrau 5 é √(0,05² + 0,05² + 0,03²) ≈ 0,077 m.
   - `v3_alvo_w` recebe o ponto com viés; `v3_alvo_idx` = B1; `v3_phi` = −1; `v3_h` = 1; `v3_phi_melhor` = 0; `v3_relogio_alvo` = 0; `v3_ponto_apoio` = 0; `v3_v_chegada` = −1; `v3_sigma` = σ do B1; `self.bota` liga.
4. B1, B2 e B3 fecham em ordem. O fecho do B3 liga `v3_bota_fechada`, e o índice vira −1 (`alvos.py:206-209`; `comando.py:498`).
5. O aviso liga no mesmo passo. Depois de `alvo_s` ~ U[0,0; 1,5] s o trecho avança para o ANDAR final.

### 5.5 Comando de velocidade do CARREGAR da fatia B

Na abertura do trecho CARREGAR da fatia B, `_publica` chama `_resample` do twist, que sorteia a célula e a velocidade (`comando.py:519,906-938`). A célula `terreno` depende do episódio de terreno, que o evento `fatia` já sorteou no reset. Em episódio de terreno a célula é `terreno`. Fora dele a célula é sorteada entre as abertas sem `terreno` (`sorteia_celulas` com `sem_tipo`). A velocidade vem de `sorteia_velocidade` (`comando.py:148-158`):

- Teto frontal = `Comando.frente_tetos[nível do CARREGAR]`, com `frente_tetos` = (0,4; 0,55; 0,7; 0,85; 1,0) m/s (`knobs.py:122`). Teto traseiro = min(frente; `tras_teto_max` = 0,5 m/s, `knobs.py:123`). Teto lateral = min(frente; `lado_teto_max` = 0,3 m/s, `knobs.py:124`). Teto de giro = `giro_tetos[min(nível; 3)]`, com `giro_tetos` = (0,3; 0,4; 0,5; 0,5) rad/s (`knobs.py:125`).
- Células `reto`, `terreno` e `arco`: escolhe uma de 4 direções com igual probabilidade (+x, −x, +y, −y), com módulo ~ U[piso; teto]. Piso em x = `piso_velocidade` = 0,25 m/s (`knobs.py:126`); piso em y = `piso_lado` = 0,2 m/s (`knobs.py:127`) (`comando.py:102-110`). Na célula `terreno`, frente e trás ficam limitadas por `terreno_v_max` = 1,0 m/s (`knobs.py:148`).
- Células `arco` e `giro_lugar`: giro = ±U[`piso_velocidade`; teto de giro] (`comando.py:113-115`). `giro_lugar` tem linear 0.
- O comando fica fixo até o fim do trecho: `time_left` = ∞ (`comando.py:940-941`).

### 5.6 Quando o episódio acaba

O episódio acaba no passo em que alguma terminação do `env_cfg.py:137-147` vale (seção 14.1). Para as fatias com caixa há quatro caminhos:

| Caminho | Condição | Ref. |
|---|---|---|
| Fim da cadeia | O trecho final avança: fatia ANDAR depois de `Cadeia.episodio_s` = 20,0 s; fatia B depois do CARREGAR (`carregar_b_s`); fatia C depois do ANDAR final (`andar_final_s` = 1,5 s). `avanca_trecho` liga `fim`, `compute` grava `v3_fim` e a terminação `time_out` lê `v3_fim` no passo seguinte. | `comando.py:213-216,614`; `terminacoes.py:53-72` |
| Tempo máximo | `episode_length_buf >= max_episode_length`. `max_episode_length` = `ceil(episodio_s / step_dt)` = `ceil(20,0 / 0,02)` = 1000 passos. | `mjlab/envs/manager_based_rl_env.py:266-283`; `knobs.py:54-55,160` |
| Queda, batente, chegada | Seção 14.1. | `terminacoes.py` |
| Caixa perdida | Com `caixa_perdida_vira_andar` ligada o episódio não acaba pela perda: o trecho congela e o episódio acaba pelo tempo máximo ou por outra terminação (seção 11.2). Com a chave desligada a terminação `caixa_perdida` acaba o episódio. | `comando.py:610-613`; `terminacoes.py:75-83` |

No reset seguinte o PEGAR e o BOTAR ainda abertos se registram (seção 13). O trecho PEGAR e o trecho BOTAR só acabam depois do aviso (fecho do P3 ou do B3) mais o atraso. Se a soma dos trechos passa de 1000 passos, o `time_out` corta a cadeia antes do `fim`.

## 6. Etapas da cadeia

### 6.1 Referências da IK (`referencias.npz`)

`alvos.Referencias` carrega o arquivo em float64 (`alvos.py:40-67`). `dados/gera_referencias.py:148-158` grava estas chaves:

| Chave | Forma | Conteúdo |
|---|---|---|
| `topos`, `meias` | (6,), (3,) | Nós da grade: altura da laje e meia aresta. Valores lidos: topos 0; 0,04; 0,15; 0,30; 0,45; 0,55 m; meias 0,07; 0,10; 0,13 m. |
| `grade_pelve`, `grade_tronco`, `grade_recuo`, `grade_aceita` | (6,3) | Altura da pelve (m), inclinação do tronco (rad) e recuo (m) de pega por (topo, meia), vindos da tabela de IK de agachado. Interpolação bilinear (`alvos.py:30-37, 69-76`). `grade_aceita` é gravada e não é lida pela classe. |
| `caixa_transporte_b` | (3,3) | Centro da caixa no frame da pelve, pose de pé com caixa, por meia aresta. Valores lidos: x ≈ 0,25 m; z ≈ 0,111 m; y ≈ 0. |
| `palmas_com_caixa_b` | (3,2,3) | Palmas no frame da pelve, pose de pé com caixa. |
| `q_com_caixa` | (3,29) | Juntas da pose de pé com caixa. |
| `pelve_com_caixa` | (3,) | Altura da pelve com caixa. Valores lidos: ≈ 0,7888 m nas três meias. |
| `q_de_pe`, `palmas_de_pe_b`, `pes_de_pe_b`, `cantos_sola`, `pelve_de_pe`, `site_pe_de_pe` | vários | Pose de pé sem caixa. `pelve_de_pe` lido: ≈ 0,7891 m. |
| `fonte`, `hash_origem` | texto | Procedência. |

As tabelas com caixa interpolam linearmente só na meia aresta `a` (`alvos.py:90-95`). A grade prende `h` e `a` nas bordas antes de interpolar (`alvos.py:22-27`).

Origem dos números (`dados/gera_referencias.py`): a grade vem da tabela de IK de agachado (`pelve_z`, `tronco_incl` em graus convertido para rad, e o recuo = norma xy entre a base e a caixa; linhas 96-112). `caixa_transporte_b`, `palmas_com_caixa_b`, `q_com_caixa` e `pelve_com_caixa` vêm das três poses de pé com caixa (uma por meia aresta); o centro da caixa sai no frame da pelve (linhas 70-73, 144-145, 151-152). `q_de_pe`, `palmas_de_pe_b`, `pes_de_pe_b`, `cantos_sola`, `pelve_de_pe` e `site_pe_de_pe` vêm da pose 0 da tabela de pé sem caixa (linhas 131-141, 153-155). `grade_pelve` (m), `grade_recuo` (m) e `grade_tronco` (rad), linhas = topos 0; 0,04; 0,15; 0,30; 0,45; 0,55 m, colunas = meias 0,07; 0,10; 0,13 m, lidas do arquivo:

| Topo (m) | `grade_pelve` (m) | `grade_recuo` (m) | `grade_tronco` (rad) |
|---|---|---|---|
| 0,00 | 0,249; 0,261; 0,268 | 0,365; 0,379; 0,390 | 0,738; 0,724; 0,676 |
| 0,04 | 0,282; 0,292; 0,298 | 0,380; 0,389; 0,392 | 0,792; 0,741; 0,655 |
| 0,15 | 0,372; 0,384; 0,394 | 0,393; 0,396; 0,399 | 0,795; 0,735; 0,675 |
| 0,30 | 0,523; 0,540; 0,555 | 0,438; 0,435; 0,433 | 0,942; 0,894; 0,847 |
| 0,45 | 0,626; 0,641; 0,655 | 0,369; 0,367; 0,366 | 0,751; 0,702; 0,653 |
| 0,55 | 0,673; 0,690; 0,705 | 0,337; 0,334; 0,331 | 0,559; 0,519; 0,473 |

O script só grava o arquivo se a FK confere: cada palma a no máximo `TOL_PALMA` = 0,05 m do plano da face lateral e a pelve da FK a no máximo `TOL_PELVE` = 0,001 m de `qpos[2]` (`gera_referencias.py:26-27, 121-129`).

### 6.2 Medidas do passo

`_mede` (`comando.py:317-346`) usa o estado verdadeiro, não o estimado:

| Medida | Definição |
|---|---|
| `h` | Altura da pelve sobre o terreno (sensor `altura_pelve`). |
| `pelve` | z da raiz menos a sola mais baixa (`z_pe_min`). |
| `palmas_c` | Sítios das palmas no frame da caixa (frame da caixa girado pelo rumo da base). |
| `incl_graus` | Ângulo entre a face mais vertical da caixa e a vertical (`alvos.py:131-135`). |
| `contatos` | (E,2): sensores `palma_E` e `palma_D` com `found > 0`. Cada sensor mede o contato do pad da palma com a geom da caixa (`cena.py:348-353`). |
| `apoio` | Sensor `apoio_caixa` com `found > 0`: contato caixa × laje (`cena.py:361-367`). |
| `sobre` | O centro da caixa, no frame da laje, tem `abs(x) <= prateleira_meia_xy` e `abs(y) <= prateleira_meia_xy`. |
| `h_caixa` | Altura da origem da caixa sobre o terreno (sensor `altura_caixa`; raios no grupo de geoms 0, `cena.py:330-340, 397`). |
| `v_rel` | Norma da velocidade linear da caixa menos a da base. |
| `tronco_graus` | Inclinação do torso (`recompensas.incl_torso`). |
| `v_caixa` | Velocidade linear da caixa, absoluta. |
| `apoio_N` | Soma, sobre os slots, do valor absoluto da componente vertical da força do sensor `apoio_caixa` (`comando.py:255-266`). |
| `peso_N` | `v3_massa` × módulo da gravidade do modelo. |
| `dpalmas` | A maior das duas distâncias entre cada palma, no frame da base, e a palma na pose de pé (`palmas_de_pe_b`). |

### 6.3 Invariante

O invariante é `desloc · incl · maos` (`comando.py:466`), calculado por `portoes_do_alvo` com os knobs finais `Alvos` (`comando.py:463-465`). Cada fator vale 1 dentro do limite e cai linearmente a 0 fora. A função `rampa(x, lim, Δ)` vale `clip(1 − (x − lim)/Δ, 0, 1)` (`potencial.py:23-29`). O fecho exige invariante igual a 1.

| Fator | Etapas | Valor 1 quando |
|---|---|---|
| `desloc` | P0, P1, P2 | O deslocamento xy da caixa desde `v3_caixa_pega_w` é no máximo `desloc_p0p2_frac · a`. `Alvos.desloc_p0p2_frac` = 1,0 (`knobs.py:222`). A rampa tem `delta_p0p2_frac · a`, com `delta_p0p2_frac` = 0,5 (`knobs.py:223`). O deslocamento zera se a tarefa é BOTAR ou a caixa foi perdida (`comando.py:460`). |
| `desloc` | B2, B3 | O deslocamento xy da caixa desde `v3_ponto_apoio` é no máximo `Alvos.desloc_b2b4` = 0,02 m (`knobs.py:224`); rampa `delta_b2b4` = 0,06 m (`knobs.py:225`). Zera se a tarefa não é BOTAR ou a caixa foi perdida (`comando.py:461`). |
| `desloc` | P3, B1 | Sempre 1. |
| `incl` | P2, P3, B1, B2 | `incl_graus <= incl_lim`. `incl_lim` é `no_degrau(30, 15, v3_degrau, 5)` (`recompensas.py:462-464`), com `Alvos.incl_inicio_graus` = 30,0 (`knobs.py:203`) e `incl_fim_graus` = 15,0 (`knobs.py:204`). Rampa `delta_incl_graus` = 15,0 (`knobs.py:205`). |
| `incl` | outras | Sempre 1. |
| `maos` | P3, B1 | Os dois sensores de palma têm contato: `contatos.sum / 2 == 1`. |
| `maos` | outras | Sempre 1. |

Resumo por etapa: P0 e P1 só `desloc`; P2 `desloc` e `incl`; P3 `incl` e `maos`; B1 `incl` e `maos`; B2 `desloc` e `incl`; B3 só `desloc`.

### 6.4 Tabela das etapas

Os limites abaixo são os do `Alvos` efetivo da zona (seção 7). A coluna "Final" é o valor do campo em `Alvos`, que vale com z = 0. O tempo é o tempo seguido em que a condição vale. A condição tem de valer em todo passo do tempo; um passo falso zera o relógio da etapa (`alvos.py:203-204`).

| Etapa | O que é, geometricamente | Condição de fecho | Final (knob, linha) | Tempo |
|---|---|---|---|---|
| P0 | As palmas ao lado da caixa, do lado do robô, 5 cm além da face próxima e afastadas da face lateral. No frame da caixa: x = −(a + `p0_frente`); y = ±(a + `palma_face` + `folga_p1`); z = 0 (`alvos.py:147-149`). | As duas palmas: `norm(palma_c − alvo) <= tol_palma` (`comando.py:468,52`), mais invariante. | `p0_frente` = 0,05 m (`knobs.py:175`); `palma_face` = 0,045 m (`knobs.py:178`); `folga_p1` = 0,02 m (`knobs.py:179`); `tol_palma` = 0,03 m (`knobs.py:183`). | 0 s (`comando.py:442`): fecha no primeiro passo em que a condição vale. |
| P1 | As palmas ao lado da face lateral, sem tocá-la. x = 0; y = ±(a + `palma_face` + `folga_p1`); z = 0. | Igual ao P0, com o alvo do P1. | Igual ao P0. | 0 s. |
| P2 | As almofadas das palmas encostadas nas faces laterais. x = 0; y = ±(a + `palma_face`); z = 0. | Os dois sensores de palma com contato (`contatos.all`, `comando.py:53`), mais invariante. | `tempo_p2` = 0,3 s (`knobs.py:182`). | `tempo_p2` efetivo. |
| P3 | Erguer a caixa e levá-la ao ponto de transporte na frente da pelve, com a pelve na altura de pé com caixa e a caixa fora da laje. | Todas: `subida >= subida_min_p3`; `d_transporte <= tol_transporte_p3`; `d_pelve <= tol_pelve_p3`; `~apoio`; invariante (`comando.py:54`). | `subida_min_p3` = 0,03 m (`knobs.py:186`); `tol_transporte_p3` = 0,05 m (`knobs.py:187`); `tol_pelve_p3` = 0,03 m (`knobs.py:188`). | `tempo_p3` = 0,5 s (`knobs.py:189`). |
| B1 | Levar a caixa ao ponto de soltura sobre a laje de destino e apoiá-la, ainda com as mãos. | Todas: `norm(caixa − alvo_B1) <= raio_b1`; `apoio_N >= apoio_b1_mg · peso_N`; `norm(v_caixa) < v_parada`; invariante (`comando.py:82-83`). | `raio_b1` = 0,10 m (`knobs.py:191`); `apoio_b1_mg` = 0,8 (`knobs.py:192`); `v_parada` = 0,05 m/s (`knobs.py:195`). | `tempo_b1` = 0,3 s (`knobs.py:194`). |
| B2 | Soltar a caixa: as palmas se afastam das faces e a caixa fica apoiada e parada. | Todas: as duas palmas com `afasta >= b2_afasta`; `apoio`; `norm(v_caixa) < v_parada`; invariante (`comando.py:84`). | `b2_afasta` = 0,05 m (`knobs.py:196`). | `tempo_b2` = 0,5 s (`knobs.py:197`). |
| B3 | Voltar à pose de pé, com as mãos ao lado do tronco. | Todas: `abs(pelve − pelve_de_pe) <= tol_pelve_b3`; `abs(tronco_graus) <= tol_tronco_b3_graus`; `dpalmas <= tol_palmas_b3`; invariante (`comando.py:85-86`). | `tol_pelve_b3` = `Detector.tol_pelve` = 0,03 m (`knobs.py:198,166`); `tol_tronco_b3_graus` = `Detector.tol_tronco_graus` = 5,0 graus (`knobs.py:199,167`); `tol_palmas_b3` = `Detector.tol_palma` = 0,05 m (`knobs.py:200,168`). | `tempo_b3` = 0,5 s (`knobs.py:201`). |

Definição das grandezas de fecho (`comando.py:468-470,473,477-478`):

- `d_palmas` por palma = distância 3D, no frame da caixa, entre `palmas_c` e `palmas_alvo_caixa(idx)`, calculada com o `Alvos` final.
- `subida` = `h_caixa − a − v3_topo`: o fundo da caixa sobre o topo da laje de pega, com a caixa verdadeira.
- `d_transporte` = distância entre o centro da caixa e o ponto de transporte. O ponto de transporte é a posição da pelve verdadeira mais `R(rumo)·caixa_transporte_b(a)[:2]` no plano, e z = chão local + `Alvos.z_transporte` = 0,90 m (`knobs.py:202`; `alvos.py:163-179`). O chão local é o z da raiz menos o sensor `altura_pelve` (`recompensas.py:487-493`). O z da tabela é descartado.
- `d_pelve` = `abs(h − pelve_com_caixa(a))`.
- `afasta` por palma = o maior, sobre os 3 eixos, de `abs(palma_c) − (a + palma_face)` (`comando.py:478`). Uma palma conta como afastada quando pelo menos um eixo tem `abs(palma_c)` maior ou igual a `a + palma_face + b2_afasta`. Os outros eixos podem ficar dentro da distância de contato.
- `alvo_B1` (no fecho do B1) = `v3_alvo_w`, o ponto de soltura com o viés de `_abre_botar` (`recompensas.py:487-493`, `alvos.py:163-179`). O fecho compara a posição verdadeira da caixa com esse ponto enviesado.
- `pelve` do B3 = `m["pelve"] − pelve_de_pe` (z da raiz menos a sola mais baixa).
- `tronco_graus` do B3 = inclinação do torso sem subtrair referência.

Efeitos do fecho (`comando.py:492-504`):
- O fecho do B1 grava `v3_ponto_apoio` (`comando.py:494-495`).
- O fecho do P2 liga `v3_pegou` (`comando.py:497`).
- O fecho do P3 liga `v3_pega_fechada` e mantém o índice em P3 (`comando.py:492-493,496`).
- O fecho do B3 liga `v3_bota_fechada`; o índice vira −1.
- Nos outros fechos o índice sobe 1 (`alvos.py:205-210`).
- Quando o índice sobe e continua válido: `v3_sigma` recebe o σ da etapa nova, `v3_h` = 1, `v3_phi_melhor` = 0 (`comando.py:502-504`).
- O relógio da etapa volta a 0 em todo avanço (`alvos.py:209`).

Só uma etapa avança por passo: um único `alvos.avanca` roda (`comando.py:482`).

Laje de destino (`eventos.py:373-413`), no momento da abertura do BOTAR:
- θ ~ U[−10°; 10°] (`Cena.destino_giro_max_graus` = 10,0, `knobs.py:87`); y ~ U[−0,10; 0,10] m (`Cena.destino_jitter_y` = 0,10, `knobs.py:86`).
- longe ~ U[0; `jitter_x_max[nível do BOTAR]`] m (`knobs.py:99`).
- topo ~ U[`botar[nível]`; `botar[0]`] m, com `Niveis.botar` = (0,55; 0,45; 0,30; 0,15; 0,04) (`knobs.py:418`).
- Se o subnível do BOTAR é 0 (menor que o último): longe ~ U[0; `subnivel_longe[0]`] e topo = `botar[0] + subnivel_extra[0]` fixo. `Niveis.subnivel_longe` = (0,10; 0,20) m (`knobs.py:423`); `subnivel_extra` = (0,10; 0,0) m (`knobs.py:422`).
- Recuo = `ref.recuo(topo, a) + longe`. O alvo no frame da base é (recuo + e(θ); y), com e(θ) = borda·cos θ + meia_xy·sen abs(θ) − borda; o centro da laje é o alvo mais (meia_xy − borda)·(cos θ, sen θ) (`alvos.py:182-192`). `Cena.destino_borda` = 0,15 m (`knobs.py:85`); meia_xy = `prateleira_meia_xy` = 0,30 m.
- A laje é teleportada para base + R(rumo)·centro, com yaw = rumo + θ e topo no mundo = z da origem + topo (`eventos.py:410`). O ponto de soltura no mundo é base + R(rumo)·alvo, com z = topo (`eventos.py:412-413`).

## 7. Zona adaptativa

A zona z de cada etapa vai de 1 (critério largo) a 0 (critério final). Todas as 7 etapas começam com z = 1,0 (`balanceador.py:105`).

### 7.1 O que o z muda

A cada iteração do PPO (quando `cur.it` muda; `comando.py:428-432`), a cadeia refaz o `Alvos` efetivo: `campo_efetivo = final + z·(largo − final)` (`no_degrau(final, largo, z, 1)`; `comando.py:433-441`). As entradas vêm de `Alvos.zona` (`knobs.py:211-220`):

| Etapa | Campo | Final (z = 0) | Largo (z = 1) | O que o z muda |
|---|---|---|---|---|
| P0 | `tol_palma` | 0,03 m | 0,05 m | Tolerância das duas palmas no fecho do P0. |
| P1 | `tol_palma` | 0,03 m | 0,05 m | Tolerância das duas palmas no fecho do P1. |
| P2 | `tempo_p2` | 0,3 s | 0,2 s | Tempo de contato das duas palmas. |
| P3 | `subida_min_p3` | 0,03 m | 0,01 m | Subida mínima da caixa. |
| P3 | `tol_transporte_p3` | 0,05 m | 0,20 m | Distância ao ponto de transporte. Também lida pela qualidade do CARREGAR. |
| P3 | `tol_pelve_p3` | 0,03 m | 0,10 m | Distância da pelve à altura de referência. |
| P3 | `tempo_p3` | 0,5 s | 0,2 s | Tempo de permanência. |
| B1 | `raio_b1` | 0,10 m | 0,20 m | Raio da caixa ao ponto de soltura. |
| B1 | `tempo_b1` | 0,3 s | 0,1 s | Tempo de permanência. |
| B2 | `b2_afasta` | 0,05 m | 0,02 m | Afastamento mínimo das palmas. |
| B2 | `tempo_b2` | 0,5 s | 0,2 s | Tempo de permanência. |
| B3 | `tol_pelve_b3` | 0,03 m | 0,06 m (2·`Detector.tol_pelve`) | Tolerância da pelve. |
| B3 | `tol_tronco_b3_graus` | 5,0 graus | 10,0 graus (2·`Detector.tol_tronco_graus`) | Tolerância do tronco. |
| B3 | `tol_palmas_b3` | 0,05 m | 0,10 m (2·`Detector.tol_palma`) | Tolerância das palmas ao lado do tronco. |
| B3 | `tempo_b3` | 0,5 s | 0,2 s | Tempo de permanência. |

A zona não afrouxa `apoio_b1_mg` nem `v_parada` (não estão em `Alvos.zona`).

Os tempos de permanência por etapa são `[0, 0, tempo_p2, tempo_p3, tempo_b1, tempo_b2, tempo_b3]`, todos efetivos (`comando.py:442-443`).

O `Alvos` efetivo vale no fecho (`ok_agora_pegar` e `ok_agora_botar`, `comando.py:474,481`), nos tempos de permanência e na qualidade do CARREGAR (`recompensas.py:587`). O Φ, a distância d, os portões, o invariante e o ponto-alvo usam o `Alvos` final (`comando.py:355-357,463-465`; `recompensas.py:545,564-568`).

### 7.2 Como o z sobe e desce

A zona de cada etapa fica salva no estado do currículo (`balanceador.py:403,425-429`). O z também entra no crítico (`privilegiados`) e decide se o registro do PEGAR e do BOTAR conta como sucesso (seção 13).

`Curriculo.fecha_iteracao` (`balanceador.py:169-176`) faz estes dois passos uma vez por iteração, depois da atualização da nota das células:

1. Para cada etapa `e` com `vis_acum[e] >= Balanceador.zona_n_min` = 200 (`knobs.py:406`): `taxa = fech_acum[e] / vis_acum[e]`; `z[e] = clip(z[e] − zona_ganho·(taxa − zona_t_alvo), 0, 1)`; os acumuladores da etapa zeram. `zona_ganho` = 0,05 (`knobs.py:405`); `zona_t_alvo` = 0,6 (`knobs.py:404`). Taxa acima de 0,6 aperta (z cai); abaixo afrouxa (z sobe).
2. Em seguida, para P1, P2, P3, B2 e B3: `z[e] = max(z[e], z[e−1] − zona_folga_cadeia)`, com `zona_folga_cadeia` = 0,2 (`knobs.py:408`). O P0 e o B1 não têm essa trava.

### 7.3 O que é uma visita

Uma visita é o período em que a etapa está aberta (`comando.py:444-452`). A cadeia mantém `vis_e` (etapa da visita, −1 sem visita) e `t_vis` (tempo desde a abertura, −1 com a visita encerrada).

- `etapa` = índice do alvo se `v3_alvo_idx >= 0` e não é (P3 com `v3_pega_fechada`) e não houve perda; senão −1 (`comando.py:447`).
- A visita abre quando `etapa` muda para um valor válido: `t_vis` recebe 0 e soma `dt` no mesmo passo, então vale `dt` no passo da abertura. Depois soma `dt` a cada passo (`comando.py:448-452`).
- Fecho com visita: `fecho = av & (t_vis >= 0)` conta uma visita e um fecho (`comando.py:483,486-490`).
- Visita sem fecho: `venceu = ~av & (t_vis >= Balanceador.zona_t_visita)` com `zona_t_visita` = 8,0 s (`knobs.py:407`) conta só uma visita. A etapa trocar com a visita aberta (a tarefa mudou ou houve perda) também conta uma visita sem fecho (`comando.py:449`).
- Depois de encerrada, a visita não reabre até a etapa mudar (`comando.py:491`).
- No reset, a visita aberta conta uma visita sem fecho (`comando.py:647-650`).

## 8. Relógios, trechos e freio

| Relógio | Onde | Regra |
|---|---|---|
| Relógio da etapa `v3_relogio_alvo` | `alvos.avanca` (`alvos.py:195-210`) | Soma `dt` enquanto o fecho vale; zera quando o fecho falha. A etapa avança quando `rel >= tempo − 1e-9` e o fecho vale e o índice é válido. |
| Relógio do trecho `relogio` | `avanca_trecho` (`comando.py:201-216`) | Seção 5.2. |
| Relógio da visita `t_vis` | `_alvos` | Seção 7.3. |
| Relógio de marcha `v3_fase_marcha` | `TwistV3._update_metrics` (`comando.py:863-874`) | Anda só com comando acima do limiar. Fora do assunto desta cadeia. |

Trechos: a tabela, as faixas e os modos estão na seção 5.1; o aviso e o congelamento por perda estão nas seções 5.2 e 11.

Freio:
- `v3_freio` é `niveis_abertos(tarefa)`, isto é, nível da tarefa + 1 (`balanceador.py:335-336`). `_publica` o grava a cada abertura de trecho para as tarefas 0 a 3 (`comando.py:513-518`).
- O termo `freio` (`recompensas.py:899-919`) vale em PEGAR, CARREGAR e BOTAR (`env_cfg.py:51`). O regime vem do comando: `total = norm(cmd_xy) + abs(cmd_z)`; parado se `total < freio_limiar_andar` = 0,05 (`knobs.py:308`); correndo se `total >= freio_limiar_correr` = 1,5 (`knobs.py:309`); senão andando. O custo é a soma, sobre as 29 juntas, de `relu(abs(q̇)/vmax − 1)²`, com `vmax` das tabelas `vel_max_parado`, `vel_max_andando` e `vel_max_correndo` (`knobs.py:311,318,325`).
- O valor devolvido é `custo · freio_fator ** v3_freio · rampa`. `freio_fator` = 1,5 (`knobs.py:305`). `rampa = min(s_t / portao, 1)`, com `s_t` a média de `s` das células abertas da tarefa (0 sem célula aberta) e `portao` = 0,75 (`knobs.py:391`). O peso da recompensa é `Recompensa.freio` = −6/29 (`knobs.py:304`).

## 9. Distância d, σ e potencial Φ

### 9.1 Distância d

`distancia_do_alvo` (`recompensas.py:224-244`). É 0 com índice −1.

| Etapa | d |
|---|---|
| P0, P1, P2, B2 | Média, sobre as duas palmas, da distância entre `palmas_c` e `palmas_alvo_caixa(idx)` no frame da caixa. |
| P3 | `hypot(norm(caixa − alvo), Δpelve · tol_transporte_p3 / tol_pelve_p3)`. A razão usa os valores finais (0,05 / 0,03). |
| B1 | `hypot(norm(caixa − alvo), escala_apoio_b1 · deficit)`, com `deficit = relu(1 − apoio_N / (apoio_b1_mg · peso_N))` e `Alvos.escala_apoio_b1` = 0,10 m (`knobs.py:193`). |
| B3 | `sqrt((Δpelve / tol_pelve_b3)² + (Δtronco / tol_tronco_b3_graus)² + (dpalmas / tol_palmas_b3)²)`. Adimensional. |

No Φ (`_phi_env`, `recompensas.py:519-569`) a referência da pelve e do tronco é `ref_do_alvo(idx)`: pelve e tronco da grade (IK) para P0, P1, P2, B1, B2; `pelve_com_caixa(a)` e tronco 0 para P3; `pelve_de_pe` e tronco 0 para B3 (`observacoes.py:116-130`).

### 9.2 σ

σ é a distância d no instante em que a etapa abre, com um piso por etapa (`comando.py:359-362`): `sigma = max(d, sigma_min_etapa[etapa])`. `Alvos.sigma_min_etapa` = (0,05; 0,05; 0,08; 0,20; 0,20; 0,05; 0,0) para (P0, P1, P2, P3, B1, B2, B3) (`knobs.py:208`). O B3 não tem piso porque o d dele é adimensional. A conta usa o estado verdadeiro e o `Alvos` final.

σ é calculado:
- Na abertura do P0 (`comando.py:392`).
- A cada avanço de etapa que deixa o índice válido (`comando.py:503`).
- Na abertura do BOTAR (`comando.py:565`).
- Na perda da caixa, com alvo B3 (`comando.py:419`).
- No reset, `v3_sigma` = 1,0 (`comando.py:655`).

A medida de σ (`_dist`, `comando.py:348-357`) usa `dpelve = h − pelve_de_pe` em qualquer etapa e `dtronco = tronco_graus` sem referência. No P3 o Φ usa `h − pelve_com_caixa(a)`. Ver a divergência na seção 16.

### 9.3 Fórmula de Φ

`phi_do_alvo` (`recompensas.py:274-283`):

`Φ = (1 − tanh(d / (largura_tanh · σ))) · fator · g_desloc · g_incl · g_maos · g_tronco · g_recuo · g_com`

- `Alvos.largura_tanh` = 2,0 (`knobs.py:221`). Com d = σ o núcleo vale `1 − tanh(0,5)` ≈ 0,538.
- σ tem piso 1e-6 só no denominador (`recompensas.py:281`).
- `fator` é o fator de pouso no B1 e 1 nas outras etapas.
- Φ vale 0 com índice −1.

Portões do Φ (`portoes_do_alvo`, `recompensas.py:247-271`, chamado com `maos` = 1 em `recompensas.py:566`):

| Portão | Etapas | Rampa |
|---|---|---|
| `desloc` | P0, P1, P2 | `rampa(desloc_pega, desloc_p0p2_frac·a, delta_p0p2_frac·a)`; desloc_pega = deslocamento xy da caixa desde `v3_caixa_pega_w`. |
| `desloc` | B2, B3 | `rampa(desloc_apoio, desloc_b2b4, delta_b2b4)`; zero de deslocamento se `v3_perdeu`. |
| `incl` | P2, P3, B1, B2 | `rampa(incl_graus, incl_lim, delta_incl_graus)`. |
| `maos` | P3, B1 | 1 sempre (o valor passado é `ones_like`). |
| `tronco` | todas menos B3 | `rampa(relu(incl_torso − tronco_ref_graus), faixa_tronco_graus, delta_tronco_graus)`; `faixa_tronco_graus` = 15,0 (`knobs.py:226`); `delta_tronco_graus` = 15,0 (`knobs.py:229`). |
| `recuo` | todas menos B3 | `rampa(abs(recuo − recuo_ref), faixa_recuo, delta_recuo)`; `faixa_recuo` = 0,08 m (`knobs.py:227`); `delta_recuo` = 0,30 m (`knobs.py:230`). `recuo` = distância xy da caixa à base. `recuo_ref` = norma xy de `caixa_transporte_b(a)` no P3 e no CARREGAR; senão `ref.recuo(topo, a)` (`recompensas.py:509-516`). |
| `com` | todas | Com os dois pés em contato: `rampa(relu(faixa_com − margem), 0, delta_com)`, com `faixa_com` = 0,03 m (`knobs.py:228`) e `delta_com` = 0,03 m (`knobs.py:231`); senão 1. `margem` é a distância com sinal do CoM às arestas do fecho convexo das solas; o CoM inclui a caixa quando `v3_pegou`, não houve perda e o alvo é P3, B1 ou B2 (`recompensas.py:419-435`). |

Fator de pouso (B1, `potencial.fator_pouso`, `potencial.py:40-52`): `1 − s·(1 − exp(−(vz/σ_pouso)²))`, com `s = clip((altura_pouso_inicio − fundo)/(altura_pouso_inicio − altura_pouso), 0, 1)`. `fundo` = z da caixa − a − (z da origem + `v3_topo_destino`). `altura_pouso_inicio` = 0,10 m (`knobs.py:370`); `altura_pouso` = 0,05 m (`knobs.py:369`). `σ_pouso = no_degrau(1,0; 0,35; v3_degrau; 5)` com `Recompensa.sigma_pouso` = (1,0; 0,35) m/s (`knobs.py:368`).

### 9.4 Para que serve o Φ: o termo `progresso`

`progresso` (`recompensas.py:816-884`) vale em todas as tarefas (`env_cfg.py:51`) e entra com peso 1 (`env_cfg.py:57`). O mjlab multiplica toda recompensa por `step_dt` = 0,02 s (`scale_rewards_by_dt` = True: `mjlab/envs/manager_based_rl_env.py:157`; `mjlab/managers/reward_manager.py:118, 127`). Valor por segundo:

`r = renda_etapa · etapa · h + renda_phi · Φ_pago + renda_conquista · conquistas`

com `Recompensa.renda_etapa` = 6,0 (`knobs.py:339`), `renda_phi` = 3,0 (`knobs.py:338`), `renda_conquista` = 2,0 (`knobs.py:343`) (`recompensas.py:286-291`).

| Condição (por precedência) | `etapa` | Φ pago |
|---|---|---|
| `v3_perdeu`, ou `~v3_pega_fechada` com tarefa diferente de PEGAR | 0 | 0 |
| `v3_bota_fechada` | 7 | 0 |
| `v3_pega_fechada` e `v3_alvo_idx < B1` (CARREGAR e atraso do PEGAR) | 4 | 0 |
| outro caso com alvo válido | índice do alvo (P0 = 0 ... B3 = 6) | Φ do alvo; no P3 é `Φ · c · q` |
| nenhum | 0 | 0 |

(`recompensas.py:851-854`). `c` = `clamp(1 − apoio_N/peso_N, 0, 1)`, o peso nas mãos; `q` = a pega firme (`recompensas.py:561-563,850`).

Conquistas, cada uma em [0; 1] (`recompensas.py:557-563`):
1. Palmas nas faces: `1 − tanh(d_faces / (largura_tanh · sigma_palmas))`, com `Recompensa.sigma_palmas` = 0,08 m (`knobs.py:344`); `d_faces` = a média de distância das palmas ao alvo do P2.
2. Pega firme `maos`: `tanh(min(F_esq; F_dir) / F_pega)`, com `F_pega = margem_pega · massa · g / (2·μ)`. `Recompensa.margem_pega` = 2,5 (`knobs.py:365`); μ = `DR.atrito_pad[0]` = 0,6 (`knobs.py:523`). As forças vêm das faces ±y da caixa (`forcas_por_face`, `recompensas.py:345-356`).
3. Peso pago: `c · maos`.

Soma (`recompensas.py:875-882`): as conquistas só pagam quando `mantem = ~sem_caixa & ~bota_fechada & (fecho_pega | alvo em (P3, B1) | alvo == P2)` (`_CAIXA_ALVO` = P3, B1; `recompensas.py:58`). O valor pago é `conquistas[0] + conquistas[1]·almofadas + conquistas[2]·com_peso`. `almofadas` = fração de palmas com contato no P2, 1 nos outros; `com_peso` = 1 no P3 e depois do fecho do P3, 0 nos outros.

Fator h (`recompensas.py:855-870`):
- Etapas P0 a B3 com Φ pago: `h ← min(1; h·exp(−dt/tau_renda) + recarga_renda · relu(Φ_pago_etapa − phi_melhor))`. `Recompensa.tau_renda` = 3,0 s (`knobs.py:347`); `recarga_renda` = 5,0 (`knobs.py:348`). `v3_phi_melhor` guarda o maior Φ da etapa.
- CARREGAR com pega fechada e sem perda (`carrega`): `h ← h + (dt/tau_renda)·(sinal − h)` (`recompensas.py:857,867`). O sinal tem dois ramos (`recompensas.py:859-866`):
  - Com comando linear ou de giro ativo (módulo maior ou igual a `Comando.limiar_canal` = 0,02, `knobs.py:138`): `g_marcha · (com_lin·clamp(r_lin; 0; 1) + com_ang·clamp(r_ang; 0; 1)) / max(nº de canais ativos; 1)`. `r_lin` e `r_ang` são o rastreio normalizado (`potencial.rastreio_normalizado`) e `g_marcha` o fator do twist (`comando.py:871`).
  - Sem comando: `clamp((1 + r_lin)·(1 + r_ang); 0; 1)`, sem `g_marcha`.
  - Nos dois ramos o sinal é multiplicado por `rampa(d_transporte; 0,05; 0,25)` (`Recompensa.qualidade_larga` = (0,05; 0,25) m, `knobs.py:351`), com `d_transporte` o `d` de `_qualidade` (`recompensas.py:572-587`).
- Fora dessas situações `h_ef` = 1 (`recompensas.py:870`).

Φ também vai ao crítico: `privilegiados` entrega `marcha_on`, `v3_h`, `relu(v3_phi_melhor − max(v3_phi, 0))` e `z` da etapa corrente (0 sem etapa) (`observacoes.py:298-318`).

## 10. Degrau de refino na cadeia

O degrau avança por tarefa em `_conta_refino` (`balanceador.py:234-248`): a tarefa tem de estar no último nível (e, nas tarefas com linha de terreno, com a célula `terreno` aberta e a linha de terreno no máximo), com a média de `s` das células abertas maior ou igual à média dos portões por `Balanceador.iters_refino` = 100 iterações seguidas (`knobs.py:400`). O degrau chega a `Balanceador.degraus` = 5 (`knobs.py:401`) e nunca desce.

`v3_degrau` publica o degrau da tarefa do env na abertura do trecho (`comando.py:516`). O degrau global é o menor degrau das quatro tarefas (`balanceador.py:341-343`).

O que o degrau aperta na cadeia, com o degrau d da tarefa corrente, salvo quando indicado:

| O que | Fórmula | Valores no degrau 0 a 5 | Ref. |
|---|---|---|---|
| Limite de inclinação da caixa (invariante, portão `incl` do Φ, qualidade do CARREGAR) | `30 + (15 − 30)·d/5` graus | 30; 27; 24; 21; 18; 15 | `recompensas.py:462-464`; `knobs.py:203-204` |
| σ do pouso no B1 (fator de pouso do Φ) | `1,0 + (0,35 − 1,0)·d/5` m/s | 1,00; 0,87; 0,74; 0,61; 0,48; 0,35 | `recompensas.py:549`; `knobs.py:368` |
| Terminação `chegada` no BOTAR: limite de velocidade de descida no primeiro toque | `1,0 + (0,25 − 1,0)·d/5` m/s | 1,00; 0,85; 0,70; 0,55; 0,40; 0,25 | `terminacoes.py:149`; `Terminacao.chegada` = (1,0; 0,25), `knobs.py:381` |
| Peso do custo `sacudida` e do custo `aperto` (P3 e B1) | `d/5` | 0; 0,2; 0,4; 0,6; 0,8; 1,0 | `recompensas.py:944,956` |
| Joelho do custo `contato_laje` (N) | `50 + (10 − 50)·d/5` | 50; 42; 34; 26; 18; 10 | `recompensas.py:890`; `Recompensa.laje_joelho_N` = (50,0; 10,0), `knobs.py:302` |
| Terminação `batente` | liga com `v3_degrau >= Refino.degrau_batente` = 1 | | `terminacoes.py:164,181`; `knobs.py:547` |

Com o degrau global g (usado em quem não é por tarefa):

| O que | Fórmula `no_degrau(ini, fim, g, 5)` | Ref. |
|---|---|---|
| Magnitudes do estimador da caixa (seção 12) | cada par `(início, refino)` | `observacoes.py:76-90, 189` |
| Deriva da odometria (seção 12) | `deriva_odo`, `deriva_rumo`, `deriva_pos_giro`, `deriva_rumo_giro` | `comando.py:247-250,623-625` |
| Viés do alvo de soltura do B1 | `vies_alvo_botar_xy`, `vies_alvo_botar_z` | `comando.py:553-557` |
| DR do refino: atraso, `rfi`, `pd`, `forca_motor`, `solref` | `Curriculo.dr()` | `balanceador.py:358-389` |

## 11. Caixa perdida

### 11.1 Detecção

`_perda` (`comando.py:394-420`) roda a cada passo, depois de `_mede`. A perda de um env vale (`comando.py:409-410`):

`perda = (caiu | arremesso | fora_da_laje) & (v3_tarefa != ANDAR) & (episode_length_buf > 0)`

`caixa_perdida_agora` (`comando.py:59-65`):

| Ramo | Condição exata | Knob |
|---|---|---|
| `caiu` | `h_caixa − a < caixa_folga_chao` (estrito) | `Terminacao.caixa_folga_chao` = 0,02 m (`knobs.py:379`) |
| `arremesso` | `v3_pegou` e `v_rel > v_solta` e `fora_do_alvo` | `Terminacao.v_solta` = 1,2 m/s (`knobs.py:380`) |
| `fora_da_laje` | índice do alvo entre P0 e P2 e o centro da caixa fora do topo da laje: `abs(x) > prateleira_meia_xy` ou `abs(y) > prateleira_meia_xy` no frame da laje | `Cena.prateleira_meia_xy` = 0,30 m (`knobs.py:89`) |

`fora_do_alvo` (`comando.py:407-408`): na tarefa BOTAR é `norm(caixa − v3_alvo_w) > Alvos.raio_b1` (final, 0,10 m, `knobs.py:191`); nas outras tarefas é sempre verdadeiro. `v_rel` é a norma da velocidade linear da caixa menos a da base.

O ramo `episode_length_buf > 0` evita a perda no primeiro passo do episódio, em que `h_caixa` ainda é o do episódio anterior (`comando.py:400-402,410`).

### 11.2 Efeito

`v3_perdeu_agora` recebe a perda do passo, para todos os envs (`comando.py:411`). Como a cadeia roda depois das terminações e das recompensas, os leitores só o veem no passo seguinte.

Com `Cadeia.caixa_perdida_vira_andar` = verdadeiro (`knobs.py:161`) (`comando.py:412-420`), nos envs com perda:

1. `v3_perdeu` = verdadeiro (fica até o reset).
2. `v3_alvo_idx` = B3; `v3_phi` = −1; `v3_h` = 1; `v3_phi_melhor` = 0; `v3_relogio_alvo` = 0; `v3_sigma` = σ do B3 (adimensional, piso 0).
3. `_publica(ANDAR)`: a tarefa vira ANDAR, com o `v3_freio` e o `v3_degrau` do ANDAR. O twist reamostra o comando, que fica zero nas fatias B e C (`comando.py:905`). Não há sorteio.
4. O trecho congela até o fim do episódio: `novo = trecho` e `avancou`, `fim` ficam falsos para quem tem `v3_perdeu` (`comando.py:610-613`). O episódio só acaba por `time_out` por tempo, por queda ou por outra terminação. O fim da cadeia não liga.
5. O B3 da caixa perdida fecha com o mesmo núcleo `ok_agora_botar`, com `invariantes` = 1 (o deslocamento é zerado e B3 não usa `incl` nem `maos`; `comando.py:454,460-461,475-481`). O fecho exige a pose de pé: `abs(pelve − pelve_de_pe) <= tol_pelve_b3`, `abs(tronco_graus) <= tol_tronco_b3_graus`, `dpalmas <= tol_palmas_b3`, mantidos por `tempo_b3`, tudo com o `Alvos` efetivo. Ao fechar, `v3_bota_fechada` liga e o índice vira −1.
6. Sucesso: a perda faz `sucesso_do_pegar` e `sucesso_do_botar` valerem falso (`comando.py:68-70,91-93`). O PEGAR aberto registra falha no reset (seção 13). O twist fecha o segmento de comando com falha (`comando.py:820`).

Com a chave desligada, `v3_perdeu` não liga e nada disso acontece; a terminação `caixa_perdida` termina o episódio no passo seguinte (`terminacoes.py:75-83`). O env_cfg passa a chave a esta terminação por `params` (`env_cfg.py:141-142`).

Outros leitores:
- `recompensas.perda_caixa`: `v3_perdeu_agora · Recompensa.perda_caixa` = 240,0 (`knobs.py:367`), com peso −1 (`env_cfg.py:57`) e escala `dt`: −4,8 por pulso (`recompensas.py:311-313,960-962`).
- `PPOPorTarefa.process_env_step` soma `γ · min(V̄_grupo, 0)` à recompensa de quem tem `v3_perdeu_agora` (`algoritmo.py:127-131`).
- Termos que pagam positivo (`vivo`, `altura_pe`, `passo`, rastreios) vão a zero com `v3_perdeu` (`recompensas.py:361-365,85-87`).
- `progresso`: etapa 0 e Φ 0 (`recompensas.py:851`).
- A observação `alvo_macro` mostra o slot "perda" (`observacoes.py:294`).

## 12. Estimador da caixa e odometria

### 12.1 Quem o usa

- Termo de observação `caixa` do ator e do crítico: 11 canais = `saida_b` (8) + alvo corrente (3). É zero em todos os canais quando a tarefa é ANDAR (`observacoes.py:172-199`). O alvo é `alvo_no_frame_da_base(v3_alvo_w, odo)`, zerado no CARREGAR e, no BOTAR, depois do aviso; fora do BOTAR, zerado sem alvo (`observacoes.py:195-197`). No PEGAR o alvo é a posição estimada gravada no P0.
- `CadeiaCmd._abre_p0`: lê `tem_leitura` e `saida_b` (seção 5.3).
- Métrica `idade_das_leituras`.

A cadeia decide fechos, perda e distâncias com a caixa verdadeira, não com a estimada. A caixa estimada define só o ponto de pega (`v3_caixa_pega_w`) e o ponto de soltura com viés (`v3_alvo_w`).

### 12.2 Quando roda

O passo do estimador roda uma vez por passo de política, na primeira leitura do termo `caixa` ou `mapa` (guarda `v3_percepcao_passo`; `observacoes.py:140-151`), com `t = common_step_counter · step_dt` (`observacoes.py:191`). O reset do estimador roda no evento `filtros`, depois da base (`eventos.py:228-241`).

### 12.3 Visibilidade

`visivel` (`estimador.py:111-126`), com a câmera D435i no site `d435` do torso (`cena.py:164-166`) e os 9 pontos da caixa (centro e 8 cantos; `observacoes.py:49-50, 166-169`):

- Cada ponto está no campo de visão se `abs(atan2(y, x)) <= radians(fov_h_graus)/2` e `abs(atan2(z, x)) <= radians(fov_v_graus)/2`, e a distância à câmera fica em `[alcance[0]; alcance[1]]`. `Estimador.fov_h_graus` = 69,0 (`knobs.py:455`); `fov_v_graus` = 42,0 (`knobs.py:456`); `alcance` = (0,28; 3,0) m (`knobs.py:457`).
- Um ponto é ocluído se o segmento câmera–ponto passa a menos que o raio de alguma das 8 cápsulas dos braços (`observacoes.py:187`).
- A caixa é visível se o centro está no campo e no alcance e pelo menos `pontos_visiveis_min` pontos estão no campo e sem oclusão. `Estimador.pontos_visiveis_min` = 3 (`knobs.py:470`).

### 12.4 Amostragem, ruído, perda, atraso

`EstimadorCaixa.passo` (`estimador.py:282-371`). O mundo da caixa verdadeira entra por posição e quaternion. As magnitudes vêm de `_magnitude` no degrau global g (`observacoes.py:76-90`): cada par `(início, refino)` vai por `no_degrau(início, refino, g, 5)`.

| Efeito | Quando sorteia | Distribuição | Magnitude (knob, início → refino) |
|---|---|---|---|
| Grade de amostras | a partir do primeiro passo depois do reset | uma amostra a cada `1/taxa_hz` s | `Estimador.taxa_hz` = 30,0 Hz (`knobs.py:451`) |
| Interpolação | por amostra | linear entre o estado do passo anterior e o atual (posição linear; quaternion interpolado e normalizado) | — |
| Ruído de posição | por amostra, por eixo | N(0; (σ)²), σ = `sigma_pos + sigma_pos_rel·dist`, com `dist` = distância câmera–caixa verdadeira | `Estimador.sigma_pos` = (0,005; 0,02) (`knobs.py:471`): 0,005 m → 0 m para `sigma_pos`; 0 → 0,02·dist para `sigma_pos_rel` |
| Viés de posição | por episódio (reset), por eixo | U[−1; 1] · magnitude | `vies_pos` = (0,0; 0,02) m (`knobs.py:460`) |
| Viés de meia aresta | por episódio | U[−1; 1] · magnitude | `vies_meia` = (0,005; 0,02) m (`knobs.py:459`) |
| Viés de orientação | por episódio | eixo uniforme na esfera; ângulo θ = mediana·exp(σ_ln·z), z ~ N(0; 1), σ_ln = ln(p90/mediana)/Φ⁻¹(0,9) | `ori_graus` = ((1,5; 2,5), (5,3; 15,3)) graus, pares (mediana; p90) de início e de refino (`knobs.py:465`) |
| Ruído de orientação por leitura | por passo, por eixo do rotvec | N(0; 1) · `radians(magnitude)` | `ori_leitura_graus` = (0,0; 1,0) graus (`knobs.py:466`) |
| Perda por amostra | por amostra | a amostra é válida se U[0; 1) >= `perda` | `perda` = (0,005; 0,022) (`knobs.py:472`) |
| Rajada de perda | por passo | um env fora de rajada abre uma se U[0; 1) < `rajada_taxa·dt`; duração ~ U[`rajada_s`]; amostra com instante antes do fim da rajada é inválida | `rajada_taxa` = (0,0; 0,035) 1/s (`knobs.py:469`); `rajada_s` = (0,3; 1,0) s (`knobs.py:468`) |
| Visibilidade | por amostra | amostra inválida se a caixa não é visível (seção 12.3). A visibilidade é calculada uma vez por passo da política, no estado do passo, e vale para todas as amostras da grade geradas nesse passo (`observacoes.py:187-192`; `estimador.py:331`) | — |
| Atraso | a amostra gerada em g é entregue quando `g <= t − latencia_s` | determinístico | `Estimador.latencia_s` = 0,033 s (`knobs.py:452`) |

Detalhes:
- A fila tem `ceil(latencia_s·taxa_hz) + 3` posições (`estimador.py:217-218`); com os valores atuais são 4.
- O registro guarda a leitura entregue mais recente (`estimador.py:343-352`).
- A idade da leitura é `t − instante`, cortada em `idade_max` = 2,0 s (`knobs.py:453`). Sem leitura, a idade é `idade_max`. As 7 primeiras saídas ficam zero se nunca houve leitura (`estimador.py:378-387`).
- Caixa presa às palmas: sem leitura nova, se a última leitura tinha `norm(caixa − meio das palmas) < meia_leitura + presa_margem`, a pose de saída anda com o meio das palmas e gira em yaw com o eixo das palmas (`estimador.py:353-371`). `Estimador.presa_margem` = 0,05 m (`knobs.py:454`). Solta, a saída fica na última leitura.
- Saída `saida_b` (`estimador.py:378-387`): posição no frame da base estimada (pela odometria), rotvec do quaternion relativo à base reduzido pela simetria do cubo (menor ângulo entre as 24 rotações, `estimador.py:59-91`), meia aresta lida e idade. Total 8 números.
- Os números `d435_pos` (`knobs.py:477`) e `d435_pitch` = 0,8307767 rad (`knobs.py:478`) posicionam o site da câmera (`cena.py:164-166`).

### 12.5 Odometria

`Odometria.passo` (`estimador.py:158-184`) roda uma vez por passo, no fim do `compute` da cadeia (`comando.py:623-625`). O incremento verdadeiro vai ao frame da base do passo anterior. Com gerador, soma ruído gaussiano N(0; 1) por canal (4 canais): `s_xyz = sqrt(deriva_odo²·norma + deriva_pos_giro²·abs(Δψ))` em x, y, z e `s_psi = sqrt(deriva_rumo²·norma + deriva_rumo_giro²·abs(Δψ))` no rumo. As magnitudes vêm de `deriva_do_degrau` (`comando.py:247-250`): `Estimador.deriva_odo` = (0,002; 0,0075) m/√m (`knobs.py:473`); `deriva_rumo` = (0,002; 0,005) rad/√m (`knobs.py:474`); `deriva_pos_giro` = (0,004; 0,008) m/√rad (`knobs.py:475`); `deriva_rumo_giro` = (0,0035; 0,014) rad/√rad (`knobs.py:476`). Roll e pitch vêm do IMU verdadeiro. No reset a pose estimada é a verdadeira (`estimador.py:148-156`).

## 13. Registro de sucesso das células PEGAR, BOTAR e CARREGAR

`_registra` (`comando.py:521-540`) manda o resultado à célula do currículo:

- PEGAR: registra quando `self.pega` está ligado (o P0 abriu). Dois momentos: na abertura do BOTAR (`falhou` = falso; `comando.py:589`) e no reset (`falhou = ~time_out`; `comando.py:643-644`).
- BOTAR: registra quando `self.bota` está ligado (o B1 abriu). Dois momentos: no fim da cadeia (`falhou` = falso; `comando.py:615-616`) e no reset (`comando.py:645`).
- Sucesso do PEGAR: `v3_pega_fechada & ~falhou & ~v3_perdeu & (max(z[0:4]) == 0)`. Sucesso do BOTAR: `v3_bota_fechada & ~falhou & ~v3_perdeu & (max(z[4:7]) == 0)` (`comando.py:68-70,91-93,534,537`). Enquanto alguma zona da tarefa não chegou a 0 exato, o registro conta como falha.
- `time_out` não é queda: `falhou` fica falso (`comando.py:643`).
- Depois do registro `self.pega` e `self.bota` desligam.
- `Curriculo.registra` soma `n` registros e `soma` de sucessos por célula (`balanceador.py:148-151`). A cada fecho de iteração, com `n > 0`, a nota é `x = soma/n`; a nota `s` da célula vira `x` na primeira medida e `s + alfa·(x − s)` nas seguintes, com `Balanceador.alfa` = 0,05 (`knobs.py:394`; `balanceador.py:159-168`). A célula passa no portão quando `s >= Balanceador.portao` = 0,75 (`knobs.py:391`; `balanceador.py:122-123`).

CARREGAR: o registro é do segmento de comando do twist. `sucesso_do_segmento` (`comando.py:227-242`) e `_fecha` (`comando.py:759-808`) julgam o CARREGAR assim:
- Canal ativo (módulo do comando linear, ou módulo do giro, maior ou igual a `Comando.limiar_canal` = 0,02, `knobs.py:138`): a razão `soma(v·ĉ)dt / soma(norma(c))dt` tem de ser maior ou igual a `Comando.razao_portao` = 0,50 (`knobs.py:146`).
- Canal sem comando: a deriva da âncora tem de ser menor que `deriva_pos_m` no linear (de `deriva_pos_inicio` = 0,25 m a `deriva_pos_m` = 0,10 m pela restrição, `knobs.py:142-143`) e que `deriva_rumo_rad` = 0,25 rad no giro (`knobs.py:144`).
- A caixa não pode ter sido perdida; no terreno, a inclinação máxima da caixa no segmento é no máximo `incl_max_terreno_graus` = 15,0 (`knobs.py:149`).
- O segmento só registra se fecha depois da âncora, isto é, depois de `ancora_s` = 1,5 s (`knobs.py:145`) desde a abertura ou desde o último empurrão (`comando.py:221-224,241`). Uma terminação que não é `time_out` registra mesmo antes da âncora, com nota 0 (`comando.py:789-790`).
- O PEGAR, o BOTAR e o CARREGAR da fatia C têm `v3_celula` = −1 e não registram por este caminho (`comando.py:901`).

## 14. O que lê a cadeia

### 14.1 Terminações (`env_cfg.py:137-147`)

| Termo | Tarefas | Condição exata | Knob |
|---|---|---|---|
| `time_out` (`fim_da_cadeia`) | todas | `episode_length_buf >= max_episode_length`, `v3_fim` ou, no ANDAR em terreno, a travessia: o `pose_ult` do `twist` a mais de `Terreno.dist_travessia` = 3,5 m da origem da célula (`terminacoes.py:53-72`) | `Cadeia.episodio_s` = 20,0 s (`knobs.py:160`) |
| `queda` | todas | ANDAR e CARREGAR: `incl > incl_marcha` ou `pelve − chão < base_min`. PEGAR e BOTAR: `incl > radians(incl_manip_graus)` ou `joelho mais baixo − chão < joelho_min` (`terminacoes.py:35-50`). A tarefa é a de `v3_tarefa`; depois da perda a tarefa é ANDAR. | `incl_marcha` = 0,8 rad (`knobs.py:375`); `base_min` = 0,20 m (`knobs.py:376`); `incl_manip_graus` = 70,0 (`knobs.py:377`); `joelho_min` = 0,05 m (`knobs.py:378`) |
| `caixa_perdida` | PEGAR, CARREGAR, BOTAR | `v3_perdeu_agora` e a chave `caixa_perdida_vira_andar` desligada (`terminacoes.py:75-83`) | `knobs.py:161` |
| `chegada` | BOTAR | primeiro contato do sensor `apoio_caixa` e `−v3_vz_caixa_ant > limite do degrau` (`terminacoes.py:128-151`) | `Terminacao.chegada` = (1,0; 0,25) m/s (`knobs.py:381`) |
| `batente` | todas | alguma das 29 juntas com `curso > batente_frac`, ou `abs(hip_yaw) > hip_yaw_max`, com `v3_degrau >= degrau_batente` (`terminacoes.py:154-183`) | `batente_frac` = 1,0 (`knobs.py:383`); `hip_yaw_max` = 1,0 rad (`knobs.py:382`); `Refino.degrau_batente` = 1 (`knobs.py:547`) |

Os termos mascaram por `v3_tarefa` com `torch.where` (`curriculo.py:93-100`).

### 14.2 Recompensas que leem buffers da cadeia

| Termo | Tarefas (`env_cfg.py:42-54`) | Buffer da cadeia lido | Forma e knobs |
|---|---|---|---|
| `progresso` | todas | `v3_alvo_idx`, `v3_sigma`, `v3_pega_fechada`, `v3_bota_fechada`, `v3_perdeu`, `v3_h`, `v3_phi_melhor`, `v3_caixa_pega_w`, `v3_ponto_apoio`, `v3_topo`, `v3_topo_destino`, `v3_tarefa`, `v3_celula`, `v3_alvo_w`, `v3_meia`, `v3_massa`, `v3_alvos` (na qualidade) | Seção 9.4. Peso 1 (`env_cfg.py:57`). |
| `freio` | PEGAR, CARREGAR, BOTAR | `v3_freio`, `v3_tarefa` | Seção 8. |
| `cerca_tronco` | todas | `v3_tarefa`, `v3_alvo_idx` | `relu(incl_torso − (tronco_ref + lim_tronco_graus))`. `tronco_ref` é o tronco do alvo corrente no PEGAR e no BOTAR (0 sem alvo) e 0 nas outras tarefas. `Recompensa.cerca_tronco` = −0,04596 (`knobs.py:272`); `lim_tronco_graus` = 15,0 (`knobs.py:279`). Multiplicada por `restricao` só no ANDAR (`recompensas.py:453-459,750-758`). |
| `cerca_pelve` | todas | `v3_tarefa`, `v3_alvo_idx`, `v3_meia` | `relu(abs(h − ref) − lim_pelve)`. `ref` = `pelve_de_pe` no ANDAR, `pelve_com_caixa(a)` no CARREGAR, pelve do alvo corrente no PEGAR e no BOTAR com alvo; sem alvo vale `pelve_de_pe`. `Recompensa.cerca_pelve` = −4,45279 (`knobs.py:275`); `lim_pelve` = 0,08 m (`knobs.py:283`) (`recompensas.py:775-788`). |
| `cerca_recuo` | PEGAR, CARREGAR, BOTAR | `v3_tarefa`, `v3_alvo_idx`, `v3_topo`, `v3_topo_destino`, `v3_meia` | `relu(abs(recuo − recuo_ref) − lim_recuo)`, valendo 0 no BOTAR com alvo B3 ou sem alvo. `Recompensa.cerca_recuo` = −10,39719 (`knobs.py:276`); `lim_recuo` = 0,08 m (`knobs.py:284`) (`recompensas.py:791-802`). |
| `cerca_com` | todas | `v3_pegou`, `v3_perdeu`, `v3_alvo_idx`, `v3_massa` | `relu(lim_com − margem)` com os dois pés em contato; a caixa entra no CoM quando `_em(alvo, P3, B1, B2)`. `Recompensa.cerca_com` = −22,09035 (`knobs.py:273`); `lim_com` = 0,03 m (`knobs.py:280`) (`recompensas.py:419-435,761-764`). |
| `contato_fora` | todas | `v3_alvo_idx` e `v3_tarefa` | Contatos do robô com o chão fora dos pés (força > `contato_fora_N` = 1,0 N, `knobs.py:300`), mais autocolisões, mais o contato robô–caixa (força > 1,0 N) fora da janela P1, P2, P3, B1, B2 e fora do CARREGAR. `Recompensa.contato_fora` = −1,0 (`knobs.py:299`) (`recompensas.py:805-813`). |
| `contato_laje` | PEGAR, BOTAR | `v3_degrau` | Soma, sobre os 3 sensores da laje (tronco, palma, dorso; `cena.py:49-50`), de `clamp((F − joelho)/(laje_saturacao_N − joelho); 0; 1)`, com `joelho` = `no_degrau(50; 10; v3_degrau; 5)` N (`laje_joelho_N`, `knobs.py:302`) e `laje_saturacao_N` = 100,0 N (`knobs.py:303`). `Recompensa.contato_laje` = −2,0 (`knobs.py:301`) (`recompensas.py:887-896`). |
| `sacudida` | PEGAR, CARREGAR, BOTAR | `v3_alvo_idx` em (P3, B1), `v3_degrau` | `Σ(Δv/(dt·5,04))² + Σ(Δω/(dt·31,56))²` da caixa, vezes `d/5`. `Recompensa.sacudida` = −0,05 (`knobs.py:353`); `sacudida_norma` = (5,04; 31,56) (`knobs.py:358`) (`recompensas.py:339-342,922-945`). |
| `aperto` | PEGAR, CARREGAR, BOTAR | `v3_alvo_idx` em (P3, B1), `v3_degrau` | `min(F_esq; F_dir)/F_ref` das faces ±y da caixa, vezes `d/5`. `Recompensa.aperto` = −0,05 (`knobs.py:355`); `F_ref` = 49,0 N (`knobs.py:359`) (`potencial.py:55-62`; `recompensas.py:948-957`). |
| `perda_caixa` | todas | `v3_perdeu_agora` | Seção 11.2. |
| `vivo` | ANDAR e CARREGAR | `v3_perdeu` | `1 − v3_perdeu` (`recompensas.py:85-87`). |
| `altura_pe`, `passo`, `rastreio_linear`, `rastreio_angular` | ANDAR e CARREGAR | `v3_perdeu`; `v3_alvos`, `v3_celula`, `v3_alvo_idx` na qualidade do rastreio | Zeram depois da perda (`_fim(..., paga=True)`). No CARREGAR o rastreio multiplica só a parte positiva pela qualidade `q = exp(−(d/tol_transporte_p3)²) · rampa(incl; incl_lim; delta_incl_graus) · contatos/2`, com `tol_transporte_p3` efetivo da zona (`recompensas.py:299-308,572-594`; `potencial.py:15-20`). |

### 14.3 Observações

| Termo | Buffers |
|---|---|
| `tarefa` (one-hot de 6) | `v3_tarefa` |
| `caixa` (11) | `v3_caixa_obs` (seção 12) |
| `alvo_macro` (11, one-hot) | `v3_alvo_idx`, `v3_pega_fechada`, `v3_bota_fechada`, `v3_perdeu`. Slots: 0 a 6 = P0 a B3; 7 = pega fechada (índice em P3); 8 = bota fechada; 9 = perda; 10 = nenhum. Prioridade: perda, bota fechada, pega fechada, índice < 0, índice (`observacoes.py:282-295`). |
| `privilegiados` (crítico) | `v3_marcha_on`, `v3_h`, `v3_phi_melhor`, `v3_phi`, `v3_alvo_idx` com `v3_cur.z` (`observacoes.py:298-318`). |
| `referencia` (crítico) | `ref_do_alvo(v3_alvo_idx)` (`observacoes.py:321-324`). |
| `degrau` | `v3_degrau` / 5 (`observacoes.py:338-340`). |

`N_ALVO_MACRO` = `len(ALVOS) + 4` = 11 (`observacoes.py:46`).

### 14.4 PPO

`PPOPorTarefa._grupo` (`algoritmo.py:98-112`) forma o grupo da vantagem pelo argmax das 6 últimas colunas do crítico (a tarefa) e pelo slot de `alvo_macro`; o ANDAR com slot "nenhum" é o grupo 0, e os outros grupos são `1 + tarefa·n + slot`, com `n` = 11 o tamanho do slot `alvo_macro`. A vantagem crua (`retornos − valores`) é normalizada dentro de cada grupo com 2 ou mais amostras: média 0 e desvio 1 (`algoritmo.py:34-60, 157-159`). `V̄_grupo` é a média de `storage.values` do grupo no rollout anterior; um grupo ausente mantém o valor anterior (`algoritmo.py:160-169`). Se `fatia_alvo` e `v_grupo` existem, o passo de recompensa soma `γ · min(V̄_grupo, 0)` à recompensa quando há terminação que não é `time_out` ou `v3_perdeu_agora` (`algoritmo.py:127-131`).

### 14.5 Métricas

`phi_<alvo>`, `tempo_<alvo>`, `erro_p3`, `deslocamento_da_caixa`, `velocidade_de_chegada`, `deslocamento_depois_do_apoio`, `aceleracao_da_caixa`, `idade_das_leituras` e `fracao_avisos` leem `v3_alvo_idx`, `v3_phi`, `v3_v_chegada`, `v3_ponto_apoio`, `v3_perdeu`, `v3_aviso` e o estimador. Elas só relatam; nada na cadeia depende delas. `_com_caixa` vale para PEGAR, CARREGAR e BOTAR (`metricas.py:239-243`).

### 14.6 Eventos que leem ou escrevem os buffers da cadeia

| Evento | Momento | Buffers | O que faz |
|---|---|---|---|
| `fatia` (`sorteia_fatia`) | reset | escreve `v3_fatia`, `v3_t_empurrao` | Antes de sortear, registra a visita e a travessia do terreno do ANDAR do episódio que acabou (`vis_ter`, `trav_ter`). Sorteia fatia e episódio de terreno (seção 2.4). Na fatia ANDAR, a coluna sai por pesos sobre o atraso da linha de cada coluna, e a linha, por arredondamento estocástico de `linha[coluna]`. `eventos.py:133-175` |
| `cena` (`posiciona_cena`) | reset | escreve `v3_topo`, `v3_massa`, `v3_caixa_xy`, `v3_rumo_cena`; lê `v3_fatia` | Seção 4.2. `eventos.py:332-367` |
| `base` (`reset_base_por_fatia`) | reset | lê `v3_fatia`, `v3_topo`, `v3_meia`, `v3_caixa_xy`, `v3_rumo_cena` | Seção 4.3. `eventos.py:211-225` |
| `carga` | reset | lê `v3_massa`, `v3_meia` | Escreve massa e inércia `(2/3)·m·a²` na física. `eventos.py:443-461` |
| `filtros` (`reinicia_filtros`) | reset | `v3_odo`, `v3_estimador` | Seção 12. `eventos.py:228-241` |
| `forca_caixa` | a cada passo, depois do comando | lê `v3_tarefa`, `v3_pegou`, `v3_perdeu` | Impulso na caixa: força por eixo ~ U[−11,5; 11,5] N (`DR.forca_caixa_N`, `knobs.py:531`), mantida por U[0,1; 0,3] s (`forca_caixa_dur_s`, `knobs.py:532`), com espera entre impulsos ~ U[3,0; 8,0] s (`forca_caixa_periodo_s`, `knobs.py:533`; `mjlab/envs/mdp/events.py:386-402`). Só age depois que `dr()["forca_caixa"]` liga, isto é, quando a soma dos níveis do PEGAR e do CARREGAR passa de 0 (`balanceador.py:202,383`). Depois do impulso o evento zera a força nos envs fora de (tarefa CARREGAR e `v3_pegou` e não `v3_perdeu`). `eventos.py:584-602`; `env_cfg.py:207-209` |
| `laje_fora` e `posiciona_destino` | chamados pela cadeia, não são eventos do cfg | `v3_topo_destino` | Seções 5.3 e 6.4. `eventos.py:322-329,393-413` |

## 15. Knobs

Todos os valores são os de `knobs.py`.

### 15.1 `Cadeia` (`knobs.py:153-161`)

| Knob | Valor | Efeito |
|---|---|---|
| `espera_s` | (0,5; 1,0) s | Faixa do ANDAR parado antes do PEGAR, nas fatias B e C (`comando.py:174`). |
| `carregar_b_s` | (4,0; 7,0) s | Duração do CARREGAR da fatia B (`comando.py:177`). |
| `carregar_c_s` | (0,5; 1,0) s | Duração do CARREGAR da fatia C (`comando.py:178`). |
| `andar_final_s` | 1,5 s | Duração do ANDAR final da fatia C, faixa (1,5; 1,5) (`comando.py:174,179`). |
| `atraso_pegar_s` | (0,0; 1,0) s | Atraso, depois do aviso do P3, até o PEGAR acabar (`comando.py:177-178`). |
| `atraso_botar_s` | (0,0; 1,5) s | Atraso, depois do aviso do B3, até o BOTAR acabar (`comando.py:179`). |
| `episodio_s` | 20,0 s | Duração do trecho da fatia ANDAR (`comando.py:176`) e do episódio (`env_cfg.py:105`). |
| `caixa_perdida_vira_andar` | verdadeiro | Liga o congelamento da perda e a tarefa ANDAR (`comando.py:412`); desliga a terminação `caixa_perdida` quando verdadeiro (`env_cfg.py:142`). |

### 15.2 `Alvos` (`knobs.py:174-231`)

| Knob | Valor | Efeito |
|---|---|---|
| `p0_frente` | 0,05 m | x do alvo das palmas no P0: −(a + p0_frente). |
| `palma_face` | 0,045 m | Distância do sítio da palma à face, almofada encostada. Entra no alvo das palmas e no `afasta` do B2. |
| `folga_p1` | 0,02 m | Folga da almofada à face no P0 e no P1. |
| `tempo_p2` | 0,3 s | Permanência do P2 (final; zona 0,2). |
| `tol_palma` | 0,03 m | Tolerância das palmas no P0 e no P1 (final; zona 0,05). |
| `subida_min_p3` | 0,03 m | Subida mínima da caixa no P3 (zona 0,01). |
| `tol_transporte_p3` | 0,05 m | Distância ao ponto de transporte no P3 e escala da qualidade do CARREGAR (zona 0,20). |
| `tol_pelve_p3` | 0,03 m | Distância da pelve no P3 (zona 0,10). |
| `tempo_p3` | 0,5 s | Permanência do P3 (zona 0,2). |
| `raio_b1` | 0,10 m | Raio ao ponto de soltura no fecho do B1 e `fora_do_alvo` da perda (zona 0,20 no fecho). |
| `apoio_b1_mg` | 0,8 | Fração do peso que a laje carrega no fecho do B1. Entra no `deficit` do d do B1. Sem zona. |
| `escala_apoio_b1` | 0,10 m | Converte o déficit de apoio em distância no d do B1. |
| `tempo_b1` | 0,3 s | Permanência do B1 (zona 0,1). |
| `v_parada` | 0,05 m/s | Caixa parada no B1 e no B2. Sem zona. |
| `b2_afasta` | 0,05 m | Afastamento das palmas no B2 (zona 0,02) e no alvo das palmas do B2 usado no d. |
| `tempo_b2` | 0,5 s | Permanência do B2 (zona 0,2). |
| `tol_pelve_b3` | 0,03 m | Pelve no B3 (zona 0,06). Também escala do d do B3. |
| `tol_tronco_b3_graus` | 5,0 graus | Tronco no B3 (zona 10,0). Também escala do d do B3. |
| `tol_palmas_b3` | 0,05 m | Palmas ao lado do tronco no B3 (zona 0,10). Também escala do d do B3. |
| `tempo_b3` | 0,5 s | Permanência do B3 (zona 0,2). |
| `z_transporte` | 0,90 m | z do ponto de transporte da caixa sobre o chão local. |
| `incl_inicio_graus` | 30,0 | Limite de inclinação da caixa no degrau 0. |
| `incl_fim_graus` | 15,0 | Limite de inclinação da caixa no degrau 5. |
| `delta_incl_graus` | 15,0 | Largura da rampa de inclinação. |
| `sigma_min_etapa` | (0,05; 0,05; 0,08; 0,20; 0,20; 0,05; 0,0) | Piso de σ por etapa. |
| `zona` | tabela da seção 7.1 | Valor largo de cada campo. |
| `largura_tanh` | 2,0 | Largura do núcleo tanh do Φ e das conquistas. |
| `desloc_p0p2_frac` | 1,0 | Deslocamento máximo da caixa em P0 a P2, vezes `a`. |
| `delta_p0p2_frac` | 0,5 | Largura da rampa do deslocamento em P0 a P2, vezes `a`. |
| `desloc_b2b4` | 0,02 m | Deslocamento máximo desde o ponto de apoio em B2 e B3. |
| `delta_b2b4` | 0,06 m | Largura da rampa desse deslocamento. |
| `faixa_tronco_graus` | 15,0 | Faixa do portão `tronco` do Φ. |
| `faixa_recuo` | 0,08 m | Faixa do portão `recuo` do Φ e jitter do recuo no reset da base. |
| `faixa_com` | 0,03 m | Faixa do portão `com` do Φ. |
| `delta_tronco_graus` | 15,0 | Largura da rampa do `tronco`. |
| `delta_recuo` | 0,30 m | Largura da rampa do `recuo`. |
| `delta_com` | 0,03 m | Largura da rampa do `com`. |

### 15.3 `Detector` (`knobs.py:164-170`)

| Knob | Valor | Efeito no treino |
|---|---|---|
| `tol_pelve` | 0,03 m | Define `Alvos.tol_pelve_b3` e o valor largo (dobro) da zona do B3 (`knobs.py:198,218`). |
| `tol_tronco_graus` | 5,0 graus | Define `Alvos.tol_tronco_b3_graus` e o valor largo (`knobs.py:199,218`). |
| `tol_palma` | 0,05 m | Define `Alvos.tol_palmas_b3` e o valor largo (`knobs.py:200,219`). |
| `estavel_s` | 0,5 s | Só é lido por `alvos.detecta_fim` (`alvos.py:228`). |
| `juntas_paradas` | 0,2 | Só é lido por `alvos.detecta_fim` (`alvos.py:225`). |

`alvos.detecta_fim` (`alvos.py:213-228`) calcula `pelve_ok & tronco_ok & palmas_ok & parado` sustentados por `estavel_s`. Nenhum módulo do pacote lido o chama.

### 15.4 `Niveis` (`knobs.py:416-424`)

| Knob | Valor | Efeito |
|---|---|---|
| `pegar` | ((0,55; 1), (0,45; 2), (0,30; 3), (0,15; 4), (0,04; 5)) | Por nível do PEGAR: (topo h_k em m; massa m_k em kg). Define o número de níveis do PEGAR (5) e as faixas de topo e massa (seção 4.2). |
| `botar` | (0,55; 0,45; 0,30; 0,15; 0,04) m | Topo por nível do BOTAR (seção 6.4). Define o número de níveis do BOTAR. |
| `topo_min_c` | 0,30 m | Piso do topo sorteado na fatia C. |
| `subnivel_extra` | (0,10; 0,0) m | Soma ao topo do PEGAR e fixa o topo do BOTAR no subnível 0. |
| `subnivel_longe` | (0,10; 0,20) m | Teto do `longe` do destino no subnível 0 (índice 0). |
| `subnivel_jitter` | (0,0; 1,0) | Fração da faixa de recuo na base da pega por subnível. |

O nível sobe depois de `Balanceador.iters_nivel` = 50 iterações seguidas com a média de `s` das células abertas no portão (`knobs.py:395`; `balanceador.py:250-278`). No nível 0, a primeira subida avança o subnível; as seguintes avançam o nível.

### 15.5 `Estimador` (`knobs.py:449-478`)

Valores, efeito e uso estão na seção 12. Os pares `(início, refino)` seguem o degrau global (seção 10). Não entram nos pares: `taxa_hz`, `latencia_s`, `idade_max`, `presa_margem`, `fov_h_graus`, `fov_v_graus`, `alcance`, `pontos_visiveis_min`, `rajada_s`, `d435_pos`, `d435_pitch`.

## 16. Divergências entre comentário e código

- ⚠ Divergência: o comentário diz que o valor `maos` de `portoes_do_alvo` vem "de quem chama" e que o `_phi_env` passa a firmeza da pega (`recompensas.py:254-255`); o código passa `torch.ones_like(maos)` (`recompensas.py:566`). O portão `maos` do Φ vale sempre 1.
- ⚠ Divergência: o comentário diz que um alvo sem tempo, "P2, tempo 0", avança no primeiro passo (`alvos.py:200`); o código dá a P2 o `tempo_p2` efetivo (`comando.py:442`). Só P0 e P1 têm tempo 0.
- ⚠ Divergência: o comentário de `_dist` diz que o alvo da caixa é o da recompensa, "uma conta só" (`comando.py:349-351`); o código usa `dpelve = h − pelve_de_pe` (`comando.py:353`) e o Φ usa `h − ref_do_alvo[:, 6]` (`recompensas.py:536`). No P3 as duas referências diferem (`pelve_de_pe` contra `pelve_com_caixa(a)`).
- ⚠ Divergência: o comentário de `eventos.py:10-11` diz que `garante_v3` é chamado em `comando.py:199`; a chamada está em `comando.py:293`.
- ⚠ Divergência: o comentário de `potencial.fator_pouso` diz que o fator multiplica o Φ "no B1 e no B2" (`potencial.py:48`); o código aplica o fator só no B1 (`_POUSO = _ix("B1")`, `recompensas.py:57,551`). O comentário de `por_degrau` cita "o fator do B2 no progresso" (`recompensas.py:448-449`); esse fator não existe no código lido.
- ⚠ Divergência: os comentários de `Cadeia.atraso_pegar_s` e `atraso_botar_s` dizem "atraso antes do PEGAR" e "antes do BOTAR" (`knobs.py:158-159`); o código usa os dois como atraso depois do aviso, isto é, depois do fecho do P3 e do B3 (`comando.py:177-179,606-609`). O comentário de `espera_s` diz "espera entre trechos" (`knobs.py:154`); o código usa a faixa só no primeiro trecho ANDAR parado das fatias B e C (`comando.py:174,177-178`).

## 17. Não determinado pelo código lido

- Não determinado pelo código lido: quem lê `v3_avancou`. Nenhum arquivo lido o consome.
- Não determinado pelo código lido: quem chama `alvos.detecta_fim`. O comentário cita `pilota_v3.py`, que não está entre as fontes.
- Não determinado pelo código lido: a definição das geoms `PALM_PAD_GEOMS`, `BOX_GEOM` e `TABLE_GEOM` e o grupo de colisão da laje; `cena.py` foi lido só nos sensores (linhas 320 a 425).
- `max_episode_length` = `ceil(episode_length_s / step_dt)` (`mjlab/envs/manager_based_rl_env.py:281-283`) com `episode_length_s` = `Cadeia.episodio_s` = 20,0 (`env_cfg.py:105`) e `step_dt` = `timestep · decimation` = 0,005 · 4 (`manager_based_rl_env.py:266-268`). Em ponto flutuante de dupla precisão, 20,0 / 0,02 = 1000,0 e o teto vale 1000 passos. O relógio do trecho do ANDAR soma `dt` em float32 e chega a 20,0 no compute do 1000º passo do episódio (o primeiro compute é o do passo do reset); `v3_fim` e `episode_length_buf >= 1000` valem então na mesma checagem de terminações. Não determinado pelo código lido: o valor exato da soma float32 dentro do torch, que foi estimado fora do código com a mesma sequência de somas.
- Não determinado pelo código lido: quem lê `grade_aceita` de `referencias.npz`. O script a grava (`gera_referencias.py:150`) e `alvos.Referencias` não a carrega.
