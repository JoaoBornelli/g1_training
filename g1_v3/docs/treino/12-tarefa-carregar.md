# Tarefa CARREGAR

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:
- `g1_v3/`: `env_cfg.py`, `tarefa.py`, `knobs.py`, `comando.py`, `eventos.py`, `balanceador.py`, `curriculo.py`, `recompensas.py`, `potencial.py`, `alvos.py`, `terminacoes.py`, `observacoes.py`, `cena.py`, `metricas.py`, `estimador.py` (linhas 100–388), `algoritmo.py`, `runner.py`.
- `mjlab/`: `envs/manager_based_rl_env.py`, `managers/event_manager.py`, `managers/command_manager.py`, `managers/reward_manager.py`, `managers/metrics_manager.py`, `managers/observation_manager.py`, `managers/curriculum_manager.py`, `managers/termination_manager.py`, `envs/mdp/events.py`, `envs/mdp/dr/actuator.py`, `envs/mdp/dr/joint.py`, `envs/mdp/dr/geom.py`, `envs/mdp/dr/body.py`, `tasks/velocity/mdp/velocity_command.py`, `tasks/velocity/config/g1/rl_cfg.py`, `utils/buffers/delay_buffer.py`, `entity/entity.py`, `terrains/terrain_generator.py`, `terrains/config.py`, `terrains/primitive_terrains.py`, `terrains/heightfield_terrains.py`.
- `rsl_rl/`: `algorithms/ppo.py`.

Convenções deste documento:
- Os caminhos de `g1_v3` são relativos a `g1_v3/`. Os de biblioteca começam por `mjlab/`.
- O passo da política dura `step_dt` = `Contrato.dt_fisica` (0,005 s, `knobs.py:54`) × `Contrato.decimacao` (4, `knobs.py:55`) = 0,02 s (`env_cfg.py:103-104`).
- O gerenciador de recompensa multiplica cada termo por `peso × step_dt` (`mjlab/managers/reward_manager.py:118,127`). Os pesos abaixo são os crus.
- O termo `v3_*` é um buffer por env que o código da v3 grava no env. Exemplo: `v3_tarefa` guarda a tarefa corrente de cada env (`curriculo.py:24`).

---

## 1. O que é a tarefa e quando um env está nela

### 1.1 Definição

CARREGAR é um valor do vetor `v3_tarefa` do env. O índice dele é 2 na lista `TAREFAS` (`knobs.py:42`, `comando.py:30`). Nessa tarefa o robô anda, ou fica parado, com a caixa já presa entre as palmas.

Um env está em CARREGAR quando `env.v3_tarefa == 2`. Toda recompensa e toda terminação da tarefa usam essa condição pela máscara `mascara` (`curriculo.py:93-100`).

### 1.2 Fatia e trecho

A fatia é o grupo de tarefas que um env recebe no reset do episódio. Há três fatias: `ANDAR`, `B` e `C` (`knobs.py:43`). O trecho é um segmento da cadeia de tarefas de uma fatia. Cada trecho publica uma tarefa (`comando.py:163-194`).

| Fatia | Trechos, em ordem (tarefa, como acaba, faixa em s) |
|---|---|
| `ANDAR` | ANDAR, por tempo, `Cadeia.episodio_s` = 20,0 (`knobs.py:160`) |
| `B` | ANDAR (espera), por tempo, `Cadeia.espera_s` = (0,5; 1,0) → PEGAR, por aviso, atraso `Cadeia.atraso_pegar_s` = (0,0; 1,0) → CARREGAR, por tempo, `Cadeia.carregar_b_s` = (4,0; 7,0) |
| `C` | ANDAR (espera), `espera_s` → PEGAR, por aviso, `atraso_pegar_s` → CARREGAR, por tempo, `Cadeia.carregar_c_s` = (0,5; 1,0) → BOTAR, por aviso, `Cadeia.atraso_botar_s` = (0,0; 1,5) → ANDAR final, por tempo, `Cadeia.andar_final_s` = 1,5 |

Referências: `comando.py:174-180`, `knobs.py:154-159`.

Só as fatias `B` e `C` têm trecho CARREGAR. A fatia `ANDAR` nunca entra em CARREGAR (`comando.py:176`).

### 1.3 Sorteio da fatia

Quando: no reset de cada env, no primeiro evento de reset, `fatia` (`env_cfg.py:154`, `eventos.py:133-163`).

Como: `torch.multinomial` com reposição sobre as três frações de `Curriculo.fracoes_fatias()` (`balanceador.py:302-304`).

De que as frações dependem (`balanceador.py:282-293`):
- Uma fatia aberta recebe a mesma parte das outras abertas. Uma fatia fechada recebe 0.
- A fatia `ANDAR` abre no começo (`balanceador.py:110`). O teto dela vale 1,0 (`balanceador.py:288`).
- Cada fatia nova tem um teto: `Balanceador.rampa_fatia_inicio` = 0,05 (`knobs.py:397`) + `Balanceador.rampa_fatia_passo` = 0,02 (`knobs.py:398`) × `_rampa[f]`.
- `_rampa[f]` sobe 1 quando `d > 0` e `d` é múltiplo de `Balanceador.rampa_fatia_its` = 12 (`knobs.py:399`). `d` é a diferença entre a iteração corrente do balanceador e a iteração em que a fatia abriu (`balanceador.py:186-193`). A rampa nunca desce.
- O código ordena as fatias abertas pelo teto, do menor ao maior. Cada uma recebe `min(teto, resto / quantas faltam)` (`balanceador.py:290-292`). O que o teto corta volta, em partes iguais, às fatias de teto maior.

Quando a fatia `B` abre: o nível do ANDAR chega a `Balanceador.nivel_abre_b` = 5 (`knobs.py:396`, `balanceador.py:209-211`). A abertura da `B` abre as células `CARREGAR/reto` e `PEGAR/pegar` juntas.

Quando a fatia `C` abre: a célula `PEGAR/pegar` passa no portão, e a célula `BOTAR/botar` abre (`balanceador.py:231-232`).

Não determinado pelo código lido: a fração dos passos de treino que ficam em CARREGAR. Ela depende do tempo que o PEGAR leva para fechar o P3 em cada episódio.

### 1.4 Sorteio da tarefa

A tarefa do env não é sorteada. Ela sai da tabela de trechos da fatia (§1.2). O reset abre o primeiro trecho da fatia (`comando.py:659-660`). O env entra em CARREGAR quando o trecho PEGAR acaba.

Condição de fim do trecho PEGAR (`comando.py:201-216`, `:604-613`):
1. O latch `v3_aviso` fica verdadeiro. O latch sobe quando `v3_pega_fechada` é verdadeiro no trecho por aviso (`comando.py:606-607`).
2. O relógio do trecho acumula `step_dt` só enquanto `v3_aviso` é verdadeiro.
3. O trecho avança quando `relogio >= alvo_s − 1e-6` e `v3_aviso` é verdadeiro. `alvo_s` é o atraso sorteado na abertura do trecho: uniforme em `atraso_pegar_s` = (0,0; 1,0).

`v3_pega_fechada` fica verdadeiro quando o alvo P3 fecha (`comando.py:492-496`). O critério de fecho do P3 está na §4.

Na abertura do trecho CARREGAR (`comando.py:568-594`):
- O código sorteia a duração `alvo_s` uniforme na faixa do trecho (`comando.py:578-579`).
- `v3_trecho` recebe o trecho novo, `v3_aviso` volta a falso, `v3_h` volta a 1,0 e `v3_phi_melhor` volta a 0,0 (`comando.py:581-583`).
- `_publica` grava `v3_tarefa` = CARREGAR, grava `v3_freio` = nível do CARREGAR + 1 e `v3_degrau` = degrau do CARREGAR, e reamostra o comando (`comando.py:508-519`). Esses dois valores ficam fixos até o próximo trecho, até a perda da caixa (`_perda` chama `_publica` com a tarefa ANDAR, `comando.py:420`) ou até o reset.
- O índice do alvo `v3_alvo_idx` continua em P3. O trecho CARREGAR não o muda (`comando.py:585-587`).
- Se a fatia é `B` e o comando sorteado não é zero, `laje_fora` move a laje para `Cena.estaciona_z` = 20,0 m acima da origem do env (`comando.py:592-594`, `eventos.py:322-329`, `knobs.py:114`). A caixa não é movida.

### 1.5 Quando o env sai de CARREGAR

| Saída | Condição | O que acontece depois |
|---|---|---|
| Fim da cadeia (fatia `B`) | O relógio do trecho passa de `alvo_s` e não há próximo trecho (`comando.py:213-215`) | `v3_fim` vira verdadeiro (`comando.py:614`). A terminação `time_out` dispara no passo seguinte (`terminacoes.py:55`, `:72`). O env reinicia |
| Troca para BOTAR (fatia `C`) | O relógio passa de `alvo_s` e há próximo trecho (`comando.py:214`) | `_abre_trecho` publica BOTAR (`comando.py:619`) |
| Caixa perdida | `perda` verdadeira em `_perda` e `Cadeia.caixa_perdida_vira_andar` = True (`knobs.py:161`, `comando.py:412-420`) | `v3_tarefa` vira ANDAR até o fim do episódio. A §8.3 detalha |
| Terminação | `queda` ou `batente` (§8) | O env reinicia no mesmo passo |
| Tempo | `episode_length_buf >= max_episode_length`, ou seja 1000 passos = `ceil(20,0 / 0,02)` com `Cadeia.episodio_s` = 20,0 (`terminacoes.py:53-55`, `mjlab/envs/manager_based_rl_env.py:283`) | `time_out`. O env reinicia |

---

## 2. Reset

### 2.1 Ordem geral do reset de um env

O reset de um env roda na ordem abaixo (`mjlab/envs/manager_based_rl_env.py:553-590`):

1. Termo de currículo `v3` (`curriculo.py:103-143`). Ele fecha a iteração do balanceador se a iteração mudou (§5.1).
2. `sim.reset` e `scene.reset`.
3. Eventos de modo `reset`, na ordem do dicionário `eventos_` de `env_cfg.py:153-211`: `fatia`, `cena`, `base`, `juntas`, `carga`, `atrito_pad`, `atrito_laje`, `vies_imu`, `pd`, `forca_motor`, `solref`, `filtros`. O evento `com_caixa` não entra no cfg porque `DR.com_caixa` = False (`knobs.py:527`, `env_cfg.py:182-183`).
4. Reset dos gerenciadores de observação (zera os buffers de atraso e de histórico dos termos do env, `mjlab/managers/observation_manager.py:248-253`), de ação, de recompensa (zera a `v_ant` do `sacudida`, `mjlab/managers/reward_manager.py:112-113`), de métrica e de currículo (`mjlab/envs/manager_based_rl_env.py:566-578`).
5. Reset dos termos de comando, nesta ordem: `cadeia`, depois `twist` (`env_cfg.py:118-128`, `mjlab/managers/command_manager.py:241-247`, `mjlab/envs/manager_based_rl_env.py:581`).
6. Reset do gerenciador de eventos: o `forca_caixa`, que é termo-classe, zera o impulso ativo e sorteia de novo a pausa (`mjlab/envs/mdp/events.py:584-606`); o `empurrao`, que é função, não reinicia (§6.1). Depois, reset das terminações, e `episode_length_buf` volta a 0 (`mjlab/envs/manager_based_rl_env.py:584-590`).

O reset acontece no mesmo passo em que a terminação dispara. Depois do reset o env publica as observações do estado novo.

`pd`, `forca_motor` e `solref` só entram no cfg com `play=False` (`env_cfg.py:169-181`).

### 2.2 Eventos de startup (uma vez, na criação do env)

Estes sorteios acontecem uma vez por env e valem pela vida inteira dele (`mjlab/envs/manager_based_rl_env.py:354`).

| Evento | O que sorteia | Distribuição e faixa | Referência |
|---|---|---|---|
| `atrito_pe` | Atrito do pé, igual em todas as geoms do pé do mesmo env | Uniforme em `DR.atrito_pe` = (0,3; 1,0), operação `abs`, `shared_random=True`; só o eixo 0 (atrito tangencial) | `env_cfg.py:185-187`, `knobs.py:508`, `mjlab/envs/mdp/dr/geom.py:118-147` |
| `massa_tronco` | Massa somada ao `torso_link`. A inércia do corpo não muda | Uniforme em `DR.massa_tronco` = (−1,0; 3,0) kg, operação `add` | `env_cfg.py:188-190`, `knobs.py:509`, `mjlab/envs/mdp/dr/body.py:285-325` |
| `vies_encoder` | Viés de encoder por junta | Uniforme em ± `DR.vies_encoder` = 0,015 rad | `env_cfg.py:191-192`, `knobs.py:510`, `mjlab/envs/mdp/dr/joint.py:209-240` |
| `tamanho_caixa` | Meia aresta da caixa, gravada em `v3_meia` | Escolha uniforme entre `Cena.caixa_n_variantes` = 8 valores igualmente espaçados em `Cena.caixa_meia_faixa` = (0,07; 0,13) m (`torch.linspace`) | `eventos.py:417-440`, `env_cfg.py:193-195`, `knobs.py:78-79` |
| `atraso` | Teto do atraso dos buffers do atuador e da observação | Teto = `DR.atraso_inicio[1]` = 0 passo (no atuador, × `Contrato.decimacao` = 4 passos de física) | `eventos.py:244-258`, `env_cfg.py:197-198`, `knobs.py:511` |

O tamanho da caixa não muda no reset. Cada env tem uma caixa de tamanho fixo.

### 2.3 Eventos de reset para um env CARREGAR (fatias `B` e `C`)

Um env que vai passar por CARREGAR começa o episódio na fatia `B` ou `C`. O reset o põe no começo do PEGAR, com a caixa na laje à frente do robô.

#### 2.3.1 `fatia` (`eventos.py:133-175`)

O evento começa por registrar a visita e a travessia do episódio de terreno da fatia `ANDAR` que acabou (`eventos.py:143-160`). Esse registro só conta a fatia `ANDAR`: não muda nada para o CARREGAR. Depois vêm os passos abaixo.

1. Sorteia a fatia (§1.3).
2. Zera `v3_t_empurrao` para −∞ (`eventos.py:163`).
3. Sorteia se o episódio é de terreno. Para a fatia `B` e `C`, a probabilidade `p` é a fração da célula `CARREGAR/terreno` no sorteio de células do CARREGAR (`_fracao_terreno`, `eventos.py:36-40`). Vale 0 enquanto a célula está fechada. O sorteio é uniforme em [0,1) contra `p` (`eventos.py:165-167`).
4. Sorteia coluna e linha do terreno (`escolhe_celula_terreno`, `eventos.py:43-74`). A fatia `ANDAR` tem regra própria: a coluna sai pelo atraso da linha de cada coluna, e a linha sai por arredondamento estocástico da linha da coluna (`eventos.py:61-66`). Para o CARREGAR, só a fatia `B` usa terreno, com os sorteios `u[2]` e `u[3]` e o `nivel_terreno("CARREGAR")` (`eventos.py:67-69`, `:168`). A fatia `C` fica na coluna 0, linha 0 (`eventos.py:71`).
   - Coluna da fatia `B`: uniforme entre as 7 colunas de `Terreno.linha_max_carregar` (`knobs.py:439-440`): `escada`, `escada_inv`, `rampa`, `rampa_inv`, `escada_aleatoria`, `obstaculos`, `caixas`. A coluna `irregular` nunca é sorteada.
   - Linha da fatia `B`: uniforme em 0..`min(nivel_terreno("CARREGAR"), teto da coluna)`, inclusive (`eventos.py:69`). Os tetos por coluna: `escada` 9, `escada_inv` 9, `rampa` 9, `rampa_inv` 9, `escada_aleatoria` 6, `obstaculos` 8, `caixas` 7 (`knobs.py:439-440`).
   - Sem episódio de terreno: coluna 0 e linha 0 (chão plano).
5. Grava `terrain_levels`, `terrain_types` e `env_origins` do env na linha e coluna sorteadas (`eventos.py:172-175`).

O terreno vale o episódio inteiro, inclusive os trechos ANDAR (espera) e PEGAR antes do CARREGAR.

Geometria do terreno (`cena.py:251-277`, `knobs.py:428-446`):
- `Terreno.linhas` = 13 linhas e `Terreno.tamanho` = (8,0; 8,0) m por célula (`knobs.py:429-430`). Cada tipo de terreno ocupa uma coluna (`curriculum=True`).
- A dificuldade da linha é `linha / (linhas − 1)`: o `difficulty_range` do gerador fica no padrão (0,0; 1,0) (`mjlab/terrains/terrain_generator.py:122`, `:256-257`). O gerador monta o terreno uma vez, na criação do env. O terreno não muda depois.
- Tetos de `Terreno.tetos_andar` (`knobs.py:432`): degrau 0,20 m, rampa 0,36, obstáculo 0,15 m (`cena.py:261-273`). Cada coluna do CARREGAR usa a dificuldade `d` (0 a 1) assim:

| Coluna | Efeito da dificuldade `d` | Referência |
|---|---|---|
| `escada`, `escada_inv` | Altura do degrau = 0 + `d` × 0,20 m (`degrau` = (0; 0,20)) | `mjlab/terrains/primitive_terrains.py:79-81`, `:256-258` |
| `rampa`, `rampa_inv` | Inclinação (subida/avanço) = 0 + `d` × 0,36. Na `rampa_inv` o sinal é negativo (plataforma embaixo) | `mjlab/terrains/heightfield_terrains.py:238-245` |
| `escada_aleatoria` | Cada degrau tem altura U(0; 0,20) m × (0,5 + 0,5·`d`), sorteada na geração do terreno | `mjlab/terrains/primitive_terrains.py:1061` |
| `obstaculos` | A altura de referência `h` = 0 + `d` × 0,15 m (`obstacle_height_range` = (0; 0,15)). Cada obstáculo escolhe sua altura entre −`h`, −`h`/2, `h`/2 e `h` (`obstacle_height_mode` = "choice": mistura de buracos e degraus) | `mjlab/terrains/heightfield_terrains.py:684-686`, `:725-727`, `:752` |
| `caixas` | Número de caixas = `int(80 × (0,5 + 0,5 × d))`, com `num_boxes` = 80 no preset; altura de cada uma = U(0; 0,15) m × (0,2 + 0,8 × `d`); rotação 0 (`box_yaw_range` = (0; 0)) | `mjlab/terrains/primitive_terrains.py:743`, `:796`, `mjlab/terrains/config.py:170-171`, `cena.py:272-273` |

- Cada coluna do CARREGAR tem plataforma plana `Terreno.plataforma` = 2,0 m (`knobs.py:441`, `cena.py:260-273`).
- `Terreno.borda` = 20,0 m de borda (`knobs.py:446`, `cena.py:275-276`).
- A origem do sub-terreno (a posição onde `fatia` põe `env_origins`) fica no centro da célula em x e y, a 4,0 m de cada borda da célula. A altura dela em z depende da coluna: `escada` = (número de degraus + 1) × altura do degrau (topo da pirâmide); `escada_inv` = o mesmo valor com sinal negativo; `escada_aleatoria` = a altura acumulada do último degrau; `rampa` = altura máxima da superfície; `rampa_inv` = −(altura máxima da superfície); `obstaculos` = 0 (`origin_z_offset` = 0,0); `caixas` = 0 (`mjlab/terrains/primitive_terrains.py:233-235`, `:410-412`, `:832`, `:1159`; `mjlab/terrains/heightfield_terrains.py:352-370`, `:840-841`; `mjlab/terrains/terrain_generator.py:321`).

Não determinado pelo código lido: o número de degraus de cada pirâmide e a altura máxima da superfície da rampa em metros, porque dependem de contas de malha que o trecho lido não fecha.

#### 2.3.2 `cena` (`eventos.py:332-367`)

Roda antes de `base`, porque a base das fatias `B` e `C` se posiciona em relação à caixa (`env_cfg.py:155-156`).

Sorteios por env (todos pelo gerador `v3_gerador`, `eventos.py:77-79`):

| Grandeza | Distribuição e faixa | Referência |
|---|---|---|
| Topo da laje, fatia `B` | Uniforme em [`h_k` + `extra`; `h_0` + `extra`]. `(h_k, m_k)` = `Niveis.pegar[nível do PEGAR]`, com `Niveis.pegar` = ((0,55; 1), (0,45; 2), (0,30; 3), (0,15; 4), (0,04; 5)) (`knobs.py:417`). `h_0` = 0,55 m. `extra` = `Niveis.subnivel_extra[subnível do PEGAR]` com `subnivel_extra` = (0,10; 0,0) (`knobs.py:422`) | `eventos.py:82-109`, `:344-346` |
| Topo da laje, fatia `C` | Uniforme em [`max(h_k; Niveis.topo_min_c)` + `extra`; `h_0` + `extra`]. `Niveis.topo_min_c` = 0,30 m (`knobs.py:419`) | `eventos.py:105-106` |
| Massa da caixa, fatia `B` | Uniforme em [`Cena.caixa_massa_base` = 1,0 kg (`knobs.py:80`); `m_k`] kg, com `m_k` do nível do PEGAR | `eventos.py:107-108` |
| Massa da caixa, fatia `C` | Uniforme em [1,0; `m_j`] kg. `j` é `nivel_passou("PEGAR")`. Com `j` = −1 o teto é 1,0 kg | `eventos.py:102`, `:107-108`, `balanceador.py:350-356` |
| Rumo da cena, só fatia `B` com terreno (coluna > 0) | `k` uniforme em {0, 1, 2, 3}; rumo = `k` × π/2 rad. Sem terreno, rumo 0 | `eventos.py:112-130` |
| Deslocamento da cena, só fatia `B` com terreno | `R(rumo)` · (`Terreno.laje_do_centro` − `Cena.prateleira_xy[0]`; 0) = `R(rumo)` · (0,75 − 0,50; 0) m. Fixo, sem sorteio (`knobs.py:94`, `:444`) | `eventos.py:127-129` |
| `dx` da caixa | `Cena.jitter_x_max[nível do PEGAR]` × U[0; 1). `jitter_x_max` = (0,20; 0,20; 0,20; 0,15; 0,08) m (`knobs.py:99`). Só para um lado | `eventos.py:358` |
| `dy` da caixa | Uniforme em `Cena.caixa_jitter_y` = (−0,18; 0,18) m (`knobs.py:97`) | `eventos.py:359` |

Colocação (`eventos.py:360-367`):
- Centro da caixa no frame da origem: `desloc` + `R(rumo)` · (`Cena.caixa_xy[0]` = 0,32 + `dx`; `Cena.caixa_xy[1]` = 0,00 + `dy`) (`knobs.py:96`). Soma-se a origem do env no mundo.
- Centro da laje: `desloc` + `R(rumo)` · (`Cena.prateleira_xy` = (0,50; 0,00)) (`knobs.py:94`). A laje é um corpo mocap. A pose é o centro do corpo: `topo` − `Cena.prateleira_meia_z` = 0,02 m (`eventos.py:276-294`, `knobs.py:90`). O rumo da laje é o rumo da cena.
- A caixa assenta no topo: z do centro = topo + `v3_meia`. O quaternion é a identidade, e a velocidade é zero (`eventos.py:297-307`).
- `v3_topo`, `v3_massa`, `v3_caixa_xy` e `v3_rumo_cena` ficam gravados para os eventos seguintes (`eventos.py:347-348`, `:364-365`).

A fatia `ANDAR` não passa por aqui para CARREGAR. Ela usa `estaciona_cena` e põe laje e caixa a `Cena.estaciona_z` = 20,0 m (`eventos.py:349-350`).

#### 2.3.3 `base` (`eventos.py:178-225`)

Para as fatias `B` e `C`, `_base_da_pega` põe o robô atrás da caixa, de frente para ela.

Sorteios (um uniforme por grandeza, por env):
- Distância `d` à caixa = `ref.recuo(v3_topo, v3_meia)` + (2·`u0` − 1) · `Alvos.faixa_recuo` · `jitter`. `Alvos.faixa_recuo` = 0,08 m (`knobs.py:227`). `jitter` = `Niveis.subnivel_jitter[subnível do PEGAR]`, com `subnivel_jitter` = (0,0; 1,0) (`knobs.py:424`) (`eventos.py:193-194`).
- Rumo (yaw) do robô = rumo da cena + U(`Cena.reset_base_manipula`["yaw"]) = U(−0,2; 0,2) rad (`knobs.py:101`, `eventos.py:195-198`).
- Altura extra `dz` = U(`reset_base_manipula`["z"]) = U(0,01; 0,05) m (`knobs.py:101`, `eventos.py:196,199`).

Pose escrita (`eventos.py:200-208`): xy = caixa − `R(rumo)` · (`d`; 0). z = z da origem + z do estado padrão da raiz + `dz`. O quaternion é o do estado padrão × o yaw. A velocidade escrita é a do estado padrão da raiz.

O z do estado padrão da raiz é `pelve_de_pe` + 0,01 (`cena.py:222-224`).

As velocidades linear e angular do estado padrão da raiz valem (0; 0; 0): `cena.py:222-227` não troca o padrão de `InitialStateCfg` (`mjlab/entity/entity.py:77-78`).

Não determinado pelo código lido: o valor numérico de `pelve_de_pe`. Ele sai de `referencias.npz`.

O valor de `ref.recuo` sai da grade `grade_recuo` de `referencias.npz` por interpolação bilinear em (topo, meia aresta) (`alvos.py:75-76`, `:30-37`). Não determinado pelo código lido: os valores da grade.

#### 2.3.4 `juntas` (`env_cfg.py:159-160`)

`reset_joints_by_offset` com `position_range` = (0,0; 0,0) e `velocity_range` = (0,0; 0,0). As 29 juntas ficam na pose padrão `q_de_pe` e com velocidade zero (`cena.py:225-226`). A posição é cortada nos limites moles das juntas (`mjlab/envs/mdp/events.py:296-299`). Não determinado pelo código lido: os valores de `q_de_pe` (arquivo `referencias.npz`).

#### 2.3.5 `carga` (`eventos.py:444-461`)

Escreve no modelo a massa de `v3_massa` e a inércia `(2/3) · m · a²` em cada um dos três eixos, com `a` = `v3_meia`. Não sorteia nada.

#### 2.3.6 `atrito_pad` e `atrito_laje` (`eventos.py:464-489`)

| Evento | Trava `portao0` desligada | Trava `portao0` ligada |
|---|---|---|
| `atrito_pad` (geoms `left_palm_pad` e `right_palm_pad`) | Fixo em `Cena.pad_atrito[0]` = 1,0 (`knobs.py:81`) | Uniforme em [`mu_min_pad`; `DR.atrito_pad[1]` = 1,0] |
| `atrito_laje` (geom `table_geom`) | Fixo em `Cena.prateleira_atrito[0]` = 1,0 (`knobs.py:91`) | Uniforme em `DR.atrito_laje` = (0,3; 1,0) (`knobs.py:524`) |

- A trava `portao0` fica verdadeira quando `nivel_passou("PEGAR") >= 0` (`balanceador.py:200`). Ela nunca volta a falso.
- `mu_min_pad` = `max(DR.mu_pad_min, DR.atrito_pad[0] − DR.passo_mu_pad × (nível do PEGAR + nível do CARREGAR))` com `mu_pad_min` = 0,5, `atrito_pad[0]` = 0,6 e `passo_mu_pad` = 0,025 (`knobs.py:523,525,526`, `balanceador.py:371-373`). Com a trava desligada, o evento não usa esse valor (`eventos.py:474`).
- A operação é `abs` na coluna 0 (atrito tangencial), com distribuição uniforme (`mjlab/envs/mdp/dr/geom.py:118-147`).
- A laje tem prioridade `Cena.prioridade_laje` = 3, acima da caixa (`Cena.caixa_priority` = 0, `knobs.py:82`, `:105`, `cena.py:113`, `:126`). A prioridade maior faz o atrito da laje valer no contato caixa–laje (semântica do campo `priority` do MuJoCo).

#### 2.3.7 `vies_imu` (`eventos.py:492-509`)

Liga quando a trava `vies` é verdadeira (`balanceador.py:201`: a célula `ANDAR/andar` passou no portão). Ligado: o viés do giro da base é uniforme em ± `DR.vies_imu_giro` = 0,05 rad/s por eixo, e a inclinação do IMU é uniforme em ± `DR.vies_imu_incl_graus` = 1,0° nos eixos x e y, com z = 0 (`knobs.py:521-522`). Desligado: giro zero e inclinação identidade.

#### 2.3.8 `pd`, `forca_motor` e `solref`

Sorteio por env no reset. Cada faixa vem de `Curriculo.dr()`, e o degrau global `g` a move. A §6 traz a tabela.

#### 2.3.9 `filtros` (`eventos.py:228-241`)

Zera o filtro de elevação do mapa, a odometria (que recomeça na pose verdadeira) e o estimador da caixa. O estimador sorteia de novo os vieses do episódio (§6.4).

### 2.4 Reset dos termos de comando

`cadeia._resample` (`comando.py:639-660`):
1. Registra o PEGAR pendente e o BOTAR pendente da cadeia do episódio que acaba, com `falhou` = não foi `time_out` (`comando.py:643-645`). Detalhe na §5.8.
2. A visita aberta da zona conta falha (`comando.py:647-650`).
3. Zera os flags `v3_pegou`, `v3_perdeu`, `v3_pega_fechada`, `v3_bota_fechada`, `v3_perdeu_agora`, `v3_avancou`, `v3_aviso` e `v3_fim` (`comando.py:651-653`).
4. `v3_relogio_alvo` = 0, `v3_sigma` = 1,0, `v3_phi` = −1,0, `v3_h` = 1,0, `v3_phi_melhor` = 0,0, `v3_v_chegada` = −1,0 (`comando.py:654-658`).
5. Abre o primeiro trecho da fatia: o ANDAR de espera, com duração uniforme em `espera_s` (`comando.py:659-660`, `:578-579`). O alvo fica em −1 (`comando.py:585-587`).

Depois, `twist.reset` fecha o segmento do episódio velho com as fotos tiradas na abertura dele, zera os filtros do episódio e reamostra (`comando.py:879-895`). O filtro `v_filt` zera, `q_filt` volta a 0,5 e `g_marcha` volta a 0. Isso está em `comando.py:886-888`.

O trecho PEGAR abre sem alvo (`v3_alvo_idx` = −1). O P0 abre no primeiro passo em que o estimador tem uma leitura válida da caixa: `tem_leitura` exige que a idade da última leitura entregue seja menor que `Estimador.idade_max` = 2,0 s (`comando.py:370-392`, `estimador.py:373-376`, `knobs.py:453`). Só nesse passo o PEGAR vira pendente de registro (`comando.py:384`). Enquanto o P0 não abre, o P3 não fecha e o relógio do trecho PEGAR não corre (§1.4).

---

## 3. Comando ou alvo

### 3.1 O que o robô recebe

O comando é o termo `twist`: um vetor de velocidade no frame da base, `vel_command_b` = (`vx`, `vy`, `wz`) (`comando.py:897-941`, `env_cfg.py:122-127`). Ele entra no ator pelo termo de observação `comando` (`observacoes.py:392`). Ele também entra nas recompensas, na marcha e no sucesso por segmento.

O alvo de posição não existe no CARREGAR. O índice `v3_alvo_idx` fica em P3, mas o ator recebe o alvo zerado (§9).

O `_update_command` do `twist` chama o do fabricante a cada passo (`comando.py:943-948`). Ele só muda o `wz` dos envs com `is_heading_env`. Só o ANDAR de terreno liga esse flag (`comando.py:903`, `:932`), e o `standing` e o `world` do fabricante ficam em False. No CARREGAR o comando fica como o sorteio o gravou.

### 3.2 Quando o comando é sorteado

| Fatia | Comando durante o CARREGAR |
|---|---|
| `B` | Sorteado uma vez, na abertura do trecho CARREGAR. Fica fixo até o fim do episódio: `time_left` vale ∞ (`comando.py:939-941`) |
| `C` | Zero. A condição `anda` só vale para CARREGAR na fatia `B` (`comando.py:905`). O env fica parado, sem célula (`v3_celula` = −1, `comando.py:901`) |

Os trechos ANDAR (espera) e PEGAR têm comando zero nas fatias `B` e `C` (`comando.py:905`).

O sorteio usa o nível do CARREGAR no instante da abertura (`comando.py:920`). Subida de nível durante o trecho não muda o comando.

`Comando.reamostra_s` = (3,0; 8,0) s (`knobs.py:137`, `env_cfg.py:123`) é o prazo que o `twist` sorteia (uniforme) a cada reamostra para o próximo reamostra automático (`mjlab/managers/command_manager.py:109-123`). Esse prazo não age no CARREGAR. Na fatia `B`, o `time_left` vale ∞ (`comando.py:939-941`). Na fatia `C`, o trecho dura no máximo `Cadeia.carregar_c_s[1]` = 1,0 s (`knobs.py:156`), menos que o mínimo de 3,0 s do prazo.

### 3.3 Escolha da célula

A célula é a classe de comando do CARREGAR. Há quatro: `reto`, `giro_lugar`, `arco` e `terreno` (`balanceador.py:22-24`).

Um env CARREGAR da fatia `B` recebe a célula assim (`comando.py:914-919`):
- Episódio de terreno (`terrain_types != 0`): célula `terreno`.
- Outro episódio: sorteio entre as células abertas do CARREGAR, com a fração da célula `terreno` posta em 0 (`sorteia_celulas(..., sem_tipo="terreno")`, `balanceador.py:306-316`). A escolha é `multinomial` com reposição.

Fração de cada célula aberta (`fracoes`, `balanceador.py:37-41`, `:295-299`):

`F_i = piso_sub + (1 − n · piso_sub) · (D_i + eps) / Σ_j (D_j + eps)`

- `n` é o número de células abertas do CARREGAR.
- `piso_sub` = `Balanceador.piso_sub` = 0,10 (`knobs.py:389`).
- `eps` = `Balanceador.eps` = 0,02 (`knobs.py:390`).
- `D_i` = `max(portao_i − s_i, 0)`. `s_i` é a nota da célula (§5.2). `portao_i` = `Balanceador.portao` = 0,75 para as células do CARREGAR (`knobs.py:391`, `balanceador.py:97-98`).

### 3.4 Tetos por nível

Os tetos vêm de `sorteia_velocidade` (`comando.py:148-158`) com `tarefa` ≠ ANDAR:
- `frente` = `Comando.frente_tetos[nível]` = (0,4; 0,55; 0,7; 0,85; 1,0) m/s (`knobs.py:122`).
- `tras` = `min(frente, Comando.tras_teto_max)`, com `tras_teto_max` = 0,5 m/s (`knobs.py:123`).
- `lado` = `min(frente, Comando.lado_teto_max)`, com `lado_teto_max` = 0,3 m/s (`knobs.py:124`).
- `giro` = `Comando.giro_tetos[min(nível, 3)]` = (0,3; 0,4; 0,5; 0,5) rad/s (`knobs.py:125`).
- Na célula `terreno`, `frente` e `tras` valem `min(valor, Comando.terreno_v_max)`, com `terreno_v_max` = 1,0 m/s (`knobs.py:148`, `comando.py:151-152`).

| Nível | `frente` (m/s) | `tras` (m/s) | `lado` (m/s) | `giro` (rad/s) |
|---|---|---|---|---|
| 0 | 0,4 | 0,4 | 0,3 | 0,3 |
| 1 | 0,55 | 0,5 | 0,3 | 0,4 |
| 2 | 0,7 | 0,5 | 0,3 | 0,5 |
| 3 | 0,85 | 0,5 | 0,3 | 0,5 |
| 4 | 1,0 | 0,5 | 0,3 | 0,5 |

### 3.5 Distribuição por célula

| Célula | Parte linear (`vx`, `vy`) | Giro (`wz`) |
|---|---|---|
| `reto` | Direção e módulo, abaixo | 0 |
| `terreno` | Igual a `reto`, com o teto da §3.4 | 0 |
| `arco` | Igual a `reto` | ± U[`piso_velocidade`; `giro`] |
| `giro_lugar` | 0 | ± U[`piso_velocidade`; `giro`] |

Referências: `comando.py:154-157`.

Parte linear (`_anda`, `comando.py:102-110`):
1. `eixo` é inteiro uniforme em {0, 1, 2, 3}. Cada direção tem 25 % de chance: 0 = +x, 1 = −x, 2 = +y, 3 = −y.
2. Teto: `frente` para o eixo 0, `tras` para o 1, `lado` para o 2 e 3.
3. Piso: `Comando.piso_velocidade` = 0,25 m/s para os eixos 0 e 1 (`knobs.py:126`); `Comando.piso_lado` = 0,2 m/s para os eixos 2 e 3 (`knobs.py:127`).
4. Módulo = piso + (teto − piso) · U[0; 1).
5. O vetor tem só um eixo diferente de zero.

Giro (`_giro`, `comando.py:113-115`): sinal uniforme em {−1, +1} e módulo = `piso_velocidade` + (`giro` − `piso_velocidade`) · U[0; 1).

A regra "só frente" do ANDAR de terreno (vx = `frente` · r, vy = 0, wz = 0, `comando.py:137-143`) não vale no CARREGAR. A célula `terreno` do CARREGAR segue o ramo `reto` (`comando.py:148-158`).

---

## 4. Etapas dentro do episódio e critério de avanço

### 4.1 A tarefa CARREGAR não tem etapas

O índice do alvo fica em P3 durante todo o CARREGAR. O avanço do alvo é desligado: `ok_agora` só vale para `tar == PEGAR` com `~v3_pega_fechada`, ou para BOTAR (`comando.py:471-482`). Em CARREGAR o relógio do alvo volta a zero a cada passo (`alvos.py:204`), e `v3_alvo_idx` não muda (`alvos.py:195-210`).

A visita da zona não conta no CARREGAR. A etapa da visita é −1 quando `idx == P3` e `v3_pega_fechada` (`comando.py:447`).

### 4.2 Como o env chega ao CARREGAR: a cadeia do PEGAR

O env só entra em CARREGAR depois do fecho do P3. O resumo da cadeia, com o critério exato, ajuda a ler o resto.

Um alvo avança quando `ok_agora` do alvo corrente vale sem interrupção por `tempo` segundos (`alvos.py:195-210`). `ok_agora` falso zera o relógio. Alvo com tempo 0 avança no primeiro passo com `ok_agora`. O tempo de cada alvo vem de `self.tempos`: 0 para P0 e P1, e o valor efetivo da zona para os outros (`tempo_p2`, `tempo_p3`, ...) (`comando.py:442-443`, `:482`).

| Alvo | `ok_agora` (`comando.py:41-56`) |
|---|---|
| P0 e P1 | Distância das duas palmas ao ponto-alvo no frame da caixa ≤ `tol_palma` do alvo, e `invariantes == 1` |
| P2 | Contato das duas palmas com a caixa (`palma_E` e `palma_D`), e `invariantes == 1` |
| P3 | `subida ≥ subida_min_p3` e `d_transporte ≤ tol_transporte_p3` e `d_pelve ≤ tol_pelve_p3` e `~apoio`, e `invariantes == 1` |

- `subida` é o fundo da caixa sobre o topo da laje da pega (`comando.py:473`).
- `d_transporte` é a distância da caixa ao ponto de transporte (`comando.py:469`).
- `d_pelve` = `|h − ref.pelve_com_caixa(v3_meia)|` (`comando.py:470`).
- `apoio` é o contato caixa–laje (`comando.py:333`).
- `invariantes` = `desloc · incl · maos`, produto de três portões que precisam valer exatamente 1,0 (`comando.py:462-466`, `recompensas.py:247-271`). Cada portão é uma rampa `clip(1 − (x − x_lim) / Δ, 0, 1)`, que só vale 1,0 com `x ≤ x_lim` (`potencial.py:23-29`). Por alvo:
  - `desloc`: nos P0 a P2, `rampa(deslocamento xy da caixa desde a pega; Alvos.desloc_p0p2_frac × a; Alvos.delta_p0p2_frac × a)`, com `desloc_p0p2_frac` = 1,0 e `delta_p0p2_frac` = 0,5 (`knobs.py:222-223`) e `a` = `v3_meia`. No P3 vale 1.
  - `incl`: nos P2 e P3, `rampa(incl_graus; incl_lim; Alvos.delta_incl_graus)`, com `incl_lim` = 30 − 3 × `degrau` graus (§5.5) e `delta_incl_graus` = 15,0 (`knobs.py:205`). Nos P0 e P1 vale 1.
  - `maos`: só no P3, igual ao número de palmas em contato dividido por 2. Nos P0 a P2 vale 1.

Valores do P3 no estado final e no estado largo (`knobs.py:186-189`, `:211-220`; o valor efetivo é `final + z · (largo − final)`, §5.6):

| Campo | Final | Largo (`z` = 1) |
|---|---|---|
| `subida_min_p3` | 0,03 m | 0,01 m |
| `tol_transporte_p3` | 0,05 m | 0,20 m |
| `tol_pelve_p3` | 0,03 m | 0,10 m |
| `tempo_p3` | 0,5 s | 0,2 s |

Fecho do P3 (`comando.py:492-496`): `v3_pega_fechada` vira verdadeiro e o índice fica em P3 (`novo = where(fecha_p3, idx, novo)`). O latch `v3_aviso` sobe no passo seguinte (§1.4). `v3_pegou` já ficou verdadeiro quando o P2 avançou (`comando.py:497`).

Na fatia `C`, o trecho CARREGAR dura `carregar_c_s` = (0,5; 1,0) s e é seguido pelo BOTAR. A §1.2 traz a tabela.

### 4.3 Entrada e saída do trecho

A §1.4 e a §1.5 descrevem a entrada e a saída. O trecho CARREGAR da fatia `B` acaba por tempo, em `carregar_b_s` = (4,0; 7,0) s, e acaba a cadeia. O trecho da fatia `C` acaba em `carregar_c_s` = (0,5; 1,0) s.

---

## 5. Progressão

O currículo é o objeto `Curriculo` em `balanceador.py`. Ele mora no env como `v3_cur` (`curriculo.py:46-47`).

### 5.1 Quando o currículo muda

O termo de currículo `v3` roda no começo de todo `_reset_idx` (`mjlab/envs/manager_based_rl_env.py:554`). Ele calcula `it = common_step_counter // Balanceador.passos_por_iteracao`. `passos_por_iteracao` = 24 (`knobs.py:402`, `curriculo.py:119`).

Quando `it` difere do `v3_iter` guardado (e `v3_iter >= 0`), ele chama `Curriculo.fecha_iteracao()` (`curriculo.py:120-122`). A primeira chamada só grava `v3_iter`. Ou seja: o balanceador fecha uma iteração a cada 24 passos do env, no primeiro reset depois da virada.

Ordem dentro de `fecha_iteracao` (`balanceador.py:153-195`):
1. EMA das células (§5.2).
2. Zona das etapas (§5.6).
3. Linha de terreno de cada coluna do ANDAR, pela taxa de travessia (`balanceador.py:177-182`; §5.3).
4. Contador do refino de cada tarefa (§5.5).
5. Aberturas (§5.3).
6. Marca de abertura das fatias e rampa das fatias (§1.3).
7. Subidas de nível e de linha de terreno (§5.4).
8. Travas do DR (§5.7).

### 5.2 Células e quem mede a nota

As quatro células do CARREGAR e a ordem de abertura estão em `LAYOUT` (`balanceador.py:22-24`): `reto`, `giro_lugar`, `arco`, `terreno`. O id da célula é o índice nessa lista: `CARREGAR/reto` = 2, `giro_lugar` = 3, `arco` = 4, `terreno` = 5.

Quem mede: o termo `twist` (`TwistV3`). Ele mede um segmento de comando e registra no currículo a nota do segmento na célula do segmento (`comando.py:759-808`). O `_abre` zera o wz do `seg_cmd` só nos envs de rumo, que só o ANDAR de terreno liga. No CARREGAR o segmento copia o comando inteiro (`comando.py:738`, `:742`).

Um segmento é o intervalo em que o comando é constante. O segmento do CARREGAR da fatia `B` dura o trecho inteiro. Ele fecha quando o contador de comandos muda (`comando.py:816-821`), no reset do episódio (`comando.py:879-884`), ou na perda da caixa (§8.3).

Nota do segmento CARREGAR, `sucesso_do_segmento` (`comando.py:227-242`) com `seg_carrega` = verdadeiro:
- `ativo_lin` = `‖cmd_xy‖ ≥ Comando.limiar_canal` = 0,02 (`knobs.py:138`). `ativo_giro` = `|cmd_z| ≥ limiar_canal`.
- `razao` = `int_proj / max(int_cmd, 1e-6)`. `int_proj[0]` = Σ (`v_xy · ĉ`) · `step_dt` no segmento inteiro. `int_cmd[0]` = Σ `‖cmd_xy‖` · `step_dt`. Para o giro: `int_proj[1]` = Σ `wz · sinal(cmd_z)` · `step_dt`, e `int_cmd[1]` = Σ `|cmd_z|` · `step_dt` (`comando.py:823-831`). A velocidade é a da base no frame da base.
- `ok_lin` = `razao[0] ≥ Comando.razao_portao` = 0,50 (`knobs.py:146`) se o canal linear é ativo. Se não, `ok_lin` = `deriva_pos < lim_pos`.
- `ok_giro` = `razao[1] ≥ 0,50` se o canal de giro é ativo. Se não, `ok_giro` = `deriva_rumo < Comando.deriva_rumo_rad` = 0,25 rad (`knobs.py:144`).
- `lim_pos` = `no_degrau(Comando.deriva_pos_inicio, Comando.deriva_pos_m, restricao, 1)` = 0,25 + (0,10 − 0,25) · `restricao` m (`knobs.py:142-143`, `comando.py:773`). `restricao` é o menor `nível / máximo` das quatro tarefas (`balanceador.py:345-348`).
- `deriva_pos` e `deriva_rumo` medem a pose no fecho contra a âncora. A âncora grava a pose em `max(abertura do segmento, último empurrão) + Comando.ancora_s` = 1,5 s (`knobs.py:145`, `comando.py:221-224`, `:833-842`).
- `sucesso = ok_lin & ok_giro & ~perdeu`. Na célula `terreno` também `~(incl_max_seg > Comando.incl_max_terreno_graus)`, com 15,0° (`knobs.py:149`, `comando.py:240`). `incl_max_seg` é o pico, no segmento, da inclinação da caixa (`comando.py:753-757`).
- O segmento registra quando `t_fecho ≥ t_ancora` (o fecho vem depois da âncora) ou quando acaba em falha (`comando.py:241`, `:790`). Um empurrão (§6.1) adia a âncora para o instante do empurrão + `ancora_s`. Se esse instante cai depois do fecho, o segmento não registra, salvo em falha (`comando.py:221-224`, `:834-837`). Falha = terminação que não é `time_out`, ou perda da caixa no segmento com caixa (`comando.py:820`, `:882-883`).
- A nota é `x` = `sucesso` (0 ou 1) × `(~falha)` (`comando.py:789`). O código chama `env.v3_cur.registra(célula, n, soma)` (`comando.py:792-794`).

A fatia `C` não registra nota de CARREGAR: o segmento tem `v3_celula` = −1 e o laço só registra células ≥ 0 (`comando.py:792`).

Fecho da iteração, EMA por célula (`balanceador.py:162-168`):
- Se a célula registrou `n > 0` segmentos na iteração: `x = soma / n`.
- Se `n_medidas == 0`: `s = x`. Senão: `s += alfa · (x − s)`, com `Balanceador.alfa` = 0,05 (`knobs.py:394`).
- `n_medidas` sobe 1. Sem registro, `s` não muda. `s` começa em 0.

### 5.3 Abertura das células

`_abre` (`balanceador.py:204-232`) roda uma vez por fecho de iteração.

| O que abre | Condição exata |
|---|---|
| `CARREGAR/reto` (e `PEGAR/pegar`) | `nivel["ANDAR"] >= Balanceador.nivel_abre_b` = 5 (`knobs.py:396`) e `reto` ainda fechada |
| `CARREGAR/giro_lugar` e depois `CARREGAR/arco` | `reto` aberta e há célula de velocidade fechada e `passa_media(abertas)`. Uma célula por fecho, na ordem `reto`, `giro_lugar`, `arco` |
| `CARREGAR/terreno` | `nivel["CARREGAR"] == max_nivel["CARREGAR"]` (= 4) e a célula ainda fechada e `nivel_terreno("ANDAR") >= 1` e `passa_media(abertas)` (`balanceador.py:218-228`) |

`nivel_terreno("ANDAR")` é o piso da média das linhas das 8 colunas de terreno do ANDAR: `floor(Σ linha[1:] / 8)` (`balanceador.py:329-333`). A linha de cada coluna sobe e desce pela taxa de travessia (`balanceador.py:177-182`), com `linha_ganho` = 2,5, `linha_n_min` = 50 e `zona_t_alvo` = 0,6 (`knobs.py:410-411`, `:404`). A célula `ANDAR/terreno` abre no nível `Balanceador.nivel_abre_terreno` = 2 do ANDAR (`balanceador.py:222`, `knobs.py:412`). A nota da célula `ANDAR/terreno` não entra na abertura do CARREGAR.

`passa_media(ids)` (`balanceador.py:125-128`): `ids` não vazio e a média de `s` das células `ids` é ≥ a média dos portões delas. Para o CARREGAR o portão vale 0,75 em todas as células (`knobs.py:391`).

Uma célula nova abre com `s` = 0 e `n_medidas` = 0. Ela não passa no mesmo fecho.

### 5.4 Subida de nível e de linha de terreno

`_sobe` (`balanceador.py:250-278`) roda uma vez por fecho de iteração, para cada tarefa.

Condição `ok` do CARREGAR:
- Se `nivel < max_nivel` (4): todas as três células de velocidade (`reto`, `giro_lugar`, `arco`) abertas e `passa_media` delas verdadeira.
- Senão, se `nivel_terreno < max_ter` (9): a célula `terreno` aberta e `s_terreno >= 0,75` (`_passa`, `balanceador.py:122-123`). `max_ter` do CARREGAR = maior valor de `Terreno.linha_max_carregar` = 9 (`balanceador.py:77-78`).
- Senão: `ok` falso.

Contagem: `seguidas[CARREGAR]` sobe 1 por fecho com `ok`. Um fecho sem `ok` zera. Com `seguidas ≥ Balanceador.iters_nivel` = 50 (`knobs.py:395`), `seguidas` volta a 0 e acontece uma subida (`balanceador.py:265-268`):
- Se `nivel == max_nivel`: `nivel_terreno["CARREGAR"] += 1`, e `n_medidas` da célula `terreno` volta a 0.
- Senão: `nivel["CARREGAR"] += 1` (o subnível do CARREGAR vale sempre o último, então não há subnível em andamento, `balanceador.py:325-327`). `n_medidas` de todas as células do CARREGAR volta a 0. O `s` fica até a primeira medida nova.

| O que sobe | O que muda no env | Quem mede |
|---|---|---|
| `nivel["CARREGAR"]` 0 → 4 | Os tetos do comando (§3.4). `v3_freio` do próximo trecho = nível + 1 (`balanceador.py:335-336`). A trava `subida` liga no 1º nível ≥ 1 do PEGAR ou do CARREGAR (§5.7). `mu_min_pad` cai 0,025 por nível, até `mu_pad_min` (§2.3.6). A conta de `restricao` (§5.7) | Notas das células `reto`, `giro_lugar`, `arco` (§5.2) |
| `nivel_terreno["CARREGAR"]` 0 → 9 | A linha máxima sorteada nos episódios de terreno da fatia `B` (§2.3.1) | Nota da célula `terreno` |

### 5.5 Degrau de refino

`_conta_refino` (`balanceador.py:234-248`) roda antes das subidas, no mesmo fecho.

Condição para o contador subir (`cont[CARREGAR] += 1`): `nivel == max_nivel` e a célula `terreno` aberta e `nivel_terreno == max_ter` (9) e `passa_media` das células abertas do CARREGAR. Se a condição falha, `cont` volta a 0.

Com `cont ≥ Balanceador.iters_refino` = 100 (`knobs.py:400`): `degrau["CARREGAR"] += 1` e `cont` volta a 0. O máximo é `Balanceador.degraus` = 5 (`knobs.py:401`). O degrau nunca desce.

O env lê o degrau do CARREGAR na abertura de cada trecho (`v3_degrau`, `comando.py:512-518`). O que o degrau `d` do CARREGAR muda (todos usam `v3_degrau` do env):

| Efeito | Valor | Referência |
|---|---|---|
| Inclinação máxima da caixa, `incl_lim` | `no_degrau(Alvos.incl_inicio_graus` = 30, `Alvos.incl_fim_graus` = 15, `d`, 5) = 30 − 3·`d` graus. Entra no portão `incl` do fecho do P3 (§4.2) e na qualidade do rastreio (§7.3) | `recompensas.py:462-464`, `comando.py:464`, `recompensas.py:587`, `knobs.py:203-204` |
| Fator da `sacudida` e do `aperto` | `d / 5` | `recompensas.py:944`, `:955` |
| Terminação `batente` liga | Quando `d ≥ Refino.degrau_batente` = 1 | `terminacoes.py:164`, `:180-181`, `knobs.py:547` |
| Observação `degrau` do crítico | `d / 5` | `observacoes.py:338-340` |

O degrau global `g` é o menor degrau das quatro tarefas (`balanceador.py:341-343`). Ele move o DR e o ruído de estimação (§6).

### 5.6 Zona adaptativa

A zona é uma escala `z` em [0, 1] por etapa do PEGAR e do BOTAR. `z` = 1 é o critério largo. `z` = 0 é o critério final (`balanceador.py:9-10`, `knobs.py:211-220`). O CARREGAR não tem zona própria. Ele lê a zona da etapa P3 (`z[3]`), pela `tol_transporte_p3` efetiva da qualidade do rastreio (§7.3).

Atualização (`balanceador.py:170-176`), uma vez por fecho de iteração:
- Se `vis_acum[e] ≥ Balanceador.zona_n_min` = 200 (`knobs.py:406`): `taxa[e] = fech_acum[e] / vis_acum[e]`; `z[e] ← clip(z[e] − Balanceador.zona_ganho · (taxa − Balanceador.zona_t_alvo), 0, 1)` com `zona_ganho` = 0,05 e `zona_t_alvo` = 0,6 (`knobs.py:404-405`); os acumuladores voltam a 0.
- Depois, para `e` em P1, P2, P3, B2 e B3: `z[e] = max(z[e], z[e−1] − Balanceador.zona_folga_cadeia)`, com `zona_folga_cadeia` = 0,2 (`knobs.py:408`).

Visitas e fechos: o termo `cadeia` conta (`comando.py:444-491`). Uma visita é uma etapa aberta. Ela acaba com o fecho, ou sem fecho quando `zona_t_visita` = 8,0 s (`knobs.py:407`) passam desde a abertura. Também acaba na perda da caixa, na troca de etapa por fora e no reset (`comando.py:647-650`).

O `Alvos` efetivo é refeito uma vez por iteração (`cur.it` mudou): `valor = final + z · (largo − final)` por campo, de `Alvos.zona` (`comando.py:428-443`, `knobs.py:211-220`). O CARREGAR lê só `tol_transporte_p3` (0,05 a 0,20 m).

### 5.7 Travas do DR, `restricao` e degrau global

As travas só ligam, nunca desligam (`balanceador.py:197-202`).

| Trava | Condição para ligar | Efeito que alcança o CARREGAR |
|---|---|---|
| `portao0` | `nivel_passou("PEGAR") >= 0` | Atrito de pad e de laje entram no sorteio (§2.3.6) |
| `vies` | Célula `ANDAR/andar` passa no portão | `vies_imu` liga (§2.3.7) |
| `subida` | `nivel["PEGAR"] + nivel["CARREGAR"] > 0` | `forca_caixa` e `rfi` ligam (§6) |

`restricao` = `min(nivel[t] / max_nivel[t])` sobre as quatro tarefas (`balanceador.py:345-348`). Os máximos: ANDAR 7, CARREGAR 4, PEGAR 4, BOTAR 4 (`balanceador.py:72-74`). Ela fixa:
- O fator de estilo `0,25 + 0,75 · restricao` (§7.3).
- O `sigma_zero_ang` do sinal de renda do CARREGAR (§7.3).
- O teto `lim_pos` da deriva (§5.2).
- O teto do atraso (§6.2).

### 5.8 Acoplamento com a nota do PEGAR

O CARREGAR não registra nota do PEGAR. Mas a queda e a perda da caixa durante o CARREGAR contam contra a célula `PEGAR/pegar`:
- O PEGAR continua pendente de registro até a abertura do BOTAR (fatia `C`) ou até o reset (`comando.py:521-540`, `:384`).
- No reset, `falhou` = o fim não foi `time_out` (`comando.py:643`). O sucesso do PEGAR = `v3_pega_fechada & ~falhou & ~v3_perdeu` e todos os `z` das etapas do PEGAR (`z[:4]`) iguais a 0 (`comando.py:537`).
- Logo: queda ou batente no CARREGAR, ou perda da caixa no CARREGAR, registram falha do PEGAR.

---

## 6. Perturbações e randomizações

Todas valem em CARREGAR salvo nota. O modo `play` as desliga em parte (`env_cfg.py:81-83`, `:200`).

### 6.1 Tabela

| Perturbação | Quando | Distribuição e faixa | Condição | Referência |
|---|---|---|---|---|
| Empurrão na base | Intervalo, por env. O tempo até o próximo empurrão é sorteado por env | Tempo uniforme em `DR.periodo_empurrao_s` = (3,0; 7,0) s. Cada eixo (`x`, `y`) soma uma velocidade uniforme em ± `DR.empurrao` = 0,5 m/s à velocidade da raiz | Só fora do `play`. O timer não reinicia no reset, porque `empurrao` é função, não classe. O primeiro prazo de cada env é sorteado na criação do env, uniforme em `periodo_empurrao_s`. Cada disparo grava `v3_t_empurrao` = `episode_length_buf × step_dt`, que adia a âncora do segmento de comando (§5.2) | `env_cfg.py:201-205`, `knobs.py:517-520`, `eventos.py:261-269`, `mjlab/envs/mdp/events.py:316-337`, `mjlab/managers/event_manager.py:226`, `:278-286`, `:367-375` |
| Força na caixa | Passo, depois do `command_manager.compute` (`mjlab/envs/manager_based_rl_env.py:456-459`). Cada env tem um timer próprio, sorteado na criação do env e de novo no reset do env | Cada componente (x, y, z) uniforme em `DR.forca_caixa_N` = (−11,5; 11,5) N. Torque (0; 0). Duração uniforme em `DR.forca_caixa_dur_s` = (0,1; 0,3) s. Pausa uniforme em `DR.forca_caixa_periodo_s` = (3,0; 8,0) s, sorteada de novo no fim de cada impulso | Trava `subida` ligada (sem a trava o termo sai antes de andar o timer) e o env em `(v3_tarefa == CARREGAR) & v3_pegou & ~v3_perdeu`. Fora disso a força é zerada logo depois de escrita e o timer segue | `env_cfg.py:207-209`, `eventos.py:584-602`, `knobs.py:530-533`, `mjlab/envs/mdp/events.py:386-552` |
| RFI (ruído de torque) | Passo, depois do comando; o valor vale nos 4 subpassos de física do passo seguinte | Por env e por junta: `(2u − 1) · s · TAU_MAX[j]`, com `u` uniforme. `s = no_degrau(DR.rfi[0], DR.rfi[1], g, 5)` = 0,05 + 0,01·`g` com a trava `subida`; senão 0 | Só fora do `play`. Com `s` = 0 o evento escreve zero | `eventos.py:605-615`, `balanceador.py:384`, `knobs.py:534` |
| Ruído da observação do ator | Todo passo | Uniforme em ± `DR.ruido_obs`: `giro_base` 0,2; `gravidade` 0,05; `juntas` 0,01; `vel_juntas` 1,5, antes da escala | Só no ator, só com a corrupção ligada | `observacoes.py:373-387`, `knobs.py:516`, `mjlab/managers/observation_manager.py:332-353` |
| Atraso da ação e da observação | Primeiro passo depois do reset; fica no episódio inteiro | Inteiro uniforme em [0; teto] por env (`delay_min_lag` = 0). Teto = `atraso_max` do `Curriculo.dr()`, em passos da política na observação e em `atraso_max × Contrato.decimacao` passos de física no atuador | Ver §6.2 | `mjlab/utils/buffers/delay_buffer.py:194-213`, `:247-270`, `observacoes.py:382-386`, `cena.py:188-197`, `eventos.py:244-258` |
| Viés do giro e da inclinação do IMU | Reset | Ver §2.3.7 | Trava `vies` | `eventos.py:492-509` |
| Atrito do pé, massa do tronco, viés de encoder, tamanho da caixa | Startup | Ver §2.2 | Sempre | `env_cfg.py:185-198` |
| Atrito do pad e da laje | Reset | Ver §2.3.6 | Trava `portao0` | `eventos.py:464-489` |
| Ganho PD | Reset | Escala uniforme em `Curriculo.dr()["pd"]`, kp e kd com sorteios separados, por env e por alvo de atuador (`mjlab/envs/mdp/dr/actuator.py:29-92`) | Só fora do `play` | `eventos.py:524-531`, `env_cfg.py:169-172` |
| Força do motor | Reset | Escala uniforme em `dr()["forca_motor"]` (pernas e cintura) e `dr()["forca_motor_bracos"]` (braços), sobre o limite de esforço, com uma amostra por env e por alvo de atuador (`mjlab/envs/mdp/dr/actuator.py:191-265`) | Só fora do `play` | `eventos.py:535-545`, `env_cfg.py:173-175` |
| `solref[0]` dos contatos | Reset | Um valor uniforme em `dr()["solref"]` por env, para pés, pads de palma, pads de dorso, laje e caixa | Só fora do `play` | `eventos.py:549-565`, `env_cfg.py:177-180` |
| Estimador da caixa e odometria | Ver §6.4 | Ver §6.4 | Sempre | `observacoes.py:76-90` |

Não determinado pelo código lido: como o ruído do lidar (`Percepcao.ruido_alcance` = 0,02 m, `ruido_angulo_graus` = 0,15°, `perda` = 0,01, `ruido_incl_graus` = 0,5°, `knobs.py:491-494`) entra no mapa. O módulo `percepcao_lidar.py` o aplica, e esse módulo não faz parte das fontes lidas. O gerador desse ruído é `v3_gerador` (`observacoes.py:229-233`).

### 6.2 Faixas por degrau global `g` (`balanceador.py:358-389`)

`no_degrau(ini, fim, d, degraus)` = `ini + (fim − ini) · d / degraus`, com `degraus` = 5 (`balanceador.py:48-51`, `knobs.py:401`).

| Grandeza | Início (`g` = 0) | Fim (`g` = 5) | Faixa no degrau `g` | Referência |
|---|---|---|---|---|
| `pd` | (1; 1) | (0,9; 1,1) (`DR.pd`) | [1 − 0,02·`g`; 1 + 0,02·`g`] | `knobs.py:535` |
| `forca_motor` | (1; 1) | (0,9; 1,1) (`DR.forca_motor`) | [1 − 0,02·`g`; 1 + 0,02·`g`] | `knobs.py:536` |
| `forca_motor_bracos` | (1; 1) | (0,8; 1,1) (`DR.forca_motor_bracos`) | [1 − 0,04·`g`; 1 + 0,02·`g`] | `knobs.py:537` |
| `solref` | `DR.solref_base` = 0,02 | (0,015; 0,03) (`DR.solref`) | [0,02 − 0,001·`g`; 0,02 + 0,002·`g`] | `knobs.py:539-540` |
| `rfi` (fração do `TAU_MAX`) | `DR.rfi[0]` = 0,05 | `DR.rfi[1]` = 0,1 | 0,05 + 0,01·`g`; vale 0 sem a trava `subida` | `knobs.py:534` |
| `atraso_max` (passos) | `ceil(DR.atraso_portao[1] · restricao − 1e-9)`, com `atraso_portao[1]` = 2 | `DR.atraso_refino[1]` = 3 | 3 se `g ≥ DR.degrau_atraso_refino` = 3; senão o valor da primeira coluna | `knobs.py:512-514` |

O termo `v3` leva `atraso_max` aos buffers só quando ele difere de `DR.atraso_inicio[1]` (`curriculo.py:123-126`).

### 6.3 Ruído de observação no ator

O ruído é uniforme e aditivo (`Unoise(n_min = −r, n_max = r)`, `observacoes.py:373-381`). Por termo, o gerenciador de observação aplica nesta ordem: ruído, corte, escala, atraso e histórico (`mjlab/managers/observation_manager.py:332-353`). Só o ator tem ruído e atraso. O atraso vale nos termos `giro_base`, `gravidade`, `juntas`, `vel_juntas` e `mapa` (`observacoes.py:55`, `:382-386`). O crítico não tem ruído nem atraso (`observacoes.py:403-404`). Além do ruído, o ator vê o `giro_base` somado ao viés fixo `v3_vies_giro` e a `gravidade` girada pela inclinação fixa `v3_incl_imu` (§2.3.7), e vê as `juntas` com o viés do encoder (`observacoes.py:202-209`, `:394`).

### 6.4 Estimador da caixa e odometria por degrau global

A observação `caixa` do ator vem do estimador, não da verdade (§9). O estimador usa as magnitudes `_magnitude(Estimador, g, 5)` (`observacoes.py:76-90`, `:189`). Cada magnitude é `no_degrau(início, refino, g, 5)`.

| Magnitude | Início | Refino | Referência |
|---|---|---|---|
| Posição, σ absoluto (m) | `Estimador.sigma_pos[0]` = 0,005 | 0 | `knobs.py:471`, `observacoes.py:84` |
| Posição, σ relativo à distância câmera–caixa | 0 | `Estimador.sigma_pos[1]` = 0,02 | `knobs.py:471`, `observacoes.py:85` |
| Viés de orientação, mediana (graus) | `Estimador.ori_graus[0][0]` = 1,5 | `ori_graus[1][0]` = 5,3 | `knobs.py:465`, `observacoes.py:83,86` |
| Viés de orientação, p90 (graus) | `ori_graus[0][1]` = 2,5 | `ori_graus[1][1]` = 15,3 | `knobs.py:465`, `observacoes.py:87` |
| Ruído de orientação por leitura (graus) | `ori_leitura_graus[0]` = 0,0 | `ori_leitura_graus[1]` = 1,0 | `knobs.py:466` |
| Probabilidade de perda da leitura | `Estimador.perda[0]` = 0,005 | `perda[1]` = 0,022 | `knobs.py:472` |
| Viés da meia aresta (m) | `vies_meia[0]` = 0,005 | `vies_meia[1]` = 0,02 | `knobs.py:459` |
| Viés de posição (m) | `vies_pos[0]` = 0,0 | `vies_pos[1]` = 0,02 | `knobs.py:460` |
| Taxa de rajada de perda (1/s) | `rajada_taxa[0]` = 0,0 | `rajada_taxa[1]` = 0,035 | `knobs.py:469` |

Outros parâmetros do estimador (`knobs.py:450-478`): `Estimador.taxa_hz` = 30,0 Hz; `latencia_s` = 0,033 s; `idade_max` = 2,0 s; `presa_margem` = 0,05 m; `rajada_s` = (0,3; 1,0) s; `pontos_visiveis_min` = 3. A leitura cai numa grade de tempo de passo 1/`taxa_hz` e chega depois de `latencia_s` (`estimador.py:204-209`, `:319-352`).

Sorteios do estimador:
- No reset do env (`estimador.py:262-280`, `eventos.py:241`), o estimador sorteia quatro quantidades fixas do episódio: `z_meia` e `z_pos` (cada eixo) uniformes em [−1; 1], o eixo do viés de orientação (uniforme na esfera, a partir de três normais normalizadas) e `z_ang` (normal padrão). A escala do degrau global entra depois, a cada passo: viés da meia aresta = `z_meia × vies_meia`, viés de posição = `z_pos × vies_pos` (por eixo), ângulo do viés de orientação = `mediana × exp(σ_ln × z_ang)` com `σ_ln = ln(p90 / mediana) / Φ⁻¹(0,9)`, com `Φ⁻¹(0,9)` = `statistics.NormalDist().inv_cdf(0.9)` (`estimador.py:14`, `:294-309`).
- A cada passo, o estimador sorteia uma rotação de leitura: cada componente do vetor de rotação vale `ori_leitura_graus × U[−1; 1]` graus (`estimador.py:307`).
- A cada amostra da grade de tempo (`estimador.py:320-338`), a posição soma `(σ_abs + σ_rel × distância câmera–caixa) × U[−1; 1]` por eixo. O σ vale como meia largura da uniforme.
- A leitura é perdida se a caixa não é visível (§9.1), se `U[0; 1) < perda`, ou se a leitura cai dentro de uma rajada (`estimador.py:331`).
- A rajada abre a cada passo, num env fora de rajada, com probabilidade `rajada_taxa × dt` (`estimador.py:311-317`). A duração é uniforme em `Estimador.rajada_s` = (0,3; 1,0) s (`knobs.py:468`). A rajada não é sorteada no reset: o reset só a fecha (`estimador.py:280`).

A odometria soma ruído normal a cada passo (`estimador.py:166-174`). O desvio de x, y e z vale `√(odo² · ‖Δp‖ + pos_giro² · |Δψ|)`, e o do rumo vale `√(rumo² · ‖Δp‖ + rumo_giro² · |Δψ|)`. As derivas vão pelo degrau global: `odo` (0,002 → 0,0075), `rumo` (0,002 → 0,005), `pos_giro` (0,004 → 0,008), `rumo_giro` (0,0035 → 0,014) (`comando.py:247-250`, `:623-625`, `estimador.py:158-184`, `knobs.py:473-476`).

---

## 7. Recompensas e penalidades

### 7.1 Regras comuns

- Cada termo é `valor × peso × step_dt` (`mjlab/managers/reward_manager.py:127`). NaN e infinito viram 0 (`:129`).
- O termo vale zero fora das suas tarefas por `torch.where(mascara, valor, 0)` (`recompensas.py:361-365`).
- `paga=True` zera o termo depois que `v3_perdeu` vira verdadeiro (`recompensas.py:363-364`). Os termos `rastreio_linear`, `rastreio_angular`, `passo` e `altura_pe` usam `paga=True`. O `vivo` zera por conta própria (`n_vivo`, `recompensas.py:85-87`).
- O fator de estilo `por_restricao(estilo_inicio)` = `no_degrau(0,25; 1,0; restricao, 1)` = 0,25 + 0,75 · `restricao`, com `Recompensa.estilo_inicio` = 0,25 (`knobs.py:260`, `recompensas.py:453-459`). Ele vale em todos os envs nos termos 8, 9, 10 e 12 (`recompensas.py:674`, `:679`, `:684`, `:697`). Nos três desvios (14 a 16) a chamada leva `so="CARREGAR"`: o fator vale só nos envs da tarefa CARREGAR, e nas outras tarefas vale 1 (`recompensas.py:457-458`, `:710`, `:716`, `:722`).
- O fator das cercas é `por_restricao(0,0; so="ANDAR")` (`recompensas.py:747,758,764,772,788`). Em CARREGAR o fator vale 1, porque `so="ANDAR"` só vale nos envs ANDAR (`recompensas.py:457-458`).
- As recompensas rodam antes de `command_manager.compute` no passo (`mjlab/envs/manager_based_rl_env.py:440,456`). Os termos de marcha (`fase`, `v_filt`, `g_marcha`) têm o valor do passo anterior.

### 7.2 Tabela dos termos registrados para CARREGAR

A tabela lista todos os 31 termos de `_RECOMPENSAS` (`env_cfg.py:42-54`) cuja tupla de tarefas contém `"CARREGAR"`. As tuplas: `ANDAR_CARREGAR` (`env_cfg.py:34`), `TODAS` (`env_cfg.py:33`) e `PEGAR_CARREGAR_BOTAR` (`env_cfg.py:36`). O peso do cfg é o knob (`env_cfg.py:131-134`), salvo `progresso` e `perda_caixa`, que usam ±1 (`env_cfg.py:57`).

O fator `dt` = 0,02 s multiplica todos. A coluna "Fatores" lista só os outros.

| # | Termo (tarefas) | Peso (knob, valor, linha) | Fórmula resumida | Fatores | Zero quando |
|---|---|---|---|---|---|
| 1 | `rastreio_linear` (ANDAR, CARREGAR) | `Recompensa.rastreio_linear` = 2,0 (`knobs.py:236`) | Tenda da fração entregue × `g_marcha` do comando linear, vezes a qualidade da caixa (§7.3) | `paga` | Tarefa fora de ANDAR e CARREGAR; `v3_perdeu` |
| 2 | `rastreio_angular` (ANDAR, CARREGAR) | `Recompensa.rastreio_angular` = 1,0 (`knobs.py:237`) | Mesma forma, para o giro | `paga` | Idem |
| 3 | `passo` (ANDAR, CARREGAR) | `Recompensa.passo` = 0,5 (`knobs.py:244`) | `Σ_pés [contato == apoio] × _avanco`, em [0; 2] | `paga` | Tarefa fora do par; `v3_perdeu`. Pago mesmo parado (§7.3) |
| 4 | `vivo` (ANDAR, CARREGAR) | `Recompensa.vivo` = 1,0 (`knobs.py:248`) | `1 − v3_perdeu` | nenhum | `v3_perdeu` |
| 5 | `altura_pe` (ANDAR, CARREGAR) | `Recompensa.altura_pe` = 1,0 (`knobs.py:249`) | `exp(−Σ_pés (h − 0,12)² · tanh(2,0 · ‖v_xy‖) / 0,0025) × _avanco`, com `h` descontada (§7.3) | `paga` | `‖comando‖ = 0` (comando exatamente zero); `v3_perdeu` |
| 6 | `escorrego` (TODAS) | `Recompensa.escorrego` = −0,2 (`knobs.py:253`) | `Σ_pés ‖v_xy‖ · contato` | nenhum | Pé no ar |
| 7 | `vz` (TODAS) | `Recompensa.vz` = −2,0 (`knobs.py:254`) | `v_z²` da base no mundo | nenhum | Nunca fica zero por regra |
| 8 | `giro_xy` (TODAS) | `Recompensa.giro_xy` = −0,05 (`knobs.py:255`) | `ω_x² + ω_y²` da base, no frame da base | estilo | Nunca por regra |
| 9 | `vel_junta` (TODAS) | `Recompensa.vel_junta` = −0,001 (`knobs.py:256`) | `Σ q̇²` nas 29 juntas | estilo | Nunca por regra |
| 10 | `acel_junta` (TODAS) | `Recompensa.acel_junta` = −5e-8 (`knobs.py:257`) | `Σ q̈²` nas 29 juntas | estilo | Nunca por regra |
| 11 | `taxa_acao` (TODAS) | `Recompensa.taxa_acao` = −0,01 (`knobs.py:258`) | `Σ (a − a_anterior)²` | nenhum | Nunca por regra |
| 12 | `energia` (TODAS) | `Recompensa.energia` = −2e-5 (`knobs.py:259`) | `Σ \|τ · q̇\|`, com `τ` = `qfrc_actuator` | estilo | Nunca por regra |
| 13 | `limite_junta` (TODAS) | `Recompensa.limite_junta` = −5,0 (`knobs.py:262`) | `Σ_j relu(\|q − c\| / m − 0,95) / 0,05` | nenhum | Junta abaixo de 95 % do curso |
| 14 | `desvio_bracos` (ANDAR, CARREGAR) | `Recompensa.desvio_bracos` = −0,1 (`knobs.py:264`) | `Σ \|q − q_com_caixa(v3_meia)\|` nas 14 juntas dos braços | estilo (só CARREGAR) | Tarefa fora do par |
| 15 | `desvio_cintura` (ANDAR, CARREGAR) | `Recompensa.desvio_cintura` = −1,0 (`knobs.py:265`) | Idem, nas 3 juntas da cintura | estilo (só CARREGAR) | Idem |
| 16 | `desvio_quadril` (ANDAR, CARREGAR) | `Recompensa.desvio_quadril` = −1,0 (`knobs.py:266`) | Idem, em `hip_roll` e `hip_yaw` das duas pernas (4 juntas) | estilo (só CARREGAR) | Idem |
| 17 | `tronco_plano` (ANDAR, CARREGAR) | `Recompensa.tronco_plano` = −20,0 (`knobs.py:267`) | `‖g_xy‖²`, gravidade projetada no frame da base | nenhum | Tarefa fora do par |
| 18 | `altura_base` (ANDAR, CARREGAR) | `Recompensa.altura_base` = −10,0 (`knobs.py:268`) | `(h_pelve − 0,78)²`, com `Recompensa.alvo_base` = 0,78 (`knobs.py:269`) | nenhum | Tarefa fora do par |
| 19 | `cerca_giro` (TODAS) | `Recompensa.cerca_giro` = −0,03965 (`knobs.py:271`) | `relu(max(−q_yawE,0)° − 30) + relu(max(q_yawD,0)° − 30)`, com `Recompensa.lim_giro_graus` = 30,0 (`knobs.py:278`) | cerca (1 em CARREGAR) | Giro para dentro ≤ 30° |
| 20 | `cerca_tronco` (TODAS) | `Recompensa.cerca_tronco` = −0,04596 (`knobs.py:272`) | `relu(incl_torso − (0 + 15°))`, com `Recompensa.lim_tronco_graus` = 15,0 (`knobs.py:279`); `tronco_IK` = 0 em CARREGAR | cerca | Inclinação do torso ≤ 15° |
| 21 | `cerca_com` (TODAS) | `Recompensa.cerca_com` = −22,09035 (`knobs.py:273`) | `relu(0,03 − margem_CoM)` com os dois pés em contato | cerca | Algum pé no ar; margem ≥ 0,03 m |
| 22 | `cerca_torque` (TODAS) | `Recompensa.cerca_torque` = −0,29615 (`knobs.py:274`) | `[Σ_soltas relu(\|τ\|/τmax − 0,8) + Σ_pares relu((\|τ_p\|+\|τ_r\|)/50 − 0,8)] / 0,2` | cerca | Torque ≤ 80 % do limite |
| 23 | `cerca_pelve` (TODAS) | `Recompensa.cerca_pelve` = −4,45279 (`knobs.py:275`) | `relu(\|h − pelve_com_caixa(v3_meia)\| − 0,08)`, com `Recompensa.lim_pelve` = 0,08 (`knobs.py:283`) | cerca | Dentro de 0,08 m |
| 24 | `cerca_recuo` (PEGAR, CARREGAR, BOTAR) | `Recompensa.cerca_recuo` = −10,39719 (`knobs.py:276`) | `relu(\|recuo − ‖caixa_transporte_b(a)[:2]‖\| − 0,08)`, com `Recompensa.lim_recuo` = 0,08 (`knobs.py:284`); `recuo = ‖xy caixa − xy pelve‖` | nenhum | Dentro de 0,08 m |
| 25 | `contato_fora` (TODAS) | `Recompensa.contato_fora` = −1,0 (`knobs.py:299`) | Nº de corpos (fora os tornozelos) com força de contato com o terreno > `Recompensa.contato_fora_N` = 1,0 N (`knobs.py:300`) + corpos em autocolisão (`recompensas.py:805-813`) | nenhum | Sem contato fora dos pés. Em CARREGAR o contato robô–caixa não conta |
| 26 | `freio` (PEGAR, CARREGAR, BOTAR) | `Recompensa.freio` = −6/29 ≈ −0,2069 (`knobs.py:304`) | `Σ_j relu(\|q̇_j\| / vmax_j − 1)²` | `Recompensa.freio_fator` = 1,5 (`knobs.py:305`) elevado a `v3_freio`; rampa `min(s / 0,75; 1)` | Velocidade de junta ≤ `vmax` |
| 27 | `progresso` (TODAS) | 1,0 (`env_cfg.py:57`) | `6 · etapa · h + 3 · Φ + 2 · conquistas`; em CARREGAR `etapa` = 4, Φ pago = 0 | nenhum | Ver §7.4 |
| 28 | `sacudida` (PEGAR, CARREGAR, BOTAR) | `Recompensa.sacudida` = −0,05 (`knobs.py:353`) | `Σ (Δv/(dt·5,04))² + Σ (Δω/(dt·31,56))²`, da caixa | `d / 5`; só com alvo P3 ou B1 | Degrau 0; `v3_alvo_idx` fora de P3 e B1 |
| 29 | `aperto` (PEGAR, CARREGAR, BOTAR) | `Recompensa.aperto` = −0,05 (`knobs.py:355`) | `min(F_esq, F_dir) / 49,0` | `d / 5`; só com alvo P3 ou B1 | Degrau 0; alvo fora de P3 e B1; sem força nas duas faces ±y |
| 30 | `perda_caixa` (TODAS) | −1,0 (`env_cfg.py:57`) | `v3_perdeu_agora · 240,0` | nenhum | Sem pulso de perda |
| 31 | `terminacao` (TODAS) | `Recompensa.terminacao` = −300,0 (`knobs.py:366`) | `1` no passo de uma terminação que não é `time_out` | nenhum | Sem terminação, ou `time_out` |

Os termos `cerca_sola` e `contato_laje` (tupla `PEGAR_BOTAR`) não valem em CARREGAR (`env_cfg.py:48`, `:51`).

Notas da tabela:
- `h_pelve` é a altura da pelve sobre o terreno, do sensor `altura_pelve` (`recompensas.py:731-733`). `h` do `altura_pe` é a folga do pé do sensor `altura_pes`, menos o desconto da §7.3 (`recompensas.py:651`).
- `q_com_caixa(v3_meia)` é a pose de braços, cintura e quadril com a caixa, interpolada pela meia aresta (`recompensas.py:412-416`, `alvos.py:84-95`). Não determinado pelo código lido: os valores (arquivo `referencias.npz`). O alvo dos desvios em CARREGAR é essa pose; em ANDAR é `q_de_pe`.
- `cerca_recuo` em CARREGAR: `_recuo_ref` devolve a norma xy de `caixa_transporte_b(a)` (`recompensas.py:509-516`). A exceção `solta` vale só para BOTAR (`recompensas.py:801`).
- `cerca_com`: o CoM inclui a caixa quando `v3_pegou & ~v3_perdeu` e o alvo está em P3, B1 ou B2 (`recompensas.py:419-435`). Em CARREGAR o índice é P3 e `v3_pegou` é verdadeiro.
- `perda_caixa`: o custo efetivo por pulso é `240,0 × 0,02` = 4,8. A cobrança sai no passo seguinte à detecção; nesse passo a tarefa já é ANDAR (§8.3).
- `terminacao`: o custo efetivo é `300,0 × 0,02` = 6,0.
- `freio`: `v3_freio` = nível do CARREGAR + 1, gravado na abertura do trecho (`comando.py:515`, `balanceador.py:335-336`). O regime sai do total do comando `‖c_xy‖ + |c_z|` (`recompensas.py:905-909`), com `freio_limiar_andar` = 0,05 e `freio_limiar_correr` = 1,5 (`knobs.py:308-309`): parado se total < 0,05, andando se 0,05 ≤ total < 1,5, correndo se total ≥ 1,5. Em CARREGAR o total fica abaixo de `Comando.frente_tetos[4]` + `Comando.giro_tetos[3]` = 1,0 + 0,5 = 1,5, porque os dois sorteios usam U[0; 1), que não chega ao teto (`comando.py:107`, `:115`). Logo o regime correndo não ocorre em CARREGAR. O teto de cada junta sai da tabela do regime (`knobs.py:311-331`). A rampa usa `s_t` = média de `s` das células abertas da tarefa CARREGAR, dividida por `Balanceador.portao` = 0,75, com teto 1 (`recompensas.py:916-918`).

### 7.3 Fórmulas completas dos termos principais

#### Rastreio (`recompensas.py:374-409`, `:590-604`)

Sejam `c` = comando no frame da base, `v̄` = velocidade da base filtrada pelo `twist` (`v_filt`) e `σ` = `Recompensa.sigma_rastreio` = 0,5 (`knobs.py:238`).

Filtro `v_filt` (`comando.py:854-858`): a cada passo, `v_filt += (1 − exp(−dt/τ)) · (v − v_filt)`, com `τ` = `Comando.tau_fracao` = 0,4 s (`knobs.py:136`). Isso dá o coeficiente 0,04877 por passo. O filtro zera no reset (`comando.py:886`).

Fator de marcha `g` (`comando.py:864-871`):
- `acerto` = média, sobre os dois pés, de `[contato == apoio]` (`n_passo`).
- `q̄` filtra o `acerto` com o mesmo `τ` = 0,4 s. `q̄` vale 0,5 no reset, e volta a 0,5 no passo em que `on` liga depois de desligado.
- `g = clip(2 · q̄ − 1, 0, 1)`.

Linear, canal com comando (`‖c_xy‖ ≥ Comando.limiar_canal` = 0,02, `knobs.py:138`):
- `den = max(‖c_xy‖, 0,02)`.
- `proj = (v̄_xy · c_xy) / den`.
- `perp² = max(‖v_xy‖² − (v_xy · c_xy / den)², 0)`, com `v_xy` a velocidade instantânea da base (`recompensas.py:404`). A projeção `proj` segue na `v̄` filtrada.
- `tenda = clip(1 − |1 − proj/den|, 0, 1) · exp(−perp² / σ²)`.
- `r_lin = tenda · g`.

Linear, canal sem comando: `r_lin = exp(−‖c_xy − v_xy‖² / σ²)`, com `v_xy` instantânea.

Giro, canal com comando (`|c_z| ≥ 0,02`): `proj = v̄_z · sinal(c_z)`, `den = max(|c_z|, 0,02)`, `r_ang = clip(1 − |1 − proj/den|, 0, 1) · g`. Sem comando: `r_ang = exp(−(c_z − wz)² / σ²)`.

Qualidade (`recompensas.py:299-308`, `:572-594`): em CARREGAR o termo vale `q · relu(r) + min(r, 0)`. Todo `r` acima é ≥ 0, então o termo vale `q · r`.

`q = exp(−(d / σ_t)²) · clip(1 − (incl − incl_lim) / 15, 0, 1) · (nº de palmas em contato) / 2`, onde:
- `d` = `‖caixa − alvo‖`. Fora da célula `terreno`, `alvo` = `base_xy + R(yaw) · caixa_transporte_b(a)[:2]` e z = `chão local + Alvos.z_transporte` = 0,90 m (`alvos.py:163-179`, `knobs.py:202`). Na célula `terreno`, `alvo` = pelve + `caixa_transporte_b(a)` no frame da pelve, com o z da tabela (`recompensas.py:572-585`).
- `σ_t` = `tol_transporte_p3` efetivo da zona = 0,05 + `z[3]` · 0,15 m (§5.6).
- `incl` é a inclinação da caixa em graus (`alvos.py:131-135`). `incl_lim` = 30 − 3·`d_CARREGAR` graus (§5.5). O denominador 15 é `Alvos.delta_incl_graus` = 15,0 (`knobs.py:205`).
- O número de palmas em contato vem dos sensores `palma_E` e `palma_D` (`recompensas.py:482-484`).

Não determinado pelo código lido: o valor de `caixa_transporte_b` (arquivo `referencias.npz`).

#### Passo (`recompensas.py:75-82`, `:619-633`)

- `on` = `‖comando‖ > Comando.limiar_marcha` = 0,1 (`knobs.py:141`). A norma usa as três componentes.
- Fase do relógio: `v3_fase_marcha` = `(tempo com on / Comando.periodo_marcha) mod 1`, com `periodo_marcha` = 0,8 s (`knobs.py:140`, `comando.py:872-873`). O relógio só avança com `on`.
- Pé esquerdo na fase `φ`. Pé direito em `(φ + 0,5) mod 1`.
- `apoio` = `(fase_do_pé < Comando.fase_apoio` = 0,55`)` ou `~on` (`knobs.py:139`).
- O termo vale `Σ_pés [contato == apoio]`. Com o relógio parado, cada pé em contato vale 1. O total vale × `_avanco` (abaixo, `recompensas.py:633`).

#### Avanço, fator do `passo` e do `altura_pe` (`recompensas.py:607-616`)

- `lin` = `‖c_xy‖ ≥ 0,02`. `ang` = `|c_z| ≥ 0,02`.
- `p` = o rastreio linear (tenda × `g_marcha`, sem a qualidade da caixa) se `lin`; senão o rastreio do giro se `ang`; senão 1 (`recompensas.py:614`).
- `piso` = `no_degrau(1,0; Recompensa.piso_avanco; nível do ANDAR; máximo do ANDAR)` = 1 − 0,75 · nível / 7, com `piso_avanco` = 0,25 (`knobs.py:245`, `recompensas.py:615`).
- Fator = `piso + (1 − piso) · p`.
- O piso segue o nível do ANDAR, e não o do CARREGAR. Em CARREGAR o nível do ANDAR vale pelo menos 5, porque a fatia `B` abre nesse nível e o nível não desce (§5.3). O piso vale 0,464 no nível 5, 0,357 no 6 e 0,25 no 7 (derivado: 1 − 0,75 · n / 7).
- Fatia `C`: comando zero, `p` = 1 e o fator vale 1. Fatia `B` com comando: marchar no lugar paga o piso, e andar o comando inteiro paga 1.

#### Altura do pé (`recompensas.py:90-97`, `:641-656`)

`exp(−Σ_pés (h − 0,12)² · tanh(2,0 · ‖v_xy‖) / 0,0025) × _avanco`, com `Recompensa.alvo_pe` = 0,12 m, `tanh_pe` = 2,0, `std_pe` = 0,0025 m² (`knobs.py:250-252`). `‖v_xy‖` é a velocidade horizontal do tornozelo. O termo vale 0 com comando exatamente zero (`recompensas.py:654-656`).

O `h` é a folga do pé do sensor `altura_pes` menos o desconto da subida do quadril pela inclinação da pelve (`recompensas.py:649-651`). Por perna: `desconto` = z no mundo do vetor pelve→quadril − z do mesmo vetor no frame da pelve. O desconto vale 0 com a pelve nivelada. Quando o tronco inclina para erguer o pé, o quadril sobe, e o desconto tira essa subida de `h`.

#### Limite de junta (`potencial.py:70-85`, `recompensas.py:700-704`)

`Σ_j relu(|q_j − c_j| / m_j − 0,95) / (1 − 0,95)`, com `Recompensa.limite_mole` = 0,95 (`knobs.py:263`). `c` e `m` são o centro e a meia faixa do limite da junta no modelo.

#### Cerca de torque (`recompensas.py:182-193`, `:767-772`)

Com `f = |τ| / τmax` por junta e `Recompensa.lim_torque` = 0,8, `banda_torque` = 0,2, `tau_par` = 50,0 (`knobs.py:281,282,298`):
- Juntas soltas: `Σ relu(f − 0,8)`. As juntas de um par ficam fora dessa soma.
- Pares (`pares_torque`, `knobs.py:293-297`: `waist_pitch` com `waist_roll`, e os dois tornozelos): `Σ_pares relu((|τ_p| + |τ_r|) / 50 − 0,8)`.
- Total dividido por 0,2.
- `τmax` por junta vem de `Recompensa.tau_max` (`knobs.py:286-292`): joelho e `hip_roll` 139; `hip_pitch`, `hip_yaw` e `waist_yaw` 88; `waist_roll`, `waist_pitch` e tornozelos 50; ombros, cotovelos e `wrist_roll` 25; `wrist_pitch` e `wrist_yaw` 5 N·m.

#### Cerca do CoM (`recompensas.py:176-179`, `:316-336`)

`relu(0,03 − m)` se os dois pés estão em contato (`pes_chao.found > 0` nos dois). `m` = distância com sinal do CoM combinado (robô + caixa presa) às arestas do fecho convexo dos 8 cantos das solas, positiva dentro. `Recompensa.lim_com` = 0,03 m (`knobs.py:280`).

#### Sacudida (`recompensas.py:339-342`, `:922-945`)

`custo = Σ_x (Δv_x / (dt · 5,04))² + Σ_x (Δω_x / (dt · 31,56))²`, com `Δ` = diferença entre a velocidade da caixa neste passo e no passo anterior (`Recompensa.sacudida_norma` = (5,04; 31,56), `knobs.py:358`). O termo vale `custo × d/5` quando `v3_alvo_idx` está em P3 ou B1. Em CARREGAR o índice é P3.

#### Aperto (`potencial.py:55-62`, `recompensas.py:948-957`)

`min(F_esq, F_dir) / Recompensa.F_ref` com `F_ref` = 49,0 N (`knobs.py:359`), vezes `d/5`. `F_esq` e `F_dir` são as somas das forças normais nas faces +y e −y da caixa, do sensor `caixa_faces` (`recompensas.py:345-356`).

#### Perda da caixa e terminação (`recompensas.py:311-313`, `:214-216`, `:960-968`)

- `perda_caixa` = `v3_perdeu_agora · Recompensa.perda_caixa` = 240,0, com peso −1,0.
- `terminacao` = `(terminado & ~time_out)`, com peso −300,0.
- Efeito extra sobre a recompensa do PPO (`algoritmo.py:114-135`): a partir do segundo rollout (quando `v_grupo` já existe, `algoritmo.py:127`, `:169`), numa terminação que não é `time_out` (`dones > 0` sem `time_outs`) e no passo da perda da caixa (`extras["v3_perdeu_agora"]`), a recompensa ganha `γ · min(V̄_g, 0)`. `γ` = 0,99 (`RslRlPpoAlgorithmCfg.gamma` em `mjlab/tasks/velocity/config/g1/rl_cfg.py:37`; `rl_cfg()` não o troca, `tarefa.py:45-50`). `V̄_g` é a média do valor do grupo da vantagem no rollout anterior (`algoritmo.py:160-169`). O grupo do CARREGAR com a pega fechada é `1 + tarefa · n + slot` = 1 + 2 × 11 + 7 = 30, com `n` = 11 canais do `alvo_macro` (`observacoes.py:46`) e `slot` = 7 (`algoritmo.py:98-112`, `observacoes.py:282-295`).

### 7.4 Fórmula completa do `progresso` em CARREGAR

Código: `recompensas.py:816-884`.

Estado do env em CARREGAR (sem caixa perdida): `v3_pega_fechada` = verdadeiro, `v3_bota_fechada` = falso, `v3_alvo_idx` = P3 (índice 3).

Resultado das condições (`recompensas.py:851-857`):
- `sem_caixa` = `v3_perdeu | (~v3_pega_fechada & tarefa ≠ PEGAR)` = falso.
- `fecho_pega` = `v3_pega_fechada & (idx < B1)` = verdadeiro.
- `etapa` = 4 (`fecho_pega`).
- `paga_phi` = falso. O Φ pago vale 0.
- `carrega` = `fecho_pega & ~sem_caixa & (tarefa == CARREGAR)` = verdadeiro.
- `com_alvo` = `paga_phi & idx ≥ 0` = falso.

Renda por passo (`recompensas.py:286-291`, `:879-882`):

`r = renda_etapa · 4 · h + renda_phi · 0 + renda_conquista · (c₀ + c₁ + c₂)`

com `Recompensa.renda_etapa` = 6,0 (`knobs.py:339`), `renda_phi` = 3,0 (`knobs.py:338`), `renda_conquista` = 2,0 (`knobs.py:343`). O valor máximo é `6 · 4 · 1 + 2 · 3` = 30 por segundo, antes do fator `dt`.

O fator `h` (`v3_h`) em CARREGAR (`recompensas.py:856-870`):
- Na abertura do trecho `h` = 1,0 (`comando.py:583`).
- A cada passo: `h ← h + (dt / τ) · (sinal − h)`, com `τ` = `Recompensa.tau_renda` = 3,0 s (`knobs.py:347`). `dt` = 0,02 s.
- `h` fica em [0, 1].

O `sinal` (`recompensas.py:859-867`):
- `com_lin` = `‖c_xy‖ ≥ 0,02`. `com_ang` = `|c_z| ≥ 0,02`.
- `r_lin` e `r_ang` são o rastreio normalizado (`potencial.py:88-106`): se `c ≥ limiar_canal`: `(expm1(−e²/σ²) − expm1(−c²/σ²)) / (−expm1(−c²/σ²))`; se não: `exp(−v²/σ₀²) − 1`. O piso é `Recompensa.piso_rastreio` = −1,0 (`knobs.py:239`). `σ` = 0,5. `σ₀` do linear = `Recompensa.sigma_zero_lin` = 0,10 (`knobs.py:241`). `σ₀` do giro = `no_degrau(sigma_zero_ang_inicio` = 0,5; `sigma_zero_ang` = 0,15; `restricao`; 1) (`knobs.py:242-243`, `recompensas.py:394`). `e²` e `v²` usam a velocidade instantânea da base.
- Com comando: `sinal = g · (com_lin · clip(r_lin, 0, 1) + com_ang · clip(r_ang, 0, 1)) / max(com_lin + com_ang, 1)`.
- Sem comando: `sinal = clip((1 + r_lin) · (1 + r_ang), 0, 1)`.
- O `sinal` é multiplicado por `rampa(d_transporte; 0,05; 0,25)` = `clip(1 − (d − 0,05) / 0,25, 0, 1)`, com `Recompensa.qualidade_larga` = (0,05; 0,25) (`knobs.py:351`). `d_transporte` é a distância da caixa ao alvo de transporte (§7.3).

As conquistas (`recompensas.py:519-563`); todas em [0, 1]:
- `c₀ = 1 − tanh(d_faces / (Alvos.largura_tanh · Recompensa.sigma_palmas))`, com `largura_tanh` = 2,0 (`knobs.py:221`) e `sigma_palmas` = 0,08 m (`knobs.py:344`). `d_faces` = média das distâncias das duas palmas aos pontos-alvo do P2 no frame da caixa. Os pontos do P2 são `x` = 0, `y` = ±(`v3_meia` + `Alvos.palma_face` = 0,045 + 0), `z` = 0 (`alvos.py:138-154`, `knobs.py:178`).
- `c₁ = tanh(min(F_esq, F_dir) / F_pega)` com `F_pega = margem_pega · massa · g / (2 · μ)`. `Recompensa.margem_pega` = 2,5 (`knobs.py:365`), `μ` = `DR.atrito_pad[0]` = 0,6 (`knobs.py:523`), `g` = `−gravidade do modelo`.
- `c₂ = clip(1 − apoio_N / peso_N, 0, 1) · c₁`. `apoio_N` = Σ |Fz| do sensor `apoio_caixa` (caixa–laje). `peso_N` = `v3_massa · g`.
- A máscara `mantem` vale verdadeiro em CARREGAR (`recompensas.py:875`). O peso `c₂` entra porque `com_peso` = 1 com `fecho_pega` (`recompensas.py:877`). O fator `almofadas` vale 1 fora do P2 (`recompensas.py:876`).

`v3_phi` (para a métrica e para o crítico) grava o Φ cru de P3 mesmo em CARREGAR, porque `idx ≥ 0` (`recompensas.py:883`). O Φ de P3 é calculado por `_phi_env` (`recompensas.py:519-569`), mas não é pago. `v3_phi_melhor` não é atualizado em CARREGAR (`recompensas.py:869`).

### 7.5 O PPO que usa estas recompensas

- O rollout tem `num_steps_per_env` = `Balanceador.passos_por_iteracao` = 24 passos por env (`tarefa.py:49`, `knobs.py:402`).
- Hiperparâmetros do mjlab (`mjlab/tasks/velocity/config/g1/rl_cfg.py:11-44`): redes do ator e do crítico com camadas (512, 256, 128) e ativação `elu`; `init_std` 1,0; `clip_param` 0,2; `num_learning_epochs` 5; `num_mini_batches` 4; `learning_rate` 1e-3 com `schedule` "adaptive" e `desired_kl` 0,01; `gamma` 0,99; `lam` 0,95; `value_loss_coef` 1,0; `max_grad_norm` 1,0.
- A v3 troca `entropy_coef` para 0,001 (`tarefa.py:48`) e a classe do algoritmo para `PPOPorTarefa` (`tarefa.py:47`).
- A vantagem é normalizada por grupo. `compute_returns` chama o PPO do `rsl_rl` e refaz a normalização sobre a vantagem crua `returns − values` (`algoritmo.py:137-159`). Cada grupo com 2 ou mais amostras sai com média 0 e desvio 1: `(a − média) / (desvio + 1e-8)` (`algoritmo.py:58-59`). Um grupo com menos de 2 amostras: o grupo 0 (ANDAR sem alvo) fica com a vantagem crua; os outros usam a média e o desvio de todas as amostras fora do grupo 0 do lote, se elas são 2 ou mais (`algoritmo.py:46-57`). O grupo vem das 6 últimas colunas do crítico e do `alvo_macro` (§7.3). Não determinado pelo código lido: o número de envs, que define o tamanho do lote.

---

## 8. Terminações

Os termos estão em `env_cfg.py:137-147`. `queda`, `caixa_perdida`, `chegada` e `batente` usam a máscara de tarefa com `torch.where` (`terminacoes.py:112`, `:125`, `:151`, `:183`). `time_out` não usa a máscara.

### 8.1 `time_out` (`fim_da_cadeia`)

- Condição: `episode_length_buf >= max_episode_length` ou `v3_fim` (`terminacoes.py:53-55`). O termo também dispara na travessia do terreno, mas só na fatia `ANDAR` com `terrain_types != 0` (`terminacoes.py:70-72`). O CARREGAR não sofre esse disparo.
- `max_episode_length` = `ceil(Cadeia.episodio_s / step_dt)` = `ceil(20,0 / 0,02)` = 1000 (`knobs.py:160`, `mjlab/envs/manager_based_rl_env.py:283`).
- `v3_fim` sobe no fim do último trecho da fatia (§1.5).
- Não passa pela máscara de tarefa. Vale em toda tarefa.
- O termo é `time_out=True` (`env_cfg.py:138`). Ele não conta como queda para a recompensa `terminacao`.

### 8.2 `queda` (CARREGAR é do grupo de marcha)

Condição (`terminacoes.py:35-50`, `:86-112`), com `Terminacao` de `knobs.py:374-383`:
- `incl_rad > Terminacao.incl_marcha` = 0,8 rad, com `incl = acos(clip(−projected_gravity_b[z], −1, 1))`; ou
- `z_pelve − z_chao_pelve < Terminacao.base_min` = 0,20 m (altura da pelve sobre o terreno, do sensor `altura_pelve`).

A condição de manipulação (70°, joelho a 0,05 m) vale só para PEGAR e BOTAR (`terminacoes.py:30-31`, `:47-50`).

### 8.3 `caixa_perdida` e a perda da caixa

O termo de terminação `caixa_perdida` vale `v3_perdeu_agora & (not Cadeia.caixa_perdida_vira_andar)` (`terminacoes.py:75-83`). Como `caixa_perdida_vira_andar` = True (`knobs.py:161`), o termo nunca termina o episódio.

A perda é detectada em `CadeiaCmd._perda` (`comando.py:394-420`). Condição, para `tarefa ≠ ANDAR` e `episode_length_buf > 0` (`comando.py:409-410`):
- `caiu`: `h_caixa − v3_meia < Terminacao.caixa_folga_chao` = 0,02 m (`knobs.py:379`, `comando.py:61-62`). `h_caixa` é a altura do centro da caixa sobre o terreno, do sensor `altura_caixa`.
- `arremesso`: `v3_pegou & (‖v_caixa − v_base‖ > Terminacao.v_solta` = 1,2 m/s`) & fora_do_alvo` (`knobs.py:380`, `comando.py:63`). Em CARREGAR `fora_do_alvo` vale sempre verdadeiro (`comando.py:408`).
- `fora_da_laje`: alvo em P0 a P2 e caixa fora do topo da laje. Não vale em CARREGAR, porque o índice é P3 (`comando.py:64`).

O que acontece na perda em CARREGAR (`comando.py:411-420`):
1. `v3_perdeu_agora` vira verdadeiro neste passo (um pulso). A recompensa `perda_caixa` o lê no passo seguinte.
2. `v3_perdeu` = verdadeiro. `v3_alvo_idx` = B3. `v3_phi` = −1, `v3_h` = 1, `v3_phi_melhor` = 0, `v3_relogio_alvo` = 0. `v3_sigma` recebe a distância do B3.
3. `_publica` grava `v3_tarefa` = ANDAR e reamostra o comando. O comando é zero (a fatia é `B` ou `C`, e a condição de `anda` exige CARREGAR na fatia `B`).
4. O trecho congela: `cong = v3_perdeu` impede o avanço e o fim (`comando.py:610-613`). O episódio vai até `time_out` por tempo (1000 passos), ou até uma `queda` ou um `batente`.
5. O segmento de comando do CARREGAR fecha como falha no mesmo passo (`falha = v3_perdeu & seg_carrega`, `comando.py:820`). A nota da célula vale 0 e ele registra mesmo antes da âncora (`comando.py:789-790`).

Depois da perda, a tarefa é ANDAR e o env se comporta assim nas recompensas:
- Os termos com `paga=True` (`rastreio_linear`, `rastreio_angular`, `passo`, `altura_pe`) e o `vivo` ficam em zero, porque `v3_perdeu` é verdadeiro (`recompensas.py:363-364`, `:85-87`).
- Os termos só de `PEGAR_CARREGAR_BOTAR` (`cerca_recuo`, `freio`, `sacudida`, `aperto`) valem zero, porque a tarefa não está na tupla (`env_cfg.py:50`, `:51-52`).
- Os termos das tuplas `ANDAR_CARREGAR` e `TODAS` sem `paga` continuam a valer, com a regra do ANDAR: por exemplo, os `desvio_*` passam a apontar para `q_de_pe` (`recompensas.py:412-416`) e a `cerca_pelve` para `pelve_de_pe` (`recompensas.py:783-784`).
- O `progresso` vale `etapa` = 0 e Φ pago = 0, porque `sem_caixa` é verdadeiro (`recompensas.py:851`, `:853-854`).
- A `perda_caixa` cobra o pulso, e a `terminacao` cobra a queda ou o batente que vierem depois.

### 8.4 `batente`

Condição (`terminacoes.py:154-164`, `:167-183`): `degrau da tarefa ≥ Refino.degrau_batente` = 1 (`knobs.py:547`) e (`alguma das 29 juntas` com `|q − c| / m > Terminacao.batente_frac` = 1,0, ou `|hip_yaw|` de alguma perna `> Terminacao.hip_yaw_max` = 1,0 rad) (`knobs.py:382-383`). `c` e `m` são o centro e a meia faixa do limite duro da junta no modelo. A comparação é estrita (`>`).

`v3_degrau` do env é o degrau do CARREGAR, gravado na abertura do trecho (§1.4). Depois da perda da caixa, `_publica` regrava `v3_degrau` com o degrau do ANDAR (`comando.py:516-518`, `:420`), e o `batente` passa a usar esse valor.

### 8.5 `chegada`

A terminação `chegada` vale só para BOTAR (`env_cfg.py:144`). Não vale em CARREGAR.

### 8.6 O que acontece depois de uma terminação

- O env reinicia no mesmo passo (§2.1).
- O `terminacao` paga −300,0 × 0,02 = −6,0 no passo da terminação, se não for `time_out` (§7.2).
- O PPO soma `γ · min(V̄_g, 0)` à recompensa desse passo (§7.3).
- Num `time_out` (por `v3_fim` ou por 1000 passos), o `rsl_rl` soma `γ × V(s)` à recompensa do passo, com `γ` = 0,99 e `V(s)` = `transition.values`, o valor que o crítico deu ao estado em que o passo começou (`rsl_rl/algorithms/ppo.py:150-155`).
- O segmento de comando e o PEGAR pendente registram falha (§5.2 e §5.8).
- `time_out` não gera falha: o segmento registra com o sucesso medido.

---

## 9. O que o ator e o crítico observam, próprio dessa tarefa

### 9.1 Ator

Os termos do ator, em ordem (`observacoes.py:389-400`, `knobs.py:58-63`): `giro_base`, `gravidade`, `comando`, `fase`, `juntas`, `vel_juntas`, `acao`, `tarefa`, `caixa`, `mapa`. Todos têm histórico de 5 passos, menos `mapa` (histórico 0). Total: 1457 números (`knobs.py:58`).

O que muda em CARREGAR:

| Termo | Valor em CARREGAR | Referência |
|---|---|---|
| `tarefa` | One-hot de 6 canais, com o canal 2 em 1. Sem ruído | `observacoes.py:135-137` |
| `comando` | `vel_command_b` do `twist`: o comando da §3. Fatia `B`: fixo no trecho. Fatia `C`: zero | `observacoes.py:392`, `comando.py:889-914` |
| `fase` | `(sen, cos)` de `2π · v3_fase_marcha`. A fase só avança com `‖comando‖ > 0,1` | `observacoes.py:298-314`, `comando.py:863,872-862` |
| `caixa` (11 canais) | Os 8 primeiros são a saída do estimador no frame da base: posição (3), rotvec (3), meia aresta (1), idade da leitura (1). Os 3 últimos são o alvo no frame da base e valem zero em CARREGAR | `observacoes.py:172-199` |

O alvo da `caixa` zera quando `tarefa == CARREGAR`, ou, para `tarefa == BOTAR`, quando o latch `v3_aviso` é verdadeiro, ou, para as outras tarefas, quando `v3_alvo_idx < 0` (`observacoes.py:196`). Em ANDAR todos os 11 canais zeram (`observacoes.py:199`). Depois da perda da caixa a tarefa é ANDAR, então os 11 canais zeram e o ator deixa de ver a caixa. O estimador entrega a leitura com ruído, perda e latência (§6.4). Sem leitura nova, a caixa presa às palmas anda com o ponto médio das palmas (`estimador.py:353-371`).

A escala e o histórico vêm de `Contrato.blocos_ator` (`knobs.py:58-63`): `giro_base` 0,2; `vel_juntas` 0,05; os outros 1,0.

Depois dessa escala, o normalizador de observação do PPO (`obs_normalization` = True, `rl_cfg.py:16`, `:26`) atua sobre o vetor do ator. Três termos do ator têm média 0 e desvio fixos: `tarefa` 1,0, `mapa` 1,0 e `caixa` 0,1 (`runner.py:32-34`). Os outros termos usam a estatística empírica, com piso de 0,05 no desvio (`algoritmo.py:63-83`). O `fase` não está em `_FIXOS` e passa pelo normalizador empírico (`runner.py:32-34`).

O `mapa` (882 canais) existe com `Percepcao.ligada` = True (`knobs.py:483`). Ele não depende da tarefa.

### 9.2 Crítico

O crítico usa os mesmos termos do ator, sem ruído e sem atraso, e acrescenta termos privilegiados (`observacoes.py:404-418`). Ele lê as `juntas` sem o viés do encoder (`observacoes.py:404`). No crítico, os termos `tarefa`, `mapa`, `alvo_macro`, `niveis`, `degrau`, `massa`, `tarefa_critico`, `privilegiados` (desvio 1,0), `caixa`, `caixa_verdade`, `alvo_verdade`, `referencia` (desvio 0,1) e `forca_caixa` (desvio 0,3) têm média 0 e desvio fixo no normalizador (`runner.py:32-34`). Termos que dependem da tarefa:

| Termo | Valor em CARREGAR sem perda | Referência |
|---|---|---|
| `tarefa_critico` | One-hot, último termo. O PPO lê as 6 últimas colunas para agrupar a vantagem | `observacoes.py:417`, `algoritmo.py:98-112`, `:149-153` |
| `alvo_macro` | One-hot de 11 canais. Em CARREGAR com pega fechada vale o canal 7 ("pega fechada"). Com a caixa perdida vale o canal 9 | `observacoes.py:282-295` |
| `referencia` | `ref_do_alvo(v3_alvo_idx = P3)`: palmas = 0, pelve = `pelve_com_caixa(a)`, tronco = 0 | `observacoes.py:116-130`, `:321-324`, `recompensas.py:501-506` |
| `privilegiados` | Quatro canais: `v3_marcha_on`, `v3_h`, `lacuna = relu(v3_phi_melhor − max(v3_phi, 0))`, `z` da etapa corrente (o `z[3]` do P3 em CARREGAR) | `observacoes.py:298-318` |
| `caixa_verdade` | Posição (3), rotvec (3) e meia aresta (1) da caixa verdadeira no frame da base. Zero em ANDAR | `observacoes.py:267-273` |
| `alvo_verdade` | `v3_alvo_w` no frame da base, se `v3_alvo_idx ≥ 0`. Em CARREGAR o índice é P3 (≥ 0). `v3_alvo_w` guarda a posição estimada da caixa na abertura do P0 e nada o muda depois (`comando.py:390-391`). O valor é o desse ponto fixo | `observacoes.py:276-279` |
| `niveis` | Nível / máximo das quatro tarefas, igual em todos os envs | `observacoes.py:327-335` |
| `degrau` | `v3_degrau / 5` | `observacoes.py:338-340` |
| `massa` | `v3_massa / 5` | `observacoes.py:343-345` |
| `forca_caixa` | Σ dos módulos das 12 fatias de força do sensor `caixa_robo`, em N, dividida por 100 | `observacoes.py:348-356`, `metricas.py:338-345` |
| `vel_lin_base` | Velocidade linear da base (sensor `imu_lin_vel`), histórico 5 | `observacoes.py:406-407` |

O `v3_alvo_w` do `alvo_verdade` é um ponto fixo no mundo. Ele não é o ponto de transporte. Esse é um fato do código, e o documento não o julga.

---

## 10. Métricas que o treino registra para essa tarefa

As métricas dos gerenciadores valem para todos os envs, e não só para CARREGAR. A coluna "Condição" diz em que passos cada uma acumula valor em CARREGAR.

### 10.1 Currículo (`curriculo.py:103-143`)

O termo `v3` devolve um dicionário, que o mjlab publica como `Curriculum/v3/<chave>` (`mjlab/managers/curriculum_manager.py:103`).

| Chave | Valor |
|---|---|
| `frac_ANDAR`, `frac_B`, `frac_C` | Frações das fatias (§1.3) |
| `s_CARREGAR_reto`, `s_CARREGAR_giro_lugar`, `s_CARREGAR_arco`, `s_CARREGAR_terreno` | Nota `s` de cada célula (§5.2) |
| `nivel_CARREGAR` | Nível do CARREGAR (0 a 4) |
| `nivel_terreno_CARREGAR` | Linha de terreno do CARREGAR (0 a 9) |
| `nivel_terreno_ANDAR` | Piso da média das linhas do ANDAR (0 a 12). Ele abre o `CARREGAR/terreno` a partir de 1 (§5.3) |
| `linha_<coluna>` | Linha de terreno do ANDAR em cada coluna de 1 a 8 (`curriculo.py:139-140`) |
| `degrau_CARREGAR` | Degrau de refino do CARREGAR (0 a 5) |
| `degrau_global` | Menor degrau das quatro tarefas |
| `z_P0` a `z_B3`, `taxa_P0` a `taxa_B3` | Zona de cada etapa e a última taxa de fecho por visita (`z_P3` é a que o CARREGAR usa) |
| `atraso_max`, `mu_min_pad`, `restricao` | Estado do DR e da restrição |
| `subnivel_PEGAR`, `subnivel_BOTAR` | Subníveis do PEGAR e do BOTAR |

### 10.2 Termos de comando (`comando.py`, `mjlab/managers/command_manager.py:241-247`)

Publicados como `Metrics/<termo>/<métrica>`, média sobre os envs que resetam.

| Termo | Métrica | Definição |
|---|---|---|
| `twist` | `error_vel_xy`, `error_vel_yaw` | `‖c_xy − v_xy‖` e `|c_z − wz|`, divididos por `reamostra_s[1] / step_dt` e somados por passo (`mjlab/tasks/velocity/mdp/velocity_command.py:61-73`) |
| `twist` | `razao_marcha` | `1 − Σ erro / Σ comando` nos passos com `‖c_xy‖ > 0,02` (`comando.py:845-850`) |
| `twist` | `acerto_relogio` | `Σ acerto · on / máx(Σ on, 1)` (`comando.py:875-877`) |
| `twist` | `eficiencia_min`, `eficiencia_media`, `segmentos` | Sobre os segmentos com pedido ≥ `Comando.pedido_min_segmento` = 0,5 (`knobs.py:147`): eficiência = `Σ(v · ĉ) / Σ‖c‖` (`comando.py:795-804`) |

### 10.3 Métricas do `metricas.termos` (`metricas.py:522-571`)

O mjlab as publica como `Episode_Metrics/<nome>` (`mjlab/managers/metrics_manager.py:116-126`). As métricas `phi_<alvo>`, `tempo_<alvo>`, `erro_p3`, `velocidade_de_chegada`, `deslocamento_depois_do_apoio`, `fracao_fora_da_plataforma` e `aceleracao_da_caixa` também logam `Episodio/<nome>` uma vez por episódio fechado, no `reset` delas (`metricas.py:81-90`).

| Métrica | Condição em CARREGAR | Referência |
|---|---|---|
| `tempo_de_voo`, `pico_de_altura`, `velocidade_de_escorrego`, `velocidade_de_junta`, `fracao_do_curso` (reduce `max`), `forca_de_pouso`, `passo_medio` (reduce `last`), `fracao_mapa_visto` | Valem em todos os passos. O `passo_medio` só conta toques de pé com `‖c_xy‖ ≥ Comando.limiar_canal` = 0,02 | `metricas.py:93-233`, `:197-213` |
| `phi_P3` e `tempo_P3` | Acumulam nos passos com `v3_alvo_idx == P3` e a tarefa com caixa. Em CARREGAR o índice é P3, então CARREGAR conta. Publicam `Episodio/phi_P3` e `Episodio/tempo_P3` | `metricas.py:245-294` |
| `erro_p3` | Idem: média no episódio do erro vertical (alvo menos caixa, com sinal), lateral (valor absoluto, no eixo y da base) e da pelve (valor absoluto de `h − pelve_com_caixa`), em m (`Episodio/erro_p3_vertical`, `_lateral`, `_pelve`) | `metricas.py:297-326` |
| `inclinacao_caixa_p3` (reduce `max`) | Pico da inclinação da caixa (rad) nos passos com `v3_alvo_idx == P3` | `metricas.py:329-335` |
| `forca_robo_caixa` | Σ dos módulos das 12 fatias do `caixa_robo`, em N | `metricas.py:338-345` |
| `aceleracao_da_caixa` (reduce `last`) | Média do custo de sacudida nos passos com alvo em P3 ou B1 (`Episodio/aceleracao_da_caixa`). Tem `v_ant` própria | `metricas.py:459-491` |
| `fracao_fora_da_plataforma` (reduce `last`) | Específica do CARREGAR: acumula nos passos com `tarefa == CARREGAR`, `fatia == B` e `terrain_types > 0`. A fração dos passos com a pelve sobre terreno a mais de 0,01 m da altura da origem (`Episodio/fracao_fora_da_plataforma`) | `metricas.py:427-456` |
| `idade_leituras` | Idade média da leitura do estimador, zero onde a tarefa não tem caixa | `metricas.py:494-507` |
| `fracao_avisos` | Fração dos passos com `v3_aviso` verdadeiro. Em CARREGAR o latch é falso (o trecho não é por aviso) | `metricas.py:510-519` |
| `phi_<alvo>` e `tempo_<alvo>` de P0, P1, P2, B1, B2 e B3 | Não acumulam em CARREGAR, porque o índice é P3 (`phi_P3` e `tempo_P3` estão acima) | `metricas.py:245-294`, `:557-560` |
| `deslocamento_caixa_laje` (reduce `max`) | Vale 0 em CARREGAR: só conta com alvo em P0 a P2 | `metricas.py:348-373` |
| `velocidade_de_chegada`, `deslocamento_depois_do_apoio` | Não acumulam em CARREGAR: o primeiro lê `v3_v_chegada` (vale −1 fora do BOTAR) e o segundo só conta com alvo em B2 ou B3 sem perda | `metricas.py:376-424` |

### 10.4 Recompensas e terminações no log

Os gerenciadores do mjlab publicam `Episode_Reward/<termo>` (soma do episódio dividida por `max_episode_length_s`, `mjlab/managers/reward_manager.py:107-110`) e `Episode_Termination/<termo>` (`mjlab/managers/termination_manager.py:95`).

---

## Divergências entre comentário e código

⚠ Divergência: a docstring de `make_env_cfg` diz que o `dr` do `k` "vale só para QUAIS eventos entram no cfg" e que as faixas dos eventos vêm do `Curriculo` (`env_cfg.py:84-87`). O código lê `k.dr` direto nas faixas dos eventos `atrito_pe`, `massa_tronco`, `vies_encoder`, `tamanho_caixa` (`k.cena`), `atraso`, `empurrao` e `forca_caixa` (`env_cfg.py:186-209`). Só `pd`, `forca_motor`, `solref`, `atrito_pad`, `atrito_laje`, `vies_imu`, `rfi` e a trava de `forca_caixa` leem o `Curriculo`.

⚠ Divergência: a docstring de `n_terminacao` diz "Peso −200 no knob" (`recompensas.py:215`). O knob `Recompensa.terminacao` vale −300,0 (`knobs.py:366`).

⚠ Divergência: a docstring do módulo `algoritmo.py` diz que o grupo da vantagem é a tarefa corrente (`algoritmo.py:14-16`). Com `fatia_alvo` definida pelo `RunnerV3`, o grupo é `1 + tarefa · n + slot` do `alvo_macro`, e só o ANDAR sem alvo ("nenhum") fica no grupo 0 (`algoritmo.py:98-112`, `runner.py:61-62`).

⚠ Divergência: o comentário de `Curriculo.__init__` diz que o ANDAR tem 10 níveis (`balanceador.py:71`). O código dá `round((1,0 − 0,3) / 0,1)` = 7 (`balanceador.py:72`, `knobs.py:122`, `:129`, `:134`).

⚠ Divergência: o comentário de `Curriculo._abre` diz que o CARREGAR abre a célula `terreno` "com o ANDAR já fora da linha 0" (`balanceador.py:220`). O código exige `nivel_terreno("ANDAR") >= 1`, ou seja, o piso da média das 8 linhas por coluna ≥ 1 (`balanceador.py:225`).

## Resumo dos pontos não determinados pelo código lido

- Fração dos passos de treino em CARREGAR.
- Valores de `referencias.npz`: `q_de_pe`, `pelve_de_pe`, `caixa_transporte_b`, `q_com_caixa`, `pelve_com_caixa`, `grade_recuo`, `cantos_sola`, `palmas_de_pe_b`.
- Número de degraus de cada pirâmide de escada e altura máxima da superfície da rampa, em metros.
- Como o ruído do lidar entra no mapa (`percepcao_lidar.py` fora das fontes lidas).
