# Tarefa ANDAR

Código descrito: `g1_v3` no commit `69e8b6b`.

Fontes lidas:
- `g1_v3/env_cfg.py`, `knobs.py`, `comando.py`, `eventos.py`, `balanceador.py`, `curriculo.py`, `recompensas.py`, `terminacoes.py`, `observacoes.py`, `cena.py`, `metricas.py`.
- `g1_v3/potencial.py`, `algoritmo.py`, `runner.py`, `tarefa.py`, `train.py`, `percepcao_lidar.py`.
- `g1_v3/estimador.py` (classe `Odometria`) e `g1_v3/alvos.py` (`Referencias`, `palmas_alvo_caixa`, `avanca`).
- `g1_v3/dados/referencias.npz` (os valores `pelve_de_pe`, `q_de_pe` e `cantos_sola`).
- `mjlab/envs/manager_based_rl_env.py`, `mjlab/envs/mdp/events.py`, `mjlab/entity/entity.py`.
- `mjlab/envs/mdp/dr/actuator.py`, `geom.py`, `body.py`, `joint.py`.
- `mjlab/managers/command_manager.py`, `event_manager.py`, `reward_manager.py`, `metrics_manager.py`.
- `mjlab/managers/observation_manager.py`, `curriculum_manager.py`, `termination_manager.py`.
- `mjlab/tasks/velocity/mdp/velocity_command.py`, `mjlab/tasks/velocity/config/g1/env_cfgs.py`, `rl_cfg.py`.
- `mjlab/terrains/config.py`, `terrain_generator.py`, `primitive_terrains.py`, `heightfield_terrains.py`, `terrain_entity.py`.
- `mjlab/utils/buffers/delay_buffer.py`, `mjlab/actuator/actuator.py`.

Convenções. O passo da política dura 0,02 s: `Contrato.dt_fisica` = 0,005 s (`knobs.py:54`) vezes `Contrato.decimacao` = 4 (`knobs.py:55`). Toda recompensa é multiplicada por esse passo (`reward_manager.py:128`). O pacote lê os números de `ATIVO`, o objeto `Knobs()` com os valores de `knobs.py` (`knobs.py:569`). Os números deste documento valem para esse objeto. Um número com "(derivado)" vem de uma conta sobre knobs.

Simulação. O env configura o MuJoCo com `Cena.njmax` = 800 (`knobs.py:107`), `Cena.nconmax` = 300 (`knobs.py:108`), `Cena.ccd_iterations` = 50 (`knobs.py:109`), `Cena.impratio` = 1,0 (`knobs.py:110`) e `Cena.cone` = "pyramidal" (`knobs.py:111`). O passo de física é `Contrato.dt_fisica` (`env_cfg.py:97-104`). A extensão da cena é `Cena.extent` = 2,5 (`knobs.py:112`, `env_cfg.py:97`).

Ação. A ação da política tem 29 valores, um por junta. O alvo de cada junta é a pose de pé mais `escala × a`: `Contrato.escala_acao` = 0,25 rad por unidade em todas as juntas, salvo `wrist_pitch` e `wrist_yaw`, que usam `Contrato.escala_acao_punho` = 0,075 (`knobs.py:52-53`, `knobs.py:589-592`, `env_cfg.py:108-109`). Um atuador de posição por linha de `Contrato.pd` segue o par (kp, kd) da linha: `hip_pitch`, `hip_yaw` e `hip_roll` (100; 2), `knee` (150; 4), `ankle_pitch` e `ankle_roll` (40; 2), `waist_yaw` (200; 5), `waist_roll` e `waist_pitch` (40; 5), ombro, cotovelo e `wrist_roll` (40; 1), `wrist_pitch` e `wrist_yaw` (40; 1) (`knobs.py:65-73`, `cena.py:176-199`).

---

## 1. O que é a tarefa e quando um env está nela

Termos usados daqui em diante:

- Env: um dos mundos de simulação que rodam em paralelo. Cada env tem um robô e uma cena.
- Tarefa: o rótulo que diz o que o env faz agora. Fica no buffer `v3_tarefa`. `TAREFAS` tem seis nomes e o ANDAR é o índice 0 (`knobs.py:42`, `comando.py:30`). O env publica só as quatro primeiras (`comando.py:513`).
- Fatia: o grupo de episódios que seguem a mesma lista de trechos. `FATIAS` = ANDAR, B, C (`knobs.py:43`). O env recebe a fatia no reset.
- Trecho: a parte do episódio que tem uma tarefa só e uma duração sorteada (`comando.py:163-194`).
- Cadeia: o termo de comando `cadeia`. Ele guarda o trecho corrente e publica a tarefa (`comando.py:282-287`).
- Twist: o termo de comando `twist`. Ele guarda o comando de velocidade (`comando.py:689`).

### 1.1 Quando o env está na tarefa ANDAR

O env está no ANDAR quando `v3_tarefa` vale 0. A tabela de trechos define quatro situações (`comando.py:175-180`).

| Situação | Fatia | Posição do trecho | Duração | Comando de velocidade |
|---|---|---|---|---|
| ANDAR pleno | ANDAR | único (`comando.py:176`) | `Cadeia.episodio_s` = 20,0 s, valor fixo (`knobs.py:160`) | sorteado (seção 3) |
| Espera antes do PEGAR | B | primeiro (`comando.py:177`) | U(`Cadeia.espera_s`) = U(0,5; 1,0) s (`knobs.py:154`) | zero |
| Espera antes do PEGAR | C | primeiro (`comando.py:178`) | U(0,5; 1,0) s | zero |
| ANDAR final | C | último (`comando.py:179`) | `Cadeia.andar_final_s` = 1,5 s, valor fixo (`knobs.py:157`) | zero |
| ANDAR depois da perda da caixa | B ou C | do instante da perda até o fim do episódio | o trecho congela (`comando.py:610-613`) | zero |

O comando é zero nas situações com caixa porque o twist só sorteia velocidade para "ANDAR na fatia ANDAR" e para "CARREGAR na fatia B" (`comando.py:905`).

A caixa perdida vira ANDAR só com `Cadeia.caixa_perdida_vira_andar` = True (`knobs.py:161`). Com essa chave, o env publica a tarefa ANDAR no passo da perda e marca `v3_perdeu` (`comando.py:412-420`). A perda vale quando a caixa cai, é arremessada ou sai da laje (`comando.py:59-65`). A condição completa fica fora deste documento.

Resumo do que a tarefa ANDAR significa no código: o robô mantém o equilíbrio e segue um comando de velocidade (vx, vy, wz) no frame da base. O comando entra no ator pelo termo `comando`.

### 1.2 Sorteio da fatia

A fatia é sorteada no reset, pelo evento `fatia` (`env_cfg.py:154`). A função `sorteia_fatia` chama `cur.sorteia_fatias(n, gerador)` (`eventos.py:161`). O sorteio é `torch.multinomial` com reposição sobre as frações das fatias (`balanceador.py:302-304`). O gerador é `v3_gerador`, um gerador de números aleatórios criado na GPU do env com a semente `cfg.seed` (0 se o cfg não tem semente) (`curriculo.py:62-64`).

As frações vêm de `Curriculo.fracoes_fatias` (`balanceador.py:282-293`):

1. Uma fatia está aberta quando alguma célula dela está aberta (`balanceador.py:287`). A fatia ANDAR abre no começo (`balanceador.py:26`).
2. O teto da fatia ANDAR é 1,0. O teto da fatia B ou C é `Balanceador.rampa_fatia_inicio` + `Balanceador.rampa_fatia_passo` × `_rampa[f]` = 0,05 + 0,02 × `_rampa[f]` (`knobs.py:397-398`, `balanceador.py:288`).
3. O código ordena as fatias abertas por teto crescente. A fatia j recebe `min(teto, resto / (n_abertas − j))`. O `resto` é 1,0 menos o que as fatias anteriores receberam (`balanceador.py:289-292`).
4. Fatia fechada recebe 0.

Com só a fatia ANDAR aberta, a fração do ANDAR é 1,0. Com a fatia B aberta e `_rampa[1]` = 0, a fatia B recebe 0,05 e o ANDAR recebe 0,95 (derivado de `balanceador.py:288-292`).

O contador `_rampa[f]` sobe 1 a cada `Balanceador.rampa_fatia_its` = 12 fechos de iteração depois da abertura da fatia, e nunca desce (`knobs.py:399`, `balanceador.py:189-193`).

A fatia B abre no fecho de iteração em que o nível do ANDAR é ≥ `Balanceador.nivel_abre_b` = 5 (`knobs.py:396`). Nesse fecho, as células `CARREGAR/reto` e `PEGAR/pegar` abrem juntas (`balanceador.py:209-211`). A fatia C abre quando a célula `PEGAR/pegar` tem nota s ≥ `Balanceador.portao` = 0,75: nesse fecho a célula `BOTAR/botar` abre (`knobs.py:391`, `balanceador.py:231-232`).

### 1.3 Sorteio da tarefa

Não existe um sorteio separado para a tarefa. A fatia escolhe o primeiro trecho da tabela, e o trecho dá a tarefa (`comando.py:660`, `comando.py:575`). Para o ANDAR pleno, o trecho único tem duração `lo + (hi − lo)·U`, com `lo` = `hi` = 20,0 s. O valor é fixo (`comando.py:578-579`). O sorteio acontece no reset, em `_abre_trecho`.

### 1.4 Sorteio do terreno e da célula do ANDAR

O terreno é a superfície em que o robô pisa. O mjlab monta uma grade de sub-terrenos: uma coluna por tipo de terreno e uma linha por nível de dificuldade (`cena.py:275-276`, `terrain_generator.py:249-263`). A grade tem `Terreno.linhas` = 13 linhas (`knobs.py:429`) e 9 colunas (`Terreno.colunas`, `knobs.py:434-435`). Cada sub-terreno mede `Terreno.tamanho` = 8,0 m × 8,0 m (`knobs.py:430`). A borda do gerador mede `Terreno.borda` = 20,0 m (`knobs.py:446`, `cena.py:275-276`).

A célula é a unidade de nota do currículo. O ANDAR tem duas células: `andar` (índice 0) e `terreno` (índice 1) (`balanceador.py:22-23`). A célula `andar` abre no começo. A célula `terreno` abre depois (seção 5).

O sorteio do terreno acontece no reset, dentro de `sorteia_fatia`, só para env da fatia ANDAR (e da fatia B, que está fora deste documento):

1. A probabilidade de terreno é `p` = a fração da célula `terreno` entre as células abertas do ANDAR, ou 0 se a célula está fechada (`eventos.py:36-41`, `eventos.py:165-166`).
2. O env vira episódio de terreno quando `U < p` (`eventos.py:167`).
3. No episódio de terreno, a coluna sai de 1 a 8 (a coluna 0, `plano`, fica de fora). O código usa o mesmo sorteio u0 e a inversa da soma acumulada de oito pesos. O peso de cada coluna é a fração `fracoes(atraso, Terreno.piso_coluna, eps)`, com `atraso = 1 − linha[coluna] / 12`, `Terreno.piso_coluna` = 0,05 (`knobs.py:443`) e `eps` = 0,02. A coluna com a linha mais baixa recebe mais episódios. Com as oito linhas iguais, cada coluna recebe 1/8 (`eventos.py:63-65`).
4. A linha da coluna sorteada é `floor(linha[coluna] + u1)`, com u1 uniforme em [0; 1) e teto em 12. É um arredondamento estocástico: a média das linhas sorteadas é igual a `linha[coluna]`, um número real que o currículo move (seção 5.6) (`eventos.py:66`).
5. Sem terreno, a coluna é 0 e a linha é 0 (`eventos.py:74`).

Antes de sortear, o mesmo evento registra a visita e a travessia do episódio que acabou (seção 11.6) (`eventos.py:143-160`).

A fração das células vem de `fracoes(deficits, piso, eps)` (`balanceador.py:37-41`, `balanceador.py:295-299`):

`F_i = piso_sub + (1 − n·piso_sub) · (D_i + eps) / Σ_j (D_j + eps)`.

O déficit `D_i` da célula `andar` é `max(portão − s, 0)`. O déficit da célula `terreno` do ANDAR não usa a nota: é a distância das oito linhas ao topo, `1 − (média das linhas de 1 a 8) / 12` (`balanceador.py:136-145`). O robô parado na plataforma tirava nota ~0,5 e zerava o déficit.

Os números: `Balanceador.piso_sub` = 0,10 (`knobs.py:389`) e `Balanceador.eps` = 0,02 (`knobs.py:390`). O portão da célula `andar` é `Balanceador.portao_andar` = 0,75 (`knobs.py:392`). O portão da célula `terreno` é `Balanceador.portao_andar_terreno` = 0,50 (`knobs.py:393`) (`balanceador.py:97-98`). Ele só vale na condição do refino (seção 5.8). O `s` é a nota da célula (seção 5).

A tabela abaixo mostra o que cada coluna sorteia. A linha `r` vira a dificuldade `t = r / 12`, com `num_rows` = 13 (`terrain_generator.py:255-257`). O mjlab usa o `difficulty_range` padrão (0; 1) porque `cena.py` não o muda.

| Coluna | Nome | Geometria em função de t | Fonte |
|---|---|---|---|
| 0 | `plano` | chão plano | `cena.py:263` |
| 1 | `escada` | degrau = 0 + t·`tetos_andar["degrau"]` = t·0,20 m. Largura do degrau 0,3 m, plataforma 2,0 m | `knobs.py:432`, `primitive_terrains.py:79-81`, `config.py:46-52`, `cena.py:264` |
| 2 | `escada_inv` | mesma regra, escada invertida | `primitive_terrains.py:256-258`, `cena.py:265` |
| 3 | `rampa` | inclinação (subida/avanço) = t·`tetos_andar["rampa"]` = t·0,36 | `heightfield_terrains.py:243`, `cena.py:266` |
| 4 | `rampa_inv` | mesma regra, invertida | `heightfield_terrains.py:239-241`, `cena.py:267` |
| 5 | `escada_aleatoria` | altura de cada degrau = U(0; 0,20 m)·(0,5 + 0,5·t). Largura 0,8 m | `primitive_terrains.py:1061`, `config.py:198-208`, `cena.py:268` |
| 6 | `irregular` | ruído de altura de (0,02·t; 0,10·t) m, com `scale_with_difficulty` ligado | `config.py:99-108`, `heightfield_terrains.py:418-420`, `cena.py:269` |
| 7 | `obstaculos` | altura-base H = t·`tetos_andar["obstaculo"]` = t·0,15 m. São 40 obstáculos de largura 0,3 m a 1,0 m; cada um tem altura sorteada com igual chance entre −H, −H/2, +H/2 e +H (buracos e degraus) | `heightfield_terrains.py:725-749`, `config.py:123-133`, `cena.py:270` |
| 8 | `caixas` | 80·(0,5 + 0,5·t) caixas, altura de cada uma = U(0; 0,15 m)·(0,2 + 0,8·t), sem giro | `primitive_terrains.py:743`, `:796`, `config.py:167-179`, `cena.py:272-273` |

As colunas 1, 2, 3, 4, 5, 7 e 8 usam plataforma plana de `Terreno.plataforma` = 2,0 m (`knobs.py:441`, `cena.py:260`). A coluna `irregular` não tem esse argumento (`cena.py:269`).

A origem do sub-terreno é o ponto em que o robô nasce: o centro xy do sub-terreno, na altura abaixo. Cada gerador do mjlab define a sua altura (`terrain_generator.py:262`).

| Coluna | Altura z da origem | Fonte |
|---|---|---|
| `plano` | 0 | `primitive_terrains.py:47` |
| `escada` | (n_degraus + 1)·altura do degrau, o topo da plataforma. Com `Terreno.tamanho` = 8,0 m, borda 1,0 m, `Terreno.plataforma` = 2,0 m e degrau de 0,3 m, n_degraus = int((8,0 − 2·1,0 − 2,0) / (2·0,3)) = 6, então z = 7·t·0,20 m (derivado) | `primitive_terrains.py:84-96`, `:233-235`, `config.py:46-52` |
| `escada_inv` | −(n_degraus + 1)·altura do degrau = −7·t·0,20 m (derivado) | `primitive_terrains.py:410-412` |
| `rampa` | altura máxima da superfície da rampa (o topo da pirâmide) | `heightfield_terrains.py:366-370` |
| `rampa_inv` | deslocamento vertical do campo de alturas (`hfield_z_offset`) | `heightfield_terrains.py:365-370` |
| `escada_aleatoria` | soma das alturas dos degraus sorteados (`current_z`) | `primitive_terrains.py:1159-1160` |
| `irregular` | (ruído mínimo + ruído máximo) / 2 = (0,02·t + 0,10·t) / 2 = 0,06·t m (derivado) | `heightfield_terrains.py:535-536` |
| `obstaculos` | `origin_z_offset` = 0,0 | `heightfield_terrains.py:707`, `:840` |
| `caixas` | 0 | `primitive_terrains.py:832` |

O robô nasce em z = z da origem + `pelve_de_pe` + 0,01 (seção 2.3). Não determinado pelo código lido: o valor numérico da altura máxima da superfície da rampa e do `hfield_z_offset` (dependem de contas de pixels do mjlab que esta leitura não seguiu até o fim).

---

## 2. Reset

O reset acontece quando o episódio termina (terminação ou `time_out`). O mjlab chama `_reset_idx` no mesmo passo em que o env termina (`manager_based_rl_env.py:445-447`). A ordem é fixa (`manager_based_rl_env.py:553-591`):

1. O currículo roda (`curriculum_manager.compute`).
2. A simulação e a cena resetam (`sim.reset`, `scene.reset`).
3. Os eventos de modo `reset` rodam, na ordem do dicionário de eventos.
4. As observações, a ação, as recompensas, as métricas, o currículo, os comandos, os eventos e as terminações resetam, nessa ordem.
5. O contador `episode_length_buf` volta a 0.

### 2.1 Currículo e atraso

O termo de currículo `v3` roda primeiro (`env_cfg.py:216`). Ele fecha a iteração do balanceador quando a iteração mudou: `it = common_step_counter // passos_por_iteracao`, com `Balanceador.passos_por_iteracao` = 24 (`knobs.py:402`, `curriculo.py:119-122`). O termo fecha a iteração uma vez, mesmo que mais de uma iteração tenha passado desde o último reset (`curriculo.py:120-122`). A primeira chamada só grava a iteração corrente e não fecha nada, porque `v3_iter` começa em −1 (`curriculo.py:58-59`, `curriculo.py:120`).

Depois, o termo lê `cur.dr()["atraso_max"]`. Se o teto do atraso é diferente de `DR.atraso_inicio[1]` = 0 (`knobs.py:511`), o termo chama `eventos.atraso` e põe o teto novo nos buffers de atraso do atuador e da observação (`curriculo.py:123-126`, `eventos.py:244-258`). O teto é `ceil(DR.atraso_portao[1] · restricao − 1e-9)` com `DR.atraso_portao[1]` = 2 (`knobs.py:512`) enquanto o degrau global é menor que `DR.degrau_atraso_refino` = 3 (`knobs.py:514`). Com o degrau global ≥ 3, o teto é `DR.atraso_refino[1]` = 3 (`knobs.py:513`, `balanceador.py:381-382`). A `restricao` é definida na seção 5.

### 2.2 Eventos de reset, na ordem

Todos os eventos de reset valem para todo env no reset. A tabela mostra o que cada um faz para um env que vai para a tarefa ANDAR na fatia ANDAR.

| # | Evento | O que faz | Faixa ou valor | Fonte |
|---|---|---|---|---|
| 1 | `fatia` | Registra a visita e a travessia do episódio de terreno que acabou, depois sorteia a fatia, o terreno (coluna e linha) e a origem do env. Zera `v3_t_empurrao` para −∞ | seção 1.2 e 1.4 | `env_cfg.py:154`, `eventos.py:133-175` |
| 2 | `cena` | Sorteia topo e massa da caixa. Na fatia ANDAR a laje e a caixa vão para `estaciona_cena` | topo = 0, massa = `Cena.caixa_massa_base` = 1,0 kg (`knobs.py:80`) | `env_cfg.py:156`, `eventos.py:332-350`, `eventos.py:103-109` |
| 3 | `base` | Põe a pose da base no ANDAR por `reset_root_state_uniform` | x U(−0,5; 0,5) m, y U(−0,5; 0,5) m, yaw U(−π; π) | `env_cfg.py:157-158`, `eventos.py:211-225` |
| 4 | `juntas` | Põe as juntas na pose de pé e zera a velocidade | offset 0 em posição e velocidade | `env_cfg.py:159-160`, `events.py:279-313` |
| 5 | `carga` | Escreve massa e inércia da caixa | massa `v3_massa`, inércia (2/3)·m·a² | `env_cfg.py:161`, `eventos.py:443-461` |
| 6 | `atrito_pad` | Sorteia o atrito das almofadas de palma | U(`mu_min_pad`; `DR.atrito_pad[1]`) com trava; sem trava, valor fixo | `env_cfg.py:162`, `eventos.py:464-475` |
| 7 | `atrito_laje` | Sorteia o atrito da laje | U(`DR.atrito_laje`) com trava; sem trava, valor fixo | `env_cfg.py:164`, `eventos.py:478-489` |
| 8 | `vies_imu` | Sorteia o viés do giro e a inclinação do IMU | seção 6, com trava | `env_cfg.py:166`, `eventos.py:492-509` |
| 9 | `pd` | Escala kp e kd das juntas | seção 6 | `env_cfg.py:170`, `eventos.py:523-531` |
| 10 | `forca_motor` | Escala o limite de esforço dos motores | seção 6 | `env_cfg.py:173-175`, `eventos.py:534-545` |
| 11 | `solref` | Sorteia o `solref[0]` de pés, almofadas, laje e caixa | seção 6 | `env_cfg.py:177-180`, `eventos.py:548-565` |
| 12 | `filtros` | Zera o filtro de elevação do lidar, a odometria e o estimador da caixa | pose da odometria = pose verdadeira | `env_cfg.py:184`, `eventos.py:228-241` |

Os eventos 9, 10 e 11 existem só no treino. Eles saem do cfg de play (`env_cfg.py:169-181`). O evento `com_caixa` não entra no cfg porque `DR.com_caixa` = False (`knobs.py:527`, `env_cfg.py:182-183`).

### 2.3 Base

O evento `base` escolhe o ramo pela fatia: o ANDAR usa `reset_root_state_uniform` e as fatias B e C usam `_base_da_pega` (`eventos.py:221-225`). Para o ANDAR, o evento usa `Cena.reset_base_andar` = x (−0,5; 0,5), y (−0,5; 0,5), yaw (−π; π) (`knobs.py:103`). O mjlab sorteia cada componente com distribuição uniforme. As chaves ausentes (z, roll, pitch) valem 0 (`events.py:27-41`).

A posição final é `default_root_state + amostra + env_origins` (`events.py:174-176`). O `default_root_state` tem z = `pelve_de_pe` + 0,01. O valor de `pelve_de_pe` é 0,78905 m (0,7890549) em `dados/referencias.npz` (`cena.py:170-173`, `cena.py:222-223`). A origem do env é a origem do sub-terreno sorteado (`eventos.py:175`). No plano, a origem é a do chão da coluna 0. O quaternion é o padrão vezes a rotação de yaw (`events.py:177-180`). A velocidade da base é a velocidade do `default_root_state` mais a amostra de `velocity_range` = {}, que soma 0 (`eventos.py:224`, `events.py:183-184`). O `default_root_state` copia posição, rotação, velocidade linear e velocidade angular de `init_state` (`entity.py:1174-1185`). A cena não define velocidade em `init_state`, e o padrão do mjlab é (0, 0, 0) para as duas (`cena.py:221-227`, `entity.py:77-78`). A base nasce parada.

Para a fatia B ou C, a base nasce atrás da caixa, em `ref.recuo` ± `Alvos.faixa_recuo` = 0,08 m (`knobs.py:227`). Esse caso fica fora deste documento, salvo o primeiro trecho ANDAR (`eventos.py:178-208`).

### 2.4 Juntas

O evento `juntas` chama `reset_joints_by_offset` com `position_range` = (0,0; 0,0) e `velocity_range` = (0,0; 0,0) (`env_cfg.py:159-160`). O resultado é a pose padrão da entidade, limitada aos limites suaves das juntas. A velocidade fica 0 (`events.py:296-302`). A pose padrão é `q_de_pe`, lida de `dados/referencias.npz` (`cena.py:225`).

### 2.5 Cena, caixa e laje

Na fatia ANDAR, a laje e a caixa não participam. A função `estaciona_cena` põe o topo da laje em `env_origins.z + Cena.estaciona_z` = 20,0 m (`knobs.py:114`, `eventos.py:317`). A laje é um corpo mocap e fica a `Cena.prateleira_meia_z` = 0,02 m abaixo do topo (`eventos.py:276-294`). A caixa nasce apoiada na laje, com o centro a `v3_meia` acima do topo e velocidade 0 (`eventos.py:297-307`, `eventos.py:319`). A posição xy da caixa é `Cena.caixa_xy` = (0,32; 0,00) m somada à origem (`knobs.py:96`).

A meia aresta `v3_meia` é sorteada no startup, entre `Cena.caixa_n_variantes` = 8 valores espaçados de `Cena.caixa_meia_faixa` = (0,07; 0,13) m (`knobs.py:78-79`, `eventos.py:429-440`). Ela não muda a cada reset.

### 2.6 Terreno

O evento `fatia` escreve `terrain_levels`, `terrain_types` e `env_origins` do env (`eventos.py:170-175`). O terreno do ANDAR segue a seção 1.4. O registro da visita e da travessia, que o evento faz antes dessa escrita, segue a seção 11.6.

### 2.7 Carga

O ANDAR não carrega nada. A caixa de 1,0 kg está parada na laje estacionada. O termo `forca_caixa` zera a força aplicada à caixa fora de "CARREGAR e com a caixa pega e sem perda" (`eventos.py:598-602`). O ANDAR nunca satisfaz essa condição.

### 2.8 Comando, trecho e estado de episódio

Depois dos eventos, o termo `cadeia` roda `_resample` (`comando.py:639-660`). Ele faz, nesta ordem:

1. Registra no currículo o PEGAR e o BOTAR que estavam abertos. O ANDAR não tem registro aqui (`comando.py:643-645`). Em seguida, conta como falha a visita de zona aberta no reset e a encerra; o ANDAR não abre visita (`comando.py:646-650`).
2. Zera os flags do episódio: `v3_pegou`, `v3_perdeu`, `v3_pega_fechada`, `v3_bota_fechada`, `v3_perdeu_agora`, `v3_avancou`, `v3_aviso`, `v3_fim` (`comando.py:651-653`).
3. Zera `v3_relogio_alvo`, põe `v3_sigma` = 1,0, `v3_phi` = −1,0, `v3_h` = 1,0, `v3_phi_melhor` = 0,0 e `v3_v_chegada` = −1,0 (`comando.py:654-658`).
4. Abre o primeiro trecho da fatia (`comando.py:660`). A abertura sorteia a duração, põe `v3_trecho`, zera o relógio do trecho e o aviso e publica a tarefa (`comando.py:568-587`).

A publicação grava `v3_tarefa` = ANDAR, `v3_freio` = nível do ANDAR + 1 e `v3_degrau` = degrau de refino do ANDAR (`comando.py:508-519`). Depois ela chama `twist._resample` (`comando.py:519`). Na abertura de um trecho ANDAR, o alvo da cadeia fica em −1 (`comando.py:585-587`). A publicação da perda da caixa não mexe nesse índice: no passo da perda ele vira B3 (índice 6) (`comando.py:415`), e o B3 da caixa perdida fecha pela pose de pé (`comando.py:454`, `comando.py:460-466`, `comando.py:475-481`). Quando o B3 fecha, o índice volta a −1 e `v3_bota_fechada` fica verdadeiro (`comando.py:492-501`, `alvos.py:195-210`).

O termo `twist` roda depois da cadeia (`env_cfg.py:118-128`). O `reset` do twist fecha o segmento do episódio anterior com `falha` = não foi `time_out` (`comando.py:882-883`). Depois ele zera `_agora`, a fase da marcha, o filtro de velocidade `v_filt`, o filtro do acerto do relógio (`q_filt` = 0,5), o fator `g_marcha` e os acumuladores (`comando.py:884-892`). Por fim, ele chama `super().reset`, que sorteia o comando de novo (`comando.py:893`). Por isso o comando é sorteado duas vezes no reset: a primeira vez pela publicação da cadeia e a segunda pelo `reset` do twist. Vale o segundo sorteio. Só nele o alvo de rumo do ANDAR de terreno é fixado: o `_agora` do twist já vale 0 e a pose da base já é a nova (seção 3.4). O twist abre o segmento com o comando vigente (`comando.py:894`).

### 2.9 Atraso e ruído de observação

O atraso é sorteado uma vez por episódio. O `DelayBuffer` zera o lag e o contador de passos no reset (`delay_buffer.py:194-208`). No primeiro `compute` depois do reset, o buffer sorteia um lag inteiro uniforme entre `min_lag` e `max_lag`, por env. Não existe novo sorteio no episódio porque `delay_update_period` = `NUNCA` = 10⁹ passos (`delay_buffer.py:247-258`, `cena.py:196-197`, `observacoes.py:383-385`, `knobs.py:47`). O teto efetivo é `max_lag`, definido na seção 2.1. Enquanto o buffer tem poucas amostras, o lag aplicado é no máximo `amostras − 1` (`delay_buffer.py:242`). Cada termo de observação do ator com atraso tem um buffer, com um lag por env (`observation_manager.py:466-475`, `delay_buffer.py:269-278`). O atuador tem um buffer por grupo de atuador, e grupos fundidos compartilham o buffer (`cena.py:188-198`, `eventos.py:253-255`, `actuator.py:314-322`).

### 2.10 O primeiro passo depois do reset

Para um env que reseta no meio de um passo de treino, o reset ocorre depois das terminações e das recompensas do mesmo passo. Depois do reset, o mjlab roda `sim.forward()` e `command_manager.compute(dt)` para todos os envs (`manager_based_rl_env.py:454-456`).

### 2.11 Ordem de um passo da política

Cada passo da política faz, nesta ordem (`manager_based_rl_env.py:420-464`):

1. Processa a ação e roda `Contrato.decimacao` = 4 passos de física (`knobs.py:55`).
2. Soma 1 em `episode_length_buf` e em `common_step_counter`.
3. Calcula as terminações, depois as recompensas, depois as métricas.
4. Reseta os envs terminados (seção 2).
5. Roda `sim.forward()`.
6. Roda os termos de comando: primeiro `cadeia`, depois `twist` (`env_cfg.py:118-128`). A seção 4.3 lista o que cada um faz.
7. Roda os eventos de modo `step` (`forca_caixa`, `rfi`) e os de modo `interval` (`empurrao`), nesta ordem (`manager_based_rl_env.py:459-461`).
8. Roda `sim.sense()` e calcula as observações.

Por isso a recompensa e a terminação de um passo leem o estado dos comandos gravado no passo anterior.

---

## 3. Comando ou alvo

O robô recebe um comando de velocidade (vx, vy, wz) no frame da base. O vx é para a frente, o vy é para a esquerda e o wz é o giro em torno do eixo vertical. O comando fica em `vel_command_b` do twist (`velocity_command.py:38`, `velocity_command.py:57-59`). O ANDAR não tem alvo de posição. O alvo da cadeia é −1 na abertura do trecho ANDAR (`comando.py:585-587`). Depois da perda da caixa ele vale B3 até o B3 fechar (seção 2.8).

### 3.1 Como o twist sorteia o comando

O twist sorteia o comando em `_resample_command` (`comando.py:897-941`). Para um env de ANDAR na fatia ANDAR:

1. A célula é a `terreno` se o env está num episódio de terreno (`terrain_types ≠ 0`). Senão, a célula é sorteada entre as células abertas do ANDAR com a fração da célula `terreno` forçada a zero (`comando.py:914-918`, `balanceador.py:306-316`). Como a célula `andar` é a única restante, o sorteio sempre devolve `andar`.
2. O nível da velocidade é `v3_cur.nivel("ANDAR")` no instante do sorteio (`comando.py:920`).
3. A função `sorteia_velocidade` devolve (vx, vy, wz) (`comando.py:921-925`).

A função `sorteia_velocidade` para o ANDAR segue estes passos (`comando.py:125-147`). Os tetos usam `n` = nível:

| Teto | Fórmula | Knobs e valores |
|---|---|---|
| `lado` | `min(andar_lado_inicio + andar_passo·n, lado_teto_max)` | `Comando.andar_lado_inicio` = 0,2 (`knobs.py:130`), `Comando.andar_passo` = 0,1 (`knobs.py:134`), `Comando.lado_teto_max` = 0,3 (`knobs.py:124`) |
| `frente` | `min(andar_inicio + andar_passo·n, frente_tetos[-1])` | `Comando.andar_inicio` = 0,3 (`knobs.py:129`), `Comando.frente_tetos[-1]` = 1,0 (`knobs.py:122`) |
| `tras` | `min(andar_inicio + andar_passo·n, tras_teto_max)` | `Comando.tras_teto_max` = 0,5 (`knobs.py:123`) |
| `giro` | `min(andar_giro_inicio + andar_passo·n, giro_tetos[-1])` | `Comando.andar_giro_inicio` = 0,3 (`knobs.py:131`), `Comando.giro_tetos[-1]` = 0,5 (`knobs.py:125`) |

Na célula `terreno`, `frente` e `tras` ainda passam por `min(·, Comando.terreno_v_max)` com `terreno_v_max` = 1,0 (`knobs.py:148`, `comando.py:131-132`). Esse corte não muda nada, porque `frente` ≤ 1,0 e `tras` ≤ 0,5. O corte vale só para a escolha dos tetos: na célula `terreno` o `tras` não entra no sorteio (adiante).

O sorteio em si (`comando.py:136-146`). Na célula `terreno` o código faz só o passo 1 e põe `vx = frente · r`, `vy = 0` e `wz = 0` (`comando.py:137-138`). O ANDAR de terreno recebe só comando para a frente. O giro dele vem do rumo (seção 3.4). Os passos 5 a 7 valem para as duas células.

Cada U abaixo é um sorteio uniforme em [0; 1) novo e independente, tirado de `v3_gerador` (`comando.py:98-99`).

1. `r = sqrt(U)` e `ang = π·(U − 0,5)`. O ângulo vai de −π/2 a +π/2.
2. O semieixo x é `+frente` com probabilidade `frente / (frente + tras)` e `−tras` com a probabilidade restante.
3. `vx = semieixo_x · r · cos(ang)` e `vy = lado · r · sin(ang)`. O ponto (vx, vy) é uniforme na área de meia elipse, com os semieixos `frente` ou `tras` em x e `lado` em y.
4. `wz = giro · (2·U − 1)`. O giro é uniforme em (−giro; giro) e independente do linear.
5. Se `|wz|` < `Comando.giro_min` = 0,1 (`knobs.py:132`), o código zera o wz.
6. Se `‖(vx, vy)‖` < `Comando.linear_min` = 0,1 (`knobs.py:133`), o código zera o vx e o vy.
7. Com probabilidade `Comando.andar_parado` = 0,10 (`knobs.py:135`), o código zera os três canais.

Os cortes dos passos 5 e 6 fazem a fração de canais zerados ficar acima de `andar_parado`.

Tetos por nível (derivados da tabela acima; unidade m/s e rad/s):

| Nível | frente | trás | lado | giro |
|---|---|---|---|---|
| 0 | 0,3 | 0,3 | 0,2 | 0,3 |
| 1 | 0,4 | 0,4 | 0,3 | 0,4 |
| 2 | 0,5 | 0,5 | 0,3 | 0,5 |
| 3 | 0,6 | 0,5 | 0,3 | 0,5 |
| 4 | 0,7 | 0,5 | 0,3 | 0,5 |
| 5 | 0,8 | 0,5 | 0,3 | 0,5 |
| 6 | 0,9 | 0,5 | 0,3 | 0,5 |
| 7 | 1,0 | 0,5 | 0,3 | 0,5 |

O nível máximo do ANDAR é `round((frente_tetos[-1] − andar_inicio) / andar_passo)` = 7 (`balanceador.py:72`).

⚠ Divergência: o comentário em `balanceador.py:71` diz que o ANDAR tem 10 níveis; o código dá 7 (`balanceador.py:72`).

Os knobs `Comando.piso_velocidade`, `Comando.piso_lado`, `Comando.frente_tetos[0:-1]` e `Comando.giro_tetos[0:-1]` não entram no sorteio do ANDAR. Eles servem ao CARREGAR (`comando.py:102-110`, `comando.py:148-158`).

### 3.2 Quando o comando muda

O twist sorteia o comando nestes momentos:

| Momento | Mecanismo | Fonte |
|---|---|---|
| No reset do episódio (duas vezes; vale a segunda) | `cadeia._publica` e `twist.reset` | `comando.py:519`, `comando.py:893` |
| Quando `time_left` do twist chega a 0 ou menos | `CommandTerm.compute` | `command_manager.py:109-115` |
| Quando o trecho muda e a cadeia publica uma tarefa nova. No ANDAR pleno o trecho nunca muda. A publicação também acontece na perda da caixa, e aí o comando do ANDAR é zero (seção 1.1) | `cadeia._publica` | `comando.py:420`, `comando.py:519` |

Cada sorteio do comando também sorteia o próximo `time_left` = U(`Comando.reamostra_s`) = U(3,0; 8,0) s, passado ao twist como `resampling_time_range` (`knobs.py:137`, `env_cfg.py:122-123`, `command_manager.py:117-123`). A média é 5,5 s (derivado). O `time_left` cai `step_dt` por passo, dentro do `compute` do twist, depois das recompensas do passo (`command_manager.py:109-112`, `manager_based_rl_env.py:440-456`). Num episódio de 20 s, o ANDAR pleno tem, em média, 20 / 5,5 ≈ 3,6 reamostras depois do comando sorteado no reset (derivado).

O twist não muda vx e vy durante o segmento. O método `_update_command` só age nos envs de rumo, que dão o wz a cada passo (seção 3.4) (`comando.py:943-948`).

Nos trechos ANDAR das fatias B e C o comando é zero (seção 1.1). O `v3_celula` do env fica em −1 nesses trechos (`comando.py:901`).

### 3.3 O que o robô observa do comando

O ator observa o comando pelo termo `comando` (3 números, histórico de 5 passos, escala 1,0) (`knobs.py:60`, `observacoes.py:392`). A observação inclui também a fase da marcha (seção 9).

### 3.4 Rumo do ANDAR de terreno

O ANDAR de terreno segue um rumo, e não um giro sorteado. `TwistV3Cfg.heading_command` = True liga a classe do fabricante (`comando.py:678`). `ranges.heading` = (−π; π) existe só porque o fabricante o exige (`env_cfg.py:127`). `rel_heading_envs` = 0: o sorteio do fabricante nunca liga o rumo, e a marca `is_heading_env` é do twist (`comando.py:679`).

1. No `_resample_command` de um env de ANDAR em terreno, `is_heading_env` fica verdadeiro se o comando linear sorteado tem `‖c_xy‖` ≥ `Comando.limiar_canal` = 0,02. Comando parado (`andar_parado` ou `linear_min`) fica sem rumo. Em todos os outros envs, a marca vira falsa (`comando.py:903`, `comando.py:932`).
2. No começo do episódio (`_agora` = 0), o alvo `heading_target` é o múltiplo de π/2 mais próximo do rumo de partida da base. O rumo vem do `qpos`, porque o `xquat` ainda é o do fim do episódio anterior. O alvo é a normal de uma face da célula (`comando.py:933-937`). O alvo vale para todo env de terreno, mesmo sem comando linear; um segmento seguinte pode ter comando. O alvo não muda nos segmentos seguintes.
3. A cada passo, `_update_command` chama o fabricante, que põe `wz = clip(heading_control_stiffness · erro_de_rumo, −0,5; 0,5)` nos envs de rumo. A rigidez vale 1,0 (`velocity_command.py:284`) e a faixa de giro vem de `ranges.ang_vel_z` = (−0,5; 0,5) (`env_cfg.py:126`, `velocity_command.py:117-123`).
4. Depois, o twist zera o wz com módulo abaixo de `giro_min` = 0,1 rad/s: uma tolerância de ±0,1 rad (±5,7°) no rumo (`comando.py:943-948`).
5. Na abertura de um segmento, a cópia do comando (`seg_cmd`) zera o wz dos envs de rumo. O canal de giro da nota do segmento fica "sem comando" e cobra a deriva, e a nota não cobra o giro de alinhamento (`comando.py:738-742`, seção 5.2).

Com o erro de rumo inicial uniforme em ±π/4 (o alvo é o múltiplo de π/2 mais próximo), a fração dos episódios com erro abaixo do corte de `giro_min` é 0,1 / (π/4) = 0,127, e a fração com wz saturado em 0,5 é 1 − 0,5 / (π/4) = 0,363 (derivado, conta direta, sem simulação). A hipótese: o robô começa com yaw uniforme em (−π; π) (seção 2.3), então o erro ao múltiplo de π/2 mais próximo é uniforme em ±π/4.

---

## 4. Etapas dentro do episódio e critério de avanço

A tarefa ANDAR não tem etapas. As etapas P0 a P3 e B1 a B3 são do PEGAR e do BOTAR (`knobs.py:44`). No ANDAR, o índice do alvo `v3_alvo_idx` vale −1 na abertura do trecho (`comando.py:585-587`). Depois da perda da caixa ele vale B3 até o B3 fechar (seção 2.8). A zona adaptativa por etapa e a renda por etapa não pagam no ANDAR: o termo `progresso` zera a etapa com a caixa perdida e paga só a etapa 7 no ANDAR final (seção 7).

O episódio do ANDAR tem dois níveis de divisão no tempo: os trechos e os segmentos.

### 4.1 Trechos

O trecho é a parte do episódio que tem uma tarefa. Na fatia ANDAR existe um trecho só. O trecho avança por tempo. A função `avanca_trecho` calcula, por passo (`comando.py:201-216`):

- `rel = relogio + dt`.
- `avancou = rel ≥ alvo − 1e-6`, com `alvo` = a duração sorteada do trecho (20,0 s no ANDAR pleno).
- `fim = avancou & (proximo < 0)`.

No ANDAR pleno, o trecho não tem próximo. Quando `avancou` fica verdadeiro, `v3_fim` marca o fim da cadeia (`comando.py:613-614`) e a terminação `time_out` dispara no passo seguinte (seção 8). Os 20,0 s do trecho correspondem a `max_episode_length` = `ceil(20,0 / 0,02)` = 1000 passos (derivado de `knobs.py:160` e `manager_based_rl_env.py:281-283`). O relógio do trecho soma `dt` no `compute` da cadeia, que roda depois das terminações (`manager_based_rl_env.py:436`, `manager_based_rl_env.py:456`, `comando.py:209`). No passo 1000, a terminação `episode_length_buf ≥ max_episode_length` dispara antes de o relógio chegar a 20,0 s (ele vale 999 × 0,02 = 19,98 s). Por isso o `time_out` do ANDAR pleno vem do contador de passos, e o `v3_fim` desse trecho não chega a disparar (derivado).

O episódio de terreno do ANDAR tem um segundo fim: a travessia. A terminação `time_out` dispara quando a base passa de `Terreno.dist_travessia` = 3,5 m da origem da célula (seção 8). Por isso o episódio de terreno dura menos que 20 s quando o robô anda (seção 11.2).

Nas fatias B e C, os trechos ANDAR avançam por tempo também: o `por_detector` desses trechos é False (`comando.py:177-179`). O trecho de espera avança quando `rel ≥ U(0,5; 1,0) s − 1e-6`. O ANDAR final da fatia C avança com 1,5 s e fecha a cadeia (`comando.py:179`).

Depois da perda da caixa, o trecho congela até o fim do episódio. A cadeia substitui o `novo` trecho pelo atual e anula `avancou` e `fim` (`comando.py:610-613`).

### 4.2 Segmentos do comando

O segmento é o intervalo em que o comando de velocidade fica igual. A fronteira de segmento é a mudança do contador de comando (`command_counter`) (`comando.py:816-821`). Cada segmento é o que o currículo mede (seção 5).

O twist abre um segmento em `_abre`: ele copia o comando, o id da célula e a marca "tarefa com caixa", e zera os acumuladores do segmento (`comando.py:735-751`). Nos envs de rumo (seção 3.4), a cópia zera o wz (`comando.py:742`). A função `_fecha` calcula a nota e a registra no currículo (seção 5.2).

A âncora é o instante a partir do qual o segmento conta a velocidade. Ela é `max(t_abertura, t_empurrão) + Comando.ancora_s` = `max(...) + 1,5 s` (`knobs.py:145`, `comando.py:221-224`, `comando.py:834-836`). A velocidade só entra na nota quando `_agora ≥ âncora` (`comando.py:855-857`). O primeiro 1,5 s de cada segmento e o 1,5 s depois de cada empurrão não entram na nota.

### 4.3 O que cada termo de comando faz em um passo

O `compute` da cadeia faz, nesta ordem (`comando.py:596-629`):

1. Mede a base, a caixa, a laje, as palmas e os contatos (`_mede`, `comando.py:317-346`).
2. Detecta a perda da caixa e escreve `v3_perdeu_agora`. Com a chave `Cadeia.caixa_perdida_vira_andar` ligada, publica o ANDAR (`_perda`, `comando.py:394-420`). A detecção exclui o env que já está na tarefa ANDAR e o primeiro passo do episódio (`comando.py:410`).
3. Abre o P0 do PEGAR quando há leitura válida da caixa. O ANDAR não entra (`_abre_p0`, `comando.py:370-392`).
4. Avança o alvo (`_alvos`, `comando.py:422-504`). No ANDAR pleno o índice é −1 e nada avança.
5. Latch do aviso, avanço do trecho, congelamento do trecho depois da perda, `v3_fim`, registro do BOTAR pendente e abertura do trecho novo (`comando.py:604-619`).
6. Dá um passo na odometria com a deriva do degrau global (`comando.py:623-625`).
7. Grava a velocidade de chegada e a velocidade vertical da caixa (`comando.py:626-629`).

O `compute` do twist (`_update_metrics`, mais o `time_left` do mjlab) faz, nesta ordem (`comando.py:810-877`, `command_manager.py:109-115`):

1. Soma `dt` em `_agora` e grava a pose do fim do passo (`comando.py:813-815`).
2. Fecha e reabre o segmento se o contador de comando mudou (`comando.py:816-821`).
3. Acumula `Σ(v·ĉ)dt`, `Σ|c|dt` e `Σ(ω·sinal)dt` do segmento e a inclinação máxima da caixa (`comando.py:822-832`).
4. Atualiza o instante da âncora e grava a pose na âncora quando o instante chega (`comando.py:833-842`).
5. Atualiza as métricas do fabricante, a razão de marcha, a velocidade desde a âncora, `seg_dur` e o filtro `v_filt` (`comando.py:843-858`).
6. Atualiza o relógio de marcha, o acerto, `q_filt`, `g_marcha`, a fase e `v3_marcha_on` (`comando.py:859-877`).
7. Subtrai `dt` de `time_left` e reamostra o comando dos envs com `time_left ≤ 0` (`command_manager.py:109-115`).

As regras de sucesso por segmento com caixa (`sucesso_do_segmento`, `Comando.razao_portao` = 0,50, `Comando.deriva_pos_m`, `Comando.deriva_pos_inicio`, `Comando.deriva_rumo_rad`, `Comando.incl_max_terreno_graus`) não valem para segmento do ANDAR. Para esse segmento o código usa a nota da seção 5.2 (`comando.py:789-790`; `knobs.py:146`, `knobs.py:142-144`, `knobs.py:149`).

---

## 5. Progressão

Esta seção descreve como a tarefa ANDAR progride durante o treino. O estado do currículo mora em `balanceador.Curriculo` (`balanceador.py:64`). O termo `curriculo.v3` liga o estado ao env e fecha a iteração (`curriculo.py:103-143`). O runner grava o estado no checkpoint (`runner.py:88-90`) e o lê na retomada (`runner.py:110`).

### 5.1 Quem mede a nota

A cadeia de medida é:

1. O twist mede cada segmento do ANDAR e calcula a nota (`comando.py:759-794`).
2. O twist chama `Curriculo.registra(célula, n, soma)`. O método acumula `n` e `soma` da célula (`balanceador.py:148-151`).
3. No fecho de iteração, `Curriculo.fecha_iteracao` atualiza a nota `s` da célula com os acumulados (`balanceador.py:153-168`).
4. O fecho de iteração roda dentro do termo de currículo `v3`, no primeiro reset depois que `common_step_counter // 24` muda (`curriculo.py:119-122`).

Uma iteração do balanceador dura `Balanceador.passos_por_iteracao` = 24 passos da política (`knobs.py:402`). O mesmo número é o `num_steps_per_env` do PPO (`tarefa.py:49`). Uma iteração de PPO coincide com uma iteração do balanceador.

### 5.2 Nota de um segmento do ANDAR

`TwistV3._fecha` calcula a nota do segmento (`comando.py:780-794`). Seja `c` o comando do segmento e `v̄` a velocidade média da base depois da âncora, em frame da base.

- `seg_nota` soma `v·dt` com `v = (vx, vy, wz)` da base, só nos passos depois da âncora. `seg_dur` soma `dt` nesses passos (`comando.py:854-857`).
- `v̄ = seg_nota / max(seg_dur, 1e-6)` (`comando.py:781`).
- `ativo_lin = ‖c_xy‖ ≥ Comando.limiar_canal` e `ativo_giro = |c_z| ≥ limiar_canal`, com `limiar_canal` = 0,02 (`knobs.py:138`, `comando.py:782-783`).
- Razão do canal linear: `clamp((v̄_xy · c_xy) / ‖c_xy‖² , 0, 1)`. Razão do canal de giro: `clamp(v̄_z · sinal(c_z) / |c_z|, 0, 1)` (`comando.py:784-786`). O denominador é o módulo do comando, com piso `limiar_canal`.
- Deriva do canal sem comando: `exp(−‖v̄_xy‖² / σ²)` no linear e `exp(−v̄_z² / σ²)` no giro, com σ = `Recompensa.sigma_rastreio` = 0,5 (`knobs.py:238`, `comando.py:787`).
- Nota do segmento: a média dos dois canais, onde cada canal vale a razão (canal com comando) ou a deriva (canal sem comando) (`comando.py:788`).
- Segmento que acaba em queda tem nota 0. O código multiplica a nota por `(~falha)` (`comando.py:789`).
- Nos envs de rumo (seção 3.4), o `c_z` do segmento vale 0: o canal de giro é "sem comando" e cobra a deriva (`comando.py:742`).
- O segmento entra no registro quando `seg_dur > 0` ou quando acaba em falha (`comando.py:790`). `falha` é "terminação que não é `time_out`" no reset (`comando.py:883`).

O twist registra a nota por célula: `registra(célula, n_segmentos, soma_das_notas)` (`comando.py:792-794`). A célula é a que o env recebeu no sorteio (`seg_cel`). Segmentos com célula −1 não são registrados (`comando.py:792`).

### 5.3 Nota da célula (`s`) e fecho de iteração

A nota `s` da célula é a média móvel exponencial (EMA) das notas dos segmentos. A EMA é a média que dá peso `alfa` ao valor novo e `1 − alfa` ao valor antigo. O fecho de iteração faz, para cada célula com `n > 0` segmentos registrados na iteração (`balanceador.py:163-168`):

- `x = soma / n`.
- Se `n_medidas = 0`: `s = x`. Senão: `s = s + alfa·(x − s)`, com `Balanceador.alfa` = 0,05 (`knobs.py:394`).
- `n_medidas` sobe 1. Os acumuladores zeram.

A célula sem segmento na iteração não muda de `s` (`balanceador.py:164`). O valor inicial de `s` é 0,0 (`balanceador.py:79`).

A ordem do fecho de iteração (`balanceador.py:153-195`). O `registra` e o fecho só leem o que o `eventos.sorteia_fatia` e o twist gravaram até o reset anterior: o fecho roda no `curriculum_manager.compute`, antes dos eventos de reset do mesmo passo (`manager_based_rl_env.py:553-563`):

1. `it` sobe 1.
2. EMA das células.
3. Zona das etapas (não age no ANDAR).
4. Linha de terreno de cada coluna do ANDAR (seção 5.6) (`balanceador.py:177-182`).
5. Contador do refino de cada tarefa.
6. Aberturas de célula e de fatia.
7. Registro da iteração de abertura das fatias.
8. Rampa das fatias.
9. Subidas de nível.
10. Travas do DR.

### 5.4 Nível de velocidade do ANDAR

O nível do ANDAR vai de 0 a 7. Os valores de cada nível estão na tabela da seção 3.1. A subida é decidida em `_sobe` (`balanceador.py:250-278`):

- Condição para subir de nível (nível < máximo): a célula `andar` está aberta e `s_andar ≥ portao_andar` = 0,75. O código usa `_passa_media` sobre a célula `andar`, que é a única de velocidade do ANDAR (`balanceador.py:29-30`, `balanceador.py:259-260`, `balanceador.py:125-128`).
- A subida acontece depois de `Balanceador.iters_nivel` = 50 fechos de iteração seguidos com a condição verdadeira. Um fecho com a condição falsa zera a contagem `_seguidas` (`knobs.py:395`, `balanceador.py:265-268`).
- Quando sobe: `_nivel["ANDAR"]` soma 1 (o ANDAR não tem subnível: `subnivel("ANDAR")` devolve `_sub_max`, que não é menor que `_sub_max`) e o `n_medidas` das duas células do ANDAR vai a 0. A EMA das células recomeça na primeira medida do nível novo (`balanceador.py:273-278`, `balanceador.py:325-327`).

O que muda no env quando o nível sobe:

- Os tetos do sorteio de velocidade (seção 3.1). A mudança vale a partir do próximo sorteio de comando.
- A `restricao` e, por ela, os fatores das cercas e do estilo (seção 5.8 e seção 7).
- O freio publicado em `v3_freio` = nível + 1 e a observação `niveis` do crítico (`comando.py:515`, `observacoes.py:327-335`).

O nível 5 do ANDAR abre a fatia B (seção 1.2).

### 5.5 Abertura da célula `terreno` do ANDAR

A célula `terreno` do ANDAR abre quando: o nível do ANDAR é ≥ `Balanceador.nivel_abre_terreno` = 2 (`knobs.py:412`), a célula ainda está fechada, e a média das células abertas do ANDAR passa no portão (`balanceador.py:218-228`, `balanceador.py:125-128`). Com só a célula `andar` aberta, a condição é `s_andar ≥ 0,75`. A nota da célula `terreno` não entra, porque ela está fechada. O código roda essa abertura antes de `_sobe`, no mesmo fecho (`balanceador.py:185`, `balanceador.py:194`). O nível 2 é o piso: o robô aprende a andar no plano antes da escada.

A célula nova entra com `s` = 0 e `n_medidas` = 0 (`balanceador.py:79`). Depois de aberta, a célula recebe fração de sorteio de terreno pela fórmula da seção 1.4. A abertura do terreno do CARREGAR exige, além disso, `nivel_terreno("ANDAR")` ≥ 1 (`balanceador.py:225-226`).

O mínimo de fechos para abrir é 101 (derivado): dois níveis de 50 fechos seguidos cada, e a abertura no fecho seguinte ao da segunda subida, porque `_abre` roda antes de `_sobe`. Isso dá 101 × 24 = 2 424 passos da política.

### 5.6 Linha de terreno por coluna

O ANDAR não tem uma linha de terreno só. O `Curriculo` guarda uma `linha` real por coluna (lista de 9 números, a posição 0 não é usada), mais as visitas e as travessias da iteração por coluna (`vis_ter`, `trav_ter`) (`balanceador.py:86-89`). Todas as linhas começam em 0,0 e vão de 0 a `_max_ter["ANDAR"]` = `Terreno.linhas − 1` = 12 (`balanceador.py:77`).

No fecho de iteração, cada coluna de 1 a 8 com `vis_ter` ≥ `Balanceador.linha_n_min` = 50 faz (`balanceador.py:177-182`):

- `taxa = trav_ter / vis_ter`.
- `linha ← clip(linha + linha_ganho · (taxa − zona_t_alvo), 0, 12)`, com `Balanceador.linha_ganho` = 2,5 (`knobs.py:411`) e `Balanceador.zona_t_alvo` = 0,6 (`knobs.py:404`). A taxa 1,0 sobe 1,0 linha. A taxa 0 desce 1,5 linha (derivado: 2,5 × −0,6). A taxa 0,6 mantém a linha.
- `vis_ter` e `trav_ter` da coluna voltam a 0. Uma coluna com menos de 50 visitas acumula para o fecho seguinte.

O que conta visita e travessia está na seção 11.6. A linha da coluna não depende do nível de velocidade nem da nota da célula `terreno`. O nível do terreno do ANDAR, `nivel_terreno("ANDAR")`, é o piso da média das oito linhas: `floor(Σ linha[1:] / 8)` (`balanceador.py:329-333`). Esse nível entra na abertura do terreno do CARREGAR, na condição do refino (seção 5.8), no estado do checkpoint e no log. A linha sorteada do episódio é `floor(linha[coluna] + u)` (seção 1.4).

O `estado()` grava a chave `terreno` (linha, visitas e travessias), e o `carrega()` a lê. Um checkpoint sem a chave volta com todas as colunas na linha 0 e sem visitas. O `nivel_ter` antigo do ANDAR não volta (`balanceador.py:404`, `balanceador.py:417`, `balanceador.py:431-436`).

A célula `terreno` do ANDAR não sobe mais de nível pelo portão: `_sobe` só move a linha do CARREGAR (`balanceador.py:250-278`).

### 5.7 Zona adaptativa e subnível

A zona adaptativa e o subnível não agem no ANDAR. O vetor de zona `Curriculo.z` tem sete posições, uma por etapa P0 a B3 (`balanceador.py:33`, `balanceador.py:105`). O subnível existe só para PEGAR e BOTAR (`balanceador.py:90-92`). O crítico recebe o valor de `z` da etapa corrente; com o índice de alvo −1 o valor é 0, e depois da perda da caixa, com o índice em B3, o valor é o `z` do B3 (`observacoes.py:316-318`).

### 5.8 Degrau de refino, degrau global e restrição

O degrau de refino é um contador de 0 a `Balanceador.degraus` = 5 (`knobs.py:401`). Ele sobe com o ANDAR no estado final. A condição (`balanceador.py:234-248`):

- O nível do ANDAR é o máximo, a célula `terreno` está aberta e `nivel_terreno("ANDAR")` = 12. Como as linhas têm teto 12, o piso da média vale 12 só com as oito colunas em 12 (`balanceador.py:239-240`).
- A média das células abertas do ANDAR tem `s` ≥ a média dos portões, isto é, `(s_andar + s_terreno)/2 ≥ (0,75 + 0,50)/2` = 0,625 (derivado de `balanceador.py:125-128`, `knobs.py:392-393`).
- O contador `_cont` soma 1 por fecho de iteração com essa condição verdadeira e zera com a condição falsa (`balanceador.py:242-245`).
- Com `_cont ≥ Balanceador.iters_refino` = 100 (`knobs.py:400`), o degrau soma 1 e o contador zera (`balanceador.py:246-248`).

O degrau do ANDAR nunca desce. O env publica o degrau no `v3_degrau` na abertura de cada trecho (`comando.py:516-518`). Efeitos:

- A terminação `batente` liga para o env quando `v3_degrau ≥ Refino.degrau_batente` = 1 (`knobs.py:547`, `terminacoes.py:164`, `terminacoes.py:177-181`).
- O crítico observa `v3_degrau / 5` (`observacoes.py:338-340`, `observacoes.py:60`).

O degrau global é o menor degrau entre as quatro tarefas (`balanceador.py:341-343`). Ele controla o refino das perturbações da seção 6 (ganhos, força do motor, `solref`, RFI, atraso, deriva da odometria e ruído do estimador da caixa) pela função `no_degrau(ini, fim, d, degraus) = ini + (fim − ini)·d/degraus` (`balanceador.py:48-51`). O ANDAR sozinho não move o degrau global enquanto as outras três tarefas estão em 0.

A restrição é o menor valor de `nível / nível_máximo` entre as quatro tarefas (`balanceador.py:345-348`). Os níveis máximos são 7 (ANDAR) e 4 (PEGAR, CARREGAR, BOTAR) (`balanceador.py:72-74`). A restrição fica em 0 enquanto uma tarefa está no nível 0.

### 5.9 Travas do DR que o ANDAR liga

As travas só ligam e nunca desligam (`balanceador.py:101-103`, `balanceador.py:197-202`):

| Trava | Condição | O que liga |
|---|---|---|
| `vies` | `s_andar ≥ portao_andar` = 0,75 em algum fecho de iteração | O evento `vies_imu` (seção 6) |

As outras travas (`portao0`, `subida`) dependem de PEGAR e CARREGAR.

### 5.10 Resumo da progressão

| Estágio | O que muda | Condição de subida | Quem mede |
|---|---|---|---|
| Nível 0 → 7 | Tetos do comando; `restricao` | `s_andar ≥ 0,75` por 50 fechos seguidos | `TwistV3._fecha` → `registra` → EMA |
| Célula `terreno` abre | Episódios de terreno entram no sorteio | Nível ≥ 2 e `s_andar ≥ 0,75` | idem |
| Linha de cada coluna 0 → 12 | Dificuldade do terreno da coluna e o peso dela no sorteio | Taxa de travessia por visita acima de 0,6 sobe; abaixo desce; a cada 50 visitas da coluna | `eventos.sorteia_fatia` → `vis_ter` e `trav_ter` |
| Degrau de refino 0 → 5 | `batente`, observação `degrau`, degrau global | Nível = 7, as oito colunas na linha 12 e média das duas células ≥ 0,625, por 100 fechos seguidos | idem |
| Abertura da fatia B | A fração do ANDAR cai | Nível do ANDAR ≥ 5 | `Curriculo._abre` |

O mínimo de fechos de iteração, derivado das contagens: 7 níveis × 50 = 350 fechos até o nível 7. Mais 5 degraus × 100 = 500 fechos até o degrau 5. A linha 12 em todas as colunas exige 12 atualizações por coluna, no mínimo, cada uma com 50 visitas e taxa 1,0. O número de fechos dessa parte depende das visitas por iteração. O código não fixa o número de envs, então o mínimo em fechos não sai dele.

⚠ Divergência: o comentário em `balanceador.py:96` diz que o portão do ANDAR no plano é 0,80. O código usa `Balanceador.portao_andar` = 0,75 (`knobs.py:392`, `balanceador.py:97-98`).

---

## 6. Perturbações e randomizações

Cada linha diz quando a perturbação acontece, a distribuição, a faixa e a condição. Todas valem para o env na tarefa ANDAR, salvo nota em contrário. O atrito do terreno e outras propriedades do terreno vêm do mjlab e não mudam por episódio.

| Perturbação | Quando | Distribuição e faixa | Condição | Fonte |
|---|---|---|---|---|
| Empurrão | Por intervalo, por env | Soma à velocidade da base em x e y (mundo) um valor U(−`DR.empurrao`; `DR.empurrao`) = U(−0,5; 0,5) m/s, por eixo. O intervalo é U(`DR.periodo_empurrao_s`) = U(3,0; 7,0) s, sorteado de novo a cada empurrão. O reset do episódio não sorteia o intervalo de novo: o tempo restante segue de um episódio para o outro (seção 6, abaixo da tabela) | Sempre ligado no treino. Sai do play. Não segue o currículo | `env_cfg.py:200-205`, `knobs.py:517`, `knobs.py:520`, `events.py:336`, `event_manager.py:278-288` |
| Atrito do pé | Startup, uma vez por env | U(`DR.atrito_pe`) = U(0,3; 1,0), valor absoluto, só no eixo tangencial (eixo 0) e único para as 14 geoms de pé do env | Sempre | `env_cfg.py:185-187`, `knobs.py:508`, `geom.py:117-147` |
| Massa do tronco | Startup, uma vez por env | Soma U(`DR.massa_tronco`) = U(−1,0; 3,0) kg ao `torso_link` | Sempre | `env_cfg.py:188-190`, `knobs.py:509`, `body.py:285-326` |
| Viés do encoder | Startup, uma vez por env e por junta | U(−`DR.vies_encoder`; `DR.vies_encoder`) = U(−0,015; 0,015) rad | Sempre. O viés entra na observação `juntas` do ator | `env_cfg.py:191-192`, `knobs.py:510`, `joint.py:209-241`, `observacoes.py:394` |
| Viés do IMU | Reset | Viés do giro: U(−0,05; 0,05) rad/s por eixo (`DR.vies_imu_giro`). Inclinação do IMU: U(−1,0°; 1,0°) em x e em y, 0 em z (`DR.vies_imu_incl_graus`) | Trava `vies` (seção 5.9). Sem a trava, giro 0 e inclinação identidade. A mesma inclinação gira os raios do lidar (`percepcao_lidar.py:56`, `:63-64`, `observacoes.py:232`) | `eventos.py:492-509`, `knobs.py:521-522` |
| Atraso | Teto no startup e no currículo. Lag sorteado uma vez por episódio | Lag inteiro uniforme em [0; teto], por env. A ação e as observações `giro_base`, `gravidade`, `juntas`, `vel_juntas` e `mapa` usam o atraso. O atuador conta o teto em passos de física: teto × `decimacao` | O teto vale 0 no startup. O currículo o sobe com a `restricao` e o degrau global (seção 2.1) | `env_cfg.py:197-198`, `eventos.py:244-258`, `observacoes.py:55`, `cena.py:188-198`, `delay_buffer.py:260-277` |
| Ganho PD (kp e kd) | Reset | Fator U(`pd`) com `pd` = (`no_degrau(1; 0,9; g; 5)`, `no_degrau(1; 1,1; g; 5)`), para `g` = degrau global. Cada junta sorteia kp e kd independentes | Só no treino | `eventos.py:523-531`, `balanceador.py:368-369`, `balanceador.py:385`, `knobs.py:535`, `actuator.py:30-130` |
| Força do motor | Reset | Fator U(`forca_motor`). Pernas e cintura: (`no_degrau(1; 0,9; g; 5)`, `no_degrau(1; 1,1; g; 5)`). Braços: (`no_degrau(1; 0,8; g; 5)`, `no_degrau(1; 1,1; g; 5)`). Uma amostra por env e por alvo de atuador | Só no treino | `eventos.py:534-545`, `balanceador.py:386-387`, `knobs.py:536-537`, `actuator.py:192-255` |
| `solref[0]` | Reset | U(`no_degrau(0,02; 0,015; g; 5)`; `no_degrau(0,02; 0,03; g; 5)`), um valor por env, em pés, almofadas, dorsos, laje e caixa | Só no treino | `eventos.py:548-565`, `balanceador.py:388`, `knobs.py:539-540`, `env_cfg.py:177-180` |
| RFI (ruído de torque) | A cada passo | Torque = (2·U − 1)·s·`TAU_MAX` por junta e por env. O `s` vale `no_degrau(DR.rfi[0]; DR.rfi[1]; g; 5)` = de 0,05 a 0,10 | Só com a trava `subida` ligada. Sem ela, `s` = 0 | `eventos.py:605-615`, `balanceador.py:384`, `knobs.py:534`, `env_cfg.py:210` |
| Ruído de observação do ator | A cada passo | Uniforme, soma antes da escala: `giro_base` ±0,2, `gravidade` ±0,05, `juntas` ±0,01, `vel_juntas` ±1,5 (`DR.ruido_obs`) | Só no ator, com corrupção ligada | `observacoes.py:373-387`, `knobs.py:516`, `env_cfg.py:112-114` |
| Ruído do lidar | A cada passo | Alcance N(0; `Percepcao.ruido_alcance`) = 0,02 m. Direção N(0; 0,15°) por componente (`ruido_angulo_graus`). Perda de raio com probabilidade `Percepcao.perda` = 0,01. Giro de inclinação N(0; `ruido_incl_graus`) = 0,5° por eixo. O raio é válido se sobrevive à perda e se o alcance fica em `Percepcao.alcance` = (0,1; 6,0) m (`knobs.py:490`) | Sempre que o lidar está ligado | `percepcao_lidar.py:58-65`, `knobs.py:491-494` |
| Deriva da odometria | A cada passo | Ruído gaussiano com desvio `sqrt(deriva_odo²·\|Δp\| + pos_giro²·\|Δψ\|)` em x, y e z e `sqrt(rumo²·\|Δp\| + rumo_giro²·\|Δψ\|)` no rumo. Os quatro valores vão de (`deriva_odo`, `deriva_rumo`, `deriva_pos_giro`, `deriva_rumo_giro`) = (0,002; 0,002; 0,004; 0,0035) no degrau global 0 a (0,0075; 0,005; 0,008; 0,014) no degrau 5, por `no_degrau` | Sempre. O resultado transporta o mapa do lidar | `estimador.py:158-184`, `comando.py:247-250`, `comando.py:623-625`, `knobs.py:473-476`, `observacoes.py:227` |
| Atrito de almofada e laje | Reset | Almofada: U(`mu_min_pad`; `DR.atrito_pad[1]`) com `DR.atrito_pad[1]` = 1,0. Laje: U(`DR.atrito_laje`) = U(0,3; 1,0). Sem a trava `portao0`, os dois valem o valor fixo da cena | Trava `portao0` (depende do PEGAR). No ANDAR, o efeito físico existe só se a palma tocar o chão | `eventos.py:464-489`, `knobs.py:523-524`, `balanceador.py:371-373` |
| Força na caixa | Passo | Zerada no ANDAR | Não vale no ANDAR | `eventos.py:598-602` |

Os intervalos do empurrão: no startup, cada env sorteia o primeiro `time_left` em U(3,0; 7,0) s. Depois de cada disparo, o env sorteia o intervalo seguinte (`event_manager.py:278-286`). O reset do episódio não sorteia o intervalo de novo. O `reset` do gerenciador de eventos roda dentro de `_reset_idx` (`manager_based_rl_env.py:584`), mas só reescreve o `time_left` dos termos de intervalo que são classe com método `reset` (`event_manager.py:217-233`). O gerenciador põe nessa lista só os termos com `reset` (`event_manager.py:359-360`). O `empurrao` é uma função (`eventos.py:261`), então fica de fora. O comentário de `knobs.py:518-519` diz o mesmo. O tempo restante segue de um episódio para o outro. O primeiro empurrão de um episódio vem entre 0 e 7 s depois do reset, conforme o tempo que restava no fim do episódio anterior (derivado). Só o primeiro episódio de cada env depois do startup tem o primeiro empurrão entre 3 e 7 s.

⚠ Divergência: o comentário em `observacoes.py:54` diz que o atraso é de 0 a 1 passo. O código sorteia o lag entre 0 e `max_lag`, e o teto chega a `DR.atraso_refino[1]` = 3 passos (`knobs.py:513`, `balanceador.py:381-382`, `eventos.py:255-258`).

O empurrão grava o instante em `v3_t_empurrao` = `episode_length_buf · step_dt` (`eventos.py:269`). A âncora da seção 4.2 usa esse instante.

Não determinado pelo código lido: o atrito dos geoms do terreno (valor padrão do mjlab).

---

## 7. Recompensas e penalidades

Cada termo de recompensa devolve uma grandeza sem sinal por passo. O gerenciador do mjlab multiplica a grandeza pelo peso e por `step_dt` = 0,02 s, troca NaN e infinito por 0 e soma (`reward_manager.py:123-132`). O gerenciador pula um termo de peso 0 (`reward_manager.py:123-125`). A coluna "Peso" mostra o peso por segundo. O valor por passo é esse peso × 0,02 (derivado).

A máscara de tarefa vale zero fora de `tarefas`: o invólucro `_fim` usa `torch.where(mascara, valor, 0)` (`recompensas.py:361-365`, `curriculo.py:93-100`). O parâmetro `paga` também zera o termo depois da perda da caixa (`v3_perdeu`).

Um termo fica no ANDAR quando a tupla de tarefas dele contém "ANDAR" (`env_cfg.py:42-54`). São 27 termos. Os 6 termos que não entram no ANDAR são `cerca_sola`, `cerca_recuo`, `contato_laje`, `freio`, `sacudida` e `aperto`.

Fatores comuns usados nas linhas da tabela:

- `estilo` = `por_restricao(env, Recompensa.estilo_inicio)` = `0,25 + 0,75·restricao`, com `estilo_inicio` = 0,25 (`knobs.py:260`, `recompensas.py:453-459`). O fator vale para todas as tarefas. No ANDAR ele multiplica `giro_xy`, `vel_junta`, `acel_junta` e `energia`. Os três desvios (`desvio_bracos`, `desvio_cintura`, `desvio_quadril`) não o usam no ANDAR: chamam `por_restricao(env, estilo_inicio, so="CARREGAR")`, e o fator vale 1 nos envs fora do CARREGAR. Eles pagam o peso cheio no ANDAR desde a iteração 0 (`recompensas.py:707-722`, `knobs.py:260`).
- `cerca` = `por_restricao(env, 0,0, so="ANDAR")` = `restricao` nos envs ANDAR e 1 nos outros. O parâmetro `so` recebe o nome da tarefa (`recompensas.py:453-459`).
- `avanço` = `_avanco(env)` = `piso + (1 − piso) · p`. O `p` é a tenda da fração entregue do comando, já multiplicada por `g_marcha`: a do canal linear se há comando linear (‖c_xy‖ ≥ `limiar_canal`), senão a do giro se há comando de giro, senão 1. O `piso` vai de 1,0 no nível 0 do ANDAR a `Recompensa.piso_avanco` = 0,25 no nível máximo, linear em `no_degrau` (`recompensas.py:607-616`, `knobs.py:245`). Só o `passo` e o `altura_pe` o usam.
- `restricao` = ver seção 5.8.

### 7.1 Tabela dos termos registrados para o ANDAR

| Termo | Peso (knob, valor) | Fórmula resumida | Fatores | Vale zero quando |
|---|---|---|---|---|
| `rastreio_linear` | `Recompensa.rastreio_linear` = 2,0 (`knobs.py:236`) | Canal com comando: tenda da fração entregue × exp(−v⊥²/σ²) × `g_marcha`, com v⊥ a velocidade perpendicular instantânea. Canal sem comando: exp(−‖cmd − v‖²/σ²) | `g_marcha`, perda da caixa | `v3_perdeu`; tarefa ≠ ANDAR/CARREGAR (`recompensas.py:597-599`, `:374-409`) |
| `rastreio_angular` | `Recompensa.rastreio_angular` = 1,0 (`knobs.py:237`) | Idem no giro, sem o fator perpendicular | `g_marcha`, perda | idem (`recompensas.py:602-604`) |
| `passo` | `Recompensa.passo` = 0,5 (`knobs.py:244`) | Σ dos 2 pés de 1[contato = apoio do relógio] × `avanço` | `avanço`, perda | `v3_perdeu` (`recompensas.py:619-633`) |
| `vivo` | `Recompensa.vivo` = 1,0 (`knobs.py:248`) | 1 − `v3_perdeu` | nenhum | `v3_perdeu` (`recompensas.py:636-638`) |
| `altura_pe` | `Recompensa.altura_pe` = 1,0 (`knobs.py:249`) | exp(−Σ_pés (h − 0,12)²·tanh(2,0·‖v_xy‖)/0,0025) × `avanço`, com h descontada da subida do quadril | comando ≠ 0, `avanço`, perda | ‖comando‖ = 0 exato; `v3_perdeu` (`recompensas.py:641-656`) |
| `escorrego` | `Recompensa.escorrego` = −0,2 (`knobs.py:253`) | Σ_pés ‖v_xy do tornozelo‖·[pé em contato] | nenhum | nenhum pé em contato (`recompensas.py:659-664`) |
| `vz` | `Recompensa.vz` = −2,0 (`knobs.py:254`) | v_z² da base no mundo | nenhum | nunca (valor 0 só se v_z = 0) (`recompensas.py:667-669`) |
| `giro_xy` | `Recompensa.giro_xy` = −0,05 (`knobs.py:255`) | ω_x² + ω_y² da base no frame da base | `estilo` | ω_x = ω_y = 0 (`recompensas.py:672-674`) |
| `vel_junta` | `Recompensa.vel_junta` = −0,001 (`knobs.py:256`) | Σ q̇² nas 29 juntas | `estilo` | juntas paradas (`recompensas.py:677-679`) |
| `acel_junta` | `Recompensa.acel_junta` = −5e-8 (`knobs.py:257`) | Σ q̈² nas 29 juntas | `estilo` | sem aceleração (`recompensas.py:682-684`) |
| `taxa_acao` | `Recompensa.taxa_acao` = −0,01 (`knobs.py:258`) | Σ (a − a_anterior)² da ação crua | nenhum | ação igual à anterior (`recompensas.py:687-690`) |
| `energia` | `Recompensa.energia` = −2e-5 (`knobs.py:259`) | Σ \|τ·q̇\| com τ = `qfrc_actuator` | `estilo` | sem movimento ou sem torque (`recompensas.py:693-697`) |
| `limite_junta` | `Recompensa.limite_junta` = −5,0 (`knobs.py:262`) | Σ_j relu(\|q − c\|/m − 0,95)/(1 − 0,95) | nenhum | todas as juntas abaixo de 95 % do curso (`recompensas.py:700-704`) |
| `desvio_bracos` | `Recompensa.desvio_bracos` = −0,1 (`knobs.py:264`) | Σ \|q − q_de_pe\| nas 14 juntas de braço | `estilo` | braços na pose de pé (`recompensas.py:707-710`) |
| `desvio_cintura` | `Recompensa.desvio_cintura` = −1,0 (`knobs.py:265`) | Σ \|q − q_de_pe\| nas 3 juntas da cintura | `estilo` | cintura na pose de pé (`recompensas.py:713-716`) |
| `desvio_quadril` | `Recompensa.desvio_quadril` = −1,0 (`knobs.py:266`) | Σ \|q − q_de_pe\| em `hip_roll` e `hip_yaw` das duas pernas (4 juntas) | `estilo` | quadril na pose de pé (`recompensas.py:719-722`) |
| `tronco_plano` | `Recompensa.tronco_plano` = −20,0 (`knobs.py:267`) | ‖g_xy‖² da gravidade projetada no frame da base | nenhum | base nivelada (`recompensas.py:725-727`) |
| `altura_base` | `Recompensa.altura_base` = −10,0 (`knobs.py:268`) | (h_pelve − 0,78)² | nenhum | h_pelve = `alvo_base` = 0,78 m (`knobs.py:269`, `recompensas.py:730-733`) |
| `cerca_giro` | `Recompensa.cerca_giro` = −0,03965 (`knobs.py:271`) | relu(max(−q_yaw_E,0)° − 30) + relu(max(q_yaw_D,0)° − 30) | `cerca` | hip_yaw para dentro ≤ 30° (`recompensas.py:744-747`) |
| `cerca_tronco` | `Recompensa.cerca_tronco` = −0,04596 (`knobs.py:272`) | relu(incl_torso° − 15) | `cerca` | inclinação do torso ≤ 15° (`recompensas.py:750-758`) |
| `cerca_com` | `Recompensa.cerca_com` = −22,09035 (`knobs.py:273`) | relu(0,03 − margem_do_CoM), só com os dois pés em contato | `cerca` | um pé no ar, ou margem ≥ 0,03 m (`recompensas.py:761-764`) |
| `cerca_torque` | `Recompensa.cerca_torque` = −0,29615 (`knobs.py:274`) | Veja a fórmula na seção 7.2 | `cerca` | torques abaixo de 80 % do limite (`recompensas.py:767-772`) |
| `cerca_pelve` | `Recompensa.cerca_pelve` = −4,45279 (`knobs.py:275`) | relu(\|h_pelve − `pelve_de_pe`\| − 0,08) | `cerca` | pelve a até 0,08 m da altura de pé (`recompensas.py:775-788`) |
| `contato_fora` | `Recompensa.contato_fora` = −1,0 (`knobs.py:299`) | nº de contatos fora dos pés com força > 1 N + corpos em autocolisão + contatos robô-caixa > 1 N | nenhum | nenhum desses contatos (`recompensas.py:805-813`) |
| `progresso` | +1,0 fixo (`env_cfg.py:57`) | No ANDAR: 0, exceto no ANDAR final da fatia C (42/s). Veja a seção 7.2 | nenhum | ANDAR pleno; espera; ANDAR depois da perda (`recompensas.py:816-884`) |
| `perda_caixa` | −1,0 fixo (`env_cfg.py:57`) | `v3_perdeu_agora` × `Recompensa.perda_caixa` = 240 (`knobs.py:367`) | nenhum | passo sem perda recém-detectada (`recompensas.py:960-962`) |
| `terminacao` | `Recompensa.terminacao` = −300,0 (`knobs.py:366`) | 1 numa terminação que não é `time_out` | nenhum | passo sem terminação (`recompensas.py:965-968`) |

Os pesos das cercas vêm de `Recompensa` (`knobs.py:271-275`). Os limites das cercas são `lim_giro_graus` = 30,0 (`knobs.py:278`), `lim_tronco_graus` = 15,0 (`knobs.py:279`), `lim_com` = 0,03 (`knobs.py:280`), `lim_torque` = 0,8 (`knobs.py:281`), `banda_torque` = 0,2 (`knobs.py:282`) e `lim_pelve` = 0,08 (`knobs.py:283`).

### 7.2 Fórmulas completas dos termos principais

Notação: `c` = comando do twist (vx, vy, wz). `v` = velocidade da base no frame da base. `v̄` = `twist.v_filt`. `σ` = `Recompensa.sigma_rastreio` = 0,5 (`knobs.py:238`). `limiar` = `Comando.limiar_canal` = 0,02 (`knobs.py:138`).

**Velocidade filtrada `v̄`.** O twist atualiza `v_filt` a cada passo: `v_filt += (1 − exp(−dt/τ))·(v − v_filt)`, com τ = `Comando.tau_fracao` = 0,4 s (`knobs.py:136`, `comando.py:854`, `comando.py:858`). O vetor `v` é (vx, vy, wz) da base. O filtro zera no reset (`comando.py:886`). A recompensa lê o filtro atualizado no passo anterior, porque o `compute` do twist roda depois das recompensas (`manager_based_rl_env.py:440`, `manager_based_rl_env.py:456`).

**Fator de marcha `g`.** O twist calcula `g_marcha = clamp(2·q̄ − 1, 0, 1)`. O `q̄` é o acerto do relógio dos dois pés, filtrado com a mesma τ (`comando.py:868-871`). O acerto é `n_passo(contato, fase, fase_apoio, on).mean(-1)` (`comando.py:868`). Quando o relógio liga (`on` e não estava ligado), `q̄` vai a 0,5 (`comando.py:869`). O `q̄` começa em 0,5 no reset (`comando.py:887`).

**`rastreio_linear` (ANDAR).** Seja `n = ‖c_xy‖`. A recompensa vale (`recompensas.py:374-409`):

- Se `n ≥ limiar`: `tenda · exp(−v⊥²/σ²) · g`, com `tenda = clamp(1 − |1 − p|, 0, 1)`, `p = (v̄_xy·c_xy)/n²` na velocidade filtrada e `v⊥² = max(‖v_xy‖² − (v_xy·c_xy/n)², 0)` na velocidade instantânea do passo (`recompensas.py:400-407`). A perpendicular instantânea nota o vaivém lateral de uma passada, que a média filtrada apagava. Parado vale 0. A entrega exata vale 1. O dobro vale 0.
- Se `n < limiar`: `exp(−‖c_xy − v_xy‖²/σ²)`, com a velocidade instantânea (`recompensas.py:389-391`, `recompensas.py:409`).

**`rastreio_angular` (ANDAR).** Seja `n = |c_z|`.

- Se `n ≥ limiar`: `clamp(1 − |1 − p|, 0, 1) · g`, com `p = v̄_z·sinal(c_z)/n` (`recompensas.py:406-407`).
- Se `n < limiar`: `exp(−(c_z − wz)²/σ²)` (`recompensas.py:392-393`, `recompensas.py:409`).

Os dois canais são decididos separadamente: um env com comando só linear paga a tenda no linear e a deriva no giro.

**`passo`.** O relógio de marcha tem período `Comando.periodo_marcha` = 0,8 s (`knobs.py:140`). A fase é `remainder(_fase_t / 0,8, 1)`, e `_fase_t` soma `dt` só quando `on` (`comando.py:872-873`). `on` = `‖c‖ > Comando.limiar_marcha` = 0,1, com a norma dos três canais (`knobs.py:141`, `comando.py:863`). O pé esquerdo usa a fase e o direito usa `fase + 0,5` (módulo 1). O apoio vale quando `fase < Comando.fase_apoio` = 0,55 (`knobs.py:139`) ou quando `on` é falso. A recompensa é `Σ_pés [contato == apoio]` (`recompensas.py:75-82`). O máximo por passo é 2, e o peso 0,5 dá 1,0 por segundo. O valor vale × `avanço`: marchar no lugar com comando paga o `piso` (0,25 no nível máximo), e atravessar paga o valor inteiro. Sem comando, o `avanço` vale 1. O `passo` lê a fase do passo anterior (`recompensas.py:619-633`, `comando.py:859-862`).

**`altura_pe`.** `h` é a altura de cada pé sobre o terreno, do sensor `altura_pes` (`recompensas.py:651`, `cena.py:392-393`). A fórmula é `exp(−Σ_pés (h − alvo_pe)² · tanh(tanh_pe · ‖v_xy‖) / std_pe)`, com `alvo_pe` = 0,12 m (`knobs.py:250`), `tanh_pe` = 2,0 (`knobs.py:252`) e `std_pe` = 0,0025 m² (`knobs.py:251`). O `v_xy` é a velocidade horizontal do tornozelo no mundo (`recompensas.py:652`, `recompensas.py:90-97`). O termo vale 0 quando `‖c‖` = 0 exato (`recompensas.py:654-656`). O termo vale × `avanço`. A altura `h` perde o desconto da subida do quadril pela inclinação da pelve: `h = altura_pes − (z_quadril − z_quadril_no_frame_da_pelve)`, em cada lado (corpos `left_hip_pitch_link` e `right_hip_pitch_link`). O desconto vale 0 com a pelve nivelada. O corpo que se inclina para erguer o pé não rende, e a perna que encurta rende (`recompensas.py:649-651`). O `alvo_pe` subiu de 0,10 para 0,12 m porque o sensor mede o tornozelo, a 3,5 cm do chão no apoio (comentário de `knobs.py:250`).

**`vivo`.** Vale 1 no ANDAR pleno. Vale 0 no ANDAR depois da perda da caixa (`recompensas.py:85-87`).

**`tronco_plano` e `altura_base`.** `tronco_plano = gx² + gy²` com `g` = gravidade projetada no frame da base (`recompensas.py:725-727`). `altura_base = (h_pelve − 0,78)²`, com `h_pelve` do sensor `altura_pelve` (`recompensas.py:730-733`).

**`limite_junta`.** `Σ_j relu(|q_j − c_j|/m_j − 0,95)/(1 − 0,95)`, com `c` = centro e `m` = meio curso de `joint_pos_limits` e 0,95 = `Recompensa.limite_mole` (`potencial.py:70-85`, `recompensas.py:700-704`, `knobs.py:263`).

**`cerca_torque`.** `f = |τ|/τ_max` por junta, com `τ_max` de `TAU_MAX` (`knobs.py:286-292`, `knobs.py:572`). Para as juntas fora dos pares: `Σ relu(f − 0,8)`. Para cada par (cintura pitch e roll, tornozelo esquerdo pitch e roll, tornozelo direito pitch e roll) (`knobs.py:293-297`): `relu((|τ_a| + |τ_b|)/50 − 0,8)`, com `tau_par` = 50 (`knobs.py:298`). O total é dividido por `banda_torque` = 0,2 (`recompensas.py:182-193`).

**`cerca_com`.** O CoM é o centro de massa da subárvore da pelve (o robô inteiro, sem a caixa no ANDAR) (`recompensas.py:419-435`). A margem é a distância com sinal do CoM às arestas do fecho convexo dos 8 cantos das solas (4 cantos por pé), positiva dentro (`recompensas.py:316-336`, `recompensas.py:473-479`). O termo vale `relu(lim_com − margem)` quando os dois pés tocam o chão e 0 caso contrário (`recompensas.py:176-179`).

**`progresso` no ANDAR.** A renda por segundo é `renda_etapa·etapa·h + renda_phi·Φ + renda_conquista·conquistas`, com `renda_etapa` = 6,0 (`knobs.py:339`), `renda_phi` = 3,0 (`knobs.py:338`) e `renda_conquista` = 2,0 (`knobs.py:343`) (`recompensas.py:286-291`). No ANDAR, a etapa vale (`recompensas.py:851-853`):

- 0 quando `v3_perdeu` ou quando a pega não está fechada. Isso cobre o ANDAR pleno, a espera e o ANDAR depois da perda. Φ pago = 0 e as conquistas valem 0.
- 7 no ANDAR final da fatia C (`v3_bota_fechada` verdadeiro). Aí `h_ef` = 1 e as conquistas valem 0. A renda é `6,0 · 7 · 1` = 42 por segundo, ou 0,84 por passo (derivado).

**`perda_caixa`.** O custo é `v3_perdeu_agora · 240`, com peso −1 (`recompensas.py:311-313`, `recompensas.py:960-962`). O buffer `v3_perdeu_agora` é escrito pela cadeia depois das recompensas e só vale a partir do passo seguinte (`comando.py:396-402`, `comando.py:411`). Um env que perde a caixa no CARREGAR paga o custo (−240 × 0,02 = −4,8 por passo, derivado) já com a tarefa ANDAR publicada, porque a tarefa troca no mesmo `compute` em que a perda é escrita (`comando.py:411-420`). O ANDAR pleno nunca paga esse termo, porque `_perda` exclui `tar == ANDAR` (`comando.py:410`).

**`terminacao`.** O termo vale 1 quando `terminated & ~time_outs` (`recompensas.py:214-216`). O custo por passo é −300 × 0,02 = −6,0 (derivado).

⚠ Divergência: a docstring de `n_terminacao` em `recompensas.py:215` diz "Peso −200 no knob". O knob `Recompensa.terminacao` vale −300,0 (`knobs.py:366`).

**`contato_fora`.** `forca` = norma da força de cada contato do sensor `nao_pes_chao` (corpos do robô, fora os tornozelos, contra o terreno). O termo conta quantos contatos passam de `Recompensa.contato_fora_N` = 1,0 N, soma os corpos em autocolisão e soma os contatos do sensor `caixa_robo` acima de 1,0 N (`recompensas.py:206-211`, `recompensas.py:805-813`, `knobs.py:300`). No ANDAR a janela da pega é falsa, então os contatos com a caixa sempre contam (`recompensas.py:812-813`).

---

## 8. Terminações

O gerenciador de terminações do mjlab separa os termos em dois grupos: `time_out` marca truncamento e os outros marcam terminação (`termination_manager.py:102-113`). O cfg tem cinco termos, nesta ordem (`env_cfg.py:137-147`).

| Termo | Tarefas | Condição exata | Vale no ANDAR |
|---|---|---|---|
| `time_out` (`fim_da_cadeia`) | todas (sem máscara) | `episode_length_buf ≥ max_episode_length`, ou `v3_fim`, ou a travessia do terreno do ANDAR (abaixo) (`terminacoes.py:53-72`) | Sim |
| `queda` | todas | No ANDAR: `incl > Terminacao.incl_marcha` ou `altura_pelve < Terminacao.base_min`, onde `incl = acos(clamp(−g_z_base, −1, 1))` (`terminacoes.py:35-50`, `terminacoes.py:97`) | Sim |
| `caixa_perdida` | PEGAR, CARREGAR, BOTAR | `v3_perdeu_agora & ~caixa_perdida_vira_andar` (`terminacoes.py:75-83`) | Não (máscara) |
| `chegada` | BOTAR | Primeiro contato da caixa com a laje com `−vz_ant > limite` (`terminacoes.py:128-151`) | Não |
| `batente` | todas | Alguma das 29 juntas com `\|q − c\|/m > batente_frac`, ou `\|hip_yaw\| > hip_yaw_max`, e `v3_degrau ≥ degrau_batente` (`terminacoes.py:154-183`) | Sim, com `v3_degrau ≥ 1` |

Os valores:

- Travessia: o `time_out` também dispara quando o env está na fatia ANDAR, `terrain_types` ≠ 0, a distância xy de `twist.pose_ult` à origem da célula passa de `Terreno.dist_travessia` = 3,5 m (`knobs.py:442`) e `episode_length_buf` > 2 (`terminacoes.py:69-72`). É um `time_out`: não paga `terminacao` e entra no PPO como truncamento. A guarda `episode_length_buf > 2` existe porque, logo depois do reset, a posição lida ainda pode ser a velha. A leitura é do `pose_ult` (seção 11.6), e não de `root_link_pos_w`. A área da travessia é um círculo de 3,5 m em torno da origem da célula. Na escada, 3,5 m fica no meio da borda plana, que vai de 3,0 a 4,0 m do centro (seção 11.5; comentários de `knobs.py:442` e `terminacoes.py:61-64`).
- `max_episode_length` = `ceil(episode_length_s / step_dt)` = `ceil(20,0 / 0,02)` = 1000 passos (`manager_based_rl_env.py:281-283`, `knobs.py:160`, `env_cfg.py:105`).
- `Terminacao.incl_marcha` = 0,8 rad (`knobs.py:375`).
- `Terminacao.base_min` = 0,20 m (`knobs.py:376`). A altura é a do sensor `altura_pelve` sobre o terreno, e não sobre a origem do mundo (`terminacoes.py:99-101`).
- `Terminacao.batente_frac` = 1,0 e `Terminacao.hip_yaw_max` = 1,0 rad (`knobs.py:382-383`). O comparador é estrito (`>`) (`terminacoes.py:162-164`).
- `Refino.degrau_batente` = 1 (`knobs.py:547`).
- `Cadeia.caixa_perdida_vira_andar` = True (`knobs.py:161`). Por isso `caixa_perdida` nunca dispara.

O `queda` do ANDAR usa a inclinação da base (raiz do robô), em radianos, medida pela gravidade projetada (`terminacoes.py:97`).

### 8.1 O que acontece depois

Depois de uma terminação ou de um `time_out`, o env reseta no mesmo passo (`manager_based_rl_env.py:445-447`). O que acontece com o ANDAR:

1. A recompensa `terminacao` paga −6,0 em terminação (seção 7). O `time_out` não paga (`recompensas.py:214-216`).
2. O twist fecha o segmento aberto. Em terminação, a nota do segmento vale 0 e o segmento entra no registro mesmo sem tempo depois da âncora. No `time_out` (inclusive o da travessia), a nota vale o cálculo normal da seção 5.2 (`comando.py:882-883`, `comando.py:789-790`).
3. O reset sorteia fatia, terreno, base e as perturbações de reset (seção 2).
4. O PPO soma `γ · min(V̄_g, 0)` à recompensa quando a terminação não é `time_out`, ou quando `v3_perdeu_agora` é verdadeiro. `V̄_g` é a média do valor do grupo da vantagem no rollout anterior. `γ` = `self.gamma`. O valor de γ no cfg do mjlab é 0,99 (`algoritmo.py:114-135`, `rl_cfg.py:37`). Para o ANDAR sem alvo, o grupo é o 0 (`algoritmo.py:106-112`). A soma só vale quando `fatia_alvo` e `v_grupo` existem (`algoritmo.py:127`).
5. O `time_out` entra no PPO como truncamento, e não como queda (`env_cfg.py:138`, `termination_manager.py:108-111`).

O mjlab registra a contagem de terminações do lote em `Episode_Termination/<termo>` (`termination_manager.py:95-97`).

---

## 9. O que o ator e o crítico observam e é próprio do ANDAR

O ator tem 10 termos, nesta ordem (`observacoes.py:389-400`). Cada termo tem canais, escala e histórico de 5 passos, salvo o `mapa`, que não tem histórico (`knobs.py:58-63`).

| Termo | Canais | Escala | Ruído do ator | Atraso | Valor no ANDAR | Fonte |
|---|---|---|---|---|---|---|
| `giro_base` | 3 | 0,2 | ±0,2 | Sim | giro da base + viés do IMU | `observacoes.py:202-204` |
| `gravidade` | 3 | 1,0 | ±0,05 | Sim | gravidade projetada, girada pela inclinação fixa do IMU | `observacoes.py:207-209` |
| `comando` | 3 | 1,0 | não | não | o comando do twist | `observacoes.py:392` |
| `fase` | 2 | 1,0 | não | não | seno e cosseno de `2π·v3_fase_marcha` | `observacoes.py:312-314` |
| `juntas` | 29 | 1,0 | ±0,01 | Sim | `joint_pos_rel` com viés do encoder | `observacoes.py:394` |
| `vel_juntas` | 29 | 0,05 | ±1,5 | Sim | `joint_vel_rel` | `observacoes.py:395` |
| `acao` | 29 | 1,0 | não | não | última ação | `observacoes.py:396` |
| `tarefa` | 6 | 1,0 | não | não | one-hot, com 1 na posição 0 (ANDAR) | `observacoes.py:135-137` |
| `caixa` | 11 | 1,0 | não | não | zeros no ANDAR | `observacoes.py:196-199` |
| `mapa` | 882 | 1,0 | não | Sim | mapa de elevação do lidar | `observacoes.py:154-157` |

O atraso vale para os termos de `_ATRASO` (`observacoes.py:55`, `observacoes.py:382-386`). A capacidade do buffer de atraso é `DR.atraso_refino[1]` = 3 passos (`knobs.py:513`, `observacoes.py:371`). O teto efetivo segue a seção 2.1. O pipeline do mjlab é: calcular, ruído, corte, escala, atraso, histórico (`observation_manager.py:325-360`).

O que é próprio do ANDAR:

- O `comando` é o comando sorteado da seção 3. No ANDAR de terreno, o vx vem do sorteio e o wz vem do rumo, que muda a cada passo (seção 3.4). Nos trechos ANDAR das fatias B e C, o comando é zero (seção 1.1).
- O `tarefa` vale [1, 0, 0, 0, 0, 0].
- O `caixa` vale zero: a função de caixa estimada zera a saída inteira onde a tarefa é ANDAR (`observacoes.py:199`).
- A `fase` do relógio de marcha anda só com comando acima de `limiar_marcha` e dá a volta em 0,8 s (`comando.py:863`, `comando.py:872-873`, `knobs.py:140-141`).
- O `mapa` é o recorte de `Percepcao.n_recorte` = 21 × 21 células de `Percepcao.celula` = 0,10 m, com 441 alturas e 441 confianças (`knobs.py:484-485`, `percepcao_lidar.py:192-200`). Altura cortada em ±`altura_clip` = 1,0 m (`knobs.py:488`). Com o lidar desligado, o termo `mapa` sai dos dois grupos (`observacoes.py:401-402`).

O crítico usa os mesmos termos do ator, sem ruído e sem atraso, e o `juntas` sem o viés do encoder (`observacoes.py:403-404`). Depois entram os termos privilegiados (`observacoes.py:405-418`):

| Termo do crítico | Canais | Valor no ANDAR | Fonte |
|---|---|---|---|
| `vel_lin_base` | 3, histórico 5 | velocidade linear do sensor `robot/imu_lin_vel` | `observacoes.py:406-407` |
| `caixa_verdade` | 7 | zeros no ANDAR | `observacoes.py:267-273` |
| `alvo_verdade` | 3 | zeros com o índice de alvo −1. Depois da perda da caixa, com o índice em B3, vale `v3_alvo_w` no frame da base, até o B3 fechar | `observacoes.py:276-279` |
| `alvo_macro` | 11 | one-hot: slot 10 ("nenhum") no ANDAR pleno e na espera; slot 8 no ANDAR final da fatia C; slot 9 depois da perda | `observacoes.py:282-295` |
| `referencia` | 8 | zeros com o índice de alvo −1. Depois da perda da caixa, com o índice em B3, as palmas valem 0, a pelve vale `pelve_de_pe` e o tronco vale 0, até o B3 fechar | `observacoes.py:321-324`, `alvos.py:145-154`, `observacoes.py:116-130` |
| `niveis` | 4 | nível/máximo de ANDAR, PEGAR, CARREGAR, BOTAR; igual em todo env | `observacoes.py:327-335` |
| `degrau` | 1 | `v3_degrau / 5` | `observacoes.py:338-340` |
| `massa` | 1 | `v3_massa / 5` = 1,0/5 = 0,2 | `observacoes.py:343-345`, `observacoes.py:61` |
| `forca_caixa` | 1 | força robô-caixa / 100 | `observacoes.py:348-356`, `observacoes.py:62` |
| `privilegiados` | 4 | `marcha_on`, `v3_h` (1,0 no ANDAR), lacuna (0 no ANDAR), z da etapa (0 com o índice de alvo −1; depois da perda, com o índice em B3, o `z` do B3) | `observacoes.py:298-318` |
| `tarefa_critico` | 6 | one-hot da tarefa; é o último termo | `observacoes.py:417` |

O grupo da vantagem do PPO usa o crítico. O ANDAR com o slot "nenhum" do `alvo_macro` forma o grupo 0. O ANDAR final (slot 8) forma o grupo `1 + 0·11 + 8` = 9. O ANDAR depois da perda (slot 9) forma o grupo 10 (`algoritmo.py:98-112`). O PPO normaliza a vantagem dentro de cada grupo (`algoritmo.py:34-60`, `algoritmo.py:158-159`). O grupo 0, quando tem menos de 2 amostras, fica com a vantagem crua (`algoritmo.py:39-45`, `algoritmo.py:54-56`). Os grupos 9 e 10, quando têm menos de 2 amostras, usam a média e o desvio de todas as amostras do lote com grupo diferente de 0, se essas amostras são 2 ou mais; senão ficam com a vantagem crua (`algoritmo.py:47-56`).

⚠ Divergência: o cabeçalho de `algoritmo.py:14-16` diz que o grupo é a tarefa corrente. Com `fatia_alvo` definida, o grupo é a tarefa mais o slot do `alvo_macro` (`algoritmo.py:98-112`, `runner.py:61-62`).

O normalizador de observação do mjlab é ligado (`rl_cfg.py:16`, `rl_cfg.py:26`), com o piso de desvio `PISO_DESVIO` = 0,05 (`algoritmo.py:66`). Os termos fixos (`tarefa`, `mapa`, `alvo_macro`, `niveis`, `degrau`, `massa`, `tarefa_critico`, `privilegiados`: desvio 1,0; `caixa`, `caixa_verdade`, `alvo_verdade`, `referencia`: desvio 0,1; `forca_caixa`: desvio 0,3) ficam fora do normalizador empírico (`runner.py:32-34`).

---

## 10. Métricas

Os termos de métrica e de log valem para todos os envs, e não só para o ANDAR. O mjlab calcula a média sobre os envs que resetam no passo (`metrics_manager.py:115-126`, `reward_manager.py:106-110`). Nenhuma métrica filtra pela tarefa ANDAR, salvo as de caixa (`metricas.py:235-243`).

### 10.1 Métricas de episódio (`Episode_Metrics/<nome>`)

O mjlab registra cada métrica de `metricas.termos` com a regra de agregação do cfg: `mean` = soma sobre o episódio / nº de passos; `last` = valor do último passo; `max` = pico (`metrics_manager.py:20-37`, `metrics_manager.py:115-126`).

| Métrica | O que mede | Agregação | Fonte |
|---|---|---|---|
| `tempo_de_voo` | Tempo médio no ar dos pés que estão no ar | mean | `metricas.py:93-101` |
| `pico_de_altura` | Altura de pico do pé no balanço, medida no pouso | mean | `metricas.py:104-133` |
| `velocidade_de_escorrego` | Velocidade horizontal do pé apoiado | mean | `metricas.py:136-142` |
| `velocidade_de_junta` | RMS de `joint_vel` nas 29 juntas | mean | `metricas.py:145-149` |
| `fracao_do_curso` | Maior `\|q − c\|/m` entre as 29 juntas | max | `metricas.py:152-165` |
| `forca_de_pouso` | Módulo da força no instante do pouso | mean | `metricas.py:168-173` |
| `passo_medio` | Comprimento médio do passo no toque, com `‖c_xy‖ ≥ limiar_canal` = 0,02 | last | `metricas.py:176-219`, `metricas.py:549-553` |
| `fracao_mapa_visto` | Fração das células do recorte com confiança > `Percepcao.limiar_visto` = 0,5 | mean | `metricas.py:222-232`, `knobs.py:503` |
| `phi_<alvo>`, `tempo_<alvo>`, `inclinacao_caixa_p3`, `forca_robo_caixa`, `deslocamento_caixa_laje`, `velocidade_de_chegada`, `deslocamento_depois_do_apoio`, `idade_leituras`, `fracao_avisos`, `fracao_fora_da_plataforma`, `aceleracao_da_caixa`, `erro_p3` | Medidas da caixa. Cada uma tem máscara de tarefa, de alvo ou de fatia, e no ANDAR pleno vale 0 ou não registra evento. A exceção é `forca_robo_caixa`, que soma a força do sensor `caixa_robo` sem máscara (`metricas.py:338-345`) | vários | `metricas.py:557-570` |

As métricas `phi_<alvo>` e `tempo_<alvo>` usam a máscara "tarefa com caixa" (`metricas.py:239-243`, `metricas.py:260`). A métrica `idade_leituras` zera onde não há caixa (`metricas.py:507`).

Alguns termos de métrica também gravam `Episodio/<nome>` em `extras["log"]`: um valor por episódio, no reset, só dos envs que tiveram o evento (`metricas.py:81-90`). São eles `phi_<alvo>`, `tempo_<alvo>`, `erro_p3_<eixo>`, `velocidade_de_chegada`, `deslocamento_depois_do_apoio`, `fracao_fora_da_plataforma` e `aceleracao_da_caixa` (`metricas.py:268`, `:293`, `:324`, `:395`, `:422`, `:453`, `:487`). Todos são medidas da caixa.

### 10.2 Métricas do comando (`Metrics/twist/<nome>`)

O gerenciador de comando registra as métricas do twist ao resetar, com média sobre os envs resetados (`command_manager.py:99-107`, `command_manager.py:241-247`).

| Métrica | O que mede | Fonte |
|---|---|---|
| `error_vel_xy` | Σ do erro linear ‖c_xy − v_xy‖ por passo, dividido por `reamostra_s[1] / step_dt` = 8,0 / 0,02 = 400 | `velocity_command.py:61-69`, `knobs.py:137` |
| `error_vel_yaw` | Idem no giro | `velocity_command.py:70-73` |
| `razao_marcha` | `1 − Σ‖c_xy − v_xy‖ / Σ‖c_xy‖`, só nos passos com `‖c_xy‖ > limiar_canal`; 0 sem comando | `comando.py:845-850` |
| `acerto_relogio` | Média do acerto do relógio de marcha nos passos com `on` | `comando.py:875-877` |
| `eficiencia_min`, `eficiencia_media`, `segmentos` | Por segmento com `Σ‖c_xy‖·dt ≥ Comando.pedido_min_segmento` = 0,5: eficiência = `Σ(v·ĉ)dt / Σ‖c_xy‖dt`. Mínimo, média e contagem | `comando.py:795-804`, `knobs.py:147` |

### 10.3 Log do currículo (`Curriculum/v3/<chave>`)

O termo `v3` devolve um dicionário, e o mjlab o publica com o prefixo `Curriculum/v3/` (`curriculo.py:127-143`, `curriculum_manager.py:97-103`). As chaves do ANDAR:

| Chave | Valor |
|---|---|
| `frac_ANDAR`, `frac_B`, `frac_C` | Fração de cada fatia (seção 1.2) |
| `s_ANDAR_andar`, `s_ANDAR_terreno` | Nota `s` de cada célula do ANDAR |
| `nivel_ANDAR` | Nível de velocidade |
| `nivel_terreno_ANDAR` | Piso da média das oito linhas por coluna (seção 5.6) |
| `linha_<coluna>` | Linha de terreno real do ANDAR em cada coluna de 1 a 8 (`escada`, `escada_inv`, `rampa`, `rampa_inv`, `escada_aleatoria`, `irregular`, `obstaculos`, `caixas`) (`curriculo.py:139-140`) |
| `degrau_ANDAR` | Degrau de refino do ANDAR |
| `degrau_global` | Menor degrau entre as quatro tarefas |
| `restricao` | A restrição da seção 5.8 |
| `atraso_max`, `mu_min_pad` | Teto do atraso e borda inferior do atrito da almofada |

Fonte: `curriculo.py:127-143`.

### 10.4 Log de recompensa e de terminação

- `Episode_Reward/<termo>` = média, sobre os envs resetados, da soma do termo no episódio dividida por `max_episode_length_s` = 20,0 s (`reward_manager.py:106-112`).
- `Episode_Termination/<termo>` = contagem de envs terminados por termo no lote (`termination_manager.py:93-97`).

---

## 11. Como o treino roda: dinâmica dos sorteios

Esta seção junta os pontos de dinâmica do treino do ANDAR. Cada número leva "(derivado)" e diz a base. As simulações rodam as funções do código do commit `69e8b6b` (`sorteia_velocidade`, `fracoes`, `escolhe_celula_terreno`), com `torch.Generator().manual_seed(0)` novo para cada nível ou caso e 200 000 sorteios. Os números dos pontos 11.1 e 11.2 são os mesmos do arquivo `04-comando-de-velocidade.md`, seção 4.6. Os níveis 0, 1, 2, 4 e 7 mostram o começo, o meio e o fim da faixa. O terreno do ANDAR só abre a partir do nível 2 (seção 5.5).

### 11.1 A mistura do comando do plano por nível

A simulação chama `sorteia_velocidade("andar", n, "ANDAR", ...)` 200 000 vezes por nível. A base de cada fração é o total de sorteios, com os parados dentro dela (derivado: simulação).

| Nível | Parado (vx, vy, wz = 0) | `\|vy\| > \|vx\|` | `\|wz\| ≥ giro_min` (0,1) | `vx < 0` |
|---|---|---|---|---|
| 0 | 0,150 | 0,263 | 0,601 | 0,376 |
| 1 | 0,119 | 0,332 | 0,675 | 0,413 |
| 2 | 0,112 | 0,279 | 0,719 | 0,420 |
| 4 | 0,110 | 0,239 | 0,719 | 0,350 |
| 7 | 0,108 | 0,194 | 0,719 | 0,281 |

Os parados passam de `andar_parado` = 0,10 porque o corte de `linear_min` zera o linear de alguns sorteios, e o corte de `giro_min` zera o giro. O nível 0 tem `frente` = 0,3: o linear com módulo abaixo de 0,1 é frequente, e o `wz` com módulo abaixo de 0,1 sai em 1/3 dos sorteios (0,1 / 0,3). Com o nível, o `frente` sobe até 1,0 e o `tras` fica em 0,5, então a fração com `vx < 0` cai depois do nível 2. Só 0,28 dos sorteios do nível 7 anda para trás. O `lado` fica em 0,3, e a fração com `|vy| > |vx|` cai de 0,33 (nível 1) a 0,19 (nível 7).

### 11.2 O comando do ANDAR de terreno por nível

O ANDAR de terreno recebe só `vx = frente · r`, com `r = sqrt(U)` (seção 3.1). O twist reamostra o comando a cada U(3; 8) s. Um episódio de 20 s tem de 3 a 7 comandos.

A simulação (derivado) usa 200 000 episódios por nível. Cada episódio tem 8 segmentos de duração U(3; 8) s, sorteados antes dos comandos, com o mesmo gerador. O comando de cada segmento vem de `sorteia_velocidade("terreno", n, "ANDAR", ...)`. Os 20 s cortam o último segmento. Hipóteses: o robô entrega a fração `f` do `vx` pedido, sem demora, em linha reta, e não cai; a base parte da origem da célula. A simulação não inclui o deslocamento de partida (±0,5 m em x e em y) nem o rumo. A distância de referência é `dist_travessia` = 3,5 m (`knobs.py:442`).

| Nível | `frente` (m/s) | Distância pedida em 20 s (m) | Pedido ≥ 3,5 m | Mediana do tempo até 3,5 m, `f` = 1,0 (s) | Passa de 3,5 m em 20 s, `f` = 0,7, sobre todos os episódios | Idem, sobre as visitas | Mediana do tempo até 3,5 m, `f` = 0,7 (s) |
|---|---|---|---|---|---|---|---|
| 0 | 0,3 | 3,47 | 0,514 | 16,1 | 0,055 | 0,107 | 19,1 |
| 1 | 0,4 | 4,73 | 0,817 | 13,2 | 0,451 | 0,551 | 16,9 |
| 2 | 0,5 | 5,95 | 0,922 | 10,8 | 0,728 | 0,790 | 14,7 |
| 4 | 0,7 | 8,38 | 0,978 | 7,7 | 0,922 | 0,943 | 11,1 |
| 7 | 1,0 | 11,99 | 0,993 | 5,3 | 0,979 | 0,986 | 7,7 |

A base de cada coluna:

- "Distância pedida" é `Σ vx · duração` em 20 s, a mesma conta que o código faz com `soma_cmd_marcha · step_dt` (`comando.py:847`, `eventos.py:155`). Ela não depende de `dist_travessia`. "Pedido ≥ 3,5 m" é a fração sobre todos os episódios. Nesta simulação, é também a fração das visitas sem queda.
- As medianas de tempo são sobre os episódios que passam de 3,5 m, e não sobre todos.
- Com `f` = 1,0, todo episódio que pede 3,5 m passa de 3,5 m: a taxa por visita vale 1,000 e a taxa sobre todos os episódios vale a coluna "Pedido ≥ 3,5 m".
- A taxa por visita é a que o currículo usa (seção 11.6). Nesta simulação, a visita é o episódio com pedido ≥ 3,5 m (não há queda). A taxa sobre todos os episódios divide pelos 200 000; a taxa por visita divide pelos que pedem 3,5 m.
- Com `f` = 0,7, a taxa por visita sobe de 0,107 (nível 0) a 0,551 (nível 1) e 0,790 (nível 2). Ela passa de `zona_t_alvo` = 0,6 só a partir do nível 2. A linha das colunas sobe quando a taxa por visita passa de 0,6 (seção 5.6).
- A duração média do episódio de terreno, sobre todos os episódios, com 20 s para os que não passam de 3,5 m (derivado): com `f` = 1,0, 18,0 / 14,8 / 12,2 / 9,0 / 6,6 s; com `f` = 0,7, 19,9 / 18,6 / 16,3 / 12,4 / 9,0 s (níveis 0, 1, 2, 4, 7).

O comentário de `nivel_abre_terreno` (`knobs.py:412`) dá a taxa por visita com `f` = 0,7: ~79 % no nível 2, ~55 % no nível 1 e ~11 % no nível 0. A simulação dá 0,790, 0,551 e 0,107.

### 11.3 Quando a célula `terreno` abre e quantos episódios são de terreno

A célula abre no fecho em que o nível do ANDAR é ≥ 2 e `s_andar ≥ 0,75` (seção 5.5). O mínimo é o fecho 101, ou 2 424 passos da política (derivado: 2 × 50 fechos seguidos mais um fecho; 24 passos por fecho).

A fração dos episódios do ANDAR que são de terreno é `F_terreno = 0,10 + 0,80 · (D_t + 0,02) / (D_a + D_t + 0,04)`, com `D_t = 1 − L̄ / 12` (L̄ é a média das oito linhas) e `D_a = max(0,75 − s_andar, 0)` (derivado: a função `fracoes` do código, duas células, `piso_sub` = 0,10 e `eps` = 0,02).

| `s_andar` | L̄ = 0 | 1 | 3 | 6 | 9 | 12 |
|---|---|---|---|---|---|---|
| 0,75 (déficit 0) | 0,885 | 0,883 | 0,880 | 0,870 | 0,845 | 0,500 |
| 0,60 | 0,786 | 0,777 | 0,755 | 0,703 | 0,591 | 0,184 |
| 0,40 | 0,687 | 0,673 | 0,640 | 0,567 | 0,438 | 0,141 |

Logo depois da abertura, L̄ = 0 e `s_andar ≥ 0,75`: 88,5 % dos episódios do ANDAR são de terreno, e só 11,5 % ficam no plano. A nota `s_andar` passa a medir só esses 11,5 %. A fração do terreno nunca passa de 0,90: o piso da célula `andar` é 0,10. Em L̄ = 12, a fração do terreno depende só da nota `s_andar`. Com a fatia B aberta, a fração dos episódios do ANDAR vale `frac_ANDAR` (seção 1.2) vezes esta fração.

A fração das colunas de escada, com as linhas iguais: cada coluna recebe 1/8 dos episódios de terreno. As colunas `escada` e `escada_inv` somam 2/8 = 0,25. Com a `escada_aleatoria`, somam 3/8 = 0,375 (derivado: conta). No começo, `frac_ANDAR` = 1,0, e então 0,885 × 0,25 = 0,22 dos episódios do ANDAR caem numa coluna `escada` ou `escada_inv` (derivado: conta). Na abertura, todas as linhas valem 0, e a linha sorteada é `floor(0 + u)` = 0. O degrau tem altura 0 m: a escada é plana. O 0,3 m é a profundidade do degrau, e não a altura. A altura sobe com a linha da coluna (seção 11.5).

### 11.4 Os pesos das colunas do terreno

Exemplo de linhas por coluna: `escada` 7,0; `escada_inv` 6,0; `rampa` 9,0; `rampa_inv` 10,0; `escada_aleatoria` 3,4; `irregular` 12,0; `obstaculos` 4,0; `caixas` 8,0. O peso de cada coluna é `fracoes(1 − linha / 12, 0,05, 0,02)` (seção 1.4). A tabela mostra o peso calculado e a frequência de 200 000 chamadas de `escolhe_celula_terreno` (derivado: simulação).

| Coluna | Linha | Peso | Frequência |
|---|---|---|---|
| `escada` | 7,0 | 0,1316 | 0,1315 |
| `escada_inv` | 6,0 | 0,1472 | 0,1470 |
| `rampa` | 9,0 | 0,1005 | 0,1006 |
| `rampa_inv` | 10,0 | 0,0849 | 0,0851 |
| `escada_aleatoria` | 3,4 | 0,1877 | 0,1877 |
| `irregular` | 12,0 | 0,0537 | 0,0537 |
| `obstaculos` | 4,0 | 0,1783 | 0,1777 |
| `caixas` | 8,0 | 0,1160 | 0,1167 |

As colunas com a linha mais baixa recebem mais episódios. As escadas somam 0,28 (sem a aleatória) ou 0,47 (com ela). A linha sorteada da `escada_aleatoria` é 3 com probabilidade 0,6 e 4 com probabilidade 0,4, e a média sorteada é 3,40 (simulação). Uma coluna com linha inteira sorteia sempre essa linha.

O mínimo por coluna é `Terreno.piso_coluna` = 0,05 mais um resto. Com uma coluna em 12 e as outras sete em 0, a coluna vencida recebe 0,0517 (derivado: conta, `0,05 + 0,60 · 0,02 / 7,16`). Com as oito colunas em 12, ou em qualquer valor igual, cada uma recebe 0,125. A coluna vencida nunca some.

### 11.5 O empurrão contra a duração de um lance de degraus

O empurrão soma U(−0,5; 0,5) m/s em x e em y à velocidade da base, no mundo. O módulo máximo é 0,71 m/s (derivado: conta). O empurrão não segue o currículo: o valor é fixo e vale em todo episódio do treino. O intervalo é U(3; 7) s, sorteado depois de cada empurrão. O reset do episódio não sorteia o intervalo de novo: o tempo restante segue de um episódio para o outro (seção 6). O primeiro empurrão de um episódio vem entre 0 e 7 s depois do reset (derivado).

A geometria da escada que o código monta (`primitive_terrains.py:84-92`, `:115-136`; `config.py:46-54`; seção 1.4). A célula mede 8,0 m, com borda de 1,0 m: a área útil mede 6,0 m. O número de anéis é `int((8,0 − 2 · 1,0 − 2,0) / (2 · 0,3))` = 6, cada um com 0,30 m de profundidade. O topo central mede 6,0 − 2 · 6 · 0,3 = 2,4 m de lado, e não os 2,0 m de `Terreno.plataforma` (derivado: conta). A partir do centro, em linha reta ao longo de um eixo:

- de 0 a 1,2 m, o topo central;
- de 1,2 a 3,0 m, os 6 anéis de degraus. São 7 espelhos de altura `h`: o primeiro a 1,2 m do centro e o último a 3,0 m;
- de 3,0 a 4,0 m, a borda plana. A célula vizinha começa a 4,0 m.

O lance de degraus mede 1,8 m e não depende de `dist_travessia`. A travessia de 3,5 m fica no meio da borda plana, depois do último espelho (comentário de `knobs.py:442`). A altura do degrau é `h` = `linha / 12 · 0,20` m: 0 m na linha 0, 0,10 m na linha 6 e 0,20 m na linha 12.

A chance de pelo menos um empurrão no lance (derivado). Com o relógio que segue entre episódios, os instantes de empurrão formam uma renovação com intervalos U(3; 7) s, e a fase do relógio no reset não depende do episódio. A chance depende só da duração do lance, `T` = 1,8 / `v`. A conta é a mesma do arquivo `03-reset-terreno-e-perturbacoes.md`, seção 9.2: 200 000 janelas com `torch.Generator().manual_seed(0)`, cada uma com começo uniforme depois de 30 s, que descarta a fase do startup. Hipóteses: o robô anda à velocidade `v` constante, na direção de uma face, e o empurrão não o atrasa.

| `v` (m/s) | Duração do lance `T` (s) | Intervalo do lance, partindo do centro (s) | Pelo menos um empurrão no lance | Empurrões esperados |
|---|---|---|---|---|
| 0,3 | 6,0 | 4,0 a 10,0 | 0,975 | 1,20 |
| 0,4 | 4,5 | 3,0 a 7,5 | 0,844 | 0,90 |
| 0,5 | 3,6 | 2,4 a 6,0 | 0,712 | 0,72 |
| 0,7 | 2,6 | 1,7 a 4,3 | 0,514 | 0,51 |
| 1,0 | 1,8 | 1,2 a 3,0 | 0,359 | 0,36 |

Para `T` ≤ 3 s há no máximo um empurrão no lance, porque o intervalo mínimo é 3 s. A chance vale então `T` / 5, com 5 s a média do intervalo (derivado: 2,57 / 5 = 0,514; 1,8 / 5 = 0,36). Os números de 6,0 s, 4,5 s, 3,6 s e 1,8 s são os do arquivo 03, seção 9.2. A coluna do intervalo do lance supõe a partida no centro; com o deslocamento de partida (±0,5 m), o intervalo muda de lugar, mas a chance não muda.

Mesmo a 1,0 m/s, um lance em cada três recebe um empurrão (0,359). A 0,3 m/s, quase todo lance recebe um. No nível 2, o `vx` pedido vai até 0,5 m/s: o lance dura pelo menos 3,6 s se o robô entrega o pedido, e a chance de um empurrão no lance é de pelo menos 0,71.

### 11.6 O que conta visita e travessia, e a ordem das leituras

O currículo conta visita e travessia por coluna no `sorteia_fatia`, no reset do env, antes de sobrescrever a fatia e o terreno (`eventos.py:143-160`). A conta usa o episódio que acabou.

- Visita: o episódio satisfaz as três condições abaixo (`eventos.py:156-157`):
  1. O episódio era ANDAR em terreno: o env estava na fatia ANDAR e numa coluna ≠ 0 (`plano` nunca conta).
  2. O episódio tinha passos: `episode_length_buf` > 0.
  3. O comando pediu ≥ `dist_travessia`, ou o episódio acabou em queda, ou a base passou de `dist_travessia`. O pedido é o acumulado `soma_cmd_marcha · step_dt`, com `dist_travessia` = 3,5 m. A queda é a terminação desse passo (`terminated` não inclui o `time_out`). A distância é a da base à origem da célula.
- Travessia: uma visita em que a distância da base à origem da célula passou de 3,5 m (`eventos.py:158`).
- O primeiro reset não conta visita nem travessia. Esse reset roda na construção do env, e o `episode_length_buf` ainda vale 0: o contador só volta a 0 no fim do `_reset_idx`, depois dos eventos (seção 2). Nesse reset, o mjlab já deu coluna e origem de terreno a cada env, o `v3_fatia` nasce 0 (ANDAR) e o `pose_ult` nasce 0. Sem a condição 2, o d seria a distância da origem da célula à origem do mundo, quase sempre maior que 3,5 m (comentário de `eventos.py:148-150`). Num reset normal, o contador vale pelo menos 1, porque ele sobe antes das terminações (seção 2.11).
- O episódio que acaba por `time_out` de 20 s, sem pedir 3,5 m e sem passar de 3,5 m, não conta. A queda sem cruzar conta como visita sem travessia e baixa a taxa. A taxa de cada coluna é `travessias / visitas`, sobre as visitas, e não sobre todos os episódios.
- A distância é a de `twist.pose_ult`, no plano xy, menos `env_origins` da célula (`eventos.py:154`). O `soma_cmd_marcha` vale o pedido do episódio inteiro: ele zera só no `reset` do twist, que roda depois do evento `fatia` (`comando.py:890`).

A ordem dentro de um passo (seção 2.11) decide o que cada leitura vê:

1. O passo de física roda. `episode_length_buf` sobe 1.
2. As terminações rodam. O `time_out` da travessia lê o `pose_ult` gravado no fim do passo anterior, no `compute` do twist (`comando.py:813-815`, `terminacoes.py:69-72`).
3. O reset roda nos envs terminados: o currículo, depois os eventos, entre eles o `fatia`, que lê o mesmo `pose_ult` e a terminação do passo (`eventos.py:151-160`).
4. O `compute` do twist roda por último e grava o `pose_ult` novo, já com a pose do reset nos envs resetados.

Por isso o término e o registro leem a mesma posição, a do fim do passo anterior. Se o término dispara porque `pose_ult` passou de 3,5 m, o registro do mesmo passo vê a mesma distância e conta a travessia. Se a terminação lesse a posição atual, o registro ainda leria a anterior, a travessia não contaria (comentário de `terminacoes.py:65-68`). O custo é um passo de atraso: 0,02 s, ou 0,6 a 2,0 cm a 0,3 a 1,0 m/s (derivado: conta). A guarda `episode_length_buf > 2` do término atrasa a leitura nos 2 primeiros passos do episódio, como a guarda do `terrain_edge_reached` do mjlab (`terminations.py:99-100` do mjlab; comentário de `terminacoes.py:63-64`). O `pose_ult` do passo do reset já é novo, porque o `compute` roda depois do reset.

## Não determinado pelo código lido

- Não determinado pelo código lido: o número de envs paralelos. O docstring de `train.py` cita `--env.scene.num-envs 4096` só como exemplo (`train.py:3`).
- Não determinado pelo código lido: o atrito dos geoms do terreno.
- Não determinado pelo código lido: o valor numérico da altura da origem das colunas `rampa` e `rampa_inv` (seção 1.4).
