# Plano de implementação da v3 (`g1_v3/`)

Data: 2026-10-14. Branch: `g1_v3`. Pedido do dono: "inicie a implementação do g1-v3 para que eu
possa começar a testar um treinamento no kaggle" (a plataforma passou ao Docker no RunPod em 22/10,
PD-22). O plano vai a um workflow de agentes Sonnet e
Haiku, orquestrado por Opus.

Base, nesta ordem de autoridade (a mesma da spec):

1. `specs/g1-limpo-comportamento.md` — o enunciado ("E §n" neste plano).
2. `specs/g1-v3-comportamento.md` — o comportamento esperado ("CV §n").
3. `specs/g1-v3.md` — a spec técnica ("S §n").
4. `specs/g1-v3-interface-real.md` — o contrato com o robô real ("I §n").
5. `docs/planos/2026-10-02-percepcao-lidar-na-v3.md` — o plano de percepção ("PERC §n").
6. `docs/relatorios/2026-10-05-dr-para-a-v3.md` — o cronograma de DR ("DR §n").
7. `vault/_sistema/inbox.md`, entradas de 14/10 — as decisões desta sessão.

Decisões do dono nesta sessão (14/10), já na inbox do vault:

- O plano cobre as Partes 0 a 4 em detalhe de tarefa. A Parte 5 (Isaac) fica fora.
- A Parte 0 roda inteira antes da Parte 1, como a S §16.
- Percepção: a fase 1 (filtro puro) e a fase 5 (lidar no treino) vêm antes da Parte 1. As fases 2
  a 4 (sim do real) e a 6 (réplica) correm em paralelo e fecham antes do robô real.
- A IK é fonte de referências macro, e não rastreio de pose. O pacote lê da IK altura da pelve,
  inclinação do tronco, recuo até a caixa, palmas e a posição da caixa no transporte. As únicas
  poses usadas inteiras são as de pé: o `q_de_pé` (offset da ação e alvo do desvio no ANDAR, S §6.3)
  e a pose de pé com caixa (alvo do desvio dos braços, da cintura e do quadril no CARREGAR, S §9.2,
  dono 05/10).
- O item 10 da Parte 0 sai: sem IK nova. P0, P1, P3, B1, B2 e B3 são alvos geométricos.
- O smoke roda uma vez, no fim da implementação, pelo dono.
- Commits locais durante a implementação estão liberados. Push continua proibido.

---

## 0. Resumo

O plano cria o pacote `g1_v3/` em nove fases, de A a I. As fases A, B e C produzem as funções
puras e os números da Parte 0. A fase D é a Parte 1 (pacote, locomoção e lidar). As fases E, F, G
e H são as Partes 2, 3, 3b e 4. A fase I é o fim: o dono roda o smoke, o orquestrador corrige, e o
dono lança o treino no RunPod.

O treino começa do zero, com 100% dos envs no ANDAR (fase 0 do currículo). O currículo abre a
cadeia B, a cadeia C e o refino sozinho, pelos balanceadores (S §11). Por isso a primeira corrida
no RunPod já usa o pacote inteiro, e os portões de play das Partes 1 a 4 (S §15) se conferem nos
checkpoints, na ordem em que o currículo abre as tarefas.

```
A preparação ──► B Parte 0 (puro) ──┐
     └─────────► C percepção fase 1 ┴─► D Parte 1 ─► E Parte 2 ─► F Parte 3 ─► G Parte 3b ─► H Parte 4 ─► I smoke + RunPod
```

Seções do enunciado que o plano atende: E §1 (ciclo do galpão, ordem de aprendizado), E §2
(cenário físico, terreno, simulador), E §3 (invariantes), E §4.1 a §4.5 (etapas).

---

## 1. Regras do workflow

### 1.1 Quem roda o quê

| ação | quem | observação |
|---|---|---|
| escrever código e testes | agentes | só os arquivos da tarefa |
| `python -m unittest discover -s g1_v3/testes -t .` | agentes e orquestrador | CPU, sem env, segundos |
| scripts da Parte 0 sobre CSV e funções puras | agentes | CPU, segundos |
| FK sem física (`mj_kinematics`) nos geradores de dados da fase A | agentes | ver PD-3; se o dono vetar, ele roda dois comandos |
| construir `make_env_cfg()` num teste | agentes | monta dataclasses; não constrói env nem compila cena |
| `ManagerBasedRlEnv`, `smoke.py`, viewer, IK, `exporta_cena` | dono | memória `feedback-quem-executa`: sem simulador na máquina dele |
| `registra_juntas_v3.py` (MuJoCo clássico, sem viewer) | dono ou orquestrador, quando o dono pedir | exceção registrada em 05/10 |
| commit local | orquestrador | depois de cada tarefa, com o teste verde |
| push, `gh`, operação remota | dono | proibido para agentes e orquestrador |
| imagem Docker, pod no RunPod e treino | dono | fase I (PD-22) |

### 1.2 Modelos

- Implementação: Sonnet (`model: "sonnet"` explícito em todo despacho).
- Cópia mecânica do `g1_limpo` e geração de dados: Haiku (marcado na tarefa).
- Orquestração, revisão de fim de Parte e correções de integração: Opus.
- Os testes de cada fase saem deste plano, com os números esperados. Uma lane de testes os escreve
  ANTES do código da mesma fase. O implementador não escreve o teste que o julga (memória
  `feedback-subagentes`: "o implementador barato escreve teste que não pode falhar").

### 1.3 Git

- O orquestrador faz um commit local por tarefa, depois do teste verde. Mensagem em Conventional
  Commits, em português, sem `Co-Authored-By` (memória `feedback-git-e-pr`). Depois de cada lote
  de agentes: `git log <base>..HEAD --format=%B | grep -ci co-authored` tem de dar 0.
- Agentes em paralelo não rodam git. O staging é global, e um `git commit` levaria o arquivo de
  outra lane.
- Stage por caminho explícito. Nunca `git add -A` nem `git add .`.
- O WIP do dono fica fora de todo commit: `vault/.obsidian/graph.json`, `vault/Untitled*.canvas`,
  `vault/_sistema/inbox.md` (exceto quando a tarefa manda editar), os CSV e `.limites.csv` da raiz,
  `reference_checkpoints/*.pt` não rastreados, `.claude/`.
- Sem push. Sem `gh`. Sem rebase nem `reset --hard`.

### 1.4 Lanes

Cada fase lista as suas lanes. Uma lane é dona de um conjunto de arquivos, e duas lanes da mesma
onda nunca tocam o mesmo arquivo. Uma falha de teste em arquivo de outra lane é reportada ao
orquestrador, e não consertada.

### 1.5 Regras de código (CLAUDE.md do repo)

- Antes de criar uma função, procure uma que já existe e possa ser copiada ou expandida.
- Cada tarefa mostra a contagem de linhas antes → depois.
- Zero import do `g1_limpo`, do `g1_poc`, do `g1_multitask` e do `g1_training` dentro do `g1_v3`.
  O `g1_v3` COPIA as peças que ficam (S §5).
- Os módulos puros (`knobs`, `alvos`, `potencial`, `balanceador`, `estimador`, `percepcao_lidar`)
  importam só `math`, `dataclasses`, `numpy`, `torch` e a stdlib. Nunca o mjlab nem o `mujoco`.
  Eles também não importam uns aos outros: recebem as seções do `knobs` como argumento (as anotações
  de tipo ficam em string, com `from __future__ import annotations`). Motivo: o sim do real e o robô
  carregam cada arquivo por caminho, sem o pacote (PD-43).
- Cada termo do mjlab (recompensa, terminação, observação) é um núcleo puro sobre tensores mais um
  invólucro fino que lê o env. O teste chama o núcleo com tensores sintéticos. O núcleo é o que se
  reescreve no Isaac Lab (S §1).
- Comentários em português, no estilo do `g1_limpo`: o porquê, com `⚠` nos pontos que já custaram
  caro e a referência da spec.
- Todo número mora no `knobs.py`. Nenhum número de treino nasce dentro de um termo.

### 1.6 Revisão

Uma revisão por Parte, no fim dela, pelo Opus, contra este plano e contra a S. As correções vão
num despacho seguinte. Nada de revisão por mini-tarefa (memória `feedback-quem-executa`, 05/10).

### 1.7 Smoke

O `g1_v3/smoke.py` cresce em cada fase: a fase escreve as seções dela, sem rodar. O dono roda o
smoke inteiro uma vez, na fase I. O padrão é o do `g1_limpo/smoke.py`: `check(nome, cond,
detalhe)` (`g1_limpo/smoke.py:50-55`), `secao(t)` (`:58-59`), envs em CPU com 2 a 64 mundos, saída
`N ok / M falhas` e `sys.exit(1)` com falha.

---

## 2. Estado atual

### 2.1 O que existe

| peça | onde | tamanho | nota |
|---|---|---|---|
| pacote antigo | `g1_limpo/` | 18 073 linhas (6 384 do smoke) | a v3 copia peças dele (§2.2) |
| tabelas de IK | `g1_limpo/ik/agachado/ref_agachado.npz` (21 poses), `g1_limpo/ik/ref_v3_de_pe.npz` (7), `g1_limpo/ik/mao_fixa/ref_v3_de_pe_theta-30_tamanhos.npz` (3) | — | `qpos` de 43 = raiz 7 + juntas 29 + caixa 7; `hash_mjcf` 4fb8671f211402f1 |
| plays com juntas, ação, força | CSV na raiz do repo (21700 a 24800, 5000 a 6050) | 130 colunas | `q_*`, `a_*`, `alvo_pd_*`, `fn_E`, `fn_D`, `fn_tot`, poses da raiz, da caixa e da laje |
| cercas medidas nos plays | `g1_limpo/ik/agachado/cercas_nos_plays.txt` | — | S §16 item 11 |
| driver antigo | `pilota.py` (657), `registra_juntas.py` (558), na raiz | — | `monta_observacao` em `pilota.py:167-274` |
| notebook antigo | `g1_limpo/kaggle/g1_limpo_zero_kaggle.ipynb` (10 células) | — | as lições que a entrada do container (D13) herda: chave `RETOMA` explícita, nome novo de `RUN`, checagem da CUDA e das versões |
| padrão do MID-360 | `~/orca/workspaces/g1_training/lidar_sim/sensores/mid360.csv` | 800 000 linhas (4 s) | colunas tempo, azimute, zênite; URL da Livox em `sensores/lidar.py:17-18` |
| montagem dos sensores | `lidar_sim/sensores/montagem.py` | — | `MID360_POS`, `MID360_QUAT`, `D435_POS`, `D435_PITCH` |
| ruído do lidar | `lidar_sim/sensores/tipos.py` | — | alcance σ 2 cm, ângulo σ 0,15°, perda 1%, D435 depth 58° vertical, 30 Hz, 33 ms |

Todo caminho `lidar_sim/...` deste plano é relativo a `~/orca/workspaces/g1_training/`: o `lidar_sim`
fica fora do repo e sem git. A v3 só lê dele na fase A (A2 copia os números para o `knobs.py`; A4
grava o `.npz` no git); depois da fase A, nada da v3 depende dele.

Ambiente: mjlab 1.5.3, rsl-rl-lib 5.4.0, mujoco 3.10.0, torch 2.13 em `.venv/`. O `pytest` não está
instalado; os testes usam `unittest` da stdlib (PD-2).

### 2.2 O que a v3 copia do `g1_limpo` (mapa dos agentes de leitura, 14/10)

| peça | origem | linhas | dependência que sai |
|---|---|---|---|
| `_spec_box`, `spec_caixa`, `spec_prateleira` | `g1_limpo/cena.py:164-234` | 67 | — |
| `_add_pad`, `add_pads_de_palma` | `g1_limpo/cena.py:238-279` | 40 | atrito do pad fixo em `:245` vira knob |
| `regroup`, `geometria_de_repouso`, `entidades` | `g1_limpo/cena.py:309-351` | 43 | `grupo_mobilia` 2 → 1 (PD-9) |
| `sensores` (11 sensores de contato) | `g1_limpo/cena.py:354-455` | 102 | — |
| `posiciona_cena`, `_topo_por_nivel`, `orientacao_de_nascimento` | `g1_limpo/eventos.py:74-211` | 135 | `garante_nivel`; z absoluto (viab. D3) |
| `tamanho_caixa`, `carga_caixa` | `g1_limpo/eventos.py:256-369` | 112 | `garante_nivel` |
| `caixa_largada` | `g1_limpo/terminacoes.py:52-120` | 69 | `_elo`, `fechou`, constantes do comando antigo |
| `Caiu`, `NoBatente` | `g1_limpo/terminacoes.py:123-196` | 72 | z do joelho contra a origem (viab. D4) |
| `contato_mesa` | `g1_limpo/recompensas.py:180-212` | 33 | — |
| `velocidade_por_regime` | `g1_limpo/recompensas.py:1179-1329` | 151 | `ESTADO_ANDAR`, `limpo_estado` |
| `_forca_das_palmas`, `_forca_ref` | `g1_limpo/recompensas.py:579-629` | 49 | — |
| kernels `_alcancar`, `_alinha` | `g1_limpo/recompensas.py:526-576`, `:680-718` | 90 | `_t`, `sigma_*` do comando antigo; só a forma fica |
| `TwistComRazaoDeMarcha` e cfg | `g1_limpo/comando.py:2276-2526` | 251 | — |
| `forca_de_apoio` | `g1_limpo/comando.py:187-220` | 34 | — |
| `garante_nivel`, `garante_freio` | `g1_limpo/curriculo.py:61-97` | 37 | — |
| métricas sem dependência | `g1_limpo/metricas.py:210-218`, `:243-276`, `:341-470`, `:585-590` | ~170 | — |
| `PPOPorElo.compute_returns` | `g1_limpo/algoritmo.py:155-224` | 70 | `elo_interno`; o one-hot fora do normalizador sai (sem normalizador) |
| `RunnerComEstadoDeCurriculo` | `g1_limpo/runner.py` | 143 | chaves novas |
| `constroi_env`, `coleta_cena`, `_mapa_atuador`, `espelha_estado`, `exporta`, `paridade` | `g1_limpo/exporta_cena.py:56-426` | ~370 | 114 → 1 447 canais |
| `carrega_cena`, `gira*`, `monta_observacao`, `Ator`, `Piloto` | `pilota.py` | 657 | observação nova |

Fatos que mudam a cópia:

- `NoBatente` nunca é construído no `g1_limpo` (`env_cfg.py` não o referencia). A v3 o liga no
  refino (fase H).
- `afasta_cena` (`g1_limpo/eventos.py:214-253`) não tem chamador no `g1_limpo`. A v3 copia a lógica dela como `estaciona_cena`: na fatia ANDAR, a laje e a caixa ficam 20 m acima da origem, fora do alcance do lidar (6 m) e sem tocar o terreno (fase E1).
- O `g1_limpo` herda do molde `push_robot` (1–3 s), `foot_friction` (0,3–1,2), `encoder_bias` e
  `reset_robot_joints` (`velocity_env_cfg.py:217-258`). A v3 escreve cada um com o valor da S §12.
- Os eventos escrevem z sem a origem do env; as terminações subtraem a origem. A v3 mede toda altura
  contra o terreno (PD-21).
- O PD do `g1_limpo` é o do mjlab (`kp = armadura·(2π·10)²`, `g1_constants.py:112-123`). A v3 usa a
  tabela da Unitree (I §3).

### 2.3 APIs do mjlab 1.5.3 que o plano usa (conferidas em 14/10)

| API | onde | uso |
|---|---|---|
| atraso de observação por termo | `managers/observation_manager.py:35-57` (`delay_min_lag`, `delay_max_lag`, `delay_update_period`, `delay_per_env_phase`) | atraso 0–1 passo sorteado por episódio, com capacidade 0–3 desde o build (PD-25, PD-28) |
| atraso de comando do atuador | `actuator/actuator.py:67-85` (em passos de física) | idem, 0–4 passos de física = 0–1 passo da política |
| sorteio do atraso | `utils/buffers/delay_buffer.py:247-258` | com `update_period` enorme e fase igual, o lag sai no 1º passo depois do reset e fica |
| histórico por termo, com preenchimento no reset | `observation_manager.py:57-60`; `CircularBuffer` copia a primeira leitura em todos os lugares | I §2.1: "no primeiro passo, os 5 lugares recebem a mesma leitura" |
| raios preparados em PyTorch fora do grafo | `sensor/raycast_sensor.py:691-747` (`prepare_rays` lê `self._local_directions` a cada passo) | padrão rotativo do MID-360 (fase 5a) |
| padrão de raios por duck typing | `raycast_sensor.py:517-521` (`pattern.generate_rays(mj_model, device)`) | classe `PadraoMid360Cfg` própria |
| `TerrainHeightSensorCfg` | `sensor/terrain_height_sensor.py` (altura do frame sobre o terreno) | altura do pé (L7), da pelve, do joelho e da caixa (PD-21) |
| sensor de contato com `pos` e `normal` | `sensor/contact_sensor.py:107-154` | régua do aperto por face (S §8.2) |
| origem do env por subterreno | `terrains/terrain_entity.py:216-233` (`terrain_levels`, `terrain_types`, `env_origins`) | terreno por tarefa (D1) |
| dificuldade por linha do gerador | `terrains/terrain_generator.py:248-264` (`t = linha/(R−1)`) | tetos por coluna (PD-20) |
| DR | `envs/mdp/dr/` (`geom_friction`, `body_mass`, `body_com_offset`, `pd_gains`, `effort_limits`, `encoder_bias`, `joint_friction`, `joint_armature`) | S §12, DR §5 |
| métricas | `MetricsTermCfg`; `extras["Episode_Metrics/<k>"]` (`metrics_manager.py:119-124`) | S §14 |

---

## 3. Estado final e como verificar

No fim da fase H:

- `g1_v3/` existe com os módulos da S §5 e as tabelas de dados em `g1_v3/dados/`.
- `python -m unittest discover -s g1_v3/testes -t .` passa, sem env.
- O registro `import g1_v3.tarefa` registra a task `Mjlab-G1-V3` sem erro e sem construir env.
- `g1_v3/smoke.py` tem uma seção por mecanismo das Partes 1 a 4.
- `g1_v3/container/entrada.sh` existe e o `test_entrada` passa em modo seco.
- `pilota_v3.py` e `registra_juntas_v3.py` existem na raiz.
- `docs/relatorios/2026-10-??-v3-parte0.md` tem os números aceitos pelo dono, já no `knobs.py`.

No fim da fase I:

- O smoke do dono passa.
- `python -m g1_v3.exporta_cena --paridade` passa (observação do env = `monta_observacao`).
- O treino roda no RunPod e os portões de play da S §15 são conferidos nos checkpoints (§13.3).

---

## 4. O que NÃO fazemos

- Parte 5 (Isaac Sim e Isaac Lab).
- REORIENTAR, LEVANTAR, prateleira, estimador real da caixa (S §4 "Fora").
- Fases 2, 3, 4 e 6 do plano de percepção (sim do real e réplica). Elas correm em paralelo, fora
  deste plano, e usam o MESMO `g1_v3/percepcao_lidar.py`, carregado por caminho (PERC §2).
- IK nova (decisão de 14/10).
- Tracking de corpo inteiro ou de pose (S §9.3).
- Mapa perfeito no crítico (PD-17).
- Cerca de largura dos pés (S §9.3, dono 09/10).
- `paridade.py` do `g1_limpo`, `inspeciona.py`, `leitura.py` e as tasks de inspeção
  por elo. A v3 tem uma task de treino e uma de play.
- Base_com (heap, [F14]) e qualquer DR que a S §12 deixa fora.
- A regra do relatório de DR "a faixa desce com sucesso < 0,50": a S §11.5 diz que o refino nunca
  desce (PD-24).

---

## 5. Decisões do plano (o dono revisa antes de despachar)

A spec não fixa estes pontos. O plano fixa um valor ou uma regra, e o agente implementa como está.
O dono pode trocar qualquer um antes do despacho.

| id | ponto | decisão | motivo |
|---|---|---|---|
| PD-1 | registro da task | `g1_v3/__init__.py` fica sem import. O registro mora em `g1_v3/tarefa.py`. | importar um módulo puro não executa o mjlab; o robô real carrega por caminho (PERC §2) |
| PD-2 | testes | `unittest` da stdlib, em `g1_v3/testes/` | o `pytest` não está no venv, e instalar é do dono |
| PD-3 | FK sem física (ACEITA pelo dono, 22/10) | o agente roda `gera_referencias.py`, `gera_capsulas.py` e os scripts da Parte 0 que usam `parte0/comum.fk` (MuJoCo clássico, só `mj_kinematics`, sem física, sem viewer, segundos) | não é simulação; evita parar o workflow. Se o dono vetar, ele roda os comandos |
| PD-4 | ponto da palma | o site `left_palm`/`right_palm` do `g1.xml` (pos 0,08 0 0 na mão) | o robô real calcula o mesmo ponto pela FK |
| PD-5 | lidar em fluxo (ACEITA pelo dono, 22/10) | 205 raios por passo de 20 ms (1 025 por 0,1 s), cada passo com o trecho seguinte do bloco do CSV; cada ponto vai ao mapa pela pose do passo em que o raio sai. No robô real, o driver entrega ao filtro, a cada passo de 20 ms, os pontos que chegaram desde o passo anterior, cada um levado ao mundo pela pose do sensor no carimbo de tempo dele (deskew: IMU, odometria, FK das juntas); o terreno parado cai no mesmo lugar no treino e no real (dono, 28/10) | mesma densidade da D6 (1 024 por 0,1 s), 1/5 do custo de raycast por passo. A densidade real (~20 mil pontos por 0,1 s) não muda a confiança (PD-42) |
| PD-6 | confiança da célula | `c = σ_ref² / (σ_ref² + P)`, σ_ref 5 cm; célula nunca vista: `c = 0` | I §2.4 deixou a fórmula para a fase 1; com o piso da PD-42, a célula vista satura em 0,862 (1 leitura no real, 2 no treino, a 2 m) e cai só pelo transporte (PD-7): 0,758 depois de 1 m |
| PD-7 | transporte e variância | `P += q_d·‖Δxy‖ + q_r·abs(Δψ) + q_t·Δt`, com `q_d` (2 cm)²/m, `q_r` (2 cm)²/rad, `q_t` (0,5 cm)²/s. O transporte também é vertical: as alturas guardadas descem `z_ref − z_ref_anterior`, com `z_ref` = z do pé mais baixo no frame da odometria (dono, 29/10) | PERC §0: cresce sobretudo com o deslocamento e devagar com o tempo. O zero da altura (I §2.4 passo 3) anda com o pé: sem o transporte vertical, a célula medida antes de um degrau fica 15 cm errada depois dele, com confiança perto de 0,8, até o lidar ler a célula de novo; no CARREGAR, a caixa tapa o chão à frente dos pés |
| PD-8 | mapa interno | 51 × 51 células de 10 cm (5,1 m), centrado na base, eixos no rumo | viab. D5 (5 × 5 m); célula central = a base |
| PD-9 | grupo da mobília | caixa e laje no grupo de geom 1; o lidar inclui os grupos 0 (terreno), 1 (mobília) e 3 (colisão do robô) | o grupo 2 tem 35 malhas visuais do robô (viabilidade D8) |
| PD-10 | filtro do corpo e caixa na mão (DONO, 22/10) | cápsulas das geoms de colisão, margem 3 cm; e todo ponto a menos de `raio_min` 0,85 m do lidar sai, em todas as tarefas (0,7 até 28/10). A caixa estimada NÃO entra no filtro. O mesmo `filtra_corpo` limpa a entrada da odometria real (KISS-ICP): um ponto que anda com o robô não entra no casamento de nuvens (dono, 28/10) | 0,85 = o braço + 5 cm: FK sem física, ponto da mão mais longe do MID-360 a 0,64 m (frente), 0,74 (lado, T), 0,45 (cima), 0,77 (baixo) e 0,80 no máximo de 20 000 poses nos limites; de pé com os braços soltos, a mão já fica a 0,74 m. O raio tira o braço e a caixa sem depender da forma das cápsulas; as cápsulas tiram pernas e pés. A caixa nas mãos fica a 0,3–0,64 m do lidar (quina de baixo da frente, com 8 cm de escorregamento); o lidar fica a 1,30 m da sola de pé e a 0,61–0,64 m no agachamento fundo. O corte não depende de estimar a caixa, da cinemática dela nem do one-hot, e é o mesmo no sim e no real. A versão anterior (volume da caixa estimada) vazava com a leitura velha presa ao mundo; a máscara pela câmera não cobre a face que só o lidar vê. Preço aceito: tudo a menos de 0,85 m do sensor some, inclusive o chão a 0,43–0,58 m na horizontal no agachamento fundo do PEGAR e do BOTAR e o topo da laje alta bem perto do robô |
| PD-11 | oclusão da D435i | 9 pontos da caixa (centro e 8 cantos) testados contra as cápsulas dos braços; a leitura vale com o centro no campo de visão (87° × 58°), na faixa 0,28–3,0 m, e com ≥ 3 pontos sem oclusão; o campo de visão é o do RGB, 69° × 42°, onde roda a detecção (DONO, 15/10; era o do depth, 87° × 58°) | S §13: reproduzir a visibilidade real; mesma lista de cápsulas da PD-10 |
| PD-12 | células do balanceador (ACEITA pelo dono, 22/10) | uma célula por TIPO de subtarefa (ANDAR: reto, parado, baixo, giro andando, giro no lugar, terreno). O nível da tarefa sobe quando todas as células abertas passam no portão e todos os tipos já abriram. A EMA da tarefa recomeça na primeira medida depois da subida. Mínimo de 50 its entre subidas | com uma célula por (tipo, nível), o piso 0,10 estoura com mais de 10 células; "o nível k sorteia entre 0,25 e o teto" já inclui os níveis de baixo |
| PD-13 | EMA do sucesso | uma atualização por iteração, com os segmentos e episódios que fecharam nela, α 0,05 | o mesmo α do balanceador do `g1_limpo` (`Cadeia.balanceador_alpha`) |
| PD-14 | déficit das fatias | ANDAR = células do ANDAR; B = células do PEGAR e do CARREGAR; C = célula do BOTAR | a S §11.1 fala em déficit "da tarefa", e as fatias juntam tarefas |
| PD-15 | sucesso com canal zero | posição deriva < 0,10 m e rumo deriva < 0,25 rad, medidas a partir de uma âncora que fica `ancora_s` 1,5 s depois da abertura do segmento e passa a 1,5 s depois de cada empurrão (DONO, 15/10); um segmento que fecha antes da âncora não registra. Antes de 15/10 a deriva contava o segmento inteiro, e a frenagem do 24800 (0,126 m de 0,5 m/s; 0,62 m de 1,5 m/s, `lidar_sim/cenarios/curva_model_24800.json`) reprovava o parado acima de ~0,5 m/s e travava o nível do ANDAR e o terreno | S §11.3 "abaixo de um limiar" sem número; 0,10 m é o do aceite do giro no lugar (S §15) |
| PD-16 | passo de lado | `L` = projeção, na direção do comando, de (pé que toca − pé de apoio), MENOS a mesma projeção na pose de pé (`q_de_pé`) | sem isto, o passo de lado recebe a largura da base (0,23 m) de graça |
| PD-17 | crítico | sem mapa perfeito; o crítico recebe o mapa do filtro | a S §19.1 só propõe; um segundo raycast dobra o custo |
| PD-18 | estimador "baixo" desde o começo | posição σ 0,5 cm, orientação σ 1°, perda 0,5%, latência 33 ms, 30 Hz, deriva da odometria 0,002 m/√m (0,4 cm em 4 m; o refino vai a 0,0075 m/√m = 1,5 cm em 4 m, o ATE do ESKF); deriva no giro 0,5° e 1 cm por volta (refino 2° e 2 cm por volta), hipótese até medir no G1 (dono, 28/10) | a S §13 diz "baixo" sem número (sanidade 4.5.2); a deriva cresce com a raiz da distância e do ângulo girado, e não com eles (D2b). Os quatro termos são o modelo de movimento de odometria (Thrun, *Probabilistic Robotics*, cap. 5); sem os termos de giro, o giro no lugar não tinha erro de rumo |
| PD-19 | mãos no invariante | `g_mãos = (mãos em contato) / 2` | S §9.1 pede rampa por grandeza; contato é contagem |
| PD-20 | terreno do CARREGAR | só nas colunas com plataforma de 2 m; R = 13 linhas (t = linha/12) e `linha_max_carregar` inteira por coluna: 9 nas escadas e nas rampas (t 0,75: degrau 0,15 m, `slope` 0,27), 8 no `obstaculos` (t 0,667: 0,10 m) e 7 no `caixas` (t 0,583: 0,15·(0,2 + 0,8·t) = 0,10 m, `primitive_terrains.py:796`) (29/10) | S §11.2 tetos do CARREGAR e aceite da S §15 (degrau 0,15 m; 15°); o gerador usa t = linha/(R−1) (`terrain_generator.py:248-264`). Com R = 10, a linha 6 dava t 0,667 = degrau 13,3 cm e `slope` 0,24, abaixo da spec. Preço: 117 ladrilhos contra 90 (+30%, o I5 mede) e 3 níveis a mais de terreno no ANDAR |
| PD-21 | alturas | pelve, joelhos, pés e caixa medidos contra o terreno por `TerrainHeightSensorCfg`, desde a Parte 1 | viab. D3, D4, L7; vale também no plano |
| PD-22 | código na plataforma de treino (DONO, 22/10) | o treino roda em Docker no RunPod (4090 ou L40S), no mjlab; o container clona a branch `g1_v3` do GitHub ao subir; o dono faz o push antes da corrida e publica a imagem. O notebook da Kaggle sai; o ponto de entrada vira um script de treino que o container chama. Os detalhes da imagem, do registro, do destino dos checkpoints e o orçamento por GPU ficam para depois do plano e da implementação | nada sobe ao GitHub sem o dono; as contas de 12–15 s/it e a cota de ~30 h/semana eram da T4 da Kaggle e serão medidas de novo |
| PD-23 | Δ das rampas (o item 2 da Parte 0 confere) | deslocamento B2–B4: 2 cm; deslocamento P0–P2: a/2; tronco: 15°; recuo: 8 cm; CoM: 3 cm; inclinação: 15° (decidido) | B04 deixou os Δ que faltam para a Parte 0 |
| PD-24 | DR | as faixas só alargam, por nível e por degrau de refino; nunca estreitam | S §11.5 "ele nunca desce" |
| PD-25 | atraso por termo | cada termo de propriocepção e o `mapa` sorteiam o próprio lag por episódio (o mapa desde 28/10: no real, os pontos chegam ao filtro com o atraso do caminho) | superconjunto do real (o real tem um lag só); evita API privada |
| PD-26 | detector "juntas quase paradas" | média de abs(q̇) das 29 juntas na janela de 0,5 s < 0,2 rad/s | proposta da I §4.2 |
| PD-27 | chave da caixa perdida | `CAIXA_PERDIDA_VIRA_ANDAR = True` | o aceite da S §15 "caixa perdida" usa a chave ligada |
| PD-28 | atraso maior por nível | a capacidade do atraso é 0–3 passos desde o build; o teto corrente sobe mudando o `max_lag` dos buffers (0–1 → 0–2 → 0–3) | o `DelayBuffer` sorteia em `[min_lag, max_lag]` lendo o atributo a cada sorteio |
| PD-29 | frame da caixa | eixo x da caixa = a normal horizontal de face mais alinhada à frente do robô (rumo da base), z = a normal de face mais perto da vertical do mundo, y = z × x | S §8.1: "o par de faces é o mais alinhado com as palmas" |
| PD-30 | regra de peso das cercas (item 3) (ACEITA pelo dono, 22/10; refeita 29/10, D4) | o peso de cada cerca faz o custo médio por segundo dela, no PEGAR dos plays do C3 e só nos passos em que ela age, valer 50% do ganho médio por segundo do `ProgressoDoAlvo` no caminho roteirizado (a média com os passos parados dava −29/m ao CoM, que age em 1,8% dos passos) | a S §16 item 11 diz "a escala por unidade sai do item 3 contra o Φ" sem a regra; o dono aceita ou troca no portão da Parte 0 |
| PD-31 | `B_alvo` (item 3) (ACEITA pelo dono, 22/10) | 1,5 × o custo dos reguladores no trecho até o alvo, no ponto de operação | S §9.4: "o bônus de cada alvo tem de cobrir o custo dos reguladores no trecho até ele"; 1,5 é a margem |
| PD-32 | peso do `progresso` (item 3) (ACEITA pelo dono, 22/10) | 1,5 × o maior quociente, entre os trechos, de custo dos reguladores por segundo sobre ganho de Φ por segundo no caminho roteirizado | o gradiente do potencial tem de vencer os reguladores no ponto de operação (CLAUDE.md, "gradiente de recompensa") |
| PD-33 | `perda_caixa` (item 3) (ACEITA pelo dono, 22/10) | 1,2 × o lado direito da desigualdade (2) da S §10; `devolve_b_alvo` = True só se a escada não fechar com custo fixo | S §10, L4.1 |
| PD-34 | `sacudida` (item 15) (ACEITA pelo dono, 22/10) | custo = (‖a_lin‖/2,9)² + (‖α‖/17)² (p50 do CARREGAR do 23250, S §11.5); peso tal que o custo no p50 valha 10% do ganho máximo do rastreio linear por segundo | S §11.5 "peso 0 → a definir; ponto de partida medido no 23250" |
| PD-35 | cerca do CoM | só com os dois pés em contato; fora disso o custo é 0 | a medida da Parte 0 (item 11) foi feita assim (`g1_limpo/ik/cercas_nos_plays.py:76`); no apoio simples da marcha o CoM sai do pé de apoio por desenho |
| PD-36 | troca de tarefa no potencial | no passo em que `v3_alvo_idx` vira −1, o `progresso` paga 0, e não `−Φ` | o `−Φ` cairia na hora em que o controlador troca a tarefa, sem relação com a ação do robô; a troca só vem depois do detector (pose estável), então não há ciclo a explorar |
| PD-37 | `qualidade` no CARREGAR | multiplica só a parte positiva do rastreio | multiplicar o rastreio negativo apagaria o custo da deriva com a caixa caída |
| PD-38 | CoM da caixa (DR) (ACEITA pelo dono, 22/10) | atrás da chave `DR.com_caixa` = False; a seção 37 do smoke testa a heap, e o dono liga | `dr.body_com_offset` corrompeu a heap no `g1_limpo` (`env_cfg.py:386`; commit `af77cf4`) |
| PD-39 | "caixa parada" no B3 | ‖v_caixa‖ < 0,05 m/s | S §8.2 sem número |
| PD-40 | gravidade ±0,1 m/s² (DR §3.7) (ACEITA pelo dono, 22/10) | sai | o mjlab 1.5.3 não tem DR de gravidade por mundo; o viés de inclinação do IMU (±1°, §6.8) cobre o mesmo efeito na observação |
| PD-41 | DR por degrau (DONO, 22/10) | um degrau GLOBAL para toda a DR do refino (PD, força do motor, `solref`, atraso, estimador, RFI): o MENOR degrau entre as tarefas. Todos os envs sorteiam com ele no reset; o `max_lag` do atraso segue o mesmo degrau. Substitui o degrau da tarefa principal da fatia | a regra mais simples: uma só, sem escolha por fatia e sem exceção para o atraso; nenhuma tarefa nova sofre DR forte cedo. Preço aceito: a tarefa adiantada refina com a DR da mais atrasada |
| PD-42 | z por célula no filtro | a MÉDIA dos z da leitura na célula, com variância `R = σ_leitura² + σ_m²/n`, `σ_leitura = σ_incl·d̄` (o erro comum da leitura; d̄ = distância média dos pontos da célula); depois do Kalman, `P ≥ σ_fixo²`, `σ_fixo` 2 cm (dono, 28/10; antes, `σ_m²/n`) | os pontos de uma leitura não erram de forma independente: a inclinação do IMU, a extrínseca e o viés do encoder (zero da altura) são os mesmos em todos, e os dois últimos em todas as leituras do episódio; a média não reduz esse erro. Com `σ_m²/n`, a confiança dependia da densidade de pontos (treino 0,854 contra real 0,992 na primeira leitura). O piso é o erro que nenhuma média tira (±1° fixo do IMU a 2 m ≈ 2 cm; viés do encoder no pé ≈ 1 cm). Além disso, o máximo de n pontos com ruído de 2 cm sobe o chão em ~+2 cm com 5 pontos; o desvio da PERC §0 ("a altura dos pontos de contato") fica na média |
| PD-43 | import dos módulos puros | não importam uns aos outros; recebem os knobs como argumento | carga por caminho no sim do real e no robô (PERC §2) |
| PD-44 | dono da odometria | o `CadeiaCmd` dá o `passo` da `v3_odo` uma vez por passo; o `atualiza_percepcao` só lê | dois termos dando o passo dobrariam a deriva |
| PD-45 | cerca de torque nos pares acoplados (DONO, 15/10) | nos pares (`waist_pitch`, `waist_roll`), (`left_ankle_pitch`, `left_ankle_roll`) e (`right_ankle_pitch`, `right_ankle_roll`), a fração é (\|τp\| + \|τr\|)/`tau_par` com `tau_par` 50; nas outras juntas segue \|τ_j\|/`tau_max_j`. O atuador fica em 50 N·m por eixo | dois motores 5020 movem o pitch e o roll de cada par juntos: com engrenagem 1:1 (hipótese do mjlab, `g1_constants.py`), o real tem \|τp\| + \|τr\| ≤ 50. O `unitree_rl_lab` treina com 25 por eixo; o `MIMIC_CFG` da Unitree, com 50. MEDIDO nos plays do C3 (torque pelo PD): o roll da cintura tem p95 de 12–16 N·m; a soma passa de 40 em 1,5–10,7% dos passos com caixa, contra 0,8–3,8% por junta. Fica a pergunta à Unitree: se o pitch puro tem só 25, os 5 kg nas lajes baixas saem (`docs/relatorios/2026-10-15-viabilidade-v3.md` §2.2, item 8) |
| PD-46 | atrito por material (DONO, 15/10) | caixa com `priority` 0 e μ 0,3; colisões do robô (`.*_collision`, pulso e tronco inclusive) e pads de dorso com μ 0,3; pés 0,6 (`priority` 1, DR 0,3–1,0); pads de palma com o DR de μ que era da caixa; laje com `priority` 3 e DR de μ 0,3–1,0. Com `priority` igual, o MuJoCo usa o maior μ e o maior condim do par (3), e o `solref` sai da mistura dos dois | com a caixa em `priority` 2, o μ sorteado da caixa (0,5–1,0) valia no pulso e no tronco, que fazem ~2/3 da força da pega no C3 (`docs/relatorios/2026-10-05-dr-de-atrito-pega-e-transferencia.md:30-42`); no real eles são carcaça. 0,3 é hipótese até alguém medir. Conta: 5 kg pedem 49 N de atrito; pads a μ 0,5 dão 40 N e o pulso a 0,3 dá ~42 N |
| PD-48 | forma do `passo` (DONO, 15/10) | r = min(relu(L)/`L_alvo`, 1)² por toque, vezes max(r_lin, 0); peso 0,5 sem mudança | por toque e linear, o termo paga por segundo cadência × L/`L_alvo` = v/`L_alvo`, o mesmo para todo L abaixo do alvo: derivada zero em L. Com o quadrado, paga v·L/`L_alvo`², que cresce até o alvo e é o mesmo de antes no alvo. MEDIDO no 24800: passo de ~0,20 m no ANDAR e de ~0,045 m (6,6 passos/s) no CARREGAR; no ponto do C3, 0,34/s no passo miúdo contra 1,5/s no alvo |
| PD-49 | σ do canal zero (DONO, 15/10) | no ramo c < `limiar` do `rastreio_normalizado`, exp(−v²/σ₀²) − 1 com `sigma_zero` 0,10 m/s no linear e 0,15 rad/s no giro; piso −1 | com o σ 0,5 da Unitree, deslizar a 0,05 m/s custava 0,01 e a derivada era −0,4 por m/s; com σ₀ 0,10, custa 0,22 e a derivada é −7,8 por m/s. É o deslocamento da base no giro no lugar e o rumo que deriva sem comando (pendência "Giro no lugar estranho") |
| PD-50 | sucesso sem perda (DONO, 15/10) | toda regra de sucesso do PEGAR, do CARREGAR e do BOTAR exige `não v3_perdeu` no instante do registro | com `caixa_perdida_vira_andar`, perder a caixa não termina o episódio: o PEGAR contava a caixa caída no CARREGAR, o segmento do CARREGAR passava a razão sem a caixa, e o B4 fechava com a caixa caindo da laje depois (enunciado :84-86) |
| PD-47 | laje de destino (DONO, 15/10) | o alvo de soltura fica a `destino_borda` 0,15 m para dentro da borda perto da laje, centrado no lado; a laje gira `destino_giro_max` ±10° em torno do alvo, contra o rumo da base; o alvo sai do rumo da base em y U[−0,10; 0,10] m; em x, o alvo fica a `recuo(h_dest, a)` + o jitter só para longe + e(θ) = 0,15·cos θ + 0,30·sen\|θ\| − 0,15 (até 5 cm a 10°), que mantém o canto mais perto à mesma distância da base [C38] | no autônomo, o erro lateral de chegada ao BOTAR mediu até 6,7 cm, e o 24800 treinou com a laje a ±0,10 m de lado (`g1_limpo/comando.py:1888-1895`). A laje fora do paralelo é o caso real (dono). Sem a borda: com o alvo no centro da laje de 0,60 m posto a 0,25–0,44 m da base, a borda perto caía de −0,05 a +0,14 m da base, dentro do corpo; o C3 punha o alvo a 0,15 m da borda (`botar_recuo_borda`, `g1_limpo/knobs.py:252`) |

---

## 6. Interfaces (o contrato que todas as lanes seguem)

Esta seção é o contrato. Um nome, um campo ou uma assinatura que não está aqui não existe. Uma lane
que precisa de algo novo para e reporta ao orquestrador, que atualiza esta seção antes de seguir.

### 6.1 Arquivos

```
g1_v3/
  __init__.py              docstring só (PD-1)
  knobs.py                 PURO: todos os números e o contrato com o robô real
  alvos.py                 PURO: referências macro, alvos P0–B4, avanço, detector de fim
  potencial.py             PURO: núcleos, rampas, Φ, progresso, aperto, pouso
  balanceador.py           PURO: os dois balanceadores, níveis e refino
  estimador.py             PURO: estimador simulado da caixa, odometria, visibilidade da D435i
  percepcao_lidar.py       PURO: filtro de elevação, pontos do lidar, filtro do corpo, padrão
  cena.py                  robô (PD da Unitree, pads, sites), caixa, laje, terreno, sensores
  observacoes.py           termos de observação do ator e do crítico
  comando.py               termos de comando: `cadeia` (tarefa, trechos, alvo macro) e `twist`
  curriculo.py             estado do currículo no env e o termo de currículo
  recompensas.py           termos de recompensa (núcleo puro + invólucro)
  terminacoes.py           termos de terminação
  eventos.py               reset, cena, DR
  metricas.py              métricas sem peso
  algoritmo.py             PPO com vantagem por tarefa
  runner.py                estado do currículo no checkpoint
  env_cfg.py               make_env_cfg
  tarefa.py                registro da task (PD-1)
  train.py, play.py        entradas de linha de comando (copiam g1_limpo/train.py e play.py)
  exporta_cena.py          cena.mjb, cena.npz, contrato.json e paridade da observação
  smoke.py                 checagens do dono (fase I)
  dados/                   gera_referencias.py, referencias.npz, gera_mid360.py,
                           mid360_passos.npz, gera_capsulas.py, capsulas.npz
  parte0/                  comum.py e os scripts dos itens da Parte 0
  testes/                  __init__.py e test_*.py (unittest)
  container/entrada.sh     o que a imagem do RunPod chama depois do clone (D13, PD-22)
pilota_v3.py               raiz do repo: driver de play e `monta_observacao` (puro + mujoco)
registra_juntas_v3.py      raiz do repo: play por roteiro, grava CSV
```

### 6.2 `knobs.py`

Dataclasses congeladas, uma por seção, e um agregador `Knobs` com `ATIVO = Knobs()`, como o
`g1_limpo/knobs.py` (agregador em `:1608-1628`). Um campo `None` quer dizer "sai da Parte 0"; o
`env_cfg.make_env_cfg` faz `assert` de que nenhum campo `None` chega ao cfg (§11, D10).

```python
JUNTAS = (  # ordem dos motores do SDK = ordem do MJCF (I §2.2)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
TAREFAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR", "REORIENTAR", "LEVANTAR")  # one-hot (I §2.2)
FATIAS = ("ANDAR", "B", "C")
ALVOS = ("P0", "P1", "P2", "P3", "P4", "B1", "B2", "B3", "B4")  # índice 0..8; −1 = nenhum

@dataclass(frozen=True)
class Contrato:                      # I §2, §3; S §6.3, §19.1
    escala_acao: float = 0.25
    dt_fisica: float = 0.005
    decimacao: int = 4               # passo da política 0,02 s
    historico: int = 5
    # (nome, canais por passo, escala, histórico) na ordem do vetor do ator (I §2.1)
    blocos_ator: tuple = (("giro_base", 3, 0.2, 5), ("gravidade", 3, 1.0, 5),
                          ("comando", 3, 1.0, 5), ("juntas", 29, 1.0, 5),
                          ("vel_juntas", 29, 0.05, 5), ("acao", 29, 1.0, 5),
                          ("tarefa", 6, 1.0, 5), ("caixa", 11, 1.0, 5), ("mapa", 882, 1.0, 0))
    # (regex das juntas, kp, kd) — I §3; UNITREE_G1_29DOF_CFG, commit 4960b84
    pd: tuple = ((r".*_hip_pitch_joint", 100.0, 2.0), (r".*_hip_yaw_joint", 100.0, 2.0),
                 (r".*_hip_roll_joint", 100.0, 2.0), (r".*_knee_joint", 150.0, 4.0),
                 (r".*_ankle_(pitch|roll)_joint", 40.0, 2.0), (r"waist_yaw_joint", 200.0, 5.0),
                 (r"waist_(roll|pitch)_joint", 40.0, 5.0),
                 (r".*_(shoulder_.*|elbow|wrist_roll)_joint", 40.0, 1.0),
                 (r".*_wrist_(pitch|yaw)_joint", 40.0, 1.0))
    referencias: str = "g1_v3/dados/referencias.npz"

def dim_ator(c: Contrato = Contrato()) -> int: ...   # 1 447 (assert no teste)
def q_de_pe(c: Contrato = Contrato()) -> "np.ndarray": ...   # (29,), lê `referencias.npz`["q_de_pe"]
def caminho(rel: str) -> "Path": ...  # relativo à raiz do repo, a partir de __file__
```

As outras seções, com os campos e os valores iniciais. A coluna "fonte" diz de onde vem o número.

| seção | campo | valor | fonte |
|---|---|---|---|
| `Cena` | `caixa_meia_faixa` | (0,07; 0,13) | S §12 |
| | `caixa_n_variantes` | 8 | `g1_limpo/knobs.py:30` |
| | `caixa_massa_base` | 1,0 | S §11.2 |
| | `pad_atrito` | (1,0; 0,02; 0,001) | `g1_limpo/cena.py:245` |
| | `caixa_priority`, `caixa_atrito`, `robo_atrito` | 0; (0,3; 0,02; 0,001); (0,3; 0,02; 0,001) | PD-46 |
| | `destino_borda`, `destino_jitter_y`, `destino_giro_max_graus` | 0,15; 0,10; 10 | PD-47 |
| | `grupo_mobilia` | 1 | PD-9 |
| | `prateleira_*`, `caixa_xy`, `caixa_jitter_y`, `jitter_x_max`, `reset_base_manipula` | os valores do `g1_limpo/knobs.py` `Cena` e `Nivel` | cópia |
| | `reset_base_andar` | (("x", (−0,5; 0,5)), ("y", (−0,5; 0,5)), ("yaw", (−π; π))), no formato do `reset_base_manipula` | D7 (29/10) |
| | `prioridade_laje` | 3 | PD-46 (29/10: sai da `cena`) |
| | `njmax`, `nconmax`, `impratio`, `cone` | 800; 300; 1,0; "pyramidal" | `g1_limpo/knobs.py:103-106`; S §12 (L2) |
| | `extent` | 2,5 | `g1_limpo/knobs.py:86` |
| `Comando` | `frente_tetos` | (0,25; 0,5; 0,75; 1,0; 1,25; 1,5) | S §11.2 |
| | `tras_lado_teto_max` | 1,0 | S §11.2 |
| | `giro_tetos` | (0,25; 0,5; 0,75; 1,0) | S §11.2 |
| | `piso_velocidade` | 0,25 | S §11.2 ("entre 0,25 e o teto") |
| | `baixo` | (0,05; 0,25) | S §19, conflito 5 |
| | `reamostra_s` | (3,0; 8,0) | S §7 |
| | `janela_media_s` | 0,8 | S §19, conflito 5 |
| | `limiar_canal` | 0,02 | S §19, conflito 5 |
| | `deriva_pos_m`, `deriva_rumo_rad`, `ancora_s` | 0,10; 0,25; 1,5 | PD-15 |
| | `razao_portao` | 0,50 | S §11.3 |
| | `pedido_min_segmento` | 0,5 | `g1_limpo/comando.py:2507` |
| | `terreno_v_max` | 1,0 | PD-12 (célula terreno usa o reto do nível, até 1,0 m/s) |
| `Cadeia` | `espera_s` | (0,5; 1,0) | S §7 |
| | `carregar_b_s` | (4,0; 7,0) | S §7 |
| | `carregar_c_s` | (0,5; 1,0) | S §7 |
| | `andar_final_s` | 1,5 | S §7 |
| | `atraso_pegar_s`, `atraso_botar_s` | (0,0; 1,0); (0,0; 1,5) | S §7 |
| | `episodio_s` | 20,0 | S §7 |
| | `caixa_perdida_vira_andar` | True | PD-27 |
| | `pega_c_topo_min` | 0,30 | S §7 |
| `Detector` | `tol_pelve`, `tol_tronco_graus`, `tol_palma` | 0,03; 5,0; 0,05 | I §4.2 |
| | `estavel_s`, `juntas_paradas` | 0,5; 0,2 | I §4.2; PD-26 |
| `Alvos` | `p0_frente`, `palma_face`, `p3_alto`, `b1_alto` | 0,05; 0,035; 0,05; 0,05 | S §8.2 |
| | `tol_palma`, `tol_caixa_p4`, `tol_pelve_p4`, `tempo_p4` | 0,03; 0,05; 0,03; 0,5 | S §8.2 |
| | `tol_b1_horizontal`, `raio_b2`, `apoio_b2_mg`, `tempo_b2` | 0,03; 0,10; 0,8; 0,3 | S §8.2 |
| | `b3_afasta`, `tempo_b3`, `tol_pelve_b4`, `tol_tronco_b4_graus`, `tempo_b4` | 0,05; 0,5; 0,03; 5,0; 0,5 | S §8.2 |
| | `z_transporte` | 0,90 | S §8.2 (sobre o chão local, PD-21) |
| | `incl_inicio_graus`, `incl_fim_graus`, `delta_incl_graus` | 30; 15; 15 | S §8.2, §9.1 |
| | `desloc_p0p2_frac`, `delta_p0p2_frac` | 1,0 (× meia aresta); 0,5 | S §8.2; PD-23 |
| | `desloc_b2b4`, `delta_b2b4` | 0,02; 0,02 | S §8.2; PD-23 |
| | `faixa_tronco_graus`, `faixa_recuo`, `faixa_com` | 15; 0,08; 0,03 | S §8.2 |
| | `delta_tronco_graus`, `delta_recuo`, `delta_com` | 15; 0,08; 0,03 | PD-23 |
| `Recompensa` | `rastreio_linear`, `rastreio_angular`, `sigma_rastreio`, `piso_rastreio` | 1,0; 0,5; 0,5; −1,0 | S §9.2, §19 |
| | `sigma_zero_lin`, `sigma_zero_ang` | 0,10; 0,15 | PD-49 |
| | `passo`, `L_min`, `fator_T` | 0,5; 0,20; (0,8; 1,4) | S §19, conflito 6 |
| | `vivo` | 0,15 | S §9.2 |
| | `altura_pe`, `alvo_pe`, `std_pe`, `tanh_pe` | 1,0; 0,10; 0,05; 2,0 | S §9.2; `foot_clearance_reward` do unitree_rl_lab |
| | `escorrego` | −0,2 | S §9.2 |
| | `vz`, `giro_xy`, `vel_junta`, `acel_junta`, `taxa_acao`, `energia` | −2,0; −0,05; −0,001; −2,5e−7; −0,05; −2e−5 | S §9.2 |
| | `limite_junta`, `limite_mole` | −5,0; 0,95 | S §9.2 |
| | `desvio_bracos`, `desvio_cintura`, `desvio_quadril`, `tronco_plano`, `altura_base`, `alvo_base` | −0,1; −1,0; −1,0; −5,0; −10,0; 0,78 | S §9.2 |
| | `cerca_sola`, `cerca_giro`, `cerca_tronco`, `cerca_com`, `cerca_torque`, `cerca_pelve`, `cerca_recuo` | `None` (Parte 0, item 3) | S §9.3; PD-30 |
| | `lim_sola_graus`, `lim_giro_graus`, `lim_tronco_graus`, `lim_com`, `lim_torque`, `banda_torque`, `lim_pelve`, `lim_recuo` | 5; 30; 15; 0,03; 0,8; 0,2; 0,08; 0,08 | S §9.3 |
| | `recuo_carregar` | 0,25 m (o `recuo_IK` do CARREGAR na `cerca_recuo`) | D5 (29/10) |
| | `tau_max` | por regex: joelho e `hip_roll` 139; `hip_pitch`, `hip_yaw`, `waist_yaw` 88; cintura e tornozelo 50; ombro, cotovelo, `wrist_roll` 25; `wrist_pitch`, `wrist_yaw` 5 | S §9.3 |
| | `pares_torque`, `tau_par` | ((`waist_pitch_joint`, `waist_roll_joint`), (`left_ankle_pitch_joint`, `left_ankle_roll_joint`), (`right_ankle_pitch_joint`, `right_ankle_roll_joint`)); 50,0 | PD-45 |
| | `contato_fora`, `contato_fora_N` | −1,0; 1,0 | S §9.2 |
| | `contato_laje`, `laje_joelho_N`, `laje_saturacao_N` | −2,0; (50; 10) (início; refino, §6.9); 100 | S §9.2; `g1_limpo/knobs.py` `Contato` |
| | `freio`, `freio_fator`, `vel_max_parado`, `vel_max_andando`, `vel_max_correndo` | −6/29; 1,5; os do `g1_limpo/knobs.py` `Tarefa` | S §9.2, §11.4 |
| | `progresso`, `b_alvo` (9 valores, um por alvo) | `None`; `None` (Parte 0, item 3) | S §9.4 |
| | `gamma` | 0,99 | S §9.1 |
| | `sacudida`, `aperto`, `F_ref` | `None`; `None` (Parte 0, item 15); 49,0 | S §8.2, §11.5 |
| | `terminacao` | −200,0 | S §10 |
| | `perda_caixa`, `devolve_b_alvo` | `None`; `None` (Parte 0, item 3) | S §10 |
| | `sigma_pouso`, `altura_pouso`, `altura_pouso_inicio` | (1,0; 0,35); 0,05; 0,10 | S §9.1; D3 da Parte 0 (dono, 29/10): o fator entra aos 0,10 m e fica pleno aos 0,05 m |
| `Terminacao` | `incl_marcha`, `base_min` | 0,8; 0,20 | S §10 |
| | `incl_manip_graus`, `joelho_min` | 70; 0,05 | S §10 |
| | `caixa_folga_chao`, `v_solta` | o valor do `g1_limpo/knobs.py` `Terminacao`; 1,2 | S §10 |
| | `chegada` | (1,0; 0,25) | S §10 |
| | `hip_yaw_max`, `batente_frac` | 1,0; 1,0 | S §10 |
| `Balanceador` | `piso_tarefa`, `piso_sub`, `eps`, `portao`, `alfa` | 0,20; 0,10; 0,02; 0,75; 0,05 | S §11; PD-13 |
| | `iters_min_nivel`, `iters_refino`, `degraus` | 50; 100; 5 | PD-12; S §11.5 |
| | `passos_por_iteracao` | 24 | o `num_steps_per_env` do `rl_cfg()` (D10) e a guarda de iteração do termo `v3` (D4) leem este campo |
| `Niveis` | `pegar` | ((0,55; 1), (0,45; 2), (0,30; 3), (0,15; 4), (0,04; 5)) | S §11.2 |
| | `botar` | (0,55; 0,45; 0,30; 0,15; 0,04) | S §11.2 |
| `Terreno` | `linhas`, `tamanho` | 13; (8,0; 8,0) | PD-20; `terrains/config.py` |
| | `tetos_andar` | degrau 0,20; rampa 0,36; obstáculo 0,15 | S §11.2 |
| | `colunas` | (`plano`, `escada`, `escada_inv`, `rampa`, `rampa_inv`, `escada_aleatoria`, `irregular`, `obstaculos`, `caixas`) | D1; a ordem das colunas do gerador (29/10: o `eventos` lê daqui, sem importar a `cena`) |
| | `linha_max_carregar` | por coluna (nome, linha): `escada`, `escada_inv`, `rampa`, `rampa_inv`, `escada_aleatoria` 9; `obstaculos` 8; `caixas` 7. Uma coluna tem plataforma de 2 m se e só se está aqui | PD-20 |
| | `plataforma` | 2,0 | S §12 |
| | `borda` | 20,0 m (`border_width` do gerador) | D1 (29/10: sai da `cena`) |
| `DR` | `atrito_pe`, `massa_tronco`, `vies_encoder` | (0,3; 1,0); (−1,0; 3,0) kg; 0,015 rad | §6.8 startup; o tamanho da caixa mora em `Cena.caixa_meia_faixa` e `caixa_n_variantes` |
| | `atraso_inicio`, `atraso_portao`, `atraso_refino`, `degrau_atraso_refino` | (0; 1); (0; 2); (0; 3) passos; 3 | §6.8; PD-25, PD-28, PD-41 |
| | `ruido_obs` | (("giro_base", 0,2), ("gravidade", 0,05), ("juntas", 0,01), ("vel_juntas", 1,5)) | §6.5 (29/10: sai do `observacoes`); a capacidade dos buffers de atraso da observação é `atraso_refino[1]` |
| | `empurrao`, `periodo_empurrao_s` | 0,5 m/s (± em x e y); 5,0 | §6.8 intervalo |
| | `vies_imu_giro`, `vies_imu_incl_graus` | 0,05 rad/s; 1,0 | §6.8 ANDAR no portão |
| | `atrito_pad`, `atrito_laje`, `passo_mu_pad`, `mu_pad_min` | (0,6; 1,0); (0,3; 1,0); 0,025; 0,5 | §6.8; PD-46 |
| | `com_caixa`, `com_caixa_frac0`, `com_caixa_frac_max` | False; (0,2; 0,2); (0,3; 0,2) (xy; z, × meia aresta) | §6.8; PD-38: a chave liga o `dr.body_com_offset` |
| | `forca_caixa_N`, `forca_caixa_dur_s`, `forca_caixa_periodo_s` | (5; 20); (0,1; 0,3); (3; 8) | §6.8 CARREGAR |
| | `rfi` | (0,05; 0,1) × `tau_max` (início; refino) | §6.8 |
| | `pd`, `forca_motor`, `forca_motor_bracos`, `solref`, `solref_base` | (0,9; 1,1); (0,9; 1,1); (0,8; 1,1); (0,015; 0,03); 0,02 (faixas no degrau global 5; degrau 0 = 1,0 e `solref_base`) | §6.8 refino; PD-41 |
| `Estimador` | `taxa_hz`, `latencia_s`, `idade_max` | 30; 0,033; 2,0 | I §2.3; `lidar_sim/sensores/tipos.py` |
| | `fov_h_graus`, `fov_v_graus`, `alcance` | 69; 42; (0,28; 3,0) | PD-11 |
| | `vies_meia`, `vies_alvo_botar_xy`, `vies_alvo_botar_z`, `ori_persistente` | início (0,005; 0,01; 0,005; True); refino (0,02; 0,05; 0,03; True) | S §13 (dono, 15/10) |
| | `pontos_visiveis_min` | 3 | PD-11 |
| | `sigma_pos`, `sigma_ori_graus`, `perda`, `deriva_odo` | início (0,005; 1,0; 0,005; 0,002 m/√m); refino: 2% da distância, mediana 5,3°, 2,2%, 0,0075 m/√m | PD-18; S §13 |
| | `deriva_rumo` | início 0,002 rad/√m; refino 0,005 rad/√m | PD-18 |
| | `deriva_pos_giro`, `deriva_rumo_giro` | início (0,004 m/√rad; 0,0035 rad/√rad) = 1 cm e 0,5° por volta; refino (0,008; 0,014) = 2 cm e 2° por volta | PD-18 (dono, 28/10) |
| | `d435_pos`, `d435_pitch` | (0,0576235; 0,01753; 0,42987); 0,8307767 | `lidar_sim/sensores/montagem.py` |
| `Percepcao` | `ligada` | True | S §4 (`PERCEPCAO_LIDAR`) |
| | `celula`, `n_recorte`, `n_interno`, `altura_clip` | 0,10; 21; 51; 1,0 | I §2.4; PD-8 |
| | `raio_min` | 0,85 | PD-10 |
| | `raios_por_passo`, `alcance` | 205; (0,1; 6,0) | PD-5 |
| | `ruido_alcance`, `ruido_angulo_graus`, `perda`, `ruido_incl_graus` | 0,02; 0,15; 0,01; 0,5 | PERC §2 |
| | `sigma_ref`, `q_desloc`, `q_giro`, `q_tempo` | 0,05; 4e−4; 4e−4; 2,5e−5 | PD-6, PD-7 |
| | `sigma_fixo` | 0,02 | PD-42 |
| | `margem_corpo` | 0,03 | PD-10 |
| | `mid360_pos`, `mid360_quat` | (0,0002835; 0,00003; 0,428434); (0; 0,99967; 0; −0,02556) | PERC §2; `montagem.py` |
| `Refino` | `regras` | tuplas (regra, tarefas, seção, campo): `laje_joelho_N` (PEGAR, BOTAR; `recompensa.laje_joelho_N`); `incl_lim` (PEGAR, CARREGAR, BOTAR; `alvos`, (`incl_inicio_graus`, `incl_fim_graus`)); `chegada` (BOTAR; `terminacao.chegada`); `sigma_pouso` (BOTAR; `recompensa.sigma_pouso`). O campo é um par (início; refino), ou dois nomes (início, fim) | §6.9; S §11.5 |
| | `fator_por_degrau`, `tarefas_fator`, `degrau_batente` | (`aperto`, `sacudida`); (PEGAR, CARREGAR, BOTAR); 1 | §6.9: peso × d/5; batente e `hip_yaw` do degrau 1 em diante |

### 6.3 Funções puras

Assinaturas exatas. `E` = número de envs. Todo tensor no device do chamador. Convenções (29/10): quaternion
em wxyz (o do mjlab); `ids` é tensor long de índices de env (o `env_ids` do mjlab), nunca máscara; `gerador` é
`torch.Generator` ou `None`, e `None` desliga ruído, perda e sorteio de inclinação (teste determinístico);
`dt` é float; ângulo de rumo é (E,) em rad; os escalares dos knobs são float.

**`alvos.py`**

```python
class Referencias:
    """Referências macro da IK (decisão de 14/10). Lê `referencias.npz`; nunca `qpos` de pega."""
    def __init__(self, caminho: str, device: str): ...
    topos: torch.Tensor          # (6,)  0; 0,04; 0,15; 0,30; 0,45; 0,55
    meias: torch.Tensor          # (3,)  0,07; 0,10; 0,13
    def pelve_z(self, h, a) -> torch.Tensor: ...     # (E,) bilinear em (h, a), com clamp na grade
    def tronco(self, h, a) -> torch.Tensor: ...      # (E,) rad
    def recuo(self, h, a) -> torch.Tensor: ...       # (E,) m, pelve → centro da caixa, no plano
    def caixa_transporte_b(self, a) -> torch.Tensor: ...    # (E,3) centro da caixa no frame da base, pose de pé com caixa
    def palmas_com_caixa_b(self, a) -> torch.Tensor: ...    # (E,2,3) palmas no frame da base, pose de pé com caixa
    def q_com_caixa(self, a) -> torch.Tensor: ...           # (E,29) juntas da pose de pé com caixa
    def pelve_com_caixa(self, a) -> torch.Tensor: ...       # (E,) altura da pelve na pose de pé com caixa
    palmas_de_pe_b: torch.Tensor   # (2,3) pose de pé sem caixa
    pes_de_pe_b: torch.Tensor      # (2,3) sites dos pés na pose de pé, frame da pelve (PD-16)
    cantos_sola: torch.Tensor      # (4,3) cantos da sola no frame do `*_ankle_roll_link` (PD-35)
    q_de_pe: torch.Tensor          # (29,)
    pelve_de_pe: float             # 0,789
    site_pe_de_pe: float           # altura do site do pé sobre a sola, na pose de pé

def frame_da_caixa(quat_caixa_w, rumo_base) -> torch.Tensor: ...   # (E,3,3), PD-29
def inclinacao_da_caixa(quat_caixa_w) -> torch.Tensor: ...         # (E,) rad, face mais perto da vertical
def palmas_alvo_caixa(indice, a, k: Alvos) -> torch.Tensor: ...    # (E,2,3) no frame da caixa:
    # P0: (−(a+p0_frente), ±(a+palma_face), 0); P1, P2: (0, ±(a+palma_face), 0);
    # B3: (0, ±(a+palma_face+b3_afasta), 0); outros: 0 (não usados; nunca NaN, §6.6)
def caixa_alvo_w(indice, caixa_na_pega_w, alvo_soltura_w, base_pos_w, rumo, chao_z, a, ref, k) -> torch.Tensor: ...
    # (E,3): P3 = caixa_na_pega + (0,0,p3_alto); P4 = base + R(rumo)·caixa_transporte_b(a), z = chao_z + z_transporte;
    # B1 = alvo_soltura + (0,0,b1_alto); B2 = alvo_soltura; outros: 0 (nunca NaN, §6.6)
def laje_destino(recuo, theta, y, meia_xy, borda) -> tuple[torch.Tensor, torch.Tensor]: ...
    # (alvo_b (E,2), centro_b (E,2)) no frame da base, PD-47: alvo em (recuo + e(θ), y), com
    # e(θ) = borda·cos θ + meia_xy·sen|θ| − borda; centro = alvo + R(θ)·(meia_xy − borda, 0)
def avanca(indice, ok_agora, relogio, dt, tempos) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    # (novo_indice, novo_relogio, avancou): avança quando `ok_agora` vale pelo tempo do alvo
    # (P2: 0; P4: tempo_p4; B2: tempo_b2; B3: tempo_b3; B4: tempo_b4); `ok_agora` falso zera o relógio
def detecta_fim(pelve_sobre_pe, tronco, palmas_b, vel_juntas, ref_pelve, ref_palmas_b,
                relogio, dt, k: Detector) -> tuple[torch.Tensor, torch.Tensor]: ...
    # (aviso, novo_relogio): I §4.2; o relógio conta enquanto a pose fecha E a média de |q̇| nas 29 juntas
    # fica abaixo do limiar a cada passo; avisa com relógio ≥ estavel_s (sem estado além do relógio; o real
    # usa a mesma regra, e por isso nunca avisa antes do treino). Depois do B4, `avanca` leva o índice a −1
```

**`potencial.py`**

```python
def nucleo(d, sigma) -> torch.Tensor: ...                 # exp(−(d/σ)²)  [C20]
def rampa(x, x_lim, delta) -> torch.Tensor: ...           # clip(1 − (x − x_lim)/delta, 0, 1)  (S §9.1, B04)
def phi(k_principal, portoes: list[torch.Tensor]) -> torch.Tensor: ...   # k · Π g
def progresso(phi_novo, phi_velho, gamma) -> torch.Tensor: ...           # γΦ(s') − Φ(s)
def fator_pouso(v_vertical, fundo_sobre_laje, sigma, altura, altura_inicio) -> torch.Tensor: ...
    # 1 − s·(1 − exp(−(v/σ)²)), s = clip((altura_inicio − fundo)/(altura_inicio − altura), 0, 1): 1 acima de
    # altura_inicio, pleno abaixo de altura (S §9.1, A10). ⚠ Gradual (D3 da Parte 0, dono 29/10): o degrau
    # em `altura` cortava Φ no passo de entrada na janela
def custo_aperto(F_esq, F_dir, F_ref) -> torch.Tensor: ...  # 1 − exp(−min(F_esq,F_dir)/F_ref); peso −w (S §8.2, B02)
def dobradica(x, limite) -> torch.Tensor: ...             # relu(x − limite)  (cercas, S §9.3)
def excesso_de_junta(q, centro, meio, mole) -> torch.Tensor: ...  # Σ relu(|q−c|/m − mole)/(1−mole)  (S §9.2)
def rastreio_normalizado(erro2, c, v2, sigma, sigma_zero, limiar, piso) -> torch.Tensor: ...
    # c ≥ limiar: (exp(−e²/σ²) − exp(−c²/σ²)) / (1 − exp(−c²/σ²)), com piso; c < limiar: exp(−v²/σ₀²) − 1  (S §19, conflito 5; PD-49)
def L_alvo(v_cmd, L_min, fator_T) -> torch.Tensor: ...    # |v|·clamp(2·L_min/|v|, 0,8, 1,4)/2  (S §19, conflito 6)
```

**`balanceador.py`**

```python
@dataclass
class Celula:
    tarefa: str        # "ANDAR" | "PEGAR" | "CARREGAR" | "BOTAR"
    tipo: str          # ANDAR: "reto","parado","baixo","giro_andando","giro_lugar","terreno";
                       # CARREGAR: "reto","giro_lugar","arco","terreno"; PEGAR: "pegar"; BOTAR: "botar"
    aberta: bool
    s: float           # EMA do sucesso (PD-13)
    n_medidas: int

class Curriculo:
    """Estado inteiro do currículo: células, níveis, fatias, refino. Puro e serializável."""
    def __init__(self, k_bal: Balanceador, k_niv: Niveis, k_com: Comando, k_ter: Terreno): ...
    celulas: list[Celula]   # público; o `celula_id` é o índice nesta lista, na ordem de abertura da B3:
                            # 0..5 ANDAR (reto, parado, baixo, giro_andando, giro_lugar, terreno),
                            # 6..9 CARREGAR (reto, giro_lugar, arco, terreno), 10 PEGAR, 11 BOTAR
    it: int                 # o contador próprio de iterações
    def registra(self, celula_id: int, n: int, n_sucesso: int) -> None: ...   # acumula na iteração
    def fecha_iteracao(self) -> None: ...
        # soma 1 ao contador próprio `self.it` (vai no `estado()`), aplica a EMA (PD-13), abre
        # células e fatias (S §11.1, §11.2), sobe níveis (PD-12), conta its no portão e avança o
        # refino (S §11.5). EMA por célula com n > 0 na iteração: x = n_sucesso/n; com n_medidas = 0
        # (célula recém-aberta, nível recém-subido) s = x; senão s += alfa·(x − s); n = 0 não muda s.
        # s começa em 0. ⚠ Não recebe o `it` do env: o `common_step_counter` volta a 0 na
        # retomada, e a carência contada da última subida salva travaria os níveis
    def fracoes_fatias(self) -> torch.Tensor: ...          # (3,) ANDAR, B, C; fatia fechada = 0
    def fracoes_celulas(self, tarefa: str) -> tuple[list[int], torch.Tensor]: ...
    def sorteia_fatias(self, n: int, gerador) -> torch.Tensor: ...     # (n,) long
    # ⚠ Os dois sorteios rodam no device do `gerador` (o `v3_gerador` mora na GPU do env; o `multinomial` de um
    # tensor da CPU com gerador CUDA falha) e devolvem no device do gerador; sem gerador, CPU (29/10).
    def id_celula(self, tarefa: str, tipo: str) -> int: ...  # o id em `celulas`; o `eventos` e o `comando` usam (29/10)
    def sorteia_celulas(self, tarefa: str, n: int, gerador, sem_tipo: str | None = None) -> torch.Tensor: ...
        # (n,) ids de célula; com `sem_tipo`, a fração da célula desse tipo vale 0 antes do sorteio (29/10:
        # o `twist` sorteia sem a célula `terreno` fora do episódio de terreno, §6.4 `v3_celula`)
    def nivel(self, tarefa: str) -> int: ...
    def nivel_terreno(self, tarefa: str) -> int: ...       # linha do gerador dentro da célula `terreno`: ANDAR
        # 0..linhas−1, CARREGAR 0..max(linha_max_carregar) (a D7 corta por coluna), PEGAR e BOTAR 0;
        # sobe pela regra dos níveis (célula `terreno` no portão, carência `iters_min_nivel`)
    def niveis_abertos(self, tarefa: str) -> int: ...      # o degrau do freio (S §11.4)
    def degrau(self, tarefa: str) -> int: ...              # 0..5
    def estado(self) -> dict: ...                          # para o checkpoint; só tipos do Python (int, float,
                                                           # bool, str, list, dict); carrega(estado()) é identidade
    def carrega(self, estado: dict) -> None: ...

def fracoes(deficits, piso, eps) -> torch.Tensor: ...      # piso + (1 − n·piso)(D+ε)/Σ(D+ε)
def deficit(s, portao) -> torch.Tensor: ...                # max(0, portão − s)
```

Regras de abertura (S §11.1, §11.2; PD-12):

| o que abre | quando |
|---|---|
| fatia ANDAR, células `reto` e `parado` do ANDAR, nível 0 | no começo |
| `baixo` → `giro_andando` → `giro_lugar` (ANDAR) | cada uma quando todas as abertas do ANDAR passam no portão |
| nível k+1 do ANDAR (tetos de velocidade) | todas as células de velocidade abertas no portão, todas abertas, ≥ `iters_min_nivel` desde a última subida |
| `terreno` (ANDAR) | depois do último nível de velocidade, com todas no portão |
| fatia B (PEGAR nível 0 e CARREGAR `reto`) | `reto` e `parado` do ANDAR no portão |
| `giro_lugar` → `arco` (CARREGAR), níveis do CARREGAR | mesma regra do ANDAR, dentro do CARREGAR |
| `terreno` (CARREGAR) | depois do último nível do CARREGAR, e quando o `terreno` do ANDAR passa no portão no nível de terreno 0 |
| fatia C (BOTAR nível 0) | PEGAR nível 0 no portão |
| nível k+1 do PEGAR e do BOTAR | célula no portão, ≥ `iters_min_nivel` |
| refino da tarefa | último nível aberto (no ANDAR e no CARREGAR, também o último nível de terreno) e a média das células da tarefa ≥ portão por `iters_refino` its seguidas (uma it fora do portão zera a contagem) |
| degrau seguinte do refino | mesmo critério, contado de novo no degrau atual; nunca desce |

**`estimador.py`**

```python
def reduz_simetria_cubo(q_rel) -> torch.Tensor: ...   # (E,4) wxyz → (E,3) vetor de rotação de menor ângulo
                                                      # entre as 24 rotações do cubo (I §2.3)
def visivel(cam_pos_w, cam_rot_w, pontos_w, capsulas_w, raios, k: Estimador) -> torch.Tensor: ...
    # (E,) bool; pontos_w (E,9,3); capsulas_w (E,K,2,3); PD-11
class Odometria:
    def __init__(self, n, device, k: Estimador, gerador): ...
    def reinicia(self, ids, pos_w, quat_w): ...                # pose estimada = a verdadeira; Δ do passo = 0
    def passo(self, pos_w, quat_w, deriva: dict): ...
        # o incremento verdadeiro do passo no frame da base (dxy_b, dψ, dz), com ruído pelo
        # deslocamento e pelo ângulo girado, integrado pela pose ESTIMADA: pos += (R(ψ_est)·dxy_b, dz),
        # ψ_est += dψ; roll e pitch saem do IMU (os verdadeiros); deriva: odo, rumo, pos_giro,
        # rumo_giro da magnitude corrente (PD-18). ⚠ O erro soma no frame da base, como no real:
        # somado no mundo, o Δ do mapa giraria pelo erro absoluto de rumo a todo passo
    def delta_b(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
        # (dxy_b (E,2), dψ (E,), dz (E,)) do último passo, com o erro; o filtro transporta por ele
    pos_w: torch.Tensor; quat_w: torch.Tensor                  # (E,3), (E,4) estimadas
class EstimadorCaixa:
    def __init__(self, n, device, k: Estimador, gerador): ...
    def reinicia(self, ids): ...
    def passo(self, t, caixa_pos_w, caixa_quat_w, meia, visivel, magnitude: dict): ...
        # leitura a taxa_hz, entregue depois de latencia_s; perda e ruído pela magnitude corrente
    def saida_b(self, odo: Odometria, base_quat_w) -> torch.Tensor: ...  # (E,8): pos_b 3, rotvec 3, meia 1, idade 1
def alvo_no_frame_da_base(alvo_w, odo: Odometria) -> torch.Tensor: ...  # (E,3)
```

**`comando.py` e `observacoes.py` (núcleos puros; 29/10)**

```python
def sorteia_velocidade(tipo, nivel, tarefa, n, gerador, k: Comando) -> torch.Tensor: ...  # (n,3): vx, vy, wz (D3)
@dataclass
class TabelaTrechos:          # a tabela da D3 achatada nas três fatias; tensores (T,)
    tarefa: torch.Tensor      # long, índice em TAREFAS publicado no trecho
    por_detector: torch.Tensor  # bool: o trecho acaba pelo aviso do detector mais o atraso
    proximo: torch.Tensor     # long, −1 = fim da cadeia
    inicio: torch.Tensor      # (3,) long, o primeiro trecho de ANDAR, B e C
def monta_tabela_trechos(k_cad: Cadeia, device) -> TabelaTrechos: ...
def avanca_trecho(trecho, relogio, duracao, aviso, atraso, dt, tabela: TabelaTrechos
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
    # (novo_trecho, novo_relogio, avancou, fim). trecho (E,) long; relogio (E,) float: o tempo no trecho por
    # tempo, ou o tempo desde o aviso no trecho por detector (0 antes); duracao (E,) a duração sorteada do
    # trecho por tempo; aviso (E,) bool, o detector já avisou neste trecho (o invólucro guarda o latch);
    # atraso (E,) o atraso sorteado na abertura do trecho por detector. Por tempo: relógio += dt, avança com
    # relógio ≥ duração. Por detector: relógio += dt só com aviso, avança com aviso e relógio ≥ atraso. Ao
    # avançar, relógio = 0; fim = avançou e proximo = −1 (o trecho fica no último). A tarefa publicada é
    # tabela.tarefa[trecho]: no atraso vale a do trecho corrente (o one-hot fica em BOTAR no atraso do BOTAR)
def tempo_da_ancora(t_abertura, t_empurrao, ancora_s) -> torch.Tensor: ...
    # (E,) max(t_abertura, t_empurrao) + ancora_s; sem empurrão no segmento, t_empurrao = t_abertura (PD-15)
def sucesso_do_segmento(cmd, int_proj, int_cmd, deriva_pos, deriva_rumo, t_fecho, t_ancora, perdeu,
                        carregar, k: Comando) -> tuple[torch.Tensor, torch.Tensor]: ...
    # (registra, sucesso), (E,) bool. cmd (E,3) o comando do segmento; int_proj, int_cmd (E,2) [linear, giro]:
    # Σ(v·ĉ)dt e Σ|c|dt, Σ(ω·sinal)dt e Σ|wz|dt; deriva_pos (E,) m e deriva_rumo (E,) rad desde a âncora;
    # t_fecho e t_ancora (E,) s; perdeu, carregar (E,) bool. registra = t_fecho ≥ t_ancora. Canal ativo
    # (|c| ≥ limiar_canal): int_proj/int_cmd ≥ razao_portao. Canal linear zero: deriva_pos < deriva_pos_m;
    # canal giro zero: deriva_rumo < deriva_rumo_rad. sucesso = todos os canais E (não carregar OU não perdeu)

def passo_de_percepcao(filtro, dist, dirs_s, sensor_pos_w, sensor_rot_w, base_pos_w, rumo, z_pe_min_w,
                       incl_imu, capsulas_w, raios, dxy_b, dpsi, z_ref, dt, k: Percepcao, gerador
                       ) -> torch.Tensor: ...
    # (E,882): pontos_no_frame_do_mapa → filtra_corpo (pontos_w, capsulas_w, sensor_pos_w) →
    # filtro.prediz(dxy_b, dpsi, z_ref, dt) → filtro.atualiza(pontos, validos, dist) → filtro.recorte()
def _calcula_percepcao(env) -> torch.Tensor: ...  # lê o env (D2) e chama passo_de_percepcao
    # `atualiza_percepcao(env)` é só a guarda: chama `_calcula_percepcao` uma vez por common_step_counter
def referencia_do_alvo(indice, h, a, ref: Referencias, k: Alvos) -> torch.Tensor: ...
    # (E,8): palmas_alvo_caixa(indice, a, k) achatadas (6); pelve e tronco da referência do alvo:
    # ref.pelve_z(h, a) e ref.tronco(h, a) de P0 a P3 e de B1 a B3, ref.pelve_com_caixa(a) e 0 no P4,
    # ref.pelve_de_pe e 0 no B4; zero com indice −1 (§6.6). h = v3_topo, a = v3_meia
```

**`recompensas.py`, `terminacoes.py` e `cena.py` (núcleos e nomes; 29/10)**

O núcleo de cada termo da tabela da D5 se chama `n_<termo>` e recebe só tensores e escalares; o invólucro
`<termo>(env, tarefas, ...)` lê o env, chama o núcleo e aplica `curriculo.mascara` (§6.6) e, nos termos que pagam
positivo (`vivo`, `altura_pe`, `passo`, rastreios), `torch.where(v3_perdeu, 0, r)`. Ângulos em graus, comprimentos
em metros, torque em fração. As assinaturas que o teste usa:

```python
def comprimento_do_passo(p_toque_w, p_apoio_w, c_w, p0_toque_b, p0_apoio_b, c_b) -> torch.Tensor: ...
    # (E,) L = (p_toque − p_apoio)·ĉ_w − (p0_toque − p0_apoio)·ĉ_b (PD-16); posições (E,3) ou (3,); ĉ unitário (E,3)
def n_passo(L, L_alvo, r_lin, c_xy, limiar) -> torch.Tensor: ...     # min(relu(L)/L_alvo, 1)²·max(r_lin, 0); 0 com c_xy < limiar
def n_vivo(perdeu) -> torch.Tensor: ...                              # 1 − perdeu (float)
def n_altura_pe(h, v_xy, alvo, std, tanh_k) -> torch.Tensor: ...     # h (E,2), v_xy (E,2,2): exp(−Σ (h − alvo)²·tanh(tanh_k·‖v‖)/std)
def n_cerca_sola(incl_graus, contato, lim_graus) -> torch.Tensor: ...  # (E,2), (E,2) bool: Σ_pés em contato relu(incl − lim)
def n_cerca_giro(q_yaw_E, q_yaw_D, lim_graus) -> torch.Tensor: ...  # rad na entrada: relu(max(−q_E,0)° − lim) + relu(max(q_D,0)° − lim)
def n_cerca_tronco(incl_graus, tronco_ik_graus, lim_graus) -> torch.Tensor: ...  # relu(incl − (tronco_ik + lim))
def n_cerca_com(com_xy, cantos_xy, dois_pes, lim) -> torch.Tensor: ...  # (E,2), (E,8,2), (E,) bool: relu(lim − margem)·dois_pes;
    # margem = distância com sinal do CoM às arestas do fecho convexo dos 8 cantos (positiva dentro) (PD-35)
def n_cerca_torque(tau, tau_max, pares, tau_par, lim, banda) -> torch.Tensor: ...  # tau (E,29), tau_max (29,), pares (P,2) long:
    # Σ_soltas relu(|τ|/τ_max − lim)/banda + Σ_pares relu((|τ_p| + |τ_r|)/tau_par − lim)/banda (PD-45)
def n_cerca_pelve(h_pelve, pelve_ik, lim) -> torch.Tensor: ...       # relu(|h − ik| − lim)
def n_cerca_recuo(recuo, recuo_ik, lim) -> torch.Tensor: ...         # relu(|r − ik| − lim)
# terminacoes.py
def n_queda(tarefa, incl_rad, z_pelve_w, z_chao_pelve, z_joelho_w, z_chao_joelho, k: Terminacao) -> torch.Tensor: ...
    # (E,) bool. ANDAR e CARREGAR: incl > incl_marcha ou z_pelve − z_chao_pelve < base_min; PEGAR e BOTAR: incl >
    # incl_manip_graus (em rad) ou z_joelho − z_chao_joelho < joelho_min. Os z do chão vêm de `altura_pelve` e
    # `altura_joelhos` (o terreno, e não o mundo); tarefa (E,) long, índice em TAREFAS
def n_fim_da_cadeia(passos, passos_max, fim) -> torch.Tensor: ...    # (E,) bool: passos ≥ passos_max ou fim
# cena.py (nomes do g1_limpo, com "laje" no lugar de "prateleira")
def spec_caixa(c: Cena) -> mujoco.MjSpec: ...; def spec_laje(c: Cena) -> mujoco.MjSpec: ...  # geoms no grupo c.grupo_mobilia
def terreno_cfg(k: Knobs) -> TerrainEntityCfg: ...  # `sub_terrains` com as chaves de `Terreno.colunas`, na ordem
```

"todas" na D5 = ANDAR, PEGAR, CARREGAR, BOTAR (REORIENTAR e LEVANTAR são canais reservados do one-hot, sem env).
Na fase D o cfg tem só os termos da tabela da D5; `contato_laje`, `freio`, `progresso`, `sacudida`, `aperto` e
`perda_caixa` entram nas fases E e H. Teto do CARREGAR na D1: escadas e rampas t = linha/(R−1); `obstaculos`
idem; `caixas` (`random_spread_boxes`) escala a altura por 0,2 + 0,8·t (`primitive_terrains.py:794`): a linha 7
dá 0,15·0,667 = 0,10. Registro: `mjlab.tasks.registry.list_tasks()`, `load_env_cfg(id)`, `load_rl_cfg(id)`; a
SemPercepcao é `make_env_cfg(k)` com `dataclasses.replace(k, percepcao=replace(k.percepcao, ligada=False))`.
O `class_name` da v3 é o do algoritmo: `rl_cfg().algorithm.class_name == algoritmo.CAMINHO`.

**`eventos.py` (núcleo puro; 29/10)**

```python
def escolhe_celula_terreno(fatia, terreno_ep, nivel_terreno, k_ter: Terreno, gerador
                           ) -> tuple[torch.Tensor, torch.Tensor]: ...
    # (coluna (E,) long, linha (E,) long). fatia (E,) long, índice em FATIAS; terreno_ep (E,) bool, o episódio
    # de terreno já sorteado fora pela fração da célula `terreno`; nivel_terreno (E,) long, o
    # `Curriculo.nivel_terreno` da tarefa (ANDAR na fatia ANDAR; CARREGAR na B); gerador torch.Generator
    # (obrigatório: é sorteio). ANDAR de terreno: coluna uniforme em 1..len(colunas)−1, linha uniforme em
    # 0..nível. B de terreno: coluna uniforme entre as de `linha_max_carregar`, linha uniforme em
    # 0..min(nível, linha_max_carregar[coluna]). Sem terreno ou fatia C: coluna 0, linha 0
```

**`percepcao_lidar.py`**

```python
class PadraoMid360:
    def __init__(self, caminho_npz: str, device: str): ...   # `mid360_passos.npz`["dirs"] (200, 205, 3)
    def direcoes(self, passo: int) -> torch.Tensor: ...      # (205,3) no frame do sensor
# Formas (29/10): dist (E,R); dirs_s (R,3) no frame do sensor (o mesmo padrão para todos os envs);
# sensor_pos_w (E,3); sensor_rot_w (E,3,3), sensor → mundo; base_pos_w (E,3); rumo (E,); z_pe_min_w (E,),
# o z do ponto mais baixo das solas; incl_imu (E,3,3). Os giros de inclinação (o fixo `incl_imu` e o sorteado
# por leitura) agem no raio em volta do sensor: p = sensor_pos_w + G_leitura·incl_imu·(sensor_rot_w·dirs_s·dist).
def pontos_no_frame_do_mapa(dist, dirs_s, sensor_pos_w, sensor_rot_w, base_pos_w, rumo,
                            z_pe_min_w, incl_imu, k: Percepcao, gerador
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    # (pontos (E,R,3), pontos_w (E,R,3), validos (E,R)): ruído de alcance e de ângulo, perda, alcance,
    # o giro fixo do IMU do episódio (incl_imu (E,3,3) = `v3_incl_imu`, o MESMO da observação da
    # gravidade; PD-42) e um giro de inclinação sorteado por leitura (σ ruido_incl_graus, PERC §2);
    # `pontos`: x e y no frame do rumo, z = altura do ponto − z do pé mais baixo (I §2.4 passo 3);
    # `pontos_w`: os MESMOS pontos, com o mesmo ruído, no frame de `sensor_pos_w` (o mundo no sim, o
    # da odometria no real), para o `filtra_corpo`
def filtra_corpo(pontos, validos, capsulas, raios, sensor_pos, raio_min,
                 margem) -> torch.Tensor: ...      # validos (E,R); PD-10; pontos, cápsulas e sensor no MESMO frame
    # pontos (E,R,3); validos (E,R) bool; capsulas (E,K,2,3) os extremos p0, p1 de cada cápsula (como no
    # `visivel`); raios (K,); sensor_pos (E,3); raio_min e margem float. Devolve o novo `validos` (E,R) bool:
    # sai o ponto a menos de raio + margem de uma cápsula e o ponto a menos de raio_min do sensor
class FiltroDeElevacao:
    def __init__(self, n, device, k: Percepcao): ...
    def reinicia(self, ids): ...                                # altura 0, variância ∞ (nunca vista); sem z_ref anterior
    # Formas: prediz(dxy_rumo (E,2), dpsi (E,), z_ref (E,), dt float); atualiza(pontos (E,R,3), validos (E,R)
    # bool, dist (E,R)); os pontos do `atualiza` estão no frame do mapa (a saída `pontos` acima)
    def prediz(self, dxy_rumo, dpsi, z_ref, dt): ...            # transporte no plano + PD-7; depois, nas células vistas,
                                                                # h −= z_ref − z_ref anterior (transporte vertical, PD-7);
                                                                # o primeiro passo depois do `reinicia` usa 0
    def atualiza(self, pontos, validos, dist): ...              # Kalman por célula; média dos z por célula e leitura,
                                                                # R com o erro comum da leitura e piso σ_fixo² (PD-42);
                                                                # d̄ = média de `dist` (E,R) dos pontos válidos da célula
    def recorte(self) -> torch.Tensor: ...                      # (E,882): 441 alturas, depois 441 confianças (I §2.4)
```

### 6.4 Buffers do env

Todos com prefixo `v3_`, criados por `curriculo.garante_v3(env)` (cópia do padrão de
`g1_limpo/curriculo.py:61-97`). Nenhum termo cria buffer próprio no env.

| buffer | forma | escrito por | lido por |
|---|---|---|---|
| `v3_cur` | `balanceador.Curriculo` | `curriculo` | todos |
| `v3_gerador` | `torch.Generator` no device do env, semente = `env.cfg.seed` (0 sem semente) (29/10) | `curriculo.garante_v3` | `estimador`, `eventos`, `comando`, `observacoes` |
| `v3_iter` | int, a última iteração vista pelo termo `v3` (29/10) | `curriculo.v3` | `curriculo` |
| `v3_ref` | `alvos.Referencias` (29/10) | `curriculo.garante_v3` | `observacoes`, `recompensas`, `comando`, `metricas` |
| `v3_fatia` | (E,) long | `eventos.sorteia_fatia` | `comando`, `eventos` |
| `v3_trecho` | (E,) long | `comando.cadeia` | `comando`, `metricas` |
| `v3_tarefa` | (E,) long, índice em `TAREFAS` | `comando.cadeia` | todos |
| `v3_celula` | (E,) long; −1 = sem célula. O episódio é de terreno se e só se `scene["terrain"].terrain_types` ≠ 0 (coluna 0 = `plano`), escrito pelo `eventos.sorteia_fatia` antes do reset dos comandos. Num trecho que anda de episódio de terreno, o `twist` fixa a célula `terreno` da tarefa; fora dele, sorteia com `sem_tipo="terreno"` (29/10) | `comando.twist` (o único) | `comando`, `curriculo` |
| `v3_alvo_idx` | (E,) long, −1 a 8 | `comando.cadeia` | `recompensas`, `observacoes`, `metricas` |
| `v3_pegou`, `v3_perdeu` | (E,) bool | `comando.cadeia` | `recompensas`, `terminacoes` (privilegiado, só sim) |
| `v3_meia`, `v3_massa` | (E,) | `eventos` | todos |
| `v3_topo`, `v3_topo_destino` | (E,) topo da laje sobre o chão local | `eventos` | `comando`, `recompensas` |
| `v3_caixa_pega_w`, `v3_alvo_w` | (E,3) | `comando.cadeia` | `observacoes`, `recompensas` |
| `v3_estimador`, `v3_odo` | objetos de `estimador.py`, criados pelo `garante_v3` com o `v3_gerador` (a `v3_odo` na fase D; o `v3_estimador` na fase E) | `observacoes.atualiza_percepcao` (estimador); `comando.cadeia` (odometria, PD-44) | `observacoes`, `metricas` |
| `v3_filtro`, `v3_padrao` | objetos de `percepcao_lidar.py` | `observacoes.atualiza_percepcao` | `observacoes` |
| `v3_percepcao_passo`, `v3_mapa_obs`, `v3_caixa_obs` | int, o `common_step_counter` da última atualização; (E,882); (E,11) | `observacoes.atualiza_percepcao` | os termos `mapa` e `caixa` do ator e do crítico, `metricas` |
| `v3_freio`, `v3_degrau` | (E,) long, da tarefa corrente | `comando.cadeia` | `recompensas`, `terminacoes` |
| `v3_fim`, `v3_aviso` | (E,) bool: fim da cadeia; aviso do detector de fim | `comando.cadeia` | `terminacoes`, `metricas` |
| `v3_sigma`, `v3_relogio_alvo`, `v3_phi` | (E,): σ do alvo corrente [C20]; relógio do avanço; Φ do passo anterior | `comando.cadeia`; `recompensas.progresso` (`v3_phi`) | `recompensas` |
| `v3_avancou`, `v3_b_pagos` | (E,) bool; (E,) soma dos `B_alvo` pagos desde o P2 | `comando.cadeia`; `recompensas` | `recompensas` |
| `v3_p4_fechado`, `v3_b4_fechado`, `v3_perdeu_agora` | (E,) bool | `comando.cadeia` | `comando`, `terminacoes`, `recompensas` |
| `v3_ponto_apoio`, `v3_vz_caixa_ant`, `v3_incl_max_seg` | (E,3); (E,); (E,) | `comando.cadeia` | `recompensas`, `terminacoes`, `comando.twist` |
| `v3_vies_giro`, `v3_incl_imu` | (E,3); (E,3,3) giro fixo do IMU | `eventos.vies_imu` | `observacoes` (`giro_base`, `gravidade`) |
| `v3_t_empurrao`, `v3_ancora` | (E,) instante do último empurrão no episódio, em s desde o começo do episódio (−∞ sem empurrão; o `sorteia_fatia` volta a −∞ no reset); (E,4) pose da âncora do canal zero (x, y, rumo, instante). O evento grava `episode_length_buf · step_dt`; o `twist` mede no relógio próprio, zerado no reset do comando, porque o mjlab reseta os comandos antes de zerar o `episode_length_buf` (`manager_based_rl_env.py:581-590`); os dois diferem em até 1 passo (29/10) | `eventos.empurrao`; `comando.twist` | `comando.twist` (PD-15) |

### 6.5 Termos (nomes exatos, na ordem do dict)

**Comandos:** `cadeia` (primeiro), `twist`.

**Observações do ator** (`actor`, `enable_corruption=True`): `giro_base`, `gravidade`, `comando`,
`juntas`, `vel_juntas`, `acao`, `tarefa`, `caixa`, `mapa`. Histórico 5 em todos menos `mapa`.
Ruído: `giro_base` ±0,2, `gravidade` ±0,05, `juntas` ±0,01 (com `biased=True`), `vel_juntas` ±1,5.
Atraso 0–1 em `giro_base`, `gravidade`, `juntas`, `vel_juntas` e `mapa` (PD-25, PD-28); os cinco
nascem com capacidade 0–3 (D2).

**Observações do crítico** (`critic`, sem corrupção): os nove do ator, sem ruído e sem atraso;
depois `vel_lin_base` (3, histórico 5); depois, sem histórico: `caixa_verdade` (7: posição da caixa
no frame da base 3, rotvec do quat da caixa no frame da base 3, meia aresta 1; zero sem caixa),
`alvo_verdade` (3), `alvo_macro` (10: one-hot de P0–B4 e "nenhum"), `referencia` (8: palmas 6, pelve
1, tronco 1), `niveis` (4, nível / último índice do nível de cada tarefa, a conta do `Curriculo`), `degrau` (1, /5), `massa` (1, /5 kg), `forca_caixa` (1, /100
N); e por ÚLTIMO `tarefa_critico` (6, one-hot da tarefa corrente, sem histórico; o grupo da
vantagem, D9).

**Recompensas:** `rastreio_linear`, `rastreio_angular`, `passo`, `vivo`, `altura_pe`, `escorrego`,
`vz`, `giro_xy`, `vel_junta`, `acel_junta`, `taxa_acao`, `energia`, `limite_junta`, `desvio_bracos`,
`desvio_cintura`, `desvio_quadril`, `tronco_plano`, `altura_base`, `cerca_sola`, `cerca_giro`,
`cerca_tronco`, `cerca_com`, `cerca_torque`, `cerca_pelve`, `cerca_recuo`, `contato_fora`,
`contato_laje`, `freio`, `progresso`, `sacudida`, `aperto`, `perda_caixa`, `terminacao`.

**Terminações:** `time_out`, `queda`, `caixa_perdida`, `chegada`, `batente`.

**Eventos:** reset, nesta ordem: `fatia`, `base`, `juntas`, `cena`, `carga`, `atrito_pad`,
`atrito_laje`, `vies_imu`, `pd`, `forca_motor`, `solref`, `com_caixa`, `filtros`; startup: `atrito_pe`,
`massa_tronco`, `vies_encoder`, `tamanho_caixa`, `atraso`; intervalo: `empurrao`, `forca_caixa`,
`rfi`. Cada fase acrescenta os seus (D: `fatia`, `base`, `juntas`, `filtros`, os startup menos
`tamanho_caixa`, `empurrao`; E: `cena`, `carga`, `atrito_pad`, `atrito_laje`, `vies_imu`, `tamanho_caixa`; H: `pd`, `forca_motor`, `solref`, `com_caixa`,
`forca_caixa`, `rfi`).

**Terminações (nota):** a chave `time_out` usa a função `fim_da_cadeia` (D6), com `time_out=True`.

**Currículo:** um termo só, `v3`.

### 6.6 Regra de tarefa por termo

Cada termo de recompensa e de terminação recebe o parâmetro `tarefas: tuple[str, ...]` e vale zero
(ou falso) nos envs cuja `v3_tarefa` não está na tupla. A máscara mora em uma função só,
`curriculo.mascara(env, tarefas) -> (E,) bool`, e o termo aplica com `torch.where(m, valor, 0)`.
⚠ Nunca multiplicar pela máscara: NaN × 0 = NaN, e um NaN de um env fora da tarefa chega à
recompensa, à entrada do ator e às métricas. Pelo mesmo motivo, nenhuma função pura devolve NaN
como "não usado" (§6.3, `alvos`). A tabela da S §9.2 vira as tuplas do `env_cfg`.
Depois da caixa perdida (`v3_perdeu`), os termos que pagam positivo por passo (`vivo`, `altura_pe`,
`passo`, rastreios) valem zero (S §9.2, §10).

### 6.7 Chaves de log e checkpoint

- Currículo (`Curriculum/v3/<k>`): `frac_ANDAR`, `frac_B`, `frac_C`, `s_<tarefa>_<tipo>`,
  `nivel_<tarefa>`, `degrau_<tarefa>`, `atraso_max`, `mu_min_pad`.
- Twist (`Metrics/twist/<k>`): `razao_marcha`, `eficiencia_min`, `eficiencia_media`, `segmentos`,
  mais os do fabricante.
- Métricas (`Episode_Metrics/<k>`): as das tarefas D8, E7, F6, G4 e H6.
- Checkpoint: `infos["v3_curriculo"] = env.v3_cur.estado()`. Sem estado de episódio (o mesmo motivo
  do `g1_limpo/runner.py:24-27`).

### 6.8 DR (S §12, DR §5; PD-24)

| quando | o que entra | valor |
|---|---|---|
| startup, desde a it 0 | atrito do pé | 0,3–1,0, compartilhado entre as geoms do pé |
| | massa do tronco | −1 a +3 kg no `torso_link` |
| | viés de encoder | ±0,015 rad |
| | tamanho da caixa | meia aresta 0,07–0,13, 8 variantes por mundo |
| reset, desde a it 0 | massa da caixa | 1 a m_k kg pelo nível do PEGAR |
| | atraso da ação, da propriocepção e do mapa | 0–1 passo, por episódio |
| | estimador | magnitude "início" (PD-18) |
| intervalo, desde a it 0 | empurrão | ±0,5 m/s em x e y, a cada 5 s |
| ANDAR no portão (nível 0) | viés por episódio no giro da base e inclinação fixa do IMU (evento `vies_imu`, E1; termos `giro_base` e `gravidade`, E8) | ±0,05 rad/s; ±1° |
| PEGAR no portão do nível 0 | atrito dos pads de palma (evento `atrito_pad`, PD-46) | [0,6; 1,0] |
| | atrito da laje (evento `atrito_laje`, PD-46) | [0,3; 1,0] |
| | CoM da caixa | ±0,2 da meia aresta |
| | atraso | 0–2 passos |
| cada nível seguinte do PEGAR e do CARREGAR | borda de baixo do μ dos pads | desce 0,025 por nível até 0,5 |
| | CoM da caixa | até ±0,3 (xy), ±0,2 (z) |
| | força na caixa (CARREGAR) | impulso 5–20 N, 0,1–0,3 s, a cada 3–8 s |
| | RFI | 0,05 × limite de torque |
| refino, por degrau (linear de 0 a 5) | ganho do PD | 1,0 → 0,9–1,1 |
| | força do motor | 1,0 → 0,9–1,1 (braços 0,8–1,1) |
| | `solref[0]` do chão e da caixa | 0,02 → 0,015–0,03 |
| | atraso | 0–2 → 0–3 |
| | RFI | 0,05 → 0,1 |
| | estimador | "início" → "refino" (§6.2 `Estimador`) |

### 6.9 Regras do refino por degrau (S §11.5)

Valor no degrau `d` = início + (fim − início) · d/5. Nas regras de termo, `d` é o degrau da tarefa
corrente do env; na linha "estimador e DR do refino", `d` é o degrau global, o menor entre as
tarefas (PD-41).

| regra | ANDAR | PEGAR | CARREGAR | BOTAR | início → fim |
|---|---|---|---|---|---|
| `laje_joelho_N` | | ✓ | | ✓ | 50 → 10 N |
| `incl_lim` dos invariantes | | ✓ | ✓ | ✓ | 30° → 15° |
| peso do `aperto` | | ✓ | ✓ | ✓ | 0 → `Recompensa.aperto` |
| peso da `sacudida` | | ✓ | ✓ | ✓ | 0 → `Recompensa.sacudida` |
| `chegada` | | | | ✓ | 1,0 → 0,25 m/s |
| `sigma_pouso` | | | | ✓ | 1,0 → 0,35 m/s |
| `batente` e `hip_yaw` > 1 rad | ✓ | ✓ | ✓ | ✓ | desligada no degrau 0, ligada do 1 em diante |
| estimador e DR do refino (degrau global, PD-41) | ✓ | ✓ | ✓ | ✓ | §6.8 |

Um peso de termo que muda por degrau vira um fator dentro do termo (`fator = d/5`), e não uma
mutação do peso no `reward_manager`: o peso do cfg é o do degrau 5.

---

## 7. Formato das tarefas

Cada tarefa tem: id, modelo, lane, arquivos (só estes), origem (o que copiar, com `arquivo:linha`),
o que fazer, teste (com os números), contagem de linhas, seção da spec e do enunciado, e a mensagem
do commit. A lane de testes da fase escreve os testes ANTES das lanes de código. O orquestrador roda
`python -m unittest discover -s g1_v3/testes -t .` depois de cada lane e commita a tarefa verde.

---

## 8. Fase A — preparação

Atende: S §5 (pacote), §6.3 (`q_de_pé`), §8.1 (tabelas), I §2 e §3 (contrato), PERC §2 e §5.
Enunciado: E §2 "Robô" (MJCF exato), E §3 "Pose de referência".

Ondas: A1 → A6 (testes) → {A2, A3, A4, A5} em paralelo → commit por tarefa.

### A1 — esqueleto do pacote (Haiku)

- Arquivos: `g1_v3/__init__.py`, `g1_v3/testes/__init__.py`, `g1_v3/dados/__init__.py`,
  `g1_v3/parte0/__init__.py`.
- O que fazer: o `__init__.py` do pacote tem só a docstring. Ela diz, em português: o objetivo da
  v3 (S §1), a regra "zero import de código do projeto", a lista dos módulos puros (§6.1) e que o
  registro da task mora em `g1_v3/tarefa.py` (PD-1). Os outros `__init__.py` ficam vazios.
- Teste: `python -c "import g1_v3, sys; assert 'mjlab' not in sys.modules"`.
- Contagem: 0 → ~25 linhas.
- Commit: `feat(v3): esqueleto do pacote g1_v3`.

### A6 — testes da fase A (Haiku, lane de testes)

- Arquivos: `g1_v3/testes/test_knobs.py`, `test_referencias.py`, `test_mid360.py`,
  `test_capsulas.py`.
- Casos (os números são o critério; o teste falha se o valor sair da tolerância):
  - `test_knobs`: `dim_ator() == 1447`; `len(JUNTAS) == 29`; `TAREFAS[0] == "ANDAR"` e
    `TAREFAS[3] == "BOTAR"`; cada junta casa com exatamente UMA regex de `Contrato.pd` e de
    `Recompensa.tau_max`; `q_de_pe()` tem forma (29,), joelho 0,162 ± 0,001, cotovelo 0,600 ±
    0,001, `left_shoulder_roll` 0,200 ± 0,001, `right_shoulder_roll` −0,200 ± 0,001 (I §2.2);
    `Knobs()` é congelado (atribuir levanta `FrozenInstanceError`); `knobs.py` não importa `torch`
    nem `mjlab` (conferir `sys.modules` num subprocesso limpo).
  - `test_referencias`: as chaves da §8 A3 existem; `pelve` na grade (chão, 0,07) = 0,249 ± 0,001 e
    (0,55; 0,13) = 0,705 ± 0,001; `tronco` (0,30; 0,10) = 51,2° ± 0,1°; `recuo` (0,30; 0,10) entre
    0,42 e 0,46; `pelve_de_pe` = 0,789 ± 0,001; `palmas_de_pe_b` simétricas (|y_E + y_D| < 1 mm);
    `caixa_transporte_b` da caixa de 20 cm com x 0,25 ± 0,03 e z +0,11 ± 0,01 (I §2.5); `q_de_pe`
    igual a `ref_v3_de_pe.npz["qpos"][0, 7:36]` em 1e−12.
  - `test_mid360`: `dirs` com forma (200, 205, 3) e norma 1 ± 1e−5; a direção (0, 0) igual à
    conversão da linha 1 do CSV (azimute 268,99°, zênite 37,838°) em 1e−6; a união dos 5 primeiros
    passos cobre ≥ 300° de azimute.
  - `test_capsulas`: K > 20 cápsulas; todo raio em (0,005; 0,25) m; o subconjunto `braco` tem
    cápsulas nos dois lados; nenhum corpo fora da árvore do `pelvis`.

### A2 — `knobs.py` (Sonnet)

- Arquivos: `g1_v3/knobs.py`.
- Origem: o formato de `g1_limpo/knobs.py` (dataclasses por seção, `Knobs`, `ATIVO`, `:1608-1628`).
- O que fazer: todas as seções e todos os campos da §6.2, com os valores da tabela. Cada campo
  tem um comentário de uma linha com a fonte (a coluna "fonte" da §6.2). Os campos que saem da
  Parte 0 ficam `None`. As tabelas por regex (`pd`, `tau_max`) são tuplas de tuplas. O módulo
  importa só `math`, `dataclasses`, `pathlib` e `numpy`.
- Teste: `test_knobs`.
- Contagem: o `g1_limpo/knobs.py` tem 1 628 linhas → meta da v3 ~700.
- S §5, §6.3, §19; I §2, §3. Commit: `feat(v3): knobs com o contrato do robô real`.

### A3 — tabela de referências macro (Sonnet)

- Arquivos: `g1_v3/dados/gera_referencias.py`, `g1_v3/dados/referencias.npz`.
- Origem: as três tabelas da §2.1; a FK com `mujoco.MjModel.from_xml_path` sobre o `g1.xml` do
  mjlab (`mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml`, `nq` = 36). As linhas da tabela têm
  `qpos` de 43: as 36 primeiras são do robô, as 7 últimas da caixa.
- O que fazer: um script `python -m g1_v3.dados.gera_referencias` que grava estas chaves:

  | chave | forma | de onde |
  |---|---|---|
  | `topos`, `meias` | (6,), (3,) | `ref_agachado.npz` linhas com `topo` ≥ 0 |
  | `grade_pelve` | (6,3) | `pelve_z` da tabela |
  | `grade_tronco` | (6,3) rad | `tronco_incl` da tabela, em rad |
  | `grade_recuo` | (6,3) | ‖`qpos[0:2]` − `caixa_xyz[0:2]`‖ |
  | `grade_aceita` | (6,3) bool | `aceita` da tabela (só relatório; decisão de 14/10: as 3 que não fecham entram como estão) |
  | `caixa_transporte_b` | (3,3) | centro da caixa no frame da pelve, `ref_v3_de_pe_theta-30_tamanhos.npz` |
  | `palmas_com_caixa_b` | (3,2,3) | FK dos sites `left_palm`, `right_palm` no frame da pelve, mesma tabela (PD-4) |
  | `q_com_caixa` | (3,29) | `qpos[:, 7:36]`, mesma tabela |
  | `pelve_com_caixa` | (3,) | `qpos[:, 2]` |
  | `q_de_pe` | (29,) | `ref_v3_de_pe.npz["qpos"][0, 7:36]` |
  | `palmas_de_pe_b` | (2,3) | FK, mesma pose |
  | `pes_de_pe_b` | (2,3) | FK dos sites `left_foot`, `right_foot` no frame da pelve, mesma pose (termo `passo`, PD-16) |
  | `cantos_sola` | (4,3) | os 4 cantos da sola no frame do `*_ankle_roll_link`, do retângulo envolvente das geoms `*_foot[1-7]_collision` (cerca do CoM, PD-35) |
  | `pelve_de_pe` | () | `qpos[0, 2]` |
  | `site_pe_de_pe` | () | z do site `left_foot` na pose de pé (a altura do site sobre a sola) |
  | `fonte`, `hash_origem` | str | caminhos e `hash_mjcf` das tabelas de origem |

  A guarda contra mudança de modelo é a tolerância nas grandezas macro, e não o hash (decisão de
  14/10): na pose de pé com caixa de cada tamanho, cada palma da FK fica entre 0 e 5 cm do plano da
  face lateral da caixa (a IK aceitou a mão a até 1 cm, S §8.1), e a pelve da FK = `qpos[2]` em 1 mm.
  Fora disso, o script falha com a medida.
- Quem roda: o agente (PD-3). O `.npz` vai para o git (é pequeno e é dado do contrato).
- Teste: `test_referencias`.
- Contagem: 0 → ~120. S §8.1, §6.3; I §2.2. Commit: `feat(v3): tabela de referências macro da IK`.

### A4 — padrão do MID-360 (Haiku)

- Arquivos: `g1_v3/dados/gera_mid360.py`, `g1_v3/dados/mid360_passos.npz`.
- Origem: `lidar_sim/sensores/lidar.py:17-31` (CSV e URL da Livox) e `:60` (fórmula da direção:
  `(sin ze·cos az, sin ze·sin az, cos ze)`, ângulos em rad).
- O que fazer: lê o CSV de `~/orca/workspaces/g1_training/lidar_sim/sensores/mid360.csv`; se não
  existir, baixa da URL. Divide as 800 000 linhas em 200 trechos de 4 000 (20 ms a 200 mil pontos
  por segundo). Em cada trecho, toma as linhas `⌊j·4000/205⌋`, j = 0..204 (PD-5). Grava `dirs`
  (200, 205, 3) float32. O `.npz` (~0,5 MB) vai para o git.
- Teste: `test_mid360`. Contagem: 0 → ~40. PERC §5; S §4 (viab. D6). Commit:
  `feat(v3): padrão do MID-360 em trechos de 20 ms`.

### A5 — cápsulas do corpo (Sonnet)

- Arquivos: `g1_v3/dados/gera_capsulas.py`, `g1_v3/dados/capsulas.npz`.
- Origem: as geoms de colisão do `g1.xml` (grupo 3, `contype` ≠ 0); PERC §7.
- O que fazer: cada geom vira uma cápsula no frame do corpo dela: cápsula e cilindro pelo eixo z
  local e meia altura; esfera com segmento nulo; caixa e malha pela cápsula envolvente no maior eixo,
  com raio = maior meia extensão dos outros dois. Grava `corpo` (K,) str, `p0`, `p1` (K,3), `raio`
  (K,), `braco` (K,) bool (ombros, cotovelos, punhos, mãos). Quem roda: o agente (PD-3).
- Teste: `test_capsulas`. Contagem: 0 → ~70. PERC §7; S §13 (oclusão). Commit:
  `feat(v3): cápsulas de colisão para o filtro do corpo`.

---

## 9. Fase B — Parte 0 (funções puras e checagens sem GPU)

Atende: S §16 Parte 0, itens 1, 2, 3, 4, 6, 8 e 15 (os itens 5, 7, 9, 11 a 14 e 16 estão feitos; o
10 saiu em 14/10). Enunciado: E §3 "Sem estátua", "Progressão na cadeia", "Caixa íntegra e
nivelada"; E §4.1 "Proibido" (passo miúdo).

Ondas: B0 (testes) → {B1, B2, B3} em paralelo → B4 → B5 (itens 1, 2, 4, 6, 8 em paralelo; depois o 3;
depois o 15) → B6.

### B0 — testes das funções puras (Haiku, lane de testes)

- Arquivos: `g1_v3/testes/test_potencial.py`, `test_alvos.py`, `test_balanceador.py`.
- Casos:
  - `potencial`: `nucleo(σ, σ) = e⁻¹`; `rampa(x_lim) = 1`, `rampa(x_lim + Δ) = 0`,
    `rampa(x_lim + Δ/2) = 0,5`; `progresso(1, 1, 0,99) = −0,01`; `custo_aperto(49, 49, 49)` =
    0,632 ± 0,001; `custo_aperto(117,5, 117,5, 49)` = 0,909 ± 0,002 e derivada 0,0019 ± 0,0001 por
    N (S §8.2); `fator_pouso(0,9, 0,02, 1,0, 0,05)` = 0,445 ± 0,001 (S §9.1: 0,44), e 1 com o fundo a
    0,10 m; `excesso_de_junta` = 0 a 90% do meio curso, 0,5 a 97,5%, 1 a 100% (S §9.2);
    `rastreio_normalizado`: c 0,5 e erro 0 → 1; robô parado (erro = c) → 0; c 0,05 e metade da
    velocidade → 0,75 ± 0,01 (S §19, conflito 5); c 0,01 e v 0 → 0; c 0,01, v 0,05 e σ₀ 0,10 →
    −0,221 ± 0,001 (PD-49); c 0,01, v 0,5 e σ₀ 0,10 → −1,000 ± 0,001; piso −1 respeitado; `L_alvo` 0,25 → 0,175; 0,5 → 0,20; 1,0 → 0,40; 1,5 → 0,60 (S §19, conflito 6).
  - `alvos`: `Referencias` devolve a grade nos nós e a média no meio de dois nós; clamp abaixo de 0 e
    acima de 0,55; `frame_da_caixa` com a caixa girada 30° e o robô a 0° dá o eixo x a 30°; com a
    caixa a 50°, dá −40°; `inclinacao_da_caixa` da caixa girada 90° em x = 0 (simetria) e girada 20°
    = 20°; `palmas_alvo_caixa` P0 com a = 0,10 → (−0,15; ±0,135; 0) e B3 → (0; ±0,185; 0); nos
    alvos sem uso, `palmas_alvo_caixa` e `caixa_alvo_w` devolvem 0, e nenhum valor é NaN;
    `caixa_alvo_w` P4 com a base na origem, rumo 0, chão 0 → z 0,90 e xy de `caixa_transporte_b`;
    `avanca` no P4: 25 passos de 0,02 s com `ok` avançam, um `ok` falso no passo 10 zera o relógio; no
    P2 avança no mesmo passo; `detecta_fim` com |q̇| médio 0,1 rad/s avisa no passo 25, com 0,3 nunca.
  - `balanceador`: `fracoes([0,0,0], 0,2, 0,02)` = 1/3 cada; com um déficit 0,75 e dois 0, toda fração
    ≥ 0,2 e a soma 1; no começo só o ANDAR abre (`fracoes_fatias` = [1, 0, 0]); a B abre só com
    `reto` E `parado` ≥ 0,75; a célula seguinte abre só com todas as abertas da tarefa no portão; o
    nível sobe, a EMA recomeça e a subida seguinte espera 50 its; a C abre com o PEGAR nível 0 ≥ 0,75;
    o refino começa só com o último nível aberto e 100 its seguidas no portão, e nunca desce;
    `estado()` → `carrega()` devolve o mesmo estado, inclusive o contador de iterações; um `Curriculo`
    carregado 20 its depois de uma subida sobe o nível seguinte só depois de mais 30 `fecha_iteracao`
    (a carência continua, a retomada não a recomeça); `sorteia_fatias` com frações (0,5; 0,3; 0,2) em
    100 000 amostras fica a 1% de cada fração.

### B1 — `potencial.py` (Sonnet)

- Origem: a forma do núcleo de `g1_limpo/recompensas.py:526-576` (`exp(−(d/σ)²)`); o resto é novo
  (S §9.1, §8.2, §19).
- O que fazer: as funções da §6.3, cada uma com docstring curta que cita a seção da spec.
- Teste: `test_potencial`. Contagem: 0 → ~150. Commit: `feat(v3): potencial, rampas e núcleos puros`.

### B2 — `alvos.py` (Sonnet)

- Origem: a regra do par de faces de `g1_limpo/recompensas.py:680-718` (`_alinha`) só como
  referência; o resto é novo (S §8; I §4.2).
- O que fazer: as funções da §6.3. `Referencias` interpola em bilinear na grade (h, a) com clamp nas
  bordas; em `a` só, interpola as três linhas da pose de pé com caixa. Nenhuma função lê `qpos` de
  pega (decisão de 14/10).
- Teste: `test_alvos`. Contagem: 0 → ~300. Commit: `feat(v3): alvos macro, avanço e detector de fim`.

### B3 — `balanceador.py` (Sonnet)

- Origem: a fórmula da S §11.1; o formato de estado serializável de `g1_limpo/runner.py:47-61`.
- O que fazer: `Curriculo` com as células da §6.3 e as regras de abertura da tabela da §6.3. A lista
  de células, em ordem de abertura: ANDAR `reto`, `parado`, `baixo`, `giro_andando`, `giro_lugar`,
  `terreno`; CARREGAR `reto`, `giro_lugar`, `arco`, `terreno`; PEGAR `pegar`; BOTAR `botar`. Os
  níveis: ANDAR e CARREGAR 0..5 (velocidade) mais 0..12 de terreno dentro da célula `terreno`
  (linha do gerador, PD-20; o CARREGAR para na `linha_max_carregar` da coluna); PEGAR e BOTAR 0..4.
- Teste: `test_balanceador`. Contagem: 0 → ~250. Commit: `feat(v3): balanceadores, níveis e refino puros`.

### B4 — utilitários da Parte 0 (Sonnet)

- Arquivos: `g1_v3/parte0/comum.py`.
- Origem: a leitura de CSV e a FK de `g1_limpo/ik/cercas_nos_plays.py` (copiar a medida das cercas).
- O que fazer: `le_csv(caminho) -> dict[str, np.ndarray]` (módulo `csv` da stdlib); `fk(q, raiz) ->
  dict` com pés, palmas, pelve, tronco (MuJoCo clássico, `mj_kinematics`, PD-3); `custos_por_s(dados,
  k)` com o custo por segundo de cada regulador da S §9.2 nos pesos da v3 (taxa de ação de `a_*`,
  velocidade e aceleração de junta por diferença de `q_*`, energia com `τ = kp·(alvo_pd − q) − kd·q̇`
  e os ganhos do mjlab, que eram os do play); `cercas_por_passo(dados, ref)` com as sete cercas da S
  §9.3 por passo; `trajetoria_pegar()` e `trajetoria_botar()`, os caminhos roteirizados do item 1,
  que os itens 2 e 3 também usam.
- Contagem: 0 → ~200. Commit: `feat(v3): utilitários das checagens da Parte 0`.

### B5 — os scripts dos itens (Sonnet; um agente por item, em paralelo)

Unidade (29/10): toda conta da Parte 0 está no retorno do mjlab (`scale_by_dt` = True): um termo de valor x e
peso w soma w·x·dt por passo. Custo ou ganho por segundo = w·média(x) (`custos_por_s`). Um bônus pago num passo
(`B_alvo` dentro do valor do `progresso`) soma `B_alvo`·dt; para cobrir um custo acumulado C (custo por s × duração
do trecho), `B_alvo` = 1,5·C/dt. O termo `progresso` soma w_p·(γΦ′ − Φ)·dt por passo; o ganho de Φ por segundo da
PD-32 usa a mesma unidade.

Cada script roda com `python -m g1_v3.parte0.<nome>`, imprime uma tabela e grava
`g1_v3/parte0/saida/<nome>.txt`. O agente roda o script e entrega a saída. Os itens 1, 2, 4, 6 e 8
correm em paralelo; o 3 vem depois deles, e o 15 depois do 3 (ele usa a `perda_caixa` do 3).

CSV de entrada (raiz do repo; o trecho sai da coluna `fase`; conferido em 29/10). O script imprime
o número por arquivo e o número de todos juntos:

| trecho | arquivos (`fase`) |
|---|---|
| PEGAR | `23250_k2`, `23250_topo004`, `24800_botar_t015`, `24800_mu06_m3`, `24800_mu10_m1`, `24800_mu10_m5` (`pegar`) |
| CARREGAR | `23250_topo004` (0,69 m/s), `24800_mu06_m3`, `24800_mu10_m1`, `24800_mu10_m5` (0,30 m/s) (`carregar`, 500 passos); no item 8 o critério do `L_alvo` não se aplica a eles: são o passo miúdo da E §4.1 (D2 da Parte 0, dono 29/10) |
| BOTAR | `23250_k2`, `24800_botar_t015` (`botar`) |
| ANDAR andando | `5000_botar`, `5350_botar` (`andar`, 300 passos, 0,65 m/s) |
| ANDAR parado | `5000_andar_parado`, `5350_andar_parado`, `5800_andar_parado` (`parado`) |
| fora | `24800_topo004` (o play acaba no PEGAR, 132 de 350 passos); `5800_botar` (o `andar` acaba em 66 de 300 passos); `excluir` |

| script | item da S §16 | entrada | o que mede | critério |
|---|---|---|---|---|
| `item1_phi.py` | 1 | trajetórias sintéticas sobre `alvos` e `potencial`: palmas da pose de pé até P0 (1,0 s), deslizam até P1 (0,5 s), contato em P2, caixa sobe 5 cm (0,5 s, P3), vai ao transporte (1,0 s) e fica 0,5 s (P4); laje 0,30 m, a 0,10. BOTAR: B1 a B4 na laje de 0,15 m | Φ e a recompensa por passo | Φ não desce dentro de cada alvo no caminho certo; pairar a 1 cm do alvo por 2 s soma ≤ 0; tombar a caixa 40° no P3 leva Φ a ≤ a rampa (0 a 45°); soltar uma mão divide Φ por 2 |
| `item2_derivadas.py` | 2 | os mesmos alvos | dΦ/dd no estado inicial de cada alvo (d = σ) e d(rampa)/dx em x_lim + Δ/2 | toda derivada ≠ 0; imprime a tabela dos Δ da PD-23 para o dono |
| `item3_escada.py` | 3 | `custos_por_s` nos CSV da tabela acima (PEGAR, CARREGAR, BOTAR, ANDAR andando e parado); o tempo de cada trecho medido nos CSV (PEGAR 0,5–2,2 s); no BOTAR, distância / 0,25 m/s (S §7) | `B_alvo` por alvo, peso do `progresso`, `perda_caixa`, `devolve_b_alvo`, pesos das cercas | `B_alvo_i` = 1,5 × custo dos reguladores no trecho até o alvo i (PD-31); peso do progresso = 1,5 × max(custo dos reguladores por s ÷ ganho de Φ por s no caminho roteirizado) (PD-32); escada da S §10: queda > perda > volta à pose de pé, queda > mergulho, as duas desigualdades de "desistir"; `perda_caixa` = 1,2 × o lado direito da desigualdade (2), e `devolve_b_alvo` = True só se ela não fechar com custo fixo (PD-33); cercas pela PD-30 |
| `item4_balanceador.py` | 4 | `Curriculo` com curvas sintéticas de sucesso por iteração (logísticas), uma presa em 0,3 | frações por iteração | nenhuma fatia trava a outra (toda fatia aberta recebe ≥ piso); com todas no portão, as frações vão a 1/3 ± 0,02 |
| `item6_comando_baixo.py` | 6 | `custos_por_s` nos CSV do ANDAR andando contra os do ANDAR parado (tabela acima) | para c = 0,05; 0,10; 0,25; 0,5; 1,0 (linear e giro): ganho por s de executar (rastreio normalizado = 1) contra a estátua (0) e contra a marcha no lugar (rastreio 0, passo 0), e o custo por s de começar e manter a marcha | executar vence a estátua e a marcha no lugar em todo c |
| `item8_passo.py` | 8 | FK dos pés nos CSV do ANDAR andando e do CARREGAR (tabela acima); toques pelo pé a < 2 cm do chão | `L_alvo` por nível de velocidade contra o passo medido; fração de voo; passo de lado com a PD-16 | `L_alvo` ≤ 1,2 × o maior passo medido na mesma velocidade e ≤ 0,60 m; o termo satura em `L_alvo` (passo maior não paga mais); o passo de lado parado (pés na largura da base) paga 0 |
| `item15_aperto.py` | 15 | S §8.2 (110–125 N por face no `model_24800`) | derivada de `custo_aperto` no ponto de operação; ganho acumulado de afrouxar de 117,5 N para 49 N em 20 s; `sacudida` pela mesma regra (PD-34) | derivada em 117,5 N ≥ 20% da derivada em `F_ref` (fica 25%, S §8.2); `w` ≤ 0,5 × `perda_caixa` SEM o aperto evitado ÷ (ganho de afrouxar numa rampa de 20 s) (D6 e D8 da Parte 0, dono 29/10); depois o item 3 põe na perda o aperto evitado com esse `w` |

### B6 — relatório e portão da Parte 0 (Opus)

- Arquivos: `docs/relatorios/<data>-v3-parte0.md`; depois do OK do dono, `g1_v3/knobs.py` (os
  campos `None`).
- O que fazer: junta as saídas da B5 numa tabela por item, com o número proposto, o critério e o
  resultado. Lista o que falhou. O dono aceita ou troca cada número. O orquestrador escreve os
  números aceitos no `knobs.py` e acrescenta uma linha na inbox do vault.
- Portão: o dono aceita os números. Sem o portão, nenhuma tarefa da fase D começa.
- Commit: `docs(v3): resultados da Parte 0` e `feat(v3): pesos da Parte 0 no knobs`.

---

## 10. Fase C — percepção, fase 1 (em paralelo com a fase B)

Atende: S §4 (mapa de elevação desde o começo), §19.1; I §2.4; PERC §0 e §9 fase 1. Enunciado:
E §2 "Terreno".

Ondas: C0 (testes) → C1.

### C0 — testes do filtro (Haiku, lane de testes)

- Arquivos: `g1_v3/testes/test_percepcao.py`.
- Casos (portão da fase 1, PERC §0):
  - um degrau de 0,15 m nos pontos sintéticos vira um salto de 0,15 ± 0,02 m entre as células
    antes e depois da borda;
  - chão plano com ruído de 2 cm: depois de 3 leituras, |h| < 1 cm e confiança = σ_ref²/(σ_ref² +
    σ_fixo²) = 0,862 ± 0,001 (o piso, PD-42);
  - transportar 0,10 m para a frente desloca o mapa em uma célula;
  - transporte vertical (PD-7): uma célula vista a +0,15 e uma a 0; `prediz` sem leitura, com o
    `z_ref` 0,15 acima do anterior (o robô subiu o degrau) → +0,15 vira 0 e 0 vira −0,15, e a
    confiança não muda; o primeiro `prediz` depois do `reinicia` não desloca nada;
  - girar 90° no lugar leva a célula (x = +0,5; y = 0) para (x = 0; y = −0,5) em ± 1 célula;
  - célula nunca vista: altura 0 e confiança 0;
  - a base inclinada 10° em pitch não muda a altura do chão (± 1 cm);
  - o robô agachado 20 cm (sensor 0,20 m mais baixo, pés no chão) vê o chão em 0 (PERC §0);
  - depois de 1 m de transporte sem leitura e com Δt = 0, a confiança de uma célula vista uma vez
    (um ponto) cai para `σ_ref² / (σ_ref² + max(R, σ_fixo²) + q_desloc·1)` ± 1e−6, com R da PD-42 (PD-6, PD-7);
  - 5 pontos na mesma célula com ruído de 2 cm dão a média, e não o máximo: |h| < 1 cm (PD-42);
  - ordem do `recorte`: o índice 0 é (x = +1,0; y = +1,0), o 440 é (−1,0; −1,0), as confianças
    começam no 441 (I §2.4);
  - `filtra_corpo`: ponto a menos de raio + margem de uma cápsula sai; ponto a 0,84 m do sensor sai
    e a 0,86 m fica; o chão a 1,2 m abaixo do sensor fica (PD-10);
  - `PadraoMid360.direcoes(200) == direcoes(0)`;
  - invariância à densidade (PD-42): chão plano com degrau de 0,15 m, a mesma trajetória a 1 m/s,
    pontos sintéticos a 205 e a 4 000 por passo com o mesmo modelo de ruído (o mesmo giro de
    inclinação por leitura nos dois); toda célula lida 2 vezes está vista nos dois; fora a coluna da
    borda do degrau, média de |Δh| ≤ 1 cm e p95 ≤ 3 cm (portão da PERC §0), e |Δconfiança| ≤ 0,03 em
    90% das células (dono, 29/10: o máximo de ±1 cm é inatingível com o ruído; a borda mistura os dois
    lados);
  - os transportes de 0,10 m e de 90° rodam com os passos reais a 50 Hz (5 × 2 cm; 75 × π/150);
  - o giro fixo do IMU (`incl_imu` de 1° em pitch) inclina o chão de 1,7 cm a 1 m em `pontos_no_frame_do_mapa`.

### C1 — `percepcao_lidar.py` (Sonnet)

- Origem: nenhuma (o `lidar_sim/percepcao/mapa.py` é voxel do Open3D e não serve, viabilidade D9).
- O que fazer: as funções e a classe da §6.3, com:
  - mapa interno 51 × 51 (PD-8); altura `h` (E,51,51), variância `P` (E,51,51), máscara `vista`;
  - `prediz`: o mapa não gira e anda só em células inteiras (dono, 29/10): a base guarda o resto `o`
    (menos de meia célula) e o rumo `psi` nos eixos do mapa; a cada passo, `a = o + R(psi)·Δxy`, o mapa
    desloca `round(a/celula)` células, `o = a − deslocamento` e `psi += Δψ`; o giro acontece no `recorte`,
    e o `atualiza` leva os pontos aos eixos do mapa pelo mesmo `o` e `psi`. ⚠ A "célula mais perto" a cada
    passo (PERC §6) congelava o mapa: a 50 Hz o passo de 2–3 cm é menor que meia célula. Fora do mapa
    vira "nunca vista";
    `P += q_desloc·|Δxy| + q_giro·|Δψ| + q_tempo·Δt` (PD-7); depois, nas células vistas,
    `h −= z_ref − z_ref_anterior` e o filtro guarda o `z_ref` (transporte vertical, PD-7);
  - `atualiza`: índice de célula de cada ponto válido; os inválidos vão a um índice de descarte
    (PERC §4 passo 4); por célula, a média dos z da leitura (`scatter_reduce` "mean") e a contagem
    n e a distância média d̄ (a média de `dist` dos pontos da célula); Kalman com `R = (σ_incl·d̄)² + σ_m²/n`, `σ_m² = ruido_alcance² +
    (d̄ · σ_ângulo)²`, `σ_incl` = `ruido_incl_graus` em rad; célula nunca vista recebe `h = z`, `P = R`;
    depois da atualização, `P = max(P, σ_fixo²)` (PD-42);
  - `recorte`: as células 15..35 em x e em y, na ordem da I §2.4; altura cortada em ±1 m; nunca
    vista → altura 0, confiança 0;
  - zero import do mjlab e do `mujoco`; o sim do real e o robô carregam o arquivo por
    `importlib.util.spec_from_file_location` (PERC §2).
- Desempenho: um script `g1_v3/parte0/custo_filtro.py` mede `prediz` + `atualiza` + `recorte` com
  4 096 envs e 205 pontos em CPU e imprime o tempo; referência: 26 ms com P 256 (viabilidade D7).
- Teste: `test_percepcao`. Contagem: 0 → ~360. Commit: `feat(v3): filtro de elevação puro (fase 1)`.

---

## 11. Fase D — Parte 1: pacote, locomoção e lidar no treino

Atende: S §16 Parte 1, §5, §6.3, §9.2 (termos do ANDAR), §10 (terminações da Unitree), §11.1 a
§11.3 (balanceadores do ANDAR), §12 (DR desde o começo), §19 e §19.1; PERC §9 fase 5; I §2 e §3.
Enunciado: E §1 "Ordem de aprendizado" (100% ANDAR), E §2 "Terreno", E §3 "Locomoção", "Pés",
"Sem estátua", "De pé, como um humano", "Empurrão", E §4.1.

Ondas:

| onda | tarefas, em paralelo | depende de |
|---|---|---|
| D-0 | D0 (testes) | fases A, B, C |
| D-1 | D1 `cena.py` | D-0 |
| D-2 | D2 `observacoes.py`, D2b `estimador.Odometria`, D3 `comando.py`, D4 `curriculo.py`, D5 `recompensas.py`, D6 `terminacoes.py`, D7 `eventos.py`, D8 `metricas.py`, D9 `algoritmo.py` + `runner.py` | D-1 |
| D-3 | D10 `env_cfg.py`, `tarefa.py`, `train.py`, `play.py` | D-2 |
| D-4 | D11 `exporta_cena.py` + `pilota_v3.py` + `registra_juntas_v3.py`; D12 smoke da Parte 1; D13 entrada do container | D-3 |
| D-5 | D14 revisão (Opus) | D-4 |

### D0 — testes da Parte 1 (Haiku, lane de testes)

- Arquivos: `g1_v3/testes/test_cfg.py`, `test_recompensas.py`, `test_terminacoes.py`,
  `test_observacoes.py`, `test_comando.py`, `test_curriculo.py`, `test_eventos.py`,
  `test_algoritmo.py`, `test_odometria.py`, `test_entrada.py`.
- Regra: os testes chamam os núcleos puros dos módulos com tensores sintéticos. Só o `test_cfg`
  chama `make_env_cfg()`, e nenhum teste constrói `ManagerBasedRlEnv`.
- Casos: os listados em cada tarefa abaixo, na linha "Teste".

### D1 — `cena.py` (Sonnet)

- Origem: `g1_limpo/cena.py:42-160` (constantes), `:164-234` (caixa e laje), `:238-279` (pads),
  `:282-305` (`robot_cfg`), `:309-351` (`regroup`, `geometria_de_repouso`, `entidades`), `:354-455`
  (`sensores`); os atuadores de `mjlab/asset_zoo/robots/unitree_g1/g1_constants.py:91-178`.
- Mudanças sobre a cópia:
  1. `robot_cfg()`: `init_state.joint_pos` = `dict(zip(JUNTAS, q_de_pe()))`; `init_state.pos` =
     (0, 0, `pelve_de_pe` + 0,01). A articulação é própria: um `BuiltinPositionActuatorCfg` por
     linha de `Contrato.pd`, com `stiffness` = kp e `damping` = kd, e `armature` e `effort_limit`
     do motor da junta (os do mjlab, `g1_constants.py:91-110`; tornozelo e cintura roll/pitch com
     o dobro, como o `G1_ACTUATOR_ANKLE` e o `G1_ACTUATOR_WAIST`). Atraso do comando em todos:
     `delay_min_lag` 0, `delay_max_lag` 12 (capacidade de 3 passos da política), `delay_update_period`
     10⁹, `delay_per_env_phase` False (PD-25, PD-28).
  2. Sites no `torso_link`: `mid360` (`Percepcao.mid360_pos`, `mid360_quat`) e `d435` (pos
     `Estimador.d435_pos`, giro de `d435_pitch` em y; o eixo x do site é o eixo óptico).
  3. `_add_pad`: atrito de `Cena.pad_atrito` no pad de palma e de `Cena.robo_atrito` no pad de
     dorso, `group` = 3 (o lidar vê o pad). Atrito por material (PD-46): a caixa nasce com
     `priority` = `Cena.caixa_priority` (0) e μ `Cena.caixa_atrito`; a `CollisionCfg` do robô ganha
     `friction` = {pés: (0,6,), `.*_collision`: `Cena.robo_atrito`}, com os pés primeiro, como o
     `condim`; a laje fica com `priority` 3. Teste (`test_cena`, sem física): no `MjModel` compilado,
     `geom_priority` da caixa = 0, da laje = 3, dos pés = 1; `geom_friction[:, 0]` = 0,3 na caixa,
     nas cápsulas do pulso, no `torso_collision` e nos pads de dorso, e 1,0 nos pads de palma.
  4. Caixa e laje no grupo `Cena.grupo_mobilia` = 1 (PD-9).
  5. `PadraoMid360Cfg` (dataclass, com `caminho`): `generate_rays(mj_model, device)` devolve
     `(zeros(205,3), dirs[0])` (duck typing, `raycast_sensor.py:517-521`).
  6. `terreno_cfg(k)`: `TerrainEntityCfg(terrain_type="generator")` com
     `TerrainGeneratorCfg(size=(8,8), num_rows=13, curriculum=True, border_width=20)` (PD-20) e as
     colunas, nesta ordem:

     | coluna | preset do mjlab (`terrains/config.py`) | parâmetros | plataforma 2 m | `linha_max_carregar` |
     |---|---|---|---|---|
     | 0 `plano` | `flat` | — | — | — (fatias de manipulação no plano) |
     | 1 `escada` | `pyramid_stairs` | `step_height_range` (0; 0,20), `platform_width` 2,0 | sim | 9 |
     | 2 `escada_inv` | `pyramid_stairs_inv` | idem | sim | 9 |
     | 3 `rampa` | `hf_pyramid_slope` | `slope_range` (0; 0,36), `platform_width` 2,0 | sim | 9 |
     | 4 `rampa_inv` | `hf_pyramid_slope_inv` | idem | sim | 9 |
     | 5 `escada_aleatoria` | `random_stairs` (`BoxRandomStairsTerrainCfg`) | `step_height_range` (0; 0,20), `platform_width` 2,0 | sim | 9 |
     | 6 `irregular` | `random_rough` | o preset, com `scale_with_difficulty=True` (o preset vem com False, `heightfield_terrains.py:404`: sem a troca, toda linha tem a mesma rugosidade de 2–10 cm) | não | — (só ANDAR) |
     | 7 `obstaculos` | `discrete_obstacles` | `obstacle_height_range` (0; 0,15), `platform_width` 2,0 | sim | 8 |
     | 8 `caixas` | `random_spread_boxes` | `box_height_range` (0; 0,15), `platform_width` 2,0 | sim | 7 |

     `COLUNAS` (tupla de dicts com nome, `plataforma`, `linha_max_carregar`) sai de `Terreno.colunas` e
     `Terreno.linha_max_carregar` (29/10); o `eventos.py` lê o `knobs`, e não a `cena`.
  7. `sensores(k)`: os 11 do `g1_limpo` e mais:
     - `altura_pes`: `TerrainHeightSensorCfg`, frames nos corpos `left_ankle_roll_link` e
       `right_ankle_roll_link` (a altura do termo da Unitree é a da origem do link, L7),
       `RingPatternCfg.single_ring(radius=0.03, num_samples=6)`, `include_geom_groups=(0,)`;
     - `altura_pelve`: frame no `pelvis`, `single_ring(0.05, 4)`, grupos (0,);
     - `altura_joelhos`: frames nos `left_knee_link` e `right_knee_link`, `single_ring(0.03, 4)`;
     - `altura_caixa`: frame no corpo `box` (entidade `box`), `single_ring(0.05, 4)`, grupos (0,);
     - `nao_pes_chao`: `ContactSensorCfg`, primário os corpos do robô menos os `*_ankle_roll_link`,
       secundário o terreno, `fields=("found","force")`, `reduce="netforce"`;
     - `caixa_faces`: primário `box_geom`, secundário a subárvore do `pelvis`,
       `fields=("found","force","pos","normal")`, `reduce="none"`, `num_slots=12`,
       `global_frame=True` (régua do aperto por face, S §8.2);
     - `lidar` (só com `Percepcao.ligada`): `RayCastSensorCfg`, frame no site `mid360`, padrão
       `PadraoMid360Cfg`, `ray_alignment="base"`, `max_distance` 6,0, `include_geom_groups=(0,1,3)`,
       `exclude_parent_body=True` (o sensor fica dentro da esfera da cabeça, `lidar_sim/sensores/
       lidar.py:38-39`).
- Teste (`test_cfg`, parte da cena): os nomes dos sensores; os grupos das geoms de caixa e laje = 1;
  as colunas do terreno na ordem da tabela; os kp e kd de cada junta = `Contrato.pd`; na
  `linha_max_carregar` de cada coluna, a fórmula do gerador (t = linha/(R−1)) dá o teto do CARREGAR
  da S §11.2: degrau 0,15 m ± 0,001, `slope` 0,27 ± 0,001, obstáculo e caixa 0,10 ± 0,005;
  `random_rough` com `scale_with_difficulty` True.
- Contagem: 482 → ~450. Commit: `feat(v3): cena com PD da Unitree, lidar e terreno`.

### D2 — `observacoes.py` (Sonnet)

- Origem: `g1_limpo/observacoes.py:59-83` (one-hot) e o padrão de termo do mjlab
  (`velocity_env_cfg.py:77-109`).
- O que fazer:
  - `grupos(k: Knobs) -> dict[str, ObservationGroupCfg]` com os termos da §6.5 (29/10: recebe o
    agregador; lê `k.contrato.blocos_ator`, `k.dr.ruido_obs`, a capacidade `k.dr.atraso_refino[1]`
    e `k.percepcao.ligada`; sem a percepção, o `mapa` sai dos dois grupos). O ruído do lidar no
    `passo_de_percepcao` usa o `v3_gerador`. `giro_base`,
    `gravidade`, `comando`, `juntas`, `vel_juntas`, `acao` e `vel_lin_base` usam as funções do mjlab
    (`mdp.builtin_sensor` "robot/imu_ang_vel" e "robot/imu_lin_vel", `mdp.projected_gravity`,
    `mdp.generated_commands` com "twist", `mdp.joint_pos_rel` com `biased=True`, `mdp.joint_vel_rel`,
    `mdp.last_action`). Escala, ruído, histórico e atraso como a §6.5. Os cinco termos com atraso
    nascem com `delay_min_lag` 0, `delay_max_lag` 3, `delay_update_period` 10⁹ e
    `delay_per_env_phase` False; o evento `atraso` da D7 baixa o teto para 1 no startup (PD-28).
    ⚠ O mjlab cria o `DelayBuffer` com a capacidade do cfg (`observation_manager.py:465-475`,
    `delay_buffer.py:151`): com `delay_max_lag` 1 no cfg, a E6 e a H4 não conseguem subir para
    0–2 e 0–3. Custo: ~60 MB na GPU com 4 096 envs (946 canais × 4 lugares).
  - `um_de_seis(env)`: one-hot de `v3_tarefa` (6 canais).
  - `caixa_zero(env)`: zeros (E,11). A fase E troca por `caixa`.
  - `atualiza_percepcao(env)`: roda uma vez por passo da política para o env inteiro. Com
    `v3_percepcao_passo` = `common_step_counter`, só volta. Senão: lê
    `env.scene.sensors["lidar"].data.distances` (E,205); calcula os pontos com
    `pontos_no_frame_do_mapa`, com o z do pé mais baixo = min(z dos sites `left_foot`, `right_foot`)
    − `site_pe_de_pe`; filtra o corpo com `pontos_w` e as cápsulas no mundo (pose dos corpos da
    `capsulas.npz`); `prediz` com `v3_odo.delta_b()` (dxy, dψ) e `z_ref` = `v3_odo.pos_w[:, 2]` +
    (z do pé mais baixo − z da base), o pé mais baixo no frame da odometria (PD-7); `atualiza` com
    as distâncias; copia `v3_padrao.direcoes(passo + 1)` em `sensor._local_directions` (`copy_`, PERC
    §5); grava `v3_mapa_obs = filtro.recorte()` e `v3_percepcao_passo`. ⚠ A guarda mora no env e
    não no termo: o mjlab cria uma instância por cfg de termo (`manager_base.py:146-147`), e o ator e
    o crítico têm o `mapa` cada um; com a guarda na instância, o filtro rodaria duas vezes por passo
    e transportaria o mapa o dobro.
  - termo `mapa(env)`: chama `atualiza_percepcao(env)` e devolve `v3_mapa_obs`; o ator e o crítico
    usam o mesmo termo. Com `Percepcao.ligada` = False, o termo não entra.
  - termos do crítico da §6.5, todos lendo os buffers `v3_*` e valendo zero onde não se aplicam
    (sem caixa, alvo −1). `tarefa_critico` é o último.
- Teste (`test_observacoes`): `um_de_seis` com `v3_tarefa` = [0, 3] dá as linhas certas; o núcleo
  do `mapa` com pontos sintéticos de chão plano devolve 882 canais com as alturas ~0; com um env
  falso (`SimpleNamespace`), duas chamadas de `atualiza_percepcao` no mesmo `common_step_counter`
  transportam o mapa uma vez só; `referencia` com alvo −1 é zero; a lista de termos do ator tem os
  nomes e a ordem da §6.5; os cinco termos com atraso têm `delay_max_lag` 3 no cfg.
- Contagem: 141 → ~300. Commit: `feat(v3): observações do ator e do crítico (§19.1)`.

### D2b — `Odometria` em `estimador.py` (Sonnet)

- O que fazer: a classe `Odometria` da §6.3. Modelo: a cada passo, o incremento verdadeiro vai ao
  frame da base do passo anterior (dxy_b, dψ, dz); dxy_b e dz recebem um ruído normal com variância
  `deriva_odo²·|Δp| + deriva_pos_giro²·|Δψ|` por eixo, e dψ `deriva_rumo²·|Δp| + deriva_rumo_giro²·|Δψ|`
  (PD-18; Δp e Δψ = o deslocamento e o giro verdadeiros do passo); a pose estimada integra
  `pos += (R(ψ_est)·dxy_b, dz)`, `ψ_est += dψ`, e `delta_b()` devolve os três do último passo.
  `deriva_odo` em m/√m (§13: 1,5 cm em 4 m → 0,0075 m/√m no refino), os termos de giro por √rad.
  O erro de rumo passa a entrar na posição absoluta, como no real; o alvo da caixa não sente,
  porque o robô fica parado antes de abrir o PEGAR e o BOTAR.
- Teste (`test_odometria`): deriva 0 → a odometria = a verdade; deriva 0,0075 e os termos de rumo
  em 0, 10 000 envs andando 4 m em passos de 1 cm → desvio padrão do erro final 1,5 ± 0,1 cm.
  Mais: só giro no lugar, magnitude do refino, uma volta → desvio do rumo 2° ± 0,1 e da posição
  2 cm ± 0,1; só translação → os termos de giro não somam nada; com o rumo estimado 10° errado e
  deriva 0, `delta_b()` = o incremento verdadeiro no frame da base (o erro absoluto de rumo não
  entra no Δ do mapa); depois do `reinicia`, `delta_b()` = 0.
- Contagem: 0 → ~75. Commit: `feat(v3): odometria simulada`.

### D3 — `comando.py` (Sonnet)

- Origem: `g1_limpo/comando.py:2276-2526` (`TwistComRazaoDeMarcha` e cfg), `:187-220`
  (`forca_de_apoio`).
- O que fazer:
  - `TwistV3` (cópia do `TwistComRazaoDeMarcha`), com:
    1. `_resample_command(ids)` SEM `super()`. Para os envs com `v3_tarefa` em ANDAR ou CARREGAR e
       trecho que anda: sorteia a célula por `v3_cur.sorteia_celulas` (a célula `terreno` vem fixa
       do reset, §D7) e a velocidade pelo núcleo `sorteia_velocidade(tipo, nivel, tarefa, n, gerador,
       k)`:
       - `reto`: uma das 4 direções; módulo U[0,25; teto], teto = `frente_tetos[k]` para a frente e
         min(`frente_tetos[k]`, 1,0) para trás e de lado; wz 0;
       - `parado`: tudo zero;
       - `baixo`: metade linear (4 direções, U[0,05; 0,25]), metade giro no lugar (±U[0,05; 0,25]);
       - `giro_andando` e `arco`: linear como o `reto` e wz ±U[0,25; `giro_tetos[min(k, 3)]`];
       - `giro_lugar`: linear 0, wz ±U[0,25; `giro_tetos[min(k, 3)]`];
       - `terreno`: como o `reto`, com teto min(teto, `terreno_v_max`).
       Para os outros envs (PEGAR, BOTAR, trechos parados), comando zero.
    2. `rel_heading_envs` 0, `rel_standing_envs` 0 (o `parado` é célula).
    3. `v_media_b` (E,3): média de (vx, vy, wz) da base nos últimos 40 passos (0,8 s), atualizada em
       `_update_metrics`; zera só no reset do episódio (S §19, conflito 5: "na troca de comando a
       janela atrasa até 0,8 s").
    4. Sucesso por segmento, no fecho do segmento (`_fecha_segmento`, `comando.py:2392-2437`):
       canal ativo (|c| ≥ 0,02): razão projetada ≥ 0,50 (linear: Σ(v·ĉ)dt / Σ|c|dt; giro: Σ(ω·sinal)dt
       / Σ|wz|dt); canal zero: deriva de posição < 0,10 m e de rumo < 0,25 rad a partir da âncora
       `v3_ancora`, gravada `ancora_s` depois da abertura do segmento e regravada `ancora_s` depois de
       cada empurrão (`v3_t_empurrao`); um segmento que fecha antes da âncora não registra (PD-15);
       no CARREGAR, todo registro exige também `não v3_perdeu` (PD-50);
       chama `v3_cur.registra(celula, 1, sucesso)`. Um segmento aberto numa terminação que não é
       `time_out` registra falha (S §11.3 "sem queda").
    5. `razao_marcha` fica, como diagnóstico.
  - `CadeiaCmd` (termo de comando novo): o vetor de comando é o one-hot (E,6) de `v3_tarefa`;
    `resampling_time_range` infinito. Estado por env: `v3_fatia`, `v3_trecho`, relógio do trecho,
    duração sorteada, relógio do detector, atraso pendente, `v3_fim`. O `CadeiaCmd` também dá o
    `passo` da `v3_odo` uma vez por passo da política; o `atualiza_percepcao` só lê (PD-44). A
    tabela de trechos é dado:

    | fatia | trechos (tarefa publicada, duração) |
    |---|---|
    | ANDAR | (ANDAR, 20 s) |
    | B | (ANDAR parado, `espera_s`), (PEGAR, detector + `atraso_pegar_s`), (CARREGAR, `carregar_b_s`, comando sorteado na abertura e fixo) |
    | C | (ANDAR parado, `espera_s`), (PEGAR, detector + `atraso_pegar_s`), (CARREGAR parado, `carregar_c_s`), (BOTAR, detector + `atraso_botar_s`, one-hot BOTAR no atraso), (ANDAR parado, `andar_final_s`) |

    O núcleo puro `avanca_trecho(trecho, relogio, duracao, aviso, atraso, dt, tabela)` decide a
    troca. Na fase D o `aviso` do detector é sempre falso (as fatias B e C ainda não abrem). Ao fim do
    último trecho, `v3_fim` = True (S §7: o episódio acaba em `time_out`).
  - `forca_de_apoio`: cópia.
- Teste (`test_comando`): `sorteia_velocidade` com 100 000 amostras: `reto` no nível 2 tem módulos em
  [0,25; 0,75] para a frente e em [0,25; 0,75] de lado, e as 4 direções com 25 ± 1%; `giro_lugar` tem
  linear 0; `baixo` fica em [0,05; 0,25]; `avanca_trecho` na fatia C passa do PEGAR ao CARREGAR
  parado só depois do aviso mais o atraso, e o one-hot fica em BOTAR no atraso do BOTAR; o sucesso
  do segmento com 60% de razão passa e com 40% falha; o parado com deriva de 0,15 m depois da âncora
  falha; o parado que anda 0,30 m nos primeiros 1,5 s (frenagem) e 0,05 m depois passa; um empurrão
  no meio regrava a âncora; um segmento de 1 s não registra.
- Contagem: 2 526 → ~600. Commit: `feat(v3): twist por célula e cadeia de trechos`.

### D4 — `curriculo.py` (Sonnet)

- Origem: `g1_limpo/curriculo.py:61-97` (padrão `garante_*`); a guarda de iteração do
  `g1_limpo` (`ultima_iter_bal`, `g1_limpo/runner.py:47-52`).
- O que fazer: `garante_v3(env)` cria todos os buffers da §6.4 (idempotente); `mascara(env,
  tarefas)`; o termo `v3(env, env_ids)`: quando `common_step_counter // passos_por_iteracao` muda,
  chama `v3_cur.fecha_iteracao()` (o `Curriculo` conta as próprias iterações, §6.3); atualiza
  e devolve o dict de log da §6.7. O `v3_freio` e o `v3_degrau` são do `comando.cadeia`, na abertura de
  cada trecho (§6.4): no reset o currículo roda antes do comando, e o termo `v3` gravaria o freio da tarefa
  do episódio anterior (29/10).
  `passos_por_iteracao` vem do `knobs` (`Balanceador.passos_por_iteracao`, 24), o mesmo campo do
  `num_steps_per_env` do `rl_cfg()`.
- Teste (`test_curriculo`): `mascara` com tarefas ("ANDAR", "CARREGAR") e `v3_tarefa` = [0,1,2,3]
  dá [True, False, True, False] (bool); `garante_v3` duas vezes não recria o `v3_cur`; com um env
  falso (`SimpleNamespace`), `fecha_iteracao` roda uma vez por iteração e não por reset.
- Contagem: 537 → ~250. Commit: `feat(v3): estado do currículo no env`.

### D5 — `recompensas.py`, termos da locomoção e cercas (Sonnet)

- Origem: `g1_limpo/recompensas.py:180-212` (`contato_mesa`, para a fase E), os termos do mjlab
  (`envs/mdp/rewards.py`: `is_alive`, `joint_acc_l2`, `action_rate_l2`, `flat_orientation_l2`), a
  medida da margem do CoM de `g1_limpo/ik/monta_pose.py` (`margem_apoio`).
- O que fazer: um núcleo puro e um invólucro por termo. Todo termo recebe `tarefas` (§6.6) e zera
  os positivos depois da caixa perdida.

  | termo | núcleo | dado do env | tarefas (S §9.2) |
  |---|---|---|---|
  | `rastreio_linear` | `potencial.rastreio_normalizado` com c = ‖cmd_xy‖, erro² = ‖cmd_xy − v̄_xy‖², v² = ‖v̄_xy‖²; vezes `qualidade(env)` (1 na fase D; a fase E troca pelo `CarregarComQualidade`) | `twist.v_media_b` | todas (no PEGAR e no BOTAR, comando zero dá só o custo da deriva) |
  | `rastreio_angular` | idem com c = abs(wz), v = v̄_wz; σ₀ = `sigma_zero_ang` (o linear usa `sigma_zero_lin`, PD-49) | idem | todas |
  | `passo` | no toque (contato agora e não no passo anterior): L = (p_toque − p_apoio)·ĉ_w − (p_toque⁰ − p_apoio⁰)·ĉ_b (PD-16); r = min(relu(L)/`L_alvo`, 1)² · max(r_lin, 0) (PD-48); zero com ‖cmd_xy‖ < 0,02 | sites dos pés; `pes_chao`; `pes_de_pe_b` da `referencias.npz` | ANDAR, CARREGAR |
  | `vivo` | 1 | — | ANDAR, CARREGAR |
  | `altura_pe` | exp(−Σ_pés (h − 0,10)²·tanh(2·‖v_xy‖)/0,05) | `altura_pes`, velocidade dos `*_ankle_roll_link` | ANDAR, CARREGAR |
  | `escorrego` | Σ_pés ‖v_xy‖·contato | idem, `pes_chao` | todas |
  | `vz`, `giro_xy` | v_z²; ω_x² + ω_y² da base | base | todas |
  | `vel_junta`, `acel_junta`, `taxa_acao` | Σ q̇²; mjlab; mjlab | — | todas |
  | `energia` | Σ abs(τ·q̇) (`qfrc_actuator`) | — | todas |
  | `limite_junta` | `potencial.excesso_de_junta` com o `jnt_range` do modelo, mole 0,95 | — | todas |
  | `desvio_bracos`, `desvio_cintura`, `desvio_quadril` | Σ abs(q − alvo) nas 14 juntas dos braços, nas 3 da cintura, nos 4 `hip_roll`/`hip_yaw`; alvo = `q_de_pe` no ANDAR e `q_com_caixa(a)` no CARREGAR | — | ANDAR, CARREGAR |
  | `tronco_plano` | mjlab `flat_orientation_l2` | — | ANDAR, CARREGAR |
  | `altura_base` | (h_pelve − 0,78)² | `altura_pelve` | ANDAR, CARREGAR |
  | `cerca_sola` | Σ_pés em contato relu(incl_graus − 5) | sites dos pés | PEGAR, BOTAR |
  | `cerca_giro` | relu(max(−q_hip_yaw_E, 0)_graus − 30) + relu(max(q_hip_yaw_D, 0)_graus − 30) | — | todas |
  | `cerca_tronco` | relu(incl_tronco_graus − (tronco_IK_graus + 15)); tronco_IK = 0 no ANDAR e no CARREGAR, a referência do alvo corrente no PEGAR e no BOTAR | `torso_link` | todas |
  | `cerca_com` | relu(0,03 − margem) só com os dois pés em contato (PD-35); margem = distância com sinal do CoM combinado (robô + caixa presa) às arestas do fecho convexo dos 8 cantos das solas | `env.sim.data.subtree_com[:, raiz]` e `env.sim.model.body_subtreemass[:, raiz]`, raiz = `robot.indexing.root_body_id` (os campos do modelo têm forma (mundos, corpos); 29/10), massa da caixa | todas |
  | `cerca_torque` | Σ_j relu(f_j − 0,8)/0,2, com f_j = abs(τ_j)/τ_max_j nas juntas soltas e, em cada par de `pares_torque`, um termo só com f = (abs(τ_p) + abs(τ_r))/`tau_par` (PD-45) | `qfrc_actuator` | todas |
  | `cerca_pelve` | relu(abs(h_pelve − pelve_IK) − 0,08); pelve_IK = `pelve_de_pe` no ANDAR, `pelve_com_caixa(a)` no CARREGAR, a referência do alvo no PEGAR e no BOTAR | `altura_pelve` | todas |
  | `cerca_recuo` | relu(abs(recuo − recuo_IK) − 0,08); recuo = ‖xy da caixa − xy da pelve‖ no mundo, a mesma medida da `grade_recuo` (`dados/gera_referencias.py:109`); recuo_IK = `Recompensa.recuo_carregar` no CARREGAR, `v3_ref.recuo(v3_topo, v3_meia)` no PEGAR e no BOTAR (29/10) | pelve, caixa | PEGAR, CARREGAR, BOTAR (a fase E liga) |
  | `contato_fora` | nº de corpos fora dos pés com força > 1 N no chão + contatos de autocolisão (+ a caixa fora da janela da pega, fase E) | `nao_pes_chao`, `auto_colisao` | todas |
  | `terminacao` | 1 numa terminação que não é `time_out`; peso −200 (× dt = −4) | — | todas |

  Unidades das cercas: ângulo em graus, comprimento em metros, torque em fração. O `item3_escada`
  usa os mesmos núcleos e as mesmas unidades.
- Teste (`test_recompensas`): passo: L = 0,20 m a 0,5 m/s com r_lin 1 → 1; L = 0,40 → 1 (satura);
  L = 0,10 → 0,25, e a soma por segundo com L = 0,10 e o dobro da cadência dá a metade da de L = 0,20 (PD-48);
  passo de lado com os pés na largura da pose de pé → 0; `altura_pe` com os pés parados → 1;
  `cerca_giro` com `hip_yaw_E` = −40° → 10; `cerca_tronco` com 20° no ANDAR → 5; `cerca_com` com o
  CoM 5 cm dentro → 0 e 1 cm dentro → 0,02; um pé só no chão → 0; `cerca_torque` com 90% → 0,5;
  `cerca_torque` com `waist_pitch` 30 N·m e `waist_roll` 15 N·m → 0,5 (soma 45 de 50), e 0 na
  regra por junta;
  `excesso_de_junta` a 97,5% → 0,5; `vivo` zera com `v3_perdeu`.
- Contagem: 1 390 → ~800 no fim da fase H (~500 na fase D). Commit:
  `feat(v3): termos da locomoção da Unitree com os desvios e as cercas`.

### D6 — `terminacoes.py` (Sonnet)

- Origem: `g1_limpo/terminacoes.py:123-155` (`Caiu`).
- O que fazer: `fim_da_cadeia` (`time_out=True`): `episode_length_buf ≥ max` ou `v3_fim`; `queda`:
  nas tarefas ANDAR e CARREGAR, inclinação > 0,8 rad ou pelve a < 0,20 m do terreno; no PEGAR e no
  BOTAR, inclinação > 70° ou o joelho mais baixo a < 0,05 m do terreno (`altura_pelve`,
  `altura_joelhos`; PD-21).
- Teste (`test_terminacoes`): os núcleos com inclinação 0,9 rad no ANDAR (cai) e no PEGAR (não cai);
  pelve a 0,15 m do terreno com a origem do subterreno a +1,4 m (cai pelo terreno, e não pelo mundo).
- Contagem: 196 → ~200 no fim da fase H. Commit: `feat(v3): terminações por tarefa contra o terreno`.

### D7 — `eventos.py` (Sonnet)

- Origem: `g1_limpo/eventos.py:35-71` (`reset_base_por_elo`, a divisão por subconjunto).
- O que fazer:
  - `sorteia_fatia` (reset, o primeiro do dict): `v3_fatia` por `v3_cur.sorteia_fatias`; o trecho
    0; para as fatias ANDAR e B, sorteia se o episódio é de terreno, com a fração da célula
    `terreno` da tarefa (ANDAR; CARREGAR na B); episódio de terreno escolhe uma coluna (ANDAR: 1 a
    8; B: só as com plataforma) e a linha ≤ o nível de terreno da tarefa (CARREGAR: ≤
    `linha_max_carregar` da coluna, PD-20), e fixa a célula `terreno`; os outros ficam na coluna 0. Escreve
    `terrain.terrain_levels[ids]`, `terrain.terrain_types[ids]` e `scene.env_origins[ids]` =
    `terrain_origins[linha, coluna]` (`terrain_entity.py:216-233`).
  - Todo sorteio usa o `v3_gerador` (§6.4). O `sorteia_fatia` não escreve `v3_celula`: escreve o
    terreno, e o `twist` lê o `terrain_types` (§6.4, 29/10).
  - `base` (reset): fatia ANDAR: `Cena.reset_base_andar` (±0,5 m em x e y e rumo livre), sobre a origem; fatias B e C: a faixa
    `Cena.reset_base_manipula`, como o `reset_base_por_elo`. z = origem + `pelve_de_pe` + 0,01.
  - `juntas` (reset): `mdp.reset_joints_by_offset` com faixa 0 (o default é o `q_de_pé`).
  - `filtros` (reset): `v3_filtro.reinicia`, `v3_odo.reinicia`.
  - startup: `atrito_pe` (`dr.geom_friction`, 0,3–1,0, compartilhado, geoms `*_foot[1-7]_collision`);
    `massa_tronco` (`dr.body_mass`, soma de −1 a +3 kg no `torso_link`); `vies_encoder`
    (`dr.encoder_bias`, ±0,015); `atraso`: põe `max_lag` = 4 nos buffers de atraso dos atuadores
    (construídos com 12, D1) e 1 nos dos termos de observação (construídos com 3, D2) (PD-28).
  - intervalo: `empurrao` (`mdp.push_by_setting_velocity`, `interval_range_s` (5, 5), x e y ±0,5),
    envolvido para gravar `v3_t_empurrao` dos envs empurrados (PD-15).
- Teste (`test_eventos`): o núcleo de escolha de coluna e linha: fatia B nunca cai na coluna 6
  (`irregular`); CARREGAR com nível de terreno 12 fica na linha ≤ 9 nas escadas e nas rampas, ≤ 8
  no `obstaculos` e ≤ 7 no `caixas`; ANDAR com nível de terreno 12 chega à linha 12; ANDAR no plano
  fica na coluna 0.
- Contagem: 550 → ~450 no fim da fase H (~200 na fase D). Commit:
  `feat(v3): reset por fatia, terreno por tarefa e DR desde o começo`.

### D8 — `metricas.py` (Haiku)

- Origem: `g1_limpo/metricas.py:210-218`, `:341-470`, `:585-590` (as sem dependência, §2.2).
- O que fazer: copia as métricas da locomoção e acrescenta `passo_medio` (L no toque),
  `fracao_mapa_visto` (células do recorte com confiança > `Percepcao.limiar_visto`, 0,5). O `atraso_medio`
  saiu (29/10): o lag sorteado mora num atributo privado do mjlab, e o `atraso_max` já vai ao log pela E6.
  `termos(k) -> dict[str, MetricsTermCfg]`.
- Contagem: 590 → ~350 no fim da fase H. Commit: `feat(v3): métricas da locomoção`.

### D9 — `algoritmo.py` e `runner.py` (Sonnet)

- Origem: `g1_limpo/algoritmo.py:155-224` (vantagem por grupo) e `g1_limpo/runner.py` inteiro.
- O que fazer:
  - `PPOPorTarefa(PPO)`: só `compute_returns`. O grupo de cada amostra é o argmax das 6 últimas
    colunas da observação do crítico (`tarefa_critico`, §6.5). A normalização por grupo e os
    casos de grupo pequeno são os do `g1_limpo`. Sem `_fixa_one_hot` (não há normalizador).
    `CAMINHO = "g1_v3.algoritmo:PPOPorTarefa"`. O núcleo `normaliza_por_grupo(adv, grupo)` é puro.
  - `RunnerV3(MjlabOnPolicyRunner)`: `save` grava `infos["v3_curriculo"] = env.v3_cur.estado()`;
    `load` restaura; checkpoint sem a chave imprime um aviso e segue com o currículo novo.
- Teste (`test_algoritmo`): `normaliza_por_grupo` com 3 grupos sintéticos dá média 0 e desvio 1 em
  cada; um grupo de 1 amostra usa a regra de grupo pequeno.
- Contagem: 372 → ~200. Commit: `feat(v3): PPO com vantagem por tarefa e runner com currículo`.

### D10 — `env_cfg.py`, `tarefa.py`, `train.py`, `play.py` (Sonnet)

- Origem: o esqueleto de `g1_limpo/env_cfg.py:148-560` e `g1_limpo/__init__.py:38-70`.
- O que fazer:
  - `make_env_cfg(k=ATIVO, play=False)`: parte de `unitree_g1_rough_env_cfg(play)` (sim e viewer) e
    TROCA cena, sensores, terreno, observações, ação (escala 0,25, `use_default_offset=True`),
    comandos (`cadeia`, `twist`), recompensas (as tuplas `tarefas` da §D5), terminações, eventos,
    currículo e métricas pelos do pacote. `njmax`, `nconmax`, `impratio`, `cone` do `knobs`;
    `episode_length_s` 20; `decimation` 4; `timestep` 0,005. `assert` de que nenhum campo `None`
    do `knobs` chega ao cfg. No play: sem corrupção, sem `empurrao`, episódio de 10⁹ s.
  - `tarefa.py`: `TASK_ID = "Mjlab-G1-V3"`, `rl_cfg()` (o `unitree_g1_ppo_runner_cfg` com
    `experiment_name` "g1_v3", `obs_normalization` False no ator e no crítico, `class_name` =
    `CAMINHO`, `num_steps_per_env` = `Balanceador.passos_por_iteracao`) e o `register_mjlab_task` com `runner_cls=RunnerV3`. Uma segunda task,
    `TASK_ID_SEM_PERCEPCAO = "Mjlab-G1-V3-SemPercepcao"`, com o mesmo cfg e `Percepcao.ligada = False`,
    só para a corrida de custo do I5 (substitui a chave `PERCEPCAO` do notebook antigo); o teste
    confere as duas no registro.
  - `train.py` e `play.py`: cópias de `g1_limpo/train.py` e `play.py` com `import g1_v3.tarefa`.
- Teste (`test_cfg`): `make_env_cfg()` monta sem erro; os nomes e a ordem dos termos da §6.5; os
  pesos iguais ao `knobs`; cada recompensa tem a tupla `tarefas` da tabela da §D5; o último termo do
  crítico é `tarefa_critico`; o `rl_cfg` tem `obs_normalization` False e o `class_name` da v3.
- Contagem: 992 + 148 + 29 → ~550. Commit: `feat(v3): montagem do env e registro da task`.

### D11 — exportador, `pilota_v3.py`, `registra_juntas_v3.py` (Sonnet; três agentes em sequência)

- Origem: `g1_limpo/exporta_cena.py:56-426`, `pilota.py`, `registra_juntas.py`.
- O que fazer:
  - `g1_v3/exporta_cena.py`: `constroi_env` com `make_env_cfg(play=True)`; `coleta_cena` com os
    blocos da §6.5; grava `cena.mjb`, `cena.npz` e `contrato.json` (ordem das juntas, `q_de_pé`, kp,
    kd, escala, dt, blocos, one-hot, mapa, `T_BASE_SENSOR`, versão do pacote) na pasta de saída;
    `paridade` compara a observação do ator do env com a de `pilota_v3.monta_observacao`, com o
    ruído desligado, bloco a bloco (tolerância 1e−5).
  - `pilota_v3.py` (raiz): carrega `knobs`, `alvos`, `estimador` e `percepcao_lidar` por caminho
    (`importlib.util.spec_from_file_location`); `monta_observacao` dos 1 447 canais com o histórico
    de 5 por bloco e o preenchimento no primeiro passo (I §2.1); o lidar no MuJoCo clássico com
    `mj_multiRay` e o mesmo padrão; o filtro com o transporte da odometria, inclusive o `z_ref` (o pé
    mais baixo no frame da odometria, PD-7); o estimador no laço; o laço de rumo k = 2 do driver; o BOTAR de um
    comando só; o detector de fim (`alvos.detecta_fim`) como status. Teclado do `pilota.py`.
  - `registra_juntas_v3.py` (raiz): o roteiro por fases do `registra_juntas.py` com as tarefas da v3;
    o CSV com as colunas de hoje mais `tarefa`, `aviso_fim`, `idade_caixa`, `fracao_mapa_visto`.
  - Decisões de 29/10 (dúvidas da D11a): o contrato entre o exportador e o driver é a seção "Contrato
    com o pilota_v3" da docstring do `exporta_cena.py` (chaves do `cena.npz` e do `contrato.json`, e a
    assinatura de `monta_observacao`, que recebe `mapa` e `caixa` de fora). O mapa do driver sai de uma
    classe `Percepcao` do `pilota_v3` (lidar por `mj_multiRay`, filtro e odometria; `reinicia(m, d)`;
    `passo(m, d) -> (882,)`), fora do `monta_observacao`. A `paridade` prova os 7 blocos sem
    `caixa`/`mapa` e o histórico (tol 1e−5) e só imprime, sem portão, a diferença entre o mapa da
    `Percepcao` e o do env (Δmax das alturas nas células vistas pelos dois e a fração de concordância
    da máscara de vistas); o dono julga na fase I. O `T_BASE_SENSOR` do `contrato.json` é a referência
    em `q_de_pé`; o driver usa a pose viva do site `mid360` (`d.site_xpos`, `d.site_xmat`), como o
    robô real usa a FK. A versão do pacote = `git rev-parse --short HEAD` na exportação
    ("desconhecida" se o git falha). O exportador confere kd = −`biasprm[2]` além de kp =
    `gainprm[0]`. O driver não simula o atraso do atuador: o robô real tem o próprio.
- Teste: `py_compile` dos três; o `monta_observacao` com uma cena falsa devolve 1 447 canais na ordem
  do `Contrato.blocos_ator`.
- Contagem: ferramentas, fora da meta. Commits: um por arquivo.

### D12 — smoke da Parte 1 (Sonnet)

- Arquivos: `g1_v3/smoke.py` (seções 1 a 15).
- Seções (o dono roda na fase I):
  1. o cfg monta e a task registra; 2. um env de 4 mundos em CPU nasce, e o ator tem 1 447 canais;
  3. a ordem e o tamanho dos blocos = `Contrato.blocos_ator`; os 5 lugares do histórico iguais
  depois do reset; 4. ação zero → alvo do PD = `q_de_pé`; kp e kd do modelo = `Contrato.pd`; 5. fase
  5a: trocar o `_local_directions` muda as distâncias no passo seguinte; 6. o mapa: parado 20
  passos no plano, nenhuma célula a menos de 0,8 m da base vista (o lidar da cabeça não vê o chão a
  menos de ~0,9 m, I §2.4); depois, com a base movida 1,0 m para a frente por escrita da pose em 50
  passos, ≥ 90% das células da metade da frente vistas (I §2.4 mediu 205 de 210) e alturas a ±2
  cm; 7. env de
  terreno na coluna e na linha certas, env do plano na coluna 0; 8. o lag do atuador fica em {0..4} e
  o da observação em {0, 1}, constantes no episódio; 9. todo termo de recompensa finito no reset, e
  `vivo` = 0,15·dt no ANDAR; 10. `queda` por inclinação e por altura sobre o terreno; 11. só a fatia
  ANDAR no começo; 12. `save` → `load` do runner devolve o mesmo `v3_cur.estado()`; 13. o
  `class_name` do algoritmo; 14. 200 passos com ação aleatória em 16 mundos sem NaN; 15. tempo por
  passo com e sem `Percepcao.ligada` (imprime; fase 5c em CPU).
- Contagem: o smoke fica fora da meta. Commit: `test(v3): smoke da Parte 1`.

### D13 — ponto de entrada do container no RunPod (Sonnet)

- Arquivos: `g1_v3/container/entrada.sh`; o teste `g1_v3/testes/test_entrada.py` sai da lane D0.
- Origem: as lições do `g1_limpo/kaggle/g1_limpo_zero_kaggle.ipynb` (células 1 a 5 e 8) e a CLI
  do mjlab 1.5.3 (`mjlab/scripts/train.py:71-95`, `mjlab/utils/os.py:23-60`), que o `train.py` da
  D10 já chama.
- Contrato com a imagem (PD-22; a imagem é do dono). A imagem traz o Python, o torch com CUDA e
  `mjlab==1.5.3` (ele pina `mujoco-warp` e `rsl-rl-lib` 5.4.0). Ao subir, ela clona
  `--branch g1_v3 --depth 1` em `$RAIZ` e chama `bash $RAIZ/g1_v3/container/entrada.sh`. O repo não
  instala pacote nenhum.
- Variáveis de ambiente do pod:

  | variável | padrão | o que faz |
  |---|---|---|
  | `RUN` | — (obrigatória) | nome da corrida; nome novo a cada mudança de conjunto |
  | `RETOMA` | — (obrigatória, 0 ou 1) | chave explícita: a retomada nunca sai da presença de um checkpoint |
  | `ITERS` | 30000 | its DESTA sessão (o rsl_rl soma à it do checkpoint: `total_it = start_it + num_learning_iterations`) |
  | `NUM_ENVS` | 4096 | envs; o I5 mede 4 096 e 8 192 |
  | `PERCEPCAO` | 1 | 0 escolhe `Mjlab-G1-V3-SemPercepcao` (só o I5) |
  | `SAIDA` | `/workspace/g1_v3` | raiz dos logs e dos checkpoints, no volume do pod; o destino final fica para depois (PD-22) |
  | `TENSORBOARD` | 1 | sobe o TensorBoard na porta 6006 em segundo plano |
  | `SECO` | 0 | 1 pula as checagens de GPU e só imprime o comando final (para o teste) |

- O que fazer, nesta ordem:
  1. `set -euo pipefail`; `export MUJOCO_GL=egl`; falha com mensagem se `RUN` ou `RETOMA` faltam, ou
     se `RETOMA` não é 0 nem 1.
  2. Sem `SECO`: `nvidia-smi` com saída não vazia; num subprocesso Python, `torch.cuda.is_available()`
     verdadeiro, `importlib.metadata.version("mjlab") == "1.5.3"` e `version("rsl-rl-lib") == "5.4.0"`
     (o `RslRlModelCfg` do 1.5.1 quebra o 5.4.0, notebook antigo célula 2); imprime as versões do
     torch e do mujoco, o nome da GPU e `git -C $RAIZ log --oneline -1`.
  3. A pasta do experimento é `$SAIDA/g1_v3` (`experiment_name` da D10). Com `RETOMA=0`: falha se já
     existe nela uma pasta `*_$RUN`. Com `RETOMA=1`: escolhe a pasta `*_$RUN` mais nova que tem um
     `model_*.pt` (uma sessão que caiu antes do primeiro checkpoint fica para trás); sem nenhuma,
     falha.
  4. Monta o comando: `python -m g1_v3.train <task> --env.scene.num-envs $NUM_ENVS --agent.run-name
     $RUN --agent.max-iterations $ITERS --agent.logger tensorboard --log-root $SAIDA`; com
     `RETOMA=1`, mais `--agent.resume --agent.load-run '^<pasta>$'` (o bool do tyro é um flag sem
     valor; o mjlab acha o `model_*.pt` de maior número dentro da pasta).
  5. Com `TENSORBOARD=1` e sem `SECO`: `tensorboard --logdir $SAIDA --port 6006 --bind_all &`.
  6. Com `SECO=1`: imprime o comando e sai com 0. Senão, `cd $RAIZ` e `exec` do comando.
- Teste (`test_entrada`, CPU, `SECO=1`, `SAIDA` num diretório temporário, por `subprocess`):
  `bash -n` passa; sem `RUN`, sem `RETOMA` ou com `RETOMA=2` → código ≠ 0; `RETOMA=0` com uma pasta
  `2026-01-01_00-00-00_x` → ≠ 0; `RETOMA=0` limpo → o comando tem `Mjlab-G1-V3 `,
  `--agent.run-name x`, `--log-root <tmp>` e não tem `--agent.resume`; `RETOMA=1` com duas pastas
  `*_x`, a mais nova sem `model_*.pt` → `--agent.load-run '^<a mais velha>$'`; `RETOMA=1` sem pasta →
  ≠ 0; `PERCEPCAO=0` → `Mjlab-G1-V3-SemPercepcao`; o texto do script não tem `git push`, `gh ` nem
  `pip`.
- Contagem: 0 → ~60 (script) e ~80 (teste). Commit: `feat(v3): ponto de entrada do container (RunPod)`.

### D14 — revisão da Parte 1 (Opus)

Contra este plano e a S: contrato da §6, unidades, máscaras de tarefa, ordem dos termos, termos
que pagam positivo no PEGAR e no BOTAR (nenhum), alturas contra o terreno, zero import do
`g1_limpo`. As correções vão num despacho seguinte.

---

## 12. Fases E a H — Partes 2, 3, 3b e 4

Mesmo formato da fase D. Cada fase começa com a lane de testes (Haiku) e termina com a revisão do
Opus. Os arquivos já existem; cada tarefa ACRESCENTA ao arquivo, e nenhuma lane da mesma onda toca o
arquivo de outra.

Os buffers das fases E a H já estão na §6.4; o `garante_v3` da fase D cria todos desde o começo.

Em cada fase, a tarefa 0 (E0, F0, G0, H0) é a lane de testes (Haiku): ela escreve, antes do código,
os casos da linha "Teste" de cada tarefa da fase, em `g1_v3/testes/test_<módulo>_<fase>.py`, e as
seções do smoke ficam com a tarefa de smoke da fase. A última tarefa (E11, F9, G7, H8) é a revisão do
Opus contra o plano e a S.

### 12.1 Fase E — Parte 2: cadeia B

Atende: S §16 Parte 2, §6.1, §6.2, §7 (fatia B), §8 (P0–P4), §9.1, §9.2 (PEGAR e CARREGAR), §10
(caixa perdida), §11.2 (níveis do PEGAR), §11.1 (fatia B), §12 (caixa), §13 (estimador). Enunciado:
E §3 "Pega", "Caixa íntegra e nivelada", "Contato", "Progressão na cadeia", E §4.3, E §4.4.

| onda | tarefas |
|---|---|
| E-0 | E0 testes |
| E-1 | E1 `eventos.py`, E2 `estimador.py`, E3 `comando.py`, E4 `recompensas.py`, E5 `terminacoes.py`, E6 `curriculo.py` + `balanceador.py` (DR por nível), E7 `metricas.py` |
| E-2 | E8 `observacoes.py` (`caixa` e crítico), E9 `pilota_v3.py` + `registra_juntas_v3.py`, E10 smoke |
| E-3 | E11 revisão |

**E1 — `eventos.py` (Sonnet).** Origem: `g1_limpo/eventos.py:74-211` (`_topo_por_nivel`,
`orientacao_de_nascimento`, `posiciona_cena`), `:214-253` (`afasta_cena` → `estaciona_cena`),
`:256-369` (`tamanho_caixa`, `carga_caixa`); `g1_limpo/env_cfg.py:412-417` (`atrito_caixa`, molde
do `atrito_pad` e do `atrito_laje`).
Mudanças: (1) toda escrita de z soma `env_origins[:, 2]` (viab. D3), e `v3_topo` guarda o topo sobre o
chão local; (2) o nível vem de `v3_cur.nivel("PEGAR")`: topo U[h_k; 0,55], na fatia C
U[max(h_k; 0,30); 0,55] (S §7); (3) massa U[1; m_k] na B; na C, U[1; m_j] com j o maior nível do PEGAR
que já passou no portão (S §11.2); (4) `estaciona_cena` na fatia ANDAR: laje e caixa 20 m acima da
origem; (5) `laje_fora(env, ids)` (chamada pelo `comando` na abertura do CARREGAR com comando que
anda, [C39]); (6) `atrito_pad` e `atrito_laje` (reset; `dr.geom_friction`, coluna 0, nos pads de
palma e na geom da laje; PD-46), ligados pelo estado de DR do `v3_cur` (§6.8); antes do portão,
pads 1,0 e laje 1,0; (7) `vies_imu` (reset): com o portão do ANDAR no estado de DR, sorteia `v3_vies_giro`
(±0,05 rad/s) e `v3_incl_imu` (giro de até 1°), senão zero e identidade. Teste: o núcleo do topo com
a origem a +1,4 m põe o topo da laje a 1,4 + h; a fatia C nunca sorteia topo < 0,30. Commit:
`feat(v3): cena da cadeia B com z sobre o chão local`.

**E2 — `estimador.py` (Sonnet).** `reduz_simetria_cubo`, `visivel`, `EstimadorCaixa` e
`alvo_no_frame_da_base` da §6.3. Teste: identidade → 0; 90° em z → 0; 100° em z → 10° em z; 30° em
z → 30°; caixa a 1 m à frente da câmera sem cápsula → visível; atrás → não; a 0,20 m → não (alcance
mínimo); cápsula de braço no segmento → não; leitura do instante t entregue em t + 33 ms; sem
leitura a idade cresce 0,02 s por passo e para em 2 s; ruído de posição com σ 0,5 cm medido em
10 000 amostras a 10%. Contagem: 60 → ~300. Commit: `feat(v3): estimador simulado da caixa e
visibilidade da D435i`.

**E3 — `comando.py`, cadeia B (Sonnet).** No `CadeiaCmd`:
1. Abertura do PEGAR: `v3_alvo_w` = centro da caixa ESTIMADA na abertura, com os erros do estimador (o controlador escolhe a caixa vista, S §6.4; dono, 15/10);
   `v3_caixa_pega_w` = o mesmo; `v3_alvo_idx` = P0; σ = distância inicial das palmas ao alvo P0.
2. A cada passo, no PEGAR: o `ok_agora` de cada alvo pela §6.2 `Alvos` (P0, P1: as duas palmas a ≤ 3
   cm; P2: as duas palmas em contato; P3: caixa sem apoio e inclinação ≤ limite do degrau; P4: caixa a
   ≤ 5 cm do alvo de transporte e pelve a ≤ 3 cm de `pelve_com_caixa(a)`, por 0,5 s), sempre com os
   invariantes dentro do limite (S §8.2: "impede o avanço"); `alvos.avanca`; no avanço: `v3_avancou`
   = True, σ do alvo seguinte = distância atual (mínimo 1 cm), `v3_pegou` = True ao passar do P2, e
   `v3_p4_fechado` = True ao fechar o P4. Depois do P4 o índice fica em P4.
3. Detector de fim (I §4.2) com a referência da pose de pé com caixa: `v3_aviso`; depois do aviso,
   espera U[0; 1] s e troca para o CARREGAR (S §7). Na fatia B, o comando do CARREGAR sai na
   abertura e fica até o fim (`time_left` = ∞); com comando que anda, `laje_fora`.
4. Caixa perdida (S §10): o núcleo `caixa_perdida_agora` (lógica de `g1_limpo/terminacoes.py:52-120`:
   caixa com o fundo a < `caixa_folga_chao`
   do terreno pelo `altura_caixa`; ou solta fora do alvo acima de 1,2 m/s; ou, de P0 a P2, fora do
   retângulo do topo da laje) liga `v3_perdeu_agora`. Com `caixa_perdida_vira_andar`: `v3_perdeu`
   = True, `v3_tarefa` = ANDAR (parado) até o fim, `v3_alvo_idx` = B4 com σ = a distância à pose de
   pé no instante da perda (a volta à pose de pé paga uma vez, como potencial; ao fechar o B4, o
   índice vai a −1). Sem a chave, a terminação `caixa_perdida` encerra (E5).
5. Sucesso do PEGAR (S §11.3): `v3_p4_fechado`, o trecho seguinte sem terminação e `não v3_perdeu` no registro (PD-50); registra no fim
   do trecho seguinte (B: no fim do episódio; C: na abertura do BOTAR) ou falha numa terminação.
Teste: o núcleo de avanço com estados sintéticos passa de P0 a P4 na ordem e não avança com a caixa
a 40°; a caixa perdida com a chave liga `v3_perdeu` e troca a tarefa; o sucesso do PEGAR conta
falha com queda no CARREGAR e com a caixa perdida no CARREGAR, chave ligada (PD-50). Commit: `feat(v3): cadeia B com alvos macro e detector de fim`.

**E4 — `recompensas.py`, manipulação (Sonnet).**

| termo | núcleo | máscara |
|---|---|---|
| `progresso` | `r = w_p·(γ·Φ(s′, i) − Φ(s, i)) + B_alvo[i]·avançou` no passo do avanço (o Φ do alvo VELHO no estado novo; D1 da Parte 0, dono 29/10: com Φ(s′, i′) o passo do avanço cobrava a queda de Φ e o `B_alvo` não cobria em 7 de 8 alvos), e `w_p·(γ·Φ(s′, i) − Φ(s, i))` nos outros passos; `Φ = k_principal · Π g`; `k_principal`: P0–P2 e B3, a média das distâncias das duas palmas aos alvos no frame da caixa; P3, P4, B1, B2, a distância da caixa ao alvo (vezes `fator_pouso` em B1 e B2); B4, `√((Δpelve/0,03)² + (Δtronco/5°)²)`; σ = `v3_sigma`. `g`: rampas da §6.2 `Alvos` (invariantes P0–P2, P2–B2, B2–B4; faixa do tronco, do recuo e do CoM, este só com os dois pés no chão, PD-35) e `g_mãos` (PD-19). No passo em que o índice vira −1 (troca de tarefa), r = 0 (PD-36) | `v3_alvo_idx` ≥ 0 |
| `qualidade` (fator do rastreio no CARREGAR) | `núcleo(‖caixa − alvo_transporte‖; 0,05) · rampa(incl) · g_mãos`, aplicado só à parte positiva do rastreio: `q·max(r,0) + min(r,0)` (PD-37) | CARREGAR |
| `contato_laje` | soma de `contato_mesa` (`g1_limpo/recompensas.py:180-212`) nos três sensores da laje, `joelho_N` pelo degrau (§6.9) | PEGAR, BOTAR |
| `freio` | cópia de `velocidade_por_regime` (`:1179-1329`) com `mascara` no lugar de `limpo_estado`, `v3_freio` no lugar de `limpo_freio`, e o comando zero lido do `twist` | PEGAR, CARREGAR, BOTAR |
| `contato_fora` (acréscimo) | + contatos robô × caixa (`caixa_robo`) fora da janela da pega (P2 a B3 e o CARREGAR) | todas |
| `perda_caixa` | no passo da perda: `perda_caixa` + (`devolve_b_alvo` ? `v3_b_pagos` : 0), peso −1 | todas |
| `cerca_recuo` | liga | PEGAR, CARREGAR, BOTAR |

A referência macro de cada alvo (para `cerca_tronco`, `cerca_pelve`, `cerca_recuo`, faixa do Φ e o
canal `referencia` do crítico): P0 a P3, a pega em (`v3_topo`, a); P4, a pose de pé com caixa; B1 a
B3, a pega em (`v3_topo_destino`, a); B4, a pose de pé sem caixa (sem recuo). Sem isto, a cerca da
pelve da pega cobraria o robô por ficar de pé no P4.
Teste: `progresso` no caminho roteirizado do item 1 bate com a saída do `item1_phi`; pairar soma ≤ 0;
o passo da troca paga 0; `qualidade` = 0 com a caixa a 45° no degrau 0 (rampa de 30° com Δ 15°) e o
rastreio negativo continua negativo;
nenhum termo paga positivo por passo no PEGAR com o robô parado (soma dos termos positivos = 0).
Commit: `feat(v3): potencial de progresso e termos da manipulação`.

**E5 — `terminacoes.py` (Sonnet).** `caixa_perdida`: lê `v3_perdeu_agora`; verdadeira só com a
chave desligada. Teste: com a chave ligada nunca termina; desligada, termina no passo da perda.
Commit: `feat(v3): terminação da caixa perdida com a chave`.

**E6 — `curriculo.py` e `balanceador.py`, DR por nível (Sonnet).** `Curriculo` ganha o estado de DR
(`mu_min_pad`, `atrito_laje`, `com_caixa`, `atraso_max`, `rfi`, `forca_caixa`) que sobe pela §6.8
quando os portões fecham, e entra no `estado()`. O termo `v3` aplica `atraso_max` nos buffers (PD-28,
PD-41). Teste: o PEGAR no portão do nível 0 liga o atrito dos pads e da laje e o atraso 0–2; cada nível seguinte desce
o μ mínimo em 0,025 até 0,5; nada desce. Commit: `feat(v3): DR por nível no currículo`.

**E7 — `metricas.py` (Haiku).** Acrescenta: Φ médio e tempo por alvo; inclinação da caixa (pico no
P3); força total robô → caixa; deslocamento da caixa na laje de P0 a P2; idade média das leituras;
fração de avisos do detector (S §14). Commit: `feat(v3): métricas da manipulação`.

**E8 — `observacoes.py` (Sonnet).** O termo `caixa` troca o `caixa_zero`: o `atualiza_percepcao`
(D2, a mesma guarda no env) ganha `visivel` (site `d435`, 9 pontos da caixa, cápsulas dos braços),
`v3_estimador.passo` e a saída `saida_b` + `alvo_no_frame_da_base` em `v3_caixa_obs` (alvo zero no
CARREGAR; os 11 canais zero no ANDAR, inclusive nas esperas, [C13]); o termo `caixa(env)` chama a
função e devolve `v3_caixa_obs`, no ator e no crítico (o estimador dá um passo só por passo da
política). O `atualiza_percepcao` passa a girar os pontos pelo `v3_incl_imu` (na fase D, a identidade; PD-42). A caixa estimada não entra no filtro do corpo (PD-10).
Os canais do crítico (`caixa_verdade`, `alvo_verdade`, `alvo_macro`, `referencia`, `massa`,
`forca_caixa`) passam a ler os buffers. `giro_base` vira função própria (o sensor do mjlab +
`v3_vies_giro`) e `gravidade` passa por `v3_incl_imu` (DR §3.4); o ruído do mjlab fica igual.
Commit: `feat(v3): bloco da caixa pelo estimador`.

**E9 — driver (Sonnet).** `pilota_v3.py`: comando PEGAR com o alvo no centro da caixa; status do
detector. `registra_juntas_v3.py`: roteiro B (ANDAR parado → PEGAR → CARREGAR). Commit:
`feat(v3): driver com a cadeia B`.

**E10 — smoke da Parte 2 (Sonnet).** Seções 16 a 24: 16. fatia B forçada: topo da laje = origem +
`v3_topo`, caixa sobre ela; 17. a abertura do PEGAR põe o alvo e o P0; 18. no PEGAR com ação zero por
50 passos, a soma dos termos que pagam positivo é 0; 19. caixa teleportada ao chão: com a chave, perda
paga uma vez e tarefa ANDAR; sem a chave, termina; 20. o estimador vê a caixa à frente e a idade cresce
com os braços na frente (juntas escritas da `q_com_caixa`); 21. o robô posto na pose de pé com caixa
dispara o aviso depois de 0,5 s; 22. a laje sai com o CARREGAR que anda; 23. o tamanho do crítico
confere com a §6.5; 24. o atraso fica em {0..8} no atuador depois do portão do PEGAR (estado forçado).

### 12.2 Fase F — Parte 3: cadeia C

Atende: S §16 Parte 3, §7 (fatia C), §8.2 (B1–B4 e o invariante B2–B4), §9.1 (pouso devagar),
§10 (chegada), §11.2 (níveis do BOTAR). Enunciado: E §4.5, E §3 "Caixa íntegra e nivelada".

| onda | tarefas |
|---|---|
| F-0 | F0 testes |
| F-1 | F1 `comando.py`, F2 `eventos.py`, F3 `terminacoes.py`, F4 `recompensas.py`, F5 `curriculo.py`, F6 `metricas.py` |
| F-2 | F7 driver, F8 smoke |
| F-3 | F9 revisão |

**F1 — `comando.py`, cadeia C (Sonnet).** Abertura do BOTAR: `posiciona_destino` (F2);
`v3_alvo_w` = o ponto de soltura do F2 no topo da laje de destino + (0, 0, a); `v3_alvo_idx` = B1. `ok_agora`: B1,
caixa a ≤ 3 cm do alvo + 5 cm na horizontal; B2, caixa a ≤ 0,10 m do alvo, nivelada, apoio ≥
0,8·m·g por 0,3 s; B3, palmas a ≥ 5 cm dos planos das faces, caixa apoiada e com ‖v‖ < 0,05 m/s
(PD-39), por 0,5 s; B4, pelve a ≤ 3 cm de `pelve_de_pe` e tronco a ≤ 5°, por 0,5 s. No primeiro apoio,
`v3_ponto_apoio` = posição da caixa (invariante B2–B4). Detector com a pose de pé sem caixa; depois do
aviso, espera U[0; 1,5] s com o one-hot em BOTAR, e o ANDAR parado de 1,5 s. Sucesso do BOTAR: B4
fechado, o ANDAR parado final sem terminação e `não v3_perdeu` no registro (PD-50). O CARREGAR parado da C não registra célula.
Commit: `feat(v3): cadeia C com B1–B4`.

**F2 — `eventos.py` (Sonnet).** `posiciona_destino(env, ids)` (PD-47): sorteia θ U[−10°; 10°] e
y U[−0,10; 0,10] m; o ponto de soltura, no frame da base, fica em x = `recuo(h_dest, a)` + o jitter
só para longe ([C38]) + e(θ), com e(θ) = `destino_borda`·cos θ + `prateleira_meia_xy`·sen|θ| −
`destino_borda`, e em y; a laje (mocap) gira para o rumo da base + θ, com o centro no ponto de
soltura + R(rumo + θ)·(`prateleira_meia_xy` − `destino_borda`, 0); topo `v3_topo_destino`
U[h_k; 0,55] pelo nível do BOTAR; a função devolve o ponto de soltura no mundo (E,3), que o F1
usa. A conta fica no núcleo puro `alvos.laje_destino` (§6.3; acréscimo ao `alvos.py` nesta
tarefa, com o jitter para longe somado ao `recuo` antes da chamada). Teste: θ = 0 e y = 0 põem
o centro da laje a recuo + 0,15 m e a borda perto a recuo − 0,15 m; θ = 10° põe o canto mais perto
a recuo − 0,15 m (± 1 mm), à mesma distância do caso θ = 0; o alvo fica a 0,15 m da borda perto, no
frame da laje, para todo θ e y. Commit: `feat(v3): laje de destino do BOTAR`.

**F3 — `terminacoes.py` (Sonnet).** `chegada`: no BOTAR, no primeiro contato caixa × laje depois da
abertura, a velocidade vertical do passo anterior (`v3_vz_caixa_ant`) acima do limite do degrau
(1,0 → 0,25 m/s). Teste: 1,1 m/s no degrau 0 termina; 0,9 não. Commit:
`feat(v3): terminação da chegada da caixa`.

**F4 — `recompensas.py` (Sonnet).** O `progresso` ganha os alvos B1–B4 com `fator_pouso` (σ pelo
degrau) e o invariante B2–B4 (rampa do deslocamento desde `v3_ponto_apoio`). Teste: pousar a 0,9 m/s
dá fator 0,44 no B2; deslocar 3 cm depois de apoiar leva a rampa a 0,5. Commit:
`feat(v3): pouso devagar e invariante depois do apoio`.

**F5 — `curriculo.py` (Haiku).** Níveis do BOTAR no sorteio do topo de destino; a abertura da C pelo
`v3_cur`. Commit: `feat(v3): níveis do BOTAR`.

**F6 — `metricas.py` (Haiku).** Velocidade de chegada; tempo por alvo B1–B4; deslocamento depois do
apoio. Commit: `feat(v3): métricas do BOTAR`.

**F7 — driver (Sonnet).** O BOTAR de um comando só no `pilota_v3` (o alvo de soltura no topo da laje);
roteiro C no `registra_juntas_v3`. Commit: `feat(v3): driver com a cadeia C`.

**F8 — smoke da Parte 3 (Sonnet).** Seções 25 a 29: 25. fatia C forçada: a laje de destino aparece na
abertura do BOTAR à frente do robô; 26. o alvo de soltura está no topo + a; 27. caixa largada de 5
cm sobre a laje não termina no degrau 0 (queda livre: 0,99 m/s < 1,0), e de 10 cm termina (1,40 m/s);
28. o one-hot fica em BOTAR durante o atraso; 29. o ANDAR parado final acaba em `time_out`.

### 12.3 Fase G — Parte 3b: CARREGAR no terreno

Atende: S §16 Parte 3b, §11.2 (TERRENO do CARREGAR), §12 (terreno); viabilidade D1, D2, D3, D4,
D10, L5. Enunciado: E §2 "Terreno", E §4.4.

Ondas: G0 → {G1, G2, G3, G4} → {G5, G6} → G7 (revisão).

**G1 — `eventos.py` (Sonnet).** Fatia B em episódio de terreno: a origem é o centro da plataforma de
2 m de uma coluna com plataforma; a laje fica a 0,75 m do centro, na direção de uma das 4 bordas; a
caixa sobre ela; o robô atrás da laje, de frente para a borda, a 0,5–0,65 m dela (viab. D10). Teste: o
núcleo põe o robô dentro da plataforma, a ≤ 0,65 m da borda e de frente para ela.

**G2 — `curriculo.py` e `balanceador.py` (Haiku).** A célula `terreno` do CARREGAR abre pela regra da
§6.3; a linha máxima por coluna pela PD-20.

**G3 — `comando.py` (Sonnet).** O sucesso do segmento do CARREGAR no terreno exige também a caixa a ≤
15° em todo passo (`v3_incl_max_seg`) e sem perda (S §11.3).

**G4 — `metricas.py` (Haiku).** `fracao_fora_da_plataforma` (viab. D10); nível de terreno por tarefa.

**G5 — smoke da Parte 3b (Sonnet).** Seções 30 a 33: 30. env da fatia B no terreno nasce na
plataforma; 31. o topo da laje = origem + h, com a origem de uma escada (+1,4 m); 32. o robô
teleportado um degrau abaixo, com a caixa no peito, não dispara a caixa perdida (viab. D4); 33. a altura da
pelve e do joelho medem contra o terreno local.

**G6 — driver (Sonnet).** Roteiro de escada e de rampa com caixa e de obstáculo no
`registra_juntas_v3` (a cena exportada já tem o terreno).

Commits: um por tarefa, prefixo `feat(v3):`.

### 12.4 Fase H — Parte 4: refino

Atende: S §16 Parte 4, §11.5, §10 (`NoBatente`), §12 (DR do refino), §13 (erro do estimador), §8.2
(`r_aperto`). Enunciado: E §3 "Sem batente", "Caixa íntegra e nivelada", "Movimento controlado".

Ondas: H0 → H1 → {H2, H3, H4, H5, H6} → H7 → H8 (revisão final). O H1 vem antes porque os outros leem
a `valor_do_degrau`.

**H1 — `curriculo.py` (Sonnet).** `valor_do_degrau(env, regra) -> (E,)`: o valor da regra da §6.9
para o degrau da tarefa corrente de cada env. Os termos e as terminações leem por ela. Teste: degrau
0 → início; 5 → fim; 3 → 60% do caminho.

**H2 — `recompensas.py` (Sonnet).**
- `sacudida`: custo da PD-34 com a aceleração linear e angular da caixa por diferença da velocidade
  (`Δv/dt`), fator d/5; máscara: caixa presa (P3 a B2 e o CARREGAR), tarefas PEGAR, CARREGAR, BOTAR.
- `aperto`: `custo_aperto(F_esq, F_dir, F_ref)`; F por face: cada contato do `caixa_faces` vai à face
  pelo argmax de |p_local|/a no frame da caixa (PD-29), e F_esq, F_dir são as somas das normais nas
  faces ±y; fator d/5; mesma máscara (S §8.2).
- Teste: `sacudida` no p50 do 23250 = 2; `aperto` com 117,5 N nas duas faces = 0,909; contato na face
  da frente não entra.

**H3 — `terminacoes.py` (Sonnet).** `batente`: cópia de `NoBatente` (`g1_limpo/terminacoes.py:
158-196`) com `frac_max` = 1,0 (o ângulo do batente da tabela, S §6.5) e |`hip_yaw`| > 1 rad; só com
degrau ≥ 1 da tarefa corrente. Teste: uma junta a 100% do meio curso no degrau 1 termina; no degrau 0
não.

**H4 — `eventos.py`, DR do refino (Sonnet).** Pelo degrau global, o menor entre as tarefas (PD-41), no reset: `dr.pd_gains` (escala 1 ± 0,1·d/5); `dr.effort_limits` (escala;
braços 0,8–1,1 no degrau 5); `solref` do chão e da caixa por evento próprio com
`@requires_model_fields("geom_solref")` (o mjlab não tem DR de `solref`; padrão de
`g1_limpo/eventos.py:256-301`); RFI por passo em `qfrc_applied` (0,05 → 0,1 × `tau_max`); força na
caixa no CARREGAR com `mdp.apply_body_impulse` (5–20 N, 0,1–0,3 s, a cada 3–8 s). A gravidade ±0,1
sai (PD-40). O atraso global sobe para 0–3 quando o degrau global chega a 3 (PD-41). O CoM da
caixa (`dr.body_com_offset`) fica atrás da chave `DR.com_caixa` = False (PD-38). Teste: os núcleos de
escala por degrau.

**H5 — estimador no refino (Sonnet).** A magnitude do `EstimadorCaixa` e da `Odometria` sobe do
"início" ao "refino" pelo degrau global, o menor entre as tarefas (PD-41; §6.2 `Estimador`), mais o viés por episódio (até ±2 cm na posição e ±2 cm na meia aresta; dono, 06/10 e 15/10) e
as rajadas de perda de 0,3–1,0 s (DR §3.5). A orientação erra por um viés por episódio sorteado da
distribuição medida (mediana 5,3°, p90 15,3°) mais 1° por leitura, no lugar do ruído independente. O
alvo do BOTAR leva viés por episódio de ±5 cm em xy e ±3 cm em z. Latência e taxa ficam em 33 ms e
30 Hz (dono, 15/10).

**H6 — métricas e log (Haiku).** Degrau por tarefa; std da ação por junta (S §14).

**H7 — smoke da Parte 4 (Sonnet).** Seções 34 a 38: 34. degrau forçado a 5: `laje_joelho_N` = 10,
chegada 0,25, σ do pouso 0,35; 35. `batente` só com degrau ≥ 1; 36. `pd_gains` dentro de 0,9–1,1 no
degrau 5; 37. `com_caixa` ligado por 200 passos em 16 mundos sem erro de heap (decide a PD-38); 38.
`save` → `load` com degraus e estado de DR.

**H8 — revisão final do código (Opus).** Revisão inteira contra o plano, a S e a I, antes da fase I.

---

## 13. Fase I — smoke, exportação e treino no RunPod

### 13.1 Antes do primeiro run

| passo | quem | comando ou ação | critério |
|---|---|---|---|
| I1 | dono | `cd ~/Documents/g1_training && .venv/bin/python -m g1_v3.smoke` | `N ok / 0 falhas`; cada falha volta ao orquestrador, que corrige e devolve |
| I2 | dono | `.venv/bin/python -m g1_v3.exporta_cena --saida ~/g1_pilota_v3/` e `--paridade --cena ~/g1_pilota_v3/` | paridade 1e−5 em todos os blocos |
| I3 | dono | `registra_juntas_v3.py` com a política aleatória, 10 s de ANDAR | o CSV sai e o robô não explode (sanidade da cena exportada) |
| I4 | dono | `git push origin g1_v3`; a imagem Docker com o contrato da D13 publicada; um pod no RunPod (4090 ou L40S) com o volume em `/workspace` e a porta 6006 exposta | o log do pod mostra o commit certo do clone, a GPU e as versões (mjlab 1.5.3, rsl-rl-lib 5.4.0) |
| I5 | dono | corridas de custo de 50 its, `RETOMA=0`: `RUN=v3_custo_sem PERCEPCAO=0` e `RUN=v3_custo_com PERCEPCAO=1`, com `NUM_ENVS=4096`; na L40S, também `RUN=v3_custo_8k NUM_ENVS=8192` | s/it (`Collection time` + `Learning time`) e memória da GPU de cada; o dono aceita o custo e fixa a GPU e o `NUM_ENVS` (fase 5c, PERC §9) |
| I6 | dono | `RUN=v3_custo_com RETOMA=1 ITERS=10` no mesmo pod | o log segue da it 50, o `RunnerV3` não avisa "checkpoint sem currículo" e o `Curriculum/*` continua de onde parou |

### 13.2 O treino

Uma corrida só, do zero, com o currículo inteiro: `RUN=v3_zero01 RETOMA=0 ITERS=30000`, com a GPU e
o `NUM_ENVS` do I5. O currículo começa com 100% de ANDAR (E §1 "Ordem de aprendizado"). Se o pod
para, a mesma corrida segue com `RETOMA=1` e `ITERS` = as its que faltam. O checkpoint sai a cada
`save_interval` (50 its) em `$SAIDA/g1_v3/<data>_v3_zero01/`, no volume do pod; copiar para fora do
pod fica com o dono até o destino final ser decidido (PD-22).

### 13.3 Portões, na ordem em que o currículo abre

| marco | o que conferir | onde | critério (S §15) |
|---|---|---|---|
| ~1 000 its | log | TensorBoard | `razao_marcha` ≥ 0,50 com o atraso ligado; nenhuma terminação nova acima de 5% |
| células `giro_lugar` e `baixo` do ANDAR no portão | play | `registra_juntas_v3` | "comando baixo" e "rumo" (laço k = 2 no driver) |
| fatia B aberta | play | idem | "espera sem agachar", "pega reta" |
| fatia C e BOTAR nível 0 no portão | play | idem | "BOTAR de um comando só" |
| `terreno` do ANDAR e do CARREGAR | play | idem | "escada sem caixa", "escada e rampa com caixa", "obstáculo" |
| refino de uma tarefa | log | TensorBoard | sucesso da tarefa não cai abaixo do portão por mais de 300 its |
| fim do treino | 10 plays por cenário | `registra_juntas_v3` | "sem batente", "sem toque na laje", "BOTAR na laje de 0,04 m", "giro de 180° com a caixa", "empurrão com a caixa", "frear com a caixa", "passo com a caixa", "caixa perdida", 9 de 10 |

Um portão que falha para o treino: diagnóstico pelo vault (`vault/_sistema/por-termo.md`),
conserto, e resume ou recomeço. A decisão é do dono.

---

## 14. Estratégia de testes

| camada | o que | quem escreve | quem roda | quando |
|---|---|---|---|---|
| núcleos puros | `unittest` em `g1_v3/testes/`, com os números deste plano | lane de testes (Haiku), antes do código | agentes e orquestrador | depois de cada lane |
| montagem do cfg | `test_cfg.py` constrói `make_env_cfg()` e confere nomes, ordem, pesos, máscaras, sensores | lane de testes | agentes | fases D a H |
| scripts da Parte 0 | `g1_v3/parte0/item*.py` | Sonnet | agentes | fase B |
| integração com o simulador | `g1_v3/smoke.py`, 38 seções, CPU | Sonnet, ao longo das fases | dono | fase I, uma vez (pedido do dono) |
| paridade treino ↔ driver | `exporta_cena --paridade` | Sonnet | dono | fase I |
| comportamento | plays com `registra_juntas_v3` contra a S §15 | — | dono | fase I, nos marcos da §13.3 |

Regra: um teste que o implementador muda para passar volta ao orquestrador, que decide se o plano
ou o código está errado. O plano só muda com nota na §20.

Comando único: `cd ~/Documents/g1_training && .venv/bin/python -m unittest discover -s g1_v3/testes -t .`

---

## 15. Desempenho

| fator | número | fonte | o que o plano faz |
|---|---|---|---|
| s/it do `g1_limpo` na Kaggle (T4, 4 096 envs) | 6,9–7,0 | notebooks | só referência; a GPU do RunPod é outra e o I5 mede de novo (PD-22) |
| terreno no modelo único (1 410 geoms) | ×1,8 a 2,1 no `env.step` em CPU | viabilidade C7, L1 | aceito pelo dono (terreno desde 06/10); com R = 13 (PD-20), 117 ladrilhos contra 90 (+30%); I5 mede na GPU |
| raycast | 205 raios por passo (PD-5), contra 1 024 por passo | viabilidade D7: 2 025 raios = 10–14% do passo em CPU | ~1/5 do custo de 1 024 por passo |
| filtro de elevação | 51 × 51 células × 4 096 envs; 205 pontos por passo | viab. D7: 26 ms com P 256 em CPU | `custo_filtro.py` mede em CPU; I5 na GPU |
| filtro do corpo | 205 pontos × ~40 cápsulas × 4 096 envs por passo | — | uma operação vetorizada; I5 mede |
| primeira camada do ator | 1 447 × 512 (contra 565 × 512) | S §19.1 | sem mudança de rede (fase 5c decide) |
| orçamento | 30 000 its ([F34]) | S §1 | na T4, 12–15 s/it dariam ~100–125 h; no RunPod, tempo e custo saem do s/it do I5 |

Risco de custo: o tempo e o custo dos 30 000 its dependem do s/it na GPU escolhida (4090 ou
L40S), medido no I5. O orçamento por GPU fica para depois do plano (PD-22). Um pod que para no
meio não perde a corrida: o checkpoint fica no volume e o `RETOMA=1` segue dele (I6).

---

## 16. Contagem de código (CLAUDE.md: antes → depois)

Comparável com a S §17: sem o smoke, sem as ferramentas (`inspeciona`, `leitura`, `exporta_cena`,
`paridade`) e sem o driver.

| módulo | `g1_limpo` | v3 (estimativa no fim da fase H) |
|---|---|---|
| `knobs.py` | 1 628 | 700 |
| `cena.py` | 482 | 450 |
| `observacoes.py` | 141 | 350 |
| `comando.py` | 2 526 | 700 |
| `curriculo.py` | 537 | 250 |
| `recompensas.py` | 1 390 | 900 |
| `terminacoes.py` | 196 | 200 |
| `eventos.py` | 550 | 500 |
| `metricas.py` | 590 | 350 |
| `algoritmo.py` + `runner.py` | 372 | 200 |
| `env_cfg.py` + `__init__.py` + `tarefa.py` + `train.py` + `play.py` | 1 169 | 600 |
| `alvos.py`, `potencial.py`, `balanceador.py` (novos) | — | 700 |
| `estimador.py`, `percepcao_lidar.py` (novos) | — | 665 |
| **total** | **9 581** | **~6 550** (−32%) |

A meta de ~5 000 da S §17 é de 02/10 e não conta o que entrou depois: o filtro de elevação (~350), o
estimador com a visibilidade (~300), as sete cercas (~200) e a observação de 1 447 canais (~150).
Sem eles a conta dá ~5 550. A S §17 manda refazer a conta na Parte 1; esta é a conta refeita.

---

## 17. Riscos

| risco | onde | o que o plano faz |
|---|---|---|
| `_local_directions` é atributo privado do mjlab | fase 5a | seção 5 do smoke; se falhar, o padrão vira um sensor com o bloco inteiro e um índice por passo (PERC §5), decisão do dono |
| `max_lag` do `DelayBuffer` mudado em tempo de execução | PD-28 | seções 8 e 24 do smoke |
| `dr.body_com_offset` corrompe a heap | PD-38 | chave desligada; seção 37 do smoke |
| o `comando.py` concentra a lógica da cadeia | fases D, E, F | núcleos puros testados (`avanca_trecho`, avanço de alvo, caixa perdida); revisão do Opus por Parte. Lição da F4 do `g1_limpo`: 80% do trabalho num arquivo deu sete defeitos, cinco silenciosos (memória `feedback-subagentes`) |
| o detector de fim nunca avisa | PEGAR e BOTAR | métrica da fração de avisos (E7); o episódio acaba em `time_out` e conta falha |
| os números da Parte 0 não fecham | fase B | o portão da B6 para o plano; o dono decide |
| o smoke só no fim junta muitos defeitos | fase I | o `test_cfg` constrói o cfg em toda fase; reservar uma rodada de correção na fase I |
| a fluência da caixa entre as palmas (4–8 cm em 6 s) | Partes 2 a 4 | aceita (L2, 14/10): a política vê a caixa e corrige |
| o custo de GPU no RunPod e a parada do pod | fase I | §15; I5 mede o s/it; I6 prova a retomada |
| a ordem das juntas do MJCF diferir da do SDK | contrato | `test_knobs` e seção 4 do smoke |
| o mapa nasce frio no reset | B e C | aceito; a espera de 0,5–1,0 s enche a frente (sanidade 4.4.1) |

---

## 18. Despacho

Modelo do prompt de cada tarefa (o orquestrador preenche os campos entre `<>`):

```
Você implementa a tarefa <id> do plano docs/planos/2026-10-14-implementacao-g1-v3.md, no repo
~/Documents/g1_training, branch g1_v3.

Leia antes de escrever: a seção da tarefa, a §1 (regras), a §6 (interfaces) e os arquivos de
origem citados na tarefa. Leia também specs/g1-v3.md nas seções citadas.

Regras:
- Toque só nestes arquivos: <lista>.
- Não rode git. Não rode o simulador, o smoke, o viewer nem a IK.
- Rode: .venv/bin/python -m unittest discover -s g1_v3/testes -t .
- Não mude teste para passar. Um teste que parece errado volta para mim, com o motivo.
- Uma interface que falta na §6 volta para mim; não invente nome nem campo.
- Comentários em português, no estilo do g1_limpo (o porquê, ⚠ nos pontos caros, a seção da spec).
- Mostre a contagem de linhas antes → depois de cada arquivo.

Entregue: os arquivos, a saída dos testes, a contagem e a lista do que ficou de fora.
```

O orquestrador, depois de cada lane: roda os testes, confere `git status` contra o snapshot (só os
arquivos da lane), faz o commit por caminho explícito, confere que não há `Co-Authored-By`.

---

## 19. Referências

- Spec: `specs/g1-v3.md` (S), `specs/g1-v3-comportamento.md` (CV), `specs/g1-v3-interface-real.md`
  (I), `specs/g1-limpo-comportamento.md` (E).
- Planos: `docs/planos/2026-10-02-percepcao-lidar-na-v3.md` (PERC).
- Relatórios: `docs/relatorios/2026-10-05-dr-para-a-v3.md` (DR), `2026-10-06-v3-sanidade-e-plano-
  de-treino.md`, `2026-10-08-revisao-spec-v3-para-implementacao.md`, `2026-10-09-viabilidade-v3.md`.
- Vault: `vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md`; `vault/_sistema/inbox.md`
  (14/10).
- Código de origem: `g1_limpo/`, `pilota.py`, `registra_juntas.py`, `lidar_sim/sensores/`.
- mjlab 1.5.3: `.venv/lib/python3.12/site-packages/mjlab/` (APIs da §2.3).

---

## 20. Mudanças neste plano

| data | mudança | motivo |
|---|---|---|
| 2026-10-14 | versão inicial | pedido do dono |
| 2026-10-14 | revisão inteira do plano: 25 correções (PD-35 a PD-44, alturas de queda livre, teste do mapa parado, referência macro por alvo, dono único da odometria, média por célula, módulos puros sem import entre si, eventos e buffers que faltavam no contrato, tabelas quebradas) | `docs/relatorios/2026-10-14-plano-v3-achados.md` |
| 2026-10-15 | cena, três decisões do dono depois da validação de viabilidade: cerca de torque pela soma nos pares acoplados da cintura e dos tornozelos (PD-45); atrito por material, com a caixa em `priority` 0 e o DR de μ nos pads e na laje (PD-46); laje de destino com o alvo a 0,15 m da borda perto, ±0,10 m de lado e giro de ±10° (PD-47). Corrige também o alvo no centro da laje de 0,60 m, que punha a borda perto dentro do corpo [C38]. Contagem: ~+20 linhas no pacote (`cena.py`, `eventos.py`, `alvos.py`, `recompensas.py`), nenhum termo novo | `docs/relatorios/2026-10-15-viabilidade-v3.md` §2.2 e §6.1; respostas do dono em 15/10 |
| 2026-10-15 | currículo e recompensa, quatro decisões do dono depois da validação de viabilidade: âncora de 1,5 s no sucesso do canal zero (PD-15); `passo` ao quadrado por toque (PD-48); σ próprio no canal zero do rastreio, 0,10 m/s e 0,15 rad/s (PD-49); `não v3_perdeu` em todo sucesso do PEGAR, do CARREGAR e do BOTAR (PD-50). Contagem: ~+14 linhas no pacote, nenhum termo novo, nenhum peso muda | `docs/relatorios/2026-10-15-viabilidade-v3.md` §2.1, itens 1 a 4; respostas do dono em 15/10 |
| 2026-10-22 | estimador e mapa, decisões do dono de 15/10 e 22/10: máscara do lidar por raio de 0,7 m em volta do sensor, sem a caixa estimada (PD-10); FOV do RGB na visibilidade (PD-11); viés de tamanho, orientação persistente, alvo do BOTAR com viés e alvo do PEGAR pela caixa estimada (H5, E); latência fica em 33 ms a 30 Hz. O rastreio da caixa por Kalman (câmera + lidar + mãos) segue em desenho, fora do plano | discussão de 15/10 a 22/10; `vault/_sistema/inbox.md` |
| 2026-10-22 | revisão das PD pelo dono: PD-3 aceita; PD-22 refeita para Docker no RunPod, clonando a branch; os detalhes de treino ficam para depois da implementação | respostas do dono em 22/10 |
| 2026-10-22 | PD-5 e PD-12 aceitas pelo dono | respostas do dono em 22/10 |
| 2026-10-22 | PD-30 a PD-34 aceitas pelo dono, com as margens propostas (50%; 1,5; 1,5; 1,2; 10%) | respostas do dono em 22/10 |
| 2026-10-22 | PD-38 e PD-40 aceitas; PD-41 refeita para um degrau global, o menor entre as tarefas, para toda a DR e o atraso. Fecha a revisão das PD | respostas do dono em 22/10 |
| 2026-10-22 | catraca do freio e do refino: o dono mantém 'nada desce' (opção A); um travamento depois de um degrau se detecta no log (s < 0,60 por 300 its depois de um degrau) e se trata à mão | resposta do dono em 22/10; viabilidade §2.1, item 6 |
| 2026-10-22 | fatos do robô real (dono): mão = pad de palma, 29 DoF; limite da cintura e do tornozelo = 50 N·m por eixo (o treino segue em 50, com a cerca pela soma do par, PD-45); envelope de produto = lajes de 0,04 a 0,55 m e cubos de 14 a 26 cm até 5 kg, igual ao do treino. Nada muda no plano | respostas do dono em 22/10 |
| 2026-10-22 | canal da caixa (dono): opção A, a política recebe a leitura da câmera (estimador simulado no treino; D435i + box-vision no real), como em 14/10. O rastreio por Kalman (câmera + lidar + mãos) fica como extensão depois do primeiro treino, sem mudar o formato dos 11 canais. Plano pronto para o despacho pela fase A | resposta do dono em 22/10 |
| 2026-10-28 | percepção treino × real (dono, 28/10): `raio_min` 0,85 m (o braço + 5 cm, medido por FK) e o mesmo `filtra_corpo` na entrada da odometria real (PD-10); a variância da medida por célula ganha o erro comum da leitura e o piso `sigma_fixo` 2 cm, e o mapa gira pelo mesmo viés do IMU da gravidade, para a confiança não depender da densidade de pontos (PD-42, PD-6; teste de invariância na C0); o driver real faz deskew por ponto e entrega os pontos a cada 20 ms (PD-5); o `mapa` entra no atraso por episódio (PD-25, §6.8); a odometria ganha os termos de giro (PD-18, D2b); contrato do detector na I §2.3. A decimação de pontos no driver foi descartada | `docs/relatorios/2026-10-22-percepcao-sim-real.md` §5 |
| 2026-10-28 | plataforma (dono, 28/10): a PD-22 chega às tarefas. A D13 deixa de ser o notebook da Kaggle e vira `g1_v3/container/entrada.sh`, o ponto de entrada que a imagem do RunPod chama depois do clone (variáveis `RUN`, `RETOMA`, `ITERS`, `NUM_ENVS`, `PERCEPCAO`, `SAIDA`, `TENSORBOARD`, `SECO`; checagem de GPU e de versões; retomada pela pasta mais nova com checkpoint), com o `test_entrada` em modo seco. A D10 registra `Mjlab-G1-V3-SemPercepcao` para o I5. A fase I troca I4 e I5 pelo pod, mede 4 096 e 8 192 envs e ganha o I6 (retomada). §0, §1.1, §2.1, §3, §6.1, §15 e §17 deixam a Kaggle | pedido do dono depois da checagem de alinhamento |
| 2026-10-29 | revisão do plano, 14 correções (o dono aprovou; a 12, tudo em Sonnet, ficou fora: a implementação segue com agentes Haiku). (1) transporte vertical do mapa pelo `z_ref`, o pé mais baixo no frame da odometria (PD-7, C0, C1, D2, D11; I §2.4 e PERC §0). (2) `atualiza_percepcao` com a guarda no env: o filtro e o estimador rodam uma vez por passo, e não uma por grupo (D2, E8, §6.4). (3) R = 13 e `linha_max_carregar` inteira: o teto do CARREGAR chega a 0,15 m e 0,27 (PD-20, D1, D7, B3, §6.2, §15). (4) atraso da observação com capacidade 0–3 desde o build (D2, D7, §2.3, §6.5). (5) máscara bool com `torch.where` e nenhum NaN como "não usado" (§6.3, §6.6, B0, D4). (6) odometria integrada no frame da base, com `delta_b()` (§6.3, D2b). (7) `pontos_w` para o `filtra_corpo` e `dist` para o d̄ (§6.3, C1). (8) PD-25 no lugar da PD-14, sem a PD-15 no `tarefa_critico`, prefixo "viab." nos itens da viabilidade. (9) sai o `raio_caixa_na_mao`. (10) estimador e DR do refino pelo degrau global (§6.9, H5). (11) lista dos CSV da Parte 0 pela coluna `fase` (B5). (13) `passos_por_iteracao` no `knobs`. (14) o `Curriculo` conta as próprias iterações, e a retomada não trava os níveis (§6.3, B0, D4). (15) `random_rough` com `scale_with_difficulty` (D1). Contagem: ~+25 linhas no pacote (`observacoes.py` +10, `estimador.py` +10, `percepcao_lidar.py` +3, `balanceador.py` +2), nenhum termo novo | `docs/relatorios/2026-10-29-revisao-plano-v3.md`; resposta do dono em 29/10 |
| 2026-10-29 | contrato do `DR` e do `Refino` na §6.2 (a A2 achou só "ver §6.8/§6.9", sem nomes de campo). Os nomes seguem os eventos da D7 e da H4 e a chave `DR.com_caixa` (PD-38); o `Refino` aponta para os pares que já moram em `Recompensa`, `Alvos` e `Terminacao`, e o `laje_joelho_N` vira par (50; 10). O tamanho da caixa fica só na `Cena`. O `jitter_z` sai da linha da `Cena`: não existe no `g1_limpo`, e o `prateleira_jitter_z` já está em `prateleira_*`. Os campos do `Knobs` são as seções em minúsculo (`contrato`, `cena`, ..., `dr`, `refino`). Contagem: ~+45 linhas no `knobs.py` | orquestrador, durante a fase A |
| 2026-10-29 | contrato da §6.3 fechado para as lanes B0 e C0: convenções (wxyz, `ids` long, `gerador` None sem ruído, `dt` float); `Curriculo.celulas` público com os ids 0..11 na ordem da B3, `it` público, a regra da EMA (s = x na primeira medida, s começa em 0) e o `estado()` só com tipos do Python; formas de `pontos_no_frame_do_mapa`, `filtra_corpo` (cápsulas (E,K,2,3), como o `visivel`), `prediz` e `atualiza`; os giros de inclinação agem em volta do sensor. Nenhuma função nova | perguntas das lanes B0 e C0 |
| 2026-10-29 | nível de terreno no `Curriculo` (a B3 pedia 0..12 dentro da célula `terreno`, sem interface na §6.3): `k_ter` no construtor e `nivel_terreno(tarefa)`; teto 12 no ANDAR e 9 no CARREGAR; o refino do ANDAR e do CARREGAR espera o último nível de terreno; "seguidas" zera a contagem | pergunta da lane B3 |
| 2026-10-29 | unidade da Parte 0 no cabeçalho da B5: o retorno do mjlab (`scale_by_dt`); custo por s = w·média(x); `B_alvo` que cobre o custo C de um trecho = 1,5·C/dt. O `custos_por_s` dividia pela duração e dava 50 vezes o custo por s | item 6 da B5 |
| 2026-10-29 | transporte do mapa (C1) e critério da invariância (C0), decisões do dono: o mapa anda em células inteiras e guarda o resto e o rumo, com o giro no `recorte` (a célula mais perto a cada passo congelava o mapa a 50 Hz); o teste de densidade compara média ≤ 1 cm e p95 ≤ 3 cm fora da borda do degrau, e |Δc| ≤ 0,03 em 90% das células. Achado: na borda lateral da vista a confiança do caso de 205 pontos fica menor (lê a célula menos vezes); a frase da PD-5 "a densidade não muda a confiança" não vale ali, e o erro é conservador. Contagem: `percepcao_lidar.py` +10 linhas; custo +7% | diagnóstico de 29/10 (`test_recorte_com_2_leituras_nao_depende_da_densidade`) |
| 2026-10-29 | portão da Parte 0, decisões do dono (`docs/relatorios/2026-10-29-v3-parte0.md` §3): D1 o `progresso` usa o Φ do alvo velho no passo do avanço; D2 o `L_alvo` fica, e o critério do item 8 não vale nos plays de CARREGAR (passo miúdo); D3 `fator_pouso` gradual de 0,10 a 0,05 m (`altura_pouso_inicio`); D4 cercas pela média só nos passos em que agem (PD-30); D5 o custo do filtro fica para o I5, em GPU; D6 o teto do `aperto` sai da perda sem o aperto evitado; D7 `sacudida` = 0,05 (spec); D8 ganho de afrouxar numa rampa de 20 s. Contagem: `potencial.py` +2, `knobs.py` +1; um campo novo (`altura_pouso_inicio`), nenhum termo novo | portão da B6 |
| 2026-10-29 | núcleo do `eventos` na §6.3 (`escolhe_celula_terreno`) e a tabela das colunas no `knobs` (`Terreno.colunas`; `linha_max_carregar` por coluna, no mesmo formato): o teste puro não importa a `cena`, que importa o mjlab. Contagem: `knobs.py` +2 | pergunta da lane D0-3 |
| 2026-10-29 | núcleos puros do `comando` e do `observacoes` na §6.3 (`sorteia_velocidade`, `TabelaTrechos`, `monta_tabela_trechos`, `avanca_trecho`, `tempo_da_ancora`, `sucesso_do_segmento`, `passo_de_percepcao`, `_calcula_percepcao`, `referencia_do_alvo`) e o buffer `v3_ref` na §6.4; o `atualiza_percepcao` vira só a guarda. Nenhum termo novo | perguntas da lane D0-2 |
| 2026-10-29 | núcleos da D5 e da D6 (`n_<termo>`, `comprimento_do_passo`, `n_queda`, `n_fim_da_cadeia`), `spec_caixa`/`spec_laje`/`terreno_cfg` da cena, "todas" = as 4 tarefas da cadeia, o conjunto de termos da fase D, a fórmula do teto da coluna `caixas` (0,2 + 0,8·t) e a API de registro, na §6.3 | perguntas da lane D0-1 |
| 2026-10-29 | §6.4: `v3_gerador` (o gerador dos sorteios e do ruído, com a semente do cfg) e `v3_iter`; o `garante_v3` cria a `v3_odo` com o `v3_gerador`; o `v3_freio` e o `v3_degrau` ficam com o `comando.cadeia`, como a §6.4 já dizia, e a D4 deixa de escrevê-los | pergunta da lane D4 |
| 2026-10-29 | D8: sai o `atraso_medio` (atributo privado do mjlab; o `atraso_max` vai ao log pela E6) e o `momento_angular` (pede sensor novo); o limiar do mapa visto vira `Percepcao.limiar_visto` = 0,5 | perguntas da lane D8 |
| 2026-10-29 | dúvidas da onda D-2: o `twist` é o único escritor de `v3_celula` e lê o episódio de terreno no `terrain_types` (`sorteia_celulas` ganha `sem_tipo`); a unidade do `v3_t_empurrao` e o relógio do `twist`; o `tamanho_caixa` vai à fase E; `grupos(k: Knobs)` (o `mapa` sai sem a percepção; ruído e capacidade do atraso no `knobs`); a composição de `caixa_verdade` e o divisor de `niveis`; o `cerca_com` usa o `subtree_com` e o `body_subtreemass` da raiz; o recuo é a distância no plano da `grade_recuo`; campos novos `Cena.reset_base_andar`, `Cena.prioridade_laje`, `Terreno.borda`, `DR.ruido_obs`, `Recompensa.recuo_carregar` | as lanes D2, D3, D5 e D7 acharam a §6 aberta; o `body_mass` tem forma (mundos, corpos) e o `root_com_pos_w` é o CoM da pelve, não o do robô |
| 2026-10-29 | `Curriculo`: os sorteios rodam no device do gerador; `id_celula(tarefa, tipo)` substitui os dois `_id_terreno` repetidos do `eventos` e do `comando` | o `v3_gerador` mora na GPU e o `multinomial` da CPU com gerador CUDA falha; função repetida (CLAUDE.md, minimização) |
| 2026-10-29 | D11: a seção "Contrato com o pilota_v3" do `exporta_cena.py` é o contrato do driver; o mapa do driver sai da `Percepcao` do `pilota_v3`, e a paridade só imprime a diferença do mapa; `T_BASE_SENSOR` é referência, o driver usa a pose viva do site; versão = hash do git; kd conferido por `biasprm[2]` | dúvidas da D11a: o plano não dizia como o mapa entra na paridade nem o que é a versão do pacote |
