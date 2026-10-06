# Inventário da v3 — o que fica, o que sai

Data: 01/10/2026. Pedido do dono: "pegar tudo que aprendemos até agora que funcionou, revisar e
aplicar em um treino refinado muito mais simples, somente o que funcionou e que é estritamente
necessário". Este inventário é a base da spec da v3. As decisões da v3 estão em
`vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md`.

## Método

- 11 agentes de leitura extraíram ~720 lições de 115 memórias, 32 documentos de
  `docs/memoria/`, 16 planos, 5 relatórios, 25 specs e o `g1_limpo/ARQUITETURA.md`.
- O revisor deduplicou, conferiu na fonte os números que decidem e fixou a classe final.
- Cada item cita a fonte original. `mem:` é uma memória em `~/.claude/memory/`; os outros
  caminhos são do repo. `§5` e `§6` são do enunciado (`specs/g1-limpo-comportamento.md`).

## Resumo

| classe | itens | o que a v3 faz |
|---|---:|---|
| Fato do simulador, do framework ou do robô | 37 | obedece |
| Conserto validado | 49 | leva |
| Remendo de economia | 15 | elimina pelo desenho |
| Rejeitado | 27 | não refaz |
| Processo | 16 | segue |
| Aberto | 26 | a spec decide |
| Contradição | 9 | resolvida aqui |

Três conclusões:

1. **O que funcionou é quase todo geometria, física e medição, e não peso.** Alvo dentro do
   alcance, σ = distância inicial, eixo de cima da caixa, distância à superfície, força de
   apoio em z, preensão contínua, a multa em rampa no contato com a mesa. Isso passa à v3 sem mudança.
2. **Quase todo remendo tem a mesma raiz: a renda paga por estado.** Pairar, estátua, pressa
   para fechar, cauda sem derivada, desvio pago. O potencial da v3 tira a raiz.
3. **Os preços que o enunciado exige nunca foram treinados.** Impacto, aperto, sacudida,
   contato com a laje abaixo de 50 N, mão única. A spec da v3 decide cada um (seção 6).

## 1. Fatos — a v3 obedece

### 1.1 mjlab e rsl_rl

| # | fato | prova | fonte |
|---|---|---|---|
| F1 | O `feet_swing_height` do mjlab não tem `reset`: o pico do pé vaza entre episódios e infla com a queda. A v3 usa a subclasse com `reset`. | MEDIDO | mem:g1-mjlab-velocity-anatomia; ARQUITETURA:1534 |
| F2 | Recompensa = bruto × peso × dt, com dt 0,02. A terminação −200 custa −4 por evento. | MEDIDO | mem:g1-mjlab-velocity-anatomia; ARQUITETURA:814 |
| F3 | `Episode_Reward/<termo>` divide pela duração máxima do episódio. Episódio curto lê diluído; `Mean reward` não divide. | MEDIDO | mem:g1-leitura-log-e-kaggle; specs/g1-limpo.md:661 |
| F4 | O currículo `command_vel` reescreve a faixa a cada tique, com degraus nas its 5000 e 10000. O cfg lido antes do launch mostra a faixa base. | MEDIDO | mem:mjlab-curriculo-de-comando-sobrescreve-a-faixa |
| F5 | O sorteador de twist do molde reamostra a cada 3–8 s e põe 10% em standing. Ele não produz giro parado sem ramo próprio. | MEDIDO | mem:g1-mjlab-velocity-anatomia |
| F6 | A única falha do molde é `bad_orientation` a 70°. O rsl_rl só faz bootstrap `+γV` em `time_out`, então terminação nova é punição forte. | MEDIDO | mem:g1-mjlab-mutacao-em-treino |
| F7 | O normalizador do rsl_rl não tem clamp: canal constante ou raro entra a ~100×. Medido: CARREGAR 94,8×, REORIENTAR 24,9×. | MEDIDO | mem:g1-limpo-one-hot-normalizado-amplifica-slot-raro |
| F8 | O `std` do rsl_rl é um vetor de 29 valores, igual em todo estado. A queda dele é mecânica do PPO, sem piso e com entropia 0,01. | MEDIDO | mem:g1-limpo-std-da-acao-por-junta; mem:g1-limpo-std-cai-cedo |
| F9 | `max_iterations` é aditivo no resume. `optimizer.load_state_dict` não confere shape. | MEDIDO | vault/G1/Transversais/Infra/g1-leitura-log-e-kaggle.md; mem:g1-mjlab-warmstart-obs-armadilhas |
| F10 | `CommandManager` e `ActionManager` não fazem deepcopy do cfg; nos outros managers só `get_term_cfg` muda o cfg depois da construção. O currículo roda no reset dos envs que resetam. | MEDIDO | mem:g1-mjlab-mutacao-em-treino |
| F11 | Entidade fixed-base vira mocap e, sem evento de reset, fica na origem do mundo. Dentro do evento de reset, `Entity.data` está obsoleto. | MEDIDO | mem:g1-mjlab-api-armadilhas; mem:g1-mjlab-reset-buffers-obsoletos |
| F12 | Pose escrita em corpo livre persiste. Corpo pinado não gera força de peso, só de penetração. | MEDIDO | mem:mjlab-pose-escrita-persiste |
| F13 | `root_com_pos_w` é a pelve. O CoM real é `subtree_com` do corpo raiz. | MEDIDO | mem:g1-mjlab-api-armadilhas |
| F14 | `base_com` corrompe a heap em CPU e GPU. `dr.body_mass` e `dr.pseudo_inertia` foram consertados no mjlab 1.5.3. | MEDIDO | ARQUITETURA:306; mem:g1-limpo-inercia-real-da-caixa |
| F15 | O G1 fica de pé sem aprender nada (`use_default_offset` + PD). Um `model_0` de pé não prova nada. | MEDIDO | mem:g1-de-pe-e-gratis-no-mjlab |
| F16 | O kernel de rastreio linear do molde (std 0,5) paga 27% do teto ao robô imóvel. | MEDIDO | mem:g1-track-lin-parado-colhe-27pct |

### 1.2 MuJoCo e física da cena

| # | fato | prova | fonte |
|---|---|---|---|
| F17 | Cone elíptico com `impratio` 10 divergiu para NaN duas vezes. Pyramidal com `impratio` 1,0 roda. | MEDIDO | mem:g1-limpo-cone-eliptico-diverge-para-nan |
| F18 | O limite de junta do MJCF é restrição mole: `ankle_roll` sai 150% do curso e `ankle_pitch` 120%, empurrados pelo contato. | MEDIDO | mem:g1-limpo-batente-e-restricao-mole |
| F19 | O apoio rígido faz a força da laje ir de 1·m·g a 0 em 2 mm de subida. Termo lido da força de apoio satura em milímetros. | MEDIDO | vault/G1/Transversais/Infra/g1-leitura-log-e-kaggle.md; auditoria de recompensa, achado 4 |
| F20 | Os pads da palma têm `condim` 3, sem atrito de torção. O `CollisionCfg` sobrescreve `condim` 4. | MEDIDO | vault/G1/Transversais/Infra/g1-auditoria-mjlab-bloco3.md; relatório do tombo §3-G |
| F21 | O MuJoCo clássico roda 96× mais rápido que o Warp em CPU: 0,065 contra 6,229 ms por subpasso. | MEDIDO | specs/g1-limpo-pilota.md |

### 1.3 Robô G1

| # | fato | prova | fonte |
|---|---|---|---|
| F22 | O `g1.xml` do mjlab é o URDF rev 1.0 exato: 29 corpos, Δ 0 mm, 33,341 kg. | MEDIDO | mem:g1-mjcf-vem-do-urdf-rev-1-0 |
| F23 | A escala da ação do punho é 0,075 rad por unidade, 6× menor que a do braço. O ruído de exploração é 2° no punho e 12° no cotovelo. O dono decidiu não mudar a escala. | MEDIDO, DONO | mem:g1-limpo-escala-da-acao-do-punho |
| F24 | Ombro, cotovelo e punho giram no mesmo eixo, palma a palma. Erguer com o braço gira a caixa ~1,6 rad por metro, e só o punho desfaz o giro. | MEDIDO | relatórios 2026-09-29 do tombo e da pega parada |
| F25 | Postura de pé por IK: pelve 0,787–0,789 m nivelada; joelho 0,15 sem caixa e 0,20–0,42 com 1–5 kg. | MEDIDO | specs/g1-limpo-postura-de-pe.md |
| F26 | Braço: ombro–cotovelo 0,198 m, cotovelo–palma 0,264 m. A IK usa 0,41–0,46 m de ombro a palma. | MEDIDO | relatório da pega parada §2; mem:g1-limpo-forma-postural |
| F27 | No robô real os atuadores são torque puro. Junta e atuador vão por nome. | MEDIDO | mem:unitree-g1-sim-mujoco-not-isaac |
| F28 | A IK alcança toda laje de 0,05 a 0,55 m com pés planos, CoM sobre os pés e 5 kg, com margem de 0,25 a 0,35 m. O BOTAR ruim não é falta de alcance. | MEDIDO | mem:g1-limpo-botar-e-bang-bang-nao-pressa |

### 1.4 Treino e infraestrutura

| # | fato | prova | fonte |
|---|---|---|---|
| F29 | O PPO aprende de passos. A fatia de dados é a fração de passos: 30% dos episódios com 24 passos deram 1,1% do gradiente. | MEDIDO | mem:g1-locomocao-primeiro-fatia-100 |
| F30 | O crítico esquece o estado que a política deixa de visitar. No BOTAR, V(alvo) − V(congelado) caiu de +15 a −8,7 em 1000 its. | MEDIDO | mem:g1-limpo-botar-esquecido-na-fase-4 |
| F31 | Uma tarefa seguinte não treinada mata o fecho da anterior: 28 de 28 fechos forçados do PEGAR terminaram em até 1,6 s. | MEDIDO | mem:g1-limpo-fechar-o-pegar-mata-o-episodio |
| F32 | A política é uma MLP sem memória (`history_length` None). | MEDIDO | docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md §7.2 |
| F33 | O gargalo é a física (89% do tempo) e a banda de memória da GPU. O Kaggle T4 roda 4096 envs; o Colab, 8192. | MEDIDO | mem:g1-treino-gargalo-banda-nao-flops; mem:g1-colab-8192-kaggle-4096 |
| F34 | O orçamento do PPO do molde é 30 000 its. Um treino de 5000 its termina no primeiro degrau de comando. | MEDIDO | mem:g1-poc-orcamento-30k-iteracoes |
| F35 | Parado, a manipulação move as juntas mais rápido que a marcha: p90 3,13 contra 0,43 rad/s. | MEDIDO | specs/g1-limpo-lento-e-estavel.md |
| F36 | A mão chegou a 2,58 m/s na pega, num pulso de 0,62 s. A ISO/TS 15066 usa 0,25 m/s. | MEDIDO | docs/planos/2026-09-23-freio-em-curriculo-pelo-nivel.md §1 |
| F37 | O choque da caixa na laje dura menos que um passo de controle (20 ms). A força lida uma vez por passo pega o pico ou não, conforme a fase: queda de 5 cm lê 2,5 × m·g, de 8 cm lê 9,0, de 10 cm lê 5,5; o pico real por subpasso é 11,8, 16,1 e 17,8. A métrica `impacto_da_caixa` não diz de que altura a caixa caiu. | MEDIDO 01/10, teste de queda na cena | relatório; decisões da v3, item 20 |

## 2. Consertos validados — a v3 leva

### 2.1 Locomoção

| # | conserto | prova | fonte |
|---|---|---|---|
| C1 | Locomoção primeiro, com 100% dos envs até o portão. É o único mecanismo com marcha medida no repo. O bloco 1 do `g1_limpo` andou em ~1000 its, com `razao_marcha` 0,553. | VALIDADO | mem:g1-locomocao-primeiro-fatia-100; mem:g1-limpo-bloco1-marcha-formou |
| C2 | O portão mede a projeção da velocidade (`razao_marcha` ≥ 0,50), e não a sobrevivência. O portão por duração abriu com o robô imóvel. | VALIDADO | mem:g1-poc-portao-forma-mede-sobrevivencia |
| C3 | `action_rate_l2` −0,10. Com −0,25 o andar travou; com −0,10 ele abriu na it 421, e as quedas caíram de 30% a 1,8%. | VALIDADO | mem:g1-push-gate-compounding-stall |
| C4 | `air_time` com peso 0. A marcha formou assim no `g1_multitask` e no bloco 1 do `g1_limpo`. | VALIDADO | mem:g1-air-time-o-incentivo-desligado; g1_limpo/knobs.py:448 |
| C5 | `giro_sem_gingado`: o rastreio angular sem wx² + wy². O canal foi de 6,5% a 38% da renda, e o erro de guinada caiu 25%. | VALIDADO | mem:g1-limpo-giro-canal-e-envelope |
| C6 | σ do giro proporcional ao comando, com piso, e envelope de ±1,6 rad/s. A derivada no topo subiu 12×. | VALIDADO | mem:g1-limpo-giro-canal-e-envelope |
| C7 | Laço de rumo em 80% dos envs (`rel_heading_envs` 0,8). A deriva de 4,4°/s da marcha reta saiu. CORRIGIDO 02/10: com k = 0,5 o rumo assenta 6–13° fora (viés/k) nos plays 21850–23250; o laço fica, o ganho é pouco. | VALIDADO | mem:g1-limpo-deriva-de-rumo-com-a-caixa |
| C8 | Freio de velocidade por junta, em soma, com degrau por nível e piso na locomoção. Validado pelo dono em 01/10. | DONO | docs/planos/2026-09-23-freio-em-curriculo-pelo-nivel.md |
| C9 | O regime parado vem do estado (`limpo_twist_zerado`), e não do comando. Pelo comando, o laço de rumo afrouxava o freio 2,7× em 31% dos passos da pega. | VALIDADO | mem:g1-limpo-regime-vem-do-estado |
| C10 | O freio soma as juntas; o `amax` derrubou dois treinos. | VALIDADO | mem:g1-limpo-reducao-e-limiar-andam-juntos |

### 2.2 Observação e algoritmo

| # | conserto | prova | fonte |
|---|---|---|---|
| C11 | Vantagem do PPO normalizada por grupo de tarefa. Na mesma iteração: `fell_over` 51,9% → 0,9%, marcha 0,484 → 0,762. | VALIDADO | g1_limpo/algoritmo.py |
| C12 | One-hot da tarefa fora do normalizador, com média 0 e desvio 1 fixos. | VALIDADO | mem:g1-limpo-one-hot-normalizado-amplifica-slot-raro |
| C13 | Canais da caixa em zero quando a tarefa é ANDAR. Sem isso, a caixa a 5 m vazava para a rede. | VALIDADO | docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md §5, §6.1 |
| C14 | Caixa no frame da base, `giro_b` em 3 canais no lugar do ângulo escalar, e `meia_aresta` observada. | VALIDADO | docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md §4.1, §6.7 |
| C15 | Crítico assimétrico: ele vê os canais privilegiados do molde e o elo interno. | VALIDADO | docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md §6.1 |
| C16 | Canal novo só por append no fim, nos dois grupos e na mesma ordem. | DONO | ARQUITETURA:829 |
| C17 | O `pose` tira ombro e cotovelo da média só enquanto seguram. O punho fica na média com σ 1,00; com 0,30 o termo morria (expoente 8,1). | VALIDADO | mem:g1-limpo-punho-vai-ao-batente |

### 2.3 Manipulação: geometria e preensão

| # | conserto | prova | fonte |
|---|---|---|---|
| C18 | Alvo de transporte com x e y no peito (frame da base) e z absoluto no mundo. Com z relativo, o robô segurava agachado. | VALIDADO | mem:g1-multitask-caixa-fora-de-alcance; ARQUITETURA:967 |
| C19 | Alvo dentro do alcance. O alvo de 1,02 m era o ombro; ele baixou a 0,85–0,95 m. | VALIDADO | mem:g1-limpo-alvo-da-pega-era-circular |
| C20 | σ do kernel = distância inicial do env, e todo env nasce em exp(−1) com derivada. Com σ fixo pequeno o termo era zero com derivada zero, porque a palma nasce a 0,339 m. | VALIDADO | mem:g1-sigma-e-a-distancia-inicial |
| C21 | Distância das palmas à superfície da caixa, e não ao centro. As duas mãos contam, sem `min` contra uma esfera. | VALIDADO | specs/g1-limpo.md:262; mem:g1-limpo-cadeia-da-pega |
| C22 | Tombo medido pelo eixo de cima capturado na mesa (`FACE_DE_PE`). A face lateral era cega ao tombo palma a palma: 93° reais, 62° lidos. | VALIDADO | mem:g1-limpo-ang-cego-ao-tombo |
| C23 | Orientação com gradiente de 0 a 180°: `½(1 − Δθ/π) + ½ exp(−(Δθ/σ)²)`. A gaussiana sozinha morria a 66°. | VALIDADO | mem:g1-limpo-sigma-do-alvo-congelado-degenera |
| C24 | A reta da orientação nos termos de aproximação zera a 90°. Com `1 − Δθ/π`, a caixa deitada valia 0,5, e o robô a deitou. | VALIDADO | mem:g1-limpo-pega-parada-na-mesa |
| C25 | No BOTAR, a referência de orientação é a vertical do mundo, e não a face congelada. Endireitar deixou de custar. | VALIDADO | docs/planos/2026-09-21-botar-referencia-de-pe.md |
| C26 | Preensão contínua pela descarga da laje (`unload`, 9,7 N → 0), e não por bit. O `grasp` booleano travou a pega por 22 mil its. | VALIDADO | mem:g1-pegar-plato-do-grasp-booleano |
| C27 | Força de apoio projetada no eixo z. Prensar a caixa de lado satisfazia a norma. | VALIDADO | docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md §6.6.2 |
| C28 | `F_ref = m·g/(2μ)`, derivado da caixa. Componente normal por mão, com o `min` das duas. | VALIDADO | ARQUITETURA:1704–1719 |
| C29 | Estágios da pega em produto, e não em soma. Com soma, o robô ganhava empurrando a caixa com o pé. | MEDIDO | ARQUITETURA:1639 |
| C30 | O BOTAR paga a caixa apoiada (`load`), e o fecho exige a caixa apoiada. Sem isso, pairar 1 cm acima pagava mais. | VALIDADO | docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md §6.6 |

### 2.4 Cena, DR e terminações

| # | conserto | prova | fonte |
|---|---|---|---|
| C31 | Contato do corpo com a mesa é multa em rampa: zero abaixo de 50 N, cheia em 100 N, peso −2 por sensor. A terminação de 28/08 matava 76% dos episódios de manipulação por ruído de exploração; com a rampa (01/09), a `descarga` foi de 0 a 0,994. | VALIDADO | g1_limpo/knobs.py `Contato`; g1_limpo/recompensas.py `contato_mesa` |
| C32 | A lista de geoms do contato inclui os pads da mão. Com a lista invertida, escorar era grátis. | VALIDADO | vault/G1/Abordagens/PPO/Currículos/C3 g1_limpo/g1-limpo-mesa-contato-zero.md |
| C33 | `|hip_yaw|` > 1,0 rad termina o episódio, e a política reaprende em ~400 its. | VALIDADO | mem:g1-limpo-limite-de-junta-nao-morde-no-curso-largo |
| C34 | Soltar a caixa fora do alvo termina o episódio, pela velocidade relativa acima de `v_solta`. | VALIDADO | specs/g1-limpo-soltar-termina.md |
| C35 | `caiu` lê o tamanho da caixa. A caixa derrubada antes da preensão não termina o episódio. | VALIDADO | docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md §6.7; docs/planos/2026-09-04-proposta-gradientes-g1-limpo.md P7 |
| C36 | DR de tamanho da caixa por mundo, no startup, e inércia real gravada no reset. | VALIDADO | docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md §6.7; mem:g1-limpo-inercia-real-da-caixa |
| C37 | Caixa em posição pegável, com jitter curto. Com jitter de 0,20 m, 60–81% dos episódios nasciam fora do alcance. | VALIDADO | mem:g1-multitask-caixa-fora-de-alcance; mem:g1-poc-auditoria-travas |
| C38 | A laje de destino só se afasta do corpo no sorteio. O toque de pé caiu de 39% a 0,6%. | DONO, MEDIDO | mem:g1-limpo-teleporte-da-laje-no-corpo |
| C39 | Laje fora da cena (+5 m) quando a tarefa não a usa. A caixa fica apoiada nela. | MEDIDO | docs/planos/2026-08-20-curriculo-progressivo-e-andar.md, adendo T1 |
| C40 | Três sensores de mesa: corpo, palma e dorso. Eles mostraram que 94,7% do contato era o tronco. | MEDIDO | mem:g1-limpo-escoro-e-o-tronco |

### 2.5 Currículo e fatia

| # | conserto | prova | fonte |
|---|---|---|---|
| C41 | A ordem dos termos de currículo é contrato: o nível lê o episódio que acabou antes do sorteio do seguinte. A ordem errada pôs o ponto fixo em 0,714. | VALIDADO | mem:g1-poc-auditoria-travas; ARQUITETURA:594 |
| C42 | Cada nível contém os anteriores, e nenhum nível é impossível. O PLR punha ~70% da amostra no nível impossível. | VALIDADO | ARQUITETURA:2042; mem:g1-lift-box-future-refinements |
| C43 | Sucesso medido só nos envs da tarefa. Gatilho que uma perturbação externa zera não mede competência. | VALIDADO | mem:g1-curriculo-congelamento-escala; specs/g1-limpo.md:454 |
| C44 | Fatia corrigida pela duração do episódio. Na v3, a cadeia fixa por índice de env resolve sem fórmula. | VALIDADO | mem:g1-locomocao-primeiro-fatia-100 |
| C45 | As tarefas abrem uma de cada vez, e a seguinte só depois de sobreviver. A fase SEGURA tornou o fecho do PEGAR sobrevivível: 13 de 19 envs vivos em 4 s na sonda, e o `s_B` chegou a 0,6 na zero17. | VALIDADO | specs/g1-limpo-curriculo-de-cadeia.md |
| C46 | Warm-start com LR 5e-4. Com LR 1e-3 o treino colapsou (`fell_over` 84). | VALIDADO | mem:warmstart-lower-lr |

### 2.6 Poses de IK

| # | conserto | prova | fonte |
|---|---|---|---|
| C47 | Gerador de IK de referência por altura de laje e tamanho de caixa: 15 poses validadas no viewer em 16/09. Regras: pelve ≤ 60°, folga de 10 mm, `FOLGA_PALMA` 0,035, sem rotação interna do hip_yaw. PEGAR e BOTAR usam a mesma tabela. Validado pelo dono em 01/10. | DONO | mem:g1-limpo-ik-referencia-botar |
| C48 | IK de pé: pelve 0,79 m nivelada e tronco vertical. | MEDIDO | specs/g1-limpo-postura-de-pe.md |
| C49 | Interpolação em espaço de junta entre poses-chave é o método da Unitree; a reta cartesiana sofre droop. Servos de posição com `implicitfast` estabilizam o braço. | VALIDADO | mem:g1-arm-ik-pd-control |

## 3. Remendos de economia — saem

| # | remendo | o que ele consertava | por que sai na v3 | fonte |
|---|---|---|---|---|
| S1 | `renda_congelada`, com a reescrita v2.1 | o elo seguinte pagar acima do congelado; o arranque | o potencial paga o salto de alvo uma vez, e a soma não depende de quando | mem:g1-limpo-renda-congelada-paga-10x-o-deficit; specs/g1-limpo-gradientes-v2.md |
| S2 | Tabela por estado (10 × 11) e a conta teto ≥ piso: BOTAR ×2, cauda ×8, CARREGAR ×3,5, PEGAR_COM ×4 | equilibrar a renda entre estados | sem renda por estado, não há o que equilibrar | specs/g1-limpo-tabela-por-estado.md |
| S3 | Rastreio por estado: 0 nas esperas e no PEGAR, ×3,5 no CARREGAR, ×1 parado (R2, K1b) | estátua com rastreio; desvio pago | duração do CARREGAR dada pelo controlador | mem:g1-limpo-carregar-parado-e-estatua |
| S4 | Caudas: CARREGAR de B e R, cauda do BOTAR, `pelve_piso_cauda`, `rampa_cauda`, `_alcancar` zerado em `soltou` | renda depois do último fecho | cada cadeia da v3 termina numa tarefa de duração fixa | specs/g1-limpo-cauda-parada-de-pe.md |
| S5 | Currículo de cadeia em 4 fases, SEGURA, balanceador `p_C` e teto do `p_C` | ordem de abertura e fatia da cadeia C | partição fixa por env e abertura em degraus | specs/g1-limpo-curriculo-de-cadeia.md; specs/g1-limpo-teto-do-p-c.md |
| S6 | Fecho por sustentação, "fechou e não avançou" para a cauda, reconferência do `perto` | fecho pagava menos que pairar | avanço por tolerância do alvo macro, e pairar paga zero | relatório do tombo, S2 |
| S7 | `staged`, `precise_pos`, `squeeze`, `unload`, `load`, `postura_ereta` e `forma_postural` como renda por passo | gradiente da pega e da pose | a proximidade entra no potencial; a forma desses termos (σ = d₀, produto, eixo de cima) fica | relatório da pega parada §1: 12,7/s no PEGAR_COM, dos quais ~11/s sem progresso |
| S8 | Os gates contra o piso da estátua (`rastreio_por_elo`, margens do smoke) | parado rendia mais que tentar | parado não ganha renda de tarefa | mem:g1-limpo-piso-da-estatua-travava-a-pega |
| S9 | Espera com publicado diferente do interno e caixa zerada | treinar ANDAR parado → PEGAR sem anunciar | o potencial não paga o agachamento antecipado; a caixa fica visível, como no §4.2 | mem:g1-limpo-janela-de-espera |
| S10 | σ pendente, recalculado na abertura do elo | σ nascia com a pose do reset | σ = d₀ fica como regra; a máquina sai com as esperas mascaradas | specs/g1-limpo-espera-sigma-e-pose.md |
| S11 | `de_pe` como portão do fecho (saiu do BOTAR, ficou no PEGAR) | fecho sem agachar | a pose de pé da altura entra no alvo macro | specs/g1-limpo-botar-fecha-e-para.md |
| S12 | `limite_de_pelve` no CARREGAR | pelve a 0,564 m com a caixa | a pelve do alvo macro vem da IK de pé com caixa | docs/planos/2026-09-15-limite-de-pelve-no-carregar.md |
| S13 | `Giro.portao_da_renda`: renda congelável × kernel de giro | deriva de rumo durante a pega | rastreio do giro zero na pega e laço de rumo | mem:g1-limpo-deriva-de-rumo-com-a-caixa |
| S14 | Profundidade do BOTAR pelo `s_C` (R4, R4b) | a laje do BOTAR herdava a do PEGAR | nível próprio do BOTAR, decisão de 01/10 | specs/g1-limpo-botar-profundidade.md |
| S15 | REORIENTAR inerte a 5% e a trava dele contra o normalizador | slot raro no one-hot | o REORIENTAR fica fora até entrar de verdade | docs/planos/2026-09-04-proposta-gradientes-g1-limpo.md P8 |

## 4. Rejeitados — não refazer

| # | ideia | motivo | fonte |
|---|---|---|---|
| R1 | Modo entrega no viewer | a caixa teleportava 4,5 m; o elo ANDAR com caixa perto não existia no treino | mem:g1-limpo-modo-entrega-viewer |
| R2 | Catraca `max(0, M − viva)` e déficit `max(0, fecho − viva)` | pinam o total, e fazer melhor não paga | §6 |
| R3 | Multiplicador por posição na cadeia | inobservável pelo crítico | §6 |
| R4 | Cone elíptico de orientação | NaN duas vezes | §6 |
| R5 | Termo novo como primeiro recurso | o conserto é a forma do termo que existe | §6 |
| R6 | Teto de tempo por tarefa | correr é o risco; demorar não é | §6 |
| R7 | Torque e energia como regra explícita | por enquanto o robô infere no treino | §6 |
| R8 | Ordem de soltura das mãos prescrita | o robô aprende | §6 |
| R9 | BOTAR → REORIENTAR como cadeia; tombo do REORIENTAR para a frente ou para o lado | sempre há ANDAR parado entre manipulações; o tombo é só para trás | §6 |
| R10 | Aborto por perda de preensão como termo | o laço de recuperação é do controlador | §6; mem:g1-limpo-aborto-por-perda-de-preensao |
| R11 | Terminação por contato com a mesa (28/08) | matava 76% dos episódios de manipulação por ruído de exploração; trocada pela rampa em 01/09 | g1_limpo/knobs.py `Contato` |
| R12 | Clip da ação do hip_yaw | o corte depois da decisão esconde o hábito e não vale no robô real | mem:g1-limpo-limite-de-junta-nao-morde-no-curso-largo |
| R13 | Escala da ação do punho igual à do cotovelo | decisão do dono, 29/09 | mem:g1-limpo-escala-da-acao-do-punho |
| R14 | `precise_ori` 4 → 0 no PEGAR | só barateia o tombo | mem:g1-limpo-pega-parada-na-mesa |
| R15 | Âncora de transporte baixa (0,75–0,85 m) | o dono a reverteu: a caixa viajava na altura do quadril | relatório da pega parada §5-A |
| R16 | Penalidade de tombo com zero em 25° dentro do `precise_ori` | −2,4/s a 30° põe a margem de erguer abaixo da que já travou a pega | mem:g1-limpo-pega-parada-na-mesa |
| R17 | Pose de IK como referência paga na espera; `pose` ×8 em regime parado | a pose de marcha é outra; renda na espera trava as tarefas | decisão do dono, 01/10 |
| R18 | Laje girada com o rumo; "mesma mesa com altura nova" | decisão do dono, 01/10 | mem:g1-limpo-teleporte-da-laje-no-corpo |
| R19 | `condim` 4 na palma | o `CollisionCfg` sobrescreve e gera 8 pontos | relatório do tombo, rejeitados |
| R20 | Reaquecer o `action_std` sozinho | reconvergiu pior em 60 its | mem:g1-poc-bloco1-treino |
| R21 | Baixar o `upright` para abrir o tronco | 64% dos envs caíram | mem:g1-lift-box-task |
| R22 | Tirar o `base_lin_vel` do ator | `dur_loco_ema` caiu de 112 a 26 | mem:g1-poc-auditoria-incentivos |
| R23 | Jitter do alvo de transporte em y | deslocava o alvo até 11° fora do eixo | ARQUITETURA:992 |
| R24 | BFM residual para a pose | o bloco fixo domina 15×; só um prior serve | mem:g1-residual-bfm-experiment |
| R25 | Handover entre controladores | um modelo só | docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md §1.1 |
| R26 | Copiar o x dos pés na IK | a perna não alcança a laje de 0,55 m | mem:g1-limpo-ik-referencia-botar |
| R27 | `forma_postural` como causa do agachamento no alvo | puxa 0,12/s para agachar contra 0,7/s para ficar de pé | mem:g1-limpo-forma-postural-agacha-no-alvo |

## 5. Processo — regras de trabalho e de medição

| # | regra | fonte |
|---|---|---|
| P1 | Julgar o movimento pelo play, e não pelo log: a métrica por média dilui. Um tombo de 42° foi lido como 8°. | mem:g1-limpo-ang-cego-ao-tombo |
| P2 | O registrador monta a observação como o treino. Play cego ao giro produziu conclusões falsas sobre o tombo. | mem:g1-limpo-registrador-cego-ao-giro |
| P3 | A marcha se lê por `seg_proj / seg_pedido`, e não pelo portão diluído. | mem:g1-limpo-portao-diluido-pela-fatia |
| P4 | Toda régua de currículo precisa de um termo com gradiente. Régua sem incentivo travou 4931 its. | mem:g1-andar-como-rastreio-de-velocidade |
| P5 | A derivada se mede no ponto de operação antes de propor o peso. | CLAUDE.md; mem:g1-limpo-auditoria-gradientes |
| P6 | Antes de chamar defeito, calcular a fração das 30 000 its já rodada. O resume não se julga antes de +200 its. | mem:g1-poc-orcamento-30k-iteracoes; vault/G1/Transversais/Infra/g1-leitura-log-e-kaggle.md |
| P7 | Sondas de rede sem simulador, por checkpoint: V(alvo) − V(congelado) e a ação comandada. | mem:g1-limpo-botar-esquecido-na-fase-4 |
| P8 | O smoke confere estrutura e aritmética; o portão do treino é play, sonda e CSV. Cada check tem de poder falhar. | docs/planos/2026-08-25-g1-limpo.md §7; ARQUITETURA:2311 |
| P9 | Comparar com o `model_0` na mesma tarefa, porque ficar de pé é grátis. | mem:g1-de-pe-e-gratis-no-mjlab |
| P10 | Observação nunca por índice fixo; junta e atuador por nome. | mem:g1-mjlab-warmstart-obs-armadilhas; mem:unitree-g1-sim-mujoco-not-isaac |
| P11 | Seed fixo entre sessões. | mem:g1-mjlab-warmstart-obs-armadilhas |
| P12 | Pins: mjlab 1.5.3, rsl-rl-lib 5.4.0, mujoco 3.10.0. Treino só no Kaggle ou no Colab; local só smoke, paridade, inspeção e leitura. | vault/G1/Transversais/Infra/g1-leitura-log-e-kaggle.md |
| P13 | Limiar de check acima de 5 mm: a gravidade baixa a caixa 2 mm entre escrita e leitura. | mem:g1-mjlab-reset-buffers-obsoletos |
| P14 | Uma mudança por bloco. A condição de fecho se testa contra a do elo anterior. | ARQUITETURA:2248; vault/G1/Abordagens/PPO/Currículos/C3 g1_limpo/g1-limpo-desenho.md |
| P15 | `Episode_Termination` é contagem. `Episode_Reward` vira taxa com × 20 / (duração × peso). | vault/G1/Transversais/Infra/g1-leitura-log-e-kaggle.md |
| P16 | Resume no Kaggle: checkpoint em pasta datada no passado, interromper o kernel sem "Stop session", conferir o commit no print `clone =`. | vault/G1/Transversais/Infra/g1-leitura-log-e-kaggle.md; sessão de 01/10 |

## 6. Abertos — a spec da v3 decide

| # | aberto | medida | proposta para a spec |
|---|---|---|---|
| A1 | Impacto da caixa sem preço | pico de 6–9× m·g por 25 mil its, mas a medida é aliasada [F37] | velocidade de chegada da caixa, limiar 0,99 m/s (queda de 5 cm) |
| A2 | Aperto sem preço acima de ~30 N, porque o `tanh` satura | 68,9 N contra 8,45 N necessários | faixa de força entre `F_ref` e k·`F_ref` no alvo da pega |
| A3 | Sacudida sem canal | nenhum termo lê a aceleração da caixa | medir no CARREGAR antes de propor |
| A4 | Contato com a laje abaixo de 50 N sem preço | coxa na laje em 137 de 150 passos do BOTAR (`model_5800`) | DECIDIDO 01/10: joelho da rampa em 10 N |
| A5 | Uma mão só paga igual; o fecho do PEGAR não confere preensão | `precise_pos` igual com uma ou duas mãos | duas mãos em todo alvo macro depois da pega |
| A6 | O fecho do BOTAR não exige as mãos livres | segurar no fecho rendia 17,9 contra 10,3/s | alvo macro "mãos abertas a 5 cm" |
| A7 | Altura de transporte sorteada em U(0,85; 0,95) | o enunciado pede 0,95 m | decidir entre o enunciado e a faixa |
| A8 | Faixa de comando assimétrica: vx de −1,5 a 2,0, vy de −1,0 a 1,0 | o enunciado pede 2 m/s nas quatro direções | seguir o enunciado ou corrigi-lo |
| A9 | `track_lin` com σ fixo 0,5 | topo do envelope quase sem derivada | σ proporcional ao comando, como o do giro |
| A10 | Giro no lugar | ~3% da frota; |wz| sorteado abaixo do piso σ 0,707 | 25% do sorteador, e |wz| acima do piso ou piso menor; medir |
| A11 | O portão da locomoção é cego ao giro | o segmento de giro puro é descartado | projeção sobre o twist inteiro |
| A12 | O ANDAR parado não tem laço de rumo | wz = 0 fixo | estender o laço |
| A13 | Braços abertos parado quase não custam | 0,017/s com dois `shoulder_roll` a 0,5 rad | σ menor de ombro e cotovelo fora da manipulação; medir |
| A14 | O `upright` põe o ótimo do tronco em ~3° | a IK do BOTAR pede 24° | tronco pelo alvo macro na manipulação; `upright` só como guarda |
| A15 | O freio taxa a frenagem no CARREGAR | não medido | medir a frenagem |
| A16 | Cair segurando custa menos que largar | +3 a +11 estimados | a terminação zera o potencial nos dois casos; medir |
| A17 | O teto do `limite_de_junta` corta a derivada | custo fixo além do teto | continuar linear depois do teto |
| A18 | A `faixa_de_pose` taxa o agachamento certo do BOTAR | ~0,48/s | decidir se a v3 precisa dela ou se o `limite_de_junta` basta |
| A19 | A massa da caixa não é observada | `F_ref` vai de 6,13 a 30,66 N | canal só no crítico |
| A20 | Tombo da caixa na subida | 42–59° nos plays do zero17 | caixa nivelada em todo alvo macro depois da pega; medir o pico no PEGAR |
| A21 | Agachamento antecipado na espera | joelho 0,67 → 1,34 rad em 0,5 s | o potencial deve zerar o ganho; conferir no play |
| A22 | Pelve baixa no CARREGAR | 0,68 m contra 0,79 m de pé | pelve do alvo macro pela IK de pé com caixa |
| A23 | `foot_slip` 20× mais barato que levantar o pé | não medido se ocorre | medir |
| A24 | A laje de 0,04 m é um degrau na frente dos pés e pode disparar a terminação | não implementado | conferir a geometria do nível mais baixo |
| A25 | REORIENTAR: k sorteado, primitivas, direção viva, fecho com 10° | latente | fora da primeira versão |
| A26 | Sim-to-real: ruído da caixa, DR de atrito e de PD, estimador da caixa, lidar, mãos reais | adiado | depois das quatro tarefas |

## 7. Contradições resolvidas

| # | contradição | resolução |
|---|---|---|
| X1 | O catálogo §5 diz que a marcha só forma com incentivo de voo. | O código tem `air_time` 0, e a marcha formou no bloco 1; no `g1_multitask` também. A linha do catálogo está errada, e a v3 começa com 0. |
| X2 | `vault/G1/Transversais/Infra/g1-mjlab-api-armadilhas.md` e `ARQUITETURA:2294` dizem que `dr.body_mass` corrompe a heap. | Consertado no mjlab 1.5.3, verificado em 14/09 (mem:g1-limpo-inercia-real-da-caixa). Vale o mais novo. |
| X3 | A spec da tabela por estado diz que a renda congelada é necessária: fechar levava a um estado que pagava menos por passo. | O argumento vale para renda por passo. Com o potencial, o fecho paga o salto uma vez, e a renda por passo da tarefa seguinte não depende do instante do fecho. |
| X4 | A espera zera a caixa (decisão de 02/09); o §4.2 diz "com a caixa já visível". | A v3 segue o enunciado, como proposta. |
| X5 | O plano de 02/09 diz que a faixa de ±2 m/s não é objetivo; o enunciado de 01/10 pede 2 m/s. | Vale o enunciado. |
| X6 | O nível por tarefa foi rejeitado em 30/09; em 01/10 o dono decidiu nível por tarefa. | Vale a decisão de 01/10. |
| X7 | Um leitor classificou o freio por nível como remendo. | O dono o validou em 01/10, e ele fica. |
| X8 | O §4.6 diz REORIENTAR inerte em 0,3 s e em metade da manipulação; o código usa 0,5 s e 5%. | Documental. O REORIENTAR sai da v3. |
| X9 | A memória da mesa, o `ARQUITETURA.md` e o comentário de `cena.py` dizem que o contato com a mesa termina o episódio. | Corrigido em 01/10, conferido no código: desde 01/09 é multa em rampa, e não há terminação de contato. A primeira versão deste inventário repetiu o erro em C31 e R11. |
