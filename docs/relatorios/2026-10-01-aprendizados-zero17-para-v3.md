# Aprendizados da zero17 e do teste do 21700 — insumo para a v3

Data: 01/10. Fonte: uma sessão de análise sobre a zero17 (`exp/g1-limpo-v2`, HEAD `39b8b82`)
e sobre o resume do `model_21700` no mesmo código. Complementa
`docs/relatorios/2026-10-01-inventario-v3.md` e `vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md`; onde um
achado confirma ou contradiz uma decisão de lá, a linha diz qual.

Método: plays no MuJoCo clássico (`registra_juntas.py`), log do treino colado pelo dono,
leitura de código e sondas de rede nos checkpoints. Quatro agentes Sonnet cegos (sem ler
relatórios nem memórias) investigaram o agachado e o BOTAR; eu revisei. Nada rodou env do
mjlab nem treino local. Cada número leva o tipo: MEDIDO (play, log ou sonda), CÓDIGO
(arquivo:linha) ou HIPÓTESE.

## 0. Resumo em nove linhas

1. O robô agacha antes do PEGAR porque a espera tem a MESMA observação do ANDAR parado e
   quase sempre vem logo depois do reset; agachar cedo rende no PEGAR e quase não custa.
2. O BOTAR não terminava porque o caminho até o fecho não tinha gradiente e o crítico não
   dava valor a um estado que nunca visitava. A laje rasa e depois a profundidade pelo
   `s_C` resolveram: `s_C` de 0,004 a 0,20.
3. A perna vira no BOTAR porque o alvo fica de lado (14–27°), passo é proibido e girar a
   pelve sobre os pés plantados não custa quase nada abaixo de 40° de `hip_yaw`.
4. O CARREGAR da zero17 anda a 70% do comando, gira 5°/s sem comando e anda agachado.
5. O `model_21700`, sob a MESMA recompensa da zero17, cai ~10× menos, pega com o tronco a
   12° (contra 27°) e não agacha na espera. Parte dos defeitos da linha zero vem do
   caminho de aprendizado, não só do ótimo da recompensa.
6. O play precisa reproduzir a observação do treino. O registrador era cego ao giro da
   caixa, e todo tombo medido em play estava confundido.
7. As métricas sem peso de funil (por cadeia e por elo) foram o que mais ajudou a decidir.
8. A linha bloco aprendeu porque recebeu as restrições depois da habilidade, por resume, e
   acumulou ~15 mil iterações de refino; a linha zero recomeça a cada mudança e enfrenta
   tudo na it 0 (seção 5b). O `model_23250` é o melhor modelo até 02/10.
9. O punho E gira na pega porque o contato o empurra além do motor de ±5 N·m (seção 3b), e o
   freio em catraca derruba a manipulação ao passar de ~1,75 para ~3 degraus (seção 5c). Os
   preços de segurança entram em níveis que podem descer.

## 1. Agachado antes de pegar

**Observado (MEDIDO, plays `*_andar_parado`, roteiro `parado:ANDAR:6.0`).** O robô agacha nos
primeiros 0,5–0,6 s depois do reset e piora com o treino:

| Checkpoint | Joelho (rad) | Pelve (m) | Depois de 4 s |
|---|---|---|---|
| keyframe (default) | 0,67 | 0,79 | — |
| `model_5000(1)` | 0,85 | 0,73 | fica |
| `model_5350` | 1,39 | 0,64 | relaxa até 1,12 |
| `model_5800` | 1,44–1,55 | 0,61 | não relaxa; cintura −0,24 |

O §3 pede joelho ~0,15 e pelve 0,78–0,79 parado.

**Mecanismo.**
- A espera antes do primeiro elo publica ANDAR, twist zero e zera os 10 canais da caixa
  (CÓDIGO: `comando.py:1128`, `observacoes.py:140`). O ator não distingue a espera do ANDAR
  parado; o crítico distingue (elo interno).
- A espera só existe nos 0,5–1,5 s depois do reset de um episódio de manipulação (CÓDIGO:
  `comando.py:1303-1307`). A locomoção nasce parada em 10% (`rel_standing_envs`).
- Com o sorteio de locomoção em 0,25 (MEDIDO, log da it 5800), logo depois do reset 95% dos
  passos "parado, sem caixa" são espera; sobre todo o ANDAR parado, 60%. A conta é
  p_m / (p_m + 0,15·p_loco). O sorteio da espera esconde QUANDO o PEGAR vem, não SE vem.
- Ganho: começar agachado adianta o fecho do PEGAR ~0,5 s, e fechar 1 s antes compra ~14 de
  renda (MEDIDO em outra sessão, memória `g1-limpo-espera-sem-agacha-o-andar-parado`).
- Custo: no ANDAR só o `pose` ×1 cobra a postura, contra o keyframe de joelho 0,669
  (CÓDIGO: `cena.py:294`). Endireitar 0,1 rad rende +0,04/s (MEDIDO). A pose de pé do §3
  rende 0,893 contra 1,0 do keyframe: o termo paga MAIS o semi-agachado do que o de pé.
  Nenhum termo mede a altura da pelve no ANDAR (`knobs.py:1330-1339`).
- Sonda de rede no `model_5350`: quanto mais agachado, mais o ator pede flexão do joelho
  (+0,67 a 0,89 rad, +0,89 a 1,07 rad); no `model_5000`, +0,02 e +0,39. Realimentação
  positiva. Ressalva: velocidades e última ação zeradas, fora da distribuição.
- Contraprova parcial: o `model_21850` (linha 21700 sob a mesma recompensa) não agacha na
  espera: pelve 0,77, joelho 0,48–0,54 (MEDIDO, `21850_c.csv`).

**Para a v3.**
- Decisões que atacam isto e o achado confirma: caixa visível na espera (§4.2) tira a
  confusão com o ANDAR parado; pose de pé parado = IK (decisão 10) põe o ótimo no §3.
- O que falta decidir: com a caixa visível, o ator SABE que o PEGAR vem, e agachar na espera
  continua rendendo pela pressa. O termo de pé na espera precisa de derivada maior que
  esse ganho no ponto de operação. A sonda que mede o ganho (V do início do PEGAR agachado
  contra de pé) não foi feita.
- O potencial `Φ(s') − Φ(s)` tira a renda por estado, mas o desconto γ ainda dá prêmio por
  chegar antes: 0,5 s são 25 passos, γ^25 = 0,78. Medir se esse prêmio basta para agachar.

## 2. BOTAR que começa e não termina

**Observado (MEDIDO, plays com o alvo real gravado).**

| Checkpoint | Menor distância ao alvo | Tombo da caixa | Nota |
|---|---|---|---|
| `model_5000(1)` | 0,20 m | 15–19° | para no meio da descida, 2 s imóvel |
| `model_5350` | 0,12 m | 16–19° | idem |
| `model_5800` (R4) | 0,04 m | 3,5–7° | pousa |
| `model_6050` (R4b) | 0,08 m | 15–18° | apoiada numa aresta: pousa tombada (§4.5 proíbe) |

**Mecanismo (antes do R4).**
- Fecho do BOTAR = `perto ∧ alinhado ∧ apoiada` por 0,5 s; de pé não entra (CÓDIGO:
  `comando.py:1528-1536`).
- No caminho só o `precise_pos` tem derivada: ~0,07/s por cm a 0,30 m do alvo (canal quase
  morto) e ~0,28/s por cm a 0,11 m. O `load` é degrau: só paga com a caixa perto e apoiada.
  84% do ganho vem depois do fecho e só chega ao robô pelo crítico (MEDIDO, agente).
- Sonda no crítico: o valor fica plano ao longo da descida, e com a caixa posta no alvo cai
  de ~+10 para −3 a −8: o crítico não valoriza um estado que a política não visita.
- A abertura teleportava a laje para dentro da perna em 31–72% das aberturas, e a laje
  herdava o topo do PEGAR (lajes baixas quase nunca).

**O que funcionou (MEDIDO, log).**

| Mudança | `s_C` |
|---|---|
| antes (it 5520) | 0,0038 |
| R4, laje rasa em metade das aberturas (it 5800) | 0,053 |
| R4b, laje independente, profundidade cresce com o `s_C` (its 6318–6320) | ~0,20 |

Funil da C na it 6320 (divisão por `c_episodio`): 55% chegam ao CARREGAR, 35% ao BOTAR, 17%
terminam com sucesso. Lajes baixas fecham menos (topo médio 0,48 m na chegada, 0,53 m no
sucesso). `caixa_largada` subiu de 0,13 (it 5800) para 0,54–0,67 com o R4b, e o log não diz
em que elo (HIPÓTESE: soltar de alto na laje baixa ou derrubar ao se levantar agachado).

**Para a v3.**
- Currículo de dificuldade pelo desempenho da própria tarefa funciona: o BOTAR só aprendeu
  quando a laje fácil apareceu. Confirma a decisão 3 (nível por tarefa).
- Sem gradiente no caminho, o fecho é penhasco. O potencial com `Φ` = proximidade ao alvo
  ataca isto; medir a derivada de `Φ` a 0,30 m antes do peso.
- O fecho precisa exigir caixa nivelada no apoio (o 6050 fecharia apoiado numa aresta a 15°)
  e o robô de pé ao fim (§4.5).
- Toda abertura que posiciona mobília confere a interpenetração com o corpo.

## 3. Perna virada no BOTAR

**Observado (MEDIDO, `21700_c.csv`, `21850_c.csv`).** No BOTAR, o alvo fica 0,40–0,44 m à
frente e 0,11–0,20 m à esquerda no frame do corpo (14–27°). A pelve gira 7–17° para o alvo
com os pés plantados; os dois `hip_yaw` ficam negativos:

| | `hip_yaw` E | `hip_yaw` D | joelho E / D |
|---|---|---|---|
| `model_21700` | −0,69 a −0,91 | −0,44 a −0,54 | 0,65 / 1,05–1,12 |
| `model_21850` | −0,38 a −0,76 | −0,20 a −0,37 | 0,72–0,80 / 1,14–1,30 |

**Mecanismo.**
- No treino, a laje do BOTAR nasce com ±0,10 m lateral e o alvo com mais ±0,05 m. No play,
  o roteiro devolve a caixa à laje da pega depois que o corpo girou 22° no PEGAR.
- Passo é proibido parado (§3 "Pés"), e o rastreio paga ficar parado no BOTAR.
- A rampa do `limite_de_junta` do `hip_yaw` começa em 0,25 do curso = ±40° = 0,69 rad
  (CÓDIGO: `knobs.py:1537`; o notebook do zero usa `(20,0, 0,15, 0,25)`). O uso medido,
  22–30°, fica abaixo: o termo vale zero. Só o `pose` cobra, ~0,15/s.

**Para a v3.**
- Abrir o BOTAR com o alvo no rumo do robô (§4.5 "parado em frente à laje"), ou dar uma rota
  legítima para alvo lateral: cintura (`waist_yaw`), que hoje também é grátis nessa faixa.
- A rampa por junta de min(15°, 30% do meio-curso) da decisão 13 morderia no `hip_yaw` a
  22–30°. Sem rota legítima, o robô procura a próxima compensação barata.
- O corpo girar 22° durante o PEGAR sem comando fere o §4.4 ("girar sem comando").

## 3b. Punho esquerdo empurrado pela caixa na pega

**Observado (MEDIDO, 05/10, `24350_B_45_acao.csv`, play com a ação do ator gravada).** Na pega,
o `wrist_yaw` E vai de +0,04 a +1,46 rad (88% do curso). O alvo do PD que a política pede fica
em +0,30 a +0,40 rad (ação +4 a +5; escala 0,075 rad/unidade). O `wrist_roll` e o
`wrist_pitch` seguem o alvo; só o yaw é arrastado. Por cinemática (`mj_forward` com as juntas
do CSV), a mão E gira 44° e desliza 7,2 cm sobre a face da caixa em 1,5 s; a D, 11° e 1,2 cm.
Com impratio 10, a mão E gira 24° e desliza 4,2 cm. A caixa não gira em relação ao corpo.

**Mecanismo.**
- O motor do `wrist_yaw` tem kp 16,8 e ±5 N·m (CÓDIGO, `cena.npz`): ele só fica ~0,30 rad
  longe do alvo antes de saturar. A junta a 1,1 rad do alvo exige um torque externo acima de
  5 N·m: é o contato com a caixa, não o comando da política.
- A política monta a configuração que carrega o punho: o roll alto (+1,29 rad pedido) e o
  aperto sem preço (~69 N contra ~8,5 N necessários, memória `g1-limpo-caixa-esmagada`).
  O momento no eixo de yaw passa do que o motor segura, e a mão rola sobre a face.
- O ±5 N·m é o limite do hardware real: o conserto não é dar torque ao punho.

**Para a v3 (dono, 05/10).**
- Dar preço ao aperto excessivo e à saturação do punho com a caixa na mão (distância entre a
  junta e o alvo do PD, ou torque no limite).
- **Esse preço entra em NÍVEIS, como o freio de velocidade.** Cheio desde a it 0, ele ensina
  a não apertar e o robô desaprende a pegar. O preço sobe quando a pega já fecha.
- O nível desse preço tem de poder descer (ver a seção 5c): em catraca, ele repete a falha
  do freio.

**O aperto medido no `model_24350` (MEDIDO, 05/10, cópia do registrador com a força normal
nas palmas, laje 0,45, caixa de 1 kg).** Segurando a caixa (t ≥ 2,2 s): palma E p50 40,3 N,
palma D p50 36,6 N. A força necessária é `F_ref = m·g/(2μ)` = 6,13 N, então o robô aperta
~6× o necessário. O `squeeze` paga `tanh(min(F)/F_ref)` = tanh(6) ≈ 1: a derivada é zero, e
o excesso não tem preço. No BOTAR o aperto segue entre 33 e 50 N, e lá o `squeeze` vale zero
(`_fora_do_botar`). Na tabela por estado o `squeeze` vale zero no CARREGAR (`knobs.py:1328`):
um preço posto dentro dele não alcança o transporte nem o pouso.

**A correção existe para impedir que o robô esmague a caixa (dono, 05/10).** O enunciado
pede "sem aperto além do necessário" (§3, "Caixa íntegra e nivelada") e lista "amassado por
aperto" entre os danos (§2). O giro do punho E (acima) é um efeito do mesmo aperto.

**Transferência para o real: a pensar (dono, 05/10).** O robô real não tem a pose exata da
caixa (estimativa por câmera) nem sensor de pressão nas mãos. Pontos para a discussão:
- O preço usa informação privilegiada (força de contato) só na RECOMPENSA. O ator não precisa
  dela no real: ele aprende um padrão de movimento cujo resultado é apertar pouco.
- O ator controla a força pela POSIÇÃO: força ≈ rigidez do PD × quanto a mão entra além da
  face. Esse "quanto" depende da largura e da pose da caixa que o ator vê. Um erro de
  estimativa da câmera vira erro de força. Treinar com ruído na pose e no tamanho da caixa
  observados (o estimador simulado da memória `g1-limpo-oraculo-da-caixa-no-real`) ensina
  margem.
- O ator já tem um sensor de força implícito: ele observa a posição das juntas e a última
  ação, então enxerga o erro do PD (alvo − junta), que é proporcional ao torque. No real o
  G1 também estima o torque das juntas pela corrente do motor. Para o comportamento treinado
  valer no real, o kp, o kd e o limite de torque da simulação têm de bater com os do robô.
- Opção a medir: dar preço ao TORQUE das juntas do braço (`kp × (alvo − q)`), e não à força de
  contato. É uma grandeza que o robô real mede, então a mesma régua valida o comportamento
  no real.

**O aperto não acompanha a massa (MEDIDO, 05/10, `model_24350`, laje 0,45, mesmo roteiro).**

| Massa | F_ref (μ 0,8) | Palma E / D | min / F_ref | Caixa na pega | `wrist_yaw` E máx |
|---|---|---|---|---|---|
| 1 kg | 6,1 N | 40,3 / 36,6 N | 6,0 | 0,898 m | +1,47 |
| 3 kg | 18,4 N | 39,5 / 36,5 N | 2,0 | 0,865 m | +1,44 |
| 5 kg | 30,7 N | 29,4 / 38,2 N | 1,0 | 0,724 m | +0,95 |

A política aperta ~37–40 N qualquer que seja a massa: esmaga a de 1 kg (6×) e fica no limite
do deslize com a de 5 kg (a caixa desce 17 cm na mão). O treino nunca mostrou atrito
diferente: palma e caixa têm μ = 1,0 fixo (`cena.py:238`, `knobs.py:33`), e o `squeeze` usa
um μ fixo de 0,8 (`knobs.py:964`). Um aperto fixo, dimensionado para a caixa pesada, basta.

**Escorrega → aperta mais; caixa pesada → aperta mais (dono, 05/10).** O comportamento
esperado é o humano: apertar conforme o peso sentido ao erguer, e apertar mais quando a
caixa começa a deslizar. No real o robô não tem sensor de tato; ele infere pela carga nas
juntas do braço, pelo erro do PD, pelo histórico e pela pose da caixa da câmera. O treino só
ensina isso se o deslize ACONTECER na simulação: o atrito e a massa têm de variar, e a
recompensa tem de conhecer o μ e a massa de cada env (informação privilegiada, fora do ator).

**A caixa é presa pelo pulso, não só pela palma (MEDIDO, 05/10, contatos na caixa em t = 3,0 s,
`model_24350`, laje 0,45).**

| Caso | Pulso D (`wrist_collision`) | Pulso E | Tronco | Pad D | Pad E |
|---|---|---|---|---|---|
| 1 kg, μ 1,0 (lateral / vertical) | 73 / +11 N | 64 / −9 N | 31 / +5 N | 36 / +3 N | 46 / 0 N |
| 5 kg, μ caixa e pads × 0,3 | 100 / +30 N | 66 / +24 N | — | 38 / +9 N | 74 / −2 N |

- A cápsula `*_wrist_collision` (corpo `wrist_pitch_link`, μ 1,0) aperta a caixa mais que o pad.
  No caso de 5 kg com atrito baixo, os pulsos sustentam 54 N de um peso de 49 N.
- O `squeeze` só mede os pads (`recompensas.py:579-616`). O aperto real fica perto de 110 N por
  lado para uma caixa de 1 kg, ~20× o necessário, e quase todo fora da régua.
- O MuJoCo usa o maior μ do par. Uma DR de atrito só na caixa e nos pads é mascarada pela
  cápsula do pulso e pelo tronco, que ficam em 1,0: com μ × 0,3 a política atual não larga a
  caixa em nenhum caso medido (μ 0,3 a 1,0; 1 a 5 kg).
- Para a v3: a DR de atrito tem de passar por TODO contato com a caixa. Um caminho é dar
  `priority` à geom da caixa (o contato usa os parâmetros dela) e prioridade maior à laje. O
  preço e a métrica do aperto têm de somar toda força do robô na caixa, e não só os pads.
- Hipótese não medida: o pad (no `wrist_yaw_link`) e a cápsula (no `wrist_pitch_link`) prendem
  a mesma face ao mesmo tempo, e isso força o giro do `wrist_yaw`.

**DR com `priority` na caixa (MEDIDO, 05/10, `model_24350`, roteiro `carregar`: 7 s de PEGAR e
10 s de CARREGAR a 0,5 m/s, laje 0,45).** A geom da caixa recebe prioridade 2, acima dos pés
(1) e do resto do robô (0). A mesa recebe prioridade 3. Todo contato do robô com a caixa passa a
usar o μ da caixa, e o μ efetivo foi conferido em `contact.friction`.

| μ da caixa | 1 kg | 3 kg | 5 kg |
|---|---|---|---|
| 1,0 | segura | segura | segura |
| 0,6 | segura | segura | segura |
| 0,45 | segura | segura | escorrega em 11,5 s (impratio 2); em 6,7 s (impratio 1) |
| 0,3 | segura | escorrega em 7,0 s e pega de novo da mesa (impratio 2) | escorrega na subida, em 2,8 s |

- A força total do robô na caixa fica em 230–250 N em todos os casos. A política não muda o
  aperto com o μ nem com a massa (F/W de 23 com 1 kg e de 4,6 com 5 kg).
- Como cai: a caixa desce devagar entre as mãos (creep), os pads perdem o contato, e a caixa
  escorrega para fora. O dono viu isso no viewer (μ 0,3, 3 kg).
- O impratio do treino (2) atrasa o escorrego, mas não o evita. O play usa 1.
- Para a DR: na faixa [0,6; 1,0] a política atual não perde a caixa, então essa faixa não
  quebra o resume, mas também não cria pressão para apertar mais. A pressão aparece com
  μ ≤ 0,45 e caixa pesada. A borda de baixo da faixa tem de descer por nível até essa região.

## 3c. DR para transferir ao real, em níveis desde o começo (dono, 05/10)

A v3 entra com randomização de domínio desde o início, em níveis que sobem com o domínio da
tarefa e podem descer:
- **Atrito** palma–caixa e caixa–laje.
- **Massa** da caixa (hoje 1 a 5 kg por nível, com inércia real desde 14/09).
- **Erros dos sensores:** ruído e atraso na pose e no tamanho da caixa estimados pela câmera,
  ruído no torque estimado das juntas e na IMU.

A recompensa usa os valores verdadeiros do env (F_ref = m·g/(2μ) por env); o ator vê só o
que o robô real mede. Ver também a memória `g1-limpo-oraculo-da-caixa-no-real` (estimador
simulado da pose da caixa).

## 4. CARREGAR da zero17 (MEDIDO, `6050_carregar.csv`, comando vx 0,5)

| Defeito | Medido |
|---|---|
| Lento | ~0,35 m/s no rumo do corpo |
| Gira sem comando | rumo de −22° a −68° em 9 s, com o laço de rumo ligado |
| Agachado (§4.4 proíbe) | pelve 0,69–0,70 m, joelho 1,05–1,25 |
| Passo curto | oscilação do joelho ~0,15 rad, contra ~0,5 no ANDAR sem caixa |
| Caixa alta | 1,02–1,03 m, 13 cm acima do alvo de 0,90 |

O mesmo checkpoint anda bem sem caixa (0,74 m/s para 0,8; pelve 0,765). Não diagnosticado.
O tombo de 50° desse play era cego ao giro (seção 6).

## 5. Resume do `model_21700` no HEAD (`21700_zero_teste`)

**Compatibilidade (CÓDIGO).** Ator 114 e crítico 131 nos dois. `observacoes.py` e `cena.py`
sem mudança desde `7b1b5ac`. O load dobra a normalização antiga do one-hot na 1ª camada
(`c0352b0`). Sem `fase_cadeia` no checkpoint, o resume entra na fase 1 (`runner.py:99-100`).
Os 3 canais de giro mudaram de significado em `04e99ca` (28/09): eixo de cima contra a
vertical, e não a face lateral. Config igual à da zero17, freio por env ligado.

**Resultado (MEDIDO, log, its 21793–21953, ainda na fase 1).**

| | 21700 no HEAD | zero17 (its 6318–6320) |
|---|---|---|
| `fell_over` | 0,08–0,17 | 1,0–1,6 |
| `caixa_largada` | 0,04–0,21 | 0,54–0,67 |
| `tronco_na_pega` | 11–14° | 26–28° |
| `precise_ori` | 0,82–1,20 | 0,42–0,50 |
| `altura_da_pelve` | 0,73 | 0,68 |
| C que abre e termina com sucesso | 70–100% | ~17% |

Play do `model_21850`: pega com 8,5° de tombo (com `giro_w`), bota a 6 cm do alvo, perna
virada (seção 3), sem agachar na espera.

**Ressalvas.** Fase 1 (90% dos fechos seguram no PEGAR, `p_C` 0,05): a mistura é mais
fácil. O freio por env nasce em 0 e sobe a partir de ~it 22100. A fase 4, comparável,
chega ~it 22500.

**Resultado na fase 4 (MEDIDO, log e plays, its 22500–23253).**

| | 21700 no HEAD (its 22800–23253) | zero17 (it 6320) |
|---|---|---|
| C: chega ao CARREGAR / BOTAR / sucesso | 63–76% / 48–71% / 44–65% | 55% / 35% / 17% |
| Desvio no fim da espera do PEGAR | 23–32% | ~15% |
| Topo médio dos sucessos do BOTAR | 0,32–0,36 m | ~0,53 m |
| `altura_da_pelve` | 0,73–0,74 | 0,68 |
| `fell_over` / `caixa_largada` | 0,33–0,54 / 0,42–0,79 | 1,0–1,6 / 0,54–0,67 |

Plays: o `model_22500` bota na laje 0,55 e congela na 0,35 (caixa 37 cm acima do alvo); o
`model_22800` bota na 0,35; o `model_23000` bota na 0,15 com 1 e 3 kg e se levanta sem
derrubar a caixa. O giro do corpo na pega caiu de ~50° (22500) para 5–7° (23000). O
`model_23250` (sha 7967fa59d9b4) é o melhor modelo até 02/10 (veredito do dono). O `s_C`
estabiliza em ~0,50, o ponto de equilíbrio do R4b (seção 8, item 6). Freio por env subindo
até 1,76 degrau sem efeito na postura.

**Para a v3.** A mesma recompensa dá comportamentos diferentes conforme o caminho. Na v3,
olhar a postura e a pega cedo, logo que cada tarefa abre, e não só no fim; um defeito que
forma cedo (o agachado foi de 0,85 a 1,49 entre as its 5000 e 5800) vira mínimo local.

**Giro no lugar no `model_23250` (pilota, veredito do dono, 02/10).** Ele gira no lugar, com
e sem caixa, mas a movimentação ainda é estranha. Causa provável: pouca exposição.
- Sem caixa (CÓDIGO, `knobs.py:551`; `sorteio` ≈ 0,28 do log): o ramo `turning` realiza
  0,10 × (1 − 0,10) ≈ 9% dos segmentos de locomoção; com ~28% dos episódios em locomoção,
  ~2,5% dos passos do treino.
- Com caixa (estimativa): só o CARREGAR andando da cadeia B sorteia comando (o da C é
  parado); com `p_C` 0,72–0,80, ~1% dos passos ou menos.
- O estágio 0 do `command_vel` (|wz| ≤ 0,5) ainda tinha o piso σ 0,707 do rastreio angular,
  onde a estátua rende o mesmo que girar (`recompensas.py:461`, medido 01/10); a linha bloco
  treinou boa parte do tempo nele.
- Na v3: decisões 5 e 8 e o sorteador único 25/25/25/25 de `g1-v3-decisoes.md` levam o giro
  no lugar de ~2,5% para ~25% da locomoção (~10×) e o põem também com a caixa; falta
  garantir |wz| acima do piso do σ.

## 5b. Por que a linha bloco aprendeu e a linha zero trava

**Resposta.** A diferença principal não é a recompensa final: é o histórico de treino. A
linha bloco aprendeu cada habilidade com regras frouxas e recebeu as restrições DEPOIS, por
resume, sobre uma política que já funcionava. A linha zero recomeça a cada mudança e enfrenta
todas as restrições desde a it 0.

| | Linha bloco (até o `model_21700`) | Linha zero (até a zero17) |
|---|---|---|
| Período (git) | 26/08 a 18/09 | 14/09 a 01/10 |
| Recomeços do zero | 1 (bloco1); depois 21 blocos em resume encadeado | ~12 (zero01 a zero17) |
| Iterações acumuladas na política | ~21 700 | ~6 300 na zero17; as outras, 1–3 mil cada, descartadas |
| 1º ciclo completo no play | `ckp9100` (~it 9100) | `model_5800` |
| Como as regras entraram | uma a uma, sobre política que andava e pegava (limite de junta exponencial no bloco18, limite de pelve no bloco20, `forma_postural` no bloco21–22) | todas desde a it 0: freio, limite de junta "do zero", cadeia completa |

**Causas.**
1. **Tempo de refino (MEDIDO, git e plays).** A linha bloco também teve os defeitos da zero:
   "anda agachado com a caixa", "caixa a 1,02 m", "BOTAR com tronco a 65°", "perna girando"
   (memórias da época). Ela os corrigiu ao longo de ~12 mil iterações, um conserto por bloco,
   sem perder o que já sabia. Por iteração, a zero17 não está atrás: completou o ciclo antes.
2. **Recomeçar joga fora o aprendizado (MEDIDO, git).** Cada mudança do conjunto de
   recompensas virou um run do zero (zero05 a zero17 em 9 dias); nenhum chegou à fase de
   refino. A linha bloco mudou a recompensa 21 vezes e sempre retomou.
3. **Armadilhas do começo do treino que a linha bloco não enfrentou (fatos MEDIDOS em
   memórias; a causa é HIPÓTESE).**
   - O freio desde a it 0 derruba o std da ação (1,0 → 0,61 na it 200 no molde; ~0,45 com
     o freio −15) e corta a exploração antes de a pega existir.
   - O limite de junta do zero com terminação no batente matava a manipulação: na zero03 a
     cintura ia ao batente em 7 passos e o PEGAR nunca treinava.
   - Estátua, agachado de antecipação (seção 1) e penhasco do BOTAR (seção 2) são mínimos
     locais que prendem uma política que ainda não sabe a tarefa.

**Prova parcial (MEDIDO, seção 5).** Com a recompensa do HEAD, o `model_21700` mantém a
pega e a postura e aprende o BOTAR até 0,15 m em ~1500 iterações. A recompensa atual
sustenta e refina uma política competente, mas não a produz a partir do zero.

**Não medido.** Uma linha zero com a mesma sequência de afrouxamento da linha bloco não foi
rodada; sem ela, a causa 3 não se separa da causa 1.

**Para a v3.**
- Treino em degraus com resume, de propósito: cada degrau abre uma tarefa com regras
  frouxas; as restrições entram quando a tarefa já fecha.
- Mudar a recompensa sem recomeçar, sempre que observação e ação permitirem (o resume do
  21700 atravessou 37 commits).
- Restrições de segurança (freio, limite de junta) em currículo desde o início, subindo com
  o fecho da tarefa (decisão 4 da v3), e não cheias na it 0.

## 5c. Freio por nível em catraca degrada a manipulação

**Observado (MEDIDO, log do `21700_zero_teste`, its 23700–24393).** O freio por env ficou
parado em ~1,75 degrau até a it ~23 750 (o contador entre degraus recomeça a cada sessão) e
subiu em linha reta até ~3,0.

| | it ~23 700 | its 24 351–24 361 | its 24 383–24 393 |
|---|---|---|---|
| `freio_degrau` | ~1,75 | ~2,93 | ~3,00 |
| `nivel` | ~3,42 (pico) | 3,08 | 3,05 |
| `fell_over` | ~0,4 | ~0,77 | ~1,33 |
| `s_B` / `s_C` | ~0,87 / ~0,53 | 0,82 / 0,39 | 0,78 / 0,41 |
| `mao_na_pega` / `velocidade_de_junta` | 0,83–0,86 / 2,03–2,10 | 0,79 / 1,85 | 0,81 / 1,87 |

A postura não mudou (pelve 0,74 m, std 0,54). O BOTAR resistiu: fecha 87–94% das C que
chegam a ele. A perda veio das quedas e da cadeia B.

**Mecanismo (CÓDIGO, `curriculo.py`, função `nivel`).** O freio de cada env sobe um degrau
quando o `nivel` do env está acima dele, e nunca desce; o `nivel` sobe e desce com o sucesso.
Laço: o freio sobe → as quedas sobem e o sucesso cai → o `nivel` cai → o freio fica, e o
preço passa do que o nível pede. A média do freio (3,07) passou a do `nivel` (3,05) na it
24 393. É o D15 do relatório de 01/10 (`2026-10-01-monotonia-do-curriculo.md`).

**Plays (dono, 05/10).** O `model_24350` tem movimentação melhor que o `model_23800`, mas o
punho E gira (seção 3b).

**Para a v3.** Toda restrição em nível (freio, preço do aperto, preço do punho) tem de poder
descer com o nível da tarefa, ou ter teto abaixo do que quebra a tarefa. A decisão 4 validou o
freio por nível; ela precisa da regra de descida.

## 6. Instrumentação

- O registrador chamava a observação com `giro_w` = 0: o ator ficava cego ao tombo e todo
  tombo de caixa em play era artefato. CORRIGIDO 01/10 (sem commit): o `giro_w` segue
  `comando.py:2102-2173`, e o CSV ganhou `giro_ang`, `alvo_x/y/z` e `ckpt` (nome:sha256).
- Diferenças que restam entre o play e o treino: a espera do roteiro publica o elo
  seguinte; o alvo do BOTAR é recalculado a cada passo pela pose da base (no treino ele
  congela na abertura); o roteiro `c` reusa a laje da pega, o que cria alvo lateral depois
  que o corpo gira.
- Dois arquivos `model_5000.pt` e `model_5000(1).pt` geravam CSVs indistinguíveis: daí a
  coluna `ckpt`.
- O fecho, o apoio e a força de contato não aparecem no play.

**Para a v3.** Teste de paridade play × treino da observação desde o dia 1; CSV que se
identifica sozinho (checkpoint, commit, alvo, giro); métricas sem peso de funil por cadeia e
de terminação por elo desde a it 0 (hoje não se sabe em que elo a caixa é largada).

## 7. Princípios confirmados nesta sessão

| Princípio | Evidência |
|---|---|
| Observação igual + futuro diferente = antecipação | agachado na espera e no ANDAR parado |
| Termo com derivada ~0 no ponto de operação é canal morto | `pose` +0,04/s por 0,1 rad; `precise_pos` 0,07/s por cm a 0,30 m |
| Fecho sem caminho pago é penhasco; o crítico só aprende o que visita | BOTAR plano até a laje rasa |
| DR sem rota legítima vira compensação na junta mais barata | `hip_yaw` a 22–30°, abaixo da rampa |
| O caminho de aprendizado importa tanto quanto o ótimo | 21700 contra a linha zero, mesma recompensa |
| Agente cego erra a conclusão sem revisão | um agente concluiu "agachar é do ANDAR parado"; a observação idêntica prediz exatamente isso |

## 8. Abertos

1. Sonda do crítico: quanto rende chegar agachado ao PEGAR (dimensiona o termo de pé na
   espera).
2. Em que elo acontecem o `caixa_largada` e o `fell_over` da zero17.
3. O tombo da pega na linha zero com o registrador novo.
4. O `21700_zero_teste` na fase 4 e com o freio cheio: plays `parado:ANDAR:6.0`,
   `carregar` e `botar`.
5. Causa do CARREGAR lento, girando e agachado.
6. **A DISCUTIR MELHOR (dono, 02/10): o currículo de profundidade do BOTAR (R4b) não
   prioriza os casos difíceis.** A profundidade é `a = clamp(s_C / 0,50, 0, 1)`
   (`comando.py:1870-1876`). Com o `s_C` em ~0,50, a laje sai uniforme do chão (0,04 m) a
   ~0,64 m e fica nisso: o R4b se equilibra em `s_C` ≈ 0,50 (MEDIDO, its 23008–23253 do
   `21700_zero_teste`) e nunca dá mais fatia à laje que falha. Um `s_C` único mistura
   todas as alturas, e o fracasso da laje baixa se dilui no sucesso da alta. Hoje o topo
   médio das chegadas (0,33 m) e dos sucessos (0,34 m) quase empata, então a falha ainda
   não aparece, mas a média não vê a cauda. Pontos para a discussão: sucesso medido por
   faixa de altura; sorteio que pese a faixa que falha; se o "nível por tarefa" da
   decisão 3 resolve isto ou só muda o nome.

7. **A DISCUTIR (dono, 06/10): a soltura arrasta a caixa.** Medido no controlador autônomo
   (cenário H, `model_24800`): depois do fecho do BOTAR a política se levanta ainda segurando a
   caixa e a arrasta 7,3 / 11,4 / 1,2 cm nas três tarefas. O treino não cobra isso: o
   `caixa_largada` só termina fora de `raio_solta` (0,18 m). Correção no treino: depois do fecho
   do BOTAR, apertar o raio para ~3 cm, ou pagar pela caixa parada durante a cauda. Em aberto:
   entrar desde o começo da v3 ou só no refino, quando a pega e a bota já estão aprendidas.
8. **Caixa pequena tomba no PEGAR (06/10, controlador autônomo):** meia aresta 0,07 e 0,08 m
   tombam ~70° com 0,5 e 1 kg; 0,09 e 0,10 m funcionam. O limite é o tamanho, não o peso. A
   faixa do treino é 0,07–0,13 m: conferir a pega na borda de baixo.

## Arquivos

- Plays: `5000_andar_parado.csv`, `5350_andar_parado.csv`, `5800_andar_parado.csv`,
  `5000_botar.csv`, `5350_botar.csv`, `6050_botar.csv`, `6050_carregar.csv`,
  `21700_c.csv`, `21850_c.csv` (raiz do repo; o `5800_botar.csv` foi regravado por outra
  sessão com outra laje).
- Memórias: `g1-limpo-espera-sem-agacha-o-andar-parado`, `g1-limpo-botar-esquecido-na-fase-4`,
  `g1-limpo-pendencias`, `g1-limpo-registrador-cego-ao-giro`.
- Registrador: `registra_juntas.py` (diff sem commit: `alvo`, `ckpt`, `giro_w`, `giro_ang`).
