# G1 v3 — interface com o robô real

Estado: 14/10, a partir da spec `specs/g1-v3.md` (§6, §13, §19) e da decisão do dono de 14/10 (entrada
mínima do ator). Nenhum código foi escrito. Cada item tem uma marca:

- **DEFINIDO**: a spec fixa o valor; o treino e o robô usam o mesmo.
- **A DEFINIR**: a spec não fixa; a linha traz a proposta. Tem de ser decidido ANTES da Parte 1, porque
  o treino grava o formato no modelo e o robô real precisa reproduzir o mesmo formato.

Regra (dono, 14/10): o modelo recebe só o que o robô real produz sem cálculo de tarefa fora do modelo:
os comandos do controlador de alto nível e os sensores. Tudo que o treino dá ao modelo e o robô real
não reproduz torna o modelo inútil no robô.

## 1. Os módulos no robô real

```
 controlador de alto nível  (eventos; decide a tarefa)
   │ tarefa · velocidade · alvo (no mundo)        ▲ status: pose, caixas vistas, tarefa concluída/falhou
   ▼                                               │
 driver da política, 50 Hz ────────────────────────┘
   ├─ lê:  LowState (29 juntas, IMU da pelve) ─────────────────────────────┐
   ├─ lê:  estimador da caixa ◄─ câmera D435i (RGB + depth) + odometria     │
   ├─ lê:  filtro de elevação ◄─ lidar MID-360 + odometria + cinemática ────┤
   ├─ monta a observação (1 447 números, §2) ───────────────────────────────┘
   ├─ roda o ator (rede 512-256-128, ELU)
   └─ escreve: LowCmd (29 alvos de junta + kp + kd, §3)
 odometria: ESKF com lidar + IMU (pose da base no mundo, para o estimador, o filtro e o controlador)
```

O driver, o estimador, o filtro e a odometria rodam NO ROBÔ. O controlador de alto nível só manda
ordens e lê o status.

## 2. Entradas do ator (1 447 números por passo de 20 ms)

### 2.1 Os blocos, na ordem do vetor

O histórico é por bloco: cada bloco guarda os últimos 5 passos, do mais velho ao mais novo, e os
blocos ficam um depois do outro (é o layout do `ObservationManager` do mjlab com `history_length` 5).
O mapa não tem histórico. No primeiro passo, os 5 lugares do histórico recebem a mesma leitura.

| # | bloco | por passo | × 5 | índices | escala | fonte no robô real | marca |
|---|---|---|---|---|---|---|---|
| 1 | velocidade angular da base | 3 | 15 | 0–14 | × 0,2 | giroscópio do IMU da pelve (`LowState.imu_state.gyroscope`; o deploy da Unitree lê esse campo como o giro da raiz) | DEFINIDO |
| 2 | gravidade projetada | 3 | 15 | 15–29 | × 1 | `Rᵀ · (0, 0, −1)`, com `R` do quaternion do IMU | DEFINIDO |
| 3 | comando de velocidade (vx, vy, wz) | 3 | 15 | 30–44 | × 1 | controlador de alto nível | DEFINIDO |
| 4 | posição das juntas, menos `q_de_pé` | 29 | 145 | 45–189 | × 1 | `LowState.motor_state[i].q` | DEFINIDO |
| 5 | velocidade das juntas | 29 | 145 | 190–334 | × 0,05 | `LowState.motor_state[i].dq` | DEFINIDO |
| 6 | última ação (a crua, antes da escala 0,25) | 29 | 145 | 335–479 | × 1 | memória do driver | DEFINIDO |
| 7 | one-hot da tarefa | 6 | 30 | 480–509 | × 1 | controlador de alto nível | DEFINIDO |
| 8 | bloco da caixa e do alvo (§2.3) | 11 | 55 | 510–564 | × 1 | estimador da caixa + controlador | DEFINIDO |
| 9 | mapa de elevação (§2.4) | 882 | — | 565–1446 | × 1 | filtro de elevação | DEFINIDO |

O ruído da tabela da §19.1 da spec existe só no treino. No robô real, os valores entram crus.

### 2.2 Os blocos 1 a 7

**Referencial da base:** o da pelve (`pelvis`), que é a raiz do robô no MJCF. O IMU da pelve fica
nesse corpo.

**Comando de velocidade** (bloco 3), no referencial da base: vx em m/s (para trás até −1,0; para a
frente até +1,5), vy em m/s (±1,0), wz em rad/s (±1,0). Nas tarefas PEGAR e BOTAR o comando é zero.
O controlador nunca pede o máximo (enunciado §4.1).

**Ordem das 29 juntas** (blocos 4, 5 e 6, e a saída da §3): a ordem dos motores do SDK da Unitree.

| índice | junta | `q_de_pé` (rad) | índice | junta | `q_de_pé` (rad) |
|---|---|---|---|---|---|
| 0 | left_hip_pitch | −0,029 | 15 | left_shoulder_pitch | 0,200 |
| 1 | left_hip_roll | −0,005 | 16 | left_shoulder_roll | 0,200 |
| 2 | left_hip_yaw | 0,001 | 17 | left_shoulder_yaw | 0,000 |
| 3 | left_knee | 0,162 | 18 | left_elbow | 0,600 |
| 4 | left_ankle_pitch | −0,134 | 19 | left_wrist_roll | 0,000 |
| 5 | left_ankle_roll | 0,005 | 20 | left_wrist_pitch | 0,000 |
| 6 | right_hip_pitch | −0,029 | 21 | left_wrist_yaw | 0,000 |
| 7 | right_hip_roll | 0,005 | 22 | right_shoulder_pitch | 0,200 |
| 8 | right_hip_yaw | −0,001 | 23 | right_shoulder_roll | −0,200 |
| 9 | right_knee | 0,162 | 24 | right_shoulder_yaw | 0,000 |
| 10 | right_ankle_pitch | −0,134 | 25 | right_elbow | 0,600 |
| 11 | right_ankle_roll | −0,005 | 26 | right_wrist_roll | 0,000 |
| 12 | waist_yaw | −0,005 | 27 | right_wrist_pitch | 0,000 |
| 13 | waist_roll | 0,000 | 28 | right_wrist_yaw | 0,000 |
| 14 | waist_pitch | 0,000 | | | |

`q_de_pé` é a pose 0 de `g1_limpo/ik/ref_v3_de_pe.npz` (spec §6.3); a tabela mostra 3 casas, e o
driver usa o valor exato do arquivo.

**One-hot** (bloco 7), nesta ordem: ANDAR, PEGAR, CARREGAR, BOTAR, REORIENTAR, LEVANTAR. Os dois
últimos ficam sempre em 0 na v3.

### 2.3 O bloco da caixa e do alvo (11 números)

| posição | dado | unidade e referencial | fonte no robô real | marca |
|---|---|---|---|---|
| 0–2 | centro da caixa | m, referencial da base | estimador: detector na imagem RGB + depth, ancorado no mundo pela odometria | DEFINIDO |
| 3–5 | orientação da caixa | vetor de rotação (eixo × ângulo, rad) da caixa contra a base, reduzido pela simetria de 90° do cubo: das 24 rotações equivalentes, a de menor ângulo | estimador: algoritmo de orientação do `g1-box-vision` | DEFINIDO |
| 6 | meia aresta da caixa | m (0,07 a 0,13) | estimador: tamanho medido pelo depth | DEFINIDO |
| 7 | idade da última leitura da caixa | s, cortada em 2 s | estimador | DEFINIDO |
| 8–10 | alvo | m, referencial da base | controlador manda o alvo no mundo; o driver converte para a base a cada passo pela odometria | DEFINIDO; zero no ANDAR e no CARREGAR |

O alvo, por tarefa:

| tarefa | alvo | os 11 números |
|---|---|---|
| ANDAR | — | todos zero (spec §6.1, [C13]) |
| PEGAR | o centro da caixa escolhida | caixa + alvo |
| CARREGAR | zero | caixa + alvo zero |
| BOTAR | onde o centro da caixa deve ficar, apoiada no destino | caixa + alvo |

**A caixa nas mãos — DEFINIDO (dono, 14/10).** A caixa vem SEMPRE da câmera e do depth, também
com a caixa nas mãos. Quando a câmera não vê a caixa (braços na frente, caixa perto demais para o
depth, de 0,175 a 0,28 m), o estimador mantém a última leitura ancorada no mundo e a idade cresce. O
modelo aprende a lidar com a leitura velha. A spec §13 dizia "caixa na mão pela cinemática das palmas
depois de `pegou`"; isso saiu, porque `pegou` é um evento do simulador que o robô real não mede.
Consequência: no treino, o estimador simulado tem de reproduzir a visibilidade real da câmera D435i
(posição na cabeça, campo de visão, alcance mínimo do depth, oclusão pelos braços), porque é ela que
decide quando a leitura envelhece.

### 2.4 O mapa de elevação (882 números)

**Como ele é gerado, passo a passo** (plano de percepção §0; o mesmo código roda no treino e no robô):

1. **Pontos.** O lidar MID-360 fica na cabeça. No robô real, o SDK da Livox entrega cerca de 20 mil
   pontos a cada 0,1 s. No treino, um lidar simplificado lança 1 024 raios a cada 0,1 s, da mesma
   pose, e o raycast vê o chão, o terreno, a caixa e o próprio robô.
2. **Referencial.** Cada ponto vai do sensor para a base pela cinemática das juntas medidas (o
   lidar está no tronco, que gira com a cintura), e da base para um referencial alinhado à gravidade
   pelo IMU. O rumo segue o da base.
3. **Zero da escala.** A altura de cada ponto é medida contra o pé mais baixo do robô (cinemática):
   chão plano = 0, degrau de 15 cm = +0,15, mesmo com o robô agachado.
4. **Filtro de Kalman por célula**, num mapa interno de 5 × 5 m com células de 10 cm:
   - a cada 20 ms, o mapa se desloca com o movimento da base (odometria), e a incerteza de cada
     célula cresce com o deslocamento;
   - a cada 0,1 s, os pontos novos atualizam a altura e a incerteza das células onde caem.
5. **Recorte.** O modelo recebe as 21 × 21 células de 2 × 2 m em volta da base: a altura (m, cortada
   em ±1 m) e a confiança (0 = nunca vista, 1 = certa).

| item | valor | marca |
|---|---|---|
| células | 21 × 21, 10 cm, centradas na base | DEFINIDO |
| eixos | x para a frente, y para a esquerda, alinhados ao rumo da base; roll e pitch fora | DEFINIDO |
| altura | ponto de contato − pé mais baixo, m, cortada em ±1 | DEFINIDO |
| confiança | de 0 a 1, da variância da célula | DEFINIDO (a fórmula: A DEFINIR na fase 1) |
| ordem no vetor | as 441 alturas, depois as 441 confianças; cada uma por linha, de x = +1,0 a −1,0 m, e dentro da linha de y = +1,0 a −1,0 m | DEFINIDO |
| escala | × 1 | DEFINIDO |

**O que a simulação mostra** (14/10, `scratchpad .../doc/img/render.py`, um degrau de 0,15 m à frente,
robô andando 1,0 m): o lidar da cabeça não vê o chão a menos de ~0,9 m em nenhuma direção. Sem caixa,
o recorte enche pela memória: 314 de 441 células vistas, e 205 de 210 na metade da frente; o chão
embaixo e atrás do robô fica sem leitura. Com a caixa de 26 cm, só 82 de 210 células da frente têm
leitura depois de 1,0 m: a frente fica desconhecida nos primeiros ~1,2 m do CARREGAR. A plataforma
plana de 2 m em volta da laje (spec §12) cobre esse trecho.

### 2.5 Exemplo de um passo

Robô de pé, parado, diante de uma superfície de 0,30 m com uma caixa de 20 cm a 0,55 m à frente. O
controlador acabou de mandar PEGAR. A gravidade, o bloco 8 e o mapa foram calculados na cena do
treino (frame da base = pelve a 0,79 m); os valores dos blocos 1, 4 e 5 são ilustrativos.

| bloco | valor |
|---|---|
| 1 velocidade angular × 0,2 | (0,002; −0,004; 0,000) — giroscópio (0,01; −0,02; 0,00) rad/s |
| 2 gravidade | (0,00; 0,00; −1,00) |
| 3 comando | (0; 0; 0) |
| 4 juntas − `q_de_pé` | 29 números perto de 0, ex.: joelho esquerdo 0,180 − 0,162 = 0,018 |
| 5 velocidade × 0,05 | 29 números perto de 0 |
| 6 última ação | os 29 números que o ator devolveu no passo anterior |
| 7 one-hot | (0; 1; 0; 0; 0; 0) |
| 8 caixa e alvo | centro (0,55; 0,00; −0,39), orientação (0; 0; 0,05), meia aresta 0,10, idade 0,03 s, alvo (0,55; 0,00; −0,39) |
| 9 mapa | ex.: a coluna central (y = 0), de x = +1,0 a −1,0, sem caixa, depois de andar até o degrau: 0,15 0,15 0,15 0,15 0,14 0,12 0,04 0 0 0, e embaixo e atrás do robô sem leitura (altura 0, confiança 0) |

E no BOTAR: robô de pé com a caixa no peito, destino numa superfície de 0,45 m a 0,55 m à frente:
alvo = (0,55; 0,00; −0,24); o bloco da caixa traz a caixa nas mãos, (0,25; 0,00; +0,11).

Cada bloco 1 a 8 entra 5 vezes no vetor (os últimos 5 passos).

## 3. Saída do ator e o que o driver manda aos motores

| item | valor | marca |
|---|---|---|
| saída da rede | `a`, 29 números, sem clip | DEFINIDO |
| alvo de cada junta | `q* = q_de_pé + 0,25 · a` (rad), na ordem da §2.2 | DEFINIDO |
| `LowCmd`, por motor | `q = q*`, `dq = 0`, `tau = 0`, `kp` e `kd` da junta | DEFINIDO |
| frequência | 50 Hz (passo de física 5 ms, decimação 4); o PD do motor roda dentro do motor | DEFINIDO |
| atraso | o treino sorteia de 0 a 1 passo (20 ms) desde o início; o driver real deve ficar abaixo de 20 ms entre ler o `LowState` e escrever o `LowCmd` | DEFINIDO no treino; medir no real |
| `kp`, `kd` | por junta, abaixo | DEFINIDO (dono, 14/10) |

**Ganhos do PD por junta** (receita da Unitree: `UNITREE_G1_29DOF_CFG` em
`unitree_rl_lab/assets/robots/unitree.py`, commit `4960b84`, o mesmo usado pela tarefa de velocidade).
O treino e o robô usam os mesmos valores.

| juntas | `kp` | `kd` | motor |
|---|---|---|---|
| `hip_pitch`, `hip_yaw` (as duas pernas) | 100 | 2 | 7520-14.3 |
| `hip_roll` (as duas pernas) | 100 | 2 | 7520-22.5 |
| `knee` | 150 | 4 | 7520-22.5 |
| `ankle_pitch`, `ankle_roll` | 40 | 2 | 5020-16 |
| `waist_yaw` | 200 | 5 | 7520-14.3 |
| `waist_roll`, `waist_pitch` | 40 | 5 | 5020-16 |
| `shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow`, `wrist_roll` | 40 | 1 | 5020-16 |
| `wrist_pitch`, `wrist_yaw` | 40 | 1 | 4010 |

Atenção: o arquivo de deploy da política v0 da Unitree
(`deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml`) traz outros valores:
cintura com `kp` 200 nos três eixos e braços com `kd` 10. Ele não foi gerado da configuração de
treino do mesmo commit. A v3 usa a tabela acima no treino e no robô, e o driver real lê os ganhos de
um arquivo exportado junto do modelo, nunca de uma tabela escrita à mão.


## 4. O controlador de alto nível

### 4.1 O que ele manda (as entradas da política que vêm dele)

| dado | quando muda | formato |
|---|---|---|
| tarefa | num evento (ex.: chegou diante da caixa) | uma das 4: ANDAR, PEGAR, CARREGAR, BOTAR |
| velocidade | no ANDAR e no CARREGAR, quando quiser | vx, vy, wz, no referencial da base; zero no PEGAR e no BOTAR |
| alvo | no comando PEGAR ou BOTAR | um ponto no mundo (m), que o driver converte para a base a cada passo |

O controlador NÃO calcula ritmo de passo, postura, posição de mão nem etapa da pega (dono, 14/10).
O laço de rumo (k = 2) do driver é do controlador: ele corrige o wz para manter a direção (spec §15,
cenário "rumo").

### 4.2 O que ele precisa receber do robô (status) — DEFINIDO (dono, 14/10)

**Detector de fim de tarefa: proximidade de uma pose de referência, estável por um tempo mínimo.**
O detector usa só sinais que o robô real tem: a cinemática das juntas medidas (pés, pelve, palmas) e
o IMU. Ele compara poucas grandezas com a pose de referência, e não as 29 juntas, porque a IK é
referência e o robô pode chegar a uma pose parecida com juntas diferentes.

| aviso | pose de referência | grandezas e tolerâncias | tempo estável |
|---|---|---|---|
| PEGAR concluído | de pé com caixa, para o tamanho da caixa (spec §8.1) | altura da pelve sobre o pé mais baixo ±3 cm; tronco ±5°; cada palma, no referencial da base, a até 5 cm da palma da referência | 0,5 s, com as juntas quase paradas |
| BOTAR concluído | de pé sem caixa (`ref_v3_de_pe.npz`, pose 0) | as mesmas, com as palmas da pose de pé sem caixa | 0,5 s, com as juntas quase paradas |

"Juntas quase paradas": a média de |velocidade| das 29 juntas na janela fica abaixo de um limiar
(proposta: 0,2 rad/s; a calibrar nos plays). As tolerâncias de pelve e tronco são as do B4 da spec
§8.2.

Os outros avisos:

| status | regra, só com sinais reais |
|---|---|
| caixa perdida | a câmera vê a caixa no chão ou longe das mãos durante PEGAR, CARREGAR ou BOTAR |
| queda | inclinação do IMU acima de 46° (ANDAR, CARREGAR) ou 70° (PEGAR, BOTAR) |
| pose e caixas | pose da base no mundo (odometria) e a lista de caixas vistas, com posição e tamanho |

**Paridade com o treino (dono, 14/10):** o treino troca PEGAR → CARREGAR e encerra o BOTAR pelo
MESMO detector, e não pelo fecho do P4 e do B4. Depois do aviso, a troca espera um atraso sorteado
de 0 a 1 s; o modelo aprende a ficar estável enquanto a ordem nova não chega. No robô real, o
controlador pode trocar no aviso ou um pouco depois. O detector roda fora do modelo e pode mudar sem
treino novo, desde que nunca avise antes do treino.

**Limite conhecido:** a pose sozinha não garante que a caixa está nas mãos. Se a caixa cair, a regra
"caixa perdida" (câmera) avisa. Uma checagem de carga nos braços (`LowState.tau_est`) pode entrar
depois, se os testes mostrarem aviso de pega sem caixa.

## 5. O que o modelo NÃO recebe (e por quê)

| dado | onde fica |
|---|---|
| força nas mãos, massa da caixa, atrito | só no crítico; o robô real não mede |
| etapa da pega (P0–P4, B1–B4) e a referência dela | só no crítico e na recompensa (dono, 14/10) |
| relógio da marcha | saiu (dono, 14/10); o robô escolhe a cadência |
| velocidade linear da base | só no crítico (receita da Unitree) |
| contato com a laje, apoio da caixa | só na recompensa e nas terminações |

## 6. Decisões em aberto, em ordem

1. FEITO (14/10): a caixa vem sempre da câmera (§2.3).
2. FEITO (14/10): detector por proximidade de pose, estável por 0,5 s, igual no treino (§4.2).
3. FEITO (14/10): orientação como vetor de rotação reduzido pela simetria do cubo (§2.3).
4. FEITO (14/10): alvo zero no ANDAR e no CARREGAR (§2.3).
5. FEITO (14/10): idade em segundos, cortada em 2 s (§2.3).
6. FEITO (14/10): escala × 1 no bloco da caixa e no mapa; ordem das células (§2.3, §2.4).
7. FEITO (14/10): `kp` e `kd` por junta, da configuração de treino da Unitree (§3).
8. FEITO (14/10): o IMU é o `imu_state` do `LowState`, lido como o da raiz (pelve), como no deploy
   da Unitree (§2.1).

Ficam para a fase 1 da percepção: a fórmula da confiança do mapa. Fica para os plays: o limiar de
"juntas quase paradas" do detector (§4.2).
