# Plano — a grade do MID-360 como percepção da v3

Data: 2026-10-02. Substitui o plano `docs/planos/2026-09-24-grade-de-ocupacao-lidar.md` para a v3.
Nenhum código foi escrito. Este plano vai ao agente que implementa a v3 (`specs/g1-v3-comportamento.md`).

## 0. Atualização de 06/10: mapa de elevação para o terreno

DECIDIDO pelo dono (06/10): a entrada da política passa a ser um MAPA DE ELEVAÇÃO 2,5D, e não a
grade 3D de ocupação da §3. O motivo é generalizar a locomoção para rampas e degraus, no ANDAR. O contrato da §2 (a mesma função pura no treino, no sim do real e no robô), o padrão de
raios da §5, o filtro do corpo da §7 e a chave da §8 continuam. A §3, a §4 e a §6 ficam substituídas
por esta seção na entrada da política.

**O que a receita da Unitree faz** (commit `4960b84`, conferido em 06/10): nada. A cena declara um
`height_scanner` (raios de 20 m acima do `torso_link`, grade de 1,6 × 1,0 m a 10 cm), mas o termo de
observação está comentado no ator e no crítico, e o terreno só tem o subterreno plano. A política do
G1 é cega. Terreno com percepção é desenho nosso, um desvio da receita (spec §19).

**A abordagem.** A política vê só o mapa em volta dela, nunca o lidar.

```
treino:      raios de um lidar simplificado, na pose do MID-360 real ──┐
sim do real: sensores/lidar.py ─────────────────────────────────────────┼─► pontos ─► MESMO filtro ─► mapa ─► política
robô real:   SDK da Livox ─────────────────────────────────────────────┘        (Kalman por célula + odometria)
```

- **A zona cega não é desenhada.** O MID-360 fica de cabeça para baixo a ~1,25 m, e não vê o chão a
  menos de ~1,0–1,3 m (medido em 23/09). No real, o mapa enche essa zona com as leituras de quando
  ela estava à frente. No treino, os raios saem da mesma pose e passam pelo MESMO filtro: a zona
  cega, a oclusão pelo corpo e o atraso aparecem sozinhos, iguais aos do real. Um mapa perfeito no
  treino ensinaria a política a confiar em célula que no real está vazia ou velha (ex.: o chão atrás
  do robô quando ele anda de costas).
- **Alternativa mais simples, NÃO escolhida:** mapa perfeito de raios vindos de cima (o
  `height_scanner` da Unitree, o padrão do Isaac Lab e do legged_gym) estragado por um modelo de erro
  escrito à mão (deslocamento, deriva, células faltando), como em Miki et al. (2022, ANYmal). É mais
  barato no simulador, mas o modelo de erro é chute até ser medido, e não reproduz a zona cega da
  cabeça do G1.

**O mapa:**

| item | valor (ponto de partida) |
|---|---|
| área | 2 × 2 m centrada no robô (raio de 1 m, dono 06/10) |
| referencial | gira só com o rumo da base; alinhado à gravidade (roll e pitch saem) |
| célula | 10 cm: 21 × 21 = 441 células; 5 cm (41 × 41 = 1 681) só se a medida de custo e de erro pedir |
| canais por célula | altura e confiança: 882 canais com 10 cm |
| altura | a altura dos PONTOS DE CONTATO dos raios com a superfície que caem na célula, com o ZERO da escala no nível do pé mais baixo (abaixo); cortada em ±1 m |
| confiança | de 0 a 1, da variância da célula; célula nunca vista = 0 |
| histórico | sem o histórico de 5 passos da §19.1 da spec: o filtro já guarda o tempo |

**A medida e o zero da escala** (esclarecido com o dono, 06/10). A MEDIDA vem do lidar: cada raio
toca uma superfície, e a altura desse ponto de contato é o que entra na célula. O ZERO da escala é o
nível do pé mais baixo do robô: valor da célula = altura do ponto de contato − altura do pé. A altura
do lidar sobre o pé sai da cinemática das juntas medidas (§2), nunca do `z` verdadeiro do simulador.
Motivo: o lidar fica na cabeça, e a cabeça sobe e desce. Com chão plano e um degrau de 15 cm:

| zero da escala | robô de pé | robô agachado 20 cm |
|---|---|---|
| o próprio lidar | chão −1,25 m; degrau −1,10 m | chão −1,05 m; degrau −0,90 m |
| o pé mais baixo | chão 0; degrau +0,15 m | chão 0; degrau +0,15 m |

Com o zero no lidar, o chão "sobe" quando o robô agacha, sem nada mudar no mundo. Com o zero no pé,
o mapa só muda quando o terreno muda.

**O filtro (a função pura, `torch`/`numpy`, sem simulador):**

1. **Predição, a cada passo da política (50 Hz):** transporta o mapa pelo deslocamento da base no
   plano (x, y, rumo), e a variância de cada célula cresce com o deslocamento (erro da odometria) e
   com o tempo (a célula envelhece até "não vista").
2. **Atualização, a cada leitura do lidar (10 Hz):** os pontos (sensor → base → gravidade) caem nas
   células; cada célula faz a atualização de Kalman da altura com a variância da medida (alcance com
   σ de 2 cm e ângulo com σ de 0,15°, a mesma da §2).
3. **Saída:** altura e confiança por célula, no frame da §0.

É o desenho do `elevation_mapping` do ETH (Fankhauser), reduzido ao que a política precisa.

**O lidar simplificado no treino:** R = 128 a 256 raios por leitura, a 10 Hz, tirados do padrão do
MID-360 (§5), na pose `T_BASE_SENSOR` (§2), com o robô incluído no raycast (§7). A odometria do
treino é o deslocamento verdadeiro mais ruído; o ruído cresce no refino até o erro do ESKF do
`lidar_sim` (ATE de 1,5 cm na volta de 4 m), como o estimador da caixa (spec §13).

**O terreno:** o gerador do mjlab (`mjlab/terrains/config.py`: escada em pirâmide, escada invertida,
rampa, escada aleatória, chão irregular), como um nível a mais do ANDAR, depois dos níveis de
velocidade no plano (dono, 06/10).

**O CARREGAR no terreno (dono, 06/10: "em algum momento"; nível futuro).** MEDIDO em 06/10
(`~/Downloads/ik_v3/sombra/sombra_caixa.py`, mapas `sombra_*.png`): raios do MID-360 na pose do URDF,
robô de pé com a caixa no peito (pose `theta-30` da IK), contra as geometrias de colisão do robô, a
caixa e o chão. As malhas visuais e a casca da cabeça saem do raio, porque no real o sensor olha por
uma janela.

| robô | chão visto mais perto | frente ±30°, 1–2 m | frente ±30°, 2–3 m | sombra da caixa no chão |
|---|---|---|---|---|
| sem caixa | 0,89 m | 100% | 100% | — |
| caixa de 14 cm | 0,89 m | 71% | 100% | 1,09–1,42 m, ±16° |
| caixa de 20 cm | 0,89 m | 40% | 100% | 1,08–1,76 m, ±26° |
| caixa de 26 cm | 0,89 m | 9% | 88% | 1,07–2,17 m, ±30° |

Os lados e a trás não mudam com a caixa. Consequência: andando para a frente com a caixa grande, o
chão some da vista a ~2,2 m, e não a ~0,9 m. A memória do mapa precisa guardar um degrau por
~2,2 m de caminho em vez de ~0,9 m: 4,4 s a 0,5 m/s, contra 1,8 s sem caixa. A odometria erra pouco
nesse trecho (ESKF: 1,5 cm em 4 m), então a sombra cabe no mesmo filtro, com duas regras:
- o raycast do treino inclui a caixa na mão (§7): a sombra aparece no treino como no real;
- a variância da célula cresce sobretudo com o DESLOCAMENTO, e devagar com o tempo: uma meia-vida
  de 1 s (a da §6 antiga) apagaria o degrau antes de o robô chegar a ele.
Não medido: a D435i olha para baixo e para a frente, justamente onde a caixa fica; com a caixa no
peito ela deve ver pouco do chão.

**A laje e a estante:** o mapa de elevação mostra o topo da laje como um degrau de altura, e serve
também à manipulação. A estante (futura) tem teto e vãos, que a elevação não representa: ela volta à
grade 3D, ou a uma camada a mais.

**Quando entra: DECIDIDO pelo dono (06/10).** O mapa entra DESDE O COMEÇO do treino, ligado em
TODOS os envs e em todas as tarefas; a chave `PERCEPCAO_LIDAR` fica ligada (substitui o "desligada
na primeira v3" de 02/10, item 1 da §1). O terreno entra como um nível a mais do ANDAR, com os
obstáculos de `mjlab/terrains/config.py`; no Isaac Lab, o gerador equivalente, que pode mudar os
tipos e parâmetros. O CARREGAR fica no plano. A entrada do ator passa de 605 para 1 487 canais
(spec §19.1); a fase 5c mede o custo antes de treinar.

**Portões que mudam** (fases da §9):
- Fase 1, sem simulador: um degrau de 0,15 m nos pontos sintéticos vira um salto de 0,15 m ± 2 cm no
  mapa; o Kalman converge em poucas leituras; transportar 0,10 m desloca o mapa uma célula; célula
  nunca vista fica com confiança 0; inclinar a base 10° não muda a altura do chão.
- Fase 6, a réplica: nas células vistas pelos dois caminhos, erro de altura de até 3 cm em 95% delas,
  e as mesmas células com confiança acima de 0,5 (concordância de pelo menos 0,90), no lugar do IoU.

## 1. O objetivo e as decisões do dono

A política ganha uma percepção do ambiente: o que existe em volta do robô, visto pelo MID-360 da
cabeça. Ela não recebe rótulo ("mesa", "prateleira"): ela vê a ocupação e decide sozinha (dono,
02/10). O oráculo da caixa (10 canais) continua; a percepção é do AMBIENTE, e não do objeto.

**A entrada da política é SÓ a grade** (dono, 02/10). A política nunca vê pontos, raios nem
distâncias. Cada lugar gera a grade com a sua fonte de pontos e a MESMA função:

```
treino:          raycast do mjlab ──┐
sim do real:     sensores/lidar.py ─┼──► pontos ──► mesma função ──► grade ──► política
robô real:       SDK da Livox ──────┘
```

O raycast é só o gerador da grade no treino, no lugar do lidar. Ele também dá a visibilidade de
graça: o raio para na primeira superfície, como o do lidar, então a grade do treino não marca o que o
lidar real não enxerga (chão embaixo da laje, face de trás de uma estante).

Decisões do dono (02/10):

1. **A grade entra na v3.** A v3 é um treino do zero, então a observação nova não custa uma linhagem extra.
   ATUALIZADO pelo dono, 06/10: a chave fica LIGADA desde o começo, em todos os envs, com o mapa de
   elevação da §0. (Antes, 02/10: a chave ficava DESLIGADA na primeira v3.) Com só a laje, a grade
   traz pouca informação nova e separa as tarefas pela cena (ANDAR sem laje, parado com laje e
   caixa). Ela entra quando o suporte variar ou o ANDAR treinar com obstáculos, por append no
   fim da entrada (`g1_poc/expande_checkpoint.py`). Detalhe: `vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md`.
2. **Remoção simples por uma chave.** Desligada, a v3 treina de novo do zero sem a grade. Não há
   estado "zerado" nem fração de envs cegos.
3. **Replicável no sim do real e no robô real.** O modelo treinado depende da grade; o sim do real
   (`volta.py`, `sensores/` no worktree `lidar_sim`) e o robô real têm de produzir a MESMA grade
   para a mesma cena e a mesma pose. Esta é a exigência central do plano; as seções 2 e 9 existem
   por causa dela.

## 2. O contrato de replicação

A grade sai de UMA função pura, igual nos três lugares. Cada entrada da função tem de existir no
robô real, e o treino tem de obtê-la pelo mesmo caminho do real, e não por um atalho do simulador.

| entrada | no robô real | no treino (v3) | no sim do real |
|---|---|---|---|
| pontos no frame do sensor | SDK da Livox, ~20 mil por varredura de 0,1 s | raycast do mjlab, R raios por leitura, padrão rotativo (§5) | `sensores/lidar.py` |
| roll e pitch da base | IMU (o ESKF da percepção) | quaternion da base + ruído de 0,5° | quaternion da base |
| altura do sensor sobre o chão | cinemática: das juntas medidas ao pé mais baixo | a MESMA cinemática, com as juntas do sim; NUNCA o `z` verdadeiro da base | a mesma cinemática |
| deslocamento da base entre passos | odometria do ESKF (fase 3 da percepção do `lidar_sim`) | deslocamento verdadeiro + ruído | o ESKF da fase 3 |
| poses dos elos (filtro do corpo) | cinemática direta das juntas medidas | poses do sim | poses do sim |
| caixa na mão | o estimador da caixa, quando existir | `pegou` do sim | `pegou` do sim |

Regras:

- **A função não importa simulador.** Só `torch` e `numpy`. Ela mora num arquivo do pacote da v3
  (proposta: `percepcao_lidar.py`), com zero import do pacote.
- **O sim do real e o robô real carregam o arquivo pelo caminho**, com
  `importlib.util.spec_from_file_location`, para não executar o `__init__` do pacote, que importa o
  mjlab. O `lidar_sim` é worktree do mesmo repositório: é o MESMO arquivo, nunca uma cópia.
- **Uma constante para a pose do sensor:** `T_BASE_SENSOR`, do `mid360_joint` do URDF rev 1.0
  (`torso_link`, `pos = (0,0002835; 0,00003; 0,428434)`, `rpy = (π; 0,0511; 0)`). O site do raycast
  do treino, a função e o `sensores/montagem.py` saem dela.
- **O mesmo ruído do sensor nos três lugares:** alcance com σ de 2 cm, ângulo com σ de 0,15° e 1%
  de perda (`sensores/tipos.py`, padrão de 24/09). No treino, o ruído entra nos pontos antes da função.

## 3. A grade (SUBSTITUÍDA pela §0 na entrada da política, 06/10; fica para a estante futura)

Um volume fixo em torno do robô, alinhado ao YAW da base: a frente da grade é a frente do robô.
Roll e pitch saem. A origem em z é o CHÃO, e não a pelve, para a grade não andar com o agachamento.

| eixo | faixa, no frame da base | células |
|---|---|---|
| x (frente) | −0,3 a +1,2 m, cubos de 0,15 m | 10 |
| y (esquerda) | −0,6 a +0,6 m, cubos de 0,15 m | 8 |
| z (altura) | uma camada de chão (−0,10 a 0,06 m), 11 camadas de 5 cm (0,06 a 0,61 m), 6 de 15 cm (0,61 a 1,51 m) | 18 |

Total: 10 × 8 × 18 = 1440 células. Índice linear `ix·144 + iy·18 + iz`.

Por que estas camadas, e não as de 24/09 (0,15 m em z):

- **A v3 desce a laje até 0,04 m.** Com camadas de 15 cm, toda laje abaixo de 0,15 m cai nas células
  do chão. As camadas de 5 cm até 0,61 m separam as lajes da v3, de 0,10 m para cima.
- **A camada do chão é grossa de propósito.** O ruído de 2 cm espalha os pontos do chão; com o
  corte em 0,06 m (3 σ), o chão não vaza para a camada de cima. A laje de 0,04 m fica na camada do
  chão: o lidar não a separa do chão com esse ruído. A política recebe a altura dela pelo alvo macro.
- O valor da célula é um escore de 0 a 1 (§6), e não um bit: ele também diz a idade da última batida.

Os tamanhos são ponto de partida. A fase 2 mede se 1,2 m à frente e 1,5 m de altura cobrem a laje
e uma prateleira ao alcance.

## 4. A função pura (SUBSTITUÍDA pelo filtro da §0, 06/10)

`pontos_para_grade(pontos, validos, roll, pitch, altura) → (E, 1440)`:

1. sensor → base, pela `T_BASE_SENSOR`;
2. desfaz roll e pitch da base; soma a altura do sensor sobre o chão: z passa a contar do chão;
3. célula de cada ponto pelas bordas da §3 (`bucketize` em z, `floor` em x e y) e a máscara `dentro`;
4. `scatter_` com os pontos inválidos ou de fora levados a uma 1441ª célula, que é descartada. Sem
   isto, um ponto inválido no índice 0 pode apagar um válido, porque o `scatter_` não garante ordem.

Ela recebe lote (E envs) no treino e E = 1 no real, com o mesmo código.

## 5. Os raios no treino: o padrão rotativo

O MID-360 real dispara as linhas do `mid360.csv` em sequência, 200 mil por segundo: uma varredura de
0,1 s é um bloco de 20 mil linhas consecutivas, e o bloco seguinte cobre outra parte da vista.

No treino, cada leitura usa R direções tiradas do bloco CORRENTE do CSV (R = 128 a 256, passo
uniforme dentro do bloco), e o bloco avança a cada leitura, como no real. A densidade é menor, mas a
geometria do padrão é a mesma. A acumulação da §6 enche a grade em algumas leituras.

Motivo: com 128 a 256 raios fixos, os raios ficam ~9° um do outro; a 1 m, isso deixa ~16 cm entre
raios, e a grade fica com buracos fixos que o real não tem.

**Verificação antes de tudo (portão da fase 5a):** o `RayCastSensor` do mjlab guarda as direções em
`self._local_directions` (`mjlab/sensor/raycast_sensor.py:446`), gerado uma vez em
`raycast_sensor.py:522`. Confirmar que trocar esse tensor no lugar (`copy_`) a cada leitura muda os
raios do kernel, sem fork do mjlab. Se não mudar, a saída é um padrão próprio com o bloco inteiro
pré-gerado e um índice de deslocamento por leitura, ou um fork pequeno do sensor; o agente reporta
ao dono antes de seguir.

## 6. O tempo: acumulação e transporte (SUBSTITUÍDA pela predição do Kalman da §0, 06/10)

O lidar lê a 10 Hz e a política age a 50 Hz. Entre leituras, a grade é transportada pelo
deslocamento da base e decai:

- na leitura: `s = max(batida, decai · s_transportado)`;
- entre leituras: `s = decai · s_transportado`.

O transporte desloca a grade pelo movimento da base no plano (x, y, yaw), por amostragem da célula
de origem. O decaimento é um knob, com a meia-vida de partida de 1 s.

É isto que guarda o chão da zona cega à frente dos pés (~1,0 m, medido em 23/09) depois que o robô
passa por ele, e a laje de destino quando a caixa sobe ao peito e a esconde.

No robô real, a varredura de 20 mil pontos enche a grade de uma vez; o transporte e o decaimento são
os mesmos. O deslocamento vem do ESKF da percepção (fase 3 do `lidar_sim`, ATE de 1,5 cm na volta
de 02/10).

## 7. O filtro do próprio corpo

- O raycast do treino INCLUI o robô, para o raio parar no ombro como no real.
- Uma lista de cápsulas por elo (ombros, braços, antebraços, mãos, tronco, coxas), gerada das
  geometrias de colisão do MJCF, com margem de 2 a 3 cm (knob).
- `filtra_corpo(pontos, poses_dos_elos, caixa=None) → validos`: distância de cada ponto a cada
  segmento; o ponto a menos do raio sai.
- A caixa na mão entra como mais um volume, ativo só com `pegou`. Antes da pega a caixa é ambiente e
  fica na grade.

## 8. A chave

Um knob booleano no pacote da v3: `PERCEPCAO_LIDAR = True`.

- `True`: o sensor entra na cena, e o termo de observação entra no FIM do `actor` e do `critic`.
- `False`: o sensor não entra, o termo não entra, e a observação volta ao tamanho da v3 sem grade.
  O treino recomeça do zero.

Nenhum termo de recompensa, terminação ou currículo lê a grade. Desligar a chave não muda mais nada.

## 9. As fases

Cada fase tem um portão. A seguinte só começa com o portão verde.

### Fase 1 — a função pura (worktree `lidar_sim`)

O arquivo nasce em `lidar_sim/percepcao/grade.py` e MUDA para o pacote da v3 na fase 5 (move, não
copia). Conteúdo: as constantes (`T_BASE_SENSOR`, bordas), `pontos_para_grade`, `filtra_corpo`,
`transporta`.

**Portão, sem simulador, com pontos sintéticos:**
- um plano a 0,30 m ocupa só a camada de 0,26–0,31 m; com a base inclinada 10° em pitch, a mesma camada;
- o chão com ruído de 2 cm fica na camada do chão em mais de 99% dos pontos;
- um ponto fora do volume não ocupa nada; um inválido no índice 0 não apaga um válido;
- o transporte de 0,15 m para a frente desloca a grade em uma célula; o decaimento segue a meia-vida.

### Fase 2 — no sim do real, robô parado

`sensores/` e o `volta.py` do `lidar_sim` chamam a função com os pontos do `sensores/lidar.py`, a
IMU e a cinemática das pernas.

**Portão:** a laje e uma estante aparecem nas células certas; inclinar o robô não move a laje na
grade; arrastar uma estrutura move as células dela.

### Fase 3 — o filtro do corpo

**Portão:** robô com a caixa no peito: os pontos sobre braços e caixa somem; a laje continua; a
parte da laje escondida pela caixa fica vazia, como no real.

### Fase 4 — acumulação e transporte, no sim do real

**Portão:** com o robô andando, a laje fica parada no mundo enquanto a grade anda; a grade não
pisca entre leituras.

### Fase 5 — no treino da v3

- **5a:** a verificação do `_local_directions` (§5). Reportar ao dono antes de seguir.
- **5b:** o padrão rotativo, o sensor (`RayCastSensorCfg` no site da `T_BASE_SENSOR`, robô incluído,
  `debug_vis` para conferir), o ruído, a cinemática da altura, o filtro do corpo, a acumulação, o
  termo de observação no fim e a chave.
- **5c — custo:** medir o passo com e sem a percepção, na CPU e na GPU do Colab, antes de treinar.
  O Colab é limitado por banda (memória `g1-treino-gargalo-banda-nao-flops`). O dono aceita ou não
  o custo; se não aceitar, a v3 sai com a chave desligada.

**Portão:** o smoke da v3 roda com a chave ligada e desligada; as checagens das fases 1 a 4 portadas;
custo medido e aceito.

### Fase 6 — a réplica

O portão que garante o requisito 3 do dono.

- Mesma cena do depósito e mesma pose do robô, nos dois caminhos: (a) o caminho do treino (raycast do
  mjlab, padrão rotativo, 1 s de acumulação, ruído) e (b) o caminho do sim do real (`sensores/lidar.py`
  denso, mesma função).
- **Portão:** IoU de pelo menos 0,90 entre as células com escore acima de 0,5 nos dois caminhos, em
  pelo menos 5 poses (parado diante da laje, diante de uma estante, andando, com a caixa no peito,
  com o robô inclinado).
- O mesmo teste roda no robô real quando ele existir, com o caminho (b) trocado pelo SDK da Livox.

## 10. Riscos

| risco | onde aparece | o que fazer |
|---|---|---|
| a altura pelo chão diverge entre treino e real | fase 2, real | a cinemática do pé mais baixo nos dois; nunca o `z` verdadeiro |
| a densidade do real enche células que o treino deixa vazias | fase 6 | o padrão rotativo + a acumulação; o IoU da fase 6 mede |
| pontos espúrios do sensor real | real | ruído e perda no treino; o escore com decaimento filtra batida isolada |
| o mjlab não deixa trocar as direções | fase 5a | padrão próprio ou fork pequeno; decisão do dono |
| o custo do raycast derruba as iterações por hora | fase 5c | medir antes; a chave desliga |
| a política ignora a grade com laje fixa | treino | esperado no começo; a grade ganha função quando o suporte variar (prateleira, plano próprio) |
| roll e pitch fora da convenção da IMU do G1 | fase 2, real | o teste da inclinação; conferir a ordem dos eixos contra a IMU antes do real |

## 11. Fora deste plano

- A prateleira como suporte do treino (currículo próprio, depois da v3 com a grade).
- Três estados por célula (ocupado, livre, não visto), com traçado do raio.
- O estimador da caixa fora de vista (plano de 24/09, §9).
