---
tipo: decisao
pai: "[[Transversais]]"
estado: ativa
verificado: 2026-10-06
resumo: "Enunciado do problema escrito pelo dono: o comportamento esperado do G1 em cada etapa (andar, pegar, carregar, botar). Todo conserto cita a regra que serve."
cluster: transversais
---
# Comportamento esperado do G1 — enunciado do problema

Este documento descreve o problema que o treino do `g1_limpo` resolve. Ele não descreve
recompensa nem código. O código é uma tentativa de resolver o problema descrito aqui.

Uso: antes de todo conserto, aponte a regra deste documento que o conserto serve e as
regras das outras etapas que o conserto toca. Um conserto que serve a um sintoma e não a
uma regra daqui não entra.

## 1. Objetivo

Um modelo só controla o G1 em cinco tarefas: ANDAR, PEGAR, CARREGAR, BOTAR e REORIENTAR.
O controlador externo escolhe a tarefa por one-hot e troca em voo, dentro do mesmo
episódio, sem handover.

O ciclo completo do produto: o robô anda até uma laje com caixas; o algoritmo externo
escolhe qual pegar; o robô pega, carrega até uma bancada padrão e bota; a câmera procura
o código de barras; o robô reorienta a caixa até a câmera ler; pega de novo, carrega até
o destino, bota, e anda de volta. O REORIENTAR gira a caixa
apoiada na laje em uma primitiva de 90° por comando, para expor outra face à câmera, e
termina largando a caixa.

**Exemplo do comportamento esperado (dono, 02/10).** O robô trabalha num galpão. O
controlador externo dá o comando de velocidade e escolhe a tarefa; o modelo executa:

| # | O que acontece | Tarefa (one-hot) |
|---|---|---|
| 1 | Anda até a mesa A e para em frente a ela | ANDAR, depois ANDAR parado |
| 2 | Pega a caixa da mesa | PEGAR |
| 3 | Com a caixa, gira 180° no lugar | CARREGAR (giro no lugar) |
| 4 | Anda até a mesa B, com alguns desvios no trajeto, e para em frente a ela | CARREGAR (andando, com curvas), depois CARREGAR parado |
| 5 | Bota a caixa sobre a mesa B | BOTAR, depois ANDAR parado |
| 6 | Reorienta a caixa até a câmera ler o código de barras | REORIENTAR × k |
| 7 | Pega a caixa de novo | PEGAR |
| 8 | Gira e anda até o pallet | CARREGAR (giro no lugar, andando) |
| 9 | Bota a caixa no pallet, que é uma laje baixa | BOTAR, depois ANDAR parado |
| 10 | Repete o ciclo | ANDAR |

O exemplo exercita todas as tarefas e as transições da lista abaixo, o giro no lugar com e
sem caixa, a curva andando com a caixa e o BOTAR em duas alturas (mesa e pallet).

**Composições treinadas.** As transições que o robô precisa no real são treinadas dentro
do episódio, em cadeias, sempre com uma espera em ANDAR parado antes do primeiro elo:

- ANDAR (parado) → PEGAR → CARREGAR
- ANDAR (parado) → PEGAR → CARREGAR parado → BOTAR → espera final em ANDAR
- FUTURO: ANDAR (parado) → REORIENTAR → REORIENTAR → … → PEGAR (o número de giros é
  sorteado). O REORIENTAR e o LEVANTAR ficam fora da v3, cada um com arquitetura própria, e
  têm o slot do one-hot reservado desde o começo (dono, 02/10).

Entre duas manipulações sem caixa na mão existe sempre um ANDAR parado. Isso simplifica
o treino e fecha bugs de transição. Por isso BOTAR → REORIENTAR não é treinado: o ciclo
real passa por BOTAR → ANDAR parado → REORIENTAR, e as duas metades já estão na lista.

CARREGAR é uma tarefa própria, com dois regimes: segurar a caixa parado, e andar com a
caixa sob comando de velocidade. O controlador compõe o ciclo completo a partir dessas
transições. Transições fora da lista, como PEGAR → BOTAR sem CARREGAR, não são treinadas.

**Ordem de aprendizado.** Primeiro o robô aprende a andar, com 100% dos envs. Cada tarefa
abre quando a tarefa de que ela depende domina o primeiro nível: a cadeia do PEGAR abre com o
andar reto e o parado dominados; a cadeia do BOTAR abre com o PEGAR dominado na laje mais alta.
Dois balanceadores repartem os envs, um entre as tarefas e outro entre as subtarefas e os níveis
de cada tarefa: quem sabe menos recebe mais envs, quem já sabe mantém um piso, e as frações
ficam iguais quando tudo está dominado. Depois que o robô faz todas as tarefas em todos os
níveis, as restrições apertam aos poucos, tarefa por tarefa (refino). O desenho está em
`specs/g1-v3.md` §11 (dono, 02/10; substitui as fases de `specs/g1-limpo-curriculo-de-cadeia.md`).

**Sucesso do ciclo.** As quatro condições valem juntas:

- O robô não cai em nenhuma etapa.
- A caixa não cai em nenhuma etapa.
- A caixa termina no alvo, dentro da tolerância, e fica estável por pelo menos 0,5 s
  sem contato com o robô.
- O robô termina de pé, livre, pronto para o próximo ciclo.

**Sucesso de um episódio de treino.** O último elo da cadeia fechou e o episódio chegou
ao fim pelo tempo, sem terminação: nem a caixa nem o robô caíram depois do fecho. Esse
sinal move o nível de dificuldade e as médias de sucesso das cadeias.


**Fora do objetivo deste modelo:** andar até a mesa com navegação (o comando de
velocidade vem de fora), o estimador real da caixa (o treino usa um estimador simulado, §2),
a grade do lidar (chave desligada na primeira v3; dono, 02/10), cena de depósito.

## 2. Cenário físico

**Robô.** Unitree G1, modelo MJCF exato do URDF rev 1.0. No robô físico os batentes de
junta são rígidos e o motor que chega neles pode danificar a junta. No treino, chegar ao
limite de uma junta não deve acontecer, em nenhuma etapa. Isso é penalizado.

**Caixa.** Cubo com meia aresta de 0,07 a 0,13 m (aresta de 0,14 a 0,26 m), massa de 1 a
5 kg, inércia real. Tamanho e massa variam por mundo. O robô não conhece a massa: ele se adapta
a ela (dono, 05/10).

A pose da caixa vem de um estimador, e não de um oráculo (dono, 02/10). No real, uma câmera
RGB, sem marcador, junto com o depth: um algoritmo detecta a caixa na imagem e outro detecta a
orientação. No treino, um estimador simulado com a mesma saída: atraso, leitura na taxa da
câmera, oclusão pelos braços e pelo alcance mínimo do depth, e erro que cresce no refino até o
medido (`specs/g1-v3.md` §13). O robô que se move guarda a caixa e o alvo no mundo pela
odometria.

O conteúdo real é frágil, portanto a caixa não pode sofrer:

- impacto (contra mesa, laje, chão ou o próprio robô);
- amassado por aperto: o robô aperta o suficiente para a caixa não escorregar nem cair, e nunca
  esmaga a caixa, entre as mãos, contra o corpo ou contra uma laje ou parede (dono, 05/10);
- sacudida (aceleração da caixa durante o transporte).

**Orientação da caixa.** A face de cima fica normal ao solo o tempo todo, o mais
próximo possível da vertical. Girar a caixa em torno do eixo vertical é permitido em
qualquer etapa; inclinar em x ou y não é. Limite: 15° no fim do treino, em toda etapa,
inclusive no transporte, porque o conteúdo não pode sofrer impacto (dono, 05/10). A única
exceção é o tombo de 90° do REORIENTAR, que termina de novo com uma face normal ao solo.

**Lajes.** O robô pega a caixa apoiada em uma laje e bota em outra. A altura de cada
laje é sorteada entre 0 e 0,55 m, de forma independente. A bancada do REORIENTAR é
sorteada entre 0,40 e 0,70 m. O robô não conhece essas alturas, e ninguém informa a ele
que existe uma laje sob a caixa: ele aprende sozinho que o fundo da caixa está apoiado.
Ele recebe um alvo de pose para pegar, um para segurar e um para soltar, e age sobre o
alvo, com ou sem laje ali.

**Posição da caixa na laje.** A caixa está em posição pegável, perto da borda, com
posição e giro em torno do vertical sorteados em faixa curta. Sem isso o robô não
generaliza em campo. Hoje a caixa é isolada. Mais tarde ela terá outra "caixa" ao redor
(laterais, topo e fundo), e o robô terá de pegar sem tocar nela com as mãos nem com a
caixa. Isto fica para o futuro, com arquitetura de comportamento própria, como o
REORIENTAR: uma estante de vários níveis, onde só os braços entram no vão (dono, 02/10).

**Altura de transporte.** No CARREGAR a caixa fica na altura do peito. O z do alvo é
0,90 m no referencial do mundo (dono, 02/10; antes, 0,95), e não no do robô, para obrigar o robô a ficar de pé. O x
e o y do alvo ficam no referencial do peito.

**Tolerância no BOTAR.** 0,10 m de posição.

## 3. Invariantes globais

Regras que valem em toda etapa, com ou sem caixa, e nas trocas entre etapas.

**Postura.** O robô fica de pé, tronco ereto, pelve na altura de pé. Nas tarefas de
manipulação o ereto é mole: o tronco pode e deve inclinar para manipular a caixa quando
o alvo exige. Agachar é permitido só para alcançar um alvo baixo, e só enquanto o alvo
está baixo. A pose de IK para uma caixa no chão não chega ao batente, portanto agachar
não justifica batente. O centro de massa fica sobre o meio dos pés sempre que possível,
e em especial quando carrega a caixa.

**De pé, como um humano.** Parado, o robô fica com a pelve nivelada, o tronco vertical e
as pernas quase retas, com a pelve a 0,78–0,79 m e o centro de massa sobre o meio dos
pés. Sem caixa, o joelho fica perto de 0,15 rad. Com a caixa na altura de transporte, de
1 a 5 kg, o joelho fica entre 0,20 e 0,42 rad, e o corpo inclina levemente para trás a
partir dos tornozelos: os pés ficam de 3 a 7 cm à frente da pelve. A pelve não inclina
para a frente com a cintura compensando o tronco. Andando, a pelve fica nivelada e o
tronco vertical; a altura da pelve e o joelho seguem a marcha. Os números vêm da IK de
30/09 (`specs/g1-limpo-postura-de-pe.md` §2).

**Movimento controlado.** Em toda tarefa o robô se move devagar e de forma controlada.
Correr para executar uma ação é risco para os objetos ao redor, para o robô e para
pessoas. Por isso não existe teto de tempo por tarefa: demorar é aceitável, correr não.

**Pés.** Parado, os dois pés ficam no chão, sob a pelve, sem passo. Andando, a marcha é
alternada, com passo de tamanho normal, sem passo miúdo, em qualquer velocidade, com e sem
caixa (dono, 05/10).

**Sem batente.** Nenhuma junta chega ao limite, em nenhuma etapa.

**Sem estátua.** Ficar parado nunca rende mais que avançar a tarefa.

**Locomoção.** O robô anda nas quatro direções e gira. O giro acontece tanto andando
quanto parado, sem deslocamento em x e y. Sem comando de movimento, o robô não gira nem
se desloca no mundo. Com comando baixo, ele executa o comando pedido: ficar parado ou marchar
no lugar não serve, e o controlador não precisa pedir mais que o necessário (dono, 05/10).

**Contato.** Só os pés tocam o chão; as mãos nunca. Só as mãos, e o tronco quando
permitido, tocam a caixa, e só da pega até soltar: antes do comando de pegar e depois de
soltar, o robô não toca a caixa, porque pode derrubá-la ou chegar perto demais de um jeito ruim
(dono, 05/10). Nada do robô toca a laje: isso danifica o robô. Uma tarefa futura de levantar do
chão terá regra própria.

**Pega.** A caixa é pega pelas faces laterais, sempre com as duas mãos. Em carga alta o
robô pode apoiar a caixa no tronco. A caixa nunca fica em uma mão só.

**Caixa íntegra e nivelada.** Sem impacto, sem aperto além do necessário, sem sacudida.
Face de cima normal ao solo.

**Pose de referência.** Nas tarefas de manipulação o robô se aproxima das poses geradas
por IK para o alvo. Sem caixa, o robô fica em pose de pé estável, mãos ao lado do tronco.

**Empurrão.** O robô resiste a empurrão em toda tarefa, na locomoção e na manipulação,
sem cair e sem largar a caixa.

**Troca de tarefa.** A troca não dá tranco no robô, e a pose muda pouco de um passo para
o outro. É por isso que as transições são treinadas.

**Progressão na cadeia.** O que faz o robô avançar é o progresso, e não o estado (dono,
01/10; substitui a renda congelada). Nas tarefas de alvo, o robô ganha ao se aproximar do alvo
macro corrente e um bônus fixo ao alcançá-lo; então o alvo seguinte passa a contar. Parar,
pairar perto de um alvo e ficar na mesa pagam zero. Quando o episódio termina, o ganho já pago
fica. Um conserto que faz ficar perto de um alvo render mais que alcançá-lo, ou que paga por
passo numa tarefa sem relógio, quebra a progressão. O desenho está em `specs/g1-v3.md` §9.

## 4. Etapas

### 4.1 ANDAR

**Entrada.** Comando de velocidade (vx, vy, wz). Sem caixa. Mãos ao lado do tronco.

**Comportamento.** Rastreia a velocidade pedida nas quatro direções, com giro andando ou
parado. Marcha alternada. Ao comando cair a zero, freia em poucos passos e fixa os pés.
Sem comando, fica parado, sem deriva de rumo nem de posição.

**Velocidade.** A faixa de treino vai até 1,5 m/s para a frente e 1,0 m/s para trás e de
lado, com giro de até 1,0 rad/s para os dois lados: o robô gira 180° em no máximo 4 s (dono,
05/10; antes, 1 m/s e 0,5 rad/s). O robô aprende o máximo que consegue executar. Em campo, o
controlador nunca pede o máximo.

**Fecho.** Não tem. ANDAR dura enquanto o controlador pedir.

**Proibido.** Deriva sem comando. Passo miúdo. Pé arrastando. Frear tarde com passo de
acerto. Braços abertos. Ignorar comando baixo: ficar parado ou marchar no lugar.

**Particularidades já pagas caro.**
- A recompensa preferia parar: parado colhia 27% do rastreio de velocidade.
- A locomoção precisa formar antes da manipulação receber orçamento.
- O currículo do fabricante sobrescreve a faixa de comando: uma queda de eficiência na
  troca de faixa é comando novo, e não regressão.

### 4.2 Espera

Antes do primeiro elo de manipulação, o robô fica em ANDAR com comando zero: parado, de
pé, mãos ao lado do tronco, de 0,5 a 1,0 s, com a caixa já visível (dono, 01/10). Isso
treina a transição ANDAR parado → PEGAR em todo episódio de manipulação.

Depois do BOTAR existe uma espera final de 1,5 s em ANDAR parado: o robô já soltou e voltou
à pose de pé dentro do BOTAR, e fica parado.

### 4.3 PEGAR

**Entrada.** Robô parado em frente à laje, caixa apoiada, alvo de pega recebido. O
controlador já levou o robô até a posição de pega.

**Comportamento.** Aproxima as duas mãos pela frente da caixa e as avança ao longo das
faces laterais, seguindo a pose de IK; o mesmo estilo serve à laje e à prateleira (dono,
02/10). O alvo
de IK escolhe o par de faces mais alinhado com as palmas; a caixa nunca é pega pela
quina. Pode corrigir os pés e deslocar um pouco a caixa na laje se a pega exigir. Fecha com a força
necessária para segurar, e não mais. Ergue a caixa reta, sem tombar, até o alvo de
transporte, e mantém ali por um período curto. O tronco inclina quando o alvo exige.

**Fecho.** Caixa presa nas duas mãos, no alvo de transporte dentro da tolerância,
nivelada, sem contato com a laje, mantida por um período curto.

**Proibido.** Pegar por cima ou com o antebraço. Apertar além do necessário. Erguer
tombada. Tocar a laje. Ficar parado com a mão na caixa sem erguer. Mergulhar sobre a
caixa. Arrastar a caixa pela laje além do ajuste de pega. Na prateleira, deslizar a caixa
no piso até ela passar a borda é a saída, e é permitido (dono, 02/10).

**Particularidades já pagas caro.**
- Aperto 4,9 vezes o necessário sem erguer: apertar pagava, erguer não.
- Tombar ao erguer era grátis: o termo de orientação tinha derivada zero longe do alvo.
- Escorar a caixa no tronco comprava o fecho sem preensão.
- A pega parada rendia mais que a pega: termos que pagavam por estar parado.
- O alvo da pega era circular e o ombro ficava a 1,02 m: o alvo ficava fora do alcance.
- O `ANG` media uma face lateral e não via o tombo em torno do eixo palma-a-palma.

### 4.4 CARREGAR

**Entrada.** Caixa nas duas mãos, no alvo de transporte. Comando de velocidade.

**Comportamento.** Dois regimes. Parado: segura a caixa na altura do peito, de pé, com a
pelve nivelada e o tronco vertical, sem deriva. Andando: recebe os mesmos comandos do ANDAR, nas quatro direções, em várias
velocidades, com giro andando ou parado, e frear em poucos passos. Rastreia a velocidade
com a caixa nas mãos, nivelada, sem sacudida, com o centro de massa sobre os pés. A caixa
fica no alvo de transporte o tempo todo. Em carga alta pode apoiar no tronco.

**Fecho.** Como cauda das cadeias B e R, não tem: CARREGAR dura enquanto o controlador
pedir. Como elo da cadeia C, fecha com a caixa no alvo de transporte, nivelada, o robô de
pé, por 0,5 s, e então abre o BOTAR.

**Proibido.** Largar. Deixar a caixa cair de altura. Andar agachado. Girar sem comando.
Segurar com uma mão. Sacudir a caixa.

**Particularidades já pagas caro.**
- Não andava com a caixa: andar somava 5,2% e parar pagava o resto.
- Andava agachado com a caixa.
- Deriva de rumo com a caixa: −0,38 rad/s custava 0,8/s e o freio afrouxava 2,7 vezes.
- Segura a caixa e cai, em vez de largar: o modo de falha trocou de largar para cair.

### 4.5 BOTAR

**Entrada.** Caixa nas mãos, robô parado em frente à laje de destino, alvo de soltura
recebido.

**Comportamento.** O controlador manda BOTAR uma vez: abaixar, apoiar, soltar e voltar de pé
acontecem dentro do BOTAR, sem outro comando (dono, 02/10). Depois de ficar de pé, o robô
continua parado no BOTAR até o controlador trocar a tarefa. Desce a caixa reta até o alvo,
seguindo a pose de IK. Inclina o tronco se o alvo é baixo. Apoia a caixa sem impacto. Quando
a caixa chega ao alvo com a orientação correta, afrouxa o aperto com as mãos ainda nas faces,
solta e volta à pose de pé estável, mãos ao lado do tronco (dono, 05/10). Na prateleira, entra
com a caixa deslizando no piso, sem tocar laterais nem teto, e tira as mãos pela frente
(dono, 02/10). O robô
aprende como soltar; a ordem das mãos não é prescrita. Pode se mover minimamente ao
soltar.

**Fecho.** Caixa apoiada dentro de 0,10 m do alvo, nivelada, parada por 0,5 s, sem
contato com o robô. Robô de pé.

**Proibido.** Soltar de altura. Pousar tombada. Empurrar a caixa depois de soltar. Tocar
a laje. Deitar o tronco no lugar de estender o braço: o tronco curvado da IK na laje baixa,
com os braços estendidos, é o desejado (dono, 01/10). Pairar com a caixa sobre o alvo sem
apoiar. Ficar apoiado depois de soltar.

**Particularidades já pagas caro.**
- Pairar pagava mais que apoiar: a renda do BOTAR não era monótona.
- O termo de alinhamento punia endireitar o tronco depois de soltar: congelava o tombo.
- O `de_pe` medido pelo joelho pagava não botar.
- Veredito do dono na 21700: tronco a 65°, braço a 55% do alcance, hip_yaw a −34°.
- O robô nunca largava: a cadeia chegava ao CARREGAR e o BOTAR não completava.
- O impacto da caixa na laje não tinha preço: soltura rápida era grátis.

### 4.6 REORIENTAR

FUTURO (dono, 02/10): o REORIENTAR fica fora da v3 e entra depois, com arquitetura de
comportamento própria. O texto abaixo fica como requisito para quando ele entrar.

**Por que existe.** Cada caixa tem um código de barras em uma das seis faces. As câmeras
do robô precisam ler esse código. Um algoritmo externo olha a imagem da caixa, decide
qual face expor e manda ao robô uma primitiva de giro. As primitivas cobrem as seis
faces.

**Entrada.** Robô parado em frente à laje, caixa apoiada, uma primitiva pedida:
girar 90° para a esquerda, girar 90° para a direita (eixo vertical), ou tombar 90° para
trás (eixo horizontal, a face da frente vai para cima).

**Comportamento.** Encosta as duas mãos na caixa e executa a primitiva, devagar. O giro
pode ser apoiado, rolando sobre a aresta, ou no ar, erguendo a caixa o suficiente para a
quina não tocar a bancada. Pode soltar e repegar no meio do tombo se o curso do punho
exige. Ao fim, a caixa está apoiada, com uma face normal ao solo, a menos de meia aresta
do lugar onde estava. Solta, afasta as mãos e volta à pose padrão, mãos ao lado do
tronco. A próxima primitiva ou o PEGAR vem em seguida.

**Cadeia.** ANDAR parado → REORIENTAR × k → PEGAR, com k sorteado.

**Fecho.** Caixa girada 90° no eixo pedido, com erro angular abaixo de 10°, apoiada,
nivelada, a menos de meia aresta do lugar, sem contato com o robô.

**Proibido.** Derrubar a caixa, em hipótese alguma. Impacto ao pousar. Apertar. Sacudir.
Girar além de 90° e voltar. Deslocar mais de meia aresta. Tocar a bancada.

**Particularidades.**
- O tombo de 90° em torno do eixo entre as palmas exige 90° de punho ou um repegue.
  Girar no ar sobre o centro da caixa varre 0,707 da aresta: para 0,13 m, a quina precisa
  de 2,7 cm de folga.
- Hoje o REORIENTAR é inerte no treino: um temporizador de 0,3 s, com placa marcando a
  face alvo. Ele sorteia metade da manipulação e come orçamento das cadeias longas.
- O tombo para trás passa por uma posição instável da caixa sobre a aresta. Ali a caixa
  cai se as mãos soltam ou se a força é assimétrica. É o ponto mais perigoso do ciclo.

## 5. Catálogo de reward hacking observado

Uma linha por hack. O objetivo é reconhecer o padrão antes de repetir o conserto.

| Etapa | O que o robô fez | O que pagou | O que o conserto quebrou ou ensinou |
|---|---|---|---|
| ANDAR | Ficou parado | `track_lin` parado colhia 27% do teto | Rastreio precisa de derivada só no movimento |
| ANDAR | Passo miúdo, sem voo | `air_time` desligado | CORRIGIDO 05/10: a marcha formou sem incentivo de voo (inventário v3, X1); o passo curto em velocidade baixa vem do período fixo da marcha, e a v3 faz o período crescer quando a velocidade cai |
| ANDAR | Ignorou comando baixo; girou estranho ou só com comando maior | Parado colhia `exp(−c²/σ²)` do rastreio: 96% a 0,1 com σ 0,5 | Rastreio normalizado pelo comando: parado paga 0 em qualquer comando (`specs/g1-v3.md` §19) |
| ANDAR | Girou devagar sem comando | O freio de rumo afrouxava 2,7× com a caixa | Regime vem do estado, não do comando |
| Todas | Estátua | Termos que pagam por estar parado: `forma_postural` 0,81/s, de pé grátis | Gatear todo termo que paga por parar |
| Todas | Batente no punho e no hip_yaw | Limite de junta a 85% não morde no curso largo | Batente é restrição mole; a rampa precisa de derivada onde o robô está |
| Todas | Pernas comprimidas, std caiu cedo | Freio −15 com σ 0,96 | O freio taxa o ruído, não a marcha |
| Todas | Bang-bang na ação | 84% do tempo congelado no BOTAR | Pose é escolha da política, não falta de alcance |
| PEGAR | Apertou 4,9× e não ergueu | Fechar pagava; erguer não | `precise_ori` ×4 não era a trava |
| PEGAR | Ergueu tombada | Gaussiana com derivada zero longe do alvo | Termo híbrido linear + gauss de 0 a 180° |
| PEGAR | Escorou a caixa no tronco | 4,9 N abria a cauda | Escorar comprava o fecho sem preensão |
| PEGAR | Pairou sem fechar | Pairar 25/s contra fechar 16,5/s | "Fechou e não avançou" pagava 32/s |
| PEGAR | Mergulhou sobre a caixa | Aborto por perda de preensão era ótimo na tabela | O laço de recuperação é plano externo |
| PEGAR | Deitou a caixa 90°, face de cima para o peito | A reta `1 − Δθ/π` ainda pagava 0,5 a 90°; endireitar exigia o punho | Reta zera a 90° (zero15); a âncora baixa foi revertida pelo dono em 30/09 (zero16) |
| PEGAR | Alvo fora do alcance | Alvo circular a 1,02 m era o ombro | σ = distância inicial; alvo no alcance |
| CARREGAR | Não andou com a caixa | Andar somava 5,2% | Renda congelada acumula 10× o déficit |
| CARREGAR | Andou agachado | `de_pe` pelo joelho | De pé é o tronco, 94,7% |
| CARREGAR | Segurou e caiu | Largar custava mais que cair | O modo de falha troca quando o preço troca |
| CARREGAR | Esmagou a caixa | 68,9 N sem preço contra 8,45 necessário | Squeeze cego |
| PEGAR, CARREGAR | Pelve 22° à frente, cintura para trás, pelve baixa a 0,72 m | A referência de forma, presa na linha da laje de 0,55, pedia pelve 0,72 e tronco 22°; o `upright` pedia tronco 0°; nada media a pelve | Referência acima da laje é a pose de pé com caixa; a pelve precisa de medida própria |
| BOTAR | Pairou sobre o alvo | Pairar 10,7 < apoiada 13,0 < espera 17,9 só depois de consertar | Renda tem de ser monótona ao longo da etapa |
| BOTAR | Deitou o tronco a 65° | `alinhado` punia endireitar | Congelava o tombo |
| BOTAR | Nunca largou | `largou` em zero o run inteiro | A cadeia parava no CARREGAR |
| BOTAR | Soltou rápido | Impacto sem preço | Impacto é regra global |
| Cauda | Ficou apoiado depois de soltar | Cauda sem derivada em 60% | Cauda paga só de pé e livre |
| Currículo | Nível saltou sobre competência zero | `frac_uniforme` empurra a média | O ponto fixo não é 50% de sucesso |
| Currículo | Renda congelada virou 57% da renda | Paga dobrado no fecho terminal e é inobservável | Congelar a queda, e canal no crítico |

## 6. Decisões fechadas

Rejeitados, com o motivo. Não reabrir sem dado novo.

- Modo entrega no viewer: rejeitado.
- Catraca `max(0, M − viva)` e déficit `max(0, fecho − viva)`: pinam o total; fazer melhor não paga.
- Multiplicador por posição na cadeia: inobservável pelo crítico.
- Cone elíptico de orientação: diverge para NaN, duas vezes.
- Termo novo de recompensa como primeiro recurso: conserta a forma de um termo que existe.
- Teto de tempo por tarefa: rejeitado; correr é o risco, demorar não.
- Torque e energia como regra explícita: por enquanto não; o robô infere no treino. Exceção:
  o custo de energia da receita da Unitree (−2e−5), que entra com a locomoção (dono, 02/10).
- Ordem de soltura das mãos prescrita: o robô aprende.
- BOTAR → REORIENTAR como cadeia: rejeitado; sempre há ANDAR parado entre manipulações.
- Tombar para frente ou para o lado no REORIENTAR: só para trás.
- Aborto por perda de preensão como termo: o laço de recuperação é do controlador.

## 7. Como usar

Antes de cada conserto:

1. Nomear o sintoma e a etapa.
2. Apontar a regra deste documento que o sintoma viola.
3. Listar as regras das outras etapas que o conserto toca, e dizer por que não quebra
   cada uma.
4. Se o conserto é um termo novo, dizer qual termo existente não serviu e por quê.
5. Procurar o padrão na seção 5. Se já aconteceu, o conserto anterior é o ponto de
   partida.
6. Depois do treino, verificar no `play` as regras tocadas, e não só o sintoma.
