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

**Composições treinadas.** As transições que o robô precisa no real são treinadas dentro
do episódio, em cadeias, sempre com uma espera em ANDAR parado antes do primeiro elo:

- ANDAR (parado) → PEGAR → CARREGAR
- ANDAR (parado) → REORIENTAR → REORIENTAR → … → PEGAR (o número de giros é sorteado)
- ANDAR (parado) → PEGAR → CARREGAR → BOTAR → espera final em ANDAR

Entre duas manipulações sem caixa na mão existe sempre um ANDAR parado. Isso simplifica
o treino e fecha bugs de transição. Por isso BOTAR → REORIENTAR não é treinado: o ciclo
real passa por BOTAR → ANDAR parado → REORIENTAR, e as duas metades já estão na lista.

CARREGAR é uma tarefa própria, com dois regimes: segurar a caixa parado, e andar com a
caixa sob comando de velocidade. O controlador compõe o ciclo completo a partir dessas
transições. Transições fora da lista, como PEGAR → BOTAR sem CARREGAR, não são treinadas.

**Sucesso do ciclo.** As quatro condições valem juntas:

- O robô não cai em nenhuma etapa.
- A caixa não cai em nenhuma etapa.
- A caixa termina no alvo, dentro da tolerância, e fica estável por pelo menos 0,5 s
  sem contato com o robô.
- O robô termina de pé, livre, pronto para o próximo ciclo.


**Fora do objetivo deste modelo:** andar até a mesa com navegação (o comando de
velocidade vem de fora), estimar a pose da caixa (oráculo no sim; estimador vem depois),
lidar, cena de depósito.

## 2. Cenário físico

**Robô.** Unitree G1, modelo MJCF exato do URDF rev 1.0. No robô físico os batentes de
junta são rígidos e o motor que chega neles pode danificar a junta. No treino, chegar ao
limite de uma junta não deve acontecer, em nenhuma etapa. Isso é penalizado.

**Caixa.** Cubo com aresta de 0,07 a 0,13 m, massa de 1 a 5 kg, inércia real. Tamanho e
massa variam por mundo. O robô recebe a pose exata da caixa a cada passo (oráculo no sim;
o estimador vem depois). O conteúdo real é frágil, portanto a caixa não pode sofrer:

- impacto (contra mesa, laje, chão ou o próprio robô);
- amassado por aperto (força de preensão além do necessário para segurar);
- sacudida (aceleração da caixa durante o transporte).

**Orientação da caixa.** A face de cima fica normal ao solo o tempo todo, o mais
próximo possível da vertical. Girar a caixa em torno do eixo vertical é permitido em
qualquer etapa; inclinar em x ou y não é. A única exceção é o tombo de 90° do
REORIENTAR, que termina de novo com uma face normal ao solo.

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
caixa.

**Altura de transporte.** No CARREGAR a caixa fica na altura do peito. O z do alvo é
0,95 m no referencial do mundo, e não no do robô, para obrigar o robô a ficar de pé. O x
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

**Movimento controlado.** Em toda tarefa o robô se move devagar e de forma controlada.
Correr para executar uma ação é risco para os objetos ao redor, para o robô e para
pessoas. Por isso não existe teto de tempo por tarefa: demorar é aceitável, correr não.

**Pés.** Parado, os dois pés ficam no chão, sob a pelve, sem passo. Andando, a marcha é
alternada, com passo de tamanho normal, sem passo miúdo.

**Sem batente.** Nenhuma junta chega ao limite, em nenhuma etapa.

**Sem estátua.** Ficar parado nunca rende mais que avançar a tarefa.

**Locomoção.** O robô anda nas quatro direções e gira. O giro acontece tanto andando
quanto parado, sem deslocamento em x e y. Sem comando de movimento, o robô não gira nem
se desloca no mundo.

**Contato.** Só os pés tocam o chão; as mãos nunca. Só as mãos, e o tronco quando
permitido, tocam a caixa. Nada do robô toca a laje: isso danifica o robô. Uma tarefa
futura de levantar do chão terá regra própria.

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

**Progressão na cadeia.** O que faz o robô avançar de uma tarefa à próxima é a renda
congelada: quando um elo fecha, a renda que ele conquistou fica mantida, e o elo seguinte
abre uma renda nova por cima. O robô aprende um elo, congela o ganho, e passa ao próximo
para ganhar mais. Um conserto que faz o elo seguinte pagar menos que o congelado do
anterior quebra a progressão: o teto do que falta fazer tem de ser maior que o piso do
que já foi feito.

## 4. Etapas

### 4.1 ANDAR

**Entrada.** Comando de velocidade (vx, vy, wz). Sem caixa. Mãos ao lado do tronco.

**Comportamento.** Rastreia a velocidade pedida nas quatro direções, com giro andando ou
parado. Marcha alternada. Ao comando cair a zero, freia em poucos passos e fixa os pés.
Sem comando, fica parado, sem deriva de rumo nem de posição.

**Velocidade.** A faixa de treino vai até 2 m/s à frente, a mesma faixa nas quatro
direções. O robô aprende o máximo que consegue executar. Em campo, o controlador nunca
pede o máximo.

**Fecho.** Não tem. ANDAR dura enquanto o controlador pedir.

**Proibido.** Deriva sem comando. Passo miúdo. Pé arrastando. Frear tarde com passo de
acerto. Braços abertos.

**Particularidades já pagas caro.**
- A recompensa preferia parar: parado colhia 27% do rastreio de velocidade.
- A locomoção precisa formar antes da manipulação receber orçamento.
- O currículo do fabricante sobrescreve a faixa de comando: uma queda de eficiência na
  troca de faixa é comando novo, e não regressão.

### 4.2 Espera

Antes do primeiro elo de manipulação, o robô fica em ANDAR com comando zero: parado, de
pé, mãos ao lado do tronco, de 0,5 a 1,5 s, com a caixa já visível. Isso treina a
transição ANDAR parado → PEGAR em todo episódio de manipulação.

Depois do BOTAR existe uma espera final: o robô solta, volta à pose de pé e fica parado
até o fim do episódio.

### 4.3 PEGAR

**Entrada.** Robô parado em frente à laje, caixa apoiada, alvo de pega recebido. O
controlador já levou o robô até a posição de pega.

**Comportamento.** Aproxima as duas mãos pelas laterais, seguindo a pose de IK. O alvo
de IK escolhe o par de faces mais alinhado com as palmas; a caixa nunca é pega pela
quina. Pode corrigir os pés e deslocar um pouco a caixa na laje se a pega exigir. Fecha com a força
necessária para segurar, e não mais. Ergue a caixa reta, sem tombar, até o alvo de
transporte, e mantém ali por um período curto. O tronco inclina quando o alvo exige.

**Fecho.** Caixa presa nas duas mãos, no alvo de transporte dentro da tolerância,
nivelada, sem contato com a laje, mantida por um período curto.

**Proibido.** Pegar por cima ou com o antebraço. Apertar além do necessário. Erguer
tombada. Tocar a laje. Ficar parado com a mão na caixa sem erguer. Mergulhar sobre a
caixa. Arrastar a caixa pela laje além do ajuste de pega.

**Particularidades já pagas caro.**
- Aperto 4,9 vezes o necessário sem erguer: apertar pagava, erguer não.
- Tombar ao erguer era grátis: o termo de orientação tinha derivada zero longe do alvo.
- Escorar a caixa no tronco comprava o fecho sem preensão.
- A pega parada rendia mais que a pega: termos que pagavam por estar parado.
- O alvo da pega era circular e o ombro ficava a 1,02 m: o alvo ficava fora do alcance.
- O `ANG` media uma face lateral e não via o tombo em torno do eixo palma-a-palma.

### 4.4 CARREGAR

**Entrada.** Caixa nas duas mãos, no alvo de transporte. Comando de velocidade.

**Comportamento.** Dois regimes. Parado: segura a caixa na altura do peito, de pé, sem
deriva. Andando: recebe os mesmos comandos do ANDAR, nas quatro direções, em várias
velocidades, com giro andando ou parado, e frear em poucos passos. Rastreia a velocidade
com a caixa nas mãos, nivelada, sem sacudida, com o centro de massa sobre os pés. A caixa
fica no alvo de transporte o tempo todo. Em carga alta pode apoiar no tronco.

**Fecho.** Não tem. CARREGAR dura enquanto o controlador pedir.

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

**Comportamento.** Desce a caixa reta até o alvo, seguindo a pose de IK. Inclina o tronco
se o alvo é baixo. Apoia a caixa sem impacto. Quando a caixa chega ao alvo com a
orientação correta, solta e volta à pose de pé estável, mãos ao lado do tronco. O robô
aprende como soltar; a ordem das mãos não é prescrita. Pode se mover minimamente ao
soltar.

**Fecho.** Caixa apoiada dentro de 0,10 m do alvo, nivelada, parada por 0,5 s, sem
contato com o robô. Robô de pé.

**Proibido.** Soltar de altura. Pousar tombada. Empurrar a caixa depois de soltar. Tocar
a laje. Deitar o tronco para alcançar. Pairar com a caixa sobre o alvo sem apoiar. Ficar
apoiado depois de soltar.

**Particularidades já pagas caro.**
- Pairar pagava mais que apoiar: a renda do BOTAR não era monótona.
- O termo de alinhamento punia endireitar o tronco depois de soltar: congelava o tombo.
- O `de_pe` medido pelo joelho pagava não botar.
- Veredito do dono na 21700: tronco a 65°, braço a 55% do alcance, hip_yaw a −34°.
- O robô nunca largava: a cadeia chegava ao CARREGAR e o BOTAR não completava.
- O impacto da caixa na laje não tinha preço: soltura rápida era grátis.

### 4.6 REORIENTAR

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
| ANDAR | Passo miúdo, sem voo | `air_time` desligado | Marcha só forma com incentivo de voo |
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
- Torque e energia como regra explícita: por enquanto não; o robô infere no treino.
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
