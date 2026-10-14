---
tipo: decisao
pai: "[[C4 v3]]"
estado: em-desenho
verificado: 2026-10-09
resumo: "Comportamento esperado da v3 com as decisões do dono de 01/10 a 09/10, sincronizado com a spec em 09/10: o que o robô vê, faz e recebe em cada trecho de cada episódio; base da spec g1-v3."
cluster: c4
---
# G1 v3 — comportamento esperado do treino (rascunho)

Estado: RASCUNHO de 01/10, SINCRONIZADO com `specs/g1-v3.md` em 09/10 (pedido do dono, depois do
achado L3 de `docs/relatorios/2026-10-09-viabilidade-v3.md`). Ele descreve o que o robô vê, o que
ele faz e o que paga em cada trecho de cada episódio. Ele não descreve código. Os números e as
fórmulas ficam na spec; aqui fica o comportamento, com a seção da spec entre parênteses. Uma
decisão nova muda os dois documentos na mesma edição.

Base: `specs/g1-limpo-comportamento.md` (enunciado), `vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md` (decisões
de 01/10) e `docs/relatorios/2026-10-01-inventario-v3.md` (o que funcionou). Os códigos entre
colchetes, como [C20] ou [A10], apontam itens do inventário.

## 1. O produto não muda

Um modelo só controla o G1. O controlador externo escolhe a tarefa por one-hot e troca a
tarefa dentro do episódio, sem handover. O ciclo do produto é o do enunciado §1.

O que o controlador manda ao robô, em campo e no treino:

- o one-hot da tarefa, com 6 slots (ANDAR, PEGAR, CARREGAR, BOTAR; REORIENTAR e LEVANTAR
  reservados e apagados);
- o comando de velocidade (vx, vy, wz), nas tarefas que andam;
- a caixa escolhida (no PEGAR) e o alvo de soltura (no BOTAR); depois disso o estimador rastreia
  os dois (§13; spec §6.4);
- o alvo definitivo da tarefa: o alvo de transporte no PEGAR e no CARREGAR, o alvo de soltura no
  BOTAR;
- o alvo macro corrente, que um gerador de IK calcula a partir da caixa e do alvo definitivo.
  O gerador roda no robô; ele é a mesma tabela de IK do PEGAR e do BOTAR de hoje [C47].

A v3 treina ANDAR, PEGAR, CARREGAR e BOTAR, com a laje como único suporte. O REORIENTAR, o
LEVANTAR e a prateleira entram depois, no futuro, cada um com arquitetura própria. Os slots deles
no one-hot ficam apagados até lá [C12].

O robô vê o terreno por um mapa de elevação 2,5D (2 × 2 m, 10 cm, altura e confiança; recorte de
um mapa interno de 5 × 5 m do filtro, dono 14/10, D5), montado
pelo lidar simplificado na pose do MID-360, ligado desde o começo em todos os envs (dono, 06/10;
supera o "desligada na primeira v3" de 02/10; spec §4, §19.1). O ANDAR e o CARREGAR andam em rampa,
escada, chão irregular e obstáculo discreto, que o robô passa por cima; o PEGAR e o BOTAR ficam no
plano (dono, 09/10; enunciado §2 "Terreno"; spec §11.2, §12).

O treino roda no mjlab; a política do mjlab passa sem treino no Isaac Sim (sim2sim), e o treino é
refeito no Isaac Lab (dono, 09/10; enunciado §2 "Simulador"; spec §1, §16 Parte 5).

## 2. Fases do treino

**Fase 0, locomoção sozinha.** 100% dos envs andam. As subtarefas do ANDAR abrem em ordem:
reto nas quatro direções e parado, comando baixo, giro andando, giro no lugar; depois os níveis
de velocidade e o nível TERRENO (spec §11.2). A fração de cada uma segue o balanceador descrito
abaixo. A fase 0 termina quando o reto e o parado do ANDAR passam no portão (0,75 de sucesso,
com o sucesso pela razão de deslocamento ≥ 0,50 [C1, C2]); aí a B abre (spec §11.1). O giro não
trava a abertura.

**Fase 1, até o fim do treino.** A tarefa de cada env sai do balanceador de tarefas (abaixo) e
fica com o env até o reset seguinte. A fração vale em ENVS, não em episódios: assim a fatia de
envs é a fatia de passos, qualquer que seja a duração dos episódios [F29, C44]. Sorteio por
episódio daria menos passos à tarefa que morre cedo, que é justamente a que erra mais.

| fatia | episódio | envs no equilíbrio |
|---|---|---|
| ANDAR | só comando de velocidade | 1/3 |
| B | ANDAR parado, PEGAR, CARREGAR com as subtarefas do ANDAR | 1/3 |
| C | ANDAR parado, PEGAR, CARREGAR parado, BOTAR, ANDAR parado | 1/3 |
| R, futuro | ANDAR parado, REORIENTAR × k, PEGAR | 1/4 cada, quando entrar |
| L, futuro | ANDAR parado, empurrão forte, LEVANTAR (L0 → L1 → L2) | uma fatia a mais, quando entrar |

**Dois balanceadores, com a mesma regra** (pedido do dono, 02/10; mecanismo em proposta).
O balanceador de tarefas reparte a frota entre as fatias abertas (ANDAR, B, C; R depois). O
balanceador de subtarefas reparte cada fatia entre as subtarefas e níveis abertos dela. Nos
dois, quem sabe menos recebe mais envs, quem já sabe mantém um piso, e as frações ficam iguais
quando tudo passa no portão: 1/3 por fatia no fim.

- **O sucesso da tarefa vem das subtarefas.** O déficit da tarefa é a média dos déficits
  `d_i` das subtarefas abertas dela, sem peso pela amostragem. O sucesso bruto nos envs da
  tarefa não serve: o balanceador de dentro põe a maior parte dos envs na subtarefa mais
  fraca e puxaria o sucesso da tarefa para baixo sozinho. Na B, entram as subtarefas dos dois
  balanceadores dela (nível do PEGAR e comando do CARREGAR).
- **Abertura.** A fase 0 é o caso em que só o ANDAR está aberto (100%). Cada tarefa tem seu
  nível (dono, 02/10), e nenhuma tarefa trava a abertura de outra. Uma fatia abre quando a
  tarefa de que ela depende domina o PRIMEIRO nível: a B abre com o reto e o parado do ANDAR
  no portão; a C abre com o PEGAR no portão no primeiro nível (laje mais alta, caixa mais leve),
  porque a pega da C usa a faixa que o PEGAR já domina. Depois, os níveis de cada tarefa sobem
  sozinhos. (Corrige a regra anterior, "abre quando todas as subtarefas abertas passam", que
  deixava um giro no lugar preso travar a C; avaliação de 02/10, R4.)
- **Piso de tarefa 0,20 (decidido pelo dono, 02/10), piso de subtarefa 0,10, ε 0,02, portão
  0,75.** Exemplo do dono,
  ANDAR e B abertos:

  | estado | ANDAR | B | a subtarefa mais fraca do ANDAR | a do PEGAR |
  |---|---|---|---|---|
  | girar em 0,40; laje 0,15 em 0,20 | 41% | 59% | 62% do ANDAR | 75% do PEGAR |
  | girar aprendeu | 25% | 75% | 25% | 75% |
  | laje 0,15 em 0,60 | 33% | 67% | 25% | 67% |
  | tudo no portão | 50% | 50% | 25% | 33% |
  | abre a laje 0,04 | 25% | 75% | 25% | 66% |

- **Abertura em ordem.** Uma subtarefa fechada tem fração zero. A seguinte abre quando as
  abertas passam no portão. ANDAR: reto e parado → comando baixo → giro andando → giro no lugar
  → níveis de velocidade → TERRENO. CARREGAR da B: reto, giro no lugar e arco, com os níveis de
  velocidade do ANDAR → TERRENO, com tetos menores (spec §11.2). O ANDAR e o CARREGAR da B têm balanceadores independentes
  (decidido pelo dono, 02/10).
- **Níveis são subtarefas do mesmo balanceador** (decidido pelo dono, 02/10): as alturas do
  PEGAR e do BOTAR, as velocidades do ANDAR e do CARREGAR e, depois, os ângulos do
  REORIENTAR. Abre só o nível mais fácil (a laje mais alta e a caixa mais leve); o seguinte
  abre quando os abertos estão dominados. A massa faz parte do nível: cada nível abre alturas
  mais baixas e massas maiores (dono, 02/10). O robô treina mais o nível que sabe menos.
- **Fração.** Com `n` subtarefas abertas, `s_i` o sucesso da subtarefa (média móvel, medido só
  nos envs dela [C43]), `d_i = max(0, portão − s_i)` e `w_i = d_i + ε`:
  `f_i = piso + (1 − n·piso) · w_i / Σw`. Quando todo `d_i` é zero, `f_i = 1/n`.
- **Números** (portão 0,75, piso 0,10, ε 0,02):

  | estado | reto | parado | giro andando | giro no lugar |
  |---|---|---|---|---|
  | abertura | 50% | 50% | — | — |
  | giro andando abre | 12% | 12% | 77% | — |
  | giro no lugar abre | 11% | 11% | 11% | 66% |
  | giro no lugar em 0,70 | 19% | 19% | 19% | 42% |
  | tudo no portão | 25% | 25% | 25% | 25% |
  | parado esquece (0,60), giro no lugar em 0,20 | 12% | 23% | 12% | 54% |

- **Proteções medidas.** A subtarefa que abre começa num nível fácil, e cada nível contém os
  anteriores [C42: o PLR punha ~70% da amostra no nível impossível]. O piso existe porque o
  crítico esquece o estado que a política deixa de visitar [F30]. Mais envs não criam
  gradiente: o giro no lugar precisa do |wz| acima do piso σ do rastreio [A10].
- **Ligação com o refino.** "Todas as subtarefas no portão" é o estado de frações iguais e é
  também o gatilho do refino (§9 item 10).

**A cadeia ordena o aprendizado sozinha.** O robô só chega ao CARREGAR depois de completar o
PEGAR, e só chega ao BOTAR depois do CARREGAR parado. Nenhuma fase de cadeia, SEGURA ou
balanceador é necessária [S5]. O motivo está na seção 5: pairar paga zero, então completar o
PEGAR sempre vale mais que ficar perto dele, mesmo antes de o CARREGAR estar treinado.

## 3. Os três episódios

Todo episódio dura no máximo 20 s, como no molde. Ao fim do último trecho da cadeia, o episódio
acaba como `time_out`, sem punição por terminar cedo.

### 3.1 ANDAR

| trecho | duração | o robô vê | o robô faz |
|---|---|---|---|
| ANDAR | 20 s; o comando muda a cada 3–8 s | one-hot ANDAR, comando, canais da caixa em zero [C13], mapa de elevação | rastreia o comando nas subtarefas abertas; marcha alternada; parado, fica de pé sem deriva; no nível TERRENO, sobe e desce escada e rampa e passa por cima do obstáculo |

### 3.2 B: PEGAR e CARREGAR

| trecho | duração | o robô vê | o robô faz |
|---|---|---|---|
| ANDAR parado | sorteada, 0,5–1,0 s | one-hot ANDAR, comando zero, a caixa visível na laje | fica de pé, mãos ao lado do tronco, sem deriva |
| PEGAR | o tempo que o robô levar | one-hot PEGAR, comando zero, a caixa, o alvo macro | percorre os alvos macro do PEGAR (seção 4) até a caixa no peito, nivelada, de pé |
| CARREGAR | fixa a partir do início dele, sorteada em 4–7 s | one-hot CARREGAR, comando sorteado entre as quatro subtarefas, a caixa, o alvo de transporte | rastreia o comando com a caixa no peito, nivelada, nas duas mãos, de pé |

O comando do CARREGAR é sorteado na abertura dele, sem herdar o zero da pega; hoje a cauda
herdava ~2,9 s parada. Ele não muda dentro do trecho: frear com a caixa vem da generalização
do ANDAR (decidido pelo dono, 02/10; conferir no play com CARREGAR a 0,5 m/s seguido de
CARREGAR parado). A laje sai da cena quando o comando pede movimento [C39]. No nível TERRENO do
CARREGAR, a pega acontece na plataforma plana do centro do subterreno (2 m), e o CARREGAR sai
dela para o terreno (dono, 09/10; spec §12).

### 3.3 C: PEGAR, CARREGAR parado, BOTAR

| trecho | duração | o robô vê | o robô faz |
|---|---|---|---|
| ANDAR parado | sorteada, 0,5–1,0 s (decisão 16) | como na B | como na B |
| PEGAR | o tempo que o robô levar | como na B; a laje sai da faixa que o PEGAR já domina, só de 0,30 m para cima | como na B |
| CARREGAR parado | sorteada, 0,5–1,0 s | one-hot CARREGAR, comando zero | segura a caixa no peito, parado, de pé |
| BOTAR | o tempo que o robô levar | one-hot BOTAR, a laje de destino à frente, o alvo de soltura, o alvo macro | percorre os alvos macro do BOTAR até soltar e ficar de pé |
| ANDAR parado | fixa, 1,5 s; a troca para ele vem de 0 a 1,5 s depois do aviso do detector de fim (pose de pé sem caixa, estável por 0,5 s; dono, 14/10; spec da interface §4.2), sorteada, com o one-hot ainda em BOTAR (dono, 05/10) | one-hot ANDAR, comando zero | fica de pé, livre, mãos ao lado do tronco; a caixa fica apoiada, sem contato |

A laje de destino aparece na abertura do BOTAR e só se afasta do corpo no sorteio [C38]. A
cadeia C fica toda no plano (spec §12).

## 4. Alvos macro

Um alvo macro é uma pose intermediária da tarefa, em poucas grandezas do espaço da tarefa. Ele
sai da caixa estimada (posição, giro e tamanho) e do alvo definitivo, então generaliza em posição e
tamanho. O peso da caixa a IK não vê: a política o aprende com a variação de 1 a 5 kg. As 29
juntas ficam livres; não há mímica. A tabela é `g1_limpo/ik/agachado/ref_agachado.npz` (dono,
08/10; trocada em 09/10; spec §8.1).

O alvo mede só a grandeza da tarefa: as palmas no P0–P2, a caixa no P3, P4, B1 e B2, as palmas
afastadas no B3 e a pose no B4. A postura (pelve, recuo, tronco, CoM, sola, giro da coxa, torque)
não é atraída para a IK: ela fica dentro de sete cercas de custo, com os valores da IK, e o PPO
escolhe a pose dentro delas (dono, 08/10; spec §9.1, §9.3).

O alvo macro avança quando o robô chega nele dentro da tolerância. Não existe relógio: demorar
é permitido, correr não (enunciado §3).

### 4.1 PEGAR

| alvo | o que vale | avança quando |
|---|---|---|
| P0 frente | as palmas 5 cm à frente da face da frente, alinhadas às faces laterais, na altura do centro | as palmas a até 3 cm |
| P1 entrada | as palmas avançam ao longo das faces laterais até o centro delas | as palmas a até 3 cm |
| P2 pega | as duas palmas nas faces; sem força mínima (dono, 05/10) | contato nas duas mãos |
| P3 erguida | a caixa 5 cm acima da laje, nivelada, sem apoio na laje | sem apoio e nivelada dentro do limite do degrau (30° → 15°) |
| P4 peito | a caixa no alvo de transporte (x, y no peito, z 0,90 m no mundo (dono, 02/10)) [C18], nivelada; o robô na pose de pé com caixa, a MESMA para toda massa (dono, 05/10; spec §8.1) | caixa a até 5 cm e pelve a até 3 cm, por 0,5 s |

O par de faces é o mais alinhado com as palmas; a caixa nunca é pega pela quina (enunciado §4.3).

### 4.2 BOTAR

O BOTAR é o PEGAR ao contrário, com a mesma tabela de IK (decisão do dono, 01/10).

O controlador manda BOTAR uma vez. Abaixar, apoiar, soltar e voltar de pé acontecem dentro do
BOTAR, sem outro comando (dono, 02/10: "não quero ficar mandando comando atrás de comando").
Depois do B4, o robô fica de pé no BOTAR até o controlador mandar a tarefa seguinte, quando
quiser.

| alvo | o que vale | avança quando |
|---|---|---|
| B1 sobre o alvo | a caixa 5 cm acima do alvo de soltura, nivelada | caixa a até 3 cm na horizontal |
| B2 apoiada | a caixa apoiada no alvo, dentro de 0,10 m, nivelada, chegada abaixo do limiar [A1] | apoio de 0,8·m·g ou mais, por 0,3 s |
| B3 solta | as duas palmas a 5 cm ou mais das faces; a caixa apoiada e parada | por 0,5 s |
| B4 de pé | a pose de pé sem caixa, mãos ao lado do tronco | pelve a até 3 cm e tronco a até 5°, por 0,5 s |

Soltar não exige afrouxar antes: o gate `F_solta` saiu (dono, 09/10; spec §8.2). O invariante de
B2 a B4 (abaixo) impede arrastar a caixa depois de apoiada.

### 4.3 Invariantes

- De P0 a P2: a caixa desloca menos de meia aresta na laje (dono, 05/10).
- De P2 a B2: caixa nivelada (30° na fase aprender, 15° no fim) e as duas mãos na caixa. Não
  existe faixa de força.
- De B2 a B4: a caixa, depois de apoiada, fica apoiada, nivelada e parada, com menos de 2 cm de
  deslocamento (dono, 09/10).

Quebrar um invariante zera a proximidade ao alvo corrente e impede o avanço. Endireitar devolve a
proximidade (spec §8.2).

**Aperto** (dono, 09/10; spec §8.2): o robô paga pelo aperto, e o menor aperto que segura custa menos,
`−w·(1−exp(−aperto/F_ref))` (dono, 14/10, B02), `F_ref` 49 N (μ 0,5), só com a caixa presa (P3 a B2), com o peso 0 → `w` no refino. O mínimo que
segura vem do conflito com o P3 e com o CARREGAR: abaixo do atrito, a caixa escorrega. A faixa
`F_ref`–k·`F_ref` [A2] não entra.

Consequência: **a caixa não chega ao peito tombada**. Ela não passa de P3 sem estar nivelada, e
pairar tombada paga zero.

## 5. O que paga

### 5.1 Tarefas de alvo: PEGAR e BOTAR

- **Só o progresso paga.** O termo de tarefa paga a mudança de um potencial: alvos já
  alcançados mais a proximidade ao alvo seguinte. A soma no trecho é o potencial no fim menos
  o do começo.
- **Parado paga zero.** Pairar perto de um alvo, ficar na mesa ou segurar sem avançar não
  rendem nada.
- **O potencial começa a contar na abertura da tarefa.** Agachar durante a espera não adianta
  ganho nenhum: ele só sobe o ponto de partida [A21].
- **O rastreio só cobra a deriva.** Com comando zero, ele vale zero parado e negativo quando o
  robô desliza ou gira sem comando. Ele não paga renda.
- **Os termos posturais da locomoção não agem.** No PEGAR e no BOTAR, o desvio postural e o
  tronco plano da Unitree valem zero; a postura fica dentro das cercas (spec §9.2, §9.3). O
  tronco plano brigaria com o tronco inclinado da IK [A14].
- **Nenhum termo paga positivo por passo** no PEGAR, no BOTAR e depois da caixa perdida: o
  `alive` e a altura do pé vão a zero (dono, 05/10; spec §9.2).

### 5.2 Tarefas por passo: ANDAR, esperas e CARREGAR

- Os termos da receita da Unitree pagam por passo (spec §19, §9.2), com dois desvios do dono
  (05/10): o rastreio é normalizado pelo comando, para o parado receber zero em comando baixo, e
  o termo da marcha paga o comprimento do passo no toque do pé, para o passo ter comprimento
  mínimo; o robô escolhe a cadência, sem relógio na observação (dono, 14/10). O desvio
  postural mira a pose de pé da IK: sem caixa no ANDAR, com caixa no CARREGAR (spec §6.3).
- No CARREGAR, o pagamento por passo é multiplicado pela qualidade da caixa: no peito,
  nivelada, nas duas mãos.
- **A duração é do controlador.** O robô não alonga nem encurta um trecho por passo. Então
  terminar o PEGAR antes não compra passos de CARREGAR, e a pressa perde a fonte.

### 5.3 Sempre

- Os reguladores da Unitree custam em toda tarefa: vz, giro xy, velocidade e aceleração de
  junta, `action_rate`, energia, escorrego do pé, contato fora dos pés filtrado pelo parceiro.
  O limite de junta é mole a 95% do curso, com rampa pela banda (dono, 08/10). As sete cercas de
  postura custam em toda tarefa; a da sola só fora da marcha (dono, 09/10; spec §9.3). O freio
  por nível [C8] age só no PEGAR, no CARREGAR e no BOTAR (dono, 05/10; spec §11.4).
- **Contato do corpo com a mesa ou a laje: multa em rampa, como hoje [C31].** O limiar começa
  em 50 N e aperta até 10 N no refino (§9 item 10; decisões do dono de 01/10 e 02/10). Não é
  terminação: a terminação de 28/08 matava 76% dos episódios de manipulação por ruído de
  exploração.
- O ganho de cada alvo macro tem de cobrir o custo dos reguladores no trecho até ele, medido no
  ponto de operação.

### 5.4 Terminações

O episódio termina quando:

- o robô cai: inclinação acima de 0,8 rad ou base abaixo de 0,2 m no ANDAR e no CARREGAR
  (Unitree); `fell_over` a 70° no PEGAR e no BOTAR (spec §10);
- a caixa cai ou sai das mãos [C35], ou o robô solta a caixa fora do alvo [C34]; com a chave
  `CAIXA_PERDIDA_VIRA_ANDAR` ligada, a perda troca para ANDAR parado em vez de terminar (§10.1);
- a caixa sai da laje antes da pega, de P0 a P2: conta como caixa perdida (dono, 05/10);
- a velocidade de chegada da caixa passa do limiar [A1]: 1,0 m/s na fase aprender, até
  0,25 m/s no refino (dono, 14/10, A10; era 1,98 → 0,99);
- `|hip_yaw|` passa de 1,0 rad [C33];
- a junta chega ao batente: SÓ NO REFINO (opção A, decidida pelo dono, 02/10).

As outras terminações valem desde o começo (decidido pelo dono, 02/10).

A terminação custa um valor fixo (−200 × dt) e o resto do episódio. **O ganho já pago fica.** Isto corrige
o que eu disse na discussão ("cair devolve tudo"). Devolver tudo faria o que está em jogo
crescer ao longo da cadeia: no BOTAR, uma queda devolveria também a pega, e o robô aprenderia
a não tentar o BOTAR. Com o ganho mantido, o que está em jogo nunca passa do que falta ganhar.

## 6. Níveis e sucesso

| tarefa | o nível controla | o sucesso que move o nível |
|---|---|---|
| ANDAR e CARREGAR | as subtarefas de comando, os níveis de velocidade e o nível TERRENO, pelo balanceador da fatia (§2) | razão entre o deslocamento projetado e o pedido, sobre o twist inteiro, ≥ 0,50, sem queda [P3, A11]; no TERRENO do CARREGAR, também a caixa a ≤ 15° e sem largar (spec §11.3) |
| PEGAR | altura da laje da pega e massa juntas: (0,55 m; 1 kg) → (0,45; 2) → (0,30; 3) → (0,15; 4) → (0,04; 5) (dono, 05/10) | PEGAR completo e o trecho seguinte sem terminação |
| BOTAR | altura da laje de destino (0,55 → 0,04 m); a massa vem da faixa já dominada pelo PEGAR (dono, 02/10) | BOTAR completo e o ANDAR parado final sem terminação |

- Cada nível contém os anteriores, e nenhum nível é impossível [C42].
- Na cadeia C, a laje da pega sai da faixa que o PEGAR já domina. A fronteira do PEGAR treina na
  cadeia B.
- O freio age só na manipulação. O degrau do freio de cada env segue o nível do grupo da tarefa
  dele, contado como o número de níveis abertos [C8] (spec §11.4).

## 7. Sintoma a sintoma

| sintoma de hoje | causa medida | o que a v3 muda |
|---|---|---|
| Não gira no lugar | ~3% da frota; |wz| abaixo do piso σ, estátua empata | fração pelo balanceador da fatia (até ~2/3 do ANDAR) e rastreio normalizado pelo comando: parado recebe zero em qualquer comando [A10] (spec §19, conflito 5) |
| A caixa tomba na pega | o braço gira a caixa ao erguer; o fecho aceitava 25°; pairar tombada pagava | nivelada é condição de todo alvo depois da pega; pairar paga zero |
| Bota mal; não bota em laje baixa | o BOTAR só treinava quando a pega da C fechava; a laje herdava a do PEGAR | nível próprio do BOTAR; a pega da C usa a faixa dominada; postura da IK da altura |
| Carrega agachado e estranho | a referência de forma presa na laje; nada media a pelve | P4 exige a pose de pé com caixa; no CARREGAR, o desvio postural e a altura da base da Unitree miram a pose de pé com caixa, e o rastreio paga vezes a qualidade da caixa |
| Agacha antes de pegar | fechar 0,5 s antes rendia ~7 | o potencial não paga o agachamento antecipado |
| No play, bota e fica agachado com as mãos na caixa (dono, 02/10) | no treino de hoje, o fecho do BOTAR não exige soltar [A6], e no fecho a máquina de estados publica ANDAR (`comando.py:1767`, espera final): o robô aprendeu a soltar e levantar só quando o one-hot vira ANDAR; o play não troca, e ele espera | abaixar, apoiar, soltar e voltar de pé ficam DENTRO do BOTAR (B1 → B4); o alvo avança sozinho pela tolerância; o controlador manda BOTAR uma vez |
| Toca a laje com a perna | o limiar de 50 N não cobra o toque leve | rampa de 50 N que aperta até 10 N no refino; a IK de referência tem folga de 10 mm da laje |

## 8. O que muda no enunciado

- §1: as cadeias passam a ser ANDAR, B e C, como na seção 2; o REORIENTAR depois.
- §2: a caixa tem meia-aresta de 0,07 a 0,13 m (o texto diz aresta).
- §3, Progressão na cadeia: o potencial substitui a renda congelada.
- §4.2: a caixa fica visível na espera; a espera inicial vai de 0,5 a 1,0 s; a espera final
  tem 1,5 s.
- §1: a pega da cadeia C só usa laje de 0,30 m para cima; as lajes baixas treinam na B.
- §4.6: o REORIENTAR fica para depois.
- §4.5: o proibido "deitar o tronco para alcançar" passa a dizer "deitar o tronco no lugar de
  estender o braço"; o tronco curvado da IK na laje baixa é o desejado.
- §5: a linha "marcha só forma com incentivo de voo" está errada [X1].

## 9. Decisões e pendências

Decidido pelo dono (01/10):

1. **Tolerância de nivelamento para avançar: 15°.** Vale para todo alvo depois de P2 e para o
   BOTAR. Hoje o fecho aceita 25°.
2. **Pose de pé parado: a IK.** Em regime parado, o `pose` mira a IK de pé: sem caixa, joelho
   ~0,15 rad; com caixa, a IK de pé com caixa para a massa (joelho 0,20–0,42). Andando, ele
   segue o keyframe do molde. Em 01/10 isto tinha sido rejeitado para o `g1_limpo` pelo risco
   da estátua; na v3 o robô não alonga a espera, então o risco não existe. ATUALIZADO (dono,
   05/10): o `pose` e o keyframe do molde saem com a receita da Unitree; a pose de pé da IK
   (joelho 0,16) é o offset da ação e o alvo do desvio postural, parado e andando (spec §6.3);
   com caixa, a mesma pose para toda massa (spec §8.1).

3. **Contato com a mesa ou a laje: joelho da rampa em 10 N.**
4. **Nenhuma junta chega ao limite; a penalização pode ser grave.**
5. **Referências de postura aprovadas: `g1_limpo/ik/ref_v3_manipulacao.npz` e
   `g1_limpo/ik/ref_v3_de_pe.npz`** (01/10, vistas de frente e de lado). O tronco curvado
   na laje baixa, com os braços estendidos, é o desejado. O proibido do enunciado §4.5
   ("deitar o tronco para alcançar") é o do 21700: deitar o tronco no lugar de estender o
   braço. A tabela nova tem 10 mm de folga com a laje (a antiga punha a perna dentro dela) e
   a cintura até 25°. Na laje de 0,04 m o robô fica 9–11 cm mais atrás, e a caixa de 14 cm
   fica rente à borda da laje. SUPERADO (dono, 08/10; tabela trocada em 09/10): as pegas vêm de
   `g1_limpo/ik/agachado/ref_agachado.npz`, e a pose de pé com caixa vem de
   `ref_v3_de_pe_theta-30_tamanhos.npz`, com a mão a −30° (spec §8.1).
6. **Durações, dentro dos 20 s do episódio** (decidido 01/10): espera 0,5–1,0 s; CARREGAR da
   B 4–7 s; CARREGAR parado da C 0,5–1,0 s; ANDAR parado final 1,5 s; a pega da C só em laje
   de 0,30 m para cima. Estimado a 0,25 m/s de mão, em linha reta, pelas poses de referência:
   PEGAR 3,0–7,0 s e BOTAR 3,2–7,2 s. Pior caso: B 15,0 s (a pega cabe até 12 s); C 17,2 s,
   com a espera de até 1,5 s depois do B4 (dono, 05/10; spec §7). O log mede o tempo real do
   PEGAR e do BOTAR e a fração de C cortada nos 20 s.
7. **Impacto pela velocidade de chegada** (decidido pelo dono, 01/10, depois do teste de
   queda). Teste de queda na cena, MuJoCo clássico: o choque dura
   menos que um passo de controle (20 ms), e o pico lido uma vez por passo depende da fase: 5 cm
   leem 2,5 × m·g e 8 cm leem 9,0 (o pico real por subpasso é 11,8 e 16,1). A medida passa a
   ser a velocidade vertical da caixa no passo antes do primeiro contato com a laje. Limiar
   final 0,99 m/s, a de uma queda de 5 cm. Não depende da massa nem do tamanho da caixa.
   ATUALIZADO 14/10 (A10): limiar 1,0 → 0,25 m/s, e o potencial do BOTAR paga o pouso devagar
   com `exp(−(v/σ)²)`, σ 1,0 → 0,35 m/s (spec §9.1).
8. **Alvo macro** (decidido 01/10): mãos e pelve atraem para a IK da altura em todos os alvos;
   tronco e pés ficam numa faixa em volta da IK; a força das mãos fica numa faixa entre `F_ref`
   e k·`F_ref`. SUPERADO: a pelve saiu do Φ e virou cerca (dono, 08/10, opção 2); tronco,
   recuo e CoM viraram cercas (spec §9.3); a faixa de força saiu, e o aperto é recompensa pelo
   menor aperto que segura (dono, 09/10; spec §8.2).
9. **Batente** (decidido 01/10): a rampa de cada junta tem largura de min(15°, 30% do
   meio-curso) a partir do batente, por lado, e sobe a ~50/s no batente; a junta que chega ao
   batente termina o episódio; a rampa nunca cobra a IK, então a da cintura vai de 25° a 30°.
   Medido 01/10: os plays de hoje passam do batente no `ankle_pitch` e no `waist_pitch`.
   Decidido pelo dono, 02/10: a rampa no batente entra no refino (item 10); na fase aprender
   só o limite físico do MJCF age. CORRIGIDO pelo dono, 02/10: a terminação e a rampa no
   batente valem desde o começo. Condições: a terminação custa mais que ficar na rampa (na
   `zero04`, morrer a −4 saía mais barato que a rampa a 53,6/s). REFUTADO 02/10 (conta com o
   `cena.npz` e o `init_std` 1,0): ler o alvo comandado termina 92,7% dos passos já no começo;
   ler o ângulo medido terminaria 4 de 7 plays do 23250 em 1,6 s, pelo tornozelo. Desde o
   começo, a terminação no batente mata o treino (`docs/relatorios/2026-10-02-avaliacao-v3.md`
   §2). DECIDIDO pelo dono, 02/10: opção A. A rampa vale desde o começo, e a terminação no
   batente entra no refino (§12.1). ATUALIZADO (dono, 08/10): a rampa vira limite mole a 95%
   do curso da tabela do robô real, com o excesso medido em fração da banda até o batente; ela
   cobre as juntas de curso largo desde o começo, sem batente virtual (spec §6.5, §9.2).
10. **Refino progressivo** (decidido pelo dono, 01/10 e 02/10). Fase aprender: um modelo faz
    todas as tarefas, pegar e botar a caixa em todas as alturas; o freio sobe por nível, como
    hoje. Quando isso acontece, começa a fase refinar, e as restrições apertam aos poucos.
    Apertam no refino: contato 50 → 10 N; nivelamento 30° →
    15°; chegada da caixa 1,0 → 0,25 m/s, σ do pouso 1,0 → 0,35 m/s; peso do `r_aperto` 0 → `w` (dono, 09/10; era
    "aperto 8 × `F_ref` → ~2,5 ×"). As cercas de postura (spec §9.3) valem cheias desde o
    começo, sem degrau (dono, 08/10). Desde o começo: as terminações do §5.4 (decidido pelo
    dono, 02/10), a rampa do limite de junta (item 9) e a caixa acima de 30° (o alvo não
    avança). O gatilho de cada degrau está no §12.1.


## 10. Garantias de comportamento (pedido do dono, 02/10)

Cada linha diz o mecanismo, quando ele age e se ele já está no desenho.

| comportamento | mecanismo | quando | estado |
|---|---|---|---|
| De pé sem caixa, pose normal, sem agachar antes de pegar | o desvio postural da Unitree mira a pose de pé da IK sem caixa (pelve 0,79, joelho 0,16), parado e andando (spec §6.3). O potencial do PEGAR conta a partir da abertura: agachar na espera não adianta ganho [A21]. A espera tem duração do controlador, então não vira estátua paga [R17] | desde o começo | no desenho |
| De pé com caixa, mãos na caixa | o P4 exige a pose de pé com caixa, a mesma para toda massa (spec §8.1). No CARREGAR, o desvio postural mira essa pose, e o rastreio paga vezes a qualidade da caixa. Os invariantes exigem as duas mãos | desde o começo | no desenho |
| Andar normal, com e sem caixa | termos da receita da Unitree: rastreio normalizado, comprimento do passo (no lugar da marcha por fase, dono 14/10), altura do pé, escorrego do pé; o CARREGAR usa os mesmos termos de pé; o `giro_sem_gingado` saiu (spec §19) | desde o começo | no desenho |
| Sem passo miúdo, em qualquer velocidade | o termo da marcha paga `min(L/L_alvo, 1)` no toque do pé, com `L_alvo = |v|·clamp(2·L_min/|v|, 0,8; 1,4)/2` (dono, 05/10; forma de 14/10, sem relógio na observação; spec §19, conflito 6). Substitui o `air_time` com duração mínima no refino | desde o começo | no desenho |
| Andar e carregar no terreno | mapa de elevação na observação; nível TERRENO no ANDAR e no CARREGAR, com tetos menores com a caixa (dono, 09/10; spec §11.2) | depois dos níveis de velocidade | no desenho |
| Caixa sem sacudir | penalidade pela aceleração linear e angular da caixa, em PEGAR, CARREGAR e BOTAR (e REORIENTAR quando entrar), a partir da diferença de velocidade entre passos de controle; a diferença de velocidade integra o choque e não sofre o aliasing da força [F37]. Medido 02/10 nos plays do 23250: no CARREGAR a 0,5 m/s, linear p50 2,9 e p90 4,8 m/s², angular p50 17 e p90 28 rad/s²; no PEGAR, linear p90 4,6–7,3 m/s², angular p90 23–45 rad/s² | refino (decidido pelo dono, 02/10) | no desenho; escala a calibrar com estes números |
| Lado de cima para cima | caixa nivelada é invariante de P2 a B2, e a caixa apoiada fica parada de B2 a B4 (30° → 15° no refino); tombo pelo eixo de cima [C22]; orientação com gradiente de 0 a 180° [C23]; vertical do mundo no BOTAR [C25]; qualidade do CARREGAR | desde o começo | no desenho |
| Sem correr | freio por nível na manipulação [C8]; a duração dos trechos por passo é do controlador, então terminar antes não compra passos; a renda congelada, que dava a pressa, sai [S1] | desde o começo | no desenho |
| Sem demorar | o desconto γ 0,99 (horizonte ~2 s) faz o progresso de agora valer mais; PEGAR e BOTAR que não fecham nos 20 s não contam sucesso, e o balanceador manda mais envs para eles. O equilíbrio entre o desconto e o freio se mede no ponto de operação | desde o começo | no desenho; calibrar |
| Resistir a empurrões | empurrão da Unitree, ±0,5 m/s a cada 5 s, em todas as tarefas; fora do play (spec §12) | desde o começo | no desenho |

### 10.1 Comportamentos novos (propostas)

- **Não se jogar atrás da caixa que cai** (o dono quer testar, com volta atrás: uma chave escolhe entre trocar para ANDAR parado e o comportamento atual, a terminação; desligar a chave devolve o treino de hoje). Hoje a caixa largada termina o episódio com o mesmo
  custo da queda do robô [C35, §5.4]. Então se jogar para salvar a caixa não custa nada a mais.
  Proposta: a caixa perdida deixa de terminar o episódio. A tarefa troca para ANDAR parado até o
  fim, como o controlador faria, com um custo fixo menor que o da queda. Largar e ficar de pé
  passa a valer mais que mergulhar e cair. Não é termo novo: é uma transição de cadeia. O
  ANDAR parado depois da perda leva à pose de pé da IK e ao ficar nela (dono, 02/10).
  ⚠ O trecho depois da perda tem de ser POBRE (plano de 21/09,
  `docs/planos/2026-09-21-aborto-por-perda-de-preensao.md`): se ele pagar por passo, derrubar
  a caixa vira lucro, sobretudo no BOTAR difícil. Regra: a volta à pose de pé paga como
  potencial, uma vez, e ficar nela paga zero; o custo fixo da perda é maior que esse ganho;
  a queda custa mais que a perda. No ANDAR os canais da caixa vão a zero [C13], então o robô
  não vê mais a caixa para persegui-la.
- **Caixa longe: dar um passo ou não pegar** (decidido pelo dono, 02/10). O alvo macro de pelve e pés sai da IK posta em
  relação à caixa. Com a caixa mais longe, os alvos andam para a frente, e o passo é o caminho
  para o potencial subir. A faixa do tronco e dos pés zera a proximidade de quem se estica fora
  dela, então o mergulho paga zero. A distância da caixa vira nível, aberto depois das caixas
  perto [C37: com jitter de 0,20 m desde o começo, 60–81% nasciam fora do alcance]. Além do
  alcance de um passo, o controlador não manda PEGAR (enunciado §4.3, entrada).
- **Levantar depois de cair** (decidido pelo dono, 02/10; FUTURO, junto com o REORIENTAR).
  Tarefa própria, com slot próprio no one-hot reservado desde o começo da v3 e
  apagado até entrar [C12]. O episódio começa de pé, parado; um empurrão forte derruba o robô.
  O dono apontou que trocar o comando no meio da queda faz o robô se debater. Mini cadeia, com
  alvos macro dentro do LEVANTAR, como o P1–P4 do PEGAR:

  | etapa | o que o robô faz | quem dispara |
  |---|---|---|
  | L0 queda | a política age na descida, como no FIRM: cobra a variação de momento dos corpos que batem no chão e o solavanco do corpo; os reguladores (`joint_acc`, `action_rate`, freio) seguram o debate. A variação de momento integra o choque e não sofre aliasing [F37] | a inclinação passa do limiar da IMU; o one-hot troca para LEVANTAR |
  | L1 deitado | fica estirado de bruços ou de costas; a pose das juntas no chão não é ditada | tronco deitado (de bruços ou de costas) e base quase parada por um tempo |
  | L2 de pé | recebe o alvo de pé e levanta até a pose de pé da IK; o caminho é dele, sem trajetória nem mímica | L1 dentro da tolerância |

  Decidido pelo dono, 02/10: o FIRM é a referência mais próxima; alvos macro dentro do
  LEVANTAR; no chão só se exige estar estirado de bruços ou de costas; "como ele chega é
  problema dele". Slot próprio no one-hot, reservado desde o começo da v3.
  Fecho: pose de pé da IK por 0,5 s; o controlador troca para ANDAR parado. Na pesquisa: o
  controlador de fábrica do G1 segue uma trajetória de três fases e só levanta de costas; o
  HumanUP também rola de bruços para de costas antes de levantar, em políticas separadas, e gera
  as poses deitadas soltando o robô de 0,5 m; o FIRM desliga os atuadores por 0,04–1,0 s no
  começo da queda e cobra impacto, momento e solavanco do corpo na descida; o HoST levanta de
  posturas variadas com vários críticos, regularização de suavidade e limite implícito de
  velocidade; o HiFAR só age depois do pouso e abre de 2D para 3D.

## 11. Laje e prateleira (02/10)

FUTURO (dono, 02/10): a prateleira sai da v3 e fica para o futuro, com arquitetura de
comportamento própria, como o REORIENTAR. A v3 treina só com a laje. O que segue fica
registrado para quando ela entrar.

Decidido pelo dono, 02/10: as mãos entram pela frente também na laje, desde o começo; a
prateleira era nível tardio da v3 (substituído acima); o enunciado registra que deslizar a caixa no piso da
prateleira é permitido (§4.3 e §4.5). A prateleira é um cubo oco sem a frente, um pouco
maior que a caixa, "15 cm cada lado talvez"; a primeira IK testa 15 cm e 7,5 cm de folga
por lado.

Pedido do dono: o suporte da caixa varia entre a laje aberta e a prateleira, com laterais e
teto (o enunciado §2 já prevê "laterais, topo e fundo" ao redor da caixa). A caixa não pode
bater na prateleira. As mãos entram pela frente. Na prateleira, a caixa não pode só subir.
A percepção é a grade do MID-360 (`docs/planos/2026-10-02-percepcao-lidar-na-v3.md`), sem
rótulo de suporte.

⚠ Nome: no código do `g1_limpo`, `prateleira` é a LAJE (`spec_prateleira`,
`prateleira_meia_xy`). Na v3, "laje" é o suporte aberto e "prateleira" é o suporte com laterais
e teto.

Proposta: uma sequência de alvos macro só, para os dois suportes. O gerador de IK põe cada
alvo de acordo com a geometria.

| alvo | o que vale | na laje | na prateleira |
|---|---|---|---|
| P0 frente | as palmas à frente da caixa, alinhadas às faces laterais, fora do suporte | igual | igual, antes da boca da prateleira |
| P1 entrada | as palmas avançam ao longo das faces laterais | igual | entre a caixa e as laterais, sem tocar |
| P2 pega | como hoje | igual | igual |
| P3 livre | a caixa livre do suporte, nivelada | sobe 5 cm | desliza para a frente até passar a borda, sem tocar laterais nem teto, e então sobe |
| P4 peito | como hoje | igual | igual |

O BOTAR espelha: alinha com a vaga, entra deslizando no piso da prateleira, apoia, solta, e as
mãos saem pela frente.

O que muda além dos alvos:

- contato da CAIXA com laterais, teto e fundo: multa em rampa, como a do corpo com a mesa
  [C31]; o piso do suporte é apoio permitido;
- deslizar a caixa no piso da prateleira passa a ser permitido (o enunciado §4.3 hoje proíbe
  arrastar além do ajuste de pega);
- o tipo de suporte e a folga (lateral e de teto, de larga a justa) viram níveis do balanceador;
  a laje abre primeiro;
- o gerador de IK recebe a geometria do suporte; no robô real ela vem da percepção ou de um mapa;
- sem a grade, a política não distingue os suportes: a prateleira só treina com a chave
  `PERCEPCAO_LIDAR` ligada.

Primeiro passo, sem treino: gerar as poses de IK na prateleira (laterais e teto como obstáculo,
folga de 10 mm), achar a folga lateral mínima em que a mão do G1 entra, e julgar no viewer.

MEDIDO 02/10 (IK do BOTAR dentro da prateleira, sementes da `ref_v3_manipulacao`, regras do
`ik_cintura25` mais folga robô × parede de 10 mm; caixa com a face da frente 2 cm para dentro
da boca; `scratchpad/ik_prateleira.py`, fotos em `g1_limpo/ik/prateleira/`):

| geometria (lateral / teto, por lado) | aceitas | onde falha |
|---|---|---|
| A: 15 / 15 cm | 9 de 15 | lajes ≤ 0,30 m: o teto; o tronco satura em 85° |
| B: 7,5 / 7,5 cm | 4 de 15 | a mão encosta na lateral (0 mm) e o corpo fura a folga em 2 mm: é o limite |
| C: 15 / 30 cm | 13 de 15 | só a laje de 0,04 m com as caixas de 14 e 26 cm |

Na C, o tronco fica igual ou abaixo do da laje aberta aprovada (0,30 m: 48–68° contra 63–70°),
porque o robô agacha mais (pelve 0,62–0,65 contra 0,68–0,69). O que limita a prateleira baixa é
o teto, e não a lateral.

REPROVADO pelo dono, 02/10: a prateleira real tem vários níveis. A parte de cima continua lá:
bate a cabeça e cega os sensores. Nas lajes baixas da geometria C, a cabeça e o tronco passam
por cima do cubo, o que uma estante de vários níveis não deixa. Regra nova: só os braços
entram no vão; tronco, cabeça e pernas ficam à frente do plano da frente da estante. Medido na
pose de pé: ombro a 1,08 m; ombro a palma 0,46 m; com o peito no plano da frente, a palma
alcança o centro da caixa de ~0,67 a ~1,3 m. Abaixo disso a pose de pé e o agachamento com
pés planos não servem (o joelho também não pode passar do plano). Em discussão: a família de
pose dos níveis baixos (agachamento aberto, meio-ajoelhado, ajoelhado).

## 12. Refino por tarefa e lacunas da spec (propostas, 02/10)

### 12.1 Critério para começar a refinar uma tarefa

Proposta: cada tarefa tem o seu contador de refino. A tarefa começa a refinar quando o último
nível dela já abriu e o sucesso médio dos níveis, sem peso pela amostragem, fica em 0,75 ou
mais por 100 its. A média deixa o refino começar mesmo que o nível mais difícil (laje de 0,04 m
com 5 kg) fique um pouco abaixo do portão; um nível que nunca passa não trava a tarefa. Cada
degrau seguinte pede o mesmo sucesso por 100 its no degrau atual; abaixo do portão, o contador
para; ele nunca desce.

Num episódio de cadeia, a regra que vale em cada passo é a da tarefa corrente, com o degrau
dela.

| regra do refino | ANDAR | PEGAR | CARREGAR | BOTAR |
|---|---|---|---|---|
| ~~passo miúdo (`air_time` com duração mínima)~~ SAIU: o passo mínimo vale desde o começo (dono, 05/10; spec §19, conflito 6) | | | | |
| contato com a laje 50 → 10 N | — | sim | — | sim |
| nivelamento 30° → 15° | — | sim | sim | sim |
| peso do `r_aperto` 0 → `w` (dono, 09/10; era "aperto 8× → 2,5× `F_ref`") | — | sim | sim | sim |
| sacudida (aceleração da caixa) | — | sim | sim | sim |
| chegada da caixa 1,0 → 0,25 m/s; σ do pouso 1,0 → 0,35 m/s | — | — | — | sim |
| terminação no batente e `hip_yaw` > 1 rad (opção A, dono, 02/10) | sim | sim | sim | sim |
| magnitude do erro do estimador (§13) | sim | sim | sim | sim |
| DR dos ganhos do PD 0,9–1,1× (depois de alinhar ao kp do deploy) (dono, 06/10) | sim | sim | sim | sim |
| DR do atraso da ação 0–1 → 0–3 passos (dono, 06/10) | sim | sim | sim | sim |
| DR do atrito seco do motor 0–0,05 N·m (dono, 06/10) | sim | sim | sim | sim |

### 12.2 Lacunas

| lacuna | proposta |
|---|---|
| canais do alvo macro | trocar o `alvo_b` de hoje (3 canais, o alvo do elo) pelo alvo macro corrente, no frame da base: as duas palmas (6), a altura da pelve (1) e a inclinação do tronco (1). Com as duas idades de leitura (§13), o bloco da caixa tem 17 canais e continua zerado no ANDAR [C13] (spec §6.1). ATUALIZADO 14/10 (dono): a referência da etapa saiu do ator e fica no crítico e na recompensa; o bloco tem 11 canais: caixa (7), idade da leitura (1) e o alvo do controlador (3); o ator recebe só os comandos de alto nível e os sensores (spec §19.1). O giro da caixa em torno da vertical entra pelas palmas, que giram com ela |
| tolerância de cada alvo | P0 e P1: palmas a até 3 cm do alvo; P2: contato nas duas mãos, sem força mínima (dono, 05/10); P3: caixa 5 cm acima da laje, sem apoio, nivelada no limite do degrau; P4: caixa a até 5 cm do alvo de transporte e pelve a até 3 cm da altura da IK, por 0,5 s; B1: caixa a até 3 cm na horizontal, 5 cm acima do alvo; B2: apoio de 0,8·m·g ou mais, a até 0,10 m do alvo; B3: palmas a 5 cm ou mais das faces e caixa parada por 0,5 s; B4: pose de pé da IK com a mesma tolerância do P4. A calibrar contra o tremor das palmas medido no play |
| P0 sem pose de IK | sai da geometria da caixa: palmas 5 cm à frente da face da frente, alinhadas às faces laterais (y = ±(a + 0,035)), na altura do centro; pelve e tronco do P0 e do P1 são os da IK da altura, como no P2 |
| gerador de IK no treino | interpolação linear da tabela (`ref_agachado.npz` por altura e tamanho; `ref_v3_de_pe_theta-30_tamanhos.npz` por tamanho, a mesma para toda massa; spec §8.1) nas grandezas do alvo (palmas no frame da caixa, altura da pelve, tronco, recuo da raiz), nunca IK por env |
| forma do potencial | `γΦ(s') − Φ(s)`, com Φ de 0 a 1 por alvo (a proximidade ao alvo corrente) e um bônus fixo ao alcançar cada alvo; com Φ acumulado, segurar o P4 por 0,5 s custaria perto de um alvo inteiro (avaliação R2) |
| sucesso do ANDAR | por subtarefa: razão entre o deslocamento projetado e o pedido, sobre o twist inteiro [P3, A11]; o parado: deriva de posição e de rumo abaixo de um limiar |
| pesos | o bônus de cada alvo cobre o custo dos reguladores no trecho até ele, medido no ponto de operação (§5.3); números na spec |

## 13. O alvo no mundo, no sim e no real (proposta, 02/10)

Pedido do dono, 02/10: hoje a caixa e o alvo chegam exatos e de graça (oráculo, 10 canais, sem
ruído nem atraso); no real eles vêm da câmera, do depth e do lidar, e o robô que se mexe tem de
saber onde o alvo está. "Esse é um problema que precisa ser resolvido por agora." Isto muda a
decisão de 24/09 (memória do oráculo: estimador só depois das tarefas).

**A política não precisa do mundo.** Ela recebe todo alvo no frame da base, como hoje. Fora dela,
um estimador guarda cada alvo ancorado no mundo e o reescreve no frame da base a cada passo:
`alvo_base(t) = T_base←mundo(t) · alvo_mundo`. O erro tem duas fontes: a DETECÇÃO (onde o alvo
estava quando foi visto: ruído, atraso, oclusão) e o MOVIMENTO (para onde o robô foi desde então:
deriva da odometria).

| alvo | referência | precisa do mundo? | fonte no real |
|---|---|---|---|
| P0–P3 | a caixa na laje | sim | câmera (tag CAIXA ou as CNNs) + depth; odometria entre leituras |
| P4 e CARREGAR | o peito | não | — |
| caixa na mão | a câmera, como antes da pega (dono, 14/10; a cinemática das mãos depois de `pegou` SAIU: `pegou` é do simulador) | sim | câmera + depth; sem leitura, a última fica e a idade cresce |
| B1–B3 | o alvo de soltura na laje de destino | sim | tag MESA (marco da sessão) ou lidar; odometria |
| B4 e esperas | de pé | não | — |

No PEGAR e no BOTAR o robô quase não sai do lugar, então a deriva da odometria em ~5 s é de
milímetros (ESKF do `lidar_sim`: ATE de 1,5 cm na volta de 4 m). O erro que pesa é o da
detecção: os braços cobrem a caixa na pega, a caixa cobre a laje ao descer, e o depth não vê
abaixo de 0,175–0,28 m.

Proposta:

1. **Contrato**, como o da grade do lidar: uma função de estimador, sem import de simulador,
   igual no sim do real e no robô. A saída por passo: cada alvo no frame da base, a idade da
   última leitura e se a leitura vale.
2. **No treino**, o oráculo vira um estimador simulado com a mesma saída: viés por episódio
   (calibração), ruído por leitura, atraso, leitura retida na taxa do sensor, oclusão (sem leitura
   nova enquanto as mãos cobrem a caixa ou a caixa cobre a laje) e deriva de odometria entre
   leituras. O crítico continua com a verdade [C15].
3. **Caixa na mão** pela câmera, como antes da pega (dono, 14/10). SUPERADO: "pela cinemática das palmas depois de `pegou`".
4. **Canais novos:** a idade da leitura da caixa e a do alvo de soltura (2), no fim do bloco.
5. **Quando entra:** a forma (atraso, taxa, oclusão, idade, troca para a cinemática) desde o
   começo, porque ela muda o laço fechado; as magnitudes (ruído, viés, deriva) crescem no refino
   até o valor medido.
6. **Calibração:** as magnitudes saem do detector da caixa e do ESKF do `lidar_sim`, medidos no
   sim do real contra a pose verdadeira.

DECIDIDO pelo dono, 02/10: a forma do estimador simulado vale desde o começo e a magnitude do
erro cresce no refino, "para deixar ele resiliente"; "o erro tem que ser em relação a
praticamente tudo". A caixa no real é vista por uma câmera RGB, sem marcador, junto com o
depth: um algoritmo detecta a caixa na imagem e outro detecta a orientação.

| o que leva erro | hoje no treino | no refino, até |
|---|---|---|
| posição da caixa | nenhum (oráculo) | erro do depth (±2% da distância até 2 m, D435i) + tremor da detecção; atraso da inferência; leitura retida na taxa da câmera; sem leitura quando os braços cobrem a caixa ou ela está abaixo do alcance mínimo do depth (0,175–0,28 m) |
| viés da câmera na caixa (dono, 06/10) | nenhum | até ±2 cm na posição e até ±2 cm no tamanho, fixos por episódio (calibração), além do tremor por leitura. O erro no tamanho vira erro de penetração do alvo da mão: caixa estimada menor → aperta mais; maior → pega frouxa. A pega tem de se ajustar pelo toque, não pela medida. O toque vem do torque PERCEBIDO pelo robô (dono, 06/10), nunca da força de contato: o ator só vê o que o G1 mede (juntas, ação anterior — o erro alvo − q dá o torque do PD). DECIDIDO (dono, 06/10): sem canal de `tau_est`; o modelo aprende a estimar o toque sozinho pelas 29 juntas. Para a relação erro de posição → força valer no real, a DR do REFINO (§12.1) inclui os ganhos do PD (0,9–1,1×, depois de alinhar ao kp do deploy), o atraso da ação (0–1 → 0–3 passos) e o atrito seco do motor (0–0,05 N·m) — faixas do `docs/relatorios/2026-10-05-dr-para-a-v3.md`; a força de contato fica só na recompensa e no crítico |
| orientação da caixa | nenhum | o orientador medido no `g1-box-vision`: mediana 5,3°, p90 15,3°, 0,2% acima de 45°; o cubo é simétrico, então o erro conta módulo 90° |
| detecção perdida | nenhuma | a taxa medida (o detector achou a caixa em 97,8%) |
| alvo de soltura | nenhum | erro da detecção da laje + deriva da odometria |
| IMU e juntas | ruído da Unitree (giro ±0,2, gravidade ±0,05, posição ±0,01, velocidade de junta ±1,5; sem velocidade linear da base no ator) e viés de encoder ±0,015 (spec §19.1); atraso de 0–1 passo na ação e na propriocepção desde a fase 0 (dono, 14/10, L4.3) | o mesmo, mais atraso de 1–2 passos |
| física | atrito do pé 0,3–1,0 e massa do tronco −1 a +3 kg (Unitree); empurrões; massa da caixa 1–5 kg | mais atrito da caixa e da palma, ganhos do PD, massa e centro de massa do robô, atraso da ação de 0–2 e 0–3 passos |
| one-hot e comando de velocidade | sem erro | sem erro: vêm exatos do controlador; ruído no one-hot faria frações entre tarefas |
