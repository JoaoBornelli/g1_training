# Contrato de troca de tarefa — g1_limpo

**Estado:** v14 EM REVISÃO pelo dono (2026-09-03). A v12 estava aprovada; a revisão de consistência de 03/09 (§6.6.1) achou três buracos de aprendizado e a v13 propõe o conserto (§13, sétima rodada); a v14 fecha o contrato do `REORIENTAR` (§8.3) sem trazer o treino dele para a v2. Nada implementado. Próximo passo: aprovar e escrever o plano de implementação na branch `exp/g1-limpo-v2`.
**Escrito:** 2026-09-02 · **Revisado:** 2026-09-03 (v14 — `ANG` vira `giro_b`, 114 canais; o `REORIENTAR` ganha contrato, quatro primitivas e o desenho do treino, e fica inerte na v2. v13 — a renda do `BOTAR` e da espera final vira monótona; `caiu` por tamanho; crítico com o elo interno; `dr.geom_size` é do próprio mjlab)
**Módulo:** `g1_limpo/`

Este documento existe para que a implementação — e o agente — saibam a todo momento
qual é o objetivo e o que já funciona. Ele **não** é um plano de implementação. O plano
vem depois, e só depois de este documento ser aprovado.

---

## 1. Objetivo macro

O robô sabe fazer **loco-manipulação de uma caixa**.

Isso se divide em duas tarefas:

- **locomoção** — com e sem caixa
- **manipulação** — pegar, botar

E numa terceira coisa, que é o assunto deste documento: **trocar entre elas com o
código rodando**, por comando externo, num modelo só.

### 1.1 Por que um modelo só

A premissa é aproveitar conhecimento prévio entre as tarefas e trocar entre elas **sem
adaptação nenhuma** — sem handover entre modelos diferentes, sem reinicializar
controlador. Se a troca exigir handover, a premissa cai.

⚠ Isto é uma decisão declarada, e ela **exclui** a saída mais simples: dois
controladores com passagem de bastão no instante em que o robô está parado. Essa saída
funcionaria hoje, sem nenhuma mudança, e foi rejeitada de propósito.

---

## 2. O que JÁ FUNCIONA — não quebrar

Medido, não suposto. Números de `bloco7`, iteração ~2514, salvo indicado.

| Coisa | Estado | Como se mede |
|---|---|---|
| **Andar** | funciona | `seg_proj/seg_pedido = 0,739` (livre de diluição) |
| **Sequenciar loco → manipulação** | funciona | `Curriculum/elo = 0,296` — a rampa chegou aos 30% de locomoção |
| **Não esquecer a locomoção** | funciona | a fatia de 30% fica reservada à locomoção pura, para sempre |
| **Recompensas e penalidades** | funcionam | a cadeia `descarga 0,965 → rampa 0,978` prova a pega; a locomoção prova o resto |
| **Pegar a caixa** | funciona parcial | `descarga = unload/squeeze = 0,965`; a caixa sai da laje |
| **Erguer de pé** | funciona parcial | `rampa = postura_ereta/unload = 0,978` |
| **Trazer ao alvo** | **não fecha** | `sustentacao ≈ 0,0005`; `precise_pos` oscila em ~7–9 cm de `tol_pos = 10 cm` |
| **Andar com a caixa** | existe, pouco exercitado | elo `CARREGAR`, 2º elo da cadeia 2 |
| **Botar** | existe, pouco exercitado | elo `BOTAR`, 2º elo da cadeia 3 |

**Regra deste trabalho: nada acima é tocado.** Locomoção, recompensas e penalidades, o
piso de 30% de locomoção, o currículo de forma e de nível, as terminações, o rastreio
por elo e a multa de mesa ficam como estão. **Decisão do dono (02/09), explícita.**

⚠ **As exceções, e cada uma tem dono e data:**

- **a cadeia 3 muda** de `(PEGAR, BOTAR)` para `(PEGAR, CARREGAR, BOTAR)` — o `botar`
  passa a vir de "segurar parado", e não direto da pega (§3.0, §6.5). As outras três
  cadeias não mudam (dono, 02/09);
- **a terminação `caixa_largada` muda em dois pontos**: na espera final, depois do
  botar, as mãos saem da caixa e `escapou` não pode disparar (dono, 02/09); e `caiu`
  passa a ler o tamanho da caixa do env, porque com a caixa variando de tamanho o
  limiar fixo de 0,10 m não acusa a queda da caixa grande (revisão de 03/09, §6.7).
  Fora disso ela não muda;
- **o tamanho da caixa varia por env** (§6.7). A pega foi aprendida com uma caixa de
  20 cm; passa a ser aprendida com 14 a 26 cm. É DR pedida pelo dono **desde o começo**,
  e é a mudança desta spec com mais efeito sobre a manipulação. A caixa **continua `box`
  primitivo**; o colisor não muda (dono, 02/09);
- ⚠⚠ **a tabela de recompensa do `BOTAR` muda, e só a dele** (§6.6.1, §6.6.2). A regra
  "recompensas não se toca" vale para o que foi **medido**: a pega e a locomoção. O
  `BOTAR` nunca fechou em run nenhuma, e a revisão de 03/09 mostrou por quê: com a tabela
  de hoje, **pairar a caixa a 1 cm da laje rende mais do que apoiá-la**, e a espera final
  da v12 alargava essa diferença de 4/s para 11,5/s. O `g1_poc` tinha caído no mesmo
  buraco e o consertou com `load` e duas máscaras; a reescrita perdeu as três. `PEGAR`,
  `CARREGAR`, `REORIENTAR` e a locomoção ficam **bit a bit iguais** (v13, em revisão).

O que este documento muda está **inteiro** na §6: observação, um canal de comando, a
cadeia 3, a terminação `caixa_largada`, a geometria da caixa, e a renda do `BOTAR` e
da espera final. Nenhum currículo.

---

## 3. O cenário de campo

O que se quer poder fazer, em ordem:

1. **Pilotar** o robô por comando de velocidade e ângulo, até uma mesa com uma caixa
   em cima.
2. Na mesa, **enviar o comando de pegar**. O robô passa a considerar a posição da
   caixa e faz a pega.
3. Com a caixa na mão, **as tarefas de locomoção não podem deixar a caixa cair**.
4. **Andar até uma posição** com a caixa, e receber um **alvo para botar**.
5. Depois de largar, o robô **"esquece" a caixa** e volta à locomoção.

### 3.0 Os quatro comportamentos — e só eles

**Princípio (dono, 02/09): treinar somente os comportamentos e transições que o robô de
fato precisa aprender. O restante é derivado — composto pelo controlador, via one-hot.**

| comportamento | o que a rede aprende | onde no treino |
|---|---|---|
| **andar** | rastrear comando linear e angular, em 0 e > 0 | 30% de locomoção pura, mais o ramo de giro (§9) |
| **pegar** | a transição parado (`andar = 0`) → pegar uma caixa | espera publicando `ANDAR` (§6.3); cadeia 0 |
| **carregar** | pegar + andar com o bit de caixa em 1 — **não deixar cair** | cadeia 2 `(PEGAR, CARREGAR)` |
| **botar** | pegar + `andar = 0` segurando + botar num alvo (**não jogar**) + `andar = 0` — **larga e fica parado** | cadeia 3 `(PEGAR, CARREGAR[v=0], BOTAR)` + espera final (§6.5, §6.6) |

Tudo comandado por **um** one-hot, passado pelo controlador. O `REORIENTAR` é habilidade
futura (§8.3) e fica sorteável pelo motivo de lá.

### 3.1 As situações que precisam existir

- robô parado recebe o comando para pegar
- robô anda com a caixa (andar e parar com ela na mão)
- robô com a caixa na mão, parado, bota em algum lugar

### 3.2 As transições que PRECISAM ser aprendidas

```
andar v=0        ->  pegar                TREINADA  (a espera publica ANDAR, §6.3)
pegar            ->  carregar, v>0        TREINADA  (cadeia 2: andar com a caixa)
pegar            ->  carregar, v=0        TREINADA  (cadeia 3: segurar parado, §6.5)
carregar, v=0    ->  botar                TREINADA  (cadeia 3, §6.5)
botar            ->  andar v=0, sem caixa TREINADA  (espera FINAL, §6.6)
```

**Nenhuma transição do cenário fica como aposta.**

⚠ **`pegar -> botar` direto NÃO é treinado.** Decisão do dono (02/09): é comportamento
derivado — o controlador nunca manda `BOTAR` a partir de `PEGAR`; ele passa por
`CARREGAR` com `v = 0` primeiro. A cadeia 3 de hoje, `(PEGAR, BOTAR)`, treinava uma
transição que não existe em campo.

### 3.3 A transição que NÃO precisa

**Andar até a mesa.** O robô não precisa aprender a caminhar com uma mesa à frente.
**Decisão do dono (02/09): isso NÃO é objetivo deste modelo** — é de outro modelo ou de
outra abordagem, se vier a ser. A razão técnica está na §7.3.

---

## 4. O CONTRATO DE ENTRADA DA REDE

Esta seção é a que importa em campo. Ela lista **tudo** que a rede recebe, e de onde
cada coisa vem no mundo real.

A observação do `actor` tem **114 canais**, nesta ordem:

| # | canal | dim | de onde vem em campo |
|---|---|---|---|
| 0 | `base_lin_vel` | 3 | IMU, a bordo |
| 1 | `base_ang_vel` | 3 | IMU, a bordo |
| 2 | `projected_gravity` | 3 | IMU, a bordo |
| 3 | `joint_pos` | 29 | encoders, a bordo |
| 4 | `joint_vel` | 29 | encoders, a bordo |
| 5 | `actions` | 29 | última ação, a bordo |
| 6 | `command` (twist) | 3 | **EXTERNO** — o piloto: `vx, vy, wz` |
| 7 | `elo` (one-hot) | 5 | **EXTERNO** — o botão de tarefa |
| 8 | `caixa` | 10 | ver §4.1 |

Total: `3+3+3+29+29+29+3+5+10 = 114`. (O **crítico** tem 119: os mesmos 114 mais o
`elo_interno` da §6.1. Ele não vai para o robô.)

⚠ **112 antes, 114 depois.** O canal `VALIDA` **sai** da observação (§6.2); ele continua
dentro do sim como porta de recompensa e de fecho de elo. O canal `meia_aresta` — o
tamanho da caixa — **entra**, no fim do termo `caixa` (§6.7). E o escalar `ANG` vira o
vetor `giro_b` (§8.3): um canal vira três. O significado da primeira camada muda, e o
checkpoint antigo não serve (§12).

### 4.1 Os 10 canais de caixa, em detalhe

| canal | dim | o que é | de onde vem em campo |
|---|---|---|---|
| `caixa_b` | 3 | **posição** da caixa, no frame da base | **PERCEPÇÃO** |
| `alvo_b` | 3 | **posição** do alvo, no frame da base | calculado A BORDO (exceto `BOTAR`) |
| `giro_b` | 3 | **vetor de giro** (eixo × ângulo, rad): a rotação que leva a normal atual da face pedida à direção pedida, no frame da base. `|giro_b|` é o erro angular | calculado A BORDO, de percepção + tarefa (§8.3) |
| `meia_aresta` | 1 | **escalar**, em metros: meio-lado da caixa (§6.7) | **PERCEPÇÃO** — o bounding box |

⚠⚠ **Os 10 canais são ZERO quando o one-hot publicado é `ANDAR`**, e vivos em todo outro
caso. Não existe terceiro estado. Esta é a **invariante que substitui o bit**, e é o que
a rede lê como "existe tarefa de caixa": canais preenchidos, ou canais em zero.

**O tamanho é input, e é decisão do dono (02/09).** Sem ele a política chutaria a abertura
das mãos — ver a tabela da §6.7.

**A posição da caixa é input em todo elo de manipulação** — `PEGAR` incluído. Ela é o
canal que guia o alcance. O que NÃO é input externo é o `alvo_b`, na maioria dos elos.

#### O `alvo_b` do `PEGAR` e do `CARREGAR`, em detalhe

Ele é calculado a bordo de dois constantes e da pose do próprio robô:

```
x, y   base + peito_b, girado pelo quat da base   ->  CONSTANTE no frame da base
z      altura_carregar, ABSOLUTO                  ->  no frame da base vira
                                                      (altura_carregar − z_da_base)
```

⚠ **O z absoluto é deliberado, e não é detalhe.** Se fosse relativo à base, o robô
satisfaria o alvo **andando agachado** — o alvo desceria junto com a pelve e a caixa
nunca precisaria subir. Com o z absoluto, o canal `alvo_b[2]` carrega **quanto o robô
está agachado**, e é o freio anti-agachamento.

Portanto `alvo_b` no `PEGAR`/`CARREGAR` **não é canal morto**: x e y são constantes,
mas o z é informação viva. Em campo ele sai de `peito_b`, `altura_carregar` e da
odometria de altura do robô — nada externo.

⚠ **Correções ao modelo mental, e elas mudam o que se manda em campo:**

- **Não existe quatérnion na entrada.** A orientação da caixa entra como **um vetor de
  giro** (`giro_b`, 3): o que falta girar, e em torno de que eixo. O quatérnion é insumo
  do cálculo a bordo, não entrada da rede. ⚠ Até a v13 havia aqui um escalar (`ANG`); a
  v14 o trocou porque um escalar diz quanto falta e não diz para que lado (§8.3).
- **O alvo não é enviado**, em quase todos os casos. No `PEGAR` e no `CARREGAR` ele é
  **ancorado na base do robô** — um offset fixo no frame dele, calculado a cada passo.
  Só o `BOTAR` precisa de alvo externo.
- **Tudo em frame da base**, e não em mundo. Isso dispensa origem global; em troca, a
  percepção tem de ser egocêntrica ou transformada.
- **`giro_b` precisa saber QUAL face.** Em sim a face é a marcada (`face_alvo_b`),
  constante. Em campo é o controlador quem escolhe a face (a próxima a explorar), e a
  percepção tem de identificar as faces da caixa — fiducial, ou geometria conhecida. Sem
  isso o `giro_b` não é computável. A face escolhida **não** entra na rede: para um cubo
  o movimento não depende dela (§8.3).
- **Não existe bit "a caixa existe".** Ele era redundante com o one-hot (§6.2). Em campo,
  "a caixa existe" é a camada de tarefa **preencher** os 10 canais ou **zerá-los** — a
  mesma regra que o sim aplica.

### 4.2 O que o operador manda, no total

```
twist        3 números    vx, vy, wz      (piloto, contínuo)
elo          5 números    one-hot         (botão de tarefa)
alvo BOTAR   3 números    só no BOTAR
                          -----
                          11 números, no máximo
```

Mais, da percepção: **posição da caixa (3)**, **orientação da caixa (4)** e **tamanho da
caixa (1)** — os dois primeiros para o cálculo a bordo de `caixa_b` e `giro_b`, o terceiro
entra direto como `meia_aresta`.

#### As regras da camada de tarefa — do lado do robô real

Quatro regras, e todas espelham algo que o **sim** faz por conta própria. Em campo,
**nada as faz sozinho** — a camada de tarefa tem de implementá-las.

1. **`elo = ANDAR` ⟹ os 10 canais de caixa em zero.** Sempre, mesmo com a caixa à vista da
   percepção. É a regra que substitui o bit. (Sim: o gate da §6.1.)
2. ⚠ **`elo ∈ {PEGAR, REORIENTAR, BOTAR}` ⟹ twist forçado a zero**, seja o que for que o
   piloto mande. (Sim: `_zera_twist_nos_parados`.) Sem isso, um joystick encostado
   durante o `PEGAR` põe a política fora de distribuição — ela nunca viu twist ≠ 0 nesses
   elos.
3. **`alvo_b` é calculado a bordo** em `PEGAR` e `CARREGAR`:
   `(peito_b.x, peito_b.y, altura_carregar − z_base)` = `(0,25, 0,00, 0,95 − z_base)`.
   É **enviado** só no `BOTAR`. (Sim: `_alvo_ancorado_na_base`.)
4. **`giro_b` é calculado a bordo** do quatérnion da caixa (percepção), da face escolhida
   pelo controlador e da direção pedida (horizontal, da caixa para o robô, §8.3).
   Recalculado a cada quadro, em malha fechada: ele encolhe conforme o robô gira. (Sim:
   `_atualiza_face`.)

Sem o bit, não há segundo canal que possa contradizer a regra 1.

### 4.3 A tabela de tarefas (`elo`)

| slot | elo | twist | canais de caixa | alvo | no cenário de campo |
|---|---|---|---|---|---|
| 0 | `ANDAR` | **ativo** — ou zero, na espera de um episódio de manipulação (§6.3) | **zero** | — | pilotar sem caixa; e o "parado antes de pegar" |
| 1 | `REORIENTAR` | zero | vivos | a própria caixa | habilidade futura (§8.3): bota na mesa, gira, pega de novo |
| 2 | `PEGAR` | zero | vivos | ancorado na base | pegar a caixa |
| 3 | `CARREGAR` | **ativo** — ou zero, na cadeia 3 (§6.5) | vivos | ancorado na base, altura do peito | andar com a caixa; e o "segurar parado" antes de botar |
| 4 | `BOTAR` | zero | vivos | topo novo, **externo** | botar a caixa |

⚠ **Um one-hot, e não dois.** A proposta de separar "tarefa" e "existe caixa" em dois
one-hots é redundante com esta tabela: o `CARREGAR` já É "locomoção com caixa", e o
`ANDAR` já É "locomoção sem caixa". A combinação está codificada no one-hot. E o
"existe caixa" é os canais estarem preenchidos — ver §4.1.

⚠ **`andar` é ±1,0 m/s hoje**, não ±2. `lin_vel_x = ±1,0`, `lin_vel_y = ±1,0`,
`ang_vel_z = ±0,5`. Se ±2 m/s é requisito de campo, é mudança de faixa de comando —
item separado, e ela mexe na locomoção, que hoje funciona.

### 4.4 Uma movimentação completa, canal a canal

Dos 114 canais, **96 são proprioceptivos e automáticos** (IMU, encoders, última ação; a
v13 dizia 85, conta errada). Os **18 que carregam tarefa** são montados pela camada de
tarefa a cada passo, a 50 Hz:

```
command   3   vx  vy  wz
elo       5   [ANDAR, REORIENTAR, PEGAR, CARREGAR, BOTAR]
caixa    10   caixa_b(3)  alvo_b(3)  giro_b(3)  meia_aresta(1)
```

Valores de exemplo com os constantes reais: `peito_b = (0,25, 0, 0,15)`,
`altura_carregar = 0,95`, pelve de pé a ~0,80 m, laje a 0,55 m, caixa de 20 cm.

```
FASE 1 — ANDAR até a mesa (piloto)
  twist    ( 0,50   0,00   0,00 )      PILOTO
  elo      [ 1  0  0  0  0 ]            OPERADOR
  caixa_b  ( 0,00   0,00   0,00 )      camada ZERA  <- a percepção pode já ver a caixa; não passa
  alvo_b   ( 0,00   0,00   0,00 )      zera
  giro_b   ( 0,00   0,00   0,00 )      zera
  meia       0,00                       zera

FASE 2 — parado na mesa (v = 0)
  twist    ( 0,00   0,00   0,00 )      PILOTO zera
  elo      [ 1  0  0  0  0 ]            ainda ANDAR
  caixa    tudo 0                       <- é o que o treino produz na "espera" (§6.3)
  duração: o operador decide

FASE 3 — PEGAR                          <- o botão
  twist    ( 0,00   0,00   0,00 )      camada FORÇA zero (regra 2)
  elo      [ 0  0  1  0  0 ]            OPERADOR
  caixa_b  ( 0,55   0,00  -0,15 )      PERCEPÇÃO: 55 cm à frente, 15 cm abaixo da pelve
  alvo_b   ( 0,25   0,00  +0,15 )      A BORDO: (0,25, 0, 0,95 − 0,80)
  giro_b   ( 0,00   0,00   0,12 )      A BORDO: torção da caixa desde a abertura do elo, aqui em Z
  meia       0,10 m                     PERCEPÇÃO: meio-lado da caixa — esta é de 20 cm
  fim: caixa erguida e segura — no sim, condição sustentada 0,5 s; em campo, o operador vê

FASE 4 — CARREGAR (andar com a caixa)
  twist    ( 0,40   0,00   0,10 )      PILOTO
  elo      [ 0  0  0  1  0 ]            OPERADOR
  caixa_b  ( 0,24   0,01  +0,12 )      PERCEPÇÃO: agora perto do peito
  alvo_b   ( 0,25   0,00  +0,15 )      A BORDO: mesma âncora — a caixa deve ficar nela
  giro_b   ( 0,00   0,02   0,05 )      A BORDO
  meia       0,10 m                     PERCEPÇÃO

FASE 5 — parado no destino, caixa na mão (v = 0)
  twist    ( 0,00   0,00   0,00 )      PILOTO zera
  elo      [ 0  0  0  1  0 ]            ainda CARREGAR — TREINADO: é a fase do meio da cadeia 3 (§6.5)
  caixa    vivos, como acima

FASE 6 — BOTAR                          <- o botão
  twist    ( 0,00   0,00   0,00 )      camada FORÇA zero (regra 2)
  elo      [ 0  0  0  0  1 ]            OPERADOR
  caixa_b  ( 0,24   0,01  +0,12 )      PERCEPÇÃO
  alvo_b   ( 0,35   0,00  -0,20 )      EXTERNO — o único alvo enviado: onde botar, no frame da base
  giro_b   ( 0,00   0,02   0,05 )      A BORDO: torção desde a abertura do BOTAR
  meia       0,10 m                     PERCEPÇÃO
  fim: caixa apoiada — no sim, força de apoio ≥ fração do peso; em campo, o operador vê

FASE 7 — ANDAR v=0 (larga e fica parado — a ESPERA final)
  twist    ( 0,00   0,00   0,00 )      PILOTO zera — o controlador manda ANDAR com v=0
  elo      [ 1  0  0  0  0 ]            OPERADOR
  caixa    tudo 0                       <- a percepção AINDA vê a caixa na laje; a camada zera (regra 1)
  o robô solta a caixa e volta à postura de pé. TREINADO: é a espera final da cadeia 3 (§6.6)

FASE 8 — ANDAR (piloto retoma)
  twist    ( 0,50   0,00   0,00 )      PILOTO
  elo      [ 1  0  0  0  0 ]
  caixa    tudo 0
```

⚠ **A rede NÃO decide quando trocar.** Quem aperta o botão é o operador, ou o algoritmo
que vier a ser escrito. Os instantes que importam: `PEGAR → CARREGAR` quando a caixa está
erguida e segura; `CARREGAR → BOTAR` depois de parar no destino; `BOTAR → ANDAR` com a
caixa apoiada. No sim esses instantes são as condições de sustain das cadeias; em campo
são olho ou sensor. Um algoritmo de troca automática é trabalho separado (§10, "o robô
decidir sozinho").

**Quais dessas trocas o treino produz** está na §7: 2→3, 3→4, 4→5, 5→6 e 6→7 são
**todas** treinadas. 7→8 é só o piloto retomar o comando de velocidade — locomoção pura.
⚠ 3→6 direto (`PEGAR → BOTAR`) não é treinado, e o controlador não o manda (§3.2).

---

## 5. O DEFEITO que impede a troca

⚠ **A observação vaza a caixa no `ANDAR`.** Medido:

```
elo = ANDAR      |caixa_b| = 4,32
```

`caixa_b`, `alvo_b` e `ANG` são publicados **sem gate**. A laje vai a +5 m no `ANDAR`
por razão **física** — não ficar na frente do robô —, mas esses 5 m estão entrando como
**informação**.

Consequência: no treino, `elo = ANDAR` ⟹ caixa a 5 m, **sempre**. Portanto a política
pode ter aprendido "ando" a partir de *a caixa está longe*, e não do one-hot. Em campo,
com a caixa numa mesa real a 0,6 m e o elo em `ANDAR`, ela vê uma combinação que nunca
existiu.

**Medido:** foi exatamente isso que fez o robô sambar quando se tentou entregar a tarefa
ao vivo com a caixa perto.

⚠ O `g1_poc` **não** tem esse furo: com o bit em zero ele zera as fatias de caixa
(`g1_poc/comando.py:390-399`).

---

## 6. O QUE MUDA

Sete grupos de mudança: **observação e canal de comando** (gate, `VALIDA` fora,
`meia_aresta` e `giro_b` dentro, `elo_interno` no crítico, publicado `ANDAR` nas duas
esperas), a **cadeia 3**, a **terminação `caixa_largada`** (guarda e `caiu` por tamanho),
a **geometria da caixa** (DR de tamanho), a **renda do `BOTAR` e da espera final**
(§6.6.1, §6.6.2), o **ramo de giro** da locomoção (§9) e o **contrato do `REORIENTAR`**
(§8.3). Nenhuma em currículo.

### 6.0 Os dois `elo` — o PUBLICADO e o INTERNO

A mudança da §6.3 depende de uma separação que **já existe** no código e que este
documento torna explícita. Medido em 02/09, por leitura de fonte:

| quem | lê | lado |
|---|---|---|
| observação `um_de_cinco` (ator e crítico) | `comando[:, ELO]` | **publicado** |
| observação `elo_interno` (**só o crítico**, §6.1) | `limpo_elo` | interno |
| gate dos canais de caixa (§6.1) | `comando[:, ELO]` | **publicado** |
| `PosturaPorElo` | `comando[:, ELO]` | **publicado** |
| `rastreio_por_elo` (`_anda_neste_elo`) | `comando[:, ELO]` | **publicado** |
| os sete incentivos, `load`, `largou` | `VALIDA` = `interno ≠ ANDAR ∧ ¬aguardando` | interno |
| máscaras do `BOTAR` (§6.6.2) | `self._elo` | interno |
| `PPOPorElo` (normalização de vantagem por grupo) | o `elo_interno` do crítico | interno |
| `_zera_twist_nos_parados` | `self._elo` | interno |
| `_aplica_elo` (laje, alvo, face) | `self._elo` | interno |
| `_fecha_elo_corrente`, `_avanca_elo` | `self._elo` | interno |
| `reset_base_por_elo`, fatia (`Curriculum/elo`) | `limpo_elo` do currículo | interno |

**O que a política vê lê o publicado. O que paga e o que é mecânica do episódio lê o
interno.** Hoje os dois são sempre iguais (`_aplica_elo` copia um no outro). A §6.3 e a
§6.6 os fazem **diferir durante as duas esperas** — e a tabela acima é o que garante que
cada consumidor lê o lado certo.

⚠ **O `VALIDA` NÃO muda de fonte** (a v12 dizia o contrário, e a revisão de 03/09
desfez). Ele continua `interno ≠ ANDAR ∧ ¬aguardando`, que é o que `_aplica_espera` já
calcula hoje. Consequência que importa: na espera **inicial** ele é 0 (aguardando), e na
espera **final** ele é **1** — os incentivos do estado "caixa apoiada no alvo" continuam
pagando depois do fecho do `BOTAR`. É isso que fecha o buraco da renda (§6.6.1). A
observação continua sem o bit (§6.2); o que a rede vê é só o one-hot publicado e os
canais de caixa gateados.

### 6.1 Gate dos canais de caixa

`caixa_b`, `alvo_b`, `giro_b` e `meia_aresta` vão a **zero** quando o **elo publicado** é
`ANDAR`. Vivos em todo outro caso. Não existe terceiro estado.

Consequências:

- Em `ANDAR`, os canais de caixa são **sempre zero**, independente de onde a caixa
  esteja fisicamente. Os 5 m voltam a ser só física.
- O one-hot passa a ser o **único** portador de identidade de tarefa.
- Em campo, o robô "esquece" a caixa literalmente: `elo = ANDAR` e ela desaparece da
  entrada. É o item 5 do cenário (§3), de graça.
- ⚠ **`ANDAR` publicado com a caixa fisicamente perto passa a ser indistinguível de
  `ANDAR` com a caixa a 5 m.** É isso que torna a §6.3 possível, e é isso que faltava ao
  modo de entrega que sambou.

**Onde:** `g1_limpo/observacoes.py`, função `caixa_no_frame_da_base`. Nos dois grupos,
`actor` e `critic`.

⚠ **O gate vale para o crítico também, mas o crítico ganha o elo INTERNO** (v13). Um
termo novo `elo_interno`, one-hot de 5 do `limpo_elo`, **só no grupo `critic`**, em
APPEND depois do `caixa`. Motivo, medido em 03/09 pela aritmética da §6.6.1: na espera
final o robô rende cerca de 18/s; um env `standing` da locomoção rende 6/s com a
**mesma** observação do ator (`ANDAR`, twist zero, caixa zero). Um crítico que vê só o
que o ator vê não separa os dois, e a função de valor erra nos dois lados — o mesmo
estado, dois retornos. Com o interno ele separa espera inicial, `standing` e espera
final. É ator-crítico assimétrico, padrão, e **não toca o deploy**: o crítico não vai
para o robô. A observação do ator fica com 114 canais; a do crítico, com 119. E
`aguardando` e `soltou` **não** precisam de canal próprio no crítico: os dois são
`interno ≠ publicado`, e o crítico vê os dois one-hots.

⚠ **E o `PPOPorElo` passa a agrupar pelo `elo_interno` do crítico**, não pelo one-hot do
ator. Senão a espera final — que carrega retorno de manipulação — entra no grupo da
locomoção e infla o desvio dele, que é exatamente o defeito que o `PPOPorElo` existe para
evitar. `fatia_do_elo` ganha o par para o grupo do crítico.

### 6.2 O `VALIDA` sai da OBSERVAÇÃO

O `VALIDA` é um bit **derivado**: `interno ≠ ANDAR ∧ ¬aguardando`. Para o que a rede
precisa saber, ele não acrescenta nada: durante a espera inicial o one-hot publicado já
diz `ANDAR` e os canais de caixa já estão em zero (§6.1); fora dela o one-hot já diz o
elo. Na espera final ele vale 1 com o publicado em `ANDAR` (§6.0) — e é justamente aí
que ele **não pode** estar na observação, porque em campo ninguém o mandaria.

**Sai da observação. Fica dentro do sim**, como a porta que multiplica os incentivos de
caixa e que impede o fecho de elo com o objetivo desligado. Um nome mais honesto para o
que ele é: `objetivo_ativo`.

**A favor de tirar:**

- some o risco de **incoerência em campo** — com dois canais (`elo` e o bit) o operador
  teria de mantê-los coerentes; sem o bit, não há par para desalinhar;
- um número a menos em campo (§4.2);
- a política deixa de ter um atalho: ela tem de ler o one-hot e a geometria, que é o que
  o deploy fornece.

⚠ **Contra tirar, e é real:** um bit é feature **linear**; "este vetor está em zero" não
é. E depois do normalizador do `rsl_rl` zero **não é zero** — vira `−μ/σ` por canal, uma
constante por canal. A rede tem de aprender "canais nesta constante = sem alvo".

**Por que a objeção é pequena:** a recompensa ensina exatamente essa distinção. Sem alvo,
alcançar em direção ao "zero fantasma" não rende nada de tarefa e custa rastreio (a
locomoção paga por ficar parado). E a separação de magnitude entre `|caixa_b| ≈
0,5–1,0 m` normalizado e a constante do zero é grande. O `g1_poc` mantém o bit — tirar é
divergir do precedente que funcionou. Aceito, com a trava da §11.1 item 2.

**Tirar um canal é de graça agora:** o gate já força reinício (§12). O `meia_aresta` da §6.7 entra no lugar, e o `giro_b` da §8.3 acrescenta dois: 114 canais.

### 6.3 A espera publica `ANDAR` — a transição TREINADA

Hoje, num episódio de manipulação, a janela de espera (`espera_s`, 0,3 a 1,0 s) mantém
`VALIDA = 0` com o one-hot já em `PEGAR`. **A mudança: durante a espera, o one-hot
publicado é `ANDAR`.** O interno segue `PEGAR`.

O que a política vê num episódio de manipulação com espera:

```
passo 0 .. espera:    one-hot ANDAR   twist 0   canais de caixa 0    <- "parado, sem tarefa"
borda:                one-hot PEGAR   twist 0   canais de caixa VIVOS <- "a tarefa chegou"
```

**É a situação 1 da §3.1 — robô parado recebe o comando de pegar — treinada em todo
episódio de manipulação.** E é exatamente o que o operador faz em campo: robô em `ANDAR`
com `v = 0`, e então o botão.

Por que isso basta, sem cadeia nova e sem tocar currículo — cada coisa que a espera
precisa **já está certa pelo elo interno**:

| a espera precisa de | quem garante | lê |
|---|---|---|
| twist zero | `_zera_twist_nos_parados` — `PEGAR ∈ elos_parados` | interno |
| mobília presente, invisível | `_aplica_elo` ramo `PEGAR` não guarda a laje; o gate zera os canais | interno + publicado |
| robô na mesa, de frente | `reset_base_por_elo` usa `faixa_manipula` para `PEGAR` | interno |
| contar como manipulação na fatia | `Curriculum/elo` lê `limpo_elo = PEGAR` | interno |
| rastreio pagar por ficar parado | `rastreio_por_elo` lê publicado = `ANDAR` ∈ `ELOS_QUE_ANDAM` | publicado |
| postura do fabricante, não neutra | `PosturaPorElo` lê publicado = `ANDAR` | publicado |
| incentivos de caixa em zero | `objetivo_ativo` = interno ≠ `ANDAR` ∧ ¬aguardando = falso (aguardando) | interno |
| o elo não fechar durante a espera | `_fecha_elo_corrente & objetivo_ativo` | interno |

**MEDIDO em 03/09, em CPU, sem implementar nada** — robô travado, `elo = PEGAR`, 8 envs.
Uma linha, `_command[:, ELO] = ANDAR`, com o interno em `PEGAR`:

```
                          obs one-hot   interno   VALIDA   twist   track_lin   staged
PEGAR normal              pegar         pegar     1        0,000   +0,0000     +2,055
passo 1, publicado=ANDAR  andar         pegar     1        0,000   +1,8326     +2,055
passo 3, publicado=ANDAR  andar         pegar     1        0,000   +1,8326     +2,055
```

A política vê `ANDAR` no passo seguinte e continua vendo (nada reescreve o canal a cada
passo — `_aplica_elo` só escreve no reset e no avanço). O twist fica zero (interno
`PEGAR`). O rastreio passa a pagar 1,83/s por velocidade zero (lê o publicado). O interno
não se move. **É a §6.0 funcionando sem nenhum código novo.**

Os dois últimos campos da tabela (`VALIDA = 1`, `staged` pagando) são **artefato da
sonda**, não defeito: a sonda escreveu o canal à mão, sem armar a espera. No treino a
espera inicial arma `_espera > 0`, e o `_aplica_espera` de hoje já zera o `VALIDA` por
`aguardando`. ⚠ A v12 concluía daqui que o `VALIDA` tinha de mudar de fonte para o
publicado; a revisão de 03/09 desfez isso (§6.0): a fonte fica no interno, porque na
espera **final** o bit precisa valer 1 (§6.6.1).

E o avanço de cadeia de hoje é a prova de que a política já vive com o one-hot mudando no
meio do episódio: `forca_avanco` em `(PEGAR, CARREGAR)` → a observação vai de
`[0,0,1,0,0]` para `[0,0,0,1,0]` no passo seguinte, e o `track_lin` de 0 para 1,83.

⚠ **A v2 deste documento propunha uma cadeia nova `(ANDAR, PEGAR)`, com o sorteio da
cadeia migrando para o currículo e a fatia excluindo-a.** Estava superconstruído: tudo
isso existe para fazer um `ANDAR` interno se comportar como manipulação — e o `PEGAR`
interno **já** se comporta assim. A única coisa que o operador vê diferente é o one-hot,
e o one-hot é publicado. Uma condição em `_aplica_espera`.

Por que é melhor que a janela de hoje:

| | janela (hoje) | espera publicando `ANDAR` |
|---|---|---|
| one-hot durante a espera | `PEGAR` | **`ANDAR`** — o que o operador ainda não mudou |
| a transição `ANDAR(v=0) → PEGAR` | aposta | **treinada** — todo episódio de manipulação |
| postura no instante da troca | resíduo não treinado | **treinada** — a postura real de parado |
| observação durante a espera | inédita (PEGAR + caixa viva + sem prêmio) | **idêntica** à de um env `standing` da locomoção — o conhecimento é reaproveitado |

**O que `espera_s` passa a significar.** Hoje, no código, é "`PEGAR` com o objetivo
desligado por 0,3 a 1,0 s". Depois da mudança é **"tempo em `ANDAR` com twist zero antes
de o `PEGAR` chegar"** — mesmo buffer, mesmo sorteio, one-hot publicado diferente.
Decisão do dono (02/09): é esta segunda coisa. **Faixa: 0,5 a 1,5 s**, sorteada para a
política não contar passos. E a faixa é **uma só para toda espera** — ver §6.5: o
"segurar parado" antes do botar usa o mesmo knob. Regra do dono: toda fase de "andar
dentro da manipulação com velocidade zero" espera de 0,5 a 1,5 s antes de o comando
trocar.

⚠ **O `_aplica_espera` recalcula o publicado a partir do interno, e não do próprio
publicado.** É a lição do bit destrutivo de 02/09: ler o que se escreveu no passo
anterior deixa o canal preso. `publicado = ANDAR se (aguardando ∨ soltou), senão interno`.

⚠ **A arma do `caixa_largada` só arma com o objetivo ativo** (v13). Hoje `_publica_pegou`
acumula o toque das duas palmas em qualquer passo. Na espera inicial, um toque na caixa
por exploração armaria `escapou` com as palmas longe, e o episódio morreria por ter
esperado. `_pegou |= tocou & objetivo_ativo`.

⚠ **O vazamento físico não morde aqui.** O robô não anda durante a espera (twist zero
pelo interno). Se a política ignorar o comando zero e derivar para a mesa, o rastreio
cobra e o `contato_tronco` cobra. É a pressão certa: "fique parado perto da mesa" **é** o
estado de campo.

### 6.4 O que SAI

- **O `VALIDA` da observação** (§6.2), e o escalar `ANG`, que vira o vetor `giro_b`
  (§8.3). A janela **fica** — ela é o mecanismo da §6.3, com o one-hot publicado trocado.
- A métrica `fracao_esperando` **fica**, e passa a ler `aguardando ∨ soltou` — "fração de
  passos em `ANDAR` publicado dentro de um episódio de manipulação", as duas esperas.
- O `recebe_tarefa` (caminho do visualizador) passa a escrever `ELO = ANDAR` além de
  rearmar a espera; a linha `VALIDA = 0` dele continua certa. Uma linha em `comando.py`;
  o evento `entrega_tarefa_no_viewer` fica intocado.
- O `rastreio_por_elo` ler `limpo_aguardando` (mudança de 02/09) **fica redundante**:
  com o publicado em `ANDAR`, `_anda_neste_elo` já devolve verdadeiro. Sai por limpeza,
  com a trava de que o rastreio paga na espera (§11.1 item 5).
- **O visualizador fica como está.** Decisão do dono (02/09): o modo de entrega e a task
  `Mjlab-G1-Limpo-Entrega` **não são tocados** na v2. Para olhar a transição, `PEGAR`
  forçado com `espera_s` alongado no cfg de play já mostra `ANDAR` publicado e depois
  `PEGAR`.

### 6.5 A cadeia 3 vira `(PEGAR, CARREGAR, BOTAR)` — o botar vem de "segurar parado"

Hoje: `CADEIAS[3] = (PEGAR, BOTAR)`. O one-hot vai direto da pega ao botar.

**Decisão do dono (02/09):** isso treina uma transição que **não existe em campo**. O
controlador, depois de uma pega bem-sucedida, manda sempre `CARREGAR` (caixa na mão, modo
locomoção). O `BOTAR` só chega quando o robô está em `CARREGAR` **com `v = 0`**. A cadeia
tem de ser essa:

```
CADEIAS[3] = (PEGAR, CARREGAR, BOTAR)        # pegar, segurar parado, botar
```

O que a política vê nos três elos:

```
PEGAR         one-hot PEGAR      twist 0    alvo = âncora do peito
CARREGAR      one-hot CARREGAR   twist 0    alvo = âncora do peito     <- "segurar parado"
BOTAR         one-hot BOTAR      twist 0    alvo = topo novo, externo
```

E a transição `CARREGAR(v=0) → BOTAR` sai da aposta e vira **treinada**.

**O que a cadeia exige — e é mais que uma linha:**

1. **`_TETO_ELOS` vira 3.** Ele é derivado de `CADEIAS` (`max(len)`), e `_ELO_EM` é
   preenchido de forma genérica. ⚠ **A verificar na implementação:** `_avanca_elo`,
   `_avanca_elo_force`, `passo_final` e o inspetor de cadeia foram escritos com 2 elos em
   mente — o código diz "derivado, nunca redigitado", mas nenhum teste hoje exercita 3.
   É o item com risco de regressão nesta spec.
2. **Twist zero no `CARREGAR` da cadeia 3.** `CARREGAR ∈ ELOS_QUE_ANDAM`, portanto o
   twist é sorteado. Na cadeia 3 ele tem de ser **zero** ("robô não deve andar de verdade
   com a caixa" antes do botar). Regra **por cadeia** no `_zera_twist_nos_parados`:
   `elo ∈ parados **ou** (elo == CARREGAR e a cadeia é a de segurar parado)`. Na cadeia 2
   o `CARREGAR` continua andando. ⚠ Sem redigitar o índice `3` no comando: a marca "esta
   cadeia segura parado" é uma tabela derivada ao lado de `CADEIAS`, pela mesma regra que
   proíbe tabela paralela escrita à mão.
3. ⚠ **O fecho do `CARREGAR` da cadeia 3 é `perto` SUSTENTADO pela espera, não por
   distância.** Hoje o `CARREGAR` fecha quando o robô **andou** `carregar_dist_m` — com
   twist zero ele nunca fecharia e o `BOTAR` nunca chegaria. Na cadeia 3 a condição é
   `perto` (a caixa na âncora do peito, sem o `andou`), e o **sustain** desse elo é a
   espera sorteada do **mesmo knob `espera_s = (0,5, 1,5)`** da espera inicial, em vez de
   `carregar_s` (decisão do dono, 02/09: toda espera é a mesma faixa). Assim o elo só
   fecha se a caixa ficou na âncora, com o robô parado, por toda a espera — que é
   "segurar parado" medido, e não só tempo passado. ⚠ A v12 dizia "fecho por tempo" sem
   `perto`; a revisão de 03/09 apontou que aí o `BOTAR` podia começar com a caixa em
   qualquer lugar. Regra **por cadeia** no `_fecha_elo_corrente` e no `_avanca_elo`. Não
   existe `segurar_s` separado.
4. **`prob_por_nivel` não muda de forma** — continuam 4 cadeias.

**O que NÃO muda:** recompensa nenhuma. Durante o "segurar parado", o rastreio paga por
velocidade zero (`CARREGAR ∈ ELOS_QUE_ANDAM`), `sustentacao` e `precise_pos` pagam por
manter a caixa na âncora do peito, e `caixa_largada` termina o episódio se ela cair. É
exatamente o "não deixar cair, parado" pedido. A mobília: o ramo `CARREGAR` do
`_aplica_elo` guarda a laje (a caixa fica na mão); o ramo `BOTAR` traz um topo novo — é o
que a cadeia 3 de hoje já faz.

⚠ **Duas regras por cadeia é o custo.** Até aqui nenhuma regra do comando dependia da
cadeia — só do elo. A alternativa seria um elo novo (`SEGURAR`), mas o contrato de campo
do dono é claro: o controlador manda `CARREGAR` com `v = 0`, e não um sexto one-hot. A
regra por cadeia é o que espelha isso.

### 6.6 A espera FINAL — depois do botar, "larga e fica parado"

**Decisão do dono (02/09):** no fim do `botar`, o comando esperado é a **ESPERA** —
`ANDAR` com velocidade zero. O robô larga a caixa e fica parado.

**É o espelho da espera inicial, com o mesmo mecanismo.** Quando o `BOTAR` fecha (caixa
apoiada, condição sustentada), o one-hot **publicado** vira `ANDAR` até o fim do
episódio. O elo **interno fica `BOTAR`**.

```
espera inicial   publicado ANDAR    twist 0   caixa 0      interno PEGAR
PEGAR            publicado PEGAR    twist 0   caixa viva
CARREGAR v=0     publicado CARREGAR twist 0   caixa viva   (segurar parado)
BOTAR            publicado BOTAR    twist 0   caixa viva, alvo = topo
espera final     publicado ANDAR    twist 0   caixa 0      interno BOTAR
```

O que o interno ficar `BOTAR` garante, e o que NÃO garante:

| a espera final precisa de | quem garante | lê |
|---|---|---|
| twist zero | `_zera_twist_nos_parados` — `BOTAR ∈ elos_parados` | interno |
| mobília no lugar, caixa na laje | o ramo `BOTAR` do `_aplica_elo` não guarda a laje | interno |
| canais de caixa zero | o gate, sobre o publicado = `ANDAR` | publicado |
| rastreio pagar por ficar parado | `rastreio_por_elo`, publicado = `ANDAR` | publicado |
| o estado "caixa apoiada no alvo" continuar pagando | `objetivo_ativo` = interno ≠ `ANDAR` ∧ ¬aguardando = **1** | interno |
| **o largar** — as palmas saírem da caixa | ⚠ **NADA, hoje.** Ver §6.6.1; o termo `largou` da §6.6.2 | interno |

⚠ A v12 atribuía o largar ao `PosturaPorElo` com o publicado em `ANDAR`. **Está errado,
e é medido no cfg do fabricante:** com twist zero o regime é `standing`, σ = 0,05 em
todas as 29 juntas. Com oito juntas de braço a 0,5 rad da default o termo vale
`exp(−27) ≈ 0`, com derivada zero. Ele não puxa nada. E `action_rate_l2` e `joint_acc`
pagam por **não** mover. Sem um incentivo próprio, o ótimo da espera final é congelar com
as mãos na caixa.

#### 6.6.1 O buraco da renda — por que o `BOTAR` de hoje não fecha

Revisão de 03/09, por leitura de `recompensas.py` e dos pesos de `knobs.py`. A condição de
fecho do `BOTAR` exige `apoiada` (a laje carrega ≥ 50% do peso). **Nenhuma recompensa paga
por `apoiada`.** Duas pagam pelo contrário: `unload = 1 − F_apoio/mg` e
`postura_ereta = rampa × unload`. Pousar a caixa zera as duas. A conta, por segundo, com
a caixa no alvo e as duas mãos nela:

| estado | HEAD (`exp/g1-limpo`) | spec v12 |
|---|---|---|
| pairar 1 cm acima do topo | 16,5 | 16,5 |
| apoio parcial, `F = 0,49·mg` (não fecha) | 16,5 | 16,5 |
| apoiada, mãos na caixa (fecha em 0,3 s) | 12,5 | 12,5 |
| depois do fecho | 12,5 | **5,0** |

Hoje o buraco custa 4/s: a política ótima paira. A espera final da v12, ao zerar os
incentivos depois do fecho, alargava a queda para 11,5/s pelo resto do episódio — e
tornava **não fechar** a melhor jogada por uma margem enorme. Como o `nível` só sobe com
sucesso de cadeia, a cadeia 3 ficaria presa em 5% do sorteio para sempre.

⚠ **O `g1_poc` caiu neste buraco e o consertou** (`g1_poc/recompensas.py:13,232,288,304`;
pesos em `g1_poc/knobs.py:249-252`): `load = clamp(F_apoio/mg) × perto`, só no `botar`,
peso 2,0; `squeeze` zero no `botar` ("apertar durante o botar paga contra soltar, −1,0/s
medido"); `unload` só no `pegar` ("ligado no botar, pagaria 2,0/s para NÃO botar"). E
mediu: "satisfazer a 3ª condição custava −3,0/s antes das máscaras". A reescrita do
`g1_limpo` perdeu as três peças. O `g1_poc` não tinha espera final, portanto não tinha o
problema do "depois"; a v13 tem de resolver os dois.

#### 6.6.2 O conserto — a renda sobe a cada passo até o fim

Princípio: **pairar < apoiar < fechar < largar**, cada estado rendendo mais que o
anterior. Cinco peças; quatro têm precedente no `g1_poc`. Todas leem o elo **interno**.

1. **`objetivo_ativo` fica como hoje** (§6.0): `interno ≠ ANDAR ∧ ¬aguardando`. A espera
   final **não** o zera. Os incentivos do estado "caixa apoiada no alvo" continuam
   pagando depois do fecho, e o rastreio entra por cima (publicado `ANDAR`, twist zero).
2. **Máscaras no `BOTAR`:** `squeeze` → 0 e `unload` → 0 quando o interno é `BOTAR`.
   `postura_ereta` é `rampa × unload` e zera junto, sem linha própria. Os três pagam por
   **segurar**; no `BOTAR` eles pagam contra a tarefa. É a máscara do `g1_poc`.
3. **`alcança ≡ 1` no `BOTAR`**, dentro de `staged` e de `precise_ori`. No `BOTAR` as
   mãos já estão na caixa: o σ do kernel cai no piso de 0,08 m e ele vale 1 por
   construção. Ele não carrega informação ali — ele só paga 3/s por **manter** as mãos
   na caixa, que é o freio contra largar. Com `alcança ≡ 1`, `staged` vira
   `3 × (1 + trazer)`: paga pela caixa ir ao alvo, indiferente às mãos. Fora do `BOTAR`
   nada muda. A forma geral da regra é `alcança ≡ 1 se (interno == BOTAR) ou soltou`
   (§8.3); na v2 as duas condições coincidem, porque só o `BOTAR` tem espera final.
4. **Termo novo `load`** = `clamp(F_apoio/mg) × perto(d ≤ 2·tol_pos)`, só com o interno
   em `BOTAR`, peso **2,0**. O espelho do `unload`, com o mesmo peso e o mesmo gate de
   posição do `g1_poc` (`load_raio_mult = 2`). O gate fecha o hack de largar a caixa em
   qualquer lugar do tampo. Sem gate de preensão: soltar **é** o objetivo. Como o interno
   segue `BOTAR` na espera final, ele continua pagando depois do fecho.
5. **Termo novo `largou`** = `soltou × load × (1 − exp(−(d_palma/σ_solta)²))`, com
   `σ_solta = 0,10 m` (knob novo), peso **1,0**. Paga por afastar as palmas com a caixa
   apoiada no alvo: 10 cm rende 0,63, 20 cm rende 0,98. Só existe na espera final. Como a
   peça 3 tirou o freio, um peso pequeno basta.

A conta depois do conserto, com os mesmos estados da §6.6.1 e mais dois:

| estado | renda /s | quem paga |
|---|---|---|
| `BOTAR`, pairar 1 cm acima, mãos na caixa | 11,5 | staged 6 · precise_pos 2 · precise_ori 1 · sustentacao 0,5 · pose 1 · upright 1 |
| `BOTAR`, apoio parcial `F = 0,49·mg` | 12,5 | + load 0,98 |
| `BOTAR`, apoiada, mãos na caixa → **fecha em 0,3 s** | 13,5 | + load 2 |
| espera final, mãos na caixa | 16,5 | + track 4 − pose 1 (σ 0,05, braços fora) |
| espera final, palmas longe, braços na default | **18,5** | + largou 1 + pose 1 |

Monótona. Cada passo em direção ao que se quer rende mais. E o mesmo `precise_pos`,
`staged` e `sustentacao` que pagam por chegar ao alvo continuam pagando por **ficar**
nele depois de largar.

**Hacks conferidos, nenhum abre:** prensar a caixa contra a laje satura `load` em 1
(`clamp`); apoiar fora do alvo perde `perto`; largar e depois empurrar a caixa perde
`precise_pos`, `load` e `largou` juntos; andar durante a espera perde rastreio; escorar
na mesa paga `contato_*`; derrubar termina (`caiu`, §6.6.3).

**Pesos e σ são ponto de partida**, não medição: 2,0 e 1,0 copiam a escala dos termos
vizinhos (`unload` = 2,0; `precise_ori` = 1,0). O smoke prova a **monotonia** da tabela
com o robô travado em cada um dos cinco estados (§11.1, item 16); a GPU prova se a política
a segue.

#### 6.6.3 O que continua igual, e a terminação

**A terminação `caixa_largada` muda em duas linhas.** Ela é `(caiu | escapou) & pegou`, e
`escapou` é "as duas palmas longe da caixa". Na espera final as mãos **têm** de sair da
caixa — `escapou` dispararia no primeiro passo e mataria o episódio por fazer a coisa
certa. Portanto:

```
caixa_largada = (caiu | (escapou & ~soltou)) & pegou
```

onde `soltou` é publicado pelo comando (`env.limpo_soltou`) quando a espera final começa.
**`caiu` continua armado:** largar é permitido, **derrubar não** — é o "não jogar" do
dono, estendido ao depois. É a primeira das duas linhas; a segunda é o `caiu` ler o
tamanho da caixa (§6.7).

**Sem cronômetro.** A espera final dura até o fim do episódio. Não há o que sortear: o
robô larga e fica de pé, e o tempo restante é prática de "parado, sem tarefa, com uma
mesa perto" — que é exatamente o estado de campo depois de botar.

**`sucesso` e `fechou` continuam marcados no fecho do `BOTAR`**, não no fim da espera. A
tarefa é botar; a espera final é o que vem depois dela.

⚠ **Por que não um 4º elo `ANDAR` na cadeia 3** (a v6 deste documento o anotava como
pendente): ele exigiria quatro regras especiais num `ANDAR` que não é `ANDAR` — não
guardar a mobília (o ramo de hoje manda a laje a +5 m em z, através das mãos), twist zero
(`ANDAR ∈ ELOS_QUE_ANDAM`), desarmar o `_pegou`, e fecho por tempo. A espera final é uma
regra e um guarda, e é o mesmo mecanismo da espera inicial.

**Consequência:** `BOTAR → ANDAR(v=0)` sai da aposta e vira treinada. **Nenhuma transição
do cenário fica como aposta** (§7.2).

### 6.7 DR de TAMANHO da caixa — desde a FASE 1

**Decisão do dono (02/09):** variar aleatoriamente o **tamanho** da caixa, não só o peso.
Independente do peso. Para todos os envs, **desde o começo** do treino de manipulação — e
não na FASE 2.

#### Por que tamanho não é um knob como massa

A massa hoje **não muda no modelo**: `carga_caixa` aplica uma **força externa** vertical
(`write_external_wrench_to_sim`) que finge a carga. O docstring diz por quê: "NUNCA
`dr.body_mass` nem `dr.pseudo_inertia`: os dois corrompem a heap (CUDA illegal memory
access). Está MEDIDO no repositório." Mutar o modelo em runtime já quebrou aqui.

Tamanho é geometria. Não há força externa que finja uma caixa maior. O `geom_size` **é**
por mundo no `mujoco_warp` (`array("*", "ngeom", vec3)`), mas escrever nele em runtime é a
mesma classe de operação que corrompeu a heap. O caminho suportado é outro:

#### O mecanismo: `geom_size` por mundo, escrito UMA vez no startup — sem mesh

**Pergunta do dono (02/09): trocar a caixa de `box` para mesh não quebra o treino?**
Não quebra a mecânica — recompensas, sensores e cadeias são por par de geom e por corpo,
e nada disso lê o tipo do geom. Mas **troca a física de contato**, e isso foi verificado
na tabela de colisão do `mujoco_warp` (`collision_driver.py`):

```
(BOX, BOX)   PRIMITIVE   — colisor analítico box_box; é o de hoje: pad-caixa, laje-caixa
(BOX, MESH)  CONVEX      — GJK/EPA; com a caixa em mesh, TODO contato dela vai por aqui
```

Menos pontos de contato por par, normais e penetrações de outra natureza, caixa apoiada
na laje com outra estabilidade, distribuição do `squeeze` diferente, e mais lento. A pega
foi aprendida contra `box_box`. O efeito no aprendizado é **desconhecido**, e só uma run
em GPU mede — o smoke só mede estática.

**E o mesh não é necessário.** Dois fatos do `mujoco_warp` e um do próprio módulo:

1. **`geom_size` é lido por mundo em TODO colisor**, primitivo ou convexo —
   `collision_core.py:101`: `geom1.size = geom_size[worldid % geom_size.shape[0], g1]`.
   O `% shape[0]` é a convenção dos campos `*`: forma `(1, ngeom)` transmite, forma
   `(nworld, ngeom)` é por mundo. O `box_box` analítico lê por esse caminho. É
   **exatamente** o que o mecanismo de variantes explora — ele só acrescenta o mesh por
   cima.
2. **`geom_rbound` e `geom_aabb` são os limites do broadphase**, e têm de acompanhar: uma
   caixa de 0,13 m com o `rbound` de 0,10 m perderia contato nos cantos. Os três campos
   são os primeiros de `VARIANT_DEPENDENT_FIELDS` do `variants.py` — é a lista do que
   depende de geometria.
3. **O `g1_limpo` já faz uma escrita dessa classe, e ela treinou a `bloco7`.** O evento
   `foot_friction` é `mode="startup"` sobre `geom_friction` — `array("*", ngeom, vec3)`,
   mesma classe de campo que `geom_size` —, por mundo, uma vez, antes do primeiro passo.
   ⚠ A corrupção de heap registrada no `carga_caixa` era **outra coisa**: `body_mass` e
   `pseudo_inertia` em **runtime**, campos de massa com derivados (`body_subtreemass`,
   `body_invweight0`) que ficam inconsistentes. Startup, geometria, campos
   auto-contidos do broadphase: é a classe do `foot_friction`, não a do `body_mass`.

**O desenho:** um evento `mode="startup"` (`tamanho_caixa`) que, para o geom da caixa,
escreve por mundo `geom_size`, `geom_rbound` e `geom_aabb` a partir do meio-lado sorteado
— e publica `env.limpo_meia_aresta`. A caixa **continua `box` primitivo**. O colisor
**não muda**. O `paridade` **não ganha divergência**.

⚠ **Três consequências, declaradas:**

1. **Fixo na run, como o `foot_friction`.** Startup, não reset. "Aleatória para todos os
   envs" vale **entre envs**: cada env vê sempre a mesma caixa durante a run. Com
   4096–8192 envs a frota cobre a distribuição. A função do mjlab aceitaria `mode="reset"`
   (é DR de primeira classe, não a classe do `body_mass`), mas o startup é o caminho já
   provado pelo `foot_friction` e basta; re-sortear no reset fica como extensão.
2. **`body_inertia` fica na de 0,10 m.** Escrever inércia é a classe de `body_mass`, e
   não se toca. Uma caixa de 0,13 m com a inércia da de 0,10 m é inconsistência **do
   mesmo tipo e do mesmo tamanho** da que o módulo já aceita: "a caixa de 5 kg fica com a
   INÉRCIA de 1 kg" (`carga_caixa`). A DR endurece a estática, não a dinâmica. Declarado.
3. **Massa não muda, por construção:** `body_mass` não é escrito. A independência do peso
   vem de graça — o wrench do `carga_caixa` segue por cima.

⚠ **A incerteza da v11 caiu, verificada em 03/09 no fonte do mjlab 1.5.1:** a escrita
existe no próprio framework. `mjlab.envs.mdp.dr.geom_size` (`envs/mdp/dr/geom.py`)
escreve `geom_size` por mundo e **recalcula `geom_rbound` e `geom_aabb`** para os
primitivos, box incluído. Os três campos são declarados por `@requires_model_fields`, e
`load_managers` os expande para `(nworld, ngeom)` **antes** dos eventos de startup e do
primeiro `forward`; o kernel de broadphase do `mujoco_warp` é cacheado pela forma por
mundo (`cache_kernel`), portanto nasce já indexando por mundo. É o mesmo caminho de DR do
`foot_friction`, e não uma escrita inédita deste módulo.

O que a função do mjlab **não** dá: cubo. Ela sorteia cada eixo de forma independente, e
`shared_random` é entre geoms, não entre eixos. Portanto o evento `tamanho_caixa` é um
**wrapper fino** (~15 linhas): sorteia um dos K valores por env, escreve `(a, a, a)` em
`geom_size` do geom da caixa, chama o `_recompute_geom_bounds` do próprio mjlab (ou repete
as duas fórmulas do box: `rbound = a·√3`, `aabb_half = (a, a, a)`), e publica
`env.limpo_meia_aresta`. Decorado com `requires_model_fields("geom_size", "geom_rbound",
"geom_aabb")`. ⚠ Confirmar que o 1.5.3 do Kaggle tem a mesma função (é mais novo; deve
ter). O smoke prova em CPU que o colisor lê o tamanho novo (§11.1 item 13). Se a GPU
reclamar — o que agora seria defeito do mjlab, não deste módulo —, o plano B está pronto:

#### Plano B: `VariantEntityCfg` com mesh

O `mjlab` tem `entity/variants.py`: K variantes de `MjSpec`, compiladas na inicialização,
com `geom_size`, `geom_rbound`, `geom_aabb`, `body_mass`, `body_inertia` espalhados por
mundo. **"Only mesh geoms can differ"** — a caixa teria de virar mesh (8 vértices por
tamanho), e todo contato dela passaria a `CONVEX`. Atribuição fixa na inicialização;
`mass=` explícito e igual em todas garante a independência do peso; a inércia varia com o
tamanho (aqui, correto). Só se o plano A falhar na GPU.

#### O desenho

```
caixa_meia_aresta_faixa  = (0,07, 0,13)   m      ±30% em torno dos 0,10 de hoje   (decisão do dono, 02/09)
caixa_n_variantes        = 8                      passo de ~0,86 cm                 (decisão do dono, 02/09)
escala                   = UNIFORME nos 3 eixos   a caixa segue cubo                (decisão)
atribuição               = aleatória por env, com semente, fixa na run
massa                    = igual em todas as variantes; a carga segue pelo wrench
```

**Escala uniforme (cubo), e não por eixo.** Por eixo daria 3 graus de liberdade e
mudaria a forma da pega (uma caixa achatada pede outra abertura de mãos). O pedido foi
"tamanho"; cubo é a leitura direta. Por eixo fica registrado como extensão.

**K = 8 tamanhos discretos, e não contínuo**, mesmo sem a restrição das variantes: com
tamanhos discretos o smoke afirma "estes 8 e só estes" e a leitura do painel por tamanho
fica possível. Contínuo é uma linha a mais, se um dia valer.

**Fonte de verdade por env: o modelo.** O evento de startup publica `env.limpo_meia_aresta`
(n, 3) a partir do que escreveu em `geom_size`. Todo consumidor lê dali. Hoje **os sítios abaixo** leem um escalar de `knobs`, e cada um passa a ler
o tensor por env:

| onde | o que lê hoje | passa a ler |
|---|---|---|
| `comando.alvos_das_palmas` | `cfg.caixa_meia_aresta` — o offset lateral das palmas | `limpo_meia_aresta[ids, 1]` |
| `comando` ramo `BOTAR` (2 sítios: `fundo`, `a[:, 2]`) | `cfg.caixa_meia_z` — fundo da caixa e z do alvo | `limpo_meia_aresta[ids, 2]` |
| `comando._laje_para` com `sobe_caixa` (ramo `ANDAR`) | `cfg.caixa_meia_z` — a caixa em cima da laje a +5 m | `limpo_meia_aresta[ids, 2]` |
| `eventos.posiciona_cena` | `caixa_meia_z` — z de repouso na laje | `limpo_meia_aresta[ids, 2]` |
| `eventos.afasta_cena` | `caixa_meia_z` | idem |
| `terminacoes.caixa_largada` (`caiu`) | `caixa_z_min = 0,10` fixo | `limpo_meia_aresta[:, 2] + folga` — ver abaixo |
| `smoke.py` (4 linhas: 106, 308, 1844, 2156) | `k.cena.caixa_meia_aresta` | a variante de referência, ou por env onde afirma o modelo |
| `inspeciona`, `paridade` | `k.cena.caixa_meia_aresta` | a variante de referência (0,10) |

⚠ **A lista acima é inventário por `grep`, não por memória** — a v12 dizia "oito sítios"
e eram mais. O plano roda `grep -n 'caixa_meia\|meia_aresta\|meia_z'` e fecha cada linha.

#### `caiu` passa a ler o tamanho (revisão de 03/09)

`caixa_z_min = 0,10` é a meia-aresta de hoje, e o smoke afirma essa igualdade: "com o
centro nessa altura a caixa está apoiada no chão". Com o tamanho variando, o limiar fixo
quebra dos dois lados: uma caixa de 0,13 m deitada no chão tem o centro em 0,13 e
**`caiu` nunca dispara** — e na espera final `escapou` está desarmado, portanto derrubar a
caixa grande não termina nada; uma caixa de 0,07 m na laje mais baixa (0,04 m) tem o
centro em 0,11, a 1 cm da terminação.

Conserto: `caiu = (z_centro − origem_z) − meia_aresta_env < caixa_folga_chao`, com
`caixa_folga_chao = 0,02 m` (knob novo; substitui `caixa_z_min`). Lê-se "o fundo da
caixa está a menos de 2 cm do chão". Caixa de 0,13 m no chão: fundo em 0, dispara. Caixa
de 0,07 m na laje a 0,04 m: fundo em 0,04, não dispara. O smoke troca a igualdade por
`caixa_folga_chao < prateleira_topo_piso`. ⚠ Limitação declarada, igual à de hoje: uma
caixa que cai e fica **tombada** sobre uma aresta tem o centro em `a·√2` e escapa ao
`caiu`; fora da espera final o `escapou` a pega, dentro dela não. Caixas param deitadas;
fica registrado.

**`paridade.py` não muda:** a caixa segue `box` primitivo, e o spec de referência (0,10 m)
é o mesmo de antes. O tamanho por env é do modelo batched, não do spec.

#### A pergunta que a DR de tamanho abre: a rede VÊ o tamanho?

Hoje `caixa_b` é o **centro** da caixa. O tamanho não está na observação. Com K tamanhos
de 14 a 26 cm, o alvo das palmas (§4.1, `alvos_das_palmas`) muda **±3 cm por lado** entre
envs — e a recompensa sabe o tamanho certo de cada env, mas a política não.

| | sem observar | observando (+1 canal) |
|---|---|---|
| o que a política aprende | uma abertura de mãos **média**, e corrige por contato | a abertura **certa** para cada caixa |
| contato como sinal | só via `joint_pos`/`joint_vel` — não há força de palma na observação | idem, mas não precisa |
| em campo | a percepção **tem** o tamanho (é um bounding box) e ele seria jogado fora | o que o campo tem, a rede recebe — é o princípio da §4 |
| custo | zero | +1 canal, gateado como os outros 9 |
| risco | pega pior em todos os tamanhos, pela média | nenhum específico |

**Decisão do dono (02/09): observar.** Um canal `meia_aresta` (o meio-lado, em metros)
no fim do termo `caixa`, gateado a zero com o publicado em `ANDAR` como os outros nove. Em
campo ele vem da percepção, como `caixa_b`. A política é sem memória (§7.2): ela não tem
como "descobrir" o tamanho ao longo do episódio — ou ela o vê, ou ela chuta. O termo
`caixa` fica com 10 canais e a observação com 114 (§4).

#### O que NÃO muda

Recompensa nenhuma **por causa do tamanho**. `unload` e `squeeze` derivam de
`limpo_massa`, que segue igual. `staged` usa `dist_palma_caixa`, que já lê
`alvos_das_palmas` — e este passa a ler o tamanho por env. `caixa_dist_max = 0,45` é
absoluto e cobre todos os tamanhos; só o `caiu` muda, acima. O currículo não muda.

⚠ **O que muda de fato é a dificuldade da pega**, e é pedido: a política deixa de poder
memorizar "20 cm". É a mudança desta spec com mais efeito sobre a manipulação, e é a
razão de o `descarga` ser sentinela na primeira run da v2 (§12).

---

## 7. O QUE É TREINADO, E O QUE É APOSTA

### 7.1 Treinado

| transição | onde | o que a política vê no pulo |
|---|---|---|
| `ANDAR(v=0) → PEGAR` | **todo** episódio de manipulação com espera > 0 (§6.3) | one-hot vira; canais de caixa acendem; twist segue zero |
| `PEGAR → CARREGAR (v>0)` | cadeia 2 | one-hot vira; twist acende; caixa segue viva |
| `PEGAR → CARREGAR (v=0)` | cadeia 3 (§6.5) | one-hot vira; twist segue zero; caixa segue viva — "segurar parado" |
| `CARREGAR (v=0) → BOTAR` | cadeia 3 (§6.5) | one-hot vira; `alvo_b` muda para o topo novo |
| `BOTAR → ANDAR (v=0)` | espera final, cadeia 3 (§6.6) | one-hot vira; canais de caixa apagam; `largou` paga por as mãos saírem da caixa, e sair não termina o episódio |

⚠ A primeira era a **aposta** da v1 deste documento. Deixa de ser: a espera a treina a
partir da postura real de parado, e o resíduo "configuração de juntas no instante da
troca" deixa de existir. (Decisão do dono, 02/09: em `v = 0` a pose é praticamente a
default — e agora isso nem precisa ser assumido.)

### 7.2 Aposta — não sobrou nenhuma

Na v5 deste documento duas transições eram aposta; na v6, uma. **Na v7, nenhuma:**

- `CARREGAR → BOTAR` — treinada pela cadeia 3 (§6.5)
- `BOTAR → ANDAR(v=0)` — treinada pela espera final (§6.6)

Toda troca do cenário de campo (§3, §4.4) acontece em algum episódio do treino. O que a
política vê em campo, ela viu no sim.

**O que sustenta isso é a política ser SEM MEMÓRIA.** Medido: `history_length = None` nos
dois grupos, e nenhum termo com histórico. O ator é um MLP. O que a política vê no
instante da troca é o corpo **agora**, e não a história — portanto "o mesmo estado" no
sim e em campo é de fato o mesmo estado. Se houvesse recorrência, o buffer carregaria a
tarefa anterior e nenhum gate consertaria.

**O que continua sem garantia, e não é transição:** a *qualidade* de cada comportamento
(a pega ainda não fecha o `sustentacao`, §2), e o sim-to-real dos contatos (§10.1).

### 7.3 Por que "andar até a mesa" fica fora

O vazamento da §5 é **informacional** e o gate o conserta. Existe um segundo, que é
**físico**: no treino, todo env de `ANDAR` que anda tem a mobília a +5 m, portanto **o
robô nunca andou com uma mesa à frente**. Gatear a observação não muda isso — em campo
ele andaria para dentro da mesa.

Consertar exige mobília presente durante o andar **em movimento**, alvo de aproximação e
freio. É trabalho de bloco, não de ajuste.

**Decisão do dono (02/09): não é objetivo deste modelo.** Quem leva o robô até a mesa é
o piloto — ou outro modelo, ou outra abordagem. Este modelo não precisa saber que a mesa
está lá. A espera da §6.3 começa **depois** disso: robô já parado na mesa.

---

## 8. LACUNAS CONHECIDAS

Registradas com custo. Nenhuma é resolvida por este trabalho.

### 8.1 `PEGAR → CARREGAR → BOTAR` — RESOLVIDA pela §6.5

Era lacuna na v5: `_TETO_ELOS = 2`, e a perna `CARREGAR → BOTAR` era aposta. Decisão do
dono (02/09): a cadeia 3 vira `(PEGAR, CARREGAR, BOTAR)`. Fica aqui só o registro de que
a lacuna existiu e de onde foi fechada.

### 8.2 `BOTAR → ANDAR` — RESOLVIDA pela §6.6

Era aposta até a v6. A espera final publica `ANDAR` depois do fecho do `BOTAR`, com o
guarda em `caixa_largada`. Registro de onde a lacuna foi fechada.

### 8.3 O `REORIENTAR` — habilidade FUTURA: a REDE fica pronta, o TREINO não entra

**Decisão do dono (02/09, refinada em 03/09): o `REORIENTAR` é habilidade futura.** É
tarefa difícil e pode custar o resto do treino; a run da v2 existe para provar que uma
política por one-hot funciona, e não para aprender a girar caixa. Portanto:

- **o CONTRATO entra agora**: o canal `giro_b` (§4.1) e o slot do one-hot. É o que faz
  a rede ficar pronta — o que falta depois é recompensa, fecho e knob, e nada disso muda
  a observação; a run da v2 pode ser retomada por warm-start no dia em que a
  reorientação virar foco;
- **o elo, a cadeia 1 e o fechamento FICAM** no código, como estão;
- **ele CONTINUA sorteável** — ver o porquê abaixo, que é contra-intuitivo;
- ⚠ **ele fica INERTE na run da v2**: `voltas_max = 0` em todo nível (e `eixo_vertical`
  falso), para a caixa nascer sempre dentro da tolerância e o elo fechar em 0,3 s sem
  trabalho, em todo nível — como já acontece hoje nos níveis 0 e 1. É o defeito abaixo
  usado de propósito como interruptor. Para o cubo isso não muda nada em `PEGAR`,
  `CARREGAR` ou `BOTAR`: um quarto de volta é simetria do cubo, e o `precise_ori` desses
  elos mede a torção desde a abertura, não a orientação de nascimento. O custo é 0,3 s de
  `REORIENTAR` antes do `PEGAR` em metade dos episódios de manipulação, como hoje. **A
  confirmar pelo dono (§13).**
- o defeito abaixo fica registrado e **não é consertado agora**.

⚠⚠ **NÃO tirar o `REORIENTAR` de `ELOS_SORTEAVEIS`**, embora fosse a saída óbvia para
"guardar para depois". Motivo medido, em `rsl_rl/modules/normalization.py:48`:

```
saída = (x − _mean) / (_std + 1e−2)        sem clamp
```

Com um canal constante em zero, `_std → 0` e `_mean → 0`. Quando o canal acende, o
primeiro `1,0` entra na rede como **100,0**.

É exatamente por isso que `fatia_loco = 0,95` e **nunca 1,00**: 5% dos episódios são de
manipulação desde o passo zero, só para os slots sorteáveis nunca ficarem constantes.
Tirar o `REORIENTAR` do sorteio recriaria o perigo que o desenho evita de propósito, e
o dia de treinar reorientação começaria com um choque de normalizador.

**Mantê-lo sorteável é o que o mantém pronto.**

#### O defeito, quantificado

```
tol_ang_deg        = 25,0°      tolerância de fechamento
desalinho_max_deg  = 15–20°     desalinho de nascimento
voltas_max         = (0, 0, 1, ...)   níveis 0 e 1 nascem sem quarto de volta
```

Nos níveis 0 e 1 a caixa nasce **dentro** da tolerância de fechamento. Somado ao `perto`
ser trivial no `REORIENTAR` (o alvo *é* a caixa), as duas condições valem no spawn e o
elo fecha depois de `sustenta_outros_s = 0,3 s` sem o robô fazer nada.

**Evidência:** `avancos = 0,1163` com `sucesso = 0,0069`, e `Curriculum/nivel = 0,945`
— exatamente a faixa onde isso morde.

**A gravidade, medida no que o defeito NÃO corrompe:** o `sucesso` só é marcado quando o
**último** elo da cadeia fecha, portanto o avanço grátis não infla o `sucesso` e não
engana o passeio de nível. O dano real: a cadeia 1 degenera em "só pegar" (a
reorientação nunca é exercitada de verdade) e há ~1,0/s de `precise_ori` pago por 0,3 s
sem trabalho. É desperdício de sorteio, não corrupção de métrica.

#### O contrato — o que a rede já recebe (ENTRA na v2)

**Visão do dono (03/09): o robô precisa saber quatro coisas, e só quatro** — girar a
caixa 90° para a esquerda ou para a direita (em torno de Z, pivotando na laje), e 90°
para cima ou para baixo (tombar, em torno de Y). Qualquer outra reorientação é repetição
dessas: a face de trás são duas à esquerda; o fundo é um tombo. Quem compõe é o
controlador externo, que lê a câmera, sabe a orientação da caixa e as faces já vistas,
escolhe a próxima face, e manda **um** giro por vez. A memória de "quais faces já vi"
mora nele, não na rede (que é sem memória, §7.2).

**O comando É o giro.** `giro_b` (§4.1): eixo × ângulo, em radianos, da rotação que leva a
normal atual da face pedida à direção pedida, no frame da base. As quatro primitivas são
os quatro vetores `(0, 0, ±π/2)` e `(0, ±π/2, 0)`, e o comprimento **encolhe** conforme o
robô gira. ⚠ Por isso não é um one-hot de quatro: a política precisa do resíduo para
saber quando parar e para desacelerar perto do fim — no tombo, para saber que a caixa
passou do ponto de equilíbrio. Com um one-hot o controlador teria de desligar o comando
e a política passaria do ponto. `|giro_b|` é o `ANG` de hoje, portanto `precise_ori` e
`alinhado` não mudam de fórmula. Até a v13 a observação tinha só o escalar: dizia quanto
faltava e não dizia para que lado — um MLP sem memória não tinha como aprender a girar.

**A direção pedida é HORIZONTAL, da caixa para o robô** — a `viva` de hoje. ⚠ Isto é
física, não preferência: uma caixa apoiada só tem normais horizontais (laterais) ou
verticais (topo, fundo). "Normal à câmera" exigiria a caixa inclinada na mão, que é outra
habilidade. A câmera do G1 vê a face da frente de cima, em ângulo, e vê o topo direto;
quem conhece a geometria da câmera é o controlador, e ele pede um passo atrás ao piloto
se o ângulo estiver ruim. Consequência boa: com a direção sempre "para a frente", os
giros são só em **Y e Z**; giro em X nunca aparece, e as seis faces são alcançáveis.

**A face pedida não entra na observação.** Um cubo é simétrico: girar 90° em Z é o
mesmo movimento seja qual for a face marcada. Em sim a face marcada segue constante
(`face_alvo_b = −X`) e a variedade vem da orientação de nascimento; em campo o
controlador escolhe a face e calcula o giro. A rotação em torno da própria normal é
livre — um QR code lê em qualquer rotação no plano — e o vetor de giro a ignora, que é
o certo.

**Nos outros elos os mesmos três canais servem.** Em `PEGAR`, `CARREGAR` e `BOTAR` a
face é congelada na abertura do elo e `giro_b` diz "quanto e em torno de que eixo a
caixa torceu desde então" — o `precise_ori` já paga por erguer sem torcer, e agora a
política vê a direção da torção.

#### O desenho do treino — PRONTO, mas NÃO entra na v2

**Decisão do dono (03/09): o `REORIENTAR` termina como o `BOTAR`** — o robô executa,
larga a caixa na mesa e tira as mãos. "Bota as mãos e tira as mãos." Portanto o desenho
é o espelho da §6.6.2, peça a peça:

1. **Fecho:** `alinhado & apoiada`, sustentado `sustenta_outros_s`. O `apoiada` entra
   porque, depois de tombar, a caixa tem de estar de novo na laje, e não no ar.
2. **Espera final** depois do fecho, igual à do `BOTAR`: publicado `ANDAR`, interno
   `REORIENTAR`, `soltou`, `escapou` desarmado, `caiu` armado, `largou` pagando por tirar
   as mãos. A cadeia 1 vira `(REORIENTAR,)` — o `PEGAR` depois dela é composição do
   controlador (`ANDAR(v=0) → PEGAR` já é treinado, §6.3), e não um segundo elo.
3. **Máscaras:** `squeeze` e `unload` zero no `REORIENTAR` (pagam por erguer; girar não
   é erguer). `load` ligado no `REORIENTAR` também (a caixa tem de voltar à laje).
4. **`alcança`:** vivo durante o elo (as mãos têm de chegar à caixa — é o gradiente de
   aproximação, e é a porta que o `precise_ori` já usa), e `≡ 1` na espera final
   (`soltou`), para não pagar por manter as mãos na caixa depois de pronto. A regra da
   §6.6.2 item 3 generaliza para `alcança ≡ 1 se (interno == BOTAR) ou soltou`.
5. **`precise_pos` no `REORIENTAR` é constante** (o alvo é a própria caixa): 2,0/s de
   renda fixa. Inofensivo, e declarado.
6. **A conta de monotonia** (girar sem apoiar < apoiada alinhada < espera final <
   largar) tem de ser refeita e afirmada pelo smoke como o item 16 da §11.1, antes de
   ligar.

**O `REORIENTAR` NÃO herda o eixo de altura do `PEGAR`** (decisão do dono, 03/09). Ele
não é tarefa de alcance. No real, se a caixa precisa ser reorientada, o robô
a larga numa mesa (`BOTAR`), gira, e a pega de novo (`PEGAR`). A sequência de campo é
`BOTAR → ANDAR(v=0) → REORIENTAR → ANDAR(v=0) → PEGAR`, e cada seta já é uma transição
treinada (§7.1): o `REORIENTAR` sempre começa da espera inicial, com a caixa apoiada e as
mãos fora. Portanto a cena dele é a de **uma mesa**, com a altura variando **de 0,45 a 0,55 m**
(decisão do dono, 03/09) só para o robô não decorar — e não a escada de 0,55 a 0,04 m
do `PEGAR`. Hoje o tombo (`eixo_vertical`) só nasce no nível 4, junto com a laje a
0,04 m; tombar uma caixa no chão não é a mesma tarefa que tombar uma na mesa. **A
variação de carga fica** (decisão do dono, 03/09): ela pesa na capacidade de girar e de
tombar, e é dificuldade legítima do `REORIENTAR`. Quando virar foco, um bloco
`knobs.Reorientar` próprio, lido por `posiciona_cena` quando o elo do env é
`REORIENTAR`:

```
topo          (0,45, 0,55) m       uniforme, independente do nível — dentro do envelope do PEGAR e do BOTAR
carga         a de hoje            `carga_caixa` pelo nível, 1 kg até `carga_max[nível]`
jitter_x      o de hoje            pelo nível
voltas        SEMPRE um quarto de volta: eixo ∈ {Y, Z}, sinal ±, probabilidade igual
desalinho     ±20°
```

⚠ A faixa de altura **não copia a do `BOTAR`**, porque o `BOTAR` de hoje varia o topo de
0,30 a 0,80 m (`botar_topo_piso`, `botar_topo_teto`), e não 10 cm; ele é o único elo que
já tem a laje separada do nível.

Duas consequências: com um quarto de volta em **todo** episódio o erro de nascimento é
90° ± 20°, nunca dentro dos 25° de tolerância — o avanço grátis morre por construção; e
o `nível` **continua** a mover com o sucesso da cadeia 1, porque a carga do `REORIENTAR`
segue o nível — só a altura saiu dele.

⚠ **Registrado, fora deste documento:** a laje do sim nunca passa de 0,55 m
(`prateleira_topo_teto`), e uma mesa real tem 0,70 a 0,80 m. A história de campo "larga
na mesa e pega de novo" acontece numa altura em que o `PEGAR` nunca treinou. É envelope
do `PEGAR`, não do `REORIENTAR`, e é assunto de knob da FASE 2 (§10.1).

**E o defeito de hoje se conserta** por dois caminhos, o primeiro de graça:

1. **Sempre um quarto de volta** (a tabela acima). O conserto antigo,
   `desalinho_max_deg > tol_ang_deg` em todo nível, deixa de ser necessário: a caixa
   nunca nasce dentro da tolerância.
2. **Exigir erro inicial mínimo** no fechamento (o elo só fecha se houve trabalho), como
   cinto de segurança. Custo: lógica nova no `_fecha_elo_corrente`.

**Por que a rede fica pronta com o que entra na v2:** tudo acima é recompensa, fecho,
cadeia e knob. Nada muda a observação. O checkpoint da v2 serve de warm-start.

---

## 9. ITEM COMPANHEIRO — girar no lugar (locomoção)

**Pedido do dono (02/09):** melhorar o aprendizado de girar no lugar. Suspeita: "lin = 0
com ang > 0" é sorteado por muito pouco tempo.

**A suspeita está certa, e é estrutural — não é pouco tempo, é quase nunca.** O
sorteador do fabricante, medido no cfg:

```
rel_standing_envs = 0,10     tudo zero                        -> não gira
rel_forward_envs  = 0,20     só frente: vx ≥ 0,3, vy = wz = 0  -> não PODE girar
rel_heading_envs  = 0,30     ang_z vem do erro de rumo, ANDANDO -> gira andando
restante          = 0,40     lin_x, lin_y, ang_z INDEPENDENTES e uniformes em ±1,0 / ±1,0 / ±0,5
```

Nos 40% uniformes, "lin ≈ 0 **e** ang ≠ 0" exige dois contínuos independentes caírem
perto de zero ao mesmo tempo. Nos outros 60% é impossível por construção: `standing` não
gira, `forward` não pode girar, `heading` gira **andando**. Girar no lugar não é um caso
do sorteio.

**Evidência no painel:** `Metrics/twist/error_vel_yaw = 1,26` contra comando de ±0,5
rad/s — o erro de guinada é maior que a faixa do comando. (Definição exata da métrica a
confirmar antes de citar como medida absoluta; a ordem de grandeza sustenta a suspeita.)

**O conserto é um ramo no sorteador, e não toca recompensa nenhuma.** O dono pôs duas
formas na mesa: (a) incluir "só velocidade angular" na lista de comandos sorteados, ou
(b) definir envs só para o giro. **Decisão: (b), `rel_turning_envs`**, análogo ao
`rel_standing_envs` — e as duas formas são o mesmo mecanismo no idioma do fabricante: as
flags `is_standing_env`, `is_forward_env` e `is_heading_env` são **re-sorteadas a cada
re-sorteio de comando** (3 a 8 s), não fixas por env. Um `is_turning_env` entra na mesma
lista: `lin = 0`, `ang_z` sorteado na faixa. O `track_angular_velocity` já paga por
rastreá-lo. O termo `TwistComRazaoDeMarcha` já é subclasse do fabricante; o ramo entra lá.

⚠ **Com `|ang_z|` mínimo**, como o ramo `forward` faz com `vx ≥ 0,3`: um giro sorteado
perto de zero seria um `standing` disfarçado. Ponto de partida: `|wz| ≥ 0,2 rad/s`.

⚠ **As flags do fabricante NÃO são uma partição, e isso muda a implementação** (lido em
`velocity_command.py`, 03/09). `is_standing_env`, `is_heading_env` e `is_forward_env` são
sorteios de Bernoulli **independentes**, e o que decide é a precedência no código:
`standing` zera o comando **todo passo**; `heading` reescreve `wz` **todo passo**;
`forward` escreve `vx ≥ 0,3, vy = wz = 0` só no resample (e um env `forward ∧ heading`
tem o `wz` reescrito pelo heading no passo seguinte). O "restante 0,40 uniforme" da
tabela acima é aproximação; as frações realizadas são outras. Portanto o `is_turning_env`
entra com precedência **explícita**: `standing > turning > forward > heading`. O
`turning` escreve `lin = 0` **todo passo** (como o `standing`), sorteia `wz` na faixa com
`|wz| ≥ 0,2`, e **sai** do `heading` (`is_heading_env[turning] = False`), senão o heading
reescreve o `wz` dele. O smoke **mede** as frações realizadas dos quatro ramos, e não as
lê do cfg.

**Confirmado no fonte, e é bom:** todas as penalidades de marcha (`foot_clearance`,
`foot_swing_height`, `foot_slip`, `soft_landing`) e o regime do `pose` gateiam por
`‖lin‖ + |wz|`, não só pela parte linear. Um env girando com `|wz| ≥ 0,2` paga
`foot_slip` e cai no regime `walking`. Não existe o hack de "girar arrastando o pé".

⚠ **Duas coisas a saber antes de ligar:**

- **O estimador de locomoção é cego a girar.** `seg_pedido` acumula `‖cmd[:, :2]‖` —
  só a parte **linear**. Um comando de giro puro tem `ativo = 0` e não entra em
  `seg_proj/seg_pedido` nem no portão da `forma`. Isso é **correto** (o portão é sobre
  andar), mas significa que **precisa de métrica própria de guinada** para ver se o
  ramo ajuda: o `error_vel_yaw` do fabricante, ou um `razao_guinada` na mesma forma do
  `seg_proj/seg_pedido`.
- **É mudança na distribuição de comando da locomoção, que hoje funciona.** Tirar
  envs dos ramos atuais para o ramo de giro muda o que a política vê em 30% dos envs.
  Ligar com fração pequena (ponto de partida: 0,10, tirado do ramo uniforme) e medir
  `seg_proj/seg_pedido` junto — se o andar cair, é o custo aparecendo.

**Decisão do dono (02/09): ENTRA, na mesma run da v2.** Uma fração dos envs de locomoção
só gira, com `lin = 0`. Fração de partida: `rel_turning_envs = 0,10`, que pela
precedência acima realiza ~0,09 dos envs (`0,10 × 0,90`, os que não são `standing`) e
encolhe o ramo uniforme na mesma medida. Sentinelas: `error_vel_yaw` (ou um `razao_guinada`) subindo, e
`seg_proj/seg_pedido` estável — se o andar cair, é o custo aparecendo.

---

## 10. O QUE VEM DEPOIS, e o que NÃO vem

**Decisão do dono (02/09): primeiro fazer funcionar no simulador; depois complicar.** As
fases abaixo **vão acontecer** — este documento as registra para que não se esqueçam. A
§10.2 é o que **não** acontece neste trabalho nem nos seguintes já previstos.

### 10.1 ROTEIRO — fases posteriores (não esquecer)

```
FASE 1  este documento     troca de tarefa em voo funcionando NO SIM
FASE 2  DR para sim-to-real      generalizar o que a FASE 1 provou
FASE 3  LIDAR em TODAS as tarefas     referenciamento de ambiente
```

#### FASE 2 — Domain Randomization

O que o `g1_limpo` **já randomiza**, herdado do molde (medido 02/09 no cfg):

```
startup    foot_friction     atrito do pé, por env, fixo no episódio
startup    encoder_bias      viés de encoder
interval   push_robot        empurrão a cada 1–3 s
reset      carga_caixa       MASSA da caixa, 1 a 5 kg por nível (próprio do módulo)
```

⚠ **Tamanho da caixa saiu desta lista:** decisão do dono (02/09), ele entra **desde a
FASE 1** (§6.7), por `geom_size` por mundo no startup.

O que **não** randomiza, e que a FASE 2 tem de avaliar — em ordem de risco para a pega:

- **Ruído de observação nos canais de caixa.** Hoje `caixa` tem `noise = None`, enquanto
  `joint_pos` tem ±0,01 e `base_lin_vel` ±0,5. A política é treinada com pose da caixa
  **perfeita**; um estimador real tem ruído de centímetros e latência. É o primeiro da
  lista porque é o único canal que vem de percepção externa.
- **Atrito da caixa e da laje.** Fixos em `cena.py`. A pega é treinada contra **um**
  coeficiente — e contato é a classe mais difícil de sim-to-real.
- **Ganho PD, atraso de ação, limite de esforço, massa do robô.** Não randomizados. São
  os itens padrão que faltam ao molde (ver memória `g1-mjlab-velocity-anatomia`).
- **Ruído nos canais de tarefa** (`command`, `elo`): hoje `None`. Provavelmente **deve**
  ficar sem ruído — são comandos digitais, não medições. Decidir na FASE 2.

⚠ **Aditivo ao contrato deste documento.** Nenhum item acima muda observação, one-hot ou
gate — só distribuições. Pode entrar sem reabrir a FASE 1. Mas **cada item pode custar o
que já funciona** (a pega foi aprendida com um coeficiente de atrito): ligar um por vez,
com `descarga` e `seg_proj/seg_pedido` como sentinelas.

#### FASE 3 — LIDAR em todas as tarefas

Entra como referenciamento de ambiente, em **todos** os elos — é o que permite a
percepção não depender só da câmera para saber onde a mesa está.

Regras **já contratadas** para quando entrar:

- canal novo é **APPEND no fim** da observação, nos dois grupos — inserção no meio
  desloca todo peso da primeira camada em silêncio (regra que `elo` e `caixa` já seguem,
  `env_cfg.py` §3e/3f);
- o warm-start de um canal novo passa por `expande_checkpoint` (precedente no projeto) e
  pelo cuidado com o normalizador da §8.3 — um canal que nasce constante explode ao
  acender. Um scan de LIDAR **não** nasce constante, mas cada feixe que nunca acerta nada
  nasce constante no seu valor de "sem retorno";
- o molde do fabricante já tem um termo de observação raycast (`foot_height_scan`) — o
  caminho de sensor está provado, falta o sensor e a geometria dos feixes;
- ⚠ **com LIDAR, a mobília a +5 m no `ANDAR` deixa de ser invisível** — o scan a vê. A
  regra "no `ANDAR` o robô não sabe da caixa" (§4.1) continua valendo para os canais de
  caixa, mas a mesa passa a existir para o robô como obstáculo. É desejado (é o
  referenciamento). **Andar até a mesa continua NÃO sendo deste modelo** (§7.3) — a
  mesa vira obstáculo conhecido para a locomoção pilotada, não alvo de aproximação.

### 10.2 FORA DE ESCOPO — não neste trabalho

- **Andar até a mesa** com a mobília presente e o robô em movimento (§7.3). **Não é
  deste modelo** — de outro modelo ou abordagem, se vier a ser.
- **O robô decidir sozinho quando está pronto** (canal de saída novo + recompensa).
- **Faixa de comando de ±2 m/s** (hoje ±1,0).
- **Fechar o `sustentacao`** — trazer a caixa ao alvo. É o gargalo do `pegar`, e é
  trabalho de recompensa, não de contrato.
- **Handover entre controladores** — rejeitado por decisão (§1.1).

---

## 11. VERIFICAÇÃO

O que prova que o contrato funciona. Tudo mecânico, sem GPU.

### 11.1 Travas de smoke

1. **O gate.** Com o elo publicado em `ANDAR`, os 10 canais de caixa são **exatamente
   zero** — e o teste teleporta a caixa para perto antes de medir, para provar que o
   zero vem do gate e não da distância. Nos dois grupos, `actor` e `critic`.
2. **A invariante que substitui o bit.** Em nenhum passo do treino existe
   `|caixa_b| = 0` com publicado ≠ `ANDAR`, nem `|caixa_b| ≠ 0` com publicado = `ANDAR`.
   Não há terceiro estado.
3. **A dimensão.** 114 canais no `actor` e 119 no `critic`: o `VALIDA` **não** está em
   nenhuma observação; o `meia_aresta` **está**, como último canal do termo `caixa`; e o
   `elo_interno` está **só** no `critic`, depois do `caixa`. O `VALIDA` continua existindo
   no comando (é a porta dos incentivos), e o smoke afirma que ele vale exatamente
   `interno ≠ ANDAR ∧ ¬aguardando` — em particular **1 na espera final**. E `meia_aresta`
   na observação bate com `limpo_meia_aresta` env a env quando o publicado ≠ `ANDAR`.
4. **A espera publica `ANDAR`.** Num episódio de manipulação, na observação do **reset**
   (antes de qualquer passo): one-hot `ANDAR`, twist zero, canais de caixa zero. Na
   borda: one-hot `PEGAR`, canais vivos, **no mesmo passo**. E o elo **interno** é `PEGAR`
   do reset ao fim.
5. **Cada consumidor lê o lado certo** (tabela da §6.0): durante a espera, o rastreio
   **paga** (lê publicado = `ANDAR`), a postura é a do fabricante (não 1,0 neutro), os
   sete incentivos são zero, o twist é zero, a mobília está na laje, o robô está na
   faixa de manipulação, e `Curriculum/elo` conta o env como **manipulação**.
6. ⚠ **O piso de 30% de locomoção PURA fica exato.** Envs de manipulação em espera
   publicam `ANDAR` mas **não** entram na conta de locomoção — a conta lê `limpo_elo`
   (interno). Medir e comparar ao `fatia_loco`. É a trava que protege a garantia contra
   esquecimento.
7. **O publicado é recalculado do interno**, não do próprio publicado — o smoke lê o
   fonte de `_aplica_espera` e afirma isso (lição do bit destrutivo de 02/09).
8. **O visualizador não mudou.** O diff da v2 contra `exp/g1-limpo` não toca em
   `entrega_tarefa_no_viewer`, `avanca_elo_no_viewer` nem na task `Mjlab-G1-Limpo-Entrega`.
   (Em `eventos.py` mudam só `posiciona_cena` e `afasta_cena`, que passam a ler o tamanho
   por env, e entra o `tamanho_caixa` — §6.7.)
9. **A cadeia 3 tem 3 elos** e `_TETO_ELOS = 3` — e a máquina de elo os percorre: com
   `cadeia_forcada = 3` e `forca_avanco` duas vezes, o elo vai `PEGAR → CARREGAR →
   BOTAR`, e `fechou` só marca no `BOTAR`.
10. **No `CARREGAR` da cadeia 3 o twist é zero** — já na observação do reset do elo, e em
    todo passo. No `CARREGAR` da cadeia 2 o twist segue sorteado.
11. **O `CARREGAR` da cadeia 3 fecha por `perto` sustentado pela espera sorteada** (0,5 a
    1,5 s), com o robô parado — e **não** fecha se a caixa sai da âncora antes do fim da
    espera. O da cadeia 2 continua fechando por distância andada com sustain `carregar_s`.
12. **A espera final.** Depois do fecho do `BOTAR`: publicado `ANDAR` **no mesmo passo
    do fecho** (escrito em `_avanca_elo_force`, sem passo de atraso), interno `BOTAR`,
    twist zero, canais de caixa zero, `limpo_soltou = 1`, e **`VALIDA = 1`**. Afastar as
    palmas da caixa **não** termina o episódio; derrubar a caixa (`caiu`) **termina**.
    Antes do fecho do `BOTAR`, afastar as palmas continua terminando (`escapou` armado). E
    `sucesso` marca no fecho do `BOTAR`, não depois.
13. **O tamanho por mundo está no modelo e o colisor o lê.** `geom_size`, `geom_rbound` e
    `geom_aabb` da caixa diferem entre mundos e são exatamente os K valores; `body_mass`
    **não** mudou (independência do peso); `limpo_meia_aresta` bate com `geom_size` env
    a env; e — a prova de que o colisor lê — uma caixa de 0,13 m apoiada na laje repousa
    com o centro 3 cm mais alto que uma de 0,10 m no env vizinho. A caixa **segue `box`**.
14. ⚠ **A pega não mudou de física.** O `descarga` do smoke (caixa apoiada → erguida) num
    env de 0,10 m fica onde estava. Sem mesh, isto é quase tautológico — e é por isso que
    o plano A é o plano A.
15. **Todo consumidor lê o tamanho por env.** Com duas variantes de tamanhos bem
    diferentes, `alvos_das_palmas` e o z de repouso da caixa na laje diferem entre os
    envs das duas — nenhum sítio ficou lendo o escalar de `knobs`.
16. ⚠⚠ **A renda do `BOTAR` é MONÓTONA** (§6.6.2). Com o robô travado e a caixa posta à
    mão em cada um dos cinco estados da tabela — pairar, apoio parcial, apoiada com as
    mãos, espera final com as mãos, espera final com as palmas longe —, a soma dos termos
    por segundo **cresce** de um estado para o seguinte. É a trava que fecha o buraco da
    §6.6.1, e ela roda a cada mudança de peso.
17. **As máscaras do `BOTAR`.** Com o interno em `BOTAR`, `squeeze`, `unload` e
    `postura_ereta` valem exatamente zero com a caixa nas mãos e fora da laje; com o
    interno em `PEGAR` ou `CARREGAR`, no mesmo estado, valem o de hoje. E `alcança` vale
    1 no `BOTAR` com as palmas a 20 cm da caixa, e vale `< 0,1` no `PEGAR` na mesma pose.
18. **`load` e `largou`.** `load` vale 0 fora do `BOTAR` e com a caixa no ar; vale 1 com a
    caixa apoiada no alvo, e **continua 1** com a mesma caixa prensada com o dobro do peso
    (o `clamp`); vale 0 com a caixa apoiada a 25 cm do alvo. `largou` vale 0 antes de
    `soltou`; com `soltou`, caixa apoiada e palmas a 20 cm, vale ≥ 0,95.
19. **`caiu` por tamanho.** Com duas variantes de tamanho bem diferentes, a caixa maior
    deitada no chão dispara `caiu`, e a caixa menor apoiada na laje a `prateleira_topo_piso`
    **não** dispara. `caixa_folga_chao < prateleira_topo_piso`.
20. **O `_pegou` só arma com o objetivo ativo.** Encostar as duas palmas na caixa durante a
    espera inicial não arma `caixa_largada`; encostar depois da borda arma.
21. **As frações do giro são medidas.** Sobre 4096 re-sorteios do twist, o smoke conta os
    envs em cada ramo realizado (`standing`, `turning`, `forward`, `heading`, uniforme) e
    afirma `turning` realizado em `0,09 ± 0,02` (`0,10 × 0,90`, §9), `|wz| ≥ 0,2` em todo
    env `turning`, `lin = 0` neles em **todo** passo, e que nenhum env `turning` está em
    `heading`.
22. **O `PPOPorElo` agrupa pelo interno.** Num lote com um env em espera final, ele cai
    no grupo `manip`, e não no `loco`.
23. **`giro_b` é o giro, e não só o ângulo.** Com a caixa girada +90° em Z no nascimento,
    `giro_b ≈ (0, 0, ∓π/2)` e `|giro_b|` bate com o `ANG` do comando em 1e-4; girada 90°
    em Y, o eixo é Y; e o sinal troca com o sentido do giro. Em `PEGAR`, no passo em que
    o elo abre, `giro_b = 0`; depois de torcer a caixa à mão em 20°, `|giro_b| ≈ 0,35`.
24. **O `REORIENTAR` está inerte na v2** (§8.3): com o cfg de treino, `voltas_max` é zero
    em todo nível, e um env de cadeia 1 avança para o `PEGAR` em `sustenta_outros_s` sem
    a caixa se mover.

### 11.2 Simulação do caminho de campo

Um teste que monta a sequência da §3 e afirma, para cada passo, que a entrada construída
é indistinguível de uma entrada que o treino produz nos canais de tarefa (`command`,
`elo`, `caixa`). Para `ANDAR(v=0) → PEGAR` isso é verdade **por construção** (§6.3); para
as outras quatro transições da §7.1 é a afirmação que o teste faz.

### 11.3 O que só o treino responde

- Se a política **usa** o one-hot em vez da distância da caixa. Antes do gate isso é
  indecidível: os dois canais dizem a mesma coisa. Depois do gate, o one-hot é a única
  fonte, e a locomoção continuar funcionando **é** a prova.
- Se a rede aprende "canais em zero = sem alvo" sem o bit (§6.2). O sintoma de falha
  seria alcance durante a espera — visível no `play` da task de entrega e no rastreio
  caindo nesses envs.
- Se girar no lugar (§9) melhora sem custar o andar: métrica de guinada subindo com
  `seg_proj/seg_pedido` estável.

---

## 12. CUSTO

⚠ **O checkpoint atual é invalidado, por três razões independentes.** Os canais de caixa,
que valiam ~4,3 no `ANDAR`, passam a valer 0 (distribuição de observação); um canal troca de
significado (sai `VALIDA`, entra `meia_aresta`); e o `ANG` vira `giro_b`, 112 → 114
(§8.3). Resume não resolve nenhuma das três. **Treino do zero.**

A run `bloco7` (it ~2514, `precise_pos` subindo, `descarga = 0,965`) fica inconsistente
com o código novo. Ela pode seguir até onde valer, mas não recebe estas mudanças. O que
ela **provou** — que a cadeia de recompensa da pega funciona — está no código, não no
checkpoint, e o reinício reaprende com as mesmas recompensas.

**Tamanho da mudança de código:** o gate (duas linhas em `observacoes.py`), o `VALIDA`
fora da observação e o `elo_interno` no crítico (§6.1, §6.2), o publicado em `ANDAR` nas
duas esperas (uma condição em `_aplica_espera`, uma linha em `_avanca_elo_force`), o
`_pegou` gateado, o ramo de giro com precedência (§9), a cadeia 3 com duas regras por
cadeia (§6.5), a espera final com o guarda e o `caiu` por tamanho em `caixa_largada`
(§6.6, §6.7), o wrapper `tamanho_caixa` sobre `dr.geom_size` e os sítios lendo tamanho por
env (§6.7), o `PPOPorElo` agrupando pelo interno, **as máscaras do `BOTAR` e os termos
`load` e `largou`** (§6.6.2), e as travas da §11. **Nenhuma mudança em currículo.** Nos
eventos de reset, só a leitura do tamanho por env (§6.7) e, pendente, `voltas_max = 0` em
`knobs.Nivel` (§8.3). Em recompensa, **só** o que a §6.6.2 lista, e só com o interno em
`BOTAR` ou em `soltou`.

⚠ **Um item com risco, nomeado:** a cadeia 3 toca a máquina de elo (`_TETO_ELOS = 3`
nunca foi exercitado; o código é derivado de `CADEIAS`, e a leitura de 03/09 não achou
nada redigitado, mas só o smoke prova). A escrita em `geom_size` deixou de ser risco deste
módulo: é a função de DR do próprio mjlab (§6.7). **Sentinelas na primeira run:**
`descarga` e `rampa` para a pega; `seg_proj/seg_pedido` para o andar; e, novos,
`Episode_Reward/load` e `Metrics/alvo_caixa/passo_final` para ver se o `BOTAR` fecha —
`sucesso` da cadeia 3 saindo de zero é o veredito da §6.6.2. A pega **não** mudou de
física — se ela cair, o suspeito é a distribuição de tamanhos, não o colisor.

O custo real é o treino, não o código.

**Onde:** na branch `exp/g1-limpo-v2`, criada de `exp/g1-limpo` HEAD. A `exp/g1-limpo`
fica **intocada como referência** — é o código que treinou a `bloco7` e funcionou. A
trava da §2 ("nada acima é tocado") vira `git diff exp/g1-limpo -- g1_limpo/` vazio em
`curriculo.py`, e nos pesos de `knobs.py` **existentes** (os novos, `load`, `largou`,
`sigma_solta`, `caixa_folga_chao`, são adições). Em `recompensas.py` o diff toca só
`_alcancar`, `squeeze`, `unload` e os dois termos novos; em `terminacoes.py` só
`caixa_largada`. O smoke da §11.1 (itens 17 e 18) afirma que fora do `BOTAR` os valores
são os de hoje.

---

## 13. DECISÕES

**Tomadas (02/09):**

- **Objetivo: só o contrato de troca.** Andar até a mesa fica fora (§7.3).
- **Recompensas, penalidades, progressão loco → manipulação, e o piso de 30% de
  locomoção pura: NÃO SE TOCA.** Funcionam, e ficam.
- **Protocolo de um estágio:** `elo` e canais de caixa chegam juntos; o operador garante
  o robô parado. Em `v = 0` a pose é praticamente a default.
- **A espera publica `ANDAR`** (§6.3). Era a única mudança na manipulação nesta rodada:
  `espera_s` deixa de ser "PEGAR com objetivo desligado" e vira "ANDAR parado, e então
  PEGAR". Sem cadeia nova, sem tocar currículo.
- **O `VALIDA` sai da observação** (§6.2). Fica no sim como porta de recompensa e de
  fecho de elo. (Com o `meia_aresta` da sexta rodada e o `giro_b` da sétima, a
  observação fica em 114.)
- **`REORIENTAR` é habilidade futura:** fica no código E no sorteio (§8.3); o defeito
  do avanço grátis fica registrado, não consertado.
- **Primeiro funcionar no sim, depois complicar.** DR é a FASE 2 e LIDAR em todas as
  tarefas é a FASE 3 do roteiro (§10.1). Registradas para não esquecer; não agora.

**Tomadas (02/09, segunda rodada):**

- **`espera_s` = tempo em `ANDAR` com twist zero antes do `PEGAR`.** (A faixa foi para
  (0,5, 1,5) na quarta rodada, §6.3.)
- **Girar no lugar ENTRA**, na mesma run: `rel_turning_envs = 0,10` (§9).
- **Próxima run já no modelo novo.** Não se espera o veredito da `bloco7`.
- **O visualizador fica como está** (§6.4).
- **Branch `exp/g1-limpo-v2`** de `exp/g1-limpo`; a `exp/g1-limpo` é a referência que
  funcionou e não se toca (§12).
- **Andar até a mesa não é deste modelo** (§3.3, §7.3, §10.2).

**Tomadas (02/09, terceira rodada):**

- **Só os quatro comportamentos** (§3.0): andar, pegar, carregar, botar. O resto é
  derivado, composto pelo controlador via one-hot.
- **`PEGAR → BOTAR` direto não é treinado.** O controlador nunca manda isso.
- **A cadeia 3 vira `(PEGAR, CARREGAR, BOTAR)`**, com o `CARREGAR` do meio em `v = 0` e
  fecho pela espera (na sétima rodada: `perto` sustentado pela espera, §6.5).
  `CARREGAR(v=0) → BOTAR` sai da aposta.
- **A espera FINAL** (§6.6): depois do `BOTAR`, publicado `ANDAR` com `v = 0` até o fim
  do episódio — o robô larga e fica parado. `escapou` desarmado, `caiu` armado. Sem
  cronômetro. `BOTAR → ANDAR(v=0)` sai da aposta. **Nenhuma aposta sobra.**

**Tomadas (02/09, quarta rodada):**

- **Toda espera é 0,5 a 1,5 s.** Um knob, `espera_s = (0,5, 1,5)`, para a espera inicial
  (§6.3) e para o "segurar parado" da cadeia 3 (§6.5). A espera final não tem
  cronômetro (§6.6). Não existe `segurar_s`.
- **Giro no lugar via `rel_turning_envs`** (§9) — flag re-sorteada a cada re-sorteio de
  comando, no idioma do fabricante. Com `|wz|` mínimo.

**Tomadas (02/09, quinta rodada):**

- **DR de tamanho da caixa desde a FASE 1** (§6.7): `geom_size` + `rbound` + `aabb` por
  mundo, escritos uma vez no startup — a classe do `foot_friction`. Cubo, 14 a 26 cm,
  K = 8, aleatório por env, fixo na run. A caixa **segue `box` primitivo**; o colisor não
  muda; `paridade` não diverge. Mesh (`VariantEntityCfg`) é o plano B, só se a GPU
  reclamar.

**Tomadas (02/09, sexta rodada — fechava a v10):**

- **A rede vê o tamanho.** Canal `meia_aresta` no fim do termo `caixa`, gateado. (Com o
  `giro_b` da sétima rodada, 114 canais — §4, §6.7.)
- **`rel_turning_envs = 0,10`**, com `|wz| ≥ 0,2 rad/s` (§9).
- **Tamanho: (0,07, 0,13) m, K = 8** (§6.7).

**Proposta da revisão de consistência (03/09, sétima rodada) — EM REVISÃO PELO DONO:**

A revisão cruzou a v12 com `comando.py`, `recompensas.py`, `terminacoes.py`, `knobs.py`,
o `g1_poc` e o fonte do mjlab 1.5.1 e do `mujoco_warp`. Achou três buracos de
aprendizado e propõe:

- **O `BOTAR` ganha a tabela do `g1_poc` e o "depois" fica pago** (§6.6.1, §6.6.2): o
  `VALIDA` continua derivado do interno (a espera final **não** o zera); `squeeze` e
  `unload` zeram no `BOTAR`; `alcança ≡ 1` no `BOTAR`; termos novos `load` (2,0) e
  `largou` (1,0). A renda vira monótona: pairar 11,5 < apoiar 13,5 < espera final 16,5 <
  largar 18,5. É a única exceção à regra "recompensas não se toca", e vale só para o
  `BOTAR`, que nunca fechou.
- **O crítico vê o elo interno** (§6.1) e o `PPOPorElo` agrupa por ele. Ator-crítico
  assimétrico; não toca o deploy.
- **`caiu` por tamanho** (§6.7): `caixa_folga_chao = 0,02` substitui `caixa_z_min`.
- **Fecho da cadeia 3** = `perto` sustentado pela espera sorteada (§6.5), sem `andou`.
- **`_pegou` só arma com o objetivo ativo** (§6.3); publicado escrito no fecho do `BOTAR`
  (§6.6); `fracao_esperando` lê as duas esperas; `recebe_tarefa` escreve `ELO` (§6.4).
- **Giro com precedência explícita** e frações medidas (§9).
- **`dr.geom_size` do mjlab** com um wrapper para o cubo (§6.7). O plano B fica.
- **O `REORIENTAR` ganha o contrato, e não o treino** (§8.3, refinado com o dono em
  03/09, v14): `ANG` vira `giro_b` — 3 canais, o giro pedido no frame da base; 114
  canais no ator, 119 no crítico. Quatro primitivas de 90° (esquerda, direita, cima,
  baixo), compostas pelo controlador externo, que também guarda "quais faces já vi".
  Direção pedida horizontal, para o robô (física da caixa apoiada). Termina como o
  `BOTAR`: larga a caixa e tira as mãos — o desenho do treino é o espelho da §6.6.2 e
  está escrito, pronto. Ele **não herda o eixo de altura do `PEGAR`**: mesa entre 0,45
  e 0,55 m, independente do nível; a variação de carga **fica**, porque pesa na
  capacidade de girar (`knobs.Reorientar`, quando virar foco). No real o robô bota na
  mesa, gira, e pega de novo. Na run da v2 ele fica **inerte** (`voltas_max = 0` em todo
  nível) e sorteável, para o slot não ficar constante; o checkpoint da v2 serve de
  warm-start quando virar foco, porque nada do que falta muda a observação.

**Pendentes: a aprovação da sétima rodada.** Em particular: os pesos 2,0 e 1,0 e o
`σ_solta = 0,10` são ponto de partida; e `voltas_max = 0` em todo nível na run da v2 é
mudança em `knobs.Nivel` que o dono confirma. Depois da aprovação, o próximo passo é o
plano de implementação.
