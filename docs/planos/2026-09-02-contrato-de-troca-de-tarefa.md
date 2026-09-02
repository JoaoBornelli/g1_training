# Contrato de troca de tarefa — g1_limpo

**Estado:** APROVADO COM EMENDAS em 2026-09-02 (§13). Nada implementado. Próximo passo: plano de implementação na branch `exp/g1-limpo-v2`.
**Escrito:** 2026-09-02 · **Revisado:** 2026-09-02 (v9 — DR de TAMANHO da caixa desde a FASE 1, via variantes de entidade)
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

⚠ **Duas exceções, pedidas pelo dono (02/09, terceira rodada):**

- **a cadeia 3 muda** de `(PEGAR, BOTAR)` para `(PEGAR, CARREGAR, BOTAR)` — o `botar`
  passa a vir de "segurar parado", e não direto da pega (§3.0, §6.5). As outras três
  cadeias não mudam;
- **a terminação `caixa_largada` ganha um guarda**: na espera final, depois do botar,
  as mãos saem da caixa e `escapou` não pode disparar; `caiu` continua armado (§6.6).
  Fora da espera final ela não muda;
- **a caixa vira entidade com variantes de tamanho** (§6.7). A pega foi aprendida com
  uma caixa de 20 cm; passa a ser aprendida com 14 a 26 cm. É DR pedida pelo dono
  **desde o começo**, e é a mudança desta spec com mais efeito sobre a manipulação.

O que este documento muda está **inteiro** na §6: observação, um canal de comando, a
cadeia 3, um guarda de terminação e a geometria da caixa. Nenhuma recompensa, nenhum
currículo.

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

A observação do `actor` tem **111 canais**, nesta ordem:

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
| 8 | `caixa` | 7 | ver §4.1 |

Total: `3+3+3+29+29+29+3+5+7 = 111`.

⚠ **Eram 112.** O canal `VALIDA` sai da observação (§6.2). Ele continua existindo
**dentro** do sim, como porta de recompensa e de fecho de elo — só não é mais entrada da
rede.

⚠ **E podem voltar a ser 112**, por outro motivo: a §6.7 propõe que o **tamanho da caixa**
entre como canal (`meia_aresta`, 1), no fim do termo `caixa`. Pendente do dono (§13).

### 4.1 Os 7 canais de caixa, em detalhe

| canal | dim | o que é | de onde vem em campo |
|---|---|---|---|
| `caixa_b` | 3 | **posição** da caixa, no frame da base | **PERCEPÇÃO** |
| `alvo_b` | 3 | **posição** do alvo, no frame da base | calculado A BORDO (exceto `BOTAR`) |
| `ANG` | 1 | **escalar**, em radianos: erro angular entre a normal da face marcada e a face pedida | calculado A BORDO, de percepção + tarefa |

⚠⚠ **Os 7 canais são ZERO quando o one-hot publicado é `ANDAR`**, e vivos em todo outro
caso. Não existe terceiro estado. Esta é a **invariante que substitui o bit**, e é o que
a rede lê como "existe tarefa de caixa": canais preenchidos, ou canais em zero.

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

- **Não existe quatérnion na entrada.** A orientação da caixa entra como **um escalar**
  (`ANG`). O quatérnion é insumo do cálculo a bordo, não entrada da rede.
- **O alvo não é enviado**, em quase todos os casos. No `PEGAR` e no `CARREGAR` ele é
  **ancorado na base do robô** — um offset fixo no frame dele, calculado a cada passo.
  Só o `BOTAR` precisa de alvo externo.
- **Tudo em frame da base**, e não em mundo. Isso dispensa origem global; em troca, a
  percepção tem de ser egocêntrica ou transformada.
- **`ANG` precisa saber QUAL face.** A caixa tem uma face marcada (`face_alvo_b`). Em
  campo, a percepção tem de identificar essa face — fiducial, ou geometria conhecida.
  Sem isso o `ANG` não é computável.
- **Não existe bit "a caixa existe".** Ele era redundante com o one-hot (§6.2). Em campo,
  "a caixa existe" é a camada de tarefa **preencher** os 7 canais ou **zerá-los** — a
  mesma regra que o sim aplica.

### 4.2 O que o operador manda, no total

```
twist        3 números    vx, vy, wz      (piloto, contínuo)
elo          5 números    one-hot         (botão de tarefa)
alvo BOTAR   3 números    só no BOTAR
                          -----
                          11 números, no máximo
```

Mais, da percepção: **posição da caixa (3)** e **orientação da caixa (4)**, para o
cálculo a bordo de `caixa_b` e `ANG`.

#### As regras da camada de tarefa — do lado do robô real

Quatro regras, e todas espelham algo que o **sim** faz por conta própria. Em campo,
**nada as faz sozinho** — a camada de tarefa tem de implementá-las.

1. **`elo = ANDAR` ⟹ os 7 canais de caixa em zero.** Sempre, mesmo com a caixa à vista da
   percepção. É a regra que substitui o bit. (Sim: o gate da §6.1.)
2. ⚠ **`elo ∈ {PEGAR, REORIENTAR, BOTAR}` ⟹ twist forçado a zero**, seja o que for que o
   piloto mande. (Sim: `_zera_twist_nos_parados`.) Sem isso, um joystick encostado
   durante o `PEGAR` põe a política fora de distribuição — ela nunca viu twist ≠ 0 nesses
   elos.
3. **`alvo_b` é calculado a bordo** em `PEGAR` e `CARREGAR`:
   `(peito_b.x, peito_b.y, altura_carregar − z_base)` = `(0,25, 0,00, 0,95 − z_base)`.
   É **enviado** só no `BOTAR`. (Sim: `_alvo_ancorado_na_base`.)
4. **`ANG` é calculado a bordo** do quatérnion da caixa (percepção) e da face pedida. A
   percepção tem de identificar a face marcada (`face_alvo_b`).

Sem o bit, não há segundo canal que possa contradizer a regra 1.

### 4.3 A tabela de tarefas (`elo`)

| slot | elo | twist | canais de caixa | alvo | no cenário de campo |
|---|---|---|---|---|---|
| 0 | `ANDAR` | **ativo** — ou zero, na espera de um episódio de manipulação (§6.3) | **zero** | — | pilotar sem caixa; e o "parado antes de pegar" |
| 1 | `REORIENTAR` | zero | vivos | a própria caixa | ⚠ não aparece no cenário — ver §8.3 |
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

Dos 111 canais, **85 são proprioceptivos e automáticos** (IMU, encoders, última ação).
Os **15 que carregam tarefa** são montados pela camada de tarefa a cada passo, a 50 Hz:

```
command   3   vx  vy  wz
elo       5   [ANDAR, REORIENTAR, PEGAR, CARREGAR, BOTAR]
caixa     7   caixa_b(3)  alvo_b(3)  ANG(1)
```

Valores de exemplo com os constantes reais: `peito_b = (0,25, 0, 0,15)`,
`altura_carregar = 0,95`, pelve de pé a ~0,80 m, laje a 0,55 m, caixa de 20 cm.

```
FASE 1 — ANDAR até a mesa (piloto)
  twist    ( 0,50   0,00   0,00 )      PILOTO
  elo      [ 1  0  0  0  0 ]            OPERADOR
  caixa_b  ( 0,00   0,00   0,00 )      camada ZERA  <- a percepção pode já ver a caixa; não passa
  alvo_b   ( 0,00   0,00   0,00 )      zera
  ANG        0,00                       zera

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
  ANG        0,12 rad                   A BORDO
  fim: caixa erguida e segura — no sim, condição sustentada 0,5 s; em campo, o operador vê

FASE 4 — CARREGAR (andar com a caixa)
  twist    ( 0,40   0,00   0,10 )      PILOTO
  elo      [ 0  0  0  1  0 ]            OPERADOR
  caixa_b  ( 0,24   0,01  +0,12 )      PERCEPÇÃO: agora perto do peito
  alvo_b   ( 0,25   0,00  +0,15 )      A BORDO: mesma âncora — a caixa deve ficar nela
  ANG        0,05 rad                   A BORDO

FASE 5 — parado no destino, caixa na mão (v = 0)
  twist    ( 0,00   0,00   0,00 )      PILOTO zera
  elo      [ 0  0  0  1  0 ]            ainda CARREGAR — TREINADO: é a fase do meio da cadeia 3 (§6.5)
  caixa    vivos, como acima

FASE 6 — BOTAR                          <- o botão
  twist    ( 0,00   0,00   0,00 )      camada FORÇA zero (regra 2)
  elo      [ 0  0  0  0  1 ]            OPERADOR
  caixa_b  ( 0,24   0,01  +0,12 )      PERCEPÇÃO
  alvo_b   ( 0,35   0,00  -0,20 )      EXTERNO — o único alvo enviado: onde botar, no frame da base
  ANG        0,05 rad                   A BORDO: face pedida para o botar
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

Seis mudanças: três em **observação e canal de comando**, uma na **cadeia 3**, um
**guarda de terminação**, e a **geometria da caixa** (DR de tamanho). Nenhuma em
recompensa ou currículo.

### 6.0 Os dois `elo` — o PUBLICADO e o INTERNO

A mudança da §6.3 depende de uma separação que **já existe** no código e que este
documento torna explícita. Medido em 02/09, por leitura de fonte:

| quem | lê | hoje |
|---|---|---|
| observação `um_de_cinco` | `comando[:, ELO]` | **publicado** |
| `PosturaPorElo` | `comando[:, ELO]` | **publicado** |
| `rastreio_por_elo` (`_anda_neste_elo`) | `comando[:, ELO]` | **publicado** |
| os sete incentivos | `VALIDA` → passa a ser `publicado ≠ ANDAR` | **publicado** |
| `PPOPorElo` (normalização de vantagem por grupo) | o one-hot da observação | **publicado** |
| `_zera_twist_nos_parados` | `self._elo` | interno |
| `_aplica_elo` (laje, alvo, face) | `self._elo` | interno |
| `_fecha_elo_corrente`, `_avanca_elo` | `self._elo` | interno |
| `reset_base_por_elo`, fatia (`Curriculum/elo`) | `limpo_elo` do currículo | interno |

**Tudo o que a política e a recompensa veem lê o publicado. Tudo o que é mecânica do
episódio lê o interno.** Hoje os dois são sempre iguais (`_aplica_elo` copia um no
outro). A §6.3 os faz **diferir durante a espera** — e a tabela acima é o que garante
que cada consumidor lê o lado certo.

⚠ Isto **não é padrão novo**. É exatamente a separação que o `VALIDA` já faz hoje: ele é
um bit **publicado** que difere do estado interno durante a janela. A mudança estende
isso do bit ao one-hot, e então o bit fica redundante (§6.2).

### 6.1 Gate dos canais de caixa

`caixa_b`, `alvo_b` e `ANG` vão a **zero** quando o **elo publicado** é `ANDAR`. Vivos em
todo outro caso. Não existe terceiro estado.

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

⚠ **Gatear o crítico também.** Não gatear daria informação privilegiada ao crítico
(assimetria legítima em ator-crítico), mas em `ANDAR` a caixa é irrelevante para o
retorno — não há ganho, e há risco de o crítico condicionar num canal que o ator não vê.

### 6.2 O `VALIDA` sai da OBSERVAÇÃO

O `VALIDA` é um bit **derivado**. Com a §6.3, ele vale **exatamente `elo publicado ≠
ANDAR`** — função pura do one-hot. E com o gate, os canais de caixa preenchidos-ou-zero
dizem a mesma coisa de novo. Ele não carrega **nada** que as outras entradas não
carreguem.

**Sai da observação. Fica dentro do sim**, como a porta que multiplica os sete incentivos
e que impede o fecho de elo com o objetivo desligado. Um nome mais honesto para o que ele
é: `objetivo_ativo`.

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

**A mudança de dimensão (112 → 111) é de graça agora:** o gate já força reinício (§12).

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
| incentivos de caixa em zero | `objetivo_ativo` = publicado ≠ `ANDAR` = falso | publicado |
| o elo não fechar durante a espera | `_fecha_elo_corrente & objetivo_ativo` | publicado |

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
anterior deixa o canal preso. `publicado = ANDAR se aguardando, senão interno`.

⚠ **O vazamento físico não morde aqui.** O robô não anda durante a espera (twist zero
pelo interno). Se a política ignorar o comando zero e derivar para a mesa, o rastreio
cobra e o `contato_tronco` cobra. É a pressão certa: "fique parado perto da mesa" **é** o
estado de campo.

### 6.4 O que SAI

- **O `VALIDA` da observação** (§6.2). Só isso sai. A janela **fica** — ela é o mecanismo
  da §6.3, com o one-hot publicado trocado.
- A métrica `fracao_esperando` **fica** — ela passa a medir "fração de passos em `ANDAR`
  publicado dentro de um episódio de manipulação", que é exatamente o que se quer ver.
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
   `elo ∈ parados **ou** (elo == CARREGAR e cadeia == 3)`. Na cadeia 2 o `CARREGAR`
   continua andando.
3. ⚠ **O fecho do `CARREGAR` da cadeia 3 é por TEMPO, não por distância.** Hoje o
   `CARREGAR` fecha quando o robô **andou** `carregar_dist_m` — com twist zero ele nunca
   fecharia e o `BOTAR` nunca chegaria. Na cadeia 3 o fecho é `t_no_elo ≥ espera`, com a
   espera sorteada do **mesmo knob `espera_s = (0,5, 1,5)` da espera inicial** (decisão do
   dono, 02/09: toda espera é a mesma faixa). Regra **por cadeia** no
   `_fecha_elo_corrente`. Não existe `segurar_s` separado.
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

Por que o interno ficar `BOTAR` dá quase tudo de graça:

| a espera final precisa de | quem garante | lê |
|---|---|---|
| twist zero | `_zera_twist_nos_parados` — `BOTAR ∈ elos_parados` | interno |
| mobília no lugar, caixa na laje | o ramo `BOTAR` do `_aplica_elo` não guarda a laje | interno |
| canais de caixa zero | o gate, sobre o publicado = `ANDAR` | publicado |
| rastreio pagar por ficar parado | `rastreio_por_elo`, publicado = `ANDAR` | publicado |
| os braços voltarem à postura de pé — **o largar** | `PosturaPorElo`, publicado = `ANDAR` → postura do fabricante | publicado |
| incentivos de caixa em zero | `objetivo_ativo` = publicado ≠ `ANDAR` = falso | publicado |

⚠ **A única coisa que NÃO vem de graça: a terminação `caixa_largada`.** Ela é
`(caiu | escapou) & pegou`, e `escapou` é "as duas palmas longe da caixa". Na espera final
as mãos **têm** de sair da caixa — `escapou` dispararia no primeiro passo e mataria o
episódio por fazer a coisa certa. Portanto:

```
caixa_largada = (caiu | (escapou & ~soltou)) & pegou
```

onde `soltou` é publicado pelo comando (`env.limpo_soltou`) quando a espera final começa.
**`caiu` continua armado:** largar é permitido, **derrubar não** — é o "não jogar" do
dono, estendido ao depois. É uma linha na terminação, e é a segunda exceção da §2.

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

#### O mecanismo: `VariantEntityCfg` — variantes na CONSTRUÇÃO

O `mjlab` tem `entity/variants.py`: uma entidade com **K variantes**, cada uma um `MjSpec`
próprio, todas com a mesma topologia cinemática. Na inicialização da simulação, cada
variante é compilada uma vez e os campos dependentes de geometria são espalhados por
mundo: `geom_size`, `geom_rbound`, `geom_aabb`, `body_mass`, `body_inertia`, e mais.

⚠ **Três restrições do mecanismo, e as três entram no desenho:**

1. **"Only mesh geoms can differ."** A caixa hoje é um `box` **primitivo**. Para variar
   por mundo ela tem de virar **mesh** — um mesh de caixa (8 vértices) por tamanho. É
   mudança em `cena.py` (`_spec_box`). ⚠ E troca o caminho de colisão: mesh convexo em
   vez de box primitivo. A pega foi aprendida com box primitivo. É o risco declarado
   desta mudança — ver verificação, §11.1 item 14.
2. **A atribuição de mundo → variante é fixa na inicialização.** Não re-sorteia no reset.
   "Aleatória para todos os envs" vale **entre envs**: cada env vê sempre a mesma caixa
   durante a run. Com 4096–8192 envs e K tamanhos, a frota cobre a distribuição (~500 a
   1000 envs por tamanho com K = 8). A política vê todos os tamanhos; cada env, um.
3. **`body_mass` é compilado por variante.** Com `mass=` **explícito e igual** em todas
   (o `get_box_spec` já recebe `caixa_massa`, não densidade), a massa não muda com o
   tamanho. A **independência do peso** é garantida na construção, e a força externa do
   `carga_caixa` segue por cima, como hoje. A **inércia** varia com o tamanho — correto
   para "mesma massa, tamanho diferente".

#### O desenho

```
caixa_meia_aresta_faixa  = (0,07, 0,13)   m      ±30% em torno dos 0,10 de hoje   (knob)
caixa_n_variantes        = 8                      passo de ~0,86 cm                 (knob)
escala                   = UNIFORME nos 3 eixos   a caixa segue cubo                (decisão)
atribuição               = aleatória por env, com semente, fixa na run
massa                    = igual em todas as variantes; a carga segue pelo wrench
```

**Escala uniforme (cubo), e não por eixo.** Por eixo daria 3 graus de liberdade e K³
combinações, e mudaria a forma da pega (uma caixa achatada pede outra abertura de mãos).
O pedido foi "tamanho"; cubo é a leitura direta. Por eixo fica registrado como extensão.

**Fonte de verdade por env: o modelo.** Uma vez, na inicialização, lê-se
`geom_size[world, geom_da_caixa]` e publica-se `env.limpo_meia_aresta` (n, 3). Todo
consumidor lê dali. Hoje **oito sítios** leem um escalar de `knobs`, e cada um passa a ler
o tensor por env:

| onde | o que lê hoje | passa a ler |
|---|---|---|
| `comando.alvos_das_palmas` | `cfg.caixa_meia_aresta` — o offset lateral das palmas | `limpo_meia_aresta[ids, 1]` |
| `comando` ramo `BOTAR` (3 sítios) | `cfg.caixa_meia_z` — fundo da caixa e z do alvo | `limpo_meia_aresta[ids, 2]` |
| `eventos.posiciona_cena` | `caixa_meia_z` — z de repouso na laje | `limpo_meia_aresta[ids, 2]` |
| `eventos.afasta_cena` | `caixa_meia_z` | idem |
| `inspeciona`, `paridade` | `k.cena.caixa_meia_aresta` | a variante de referência (0,10) |

⚠ **`paridade.py` passa a ter uma divergência declarada:** a caixa de referência é um
`box` primitivo no `g1_poc`; aqui vira mesh. O `paridade` compara a variante de 0,10 m e
aceita a diferença de tipo de geom como divergência **nomeada**, não como falha.

#### A pergunta que a DR de tamanho abre: a rede VÊ o tamanho?

Hoje `caixa_b` é o **centro** da caixa. O tamanho não está na observação. Com K tamanhos
de 14 a 26 cm, o alvo das palmas (§4.1, `alvos_das_palmas`) muda **±3 cm por lado** entre
envs — e a recompensa sabe o tamanho certo de cada env, mas a política não.

| | sem observar | observando (+1 canal) |
|---|---|---|
| o que a política aprende | uma abertura de mãos **média**, e corrige por contato | a abertura **certa** para cada caixa |
| contato como sinal | só via `joint_pos`/`joint_vel` — não há força de palma na observação | idem, mas não precisa |
| em campo | a percepção **tem** o tamanho (é um bounding box) e ele seria jogado fora | o que o campo tem, a rede recebe — é o princípio da §4 |
| custo | zero | +1 canal, gateado como os outros 7; 111 → 112 |
| risco | pega pior em todos os tamanhos, pela média | nenhum específico |

**Recomendação: observar.** Um canal `meia_aresta` (o meio-lado, em metros) no fim do
termo `caixa`, gateado a zero com o publicado em `ANDAR` como os outros sete. Em campo
ele vem da percepção, como `caixa_b`. A política é sem memória (§7.2): ela não tem como
"descobrir" o tamanho ao longo do episódio — ou ela o vê, ou ela chuta. **Pendente do dono
(§13).**

#### O que NÃO muda

Recompensa nenhuma. `unload` e `squeeze` derivam de `limpo_massa`, que segue igual.
`staged` usa `dist_palma_caixa`, que já lê `alvos_das_palmas` — e este passa a ler o
tamanho por env. As terminações não mudam (`caixa_dist_max = 0,45` é absoluto e cobre
todos os tamanhos). O currículo não muda.

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
| `BOTAR → ANDAR (v=0)` | espera final, cadeia 3 (§6.6) | one-hot vira; canais de caixa apagam; as mãos saem da caixa sem terminar o episódio |

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

### 8.3 O `REORIENTAR` avança de graça — habilidade FUTURA, defeito registrado

**Decisão do dono (02/09): o `REORIENTAR` é habilidade futura.** Não é foco agora, mas
o código fica pronto para treiná-la eventualmente. Portanto:

- **o elo, o slot do one-hot, a cadeia 1 e o fechamento FICAM** no código
- **ele CONTINUA sorteável** — ver o porquê abaixo, que é contra-intuitivo
- o defeito abaixo fica registrado e **não é consertado agora**

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

**Quando a reorientação virar foco**, o conserto registrado, em ordem de custo:

1. **`desalinho_max_deg > tol_ang_deg`** em todo nível, para a caixa nunca nascer
   dentro da tolerância de fechamento. Custo: uma tabela de números.
2. **Exigir erro inicial mínimo** no fechamento (o elo só fecha se houve trabalho).
   Custo: lógica nova no `_fecha_elo_corrente`.

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
só gira, com `lin = 0`. Fração de partida: 0,10, tirada do ramo uniforme (que vai de 0,40
a 0,30). Sentinelas: `error_vel_yaw` (ou um `razao_guinada`) subindo, e
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
FASE 1** (§6.7), por variantes de entidade.

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

1. **O gate.** Com o elo publicado em `ANDAR`, os 7 canais de caixa são **exatamente
   zero** — e o teste teleporta a caixa para perto antes de medir, para provar que o
   zero vem do gate e não da distância. Nos dois grupos, `actor` e `critic`.
2. **A invariante que substitui o bit.** Em nenhum passo do treino existe
   `|caixa_b| = 0` com publicado ≠ `ANDAR`, nem `|caixa_b| ≠ 0` com publicado = `ANDAR`.
   Não há terceiro estado.
3. **A dimensão.** 111 canais no `actor`; o `VALIDA` não está na observação. Ele
   continua existindo no comando (é a porta dos sete incentivos), e o smoke afirma que
   ele vale exatamente `publicado ≠ ANDAR`.
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
   `eventos.py` (modo de entrega), nem na task `Mjlab-G1-Limpo-Entrega`.
9. **A cadeia 3 tem 3 elos** e `_TETO_ELOS = 3` — e a máquina de elo os percorre: com
   `cadeia_forcada = 3` e `forca_avanco` duas vezes, o elo vai `PEGAR → CARREGAR →
   BOTAR`, e `fechou` só marca no `BOTAR`.
10. **No `CARREGAR` da cadeia 3 o twist é zero** — já na observação do reset do elo, e em
    todo passo. No `CARREGAR` da cadeia 2 o twist segue sorteado.
11. **O `CARREGAR` da cadeia 3 fecha por tempo**, dentro da faixa de `segurar_s`, com o
    robô parado — e o da cadeia 2 continua fechando por distância andada.
12. **A espera final.** Depois do fecho do `BOTAR`: publicado `ANDAR`, interno `BOTAR`,
    twist zero, canais de caixa zero, `limpo_soltou = 1`. Afastar as palmas da caixa
    **não** termina o episódio; derrubar a caixa (`caiu`) **termina**. Antes do fecho do
    `BOTAR`, afastar as palmas continua terminando (`escapou` armado). E `sucesso` marca
    no fecho do `BOTAR`, não depois.
13. **As variantes existem e são K.** `geom_size` da caixa difere entre mundos; a
    contagem de mundos por variante bate com a atribuição; **`body_mass` é igual em
    todas** (independência do peso); `limpo_meia_aresta` bate com `geom_size` env a env.
14. ⚠ **A pega sobrevive ao mesh.** O `descarga` medido no smoke (caixa apoiada → erguida)
    na variante de referência (0,10 m) fica onde estava com o `box` primitivo. É a trava
    contra o risco 1 da §6.7.
15. **Todo consumidor lê o tamanho por env.** Com duas variantes de tamanhos bem
    diferentes, `alvos_das_palmas` e o z de repouso da caixa na laje diferem entre os
    envs das duas — nenhum sítio ficou lendo o escalar de `knobs`.

### 11.2 Simulação do caminho de campo

Um teste que monta a sequência da §3 e afirma, para cada passo, que a entrada construída
é indistinguível de uma entrada que o treino produz nos canais de tarefa (`command`,
`elo`, `caixa`). Para `ANDAR(v=0) → PEGAR` isso é verdade **por construção** (§6.3); para
as duas apostas da §7.2 é a afirmação que o teste faz.

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

⚠ **O checkpoint atual é invalidado, por duas razões independentes.** Sete canais que
valiam ~4,3 no `ANDAR` passam a valer 0 (distribuição de observação), e a observação
perde um canal (112 → 111, a primeira camada muda de forma). Resume não resolve
nenhuma das duas. **Treino do zero.**

A run `bloco7` (it ~2514, `precise_pos` subindo, `descarga = 0,965`) fica inconsistente
com o código novo. Ela pode seguir até onde valer, mas não recebe estas mudanças. O que
ela **provou** — que a cadeia de recompensa da pega funciona — está no código, não no
checkpoint, e o reinício reaprende com as mesmas recompensas.

**Tamanho da mudança de código:** o gate (duas linhas em `observacoes.py`), o `VALIDA`
fora da observação (uma linha), o publicado em `ANDAR` na espera (uma condição em
`_aplica_espera`), o ramo de giro (§9), a cadeia 3 com duas regras por cadeia (§6.5), a espera final com
um guarda em `caixa_largada` (§6.6), a caixa como entidade com variantes e oito sítios
lendo tamanho por env (§6.7), e as travas da §11. **Nenhuma mudança em recompensa,
currículo ou reset.**

⚠ **Dois itens com risco de regressão, e os dois estão nomeados:** a cadeia 3 toca a
máquina de elo (`_TETO_ELOS = 3` nunca foi exercitado), e a caixa vira mesh (a pega foi
aprendida com box primitivo). **Sentinelas na primeira run:** `descarga` e `rampa` para a
pega; `seg_proj/seg_pedido` para o andar. Se a pega cair e o andar não, o suspeito é a
§6.7 antes da §6.5.

O custo real é o treino, não o código.

**Onde:** na branch `exp/g1-limpo-v2`, criada de `exp/g1-limpo` HEAD. A `exp/g1-limpo`
fica **intocada como referência** — é o código que treinou a `bloco7` e funcionou. A
trava da §2 ("nada acima é tocado") vira `git diff exp/g1-limpo -- g1_limpo/` vazio em
`recompensas.py`, `terminacoes.py`, `curriculo.py` e nos pesos de `knobs.py`.

---

## 13. DECISÕES

**Tomadas (02/09):**

- **Objetivo: só o contrato de troca.** Andar até a mesa fica fora (§7.3).
- **Recompensas, penalidades, progressão loco → manipulação, e o piso de 30% de
  locomoção pura: NÃO SE TOCA.** Funcionam, e ficam.
- **Protocolo de um estágio:** `elo` e canais de caixa chegam juntos; o operador garante
  o robô parado. Em `v = 0` a pose é praticamente a default.
- **A espera publica `ANDAR`** (§6.3). É a única mudança na manipulação: `espera_s`
  deixa de ser "PEGAR com objetivo desligado" e vira "ANDAR parado, e então PEGAR". Sem
  cadeia nova, sem tocar currículo.
- **O `VALIDA` sai da observação** (§6.2). Fica no sim como porta de recompensa e de
  fecho de elo. 111 canais.
- **`REORIENTAR` é habilidade futura:** fica no código E no sorteio (§8.3); o defeito
  do avanço grátis fica registrado, não consertado.
- **Primeiro funcionar no sim, depois complicar.** DR é a FASE 2 e LIDAR em todas as
  tarefas é a FASE 3 do roteiro (§10.1). Registradas para não esquecer; não agora.

**Tomadas (02/09, segunda rodada):**

- **`espera_s` = tempo em `ANDAR` com twist zero antes do `PEGAR`.** Faixa (0,3, 1,0) na
  primeira run da v2 (§6.3).
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
  fecho por tempo (§6.5). `CARREGAR(v=0) → BOTAR` sai da aposta.
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

- **DR de tamanho da caixa desde a FASE 1** (§6.7): variantes de entidade, escala uniforme
  (cubo), 14 a 26 cm, K = 8, atribuição aleatória por env fixa na run, massa igual em
  todas. A caixa vira mesh.

**Pendentes:**

1. **A rede vê o tamanho?** Recomendação: sim, +1 canal `meia_aresta` gateado (111 → 112).
   Ver a tabela da §6.7. Sem ele a política chuta a abertura das mãos.
2. **`rel_turning_envs`** — 0,10 é ponto de partida; ajustar com `error_vel_yaw`. E o
   `|wz|` mínimo: 0,2 rad/s de partida.
3. **Faixa e K do tamanho** — (0,07, 0,13) m e 8 são pontos de partida.
