# Contrato de troca de tarefa — g1_limpo

**Estado:** APROVADO COM EMENDAS em 2026-09-02 (§13). Nada implementado. Próximo passo: plano de implementação na branch `exp/g1-limpo-v2`.
**Escrito:** 2026-09-02 · **Revisado:** 2026-09-02 (v5 — decisões do dono: branch v2, girar no lugar entra, andar até a mesa não é deste modelo, viewer intocado)
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

**Regra deste trabalho: nada acima é tocado.** Locomoção, recompensas e penalidades, as
quatro cadeias, o piso de 30% de locomoção, o currículo de forma e de nível, as
terminações, o rastreio por elo e a multa de mesa ficam como estão. **Decisão do dono
(02/09), explícita.**

O que este documento muda está **inteiro** na §6, e é observação e um canal de comando.
Nenhuma recompensa, nenhuma cadeia, nenhum currículo.

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

### 3.1 As situações que precisam existir

- robô parado recebe o comando para pegar
- robô anda com a caixa (andar e parar com ela na mão)
- robô com a caixa na mão, parado, bota em algum lugar

### 3.2 As transições que PRECISAM ser aprendidas

```
andar v=0  ->  pegar                 TREINADA  (a espera publica ANDAR, §6.3)
pegar      ->  andar com a caixa     TREINADA  (cadeia 2, já existe)
pegar      ->  botar                 TREINADA  (cadeia 3, já existe)
carregar   ->  botar                 aposta    (§7.2)
botar      ->  andar, sem caixa      aposta    (§7.2)
```

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
| 3 | `CARREGAR` | **ativo** | vivos | ancorado na base, altura do peito | andar com a caixa |
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
  elo      [ 0  0  0  1  0 ]            ainda CARREGAR — treinado: ~10% dos envs são standing
  caixa    vivos, como acima

FASE 6 — BOTAR                          <- o botão
  twist    ( 0,00   0,00   0,00 )      camada FORÇA zero (regra 2)
  elo      [ 0  0  0  0  1 ]            OPERADOR
  caixa_b  ( 0,24   0,01  +0,12 )      PERCEPÇÃO
  alvo_b   ( 0,35   0,00  -0,20 )      EXTERNO — o único alvo enviado: onde botar, no frame da base
  ANG        0,05 rad                   A BORDO: face pedida para o botar
  fim: caixa apoiada — no sim, força de apoio ≥ fração do peso; em campo, o operador vê

FASE 7 — ANDAR (esquecer a caixa)
  twist    ( 0,50   0,00   0,00 )      PILOTO
  elo      [ 1  0  0  0  0 ]            OPERADOR
  caixa    tudo 0                       <- a percepção AINDA vê a caixa na laje; a camada zera (regra 1)
```

⚠ **A rede NÃO decide quando trocar.** Quem aperta o botão é o operador, ou o algoritmo
que vier a ser escrito. Os instantes que importam: `PEGAR → CARREGAR` quando a caixa está
erguida e segura; `CARREGAR → BOTAR` depois de parar no destino; `BOTAR → ANDAR` com a
caixa apoiada. No sim esses instantes são as condições de sustain das cadeias; em campo
são olho ou sensor. Um algoritmo de troca automática é trabalho separado (§10, "o robô
decidir sozinho").

**Quais dessas trocas o treino produz** está na §7: as fases 2→3, 3→4 e 3→6 são
treinadas; 4→6 (via 5) e 6→7 são a aposta da §7.2.

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

Três mudanças, todas em **observação e canal de comando**. Nenhuma em recompensa,
cadeia ou currículo.

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
Decisão do dono (02/09): é esta segunda coisa. Faixa: manter (0,3, 1,0) na primeira run
da v2, para que a única diferença nos episódios de manipulação contra a `bloco7` seja o
one-hot — atribuição limpa. Alongar depois, se a transição precisar de mais exposição.
Sorteada, para a política não contar passos.

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

---

## 7. O QUE É TREINADO, E O QUE É APOSTA

### 7.1 Treinado

| transição | onde | o que a política vê no pulo |
|---|---|---|
| `ANDAR(v=0) → PEGAR` | **todo** episódio de manipulação com espera > 0 (§6.3) | one-hot vira; canais de caixa acendem; twist segue zero |
| `PEGAR → CARREGAR` | cadeia 2 | one-hot vira; twist acende; caixa segue viva |
| `PEGAR → BOTAR` | cadeia 3 | one-hot vira; `alvo_b` muda para o topo novo |

⚠ A primeira era a **aposta** da v1 deste documento. Deixa de ser: a espera a treina a
partir da postura real de parado, e o resíduo "configuração de juntas no instante da
troca" deixa de existir. (Decisão do dono, 02/09: em `v = 0` a pose é praticamente a
default — e agora isso nem precisa ser assumido.)

### 7.2 Aposta — o que resta dela

Duas transições do cenário **não** são treinadas:

| transição | por que é aposta razoável |
|---|---|
| `CARREGAR → BOTAR` | o estado de chegada — caixa na mão, robô parado, `alvo_b` num topo — é o que a cadeia 3 produz na abertura do `BOTAR`. Só o one-hot de partida difere. |
| `BOTAR → ANDAR` | com o gate, o estado de chegada é locomoção pura: canais zero, twist ativo. É 30% do treino. |

**Por que a aposta é razoável: a política é SEM MEMÓRIA.** Medido: `history_length =
None` nos dois grupos, e nenhum termo com histórico. O ator é um MLP. O que a política vê
no instante da troca é o corpo **agora**, e não a história. Se houvesse recorrência, o
buffer carregaria a tarefa anterior e nenhum gate consertaria.

**Se a aposta falhar, o conserto é conhecido:** uma cadeia de 3 elos
`(PEGAR, CARREGAR, BOTAR)` com `_TETO_ELOS = 3`, e um 3º elo `ANDAR` na cadeia 3.
Decidir com o painel, não agora (§13).

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

### 8.1 `PEGAR → CARREGAR → BOTAR` não existe como sequência

`_TETO_ELOS = 2`: toda cadeia tem no máximo 2 elos. O cenário §3 pede pegar → andar com
a caixa → botar, que são 3. Hoje treina-se `PEGAR→CARREGAR` e `PEGAR→BOTAR` separados,
e a perna `CARREGAR → BOTAR` é a aposta da §7.2. **Custo se não bastar:** uma cadeia de
3 elos, `_TETO_ELOS = 3`.

### 8.2 `BOTAR → ANDAR` não é treinado

Nenhuma cadeia volta ao `ANDAR`. Aposta da §7.2, de baixo risco. **Custo se não bastar:**
um 3º elo `ANDAR` na cadeia 3.

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

**O conserto é um ramo no sorteador, e não toca recompensa nenhuma:** um
`rel_turning_envs` análogo ao `rel_standing_envs` — fração de envs com `lin = 0` e
`ang_z` sorteado na faixa. O `track_angular_velocity` já paga por rastreá-lo. O termo
`TwistComRazaoDeMarcha` já é subclasse do fabricante; o ramo entra lá.

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

**Tamanho da mudança de código, e ele é pequeno:** o gate (duas linhas em
`observacoes.py`), o `VALIDA` fora da observação (uma linha), o publicado em `ANDAR` na
espera (uma condição em `_aplica_espera`), e as travas da §11. **Nenhuma mudança em
recompensa, cadeia, currículo ou reset.** O item §9 é separado e opcional.

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

**Pendentes:**

1. **Cadeia de 3 elos.** Os elos seriam `(PEGAR, CARREGAR, BOTAR)` — pegar, andar com
   ela, botar: o cenário de campo inteiro menos a aproximação. E para "esquecer a caixa",
   um 3º elo `ANDAR` na cadeia 3: `(PEGAR, BOTAR, ANDAR)`. ⚠ Este segundo NÃO é de graça:
   o ramo `ANDAR` do `_aplica_elo` manda a laje a +5 m em z — com a caixa recém-apoiada e
   as mãos sobre ela, a laje subiria através das mãos. Precisaria de um `ANDAR` que não
   guarda a mobília. **Recomendação:** nenhum dos dois na primeira run da v2; a aposta da
   §7.2 os cobre, e eles tocam a máquina de cadeia, que funciona. Medir primeiro.
2. **`rel_turning_envs`** — 0,10 é ponto de partida; ajustar com `error_vel_yaw`.
