# g1_limpo — especificação

**Branch:** `exp/g1-limpo` (base `exp/g1-poc` @ `3ff4847`)
**Data:** 2026-08-25
**Estado:** F0, F1 e F2 implementadas e verificadas (`smoke` 187 ok / 0 falhas, `paridade`
85 campos / 0 diferenças, `inspeciona --tabela` 0 falhas nos 5 elos e nos 7 níveis,
`leitura --demo` ok). O alvo do `REORIENTAR` foi redesenhado em 2026-08-26 (§4.3).
**Nenhuma run rodada ainda** — o portão de treino é remoto, e o venv local não roda
PPO (ver §11). F3 a F6 pendentes.

Um quarto módulo de treino, novo e isolado. `g1_multitask/` e `g1_poc/` ficam **intocados**,
os dois como referência.

---

## 1. Por que ele existe

Dois resultados comportamentais medidos, em módulos diferentes, e nenhum módulo tem os dois.

| módulo | conseguiu | não conseguiu |
|---|---|---|
| `g1_poc` | abriu `pegar` e `reorientar`; **reorientava a caixa bem e erguia ao alvo** | nunca andou; o robô fica imóvel |
| `g1_multitask` | **andou** e **resistiu a empurrão** (rastreava velocidade e não caía) | nunca girou; código que o dono chama de bloat |

O `g1_limpo` existe para preservar os dois e descartar o resto.

---

## 2. Restrições duras

**R1 — zero import de código do projeto.** Não consome `g1_training/`, nem `g1_multitask/`,
nem `g1_poc/`. Lê os três como referência humana e transcreve. O `mjlab` pode ser importado:
é framework.

Ganho colateral: o `experiment_name` compartilhado (`g1_lifting_box`, em
`g1_training/rl_cfg.py`) deixa de acoplar as runs.

Custo: transcrição à mão é a via mais provável de perder o andar. Ver §11.

**R2 — um objetivo só na manipulação.** Não cinco tarefas com cinco conjuntos de alvo.
Um objetivo: a caixa em algum lugar, com uma orientação, segurando ou solta.

⚠ Nem o `g1_poc` implementa isso hoje — ele tem quatro regras de alvo por elo, mais o alvo
de orientação, mais o twist. R2 é o alvo do desenho novo, não algo a copiar.

**R3 — incentivo, não penalização.** Toda ação precisa de um termo positivo e contínuo que
cresça a partir do repouso. Penalidade só limita **como** fazer o que já existe; ela não
ensina. Booleano é platô — preferir contínuo.

---

## 3. Fundação: molde `mjlab.tasks.velocity`

Decidido, e o argumento é um item só, não volume de código.

As três tabelas de σ por junta do `variable_posture` do G1 vivem **apenas** dentro de
`unitree_g1_rough_env_cfg` (`mjlab/tasks/velocity/config/g1/env_cfgs.py:107-146`). Não existe
constante exportada. Sob R1 elas podem ser **colhidas** do cfg do fabricante, porque `mjlab`
é biblioteca. Redigitá-las custa ~40 linhas de calibração — 14 padrões de junta × 3 regimes —
sem nenhum teste que pegue um dígito trocado. Um σ de joelho de 0,35 digitado 0,035 não
quebra nada: ele achata o passo, e a run morre 1200 iterações depois num painel.

Custo de chegar ao ponto de partida a partir do molde: **~39 linhas** (medido no
`g1_poc/env_cfg.py:86-124`).

### 3.1 O que o molde instala e precisa sair

| item | ação | por quê |
|---|---|---|
| `terrain_scan`, `height_scan`, `out_of_terrain_bounds`, currículo `terrain_levels` | já saem | `unitree_g1_flat_env_cfg` remove os quatro. É a receita flat, não desvio |
| `base_com` (`dr.body_com_offset`) | `pop` | corrompe memória em CPU **e** GPU; derruba a task do próprio fabricante |
| `randomize_terrain` (ramo de play) | `pop` | roda depois dos eventos de cena e mexe na origem do env; dessincroniza mobília de pose absoluta |
| `commands_vel` no play | `pop` | `CommandManager.__init__` faz `self.cfg = cfg` **sem** deepcopy, e `commands_vel` escreve em `cfg.ranges` — apaga velocidade pinada à mão |
| `reset_base` com yaw ±3,14 | bifurcar por modo | ±3,14 na locomoção (a mobília sobe 5 m, não há com que alinhar); ±0,2 na manipulação |
| física: `njmax=300`, `nconmax=None` | sobrescrever | insuficiente para manipulação. `njmax=800`, `nconmax=300`, `impratio=1.0`, `cone="pyramidal"` — `elliptic`/`impratio=10` divergia para NaN no reset parcial (cicatriz de 2026-07-15) |

---

## 4. Estrutura do treino

### 4.1 Um objetivo, alcançado por CADEIAS de no máximo 2 elos

O objetivo é um só: **a caixa onde ela deve estar, com a orientação pedida.** Mas ele é
alcançado por fases, e **as fases trocam DENTRO do episódio.** É isso que treina as
transições, e é o desenho sob o qual o `g1_poc` reorientou bem e ergueu ao alvo.

Transcrito de `g1_poc/comando.py:60,63`:

```
ELOS:     REORIENTAR   PEGAR   CARREGAR   BOTAR
CADEIAS:  (PEGAR)                      ← 1 elo
          (REORIENTAR → PEGAR)
          (PEGAR      → CARREGAR)
          (PEGAR      → BOTAR)
```

**Teto de 2 elos por cadeia.** A cadeia é sorteada no reset; os elos avançam dentro do
episódio, **sem reset e sem resample** (`_avanca_elo`, `comando.py:472`). A distribuição das
cadeias é uma tabela `[níveis × cadeias]` indexada pelo **nível** (`comando.py:126`) — é
assim que o currículo gradua *qual transição* o robô pratica, sem tarefa nova e sem grafo.

O `pegar` é o eixo de tudo: ele aparece em todas as cadeias, como 1º ou 2º elo.

**Anti-esquecimento por construção:** toda cadeia de 2 elos passa pelo elo anterior. Não se
esquece o `pegar` enquanto se treina o `botar`, porque não se chega ao `botar` sem pegar.
Isso vale mais que qualquer piso de amostragem, e não custa knob.

**Fechamento por elo** (`comando.py:435-438`):

| elo | fecha quando | sustain |
|---|---|---|
| `REORIENTAR` | perto & alinhado | `sustenta_outros_s` |
| `PEGAR` | perto & alinhado & **de pé** | `sustenta_pegar_s` |
| `CARREGAR` | perto | `carregar_s` (mínimo de tempo, `comando.py:455`) |
| `BOTAR` | perto & alinhado & **apoiada** | `sustenta_outros_s` |

**O que cada avanço de elo reescreve** (`comando.py:499-560`): o `PEGAR` pede a caixa na âncora do peito
(x,y relativos ao robô, z absoluto); o `CARREGAR` move a prateleira **+5 m** por escrita de mocap, e o chão fica livre para
andar; o `BOTAR` sorteia topo novo com teto efetivo no fundo da caixa menos a folga — sem
esse teto a laje nasceria **dentro** da caixa.

⚠ E a cada avanço de elo os σ de `bringing`, `reaching` e `ori` são recalculados **contra a
pose fresca** (`comando.py:556-560`). É isso que impede os níveis difíceis de virarem sorte:
com σ fixo de 0,40 rad, 90° dá 2,0e−7.

### 4.2 O one-hot — 5 slots, um por estado

| slot | estado | twist ativo? | caixa nas mãos? |
|---|---|---|---|
| 0 | `ANDAR` (locomoção pura, sem caixa) | **sim** | não |
| 1 | `REORIENTAR` | não | não |
| 2 | `PEGAR` | não | não → sim |
| 3 | `CARREGAR` | **sim** | **sim** |
| 4 | `BOTAR` | não | sim |

⚠ O one-hot é escrito **por passo, do elo corrente** — não uma vez no resample. Era a única
incompatibilidade real entre a máquina de elo e o one-hot do `g1_multitask`, e ela é de uma
linha.

**O ALVO DO `PEGAR` E O DO `CARREGAR` SÃO EXATAMENTE IGUAIS** (revisado 2026-08-25):

```
x, y   RELATIVOS ao robô   — a caixa está nas mãos, tem de acompanhá-lo
z      ABSOLUTO, 0,95 m    — agachar NÃO pode baixar o alvo
```

O `z` absoluto é o que obriga a erguer: um alvo relativo em z desceria com a pelve, e o robô
satisfaria agachando até a caixa. Os `0,95` são derivados — pelve do keyframe em `0,798`
(medido) mais a âncora do peito `+0,15`. E **não há jitter no alvo**: ±0,05 em y sobre
x = 0,25 desloca `atan(0,05/0,25) = 11°` fora do eixo, e o alvo aparece "de lado".

**O que impede o robô de ANDAR com a caixa no `pegar` é o COMANDO DE VELOCIDADE EM ZERO**, e
não a forma do alvo. Os elos `reorientar`, `pegar` e `botar` têm o twist forçado a zero (a
coluna "twist ativo?" da tabela acima); o `andar` e o `carregar` o têm ativo. É a configuração
do `g1_poc`, cuja manipulação funcionou.

⚠ Consequência, e ela DERRUBA uma decisão anterior desta spec: os dois `track_*` ficam
**ATIVOS em todos os elos**. Uma versão anterior mandava gateá-los em `ANDAR | CARREGAR`, para
um env de `PEGAR` não receber o kernel cheio por ficar imóvel. Com o twist em zero isso se
inverte — gatear os termos fora removeria a única coisa que recompensa ficar parado, e o twist
zerado não impediria nada.

⚠ Preço, MEDIDO (2026-08-26, robô travado — com ação zero ele desaba e a velocidade da queda
entra no erro, o que dá 2,14 e não é o piso): um robô imóvel num elo de manipulação colhe
**3,82/s** dos dois `track_*`, e o env inteiro colhe 5,82/s. Num elo que anda a mesma estátua
colhe 2,33/s de `track_*` e 4,20/s no total — o twist não é zero ali, portanto ficar parado
paga menos.

⚠ E a POSTURA não entra nesse piso, porque na F2 ela ficou **neutra** nos elos de manipulação.
Não é um 4º regime de σ, que era o desenho anterior: `exp(−média(erro²/σ²))` sobre 29 juntas é
um produto de 17 gaussianas quando os braços saem do default, e colapsa a zero para QUALQUER σ
— nem `std_running × 5` sobrevive a 40% da faixa de junta. O `std_standing` do G1 é uma entrada
só (`.*` = 0,05) e o `walking_threshold` dele é 0,05, logo com o twist em zero o regime
`standing` é **certo**, e o termo vale exatamente zero já a 10% da faixa, **com gradiente
zero**: canal morto, não penalidade forte.

Portanto o termo não tem o que dizer num elo de manipulação, e devolve **1,0** ali (neutro —
zero seria uma penalidade por sorteio de elo, e desalinharia a escala de retorno que o
controlador de fatia da F5 lê). Quem segura o robô de pé passa a ser o `upright` do fabricante
mais a condição de fechamento do elo, que exige "de pé". É R3 na forma mais limpa, e custa
zero knobs.

⚠ E a fatia de locomoção nunca é 1,00: com 1,00 os slots de manipulação do one-hot são
constantes em zero, e o normalizador do `rsl_rl` divide por `_std + 1e−2` **sem clamp**
(`rsl_rl/modules/normalization.py:48`) — ao acender, 1,0 entra na rede como **100,0**. Com
`fatia_loco = 0,95`, 5% dos episódios são de manipulação desde o passo 0. Os slots `CARREGAR` e
`BOTAR` seguem constantes até a F4, porque só existem como 2º elo de cadeia, e isso está
declarado com mitigação pré-registrada.

⚠ O sorteio de elo vive no **currículo**, e não num evento: a ordem de reset é currículo →
eventos → comando, e o reset de pose da base (evento) e o alvo (comando) os dois leem o elo. Contra o teto de ~12,5/s dos sete termos de tarefa isso é um **piso, e não um
concorrente** — mexer os braços não move a base. E é a configuração do `g1_poc`.

A razão de engenharia do one-hot passa a ser outra, e continua suficiente: ele diz à política
QUAL objetivo está ativo, e gateia os sete termos de tarefa — que sem gate pagariam o máximo
com os canais de caixa zerados, porque `exp(0) = 1`.

⚠ Isto **não** são 5 tarefas no sentido do `g1_multitask`. São 5 estados de **um** objetivo:
uma tabela de nível, um sorteador de cadeia, um alvo. Sem eixo por tarefa, sem grafo `PAIS`,
sem equalização de orçamento. R2 continua respeitada.

**O one-hot não leva o crédito do andar.** O `g1_poc` já tinha o equivalente funcional (o bit
`caixa_valida` em `command[:,9]`, `env_cfg.py:243-246`, mais o twist forçado a zero,
`comando.py:826`) e não andou.

### 4.3 O alvo do `REORIENTAR` — uma face marcada, no máximo um quarto de volta

Revisado 2026-08-26, e o desenho **inverteu**: a dificuldade não está no alvo, está na
orientação de nascimento da caixa.

**A face pedida é sempre a mesma, e ela é pintada.** `face_alvo_b = (−1, 0, 0)` no
referencial da caixa, com uma placa visual verde no geom `face_alvo`. O comando publica em
`FACE` a direção **desejada** (caixa→robô, projetada no horizontal) e em `ANG` o erro angular
corrente `arccos(normal_marcada · desejada)`.

Não se sorteia qual face. Sortear a face não ensinaria nada a mais: o cubo é simétrico, e o
que o robô precisa aprender é **girar**, não reconhecer qual lado é qual.

**O robô precisa aprender a girar no máximo 90°** (decisão do dono). A face não precisa ir
para o lado oposto. Qualquer uma das 6 faces chega à frente do robô com um quarto de volta em
Z, ou um quarto de volta em X/Y — e as duas coisas se fazem com a caixa **na laje**.

**A dificuldade gradua pelo nascimento** (`eventos.orientacao_de_nascimento`):

| nível | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `voltas_max` | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| `eixo_vertical` | não | não | não | não | **sim** | sim | sim |
| `desalinho_max_deg` | 15 | 20 | 20 | 20 | 20 | 20 | 20 |

As voltas são sorteadas uniformes em `0..voltas_max`, então **cada nível contém o anterior**.
O Z vem antes do Y por razão física: girar em Z é **pivotar** sobre a laje; girar em Y é
**tombar** um cubo de 20 cm numa laje de 4 cm.

⚠ Este eixo **satura no nível 4**. Acima dele só a altura e a carga graduam.

⚠ O erro tem **três** parcelas, e a terceira não é knob: `voltas × 90 + desalinho + azimute`.
A caixa nasce em `y ∈ ±0,18` com `x ≈ 0,32`, então a direção caixa→robô foge até
`atan(0,18/0,32) = 29°` do eixo. Sem essa parcela o teto medido acusa o nível 0.

**Diagnóstico de graça:** `erro == 90,0` exato significa caixa tombada. Quando a volta é em Y
a face marcada aponta para cima, e o ângulo entre um vetor vertical e *qualquer* direção
horizontal é exatamente 90°, seja qual for o azimute.

**A placa visual é a primeira divergência deliberada contra a referência**, e ela é
delimitada em vez de tolerada. `contype = 0`, `conaffinity = 0`, `density = 0` → massa **e
inércia bit-idênticas** à caixa sem ela. O `paridade.py` não compara geom a geom: ele compara
a física do corpo, compara o geom de colisão no índice 0, e **afirma** que o marcador é
inerte. Sem a placa a inspeção do `reorientar` seria cega — um cubo uniforme girado 90° é
visualmente idêntico ao original.

### 4.4 Duração de episódio — um corte só

`episode_length_s = 20,0` → 1000 passos a 50 Hz (`timestep 0,005 × decimation 4 = dt 0,02`).

⚠ `max_episode_length` é **escalar** (`mjlab/envs/manager_based_rl_env.py:281-283`): não
existe duração por env. E como um episódio de cadeia contém **dois** estados, **não existe
corte por modo** — o que gradua o tempo é o `_sust_alvo` por elo, não o teto do episódio.

O único corte por estado que sobra é o mínimo do `CARREGAR` (`carregar_s`), que é um piso de
tempo, não um teto.

---

## 5. Locomoção

### 5.1 A tabela

Derivada da comparação **medida** entre os dois módulos (cfgs montados e impressos em
2026-08-25). Os termos idênticos nos dois estão fora da bissecção e ficam como estão.

| termo | peso | origem | evidência |
|---|---:|---|---|
| `track_linear_velocity` | +2,0 | fabricante, idêntico nos dois | — |
| `track_angular_velocity` | +2,0 | fabricante, idêntico nos dois | — |
| `upright` | +1,0 | fabricante, idêntico nos dois | — |
| `pose` (`variable_posture`) | +1,0 | **fabricante / `g1_poc`** | **DECIDIDO** — §5.2 |
| `terminacao` | **−200** | **`g1_multitask` (andou)** | **DECIDIDO** — §5.3 |
| `action_rate_l2` | −0,1 | idêntico nos dois | plano, sem cronograma e sem fator por braço |
| `foot_clearance` | −2,0 | idêntico nos dois | — |
| `foot_swing_height` | −0,25 | idêntico nos dois | — |
| slip do pé | −0,1 | idêntico nos dois | — |
| `soft_landing` | −1e−5 | idêntico nos dois | — |
| `body_ang_vel` | −0,05 | idêntico nos dois | — |
| `angular_momentum` | −0,02 | idêntico nos dois | — |
| `self_collisions` | −1,0 | idêntico nos dois | — |
| `joint_acc` | −2,5e−7 | `g1_multitask` (andou) | desprezível; entra por paridade |
| `air_time` | **0,0** | **ausente no que andou, 0,0 no que não andou** | §5.4 |
| `joint_vel_hinge` | **sai** | só no `g1_poc` | penalidade que ENSINA; e a −1,0 consumiu 99,1% das penalidades num bloco |

**Aberto, para decidir:** `dof_pos_limits` (−0,5 no que andou, −1,0 no fabricante e no poc) e
**escala de ação** (0,8 no que andou, 1,0 no poc). Ver §12.

### 5.2 O termo postural — decidido, com risco declarado

`variable_posture` do fabricante, peso 1,0. Os σ do regime `walking` (joelho 0,35, hip_pitch
0,3, ankle_pitch 0,25) afrouxam exatamente as juntas que precisam se mover na passada. O
`posture` do `g1_multitask` é um puxão plano para o keyframe joelhos-flexionados, sem regime,
e pune amplitude de perna uniformemente.

⚠ **Risco aceito:** este é o item nº 1 da bissecção, e a escolha é o lado do módulo que
**não** andou.

**Mitigação obrigatória — 4º regime de σ, gateado pela TAREFA** (slot do one-hot), nunca por
limiar de demanda de velocidade. Sem ele, com o twist zerado na manipulação o robô cai no
regime `standing` (σ = 0,05) e paga **~0,93/s por terminar a tarefa** — com a caixa segurada
a 0,82 m o `pose` vale ~0,93 no regime de manipulação contra ~0,00 em `standing`, e
`exp(−27,6) ≈ 1e−12` é zero em float32, com gradiente zero.

⚠ **Não** gatear por limiar de demanda: foi penhasco medido (it 5217, demanda 1,60 contra
limiar 1,5, Δθ preso em 14,2° contra 20° de aceite).

### 5.3 `terminacao` — decidido

−200, que a dt = 0,02 com `scale_rewards_by_dt` vale **−4,0 por evento de queda**. É a config
do módulo que andou; o `g1_poc` não tinha.

⚠ Tensão declarada: por R3 uma penalidade que ensina "não cair" deveria sair. Ela fica porque
está na única config com marcha medida. Decisão do dono.

### 5.4 `air_time` fica DESLIGADO

Medido em 2026-08-25 montando os dois cfgs:

```
air_time    g1_multitask (ANDOU): AUSENTE      g1_poc (NÃO ANDOU): 0.0
```

`g1_training/base_env.py:141-145` apaga todo reward fora de `_BALANCE` e o `g1_multitask`
nunca o reinstala. **Os dois tinham o termo desligado, portanto ele não é o diferenciador.**

O `air_time` é o único termo **positivo** de marcha do mjlab e o único com derivada não-nula
no repouso, e por R3 ele seria o candidato natural. Mas ligá-lo é **desviar da única
configuração com marcha medida**. Ele entra como experimento declarado (§12), com A/B, nunca
como "o conserto".

### 5.5 Guinada: `ang_vel_z = (−0,5; 0,5)`

**Não** zerar. Com `wz_cmd ≡ 0` a estátua tem erro zero e colhe o kernel **cheio**.

| config | estátua | andador (ω_xy ≈ 0,3) | custo de andar |
|---|---:|---:|---:|
| `(−0,5; 0,5)` | 1,711/s | 1,671/s | **−0,041/s** |
| `(0, 0)` | **2,000/s** | 1,671/s | **−0,329/s** |

Zerar entrega o termo inteiro à estátua e torna o viés anti-marcha **8× maior**.

### 5.6 A conta da estátua, para registro

Medido por Monte Carlo (4 M amostras, `rel_standing_envs = 0,10`, kernels do fabricante):

| estágio do `commands_vel` | `track_lin` ×2 | `track_ang` ×2 | soma |
|---|---:|---:|---:|
| it 0 — x(−1;1) wz(±0,5) | 0,550 | 1,740 | **2,290/s** |
| it 5000 — x(−1,5;2,0) wz(±0,7) | 0,401 | 1,551 | **1,952/s** |
| it 10000 — x(−2,0;3,0) | 0,340 | 1,551 | **1,892/s** |

O currículo de comando reduz a colheita da estátua em **17% ao longo de 10 mil iterações**.
Não é conserto. Somando `upright` + `pose`, a estátua colhe ~3,7/s contra ~1,1/s de
penalidade dominante.

**Conclusão honesta:** ficar parado é ótimo **local**, não inescapável. O `g1_multitask`
escapou dele **sem nenhum termo positivo de marcha** — por exploração. Com σ ≈ 0,5 em 29
juntas, 4096 envs e 30 mil iterações, o ruído tropeça num passo, e daí os freios moldam e o
`track_lin` paga o deslocamento.

### 5.7 Push — requisito, não knob

Resistência a empurrão é comportamento medido no `g1_multitask` e é requisito.

⚠ E há um bug de medição a **não** repetir: no `g1_multitask` o `push_robot` **resetava o
gatilho de sucesso**. O contador só avança enquanto `_base & _condicao`, e `_base` inclui o
teste de erro angular; um push de ±0,78 rad/s a cada 1-3 s estoura o teste e zera o contador.
O `perf[locomover]` ficou 0 nas iterações 13.700 e 17.297 e só descolou na 22.296 — enquanto
o robô, segundo o dono, **já andava há algum tempo**. O conserto aplicado na época foi no
sintoma (`tol_w` 0,35 → 0,70).

**Regra que sai disto:** um gatilho de sucesso que uma perturbação **externa** pode zerar não
mede competência — mede ausência de perturbação. Se o push é requisito, ele não pode viver
dentro da integral do critério. Régua e perturbação em compartimentos separados.

---

## 6. Manipulação — os sete incentivos

Todos positivos e contínuos, o que é R3. É o desenho sob o qual o `g1_poc` reorientou bem e
ergueu ao alvo.

| termo | forma | peso | onde tem gradiente |
|---|---|---:|---|
| `staged` | `reaching × (1 + bringing)`, σ por env | +3,0 | distância palma→face e caixa→alvo |
| `precise_pos` | `exp(−‖caixa−alvo‖²/0,05²)` | +2,0 | posição da caixa |
| `precise_ori` | `reaching × exp(−Δθ²/σ²)`, σ por env | +1,0 | ângulo da face |
| `squeeze` | `tanh(min(F_esq,F_dir)/F_ref)` | +1,0 | **força de palma** |
| `unload` | `1 − F_apoio/(m·g)` | +2,0 | **força de apoio** — a ponte |
| `postura_ereta` | rampa dupla na pelve × preensão × descarga | +2,0 | altura da pelve |
| `sustentacao` | `t_na_condição / alvo` | +0,5 | **tempo na condição** |

Piso da estátua: **~4,0/s** dos dois `track_*` com o twist em zero (§4.2), e os sete termos
multiplicam pelo slot 1. **Esse gate é obrigatório:** com os canais de caixa zerados,
`exp(0) = 1` e "não existe caixa" pagaria o máximo.

Restam `upright` +1,0 e `pose` +1,0 no 4º regime como piso de sobrevivência legítimo.

**Fora:** `load` (soltar não é exigido — §4.2); `precise_pos` só entra porque o alvo é ponto
de mundo, e não a própria caixa. O `unload` já existe nos dois módulos com a mesma fórmula, e
ter convergido de forma independente é evidência a favor dele.

⚠ O `precise_ori` com σ **fixo** torna os níveis 4+ sorte: com σ = 0,40 rad, 90° dá 2,0e−7.
O σ por env é obrigatório, e ele exige resolver os sites das palmas **dentro** do termo de
comando, contra pose fresca — no reset o command manager roda depois dos eventos que
reposicionam a caixa.

---

## 7. Cena

| item | valor |
|---|---|
| caixa | `MjSpec` próprio, freejoint, geom box, meia-aresta (0,10; 0,10; 0,10), massa 1,0 kg default, `condim=3`, atrito (1,0; 0,02; 0,001) |
| prateleira | `MjSpec` próprio, body **sem** freejoint → mjlab auto-envolve em **mocap**: cinemático, posicionável por env, flutua em qualquer z, sem massa. Geom box (0,30; 0,30; **0,02**) |
| laje fina | deliberado. Paredão permite escorar; laje de 2 cm não |
| pads de palma | deletar as cápsulas `left_hand_collision` / `right_hand_collision` e acrescentar 4 pads. ⚠ API: `spec.delete(geom)`, nunca `geom.delete()` |
| sensores | palma e dorso e apoio precisam do campo **`force`**, não só `found` — sem isso o `squeeze` e o `unload` são impossíveis |
| locomoção | mobília sobe **+5 m**, e por isso o yaw de reset pode ser ±3,14 sem alinhar nada |

**Não** transcrever `get_table_spec` (meia-aresta 0,275, massa 20 kg, free-body): o próprio
docstring o marca legado e ele tem zero consumidores nos três módulos.

---

## 8. Currículo — três relógios, e a ordem de aprendizado

**Nada é "desbloqueado" no sentido de ligar e desligar.** Três mecanismos rodam ao mesmo
tempo e não se falam. O que muda é a **fatia** e a **dificuldade**, nunca a existência.

| | parte | o que move | o que a dispara |
|---|---|---|---|
| A | `command_vel` | alarga a caixa de comando de velocidade | passo global: it 0 / 5000×24 / 10000×24. Monotônica, sem realimentação |
| B | `forma` | a **fatia** locomoção × cadeia | `razao_marcha_ema ≥ 0,50` (§9) |
| C | `nivel` | a dificuldade do objetivo **e a distribuição das cadeias** | sucesso/fracasso por env, passeio ±1, `NIVEL_MAX = 6` |

### 8.0 A linha do tempo

```
it 0–200      alvo = 0,95    locomoção 95% / cadeia 5%    carência: o balanço nem começou
it ~200       o portão testa razao_marcha_ema >= 0,50
                 fechado -> fica em 0,95, indefinidamente
                 aberto  -> desce 0,02 a cada 12 iterações
it ~600       alvo = 0,30    locomoção 30% / cadeia 70%   regime
```

O relógio C roda o tempo todo, sobre as cadeias que existirem. A 5% de fatia isso são
~4.900 transições por iteração — não é zero, e é onde o nível 0 é praticado.

**A ordem de aprendizado das HABILIDADES não é uma escada de tarefas — é a tabela de
cadeias por nível** (`[níveis × cadeias]`, §4.1). O nível baixo concentra a probabilidade na
cadeia de 1 elo (`PEGAR`); o nível alto abre as cadeias de 2 elos, que são as **transições**.
Portanto o que o currículo gradua é *qual transição praticar*, e não *qual tarefa existe*.

**Ordem no dict é contrato e tem teste:** `command_vel` → `nivel` → `forma`. Invertida (bug
medido em 20/08), `p_up` caía de `p` para `0,7·p`, o ponto fixo do nível saía de 0,5 para
0,714, e um episódio de **locomoção** rebaixava o nível em 70% das vezes.

**Nível:** sobe com sucesso, desce sem, `clamp(0, 6)`. Só episódios de **cadeia** movem o
nível — um episódio de locomoção pura é ensaio. A propriedade que é o motivo da regra: o
nível se equilibra onde a taxa de sucesso é ~50% (ponto fixo do passeio), portanto **nenhum
limiar é escolhido à mão**.

### 8.1 Os três pisos de anti-esquecimento

São **três coisas diferentes**, e confundi-las foi o que quebrou o `g1_multitask`.

| piso | garante | valor |
|---|---|---|
| **de fatia** | nenhum dos dois lados vai a zero | `alvo_loco_max = 0,95` (a cadeia nunca fica em 0%) e `alvo_loco_min = 0,30` (a locomoção nunca cai abaixo de 30%) |
| **de nível** | o nível fácil não desaparece do treino | fração dos envs sorteada **uniformemente** sobre os níveis abertos |
| **de elo** | o elo anterior nunca é esquecido | **estrutural, sem knob**: toda cadeia de 2 elos passa pelo 1º. Não se chega ao `BOTAR` sem `PEGAR` |

⚠ O piso de nível **não** é o `rho = 0,30` do `g1_multitask`. Aquele era um piso uniforme
sobre **tarefas**, e foi ele que tornou o alvo de fatia inalcançável: com 5 tarefas o piso
ocupava 0,75 do sorteio e o teto da locomoção ficava em 0,55, contra os 0,945 que a fatia de
30% exigia. **Piso de nível e piso de fatia são eixos ortogonais** — o de nível não toca a
divisão locomoção × cadeia.

⚠ O rebaixamento do passeio ±1 espalha os envs, mas é **distribuição, não garantia**: se a
política ficar boa, os envs empilham no topo e o nível 0 sai do treino. O piso é seguro
barato. (Correção: uma versão anterior desta spec afirmava que o rebaixamento dispensava o
piso.)

**Tabela de células — só o piso desce, o teto é fixo:**

| nível | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `topo_min` (m) | 0,55 | 0,45 | 0,30 | 0,15 | 0,04 | 0,04 | 0,04 |
| `carga_max` (kg) | 1,0 | 2,0 | 3,0 | 4,0 | 5,0 | 5,0 | 5,0 |
| `jitter_x_max` (m) | 0,20 | 0,20 | 0,20 | 0,15 | 0,08 | 0,08 | 0,08 |
| `voltas_max` | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| `eixo_vertical` | não | não | não | não | sim | sim | sim |
| `desalinho_max_deg` | 15 | 20 | 20 | 20 | 20 | 20 | 20 |

Topo sorteado em `U(topo_min; 0,55)`, carga em `1,0 + (teto−1,0)·U(0,1)`, voltas uniformes em
`0..voltas_max`: **cada nível contém o anterior**. O `ang_max_deg` de uma versão anterior foi
**deletado** — ele fazia o `reorientar` nascer satisfeito em 3 dos 7 níveis (§4.3). A tabela discreta do `g1_multitask` tem dois defeitos — no nível 6 a laje fica
enterrada (centro em −0,02 m), e a altura fácil desaparece do treino no instante da promoção.

**Nada do currículo vai para o checkpoint.** Depois de um resume o balanço recomeça no piso e
recalibra. É o comportamento seguro (um freio recomeça solto, não apertado) e é declarado.

---

## 9. A fatia de transições — o mecanismo central

Esta é a explicação causal do "andou / não andou", e ela **valida a hipótese original do
dono**: o problema não é dividir, é treinar as duas ao mesmo tempo, cedo demais.

O `g1_multitask` é sequencial por construção:

```
g1_multitask/curriculum.py:118   self.abertas = [T.LOCOMOVER]
g1_multitask/curriculum.py:389   if perf[cel][topo] < self.limiar: continue
g1_multitask/knobs.py:756        limiar_competencia = 0.90
```

Só `LOCOMOVER` nasce aberta, e nenhuma outra tarefa abre antes de `perf[locomover] ≥ 0,90`.
A locomoção recebeu **100% dos dados** por 22.296 iterações. O `g1_poc` fez o oposto: o
portão media **sobrevivência**, abriu com o robô imóvel, e entregou 70% das transições à
manipulação por volta da iteração 420.

### 9.1 A aritmética

`frac_locomocao` **não** é probabilidade de sorteio. É a fatia de **transições** alvo, e o
sorteio é resolvido das durações medidas:

```
f = alvo·Tm / (Tl·(1−alvo) + alvo·Tm)        Tl, Tm = EMAs, ema = 0,99
```

O sorteio é por **episódio** e o PPO aprende por **transição**:

| Tl (loco) | Tm (manip) | sorteio 0,30 entrega | para entregar 0,30 precisa sortear |
|---:|---:|---:|---:|
| 24 | 961 | **1,06%** ← a armadilha medida | 0,9449 |
| 150 | 500 | 11,4% | 0,5882 |
| 400 | 500 | 25,5% | 0,3488 |
| 1000 | 500 | 46,2% | 0,1765 |

Clamps do sorteio: `[0,10 ; 0,95]`.

### 9.2 O portão e o balanço

| parâmetro | valor | razão |
|---|---|---|
| **portão** | `razao_marcha_ema ≥ 0,50` | um sinal, adimensional, do fabricante (`terrain_levels_vel` rebaixa quem anda menos de metade da comandada). Imune ao alargamento das faixas na it 5000 |
| `alvo_loco_max` (piso inicial) | **0,95**, não 1,00 | com 1,00 o slot `MANIPULAR` é constante zero por 200+ iterações, e um canal constante dá `_std = 0` no normalizador: ao acender, 1,0 entra como 100,0. Com 0,95, 5% dos episódios são de manipulação desde o passo 0 e nenhum slot é constante |
| `alvo_loco_min` | 0,30 | destino |
| `alvo_passo` / `iters_entre_degraus` | 0,02 / 12 | 33 degraus ⇒ ≥ 396 iterações na rampa |
| carência | 200 iterações | contada de quando o **balanço começou**, nunca de passo global absoluto |
| histerese | sobe fatia se sinal < 0,80 × limiar | assimétrico: lento para avançar, rápido para defender |

**Estado inicial das EMAs, deliberadamente assimétrico:** durações nascem **neutras**
(episódio cheio), porque governam a fatia e um erro ali só desafina o sorteio por ~τ;
`razao_marcha` nasce **pessimista em 0,0**, porque governa o **portão**, e um portão que
nasce aprovando entrega a locomoção antes de existir marcha — foi exatamente o que a
`dur_loco_ema` neutra em 1000 passos fez.

⚠ **Um sinal só no portão.** Dois sinais conjuntivos já travaram uma rampa para sempre: o
`erro_giro_ema ≤ 0,30` ficou plano em 0,587 por 390 iterações enquanto a `razao_giro` marcava
0,373.

### 9.3 Fatia esperada por fase

`4096 envs × 24 passos = 98.304 transições/iteração`.

| fase | alvo | Tl / Tm | sorteio | transições LOCO | MANIP |
|---|---|---|---|---:|---:|
| 1 — locomoção quase pura | 0,95 | 1000 / 500 | 0,95 | ~93.400 | ~4.900 |
| 2 — rampa | 0,60 | 150 / 500 | 0,833 | ~59.000 | ~39.300 |
| 3 — regime | 0,30 | 1000 / 500 | 0,176 | ~29.500 | ~68.800 |

---

## 10. Leitura

⚠ `Episode_Reward/<termo>` do rsl_rl é a soma do episódio dividida por
`max_episode_length_s`. Com episódios de 2,05 s num teto de 20 s, todo valor sai dividido por
0,1027: `action_rate_l2 = −0,34` no painel é **−3,31/s** de verdade. Ler o painel sem
desfazer isso foi o que deixou dois freios consumirem 55% do sinal positivo por 5000
iterações sem ninguém ver.

⚠ **Medição nunca mora dentro de penalidade.** `reward_manager.py:122` **pula** termo com
peso 0, portanto desligar uma penalidade apaga o log em silêncio. Cinco termos do fabricante
escrevem `Metrics/*` de dentro da função — eles viram `MetricsTermCfg`.

⚠ **Nada é gateado em `peak_height_mean`:** é média global sobre pousos, dominada pelos envs
de manipulação com o pé plantado, e **infla** com queda porque `feet_swing_height` não tem
`reset()` (bug do mjlab 1.5.1 — 3 linhas de subclasse consertam).

### Escada de corte

| iteração | chave | ≥ | alvo | falhar significa |
|---|---|---|---|---|
| 200 | `Policy/mean_noise_std` | ≥ | 0,85 | as penalidades dominam; algum termo ficou pesado |
| 1000 | `forma/dur_loco_ema` | ≥ | 150 | o robô de locomoção não sobrevive (sobrevivência, não marcha) |
| 2000 | `forma/razao_marcha_ema` | ≥ | 0,50 | **o portão.** Não fecha metade da velocidade comandada, e o balanço nunca libera fatia |
| 3000 | `sucesso_manipulacao` | ≥ | 0,30 | os consertos machucaram a manipulação |

⚠ O log reporta a fatia de **transições**, não o sorteio. As chaves `Metrics/*` do comando
saem diluídas pela **composição dos resets**, não pela população: num bloco medido, 8,89% dos
resets eram de manipulação contra 70,1% da população, porque o episódio é 24× mais longo.
Ler `episode_success = 0,033` como "3%" foi erro — o valor condicionado era 37%.

---

## 11. Paridade — como não perder o andar na transcrição

Sob R1 cada valor é transcrito à mão, e um número errado **não levanta erro**: ele muda o
comportamento em silêncio.

O guarda-corpo que já existe no `g1_poc` (`smoke.py:766-828`) cobre pouco: `.weight` de 14
termos, o dict da escala de ação, 6 escalares e 3 faixas do twist, e conjuntos de **nomes**.
Ele **não** cobre — e cada omissão é uma via de perder o andar:

- `params` de nenhum termo: o `std` do `track_linear_velocity`, os três dicts de σ do `pose`,
  o `target_height` do `foot_clearance`. **Peso certo com σ errado passa.**
- `func` dos termos: reimplementação com o mesmo peso e outro kernel passa.
- `params` de terminação e de evento: `fell_over.limit_angle`, `push_robot.velocity_range`.
- o grupo de **observação** inteiro: nomes, ordem, escala, ruído.
- **toda a cena.** Massa, atrito, meia-aresta, altura da laje, grupo de geom, os pads e os
  `fields` dos sensores vivem dentro de lambdas de `spec_fn`, e comparação de cfg não penetra.

E há um limite estrutural: paridade contra o **fabricante** não pode pegar erro na metade de
**manipulação**, porque o fabricante não tem caixa nem mesa.

**O verificador proposto:** um teste — e só o teste — importa os módulos de referência,
compila os dois `MjSpec` e diferencia o `mjModel`: `body_mass`, `geom_size`, `geom_friction`,
`geom_condim`, `geom_group`, `site_pos`, e os `fields`/`adr` de cada sensor. É diagnóstico
descartável, não dependência do treino. Se isso for recusado, a paridade fica por conferência
manual dupla, e o risco fica declarado.

⚠ **O venv local não roda PPO, e isso não é defeito deste módulo** (medido 2026-08-26). Duas
causas independentes: `mjlab/scripts/train.py` chama `select_gpus`, que estoura sem placa; e o
`mjlab 1.5.1` local manda `cnn_cfg`/`rnn_type`/`rnn_num_layers` ao `MLPModel` do
`rsl-rl-lib 5.4.0`, que não os aceita — o que derruba a task do PRÓPRIO fabricante. O
ambiente de treino instala `mjlab==1.5.3` com o mesmo `rsl-rl-lib` e roda. Consequência para
a paridade: **ela vale onde roda.** O `smoke` e o `paridade` têm de rodar no ambiente remoto,
na primeira célula, antes do treino — rodá-los só aqui prova a versão errada do `mjlab`.

⚠ **Divergência deliberada, e como ela se declara.** A nossa caixa tem um geom a mais que a
referência: a placa da face alvo (§4.3). Quando isso aconteceu o `paridade.py` **não** ganhou
tolerância — ele trocou de eixo: compara a física do corpo (massa e inércia, bit-idênticas),
compara o geom de **colisão** no índice 0, e afirma `contype = conaffinity = 0` no marcador. É
o padrão para toda divergência futura: delimitada e com o limite afirmado por teste, nunca
tolerada por `atol`.

---

## 12. O que só uma run decide

| item | opções | nota |
|---|---|---|
| `dof_pos_limits` | −0,5 (andou) contra −1,0 (fabricante e poc) | a paridade dirá "idêntico" sem poder dizer se o andar dependia do desvio |
| **escala de ação** | 0,8 (andou) contra 1,0 (poc) | vereditos **opostos** documentados: o commit `3fa588a` chama o 0,8 de *"o desvio que eu levei mais tempo para achar, e possivelmente o mais caro"*. Não se herdam os dois — exige escolha declarada ou A/B |
| `air_time` | 0,0 (as duas referências) contra ligar | §5.4. Se ligar, é experimento com A/B e régua `Metrics/air_time_mean` |
| orçamento do bloco 1 | 3000 / 5000 / 10000+ iterações | abaixo de 5000 o `commands_vel` do fabricante nunca endurece; a rampa do balanço já consome ~600 |

**Não estabelecido, e declarado:** que a marcha apareça. O smoke não mede convergência. E
**ninguém neste projeto rodou a tarefa do fabricante sem modificação** — portanto não se sabe
se ela anda nesta máquina, e o experimento de controle segue não executado.

---

## 13. Fases, cada uma com portão antes da seguinte

| fase | entrega | portão |
|---|---|---|
| F0 | esqueleto: cfg registrado, cena, ação, observação, smoke | smoke verde; `mjModel` diferenciado contra a referência (§11) |
| F1 | locomoção pura, sem caixa, sem one-hot ativo | `dur_loco_ema ≥ 150` na it 1000; `razao_marcha` medida e logada |
| F2 | one-hot de 5 slots escrito por passo, twist zerado nos elos parados, 4º regime da postura | smoke: o twist é **0** num env de `PEGAR` e ativo num de `CARREGAR`; o slot muda no meio do episódio sem reset |
| F3 | os sete incentivos, cena com caixa e mesa, cadeia de 1 elo (`PEGAR`) só | `sucesso_manipulacao ≥ 0,30` na cadeia curta |
| F3b | máquina de elo: as 3 cadeias de 2 elos, `_avanca_elo`, σ recalculado por elo | smoke: o elo avança sem reset; a prateleira sobe +5 m no `CARREGAR`; o topo do `BOTAR` nunca nasce dentro da caixa |
| F4 | balanço de forma: portão `razao_marcha ≥ 0,50`, rampa 0,95 → 0,30 | fatia de transições logada e batendo com a tabela de §9.3 |
| F5 | currículo de nível, tabela de células | níveis sobem e descem; ponto fixo perto de 50% |

---

## Apêndice — o que fica de fora, e por quê

| mecanismo | origem | por que sai |
|---|---|---|
| `_equaliza_orcamento` (4,0/passo) | `g1_multitask` | consequência de R2: não há 5 orçamentos a igualar |
| grafo `PAIS` + `abertos` no checkpoint | `g1_multitask` | consequência de R2 |
| `NIVEIS_ATIVOS` | `g1_multitask` | consequência de R2 |
| `_dist_tarefas` inversa à competência | `g1_multitask` | consequência de R2; a distribuição vira locomoção × manipulação |
| `tol_v`/`tol_w` na régua de sucesso | `g1_multitask` | são dois escalares **globais**, não por tarefa; e o push resetava o gatilho (§5.7). Régua e perturbação separadas |
| `LOCOMOVER_CARREGANDO` como **tarefa** | `g1_multitask` | não sai o comportamento, sai a forma: ele entra como o **elo `CARREGAR`** (§4.1), dentro de uma cadeia, não como 5ª tarefa com eixo próprio |
| `joint_vel_hinge` / `peso_por_competencia` | `g1_poc` | penalidade que ensina; e a rampa tinha um cliente só |
| `load` | `g1_poc` | soltar não é exigido (§4.2) |
| `air_time` ligado | proposta | §5.4 — desviaria da única config com marcha medida |
| `expande_checkpoint` | `g1_poc` | módulo novo, sem checkpoint legado a converter |
