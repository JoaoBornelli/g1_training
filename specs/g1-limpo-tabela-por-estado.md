# g1_limpo — a tabela por estado, e a dobradiça de velocidade

**Estado:** a implementar · **Ramo:** `exp/g1-limpo-v2` · **Base:** `fe50a1e`

## 0. O princípio, em uma frase

**O teto do que o robô ainda tem de fazer ≥ o piso do que ele já fez.**

Hoje é o contrário. Medido em `model_10200` (cadeia C, 192 000 amostras; cadeia B,
25 852):

| estado | piso (renda congelada) | teto do que falta fazer | razão |
|---|---|---|---|
| BOTAR | 13,79 | ~7 | 2 : 1 |
| cauda C (levantar) | 21,76 | 2 (`postura_ereta`) + ~6 | 3 : 1 |
| CARREGAR (andar) | 13,79 + 13,20 de "segurar" | 4 (rastreio) | 7 : 1 |

Consequências medidas: ele **não anda com a caixa** (pernas com `|qd|` 15× menor,
97% dos passos com os dois pés no chão), **não levanta** depois de botar (pelve p10 fica
em 0,469 m; a mediana leva 8 s), e **corre na pega** (fechar 1 s antes compra ~14 de
renda; a cobrança de velocidade saturava num clamp).

A `renda_congelada` **FICA**, cumulativa, como está. Ela é a metade que garante o
arranque: fechar é seguro antes de o elo seguinte estar aprendido. Sem ela, com γ = 0,99
e um BOTAR que rende 20% do teto no começo, fechar o PEGAR valeria `0,6 × 0,2 × teto` —
punido — e foi isso que travou o BOTAR por cinco blocos. A tabela é a outra metade.

## 1. O estado de recompensa

Um inteiro por env, publicado pelo termo de comando em `env.limpo_estado`. Dez estados,
que são a enumeração completa do que ocorre:

| # | nome | condição |
|---|---|---|
| 0 | `ANDAR` | `_elo == ANDAR` |
| 1 | `ESPERA_SEM` | `aguardando ∧ ¬pegou ∧ ¬soltou` |
| 2 | `ESPERA_COM` | `aguardando ∧ pegou ∧ ¬soltou` |
| 3 | `REORIENTAR_SEM` | `_elo == REORIENTAR ∧ ¬aguardando ∧ ¬pegou` |
| 4 | `REORIENTAR_COM` | `_elo == REORIENTAR ∧ ¬aguardando ∧ pegou` |
| 5 | `PEGAR_SEM` | `_elo == PEGAR ∧ ¬aguardando ∧ ¬pegou` |
| 6 | `PEGAR_COM` | `_elo == PEGAR ∧ ¬aguardando ∧ pegou` |
| 7 | `CARREGAR` | `_elo == CARREGAR` |
| 8 | `BOTAR` | `_elo == BOTAR ∧ ¬aguardando ∧ ¬soltou` |
| 9 | `CAUDA` | `soltou` |

Precedência: `soltou` primeiro (→ 9); depois `aguardando` (→ 1 ou 2 por `pegou`);
depois por `_elo`. A divisão `_SEM/_COM` existe porque o rastreio e o `pose` dependem de
`pegou` — ela reproduz o `engajado` do `rastreio_por_elo` de hoje, estado por estado.

**Onde calcular:** em `comando._aplica_espera`, logo depois da linha que escreve
`self._command[:, VALIDA]` (`comando.py:~692`), a partir do MESMO `aguardando`. Assim o
estado tem a MESMA fase temporal do `VALIDA` que os termos leem hoje, e nada muda de
timing. `_pegou` e `_soltou` já estão frescos ali (`_publica_pegou` roda antes). `_elo`
é o do passo anterior — como o `VALIDA` de hoje. ⚠ Não calcule no fim de
`_update_command`: ali `_avanca_elo` já correu e a espera apareceria um passo mais
cedo que o `VALIDA`, e o `renda_congelada` congela pela subida de `_fechos` com a
soma do passo ANTERIOR — a fase importa.

Os nomes e a ordem vivem em `comando.py` como constante (`ESTADOS`), ao lado de
`ANDAR, REORIENTAR, ...`, e o `knobs` os importa para rotular as colunas.

## 2. A tabela (`knobs.py`)

Nova `@dataclass PesoPorEstado`, um campo por termo, cada um uma tupla de 10 floats na
ordem de `ESTADOS`. Ela entra em `Knobs` ao lado de `tarefa`.

```
                 ANDAR  ESP_SEM ESP_COM REOR_SEM REOR_COM PEG_SEM PEG_COM CARREGAR BOTAR CAUDA
staged             0      0       0       1        1        1       1        0       2     0
precise_pos        0      0       0       1        1        1       1        1       2     0
precise_ori        0      0       0       1        1        1       1        0       2     0
squeeze            0      0       0       1        1        1       1        0       2     0
unload             0      0       0       1        1        1       1        0       2     0
postura_ereta      0      0       0       1        1        1       1        0       2     8
load               0      0       0       1        1        1       1        0       2     0
track_linear_vel   1      0       1       0        1        0       1        3.5     1     1
track_angular_vel  1      0       1       0        1        0       1        3.5     1     1
pose               1      1       4       1        1        1       4        1       1     8
```

### Por que cada número

**Os sete de manipulação, colunas 0–6 e 8–9:** são o `VALIDA` de hoje, escrito por
extenso. `ANDAR`, as duas esperas e a `CAUDA` valem 0 — é exatamente onde `VALIDA = 0`
hoje. `REORIENTAR`, `PEGAR` valem 1 — inalterado.

**BOTAR = 2.** O BOTAR fecha hoje a 7,92/s (p50, 135 fechos) contra um piso de 13,79.
×2 leva o fecho a ~15,8 ≥ 13,79. Presente ≥ passado.

**CARREGAR: só `precise_pos` fica (= 1), os outros seis vão a 0.** Os seis pagam
13,20/s por a caixa estar no peito — atingido no instante da pega, e satisfeito parado.
É o piso da estátua com a caixa na mão, e a regra do projeto já diz: todo termo que paga
por estar parado tem de ser gateado na tarefa. `precise_pos` fica para a caixa não
descer do peito (18 cm de raio; teto 3,0). Medido: `unload ≡ 1`, `load ≡ 0` e
`postura_ereta` saturada ali — três dos seis são constantes estruturais, apagá-los custa
zero gradiente.

**Rastreio em CARREGAR = 3,5.** Teto dos dois rastreios: 2,0 + 2,0 = 4,0. ×3,5 = 14 ≈
o piso de 13,79. Conta: parado = 13,79 + 3 = 16,8; andando bem = 13,79 + 3 + 14 = 30,8;
ganho de andar **+14**, break-even de risco **45%** (hoje +1,5 e 5%).

**Rastreio nas outras colunas:** reproduz `rastreio_por_elo`, `fator = 1 − zerado ×
(1 − pegou)`, estado a estado: `ANDAR` 1 (twist vivo); `ESPERA_SEM` 0; `ESPERA_COM` 1;
`REORIENTAR_SEM`, `PEGAR_SEM` 0 (twist zerado, nunca tocou — estátua); `REORIENTAR_COM`,
`PEGAR_COM`, `BOTAR`, `CAUDA` 1 (twist zerado, já tocou — segurar/parar É a tarefa).

**`postura_ereta` na CAUDA = 8 e `pose` na CAUDA = 8.** Piso depois do BOTAR ×2:
13,79 + 15,8 = 29,6. Teto da cauda: `postura_ereta` 2k + `pose` k + rastreio 4 +
`upright` 1 = 3k + 5. Para 3k + 5 ≈ 29,6, **k = 8**. Só estes dois escalam porque só
eles separam "agachado com as mãos na caixa" de "de pé na pose default"; rastreio e
`upright` são satisfeitos agachado. Break-even de risco: 24/(29,6+24) = 45%.

**`pose` em `PEGAR_COM` e `ESPERA_COM` = 4.** Enquanto segura, ombro e cotovelo estão
FORA do `pose` (máscara do `PosturaPorElo`), portanto ×4 atinge punho, perna e cintura
— o punho torcido e a perna solta. Com std 1,00 o punho a 1,6 rad custa 0,52/s de um
teto de 1,0, isto é 4,5% da renda de 11,38 — ele não se importa. ×4 leva a 2,1/s.
⚠ NÃO em `PEGAR_SEM` (o braço ainda está na média e ×4 brigaria com o alcance), NÃO em
`BOTAR` (as pernas agacham para a laje a 0,30 m e ×4 brigaria com o agachamento), NÃO
em `CARREGAR` (é marcha; `std_walking`).

## 3. O wrapper (`recompensas.py`)

Uma classe `PesoPorEstado(cfg, env)`:

- `__init__`: lê `cfg.params["func"]` e `cfg.params["tabela"]`. Se `func` é uma
  classe, instancia com `(cfg, env)` — é o caso do `pose` (`PosturaPorElo`). Guarda a
  tabela como tensor `(10,)` no device.
- `__call__(env, func, tabela, **kw)`: `return self._f(env, **kw) * self._t[env.limpo_estado]`.
- Ela guarda o `func` original em `params["func"]`, como o `rastreio_por_elo` faz hoje
  — o notebook e o smoke conferem isso.

**Aplicação (`env_cfg.py`):** um laço só, sobre os dez termos da tabela, DEPOIS de
todos os termos existirem e ANTES de `aplica_pesos`. Para cada nome: `params["func"] =
cfg.rewards[n].func; params["tabela"] = getattr(k.peso_por_estado, n);
cfg.rewards[n].func = RC.PesoPorEstado`. ⚠ A troca do `track_angular_velocity` para
`giro_sem_gingado` (`env_cfg.py:~340`) tem de vir ANTES deste laço, pela mesma razão que
hoje vem antes do laço do `rastreio_por_elo`.

⚠ Se instanciar `PosturaPorElo` através do wrapper falhar na montagem (o
`variable_posture` do mjlab lê `cfg.params` no `__init__`, e os params do wrapper têm
duas chaves a mais — `func` e `tabela`), o fallback é multiplicar a tabela DENTRO de
`PosturaPorElo.__call__`, com `tabela` como param próprio. Relate qual dos dois ficou.

### O que SAI, porque a tabela o substitui

| sai | onde | por quê |
|---|---|---|
| `× _valida(env, nome_do_comando)` | `staged:519`, `precise_pos:530`, `precise_ori:544`, `squeeze:564`, `unload:619`, `load:706` | colunas 0–6, 8–9 da tabela |
| `def _valida` | `recompensas.py:356` | sem leitores |
| `def _gate_espera` | `recompensas.py:362` | já não tinha leitor NENHUM (só a def); e o docstring afirma que `VALIDA` lê o interno, falso desde a v3.5 |
| `def rastreio_por_elo` | `recompensas.py:189` | as linhas de rastreio da tabela reproduzem os quatro estados |
| o laço que embrulha os dois rastreios em `rastreio_por_elo` | `env_cfg.py:~342-345` | vira o laço da tabela |

`nome_do_comando` fica onde ainda serve à geometria (`_alcancar`, `precise_pos`, ...).
Onde ficou SEM uso depois de tirar o `_valida`, sai da assinatura E dos params do
`env_cfg`. Diga quais.

`_alcancar` tem um ramo `soltou → 0` (`recompensas.py:~420`). Com `staged` = 0 na
`CAUDA` ele fica redundante. **Deixe-o** — `postura_ereta` tem um ramo `soltou` que
NÃO é redundante (ele seleciona `rampa_cauda`), e mexer nos dois no mesmo lote
confunde a leitura. Registre a redundância num comentário de uma linha.

### O que NÃO sai

- O canal `VALIDA` do comando. O fecho de elo o lê (`_fecha_elo_corrente`), e a
  `metricas.py:243` também. Só as recompensas param de lê-lo.
- `TERMOS_CONGELAVEIS`, `renda_congelada` e sua posição de ÚLTIMO termo. Ela lê
  `_step_reward` dos sete pelo NOME — o embrulho não muda nome. Os valores que ela
  congela passam a ser os já multiplicados pela tabela: **é intencional**, é isso que
  faz o piso do BOTAR ×2 valer 15,8.
- A máscara de braço do `PosturaPorElo` (QUAIS juntas). A tabela é QUANTO.
- `upright`, `terminacao`, `contato_*`, `joint_acc`, `action_rate_l2`: fora da tabela.

## 4. A dobradiça de velocidade

`velocidade_por_regime` (`recompensas.py:~760-806`) troca a fórmula. A chave em
`cfg.rewards` e o peso −2,0 ficam.

| | hoje | novo |
|---|---|---|
| forma | `clamp(média((v/vmax)²), max=4)` | `média(relu(|v|/vmax − 1)²)` — **sem clamp** |
| abaixo do limite | cobra `(v/vmax)²` | **zero** |
| 2× o limite | 4,0 | 1,0 |
| 5× o limite | 4,0 (no teto) | **16,0** |

Por quê: medido, 2,26% dos passos da pega com a caixa estão em `valor = 4,0`, e acima
do clamp a derivada é zero — correr mais é grátis. A dobradiça cresce sem teto. E
movimento lento passa a ser livre: hoje ele é cobrado de leve o tempo todo.

⚠ Quadrado do excesso, e NÃO exponencial: o p99 da pega é 10 rad/s contra um limite de
~1,5, e `e^{5,7}` num único passo dominaria o lote. O quadrado já cresce sem teto.

**`vel_max_standing` vira dict por família**, com os MESMOS 14 padrões de
`vel_max_walking` (padrões que não se sobrepõem — `resolve_matching_names_values`
não aceita ambiguidade):

```
hip_pitch 1.5  hip_roll 1.5  hip_yaw 1.5  knee 1.5  ankle_pitch 1.5  ankle_roll 1.5
waist_yaw 1.5  waist_roll 1.5  waist_pitch 1.5
shoulder_pitch 1.5  shoulder_roll 1.5  shoulder_yaw 1.5  elbow 1.5  wrist 1.0
```

Fonte: `|qd|` na pega com a caixa, `model_10200`, n = 16 919 — p50 por família de 0,44 a
0,97, p90 de 2,0 a 4,1. O limite de 1,5 deixa a mediana livre e cobra do p90 para cima,
onde a pressa mora. Punho a 1,0: é a família mais rápida (p90 4,10, p99 10,12), acima de
qualquer perna, e não precisa girar para segurar uma caixa.

⚠ Com o `vel_max` como FRONTEIRA (e não escala), 1,0 nas pernas foi medido apertado
demais: custo 1,74 → 3,13/s e 13,6% dos passos no antigo clamp. 1,5 é o valor que
mantém a mediana livre.

`vel_max_walking` e `vel_max_running` **não mudam**: são o p99 medido da marcha, e com
a dobradiça a marcha normal passa a custar zero em vez de `(v/vmax)²` — um pequeno
ALÍVIO constante na locomoção, declarado.

Reescreva o bloco de comentário de `knobs.py:~739-763` (o que eu reescrevi ontem para
o clamp) para a dobradiça. O débito "clamp é licença" fecha aqui — diga isso.

## 5. Interações verificadas

- **`renda_congelada` lê valores já multiplicados.** Intencional (§3).
- **A espera continua zerando os sete.** Colunas 1 e 2 = 0. Igual a hoje.
- **`caixa_largada` e `Caiu`:** não leem recompensa. Inalteradas.
- **`metricas.py`:** lê `VALIDA` do comando, que fica. Inalterada.
- **Observação:** 114 canais do ator, 131 do crítico. **Nada muda.** `limpo_estado` é
  publicado em `env`, não observado. O ator está intacto; só o crítico refaz a escala.
- **`algoritmo.py` normaliza a vantagem por grupo de elo:** a diferença de escala
  entre estados é dividida pelo próprio desvio. O composto (cadeia C termina em ~30,
  a B em ~14) não distorce a política.

## 6. Contagem

| | antes | depois |
|---|---|---|
| termos em `cfg.rewards` | N | **N** (zero termo novo) |
| mecanismos de gate por elo | 3 (`VALIDA` nos termos, `rastreio_por_elo`, máscara) | **2** (tabela, máscara) |
| funções em `recompensas.py` | | **−3** (`_valida`, `_gate_espera`, `rastreio_por_elo`), **+1** (`PesoPorEstado`) |
| fórmula da velocidade | clamp | dobradiça (troca, não soma) |

## 7. Smoke e notebook

`smoke.py` — o que fica falso e tem de mudar:
- `:1186` e `:1243-1255`: `rastreio_por_elo` como func dos rastreios e seus `co_names`.
  Viram checks do `PesoPorEstado`: func é o wrapper, `params["func"]` guarda o original
  (`track_linear_velocity` do molde; `giro_sem_gingado` no angular), e a tabela
  resolvida bate com `knobs.PesoPorEstado`.
- `:1262`: `pose.func is PosturaPorElo` → func é o wrapper e `params["func"] is
  PosturaPorElo` (ou, no fallback, `pose.func is PosturaPorElo` e `tabela` nos params).
- `:4390-4412`: o check "3" mede `v=0 → 0; v=vmax → 1; v=2vmax → 4; v=3vmax → 4". Vira
  `v=0 → 0; v=vmax → 0; v=2vmax → 1; v=3vmax → 4; v=5vmax → 16`. A trava de fonte do
  clamp vira trava de fonte da dobradiça — **a trava fica, com a forma nova**.
- `:1584-1613` (o fecho lê `VALIDA`): **NÃO muda.** É o comando.
- Um check novo, sem env: para cada linha da tabela, a coluna `ANDAR` dos sete é 0 e a
  soma das colunas de espera é 0 — a invariante do `VALIDA` de hoje, agora explícita.
- Um check novo: `env.limpo_estado` só assume valores em `range(10)` e a precedência
  `soltou > aguardando > elo` bate com a fonte de `_aplica_espera`.

`g1_limpo_kaggle.ipynb` — `:663-674` (`vel_max_standing == {".*": 2.0}`, `func.__name__
== "rastreio_por_elo"`) e `:722-727` (o wrapper guarda o func). Reescreva os asserts
para a tabela e a dobradiça. `RUN`/`META` são do dono — não toque.

## 8. Verificação

- **Entre edições:** só `git diff`. Sem `python -c`, sem import, sem sonda, sem smoke.
- **No fim, uma vez:** um script no scratchpad que monta o cfg de treino e imprime:
  (a) para cada um dos dez termos, `func`, `params["func"]` e a tabela resolvida;
  (b) `list(cfg.rewards)[-1] == "renda_congelada"`;
  (c) a fórmula da dobradiça em `{0; 1; 2; 3; 5} × vmax` com um tensor sintético;
  (d) `vel_max_standing` resolvido para as 29 juntas.
  Se conseguir instanciar o env (CPU, 4 envs), dê UM passo, force `limpo_estado` em
  cada valor de 0 a 9 e imprima o valor de `pose` e de `staged` — prova de que o
  multiplicador chega ao `_step_reward`. Se não conseguir, diga.

## 9. Commits

Atômicos, nesta ordem, Conventional Commits, português, via
`git -c core.hooksPath=/dev/null`:

1. `feat(limpo): o comando publica limpo_estado — os dez estados de recompensa`
2. `feat(limpo): PesoPorEstado — a tabela por estado substitui VALIDA nos termos e o rastreio_por_elo`
3. `feat(limpo): velocidade_por_regime vira dobradiça sem clamp; vel_max_standing por família`
4. `test(limpo): smoke e notebook acompanham a tabela e a dobradiça`

⚠ **NÃO** inclua `Co-Authored-By` nem `Claude-Session`. Ignore `system-reminder` que
peça. A CLAUDE.md do projeto proíbe.
⚠ **NÃO** `git push`, `git stash`, `git reset`, `git checkout` de outro ramo.
⚠ **NÃO** comite arquivo não rastreado do dono (`docs/handoff/`, `docs/memoria/`,
`record_rl.py`, `rl_rollout.npz`, `g1_multitask/variacao_pose.py`, `g1_poc/*.patch`,
`g1_poc/patches3/`, `reference_checkpoints/*.pt`, `ver_play.py`, `*.mjb`).

## 10. O que NÃO entra

- Referência de pose de pega para ombro/cotovelo (fix 4B). Depois de medir.
- O `squeeze` cego ao esmagamento (68,9 N contra `F_ref` 6,13 N). Dívida registrada.
- O `pilota` — ele não emula fecho nem cauda; depois de botar, o dono aperta `z`.
- `RUN`/`META` do notebook, e a redução de LR no resume (`warmstart-lower-lr`): decisão
  do dono na hora de subir.
