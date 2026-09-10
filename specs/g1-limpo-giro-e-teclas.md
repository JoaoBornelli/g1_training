# g1_limpo — o giro sai do gingado, e as teclas do pilota (v1)

Dois consertos independentes. O primeiro muda a recompensa de locomoção; o segundo é
só ferramenta.

## §0 PARTE A — o giro

### 0.1 O problema, no código

`mjlab/tasks/velocity/mdp/rewards.py:57-64`:

```python
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + xy_error
    return torch.exp(-ang_vel_error / std**2)
```

`xy_error` é o roll e o pitch da base. **Eles nunca são comandados.** Um humanoide
andando balança a pelve a cada passo, e esse balanço entra no MESMO expoente do erro
de guinada.

E ele é cobrado DUAS VEZES: `body_ang_vel` (peso −0,05) já pune velocidade angular do
corpo.

### 0.2 O tamanho, MEDIDO na `bloco14` it 9847

| termo | painel | /s | % do máximo (peso 2,0) |
|---|---|---|---|
| `track_linear_velocity` | 0,9951 | 1,18 | **59%** |
| `track_angular_velocity` | 0,1095 | 0,13 | **6,5%** |

Invertendo `exp(−err²/std²)` com os `std²` do molde (0,25 linear, 0,50 angular):

| | erro² total |
|---|---|
| linear | 0,132 |
| angular | **1,37** |

Dez vezes maior. O canal angular vive a 6,5% do máximo — saturado perto de zero, onde
a derivada quase some. É o padrão de `g1-sigma-e-a-distancia-inicial` e do penhasco do
`std_standing`, e explica o que o dono viu no `pilota`: **o robô não gira.**

⚠ A fração de envs com giro é baixa (~13%: 31% de locomoção × 90% não-parados × 60%
não-heading, e o `heading` decai a zero ao alinhar), mas ela NÃO é o gargalo. Nesses
13% o gradiente já está morto. Subir a fração antes de destravar o termo compra pouco.
Fica registrado e **fora deste lote**.

### 0.3 A MUDANÇA

`recompensas.py`, uma função nova, ao lado do `rastreio_por_elo`:

```python
def giro_sem_gingado(env, std: float, command_name: str,
                     asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """`exp(−(wz_cmd − wz)²/σ²)`. O `track_angular_velocity` do fabricante SEM o
    roll e o pitch da base.

    ⚠⚠ POR QUE. O termo do molde soma `wx² + wy²` ao erro de guinada
    (`velocity/mdp/rewards.py:62`). Roll e pitch da base NUNCA são comandados: são o
    gingado da marcha. MEDIDO na `bloco14` it 9847: o canal angular vale 6,5% do
    máximo contra 59% do linear, e o erro² angular é 1,37 contra 0,132 — o expoente é
    dominado pelo que não se pede, e a derivada em relação à guinada quase some. O
    robô não gira, e o `play` confirma.

    ⚠ O gingado CONTINUA punido, uma vez só: `body_ang_vel` (peso −0,05) já mede
    velocidade angular do corpo. No molde ele era cobrado duas vezes.

    ⚠ `std` NÃO muda: fica em `sqrt(0.5)` do molde. Sem o `xy_error`, um erro de
    guinada de 0,7 rad/s — o topo do envelope — dá `exp(−0,49/0,5) = 0,37`. O kernel
    já cobre a faixa; mexer no σ junto misturaria duas mudanças numa medição.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"comando '{command_name}' não existe"
    wz = asset.data.root_link_ang_vel_b[:, 2]
    return torch.exp(-torch.square(command[:, 2] - wz) / std**2)
```

`env_cfg.py`, **antes** do laço que envolve os dois rastreios (~linha 332):

```python
    # ⚠ v1 do lote `giro-e-teclas`: o giro perde o gingado ANTES de ser envolvido. A
    # ordem importa — o laço abaixo guarda `_t.func` em `params["func"]`, portanto a
    # troca tem de acontecer aqui, ou o wrapper chamaria o termo velho.
    cfg.rewards["track_angular_velocity"].func = RC.giro_sem_gingado

    for _nome_rastreio in ("track_linear_velocity", "track_angular_velocity"):
        ...
```

⚠ `track_linear_velocity` **NÃO muda**. Ele soma `vz²`, que também não é comandado,
mas o número diz que ele não está saturado: 59% do máximo, erro² 0,132. Sem problema
medido, sem mudança.

### 0.4 A contagem

| | antes | depois |
|---|---|---|
| operandos no erro angular | 2 (`z_error + xy_error`) | **1** |
| funções em `recompensas.py` | n | **n + 1** |
| termos de recompensa | 28 | 28 |
| knobs | — | 0 |

Honesto: **uma função entra.** O que sai é o `xy_error`, que era cobrado duas vezes.
O dono autorizou adição neste ciclo de fechamento da arquitetura.

### 0.5 O que NÃO muda

Pesos, `std`, envelope do currículo, `rel_standing_envs`, `rel_heading_envs`, a fração
de envs de locomoção, o piso de 30%, `track_linear_velocity`, `body_ang_vel`, e a
função do molde (ela fica intocada; só deixa de ser chamada).

## §0b PARTE C — o robô não precisa aprender a correr

### 0b.1 O problema, MEDIDO na própria `bloco14`

`mjlab/tasks/velocity/velocity_env_cfg.py:405` — os estágios do envelope são por PASSO
FIXO, não por desempenho:

```python
{"step": 0,          "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
{"step": 5000 * 24,  "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
{"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},          # <- o terceiro
```

`common_step_counter` conta passos de controle e há 24 por iteração, logo o terceiro
estágio dispara na **iteração 10000**. A `bloco14` cruzou, e o topo linear foi de 2,0
para 3,0 m/s.

O custo, em 452 iterações (9847 → 10299):

| | 9847 | 10299 |
|---|---|---|
| `Curriculum/forma/s_C` | 0,1952 | **0,1418** |
| `Metrics/alvo_caixa/sucesso` | 0,3062 | 0,2230 |
| Mean reward | 156,3 | **126,6** |
| `Episode_Termination/caixa_largada` | 10,7% | **16,3%** |
| Mean action std | 0,60 | **0,67** |
| entropy loss | 25,2 | 28,5 |

O std e a entropia subindo são re-exploração depois de choque de distribuição. E a
cauda de B e R sorteia do MESMO envelope: o robô anda com a caixa a 3 m/s e a perde —
daí o `caixa_largada` subindo 50%.

Decisão do dono, 2026-09-10: **"corta o estágio, o robô não precisa aprender a
correr."** Ele quer caminhar com caixa e apoiar em mesa; 3 m/s não serve a nenhum dos
dois e cobra dos dois.

### 0b.2 A MUDANÇA

`env_cfg.py`, junto do bloco que monta o currículo (~linha 443, ANTES de
`cfg.curriculum["forma"]`):

```python
    # ⚠⚠ O TERCEIRO ESTÁGIO DO ENVELOPE SAI (decisão do dono, 2026-09-10). O molde
    # sobe `lin_vel_x` para (−2,0; 3,0) na iteração 10000, por PASSO FIXO e não por
    # desempenho (`velocity_env_cfg.py:405`). MEDIDO na `bloco14`: ao cruzar,
    # `s_C` caiu 0,195 -> 0,142, o reward 156 -> 127 e o `caixa_largada` subiu de
    # 10,7% para 16,3% — a cauda de B e R sorteia do MESMO envelope e o robô perde a
    # caixa andando a 3 m/s. A tarefa é caminhar com caixa e apoiar em mesa.
    # ⚠ ISTO QUEBRA A PARIDADE com o molde, de propósito. O assert do notebook muda
    # junto (§2). Os dois primeiros estágios ficam INTOCADOS.
    _ests = cfg.curriculum["command_vel"].params["velocity_stages"]
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
        e for e in _ests if e["step"] < 10000 * 24]
```

⚠ O filtro é por `step`, e não `[:2]`: um upgrade do `mjlab` que acrescente um estágio
intermediário quebraria o corte por índice em silêncio.

⚠ **NÃO** mexa em `ang_vel_z`. O terceiro estágio não o toca — ele fica em (−0,7; 0,7)
desde o segundo. A Parte A é que trata o giro.

⚠ O bloco de `inspecao` em `env_cfg.py:777` faz `cfg.curriculum.pop("command_vel")`.
Ponha o corte ANTES desse ponto, ou proteja com `if "command_vel" in cfg.curriculum`.

### 0b.3 O retomar

O dono parou a `bloco14` na ~10300. Com o estágio cortado, o `commands_vel` volta a
aplicar o segundo estágio: o envelope cai para (−1,5; 2,0). O checkpoint carrega a
política já exposta a 3 m/s, e isso é inofensivo — ela só deixa de receber o pedido.

**Retome do checkpoint mais alto.** Os 300 passos de exposição são desvio de
distribuição, não corrupção, e voltar para a ~9850 jogaria fora 450 iterações do resto.

### 0b.4 A contagem

| | antes | depois |
|---|---|---|
| estágios do envelope | 3 | **2** |
| termos de recompensa | 28 | 28 |
| knobs | — | 0 |

Subtrativo.

## §1 PARTE B — as teclas do `pilota`

### 1.1 O problema

O `key_callback` do `launch_passive` é passado ao `_Simulate` em C++, que trata as
próprias teclas ANTES. O dono confirmou no uso: `0`–`4` mudam a visualização da cena.

Reservadas pelo `simulate`, e a spec do `pilota` usou **oito** delas: `0`–`9`,
**espaço**, as **quatro setas**, **Backspace**, **`[`**, **`]`**, Tab e F1–F5.

### 1.2 O mapa novo, só letras

| tecla | efeito |
|---|---|
| `w` `s` | `vx` ± passo |
| `a` `d` | `vy` ± passo |
| `q` `e` | `wz` ± passo |
| `x` | zera o twist |
| `z` `c` `v` `b` `n` | elo: ANDAR · REORIENTAR · PEGAR · CARREGAR · BOTAR |
| `-` `=` | os três tetos × 0,8 e × 1,25 |
| `,` `.` | fator de tempo ÷2 e ×2 |
| `p` | reset |

Nenhuma é atalho global do `simulate`.

### 1.3 Duas defesas, porque eu já errei este mapa uma vez

**O mapa vira dado.** Um dict no topo do módulo, `TECLAS = {"w": "vx+", ...}`, com uma
bandeira `--teclas "w=vx+,s=vx-"` que sobrepõe entradas. Se alguma ainda colidir, o
dono troca por bandeira em vez de esperar um lote.

**Tecla desconhecida se anuncia.** No `key_callback`, uma tecla fora do mapa imprime
`[pilota] tecla <código> ('<char>') não está no mapa`. Assim ele descobre na hora qual
o `simulate` engoliu, em vez de achar que o pilota travou.

E o cabeçalho impresso na partida lista o mapa CORRENTE, montado do dict — não um
texto escrito à mão que sai de sincronia.

## §2 O notebook

A tabela de recompensa muda (Parte A) e o currículo muda (Parte C), portanto:

1. `RUN = "bloco15"` nas DUAS células que carregam o nome.
2. `META = 12500`. O dono parou na ~10300; isso dá ~2200 iterações.
3. **O assert de paridade dos estágios TEM de mudar** — hoje ele é:

```python
assert cu["command_vel"].params["velocity_stages"] == \
    fab.curriculum["command_vel"].params["velocity_stages"]
```

e passa a ser a afirmação do que queremos, com o motivo:

```python
# ⚠ PARIDADE QUEBRADA DE PROPÓSITO (spec `g1-limpo-giro-e-teclas.md` §0b): o terceiro
# estágio do molde sobe `lin_vel_x` a 3,0 m/s na iteração 10000, por passo fixo. MEDIDO
# na bloco14: `s_C` 0,195 -> 0,142 e `caixa_largada` 10,7% -> 16,3% ao cruzar.
_ests = cu["command_vel"].params["velocity_stages"]
assert len(_ests) == 2 and _ests[-1]["lin_vel_x"] == (-1.5, 2.0), \
    f"o envelope de velocidade não está cortado no 2º estágio: {_ests}"
assert _ests == [e for e in fab.curriculum["command_vel"].params["velocity_stages"]
                 if e["step"] < 10000 * 24], \
    "os dois primeiros estágios divergiram do molde — só o TERCEIRO deve sair"
```

4. O discriminador da Parte A, junto aos de v3.5:

```python
assert rw["track_angular_velocity"].params["func"].__name__ == "giro_sem_gingado", \
    "clone anterior ao lote do giro: o rastreio angular ainda soma roll e pitch da " \
    "base ao erro de guinada — canal a 6,5% do máximo, sem derivada"
```

⚠ Confira que `params["func"]` é mesmo onde o wrapper guarda a função; se o campo for
outro, ajuste o assert, não o código.

Não mexa na LR. A célula já baixa para 5e-4.

## §3 A VERIFICAÇÃO

**Parte A** — uma execução, no fim: 16 envs, 50 passos, CPU, `make_env_cfg()` padrão.
Sem exceção; `cfg.rewards["track_angular_velocity"].params["func"].__name__ ==
"giro_sem_gingado"`; e o termo devolve shape `(16,)` em `[0, 1]` sem NaN.

**Parte B** — `python -c "import ast,sys; ast.parse(open('pilota.py').read())"` e a
checagem que o conserto de `726b817` motivou: `pilota.main.__code__.co_varnames` não
contém `mujoco`. **Não abra o viewer.**

⚠ O `--paridade` do `exporta_cena` NÃO precisa rodar: a Parte B não toca observação,
alvo nem `ctrl`. Se você mudar qualquer um dos três, rode.

## §4 O que este lote NÃO toca

`comando.py`, `terminacoes.py`, `curriculo.py`, `knobs.py`, `observacoes.py`,
`runner.py`, `algoritmo.py`, `exporta_cena.py`, as specs anteriores. Não roda
`smoke.py`. Não muda o contrato de observação nem o envelope do currículo.

## §5 Commits

Atômicos, Conventional Commits, mensagem em português, **sem trailer de atribuição**
(`git -c core.hooksPath=/dev/null`). Três:

1. `fix(limpo): o rastreio de guinada perde o roll e o pitch da base`
2. `fix(pilota): teclas que o simulate não reserva, mapa configurável`
3. `docs(specs): g1-limpo giro e teclas`

O notebook entra no commit 1.

Nunca commite: `docs/handoff/`, `docs/memoria/`, `record_rl.py`, `rl_rollout.npz`,
`g1_multitask/variacao_pose.py`, `g1_poc/*.patch`, `g1_poc/patches3/`,
`reference_checkpoints/*.pt`, `ver_play.py`, `*.mjb`. Nunca `git push`.
