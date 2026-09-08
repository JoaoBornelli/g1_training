# Spec — G1: o rastreio volta nos elos parados. G2: limite de velocidade por regime

Data: 2026-09-08. Repo `g1_training`, branch `exp/g1-limpo-v2`. Só a pasta `g1_limpo/`.

Aprovado pelo dono em 2026-09-08. Objetivo declarado por ele: **nas tarefas de manipulação o robô inteiro se move de maneira lenta e estável — pernas, braços e tronco.**

⚠ Esta spec entra DEPOIS da `specs/g1-limpo-espera-sigma-e-pose.md` (F1/F2). As duas tocam `recompensas.py`, `env_cfg.py` e `knobs.py`. Não rode as duas em paralelo.

## 0. A medição que calibra tudo

Sonda: `scratchpad/wt-medicao/mede_vel_junta.py`. `model_4999` do `bloco9`, 32 envs, 600 passos, CPU, worktree no commit `8f7eca1`.

`|v|` de junta em rad/s, sobre junta × env × passo:

| fase | p50 | p90 | p99 | máx |
|---|---|---|---|---|
| manipulação parada | 0,58 | **3,13** | **8,19** | 23,80 |
| locomoção parada | 0,01 | 0,43 | 2,24 | 15,06 |
| andando | 0,40 | 2,00 | 4,49 | 15,17 |
| correndo | 0,70 | 2,50 | 7,40 | 14,81 |

**O robô se move mais rápido parado com a caixa do que correndo.** E a locomoção parada recebe a MESMA ordem (comando zero) e produz 7,3× menos no p90.

A diferença é o `rastreio_por_elo`: ele paga 4,0/s por comando zero na locomoção e ZERO na manipulação. Foi o P4 da v2.1.

As pernas correm, não os braços. Manipulação parada, p90 por junta: joelho 6,22; tornozelo pitch 7,05; quadril pitch 4,14; ombro pitch 2,68; cotovelo 2,43. O joelho corre 1,8× mais parado com a caixa do que andando (3,54).

## 1. Regras para quem implementa

- Python é `.venv/bin/python`. Portão: `cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.smoke 2>&1 | tail -30`, timeout 600000 ms.
- **O smoke leva ~4,5 min. Rode-o UMA VEZ, no fim.** No desenvolvimento use script pontual no scratchpad com UM env sintético de 16 envs. Falha intermitente pré-existente na seção 19 ("topo do BOTAR") não é sua.
- `g1_limpo/` NUNCA importa `g1_training`, `g1_poc` nem `g1_multitask`.
- Estilo do módulo: docstring em português, um `⚠` por fato medido ou armadilha, sem comentário decorativo, sem emoji.
- TDD: check primeiro, veja falhar, implemente, veja passar.
- Simplicidade: EXATAMENTE o que esta spec lista.
- Git: NÃO commitar, stash, reset nem checkout. Não tocar em untracked.
- Impasse: implemente o resto e descreva com arquivo:linha.

## 2. G1 — o rastreio volta onde a tarefa manda ficar parado

### O estado de hoje

`recompensas.rastreio_por_elo`, depois do P4:

```python
def rastreio_por_elo(env, *, func, **kwargs) -> torch.Tensor:
    return func(env, **kwargs) * (1.0 - env.limpo_twist_zerado)
```

A regra do P4 era "twist zerado PELA TAREFA não rende rastreio". Ela removeu os 4,0/s que as duas esperas e o segurar-parado cobravam por velocidade zero forçada. Isso estava certo para a espera. Estava ERRADO para o `PEGAR`, o `BOTAR` e o segurar-parado: neles ficar parado É a tarefa, e a medição da §0 mostra o preço.

### A mudança

```python
def rastreio_por_elo(env, *, func, nome_do_comando: str, **kwargs) -> torch.Tensor:
    valor = func(env, **kwargs)
    # ⚠⚠ TRÊS ESTADOS, e não dois. O P4 tinha só dois e custou 7,3× de velocidade de
    # junta (medido 2026-09-08, ver a spec §0):
    #
    #   locomoção, com ou sem comando       paga cheio — é rastreio de verdade
    #   espera, `VALIDA = 0`                paga ZERO  — não existe tarefa ainda
    #   elo parado ATIVO, `VALIDA = 1`      paga cheio — ficar parado É a tarefa
    #
    # O fator é branchless. Confira os três: zerado=0 -> 1; zerado=1 e valida=1 -> 1;
    # zerado=1 e valida=0 -> 0.
    fator = 1.0 - env.limpo_twist_zerado * (1.0 - _valida(env, nome_do_comando))
    return valor * fator
```

`env_cfg.py`, no laço dos dois rastreios: devolver `nome_do_comando="alvo_caixa"` aos params. O P4 o havia removido.

### O que isto cobre, e o que não cobre

Cobre: velocidade da BASE nos elos parados ativos. Portanto conserta também o arrasto lateral com a caixa na mão, que foi a outra observação do dono.

Não cobre: a espera. Lá o `VALIDA` é zero e o fator é zero, de propósito. Quem cuida da pose na espera é o F2 da outra spec (`pose_de_braco`).

⚠ No treino o `ANDAR` **não** está em `elos_parados` — o append de `CMD.ANDAR` vive dentro do ramo `if entrega_apos_s is not None` (`env_cfg.py`, visualizador de entrega). Portanto env de locomoção tem `limpo_twist_zerado = 0`, inclusive os 10% de `rel_standing_envs` que sorteiam comando zero. Não mexa nisso.

### Smoke do G1

Seção nova `--- G1/G2: lento e estável`:

1. Os dois rastreios têm `nome_do_comando` nos params e `func=RC.rastreio_por_elo`.
2. Num env `elo=PEGAR`, 16 envs, DURANTE a espera (sem `_passa_janela`): `limpo_twist_zerado == 1`, `VALIDA == 0`, e `track_linear_velocity == 0`.
3. No MESMO env, depois de `_passa_janela`: `limpo_twist_zerado == 1`, `VALIDA == 1`, e `track_linear_velocity > 0`. **Este é o check que falha hoje.**
4. Num env `elo=ANDAR` de locomoção: `limpo_twist_zerado == 0` e o rastreio é igual ao termo do molde.
5. Num env `elo=PEGAR, cadeia=3`, depois de `forca_avanco` (→ CARREGAR de segurar-parado): `limpo_twist_zerado == 1`, `VALIDA == 1`, rastreio `> 0`.

## 3. G2 — `velocidade_por_regime`

### O desenho, e ele é o do dono

Espelhe o `variable_posture` do fabricante (`mjlab/tasks/velocity/mdp/rewards.py:385`), trocando erro de POSIÇÃO por VELOCIDADE:

- Três dicts de limite: `standing`, `walking`, `running`, resolvidos por `resolve_matching_names_values` no `__init__`, como o molde faz.
- O regime vem do COMANDO, não da velocidade medida: `total = ‖cmd[:2]‖ + |cmd[2]|`; `standing` se `total < walking_threshold`; `running` se `total >= running_threshold`.
- `walking_threshold = 0.05` e `running_threshold = 1.5`, os mesmos do molde.
- Retorno: `exp(−média(v² / vmax²))`, em [0, 1]. 1,0 = parado.

⚠⚠ **O regime JÁ É o gate da tarefa, e não precisa de um segundo.** Em todo elo de manipulação o `_zera_twist_nos_parados` escreve zero no comando, portanto `total = 0 < 0,05` e o regime é `standing`. Não acrescente gate por `limpo_twist_zerado` nem por `VALIDA`.

### Os números, todos medidos

`knobs.Tarefa`:

```python
# ⚠⚠ O LIMITE DO `standing` É UMA ENTRADA SÓ, como o `std_standing = {".*": 0.05}` do
# fabricante: a ordem do dono é "o robô inteiro lento", pernas, braços e tronco.
#
# ⚠ E o 2,0 NÃO é escolhido: é o p99 da LOCOMOÇÃO PARADA medido em 2026-09-08. A régua
# é "fique tão parado quanto você já fica sem a caixa". Medido no `model_4999`: com
# vmax 2,0 a mediana do comportamento de hoje cai em 0,474 — o meio da faixa, onde a
# derivada é máxima. Com 1,0 ela cai em 0,051 e o termo vira canal morto, que é o
# defeito medido no `PosturaPorElo`. Com 5,0 ela sobe a 0,887 e o termo satura.
vel_max_standing: dict = field(default_factory=lambda: {".*": 2.0})

# ⚠ Os dois de baixo são o p99 MEDIDO de cada padrão naquele regime. O termo paga 0,96
# na marcha normal: ele não taxa a locomoção, ele morde o excesso.
# ⚠ E o `running` é `max(p99_running, p99_walking)` por padrão, e isso é decisão: o p99
# de `running` saiu MENOR que o de `walking` em cinco padrões, porque a amostra veio do
# transiente do `velocity_stages` novo (lin_vel_x foi a 2,0 m/s na última iteração da
# bloco9). Limite de correr mais apertado que o de andar puniria correr, ao contrário.
vel_max_walking: dict = field(default_factory=lambda: {
    r".*hip_pitch.*": 4.0,  r".*hip_roll.*": 5.0,   r".*hip_yaw.*": 4.0,
    r".*knee.*": 6.5,       r".*ankle_pitch.*": 6.0, r".*ankle_roll.*": 4.0,
    r".*waist_yaw.*": 3.0,  r".*waist_roll.*": 5.0, r".*waist_pitch.*": 2.5,
    r".*shoulder_pitch.*": 3.0, r".*shoulder_roll.*": 4.0,
    r".*shoulder_yaw.*": 2.0, r".*elbow.*": 3.0, r".*wrist.*": 2.5,
})
vel_max_running: dict = field(default_factory=lambda: {
    r".*hip_pitch.*": 5.0,  r".*hip_roll.*": 5.0,   r".*hip_yaw.*": 4.0,
    r".*knee.*": 10.0,      r".*ankle_pitch.*": 9.0, r".*ankle_roll.*": 4.0,
    r".*waist_yaw.*": 3.0,  r".*waist_roll.*": 5.0, r".*waist_pitch.*": 2.5,
    r".*shoulder_pitch.*": 3.5, r".*shoulder_roll.*": 4.5,
    r".*shoulder_yaw.*": 2.5, r".*elbow.*": 3.0, r".*wrist.*": 2.5,
})
# ⚠ Peso 2,0, igual ao `postura_ereta` e ao `unload`. Medido: o comportamento de hoje
# paga 0,95 dos 2,0, portanto ele perde 1,05/s de uma renda de manipulação de ~24/s.
velocidade_por_regime: float = 2.0
```

Use `dataclasses.field(default_factory=...)` — dict mutável como default de dataclass levanta erro.

### A fiação

`env_cfg.py`, na seção 3g, **antes** do `renda_congelada`:

```python
cfg.rewards["velocidade_por_regime"] = RewardTermCfg(
    func=RC.velocidade_por_regime, weight=tr.velocidade_por_regime,
    params={"vel_max_standing": tr.vel_max_standing,
            "vel_max_walking": tr.vel_max_walking,
            "vel_max_running": tr.vel_max_running,
            "command_name": "twist",
            "walking_threshold": 0.05, "running_threshold": 1.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
```

⚠ `renda_congelada` continua o ÚLTIMO termo do dict. ⚠ `velocidade_por_regime` **NÃO** entra em `TERMOS_CONGELAVEIS`: ele não depende do elo, ele depende do regime de comando.

⚠ O `walking_threshold` e o `running_threshold` NÃO viram knob. Eles são os do molde, e escrever um segundo lugar para o mesmo número é como o `std_standing` do `pose` deriva em silêncio num upgrade. Leia-os de `cfg.rewards["pose"].params` se preferir uma fonte só; se ler, afirme no smoke que os dois termos usam o mesmo valor.

### A métrica

`metricas.py`: `velocidade_de_junta`, RMS de `joint_vel` sobre as 29 juntas, por env. `reduce="mean"`. Sem peso.

⚠ Ela existe para recalibrar os três dicts sem precisar da sonda de CPU. Foi a ausência dela que obrigou a sonda desta vez.

### Smoke do G2

6. Os três dicts existem em `knobs.Tarefa`, o `standing` tem UMA entrada `".*"` com valor 2,0, e todo padrão do `walking` tem limite `<=` o do `running`.
7. `cfg.rewards["velocidade_por_regime"]` existe, resolve as **29** juntas, e `list(cfg.rewards)[-1] == "renda_congelada"` segue verdadeiro.
8. `"velocidade_por_regime" not in TERMOS_CONGELAVEIS`.
9. Fórmula, sem simulador: com todas as juntas em `vmax`, `exp(−1) = 0,3679`; com todas em `vmax/2`, `exp(−0,25) = 0,7788`; com todas paradas, 1,0. Afirme os três com tolerância 1e−3.
10. Num env `elo=PEGAR` depois de `_passa_janela`: o comando do twist é zero, portanto o regime é `standing`. Confirme que o `vmax` efetivo usado é o do `standing` — por exemplo escrevendo `joint_vel` a 2,0 rad/s em todas as juntas e afirmando que o termo vale `exp(−1)` com tolerância 0,02.
11. `cfg.metrics["velocidade_de_junta"]` existe.

## 4. Lote de revisão

Revisar o `git diff` contra esta spec:

- `rastreio_por_elo` tem os TRÊS estados, e o fator é `1 − zerado × (1 − VALIDA)`. Os dois termos recebem `nome_do_comando`.
- `velocidade_por_regime` espelha o `variable_posture`: `resolve_matching_names_values` no `__init__`, regime pelo COMANDO, três máscaras somadas, `exp(−média(v²/vmax²))`.
- Nenhum gate a mais no `velocidade_por_regime` além do regime.
- `standing` é uma entrada só. `running >= walking` em todo padrão.
- `velocidade_por_regime` fora de `TERMOS_CONGELAVEIS`; `renda_congelada` é o último termo.
- Os dicts usam `field(default_factory=...)`.
- Smoke verde fora a intermitente da seção 19. Nenhum check antigo apagado sem substituto.

Saída: APROVADO, ou lista de correções com arquivo e linha.
