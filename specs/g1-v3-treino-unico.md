# g1_v3: um treino do zero que percorre o currículo inteiro (dono, 11/11)

Esta spec é o contrato do lote de 11/11. Ela registra as decisões e as edições exatas.

## 1. Decisões do dono (11/11)

1. O treino roda do zero, sem intervenção, e segue o currículo: ANDAR, PEGAR, BOTAR e CARREGAR,
   e depois o refino (§11 de `g1-v3.md`).
2. As restrições ficam baixas até o robô aprender as quatro tarefas em alguns níveis.
3. As restrições sobem depois, sem passo agressivo, pela regra do freio (§11.4): seguem o nível.
4. A recompensa do início paga ficar de pé e andar.

Motivo medido (11/11): com ação zero o robô cai em 1,12 s e viver rende −1,6/s sem a terminação;
a marcha ficou no nível 0 da iteração 4600 à 11231; o portão conjuntivo das células nunca abriu;
a abertura da fatia B desfez a marcha; o P4 nunca fechou (caixa 24 a 26 cm abaixo, 11 a 13 cm de
lado, pelve a 3,2 a 4,7 cm com tolerância de 3 cm e sem termo abaixo de 8 cm); o punho pedia
10 N·m por unidade de ação com limite de 5 N·m.

## 2. O número `restricao`

`restricao = min, nas 4 tarefas, de nivel(tarefa) / max_nivel[tarefa]`, em [0, 1].

- Vale 0 até as quatro tarefas passarem do nível 0. Nunca desce.
- Com `frente_tetos` de 5 valores, as quatro tarefas têm `max_nivel` 4: os valores possíveis são
  0; 0,25; 0,5; 0,75; 1.
- Todo par `(início, fim)` desta spec vale `balanceador.no_degrau(início, fim, restricao, 1)`.

| O que segue a `restricao` | início (0) | fim (1) |
|---|---|---|
| `vel_junta`, `acel_junta`, `energia`, `desvio_bracos`, `desvio_cintura`, `desvio_quadril`, `giro_xy` | fator `estilo_inicio` 0,25 | fator 1 |
| `cerca_giro`, `cerca_tronco`, `cerca_com`, `cerca_torque`, `cerca_pelve`, só nos envs da tarefa ANDAR | fator 0 | fator 1 |
| σ₀ do giro sem comando | `sigma_zero_ang_inicio` 0,5 | `sigma_zero_ang` 0,15 |
| deriva de posição do parado (sucesso do segmento) | `deriva_pos_inicio` 0,25 m | `deriva_pos_m` 0,10 m |
| teto do atraso (passos) | 0 | `ceil(atraso_portao[1] · restricao)`: 1, 1, 2, 2 |

O refino de 5 degraus (§11.5) fica como está. Com o degrau global ≥ `degrau_atraso_refino`, o teto
do atraso é `atraso_refino[1]`, como hoje.

## 3. Regras do lote (valem para toda tarefa)

- Trabalhe só no worktree `/home/joaobornelli/workbench/projects/g1-treino-v3/g1_training`, branch `g1_v3`.
- Commit local no fim da tarefa, com os arquivos da tarefa por caminho explícito. PROIBIDO: push,
  `gh`, qualquer operação no remoto, `git stash`, `Co-Authored-By` e qualquer trailer de coautoria.
- Durante as edições, não rode teste. Faça todas as edições da tarefa, confira com `git diff`, e
  rode os testes uma vez, no fim.
- Comando dos testes: `.venv/bin/python -m unittest discover -s g1_v3/testes -t .`
  A falha `test_mid360.TestMid360.test_primeira_direcao_bate_com_o_csv` é antiga (CSV fora do repo)
  e não conta. Qualquer outra falha conta.
- Um teste que fixa um valor antigo desta spec é atualizado para o valor novo. Não apague teste.
  Não enfraqueça um assert para ele passar.
- Não mude nada fora do que a tarefa lista. Não refatore. Não renomeie.
- Todo número novo mora em `g1_v3/knobs.py`.
- Relate no fim: os arquivos mudados, o comando de teste, as últimas 15 linhas da saída e o hash do commit.

## 4. Tarefa T1: knobs do início e escala do punho

Arquivos: `g1_v3/knobs.py`, `g1_v3/env_cfg.py`, `g1_v3/smoke.py`, `g1_v3/testes/`.

### 4.1 `g1_v3/knobs.py`

| Classe | Campo | Valor novo |
|---|---|---|
| `Contrato` | `escala_acao_punho` (campo novo, logo depois de `escala_acao`) | `0.075` |
| `Comando` | `frente_tetos` | `(0.5, 0.75, 1.0, 1.25, 1.5)` |
| `Comando` | `janela_media_s` | `0.8` (fica; ver a nota abaixo da tabela) |
| `Comando` | `deriva_pos_inicio` (campo novo, logo depois de `deriva_pos_m`) | `0.25` |
| `Recompensa` | `rastreio_linear` | `2.0` |
| `Recompensa` | `rastreio_angular` | `2.0` |
| `Recompensa` | `vivo` | `1.0` |
| `Recompensa` | `sigma_zero_ang_inicio` (campo novo, logo depois de `sigma_zero_ang`) | `0.5` |
| `Recompensa` | `estilo_inicio` (campo novo, logo depois de `energia`) | `0.25` |
| `Balanceador` | `rampa_fatia_inicio` (campo novo, depois de `iters_min_nivel`) | `0.05` |
| `Balanceador` | `rampa_fatia_passo` (campo novo) | `0.02` |
| `Balanceador` | `rampa_fatia_its` (campo novo, `int`) | `12` |
| `DR` | `atraso_inicio` | `(0, 0)` |

Nota sobre `janela_media_s` (verificação de 11/11): a T1 aplicou 0,02 e a medida local com ação zero e com ação
N(0,1) reprovou o valor. Com 0,02 o `rastreio_linear` do robô sem treino foi de −0,24 para −0,70/s na célula reto e
de −1,04 para −1,69/s na parado, e o `rastreio_angular` de −0,68 para −1,79/s. O knob voltou a 0,8.

Cada campo novo ou mudado leva um comentário na linha com "11/11" e o motivo em poucas palavras
(use a §1 e a §2). Os tipos são `float`, salvo os indicados.

Função nova, no fim do arquivo, depois de `q_de_pe`:

```python
def escala_por_junta(c: Contrato = Contrato()) -> tuple:  # I §3; 11/11: (29,) rad por unidade, na ordem de JUNTAS
    # o punho (wrist_pitch e wrist_yaw) usa a escala do mjlab; com 0,25 e kp 40 ele pedia 10 N·m com limite de 5
    return tuple(c.escala_acao_punho if re.fullmatch(r".*_wrist_(pitch|yaw)_joint", j) else c.escala_acao
                 for j in JUNTAS)
```

### 4.2 `g1_v3/env_cfg.py`

- No import de `g1_v3.knobs`, acrescente `JUNTAS` e `escala_por_junta`.
- Em `cfg.actions`, troque `scale=k.contrato.escala_acao` por
  `scale=dict(zip(JUNTAS, escala_por_junta(k.contrato)))`.

### 4.3 `g1_v3/smoke.py`

- Importe `escala_por_junta` de `g1_v3.knobs`.
- Na linha `acao = (q - qde + robo.data.encoder_bias[:, idx]) / ATIVO.contrato.escala_acao`, troque o
  divisor por `torch.tensor(escala_por_junta(), device=q.device, dtype=q.dtype)`.

### 4.4 Testes

- `test_knobs.py`: teste novo: `escala_por_junta()` tem 29 valores; 4 valem 0,075 (os dois
  `wrist_pitch` e os dois `wrist_yaw`) e 25 valem 0,25.
- `test_cfg.py`: teste novo: o `scale` do termo `joint_pos` é um dict com as 29 juntas de `JUNTAS`
  e os valores de `escala_por_junta()`.
- Ajuste os testes que fixam os valores antigos.

Commit: `feat(v3): knobs do início do treino único e escala do punho por junta`

## 5. Tarefa T2: currículo (`balanceador.py`, `curriculo.py`)

### 5.1 Média das células no lugar do portão conjuntivo

Método novo em `Curriculo`, logo depois de `_passa`:

```python
    def _passa_media(self, ids):
        # 11/11: a média das células no portão. O portão conjuntivo (todas ≥ portão ao mesmo tempo) nunca abriu
        # em 13 TensorBoards; o refino (`_conta_refino`) já usava a média.
        return bool(ids) and sum(self.celulas[i].s for i in ids) / len(ids) >= self.k.portao
```

Trocas (só estas):

- `_abre`, fatia B: `self._passa(andar_reto) and self._passa(andar_parado)` vira
  `self._passa_media([andar_reto, andar_parado])`.
- `_abre`, cadeias: `all(self._passa(i) for i in abertas)` vira `self._passa_media(abertas)`.
- `_abre`, terreno: `all(self._passa(i) for i in self._abertos(t))` vira `self._passa_media(self._abertos(t))`.
- `_sobe`: `all(self._passa(i) for i in ids)` vira `self._passa_media(ids)`.
- `_conta_refino`: a conta `media = ...` e o teste `ultimo and abertas and media >= self.k.portao` viram
  `ultimo and self._passa_media(abertas)`.

Não mude: a abertura da fatia C (`self._passa(_POS[("PEGAR", "pegar")])`), a condição do terreno do
ANDAR no CARREGAR, o `_trava_dr` e o `nivel_passou`.

### 5.2 `restricao`

Método novo em `Curriculo`, logo depois de `degrau_global`:

```python
    def restricao(self) -> float:
        # 11/11 (specs/g1-v3-treino-unico.md §2): as restrições seguem o progresso, como o freio segue o nível.
        # O menor nível relativo entre as 4 tarefas: 0 até todas passarem do nível 0; nunca desce.
        return min(self._nivel[t] / self.max_nivel[t] for t in TAREFAS)
```

### 5.3 Rampa da fatia

- `__init__`: `self._it_abertura = [0, None, None]` (iteração em que cada fatia abriu; o ANDAR abre na 0).
- `fecha_iteracao`, logo depois de `self._abre()`: para `f` em `(1, 2)`, se `self._it_abertura[f] is None` e
  existe célula aberta com `FATIA_DE[c.tarefa] == f`, então `self._it_abertura[f] = self.it`.
- `fracoes_fatias`, antes do `return out` final: para `f` em `(1, 2)` com `self._it_abertura[f]` diferente de
  `None`:

```python
            teto = min(1.0, self.k.rampa_fatia_inicio
                       + self.k.rampa_fatia_passo * ((self.it - self._it_abertura[f]) // self.k.rampa_fatia_its))
            excesso = max(0.0, float(out[f]) - teto)
            out[f] -= excesso
            out[0] += excesso  # o excedente volta para o ANDAR
```

- `estado()`: chave nova `"it_abertura": list(self._it_abertura)`.
- `carrega()`: `self._it_abertura = list(estado.get("it_abertura", [0, None, None]))`.

### 5.4 Atraso em `dr()`

Troque o valor de `"atraso_max"` por:

```python
            "atraso_max": (d.atraso_refino[1] if g >= d.degrau_atraso_refino
                           else math.ceil(d.atraso_portao[1] * self.restricao() - 1e-9)),
```

Acrescente `import math` no topo. O `p0` continua a valer para `atrito_pad`, `atrito_laje`, `mu` e `com`.

### 5.5 `g1_v3/curriculo.py`

Na função `v3`, na linha `log.update(atraso_max=teto, ...)`, acrescente `restricao=cur.restricao()`.

### 5.6 Testes (em `test_balanceador*.py` e `test_curriculo*.py`)

- Média: com reto 0,60 e parado 0,95 a fatia B abre; com reto 0,60 e parado 0,80 não abre.
- Média: o nível sobe com as células de velocidade abertas e a média ≥ 0,75, mesmo com uma célula em 0,70.
- `restricao`: 0 com todos os níveis em 0; 0 com três tarefas no nível 1 e uma no 0; 0,25 com as quatro no
  nível 1; 1 com as quatro no último nível.
- Rampa: na iteração em que a fatia B abre, `fracoes_fatias()[1]` vale no máximo 0,05; 12 fechos depois,
  no máximo 0,07; a soma das três frações continua 1.
- Estado: `estado()` e `carrega()` guardam e repõem `it_abertura`; um estado sem a chave carrega com o padrão.
- Atraso: `dr()["atraso_max"]` vale 0 com `restricao` 0; 1 com 0,25 e com 0,5; 2 com 0,75 e com 1.
- Ajuste os testes que esperavam o portão conjuntivo e o atraso pelo `portao0`.

Commit: `feat(v3): currículo pela média das células, restricao pelo nível e fatia em rampa`

## 6. Tarefa T3: recompensa e sucesso (`recompensas.py`, `comando.py`)

### 6.1 Função nova em `recompensas.py`, logo depois de `por_degrau`

```python
def por_restricao(env, ini, so_andar=False):
    """Fator (E,) das restrições que seguem o progresso (11/11; specs/g1-v3-treino-unico.md §2): `ini` com
    `restricao` 0 e 1 com `restricao` 1. `so_andar`: o fator vale só nos envs da tarefa ANDAR; nos outros, 1."""
    f = torch.full((env.num_envs,), balanceador.no_degrau(ini, 1.0, env.v3_cur.restricao(), 1), device=env.device)
    if so_andar:
        f = torch.where(env.v3_tarefa == TAREFAS.index("ANDAR"), f, torch.ones_like(f))
    return f
```

### 6.2 Termos

- Em `vel_junta`, `acel_junta`, `energia`, `desvio_bracos`, `desvio_cintura`, `desvio_quadril` e `giro_xy`: o
  `return _fim(env, tarefas, X)` vira `return _fim(env, tarefas, X) * por_restricao(env, R.estilo_inicio)`.
- Em `cerca_giro`, `cerca_tronco`, `cerca_com`, `cerca_torque` e `cerca_pelve`: o `return _fim(env, tarefas, X)`
  vira `return _fim(env, tarefas, X) * por_restricao(env, 0.0, so_andar=True)`.
- Não mude `cerca_sola`, `cerca_recuo` nem os outros termos.

### 6.3 σ₀ do giro em `_rastreio`

No ramo `else` (angular), `sigma_zero = R.sigma_zero_ang` vira
`sigma_zero = balanceador.no_degrau(R.sigma_zero_ang_inicio, R.sigma_zero_ang, env.v3_cur.restricao(), 1)`.

### 6.4 Pelve no P4, em `distancia_do_alvo`

Hoje o P4 usa `d_caixa`. O fecho do P4 exige também a pelve a `tol_pelve_p4` (3 cm), e nenhum termo puxa a
pelve abaixo de 8 cm. A distância do P4 passa a somar o erro da pelve, na escala das duas tolerâncias:

```python
    d_p4 = torch.hypot(d_caixa, dpelve * (k.tol_caixa_p4 / k.tol_pelve_p4))
    d = torch.where(_em(indice, _PALMA_ALVO), d_palma,
                    torch.where(indice == _P4, d_p4, torch.where(_em(indice, _CAIXA_ALVO), d_caixa, d_pose)))
```

Atualize a docstring (P4: caixa e pelve). O `dpelve` já chega pelos dois chamadores (`_phi_env` e
`comando.py`, que calcula o σ); confira que o `dpelve` do `comando.py` no P4 é a altura da pelve menos a
referência (`pelve_de_pe` e `pelve_com_caixa` valem 0,789 m nos três tamanhos).

A mesma conta existe em `g1_v3/parte0/item1_phi.py`, função `_distancia`, e os testes
`test_recompensas_e.py::TestPassoAPassoComItem1` comparam as duas. Aplique lá a mesma regra:

```python
    d_p4 = np.hypot(d_caixa, (tr["pelve"] - refs["pelve_pe"]) * (K.tol_caixa_p4 / K.tol_pelve_p4))
    return np.where(np.isin(idx, PALMA), d_palma,
                    np.where(idx == NOMES.index("P4"), d_p4, np.where(np.isin(idx, CAIXA), d_caixa, d_pose)))
```

e atualize a docstring dela. Não mude mais nada nesse arquivo.

### 6.5 Deriva do parado em `comando.py`

- `sucesso_do_segmento` ganha o parâmetro final `deriva_pos_m=None`. Dentro:
  `lim_pos = k.deriva_pos_m if deriva_pos_m is None else deriva_pos_m`, e o `deriva_pos < k.deriva_pos_m` vira
  `deriva_pos < lim_pos`.
- Em `_fecha`, a chamada passa
  `deriva_pos_m=balanceador.no_degrau(k.deriva_pos_inicio, k.deriva_pos_m, env.v3_cur.restricao(), 1)`.
- Não mude a chamada de `g1_v3/mede_deriva.py`.

### 6.6 Testes

- `por_restricao`: com `restricao` 0 e `ini` 0,25 o fator é 0,25 em todo env; com `restricao` 1 é 1; com
  `so_andar` e `ini` 0, o env do ANDAR recebe 0 e o do PEGAR recebe 1.
- Um termo de estilo (`vel_junta`) e uma cerca (`cerca_torque`): o valor com `restricao` 0 é o valor de
  antes vezes o fator.
- `_rastreio` angular sem comando: com `restricao` 0 o σ₀ é 0,5; com 1 é 0,15.
- `distancia_do_alvo` no P4: com `d_caixa` 0 e `dpelve` 0,03 a distância é 0,05; com `dpelve` 0 é o `d_caixa`;
  nos outros alvos o valor não muda.
- `sucesso_do_segmento`: deriva de 0,20 m no parado passa com `deriva_pos_m=0.25` e falha sem o parâmetro.
- Os envs falsos dos testes existentes ganham `v3_cur` com `restricao()`, `num_envs`, `device` e `v3_tarefa`
  onde o termo passou a usá-los.

Commit: `feat(v3): restrições pelo progresso nos termos, pelve no P4 e deriva do parado`

## 7. Tarefa T4: normalizador e vantagem (`algoritmo.py`, `runner.py`)

### 7.1 Escala fixa por termo

- `runner._FIXOS` vira um dict `nome → desvio`:

```python
_FIXOS = {"tarefa": 1.0, "mapa": 1.0, "alvo_macro": 1.0, "niveis": 1.0, "degrau": 1.0, "massa": 1.0,
          "tarefa_critico": 1.0, "caixa": 0.1, "caixa_verdade": 0.1, "alvo_verdade": 0.1, "referencia": 0.1,
          "forca_caixa": 0.3}
```

  Comentário: os termos da caixa valem zero enquanto só há ANDAR; com a estatística empírica eles entrariam
  com ganho de 16,7 na abertura da fatia B. Os desvios vêm da medida no model_11200.
- `runner._fixa_obs`: a lista de cada grupo passa a guardar pares `(slice(i, i + n), _FIXOS[nome])`.
- `algoritmo.fixa_normalizador(nz, fatias, piso)`: `fatias` é uma lista de pares `(fatia, desvio)`. Para cada
  par: `nz._mean[..., sl] = 0.0`, `nz._std[..., sl] = dv`, `nz._var[..., sl] = dv * dv`.
- `PPOPorTarefa.fatias_fixas` continua `{"actor": [...], "critic": [...]}`, agora com pares.

### 7.2 Vantagem por tarefa e alvo

- `PPOPorTarefa`: atributo de classe novo `fatia_alvo = None` (a fatia do one-hot `alvo_macro` no crítico).
- `runner._fixa_obs`: no grupo `critic`, quando o termo é `"alvo_macro"`, guarde
  `self.alg.fatia_alvo = slice(i, i + n)`.
- `compute_returns`: depois do `assert` do one-hot da tarefa, o grupo passa a ser:

```python
        tarefa = bloco.argmax(-1)
        grupo = tarefa
        if self.fatia_alvo is not None:
            # 11/11: um grupo por tarefa e alvo, como o g1_limpo normalizava por elo interno. Com um grupo por
            # tarefa, o P4 tinha 40 % das amostras do PEGAR e 6,7 % da magnitude da vantagem. O ANDAR fica em um
            # grupo só (o 0, que o `normaliza_por_grupo` trata como ANDAR).
            alvo = st.observations["critic"][..., self.fatia_alvo].argmax(-1)
            n_alvos = self.fatia_alvo.stop - self.fatia_alvo.start
            grupo = torch.where(tarefa == ANDAR, torch.zeros_like(tarefa), 1 + tarefa * n_alvos + alvo)
```

  e a chamada vira `normaliza_por_grupo(crua.reshape(-1), grupo.reshape(-1))`.
- Não mude `normaliza_por_grupo`.

### 7.3 Testes

- `fixa_normalizador` com par `(slice, 0.1)`: média 0, desvio 0,1 e variância 0,01 na fatia; a entrada 0,1
  sai como ~0,909 (`0,1 / (0,1 + 0,01)`).
- `_fixa_obs`: os pares têm o desvio do `_FIXOS`; a `fatia_alvo` é a do termo `alvo_macro` do crítico.
- Grupo: com `fatia_alvo` definida, duas amostras do PEGAR com alvos diferentes caem em grupos diferentes;
  duas do ANDAR com alvos diferentes caem no grupo 0; sem `fatia_alvo` o grupo é a tarefa.
- Ajuste os testes de 11/11 que passavam `slice` puro.

Commit: `feat(v3): escala fixa dos termos da caixa no normalizador e vantagem por tarefa e alvo`

## 8. Tarefa T5: métrica do P4 (`metricas.py`)

Classe nova, logo depois de `tempo_no_alvo`, no padrão de `phi_medio_do_alvo`:

```python
class erro_p4:
    """Erro médio no P4, por episódio, em metros (11/11): vertical e lateral da caixa ao alvo, e da pelve à
    referência. Sem peso. `Episodio/erro_p4_vertical`, `_lateral` e `_pelve`; o `__call__` devolve "teve o P4"."""

    def __init__(self, cfg, env):
        self.env = env
        self.soma = torch.zeros(env.num_envs, 3, device=env.device)
        self.n = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env) -> torch.Tensor:
        robo = env.scene["robot"]
        ok = (env.v3_alvo_idx == P4) & _com_caixa(env)
        e = env.scene[CAIXA].data.root_link_pos_w - recompensas.alvo_da_caixa_w(env, torch.full_like(env.v3_alvo_idx, P4))
        yaw = recompensas.estimador.yaw(robo.data.root_link_quat_w)
        lateral = (-torch.sin(yaw) * e[:, 0] + torch.cos(yaw) * e[:, 1]).abs()
        pelve = (env.scene["altura_pelve"].data.heights[:, 0] - env.v3_ref.pelve_com_caixa(env.v3_meia)).abs()
        erro = torch.stack([-e[:, 2], lateral, pelve], dim=-1)
        self.soma = self.soma + torch.where(ok[:, None], erro, torch.zeros_like(erro))
        self.n = self.n + ok.float()
        return (self.n > 0).float()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        media = self.soma / self.n.clamp(min=1.0)[:, None]
        for i, nome in enumerate(("vertical", "lateral", "pelve")):
            _loga_episodios(self.env, "erro_p4_" + nome, media[:, i], self.n > 0, env_ids)
        self.soma[env_ids] = 0.0
        self.n[env_ids] = 0.0
```

- Importe `P4` de `g1_v3.alvos` (a linha de import já traz `P0, P1, P2, P3`).
- Se `recompensas` não expõe `estimador`, importe `from g1_v3 import estimador` no topo e use `estimador.yaw`.
- Confira a forma de `env.v3_ref.pelve_com_caixa(env.v3_meia)`: tem de ser (E,), no device do env (`.to(...)`).
- `__all__`: acrescente `"erro_p4"`.
- `termos`: `m["erro_p4"] = MetricsTermCfg(func=erro_p4, reduce="last")`.

Teste (em `test_metricas_h.py`): com um env falso de 2 envs, um no P4 e um no P0, o `reset` loga os três
nomes só para o env do P4, com os valores esperados de um caso de conta à mão.

Em `specs/g1-v3.md`, no fim da §19 (antes de "### 19.1"), acrescente uma linha: "Desde 11/11, o treino único
segue `specs/g1-v3-treino-unico.md`: restrições pelo progresso, currículo pela média, fatia em rampa, pelve no
P4, vantagem por tarefa e alvo e escala do punho por junta."

Commit: `feat(v3): métrica do erro do P4 e ponteiro da spec do treino único`

## 9. Tarefa T6: correções da revisão (só comentários e testes)

A revisão do lote não achou defeito no código de produção. Ela achou comentários falsos e testes fracos.
Esta tarefa NÃO muda lógica de produção: só o texto de comentários e os testes.

### 9.1 Comentários

| Arquivo | Onde | Texto certo |
|---|---|---|
| `g1_v3/knobs.py` | comentário de `deriva_pos_inicio` | o limite DESCE de 0,25 m (restricao 0) a `deriva_pos_m` (restricao 1) |
| `g1_v3/knobs.py` | comentário de `sigma_zero_ang_inicio` | o σ₀ DESCE de 0,5 (restricao 0) a `sigma_zero_ang` (restricao 1) |
| `g1_v3/knobs.py` | comentário de `atraso_portao` | 11/11: o teto do atraso é `ceil(atraso_portao[1] · restricao)`; antes ligava quando o PEGAR passava no nível 0 |
| `g1_v3/balanceador.py` | comentário do `max_nivel` no `__init__` ("6 níveis de velocidade") | 5 níveis de velocidade |
| `g1_v3/balanceador.py` | comentário do `_trava_dr` ("o mesmo fecho que abre a fatia B liga o vies") | desde 11/11 a fatia B abre pela média e o "vies" exige o reto e o parado no portão: o "vies" pode ligar depois da fatia B |
| `g1_v3/env_cfg.py` | comentário do evento `atraso` (teto de 1 passo) | o teto inicial é `atraso_inicio[1]`, que vale 0 desde 11/11 |
| `g1_v3/observacoes.py` | docstring do topo (o evento `atraso` baixa o teto para 1) | o teto inicial é `atraso_inicio[1]`, que vale 0 desde 11/11 |
| `g1_v3/testes/test_comando.py` | comentário "o limite do parado sobe de 0,25 a 0,10" | desce |

### 9.2 Testes

1. `test_balanceador.py`, o teste da rampa (`test_rampa_da_fatia_b_entra_com_teto_e_sobe_por_degrau`): hoje ele só
   afirma "no máximo". Monte o caso com o déficit da fatia B alto (a fração sem a rampa passa de 0,5), e afirme com
   `assertAlmostEqual`: no fecho da abertura a fração da B é 0,05 e a do ANDAR é 0,95; 12 fechos depois a da B é
   0,07 e a do ANDAR é 0,93; a fração da fatia C fechada é 0. Teste novo: quando a fatia C abre, a fração dela
   é 0,05 no fecho da abertura.
2. `test_metricas_h.py`, `ErroP4Test`: teste novo com yaw de 90° (quaternion wxyz `(0.70710678, 0, 0, 0.70710678)`):
   com `e = (0,10; 0; −0,20)` no mundo, o erro lateral é 0,10 (com yaw 0 seria 0). Teste novo com DOIS passos de
   `__call__` no P4 com erros verticais 0,20 e 0,40: a média logada é 0,30. Teste novo: `reset` só do env 0 não zera
   a soma do env 1.
3. `test_comando_g.py` (ou o arquivo onde está o env falso do `_fecha`): teste novo que chama o `_fecha` com um
   segmento parado de deriva 0,20 m: com `restricao=lambda: 0.0` o segmento registra sucesso; com
   `restricao=lambda: 1.0` registra falha. Use o mesmo env falso do teste `FechoDoSegmentoNoTerreno`.
4. `test_recompensas_e.py`, `TestDistanciaDoP4`: teste novo: no P3 (índice 3) e no B1 (índice 5), com `dpelve` 0,10
   a distância é igual ao `d_caixa` (a pelve só entra no P4).
5. `test_curriculo_h.py`, `test_log_traz_degraus_degrau_global_atraso_e_mu`: afirme também que o log tem a chave
   `restricao` com o valor de `cur.restricao()`.

Se um teste novo desta lista FALHAR contra o código de produção atual, não mude o código de produção: pare e relate
o caso, com os números.

Commit: `test(v3): reforço dos testes do treino único e comentários corrigidos`
