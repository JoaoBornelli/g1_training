# Plano — fazer o treino seguir o currículo especificado

**Data:** 2026-08-20
**Responsável:** João Bornelli
**Estado do código:** `exp/g1-poc` em `c4e87bb`, árvore limpa, `smoke.py` 52 ok / 0 falhas
**Referências:** `ESPECIFICACAO-g1_poc.md` §7, §8, §9, §10, §17 · `docs/adrs/0001-*.md`

---

## 0. Objetivo

O treino tem de progredir nos três eixos do currículo (altura, carga, rotação), abrir as
cadeias de elos, e aprender a andar. Hoje ele não faz nenhuma das três coisas, e as causas
são distintas.

Este plano é a lista fechada do que muda, em que ordem, e qual medida abre o passo seguinte.

---

## 1. Comparativo: o especificado × o implementado

### 1.1 §10.1 — o nível por env

| coluna da tabela de células | especificado | implementado |
|---|---|---|
| regra de promoção `clamp(nivel ± 1, 0, 6)` | sim | **sim** — `curriculo.py:50` |
| topo da prateleira 0,55 → 0,04 por nível | sim | **não** — fixo em 0,55 |
| carga 1 → 5 kg por nível | sim | **não** — fixa em 1 kg |
| rotação 0° → 180° por nível | sim | **não** — fixa em 0° |
| cadeias e frações por nível | sim | **não** — só `pegar` |

`env.poc_nivel` é escrito por `nivel_caixa` e **não é lido por ninguém**. As três colunas
estão presas nos parâmetros do `env_cfg`:

- `env_cfg.py:346` passa `topo_piso = kc.prateleira_topo_teto` — a faixa é degenerada em 0,55
- `env_cfg.py:360` passa `faixa_kg = kd.carga_kg = (1.0, 1.0)`
- `comando.py:179` escreve `self._ang[env_ids] = 0.0`

### 1.2 §7 — os elos e as cadeias

| elo | especificado | implementado |
|---|---|---|
| `pegar` | 4 condições, 1,0 s | **sim** — `comando.py:236-249` |
| `reorientar` | 2 condições, 0,5 s | não |
| `carregar` | 6 s, alvo no frame da base | não |
| `botar` | 3 condições, 0,5 s | não |
| troca de elo dentro do episódio (§7.5) | sim | não |
| prateleira se move no fecho (§7.3) | sim | não |

Os números dos elos que faltam **já estão** em `knobs.py`: `Alvo.peito_b`, `Alvo.botar_x/y`,
`Alvo.botar_topo_piso/teto`, `Tolerancia.sustenta_outros_s`, `Tolerancia.carregar_s`,
`Tolerancia.fracao_apoio_botar`. O desenho existe; o código não.

### 1.3 §10.2 — a locomoção

O cronograma está correto desde `c4e87bb` (degraus em 8000 e 12000). O que não está
resolvido é a fatia de dados:

```
fatia de locomoção = (0,30 × 24 passos) / (0,30 × 24 + 0,70 × 961) = 1,06%
```

A métrica que mede isto **já existe e já apontava o problema**: `frac_manipula`
(`curriculo.py:47`) é `env.poc_manipula.float().mean()` sobre **todos** os envs, portanto
ela é a fração populacional — que para um processo em regime é a mesma coisa que a fatia de
transições. Ela lia ≈ 0,99 e subia. O nome dela sugere o sorteio, e o sorteio é
`frac_locomocao = 0,30` no knob. Essa ambiguidade de nome atrasou o diagnóstico.

O laço se auto-sustenta: episódio curto → pouco dado → não aprende → episódio curto. Os
cinco termos de marcha são gateados por comando, portanto valem **exatamente zero** nos 99%
dos passos em que o twist está zerado.

### 1.4 As correções já aplicadas e ainda não medidas em treino

| commit | mudança | medida em treino? |
|---|---|---|
| `96761c2` | `unload` — a ponte do platô do grasp | sim (it 3080: 1º `episode_success` do projeto) |
| `d4c0726` | degrau do `hinge` 3000 → 10000 | não |
| `c3c306e` | push 1–3 s → 10–20 s | **não** — medido só na sonda |
| `c4e87bb` | twist volta ao estágio 0 | **não** |

A sonda, no `model_5100` com push em 10–20 s, mediu `episode_success = 0,750` e
`fecha_todas = 63,4%` em 8 envs. Ou seja: em condição de play a política **fecha** o elo. O
treino, na it 5217, media `episode_success ≈ 0,006`. Parte dessa distância são duas
correções que ainda não rodaram.

---

## 2. O gargalo raiz — corrigido em relação ao diagnóstico de ontem

A promoção de nível só sobe com `episode_success`, e o sucesso exige `de_pe`
(pelve ≥ 0,65 m). Medido na it 5217:

```
pelve_z = 0,6345      fecha_de_pe = 0,2388      erro_angulo_deg = 14,2°
```

`fecha_de_pe` é `de_pe × valida`, portanto 0,2388 / 0,70 = **34% dos envs de manipulação
estão de pé**. A pelve está **1,5 cm** abaixo do critério, e não muito abaixo.

### 2.1 A §9.4 promete o gradiente que levanta o robô, e ele não existe

A §9.4 diz: a demanda cai a zero quando a caixa chega ao alvo, o regime volta para
`standing`, e a `pose` puxa o robô para a pose de pé.

**Isto não funciona, e o motivo é o valor de `std_standing`.** O G1 usa
`std_standing = {".*": 0.05}` (`mjlab/tasks/velocity/config/g1/env_cfgs.py:107`) — 0,05 rad
para **toda** junta, braços incluídos.

Com a caixa segurada a 0,82 m, quatro juntas de braço ficam a ≈ 0,7 rad do default:

| regime | contribuição do braço | `pose` |
|---|---|---|
| `std_manipulando` (ombro/cotovelo σ = 1,00) | 4 × 0,49/1,00 = 1,96, sobre 29 juntas | **≈ 0,93** |
| `standing` (σ = 0,05) | 4 × 0,49/0,0025 = 784, sobre 29 juntas | **≈ 0** |

Trocar de regime com a caixa na mão **destrói** o termo `pose` inteiro (peso 1,0). O que a
§9.4 chama de "gradiente que levanta o robô" é, na prática, um penhasco que **cobra 0,93/s
por terminar a tarefa**.

E o penhasco está exatamente onde o robô parou. Com `peso_dist = 10`, `peso_ang = 6` e
`limiar = 1,5`:

```
demanda medida = 10 × 0,0106 + 6 × 0,248 rad = 0,11 + 1,49 = 1,60
```

O regime só troca com `demanda < 1,5`, ou seja com `Δθ < 13,3°`. A régua de sucesso aceita
`Δθ < 20°`. **Existe uma banda de 13,3° a 20° em que o elo fecha e o penhasco não é
pisado** — e o ângulo medido, 14,2°, está dentro dela.

Verificável no log sem treinar nada novo: `Metrics/postura_frac_manipulando` tem de estar
≈ 1,0 (o robô praticamente nunca entra em `standing`).

### 2.2 Consequência para o desenho

Ontem eu propus alinhar a demanda com a régua (demanda = 0 quando as condições fecham).
**Isso seria pior:** poria o penhasco de −0,93/s exatamente no ponto de sucesso.

O desenho correto é o inverso:

1. **Tirar o penhasco.** O regime de postura passa a ser escolhido pela FORMA do episódio
   (`caixa_valida`), e não pela demanda residual. Um episódio de manipulação usa
   `std_manipulando` de ponta a ponta. Os três regimes de velocidade continuam intactos nos
   episódios de locomoção, portanto a marcha validada não muda.
2. **Pagar pela pelve, de forma contínua e explícita.** Um termo novo, com rampa, gateado
   por preensão bimanual. É o idioma que o `unload` já validou: quando falta gradiente numa
   coordenada, entra **um** termo que tem gradiente nessa coordenada.

As duas mudanças são **uma só** mudança de mecanismo e vão no mesmo bloco. Separá-las não
produz informação: sem (1) o termo novo luta contra −0,93/s; sem (2) nada levanta o robô.
Esta é uma exceção declarada à regra "uma mudança por bloco" da §17, e a razão está escrita.

---

## 3. Defeitos que aparecem só quando as cadeias entrarem

Três incompatibilidades reais entre o código de hoje e a §7. Nenhuma se manifesta agora,
porque só existe o elo `pegar`.

| # | defeito | onde |
|---|---|---|
| 1 | **`unload` e `botar` são opostos.** `unload = 1 − F_apoio/m·g` paga por descarregar; o fecho do `botar` exige `F_apoio ≥ 0,8·m·g`. Os dois ativos ao mesmo tempo pagam para não botar. | `recompensas.py:241` × §7.2 |
| 2 | **`caixa_largada` usa o sucesso da cadeia, não do `pegar`.** O gate é `poc_success > 0.5`, que só vira 1 no ÚLTIMO elo. Durante `carregar`, largar a caixa não termina o episódio. | `terminacoes.py:51` |
| 3 | **O gate `nao_caiu` do `unload` referencia `env.poc_topo`.** Quando a prateleira se move (+5 m no `carregar`, topo novo no `botar`), a referência muda e o gate passa a mentir. | `recompensas.py:252` |

Mais três knobs mortos, que enganam quem for mexer:

| knob | por que está morto |
|---|---|
| `CaixaAlvoCommandCfg.frac_locomocao` | `comando.py:305` declara, `_resample_command` lê `env.poc_manipula`. Quem ajustar aqui não muda nada. |
| `DR.push_janela_livre_s` | já marcado como morto no próprio knob |
| `Treino.lr_warm_start` | ninguém lê; o ADR-0001 declarou e nunca aplicou |

---

## 4. O grafo de dependências

```
[F2] o robô fica de pé            ← destrava episode_success
      │
      ├── sem isto, o nível NUNCA sobe, e [F1] fica inerte
      ▼
[F1] tabela de células            ← no-op comprovado no nível 0
      │
      ▼
[F3] o andar                      ← independente de F1/F2, mas obrigatório antes de `carregar`
      │
      ▼
[F4] as cadeias                   ← exercitadas só do nível 3 para cima
      │
      ▼
[F5] refino de pose               ← passo 6 da §17, o último
```

`F1` entra **junto** com `F2` porque no nível 0 a célula é, número por número, a cena de
hoje (topo 0,55 / carga 1 kg / ângulo 0°). Todos os envs começam no nível 0. Portanto `F1`
não é uma segunda mudança mensurável naquele bloco: ela é a fiação que faz `F2` render
progressão de nível no instante em que funcionar, sem gastar um bloco só para isso.

---

## Fase 1 — a tabela de células (§10.1)

### Objetivo

`env.poc_nivel` passa a selecionar altura, carga e rotação. No nível 0 a cena não muda.

### Mudanças

#### 1. `g1_poc/knobs.py`

```python
@dataclass
class Celulas:
    """§10.1 — a célula de cada nível. O teto do topo e o piso da carga não mudam."""
    topo_min:    tuple = (0.55, 0.45, 0.30, 0.15, 0.04, 0.04, 0.04)
    carga_max:   tuple = (1.0,  2.0,  3.0,  4.0,  5.0,  5.0,  5.0)
    ang_max_deg: tuple = (0.0,  0.0,  0.0,  45.0, 90.0, 180.0, 180.0)
    # o nível 6 gira no eixo horizontal (§10.1) — Risco 1 da §19, opcional.
    # Fica igual ao 5 até haver mão.
```

Entra em `Knobs` como `celulas`.

#### 2. `g1_poc/eventos.py`

`reset_cena` troca `topo_piso: float` por `topo_min_por_nivel: tuple` e resolve por env:

```python
nivel = getattr(env, "poc_nivel", None)
if nivel is None:
    piso = torch.full((n,), topo_min_por_nivel[0], device=dev)
else:
    tabela = torch.tensor(topo_min_por_nivel, device=dev)
    piso = tabela[nivel[env_ids]]
topo = piso + (topo_teto - piso) * torch.rand(n, device=dev)
```

`carga_caixa` faz o mesmo com `carga_max_por_nivel`, mantendo o piso em `massa_base`.

⚠ O `clamp(min=topo_piso)` de `eventos.py:66` passa a ser `clamp(min=piso)`, por env.

#### 3. `g1_poc/comando.py`

`_resample_command` sorteia o ângulo da célula em vez de escrever 0,0:

```python
ang_max = torch.deg2rad(self._ang_max_tabela[nivel[env_ids]])
self._ang[env_ids] = (2.0 * torch.rand(n, device=self.device) - 1.0) * ang_max
```

A tabela entra no `CaixaAlvoCommandCfg` como `ang_max_deg_por_nivel`.

#### 4. `g1_poc/env_cfg.py`

Passa `k.celulas.topo_min` / `carga_max` / `ang_max_deg` para os três lugares. O
`prateleira_topo_teto` continua sendo o teto em todos os níveis.

#### 5. `g1_poc/smoke.py` — seção 14 nova

### Critérios de aceite

#### Automático

- [ ] `python -m g1_poc.smoke` — 0 falhas, e a contagem sobe para ≥ 57
- [ ] o smoke prova o **no-op no nível 0**: com `poc_nivel = 0` em todos os envs, `poc_topo`
      cai em `[0,53 ; 0,57]` (0,55 ± jitter), `poc_massa == 1,0` e `cmd._ang == 0`
- [ ] o smoke prova a **progressão**: forçar cada nível de 0 a 6 e conferir
      `poc_topo.min() ≥ topo_min[n] − jitter`, `poc_massa.max() ≤ carga_max[n]`,
      `|_ang|.max() ≤ ang_max_deg[n]`
- [ ] o smoke prova que **a promoção muda a cena**: nível 0 dá topo ≈ 0,55 / carga 1,0;
      nível 4 dá topo ≤ 0,10 / carga ≥ 4,0

#### Manual

- [ ] `python -m g1_poc.play --pegar` com o nível forçado em 4: a laje a 0,04 m apoia no
      chão e a caixa nasce em cima dela (passo 1 da §18)

---

## Fase 2 — o robô fica de pé

### Objetivo

Tirar o penhasco de −0,93/s do fim da tarefa e pagar pela altura da pelve com uma rampa.
Isto é o que destrava `episode_success`, e portanto a promoção de nível.

### Mudanças

#### 1. `g1_poc/postura.py` — o regime pela FORMA, não pela demanda residual

```python
# o 4º regime vale no episódio de manipulação inteiro. A troca por demanda
# residual era um PENHASCO: com a caixa na mão, `standing` (σ = 0,05 em toda
# junta) leva `pose` de 0,93 para ≈ 0, portanto cobrava 0,93/s por terminar a
# tarefa. Medido: demanda no fecho = 1,60 contra limiar 1,5, ou seja o robô
# ficava do lado lucrativo do penhasco, com Δθ preso em 14,2°.
m_manip = (bit > 0.5).float().unsqueeze(1)
```

`peso_dist`, `peso_ang` e `limiar` saem de `Postura` e do `env_cfg`. O log
`Metrics/postura_demanda_caixa` fica (é diagnóstico útil); `postura_frac_manipulando` passa
a ser a fração de manipulação.

#### 2. `g1_poc/recompensas.py` — termo novo `postura_ereta`

```python
def postura_ereta(env, command_name, palm_sensors, pelve_min, rampa) -> torch.Tensor:
    """Rampa contínua na altura da pelve, gateada por preensão bimanual.

    A régua exige pelve ≥ `pelve_min` (§7.2, condição 3) e NENHUM termo pagava por
    ela: o `upright` mede inclinação, e o `precise_pos` — o maior termo do pacote —
    é maximizado agachando, porque agachar encurta caixa→alvo.

    O gate de preensão é o que torna o termo compatível com a pega baixa: antes de
    ter a caixa na mão o robô PRECISA agachar (a prateleira desce a 0,04 m no nível
    4), e ali o termo vale zero. Depois de pegar, subir paga.

    A rampa satura em `pelve_min`. Acima disso não há prêmio: a régua não pede mais.
    """
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    fracao = ((z - (pelve_min - rampa)) / rampa).clamp(0.0, 1.0)
    pega = ...  # mesma preensão bimanual do `unload`
    return fracao * pega.float() * _valida(env, command_name)
```

Knobs: `Recompensa.postura_ereta = 1.0`, `Recompensa.postura_ereta_rampa = 0.20`
(a rampa vai de 0,45 m a 0,65 m; a pelve medida, 0,6345, entra em 92% dela — logo há
gradiente exatamente onde o robô está).

#### 3. `g1_poc/env_cfg.py`

Registra o termo. **A contagem de recompensas vai de 20 para 21.**

#### 4. `g1_poc/smoke.py`

- contagem 20 → 21, e `postura_ereta` na lista de termos de tarefa
- o teste de bit=0 passa a cobrir `postura_ereta`
- teste novo: com as palmas longe da caixa, `postura_ereta == 0` mesmo com o robô de pé
- teste novo: `postura_frac_manipulando` é 1,0 nos envs de manipulação e 0,0 nos de
  locomoção

#### 5. `ESPECIFICACAO-g1_poc.md`

- §8 header: 20 → 21 termos; a tabela da §8.2 ganha `postura_ereta` com uma §8.2.3
- **§9.4 é retificada.** O texto atual afirma um gradiente que a aritmética de
  `std_standing = 0,05` nega. Retificar, com os dois números da tabela da §2.1 deste plano.
- §9.3: registrar que a demanda não escolhe mais o regime, e por quê

### Critérios de aceite

#### Automático

- [ ] `python -m g1_poc.smoke` — 0 falhas, 21 termos
- [ ] `python -m g1_poc.sonda --envs 16` no `model_5100` (checkpoint de antes da mudança):
      registra a linha de base de `fecha_de_pe`, `pelve_z` e `episode_success`

#### Manual

- [ ] bloco de treino, warm-start do melhor checkpoint com `learning_rate = 5e-4`:
      `Episode_Reward/postura_ereta` **sai de zero** e `Metrics/…/pelve_z` sobe acima de 0,65
- [ ] `fecha_de_pe / frac_manipula` passa de 0,34 para > 0,60
- [ ] `episode_success` passa de 0,006 para > 0,10
- [ ] `Curriculum/nivel_medio` **sai de zero** — é a primeira prova de que o eixo do nível
      funciona ponta a ponta
- [ ] a sonda, no checkpoint novo, não mostra queda de `fecha_todas` em relação à linha de
      base

**Portão:** o passo seguinte só abre com `nivel_medio > 0`. Se `episode_success` subir e o
nível não seguir, o defeito está na Fase 1, não aqui.

---

## Fase 3 — o andar

### Objetivo

Romper o laço de 1,06%. Duas mudanças: uma de receita de treino, uma de código.

### Mudanças

#### 1. Receita — um bloco com a fatia invertida

`Episodio.frac_locomocao` vai a **0,85** por um bloco, e volta a 0,30 depois. Não é 1,00
porque 0,15 de manipulação mantém o `pegar` ensaiado; com o episódio de andar em 24 passos
isso já dá ≈ 12% de fatia de locomoção (11× a de hoje), e a fatia cresce sozinha à medida
que o episódio de andar se alonga.

Zero código novo: `frac_locomocao` é o parâmetro do termo `forma`.

⚠ Voltar a 0,30 depois **não** desfaz o ganho. A fatia é governada pelo tempo de vida do
episódio: com o robô andando, o episódio de locomoção chega perto dos 20 s, e o sorteio
70/30 entrega os 30% de transições que ele promete.

#### 2. Código — gate por competência no cronograma do twist (dívida da §10.3)

O cronograma por passo global já saiu de fase duas vezes (`hinge` e `twist`). O gate fecha
a classe do problema, e não só esta instância.

`g1_poc/curriculo.py`, termo novo:

```python
def twist_por_competencia(env, env_ids, command_name, velocity_stages,
                          duracao_min_frac, ema):
    """Avança o estágio do twist só quando o robô SUSTENTA o teto atual.

    O sinal de competência é a duração do episódio de LOCOMOÇÃO. Ele é direto: um
    robô que não anda cai em 24 passos, e um que anda chega ao time_out. É também a
    grandeza que governa a fatia de dados, portanto é a que precisa subir.

    ⚠ Este termo tem de vir ANTES de `forma` no dict de currículo. O `sorteia_forma`
    sobrescreve `poc_manipula`, e aqui precisamos da forma do episódio que ACABOU.
    ⚠ `episode_length_buf[env_ids]` só é zerado no fim do `_reset_idx`, portanto
    aqui ele ainda vale a duração final.
    """
```

Regra de avanço: `estagio` sobe quando `common_step_counter >= stage["step"]`
**e** `ema_duracao_loco >= duracao_min_frac × max_episode_length`. Guardado em
`env.poc_estagio_twist`, monotônico. Knobs: `Cronograma.twist_duracao_min_frac = 0.60`,
`Cronograma.twist_ema = 0.99`.

O termo `twist_ranges` do `env_cfg` troca `mdp.commands_vel` por este.

#### 3. Renomear a métrica que enganou

`frac_manipula` → `frac_manipula_pop`, com o comentário de que ela é a fração
**populacional** (≈ fatia de transições) e que o sorteio é `frac_locomocao`. É o número que
gateia esta fase.

### Critérios de aceite

#### Automático

- [ ] `python -m g1_poc.smoke` — 0 falhas; teste novo de que `twist_por_competencia` vem
      antes de `forma` no dict de currículo
- [ ] teste novo: com `ema` baixa, o estágio não avança mesmo com `common_step_counter`
      acima do degrau

#### Manual

- [ ] bloco com `frac_locomocao = 0,85`: `Metrics/peak_height_mean` passa de 2,7 mm para
      > 50 mm
- [ ] `frac_manipula_pop` cai abaixo de 0,80
- [ ] a duração média do episódio de locomoção passa de 24 passos para > 400
- [ ] `Curriculum/lin_vel_x_max` **continua em 1,0** enquanto a duração não passar de 60%
      (prova de que o gate segura)
- [ ] de volta em `frac_locomocao = 0,30`, a sonda mostra o `pegar` preservado —
      `fecha_todas` dentro da variação amostral da linha de base da Fase 2

---

## Fase 4 — as cadeias (§7)

### Objetivo

Os elos `reorientar`, `carregar` e `botar`, a troca de elo dentro do episódio, e a
prateleira que se move. É o maior item do plano, e vai em três entregas.

### Estado novo, por env

| buffer | uso |
|---|---|
| `env.poc_cadeia` | 0 `pegar` · 1 `reorientar→pegar` · 2 `pegar→carregar` · 3 `pegar→botar` |
| `env.poc_elo` | índice do elo ativo dentro da cadeia |
| `env.poc_pegou` | o elo `pegar` fechou (≠ a cadeia toda ter fechado) |

A cadeia é sorteada no reset, das frações da §10.1:

| nível | `pegar` | `reorientar→pegar` | `pegar→carregar` | `pegar→botar` |
|---|---|---|---|---|
| 0–2 | 1,00 | — | — | — |
| 3 | 0,50 | 0,50 | — | — |
| 4 | 0,40 | 0,25 | 0,35 | — |
| 5–6 | 0,30 | 0,20 | 0,25 | 0,25 |

Entra em `knobs.Celulas` como `cadeias: tuple[tuple[float, ...], ...]`.

### A máquina de elo

Vive no `CaixaAlvoCommand._update_command`, e **não** usa `_resample`: o `_resample` zera
`episode_success`. Um método próprio, `_avanca_elo(ids)`, faz:

1. escreve o alvo do elo novo (regra por elo, ver abaixo)
2. zera `_sustenta[ids]`
3. recalcula `dist_inicial[ids]` (é o σ do `bringing`, §8.2)
4. atualiza `poc_twist_zero[ids]` — só o `carregar` libera o twist
5. move a prateleira, quando a cadeia pede (§7.3)

| elo | alvo | fecho |
|---|---|---|
| `pegar` | mundo, sorteado uma vez | 4 condições, 1,0 s |
| `reorientar` | posição ATUAL da caixa, `dir_alvo` girado pelo ângulo da célula | 2 condições, 0,5 s |
| `carregar` | `base + peito_b`, recalculado **a cada passo** | 6 s, e `erro_pos < raio` no instante |
| `botar` | topo novo + 0,10 · x 0,30–0,40 · y ±0,12 | 3 condições, 0,5 s |

O `carregar` obriga um ramo por env no `_update_command`: alvo ancorado no mundo para três
elos, ancorado na base para um.

Movimento da prateleira no fecho do `pegar` (§7.3), com `write_mocap_pose_to_sim`:

- cadeia 2 (`carregar`): +5 m
- cadeia 3 (`botar`): topo novo sorteado em **0,30–0,80** (`Alvo.botar_topo_piso/teto`) — a
  faixa da COLOCAÇÃO, diferente da faixa da pega
- cadeias 0 e 1: não se move

⚠ A escrita acontece no `_update_command`, que roda depois do `sim.forward()` do
`step()`. A pose nova vale a partir do passo seguinte. A regra do mjlab — "não leia grandeza
derivada na mesma função que escreve estado" — é respeitada porque a máquina de elo lê pose
de **caixa e robô**, e escreve pose de **prateleira**.

### Os três defeitos da §3 deste plano, consertados aqui

1. **`unload` só vale no elo `pegar`.** Ele é a ponte do platô do grasp; nos outros elos a
   caixa já está fora da prateleira. No `botar` ele é o **oposto** da condição de fecho
   (`F_apoio ≥ 0,8·m·g`), portanto mantê-lo ligado pagaria para não botar. O gate entra como
   máscara na função, sem termo novo.
2. **`caixa_largada` passa a gatear em `poc_pegou & ~sucesso`.** Assim largar a caixa
   durante o `carregar` termina o episódio, e soltar a caixa **depois** do `botar` fechar
   não termina — o que respeitaria a §7.5, que proíbe terminar no sucesso.
3. **`env.poc_topo` é reescrito quando a prateleira se move**, para o gate `nao_caiu` do
   `unload` continuar referenciando a cena real. Com (1), o ponto é acadêmico no `carregar`
   e no `botar`, mas o buffer é observado pelo crítico (`topo_prateleira`) e tem de ser
   verdade.

### Entregas

| entrega | elo | portão da §17 |
|---|---|---|
| 4a | `reorientar` (nível 3) | `nivel_medio > 3` |
| 4b | `carregar` (nível 4) | `nivel_medio > 4` — exige a Fase 3 fechada |
| 4c | `botar` (níveis 5–6) | os cinco critérios da §0 |

### Critérios de aceite (por entrega)

#### Automático

- [ ] `python -m g1_poc.smoke` — 0 falhas; seções novas para a máquina de elo:
      o fecho do elo N escreve o alvo do elo N+1; `episode_success` **não** é zerado na
      troca; `_sustenta` **é** zerado
- [ ] teste de que o fecho do último elo trava `episode_success` e o episódio **continua**
- [ ] teste de que `unload == 0` quando o elo ativo é `botar`
- [ ] teste de que `caixa_largada` é falso antes de `poc_pegou` e depois do sucesso
- [ ] `python -m g1_poc.play --pegar` sobe sem exceção com cada cadeia forçada

#### Manual

- [ ] no `play`, com a cadeia forçada, a prateleira se move no fecho do `pegar` e a caixa
      **não** se move com ela
- [ ] a sonda mostra a cadeia fechando elo por elo, com a decomposição por elo
- [ ] bloco de treino: `nivel_medio` passa do portão da entrega

---

## Fase 5 — refino de pose

Passo 6 da §17, e só aqui. O `hinge` de −1,00 está em 10 000 iterações e o `action_rate` de
−0,25 em 3 000. Os dois têm de ser reconferidos contra o `nivel_medio` real quando a
Fase 4 fechar, e o gate por competência da Fase 3 deve ser estendido a eles — a §10.3 já
registra a dívida.

Sem critério de aceite fechado agora: ele depende da fase medida no fim da Fase 4.

---

## 5. O que este plano NÃO faz

| item | por quê |
|---|---|
| nível 6 com rotação no eixo horizontal | Risco 1 da §19; o G1 não tem mão. A célula do 6 fica igual à do 5. |
| mexer em `pelve_min = 0,65` | o critério é a régua da POC, e a decisão do usuário é que o robô **tem** que ficar de pé |
| `elliptic` / `impratio = 10` | Risco 6 da §19, depois da POC |
| inércia real da carga | `dr.body_mass` corrompe a heap; gap declarado |
| navegação (cadeia de 3 elos) | §7.4 a exclui |
| sim-to-real (latência, viés, perda de rastreio) | Riscos 7 e 8 da §19 |
| `box_shake`, `com_balance`, separar o `bringing` | ficam na reserva da §8.5, com gatilho escrito |

---

## 6. Orçamento

| fase | código | blocos de treino |
|---|---|---|
| 1 + 2 | pequeno (1 termo, 1 função, 3 fiações) | 1 |
| 3 | pequeno (1 termo de currículo) | 2 (o de 0,85 e o de volta a 0,30) |
| 4a / 4b / 4c | **grande** (máquina de elo, 3 elos, prateleira móvel) | 3 |
| 5 | pequeno | 1 |

Sete blocos, e o caminho crítico é a Fase 4.

---

## 7. Disciplina

1. Uma mudança por bloco. A exceção é a Fase 2, e a razão está na §2.2.
2. Warm-start sempre com `learning_rate = 5e-4`.
3. Nenhuma fase abre sem o portão da anterior medido.
4. A régua de campo é a `sonda.py`, e ela mede força e fato físico — não recompensa.

---

# Adendo — 20/08, pós-auditoria (4 auditorias independentes)

Quatro auditorias antes de implementar: API do mjlab, viabilidade mecânica, travas e
incentivos. Vereditos que MUDAM este plano:

## Corrigido no próprio plano

- **A justificativa da Fase 2 estava errada.** Este plano diz que "o `precise_pos`
  paga por agachar". Não paga: o alvo é ACIMA do repouso, então subir com a caixa
  aproxima do alvo. O `precise_pos` é INDIFERENTE à pelve abaixo do alvo e CONTRÁRIO
  acima (−16,2/m no fecho). O mecanismo real do platô: NADA paga pela pelve (só a
  `pose`, a ~0,73/m) e agachar custa 0,119/s. A conclusão (rampa de pelve) fica; o
  mecanismo escrito era outro.
- **Bug novo, medido: a promoção de nível usava a forma do episódio SEGUINTE.**
  `nivel` vinha depois de `forma` no dict de currículo, e `forma` sobrescreve
  `poc_manipula`. Efeito: p_up = 0,7·p (ponto fixo em 0,714, não 0,5) e um episódio
  de locomoção rebaixava o nível em 70% das vezes — **a Fase 3 (frac_locomocao 0,85)
  teria apagado a escada da Fase 2 mesmo com manipulação perfeita (teto 0,214)**.
  Conserto: ordem `twist_ranges, nivel, forma, hinge, action_rate`.
- **Bug confirmado, escopo menor: o alias `env.poc_success`.** Só a `caixa_largada`
  era cega; o `nivel_caixa` lê `cmd.episode_success` direto e nunca foi afetado.
- **T1 do auditor de travas ("a caixa afastada cai na frente do robô") foi REFUTADA
  por medição**: a caixa fica APOIADA na laje a 5 m (z = 5,099 após 120 passos, com
  5 kg). O auditor simulou sem a prateleira. `afasta_cena` fica como está.

## Acréscimos ao MACRO 1 (já no spec v2)

1. **σ do `reaching` por elo** (`max(0,20; d_inicial palma→face)`): sem isto o
   gradiente de aproximação cai 1391× do nível 0 ao 4 e os níveis 3+ viram sorte.
2. **`postura_ereta` recalibrado**: rampa em DUAS partes (longa 0,20→0,65 + fina
   0,57→0,65), peso 2,0 (2,2/m e 14,7/m), e gate extra de DESCARGA
   (F_apoio < 0,2·m·g) — sem ele, encostar e ficar de pé com a caixa apoiada
   pagaria a rampa inteira no exato platô do bloco 1.
3. **Termo `sustentacao`** (+0,5, `clamp(t/1s)`): 0,98 s e 0,00 s pagavam o mesmo, e
   o push — o único fator que degradava o sucesso — ataca exatamente o cronômetro.
4. **Gate do twist com DESCIDA** (histerese 0,8×alvo), teto de 1 degrau/12
   iterações (sem isso, um warm-start além dos degraus re-explodiria 0→2 em 0,08
   iteração), EMA sem os envs parados, EMA como tensor no device.
5. **`jitter_x` da caixa entra na célula** (0,20 → 0,08 nos níveis ≥4): a 0,04 m o
   alcance acaba em x≈0,45 e 60% dos episódios exigiriam um passo com twist zerado.
6. **Portão da Fase 2 vira `nivel_medio ≥ 1,0`** + histograma
   (`nivel_frac_0`/`nivel_frac_3mais`): "sair de zero" já vale hoje com p = 0,006
   (0,0042), e `nivel_max` satura com um env sortudo.
7. São **22 termos** de recompensa (13 + 1 + 8).

## Registrado para a MACRO 2 (as cadeias) — decisões de desenho fechadas

- **`load` para o `botar`** (+2,0, espelho do `unload`, ativo só no elo): sem ele o
  fecho do `botar` tem saldo −3,0/s antes da máscara e 0,0/s depois — sorte pura.
  E mascarar o `squeeze` no mesmo elo.
- **`caixa_largada` por cadeia**: armar em `poc_pegou`; desarmar o ramo "escapou"
  quando a CADEIA fecha (no `botar`, soltar é o objetivo); mascarar durante a
  janela de sustentação do elo terminal (0,5 s — senão 0,56 m/s de recuo de mão já
  termina o episódio). ⚠ O gate que este plano propunha (`poc_pegou & ~sucesso`)
  está ERRADO: na cadeia de um elo é identicamente falso.
- **`carregar` fecha SUSTENTADO** (0,5 s, knob já existe), não no instante — um
  fecho instantâneo com no_alvo ≈ 57% é uma moeda e o nível viraria passeio sem
  deriva, invalidando o portão da 4b.
- **Bloco preparatório antes da 4b**: hoje 0,00% das transições têm twist ≠ 0 E
  caixa_valida = 1 (por construção). "Segure e ande" com uma fração dos envs de
  manipulação com twist liberado, antes do elo novo.
- **`reorientar` é empurrar APOIADA, não girar em mão** (diagonal 0,283 m > vão das
  palmas; torque de escorregamento ≈ 0,49 N·m). O ator precisa de 3 canais com a
  orientação da caixa (112 → 115) antes da 4a.
- **`std_carregando`** (5º dicionário: pernas do walking + braços do manipulando) —
  `std_walking` com caixa na mão é penhasco de −0,885/s; `std_manipulando` tem 1/7
  da moldagem de marcha.
- **A "correção 3" da Fase 4 deste plano CAI**: ninguém observa `poc_topo` (o
  crítico lê a pose real da mesa); com o `unload` mascarado ao `pegar`, reescrever
  `poc_topo` é desnecessário — e fazê-lo ANTES da máscara zeraria o `unload` na
  cadeia do `carregar` (ordem importa).
- ⚠ Docstring de `g1_training/common/robot.py` diz "±Z local" para a normal do pad;
  o código produz ±Y. O `squeeze` usa ±Y e `.abs()` — correto, mas a mina fica para
  quem mexer no `reorientar`. (Não editamos `g1_training/`.)

## Contingências (não implementar agora; régua definida)

- **`contato_ilegal` × nível 2**: o Risco 3 da §19 está INVERTIDO — o pico é no topo
  0,30 m (20–25% das poses de pega), não a 0,04 m (0–3%). Se a escada travar em
  `nivel_medio ≈ 2` com `Episode_Termination/contato_ilegal` alto, a mitigação é
  prateleira em x = 0,60 (que arrasta `caixa_xy` junto — mudança de geometria, um
  bloco próprio).
- **`base_lin_vel` invisível ao ator**: 50% do sinal de locomoção é função de estado
  que o ator não observa num quadro só (o fabricante TEM o canal no ator; nós
  tiramos e não pusemos histórico). Conserto candidato: `history_length = 5` no
  grupo actor (112 → 560 canais) — QUEBRA warm-start e contrato de deploy. Decisão
  do usuário, para um bloco de fronteira.
- **Persistência do gate do twist**: `poc_estagio_twist`/EMA não vão ao checkpoint
  (o runner do mjlab só salva `common_step_counter`; não editamos o pacote). Com a
  descida + teto de degrau, o gate se recalibra em ~12 iterações após cada resume.
  Declarado.
