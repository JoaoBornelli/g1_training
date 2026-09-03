# `g1_limpo` — como o treinamento está montado

> Leitura minuciosa do código em `/home/joaobornelli/Documents/g1_training/g1_limpo`,
> arquivo por arquivo, função por função, na ordem em que o código realmente executa.
>
> Gerado em 2026-08-31. 18 arquivos `.py`, ~450 KB de fonte.

---

## Sumário

0. [Python orientado a objetos — o mínimo para ler este código](#0)
1. [O mapa dos arquivos: quem importa quem](#1)
2. [Onde o código começa: a cadeia de entrada](#2)
3. [O que `make_env_cfg` monta, na ordem exata](#3)
4. [A cena: robô, caixa, laje, sensores](#4)
5. [A ordem de execução de UM reset](#5)
6. [A ordem de execução de UM passo](#6)
7. [A observação: o que a política vê](#7)
8. [O comando: os 5 elos e as 4 cadeias](#8)
9. [As recompensas: 9 termos próprios + a tabela do fabricante](#9)
10. [As penalidades e as terminações](#10)
11. [O currículo: três relógios independentes](#11)
12. [A ordem do treinamento: as fases F0 → F6](#12)
13. [As limitações declaradas](#13)
14. [As ferramentas: smoke, inspeciona, leitura, paridade](#14)
15. [Divergências entre o código e a especificação do projeto](#15)

---

<a id="0"></a>
## 0. Python orientado a objetos — o mínimo para ler este código

Cinco padrões aparecem em todo o pacote. Sem eles, o resto não se lê.

### 0.1 `class` + `__init__` + `self`

```python
class AlvoCaixaCmd(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._elo = torch.full((n,), PEGAR, ...)
```

- `class X:` declara um **molde**. `X(...)` cria um **objeto** desse molde.
- `__init__` roda **uma vez**, no instante da criação. É onde os buffers nascem.
- `self` é o objeto em si. `self._elo = ...` guarda um valor **dentro** do objeto, e ele
  sobrevive entre chamadas. É isto que dá memória ao termo.
- `_elo` com underscore na frente é convenção: "isto é interno, não mexa de fora".
  O Python não impede — é só sinalização.

### 0.2 Herança: `class Filho(Pai)` e `super()`

```python
class AlturaDeBalanco(feet_swing_height):
    def reset(self, env_ids=None):
        self.peak_heights[env_ids] = 0.0
```

`AlturaDeBalanco` **é** um `feet_swing_height` — herda todo o comportamento dele — e
acrescenta um método `reset` que o pai não tinha. Nada mais muda.

```python
class PosturaPorElo(variable_posture):
    def __call__(self, env, *args, canal_do_elo, ...):
        valor = super().__call__(env, *args, **kwargs)   # <- chama o PAI
        return torch.where(anda, valor, torch.ones_like(valor))
```

`super().__call__(...)` executa a versão do **pai** e devolve o resultado. Aqui o filho
usa o valor do pai e depois o **substitui por 1,0** nos envs que estão num elo de
manipulação. É o idioma dominante do pacote: *reusar o fabricante e alterar um ponto*.

### 0.3 Objeto chamável: `__call__`

```python
class sustentacao:
    def __init__(self, cfg, env): self.t = torch.zeros(env.num_envs)
    def __call__(self, env, ...):  ...      # chamada como se fosse função
    def reset(self, env_ids=None): self.t[env_ids] = 0.0
```

Um objeto com `__call__` pode ser chamado como função: `obj(env, ...)`. É como o mjlab
aceita um **termo de recompensa com estado**. Uma função pura não guarda cronômetro; esta
classe guarda `self.t`. E o `reset` é o que impede o cronômetro de um episódio vazar para o
seguinte.

> ⚠ Regra do mjlab que aparece três vezes no código: `reward_manager.py:174` só chama
> `reset()` em termo de **classe que tenha** um método `reset`. Uma classe sem `reset`
> nunca é zerada. Foi assim que nasceu o bug do `feet_swing_height` (§9.1).

### 0.4 `@dataclass` — configuração declarativa

```python
@dataclass
class Tarefa:
    staged: float = 3.0
    precise_pos: float = 2.0
```

Um `dataclass` é um saco de campos com valores padrão. `Tarefa()` já vem com tudo
preenchido; `Tarefa(staged=5.0)` muda um. Todo o `knobs.py` é isto — nada de lógica, só
números com nome.

`field(default_factory=Cena)` aparece quando o padrão é um objeto mutável: cada instância
precisa do **seu** dicionário, senão todas compartilham o mesmo.

### 0.5 O padrão Cfg → `build()`

```python
@dataclass(kw_only=True)
class AlvoCaixaCmdCfg(CommandTermCfg):
    peito_b: tuple = (0.25, 0.00, 0.15)
    def build(self, env): return AlvoCaixaCmd(self, env)
```

Existem **dois** objetos por termo:

| objeto | papel | quando existe |
|---|---|---|
| `AlvoCaixaCmdCfg` | a receita: só números, serializável | na montagem do cfg |
| `AlvoCaixaCmd` | o termo vivo: tensores, buffers, lógica | quando o env é criado |

O mjlab chama `cfg.build(env)` para transformar um no outro. O comentário no fim de
`comando.py` avisa: o mjlab **não** usa um atributo `class_type` — quem escrever
`class_type = ...` cria um campo morto, o `build` **herdado** roda, e o treino usa o
twist do fabricante sem nenhum erro.

---

<a id="1"></a>
## 1. O mapa dos arquivos: quem importa quem

**Invariante do pacote, declarado no `__init__.py`:** *zero import de código do projeto.*
Nada aqui importa `g1_training`, `g1_poc` ou `g1_multitask`. A única exceção é
`paridade.py`, que é um verificador descartável e não roda em treino.

```
                        train.py / play.py
                                │
                                ▼
                         __init__.py  ──────────────┐  registra a task no gym
                                │                   │  + runner_cls
                                ▼                   ▼
                          env_cfg.py            runner.py
                                │
         ┌──────────┬───────────┼───────────┬──────────┬──────────┐
         ▼          ▼           ▼           ▼          ▼          ▼
      cena.py   comando.py  recompensas  curriculo  eventos   terminacoes
         │          │        observacoes      │      metricas
         │          │                         │
         └──────────┴────────► knobs.py ◄─────┘
                            (todos os números)
```

### Dependências reais (não circulares)

| arquivo | importa do pacote | é importado por |
|---|---|---|
| `knobs.py` | — (folha) | todos |
| `cena.py` | `knobs` | `env_cfg` |
| `curriculo.py` | — (**não** importa `comando`) | `env_cfg`, `eventos`, `comando`, `runner` |
| `comando.py` | `curriculo` (`garante_elo`, `garante_nivel`) | `env_cfg`, `__init__`, `eventos` |
| `eventos.py` | `curriculo` | `env_cfg` |
| `recompensas.py` | `comando` (imports **dentro** das funções) | `env_cfg` |
| `observacoes.py` | `comando` (import dentro da função) | `env_cfg` |
| `terminacoes.py` | — | `env_cfg` |
| `metricas.py` | — | `env_cfg` |
| `env_cfg.py` | todos os acima | `__init__` |
| `runner.py` | `curriculo` (dentro do método `load`) | `__init__` |

**Por que `curriculo.py` não importa `comando.py`:** seria ciclo — `comando.py` importa
`garante_nivel` de lá. A solução é passar os ids dos elos por **parâmetro**, do `env_cfg`,
que é quem importa os dois. E é também por isso que `recompensas.py` e `observacoes.py`
fazem `from g1_limpo.comando import VALIDA` **dentro** da função, e não no topo.

### Os buffers publicados no `env` (a cola entre módulos)

O `env` do mjlab é usado como quadro de avisos. Quem escreve, quem lê:

| atributo | escrito por | lido por |
|---|---|---|
| `env.limpo_nivel` | `curriculo.nivel` | `eventos.*`, `comando`, o desenho |
| `env.limpo_elo` | `curriculo.sorteia_elo` | `eventos.reset_base_por_elo`, `comando`, `curriculo.forma` |
| `env.limpo_forma` | `curriculo.forma` | `curriculo.sorteia_elo`, `runner` |
| `env.limpo_topo` | `eventos.posiciona_cena`, `comando._laje_para` | inspeção |
| `env.limpo_massa` | `eventos.carga_caixa` | `recompensas.unload`, `_forca_ref`, `comando._fecha_elo` |
| `env.limpo_pegou` | `comando._publica_pegou` | `terminacoes.caixa_largada` |
| `env.limpo_ids_palma` | `comando.__init__` | `terminacoes.caixa_largada` |
| `env.limpo_voltas` | `eventos.orientacao_de_nascimento` | o desenho de debug |

---

<a id="2"></a>
## 2. Onde o código começa: a cadeia de entrada

### `train.py` — 15 linhas, e não faz nada

```python
import g1_limpo                              # ← o import REGISTRA a task
from mjlab.scripts.train import main as train_main
if __name__ == "__main__":
    train_main()
```

Rodar:

```bash
python -m g1_limpo.train --env.scene.num-envs 4096
```

> ⚠ `num_envs` **não tem default útil**: o default do mjlab é **1**. Tem de passar no CLI.

### `__init__.py` — o registro

O trabalho todo acontece no *import*. Ele registra **9 tasks**:

| task | env_cfg | para quê |
|---|---|---|
| `Mjlab-G1-Limpo` | `make_env_cfg()` | **o treino** |
| `Mjlab-G1-Limpo-Inspecao-{Andar,Reorientar,Pegar,Carregar,Botar}` | `make_env_cfg(inspecao=True, elo=i)` | ver 1 elo parado |
| `Mjlab-G1-Limpo-Inspecao-Cadeia-{1,2,3}` | `make_env_cfg(inspecao=True, elo=CADEIAS[i][0], cadeia=i, avanca_apos_s=3.0)` | ver a **transição** |

Três decisões declaradas ali:

1. **`experiment_name = "g1_limpo"`** não é cosmético. O `load_run` default é o regex
   `.*`, portanto um `--agent.resume` casaria com a run de **outro pacote**. Já custou uma
   sessão ao repositório.
2. **`runner_cls=RunnerComEstadoDeCurriculo`** também não. Sem ele o estado do currículo
   **não vai para o checkpoint**, e Colab/Kaggle matam sessão no meio de um bloco de 5000
   iterações — a rampa de ~400 iterações seria re-paga a cada reinício.
3. **Uma task por cadeia.** O `run_play` do mjlab carrega o cfg **registrado** e roda o
   próprio laço; ele não expõe gancho para mutar o cfg. Registrar a variante é o único
   caminho suportado.

### `runner.py` — o que vai para o checkpoint

`RunnerComEstadoDeCurriculo(MjlabOnPolicyRunner)` sobrescreve dois métodos:

```python
def save(self, path, infos=None):
    estado = {"forma": {...}, "limpo_nivel": ..., "limpo_elo": ...}
    super().save(path, {**infos, "limpo_curriculo": estado})

def load(self, path, *a, **kw):
    carregado = super().load(path, *a, **kw)
    # restaura env.limpo_forma, env.limpo_nivel, env.limpo_elo
```

| vai | não vai |
|---|---|
| `alvo, dur_loco, dur_manip, razao` (as EMAs) | os buffers do termo de comando |
| `passo_inicial, ultimo_degrau, iters_balanco` | `cadeia`, `passo`, `sustenta`, os σ |
| `abriu`, `sorteio` | — |
| `limpo_nivel`, `limpo_elo` (por env) | — |

Motivo do "não vai": aqueles são estado de **episódio**, não de currículo. Um resume começa
com todos os envs em reset, portanto eles nascem corretos de qualquer forma — e salvá-los
criaria a chance de restaurar um σ de uma pose que não existe mais.

**Tolerância a mudança de nº de envs:** o Kaggle dá 2 GPUs num dia e 1 no outro. O `load`
copia `k = min(buf.numel(), salvo.numel())` e avisa quantos ficaram no default.

---

<a id="3"></a>
## 3. O que `make_env_cfg` monta, na ordem exata

`env_cfg.py::make_env_cfg(k=None, play=False, *, inspecao=False, elo=None, cadeia=None, avanca_apos_s=None)`

Ela parte do **molde do fabricante** e o modifica. Passo 0 é a linha mais importante:

```python
cfg = unitree_g1_flat_env_cfg(play=play)
```

**Por que o molde e não `ManagerBasedRlEnvCfg` do zero:** as três tabelas de σ por junta do
`variable_posture` do G1 existem **apenas dentro** de `unitree_g1_rough_env_cfg`. Não há
constante exportada. Redigitá-las custaria ~40 linhas (14 padrões × 3 regimes) sem nenhum
teste que pegue um dígito trocado — *"um σ de joelho de 0,35 digitado 0,035 achata o passo,
e a run morre 1200 iterações depois num painel."*

A variante `flat` já remove de graça o `terrain_scan`, o `height_scan`, o
`out_of_terrain_bounds` e o currículo `terrain_levels`.

### A sequência completa

| § | o que faz | detalhe que importa |
|---|---|---|
| 1 | `cfg.scene.entities = C.entidades(k)` | robô + caixa + laje |
| 1 | `cfg.scene.sensors += C.sensores()` | **os do fabricante FICAM**; os nossos são adição |
| 1 | física: `njmax=800, nconmax=300, impratio=1.0, cone="pyramidal"` | cicatriz de 15/07: `elliptic` + `impratio=10` divergiu para NaN |
| 2 | `acao.scale = {p: v * escala_acao_mult}` | `G1_ACTION_SCALE` tem **16 padrões regex**, não 29 nomes de junta |
| 2b | acrescenta `terminacao` (`is_terminated`) e `joint_acc` | os dois termos do `g1_multitask`, o módulo que **andou** |
| 2f | **3 terminações de mesa**, uma por grupo de geom | itera `C.MESA_POR_GRUPO` |
| 2f | `caixa_largada` | armada pela 1ª preensão |
| — | `foot_swing_height.func = RC.AlturaDeBalanco` | conserta o `reset` que falta |
| — | `pose.func = RC.PosturaPorElo` | neutro nos elos de manipulação |
| — | `aplica_pesos(cfg, k.recompensa)` | com `assert nome in cfg.rewards` |
| 2c | `cfg.metrics.update(MT.termos(...))` | as métricas saem de dentro da recompensa |
| 2d | reconstrói `cfg.commands["twist"]` como subclasse | campo a campo por `dataclasses.fields` |
| 3 | `cfg.events.pop("base_com")` | corrompe a heap em CPU **e** GPU |
| 3 | `posiciona_cena`, `carga_caixa`, `reset_base_por_elo` | um evento por entidade |
| 3b | `cfg.curriculum["forma"] / ["nivel"] / ["elo"]` | **a ordem é contrato** |
| 3c | `cfg.commands["alvo_caixa"] = CMD.AlvoCaixaCmdCfg(...)` | a única fonte de verdade do alvo |
| 3e | obs `["elo"]` nos **dois** grupos | one-hot de 5 |
| 3f | obs `["caixa"]` nos **dois** grupos | 10 canais, gateados pelo elo **publicado** (spec §6.1) |
| 3g | 7 termos de recompensa de tarefa | todos gateados por `VALIDA` |
| 3h | obs `["elo_interno"]` **só no `critic`** | one-hot do elo interno; ator 114, crítico 131 — o crítico do fabricante já tem 12 canais privilegiados de pé (spec §6.1) |
| 3i | `load`, `largou` | só no `BOTAR` e na espera final; leem o elo interno (spec §6.6.2) |
| 3d | ramo inspeção | `trava_robo`, `terminations = {}` |
| 4 | ramo play | remove `randomize_terrain` e `commands_vel` |

### Os dois `assert` que valem a leitura

```python
def aplica_pesos(cfg, r):
    for nome, peso in dataclasses.asdict(r).items():
        if nome == "altura_de_balanco": continue     # não é peso, é alvo em metros
        assert nome in cfg.rewards, f"termo '{nome}' não existe no molde"
        cfg.rewards[nome].weight = peso
```

Sem o `assert`, um nome que o mjlab renomeie num upgrade passaria como
`cfg.rewards["nome_velho"].weight = x` num dict — criando um termo **órfão sem `func`**, ou
pior, deixando o termo real com o peso do molde. O `assert` transforma isso num erro na
montagem, e não num painel estranho 3000 iterações depois.

### As três constantes de fase (no topo de `env_cfg.py`, não no `knobs.py`)

```python
ELO_DE_TREINO   = CMD.ANDAR                      # o elo da MAIORIA
ELOS_QUE_ANDAM  = (CMD.ANDAR, CMD.CARREGAR)      # governa 3 coisas de uma vez
ELOS_SORTEAVEIS = (CMD.REORIENTAR, CMD.PEGAR)    # o que um RESET pode entregar
```

`ELOS_QUE_ANDAM` governa: (a) a faixa de yaw do reset da base, (b) a neutralidade da
postura, (c) quais elos **não** têm o twist zerado. É de propósito que a lista seja uma só.

`ELOS_SORTEAVEIS` exclui `CARREGAR` e `BOTAR` porque eles **começam com a caixa nas mãos** —
existem só como 2º elo de uma cadeia. Consequência declarada: os **slots 3 e 4 do one-hot
ficam constantes em zero** até a F4, e `rsl_rl/modules/normalization.py:48` faz
`(x − _mean)/(_std + 1e−2)` sem clamp — logo o primeiro `1,0` entra na rede como **100,0**.
A decisão foi aceitar o transiente: pôr um env em `CARREGAR` com a caixa na prateleira
ensinaria à política que o slot 3 significa "caixa na prateleira, ande", e a F4 teria de
**desaprender** isso.

### `elo=None` vs `elo=X` — a distinção que mata a F2 em silêncio

```python
elo_explicito = elo is not None
elo_alvo = ELO_DE_TREINO if elo is None else int(elo)
```

- `elo=X` → **inspetor e play**: o elo é FORÇADO, igual em todos os envs.
- `elo=None` → **treino**: o elo é SORTEADO por env. Forçar o `ELO_DE_TREINO` no comando
  aqui anularia o sorteio inteiro sem uma linha de erro.

É por isso que `elo_forcado=elo_alvo if elo_explicito else None` aparece nos dois
consumidores (currículo e comando).

---

<a id="4"></a>
## 4. A cena: robô, caixa, laje, sensores

`cena.py` — tudo aqui foi **transcrito à mão** de `g1_training/common/box.py`,
`common/robot.py`, `base_env.py` e `g1_poc/cena.py`. O `paridade.py` prova que a
transcrição bate, comparando o `mjModel` **compilado**.

### 4.1 O robô — `robot_cfg()`

```python
CollisionCfg(
    geom_names_expr=(".*_collision", ".*_pad"),
    condim={r"^(left|right)_foot[1-7]_collision$": 3,
            r".*_palm_pad$": 3, r".*_hand_back_pad$": 1, ".*_collision": 1},
    priority={r"^(left|right)_foot[1-7]_collision$": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)})
EntityCfg(init_state=KNEES_BENT_KEYFRAME,
          spec_fn=lambda: add_pads_de_palma(get_spec()), ...)
```

> ⚠ Os pads terminam em `_pad`, portanto **não** casam com `.*_collision`. Sem entrarem
> explicitamente em `geom_names_expr`, eles não participam da colisão e os sensores de
> palma nunca disparam.

### 4.2 `add_pads_de_palma(spec)` — o refinamento das mãos

1. **Apaga** as cápsulas `left_hand_collision` e `right_hand_collision`. Elas são
   radialmente simétricas: não distinguem palma de verso, e deixariam o **dorso** tocar a
   caixa e contar como pega.
2. Acrescenta 4 geoms box em cada mão:

| pad | `condim` | offset | significado |
|---|---|---|---|
| `left_palm_pad` | 3 | `y = −0,015` | palma, com atrito |
| `right_palm_pad` | 3 | `y = +0,015` | palma, com atrito |
| `left_hand_back_pad` | 1 | `y = +0,015` | dorso, escorregadio |
| `right_hand_back_pad` | 1 | `y = −0,015` | dorso, escorregadio |

Meia-aresta do pad: `(0.035, 0.008, 0.045)` — laje fina. Atrito `(1.0, 0.02, 0.001)`.

> ⚠ **Discrepância na referência, e o código está certo.** `common/robot.py` comenta
> "palma: −Z local" para as **duas** mãos, mas o código usa **sinais opostos**. Os frames
> dos dois punhos são simétricos em qpos0, portanto o sinal oposto é o que produz palmas
> **voltadas uma para a outra**. O `paridade.py` compara `geom_pos` e travaria se o
> comentário tivesse sido copiado.

> ⚠ Nota de API (mujoco 3.10 / mjlab 1.5): remover um nó é `spec.delete(geom)`, e **nunca**
> `geom.delete()`.

### 4.3 A caixa — `spec_caixa(c)`

| item | valor | knob |
|---|---|---|
| forma | cubo, meia-aresta 0,10 m | `caixa_meia_aresta` |
| massa | 1,0 kg | `caixa_massa` |
| junta | **free joint** (6 DoF), `nq = 7` | — |
| `condim` / atrito | 3 / `(1.0, 0.02, 0.001)` | `caixa_condim`, `caixa_atrito` |
| grupo de geom | **2** | `grupo_mobilia` |
| nasce em | `x = 0,32`, `y = 0,00` | `caixa_xy` |

Mais uma **placa visual** na face alvo (`face_alvo` geom): `contype = 0`,
`conaffinity = 0`, `density = 0.0`. Massa e inércia ficam **bit-idênticas** à caixa sem
ela (o `paridade.py` afirma isso). Sem a placa, a inspeção do `reorientar` seria cega — um
cubo uniforme girado 90° é visualmente idêntico ao original.

### 4.4 A laje — `spec_prateleira(c)`

**Sem free joint** → o mjlab a auto-envolve em **MOCAP**. Três consequências, e as três são
o motivo:

1. corpo **cinemático**, posicionável por env em runtime (`write_mocap_pose_to_sim`);
2. flutua em qualquer z **sem** tocar o chão;
3. não é movida por contato, portanto dispensa massa.

E ela é **fina em z** (`prateleira_meia_z = 0,02` → 4 cm de espessura): é uma prateleira,
não um paredão — o que mata o atalho de escorar o robô nela.

> ⚠ O nome do geom é `table_geom` e o do body é `table`, apesar de ser uma laje. Mantido de
> propósito: é o nome que os padrões de sensor casam.

> ⚠ **O grupo 2 é obrigatório.** `regroup(spec, 2)` põe todos os geoms da mobília fora do
> grupo 0, porque o `foot_height_scan` do fabricante usa `include_geom_groups=(0,)` — ele
> leria a prateleira **como chão**, e o robô "veria" um degrau que na verdade é a mesa dele.

### 4.5 `geometria_de_repouso(c)` — as alturas derivadas num lugar só

```python
topo               = prateleira_topo_teto            # 0,55
centro_prateleira  = topo − prateleira_meia_z        # 0,53
fundo_prateleira   = topo − 2·prateleira_meia_z      # 0,51
caixa_z            = topo + caixa_meia_aresta[2]     # 0,65
```

A pose de um corpo mocap é o **centro**, não o topo. Esta é a conta que o
`inspeciona.py` confere.

### 4.6 Os 10 sensores de contato — `sensores()`

| nome | primário | secundário | campos | `reduce` |
|---|---|---|---|---|
| `palma_E`, `palma_D` | `*_palm_pad` | `box_geom` | `found`, **`force`** | netforce |
| `dorso_E`, `dorso_D` | `*_hand_back_pad` | `box_geom` | `found` | none |
| `corpo_prateleira` | pelve, tronco, quadril, coxa | `table_geom` | `found`, `force` | netforce |
| `palma_prateleira` | `.*_palm_pad` | `table_geom` | `found`, `force` | netforce |
| `dorso_prateleira` | `.*_hand_back_pad` | `table_geom` | `found`, `force` | netforce |
| `apoio_caixa` | `box_geom` | `table_geom` | `found`, **`force`** | netforce |
| `auto_colisao` | subtree `pelvis` | subtree `pelvis` | `found`, `force` | none, hist 4 |
| `pes_chao` | tornozelos | **`None`** = qualquer | `found`, `force`, `track_air_time` | netforce |

Quatro decisões dentro dessa tabela:

- **`force` onde a magnitude importa.** É o ponto que `g1_training/base_env.py` **não**
  resolve: o `_pad_contact_sensor` de lá pede só `fields=("found",)`, e com isso o `squeeze`
  e o `unload` são **impossíveis** de escrever.
- **O campo `normal` NÃO é pedido.** Com `reduce="netforce"` todos os contatos somam num
  wrench só, e "a normal do contato" perde significado. Quem precisa da normal da palma a
  calcula da **orientação do sítio**, que é exata. E a força sai no frame **global**, porque
  netforce implica global.
- **`secondary=None` no `pes_chao`** → qualquer contato conta como chão. Pisar na
  prateleira conta, e é o que queremos: sem isso o slip do pé fica cego justamente no nível
  em que a laje é um degrau de 4 cm.
- **Os sensores do fabricante ficam todos.** Tentar substituir `feet_ground_contact` e
  `self_collision` pelos nossos **explode na montagem**: o termo de obs `foot_air_time`
  referencia `feet_ground_contact` **por nome**, e o reward `self_collisions` referencia
  `self_collision`. Preço de manter os dois: um sensor de contato de pé duplicado. É barato,
  e mantém intacto o contrato que fez o robô andar.

### 4.7 Três sensores de mesa em vez de um

```python
GRUPO_TRONCO = (r"pelvis_collision", r"torso_collision",
                r".*_hip_collision", r".*_thigh_collision")
GRUPO_PALMA  = (r".*_palm_pad",)
GRUPO_DORSO  = (r".*_hand_back_pad",)
CORPOS_QUE_NAO_ESCORAM = GRUPO_TRONCO + GRUPO_PALMA + GRUPO_DORSO
MESA_POR_GRUPO = ((SENSOR_CORPO_PRATELEIRA, "contato_tronco"),
                  (SENSOR_PALMA_PRATELEIRA, "contato_palma"),
                  (SENSOR_DORSO_PRATELEIRA, "contato_dorso"))
```

**A união dos três é idêntica ao termo único de 28/08.** A partição é puramente de
**medição**: com `reduce="netforce"` um sensor entrega **um** número, portanto uma lista só
diz "encostou" e não diz **com o quê**. No bloco 4 isso era ~46% dos episódios de
manipulação e o log não distinguia tronco de coxa de pad da palma — qualquer conserto
seguinte seria chute. O robô não vê diferença nenhuma: mesmos geoms, mesmo limiar de 50 N,
mesmo `terminacao = −200`.

**A história da lista** (vale ler, é o padrão de erro do projeto):

> A lista de **corpo inteiro** (`.*_collision`, 33 geoms) **rodou e falhou**. Medido no
> bloco 3, it 4251: `contato_ilegal` fez 18,5% das terminações contra uma fatia de
> manipulação de 24,7% — ~75% dos episódios de manipulação morriam na mesa, com `squeeze`
> em 0,0002 depois de 3200 iterações. O sinal estava **invertido**, por duas razões que se
> somavam:
> 1. `add_pads_de_palma` apaga `*_hand_collision`. Nenhum `_pad` casa com `.*_collision`,
>    portanto a **mão não tinha geom nenhum** neste sensor — era a única superfície do
>    corpo com custo **zero** para escorar, exatamente a que o dono viu escorando no `play`.
> 2. `.*_collision` cobre **punho, cotovelo e ombro** — as peças que **têm** de chegar perto
>    do tampo para pegar. Aproximar terminava; escorar era grátis.
>
> A lista de hoje inverte os dois: **os pads entram, o punho e o cotovelo saem.** O pé fica
> fora. E o que torna isso seguro é o **limiar de força**, não a lista.

> ⚠ **Acoplamento latente, declarado:** com `topo_min` chegando a 0,04 m nos níveis 4 a 6, a
> laje é um degrau de 4 cm na frente dos pés, e **pisar nela passa dos 50 N**. Nesses níveis
> a mesa precisaria deixar de existir, e **isso não está implementado**. Os blocos 1 a 3
> nunca saíram do nível 0, onde o topo é ≥ 0,55 m — portanto é risco **adiado**, não ativo.

---

<a id="5"></a>
## 5. A ordem de execução de UM reset

Esta é a seção mais importante do documento. **A ordem é contrato, e ela tem teste.**

```
   env._reset_idx(env_ids)
        │
        │  (mjlab: manager_based_rl_env.py)
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 1. curriculum_manager.compute()          (linha :554)         │
  │                                                               │
  │    ordem do dict = ordem de INSERÇÃO em env_cfg.py:           │
  │                                                               │
  │    a) command_vel   (do fabricante) alarga a faixa de twist   │
  │    b) forma         mede o episódio que ACABOU → sorteio      │
  │    c) nivel         mede o episódio que ACABOU → env.limpo_nivel│
  │    d) elo           escreve o episódio que COMEÇA → env.limpo_elo│
  └───────────────────────────────────────────────────────────────┘
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 2. event_manager.apply(mode="reset")     (linha :560)         │
  │                                                               │
  │    a) reset_base_por_elo    LÊ limpo_elo → faixa de yaw       │
  │    b) posiciona_cena        LÊ limpo_nivel → topo + caixa     │
  │    c) carga_caixa           LÊ limpo_nivel → limpo_massa      │
  │    d) [segura_caixa]        só inspeção de CARREGAR/BOTAR     │
  │    e) push_robot etc.       do fabricante                     │
  └───────────────────────────────────────────────────────────────┘
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 3. command_manager.reset()               (linha :581)         │
  │                                                               │
  │    twist._resample()             sorteia vx, vy, wz           │
  │    alvo_caixa._resample_command()                             │
  │       · zera _pegou                                           │
  │       · LÊ limpo_elo → _elo                                   │
  │       · sorteia a CADEIA compatível com esse elo              │
  │       · zera _passo, _sust, avancou, fechou                   │
  │       · _aplica_elo(env_ids)   ← escreve o alvo               │
  │       · _pendente = True       ← a parte de POSE fica p/ depois│
  └───────────────────────────────────────────────────────────────┘
```

### 5.1 Por que `forma → nivel → elo`, e não o contrário

Os dois primeiros medem o episódio que **acabou**; o `elo` escreve o do episódio que
**começa**. Os dois primeiros precisam ler `env.limpo_elo` **antes** de o terceiro
sobrescrevê-lo:

- o `forma` atribui as durações medidas ao lado certo (locomoção ou manipulação);
- o `nivel` precisa saber se o episódio era de locomoção para **não** mover o nível por
  causa dele.

**Bug medido em 20/08 com a ordem invertida:** a probabilidade de subir de nível caía de
`p` para `0,7·p`, o ponto fixo saía de 0,5 para **0,714**, e um episódio de **locomoção**
rebaixava o nível em 70% das vezes.

> ⚠ O docstring da função `curriculo.forma` ainda diz `command_vel → elo → nivel → forma`.
> **Está desatualizado.** A ordem real, a que o `env_cfg` insere e o `smoke` afirma
> (seção 20), é `command_vel → forma → nivel → elo`.

### 5.2 Por que o `elo` é currículo e não evento

Porque o **reset de pose da base** depende dele (`reset_base_por_elo` decide a faixa de yaw)
e o **alvo** depende dele. Todo termo de currículo roda antes de todo evento. Num evento, o
elo chegaria **depois** de o reset de pose já ter acontecido.

### 5.3 A armadilha do `_pendente` — pose obsoleta no reset

**Medida em 25/08, e custa uma sessão a quem não souber.**

No reset o command manager roda **depois** dos eventos que reposicionam a caixa e a laje,
mas os buffers de `data` das entidades **ainda não foram recomputados**. Portanto tudo que
depende de pose é lixo ali.

O que isso quebrou: o teto do `BOTAR` é `min(fundo_da_caixa − folga, teto_do_knob)`, e com a
pose obsoleta o fundo deu **negativo** — o teto colapsou no piso e o topo saiu 0,300 nos
oito envs. Só o `maximum(teto, piso)` impediu uma laje enterrada. O alvo do `ANDAR` saiu
0,000 pelo mesmo motivo.

**A solução:** marcar o env como `_pendente = True` no resample, e concluir a parte
dependente de pose no **primeiro `_update_command`**, quando a pose está fresca:

```python
def _update_command(self):
    pend = todos[self._pendente]
    if len(pend):
        self._aplica_elo(pend, so_pose=True)      # re-sorteia topo e alvo do BOTAR
        self._recalcula_sigmas(pend)              # o σ SÓ é calculado aqui
        self._pos_no_elo[pend] = self.robot.data.root_link_pos_w[pend]
        self._pendente[pend] = False
```

O mesmo defeito aparece no `eventos.segura_caixa`, que por isso **não lê pose nenhuma** —
usa a constante `POSE_TRAVADA = (0.0, 0.0, 0.80)` e soma o `peito_b`.

### 5.4 `eventos.reset_base_por_elo` — um despachante, não uma reimplementação

```python
anda = torch.isin(elo, tensor(elos_que_andam))
for mascara, faixa in ((anda, faixa_loco), (~anda, faixa_manipula)):
    reset_root_state_uniform(env, env_ids[mascara], pose_range=faixa, ...)
```

| elo | faixa de yaw | x, y | motivo |
|---|---|---|---|
| ANDAR, CARREGAR | **±3,14** | ±0,50 | a mobília está a +5 m, não há com que alinhar o rumo |
| REORIENTAR, PEGAR, BOTAR | **±0,2** | −0,10..0, ±0,10 | mobília de pose absoluta, à frente do robô |

Os dois defeitos espelhados que essa divisão conserta:

- **±0,2 global** foi o defeito central de um bloco medido: o erro de rumo era sempre
  minúsculo, o `track_angular_velocity` era satisfeito sem o robô fazer nada, e o canal de
  guinada nunca foi exercitado. Quando a política derivou para o giro, ela **não tinha
  autoridade** para sair.
- **±3,14 global**: um env de manipulação nasceria de costas para a prateleira, e a tarefa
  viraria sorte de sorteio.

> ⚠ Ele **não reimplementa amostragem**. Reescrever à mão perderia o `default_root_state`, o
> `env_origins` e o `quat_mul` com a orientação default — três coisas que o fabricante faz e
> que um transcritor esquece.

> ⚠ **A base reseta EM REPOUSO, nos dois modos.** O knob `reset_base_vel_manipula` existia
> com o comentário "o robô chega andando, não parado" e **nenhum consumidor** — o `env_cfg`
> passa `velocidade: {}`. Foi **removido** em 26/08. Um knob morto num arquivo cuja premissa
> é reprodutibilidade por `git diff` é pior que ausente.

### 5.5 `eventos.posiciona_cena` — um evento por entidade

> ⚠ **Dois eventos que escrevem a pose da MESMA entidade no MESMO reset não se somam: o
> segundo APAGA o primeiro, sem erro e sem log.** Por isso `posiciona_cena` faz a
> prateleira **e** a caixa, e não existe um segundo evento tocando nenhuma das duas.

```python
topo = _topo_por_nivel(...)      # publica env.limpo_topo
mesa.write_mocap_pose_to_sim(pose com z = topo − meia_z)
caixa.write_root_link_pose_to_sim(pose com z = topo + caixa_meia_z,
                                  quat = orientacao_de_nascimento(...))
caixa.write_root_link_velocity_to_sim(zeros)
```

`_topo_por_nivel`:

```python
piso = topo_min[nivel]                                    # só o PISO desce
topo = piso + (topo_teto − piso) · rand()                 # teto SEMPRE 0,55
topo = topo + (2·rand() − 1) · jitter_z                   # ±0,02
topo = maximum(topo, piso)                                # o jitter nunca desce do piso
```

Consequência deliberada: **cada nível CONTÉM o anterior**, e a altura fácil nunca desaparece
do treino no instante da promoção. É o defeito da tabela discreta do `g1_multitask`.

Jitter da caixa:

```python
dx = jitter_x_max[nivel] · rand()          # de UM LADO só: afasta, nunca aproxima
dy = uniform(−0.18, +0.18)
```

O `dx` **aperta** com o nível (`0,20 → 0,15 → 0,08`): com o topo a 0,04 m as poses de pega
só existem até x relativo ~0,45 m. Sem apertar, os níveis altos sorteariam poses fora de
alcance e a competência viraria sorte.

### 5.6 `eventos.orientacao_de_nascimento` — a dificuldade do `reorientar`

O eixo é em **quartos de volta**, e não em graus (decidido 26/08):

```python
voltas = floor(rand() · (voltas_max[nivel] + 1)).clamp(max=1)   # 0 ou 1
usa_y  = eixo_vertical[nivel] & (rand() < 0.5)
sinal  = ±1
ang    = voltas · (π/2) · sinal                                 # no máximo ±90°
pitch  = ang se usa_y senão 0
yaw    = ang se não usa_y senão 0
yaw   += (2·rand() − 1) · desalinho_max_deg[nivel]              # o desalinho SEMPRE entra
```

Duas decisões:

- **Teto de UMA volta.** A face marcada nunca nasce do lado oposto: o robô só precisa
  aprender a girar no máximo 90°. A primitiva atômica **é** o quarto de volta; compor voltas
  sai de graça aplicando a primitiva outra vez.
- **O eixo vertical (Y) entra depois do horizontal (Z)**, e a razão é física: girar em Z é
  **pivotar** sobre a laje, e dá para empurrar com uma mão; girar em Y é **tombar**, e exige
  erguer uma aresta de um cubo de 20 cm.

O desalinho residual entra em todos os níveis (`15° → 20°`). Antes de 26/08 o eixo era
`ang_max_deg = (0, 0, 0, 45, 90, 180, 180)`, e com **zero** nos três primeiros níveis o
`reorientar` ficava **satisfeito em t = 0** — não fazia nada em 3 dos 7 níveis.

### 5.7 `eventos.carga_caixa` — a carga como força externa

```python
teto = carga_max[nivel]
kg = massa_base + (teto − massa_base).clamp(min=0) · rand()
env.limpo_massa[env_ids] = kg                     # em KG, não em newtons
forcas[:, 0, 2] = −(kg − massa_base) · 9.81
caixa.write_external_wrench_to_sim(forces=forcas, torques=zeros)
```

> ⚠ **Nunca `dr.body_mass` nem `dr.pseudo_inertia`:** os dois corrompem a heap (CUDA illegal
> memory access). Está medido no repositório, e é o mesmo tipo de defeito do `base_com`.

> ⚠ **Limitação declarada:** a caixa de 5 kg fica com a **inércia de 1 kg**. A randomização
> endurece a **estática**, não a dinâmica.

Publica **kg** e não newtons de propósito: o `unload` deriva `m·g` e o `squeeze` deriva
`F_ref = m·g/(2µ)`. Publicar newtons obrigaria um dos dois a desfazer a conta, *"e é assim
que se erra um fator 9,81 em silêncio"*.

---

<a id="6"></a>
## 6. A ordem de execução de UM passo

```
  env.step(action)
     │
     ├─ action_manager.process(action)      escala + offset da pose default
     │
     ├─ decimation × 4:  sim.step()         timestep 0,005 s → dt de controle 0,02 s (50 Hz)
     │
     ├─ command_manager.compute()
     │     ├─ twist:       _update_metrics()  ← razao_marcha + eficiencia por segmento
     │     │               _resample() se time_left <= 0
     │     └─ alvo_caixa:  _update_command()
     │            1. os _pendente  → _aplica_elo(so_pose=True) + _recalcula_sigmas
     │            2. _atualiza_face(todos)             normal + erro angular
     │            3. _alvo_ancorado_na_base(CARREGAR)  alvo anda com o robô
     │            4. alvo do REORIENTAR/ANDAR = a própria caixa
     │            5. _alvo_ancorado_na_base(PEGAR)
     │            6. _zera_twist_nos_parados()   ← SOBRESCREVE tw.vel_command_b
     │            7. _publica_pegou()            ← arma o caixa_largada
     │            8. _avanca_elo()               ← a máquina de estados
     │
     ├─ observation_manager.compute()      actor (com ruído) e critic (sem)
     │
     ├─ reward_manager.compute()           soma ponderada × dt
     │
     ├─ termination_manager.compute()      time_out, fell_over, 3× mesa, caixa_largada
     │
     ├─ metrics_manager.compute()
     │
     └─ se algum terminou → _reset_idx(ids)   → volta para a §5
```

### 6.1 `_zera_twist_nos_parados` — a decisão central do desenho

```python
parados = torch.isin(self._elo, tensor(self.cfg.elos_parados))   # (1, 2, 4)
tw = self._env.command_manager.get_term("twist")
tw.vel_command_b[parados] = 0.0
```

**É isto que impede o robô de andar com a caixa no `pegar`, no `reorientar` e no `botar` —
e não a forma do alvo.** O `pegar` e o `carregar` pedem **exatamente o mesmo ponto**; a
diferença entre os dois elos é só o comando de velocidade.

Duas propriedades:

- **A ordem funciona porque o dict é ordenado por inserção.** O `twist` vem do molde do
  fabricante, portanto está inserido **antes** de `alvo_caixa`. O `compute` dele já rodou
  quando este roda, e a escrita não é sobrescrita no mesmo passo.
- **A escrita é destrutiva de propósito.** Qualquer métrica que gateie por "comando ativo"
  passa a **não** contar estes passos, que é o correto — eles são passos de comando zero de
  verdade, e não passos mascarados na leitura.

### 6.2 `scale_rewards_by_dt = True` — o peso É o valor por segundo

`dt = 0,005 × 4 = 0,02 s`. O mjlab multiplica cada termo por `dt` e o `Episode_Reward` é a
soma. Duas consequências que já custaram sessões:

- **`terminacao = −200` NÃO custa 200.** O passo que termina paga `−200 × 0,02 = −4,0`.
  Contra os ~5,0/s do teto positivo, cair custa ~0,8 s de tudo.
- **`joint_acc = −2,5e−7` é desprezível de propósito** — medido em 0,0006/s. Ele entra por
  paridade com o módulo que andou, não por efeito.

E o `Episode_Reward` do painel tem uma **diluição** que precisa ser desfeita na leitura —
ver §14.3.

---

<a id="7"></a>
## 7. A observação: o que a política vê

`observacoes.py` — 2 termos, ~90 linhas.

**Contrato do layout, declarado no topo do arquivo:** *canal novo entra sempre POR ÚLTIMO,
e nos DOIS grupos, na MESMA ordem.* Assim migrar um checkpoint é um **append de colunas**, e
nunca uma inserção no meio — uma inserção no meio desloca todo peso da primeira camada em
silêncio, e a política sai andando de lado sem uma linha de erro.

```
  grupo "actor"  (com enable_corruption)      grupo "critic" (sem ruído)
  ┌────────────────────────────────┐          ┌────────────────────────────────┐
  │ ... termos do FABRICANTE ...   │          │ ... termos do FABRICANTE ...   │
  │  base_ang_vel, projected_grav, │          │  (idem, mais os privilegiados) │
  │  joint_pos, joint_vel,         │          │                                │
  │  last_action, twist_cmd,       │          │                                │
  │  foot_air_time, ...            │          │                                │
  ├────────────────────────────────┤          ├────────────────────────────────┤
  │ "elo"    → um_de_cinco     5   │  ← 3e    │ "elo"    → um_de_cinco     5   │
  ├────────────────────────────────┤          ├────────────────────────────────┤
  │ "caixa"  → caixa_no_frame  8   │  ← 3f    │ "caixa"  → caixa_no_frame  8   │
  └────────────────────────────────┘          └────────────────────────────────┘
```

### 7.1 `um_de_cinco(env, command_name, canal_do_elo)` — 5 canais

```python
elo = comando[:, canal_do_elo].long().clamp(0, 4)
return F.one_hot(elo, num_classes=5).float()
```

| slot | elo |
|---|---|
| 0 | ANDAR |
| 1 | REORIENTAR |
| 2 | PEGAR |
| 3 | CARREGAR |
| 4 | BOTAR |

Três decisões:

- **Ele é lido do COMANDO, por passo**, e não de um buffer de reset. É isto que permite o
  elo **trocar dentro do episódio** na F4, sem reset e sem resample. Era a única
  incompatibilidade real entre a máquina de elo do `g1_poc` e o one-hot do `g1_multitask`,
  e ela é de uma linha.
- **Sem `noise` e sem `scale`.** Ruído num one-hot produziria frações entre slots, isto é,
  estados que **não existem**. E `scale` num canal já em [0,1] só desalinharia a escala
  contra o normalizador.
- **O one-hot não leva o crédito do andar.** O `g1_poc` já tinha o equivalente funcional —
  o bit `caixa_valida` + o twist forçado a zero — e não andou. A razão de engenharia dele é
  outra: ele diz **qual objetivo está ativo**, e gateia os sete termos de tarefa, que sem
  gate pagariam o **máximo** com os canais de caixa zerados, porque `exp(0) = 1`.

### 7.2 `caixa_no_frame_da_base(env, command_name)` — 8 canais

```python
p, q = robo.data.root_link_pos_w, robo.data.root_link_quat_w
caixa_b = quat_apply_inverse(q, caixa_pos_w − p)
alvo_b  = quat_apply_inverse(q, cmd[:, ALVO] − p)
giro_b  = quat_apply_inverse(q, cmd[:, GIRO])          # eixo × ângulo do giro pedido
return cat([caixa_b, alvo_b, giro_b, limpo_meia_aresta[:, :1]]) * (cmd[:, ELO] != ANDAR)
```

| fatia | conteúdo |
|---|---|
| `[0:3]` | caixa − base, **no frame da base** |
| `[3:6]` | alvo − base, **no frame da base** |
| `[6]` | erro angular da face pedida, em **radianos** |
| `[7]` | `valida`: 1 nos elos com caixa, 0 no `ANDAR` |

Três decisões:

- **Tudo no frame da base, e não em mundo.** Coordenada de mundo carrega a **origem do
  env**, que é diferente em cada um dos 4096 — a política teria de aprender 4096
  deslocamentos. E carrega o rumo: o mesmo problema geométrico visto de dois yaws daria dois
  vetores diferentes.
- **O σ não entra aqui.** Ele diz "este env é fácil ou difícil", e a política condicionaria
  a ação à **forma da recompensa** em vez de à tarefa. *σ é moldagem; a observação é estado
  do mundo.*
- **O erro angular entra em radianos e sem normalizar.** Ele já vive em [0, π].

> ⚠ O `smoke.py` **não** afirma o total de canais do ator. Ele afirma a **ordem** e os
> **nomes** dos termos, e as constantes `N_SLOTS = 5` / `N_CAIXA = 10`. O fatiamento nos
> testes é **calculado** (`fatia_do_elo(114) == slice(99, 104)`), não digitado. O `VALIDA`
> **não** está na observação desde a v2 (spec §6.2); o `ANG` virou o vetor `giro_b`
> (spec §8.3); o `meia_aresta` é o último canal (spec §6.7).

---

<a id="8"></a>
## 8. O comando: os 5 elos e as 4 cadeias

`comando.py`, 1335 linhas — o arquivo mais denso do pacote. Ele é a **única fonte de verdade
do alvo**, e o desenho de debug mora **dentro** dele: um visualizador que reimplementasse o
sorteio seria uma segunda fonte, e mentiria no dia em que as duas divergissem.

### 8.1 O layout do comando — 9 canais

```python
ALVO   = slice(0, 3)    # posição alvo da caixa, em MUNDO
FACE   = slice(3, 6)    # normal DESEJADA da face marcada, em MUNDO (unitária)
ANG    = 6              # erro angular ATUAL, em radianos (= |GIRO|)
VALIDA = 7              # 1,0 se o objetivo de caixa está ativo; 0,0 no ANDAR e na espera inicial
ELO    = 8              # o elo PUBLICADO, como float (ANDAR nas duas esperas; spec §6.0)
GIRO   = slice(9, 12)   # eixo × ângulo do giro pedido, em MUNDO (spec §8.3) — append da v2
DIM    = 12
```

> ⚠ **Dois elos desde a v2 (spec §6.0).** `_command[:, ELO]` é o **publicado**: o que a rede
> vê. `AlvoCaixaCmd._elo` (publicado em `env.limpo_elo_interno`) é o **interno**: a mecânica
> do episódio e o que paga. Eles diferem nas duas esperas: a inicial (`aguardando`, publicado
> `ANDAR`, `VALIDA = 0`) e a final depois do fecho do `BOTAR` (`soltou`, publicado `ANDAR`,
> `VALIDA = 1` — os incentivos do estado apoiado continuam pagando).

### 8.2 Os 5 elos, e o alvo de cada um é uma coisa diferente

```python
ANDAR, REORIENTAR, PEGAR, CARREGAR, BOTAR = 0, 1, 2, 3, 4
```

| elo | `VALIDA` | twist | alvo publicado | laje |
|---|---|---|---|---|
| **ANDAR** | 0 | sorteado | a própria caixa (inerte) | vai a **+5 m**, com a caixa |
| **REORIENTAR** | 1 | **zero** | a própria caixa (pede-se **atitude**) | fica |
| **PEGAR** | 1 | **zero** | `base + peito_b` em x,y · **z = 0,95 absoluto** | fica |
| **CARREGAR** | 1 | sorteado | **idêntico ao PEGAR** | vai a **+5 m** |
| **BOTAR** | 1 | **zero** | lateral, em cima de um **topo novo** | topo novo sorteado |

```python
elos_parados = (1, 2, 4)   # REORIENTAR, PEGAR, BOTAR → twist forçado a zero
```

### 8.3 O alvo do `PEGAR`/`CARREGAR` — referencial dividido por eixo

```python
def _alvo_ancorado_na_base(self, ids):
    p = tensor(self.cfg.peito_b).expand(len(ids), 3)     # (0.25, 0.00, 0.15)
    a = base_p + quat_apply(base_q, p)
    a[:, 2] = self.cfg.altura_carregar                    # 0.95 — ABSOLUTO
    self._command[ids, ALVO] = a
```

| eixo | referencial | por quê |
|---|---|---|
| x, y | **relativos ao robô**, reescritos a cada passo | a caixa está nas mãos e tem de acompanhá-lo |
| z | **absoluto**, `0,95 m` | **agachar não pode baixar o alvo** |

Se o z fosse relativo, o robô satisfaria o alvo **andando agachado**: o alvo desceria junto
com a pelve e a caixa nunca precisaria subir.

**Derivação do 0,95:** a pelve do keyframe joelhos-flexionados fica em `z = 0,798` (medido),
e `peito_b.z = 0,15`. Logo `0,798 + 0,15 = 0,948`. O `smoke` confere esta soma contra a pose
default do robô, para o número não derivar em silêncio.

**Histórico** (é o padrão de erro mais instrutivo do arquivo): o alvo do `pegar` era
absoluto e **fixo** em `z = (0,78; 0,85)`, transcrito da skill Lift, cujo `shelf_top = 0,55`
era **fixo**. Ali significava "erguer 13 a 20 cm da mesa", e estava certo. Aqui a laje varia
de 0,55 a 0,04 por nível, e o alvo continuou absoluto:

```
  nível 0   laje 0,55   caixa 0,65   →  erguer 0,13–0,20 m
  nível 6   laje 0,04   caixa 0,14   →  erguer 0,64–0,71 m
```

O mesmo termo passou a significar 5× trabalhos diferentes, e o eixo `topo_min` graduava
**duas** coisas: de onde pegar **e** quanto erguer. *"Isso vazou, não foi decidido."*

**Não existe jitter no alvo** (decisão do dono, 25/08): um jitter em y de ±0,05 sobre
x = 0,25 deslocava o alvo até 11° fora do eixo do robô, e ele aparecia "de lado" no viewer.
A variedade do episódio vem da **caixa** (jitter em x, y, orientação) e do **nível**.

### 8.4 O alvo do `BOTAR` — o limite físico vence o knob

```python
fundo = caixa_z − caixa_meia_z
teto  = clamp(fundo − botar_folga_laje, max=botar_topo_teto)      # 0,05 / 0,80
piso  = clamp(full(botar_topo_piso), max=teto)                    # 0,30 CEDE
piso  = maximum(piso, prateleira_topo_piso)                       # nunca enterrada (0,04)
teto  = maximum(teto, piso)
topo  = piso + (teto − piso) · rand()
```

Depois: `_laje_para(m, topo)` e o alvo é **lateral**, em cima do topo novo:

```python
a[:, 0] = org.x + uniform(0.30, 0.40)      # botar_x
a[:, 1] = org.y + uniform(−0.12, 0.12)     # botar_y
a[:, 2] = topo + caixa_meia_z
```

> ⚠ **Defeito medido em 26/08** ao estender o inspetor para os 7 níveis: a versão anterior
> fazia `piso = botar_topo_piso` (0,30) e depois `teto = maximum(teto, piso)`. Com a caixa
> segurada **baixa** — o que acontece nos níveis altos — o `fundo − folga` cai abaixo de
> 0,30, e aquele `maximum` **sobrepunha o limite físico com o knob**: a laje nascia em 0,300
> contra um `fundo − folga` de 0,067. **Dentro da caixa.** O check da F4 não pegou porque
> rodava um nível só.

O alvo é lateral e não frontal porque o frontal exigiria alcançar **por cima de 20 cm de
tampo** — defeito medido em 16/07.

**Caso declarado:** se a caixa está segurada mais baixa que a laje mais fina possível,
nenhum topo satisfaz as duas coisas. Aí a laje vai ao chão e o alvo fica acima do fundo da
caixa — geometricamente impossível de satisfazer, *"e é melhor declarar que violar em
silêncio"*.

### 8.5 A face e o erro angular — dois regimes

```python
self._face_viva[ids] = (self._elo[ids] == REORIENTAR)
self._congela_face(ids[self._elo[ids] != REORIENTAR])
```

| elo | direção pedida | o termo pergunta |
|---|---|---|
| **REORIENTAR** | **VIVA** — da caixa para o robô, na horizontal, recalculada todo passo | "vire a face para mim" |
| todos os outros | **CONGELADA** na normal do instante em que o elo abriu | "a caixa girou desde então?" |

```python
def _atualiza_face(self, ids):
    normal_w   = quat_apply(caixa_quat, face_alvo_b)          # o que ela É
    para_robo  = (robo_pos − caixa_pos); para_robo.z = 0      # HORIZONTAL
    viva       = normalize(para_robo)
    desejada   = where(self._face_viva, viva, self._face_alvo_w)
    self._command[ids, FACE] = desejada
    self._command[ids, ANG]  = acos(clamp(normal_w · desejada, −1, 1))
```

> ⚠ Até 28/08 a direção era **viva em todo elo**, e o `precise_ori` (peso 1,0) ficava
> **inerte**: no nível 0 a caixa nasce alinhada (`voltas_max = 0`, desalinho ≤ 15°) e
> `sigma_ori` tem piso de 0,20 rad, portanto o termo nascia satisfeito com derivada ~zero.
> Pior: o alvo se movia com o **robô** — andar em volta da caixa mudava o termo sem tocar
> nela. O `precise_ori` congelado é o que substitui o `box_shake = −0,15` do `g1_poc`.

O erro é o ângulo entre dois vetores **3D**, e não uma rotação em torno de Z: se a caixa
estiver **tombada**, a normal aponta para cima e o erro tem de acusar 90°, não 0.

**A face pedida é CONSTANTE** — `face_alvo_b = (−1, 0, 0)`, a face `−X` da caixa. Com a
caixa nascendo à frente do robô e quatérnion identidade, o `−X` dela aponta de volta para
ele; "zero voltas" é a orientação de nascimento neutra. A dificuldade mora na **orientação
de nascimento**, não em qual face se pede. `FACE_AXES` (as 6 faces) fica no arquivo só como
documentação.

### 8.6 Os σ por env — a decisão de maior consequência

```python
def _recalcula_sigmas(self, ids):
    d_palma = self.dist_palma_caixa(ids)
    d_alvo  = norm(caixa_pos − self._command[ids, ALVO])
    self.sigma_alcance[ids] = (d_palma · sigma_fator).clamp(min=0.08)
    self.sigma_trazer[ids]  = (d_alvo  · sigma_fator).clamp(min=0.08)
    self._atualiza_face(ids)
    self.sigma_ori[ids]     = (self._command[ids, ANG] · sigma_fator).clamp(min=0.20)
```

**Os σ NÃO são knobs. Cada um é a distância inicial daquele env.**

Medição que justifica: a palma nasce a **0,339 m** da caixa (mín 0,211, máx 0,481). Com σ
**fixo** de 0,10 m o kernel `exp(−d²/σ²)` vale **1e−05** ali, **e a derivada é ZERO** — o
robô move a mão 1 cm para perto e nada muda, 1 cm para longe e nada muda. **Não existe
pista de onde ir.** Foi isto que travou o `g1_poc`, e não uma preferência do robô por ficar
parado.

Com `σ = d₀`, todo env nasce em `exp(−1) = 0,368` com derivada `2/d₀ × 0,368`: **3,49** no
env mais perto e **1,53** no mais longe. Vivo nos dois extremos, e sem número mágico.
(O `smoke` mede 0,3679 a 0,3708 em 32 envs.)

**Onde eles são calculados:** só em dois pontos, e os dois têm pose fresca — na passada do
`_pendente` (1º passo depois do reset) e no `_avanca_elo_force` (avanço de elo).

**Pré-registrado:** se o alcance não aparecer, `sigma_fator` vai a **1,5**. É o primeiro e
único número a mover, e **nunca o peso** — tornar o 1º centímetro positivo exigiria peso
> 12, quatro vezes o da locomoção, e o robô pararia de andar.

### 8.7 `dist_palma_caixa` — bimanual e lateral

```python
def alvos_das_palmas(self, ids):
    off = zeros_like(caixa); off[:, 1] = caixa_meia_aresta   # 0,10
    off = quat_apply(caixa_quat, off)                         # gira COM a caixa
    return stack((caixa + off, caixa − off), dim=1)           # [k, 2, 3]

def dist_palma_caixa(self, ids):
    palmas = robo.data.site_pos_w[ids][:, self._ids_palma, :]
    return norm(palmas − self.alvos_das_palmas(ids), dim=−1).mean(dim=1)
```

O alvo de **cada** palma é o centro da **sua** face lateral. A esquerda mira `+y` da caixa,
a direita mira `−y` — e isso casa com a geometria dos pads (pad esquerdo em `y = −0,015`,
direito em `+0,015`), portanto as duas palmas **olham uma para a outra**.

> ⚠ Até 28/08 isto era `min` sobre as palmas contra a **superfície de uma esfera**:
> `d = ‖palma − centro‖.min(palmas) − meia_aresta`. Dois buracos, e o bloco 3 caiu nos dois:
>
> 1. Com `min`, **uma** mão satura o kernel e a segunda não tem gradiente nenhum — mas o
>    `squeeze` exige as **duas** (ele é `min` das forças). A cadeia ficava sem ponte entre
>    "uma mão encosta" e "as duas apertam": `staged` parado no valor de nascimento e
>    `squeeze` em **0,0002** depois de 3200 iterações.
> 2. Com a esfera, tocar o **topo**, a **frente** ou a **base** paga igual a tocar a
>    lateral — portanto não existe gradiente para a pose de pega.
>
> **A MÉDIA é o que acopla as duas mãos:** uma mão atrasada derruba o termo, portanto as
> duas se aproximam juntas. O máximo é a pose **pré-pega**, com as palmas flanqueando a
> caixa — e é ela que torna o pad de **dorso** geometricamente errado, que é como se
> dispensa o `back_penalty`.

E **não subtrai mais a meia-aresta**: o alvo já está na superfície.

### 8.8 As 4 cadeias

```python
CADEIAS = ((PEGAR,),
           (REORIENTAR, PEGAR),
           (PEGAR, CARREGAR),
           (PEGAR, CARREGAR, BOTAR))   # v2: pegar, SEGURAR PARADO, botar (spec §6.5)
CADEIA_NENHUMA = −1        # o ANDAR
# `_SEGURA_PARADO`, derivado de CADEIAS: a cadeia em que CARREGAR é seguido de BOTAR.
# Nela o CARREGAR tem twist zero e fecha por `perto` sustentado pela espera sorteada.
```

**Teto de 2 elos.** Uma cadeia de 3 exigiria navegação. **O `PEGAR` aparece em todas** — é
o eixo do qual não se esquece, e isso é o **piso anti-esquecimento estrutural**, sem knob:
toda cadeia de 2 elos passa pelo 1º.

Tabelas derivadas de `CADEIAS`, **nunca redigitadas** (`_PRIMEIRO_ELO`, `_N_ELOS`,
`_ELO_EM`) — *"uma tabela paralela escrita à mão sai de sincronia no dia em que uma cadeia
mudar."*

### 8.9 Como a cadeia é sorteada — o ponto mais delicado do arquivo

```python
elo_atual = self._elo[env_ids]                      # do currículo (a FATIA)
tab    = tensor(prob_por_nivel)                     # [7 níveis × 4 cadeias]
linha  = tab[nivel]
compat = _PRIMEIRO_ELO.unsqueeze(0) == elo_atual.unsqueeze(1)   # só cadeias compatíveis
pesos  = linha · compat.float()
tem    = pesos.sum(dim=−1) > 0
escolha = torch.multinomial(where(tem, pesos, ones), 1)
self._cadeia[env_ids] = where(tem, escolha, CADEIA_NENHUMA)
```

> ⚠⚠ **Uma versão anterior inverteu isto**: ela sorteava a cadeia e depois **sobrescrevia**
> `self._elo` com o 1º elo dela. Como o 1º elo de **três das quatro** cadeias é o `PEGAR`,
> **todos** os envs viravam `PEGAR` — e a fatia de locomoção de 95% era **apagada**. O
> módulo inteiro existe para não entregar as transições à manipulação cedo demais, e aquele
> `=` fazia exatamente isso, sem uma linha de log.

**A ordem correta é a inversa:** quem decide se o env é de **locomoção** ou de
**manipulação** é o currículo (a fatia). A cadeia só escolhe **qual transição praticar**,
**dentro** da manipulação — e ela tem de **começar** no elo que o currículo já sorteou.

> ⚠ **Sem `prob_por_nivel` no cfg a máquina de elo é INERTE, e em silêncio.** O default do
> campo é `()`, e o sorteio cai no ramo "não há cadeia". Foi o que aconteceu na primeira
> entrega — toda cadeia saía 0 e nenhum env avançava, sem nenhum erro.

`prob_por_nivel` (`knobs.Cadeia`), 7 linhas × 4 colunas, cada linha soma 1,0:

| nível | `(PEGAR,)` | `REORIENTAR→PEGAR` | `PEGAR→CARREGAR` | `PEGAR→BOTAR` |
|---|---|---|---|---|
| 0 | **0,80** | 0,10 | 0,05 | 0,05 |
| 1 | 0,75 | 0,10 | 0,10 | 0,05 |
| 2 | 0,60 | 0,20 | 0,10 | 0,10 |
| 3 | 0,40 | 0,25 | 0,20 | 0,15 |
| 4 | 0,20 | 0,25 | **0,30** | 0,25 |
| 5 | 0,15 | 0,25 | **0,35** | 0,25 |
| 6 | 0,10 | 0,25 | **0,35** | 0,30 |

Nível baixo concentra na cadeia de 1 elo (robustecer a pega antes de transições); nível alto
favorece as de 2 (com dificuldade física alta, **a transição é o aprendizado**).

### 8.10 As condições de fechamento — `_fecha_elo_corrente`

```python
perto    = ‖caixa − alvo‖ <= tol_pos                          # 0,10 m
alinhado = ANG <= radians(tol_ang_deg)                        # 25°
de_pe    = pelve_z >= pelve_alvo                              # 0,75 m
apoiada  = F_apoio >= fracao_do_peso_apoiada · m·g            # 0,5 · m·g
andou    = ‖base_xy − _pos_no_elo_xy‖ >= carregar_dist_m       # 0,50 m
```

| elo | condição | sustain |
|---|---|---|
| REORIENTAR | `perto & alinhado` | 0,3 s |
| **PEGAR** | `perto & alinhado & de_pe` | **0,5 s** |
| CARREGAR | `perto & andou` | 1,5 s |
| BOTAR | `perto & alinhado & apoiada` | 0,3 s |

**O `PEGAR` tem o sustain MAIOR de propósito:** ele é o elo do qual todas as cadeias
dependem, e um fecho por acidente de um frame propagaria para os outros três. (Um comentário
anterior no `knobs.py` dizia o contrário do que os valores fazem — foi corrigido.)

> ⚠⚠ **`CARREGAR` exige DESLOCAMENTO, e não só tempo.** Defeito medido em 26/08: o `pegar`
> e o `carregar` publicam **exatamente o mesmo alvo**, e as condições eram
> `PEGAR = perto & alinhado & de_pé` e `CARREGAR = perto`. `perto` é **subconjunto** da
> condição do `pegar` sobre um alvo que não muda. Portanto no instante em que o `pegar`
> fechava, o `carregar` **já estava satisfeito**: o robô ficava parado 1,5 s e a cadeia era
> marcada como **sucesso** — o que **move o currículo de nível**. A cadeia `pegar → carregar`
> treinava **"não andar"**.
>
> Derivação do 0,50 m: a faixa de comando do fabricante vai a 1,0 m/s e o portão da F1 exige
> rastrear metade dela, logo um robô aprovado cobre ~0,5 m em 1,0 s. Com `carregar_s = 1,5 s`
> o pedido é conservador.

> ⚠ **Sem `try/except` na condição `apoiada`.** Uma versão anterior lia um sensor
> inexistente (`contact_caixa_laje`), pelo método errado (`robot.find_sites` — um **sensor**
> não é um **sítio**, e ele vive na **cena**), dentro de um `try` cujo `except` deixava
> `apoiada = True`. Resultado: o `BOTAR` fechava com `perto & alinhado` apenas, e sem uma
> linha de erro. *"Se o sensor não existir, isto TEM de explodir."*

> ⚠ O limiar de "apoiada" é uma **fração do peso**, não newtons. Um limiar fixo de 2 N
> significaria "apoiada" com carga de 1 kg (9,8 N) e "no ar" com 5 kg mal encostada.

### 8.11 A máquina de estados — `_avanca_elo`

```
                         a cada passo, dentro de _update_command

   ┌──────────────────────────────────────────────────────────────────┐
   │  nao_fechou = todos[~self.fechou]                                │
   │  fecha = _fecha_elo_corrente(nao_fechou)                         │
   │  _sust = where(fecha, _sust + dt, 0.0)      ← ZERA se soltar     │
   │  sustain_alvo = por elo (0,5 / 1,5 / 0,3)                        │
   │  tem_cadeia = _cadeia >= 0                  ← o ANDAR fica fora  │
   │  deve_avancar = (_sust >= sustain_alvo) & tem_cadeia              │
   └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼  _avanca_elo_force(ids)
        ┌──────────────────────────┴──────────────────────────┐
        │ pode = tem & (passo + 1 < n_elos)                   │
        ▼                                                      ▼
   TEM 2º ELO                                            É O ÚLTIMO
   _passo += 1                                           fechou = True
   _elo = _ELO_EM[cad, passo]                            metrics["sucesso"] = 1
   _sust = 0                                             (o episódio CONTINUA)
   _aplica_elo(so_pose=False)   ← novo alvo, laje se move
   _recalcula_sigmas            ← σ do novo elo
   _pos_no_elo = base_pos       ← âncora de deslocamento
   avancou = True               ← lido por recompensas.sustentacao
```

Quatro armadilhas consertadas neste bloco:

1. **`tem_cadeia` é obrigatório.** Sem ele, um env de `ANDAR` entrava em
   `_avanca_elo_force` **a cada passo**: o laço por elo não cobre o `ANDAR`, portanto o
   `sustain_alvo` ficava no zero do `torch.zeros`, e `0 >= 0` é `True`. Com 95% dos envs em
   locomoção, isso eram ~95% dos envs entrando na função todo passo de física.
2. **`CADEIAS[-1]` devolve a ÚLTIMA cadeia em Python.** Um env de `ANDAR` era tratado como
   `(PEGAR, BOTAR)`, em silêncio. Por isso `elo_de()` lê o **buffer** `_elo` e nunca
   reconstrói de `CADEIAS[cadeia][passo]`, e `n_elos_da_cadeia()` devolve **1** para quem não
   tem cadeia. O `__init__.py` registra tasks de inspeção para os **cinco** elos, três dos
   quais caem em `−1`.
3. **`f = ids[tem & ~pode]`, e não `~pode`.** Sem o `tem`, um env de `ANDAR` em que o
   inspetor chamasse `forca_avanco` seria marcado como **sucesso de manipulação**.
4. **`dt` vem do env (`self._env.step_dt`), não de `1.0/50.0`.** O literal acerta hoje e
   passa a mentir no dia em que a decimação ou o timestep mudar — e o cronômetro erraria em
   silêncio. O `smoke` afirma que a string `"1.0 / 50.0"` **não aparece** em `comando.py`.

**O avanço não reseta o episódio.** Não há reset nem resample: o robô, a caixa e as
velocidades continuam. O one-hot acompanha o elo sem corte de episódio, porque ele é lido do
comando por passo (§7.1).

**O episódio não termina no sucesso.** `fechou = True` trava a métrica e para o avanço; o
episódio segue até o `time_out`. Terminar cedo descartaria a recompensa do tempo restante, e
isso **puniria** o sucesso.

**As métricas são escritas todo passo, para todos os envs:**

```python
self.metrics["fatia_cadeia"][:] = (n_el > 1).float()
self.metrics["passo_final"][:]  = self._passo.float()
```

Antes elas só eram escritas dentro do `_avanca_elo_force`, isto é só no instante de um
avanço — logo um env de cadeia que **nunca** fechasse o 1º elo nunca escrevia
`fatia_cadeia = 1`, e o painel ficava perto de zero no começo do treino. A escada leria isso
e diagnosticaria *"as cadeias não estão sendo sorteadas"*, que é o oposto do que estaria
acontecendo.

### 8.12 O movimento da laje — `_laje_para`

```python
pose[:, 2] = topo − prateleira_meia_z          # a pose mocap é o CENTRO
self.prateleira.write_mocap_pose_to_sim(pose, env_ids=ids)
if sobe_caixa:
    pc[:, 2] = topo + caixa_meia_z
    self.caixa.write_root_link_pose_to_sim(pc, ...)
```

| elo | destino da laje | leva a caixa? |
|---|---|---|
| ANDAR | `afasta_z = 5,0 m` | **sim** (senão o robô tropeça nela) |
| CARREGAR | `afasta_z = 5,0 m` | não (a caixa está nas mãos) |
| BOTAR | topo novo sorteado | não |
| REORIENTAR, PEGAR | não se move | — |

**Isto não é um teleporte da caixa.** No `ANDAR` a caixa vai junto porque a laje é o apoio
dela; nos elos de manipulação o treino move só a mobília.

### 8.13 A arma da terminação — `_publica_pegou`

```python
tocou = AND sobre os dois sensores de palma de (found > 0).any(-1)
self._pegou |= tocou                       # MONOTÔNICO dentro do episódio
self._env.limpo_pegou = self._pegou.float()
```

"As duas palmas já tocaram a caixa **neste** episódio." Zerada só no `_resample_command`.

- **Monotônica de propósito:** soltar a caixa para reposicionar não desarma a terminação —
  se desarmasse, largar de vez deixaria de terminar.
- **Mora no comando, e não na terminação,** porque aqui existe escopo de episódio. Um termo
  de terminação é uma função sem `reset`, e o estado dele vazaria de um episódio para o
  outro.
- **Republica o tensor todo passo** em vez de guardar referência: `.float()` cria um tensor
  **novo**, portanto uma publicação única no `__init__` congelaria o valor em zero para
  sempre.

### 8.14 `resampling_time_range = (1e9, 1e9)` — nunca resampleia no meio

> ⚠ **Nunca igual à duração do episódio.** Com `(20, 20)` o `time_left` do comando cruza
> zero no passo 999 e o `time_out` da terminação só dispara no 1000: o resample rodava **um
> passo antes do fim** e zerava o sucesso do episódio. O nível lia sucesso 0 em **todo**
> episódio que chegava ao time_out.
>
> A meta é 1 resample por episódio, e **quem resampleia é o RESET**.

### 8.15 `TwistComRazaoDeMarcha` — as duas réguas da locomoção

Subclasse de `UniformVelocityCommand`, reconstruída campo a campo por `dataclasses.fields`:

```python
antigo = cfg.commands["twist"]
campos = {f.name: getattr(antigo, f.name) for f in dataclasses.fields(antigo)}
cfg.commands["twist"] = CMD.TwistComRazaoDeMarchaCfg(**campos, limiar_comando=0.05,
                                                     pedido_min_segmento=0.5)
```

Copiar campos à mão perderia em silêncio qualquer campo que um upgrade de mjlab adicione —
e um `rel_standing_envs` perdido mudaria 10% dos envs sem uma linha de log.

**JUIZ (desde 27/08) — `eficiencia_min`.** Por **segmento** de comando:

```
e_s = Σ(v_real · v̂_cmd)·dt / Σ‖v_cmd‖·dt        no segmento s
portão = min(e_s)
```

*"Em média, que fração da velocidade comandada você entregou na direção comandada, no pior
segmento?"* 1,0 = entregou tudo. Negativo = foi para o lado errado. Perpendicular = 0.

**DIAGNÓSTICO — `razao_marcha`.** `1 − Σ‖v_cmd_xy − v_xy‖ / Σ‖v_cmd_xy‖`.

> ⚠ **Por que a troca, e é medição.** A `razao_marcha` é soma de **normas**, e norma nunca
> cancela: com `v_real = v_politica + ruído`, `Σ‖(v_cmd − v_pol) − ruído‖ > Σ‖v_cmd − v_pol‖`
> para ruído de média zero. Logo ruído **sempre** a infla, monotonicamente.
>
> **Medido no bloco 1:** o `std` subiu de 0,43 (it 1525) para 0,61 (it 4999) — porque a
> manipulação entrou e exploração voltou a valer — e a razão caiu de **0,514 para 0,426**,
> enquanto **duração** (984 → 988) e **queda** (0,000 → 0,167) não se moveram e o `play`
> determinístico andava bem. O portão congelou na banda morta e a rampa deu **um** degrau em
> 1341 iterações. *Ele leu ruído de ação como incompetência.*
>
> A projeção conserta pela forma: `Σ(ruído · v̂_cmd)` tem média zero e encolhe com `1/√N`.

Três outras decisões:

- **Por segmento, e não por episódio.** Média sobre o episódio inteiro deixaria o robô
  **trocar tempo** (parar 10 s e compensar depois). Cada segmento é pontuado sozinho.
- **`min` e não média.** Um robô que anda reto e não gira mostra média alta e mínimo baixo, e
  é o mínimo que responde "sabe andar". O `error_vel_yaw` está em ~2,5 há 5000 iterações e
  nenhum portão olhava para ele.
- **As duas nascem em 0,0 (pessimistas).** Robô imóvel projeta zero e tem erro igual ao
  comando. É o oposto exato do portão que media **duração de episódio**: aquele dava nota
  máxima à estátua, porque estátua não cai.

**Fronteira de segmento**, detectada pelo `command_counter`:

```python
atual = self.command_counter
mudou = atual != self.metrics["seg_visto"]
if mudou.any():
    self._fecha_segmento(mudou); self.metrics["seg_visto"][:] = atual
```

A ordem do mjlab torna isso correto: `CommandTerm.compute` chama `_update_metrics`
**antes** do `_resample`, portanto nesta chamada o comando e o contador ainda são os do
segmento que está correndo.

**Validade do segmento por `seg_pedido >= 0,5`** — uma regra descarta dois casos de uma vez:
o comando quase nulo (`is_standing_env`, 10% dos envs), cujo denominador é ruído; e o
fragmento curto do fim do episódio, onde o ruído de ação ainda não cancelou.

**A máscara é obrigatória** em `_fecha_segmento`: num lote de 4096 cada env re-sorteia no seu
próprio instante, então sem a máscara o zeramento apagaria o acumulador dos envs que estão
**no meio** do segmento deles.

**E o primeiro segmento não pode entrar no `min` como se houvesse um anterior:** com
`segmentos == 0` o mínimo **é** a eficiência dele, e não `min(e, 0.0)` — que travaria a
métrica em zero para sempre.

**As somas vivem em `self.metrics`**, e isso não é conveniência: `CommandTerm.reset` lê a
métrica, tira a média dos envs e **só depois** zera — e o `reset` do comando roda **depois**
do currículo. Portanto o consumidor lê o episódio que **acabou**.

---

<a id="9"></a>
## 9. As recompensas: 9 termos próprios + a tabela do fabricante

O conjunto final é: **todos os termos do molde `unitree_g1_flat_env_cfg`**, com os pesos
reescritos por `aplica_pesos`, **mais 9 nomes novos**. O `smoke.py` afirma exatamente isso:

```python
set(cfg.rewards) − set(fabricante) == {
    "terminacao", "joint_acc",                                   # os 2 da F1
    "staged", "precise_pos", "precise_ori",                      # os 7 da F3
    "squeeze", "unload", "postura_ereta", "sustentacao"}
```

e **nada do molde foi removido**.

### 9.1 A tabela da locomoção — `knobs.Recompensa`

Todos os pesos são **por segundo** (`scale_rewards_by_dt = True`).

| termo | peso | nota |
|---|---:|---|
| `track_linear_velocity` | **+2,0** | do fabricante |
| `track_angular_velocity` | **+2,0** | do fabricante |
| `upright` | **+1,0** | é o que segura o robô de pé nos elos de manipulação |
| `pose` | **+1,0** | `variable_posture`, σ **colhidos** do molde |
| `air_time` | **0,0** | o único positivo de marcha, e o fabricante o entrega em zero |
| `action_rate_l2` | **−0,10** | destravou o andar num bloco medido |
| `dof_pos_limits` | **−1,0** | |
| `foot_clearance` | **−2,0** | |
| `foot_swing_height` | **−0,25** | alvo 0,10 m; `func` trocado por `AlturaDeBalanco` |
| `foot_slip` | **−0,1** | |
| `soft_landing` | **−1e−5** | |
| `body_ang_vel` | **−0,05** | |
| `angular_momentum` | **−0,02** | |
| `self_collisions` | **−1,0** | |
| `terminacao` | **−200,0** | `is_terminated` — **novo**, do `g1_multitask` |
| `joint_acc` | **−2,5e−7** | **novo**; medido em 0,0006/s, entra por paridade |

**A tabela é escrita INTEIRA, inclusive os pesos idênticos ao default.** Não é redundância:
é a trava contra deriva silenciosa. Um upgrade de mjlab que mude um default passaria sem
erro e sem log — e o `escala_acao_mult` já é o único número pré-registrado para mexer se o
portão falhar; não pode haver um segundo suspeito entrando pela porta de trás.

**`terminacao = −200` não custa 200:** o passo que termina paga `−200 × 0,02 = −4,0`. E
`is_terminated` **exclui o `time_out`** (ele lê `termination_manager.terminated`), portanto
terminar o episódio pelo tempo **não** é punido — só cair é. Este único termo é o que cobra
**todas** as terminações de falha; nenhuma penalidade é somada em cima.

**`AlturaDeBalanco(feet_swing_height)` — um bug do molde, e ele é silencioso.**

O termo do fabricante acumula `peak_heights` por pé e só zera no primeiro contato. Mas
`reward_manager.py:174` só registra um termo de classe em `_class_term_cfgs` — a lista dos
que recebem `reset(env_ids)` — quando a classe **tem** um método `reset`. O
`feet_swing_height` não tem.

Consequência: quando o episódio termina com um pé no ar (isto é, **toda vez que o robô
cai**), o pico daquele pé sobrevive ao reset e entra no episódio seguinte. O
`Metrics/peak_height_mean` **infla com queda**, e o painel mostra "o passo está subindo"
exatamente quando o robô está caindo mais. *"Foi assim que um bloco leu `peak_height` em
alta durante 5000 iterações com o robô imóvel: a altura vinha do vôo da queda."*

O conserto tem **três linhas e nenhum número**.

### 9.2 `PosturaPorElo` — o termo que fica calado

```python
def __call__(self, env, *args, canal_do_elo, nome_do_comando, elos_que_andam, **kw):
    valor = super().__call__(env, *args, **kw)
    elo = env.command_manager.get_command(nome_do_comando)[:, canal_do_elo].long()
    anda = torch.isin(elo, tensor(elos_que_andam))
    return torch.where(anda, valor, torch.ones_like(valor))
```

**Por que NÃO é um 4º regime de σ**, que era o desenho do plano original: porque foi medido,
e **nenhum σ resolve**. O termo é `exp(−média(erro²/σ²))` sobre 29 juntas; com 17 delas fora
do default ele é um produto de 17 gaussianas, e **colapsa para qualquer σ**.

Medido em 26/08, com a excursão em fração da faixa real de cada junta (faixa média das 17
juntas de manipulação: 3,77 rad):

| fração da faixa | standing | walking | running | run×3 | run×5 |
|---:|---:|---:|---:|---:|---:|
| 0,10 | 0,000 | 0,014 | 0,184 | 0,829 | 0,935 |
| 0,20 | 0,000 | 0,000 | 0,001 | 0,471 | 0,763 |
| 0,30 | 0,000 | 0,000 | 0,000 | 0,184 | 0,544 |
| 0,40 | 0,000 | 0,000 | 0,000 | 0,049 | 0,338 |

Três coisas saem daí:

1. `std_standing` é **uma entrada só**, `.*` = 0,05, para **todas** as 29 juntas. E o
   `walking_threshold` do G1 é **0,05**, não 0,5 (medido no cfg). Com o twist forçado a zero
   num elo de manipulação, `total_speed = 0 < 0,05` **sempre** — logo o regime `standing` é
   **certo**, não provável.
2. O termo não vale 0,93/s a menos: ele vale **exatamente zero**, já a 10% da faixa.
   `exp(−muito)` é 0 em float32, **com gradiente zero**. Não é penalidade forte, é **canal
   morto**.
3. Nem `running×5` sobrevive a 40% da faixa. Um multiplicador só empurra o penhasco alguns
   centímetros para a direita.

E **excluir os braços não basta**: com os braços fora da média, um braço esticado custa
0,000 — ótimo — mas um **agachamento** com as pernas em `running` dá 0,128 a 10% da faixa e
0,000 a 20%. E o nível 4+ põe a laje a 0,04 m, o que **exige** agachar.

**Portanto o termo não tem o que dizer num elo de manipulação, e a resposta certa é ficar
calado.** O que segura o robô de pé passa a ser o `upright` (+1,0, independente de elo) mais
a própria condição de fechamento do elo — o `PEGAR` só fecha "de pé".

> ⚠ **Retorna 1,0, e não 0,0.** Zero faria o env de manipulação pagar 1,0/s a menos só por
> estar naquele elo — uma **penalidade por sorteio**. Um faz o termo **neutro**, e mantém a
> escala de retorno comparável entre elos, que é o que o controlador de fatia da F5 lê.

Os braços seguem contidos por cinco termos que **não** dependem de elo: `action_rate_l2`,
`joint_acc`, `angular_momentum`, `body_ang_vel`, `dof_pos_limits` e `self_collisions`. *"Não
é terra sem lei — é só a instrução 'volte à pose default' que sai."*

**Os σ são COLHIDOS, não redigitados** (`colhe_sigmas_de_postura`). O `smoke` prova por
identidade de objeto, e vai além: ele afirma que a palavra `knee` **não aparece em nenhum
fonte do pacote** (excluindo `smoke.py` e `paridade.py`, que se auto-acusariam).

### 9.3 Os 7 incentivos de manipulação — `knobs.Tarefa`

**Todos positivos e contínuos. Nenhuma penalidade, nenhum booleano.**

| termo | peso | forma |
|---|---:|---|
| `staged` | **+3,0** | `alcançar × (1 + trazer)` — o motor da fase inicial |
| `precise_pos` | **+2,0** | `exp(−‖caixa−alvo‖²/0,05²)` — σ **fixo** |
| `precise_ori` | **+1,0** | `alcançar × exp(−(Δθ/σ_ori)²)` |
| `squeeze` | **+1,0** | `tanh(min(F_n_E, F_n_D)/F_ref)` |
| `unload` | **+2,0** | `(1 − F_apoio/mg) × preensão` |
| `postura_ereta` | **+2,0** | `rampa_pelve × unload` |
| `sustentacao` | **+0,5** | `t_na_condição / 1,0 s` |
| **soma** | **11,5/s** | contra o **piso da estátua de 5,81/s** (medido 26/08) |

Razão ~2:1 no fecho completo, e é a resposta à pergunta *"ficar parado paga mais que agir?"*.

Três regras que valem para os sete:

1. **Todos multiplicam por `VALIDA`.** Sem o gate, um env de `ANDAR` pagaria o **máximo**:
   com os canais de caixa zerados, `exp(0) = 1`. O `smoke` mede: no `ANDAR` os sete valem
   exatamente 0 (`< 1e−9`).
2. **Todos os σ vêm do termo de comando, por env** (§8.6) — nunca do knob. A única exceção é
   o `precise_pos`.
3. **Nenhuma penalidade aqui, e é princípio (R3):** *penalidade limita COMO fazer o que já
   existe; ela não ensina a fazer.* E booleano é platô — o `pegar` do `g1_poc` travou 22 mil
   iterações num `squeeze` booleano.

#### `staged` — produto, e não soma

```python
alcanca = exp(−(dist_palma_caixa / sigma_alcance)²)
traz    = exp(−(dist_caixa_alvo  / sigma_trazer)²)
return alcanca · (1 + traz) · VALIDA
```

> ⚠ **A forma é PRODUTO.** `trazer` só paga se a mão já estiver perto. Com soma, o robô
> ganharia por **empurrar a caixa até o alvo com o pé** — e foi assim que uma run antiga
> aprendeu a **chutar a caixa**.

Teto **2,0**, não 1,0. Com peso 3,0 ele contribui até **6,0/s**. É o maior termo do conjunto
de propósito: **ele é o único que tem gradiente na pose de repouso.**

#### `precise_pos` — o único com σ fixo

```python
return exp(−(dist_caixa_alvo / 0.05)²) · VALIDA
```

Ele responde *"a caixa está NO alvo?"*, que é uma **tolerância de aceite**, não uma rampa de
aproximação. Quem faz a rampa é o `staged`, com σ por env. **Dois termos, duas perguntas.**

#### `precise_ori` — gateado pelo alcance

```python
return alcancar · exp(−(ANG / sigma_ori)²) · VALIDA
```

Girar a caixa **sem tocá-la** não é a tarefa. E o σ é o ângulo inicial daquele env: com σ
fixo de 0,40 rad um pedido de 90° dava `exp(−(1,57/0,40)²) = 2,0e−7`, isto é **zero** — era
a "sorte de nível 3+" medida no `g1_poc`.

#### `squeeze` — o termo com gradiente na coordenada do aperto

```python
def _forca_das_palmas(env, sensores, asset_cfg):
    quat   = robo.data.site_quat_w[:, asset_cfg.site_ids]           # [B,2,4]
    locais = tensor([[0,−1,0], [0,+1,0]])                            # E olha −y, D olha +y
    normais = quat_apply(quat, locais)                               # [B,2,3]
    for i, s in enumerate(sensores):
        f = env.scene[s].data.force                                  # global (netforce)
        fs.append(sum(f · normais[:, i]).abs().sum(−1))
    return stack(fs).min(−1).values                                  # MIN, não soma

def _forca_ref(env, mu):
    return (env.limpo_massa · 9.81 / (2·mu)).clamp(min=1e−3)          # m·g/(2μ)

squeeze = tanh(_forca_das_palmas(...) / _forca_ref(...)) · VALIDA
```

**O diagnóstico que justifica o termo.** A política só age em alvos de junta. Para a caixa
subir, a força de atrito tem de vencer o peso. Antes disso a caixa não se move, e nenhuma
recompensa muda:

| derivada da recompensa em relação a… | valor |
|---|---|
| a altura da caixa | grande |
| **a força de aperto** | **zero** |

É um **degrau**, não uma rampa. O `reaching` não conserta: a palma não penetra a caixa,
portanto ele **satura no contato**, e apertar mais forte não o move. O `squeeze` é o único
termo com derivada positiva de 0 N até `F_ref` — exatamente o vão que não pagava nada.

Quatro decisões dentro dele:

- **`min` das duas palmas.** Uma palma sozinha **empurra** a caixa, ela não a segura. Com
  soma, apertar forte com uma mão pagaria tanto quanto pegar com as duas.
- **Só a componente NORMAL ao pad.** Este é o anti-hack, e a reescrita o tinha **perdido**:
  até 28/08 isto era `‖F‖`, a magnitude inteira. Com a magnitude, **apertar a caixa para
  baixo contra a prateleira paga como preensão** — e aquela força é **tangencial** ao pad. A
  projeção a descarta sem precisar de um segundo termo.
- **A normal vem da orientação do SÍTIO**, e não do campo `normal` do sensor: com
  `reduce="netforce"` "a normal do contato" perde significado. `abs()` em vez de sinal —
  errar a convenção zeraria o termo em silêncio.
- **`F_ref` é DERIVADO, não escolhido.** Com `m = 1,0 kg` e `μ = 0,8` dá **6,13 N**. Até
  28/08 era um knob fixo de **12,0 N** sem derivação: pedia o **dobro** do necessário e
  pagava **metade** no primeiro newton — justo a faixa em que a preensão tem de nascer. E a
  massa é **por env**, portanto uma caixa mais pesada exige mais aperto e o termo acompanha
  sozinho.

- **`tanh` e não limiar.** Contínuo desde a primeira décima de newton, portanto existe
  gradiente **antes** de a preensão "existir".

#### `unload` — a ponte do `pegar`, com porteiro

```python
f        = norm(env.scene["apoio_caixa"].data.force).squeeze(−1)
descarga = (1 − f / (m·g)).clamp(0, 1)
preensao = tanh(_forca_das_palmas(...) / _forca_ref(...))      # ← O PORTEIRO
return descarga · preensao · VALIDA
```

Medido 27/08 erguendo a caixa da laje: `F_apoio` 9,80 N → 0,00 N e o termo 0,0005 → 1,0.

> ⚠ **Mas a transição é ESTREITA**, e isto corrige o que o docstring afirmava antes:
> erguendo a caixa em degraus, ela salta de 0 a 1 em **2 mm**. O contato é rígido, e a força
> de apoio não passeia por valores intermediários. **Como rampa de altura, o `unload` é
> quase um booleano.**
>
> E o método de medição não decide o caso que importa: a caixa foi **teleportada**, portanto
> o teste não modela **partilha de carga**. Numa pega real a força da palma sobe enquanto a
> do apoio desce, e as duas somam `m·g` — ali a força de apoio **passa** pelos valores do
> meio. Isso só uma run com preensão mede.
>
> Consequência de desenho: o gradiente de **aproximação** vem do `staged` e o de **força**
> vem do `squeeze`. O `unload` marca *"a caixa saiu da laje"* — é um **bônus**, não a rampa.
> É bom que os três não dependam um do outro.

> ⚠⚠ **O porteiro de preensão, acrescentado em 28/08 por medição.** Sem ele o termo lê só "a
> caixa não pesa na laje", e **derrubar a caixa satisfaz isso perfeitamente**: uma vez no
> chão, `F_apoio` é zero para sempre e o termo paga **2,0/s pelo resto do episódio, sem mão
> nenhuma**. No bloco 3, it 4251, o `unload` marcava 0,0995 com o `squeeze` em 0,0002 —
> **descarga sem preensão**, que é a assinatura desse atalho.
>
> O porteiro é `tanh(F_palmas/F_ref)`, o **mesmo** fator do `squeeze`, portanto contínuo
> desde o primeiro newton: fecha o atalho sem virar degrau.

A **outra metade** do porteiro é a terminação `caixa_largada` (§10.3): o porteiro tira o
pagamento de *"derrubar sem pegar"*, a terminação tira o de *"pegar e largar"*.

#### `postura_ereta` — paga por erguer SEM agachar

```python
z     = pelve_z − env_origin_z
rampa = ((z − 0.45) / (0.75 − 0.45)).clamp(0, 1)          # pelve_piso, pelve_alvo
return rampa · unload(...)                                 # o unload já traz VALIDA e preensão
```

O alvo já tem z absoluto, o que remove o atalho de **baixar o alvo**; este termo remove o
atalho de **baixar o corpo** para encurtar o alcance.

- **Rampa de dois lados (`clamp(0,1)`):** zero abaixo do piso, um acima do alvo, linear no
  meio. Sem o clamp superior, esticar-se além do alvo pagaria cada vez mais, e o robô
  aprenderia a ficar **na ponta dos pés**.
- **Multiplicada pela descarga, não somada.** Somado, o robô colheria a rampa só por ficar
  de pé sem tocar a caixa — que é exatamente o que ele já faz de graça.
- **A preensão não entra de novo**: o `unload` já a traz desde 28/08. Multiplicar aqui daria
  `preensão²`, que aperta a rampa sem acrescentar informação.

#### `sustentacao` — a classe com cronômetro

```python
class sustentacao:
    def __init__(self, cfg, env): self.t = torch.zeros(env.num_envs); self.dt = env.step_dt
    def __call__(self, env, nome, tol_pos, tol_ang, sustenta_s):
        na_condicao = perto & alinhado & (VALIDA > 0.5)
        self.t = where(na_condicao, self.t + self.dt, zeros)
        avancou = getattr(_t(env, nome), "avancou", None)
        if avancou is not None:
            self.t = where(avancou, zeros, self.t)        # ⚠ ZERA NO AVANÇO DE ELO
        return (self.t / sustenta_s).clamp(max=1.0)
    def reset(self, env_ids=None): self.t[env_ids] = 0.0
```

Paga por **ficar** lá, e não só por passar por lá.

> ⚠⚠ **Zera no avanço de elo.** Sem isto o crédito **vaza** de um elo para o seguinte: o
> `pegar` e o `carregar` pedem o **mesmo** alvo, portanto no instante em que o elo avança a
> condição continua valendo e o `self.t` já está saturado — o `carregar` **nasceria pago**
> sem nenhum trabalho novo. O termo tem o seu **próprio** cronômetro, separado do `_sust` do
> comando, e por isso precisa do seu próprio zeramento.

> ⚠ **O cronômetro lê SÓ a condição da tarefa.** No `g1_multitask` ele lia também o erro
> angular da base, e o `push_robot` (±0,78 rad/s a cada 1 a 3 s) estourava o teste e
> **zerava** o contador: o `perf` do locomover marcou **0** nas iterações 13.700 e 17.297
> **com o robô já andando**. Uma régua que uma perturbação externa zera não mede competência.
> *Push e régua ficam em compartimentos separados.*

### 9.4 O `SceneEntityCfg` — três armadilhas do mjlab num lugar

```python
def _palmas() -> SceneEntityCfg:
    return SceneEntityCfg("robot", site_names=list(C.PALM_SITES))

cfg.rewards["squeeze"] = RewardTermCfg(..., params={..., "asset_cfg": _palmas()})
cfg.rewards["unload"] = RewardTermCfg(..., params={..., "asset_cfg": _palmas()})
cfg.rewards["postura_ereta"] = RewardTermCfg(..., params={..., "asset_cfg": _palmas()})
```

1. **Tem de viver em `params`.** `manager_base.py:141-145` só resolve os que estão lá. Como
   argumento default de função ele **nunca** é resolvido, `site_ids` fica `slice(None)`, e a
   projeção leria os **seis** sítios do robô em vez das duas palmas — sem erro, e com o termo
   medindo outra coisa. (O mesmo bug custou a primeira execução do `metricas.py`, e lá só
   apareceu por sorte de forma: 6 contra 2.)
2. **Uma instância POR TERMO**, não uma compartilhada: o `manager_base` **resolve os ids
   dentro do objeto**, portanto um `SceneEntityCfg` compartilhado por três termos é estado
   mutável compartilhado entre managers. O `smoke` afirma `len({id(...)}) == 3`.
3. **A fonte é `C.PALM_SITES`**, a mesma que o comando recebe em `sitios_palma`. Um segundo
   lugar com a lista seria uma segunda fonte de verdade, e a projeção da força passaria a
   ler outros sítios que o kernel de alcance.

### 9.5 As métricas fora da recompensa — `metricas.py`

**Por que elas saem de dentro dos termos de recompensa.** O fabricante escreve cinco
`Metrics/*` de **dentro** de termos de recompensa. E `reward_manager.py:122` **pula termo com
peso 0**:

```python
if term_cfg.weight == 0.0:
    self._step_reward[:, term_idx] = 0.0
    continue
```

Portanto **desligar uma penalidade apaga a medição dela, em silêncio.** Não é hipótese: o
`air_time` do molde já vem com peso 0,0, e por isso o `Metrics/air_time_mean` do fabricante
**não existe** no painel de quem roda a receita dele. A métrica que responderia *"o robô
levanta o pé?"* está desligada junto com o incentivo — e é exatamente a pergunta da F1.

Os 7 termos, todos **sem peso**:

| métrica | o que mede |
|---|---|
| `momento_angular` | módulo do momento angular do corpo — proxy de balanço de braço |
| `tempo_de_voo` | tempo de voo médio dos pés no ar, em s. **A métrica central da F1** |
| `pico_de_altura` | altura de pico do pé no balanço, medida **no pouso**. Classe **com `reset`** |
| `velocidade_de_escorrego` | velocidade horizontal do pé **apoiado** (deveria ser zero) |
| `forca_de_pouso` | módulo da força no instante do pouso, em N — proxy de impacto |
| `palmas_em_contato` | **0 / 0,5 / 1,0** — nenhuma, uma, ou as **duas** palmas na caixa |
| `dorso_em_contato` | o pad de dorso tocando a caixa. **Tem de ser zero** |

`palmas_em_contato` é **fração, e não `min` nem `any`** — e a razão é uma confissão no
código: *"o `squeeze` usa `min` das forças, portanto uma palma sozinha e nenhuma palma dão o
MESMO zero exato — e essa ambiguidade já me fez ler abandono da tarefa onde havia uma mão na
caixa."*

`dorso_em_contato` existe porque o freio do dorso é **geométrico** (o alcance bimanual põe as
palmas viradas uma para a outra); esta métrica confere se o freio funciona. Se ela sair de
zero, o freio não bastou.

**`pico_de_altura` tem buffer PRÓPRIO**, independente do da recompensa. Não é duplicação por
descuido: se lesse o buffer do termo de recompensa, desligar o peso do `foot_swing_height`
**congelaria a métrica** — que é o defeito que este arquivo existe para consertar.

**Divergência declarada, e é melhoria:** as médias do fabricante são escalares de **lote**
(`sum(...)/num_in_air` sobre o batch), portanto não têm eixo de env. Aqui cada métrica é
**por env**, e o `MetricsManager` faz a média de envs no fim do episódio. O número final é
comparável; a rota é melhor.

> ⚠ E o `MetricsManager` divide por `step_count`, **não** por `max_episode_length_s`. Métrica
> em [0,1] fica em [0,1] no log — **não** existe a diluição que o `Episode_Reward/*` tem
> (§14.3).

---

<a id="10"></a>
## 10. As penalidades e as terminações

### 10.1 O princípio: terminar em vez de penalizar

*Uma trajetória inválida acaba; ela não paga multa.*

> ⚠ **A reescrita perdeu o princípio junto com os termos.** Até 27/08 o `g1_limpo` não tinha
> terminação própria **nenhuma** — só as duas do molde. E o freio do escoro entrou primeiro
> como **penalidade** (`contato_prateleira = −1,5`).
>
> **O bloco 2 rodou 405 iterações com ela e o resultado decidiu:** o contato do tronco caiu
> monotonicamente (7,5% → 3,8% → 2,0% dos passos) **e a manipulação caiu com ele**
> (`staged` 0,36 → 0,17). *Uma multa que o robô pode pagar é uma multa que ele ORÇA* — e com
> o `action_rate_l2` medindo −2,04/s, **escorar sai mais barato que se mover**.
>
> O `g1_poc` registra o mesmo mecanismo do outro lado: quando o movimento encareceu (o
> `action_rate` degrau para −1,00), o `contato_ilegal` dele **subiu de 6,4% para 17,5%** das
> terminações. A terminação não é orçável.

### 10.2 As 7 terminações

| termo | condição | `time_out` | origem |
|---|---|---|---|
| `time_out` | passam 20 s | **True** | molde |
| `fell_over` | inclinação do tronco passa do limite | False | molde |
| `contato_tronco` | pelve/tronco/quadril/coxa × mesa > **50 N** | False | próprio |
| `contato_palma` | pad de palma × mesa > 50 N | False | próprio |
| `contato_dorso` | pad de dorso × mesa > 50 N | False | próprio |
| `caixa_largada` | caixa abaixo de 0,10 m **ou** longe das duas palmas | False | próprio |
| — | `out_of_terrain_bounds` do molde é **removido** pela variante `flat` | | |

**Nenhuma penalidade é somada em cima.** O `terminacao = −200` já cobra todas por
`is_terminated`, que exclui o `time_out`. Portanto o preço de cair ou escorar é **−4,0 no
passo mais todo o retorno futuro perdido**.

### 10.3 `caixa_largada` — armada pela primeira preensão

```python
pegou = getattr(env, "limpo_pegou", None)
if pegou is None: return zeros(bool)
dist    = norm(palmas − caixa.unsqueeze(1), dim=−1)             # [B, 2]
caiu    = (caixa_z − env_origins_z) < z_min                    # 0,10 m
escapou = (dist > dist_max).all(dim=−1)                        # 0,45 m, AMBAS
return (caiu | escapou) & (pegou > 0.5)
```

Quatro detalhes:

- **A arma é obrigatória.** Sem ela **todo episódio começaria terminando**: no reset a caixa
  está na laje e as palmas estão longe, que é exatamente a condição de `escapou`.
- **`escapou` exige as DUAS palmas longe (`.all`)**, e não uma. Uma mão que solta para
  reposicionar é parte de uma pega, não o fim dela.
- **O z é RELATIVO à origem do env.** Com `env_spacing` os envs não estão todos em z = 0, e
  um limiar absoluto acusaria queda no env errado.
- **`z_min = 0,10` é a meia-aresta**, e não uma tolerância escolhida: com o centro nessa
  altura a caixa está **apoiada no chão**. E `dist_max = 0,45` é **maior** que a distância de
  nascimento típica (0,339) — os dois freios são independentes de propósito.

### 10.4 `contato_ilegal` — o limiar é o que torna a lista segura

```python
f = env.scene[sensor_name].data.force
return torch.norm(f, dim=−1).amax(dim=−1) > limiar_N          # 50,0
```

- **Limiar de força, e não booleano.** Roçar o tampo ao alcançar **não** termina; apoiar o
  peso termina. Com booleano, a lista tornaria pose baixa inganhável.
- **50 N é medido no `g1_poc`**, onde a mesma terminação rodou. Não é número novo.
- **`amax` sobre os slots, e não `sum`:** a pergunta é "algum ponto de contato passa de
  50 N", não "a soma de todos passa". Com `netforce` e `num_slots=1` os dois coincidem hoje;
  o `amax` continua certo se alguém subir os slots.

> ⚠ **Espere esta terminação disparar.** O peso do `action_rate_l2` no `g1_limpo` é −0,10 (o
> do fabricante), mas o `Episode_Reward` dele mede **−2,3/s**, a maior conta do conjunto. A
> mesma pressão de *"escorar economiza esforço"* existe. Leia a **fração** dela antes de
> concluir que o robô piorou.

### 10.5 O que NÃO existe como penalidade neste módulo

| termo de módulos anteriores | por que saiu |
|---|---|
| `grasp` booleano | o `squeeze` cobre a mesma faixa, de forma contínua |
| `box_at_peito`, `box_at_prateleira` | é o `precise_pos`, com outro alvo |
| `box_shake`, `box_shake_pegar` | é o `precise_ori` **congelado** (§8.5) |
| `back_penalty` | é o alcance **bimanual e lateral** (§8.7) |
| `com_balance` / `com_over_feet` | é o `upright` + a terminação `fell_over`. Medido: um mergulho de 30 cm custa `0,25² × 2,0 = 0,125/s` contra 11,5/s de tarefa — **1,1%**, quase inerte |
| `table_contact` | é a terminação `contato_ilegal` |
| `contato_prateleira = −1,5` | virou terminação, por medição (§10.1) |
| `sucesso_denso` | é o `precise_pos` |
| `hold_still` | é o regime `standing` da `pose` |
| `hip_deviation` | são os σ laterais apertados |
| equalização de orçamento por tarefa | existe **uma** tarefa |
| `T.gated` / `TERMOS_DE_TAREFA` | o auto-gate por `VALIDA` e pelo comando |

---

<a id="11"></a>
## 11. O currículo: três relógios independentes

`curriculo.py` — **três relógios, e eles não se falam.**

| relógio | eixo | agendado por | onde |
|---|---|---|---|
| **A. `command_vel`** | a faixa do twist | passo global | do fabricante, fica como está |
| **B. `forma`** | a fatia locomoção × manipulação | laço fechado, por sinal medido | `curriculo.forma` |
| **C. `nivel`** | a dificuldade física do objetivo | passeio aleatório por sucesso | `curriculo.nivel` |

A separação é deliberada: **o que a tarefa pede** adapta por sucesso; **quão limpo** o
movimento deve ser não adapta, porque apertar sempre baixa o sucesso.

### 11.1 Relógio C — o nível, por env

```python
def nivel(env, env_ids, *, n_niveis, forcado, frac_uniforme, nome_do_comando):
    buf = garante_nivel(env)
    if forcado is not None:
        buf[env_ids] = clamp(forcado); return media

    # --- O PASSEIO ALEATÓRIO ±1 ---
    cmd = env.command_manager.get_term(nome_do_comando)
    de_cadeia = cmd._cadeia[env_ids] >= 0          # ← só episódios de CADEIA movem
    sucesso   = cmd.fechou[env_ids]
    passo = where(sucesso, 1, −1) · de_cadeia.long()
    buf[env_ids] = (buf[env_ids] + passo).clamp(0, n_niveis − 1)

    # --- O PISO ANTI-ESQUECIMENTO ---
    abertos = int(buf.max()) + 1
    sorteia = rand(len(env_ids)) < frac_uniforme    # 0,20
    buf[env_ids[sorteia]] = randint(abertos, ...)
    return float(buf.float().mean())
```

**Duas propriedades do passeio ±1:**

1. **O nível se equilibra onde a taxa de sucesso é ≈ 50%.** É um passeio aleatório ±1 com
   probabilidade de subir igual a `p(sucesso)`; o ponto fixo é `p = 0,5` **por construção**.
   É por isso que **não existe limiar escolhido à mão** — o `smoke` afirma que não há
   `limiar_competencia` nos knobs.
2. **O rebaixamento é o anti-esquecimento.** Os envs se espalham pelos níveis; sempre há
   envs nos casos fáceis.

**Mas o rebaixamento é distribuição, não garantia** — se a política ficar boa, os envs
empilham no topo e o nível 0 sai do treino. Daí o **piso de 20% sorteado uniformemente**
sobre os níveis abertos. O `smoke` mede os dois casos: com piso sobram envs fora do topo
mesmo com 100% de sucesso; sem piso, **todos** colam no topo.

> ⚠ **Só episódios de CADEIA movem o nível.** Um episódio de locomoção não tem cadeia, não
> tem o que fechar, e portanto "fracassaria" sempre — com a fatia de locomoção em 95%, o
> nível seria empurrado ao piso por episódios que **nem tentaram** a tarefa. O `smoke` chama
> este de *"o check que mais importa da F6"*.

> ⚠ **Piso de NÍVEL ≠ o `rho = 0,30` do `g1_multitask`.** Aquele era piso sobre **tarefas**,
> e com 5 tarefas ocupava 0,75 do sorteio: o teto da locomoção ficava em 0,55 contra os 0,945
> que a fatia de 30% exigia, e a fatia alvo virava **inalcançável**. Piso de nível e piso de
> fatia são eixos **ortogonais**.

> ⚠ **Forçar o buffer de fora NÃO funciona:** o termo de currículo roda no reset e aplicaria
> o passeio por cima. Tem de ser knob (`Nivel.forcado`).

### 11.2 A tabela de níveis — `knobs.Nivel`

**Só o PISO desce; o TETO é fixo. Portanto cada nível CONTÉM o anterior.**

| nível | `topo_min` | `carga_max` | `voltas_max` | `eixo_vertical` | `desalinho_max_deg` | `jitter_x_max` |
|---:|---:|---:|---:|:---:|---:|---:|
| 0 | 0,55 | 1,0 | 0 | — | 15,0 | 0,20 |
| 1 | 0,45 | 2,0 | 0 | — | 20,0 | 0,20 |
| 2 | 0,30 | 3,0 | 1 | — | 20,0 | 0,20 |
| 3 | 0,15 | 4,0 | 1 | — | 20,0 | 0,15 |
| 4 | **0,04** | **5,0** | 1 | **sim** | 20,0 | 0,08 |
| 5 | 0,04 | 5,0 | 1 | sim | 20,0 | 0,08 |
| 6 | 0,04 | 5,0 | 1 | sim | 20,0 | 0,08 |

Quatro leituras da tabela:

- O **máximo** do topo continua **0,55 m** em todos os níveis, e o **mínimo** da carga
  continua **1 kg**. O robô treina a altura e a carga que ele domina.
- O piso de 0,04 m existe porque a laje tem 4 cm: com o topo em 0,04 a laje **apoia** no
  chão em vez de atravessá-lo. Dois corpos estáticos em contato gastam slots de contato.
- **O eixo do `reorientar` SATURA no nível 4.** Acima dele o que gradua é a altura da laje e
  a carga, não a orientação. (O `smoke` afirma `voltas_max[4] == voltas_max[−1]`.)
- **A tabela discreta do `g1_multitask` (0,55 … 0,00) tinha dois defeitos:** no nível 6 a
  laje ficava **enterrada** (centro em −0,02 m), e a altura fácil **desaparecia** do treino
  no instante da promoção.

### 11.3 Relógio B — o controlador de fatia, e a armadilha de 40×

Este é o mecanismo central do módulo. Ele resolve **um** problema:

> **O sorteio é por EPISÓDIO. O PPO aprende por TRANSIÇÃO.**

```python
def resolve_sorteio(alvo, dur_loco, dur_manip, lo, hi):
    """f = alvo·Tm / (Tl·(1−alvo) + alvo·Tm)"""
    tl, tm = max(dur_loco, 1.0), max(dur_manip, 1.0)
    a = clamp(alvo, 0, 1)
    return clamp(a·tm / (tl·(1−a) + a·tm), lo, hi)
```

| `Tl` | `Tm` | um sorteio de 0,30 **entrega** | para entregar 0,30, **sortear** |
|---:|---:|---:|---:|
| 24 | 961 | **1,06 %** | **0,9449** |
| 150 | 500 | 11,4 % | 0,5882 |
| 400 | 500 | 25,5 % | 0,3488 |
| 1000 | 500 | 46,2 % | 0,1765 |

**O `g1_poc` entregou 70% das transições à manipulação achando que entregava 30%.** Um erro
de **40×**. É o defeito que este módulo existe para não repetir.

A função é **pura** de propósito — nenhum tensor, nenhum env. É o que permite testá-la contra
a tabela da spec **sem simulador**, e é o único jeito de saber que a aritmética está certa
antes de gastar GPU. O `smoke.py` seção 20 verifica as 4 linhas, nos dois sentidos.

### 11.4 O laço fechado da `forma`

```
                       ┌──────────────────────────────────────────────┐
                       │  1. as EMAs de DURAÇÃO (α = 0,99)            │
   episódios que ──────┤     dur_loco  ← episódios de ANDAR           │
   ACABARAM            │     dur_manip ← episódios de manipulação     │
                       └──────────────────┬───────────────────────────┘
                                          │
   twist.metrics ───► 2. o SINAL DO PORTÃO │
   ["eficiencia_min"]    razao ← EMA(eficiencia_min)   [nasce em 0,0]
                                          │
                       ┌──────────────────▼───────────────────────────┐
                       │  3. o PORTÃO e a RAMPA                       │
                       │                                              │
                       │  se iters_balanco < 200:  não mexe (CARÊNCIA)│
                       │                                              │
                       │  razao < 0,80 × 0,50  →  alvo += 0,02        │
                       │                          (DEVOLVE fatia)     │
                       │  razao >= 0,50        →  abriu = 1            │
                       │                          alvo −= 0,02        │
                       │                          (1 degrau por janela)│
                       │                                              │
                       │  alvo clampeado em [0,30 ; 0,95]             │
                       └──────────────────┬───────────────────────────┘
                                          │
                       ┌──────────────────▼───────────────────────────┐
                       │  4. sorteio = resolve_sorteio(alvo, Tl, Tm)  │
                       │     clampeado em [0,10 ; 0,95]               │
                       └──────────────────┬───────────────────────────┘
                                          ▼
                            env.limpo_forma["sorteio"]
                                          │
                                          ▼
                            curriculo.sorteia_elo lê daqui
```

`knobs.Forma`:

| campo | valor | papel |
|---|---:|---|
| `fatia_loco` | 0,95 | valor de partida (e o usado com o controlador desligado) |
| `controla` | True | liga o laço fechado |
| `alvo_loco_max` | 0,95 | o piso inicial |
| `alvo_loco_min` | 0,30 | **o destino** |
| `alvo_passo` | 0,02 | 33 degraus de 0,95 a 0,30 |
| `iters_entre_degraus` | 12 | ⇒ ≥ **396 iterações** de rampa |
| `sorteio_min` / `sorteio_max` | 0,10 / 0,95 | clamps do **sorteio**, não do alvo |
| `limiar_portao` | 0,50 | **provisório e não medido** na escala nova |
| `histerese` | 0,80 | devolve fatia se o sinal cai abaixo de `0,80 × limiar` |
| `carencia_iters` | 200 | contada de quando o **balanço** começou |
| `ema` | 0,99 | |
| `dur_inicial_passos` | 1000,0 | |
| `passos_por_iteracao` | 24 | o `num_steps_per_env` do PPO |

**Cinco decisões dentro desse laço:**

1. **`fatia_loco` nunca é 1,00 — e o motivo não é o treino, é o NORMALIZADOR.** Com 1,00 os
   slots de manipulação do one-hot são constantes em zero, e
   `rsl_rl/modules/normalization.py:48` faz `(x − _mean)/(_std + 1e−2)` **sem clamp**: ao
   acender, `1,0` entra na rede como **100,0**. Com 0,95, 5% dos episódios são de manipulação
   desde o passo 0.

2. **O estado inicial é ASSIMÉTRICO, e é decisão medida.**
   - As **durações** nascem **neutras** (episódio cheio). Elas governam a **fatia**, e um erro
     ali só desafina o sorteio por ~τ.
   - O **sinal do portão** nasce **pessimista em 0,0**. Ele governa a **entrega**, e um portão
     que nasce aprovando entrega a locomoção **antes de existir marcha**. Foi exatamente o que
     a `dur_loco_ema` neutra em 1000 passos fez: *"ela dava nota máxima à estátua, porque
     estátua não cai."*

3. **Histerese assimétrica: lento para avançar, rápido para defender.**

4. **Um degrau por JANELA, não por chamada.** Este termo roda **várias vezes por iteração de
   PPO** (uma por passo em que algum env reseta), logo um `% iters_entre_degraus == 0`
   desceria a rampa muitas vezes na mesma iteração. O `ultimo_degrau` garante uma descida por
   janela.

5. **⚠⚠ A iteração é DERIVADA de `env.common_step_counter`, e não incrementada aqui.**
   Defeito medido em 26/08 (achado por code review):

   > O termo de currículo roda em `curriculum_manager.compute`, que o `_reset_idx` chama — e
   > o `_reset_idx` roda a **cada passo em que ALGUM env reseta**. Com os episódios
   > dessincronizados isso é quase todo passo: medido com 128 envs, o contador subiu para
   > **48,8% dos passos** em 400 passos, e com 4096 envs tenderia a 100%.
   >
   > Portanto um `contador += 1` conta **passos**, não iterações. Consequência medida: a
   > carência de "200 iterações" era atingida em **~17**, e a rampa de "396 iterações" em
   > **~34**. A fatia colapsaria de 0,95 para 0,30 em algumas dezenas de iterações —
   > exatamente a falha que este módulo existe para evitar.
   >
   > O conserto: `iters_balanco = (passo − passo_inicial) / passos_por_iteracao`, com
   > `passo = env.common_step_counter`, que o mjlab **já persiste** no checkpoint. A rampa
   > fica resume-safe de graça e **monotônica por construção**.

**Um sinal só no portão.** Dois sinais conjuntivos já travaram uma rampa para sempre: o
`erro_giro_ema <= 0,30` ficou plano em 0,587 por 390 iterações enquanto a `razao_giro` marcava
0,373.

### 11.5 `sorteia_elo` — quem decide loco × manipulação

```python
def sorteia_elo(env, env_ids, *, elo_loco, elos_manip, fatia_loco, forcado):
    buf = garante_elo(env, elo_loco)
    if forcado is not None: buf[env_ids] = forcado; return forcado

    est = getattr(env, "limpo_forma", None)
    if est is not None and "sorteio" in est:
        fatia_loco = est["sorteio"]              # ← O SORTEIO RESOLVIDO, não o alvo

    sorteio = rand(n)
    k = randint(len(elos_manip), (n,))           # uniforme entre os sorteáveis
    buf[env_ids] = where(sorteio < fatia_loco, elo_loco, tabela[k])
    return float((buf == elo_loco).float().mean())
```

> ⚠ **O que chega aqui é o SORTEIO JÁ RESOLVIDO, não a fatia alvo.** Os dois são coisas
> diferentes: `alvo` é fatia de **transições**, `sorteio` é probabilidade por **episódio**.
> Usar o alvo direto aqui **é a armadilha de 40× deste projeto**.

Ele roda no **currículo** e não num evento porque o elo tem de existir **antes dos eventos**
(o reset de pose da base depende dele) e **antes do comando** (o alvo depende dele).

---

<a id="12"></a>
## 12. A ordem do treinamento: as fases F0 → F6

O pacote foi construído em fases, e os comentários do código as nomeiam. **Todas estão
implementadas hoje** — o que muda de fase para fase é o que o portão do bloco olha.

| fase | o que entrou | portão de aceite |
|---|---|---|
| **F0** | esqueleto, cena, ação, física, remoções, contrato de não-import | `smoke.py` verde; `inspeciona --tabela` sem falha |
| **F1** | a tabela de recompensa da locomoção, as métricas de marcha, as duas réguas | **locomoção pura**: mobília a +5 m, `valida = 0`, nenhum alvo de caixa. A escada de corte (§14.3) |
| **F2** | o one-hot de 5 elos, o sorteio de elo por env, `PosturaPorElo` | a fatia medida bate com o knob; slots 3 e 4 constantes em zero |
| **F3** | os 7 incentivos, os canais de caixa na obs, os σ por env | `Contrib/squeeze` sai de zero **antes** de `precise_pos` |
| **F4** | a máquina de elo: as 4 cadeias, `_avanca_elo`, `prob_por_nivel` | a cadeia **não destrói** a fatia de locomoção (±0,06) — *"o invariante mais importante da F4"* |
| **F5** | o controlador de fatia (`forma`), o piso de nível, o estado no checkpoint | robô parado **não** abre o portão em 40 degraus de folga — *"o check que mais importa da F5"* |
| **F6** | o passeio aleatório de nível | um episódio de **locomoção** não move o nível |

### 12.1 A hipótese central, e ela foi validada por medição

**F1 = LOCOMOÇÃO PURA.**

> O `g1_multitask` **andou** porque a fase 1 dele não tinha manipulação nenhuma (fatia de
> 100% para a locomoção). O `g1_poc` **não andou** porque entregou **70% das transições** à
> manipulação por volta da iteração 420, com o robô imóvel.

Por isso `ELO_DE_TREINO = CMD.ANDAR` e `alvo_loco_max = 0,95`: o treino **abre** em
locomoção, e só entrega manipulação quando o portão de marcha aprova.

### 12.2 As duas regras de disciplina

O repositório já pagou por elas:

1. **Uma mudança por bloco.**
2. **Warm-start sempre com `learning_rate = 5e-4`.**

### 12.3 Os números pré-registrados

*Pré-registrado* = decidido **antes** de ver o resultado, para não virar pesca de hipótese.

| se falhar | mover | para | e nada mais |
|---|---|---|---|
| o portão da F1 (o robô não anda) | `cena.escala_acao_mult` | 0,8 | ✔ |
| o alcance não aparece na F3 | `tarefa.sigma_fator` | 1,5 | ✔ — **nunca o peso** |
| aparece um hack de recompensa | volte **um** termo | — | não seis |

O `sigma_fator` e nunca o peso porque *"tornar o 1º centímetro positivo exigiria peso > 12,
quatro vezes o da locomoção, e o robô pararia de andar."*

### 12.4 O PPO

`mjlab/tasks/velocity/config/g1/rl_cfg.py::unitree_g1_ppo_runner_cfg()`, **sem mudança**,
com `experiment_name = "g1_limpo"`.

| campo | valor |
|---|---|
| `num_steps_per_env` | **24** (é o `passos_por_iteracao` do knob) |
| `gamma` / `lam` | 0,99 / 0,95 |
| `learning_rate` | 1,0e−3, `schedule = "adaptive"`, `desired_kl = 0,01` |
| `clip_param` | 0,2 |
| `entropy_coef` | 0,01 |
| épocas / minibatches | 5 / 4 |
| ator e crítico | MLP (512, 256, 128), ELU |
| distribuição | Gaussiana, `init_std = 1,0`, `std_type = "scalar"` |
| normalização de obs | `EmpiricalNormalization`, nos dois grupos |

---

<a id="13"></a>
## 13. As limitações declaradas

Estas estão escritas no próprio código, e vale conhecê-las antes de diagnosticar um painel.

| # | limitação | onde está declarada |
|---|---|---|
| 1 | **A caixa de 5 kg tem a inércia de 1 kg.** A carga é força externa, porque `dr.body_mass` corrompe a heap. A DR endurece a **estática**, não a dinâmica | `eventos.carga_caixa` |
| 2 | **`topo_min = 0,04` põe a laje como degrau de 4 cm na frente dos pés, e PISAR nela passa dos 50 N.** Nesses níveis a mesa precisaria deixar de existir, e **isso não está implementado**. Risco **adiado**: os blocos 1–3 nunca saíram do nível 0 | `cena.CORPOS_QUE_NAO_ESCORAM` |
| 3 | **Os slots 3 e 4 do one-hot são constantes em zero** até uma cadeia de 2 elos fechar o 1º elo. O normalizador do rsl_rl fará o primeiro `1,0` entrar como **100,0** | `curriculo.py`, topo |
| 4 | **`limiar_portao = 0,50` é provisório e não medido** na escala nova. O 0,50 vinha da `razao_marcha` e coincide de escala, mas ninguém mediu quanto a política que o `play` mostrou andando bem marca em `eficiencia_min` | `knobs.Forma.limiar_portao` |
| 5 | **O `unload` é quase booleano como rampa de altura** (salta de 0 a 1 em 2 mm), e a medição que o avaliou **teleportou** a caixa, portanto não modela partilha de carga | `recompensas.unload` |
| 6 | **`cone = pyramidal` modela pior o cone de atrito de uma pega.** `elliptic` + `impratio = 10` é o par que a tarefa de manipulação do mjlab usa, e que **divergiu para NaN** aqui em 15/07 | `knobs.Cena` |
| 7 | **`base_com` (`dr.body_com_offset`) está removido.** Preço declarado: perder ±2,5 cm de randomização de CoM no torso | `env_cfg`, §3 |
| 8 | **A base reseta EM REPOUSO**, nos dois modos. Não existe "entrega do navegador" com velocidade residual | `knobs.Cena`, comentário do knob removido |
| 9 | **A pose da caixa é verdade absoluta do simulador**, com ruído de ±0,01 m. Falta latência, viés e perda de rastreio | fora do escopo, spec §19 |
| 10 | **`so_pose` não é lido pelo corpo de `_aplica_elo`** — a função refaz **tudo**. Hoje isso é correto e desejado, mas quem acrescentar ali um sorteio que deva sobreviver à passada do `_pendente` **tem** de passar a lê-lo | `comando._aplica_elo` |

---

<a id="14"></a>
## 14. As ferramentas: smoke, inspeciona, leitura, paridade

### 14.1 `smoke.py` — o portão de cada fase

```bash
python -m g1_limpo.smoke
```

*"É o portão de cada fase do plano: nenhuma fase começa com o smoke vermelho."*

Não é um framework de testes: é um **script linear** de ~530 chamadas a
`check(nome, cond, detalhe)`, agrupadas em **21 seções** por `secao(titulo)`. Imprime
`N ok / M falhas` e sai com 1 se houver qualquer falha. **Não dá para rodar um subconjunto.**

As 21 seções, por tema:

| tema | seções | o que afirmam |
|---|---|---|
| cena e física | 1–7 | 3 entidades; caixa `nq = 7`, laje `nq = 0`; mobília fora do grupo 0; a laje apoia no chão no piso do currículo; 8 sensores; `njmax/nconmax/impratio/cone`; 16 padrões em `G1_ACTION_SCALE` |
| ramo play | 8 | `randomize_terrain`, `commands_vel` e `push_robot` fora |
| σ da postura | 9 | os três dicts são **os mesmos objetos** do fabricante; e a palavra `knee` não aparece em nenhum fonte |
| contrato de não-import | 10 | nenhum `.py` importa `g1_training`/`g1_poc`/`g1_multitask` |
| recompensa | 11, 14, 18 | `set(rewards) − set(fabricante)` == os **9** nomes; `terminacao = −200`, `joint_acc = −2,5e−7`; a subclasse `AlturaDeBalanco` **tem** `reset` e a do fabricante **não**; os 7 incentivos somam **11,5/s**; o σ é a distância inicial e o kernel vale **0,368** na abertura |
| comando e elo | 12, 13, 16 | layout `(0:3, 3:6, 6, 7, 8, 9)`; 5 elos; `ALCANCE_R = 0,85`; `elos_parados = (1,2,4)`; o desenho produz **2 frames, 4 spheres, 8 arrows, 2 boxes** com 2 envs |
| piso da estátua | 17 | a estátua num elo parado colhe **3,4 a 4,2/s** dos dois `track_*`; o piso total do parado é **< 12,5/s** |
| máquina de elo | 19 | 4 cadeias, teto de 2; `PEGAR` em todas; `prob_por_nivel` é 7×4 e cada linha soma 1,0; a cadeia **não destrói** a fatia (±0,06); o avanço **não reseta** o episódio |
| balanço e checkpoint | 20 | a tabela de `resolve_sorteio` (4 linhas × 2 sentidos); **33 degraus e ≥ 396 iterações**; a ordem `command_vel → forma → nivel → elo`; ciclo real de save/load com valores reconhecíveis |
| passeio de nível | 21 | 100% → topo; 0% → piso; 50% → **difunde**; locomoção **não move** o nível |
| leitura | 15 | roda `leitura._demo()` e exige 0 |

**O check que o autor marca como o mais importante:** *"com `caixa_valida = 0`, os quatro
canais da caixa são zero E os termos de tarefa são zero."*

> ⚠ A docstring do `smoke.py` diz `FASES COBERTAS: F0`. **Está desatualizada** — o corpo
> cobre F0 a F6.

> ⚠ Um teste fixa um número que o próprio autor declara provisório: o `limiar_portao = 0,50`.

### 14.2 `inspeciona.py` — a revisão antes de gastar GPU

*"A GPU é alugada, e um erro de geometria descoberto na iteração 800 de um Kaggle custa a
sessão."* A lição de 16/07 é **"ataque a geometria, e não o sintoma"**.

```bash
python -m g1_limpo.inspeciona --tabela            # os 5 elos, nível 0   ← o DEFAULT
python -m g1_limpo.inspeciona --tabela pegar      # o pegar nos 7 níveis
python -m g1_limpo.inspeciona --tabela --cadeia 2 # + a seção PÓS-AVANÇO (7 níveis × 3 cadeias)
python -m g1_limpo.inspeciona --viewer pegar
python -m g1_limpo.inspeciona --viewer --cadeia 2 --avanca-elo
```

Flags: `--tabela`, `--viewer`, `--cadeia N|NOME`, `--avanca-elo`, `--nivel N`, `--envs`
(1), `--envs-tabela` (8), `--device` (cpu).

**Ordem recomendada:** 1) `--tabela` sem argumento (o portão barato); 2) `--tabela <elo>` no
elo suspeito; 3) `--tabela --cadeia N`; 4) `--viewer <elo>`; 5) `--viewer --cadeia N
--avanca-elo`.

Colunas da tabela (cada célula é `min–max` sobre os envs):
`elo | niv | valida | topo laje | caixa z | alvo z | caixa->alvo | pelve->alvo | voltas |
azimute | erro graus`.

Ele instancia o **mesmo** `make_env_cfg` do treino — nada de mock. As checagens de sanidade
mais instrutivas:

- **O robô está travado:** `std(robo_z) <= 1e−3`. O limiar é **derivado**: `½·g·dt² = 0,5 ×
  9,81 × 0,02² = 2,0 mm` de queda por construção. *"Um limiar de 1e−4 acusava a própria
  gravidade."*
- **O erro angular tem teto de TRÊS parcelas:** `voltas × 90° + desalinho_max_deg[niv] +
  azimute`, com folga de 1°. O azimute é **geometria, não knob**: a caixa nasce em y até
  ±0,18 com x ≈ 0,32, logo `atan(0,18/0,32) = 29°`. Sem ele o check acusava o nível 0 com 25°
  contra teto de 15°.
- **`|lateral| <= 5e−3`:** o alvo tem de estar à frente, no eixo do robô. Defeito visto no
  viewer em 25/08.
- **`BOTAR`: a laje não nasce dentro da caixa** — comparação **elemento a elemento**, com a
  tolerância sendo a **queda medida** da caixa naquele passo, não um número chutado.

> ⚠ **Cicatriz:** o `except` do pós-avanço foi **estreitado** para `(AttributeError,
> KeyError)`. Era `except Exception` e **engoliu um `NameError` do próprio código**
> (`_z_antes` não definido), fazendo toda a geometria do `BOTAR` "passar por omissão" nos 7
> níveis.

> ⚠ **Duplicação real:** `_resolve_cadeia()` é usada só pelo `viewer`; a função `tabela`
> reimplementa a mesma lógica inline, com `cadeias_ids = [0,1,2,3]` **hardcoded** em vez de
> derivada de `CMD.CADEIAS` — o que contradiz o comentário do próprio arquivo sobre listas
> paralelas saindo de sincronia.

### 14.3 `leitura.py` — a diluição do painel, desfeita

```bash
python g1_limpo/leitura.py                 # o run mais recente sob ./logs
python g1_limpo/leitura.py CAMINHO/run
python g1_limpo/leitura.py --demo          # autoteste da aritmética, sem log
```

> ⚠ **Rode como arquivo, não com `-m`.** `-m g1_limpo.leitura` importa o `__init__.py`, que
> registra a task e exige o `mjlab`.

**A fórmula, e o motivo de o arquivo existir.** O `Episode_Reward/<termo>` do rsl_rl é a soma
do episódio dividida por `max_episode_length_s` — e **não** pela duração real:

```
taxa por segundo = Episode_Reward × PASSOS_CHEIOS / passos_medios

DT = 0,02      MAX_EP_S = 20,0      PASSOS_CHEIOS = 1000
```

**O número da cicatriz:** com episódios de 2,05 s (102,5 passos) num teto de 20 s, todo valor
sai multiplicado por **0,1027**; o fator de correção é **9,756**. Um `action_rate_l2 = −0,34`
no painel é um custo real de **−3,31/s**. Ler o painel sem desfazer isso *"deixou dois freios
consumirem 55% do sinal positivo por 5000 iterações sem ninguém ver."*

> ⚠ O `Episode_Metrics/*` **não** tem essa diluição. Só a recompensa precisa da correção.
> E `passos_medios` vem em **passos** (`Train/mean_episode_length`), não em segundos —
> confundir os dois erra por **50×**.

Os 4 painéis: **(a)** recompensa des-diluída, por termo, ordenada por taxa, com `SOMA
positiva / SOMA negativa / LÍQUIDO` e o aviso *"os freios consomem X% do sinal positivo"*;
**(b)** marcha, 17 linhas (as 4 terminações de contato, palmas/dorso em contato, as três
réguas, tempo de voo, pico, escorrego, força de pouso, duração, nível); **(c)** cadeia
(`sucesso`, `passo_final`, `avancos`, `fatia_cadeia`, `Curriculum/elo`); **(d)** a escada.

**A ESCADA de corte:**

| it | chave | comp | alvo | falhar significa |
|---:|---|:---:|---:|---|
| 200 | `Policy/mean_std` | ≥ | **0,60** | as penalidades dominam desde o começo |
| 1000 | `Train/mean_episode_length` | ≥ | **150** passos | o robô não sobrevive ao episódio |
| 2000 | `Metrics/twist/eficiencia_min` | ≥ | **0,50** | **o robô não anda** (alvo não calibrado) |
| 1000 | `Episode_Metrics/tempo_de_voo` | > | **0,0** | nenhum pé sai do chão: não há marcha, há arrasto |
| fim | `Metrics/alvo_caixa/fatia_cadeia` | ≥ | **0,10** | as cadeias de 2 elos não estão sendo sorteadas |
| fim | `Metrics/alvo_caixa/sucesso` | > | **0,0** | nenhuma cadeia de 2 elos fechou |

Três robustezes:

- **Chave ausente é FALHA** (`portão CEGO`), e é tratada **antes** do `it`. A cicatriz: com
  `it=None` e chave ausente, `ultimo = −1` fazia o comparador dar `TypeError`, **matando o
  leitor inteiro** em vez de reportar dois portões cegos.
- **`it = None` = fim de run**, porque o contador do rsl_rl **acumula entre blocos**
  (`total_it = start_it + num_learning_iterations`) — "na iteração 5000" para a F4 dispararia
  no instante em que o bloco começa.
- **`_em(serie, it)` não olha para o futuro:** devolve o último valor com `passo <= it`.

> ⚠ **Cicatriz de chave:** a escada do `g1_poc` usa `Policy/mean_noise_std`, **que não existe**
> no rsl_rl 5.4.0 — o logger escreve `Policy/mean_std`. *"Aquela linha da escada nunca
> disparou."*

O `--demo` é um autoteste em três blocos, com uma confissão útil: *"uma versão anterior
'testava' `0,10 >= 0,05`, que exercita o operador do Python e não pode falhar, logo não era
teste."*

### 14.4 `paridade.py` — o descartável que prova a transcrição

```bash
python -m g1_limpo.paridade
```

**É o único arquivo do pacote que importa código do projeto**, e é declaradamente
descartável: não roda em treino, não é importado por ninguém.

**Por que compara `mjModel` e não `cfg`:** massa, inércia, atrito, meia-aresta, altura da
laje, grupo de geom e posição dos pads vivem dentro de **lambdas** de `spec_fn`, e
**comparação de cfg não penetra lambda**. Só o modelo compilado vê esses números.

5 seções: **(1)** a caixa, em três partes — física **bit-idêntica**, o geom de colisão
idêntico, e o marcador provadamente inerte (`contype = 0`, `conaffinity = 0`); **(2)** a laje,
11 campos; **(3)** o robô com os pads, 11 campos + os **nomes na mesma ordem** + as cápsulas
removidas nos dois; **(4)** 5 constantes de nome; **(5)** o contrato dos 5 campos de cada
sensor contra o `g1_poc`, com a divergência dos 2 sensores extras **nomeada**.

**Limite estrutural declarado:** paridade contra o fabricante **não pega** erro na metade de
manipulação, porque o fabricante não tem caixa nem mesa.

---

<a id="15"></a>
## 15. Divergências entre o código e a especificação do projeto

A `especificacao-g1_poc.md` no projeto descreve o **`g1_poc`**. O `g1_limpo` é uma
**reescrita posterior**, e reverteu várias decisões dela por medição. Se você ler a spec como
se descrevesse o código de hoje, vai se enganar em 14 pontos.

| # | a especificação diz | o código faz | por quê |
|---|---|---|---|
| 1 | obs: bit **`caixa_valida`** (1 canal), ator com 112 canais | **one-hot de 5 elos** (5 canais) + **8 canais de caixa no frame da base** | o bit não distingue *qual* objetivo está ativo; a troca de elo dentro do episódio pede o one-hot lido do comando |
| 2 | **18 termos** de recompensa (13 + 5) | molde + **9** novos: `terminacao`, `joint_acc` + os **7** de tarefa | `unload`, `postura_ereta` e `sustentacao` entraram (a spec os tinha "em reserva" ou não os tinha) |
| 3 | **`terminacao (−200)` sai**; "a falha passa a ser fim de episódio" | `terminacao = −200,0` **existe** | é `is_terminated`, que **exclui** o `time_out`: ele é a forma de cobrar as terminações sem somar penalidade em cima |
| 4 | um **4º regime de σ** (`std_manipulando`, 12 valores) | `PosturaPorElo` retorna **1,0** (neutro) | medido: nenhum σ resolve — o termo é produto de 17 gaussianas e **colapsa** (§9.2) |
| 5 | alvo do `pegar`: **z absoluto 0,78–0,85** de mundo | `peito_b` relativo em x,y + **z = 0,95 absoluto**, idêntico ao `carregar` | com a laje variando de 0,55 a 0,04, o alvo fixo fazia "erguer" valer 5× coisas diferentes (§8.3) |
| 6 | `carregar` fecha com **6 s** de tempo | **1,5 s E deslocamento ≥ 0,50 m** | `perto` é subconjunto da condição do `pegar` sobre o mesmo alvo: a cadeia treinava **"não andar"** (§8.10) |
| 7 | tolerâncias **0,05 m / 20°**; `sustenta_pegar = 1,0 s` | **0,10 m / 25°**; `sustenta_pegar = 0,5 s` | — |
| 8 | `reorientar` em **graus** (0–180°) | **quartos de volta**, teto de **1** (±90°) | a primitiva atômica é o quarto de volta; compor voltas sai de graça |
| 9 | fatia **30% loco / 70% manip**, fixa | **laço fechado** 0,95 → 0,30, gateado por `eficiencia_min`, com conversão transição↔episódio | a armadilha de 40× (§11.3) |
| 10 | `escala_acao = G1_ACTION_SCALE × 0,8` | **× 1,0** | *"o 0,8 cortava 20% da autoridade de junta, e autoridade é o que uma fase de balanço precisa"*. Pré-registrado para voltar a 0,8 se o portão falhar |
| 11 | `dof_pos_limits = −10,0` | **−1,0** | — |
| 12 | `contato_ilegal` exclui antebraço, mão, pad e pé | **1 sensor → 3**; os **pads entram**, punho e cotovelo **saem** | a lista de corpo inteiro rodou e falhou: 75% dos episódios de manipulação morriam na mesa (§4.7) |
| 13 | `F_ref = m·g/(2μ)` com **μ = 1,0** (≈6 N) | **μ = 0,8** (6,13 N) | μ pessimista da faixa de atrito |
| 14 | §11.1: reset com **velocidade residual** e erro de rumo (a "entrega do navegador") | a base reseta **em repouso** | o knob existia sem consumidor e foi **removido** em 26/08 |

**O que a spec e o código concordam** (e são as decisões estruturais):

- o alvo é um **comando**, e o robô o observa;
- **terminar em vez de penalizar**;
- o episódio **não termina no sucesso** — `fechou` trava e o episódio segue até o `time_out`;
- as **4 cadeias**, teto de 2 elos, com o `PEGAR` em todas;
- o **passeio ±1** de nível, com ponto fixo em `p = 0,5`;
- a carga como **força externa**, nunca `dr.body_mass`;
- a mobília no **grupo de geom 2**;
- `cone = pyramidal`, `impratio = 1,0`;
- o PPO do fabricante sem mudança.

### Duas inconsistências internas no código, para corrigir quando der

1. **`curriculo.forma`, docstring:** diz `command_vel → elo → nivel → forma`. A ordem real —
   a que `env_cfg` insere e o `smoke` seção 20 afirma — é `command_vel → forma → nivel → elo`.
2. **`smoke.py`, docstring:** diz `FASES COBERTAS: F0`. O corpo cobre F0 a F6.

Nenhuma das duas muda comportamento. As duas enganam quem lê.
