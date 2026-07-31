# Células do Kaggle — experimento residual BFM + PPO

Notebook novo, GPU T4. **Ligue a internet** nas configurações do notebook.

---

## Antes de abrir o Kaggle: duas coisas no seu PC

**1. Suba o repo.** A branch é `exp/residual-bfm`.

```bash
cd ~/Documents/g1_training
git push -u origin exp/residual-bfm
```

**2. Suba o peso como Kaggle Dataset.** O arquivo tem 122 MB e está fora do git de
propósito (o GitHub recusa acima de 100 MB).

- Arquivo: `~/Documents/g1_training/g1_residual/peso/bfm_ator.pt`
- Em <https://www.kaggle.com/datasets> → **New Dataset** → suba o arquivo só.
- Nome sugerido: `bfm-ator-g1`. Ele vai aparecer em
  `/kaggle/input/bfm-ator-g1/bfm_ator.pt`.
- No notebook: **Add Input** → o dataset.

Se o nome ficar diferente, ajuste a célula 3.

---

## Célula 0 — a GPU

Só `nvidia-smi`. **Não importe torch aqui.** Importar torch antes do `pip` deixa a
extensão já registrada, e um `reload` depois levanta
`Only a single TORCH_LIBRARY can be used to register the namespace triton`.

```python
!nvidia-smi
```

---

## Célula 1 — o repo

```python
!rm -rf /kaggle/working/g1
!git clone -q --branch exp/residual-bfm \
    https://github.com/JoaoBornelli/g1_training.git /kaggle/working/g1
!cd /kaggle/working/g1 && git log --oneline -3
```

---

## Célula 2 — dependências

`torch` fica **sem pin** de propósito: a imagem do Kaggle já traz uma versão casada
com o CUDA da máquina, e forçar outra quebra o `warp`.

```python
!pip install -q mjlab rsl-rl-lib gymnasium pydantic safetensors
!pip list 2>/dev/null | grep -Ei "^(mjlab|rsl-rl|warp-lang|torch|gymnasium|pydantic|safetensors) "
```

---

## Célula 3 — o peso do BFM

```python
import pathlib, shutil
destino = pathlib.Path('/kaggle/working/g1/g1_residual/peso')
destino.mkdir(parents=True, exist_ok=True)
origem = pathlib.Path('/kaggle/input/bfm-ator-g1/bfm_ator.pt')
assert origem.is_file(), f'não achei {origem} — confira o Add Input e o nome do dataset'
shutil.copy(origem, destino / 'bfm_ator.pt')
print(f"{(destino / 'bfm_ator.pt').stat().st_size / 2**20:.1f} MB no lugar")
```

---

## Célula 4 — o ambiente está de pé?

Subprocesso, para não importar torch no kernel do notebook antes da hora.

```python
import subprocess
print(subprocess.run(
    ['python', '-c', 'import torch;'
     'print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),'
     '"|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")'],
    cwd='/kaggle/working/g1', capture_output=True, text=True).stdout)
```

---

## Célula 5 — TESTE DE FUMAÇA: o BFM anda no nosso env?

**Rode isto antes do treino.** Residual em zero, nenhum treino, ~2 min. Ele responde
se o BFM fica de pé e se ele aguenta a caixa.

```python
!cd /kaggle/working/g1 && python g1_residual/fumaca.py 256 300
```

O que esperar, medido em CPU com 16 envs:

| `z` | de pé no fim | passos de pé |
|---|---|---|
| `move-ego-0-0` | 100% | 150 de 150 |
| `raisearms-m-m` | 100% | 150 de 150 |
| `crouch-0` | 0% | 10 |
| `crouch-0.25` | 0% | 9 |

**Se o `move-ego-0-0` não der ~100%, pare aqui.** Algo do plant está errado e o
treino só vai gastar cota. Suspeitos, nessa ordem: o peso não é o certo, a DR está
diferente, ou a cena empurra o robô.

Os agachamentos darem 0% é **esperado** — é o achado que trocou o prior do `pegar`.

---

## Célula 6 — o treino

Curto de propósito. 3000 iterações, e olhe o `steps per second` no primeiro bloco de
saída antes de decidir subir o `num_envs`.

```python
!cd /kaggle/working/g1 && python g1_residual/train.py \
    --env.scene.num-envs 2048 \
    --agent.max-iterations 3000
```

O que vigiar, na ordem:

1. **`[RESIDUAL] ação 49 = 29 residual + 20 comportamento`** na primeira linha. Se
   disser 29, o termo de ação velho foi construído e o BFM não está no laço.
2. **`Mean episode length`** — tem que começar ALTO, em centenas. O BFM sozinho fica
   150 de 150 passos de pé no teste de fumaça, e o sucesso do `parado` é sobreviver
   20 s. Se começar perto de 30-60 passos, o RESIDUAL está derrubando o BFM: olhe
   `arm_vel` (a run que falhou dava −5,49 contra −0,24 da monolítica) e baixe a
   `escala_delta`.
3. **`retorno ÷ episódio`** — se ficar negativo, o episódio encurta, porque
   terminação zera o valor futuro. É transitório e se resolve sozinho quando cruza o
   zero; **não mexa em peso por causa disso**.
4. **`Episode_Metrics/sucesso`** — sai do zero?
5. **`Episode_Metrics/taxa_alvo`** e **`Episode_Reward/joint_acc`** — se os dois
   subirem juntos e não descerem, o residual e o reflexo do BFM estão oscilando. O
   `action_rate_l2` é **cego** para isso, porque lê só a saída da política.

---

## Célula 7 — o que a rede escolheu de comportamento

```python
!cd /kaggle/working/g1 && ls -t logs/g1_residual/ | head -3
```

No TensorBoard procure `z_graus/2` (quantos graus o `pegar` andou desde o prior) e as
chaves `z_perto/2/<nome>`. Elas dizem para qual dos 41 comportamentos a política foi.

**É esta a resposta para "agachar é a melhor maneira de pegar a caixa?".** Se ela
andou para `crouch-*`, agachar era certo e ela aprendeu a estabilizar a descida. Se
ficou perto de `move-ego-0-0`, ela achou outro jeito.

---

## Célula 8 — baixar o checkpoint para rodar o `play` no seu PC

O `play` precisa de janela, então ele roda no seu PC, não aqui.

```python
import pathlib, shutil
raiz = pathlib.Path('/kaggle/working/g1/logs/g1_residual')
run = sorted(raiz.iterdir(), key=lambda p: p.stat().st_mtime)[-1]
ckpts = sorted(run.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
print('run:', run.name)
for c in ckpts[-3:]:
    print(f'  {c.name}  {c.stat().st_size / 2**20:.1f} MB')
shutil.copy(ckpts[-1], f'/kaggle/working/{ckpts[-1].name}')
print(f'\nbaixe /kaggle/working/{ckpts[-1].name} pelo painel de Output')
```

No seu PC:

```bash
cd ~/Documents/g1_training
python play.py --task Mjlab-Residual-Unitree-G1 \
               --checkpoint ~/Downloads/model_XXXX.pt
```

---

## Se precisar mexer em algo entre tentativas

Os três knobs, em ordem do que eu mexeria primeiro:

| knob | onde | quando |
|---|---|---|
| `escala_delta` 0,15 → 0,25 | `acao.py::ResidualBFMActionCfg` | o `pegar` não sai do lugar; nenhum agachamento do BFM sobrevive sozinho, então o residual tem trabalho real ali |
| `prior_unico=True` | `env_residual.py::build_env_residual` | tirar meu palpite de prior do caminho |
| `escala_c` 0,3 → 0,6 | `base_z.py::ESCALA_C` | `z_graus` fica preso perto de zero |

Todos são mudança de config. Mas o espaço de ação **não** pode mudar — isso é
Categoria C e obriga a recomeçar do zero.
