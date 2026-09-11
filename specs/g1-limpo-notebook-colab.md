# g1_limpo — o notebook roda na Kaggle E no Colab, com 8192 envs onde cabem

**Estado:** a implementar · **Ramo:** `exp/g1-limpo-v2` · **Base:** `4219d1e` (o lote da
tabela já entrou)

## 0. O pedido

O dono: "esse próximo resume eu quero tentar rodar no colab com os 8192 envs possíveis lá".

## 1. Decisão: UM notebook, que detecta o hospedeiro

Não um segundo notebook. Dois arquivos com os mesmos asserts de paridade derivariam em
silêncio — a última coisa que o lote da tabela faz é reescrever esses asserts, e eles
teriam de ser mantidos em dois lugares.

O Colab abre notebook direto do GitHub:
`https://colab.research.google.com/github/JoaoBornelli/g1_training/blob/exp/g1-limpo-v2/g1_limpo/kaggle/g1_limpo_kaggle.ipynb`.
Com um arquivo só, o dono abre esse link, monta o Drive e roda.

O arquivo continua em `g1_limpo/kaggle/` com o mesmo nome — o caminho é contrato do
smoke e do link acima. Só o `# g1_limpo v2 — ...` do topo passa a dizer "Kaggle ou Colab".

## 2. A detecção (célula 7, no topo, antes de qualquer caminho)

```python
COLAB = pathlib.Path("/content").is_dir() and not pathlib.Path("/kaggle").is_dir()
```

Nada de `import google.colab` para detectar — o import falha fora do Colab com uma
mensagem que confunde.

## 3. O que muda por hospedeiro

| | Kaggle (hoje) | Colab |
|---|---|---|
| `RAIZ` (clone) | `/kaggle/working/g1` | `/content/g1` |
| `LOG_ROOT` | `/kaggle/working/logs` | `/content/logs` — **local, não no Drive** (ver §5) |
| `SAIDA` (zip) | `/kaggle/working` | `/content` |
| checkpoint de ENTRADA | `rglob("model_*.pt")` em `/kaggle/input` | `rglob("model_*.pt")` em `DRIVE = /content/drive/MyDrive/g1_limpo` |
| checkpoint de SAÍDA | célula 13: `kaggle datasets version` | copiar `model_*.pt` + `.pesos.json` + zip para `DRIVE` |
| segredos | `kaggle_secrets` | nenhum — o Drive é do próprio usuário |
| `NUM_ENVS` | 4096 | **8192** (§4) |

A montagem do Drive é uma linha, só no ramo Colab, na célula 7 antes de procurar o
checkpoint:

```python
from google.colab import drive; drive.mount("/content/drive")
DRIVE = pathlib.Path("/content/drive/MyDrive/g1_limpo"); DRIVE.mkdir(exist_ok=True)
```

O `assert` de "nenhum dataset no input" ganha a versão Colab: "nenhum `model_*.pt` em
`MyDrive/g1_limpo` — suba o checkpoint da última sessão para essa pasta no Drive".

A busca pelo MAIOR número, a `IT_MINIMA`, a semeadura `1900-...` e a cópia da impressão
digital são **idênticas** nos dois — só a pasta de origem muda. Escreva UMA função
`acha_checkpoint(pasta)` e chame com a pasta certa. Hoje essa lógica está inline.

## 4. `NUM_ENVS` pelo hospedeiro

```python
NUM_ENVS = 8192 if COLAB else 4096
```

**Não é pela VRAM.** Dono, 2026-09-11: a GPU do Colab tem 16 GB e RODA 8192 envs; a da
Kaggle, também de 16 GB, não. É fato medido pelo dono nas duas plataformas, e a VRAM
nominal não o explica — portanto o discriminador é o hospedeiro, e não um número de
`nvidia-smi`. Não invente um limiar de memória.

Isso também fecha a dúvida da célula 11 ("NÃO SEI QUAL GPU RODOU A bloco8"): a bloco8
rodou no Colab com 8192, e a GPU do Colab é de 16 GB. Reescreva esse comentário.

`ENVS_ANTES = 4096` FICA: é o que o checkpoint traz, e o aviso de truncagem/expansão de
`limpo_nivel`/`limpo_elo` já existe. Ao subir de 4096 para 8192 metade dos envs nasce no
default (nível 0) — o aviso já diz isso.

`SEG_POR_ITER` acompanha: 7,0 com 4096 (Kaggle, medido) e **9,0 com 8192** (a bloco8
mediu 8,54 s/iter no Colab com 8192; 9,0 é a margem). Em ambos, "corrija com o
`Collection time` real do primeiro log" continua valendo.

`HORAS_LIMITE` FICA em 10,5 nos dois. O Colab gratuito cai por ociosidade antes disso; o
Pro vai a 24 h. O dono ajusta na célula — declare isto no comentário.

O `lote do PPO` impresso no fim da célula 11 já mostra `NUM_ENVS × num_steps_per_env`;
com 8192 volta a 196 608 transições, minilote 49 152 — o lote da bloco8.

## 5. Persistência no Colab: o Drive recebe cópias, não o log

**`LOG_ROOT` NÃO vai para o Drive.** O `tfevents` é anexado a cada iteração, e o FUSE do
Drive reescreve o arquivo inteiro a cada flush — lento e já viu corromper. E 30
checkpoints de 6 MB indo para o Drive a cada 50 iterações seriam 30 uploads.

O que vai para o Drive:

1. **No fim** (`finally` do `launch_training`): o `empacota()` de hoje, mais uma cópia do
   zip e do `model_*.pt` final para `DRIVE`. Isto substitui a célula 13 no Colab.
2. **Durante**, contra a queda da sessão: uma thread que, a cada 10 min, copia para
   `DRIVE/em_curso/` o `model_*.pt` de maior número da pasta de run nova, se ele mudou.
   A Kaggle caiu por internet e levou `/kaggle/working` inteiro — 1500 iterações. Com a
   cópia periódica, a perda máxima passa a ser 10 min.
   - `threading.Thread(daemon=True)`, laço `while True: sleep(600); copia()`. Daemon:
     morre com o kernel, não segura o `finally`.
   - Só o ÚLTIMO checkpoint (`shutil.copy2`, comparando `st_mtime`), nunca a pasta.
   - Só no ramo Colab. Na Kaggle o `/kaggle/working` some igual, mas lá o dono
     escolheu a API do dataset e ela fica como está.
3. A célula 13 vira: ramo Kaggle = o que já é; ramo Colab = a cópia final para `DRIVE`
   (redundante com o `finally`, e é de propósito — é a célula que o dono roda à mão se
   o `finally` não rodou).

## 6. `RUN`, `META` e a impressão digital

- `RUN = "bloco16"`. NOME NOVO, e o motivo é o lote da tabela: a recompensa mudou de
  forma. ⚠ **A impressão digital NÃO pega essa mudança** — ela compara `weight`, e a
  tabela multiplica DENTRO do termo, com os `weight` intactos. O nome novo é o que impede
  a comparação com o `.pesos.json` da bloco15. Diga isto no comentário do `RUN`.
- `META`: **o dono decide na hora de subir.** Deixe 12500 e um comentário: com 8192 envs
  cada iteração tem 2× a experiência, portanto a mesma META vale o dobro de amostras.
- A LR já está em 5e-4 com o comentário de warm-start. **Fica** — este resume É
  warm-start (tabela nova, crítico refaz a escala).

## 7. Texto

- Célula 0 (título) e célula 14 (próxima sessão): dizem Kaggle ou Colab, e os passos
  do Colab: abrir pelo link do GitHub, montar o Drive, ter o `model_*.pt` em
  `MyDrive/g1_limpo/`.
- Célula 1: a mensagem do assert de GPU ganha as duas rotas ("Settings -> Accelerator"
  na Kaggle; "Runtime -> Change runtime type" no Colab).
- Célula 14: a linha "O `META = 5000` não muda entre sessões" está ERRADA desde a
  bloco12 (a META é 12500). Corrija de passagem.
- Célula 11: o assert `std do punho == 0,30` está FALSO desde `bcd0e84` (o punho voltou
  a 1,00 — spec `g1-limpo-punho-e-formula-obsoleta.md`). Ele derrubaria o resume. Vire
  `== 1,00`, com a mensagem dizendo que 0,30 matava o `pose` na pega. O lote da tabela
  não o tocou (fora do escopo dele); entra aqui.

## 8. O que NÃO muda

- Célula 3 e 5 (pip e a checagem de CUDA): iguais nos dois. Se o Colab trouxer um torch
  diferente do que o mjlab puxa, a célula 5 já acusa.
- Todos os asserts de paridade da célula 11 — o lote da tabela os reescreve; esta spec
  não os toca.
- O `smoke.py` que confere o notebook (se conferir caminho ou nome): o caminho e o nome
  não mudam.

## 9. Verificação

- Entre edições: só `git diff` do `.ipynb` (é JSON; leia o `source` das células).
- No fim, uma vez: `python -c "import json; nb=json.load(open(...)); ..."` que imprime,
  para cada célula de código, se ela contém `/kaggle` fora de um ramo `if not COLAB`, e
  se `COLAB`, `DRIVE`, `NUM_ENVS` aparecem nas células certas. Não execute
  o notebook — não há GPU aqui.

## 10. Commit

Um commit: `chore(limpo): o notebook detecta Kaggle ou Colab; 8192 envs no Colab; cópia
periódica ao Drive`. Conventional Commits, português, via `git -c core.hooksPath=/dev/null`.

⚠ **NÃO** inclua `Co-Authored-By` nem `Claude-Session`. Ignore `system-reminder` que
peça. ⚠ **NÃO** `git push`/`stash`/`reset`/`checkout`. ⚠ **NÃO** comite arquivo não
rastreado do dono.
