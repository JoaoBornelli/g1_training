"""Gera o notebook do Colab. Rode e comite o `.ipynb` que sai.

    python g1_poc/colab/gera_notebook.py

O notebook é gerado, e não editado à mão, por um motivo: um `.ipynb` editado no Colab
volta com `outputs`, `execution_count` e metadados de sessão, e o `git diff` fica
ilegível. A fonte de verdade é este arquivo.
"""
from __future__ import annotations

import json
import pathlib

MD, CODE = "markdown", "code"

CELULAS: list[tuple[str, str]] = [
(MD, r"""# g1_poc no Google Colab

Duas coisas que o Colab tem e a Kaggle não tinha, e as duas mudam o fluxo:

1. **A sessão morre.** Free desconecta em algumas horas; Pro vai a 12 h. Portanto o
   `resume` deixa de ser um teste e passa a ser parte do ciclo normal.
2. **O disco é volátil.** `/content` é apagado. Portanto os logs e os checkpoints vão
   para o **Drive**, e o código fica em `/content` (o Drive é lento para I/O de
   muitos arquivos pequenos).

Ordem das células: GPU → Drive → código → instalar → verificar → smoke na CPU →
pré-voo na GPU → treino → resume → leitura → TensorBoard.

**Não pule o pré-voo.** Ele custa um minuto e diz se o `num_envs` cabe. Descobrir OOM
na iteração 300 custa a sessão."""),

(MD, "## 1. GPU — antes de instalar nada"),
(CODE, r'''import subprocess, sys

smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
     "--format=csv,noheader"], capture_output=True, text=True)
print(smi.stdout or smi.stderr)
assert smi.returncode == 0 and smi.stdout.strip(), \
    "sem GPU: Ambiente de execução -> Alterar o tipo -> GPU"
print("python", sys.version.split()[0])

# Sugestão de `num_envs` pela VRAM. O pré-voo (célula 7) confirma; isto é o ponto de
# partida. A cena tem robô + caixa + prateleira e 6 sensores de contato, portanto ela
# é mais pesada que a `velocity` pura do mjlab.
nome, vram = smi.stdout.split(",")[0].strip(), smi.stdout.split(",")[1].strip()
mb = int("".join(c for c in vram if c.isdigit()))
NUM_ENVS = 2048 if mb < 20000 else (4096 if mb < 45000 else 8192)
print(f"\nGPU {nome} com {mb} MiB -> comece com num_envs = {NUM_ENVS}")
print("Se der OOM, caia para a metade. Se sobrar VRAM no pré-voo, dobre.")'''),

(MD, r"""## 2. Drive — é aqui que o treino sobrevive à sessão

`RAIZ_DRIVE` guarda três coisas:

- `logs/` — para onde o mjlab escreve. É o que o `resume` lê.
- `entrada/` — onde você põe um checkpoint à mão, se quiser retomar um de outra
  sessão.

⚠ O mjlab resolve o resume como `<log_root>/<experiment_name>/<load_run>/<checkpoint>`
(`scripts/train.py:133` → `utils/os.py::get_checkpoint_path`). Os **três** níveis têm
de bater entre a célula de treino e a de resume. Elas derivam o caminho do mesmo
lugar, e não o digitam — era o bug do notebook anterior."""),
(CODE, r'''import pathlib
from google.colab import drive

drive.mount("/content/drive")

RAIZ_DRIVE = pathlib.Path("/content/drive/MyDrive/g1_poc")
LOG_ROOT = RAIZ_DRIVE / "logs"
ENTRADA = RAIZ_DRIVE / "entrada"
for d in (LOG_ROOT, ENTRADA):
    d.mkdir(parents=True, exist_ok=True)

print("LOG_ROOT =", LOG_ROOT)
print("ENTRADA  =", ENTRADA)
print("\nconteúdo de LOG_ROOT:")
for p in sorted(LOG_ROOT.rglob("*"))[:20]:
    print("  ", p.relative_to(LOG_ROOT))'''),

(MD, r"""## 3. Código — do GitHub, não de um dataset

O repo é público, portanto `git clone` basta. Rodar de novo esta célula faz `pull`."""),
(CODE, r'''import pathlib, subprocess

BRANCH = "exp/g1-poc"
RAIZ = pathlib.Path("/content/g1_training")

if not RAIZ.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH,
                    "https://github.com/JoaoBornelli/g1_training.git", str(RAIZ)],
                   check=True)
else:
    subprocess.run(["git", "-C", str(RAIZ), "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(RAIZ), "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", str(RAIZ), "reset", "--hard", f"origin/{BRANCH}"],
                   check=True)

print(subprocess.run(["git", "-C", str(RAIZ), "log", "--oneline", "-3"],
                     capture_output=True, text=True).stdout)'''),

(MD, "## 4. Instalar"),
(CODE, r'''REQ = RAIZ / "g1_poc" / "colab" / "requirements.txt"
assert REQ.exists(), REQ
print(REQ.read_text())
!pip install -q -r {REQ}'''),

(MD, r"""## 5. Verificar — num interpretador NOVO

O `pip` da célula acima pode ter trocado alguma biblioteca já importada neste kernel.
Perguntar ao kernel atual daria a resposta velha. Portanto a verificação roda num
**subprocesso**.

Se o torch estiver abaixo de 2.7.0, o mjlab não roda. No Colab é seguro atualizar
(`pip install -U torch`), mas **é obrigatório reiniciar o runtime depois** — o torch
registra operadores C++ no import, e recarregar levanta
`Only a single TORCH_LIBRARY can be used to register the namespace triton`."""),
(CODE, r'''CHECK = r"""
import torch, mujoco, warp, mjlab, sys
print("torch      ", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device     ", torch.cuda.get_device_name(0))
    print("capability ", torch.cuda.get_device_capability(0))
print("mujoco     ", mujoco.__version__)
print("warp       ", warp.config.version)
print("mjlab      ", getattr(mjlab, "__version__", "?"))
assert torch.cuda.is_available(), "torch NAO ve a GPU"
major, minor = (int(x) for x in torch.__version__.split(".")[:2])
assert (major, minor) >= (2, 7), (
    f"mjlab exige torch>=2.7.0 e este e {torch.__version__}.\\n"
    "  rode: pip install -U torch  e REINICIE o runtime")
print("\nok")
"""
import subprocess, sys
r = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True)
print(r.stdout)
print(r.stderr[-2000:] if r.returncode else "")
assert r.returncode == 0, "verificação falhou — leia o erro acima"'''),

(MD, r"""## 6. Smoke na CPU — segundos, sem GPU

Ele confere os contratos e os invariantes. Roda antes de gastar um minuto de GPU."""),
(CODE, r'''!cd {RAIZ} && python -m g1_poc.smoke 2>&1 | tail -40'''),

(MD, r"""## 7. Pré-voo na GPU — **o item 0**

Instancia a cena com `NUM_ENVS`, dá 20 passos, e mede VRAM e passos por segundo.

Se estourar a memória, a célula falha aqui, em um minuto, e não na iteração 300."""),
(CODE, r'''import os, sys, time, importlib
os.environ.setdefault("MUJOCO_GL", "egl")   # headless; inerte se nada renderizar
sys.path.insert(0, str(RAIZ))

import torch
import g1_poc
from g1_poc.env_cfg import make_g1_poc_env_cfg, OBS_ATOR, OBS_CRITICO
from mjlab.envs import ManagerBasedRlEnv

cfg = make_g1_poc_env_cfg()
cfg.scene.num_envs = NUM_ENVS
torch.cuda.reset_peak_memory_stats()

t0 = time.time()
env = ManagerBasedRlEnv(cfg, device="cuda")
obs, _ = env.reset()
print(f"instanciou em {time.time()-t0:.1f} s")
print(f"obs ator    = {obs['actor'].shape[-1]}  (esperado {OBS_ATOR})")
print(f"obs crítico = {obs['critic'].shape[-1]}  (esperado {OBS_CRITICO})")
assert obs["actor"].shape[-1] == OBS_ATOR
assert obs["critic"].shape[-1] == OBS_CRITICO

acao = torch.zeros(NUM_ENVS, env.action_manager.total_action_dim, device="cuda")
for _ in range(5):                       # aquece o JIT do warp
    env.step(acao)
torch.cuda.synchronize()
t0 = time.time()
N = 20
for _ in range(N):
    env.step(acao)
torch.cuda.synchronize()
dt = time.time() - t0

pico = torch.cuda.max_memory_allocated() / 2**30
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"\npassos/s    = {N * NUM_ENVS / dt:,.0f}")
print(f"VRAM pico   = {pico:.2f} GiB de {total:.2f} GiB  ({100*pico/total:.0f}%)")
print(f"\numa iteração de 24 passos leva ~{24*dt/N:.2f} s")
print(f"5000 iterações levam ~{5000*24*dt/N/3600:.1f} h")
if pico / total > 0.85:
    print("\n⚠ acima de 85% da VRAM. Caia o num_envs para a metade.")
env.close()
del env
torch.cuda.empty_cache()'''),

(MD, r"""## 8. Treino — do zero

`resume = False`. O checkpoint do bloco 1 **não** serve de ponto de partida: a
política aprendeu a não se mover (σ 0,46 contra 1,0 inicial, pico do pé em 4 mm), e
tirar a penalidade não desfaz o comportamento aprendido — só para de cobrar por ele.

Os asserts do `[CONFERE]` são o contrato deste pacote. Dois deles custaram 6,6 horas
de GPU e estão marcados."""),
(CODE, r'''import dataclasses, os, sys, importlib

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["TORCHRUNX_LOG_DIR"] = "/content/torchrunx"   # local: log de rank
sys.path.insert(0, str(RAIZ))

import g1_poc
from g1_poc import curriculo as CU
from g1_poc import eventos as EV
from g1_poc import recompensas as R
from mjlab.scripts.train import TrainConfig, launch_training

TASK = g1_poc.TASK_ID                     # "Mjlab-G1-Poc"
RUN = "bloco2"                            # o nome da run, FIXO. O resume o reusa.

cfg = TrainConfig.from_task(TASK)
cfg = dataclasses.replace(cfg, log_root=str(LOG_ROOT))
cfg.env.scene.num_envs = NUM_ENVS
cfg.agent.max_iterations = 3000
cfg.agent.resume = False
cfg.agent.run_name = RUN

e, rw, cu, tm = cfg.env.events, cfg.env.rewards, cfg.env.curriculum, cfg.env.terminations

print(f"[CONFERE] experiment    = {cfg.agent.experiment_name}")
print(f"[CONFERE] num_envs      = {cfg.env.scene.num_envs}")
print(f"[CONFERE] comandos      = {list(cfg.env.commands)}")
print(f"[CONFERE] recompensas   = {len(rw)}")
print(f"[CONFERE] terminações   = {len(tm)}  {sorted(tm)}")
print(f"[CONFERE] eventos       = {list(e)}")
print(f"[CONFERE] currículo     = {list(cu)}")
print(f"[CONFERE] hinge         = {rw['joint_vel_hinge'].func.__name__}")
print(f"[CONFERE] tetos do hinge= {len(rw['joint_vel_hinge'].params['max_vel_manipulando'])} padrões")

# --- o contrato do pacote ---
assert cfg.agent.experiment_name == g1_poc.EXPERIMENT
# ORDEM: `caixa_alvo` resolve `poc_twist_zero`, e o `twist` lê no mesmo passo
assert list(cfg.env.commands)[0] == "caixa_alvo"
assert tm["time_out"].time_out is True, "sem isso o rsl_rl trata o fim como fracasso"

# ⚠ CUSTOU 6,6 h #1 — o wipe do passo 999. Com o resample do comando igual à duração
# do episódio, o `time_left` cruza zero UM PASSO antes do `time_out` e zera
# `episode_success`. O nível lia sucesso 0 em todo episódio que chegava ao fim.
assert cfg.env.commands["caixa_alvo"].resampling_time_range[0] > cfg.env.episode_length_s

# ⚠ CUSTOU 6,6 h #2 — os dois freios de movimento por passo global. Eles consumiram
# 96% da penalidade e 55% do sinal positivo. Agora o gate é por competência.
assert cu["hinge"].func is CU.peso_por_competencia
assert cu["action_rate"].func is CU.peso_por_competencia
assert rw["joint_vel_hinge"].func is R.hinge_por_forma, "o hinge tem de ser por forma"

# a ordem do currículo: quem lê a forma do episódio que ACABOU vem antes de `forma`
ordem = list(cu)
assert ordem.index("command_vel") < ordem.index("forma")
assert ordem.index("nivel") < ordem.index("forma")
assert ordem.index("forma") < ordem.index("hinge")

# a entrega do navegador é SÓ da manipulação
assert e["reset_base"].func is EV.reset_base_por_forma
assert "entrega_do_navegador" in e
assert list(e).index("reset_base") < list(e).index("entrega_do_navegador")

# ⚠ CUSTOU 8,6 h #3 — o bloco 2 colapsou por GIRO. Quatro consertos, e os quatro
# têm de estar presentes juntos; qualquer um sozinho não resolve.
#   1. o yaw do reset volta ao ±3,14 do fabricante NA LOCOMOÇÃO
assert e["reset_base"].params["faixa_loco"]["yaw"][1] > 3.0
assert e["reset_base"].params["faixa_manipula"]["yaw"][1] < 0.5
#   2. a locomoção é a do FABRICANTE: nenhum termo inventado, escala de ação pura
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
_fab = unitree_g1_flat_env_cfg(play=False)
assert set(rw) - set(_fab.rewards) == {
    "staged", "precise_pos", "precise_ori", "squeeze", "unload", "load",
    "postura_ereta", "sustentacao", "joint_vel_hinge"}, "termo inventado na locomoção"
assert cfg.env.actions["joint_pos"].scale == _fab.actions["joint_pos"].scale
assert cu["command_vel"].params["velocity_stages"] == \
    _fab.curriculum["command_vel"].params["velocity_stages"]
assert "action_rate" not in cu and "twist_ranges" not in cu
#   3. a janela de espera: todo episódio começa PARADO
assert cfg.env.commands["caixa_alvo"].espera_s[1] > 0.0
#   4. o balanço de forma começa em locomoção PURA e desce por competência
assert cu["forma"].params["balanco"] is not None, "auto_balanco tem de estar ligado"
assert cu["forma"].params["frac_locomocao"] == 1.0, "o bloco começa em locomoção pura"
# e o `dof_pos_limits` volta ao valor do fabricante (era 10x)
assert rw["dof_pos_limits"].weight == -1.0
assert list(e).index("reset_cena") < list(e).index("afasta_cena")

# `dr.body_com_offset` corrompe a heap — medido em CPU e em GPU
assert "base_com" not in e
# a DR de atrito é por EPISÓDIO, compartilhada, e na CAIXA
fr = e["caixa_friction"]
assert fr.mode == "reset" and fr.params["shared_random"] is True
assert fr.params["asset_cfg"].name == "box"

print("\n[CONFERE] tudo ok. lançando.\n")
launch_training(TASK, cfg)'''),

(MD, r"""## 9. Resume — o portão da sessão de Colab

Rode esta célula quando a sessão cair. Ela é a **mesma** configuração da célula 8, com
`resume = True`.

⚠ **`load_run` é um regex ANCORADO NO INÍCIO** (`re.match`, `utils/os.py:42`), e o
`launch_training` nomeia o diretório como `<timestamp>_<run_name>`
(`scripts/train.py:187-189`). Portanto:

    load_run = "bloco2"      ->  NUNCA casa com `2026-08-21_11-45-00_bloco2`
    load_run = ".*bloco2"    ->  casa, e o mjlab pega o alfabeticamente ÚLTIMO,
                                 que com timestamp no nome é o mais recente

Dentro da run, os checkpoints são ordenados com `f"{m:0>15}"` — zero-padded, portanto
`model_9.pt` vem antes de `model_22296.pt`, e não depois. Isso o mjlab já faz certo.

Cada sessão de Colab cria um diretório novo. O TensorBoard da célula 11 aponta para o
`LOG_ROOT` inteiro, portanto ele mostra todos eles juntos."""),
(CODE, r'''import dataclasses, sys
sys.path.insert(0, str(RAIZ))

import g1_poc
from mjlab.scripts.train import TrainConfig, launch_training

TASK = g1_poc.TASK_ID
RUN = "bloco2"                            # o MESMO da célula 8

# `LOAD_RUN` pode ter sido fixado pela célula 9b (semente de fora). Senão, ele é o
# regex que casa qualquer diretório terminado em `_bloco2`.
LOAD_RUN = globals().get("LOAD_RUN") or f".*{RUN}"

# ⚠ Os TRÊS níveis têm de bater: <log_root>/<experiment>/<load_run>/<checkpoint>.
# Derivar aqui, e não digitar, é o que impede as duas células de divergirem.
EXP_DIR = LOG_ROOT / g1_poc.EXPERIMENT
assert EXP_DIR.exists(), f"não existe {EXP_DIR} — rodou a célula 8?"

import re
runs = sorted(d for d in EXP_DIR.iterdir() if d.is_dir() and re.match(LOAD_RUN, d.name))
assert runs, (f"nenhum diretório em {EXP_DIR} casa com o regex {LOAD_RUN!r}.\n"
              f"  existem: {[d.name for d in EXP_DIR.iterdir() if d.is_dir()]}")
ESCOLHIDA = runs[-1]                      # o mjlab ordena alfabeticamente e pega o último
pts = sorted(ESCOLHIDA.glob("model_*.pt"),
             key=lambda f: int("".join(c for c in f.stem if c.isdigit()) or 0))
assert pts, f"nenhum model_*.pt em {ESCOLHIDA}"
print(f"log_root   = {LOG_ROOT}")
print(f"experiment = {g1_poc.EXPERIMENT}")
print(f"load_run   = {LOAD_RUN!r}  ->  {ESCOLHIDA.name}")
print(f"checkpoints= {[p.name for p in pts[-3:]]}  (o mjlab pega o de maior número)")

cfg = TrainConfig.from_task(TASK)
cfg = dataclasses.replace(cfg, log_root=str(LOG_ROOT))
cfg.env.scene.num_envs = NUM_ENVS
cfg.agent.max_iterations = 3000
cfg.agent.resume = True
cfg.agent.run_name = RUN
cfg.agent.load_run = LOAD_RUN
cfg.agent.load_checkpoint = pts[-1].name

# ⚠ Warm-start SEMPRE com 5e-4. Sem isso os primeiros updates destroem o equilíbrio e
# o `fell_over` dispara. Lição do repositório, e o ADR-0001 declarou a mitigação sem
# nunca a aplicar.
cfg.agent.algorithm.learning_rate = 5e-4

# ⚠ O que NÃO volta do checkpoint: `poc_nivel`, o estágio dos gates e as EMAs. O
# runner só salva `common_step_counter`. Portanto o nível recomeça em 0 e os freios
# recomeçam SOLTOS. É o lado seguro, e é declarado.
print(f"\nretomando de {cfg.agent.load_checkpoint} com lr = "
      f"{cfg.agent.algorithm.learning_rate}\n")
launch_training(TASK, cfg)'''),

(MD, r"""### 9b. Trazer um checkpoint de fora (opcional)

Use quando o `.pt` veio de outra sessão ou de outra máquina.

Ele vai para um diretório de run PRÓPRIO, chamado `semente`, e a célula fixa
`LOAD_RUN = "semente"`. Duas razões:

- o `load_run` é ancorado no início, portanto o nome exato casa;
- separar a semente das runs com timestamp impede que o alfabético escolha a run
  errada.

Rode esta célula **antes** da 9."""),
(CODE, r'''import shutil, sys
sys.path.insert(0, str(RAIZ))
import g1_poc

RUN_DIR = LOG_ROOT / g1_poc.EXPERIMENT / "semente"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ordena pelo NÚMERO, e não pelo nome: `model_9.pt` viria depois de `model_22296.pt`
# em ordem alfabética.
achados = sorted(ENTRADA.rglob("model_*.pt"),
                 key=lambda f: int("".join(c for c in f.stem if c.isdigit()) or 0))
assert achados, f"nenhum model_*.pt em {ENTRADA} — suba o checkpoint para o Drive"
for f in achados:
    print(f"  {f.name}   ({f.stat().st_size / 2**20:.1f} MB)")

ORIGEM = achados[-1]
DESTINO = RUN_DIR / ORIGEM.name
if DESTINO.resolve() != ORIGEM.resolve():
    shutil.copy(ORIGEM, DESTINO)

LOAD_RUN = "semente"        # a célula 9 lê esta variável

print(f"""
log_root   = {LOG_ROOT}
experiment = {g1_poc.EXPERIMENT}
load_run   = {LOAD_RUN}
checkpoint = {DESTINO.name}

no lugar   = {DESTINO}

agora rode a célula 9.""")'''),

(MD, r"""## 10. Leitura — a tabela que decide o bloco

O `Episode_Reward/*` do rsl_rl é a soma do episódio dividida por
`max_episode_length_s`. Com episódios curtos, todo valor sai dividido por um número
pequeno — e foi por isso que dois freios consumiram 55% do sinal positivo por 5000
iterações sem aparecer no painel.

Esta célula desfaz a normalização, desdilui as métricas de caixa e roda a escada de
corte da §17.

`--demo` roda a análise nos números medidos na it 5000 do bloco 1. É o autoteste."""),
(CODE, r'''!cd {RAIZ} && python g1_poc/leitura.py {LOG_ROOT}/g1_poc'''),

(MD, "## 11. TensorBoard"),
(CODE, r'''%load_ext tensorboard
%tensorboard --logdir {LOG_ROOT}'''),
]


def main() -> None:
    celulas = []
    for tipo, fonte in CELULAS:
        linhas = fonte.split("\n")
        src = [l + "\n" for l in linhas[:-1]] + [linhas[-1]]
        c = {"cell_type": tipo, "metadata": {}, "source": src}
        if tipo == CODE:
            c["outputs"] = []
            c["execution_count"] = None
        celulas.append(c)

    nb = {
        "cells": celulas,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    saida = pathlib.Path(__file__).parent / "g1_poc_colab.ipynb"
    saida.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(f"escrito: {saida}  ({len(celulas)} células)")


if __name__ == "__main__":
    main()
