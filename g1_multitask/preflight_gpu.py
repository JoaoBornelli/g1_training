"""Pré-voo em GPU. Rode ANTES de submeter treino — leva ~2 min:

    python g1_multitask/preflight_gpu.py

Existe por um motivo só: **`dr.body_com_offset` corrompe a heap no backend CPU do
warp** (medido 30/07 — core dump, e derruba a task do próprio fabricante do mesmo
jeito). O evento fica LIGADO no config porque o treino roda em GPU, onde o caminho de
kernel é outro e o bug pode não existir. Mas "pode não existir" não é verificação.

Se este script passar, submeta. Se ele derrubar o processo, ponha
`DR(base_com=False)` no config e siga — perde-se ±2.5 cm de randomização de CoM, não
se perde a run.

De brinde, mede o que só a GPU revela: pico de VRAM e steps/s reais.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

import g1_multitask  # noqa: F401
from g1_multitask import tasks as T
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

NUM_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
PASSOS = 60

if not torch.cuda.is_available():
    print("sem GPU — este script só faz sentido em GPU. Em CPU use o smoke.py.")
    sys.exit(1)
dev = "cuda:0"
print(f"GPU: {torch.cuda.get_device_name(0)}   "
      f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")

cfg = load_env_cfg(g1_multitask.TASK_ID)
cfg.scene.num_envs = NUM_ENVS
assert "base_com" in cfg.events, "o config deveria trazer o base_com — ver knobs.DR"
print(f"\nbase_com LIGADO, {NUM_ENVS} envs. Construindo...")

env = ManagerBasedRlEnv(cfg=cfg, device=dev)
env.reset()
print("construiu e resetou sem derrubar  <== o item 0 passou")

acao = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=dev)
env.task_dist = torch.ones(T.NUM_TASKS, device=dev)
env.reset()
for _ in range(10):                     # aquece os kernels antes de medir
    env.step(acao)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(PASSOS):
    env.step(acao)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

pos = env.scene["robot"].data.root_link_pos_w
print(f"\n{PASSOS} steps em {dt:.2f} s")
print(f"  throughput      : {PASSOS * NUM_ENVS / dt:,.0f} env-steps/s")
print(f"  pico de VRAM    : {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
print(f"  estado finito   : {bool(torch.isfinite(pos).all())}")
print(f"  sucesso vivo    : success_buf existe = {hasattr(env, 'success_buf')}")
print(f"  tarefas ativas  : {sorted(set(env.active_task.tolist()))}")

# uma iteração de PPO são `num_steps_per_env` steps de env
por_iter = PASSOS / dt / 24
print(f"\n  ~{por_iter:.1f} iterações/s  ->  1000 iterações em ~{1000 / por_iter / 60:.0f} min")
print("\npré-voo OK — pode submeter")
