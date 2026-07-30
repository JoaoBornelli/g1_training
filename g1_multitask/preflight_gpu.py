"""Pré-voo em GPU. Rode ANTES de submeter treino — leva ~2 min:

    python g1_multitask/preflight_gpu.py [num_envs]

Mede o que **só a GPU revela**: pico de VRAM e steps/s reais, que é o que dimensiona
os blocos de 2k-3k de verdade. E guarda que o `base_com` continua fora.

Ele nasceu para testar o item 0 — `dr.body_com_offset` — e esse item **fechou em
30/07**: o A/B em processos separados com `CUDA_LAUNCH_BLOCKING=1` mostrou que o
evento corrompe memória em GPU também, não só em CPU, e a resposta foi desligar. O
assert aqui virou o contrário do que era: agora ele falha se o evento VOLTAR.

Se ele derrubar o processo mesmo com o `base_com` fora, o suspeito passa a ser escala
— rode com `1024`, depois `2048`, depois `4096`, e veja onde estoura.
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
# O `base_com` tem que estar FORA: `dr.body_com_offset` corrompe memória em CPU e em
# GPU (fechado 30/07 por A/B com CUDA_LAUNCH_BLOCKING=1). Se ele reaparecer aqui,
# alguém religou sem rodar o A/B — e o sintoma é um `illegal memory access` cujo
# traceback aponta pro lugar errado, em `curriculum.py::_medir`.
assert "base_com" not in cfg.events, (
    "base_com voltou ao config e ele corrompe memória — ver knobs.DR.base_com")
dr_ativa = [e for e in ("foot_friction", "encoder_bias") if e in cfg.events]
print(f"\nDR ativa: {dr_ativa}  (base_com fora, de propósito)")
print(f"{NUM_ENVS} envs. Construindo...")

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
