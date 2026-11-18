"""Mede por que o P4 do PEGAR não fecha numa política treinada da v3 (18/11).

Responde: no P4 a caixa fica ~0,16 m abaixo do alvo e o Φ_P4 vale 0,03–0,05, quando o núcleo sozinho daria 0,5–0,8.
Qual portão corta o Φ? Em cada passo com o alvo no P4 (PEGAR, sem caixa perdida), grava o d e o σ do alvo, o
núcleo, os cinco portões (desloc, incl, maos, tronco, recuo, com), o Φ, as forças de palma por face (F_esq, F_dir),
o F_pega, a força total robô → caixa (palmas + resto do corpo) e o erro da caixa ao alvo no frame da base
(frente, lado, z). Todas as fatias vão para a B (a política não muda; só o sorteio da fatia).

Uso:
    python -m g1_v3.mede_p4 --checkpoint CAMINHO/model_N.pt [--num-envs 1024] [--passos 3000] [--amostra]
                            [--saida ./mede_p4_saida] [--device cuda:0]
"""
import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import g1_v3.tarefa as tarefa  # registra a task
from g1_v3 import alvos, estimador, recompensas
from g1_v3.alvos import P4
from g1_v3.env_cfg import make_env_cfg
from g1_v3.knobs import ATIVO, TAREFAS
from g1_v3.metricas import CAIXA_ROBO
from g1_v3.runner import RunnerV3
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--num-envs", type=int, default=1024)
ap.add_argument("--passos", type=int, default=3000)
ap.add_argument("--amostra", action="store_true", help="ação amostrada com o std treinado (como no treino)")
ap.add_argument("--saida", default="./mede_p4_saida")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
saida = Path(args.saida)
saida.mkdir(parents=True, exist_ok=True)

env_cfg = make_env_cfg(play=False)
env_cfg.scene.num_envs = args.num_envs
rl = tarefa.rl_cfg()
env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
env = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
runner = RunnerV3(env, asdict(rl), device=args.device)
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
policy = runner.get_inference_policy(device=args.device)
u = env.unwrapped

# todo episódio novo cai na fatia B (ANDAR, B, C); os já em curso seguem até o reset
u.v3_cur.fracoes_fatias = lambda: torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)

# o `_phi_env` chama `phi_do_alvo` pelo módulo: a troca guarda d, σ e os portões da última chamada
phi_original, cap = recompensas.phi_do_alvo, {}


def phi_capturando(indice, d, sigma, fator, portoes):
    cap.update(d=d, sigma=sigma, **portoes)
    return phi_original(indice, d, sigma, fator, portoes)


recompensas.phi_do_alvo = phi_capturando
u.episode_length_buf = torch.randint_like(u.episode_length_buf, high=int(u.max_episode_length))
pegar, g = TAREFAS.index("PEGAR"), -float(u.sim.mj_model.opt.gravity[2])
linhas = []

with torch.inference_mode():
    obs = env.get_observations()
    for _ in range(args.passos):
        obs, _, _, _ = env.step(policy(obs, stochastic_output=args.amostra))
        m = (u.v3_alvo_idx == P4) & (u.v3_tarefa == pegar) & ~u.v3_perdeu
        if not bool(m.any()):
            continue
        p4 = torch.full_like(u.v3_alvo_idx, P4)
        phi = recompensas._phi_env(u, p4, u.v3_sigma)  # o mesmo Φ do progresso, com o alvo no P4
        robo, caixa = u.scene["robot"], u.scene["box"]
        yaw = estimador.yaw(robo.data.root_link_quat_w)
        e = caixa.data.root_link_pos_w - recompensas.alvo_da_caixa_w(u, p4)
        R_caixa = alvos.frame_da_caixa(caixa.data.root_link_quat_w, yaw)
        faces = u.scene["caixa_faces"].data
        f_esq, f_dir = recompensas.forcas_por_face(faces.normal, faces.force, faces.found > 0, R_caixa)
        sigma = cap["sigma"]
        linhas.append({n: v[m].float().cpu() for n, v in {
            "d": cap["d"], "sigma": sigma, "nucleo": 1 - torch.tanh(cap["d"] / (ATIVO.alvos.largura_tanh * sigma)),
            **{f"p_{n}": cap[n] for n in ("desloc", "incl", "maos", "tronco", "recuo", "com")}, "phi": phi,
            "F_esq": f_esq, "F_dir": f_dir,
            "F_pega": ATIVO.recompensa.margem_pega * u.v3_massa * g / (2.0 * ATIVO.dr.atrito_pad[0]),
            "F_total": u.scene[CAIXA_ROBO].data.force.norm(dim=-1).sum(-1),
            "e_frente": torch.cos(yaw) * e[:, 0] + torch.sin(yaw) * e[:, 1],
            "e_lado": -torch.sin(yaw) * e[:, 0] + torch.cos(yaw) * e[:, 1], "e_z": e[:, 2],
        }.items()})

d = {n: torch.cat([l[n] for l in linhas]).numpy() for n in linhas[0]} if linhas else {}
if not d:
    raise SystemExit("nenhum passo com o alvo no P4")
np.savez(saida / "passos_p4.npz", **d)
q = lambda x, p: float(np.percentile(x, p))
print(f"política: {'amostrada' if args.amostra else 'média'}; passos={args.passos}; envs={args.num_envs}; "
      f"passos com P4={len(d['d'])}")
print("grandeza | média | p10 | p50 | p90")
for n, v in d.items():
    print(f"{n} | {v.mean():.3f} | {q(v, 10):.3f} | {q(v, 50):.3f} | {q(v, 90):.3f}")
portoes = [n for n in d if n.startswith("p_")]
menor = np.stack([d[n] for n in portoes]).argmin(0)
print("portão que limita (o menor do passo): " + ", ".join(
    f"{n[2:]} {100 * (menor == i).mean():.1f}%" for i, n in enumerate(portoes)))
print(f"F palmas (esq+dir) / F total robô→caixa, p50: {q((d['F_esq'] + d['F_dir']) / np.maximum(d['F_total'], 1e-6), 50):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for n in portoes + ["nucleo", "phi"]:
    ax[0].hist(d[n], bins=40, range=(0, 1), histtype="step", label=n)
ax[0].legend(fontsize=7)
ax[0].set_title("portões, núcleo e Φ no P4", fontsize=9)
for n in ("F_esq", "F_dir", "F_total", "F_pega"):
    ax[1].hist(d[n], bins=40, histtype="step", label=n)
ax[1].legend(fontsize=7)
ax[1].set_title("forças (N)", fontsize=9)
for n in ("e_frente", "e_lado", "e_z"):
    ax[2].hist(d[n], bins=40, histtype="step", label=n)
ax[2].legend(fontsize=7)
ax[2].set_title("erro caixa − alvo no frame da base (m)", fontsize=9)
fig.tight_layout()
fig.savefig(saida / "p4.png", dpi=110)
print(f"saída em {saida.resolve()}")
