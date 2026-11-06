"""Mede a deriva de posição e de rumo por segmento de comando de uma política treinada da v3.

Responde: por que `ANDAR parado` e `ANDAR reto` ficam abaixo do portão? Cada segmento fechado grava dpos, rumo com
sinal, tempo desde a âncora e quais canais (linear, giro) reprovaram, nas condições do treino.

Uso:
    python -m g1_v3.mede_deriva --checkpoint CAMINHO/model_N.pt [--num-envs 1024] [--passos 3000] [--amostra]
                                [--saida ./mede_deriva_saida] [--device cuda:0]
"""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import g1_v3.tarefa as tarefa  # registra a task
from g1_v3.comando import sucesso_do_segmento
from g1_v3.env_cfg import make_env_cfg
from g1_v3.runner import RunnerV3
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import wrap_to_pi

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--num-envs", type=int, default=1024)
ap.add_argument("--passos", type=int, default=3000)
ap.add_argument("--amostra", action="store_true", help="ação amostrada com o std treinado (como no treino)")
ap.add_argument("--saida", default="./mede_deriva_saida")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
saida = Path(args.saida)
saida.mkdir(parents=True, exist_ok=True)

# Treino (play=False): o portão é medido com empurrão, DR e âncora que se move depois do empurrão.
env_cfg = make_env_cfg(play=False)
env_cfg.scene.num_envs = args.num_envs
rl = tarefa.rl_cfg()
env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
env = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
runner = RunnerV3(env, asdict(rl), device=args.device)
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
policy = runner.get_inference_policy(device=args.device)

cur = env.unwrapped.v3_cur
nomes = [f"{c.tarefa} {c.tipo}" for c in cur.celulas]
twist = env.unwrapped.command_manager.get_term("twist")
k = twist.k
fecha_original = twist._fecha
linhas = []  # um dict de tensores (em CPU) por chamada de `_fecha`


def fecha_medindo(ids, falha):
    # Lê o estado ANTES do original: ele zera seg_proj/seg_int.
    env_, ancora = env.unwrapped, env.unwrapped.v3_ancora[ids]
    dpos = (twist.pose_ult[ids, :2] - ancora[:, :2]).norm(dim=1)
    rumo = wrap_to_pi(twist.pose_ult[ids, 2] - ancora[:, 2])
    cel = twist.seg_cel[ids]
    terreno = cel == cur.id_celula("CARREGAR", "terreno")
    # mesmos argumentos do `_fecha`
    reg, suc = sucesso_do_segmento(
        twist.seg_cmd[ids], twist.seg_proj[ids], twist.seg_int[ids], dpos, rumo.abs(), twist._agora[ids], ancora[:, 3],
        env_.v3_perdeu[ids], twist.seg_carrega[ids], env_.v3_incl_max_seg[ids], terreno, k)
    cmd = twist.seg_cmd[ids]
    razao = twist.seg_proj[ids] / twist.seg_int[ids].clamp(min=1e-6)
    ativo_lin = cmd[:, :2].norm(dim=1) >= k.limiar_canal
    ativo_giro = cmd[:, 2].abs() >= k.limiar_canal
    ok_lin = torch.where(ativo_lin, razao[:, 0] >= k.razao_portao, dpos < k.deriva_pos_m)
    ok_giro = torch.where(ativo_giro, razao[:, 1] >= k.razao_portao, rumo.abs() < k.deriva_rumo_rad)
    linhas.append({
        "cel": cel, "vx": cmd[:, 0], "vy": cmd[:, 1], "wz": cmd[:, 2], "dpos": dpos, "rumo": rumo,
        "t_ancora": twist._agora[ids] - ancora[:, 3], "reg": reg | falha, "suc": suc & ~falha,
        "ok_lin": ok_lin, "ok_giro": ok_giro, "falha": falha.clone(), "razao_lin": razao[:, 0],
    })
    fecha_original(ids, falha)


twist._fecha = fecha_medindo

with torch.inference_mode():
    obs = env.get_observations()
    for _ in range(args.passos):
        acao = policy(obs, stochastic_output=args.amostra)
        obs, _, _, _ = env.step(acao)

d = {n: torch.cat([l[n] for l in linhas]).float().cpu().numpy() for n in linhas[0]}
nome_cel = np.array([nomes[int(c)] if c >= 0 else "sem célula" for c in d["cel"]])
cols = [n for n in d if n != "cel"]
with open(saida / "segmentos.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["celula"] + cols)
    w.writerows([[nome_cel[i]] + [d[n][i] for n in cols] for i in range(len(nome_cel))])

# `reg` verdadeiro: só o que entra no currículo
m = (d["reg"] > 0) & (nome_cel != "sem célula")
d = {n: v[m] for n, v in d.items()}
nome_cel = nome_cel[m]
arumo = np.abs(d["rumo"])
ok_lin, ok_giro, suc = d["ok_lin"] > 0, d["ok_giro"] > 0, d["suc"] > 0
com_t = d["t_ancora"] > 0  # os percentis de taxa só valem com âncora feita (t ≤ 0: fechou antes da âncora)
t = np.where(com_t, d["t_ancora"], np.nan)
q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")

celulas = sorted(set(nome_cel))
print(f"política: {'amostrada' if args.amostra else 'média'}; passos={args.passos}; envs={args.num_envs}; "
      f"segmentos registrados={len(nome_cel)}")
cab = ["celula", "n", "sucesso", "%rep_lin", "%rep_giro", "|rumo|p50", "|rumo|p90", "dpos p50", "dpos p90",
       "|rumo|/t p50", "|rumo|/t p90", "dpos/t p50", "dpos/t p90", "viés rumo/t"]
print(" | ".join(cab))
for cel in celulas:
    g = nome_cel == cel
    gt = g & com_t
    rv, pv = arumo[gt] / t[gt], d["dpos"][gt] / t[gt]
    vies = float(np.mean(d["rumo"][gt] / t[gt])) if gt.any() else float("nan")
    v = [cel, int(g.sum()), suc[g].mean(), 100 * (~ok_lin[g]).mean(), 100 * (~ok_giro[g]).mean(),
         q(arumo[g], 50), q(arumo[g], 90), q(d["dpos"][g], 50), q(d["dpos"][g], 90),
         q(rv, 50), q(rv, 90), q(pv, 50), q(pv, 90), vies]
    print(" | ".join(x if isinstance(x, str) else str(x) if isinstance(x, int) else f"{x:.3f}" for x in v))

if celulas:
    fig, ax = plt.subplots(len(celulas), 2, figsize=(10, 2.6 * len(celulas)), squeeze=False)
    for i, cel in enumerate(celulas):
        g = nome_cel == cel
        ax[i, 0].hist(arumo[g], bins=40, color="tab:blue")
        ax[i, 0].axvline(k.deriva_rumo_rad, color="red")
        ax[i, 0].set_title(f"{cel}: |rumo| (rad), n={int(g.sum())}", fontsize=9)
        ax[i, 1].hist(d["dpos"][g], bins=40, color="tab:green")
        ax[i, 1].axvline(k.deriva_pos_m, color="red")
        ax[i, 1].set_title(f"{cel}: dpos (m)", fontsize=9)
    fig.tight_layout()
    fig.savefig(saida / "deriva.png", dpi=110)
print(f"saída em {saida.resolve()}")
