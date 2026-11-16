"""Mede a deriva de posição e de rumo por segmento de comando de uma política treinada da v3.

Responde: por que o ANDAR fica abaixo do portão? Cada segmento fechado grava dpos, rumo com sinal, tempo desde a
âncora e os canais da nota (linear e giro, núcleo médio desde a âncora), nas condições do treino. 16/11, T5 em estudo:
grava também o giro pelo ω_z médio do segmento, para comparar a nota de hoje com a nota que a T5 daria.

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
from g1_v3 import balanceador
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
wz_soma = torch.zeros(env.unwrapped.num_envs, device=args.device)  # Σ ω_z·dt desde a âncora, por env (giro da T5)


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
        env_.v3_perdeu[ids], twist.seg_carrega[ids], env_.v3_incl_max_seg[ids], terreno, k,
        deriva_pos_m=balanceador.no_degrau(k.deriva_pos_inicio, k.deriva_pos_m, cur.restricao(), 1))
    cmd = twist.seg_cmd[ids]
    # os canais da nota que o `_fecha` registra e o giro da T5: o núcleo do ω_z médio do segmento, que ignora o balanço
    # da pelve dentro da passada
    dur = twist.seg_dur[ids].clamp(min=1e-6)
    nota = twist.seg_nota[ids] / dur[:, None]
    wz_med = wz_soma[ids] / dur
    wz_soma[ids] = 0.0
    giro_t5 = torch.exp(-(cmd[:, 2] - wz_med) ** 2 / k.sigma_nota ** 2)
    carrega, ok = twist.seg_carrega[ids], (~falha).float()
    linhas.append({
        "cel": cel, "vx": cmd[:, 0], "vy": cmd[:, 1], "wz": cmd[:, 2], "dpos": dpos, "rumo": rumo,
        "t_ancora": twist._agora[ids] - ancora[:, 3], "falha": falha.clone(),
        "reg": (reg & ((twist.seg_dur[ids] > 0) | carrega)) | falha,  # o registro do `_fecha`
        "nota_lin": nota[:, 0], "nota_giro": nota[:, 1], "wz_med": wz_med, "giro_t5": giro_t5,
        # o x que o `_fecha` registra hoje e o x com o giro da T5 (a tarefa com caixa segue com o sucesso)
        "x_hoje": torch.where(carrega, suc.float(), nota.min(dim=1).values) * ok,
        "x_t5": torch.where(carrega, suc.float(), torch.minimum(nota[:, 0], giro_t5)) * ok,
    })
    fecha_original(ids, falha)


twist._fecha = fecha_medindo

with torch.inference_mode():
    obs = env.get_observations()
    for _ in range(args.passos):
        acao = policy(obs, stochastic_output=args.amostra)
        obs, _, _, _ = env.step(acao)
        # o ω_z do passo entra no segmento corrente depois da âncora, como o `seg_nota` (o reset e o fecho vêm antes, no step)
        apos = twist._agora >= env.unwrapped.v3_ancora[:, 3]
        wz_soma += twist.robot.data.root_link_ang_vel_b[:, 2] * apos * env.unwrapped.step_dt

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
p = cur.k.portao_andar
com_t = d["t_ancora"] > 0  # os percentis de taxa só valem com âncora feita (t ≤ 0: fechou antes da âncora)
t = np.where(com_t, d["t_ancora"], np.nan)
q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")

celulas = sorted(set(nome_cel))
print(f"política: {'amostrada' if args.amostra else 'média'}; passos={args.passos}; envs={args.num_envs}; "
      f"segmentos registrados={len(nome_cel)}")
cab = ["celula", "n", "nota hoje", "nota T5", "%seg≥portão hoje", "%seg≥portão T5", "lin", "giro", "giro T5",
       "|rumo|p50", "|rumo|p90", "dpos p50", "dpos p90", "|rumo|/t p50", "|rumo|/t p90", "dpos/t p50", "dpos/t p90",
       "viés rumo/t"]
print(" | ".join(cab))
for cel in celulas:
    g = nome_cel == cel
    gt = g & com_t
    rv, pv = arumo[gt] / t[gt], d["dpos"][gt] / t[gt]
    vies = float(np.mean(d["rumo"][gt] / t[gt])) if gt.any() else float("nan")
    v = [cel, int(g.sum()), d["x_hoje"][g].mean(), d["x_t5"][g].mean(), 100 * (d["x_hoje"][g] >= p).mean(),
         100 * (d["x_t5"][g] >= p).mean(), d["nota_lin"][g].mean(), d["nota_giro"][g].mean(), d["giro_t5"][g].mean(),
         q(arumo[g], 50), q(arumo[g], 90), q(d["dpos"][g], 50), q(d["dpos"][g], 90),
         q(rv, 50), q(rv, 90), q(pv, 50), q(pv, 90), vies]
    print(" | ".join(x if isinstance(x, str) else str(x) if isinstance(x, int) else f"{x:.3f}" for x in v))

if celulas:
    fig, ax = plt.subplots(len(celulas), 2, figsize=(10, 2.6 * len(celulas)), squeeze=False)
    for i, cel in enumerate(celulas):
        g = nome_cel == cel
        for j, (a, b, ra, rb, tit) in enumerate([("x_hoje", "x_t5", "hoje", "T5", "nota do segmento"),
                                                 ("nota_giro", "giro_t5", "giro hoje", "giro T5", "canal do giro")]):
            ax[i, j].hist(d[a][g], bins=40, range=(0, 1), alpha=0.6, label=ra)
            ax[i, j].hist(d[b][g], bins=40, range=(0, 1), alpha=0.6, label=rb)
            ax[i, j].axvline(p, color="red")
            ax[i, j].legend(fontsize=8)
            ax[i, j].set_title(f"{cel}: {tit}, n={int(g.sum())}", fontsize=9)
    fig.tight_layout()
    fig.savefig(saida / "nota.png", dpi=110)
print(f"saída em {saida.resolve()}")
