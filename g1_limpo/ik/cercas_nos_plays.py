"""Parte 0, itens 11 e 14 (spec v3 §16): as sete cercas da §9.3 e a rampa do limite de junta (§9.2)
medidas nos CSV de play (`registra_juntas`) da raiz do repo. Reconstrói cada passo no MuJoCo (base,
29 juntas, caixa) e calcula a grandeza e o custo de cada cerca com peso 1, por arquivo e por fase.
Torque = PD da cena (kp, kd do mjb) sobre `alvo_pd` e `q`, com q̇ por diferença finita, cortado no
forcerange; arquivos sem `alvo_pd` ficam sem torque. Pé "em contato" = site do pé a menos de 2 cm
da altura de pé. Caixa "presa" = fora do apoio e força total > 20 N. MEDIDO 09/10: saída em
`agachado/cercas_nos_plays.txt`.

    cd ~/Documents/g1_training && .venv/bin/python g1_limpo/ik/cercas_nos_plays.py
"""
import csv
import glob
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from g1_limpo.ik import ik_mao_fixa as F, gera_botar as G, ik_entre_pernas as E, monta_pose as MP
c = G.Cena(F.CENA); E.prepara(c); m, d = c.m, c.d
jn = lambda s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + s)
act_j = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
act_q = np.array([int(m.jnt_qposadr[j]) for j in act_j]); act_dof = np.array([int(m.jnt_dofadr[j]) for j in act_j])
act_nome = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).split("/")[-1] for j in act_j]
kp = m.actuator_gainprm[:, 0]; kd = -m.actuator_biasprm[:, 2]; fmax = m.actuator_forcerange[:, 1]
lo, hi = m.jnt_range[1:30, 0], m.jnt_range[1:30, 1]
LIM_TORQUE, LIM_RAMPA, SOLA, GIRO, TRONCO_MAIS, COM_PISO, RECUO, PELVE = 0.8, 0.95, 5.0, 30.0, 15.0, 0.03, 0.08, 0.08
# referências da IK (aresta 20 cm) por topo
T = np.load("g1_limpo/ik/agachado/ref_agachado.npz")
ref = {}
for k in range(len(T["topo"])):
    if abs(float(T["meia"][k]) - 0.10) > 1e-6 or T["nas_maos"][k]: continue
    d.qpos[:] = T["qpos"][k]; mujoco.mj_kinematics(m, d)
    ref[float(T["topo"][k])] = (float(T["tronco_incl"][k]), float(T["pelve_z"][k]), float(T["caixa_xyz"][k][0] - d.xpos[c.bid_pelve][0]))
topos = np.array(sorted(ref)); R = np.array([ref[t] for t in topos])
def ref_ik(topo): return [float(np.interp(topo, topos, R[:, i])) for i in range(3)]
d.qpos[:] = T["qpos"][-1]; mujoco.mj_kinematics(m, d)
Z_PE = float(d.site_xpos[c.sid["left_foot"]][2]); PELVE_PE = float(d.xpos[c.bid_pelve][2])
print(f"site do pé de pé z={Z_PE:.3f}; pelve de pé {PELVE_PE:.3f}; refs IK por topo: " + "; ".join(f"{t:.2f}: tronco {a:.0f}° pelve {b:.2f} recuo {r:.2f}" for t, (a, b, r) in zip(topos, R)))
i_hy = {l: act_nome.index(f"{l}_hip_yaw_joint") for l in ("left", "right")}
def incl(bid): return float(np.degrees(G.inclinacao(c, bid)))
gt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")

def mede_csv(f):
    with open(f) as fh:
        r = csv.reader(fh); cab = next(r); rows = [[x for x in row] for row in r]
    ix = {n: k for k, n in enumerate(cab)}
    col = lambda n: np.array([float(row[ix[n]]) for row in rows])
    q = np.stack([col("q_" + n) for n in act_nome], 1)
    tem_pd = ("alvo_pd_" + act_nome[0]) in ix
    alvo = np.stack([col("alvo_pd_" + n) for n in act_nome], 1) if tem_pd else np.full_like(q, np.nan)
    t = col("t"); dt = float(np.median(np.diff(t)))
    qd = np.gradient(q, dt, axis=0)
    base = np.stack([col(k) for k in ("raiz_x", "raiz_y", "raiz_z", "raiz_qw", "raiz_qx", "raiz_qy", "raiz_qz")], 1)
    caixa = np.stack([col(k) for k in ("caixa_x", "caixa_y", "caixa_z", "caixa_qw", "caixa_qx", "caixa_qy", "caixa_qz")], 1)
    laje_z = col("laje_z"); fase = [row[ix["fase"]] for row in rows]
    fn = col("fn_tot") if "fn_tot" in ix else None
    tau = np.clip(kp * (alvo - q) - kd * qd, -fmax, fmax) if tem_pd else np.zeros_like(q)
    out = []
    for i in range(len(rows)):
        d.qpos[:7] = base[i]; d.qpos[act_q] = q[i]; d.qpos[c.adr_caixa:c.adr_caixa + 7] = caixa[i]
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        topo = laje_z[i] + c.meia_laje
        em_pega = fase[i] in ("pegar", "botar")
        tr_ik, pz_ik, rc_ik = ref_ik(topo) if em_pega else (0.0, PELVE_PE, np.nan)
        pes = {l: d.site_xpos[c.sid[f"{l}_foot"]][2] < Z_PE + 0.02 for l in ("left", "right")}
        sola = sum(max(0.0, incl(c.bid_pe[l]) - SOLA) for l in ("left", "right") if pes[l])
        giro_q = {"left": -np.degrees(q[i, i_hy["left"]]), "right": np.degrees(q[i, i_hy["right"]])}
        giro = sum(max(0.0, g - GIRO) for g in giro_q.values())
        tronco = incl(c.bid_torso); c_tronco = max(0.0, tronco - (tr_ik + TRONCO_MAIS))
        solta = caixa[i, 2] - 0.10 > topo + 0.01          # caixa fora do apoio
        presa = solta and (fn is None or fn[i] > 20.0)
        mc = 5.0 if presa else 0.0
        com = (c.m_robo * d.subtree_com[c.bid_pelve] + mc * caixa[i, :3]) / (c.m_robo + mc)
        margem = MP.margem_apoio(c, com[:2]) if (pes["left"] and pes["right"]) else np.nan
        c_com = max(0.0, COM_PISO - margem) * 100 if np.isfinite(margem) else 0.0
        frac = np.abs(tau[i]) / fmax if tem_pd else np.full(m.nu, np.nan)
        c_torque = float(np.sum(np.maximum(0.0, frac - LIM_TORQUE) / (1 - LIM_TORQUE))) if tem_pd else np.nan
        fr_j = np.abs(q[i] - (hi[act_dof - 6] + lo[act_dof - 6]) / 2) / ((hi[act_dof - 6] - lo[act_dof - 6]) / 2)
        c_rampa = float(np.sum(np.maximum(0.0, fr_j - LIM_RAMPA) / (1 - LIM_RAMPA)))
        recuo = float(np.linalg.norm(caixa[i, :2] - d.xpos[c.bid_pelve][:2]))
        c_recuo = max(0.0, abs(recuo - rc_ik) - RECUO) * 100 if em_pega else 0.0
        pz = float(d.xpos[c.bid_pelve][2]); c_pelve = max(0.0, abs(pz - pz_ik) - PELVE) * 100
        out.append((fase[i], sola, giro, c_tronco, c_com, c_torque, c_rampa, c_recuo, c_pelve,
                    max(incl(c.bid_pe[l]) for l in ("left", "right") if pes[l]) if any(pes.values()) else np.nan,
                    max(giro_q.values()), tronco, margem * 100 if np.isfinite(margem) else np.nan, (frac.max() if tem_pd else np.nan), (act_nome[int(np.argmax(frac))].replace("_joint", "") if tem_pd else "sem PD"), fr_j.max(),
                    recuo, pz, int(pes["left"]) + int(pes["right"]), presa))
    return out

NOMES = ["sola", "giro", "tronco", "CoM", "torque", "rampa", "recuo", "pelve"]
UNID = ["°", "°", "°", "cm", "", "", "cm", "cm"]
tot = {}
linhas = []
for f in sorted(glob.glob("*.csv")):
    if f.endswith(".limites.csv") or f in ("juntas.csv", "excluir.csv"): continue
    out = mede_csv(f)
    fases = sorted(set(o[0] for o in out), key=lambda s: ["espera", "pegar", "carregar", "botar", "andar", "parado"].index(s) if s in ("espera", "pegar", "carregar", "botar", "andar", "parado") else 9)
    for fs in fases:
        sel = [o for o in out if o[0] == fs]; n = len(sel)
        custos = np.array([o[1:9] for o in sel], dtype=float)
        gr = np.array([[o[9], o[10], o[11], o[12], o[13], o[15], o[16], o[17]] for o in sel], dtype=float)
        pes2 = np.mean([o[18] == 2 for o in sel]); presa = np.mean([o[19] for o in sel])
        jmax = max(set(o[14] for o in sel), key=[o[14] for o in sel].count)
        s = (f"{f.replace('.csv',''):22s} {fs:8s} n={n:4d} 2pés={pes2*100:3.0f}% presa={presa*100:3.0f}% | "
             + " | ".join((f"{nm} {np.mean(cu > 0)*100:3.0f}% méd {cu.mean():5.2f}{u} máx {cu.max():5.1f}" if np.isfinite(cu).all() else f"{nm} sem PD") for nm, u, cu in zip(NOMES, UNID, custos.T))
             + f" || sola p95 {np.nanpercentile(gr[:,0],95):4.1f}° giro p95 {np.percentile(gr[:,1],95):4.0f}° tronco p50/p95 {np.percentile(gr[:,2],50):3.0f}/{np.percentile(gr[:,2],95):3.0f}° CoM p05 {np.nanpercentile(gr[:,3],5):5.1f}cm frac_torque p95 {np.nanpercentile(gr[:,4],95)*100 if np.isfinite(gr[:,4]).any() else float("nan"):3.0f}% ({jmax}) curso p95 {np.percentile(gr[:,5],95)*100:3.0f}% recuo p50 {np.percentile(gr[:,6],50):.2f} pelve p50 {np.percentile(gr[:,7],50):.2f}")
        linhas.append(s); print(s, flush=True)
        tot.setdefault(fs, []).append(custos)
print("\nRESUMO por fase (custo médio por passo com peso 1; % de passos com custo > 0):")
for fs, lst in tot.items():
    cu = np.vstack(lst)
    print(f"{fs:8s} n={len(cu):5d} | " + " | ".join(f"{nm} {np.mean(x[np.isfinite(x)] > 0)*100:3.0f}% {np.nanmean(x):5.2f}{u}" for nm, u, x in zip(NOMES, UNID, cu.T)))
