"""Parte 0, itens 12 e 16 (spec v3 §16): as 21 poses da tabela em física (MuJoCo clássico), com os
servos PD da cena (kp/kd do mjb, forcerange do atuador). No lugar da política, um integrador do
erro de junta a 50 Hz sobre o alvo do servo, partindo do feedforward estático (`torques()` / kp).
No lugar do equilíbrio pelo IMU, um arnês mola-amortecedor FRACO no torso; a força dele sai na
tabela (5 a 20 N nas pegas, 2 N de pé). Modo `carga`: caixa longe, 5 kg pendurados nos punhos
(item 12, torque dinâmico). Modo `caixa`: caixa no lugar da IK, apoiada, sem carga (item 16,
força das palmas em repouso, por face). MEDIDO 09/10: saída em `agachado/segura_pose.txt`.

    .venv/bin/python g1_limpo/ik/segura_pose.py g1_limpo/ik/agachado/ref_agachado.npz [alfa=0.1]
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from g1_limpo.ik import ik_mao_fixa as F, gera_botar as G, ik_entre_pernas as E
c = G.Cena(F.CENA); E.prepara(c); m, d = c.m, c.d
T = np.load(sys.argv[1]); assert T["hash_mjcf"] == c.hash
ALFA, T_FORMA, T_MEDE, DEC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1, 4.0, 2.0, 4
# ARNÊS: mola-amortecedor fraco na pelve, no lugar do equilíbrio que a política faz pelo IMU. Um PD de
# junta não equilibra o pêndulo (kp do tornozelo 28 N·m/rad contra m·g·h ~ 240). Reporta a força.
K_LIN, D_LIN, K_ANG, D_ANG = 1000.0, 300.0, 100.0, 5.0
BH = c.bid_torso   # no torso: na pelve (3,8 kg, I 0,01) o amortecimento explícito explode
v6 = np.zeros(6); q4 = np.zeros(4); rv = np.zeros(3)
act_q = np.array([int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)])
act_dof = np.array([int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)])
act_nome = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).split("/")[-1].replace("_joint", "") for i in range(m.nu)]
kp, fmax = m.actuator_gainprm[:, 0].copy(), m.actuator_forcerange[:, 1].copy()
clo, chi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
gid_robo = set(g for g in range(m.ngeom) if m.geom_contype[g] and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("robot/"))
f6 = np.zeros(6)
i_wp = act_nome.index("waist_pitch")

def forcas_caixa():
    """Normais robô→caixa por face (esq y+, dir y−, frente x−, trás x+, topo z+)."""
    out = dict(esq=0.0, dir=0.0, frente=0.0, tras=0.0, topo=0.0)
    cx = d.xpos[c.bid_caixa]
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = con.geom1, con.geom2
        if c.gid_caixa not in (g1, g2) or not ({g1, g2} & gid_robo):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        fn = abs(f6[0])
        p = con.pos - cx
        ax = int(np.argmax(np.abs(p)))
        face = ("tras" if p[0] > 0 else "frente") if ax == 0 else ("esq" if p[1] > 0 else "dir") if ax == 1 else "topo"
        out[face] += fn
    return out

def roda(k, modo):
    topo, a, q_ik, caixa, massa = float(T["topo"][k]), float(T["meia"][k]), np.array(T["qpos"][k]), np.array(T["caixa_xyz"][k]), float(T["massa"][k])
    mujoco.mj_resetData(m, d)
    c.cenario(topo if topo > 0 else -5.0, a)
    m.body_mass[c.bid_caixa] = massa; m.body_inertia[c.bid_caixa] = massa * (2 * a) ** 2 / 6.0
    mujoco.mj_setConst(m, d)
    d.qpos[:] = q_ik; d.qvel[:] = 0.0
    if modo == "carga":
        d.qpos[c.adr_caixa:c.adr_caixa + 3] = (3.0, 0.0, a)
    mujoco.mj_forward(m, d)
    tau, _ = E.torques(c, caixa, massa if modo == "carga" else 0.0)
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    z0, t0 = float(d.xpos[c.bid_pelve][2]), float(np.degrees(G.inclinacao(c, c.bid_torso)))
    ff = q_ik[act_q] + tau[act_dof - 6] / kp
    z = np.zeros(m.nu)
    d.xfrc_applied[:] = 0.0
    if modo == "carga":
        for l in E.LADOS:
            d.xfrc_applied[c.bid_punho[l], 2] = -massa * 9.81 / 2
    n = int((T_FORMA + T_MEDE) / m.opt.timestep); n_mede = int(T_MEDE / m.opt.timestep)
    acc = dict(err=np.zeros(m.nu), frac=np.zeros(m.nu), sat=np.zeros(m.nu), z=0.0, t=0.0, n=0, fc=dict(esq=0.0, dir=0.0, frente=0.0, tras=0.0, topo=0.0))
    caixa0 = d.xpos[c.bid_caixa].copy(); caiu = False; traj = []
    p0, R0 = d.xpos[BH].copy(), d.xmat[BH].reshape(3, 3).copy()
    acc["arn"] = np.zeros(2)
    for i in range(n):
        if i % DEC == 0:
            z += ALFA * (q_ik[act_q] - d.qpos[act_q])
            d.ctrl[:] = np.clip(ff + z, clo, chi)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, BH, v6, 0)
        dp = d.xpos[BH] - p0
        mujoco.mju_mat2Quat(q4, (d.xmat[BH].reshape(3, 3) @ R0.T).ravel()); mujoco.mju_quat2Vel(rv, q4, 1.0)
        d.xfrc_applied[BH, :3] = -K_LIN * dp - D_LIN * v6[3:]
        d.xfrc_applied[BH, 3:] = -K_ANG * rv - D_ANG * v6[:3]
        mujoco.mj_step(m, d)
        if i >= n - n_mede:
            acc["arn"] += (np.linalg.norm(d.xfrc_applied[BH, :3]), np.linalg.norm(d.xfrc_applied[BH, 3:]))
        if i % 100 == 0:
            traj.append(float(np.degrees(np.abs(q_ik[act_q] - d.qpos[act_q]).max())))
        zp = float(d.xpos[c.bid_pelve][2])
        if zp < 0.5 * z0 or zp < 0.15:
            caiu = True; break
        if i >= n - n_mede:
            fr = np.abs(d.actuator_force) / fmax
            acc["err"] += np.abs(q_ik[act_q] - d.qpos[act_q]); acc["frac"] += fr; acc["sat"] += fr >= 0.98
            acc["z"] += zp; acc["t"] += float(np.degrees(G.inclinacao(c, c.bid_torso))); acc["n"] += 1
            if modo == "caixa":
                for kk, v in forcas_caixa().items(): acc["fc"][kk] += v
    nome = "de pé" if topo < 0 else ("chão" if topo == 0 else f"{topo:.2f}")
    if caiu or acc["n"] == 0:
        return f"{nome} {2*a*100:.0f}cm | {modo} | CAIU em {i*m.opt.timestep:.1f} s | traj {' '.join(f'{v:.0f}' for v in traj)}"
    N = acc["n"]; err = np.degrees(acc["err"] / N); frac = acc["frac"] / N; sat = acc["sat"] / N
    j = int(np.argmax(err)); jf = int(np.argmax(frac))
    altos = ", ".join(f"{act_nome[i]} {frac[i]*100:.0f}%" for i in np.argsort(-frac)[:3])
    s = (f"{nome} {2*a*100:.0f}cm | {modo} | erro máx {err[j]:.1f}° ({act_nome[j]}) | pelve {acc['z']/N - z0:+.3f} m | tronco {acc['t']/N - t0:+.1f}° | "
         f"torque: {altos} | cintura {frac[i_wp]*100:.0f}% | saturação {sat.max()*100:.0f}% do tempo ({act_nome[int(np.argmax(sat))]}) | arnês {acc['arn'][0]/N:.0f} N {acc['arn'][1]/N:.0f} N·m")
    if modo == "caixa":
        fc = {kk: v / N for kk, v in acc["fc"].items()}
        dc = d.xpos[c.bid_caixa] - caixa0
        s += f" | robô→caixa: esq {fc['esq']:.1f} dir {fc['dir']:.1f} frente {fc['frente']:.1f} trás {fc['tras']:.1f} topo {fc['topo']:.1f} N; caixa moveu {np.linalg.norm(dc[:2])*100:.1f} cm"
    return s

linhas = []
for k in range(len(T["topo"])):
    for modo in ("carga", "caixa"):
        if modo == "caixa" and T["nas_maos"][k]:
            continue
        linhas.append(roda(k, modo)); print(linhas[-1], flush=True)
