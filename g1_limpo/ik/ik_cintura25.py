"""Gera a tabela de IK com a cintura limitada, partindo das poses aprovadas.

Igual ao `g1_limpo.ik.gera_botar`, com duas diferenças:
1. o `waist_pitch` (e o `waist_roll`) ficam em |q| <= --cintura-max graus;
2. o solver parte da pose aprovada da `ref_botar.npz` (mesma laje e caixa), e não da pose
   de repouso: a pose nova fica perto da aprovada e só muda o necessário.
"""
import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

import g1_limpo.ik.gera_botar as G


def euler_xy(quat):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    p = np.arcsin(np.clip(R[0, 2], -1, 1))
    r = np.arctan2(-R[1, 2], R[2, 2])
    q2 = np.zeros(4)
    mujoco.mju_euler2Quat(q2, np.array([r, p, 0.0]), b"xyz")
    if min(np.abs(q2 - quat).max(), np.abs(q2 + quat).max()) > 1e-3:
        r = np.arctan2(R[2, 1], R[1, 1])
        mujoco.mju_euler2Quat(q2, np.array([r, p, 0.0]), b"xyz")
    assert min(np.abs(q2 - quat).max(), np.abs(q2 + quat).max()) < 1e-3, (quat, q2)
    return r, p


CAIXA_NA_BORDA = False


def caixa_xyz(c, topo, a):
    if CAIXA_NA_BORDA:
        borda = G.AVANCO_LAJE - float(c.m.geom_size[mujoco.mj_name2id(
            c.m, mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")][0])
        return np.array([borda + a, 0.0, topo + a])
    return np.array([G.AVANCO_LAJE - G.RECUO_BORDA, 0.0, topo + a])


RAIZ_X = False


def poe(c, st, caixa):
    c.poe(st[:29], st[29], st[30], st[31], caixa)
    if RAIZ_X:
        c.d.qpos[0] = st[32]
        mujoco.mj_kinematics(c.m, c.d)
        mujoco.mj_comPos(c.m, c.d)


def resolve(c, topo, a, st, lo_c, hi_c, iters):
    c.cenario(topo, a)
    caixa = caixa_xyz(c, topo, a)
    st = st.copy()
    if RAIZ_X and len(st) == 32:
        st = np.concatenate([st, [0.0]])
    nv = len(st)
    st[:29] = np.clip(st[:29], lo_c, hi_c)
    for _ in range(iters):
        poe(c, st, caixa)
        ativos = np.flatnonzero(c.folgas() < c.folga_par + G.BANDA_ATIVA)
        f0 = G.residuo(c, caixa, a, G.PESO_REPOUSO, ativos)
        J = np.zeros((len(f0), nv))
        for k in range(nv):
            s = st.copy(); s[k] += 1e-5
            poe(c, s, caixa)
            J[:, k] = (G.residuo(c, caixa, a, G.PESO_REPOUSO, ativos) - f0) / 1e-5
        passo = J.T @ np.linalg.solve(J @ J.T + 2e-2 * np.eye(len(f0)), -f0)
        st = st + np.clip(passo, -0.15, 0.15)
        st[:29] = np.clip(st[:29], lo_c, hi_c)
        st[29] = np.clip(st[29], 0.25, 1.05)
        if nv == 33:
            st[32] = np.clip(st[32], -0.30, 0.10)
    poe(c, st, caixa)
    f = G.residuo(c, caixa, a)
    n = c.n_pe
    return float(np.abs(f[:n]).max()), float(np.abs(f[n:n + 6]).max()), st, caixa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cena", default="~/g1_pilota/")
    ap.add_argument("--ref", default="g1_limpo/ik/ref_botar.npz")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--cintura-max", type=float, default=25.0)
    ap.add_argument("--iters", type=int, default=240)
    ap.add_argument("--so", type=int, nargs="*", default=None, help="índices de pose")
    ap.add_argument("--folga-laje", type=float, default=None,
                    help="se dado, a laje entra nos pares de folga com esta distância mínima (m)")
    ap.add_argument("--raiz-x", action="store_true", help="a pelve pode recuar ou avançar em x")
    ap.add_argument("--topo", type=float, default=None, help="força o topo da laje (0 = caixa no chão)")
    ap.add_argument("--caixa-na-borda", action="store_true",
                    help="a face da caixa voltada ao robô fica rente à borda da laje")
    ap.add_argument("--sementes", type=int, default=0,
                    help="sementes extras a partir da pose de repouso, como o gerador original")
    args = ap.parse_args()
    global CAIXA_NA_BORDA, RAIZ_X
    CAIXA_NA_BORDA = args.caixa_na_borda
    RAIZ_X = args.raiz_x

    c = G.Cena(Path(args.cena).expanduser())
    ref = np.load(args.ref)
    assert ref["hash_mjcf"] == c.hash, "npz de outro MJCF"
    c.pes_de(Path(args.ref), 3)
    if args.folga_laje is not None:
        gl = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")
        robo = [g for g in range(c.m.ngeom) if c.m.geom_contype[g]
                and (mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("robot/")]
        c.pares = np.vstack([c.pares, np.array([(g, gl) for g in robo])])
        c.folga_par = np.concatenate([c.folga_par, np.full(len(robo), args.folga_laje)])
        print(f"[laje] {len(robo)} pares robô × laje, folga {args.folga_laje*1000:.0f} mm")
    folga = 0.10 * (c.hi - c.lo) / 2.0
    lo_c, hi_c = c.lo + folga, c.hi - folga
    nomes = [mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, 30)]
    for nome in ("robot/waist_pitch_joint", "robot/waist_roll_joint"):
        i = nomes.index(nome)
        lim = np.radians(args.cintura_max)
        lo_c[i], hi_c[i] = max(lo_c[i], -lim), min(hi_c[i], lim)
    iw = nomes.index("robot/waist_pitch_joint")
    ih = [nomes.index("robot/left_hip_pitch_joint"), nomes.index("robot/right_hip_pitch_joint")]

    ks = range(len(ref["topo"])) if args.so is None else args.so
    linhas = []
    for k in ks:
        topo, a = float(ref["topo"][k]), float(ref["meia"][k])
        if args.topo is not None:
            topo = args.topo
        q = ref["qpos"][k]
        r, p = euler_xy(q[3:7])
        st0 = np.concatenate([q[7:36], [q[2]], [r, p]])
        t0 = time.time()
        cand = [resolve(c, topo, a, st0, lo_c, hi_c, args.iters)]
        for sem in range(args.sementes):
            rng = np.random.default_rng(sem)
            q0 = np.clip(c.q_rep, c.lo + 1e-3, c.hi - 1e-3)
            ruido = (rng.random(c.n_juntas) - 0.5) * (0.7 if sem else 0.0)
            s0 = np.concatenate([np.clip(q0 + ruido, c.lo + 1e-3, c.hi - 1e-3), [0.78], [0.0, 0.0]])
            cand.append(resolve(c, topo, a, s0, lo_c, hi_c, args.iters))
        def nota(r):
            poe(c, r[2], r[3])
            return max(r[0], r[1], max(0.0, -float(c.folgas().min())))
        e_pe, e_mao, st, caixa = min(cand, key=nota)
        poe(c, st, caixa)
        g1, g2, g4, gp = G.grandezas(c, a, caixa)
        frac = float((np.abs(st[:29] - (c.hi + c.lo) / 2) / ((c.hi - c.lo) / 2)).max())
        fol = float(c.folgas().min())
        larg = float(abs(c.d.site_xpos[c.sid["left_foot"]][1] - c.d.site_xpos[c.sid["right_foot"]][1]))
        ok = (e_pe < 0.01 and e_mao < 0.02 and frac < 0.95 and gp < G.PELVE_PORTAO and fol > -0.002)
        print(f"pose {k:02d} laje {topo:.2f} caixa {2*a*100:.0f}cm | res_pe {e_pe:.4f} res_mao {e_mao:.4f} "
              f"| pelve {ref['pelve_z'][k]:.2f}->{g1:.2f} tronco {ref['tronco_incl'][k]:.0f}->{g2:.0f}° "
              f"| cintura {np.degrees(q[7+iw]):.0f}->{np.degrees(st[iw]):.0f}° "
              f"hip_pitch {np.degrees(q[7+ih[0]]):.0f}->{np.degrees(st[ih[0]]):.0f}° "
              f"| raiz x {c.d.qpos[0]:+.2f} | folga {fol*1000:.1f}mm {'ok' if ok else 'REJEITADA'} ({time.time()-t0:.0f}s)", flush=True)
        linhas.append(dict(topo=topo, meia=a, residuo=e_mao, residuo_pe=e_pe, pelve_z=g1, pelve_incl=gp,
                           folga_min=fol, pes_larg=larg, tronco_incl=g2, com_offset=g4, frac_max=frac,
                           aceita=ok, qpos=c.d.qpos.copy()))
    np.savez(args.saida, hash_mjcf=c.hash,
             **{key: np.array([l[key] for l in linhas]) for key in
                ("topo", "meia", "residuo", "residuo_pe", "pelve_z", "pelve_incl", "folga_min",
                 "pes_larg", "tronco_incl", "com_offset", "frac_max", "aceita", "qpos")})
    print(f"{sum(l['aceita'] for l in linhas)} de {len(linhas)} aceitas -> {args.saida}")


if __name__ == "__main__":
    main()
