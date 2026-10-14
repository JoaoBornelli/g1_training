"""IK do BOTAR dentro de uma prateleira: um cubo oco sem a frente, sobre a laje.

O cubo tem piso (a própria laje), duas laterais, teto e fundo. A boca fica na borda da
laje voltada para o robô. A caixa fica dentro, com a face da frente 2 cm para dentro da
boca, apoiada no piso. As palmas ficam nas faces laterais, como na tabela v3.

O `cena.mjb` é compilado e não aceita geom novo. As paredes entram no IK reposicionando o
geom da laje como cada parede, uma por vez, só para o `mj_geomDistance`, e devolvendo a
laje ao lugar. As regras são as do `ik_cintura25.py` (cintura <= 25°, folga robô x laje de
10 mm, raiz livre em x), mais a folga robô x parede de 10 mm.

    PYTHONPATH=$PWD python ik_prateleira.py --folga 0.15 --saida prat_015.npz
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from g1_limpo.ik import ik_cintura25 as K  # noqa: E402

G = K.G
FOLGA_PAREDE = 0.010
DENTRO = 0.02          # a face da frente da caixa fica 2 cm para dentro da boca
FUNDO = 0.03           # folga entre a face de trás da caixa e o fundo
ESP = 0.01             # meia espessura das paredes

PAREDES: list[tuple[np.ndarray, np.ndarray]] = []
CAIXA = np.zeros(3)


def monta(c, topo, a, folga, folga_teto):
    """Calcula a caixa e as quatro paredes do cubo para esta laje e esta caixa."""
    global PAREDES, CAIXA
    gl = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")
    borda = G.AVANCO_LAJE - float(c.m.geom_size[gl][0])
    xc = borda + DENTRO + a
    CAIXA = np.array([xc, 0.0, topo + a])
    x0, x1 = borda, xc + a + FUNDO
    w = a + folga                       # meia largura interna
    h = 2 * a + folga_teto              # altura interna
    xm, dx = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
    PAREDES = [
        (np.array([xm, +(w + ESP), topo + h / 2]), np.array([dx + ESP, ESP, h / 2 + ESP])),
        (np.array([xm, -(w + ESP), topo + h / 2]), np.array([dx + ESP, ESP, h / 2 + ESP])),
        (np.array([xm, 0.0, topo + h + ESP]), np.array([dx + ESP, w + 2 * ESP, ESP])),
        (np.array([x1 + ESP, 0.0, topo + h / 2]), np.array([ESP, w + 2 * ESP, h / 2 + ESP])),
    ]


def dist_paredes(c, robo):
    """Distância de cada geom do robô a cada parede, com o geom da laje emprestado."""
    m, d = c.m, c.d
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")
    salvo = (m.geom_size[gl].copy(), float(m.geom_rbound[gl]), m.geom_aabb[gl].copy(),
             d.geom_xpos[gl].copy(), d.geom_xmat[gl].copy())
    ft = np.zeros(6)
    out = []
    try:
        for centro, meia in PAREDES:
            m.geom_size[gl] = meia
            m.geom_rbound[gl] = float(np.linalg.norm(meia))
            m.geom_aabb[gl] = np.concatenate([np.zeros(3), meia])
            d.geom_xpos[gl] = centro
            d.geom_xmat[gl] = np.eye(3).ravel()
            out.append([mujoco.mj_geomDistance(m, d, int(g), gl, G.DIST_MAX, ft) for g in robo])
    finally:
        m.geom_size[gl], m.geom_rbound[gl], m.geom_aabb[gl] = salvo[0], salvo[1], salvo[2]
        d.geom_xpos[gl], d.geom_xmat[gl] = salvo[3], salvo[4]
    return np.array(out)


def dist_pares(c, robo, pares):
    """Como `dist_paredes`, só para os pares (parede, geom) dados."""
    if len(pares) == 0:
        return np.zeros(0)
    m, d = c.m, c.d
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")
    salvo = (m.geom_size[gl].copy(), float(m.geom_rbound[gl]), m.geom_aabb[gl].copy(),
             d.geom_xpos[gl].copy(), d.geom_xmat[gl].copy())
    ft = np.zeros(6)
    out = np.zeros(len(pares))
    try:
        for w in np.unique(pares[:, 0]):
            centro, meia = PAREDES[w]
            m.geom_size[gl] = meia
            m.geom_rbound[gl] = float(np.linalg.norm(meia))
            m.geom_aabb[gl] = np.concatenate([np.zeros(3), meia])
            d.geom_xpos[gl] = centro
            d.geom_xmat[gl] = np.eye(3).ravel()
            for i in np.flatnonzero(pares[:, 0] == w):
                out[i] = mujoco.mj_geomDistance(m, d, int(robo[pares[i, 1]]), gl, G.DIST_MAX, ft)
    finally:
        m.geom_size[gl], m.geom_rbound[gl], m.geom_aabb[gl] = salvo[0], salvo[1], salvo[2]
        d.geom_xpos[gl], d.geom_xmat[gl] = salvo[3], salvo[4]
    return out


def resolve(c, topo, a, st, lo_c, hi_c, iters, raiz_max):
    """O `ik_cintura25.resolve`, com o avanço máximo da raiz como parâmetro."""
    c.cenario(topo, a)
    caixa = CAIXA.copy()
    st = st.copy()
    nv = len(st)
    st[:29] = np.clip(st[:29], lo_c, hi_c)
    for _ in range(iters):
        K.poe(c, st, caixa)
        ativos = np.flatnonzero(c.folgas() < c.folga_par + G.BANDA_ATIVA)
        f0 = G.residuo(c, caixa, a, G.PESO_REPOUSO, ativos)
        J = np.zeros((len(f0), nv))
        for k in range(nv):
            s = st.copy(); s[k] += 1e-5
            K.poe(c, s, caixa)
            J[:, k] = (G.residuo(c, caixa, a, G.PESO_REPOUSO, ativos) - f0) / 1e-5
        passo = J.T @ np.linalg.solve(J @ J.T + 2e-2 * np.eye(len(f0)), -f0)
        st = st + np.clip(passo, -0.15, 0.15)
        st[:29] = np.clip(st[:29], lo_c, hi_c)
        st[29] = np.clip(st[29], 0.25, 1.05)
        st[32] = np.clip(st[32], -0.30, raiz_max)
    K.poe(c, st, caixa)
    f = G.residuo(c, caixa, a)
    n = c.n_pe
    return float(np.abs(f[:n]).max()), float(np.abs(f[n:n + 6]).max()), st, caixa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cena", default="~/g1_pilota/")
    ap.add_argument("--ref", default="g1_limpo/ik/ref_v3_manipulacao.npz")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--folga", type=float, default=0.15, help="folga lateral por lado (m)")
    ap.add_argument("--folga-teto", type=float, default=None, help="folga acima da caixa (m)")
    ap.add_argument("--so", type=int, nargs="*", default=None, help="índices da ref")
    ap.add_argument("--iters", type=int, default=240)
    ap.add_argument("--sementes", type=int, default=0)
    ap.add_argument("--raiz-max", type=float, default=0.25, help="avanço máximo da raiz em x (m)")
    ap.add_argument("--palma-frente", type=float, default=0.0,
                    help="fração da meia aresta que as palmas recuam para a boca (0 = centro da face)")
    args = ap.parse_args()
    folga_teto = args.folga if args.folga_teto is None else args.folga_teto

    c = G.Cena(Path(args.cena).expanduser())
    ref = np.load(args.ref)
    assert ref["hash_mjcf"] == c.hash, "npz de outro MJCF"
    c.pes_de(Path("g1_limpo/ik/ref_botar.npz"), 3)
    gl = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")
    robo = [g for g in range(c.m.ngeom) if c.m.geom_contype[g]
            and (mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("robot/")]
    c.pares = np.vstack([c.pares, np.array([(g, gl) for g in robo])])
    c.folga_par = np.concatenate([c.folga_par, np.full(len(robo), 0.010)])

    K.RAIZ_X = True
    orig = G.residuo

    cache = {"chave": None, "pares": None}

    def residuo_com_paredes(cc, caixa, a, peso_rep=G.PESO_REPOUSO, ativos=None):
        e = orig(cc, caixa, a, peso_rep, ativos)
        dx = args.palma_frente * a
        e[cc.n_pe] += dx
        e[cc.n_pe + 3] += dx
        if ativos is None:
            dist = dist_paredes(cc, robo).ravel()
        else:
            if cache["chave"] is not ativos:
                completo = dist_paredes(cc, robo)
                cache["chave"] = ativos
                cache["pares"] = np.argwhere(completo < FOLGA_PAREDE + G.BANDA_ATIVA)
            dist = dist_pares(cc, robo, cache["pares"])
        return np.concatenate([e, np.maximum(0.0, FOLGA_PAREDE - dist)])

    G.residuo = residuo_com_paredes
    K.caixa_xyz = lambda cc, topo, a: CAIXA.copy()

    folga_j = 0.10 * (c.hi - c.lo) / 2.0
    lo_c, hi_c = c.lo + folga_j, c.hi - folga_j
    nomes = [mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, 30)]
    for nome in ("robot/waist_pitch_joint", "robot/waist_roll_joint"):
        i = nomes.index(nome)
        lim = np.radians(25.0)
        lo_c[i], hi_c[i] = max(lo_c[i], -lim), min(hi_c[i], lim)
    iw = nomes.index("robot/waist_pitch_joint")

    ks = range(len(ref["topo"])) if args.so is None else args.so
    linhas = []
    for k in ks:
        topo, a = float(ref["topo"][k]), float(ref["meia"][k])
        c.cenario(topo, a)
        monta(c, topo, a, args.folga, folga_teto)
        q = ref["qpos"][k]
        r, p = K.euler_xy(q[3:7])
        st0 = np.concatenate([q[7:36], [q[2]], [r, p], [q[0]]])
        t0 = time.time()
        cand = [resolve(c, topo, a, st0, lo_c, hi_c, args.iters, args.raiz_max)]
        for sem in range(args.sementes):
            rng = np.random.default_rng(sem)
            q0 = np.clip(c.q_rep, c.lo + 1e-3, c.hi - 1e-3)
            ruido = (rng.random(c.n_juntas) - 0.5) * (0.7 if sem else 0.0)
            s0 = np.concatenate([np.clip(q0 + ruido, c.lo + 1e-3, c.hi - 1e-3), [0.78], [0.0, 0.0], [q[0]]])
            cand.append(resolve(c, topo, a, s0, lo_c, hi_c, args.iters, args.raiz_max))

        def nota(res):
            K.poe(c, res[2], res[3])
            dp = float(dist_paredes(c, robo).min())
            return max(res[0], res[1], max(0.0, -float(c.folgas().min())), max(0.0, -dp))

        e_pe, e_mao, st, caixa = min(cand, key=nota)
        K.poe(c, st, caixa)
        g1, g2, g4, gp = G.grandezas(c, a, caixa)
        frac = float((np.abs(st[:29] - (c.hi + c.lo) / 2) / ((c.hi - c.lo) / 2)).max())
        fol = float(c.folgas().min())
        dpar = dist_paredes(c, robo)
        dp = float(dpar.min())
        qual = ["esq", "dir", "teto", "fundo"][int(np.argmin(dpar.min(axis=1)))]
        larg = float(abs(c.d.site_xpos[c.sid["left_foot"]][1] - c.d.site_xpos[c.sid["right_foot"]][1]))
        ok = (e_pe < 0.01 and e_mao < 0.02 and frac < 0.95 and gp < G.PELVE_PORTAO
              and fol > -0.002 and dp > -0.002)
        print(f"laje {topo:.2f} caixa {2*a*100:.0f}cm folga {args.folga*100:.1f}cm | res_pe {e_pe:.4f} "
              f"res_mao {e_mao:.4f} | pelve {g1:.2f} tronco {g2:.0f}° cintura {np.degrees(st[iw]):.0f}° "
              f"| raiz x {c.d.qpos[0]:+.2f} | folga corpo {fol*1000:.1f}mm parede {dp*1000:.1f}mm ({qual}) "
              f"| {'ok' if ok else 'REJEITADA'} ({time.time()-t0:.0f}s)", flush=True)
        linhas.append(dict(topo=topo, meia=a, residuo=e_mao, residuo_pe=e_pe, pelve_z=g1, pelve_incl=gp,
                           folga_min=fol, folga_parede=dp, pes_larg=larg, tronco_incl=g2, com_offset=g4,
                           frac_max=frac, aceita=ok, qpos=c.d.qpos.copy(), caixa_xyz=caixa,
                           paredes=np.array([np.concatenate(pw) for pw in PAREDES])))
    G.residuo = orig
    np.savez(args.saida, hash_mjcf=c.hash, folga=np.array([args.folga]), folga_teto=np.array([folga_teto]),
             **{key: np.array([l[key] for l in linhas]) for key in linhas[0]})
    print(f"{sum(l['aceita'] for l in linhas)} de {len(linhas)} aceitas -> {args.saida}")


if __name__ == "__main__":
    main()
