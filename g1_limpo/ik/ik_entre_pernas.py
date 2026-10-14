"""IK de teste: agachamento fundo, pés mais abertos, caixa no CHÃO entre as pernas.

Pedido do dono (08/10): a foto de outra equipe mostra o G1 agachado e estável pegando uma
lata no chão. Teste: pelve baixa, pés mais abertos que os 0,40 m da v3, mãos entre as
pernas nas faces laterais da caixa.

Reaproveita as tarefas do `gera_botar.residuo` (pés planos no chão, palmas nas faces,
normal do pad, CoM combinado sobre os pés, pelve <= 60°, sem rotação interna do joelho,
folgas robô x robô e robô x caixa) e as regras da v3 do `ik_mao_fixa` (mão a θ, cintura
<= 25°, margem do curso). Acrescenta: largura do apoio pedida, pés alinhados em x com
rumo simétrico, caixa no meio dos pés (mais `--dx` à frente), pelve pedida e folga de
10 mm entre o robô e o chão (fora os pés).

Limites de junta: a tabela da spec da v3 (§6.5), que é a do `cena.mjb`. `--margem 0` e
`--cintura 30` usam o robô real inteiro; o padrão (0,10 e 25°) são as regras da IK da v3.
`--hip-pitch-min -154` é a variante da página do produto (§6.5, divergência em aberto).

Mede também o torque ESTÁTICO de cada junta com a caixa nas mãos: gravidade do robô e
da caixa, reação do chão vertical nos dois pés, centro de pressão sob o CoM.

    cd ~/Documents/g1_training
    MUJOCO_GL=egl .venv/bin/python g1_limpo/ik/ik_entre_pernas.py
"""
import argparse
import sys
import time
from pathlib import Path

# ⚠ A RAIZ DO REPO NO `sys.path`, o idioma do `ve_botar.py`: o arquivo roda como
# `python g1_limpo/ik/<nome>.py` e como `python -m g1_limpo.ik.<nome>`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
AQUI = Path(__file__).resolve().parent

import mujoco                                            # noqa: E402
import numpy as np                                       # noqa: E402

import g1_limpo.ik.gera_botar as G                      # noqa: E402
from g1_limpo.ik import ik_mao_fixa as F                 # noqa: E402
from g1_limpo.ik.ve_botar import _grava_png              # noqa: E402

CAIXA_X = 0.30          # m: a caixa fica parada no mundo; a raiz anda livre em x
FOLGA_CHAO = 0.010
RUMO_MAX = np.radians(30.0)
SEM_MAOS = False
LADOS = ("left", "right")


def prepara(c):
    """Pares robô x chão (fora os elos do pé), punho e corpos do pé."""
    m = c.m
    nome = lambda t, i: mujoco.mj_id2name(m, t, i) or ""                       # noqa: E731
    chao = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    c.bid_pe = {l: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot/{l}_ankle_roll_link")
                for l in LADOS}
    robo = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] not in c.bid_pe.values()
            and nome(mujoco.mjtObj.mjOBJ_GEOM, g).startswith("robot/")]
    c.n_par_corpo = len(c.pares)
    c.pares = np.vstack([c.pares, np.array([(g, chao) for g in robo])])
    c.folga_par = np.concatenate([c.folga_par, np.full(len(robo), FOLGA_CHAO)])
    c.bid_punho = {l: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot/{l}_wrist_yaw_link")
                   for l in LADOS}
    c.geom_pe = [g for g in range(m.ngeom) if m.geom_bodyid[g] in c.bid_pe.values() and m.geom_contype[g]]
    c.nomes = [nome(mujoco.mjtObj.mjOBJ_JOINT, j).split("/")[-1].replace("_joint", "") for j in range(1, 30)]
    c.m_robo = float(m.body_mass[1:1 + 30].sum())


def semente(c, s, lo, hi):
    """Agachamento aberto de partida: quadril e joelho dobrados, coxas para fora."""
    q = c.q_rep.copy()
    alvo = {"hip_pitch": (-2.0, -2.0), "knee": (2.3, 2.3), "ankle_pitch": (-0.65, -0.65),
            "hip_roll": (0.3, -0.3), "hip_yaw": (0.3, -0.3)}       # esquerda, direita: para FORA
    for i, n in enumerate(c.nomes):
        for k, (e, d) in alvo.items():
            if n == f"left_{k}": q[i] = e
            if n == f"right_{k}": q[i] = d
    ruido = (np.random.default_rng(s).random(29) - 0.5) * (0.5 if s else 0.0)
    return np.clip(q + ruido, lo, hi)


def rumo(c, s):
    R = c.d.site_xmat[c.sid[s]].reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


JOELHO = None          # rad: dobra pedida do joelho (alvo mole, peso da pelve); None = livre


def residuo(c, caixa, a, ativos, theta, pelve, dx):
    d = c.d
    e = list(G.residuo(c, caixa, a, G.PESO_REPOUSO, ativos))
    if SEM_MAOS:
        e[9:19] = [0.0] * 10            # palmas (6) e normal do pad (4) fora: agachamento puro
    lf, rf = d.site_xpos[c.sid["left_foot"]], d.site_xpos[c.sid["right_foot"]]
    # pés alinhados em x, rumo simétrico, caixa entre os pés. A ponta do pé abre para fora
    # entre 0 e RUMO_MAX: sem o teto o solver gira o hip_roll a 158° e o pé a 155°
    re = rumo(c, "left_foot")
    e += [lf[0] - rf[0], re + rumo(c, "right_foot"), caixa[0] - 0.5 * (lf[0] + rf[0]) - dx,
          max(0.0, re - RUMO_MAX), max(0.0, -re)]
    if theta is not None:
        e += [F.W_MAO * (F.elev_dedos(c, l) - theta) for l in LADOS]
    if pelve is not None:
        e += [F.W_PELVE * (float(d.xpos[c.bid_pelve][2]) - pelve)]
    if JOELHO is not None:
        q = d.qpos[7:36]
        e += [0.1 * F.W_PELVE * (q[c.nomes.index(f"{l}_knee")] - JOELHO) for l in LADOS]
    return np.array(e)


def resolve(c, a, larg, theta, pelve, dx, s, iters, z):
    """Newton amortecido sobre 33 variáveis: 29 juntas, pelve z, roll, pitch e x da raiz."""
    G.LARGURA_MIN = G.LARGURA_MAX = larg
    caixa = np.array([CAIXA_X, 0.0, z])
    lo, hi = F.limites(c)
    # sementes pares partem com a pelve inclinada 29°, ímpares com 9°
    q0 = semente(c, s, lo, hi)
    if JOELHO is not None:
        for i, n in enumerate(c.nomes):
            if n.endswith("_knee"): q0[i] = min(JOELHO, hi[i])
            if n.endswith("_ankle_pitch"): q0[i] = lo[i] + 0.05
    st = np.concatenate([q0, [pelve or 0.40, 0.0, (0.5, 0.15)[s % 2], CAIXA_X - dx - 0.05]])
    for _ in range(iters):
        F.poe_pega(c, st, caixa, st[32])
        ativos = np.flatnonzero(c.folgas() < c.folga_par + G.BANDA_ATIVA)
        f0 = residuo(c, caixa, a, ativos, theta, pelve, dx)
        J = np.zeros((len(f0), 33))
        for j in range(33):
            v = st.copy(); v[j] += 1e-5
            F.poe_pega(c, v, caixa, v[32])
            J[:, j] = (residuo(c, caixa, a, ativos, theta, pelve, dx) - f0) / 1e-5
        st = st + np.clip(J.T @ np.linalg.solve(J @ J.T + 2e-2 * np.eye(len(f0)), -f0), -0.15, 0.15)
        st[:29] = np.clip(st[:29], lo, hi)
        st[29] = np.clip(st[29], 0.20, 1.05)
    F.poe_pega(c, st, caixa, st[32])
    f = G.residuo(c, caixa, a)
    g = residuo(c, caixa, a, np.array([], int), None, None, dx)[-5:]
    return dict(e_pe=float(np.abs(f[:9]).max()), e_mao=0.0 if SEM_MAOS else float(np.abs(f[9:15]).max()),
                e_geo=float(np.abs(g).max()), qpos=c.d.qpos.copy(), caixa=caixa)


def torques(c, caixa, m_caixa):
    """Torque estático das 29 juntas. Reação do chão vertical, com o centro de pressão de
    cada pé no x do CoM combinado; a divisão entre os pés fecha o momento em x. O resíduo
    nos 6 graus da base mede o quanto essa hipótese falha."""
    m, d = c.m, c.d
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    peso = 9.81 * np.array([c.m_robo, m_caixa])
    com = (c.m_robo * d.subtree_com[c.bid_pelve] + m_caixa * caixa) / (c.m_robo + m_caixa)
    y = {l: float(d.site_xpos[c.sid[f"{l}_foot"]][1]) for l in LADOS}
    W = peso.sum()
    f_e = W * (com[1] - y["right"]) / (y["left"] - y["right"])
    forcas = [(c.bid_pe["left"], np.array([com[0], y["left"], 0.0]), np.array([0, 0, f_e])),
              (c.bid_pe["right"], np.array([com[0], y["right"], 0.0]), np.array([0, 0, W - f_e]))]
    for l in LADOS:
        forcas.append((c.bid_punho[l], d.site_xpos[c.sid[f"{l}_palm"]].copy(),
                       np.array([0, 0, -peso[1] / 2])))
    jac = np.zeros((3, m.nv))
    tau = d.qfrc_bias[:35].copy()
    for b, p, fz in forcas:
        mujoco.mj_jac(m, d, jac, None, p, b)
        tau -= jac[:, :35].T @ fz
    return tau[6:], float(np.abs(tau[:6]).max())


def margem_sagital(c, com_x):
    """Distância do CoM à borda da frente ou de trás do apoio, em x."""
    xs = []
    for g in c.geom_pe:
        R = c.d.geom_xmat[g].reshape(3, 3); h = c.m.geom_size[g][1]
        xs += [c.d.geom_xpos[g][0] + s * h * R[0, 2] for s in (-1, 1)]
    return min(com_x - min(xs), max(xs) - com_x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--larguras", type=float, nargs="*", default=[0.40, 0.50, 0.60],
                    help="separação lateral dos pés em m (v3 = 0,40)")
    ap.add_argument("--pelve-z", type=float, nargs="*", default=[0.0, 0.40, 0.32],
                    help="altura pedida da pelve; 0 = livre")
    ap.add_argument("--meia", type=float, default=0.10, help="meia aresta da caixa")
    ap.add_argument("--caixa-z", type=float, default=0.0,
                    help="altura do CENTRO da caixa; 0 = apoiada no chão (z = meia)")
    ap.add_argument("--massa", type=float, default=5.0, help="kg da caixa no torque")
    ap.add_argument("--dx", type=float, default=0.0, help="caixa à frente do meio dos pés, m")
    ap.add_argument("--theta", type=float, default=-30.0, help="dedos contra o solo, graus; 999 = livre")
    ap.add_argument("--margem", type=float, default=0.10)
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--iters", type=int, default=240)
    ap.add_argument("--saida", default=str(AQUI / "entre_pernas"))
    ap.add_argument("--sem-fotos", action="store_true")
    ap.add_argument("--hip-pitch-min", type=float, default=None,
                    help="flexão do hip_pitch em graus; padrão = tabela da §6.5 (-145); -154 = página do produto")
    ap.add_argument("--rumo-max", type=float, default=30.0,
                    help="abertura máxima da ponta do pé, graus; regra de postura do teste, não limite do robô")
    ap.add_argument("--joelho", type=float, default=None, help="dobra pedida do joelho, rad (batente 2,88)")
    ap.add_argument("--cintura", type=float, default=25.0,
                    help="teto da cintura em graus; 25 = regra da IK da v3, 30 = robô (§6.5)")
    ap.add_argument("--sem-maos", action="store_true",
                    help="tira as tarefas das palmas: mede só o agachamento (use --massa 0 --caixa-z 3)")
    args = ap.parse_args()
    global SEM_MAOS, JOELHO, RUMO_MAX
    SEM_MAOS = args.sem_maos
    RUMO_MAX = np.radians(args.rumo_max)
    JOELHO = args.joelho
    F.MARGEM = args.margem
    F.CINTURA = np.radians(args.cintura)
    G.MASSA_CAIXA = args.massa
    theta = None if args.theta >= 999 else np.radians(args.theta)
    out = Path(args.saida).expanduser(); out.mkdir(parents=True, exist_ok=True)

    c = G.Cena(F.CENA)
    prepara(c)
    if args.hip_pitch_min is not None:
        for l in LADOS:
            c.lo[c.nomes.index(f"{l}_hip_pitch")] = np.radians(args.hip_pitch_min)
    c.cenario(-5.0, args.meia)                      # a laje fica enterrada, longe de tudo
    lim = np.array([abs(c.m.actuator_forcerange[np.flatnonzero(c.m.actuator_trnid[:, 0] == j + 1)[0], 1])
                    for j in range(29)])
    a = args.meia
    z = args.caixa_z if args.caixa_z > 0 else a
    rot_th = "livre" if theta is None else f"{args.theta:+.0f}°"
    linhas = [f"caixa {2*a*100:.0f} cm com o centro a {z:.2f} m, {args.massa:.0f} kg, {args.dx*100:+.0f} cm à frente do meio "
              f"dos pés; mão {rot_th}; margem {args.margem:.2f}; cintura <= {args.cintura:.0f}°; folga do chão "
              f"{FOLGA_CHAO*1000:.0f} mm; pelve <= 60°; hip_pitch >= {np.degrees(c.lo[c.nomes.index('left_hip_pitch')]):.0f}°",
              "pés | pelve pedida | veredito | res pe/mão/geo [cm] | pelve z, incl | tronco | mão E/D | "
              "joelho hip_pitch hip_roll hip_yaw ankle [rad] | rumo pé | folga corpo, chão [mm] | "
              "CoM–borda [cm] | torque % joelho hip_p hip_r ankle_p (res base N) | perto do batente"]
    print(linhas[0]); print(linhas[1])
    poses = []
    for larg in args.larguras:
        for pz in args.pelve_z:
            t0 = time.time()
            pelve = None if pz <= 0 else pz
            melhor = None
            for s in range(args.sementes):
                r = resolve(c, a, larg, theta, pelve, args.dx, s, args.iters, z)
                if melhor is None or max(r["e_pe"], r["e_mao"], r["e_geo"]) < \
                        max(melhor["e_pe"], melhor["e_mao"], melhor["e_geo"]):
                    melhor = r
            r = melhor
            c.d.qpos[:] = r["qpos"]; mujoco.mj_kinematics(c.m, c.d); mujoco.mj_comPos(c.m, c.d)
            mm = F.mede(c, r["qpos"])
            fol = c.folgas()
            f_corpo, f_chao = float(fol[:c.n_par_corpo].min()), float(fol[c.n_par_corpo:].min())
            gn = lambda g: (mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "?").split("/")[-1].replace("_collision", "")  # noqa: E731
            par = c.pares[int(np.argmin(fol[:c.n_par_corpo]))]
            com = (c.m_robo * c.d.subtree_com[c.bid_pelve] + args.massa * r["caixa"]) / (c.m_robo + args.massa)
            borda = margem_sagital(c, com[0])
            tau, res_base = torques(c, r["caixa"], args.massa)
            pct = 100 * np.abs(tau) / lim
            q = r["qpos"][7:36]; ix = lambda n: c.nomes.index(n)                # noqa: E731
            ok = (r["e_pe"] <= F.TOL_PE and r["e_mao"] <= F.TOL_MAO and r["e_geo"] <= 0.01
                  and f_corpo > -0.002 and f_chao > -0.002 and mm["frac"].max() <= 1.0 - args.margem + 0.005)
            alto = [f"{c.nomes[i]} {mm['resto'][i]:.0f}°" for i in np.argsort(-mm["frac"]) if mm["frac"][i] > 0.80]
            pior = lambda k: max(pct[ix(f"left_{k}")], pct[ix(f"right_{k}")])   # noqa: E731
            linhas.append(
                f"{larg:.2f} | {'livre' if pelve is None else f'{pelve:.2f}'} | {'ok' if ok else 'REJEITADA'} | "
                f"{r['e_pe']*100:.1f}/{r['e_mao']*100:.1f}/{r['e_geo']*100:.1f} | {mm['pelve']:.2f}, {mm['pelve_i']:.0f}° | "
                f"{mm['tronco']:.0f}° | {mm['mao'][0]:+.0f}/{mm['mao'][1]:+.0f} | "
                f"{q[ix('left_knee')]:.2f} {q[ix('left_hip_pitch')]:.2f} {q[ix('left_hip_roll')]:+.2f} "
                f"{q[ix('left_hip_yaw')]:+.2f} {q[ix('left_ankle_pitch')]:.2f} | "
                f"{np.degrees(rumo(c, 'left_foot')):+.0f}° | {f_corpo*1000:.0f} ({gn(par[0])} x {gn(par[1])}), {f_chao*1000:.0f} | {borda*100:.1f} | "
                f"{pior('knee'):.0f} {pior('hip_pitch'):.0f} {pior('hip_roll'):.0f} {pior('ankle_pitch'):.0f} "
                f"({res_base:.1f}) | {', '.join(alto) if alto else 'nenhuma'}  [{time.time()-t0:.0f}s]")
            print(linhas[-1], flush=True)
            poses.append(dict(larg=larg, pelve=pz, ok=ok, qpos=r["qpos"], tau=tau))
    (out / "entre_pernas.txt").write_text("\n".join(linhas) + "\n")
    np.savez(out / "entre_pernas.npz", hash_mjcf=c.hash, meia=a, massa=args.massa, dx=args.dx,
             caixa_xyz=np.array([CAIXA_X, 0.0, z]),
             **{k: np.array([p[k] for p in poses]) for k in ("larg", "pelve", "ok", "qpos", "tau")})
    if not args.sem_fotos:
        H, W = 420, 480
        rend = mujoco.Renderer(c.m, height=H, width=W)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[0] = opt.geomgroup[2] = 1
        tiras = []
        for p in poses:
            imgs = []
            for azim, elev in ((90.0, -8.0), (180.0, -12.0), (145.0, -25.0)):   # lado, frente, 3/4 de cima
                cam.distance, cam.azimuth, cam.elevation = 1.7, azim, elev
                c.d.qpos[:] = p["qpos"]; mujoco.mj_forward(c.m, c.d)
                cam.lookat[:] = (CAIXA_X - 0.10, 0.0, 0.35)
                rend.update_scene(c.d, camera=cam, scene_option=opt)
                imgs.append(rend.render().copy())
            sep = np.full((H, 8, 3), 255, np.uint8)
            tiras.append(np.concatenate([imgs[0], sep, imgs[1], sep, imgs[2]], axis=1))
        rend.close()
        grade = tiras[0]
        for t in tiras[1:]:
            grade = np.concatenate([grade, np.full((8, grade.shape[1], 3), 255, np.uint8), t], axis=0)
        _grava_png(out / "entre_pernas.png", grade)
        print(f"foto em {out / 'entre_pernas.png'} (linhas na ordem da tabela; colunas lado | frente | 3/4 de cima)")


if __name__ == "__main__":
    main()
