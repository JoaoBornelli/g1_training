"""IK do agachamento a partir da pose montada à mão pelo dono (`poses_manuais/agachado 00.npz`).

ALVO DE POSTURA (o termo de repouso do `gera_botar.residuo`, peso `--peso-pose`): a mistura da
pose do dono (caixa no chão) com a pose de pé com caixa aprovada (caixa a 0,90 m), com peso
t = (z da caixa − meia) / (0,90 − meia). SEMENTE: no chão a pose do dono; na laje a pega aprovada
da v3 da mesma laje e tamanho, com a mão acima do tampo. As tarefas são as da IK da v3: solas planas
no chão, palmas nas faces da caixa com a normal do pad, dedos a θ, CoM combinado a `--com-piso`
da borda das solas (a cerca da spec §9.3; `--com-piso 0` volta a "CoM no meio dos pés"), pelve
<= 60°, folgas (corpo, caixa, chão), juntas a 90% do curso da tabela da spec (§6.5) e cintura
<= 25°. Os pés não copiam a pose do dono: ponta livre, simétricos e alinhados; largura presa à
mistura de postura ± 3 cm (ou livre entre 0,20 m e `--largura-max` com `--largura-livre`).

Varre a altura do apoio (chão e as lajes da v3), o tamanho e a massa da caixa. A caixa fica no
x da v3 (laje 0,15 m à frente dela); a raiz anda em x. Foto: linha 0 = a pose do dono como foi
salva; depois uma linha por altura, com o primeiro tamanho e a primeira massa; por último a pose
de pé com caixa aprovada. No npz, `topo` = -1 e `nas_maos` = True marcam a pose de pé. Colunas: lado | frente | 3/4 de cima.

    cd ~/Documents/g1_training
    MUJOCO_GL=egl .venv/bin/python g1_limpo/ik/ik_agachado.py
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
from g1_limpo.ik import ik_entre_pernas as E             # noqa: E402
from g1_limpo.ik import ik_mao_fixa as F                 # noqa: E402
from g1_limpo.ik import monta_pose as MP                 # noqa: E402
from g1_limpo.ik.ve_botar import _grava_png              # noqa: E402

LADOS = ("left", "right")


COM_PISO = None        # m: CoM a esta distância ou mais da borda das solas (cerca da spec §9.3); None = no meio dos pés
PESO_COM = 1.0         # multiplica a cerca do CoM
SOLA_MAX = 5.0         # graus: aceite da inclinação da sola, o mesmo da cerca da spec (§9.3)
PESO_MAO = 1.0         # multiplica o resíduo das palmas (6) e da normal do pad (4)


def residuo(c, caixa, a, ativos, theta, rumo0, peso):
    e = G.residuo(c, caixa, a, peso, ativos)
    e[22:24] = 0.0                   # giro da coxa livre: a restrição é a sola plana (spec v3 §6.5)
    e[9:19] *= PESO_MAO
    d = c.d
    if COM_PISO is not None:
        # a tarefa "CoM no meio dos pés" (19, 20) vira a cerca da spec: dentro do polígono das
        # solas com margem; o y continua centrado (pernas simétricas)
        m_robo = float(c.m.body_mass[1:1 + 30].sum())
        com = (m_robo * d.subtree_com[c.bid_pelve] + G.MASSA_CAIXA * caixa) / (m_robo + G.MASSA_CAIXA)
        e[19] = PESO_COM * max(0.0, COM_PISO - MP.margem_apoio(c, com[:2]))
        e[20] = com[1] - 0.5 * (d.site_xpos[c.sid["left_foot"]][1] + d.site_xpos[c.sid["right_foot"]][1])
    lf, rf = d.site_xpos[c.sid["left_foot"]], d.site_xpos[c.sid["right_foot"]]
    re, rd = E.rumo(c, "left_foot"), E.rumo(c, "right_foot")
    extra = [lf[0] - rf[0], re + rd]               # pés alinhados e simétricos; largura e ponta livres
    extra += [F.W_MAO * (F.elev_dedos(c, l) - theta) for l in LADOS]
    return np.concatenate([e, extra])


def resolve(c, q_ref, caixa, a, theta, rumo0, peso, s, iters):
    """Newton amortecido sobre 33 variáveis: 29 juntas, pelve z, roll, pitch e x da raiz."""
    lo, hi = F.limites(c)
    from g1_limpo.ik.ik_cintura25 import euler_xy
    r, p = euler_xy(q_ref[3:7])
    ruido = (np.random.default_rng(s).random(29) - 0.5) * (0.2 if s else 0.0)
    st = np.concatenate([np.clip(q_ref[7:36] + ruido, lo, hi), [q_ref[2], r, p, q_ref[0]]])
    for _ in range(iters):
        F.poe_pega(c, st, caixa, st[32])
        ativos = np.flatnonzero(c.folgas() < c.folga_par + G.BANDA_ATIVA)
        f0 = residuo(c, caixa, a, ativos, theta, rumo0, peso)
        J = np.zeros((len(f0), 33))
        for j in range(33):
            v = st.copy(); v[j] += 1e-5
            F.poe_pega(c, v, caixa, v[32])
            J[:, j] = (residuo(c, caixa, a, ativos, theta, rumo0, peso) - f0) / 1e-5
        st = st + np.clip(J.T @ np.linalg.solve(J @ J.T + 2e-2 * np.eye(len(f0)), -f0), -0.15, 0.15)
        st[:29] = np.clip(st[:29], lo, hi)
        st[29] = np.clip(st[29], 0.20, 1.05)
    F.poe_pega(c, st, caixa, st[32])
    f = G.residuo(c, caixa, a)
    g = residuo(c, caixa, a, np.array([], int), theta, rumo0, 0.0)[-4:-2]
    pe = np.abs(np.concatenate([f[:9], g]))
    nomes_pe = ["E z", "E incl x", "E incl y", "D z", "D incl x", "D incl y", "simetria y", "largura >", "largura <", "alinh x", "rumo sim"]
    # ⚠ a inclinação da sola (índices 1, 2, 4, 5: seno do ângulo) tem aceite próprio, em graus,
    # igual à cerca da spec (§9.3: 5°); o resto do pé (altura, simetria, largura, alinhamento) em metros
    incl = np.degrees(np.arcsin(np.clip(pe[[1, 2, 4, 5]].max(), 0, 1)))
    resto = pe[[0, 3, 6, 7, 8, 9, 10]].max()
    return dict(pe_pior=nomes_pe[int(np.argmax(pe))], e_pe=float(resto), sola=float(incl),
                e_mao=float(np.abs(f[9:15]).max()), qpos=c.d.qpos.copy())


def mede(c, qpos, caixa, massa, q_ref, lim):
    c.d.qpos[:] = qpos; mujoco.mj_kinematics(c.m, c.d); mujoco.mj_comPos(c.m, c.d)
    d = c.d
    q = qpos[7:36]
    frac = np.abs(q - (c.hi + c.lo) / 2) / ((c.hi - c.lo) / 2)
    resto = np.degrees(np.minimum(q - c.lo, c.hi - q))
    com = (c.m_robo * d.subtree_com[c.bid_pelve] + massa * caixa) / (c.m_robo + massa)
    apoio = MP.margem_apoio(c, com[:2])
    fol = c.folgas()
    nome = lambda g: (mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "?").split("/")[-1].replace("_collision", "")  # noqa: E731
    i = int(np.argmin(fol))
    desvio = np.degrees(np.abs(q - q_ref[7:36]))
    k = int(np.argmax(desvio))
    tau, _ = E.torques(c, caixa, massa)
    pct = 100 * np.abs(tau) / lim
    tq = lambda j: max(pct[c.nomes.index(f"left_{j}")], pct[c.nomes.index(f"right_{j}")])  # noqa: E731
    mujoco.mj_kinematics(c.m, c.d); mujoco.mj_comPos(c.m, c.d)
    return dict(
        pelve=float(d.xpos[c.bid_pelve][2]), pelve_i=float(np.degrees(G.inclinacao(c, c.bid_pelve))),
        tronco=float(np.degrees(G.inclinacao(c, c.bid_torso))), apoio=apoio, folga=float(fol[i]),
        par=f"{nome(c.pares[i][0])} x {nome(c.pares[i][1])}", frac=float(frac.max()),
        joelho=float(np.degrees(q[c.nomes.index("left_knee")])),
        hip=float(np.degrees(q[c.nomes.index("left_hip_pitch")])),
        desvio=f"{c.nomes[k]} {desvio[k]:.0f}°",
        torque=f"{tq('knee'):.0f} {tq('hip_pitch'):.0f} {tq('ankle_pitch'):.0f} "
               f"{pct[c.nomes.index('waist_pitch')]:.0f} {tq('shoulder_pitch'):.0f} {tq('elbow'):.0f}",
        alto=", ".join(f"{c.nomes[j]} {resto[j]:.0f}°" for j in np.argsort(-frac) if frac[j] > 0.9) or "nenhuma")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=Path, default=AQUI / "poses_manuais" / "agachado 00.npz")
    ap.add_argument("--alturas", type=float, nargs="*", default=[0.0, 0.04, 0.15, 0.30, 0.45, 0.55],
                    help="topo do apoio da caixa, m; 0 = chão (sem laje), > 0 = laje da v3")
    ap.add_argument("--arestas", type=float, nargs="*", default=[0.20])
    ap.add_argument("--massas", type=float, nargs="*", default=[5.0])
    ap.add_argument("--nome", default="ref_agachado", help="nome base dos arquivos de saída")
    ap.add_argument("--largura-max", type=float, default=0.50, help="teto da largura dos pés na IK, m (modo livre)")
    ap.add_argument("--largura-livre", action="store_true",
                    help="largura dos pés livre entre 0,20 m e --largura-max; o padrão prende a largura à "
                         "mistura de postura (pose do dono no chão -> pose de pé no alto), com 3 cm de folga")
    ap.add_argument("--de-pe", type=Path, default=AQUI / "mao_fixa" / "ref_v3_de_pe_theta-30_tamanhos.npz",
                    help="pose de pé com caixa a 0,90 m, por tamanho (aprovada 05/10)")
    ap.add_argument("--lajes", type=Path, default=AQUI / "mao_fixa" / "ref_v3_manipulacao_theta-30.npz",
                    help="pegas aprovadas da v3, semente nas lajes")
    ap.add_argument("--theta", type=float, default=-30.0, help="dedos contra o solo, graus (v3: -30)")
    ap.add_argument("--peso-pose", type=float, default=0.10, help="atração à pose do dono, por rad")
    ap.add_argument("--com-piso", type=float, default=0.03,
                    help="m: CoM a esta distância ou mais da borda das solas, no lugar de 'no meio dos pés'")
    ap.add_argument("--peso-mao", type=float, default=3.0, help="peso do resíduo das palmas")
    ap.add_argument("--peso-com", type=float, default=3.0, help="peso da cerca do CoM (com --com-piso)")
    ap.add_argument("--tol-mao", type=float, default=0.01, help="aceite da mão, m (dono 08/10: 1 cm)")
    ap.add_argument("--margem", type=float, default=0.10, help="0,10 = 90%% do curso, regra da IK da v3")
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--saida", type=Path, default=AQUI / "agachado")
    args = ap.parse_args()
    F.MARGEM = args.margem
    global COM_PISO, PESO_MAO, PESO_COM
    COM_PISO, PESO_MAO, PESO_COM = (args.com_piso if args.com_piso > 0 else None), args.peso_mao, args.peso_com
    out = args.saida.expanduser(); out.mkdir(parents=True, exist_ok=True)

    c = G.Cena(F.CENA)
    E.prepara(c)
    gl = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")
    robo = [g for g in range(c.m.ngeom) if c.m.geom_contype[g]
            and (mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("robot/")]
    c.pares = np.vstack([c.pares, np.array([(g, gl) for g in robo])])
    c.folga_par = np.concatenate([c.folga_par, np.full(len(robo), F.FOLGA_LAJE)])
    lim = np.array([abs(c.m.actuator_forcerange[np.flatnonzero(c.m.actuator_trnid[:, 0] == j + 1)[0], 1])
                    for j in range(29)])
    P = np.load(args.pose.expanduser())
    assert P["hash_mjcf"] == c.hash, "pose de outro MJCF"
    q_ref = np.array(P["qpos"][0], dtype=float)
    cx = float(P["caixa_xyz"][0][0])
    BX = G.AVANCO_LAJE - G.RECUO_BORDA            # x da caixa na cena da v3 (a laje fica 0,15 à frente)
    q_seed = q_ref.copy(); q_seed[0] += BX - cx
    c.q_rep = q_ref[7:36].copy()                  # o termo de repouso aponta para a pose do dono
    c.d.qpos[:] = q_ref; mujoco.mj_kinematics(c.m, c.d)
    rumo0 = E.rumo(c, "left_foot")
    D = np.load(args.de_pe.expanduser()); M = np.load(args.lajes.expanduser())
    assert D["hash_mjcf"] == c.hash and M["hash_mjcf"] == c.hash, "tabela de outro MJCF"
    larg_dono = float(abs(c.d.site_xpos[c.sid["left_foot"]][1] - c.d.site_xpos[c.sid["right_foot"]][1]))
    c.d.qpos[:] = D["qpos"][1]; mujoco.mj_kinematics(c.m, c.d)
    larg_pe = float(abs(c.d.site_xpos[c.sid["left_foot"]][1] - c.d.site_xpos[c.sid["right_foot"]][1]))
    c.d.qpos[:] = q_ref; mujoco.mj_kinematics(c.m, c.d)
    # os pés NÃO copiam a pose do dono (dono, 08/10: "o pé não precisa ficar exato"): largura numa
    # faixa e ponta livre; a semelhança vem do alvo de postura
    G.LARGURA_MIN, G.LARGURA_MAX = 0.20, args.largura_max
    th = np.radians(args.theta)
    q_dono = q_ref[7:36].copy()
    linhas = [f"pose do dono {args.pose.name}: pés a {larg_dono:.2f} m, ponta {np.degrees(rumo0):+.0f}° (pés livres: "
              f"largura {'livre até ' + format(args.largura_max, '.2f') if args.largura_livre else f'presa à mistura {larg_dono:.2f} -> {larg_pe:.2f} m ± 3 cm'}, ponta livre), caixa "
              f"a {cx:.2f} m dos pés; dedos {args.theta:+.0f}°; margem {args.margem:.2f}; cintura <= 25°; "
              f"giro da coxa livre; peso da pose {args.peso_pose}",
              "apoio caixa massa | veredito | res pé/mão [cm], sola [°] | pelve z, incl | tronco | joelho hip_pitch [°] | "
              "CoM-borda [cm] | folga mín [mm] (par) | maior desvio do alvo de postura | "
              "torque % joelho hip_p tornoz cintura ombro_p cotovelo | perto do batente"]
    print(linhas[0]); print(linhas[1])
    c.cenario(-5.0, float(P["meia"]))
    m0 = mede(c, q_ref, np.array(P["caixa_xyz"][0]), args.massas[0], q_ref, lim)
    linhas.append(f"DONO chão {2*float(P['meia'])*100:.0f}cm {args.massas[0]:.1f}kg | como salva | — | "
                  f"{m0['pelve']:.2f}, {m0['pelve_i']:.0f}° | {m0['tronco']:.0f}° | {m0['joelho']:.0f} {m0['hip']:.0f} | "
                  f"{m0['apoio']*100:+.1f} | {m0['folga']*1000:.0f} ({m0['par']}) | — | {m0['torque']} | {m0['alto']}")
    print(linhas[-1], flush=True)
    regs, fotos = [], [(q_ref, float(P["meia"]), -5.0)]
    for topo in args.alturas:
      for aresta in args.arestas:
        a = aresta / 2
        laje = topo if topo > 0 else -5.0          # chão: a laje fica enterrada
        c.cenario(laje, a)
        caixa = np.array([BX, 0.0, max(topo, 0.0) + a])
        # ALVO DE POSTURA: mistura da pose do dono (caixa no chão) com a pose de pé aprovada
        # (caixa a 0,90 m), com peso pela altura do centro da caixa
        t = float(np.clip((caixa[2] - a) / (0.90 - a), 0.0, 1.0))
        q_pe = np.array(D["qpos"][int(np.argmin(np.abs(D["meia"] - a)))], dtype=float)
        c.q_rep = (1.0 - t) * q_dono + t * q_pe[7:36]
        if not args.largura_livre:
            # ⚠ A LARGURA DOS PÉS É VARIÁVEL LIVRE DA IK (medido 08/10): solta, ela desliza até o
            # teto (0,50 m nas lajes de 0,30 a 0,55) sem mudar pelve, tronco, torque nem folga.
            # Presa à mistura de postura ela segue o dono no chão (0,43) e a pose de pé no alto (0,23).
            larg = (1.0 - t) * larg_dono + t * larg_pe
            G.LARGURA_MIN, G.LARGURA_MAX = larg - 0.03, larg + 0.03
        # SEMENTE: no chão a pose do dono; na laje a pega aprovada da v3 (mão acima do tampo)
        if topo > 0:
            k = int(np.argmin(np.abs(M["topo"] - topo) + np.abs(M["meia"] - a)))
            q_seed = np.array(M["qpos"][k], dtype=float); q_seed[0] += BX - float(M["caixa_xyz"][k][0])
        else:
            q_seed = q_ref.copy(); q_seed[0] += BX - cx
        for massa in args.massas:
            t0 = time.time()
            G.MASSA_CAIXA = massa
            melhor = None
            for s in range(args.sementes):
                r = resolve(c, q_seed, caixa, a, th, rumo0, args.peso_pose, s, args.iters)
                if melhor is None or max(r["e_pe"], r["e_mao"]) < max(melhor["e_pe"], melhor["e_mao"]):
                    melhor = r
            r = melhor
            q_alvo = q_ref.copy(); q_alvo[7:36] = c.q_rep
            m = mede(c, r["qpos"], caixa, massa, q_alvo, lim)
            ok = (r["e_pe"] <= F.TOL_PE and r["sola"] <= SOLA_MAX and r["e_mao"] <= args.tol_mao and m["folga"] > -0.002
                  and m["frac"] <= 1.0 - args.margem + 0.005 and m["apoio"] > 0.0)
            linhas.append(f"{'chão' if topo <= 0 else f'{topo:.2f}'} {aresta*100:.0f}cm {massa:.1f}kg t={t:.2f} | {'ok' if ok else 'REJEITADA'} | "
                          f"{r['e_pe']*100:.1f}/{r['e_mao']*100:.1f} sola {r['sola']:.1f}° | {m['pelve']:.2f}, {m['pelve_i']:.0f}° | "
                          f"{m['tronco']:.0f}° | {m['joelho']:.0f} {m['hip']:.0f} | {m['apoio']*100:+.1f} | "
                          f"{m['folga']*1000:.0f} ({m['par']}) | {m['desvio']} | {m['torque']} | {m['alto']} | pé: {r['pe_pior']}  "
                          f"[{time.time()-t0:.0f}s]")
            print(linhas[-1], flush=True)
            regs.append(dict(topo=topo, nas_maos=False, meia=a, massa=massa, aceita=ok, residuo=r["e_mao"], residuo_pe=r["e_pe"],
                             pelve_z=m["pelve"], pelve_incl=m["pelve_i"], tronco_incl=m["tronco"],
                             folga_min=m["folga"], frac_max=m["frac"], caixa_xyz=caixa, qpos=r["qpos"]))
            if massa == args.massas[0] and aresta == args.arestas[0]:
                fotos.append((r["qpos"], a, laje))
    # DE PÉ com a caixa a 0,90 m (aprovada 05/10), uma linha por tamanho, com a caixa nas mãos
    for aresta in args.arestas:
        a = aresta / 2
        k = int(np.argmin(np.abs(D["meia"] - a)))
        q = np.array(D["qpos"][k], dtype=float)
        c.cenario(-5.0, float(D["meia"][k]))
        caixa = q[c.adr_caixa:c.adr_caixa + 3].copy()
        m = mede(c, q, caixa, args.massas[0], q, lim)
        linhas.append(f"de pé {2*float(D['meia'][k])*100:.0f}cm {args.massas[0]:.1f}kg | aprovada 05/10 | "
                      f"{float(D['residuo_pe'][k])*100:.1f}/{float(D['residuo'][k])*100:.1f} | {m['pelve']:.2f}, "
                      f"{m['pelve_i']:.0f}° | {m['tronco']:.0f}° | {m['joelho']:.0f} {m['hip']:.0f} | {m['apoio']*100:+.1f} | "
                      f"{m['folga']*1000:.0f} ({m['par']}) | — | {m['torque']} | {m['alto']}")
        print(linhas[-1], flush=True)
        regs.append(dict(topo=-1.0, nas_maos=True, meia=float(D["meia"][k]), massa=args.massas[0], aceita=True,
                         residuo=float(D["residuo"][k]), residuo_pe=float(D["residuo_pe"][k]),
                         pelve_z=m["pelve"], pelve_incl=m["pelve_i"], tronco_incl=m["tronco"],
                         folga_min=m["folga"], frac_max=m["frac"], caixa_xyz=caixa, qpos=q))
        if aresta == args.arestas[0]:
            fotos.append((q, float(D["meia"][k]), -5.0))
    (out / f"{args.nome}.txt").write_text("\n".join(linhas) + "\n")
    np.savez(out / f"{args.nome}.npz", hash_mjcf=c.hash, theta_mao=args.theta, pose_dono=q_ref,
             **{k: np.array([g[k] for g in regs]) for k in regs[0]})
    H, W = 460, 540
    rend = mujoco.Renderer(c.m, height=H, width=W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[0] = opt.geomgroup[2] = 1
    tiras = []
    for qpos, a, laje in fotos:
        c.cenario(laje, a)
        imgs = []
        for az, el in ((90.0, -5.0), (180.0, -10.0), (145.0, -25.0)):
            cam.distance, cam.azimuth, cam.elevation = 1.6, az, el
            c.d.qpos[:] = qpos; mujoco.mj_forward(c.m, c.d)
            cam.lookat[:] = (qpos[0] + 0.10, 0.0, 0.40)
            rend.update_scene(c.d, camera=cam, scene_option=opt)
            imgs.append(rend.render().copy())
        sep = np.full((H, 8, 3), 255, np.uint8)
        tiras.append(np.concatenate([imgs[0], sep, imgs[1], sep, imgs[2]], axis=1))
    rend.close()
    grade = tiras[0]
    for t in tiras[1:]:
        grade = np.concatenate([grade, np.full((8, grade.shape[1], 3), 255, np.uint8), t], axis=0)
    _grava_png(out / f"{args.nome}.png", grade)
    print(f"-> {out}/{args.nome}.txt, .npz, .png")


if __name__ == "__main__":
    main()
