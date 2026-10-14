"""IK com o ângulo da mão FIXO em relação à caixa, da pega ao transporte.

Parte das poses aprovadas (`ref_v3_manipulacao.npz` e `ref_v3_de_pe.npz`): mesma caixa,
mesma laje, mesmo recuo da raiz. Acrescenta UMA tarefa: a inclinação dos dedos (eixo x do
`wrist_yaw_link`) igual a THETA nas duas mãos. A caixa nasce sem giro, então o frame dela é
o do mundo.

    python g1_limpo/ik/ik_mao_fixa.py --theta -35 --saida pasta
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

# ⚠ A RAIZ DO REPO NO `sys.path`, o idioma do `ve_botar.py`: o arquivo roda como
# `python g1_limpo/ik/<nome>.py` e como `python -m g1_limpo.ik.<nome>`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
AQUI = Path(__file__).resolve().parent
import g1_limpo.ik.gera_botar as G                      # noqa: E402
from g1_limpo.ik import gera_de_pe as GP                 # noqa: E402
from g1_limpo.ik.ve_botar import _grava_png              # noqa: E402

W_MAO = 0.10            # rad -> m: 0,1 rad de erro custa 1 cm
MARGEM = 0.15           # a IK fica a 85% do meio-curso: dentro do limite mole 0,9 da Unitree
CINTURA = np.radians(25.0)
FOLGA_LAJE = 0.010
CENA = Path("~/g1_pilota/").expanduser()


def elev_dedos(c, lado):
    R = c.d.xmat[c.bid_punho[lado]].reshape(3, 3)
    return float(np.arcsin(np.clip(R[2, 0], -1, 1)))


def limites(c):
    folga = MARGEM * (c.hi - c.lo) / 2.0
    lo, hi = c.lo + folga, c.hi - folga
    nomes = [mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, 30)]
    for n in ("robot/waist_pitch_joint", "robot/waist_roll_joint"):
        i = nomes.index(n)
        lo[i], hi[i] = max(lo[i], -CINTURA), min(hi[i], CINTURA)
    return lo, hi


# ---------------------------------------------------------------- pega (laje)
def poe_pega(c, st, caixa, raiz_x):
    c.poe(st[:29], st[29], st[30], st[31], caixa)
    c.d.qpos[0] = raiz_x
    mujoco.mj_kinematics(c.m, c.d)
    mujoco.mj_comPos(c.m, c.d)


PELVE_ALVO = None      # m: altura pedida da pelve na pega (agachamento); None = livre
W_PELVE = 0.5


def res_pega(c, caixa, a, theta, ativos):
    e = list(G.residuo(c, caixa, a, G.PESO_REPOUSO, ativos))
    e += [W_MAO * (elev_dedos(c, l) - theta) for l in ("left", "right")]
    if PELVE_ALVO is not None:
        e += [W_PELVE * (float(c.d.xpos[c.bid_pelve][2]) - PELVE_ALVO)]
    return np.array(e)


RAIZ_LIVRE = 0.0       # m: quanto a raiz pode andar em x em volta da pose aprovada


def resolve_pega(c, k, ref, theta, iters):
    topo, a = float(ref["topo"][k]), float(ref["meia"][k])
    caixa = np.array(ref["caixa_xyz"][k]); q = ref["qpos"][k]; raiz0 = float(q[0])
    c.cenario(topo, a)
    from g1_limpo.ik.ik_cintura25 import euler_xy
    r, p = euler_xy(q[3:7])
    st = np.concatenate([q[7:36], [q[2]], [r, p], [raiz0]])
    nv = 33 if RAIZ_LIVRE > 0 else 32
    lo, hi = limites(c)
    st[:29] = np.clip(st[:29], lo, hi)
    for _ in range(iters):
        poe_pega(c, st, caixa, st[32])
        ativos = np.flatnonzero(c.folgas() < c.folga_par + G.BANDA_ATIVA)
        f0 = res_pega(c, caixa, a, theta, ativos)
        J = np.zeros((len(f0), nv))
        for j in range(nv):
            s = st.copy(); s[j] += 1e-5
            poe_pega(c, s, caixa, s[32])
            J[:, j] = (res_pega(c, caixa, a, theta, ativos) - f0) / 1e-5
        passo = J.T @ np.linalg.solve(J @ J.T + 2e-2 * np.eye(len(f0)), -f0)
        st[:nv] = st[:nv] + np.clip(passo, -0.15, 0.15)
        st[:29] = np.clip(st[:29], lo, hi)
        st[29] = np.clip(st[29], 0.25, 1.05)
        st[32] = np.clip(st[32], raiz0 - RAIZ_LIVRE, raiz0 + RAIZ_LIVRE)
    poe_pega(c, st, caixa, st[32])
    f = G.residuo(c, caixa, a)
    return dict(e_pe=float(np.abs(f[:c.n_pe]).max()), e_mao=float(np.abs(f[c.n_pe:c.n_pe + 6]).max()),
                qpos=c.d.qpos.copy(), topo=topo, meia=a, raiz_dx=float(st[32] - raiz0))


# ---------------------------------------------------------------- de pé com caixa
class PeMao(GP.Pe):
    theta = 0.0

    def residuo(self, st, h, ativos):
        e, n_pe, n_mao, n_tar = super().residuo(st, h, ativos)
        extra = [W_MAO * (elev_dedos(self.c, l) - self.theta) for l in ("left", "right")]
        # a tarefa da mão entra no bloco das TAREFAS (antes do secundário)
        e = np.concatenate([e[:n_tar], extra, e[n_tar:]])
        return e, n_pe, n_mao, n_tar + 2


def resolve_de_pe(theta, h, sementes, iters, semente_q, meia=0.10):
    p = PeMao(CENA, 3.0, meia)
    p.theta = theta
    p.c.bid_punho = {l: mujoco.mj_name2id(p.c.m, mujoco.mjtObj.mjOBJ_BODY, f"robot/{l}_wrist_yaw_link")
                     for l in ("left", "right")}
    lo, hi = limites(p.c)
    p.c.lo_ik, p.c.hi_ik = lo, hi
    # semente extra: a pose aprovada de pé com caixa
    q_rep0 = p.c.q_rep.copy()
    p.c.q_rep = semente_q[7:36].copy()
    st, e_pe, e_mao = p.resolve(h, sementes, iters, margem=MARGEM)
    p.c.q_rep = q_rep0
    p.c.poe(st[:29], st[29], st[30], st[31], p.caixa(st, h))
    fol = float(p.c.folgas().min())
    return p, dict(e_pe=e_pe, e_mao=e_mao, qpos=p.c.d.qpos.copy(), topo=5.0, meia=meia, folga=fol)


# ---------------------------------------------------------------- medidas e fotos
def mede(c, qpos):
    c.d.qpos[:] = qpos; mujoco.mj_kinematics(c.m, c.d)
    q = qpos[7:36]
    meio, meia_faixa = (c.hi + c.lo) / 2, (c.hi - c.lo) / 2
    frac = np.abs(q - meio) / meia_faixa
    resto = np.degrees(np.minimum(q - c.lo, c.hi - q))
    return dict(frac=frac, resto=resto, mao=[np.degrees(elev_dedos(c, l)) for l in ("left", "right")],
                pelve=float(c.d.xpos[c.bid_pelve][2]),
                tronco=float(np.degrees(G.inclinacao(c, c.bid_torso))),
                pelve_i=float(np.degrees(G.inclinacao(c, c.bid_pelve))))


def foto(c, paineis_in, azim, nome, W=640, H=560):
    """`paineis_in`: lista de (qpos, topo_da_laje, meia). Um painel por pose, lado a lado."""
    r = mujoco.Renderer(c.m, height=H, width=W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 2.0, azim, -12.0
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[0] = opt.geomgroup[2] = 1
    paineis = []
    for q, topo, a in paineis_in:
        c.cenario(topo, a)
        c.d.qpos[:] = q; mujoco.mj_forward(c.m, c.d)
        cam.lookat[:] = (0.15, 0.0, 0.60)
        r.update_scene(c.d, camera=cam, scene_option=opt)
        paineis.append(r.render().copy())
    r.close()
    sep = np.full((H, 8, 3), 255, np.uint8)
    out = paineis[0]
    for pn in paineis[1:]:
        out = np.concatenate([out, sep, pn], axis=1)
    _grava_png(nome, out)


def cadeia(args, c, M, D, nomes, out):
    """Para cada pega aprovada: a mão fica no ângulo DELA, e a IK de pé com caixa segue esse
    ângulo. Foto: pega aprovada | de pé com a mesma mão."""
    linhas = ["pega aprovada -> de pé com caixa a 0,90 m, mão no MESMO ângulo da pega "
              f"(margem da IK {MARGEM:.2f} do meio-curso)",
              "laje | mão na pega E/D [°] | de pé: res_pe res_mao, mão E/D [°] | "
              "juntas de pé acima de 80% do meio-curso (fração; graus até o batente)"]
    for k in args.poses:
        t0 = time.time()
        if args.fixo:
            # a pega TAMBÉM com a mão no ângulo pedido, a partir da pose aprovada
            c.pes_de(AQUI / "ref_v3_manipulacao.npz", k)
            rp = resolve_pega(c, k, M, np.radians(args.theta), args.iters)
            c.pes_alvo, c.n_pe = None, 9
            q_pega = rp["qpos"]; res_pega = f"pega res_pe {rp['e_pe']:.4f} res_mao {rp['e_mao']:.4f}"
        else:
            q_pega, res_pega = M["qpos"][k], "pega aprovada"
        mm_p = mede(c, q_pega)
        th = np.radians(0.5 * (mm_p["mao"][0] + mm_p["mao"][1]))
        p, r = resolve_de_pe(th, 0.90, args.sementes, args.de_pe_iters, D["qpos"][2])
        mm = mede(c, r["qpos"])
        alto = [f"{nomes[i]} {mm['frac'][i]:.2f} ({mm['resto'][i]:.0f}°)"
                for i in np.argsort(-mm["frac"]) if mm["frac"][i] > 0.80]
        topo = float(M["topo"][k])
        alto_p = [f"{nomes[i]} {mm_p['frac'][i]:.2f} ({mm_p['resto'][i]:.0f}°)"
                  for i in np.argsort(-mm_p["frac"]) if mm_p["frac"][i] > 0.80]
        linhas.append(f"{topo:.2f} | {res_pega}; tronco {mm_p['tronco']:.0f}°, pelve {mm_p['pelve_i']:.0f}°; "
                      f"juntas: {', '.join(alto_p) if alto_p else 'nenhuma'}")
        linhas.append(f"{topo:.2f} | {mm_p['mao'][0]:+.0f}/{mm_p['mao'][1]:+.0f} | "
                      f"{r['e_pe']:.4f} {r['e_mao']:.4f}, {mm['mao'][0]:+.0f}/{mm['mao'][1]:+.0f} | "
                      + (", ".join(alto) if alto else "nenhuma"))
        print(linhas[-1], f"({time.time()-t0:.0f}s)", flush=True)
        if not args.sem_fotos:
            for azim, vista in ((90.0, "lado"), (150.0, "frente")):
                foto(c, [(q_pega, topo, float(M["meia"][k])), (r["qpos"], 5.0, 0.10)], azim,
                     out / f"cadeia{args.tag}_{topo:.2f}_{vista}.png")
    txt = "\n".join(linhas)
    (out / f"cadeia{args.tag}_limites.txt").write_text(txt + "\n")
    print(txt)


def angulos(args, c, M, D, nomes, out):
    """Uma laje só (`--poses k`), vários ângulos da mão. Para cada ângulo: a pega com a mão
    fixa nele e o robô de pé com a caixa a 0,90 m com a MESMA mão. Grade: linhas = ângulos,
    colunas = pega | de pé."""
    k = args.poses[0]
    topo, a = float(M["topo"][k]), float(M["meia"][k])
    linhas = [f"laje {topo:.2f} m, caixa {2*a*100:.0f} cm, margem da IK {MARGEM:.2f} do meio-curso; "
              "mão = inclinação dos dedos contra o solo (0 = horizontal, -90 = para baixo)"]
    paineis = []
    for th_g in args.angulos:
        c.pes_de(AQUI / "ref_v3_manipulacao.npz", k)
        rp = resolve_pega(c, k, M, np.radians(th_g), args.iters)
        c.pes_alvo, c.n_pe = None, 9
        mp = mede(c, rp["qpos"])
        p, rd = resolve_de_pe(np.radians(th_g), 0.90, args.sementes, args.de_pe_iters, D["qpos"][2])
        md = mede(c, rd["qpos"])
        def alto(mm):
            v = [f"{nomes[i]} {mm['resto'][i]:.0f}°" for i in np.argsort(-mm["frac"]) if mm["frac"][i] > 0.80]
            return ", ".join(v) if v else "nenhuma"
        ok_p = rp["e_pe"] < 0.01 and rp["e_mao"] < 0.02
        ok_d = rd["e_pe"] < 0.01 and rd["e_mao"] < 0.02
        linhas += [f"mão {th_g:+.0f}°",
                   f"  pega : {'ok' if ok_p else 'NÃO FECHA'} (res_pe {rp['e_pe']:.4f}, res_mao {rp['e_mao']:.4f}); "
                   f"raiz {rp['raiz_dx']*100:+.0f} cm; mão {mp['mao'][0]:+.0f}/{mp['mao'][1]:+.0f}; tronco {mp['tronco']:.0f}°; pelve z {mp['pelve']:.2f}, "
                   f"incl {mp['pelve_i']:.0f}°; perto do batente: {alto(mp)}",
                   f"  de pé: {'ok' if ok_d else 'NÃO FECHA'} (res_pe {rd['e_pe']:.4f}, res_mao {rd['e_mao']:.4f}); "
                   f"mão {md['mao'][0]:+.0f}/{md['mao'][1]:+.0f}; perto do batente: {alto(md)}"]
        print("\n".join(linhas[-3:]), flush=True)
        paineis.append([(rp["qpos"], topo, a), (rd["qpos"], 5.0, 0.10)])
    (out / f"angulos_{topo:.2f}.txt").write_text("\n".join(linhas) + "\n")
    if not args.sem_fotos:
        for azim, vista in ((90.0, "lado"), (150.0, "frente")):
            tiras = []
            for i, par in enumerate(paineis):
                nome = out / f"_tmp_{i}.png"
                tiras.append(foto_rgb(c, par, azim))
            sep = np.full((8, tiras[0].shape[1], 3), 255, np.uint8)
            grade = tiras[0]
            for t in tiras[1:]:
                grade = np.concatenate([grade, sep, t], axis=0)
            _grava_png(out / f"angulos_{topo:.2f}_{vista}.png", grade)
        print(f"fotos em {out}")


def foto_rgb(c, paineis_in, azim, W=560, H=420):
    r = mujoco.Renderer(c.m, height=H, width=W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 2.3, azim, -10.0
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[0] = opt.geomgroup[2] = 1
    imgs = []
    for q, topo, a in paineis_in:
        c.cenario(topo, a)
        c.d.qpos[:] = q; mujoco.mj_forward(c.m, c.d)
        cam.lookat[:] = (0.15, 0.0, 0.62)
        r.update_scene(c.d, camera=cam, scene_option=opt)
        imgs.append(r.render().copy())
    r.close()
    sep = np.full((H, 8, 3), 255, np.uint8)
    return np.concatenate([imgs[0], sep, imgs[1]], axis=1)


def agacha(args, c, M, nomes, out):
    """Uma laje só (`--poses k`), mão fixa em `--theta`, e a pelve pedida em cada altura de
    `--pelve-z`. Todas as outras restrições ficam. Grade: linhas = alturas, colunas = lado | frente."""
    global PELVE_ALVO
    k = args.poses[0]
    topo, a = float(M["topo"][k]), float(M["meia"][k])
    th = np.radians(args.theta)
    linhas = [f"laje {topo:.2f} m, caixa {2*a*100:.0f} cm, mão {args.theta:+.0f}°, margem {MARGEM:.2f}, "
              f"raiz livre ±{RAIZ_LIVRE*100:.0f} cm, cintura ≤ 25°, folga com a laje {FOLGA_LAJE*1000:.0f} mm"]
    linhas.append("pelve pedida | veredito | res_pe res_mao | pelve z, incl | tronco | joelho, hip_pitch | "
                  "CoM-pés | folga mín | raiz | perto do batente")
    poses = []
    for z in args.pelve_z:
        PELVE_ALVO = None if z <= 0 else z
        c.pes_de(AQUI / "ref_v3_manipulacao.npz", k)
        rp = resolve_pega(c, k, M, th, args.iters)
        c.pes_alvo, c.n_pe = None, 9
        mp = mede(c, rp["qpos"])
        c.d.qpos[:] = rp["qpos"]; mujoco.mj_kinematics(c.m, c.d); mujoco.mj_comPos(c.m, c.d)
        _, _, com, _ = G.grandezas(c, a, np.array(M["caixa_xyz"][k]))
        fol = float(c.folgas().min())
        q = rp["qpos"][7:36]
        joelho = q[nomes.index("left_knee")]; hip = q[nomes.index("left_hip_pitch")]
        ok = rp["e_pe"] < 0.01 and rp["e_mao"] < 0.02 and fol > -0.002
        alto = [f"{nomes[i]} {mp['resto'][i]:.0f}°" for i in np.argsort(-mp["frac"]) if mp["frac"][i] > 0.80]
        rot = "livre" if z <= 0 else f"{z:.2f}"
        linhas.append(f"{rot} | {'ok' if ok else 'NÃO FECHA'} | {rp['e_pe']:.4f} {rp['e_mao']:.4f} | "
                      f"{mp['pelve']:.2f}, {mp['pelve_i']:.0f}° | {mp['tronco']:.0f}° | "
                      f"{joelho:.2f}, {hip:.2f} | {com*100:.1f} cm | {fol*1000:.0f} mm | "
                      f"{rp['raiz_dx']*100:+.0f} cm | {', '.join(alto) if alto else 'nenhuma'}")
        print(linhas[-1], flush=True)
        poses.append(rp["qpos"])
    PELVE_ALVO = None
    (out / f"agacha_{topo:.2f}.txt").write_text("\n".join(linhas) + "\n")
    if not args.sem_fotos:
        tiras = []
        for q in poses:
            r = mujoco.Renderer(c.m, height=420, width=560)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[0] = opt.geomgroup[2] = 1
            imgs = []
            for azim in (90.0, 150.0):
                cam.distance, cam.azimuth, cam.elevation = 2.0, azim, -10.0
                c.cenario(topo, a); c.d.qpos[:] = q; mujoco.mj_forward(c.m, c.d)
                cam.lookat[:] = (0.15, 0.0, 0.45)
                r.update_scene(c.d, camera=cam, scene_option=opt)
                imgs.append(r.render().copy())
            r.close()
            tiras.append(np.concatenate([imgs[0], np.full((420, 8, 3), 255, np.uint8), imgs[1]], axis=1))
        grade = tiras[0]
        for t in tiras[1:]:
            grade = np.concatenate([grade, np.full((8, grade.shape[1], 3), 255, np.uint8), t], axis=0)
        _grava_png(out / f"agacha_{topo:.2f}.png", grade)
        print(f"foto em {out / f'agacha_{topo:.2f}.png'}")


TOL_PE, TOL_MAO = 0.025, 0.025   # aceite da tabela (dono, 05/10: "pés a 2 cm não é problema")


def tabela(args, c, M, D, nomes, out):
    """As 15 pegas da `ref_v3_manipulacao` com a mão fixa em `--theta`, e UMA pose de pé com
    caixa a 0,90 m com a mesma mão. Salva no esquema das tabelas aprovadas. Fotos: uma linha
    por laje (caixa de meia 0,10), colunas pega | de pé, de lado e de frente."""
    th = np.radians(args.theta)
    p, rd = resolve_de_pe(th, 0.90, args.sementes, args.de_pe_iters, D["qpos"][2])
    md = mede(c, rd["qpos"])
    def alto(mm):
        v = [f"{nomes[i]} {mm['resto'][i]:.0f}°" for i in np.argsort(-mm["frac"]) if mm["frac"][i] > 0.80]
        return ", ".join(v) if v else "nenhuma"
    linhas = [f"mão {args.theta:+.0f}° (dedos contra o solo), margem {MARGEM:.2f} do meio-curso, raiz livre "
              f"±{RAIZ_LIVRE*100:.0f} cm, aceite pés e mãos ≤ {TOL_PE*100:.1f} cm",
              f"DE PÉ 0,90 m: res_pe {rd['e_pe']*100:.2f} cm, res_mao {rd['e_mao']*100:.2f} cm; mão "
              f"{md['mao'][0]:+.0f}/{md['mao'][1]:+.0f}; perto do batente: {alto(md)}",
              "laje caixa | veredito | res_pe res_mao [cm] | raiz | mão | pelve z, incl | tronco | perto do batente"]
    regs = []
    for k in range(len(M["topo"])):
        c.pes_de(AQUI / "ref_v3_manipulacao.npz", k)
        rp = resolve_pega(c, k, M, th, args.iters)
        c.pes_alvo, c.n_pe = None, 9
        mp = mede(c, rp["qpos"])
        c.d.qpos[:] = rp["qpos"]; mujoco.mj_kinematics(c.m, c.d); mujoco.mj_comPos(c.m, c.d)
        fol = float(c.folgas().min())
        ok = rp["e_pe"] <= TOL_PE and rp["e_mao"] <= TOL_MAO and fol > -0.002 and mp["frac"].max() <= 0.905
        linhas.append(f"{rp['topo']:.2f} {2*rp['meia']*100:.0f}cm | {'ok' if ok else 'REJEITADA'} | "
                      f"{rp['e_pe']*100:.2f} {rp['e_mao']*100:.2f} | {rp['raiz_dx']*100:+.0f} cm | "
                      f"{mp['mao'][0]:+.0f}/{mp['mao'][1]:+.0f} | {mp['pelve']:.2f}, {mp['pelve_i']:.0f}° | "
                      f"{mp['tronco']:.0f}° | {alto(mp)}")
        print(linhas[-1], flush=True)
        regs.append(dict(topo=rp["topo"], meia=rp["meia"], residuo=rp["e_mao"], residuo_pe=rp["e_pe"],
                         pelve_z=mp["pelve"], pelve_incl=mp["pelve_i"], tronco_incl=mp["tronco"],
                         folga_min=fol, frac_max=float(mp["frac"].max()), aceita=ok, qpos=rp["qpos"],
                         caixa_xyz=np.array(M["caixa_xyz"][k])))
    txt = "\n".join(linhas)
    print(txt.split("\n")[1])
    tag = f"theta{args.theta:+.0f}"
    (out / f"tabela_{tag}.txt").write_text(txt + "\n")
    np.savez(out / f"ref_v3_manipulacao_{tag}.npz", hash_mjcf=c.hash, theta_mao=args.theta,
             **{key: np.array([r[key] for r in regs]) for key in regs[0]})
    np.savez(out / f"ref_v3_de_pe_{tag}.npz", hash_mjcf=c.hash, theta_mao=args.theta, massa=np.array([3.0]),
             topo=np.array([0.80]), meia=np.array([0.10]), qpos=np.array([rd["qpos"]]),
             residuo=np.array([rd["e_mao"]]), residuo_pe=np.array([rd["e_pe"]]))
    if not args.sem_fotos:
        idx = [k for k in range(len(M["topo"])) if abs(float(M["meia"][k]) - 0.10) < 1e-6]
        for azim, vista in ((90.0, "lado"), (150.0, "frente")):
            tiras = [foto_rgb(c, [(regs[k]["qpos"], regs[k]["topo"], regs[k]["meia"]), (rd["qpos"], 5.0, 0.10)], azim)
                     for k in idx]
            grade = tiras[0]
            for t in tiras[1:]:
                grade = np.concatenate([grade, np.full((8, grade.shape[1], 3), 255, np.uint8), t], axis=0)
            _grava_png(out / f"tabela_{tag}_{vista}.png", grade)
        print(f"fotos em {out}")


def tamanhos(args, c, M, D, nomes, out):
    """Pose de pé com caixa (mão em `--theta`) para cada tamanho de caixa, ao lado das pegas do
    mesmo tamanho na tabela `ref_v3_manipulacao_theta<θ>.npz` já gerada. Uma grade por tamanho."""
    th = np.radians(args.theta)
    tag = f"theta{args.theta:+.0f}"
    T = np.load(out / f"ref_v3_manipulacao_{tag}.npz")
    def alto(mm):
        v = [f"{nomes[i]} {mm['resto'][i]:.0f}°" for i in np.argsort(-mm["frac"]) if mm["frac"][i] > 0.80]
        return ", ".join(v) if v else "nenhuma"
    linhas = [f"mão {args.theta:+.0f}°; de pé com caixa a 0,90 m por tamanho; pegas da tabela {tag}"]
    pes = {}
    for meia in (0.07, 0.10, 0.13):
        p, rd = resolve_de_pe(th, 0.90, args.sementes, args.de_pe_iters, D["qpos"][2], meia)
        md = mede(c, rd["qpos"])
        ok = rd["e_pe"] < 0.01 and rd["e_mao"] < 0.02 and rd["folga"] > -0.002
        q = rd["qpos"][7:36]
        linhas.append(f"DE PÉ caixa {2*meia*100:.0f} cm | {'ok' if ok else 'REJEITADA'} | res_pe {rd['e_pe']*100:.2f} "
                      f"res_mao {rd['e_mao']*100:.2f} cm | folga mín {rd['folga']*1000:.0f} mm | mão "
                      f"{md['mao'][0]:+.0f}/{md['mao'][1]:+.0f} | cotovelo {q[nomes.index('left_elbow')]:.2f} | "
                      f"ombro roll {q[nomes.index('left_shoulder_roll')]:.2f} | perto do batente: {alto(md)}")
        print(linhas[-1], flush=True)
        pes[meia] = rd
    for k in range(len(T["topo"])):
        mm = mede(c, T["qpos"][k])
        linhas.append(f"pega {T['topo'][k]:.2f} caixa {2*T['meia'][k]*100:.0f} cm | "
                      f"{'ok' if T['aceita'][k] else 'REJEITADA'} | tronco {T['tronco_incl'][k]:.0f}° | "
                      f"folga {T['folga_min'][k]*1000:.0f} mm | perto do batente: {alto(mm)}")
    (out / f"tamanhos_{tag}.txt").write_text("\n".join(linhas) + "\n")
    print("\n".join(linhas[4:]))
    if not args.sem_fotos:
        for meia, rd in pes.items():
            idx = [k for k in range(len(T["topo"])) if abs(float(T["meia"][k]) - meia) < 1e-6]
            for azim, vista in ((90.0, "lado"), (150.0, "frente")):
                tiras = [foto_rgb(c, [(T["qpos"][k], float(T["topo"][k]), meia), (rd["qpos"], 5.0, meia)], azim)
                         for k in idx]
                grade = tiras[0]
                for t in tiras[1:]:
                    grade = np.concatenate([grade, np.full((8, grade.shape[1], 3), 255, np.uint8), t], axis=0)
                _grava_png(out / f"tamanho_{2*meia*100:.0f}cm_{vista}.png", grade)
        np.savez(out / f"ref_v3_de_pe_{tag}_tamanhos.npz", hash_mjcf=c.hash, theta_mao=args.theta,
                 meia=np.array(list(pes)), qpos=np.array([r["qpos"] for r in pes.values()]),
                 residuo=np.array([r["e_mao"] for r in pes.values()]),
                 residuo_pe=np.array([r["e_pe"] for r in pes.values()]))
        print(f"fotos em {out}")


def main():
    global MARGEM, RAIZ_LIVRE
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.0, help="inclinação dos dedos, graus")
    ap.add_argument("--angulos", type=float, nargs="*", default=None,
                    help="varre ângulos da mão numa laje só (--poses k): pega | de pé")
    ap.add_argument("--cadeia", action="store_true",
                    help="de pé com a mão no ângulo de cada pega aprovada; foto pega | de pé")
    ap.add_argument("--pelve-z", type=float, nargs="*", default=None,
                    help="agachamento: alturas pedidas da pelve na pega (0 = livre), com --theta e --poses k")
    ap.add_argument("--tamanhos", action="store_true", help="pose de pé por tamanho de caixa, com as pegas da tabela")
    ap.add_argument("--tabela", action="store_true", help="as 15 pegas e a pose de pé com --theta")
    ap.add_argument("--raiz-livre", type=float, default=0.0, help="m que a raiz anda em x na pega")
    ap.add_argument("--fixo", action="store_true", help="com --cadeia: a pega também usa --theta")
    ap.add_argument("--tag", default="", help="sufixo dos arquivos do modo --cadeia")
    ap.add_argument("--poses", type=int, nargs="*", default=[1, 4, 7, 10, 13],
                    help="índices da ref_v3_manipulacao (meia 0,10: 1 4 7 10 13)")
    ap.add_argument("--iters", type=int, default=240)
    ap.add_argument("--de-pe-iters", type=int, default=400)
    ap.add_argument("--sementes", type=int, default=2)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--sem-fotos", action="store_true")
    ap.add_argument("--margem", type=float, default=0.15, help="fração do meio-curso fora do alcance da IK")
    args = ap.parse_args()
    MARGEM = args.margem
    RAIZ_LIVRE = args.raiz_livre
    th = np.radians(args.theta)
    out = Path(args.saida).expanduser(); out.mkdir(parents=True, exist_ok=True)

    M = np.load(AQUI / "ref_v3_manipulacao.npz"); D = np.load(AQUI / "ref_v3_de_pe.npz")
    c = G.Cena(CENA)
    assert M["hash_mjcf"] == c.hash
    gl = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")
    robo = [g for g in range(c.m.ngeom) if c.m.geom_contype[g]
            and (mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("robot/")]
    c.pares = np.vstack([c.pares, np.array([(g, gl) for g in robo])])
    c.folga_par = np.concatenate([c.folga_par, np.full(len(robo), FOLGA_LAJE)])
    c.bid_punho = {l: mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_BODY, f"robot/{l}_wrist_yaw_link")
                   for l in ("left", "right")}
    nomes = [mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_JOINT, j).split("/")[-1].replace("_joint", "")
             for j in range(1, 30)]

    if args.tamanhos:
        tamanhos(args, c, M, D, nomes, out)
        return
    if args.tabela:
        tabela(args, c, M, D, nomes, out)
        return
    if args.pelve_z:
        agacha(args, c, M, nomes, out)
        return
    if args.angulos:
        angulos(args, c, M, D, nomes, out)
        return
    if args.cadeia:
        cadeia(args, c, M, D, nomes, out)
        return
    res = []
    for k in args.poses:
        t0 = time.time()
        c.pes_de(AQUI / "ref_v3_manipulacao.npz", k)   # os pés da pose aprovada
        r = resolve_pega(c, k, M, th, args.iters)
        r.update(nome=f"pega_{r['topo']:.2f}", antes=M["qpos"][k]); res.append(r)
        c.pes_alvo, c.n_pe = None, 9
        print(f"{r['nome']}: res_pe {r['e_pe']:.4f} res_mao {r['e_mao']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    p, r = resolve_de_pe(th, 0.90, args.sementes, args.de_pe_iters, D["qpos"][2])
    r.update(nome="de_pe_0.90", antes=D["qpos"][2]); res.append(r)
    print(f"{r['nome']}: res_pe {r['e_pe']:.4f} res_mao {r['e_mao']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    linhas = [f"theta = {args.theta:.0f} graus (dedos; 0 = horizontal, negativo = para baixo)",
              "pose | res_pe res_mao | mão E/D [°] | pelve z | tronco [°] | pelve incl [°] | "
              "juntas acima de 80% do meio-curso (fração; graus até o batente)"]
    for r in res:
        for tag, q in (("ANTES", r["antes"]), ("NOVA ", r["qpos"])):
            mm = mede(c, q)
            alto = [f"{nomes[i]} {mm['frac'][i]:.2f} ({mm['resto'][i]:.0f}°)"
                    for i in np.argsort(-mm["frac"]) if mm["frac"][i] > 0.80]
            resid = f"{r['e_pe']:.4f} {r['e_mao']:.4f}" if tag.startswith("NOVA") else "   —      —   "
            linhas.append(f"{r['nome']} {tag} | {resid} | "
                          f"{mm['mao'][0]:+.0f}/{mm['mao'][1]:+.0f} | {mm['pelve']:.3f} | {mm['tronco']:.0f} | "
                          f"{mm['pelve_i']:.0f} | " + (", ".join(alto) if alto else "nenhuma"))
    txt = "\n".join(linhas)
    print(txt)
    (out / f"limites_theta{args.theta:+.0f}.txt").write_text(txt + "\n")
    np.savez(out / f"mao_fixa_theta{args.theta:+.0f}.npz", hash_mjcf=c.hash,
             nome=np.array([r["nome"] for r in res]), topo=np.array([r["topo"] for r in res]),
             meia=np.array([r["meia"] for r in res]), qpos=np.array([r["qpos"] for r in res]),
             antes=np.array([r["antes"] for r in res]))
    if not args.sem_fotos:
        for r in res:
            for azim, vista in ((90.0, "lado"), (150.0, "frente")):
                foto(c, [(r["antes"], r["topo"], r["meia"]), (r["qpos"], r["topo"], r["meia"])], azim,
                     out / f"{r['nome']}_{vista}.png")
        print(f"fotos em {out}")


if __name__ == "__main__":
    main()
