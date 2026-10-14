"""Monta uma pose do G1 à mão, com sliders no navegador, e mede o que importa nela.

    cd ~/Documents/g1_training
    .venv/bin/python g1_limpo/ik/monta_pose.py
    # abre http://localhost:8080

    --de ARQ.npz --i K   abre a pose K de um npz (tabelas da v3, ref_agachado, poses salvas). Com
                         mais de uma pose aparece o slider "pose da tabela", que troca de pose.
    --aresta 0.20        aresta inicial da caixa em m (padrão: a da pose aberta, ou 0,20)
    --massa 5            massa inicial da caixa em kg (padrão: a da pose aberta, ou 5)

Sliders em graus, com a faixa da tabela de juntas da spec da v3 (§6.5), que é a do `cena.mjb`.
"espelhar" ligado: os sliders da direita seguem os da esquerda (pitch, joelho e cotovelo
iguais; roll e yaw com o sinal trocado) e ficam travados.

A raiz não tem slider: os PÉS ficam no chão. A cada mudança o script acha a pelve (altura,
roll, pitch) que deixa as duas solas planas no chão e põe o meio dos pés na origem. Com as
pernas assimétricas uma sola não fecha, e o painel mostra quanto. Uma pose da IK que deixou o
pé fora do lugar aparece aqui com o pé no chão e a pelve recalculada.

Caixa: tamanho e massa por slider. Com "altura do apoio" > 0 a laje da v3 aparece sob a caixa,
com o centro 0,15 m à frente dela, como na cena do treino. A caixa é desenhada pelo viser, e não
pela malha do mjviser, porque a malha nasce com um tamanho só.

"salvar pose" grava em `g1_limpo/ik/poses_manuais/<nome>.npz`, com `qpos` (1, nq),
`hash_mjcf`, a caixa, o apoio e a massa: o mesmo esquema das tabelas.

⚠ Só cinemática (`mj_forward`, nunca `mj_step`): a pose não cai porque não há física. O
equilíbrio é o ESTÁTICO: CoM combinado (robô + caixa) dentro do polígono das solas.
"""
import argparse
import sys
import threading
from pathlib import Path

# ⚠ A RAIZ DO REPO NO `sys.path`, o idioma do `ve_botar.py`: o arquivo roda como
# `python g1_limpo/ik/<nome>.py` e como `python -m g1_limpo.ik.<nome>`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
AQUI = Path(__file__).resolve().parent

import mujoco                                            # noqa: E402
import numpy as np                                       # noqa: E402
import viser                                             # noqa: E402
from mjviser import ViserMujocoScene                     # noqa: E402

import g1_limpo.ik.gera_botar as G                      # noqa: E402
from g1_limpo.ik import ik_entre_pernas as E             # noqa: E402
from g1_limpo.ik import ik_mao_fixa as F                 # noqa: E402

# no lado direito, estas juntas são as da esquerda com o sinal trocado
TROCA = ("hip_roll", "hip_yaw", "ankle_roll", "shoulder_roll", "shoulder_yaw", "wrist_roll", "wrist_yaw")
PERNA = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
BRACO = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
CINTURA = ("waist_yaw", "waist_roll", "waist_pitch")
SAIDA = AQUI / "poses_manuais"
COR_CAIXA = (214, 132, 48)


def casco(P):
    """Casco convexo 2D, anti-horário (cadeia monótona)."""
    P = sorted(map(tuple, P))
    giro = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa: E731
    baixo, cima = [], []
    for p in P:
        while len(baixo) >= 2 and giro(baixo[-2], baixo[-1], p) <= 0:
            baixo.pop()
        baixo.append(p)
    for p in reversed(P):
        while len(cima) >= 2 and giro(cima[-2], cima[-1], p) <= 0:
            cima.pop()
        cima.append(p)
    return np.array(baixo[:-1] + cima[:-1])


def margem_apoio(c, p):
    """Distância do ponto `p` (xy) à borda do polígono das solas: positiva dentro, negativa fora."""
    pts = []
    for g in c.geom_pe:
        R = c.d.geom_xmat[g].reshape(3, 3); h = c.m.geom_size[g][1]
        pts += [(c.d.geom_xpos[g] + s * h * R[:, 2])[:2] for s in (-1, 1)]
    H = casco(np.array(pts))
    m = np.inf
    for a, b in zip(H, np.roll(H, -1, axis=0)):
        t = b - a; n = np.array([-t[1], t[0]]) / np.linalg.norm(t)    # normal para DENTRO
        m = min(m, float(np.dot(p - a, n)))
    return m


class Pose:
    def __init__(self, c, meia, massa, lim):
        self.c, self.massa, self.lim = c, massa, lim
        self.raiz = np.array([0.78, 0.0, 0.0])                       # pelve z, roll, pitch
        self.caixa_x, self.apoio, self.mostra, self.nas_maos = 0.10, 0.0, True, False
        self.tamanho(meia)
        self.caixa = np.array([5.0, 0.0, meia])

    def tamanho(self, meia):
        """Muda a meia aresta da caixa no modelo (folgas e CoM usam este tamanho)."""
        self.meia = meia
        self.c.cenario(-5.0, meia)

    def _sola(self):
        c, d = self.c, self.c.d
        e = []
        for s in ("left_foot", "right_foot"):
            R = d.site_xmat[c.sid[s]].reshape(3, 3)
            e += [d.site_xpos[c.sid[s]][2] - G.Z_SITE_ALVO, R[0, 2], R[1, 2]]
        return np.array(e)

    def monta(self, q):
        """Juntas `q` (29, rad) -> pose com as solas no chão, meio dos pés na origem, a caixa e a laje."""
        c, d = self.c, self.c.d
        longe = np.array([5.0, 0.0, self.meia])
        r = self.raiz.copy()
        for _ in range(12):                                          # Gauss-Newton em 3 variáveis
            c.poe(q, *r, longe); f0 = self._sola()
            J = np.zeros((6, 3))
            for k in range(3):
                s = r.copy(); s[k] += 1e-5
                c.poe(q, *s, longe); J[:, k] = (self._sola() - f0) / 1e-5
            r = r + np.linalg.lstsq(J, -f0, rcond=None)[0]
        c.poe(q, *r, longe)
        f = self._sola().reshape(2, 3)                    # por pé: altura, R[0,2], R[1,2]
        self.raiz = r
        self.sola_alt = float(np.abs(f[:, 0]).max())
        self.sola_giro = float(np.degrees(np.arcsin(np.clip(np.abs(f[:, 1:]).max(), 0, 1))))
        meio = 0.5 * (d.site_xpos[c.sid["left_foot"]] + d.site_xpos[c.sid["right_foot"]])
        d.qpos[0] -= meio[0]; d.qpos[1] -= meio[1]
        mujoco.mj_kinematics(c.m, d)
        if self.nas_maos:
            caixa = 0.5 * (d.site_xpos[c.sid["left_palm"]] + d.site_xpos[c.sid["right_palm"]])
        elif self.mostra:
            caixa = np.array([self.caixa_x, 0.0, self.apoio + self.meia])
        else:
            caixa = longe
        d.qpos[c.adr_caixa:c.adr_caixa + 3] = caixa
        d.qpos[c.adr_caixa + 3:c.adr_caixa + 7] = (1.0, 0.0, 0.0, 0.0)
        # a laje da v3 sob a caixa: centro 0,15 m à frente dela (gera_botar.RECUO_BORDA)
        if self.mostra and not self.nas_maos and self.apoio > 0:
            d.mocap_pos[c.id_mocap] = (self.caixa_x + G.RECUO_BORDA, 0.0, self.apoio - c.meia_laje)
        else:
            d.mocap_pos[c.id_mocap] = (5.0, 0.0, -5.0)
        mujoco.mj_forward(c.m, d)
        mujoco.mj_comPos(c.m, d)
        self.caixa = caixa.copy()

    def texto(self):
        c, d = self.c, self.c.d
        q = d.qpos[7:36].copy()
        com_caixa = self.mostra or self.nas_maos
        massa = self.massa if com_caixa else 0.0
        com = (c.m_robo * d.subtree_com[c.bid_pelve] + massa * self.caixa) / (c.m_robo + massa)
        apoio = margem_apoio(c, com[:2])
        lf, rf = d.site_xpos[c.sid["left_foot"]], d.site_xpos[c.sid["right_foot"]]
        rumo = [np.degrees(E.rumo(c, s)) for s in ("left_foot", "right_foot")]
        # folgas: corpo x corpo, corpo x caixa, corpo x chão ou laje (fora os pés no chão)
        fol = c.folgas()
        eh_caixa = c.pares[:, 1] == c.gid_caixa
        eh_chao = np.arange(len(fol)) >= c.n_par_corpo
        nome = lambda g: (mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "?").split("/")[-1].replace("_collision", "")  # noqa: E731

        def pior(mask):
            i = np.flatnonzero(mask)[np.argmin(fol[mask])]
            return f"{fol[i]*1000:.0f} mm ({nome(c.pares[i][0])} x {nome(c.pares[i][1])})"
        tau, _ = E.torques(c, self.caixa, massa)
        mujoco.mj_kinematics(c.m, d); mujoco.mj_comPos(c.m, d)
        pct = 100 * np.abs(tau) / self.lim
        tq = lambda j: max(pct[c.nomes.index(f"left_{j}")], pct[c.nomes.index(f"right_{j}")])  # noqa: E731
        frac = np.abs(q - (c.hi + c.lo) / 2) / ((c.hi - c.lo) / 2)
        resto = np.degrees(np.minimum(q - c.lo, c.hi - q))
        alto = [f"{c.nomes[i]} {resto[i]:.0f}°" for i in np.argsort(-frac) if frac[i] > 0.9]
        linhas = [
            f"**Pelve** {d.xpos[c.bid_pelve][2]:.3f} m · inclinação {np.degrees(G.inclinacao(c, c.bid_pelve)):.0f}° "
            f"(queda do treino: 70°) · **tronco** {np.degrees(G.inclinacao(c, c.bid_torso)):.0f}°",
            f"**Solas** fora do chão {self.sola_alt*100:.1f} cm · inclinadas {self.sola_giro:.1f}° · "
            f"pés a {abs(lf[1]-rf[1]):.2f} m · "
            f"ponta {rumo[0]:+.0f}° / {rumo[1]:+.0f}°",
            (f"**Equilíbrio** CoM dentro do apoio, margem {apoio*100:.1f} cm" if apoio >= 0 else
             f"**Equilíbrio: CoM FORA do apoio por {-apoio*100:.1f} cm** (cai)")
            + f" · caixa {2*self.meia*100:.0f} cm, {massa:.1f} kg",
            f"**Torque** joelho {tq('knee'):.0f}% · hip_pitch {tq('hip_pitch'):.0f}% · hip_roll {tq('hip_roll'):.0f}% "
            f"· tornozelo {tq('ankle_pitch'):.0f}% · cintura {pct[c.nomes.index('waist_pitch')]:.0f}% "
            f"· ombro {tq('shoulder_pitch'):.0f}% · cotovelo {tq('elbow'):.0f}%",
        ]
        if self.mostra and not self.nas_maos:
            lado = self.meia + G.FOLGA_PALMA
            dm = [np.linalg.norm(d.site_xpos[c.sid[f"{l}_palm"]] - (self.caixa + (0.0, s * lado, 0.0))) * 100
                  for l, s in (("left", 1), ("right", -1))]
            linhas.append(f"**Mãos até a face** E {dm[0]:.1f} cm · D {dm[1]:.1f} cm · dedos "
                          f"{np.degrees(F.elev_dedos(c, 'left')):+.0f}° / {np.degrees(F.elev_dedos(c, 'right')):+.0f}°")
        linhas += [f"**Folga** corpo {pior(~eh_caixa & ~eh_chao)} · caixa {pior(eh_caixa)} · "
                   f"chão/laje {pior(eh_chao)}",
                   f"**Perto do batente** (> 90% do curso): {', '.join(alto) if alto else 'nenhuma'}"]
        return "\n\n".join(linhas)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--de", type=Path, default=None, help="npz com `qpos` (N, nq) e `hash_mjcf`")
    ap.add_argument("--i", type=int, default=0, help="índice da pose no --de")
    ap.add_argument("--aresta", type=float, default=None)
    ap.add_argument("--massa", type=float, default=None)
    ap.add_argument("--porta", type=int, default=8080)
    args = ap.parse_args()

    c = G.Cena(F.CENA)
    E.prepara(c)
    # laje nos pares de folga (10 mm, como nas tabelas da v3); entra no grupo "chão/laje"
    gl = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")
    robo = [g for g in range(c.m.ngeom) if c.m.geom_contype[g]
            and (mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("robot/")]
    c.pares = np.vstack([c.pares, np.array([(g, gl) for g in robo])])
    c.folga_par = np.concatenate([c.folga_par, np.full(len(robo), F.FOLGA_LAJE)])
    c.m.geom_rgba[c.gid_caixa, 3] = 0.0          # a caixa é desenhada pelo viser (tamanho variável)
    lim = np.array([abs(c.m.actuator_forcerange[np.flatnonzero(c.m.actuator_trnid[:, 0] == j + 1)[0], 1])
                    for j in range(29)])

    R = None
    if args.de is not None:
        R = np.load(args.de.expanduser())
        assert R["hash_mjcf"] == c.hash, "npz de outro MJCF"
    campo = lambda k, i: np.atleast_1d(R[k])[min(i, np.atleast_1d(R[k]).size - 1)]  # noqa: E731
    tem = lambda k: R is not None and k in R                                          # noqa: E731
    n_poses = len(R["qpos"]) if R is not None else 0
    q0 = np.array(R["qpos"][args.i][7:36], dtype=float) if R is not None else c.q_rep.copy()
    meia0 = args.aresta / 2 if args.aresta else (float(campo("meia", args.i)) if tem("meia") else 0.10)
    massa0 = args.massa if args.massa is not None else (float(campo("massa", args.i)) if tem("massa") else 5.0)
    pose = Pose(c, meia0, massa0, lim)

    # ⚠ só nesta máquina: o padrão do viser é 0.0.0.0, e a rede inteira podia salvar poses
    server = viser.ViserServer(host="127.0.0.1", port=args.porta)
    scene = ViserMujocoScene(server, c.m, num_envs=1)
    sujo = threading.Event()
    faixa = {n: (float(np.degrees(c.lo[i])), float(np.degrees(c.hi[i]))) for i, n in enumerate(c.nomes)}
    sl = {}

    def novo(n, rotulo):
        lo, hi = faixa[n]
        v = float(np.clip(np.degrees(q0[c.nomes.index(n)]), lo, hi))
        sl[n] = server.gui.add_slider(rotulo, round(lo, 1), round(hi, 1), 0.5, round(v, 1))

    with server.gui.add_folder("Pose"):
        espelho = server.gui.add_checkbox("espelhar (direita segue a esquerda)", True)
        nome = server.gui.add_text("nome", "agachado")
        salvar = server.gui.add_button("salvar pose")
        de_pe = server.gui.add_button("voltar de pé")
    sel = info = None
    if n_poses > 1:
        with server.gui.add_folder("Tabela"):
            sel = server.gui.add_slider("pose da tabela", 0, n_poses - 1, 1, args.i)
            info = server.gui.add_markdown("")
    with server.gui.add_folder("Caixa"):
        mostra = server.gui.add_checkbox("mostrar", True)
        nas_maos = server.gui.add_checkbox("presa entre as palmas", False)
        aresta = server.gui.add_slider("aresta [cm]", 10.0, 50.0, 1.0, round(2 * meia0 * 100, 1))
        massa = server.gui.add_slider("massa [kg]", 0.0, 15.0, 0.5, massa0)
        cx = server.gui.add_slider("à frente do meio dos pés [m]", -0.30, 0.80, 0.01, pose.caixa_x)
        apoio = server.gui.add_slider("altura do apoio [m]", 0.0, 0.80, 0.01, pose.apoio)
    painel = server.gui.add_markdown("")
    for lado, rot in (("left", "esquerda"), ("right", "direita")):
        with server.gui.add_folder(f"Perna {rot}"):
            for j in PERNA:
                novo(f"{lado}_{j}", j)
    with server.gui.add_folder("Cintura"):
        for j in CINTURA:
            novo(j, j)
    for lado, rot in (("left", "esquerda"), ("right", "direita")):
        with server.gui.add_folder(f"Braço {rot}"):
            for j in BRACO:
                novo(f"{lado}_{j}", j)
    with server.gui.add_folder("Vista", expand_by_default=False):
        scene.create_visualization_gui()

    def espelha():
        for j in PERNA + BRACO:
            lo, hi = faixa[f"right_{j}"]
            v = float(np.clip(sl[f"left_{j}"].value * (-1 if j in TROCA else 1), lo, hi))
            if sl[f"right_{j}"].value != v:
                sl[f"right_{j}"].value = v

    def simetrica():
        return all(abs(sl[f"right_{j}"].value - sl[f"left_{j}"].value * (-1 if j in TROCA else 1)) < 1.0
                   for j in PERNA + BRACO)

    def mudou(_=None):
        if espelho.value:
            espelha()
        sujo.set()

    def trava_direita(_=None):
        for j in PERNA + BRACO:
            sl[f"right_{j}"].disabled = espelho.value
        mudou()

    def caixa(_=None):
        pose.mostra, pose.nas_maos = mostra.value, nas_maos.value
        pose.caixa_x, pose.apoio, pose.massa = cx.value, apoio.value, massa.value
        if abs(aresta.value / 200 - pose.meia) > 1e-6:
            pose.tamanho(aresta.value / 200)
        sujo.set()

    def carrega(k):
        """Pose k do npz nos sliders: juntas, caixa (tamanho, massa, posição) e apoio."""
        q = np.array(R["qpos"][k], dtype=float)
        c.d.qpos[:] = q; mujoco.mj_kinematics(c.m, c.d)
        meio_x = 0.5 * (c.d.site_xpos[c.sid["left_foot"]][0] + c.d.site_xpos[c.sid["right_foot"]][0])
        espelho.value = False                     # sem espelho durante a carga: a direita vem do npz
        for i, n in enumerate(c.nomes):
            lo, hi = faixa[n]
            sl[n].value = float(np.clip(round(np.degrees(q[7 + i]), 1), lo, hi))
        if tem("meia"):
            aresta.value = float(np.clip(round(float(campo("meia", k)) * 200, 1), 10.0, 50.0))
        if tem("massa"):
            massa.value = float(campo("massa", k))
        presa = bool(campo("nas_maos", k)) if tem("nas_maos") else False
        topo = float(campo("topo", k)) if tem("topo") else 0.0
        mostra.value = True
        nas_maos.value = presa
        apoio.value = round(max(topo, 0.0), 2)
        cx.value = float(np.clip(round(float(q[c.adr_caixa] - meio_x), 2), -0.30, 0.80))
        if info is not None:
            onde = "de pé, caixa nas mãos" if presa else ("chão" if topo <= 0 else f"laje {topo:.2f} m")
            partes = [f"**Pose {k}** de 0–{n_poses - 1}: {onde}"]
            if tem("aceita"):
                partes.append("IK ok" if bool(campo("aceita", k)) else "IK REJEITADA")
            if tem("residuo_pe"):
                partes.append(f"resíduo da IK: pé {float(campo('residuo_pe', k))*100:.1f} cm, "
                              f"mão {float(campo('residuo', k))*100:.1f} cm")
            info.content = " · ".join(partes)
        espelho.value = simetrica()
        trava_direita()

    for s in sl.values():
        s.on_update(mudou)
    espelho.on_update(trava_direita)
    for h in (mostra, nas_maos, aresta, massa, cx, apoio):
        h.on_update(caixa)
    if sel is not None:
        sel.on_update(lambda _: carrega(int(sel.value)))

    @de_pe.on_click
    def _(_):
        for i, n in enumerate(c.nomes):
            lo, hi = faixa[n]
            sl[n].value = float(np.clip(round(np.degrees(c.q_rep[i]), 1), lo, hi))

    @salvar.on_click
    def _(_):
        # ⚠ só o último componente do nome: "../agachado/ref_agachado" sobrescrevia a tabela da v3
        base = Path(nome.value).name
        if base in ("", ".", ".."):
            print(f"[monta_pose] nome inválido: {nome.value!r}", flush=True); return
        SAIDA.mkdir(parents=True, exist_ok=True)
        f = SAIDA / f"{base}.npz"
        np.savez(f, hash_mjcf=c.hash, qpos=c.d.qpos[None].copy(), caixa_xyz=pose.caixa[None].copy(),
                 meia=np.array([pose.meia]), massa=np.array([pose.massa]), topo=np.array([pose.apoio]),
                 nas_maos=np.array([pose.nas_maos]), painel=painel.content)
        print(f"[monta_pose] salvo {f}", flush=True)

    if R is not None:
        carrega(args.i)
    else:
        trava_direita()
    print(f"[monta_pose] abra http://localhost:{args.porta}  (Ctrl+C encerra)", flush=True)
    while True:
        if sujo.wait(0.1):
            sujo.clear()
            q = np.radians([sl[n].value for n in c.nomes])
            pose.monta(q)
            painel.content = pose.texto()
            scene.update_from_mjdata(c.d)
            server.scene.add_box("/caixa", color=COR_CAIXA, dimensions=(2 * pose.meia,) * 3,
                                 position=tuple(float(v) for v in pose.caixa),
                                 visible=bool(pose.mostra or pose.nas_maos))


if __name__ == "__main__":
    main()
