"""Gera a pose de REFERÊNCIA DE PÉ COM CAIXA por IK: a caixa no alvo de transporte.

    python -m g1_limpo.ik.gera_de_pe --cena ~/g1_pilota_zero15/ --saida g1_limpo/ik/ref_de_pe.npz

Spec `specs/g1-limpo-postura-de-pe.md` §2 e §4; enunciado §3 ("De pé, como um humano").

⚠ O ESQUEMA É O DO `ref_botar.npz`, e a chave também: `topo + meia` é a altura do centro
da caixa. Aqui não há laje: `topo = h − meia` só existe para a `FormaPostural` juntar as
duas tabelas numa chave só.

⚠⚠ PRIORIDADE DE TAREFAS, e não o Newton amortecido do `gera_botar`. Com a pose de pé o
sistema é subdeterminado, e o custo secundário (torque e repouso) pesa pouco perto do
amortecimento: MEDIDO em 30/09, o Newton simples parava perto da SEMENTE (joelho 0,58
contra 0,15 do ótimo). Aqui as tarefas convergem primeiro, e o secundário desce só no
espaço nulo delas, em três fases: só tarefas, tarefas + secundário, só tarefas.

⚠⚠ O SECUNDÁRIO É O TORQUE ESTÁTICO POR GRUPO: normalizado pelo limite de cada motor, com
peso 1 nas 15 juntas de perna e cintura e 0,3 nas 14 do braço. Com peso igual, os motores
pequenos (ombro 25 N·m, punho 5 N·m) dominavam a norma, e com 5 kg a IK trocava carga no
braço por AGACHAR: joelho 21,9 N·m contra 6,4 de pé (MEDIDO, 30/09).

⚠ `mj_kinematics` e `mj_comPos` apenas, como no `gera_botar`: a pose é cinemática.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from g1_limpo.ik import gera_botar as GB
from g1_limpo.knobs import Knobs

LARGURA = (0.20, 0.30)          # m, faixa da largura dos pés de pé (o keyframe apoia 0,237)
L_MOM = 0.05                    # m, escala dos momentos do pé no mínimo-norma da reação
W_REP, W_ORI = 0.05, 1.0        # repouso fraco; pelve nivelada e tronco vertical
FASE_1, FASE_3 = 120, 80        # iterações só de tarefas, no começo e no fim
LONGE = np.array([5.0, 0.0, 5.0])


class Pe:
    """A cena do `gera_botar`, mais o torque estático e o alvo de transporte."""

    def __init__(self, pasta: Path, massa: float, meia: float):
        self.c = c = GB.Cena(pasta)
        c.cenario(5.0, meia)                          # a laje fica fora, a 5 m
        m = c.m
        self.nomes = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).split("/")[-1]
                      .replace("_joint", "") for j in range(1, 1 + c.n_juntas)]
        self.ix = {n: i for i, n in enumerate(self.nomes)}
        self.dof = np.array([m.jnt_dofadr[j] for j in range(1, 1 + c.n_juntas)])
        self.tau_max = np.zeros(c.n_juntas)
        for a in range(m.nu):
            self.tau_max[int(m.actuator_trnid[a, 0]) - 1] = m.actuator_forcerange[a, 1]
        self.w_grupo = np.array([1.0] * 15 + [0.3] * 14)
        self.m_robo = float(m.body_mass[1:1 + 30].sum())
        self.massa, self.meia = massa, meia
        self.peito = np.array(Knobs().alvo.peito_b)
        # a referência fraca de repouso: pernas retas, braços soltos ao lado do corpo
        self.q_hum = np.zeros(c.n_juntas)
        for n, v in (("left_shoulder_roll", 0.15), ("right_shoulder_roll", -0.15),
                     ("left_elbow", 0.3), ("right_elbow", 0.3),
                     ("left_knee", 0.1), ("right_knee", 0.1)):
            self.q_hum[self.ix[n]] = v

    def _jac(self, sid):
        jp, jr = np.zeros((3, self.c.m.nv)), np.zeros((3, self.c.m.nv))
        mujoco.mj_jacSite(self.c.m, self.c.d, jp, jr, sid)
        return jp, jr

    def torque(self):
        """(29,) N·m para segurar a pose parada: gravidade, pés como apoio, caixa nas palmas.

        Equilíbrio com velocidade zero: `qfrc_bias = τ + Σ Jᵀ w`. As 6 linhas da raiz não
        têm motor e fixam as reações dos dois pés, divididas pelo mínimo-norma (momentos
        na escala `L_MOM`). O peso da caixa entra como m·g/2 para baixo em cada palma.
        """
        c, m = self.c, self.c.m
        g = np.zeros(m.nv); mujoco.mj_rne(m, c.d, 0, g)
        fp = np.zeros(m.nv)
        for s in ("left_palm", "right_palm"):
            fp += self._jac(c.sid[s])[0].T @ np.array([0.0, 0.0, -self.massa * 9.81 / 2])
        JL = np.vstack(self._jac(c.sid["left_foot"])); JR = np.vstack(self._jac(c.sid["right_foot"]))
        A = np.hstack([JL[:, :6].T, JR[:, :6].T])
        D = np.diag([1, 1, 1, L_MOM, L_MOM, L_MOM] * 2)
        w = D @ np.linalg.lstsq(A @ D, g[:6] - fp[:6], rcond=None)[0]
        return (g - JL.T @ w[:6] - JR.T @ w[6:] - fp)[self.dof]

    def caixa(self, st, h):
        """O alvo de transporte (`comando._alvo_ancorado_na_base`): base + R·peito_b, z em h."""
        quat = np.zeros(4); mujoco.mju_euler2Quat(quat, np.array([st[30], st[31], 0.0]), b"xyz")
        a = np.zeros(3); mujoco.mju_rotVecQuat(a, self.peito, quat)
        return np.array([a[0], a[1], h])

    def residuo(self, st, h, ativos):
        """(e, n_pe, n_mao, n_tarefas): as tarefas primeiro, o secundário no fim."""
        c, d = self.c, self.c.d
        cx = self.caixa(st, h)
        c.poe(st[:29], st[29], st[30], st[31], cx)
        e = []
        for s in ("left_foot", "right_foot"):
            p = d.site_xpos[c.sid[s]]; R = d.site_xmat[c.sid[s]].reshape(3, 3)
            e += [p[2] - GB.Z_SITE_ALVO, R[0, 2], R[1, 2], np.arctan2(R[1, 0], R[0, 0])]
        ly, ry = d.site_xpos[c.sid["left_foot"]][1], d.site_xpos[c.sid["right_foot"]][1]
        e += [ly + ry, max(0.0, (ly - ry) - LARGURA[1]), max(0.0, LARGURA[0] - (ly - ry)),
              d.site_xpos[c.sid["left_foot"]][0] - d.site_xpos[c.sid["right_foot"]][0]]
        n_pe = len(e)
        lado = self.meia + GB.FOLGA_PALMA
        e += list(d.site_xpos[c.sid["left_palm"]] - (cx + (0.0, +lado, 0.0)))
        e += list(d.site_xpos[c.sid["right_palm"]] - (cx + (0.0, -lado, 0.0)))
        for s in ("left_palm", "right_palm"):
            R = d.site_xmat[c.sid[s]].reshape(3, 3); e += [R[0, 1], R[2, 1]]
        n_mao = len(e)
        com = (self.m_robo * d.subtree_com[c.bid_pelve] + self.massa * cx) / (self.m_robo + self.massa)
        mid = 0.5 * (d.site_xpos[c.sid["left_foot"]] + d.site_xpos[c.sid["right_foot"]])
        e += [com[0] - mid[0], com[1] - mid[1]]
        e += [max(0.0, GB.inclinacao(c, c.bid_pelve) - GB.PELVE_MAX)]
        q = st[:29]
        e += [max(0.0, -q[c.i_yaw["left"]]), max(0.0, q[c.i_yaw["right"]])]
        if ativos is not None and len(ativos):
            e += list(np.maximum(0.0, c.folga_par[ativos] - c.folgas(ativos)))
        Rt = d.xmat[c.bid_torso].reshape(3, 3)
        e += [W_ORI * st[30], W_ORI * st[31], W_ORI * Rt[0, 2], W_ORI * Rt[1, 2]]
        n_tar = len(e)
        e += list(self.w_grupo * self.torque() / self.tau_max)
        e += list(W_REP * (q - self.q_hum))
        return np.array(e), n_pe, n_mao, n_tar

    def resolve(self, h, sementes=4, iters=520, margem=0.10):
        """A melhor pose das sementes: tarefas dentro da tolerância, e o menor torque."""
        c = self.c
        melhor = None
        for s in range(sementes):
            rng = np.random.default_rng(s)
            base = self.q_hum if s == 1 else c.q_rep
            ruido = (rng.random(c.n_juntas) - 0.5) * (0.5 if s >= 2 else 0.0)
            st = np.concatenate([np.clip(base + ruido, c.lo + 1e-3, c.hi - 1e-3),
                                 [0.80 if s == 1 else 0.78], [0.0, 0.0]])
            for it in range(iters):
                usa_d2 = FASE_1 <= it < iters - FASE_3
                c.poe(st[:29], st[29], st[30], st[31], self.caixa(st, h))
                ativos = np.flatnonzero(c.folgas() < c.folga_par + GB.BANDA_ATIVA)
                f0, _, _, nt = self.residuo(st, h, ativos)
                J = np.zeros((len(f0), 32))
                for k in range(32):
                    s2 = st.copy(); s2[k] += 1e-5
                    J[:, k] = (self.residuo(s2, h, ativos)[0] - f0) / 1e-5
                Jt, et, Js, es = J[:nt], f0[:nt], J[nt:], f0[nt:]
                d1 = Jt.T @ np.linalg.solve(Jt @ Jt.T + 2e-2 * np.eye(nt), -et)
                d2 = np.zeros(32)
                if usa_d2:
                    N = np.eye(32) - np.linalg.pinv(Jt, rcond=1e-4) @ Jt
                    Jh = Js @ N
                    d2 = 0.5 * N @ (Jh.T @ np.linalg.solve(Jh @ Jh.T + 1e-3 * np.eye(len(es)),
                                                           -es - Js @ d1))
                st = st + np.clip(d1 + np.clip(d2, -0.05, 0.05), -0.15, 0.15)
                folga = margem * (c.hi - c.lo) / 2.0
                st[:29] = np.clip(st[:29], c.lo + folga, c.hi - folga)
                st[29] = np.clip(st[29], 0.25, 1.05)
            f, n_pe, n_mao, n_tar = self.residuo(st, h, None)
            e_pe = float(np.abs(f[:n_pe]).max()); e_mao = float(np.abs(f[n_pe:n_mao]).max())
            e_tar = float(np.abs(f[n_mao:n_tar]).max())
            custo = float(np.linalg.norm(f[n_tar:n_tar + c.n_juntas]))
            ok = e_pe < 0.01 and e_mao < 0.02 and e_tar < 0.02
            chave = (not ok, custo if ok else max(e_pe, e_mao, e_tar))
            if melhor is None or chave < melhor[0]:
                melhor = (chave, st.copy(), e_pe, e_mao)
        return melhor[1], melhor[2], melhor[3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cena", required=True, help="pasta com cena.mjb e cena.npz")
    ap.add_argument("--saida", default="ref_de_pe.npz")
    ap.add_argument("--alturas", type=float, nargs="*", default=[0.85, 0.90, 0.95, 1.00],
                    help="altura do CENTRO da caixa no mundo, m")
    ap.add_argument("--massa", type=float, default=3.0, help="kg; o meio da faixa de 1 a 5")
    ap.add_argument("--meia", type=float, default=0.10)
    ap.add_argument("--sementes", type=int, default=4)
    ap.add_argument("--iters", type=int, default=520)
    args = ap.parse_args()

    p = Pe(Path(args.cena).expanduser(), args.massa, args.meia)
    c = p.c
    print(f"[gera_de_pe] {c.caminho}  hash {c.hash}  massa {args.massa} kg  meia {args.meia} m")
    print(f"{'h':>6} {'res_pe':>8} {'res_mao':>8} {'pelve_z':>8} {'pelve_i':>8} {'tronco':>8} "
          f"{'joelho':>7} {'pes_x':>7} {'folga_mm':>8}  veredito")
    linhas = []
    for h in args.alturas:
        st, e_pe, e_mao = p.resolve(h, args.sementes, args.iters)
        cx = p.caixa(st, h)
        c.poe(st[:29], st[29], st[30], st[31], cx)
        d = c.d
        lf, rf = d.site_xpos[c.sid["left_foot"]], d.site_xpos[c.sid["right_foot"]]
        mid = 0.5 * (lf + rf)
        com = (p.m_robo * d.subtree_com[c.bid_pelve] + args.massa * cx) / (p.m_robo + args.massa)
        pelve_i = float(np.degrees(GB.inclinacao(c, c.bid_pelve)))
        frac = float((np.abs(st[:29] - (c.hi + c.lo) / 2) / ((c.hi - c.lo) / 2)).max())
        folga = float(c.folgas().min())
        ok = (e_pe < 0.01 and e_mao < 0.02 and frac < 0.95 and pelve_i < GB.PELVE_PORTAO
              and folga > -0.002)
        l = dict(topo=h - args.meia, meia=args.meia, massa=args.massa, residuo=e_mao,
                 residuo_pe=e_pe, pelve_z=float(d.xpos[c.bid_pelve][2]), pelve_incl=pelve_i,
                 folga_min=folga, pes_larg=float(abs(lf[1] - rf[1])),
                 tronco_incl=float(np.degrees(GB.inclinacao(c, c.bid_torso))),
                 com_offset=float(np.hypot(com[0] - mid[0], com[1] - mid[1])),
                 frac_max=frac, aceita=ok, qpos=d.qpos.copy())
        linhas.append(l)
        print(f"{h:6.2f} {e_pe:8.4f} {e_mao:8.4f} {l['pelve_z']:8.3f} {pelve_i:8.1f} "
              f"{l['tronco_incl']:8.1f} {st[p.ix['left_knee']]:7.2f} {mid[0] - d.xpos[c.bid_pelve][0]:+7.3f} "
              f"{folga * 1000:8.1f}  {'ok' if ok else 'REJEITADA'}  |frac|max {frac:.2f}")
    print(f"\n{sum(l['aceita'] for l in linhas)} de {len(linhas)} poses aceitas")
    np.savez(args.saida, hash_mjcf=c.hash,
             **{k: np.array([l[k] for l in linhas]) for k in linhas[0]})
    print(f"-> {args.saida}")


if __name__ == "__main__":
    main()
