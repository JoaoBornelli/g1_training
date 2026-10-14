"""A pose de pé SEM CAIXA, pelo mesmo solver do `gera_de_pe` (prioridade de tarefas + torque).

Troca as tarefas das palmas por uma tarefa de braço: as 14 juntas do braço no keyframe do
molde (mãos ao lado do tronco, enunciado §3). A caixa sai da cena e a massa dela é zero.
O resto fica igual: pés planos e paralelos, CoM sobre o meio dos pés, pelve nivelada,
tronco vertical, sem rotação interna do quadril, folga entre peças, torque estático mínimo.
"""
import argparse
from pathlib import Path

import numpy as np

from g1_limpo.ik import gera_de_pe as P


class PeSemCaixa(P.Pe):
    def residuo(self, st, h, ativos):
        c, d = self.c, self.c.d
        longe = np.array([5.0, 0.0, 5.0])
        c.poe(st[:29], st[29], st[30], st[31], longe)
        e = []
        for s in ("left_foot", "right_foot"):
            p = d.site_xpos[c.sid[s]]; R = d.site_xmat[c.sid[s]].reshape(3, 3)
            e += [p[2] - P.GB.Z_SITE_ALVO, R[0, 2], R[1, 2], np.arctan2(R[1, 0], R[0, 0])]
        ly, ry = d.site_xpos[c.sid["left_foot"]][1], d.site_xpos[c.sid["right_foot"]][1]
        e += [ly + ry, max(0.0, (ly - ry) - P.LARGURA[1]), max(0.0, P.LARGURA[0] - (ly - ry)),
              d.site_xpos[c.sid["left_foot"]][0] - d.site_xpos[c.sid["right_foot"]][0]]
        n_pe = len(e)
        q = st[:29]
        e += list(q[15:29] - c.q_rep[15:29])          # braço no keyframe do molde
        n_mao = len(e)
        com = d.subtree_com[c.bid_pelve]
        mid = 0.5 * (d.site_xpos[c.sid["left_foot"]] + d.site_xpos[c.sid["right_foot"]])
        e += [com[0] - mid[0], com[1] - mid[1]]
        e += [max(0.0, P.GB.inclinacao(c, c.bid_pelve) - P.GB.PELVE_MAX)]
        e += [max(0.0, -q[c.i_yaw["left"]]), max(0.0, q[c.i_yaw["right"]])]
        if ativos is not None and len(ativos):
            e += list(np.maximum(0.0, c.folga_par[ativos] - c.folgas(ativos)))
        Rt = d.xmat[c.bid_torso].reshape(3, 3)
        e += [P.W_ORI * st[30], P.W_ORI * st[31], P.W_ORI * Rt[0, 2], P.W_ORI * Rt[1, 2]]
        n_tar = len(e)
        e += list(self.w_grupo * self.torque() / self.tau_max)
        e += list(P.W_REP * (q - self.q_hum))
        return np.array(e), n_pe, n_mao, n_tar

    def caixa(self, st, h):
        return np.array([5.0, 0.0, 5.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cena", default="~/g1_pilota/")
    ap.add_argument("--saida", required=True)
    args = ap.parse_args()
    p = PeSemCaixa(Path(args.cena).expanduser(), 0.0, 0.10)
    c = p.c
    st, e_pe, e_braco = p.resolve(0.95)
    c.poe(st[:29], st[29], st[30], st[31], np.array([5.0, 0.0, 5.0]))
    q = st[:29]
    tronco = float(np.degrees(P.GB.inclinacao(c, c.bid_torso)))
    pelve_i = float(np.degrees(P.GB.inclinacao(c, c.bid_pelve)))
    folga = float(c.folgas().min())
    larg = float(abs(c.d.site_xpos[c.sid["left_foot"]][1] - c.d.site_xpos[c.sid["right_foot"]][1]))
    ok = e_pe < 0.01 and e_braco < 0.02 and folga > -0.002
    print(f"res_pe {e_pe:.4f} res_braco {e_braco:.4f} pelve {st[29]:.3f} tronco {tronco:.1f}° "
          f"pelve_incl {pelve_i:.1f}° | quadril {q[p.ix['left_hip_pitch']]:+.2f} "
          f"joelho {q[p.ix['left_knee']]:.2f} tornozelo {q[p.ix['left_ankle_pitch']]:+.2f} "
          f"| pés {larg:.3f} m | folga {folga*1000:.1f} mm | {'ok' if ok else 'REJEITADA'}")
    np.savez(args.saida, hash_mjcf=c.hash, topo=np.array([-1.0]), meia=np.array([0.10]),
             massa=np.array([0.0]), residuo=np.array([e_braco]), residuo_pe=np.array([e_pe]),
             pelve_z=np.array([float(c.d.xpos[c.bid_pelve][2])]), pelve_incl=np.array([pelve_i]),
             folga_min=np.array([folga]), pes_larg=np.array([larg]), tronco_incl=np.array([tronco]),
             com_offset=np.array([0.0]), frac_max=np.array([float((np.abs(q - (c.hi + c.lo) / 2) / ((c.hi - c.lo) / 2)).max())]),
             aceita=np.array([ok]), qpos=c.d.qpos.copy()[None, :])


if __name__ == "__main__":
    main()
