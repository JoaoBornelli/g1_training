"""Sombra da caixa e do corpo na visão do MID-360, robô de pé com a caixa no peito.

Raios do MID-360 (FOV vertical -7..+52 graus no frame do sensor, montado invertido) a partir da
pose do URDF (`mid360_joint`, no `torso_link`). Para cada raio: onde bate primeiro (chão, caixa,
corpo) e onde bateria no chão sem obstáculo. Saída: mapa de cima (PNG) e números por setor.
"""
import sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
AQUI = Path(__file__).resolve().parent
import g1_limpo.ik.gera_botar as G
from g1_limpo.ik.ve_botar import _grava_png

POS = np.array([0.0002835, 0.00003, 0.428434]); RPY = (np.pi, 0.0511, 0.0)
c = G.Cena(Path("~/g1_pilota/").expanduser()); m, d = c.m, c.d
bt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
qr = np.zeros(4); mujoco.mju_euler2Quat(qr, np.array(RPY), b"xyz"); Rs = np.zeros(9); mujoco.mju_quat2Mat(Rs, qr); Rs = Rs.reshape(3, 3)
nome = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
# ⚠ as malhas VISUAIS (sem nome, contype 0) saem do raio: o MID-360 fica DENTRO da casca da cabeça,
# e no real ele olha por uma janela. Vão para o grupo 5, que o raio não vê. Ficam: o chão, a caixa e
# as geometrias de COLISÃO do robô.
for g in range(m.ngeom):
    if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
        m.geom_group[g] = 5
GRUPOS = np.array([1, 1, 1, 1, 1, 0], np.uint8)
SAIDA = 0.12
corpo_de = lambda g: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "") if g >= 0 else ""
az = np.radians(np.arange(0, 360, 1.0)); el = np.radians(np.arange(-7, 52.01, 0.5))
A, E = np.meshgrid(az, el); dirs_s = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)

def mede(qpos, meia, rotulo, out):
    c.cenario(5.0, meia); d.qpos[:] = qpos; mujoco.mj_forward(m, d)
    Rt = d.xmat[bt].reshape(3, 3); p0 = d.xpos[bt] + Rt @ POS; R = Rt @ Rs
    dirs = dirs_s @ R.T; n = len(dirs)
    geomid = np.full(n, -1, np.int32); dist = np.full(n, -1.0)
    # ⚠ a colisão da CABEÇA (no `torso_link`) envolve o sensor; no real ele olha por uma janela.
    # Cada raio sai de um ponto próprio, SAIDA m adiante na direção dele, fora da cabeça.
    for i in range(n):
        geomid[i:i + 1] = -1
        dist[i] = mujoco.mj_ray(m, d, p0 + SAIDA * dirs[i], dirs[i], GRUPOS, 1, -1, geomid[i:i + 1])
    dist = np.where(dist >= 0, dist + SAIDA, dist)
    chao_t = np.where(dirs[:, 2] < -1e-6, -p0[2] / dirs[:, 2], np.inf)          # chão em z = 0
    pchao = p0 + dirs * chao_t[:, None]
    r = np.hypot(pchao[:, 0] - d.xpos[c.bid_pelve][0], pchao[:, 1] - d.xpos[c.bid_pelve][1])
    alvo = np.isfinite(chao_t) & (r < 3.0)
    nomes = np.array([corpo_de(g) for g in geomid])
    caixa = np.char.startswith(nomes, "box/"); corpo = np.char.startswith(nomes, "robot/")
    perto = corpo & (dist < 0.06)
    if perto.any():
        print(f"   ⚠ {perto.sum()} raios batem no próprio corpo a menos de 6 cm do sensor: "
              f"{sorted(set(nomes[perto]))[:4]}")
    bloq_cx = alvo & caixa & (dist < chao_t); bloq_cp = alvo & corpo & (dist < chao_t); visto = alvo & ~bloq_cx & ~bloq_cp
    # setores, em azimute do mundo em volta da pelve (0 = frente)
    ang = np.degrees(np.arctan2(pchao[:, 1] - d.xpos[c.bid_pelve][1], pchao[:, 0] - d.xpos[c.bid_pelve][0]))
    print(f"== {rotulo}: lidar a {p0[2]:.2f} m; chão mais perto visto a {r[visto].min():.2f} m")
    for nome_s, lo, hi in (("frente ±30°", -30, 30), ("lados 30–150°", 30, 150), ("trás", 150, 180)):
        sel = alvo & (np.abs(ang) >= lo) & (np.abs(ang) < hi)
        for a, b in ((1.0, 2.0), (2.0, 3.0)):
            s2 = sel & (r >= a) & (r < b)
            if s2.sum():
                print(f"   {nome_s:13s} {a:.0f}–{b:.0f} m: visto {100*visto[s2].mean():5.1f}% | caixa {100*bloq_cx[s2].mean():5.1f}% | corpo {100*bloq_cp[s2].mean():5.1f}%")
    fr = alvo & (np.abs(ang) < 30)
    if bloq_cx[fr].any():
        print(f"   sombra da caixa na frente: de {r[fr & bloq_cx].min():.2f} a {r[fr & bloq_cx].max():.2f} m, "
              f"azimute ±{np.abs(ang[fr & bloq_cx]).max():.0f}°")
    # mapa de cima, 6 x 6 m, 5 cm
    N = 120; img = np.full((N, N, 3), 235, np.uint8)
    ij = lambda p: (np.clip(((3 - (p[:, 0] - d.xpos[c.bid_pelve][0])) / 0.05).astype(int), 0, N - 1),
                    np.clip(((3 - (p[:, 1] - d.xpos[c.bid_pelve][1])) / 0.05).astype(int), 0, N - 1))
    for mask, cor in ((visto, (40, 160, 40)), (bloq_cp, (240, 150, 30)), (bloq_cx, (200, 30, 30))):
        i, j = ij(pchao[mask]); img[i, j] = cor
    k = N // 2; img[k - 1:k + 2, k - 1:k + 2] = (0, 0, 0)                          # o robô
    for rr in (1.0, 2.0):                                                           # anéis de 1 e 2 m
        t = np.linspace(0, 2 * np.pi, 400); ii = (k - rr / 0.05 * np.cos(t)).astype(int); jj = (k - rr / 0.05 * np.sin(t)).astype(int)
        img[ii.clip(0, N - 1), jj.clip(0, N - 1)] = (90, 90, 90)
    big = np.kron(img, np.ones((4, 4, 1), np.uint8))
    _grava_png(out, big)

T = np.load(AQUI.parent / "mao_fixa" / "ref_v3_de_pe_theta-30_tamanhos.npz")
D = np.load("g1_limpo/ik/ref_v3_de_pe.npz")
out = AQUI; out.mkdir(exist_ok=True)
mede(D["qpos"][0], 0.10, "de pé SEM caixa", out / "sombra_sem_caixa.png")
for i, a in enumerate(T["meia"]):
    mede(T["qpos"][i], float(a), f"de pé com caixa de {2*a*100:.0f} cm", out / f"sombra_caixa_{2*a*100:.0f}cm.png")
print("mapas em", out, "(verde = chão visto; vermelho = sombra da caixa; laranja = sombra do corpo; anéis de 1 e 2 m; frente para cima)")
