"""Gera g1_v3/dados/referencias.npz: a tabela de referências macro da IK (A3; S §8.1, §6.3, I §2.2).

Lê, por caminho, as tabelas aprovadas do g1_limpo (ik/agachado, ik/mao_fixa e ik/ref_v3_de_pe).
O script não importa o g1_limpo nem o g1_v3. A FK é sem física: o modelo só lê as poses
(`mj_kinematics`, PD-3); não há passo de simulação nem viewer.

A guarda é a tolerância nas grandezas macro, e não o hash do MJCF (decisão de 14/10): na pose
de pé com caixa de cada tamanho, cada palma fica a até 5 cm do plano da face lateral da caixa
(a IK aceita a mão a até 1 cm, S §8.1), e a pelve da FK bate com `qpos[2]` em 1 mm. Fora disso
o script para com a medida, sem gravar o .npz.
"""
import importlib.util
from pathlib import Path

import mujoco
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
IK = RAIZ / "g1_limpo" / "ik"
AGACHADO = IK / "agachado" / "ref_agachado.npz"
TAMANHOS = IK / "mao_fixa" / "ref_v3_de_pe_theta-30_tamanhos.npz"
DE_PE = IK / "ref_v3_de_pe.npz"
SAIDA = Path(__file__).resolve().parent / "referencias.npz"
PALMAS = ("left_palm", "right_palm")
PES = ("left_foot", "right_foot")
TOL_PELVE = 1e-3  # m (§8.1)
TOL_PALMA = 0.05  # m, distância da palma ao plano da face lateral (§8.1)


def _xml_g1():
    spec = importlib.util.find_spec("mjlab")  # acha a pasta sem executar o pacote
    assert spec is not None and spec.submodule_search_locations, "mjlab não instalado"
    return Path(list(spec.submodule_search_locations)[0]) / "asset_zoo/robots/unitree_g1/xmls/g1.xml"


def _npz(caminho):
    return dict(np.load(caminho, allow_pickle=False))


def _rot(quat):
    """Matriz 3×3 de um quaternion wxyz (convenção do qpos do MuJoCo)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    return R.reshape(3, 3)


def _fk(m, q):
    """Cinemática direta da pose q (36 juntas da raiz e do robô; o resto é a caixa)."""
    d = mujoco.MjData(m)
    d.qpos[:] = q[:m.nq]
    mujoco.mj_kinematics(m, d)
    return d


def _corpo(m, d, nome):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nome)
    return d.xpos[i], d.xmat[i].reshape(3, 3)


def _sitio(m, d, nome):
    return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, nome)]


def _na_pelve(m, d, nomes):
    """Sítios (n,3) no frame da pelve: R^T (p − origem), com R e origem do corpo `pelvis`."""
    origem, R = _corpo(m, d, "pelvis")
    return np.array([(_sitio(m, d, n) - origem) @ R for n in nomes])


def _centro_na_pelve(m, q):
    """Centro da caixa (qpos[36:39]) no frame da pelve (I §2.5: x 0,25 e z +0,11 para 20 cm)."""
    origem, R = _corpo(m, _fk(m, q), "pelvis")
    return (q[36:39] - origem) @ R


def _distancias_da_face(m, d, q, a):
    """Distância (m) de cada palma ao plano da face lateral da caixa de meia aresta a.

    As faces laterais são as de ±y no frame da caixa (palmas_alvo_caixa, §6.3). A caixa é
    `qpos[36:43]`: centro e quaternion.
    """
    loc = (np.array([_sitio(m, d, n) for n in PALMAS]) - q[36:39]) @ _rot(q[39:43])
    return np.abs(np.abs(loc[:, 1]) - a)


def _capsulas(m, d, lado):
    """Extremos (7,2,3) e raios (7,) das 7 cápsulas de sola de um pé, no mundo (PD-35)."""
    gs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{lado}_foot{k}_collision") for k in range(1, 8)]
    seg = []
    for g in gs:
        meio_eixo = m.geom_size[g, 1] * d.geom_xmat[g].reshape(3, 3)[:, 2]
        seg.append((d.geom_xpos[g] - meio_eixo, d.geom_xpos[g] + meio_eixo))
    return np.array(seg), m.geom_size[gs, 0]


def _grade(ag):
    """Grade (topo, meia) das linhas com topo ≥ 0; topo = −1 são as poses de pé com caixa."""
    ok = ag["topo"] >= 0
    topos, meias = np.unique(ag["topo"][ok]), np.unique(ag["meia"][ok])
    shape = (len(topos), len(meias))
    g = {k: np.zeros(shape) for k in ("grade_pelve", "grade_tronco", "grade_recuo")}
    aceita, vistos = np.zeros(shape, bool), np.zeros(shape, int)
    for r in np.flatnonzero(ok):
        i = int(np.flatnonzero(np.isclose(topos, ag["topo"][r]))[0])
        j = int(np.flatnonzero(np.isclose(meias, ag["meia"][r]))[0])
        vistos[i, j] += 1
        g["grade_pelve"][i, j] = ag["pelve_z"][r]
        g["grade_tronco"][i, j] = np.radians(ag["tronco_incl"][r])  # a tabela de origem está em graus
        g["grade_recuo"][i, j] = np.linalg.norm(ag["qpos"][r, 0:2] - ag["caixa_xyz"][r, 0:2])
        aceita[i, j] = ag["aceita"][r]
    assert (vistos == 1).all(), "a grade não tem exatamente um nó por (topo, meia)"
    return topos, meias, g, aceita


def gera():
    m = mujoco.MjModel.from_xml_path(str(_xml_g1()))
    assert m.nq == 36, m.nq
    ag, tm, de = _npz(AGACHADO), _npz(TAMANHOS), _npz(DE_PE)
    assert ag["hash_mjcf"] == tm["hash_mjcf"] == de["hash_mjcf"], "tabelas de origem com MJCF diferente"

    # Guarda (§8.1): medidas de cada tamanho de caixa, antes de gravar.
    for k, a in enumerate(tm["meia"]):
        d = _fk(m, tm["qpos"][k])
        dif_pelve = abs(_corpo(m, d, "pelvis")[0][2] - tm["qpos"][k][2])
        dist = _distancias_da_face(m, d, tm["qpos"][k], a)
        print(f"meia {a:.2f}: palmas a {dist[0]*100:.2f} e {dist[1]*100:.2f} cm da face; "
              f"pelve FK − qpos[2] = {dif_pelve*1000:.3f} mm")
        if (dist < 0).any() or (dist > TOL_PALMA).any() or dif_pelve > TOL_PELVE:
            raise SystemExit(f"guarda falhou na meia {a}: palmas {dist} m, pelve {dif_pelve} m")

    # Pose de pé sem caixa (pose 0 de ref_v3_de_pe): palmas, pés, sola e altura do site.
    qde = de["qpos"][0]
    d = _fk(m, qde)
    sola, r_sola = _capsulas(m, d, "left")
    sola_z = float((sola[..., 2] - r_sola[:, None]).min())  # ponto mais baixo da sola, no mundo
    origem, R = _corpo(m, d, "left_ankle_roll_link")
    loc = (sola - origem) @ R  # cantos no frame do tornozelo
    lo = (loc[..., :2] - r_sola[:, None, None]).reshape(-1, 2).min(0)
    hi = (loc[..., :2] + r_sola[:, None, None]).reshape(-1, 2).max(0)
    z0 = float((loc[..., 2] - r_sola[:, None]).min())
    cantos = np.array([[lo[0], lo[1], z0], [hi[0], lo[1], z0], [hi[0], hi[1], z0], [lo[0], hi[1], z0]])

    topos, meias, grade, aceita = _grade(ag)
    caixa = np.array([_centro_na_pelve(m, q) for q in tm["qpos"]])
    palmas_caixa = np.array([_na_pelve(m, _fk(m, q), PALMAS) for q in tm["qpos"]])
    print(f"pelve de pé: {qde[2]:.4f} m; site left_foot a {(_sitio(m, d, 'left_foot')[2] - sola_z)*1000:.1f} mm da sola")

    np.savez(
        SAIDA,
        topos=topos, meias=meias, **grade, grade_aceita=aceita,
        caixa_transporte_b=caixa, palmas_com_caixa_b=palmas_caixa,
        q_com_caixa=tm["qpos"][:, 7:36], pelve_com_caixa=tm["qpos"][:, 2],
        q_de_pe=qde[7:36], palmas_de_pe_b=_na_pelve(m, d, PALMAS), pes_de_pe_b=_na_pelve(m, d, PES),
        cantos_sola=cantos, pelve_de_pe=qde[2],
        site_pe_de_pe=_sitio(m, d, "left_foot")[2] - sola_z,
        fonte=np.array(f"{AGACHADO.relative_to(RAIZ)}; {TAMANHOS.relative_to(RAIZ)}; {DE_PE.relative_to(RAIZ)}"),
        hash_origem=np.array(str(ag["hash_mjcf"])),
    )
    return len(grade)


if __name__ == "__main__":
    print(f"referências gravadas em {SAIDA}")
    gera()
