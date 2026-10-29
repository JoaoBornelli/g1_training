"""Gera g1_v3/dados/capsulas.npz: uma cápsula por geom de colisão do g1.xml do mjlab.

As cápsulas são o filtro do próprio corpo (PERC §7): o raio do treino não pode ver o robô
como alvo. Só o grupo 3 com contype != 0 entra (as geoms de colisão). Não há física aqui:
o modelo é compilado só para ler tamanhos, tipos e poses (PD-3).

Cada cápsula fica no frame do corpo da geom. Esfera tem segmento nulo; cápsula e cilindro
seguem o eixo z local; caixa e malha viram a cápsula envolvente no maior eixo, com raio =
maior meia extensão dos outros dois (a malha usa a caixa envolvente dos vértices).
"""
import importlib.util
from pathlib import Path

import mujoco
import numpy as np

SAIDA = Path(__file__).resolve().parent / "capsulas.npz"
ESFERA, CAPSULA, CILINDRO, CAIXA, MALHA = 2, 3, 5, 6, 7  # mjtGeom do MuJoCo
BRACO = ("shoulder", "elbow", "wrist", "hand")  # S §13: ombros, cotovelos, punhos, mãos


def _xml_g1():
    spec = importlib.util.find_spec("mjlab")  # acha a pasta sem executar o pacote
    assert spec is not None and spec.submodule_search_locations, "mjlab não instalado"
    return Path(list(spec.submodule_search_locations)[0]) / "asset_zoo/robots/unitree_g1/xmls/g1.xml"


def _forma(m, g):
    """(centro, meia do segmento, raio, eixo) no frame local da geom; segmento = centro ± meia·eixo."""
    t, s = m.geom_type[g], m.geom_size[g]
    z = np.array([0.0, 0.0, 1.0])
    if t == ESFERA:
        return np.zeros(3), 0.0, s[0], z
    if t in (CAPSULA, CILINDRO):
        return np.zeros(3), s[1], s[0], z
    if t == CAIXA:
        meia, centro = s[:3], np.zeros(3)
    else:  # malha: caixa envolvente dos vértices, no frame da geom
        i, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
        v = m.mesh_vert[i:i + n]
        meia, centro = (v.max(0) - v.min(0)) / 2, (v.max(0) + v.min(0)) / 2
    k = int(np.argmax(meia))  # maior eixo vira o segmento; o raio é o maior dos outros dois
    eixo = np.eye(3)[k]
    r = float(np.max(np.delete(meia, k)))
    return centro, max(float(meia[k]) - r, 0.0), r, eixo


def gera():
    m = mujoco.MjModel.from_xml_path(str(_xml_g1()))
    corpos, p0, p1, raio, braco = [], [], [], [], []
    for g in range(m.ngeom):
        if m.geom_group[g] != 3 or m.geom_contype[g] == 0:
            continue
        nome = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        centro, meia, r, eixo = _forma(m, g)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        R = R.reshape(3, 3)
        pos = m.geom_pos[g]
        corpos.append(nome)
        p0.append(pos + R @ (centro - meia * eixo))
        p1.append(pos + R @ (centro + meia * eixo))
        raio.append(r)
        braco.append(any(b in nome for b in BRACO))
    np.savez(SAIDA, corpo=np.array(corpos), p0=np.array(p0), p1=np.array(p1),
             raio=np.array(raio), braco=np.array(braco, dtype=bool))
    return len(corpos)


if __name__ == "__main__":
    print(f"{gera()} cápsulas -> {SAIDA}")
