"""Folhas de contato das poses de IK na prateleira: uma de lado, uma de frente.

Cinco linhas (lajes) x três colunas (caixas). As paredes do cubo entram como geoms de
desenho, semitransparentes, porque o `cena.mjb` não as tem.

    PYTHONPATH=$PWD python foto_prateleira.py prat_C.npz pasta_saida
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

import g1_limpo.ik.gera_botar as G
from g1_limpo.ik.ve_botar import _grava_png

W, H = 400, 300


def folha(c, ref, azimute, nome):
    r = mujoco.Renderer(c.m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 2.4, azimute, -10.0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[2] = 1
    n = len(ref["topo"])
    tiles = []
    for k in range(n):
        topo, a = float(ref["topo"][k]), float(ref["meia"][k])
        c.cenario(topo, a)
        c.d.qpos[:] = ref["qpos"][k]
        mujoco.mj_forward(c.m, c.d)
        cam.lookat[:] = (0.15, 0.0, 0.55)
        r.update_scene(c.d, camera=cam, scene_option=opt)
        for linha in ref["paredes"][k]:
            g = r.scene.geoms[r.scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, linha[3:6], linha[0:3],
                                np.eye(3).ravel(), np.array([0.55, 0.45, 0.30, 0.35], dtype=np.float32))
            r.scene.ngeom += 1
        img = r.render().copy()
        cor = (0, 160, 0) if bool(ref["aceita"][k]) else (200, 0, 0)
        img[:6, :] = cor
        img[-6:, :] = cor
        tiles.append(img)
    r.close()
    linhas = [np.concatenate(tiles[i:i + 3], axis=1) for i in range(0, n, 3)]
    out = np.concatenate(linhas, axis=0)
    _grava_png(nome, out)
    print(f"[folha] {nome}")


def main():
    arq, pasta = Path(sys.argv[1]), Path(sys.argv[2]).expanduser()
    pasta.mkdir(parents=True, exist_ok=True)
    c = G.Cena(Path("~/g1_pilota/").expanduser())
    ref = np.load(arq)
    base = arq.stem
    folha(c, ref, 90.0, pasta / f"{base}_lado.png")
    folha(c, ref, 150.0, pasta / f"{base}_frente.png")


if __name__ == "__main__":
    main()
