"""Entidades da cena como specs do mjlab: a CAIXA (leve, a erguer) e a MESA
(pesada, suporte). Cada uma é um corpo LIVRE (free joint) num MjSpec PRÓPRIO.

Diferença pro scene.py de preview: lá a caixa e a mesa eram bodies dentro do
MESMO spec do robô, só pra visualizar num modelo só. No mjlab, cada objeto é
uma ENTIDADE separada (um MjSpec) — o mjlab combina as entidades numa cena e
as replica por-ambiente. Por isso aqui cada função devolve seu próprio spec.

Rodar como smoke (compila as duas):  python g1_training/common/box.py
"""
from __future__ import annotations

import mujoco

BOX_GEOM = "box_geom"
TABLE_GEOM = "table_geom"


def _free_box(body, geom, joint, half, mass, rgba, condim=3) -> mujoco.MjSpec:
    """Um box primitivo como corpo livre (free joint) num MjSpec próprio."""
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name=body)
    b.add_freejoint(name=joint)                 # 7 DOF: pode transladar+girar livre
    b.add_geom(
        name=geom, type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(half), mass=mass,
        condim=condim, friction=(1.0, 0.02, 0.001), rgba=rgba,
    )
    return spec


def get_box_spec(half=(0.10, 0.10, 0.10), mass: float = 1.0) -> mujoco.MjSpec:
    """Caixa LEVE a erguer (~0.20 m de lado, 1 kg)."""
    return _free_box("box", BOX_GEOM, "box_joint", half, mass, (0.8, 0.5, 0.2, 1.0))


def get_table_spec(half=(0.30, 0.30, 0.275), mass: float = 20.0) -> mujoco.MjSpec:
    """MESA = caixa PESADA e larga (corpo livre; pesada → fica no lugar quando tocada)."""
    return _free_box("table", TABLE_GEOM, "table_joint", half, mass, (0.5, 0.5, 0.55, 1.0))


if __name__ == "__main__":
    for name, spec in (("box", get_box_spec()), ("table", get_table_spec())):
        m = spec.compile()
        print(f"{name:5s}: compilou  ngeom={m.ngeom}  nq={m.nq} (7=free joint)  "
              f"massa={m.body_mass.sum():.2f} kg")
    print("OK brick 1: caixa e mesa compilam como entidades separadas")
