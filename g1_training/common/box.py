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
    """MESA = caixa PESADA e larga (corpo livre; pesada → fica no lugar quando tocada).
    LEGADO (free-body). A cena nova usa `get_shelf_spec` (prateleira mocap)."""
    return _free_box("table", TABLE_GEOM, "table_joint", half, mass, (0.5, 0.5, 0.55, 1.0))


def get_shelf_spec(half=(0.30, 0.30, 0.02)) -> mujoco.MjSpec:
    """PRATELEIRA fina FIXA (sem free joint) → o mjlab auto-wrappa em MOCAP
    (`auto_wrap_fixed_base_mocap`): corpo cinemático, posicionável por-env em runtime
    (write_mocap_pose), flutua em qualquer z SEM tocar o chão (mata o clip de fundo que
    travava o `table_half` compile-time) e não é movido por contato (dispensa massa).
    A caixa (livre) repousa em cima → continua LEVANTÁVEL. Fina em z = plateleira, não
    paredão (mata o crutch de escorar). Mesmo geom name (TABLE_GEOM) → sensores
    box_support/body_table seguem casando."""
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name="table")          # SEM add_freejoint -> fixed -> mocap
    b.add_geom(
        name=TABLE_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(half),
        condim=3, friction=(1.0, 0.02, 0.001), rgba=(0.5, 0.5, 0.55, 1.0),
    )
    return spec


if __name__ == "__main__":
    for name, spec in (("box", get_box_spec()), ("shelf", get_shelf_spec())):
        m = spec.compile()
        print(f"{name:5s}: compilou  ngeom={m.ngeom}  nq={m.nq}  (box=7 free / shelf=0 fixed)")
    print("OK brick 1: caixa (livre) e prateleira (fixa->mocap) compilam")
