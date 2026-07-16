"""G1 refinado pra manipulação: substitui a cápsula da mão por pads de
palma (face, com friction pra segurar) e de dorso (para detectar/bloquear
contato do verso). Herda o resto de get_g1_robot_cfg()."""
from __future__ import annotations

import mujoco

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ARTICULATION,
    KNEES_BENT_KEYFRAME,
    get_spec,
)
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

PALM_SITES = ("left_palm", "right_palm")
PALM_PAD_GEOMS = ("left_palm_pad", "right_palm_pad")
BACK_PAD_GEOMS = ("left_hand_back_pad", "right_hand_back_pad")

# Geometria CONFIRMADA por render (2026-07-13): o frame da mão (wrist_yaw_link) é
# ~alinhado ao mundo em qpos0; a mão é uma pá chata cujas faces largas (palma/dorso)
# apontam em ±Z LOCAL. Palma (lado onde os dedos curvam) = -Z; dorso = +Z. Os pads
# são offsetados em Z (o bug que você viu no play era offset em Y → pads na borda).
_PAD_HALF = (0.035, 0.008, 0.045)  # laje fina no plano XY (thin em z), cobre a palma
_PALM_DZ = 0.015   # offset ao longo da normal da palma (Z local)
_PALM_X = 0.10    # ao longo da mão, cobrindo a região da palma


def _add_pad(hand_body, name: str, dz: float, condim: int) -> None:
    hand_body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(_PALM_X, dz, 0.0),
        size=_PAD_HALF,
        condim=condim,
        friction=(1.0, 0.02, 0.001),
        rgba=(0.1, 0.6, 0.1, 0.6) if "palm" in name else (0.6, 0.1, 0.1, 0.4),
    )


def add_palm_pads(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Remove as cápsulas *_hand_collision e adiciona pads de palma/dorso.

    Palma = face -Z local (lado dos dedos), dorso = +Z local; igual pros dois
    wrists (frames simétricos em qpos0). Confirmado por render.

    Nota de API (mujoco 3.10 / mjlab 1.5.0): o método de remoção de um nó vive
    no MjSpec, não no nó — é `spec.delete(geom)`, não `geom.delete()`. `site.parent`
    funciona como documentado no brief (retorna o MjsBody dono do site).
    """
    # Remover as cápsulas de colisão originais (radialmente simétricas: não
    # distinguem palma de verso, e deixariam o verso tocar a caixa).
    for geom in list(spec.geoms):
        if geom.name in ("left_hand_collision", "right_hand_collision"):
            spec.delete(geom)

    # Encontrar os bodies das mãos (o site *_palm mora neles).
    for site_name, palm_pad, back_pad in (
        ("left_palm", "left_palm_pad", "left_hand_back_pad"),
        ("right_palm", "right_palm_pad", "right_hand_back_pad"),
    ):
        if site_name == "right_palm":
            site = spec.site(site_name)
            hand_body = site.parent
            _add_pad(hand_body, palm_pad, dz=+_PALM_DZ, condim=3)  # palma: -Z local
            _add_pad(hand_body, back_pad, dz=-_PALM_DZ, condim=1)  # dorso: +Z local
        else:
            site = spec.site(site_name)
            hand_body = site.parent
            _add_pad(hand_body, palm_pad, dz=-_PALM_DZ, condim=3)  # palma: -Z local
            _add_pad(hand_body, back_pad, dz=+_PALM_DZ, condim=1)  # dorso: +Z local
    return spec


def get_lift_box_robot_cfg() -> EntityCfg:
    """G1 (get_g1_robot_cfg) com as mãos refinadas + os pads na CollisionCfg.

    Os pads terminam em '_pad', então NÃO batem em '.*_collision' do
    FULL_COLLISION padrão — precisamos incluí-los explicitamente (condim já
    vem setado no geom; a CollisionCfg garante que participam da colisão)."""
    collisions = CollisionCfg(
        geom_names_expr=(".*_collision", ".*_pad"),
        condim={
            r"^(left|right)_foot[1-7]_collision$": 3,
            r".*_palm_pad$": 3,
            r".*_hand_back_pad$": 1,
            ".*_collision": 1,
        },
        priority={r"^(left|right)_foot[1-7]_collision$": 1},
        friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
    )

    def spec_fn() -> mujoco.MjSpec:
        return add_palm_pads(get_spec())

    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(collisions,),
        spec_fn=spec_fn,
        articulation=G1_ARTICULATION,
    )
