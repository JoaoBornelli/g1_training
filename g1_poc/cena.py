"""A cena do g1_poc: robô com pads, caixa, prateleira mocap, e os sensores.

Nada aqui é novo. Tudo vem de `g1_training/common/`, que já rodou.

⚠ NÃO editar `g1_training/`. Este pacote só CONSOME.

Rodar como smoke de geometria (compila as duas entidades, sem mjlab RL):
    python -m g1_poc.cena
"""
from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from g1_training.common.box import TABLE_GEOM, get_box_spec, get_shelf_spec
from g1_training.common.robot import (
    BACK_PAD_GEOMS,
    PALM_PAD_GEOMS,
    PALM_SITES,
    get_lift_box_robot_cfg,
)

from g1_poc.knobs import Knobs

__all__ = [
    "PALM_SITES", "PALM_PAD_GEOMS", "BACK_PAD_GEOMS", "TABLE_GEOM",
    "SENSOR_PALMA", "SENSOR_DORSO", "SENSOR_APOIO", "SENSOR_CORPO_PRATELEIRA",
    "SENSOR_AUTO_COLISAO", "SENSOR_PES",
    "regroup", "entidades", "sensores", "robot_cfg",
]

SENSOR_PALMA = ("palma_E", "palma_D")
SENSOR_DORSO = ("dorso_E", "dorso_D")
SENSOR_APOIO = "apoio_caixa"
SENSOR_CORPO_PRATELEIRA = "corpo_prateleira"
SENSOR_AUTO_COLISAO = "auto_colisao"
SENSOR_PES = "pes_chao"

# Corpos que NÃO podem escorar na prateleira. O antebraço, a mão e o pé ficam de
# fora: numa pega a 0.04 m o antebraço passa perto do tampo, e esse contato é
# normal. Escorar o TRONCO ou a COXA é o defeito medido no repo.
#
# ⚠ A CANELA também fica de fora, e por medida: `prateleira_topo_piso = 0.04`, então
# no nível mais baixo do currículo a laje é um degrau de 4 cm na frente dos pés.
# Cobrar canela/pé ali repetiria o A11 do multitask — o robô pagava por PISAR na
# prateleira, que é o contato que o `feet_slip` precisa ver.
#
# Os nomes são os do modelo, não os do URDF: o G1 não tem geom de colisão de cintura
# nem de joelho (o tronco cobre a cintura; a perna vai de `thigh` a `shin`), e o do
# quadril é `left_hip_collision` — sem segundo `_`. Padrão que casa zero geom é
# ValueError no `resolve_matching_names`, não aviso.
CORPOS_QUE_NAO_ESCORAM = (
    r"pelvis_collision",
    r"torso_collision",
    r".*_hip_collision",
    r".*_thigh_collision",
)


def regroup(spec: mujoco.MjSpec, group: int) -> mujoco.MjSpec:
    """Põe todos os geoms do spec num grupo de geom.

    `common/box.py` não passa `group=`, portanto a mobília nasce no grupo 0. O
    `foot_height_scan` do fabricante usa `include_geom_groups=(0,)` e leria a
    prateleira COMO CHÃO. Ver §3.3.
    """
    for geom in spec.geoms:
        geom.group = group
    return spec


def robot_cfg() -> EntityCfg:
    """O G1 com os pads de palma e de dorso. Sem mudança de física."""
    return get_lift_box_robot_cfg()


def entidades(k: Knobs) -> dict[str, EntityCfg]:
    c = k.cena
    topo = c.prateleira_topo_teto
    centro_prateleira_z = topo - c.prateleira_meia_z
    caixa_z = topo + c.caixa_meia_aresta[2]
    meia_prateleira = (c.prateleira_meia_xy, c.prateleira_meia_xy, c.prateleira_meia_z)
    grupo = c.grupo_mobilia

    return {
        "robot": robot_cfg(),
        "box": EntityCfg(
            spec_fn=lambda: regroup(
                get_box_spec(c.caixa_meia_aresta, c.caixa_massa), grupo),
            init_state=EntityCfg.InitialStateCfg(
                pos=(c.caixa_xy[0], c.caixa_xy[1], caixa_z)),
        ),
        "table": EntityCfg(
            spec_fn=lambda: regroup(get_shelf_spec(meia_prateleira), grupo),
            init_state=EntityCfg.InitialStateCfg(
                pos=(c.prateleira_xy[0], c.prateleira_xy[1], centro_prateleira_z)),
        ),
    }


def sensores() -> tuple[ContactSensorCfg, ...]:
    """Os 6 sensores da §3.5.

    `force` é pedido onde a MAGNITUDE importa: as palmas (o `squeeze`), o apoio
    (o fecho do `botar`), a prateleira (o `contato_ilegal`) e a auto-colisão.
    """
    palmas = tuple(
        ContactSensorCfg(
            name=nome,
            primary=ContactMatch(mode="geom", pattern=pad, entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
            # ⚠ `reduce="netforce"` soma todos os contatos num wrench só, e a força
            # sai no frame GLOBAL. Portanto o campo `normal` NÃO é pedido: com a
            # redução em netforce ele perde significado (qual das normais?). O
            # `squeeze` calcula a normal da palma da orientação do site, que é
            # exata. Ver `recompensas.squeeze`.
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        )
        for nome, pad in zip(SENSOR_PALMA, PALM_PAD_GEOMS)
    )
    dorsos = tuple(
        ContactSensorCfg(
            name=nome,
            primary=ContactMatch(mode="geom", pattern=pad, entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        )
        for nome, pad in zip(SENSOR_DORSO, BACK_PAD_GEOMS)
    )
    apoio = ContactSensorCfg(
        name=SENSOR_APOIO,
        primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
        secondary=ContactMatch(mode="geom", pattern=TABLE_GEOM, entity="table"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    corpo = ContactSensorCfg(
        name=SENSOR_CORPO_PRATELEIRA,
        primary=ContactMatch(
            mode="geom", pattern=CORPOS_QUE_NAO_ESCORAM, entity="robot"),
        secondary=ContactMatch(mode="geom", pattern=TABLE_GEOM, entity="table"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    auto = ContactSensorCfg(
        name=SENSOR_AUTO_COLISAO,
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    pes = ContactSensorCfg(
        name=SENSOR_PES,
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        # secondary=None -> qualquer contato conta como chão. Pisar na prateleira
        # conta, e é o que queremos: sem isto o `feet_slip` fica cego.
        secondary=None,
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    return palmas + dorsos + (apoio, corpo, auto, pes)


if __name__ == "__main__":
    k = Knobs()
    c = k.cena
    meia = (c.prateleira_meia_xy, c.prateleira_meia_xy, c.prateleira_meia_z)
    for nome, spec in (
        ("caixa", regroup(get_box_spec(c.caixa_meia_aresta, c.caixa_massa), c.grupo_mobilia)),
        ("prateleira", regroup(get_shelf_spec(meia), c.grupo_mobilia)),
    ):
        m = spec.compile()
        grupos = sorted({int(m.geom_group[i]) for i in range(m.ngeom)})
        print(f"{nome:11s} compilou  ngeom={m.ngeom}  nq={m.nq}  grupos={grupos}")
    print()
    print("geometria da §18:")
    print(f"  topo da prateleira no piso      = {c.prateleira_topo_piso:.2f} m")
    print(f"  fundo da laje nesse piso        = {c.prateleira_topo_piso - 2*c.prateleira_meia_z:+.2f} m")
    print(f"  centro da caixa no piso         = {c.prateleira_topo_piso + c.caixa_meia_aresta[2]:.2f} m")
    print(f"  a prateleira ocupa x de         = {c.prateleira_xy[0]-c.prateleira_meia_xy:.2f} a "
          f"{c.prateleira_xy[0]+c.prateleira_meia_xy:.2f} m")
    print(f"  a caixa nasce em x              = {c.caixa_xy[0]:.2f} m")
    print("OK: as duas entidades compilam e estão fora do grupo 0")
