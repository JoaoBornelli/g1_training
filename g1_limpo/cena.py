"""A cena do g1_limpo: robô com pads de palma, caixa, prateleira mocap, sensores.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Tudo aqui foi TRANSCRITO de:
    g1_training/common/box.py     (specs da caixa e da prateleira)
    g1_training/common/robot.py   (pads de palma e de dorso)
    g1_training/base_env.py       (montagem das entidades)
    g1_poc/cena.py                (os 6 sensores, o regroup, os corpos que não escoram)

O `paridade.py` prova que a transcrição está correta comparando o `mjModel` compilado.

Rodar como smoke de geometria (compila as duas entidades, sem mjlab RL):
    python -m g1_limpo.cena
"""
from __future__ import annotations

import mujoco

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ARTICULATION,
    KNEES_BENT_KEYFRAME,
    get_spec,
)
from mjlab.entity import EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from g1_limpo.knobs import Cena, Knobs

__all__ = [
    "BOX_GEOM", "TABLE_GEOM",
    "PALM_SITES", "PALM_PAD_GEOMS", "BACK_PAD_GEOMS", "FOOT_SITES",
    "SENSOR_PALMA", "SENSOR_DORSO", "SENSOR_APOIO", "SENSOR_CORPO_PRATELEIRA",
    "SENSOR_AUTO_COLISAO", "SENSOR_PES",
    "spec_caixa", "spec_prateleira", "add_pads_de_palma", "robot_cfg",
    "regroup", "entidades", "sensores", "geometria_de_repouso",
]

# ---------------------------------------------------------------- nomes de geom
BOX_GEOM = "box_geom"
TABLE_GEOM = "table_geom"

# ⚠ O nome do geom da prateleira é `table_geom` e o do body é `table`, apesar de a
# entidade ser uma LAJE e não uma mesa. Mantido de propósito: é o nome que os
# padrões de sensor casam, e mudar aqui quebraria a paridade sem ganho nenhum.

# ------------------------------------------------------------- sites e pads
PALM_SITES = ("left_palm", "right_palm")
PALM_PAD_GEOMS = ("left_palm_pad", "right_palm_pad")
BACK_PAD_GEOMS = ("left_hand_back_pad", "right_hand_back_pad")
FOOT_SITES = ("left_foot", "right_foot")

# Geometria CONFIRMADA por render (2026-07-13): o frame da mão (`wrist_yaw_link`) é
# ~alinhado ao mundo em qpos0; a mão é uma pá chata cujas faces largas apontam em
# ±Z LOCAL. Os pads são offsetados em Z — offset em Y punha os pads na BORDA, que
# foi um bug real.
_PAD_HALF = (0.035, 0.008, 0.045)   # laje fina, cobre a palma
_PALM_DZ = 0.015                    # offset ao longo da normal da palma
_PALM_X = 0.10                      # ao longo da mão, na região da palma

# ------------------------------------------------------------------- sensores
SENSOR_PALMA = ("palma_E", "palma_D")
SENSOR_DORSO = ("dorso_E", "dorso_D")
SENSOR_APOIO = "apoio_caixa"
SENSOR_CORPO_PRATELEIRA = "corpo_prateleira"
SENSOR_AUTO_COLISAO = "auto_colisao"
SENSOR_PES = "pes_chao"

# Corpos que NÃO podem escorar na prateleira. O antebraço, a mão e o pé ficam de
# fora: numa pega a 0,04 m o antebraço passa perto do tampo, e esse contato é
# normal. Escorar o TRONCO ou a COXA é o defeito medido no repo.
#
# ⚠ A CANELA também fica de fora, e por medida: com `prateleira_topo_piso = 0.04` a
# laje é um degrau de 4 cm na frente dos pés. Cobrar canela ou pé ali faria o robô
# pagar por PISAR na prateleira, que é exatamente o contato que o `pes_chao` precisa
# ver.
#
# ⚠ Os nomes são os do MODELO, não os do URDF: o G1 não tem geom de colisão de
# cintura nem de joelho (o tronco cobre a cintura; a perna vai de `thigh` a `shin`),
# e o do quadril é `left_hip_collision` — sem segundo `_`. Um padrão que casa ZERO
# geom levanta `ValueError` no `resolve_matching_names`, não um aviso.
CORPOS_QUE_NAO_ESCORAM = (
    r"pelvis_collision",
    r"torso_collision",
    r".*_hip_collision",
    r".*_thigh_collision",
)


# ============================================================ specs de entidade
def _spec_box(body: str, geom: str, joint: str | None, half, mass, rgba,
              condim: int, atrito) -> mujoco.MjSpec:
    """Um box primitivo num MjSpec PRÓPRIO.

    No mjlab cada objeto é uma ENTIDADE separada (um MjSpec), e o mjlab combina as
    entidades numa cena e as replica por ambiente.

    `joint=None` -> corpo SEM free joint -> o mjlab auto-envolve em MOCAP.
    """
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name=body)
    if joint is not None:
        b.add_freejoint(name=joint)       # 7 DoF: transladar e girar livre
    kwargs = dict(
        name=geom, type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(half),
        condim=condim, friction=tuple(atrito), rgba=tuple(rgba),
    )
    if mass is not None:
        kwargs["mass"] = mass
    b.add_geom(**kwargs)
    return spec


def spec_caixa(c: Cena) -> mujoco.MjSpec:
    """Caixa LEVE a erguer: corpo LIVRE, ~0,20 m de lado, 1 kg."""
    return _spec_box("box", BOX_GEOM, "box_joint", c.caixa_meia_aresta,
                     c.caixa_massa, c.caixa_rgba, c.caixa_condim, c.caixa_atrito)


def spec_prateleira(c: Cena) -> mujoco.MjSpec:
    """PRATELEIRA fina, SEM free joint -> o mjlab auto-envolve em MOCAP.

    Consequências, e as três são o motivo de ela ser assim:
      - corpo CINEMÁTICO, posicionável por env em runtime (`write_mocap_pose_to_sim`)
      - flutua em qualquer z SEM tocar o chão
      - não é movida por contato, portanto dispensa massa

    A caixa (livre) repousa em cima, e continua ERGUÍVEL. E ela é FINA em z: é uma
    prateleira, não um paredão — o que mata o atalho de escorar o robô nela.
    """
    half = (c.prateleira_meia_xy, c.prateleira_meia_xy, c.prateleira_meia_z)
    return _spec_box("table", TABLE_GEOM, None, half, None,
                     c.prateleira_rgba, c.prateleira_condim, c.prateleira_atrito)


# ===================================================================== o robô
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


def add_pads_de_palma(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Remove as cápsulas `*_hand_collision` e acrescenta pads de palma e de dorso.

    As cápsulas originais são radialmente simétricas: elas não distinguem palma de
    verso, e deixariam o DORSO tocar a caixa e contar como pega.

    ⚠ Nota de API (mujoco 3.10 / mjlab 1.5): o método de remoção de um nó vive no
    MjSpec, não no nó — é `spec.delete(geom)`, e NUNCA `geom.delete()`.

    ⚠ DISCREPÂNCIA NA REFERÊNCIA, e eu transcrevi o CÓDIGO, não o comentário.
    `g1_training/common/robot.py:65-71` comenta "palma: -Z local" para as DUAS mãos,
    mas o código põe o pad de palma da DIREITA em `+_PALM_DZ` e o da ESQUERDA em
    `-_PALM_DZ`. Os frames dos dois punhos são simétricos em qpos0, portanto o sinal
    OPOSTO é o que produz palmas voltadas uma para a outra — o comentário é que está
    errado. O `paridade.py` compara `geom_pos` e travaria se eu tivesse copiado o
    comentário.
    """
    for geom in list(spec.geoms):
        if geom.name in ("left_hand_collision", "right_hand_collision"):
            spec.delete(geom)

    # (site, pad de palma, pad de dorso, sinal do dz da palma)
    for site_name, palm_pad, back_pad, sinal in (
        ("left_palm", "left_palm_pad", "left_hand_back_pad", -1.0),
        ("right_palm", "right_palm_pad", "right_hand_back_pad", +1.0),
    ):
        hand_body = spec.site(site_name).parent
        _add_pad(hand_body, palm_pad, dz=sinal * _PALM_DZ, condim=3)
        _add_pad(hand_body, back_pad, dz=-sinal * _PALM_DZ, condim=1)
    return spec


def robot_cfg() -> EntityCfg:
    """O G1 com as mãos refinadas, e os pads dentro da `CollisionCfg`.

    ⚠ Os pads terminam em `_pad`, portanto NÃO casam com `.*_collision` do
    FULL_COLLISION padrão — eles precisam entrar explicitamente, senão não
    participam da colisão e os sensores de palma nunca disparam.
    """
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
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(collisions,),
        spec_fn=lambda: add_pads_de_palma(get_spec()),
        articulation=G1_ARTICULATION,
    )


# ================================================================== a cena
def regroup(spec: mujoco.MjSpec, group: int) -> mujoco.MjSpec:
    """Põe todos os geoms do spec num grupo de geom.

    ⚠ Os specs da mobília nascem no grupo 0, e o `foot_height_scan` do fabricante
    usa `include_geom_groups=(0,)` — ele leria a prateleira COMO CHÃO, e o robô
    "veria" um degrau que na verdade é a mesa dele.
    """
    for geom in spec.geoms:
        geom.group = group
    return spec


def geometria_de_repouso(c: Cena) -> dict[str, float]:
    """As alturas derivadas, num lugar só. O `inspeciona.py` confere estas contas.

    A prateleira é MOCAP e a sua pose é o CENTRO do corpo, não o topo. Portanto o
    centro fica sempre `meia_z` abaixo do topo pedido.
    """
    topo = c.prateleira_topo_teto
    return {
        "topo": topo,
        "centro_prateleira_z": topo - c.prateleira_meia_z,
        "fundo_prateleira_z": topo - 2.0 * c.prateleira_meia_z,
        "caixa_z": topo + c.caixa_meia_aresta[2],
    }


def entidades(k: Knobs) -> dict[str, EntityCfg]:
    c = k.cena
    g = geometria_de_repouso(c)
    return {
        "robot": robot_cfg(),
        "box": EntityCfg(
            spec_fn=lambda: regroup(spec_caixa(c), c.grupo_mobilia),
            init_state=EntityCfg.InitialStateCfg(
                pos=(c.caixa_xy[0], c.caixa_xy[1], g["caixa_z"])),
        ),
        "table": EntityCfg(
            spec_fn=lambda: regroup(spec_prateleira(c), c.grupo_mobilia),
            init_state=EntityCfg.InitialStateCfg(
                pos=(c.prateleira_xy[0], c.prateleira_xy[1], g["centro_prateleira_z"])),
        ),
    }


def sensores() -> tuple[ContactSensorCfg, ...]:
    """Os 6 sensores.

    `force` é pedido onde a MAGNITUDE importa: as palmas (o `squeeze`), o apoio (a
    ponte do `unload` e o fecho do `botar`), a prateleira (o contato ilegal) e a
    auto-colisão.

    ⚠ Este é o ponto que o `g1_training/base_env.py` NÃO resolve: o
    `_pad_contact_sensor` de lá pede `fields=("found",)`, sem força, e com isso o
    `squeeze` e o `unload` são IMPOSSÍVEIS de escrever. Aqui a força entra desde o
    início.

    ⚠ `reduce="netforce"` soma todos os contatos num wrench só, e a força sai no
    frame GLOBAL. Portanto o campo `normal` NÃO é pedido: com a redução em netforce
    ele perde significado (qual das normais?). Quem precisa da normal da palma a
    calcula da orientação do site, que é exata.
    """
    palmas = tuple(
        ContactSensorCfg(
            name=nome,
            primary=ContactMatch(mode="geom", pattern=pad, entity="robot"),
            secondary=ContactMatch(mode="geom", pattern=BOX_GEOM, entity="box"),
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
            secondary=ContactMatch(mode="geom", pattern=BOX_GEOM, entity="box"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        )
        for nome, pad in zip(SENSOR_DORSO, BACK_PAD_GEOMS)
    )
    apoio = ContactSensorCfg(
        name=SENSOR_APOIO,
        primary=ContactMatch(mode="geom", pattern=BOX_GEOM, entity="box"),
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
        # ⚠ `secondary=None` -> QUALQUER contato conta como chão. Pisar na
        # prateleira conta, e é o que queremos: sem isto o slip do pé fica cego
        # justamente no nível em que a laje é um degrau de 4 cm.
        secondary=None,
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    return palmas + dorsos + (apoio, corpo, auto, pes)


# ===================================================================== smoke
if __name__ == "__main__":
    k = Knobs()
    c = k.cena
    for nome, spec in (("caixa", regroup(spec_caixa(c), c.grupo_mobilia)),
                       ("prateleira", regroup(spec_prateleira(c), c.grupo_mobilia))):
        m = spec.compile()
        grupos = sorted({int(m.geom_group[i]) for i in range(m.ngeom)})
        print(f"{nome:11s} compilou  ngeom={m.ngeom}  nq={m.nq}  grupos={grupos}")

    g = geometria_de_repouso(c)
    print()
    print("geometria de repouso:")
    print(f"  topo da prateleira            = {g['topo']:.3f} m")
    print(f"  centro do corpo mocap         = {g['centro_prateleira_z']:.3f} m")
    print(f"  fundo da laje                 = {g['fundo_prateleira_z']:+.3f} m")
    print(f"  centro da caixa               = {g['caixa_z']:.3f} m")
    print(f"  no PISO do currículo, o fundo = "
          f"{c.prateleira_topo_piso - 2*c.prateleira_meia_z:+.3f} m")
    print(f"  a prateleira ocupa x de       = "
          f"{c.prateleira_xy[0]-c.prateleira_meia_xy:.2f} a "
          f"{c.prateleira_xy[0]+c.prateleira_meia_xy:.2f} m")
    print(f"  a caixa nasce em x            = {c.caixa_xy[0]:.2f} m")
    print()
    print("OK: as duas entidades compilam e estão fora do grupo 0")
