"""A cena da v3: robô G1 com o PD da Unitree, pads de palma e de dorso, sites do lidar e da D435i,
caixa, laje, terreno por colunas e sensores (plano D1, §6.1 e §6.2).

⚠ ZERO IMPORT DO g1_limpo. As peças são COPIAS de g1_limpo/cena.py, com as mudanças da D1 sobre a cópia:
o PD e os atuadores da Unitree (I §3), o atrito por material (PD-46), a mobília no grupo 1 (PD-9),
o terreno por colunas com teto de CARREGAR (PD-20), os sensores de altura (PD-21) e o lidar (PERC §2).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION, get_spec
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import (ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg, RingPatternCfg,
                          TerrainHeightSensorCfg)
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import (discrete_obstacles, flat, hf_pyramid_slope, hf_pyramid_slope_inv,
                                   pyramid_stairs, pyramid_stairs_inv, random_rough, random_spread_boxes,
                                   random_stairs)
from mjlab.utils.spec_config import CollisionCfg

from g1_v3.knobs import ATIVO, JUNTAS, Cena, Knobs, Terreno, caminho, q_de_pe

# ------------------------------------------------------------------ nomes e grupos
BOX_GEOM = "box_geom"
# ⚠ O geom da LAJE se chama `table_geom` e o corpo `table`, apesar de a entidade ser uma laje.
# É o nome que os padrões de sensor casam; mudar quebraria a paridade (g1_limpo/cena.py:33-37).
TABLE_GEOM = "table_geom"
PALM_PAD_GEOMS = ("left_palm_pad", "right_palm_pad")
BACK_PAD_GEOMS = ("left_hand_back_pad", "right_hand_back_pad")

# Grupos de geom. A mobília sai do grupo 0 (o foot_height_scan lê o 0 como chão; PD-9).
# Os pads ficam no grupo 3 para o lidar vê-los (`include_geom_groups` = (0, 1, 3)).
_GRUPO_PAD = 3

# Os três sensores de mesa, um por grupo de geom: com `reduce="netforce"` um sensor só diz "encostou"
# e não diz com o quê. A união dos três é `CORPOS_QUE_NAO_ESCORAM`.
GRUPO_TRONCO = (r"pelvis_collision", r"torso_collision", r".*_hip_collision", r".*_thigh_collision")
GRUPO_PALMA = (r".*_palm_pad",)
GRUPO_DORSO = (r".*_hand_back_pad",)
CORPOS_QUE_NAO_ESCORAM = GRUPO_TRONCO + GRUPO_PALMA + GRUPO_DORSO
MESA_POR_GRUPO = (("corpo_prateleira", "contato_tronco"), ("palma_prateleira", "contato_palma"),
                  ("dorso_prateleira", "contato_dorso"))

# ⚠ A lista de corpo inteiro (`.*_collision`) falhou em medição (g1_limpo/cena.py:45-75): o pad de palma
# entra e o punho e o cotovelo saem. O que deixa isso seguro é o LIMIAR DE FORÇA da terminação, e não a
# lista: roçar não termina; escorar termina.

_PAD_HALF = (0.035, 0.008, 0.045)  # laje fina, cobre a palma (g1_limpo/cena.py:90)
_PALM_DZ = 0.015  # offset ao longo da normal da palma
_PALM_X = 0.10  # ao longo da mão, na região da palma
# ⚠ O sinal de dz das palmas é o do CÓDIGO (g1_limpo/cena.py:add_pads_de_palma), não o do comentário
# da fonte: o pad de palma da direita sai em +dz e o da esquerda em −dz. O comentário estava errado.


# ============================================================ specs de entidade
def _meia_caixa(c: Cena) -> float:
    """Meia aresta de REFERÊNCIA da caixa: o meio da faixa de DR (g1_limpo: a variante do meio)."""
    return sum(c.caixa_meia_faixa) / 2.0


def _spec_box(body: str, geom: str, joint: str | None, half, atrito, priority: int, condim: int,
              massa: float | None = None, rgba=None) -> mujoco.MjSpec:
    """Um box primitivo num MjSpec PRÓPRIO. `joint=None` -> sem free joint -> o mjlab põe em MOCAP.

    ⚠ `priority` > a do robô (0; pés 1) faz todo contato com o robô usar o μ, o condim e o solref
    DESTE geom, e não o máximo do par. É o que dá sentido ao DR de μ da caixa (PD-46).
    """
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name=body)
    if joint is not None:
        b.add_freejoint(name=joint)
    kwargs = dict(name=geom, type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(half), condim=condim,
                  friction=tuple(atrito), priority=priority)
    if massa is not None:
        kwargs["mass"] = massa
    if rgba is not None:
        kwargs["rgba"] = tuple(rgba)
    b.add_geom(**kwargs)
    return spec


def regroup(spec: mujoco.MjSpec, group: int) -> mujoco.MjSpec:
    """Põe todos os geoms do spec num grupo de geom.

    ⚠ Sem isto a mobília nasce no grupo 0, e o `foot_height_scan` leria a laje COMO CHÃO: o robô
    "veria" um degrau que é a própria mesa dele.
    """
    for geom in spec.geoms:
        geom.group = group
    return spec


def spec_caixa(c: Cena) -> mujoco.MjSpec:
    """Caixa LEVE a erguer: corpo LIVRE, cubo na meia aresta de referência, `caixa_massa_base` kg.

    ⚠ `priority` = `caixa_priority` (0) e μ = `caixa_atrito` (PD-46). O condim é 3, o padrão do MuJoCo
    (g1_limpo/knobs.py Cena.caixa_condim).
    """
    m = _meia_caixa(c)
    spec = _spec_box("box", BOX_GEOM, "box_joint", (m, m, m), c.caixa_atrito, c.caixa_priority, 3,
                     massa=c.caixa_massa_base)
    return regroup(spec, c.grupo_mobilia)


def spec_laje(c: Cena) -> mujoco.MjSpec:
    """Laje FINA, sem free joint -> o mjlab põe em MOCAP.

    Corpo cinemático: posicionável por env em runtime, flutua em qualquer z, não é movido por contato,
    dispensa massa. A caixa repousa em cima e continua erguível. Fina em z: é uma laje, não um paredão,
    o que mata o atalho de escorar o robô nela.
    """
    half = (c.prateleira_meia_xy, c.prateleira_meia_xy, c.prateleira_meia_z)
    spec = _spec_box("table", TABLE_GEOM, None, half, c.prateleira_atrito, c.prioridade_laje,
                     c.prateleira_condim, rgba=c.prateleira_rgba)
    return regroup(spec, c.grupo_mobilia)


# ===================================================================== o robô
def _add_pad(corpo_mao, name: str, dz: float, condim: int, atrito, rgba) -> None:
    corpo_mao.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=(_PALM_X, dz, 0.0),
                       size=_PAD_HALF, condim=condim, friction=tuple(atrito), group=_GRUPO_PAD,
                       rgba=rgba)


def add_pads_de_palma(spec: mujoco.MjSpec, c: Cena) -> mujoco.MjSpec:
    """Remove as cápsulas `*_hand_collision` e acrescenta pads de palma e de dorso.

    As cápsulas são simétricas: não distinguem palma de verso, e deixariam o DORSO tocar a caixa e
    contar como pega. ⚠ A API de remoção é `spec.delete(geom)`, nunca `geom.delete()`.
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
        _add_pad(hand_body, palm_pad, sinal * _PALM_DZ, 3, c.pad_atrito, (0.1, 0.6, 0.1, 0.6))
        _add_pad(hand_body, back_pad, -sinal * _PALM_DZ, 1, c.robo_atrito, (0.6, 0.1, 0.1, 0.4))
    return spec


def _sitios_do_torso(spec: mujoco.MjSpec, k: Knobs) -> mujoco.MjSpec:
    """Sites do lidar (MID-360) e da D435i, no `torso_link` (PERC §2; `lidar_sim/sensores/montagem.py`)."""
    torso = spec.body("torso_link")
    e = k.estimador
    # ⚠ quaternion (w, x, y, z); a D435i gira `d435_pitch` em torno de y, e o eixo x do site é o óptico.
    meio = e.d435_pitch / 2.0
    torso.add_site(name="mid360", pos=k.percepcao.mid360_pos, quat=k.percepcao.mid360_quat)
    torso.add_site(name="d435", pos=e.d435_pos, quat=(math.cos(meio), 0.0, math.sin(meio), 0.0))
    return spec


def _pelve_de_pe(k: Knobs) -> float:
    """Altura da pelve na pose de pé: `pelve_de_pe` do `referencias.npz`, como o `q_de_pe` (I §2.2)."""
    with np.load(caminho(k.contrato.referencias)) as npz:
        return float(npz["pelve_de_pe"])


def _atuadores(k: Knobs) -> tuple:
    """Um BuiltinPositionActuatorCfg por linha de `Contrato.pd`: kp e kd da linha, e armature e
    effort_limit do motor da junta (os do mjlab, `g1_constants.py:91-178`; tornozelo e cintura
    roll/pitch já vêm com o dobro nos grupos). Atraso do comando em todos: capacidade de 3 passos da
    política, em passos de física (PD-25, PD-28).
    """
    grupos = G1_ARTICULATION.actuators

    def motor(regex: str):
        junta = next(j for j in JUNTAS if re.fullmatch(regex, j))
        return next(g for g in grupos if any(re.fullmatch(e, junta) for e in g.target_names_expr))

    atraso_max = k.dr.atraso_refino[1] * k.contrato.decimacao  # 3 passos da política × 4
    out = []
    for regex, kp, kd in k.contrato.pd:
        m = motor(regex)
        out.append(BuiltinPositionActuatorCfg(
            target_names_expr=(regex,), stiffness=kp, damping=kd,
            effort_limit=m.effort_limit, armature=m.armature,
            delay_min_lag=0, delay_max_lag=atraso_max,
            # ⚠ 10⁹ passos = nunca reamostra: o lag sorteado no reset (eventos) fica no episódio inteiro.
            delay_update_period=10**9, delay_per_env_phase=False,
        ))
    return tuple(out)


def robot_cfg(k: Knobs = ATIVO) -> EntityCfg:
    """O G1 com as mãos refinadas, os pads, os sites e o PD da Unitree.

    ⚠ Os pads terminam em `_pad`, portanto NÃO casam com `.*_collision` do FULL_COLLISION: entram
    explicitamente, senão não participam da colisão e os sensores de palma nunca disparam.
    """
    c = k.cena
    collisions = CollisionCfg(
        geom_names_expr=(".*_collision", ".*_pad"),
        condim={
            r"^(left|right)_foot[1-7]_collision$": 3,
            r".*_palm_pad$": 3,
            r".*_hand_back_pad$": 1,
            ".*_collision": 1,
        },
        priority={r"^(left|right)_foot[1-7]_collision$": 1},
        # ⚠ os pés vêm PRIMEIRO: o primeiro padrão que casa vence, como no condim.
        friction={r"^(left|right)_foot[1-7]_collision$": (0.6,), ".*_collision": c.robo_atrito},
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, _pelve_de_pe(k) + 0.01),
            joint_pos=dict(zip(JUNTAS, q_de_pe(k.contrato))),
            joint_vel={".*": 0.0},
        ),
        collisions=(collisions,),
        spec_fn=lambda: _sitios_do_torso(add_pads_de_palma(get_spec(), c), k),
        articulation=EntityArticulationInfoCfg(
            actuators=_atuadores(k),
            soft_joint_pos_limit_factor=G1_ARTICULATION.soft_joint_pos_limit_factor,
        ),
    )


# ================================================================== o terreno

def colunas(t: Terreno) -> tuple:
    """As colunas do gerador, na ordem de `Terreno.colunas`. Uma coluna tem plataforma de 2 m se e só se
    está em `linha_max_carregar` (PD-20): é ela que o CARREGAR usa.
    """
    lim = dict(t.linha_max_carregar)
    return tuple({"nome": n, "plataforma": n in lim, "linha_max_carregar": lim.get(n)}
                 for n in t.colunas)


COLUNAS = colunas(ATIVO.terreno)


def terreno_cfg(k: Knobs = ATIVO) -> TerrainEntityCfg:
    """Gerador com um sub-terreno por coluna (`curriculum=True`: cada tipo ganha exatamente uma coluna).

    Os tetos vêm de `Terreno.tetos_andar`; a linha de cada coluna (t = linha/(R−1)) é o que dá o teto
    de CARREGAR (`linha_max_carregar`). ⚠ `scale_with_difficulty=True` no `irregular`: o preset vem
    com False (`heightfield_terrains.py:404`), e sem a troca toda linha tem a mesma rugosidade.
    """
    t = k.terreno
    teto = dict(t.tetos_andar)
    pl = t.plataforma
    degrau = (0.0, teto["degrau"])
    sub = {
        "plano": flat(),
        "escada": pyramid_stairs(step_height_range=degrau, platform_width=pl),
        "escada_inv": pyramid_stairs_inv(step_height_range=degrau, platform_width=pl),
        "rampa": hf_pyramid_slope(slope_range=(0.0, teto["rampa"]), platform_width=pl),
        "rampa_inv": hf_pyramid_slope_inv(slope_range=(0.0, teto["rampa"]), platform_width=pl),
        "escada_aleatoria": random_stairs(step_height_range=degrau, platform_width=pl),
        "irregular": random_rough(scale_with_difficulty=True),
        "obstaculos": discrete_obstacles(obstacle_height_range=(0.0, teto["obstaculo"]), platform_width=pl),
        "caixas": random_spread_boxes(box_height_range=(0.0, teto["obstaculo"]), platform_width=pl),
    }
    gerador = TerrainGeneratorCfg(size=t.tamanho, num_rows=t.linhas, curriculum=True,
                                  border_width=t.borda, sub_terrains=sub)
    return TerrainEntityCfg(terrain_type="generator", terrain_generator=gerador)


# ================================================================== a cena
def geometria_de_repouso(c: Cena) -> dict[str, float]:
    """As alturas derivadas, num lugar só. A laje é MOCAP: a pose é o CENTRO do corpo, `meia_z` abaixo
    do topo pedido."""
    topo = c.prateleira_topo_teto
    return {
        "topo": topo,
        "centro_prateleira_z": topo - c.prateleira_meia_z,
        "fundo_prateleira_z": topo - 2.0 * c.prateleira_meia_z,
        "caixa_z": topo + _meia_caixa(c),
    }


def entidades(k: Knobs = ATIVO) -> dict[str, EntityCfg]:
    c = k.cena
    g = geometria_de_repouso(c)
    return {
        "robot": robot_cfg(k),
        "box": EntityCfg(
            spec_fn=lambda: spec_caixa(c),
            init_state=EntityCfg.InitialStateCfg(pos=(c.caixa_xy[0], c.caixa_xy[1], g["caixa_z"])),
        ),
        # O nome da entidade é `table` (os sensores de mesa a referem assim); o que muda é a geometria.
        "table": EntityCfg(
            spec_fn=lambda: spec_laje(c),
            init_state=EntityCfg.InitialStateCfg(
                pos=(c.prateleira_xy[0], c.prateleira_xy[1], g["centro_prateleira_z"])),
        ),
    }


@dataclass
class PadraoMid360Cfg:
    """Padrão de raios do MID-360 (205 raios por passo). Duck typing com os padrões do mjlab:
    `RayCastSensor` só chama `generate_rays(mj_model, device)` (`raycast_sensor.py:517-521`).

    ⚠ Devolve o PRIMEIRO passo. A fase 5a troca `_local_directions` a cada passo (`prepare_rays`,
    `raycast_sensor.py:691-747`), e não este cfg.
    """
    caminho: str

    def generate_rays(self, mj_model, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        del mj_model
        dirs = np.load(self.caminho)["dirs"][0]  # (205, 3) no frame do sensor
        offsets = torch.zeros((dirs.shape[0], 3), dtype=torch.float32, device=device)
        return offsets, torch.as_tensor(dirs, dtype=torch.float32, device=device)


def _altura(nome: str, corpos: tuple[ObjRef, ...], raio: float, amostras: int,
             grupos: tuple[int, ...] = (0,)) -> TerrainHeightSensorCfg:
    """Altura sobre o TERRENO (grupo 0) de um anel de raios para baixo.

    ⚠ `ray_alignment="yaw"`: os raios descem na vertical mesmo com o corpo inclinado, e a altura é a
    da origem do frame (L7: o termo da Unitree é a origem do link). É o que o mjlab usa no
    `foot_height_scan` (`velocity_env_cfg.py`).
    """
    return TerrainHeightSensorCfg(name=nome, frame=corpos,
                                  pattern=RingPatternCfg.single_ring(radius=raio, num_samples=amostras),
                                  ray_alignment="yaw", include_geom_groups=grupos)


def sensores(k: Knobs = ATIVO) -> tuple:
    """Os sensores da cena: os 11 do g1_limpo, os de altura, o de contato com o chão, a régua do aperto
    por face e, com `Percepcao.ligada`, o lidar (18 no total)."""
    c = k.cena
    robo = "robot"
    palmas = tuple(
        ContactSensorCfg(name=nome, primary=ContactMatch(mode="geom", pattern=pad, entity=robo),
                         secondary=ContactMatch(mode="geom", pattern=BOX_GEOM, entity="box"),
                         fields=("found", "force"), reduce="netforce", num_slots=1)
        for nome, pad in zip(("palma_E", "palma_D"), PALM_PAD_GEOMS)
    )
    # ⚠ Dorsos: só `found`, com `reduce="none"`. Medem o toque, não a força.
    dorsos = tuple(
        ContactSensorCfg(name=nome, primary=ContactMatch(mode="geom", pattern=pad, entity=robo),
                         secondary=ContactMatch(mode="geom", pattern=BOX_GEOM, entity="box"),
                         fields=("found",), reduce="none", num_slots=1)
        for nome, pad in zip(("dorso_E", "dorso_D"), BACK_PAD_GEOMS)
    )
    apoio = ContactSensorCfg(name="apoio_caixa",
                             primary=ContactMatch(mode="geom", pattern=BOX_GEOM, entity="box"),
                             secondary=ContactMatch(mode="geom", pattern=TABLE_GEOM, entity="table"),
                             fields=("found", "force"), reduce="netforce", num_slots=1)
    mesa = tuple(
        ContactSensorCfg(name=nome, primary=ContactMatch(mode="geom", pattern=grupo, entity=robo),
                         secondary=ContactMatch(mode="geom", pattern=TABLE_GEOM, entity="table"),
                         fields=("found", "force"), reduce="netforce", num_slots=1)
        for (nome, _), grupo in zip(MESA_POR_GRUPO, (GRUPO_TRONCO, GRUPO_PALMA, GRUPO_DORSO))
    )
    # ⚠ FATIAS, e não `netforce`: as duas palmas apertam em sentidos OPOSTOS e a soma se cancela.
    caixa_robo = ContactSensorCfg(name="caixa_robo",
                                  primary=ContactMatch(mode="geom", pattern=BOX_GEOM, entity="box"),
                                  secondary=ContactMatch(mode="subtree", pattern="pelvis", entity=robo),
                                  fields=("force", "found"), reduce="maxforce", num_slots=12)
    auto = ContactSensorCfg(name="auto_colisao",
                            primary=ContactMatch(mode="subtree", pattern="pelvis", entity=robo),
                            secondary=ContactMatch(mode="subtree", pattern="pelvis", entity=robo),
                            fields=("found", "force"), reduce="none", num_slots=1, history_length=4)
    # ⚠ `secondary=None`: qualquer contato do pé conta como chão, inclusive a laje.
    pes = ContactSensorCfg(name="pes_chao",
                           primary=ContactMatch(mode="subtree",
                                                pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                                                entity=robo),
                           secondary=None, fields=("found", "force"), reduce="netforce", num_slots=1,
                           track_air_time=True)

    # Alturas (PD-21, S §8.2): o pé pela origem do link (L7); a pelve, os joelhos e a caixa.
    altura_pes = _altura("altura_pes", tuple(ObjRef(type="body", name=n, entity=robo) for n in (
        "left_ankle_roll_link", "right_ankle_roll_link")), 0.03, 6)
    altura_pelve = _altura("altura_pelve", (ObjRef(type="body", name="pelvis", entity=robo),), 0.05, 4)
    altura_joelhos = _altura("altura_joelhos", tuple(ObjRef(type="body", name=n, entity=robo) for n in (
        "left_knee_link", "right_knee_link")), 0.03, 4)
    altura_caixa = _altura("altura_caixa", (ObjRef(type="body", name="box", entity="box"),), 0.05, 4)

    # Contato do robô com o chão, sem os pés: a mão, o joelho ou o tronco no chão (S §9.2).
    nao_pes_chao = ContactSensorCfg(
        name="nao_pes_chao",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity=robo,
                             exclude=(r".*_ankle_roll_link",)),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1,
    )
    # Régua do aperto por face (S §8.2): quem toca cada face da caixa, com posição e normal no mundo.
    caixa_faces = ContactSensorCfg(
        name="caixa_faces",
        primary=ContactMatch(mode="geom", pattern=BOX_GEOM, entity="box"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity=robo),
        fields=("found", "force", "pos", "normal"), reduce="none", num_slots=12, global_frame=True,
    )

    fixos = palmas + dorsos + mesa + (apoio, caixa_robo, auto, pes)
    alturas = (altura_pes, altura_pelve, altura_joelhos, altura_caixa, nao_pes_chao, caixa_faces)
    if not k.percepcao.ligada:
        return fixos + alturas
    # Lidar: grupos 0 (terreno), 1 (mobília) e 3 (pads). `exclude_parent_body` deixa o sensor dentro da
    # esfera da cabeça (`lidar_sim/sensores/lidar.py:38-39`); o filtro do corpo é da D6 (PD-10).
    lidar = RayCastSensorCfg(
        name="lidar", frame=ObjRef(type="site", name="mid360", entity=robo),
        pattern=PadraoMid360Cfg(caminho=str(caminho("g1_v3/dados/mid360_passos.npz"))),
        ray_alignment="base", max_distance=k.percepcao.alcance[1],
        include_geom_groups=(0, 1, _GRUPO_PAD), exclude_parent_body=True,
    )
    return fixos + alturas + (lidar,)
