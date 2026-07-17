"""A INVARIANTE: tudo que TODA skill (Stand, Lift, Place, ...) compartilha.

Contrato central do projeto — **shape de observação e ação idêntico entre
skills** é o que torna o warm-start possível (`--agent.resume --agent.load-run`
entre Stand -> Lift -> Place). Este módulo é o ÚNICO lugar que monta esse
contrato; skills NUNCA re-implementam obs/ação, só herdam `build_base_env` e
mutam recompensa/cena por cima.

O que mora aqui (mecânica, estável): entidades robô+caixa+mesa (a caixa/mesa
existem em TODA skill, mesmo Stand, pra shape de obs não variar), física de
manipulação, fix de reset broadcast-safe, terminação anti-NaN, comando de
sustentação, observação em frame da base, sensores de contato (sempre
presentes — alimentam GATES de recompensa, nunca a obs), e a fundação de
equilíbrio (upright/ang_vel/momentum/limits/action_rate/self_collisions +
postura + slip dos pés). Pesos de fundação que VARIAM por skill (postura,
action_rate, ang_vel/momentum, push) são parâmetros de entrada — os NÚMEROS
ficam nos `knobs.py`/`configs/` de cada skill, não aqui.

O que NÃO mora aqui: recompensa de TAREFA (reaching/grasp/lift/... — só na
Lift, ver skills/lift/), DR de empurrão fora do baseline, push_force.
"""
from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.envs.mdp import rewards as base_rewards
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from .common import events as lift_events
from .common import observations as obs
from .common import rewards as foundation
from .common import terminations as lift_terms
from .common.box import BOX_GEOM, TABLE_GEOM, get_box_spec, get_shelf_spec
from .common.commands import LiftBoxCommandCfg
from .common.robot import BACK_PAD_GEOMS, PALM_PAD_GEOMS, get_lift_box_robot_cfg

# --- sensores de contato (nomes usados por skills/lift/rewards.py) ---
PALM_SENSORS = ("palm_L_box", "palm_R_box")
BACK_SENSORS = ("back_L_box", "back_R_box")
SUPPORT_SENSOR = "box_support"
BODY_TABLE_SENSOR = "body_table"
FOOT_SITES = foundation.FOOT_SITES


def _pad_contact_sensor(name: str, pad_geom: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="geom", pattern=pad_geom, entity="robot"),
        secondary=ContactMatch(mode="geom", pattern=BOX_GEOM, entity="box"),
        fields=("found",), reduce="netforce", num_slots=1,
    )


def build_base_env(
    *,
    play: bool,
    box_pos: tuple[float, float, float],
    table_pos: tuple[float, float, float],
    box_half: tuple[float, float, float] = (0.10, 0.10, 0.10),
    box_mass: float = 1.0,
    shelf_half: tuple[float, float, float] = (0.30, 0.30, 0.02),
    box_pose_range: dict | None = None,
    reset_base_pose_range: dict,
    posture_weight: float,
    posture_joints: list[str],
    posture_std: float = 0.5,
) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_flat_env_cfg(play=play)

    # 1. ENTIDADES: robô com pads + caixa (LIVRE, levantável) + PRATELEIRA (fina,
    #    fixa -> auto-mocap: cinemática, posicionável por-env em qualquer z sem tocar
    #    o chão). Caixa/prateleira existem em TODA skill pra shape de obs não variar;
    #    só a POSIÇÃO muda (cada skill via box_pos/table_pos).
    cfg.scene.entities = {
        "robot": get_lift_box_robot_cfg(),
        "box": EntityCfg(
            spec_fn=lambda: get_box_spec(box_half, box_mass),
            init_state=EntityCfg.InitialStateCfg(pos=box_pos),
        ),
        "table": EntityCfg(
            spec_fn=lambda: get_shelf_spec(shelf_half),
            init_state=EntityCfg.InitialStateCfg(pos=table_pos),
        ),
    }

    # 2. GOTCHA por-ambiente: sem evento de reset a entidade fica no world-origin
    #    pra TODOS os envs. `reset_root_state_uniform` é POLIMÓRFICO — trata a caixa
    #    (free-body: write_root_state) E a prateleira (mocap: write_mocap_pose) no
    #    mesmo código, somando a amostra de pose_range à posição de repouso +
    #    env_origin. É o que dá o jitter da caixa (só a caixa; prateleira pose_range={}).
    for name in ("box", "table"):
        rng = (box_pose_range or {}) if name == "box" else {}
        cfg.events[f"reset_{name}"] = EventTermCfg(
            func=vel_mdp.reset_root_state_uniform, mode="reset",
            params={"pose_range": rng, "velocity_range": {}, "asset_cfg": SceneEntityCfg(name)},
        )

    # 3. reset da BASE (task parada; range decidido por skill — Stand fica
    #    mais apertado, mesma família de faixa em todas).
    cfg.events["reset_base"].params["pose_range"] = reset_base_pose_range

    # 4. FÍSICA de manipulação (cicatriz 2026-07-15: elliptic/impratio=10
    #    divergia pra NaN no reset parcial; pyramidal/1.0 digere).
    cfg.sim.njmax = 800
    cfg.sim.nconmax = 300
    cfg.sim.mujoco.impratio = 1.0
    cfg.sim.mujoco.cone = "pyramidal"

    # 5. DR de startup OFF por ora + fix broadcast-safe do reset de juntas
    #    (bug upstream: soft_joint_pos_limits colapsa (1,nj,2) sem DR).
    for ev in ("foot_friction", "encoder_bias", "base_com"):
        cfg.events.pop(ev, None)
    cfg.events["reset_robot_joints"].func = lift_events.reset_joints_by_offset
    if play:
        cfg.events.pop("push_robot", None)

    # 5c. GUARDA defesa-em-profundidade: mundo não-finito termina e reseta.
    cfg.terminations["nonfinite"] = TerminationTermCfg(
        func=lift_terms.nonfinite_state, time_out=False)

    # 6. COMANDO: alvo de sustentação (+ bit de fase reservado). debug_vis=True
    #    desenha uma esfera translúcida no alvo no viewer/play/vídeo (mesmo padrão
    #    do LiftingCommand do mjlab) — sem custo em treino headless (só desenha
    #    quando tem visualizer real).
    cfg.commands = {
        "lift_target": LiftBoxCommandCfg(
            entity_name="box", debug_vis=True,
            resampling_time_range=(cfg.episode_length_s, cfg.episode_length_s),
        )
    }

    # 7. FUNDAÇÃO: mantém só o equilíbrio base; apaga marcha (lê o twist).
    _BALANCE = ("upright", "body_ang_vel", "angular_momentum",
                "dof_pos_limits", "action_rate_l2", "self_collisions")
    for name in list(cfg.rewards):
        if name not in _BALANCE:
            del cfg.rewards[name]

    # 7a. POSTURA (fundação command-free): mantém as juntas perto do keyframe
    #     joelhos-flexionados. Escopo/peso VARIAM por skill (parâmetro).
    cfg.rewards["posture"] = RewardTermCfg(
        func=base_rewards.posture, weight=posture_weight,
        params={"std": {".*": posture_std},
                "asset_cfg": SceneEntityCfg("robot", joint_names=posture_joints)},
    )

    # 7b. SLIP DOS PÉS — FUNDAÇÃO FIXA, sempre-on em TODA skill (user 07-16:
    #     ataca o "espacato" na raiz — pé escorregando pra alargar a base —
    #     em vez de só punir o sintoma via upright/postura). Não é knob de
    #     config: presença e peso são fixos aqui, não desligam por fine-tune.
    cfg.rewards["feet_slip"] = RewardTermCfg(
        func=foundation.feet_slip_standing, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES)},
    )

    # 8. OBSERVAÇÃO (frame da BASE): contrato sim-to-real, shape FIXO.
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        del terms["command"]
        terms["target_pos_b"] = ObservationTermCfg(
            func=obs.target_pos_b, params={"command_name": "lift_target"})
        terms["box_pos_b"] = ObservationTermCfg(
            func=obs.object_pos_b, params={"object_name": "box"})
        terms["phase"] = ObservationTermCfg(
            func=obs.command_phase, params={"command_name": "lift_target"})
        terms["joint_torque"] = ObservationTermCfg(func=obs.joint_torque)

    # 9. currículo de velocidade da velocity referencia o twist -> fora.
    cfg.curriculum.pop("command_vel", None)

    # 10. SENSORES (sempre presentes; alimentam GATES da recompensa de
    #     tarefa da Lift, não a obs — inofensivos/baratos nas outras skills).
    palm_sensors = (_pad_contact_sensor(PALM_SENSORS[0], PALM_PAD_GEOMS[0]),
                    _pad_contact_sensor(PALM_SENSORS[1], PALM_PAD_GEOMS[1]))
    back_sensors = (_pad_contact_sensor(BACK_SENSORS[0], BACK_PAD_GEOMS[0]),
                    _pad_contact_sensor(BACK_SENSORS[1], BACK_PAD_GEOMS[1]))
    box_support = ContactSensorCfg(
        name=SUPPORT_SENSOR,
        primary=ContactMatch(mode="geom", pattern=BOX_GEOM, entity="box"),
        secondary=ContactMatch(mode="geom", pattern=TABLE_GEOM, entity="table"),
        fields=("found",), reduce="netforce", num_slots=1,
    )
    body_table = ContactSensorCfg(
        name=BODY_TABLE_SENSOR,
        primary=ContactMatch(mode="geom", pattern=".*_collision", entity="robot"),
        secondary=ContactMatch(mode="geom", pattern=TABLE_GEOM, entity="table"),
        fields=("found",), reduce="netforce", num_slots=1,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (*palm_sensors, *back_sensors, box_support, body_table)

    return cfg
