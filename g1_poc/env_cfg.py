"""Monta o env do g1_poc a partir do `velocity` do mjlab.

Estratégia: partir de `make_velocity_env_cfg()` e MUTAR. Assim a fundação de
locomoção — observação proprioceptiva, ação, os 13 termos de recompensa, a DR de
startup, o push, as terminações — entra sem uma linha reescrita, com os pesos do G1.

O que este arquivo muda:
    1. terreno plano, e a mobília entra na cena
    2. os sensores de contato
    3. o comando: `caixa_alvo` (novo, PRIMEIRO) e `twist` (subclasse)
    4. a observação: sai `base_lin_vel` do ator, entram os 5 canais de caixa + `face_normal_b`
    5. o crítico: 13 canais privilegiados
    6. a `posture` ganha o quarto regime (FORMA do episódio)
    7. os 9 termos de tarefa
    8. as 2 terminações próprias
    9. os 3 eventos de cena
   10. o currículo em 4 partes: forma, nível, gate por competência, qualidade
   11. a máquina de elo (§7)

Contrato de observação: **ator 115, crítico 128**. Ele é derivado, não digitado —
o `smoke.py` o confere contra o `observation_manager`.
"""
from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab.asset_zoo.robots import G1_ACTION_SCALE

from g1_poc import cena as C
from g1_poc import curriculo as CU
from g1_poc import eventos as EV
from g1_poc import observacoes as OBS
from g1_poc import recompensas as R
from g1_poc import terminacoes as T
from g1_poc.comando import CaixaAlvoCommandCfg, TwistPocCfg
from g1_poc.knobs import ATIVO, Knobs
from g1_poc.postura import postura_manipulacao

CMD_CAIXA = "caixa_alvo"
CMD_TWIST = "twist"

# largura esperada da observação. Derivada, e conferida no smoke.
# (+3 do canal `face_normal_b`, que a cirurgia de checkpoint appenda às colunas)
OBS_ATOR = 115
OBS_CRITICO = 128


def _g1_pesos_de_postura(cfg) -> dict:
    """Colhe os três dicionários de σ do G1 do cfg do fabricante.

    Eles são calibrados por robô e NÃO são redigitados aqui.
    """
    p = cfg.rewards["pose"].params
    return {
        "std_standing": p["std_standing"],
        "std_walking": p["std_walking"],
        "std_running": p["std_running"],
    }


def make_g1_poc_env_cfg(k: Knobs | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    k = k or ATIVO
    kc, ka, kt, kr, kp, ke, kd, kcm = (
        k.cena, k.alvo, k.tol, k.reward, k.postura, k.episodio, k.dr, k.comando)
    lateral = kr.lateral_offset if kr.lateral_offset is not None else kc.caixa_meia_aresta[1]

    # -------------------------------------------------- 0. a fundação do fabricante
    # `unitree_g1_flat_env_cfg` já resolve os sítios, os grupos e os σ do G1.
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

    cfg = unitree_g1_flat_env_cfg(play=play)
    postura_std = _g1_pesos_de_postura(cfg)

    site_pes = ("left_foot", "right_foot")

    # -------------------------------------------------- 1. a cena
    cfg.scene.entities = C.entidades(k)
    cfg.scene.extent = 2.5

    # o `foot_height_scan` do fabricante fica; ele lê o grupo 0, e a mobília está
    # no grupo 2 (ver `cena.regroup`). Os nossos sensores entram por cima.
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if not isinstance(s, RayCastSensorCfg) or s.name == "foot_height_scan"
    ) + C.sensores()
    # o `feet_ground_contact` e o `self_collision` do fabricante saem: os nossos
    # (`pes_chao` e `auto_colisao`) os substituem, com nomes do pacote.
    cfg.scene.sensors = tuple(
        s for s in cfg.scene.sensors
        if s.name not in ("feet_ground_contact", "self_collision")
    )

    # física de manipulação (cicatriz de 15/07: `elliptic`/`impratio=10` divergiu
    # para NaN no reset parcial; `pyramidal`/1,0 é o par que roda)
    cfg.sim.njmax = 800
    cfg.sim.nconmax = 300
    cfg.sim.mujoco.impratio = 1.0
    cfg.sim.mujoco.cone = "pyramidal"

    # -------------------------------------------------- 2. ação
    acao = cfg.actions["joint_pos"]
    assert isinstance(acao, JointPositionActionCfg)
    acao.scale = {j: v * kc.escala_acao_mult for j, v in G1_ACTION_SCALE.items()}

    # -------------------------------------------------- 3. comandos
    # ORDEM: `caixa_alvo` PRIMEIRO. Ele resolve `env.poc_twist_zero`, e o `twist`
    # lê esse buffer no mesmo passo. Dict é ordenado por inserção.
    ep = ke.duracao_s
    cfg.commands = {
        CMD_CAIXA: CaixaAlvoCommandCfg(
            # ⚠ 10×, e NUNCA igual à duração do episódio. Com (20, 20) o time_left
            # do comando cruza zero no passo 999 e o time_out da terminação só
            # dispara no passo 1000: o _resample rodava UM PASSO antes do fim e
            # zerava episode_success/pegou/alvo — o nível lia sucesso 0 em todo
            # episódio que chegava ao time_out (medido it 5306: pegou 0,97 com
            # sucesso 0,00), e a escada ficava esfomeada. O "sucesso 0,006 no
            # treino vs 0,75 na sonda" do bloco 2 era em grande parte isto. A meta
            # é 1 por episódio: quem resampleia é o RESET.
            resampling_time_range=(10.0 * ep, 10.0 * ep),
            debug_vis=True,
            pegar_range=(ka.pegar_x, ka.pegar_y, ka.pegar_z),
            raio_sucesso=kt.raio_sucesso,
            angulo_sucesso_rad=kt.angulo_sucesso_rad,
            sustenta_pegar_s=kt.sustenta_pegar_s,
            pelve_min=kt.pelve_min,
            inclinacao_max_rad=kt.inclinacao_max_rad,
            bringing_std_piso=kr.bringing_std_piso,
            palm_sites=C.PALM_SITES,
            lateral_offset=lateral,
            reaching_std_piso=kr.reaching_std,
            cadeias=k.celulas.cadeias,
            ang_max_deg=k.celulas.ang_max_deg,
            sustenta_outros_s=kt.sustenta_outros_s,
            carregar_s=kt.carregar_s,
            fracao_apoio_botar=kt.fracao_apoio_botar,
            peito_b=ka.peito_b,
            botar_x=ka.botar_x,
            botar_y=ka.botar_y,
            botar_topo_piso=ka.botar_topo_piso,
            botar_topo_teto=ka.botar_topo_teto,
            botar_folga_laje=ka.botar_folga_laje,
            caixa_meia_z=kc.caixa_meia_aresta[2],
            prateleira_meia_z=kc.prateleira_meia_z,
            prateleira_xy=kc.prateleira_xy,
            afasta_z=kc.afasta_z,
            support_sensor=C.SENSOR_APOIO,
            precise_ori_std_piso=kr.precise_ori_std,
            frac_twist_livre=ke.frac_twist_livre_manipula,
            twist_livre_nivel_min=ke.twist_livre_nivel_min,
        ),
        CMD_TWIST: TwistPocCfg(
            entity_name="robot",
            resampling_time_range=kcm.resample_s,
            rel_standing_envs=kcm.rel_standing,
            rel_heading_envs=kcm.rel_heading,
            rel_forward_envs=kcm.rel_forward,
            heading_command=True,
            heading_control_stiffness=kcm.heading_stiffness,
            debug_vis=True,
            ranges=TwistPocCfg.Ranges(
                lin_vel_x=kcm.lin_vel_x,
                lin_vel_y=kcm.lin_vel_y,
                ang_vel_z=kcm.ang_vel_z,
                heading=(-math.pi, math.pi),
            ),
            frac_giro_no_standing=kcm.frac_giro_no_standing,
            piso_giro_rad_s=kcm.piso_giro_rad_s,
        ),
    }

    # -------------------------------------------------- 4. observação do ator
    ator = cfg.observations["actor"].terms
    critico = cfg.observations["critic"].terms

    # `base_lin_vel` SAI do ator. Num humanoide real ela não é medida de forma
    # confiável — é por isso que a task de tracking da Unitree se chama
    # "No-State-Estimation". Ela fica só no crítico, que é descartado.
    ator.pop("base_lin_vel", None)

    # os termos de pé do crítico do fabricante saem: as dimensões deles não são
    # deriváveis deste arquivo, e o contrato tem de ser um número exato. Podem
    # voltar depois — eles não tocam o deploy.
    for nome in ("foot_height", "foot_air_time", "foot_contact", "foot_contact_forces"):
        critico.pop(nome, None)

    caixa_ator = {
        "palmas_para_caixa": ObservationTermCfg(
            func=OBS.palmas_para_caixa,
            params={
                "command_name": CMD_CAIXA,
                "object_name": "box",
                "lateral_offset": lateral,
                "asset_cfg": SceneEntityCfg("robot", site_names=C.PALM_SITES),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "caixa_para_alvo": ObservationTermCfg(
            func=OBS.caixa_para_alvo,
            params={"command_name": CMD_CAIXA, "object_name": "box"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "face_alvo": ObservationTermCfg(
            func=OBS.fatia_comando,
            params={"command_name": CMD_CAIXA, "lo": 3, "hi": 6},
        ),
        "dir_alvo": ObservationTermCfg(
            func=OBS.fatia_comando,
            params={"command_name": CMD_CAIXA, "lo": 6, "hi": 9},
        ),
        "caixa_valida": ObservationTermCfg(
            func=OBS.fatia_comando,
            params={"command_name": CMD_CAIXA, "lo": 9, "hi": 10},
        ),
        "face_normal_b": ObservationTermCfg(
            func=OBS.face_normal_b,
            params={"command_name": CMD_CAIXA, "object_name": "box"},
        ),
    }
    ator.update(caixa_ator)
    # no crítico, copiar caixa_ator mas remover face_normal_b, que entra depois de topo_prateleira
    critico.update({n: t for n, t in caixa_ator.items() if n != "face_normal_b"})

    # -------------------------------------------------- 5. crítico privilegiado
    critico["base_lin_vel"] = ObservationTermCfg(
        func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"})
    critico["forca_palmas"] = ObservationTermCfg(
        func=OBS.forca_palmas, params={"sensores": C.SENSOR_PALMA})
    critico["forca_apoio"] = ObservationTermCfg(
        func=OBS.forca_apoio, params={"sensor_name": C.SENSOR_APOIO})
    critico["vel_caixa"] = ObservationTermCfg(
        func=OBS.vel_caixa, params={"object_name": "box"})
    critico["topo_prateleira"] = ObservationTermCfg(
        func=OBS.topo_prateleira, params={"meia_z": kc.prateleira_meia_z})
    critico["face_normal_b"] = ObservationTermCfg(
        func=OBS.face_normal_b,
        params={"command_name": CMD_CAIXA, "object_name": "box"},
    )

    # -------------------------------------------------- 6. postura, 4º regime
    pose = cfg.rewards["pose"]
    pose.func = postura_manipulacao
    pose.weight = kr.postura
    pose.params = {
        **postura_std,
        "std_manipulando": kp.std_manipulando,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": CMD_TWIST,
        "caixa_command_name": CMD_CAIXA,
        "walking_threshold": 0.05,
        "running_threshold": kp.running_threshold,
    }

    # os pesos da fundação que o pacote muda em relação ao `velocity`
    cfg.rewards["dof_pos_limits"].weight = kr.dof_pos_limits   # −1,0 -> −10,0
    cfg.rewards["action_rate_l2"].weight = kr.action_rate
    cfg.rewards["upright"].weight = kr.upright
    cfg.rewards["body_ang_vel"].weight = kr.body_ang_vel
    cfg.rewards["angular_momentum"].weight = kr.angular_momentum
    cfg.rewards["foot_clearance"].weight = kr.foot_clearance
    cfg.rewards["foot_swing_height"].weight = kr.foot_swing_height
    cfg.rewards["foot_slip"].weight = kr.foot_slip
    cfg.rewards["soft_landing"].weight = kr.soft_landing
    cfg.rewards["track_linear_velocity"].weight = kr.track_lin
    cfg.rewards["track_angular_velocity"].weight = kr.track_ang
    # o `air_time` fica em 0 (o G1 já o desliga); ele é ruído de gait aqui.
    cfg.rewards["air_time"].weight = 0.0

    # renomeia os sensores de pé/auto-colisão nos termos herdados
    for nome in ("foot_slip", "foot_swing_height", "soft_landing", "air_time"):
        if nome in cfg.rewards and "sensor_name" in cfg.rewards[nome].params:
            cfg.rewards[nome].params["sensor_name"] = C.SENSOR_PES
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=kr.self_collisions,
        params={"sensor_name": C.SENSOR_AUTO_COLISAO,
                "force_threshold": k.term.auto_colisao_N},
    )

    # -------------------------------------------------- 7. os 6 termos de tarefa
    palmas = SceneEntityCfg("robot", site_names=C.PALM_SITES)
    cfg.rewards["staged"] = RewardTermCfg(
        func=R.staged, weight=kr.staged,
        params={"command_name": CMD_CAIXA, "object_name": "box",
                "reaching_std": kr.reaching_std, "lateral_offset": lateral,
                "asset_cfg": palmas},
    )
    cfg.rewards["precise_pos"] = RewardTermCfg(
        func=R.precise_pos, weight=kr.precise_pos,
        params={"command_name": CMD_CAIXA, "object_name": "box",
                "std": kr.precise_pos_std},
    )
    cfg.rewards["precise_ori"] = RewardTermCfg(
        func=R.precise_ori, weight=kr.precise_ori,
        params={"command_name": CMD_CAIXA, "object_name": "box",
                "std": kr.precise_ori_std, "lateral_offset": lateral,
                "reaching_std": kr.reaching_std, "asset_cfg": palmas},
    )
    cfg.rewards["squeeze"] = RewardTermCfg(
        func=R.squeeze, weight=kr.squeeze,
        params={"command_name": CMD_CAIXA, "palm_sensors": C.SENSOR_PALMA,
                "massa_attr": "poc_massa", "mu": kr.squeeze_mu,
                "asset_cfg": palmas},
    )
    # a PONTE do platô do grasp. Ele SOMA ao `squeeze`, e não o substitui: o aperto já
    # está resolvido (6× F_ref na it 1884), o que faltava era pagar por DESCARREGAR.
    cfg.rewards["unload"] = RewardTermCfg(
        func=R.unload, weight=kr.unload,
        params={"command_name": CMD_CAIXA, "object_name": "box",
                "support_sensor": C.SENSOR_APOIO, "palm_sensors": C.SENSOR_PALMA,
                "massa_attr": "poc_massa",
                "caixa_meia_z": kc.caixa_meia_aresta[2],
                "tol_queda": kr.unload_tol_queda},
    )
    # a RAMPA da pelve (§8.2.3): a condição 3 do fecho, que nenhum termo pagava.
    cfg.rewards["postura_ereta"] = RewardTermCfg(
        func=R.postura_ereta, weight=kr.postura_ereta,
        params={"command_name": CMD_CAIXA, "palm_sensors": C.SENSOR_PALMA,
                "support_sensor": C.SENSOR_APOIO, "massa_attr": "poc_massa",
                "pelve_min": kt.pelve_min, "rampa": kr.postura_ereta_rampa,
                "rampa_fina": kr.postura_ereta_rampa_fina,
                "frac_descarga": kr.postura_ereta_frac_descarga,
                "asset_cfg": SceneEntityCfg("robot")},
    )
    # a RAMPA da sustentação (§8.2.4): 0,98 s e 0,00 s pagavam o mesmo.
    cfg.rewards["sustentacao"] = RewardTermCfg(
        func=R.sustentacao, weight=kr.sustentacao,
        params={"command_name": CMD_CAIXA},
    )
    # o ESPELHO do unload, só no elo `botar` (§8.2.5)
    cfg.rewards["load"] = RewardTermCfg(
        func=R.load, weight=kr.load,
        params={"command_name": CMD_CAIXA, "object_name": "box",
                "support_sensor": C.SENSOR_APOIO, "massa_attr": "poc_massa",
                "raio_sucesso": kt.raio_sucesso, "raio_mult": kr.load_raio_mult},
    )
    cfg.rewards["joint_vel_hinge"] = RewardTermCfg(
        func=R.joint_vel_hinge, weight=kr.joint_vel_hinge,
        params={"max_vel": kr.joint_vel_max,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )

    # -------------------------------------------------- 8. terminações
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["fell_over"].params["limit_angle"] = k.term.fell_over_rad
    cfg.terminations["caixa_largada"] = TerminationTermCfg(
        func=T.caixa_largada,
        params={"object_name": "box", "z_min": k.term.caixa_largada_z,
                "dist_max": k.term.caixa_largada_dist, "asset_cfg": palmas},
    )
    cfg.terminations["contato_ilegal"] = TerminationTermCfg(
        func=T.contato_ilegal,
        params={"sensor_name": C.SENSOR_CORPO_PRATELEIRA,
                "limiar_N": k.term.contato_ilegal_N},
    )
    cfg.episode_length_s = ke.duracao_s

    # -------------------------------------------------- 9. eventos
    # ⚠ base_com fica FORA: `dr.body_com_offset` corrompe a heap (medido em CPU e
    # em GPU). Perde-se ±2,5 cm de CoM no torso — gap declarado.
    if not kd.base_com:
        cfg.events.pop("base_com", None)
    cfg.events["reset_base"].params["pose_range"] = kc.reset_base_pose
    # §11.1 — a entrega do navegador: o robô chega andando, não parado.
    cfg.events["reset_base"].params["velocity_range"] = kc.reset_base_vel_manipulacao
    # ⚠ O `push_robot` NÃO existe no modo play: o próprio cfg do G1 o remove
    # (`config/g1/env_cfgs.py:169`). Portanto ele precisa de guarda — sem ela o
    # registro da task quebra, porque o `__init__` monta o cfg de play também.
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].interval_range_s = kd.push_intervalo_s

    # atrito da CAIXA, POR EPISÓDIO, compartilhado entre os dois pads.
    # ⚠ O treino atual registra isto como `startup` e nos pads do robô, sem
    # `shared_random`: uma amostra por env para a run inteira, e μ diferente em cada
    # palma, de forma permanente. Aqui é `reset` e compartilhado.
    cfg.events["caixa_friction"] = EventTermCfg(
        mode="reset",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("box", geom_names=("box_geom",)),
            "operation": "abs",
            "ranges": kd.caixa_friction,
            "shared_random": True,
        },
    )

    # ordem: cena -> carga -> afasta. O `afasta` tem de vir por último.
    cfg.events["reset_cena"] = EventTermCfg(
        func=EV.reset_cena, mode="reset",
        params={
            "topo_min_por_nivel": k.celulas.topo_min,
            "topo_teto": kc.prateleira_topo_teto,
            "jitter_z": kc.prateleira_jitter_z,
            "meia_z": kc.prateleira_meia_z,
            "caixa_meia_z": kc.caixa_meia_aresta[2],
            "caixa_xy": kc.caixa_xy,
            "prateleira_xy": kc.prateleira_xy,
            "jitter_x_max_por_nivel": k.celulas.jitter_x_max,
            "jitter_y": kc.caixa_jitter_y,
            "jitter_yaw_deg": kc.caixa_jitter_yaw_deg,
        },
    )
    cfg.events["carga_caixa"] = EventTermCfg(
        func=EV.carga_caixa, mode="reset",
        params={"carga_max_por_nivel": k.celulas.carga_max, "massa_base": kc.caixa_massa},
    )
    cfg.events["afasta_cena"] = EventTermCfg(
        func=EV.afasta_cena, mode="reset",
        params={"altura": kc.afasta_z, "meia_z": kc.prateleira_meia_z,
                "caixa_meia_z": kc.caixa_meia_aresta[2],
                "caixa_xy": kc.caixa_xy, "prateleira_xy": kc.prateleira_xy},
    )

    # -------------------------------------------------- 10. currículo
    cr = k.cronograma
    cfg.curriculum = {
        # ⚠ ORDEM: `twist_ranges` e `nivel` leem a forma do episódio que ACABOU, e
        # `forma` a SOBRESCREVE com o sorteio do episódio novo. Com `nivel` depois
        # de `forma` (o bug), a promoção era gateada pela forma do episódio
        # SEGUINTE: p_up = 0,7·p, o ponto fixo saía de 0,5 para 0,714, e um bloco
        # com frac_locomocao = 0,85 limitaria nivel_medio a 0,214 mesmo com
        # manipulação perfeita. Os eventos leem a forma NOVA e rodam DEPOIS de todo o
        # currículo, portanto não são afetados.
        "twist_ranges": CurriculumTermCfg(
            func=CU.twist_por_competencia,
            params={"command_name": CMD_TWIST, "velocity_stages": cr.locomocao,
                    "duracao_min_frac": cr.twist_duracao_min_frac,
                    "desce_frac": cr.twist_desce_frac, "ema": cr.twist_ema,
                    "iters_entre_degraus": cr.twist_iters_entre_degraus},
        ),
        "nivel": CurriculumTermCfg(
            func=CU.nivel_caixa,
            params={"command_name": CMD_CAIXA, "nivel_forcado": None},
        ),
        "forma": CurriculumTermCfg(
            func=CU.sorteia_forma,
            params={"frac_locomocao": ke.frac_locomocao,
                    "frac_loco_min": ke.frac_loco_min,
                    "frac_loco_max": ke.frac_loco_max,
                    "ema": ke.forma_ema},
        ),
        "hinge": CurriculumTermCfg(
            func=envs_mdp.reward_curriculum,
            params={"reward_name": "joint_vel_hinge", "stages": cr.hinge},
        ),
        "action_rate": CurriculumTermCfg(
            func=envs_mdp.reward_curriculum,
            params={"reward_name": "action_rate_l2", "stages": cr.action_rate},
        ),
    }

    # -------------------------------------------------- play
    if play:
        # no play o episódio é infinito e SEM currículo de nível: o resample de
        # 20 s volta, para o viewer ciclar metas (o wipe não tem o que esfomear)
        cfg.commands[CMD_CAIXA].resampling_time_range = (ep, ep)
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        # ⚠ O play do G1 ACRESCENTA `randomize_terrain` (`env_cfgs.py:172`), e ele
        # entra no FIM do dict — portanto rodaria DEPOIS do `reset_cena` e do
        # `afasta_cena`. Ele mexe na origem do env, e a nossa mobília é posicionada
        # com pose ABSOLUTA. O resultado seria a caixa e a prateleira dessincronizadas
        # do robô. Aqui não há terreno gerado, portanto ele não tem função.
        cfg.events.pop("randomize_terrain", None)
        # os dois cronogramas por passo global saem; `forma` e `nivel` FICAM, porque
        # o `afasta_cena` e o comando leem `env.poc_manipula`.
        cfg.curriculum.pop("twist_ranges", None)
        cfg.curriculum.pop("hinge", None)
        cfg.curriculum.pop("action_rate", None)

    return cfg
