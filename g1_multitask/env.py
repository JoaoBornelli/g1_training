"""Monta o env do multi-tarefa por COMPOSIÇÃO sobre o contrato do `g1_training`.

Sem herança: `build_base_env` é função, e o padrão do repo é chamar + mutar. O que
esta função faz, na ordem:

1. chama `build_base_env` -> ganha de graça o contrato de obs, a física de
   manipulação, os sensores de contato, a terminação anti-NaN e a fundação de
   equilíbrio;
2. DESFAZ 4 escolhas da base que o multi-tarefa não pode herdar — o comando de
   4 números, o bit `phase` na obs, o grupo de geom da mobília e a DR de startup
   que a base removeu com `pop`;
3. acrescenta comando, observação, recompensa, terminação e currículo próprios.

Nada em `g1_training/` é alterado. O passo 2 existe justamente por isso.
"""
from __future__ import annotations

import math
from copy import deepcopy

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import rewards as base_rewards
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from g1_training.base_env import (
    PALM_SENSORS,
    BACK_SENSORS,
    BODY_IMPACT_SENSOR,
    BODY_TABLE_SENSOR,
    FOOT_SITES,
    build_base_env,
)
from g1_training.common.box import get_box_spec, get_shelf_spec
from g1_training.common.robot import PALM_SITES
from g1_training.skills.lift import rewards as LR

from . import curriculum as C
from . import events as MT_events
from . import metrics as MT_metrics
from . import observability as MT_obs
from . import observations as obs
from . import rewards as R
from . import tasks as T
from . import terminations as MT_terms
from .commands import DesiredTwistCommandCfg, LiftTargetCommandCfg
from .knobs import MultitaskKnobs
from .pregrasp import POSE_PRE_GRASP
from .scene import regroup
from .tasks import (
    ANDAR,
    ANDAR_CAIXA,
    BOTAR,
    PARADO,
    PARADO_CAIXA,
    PEGAR,
    REORIENTAR,
)

_RESET_BASE_POSE_RANGE = {
    "x": (-0.10, 0.0), "y": (-0.10, 0.10), "z": (0.01, 0.05), "yaw": (-0.2, 0.2),
}
_TOKENS_PERNA = ("hip", "knee", "ankle", "waist")


def _so_pernas(std: dict) -> dict:
    """Filtra um dict de `std` por junta pro escopo perna+cintura.

    Obrigatório, não cosmético: o `resolve_matching_names_values` do mjlab EXIGE que
    toda chave case pelo menos uma junta. Deixar `.*shoulder.*` num escopo sem braço
    levanta erro em vez de ser ignorado."""
    return {k: v for k, v in std.items() if any(t in k for t in _TOKENS_PERNA)}


def build_multitask_env(
    knobs: MultitaskKnobs, play: bool = False
) -> ManagerBasedRlEnvCfg:
    s = knobs.scene
    box_z = s.shelf_top + s.box_half[2]              # caixa repousa no topo da prateleira
    shelf_center_z = s.shelf_top - s.shelf_half_z    # centro do slab fino (mocap)
    yaw = math.radians(s.box_jitter_yaw_deg)

    cfg = build_base_env(
        play=play,
        box_pos=(s.box_xy[0], s.box_xy[1], box_z),
        table_pos=(s.table_xy[0], s.table_xy[1], shelf_center_z),
        box_half=s.box_half,
        box_mass=s.box_mass,
        shelf_half=(s.shelf_half_xy, s.shelf_half_xy, s.shelf_half_z),
        # o jitter de yaw é GERAL (§14), não filtro de tarefa: o `reorientar` e o
        # `pegar` compartilham a MESMA cena, só o comando difere.
        box_pose_range={
            "x": tuple(s.box_jitter_x),
            "y": tuple(s.box_jitter_y),
            "yaw": (-yaw, yaw),
        },
        reset_base_pose_range=_RESET_BASE_POSE_RANGE,
        posture_weight=knobs.reward.postura,
        posture_joints=list(knobs.reward.postura_joints),
    )

    # --- 2a. COMANDO: substitui o `lift_target` de 4 números que a base instalou ---
    # ORDEM IMPORTA: `lift_target` primeiro, porque ele resolve `env.active_task` e o
    # `twist` lê esse buffer no mesmo passo. Dict é ordenado por inserção, e o
    # command manager percorre nessa ordem.
    c, s, t_ = knobs.command, knobs.scene, knobs.tolerancia
    ep = cfg.episode_length_s
    cfg.commands = {
        "lift_target": LiftTargetCommandCfg(
            debug_vis=True,
            resampling_time_range=(ep, ep),      # 1 meta por episódio
            box_half_z=s.box_half[2],
            shelf_half_z=s.shelf_half_z,
            atraso_gatilho_s=c.atraso_gatilho_s,
        ),
        "twist": DesiredTwistCommandCfg(
            debug_vis=False,
            resampling_time_range=(ep, ep),      # derivado: nunca sorteia
            v_max=c.v_max, w_max=c.w_max, heading_gain=c.heading_gain,
            d_morto_andar=c.d_morto_andar, d_morto_manipula=c.d_morto_manipula,
            d_freio_extra=c.d_freio_extra,
        ),
    }

    # --- 2b. o bit `phase` SAI (F10) ---
    # Ele era a reserva da skill Place, constante em 0, e vivia na fatia [3:4] do
    # comando antigo. Com o layout novo essa fatia é `face_alvo.x` — deixar o termo
    # não daria erro, só alimentaria a rede com um número que significa outra coisa.
    # O one-hot torna o bit redundante, e repurposá-lo como "segurando" quebraria o
    # contrato sim-to-real (segurar é INFERIDO de box_pos_b + joint_torque).
    for grupo in ("actor", "critic"):
        cfg.observations[grupo].terms.pop("phase", None)

    # --- 3. OBSERVAÇÃO: +20 canais, 131 -> 151 ---
    # Os DOIS grupos ganham os mesmos canais, na mesma ordem. Se um canal de comando
    # entrasse só no ator, o crítico estimaria valor sem saber qual tarefa está
    # rodando — e o crítico é quem calcula a vantagem.
    for grupo in ("actor", "critic"):
        termos = cfg.observations[grupo].terms
        termos["box_rot_b"] = ObservationTermCfg(
            func=obs.object_rot_b, params={"object_name": "box"})
        termos["face_alvo"] = ObservationTermCfg(
            func=obs.command_slice,
            params={"command_name": "lift_target", "lo": 3, "hi": 6})
        termos["dir_alvo"] = ObservationTermCfg(
            func=obs.command_slice,
            params={"command_name": "lift_target", "lo": 6, "hi": 9})
        termos["task_onehot"] = ObservationTermCfg(
            func=obs.command_slice,
            params={"command_name": "lift_target", "lo": 9, "hi": 17})

        # ESCALA MANUAL do `target_pos_b` (§9b). O normalizador empírico normalmente
        # cuidaria disso, mas nos canais de comando ele é CONGELADO (ver runner.py),
        # e é o congelamento que cria a obrigação de escalar na mão. ÷2.0 põe a
        # faixa útil (0 a ~2 m de destino) em O(1).
        # Nota: a §9b também manda "peso da caixa ÷ 5.0", mas NÃO existe canal de
        # peso na obs — a política infere carga do `joint_torque`, que é o que o
        # robô real mede. Sem canal, sem escala.
        termos["target_pos_b"].scale = 0.5

    # --- 4. CENA: mobília fora do grupo 0 (item 7, metade 1) ---
    # Refaz as duas entidades com o spec regrupado. `build_base_env` as criou no
    # grupo 0 porque `common/box.py` não passa `group=`, e o `foot_height_scan` do
    # fabricante (`include_geom_groups=(0,)`) então lê a prateleira COMO CHÃO.
    shelf_half = (s.shelf_half_xy, s.shelf_half_xy, s.shelf_half_z)
    grupo = s.grupo_mobilia
    cfg.scene.entities["box"] = EntityCfg(
        spec_fn=lambda: regroup(get_box_spec(s.box_half, s.box_mass), grupo),
        init_state=EntityCfg.InitialStateCfg(pos=(s.box_xy[0], s.box_xy[1], box_z)),
    )
    cfg.scene.entities["table"] = EntityCfg(
        spec_fn=lambda: regroup(get_shelf_spec(shelf_half), grupo),
        init_state=EntityCfg.InitialStateCfg(
            pos=(s.table_xy[0], s.table_xy[1], shelf_center_z)),
    )

    # --- 5. CENA: contato do pé casa QUALQUER coisa (item 7, metade 2) ---
    # O `feet_ground_contact` vem com `secondary=ContactMatch(mode="body",
    # pattern="terrain")` — só registra contato pé↔terreno. Pisar na prateleira não
    # conta, e aí o `feet_slip` fica cego e o `feet_air_time` acha o pé no ar.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "feet_ground_contact":
            sensor.secondary = None

    # --- 6. DR DE STARTUP DE VOLTA (item 3e) ---
    # O `base_env` remove os 3 com `pop`. Voltam COLHIDOS de uma chamada limpa do
    # cfg do fabricante, não redigitados: os ranges e o `asset_cfg` já preenchido
    # por robô (geom dos pés, `torso_link`) são a referência testada.
    #
    # Por que tem que estar ligada desde a 1ª iteração e não depois: ela muda a
    # DISTRIBUIÇÃO de treino, e a catraca do anelamento guarda o pico de
    # competência. Ligar depois faria a catraca retirar a muleta num nível que o
    # robô já não sustenta sob a distribuição nova.
    if not play:
        fabricante = unitree_g1_flat_env_cfg(play=False)
        for nome in ("foot_friction", "encoder_bias", "base_com"):
            if getattr(knobs.dr, nome) and nome in fabricante.events:
                cfg.events[nome] = deepcopy(fabricante.events[nome])

    # --- 7. LOCOMOÇÃO DE VOLTA (itens 8, 10, 11) ---
    # O `base_env` apaga tudo que não é equilíbrio, porque a Lift é uma task parada.
    # Aqui a marcha volta COLHIDA do fabricante e apontada pro `"twist"`, sem
    # reescrever nenhuma função: `command_name="twist"` já é o default delas, e o
    # nome do nosso termo de comando é o mesmo do cfg de velocity de propósito.
    r = knobs.reward
    fab = unitree_g1_flat_env_cfg(play=play)
    for nome, peso in (("track_angular_velocity", r.track_angular_velocity),
                       ("foot_clearance", r.foot_clearance),
                       ("foot_swing_height", r.foot_swing_height)):
        cfg.rewards[nome] = deepcopy(fab.rewards[nome])
        cfg.rewards[nome].weight = peso

    # `track_linear_velocity` entra na variante com freio de z (item 11): mesma
    # assinatura e mesmos params do fabricante, uma linha de diferença no meio.
    cfg.rewards["track_linear_velocity"] = deepcopy(fab.rewards["track_linear_velocity"])
    cfg.rewards["track_linear_velocity"].func = R.track_linear_velocity_freio_z
    cfg.rewards["track_linear_velocity"].weight = r.track_linear_velocity

    # DOIS `soft_landing`, sensores diferentes, nomes distintos. A colisão de nome do
    # item 13 não existe aqui porque este cfg nomeia os termos dele.
    cfg.rewards["soft_landing_feet"] = deepcopy(fab.rewards["soft_landing"])
    cfg.rewards["soft_landing_feet"].weight = r.soft_landing_feet
    cfg.rewards["soft_landing_table"] = RewardTermCfg(
        func=vel_mdp.soft_landing, weight=r.soft_landing_table,
        params={"sensor_name": BODY_IMPACT_SENSOR},
    )

    # Anti-dinâmica que a §14 lista e o `base_env` não instala.
    cfg.rewards["arm_vel"] = RewardTermCfg(
        func=base_rewards.joint_vel_l2, weight=r.arm_vel,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[".*(shoulder|elbow|wrist).*"])},
    )
    cfg.rewards["joint_acc"] = RewardTermCfg(
        func=base_rewards.joint_acc_l2, weight=r.joint_acc,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Pesos da §14 nos termos que o `base_env` herdou com o valor da Lift.
    for nome, peso in (("upright", r.upright),
                       ("action_rate_l2", r.action_rate_l2),
                       ("body_ang_vel", r.body_ang_vel),
                       ("angular_momentum", r.angular_momentum),
                       ("self_collisions", r.self_collisions),
                       ("dof_pos_limits", r.dof_pos_limits),
                       ("feet_slip", r.feet_slip)):
        if nome in cfg.rewards:
            cfg.rewards[nome].weight = peso

    # --- 8. POSTURA EM 3 ESCOPOS (item 12, resolvido por REUSO) ---
    # O `base_env` instala UM `posture`. Ele sai e entram três, todos com a MESMA
    # função do fabricante (`base_rewards.posture`) — nenhuma função nova.
    #
    # Por que três e não um `variable_posture`: aquele troca a tolerância pela
    # velocidade COMANDADA, e `parado` e manipulação têm as duas velocidade ~0 —
    # ele não distingue os dois casos, que precisam de std 0.05 e 0.5.
    # Por que não `variable_posture` gateado: ele é uma CLASSE, e o `gated()` chama
    # `inner(env, **kw)` — passar a classe construiria uma instância em vez de
    # chamar. Usar a função simples três vezes é mais barato que fazer o `gated()`
    # entender classe.
    #
    # O `std_walking` é COLHIDO do cfg do fabricante (dict por junta, calibrado por
    # robô), não redigitado.
    del cfg.rewards["posture"]
    std_walking = deepcopy(fab.rewards["pose"].params["std_walking"])
    pernas = SceneEntityCfg("robot", joint_names=list(r.postura_joints))
    corpo_todo = SceneEntityCfg("robot", joint_names=[".*"])

    # São QUATRO termos, não três, porque o escopo e o std são independentes:
    #   escopo   <- a mão está ocupada? (braço é efetuador, não pode ser travado)
    #   std      <- qual o regime de velocidade? (parado / andando / manipulando)
    # As 4 combinações com conteúdo real são as de baixo. Colapsar `andar` com
    # `andar c/ caixa` daria std de marcha com braço travado numa tarefa que carrega
    # caixa; colapsar `andar c/ caixa` com manipulação daria std 0.5 numa tarefa que
    # anda. Nenhuma função nova em nenhum dos quatro.
    for nome, tarefas, std, escopo in (
        # mãos vazias: o braço PODE ser travado, e no `andar` deve — o `std_walking`
        # do fabricante shapa balanço de braço nas entradas de ombro/cotovelo/punho.
        ("posture_parado", (PARADO,), {".*": r.postura_std_parado}, corpo_todo),
        ("posture_anda", (ANDAR,), std_walking, corpo_todo),
        # mão ocupada (ou ocupando): braço LIVRE, escopo perna+cintura.
        ("posture_manip", (PEGAR, BOTAR, REORIENTAR, PARADO_CAIXA),
         {".*": r.postura_std_manipula}, pernas),
        ("posture_carrega", (ANDAR_CAIXA,), _so_pernas(std_walking), pernas),
    ):
        cfg.rewards[nome] = RewardTermCfg(
            func=R.gated, weight=r.postura,
            params={"inner": base_rewards.posture, "tasks": tuple(tarefas),
                    "std": std, "asset_cfg": escopo},
        )

    # --- 9. ANTI-HACKS, com os 2 gates explícitos (§6b) ---
    # Só estes dois precisam de máscara. `foot_clearance`, `foot_swing_height` e
    # `soft_landing_feet` já são auto-gateados por ‖v_desejada‖ pelo fabricante.
    cfg.rewards["com_balance"] = RewardTermCfg(
        func=R.gated, weight=r.com_balance,
        params={"inner": LR.com_over_feet_penalty, "tasks": T.exceto(*T.ANDA),
                "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
                "forward_margin": r.com_margin},
    )
    cfg.rewards["box_shake"] = RewardTermCfg(
        func=R.gated, weight=r.box_shake,
        params={"inner": LR.box_shake_penalty, "tasks": T.exceto(REORIENTAR),
                "object_name": "box"},
    )
    # Estes dois valem em todas as tarefas -> sem gate. O `back_penalty` fica inerte
    # sem caixa na mão (o sensor não dispara), e isso é aceitável.
    cfg.rewards["table_contact"] = RewardTermCfg(
        func=LR.table_contact_penalty, weight=r.table_contact,
        params={"sensor_name": BODY_TABLE_SENSOR},
    )
    cfg.rewards["back_penalty"] = RewardTermCfg(
        func=LR.back_penalty, weight=r.back_penalty,
        params={"back_sensors": BACK_SENSORS},
    )

    # --- 10. RECOMPENSAS DE TAREFA (§6b bloco B) ---
    # Os gates saem da tabela da §6b, coluna por coluna. Os kernels vêm da Lift por
    # import; os NÚMEROS são revisados na Tarefa 12 (ver marcas `T12:` no knobs.py).
    pega = dict(palm_sensors=PALM_SENSORS, back_sensors=BACK_SENSORS)
    palmas = SceneEntityCfg("robot", site_names=PALM_SITES)

    # `lift` — progresso de altura, só no `pegar`. `rest_z_attr` faz o zero do
    # progresso ser POR-ENV: com currículo de altura, cada altura tem seu zero, e um
    # `rest_z` fixo deixaria as alturas baixas sem gradiente nenhum.
    cfg.rewards["lift"] = RewardTermCfg(
        func=R.gated, weight=r.lift,
        params={"inner": LR.lift_reward, "tasks": (PEGAR,),
                "object_name": "box", "command_name": "lift_target",
                "rest_z": box_z, "upright_std": r.upright_std,
                "rest_z_attr": None, **pega},
    )
    # `reaching` — shaping. ANELA com catraca (item 22): o currículo baixa este peso
    # conforme a competência sobe, e ele nunca volta a subir. Para em 0.01, não em 0,
    # porque `weight == 0.0` faz o RewardManager PULAR o termo e ele sai do log.
    cfg.rewards["reaching"] = RewardTermCfg(
        func=R.gated, weight=r.reaching,
        params={"inner": LR.reaching_reward, "tasks": (REORIENTAR, PEGAR, BOTAR),
                "std_coarse": r.std_coarse, "std_fine": r.std_fine,
                "object_name": "box", "asset_cfg": palmas,
                "lateral_offset": s.box_half[1]},
    )
    cfg.rewards["grasp"] = RewardTermCfg(
        func=R.gated, weight=r.grasp,
        params={"inner": LR.grasp_reward, "tasks": (PEGAR,), **pega},
    )
    cfg.rewards["box_at_peito"] = RewardTermCfg(
        func=R.gated, weight=r.box_at_peito,
        params={"inner": R.box_at_peito,
                "tasks": (PEGAR, PARADO_CAIXA, ANDAR_CAIXA),
                "std": r.sustain_std, "object_name": "box",
                "alvo_peito_b": c.alvo_peito_b, **pega},
    )
    cfg.rewards["box_at_prateleira"] = RewardTermCfg(
        func=R.gated, weight=r.box_at_prateleira,
        params={"inner": R.box_at_prateleira, "tasks": (BOTAR,),
                "std": r.sustain_std, "object_name": "box",
                "command_name": "lift_target",
                "fracao_solta": r.botar_fracao_solta, **pega},
    )
    cfg.rewards["orienta_face"] = RewardTermCfg(
        func=R.gated, weight=r.kernel_angulo,
        params={"inner": R.orienta_face, "tasks": (REORIENTAR,),
                "command_name": "lift_target",
                "std_grosso_deg": r.angulo_std_grosso_deg,
                "std_fino_deg": r.angulo_std_fino_deg,
                "xy_std": r.reorienta_xy_std},
    )
    cfg.rewards["hold_still"] = RewardTermCfg(
        func=R.gated, weight=r.hold_still,
        params={"inner": LR.hold_still_bonus,
                "tasks": (PARADO_CAIXA, ANDAR_CAIXA, PEGAR),
                "object_name": "box", "command_name": "lift_target",
                "gate_std": 0.25, "still_std": 0.5, **pega},
    )

    # --- 11. CURRÍCULO: quem sorteia a tarefa ---
    # Tem que ser curriculum term, não evento nem comando: no reset a ordem é
    # currículo (:554) -> eventos (:560) -> comando (:581), e o evento de spawn
    # segurando precisa saber a tarefa antes de rodar.
    cfg.curriculum["orquestrador"] = CurriculumTermCfg(
        func=C.Orquestrador,
        params={"curriculum": knobs.curriculum, "min_amostras_evento": 200},
    )

    # --- 12a. AFASTA a prateleira de quem não a usa ---
    # Ela mora em x=0.50 com meia-extensão 0.30, ou seja x de 0.20 a 0.80 com topo em
    # 0.55 m — altura de joelho. E o destino do `andar` é
    # `pos_robô + (d·cos(head), d·sin(head), 0)` com d até 2.0 m: com heading perto de
    # zero o robô anda DIRETO CONTRA ela. Ela não é parte da tarefa `andar`.
    # Descoberto no `play` em 30/07; a run monolítica tem o mesmo bug e só não mostrou
    # porque nunca abriu o `andar`.
    #
    # ANTES do `reset_segurando`: as tarefas de `SPAWN_SEGURANDO` têm a caixa afastada
    # aqui e o evento seguinte a traz de volta para as palmas. Invertido, ela sairia
    # das mãos.
    cfg.events["afasta_cena"] = EventTermCfg(
        func=MT_events.afasta_cena, mode="reset",
        params={"tarefas_com_prateleira": T.MANIPULA,
                # a caixa só fica onde ela é usada de fato: quem manipula, e quem
                # nasce segurando (esses o `reset_segurando` reposiciona logo depois)
                "tarefas_com_caixa": tuple(set(T.MANIPULA) | set(T.SPAWN_SEGURANDO)),
                "shelf_half_z": s.shelf_half_z,
                "box_half_z": s.box_half[2],
                "distancia": s.afasta_distancia},
    )

    # --- 12b. SPAWN SEGURANDO das 3 tarefas c/ caixa ---
    # POR ÚLTIMO no dict de eventos, de propósito: ele sobrescreve o que o
    # `reset_robot_joints` e o `reset_box` da base acabaram de escrever, e o
    # event manager percorre na ordem de inserção.
    cfg.events["reset_segurando"] = EventTermCfg(
        func=MT_events.reset_segurando, mode="reset",
        params={"pose_bracos": POSE_PRE_GRASP, "tarefas": T.SPAWN_SEGURANDO},
    )

    # --- 12c. a QUEDA deixa de encerrar o episódio ---
    # Ver `knobs.Tolerancia.termina_ao_cair` para o porquê e para a medição que autoriza.
    # Em resumo: terminar zera o valor futuro, e com reward por passo negativa isso faz
    # morrer cedo RENDER MAIS — foi o vale que as duas runs de 30/07 atravessaram. E a
    # política nunca via o estado caído, então não podia aprender a levantar.
    # O `metrics.Sucesso` não depende desta terminação: ele replica o critério de 70° numa
    # flag própria, então o sucesso do `parado` continua significando "não caiu".
    if not knobs.tolerancia.termina_ao_cair:
        cfg.terminations.pop("fell_over", None)

    # --- 13. TERMINAÇÕES (item 21) ---
    # `time_out`, `fell_over` (70°) e `nonfinite` já vêm herdados. As 3 daqui são as
    # que faltam da §6b/D, e as duas primeiras são GATEADAS: a distinção `pegar` ×
    # `carregar` é o ponto — largar no `pegar` deve dar nova chance no mesmo episódio.
    cfg.terminations["largou"] = TerminationTermCfg(
        func=MT_terms.largou, time_out=False,
        params={"tasks": T.COM_CAIXA, "z_min": t_.largou_z},
    )
    cfg.terminations["caixa_caiu"] = TerminationTermCfg(
        func=MT_terms.caixa_caiu, time_out=False,
        params={"tasks": (REORIENTAR,), "margem": s.box_half[2],
                "shelf_half_z": s.shelf_half_z},
    )
    cfg.terminations["fora_da_area"] = TerminationTermCfg(
        func=MT_terms.fora_da_area, time_out=False,
        params={"raio": t_.area_raio},
    )

    # --- 14. SUCESSO por tarefa -> env.success_buf (itens 17 e 18) ---
    # Métrica FÍSICA, fora do reward manager. É ela que faz "ajustar peso entre
    # blocos" ser Categoria A (grátis) em vez de Categoria C disfarçada.
    # `reduce="last"` porque sucesso é binário do EPISÓDIO: a média por passo diria
    # "0.3 de sucesso" pra um episódio que teve sucesso nos últimos 30%% dos passos.
    cfg.metrics["sucesso"] = MetricsTermCfg(
        func=MT_metrics.Sucesso, reduce="last",
        params={"tol": t_, "alvo_peito_b": c.alvo_peito_b},
    )
    # F3: a deriva do `parado` vira LOG, não portão.
    cfg.metrics["deriva_parado"] = MetricsTermCfg(
        func=MT_metrics.deriva_parado, reduce="max",
    )

    # --- 15. OBSERVABILIDADE por tarefa x termo (tarefa nova, fora da §13) ---
    # O log default do mjlab dilui: `Episode_Reward/<termo>` é média sobre TODOS os
    # envs, e com 7 tarefas intercaladas um termo gateado aparece com ~1/7 da
    # magnitude real. Sem esta separação não há como saber qual termo domina em qual
    # tarefa, que é a pergunta que se faz entre blocos de 2k-3k.
    #
    # DOIS termos cooperando: o de métrica acumula a cada passo (de graça, lendo o
    # `_step_reward` que o RewardManager já preenche), e o de currículo emite o dict
    # no reset — que é o único canal do mjlab pra log de chave/valor arbitrário.
    cfg.metrics["reward_total"] = MetricsTermCfg(func=MT_obs.Contribuicao)
    cfg.curriculum["contrib"] = CurriculumTermCfg(
        func=MT_obs.Relatorio, params={"min_amostras": 500, "top": 0},
    )

    # ESCALA DE AÇÃO (estrutural): encolhe o G1_ACTION_SCALE por-junta por um fator
    # global. Não entra na soma de reward -> não compete com nenhum termo.
    mult = knobs.foundation.action_scale_mult
    if mult != 1.0:
        act = cfg.actions["joint_pos"]
        if isinstance(act.scale, dict):
            act.scale = {k: v * mult for k, v in act.scale.items()}
        else:
            act.scale = act.scale * mult

    return cfg
