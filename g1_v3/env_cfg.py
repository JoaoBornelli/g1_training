"""Monta o env da v3 sobre o molde de locomoção do FABRICANTE (D10 do plano; §6.5, §6.6, §6.8; S §19.1).

⚠ ZERO IMPORT DE CÓDIGO DO g1_limpo. O molde é `unitree_g1_rough_env_cfg` do mjlab (sim e viewer). Tudo o que
é nosso entra por TROCA de campo: cena, sensores, terreno, observações, ação, comandos, recompensas, terminações,
eventos, currículo e métricas. O molde não fica no cfg: o `rewards`, o `events` e os demais dicts são substituídos
inteiros, para que nenhum termo do fabricante sobreviva por engano.

⚠ Todo número vem do `knobs.py`. As constantes deste módulo são as tuplas de tarefa de cada termo (§6.6), os geoms
dos pés e o episódio de play (D10).
"""
from __future__ import annotations

import dataclasses
import re

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from g1_v3 import cena, comando, curriculo, eventos, metricas, observacoes, recompensas, terminacoes
from g1_v3.knobs import ATIVO, Knobs

# Tarefas de cada termo (§6.6; a tabela da D5 e a S §9.2). "todas" = as quatro da cadeia.
TODAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")
ANDAR_CARREGAR = ("ANDAR", "CARREGAR")
PEGAR_BOTAR = ("PEGAR", "BOTAR")
PEGAR_CARREGAR_BOTAR = ("PEGAR", "CARREGAR", "BOTAR")
# E11 R1 (29/10): o `progresso` não vale no CARREGAR (o índice fica no P4 e o termo cobraria (1 − γ)·w_p·Φ(P4)).
ANDAR_PEGAR_BOTAR = ("ANDAR", "PEGAR", "BOTAR")

# §6.5, na ordem do dict. Nome = função em `recompensas` = campo de `knobs.Recompensa` (o peso cru).
_RECOMPENSAS = (
    ("rastreio_linear", TODAS), ("rastreio_angular", TODAS), ("passo", ANDAR_CARREGAR),
    ("vivo", ANDAR_CARREGAR), ("altura_pe", ANDAR_CARREGAR), ("escorrego", TODAS), ("vz", TODAS),
    ("giro_xy", TODAS), ("vel_junta", TODAS), ("acel_junta", TODAS), ("taxa_acao", TODAS),
    ("energia", TODAS), ("limite_junta", TODAS), ("desvio_bracos", ANDAR_CARREGAR),
    ("desvio_cintura", ANDAR_CARREGAR), ("desvio_quadril", ANDAR_CARREGAR),
    ("tronco_plano", ANDAR_CARREGAR), ("altura_base", ANDAR_CARREGAR), ("cerca_sola", PEGAR_BOTAR),
    ("cerca_giro", TODAS), ("cerca_tronco", TODAS), ("cerca_com", TODAS), ("cerca_torque", TODAS),
    ("cerca_pelve", TODAS), ("cerca_recuo", PEGAR_CARREGAR_BOTAR), ("contato_fora", TODAS),
    ("contato_laje", PEGAR_BOTAR), ("freio", PEGAR_CARREGAR_BOTAR), ("progresso", ANDAR_PEGAR_BOTAR),
    ("sacudida", PEGAR_CARREGAR_BOTAR), ("aperto", PEGAR_CARREGAR_BOTAR),
    ("perda_caixa", TODAS), ("terminacao", TODAS),
)
# Pesos que NÃO vêm do `knobs`: o `progresso` e o `perda_caixa` entram com ±1 no cfg, e a escala mora no núcleo
# (§6.3, bloco da fase E: w_p, B_alvo e o custo da perda ficam em `recompensas.py`).
_PESO_FIXO = {"progresso": 1.0, "perda_caixa": -1.0}

# Geoms de colisão dos pés (o mesmo nome que o molde do mjlab usa em `foot_friction`).
_GEOMS_DOS_PES = tuple(f"{lado}_foot{i}_collision" for lado in ("left", "right") for i in range(1, 8))

# ⚠ Episódio "infinito" no play (§6.5, D10): quem termina é o `v3_fim`, não o relógio.
_EPISODIO_PLAY_S = int(1e9)


def _sem_none(obj, caminho: str = "knobs") -> None:
    """⚠ `None` no knobs quer dizer "sai da Parte 0" (knobs.py:12): nenhum `None` pode chegar ao cfg.

    Desce pelas dataclasses e pelas tuplas. O assert aponta o caminho do campo, para o erro dizer onde está.
    """
    assert obj is not None, f"{caminho} é None: campo da Parte 0 sem valor"
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            _sem_none(getattr(obj, f.name), f"{caminho}.{f.name}")
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            _sem_none(v, f"{caminho}[{i}]")


def make_env_cfg(k: Knobs = ATIVO, play: bool = False) -> ManagerBasedRlEnvCfg:
    """O cfg da v3. `play=True`: ator sem corrupção, sem empurrão nem as DR do refino (P24), sem o currículo `v3`,
    episódio de 10⁹ s (o `v3_fim` acaba a cadeia).

    Do `k` só valem os pesos (`recompensa`), `comando`, `cadeia`, `estimador`, `terreno`, `cena`, `contrato` e
    `percepcao.ligada`. O `dr` do `k` vale só para QUAIS eventos entram no cfg (o `com_caixa`, por exemplo); as faixas
    dos eventos vêm do `Curriculo`, que lê o `ATIVO.dr` (`curriculo.py:47`). O resto dos termos (σ, limites, ruído do
    lidar, terminação, balanceador, currículo) também lê o `ATIVO` do módulo: um `k` diferente não o muda.
    """
    _sem_none(k)
    c = k.cena
    cfg = unitree_g1_rough_env_cfg(play=play)

    # ------------------------------------------------------------ 1. física, cena e terreno
    cfg.scene.entities = cena.entidades(k)
    cfg.scene.sensors = cena.sensores(k)  # os 11 do g1_limpo + os 7 de altura/contato; o lidar só com Percepcao.ligada
    cfg.scene.terrain = cena.terreno_cfg(k)
    cfg.scene.extent = c.extent
    cfg.sim.njmax = c.njmax
    cfg.sim.nconmax = c.nconmax
    cfg.sim.mujoco.ccd_iterations = c.ccd_iterations
    cfg.sim.mujoco.impratio = c.impratio
    cfg.sim.mujoco.cone = c.cone
    cfg.sim.mujoco.timestep = k.contrato.dt_fisica
    cfg.decimation = k.contrato.decimacao
    cfg.episode_length_s = _EPISODIO_PLAY_S if play else k.cadeia.episodio_s

    # ------------------------------------------------------------ 2. ação (§6.2: escala 0,25, pose de pé como offset)
    cfg.actions = {"joint_pos": JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=k.contrato.escala_acao, use_default_offset=True)}

    # ------------------------------------------------------------ 3. observações (§6.5: ator e crítico)
    cfg.observations = observacoes.grupos(k)
    if play:
        cfg.observations["actor"].enable_corruption = False

    # ------------------------------------------------------------ 4. comandos: `cadeia` primeiro, `twist` depois
    teto_frente, teto_lado, teto_giro = k.comando.frente_tetos[-1], k.comando.tras_lado_teto_max, k.comando.giro_tetos[-1]
    cfg.commands = {
        "cadeia": comando.CadeiaCmdCfg(k=k.cadeia, est=k.estimador, entity_name="robot"),
        # `ranges` é o envelope do `knobs` que o exportador lê (`envelope_treino`); o sorteio não o usa.
        # `resampling_time_range` = `Comando.reamostra_s` (§6.2).
        "twist": comando.TwistV3Cfg(
            k=k.comando, entity_name="robot", resampling_time_range=k.comando.reamostra_s,
            ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-teto_frente, teto_frente),
                                                    lin_vel_y=(-teto_lado, teto_lado),
                                                    ang_vel_z=(-teto_giro, teto_giro))),
    }

    # ------------------------------------------------------------ 5. recompensas (peso cru: o mjlab multiplica por dt)
    cfg.rewards = {nome: RewardTermCfg(func=getattr(recompensas, nome),
                                       weight=_PESO_FIXO[nome] if nome in _PESO_FIXO else getattr(k.recompensa, nome),
                                       params={"tarefas": tarefas})
                   for nome, tarefas in _RECOMPENSAS}

    # ------------------------------------------------------------ 6. terminações (§6.5; `time_out` não conta como queda)
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=terminacoes.fim_da_cadeia, time_out=True),
        # A queda vale nas quatro tarefas (D6): o `n_queda` separa ANDAR/CARREGAR de PEGAR/BOTAR por dentro, por par.
        "queda": TerminationTermCfg(func=terminacoes.queda, params={"tarefas": TODAS}),
        "caixa_perdida": TerminationTermCfg(func=terminacoes.caixa_perdida, params={
            "tarefas": PEGAR_CARREGAR_BOTAR, "vira_andar": k.cadeia.caixa_perdida_vira_andar}),
        # F-2: a caixa tocou a laje descendo acima do limite do degrau (só no BOTAR).
        "chegada": TerminationTermCfg(func=terminacoes.chegada, params={"tarefas": ("BOTAR",)}),
        # H (P13/P14): junta fora do batente, nas 4 tarefas; liga a partir do degrau (§6.9), dentro da função.
        "batente": TerminationTermCfg(func=terminacoes.batente, params={"tarefas": TODAS}),
    }

    # ------------------------------------------------------------ 7. eventos (§6.5 na ordem; §6.8 os da fase D)
    # ⚠ P18: `actuator_ids` indexa os GRUPOS de atuador (um por linha de `Contrato.pd`): braços = shoulder, elbow, wrist.
    bracos = [i for i, (regex, _, _) in enumerate(k.contrato.pd) if re.search("shoulder|elbow|wrist", regex)]
    pernas = [i for i in range(len(k.contrato.pd)) if i not in bracos]
    eventos_ = {
        "fatia": EventTermCfg(func=eventos.sorteia_fatia, mode="reset", params={"k_ter": k.terreno}),
        # ⚠ E11 S4 (29/10): a `cena` roda ANTES da `base`: a base das fatias B e C se põe em relação à caixa.
        "cena": EventTermCfg(func=eventos.posiciona_cena, mode="reset"),
        "base": EventTermCfg(func=eventos.reset_base_por_fatia, mode="reset", params={
            "faixa_andar": dict(c.reset_base_andar), "faixa_manipula": dict(c.reset_base_manipula)}),
        "juntas": EventTermCfg(func=reset_joints_by_offset, mode="reset",
                               params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}),
        "carga": EventTermCfg(func=eventos.carga, mode="reset"),
        "atrito_pad": EventTermCfg(func=eventos.atrito_pad, mode="reset", params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=cena.PALM_PAD_GEOMS)}),
        "atrito_laje": EventTermCfg(func=eventos.atrito_laje, mode="reset", params={
            "asset_cfg": SceneEntityCfg("table", geom_names=(cena.TABLE_GEOM,))}),
        "vies_imu": EventTermCfg(func=eventos.vies_imu, mode="reset"),
        # H (30/10): pd, forca_motor e solref saem do play (P24); o com_caixa só entra com a chave (PD-38, P19).
        # O `**` põe cada grupo na posição da §6.5 nos dois casos (play e treino).
        **({
            "pd": EventTermCfg(func=eventos.pd, mode="reset", params={"asset_cfg": SceneEntityCfg("robot")}),
            # M10: com só `actuator_ids`, o resolve põe em `actuator_names` os nomes por junta (não por grupo); o
            # `effort_limits` só lê os ids, então o nome não entra na conta.
            "forca_motor": EventTermCfg(func=eventos.forca_motor, mode="reset", params={
                "asset_cfg_pernas": SceneEntityCfg("robot", actuator_ids=pernas),
                "asset_cfg_bracos": SceneEntityCfg("robot", actuator_ids=bracos)}),
            # P17: o solref das geoms que mandam no contato (pés, pads, laje e caixa); terreno fica de fora.
            "solref": EventTermCfg(func=eventos.solref, mode="reset", params={
                "asset_cfg_robo": SceneEntityCfg("robot", geom_names=_GEOMS_DOS_PES + cena.PALM_PAD_GEOMS + cena.BACK_PAD_GEOMS),
                "asset_cfg_laje": SceneEntityCfg("table", geom_names=(cena.TABLE_GEOM,)),
                "asset_cfg_caixa": SceneEntityCfg("box", geom_names=(cena.BOX_GEOM,))}),
        } if not play else {}),
        **({"com_caixa": EventTermCfg(func=eventos.com_caixa, mode="reset", params={
            "asset_cfg": SceneEntityCfg("box", body_names=("box",))})} if k.dr.com_caixa else {}),
        "filtros": EventTermCfg(func=eventos.reinicia_filtros, mode="reset"),
        "atrito_pe": EventTermCfg(func=dr.geom_friction, mode="startup", params={
            "ranges": k.dr.atrito_pe, "operation": "abs", "shared_random": True,
            "asset_cfg": SceneEntityCfg("robot", geom_names=_GEOMS_DOS_PES)}),
        "massa_tronco": EventTermCfg(func=dr.body_mass, mode="startup", params={
            "ranges": k.dr.massa_tronco, "operation": "add",
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))}),
        "vies_encoder": EventTermCfg(func=dr.encoder_bias, mode="startup", params={
            "bias_range": (-k.dr.vies_encoder, k.dr.vies_encoder), "asset_cfg": SceneEntityCfg("robot")}),
        "tamanho_caixa": EventTermCfg(func=eventos.tamanho_caixa, mode="startup", params={
            "faixa": k.cena.caixa_meia_faixa, "n_variantes": k.cena.caixa_n_variantes,
            "asset_cfg": SceneEntityCfg("box", geom_names=(cena.BOX_GEOM,))}),
        # ⚠ `atraso` baixa o teto do buffer para `atraso_inicio[1]` (1 passo); a capacidade do build fica (observacoes.py).
        "atraso": EventTermCfg(func=eventos.atraso, mode="startup", params={
            "passos_max": k.dr.atraso_inicio[1], "decimacao": k.contrato.decimacao}),
    }
    if not play:  # ⚠ no play saem o empurrão e as DR do refino: a corrida é a cadeia, o robô é nominal (P24)
        eventos_["empurrao"] = EventTermCfg(
            func=eventos.empurrao, mode="interval",
            interval_range_s=(k.dr.periodo_empurrao_s, k.dr.periodo_empurrao_s),
            params={"velocity_range": {"x": (-k.dr.empurrao, k.dr.empurrao),
                                       "y": (-k.dr.empurrao, k.dr.empurrao)}})
        # H (P3, P4, P15): passo, depois do comando, nos 4 subpassos. A força é mascarada dentro da classe.
        eventos_["forca_caixa"] = EventTermCfg(func=eventos.forca_caixa, mode="step", params={
            "force_range": k.dr.forca_caixa_N, "torque_range": (0.0, 0.0), "duration_s": k.dr.forca_caixa_dur_s,
            "cooldown_s": k.dr.forca_caixa_periodo_s, "asset_cfg": SceneEntityCfg("box", body_names=("box",))})
        eventos_["rfi"] = EventTermCfg(func=eventos.rfi, mode="step")
    cfg.events = eventos_

    # ------------------------------------------------------------ 8. currículo (um termo só, `v3`) e métricas
    # D14 (29/10): no play não há o termo `v3`. O estado carregado do checkpoint fica congelado; os buffers da cadeia
    # nascem no `CadeiaCmd.__init__`, e não no termo.
    cfg.curriculum = {} if play else {"v3": CurriculumTermCfg(func=curriculo.v3)}
    cfg.metrics = metricas.termos(k)

    return cfg
