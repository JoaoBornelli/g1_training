"""Sucesso por tarefa — a RÉGUA do currículo, e ela é física.

Este módulo é pré-requisito do workflow de blocos de 2k-3k, não um extra. O §15
divide as intervenções em 3 categorias de custo, e mudar **peso de reward** só é
Categoria A ("grátis, retoma do checkpoint") porque o sucesso mora aqui, em
`env.success_buf`, como **fato físico** — mexer em peso não move a régua e não
invalida nenhuma EMA nem nenhum limiar do orquestrador.

Se o sucesso fosse soma de reward, como o `PlrHeights` da Lift faz
(`reward_manager._episode_sums["sustain_precise"]`), toda mudança de peso seria
**Categoria C disfarçada** — recomeçar do zero — e o ajuste entre blocos deixaria de
existir como ferramenta.

Uma computação, dois consumidores: o currículo lê `env.success_buf`, o
`MetricsManager` loga o retorno com `reduce="last"` (que o docstring do mjlab
descreve como sendo exatamente pra métrica binária de sucesso).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from g1_training.base_env import BACK_SENSORS, PALM_SENSORS, SUPPORT_SENSOR
from g1_training.skills.lift.rewards import _contact, _grasp

from . import tasks as T
from .terminations import de_pe

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.managers.metrics_manager import MetricsTermCfg

ALVO = slice(0, 3)


class Sucesso:
    """Critério de sucesso por tarefa, com sustentação temporal (§6b/E).

    Stateful: o mjlab auto-instancia com `(cfg, env)`. Guarda um contador de passos
    em que a condição vale seguido, e um flag `conquistado` que **persiste até o fim
    do episódio** — o sucesso é do EPISÓDIO, não do passo.

    ⚠️ **Detecta o reset sozinho.** O `MetricsManager.reset` limpa só os buffers
    dele; ele NÃO chama `reset()` nos termos-classe. Então aqui a gente vê o reset
    pela QUEDA do `episode_length_buf`, o que também é robusto ao
    `init_at_random_ep_len=True` que o `train.py` do mjlab passa (ele randomiza o
    comprimento no começo do treino, e um teste de `== 0` daria falso negativo)."""

    def __init__(self, cfg: "MetricsTermCfg", env: "ManagerBasedRlEnv"):
        dev = env.device
        p = cfg.params
        self.tol = p["tol"]
        self.alvo_peito_b = torch.tensor(p["alvo_peito_b"], device=dev)
        self.dt = env.step_dt

        n = env.num_envs
        self._contador = torch.zeros(n, device=dev)
        self._conquistado = torch.zeros(n, dtype=torch.bool, device=dev)
        self._len_ant = torch.zeros(n, dtype=torch.long, device=dev)
        self._nunca_caiu = torch.ones(n, dtype=torch.bool, device=dev)

        self._palmas = SceneEntityCfg("robot", site_names=[])
        # buffer criado ANTES do 1º reset: o currículo e o orquestrador leem daqui
        env.success_buf = torch.zeros(n, device=dev)

    # ------------------------------------------------------------- primitivas
    def _condicao(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        """[B] bool — a condição da tarefa ATIVA de cada env vale AGORA."""
        tol = self.tol
        robo: Entity = env.scene["robot"]
        caixa: Entity = env.scene["box"]
        meta = env.command_manager.get_term("lift_target")
        tarefa = env.active_task

        # --- primitivas compartilhadas, computadas uma vez ---
        parado_de_pe = de_pe(env, tol.de_pe_z, tol.de_pe_tilt_rad)
        preensao = _grasp(env, PALM_SENSORS, BACK_SENSORS) > 0.5
        apoiada = _contact(env, SUPPORT_SENSOR) > 0.5
        caixa_quieta = caixa.data.root_link_lin_vel_w.norm(dim=-1) < tol.caixa_quieta_v

        peito_w = robo.data.root_link_pos_w + quat_apply(
            robo.data.root_link_quat_w,
            self.alvo_peito_b.expand(env.num_envs, 3))
        no_peito = (caixa.data.root_link_pos_w - peito_w).norm(dim=-1) < tol.caixa_no_alvo
        no_alvo = (caixa.data.root_link_pos_w
                   - meta.command[:, ALVO]).norm(dim=-1) < tol.caixa_no_alvo
        chegou = (robo.data.root_link_pos_w[:, :2]
                  - meta.command[:, ALVO][:, :2]).norm(dim=-1) < tol.andar_raio
        orientada = ((meta.erro_angulo_deg() < tol.reorienta_angulo_deg)
                     & (meta.desvio_xy() < tol.reorienta_xy) & apoiada)

        # --- combinação por tarefa (§6b/E) ---
        cond = torch.zeros_like(parado_de_pe)
        # `parado`: sobreviveu os 20 s. F3 — a deriva de 0.20 m é LOG, não portão:
        # como portão ela comprimia a taxa de sucesso a ~0 sob push nível 4.
        cond = torch.where(tarefa == T.PARADO,
                           env.termination_manager.time_outs & self._nunca_caiu, cond)
        # `andar`: chegou no raio e ficou de pé. Não há limiar de "quieto" porque o
        # `d_morto` leva a velocidade comandada a ZERO no alvo — o robô para pelo
        # perfil, não por penalidade. Ver §4.
        cond = torch.where(tarefa == T.ANDAR, chegou & parado_de_pe, cond)
        cond = torch.where(tarefa == T.PEGAR, no_peito & parado_de_pe, cond)
        cond = torch.where(tarefa == T.BOTAR,
                           no_alvo & ~preensao & caixa_quieta & parado_de_pe, cond)
        cond = torch.where(tarefa == T.REORIENTAR, orientada, cond)
        cond = torch.where(tarefa == T.PARADO_CAIXA,
                           no_peito & preensao & parado_de_pe, cond)
        cond = torch.where(tarefa == T.ANDAR_CAIXA,
                           chegou & parado_de_pe & no_peito & preensao, cond)
        return cond

    def _exigencia_s(self, env) -> torch.Tensor:
        """[B] — quantos segundos a condição tem que valer seguido, por tarefa."""
        tol, tarefa = self.tol, env.active_task
        s = torch.full_like(env.success_buf, tol.sustenta_pegar_s)
        s = torch.where(tarefa == T.PARADO, torch.zeros_like(s), s)   # instantâneo
        s = torch.where(tarefa == T.ANDAR, torch.full_like(s, tol.sustenta_andar_s), s)
        s = torch.where(tarefa == T.ANDAR_CAIXA,
                        torch.full_like(s, tol.sustenta_andar_s), s)
        s = torch.where(tarefa == T.BOTAR, torch.full_like(s, tol.sustenta_botar_s), s)
        s = torch.where(tarefa == T.REORIENTAR,
                        torch.full_like(s, tol.sustenta_reorienta_s), s)
        return s

    # ------------------------------------------------------------------ termo
    def __call__(self, env: "ManagerBasedRlEnv", **params) -> torch.Tensor:
        # `**params` engole `tol` e `alvo_peito_b`: o manager repassa os `cfg.params`
        # como kwargs em toda chamada, e eles já foram resolvidos no `__init__`.
        del params
        # 1. reset detectado pela QUEDA do comprimento do episódio
        caiu = env.episode_length_buf < self._len_ant
        if bool(caiu.any()):
            self._contador[caiu] = 0.0
            self._conquistado[caiu] = False
            self._nunca_caiu[caiu] = True
        self._len_ant.copy_(env.episode_length_buf)

        # 2. quem CAIU nunca mais "sobreviveu" neste episódio.
        #
        # ⚠️ Duas fontes, e a primeira é obrigatória: com `termina_ao_cair = False` a
        # queda NÃO entra em `terminated`, então ler só o manager deixaria o `parado`
        # dar sucesso a um robô que passou 18 s no chão. O critério replicado é o do
        # `fell_over` do fabricante — inclinação do torso acima de `limite_queda`.
        #
        # `projected_gravity_b[:, 2]` é −cos(inclinação): vale −1 em pé e sobe pra 0
        # deitado. A 70° o corte é −cos(70°) = −0.342.
        caiu_agora = (env.scene["robot"].data.projected_gravity_b[:, 2]
                      > -math.cos(self.tol.limite_queda_rad))
        self._nunca_caiu &= ~caiu_agora
        # a segunda pega `fora_da_area` e `nonfinite`, que continuam terminando
        self._nunca_caiu &= ~env.termination_manager.terminated

        # 3. sustentação: soma enquanto vale, ZERA quando quebra
        cond = self._condicao(env)
        self._contador = torch.where(cond, self._contador + self.dt,
                                     torch.zeros_like(self._contador))
        self._conquistado |= cond & (self._contador >= self._exigencia_s(env))

        env.success_buf.copy_(self._conquistado.float())
        return env.success_buf


def deriva_parado(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Deriva do robô em relação à origem do ambiente. [B] — SÓ LOG (F3).

    Era o critério de sucesso antigo do `parado` (< 0.20 m). Saiu porque comprime a
    taxa de sucesso pra perto de zero sob push nível 4, e o modo de falha é
    silencioso: o orquestrador leria "não competente" quando o robô está de pé e
    firme, só deslocado. Fica logada porque o número continua interessante."""
    robo: Entity = env.scene["robot"]
    return (robo.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).norm(dim=-1)
