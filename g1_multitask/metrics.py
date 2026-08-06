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
        # --- `parado`: acumulador CUMULATIVO de "de pé e devagar" ---
        # Cumulativo e não consecutivo: o user pediu 80% dos passos, e um empurrão no
        # meio do episódio não deve zerar o que já valeu. O `_contador` acima é
        # consecutivo (zera quando quebra) e serve às outras tarefas.
        self._quieto_passos = torch.zeros(n, device=dev)
        self._frac_quieto = torch.zeros(n, device=dev)
        # --- `andar` (S6): flags de DISPARO da histerese, travados pelo episódio ---
        # `chegou` e `alinhado` disparam num limiar apertado e depois passam a ser
        # avaliados num limiar folgado. O flag guarda o disparo; a condição continua
        # sendo reavaliada, com o raio de manutenção. Ver `_condicao`.
        self._chegou_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self._alinhado_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        # --- `parado` (S7): segundos ACUMULADOS fora de `de pé` ---
        # Acumulado e não consecutivo: sob push o robô sai e volta várias vezes, e um
        # contador consecutivo perdoaria dez quedas curtas seguidas.
        self._fora_de_pe_s = torch.zeros(n, device=dev)

        self._palmas = SceneEntityCfg("robot", site_names=[])
        # --- item 4: diagnóstico por tarefa, emitido pelo `observability.Relatorio` ---
        # coluna 0 = `cond_fisica`: a condição da tarefa SORTEADA vale agora?
        # coluna 1 = `atribuicao_divergente`: houve sucesso com a condição dela FALSA?
        # Existe porque o log de 31/07 mostrou `perf[pegar] = 0,98` com `grasp = 0`, e
        # levaram 7 simulações locais pra estreitar isso. Com estas duas linhas o log
        # responde na iteração 1: se `perf` sobe e `cond_fisica` fica em zero, o crédito
        # é falso.
        env.diag_soma = torch.zeros(T.NUM_TASKS, 2, device=dev)
        env.diag_cont = torch.zeros(T.NUM_TASKS, device=dev)
        # buffer criado ANTES do 1º reset: o currículo e o orquestrador leem daqui
        env.success_buf = torch.zeros(n, device=dev)

    # ------------------------------------------------------------- primitivas
    def _condicao(self, env: "ManagerBasedRlEnv",
                  tarefa: torch.Tensor | None = None) -> torch.Tensor:
        """[B] bool — a condição de uma tarefa vale AGORA, por env.

        `tarefa=None` usa `env.active_task`, que é o que PONTUA. Passando
        `env.tarefa_sorteada` sai a condição da tarefa que o CURRÍCULO credita — as duas
        divergem na janela de pré-gatilho, e é dessa diferença que sai o
        `atribuicao_divergente` do log."""
        tol = self.tol
        robo: Entity = env.scene["robot"]
        caixa: Entity = env.scene["box"]
        meta = env.command_manager.get_term("lift_target")
        tarefa = env.active_task if tarefa is None else tarefa

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
        d_alvo = (robo.data.root_link_pos_w[:, :2]
                  - meta.command[:, ALVO][:, :2]).norm(dim=-1)
        chegou = d_alvo < tol.andar_raio
        # --- `andar` (S6): histerese em `chegou` e em `alinhado` ---
        # O flag de disparo é escrito no `__call__`; aqui só se lê. A condição CONTINUA
        # sendo reavaliada, mas contra o raio de MANUTENÇÃO, que é mais folgado.
        #
        # ⚠️ Por que histerese e não um raio único: o `cond` sustentado zera o contador
        # quando a condição quebra. Sob push nível 4 — 50 N por até 3 s — o robô sai de
        # um círculo de 0.10 m e o contador reinicia sem parar. O `andar` nunca chegaria
        # a 0.90, e ele é pai de `pegar` e de `reorientar`.
        #
        # ⚠️ Por que não travar a condição de vez: travada, o robô chega, sai andando
        # para longe e ainda pontua se ficar 3 s de pé. A histerese fecha esse buraco e
        # continua dando a folga que o push exige.
        #
        # `parado_de_pe` NÃO entra na histerese, de propósito: cair tem que zerar.
        chegou_andar = self._chegou_ok & (d_alvo < tol.andar_raio_mantem)
        alinhado = self._alinhado_ok & (meta.erro_rumo_deg() < tol.alinhado_mantem_deg)
        orientada = ((meta.erro_angulo_deg() < tol.reorienta_angulo_deg)
                     & (meta.desvio_xy() < tol.reorienta_xy) & apoiada)

        # --- combinação por tarefa (§6b/E) ---
        cond = torch.zeros_like(parado_de_pe)
        # `parado`: sobreviveu os 20 s E está DE PÉ no fim. F3 — a deriva de 0.20 m é
        # LOG, não portão: como portão ela comprimia a taxa de sucesso a ~0 sob push
        # nível 4.
        #
        # ⚠️ O `de_pe` é obrigatório, e ele faltava. "Sobreviveu" sozinho aprova SENTAR:
        # o `_nunca_caiu` depende de `terminated`, e a única terminação de queda é o
        # `fell_over`, que mede só INCLINAÇÃO do tronco (70°). Sentado com o tronco
        # vertical a inclinação é ~0°, então o robô nunca "cai" e o `parado` pontuava
        # 20 s no chão — visto no `play` em 31/07, com `sucesso = 0,9553`.
        #
        # O `de_pe` fecha isso pela ALTURA: pelve >= 0,65 m, e sentado ela fica ~0,30 m.
        #
        # Ressalva registrada, não consertada: o teste é INSTANTÂNEO no passo 1000,
        # porque a exigência de sustentação do `parado` é 0 s. Um robô que passa o
        # episódio sentado e levanta no último passo ainda aprova. Fechar isso exige
        # fração do episódio, que é mudança de definição de sucesso (Categoria C).
        # 🔧 03/08/2026 — o portão é VELOCIDADE, não posição. Pedido do user: "Ele não
        # precisa ficar no mesmo lugar, mas sim não se mover." A deriva continua só
        # logada (`deriva_parado`), e agora entra a fração do episódio em que ele
        # esteve de pé E devagar. O `_frac_quieto` é acumulado no `__call__`, não aqui:
        # este método roda DUAS vezes por passo (uma pela tarefa ativa, uma pela
        # sorteada, para o diagnóstico), e acumular aqui contaria em dobro.
        #
        # A fração também fecha o furo do teste instantâneo. A exigência de sustentação
        # do `parado` é 0 s, então o critério vale num passo só — o do `time_out`. Sem
        # a fração, andar 20 s e parar no último passo aprovaria, e o mesmo para passar
        # o episódio sentado e levantar no fim (o `de_pe` entra na mesma fração).
        #
        # 🔧 S7 — entra o TEMPO FORA DE PÉ. O `_frac_quieto` exige `devagar E de pé`
        # junto, então um robô agachado e imóvel some do numerador mas o critério não
        # diz POR QUE. O acumulador separa as duas falhas: quem se move perde fração,
        # quem agacha estoura o tempo fora. Sem ele, agachar e ficar agachado ainda
        # aprova se a fração for tolerante o bastante.
        cond = torch.where(tarefa == T.PARADO,
                           env.termination_manager.time_outs & self._nunca_caiu
                           & parado_de_pe
                           & (self._frac_quieto >= tol.parado_fracao)
                           & (self._fora_de_pe_s < tol.limite_fora_de_pe_s), cond)
        # `andar`: chegou no raio, apontado, e de pé. Não há limiar de "quieto" porque
        # o `d_morto` leva a velocidade comandada a ZERO no alvo — o robô para pelo
        # perfil, não por penalidade. Ver §4.
        #
        # O `andar c/ caixa` (abaixo) continua com o `chegou` de raio único. A S6 trata
        # só o `andar`; a diferença fica sendo o disparo, porque o raio de manutenção
        # da histerese é o mesmo 0.25 do raio único.
        cond = torch.where(tarefa == T.ANDAR,
                           chegou_andar & alinhado & parado_de_pe, cond)
        # ⚠️ `preensao` é OBRIGATÓRIA aqui, e faltava. O `pegar` era a única das sete
        # sem ela — `parado c/ caixa` e `andar c/ caixa` exigem, `botar` exige o
        # contrário. Sem ela o critério passa ENCOSTANDO o peito na caixa parada na
        # prateleira: na fronteira do `de_pe` (pelve 0,65, inclinação 20°) o alvo do
        # peito desce para z = 0,723, e a caixa na prateleira está em 0,65 — Δz = 0,073,
        # dentro dos 0,10 m. Medido em 31/07: `pegar` marcava 98,6% de sucesso com
        # `grasp = 0`, `lift = 0` e a caixa subindo 3,8 cm.
        cond = torch.where(tarefa == T.PEGAR,
                           no_peito & parado_de_pe & preensao, cond)
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
            self._quieto_passos[caiu] = 0.0
            self._chegou_ok[caiu] = False
            self._alinhado_ok[caiu] = False
            self._fora_de_pe_s[caiu] = 0.0
        self._len_ant.copy_(env.episode_length_buf)

        # 2. quem foi terminado por falha nunca mais "sobreviveu" neste episódio
        self._nunca_caiu &= ~env.termination_manager.terminated

        # 2b. `parado`: acumula os passos em que ele está DE PÉ e DEVAGAR, e converte
        # em fração do episódio corrido. Fica aqui, e não no `_condicao`, porque aquele
        # roda duas vezes por passo (tarefa ativa e tarefa sorteada) e contaria dobrado.
        robo: Entity = env.scene["robot"]
        devagar = (robo.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
                   < self.tol.parado_v_max)
        _em_pe = de_pe(env, self.tol.de_pe_z, self.tol.de_pe_tilt_rad)
        quieto = devagar & _em_pe
        self._quieto_passos += quieto.float()
        # S7: o tempo fora de `de pé` acumula aqui, no mesmo lugar e pelo mesmo motivo
        # (o `_condicao` roda duas vezes por passo e contaria dobrado).
        self._fora_de_pe_s += (~_em_pe).float() * self.dt
        self._frac_quieto = (self._quieto_passos
                             / env.episode_length_buf.clamp(min=1).float())

        # 2c. `andar` (S6): dispara a histerese de `chegou` e de `alinhado`.
        # Fica aqui pelo mesmo motivo do `_frac_quieto`: o `_condicao` roda duas vezes
        # por passo (tarefa ativa e tarefa sorteada). Aqui o efeito seria idempotente
        # (`|=` com a mesma geometria), mas manter toda escrita de estado num lugar só
        # evita que a próxima mudança reintroduza a contagem dobrada.
        _meta = env.command_manager.get_term("lift_target")
        _d_alvo = (robo.data.root_link_pos_w[:, :2]
                   - _meta.command[:, ALVO][:, :2]).norm(dim=-1)
        self._chegou_ok |= _d_alvo < self.tol.andar_raio_chega
        self._alinhado_ok |= _meta.erro_rumo_deg() < self.tol.alinhado_chega_deg

        # 3. sustentação: soma enquanto vale, ZERA quando quebra.
        #
        # ⚠️ **O pré-gatilho não pontua.** Nos até 2 s antes do comando chegar a tarefa
        # ATIVA é `parado` (ou `parado c/ caixa`), e o critério do `parado` é
        # `time_out & de pé` com sustentação **0 s** — fecha num passo só. Sem este
        # `& disparou`, uma tarefa registra sucesso por um critério que não é o dela.
        # Apontado pelo user em 31/07: "os 2 s parado não devem pontuar, a tarefa é pegar
        # a caixa". Também impede o contador de sustentação de acumular na espera.
        meta = env.command_manager.get_term("lift_target")
        cond = self._condicao(env) & meta.disparou
        self._contador = torch.where(cond, self._contador + self.dt,
                                     torch.zeros_like(self._contador))
        antes = self._conquistado.clone()
        self._conquistado |= cond & (self._contador >= self._exigencia_s(env))

        # 4. diagnóstico: a condição da tarefa que o CURRÍCULO vai creditar
        srt = getattr(env, "tarefa_sorteada", env.active_task)
        cond_srt = self._condicao(env, srt)
        sem_condicao = (self._conquistado & ~antes) & ~cond_srt
        env.diag_soma.index_add_(
            0, srt, torch.stack([cond_srt.float(), sem_condicao.float()], dim=-1))
        env.diag_cont.index_add_(0, srt, torch.ones_like(cond_srt, dtype=torch.float))

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
