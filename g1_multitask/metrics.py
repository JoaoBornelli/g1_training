"""Sucesso por tarefa — a RÉGUA do currículo, e ela é física.

Este módulo é pré-requisito do workflow de blocos de 2k-3k, não um extra. Mudar
**peso de reward** só é Categoria A ("grátis, retoma do checkpoint") porque o sucesso
mora aqui, em `env.success_buf`, como **fato físico** — mexer em peso não move a régua
e não invalida nenhuma EMA nem nenhum limiar do orquestrador.

Se o sucesso fosse soma de reward, como o `PlrHeights` da Lift faz, toda mudança de
peso seria **Categoria C disfarçada** — recomeçar do zero.

Uma computação, dois consumidores: o currículo lê `env.success_buf`, o
`MetricsManager` loga o retorno com `reduce="last"`.

--------------------------------------------------------------------------------
O CRITÉRIO NOVO (§8)
--------------------------------------------------------------------------------
O critério base vale para as cinco tarefas, e ele é ERRO MÉDIO DE VELOCIDADE:

    erro_lin = (1/T) ∫ ‖v_cmd_xy − v_xy‖ dt        # m/s
    erro_ang = (1/T) ∫ |ωz_cmd − ωz|   dt          # rad/s

    base = não_caiu & (erro_lin ≤ TOL_V) & (erro_ang ≤ TOL_W)

Ele é físico: mede metros por segundo. A recompensa calcula `exp(−erro²/σ²)`. As duas
funções são diferentes, então o `σ` da recompensa não move a régua.

Com comando zero — as três tarefas de manipulação — o critério mede a DERIVA e o
rebolado de pelve. É de graça, e é o que fecha o buraco que o `hold_still` deixou.

**Alternativa rejeitada antes da implementação:** o critério por deslocamento. Com
reamostragem a cada 3 a 8 s as direções se cancelam, e um robô parado passaria.

Saíram: `chegou`, `chegou_andar`, `alinhado` e a histerese dos dois; o `_frac_quieto`,
o `_fora_de_pe_s` e o `deriva_parado` do antigo `parado`; o `disparou` do gatilho.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1_training.base_env import BACK_SENSORS, PALM_SENSORS, SUPPORT_SENSOR
from g1_training.skills.lift.rewards import _contact, _grasp

from . import tasks as T
from .rewards import alvo_peito_w, alvo_z_pegar
from .terminations import de_pe

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.managers.metrics_manager import MetricsTermCfg

ALVO = slice(0, 3)
PASSOS_MIN_S = 1.0
"""Segundos mínimos de episódio antes de a média de erro valer alguma coisa.

Sem este piso, o critério base dispararia no passo 1, onde a média é um número só."""

FILTRO_ANG_TAU_S = 0.5
"""Constante de tempo do passa-baixa do erro angular FILTRADO (só-log, 11/08).

O erro cru tem um piso sistêmico de 0.62-0.73 rad/s que não é rastreio ruim: a
oscilação de guinada da própria marcha (~1.5-2 Hz) e o push do fabricante moram
dentro da integral — foi por isso que o `tol_w` subiu pra 0.70, virando guarda de
sanidade em vez de teste de rastreio. Esta versão filtra o erro ASSINADO com EMA de
0.5 s antes do |·|: a oscilação de marcha (período 0.5-0.7 s) atenua ~7×, o
seguir-comando (janelas de 3-8 s) passa inteiro. É o número que responde "ele segue
o ωz ou está se perdendo?" — se o currículo avançar com ele alto, o `tol_w` aperta
com dado na mão (os destraves são catraca; só as EMAs recalibram)."""


def condicao_tarefa(env: "ManagerBasedRlEnv", tol, alvo_peito_b,
                    tarefa: torch.Tensor | None = None) -> torch.Tensor:
    """[B] bool — a condição FÍSICA da tarefa vale AGORA, por env.

    **Função de módulo desde 17/08, e isso é estrutural.** Ela tem DOIS consumidores: o
    critério de sucesso (`Sucesso`, que alimenta o currículo) e a recompensa densa
    (`sucesso_denso`). Se cada um tivesse a sua cópia, reward e critério podiam divergir
    — que é a classe de defeito C1/C2 do bloco 1 (recompensa máxima num estado que o
    critério reprova).

    Não pode ser método: o `RewardManager` roda ANTES do `MetricsManager` no `step()`
    (`manager_based_rl_env.py:439-440`), então ler um buffer que o `Sucesso` escreve
    daria a condição do passo ANTERIOR — e, no primeiro passo após o reset, a do
    episódio anterior."""
    robo: Entity = env.scene["robot"]
    caixa: Entity = env.scene["box"]
    meta = env.command_manager.get_term("lift_target")
    tarefa = env.active_task if tarefa is None else tarefa

    # --- primitivas compartilhadas, computadas uma vez ---
    parado_de_pe = de_pe(env, tol.de_pe_z, tol.de_pe_tilt_rad)
    preensao = _grasp(env, PALM_SENSORS, BACK_SENSORS) > 0.5
    apoiada = _contact(env, SUPPORT_SENSOR) > 0.5
    caixa_quieta = caixa.data.root_link_lin_vel_w.norm(dim=-1) < tol.caixa_quieta_v

    # ⚠️ **PISO em z, não esfera 3D (17/08).** O `pegar` mede altura de MUNDO contra o
    # alvo graduado pelo eixo `alvo`. A esfera de 0,10 m reprovava o robô por fazer
    # MAIS: com o alvo em +5 cm e a caixa a +26 cm, a distância dá 21 cm. E ele não
    # observa o nível, então não teria como parar na altura certa. O piso é monotônico.
    erguida = (caixa.data.root_link_pos_w[:, 2]
               >= alvo_z_pegar(env) - tol.alvo_tol_z)
    # o alvo do peito ficou SÓ no `locomover_carregando`, e voltou ao frame da base
    peito_w = alvo_peito_w(env, alvo_peito_b)
    no_peito = (caixa.data.root_link_pos_w - peito_w).norm(dim=-1) < tol.caixa_no_alvo
    no_alvo = (caixa.data.root_link_pos_w
               - meta.command[:, ALVO]).norm(dim=-1) < tol.caixa_no_alvo
    orientada = ((meta.erro_angulo_deg() < tol.reorienta_angulo_deg)
                 & (meta.desvio_xy() < tol.reorienta_xy) & apoiada)

    # --- condição ADICIONAL por tarefa (§8) ---
    # O `locomover` não tem nenhuma: o critério base já mede a tarefa dele.
    #
    # ⚠️ O `de_pé` SAIU do critério de locomoção. O `pose` e o `upright` já
    # produzem a postura, de forma contínua, e o `fell_over` já termina a queda.
    cond = torch.ones_like(parado_de_pe)
    # ⚠️ `preensao` e `de_pé` ficam no `pegar` mesmo com o critério de altura simples.
    # Sem `preensao` o robô aninha a caixa no vão dos antebraços contra o tronco e ergue
    # sem pegar — o hack que o `exige_grasp` do `gated` documenta como MEDIDO. O
    # `_grasp` exige contato de palma e proíbe contato de dorso, então ele fecha esse
    # caminho. Sem `de_pé`, o estado final do `pegar` deixa de ser o estado inicial
    # canônico do `locomover_carregando` e do `botar`.
    cond = torch.where(tarefa == T.PEGAR,
                       erguida & parado_de_pe & preensao, cond)
    cond = torch.where(tarefa == T.BOTAR,
                       no_alvo & ~preensao & caixa_quieta & parado_de_pe, cond)
    cond = torch.where(tarefa == T.REORIENTAR, orientada, cond)
    cond = torch.where(tarefa == T.LOCOMOVER_CARREGANDO,
                       no_peito & preensao, cond)
    return cond


def sucesso_denso(env: "ManagerBasedRlEnv", tol, alvo_peito_b,
                  tasks: tuple[int, ...]) -> torch.Tensor:
    """[B] — 1,0 enquanto a condição da tarefa vale, nas tarefas listadas. (17/08)

    **Recompensa, e vive aqui de propósito.** O alvo não pagava nada: o `cond_fisica`
    era só diagnóstico, e o bloco 3 rodou 22 mil iterações com ele em 0,0000 sem um
    único termo de recompensa olhando para ele. Pôr o bônus no mesmo módulo do critério
    é o que garante que os dois medem a mesma coisa — ver `condicao_tarefa`.

    ⚠️ Ele é gateado por tarefa AQUI, e não pelo `gated`, porque a máscara do `gated`
    lê o one-hot do comando e este termo já precisa do `active_task` para a condição. Um
    gate só, uma fonte só.

    ⚠️ O `locomover` fica fora: a condição dele é `ones_like` e ele coletaria o bônus
    inteiro de graça. O sinal denso dele são os dois `track_*`.

    ⚠️ Fora de `TERMOS_DE_TAREFA`: dentro, o orçamento do `pegar` iria a 9,5 e o bônus
    se diluiria para 2,1. Ver `tasks.TERMOS_DE_TAREFA`."""
    quais = torch.as_tensor(list(tasks), dtype=torch.long, device=env.device)
    dentro = (env.active_task.unsqueeze(-1) == quais).any(dim=-1)
    return condicao_tarefa(env, tol, alvo_peito_b).float() * dentro.float()


class Sucesso:
    """Critério de sucesso por tarefa, com sustentação temporal.

    Stateful: o mjlab auto-instancia com `(cfg, env)`. Guarda o acumulador de erro de
    velocidade, um contador de passos em que a condição vale seguido, e um flag
    `conquistado` que **persiste até o fim do episódio** — o sucesso é do EPISÓDIO.

    ⚠️ **Detecta o reset sozinho.** O `MetricsManager.reset` limpa só os buffers dele;
    ele NÃO chama `reset()` nos termos-classe. Então aqui a gente vê o reset pela
    QUEDA do `episode_length_buf`, o que também é robusto ao
    `init_at_random_ep_len=True` que o `train.py` do mjlab passa."""

    def __init__(self, cfg: "MetricsTermCfg", env: "ManagerBasedRlEnv"):
        dev = env.device
        p = cfg.params
        self.tol = p["tol"]
        self.alvo_peito_b = torch.tensor(p["alvo_peito_b"], device=dev)
        self.dt = env.step_dt
        self._passos_min = PASSOS_MIN_S / self.dt

        n = env.num_envs
        self._contador = torch.zeros(n, device=dev)
        self._conquistado = torch.zeros(n, dtype=torch.bool, device=dev)
        self._len_ant = torch.zeros(n, dtype=torch.long, device=dev)
        self._nunca_caiu = torch.ones(n, dtype=torch.bool, device=dev)
        # --- acumuladores do critério base (§8) ---
        self._erro_lin = torch.zeros(n, device=dev)
        self._erro_ang = torch.zeros(n, device=dev)
        self._passos = torch.zeros(n, device=dev)
        # estado do passa-baixa do erro ASSINADO (só-log; ver FILTRO_ANG_TAU_S)
        self._erro_ang_filt = torch.zeros(n, device=dev)
        self._alpha_filt = min(1.0, self.dt / FILTRO_ANG_TAU_S)

        self._palmas = SceneEntityCfg("robot", site_names=[])
        # --- diagnóstico por tarefa, emitido pelo `observability.Relatorio` ---
        # coluna 0 = `cond_fisica`: a condição da tarefa sorteada vale agora?
        # coluna 1 = `atribuicao_divergente`: houve sucesso com a condição dela FALSA?
        # Existe porque o log de 31/07 mostrou `perf[pegar] = 0,98` com `grasp = 0`.
        # colunas 2 e 3 = erro de velocidade linear/angular POR TAREFA (10/08). O
        # `Episode_Metrics/erro_vel_*` global dilui: com 3 tarefas abertas, 2 têm
        # comando zero, e o 0,89 do bloco 2 não dizia QUEM reprovava na régua. É
        # contra estas colunas que `tol_v`/`tol_w` se calibram.
        # coluna 4 = |erro angular FILTRADO| (11/08) — rastreio sem a oscilação de
        # marcha; é quem vigia se o tol_w 0.70 deixou passar um não-rastreador.
        env.diag_soma = torch.zeros(T.NUM_TASKS, 5, device=dev)
        env.diag_cont = torch.zeros(T.NUM_TASKS, device=dev)
        # buffer criado ANTES do 1º reset: o currículo lê daqui
        env.success_buf = torch.zeros(n, device=dev)

    # ------------------------------------------------------------- primitivas
    def _base(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        """[B] bool — o critério base: não caiu e rastreou a velocidade comandada."""
        passos = self._passos.clamp(min=1.0)
        return (self._nunca_caiu
                & (self._passos >= self._passos_min)
                & (self._erro_lin / passos <= self.tol.tol_v)
                & (self._erro_ang / passos <= self.tol.tol_w))

    def _condicao(self, env: "ManagerBasedRlEnv",
                  tarefa: torch.Tensor | None = None) -> torch.Tensor:
        """[B] bool — a condição de uma tarefa vale AGORA, por env.

        `tarefa=None` usa `env.active_task`, que é o que PONTUA. Passando
        `env.tarefa_sorteada` sai a condição que o CURRÍCULO credita."""
        return condicao_tarefa(env, self.tol, self.alvo_peito_b, tarefa)

    def _exigencia_s(self, env) -> torch.Tensor:
        """[B] — quantos segundos a condição tem que valer seguido, por tarefa."""
        tol, tarefa = self.tol, env.active_task
        s = torch.full_like(env.success_buf, tol.sustenta_pegar_s)
        # O `locomover` fecha no `time_out`: o critério dele é a média do episódio.
        s = torch.where(tarefa == T.LOCOMOVER, torch.zeros_like(s), s)
        s = torch.where(tarefa == T.LOCOMOVER_CARREGANDO,
                        torch.full_like(s, tol.sustenta_carregar_s), s)
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
            self._erro_lin[caiu] = 0.0
            self._erro_ang[caiu] = 0.0
            self._erro_ang_filt[caiu] = 0.0
            self._passos[caiu] = 0.0
        self._len_ant.copy_(env.episode_length_buf)

        # 2. quem foi terminado por falha nunca mais "sobreviveu" neste episódio
        self._nunca_caiu &= ~env.termination_manager.terminated

        # 3. acumula o erro de rastreio. É a integral do §8, e ela roda AQUI, uma vez
        # por passo — o `_condicao` roda duas vezes (tarefa ativa e sorteada) e
        # contaria dobrado.
        robo: Entity = env.scene["robot"]
        cmd = env.command_manager.get_command("twist")
        e_ang_sinal = cmd[:, 2] - robo.data.root_link_ang_vel_b[:, 2]
        e_lin = (cmd[:, :2] - robo.data.root_link_lin_vel_b[:, :2]).norm(dim=-1)
        e_ang = e_ang_sinal.abs()
        self._erro_lin += e_lin
        self._erro_ang += e_ang
        # o filtro roda no erro ASSINADO — filtrar depois do |·| não cancela a
        # oscilação (o retificador vira tudo pro mesmo lado)
        self._erro_ang_filt += self._alpha_filt * (e_ang_sinal - self._erro_ang_filt)
        self._passos += 1.0

        # 4. sustentação: soma enquanto vale, ZERA quando quebra.
        cond = self._base(env) & self._condicao(env)
        # O `locomover` tem exigência 0 s, então ele fecharia em qualquer passo em que
        # a média já estivesse boa. O `time_out` amarra o critério ao FIM do episódio,
        # que é onde a média `(1/T) ∫` do §8 é definida.
        so_locomover = env.active_task == T.LOCOMOVER
        cond = torch.where(so_locomover,
                           cond & env.termination_manager.time_outs, cond)

        self._contador = torch.where(cond, self._contador + self.dt,
                                     torch.zeros_like(self._contador))
        antes = self._conquistado.clone()
        self._conquistado |= cond & (self._contador >= self._exigencia_s(env))

        # 5. diagnóstico: a condição da tarefa que o CURRÍCULO vai creditar
        srt = getattr(env, "tarefa_sorteada", env.active_task)
        cond_srt = self._condicao(env, srt)
        sem_condicao = (self._conquistado & ~antes) & ~cond_srt
        env.diag_soma.index_add_(
            0, srt, torch.stack([cond_srt.float(), sem_condicao.float(),
                                 e_lin, e_ang,
                                 self._erro_ang_filt.abs()], dim=-1))
        env.diag_cont.index_add_(0, srt, torch.ones_like(cond_srt, dtype=torch.float))

        env.success_buf.copy_(self._conquistado.float())
        return env.success_buf


def erro_vel_linear(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Erro instantâneo de rastreio linear, em m/s. [B] — SÓ LOG.

    O `UniformVelocityCommand` já acumula `error_vel_xy` como métrica de comando, mas
    aquela normaliza pela janela de reamostragem, não pelo episódio. Esta aqui é a
    grandeza crua, e o `reduce="mean"` do `MetricsManager` faz a média do episódio —
    que é exatamente a régua do §8."""
    robo: Entity = env.scene["robot"]
    cmd = env.command_manager.get_command("twist")
    return (cmd[:, :2] - robo.data.root_link_lin_vel_b[:, :2]).norm(dim=-1)


def erro_vel_angular(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Erro instantâneo de rastreio angular, em rad/s. [B] — SÓ LOG.

    Com comando zero — a manipulação — ele mede rebolado de pelve, que é o buraco que
    o `hold_still` deixou ao sair (§10)."""
    robo: Entity = env.scene["robot"]
    cmd = env.command_manager.get_command("twist")
    return (cmd[:, 2] - robo.data.root_link_ang_vel_b[:, 2]).abs()
