"""Os dois termos de comando: a META (17 números) e o TWIST do fabricante (3).

    lift_target                    17 números  — a meta que a política persegue
      [0:3]   alvo_pos    destino em MUNDO (a obs converte pra base)
      [3:6]   face_alvo   eixo do CORPO da caixa que é a face alvo
      [6:9]   dir_alvo    direção alvo, já em frame da BASE
      [9:17]  one-hot     8 slots, 5 em uso

    twist                           3 números  — `[vx, vy, ωz]`, SORTEADO
      É o `UniformVelocityCommand` do mjlab, com uma subclasse fina.

Duas escolhas de layout que economizam código:

- **`alvo_pos` fica em `[0:3]`** de propósito, porque assim
  `g1_training/common/observations.py::target_pos_b` é reusado sem uma linha de
  mudança.
- **o twist chama-se `"twist"`**, o mesmo nome do cfg de velocity do fabricante, então
  os termos de marcha dele encontram o comando sem uma linha de fiação.

--------------------------------------------------------------------------------
O QUE A REFORMA DE 07/08 TIROU DAQUI  (ver `EXPERIMENTO.md` §10b)
--------------------------------------------------------------------------------
O `DesiredTwistCommand` inteiro, com a quíntica, a banda de frenagem, o limitador de
taxa e os dois `d_morto`. Ele derivava `[vx, vy, ωz]` de um DESTINO, e a linha
`v_alvo *= cos(erro_rumo)` fechava um ciclo: sem girar, o comando ia a zero; sem
comando, a política não aprendia a andar. Medido: comando efetivo de 0,052 m/s contra
`v_max` 0,5.

Saíram junto o `erro_rumo_deg` e todo o maquinário de destino (`_destino_w`, `_dist`,
`_head`), porque o `andar` deixou de ser "ir a um lugar" e virou "rastrear uma
velocidade".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from . import tasks as T

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

ALVO = slice(0, 3)
FACE = slice(3, 6)
DIR = slice(6, 9)
ONEHOT = slice(9, 17)
COMMAND_DIM = 17

FACE_AXES = (
    (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),      # 0,1 — laterais (giro em torno de z)
    (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),      # 2,3 — laterais
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),      # 4,5 — topo e fundo (exigem erguer)
)
LATERAIS = (0, 1, 2, 3)
TOPO_FUNDO = (4, 5)


def _nivel(env, eixo: str, task_idx: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    """Índice de nível nesse eixo, shape igual a `task_idx`.

    Fallback: o índice INICIAL da tarefa. O termo de comando é construído ANTES do
    termo de currículo, então `env.nivel` pode não existir na primeira leitura."""
    tabela = getattr(env, "nivel", None)
    if tabela is not None and eixo in tabela:
        return tabela[eixo][env_ids]
    inicio = torch.zeros_like(task_idx)
    for t, eixos in T.AXES.items():
        if eixo in eixos:
            inicio = torch.where(task_idx == t, eixos[eixo], inicio)
    return inicio


def _rot_z(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Roda `v` [B,3] em torno de z por `ang` [B]. Sem quaternion — é rotação plana."""
    c, s = torch.cos(ang), torch.sin(ang)
    return torch.stack((v[:, 0] * c - v[:, 1] * s,
                        v[:, 0] * s + v[:, 1] * c,
                        v[:, 2]), dim=-1)


class LiftTargetCommand(CommandTerm):
    """A meta: qual tarefa, onde é o destino, e qual face vai pra qual direção."""

    cfg: "LiftTargetCommandCfg"

    def __init__(self, cfg: "LiftTargetCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene["robot"]
        self.box: Entity = env.scene[cfg.box_name]
        self.table: Entity = env.scene[cfg.table_name]

        self._command = torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)
        self._face_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._spawn_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._dir_w = torch.zeros(self.num_envs, 3, device=self.device)
        """Direção alvo da face, em MUNDO. 🔧 Conserto de 31/07: era guardada no frame
        da BASE e comparada contra a normal recomputada na base a cada passo — então
        **girar o ROBÔ mudava o erro com a caixa imóvel**. Medido, 64 envs, ação zero: a
        caixa não sai do lugar (`desvio_xy = 0,0009 m`) e a fração dentro da tolerância
        de 10° sobe de 0/64 no spawn para 19/64 (30%) no passo 200 — o `reorientar` era
        aprovado pelo robô virar. Em mundo, só a rotação da CAIXA move o erro."""

        # INTENÇÃO sorteada, resolvida contra a pose só depois (ver `_resolver`).
        self._ang = torch.zeros(self.num_envs, device=self.device)
        self._topo = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pendente = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ativa = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        """A tarefa ATIVA agora. Escrita SEMPRE in-place, porque `env.active_task`
        guarda a referência a este tensor."""

        # Disciplina do §15: os rewards leem `active_task` antes do 1º reset, então
        # o buffer TEM que existir aqui.
        env.active_task = self._ativa

        self.metrics["erro_posicao"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["erro_angulo_deg"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    # ------------------------------------------------------------------ resample
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Sorteia a INTENÇÃO. Nada que dependa de pose é resolvido aqui.

        ⚠️ Por que a separação existe (bug medido 30/07): no reset o command manager
        roda DEPOIS dos eventos que reposicionam robô e caixa
        (`manager_based_rl_env.py:581` contra `:560`), mas as grandezas DERIVADAS —
        `root_link_pos_w`, `root_link_quat_w` — ainda não foram recalculadas. Ler pose
        aqui devolve a pose ANTERIOR ao reset.

        O sintoma era silencioso e destruía o eixo de giro: com `dir_alvo` derivado da
        orientação velha da caixa, o erro de ângulo no nível 15° saía espalhado de
        0,35° a 40,65° em vez de 15° exatos."""
        n = len(env_ids)
        if n == 0:
            return

        # A tarefa foi sorteada pelo CURRÍCULO (`:554`), não aqui (`:581`): o evento
        # `reset_segurando` (`:560`) precisa dela pra decidir quem nasce com a caixa
        # nas mãos, e isso acontece antes deste método rodar.
        tarefa = self._env.tarefa_sorteada[env_ids]
        self._ativa[env_ids] = tarefa

        # ORIENTAÇÃO (só o `reorientar` usa). Nível 0-3 = giro em torno de z sobre uma
        # face LATERAL; nível 4 = topo/fundo, salto qualitativo que exige a mão.
        giro = _nivel(self._env, "giro", tarefa, env_ids)
        ang = torch.tensor([math.radians(a) for a in T.LEVELS["giro"]],
                           device=self.device)[giro]
        sinal = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        self._ang[env_ids] = ang * sinal
        topo = giro >= len(T.LEVELS["giro"]) - 1
        self._topo[env_ids] = topo
        lat = torch.randint(0, len(LATERAIS), (n,), device=self.device)
        tf = torch.randint(0, len(TOPO_FUNDO), (n,), device=self.device) + TOPO_FUNDO[0]
        self._face_idx[env_ids] = torch.where(topo, tf, lat)

        self._pendente[env_ids] = True

    def _resolver(self) -> None:
        """Resolve contra a pose FRESCA o que o resample não podia resolver.

        Chamado de todo ponto de leitura, e é no-op quando não há nada pendente."""
        if not bool(self._pendente.any()):
            return
        ids = self._pendente.nonzero().flatten()
        tarefa = self._ativa[ids]

        # alvo x,y do `reorientar` = onde a caixa NASCEU (pose real pós-reset).
        self._spawn_xy[ids] = self.box.data.root_link_pos_w[ids, :2]

        face_b = torch.tensor(FACE_AXES, device=self.device)[self._face_idx[ids]]
        # ⚠️ TUDO em MUNDO daqui pra baixo. A normal da face vive no frame da CAIXA
        # (`face_b` é constante), então levá-la a mundo é uma rotação só.
        normal_w = quat_apply(self.box.data.root_link_quat_w[ids], face_b)
        alvo_lateral = _rot_z(normal_w, self._ang[ids])
        # topo/fundo: a face tem que apontar PRA MIM. Resolvido no SPAWN e congelado —
        # é alvo relativo ao robô por natureza, e congelar mantém o critério medindo
        # rotação da caixa, igual ao `_spawn_xy`.
        para_mim = (self.robot.data.root_link_pos_w[ids, :2]
                    - self.box.data.root_link_pos_w[ids, :2])
        alvo_tf = torch.zeros_like(alvo_lateral)
        alvo_tf[:, :2] = para_mim / para_mim.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dir_alvo = torch.where(self._topo[ids].unsqueeze(-1), alvo_tf, alvo_lateral)
        dir_alvo = dir_alvo / dir_alvo.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # zera face/dir de quem não é `reorientar`
        so_reorienta = (tarefa == T.REORIENTAR).unsqueeze(-1)
        self._command[ids, FACE] = torch.where(
            so_reorienta, face_b, torch.zeros_like(face_b))
        self._dir_w[ids] = torch.where(
            so_reorienta, dir_alvo, torch.zeros_like(dir_alvo))
        # ⚠️ `DIR` da OBS não é escrito aqui: o `_update_command` o reescreve a CADA
        # passo, convertendo `_dir_w` para o frame da base atual.
        self._pendente[ids] = False

    # -------------------------------------------------------------------- update
    def _update_command(self) -> None:
        """Reavalia `alvo_pos` a cada passo pela regra da tarefa (F5).

        🔧 **F5 — `alvo_pos` não é buffer, é regra avaliada a cada passo.** Gravado no
        reset, quando o robô se move aquele vetor passa a ser o vetor de VOLTA ao
        spawn — o vetor de deriva. Informar deriva implica uma tarefa ("volte") pela
        qual a política não é recompensada."""
        self._resolver()
        tarefa = self._ativa

        one = torch.zeros(self.num_envs, T.ONEHOT_DIM, device=self.device)
        one.scatter_(1, tarefa.unsqueeze(-1), 1.0)
        self._command[:, ONEHOT] = one

        robo = self.robot.data.root_link_pos_w
        caixa = self.box.data.root_link_pos_w
        # ponto de pouso na prateleira: centro dela + meia altura da caixa
        prateleira = self.table.data.root_link_pos_w.clone()
        prateleira[:, 2] += self.cfg.shelf_half_z + self.cfg.box_half_z

        # ⚠️ As duas tarefas de locomoção NÃO têm alvo de posição. Elas rastreiam
        # velocidade. O default `alvo = robo` dá `target_pos_b = 0` na obs, e o
        # `target_pos_b_gateado` do ator zera o canal de qualquer forma.
        alvo = robo.clone()
        for t in (T.PEGAR, T.REORIENTAR):
            alvo = torch.where((tarefa == t).unsqueeze(-1), caixa, alvo)
        alvo = torch.where((tarefa == T.BOTAR).unsqueeze(-1), prateleira, alvo)
        self._command[:, ALVO] = alvo

        # `dir_alvo` da OBS: o alvo vive em MUNDO (`_dir_w`) e a obs é egocêntrica,
        # então a conversão é por passo. Sem isso a política veria o vetor do spawn,
        # que deixa de apontar pro lugar certo assim que o robô gira.
        self._command[:, DIR] = quat_apply_inverse(
            self.robot.data.root_link_quat_w, self._dir_w)

    def _update_metrics(self) -> None:
        self._resolver()
        passos = self.cfg.resampling_time_range[1] / self._env.step_dt
        err = torch.norm(self._command[:, ALVO] - self.box.data.root_link_pos_w, dim=-1)
        self.metrics["erro_posicao"] += err / passos
        self.metrics["erro_angulo_deg"] += self.erro_angulo_deg() / passos

    # ------------------------------------------------------------------ leitores
    def erro_angulo_deg(self) -> torch.Tensor:
        """Ângulo entre a normal da face alvo e `dir_alvo`, **em MUNDO**, em graus.

        UM escalar, e a simetria do cubo se resolve sozinha: girar em torno da normal
        da face não muda o vetor, então o erro não se move — e como essa rotação é
        irrelevante pro objetivo, a métrica CONCORDA com o objetivo.

        🔧 **Conserto de 31/07: era comparado no frame da BASE, e girar o ROBÔ mudava o
        erro com a caixa parada.** Em mundo, só a rotação da CAIXA move o erro."""
        self._resolver()
        face_b = self._command[:, FACE]
        normal_w = quat_apply(self.box.data.root_link_quat_w, face_b)
        cos = (normal_w * self._dir_w).sum(dim=-1).clamp(-1.0, 1.0)
        ang = torch.rad2deg(torch.acos(cos))
        # sem comando de orientação (face zerada) o erro não existe -> 0
        return torch.where(face_b.abs().sum(dim=-1) > 0.0, ang, torch.zeros_like(ang))

    def spawn_xy_b(self) -> torch.Tensor:
        """O alvo x,y do `reorientar` em frame da base, [B,2]. Convertido a cada passo."""
        self._resolver()
        robo = self.robot.data.root_link_pos_w
        delta = torch.cat((self._spawn_xy - robo[:, :2],
                           torch.zeros(self.num_envs, 1, device=self.device)), dim=-1)
        return quat_apply_inverse(self.robot.data.root_link_quat_w, delta)[:, :2]

    def desvio_xy(self) -> torch.Tensor:
        """Quanto a caixa saiu do x,y onde nasceu, em metros. [B]

        Distância é invariante de frame, então sai direto do mundo. O alvo é ponto
        ESTÁTICO do mundo (onde a caixa nasceu), e por isso não recai no F5."""
        self._resolver()
        return (self.box.data.root_link_pos_w[:, :2] - self._spawn_xy).norm(dim=-1)

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Esfera no destino + seta da direção alvo.

        A face alvo fica VISÍVEL no `play` sem mudar geometria nem recompilar."""
        idx = visualizer.get_env_indices(self.num_envs)
        if not idx:
            return
        for b in idx:
            visualizer.add_sphere(
                center=self._command[b, ALVO].cpu().numpy(), radius=0.03,
                color=self.cfg.viz.cor_alvo, label=f"alvo_{b}")
            if float(self._command[b, FACE].abs().sum()) > 0.0:
                base = self.box.data.root_link_pos_w[b]
                dir_w = quat_apply(self.robot.data.root_link_quat_w[b: b + 1],
                                   self._command[b: b + 1, DIR])[0]
                visualizer.add_sphere(
                    center=(base + 0.25 * dir_w).cpu().numpy(), radius=0.02,
                    color=self.cfg.viz.cor_face, label=f"dir_alvo_{b}")


@dataclass(kw_only=True)
class LiftTargetCommandCfg(CommandTermCfg):
    box_name: str = "box"
    table_name: str = "table"
    box_half_z: float = 0.10
    shelf_half_z: float = 0.02

    @dataclass
    class VizCfg:
        cor_alvo: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.3)
        cor_face: tuple[float, float, float, float] = (0.2, 0.8, 1.0, 0.6)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: "ManagerBasedRlEnv") -> LiftTargetCommand:
        return LiftTargetCommand(self, env)


# ============================================================================
#  O TWIST — subclasse fina do comando do fabricante
# ============================================================================
class TwistMultitarefa(UniformVelocityCommand):
    """`UniformVelocityCommand` do mjlab com três acréscimos, e nada mais.

    Tudo que o fabricante entrega continua valendo: sorteio uniforme, reamostragem a
    cada 3 a 8 s, `rel_standing_envs`, `rel_forward_envs`, `rel_heading_envs`, a lei
    de realimentação do heading, as métricas `error_vel_xy` e `error_vel_yaw`, e o
    joystick do viewer.

    **1. Comando ZERO nas tarefas de manipulação.** O `pegar`, o `botar` e o
    `reorientar` não andam. A zeragem acontece no `_update_command`, depois do
    heading e depois da zeragem de standing, então ela vence as duas.

    **2. Teto de velocidade POR ENV, só no `vx`.** O eixo `velocidade` do currículo é
    por tarefa, e o `locomover` e o `locomover_carregando` têm células
    independentes. O fabricante sorteia com uma faixa ESCALAR para todos os envs
    (`velocity_command.py:77`), então o teto por env entra como reescala depois do
    `super()`. Reescalar preserva o formato do `rel_forward_envs`: o piso de 0,3
    vira `0,3 × escala`.

    ⚠️ **SÓ a coluna do `vx` (10/08).** A progressão do próprio fabricante pro G1
    alarga `lin_vel_x` até (−1.5, 2.0) e o yaw até 0.7 SEM tocar o `lin_vel_y`
    (`config/g1/env_cfgs.py:217`) — lateral acima de ±1.0 não é envelope testado.
    Até o bloco 2 a reescala pegava as DUAS colunas: no nível 1 o lateral ia a
    ±1.5 m/s, e o rastreio de guinada do `locomover` degradou junto
    (`contrib/locomover/track_angular_velocity` 0.44 → 0.21, medido 10/08).

    ⚠️ O `ang_vel_z` **não** é reescalado. O eixo é de velocidade linear. E a lei do
    heading satura na faixa ESCALAR `cfg.ranges.ang_vel_z`, então escalar o sorteio
    por env deixaria os dois modos inconsistentes.

    **3. GIRO PARADO.** Uma fração dos envs recebe `vx = 0`, `vy = 0` e `|ωz|` acima
    de um piso. Ela sai de dentro do `rel_standing_envs`, e não de uma fração nova: o
    regime parado é o mais simples dos três, e reduzi-lo custa menos que reduzir dado
    de marcha.

    Três fatos medidos que justificam o desenho:

      - o sorteio sozinho quase não produz giro parado. `P(|vx| < 0,05 e |vy| < 0,05)`
        = 0,05 × 0,05 = **0,25%**;
      - o **piso é obrigatório**. O gate dos quatro termos de marcha é
        `‖cmd_xy‖ + |ωz| > 0,05`, e 10% dos sorteios de `ωz ~ U(−0,5, 0,5)` ficam
        abaixo disso. Sem piso, esses envs ficam sem sinal de tarefa nenhum;
      - o piso 0,15 é **derivado**: o `rel_forward_envs` do fabricante trava
        `lin_vel_x ≥ 0,3` num teto de 1,0, ou seja 30% do teto. Trinta por cento do
        teto de `ωz` (0,5) dá 0,15.

    ⚠️ **Os envs de giro saem da máscara de standing.** O `_update_command` do
    fabricante zera `is_standing_env` a cada passo (`velocity_command.py:136`). Sem
    tirá-los de lá, o comando de giro seria apagado todo passo.

    ⚠️ **Eles NÃO saem da máscara de heading, e isso é de propósito.** Um env que caia
    nos dois recebe "gire parado até apontar para X, depois pare" — que é o caso de
    uso da navegação que motivou o giro parado."""

    cfg: "TwistMultitarefaCfg"

    def __init__(self, cfg: "TwistMultitarefaCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.is_pivot_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._cmd_zero = torch.tensor(
            list(cfg.tarefas_cmd_zero), dtype=torch.long, device=self.device)
        self._teto_nominal = float(cfg.ranges.lin_vel_x[1])
        assert self._teto_nominal > 0.0, "lin_vel_x[1] tem de ser positivo"
        self.metrics["frac_giro_parado"] = torch.zeros(self.num_envs, device=self.device)

    # -------------------------------------------------------------- resample
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)

        # --- teto por env: reescala SÓ o vx sorteado (ver docstring da classe) ---
        # O `vel_command_w` acompanha porque no resample ele é CÓPIA do sorteio
        # (`velocity_command.py:90`) — escalar só um dos dois quebraria os
        # world-envs se um dia forem ligados.
        teto = getattr(self._env, "teto_velocidade", None)
        if teto is not None:
            escala = teto[env_ids] / self._teto_nominal
            self.vel_command_b[env_ids, 0] *= escala
            self.vel_command_w[env_ids, 0] *= escala

        # --- giro parado: sai de DENTRO da fração de standing ---
        parado = self.is_standing_env[env_ids]
        vira_giro = parado & (r.uniform_(0.0, 1.0) < self.cfg.frac_giro_no_standing)
        self.is_standing_env[env_ids] = parado & ~vira_giro
        self.is_pivot_env[env_ids] = vira_giro

        ids = env_ids[vira_giro]
        if len(ids) > 0:
            self.vel_command_b[ids, :2] = 0.0
            self.vel_command_w[ids, :2] = 0.0
            wz = self.vel_command_b[ids, 2]
            # `sign` explícito: `torch.sign(0) = 0` mataria o comando de um env que
            # tirou exatamente zero.
            sinal = torch.where(wz >= 0.0, 1.0, -1.0)
            self.vel_command_b[ids, 2] = sinal * wz.abs().clamp(
                min=self.cfg.piso_giro_rad_s)
            self.vel_command_w[ids, 2] = self.vel_command_b[ids, 2]

    # ---------------------------------------------------------------- update
    def _update_command(self) -> None:
        super()._update_command()
        # A manipulação não anda. Roda DEPOIS do `super()`, portanto vence o heading
        # e vence a zeragem de standing.
        tarefa = getattr(self._env, "active_task", None)
        if tarefa is None or len(self._cmd_zero) == 0:
            return
        zera = (tarefa.unsqueeze(-1) == self._cmd_zero).any(dim=-1)
        self.vel_command_b[zera] = 0.0
        self.vel_command_w[zera] = 0.0

    def _update_metrics(self) -> None:
        super()._update_metrics()
        passos = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["frac_giro_parado"] += self.is_pivot_env.float() / passos


@dataclass(kw_only=True)
class TwistMultitarefaCfg(UniformVelocityCommandCfg):
    tarefas_cmd_zero: tuple[int, ...] = T.CMD_ZERO
    """Tarefas em que o comando é forçado a `[0, 0, 0]`."""

    frac_giro_no_standing: float = 0.5
    """Que fração do `rel_standing_envs` vira giro parado.

    Com `rel_standing_envs = 0,1` e 0,5 aqui, o resultado é 0,05 parado e 0,05 giro."""

    piso_giro_rad_s: float = 0.15
    """Piso de `|ωz|` no giro parado, em rad/s.

    DERIVADO, não escolhido: o `rel_forward_envs` do fabricante trava `lin_vel_x` em
    30% do teto (0,3 de 1,0). Trinta por cento do teto de `ωz` (0,5) dá 0,15.

    Ele tem de ficar acima de 0,05, que é o `command_threshold` dos quatro termos de
    marcha. Abaixo disso o env fica sem sinal de tarefa nenhum."""

    def build(self, env: "ManagerBasedRlEnv") -> TwistMultitarefa:
        return TwistMultitarefa(self, env)
