"""Os dois comandos do g1_poc.

`CaixaAlvoCommand` — 10 números (§6.2):
    [0:3] a posição do alvo, em MUNDO
    [3:6] `face_alvo`  — qual das 6 faces, no frame da caixa
    [6:9] `dir_alvo`   — para onde a normal dessa face aponta, no frame da base
    [9:10] `caixa_valida` — 0 ou 1

`TwistPoc` — o `UniformVelocityCommand` do mjlab, com dois acréscimos:
    (a) o twist é forçado a zero nos elos de manipulação;
    (b) metade dos envs PARADOS recebe giro no lugar.

ORDEM IMPORTA no dict de comandos: `caixa_alvo` vem PRIMEIRO, porque ele resolve
`env.poc_twist_zero`, e o `twist` lê esse buffer no mesmo passo.

ESTADO DESTE ARQUIVO — ESQUELETO (passo 2 da §17):
    um elo só, `pegar`; sem cadeia; sem nível; a prateleira não se move.
    A forma do episódio (30% locomoção / 70% manipulação) JÁ está aqui, porque o
    bit `caixa_valida` depende dela e o smoke o verifica.
    As cadeias e o currículo entram no passo 4, depois do portão do passo 3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

# fatias do comando
ALVO = slice(0, 3)
FACE = slice(3, 6)
DIR = slice(6, 9)
VALIDA = slice(9, 10)
COMANDO_DIM = 10

# As 6 faces, como eixos no frame da CAIXA. As 4 primeiras são LATERAIS; as duas
# últimas são topo e fundo, e exigem tombar a caixa (salto qualitativo, nível 6).
FACE_AXES = (
    (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
)
N_LATERAIS = 4


def _rot_z(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Roda `v` [B,3] em torno de z pelo ângulo `ang` [B]."""
    c, s = torch.cos(ang), torch.sin(ang)
    out = torch.empty_like(v)
    out[:, 0] = c * v[:, 0] - s * v[:, 1]
    out[:, 1] = s * v[:, 0] + c * v[:, 1]
    out[:, 2] = v[:, 2]
    return out


class CaixaAlvoCommand(CommandTerm):
    """Publica o alvo da caixa, o bit de validade, e resolve a forma do episódio."""

    cfg: CaixaAlvoCommandCfg

    def __init__(self, cfg: CaixaAlvoCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene["robot"]
        self.caixa: Entity = env.scene["box"]
        self.prateleira: Entity = env.scene["table"]

        self._command = torch.zeros(self.num_envs, COMANDO_DIM, device=self.device)
        self._face_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._dir_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._ang = torch.zeros(self.num_envs, device=self.device)
        self._pendente = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # a distância comandada no começo do elo. Ela é o σ do `bringing` (§8.2).
        self.dist_inicial = torch.full(
            (self.num_envs,), cfg.bringing_std_piso, device=self.device)

        # sucesso TRAVADO. O episódio NÃO termina no sucesso (§7.5).
        self.episode_success = torch.zeros(self.num_envs, device=self.device)
        self._sustenta = torch.zeros(self.num_envs, device=self.device)

        # a forma do episódio. `manipula` é o inverso de `locomocao`.
        self.manipula = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # buffers que outros managers leem. Existem ANTES do 1º reset, porque as
        # recompensas leem `env.poc_valida` no primeiro passo.
        env.poc_valida = self._command[:, VALIDA].squeeze(-1)
        env.poc_twist_zero = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device)
        env.poc_dist_inicial = self.dist_inicial
        env.poc_success = self.episode_success

        self.metrics["erro_posicao"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["erro_angulo_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["no_alvo"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["frac_manipula"] = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ leitura
    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def valida(self) -> torch.Tensor:
        """[B] — 1,0 quando existe tarefa de caixa."""
        return self._command[:, 9]

    def erro_pos(self) -> torch.Tensor:
        return torch.norm(
            self._command[:, ALVO] - self.caixa.data.root_link_pos_w, dim=-1)

    def erro_ang(self) -> torch.Tensor:
        """Ângulo entre a normal da face alvo, em MUNDO, e o `dir_alvo`, em MUNDO.

        UM escalar, e a simetria do cubo se resolve sozinha: girar em torno da
        normal da face não muda o vetor. A métrica CONCORDA com o objetivo.
        (Mesma formulação do `erro_angulo_deg` do g1_multitask.)
        """
        face_b = torch.tensor(FACE_AXES, device=self.device)[self._face_idx]
        normal_w = quat_apply(self.caixa.data.root_link_quat_w, face_b)
        cos = torch.sum(normal_w * self._dir_w, dim=-1).clamp(-1.0, 1.0)
        return torch.acos(cos)

    def de_pe(self) -> torch.Tensor:
        """[B] bool — a pelve alta e o tronco pouco inclinado."""
        z = self.robot.data.root_link_pos_w[:, 2]
        g = self.robot.data.projected_gravity_b
        inclinacao = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
        return (z >= self.cfg.pelve_min) & (inclinacao <= self.cfg.inclinacao_max_rad)

    # ------------------------------------------------------------------ resample
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return

        # --- a forma do episódio ---
        # ⚠ Ela é sorteada pelo CURRÍCULO (`curriculo.sorteia_forma`), e não aqui.
        # No reset a ordem do mjlab é currículo → eventos → comando, e o
        # `afasta_cena` precisa da forma ANTES deste método rodar.
        forma = getattr(self._env, "poc_manipula", None)
        if forma is None:
            manipula = torch.ones(n, dtype=torch.bool, device=self.device)
        else:
            manipula = forma[env_ids]
        self.manipula[env_ids] = manipula

        # --- o alvo do elo `pegar`, em MUNDO ---
        # Altura ABSOLUTA. Agachar não move este alvo, portanto o robô tem de
        # ficar de pé. É a decisão central do ADR-0001, e ela está certa.
        r = self.cfg.pegar_range
        lo = torch.tensor([r[0][0], r[1][0], r[2][0]], device=self.device)
        hi = torch.tensor([r[0][1], r[1][1], r[2][1]], device=self.device)
        alvo = lo + (hi - lo) * torch.rand(n, 3, device=self.device)
        # o x,y do alvo é relativo à origem do env; o z é do MUNDO
        origem = self._env.scene.env_origins[env_ids]
        alvo[:, 0] += origem[:, 0]
        alvo[:, 1] += origem[:, 1]
        self._command[env_ids, ALVO] = alvo

        # --- a face e o giro pedido ---
        # ESQUELETO: nível 0 pede 0°. `dir_alvo` recebe a normal ATUAL da face,
        # o que significa "erga sem torcer".
        self._face_idx[env_ids] = torch.randint(
            0, N_LATERAIS, (n,), device=self.device)
        self._ang[env_ids] = 0.0

        # --- o bit ---
        self._command[env_ids, 9] = manipula.float()

        # zera o sucesso e o cronômetro de sustentação
        self.episode_success[env_ids] = 0.0
        self._sustenta[env_ids] = 0.0
        self._pendente[env_ids] = True

    def _resolver(self) -> None:
        """Resolve, contra a pose FRESCA, o que o resample não podia resolver.

        ⚠ No reset o command manager roda DEPOIS dos eventos que reposicionam a
        caixa, mas as grandezas derivadas ainda não foram recalculadas. Ler pose
        no `_resample_command` devolve a pose ANTERIOR ao reset. Bug medido em
        30/07 no g1_multitask: o erro de ângulo saía espalhado em vez de exato.
        """
        if not bool(self._pendente.any()):
            return
        ids = self._pendente.nonzero().flatten()

        face_b = torch.tensor(FACE_AXES, device=self.device)[self._face_idx[ids]]
        normal_w = quat_apply(self.caixa.data.root_link_quat_w[ids], face_b)
        dir_w = _rot_z(normal_w, self._ang[ids])
        self._dir_w[ids] = dir_w / dir_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self._command[ids, FACE] = face_b

        # σ do `bringing` = a distância a vencer no começo do elo (§8.2).
        # O σ fixo de 0,30 m do mjlab já está saturado quando a caixa sobe 0,17 m.
        d = torch.norm(
            self._command[ids, ALVO] - self.caixa.data.root_link_pos_w[ids], dim=-1)
        self.dist_inicial[ids] = torch.clamp(d, min=self.cfg.bringing_std_piso)

        self._pendente[ids] = False

    # -------------------------------------------------------------------- update
    def _update_command(self) -> None:
        self._resolver()

        # `dir_alvo` da OBS: o alvo vive em MUNDO e a obs é egocêntrica, portanto
        # a conversão é por passo. Sem isto a política veria o vetor do spawn, que
        # deixa de apontar para o lugar certo assim que o robô gira.
        self._command[:, DIR] = quat_apply_inverse(
            self.robot.data.root_link_quat_w, self._dir_w)

        # com o bit em 0 as três fatias de caixa são zeradas (§5.1.1)
        zero = self.valida.unsqueeze(-1) < 0.5
        self._command[:, FACE] = torch.where(
            zero, torch.zeros_like(self._command[:, FACE]), self._command[:, FACE])
        self._command[:, DIR] = torch.where(
            zero, torch.zeros_like(self._command[:, DIR]), self._command[:, DIR])

        # o twist é zero nos elos de manipulação (ESQUELETO: o único elo é `pegar`)
        self._env.poc_twist_zero = self.manipula.clone()

        # --- o fecho do elo `pegar`: 4 condições, sustentadas ---
        fecha = (
            (self.erro_pos() < self.cfg.raio_sucesso)
            & (self.erro_ang() < self.cfg.angulo_sucesso_rad)
            & self.de_pe()
            & self.manipula
        )
        dt = self._env.step_dt
        self._sustenta = torch.where(fecha, self._sustenta + dt,
                                     torch.zeros_like(self._sustenta))
        # sucesso TRAVADO. O episódio continua — terminar aqui puniria o sucesso.
        self.episode_success = torch.maximum(
            self.episode_success,
            (self._sustenta >= self.cfg.sustenta_pegar_s).float(),
        )

    def _update_metrics(self) -> None:
        self._resolver()
        self.metrics["erro_posicao"] = self.erro_pos() * self.valida
        self.metrics["erro_angulo_deg"] = torch.rad2deg(self.erro_ang()) * self.valida
        self.metrics["no_alvo"] = (self.erro_pos() < self.cfg.raio_sucesso).float() * self.valida
        self.metrics["episode_success"] = self.episode_success
        self.metrics["frac_manipula"] = self.manipula.float()

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        for i in visualizer.get_env_indices(self.num_envs):
            if float(self.valida[i]) < 0.5:
                continue
            visualizer.add_sphere(
                center=self._command[i, ALVO].cpu().numpy(),
                radius=self.cfg.raio_sucesso,
                color=(1.0, 0.5, 0.0, 0.3),
                label=f"alvo_{i}",
            )


@dataclass(kw_only=True)
class CaixaAlvoCommandCfg(CommandTermCfg):
    # faixas do alvo do `pegar`, em MUNDO (x, y, z)
    pegar_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    frac_locomocao: float = 0.30
    raio_sucesso: float = 0.05
    angulo_sucesso_rad: float = math.radians(20.0)
    sustenta_pegar_s: float = 1.0
    pelve_min: float = 0.65
    inclinacao_max_rad: float = math.radians(20.0)
    bringing_std_piso: float = 0.10

    def build(self, env: ManagerBasedRlEnv) -> CaixaAlvoCommand:
        return CaixaAlvoCommand(self, env)


class TwistPoc(UniformVelocityCommand):
    """O twist do mjlab, com dois acréscimos.

    (a) ZERO nos elos de manipulação. Aplicado DEPOIS do `super()`, portanto vence
        o sorteio de standing e de heading.
    (b) GIRO NO LUGAR: metade dos envs parados recebe vx = vy = 0 e |ωz| ≥ piso.
        Sem isto os envs parados nunca giram, e o robô não aprende a girar no
        lugar. (Acréscimo do g1_multitask, e ele fica.)
    """

    cfg: TwistPocCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        # (b) giro no lugar, dentro do subconjunto parado
        parado = self.is_standing_env[env_ids]
        sorteio = torch.rand(len(env_ids), device=self.device)
        giro = parado & (sorteio < self.cfg.frac_giro_no_standing)
        ids_giro = env_ids[giro]
        if len(ids_giro) > 0:
            self.vel_command_b[ids_giro, 0] = 0.0
            self.vel_command_b[ids_giro, 1] = 0.0
            w = self.vel_command_b[ids_giro, 2]
            sinal = torch.where(w >= 0, 1.0, -1.0)
            piso = self.cfg.piso_giro_rad_s
            self.vel_command_b[ids_giro, 2] = sinal * torch.clamp(w.abs(), min=piso)
            # Sai das TRÊS máscaras. Só sair da de standing não basta: o
            # `_update_command` do fabricante sobrescreve o ωz dos envs de heading,
            # e reprojeta o vx,vy dos envs de mundo. Sem isto o piso de giro é
            # apagado em silêncio.
            self.is_standing_env[ids_giro] = False
            self.is_heading_env[ids_giro] = False
            self.is_world_env[ids_giro] = False

    def _update_command(self) -> None:
        super()._update_command()
        # (a) zero nos elos de manipulação
        zero = getattr(self._env, "poc_twist_zero", None)
        if zero is not None:
            self.vel_command_b[zero] = 0.0


@dataclass(kw_only=True)
class TwistPocCfg(UniformVelocityCommandCfg):
    frac_giro_no_standing: float = 0.5
    piso_giro_rad_s: float = 0.15

    def build(self, env: ManagerBasedRlEnv) -> TwistPoc:
        return TwistPoc(self, env)
