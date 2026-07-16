"""Comando da task Levanta-Caixa: a META que a política persegue.

Layout do vetor de comando (num_envs, 4):
  [0:3] target_pos -> onde a CAIXA deve estar (frame do MUNDO, já com env_origin)
  [3]   phase      -> 0 = segurar (hold) agora; 1 = largar (place) no futuro

A coluna de FASE é reservada desde já (constante 0). Ela existe pra que, quando
o place chegar, o SHAPE de obs/ação continue idêntico -> o checkpoint da política
de pega-e-segura transfere (warm-start), em vez de retreino do zero. Caveat: o bit
constante é ignorado pela rede até um fine-tune com ele variando (fase de place).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

TARGET_IDX = slice(0, 3)
PHASE_IDX = 3
COMMAND_DIM = 4


class LiftBoxCommand(CommandTerm):
    """Alvo 3D de sustentação da caixa (+ bit de fase reservado)."""

    cfg: "LiftBoxCommandCfg"

    def __init__(self, cfg: "LiftBoxCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.box: Entity = env.scene[cfg.entity_name]
        # (1) GUARDAR a meta: um tensor [B, 4], zerado (a fase já nasce 0).
        self._command = torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)
        # (3) MÉTRICA pro TensorBoard: erro de posição caixa->alvo.
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # (2) SORTEAR meta nova (só pros envs que resetaram): alvo uniforme nos ranges.
        n = len(env_ids)

        def _u(rng: tuple[float, float]) -> torch.Tensor:
            return torch.empty(n, device=self.device).uniform_(rng[0], rng[1])

        target = torch.stack([_u(self.cfg.target_x), _u(self.cfg.target_y),
                              _u(self.cfg.target_z)], dim=-1)
        # alvo no MUNDO = alvo local + origem do ambiente (por-env, igual à caixa/mesa)
        self._command[env_ids, TARGET_IDX] = target + self._env.scene.env_origins[env_ids]
        self._command[env_ids, PHASE_IDX] = self.cfg.phase_value  # 0 = hold (reservado)

    def _update_command(self) -> None:
        pass  # alvo estático entre resamples

    def _update_metrics(self) -> None:
        max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        box_pos = self.box.data.root_link_pos_w
        err = torch.norm(self._command[:, TARGET_IDX] - box_pos, dim=-1)
        self.metrics["position_error"] += err / max_step

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Esfera translúcida no alvo -- mesmo padrão do `LiftingCommand` do mjlab
        (manipulation/mdp/commands.py), só que pro nosso vetor de comando (4D,
        alvo em [0:3]). Só desenha se `cfg.debug_vis=True` (o manager já filtra)."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        for batch in env_indices:
            target_pos = self._command[batch, TARGET_IDX].cpu().numpy()
            visualizer.add_sphere(
                center=target_pos, radius=0.03,
                color=self.cfg.viz.target_color, label=f"lift_box_target_{batch}",
            )


@dataclass(kw_only=True)
class LiftBoxCommandCfg(CommandTermCfg):
    entity_name: str
    """Entidade cuja pose alimenta a métrica de erro (a caixa)."""
    target_x: tuple[float, float] = (0.40, 0.50)
    target_y: tuple[float, float] = (-0.05, 0.05)
    target_z: tuple[float, float] = (0.78, 0.85)  # acima do topo da mesa (0.55) => erguer
    phase_value: float = 0.0                       # 0=hold agora; 1=place (futuro)

    @dataclass
    class VizCfg:
        target_color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.3)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: "ManagerBasedRlEnv") -> LiftBoxCommand:
        return LiftBoxCommand(self, env)
