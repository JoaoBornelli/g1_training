"""§9 — o quarto regime do `variable_posture`.

O `variable_posture` do mjlab escolhe o σ por regime de VELOCIDADE comandada:
`standing`, `walking`, `running`. O elo `pegar` roda com o twist em zero, portanto
ele cai em `standing`, e o `std_standing` do G1 é `{".*": 0,05}`.

Um ombro deslocado 0,5 rad dá 0,25/0,0025 = 100. Portanto o termo cobra do robô por
esticar o braço.

Uma prateleira a 0,04 m exige um agachamento acima de 1,5 rad no joelho. O
`std_running` do G1 dá 0,6 rad ao joelho. Portanto o termo cobra do robô por agachar.

É o defeito que o repositório mediu em 17/07: "posture 0,8 briga com o squat".

A solução é um quarto regime, e ele lê a DEMANDA DA CAIXA:

    demanda = peso_dist · ‖caixa − alvo‖ + peso_ang · Δθ

"Demanda" é quanto trabalho falta. O braço fica livre enquanto há trabalho, e a pose
aperta quando o trabalho acaba — o que dá um GRADIENTE que levanta o robô no fim da
manobra. É o `hold_still_bonus` do repo, agora de graça.

Os três dicionários do G1 ficam INTOCADOS. A marcha validada não muda.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import variable_posture
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class postura_manipulacao(variable_posture):  # noqa: N801 (idioma do mjlab)
    """`variable_posture` com um quarto regime, escolhido pela demanda da caixa."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Entity = env.scene[asset_cfg.name]
        _, joint_names = asset.find_joints(asset_cfg.joint_names)
        _, _, std_m = resolve_matching_names_values(
            data=cfg.params["std_manipulando"],
            list_of_strings=joint_names,
        )
        self.std_manipulando = torch.tensor(
            std_m, device=env.device, dtype=torch.float32)

    def __call__(  # type: ignore[override]
        self,
        env: ManagerBasedRlEnv,
        std_standing,
        std_walking,
        std_running,
        std_manipulando,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        caixa_command_name: str,
        peso_dist: float,
        peso_ang: float,
        limiar: float,
        walking_threshold: float = 0.05,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running, std_manipulando  # resolvidos no init

        asset: Entity = env.scene[asset_cfg.name]
        twist = env.command_manager.get_command(command_name)
        assert twist is not None

        # --- os três regimes do mjlab, pela velocidade comandada ---
        vel = torch.norm(twist[:, :2], dim=1) + torch.abs(twist[:, 2])
        m_stand = (vel < walking_threshold).float().unsqueeze(1)
        m_walk = ((vel >= walking_threshold) & (vel < running_threshold)).float().unsqueeze(1)
        m_run = (vel >= running_threshold).float().unsqueeze(1)
        std_vel = (
            self.std_standing * m_stand
            + self.std_walking * m_walk
            + self.std_running * m_run
        )

        # --- o quarto regime, pela demanda da caixa ---
        caixa_cmd = env.command_manager.get_term(caixa_command_name)
        bit = caixa_cmd.command[:, 9]
        demanda = (
            peso_dist * caixa_cmd.erro_pos() + peso_ang * caixa_cmd.erro_ang()
        ) * bit
        m_manip = (demanda >= limiar).float().unsqueeze(1)

        std = m_manip * self.std_manipulando + (1.0 - m_manip) * std_vel

        env.extras["log"]["Metrics/postura_frac_manipulando"] = m_manip.mean()
        env.extras["log"]["Metrics/postura_demanda_caixa"] = demanda.mean()

        q = asset.data.joint_pos[:, asset_cfg.joint_ids]
        q0 = self.default_joint_pos[:, asset_cfg.joint_ids]
        erro_sq = torch.square(q - q0)
        return torch.exp(-torch.mean(erro_sq / (std**2), dim=1))
