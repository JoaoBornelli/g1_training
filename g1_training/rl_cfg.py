"""PPO runner da Levanta-Caixa: reusa o padrão do G1 (velocity), só troca o logger
(tensorboard, pra não travar no login do wandb) e o experiment_name (compartilhado
entre as fases -> warm-start cruza via --agent.load-run)."""
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def lift_box_ppo_runner_cfg(run_name: str = ""):
    cfg = unitree_g1_ppo_runner_cfg()
    cfg.logger = "tensorboard"
    cfg.experiment_name = "g1_lifting_box"
    cfg.run_name = run_name
    return cfg
