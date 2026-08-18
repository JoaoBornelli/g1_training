"""g1_poc — a POC de loco-manipulação de caixa do G1.

Desenho em uma frase: **uma tarefa, dois comandos, e o alvo da caixa define o
comportamento.** Não existe one-hot de tarefa, não existe orçamento equalizado, e
não existe gate por tarefa.

    twist(3)      ->  para onde andar. Zero = ficar parado.
    caixa_alvo(10) ->  onde a caixa deve ficar, e se existe caixa.

"Pegar", "carregar", "botar" e "reorientar" são POSIÇÕES DE ALVO, e não tarefas.

Especificação completa: ESPECIFICACAO-g1_poc.md.

Importar este módulo registra a task:

    import g1_poc     # registra Mjlab-G1-Poc
"""
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from g1_poc.env_cfg import make_g1_poc_env_cfg

TASK_ID = "Mjlab-G1-Poc"
EXPERIMENT = "g1_poc"


def rl_cfg():
    """O PPO do mjlab, sem mudança, com `experiment_name` próprio.

    ⚠ O `experiment_name` próprio NÃO é cosmético. O `load_run` default é o regex
    `.*`, portanto um `--agent.resume` casaria com a run errada de outro pacote. O
    engano de 04/08 custou uma sessão ao repositório.
    """
    cfg = unitree_g1_ppo_runner_cfg()
    cfg.experiment_name = EXPERIMENT
    return cfg


register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_g1_poc_env_cfg(),
    play_env_cfg=make_g1_poc_env_cfg(play=True),
    rl_cfg=rl_cfg(),
)

__all__ = ["TASK_ID", "EXPERIMENT", "make_g1_poc_env_cfg", "rl_cfg"]
