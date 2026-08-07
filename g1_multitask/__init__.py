"""Registra a task do multi-tarefa. Pacote IRMÃO do `g1_training`, não parte dele.

O isolamento é a razão de existir deste pacote: `g1_training/` fica intocado, e
tudo que o multi-tarefa precisa dele entra por import. `register_mjlab_task` é
função comum do mjlab (registro é um dict de módulo, sem decorator e sem
auto-discovery), então registrar daqui não exige tocar no `__init__.py` de lá.

Ativação: `import g1_multitask` antes de qualquer `list_tasks()`. O `train`/`play`
do mjlab só fazem `import mjlab.tasks`, então quem chama tem que importar isto
primeiro — ver `train.py`.

⚠️ `experiment_name` compartilhado (`g1_lifting_box`, em `g1_training/rl_cfg.py`)
NÃO significa que o checkpoint da Lift carrega aqui: a obs do multi-tarefa tem
154 números contra 132 da Lift, e largura diferente é Categoria C ("recomeçar do
zero", §15). O treino do multi-tarefa começa do zero, e é a última vez que isso
é obrigatório.
"""
from mjlab.tasks.registry import register_mjlab_task

from g1_training.rl_cfg import lift_box_ppo_runner_cfg

from .configs import ACTIVE
from .env import build_multitask_env
from .runner import MultitaskRunner

TASK_ID = "Mjlab-Multitask-Unitree-G1"

EXPERIMENT = "g1_multitask"
"""`experiment_name` PRÓPRIO, e não o `g1_lifting_box` compartilhado.

O `lift_box_ppo_runner_cfg` do `g1_training` põe `g1_lifting_box` de propósito: as
skills Stand/Stand-Step/Lift compartilham obs e ação, então compartilhar o
`experiment_name` permite warm-start entre elas via `--agent.load-run`.

Aqui isso não vale, e o risco é real. O `load_run` default é o regex `.*`, e o
`launch_training` monta o diretório como `log_root/experiment_name/<timestamp>_<run>`.
Com o nome compartilhado, um `--agent.resume True` num log_root que também tenha runs
da Lift casaria com uma delas e tentaria carregar um checkpoint de **132** números na
rede de **154** — `size mismatch` na primeira camada.

E não há nada a perder: obs de largura diferente é Categoria C (§15), então checkpoint
da Lift não carregaria aqui de jeito nenhum. Nome separado = o resume só acha o que é
dele."""


def _rl_cfg():
    cfg = lift_box_ppo_runner_cfg(run_name="multitask")
    cfg.experiment_name = EXPERIMENT
    return cfg

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=build_multitask_env(ACTIVE, play=False),
    play_env_cfg=build_multitask_env(ACTIVE, play=True),
    rl_cfg=_rl_cfg(),
    # O `runner_cls` serve DUAS coisas: o congelamento do normalizador e o estado do
    # currículo no checkpoint. E de brinde resolve o `map_location` no `play`, que
    # já passa `map_location=device` (`play.py:206`).
    runner_cls=MultitaskRunner,
)
