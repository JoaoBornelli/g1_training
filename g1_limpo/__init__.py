"""g1_limpo — loco-manipulação de caixa do G1, reescrita sem import de projeto.

Desenho em uma frase: **um objetivo, alcançado por cadeias de no máximo dois elos,
e os elos trocam DENTRO do episódio.**

    ANDAR  REORIENTAR  PEGAR  CARREGAR  BOTAR      <- os 5 estados do one-hot
    (PEGAR)  (REORIENTAR->PEGAR)  (PEGAR->CARREGAR)  (PEGAR->BOTAR)   <- as cadeias

O que este pacote NÃO tem, e por quê: orçamento equalizado por tarefa, grafo de
pais entre tarefas, eixos de currículo congelados, distribuição inversa à
competência, e régua de sucesso com tolerância de velocidade. Todos existiam para
resolver o problema de haver CINCO tarefas com cinco conjuntos de alvo. Com um
objetivo só, eles caem por consequência.

⚠ INVARIANTE DO PACOTE: zero import de código do projeto. Não importa
`g1_training`, nem `g1_poc`, nem `g1_multitask`. A única exceção é `paridade.py`,
que é um verificador descartável e não roda em treino.

Especificação: `specs/g1-limpo.md`. Plano: `docs/planos/2026-08-25-g1-limpo.md`.

Importar este módulo registra a task:

    import g1_limpo     # registra Mjlab-G1-Limpo
"""
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from g1_limpo.env_cfg import make_env_cfg

TASK_ID = "Mjlab-G1-Limpo"
EXPERIMENT = "g1_limpo"


def rl_cfg():
    """O PPO do mjlab, sem mudança, com `experiment_name` próprio.

    ⚠ O `experiment_name` próprio NÃO é cosmético. O `load_run` default é o regex
    `.*`, portanto um `--agent.resume` casaria com a run errada de OUTRO pacote. O
    engano já custou uma sessão a este repositório, e o `g1_training/rl_cfg.py`
    ainda compartilha `g1_lifting_box` entre módulos. Aqui o nome é isolado.
    """
    cfg = unitree_g1_ppo_runner_cfg()
    cfg.experiment_name = EXPERIMENT
    return cfg


# ⚠ O `runner_cls` NÃO é cosmético: sem ele o estado do currículo NÃO vai para o
# checkpoint, e o Colab/Kaggle matam sessão no meio de um bloco de 5000 iterações. A
# rampa de ~400 iterações seria re-paga a cada reinício, e o currículo ficaria
# não-monotônico. Ver `runner.py`.
from g1_limpo.runner import RunnerComEstadoDeCurriculo  # noqa: E402

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=rl_cfg(),
    runner_cls=RunnerComEstadoDeCurriculo,
)

# ---------------------------------------------------------------- INSPEÇÃO
# Duas tasks a mais, SÓ para a revisão visual. Elas usam o MESMO `make_env_cfg`,
# com o robô travado e as terminações desligadas — nada de cena ou de alvo é
# recalculado. Ver `inspeciona.py`.
# UMA task por ELO. Assim `inspeciona --viewer pegar` mostra o alvo do `pegar`, e
# `--viewer botar` mostra o do `botar`, cada um com a cena que aquele elo tem.
from g1_limpo.comando import ELOS  # noqa: E402

TASK_INSPECAO = {
    nome: f"Mjlab-G1-Limpo-Inspecao-{nome.capitalize()}" for nome in ELOS
}

for _i, _nome in enumerate(ELOS):
    register_mjlab_task(
        task_id=TASK_INSPECAO[_nome],
        env_cfg=make_env_cfg(inspecao=True, elo=_i),
        play_env_cfg=make_env_cfg(play=True, inspecao=True, elo=_i),
        rl_cfg=rl_cfg(),
    )

__all__ = ["TASK_ID", "TASK_INSPECAO", "EXPERIMENT", "make_env_cfg", "rl_cfg"]
