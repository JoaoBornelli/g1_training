"""Launcher do treino na Kaggle. A run é FATIADA em blocos de 2k-3k iterações.

    # bloco 1 — do zero
    python g1_multitask/train.py --env.scene.num-envs 4096 \
                                 --agent.max-iterations 2500

    # blocos seguintes — retoma do checkpoint
    python g1_multitask/train.py --env.scene.num-envs 4096 \
                                 --agent.max-iterations 2500 \
                                 --agent.resume True

    # dual T4
    python g1_multitask/train.py --gpu-ids "all" --env.scene.num-envs 4096

⚠️ **`num_envs` é POR RANK, não total.** Com dual T4 e 4096 por rank são 8192 envs, e
o orçamento de 30 000 iterações da §14 se esgota em ~15 000. É por isso que a alavanca
de corte existe (altura 7 → 4 níveis, distância 4 → 3, ~20 dos 60 eventos): melhor
puxar deliberadamente no bloco 1, com o número na mão, do que descobrir na semana 3.

**Por que este arquivo existe em vez de chamar o `train` do mjlab direto.** O
`main()` do mjlab só faz `import mjlab.tasks` antes de listar as tasks, e o registro
do `g1_multitask` acontece no import DELE. Como o `_REGISTRY` é um dict de módulo,
importar aqui antes de delegar basta — a task aparece na lista sem tocar em nada do
mjlab.

**Depois de cada bloco**, rode o relatório antes de decidir o próximo:

    python g1_multitask/entre_blocos.py

⚠️ **`torchrunx` manda o stdout dos workers pra arquivo** (`TORCHRUNX_LOG_DIR`, default
`{log_dir}/torchrunx`) → as linhas `[CURRICULO]` e o alarme de estagnação **não
aparecem na saída da célula**, e o rank 1 fica invisível. Aponte pra dentro de
`/kaggle/working` e conte com ler dois arquivos.
"""
import os
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

# O import é o que registra a task. Tem que vir antes de qualquer `list_tasks()`.
import g1_multitask  # noqa: E402, F401

from mjlab.scripts.train import main as mjlab_train_main  # noqa: E402

# Sem isto o alarme de estagnação e o log do currículo somem no DDP.
os.environ.setdefault("TORCHRUNX_LOG_DIR", str(RAIZ / "logs" / "torchrunx"))

if __name__ == "__main__":
    # o `main()` do mjlab lê o task id como 1º argumento posicional
    sys.argv = [sys.argv[0], g1_multitask.TASK_ID, *sys.argv[1:]]
    mjlab_train_main()
