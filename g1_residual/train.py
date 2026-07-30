"""Launcher do experimento residual na Kaggle.

    # treino curto — é experimento, não a run de produção
    python g1_residual/train.py --env.scene.num-envs 2048 \
                                --agent.max-iterations 3000

    # depois, para olhar
    python play.py --task Mjlab-Residual-Pegar-Unitree-G1 \
                   --checkpoint logs/g1_residual_pegar/<run>/model_<n>.pt

⚠️ **`num_envs` é POR RANK.** E aqui cada passo roda o ator do BFM (31,9 M de
parâmetros, congelado) além da política, então o custo por passo sobe ~17%. Comece
com 2048 e olhe o `steps per second` antes de subir.

**Pré-requisito que não dá para pular:** `g1_residual/peso/bfm_ator.pt` (122 MB). Ele
sai do `extrai_ator.py`, que precisa do checkpoint de 3,15 GB — ou seja **roda no seu
PC, não aqui**. Na Kaggle ele entra como Dataset. Sem o arquivo o `AtorBFM` levanta
na hora, com a mensagem certa.

**O portão de referência também não roda aqui** (`referencia.py` precisa do ambiente
do BFM-Zero). Ele é local, e tem que ter passado ANTES de subir: erro na montagem da
obs do BFM não levanta exceção, só sai comportamento ruim.
"""
import os
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

# O import é o que registra a task. Tem que vir antes de qualquer `list_tasks()`.
import g1_residual  # noqa: E402, F401

from mjlab.scripts.train import main as mjlab_train_main  # noqa: E402

os.environ.setdefault("TORCHRUNX_LOG_DIR", str(RAIZ / "logs" / "torchrunx"))

if __name__ == "__main__":
    sys.argv = [sys.argv[0], g1_residual.TASK_ID, *sys.argv[1:]]
    mjlab_train_main()
