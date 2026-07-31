"""Visualiza a política residual no viewer. Roda no SEU PC, precisa de display.

    python g1_residual/play.py ~/Downloads/model_1500.pt
    python g1_residual/play.py ~/Downloads/model_1500.pt --tarefa pegar
    python g1_residual/play.py ~/Downloads/model_1500.pt --envs 4
    python g1_residual/play.py ~/Downloads/model_1500.pt --video     # sem janela

Não precisa de GPU: 1 robô roda em CPU sem problema, e o ator do BFM é congelado.

**Por que este arquivo e não o `play.py` da raiz.** O `play.py` da raiz tem
`--task` restrito a `("Stand", "Stand-Step", "Lift")` e importa só o `g1_training`.
A task residual é registrada no import do `g1_residual`, então ela nem aparece lá.

`--tarefa` força UMA das 7 no viewer. Sem ela o currículo sorteia, e num robô só você
vê a que caiu no sorteio. Ele mexe no `abertas` do orquestrador, que é a única forma
que o currículo respeita — escrever em `env.tarefa_sorteada` não tem efeito, porque o
`_amostrar` sobrescreve no reset.

No viewer nativo: Ctrl+arrasto empurra o robô, SPACE pausa, setinha dá 1 passo,
`-`/`=` muda a velocidade.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import g1_residual  # noqa: E402  (o import registra a task)
from g1_multitask import tasks as T  # noqa: E402
from g1_multitask.runner import MultitaskRunner  # noqa: E402
from g1_residual.env_residual import OrquestradorPegar  # noqa: E402
from mjlab.scripts.play import PlayConfig, run_play  # noqa: E402
from mjlab.tasks.registry import register_mjlab_task  # noqa: E402

NOMES = {
    "parado": T.PARADO, "andar": T.ANDAR, "pegar": T.PEGAR, "botar": T.BOTAR,
    "reorientar": T.REORIENTAR, "parado_caixa": T.PARADO_CAIXA,
    "andar_caixa": T.ANDAR_CAIXA,
}


def _forca_tarefa(tarefa: int) -> None:
    """Re-registra a task com o orquestrador abrindo só a tarefa pedida.

    O `OrquestradorPegar` já é a subclasse que fixa a lista `abertas`; aqui só troco
    QUAL tarefa ele fixa. O `super()` explícito pula o `__init__` dele e vai direto no
    `Orquestrador`, senão a lista voltaria para o `pegar`."""

    class _Uma(OrquestradorPegar):
        def __init__(self, cfg, env):
            super(OrquestradorPegar, self).__init__(cfg, env)
            self.abertas = [tarefa]
            env.tarefa_sorteada[:] = tarefa

    cfg = g1_residual.build_env_residual(play=True)
    cfg.curriculum["orquestrador"].func = _Uma
    # `register_mjlab_task` sobrescreve a entrada do dict, então re-registrar com o
    # MESMO id é o jeito de trocar o cfg sem tocar em nada do mjlab.
    register_mjlab_task(
        task_id=g1_residual.TASK_ID, env_cfg=cfg, play_env_cfg=cfg,
        rl_cfg=g1_residual._rl_cfg(), runner_cls=MultitaskRunner)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str, help="caminho pro model_<iter>.pt")
    p.add_argument("--tarefa", choices=sorted(NOMES), default=None,
                   help="força uma das 7 no viewer (default: o currículo sorteia)")
    p.add_argument("--envs", type=int, default=1)
    p.add_argument("--video", action="store_true", help="grava mp4, sem janela")
    p.add_argument("--video-length", type=int, default=500)
    args = p.parse_args()

    ckpt = pathlib.Path(args.checkpoint).expanduser()
    assert ckpt.is_file(), f"não achei {ckpt}"

    if args.tarefa:
        _forca_tarefa(NOMES[args.tarefa])
        print(f"[PLAY] tarefa forçada: {args.tarefa}")

    run_play(g1_residual.TASK_ID, PlayConfig(
        agent="trained", checkpoint_file=str(ckpt), num_envs=args.envs,
        viewer="native", video=args.video, video_length=args.video_length,
    ))


if __name__ == "__main__":
    main()
