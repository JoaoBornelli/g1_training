"""Visualiza uma política treinada no viewer.

Uso (do venv do projeto, numa máquina COM display — não precisa de GPU):

    python play.py CAMINHO/model_999.pt
    python play.py CAMINHO/model_999.pt --task Lift
    python play.py CAMINHO/model_999.pt --video   # grava mp4 (headless)

O checkpoint é o `model_<iter>.pt` que o treino salva em
<log_root>/g1_lifting_box/<timestamp>_<fase>/.
"""
import argparse
import pathlib
import sys

# põe a raiz do repo no path pra `import g1_training` (o pacote) resolver de qualquer cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import g1_training  # noqa: F401  (o import registra as tasks Stand/Stand-Step/Lift)
from mjlab.scripts.play import PlayConfig, run_play


def main() -> None:
    p = argparse.ArgumentParser(description="Play de uma política treinada.")
    p.add_argument("checkpoint", type=str, help="caminho pro model_<iter>.pt")
    p.add_argument("--task", choices=("Stand", "Stand-Step", "Lift"), default="Stand",
                   help="fase treinada no checkpoint (default: Stand)")
    p.add_argument("--envs", type=int, default=1, help="nº de robôs na cena (default: 1)")
    p.add_argument("--video", action="store_true",
                   help="grava mp4 em vez de abrir janela (use em máquina headless)")
    p.add_argument("--video-length", type=int, default=500, help="steps do vídeo (default: 500)")
    args = p.parse_args()

    ckpt = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        p.error(f"checkpoint não encontrado: {ckpt}")

    cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(ckpt),
        num_envs=args.envs,
        viewer="native",
        video=args.video,
        video_length=args.video_length,
    )
    run_play(f"Mjlab-Lift-Box-Unitree-G1-{args.task}", cfg)


if __name__ == "__main__":
    main()
