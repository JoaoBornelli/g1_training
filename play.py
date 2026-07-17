"""Visualiza uma política treinada no viewer.

Uso (do venv do projeto, numa máquina COM display — não precisa de GPU):

    python play.py CAMINHO/model_999.pt --task Lift
    python play.py CAMINHO/model_999.pt --task Lift --shelf-top 0.45   # testa a caixa mais baixa
    python play.py CAMINHO/model_999.pt --task Lift --rehearsal        # inclui os envs "só de pé"
    python play.py CAMINHO/model_999.pt --task Stand-Step
    python play.py CAMINHO/model_999.pt --task Lift --video            # grava mp4 (headless)

Por padrão, na Lift o rehearsal fica DESLIGADO no play (a caixa sempre perto, pra
você inspecionar a pega/erguer limpo). `--shelf-top X` põe a prateleira mocap em
qualquer altura (testa o currículo). `--rehearsal` mantém os envs "só ficar de pé".

Interação no viewer nativo: Ctrl+arrasto = empurrar o robô/mover a caixa; SPACE
pausa; setinha = 1 step; -/= = devagar/rápido; R = mostra/esconde o alvo (esfera).

O checkpoint é o `model_<iter>.pt` salvo em <log_root>/g1_lifting_box/<timestamp>_<fase>/.
"""
import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import g1_training  # noqa: F401  (o import registra as tasks Stand/Stand-Step/Lift)
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import register_mjlab_task

from g1_training.rl_cfg import lift_box_ppo_runner_cfg
from g1_training.skills.lift.configs import ACTIVE as LIFT_ACTIVE
from g1_training.skills.lift.env import build_lift_env

_CUSTOM_TASK = "Mjlab-Lift-Box-Unitree-G1-LiftPlay"


def _register_custom_lift(shelf_top: float | None, rehearsal: bool,
                          weight: float | None = None) -> str:
    """Registra uma task de play da Lift com knobs ajustados (altura da prateleira /
    rehearsal on-off / peso da caixa), pra inspecionar sem mexer no config de treino."""
    over: dict = {}
    scene_over: dict = {}
    plr = LIFT_ACTIVE.plr
    if plr.shelf_levels:
        # PLR ativo: FIXA uma altura só pra inspecionar limpo (--shelf-top, senão a mais
        # BAIXA = a mais nova/difícil). level_jitter_z=0 → a caixa cai exatamente na altura.
        pin = shelf_top if shelf_top is not None else min(plr.shelf_levels)
        over["plr"] = dataclasses.replace(plr, shelf_levels=(pin,), level_jitter_z=0.0)
    else:
        if shelf_top is not None:
            scene_over["shelf_top"] = shelf_top
        if not rehearsal:
            scene_over["rehearsal_fraction"] = 0.0   # caixa sempre perto, pega/erguer limpo
    if weight is not None:
        # FIXA o peso da caixa (payload) nesse valor pra inspecionar um peso específico.
        # Sem --weight, o config varia o peso por episódio (aleatório no range).
        scene_over["box_weight_range"] = (weight, weight)
        print(f"[play] peso da caixa FIXADO em {weight} kg (payload)")
    if scene_over:
        over["scene"] = dataclasses.replace(LIFT_ACTIVE.scene, **scene_over)
    knobs = dataclasses.replace(LIFT_ACTIVE, **over)
    register_mjlab_task(
        task_id=_CUSTOM_TASK,
        env_cfg=build_lift_env(knobs, play=False),
        play_env_cfg=build_lift_env(knobs, play=True),
        rl_cfg=lift_box_ppo_runner_cfg(run_name="lift"),
    )
    return _CUSTOM_TASK


def main() -> None:
    p = argparse.ArgumentParser(description="Play de uma política treinada.")
    p.add_argument("checkpoint", type=str, help="caminho pro model_<iter>.pt")
    p.add_argument("--task", choices=("Stand", "Stand-Step", "Lift"), default="Stand",
                   help="fase treinada no checkpoint (default: Stand)")
    p.add_argument("--envs", type=int, default=1, help="nº de robôs na cena (default: 1)")
    p.add_argument("--shelf-top", type=float, default=None,
                   help="[Lift] altura da prateleira mocap a inspecionar. Com PLR ativo, "
                        "FIXA nessa altura (default = a mais baixa da lista de níveis)")
    p.add_argument("--rehearsal", action="store_true",
                   help="[Lift] mantém os envs 'só ficar de pé' (default: off no play)")
    p.add_argument("--weight", type=float, default=None,
                   help="[Lift] FIXA o peso da caixa em X kg (payload) pra inspecionar. "
                        "Sem isso, o config varia o peso por episódio")
    p.add_argument("--video", action="store_true",
                   help="grava mp4 em vez de abrir janela (use em máquina headless)")
    p.add_argument("--video-length", type=int, default=500, help="steps do vídeo (default: 500)")
    args = p.parse_args()

    ckpt = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        p.error(f"checkpoint não encontrado: {ckpt}")

    # Lift: por padrão desliga o rehearsal (caixa sempre perto). Com --shelf-top/--weight
    # ou sem --rehearsal, registra uma task custom; senão usa a registrada normal.
    if args.task == "Lift" and (args.shelf_top is not None or args.weight is not None
                                or not args.rehearsal):
        task_id = _register_custom_lift(args.shelf_top, rehearsal=args.rehearsal,
                                        weight=args.weight)
    else:
        task_id = f"Mjlab-Lift-Box-Unitree-G1-{args.task}"

    cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(ckpt),
        num_envs=args.envs,
        viewer="native",
        video=args.video,
        video_length=args.video_length,
    )
    run_play(task_id, cfg)


if __name__ == "__main__":
    main()
