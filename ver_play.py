"""Lançador do `play` para o g1_limpo, com elo forçado e robô LIVRE.

    python ver_play.py ~/Downloads/model_3800.pt              # elo sorteado
    python ver_play.py ~/Downloads/model_3800.pt andar --viser
    python ver_play.py ~/Downloads/model_3800.pt carregar --viser

Duas coisas justificam este arquivo em vez do CLI `play`:

1. O `play` do mjlab não descobre a nossa task — ela registra por efeito colateral do
   `import g1_limpo`, e o `--registry-name` dele é só para artefato de W&B.

2. ⚠ As `TASK_INSPECAO` do pacote NÃO servem para ver marcha: elas usam
   `inspecao=True`, que instala o `trava_robo` (`env_cfg.py:450`) e pina o robô na
   pose de reset a cada passo. Aqui registramos as variantes com `inspecao=False`, em
   que `elo=N` força o elo no CURRÍCULO (`env_cfg.py:328`) e no COMANDO
   (`env_cfg.py:369`) sem travar nada.

⚠ Só o viewer `viser` tem os sliders de comando. O `create_gui` do twist é exclusivo
dele, e o `compute` escreve o slider DEPOIS do `super().compute(dt)`, portanto o valor
pinado vence o re-sorteio de 3-8 s, os 10% de `standing_env` e o controlador de rumo.

⚠ Nos elos fora de (ANDAR, CARREGAR) o slider não tem efeito: o nosso termo `alvo_caixa`
roda depois do twist e chama `_zera_twist_nos_parados()`.
"""
from __future__ import annotations

import sys
from pathlib import Path

PREFIXO = "Mjlab-G1-Limpo-Livre-"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    ckpt = Path(sys.argv[1]).expanduser()
    if not ckpt.is_file():
        print(f"checkpoint não encontrado: {ckpt}")
        return 2

    resto = sys.argv[2:]
    quer_viser = "--viser" in resto
    envs = 1
    if "--envs" in resto:
        envs = int(resto[resto.index("--envs") + 1])

    import g1_limpo  # registra as tasks do pacote
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
    from g1_limpo.comando import ELOS
    from g1_limpo.env_cfg import make_env_cfg
    from g1_limpo.runner import RunnerComEstadoDeCurriculo
    from mjlab.scripts.play import PlayConfig, run_play

    nome_elo = next((a for a in resto if a in ELOS), None)

    # UMA task LIVRE por elo: elo forçado, robô solto, sem `trava_robo`.
    for i, nome in enumerate(ELOS):
        register_mjlab_task(
            task_id=f"{PREFIXO}{nome.capitalize()}",
            env_cfg=make_env_cfg(elo=i),
            play_env_cfg=make_env_cfg(play=True, elo=i),
            rl_cfg=unitree_g1_ppo_runner_cfg(),
            runner_cls=RunnerComEstadoDeCurriculo,
        )

    task = g1_limpo.TASK_ID if nome_elo is None else f"{PREFIXO}{nome_elo.capitalize()}"

    print(f"task       : {task}")
    print(f"checkpoint : {ckpt}")
    print(f"envs       : {envs}   viewer: {'viser' if quer_viser else 'native'}")
    if quer_viser:
        print("\nNo painel: pasta 'Twist' -> marque 'Enable' -> sliders vx, vy, wz.")
    else:
        print("\n⚠ Sem sliders no viewer native. Use --viser para mudar a direção.")
    print()

    try:
        run_play(task, PlayConfig(
            agent="trained",
            checkpoint_file=str(ckpt),
            num_envs=envs,
            device="cpu",
            viewer="viser" if quer_viser else "native",
        ))
    except (RuntimeError, TypeError) as e:
        msg = str(e)
        print("\n=== FALHOU ===")
        print(msg[:1200])
        if any(s in msg for s in ("cnn_cfg", "rnn_type", "unexpected keyword")):
            print("\nIncompatibilidade de versão: o `RslRlModelCfg` do mjlab local manda")
            print("campos que o `MLPModel` do rsl-rl-lib rejeita.")
            print("Conserto:  uv pip install mjlab==1.5.3")
        elif any(s in msg for s in ("state_dict", "size mismatch", "Missing key")):
            print("\nA arquitetura do checkpoint não casa com o cfg local.")
            print("Provável: mjlab 1.5.1 aqui contra 1.5.3 no treino.")
            print("Conserto:  uv pip install mjlab==1.5.3")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
