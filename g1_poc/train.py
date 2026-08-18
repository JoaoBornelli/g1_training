"""Treino do g1_poc. Delega ao `mjlab.scripts.train`.

    python -m g1_poc.train --task Mjlab-G1-Poc --env.scene.num-envs 4096

⚠ Três coisas que o repositório já pagou por esquecer:

1. **`num_envs`.** O default do mjlab é **1**. Esquecer a flag roda 1 env, em
   silêncio, sem uma linha de erro.
2. **Warm-start com `learning_rate = 5e-4`.** Sempre. Sem isso os primeiros updates
   destroem o equilíbrio e o `fell_over` dispara. O ADR-0001 declarou esta mitigação
   e ela nunca existiu no código:

       --agent.algorithm.learning-rate 5e-4

3. **`--agent.resume`.** O `load_run` default é o regex `.*`. O `experiment_name`
   próprio (`g1_poc`) impede casar com a run de outro pacote.
"""
from __future__ import annotations

import g1_poc  # noqa: F401  (registra a task)


def main() -> None:
    from mjlab.scripts.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
