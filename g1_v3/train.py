"""Entrada de treino da v3.

    python -m g1_v3.train --env.scene.num-envs 4096

Ela não faz nada além de registrar a task e chamar o treino do mjlab. Toda a configuração mora em `knobs.py`,
e todo o resto do pacote é montado por `env_cfg.make_env_cfg`.
"""
from __future__ import annotations

import g1_v3.tarefa  # noqa: F401  (o import registra a task)
from mjlab.scripts.train import main as train_main

if __name__ == "__main__":
    train_main()
