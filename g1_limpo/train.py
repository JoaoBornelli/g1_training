"""Entrada de treino do g1_limpo.

    python -m g1_limpo.train --env.scene.num-envs 4096

Ela não faz nada além de registrar a task e chamar o treino do mjlab. Toda a
configuração mora em `knobs.py`, e todo o resto do pacote é montado por
`env_cfg.make_env_cfg`.
"""
from __future__ import annotations

import g1_limpo  # noqa: F401  (o import registra a task)
from mjlab.scripts.train import main as train_main

if __name__ == "__main__":
    train_main()
