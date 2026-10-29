"""Entrada de inspeção COM política treinada.

    python -m g1_v3.play --checkpoint-file CAMINHO/model_1000.pt
"""
from __future__ import annotations

import g1_v3.tarefa  # noqa: F401  (o import registra a task)
from mjlab.scripts.play import main as play_main

if __name__ == "__main__":
    play_main()
