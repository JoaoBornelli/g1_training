"""Entrada de inspeção COM política treinada.

    python -m g1_limpo.play --checkpoint-file CAMINHO/model_1000.pt

Para revisar a CENA e os ALVOS sem política, use o `inspeciona.py`: ele trava o robô
e não precisa de checkpoint.
"""
from __future__ import annotations

import g1_limpo  # noqa: F401  (o import registra a task)
from mjlab.scripts.play import main as play_main

if __name__ == "__main__":
    play_main()
