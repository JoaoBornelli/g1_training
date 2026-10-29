"""Padrão do MID-360 em trechos de 20 ms (A4; PERC §5, S §4).

Lê as 800 000 direções da Livox (4 s de varredura) e grava 200 trechos de 20 ms, com 205
direções unitárias cada. O passo de física do lidar usa um trecho por passo (PD-5).
"""
import urllib.request
from pathlib import Path

import numpy as np

URL_CSV = "https://raw.githubusercontent.com/Livox-SDK/livox_laser_simulation/main/scan_mode/mid360.csv"
CSV = Path.home() / "orca/workspaces/g1_training/lidar_sim/sensores/mid360.csv"
SAIDA = Path(__file__).resolve().parent / "mid360_passos.npz"

TRECHOS = 200              # 200 × 20 ms = 4 s, uma volta do padrão
LINHAS_POR_TRECHO = 4000   # 20 ms a 200 mil pontos por segundo
RAIOS = 205                # PD-5: raios por passo de 20 ms


def _angulos() -> np.ndarray:
    """(800 000, 2) azimute e zênite em radianos. Baixa o CSV se ele não existir."""
    if not CSV.exists():
        try:
            urllib.request.urlretrieve(URL_CSV, CSV)
        except OSError as erro:
            CSV.unlink(missing_ok=True)
            raise RuntimeError(f"falha ao baixar {URL_CSV}: {erro}") from erro
    # O CSV vem em graus (cabeçalho Time/s,Azimuth/deg,Zenith/deg); a fórmula de lidar.py:60 pede rad.
    return np.radians(np.loadtxt(CSV, delimiter=",", skiprows=1, usecols=(1, 2)))


def gera() -> np.ndarray:
    ang = _angulos()
    if ang.shape != (TRECHOS * LINHAS_POR_TRECHO, 2):
        raise ValueError(f"CSV com {len(ang)} linhas; esperado {TRECHOS * LINHAS_POR_TRECHO}")
    # Linha j do trecho é ⌊j·4000/205⌋, j = 0..204: o máximo é 3 980, dentro do trecho.
    dentro = (np.arange(RAIOS) * LINHAS_POR_TRECHO) // RAIOS
    idx = np.arange(TRECHOS)[:, None] * LINHAS_POR_TRECHO + dentro[None, :]
    az, ze = ang[idx, 0], ang[idx, 1]
    # Mesma fórmula de lidar.py:60, em float32 (o treino não precisa de mais precisão).
    dirs = np.stack([np.sin(ze) * np.cos(az), np.sin(ze) * np.sin(az), np.cos(ze)],
                    axis=-1).astype(np.float32)
    np.savez(SAIDA, dirs=dirs)
    return dirs


if __name__ == "__main__":
    print(f"{SAIDA}: {gera().shape}")
