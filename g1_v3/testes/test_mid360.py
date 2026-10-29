"""Testes do padrão do MID-360 (A4 do plano). Escritos antes do código (lane de testes, A6).

O padrão é 200 trechos de 20 ms, cada um com 205 direções. O trecho 0 começa na primeira linha
do CSV da Livox (A4: linha ⌊j·4000/205⌋, com j = 0).
"""
import csv
import math
import unittest
from pathlib import Path

import numpy as np

DADOS = Path(__file__).resolve().parent.parent / "dados"
# CSV da Livox (A4). Se faltar, o teste falha: não há pulo silencioso.
CSV = Path("/home/joaobornelli/orca/workspaces/g1_training/lidar_sim/sensores/mid360.csv")


def _primeira_linha_do_csv():
    """Azimute e zênite (graus) da linha 1 de dados, com o cabeçalho 'Time/s,Azimuth/deg,Zenith/deg'."""
    assert CSV.exists(), f"CSV do MID-360 ausente: {CSV}"
    with open(CSV, newline="") as f:
        leitor = csv.reader(f)
        next(leitor)  # cabeçalho
        linha = next(leitor)
    return float(linha[1]), float(linha[2])


class TestMid360(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dirs = np.load(DADOS / "mid360_passos.npz", allow_pickle=False)["dirs"]

    def test_forma(self):
        # 200 trechos de 20 ms, 205 raios por trecho, xyz (A4).
        self.assertEqual(self.dirs.shape, (200, 205, 3))

    def test_normas_unitarias(self):
        # Cada direção é um vetor unitário; float32 dá erro de ~1e-7, a tolerância é 1e-5.
        normas = np.linalg.norm(self.dirs.astype(np.float64), axis=-1)
        np.testing.assert_allclose(normas, 1.0, rtol=0, atol=1e-5)

    def test_primeira_direcao_bate_com_o_csv(self):
        # Fórmula de lidar.py:60: (sin ze·cos az, sin ze·sin az, cos ze), ângulos em rad.
        # Linha 1 do CSV: azimute 268,99°, zênite 37,838°. Tolerância 1e-6 (float32 de sobra).
        az_deg, ze_deg = _primeira_linha_do_csv()
        self.assertAlmostEqual(az_deg, 268.99, delta=0.005)  # confere o CSV que o teste lê
        self.assertAlmostEqual(ze_deg, 37.838, delta=0.0005)
        az, ze = math.radians(az_deg), math.radians(ze_deg)
        esperado = np.array([math.sin(ze) * math.cos(az),
                             math.sin(ze) * math.sin(az),
                             math.cos(ze)])
        np.testing.assert_allclose(self.dirs[0, 0].astype(np.float64), esperado, rtol=0, atol=1e-6)

    def test_cinco_primeiros_trechos_cobrem_300_graus(self):
        # Cobertura em azimute = 360° − maior vão entre azimutes consecutivos (no círculo).
        # Se o padrão colapsa num setor, o vão passa de 60° e o teste cai.
        xy = self.dirs[:5].reshape(-1, 3).astype(np.float64)
        az = np.sort(np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) % 360.0)
        vaos = np.diff(az)
        vao_fechamento = az[0] + 360.0 - az[-1]
        cobertura = 360.0 - max(vaos.max(), vao_fechamento)
        self.assertGreaterEqual(cobertura, 300.0)


if __name__ == "__main__":
    unittest.main()
