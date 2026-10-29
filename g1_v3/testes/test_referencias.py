"""Testes da tabela de referências macro (A3 do plano). Escritos antes do código (lane de testes, A6).

Os números são os do plano (A6). Os índices de grade são achados pelo valor (topo, meia nas
chaves `topos` e `meias`), e não pela posição, para o teste dizer o que mede.
"""
import unittest
from pathlib import Path

import numpy as np

from g1_v3.knobs import q_de_pe

DADOS = Path(__file__).resolve().parent.parent / "dados"
# Pose de pé do g1_limpo: a fonte de q_de_pe (I §2.2) e da pelve de pé.
REF_DE_PE = Path(__file__).resolve().parents[2] / "g1_limpo" / "ik" / "ref_v3_de_pe.npz"

# Chaves da §8 A3 (as 18 do script gera_referencias).
CHAVES = (
    "topos", "meias", "grade_pelve", "grade_tronco", "grade_recuo", "grade_aceita",
    "caixa_transporte_b", "palmas_com_caixa_b", "q_com_caixa", "pelve_com_caixa",
    "q_de_pe", "palmas_de_pe_b", "pes_de_pe_b", "cantos_sola", "pelve_de_pe",
    "site_pe_de_pe", "fonte", "hash_origem",
)


def _indice(valores, alvo):
    """Posição do nó da grade com esse valor. Exige exatamente um nó."""
    idx = np.flatnonzero(np.isclose(valores, alvo, atol=1e-9))
    assert len(idx) == 1, f"{alvo} não está na grade {valores}"
    return int(idx[0])


class TestReferencias(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.d = np.load(DADOS / "referencias.npz", allow_pickle=False)

    def test_chaves_existem(self):
        faltam = [c for c in CHAVES if c not in self.d.files]
        self.assertEqual(faltam, [])

    def test_grade_pelve(self):
        # Grade (topo, meia): o chão é o topo 0.
        g = self.d["grade_pelve"]
        i, j = _indice(self.d["topos"], 0.0), _indice(self.d["meias"], 0.07)
        self.assertAlmostEqual(g[i, j], 0.249, delta=0.001)  # chão, meia 0,07
        i, j = _indice(self.d["topos"], 0.55), _indice(self.d["meias"], 0.13)
        self.assertAlmostEqual(g[i, j], 0.705, delta=0.001)  # topo 0,55, meia 0,13

    def test_grade_tronco_em_graus(self):
        # A grade está em rad; a conversão para graus cai no número da tabela de origem (51,2°).
        g = self.d["grade_tronco"]
        i, j = _indice(self.d["topos"], 0.30), _indice(self.d["meias"], 0.10)
        self.assertAlmostEqual(np.degrees(g[i, j]), 51.2, delta=0.1)

    def test_grade_recuo(self):
        # Recuo da pelve em relação à caixa, no nó (0,30; 0,10): 0,42 a 0,46 (plano, A6).
        g = self.d["grade_recuo"]
        i, j = _indice(self.d["topos"], 0.30), _indice(self.d["meias"], 0.10)
        self.assertGreaterEqual(g[i, j], 0.42)
        self.assertLessEqual(g[i, j], 0.46)

    def test_pelve_de_pe(self):
        # Pelve de pé: 0,789 m, igual a qpos[0, 2] de ref_v3_de_pe.
        self.assertAlmostEqual(float(self.d["pelve_de_pe"]), 0.789, delta=0.001)

    def test_palmas_de_pe_simetricas(self):
        # Linhas: esquerda (E) e direita (D); colunas x, y, z. Os y das palmas são simétricos a 2 mm.
        # ⚠ A A6 pedia 1 mm, mas a pose de pé da IK tem waist_yaw = −0,005 rad: o giro leva as
        # duas palmas ~0,75 mm para −y (soma −1,52 mm). Na pose zero o g1.xml dá 0,01 mm. O limite
        # ainda pega frame trocado ou palma espelhada, que erram em centímetros (orquestrador, 29/10).
        p = self.d["palmas_de_pe_b"]
        self.assertEqual(p.shape, (2, 3))
        self.assertLess(abs(p[0, 1] + p[1, 1]), 0.002)

    def test_caixa_transporte_b_20cm(self):
        # Caixa de 20 cm = meia aresta 0,10 (I §2.5). Linha pela chave `meias`.
        c = self.d["caixa_transporte_b"]
        self.assertEqual(c.shape, (3, 3))
        i = _indice(self.d["meias"], 0.10)
        self.assertAlmostEqual(c[i, 0], 0.25, delta=0.03)  # x
        self.assertAlmostEqual(c[i, 2], 0.11, delta=0.01)  # z

    def test_q_de_pe_igual_a_pose_zero_do_arquivo(self):
        # Exato em 1e-12: o driver usa o valor do arquivo, sem arredondar (I §2.2).
        ref = np.load(REF_DE_PE, allow_pickle=False)["qpos"][0, 7:36]
        np.testing.assert_allclose(self.d["q_de_pe"], ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(q_de_pe(), ref, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
