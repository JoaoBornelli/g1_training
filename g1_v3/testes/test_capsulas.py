"""Testes das cápsulas de colisão do corpo (A5 do plano). Escritos antes do código (lane de testes, A6).

As cápsulas vêm das geoms de colisão do g1.xml do mjlab (PERC §7). O XML é lido com
xml.etree (stdlib): o teste não importa mjlab nem mujoco.
"""
import importlib.util
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DADOS = Path(__file__).resolve().parent.parent / "dados"


def _nomes_da_arvore_do_pelvis():
    """Nomes do corpo `pelvis` e de todos os seus descendentes (o próprio pelvis incluído)."""
    spec = importlib.util.find_spec("mjlab")  # não executa o pacote, só acha a pasta
    assert spec is not None and spec.submodule_search_locations, "mjlab não instalado"
    xml = Path(list(spec.submodule_search_locations)[0]) / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
    assert xml.exists(), f"g1.xml ausente: {xml}"
    raiz = ET.parse(xml).getroot()
    pelvis = [b for b in raiz.iter("body") if b.get("name") == "pelvis"]
    assert len(pelvis) == 1, "o g1.xml deveria ter um único corpo pelvis"
    return {b.get("name") for b in pelvis[0].iter("body")}


class TestCapsulas(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        d = np.load(DADOS / "capsulas.npz", allow_pickle=False)
        cls.corpo = d["corpo"]
        cls.raio = d["raio"]
        cls.braco = d["braco"]

    def test_quantidade(self):
        # Mais de 20 cápsulas: pernas, tronco, braços e mãos (PERC §7).
        self.assertGreater(len(self.raio), 20)

    def test_raios_na_faixa(self):
        # Entre 5 mm (geom fina) e 25 cm (a maior envolvente do corpo). Intervalo aberto.
        self.assertTrue(np.all(self.raio > 0.005), self.raio.min())
        self.assertTrue(np.all(self.raio < 0.25), self.raio.max())

    def test_braco_nos_dois_lados(self):
        # O filtro do corpo precisa ver os dois braços (S §13, oclusão).
        nomes_braco = [str(c) for c, b in zip(self.corpo, self.braco) if b]
        self.assertTrue(any("left" in n for n in nomes_braco), nomes_braco)
        self.assertTrue(any("right" in n for n in nomes_braco), nomes_braco)

    def test_nenhum_corpo_fora_da_arvore_do_pelvis(self):
        # Todo corpo com cápsula é descendente do pelvis (a raiz do robô, I §2.2).
        arvore = _nomes_da_arvore_do_pelvis()
        fora = sorted({str(c) for c in self.corpo} - arvore)
        self.assertEqual(fora, [])


if __name__ == "__main__":
    unittest.main()
