"""Testes do núcleo puro escolhe_celula_terreno de g1_v3/eventos.py (D7 do plano, §6.3 "eventos.py (núcleo
puro; 29/10)"; S §11.2, §12; PD-20). Escritos antes do código.

Convenções: fatia é índice em FATIAS (0 ANDAR, 1 B, 2 C); a tabela de colunas e os tetos de CARREGAR
vêm de `ATIVO.terreno` (`colunas`, `linha_max_carregar` por nome). O gerador é fixo (semente 0) e E = 20 000
amostras: com esse tamanho, cada coluna recebe ~2 000 sorteios, o bastante para que o máximo de uma
faixa uniforme chegue ao teto com folga. Os casos seguem a linha "Teste" da D7.
"""
import unittest

import torch

from g1_v3.eventos import escolhe_celula_terreno
from g1_v3.knobs import ATIVO

T = ATIVO.terreno
NOMES = list(T.colunas)
IRREGULAR = NOMES.index("irregular")  # coluna 6 (rugosidade aleatória, sem plataforma, só ANDAR)
LIMITE = {NOMES.index(nome): linha for nome, linha in T.linha_max_carregar}  # coluna -> linha máxima CARREGAR
ANDAR, B, C = 0, 1, 2
E = 20_000
NIVEL_MAX = 12  # linhas 0..12 do gerador (Terreno.linhas = 13, PD-20)


def sorteia(fatia, terreno, nivel, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.full((E,), fatia, dtype=torch.long)
    t = torch.full((E,), terreno, dtype=torch.bool)
    n = torch.full((E,), nivel, dtype=torch.long)
    return escolhe_celula_terreno(f, t, n, T, g)


class TestEscolheCelulaTerreno(unittest.TestCase):
    def test_fatia_b_nunca_cai_no_irregular(self):
        # S §11.2: o `irregular` não tem plataforma, então a fatia B (que pede plataforma) nunca o sorteia
        coluna, _ = sorteia(B, True, NIVEL_MAX)
        self.assertFalse((coluna == IRREGULAR).any())
        self.assertTrue(set(coluna.tolist()) <= set(LIMITE))

    def test_carregar_nivel_12_respeita_o_teto_de_cada_coluna(self):
        # PD-20: escadas e rampas ≤ 9 (degrau de 0,15 m), obstáculos ≤ 8, caixas ≤ 7
        coluna, linha = sorteia(B, True, NIVEL_MAX)
        for c, teto in LIMITE.items():
            sel = linha[coluna == c]
            self.assertGreater(sel.numel(), 0, NOMES[c])
            self.assertLessEqual(int(sel.max()), teto, NOMES[c])
            self.assertEqual(int(sel.max()), teto, f"{NOMES[c]} não chegou ao teto {teto}")

    def test_andar_nivel_12_chega_a_linha_12(self):
        # ANDAR de terreno: linha uniforme em 0..nível; o máximo sorteado é o nível 12 (S §11.2)
        coluna, linha = sorteia(ANDAR, True, NIVEL_MAX)
        self.assertEqual(int(linha.max()), NIVEL_MAX)
        self.assertGreaterEqual(int(coluna.min()), 1)  # coluna 0 (plano) só no episódio sem terreno
        self.assertLessEqual(int(coluna.max()), len(NOMES) - 1)

    def test_andar_no_plano_fica_na_coluna_0(self):
        # sem terreno no episódio: coluna 0 e linha 0, mesmo com nível 12 (D7, `flat`)
        coluna, linha = sorteia(ANDAR, False, NIVEL_MAX)
        self.assertTrue((coluna == 0).all())
        self.assertTrue((linha == 0).all())

    def test_fatia_c_sempre_zero(self):
        # a fatia C não tem terreno (S §7): (0, 0) qualquer que seja o nível e o episódio
        coluna, linha = sorteia(C, True, NIVEL_MAX)
        self.assertTrue((coluna == 0).all())
        self.assertTrue((linha == 0).all())


if __name__ == "__main__":
    unittest.main()
