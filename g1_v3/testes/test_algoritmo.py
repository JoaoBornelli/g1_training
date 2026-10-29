"""Testes de g1_v3/algoritmo.py, núcleo normaliza_por_grupo (D9 do plano, §11; g1_limpo/algoritmo.py:155-224).
Escritos antes do código.

Convenções: `grupo` é o índice de TAREFAS (§6.2; ANDAR = 0), o argmax das 6 últimas colunas do crítico
(D9). `adv` e `grupo` são (N,) e a saída é (N,). O desvio é o `std()` do torch (sem correção de Bessel
unbiased, o padrão do torch), como no g1_limpo (`a.std()`); por isso a saída tem `std()` = 1.

Grupo pequeno (< 2 amostras), como no g1_limpo: o ANDAR fica com a vantagem CRUA; os outros grupos
são normalizados pela "manipulação inteira" (todas as amostras não-ANDAR do lote, o próprio grupo
ralo incluído), que é o fallback que evita o NaN do std de uma amostra. Os testes repetem essa conta.
"""
import unittest

import torch

from g1_v3.algoritmo import normaliza_por_grupo

ANDAR, PEGAR, CARREGAR = 0, 1, 2


class TestNormalizaPorGrupo(unittest.TestCase):
    def test_tres_grupos_media_zero_e_desvio_um(self):
        # três grupos com escalas e médias diferentes: cada um sai com média 0 e desvio 1
        g = torch.Generator().manual_seed(0)
        blocos = [(ANDAR, 40, 0.0, 0.5), (PEGAR, 30, 3.0, 2.0), (CARREGAR, 20, -1.0, 10.0)]
        adv = torch.cat([torch.randn(n, generator=g) * esc + loc for _, n, loc, esc in blocos])
        grupo = torch.cat([torch.full((n,), t, dtype=torch.long) for t, n, _, _ in blocos])
        out = normaliza_por_grupo(adv, grupo)
        self.assertEqual(out.shape, adv.shape)
        for t, _, _, _ in blocos:
            m = out[grupo == t]
            self.assertAlmostEqual(float(m.mean()), 0.0, places=4)
            self.assertAlmostEqual(float(m.std()), 1.0, places=4)

    def test_grupo_de_uma_amostra_nao_andar_usa_a_manipulacao_inteira(self):
        # CARREGAR com uma amostra só (7,0). O fallback é a média e o desvio de todas as não-ANDAR
        # do lote (2,0; 4,0; 7,0), e o valor sai normalizado por eles, sem NaN (g1_limpo:`n < 2`).
        adv = torch.tensor([0.5, -0.5, 1.0, 2.0, 4.0, 7.0])
        grupo = torch.tensor([ANDAR, ANDAR, ANDAR, PEGAR, PEGAR, CARREGAR])
        out = normaliza_por_grupo(adv, grupo)
        self.assertTrue(torch.isfinite(out).all())
        manip = adv[grupo != ANDAR]
        esperado = (7.0 - manip.mean()) / (manip.std() + 1e-8)
        self.assertAlmostEqual(float(out[5]), float(esperado), places=5)
        # o PEGAR, com duas amostras, segue a regra do grupo normal (média 0)
        self.assertAlmostEqual(float(out[3] + out[4]), 0.0, places=5)

    def test_grupo_de_uma_amostra_andar_fica_com_a_vantagem_crua(self):
        # o ANDAR é o único grupo que não usa o fallback: a amostra sozinha sai como veio (0,8)
        adv = torch.tensor([0.8, 1.0, 2.0, 3.0])
        grupo = torch.tensor([ANDAR, PEGAR, PEGAR, PEGAR])
        out = normaliza_por_grupo(adv, grupo)
        self.assertAlmostEqual(float(out[0]), 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
