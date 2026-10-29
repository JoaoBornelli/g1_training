"""Testes da terminação de chegada da caixa no BOTAR, `terminacoes.n_chegada` (F3 do plano; §6.3 "Fase F: núcleos
puros (29/10)", §12.2 F3; S §10; D3 da Parte 0). Escritos antes do código: o AttributeError de `n_chegada` é
esperado até a lane F3.

Convenções: tensores sintéticos, E = 1, sem env. `vz_ant` é a velocidade vertical do passo anterior (m/s, negativa
na descida, `v3_vz_caixa_ant`). `limite` é o do degrau 0 de `Terminacao.chegada` (início = 1,0 m/s), que é o valor
que `recompensas.por_degrau` devolve no degrau 0.
"""
import unittest

import torch

from g1_v3 import terminacoes
from g1_v3.knobs import ATIVO

T = ATIVO.terminacao
LIMITE_DEGRAU_0 = T.chegada[0]  # 1,0 m/s: no degrau 0 o limite é o início (refino em 0,25 m/s)


def chega(primeiro_contato, vz_ant, limite=LIMITE_DEGRAU_0):
    return terminacoes.n_chegada(torch.tensor([primeiro_contato]), torch.tensor([float(vz_ant)]),
                                 torch.tensor([float(limite)])).item()


class Chegada(unittest.TestCase):
    def test_um_virgula_um_m_s_no_degrau_0_termina(self):
        # Teste (F3): 1,1 m/s no degrau 0 (limite 1,0) termina.
        self.assertTrue(chega(True, -1.1))

    def test_zero_virgula_nove_m_s_no_degrau_0_nao_termina(self):
        # Teste (F3): 0,9 m/s no degrau 0 não termina.
        self.assertFalse(chega(True, -0.9))

    def test_sem_primeiro_contato_nao_termina(self):
        # Teste (F3): sem primeiro contato caixa × laje, mesmo 1,1 m/s não termina.
        self.assertFalse(chega(False, -1.1))


if __name__ == "__main__":
    unittest.main()
