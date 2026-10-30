"""Testes do núcleo n_caixa_perdida de g1_v3/terminacoes.py (E5 do plano; §6.3 "Fase E: núcleos puros (29/10)",
§12.1; S §10; PD-27). Escritos antes do código: o ImportError é esperado até a lane E5.

A chave `caixa_perdida_vira_andar` (Cadeia, PD-27) entra como argumento `vira_andar`: ligada, a perda vira ANDAR e
não termina o episódio; desligada, a perda termina no passo em que acontece. Tensores sintéticos, sem env.
"""
import unittest

import torch

from g1_v3 import terminacoes


class TestNCaixaPerdida(unittest.TestCase):

    def test_com_a_chave_ligada_nunca_termina(self):
        # E5, linha "Teste": com a chave ligada, a caixa perdida nunca termina o episódio
        perdeu = torch.tensor([False, True, True, False])
        fim = terminacoes.n_caixa_perdida(perdeu, vira_andar=True)
        self.assertEqual(fim.dtype, torch.bool)
        self.assertEqual(fim.shape, perdeu.shape)
        self.assertFalse(bool(fim.any()))

    def test_com_a_chave_desligada_termina_no_passo_da_perda(self):
        # E5, linha "Teste": com a chave desligada, termina no passo em que o sinal de perda sobe (um passo depois da perda física, caso 4)
        perdeu = torch.tensor([False, False, True, False])
        fim = terminacoes.n_caixa_perdida(perdeu, vira_andar=False)
        self.assertEqual(fim.tolist(), [False, False, True, False])

    def test_com_a_chave_desligada_so_termina_o_env_que_perdeu(self):
        # E5, §6.3: `perdeu_agora & ~vira_andar` é por env. Com três envs, só o que perdeu termina.
        perdeu = torch.tensor([True, False, False])
        fim = terminacoes.n_caixa_perdida(perdeu, vira_andar=False)
        self.assertEqual(fim.tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
