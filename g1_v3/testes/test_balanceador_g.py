"""Conferência da G2 (fase G; §6.3 "Fase G", PD-20, S §11.2): as regras do terreno do CARREGAR já estão no código
da fase D (`balanceador.py:143-151`, `eventos.escolhe_celula_terreno`). Estes testes só as provam; não mudam código.

Os estados são postos à mão (`_nivel`, `s`, `aberta`), como nos outros testes do balanceador. `SEM_CARENCIA`
trava as subidas por conta própria, para o nível que o teste põe ficar o que ele põe.
"""
import dataclasses
import unittest

import torch

from g1_v3 import balanceador
from g1_v3.eventos import escolhe_celula_terreno
from g1_v3.knobs import ATIVO

B = ATIVO.balanceador
N = ATIVO.niveis
COM = ATIVO.comando
TER = ATIVO.terreno
SEM_CARENCIA = dataclasses.replace(B, iters_min_nivel=10 ** 6)  # nenhum nível sobe sozinho
SEM_ESPERA = dataclasses.replace(B, iters_min_nivel=0)  # a linha de terreno pode subir a cada fecho
MAX_CARREGAR = balanceador.Curriculo(SEM_CARENCIA, N, COM, TER).max_nivel["CARREGAR"]
TETO_LINHA = max(v for _n, v in TER.linha_max_carregar)  # 9 (PD-20)
ANDAR, FATIA_B = 0, 1


def novo(k_bal=SEM_CARENCIA):
    return balanceador.Curriculo(k_bal, N, COM, TER)


def cel(cur, tarefa, tipo):
    return next(c for c in cur.celulas if c.tarefa == tarefa and c.tipo == tipo)


def prepara_carregar_no_portao(cur):
    # as células de velocidade do CARREGAR abertas e no portão; a célula `terreno` do ANDAR passada (s ≥ portão)
    for tp in balanceador.VELOCIDADE["CARREGAR"]:
        c = cel(cur, "CARREGAR", tp)
        c.aberta, c.s = True, 1.0
    cel(cur, "ANDAR", "terreno").s = 1.0


class TestCelulaTerrenoDoCarregar(unittest.TestCase):

    def test_terreno_do_carregar_so_abre_no_ultimo_nivel_de_velocidade(self):
        # §6.3 (regra 1): antes do último nível do CARREGAR, a célula `terreno` fica fechada, mesmo com tudo
        # no portão e o terreno do ANDAR passado. Só no último nível ela abre.
        cur = novo()
        prepara_carregar_no_portao(cur)
        cur._nivel["CARREGAR"] = MAX_CARREGAR - 1
        cur.fecha_iteracao()
        self.assertFalse(cel(cur, "CARREGAR", "terreno").aberta)
        cur._nivel["CARREGAR"] = MAX_CARREGAR
        cur.fecha_iteracao()
        self.assertTrue(cel(cur, "CARREGAR", "terreno").aberta)

    def test_terreno_do_carregar_espera_o_terreno_do_andar_passar(self):
        # S §11.2: no último nível, a célula só abre com o terreno do ANDAR passado (s ≥ portão) ou com o
        # nível de terreno do ANDAR já em 1 ou mais. Sem nenhum dos dois, fica fechada.
        cur = novo()
        prepara_carregar_no_portao(cur)
        cur._nivel["CARREGAR"] = MAX_CARREGAR
        cel(cur, "ANDAR", "terreno").s = 0.0
        cur.fecha_iteracao()
        self.assertFalse(cel(cur, "CARREGAR", "terreno").aberta)
        cur._nivel_ter["ANDAR"] = 1  # a linha do ANDAR já subiu: o terreno do ANDAR conta como passado
        cur.fecha_iteracao()
        self.assertTrue(cel(cur, "CARREGAR", "terreno").aberta)

    def test_terreno_do_carregar_espera_as_abertas_no_portao(self):
        # §6.3 (regra 1): com o último nível e o ANDAR passado, uma célula de velocidade aberta abaixo do
        # portão segura a célula `terreno` do CARREGAR.
        cur = novo()
        prepara_carregar_no_portao(cur)
        cur._nivel["CARREGAR"] = MAX_CARREGAR
        cel(cur, "CARREGAR", "arco").s = 0.5
        cur.fecha_iteracao()
        self.assertFalse(cel(cur, "CARREGAR", "terreno").aberta)

    def test_linha_de_terreno_do_carregar_para_no_maximo_da_tabela(self):
        # §6.3 e PD-20 (regra 2): a linha do CARREGAR sobe até max(linha_max_carregar) e não passa dele,
        # mesmo com a célula sempre no portão e o nível de velocidade no máximo.
        cur = novo(SEM_ESPERA)
        prepara_carregar_no_portao(cur)
        cur._nivel["CARREGAR"] = MAX_CARREGAR
        c = cel(cur, "CARREGAR", "terreno")
        c.aberta, c.s = True, 1.0
        for _ in range(3 * TETO_LINHA + 5):
            cur.fecha_iteracao()
            self.assertLessEqual(cur.nivel_terreno("CARREGAR"), TETO_LINHA)
        self.assertEqual(cur.nivel_terreno("CARREGAR"), TETO_LINHA)


class TestSorteioDeTerrenoDaFatiaB(unittest.TestCase):

    def test_fatia_b_so_sorteia_coluna_com_plataforma_e_linha_ate_o_teto_da_coluna(self):
        # regra 3 (§6.3, PD-20, S §11.2): a fatia B sorteia só colunas de `linha_max_carregar` (com plataforma)
        # e a linha até o teto dessa coluna; nunca passa do nível de terreno corrente.
        nomes = list(TER.colunas)
        teto_da_coluna = torch.full((len(nomes),), -1, dtype=torch.long)  # −1 = sem plataforma
        for nome, lim in TER.linha_max_carregar:
            teto_da_coluna[nomes.index(nome)] = lim
        n = 20_000
        for nivel in (12, 3):
            g = torch.Generator().manual_seed(0)
            coluna, linha = escolhe_celula_terreno(
                torch.full((n,), FATIA_B, dtype=torch.long), torch.ones(n, dtype=torch.bool),
                torch.full((n,), nivel, dtype=torch.long), TER, g)
            teto = teto_da_coluna[coluna]
            self.assertTrue((teto >= 0).all(), "a fatia B sorteou coluna sem plataforma")
            self.assertTrue((linha <= teto).all(), "a linha passou do teto da coluna")
            self.assertTrue((linha <= nivel).all(), "a linha passou do nível de terreno")


if __name__ == "__main__":
    unittest.main()
