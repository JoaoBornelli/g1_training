"""Testes da lane H5 da fase H (`comando.py`; plano §6.3 "Fase H", §12.4 H5; PD-44, P22): a deriva da odometria pelo
degrau global (`deriva_do_degrau`), o `passo` da `v3_odo` no `compute` e o viés do alvo do BOTAR. Escritos antes do código.

Convenções: tensores sintéticos em CPU; o env é um `types.SimpleNamespace` e as partes que leem a cena ficam em `mock`.
"""
import types
import unittest
from unittest import mock

import torch

from g1_v3 import comando, eventos
from g1_v3.alvos import B1
from g1_v3.comando import CadeiaCmd
from g1_v3.knobs import ATIVO, Estimador

K = ATIVO.estimador
DT = ATIVO.contrato.decimacao * ATIVO.contrato.dt_fisica  # 0,02 s, passo da política


def _cadeia_sem_env(env):
    # `CadeiaCmd` sem o `__init__` (que pede a cena): só os atributos que os métodos lidos aqui usam
    obj = CadeiaCmd.__new__(CadeiaCmd)
    obj._env = env
    obj.pega = torch.zeros(1, dtype=torch.bool)
    obj.cfg = types.SimpleNamespace(est=Estimador())
    return obj


class TestDerivaDoDegrau(unittest.TestCase):

    def test_deriva_no_degrau_0_3_e_5(self):
        # §12.4 H5: g = 0 → início; g = 5 → refino; g = 3 → 60 % do caminho (PD-18, PD-41)
        for g, esperado in ((0, (0.002, 0.002, 0.004, 0.0035)),
                            (5, (0.0075, 0.005, 0.008, 0.014)),
                            (3, (0.0053, 0.0038, 0.0064, 0.0098))):
            d = comando.deriva_do_degrau(K, g, ATIVO.balanceador.degraus)
            self.assertEqual(set(d), {"odo", "rumo", "pos_giro", "rumo_giro"})
            for chave, valor in zip(("odo", "rumo", "pos_giro", "rumo_giro"), esperado):
                self.assertAlmostEqual(d[chave], valor, places=6, msg=f"g={g} {chave}")


class TestComputeDaOPassoDaOdometria(unittest.TestCase):

    def test_compute_passa_a_deriva_do_degrau_global_uma_vez_com_a_pose_da_raiz(self):
        # PD-44 e §6.4: um `v3_odo.passo` por `compute`, com a pose da raiz e a deriva de g = 3 (o degrau global)
        env = types.SimpleNamespace(
            v3_trecho=torch.zeros(1, dtype=torch.long), v3_aviso=torch.zeros(1, dtype=torch.bool),
            v3_fim=torch.zeros(1, dtype=torch.bool), v3_perdeu=torch.zeros(1, dtype=torch.bool),
            v3_tarefa=torch.zeros(1, dtype=torch.long), v3_p4_fechado=torch.zeros(1, dtype=torch.bool),
            v3_b4_fechado=torch.zeros(1, dtype=torch.bool), v3_vz_caixa_ant=torch.zeros(1), v3_odo=mock.Mock(),
            v3_cur=types.SimpleNamespace(degrau_global=lambda: 3, restricao=lambda: 1.0),
            scene={"box": types.SimpleNamespace(data=types.SimpleNamespace(root_link_lin_vel_w=torch.zeros(1, 3)))})
        obj = _cadeia_sem_env(env)
        pos, quat = torch.tensor([[1.0, 2.0, 0.5]]), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        obj.robot = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=pos, root_link_quat_w=quat))
        obj.tab = types.SimpleNamespace(por_detector=torch.zeros(1, dtype=torch.bool))  # só o latch do aviso lê a tabela
        obj.relogio = torch.zeros(1)
        obj.alvo_s = torch.zeros(1)
        obj.bota = torch.zeros(1, dtype=torch.bool)
        ordem = []
        env.v3_odo.passo.side_effect = lambda *a: ordem.append("odo")
        obj._mede = mock.Mock(return_value={})
        obj._perda = mock.Mock()
        obj._abre_p0 = mock.Mock()
        obj._alvos = mock.Mock()
        obj._grava_chegada = mock.Mock(side_effect=lambda: ordem.append("chegada"))
        nao_avancou = (torch.zeros(1, dtype=torch.long), torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                       torch.zeros(1, dtype=torch.bool))
        with mock.patch.object(comando, "avanca_trecho", return_value=nao_avancou):
            obj.compute(DT)
        env.v3_odo.passo.assert_called_once()
        args = env.v3_odo.passo.call_args[0]
        self.assertIs(args[0], pos)
        self.assertIs(args[1], quat)
        self.assertEqual(args[2].keys(), {"odo", "rumo", "pos_giro", "rumo_giro"})
        self.assertAlmostEqual(args[2]["odo"], 0.0053, places=6)
        self.assertAlmostEqual(args[2]["rumo"], 0.0038, places=6)
        self.assertAlmostEqual(args[2]["pos_giro"], 0.0064, places=6)
        self.assertAlmostEqual(args[2]["rumo_giro"], 0.0098, places=6)
        self.assertEqual(ordem, ["odo", "chegada"])  # a odometria anda antes das escritas do fim


class TestAbreBotarComViesDoAlvo(unittest.TestCase):
    # P22: com gerador, o alvo do BOTAR sai com viés U[±vies_alvo_botar] por eixo (x, y: xy; z: z), no degrau global

    N = 2000
    A = 0.10  # meia aresta (m) sobre a laje: o alvo sobe `a` (a caixa com o centro a meia aresta acima)

    def _abre(self, g, gerador):
        n = self.N
        z = torch.zeros(n)
        env = types.SimpleNamespace(
            v3_gerador=gerador, v3_cur=types.SimpleNamespace(degrau_global=lambda: g, restricao=lambda: 1.0),
            v3_alvo_w=torch.zeros(n, 3), v3_meia=torch.full((n,), self.A), v3_alvo_idx=torch.zeros(n, dtype=torch.long),
            v3_phi=z.clone(), v3_h=z.clone(), v3_phi_melhor=z.clone(), v3_relogio_alvo=z.clone(), v3_ponto_apoio=torch.zeros(n, 3), v3_v_chegada=z.clone(),
            v3_sigma=z.clone())
        obj = _cadeia_sem_env(env)
        obj.bota = torch.zeros(n, dtype=torch.bool)
        obj._sigma = lambda m, indice: torch.ones(n)
        with mock.patch.object(eventos, "posiciona_destino", return_value=torch.zeros(n, 3)):
            obj._abre_botar(torch.arange(n), {})
        self.assertEqual(env.v3_alvo_idx.tolist(), [B1] * n)  # o alvo corrente é o B1, com ou sem viés
        return env.v3_alvo_w - torch.tensor([0.0, 0.0, self.A])

    def test_vies_do_alvo_por_degrau_e_uniforme(self):
        # §12.4 H5: x e y com U[±vies_xy(g)], z com U[±vies_z(g)]; desvio de x = lim/√3 (5 cm → 0,0289 m)
        for g, lim_xy, lim_z in ((5, 0.05, 0.03), (3, 0.034, 0.02), (0, 0.01, 0.005)):
            vies = self._abre(g, torch.Generator().manual_seed(0))
            self.assertLessEqual(float(vies[:, 0].abs().max()), lim_xy + 1e-6, msg=f"g={g} x")
            self.assertLessEqual(float(vies[:, 1].abs().max()), lim_xy + 1e-6, msg=f"g={g} y")
            self.assertLessEqual(float(vies[:, 2].abs().max()), lim_z + 1e-6, msg=f"g={g} z")
            self.assertAlmostEqual(float(vies[:, 0].std()), lim_xy / 3 ** 0.5, delta=0.002, msg=f"g={g}")
            self.assertAlmostEqual(float(vies[:, 2].std()), lim_z / 3 ** 0.5, delta=0.002, msg=f"g={g}")

    def test_sem_gerador_o_alvo_do_botar_e_so_a_soltura_mais_a(self):
        alvo = self._abre(5, None)
        self.assertTrue(torch.allclose(alvo, torch.zeros(self.N, 3)))


if __name__ == "__main__":
    unittest.main()
