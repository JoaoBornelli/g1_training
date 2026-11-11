"""Testes da fase H6 de g1_v3/runner.py (plano §12.4 H6; S §14; §6.7).

O rsl_rl grava só `Policy/mean_std` (logger.py:211). `RunnerV3` embrulha `logger.log` e grava `Policy/std/<junta>`,
a média em 0 do `action_std`, um escalar por junta, pelos nomes da ação. O runner é montado sem `__init__`, com
logger, env e writer falsos. Escritos antes do código: `_embrulha_log` ainda não existe (AttributeError).
"""
import types
import unittest
from unittest import mock

import torch
from mjlab.rl import MjlabOnPolicyRunner

from g1_v3 import knobs
from g1_v3.runner import RunnerV3


class LoggerFalso:
    def __init__(self, writer):
        self.writer = writer
        self.chamadas = []

    def log(self, **kw):  # o original do rsl_rl, só para registrar a chamada
        self.chamadas.append(kw)


class WriterFalso:
    def __init__(self):
        self.escalares = []

    def add_scalar(self, tag, valor, passo):
        self.escalares.append((tag, valor, passo))


def runner_falso(writer):
    obj = object.__new__(RunnerV3)
    obj.logger = LoggerFalso(writer)
    termo = types.SimpleNamespace(target_names=list(knobs.JUNTAS))
    am = types.SimpleNamespace(get_term=mock.Mock(return_value=termo))
    obj.env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(action_manager=am))
    return obj


class RunnerHTest(unittest.TestCase):

    def test_init_chama_o_pai_e_depois_embrulha_o_log(self):
        with mock.patch.object(MjlabOnPolicyRunner, "__init__", return_value=None) as pai, \
                mock.patch.object(RunnerV3, "_embrulha_log") as embrulha, \
                mock.patch.object(RunnerV3, "_fixa_obs") as fixa:
            RunnerV3(1, key=2)
        pai.assert_called_once_with(1, key=2)  # o mock não é método: sem o self
        embrulha.assert_called_once_with()
        fixa.assert_called_once_with()

    def _runner_com_obs(self, dim_ator=None):
        # ator: giro(15) tarefa(30) caixa(55) mapa(882); crítico: giro(15) tarefa(30) niveis(4) tarefa_critico(6)
        om = types.SimpleNamespace(
            active_terms={"actor": ["giro_base", "tarefa", "caixa", "mapa"],
                          "critic": ["giro_base", "tarefa", "niveis", "tarefa_critico"]},
            group_obs_term_dim={"actor": [(15,), (30,), (55,), (21, 42)], "critic": [(15,), (30,), (4,), (6,)]})
        nz = lambda d: types.SimpleNamespace(_mean=torch.zeros(1, d))
        obj = object.__new__(RunnerV3)
        obj.env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(observation_manager=om))
        obj.alg = types.SimpleNamespace(actor=types.SimpleNamespace(obs_normalizer=nz(dim_ator or 982)),
                                        critic=types.SimpleNamespace(obs_normalizer=torch.nn.Identity()))
        return obj

    def test_fixa_obs_acha_as_fatias_pelo_nome_do_termo(self):
        obj = self._runner_com_obs()
        obj._fixa_obs()
        self.assertEqual(obj.alg.fatias_fixas, {"actor": [slice(15, 45), slice(100, 982)],
                                                "critic": [slice(15, 45), slice(45, 49), slice(49, 55)]})

    def test_fixa_obs_falha_se_a_entrada_da_rede_nao_e_a_soma_dos_termos(self):
        with self.assertRaises(AssertionError):
            self._runner_com_obs(dim_ator=981)._fixa_obs()

    def test_embrulha_chama_o_original_com_os_mesmos_argumentos(self):
        obj = runner_falso(WriterFalso())
        obj._embrulha_log()
        std = torch.rand(3, len(knobs.JUNTAS))
        obj.logger.log(it=7, action_std=std, loss_dict={})
        self.assertEqual(len(obj.logger.chamadas), 1)
        self.assertEqual(obj.logger.chamadas[0]["it"], 7)
        self.assertIs(obj.logger.chamadas[0]["action_std"], std)

    def test_grava_std_por_junta_com_a_media_da_coluna_no_it(self):
        writer = WriterFalso()
        obj = runner_falso(writer)
        obj._embrulha_log()
        std = torch.rand(3, len(knobs.JUNTAS))
        obj.logger.log(it=7, action_std=std, loss_dict={})
        self.assertEqual(len(writer.escalares), len(knobs.JUNTAS))
        self.assertEqual([t for t, _, _ in writer.escalares], [f"Policy/std/{j}" for j in knobs.JUNTAS])
        self.assertTrue(all(p == 7 for _, _, p in writer.escalares))
        esperado = std.mean(dim=0).tolist()
        self.assertTrue(all(abs(v - e) < 1e-6 for (_, v, _), e in zip(writer.escalares, esperado)))

    def test_sem_writer_nao_grava_nada_e_chama_o_original(self):
        obj = runner_falso(None)
        obj._embrulha_log()
        obj.logger.log(it=7, action_std=torch.rand(3, len(knobs.JUNTAS)), loss_dict={})
        self.assertEqual(len(obj.logger.chamadas), 1)


if __name__ == "__main__":
    unittest.main()
