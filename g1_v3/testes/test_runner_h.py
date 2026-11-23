"""Testes da fase H6 de g1_v3/runner.py (plano §12.4 H6; S §14; §6.7).

O rsl_rl grava só `Policy/mean_std` (logger.py:211). `RunnerV3` embrulha `logger.log` e grava `Policy/std/<junta>`,
a média em 0 do `action_std`, um escalar por junta, pelos nomes da ação. O runner é montado sem `__init__`, com
logger, env e writer falsos. Escritos antes do código: `_embrulha_log` ainda não existe (AttributeError).
"""
import os
import tempfile
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

    def _runner_com_obs(self, dim_ator=None, com_alvo=False):
        # ator: giro(15) tarefa(30) caixa(55) mapa(882); crítico: giro(15) tarefa(30) niveis(4) tarefa_critico(6)
        # com `com_alvo`, o crítico ganha o alvo_macro(10) depois da tarefa
        critico = ["giro_base", "tarefa", "alvo_macro", "niveis", "tarefa_critico"] if com_alvo \
            else ["giro_base", "tarefa", "niveis", "tarefa_critico"]
        dims_critico = [(15,), (30,), (10,), (4,), (6,)] if com_alvo else [(15,), (30,), (4,), (6,)]
        om = types.SimpleNamespace(
            active_terms={"actor": ["giro_base", "tarefa", "caixa", "mapa"], "critic": critico},
            group_obs_term_dim={"actor": [(15,), (30,), (55,), (21, 42)], "critic": dims_critico})
        nz = lambda d: types.SimpleNamespace(_mean=torch.zeros(1, d))
        obj = object.__new__(RunnerV3)
        obj.env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(observation_manager=om))
        obj.alg = types.SimpleNamespace(actor=types.SimpleNamespace(obs_normalizer=nz(dim_ator or 982)),
                                        critic=types.SimpleNamespace(obs_normalizer=torch.nn.Identity()))
        return obj

    def test_fixa_obs_acha_as_fatias_pelo_nome_do_termo_com_o_desvio(self):
        obj = self._runner_com_obs()
        obj._fixa_obs()
        self.assertEqual(obj.alg.fatias_fixas, {
            "actor": [(slice(15, 45), 1.0), (slice(45, 100), 0.1), (slice(100, 982), 1.0)],
            "critic": [(slice(15, 45), 1.0), (slice(45, 49), 1.0), (slice(49, 55), 1.0)]})

    def test_fixa_obs_guarda_a_fatia_do_alvo_macro_no_critico(self):
        obj = self._runner_com_obs(com_alvo=True)
        obj._fixa_obs()
        self.assertEqual(obj.alg.fatia_alvo, slice(45, 55))
        self.assertEqual(obj.alg.fatias_fixas["critic"], [(slice(15, 45), 1.0), (slice(45, 55), 1.0),
                                                          (slice(55, 59), 1.0), (slice(59, 65), 1.0)])

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


class LoadComCriticoDeOutraFormaTest(unittest.TestCase):
    """23/11, T14: o `alvo_macro` do crítico passou de 13 para 11 canais. `RunnerV3.load` carrega só o ator e a iteração
    quando a forma do crítico do checkpoint difere da atual (o `load_cfg` do rsl_rl 5.4.0, `ppo.py:372`)."""

    def _load(self, entrada_do_checkpoint):
        # runner sem `__init__`; o crítico atual tem a 1ª camada com 1518 entradas, e o `load` do pai (o do mjlab) é um mock
        obj = object.__new__(RunnerV3)
        atual = {"mlp.0.weight": torch.zeros(4, 1518), "obs_normalizer._mean": torch.zeros(1, 1518)}
        obj.alg = types.SimpleNamespace(_raw_critic=types.SimpleNamespace(state_dict=lambda: atual))
        estado = {"it": 7}
        cur = mock.Mock()
        obj.env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(v3_cur=cur))
        ckpt = {"critic_state_dict": {"mlp.0.weight": torch.zeros(4, entrada_do_checkpoint),
                                      "obs_normalizer._mean": torch.zeros(1, entrada_do_checkpoint)},
                "actor_state_dict": {}, "iter": 7, "infos": {"v3_curriculo": estado}}
        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "model_7.pt")
            torch.save(ckpt, caminho)
            with mock.patch.object(MjlabOnPolicyRunner, "load", return_value={"v3_curriculo": estado}) as pai:
                obj.load(caminho)
        return pai, cur, estado

    def test_critico_de_13_canais_de_alvo_macro_carrega_so_o_ator_e_a_iteracao(self):
        pai, cur, estado = self._load(1520)  # 1518 + 2: o `alvo_macro` com 13 canais
        self.assertEqual(pai.call_args.args[1], {"actor": True, "critic": False, "optimizer": False, "iteration": True})
        cur.carrega.assert_called_once_with(estado)  # o currículo segue restaurado

    def test_critico_da_mesma_forma_carrega_tudo(self):
        pai, cur, _ = self._load(1518)
        self.assertIsNone(pai.call_args.args[1])  # sem `load_cfg`: o pai carrega o ator, o crítico, o otimizador e a iteração


if __name__ == "__main__":
    unittest.main()
