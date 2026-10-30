"""Testes da lane H5b da fase H (`observacoes.py`; plano §6.3 "Fase H", §6.5, §12.4 H5b; P7).

`_magnitude` com o degrau global (g = 0, 3 e 5 de 5), o canal `forca_caixa` do crítico e a magnitude que o
`_calcula_caixa` passa ao `EstimadorCaixa.passo`. Env falso com `types.SimpleNamespace`, como os vizinhos.
"""
import types
import unittest
from unittest import mock

import torch

from g1_v3 import observacoes
from g1_v3.knobs import ATIVO, TAREFAS

K = ATIVO
DEGRAUS = K.balanceador.degraus  # 5
ANDAR = TAREFAS.index("ANDAR")


class MagnitudeTest(unittest.TestCase):
    def test_degrau_zero_e_o_inicio(self):
        m = observacoes._magnitude(K.estimador, 0, DEGRAUS)
        esperado = {"sigma_pos": 0.005, "sigma_pos_rel": 0.0, "ori_mediana_graus": 1.5, "ori_p90_graus": 2.5,
                    "ori_leitura_graus": 0.0, "perda": 0.005, "vies_meia": 0.005, "vies_pos": 0.0,
                    "rajada_taxa": 0.0}
        self._confere(m, esperado)

    def test_degrau_tres_de_cinco(self):
        # §12.4 H5b, com a conta do `balanceador.no_degrau`: (ini + (fim − ini)·3/5)
        m = observacoes._magnitude(K.estimador, 3, DEGRAUS)
        esperado = {"sigma_pos": 0.002, "sigma_pos_rel": 0.012, "ori_mediana_graus": 3.78, "ori_p90_graus": 10.18,
                    "ori_leitura_graus": 0.6, "perda": 0.0152, "vies_meia": 0.014, "vies_pos": 0.012,
                    "rajada_taxa": 0.021}
        self._confere(m, esperado)

    def test_degrau_cinco_e_o_refino(self):
        m = observacoes._magnitude(K.estimador, 5, DEGRAUS)
        esperado = {"sigma_pos": 0.0, "sigma_pos_rel": 0.02, "ori_mediana_graus": 5.3, "ori_p90_graus": 15.3,
                    "ori_leitura_graus": 1.0, "perda": 0.022, "vies_meia": 0.02, "vies_pos": 0.02,
                    "rajada_taxa": 0.035}
        self._confere(m, esperado)

    def _confere(self, m, esperado):
        self.assertEqual(set(m), set(esperado))  # sem `sigma_ori_graus` nem `ori_persistente` (P21b)
        for chave, valor in esperado.items():
            self.assertAlmostEqual(m[chave], valor, places=6, msg=chave)


def env_caixa_robo(forca):
    # `forca` (E, 12, 3): o sensor `caixa_robo` com as 12 fatias de força (N)
    caixa = types.SimpleNamespace(data=types.SimpleNamespace(force=forca))
    return types.SimpleNamespace(num_envs=forca.shape[0], device="cpu", scene={"caixa_robo": caixa})


class ForcaCaixaTest(unittest.TestCase):
    def test_duas_fatias_opostas_somam_os_modulos(self):
        # P7: as duas palmas apertam em sentidos opostos; a soma dos módulos é 235 N → 2,35 na escala de 100 N
        f = torch.zeros(2, 12, 3)
        f[0, 0, 1], f[0, 1, 1] = 117.5, -117.5
        out = observacoes.forca_caixa(env_caixa_robo(f))
        self.assertEqual(tuple(out.shape), (2, 1))
        self.assertAlmostEqual(out[0, 0].item(), 2.35, places=5)
        self.assertEqual(out[1, 0].item(), 0.0)


class CalculaCaixaComDegrauTest(unittest.TestCase):
    def test_estimador_recebe_a_magnitude_do_degrau_global(self):
        # H5b: a magnitude do `passo` sai do degrau global do curriculo, e não mais do início fixo
        robo = mock.MagicMock()
        robo.find_sites.return_value = ([0], None)
        robo.data.site_pos_w = torch.zeros(1, 1, 3)
        robo.data.site_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        robo.data.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        caixa = types.SimpleNamespace(data=types.SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.5]]), root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
        estimador = mock.MagicMock()
        estimador.saida_b.return_value = torch.zeros(1, 8)
        env = types.SimpleNamespace(scene={"robot": robo, "box": caixa}, common_step_counter=3, step_dt=0.02,
                                    v3_meia=torch.tensor([0.1]), v3_odo=None, v3_estimador=estimador,
                                    v3_tarefa=torch.tensor([ANDAR]), v3_alvo_idx=torch.tensor([-1]),
                                    v3_alvo_w=torch.zeros(1, 3), v3_aviso=torch.tensor([False]),
                                    v3_cur=types.SimpleNamespace(degrau_global=lambda: 3))
        with mock.patch.object(observacoes, "_capsulas_w", return_value=(torch.zeros(1, 33, 2, 3), torch.zeros(33))), \
                mock.patch.object(observacoes, "visivel", return_value=torch.ones(1, dtype=torch.bool)), \
                mock.patch.object(observacoes, "alvo_no_frame_da_base", return_value=torch.ones(1, 3)):
            observacoes._calcula_caixa(env)
        magnitude = estimador.passo.call_args.args[5]
        self.assertEqual(magnitude, observacoes._magnitude(K.estimador, 3, DEGRAUS))


if __name__ == "__main__":
    unittest.main()
