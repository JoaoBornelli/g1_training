"""Testes da marcha por relógio (13/11, M4): o portão de velocidade do termo `passo` e o relógio do `TwistV3`.
Env falso com `types.SimpleNamespace`, como os vizinhos. O `_update_metrics` do fabricante fica fora (mock).
"""
import types
import unittest
from unittest import mock

import torch

from g1_v3 import comando, recompensas
from g1_v3.knobs import ATIVO, TAREFAS

C = ATIVO.comando
DT = 0.02


def _env_passo(cmd, v_media, fase=0.0, contato=(True, True), perdeu=False):
    # fase 0: os dois pés em apoio no relógio; com os dois no chão o acerto é 2 e o termo vale 2 × o portão
    twist = types.SimpleNamespace(v_media_b=torch.tensor([v_media], dtype=torch.float32))
    return types.SimpleNamespace(
        device="cpu", v3_tarefa=torch.tensor([TAREFAS.index("ANDAR")]), v3_perdeu=torch.tensor([perdeu]),
        v3_fase_marcha=torch.tensor([fase]), v3_acerto_marcha=torch.zeros(1),
        scene={"pes_chao": types.SimpleNamespace(data=types.SimpleNamespace(found=torch.tensor([[int(c) for c in contato]])))},
        command_manager=types.SimpleNamespace(get_command=lambda _: torch.tensor([cmd], dtype=torch.float32),
                                              get_term=lambda _: twist))


def _passo(env):
    return float(recompensas.passo(env, ("ANDAR", "CARREGAR"))[0])


class PortaoDeVelocidade(unittest.TestCase):

    def test_velocidade_igual_ao_comando_da_o_termo_inteiro(self):
        self.assertAlmostEqual(_passo(_env_passo((0.5, 0.0, 0.0), (0.5, 0.0, 0.0))), 2.0, places=5)

    def test_parado_com_comando_da_zero(self):
        self.assertAlmostEqual(_passo(_env_passo((0.5, 0.0, 0.0), (0.0, 0.0, 0.0))), 0.0, places=5)

    def test_acima_do_comando_satura_em_1(self):
        self.assertAlmostEqual(_passo(_env_passo((0.5, 0.0, 0.0), (1.0, 0.0, 0.0))), 2.0, places=5)

    def test_metade_da_velocidade_paga_metade(self):
        self.assertAlmostEqual(_passo(_env_passo((0.5, 0.0, 0.0), (0.25, 0.0, 0.0))), 1.0, places=5)

    def test_andar_para_tras_do_comando_da_zero(self):
        self.assertAlmostEqual(_passo(_env_passo((0.5, 0.0, 0.0), (-0.3, 0.0, 0.0))), 0.0, places=5)

    def test_sem_comando_o_portao_vale_1(self):
        self.assertAlmostEqual(_passo(_env_passo((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))), 2.0, places=5)

    def test_giro_no_lugar_usa_o_sinal_do_comando(self):
        self.assertAlmostEqual(_passo(_env_passo((0.0, 0.0, -1.0), (0.0, 0.0, -1.0))), 2.0, places=5)
        self.assertAlmostEqual(_passo(_env_passo((0.0, 0.0, -1.0), (0.0, 0.0, 1.0))), 0.0, places=5)
        self.assertAlmostEqual(_passo(_env_passo((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))), 0.0, places=5)

    def test_grava_o_acerto_sem_o_portao_e_zera_com_a_caixa_perdida(self):
        env = _env_passo((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), contato=(True, False))  # fase 0: E apoio, D apoio
        self.assertAlmostEqual(_passo(env), 0.0, places=5)
        self.assertAlmostEqual(float(env.v3_acerto_marcha[0]), 0.5, places=5)
        self.assertAlmostEqual(_passo(_env_passo((0.5, 0.0, 0.0), (0.5, 0.0, 0.0), perdeu=True)), 0.0, places=5)


def _twist(cmd, fase=0.0, on=False, n=1):
    # `TwistV3` sem o `__init__`: só o que `_update_metrics` lê, com o fabricante e a âncora fora
    env = types.SimpleNamespace(
        step_dt=DT, v3_fase_marcha=torch.full((n,), fase), v3_marcha_on=torch.full((n,), on, dtype=torch.bool),
        v3_acerto_marcha=torch.ones(n), v3_t_empurrao=torch.full((n,), float("-inf")), v3_ancora=torch.zeros(n, 4),
        v3_incl_max_seg=torch.zeros(n),
        scene={"box": types.SimpleNamespace(data=types.SimpleNamespace(
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)))})
    obj = comando.TwistV3.__new__(comando.TwistV3)
    obj._env, obj.k = env, C
    obj.robot = types.SimpleNamespace(data=types.SimpleNamespace(
        root_link_pos_w=torch.zeros(n, 3), heading_w=torch.zeros(n), root_link_lin_vel_b=torch.zeros(n, 3),
        root_link_ang_vel_b=torch.zeros(n, 3)))
    obj.vel_command_b = torch.tensor([cmd], dtype=torch.float32).repeat(n, 1)
    obj._agora, obj.t_ab, obj.ancora_feita = torch.zeros(n), torch.zeros(n), torch.zeros(n, dtype=torch.bool)
    obj.command_counter, obj.seg_visto = torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)
    obj.seg_proj, obj.seg_int = torch.zeros(n, 2), torch.zeros(n, 2)
    obj.soma_erro_marcha, obj.soma_cmd_marcha = torch.zeros(n), torch.zeros(n)
    obj.soma_acerto, obj.passos_on, obj.pose_ult = torch.zeros(n), torch.zeros(n), torch.zeros(n, 3)
    obj.metrics = {"razao_marcha": torch.zeros(n), "acerto_relogio": torch.zeros(n)}
    obj._hist, obj._i, obj.v_media_b = torch.zeros(n, 4, 3), 0, torch.zeros(n, 3)
    return obj


def _passo_do_relogio(obj):
    with mock.patch.object(comando.UniformVelocityCommand, "_update_metrics"):
        obj._update_metrics()


class RelogioDoTwist(unittest.TestCase):

    def test_periodo_pela_velocidade(self):
        # T = 2/(1,4 + 1,2·v): v 0,5 dá 1,0 s; v 1,0 dá ≈ 0,769 s. O primeiro passo com comando soma dt/T à fase 0.
        for v, T in ((0.5, 1.0), (1.0, 2.0 / 2.6)):
            obj = _twist((v, 0.0, 0.0))
            _passo_do_relogio(obj)
            self.assertAlmostEqual(float(obj._env.v3_fase_marcha[0]), DT / T, places=6)

    def test_giro_sem_comando_linear_usa_v_03(self):
        obj = _twist((0.0, 0.0, 1.0))
        _passo_do_relogio(obj)
        self.assertAlmostEqual(float(obj._env.v3_fase_marcha[0]), DT * (C.cadencia_base + C.cadencia_por_v * 0.3) / 2.0, places=6)

    def test_recomeca_em_0_quando_o_comando_passa_do_limiar(self):
        # relógio desligado com fase velha 0,7: ao ligar, a fase parte de 0 (e não de 0,7)
        obj = _twist((0.5, 0.0, 0.0), fase=0.7, on=False)
        _passo_do_relogio(obj)
        self.assertAlmostEqual(float(obj._env.v3_fase_marcha[0]), DT / 1.0, places=6)
        self.assertTrue(bool(obj._env.v3_marcha_on[0]))

    def test_com_o_relogio_ligado_a_fase_integra_e_da_a_volta(self):
        obj = _twist((0.5, 0.0, 0.0), fase=0.99, on=True)
        _passo_do_relogio(obj)
        self.assertAlmostEqual(float(obj._env.v3_fase_marcha[0]), (0.99 + DT) % 1.0, places=5)

    def test_comando_zero_para_o_relogio_em_zero(self):
        obj = _twist((0.0, 0.0, 0.0), fase=0.4, on=True)
        _passo_do_relogio(obj)
        self.assertEqual(float(obj._env.v3_fase_marcha[0]), 0.0)
        self.assertFalse(bool(obj._env.v3_marcha_on[0]))

    def test_metrica_acerto_conta_so_os_passos_com_comando(self):
        obj = _twist((0.5, 0.0, 0.0))
        obj._env.v3_acerto_marcha[:] = 0.5
        _passo_do_relogio(obj)
        obj.vel_command_b[:] = 0.0
        _passo_do_relogio(obj)  # sem comando: não entra na média
        self.assertAlmostEqual(float(obj.metrics["acerto_relogio"][0]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
