"""Testes da marcha por relógio (16/11, R: forma do `feet_gait` da Unitree): o termo `passo` e o relógio do `TwistV3`.
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


def _env_passo(cmd, fase=0.2, contato=(True, False), perdeu=False):
    # fase 0,2: o pé E está em apoio (0,2 < 0,55) e o D em balanço (0,7); contato (E no chão, D no ar) acerta os dois.
    # Sem `get_term`: o termo não lê mais a velocidade do robô (sem portão).
    return types.SimpleNamespace(
        device="cpu", v3_tarefa=torch.tensor([TAREFAS.index("ANDAR")]), v3_perdeu=torch.tensor([perdeu]),
        v3_fase_marcha=torch.tensor([fase]), v3_acerto_marcha=torch.zeros(1),
        scene={"pes_chao": types.SimpleNamespace(data=types.SimpleNamespace(found=torch.tensor([[int(c) for c in contato]])))},
        command_manager=types.SimpleNamespace(get_command=lambda _: torch.tensor([cmd], dtype=torch.float32)))


def _passo(env):
    return float(recompensas.passo(env, ("ANDAR", "CARREGAR"))[0])


class TermoPasso(unittest.TestCase):

    def test_comando_no_limiar_ou_abaixo_da_zero(self):
        self.assertAlmostEqual(_passo(_env_passo((0.0, 0.0, 0.0))), 0.0, places=5)
        self.assertAlmostEqual(_passo(_env_passo((0.05, 0.0, 0.0))), 0.0, places=5)

    def test_com_comando_paga_a_soma_dos_acertos(self):
        self.assertAlmostEqual(_passo(_env_passo((0.5, 0.0, 0.0))), 2.0, places=5)
        self.assertAlmostEqual(_passo(_env_passo((0.5, 0.0, 0.0), contato=(True, True))), 1.0, places=5)
        self.assertAlmostEqual(_passo(_env_passo((0.5, 0.0, 0.0), contato=(False, True))), 0.0, places=5)

    def test_a_norma_do_comando_inteiro_liga_o_termo(self):
        # só o giro, e vx e vy juntos: cada canal sozinho passa, ‖c‖ > 0,1 em todos
        self.assertAlmostEqual(_passo(_env_passo((0.0, 0.0, 0.5))), 2.0, places=5)
        self.assertAlmostEqual(_passo(_env_passo((0.08, 0.08, 0.0))), 2.0, places=5)

    def test_robo_parado_com_comando_recebe_pelos_acertos_sem_portao(self):
        # o env não tem velocidade nenhuma: o termo não a lê, e o comando de 0,5 m/s paga os 2 acertos
        self.assertAlmostEqual(_passo(_env_passo((0.5, 0.0, 0.0))), 2.0, places=5)

    def test_grava_o_acerto_mesmo_sem_comando_e_zera_com_a_caixa_perdida(self):
        env = _env_passo((0.0, 0.0, 0.0), contato=(True, True))  # E acerta, D erra
        self.assertAlmostEqual(_passo(env), 0.0, places=5)
        self.assertAlmostEqual(float(env.v3_acerto_marcha[0]), 0.5, places=5)
        self.assertAlmostEqual(_passo(_env_passo((0.5, 0.0, 0.0), perdeu=True)), 0.0, places=5)


def _twist(cmd, n=1):
    # `TwistV3` sem o `__init__`: só o que `_update_metrics` lê, com o fabricante e a âncora fora
    env = types.SimpleNamespace(
        step_dt=DT, v3_fase_marcha=torch.zeros(n), v3_marcha_on=torch.zeros(n, dtype=torch.bool),
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


def _passos_do_relogio(obj, n):
    with mock.patch.object(comando.UniformVelocityCommand, "_update_metrics"):
        for _ in range(n):
            obj._update_metrics()


class RelogioDoTwist(unittest.TestCase):

    def test_fase_e_o_tempo_sobre_08_com_comando_ligado_e_desligado(self):
        # 50 passos de 0,02 s = 1,0 s → fase (1,0 / 0,8) mod 1 = 0,25; o comando não muda o período
        for cmd in ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)):
            obj = _twist(cmd)
            _passos_do_relogio(obj, 50)
            self.assertAlmostEqual(float(obj._env.v3_fase_marcha[0]), 0.25, places=4)

    def test_sem_recomeco_quando_o_comando_passa_do_limiar(self):
        # 10 passos sem comando e 1 com: a fase segue o tempo (11 · dt / 0,8), e não parte de 0
        obj = _twist((0.0, 0.0, 0.0))
        _passos_do_relogio(obj, 10)
        obj.vel_command_b[:] = torch.tensor([0.5, 0.0, 0.0])
        _passos_do_relogio(obj, 1)
        self.assertAlmostEqual(float(obj._env.v3_fase_marcha[0]), 11 * DT / 0.8, places=5)
        self.assertTrue(bool(obj._env.v3_marcha_on[0]))

    def test_a_fase_da_a_volta_a_cada_08_s(self):
        obj = _twist((0.5, 0.0, 0.0))
        _passos_do_relogio(obj, 45)  # 0,9 s → 0,125
        self.assertAlmostEqual(float(obj._env.v3_fase_marcha[0]), 0.125, places=4)

    def test_on_pela_norma_do_comando_inteiro(self):
        for cmd, esperado in (((0.05, 0.0, 0.0), False), ((0.0, 0.0, 0.5), True), ((0.08, 0.08, 0.0), True),
                              ((0.0, 0.0, 0.0), False)):
            obj = _twist(cmd)
            _passos_do_relogio(obj, 1)
            self.assertEqual(bool(obj._env.v3_marcha_on[0]), esperado, cmd)

    def test_metrica_acerto_conta_so_os_passos_com_comando(self):
        obj = _twist((0.5, 0.0, 0.0))
        obj._env.v3_acerto_marcha[:] = 0.5
        _passos_do_relogio(obj, 1)
        obj.vel_command_b[:] = 0.0
        _passos_do_relogio(obj, 1)  # sem comando: não entra na média
        self.assertAlmostEqual(float(obj.metrics["acerto_relogio"][0]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
