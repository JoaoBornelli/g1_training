"""Testes da marcha por relógio (16/11, R: forma do `feet_gait` da Unitree): o termo `passo` e o relógio do `TwistV3`; e da
nota do segmento do ANDAR (16/11, A1), que o mesmo `_update_metrics` acumula.
Env falso com `types.SimpleNamespace`, como os vizinhos. O `_update_metrics` do fabricante fica fora (mock).
"""
import math
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
        v3_incl_max_seg=torch.zeros(n), v3_perdeu=torch.zeros(n, dtype=torch.bool),
        v3_cur=types.SimpleNamespace(id_celula=lambda tarefa, tipo: 1, restricao=lambda: 1.0, registra=mock.Mock()),
        scene={"box": types.SimpleNamespace(data=types.SimpleNamespace(
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)))})
    obj = comando.TwistV3.__new__(comando.TwistV3)
    obj._env, obj.k = env, C
    obj.robot = types.SimpleNamespace(data=types.SimpleNamespace(
        root_link_pos_w=torch.zeros(n, 3), heading_w=torch.zeros(n), root_link_lin_vel_b=torch.zeros(n, 3),
        root_link_ang_vel_b=torch.zeros(n, 3)))
    obj.vel_command_b = torch.tensor([cmd], dtype=torch.float32).repeat(n, 1)
    obj._agora, obj.t_ab, obj.ancora_feita = torch.zeros(n), torch.zeros(n), torch.zeros(n, dtype=torch.bool)
    obj._fase_t = torch.zeros(n)
    obj.command_counter, obj.seg_visto = torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)
    obj.seg_proj, obj.seg_int = torch.zeros(n, 2), torch.zeros(n, 2)
    obj.seg_nota, obj.seg_dur = torch.zeros(n, 2), torch.zeros(n)
    obj.seg_cmd, obj.seg_cel, obj.seg_carrega = obj.vel_command_b.clone(), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.bool)
    obj.soma_erro_marcha, obj.soma_cmd_marcha = torch.zeros(n), torch.zeros(n)
    obj.soma_acerto, obj.passos_on, obj.pose_ult = torch.zeros(n), torch.zeros(n), torch.zeros(n, 3)
    obj.metrics = {k: torch.zeros(n) for k in ("razao_marcha", "acerto_relogio", "eficiencia_min", "eficiencia_media", "segmentos")}
    return obj


def _passos_do_relogio(obj, n):
    with mock.patch.object(comando.UniformVelocityCommand, "_update_metrics"):
        for _ in range(n):
            obj._update_metrics()


class RelogioDoTwist(unittest.TestCase):

    def test_fase_e_o_tempo_sobre_08_com_comando_ligado_e_desligado(self):
        # 19/11, T12: 50 passos de 0,02 s = 1,0 s com comando → fase (1,0 / 0,8) mod 1 = 0,25; com comando zero, 0
        for cmd in ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0)):
            obj = _twist(cmd)
            _passos_do_relogio(obj, 50)
            self.assertAlmostEqual(float(obj._env.v3_fase_marcha[0]), 0.25, places=4)
        obj = _twist((0.0, 0.0, 0.0))
        _passos_do_relogio(obj, 50)
        self.assertEqual(float(obj._env.v3_fase_marcha[0]), 0.0)

    def test_sem_recomeco_quando_o_comando_passa_do_limiar(self):
        # 10 passos sem comando e 1 com: a fase conta só o passo com comando (1 · dt / 0,8)
        obj = _twist((0.0, 0.0, 0.0))
        _passos_do_relogio(obj, 10)
        obj.vel_command_b[:] = torch.tensor([0.5, 0.0, 0.0])
        _passos_do_relogio(obj, 1)
        self.assertAlmostEqual(float(obj._env.v3_fase_marcha[0]), 1 * DT / 0.8, places=5)
        self.assertTrue(bool(obj._env.v3_marcha_on[0]))

    def test_comando_zero_comando_soma_so_os_passos_com_comando(self):
        # 20 passos com comando (fase 20 · dt / 0,8 = 0,5), 30 sem (a fase fica em 0,5) e 10 com (30 passos de comando:
        # 30 · dt / 0,8 = 0,75)
        obj = _twist((0.5, 0.0, 0.0))
        _passos_do_relogio(obj, 20)
        obj.vel_command_b[:] = 0.0
        _passos_do_relogio(obj, 30)
        self.assertAlmostEqual(float(obj._env.v3_fase_marcha[0]), 0.5, places=5)
        obj.vel_command_b[:] = torch.tensor([0.5, 0.0, 0.0])
        _passos_do_relogio(obj, 10)
        self.assertAlmostEqual(float(obj._env.v3_fase_marcha[0]), 0.75, places=5)

    def test_reset_zera_a_fase(self):
        # 19/11, T12: o `_fase_t` zera no reset do episódio; o 1º passo depois dele soma só o próprio dt (o fabricante,
        # o fecho e a abertura do segmento ficam fora)
        obj = _twist((0.5, 0.0, 0.0))
        _passos_do_relogio(obj, 10)
        obj._env.termination_manager = types.SimpleNamespace(time_outs=torch.zeros(1, dtype=torch.bool))
        with mock.patch.object(comando.TwistV3, "_fecha"), mock.patch.object(comando.TwistV3, "_abre"), \
                mock.patch.object(comando.UniformVelocityCommand, "reset", return_value={}):
            obj.reset(torch.tensor([0]))
        self.assertEqual(float(obj._fase_t[0]), 0.0)
        _passos_do_relogio(obj, 1)
        self.assertAlmostEqual(float(obj._env.v3_fase_marcha[0]), DT / 0.8, places=5)

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


class NotaDoSegmento(unittest.TestCase):
    # 16/11, A1, T3 e T4; 17/11, T5: a nota do ANDAR é o MÍNIMO de dois canais, depois da âncora. Linear: o núcleo médio
    # exp(−‖c_xy − v_xy‖²/σ²), com a velocidade instantânea da base. Giro: o núcleo do erro médio, exp(−(média de c_z − ω_z)²/σ²),
    # que ignora o balanço da pelve dentro da passada. σ = 0,5.
    # 100 passos de 0,02 s = 2 s: a âncora (1,5 s) deixa 0,5 s de nota.

    def _nota(self, cmd, vel, falha=False):
        # devolve (a nota acumulada pelo `_update_metrics`, o que o `_fecha` registra para o segmento)
        obj = _twist(cmd)
        obj.robot.data.root_link_lin_vel_b[:] = torch.tensor(vel[:2] + (0.0,))
        obj.robot.data.root_link_ang_vel_b[:, 2] = vel[2]
        _passos_do_relogio(obj, 100)
        media = obj.seg_nota[0] / obj.seg_dur[0].clamp(min=1e-6)
        acumulada = min(float(media[0]), math.exp(-float(media[1]) ** 2 / C.sigma_nota ** 2))
        obj._fecha(torch.tensor([0]), torch.tensor([falha]))
        registra = obj._env.v3_cur.registra
        registra.assert_called_once()
        celula, n, soma = registra.call_args.args
        self.assertEqual((celula, n), (0, 1))
        return acumulada, float(soma)

    def test_rastreio_perfeito_da_nota_1(self):
        acumulada, registrada = self._nota((0.5, 0.0, 0.3), (0.5, 0.0, 0.3))
        self.assertAlmostEqual(acumulada, 1.0, places=5)
        self.assertAlmostEqual(registrada, 1.0, places=5)

    def test_robo_parado_com_comando_de_0_5_m_s_da_exp_menos_1(self):
        # linear: exp(−0,25/0,25) = e⁻¹; giro (comando 0, parado): 1 → o mínimo é e⁻¹ ≈ 0,368 (pela média dava 0,684)
        acumulada, registrada = self._nota((0.5, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(acumulada, math.exp(-1), places=5)
        self.assertAlmostEqual(registrada, math.exp(-1), places=5)

    def test_andar_certo_no_linear_sem_girar_com_comando_de_giro_0_5_da_exp_menos_1(self):
        # T4: o linear rende 1, o giro (comando 0,5 rad/s, ω = 0) rende exp(−0,25/0,25) = e⁻¹: o mínimo é e⁻¹.
        # Pela média, a nota seria 0,684 e passaria o portão de 0,8 só com esse erro pequeno de giro nos níveis baixos
        acumulada, registrada = self._nota((0.5, 0.0, 0.5), (0.5, 0.0, 0.0))
        self.assertAlmostEqual(acumulada, math.exp(-1), places=5)
        self.assertAlmostEqual(registrada, math.exp(-1), places=5)

    def test_queda_da_nota_0_mesmo_com_rastreio_perfeito_acumulado(self):
        acumulada, registrada = self._nota((0.5, 0.0, 0.0), (0.5, 0.0, 0.0), falha=True)
        self.assertAlmostEqual(acumulada, 1.0, places=5)  # o segmento rastreou até cair
        self.assertEqual(registrada, 0.0)

    def test_empurrao_depois_da_ancora_nao_tira_o_registro(self):
        # 17/11, T6: o empurrão leva a âncora a 3,4 s, à frente do relógio (2,02 s). Pela regra antiga, o segmento
        # fechava antes da âncora e não registrava; pela nota, registra os 0,5 s depois da âncora de antes do empurrão.
        obj = _twist((0.5, 0.0, 0.0))
        obj.robot.data.root_link_lin_vel_b[:, 0] = 0.5
        _passos_do_relogio(obj, 100)
        obj._env.v3_t_empurrao[:] = 1.9
        _passos_do_relogio(obj, 1)
        obj._fecha(torch.tensor([0]), torch.tensor([False]))
        registra = obj._env.v3_cur.registra
        registra.assert_called_once()
        celula, n, soma = registra.call_args.args
        self.assertEqual((celula, n), (0, 1))
        self.assertAlmostEqual(float(soma), 1.0, places=5)

    def test_balanco_do_giro_em_torno_do_comando_nao_baixa_a_nota(self):
        # ω̄ = c_z = 0,3: erro médio zero, o giro rende 1; o linear (perfeito) rende 1. Pelo núcleo instantâneo, dava ≈ 0.
        # `t_ab` em −10 põe a âncora (1,5 s) antes dos 100 passos, e 100 é par: a oscilação fecha a média em zero
        obj = _twist((0.5, 0.0, 0.3))
        obj.robot.data.root_link_lin_vel_b[:, 0] = 0.5
        obj.t_ab[:] = -10.0
        with mock.patch.object(comando.UniformVelocityCommand, "_update_metrics"):
            for i in range(100):
                obj.robot.data.root_link_ang_vel_b[:, 2] = 0.3 + (1.65 if i % 2 == 0 else -1.65)
                obj._update_metrics()
        obj._fecha(torch.tensor([0]), torch.tensor([False]))
        registra = obj._env.v3_cur.registra
        registra.assert_called_once()
        self.assertAlmostEqual(float(registra.call_args.args[2]), 1.0, places=5)

    def test_vies_de_giro_com_balanco_cobra_pelo_erro_medio(self):
        # ω̄ = 0,5 e c_z = 0,3: erro médio −0,2, o giro rende exp(−0,04/0,25) = exp(−0,16) ≈ 0,8521; o linear rende 1
        # `t_ab` em −10, como no teste anterior: os 100 passos contam depois da âncora
        obj = _twist((0.5, 0.0, 0.3))
        obj.robot.data.root_link_lin_vel_b[:, 0] = 0.5
        obj.t_ab[:] = -10.0
        with mock.patch.object(comando.UniformVelocityCommand, "_update_metrics"):
            for i in range(100):
                obj.robot.data.root_link_ang_vel_b[:, 2] = 0.5 + (1.65 if i % 2 == 0 else -1.65)
                obj._update_metrics()
        obj._fecha(torch.tensor([0]), torch.tensor([False]))
        registra = obj._env.v3_cur.registra
        registra.assert_called_once()
        self.assertAlmostEqual(float(registra.call_args.args[2]), math.exp(-0.16), places=3)


if __name__ == "__main__":
    unittest.main()
