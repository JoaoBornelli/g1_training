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

    def test_relogio_parado_paga_um_por_pe_em_contato_com_o_chao(self):
        # 24/11, T16: comando zero ou até o limiar (‖c‖ ≤ 0,1): o alvo dos dois pés é o chão, em qualquer fase. Dois pés no chão
        # pagam o máximo (2, o 1,0/s do peso 0,5), um pé no ar paga metade, os dois no ar pagam 0
        for cmd in ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.1, 0.0, 0.0)):
            for fase in (0.2, 0.7):
                self.assertAlmostEqual(_passo(_env_passo(cmd, fase=fase, contato=(True, True))), 2.0, places=5, msg=str(cmd))
                self.assertAlmostEqual(_passo(_env_passo(cmd, fase=fase, contato=(True, False))), 1.0, places=5, msg=str(cmd))
                self.assertAlmostEqual(_passo(_env_passo(cmd, fase=fase, contato=(False, False))), 0.0, places=5, msg=str(cmd))

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

    def test_zera_com_a_caixa_perdida_e_nao_grava_o_acerto(self):
        # o acerto que o g lê mora no `TwistV3._update_metrics`: o termo não o grava (o mjlab pula um termo de peso 0)
        env = _env_passo((0.5, 0.0, 0.0))
        self.assertAlmostEqual(_passo(env), 2.0, places=5)
        self.assertEqual(float(env.v3_acerto_marcha[0]), 0.0)
        self.assertAlmostEqual(_passo(_env_passo((0.5, 0.0, 0.0), perdeu=True)), 0.0, places=5)
        self.assertAlmostEqual(_passo(_env_passo((0.0, 0.0, 0.0), contato=(True, True), perdeu=True)), 0.0, places=5)


def _twist(cmd, n=1, contato=(True, True)):
    # `TwistV3` sem o `__init__`: só o que `_update_metrics` lê, com o fabricante e a âncora fora. `contato`: os pés em contato
    # (T16: o acerto, q̄ e g saem daqui); padrão, os dois no chão
    env = types.SimpleNamespace(
        step_dt=DT, v3_fase_marcha=torch.zeros(n), v3_marcha_on=torch.zeros(n, dtype=torch.bool),
        v3_acerto_marcha=torch.ones(n), v3_t_empurrao=torch.full((n,), float("-inf")), v3_ancora=torch.zeros(n, 4),
        v3_incl_max_seg=torch.zeros(n), v3_perdeu=torch.zeros(n, dtype=torch.bool),
        v3_cur=types.SimpleNamespace(id_celula=lambda tarefa, tipo: 1, restricao=lambda: 1.0, registra=mock.Mock()),
        scene={"box": types.SimpleNamespace(data=types.SimpleNamespace(
                   root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1))),
               "pes_chao": types.SimpleNamespace(data=types.SimpleNamespace(
                   found=torch.tensor([[int(c) for c in contato]]).repeat(n, 1)))})
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
    obj.seg_nota, obj.seg_dur, obj.v_filt = torch.zeros(n, 3), torch.zeros(n), torch.zeros(n, 3)
    obj.q_filt, obj.g_marcha = torch.zeros(n), torch.zeros(n)
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
        # fase 0,2: E em apoio, D em balanço; os dois no chão: E acerta e D erra, acerto 0,5
        obj = _twist((0.5, 0.0, 0.0))
        obj._env.v3_fase_marcha[:] = 0.2
        _passos_do_relogio(obj, 1)
        self.assertAlmostEqual(float(obj._env.v3_acerto_marcha[0]), 0.5, places=6)
        obj.vel_command_b[:] = 0.0
        _passos_do_relogio(obj, 1)  # sem comando: o acerto mede os pés no chão (1,0) e não entra na média
        self.assertAlmostEqual(float(obj.metrics["acerto_relogio"][0]), 0.5, places=6)


class FiltroDaVelocidade(unittest.TestCase):
    # 24/11, T15: a recompensa lê a velocidade da base filtrada (média móvel exponencial, τ = `tau_fracao` = 0,4 s) que o `twist`
    # guarda por ambiente; ela zera no reset do episódio

    def _recompensa(self, obj):
        # o `_rastreio` real sobre o `twist` de teste: comando, velocidade instantânea e a filtrada (g = 1: aqui só a tenda)
        env = types.SimpleNamespace(
            command_manager=types.SimpleNamespace(get_command=lambda _: obj.vel_command_b,
                                                  get_term=lambda _: types.SimpleNamespace(v_filt=obj.v_filt, g_marcha=torch.ones(1))),
            scene={"robot": obj.robot}, v3_cur=types.SimpleNamespace(restricao=lambda: 1.0))
        return float(recompensas._rastreio(env, True)[0])

    def test_balanco_de_0_15_m_s_a_1_25_hz_em_torno_do_alvo_paga_0_9_ou_mais(self):
        # a passada balança a velocidade; o filtro deixa ~30 % da oscilação e a tenda quase não a cobra. Instantânea, o mesmo
        # balanço rende ~0,8
        obj = _twist((0.5, 0.0, 0.0))
        pagas = []
        with mock.patch.object(comando.UniformVelocityCommand, "_update_metrics"):
            for i in range(300):
                obj.robot.data.root_link_lin_vel_b[:, 0] = 0.5 + 0.15 * math.sin(2 * math.pi * 1.25 * i * DT)
                obj._update_metrics()
                if i >= 100:  # depois do transitório do filtro (τ 0,4 s)
                    pagas.append(self._recompensa(obj))
        self.assertGreaterEqual(sum(pagas) / len(pagas), 0.9)
        self.assertGreaterEqual(min(pagas), 0.85)

    def test_o_mesmo_balanco_no_alvo_de_1_m_s_separa_a_tenda_filtrada_do_nucleo_antigo(self):
        # medido (alvo 1,0, balanço ±0,15 m/s a 1,25 Hz, depois do transitório): tenda filtrada média 0,971 e mínimo 0,952; o núcleo
        # instantâneo de antes, exp(−e²/σ²), média 0,957 e mínimo 0,914; a tenda sem filtro, média 0,905 e mínimo 0,850. Com mínimo
        # ≥ 0,94 e média ≥ 0,96 só a tenda filtrada passa
        obj = _twist((1.0, 0.0, 0.0))
        pagas = []
        with mock.patch.object(comando.UniformVelocityCommand, "_update_metrics"):
            for i in range(300):
                obj.robot.data.root_link_lin_vel_b[:, 0] = 1.0 + 0.15 * math.sin(2 * math.pi * 1.25 * i * DT)
                obj._update_metrics()
                if i >= 100:
                    pagas.append(self._recompensa(obj))
        self.assertGreaterEqual(sum(pagas) / len(pagas), 0.96)
        self.assertGreaterEqual(min(pagas), 0.94)

    def test_a_velocidade_filtrada_sobe_com_a_constante_de_tempo(self):
        # degrau de 0,5 m/s: depois de τ = 0,4 s (20 passos) o filtro chega a 1 − e⁻¹ do degrau
        obj = _twist((0.5, 0.0, 0.0))
        obj.robot.data.root_link_lin_vel_b[:, 0] = 0.5
        _passos_do_relogio(obj, 20)
        self.assertAlmostEqual(float(obj.v_filt[0, 0]), 0.5 * (1 - math.exp(-1)), places=4)

    def test_reset_zera_o_filtro_e_nao_vaza_para_o_episodio_seguinte(self):
        obj = _twist((0.5, 0.0, 0.3))
        obj.robot.data.root_link_lin_vel_b[:, 0] = 0.5
        obj.robot.data.root_link_ang_vel_b[:, 2] = 0.3
        _passos_do_relogio(obj, 30)
        self.assertGreater(float(obj.v_filt[0, 0]), 0.3)
        obj._env.termination_manager = types.SimpleNamespace(time_outs=torch.zeros(1, dtype=torch.bool))
        with mock.patch.object(comando.TwistV3, "_fecha"), mock.patch.object(comando.TwistV3, "_abre"), \
                mock.patch.object(comando.UniformVelocityCommand, "reset", return_value={}):
            obj.reset(torch.tensor([0]))
        self.assertEqual(obj.v_filt[0].tolist(), [0.0, 0.0, 0.0])
        obj.robot.data.root_link_lin_vel_b[:] = 0.0  # o robô novo nasce parado
        obj.robot.data.root_link_ang_vel_b[:] = 0.0
        _passos_do_relogio(obj, 1)
        self.assertEqual(obj.v_filt[0].tolist(), [0.0, 0.0, 0.0])


class AcertoFiltradoEG(unittest.TestCase):
    # 24/11, T16: o acerto do relógio dos dois pés e q̄ (filtro de τ `tau_fracao`) moram no `_update_metrics`; g = clamp(2·q̄ − 1, 0, 1)

    def test_g_vale_sem_o_termo_passo(self):
        # o `passo` nunca roda aqui (o mjlab pula um termo de peso 0): o acerto, q̄ e g saem só do `_update_metrics`.
        # Comando zero e os dois pés no chão: acerto 1, q̄ = 1 − e⁻¹ aos 20 passos (τ 0,4 s), g = 2q̄ − 1
        obj = _twist((0.0, 0.0, 0.0))
        _passos_do_relogio(obj, 20)
        q = 1 - math.exp(-1)
        self.assertAlmostEqual(float(obj.q_filt[0]), q, places=4)
        self.assertAlmostEqual(float(obj.g_marcha[0]), 2 * q - 1, places=4)

    def test_g_e_zero_no_acaso_e_um_no_acerto_total(self):
        # comando zero (alvo: o chão): só o pé E no chão acerta 0,5 → q̄ → 0,5, g → 0; os dois no chão → g → 1
        obj = _twist((0.0, 0.0, 0.0), contato=(True, False))
        _passos_do_relogio(obj, 400)
        self.assertAlmostEqual(float(obj.q_filt[0]), 0.5, places=3)
        self.assertAlmostEqual(float(obj.g_marcha[0]), 0.0, places=2)
        obj = _twist((0.0, 0.0, 0.0))
        _passos_do_relogio(obj, 400)
        self.assertAlmostEqual(float(obj.g_marcha[0]), 1.0, places=2)
        # acerto 0 (os dois pés no ar, alvo o chão): q̄ = 0 e g não fica negativo
        obj = _twist((0.0, 0.0, 0.0), contato=(False, False))
        _passos_do_relogio(obj, 50)
        self.assertEqual(float(obj.g_marcha[0]), 0.0)

    def test_acerto_com_comando_e_o_do_relogio_com_a_fase_do_passo_anterior(self):
        # fase 0,2: E em apoio, D em balanço. (E no chão, D no ar) acerta os dois; (E, D no chão) acerta um
        for contato, esperado in (((True, False), 1.0), ((True, True), 0.5), ((False, True), 0.0)):
            obj = _twist((0.5, 0.0, 0.0), contato=contato)
            obj._env.v3_fase_marcha[:] = 0.2
            _passos_do_relogio(obj, 1)
            self.assertAlmostEqual(float(obj._env.v3_acerto_marcha[0]), esperado, places=6, msg=str(contato))

    def test_q_parte_do_acaso_na_troca_de_parado_para_comando(self):
        # parado muito tempo com os dois pés no chão: q̄ ≈ 1. Quando o comando volta ele parte de 0,5 (g = 0 no instante, e já sobe no
        # 1º passo), e não de ≈ 1 (os passinhos pagariam a velocidade quase cheia nos primeiros 0,4 s) nem de 0 (~0,3 s morto
        # para a marcha certa). 1º passo com comando, fase 0,2, (E no chão, D no ar): acerto 1
        obj = _twist((0.0, 0.0, 0.0))
        _passos_do_relogio(obj, 100)
        self.assertGreater(float(obj.q_filt[0]), 0.95)
        obj.vel_command_b[:] = torch.tensor([0.5, 0.0, 0.0])
        obj._env.scene["pes_chao"].data.found[:] = torch.tensor([[1, 0]])
        obj._env.v3_fase_marcha[:] = 0.2
        _passos_do_relogio(obj, 1)
        k = 1 - math.exp(-DT / 0.4)
        self.assertAlmostEqual(float(obj.q_filt[0]), 0.5 + k * 0.5, places=5)
        self.assertAlmostEqual(float(obj.g_marcha[0]), k, places=5)  # 2q − 1: já sobe no 1º passo
        # e na troca de comando para parado (sem prêmio a vigiar): o filtro segue sem zerar
        obj.vel_command_b[:] = 0.0
        q = float(obj.q_filt[0])
        _passos_do_relogio(obj, 1)
        self.assertGreater(float(obj.q_filt[0]), 0.9 * q)

    def test_reset_poe_q_no_acaso_e_g_em_zero_e_nao_vaza(self):
        obj = _twist((0.0, 0.0, 0.0))
        _passos_do_relogio(obj, 100)
        self.assertGreater(float(obj.g_marcha[0]), 0.5)
        obj._env.termination_manager = types.SimpleNamespace(time_outs=torch.zeros(1, dtype=torch.bool))
        with mock.patch.object(comando.TwistV3, "_fecha"), mock.patch.object(comando.TwistV3, "_abre"), \
                mock.patch.object(comando.UniformVelocityCommand, "reset", return_value={}):
            obj.reset(torch.tensor([0]))
        self.assertEqual((float(obj.q_filt[0]), float(obj.g_marcha[0])), (0.5, 0.0))
        obj._env.scene["pes_chao"].data.found[:] = 0  # o robô novo nasce com os pés no ar: nada do episódio antigo sobra
        _passos_do_relogio(obj, 1)
        self.assertAlmostEqual(float(obj.q_filt[0]), 0.5 * math.exp(-DT / 0.4), places=5)
        self.assertEqual(float(obj.g_marcha[0]), 0.0)

    def test_acerto_com_comando_zero_mede_os_pes_no_chao_em_qualquer_fase(self):
        # T16: fase 0,7 (o E "deveria" estar no ar pelo relógio, o D em apoio) e o relógio parado: os dois no chão acertam os dois
        obj = _twist((0.0, 0.0, 0.0))
        obj._env.v3_fase_marcha[:] = 0.7
        _passos_do_relogio(obj, 1)
        self.assertAlmostEqual(float(obj._env.v3_acerto_marcha[0]), 1.0, places=6)


class NotaDoSegmento(unittest.TestCase):
    # 24/11, T15: a nota do ANDAR é a média dos dois canais. Canal comandado: min(razão, 1), com a razão desde a âncora: (v̄·ĉ)/‖c‖ no
    # linear e ω̄/c_z no giro (razão negativa vale 0). Canal sem comando: a deriva exp(−v̄²/σ²) da média do segmento. Queda: 0. 100 passos de 0,02 s = 2 s: a âncora (1,5 s) deixa 0,5 s de nota.

    def _nota(self, cmd, vel, falha=False, antes=None):
        # devolve o que o `_fecha` registra para o segmento. `antes`: velocidade nos 70 passos antes da âncora (os 30 restantes
        # passam por ela)
        obj = _twist(cmd)
        if antes is not None:
            obj.robot.data.root_link_lin_vel_b[:] = torch.tensor(antes[:2] + (0.0,))
            obj.robot.data.root_link_ang_vel_b[:, 2] = antes[2]
            _passos_do_relogio(obj, 70)
        obj.robot.data.root_link_lin_vel_b[:] = torch.tensor(vel[:2] + (0.0,))
        obj.robot.data.root_link_ang_vel_b[:, 2] = vel[2]
        _passos_do_relogio(obj, 100 - (70 if antes is not None else 0))
        obj._fecha(torch.tensor([0]), torch.tensor([falha]))
        registra = obj._env.v3_cur.registra
        registra.assert_called_once()
        celula, n, soma = registra.call_args.args
        self.assertEqual((celula, n), (0, 1))
        return float(soma)

    def test_rastreio_perfeito_da_nota_1(self):
        self.assertAlmostEqual(self._nota((0.5, 0.0, 0.3), (0.5, 0.0, 0.3)), 1.0, places=5)

    def test_robo_parado_com_comando_de_0_5_m_s_da_0_5(self):
        # linear comandado: razão 0; giro sem comando e parado: deriva 1; média 0,5 (antes, e⁻¹ pelo núcleo)
        self.assertAlmostEqual(self._nota((0.5, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.5, places=5)

    def test_media_dos_dois_canais(self):
        # linear perfeito (1) e giro de 0,5 rad/s sem girar (0): média 0,5
        self.assertAlmostEqual(self._nota((0.5, 0.0, 0.5), (0.5, 0.0, 0.0)), 0.5, places=5)
        # linear 0,8 e giro sem comando e parado (deriva 1): média 0,9
        self.assertAlmostEqual(self._nota((0.5, 0.0, 0.0), (0.4, 0.0, 0.0)), 0.9, places=5)

    def test_canal_sem_comando_entra_na_media_com_a_deriva(self):
        # linear comandado com razão 1 e giro sem comando derivando 0,5 rad/s: (1 + e⁻¹) / 2. O giro parado que gira é cobrado
        self.assertAlmostEqual(self._nota((0.5, 0.0, 0.0), (0.5, 0.0, 0.5)), (1 + math.exp(-1)) / 2, places=5)
        # giro comandado com razão 1 e linear sem comando derivando 0,5 m/s: o mesmo
        self.assertAlmostEqual(self._nota((0.0, 0.0, 0.3), (0.5, 0.0, 0.3)), (1 + math.exp(-1)) / 2, places=5)

    def test_razao_de_1_3_vale_1_e_razao_negativa_vale_0(self):
        self.assertAlmostEqual(self._nota((0.5, 0.0, 0.0), (0.65, 0.0, 0.0)), 1.0, places=5)
        self.assertAlmostEqual(self._nota((0.5, 0.0, 0.3), (0.5, 0.0, 0.39)), 1.0, places=5)
        # (o outro canal, sem comando e sem deriva, vale 1: a média é 0,5)
        self.assertAlmostEqual(self._nota((0.5, 0.0, 0.0), (-0.3, 0.0, 0.0)), 0.5, places=5)
        self.assertAlmostEqual(self._nota((0.0, 0.0, 0.3), (0.0, 0.0, -0.3)), 0.5, places=5)
        self.assertAlmostEqual(self._nota((0.5, 0.0, 0.3), (-0.3, 0.0, -0.3)), 0.0, places=5)
        # ré e giro negativo: a mesma razão que frente e giro positivo
        self.assertAlmostEqual(self._nota((-0.5, 0.0, -0.3), (-0.4, 0.0, -0.24)), 0.8, places=5)

    def test_a_conta_vem_da_ancora(self):
        # os 70 passos antes da âncora (parado) não contam: depois dela, o comando é cumprido
        self.assertAlmostEqual(self._nota((0.5, 0.0, 0.0), (0.5, 0.0, 0.0), antes=(0.0, 0.0, 0.0)), 1.0, places=5)

    def test_comando_abaixo_do_limiar_do_canal_conta_como_sem_comando(self):
        # `limiar_canal` = 0,02: um comando de 0,015 (linear ou giro) é canal sem comando, e parado vale a deriva 1; de 0,05 é
        # comandado e parado vale razão 0 (média 0,5)
        self.assertAlmostEqual(self._nota((0.015, 0.0, 0.0), (0.0, 0.0, 0.0)), 1.0, places=5)
        self.assertAlmostEqual(self._nota((0.0, 0.0, 0.015), (0.0, 0.0, 0.0)), 1.0, places=5)
        self.assertAlmostEqual(self._nota((0.05, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.5, places=5)
        self.assertAlmostEqual(self._nota((0.0, 0.0, 0.05), (0.0, 0.0, 0.0)), 0.5, places=5)

    def test_comando_zero_usa_a_media_das_duas_derivas(self):
        self.assertAlmostEqual(self._nota((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 1.0, places=5)
        self.assertAlmostEqual(self._nota((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)), (1 + math.exp(-1)) / 2, places=5)
        self.assertAlmostEqual(self._nota((0.0, 0.0, 0.0), (0.0, 0.0, 0.5)), (1 + math.exp(-1)) / 2, places=5)
        self.assertAlmostEqual(self._nota((0.0, 0.0, 0.0), (0.5, 0.0, 0.5)), math.exp(-1), places=5)

    def test_queda_da_nota_0_mesmo_com_rastreio_perfeito_acumulado(self):
        self.assertEqual(self._nota((0.5, 0.0, 0.0), (0.5, 0.0, 0.0), falha=True), 0.0)

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

    def test_segmento_sem_tempo_depois_da_ancora_nao_registra_mas_a_queda_registra(self):
        obj = _twist((0.5, 0.0, 0.0))
        _passos_do_relogio(obj, 10)  # 0,2 s: antes da âncora
        obj._fecha(torch.tensor([0]), torch.tensor([False]))
        obj._env.v3_cur.registra.assert_not_called()
        obj._fecha(torch.tensor([0]), torch.tensor([True]))
        obj._env.v3_cur.registra.assert_called_once()

    def test_balanco_do_giro_em_torno_do_comando_nao_baixa_a_nota(self):
        # ω̄ = c_z = 0,3: razão 1; o linear (perfeito) rende 1. `t_ab` em −10 põe a âncora (1,5 s) antes dos 100 passos, e 100
        # é par: a oscilação fecha a média em zero
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


if __name__ == "__main__":
    unittest.main()
