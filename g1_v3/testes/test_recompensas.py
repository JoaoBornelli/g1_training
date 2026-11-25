"""Testes dos núcleos de recompensa da D5 (recompensas.py) e de potencial.excesso_de_junta. Escritos antes do código.

Os números são os da linha "Teste" de D5 (plano §11) e dos knobs. Tensores sintéticos; nenhum env é construído.
Ângulos em graus, comprimentos em metros, torque em fração (§6.3).
"""
import math
import re
import types
import unittest

import torch

from g1_v3 import potencial, recompensas
from g1_v3.knobs import ATIVO, JUNTAS, caminho

R = ATIVO.recompensa
DEV = "cpu"
F = torch.float32


def t(*valores):
    return torch.tensor(valores, dtype=F)


def _tau_max():
    """Vetor (29,) do τ_max pela primeira regex que casa com a junta (S §9.3)."""
    return t(*[next(v for rx, v in R.tau_max if re.fullmatch(rx, j)) for j in JUNTAS])


class TestPasso(unittest.TestCase):
    """16/11, R: n_passo é o relógio do `feet_gait`. Pé E na `fase`, pé D na fase + 0,5; apoio com fase < 0,55."""

    def n(self, contato, fase=0.2, on=True):
        return recompensas.n_passo(torch.tensor([contato]), t(fase), ATIVO.comando.fase_apoio, torch.tensor([on]))[0].tolist()

    def test_fase_02_e_apoio_e_d_em_balanco(self):
        # E: 0,2 < 0,55 apoio; D: 0,7 balanço
        self.assertEqual(self.n((True, False)), [1.0, 1.0])
        self.assertEqual(self.n((True, True)), [1.0, 0.0])

    def test_fase_070_inverte_os_papeis(self):
        # E: 0,7 balanço; D: 0,2 apoio
        self.assertEqual(self.n((False, True), fase=0.7), [1.0, 1.0])

    def test_dois_pes_no_chao_acertam_055_do_ciclo_por_pe(self):
        fases = torch.arange(1000, dtype=F) / 1000.0
        acerto = recompensas.n_passo(torch.ones(1000, 2, dtype=torch.bool), fases, ATIVO.comando.fase_apoio,
                                     torch.ones(1000, dtype=torch.bool))
        for pe in (0, 1):
            self.assertAlmostEqual(acerto[:, pe].mean().item(), 0.55, places=2)

    def test_relogio_parado_o_alvo_e_o_chao_em_qualquer_fase(self):
        # 24/11, T16: com `on` falso cada pé em contato vale 1 e cada pé no ar vale 0, seja qual for a fase
        for fase in (0.2, 0.7):
            self.assertEqual(self.n((True, True), fase=fase, on=False), [1.0, 1.0])
            self.assertEqual(self.n((True, False), fase=fase, on=False), [1.0, 0.0])


class TestVivo(unittest.TestCase):
    def test_zera_com_caixa_perdida(self):
        self.assertEqual(recompensas.n_vivo(torch.tensor([True, False])).tolist(), [0.0, 1.0])


class TestAlturaPe(unittest.TestCase):
    def test_pes_parados_dao_1(self):
        h = t(0.0, 0.0).reshape(1, 2)
        v = torch.zeros(1, 2, 2, dtype=F)
        out = recompensas.n_altura_pe(h, v, R.alvo_pe, R.std_pe, R.tanh_pe)
        self.assertAlmostEqual(out.item(), 1.0, places=5)


class TestCercas(unittest.TestCase):
    def test_cerca_giro_hip_yaw_e_menos_40_da_10(self):
        out = recompensas.n_cerca_giro(t(math.radians(-40.0)), t(0.0), R.lim_giro_graus)
        self.assertAlmostEqual(out.item(), 10.0, places=4)

    def test_cerca_tronco_20_graus_no_andar_da_5(self):
        out = recompensas.n_cerca_tronco(t(20.0), t(0.0), R.lim_tronco_graus)
        self.assertAlmostEqual(out.item(), 5.0, places=4)

    def _retangulo(self):
        # Solas: retângulo ±0,10 m × ±0,05 m (os 8 cantos, dois pés com o mesmo contorno).
        canto = [(0.10, 0.05), (0.10, -0.05), (-0.10, 0.05), (-0.10, -0.05)]
        return torch.tensor([canto + canto], dtype=F)

    def test_com_5_cm_dentro_da_0(self):
        out = recompensas.n_cerca_com(t(0.0, 0.0).reshape(1, 2), self._retangulo(),
                                      torch.tensor([True]), R.lim_com)
        self.assertAlmostEqual(out.item(), 0.0, places=6)

    def test_com_1_cm_dentro_da_002(self):
        out = recompensas.n_cerca_com(t(0.0, 0.04).reshape(1, 2), self._retangulo(),
                                      torch.tensor([True]), R.lim_com)
        self.assertAlmostEqual(out.item(), 0.02, places=5)

    def test_um_pe_so_no_chao_da_0(self):
        out = recompensas.n_cerca_com(t(0.0, 0.04).reshape(1, 2), self._retangulo(),
                                      torch.tensor([False]), R.lim_com)
        self.assertAlmostEqual(out.item(), 0.0, places=6)

    def test_torque_a_90_por_cento_da_0_5(self):
        tau = torch.zeros(1, len(JUNTAS), dtype=F)
        tau[0, JUNTAS.index("left_knee_joint")] = 0.9 * _tau_max()[JUNTAS.index("left_knee_joint")]
        out = recompensas.n_cerca_torque(tau, _tau_max(), torch.zeros((0, 2), dtype=torch.long),
                                         R.tau_par, R.lim_torque, R.banda_torque)
        self.assertAlmostEqual(out.item(), 0.5, places=5)

    def test_torque_do_par_da_cintura_soma_45_de_50_da_0_5(self):
        tau = torch.zeros(1, len(JUNTAS), dtype=F)
        tau[0, JUNTAS.index("waist_pitch_joint")] = 30.0
        tau[0, JUNTAS.index("waist_roll_joint")] = 15.0
        pares = torch.tensor([[JUNTAS.index(a), JUNTAS.index(b)] for a, b in R.pares_torque], dtype=torch.long)
        out = recompensas.n_cerca_torque(tau, _tau_max(), pares, R.tau_par, R.lim_torque, R.banda_torque)
        self.assertAlmostEqual(out.item(), 0.5, places=5)

    def test_torque_da_cintura_sem_regra_de_par_da_0(self):
        # 30/50 = 0,6 e 15/50 = 0,3, ambos abaixo de 0,8 por junta.
        tau = torch.zeros(1, len(JUNTAS), dtype=F)
        tau[0, JUNTAS.index("waist_pitch_joint")] = 30.0
        tau[0, JUNTAS.index("waist_roll_joint")] = 15.0
        out = recompensas.n_cerca_torque(tau, _tau_max(), torch.zeros((0, 2), dtype=torch.long),
                                         R.tau_par, R.lim_torque, R.banda_torque)
        self.assertAlmostEqual(out.item(), 0.0, places=6)


class TestInvolucrosDeTorque(unittest.TestCase):
    """Invólucros com env falso (SimpleNamespace). B1: `qfrc_actuator` do mjlab já tem só as 29 juntas
    (entity/data.py:415-422); o fatiamento `[:, 6:]` dava (E,23) e quebrava com (E,29) de `joint_vel`."""
    TODAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")

    def _env(self, tau, qd):
        robo = types.SimpleNamespace(data=types.SimpleNamespace(qfrc_actuator=tau, joint_vel=qd))
        return types.SimpleNamespace(device="cpu", num_envs=4, v3_tarefa=torch.tensor([0, 1, 2, 3]),
                                     v3_cur=types.SimpleNamespace(restricao=lambda: 1.0),  # 1: valor antigo dos termos
                                     scene={"robot": robo})

    def test_energia_com_29_juntas_da_forma_E(self):
        tau = torch.linspace(0.0, 1.0, 4 * len(JUNTAS), dtype=F).reshape(4, len(JUNTAS))
        qd = torch.full((4, len(JUNTAS)), 0.5, dtype=F)
        out = recompensas.energia(self._env(tau, qd), self.TODAS)
        self.assertEqual(tuple(out.shape), (4,))
        self.assertTrue(torch.allclose(out, (tau * qd).abs().sum(-1)))

    def test_cerca_torque_com_29_juntas_da_forma_E(self):
        # Uma junta solta a 90% do τ_max dá 0,5 (a mesma conta de TestCercas); o resto é zero.
        tau = torch.zeros(4, len(JUNTAS), dtype=F)
        tau[:, JUNTAS.index("left_knee_joint")] = 0.9 * _tau_max()[JUNTAS.index("left_knee_joint")]
        qd = torch.zeros(4, len(JUNTAS), dtype=F)
        out = recompensas.cerca_torque(self._env(tau, qd), self.TODAS)
        self.assertEqual(tuple(out.shape), (4,))
        self.assertTrue(torch.allclose(out, torch.full((4,), 0.5, dtype=F), atol=1e-5))


class TestLimiteJunta(unittest.TestCase):
    def test_excesso_a_97_5_por_cento_da_0_5(self):
        # Núcleo de limite_junta: potencial.excesso_de_junta (tabela da D5).
        q = t(0.975).reshape(1, 1)
        out = potencial.excesso_de_junta(q, t(0.0), t(1.0), R.limite_mole)
        self.assertAlmostEqual(out.item(), 0.5, places=5)


class TestRastreioUnitree(unittest.TestCase):
    """T3 (16/11) e T15 (24/11): `_rastreio` paga a tenda da fração entregue (velocidade filtrada `v_filt` do `twist`) no canal com
    comando e o núcleo da Unitree (velocidade instantânea) no canal sem comando; `normalizado=True` é a conta do sinal de
    progresso do CARREGAR e segue com a velocidade instantânea."""

    def _env(self, cmd, v_xy, wz, v_filt=None, g=1.0):
        # `v_filt` (vx, vy, wz): a velocidade filtrada; sem ela, vale a instantânea (a base em regime). `g`: o `g_marcha` do `twist`
        dados = types.SimpleNamespace(root_link_lin_vel_b=t(*v_xy, 0.0).reshape(1, 3),
                                      root_link_ang_vel_b=t(0.0, 0.0, wz).reshape(1, 3))
        twist = types.SimpleNamespace(v_filt=t(*(v_filt or (*v_xy, wz))).reshape(1, 3), g_marcha=t(g))
        return types.SimpleNamespace(
            command_manager=types.SimpleNamespace(get_command=lambda nome: t(*cmd).reshape(1, 3),
                                                  get_term=lambda nome: twist),
            scene={"robot": types.SimpleNamespace(data=dados)},
            v3_cur=types.SimpleNamespace(restricao=lambda: 1.0))

    def _r(self, cmd, v_filt, linear=True, v_inst=(0.0, 0.0, 0.0), g=1.0):
        env = self._env(cmd, v_inst[:2], v_inst[2], v_filt, g)
        return float(recompensas._rastreio(env, linear)[0])

    def test_parado_com_comando_paga_0_em_qualquer_alvo(self):
        # T15: o núcleo de hoje pagava 70 % parado com lado 0,3 e 37 % com giro 0,5; a tenda paga 0 nos dois canais
        for alvo in (0.1, 0.3, 0.5, 1.0):
            self.assertAlmostEqual(self._r((alvo, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.0, places=6, msg=f"linear {alvo}")
            self.assertAlmostEqual(self._r((0.0, alvo, 0.0), (0.0, 0.0, 0.0)), 0.0, places=6, msg=f"lado {alvo}")
            self.assertAlmostEqual(self._r((0.0, 0.0, alvo), (0.0, 0.0, 0.0), linear=False), 0.0, places=6, msg=f"giro {alvo}")

    def test_tenda_vale_1_no_alvo_e_0_no_dobro_e_nunca_negativa(self):
        for alvo in (0.3, 0.5, 1.0):
            self.assertAlmostEqual(self._r((alvo, 0.0, 0.0), (alvo, 0.0, 0.0)), 1.0, places=5)
            self.assertAlmostEqual(self._r((alvo, 0.0, 0.0), (0.5 * alvo, 0.0, 0.0)), 0.5, places=5)
            self.assertAlmostEqual(self._r((alvo, 0.0, 0.0), (1.5 * alvo, 0.0, 0.0)), 0.5, places=5)
            self.assertAlmostEqual(self._r((alvo, 0.0, 0.0), (2.0 * alvo, 0.0, 0.0)), 0.0, places=5)
            self.assertAlmostEqual(self._r((alvo, 0.0, 0.0), (3.0 * alvo, 0.0, 0.0)), 0.0, places=5)
            self.assertAlmostEqual(self._r((0.0, 0.0, alvo), (0.0, 0.0, alvo), linear=False), 1.0, places=5)
            self.assertAlmostEqual(self._r((0.0, 0.0, alvo), (0.0, 0.0, 2.0 * alvo), linear=False), 0.0, places=5)

    def test_canal_sem_comando_fica_com_o_nucleo_da_velocidade_instantanea(self):
        # parado vale 1 e a deriva é cobrada (exp(−v²/σ²), σ = 0,5); o filtro não entra aqui
        self.assertAlmostEqual(self._r((0.0, 0.0, 0.3), (0.0, 0.0, 0.0), linear=True), 1.0, places=5)
        self.assertAlmostEqual(self._r((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), linear=False), 1.0, places=5)
        self.assertAlmostEqual(self._r((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), v_inst=(0.5, 0.0, 0.0)), math.exp(-1), places=5)
        self.assertAlmostEqual(self._r((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), linear=False, v_inst=(0.0, 0.0, 0.5)), math.exp(-1), places=5)
        self.assertAlmostEqual(self._r((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), v_inst=(0.0, 0.0, 0.0)), 1.0, places=5)  # o filtro não conta

    def test_velocidade_perpendicular_ao_comando_reduz_o_termo(self):
        # comando 0,5 em x, entrega 0,5 em x e 0,5 em y: a tenda dá 1 e o perpendicular multiplica por exp(−0,25/0,25)
        self.assertAlmostEqual(self._r((0.5, 0.0, 0.0), (0.5, 0.5, 0.0)), math.exp(-1), places=5)
        # comando na diagonal: a parte perpendicular de (0,3; 0,3) sobre ĉ = (1, 0) é 0,3
        self.assertAlmostEqual(self._r((0.3, 0.0, 0.0), (0.3, 0.3, 0.0)), math.exp(-0.09 / 0.25), places=5)

    def test_re_lado_e_giro_negativo_dao_a_mesma_fracao_que_frente_e_giro_positivo(self):
        ref = self._r((0.4, 0.0, 0.0), (0.3, 0.0, 0.0))
        self.assertAlmostEqual(ref, 0.75, places=5)
        self.assertAlmostEqual(self._r((-0.4, 0.0, 0.0), (-0.3, 0.0, 0.0)), ref, places=5)
        self.assertAlmostEqual(self._r((0.0, 0.4, 0.0), (0.0, 0.3, 0.0)), ref, places=5)
        self.assertAlmostEqual(self._r((0.0, -0.4, 0.0), (0.0, -0.3, 0.0)), ref, places=5)
        self.assertAlmostEqual(self._r((0.0, 0.0, -0.4), (0.0, 0.0, -0.3), linear=False), ref, places=5)
        # o sentido errado paga 0
        self.assertAlmostEqual(self._r((-0.4, 0.0, 0.0), (0.3, 0.0, 0.0)), 0.0, places=5)
        self.assertAlmostEqual(self._r((0.0, 0.0, -0.4), (0.0, 0.0, 0.3), linear=False), 0.0, places=5)

    def test_g_da_marcha_multiplica_a_tenda_do_canal_com_comando(self):
        # T16: velocidade no alvo; acerto 0,5 (g 0) paga 0, acerto 1 (g 1) paga o valor de antes; 0,61 → g 0,22 e 0,91 → g 0,82
        for linear, cmd in ((True, (0.5, 0.0, 0.0)), (False, (0.0, 0.0, 0.5))):
            for q, esperado in ((0.5, 0.0), (1.0, 1.0), (0.61, 0.22), (0.91, 0.82)):
                g = min(max(2 * q - 1, 0.0), 1.0)
                self.assertAlmostEqual(self._r(cmd, cmd, linear=linear, g=g), esperado, places=5, msg=f"{cmd} q {q}")

    def test_g_nao_muda_o_canal_sem_comando_nem_o_normalizado(self):
        # T16: o canal sem comando (núcleo da velocidade instantânea) e o `normalizado` valem o mesmo com g = 0
        self.assertAlmostEqual(self._r((0.0, 0.0, 0.3), (0.0, 0.0, 0.0), g=0.0), 1.0, places=5)
        self.assertAlmostEqual(self._r((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), v_inst=(0.5, 0.0, 0.0), g=0.0), math.exp(-1), places=5)
        env = self._env((0.5, 0.0, 0.0), (0.5, 0.0), 0.0, g=0.0)
        self.assertAlmostEqual(float(recompensas._rastreio(env, True, normalizado=True)[0]), 1.0, places=5)

    def test_normalizado_parado_com_comando_vale_0(self):
        env = self._env((0.5, 0.0, 0.3), (0.0, 0.0), 0.0)
        self.assertAlmostEqual(float(recompensas._rastreio(env, True, normalizado=True)[0]), 0.0, places=5)
        self.assertAlmostEqual(float(recompensas._rastreio(env, False, normalizado=True)[0]), 0.0, places=5)

    def test_normalizado_usa_a_velocidade_instantanea(self):
        # cumprido: 1; erro de 0,3 m/s com comando 0,5: (exp(−0,09/0,25) − exp(−1)) / (1 − exp(−1)) do `rastreio_normalizado`
        env = self._env((0.5, 0.0, 0.0), (0.2, 0.0), 0.0, v_filt=(0.5, 0.0, 0.0))
        esperado = (math.exp(-0.09 / 0.25) - math.exp(-1)) / (1 - math.exp(-1))
        self.assertAlmostEqual(float(recompensas._rastreio(env, True, normalizado=True)[0]), esperado, places=5)


if __name__ == "__main__":
    unittest.main()
