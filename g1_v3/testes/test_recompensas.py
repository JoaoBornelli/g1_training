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
    """13/11, M4: n_passo é o relógio do `feet_gait`. Pé E na `fase`, pé D na fase + 0,5; apoio com fase < 0,55."""

    def n(self, contato, fase=0.2, on=True):
        return recompensas.n_passo(torch.tensor([contato]), t(fase), torch.tensor([on]), ATIVO.comando.fase_apoio)[0].tolist()

    def test_fase_02_com_comando_e_apoio_e_d_em_balanco(self):
        # E: 0,2 < 0,55 apoio; D: 0,7 balanço
        self.assertEqual(self.n((True, False)), [1.0, 1.0])
        self.assertEqual(self.n((True, True)), [1.0, 0.0])

    def test_sem_comando_os_dois_pes_em_apoio(self):
        self.assertEqual(self.n((True, True), on=False), [1.0, 1.0])
        self.assertEqual(self.n((False, True), on=False), [0.0, 1.0])

    def test_fase_070_inverte_os_papeis(self):
        # E: 0,7 balanço; D: 0,2 apoio
        self.assertEqual(self.n((False, True), fase=0.7), [1.0, 1.0])


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


if __name__ == "__main__":
    unittest.main()
