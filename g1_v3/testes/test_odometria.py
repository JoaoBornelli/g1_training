"""Testes de g1_v3/estimador.py, classe Odometria (D2b do plano, §6.3; PD-18; S §13). Escritos antes do código.

Convenções: quaternion wxyz; o robô anda no plano, então só há giro em z. `deriva` é o dict das magnitudes
correntes com as chaves de §6.3 (odo, rumo, pos_giro, rumo_giro). Os valores do refino são o índice 1 de
`Estimador` (PD-18). Desvio de um eixo com passos de tamanho d: deriva·√(Σ|Δ|). Assim, 4 m a 0,0075 m/√m dão
1,5 cm; uma volta a 0,014 rad/√rad dá 2,0° (√(2π)·0,014 rad); a posição no giro, 0,008·√(2π) m = 2,0 cm.
"""
import math
import unittest

import torch

from g1_v3.estimador import Odometria
from g1_v3.knobs import ATIVO

K = ATIVO.estimador
R = 1  # índice do refino em `Estimador` (início, refino)
N = 10_000  # envs: o desvio medido tem erro relativo ~1/√(2N), abaixo da tolerância de 0,1 cm
DEV = "cpu"


def quat_de_yaw(psi):
    z = torch.zeros_like(psi)
    return torch.stack([torch.cos(psi / 2), z, z, torch.sin(psi / 2)], dim=-1)


def yaw_de(q):
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                       1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def deriva(odo=0.0, rumo=0.0, pos_giro=0.0, rumo_giro=0.0):
    return {"odo": odo, "rumo": rumo, "pos_giro": pos_giro, "rumo_giro": rumo_giro}


def novo(n):
    return Odometria(n, DEV, K, torch.Generator().manual_seed(0))


class OdometriaTest(unittest.TestCase):

    def test_sem_deriva_a_odometria_e_a_verdade(self):
        n = 4
        odo = novo(n)
        ids = torch.arange(n)
        psi = torch.zeros(n)
        pos = torch.zeros(n, 3)
        odo.reinicia(ids, pos, quat_de_yaw(psi))
        for k in range(60):  # gira para os dois lados e anda 1 cm por passo
            meio = psi + 0.03 * (k % 3 - 1) / 2
            pos = pos + torch.stack([0.01 * torch.cos(meio), 0.01 * torch.sin(meio), torch.zeros(n)], -1)
            psi = psi + 0.03 * (k % 3 - 1)
            odo.passo(pos, quat_de_yaw(psi), deriva())
        self.assertLess(float((odo.pos_w - pos).abs().max()), 1e-5)
        self.assertLess(float((yaw_de(odo.quat_w) - psi).abs().max()), 1e-5)

    def test_refino_4m_de_andar_erra_1_5_cm_por_eixo(self):
        # σ por eixo = 0,0075·√Σ|Δp| = 0,0075·√4 = 1,5 cm (PD-18, §13 refino)
        odo = novo(N)
        ids = torch.arange(N)
        pos = torch.zeros(N, 3)
        q = quat_de_yaw(torch.zeros(N))
        odo.reinicia(ids, pos, q)
        d = deriva(odo=K.deriva_odo[R])
        for _ in range(400):
            pos = pos + torch.tensor([0.01, 0.0, 0.0])
            odo.passo(pos, q, d)
        err = odo.pos_w - pos
        self.assertAlmostEqual(float(err[:, 0].std()), 0.015, delta=0.001)
        self.assertAlmostEqual(float(err[:, 1].std()), 0.015, delta=0.001)

    def test_giro_no_lugar_uma_volta_erra_2_graus_e_2_cm(self):
        # σ do rumo = 0,014·√(2π) rad ≈ 2,0°; σ da posição = 0,008·√(2π) m ≈ 2,0 cm (PD-18, refino)
        odo = novo(N)
        ids = torch.arange(N)
        pos = torch.zeros(N, 3)
        psi = torch.zeros(N)
        odo.reinicia(ids, pos, quat_de_yaw(psi))
        d = deriva(pos_giro=K.deriva_pos_giro[R], rumo_giro=K.deriva_rumo_giro[R])
        for _ in range(100):  # 100 passos de 0,0628 rad = uma volta, no lugar
            psi = psi + 2 * math.pi / 100
            odo.passo(pos, quat_de_yaw(psi), d)
        erro_rumo = yaw_de(odo.quat_w) - psi
        erro_rumo = torch.atan2(torch.sin(erro_rumo), torch.cos(erro_rumo))  # rumo em (−π, π]
        self.assertAlmostEqual(float(erro_rumo.std()) * 180 / math.pi, 2.0, delta=0.1)
        err = odo.pos_w - pos
        self.assertAlmostEqual(float(err[:, 0].std()), 0.02, delta=0.001)

    def test_so_translacao_os_termos_de_giro_nao_somam(self):
        # Δψ = 0 em todo passo: a variância dos termos de giro é zero, mesmo com deriva alta
        odo = novo(N)
        ids = torch.arange(N)
        pos = torch.zeros(N, 3)
        q = quat_de_yaw(torch.zeros(N))
        odo.reinicia(ids, pos, q)
        d = deriva(pos_giro=0.5, rumo_giro=0.5)
        for _ in range(100):
            pos = pos + torch.tensor([0.01, 0.0, 0.0])
            odo.passo(pos, q, d)
        self.assertLess(float((odo.pos_w - pos).abs().max()), 1e-6)
        self.assertLess(float(yaw_de(odo.quat_w).abs().max()), 1e-6)

    def test_delta_b_sem_deriva_e_o_incremento_no_frame_da_base(self):
        # Δ verdadeiro no frame da base do passo anterior: R(−ψ₀)·Δp; dψ e dz verdadeiros (§6.3)
        n = 4
        odo = novo(n)
        ids = torch.arange(n)
        psi0 = 0.4
        pos0 = torch.zeros(n, 3)
        odo.reinicia(ids, pos0, quat_de_yaw(torch.full((n,), psi0)))
        dp = torch.tensor([0.03, -0.02, 0.01])
        pos1 = pos0 + dp
        odo.passo(pos1, quat_de_yaw(torch.full((n,), psi0 + 0.1)), deriva())
        dxy, dpsi, dz = odo.delta_b()
        c, s = math.cos(psi0), math.sin(psi0)
        esperado = torch.tensor([c * dp[0] + s * dp[1], -s * dp[0] + c * dp[1]])
        self.assertLess(float((dxy - esperado).abs().max()), 1e-5)
        self.assertLess(float((dpsi - 0.1).abs().max()), 1e-5)
        self.assertLess(float((dz - dp[2]).abs().max()), 1e-5)

    def test_rumo_estimado_10_graus_errado_nao_entra_no_delta(self):
        # Item 7 (29/10). `reinicia` recebe a pose verdadeira; depois o `quat_w` estimado é girado 10° em yaw
        # (atributo público, §6.3). A odometria guarda a verdade à parte para tirar o incremento.
        n = 4
        odo = novo(n)
        ids = torch.arange(n)
        psi0 = 0.4
        dez = math.radians(10)
        odo.reinicia(ids, torch.zeros(n, 3), quat_de_yaw(torch.full((n,), psi0)))
        odo.quat_w[:] = quat_de_yaw(torch.full((n,), psi0 + dez))  # estimativa errada
        dp = torch.tensor([0.03, -0.02, 0.01])
        odo.passo(torch.zeros(n, 3) + dp, quat_de_yaw(torch.full((n,), psi0 + 0.1)), deriva())
        # (a) Δ = incremento verdadeiro no frame da base VERDADEIRA (R(ψ₀)^T·Δp); o erro não entra
        dxy, dpsi, dz = odo.delta_b()
        c, s = math.cos(psi0), math.sin(psi0)
        esperado = torch.tensor([c * dp[0] + s * dp[1], -s * dp[0] + c * dp[1]])
        self.assertLess(float((dxy - esperado).abs().max()), 1e-5)
        self.assertLess(float((dpsi - 0.1).abs().max()), 1e-5)
        self.assertLess(float((dz - dp[2]).abs().max()), 1e-5)
        # (b) a posição absoluta estimada anda na direção girada 10°: R(10°)·Δp no plano (§6.3, D2b)
        c10, s10 = math.cos(dez), math.sin(dez)
        girado = torch.tensor([c10 * dp[0] - s10 * dp[1], s10 * dp[0] + c10 * dp[1], dp[2]])
        self.assertLess(float((odo.pos_w - girado).abs().max()), 1e-5)

    def test_reinicia_zera_o_delta(self):
        n = 4
        odo = novo(n)
        ids = torch.arange(n)
        q = quat_de_yaw(torch.zeros(n))
        odo.reinicia(ids, torch.zeros(n, 3), q)
        odo.passo(torch.full((n, 3), 0.05), q, deriva(odo=K.deriva_odo[R]))
        odo.reinicia(ids, torch.zeros(n, 3), q)
        dxy, dpsi, dz = odo.delta_b()
        for t in (dxy, dpsi, dz):
            self.assertEqual(float(t.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
