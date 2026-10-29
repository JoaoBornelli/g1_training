"""Testes de g1_v3/alvos.py (B0 do plano, §6.3; S §8, §8.2, I §4.2). Escritos antes do código.

Convenções que o teste assume (o plano não fixa todas; ver a resposta de B0):
- todo tensor tem forma (E,), com E = 1 aqui; quaternion em wxyz; `indice` é long (E,) com a posição
  em ALVOS; `tempos` é (9,) na ordem de ALVOS.
"""
import math
import unittest

import torch

from g1_v3 import alvos
from g1_v3.knobs import ALVOS, ATIVO, caminho

K_AL = ATIVO.alvos
K_DET = ATIVO.detector
DEV = "cpu"
T = torch.float64
IDX = {nome: i for i, nome in enumerate(ALVOS)}  # P0=0 … P4=4, B1=5, B2=6, B3=7, B4=8


def t(*valores):
    return torch.tensor(list(valores), dtype=T)


def quat_eixo_z(graus):
    # rotação em torno de z, wxyz
    a = math.radians(graus) / 2
    return torch.tensor([[math.cos(a), 0.0, 0.0, math.sin(a)]], dtype=T)


def quat_eixo_x(graus):
    # rotação em torno de x, wxyz
    a = math.radians(graus) / 2
    return torch.tensor([[math.cos(a), math.sin(a), 0.0, 0.0]], dtype=T)


class TestReferencias(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ref = alvos.Referencias(str(caminho(ATIVO.contrato.referencias)), DEV)

    def _h(self, valor):
        return torch.tensor([valor], dtype=T)

    def test_nos_da_grade_batem_com_o_npz(self):
        # Nós (topo, meia) devolvem a grade do referencias.npz: grade_pelve/tronco/recuo, shape (6, 3).
        import numpy as np
        d = np.load(str(caminho(ATIVO.contrato.referencias)), allow_pickle=False)
        ref = self.ref
        for k, h in enumerate(d["topos"]):
            for j, a in enumerate(d["meias"]):
                ht = self._h(float(h))
                at = torch.tensor([float(a)], dtype=T)
                with self.subTest(topo=k, meia=j):
                    self.assertAlmostEqual(ref.pelve_z(ht, at).item(), d["grade_pelve"][k, j], places=6)
                    self.assertAlmostEqual(ref.tronco(ht, at).item(), d["grade_tronco"][k, j], places=6)
                    self.assertAlmostEqual(ref.recuo(ht, at).item(), d["grade_recuo"][k, j], places=6)

    def test_media_no_meio_de_dois_nos_de_altura(self):
        # Interpolação bilinear: no meio de dois nós de h, a pelve é a média dos dois nós.
        ref = self.ref
        a = ref.meias[0].reshape(1)
        for k in range(len(ref.topos) - 1):
            h0, h1 = ref.topos[k].item(), ref.topos[k + 1].item()
            meio = ref.pelve_z(self._h((h0 + h1) / 2), a)
            media = (ref.pelve_z(self._h(h0), a) + ref.pelve_z(self._h(h1), a)) / 2
            with self.subTest(k=k):
                self.assertAlmostEqual(meio.item(), media.item(), places=6)

    def test_media_no_meio_de_duas_meias_arestas(self):
        # No meio de duas meias arestas (0,07 e 0,10), a pelve é a média dos dois nós de a.
        ref = self.ref
        h = self._h(ref.topos[2].item())
        a0, a1 = ref.meias[0].item(), ref.meias[1].item()
        meio = ref.pelve_z(h, self._h((a0 + a1) / 2))
        media = (ref.pelve_z(h, self._h(a0)) + ref.pelve_z(h, self._h(a1))) / 2
        self.assertAlmostEqual(meio.item(), media.item(), places=6)

    def test_clamp_abaixo_de_zero_e_acima_de_055(self):
        # S §8.1: h fora da grade é preso nas bordas 0 e 0,55 m.
        ref = self.ref
        a = ref.meias[1].reshape(1)
        self.assertAlmostEqual(ref.pelve_z(self._h(-0.1), a).item(),
                               ref.pelve_z(self._h(0.0), a).item(), places=9)
        self.assertAlmostEqual(ref.pelve_z(self._h(0.7), a).item(),
                               ref.pelve_z(self._h(0.55), a).item(), places=9)

    def test_clamp_da_meia_aresta(self):
        # a abaixo de 0,07 ou acima de 0,13 é preso nas bordas da grade (§6.3 "com clamp na grade").
        ref = self.ref
        h = self._h(ref.topos[2].item())
        self.assertAlmostEqual(ref.pelve_z(h, self._h(0.0)).item(),
                               ref.pelve_z(h, self._h(ref.meias[0].item())).item(), places=9)
        self.assertAlmostEqual(ref.pelve_z(h, self._h(0.2)).item(),
                               ref.pelve_z(h, self._h(ref.meias[-1].item())).item(), places=9)


class TestFrameDaCaixa(unittest.TestCase):

    def test_caixa_girada_30_graus_dá_eixo_x_a_30(self):
        # PD-29: eixo x = coluna 0 da matriz; robô a 0°.
        m = alvos.frame_da_caixa(quat_eixo_z(30.0), t(0.0))
        ang = math.degrees(math.atan2(m[0, 1, 0].item(), m[0, 0, 0].item()))
        self.assertAlmostEqual(ang, 30.0, delta=0.01)

    def test_caixa_girada_50_graus_dá_eixo_x_a_menos_40(self):
        # Face mais perta do robô: 50° − 90° = −40°.
        m = alvos.frame_da_caixa(quat_eixo_z(50.0), t(0.0))
        ang = math.degrees(math.atan2(m[0, 1, 0].item(), m[0, 0, 0].item()))
        self.assertAlmostEqual(ang, -40.0, delta=0.01)


class TestInclinacaoDaCaixa(unittest.TestCase):

    def test_girada_90_em_x_tem_inclinacao_zero(self):
        # Simetria do cubo: uma face fica na vertical; o menor ângulo entre faces e vertical é 0.
        inc = alvos.inclinacao_da_caixa(quat_eixo_x(90.0))
        self.assertAlmostEqual(inc.item(), 0.0, delta=1e-4)

    def test_girada_20_em_x_tem_inclinacao_20(self):
        # Eixo x escolhido: em torno de z a inclinação seria 0. A spec não fixa o eixo.
        inc = alvos.inclinacao_da_caixa(quat_eixo_x(20.0))
        self.assertAlmostEqual(inc.item(), math.radians(20.0), delta=1e-4)


class TestPalmasAlvoCaixa(unittest.TestCase):

    def test_P0_com_a_010(self):
        # S §8.2: P0 = (−(a + p0_frente), ±(a + palma_face), 0) = (−0,15; ±0,135; 0).
        out = alvos.palmas_alvo_caixa(torch.tensor([IDX["P0"]]), torch.tensor([0.10], dtype=T), K_AL)
        self.assertEqual(tuple(out.shape), (1, 2, 3))
        for p in out[0].tolist():
            self.assertAlmostEqual(p[0], -0.15, delta=1e-6)
            self.assertAlmostEqual(abs(p[1]), 0.135, delta=1e-6)
            self.assertAlmostEqual(p[2], 0.0, delta=1e-6)
        self.assertLess(out[0, 0, 1].item() * out[0, 1, 1].item(), 0.0)  # uma palma de cada lado

    def test_B3_com_a_010(self):
        # S §8.2: B3 = (0, ±(a + palma_face + b3_afasta), 0) = (0; ±0,185; 0).
        out = alvos.palmas_alvo_caixa(torch.tensor([IDX["B3"]]), torch.tensor([0.10], dtype=T), K_AL)
        for p in out[0].tolist():
            self.assertAlmostEqual(p[0], 0.0, delta=1e-6)
            self.assertAlmostEqual(abs(p[1]), 0.185, delta=1e-6)
            self.assertAlmostEqual(p[2], 0.0, delta=1e-6)

    def test_alvos_sem_uso_dao_zero_e_nao_nan(self):
        # §6.3: P3, P4, B1, B2 e B4 não usam as palmas; devolvem 0, nunca NaN (§6.6).
        for nome in ("P3", "P4", "B1", "B2", "B4"):
            with self.subTest(alvo=nome):
                out = alvos.palmas_alvo_caixa(torch.tensor([IDX[nome]]), torch.tensor([0.10], dtype=T), K_AL)
                self.assertFalse(torch.isnan(out).any().item())
                self.assertEqual(out.abs().sum().item(), 0.0)


class TestCaixaAlvoW(unittest.TestCase):

    def test_P4_com_base_na_origem_e_rumo_zero(self):
        # S §8.2: P4 com z = chão + z_transporte (0,90), e xy de caixa_transporte_b(a).
        ref = alvos.Referencias(str(caminho(ATIVO.contrato.referencias)), DEV)
        a = torch.tensor([0.10], dtype=T)
        zeros3 = torch.zeros(1, 3, dtype=T)
        zero1 = torch.zeros(1, dtype=T)
        out = alvos.caixa_alvo_w(torch.tensor([IDX["P4"]]), zeros3, zeros3, zeros3,
                                 zero1, zero1, a, ref, K_AL)
        self.assertAlmostEqual(out[0, 2].item(), 0.90, delta=1e-9)
        xy = ref.caixa_transporte_b(a)[0, :2]
        self.assertTrue(torch.allclose(out[0, :2], xy, atol=1e-6))

    def test_alvos_sem_uso_dao_zero_e_nao_nan(self):
        # §6.3: P0, P1, P2, B3 e B4 não usam a caixa-alvo; devolvem 0, nunca NaN.
        ref = alvos.Referencias(str(caminho(ATIVO.contrato.referencias)), DEV)
        a = torch.tensor([0.10], dtype=T)
        zeros3 = torch.zeros(1, 3, dtype=T)
        zero1 = torch.zeros(1, dtype=T)
        for nome in ("P0", "P1", "P2", "B3", "B4"):
            with self.subTest(alvo=nome):
                out = alvos.caixa_alvo_w(torch.tensor([IDX[nome]]), zeros3, zeros3, zeros3,
                                         zero1, zero1, a, ref, K_AL)
                self.assertFalse(torch.isnan(out).any().item())
                self.assertEqual(out.abs().sum().item(), 0.0)


def _tempos():
    # tempo de permanência de cada alvo, na ordem de ALVOS (§8.2); 0 onde o plano não dá tempo.
    return torch.tensor([0.0, 0.0, 0.0, 0.0, K_AL.tempo_p4, 0.0,
                         K_AL.tempo_b2, K_AL.tempo_b3, K_AL.tempo_b4], dtype=T)


class TestAvanca(unittest.TestCase):

    def test_P4_avanca_no_passo_25_com_ok_sempre(self):
        # §8.2: tempo_p4 = 0,5 s; com passo de 0,02 s, são 25 passos. Antes do 25, fica em P4.
        idx = torch.tensor([IDX["P4"]])
        rel = torch.zeros(1, dtype=T)
        ok = torch.ones(1, dtype=torch.bool)
        for passo in range(1, 26):
            idx, rel, _ = alvos.avanca(idx, ok, rel, 0.02, _tempos())
            if passo < 25:
                self.assertEqual(idx.item(), IDX["P4"], f"avançou cedo no passo {passo}")
        self.assertEqual(idx.item(), IDX["B1"])

    def test_ok_falso_no_passo_10_zera_o_relogio(self):
        # Sem o reset, o avanço sairia no passo 25. Com o reset no 10, a contagem recomeça e o
        # avanço sai no passo 35 (25 passos ok depois do 10). Checa 25 (ainda não) e 35 (já).
        idx = torch.tensor([IDX["P4"]])
        rel = torch.zeros(1, dtype=T)
        for passo in range(1, 36):
            ok = torch.tensor([passo != 10])
            idx, rel, _ = alvos.avanca(idx, ok, rel, 0.02, _tempos())
            if passo == 25:
                self.assertEqual(idx.item(), IDX["P4"], "o relógio não zerou no passo 10")
        self.assertEqual(idx.item(), IDX["B1"])

    def test_P2_avanca_no_mesmo_passo(self):
        # §6.3 e B0: P2 tem tempo 0 e avança no primeiro passo com ok.
        idx = torch.tensor([IDX["P2"]])
        rel = torch.zeros(1, dtype=T)
        idx, _, _ = alvos.avanca(idx, torch.ones(1, dtype=torch.bool), rel, 0.02, _tempos())
        self.assertEqual(idx.item(), IDX["P3"])


class TestDetectaFim(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ref = alvos.Referencias(str(caminho(ATIVO.contrato.referencias)), DEV)

    def _entradas(self, vel_juntas):
        ref = self.ref
        pelve = torch.tensor([ref.pelve_de_pe], dtype=T)
        tronco = torch.zeros(1, dtype=T)  # desvio do tronco da vertical (rad); 0 na pose de pé
        palmas = ref.palmas_de_pe_b.unsqueeze(0).to(T)
        ref_pelve = torch.tensor([ref.pelve_de_pe], dtype=T)
        ref_palmas = ref.palmas_de_pe_b.unsqueeze(0).to(T)
        juntas = torch.full((1, 29), vel_juntas, dtype=T)
        return pelve, tronco, palmas, juntas, ref_pelve, ref_palmas

    def _roda(self, vel_juntas, passos):
        pelve, tronco, palmas, juntas, ref_pelve, ref_palmas = self._entradas(vel_juntas)
        rel = torch.zeros(1, dtype=T)
        avisos = []
        for _ in range(passos):
            aviso, rel = alvos.detecta_fim(pelve, tronco, palmas, juntas, ref_pelve, ref_palmas,
                                           rel, 0.02, K_DET)
            avisos.append(bool(aviso.reshape(-1)[0].item()))
        return avisos

    def test_juntas_com_0_1_avisa_no_passo_25(self):
        # I §4.2: estável por 0,5 s com |q̇| médio 0,1 < 0,2 → 25 passos de 0,02 s.
        avisos = self._roda(0.1, 26)
        self.assertFalse(avisos[23], "avisou antes de 0,5 s")
        self.assertTrue(avisos[24], "não avisou no passo 25")

    def test_juntas_com_0_3_nunca_avisa(self):
        # |q̇| médio 0,3 > 0,2 (juntas_paradas): não está parado, nunca avisa.
        self.assertFalse(any(self._roda(0.3, 100)))


if __name__ == "__main__":
    unittest.main()
