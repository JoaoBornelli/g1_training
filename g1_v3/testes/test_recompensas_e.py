"""Testes da tarefa E4 do plano (§12.1, `recompensas.py` da fase E): distancia_do_alvo, portoes_do_alvo,
phi_do_alvo, n_progresso, n_qualidade, aplica_qualidade e n_perda_caixa (§6.3, §6.2 `Alvos`/`Recompensa`).
Escritos antes do código: as funções ainda não existem, então os testes falham com AttributeError até a E4.

As tabelas de passo a passo usam o caminho roteirizado do item 1 da Parte 0 (`parte0/item1_phi.py`, com
`comum.trajetoria_pegar` e `trajetoria_botar`). Esse módulo importa `comum`, que importa `mujoco` e `scipy`,
e `referencias()` faz FK com o g1.xml: é importação e cinemática, não simulador nem env.
Ângulos em graus nas tabelas; comprimentos em metros.
"""
import types
import unittest
from unittest import mock

import numpy as np
import torch

from g1_v3 import alvos, potencial, recompensas
from g1_v3.knobs import ATIVO, TAREFAS
from g1_v3.parte0 import comum as C
from g1_v3.parte0 import item1_phi as I1

K = ATIVO.alvos
R = ATIVO.recompensa
A = C.A_CAIXA
DT = C.DT
TOL = 1e-5
POUSO = [5, 6]  # B1 e B2: k × fator_pouso


def T(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


class _Caminho:
    """Entradas do núcleo montadas como o invólucro montaria, a partir do caminho do item 1."""

    def __init__(self, tr):
        tr["incl"] = np.zeros(len(tr["t"]))
        self.tr = tr
        self.refs = I1.referencias()
        self.pts = I1.pontos_de(tr)
        self.res = I1.avalia(tr, self.refs, self.pts)  # referência: d, σ e Φ do item 1
        idx = tr["indice"]
        self.idx = T(idx).long()
        self.n = len(idx)
        self.caixa = tr["caixa"]
        self.palmas_c = tr["palmas"] - tr["caixa"][:, None, :]  # caixa sem giro: frame da caixa = mundo
        self.caixa_alvo = self.caixa[self.res["fim"][self.res["trecho"]]]  # alvo = fim do trecho corrente
        self.a = T(np.full(self.n, A))
        self.dpelve = tr["pelve"] - self.refs["pelve_pe"]
        self.dtronco = np.degrees(tr["tronco"] - self.refs["tronco"][idx])

    def distancia(self):
        return recompensas.distancia_do_alvo(self.idx, T(self.palmas_c), T(self.caixa), T(self.caixa_alvo),
                                             T(self.dpelve), T(self.dtronco), self.a, K)

    def phi(self):
        z = self.caixa[:, 2]
        v = np.diff(z, prepend=z[0]) / DT
        pouso = potencial.fator_pouso(T(v), T(z - A - C.TOPO_BOTAR), R.sigma_pouso[0], R.altura_pouso,
                                      R.altura_pouso_inicio).numpy()
        fator = np.where(np.isin(self.tr["indice"], POUSO), pouso, 1.0)
        xy = self.caixa[:, :2]
        contatos = np.linalg.norm(self.palmas_c - I1.ALVO_P1, axis=2) <= K.tol_palma  # contato: palma no P1
        rec = self.refs["recuo"][self.tr["indice"]]
        portoes = recompensas.portoes_do_alvo(
            self.idx,
            T(np.linalg.norm(xy - self.pts["caixa0"][:2], axis=1)),
            T(np.linalg.norm(xy - self.pts["apoio"][:2], axis=1)),
            T(self.tr["incl"]), K.incl_inicio_graus, T(contatos),
            T(np.abs(self.dtronco)), T(np.abs(np.linalg.norm(xy, axis=1) - np.nan_to_num(rec))),
            torch.zeros(self.n, dtype=torch.float64), torch.zeros(self.n, dtype=torch.bool), self.a, K)
        return recompensas.phi_do_alvo(self.idx, T(self.res["d"]), T(self.res["sigma"]), T(fator), portoes)


class TestPassoAPassoComItem1(unittest.TestCase):
    """Distância e Φ do caminho roteirizado, passo a passo, contra `item1_phi.avalia` (tolerância 1e-5)."""

    @classmethod
    def setUpClass(cls):
        cls.pegar = _Caminho(C.trajetoria_pegar())
        cls.botar = _Caminho(C.trajetoria_botar())

    def test_distancia_pegar_bate_com_item1(self):
        # Teste E4: distancia_do_alvo no caminho do PEGAR bate, passo a passo, com item1_phi (1e-5)
        c = self.pegar
        np.testing.assert_allclose(c.distancia().numpy(), c.res["d"], atol=TOL, rtol=0)

    def test_distancia_botar_bate_com_item1(self):
        # Teste E4: distancia_do_alvo no caminho do BOTAR bate, passo a passo, com item1_phi (1e-5)
        c = self.botar
        np.testing.assert_allclose(c.distancia().numpy(), c.res["d"], atol=TOL, rtol=0)

    def test_phi_pegar_bate_com_item1(self):
        # Teste E4: phi_do_alvo no caminho do PEGAR bate, passo a passo, com item1_phi["phi"] (1e-5)
        c = self.pegar
        np.testing.assert_allclose(c.phi().numpy(), c.res["phi"], atol=TOL, rtol=0)

    def test_phi_botar_bate_com_item1(self):
        # Teste E4: phi_do_alvo no caminho do BOTAR bate, passo a passo, com item1_phi["phi"] (1e-5)
        c = self.botar
        np.testing.assert_allclose(c.phi().numpy(), c.res["phi"], atol=TOL, rtol=0)


class TestDistancia(unittest.TestCase):
    def test_palmas_sobre_o_alvo_da_zero(self):
        # Borda E4: P0 com as palmas no alvo do frame da caixa → 0
        a = torch.tensor([A], dtype=torch.float64)
        palmas = alvos.palmas_alvo_caixa(torch.tensor([0]), a, K)  # (1, 2, 3)
        d = recompensas.distancia_do_alvo(torch.tensor([0]), palmas, torch.zeros(1, 3, dtype=torch.float64),
                                          torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
                                          torch.zeros(1, dtype=torch.float64), a, K)
        self.assertAlmostEqual(float(d[0]), 0.0, places=6)

    def test_p3_mede_a_caixa_ao_alvo(self):
        # Borda E4: P3 usa ‖caixa − alvo‖ (0,30 m em x → 0,30)
        z = torch.zeros(1, dtype=torch.float64)
        d = recompensas.distancia_do_alvo(torch.tensor([3]), torch.zeros(1, 2, 3, dtype=torch.float64),
                                          torch.tensor([[0.30, 0.0, 0.0]], dtype=torch.float64),
                                          torch.zeros(1, 3, dtype=torch.float64), z, z,
                                          torch.tensor([A], dtype=torch.float64), K)
        self.assertAlmostEqual(float(d[0]), 0.30, places=6)

    def test_b4_normaliza_pelve_e_tronco(self):
        # Borda E4: B4 = √((Δpelve/0,03)² + (Δtronco/5°)²); Δpelve 3 cm com Δtronco 0 → 1
        z = torch.zeros(1, dtype=torch.float64)
        d = recompensas.distancia_do_alvo(torch.tensor([8]), torch.zeros(1, 2, 3, dtype=torch.float64),
                                          torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64),
                                          torch.tensor([0.03], dtype=torch.float64), z,
                                          torch.tensor([A], dtype=torch.float64), K)
        self.assertAlmostEqual(float(d[0]), 1.0, places=6)

    def test_indice_menos_um_da_zero(self):
        # Borda E4: −1 (nenhum alvo) → 0, nunca NaN
        z = torch.zeros(1, dtype=torch.float64)
        d = recompensas.distancia_do_alvo(torch.tensor([-1]), torch.zeros(1, 2, 3, dtype=torch.float64),
                                          torch.full((1, 3), 0.5, dtype=torch.float64),
                                          torch.zeros(1, 3, dtype=torch.float64), z, z,
                                          torch.tensor([A], dtype=torch.float64), K)
        self.assertEqual(float(d[0]), 0.0)


class TestProgresso(unittest.TestCase):
    def test_pairar_soma_nao_positivo(self):
        # Teste E4: pairar (Φ constante) soma ≤ 0 em n_progresso com γ < 1
        phi = torch.full((1,), 0.6, dtype=torch.float64)
        fora = torch.zeros(1, dtype=torch.bool)
        b = torch.zeros(1, dtype=torch.float64)
        soma = sum(float(recompensas.n_progresso(phi, phi, fora, b, fora, fora, R.progresso, R.gamma)[0])
                   for _ in range(100))
        self.assertLessEqual(soma, 1e-9)

    def test_troca_sem_avanco_paga_zero(self):
        # Teste E11 (correção 2): troca sem avanço não paga, nem o ΔΦ nem o B_alvo[B4]
        nao = torch.zeros(1, dtype=torch.bool)
        sim = torch.ones(1, dtype=torch.bool)
        r = recompensas.n_progresso(torch.tensor([0.9]), torch.tensor([0.1]), nao, torch.tensor([0.5]), sim, nao,
                                    1.0, R.gamma)
        self.assertEqual(float(r[0]), 0.0)

    def test_troca_com_fecho_do_b4_sem_perda_paga_b_alvo(self):
        # Teste E11 (R3): o fecho do B4 (troca com avanço, sem `v3_perdeu`) paga o B_alvo[B4]; o ΔΦ vai a zero (PD-36)
        nao = torch.zeros(1, dtype=torch.bool)
        sim = torch.ones(1, dtype=torch.bool)
        r = recompensas.n_progresso(torch.tensor([0.9]), torch.tensor([0.1]), sim, torch.tensor([0.5]), sim, nao,
                                    1.0, R.gamma)
        self.assertAlmostEqual(float(r[0]), 0.5, places=6)

    def test_troca_com_fecho_do_b4_depois_da_perda_paga_zero(self):
        # Teste E11 (correção 2; S §10): a volta à pose de pé depois da perda não paga o B_alvo[B4] (a perda devolve)
        sim = torch.ones(1, dtype=torch.bool)
        r = recompensas.n_progresso(torch.tensor([0.9]), torch.tensor([0.1]), sim, torch.tensor([0.5]), sim, sim,
                                    1.0, R.gamma)
        self.assertEqual(float(r[0]), 0.0)

    def test_entrada_no_alvo_paga_zero(self):
        # Teste E4: o passo com `phi_ant` = −1 (sem alvo antes) é a entrada no alvo e paga 0, sem o salto de Φ + 1
        nao = torch.tensor([False])
        r = recompensas.n_progresso(torch.tensor([0.5]), torch.tensor([-1.0]), nao,
                                    torch.tensor([0.0]), nao, nao, 1.0, R.gamma)
        self.assertEqual(float(r[0]), 0.0)

    def test_avanco_usa_phi_do_alvo_velho_no_estado_novo_mais_b_alvo(self):
        # Teste E4 (D1 da Parte 0): no passo do avanço, Φ(s′, i) do alvo VELHO + B_alvo; 0,99·0,8 − 0,6 + 0,5
        nao = torch.tensor([False])
        r = recompensas.n_progresso(torch.tensor([0.8]), torch.tensor([0.6]), torch.tensor([True]),
                                    torch.tensor([0.5]), nao, nao, 1.0, 0.99)
        self.assertAlmostEqual(float(r[0]), 0.692, places=5)


class TestQualidade(unittest.TestCase):
    def test_qualidade_zero_com_caixa_a_45_graus(self):
        # Teste E4: n_qualidade = 0 com a caixa a 45° e incl_lim 30° (rampa com Δ 15°)
        q = recompensas.n_qualidade(torch.tensor([0.0]), torch.tensor([45.0]), K.incl_inicio_graus,
                                    torch.tensor([[True, True]]), K)
        self.assertAlmostEqual(float(q[0]), 0.0, places=6)

    def test_qualidade_um_com_caixa_nivelada_e_duas_maos(self):
        # Borda E4: d 0, inclinação dentro do limite e as duas mãos → 1
        q = recompensas.n_qualidade(torch.tensor([0.0]), torch.tensor([30.0]), K.incl_inicio_graus,
                                    torch.tensor([[True, True]]), K)
        self.assertAlmostEqual(float(q[0]), 1.0, places=6)

    def test_aplica_qualidade_mantem_negativo_e_escala_positivo(self):
        # Teste E4: aplica_qualidade deixa o rastreio negativo como está e escala o positivo (q·relu(r) + min(r,0))
        r = torch.tensor([-0.3, 0.4])
        q = torch.tensor([0.5, 0.5])
        out = recompensas.aplica_qualidade(r, q)
        self.assertAlmostEqual(float(out[0]), -0.3, places=6)
        self.assertAlmostEqual(float(out[1]), 0.2, places=6)


class TestPerdaCaixa(unittest.TestCase):
    def test_perda_sem_devolver_b_pagos(self):
        # Teste E4: n_perda_caixa com devolve False: só o custo, no passo da perda
        out = recompensas.n_perda_caixa(torch.tensor([True, False]), torch.tensor([0.5, 0.5]),
                                        R.perda_caixa, False)
        self.assertAlmostEqual(float(out[0]), R.perda_caixa, places=4)
        self.assertEqual(float(out[1]), 0.0)

    def test_perda_devolvendo_b_pagos(self):
        # Teste E4: n_perda_caixa com devolve True: custo + b_pagos, no passo da perda
        out = recompensas.n_perda_caixa(torch.tensor([True, False]), torch.tensor([0.5, 0.5]),
                                        R.perda_caixa, True)
        self.assertAlmostEqual(float(out[0]), R.perda_caixa + 0.5, places=4)
        self.assertEqual(float(out[1]), 0.0)


class TestPhiDoAlvo(unittest.TestCase):
    def test_indice_menos_um_da_zero(self):
        # Borda E4: phi_do_alvo com −1 → 0, nunca NaN (§6.6)
        portoes = {"desloc": torch.ones(1, dtype=torch.float64), "incl": torch.ones(1, dtype=torch.float64)}
        phi = recompensas.phi_do_alvo(torch.tensor([-1]), torch.tensor([0.5], dtype=torch.float64),
                                      torch.tensor([0.1], dtype=torch.float64),
                                      torch.ones(1, dtype=torch.float64), portoes)
        self.assertEqual(float(phi[0]), 0.0)


class TestPortoes(unittest.TestCase):
    """portoes_do_alvo: cada termo em [0, 1]; "maos" e "com" pelas regras do plano (PD-19, PD-35)."""

    def _p(self, idx, contatos=((True, True),), dois_pes=False, com=0.0):
        """Portões de um alvo com os outros desvios em 0 (dentro de todas as faixas)."""
        def um(v):
            return torch.tensor([v], dtype=torch.float64)
        return recompensas.portoes_do_alvo(
            torch.tensor([idx]), um(0.0), um(0.0), um(0.0), K.incl_inicio_graus,
            torch.tensor(contatos), um(0.0), um(0.0), um(com), torch.tensor([dois_pes]),
            torch.tensor([A], dtype=torch.float64), K)

    def test_maos_meio_com_um_contato_no_p3(self):
        # Teste E4: maos = 0,5 com um contato no P3 (P2–B2: contatos.sum/2)
        self.assertAlmostEqual(float(self._p(3, [(True, False)])["maos"][0]), 0.5, places=6)

    def test_maos_um_fora_de_p2_b2(self):
        # Teste E4: maos = 1 fora de P2–B2 (P0 e B3), mesmo sem contato
        self.assertEqual(float(self._p(0, [(False, False)])["maos"][0]), 1.0)
        self.assertEqual(float(self._p(7, [(False, False)])["maos"][0]), 1.0)

    def test_com_um_sem_os_dois_pes(self):
        # Teste E4: com = 1 sem os dois pés no chão, mesmo com desvio do CoM fora da faixa
        self.assertEqual(float(self._p(4, com=0.5)["com"][0]), 1.0)

    def test_com_zero_com_os_dois_pes_e_desvio_fora_da_faixa(self):
        # Borda E4: com os dois pés e desvio = faixa + Δ (0,06 m) → 0
        self.assertAlmostEqual(float(self._p(4, dois_pes=True, com=K.faixa_com + K.delta_com)["com"][0]),
                               0.0, places=6)


TODAS_TAREFAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")


def _env_progresso(idx, av, p4_fechado, phi_ant, sigma, sigma_velho):
    """Env falso (SimpleNamespace) com os buffers que `recompensas.progresso` lê. Um passo, E = 1."""
    return types.SimpleNamespace(
        device="cpu", v3_tarefa=torch.tensor([TAREFAS.index("PEGAR")]),
        v3_alvo_idx=torch.tensor([idx]), v3_avancou=torch.tensor([av]), v3_p4_fechado=torch.tensor([p4_fechado]),
        v3_phi=torch.tensor([phi_ant], dtype=torch.float32), v3_sigma=torch.tensor([sigma], dtype=torch.float32),
        v3_sigma_velho=torch.tensor([sigma_velho], dtype=torch.float32), v3_b_pagos=torch.zeros(1),
        v3_perdeu=torch.zeros(1, dtype=torch.bool))


class TestInclLim(unittest.TestCase):
    """B1 (revisão E11): `incl_lim` lia `K`, que não existia no módulo, e o primeiro reset caía com NameError."""

    def test_incl_lim_roda_e_desce_pelo_degrau(self):
        env = types.SimpleNamespace(v3_degrau=torch.tensor([0, ATIVO.balanceador.degraus]))
        out = recompensas.incl_lim(env).double().numpy()
        np.testing.assert_allclose(out, [K.incl_inicio_graus, K.incl_fim_graus], atol=1e-4)


class TestProgressoNoFechoDoP4(unittest.TestCase):
    """S2 (revisão E11): no fecho do P4 a cadeia põe `v3_avancou` = True e mantém o índice em P4. O progresso tem
    de pagar Φ(P4) e B_alvo[P4]; o código velho pagava o alvo P3 (`idx − 1`)."""

    def test_fecho_do_p4_paga_b_alvo_do_p4(self):
        # Φ(s′, P4) = 0,8; Φ_ant = 0,6 (Φ(s, P4)). Esperado: w·(γ·0,8 − 0,6) + B_alvo[P4].
        phi = [0.0] * 9  # um Φ por alvo (ALVOS); só P4 importa aqui, e P3 fica em 0 (a faixa do tronco da pega)
        phi[alvos.P4] = 0.8

        def phi_por_alvo(env, indice, sigma=None):
            tab = torch.tensor(phi, dtype=torch.float32)
            return torch.where(indice >= 0, tab[indice.clamp(min=0)], torch.zeros(indice.shape))

        env = _env_progresso(alvos.P4, av=True, p4_fechado=True, phi_ant=0.6, sigma=0.3, sigma_velho=0.3)
        with mock.patch.object(recompensas, "_phi_env", phi_por_alvo):
            r = recompensas.progresso(env, TODAS_TAREFAS)
        esperado = R.progresso * (R.gamma * 0.8 - 0.6) + R.b_alvo[alvos.P4]
        self.assertAlmostEqual(float(r[0]), esperado, places=4)


class TestProgressoSigmaDoAlvoVelho(unittest.TestCase):
    """S3 (revisão E11): no avanço P1 → P2, Φ(s′, P1) usa o σ do alvo VELHO (`v3_sigma_velho`), e não o σ novo."""

    def test_avanco_p1_para_p2_mede_o_velho_com_sigma_velho(self):
        # σ velho (P1) = 0,15; σ novo (P2) = 0,02. Φ(s′, P1) = σ usado. Esperado: w·(γ·0,15 − 0,5) + B_alvo[P1].
        def phi_pelo_sigma(env, indice, sigma=None):
            return env.v3_sigma if sigma is None else sigma

        env = _env_progresso(alvos.P2, av=True, p4_fechado=False, phi_ant=0.5, sigma=0.02, sigma_velho=0.15)
        with mock.patch.object(recompensas, "_phi_env", phi_pelo_sigma):
            r = recompensas.progresso(env, TODAS_TAREFAS)
        esperado = R.progresso * (R.gamma * 0.15 - 0.5) + R.b_alvo[alvos.P1]
        self.assertAlmostEqual(float(r[0]), esperado, places=4)


class TestCercaRecuoNoP4(unittest.TestCase):
    """S7 (revisão E11): no PEGAR e no BOTAR com o alvo P4, a `cerca_recuo` mede contra a referência do transporte
    (`recuo_carregar`, 0,25 m), e não contra `ref.recuo` (0,40 m, o recuo da pega). A caixa está a 0,25 m: custo 0."""

    def _env(self, tarefa):
        robo = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=torch.zeros(1, 3)))
        caixa = types.SimpleNamespace(
            data=types.SimpleNamespace(root_link_pos_w=torch.tensor([[R.recuo_carregar, 0.0, 0.0]])))

        class Referencia:
            def recuo(self, topo, a):
                return torch.full((1,), 0.40)

            def caixa_transporte_b(self, a):
                return torch.tensor([[R.recuo_carregar, 0.0, 0.0]])

        return types.SimpleNamespace(
            device="cpu", scene={"robot": robo, "box": caixa}, v3_tarefa=torch.tensor([TAREFAS.index(tarefa)]),
            v3_alvo_idx=torch.tensor([alvos.P4]), v3_perdeu=torch.zeros(1, dtype=torch.bool),
            v3_topo=torch.zeros(1), v3_topo_destino=torch.zeros(1), v3_meia=torch.full((1,), A),
            v3_ref=Referencia())

    def test_cerca_recuo_no_p4_mede_contra_o_transporte(self):
        for tarefa in ("PEGAR", "BOTAR"):
            with self.subTest(tarefa=tarefa):
                r = recompensas.cerca_recuo(self._env(tarefa), ("PEGAR", "CARREGAR", "BOTAR"))
                self.assertAlmostEqual(float(r[0]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
