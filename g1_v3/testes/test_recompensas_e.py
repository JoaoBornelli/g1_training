"""Testes da tarefa E4 do plano (§12.1, `recompensas.py` da fase E): distancia_do_alvo, portoes_do_alvo,
phi_do_alvo, n_progresso (renda por segundo e tabela de etapas), n_qualidade, aplica_qualidade e n_perda_caixa (§6.3, §6.2 `Alvos`/`Recompensa`).
Escritos antes do código: as funções ainda não existem, então os testes falham com AttributeError até a E4.

As tabelas de passo a passo usam o caminho roteirizado do item 1 da Parte 0 (`parte0/item1_phi.py`, com
`comum.trajetoria_pegar` e `trajetoria_botar`). Esse módulo importa `comum`, que importa `mujoco` e `scipy`,
e `referencias()` faz FK com o g1.xml: é importação e cinemática, não simulador nem env.
Ângulos em graus nas tabelas; comprimentos em metros.
"""
import math
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
                                             T(self.dpelve), T(self.dtronco), torch.zeros(self.n, dtype=torch.float64),
                                             torch.zeros(self.n, dtype=torch.float64), self.a, K)

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
            T(self.tr["incl"]), K.incl_inicio_graus, T(contatos.sum(1) / 2.0),
            T(np.maximum(self.dtronco, 0)),  # F9 P2: o portão do tronco é de um lado só; o chamador passa relu(incl − IK)
            T(np.abs(np.linalg.norm(xy, axis=1) - np.nan_to_num(rec))),
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
                                          torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
                                          torch.zeros(1, dtype=torch.float64), a, K)
        self.assertAlmostEqual(float(d[0]), 0.0, places=6)

    def test_p3_mede_a_caixa_ao_alvo(self):
        # Borda E4: P3 usa ‖caixa − alvo‖ (0,30 m em x → 0,30)
        z = torch.zeros(1, dtype=torch.float64)
        d = recompensas.distancia_do_alvo(torch.tensor([3]), torch.zeros(1, 2, 3, dtype=torch.float64),
                                          torch.tensor([[0.30, 0.0, 0.0]], dtype=torch.float64),
                                          torch.zeros(1, 3, dtype=torch.float64), z, z, z, z,
                                          torch.tensor([A], dtype=torch.float64), K)
        self.assertAlmostEqual(float(d[0]), 0.30, places=6)

    def test_b4_normaliza_pelve_e_tronco(self):
        # Borda E4: B4 = √((Δpelve/0,03)² + (Δtronco/5°)² + (dpalmas/0,05)²); Δpelve 3 cm com o resto 0 → 1
        z = torch.zeros(1, dtype=torch.float64)
        d = recompensas.distancia_do_alvo(torch.tensor([8]), torch.zeros(1, 2, 3, dtype=torch.float64),
                                          torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64),
                                          torch.tensor([0.03], dtype=torch.float64), z, z, z,
                                          torch.tensor([A], dtype=torch.float64), K)
        self.assertAlmostEqual(float(d[0]), 1.0, places=6)

    def _d_b(self, indice, dpalmas=0.0, deficit=0.0, caixa_x=0.0):
        # `distancia_do_alvo` no índice dado, com a caixa em (caixa_x, 0, 0), o alvo na origem e pelve e tronco no alvo
        z = torch.zeros(1, dtype=torch.float64)
        return float(recompensas.distancia_do_alvo(
            torch.tensor([indice]), torch.zeros(1, 2, 3, dtype=torch.float64),
            torch.tensor([[caixa_x, 0.0, 0.0]], dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64), z, z,
            torch.tensor([dpalmas], dtype=torch.float64), torch.tensor([deficit], dtype=torch.float64),
            torch.tensor([A], dtype=torch.float64), K)[0])

    def test_b4_cresce_com_as_palmas_longe_do_tronco(self):
        # 13/11, A9: com pelve e tronco no alvo, d = dpalmas/tol_palmas_b4 (0,05 m → 1; 0,10 m → 2), e cresce com dpalmas
        self.assertAlmostEqual(self._d_b(8, dpalmas=0.0), 0.0, places=6)
        self.assertAlmostEqual(self._d_b(8, dpalmas=K.tol_palmas_b4), 1.0, places=6)
        self.assertAlmostEqual(self._d_b(8, dpalmas=2 * K.tol_palmas_b4), 2.0, places=6)

    def test_b2_soma_o_deficit_de_apoio_a_distancia_da_caixa(self):
        # 13/11, A3: déficit 0 dá d_caixa; déficit 1 dá hypot(d_caixa, escala_apoio_b2 = 0,10)
        self.assertAlmostEqual(self._d_b(6, deficit=0.0, caixa_x=0.05), 0.05, places=6)
        self.assertAlmostEqual(self._d_b(6, deficit=1.0, caixa_x=0.05), math.hypot(0.05, K.escala_apoio_b2), places=6)
        self.assertAlmostEqual(self._d_b(6, deficit=1.0, caixa_x=0.0), K.escala_apoio_b2, places=6)

    def test_deficit_e_dpalmas_so_valem_no_b2_e_no_b4(self):
        # 13/11, A3/A9: no B1 (5) e no P3 (3) a distância segue a da caixa, com déficit 1 e dpalmas altos
        for indice in (3, 5):
            self.assertAlmostEqual(self._d_b(indice, dpalmas=1.0, deficit=1.0, caixa_x=0.30), 0.30, places=6)

    def test_indice_menos_um_da_zero(self):
        # Borda E4: −1 (nenhum alvo) → 0, nunca NaN
        z = torch.zeros(1, dtype=torch.float64)
        d = recompensas.distancia_do_alvo(torch.tensor([-1]), torch.zeros(1, 2, 3, dtype=torch.float64),
                                          torch.full((1, 3), 0.5, dtype=torch.float64),
                                          torch.zeros(1, 3, dtype=torch.float64), z, z, z, z,
                                          torch.tensor([A], dtype=torch.float64), K)
        self.assertEqual(float(d[0]), 0.0)


def _buffers_m5(n):
    """Buffers e relógio que o `progresso` lê desde a M5 (13/11): h cheio, melhor Φ zerado, comando zero."""
    return dict(v3_h=torch.ones(n), v3_phi_melhor=torch.zeros(n), step_dt=DT,
                command_manager=types.SimpleNamespace(get_command=lambda nome: torch.zeros(n, 3)))


def _sinal_m5(n):
    """Rastreio zero e qualidade larga 1 (caixa no alvo): o sinal do CARREGAR vale 1 e o h fica em 1."""
    return mock.patch.multiple(recompensas, _rastreio=lambda e, linear: torch.zeros(n),
                               _qualidade=lambda e: (torch.ones(n), torch.zeros(n)))


class TestProgresso(unittest.TestCase):
    """Núcleo da renda por segundo: w_etapa·etapa·h + w_phi·Φ, com os pesos do knob (3,0 e 6,0), valores à mão."""

    def _n(self, phi, etapa, h=1.0):
        return float(recompensas.n_progresso(torch.tensor([phi]), torch.tensor([float(etapa)]),
                                             R.renda_phi, R.renda_etapa, torch.tensor([h]))[0])

    def test_h_so_multiplica_a_etapa(self):
        # 13/11, M5: 6·n·h + 3·Φ, com n = 5, h = 0,5 e Φ = 0,4: 15 + 1,2
        self.assertAlmostEqual(self._n(0.4, 5, h=0.5), 16.2, places=5)

    def test_salto_do_fecho_nunca_cai_abaixo_de_3(self):
        # M5: para n de 1 a 8, h e Φ em [0, 1], a renda da etapa seguinte (h = 1, Φ = 0) passa a atual por ≥ 3
        for n in range(1, 9):
            for h in (0.0, 0.25, 0.5, 1.0):
                for phi in (0.0, 0.5, 1.0):
                    self.assertGreaterEqual(6 * (n + 1) - self._n(phi, n, h), 3.0 - 1e-6, f"n={n} h={h} Φ={phi}")

    def test_um_valor_por_linha_da_tabela_de_etapas(self):
        self.assertAlmostEqual(self._n(0.8, 0), 2.4, places=5)   # P0 com Φ 0,8
        self.assertAlmostEqual(self._n(0.0, 1), 6.0, places=5)   # P1 com Φ 0
        self.assertAlmostEqual(self._n(0.0, 5), 30.0, places=5)  # CARREGAR: só a etapa 5 paga
        self.assertAlmostEqual(self._n(0.5, 7), 43.5, places=5)  # B3 com Φ 0,5
        self.assertAlmostEqual(self._n(0.0, 9), 54.0, places=5)  # depois do B4
        self.assertAlmostEqual(self._n(0.5, 0), 1.5, places=5)   # etapa 0 com Φ 0,5 (o núcleo soma; a perda zera o Φ em `progresso`)

    def test_pairar_no_alvo_paga_renda_positiva(self):
        # Φ constante > 0 no P0: a renda paga a cada passo e a soma cresce com o tempo (o oposto do termo por diferença)
        soma = sum(self._n(0.6, 0) for _ in range(100))
        self.assertAlmostEqual(soma, 180.0, places=3)

    def test_o_fecho_sempre_sobe_a_renda(self):
        # Para cada par de alvos consecutivos, a renda do próximo com Φ = 0 passa a do atual com Φ = 1.
        # Etapas: P0..P4 = 0..4, B1..B4 = 5..8, depois do B4 = 9 (o CARREGAR, etapa 5, fica entre o P4 e o B1).
        etapas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for atual, proximo in zip(etapas[:-1], etapas[1:]):
            self.assertGreater(self._n(0.0, proximo), self._n(1.0, atual), f"etapa {atual} → {proximo}")
        # P4 (Φ = 1) → CARREGAR (etapa 5, Φ = 0) sobe; CARREGAR (30) → B1 (etapa 5, Φ ≥ 0) não cai
        self.assertGreater(self._n(0.0, 5), self._n(1.0, 4))
        self.assertGreaterEqual(self._n(0.0, 5), 30.0)


class TestPressa(unittest.TestCase):
    """`n_pressa` (M5): h·exp(−dt/τ) + recarga·ganho, com teto 1."""

    def _h(self, h, ganho):
        return float(recompensas.n_pressa(torch.tensor([h]), torch.tensor([ganho]), DT, R.tau_renda, R.recarga_renda)[0])

    def test_sem_ganho_h_cai_por_exp_a_cada_passo(self):
        self.assertAlmostEqual(self._h(1.0, 0.0), math.exp(-DT / R.tau_renda), places=6)
        self.assertAlmostEqual(self._h(0.5, 0.0), 0.5 * math.exp(-DT / R.tau_renda), places=6)

    def test_tres_segundos_sem_ganho_levam_h_a_e_menos_1(self):
        h = torch.tensor([1.0])
        for _ in range(round(3.0 / DT)):
            h = recompensas.n_pressa(h, torch.zeros(1), DT, R.tau_renda, R.recarga_renda)
        self.assertAlmostEqual(float(h[0]), math.exp(-1.0), places=4)

    def test_ganho_de_0_2_devolve_h_a_1_e_o_teto_vale_1(self):
        self.assertAlmostEqual(self._h(0.0, 0.2), 1.0, places=6)  # 5 · 0,2 = 1
        self.assertAlmostEqual(self._h(1.0, 5.0), 1.0, places=6)


class TestEtapaDoProgresso(unittest.TestCase):
    """`recompensas.progresso` com env falso e `_phi_env` trocado por um Φ fixo por env: a tabela de etapas inteira."""

    # (alvo, tarefa, perdeu, b4_fechado, p4_fechado, Φ do alvo) -> (renda esperada à mão, v3_phi gravado[, h de entrada,
    # 1 se faltar]). C3: nas etapas 5 (PEGAR) e 9 o h de entrada é 0,3 e a renda pede h = 1; sem o `h_ef = 1` ela cai.
    CASOS = [
        (0, "PEGAR", False, False, False, 0.8, 2.4, 0.8),        # P0, Φ 0,8
        (1, "PEGAR", False, False, False, 0.0, 6.0 * math.exp(-DT / R.tau_renda), 0.0),  # P1, Φ 0: h cai um passo
        (4, "PEGAR", False, False, True, 0.9, 30.0, 0.9, 0.3),   # PEGAR com p4_fechado (alvo P4): etapa 5, Φ não paga, h = 1
        (4, "CARREGAR", False, False, True, 0.9, 30.0, 0.9),     # CARREGAR com p4_fechado (alvo P4): etapa 5
        (-1, "CARREGAR", False, False, False, 0.0, 0.0, -1.0),   # CARREGAR sem a pega: 0
        (5, "BOTAR", False, False, False, 0.7, 0.0, 0.7),        # B1 no BOTAR sem p4_fechado: 0
        (7, "BOTAR", False, False, True, 0.5, 43.5, 0.5),        # B3, Φ 0,5, com a pega feita
        (6, "BOTAR", False, False, True, 0.5, 37.5, 0.5),        # B2 com p4_fechado True: etapa 6, Φ pago (fatia C)
        (-1, "ANDAR", False, True, True, 0.0, 54.0, -1.0, 0.3),  # ANDAR final depois do B4: etapa 9, h = 1
        (-1, "ANDAR", False, True, False, 0.0, 0.0, -1.0),       # b4_fechado sem p4_fechado: 0
        (8, "ANDAR", True, False, True, 0.5, 0.0, 0.5),          # caixa perdida, alvo B4, Φ 0,5: 0
        (-1, "ANDAR", True, True, True, 0.0, 0.0, -1.0),         # perdeu com b4_fechado: 0
        (-1, "ANDAR", False, False, False, 0.0, 0.0, -1.0),      # ANDAR da fatia ANDAR
        (-1, "PEGAR", False, False, False, 0.0, 0.0, -1.0),      # PEGAR na espera, sem alvo
    ]

    def test_tabela_de_etapas(self):
        n = len(self.CASOS)
        col = lambda i, dtype: torch.tensor([c[i] for c in self.CASOS], dtype=dtype)
        env = types.SimpleNamespace(
            device="cpu", v3_alvo_idx=col(0, torch.long), v3_tarefa=torch.tensor([TAREFAS.index(c[1]) for c in self.CASOS]),
            v3_perdeu=col(2, torch.bool), v3_b4_fechado=col(3, torch.bool), v3_p4_fechado=col(4, torch.bool),
            v3_sigma=torch.ones(n), v3_phi=torch.zeros(n),
            **{**_buffers_m5(n), "v3_h": torch.tensor([c[8] if len(c) > 8 else 1.0 for c in self.CASOS])})
        phi = col(5, torch.float32)
        with mock.patch.object(recompensas, "_phi_env",
                               lambda e, indice, sigma: torch.where(indice >= 0, phi, torch.zeros_like(phi))), \
                _sinal_m5(n):
            r = recompensas.progresso(env, TODAS_TAREFAS)
        for i, c in enumerate(self.CASOS):
            self.assertAlmostEqual(float(r[i]), c[6], places=5, msg=str(c))
            self.assertAlmostEqual(float(env.v3_phi[i]), c[7], places=5, msg=str(c))

    def test_tarefa_fora_da_mascara_paga_zero(self):
        env = types.SimpleNamespace(
            device="cpu", v3_alvo_idx=torch.tensor([0]), v3_tarefa=torch.tensor([TAREFAS.index("PEGAR")]),
            v3_perdeu=torch.zeros(1, dtype=torch.bool), v3_b4_fechado=torch.zeros(1, dtype=torch.bool),
            v3_p4_fechado=torch.zeros(1, dtype=torch.bool), v3_sigma=torch.ones(1), v3_phi=torch.zeros(1),
            **_buffers_m5(1))
        with mock.patch.object(recompensas, "_phi_env", lambda e, indice, sigma: torch.ones(1)), _sinal_m5(1):
            r = recompensas.progresso(env, ("ANDAR",))
        self.assertEqual(float(r[0]), 0.0)

    def test_phi_melhor_sobe_e_h_recarrega_so_com_ganho(self):
        # M5, P2 (etapa 2): env 0 com Φ 0,3 abaixo do melhor 0,5 (sem ganho: h cai); env 1 com Φ 0,6 sobre o melhor 0,1
        # (ganho 0,5 × 5 = 2,5: h volta ao teto 1). O melhor Φ guarda o máximo.
        env = types.SimpleNamespace(
            device="cpu", v3_alvo_idx=torch.tensor([2, 2]), v3_tarefa=torch.full((2,), TAREFAS.index("PEGAR")),
            v3_perdeu=torch.zeros(2, dtype=torch.bool), v3_b4_fechado=torch.zeros(2, dtype=torch.bool),
            v3_p4_fechado=torch.zeros(2, dtype=torch.bool), v3_sigma=torch.ones(2), v3_phi=torch.zeros(2),
            **{**_buffers_m5(2), "v3_h": torch.tensor([0.8, 0.8]), "v3_phi_melhor": torch.tensor([0.5, 0.1])})
        phi = torch.tensor([0.3, 0.6])
        with mock.patch.object(recompensas, "_phi_env", lambda e, indice, sigma: phi), _sinal_m5(2):
            r = recompensas.progresso(env, TODAS_TAREFAS)
        self.assertAlmostEqual(float(env.v3_h[0]), 0.8 * math.exp(-DT / R.tau_renda), places=5)
        self.assertAlmostEqual(float(env.v3_h[1]), 1.0, places=5)
        self.assertEqual(env.v3_phi_melhor.tolist(), [0.5, torch.tensor(0.6).item()])
        self.assertAlmostEqual(float(r[0]), 6 * 2 * float(env.v3_h[0]) + 3 * 0.3, places=5)

    def test_sinal_do_carregar_com_e_sem_comando(self):
        # M5, CARREGAR com a pega feita (p4_fechado, alvo P4), h inicial 0,5. Env 0: comando linear 0,5, rastreio linear
        # 0,6 e o angular −0,8 fora (canal sem comando), caixa no alvo (d 0): sinal 0,6. Env 1: sem comando, r_lin −0,5 e
        # r_ang −0,2: (1 − 0,5)(1 − 0,2) = 0,4, com a caixa a 0,20 m: rampa 0,4, sinal 0,16.
        env = types.SimpleNamespace(
            device="cpu", v3_alvo_idx=torch.tensor([4, 4]), v3_tarefa=torch.full((2,), TAREFAS.index("CARREGAR")),
            v3_perdeu=torch.zeros(2, dtype=torch.bool), v3_b4_fechado=torch.zeros(2, dtype=torch.bool),
            v3_p4_fechado=torch.ones(2, dtype=torch.bool), v3_sigma=torch.ones(2), v3_phi=torch.zeros(2),
            **{**_buffers_m5(2), "v3_h": torch.tensor([0.5, 0.5])})
        env.command_manager = types.SimpleNamespace(
            get_command=lambda nome: torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        r_lin, r_ang = torch.tensor([0.6, -0.5]), torch.tensor([-0.8, -0.2])
        with mock.patch.object(recompensas, "_phi_env", lambda e, indice, sigma: torch.ones(2)), \
                mock.patch.object(recompensas, "_rastreio", lambda e, linear: r_lin if linear else r_ang), \
                mock.patch.object(recompensas, "_qualidade", lambda e: (torch.ones(2), torch.tensor([0.0, 0.20]))):
            r = recompensas.progresso(env, TODAS_TAREFAS)
        passo = DT / R.tau_renda
        esperado = [0.5 + passo * (0.6 - 0.5), 0.5 + passo * (0.16 - 0.5)]
        for i in range(2):
            self.assertAlmostEqual(float(env.v3_h[i]), esperado[i], places=5)
            self.assertAlmostEqual(float(r[i]), 30.0 * esperado[i], places=4)  # etapa 5, Φ não paga


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
    def test_perda_paga_so_o_custo(self):
        # n_perda_caixa: só o custo, no passo em que o sinal de perda sobe (um passo depois da perda, caso 4)
        out = recompensas.n_perda_caixa(torch.tensor([True, False]), R.perda_caixa)
        self.assertAlmostEqual(float(out[0]), R.perda_caixa, places=4)
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

    def _p(self, idx, contatos=((True, True),), dois_pes=False, com=0.0, incl=0.0):
        """Portões de um alvo com os outros desvios em 0 (dentro de todas as faixas). `maos` entra como a cadeia o
        passa: contatos/2 (11/11: o portão recebe o valor pronto; a recompensa passa a firmeza da pega)."""
        def um(v):
            return torch.tensor([v], dtype=torch.float64)
        return recompensas.portoes_do_alvo(
            torch.tensor([idx]), um(0.0), um(0.0), um(incl), K.incl_inicio_graus,
            torch.tensor(contatos).double().sum(-1) / 2.0, um(0.0), um(0.0), um(com), torch.tensor([dois_pes]),
            torch.tensor([A], dtype=torch.float64), K)

    def test_maos_meio_com_um_contato_no_p3(self):
        # Teste E4: maos = 0,5 com um contato no P3 (P2–B2: contatos.sum/2)
        self.assertAlmostEqual(float(self._p(3, [(True, False)])["maos"][0]), 0.5, places=6)

    def test_maos_um_fora_de_p2_b2(self):
        # Teste E4: maos = 1 fora de P2–B2 (P0 e B3), mesmo sem contato
        self.assertEqual(float(self._p(0, [(False, False)])["maos"][0]), 1.0)
        self.assertEqual(float(self._p(7, [(False, False)])["maos"][0]), 1.0)

    def test_maos_um_no_p2_e_zero_no_p3_sem_contato(self):
        # 13/11, A1: o Φ do P2 não zera sem força (maos = 1); o P3 a B2 seguem exigindo o contato
        self.assertEqual(float(self._p(2, [(False, False)])["maos"][0]), 1.0)
        self.assertEqual(float(self._p(3, [(False, False)])["maos"][0]), 0.0)

    def test_incl_exige_a_caixa_nivelada_ate_o_b3(self):
        # 13/11, A6: a inclinação acima de limite + delta zera o `incl` no B3 e no P2; no B4 o portão vale 1
        fora = K.incl_inicio_graus + K.delta_incl_graus + 1.0
        self.assertEqual(float(self._p(7, incl=fora)["incl"][0]), 0.0)
        self.assertEqual(float(self._p(2, incl=fora)["incl"][0]), 0.0)
        self.assertEqual(float(self._p(8, incl=fora)["incl"][0]), 1.0)

    def test_janela_da_pega_comeca_no_p1(self):
        # 13/11, A2: o alvo do P1 é a palma encostada, então o contato com a caixa no P1 não é contato fora
        self.assertIn(1, recompensas._JANELA_PEGA)
        self.assertNotIn(0, recompensas._JANELA_PEGA)

    def test_com_um_sem_os_dois_pes(self):
        # Teste E4: com = 1 sem os dois pés no chão, mesmo com desvio do CoM fora da faixa
        self.assertEqual(float(self._p(4, com=0.5)["com"][0]), 1.0)

    def test_com_zero_com_os_dois_pes_e_desvio_fora_da_faixa(self):
        # Borda E4: com os dois pés e desvio = faixa + Δ (0,06 m) → 0
        self.assertAlmostEqual(float(self._p(4, dois_pes=True, com=K.faixa_com + K.delta_com)["com"][0]),
                               0.0, places=6)


TODAS_TAREFAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")


class TestInclLim(unittest.TestCase):
    """B1 (revisão E11): `incl_lim` lia `K`, que não existia no módulo, e o primeiro reset caía com NameError."""

    def test_incl_lim_roda_e_desce_pelo_degrau(self):
        env = types.SimpleNamespace(v3_degrau=torch.tensor([0, ATIVO.balanceador.degraus]))
        out = recompensas.incl_lim(env).double().numpy()
        np.testing.assert_allclose(out, [K.incl_inicio_graus, K.incl_fim_graus], atol=1e-4)


class TestCercaRecuoNoP4(unittest.TestCase):
    """S7 (revisão E11): no PEGAR e no BOTAR com o alvo P4, a `cerca_recuo` mede contra a referência do transporte
    (`ref.caixa_transporte_b`, norma xy de 0,25 m no mock), e não contra `ref.recuo` (0,40 m, o recuo da pega). A caixa
    está a 0,25 m: custo 0. (F4: o `Recompensa.recuo_carregar` saiu; a referência é só a do `_recuo_ref`.)"""

    def _env(self, tarefa):
        transporte = torch.tensor([[0.25, 0.0, 0.0]])  # o que o mock de referência devolve no CARREGAR
        robo = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=torch.zeros(1, 3)))
        caixa = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=transporte))

        class Referencia:
            def recuo(self, topo, a):
                return torch.full((1,), 0.40)

            def caixa_transporte_b(self, a):
                return transporte

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


class TestFreioRampa(unittest.TestCase):
    """O preço do freio cresce com o `s` das células abertas da tarefa, até o portão (colapso da it 5140)."""

    def _custo(self, s):
        # PEGAR com todas as juntas a 10 rad/s (acima do teto); só a célula aberta do PEGAR conta
        cel = [types.SimpleNamespace(tarefa="PEGAR", aberta=True, s=s),
               types.SimpleNamespace(tarefa="PEGAR", aberta=False, s=1.0)]
        robo = types.SimpleNamespace(data=types.SimpleNamespace(joint_vel=torch.full((1, 29), 10.0)))
        cmd = types.SimpleNamespace(get_command=lambda nome: torch.zeros(1, 3))
        env = types.SimpleNamespace(
            device="cpu", scene={"robot": robo}, command_manager=cmd, v3_cur=types.SimpleNamespace(celulas=cel),
            v3_tarefa=torch.tensor([TAREFAS.index("PEGAR")]), v3_freio=torch.zeros(1, dtype=torch.long))
        return float(recompensas.freio(env, ("PEGAR",))[0])

    def test_rampa_zero_cheia_e_meia(self):
        portao = ATIVO.balanceador.portao
        cheio = self._custo(portao)
        self.assertGreater(cheio, 0.0)
        self.assertEqual(self._custo(0.0), 0.0)
        self.assertAlmostEqual(self._custo(portao / 2), cheio / 2, places=4)


class TestDistanciaDoP4(unittest.TestCase):
    def test_p4_soma_a_pelve_na_escala_das_tolerancias(self):
        # §6.4: P4 = √(d_caixa² + (Δpelve · tol_caixa/tol_pelve)²); caixa no alvo e pelve 3 cm → 0,05
        z = torch.zeros(1, dtype=torch.float64)
        d = recompensas.distancia_do_alvo(torch.tensor([4]), torch.zeros(1, 2, 3, dtype=torch.float64), z.repeat(3).reshape(1, 3),
                                          z.repeat(3).reshape(1, 3), torch.tensor([0.03], dtype=torch.float64), z, z, z,
                                          torch.tensor([A], dtype=torch.float64), K)
        self.assertAlmostEqual(float(d[0]), 0.05, places=6)

    def test_p4_com_pelve_no_alvo_e_a_caixa(self):
        # §6.4: com Δpelve 0, o P4 é a distância da caixa (0,30 m em x → 0,30)
        z = torch.zeros(1, dtype=torch.float64)
        d = recompensas.distancia_do_alvo(torch.tensor([4]), torch.zeros(1, 2, 3, dtype=torch.float64),
                                          torch.tensor([[0.30, 0.0, 0.0]], dtype=torch.float64),
                                          torch.zeros(1, 3, dtype=torch.float64), z, z, z, z,
                                          torch.tensor([A], dtype=torch.float64), K)
        self.assertAlmostEqual(float(d[0]), 0.30, places=6)
    def test_p3_e_b1_nao_somam_a_pelve(self):
        # §6.4: a pelve só entra no P4. Com Δpelve 0,10 no P3 (índice 3) e no B1 (índice 5), d é o d_caixa (0,30 em x)
        z = torch.zeros(1, dtype=torch.float64)
        for indice in (3, 5):
            with self.subTest(indice=indice):
                d = recompensas.distancia_do_alvo(torch.tensor([indice]), torch.zeros(1, 2, 3, dtype=torch.float64),
                                                  torch.tensor([[0.30, 0.0, 0.0]], dtype=torch.float64),
                                                  torch.zeros(1, 3, dtype=torch.float64),
                                                  torch.tensor([0.10], dtype=torch.float64), z, z, z,
                                                  torch.tensor([A], dtype=torch.float64), K)
                self.assertAlmostEqual(float(d[0]), 0.30, places=6)


if __name__ == "__main__":
    unittest.main()
