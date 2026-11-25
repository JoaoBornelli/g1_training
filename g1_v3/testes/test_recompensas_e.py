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
POUSO = [4]  # B1: k × fator_pouso (23/11, T14)


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

    def test_b3_normaliza_pelve_e_tronco(self):
        # Borda E4: B3 = √((Δpelve/0,03)² + (Δtronco/5°)² + (dpalmas/0,05)²); Δpelve 3 cm com o resto 0 → 1
        z = torch.zeros(1, dtype=torch.float64)
        d = recompensas.distancia_do_alvo(torch.tensor([6]), torch.zeros(1, 2, 3, dtype=torch.float64),
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

    def test_b3_cresce_com_as_palmas_longe_do_tronco(self):
        # 13/11, A9: com pelve e tronco no alvo, d = dpalmas/tol_palmas_b3 (0,05 m → 1; 0,10 m → 2), e cresce com dpalmas
        self.assertAlmostEqual(self._d_b(6, dpalmas=0.0), 0.0, places=6)
        self.assertAlmostEqual(self._d_b(6, dpalmas=K.tol_palmas_b3), 1.0, places=6)
        self.assertAlmostEqual(self._d_b(6, dpalmas=2 * K.tol_palmas_b3), 2.0, places=6)

    def test_b1_soma_o_deficit_de_apoio_a_distancia_da_caixa(self):
        # 13/11, A3; 23/11, T14: déficit 0 dá d_caixa; déficit 1 dá hypot(d_caixa, escala_apoio_b1 = 0,10)
        self.assertAlmostEqual(self._d_b(4, deficit=0.0, caixa_x=0.05), 0.05, places=6)
        self.assertAlmostEqual(self._d_b(4, deficit=1.0, caixa_x=0.05), math.hypot(0.05, K.escala_apoio_b1), places=6)
        self.assertAlmostEqual(self._d_b(4, deficit=1.0, caixa_x=0.0), K.escala_apoio_b1, places=6)

    def test_deficit_e_dpalmas_so_valem_no_b1_e_no_b3(self):
        # 13/11, A3/A9: no P3 (3) a distância segue a da caixa (a pelve está no alvo), com déficit 1 e dpalmas altos
        self.assertAlmostEqual(self._d_b(3, dpalmas=1.0, deficit=1.0, caixa_x=0.30), 0.30, places=6)

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
    # 18/11, T11b: o `progresso` lê os sensores das almofadas (`_contato_palmas`); aqui as duas tocam a caixa
    toque = types.SimpleNamespace(data=types.SimpleNamespace(found=torch.ones(n, 1)))
    return dict(v3_h=torch.ones(n), v3_phi_melhor=torch.zeros(n), step_dt=DT, scene={"palma_E": toque, "palma_D": toque},
                command_manager=types.SimpleNamespace(get_command=lambda nome: torch.zeros(n, 3),
                                                       get_term=lambda nome: types.SimpleNamespace(g_marcha=torch.ones(n))))


def _sinal_m5(n):
    """Rastreio zero e qualidade larga 1 (caixa no alvo): o sinal do CARREGAR vale 1 e o h fica em 1."""
    return mock.patch.multiple(recompensas, _rastreio=lambda e, linear, normalizado=False: torch.zeros(n),
                               _qualidade=lambda e: (torch.ones(n), torch.zeros(n)))


def _renda_progresso(alvo, tarefa, pega_fechada, phi, conquistas):
    """Renda por segundo de um env com o `progresso` real (etapa, Φ pago e conquistas pelas regras da seção 5 da spec),
    h = 1 e `_phi_env` trocado por Φ e conquistas [palmas, pega firme, peso pago] dados."""
    env = types.SimpleNamespace(
        device="cpu", v3_alvo_idx=torch.tensor([alvo]), v3_tarefa=torch.tensor([TAREFAS.index(tarefa)]),
        v3_perdeu=torch.zeros(1, dtype=torch.bool), v3_bota_fechada=torch.zeros(1, dtype=torch.bool),
        v3_pega_fechada=torch.tensor([pega_fechada]), v3_sigma=torch.ones(1), v3_phi=torch.zeros(1), **_buffers_m5(1))
    with mock.patch.object(recompensas, "_phi_env",
                           lambda e, indice, sigma: (torch.tensor([phi]), torch.tensor([conquistas]))), \
            _sinal_m5(1), mock.patch.object(recompensas, "n_pressa", lambda h, ganho, dt, tau, rec: h):
        return float(recompensas.progresso(env, ("ANDAR", "PEGAR", "CARREGAR", "BOTAR"))[0])


class TestProgresso(unittest.TestCase):
    """Núcleo da renda por segundo: w_etapa·etapa·h + w_phi·Φ, com os pesos do knob (3,0 e 6,0), valores à mão."""

    def _n(self, phi, etapa, h=1.0, conquistas=0.0):
        return float(recompensas.n_progresso(torch.tensor([phi]), torch.tensor([float(etapa)]), R.renda_phi,
                                             R.renda_etapa, torch.tensor([h]), torch.tensor([conquistas]),
                                             R.renda_conquista)[0])

    def test_conquistas_somam_fora_do_h(self):
        # 18/11, T10: 6·n·h + 3·Φ + 2·conquistas, com n = 3, h = 0,5, Φ = 0 e as duas conquistas cheias: 9 + 0 + 4
        self.assertAlmostEqual(self._n(0.0, 3, h=0.5, conquistas=2.0), 13.0, places=5)

    def test_escada_das_trocas_que_o_robo_controla(self):
        # 23/11, T14 (seção 5 da spec): pelo `progresso` (mapa de etapas, Φ pago e conquistas pagas), no pior caso de cada
        # troca (h = 1 na etapa nova, Φ máximo na que fecha, as três conquistas cheias) a renda de depois passa a de antes.
        # Conquistas [palmas, pega, peso pago]; o peso pago é c · q (24/11)
        phi0 = 1 - math.tanh(1 / K.largura_tanh)
        r = _renda_progresso
        # P2 → P3 com a caixa na laje (c = 0): o Φ do P3 paga 0 (Φ · c · q); 12 + 3 + 2·2 = 19 → 18 + 0 + 2·2 = 22. A igualdade
        # confere o Φ pago (o saldo sozinho só sobe se o P3 pagasse Φ cru)
        antes, depois = r(2, "PEGAR", False, 1.0, [1.0, 1.0, 0.0]), r(3, "PEGAR", False, 1.0, [1.0, 1.0, 0.0])
        self.assertAlmostEqual(depois, 18.0 + 2 * 2.0, places=4)
        self.assertGreaterEqual(depois - antes, 3.0 - 1e-4)
        # P3 → pega fechada (c · q = 1): 18 + 3 + 2·3 = 27 → 24 + 2·3 = 30; a etapa da pega fechada é 4
        antes, depois = r(3, "PEGAR", False, 1.0, [1.0, 1.0, 1.0]), r(3, "PEGAR", True, 1.0, [1.0, 1.0, 1.0])
        self.assertAlmostEqual(depois, 24.0 + 2 * 3.0, places=4)
        self.assertGreaterEqual(depois - antes, 3.0 - 1e-4)
        # B1 → B2: o B1 soma palmas e pega, e não o peso (24 + 3 + 2·2 = 31); o B2 não soma nenhuma (30 + 3·Φ_B2)
        antes, depois = r(4, "BOTAR", True, 1.0, [1.0, 1.0, 1.0]), r(5, "BOTAR", True, phi0, [1.0, 1.0, 1.0])
        self.assertAlmostEqual(antes, 24.0 + 3.0 + 2 * 2.0, places=4)
        self.assertGreater(depois, antes)
        # B2 → B3
        antes, depois = r(5, "BOTAR", True, 1.0, [1.0, 1.0, 1.0]), r(6, "BOTAR", True, phi0, [1.0, 1.0, 1.0])
        self.assertGreaterEqual(depois - antes, 3.0 - 1e-4)

    def test_h_so_multiplica_a_etapa(self):
        # 13/11, M5: 6·n·h + 3·Φ, com n = 5, h = 0,5 e Φ = 0,4: 15 + 1,2
        self.assertAlmostEqual(self._n(0.4, 5, h=0.5), 16.2, places=5)

    def test_salto_do_fecho_nunca_cai_abaixo_de_3(self):
        # M5: para n de 1 a 6, h e Φ em [0, 1], a renda da etapa seguinte (h = 1, Φ = 0) passa a atual por ≥ 3
        for n in range(1, 7):
            for h in (0.0, 0.25, 0.5, 1.0):
                for phi in (0.0, 0.5, 1.0):
                    self.assertGreaterEqual(6 * (n + 1) - self._n(phi, n, h), 3.0 - 1e-6, f"n={n} h={h} Φ={phi}")

    def test_um_valor_por_linha_da_tabela_de_etapas(self):
        self.assertAlmostEqual(self._n(0.8, 0), 2.4, places=5)   # P0 com Φ 0,8
        self.assertAlmostEqual(self._n(0.0, 1), 6.0, places=5)   # P1 com Φ 0
        self.assertAlmostEqual(self._n(0.0, 4), 24.0, places=5)  # CARREGAR: só a etapa 4 paga
        self.assertAlmostEqual(self._n(0.5, 6), 37.5, places=5)  # B3 com Φ 0,5
        self.assertAlmostEqual(self._n(0.0, 7), 42.0, places=5)  # depois do B3
        self.assertAlmostEqual(self._n(0.5, 0), 1.5, places=5)   # etapa 0 com Φ 0,5 (o núcleo soma; a perda zera o Φ em `progresso`)

    def test_pairar_no_alvo_paga_renda_positiva(self):
        # Φ constante > 0 no P0: a renda paga a cada passo e a soma cresce com o tempo (o oposto do termo por diferença)
        soma = sum(self._n(0.6, 0) for _ in range(100))
        self.assertAlmostEqual(soma, 180.0, places=3)

    def test_o_fecho_sempre_sobe_a_renda(self):
        # Para cada par de alvos consecutivos, a renda do próximo com Φ = 0 passa a do atual com Φ = 1.
        # Etapas: P0..P3 = 0..3, B1..B3 = 4..6, depois do B3 = 7 (o CARREGAR, etapa 4, fica entre o P3 e o B1).
        etapas = [0, 1, 2, 3, 4, 5, 6, 7]
        for atual, proximo in zip(etapas[:-1], etapas[1:]):
            self.assertGreater(self._n(0.0, proximo), self._n(1.0, atual), f"etapa {atual} → {proximo}")
        # P3 (Φ = 1) → CARREGAR (etapa 4, Φ = 0) sobe; CARREGAR (24) → B1 (etapa 4, Φ ≥ 0) não cai
        self.assertGreater(self._n(0.0, 4), self._n(1.0, 3))
        self.assertGreaterEqual(self._n(0.0, 4), 24.0)


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

    # (alvo, tarefa, perdeu, bota_fechada, pega_fechada, Φ do alvo) -> (renda esperada à mão, v3_phi gravado[, h de entrada,
    # 1 se faltar]). C3: nas etapas 4 (PEGAR) e 7 o h de entrada é 0,3 e a renda pede h = 1; sem o `h_ef = 1` ela cai.
    CASOS = [
        (0, "PEGAR", False, False, False, 0.8, 2.4, 0.8),        # P0, Φ 0,8
        (1, "PEGAR", False, False, False, 0.0, 6.0 * math.exp(-DT / R.tau_renda), 0.0),  # P1, Φ 0: h cai um passo
        (3, "PEGAR", False, False, True, 0.9, 24.0, 0.9, 0.3),   # PEGAR com pega_fechada (alvo P3): etapa 4, Φ não paga, h = 1
        (3, "CARREGAR", False, False, True, 0.9, 24.0, 0.9),     # CARREGAR com pega_fechada (alvo P3): etapa 4
        (-1, "CARREGAR", False, False, False, 0.0, 0.0, -1.0),   # CARREGAR sem a pega: 0
        (4, "BOTAR", False, False, False, 0.7, 0.0, 0.7),        # B1 no BOTAR sem pega_fechada: 0
        (6, "BOTAR", False, False, True, 0.5, 37.5, 0.5),        # B3, Φ 0,5, com a pega feita
        (5, "BOTAR", False, False, True, 0.5, 31.5, 0.5),        # B2 com pega_fechada True: etapa 5, Φ pago (fatia C)
        (-1, "ANDAR", False, True, True, 0.0, 42.0, -1.0, 0.3),  # ANDAR final depois do B3: etapa 7, h = 1
        (-1, "ANDAR", False, True, False, 0.0, 0.0, -1.0),       # bota_fechada sem pega_fechada: 0
        (6, "ANDAR", True, False, True, 0.5, 0.0, 0.5),          # caixa perdida, alvo B3, Φ 0,5: 0
        (-1, "ANDAR", True, True, True, 0.0, 0.0, -1.0),         # perdeu com bota_fechada: 0
        (-1, "ANDAR", False, False, False, 0.0, 0.0, -1.0),      # ANDAR da fatia ANDAR
        (-1, "PEGAR", False, False, False, 0.0, 0.0, -1.0),      # PEGAR na espera, sem alvo
    ]

    def test_tabela_de_etapas(self):
        n = len(self.CASOS)
        col = lambda i, dtype: torch.tensor([c[i] for c in self.CASOS], dtype=dtype)
        env = types.SimpleNamespace(
            device="cpu", v3_alvo_idx=col(0, torch.long), v3_tarefa=torch.tensor([TAREFAS.index(c[1]) for c in self.CASOS]),
            v3_perdeu=col(2, torch.bool), v3_bota_fechada=col(3, torch.bool), v3_pega_fechada=col(4, torch.bool),
            v3_sigma=torch.ones(n), v3_phi=torch.zeros(n),
            **{**_buffers_m5(n), "v3_h": torch.tensor([c[8] if len(c) > 8 else 1.0 for c in self.CASOS])})
        phi = col(5, torch.float32)
        with mock.patch.object(recompensas, "_phi_env",
                               lambda e, indice, sigma: (torch.where(indice >= 0, phi, torch.zeros_like(phi)),
                                                         torch.zeros(n, 3))), \
                _sinal_m5(n):
            r = recompensas.progresso(env, TODAS_TAREFAS)
        for i, c in enumerate(self.CASOS):
            self.assertAlmostEqual(float(r[i]), c[6], places=5, msg=str(c))
            self.assertAlmostEqual(float(env.v3_phi[i]), c[7], places=5, msg=str(c))

    def test_conquistas_so_somam_onde_a_pega_tem_de_continuar(self):
        # 18/11, T10: com Φ 0, q_palmas 0,3, q_pega 0,9 e (T14, 24/11) o peso pago c · q = 0,5 · 0,9 = 0,45, cada caso paga 6·n (h = 1) mais
        # as conquistas do lugar: palmas + pega no B1; palmas + pega + peso no P3 e no CARREGAR (ou no PEGAR) com a pega
        # fechada: 2·(0,3 + 0,9 + 0,45) = 3,3. B2, B3 e o resto não pagam conquista. (alvo, tarefa, perdeu, bota_fechada, pega_fechada,
        # renda[, almofadas em contato])
        # T11b: o P2 também soma, e nele SÓ a pega vale pela fração das almofadas na caixa: duas 12 + 2·(0,3 + 0,9) = 14,4;
        # uma 12 + 2·(0,3 + 0,45) = 13,5; nenhuma 12 + 2·0,3 = 12,6. O P1 segue sem conquistas; o P3 sem almofada paga a
        # pega inteira; o P2 sem a caixa (CARREGAR aberto depois de um PEGAR que parou no P2) paga 0. O peso não paga no P2
        casos = [(2, "PEGAR", False, False, False, 14.4), (2, "PEGAR", False, False, False, 13.5, (True, False)),
                 (2, "PEGAR", False, False, False, 12.6, (False, False)), (1, "PEGAR", False, False, False, 6.0),
                 (2, "CARREGAR", False, False, False, 0.0), (2, "PEGAR", True, False, False, 0.0),
                 (3, "PEGAR", False, False, False, 21.3, (False, False)), (3, "PEGAR", False, False, False, 21.3),
                 (3, "PEGAR", False, False, True, 27.3), (3, "CARREGAR", False, False, True, 27.3),
                 (4, "BOTAR", False, False, True, 26.4), (5, "BOTAR", False, False, True, 30.0),
                 (6, "BOTAR", False, False, True, 36.0), (-1, "CARREGAR", False, False, False, 0.0),
                 (3, "ANDAR", True, False, False, 0.0), (-1, "ANDAR", False, True, True, 42.0)]
        n = len(casos)
        col = lambda i, dtype: torch.tensor([c[i] for c in casos], dtype=dtype)
        env = types.SimpleNamespace(
            device="cpu", v3_alvo_idx=col(0, torch.long), v3_tarefa=torch.tensor([TAREFAS.index(c[1]) for c in casos]),
            v3_perdeu=col(2, torch.bool), v3_bota_fechada=col(3, torch.bool), v3_pega_fechada=col(4, torch.bool),
            v3_sigma=torch.ones(n), v3_phi=torch.zeros(n), **{**_buffers_m5(n), "v3_h": torch.ones(n)})
        for k, nome in enumerate(("palma_E", "palma_D")):
            achou = torch.tensor([[float(c[6][k]) if len(c) > 6 else 1.0] for c in casos])
            env.scene[nome] = types.SimpleNamespace(data=types.SimpleNamespace(found=achou))
        with mock.patch.object(recompensas, "_phi_env", lambda e, indice, sigma: (torch.zeros(n), torch.tensor([[0.3, 0.9, 0.5 * 0.9]]).expand(n, 3))), \
                _sinal_m5(n), mock.patch.object(recompensas, "n_pressa", lambda h, ganho, dt, tau, rec: h):
            r = recompensas.progresso(env, TODAS_TAREFAS)
        for i, c in enumerate(casos):
            self.assertAlmostEqual(float(r[i]), c[5], places=4, msg=str(c))

    def test_phi_do_p3_paga_vezes_o_peso_pago_e_o_do_b1_nao(self):
        # 23/11, T14; 24/11: no P3 o Φ pago é Φ · c · q (e o peso soma 2·c·q, com q a pega firme); no B1 o Φ paga inteiro e o
        # peso não soma. Φ 0,8, palmas 0, pega q = 0,5, h = 1: P3 paga 18 + 3·0,8·cq + 2·(q + cq); B1 (com a pega feita) paga
        # 24 + 3·0,8 + 2·q. Com c = 1 e q = 0 (caixa escorada, sem aperto) o P3 paga só 18. `v3_phi` guarda o Φ cru (as
        # métricas `phi_<alvo>` o leem)
        for c, q in ((0.0, 0.5), (0.5, 0.5), (1.0, 0.5), (1.0, 0.0)):
            casos = [(3, "PEGAR", False), (4, "BOTAR", True)]  # (alvo, tarefa, pega fechada)
            env = types.SimpleNamespace(
                device="cpu", v3_alvo_idx=torch.tensor([x[0] for x in casos]),
                v3_tarefa=torch.tensor([TAREFAS.index(x[1]) for x in casos]),
                v3_perdeu=torch.zeros(2, dtype=torch.bool), v3_bota_fechada=torch.zeros(2, dtype=torch.bool),
                v3_pega_fechada=torch.tensor([x[2] for x in casos]), v3_sigma=torch.ones(2), v3_phi=torch.zeros(2),
                **_buffers_m5(2))
            conquistas = torch.tensor([[0.0, q, c * q]]).expand(2, 3)
            with mock.patch.object(recompensas, "_phi_env", lambda e, indice, sigma: (torch.full((2,), 0.8), conquistas)), \
                    _sinal_m5(2), mock.patch.object(recompensas, "n_pressa", lambda h, ganho, dt, tau, rec: h):
                r = recompensas.progresso(env, TODAS_TAREFAS)
            with self.subTest(c=c, q=q):
                self.assertAlmostEqual(float(r[0]), 18.0 + 3 * 0.8 * c * q + 2.0 * (q + c * q), places=4)
                self.assertAlmostEqual(float(r[1]), 24.0 + 3 * 0.8 + 2.0 * q, places=4)
                self.assertEqual(env.v3_phi.tolist(), [torch.tensor(0.8).item()] * 2)

    def test_tarefa_fora_da_mascara_paga_zero(self):
        env = types.SimpleNamespace(
            device="cpu", v3_alvo_idx=torch.tensor([0]), v3_tarefa=torch.tensor([TAREFAS.index("PEGAR")]),
            v3_perdeu=torch.zeros(1, dtype=torch.bool), v3_bota_fechada=torch.zeros(1, dtype=torch.bool),
            v3_pega_fechada=torch.zeros(1, dtype=torch.bool), v3_sigma=torch.ones(1), v3_phi=torch.zeros(1),
            **_buffers_m5(1))
        with mock.patch.object(recompensas, "_phi_env", lambda e, indice, sigma: (torch.ones(1), torch.zeros(1, 3))), \
                _sinal_m5(1):
            r = recompensas.progresso(env, ("ANDAR",))
        self.assertEqual(float(r[0]), 0.0)

    def test_phi_melhor_sobe_e_h_recarrega_so_com_ganho(self):
        # M5, P2 (etapa 2): env 0 com Φ 0,3 abaixo do melhor 0,5 (sem ganho: h cai); env 1 com Φ 0,6 sobre o melhor 0,1
        # (ganho 0,5 × 5 = 2,5: h volta ao teto 1). O melhor Φ guarda o máximo.
        env = types.SimpleNamespace(
            device="cpu", v3_alvo_idx=torch.tensor([2, 2]), v3_tarefa=torch.full((2,), TAREFAS.index("PEGAR")),
            v3_perdeu=torch.zeros(2, dtype=torch.bool), v3_bota_fechada=torch.zeros(2, dtype=torch.bool),
            v3_pega_fechada=torch.zeros(2, dtype=torch.bool), v3_sigma=torch.ones(2), v3_phi=torch.zeros(2),
            **{**_buffers_m5(2), "v3_h": torch.tensor([0.8, 0.8]), "v3_phi_melhor": torch.tensor([0.5, 0.1])})
        phi = torch.tensor([0.3, 0.6])
        with mock.patch.object(recompensas, "_phi_env", lambda e, indice, sigma: (phi, torch.zeros(2, 3))), _sinal_m5(2):
            r = recompensas.progresso(env, TODAS_TAREFAS)
        self.assertAlmostEqual(float(env.v3_h[0]), 0.8 * math.exp(-DT / R.tau_renda), places=5)
        self.assertAlmostEqual(float(env.v3_h[1]), 1.0, places=5)
        self.assertEqual(env.v3_phi_melhor.tolist(), [0.5, torch.tensor(0.6).item()])
        self.assertAlmostEqual(float(r[0]), 6 * 2 * float(env.v3_h[0]) + 3 * 0.3, places=5)

    def test_no_p3_o_ganho_do_phi_cru_so_recarrega_o_h_com_o_peso_pago(self):
        # 23/11, T14; 24/11: arrastar a caixa pela laje, ou escorá-la no tronco sem aperto lateral, sobe o Φ cru do P3 com
        # c · q = 0. A recarga do h e o melhor Φ leem Φ · c · q: com c · q = 0 (c = 1, q = 0) um ganho de 0,5 no Φ cru não
        # recarrega o h (que só cai) nem mexe no melhor Φ; com c = q = 1 o mesmo ganho recarrega o h ao teto e sobe o melhor
        # Φ. `v3_phi` guarda o Φ cru nos dois casos
        for c, q, h_esperado, melhor_esperado in ((1.0, 0.0, 0.8 * math.exp(-DT / R.tau_renda), 0.1), (1.0, 1.0, 1.0, 0.6)):
            env = types.SimpleNamespace(
                device="cpu", v3_alvo_idx=torch.tensor([3]), v3_tarefa=torch.tensor([TAREFAS.index("PEGAR")]),
                v3_perdeu=torch.zeros(1, dtype=torch.bool), v3_bota_fechada=torch.zeros(1, dtype=torch.bool),
                v3_pega_fechada=torch.zeros(1, dtype=torch.bool), v3_sigma=torch.ones(1), v3_phi=torch.zeros(1),
                **{**_buffers_m5(1), "v3_h": torch.tensor([0.8]), "v3_phi_melhor": torch.tensor([0.1])})
            conquistas = torch.tensor([[0.0, q, c * q]])
            with mock.patch.object(recompensas, "_phi_env", lambda e, indice, sigma: (torch.tensor([0.6]), conquistas)), \
                    _sinal_m5(1):
                recompensas.progresso(env, TODAS_TAREFAS)
            with self.subTest(c=c, q=q):
                self.assertAlmostEqual(float(env.v3_h[0]), h_esperado, places=5)
                self.assertAlmostEqual(float(env.v3_phi_melhor[0]), melhor_esperado, places=5)
                self.assertAlmostEqual(float(env.v3_phi[0]), 0.6, places=5)

    def test_sinal_do_carregar_com_e_sem_comando(self):
        # M5, CARREGAR com a pega feita (pega_fechada, alvo P3), h inicial 0,5. Env 0: comando linear 0,5, rastreio linear
        # 0,6 e o angular −0,8 fora (canal sem comando), caixa no alvo (d 0): sinal 0,6. Env 1: sem comando, r_lin −0,5 e
        # r_ang −0,2: (1 − 0,5)(1 − 0,2) = 0,4, com a caixa a 0,20 m: rampa 0,4, sinal 0,16.
        env = types.SimpleNamespace(
            device="cpu", v3_alvo_idx=torch.tensor([3, 3]), v3_tarefa=torch.full((2,), TAREFAS.index("CARREGAR")),
            v3_perdeu=torch.zeros(2, dtype=torch.bool), v3_bota_fechada=torch.zeros(2, dtype=torch.bool),
            v3_pega_fechada=torch.ones(2, dtype=torch.bool), v3_sigma=torch.ones(2), v3_phi=torch.zeros(2),
            **{**_buffers_m5(2), "v3_h": torch.tensor([0.5, 0.5])})
        env.command_manager = types.SimpleNamespace(
            get_command=lambda nome: torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            get_term=lambda nome: types.SimpleNamespace(g_marcha=torch.ones(2)))
        r_lin, r_ang = torch.tensor([0.6, -0.5]), torch.tensor([-0.8, -0.2])
        with mock.patch.object(recompensas, "_phi_env", lambda e, indice, sigma: (torch.ones(2), torch.zeros(2, 3))), \
                mock.patch.object(recompensas, "_rastreio",
                                  lambda e, linear, normalizado=False: r_lin if linear else r_ang), \
                mock.patch.object(recompensas, "_qualidade", lambda e: (torch.ones(2), torch.tensor([0.0, 0.20]))):
            r = recompensas.progresso(env, TODAS_TAREFAS)
        passo = DT / R.tau_renda
        esperado = [0.5 + passo * (0.6 - 0.5), 0.5 + passo * (0.16 - 0.5)]
        for i in range(2):
            self.assertAlmostEqual(float(env.v3_h[i]), esperado[i], places=5)
            self.assertAlmostEqual(float(r[i]), 24.0 * esperado[i], places=4)  # etapa 4, Φ não paga

    def test_g_da_marcha_multiplica_so_o_sinal_comandado_do_h(self):
        # T16: mesmo caso do teste anterior com g = 0,5. Env 0 (comando): sinal 0,5 · 0,6 = 0,3. Env 1 (deslize parado): 0,16, sem g
        env = types.SimpleNamespace(
            device="cpu", v3_alvo_idx=torch.tensor([3, 3]), v3_tarefa=torch.full((2,), TAREFAS.index("CARREGAR")),
            v3_perdeu=torch.zeros(2, dtype=torch.bool), v3_bota_fechada=torch.zeros(2, dtype=torch.bool),
            v3_pega_fechada=torch.ones(2, dtype=torch.bool), v3_sigma=torch.ones(2), v3_phi=torch.zeros(2),
            **{**_buffers_m5(2), "v3_h": torch.tensor([0.5, 0.5])})
        env.command_manager = types.SimpleNamespace(
            get_command=lambda nome: torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            get_term=lambda nome: types.SimpleNamespace(g_marcha=torch.full((2,), 0.5)))
        r_lin, r_ang = torch.tensor([0.6, -0.5]), torch.tensor([-0.8, -0.2])
        with mock.patch.object(recompensas, "_phi_env", lambda e, indice, sigma: (torch.ones(2), torch.zeros(2, 3))), \
                mock.patch.object(recompensas, "_rastreio",
                                  lambda e, linear, normalizado=False: r_lin if linear else r_ang), \
                mock.patch.object(recompensas, "_qualidade", lambda e: (torch.ones(2), torch.tensor([0.0, 0.20]))):
            recompensas.progresso(env, TODAS_TAREFAS)
        passo = DT / R.tau_renda
        for i, sinal in enumerate((0.3, 0.16)):
            self.assertAlmostEqual(float(env.v3_h[i]), 0.5 + passo * (sinal - 0.5), places=5)


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
        # Teste E4: maos = 0,5 com um contato no P3 (P3 e B1: contatos.sum/2)
        self.assertAlmostEqual(float(self._p(3, [(True, False)])["maos"][0]), 0.5, places=6)

    def test_maos_um_fora_de_p3_e_b1(self):
        # Teste E4; 23/11, T14: maos = 1 fora do P3 e do B1 (P0, B2 e B3), mesmo sem contato
        for idx in (0, 5, 6):
            self.assertEqual(float(self._p(idx, [(False, False)])["maos"][0]), 1.0)

    def test_maos_um_no_p2_e_zero_no_p3_sem_contato(self):
        # 13/11, A1: o Φ do P2 não zera sem força (maos = 1); o P3 e o B1 seguem exigindo o contato
        self.assertEqual(float(self._p(2, [(False, False)])["maos"][0]), 1.0)
        self.assertEqual(float(self._p(3, [(False, False)])["maos"][0]), 0.0)
        self.assertEqual(float(self._p(4, [(False, False)])["maos"][0]), 0.0)

    def test_incl_exige_a_caixa_nivelada_ate_o_b2(self):
        # 13/11, A6: a inclinação acima de limite + delta zera o `incl` no B2 e no P2; no B3 o portão vale 1
        fora = K.incl_inicio_graus + K.delta_incl_graus + 1.0
        self.assertEqual(float(self._p(5, incl=fora)["incl"][0]), 0.0)
        self.assertEqual(float(self._p(2, incl=fora)["incl"][0]), 0.0)
        self.assertEqual(float(self._p(6, incl=fora)["incl"][0]), 1.0)

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


class TestCercaRecuoNoP3(unittest.TestCase):
    """S7 (revisão E11): no PEGAR e no BOTAR com o alvo P3 (o P4 antigo), a `cerca_recuo` mede contra a referência do transporte
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
            v3_alvo_idx=torch.tensor([alvos.P3]), v3_perdeu=torch.zeros(1, dtype=torch.bool),
            v3_topo=torch.zeros(1), v3_topo_destino=torch.zeros(1), v3_meia=torch.full((1,), A),
            v3_ref=Referencia())

    def test_cerca_recuo_no_p3_mede_contra_o_transporte(self):
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


class TestDistanciaDoP3(unittest.TestCase):
    def test_p3_soma_a_pelve_na_escala_das_tolerancias(self):
        # §6.4; 23/11, T14: P3 = √(d_caixa² + (Δpelve · tol_transporte_p3/tol_pelve_p3)²); caixa no alvo e pelve 3 cm → 0,05
        z = torch.zeros(1, dtype=torch.float64)
        d = recompensas.distancia_do_alvo(torch.tensor([3]), torch.zeros(1, 2, 3, dtype=torch.float64), z.repeat(3).reshape(1, 3),
                                          z.repeat(3).reshape(1, 3), torch.tensor([0.03], dtype=torch.float64), z, z, z,
                                          torch.tensor([A], dtype=torch.float64), K)
        self.assertAlmostEqual(float(d[0]), 0.05, places=6)
        self.assertAlmostEqual(float(d[0]), K.tol_transporte_p3 * 0.03 / K.tol_pelve_p3, places=6)

    def test_p3_com_pelve_no_alvo_e_a_caixa(self):
        # §6.4: com Δpelve 0, o P3 é a distância da caixa (0,30 m em x → 0,30)
        z = torch.zeros(1, dtype=torch.float64)
        d = recompensas.distancia_do_alvo(torch.tensor([3]), torch.zeros(1, 2, 3, dtype=torch.float64),
                                          torch.tensor([[0.30, 0.0, 0.0]], dtype=torch.float64),
                                          torch.zeros(1, 3, dtype=torch.float64), z, z, z, z,
                                          torch.tensor([A], dtype=torch.float64), K)
        self.assertAlmostEqual(float(d[0]), 0.30, places=6)

    def test_b1_nao_soma_a_pelve(self):
        # §6.4: a pelve só entra no P3. Com Δpelve 0,10 no B1 (índice 4), d é o d_caixa (0,30 em x)
        z = torch.zeros(1, dtype=torch.float64)
        d = recompensas.distancia_do_alvo(torch.tensor([4]), torch.zeros(1, 2, 3, dtype=torch.float64),
                                          torch.tensor([[0.30, 0.0, 0.0]], dtype=torch.float64),
                                          torch.zeros(1, 3, dtype=torch.float64),
                                          torch.tensor([0.10], dtype=torch.float64), z, z, z,
                                          torch.tensor([A], dtype=torch.float64), K)
        self.assertAlmostEqual(float(d[0]), 0.30, places=6)


class TestConjuntosDeAlvo(unittest.TestCase):
    """Os conjuntos de índices de `recompensas.py` (`_P3_B2`, `_B1_B2`, `_JANELA_PEGA`) pelas bordas: um env por alvo,
    P0..P3 = 0..3 e B1..B3 = 4..6 (23/11, T14). Um índice deslocado em qualquer conjunto muda um destes vetores."""

    ALVOS = torch.arange(7)

    def test_a_massa_da_caixa_entra_no_com_do_p3_ao_b2_e_nao_no_p2_nem_no_b3(self):
        # robô de 40 kg com o CoM na origem, caixa de 10 kg em x = 1: com a caixa, x = 10 / 50 = 0,2; sem ela, 0
        n = len(self.ALVOS)
        env = types.SimpleNamespace(
            scene={"robot": types.SimpleNamespace(indexing=types.SimpleNamespace(root_body_id=0)),
                   "box": types.SimpleNamespace(data=types.SimpleNamespace(root_com_pos_w=torch.tensor([[1.0, 0.0, 0.0]]).expand(n, 3)))},
            sim=types.SimpleNamespace(data=types.SimpleNamespace(subtree_com=torch.zeros(n, 1, 3)),
                                      model=types.SimpleNamespace(body_subtreemass=torch.full((1, 1), 40.0))),
            v3_pegou=torch.ones(n, dtype=torch.bool), v3_perdeu=torch.zeros(n, dtype=torch.bool),
            v3_massa=torch.full((n,), 10.0), v3_alvo_idx=self.ALVOS)
        com_x = recompensas._com_combinado(env)[:, 0]
        for i, esperado in enumerate((0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0)):
            self.assertAlmostEqual(float(com_x[i]), esperado, places=6, msg=f"alvo {i}")

    def test_o_topo_de_referencia_e_o_do_destino_no_b1_e_no_b2_e_o_da_pega_nos_outros(self):
        env = types.SimpleNamespace(v3_topo=torch.full((7,), 1.0), v3_topo_destino=torch.full((7,), 2.0))
        self.assertEqual(recompensas._topo_do(env, self.ALVOS).tolist(), [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0])

    def test_o_contato_com_a_caixa_e_livre_do_p1_ao_b2_e_conta_no_p0_e_no_b3(self):
        # uma fatia de 5 N (acima de `contato_fora_N`) na caixa; sem chão fora dos pés e sem autocolisão. Tarefa PEGAR até
        # o P3 e BOTAR do B1 em diante: o CARREGAR também abre a janela e fica de fora
        n = len(self.ALVOS)
        caixa = torch.zeros(n, 12, 3)
        caixa[:, 0, 0] = 5.0
        tarefa = torch.tensor([TAREFAS.index("PEGAR")] * 4 + [TAREFAS.index("BOTAR")] * 3)
        env = types.SimpleNamespace(
            device="cpu", v3_tarefa=tarefa, v3_alvo_idx=self.ALVOS,
            scene={"nao_pes_chao": types.SimpleNamespace(data=types.SimpleNamespace(force=torch.zeros(n, 1, 3))),
                   "auto_colisao": types.SimpleNamespace(data=types.SimpleNamespace(found=torch.zeros(n, 1))),
                   "caixa_robo": types.SimpleNamespace(data=types.SimpleNamespace(force=caixa))})
        r = recompensas.contato_fora(env, TODAS_TAREFAS)
        self.assertEqual(r.tolist(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
