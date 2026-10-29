"""Testes do filtro de elevação (C0 do plano; PERC §0, §4, §7; PD-6, PD-7, PD-10, PD-42; I §2.4).

Escritos antes do código (lane de testes). O módulo g1_v3/percepcao_lidar.py ainda não existe:
o ImportError abaixo é o esperado agora. Contrato: plano §6.3 (commit 46c8681).
"""
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from g1_v3.knobs import Percepcao
from g1_v3 import percepcao_lidar  # ImportError esperado até a C1

K = Percepcao()  # fonte única dos números: nada de número de teste fora do knobs
DADOS = Path(__file__).resolve().parent.parent / "dados"
SIG_REF2 = K.sigma_ref ** 2  # σ_ref² = (5 cm)² (PD-6)
CELULA = K.celula  # 0,10 m
N_REC = K.n_recorte  # 21 células por lado no recorte
N_CEL = N_REC * N_REC  # 441
DEGRAU_X, DEGRAU_H = 1.0, 0.15  # cenário do teste de densidade: degrau de 0,15 m em x = 1,0 m
N_PASSOS = 60  # 60 × 20 ms a 1 m/s


def _idx(x, y):
    """Índice no recorte (I §2.4): linha de x = +1,0 a −1,0; dentro da linha, y = +1,0 a −1,0."""
    return round((1.0 - x) / CELULA) * N_REC + round((1.0 - y) / CELULA)


def _R(d, n=1):
    """R = (σ_incl·d̄)² + σ_m²/n (PD-42, C1). σ_incl e σ_ângulo entram em rad."""
    sig_leitura2 = (math.radians(K.ruido_incl_graus) * d) ** 2
    sig_m2 = K.ruido_alcance ** 2 + (d * math.radians(K.ruido_angulo_graus)) ** 2
    return sig_leitura2 + sig_m2 / n


def _leitura(pontos, dist=None):
    """Pontos (R,3) → (pontos (1,R,3), validos (1,R), dist (1,R)). Sem `dist`, usa a norma."""
    p = np.asarray(pontos, dtype=np.float64).reshape(-1, 3)
    d = np.linalg.norm(p, axis=1) if dist is None else np.asarray(dist, dtype=np.float64)
    return (torch.tensor(p, dtype=torch.float32)[None],
            torch.ones(1, len(p), dtype=torch.bool),
            torch.tensor(d, dtype=torch.float32)[None])


def _filtro():
    return percepcao_lidar.FiltroDeElevacao(1, "cpu", K)


def _prediz(f, dxy=(0.0, 0.0), dpsi=0.0, z_ref=0.0, dt=0.0):
    """dt é float (§6.3); os demais são (E,)."""
    f.prediz(torch.tensor([dxy], dtype=torch.float32), torch.tensor([dpsi], dtype=torch.float32),
             torch.tensor([z_ref], dtype=torch.float32), float(dt))


def _mapa(f):
    """(altura, confiança) do recorte do env 0, 441 valores cada."""
    r = f.recorte()[0].numpy()
    return r[:N_CEL], r[N_CEL:]


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rodrigues(v):
    """Matriz de rotação de um vetor de giro v (rad). Usada para sortear o giro por leitura."""
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * kx + (1.0 - math.cos(th)) * (kx @ kx)


def _raios(n, seed):
    """Direções no frame do sensor: azimute ±60°, 25° a 65° abaixo do horizonte (para o chão)."""
    rng = np.random.default_rng(seed)
    az = rng.uniform(-math.pi / 3, math.pi / 3, n)
    el = rng.uniform(math.radians(25), math.radians(65), n)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)], axis=1)


def _trace(o, d, degrau=True):
    """Distância (N,) de cada raio d (mundo) a partir de o até o chão z = 0. Com `degrau`, o chão
    sobe 0,15 m para x ≥ 1,0 (face vertical em x = 1,0). Sem acerto: 100 (fora do alcance).
    Só gera `dist` das leituras sintéticas; a geometria vale para raios para baixo e à frente."""
    dx, dz = d[:, 0], d[:, 2]
    ok = dz < 0
    t = np.full(len(d), 100.0)
    tf = -o[2] / np.where(ok, dz, -1.0)
    if not degrau:
        t[ok] = tf[ok]
        return t
    xf = o[0] + tf * dx
    hit_f = ok & (xf < DEGRAU_X)
    tr = (DEGRAU_X - o[0]) / dx  # dx > 0 nos raios desta cena
    zr = o[2] + tr * dz
    hit_r = ~hit_f & (zr >= 0) & (zr <= DEGRAU_H)
    tt = (DEGRAU_H - o[2]) / np.where(ok, dz, -1.0)
    xt = o[0] + tt * dx
    hit_t = ~hit_f & ~hit_r & ok & (xt >= DEGRAU_X)
    t[hit_f] = tf[hit_f]
    t[hit_r] = tr[hit_r]
    t[hit_t] = tt[hit_t]
    return t


def _pontos(dist, dirs, pos, base, rot=None, incl=None):
    """Chama pontos_no_frame_do_mapa com gerador None (sem ruído sorteado, sem perda)."""
    rot = np.eye(3) if rot is None else rot
    incl = np.eye(3) if incl is None else incl
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    return percepcao_lidar.pontos_no_frame_do_mapa(
        t(dist)[None], t(dirs), t(pos)[None], t(rot)[None], t(base)[None],
        torch.zeros(1), torch.zeros(1), t(incl)[None], K, None)


class TestFiltroDeElevacao(unittest.TestCase):

    def test_degrau_de_0_15_vira_salto_de_0_15(self):
        # Pontos sem ruído: centro de célula em x de −0,5 a +0,7; o degrau fica entre x = 0,1
        # (antes) e x = 0,2 (depois). PERC §0: salto de 0,15 ± 0,02 m.
        pts = [(round(i * CELULA, 4), 0.0, 0.15 if i >= 2 else 0.0) for i in range(-5, 8)]
        f = _filtro()
        f.atualiza(*_leitura(pts))
        h, _ = _mapa(f)
        self.assertAlmostEqual(h[_idx(0.3, 0.0)] - h[_idx(0.1, 0.0)], 0.15, delta=0.02)
        self.assertAlmostEqual(h[_idx(-0.5, 0.0)], 0.0, delta=1e-4)

    def test_chao_plano_com_ruido_de_2cm_depois_de_3_leituras(self):
        # 50 pontos por célula: o erro da média (2 cm / √150 ≈ 0,16 cm) fica longe de 1 cm.
        # Com 50 pontos o piso σ_fixo domina R, e a confiança satura em 0,862 (PD-42).
        rng = np.random.default_rng(0)
        cx, cy = np.meshgrid(1.0 - CELULA * np.arange(N_REC), 1.0 - CELULA * np.arange(N_REC),
                             indexing="ij")
        centros = np.stack([cx.ravel(), cy.ravel()], axis=1)  # mesma ordem de _idx
        f = _filtro()
        for _ in range(3):
            xy = np.repeat(centros, 50, axis=0) + rng.uniform(-0.04, 0.04, size=(N_CEL * 50, 2))
            z = rng.normal(0.0, 0.02, size=(N_CEL * 50, 1))
            f.atualiza(*_leitura(np.hstack([xy, z])))
        h, c = _mapa(f)
        self.assertLess(np.abs(h).max(), 0.01)
        esperado = SIG_REF2 / (SIG_REF2 + K.sigma_fixo ** 2)
        self.assertAlmostEqual(esperado, 0.862, delta=0.001)  # PD-42: o piso, 2 cm
        np.testing.assert_allclose(c, esperado, rtol=0, atol=0.001)

    def test_numero_pd6_com_piso_e_um_metro(self):
        # PD-6: célula no piso, depois de 1 m de transporte: 0,758. Só confere a fórmula.
        esperado = SIG_REF2 / (SIG_REF2 + K.sigma_fixo ** 2 + K.q_desloc * 1.0)
        self.assertAlmostEqual(esperado, 0.758, delta=0.001)

    def test_transporte_de_0_10_desloca_uma_celula(self):
        # Transporte de 0,10 m para a frente em 5 passos de 2 cm (1 m/s a 50 Hz): x = +0,5 vai a x = +0,4.
        f = _filtro()
        f.atualiza(*_leitura([(0.5, 0.0, 0.15)], dist=[0.5]))
        for _ in range(5):
            _prediz(f, dxy=(0.02, 0.0))
        h, c = _mapa(f)
        self.assertAlmostEqual(h[_idx(0.4, 0.0)], 0.15, delta=0.01)
        self.assertGreater(c[_idx(0.4, 0.0)], 0.0)
        self.assertEqual(c[_idx(0.5, 0.0)], 0.0)  # a célula nova em x = +0,5 vem de fora do campo visto

    def test_transporte_vertical_pd7(self):
        # PD-7: o pé mais baixo sobe 0,15 (z_ref). A célula em +0,15 vira 0; a em 0 vira −0,15.
        f = _filtro()
        _prediz(f, z_ref=0.0)  # primeiro passo: guarda z_ref = 0
        f.atualiza(*_leitura([(0.5, 0.0, 0.15), (-0.5, 0.0, 0.0)], dist=[0.5, 0.5]))
        _, c0 = _mapa(f)
        _prediz(f, z_ref=0.15)
        h1, c1 = _mapa(f)
        self.assertAlmostEqual(h1[_idx(0.5, 0.0)], 0.0, delta=1e-5)
        self.assertAlmostEqual(h1[_idx(-0.5, 0.0)], -0.15, delta=1e-5)
        np.testing.assert_allclose(c1, c0, rtol=0, atol=1e-6)  # só a altura muda, não a confiança

    def test_reinicia_limpa_e_primeiro_prediz_nao_desloca(self):
        # Depois do reinicia não há z_ref anterior: o primeiro prediz usa 0 e não desloca.
        # Fraco por construção: o mapa está vazio nesse passo. Confere o reinicia e o zero.
        f = _filtro()
        f.atualiza(*_leitura([(0.5, 0.0, 0.15)], dist=[0.5]))
        f.reinicia(torch.tensor([0]))
        h, c = _mapa(f)
        self.assertTrue(np.all(h == 0.0) and np.all(c == 0.0))
        _prediz(f, z_ref=0.15)
        f.atualiza(*_leitura([(0.5, 0.0, 0.15)], dist=[0.5]))
        _prediz(f, z_ref=0.15)  # mesmo z_ref: nenhum deslocamento vertical
        h, _ = _mapa(f)
        self.assertAlmostEqual(h[_idx(0.5, 0.0)], 0.15, delta=1e-5)

    def test_giro_de_90_graus_no_lugar(self):
        # Δψ = +90° (CCW): a célula nova (0; −0,5) vem da célula velha R(Δψ)·p = (0,5; 0).
        # Tolerância de ±1 célula (C0).
        f = _filtro()
        f.atualiza(*_leitura([(0.5, 0.0, 0.15)], dist=[0.5]))
        for _ in range(75):  # 1,05 rad/s a 50 Hz
            _prediz(f, dpsi=math.pi / 150)
        h, c = _mapa(f)
        vistos = np.flatnonzero(c > 0.0)
        self.assertEqual(len(vistos), 1)
        alvo = _idx(0.0, -0.5)
        i = vistos[0]
        self.assertLessEqual(abs(i // N_REC - alvo // N_REC), 1)
        self.assertLessEqual(abs(i % N_REC - alvo % N_REC), 1)
        self.assertAlmostEqual(h[i], 0.15, delta=0.01)

    def test_celula_nunca_vista_fica_zero(self):
        # Mapa novo, e depois de um transporte: célula sem leitura tem altura 0 e confiança 0.
        f = _filtro()
        h, c = _mapa(f)
        self.assertTrue(np.all(h == 0.0) and np.all(c == 0.0))
        _prediz(f, dxy=(1.0, 0.0))
        h, c = _mapa(f)
        self.assertTrue(np.all(h == 0.0) and np.all(c == 0.0))

    def test_um_metro_de_transporte_reduz_a_confianca(self):
        # Um ponto a 2 m (R acima do piso, n = 1). Depois de 1 m sem leitura e Δt = 0:
        # c = σ_ref² / (σ_ref² + max(R, σ_fixo²) + q_desloc·1). PD-6, PD-7.
        f = _filtro()
        f.atualiza(*_leitura([(2.0, 0.0, 0.0)], dist=[2.0]))
        _prediz(f, dxy=(1.0, 0.0), dt=0.0)
        _, c = _mapa(f)
        esperado = SIG_REF2 / (SIG_REF2 + max(_R(2.0), K.sigma_fixo ** 2) + K.q_desloc * 1.0)
        self.assertAlmostEqual(c[_idx(1.0, 0.0)], esperado, delta=1e-6)

    def test_cinco_pontos_na_mesma_celula_dao_a_media(self):
        # Ruído de 2 cm, valores fixos: média 0,008 m; o máximo seria 0,03 (PD-42).
        z = [0.03, 0.01, -0.01, 0.02, -0.01]
        pts = [(0.50 + dx, 0.50, zi) for dx, zi in zip([-0.02, -0.01, 0.0, 0.01, 0.02], z)]
        f = _filtro()
        f.atualiza(*_leitura(pts))
        h, _ = _mapa(f)
        self.assertAlmostEqual(h[_idx(0.5, 0.5)], np.mean(z), delta=1e-5)
        self.assertLess(abs(h[_idx(0.5, 0.5)]), 0.01)

    def test_ordem_do_recorte(self):
        # I §2.4: índice 0 = (x +1,0; y +1,0); 20 = (+1,0; −1,0); 440 = (−1,0; −1,0).
        # Altura nos índices 0..440; confiança em 441..881.
        f = _filtro()
        f.atualiza(*_leitura([(1.0, 1.0, 0.3), (1.0, -1.0, 0.1), (-1.0, -1.0, -0.3)]))
        r = f.recorte()[0].numpy()
        self.assertEqual(r.shape, (2 * N_CEL,))
        h, c = r[:N_CEL], r[N_CEL:]
        self.assertAlmostEqual(h[0], 0.3, delta=1e-5)
        self.assertAlmostEqual(h[20], 0.1, delta=1e-5)
        self.assertAlmostEqual(h[440], -0.3, delta=1e-5)
        self.assertGreater(c[0], 0.0)
        self.assertGreater(c[440], 0.0)
        self.assertEqual(h[1], 0.0)
        self.assertEqual(c[1], 0.0)


class TestPontosNoFrame(unittest.TestCase):
    """pontos_no_frame_do_mapa com geometria de chão plano; ruído sorteado desligado (gerador None)."""

    def _chao(self, z_sensor, rot, dirs, incl=None):
        pos = np.array([0.0, 0.0, z_sensor])
        dist = _trace(pos, dirs @ rot.T, degrau=False)  # rot: sensor → mundo
        return _pontos(dist, dirs, pos, base=np.array([0.0, 0.0, 0.8]), rot=rot, incl=incl)

    def test_base_inclinada_10_graus_nao_muda_o_chao(self):
        # Pitch de 10° na pose verdadeira do sensor: o chão continua em 0 ± 1 cm.
        pontos, _, validos = self._chao(1.2, _rot_y(math.radians(10)), _raios(205, 5))
        v = validos[0].numpy()
        self.assertGreater(v.sum(), 100)
        self.assertLess(np.abs(pontos[0].numpy()[v, 2]).max(), 0.01)

    def test_agachado_20_cm_ve_o_chao_em_zero(self):
        # Sensor 0,20 m mais baixo com os pés no chão: o z do ponto mais baixo anda com o pé,
        # então o chão continua em 0 (PERC §0).
        dirs = _raios(205, 5)
        for z_sensor in (1.2, 1.0):
            pontos, _, validos = self._chao(z_sensor, np.eye(3), dirs)
            v = validos[0].numpy()
            self.assertGreater(v.sum(), 100)
            self.assertLess(np.abs(pontos[0].numpy()[v, 2]).max(), 0.01)

    def test_giro_do_imu_de_1_grau_inclina_o_chao_1_7_cm_a_1_m(self):
        # Um raio que bate no chão a x = +1,0 m (tan φ = 1,2 / 1,0). Com o giro fixo do IMU de 1°
        # em pitch, z muda de 0 para ≈ −1,7 cm (PD-42, plano §11 D0).
        phi = math.atan(1.2)
        dirs = np.array([[math.cos(phi), 0.0, -math.sin(phi)]])
        incl = _rot_y(math.radians(1.0))
        pos = np.array([0.0, 0.0, 1.2])
        z0 = self._chao(1.2, np.eye(3), dirs)[0].numpy()[0, 0, 2]
        pontos, _, validos = self._chao(1.2, np.eye(3), dirs, incl=incl)
        self.assertTrue(validos[0, 0].item())
        z1 = pontos.numpy()[0, 0, 2]
        esperado = pos[2] + (incl @ np.array([1.0, 0.0, -1.2]))[2]  # o mesmo giro, à mão
        self.assertAlmostEqual(z0, 0.0, delta=1e-4)
        self.assertAlmostEqual(z1, esperado, delta=1e-4)
        self.assertAlmostEqual(abs(z1), 0.017, delta=0.002)


class TestFiltraCorpo(unittest.TestCase):

    def test_raio_mais_margem_e_raio_minimo(self):
        # Sensor na origem. Cápsula horizontal em x = 2,0, z = 0,5, de y = −1 a +1, raio 0,10.
        # Sai o ponto a menos de raio + margem (0,13) da cápsula; sai o ponto a menos de
        # raio_min (0,85) do sensor (PD-10). O chão a 1,2 m abaixo do sensor fica.
        pontos = torch.tensor([[[0.84, 0.0, 0.0],   # < raio_min: sai
                                [0.86, 0.0, 0.0],   # > raio_min: fica
                                [2.0, 0.0, 0.62],   # 0,12 da cápsula < 0,13: sai
                                [2.0, 0.0, 0.64],   # 0,14 da cápsula > 0,13: fica
                                [1.0, 0.0, -1.2]]])  # chão a 1,2 m abaixo: fica
        validos = torch.ones(1, 5, dtype=torch.bool)
        capsulas = torch.tensor([[[[2.0, -1.0, 0.5], [2.0, 1.0, 0.5]]]])  # (E,K,2,3)
        raios = torch.tensor([0.10])
        sensor = torch.zeros(1, 3)
        novo = percepcao_lidar.filtra_corpo(pontos, validos, capsulas, raios, sensor,
                                            K.raio_min, K.margem_corpo)
        self.assertEqual(novo.dtype, torch.bool)
        self.assertEqual(novo[0].tolist(), [False, True, False, True, True])


class TestInvarianciaADensidade(unittest.TestCase):

    def _passeio(self, n_raios, tilts):
        """60 passos de 20 ms, sensor a 1,2 m andando 1 m/s em +x sobre um degrau de 0,15 m em
        x = 1,0. Por passo: raios por leitura com ruído de alcance (2 cm) e de ângulo (0,15°) no
        teste; o giro de inclinação `tilts[k]` é o mesmo nas duas densidades (gerador None)."""
        dirs = _raios(n_raios, 3)
        rng = np.random.default_rng(100 + n_raios)
        f = _filtro()
        contagem = {}
        for k in range(N_PASSOS):
            x = -0.6 + 0.02 * k
            pos = np.array([x, 0.0, 1.2])
            dist = _trace(pos, dirs)
            dist_r = dist + rng.normal(0.0, K.ruido_alcance, len(dist))
            dr = dirs + rng.normal(0.0, math.radians(K.ruido_angulo_graus), dirs.shape)
            dr /= np.linalg.norm(dr, axis=1, keepdims=True)
            pontos, _, validos = _pontos(dist_r, dr, pos, base=np.array([x, 0.0, 0.8]),
                                         incl=tilts[k])
            if k > 0:
                _prediz(f, dxy=(0.02, 0.0), dt=0.02)
            f.atualiza(pontos, validos, torch.tensor(dist_r, dtype=torch.float32)[None])
            # Contagem de leituras por célula do mundo (10 cm), a partir dos acertos válidos.
            v = validos[0].numpy()
            hits = pos + dirs[v] * dist[v][:, None]
            for c in set(map(tuple, np.floor(hits[:, :2] / CELULA).astype(int))):
                contagem[c] = contagem.get(c, 0) + 1
        # Contagem de cada célula do recorte final, pelo centro dela no mundo.
        x_fim = -0.6 + 0.02 * (N_PASSOS - 1)
        cont = np.zeros(N_CEL, dtype=int)
        for i in range(N_CEL):
            wx = x_fim + (1.0 - CELULA * (i // N_REC))
            wy = 1.0 - CELULA * (i % N_REC)
            cont[i] = contagem.get((math.floor(wx / CELULA + 1e-9), math.floor(wy / CELULA + 1e-9)), 0)
        return f, cont

    def test_recorte_com_2_leituras_nao_depende_da_densidade(self):
        # PD-42: 205 e 4 000 pontos por passo, mesmo giro de inclinação por leitura nos dois.
        # Compara as células com ≥ 2 leituras nos dois casos: altura ± 1 cm, confiança ± 0,03.
        # O giro é sorteado uma vez por passo (rng próprio, igual nos dois casos) e passado como
        # incl_imu; o ruído por ponto é aplicado no teste (gerador None). Perda não entra.
        rng_g = np.random.default_rng(11)
        tilts = [_rodrigues(v) for v in rng_g.normal(0.0, math.radians(K.ruido_incl_graus),
                                                     (N_PASSOS, 3))]
        f1, cont1 = self._passeio(205, tilts)
        f2, cont2 = self._passeio(4000, tilts)
        h1, c1 = _mapa(f1)
        h2, c2 = _mapa(f2)
        sel = (cont1 >= 2) & (cont2 >= 2)
        self.assertGreater(sel.sum(), 20)  # sem isto, o teste passa por vazio
        self.assertTrue((c1[sel] > 0).all() and (c2[sel] > 0).all())  # lida 2 vezes = vista: o transporte leva a leitura
        self.assertGreater(h2[sel].max(), 0.10)  # o degrau está no recorte
        # Altura: a célula da borda mistura chão e degrau na proporção que os raios sorteiam, e fica fora. Nas
        # outras, cada célula erra ~0,9 cm (dp) nos dois casos: ±1 cm no viés; ±3 cm em 95% (réplica, PERC §0).
        x_fim = -0.6 + 0.02 * (N_PASSOS - 1)
        borda = np.abs(x_fim + 1.0 - CELULA * (np.arange(N_CEL) // N_REC) - DEGRAU_X) < CELULA / 2
        dh = (h1 - h2)[sel & ~borda]
        self.assertLessEqual(abs(dh.mean()), 0.01)
        self.assertLessEqual(np.percentile(np.abs(dh), 95), 0.03)
        # Confiança ±0,03 em 90% das células. O resto fica na borda da vista: a 205 raios a célula é lida mais
        # raramente e envelhece pelo transporte (PD-7). O n da leitura já não entra (PD-42).
        self.assertLessEqual(np.mean(np.abs(c1[sel] - c2[sel]) > 0.03), 0.10)


class TestPadraoMid360(unittest.TestCase):

    def test_trecho_se_repete_a_cada_200_passos(self):
        # 200 trechos de 20 ms: o passo 200 é o passo 0 (PD-5).
        padrao = percepcao_lidar.PadraoMid360(str(DADOS / "mid360_passos.npz"), "cpu")
        a, b = padrao.direcoes(200), padrao.direcoes(0)
        self.assertEqual(tuple(a.shape), (205, 3))
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
