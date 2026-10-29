"""Testes de g1_v3/comando.py (D3 do plano; S §7, §11.2, §19 conflito 5; PD-15). Escritos antes do código.

Núcleos da §6.3 (29/10): `sorteia_velocidade`, `monta_tabela_trechos`/`avanca_trecho` (cadeia), `tempo_da_ancora`
e `sucesso_do_segmento` (canal zero, PD-15). Convenção: comandos e integrais em colunas (vx, vy, wz) e
(linear, giro). Ordem de TAREFAS (knobs): 0 ANDAR, 1 PEGAR, 2 CARREGAR, 3 BOTAR.
"""
import unittest

import torch

from g1_v3 import comando
from g1_v3.knobs import ATIVO

K = ATIVO.comando
N = 100_000  # amostras: o erro de 25 ± 1% fica em ~7σ
CPU = "cpu"


def amostra(tipo, nivel, tarefa="ANDAR", seed=0):
    g = torch.Generator().manual_seed(seed)
    return comando.sorteia_velocidade(tipo, nivel, tarefa, N, g, K)


def t1(v, dtype=torch.float32):
    return torch.tensor([v], dtype=dtype)


def trechos_de(tab, ini):
    # segue `proximo` desde `ini` até −1 (fim da cadeia)
    out, t = [], int(ini)
    while t != -1:
        out.append(t)
        t = int(tab.proximo[t])
    return out


def passo(tab, trecho, relogio, aviso, atraso=0.4, dt=0.2, duracao=0.0):
    return comando.avanca_trecho(trecho, relogio, t1(duracao), t1(aviso, torch.bool), t1(atraso),
                                 t1(dt), tab)


def segmento(cmd, iproj, icmd, dpos, drum, tf, ta, perdeu=False, carregar=False):
    return comando.sucesso_do_segmento(
        torch.tensor([cmd]), torch.tensor([iproj]), torch.tensor([icmd]),
        t1(dpos), t1(drum), t1(tf), t1(ta), t1(perdeu, torch.bool), t1(carregar, torch.bool), K)


class SorteiaVelocidadeTest(unittest.TestCase):

    def test_reto_nivel_2_modulos_e_quatro_direcoes(self):
        # nível 2 → frente_tetos[2] = 0,75 para frente, e min(0,75; 1,0) = 0,75 para trás e de lado (S §11.2)
        v = amostra("reto", 2)
        self.assertTrue(bool((v[:, 2] == 0).all()))  # reto não gira
        modulo = v[:, :2].norm(dim=1)
        self.assertGreaterEqual(float(modulo.min()), 0.25 - 1e-6)
        self.assertLessEqual(float(modulo.max()), 0.75 + 1e-6)
        self.assertGreater(float(modulo.max()), 0.74)  # o teto é atingido, não só o piso
        eixo_x = v[:, 0].abs() > v[:, 1].abs()
        dirs = torch.stack([eixo_x & (v[:, 0] > 0), eixo_x & (v[:, 0] < 0),
                            ~eixo_x & (v[:, 1] > 0), ~eixo_x & (v[:, 1] < 0)]).float().mean(dim=1)
        for f in dirs.tolist():
            self.assertAlmostEqual(f, 0.25, delta=0.01)

    def test_giro_lugar_nao_anda(self):
        v = amostra("giro_lugar", 2)
        self.assertTrue(bool((v[:, :2] == 0).all()))

    def test_baixo_fica_entre_0_05_e_0_25(self):
        # metade linear (U[0,05; 0,25] em 4 direções), metade giro no lugar (±U[0,05; 0,25]) (D3)
        v = amostra("baixo", 0)
        lin = v[:, :2].norm(dim=1)
        w = v[:, 2].abs()
        so_linear = (lin > 0) & (w == 0) & (lin >= 0.05 - 1e-6) & (lin <= 0.25 + 1e-6)
        so_giro = (lin == 0) & (w >= 0.05 - 1e-6) & (w <= 0.25 + 1e-6)
        self.assertTrue(bool((so_linear | so_giro).all()))


class CadeiaDeTrechosTest(unittest.TestCase):

    def test_fatia_c_pegar_por_detector_e_botar_no_atraso(self):
        tab = comando.monta_tabela_trechos(ATIVO.cadeia, CPU)
        seq = trechos_de(tab, tab.inicio[2])
        # §7, fatia C: ANDAR parado, PEGAR (detector), CARREGAR parado, BOTAR (detector), ANDAR parado
        self.assertEqual([int(tab.tarefa[t]) for t in seq], [0, 1, 2, 3, 0])
        self.assertEqual([bool(tab.por_detector[t]) for t in seq], [False, True, False, True, False])

        # PEGAR: sem aviso, não avança e o relógio fica em 0
        pegar = seq[1]
        trecho, relogio = t1(pegar, torch.long), t1(0.0)
        for _ in range(5):
            trecho, relogio, avancou, _ = passo(tab, trecho, relogio, aviso=False)
            self.assertFalse(bool(avancou))
            self.assertEqual(int(trecho), pegar)
            self.assertEqual(float(relogio), 0.0)

        # aviso verdadeiro, atraso 0,4 s, passo 0,2 s: avança só quando o relógio desde o aviso chega a 0,4
        trecho, relogio, avancou, _ = passo(tab, trecho, relogio, aviso=True)
        self.assertFalse(bool(avancou))
        trecho, relogio, avancou, _ = passo(tab, trecho, relogio, aviso=True)
        self.assertTrue(bool(avancou))
        self.assertEqual(int(trecho), seq[2])  # o CARREGAR parado, por tempo
        self.assertEqual(float(relogio), 0.0)

        # BOTAR: durante o atraso, o trecho publica BOTAR (one-hot = tabela.tarefa[trecho])
        botar = seq[3]
        trecho = t1(botar, torch.long)
        relogio = t1(0.0)
        for _ in range(3):
            trecho, relogio, avancou, _ = passo(tab, trecho, relogio, aviso=False)
            self.assertEqual(int(trecho), botar)
            self.assertEqual(int(tab.tarefa[trecho]), 3)  # BOTAR


class SegmentoTest(unittest.TestCase):

    def test_razao_0_60_passa_e_0_40_falha(self):
        # canal linear ativo (|c| = 1 ≥ 0,02); razão = Σ(v·ĉ)dt / Σ|c|dt ≥ 0,50 (S §11.3)
        cmd = [0.5, 0.0, 0.0]
        _, ok60 = segmento(cmd, [0.6, 0.0], [1.0, 0.0], 0.0, 0.0, 5.0, 1.5)
        _, ok40 = segmento(cmd, [0.4, 0.0], [1.0, 0.0], 0.0, 0.0, 5.0, 1.5)
        self.assertTrue(bool(ok60))
        self.assertFalse(bool(ok40))

    def test_parado_com_deriva_0_15_m_falha(self):
        # canal linear zero: deriva da âncora < 0,10 m (PD-15); o giro zero passa (rumo 0)
        _, ok = segmento([0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.15, 0.0, 5.0, 1.5)
        self.assertFalse(bool(ok))

    def test_frenagem_medida_desde_a_ancora_passa(self):
        # o parado que anda 0,30 m nos primeiros 1,5 s não entra: a deriva conta só desde a âncora (t = 1,5 s)
        _, ok = segmento([0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.05, 0.0, 5.0, 1.5)
        self.assertTrue(bool(ok))

    def test_empurrao_regrava_a_ancora_e_segmento_curto_nao_registra(self):
        # empurrão em 2,0 s com abertura em 0: âncora em 3,5 s (PD-15)
        ancora = comando.tempo_da_ancora(t1(0.0), t1(2.0), 1.5)
        self.assertAlmostEqual(float(ancora[0]), 3.5, places=6)
        # sem empurrão: âncora = abertura + 1,5 s
        self.assertAlmostEqual(float(comando.tempo_da_ancora(t1(1.0), t1(1.0), 1.5)[0]), 2.5, places=6)
        # fecho em 3,0 s < âncora 3,5 s: não registra
        reg, _ = segmento([0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.0, 0.0, 3.0, 3.5)
        self.assertFalse(bool(reg))
        # segmento de 1,0 s com âncora 1,5 s: não registra
        reg, _ = segmento([0.5, 0.0, 0.0], [0.6, 0.0], [1.0, 0.0], 0.0, 0.0, 1.0, 1.5)
        self.assertFalse(bool(reg))

    def test_carregar_perdido_nao_tem_sucesso(self):
        # sucesso = todos os canais e (não carregar OU não perdeu) (§6.3)
        _, ok = segmento([0.5, 0.0, 0.0], [0.6, 0.0], [1.0, 0.0], 0.0, 0.0, 5.0, 1.5,
                         perdeu=True, carregar=True)
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    unittest.main()
