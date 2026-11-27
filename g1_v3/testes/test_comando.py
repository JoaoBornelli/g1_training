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


def amostra(tipo, nivel, tarefa="CARREGAR", seed=0):
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
        t1(dpos), t1(drum), t1(tf), t1(ta), t1(perdeu, torch.bool), t1(carregar, torch.bool), t1(0.0),
        t1(False, torch.bool), K)


class SorteiaVelocidadeTest(unittest.TestCase):

    def test_reto_nivel_2_modulos_e_quatro_direcoes(self):
        # CARREGAR: nível 2 → frente_tetos[2] = 0,7 para frente (24/11, T15), min(0,7; 0,5) = 0,5 para trás e min(0,7; 0,3) = 0,3
        # de lado (S §11.2); o piso é 0,25 e o do lado, 0,2
        v = amostra("reto", 2)
        self.assertTrue(bool((v[:, 2] == 0).all()))  # reto não gira
        modulo = v[:, :2].norm(dim=1)
        self.assertGreaterEqual(float(modulo.min()), K.piso_lado - 1e-6)
        self.assertLessEqual(float(modulo.max()), 0.7 + 1e-6)
        self.assertGreater(float(modulo.max()), 0.69)  # o teto é atingido, não só o piso
        eixo_x = v[:, 0].abs() > v[:, 1].abs()
        dirs = torch.stack([eixo_x & (v[:, 0] > 0), eixo_x & (v[:, 0] < 0),
                            ~eixo_x & (v[:, 1] > 0), ~eixo_x & (v[:, 1] < 0)]).float().mean(dim=1)
        for f in dirs.tolist():
            self.assertAlmostEqual(f, 0.25, delta=0.01)

    def test_carregar_nenhum_sorteio_passa_dos_tetos_por_canal(self):
        # 24/11, T15: frente 1,0, trás 0,5, lado 0,3 m/s e giro 0,5 rad/s, em todos os níveis e tipos do CARREGAR
        for nivel in range(len(K.frente_tetos)):
            for tipo in ("reto", "terreno", "arco", "giro_lugar"):
                v = amostra(tipo, nivel)
                with self.subTest(nivel=nivel, tipo=tipo):
                    self.assertLessEqual(float(v[:, 0].max()), min(K.frente_tetos[nivel], 1.0) + 1e-6)
                    self.assertGreaterEqual(float(v[:, 0].min()), -min(K.frente_tetos[nivel], 0.5) - 1e-6)
                    self.assertLessEqual(float(v[:, 1].abs().max()), min(K.frente_tetos[nivel], 0.3) + 1e-6)
                    self.assertLessEqual(float(v[:, 2].abs().max()), 0.5 + 1e-6)
        # no último nível os tetos são atingidos
        v = amostra("arco", len(K.frente_tetos) - 1)
        self.assertAlmostEqual(float(v[:, 0].max()), 1.0, delta=0.01)
        self.assertAlmostEqual(float(v[:, 0].min()), -0.5, delta=0.01)
        self.assertAlmostEqual(float(v[:, 1].abs().max()), 0.3, delta=0.01)
        self.assertAlmostEqual(float(v[:, 2].abs().max()), 0.5, delta=0.01)

    def test_giro_lugar_nao_anda(self):
        v = amostra("giro_lugar", 2)
        self.assertTrue(bool((v[:, :2] == 0).all()))

    def test_andar_nivel_0_comeca_em_0_3_com_10_por_cento_parado(self):
        # 24/11, T15: o nível 0 do ANDAR começa em 0,3 (frente e trás), 0,2 (lado) e 0,3 rad/s (giro), 10 % zerados; (vx, vy)
        # na elipse e wz independente
        v = amostra("andar", 0, "ANDAR")
        self.assertLessEqual(float(v[:, 0].abs().max()), K.andar_inicio + 1e-6)
        self.assertLessEqual(float(v[:, 1].abs().max()), K.andar_lado_inicio + 1e-6)
        self.assertLessEqual(float(v[:, 2].abs().max()), K.andar_giro_inicio + 1e-6)
        self.assertGreater(float(v[:, 0].max()), 0.29)  # a faixa é usada até o teto, nos dois sentidos
        self.assertLess(float(v[:, 0].min()), -0.29)
        # 24/11, T16: os 10 % mais o sorteio com ‖c_xy‖ < `linear_min` (a elipse 0,3 × 0,2 perde o disco de 0,1: 1/6) e |giro| < `giro_min` (1/3)
        parado = K.andar_parado + (1 - K.andar_parado) * K.linear_min ** 2 / (K.andar_inicio * K.andar_lado_inicio) \
            * K.giro_min / K.andar_giro_inicio
        self.assertAlmostEqual(float((v == 0).all(dim=1).float().mean()), parado, delta=0.01)
        # independentes: um canal ligado não liga os outros; o giro anda junto com o linear
        ativo = ~(v == 0).all(dim=1)
        self.assertGreater(float(((v[:, 2] != 0) & (v[:, :2].norm(dim=1) > 0.1))[ativo].float().mean()), 0.5)

    def test_andar_nivel_7_vai_aos_tetos_finais_e_nenhum_nivel_passa_deles(self):
        # 24/11, T15: frente 1,0, trás 0,5, lado 0,3 m/s e giro 0,5 rad/s; o nível 7 é o máximo do plano
        for nivel in (7, 12):
            v = amostra("andar", nivel, "ANDAR")
            self.assertAlmostEqual(float(v[:, 0].max()), 1.0, delta=0.01)
            self.assertAlmostEqual(float(v[:, 0].min()), -0.5, delta=0.01)
            self.assertAlmostEqual(float(v[:, 1].abs().max()), 0.3, delta=0.01)
            self.assertAlmostEqual(float(v[:, 2].abs().max()), 0.5, delta=0.01)
            self.assertAlmostEqual(float((v == 0).all(dim=1).float().mean()), K.andar_parado, delta=0.01)
        for nivel in range(13):
            for tipo in ("andar", "terreno"):
                v = amostra(tipo, nivel, "ANDAR", seed=nivel)
                with self.subTest(nivel=nivel, tipo=tipo):
                    self.assertLessEqual(float(v[:, 0].max()), 1.0 + 1e-6)
                    self.assertGreaterEqual(float(v[:, 0].min()), -0.5 - 1e-6)
                    self.assertLessEqual(float(v[:, 1].abs().max()), 0.3 + 1e-6)
                    self.assertLessEqual(float(v[:, 2].abs().max()), 0.5 + 1e-6)

    def test_andar_sorteia_vx_vy_na_elipse_dos_tetos(self):
        # 17/11, T7: (vx, vy) uniforme na elipse dos tetos; no quadrado de antes, (vx/semieixo)² + (vy/lado)² chegava a 2
        v = amostra("andar", 7, "ANDAR")
        ativo = v[:, :2].norm(dim=-1) > 0  # com linear: T16, o linear zerado (parado ou ‖c_xy‖ < `linear_min`) sai da conta
        vx, vy = v[ativo, 0], v[ativo, 1]
        frente, tras = K.frente_tetos[-1], K.tras_teto_max
        # T16: o disco ‖c_xy‖ < `linear_min` sai da elipse; P(corte) = 2·r²/((frente + trás)·lado), as duas metades somadas
        corte = 2 * K.linear_min ** 2 / ((frente + tras) * K.lado_teto_max)
        rho2 = (vx / torch.where(vx > 0, frente, tras)) ** 2 + (vy / K.lado_teto_max) ** 2
        self.assertLessEqual(float(rho2.max()), 1.0 + 1e-5)
        # uniforme na área: metade da elipse fica em rho² ≤ 0,5 (menos o disco cortado, que cai todo aí); a frente tem
        # (frente − r²/lado)/(frente + trás − 2r²/lado) do que sobra
        self.assertAlmostEqual(float((rho2 <= 0.5).float().mean()), (0.5 - corte) / (1 - corte), delta=0.01)
        self.assertAlmostEqual(float((vx > 0).float().mean()), (frente - K.linear_min ** 2 / K.lado_teto_max)
                               / (frente + tras - 2 * K.linear_min ** 2 / K.lado_teto_max), delta=0.01)

    def test_andar_terreno_limita_vx_a_terreno_v_max(self):
        v = amostra("terreno", 7, "ANDAR")
        self.assertLessEqual(float(v[:, 0].max()), K.terreno_v_max + 1e-6)
        self.assertGreater(float(v[:, 0].max()), K.terreno_v_max - 0.01)
        self.assertGreaterEqual(float(v[:, 0].min()), 0.0)
        # 27/11, T19: vx = frente·r (r = √U): zero também quando vx < `linear_min`, P = (linear_min/frente)²
        frente = min(K.frente_tetos[-1], K.terreno_v_max)
        parado = K.andar_parado + (1 - K.andar_parado) * (K.linear_min / frente) ** 2
        self.assertAlmostEqual(float((v == 0).all(dim=1).float().mean()), parado, delta=0.01)

    def test_andar_terreno_e_so_frente_e_rumo(self):
        # 27/11, T19: no terreno do ANDAR o sorteio é "frente + rumo": vy = 0 e wz = 0 (o rumo é do `TwistV3._update_command`), vx de 0 até
        # `frente` (o teto do nível, com `terreno_v_max`), e as regras `linear_min` e `andar_parado` continuam. O `andar` (plano) não muda
        for nivel in range(8):
            v = amostra("terreno", nivel, "ANDAR", seed=nivel)
            frente = min(K.andar_inicio + K.andar_passo * nivel, K.frente_tetos[-1], K.terreno_v_max)
            with self.subTest(nivel=nivel):
                self.assertTrue(bool((v[:, 1] == 0).all()))
                self.assertTrue(bool((v[:, 2] == 0).all()))
                self.assertGreaterEqual(float(v[:, 0].min()), 0.0)
                self.assertAlmostEqual(float(v[:, 0].max()), frente, delta=0.01)
                self.assertFalse(bool(((v[:, 0] > 0) & (v[:, 0] < K.linear_min)).any()))
        v = amostra("andar", 7, "ANDAR")
        self.assertGreater(float(v[:, 1].abs().max()), 0.2)  # fora do terreno vy e wz seguem sorteados
        self.assertGreater(float(v[:, 2].abs().max()), 0.4)

    def test_andar_giro_e_zero_ou_pelo_menos_giro_min(self):
        # 24/11, T15: abaixo de 0,1 rad/s a razão ω̄/c_z é ruído; o sorteio que cai aí vira 0. 10 000 sorteios por nível
        g = torch.Generator().manual_seed(3)
        for nivel in range(9):
            for tipo in ("andar", "terreno"):
                v = comando.sorteia_velocidade(tipo, nivel, "ANDAR", 10_000, g, K)
                with self.subTest(nivel=nivel, tipo=tipo):
                    wz = v[:, 2].abs()
                    self.assertFalse(bool(((wz > 0) & (wz < K.giro_min - 1e-7)).any()))
                    if tipo == "andar":  # 27/11, T19: no terreno o wz sorteado é 0 (o rumo faz o giro)
                        self.assertGreater(float((wz >= K.giro_min - 1e-7).float().mean()), 0.3)  # o giro não some
        # nível 0 (teto 0,3): o terço do sorteio em |wz| < 0,1 vira zero, entre os segmentos com comando linear
        v = amostra("andar", 0, "ANDAR")
        ativo = (v[:, :2] != 0).any(dim=1)
        self.assertAlmostEqual(float((v[ativo, 2] == 0).float().mean()), 1 / 3, delta=0.02)

    def test_andar_linear_e_zero_ou_pelo_menos_linear_min(self):
        # 24/11, T16: com ‖c‖ ≤ `limiar_marcha` o relógio fica parado e o g ≈ 1 pagaria a velocidade sem marcha. 10 000 sorteios por
        # nível: nenhum com 0 < ‖c_xy‖ < `linear_min`, e nenhum comando (linear + giro) com ‖c‖ ≤ `limiar_marcha` e um canal
        # ≥ `limiar_canal`. Vale também para o CARREGAR (pisos 0,25, 0,2 e 0,25)
        g = torch.Generator().manual_seed(5)
        casos = [("ANDAR", tipo, n) for n in range(9) for tipo in ("andar", "terreno")]
        casos += [("CARREGAR", tipo, n) for n in range(len(K.frente_tetos)) for tipo in ("reto", "arco", "giro_lugar", "terreno")]
        for tarefa, tipo, nivel in casos:
            v = comando.sorteia_velocidade(tipo, nivel, tarefa, 10_000, g, K)
            with self.subTest(tarefa=tarefa, nivel=nivel, tipo=tipo):
                lin = v[:, :2].norm(dim=-1)
                self.assertFalse(bool(((lin > 0) & (lin < K.linear_min)).any()))
                canal = (v[:, :2].abs() >= K.limiar_canal).any(dim=-1) | (v[:, 2].abs() >= K.limiar_canal)
                self.assertFalse(bool((canal & (v.norm(dim=-1) <= K.limiar_marcha)).any()))
                if tarefa == "ANDAR":
                    self.assertGreater(float((lin >= K.linear_min).float().mean()), 0.3)  # o linear não some

    def test_carregar_giro_respeita_o_piso(self):
        # o CARREGAR sorteia o giro em ±U[0,25; teto]: já está acima de `giro_min`
        for nivel in range(len(K.frente_tetos)):
            for tipo in ("arco", "giro_lugar"):
                wz = amostra(tipo, nivel)[:, 2].abs()
                self.assertGreaterEqual(float(wz.min()), max(K.piso_velocidade, K.giro_min) - 1e-6)

    def test_carregar_lado_tem_piso_de_0_2_e_frente_de_0_25(self):
        # 24/11, T15: o piso do lado é `piso_lado` (0,2) e o do resto do linear, `piso_velocidade` (0,25)
        v = amostra("reto", len(K.frente_tetos) - 1)
        lado = (v[:, 0] == 0) & (v[:, 1] != 0)
        resto = v[:, 1] == 0
        self.assertGreaterEqual(float(v[lado, 1].abs().min()), K.piso_lado - 1e-6)
        self.assertLess(float(v[lado, 1].abs().min()), K.piso_lado + 0.01)  # o piso é atingido
        self.assertGreaterEqual(float(v[resto, 0].abs().min()), K.piso_velocidade - 1e-6)

    def test_andar_sobe_0_1_por_nivel(self):
        # cada nível soma `andar_passo` ao teto de cada canal: no nível 1, frente e trás 0,4, lado 0,3 e giro 0,4
        v = amostra("andar", 1, "ANDAR")
        self.assertAlmostEqual(float(v[:, 0].max()), 0.4, delta=0.01)
        self.assertAlmostEqual(float(v[:, 0].min()), -0.4, delta=0.01)
        self.assertAlmostEqual(float(v[:, 1].abs().max()), 0.3, delta=0.01)
        self.assertAlmostEqual(float(v[:, 2].abs().max()), 0.4, delta=0.01)


class CadeiaDeTrechosTest(unittest.TestCase):

    def test_fatia_c_pegar_por_aviso_e_botar_no_atraso(self):
        tab = comando.monta_tabela_trechos(ATIVO.cadeia, CPU)
        seq = trechos_de(tab, tab.inicio[2])
        # §7, fatia C: ANDAR parado, PEGAR (aviso), CARREGAR parado, BOTAR (aviso), ANDAR parado
        self.assertEqual([int(tab.tarefa[t]) for t in seq], [0, 1, 2, 3, 0])
        self.assertEqual([bool(tab.por_detector[t]) for t in seq], [False, True, False, True, False])

        # PEGAR (13/11, M6: o aviso é o latch do fecho do P3): sem o latch, não avança e o relógio fica em 0
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


class DerivaDoParado(unittest.TestCase):
    def _parado(self, dpos, deriva_pos_m=None):
        # Comando zero (canal lin inativo): só a deriva de posição decide o eixo linear
        return comando.sucesso_do_segmento(
            torch.zeros(1, 3), torch.zeros(1, 2), torch.ones(1, 2), t1(dpos), t1(0.0), t1(5.0), t1(1.5),
            t1(False, torch.bool), t1(False, torch.bool), t1(0.0), t1(False, torch.bool), K,
            deriva_pos_m=deriva_pos_m)

    def test_deriva_de_20_cm_passa_com_o_limite_de_25_cm(self):
        # §6.5: o limite do parado desce de deriva_pos_inicio (0,25 m) a deriva_pos_m (0,10 m) pela restrição
        _, suc = self._parado(0.20, deriva_pos_m=0.25)
        self.assertTrue(bool(suc[0]))

    def test_deriva_de_20_cm_falha_sem_o_parametro(self):
        # Sem o parâmetro vale k.deriva_pos_m (0,10 m): 0,20 m falha
        _, suc = self._parado(0.20)
        self.assertFalse(bool(suc[0]))

if __name__ == "__main__":
    unittest.main()
