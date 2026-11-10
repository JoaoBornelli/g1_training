"""Testes da lane G3 da fase G de g1_v3/comando.py (§6.3 "Fase G"): o sucesso do segmento do CARREGAR no terreno, o
`v3_incl_max_seg` do twist e o B4 da caixa perdida medido pela pelve sobre o pé mais baixo (F9, R6 da E11).
Escritos antes do conserto. Os testes de env usam env falso (`types.SimpleNamespace`), sem a cena.
"""
import dataclasses
import math
import types
import unittest
from unittest import mock

import torch

from g1_v3 import alvos, comando, recompensas
from g1_v3.knobs import ATIVO

A = ATIVO.alvos
# k próprio do núcleo: os campos de `ATIVO.comando` (inclui `incl_max_terreno_graus`, G1)
K = types.SimpleNamespace(**dataclasses.asdict(ATIVO.comando))
P0, P1, P2, P3, P4, B1, B2, B3, B4 = range(9)
CARREGAR_TERRENO = 3  # id de célula falso: ("CARREGAR", "terreno")


def segmento_g(incl, terreno, carregar=True, cmd=(0.5, 0.0, 0.0), k=K):
    # Segmento que passa em todos os canais (razão 0,6 no linear) e só varia a inclinação e o terreno
    t = lambda v, dt=torch.float32: torch.tensor([v], dtype=dt)
    return comando.sucesso_do_segmento(
        torch.tensor([list(cmd)]), torch.tensor([[0.6, 0.0]]), torch.tensor([[1.0, 0.0]]),
        t(0.0), t(0.0), t(5.0), t(1.5), t(False, torch.bool), t(carregar, torch.bool), t(incl), t(terreno, torch.bool),
        k)


def _quat_x(graus):
    # wxyz: rotação em torno de x. A face mais vertical da caixa sai da vertical por `graus`
    r = math.radians(graus) / 2
    return torch.tensor([[math.cos(r), math.sin(r), 0.0, 0.0]])


class SucessoDoSegmentoNoTerreno(unittest.TestCase):

    def test_carregar_no_terreno_com_16_graus_falha(self):
        _, suc = segmento_g(incl=16.0, terreno=True)
        self.assertFalse(bool(suc))

    def test_carregar_no_terreno_com_14_graus_passa(self):
        _, suc = segmento_g(incl=14.0, terreno=True)
        self.assertTrue(bool(suc))

    def test_fora_do_terreno_com_16_graus_passa(self):
        _, suc = segmento_g(incl=16.0, terreno=False)
        self.assertTrue(bool(suc))

    def test_andar_no_terreno_com_16_graus_passa(self):
        # ANDAR sem caixa: o limiar da inclinação não vale
        _, suc = segmento_g(incl=16.0, terreno=True, carregar=False)
        self.assertTrue(bool(suc))


def _twist_falso(env, n=1):
    # `TwistV3` sem o `__init__` (que pede a cena e o fabricante): só o que `_mede_incl`, `_abre` e `_fecha` leem
    obj = comando.TwistV3.__new__(comando.TwistV3)
    obj._env = env
    obj._agora = torch.zeros(n)
    obj.vel_command_b = torch.zeros(n, 3)
    obj.seg_cmd = torch.zeros(n, 3)
    obj.seg_cel = torch.full((n,), -1, dtype=torch.long)
    obj.seg_carrega = torch.zeros(n, dtype=torch.bool)
    obj.seg_visto = torch.zeros(n, dtype=torch.long)
    obj.command_counter = torch.zeros(n, dtype=torch.long)
    obj.seg_proj = torch.zeros(n, 2)
    obj.seg_int = torch.zeros(n, 2)
    obj.ancora_feita = torch.zeros(n, dtype=torch.bool)
    obj.t_ab = torch.zeros(n)
    return obj


def _caixa_falsa(graus):
    return types.SimpleNamespace(data=types.SimpleNamespace(root_link_quat_w=_quat_x(graus)))


class InclMaxDoSegmento(unittest.TestCase):

    def test_caixa_a_10_20_5_graus_guarda_o_maior(self):
        env = types.SimpleNamespace(v3_incl_max_seg=torch.zeros(1), scene={"box": _caixa_falsa(10)})
        obj = _twist_falso(env)
        for graus in (10, 20, 5):
            env.scene["box"] = _caixa_falsa(graus)
            obj._mede_incl()
        self.assertAlmostEqual(float(env.v3_incl_max_seg[0]), 20.0, places=3)

    def test_abertura_do_segmento_zera_o_incl_max(self):
        env = types.SimpleNamespace(v3_incl_max_seg=torch.tensor([20.0]), v3_celula=torch.tensor([CARREGAR_TERRENO]),
                                    v3_tarefa=torch.tensor([comando.CARREGAR]))
        obj = _twist_falso(env)
        obj._abre(torch.arange(1))
        self.assertEqual(env.v3_incl_max_seg.tolist(), [0.0])


class FechoDoSegmentoNoTerreno(unittest.TestCase):
    # `_fecha` real, com o núcleo real: o `terreno` sai da célula do segmento e o `incl_max_seg` do buffer do env

    def _fecha(self, incl, cel, carregar):
        env = types.SimpleNamespace(
            v3_ancora=torch.tensor([[0.0, 0.0, 0.0, 1.5]]), v3_perdeu=torch.zeros(1, dtype=torch.bool),
            v3_incl_max_seg=torch.tensor([float(incl)]),
            v3_cur=types.SimpleNamespace(
                id_celula=lambda tarefa, tipo: CARREGAR_TERRENO if (tarefa, tipo) == ("CARREGAR", "terreno") else -1,
                registra=mock.Mock()))
        obj = _twist_falso(env)
        obj._env, obj.k = env, K
        obj.pose_ult = torch.zeros(1, 3)
        obj._agora = torch.tensor([5.0])
        obj.seg_cmd = torch.tensor([[0.5, 0.0, 0.0]])
        obj.seg_cel = torch.tensor([cel])
        obj.seg_carrega = torch.tensor([carregar])
        obj.seg_proj = torch.tensor([[0.6, 0.0]])
        obj.seg_int = torch.tensor([[1.0, 0.0]])
        obj.metrics = {k: torch.zeros(1) for k in ("eficiencia_min", "eficiencia_media", "segmentos")}
        obj._fecha(torch.tensor([0]), torch.tensor([False]))
        return env.v3_cur.registra

    def test_carregar_no_terreno_a_16_graus_registra_sem_sucesso(self):
        registra = self._fecha(16.0, CARREGAR_TERRENO, True)
        registra.assert_called_once_with(CARREGAR_TERRENO, 1, 0)

    def test_carregar_no_terreno_a_14_graus_registra_com_sucesso(self):
        registra = self._fecha(14.0, CARREGAR_TERRENO, True)
        registra.assert_called_once_with(CARREGAR_TERRENO, 1, 1)

    def test_celula_que_nao_e_terreno_a_16_graus_registra_com_sucesso(self):
        registra = self._fecha(16.0, CARREGAR_TERRENO + 4, True)
        registra.assert_called_once_with(CARREGAR_TERRENO + 4, 1, 1)


def _cadeia_falsa(env, n=1):
    # `CadeiaCmd` sem o `__init__`: só o que `_alvos`, `_detector` e `_abre_p0` leem
    obj = comando.CadeiaCmd.__new__(comando.CadeiaCmd)
    obj._env = env
    obj.pega = torch.zeros(n, dtype=torch.bool)
    obj.tempos = torch.tensor(comando._TEMPOS, dtype=torch.float32)
    obj._sigma = lambda m, indice: torch.zeros(n)
    return obj


def _ref_falsa():
    # pelve_de_pe = 0,789 (a referência do robô); a pelve com caixa e as palmas não entram neste teste
    return types.SimpleNamespace(
        pelve_de_pe=0.789, pelve_com_caixa=lambda a: torch.zeros(1),
        palmas_de_pe_b=torch.zeros(1, 2, 3), palmas_com_caixa_b=lambda a: torch.zeros(1, 2, 3))


class B4DaCaixaPerdidaPelaPelve(unittest.TestCase):

    def test_b4_mede_a_pelve_sobre_o_pe_mais_baixo_e_nao_o_h(self):
        # Degrau: `h` (altura sob a pelve) = 0,50, mas a pelve sobre o pé mais baixo = 1,00 − 0,10 = 0,90.
        # O B4 segue a pelve sobre o pé: dpelve = 0,90 − 0,789 = 0,111 (e não 0,50 − 0,789 = −0,289).
        z = torch.zeros(1)
        env = types.SimpleNamespace(
            v3_alvo_idx=torch.tensor([B4]), v3_tarefa=torch.tensor([comando.ANDAR]),
            v3_perdeu=torch.tensor([True]), v3_meia=torch.tensor([0.1]), v3_ref=_ref_falsa(),
            v3_p4_fechado=torch.zeros(1, dtype=torch.bool), v3_pegou=torch.zeros(1, dtype=torch.bool),
            v3_b4_fechado=torch.zeros(1, dtype=torch.bool), v3_avancou=torch.zeros(1, dtype=torch.bool),
            v3_relogio_alvo=z.clone(), v3_alvo_w=torch.zeros(1, 3), v3_caixa_pega_w=torch.zeros(1, 3),
            v3_ponto_apoio=torch.zeros(1, 3), v3_sigma=torch.ones(1),
            v3_phi=torch.zeros(1))
        m = dict(h=torch.full((1,), 0.50), pelve=torch.full((1,), 0.90), caixa_p=torch.zeros(1, 3),
                 palmas_c=torch.zeros(1, 2, 3), contatos=torch.zeros(1, 2, dtype=torch.bool),
                 apoio=torch.zeros(1, dtype=torch.bool), incl_graus=z, tronco_graus=z,
                 v_caixa=torch.zeros(1, 3), apoio_N=z.clone(), peso_N=torch.ones(1))
        obj = _cadeia_falsa(env)
        ok_todos = dict(desloc=torch.ones(1), incl=torch.ones(1), maos=torch.ones(1))
        with mock.patch.object(comando, "ok_agora_botar", return_value=torch.zeros(1, dtype=torch.bool)) as botar, \
                mock.patch.object(comando, "ok_agora_pegar", return_value=torch.zeros(1, dtype=torch.bool)), \
                mock.patch.object(alvos, "palmas_alvo_caixa", return_value=torch.zeros(1, 2, 3)), \
                mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=torch.zeros(1, 3)), \
                mock.patch.object(recompensas, "incl_lim", return_value=torch.tensor([30.0])), \
                mock.patch.object(recompensas, "portoes_do_alvo", return_value=ok_todos):
            obj._alvos(m, 0.02)
        dpelve = botar.call_args.args[8]  # posição de `dpelve` em `ok_agora_botar`
        self.assertAlmostEqual(float(dpelve[0]), 0.111, places=6)

    def test_detector_usa_a_pelve_da_medida(self):
        # `_detector` e `_alvos` leem a mesma pelve: a de `_mede` (`m["pelve"]`). Aqui `z_pe_min` é mockado para
        # 0,10, e `m["pelve"]` = 0,42 difere de root_z − 0,10 = 0,90: o detector tem de receber 0,42.
        env = types.SimpleNamespace(
            v3_ref=_ref_falsa(), v3_meia=torch.tensor([0.1]), v3_tarefa=torch.tensor([comando.ANDAR]),
            v3_alvo_idx=torch.tensor([-1]), v3_trecho=torch.tensor([0]), v3_aviso=torch.zeros(1, dtype=torch.bool))
        obj = _cadeia_falsa(env)
        obj.robot = types.SimpleNamespace(data=types.SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 1.0]]), joint_vel=torch.zeros(1, 4)))
        obj.rel_fim = torch.zeros(1)
        obj.tab = types.SimpleNamespace(por_detector=torch.tensor([True]))
        m = dict(base_q=torch.tensor([[1.0, 0.0, 0.0, 0.0]]), base_p=torch.zeros(1, 3),
                 palmas_w=torch.zeros(1, 2, 3), tronco_graus=torch.zeros(1), pelve=torch.tensor([0.42]))
        with mock.patch.object(comando, "z_pe_min", return_value=torch.tensor([0.10])), \
                mock.patch.object(alvos, "detecta_fim", return_value=(torch.zeros(1, dtype=torch.bool),
                                                                      torch.zeros(1))) as det:
            obj._detector(m, 0.02)
        self.assertAlmostEqual(float(det.call_args.args[0][0]), 0.42, places=6)


if __name__ == "__main__":
    unittest.main()
