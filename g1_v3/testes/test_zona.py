"""Testes da T13 (specs/g1-v3-curriculo-unitree.md §16): a zona adaptativa do fecho de cada etapa do PEGAR e do BOTAR.

O controle do z mora no `balanceador.Curriculo`; as visitas, o `Alvos` efetivo e os tempos moram na `CadeiaCmd`. A cadeia e o
env são falsos, como nos vizinhos (`test_comando_e`): só `_alvos`, `_registra`, `_resample` e `_sigma` rodam, com a cena fora.
"""
import contextlib
import math
import types
import unittest
from unittest import mock

import torch

from g1_v3 import alvos, balanceador, comando, curriculo, observacoes, recompensas
from g1_v3.knobs import ALVOS, ATIVO
from g1_v3.testes import test_comando_e as ce
from g1_v3.testes import test_recompensas_g as rg

A, B = ATIVO.alvos, ATIVO.balanceador
P0, P1, P2, P3, P4, B1, B2, B3, B4 = range(9)
N = balanceador.N_ETAPAS


def novo():
    return balanceador.Curriculo(B, ATIVO.niveis, ATIVO.comando, ATIVO.terreno)


def fecha(cur, etapa, visitas, fechos):
    cur.vis_acum[etapa], cur.fech_acum[etapa] = visitas, fechos
    cur.fecha_iteracao()


class TestControleDoZ(unittest.TestCase):

    def test_comeca_em_1_e_as_etapas_sao_as_de_alvos(self):
        self.assertEqual(N, len(ALVOS))
        self.assertEqual(novo().z, [1.0] * N)

    def test_taxa_acima_do_alvo_aperta_e_abaixo_afrouxa(self):
        # z ← z − k·(taxa − t_alvo): taxa 1 com t_alvo 0,6 e k 0,05 tira 0,02; taxa 0 põe 0,03
        cur = novo()
        fecha(cur, P3, 200, 200)
        self.assertAlmostEqual(cur.z[P3], 1.0 - 0.02, places=9)
        cur.z = [0.5] * N
        fecha(cur, P3, 200, 0)
        self.assertAlmostEqual(cur.z[P3], 0.5 + 0.03, places=9)
        self.assertEqual(cur.z[P2], 0.5)  # as outras etapas não mudam

    def test_z_fica_em_0_e_1(self):
        cur = novo()
        cur.z = [0.01] * N
        fecha(cur, P3, 200, 200)
        self.assertEqual(cur.z[P3], 0.0)
        cur.z = [0.99] * N
        fecha(cur, P3, 200, 0)
        self.assertEqual(cur.z[P3], 1.0)

    def test_sem_atualizacao_com_menos_de_n_min_visitas(self):
        cur = novo()
        n = B.zona_n_min - 1
        fecha(cur, P3, n, n)
        self.assertEqual(cur.z[P3], 1.0)
        self.assertEqual((cur.vis_acum[P3], cur.fech_acum[P3]), (n, n))  # as visitas esperam a próxima iteração
        fecha(cur, P3, n + 1, n + 1)  # 200 visitas acumuladas: agora atualiza
        self.assertAlmostEqual(cur.z[P3], 0.98, places=9)

    def test_acumuladores_zeram_so_na_etapa_atualizada(self):
        cur = novo()
        cur.vis_acum[P4], cur.fech_acum[P4] = 50, 10
        fecha(cur, P3, 200, 120)
        self.assertEqual((cur.vis_acum[P3], cur.fech_acum[P3]), (0, 0))
        self.assertEqual((cur.vis_acum[P4], cur.fech_acum[P4]), (50, 10))
        self.assertAlmostEqual(cur.taxa[P3], 0.6, places=9)  # a última taxa fica para o log; com 0,6 o z não se mexe
        self.assertEqual(cur.z[P3], 1.0)

    def test_ordem_da_cadeia_uma_etapa_nao_fica_mais_apertada_que_a_anterior_por_mais_de_0_2(self):
        cur = novo()
        cur.z = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0]
        cur.fecha_iteracao()
        for obtido, esperado in zip(cur.z, (1.0, 0.8, 0.6, 0.4, 0.2, 1.0, 0.8, 0.6, 0.4)):
            self.assertAlmostEqual(obtido, esperado, places=9)
        cur.z = [0, 0, 0, 0, 0.5, 0, 0, 0, 0]  # o B1 não segue o P4: a cadeia do BOTAR é outra
        cur.fecha_iteracao()
        self.assertEqual(cur.z[B1], 0.0)

    def test_a_ordem_vale_depois_da_atualizacao(self):
        # P2 em 0,3 com o P1 em 1: a atualização aperta o P2 a 0,28, e a ordem o põe em 0,8
        cur = novo()
        cur.z = [1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        fecha(cur, P2, 200, 200)
        self.assertAlmostEqual(cur.z[P2], 0.8, places=9)

    def test_estado_com_e_sem_a_chave(self):
        cur = novo()
        cur.z[2], cur.vis_acum[3], cur.fech_acum[3] = 0.4, 17, 5
        est = cur.estado()
        outro = novo()
        outro.carrega(est)
        self.assertEqual((outro.z, outro.vis_acum, outro.fech_acum), (cur.z, cur.vis_acum, cur.fech_acum))
        self.assertEqual(outro.estado(), est)
        del est["zona"]  # checkpoint de antes da T13: z = 1 e acumuladores em 0
        outro.carrega(est)
        self.assertEqual((outro.z, outro.vis_acum, outro.fech_acum), ([1.0] * N, [0] * N, [0] * N))


class TestTabelaDaZona(unittest.TestCase):

    def test_largo_de_cada_criterio_e_o_da_spec(self):
        esperado = {("P0", "tol_palma"): 0.05, ("P1", "tol_palma"): 0.05, ("P2", "tempo_p2"): 0.2,
                    ("P3", "tol_caixa_p3"): 0.04, ("P3", "tempo_p3"): 0.1, ("P4", "tol_caixa_p4"): 0.20,
                    ("P4", "tol_pelve_p4"): 0.10, ("P4", "tempo_p4"): 0.2, ("B1", "tol_b1_horizontal"): 0.10,
                    ("B2", "raio_b2"): 0.20, ("B2", "apoio_b2_mg"): 0.5, ("B2", "tempo_b2"): 0.1,
                    ("B3", "b3_afasta"): 0.02, ("B3", "tempo_b3"): 0.2,
                    ("B4", "tol_pelve_b4"): 2 * ATIVO.detector.tol_pelve,
                    ("B4", "tol_tronco_b4_graus"): 2 * ATIVO.detector.tol_tronco_graus,
                    ("B4", "tol_palmas_b4"): 2 * ATIVO.detector.tol_palma, ("B4", "tempo_b4"): 0.2}
        self.assertEqual({(e, c): v for e, c, v in A.zona}, esperado)
        for _, campo, _ in A.zona:
            self.assertTrue(hasattr(A, campo), campo)

    def test_piso_do_sigma_por_etapa(self):
        self.assertEqual(A.sigma_min_etapa, (0.05, 0.05, 0.08, 0.05, 0.20, 0.10, 0.20, 0.05, 0.0))
        self.assertFalse(hasattr(A, "sigma_min"))


# --- a cadeia: Alvos efetivo, tempos, visitas -------------------------------------------------------------------

def _cadeia(idx, z, tarefa=comando.PEGAR):
    env = ce._env_falso(idx, tarefa)
    env.v3_cur.z = list(z)
    obj = ce._cadeia_sem_env(env)
    obj._sigma = lambda m, indice: torch.ones(1)
    return obj, env


def _alvos(obj, m, dt=ce.DT, ok_pegar=None):
    # `_alvos` com a cena de fora; `ok_pegar` (bool) força o fecho do PEGAR, sem ele o núcleo real decide
    patches = [mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=torch.zeros(1, 3)),
               mock.patch.object(recompensas, "incl_lim", return_value=torch.tensor([30.0])),
               mock.patch.object(recompensas, "portoes_do_alvo",
                                 return_value=dict(desloc=torch.ones(1), incl=torch.ones(1), maos=torch.ones(1))),
               mock.patch.object(comando, "ok_agora_botar", return_value=torch.tensor([False]))]
    if ok_pegar is not None:
        patches.append(mock.patch.object(comando, "ok_agora_pegar", return_value=torch.tensor([ok_pegar])))
    with contextlib.ExitStack() as pilha:
        for p in patches:
            pilha.enter_context(p)
        obj._alvos(m, dt)


def _medidas_p3(subida):
    m = ce._medidas()
    m["apoio"] = torch.zeros(1, dtype=torch.bool)  # a caixa fora da laje
    m["h_caixa"] = torch.tensor([0.1 + subida])  # `a` = 0,1 e o topo = 0: subida = h_caixa − a − topo
    return m


class TestCriterioEfetivo(unittest.TestCase):

    def _p3(self, z, subida, passos):
        obj, env = _cadeia(P3, [z] * N)
        m = _medidas_p3(subida)
        for _ in range(passos):
            _alvos(obj, m)
        return env.v3_alvo_idx.item()

    def test_com_z_1_o_p3_fecha_com_1_cm_por_0_1_s_e_nao_fecha_com_0_5_cm(self):
        self.assertEqual(self._p3(1.0, 0.0105, 6), P4)
        self.assertEqual(self._p3(1.0, 0.005, 30), P3)

    def test_com_z_0_o_p3_e_o_de_hoje_3_cm_por_0_3_s(self):
        self.assertEqual(self._p3(0.0, 0.0105, 30), P3)  # 1 cm não basta
        self.assertEqual(self._p3(0.0, 0.035, 10), P3)  # 3 cm, mas ainda sem os 0,3 s
        self.assertEqual(self._p3(0.0, 0.035, 16), P4)

    def test_os_tempos_acompanham_o_z_e_so_mudam_na_virada_da_iteracao(self):
        obj, env = _cadeia(P3, [1.0] * N)
        m = _medidas_p3(0.0)
        _alvos(obj, m)
        self.assertEqual([round(t, 6) for t in obj.tempos.tolist()], [0, 0, 0.2, 0.1, 0.2, 0, 0.1, 0.2, 0.2])
        self.assertAlmostEqual(env.v3_alvos.tol_caixa_p4, 0.20)
        env.v3_cur.z = [0.5] * N
        _alvos(obj, m)  # a mesma iteração: nada muda
        self.assertAlmostEqual(obj.tempos[P3].item(), 0.1)
        env.v3_cur.it += 1  # a iteração virou
        _alvos(obj, m)
        self.assertAlmostEqual(obj.tempos[P3].item(), 0.2)
        self.assertAlmostEqual(env.v3_alvos.tol_caixa_p3, 0.03)
        env.v3_cur.z = [1.0] * N
        _alvos(obj, m)  # a iteração 1 já foi lida: o z novo espera a iteração 2
        self.assertAlmostEqual(obj.tempos[P3].item(), 0.2)
        env.v3_cur.z = [0.0] * N
        env.v3_cur.it += 1
        _alvos(obj, m)
        self.assertEqual([round(t, 6) for t in obj.tempos.tolist()], [0, 0, 0.3, 0.3, 0.5, 0, 0.3, 0.5, 0.5])
        for _, campo, _ in A.zona:  # com z = 0 cada critério volta ao final
            if campo != "tol_palma":
                self.assertAlmostEqual(getattr(env.v3_alvos, campo), getattr(A, campo), places=9, msg=campo)
        for v in env.v3_alvos.tol_palma.tolist():
            self.assertAlmostEqual(v, A.tol_palma, places=6)

    def test_p0_e_p1_usam_o_z_de_cada_uma(self):
        # as duas palmas a 4 cm da pose: entre o final (3 cm) e o largo (5 cm). P0 com z = 1 fecha; P1 com z = 0 não
        m = ce._medidas()
        m["palmas_c"] = torch.tensor([[[0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]])
        for idx, z, fecha_ in ((P0, [1.0, 0.0] + [0.0] * 7, True), (P1, [1.0, 0.0] + [0.0] * 7, False),
                               (P0, [0.0, 1.0] + [0.0] * 7, False), (P1, [0.0, 1.0] + [0.0] * 7, True)):
            with self.subTest(idx=idx, z=z[:2]):
                obj, env = _cadeia(idx, z)
                with mock.patch.object(alvos, "palmas_alvo_caixa", return_value=torch.zeros(1, 2, 3)):
                    _alvos(obj, m)
                self.assertEqual(env.v3_alvo_idx.item(), idx + 1 if fecha_ else idx)

    def test_o_d_e_o_phi_ficam_no_valor_final(self):
        obj, env = _cadeia(P4, [1.0] * N)
        m = ce._medidas()
        _alvos(obj, m)
        self.assertIsNot(env.v3_alvos, A)
        self.assertAlmostEqual(env.v3_alvos.tol_caixa_p4, 0.20)
        with mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=torch.zeros(1, 3)), \
                mock.patch.object(recompensas, "distancia_do_alvo", return_value=torch.zeros(1)) as dist:
            obj._dist(m, torch.tensor([P4]))
        self.assertIs(dist.call_args.args[-1], A)  # o `d` da cadeia (o σ) lê o final

    def test_a_qualidade_do_carregar_muda_com_o_z_do_p4(self):
        # a caixa a 17 cm do ponto de transporte: com o P4 largo (20 cm) a qualidade fica perto de 0,49; com o final, 0
        obj, env = _cadeia(P4, [0.0] * 4 + [1.0] + [0.0] * 4)
        _alvos(obj, ce._medidas())
        e = rg.env_falso(-1, yaw=math.pi / 2, caixa_z=rg.Z_ALVO_DE_HOJE + 0.17)
        e.v3_alvos = env.v3_alvos
        self.assertAlmostEqual(recompensas._qualidade(e)[0].item(), math.exp(-((0.17 / 0.20) ** 2)), places=3)
        obj, env = _cadeia(P4, [0.0] * N)
        _alvos(obj, ce._medidas())
        e.v3_alvos = env.v3_alvos
        self.assertLess(recompensas._qualidade(e)[0].item(), 1e-3)

    def test_sigma_tem_piso_por_etapa_e_o_b4_fica_sem_piso(self):
        obj = comando.CadeiaCmd.__new__(comando.CadeiaCmd)
        obj.sigma_min = torch.tensor(A.sigma_min_etapa)
        obj._dist = lambda m, indice: torch.full((N,), 0.001)
        sigma = comando.CadeiaCmd._sigma(obj, {}, torch.arange(N))
        self.assertEqual([round(s, 6) for s in sigma.tolist()], [0.05, 0.05, 0.08, 0.05, 0.20, 0.10, 0.20, 0.05, 0.001])
        obj._dist = lambda m, indice: torch.full((N,), 0.5)
        self.assertEqual(comando.CadeiaCmd._sigma(obj, {}, torch.arange(N)).tolist(), [0.5] * N)  # acima do piso: o d


class TestVisitas(unittest.TestCase):

    def _contas(self, env):
        return env.v3_cur.vis_acum[P3], env.v3_cur.fech_acum[P3]

    def test_fecho_conta_visita_e_fecho(self):
        obj, env = _cadeia(P1, [0.0] * N)
        _alvos(obj, ce._medidas(), ok_pegar=True)  # o P1 tem tempo 0: abre e fecha no mesmo passo
        self.assertEqual((env.v3_cur.vis_acum[P1], env.v3_cur.fech_acum[P1]), (1, 1))
        self.assertEqual(env.v3_alvo_idx.item(), P2)
        _alvos(obj, ce._medidas(), ok_pegar=True)  # o P2 abriu: a visita está aberta e ainda não acabou
        self.assertEqual(env.v3_cur.vis_acum, [0, 1] + [0] * 7)
        self.assertEqual(env.v3_cur.fech_acum, [0, 1] + [0] * 7)

    def test_t_visita_sem_fecho_conta_falha_uma_vez_so(self):
        obj, env = _cadeia(P3, [0.0] * N)
        m = _medidas_p3(0.0)
        for _ in range(7):
            _alvos(obj, m, dt=1.0, ok_pegar=False)
        self.assertEqual(self._contas(env), (0, 0))
        _alvos(obj, m, dt=1.0, ok_pegar=False)  # 8 s desde a abertura
        self.assertEqual(self._contas(env), (1, 0))
        for _ in range(20):
            _alvos(obj, m, dt=1.0, ok_pegar=False)
        self.assertEqual(self._contas(env), (1, 0))  # a visita acabou: a etapa aberta não conta de novo
        _alvos(obj, m, dt=1.0, ok_pegar=True)  # o fecho depois do T_visita não vira fecho da visita que já acabou
        self.assertEqual(env.v3_alvo_idx.item(), P4)
        self.assertEqual(self._contas(env), (1, 0))

    def test_perda_da_caixa_conta_falha(self):
        obj, env = _cadeia(P3, [0.0] * N)
        m = _medidas_p3(0.0)
        _alvos(obj, m, ok_pegar=False)
        env.v3_perdeu[:] = True  # o `_perda` marca a perda e põe o alvo no B4
        env.v3_alvo_idx[:] = B4
        _alvos(obj, m, ok_pegar=False)
        self.assertEqual(self._contas(env), (1, 0))
        _alvos(obj, m, dt=10.0, ok_pegar=False)  # 10 s com a caixa perdida: o B4 dela não é visita e não vence
        self.assertEqual(self._contas(env), (1, 0))
        self.assertEqual(env.v3_cur.vis_acum[B4], 0)

    def test_troca_de_tarefa_sem_fecho_conta_falha(self):
        obj, env = _cadeia(P3, [0.0] * N)
        m = _medidas_p3(0.0)
        _alvos(obj, m, ok_pegar=False)
        env.v3_tarefa[:] = comando.CARREGAR  # o trecho acabou por fora: o alvo sai de cena
        env.v3_alvo_idx[:] = -1
        _alvos(obj, m, ok_pegar=False)
        self.assertEqual(self._contas(env), (1, 0))

    def test_reset_conta_falha_e_nada_conta_duas_vezes(self):
        obj = ce._cadeia_de_trechos()
        env = obj._env
        env.v3_tarefa[:] = comando.PEGAR
        env.v3_alvo_idx[:] = P3
        _alvos(obj, _medidas_p3(0.0), ok_pegar=False)
        self.assertEqual(env.v3_cur.vis_acum[P3], 0)  # a visita está aberta
        obj._resample(torch.tensor([0]))
        self.assertEqual((env.v3_cur.vis_acum[P3], env.v3_cur.fech_acum[P3]), (1, 0))
        obj._resample(torch.tensor([0]))  # um segundo reset sem visita aberta
        self.assertEqual(env.v3_cur.vis_acum[P3], 1)
        self.assertEqual(obj.vis_e.tolist(), [-1])

    def test_p4_fechado_nao_e_visita_aberta(self):
        obj, env = _cadeia(P4, [0.0] * N)
        env.v3_p4_fechado[:] = True
        _alvos(obj, ce._medidas(), dt=10.0, ok_pegar=False)  # 10 s com o P4 fechado: nada vence
        self.assertEqual(env.v3_cur.vis_acum, [0] * N)

    def test_bota_fica_visitada_com_o_p4_do_pega_fechado_no_episodio(self):
        # o `v3_p4_fechado` do PEGAR fica True no episódio inteiro; o B1 do mesmo episódio é visita
        obj, env = _cadeia(B1, [0.0] * N, tarefa=comando.BOTAR)
        env.v3_p4_fechado[:] = True
        _alvos(obj, ce._medidas(), ok_pegar=False)
        self.assertEqual(obj.vis_e.tolist(), [B1])


class TestSucessoSoComOCriterioFinal(unittest.TestCase):

    def _registra(self, z_pega, z_bota, botar):
        obj = ce._cadeia_de_trechos()
        env = obj._env
        env.v3_cur.z = list(z_pega) + list(z_bota)
        env.v3_p4_fechado[:] = True
        env.v3_b4_fechado[:] = True
        obj.pega[:] = obj.bota[:] = True
        obj._registra(torch.tensor([0]), torch.tensor([False]), botar=botar)
        return env.v3_cur.registros

    def test_pegar_com_z_acima_de_0_registra_falha(self):
        self.assertEqual(self._registra([0, 0, 0, 0, 0.3], [0] * 4, False), [(1, 0)])
        self.assertEqual(self._registra([0.3, 0, 0, 0, 0], [0] * 4, False), [(1, 0)])
        self.assertEqual(self._registra([0] * 5, [0] * 4, False), [(1, 1)])

    def test_botar_so_olha_o_z_do_botar(self):
        self.assertEqual(self._registra([0] * 5, [0, 0, 0.2, 0], True), [(1, 0)])
        self.assertEqual(self._registra([0.5] * 5, [0] * 4, True), [(1, 1)])  # o PEGAR aberto não trava o BOTAR
        self.assertEqual(self._registra([0] * 5, [0.5] * 4, False), [(1, 1)])  # nem o BOTAR trava o PEGAR


class TestObservacaoEPublicacao(unittest.TestCase):

    def test_critico_ve_o_z_da_etapa_e_zero_fora_de_etapa(self):
        env = types.SimpleNamespace(
            v3_fase_marcha=torch.zeros(3), v3_marcha_on=torch.zeros(3, dtype=torch.bool), v3_h=torch.ones(3),
            v3_phi_melhor=torch.zeros(3), v3_phi=torch.zeros(3), v3_sigma=torch.full((3,), 7.0),
            v3_alvo_idx=torch.tensor([P0, B4, -1]), v3_cur=types.SimpleNamespace(z=[0.1 * (i + 1) for i in range(N)]))
        self.assertTrue(torch.allclose(observacoes.privilegiados(env)[:, 3], torch.tensor([0.1, 0.9, 0.0])))

    def test_log_traz_z_e_taxa_de_cada_etapa(self):
        env = types.SimpleNamespace(num_envs=4, device="cpu", common_step_counter=0, v3_tarefa=torch.zeros(4, dtype=torch.long))
        curriculo.garante_v3(env)
        env.v3_cur.z[P3], env.v3_cur.taxa[P3] = 0.4, 0.55
        log = curriculo.v3(env, torch.arange(4))
        for nome in ALVOS:
            self.assertIn(f"z_{nome}", log)
            self.assertIn(f"taxa_{nome}", log)
        self.assertEqual((log["z_P3"], log["taxa_P3"], log["z_P0"]), (0.4, 0.55, 1.0))


if __name__ == "__main__":
    unittest.main()
