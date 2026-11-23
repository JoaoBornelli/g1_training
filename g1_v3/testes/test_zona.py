"""Testes da T13 (specs/g1-v3-curriculo-unitree.md §16): a zona adaptativa do fecho de cada etapa do PEGAR e do BOTAR.

O controle do z mora no `balanceador.Curriculo`; as visitas, o `Alvos` efetivo e os tempos moram na `CadeiaCmd`. A cadeia e o
env são falsos, como nos vizinhos (`test_comando_e`): só `_alvos`, `_registra`, `_resample` e `_sigma` rodam, com a cena fora.
"""
import contextlib
import math
import types
import unittest
from dataclasses import replace
from unittest import mock

import torch

from g1_v3 import alvos, balanceador, comando, curriculo, observacoes, recompensas
from g1_v3.knobs import ALVOS, ATIVO
from g1_v3.testes import test_comando_e as ce
from g1_v3.testes import test_recompensas_g as rg

A, B = ATIVO.alvos, ATIVO.balanceador
P0, P1, P2, P3, B1, B2, B3 = range(7)
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
        self.assertEqual(novo().taxa, [-1.0] * N)  # sem medida: o log não mostra 0 antes da primeira atualização

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
        self.assertEqual(cur.taxa[P3], -1.0)
        fecha(cur, P3, n + 1, n + 1)  # 200 visitas acumuladas: agora atualiza
        self.assertAlmostEqual(cur.z[P3], 0.98, places=9)

    def test_acumuladores_zeram_so_na_etapa_atualizada(self):
        cur = novo()
        cur.vis_acum[B1], cur.fech_acum[B1] = 50, 10
        fecha(cur, P3, 200, 120)
        self.assertEqual((cur.vis_acum[P3], cur.fech_acum[P3]), (0, 0))
        self.assertEqual((cur.vis_acum[B1], cur.fech_acum[B1]), (50, 10))
        self.assertAlmostEqual(cur.taxa[P3], 0.6, places=9)  # a última taxa fica para o log; com 0,6 o z não se mexe
        self.assertEqual(cur.z[P3], 1.0)

    def test_ordem_da_cadeia_uma_etapa_nao_fica_mais_apertada_que_a_anterior_por_mais_de_0_2(self):
        cur = novo()
        cur.z = [1.0, 0, 0, 0, 1.0, 0, 0]
        cur.fecha_iteracao()
        for obtido, esperado in zip(cur.z, (1.0, 0.8, 0.6, 0.4, 1.0, 0.8, 0.6)):
            self.assertAlmostEqual(obtido, esperado, places=9)
        cur.z = [0, 0, 0, 0.5, 0, 0, 0]  # o B1 não segue o P3: a cadeia do BOTAR é outra
        cur.fecha_iteracao()
        self.assertEqual(cur.z[B1], 0.0)

    def test_a_ordem_vale_depois_da_atualizacao(self):
        # P2 em 0,3 com o P1 em 1: a atualização aperta o P2 a 0,28, e a ordem o põe em 0,8
        cur = novo()
        cur.z = [1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0]
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

    def test_estado_de_9_etapas_converte_para_7(self):
        # 23/11, T14: um checkpoint de antes da T14 guarda a zona de 9 etapas. O P0, P1 e P2 têm os mesmos índices e levam o z, as
        # visitas e os fechos; as outras etapas recomeçam largas (z = 1), sem visitas, sem fechos e com a taxa em −1
        est = novo().estado()
        est["zona"] = {"z": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "visitas": [10, 20, 30, 40, 50, 60, 70, 80, 90],
                       "fechos": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
        cur = novo()
        cur.taxa = [0.5] * N  # um valor velho: a conversão o devolve a −1
        cur.carrega(est)
        self.assertEqual(cur.z, [0.1, 0.2, 0.3] + [1.0] * 4)
        self.assertEqual(cur.vis_acum, [10, 20, 30] + [0] * 4)
        self.assertEqual(cur.fech_acum, [1, 2, 3] + [0] * 4)
        self.assertEqual(cur.taxa, [-1.0] * N)


class TestTabelaDaZona(unittest.TestCase):

    def test_largo_de_cada_criterio_e_o_da_spec(self):
        esperado = {("P0", "tol_palma"): 0.05, ("P1", "tol_palma"): 0.05, ("P2", "tempo_p2"): 0.2,
                    ("P3", "subida_min_p3"): 0.01, ("P3", "tol_transporte_p3"): 0.20, ("P3", "tol_pelve_p3"): 0.10,
                    ("P3", "tempo_p3"): 0.2, ("B1", "raio_b1"): 0.20, ("B1", "tempo_b1"): 0.1,
                    ("B2", "b2_afasta"): 0.02, ("B2", "tempo_b2"): 0.2,
                    ("B3", "tol_pelve_b3"): 2 * ATIVO.detector.tol_pelve,
                    ("B3", "tol_tronco_b3_graus"): 2 * ATIVO.detector.tol_tronco_graus,
                    ("B3", "tol_palmas_b3"): 2 * ATIVO.detector.tol_palma, ("B3", "tempo_b3"): 0.2}
        self.assertEqual({(e, c): v for e, c, v in A.zona}, esperado)
        for _, campo, _ in A.zona:
            self.assertTrue(hasattr(A, campo), campo)

    def test_piso_do_sigma_por_etapa(self):
        self.assertEqual(A.sigma_min_etapa, (0.05, 0.05, 0.08, 0.20, 0.20, 0.05, 0.0))
        self.assertFalse(hasattr(A, "sigma_min"))


# --- a cadeia: Alvos efetivo, tempos, visitas -------------------------------------------------------------------

def _cadeia(idx, z, tarefa=comando.PEGAR):
    env = ce._env_falso(idx, tarefa)
    env.v3_cur.z = list(z)
    obj = ce._cadeia_sem_env(env)
    obj._sigma = lambda m, indice: torch.ones(1)
    return obj, env


def _alvos(obj, m, dt=ce.DT, ok_pegar=None, ok_botar=False):
    # `_alvos` com a cena de fora; `ok_pegar` (bool) força o fecho do PEGAR, sem ele o núcleo real decide. `ok_botar`
    # força o do BOTAR (False por padrão); None deixa o núcleo real decidir, com o `Alvos` efetivo
    patches = [mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=torch.zeros(1, 3)),
               mock.patch.object(recompensas, "incl_lim", return_value=torch.tensor([30.0])),
               mock.patch.object(recompensas, "portoes_do_alvo",
                                 return_value=dict(desloc=torch.ones(1), incl=torch.ones(1), maos=torch.ones(1)))]
    if ok_botar is not None:
        patches.append(mock.patch.object(comando, "ok_agora_botar", return_value=torch.tensor([ok_botar])))
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
        # o fecho do P3 deixa o índice no P3 (o CARREGAR segue com ele) e marca `v3_pega_fechada`
        obj, env = _cadeia(P3, [z] * N)
        m = _medidas_p3(subida)
        for _ in range(passos):
            _alvos(obj, m)
        return env.v3_pega_fechada.item()

    def test_com_z_1_o_p3_fecha_com_1_cm_por_0_2_s_e_nao_fecha_com_0_5_cm(self):
        self.assertTrue(self._p3(1.0, 0.0105, 11))
        self.assertFalse(self._p3(1.0, 0.005, 40))

    def test_com_z_0_o_p3_e_o_final_3_cm_por_0_5_s(self):
        self.assertFalse(self._p3(0.0, 0.0105, 40))  # 1 cm não basta
        self.assertFalse(self._p3(0.0, 0.035, 20))  # 3 cm, mas ainda sem os 0,5 s
        self.assertTrue(self._p3(0.0, 0.035, 26))

    def test_os_tempos_acompanham_o_z_e_so_mudam_na_virada_da_iteracao(self):
        obj, env = _cadeia(P3, [1.0] * N)
        m = _medidas_p3(0.0)
        _alvos(obj, m)
        self.assertEqual([round(t, 6) for t in obj.tempos.tolist()], [0, 0, 0.2, 0.2, 0.1, 0.2, 0.2])
        self.assertAlmostEqual(env.v3_alvos.tol_transporte_p3, 0.20)
        env.v3_cur.z = [0.5] * N
        _alvos(obj, m)  # a mesma iteração: nada muda
        self.assertAlmostEqual(obj.tempos[P3].item(), 0.2)
        env.v3_cur.it += 1  # a iteração virou
        _alvos(obj, m)
        self.assertAlmostEqual(obj.tempos[P3].item(), 0.35)
        self.assertAlmostEqual(env.v3_alvos.tol_transporte_p3, 0.125)
        env.v3_cur.z = [1.0] * N
        _alvos(obj, m)  # a iteração 1 já foi lida: o z novo espera a iteração 2
        self.assertAlmostEqual(obj.tempos[P3].item(), 0.35)
        env.v3_cur.z = [0.0] * N
        env.v3_cur.it += 1
        _alvos(obj, m)
        self.assertEqual([round(t, 6) for t in obj.tempos.tolist()], [0, 0, 0.3, 0.5, 0.3, 0.5, 0.5])
        for _, campo, _ in A.zona:  # com z = 0 cada critério volta ao final
            if campo != "tol_palma":
                self.assertAlmostEqual(getattr(env.v3_alvos, campo), getattr(A, campo), places=9, msg=campo)
        for v in env.v3_alvos.tol_palma.tolist():
            self.assertAlmostEqual(v, A.tol_palma, places=6)

    def test_p0_e_p1_usam_o_z_de_cada_uma(self):
        # as duas palmas a 4 cm da pose: entre o final (3 cm) e o largo (5 cm). P0 com z = 1 fecha; P1 com z = 0 não
        m = ce._medidas()
        m["palmas_c"] = torch.tensor([[[0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]])
        for idx, z, fecha_ in ((P0, [1.0, 0.0] + [0.0] * 5, True), (P1, [1.0, 0.0] + [0.0] * 5, False),
                               (P0, [0.0, 1.0] + [0.0] * 5, False), (P1, [0.0, 1.0] + [0.0] * 5, True)):
            with self.subTest(idx=idx, z=z[:2]):
                obj, env = _cadeia(idx, z)
                with mock.patch.object(alvos, "palmas_alvo_caixa", return_value=torch.zeros(1, 2, 3)):
                    _alvos(obj, m)
                self.assertEqual(env.v3_alvo_idx.item(), idx + 1 if fecha_ else idx)

    def test_o_fecho_do_botar_usa_o_alvos_efetivo(self):
        # o núcleo real do BOTAR, sem mock: cada medida fica entre o final e o largo, e só o z = 1 fecha. Com o `Alvos`
        # final no lugar do efetivo (`env.v3_alvos`), o z = 1 não fecharia. O apoio do B1 fica em 100 % do peso: o peso
        # não é da zona (`test_comando_f`)
        m = ce._medidas()
        m["caixa_p"] = torch.tensor([[0.15, 0.0, 0.0]])  # B1: 15 cm do ponto (final 10, largo 20)
        m["apoio_N"] = torch.tensor([1.0])
        casos = [(B1, m, 6)]
        m2 = ce._medidas()
        m2["palmas_c"] = torch.full((1, 2, 3), 0.1 + A.palma_face + 0.03)  # B2: palmas a 3 cm da face (final 5, largo 2)
        casos.append((B2, m2, 15))
        for idx, medidas, passos in casos:
            for z, fecha_ in ((1.0, True), (0.0, False)):
                with self.subTest(idx=idx, z=z):
                    obj, env = _cadeia(idx, [z] * N, tarefa=comando.BOTAR)
                    for _ in range(passos):
                        _alvos(obj, medidas, ok_botar=None)
                    self.assertEqual(env.v3_alvo_idx.item() == idx + 1, fecha_)

    def test_o_phi_nao_muda_com_o_z(self):
        # o Φ lê o `Alvos` final (`K`), não o efetivo: com `env.v3_alvos` largo (z = 1) ele fica igual ao do final. P3: a razão
        # tol_transporte/tol_pelve da distância; B1: o `apoio_b1_mg` do déficit (65 % do peso no apoio, entre o final e o largo)
        class Cena(dict):
            env_origins = torch.zeros(1, 3)

        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        robo = types.SimpleNamespace(
            data=types.SimpleNamespace(root_link_pos_w=torch.zeros(1, 3), root_link_quat_w=quat,
                                       site_pos_w=torch.zeros(1, 2, 3)),
            find_sites=lambda nomes: ([0, 1], None))
        caixa = types.SimpleNamespace(data=types.SimpleNamespace(  # a caixa a 1 m: fora da altura do pouso
            root_link_pos_w=torch.tensor([[0.0, 0.0, 1.0]]), root_link_quat_w=quat, root_link_lin_vel_w=torch.zeros(1, 3)))
        faces = types.SimpleNamespace(data=types.SimpleNamespace(
            normal=torch.zeros(1, 12, 3), force=torch.zeros(1, 12, 3), found=torch.zeros(1, 12)))
        peso = 1.0 * 9.81
        cena = Cena(robot=robo, box=caixa, caixa_faces=faces,
                    altura_pelve=types.SimpleNamespace(data=types.SimpleNamespace(heights=torch.full((1, 1), 0.05))),
                    apoio_caixa=types.SimpleNamespace(data=types.SimpleNamespace(force=torch.tensor([[[0.0, 0.0, 0.65 * peso]]]))))
        largo = replace(A, tol_transporte_p3=0.20, tol_pelve_p3=0.10, apoio_b1_mg=0.5)
        ones = dict.fromkeys(("desloc", "incl", "maos", "tronco", "recuo", "com"), torch.ones(1))
        for idx in (P3, B1):
            phis = []
            for k in (A, largo):
                env = types.SimpleNamespace(
                    device="cpu", scene=cena, v3_alvos=k, v3_meia=torch.tensor([0.1]), v3_topo_destino=torch.zeros(1),
                    v3_degrau=torch.zeros(1, dtype=torch.long), v3_caixa_pega_w=torch.zeros(1, 3),
                    v3_perdeu=torch.zeros(1, dtype=torch.bool), v3_ponto_apoio=torch.zeros(1, 3),
                    v3_alvo_idx=torch.tensor([idx]), v3_sigma=torch.tensor([0.3]),
                    v3_ref=types.SimpleNamespace(palmas_de_pe_b=torch.zeros(2, 3)), v3_massa=torch.ones(1),
                    sim=types.SimpleNamespace(mj_model=types.SimpleNamespace(opt=types.SimpleNamespace(gravity=(0.0, 0.0, -9.81)))))
                with mock.patch.object(recompensas, "ref_do_alvo", return_value=torch.zeros(1, 8)), \
                        mock.patch.object(recompensas, "incl_torso", return_value=torch.zeros(1)), \
                        mock.patch.object(recompensas, "_pes_e_com",
                                          return_value=(torch.zeros(1, 8, 2), torch.zeros(1, 2), torch.ones(1, dtype=torch.bool))), \
                        mock.patch.object(recompensas, "_recuo_ref", return_value=torch.zeros(1)), \
                        mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=torch.tensor([[0.0, 0.0, 1.0]])), \
                        mock.patch.object(recompensas, "portoes_do_alvo", return_value=ones):
                    phis.append(recompensas._phi_env(env, env.v3_alvo_idx, env.v3_sigma)[0].item())
            with self.subTest(idx=idx):
                self.assertGreater(phis[0], 0.5)
                self.assertEqual(phis[0], phis[1])

    def test_o_d_e_o_phi_ficam_no_valor_final(self):
        obj, env = _cadeia(P3, [1.0] * N)
        m = ce._medidas()
        _alvos(obj, m)
        self.assertIsNot(env.v3_alvos, A)
        self.assertAlmostEqual(env.v3_alvos.tol_transporte_p3, 0.20)
        with mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=torch.zeros(1, 3)), \
                mock.patch.object(recompensas, "distancia_do_alvo", return_value=torch.zeros(1)) as dist:
            obj._dist(m, torch.tensor([P3]))
        self.assertIs(dist.call_args.args[-1], A)  # o `d` da cadeia (o σ) lê o final

    def test_a_qualidade_do_carregar_muda_com_o_z_do_p3(self):
        # a caixa a 17 cm do ponto de transporte: com o P3 largo (20 cm) a qualidade fica perto de 0,49; com o final, 0
        obj, env = _cadeia(P3, [0.0] * 3 + [1.0] + [0.0] * 3)
        _alvos(obj, ce._medidas())
        e = rg.env_falso(-1, yaw=math.pi / 2, caixa_z=rg.Z_ALVO_DE_HOJE + 0.17)
        e.v3_alvos = env.v3_alvos
        self.assertAlmostEqual(recompensas._qualidade(e)[0].item(), math.exp(-((0.17 / 0.20) ** 2)), places=3)
        obj, env = _cadeia(P3, [0.0] * N)
        _alvos(obj, ce._medidas())
        e.v3_alvos = env.v3_alvos
        self.assertLess(recompensas._qualidade(e)[0].item(), 1e-3)

    def test_sigma_tem_piso_por_etapa_e_o_b3_fica_sem_piso(self):
        obj = comando.CadeiaCmd.__new__(comando.CadeiaCmd)
        obj.sigma_min = torch.tensor(A.sigma_min_etapa)
        obj._dist = lambda m, indice: torch.full((N,), 0.001)
        sigma = comando.CadeiaCmd._sigma(obj, {}, torch.arange(N))
        self.assertEqual([round(s, 6) for s in sigma.tolist()], [0.05, 0.05, 0.08, 0.20, 0.20, 0.05, 0.001])
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
        self.assertEqual(env.v3_cur.vis_acum, [0, 1] + [0] * 5)
        self.assertEqual(env.v3_cur.fech_acum, [0, 1] + [0] * 5)

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
        self.assertTrue(env.v3_pega_fechada.item())
        self.assertEqual(self._contas(env), (1, 0))

    def test_perda_da_caixa_conta_falha(self):
        obj, env = _cadeia(P3, [0.0] * N)
        m = _medidas_p3(0.0)
        _alvos(obj, m, ok_pegar=False)
        env.v3_perdeu[:] = True  # o `_perda` marca a perda e põe o alvo no B3
        env.v3_alvo_idx[:] = B3
        _alvos(obj, m, ok_pegar=False)
        self.assertEqual(self._contas(env), (1, 0))
        _alvos(obj, m, dt=10.0, ok_pegar=False)  # 10 s com a caixa perdida: o B3 dela não é visita e não vence
        self.assertEqual(self._contas(env), (1, 0))
        self.assertEqual(env.v3_cur.vis_acum[B3], 0)

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
        self.assertEqual(env.v3_cur.vis_acum, [0] * N)  # a visita está aberta
        obj._resample(torch.tensor([0]))
        um_p3 = [0, 0, 0, 1, 0, 0, 0]
        self.assertEqual((env.v3_cur.vis_acum, env.v3_cur.fech_acum), (um_p3, [0] * N))
        obj._resample(torch.tensor([0]))  # um segundo reset sem visita aberta: sem contagem em nenhuma etapa
        self.assertEqual((env.v3_cur.vis_acum, env.v3_cur.fech_acum), (um_p3, [0] * N))
        self.assertEqual(obj.vis_e.tolist(), [-1])

    def test_reset_depois_da_visita_vencida_nao_conta_de_novo(self):
        obj = ce._cadeia_de_trechos()
        env = obj._env
        env.v3_tarefa[:] = comando.PEGAR
        env.v3_alvo_idx[:] = P3
        _alvos(obj, _medidas_p3(0.0), ok_pegar=False)
        _alvos(obj, _medidas_p3(0.0), dt=10.0, ok_pegar=False)  # a visita passa de T_visita: vence e conta falha
        self.assertEqual(env.v3_cur.vis_acum, [0, 0, 0, 1, 0, 0, 0])
        obj._resample(torch.tensor([0]))  # o `vis_e` ainda guarda o P3, mas `t_vis` < 0: a visita já foi contada
        self.assertEqual((env.v3_cur.vis_acum, env.v3_cur.fech_acum), ([0, 0, 0, 1, 0, 0, 0], [0] * N))

    def test_pega_fechada_nao_e_visita_aberta(self):
        obj, env = _cadeia(P3, [0.0] * N)
        env.v3_pega_fechada[:] = True
        _alvos(obj, ce._medidas(), dt=10.0, ok_pegar=False)  # 10 s com a pega fechada: nada vence
        self.assertEqual(env.v3_cur.vis_acum, [0] * N)

    def test_bota_fica_visitada_com_a_pega_fechada_no_episodio(self):
        # o `v3_pega_fechada` do PEGAR fica True no episódio inteiro; o B1 do mesmo episódio é visita
        obj, env = _cadeia(B1, [0.0] * N, tarefa=comando.BOTAR)
        env.v3_pega_fechada[:] = True
        _alvos(obj, ce._medidas(), ok_pegar=False)
        self.assertEqual(obj.vis_e.tolist(), [B1])


class TestSucessoSoComOCriterioFinal(unittest.TestCase):

    def _registra(self, z_pega, z_bota, botar):
        obj = ce._cadeia_de_trechos()
        env = obj._env
        env.v3_cur.z = list(z_pega) + list(z_bota)
        env.v3_pega_fechada[:] = True
        env.v3_bota_fechada[:] = True
        obj.pega[:] = obj.bota[:] = True
        obj._registra(torch.tensor([0]), torch.tensor([False]), botar=botar)
        return env.v3_cur.registros

    def test_pegar_com_z_acima_de_0_registra_falha(self):
        self.assertEqual(self._registra([0, 0, 0, 0.3], [0] * 3, False), [(1, 0)])
        self.assertEqual(self._registra([0.3, 0, 0, 0], [0] * 3, False), [(1, 0)])
        self.assertEqual(self._registra([0] * 4, [0] * 3, False), [(1, 1)])

    def test_botar_so_olha_o_z_do_botar(self):
        self.assertEqual(self._registra([0] * 4, [0, 0.2, 0], True), [(1, 0)])
        self.assertEqual(self._registra([0.5] * 4, [0] * 3, True), [(1, 1)])  # o PEGAR aberto não trava o BOTAR
        self.assertEqual(self._registra([0] * 4, [0.5] * 3, False), [(1, 1)])  # nem o BOTAR trava o PEGAR


class TestObservacaoEPublicacao(unittest.TestCase):

    def test_critico_ve_o_z_da_etapa_e_zero_fora_de_etapa(self):
        env = types.SimpleNamespace(
            v3_fase_marcha=torch.zeros(3), v3_marcha_on=torch.zeros(3, dtype=torch.bool), v3_h=torch.ones(3),
            v3_phi_melhor=torch.zeros(3), v3_phi=torch.zeros(3), v3_sigma=torch.full((3,), 7.0),
            v3_alvo_idx=torch.tensor([P0, B3, -1]), v3_cur=types.SimpleNamespace(z=[0.1 * (i + 1) for i in range(N)]))
        self.assertTrue(torch.allclose(observacoes.privilegiados(env)[:, 3], torch.tensor([0.1, 0.7, 0.0])))

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
