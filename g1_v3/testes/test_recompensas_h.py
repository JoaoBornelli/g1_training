"""Testes da sacudida, do aperto e do degrau da tarefa (H2 do plano; §6.3 "Fase H: núcleos puros (30/10)", §6.5,
§12.4 H2; S §8.2 e §11.5; P1, P3, P4, P8 a P12, P20 da Parte 4).

Núcleos com tensores sintéticos; invólucros com env falso (`types.SimpleNamespace`), como os vizinhos.
Convenções: E = 1 salvo onde a conta pede outro; `DT` = 0,02 s (`env.step_dt`).
"""
import types
import unittest

import torch

from g1_v3 import recompensas
from g1_v3.knobs import ALVOS, ATIVO, TAREFAS

DT = 0.02
NORMA_LIN, NORMA_ANG = ATIVO.recompensa.sacudida_norma  # (5,04; 31,56): o p50 medido no 23250
P4, P0, P1, P2, B3, B4 = (ALVOS.index(n) for n in ("P4", "P0", "P1", "P2", "B3", "B4"))
CARGA = ("PEGAR", "CARREGAR", "BOTAR")  # tarefas da sacudida e do aperto (§6.5)
S = 12  # fatias do sensor `caixa_faces` (cena.py, num_slots=12)
IDENTIDADE = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # quat wxyz
P50_DV = 5.04 * DT  # |Δv| do p50 medido: m/s entre duas chamadas
P50_DW = 31.56 * DT  # |Δω| do p50 medido: rad/s entre duas chamadas
R90 = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])  # caixa girada 90° em yaw


def env_falso(tarefa="CARREGAR", alvo=P4, degrau=5, faces=None, e=1):
    """Env de `e` envs: caixa com velocidade zero e quat identidade; robô com quat identidade."""
    vel = torch.zeros(e, 6)
    box = types.SimpleNamespace(data=types.SimpleNamespace(root_link_vel_w=vel, root_link_quat_w=IDENTIDADE.repeat(e, 1)))
    robo = types.SimpleNamespace(data=types.SimpleNamespace(root_link_quat_w=IDENTIDADE.repeat(e, 1)))
    cena = {"box": box, "robot": robo}
    if faces is not None:
        cena["caixa_faces"] = faces
    return types.SimpleNamespace(
        scene=cena, device="cpu", step_dt=DT, num_envs=e,
        v3_tarefa=torch.full((e,), TAREFAS.index(tarefa), dtype=torch.long),
        v3_alvo_idx=torch.full((e,), alvo, dtype=torch.long),
        v3_degrau=torch.full((e,), degrau, dtype=torch.long))


def vel_caixa(lin=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0)):
    """(1,6) de `root_link_vel_w`: [lin xyz, ang xyz] (mjlab, `entity/data.py`)."""
    return torch.tensor([[*lin, *ang]])


def sacudida_p50(degrau, alvo=P4, tarefa="CARREGAR"):
    """Duas chamadas do termo: a 1ª com v = 0; a 2ª com o Δ do p50 medido (|Δv| = 5,04·dt, |Δω| = 31,56·dt)."""
    env = env_falso(tarefa=tarefa, alvo=alvo, degrau=degrau)
    termo = recompensas.sacudida(cfg=None, env=env)
    termo(env, CARGA)
    env.scene["box"].data.root_link_vel_w = vel_caixa(lin=(P50_DV, 0.0, 0.0), ang=(0.0, P50_DW, 0.0))
    return termo(env, CARGA).item()


def fatias(contatos, ruido=()):
    """(normal, força, achado) do sensor com 12 fatias: `contatos` = [(normal, força)] nas primeiras fatias com
    achado; `ruido` = índices de fatias SEM achado com NaN na normal e na força."""
    normal, forca = torch.zeros(1, S, 3), torch.zeros(1, S, 3)
    achado = torch.zeros(1, S, dtype=torch.bool)
    for i, (n, f) in enumerate(contatos):
        normal[0, i], forca[0, i], achado[0, i] = torch.tensor(n), torch.tensor(f), True
    for i in ruido:
        normal[0, i], forca[0, i] = float("nan"), float("nan")
    return normal, forca, achado


def aperto_de(contatos, degrau=5, alvo=P4, tarefa="CARREGAR", ruido=()):
    """Valor do `aperto` com as faces de `contatos` no sensor `caixa_faces` (robô e caixa com quat identidade)."""
    normal, forca, achado = fatias(contatos, ruido)
    faces = types.SimpleNamespace(data=types.SimpleNamespace(normal=normal, force=forca, found=achado.long()))
    env = env_falso(tarefa=tarefa, alvo=alvo, degrau=degrau, faces=faces)
    return recompensas.aperto(env, CARGA)


class NSacudida(unittest.TestCase):
    def test_delta_do_p50_vale_2(self):
        # Teste (H2): ‖Δv‖ = 5,04·dt e ‖Δω‖ = 31,56·dt → (1)² + (1)² = 2,0 (o p50 medido no 23250).
        v = recompensas.n_sacudida(torch.tensor([[P50_DV, 0.0, 0.0]]), torch.tensor([[0.0, P50_DW, 0.0]]),
                                   DT, NORMA_LIN, NORMA_ANG)
        self.assertAlmostEqual(v.item(), 2.0, delta=1e-3)

    def test_p50_da_spec_vale_0_621(self):
        # Teste (H2): a spec (2,9 m/s²; 17 rad/s²) → (2,9/5,04)² + (17/31,56)² = 0,621.
        v = recompensas.n_sacudida(torch.tensor([[2.9 * DT, 0.0, 0.0]]), torch.tensor([[17.0 * DT, 0.0, 0.0]]),
                                   DT, NORMA_LIN, NORMA_ANG)
        self.assertAlmostEqual(v.item(), 0.621, delta=1e-3)

    def test_delta_zero_vale_zero(self):
        # Teste (H2): sem variação entre as chamadas, a sacudida é zero.
        z = torch.zeros(1, 3)
        self.assertEqual(recompensas.n_sacudida(z, z, DT, NORMA_LIN, NORMA_ANG).item(), 0.0)

    def test_direcao_de_dv_nao_muda_o_valor(self):
        # Teste (H2): |Δv| igual em outra direção dá o mesmo valor (a conta é a soma dos quadrados).
        dw = torch.zeros(1, 3)
        a = recompensas.n_sacudida(torch.tensor([[P50_DV, 0.0, 0.0]]), dw, DT, NORMA_LIN, NORMA_ANG)
        b = recompensas.n_sacudida(torch.tensor([[0.0, 0.0, P50_DV]]), dw, DT, NORMA_LIN, NORMA_ANG)
        self.assertAlmostEqual(a.item(), b.item(), delta=1e-6)


class Sacudida(unittest.TestCase):
    def test_primeira_chamada_com_v_zero_vale_zero(self):
        # Teste (H2): a v_ant começa em zero, e a 1ª chamada com v = 0 dá Δ = 0.
        env = env_falso()
        termo = recompensas.sacudida(cfg=None, env=env)
        self.assertEqual(termo(env, CARGA).item(), 0.0)

    def test_p50_vale_2_no_degrau_5_1_2_no_3_e_0_no_0(self):
        # Teste (H2): o fator d/5 do degrau da tarefa mora no termo: 2,0 · d/5.
        self.assertAlmostEqual(sacudida_p50(5), 2.0, delta=1e-3)
        self.assertAlmostEqual(sacudida_p50(3), 1.2, delta=1e-3)
        self.assertEqual(sacudida_p50(0), 0.0)

    def test_zero_com_alvo_que_nao_pesa_e_fora_das_tarefas(self):
        # Teste (H2): só P3, P4, B1 e B2 pagam (§6.5, P3/P4/B1/B2); alvo −1, P0–P2, B3, B4 e ANDAR dão zero.
        for alvo in (-1, P0, P1, P2, B3, B4):
            with self.subTest(alvo=alvo):
                self.assertEqual(sacudida_p50(5, alvo=alvo), 0.0)
        self.assertEqual(sacudida_p50(5, tarefa="ANDAR"), 0.0)

    def test_custo_atualiza_v_ant_mesmo_mascarado(self):
        # Teste (H2): com o alvo P0 (mascarado), a v_ant já guarda a velocidade da chamada.
        env = env_falso(alvo=P0)
        termo = recompensas.sacudida(cfg=None, env=env)
        v2 = vel_caixa(lin=(1.0, 2.0, 3.0), ang=(0.1, 0.2, 0.3))
        env.scene["box"].data.root_link_vel_w = v2
        termo(env, CARGA)
        self.assertTrue(torch.equal(termo.v_ant, v2))

    def test_reset_zera_so_o_env_pedido(self):
        # Teste (H2): reset([0]) zera a v_ant do env 0 e deixa a do env 1.
        env = env_falso(e=2)
        termo = recompensas.sacudida(cfg=None, env=env)
        env.scene["box"].data.root_link_vel_w = torch.tensor([[1.0, 0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0, 0]])
        termo(env, CARGA)
        termo.reset(env_ids=torch.tensor([0]))
        self.assertTrue(torch.equal(termo.v_ant[0], torch.zeros(6)))
        self.assertEqual(termo.v_ant[1, 1].item(), 2.0)


class ForcasPorFace(unittest.TestCase):
    def test_117_5_nas_faces_mais_y_e_menos_y(self):
        # Teste (H2): 117,5 N na normal +y e 117,5 N na −y → (117,5; 117,5), cada uma na sua palma.
        normal, forca, achado = fatias([((0, 1, 0), (0, 117.5, 0)), ((0, -1, 0), (0, -117.5, 0))])
        esq, dir_ = recompensas.forcas_por_face(normal, forca, achado, torch.eye(3)[None])
        self.assertAlmostEqual(esq.item(), 117.5, delta=1e-4)
        self.assertAlmostEqual(dir_.item(), 117.5, delta=1e-4)

    def test_contato_na_frente_e_em_cima_nao_entra(self):
        # Teste (H2): 200 N na face −x (frente) e 200 N na +z não mudam F_esq nem F_dir (S §8.2).
        normal, forca, achado = fatias([((0, 1, 0), (0, 117.5, 0)), ((0, -1, 0), (0, -117.5, 0)),
                                        ((-1, 0, 0), (-200, 0, 0)), ((0, 0, 1), (0, 0, 200))])
        esq, dir_ = recompensas.forcas_por_face(normal, forca, achado, torch.eye(3)[None])
        self.assertAlmostEqual(esq.item(), 117.5, delta=1e-4)
        self.assertAlmostEqual(dir_.item(), 117.5, delta=1e-4)

    def test_fatia_sem_achado_com_nan_nao_entra(self):
        # Teste (H2): fatia sem achado com força e normal NaN não entra, e a saída é finita (torch.where, §6.6).
        normal, forca, achado = fatias([((0, 1, 0), (0, 117.5, 0)), ((0, -1, 0), (0, -117.5, 0))], ruido=(7,))
        esq, dir_ = recompensas.forcas_por_face(normal, forca, achado, torch.eye(3)[None])
        self.assertTrue(torch.isfinite(esq).all() and torch.isfinite(dir_).all())
        self.assertAlmostEqual(esq.item(), 117.5, delta=1e-4)
        self.assertAlmostEqual(dir_.item(), 117.5, delta=1e-4)

    def test_caixa_girada_90_graus_em_yaw_da_o_mesmo(self):
        # Teste (H2): as normais no mundo giradas pelo mesmo R dão o mesmo resultado (n_c = Rᵀ·normal_w).
        normal_c, forca_c, achado = fatias([((0, 1, 0), (0, 117.5, 0)), ((0, -1, 0), (0, -117.5, 0))])
        normal_w = torch.einsum("eij,esj->esi", R90, normal_c)
        forca_w = torch.einsum("eij,esj->esi", R90, forca_c)
        esq, dir_ = recompensas.forcas_por_face(normal_w, forca_w, achado, R90)
        self.assertAlmostEqual(esq.item(), 117.5, delta=1e-4)
        self.assertAlmostEqual(dir_.item(), 117.5, delta=1e-4)

    def test_forca_que_nao_e_normal_conta_so_a_componente(self):
        # Teste (H2): força (0; 100; 30) na normal +y → 100 (F_n = |⟨força, normal⟩|), só na palma esquerda.
        normal, forca, achado = fatias([((0, 1, 0), (0, 100, 30))])
        esq, dir_ = recompensas.forcas_por_face(normal, forca, achado, torch.eye(3)[None])
        self.assertAlmostEqual(esq.item(), 100.0, delta=1e-4)
        self.assertEqual(dir_.item(), 0.0)

    def test_dois_contatos_na_mais_y_somam(self):
        # Teste (H2): 60 N e 40 N na normal +y somam 100 N na palma esquerda.
        normal, forca, achado = fatias([((0, 1, 0), (0, 60, 0)), ((0, 1, 0), (0, 40, 0))])
        esq, _ = recompensas.forcas_por_face(normal, forca, achado, torch.eye(3)[None])
        self.assertAlmostEqual(esq.item(), 100.0, delta=1e-4)


class Aperto(unittest.TestCase):
    def test_117_5_nas_duas_faces_no_degrau_5_3_e_0(self):
        # Teste (H2): 117,5 N nas duas faces → 0,909 no degrau 5; 0,545 no 3; 0 no 0 (S §8.2, F_ref = 49 N).
        contatos = [((0, 1, 0), (0, 117.5, 0)), ((0, -1, 0), (0, -117.5, 0))]
        self.assertAlmostEqual(aperto_de(contatos, degrau=5).item(), 0.909, delta=1e-3)
        self.assertAlmostEqual(aperto_de(contatos, degrau=3).item(), 0.545, delta=1e-3)
        self.assertEqual(aperto_de(contatos, degrau=0).item(), 0.0)

    def test_49_nas_duas_faces_vale_0_632(self):
        # Teste (H2): 49 N nas duas faces = F_ref → 1 − e⁻¹ = 0,632.
        contatos = [((0, 1, 0), (0, 49, 0)), ((0, -1, 0), (0, -49, 0))]
        self.assertAlmostEqual(aperto_de(contatos, degrau=5).item(), 0.632, delta=1e-3)

    def test_uma_face_so_vale_zero(self):
        # Teste (H2): aperto com uma palma só não aperta a caixa: o mínimo das duas é zero.
        self.assertEqual(aperto_de([((0, 1, 0), (0, 117.5, 0))]).item(), 0.0)

    def test_alvo_que_nao_pesa_vale_zero(self):
        # Teste (H2): alvo −1 (sem alvo) não paga aperto.
        contatos = [((0, 1, 0), (0, 117.5, 0)), ((0, -1, 0), (0, -117.5, 0))]
        self.assertEqual(aperto_de(contatos, alvo=-1).item(), 0.0)

    def test_env_fora_da_tarefa_com_nan_vale_zero_e_finito(self):
        # Teste (H2): fora da tarefa (ANDAR), com NaN na força, o termo vale 0 e é finito (torch.where, §6.6).
        valor = aperto_de([((0, 1, 0), (0, float("nan"), 0)), ((0, -1, 0), (0, -117.5, 0))], tarefa="ANDAR")
        self.assertEqual(valor.item(), 0.0)
        self.assertTrue(torch.isfinite(valor).all())


class PorDegrau(unittest.TestCase):
    def test_por_degrau_com_degrau_por_env(self):
        # Teste (H2): (ini, fim) = (50, 10) com degrau [0, 3, 5] → [50, 26, 10]: a fórmula de `balanceador.no_degrau`.
        env = types.SimpleNamespace(v3_degrau=torch.tensor([0, 3, 5]))
        valor = recompensas.por_degrau(env, 50, 10)
        self.assertTrue(torch.allclose(valor, torch.tensor([50.0, 26.0, 10.0])))

    def test_tau_max_sai_do_modulo(self):
        # Teste (H2, P20): o `_TAU_MAX` do módulo sai; a `cerca_torque` lê `knobs.TAU_MAX`.
        self.assertFalse(hasattr(recompensas, "_TAU_MAX"))


if __name__ == "__main__":
    unittest.main()
