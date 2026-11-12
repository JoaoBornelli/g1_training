"""Testes de g1_v3/metricas.py da lane H5b da fase H (§6.7, §12.4 H5b; P13, P23; S §14).

`aceleracao_da_caixa` (custo da sacudida nos passos presos, por episódio) e `fracao_do_curso` lendo a
`potencial.curso`. Env falso com `types.SimpleNamespace`, como o `test_metricas_g.py`; `reset` antes do `__call__`
seguindo a ordem do mjlab (passos → `reset` → `__call__`).
"""
import types
import unittest
from unittest import mock

import torch

from g1_v3 import alvos, knobs, metricas, potencial, recompensas

DT = 0.02
P4, P0 = alvos.P4, alvos.P0
# Δ do p50 medido (knobs: `sacudida_norma` = (5,04; 31,56) m/s² e rad/s² × dt): ‖Δv‖ = 5,04·0,02, ‖Δω‖ = 31,56·0,02
DELTA = torch.tensor([5.04 * DT, 0.0, 0.0, 31.56 * DT, 0.0, 0.0])
CHAVE = "Episodio/aceleracao_da_caixa"


def env_falso(alvo):
    # `alvo`: lista (E,) de índices de `alvos`; a velocidade da caixa (E, 6) é trocada a cada passo do teste
    n = len(alvo)
    caixa = types.SimpleNamespace(data=types.SimpleNamespace(root_link_vel_w=torch.zeros(n, 6)))
    return types.SimpleNamespace(num_envs=n, device="cpu", step_dt=DT, scene={"box": caixa},
                                 extras={"log": {}}, v3_alvo_idx=torch.tensor(alvo))


class AceleracaoDaCaixaTest(unittest.TestCase):
    def test_caixa_presa_loga_o_custo_da_sacudida_do_p50(self):
        env = env_falso([P4, P0])  # env 0 preso (P4); env 1 não entra
        m = metricas.aceleracao_da_caixa(None, env)
        for k in (1, 2, 3):  # velocidades k·Δ: cada passo muda a velocidade por Δ, e o custo do passo é 2 (p50)
            env.scene["box"].data.root_link_vel_w = k * DELTA.expand(2, 6)
            flag = m(env)
        self.assertEqual(flag.tolist(), [1.0, 0.0])  # o `__call__` devolve a flag "teve passo preso"
        m.reset()
        log = env.extras["log"][CHAVE]
        self.assertEqual(log.numel(), 1)  # só o env 0 fecha com evento
        self.assertAlmostEqual(log.item(), 2.0, delta=1e-4)

    def test_sem_passo_preso_a_chave_nao_sai(self):
        env = env_falso([P0, P0])
        m = metricas.aceleracao_da_caixa(None, env)
        for k in (1, 2):
            env.scene["box"].data.root_link_vel_w = k * DELTA.expand(2, 6)
            m(env)
        m.reset()
        self.assertNotIn(CHAVE, env.extras["log"])

    def test_mascara_e_a_do_recompensas_caixa_alvo(self):
        # R3: a máscara "caixa presa" é a do termo (`recompensas._CAIXA_ALVO`), e não um conjunto escrito de novo.
        # Com o grupo trocado para só (P3,), o passo com o alvo em P4 não entra (falha antes do código: P4 é fixo).
        env = env_falso([P4, alvos.P3])
        with mock.patch.object(recompensas, "_CAIXA_ALVO", (alvos.P3,)):
            m = metricas.aceleracao_da_caixa(None, env)
            for k in (1, 2):
                env.scene["box"].data.root_link_vel_w = k * DELTA.expand(2, 6)
                flag = m(env)
        self.assertEqual(flag.tolist(), [0.0, 1.0])

    def test_termo_registrado_com_reduce_last(self):
        cfg = metricas.termos(knobs.ATIVO)["aceleracao_da_caixa"]
        self.assertIs(cfg.func, metricas.aceleracao_da_caixa)
        self.assertEqual(cfg.reduce, "last")


def env_do_p4_e_p0():
    # env 0 no P4: caixa 0,20 m abaixo do alvo, 0,10 m ao lado (yaw 0), pelve 0,04 m fora da referência.
    # env 1 no P0, com a caixa longe: se entrasse no log, o valor mudaria.
    robo = types.SimpleNamespace(data=types.SimpleNamespace(root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)))
    caixa = types.SimpleNamespace(data=types.SimpleNamespace(
        root_link_pos_w=torch.tensor([[1.0, 2.10, 0.30], [5.0, 5.0, 5.0]])))
    altura = types.SimpleNamespace(data=types.SimpleNamespace(heights=torch.tensor([[0.80], [0.30]])))
    ref = types.SimpleNamespace(pelve_com_caixa=lambda a: torch.tensor([0.76, 0.25]))
    return types.SimpleNamespace(num_envs=2, device="cpu", step_dt=DT, extras={"log": {}},
                                 scene={"robot": robo, "box": caixa, "altura_pelve": altura},
                                 v3_alvo_idx=torch.tensor([P4, P0]),
                                 v3_tarefa=torch.tensor([metricas.CARREGAR] * 2),
                                 v3_ref=ref, v3_meia=torch.tensor([0.5, 0.5]))


class ErroP4Test(unittest.TestCase):
    def test_erros_do_p4_so_com_o_env_do_p4(self):
        # alvo da caixa fixo em (1,0; 2,0; 0,5) para os dois envs: e = caixa − alvo
        # vertical = −e_z = −(0,30 − 0,50) = 0,20; lateral = |e_y| = |2,10 − 2,00| = 0,10; pelve = |0,80 − 0,76| = 0,04
        env = env_do_p4_e_p0()
        alvo = torch.tensor([[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]])
        m = metricas.erro_p4(None, env)
        with mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=alvo):
            flag = m(env)
        self.assertEqual(flag.tolist(), [1.0, 0.0])  # o `__call__` devolve a flag "teve o P4"
        m.reset()
        log = env.extras["log"]
        for nome, esperado in (("vertical", 0.20), ("lateral", 0.10), ("pelve", 0.04)):
            self.assertAlmostEqual(log["Episodio/erro_p4_" + nome].item(), esperado, delta=1e-5)


    def test_erro_lateral_com_yaw_de_90_graus(self):
        # yaw de 90° (wxyz (0,7071; 0; 0; 0,7071)): sen = 1, cos = 0, então lateral = |e_x|. Caixa 0,10 m à frente
        # do alvo em x e 0 em y, 0,20 m abaixo: lateral 0,10 (com yaw 0 seria 0, pois e_y = 0)
        env = env_do_p4_e_p0()
        env.scene["robot"].data.root_link_quat_w = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]] * 2)
        env.scene["box"].data.root_link_pos_w = torch.tensor([[1.10, 2.00, 0.30], [5.0, 5.0, 5.0]])
        alvo = torch.tensor([[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]])
        m = metricas.erro_p4(None, env)
        with mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=alvo):
            m(env)
        m.reset()
        self.assertAlmostEqual(env.extras["log"]["Episodio/erro_p4_lateral"].item(), 0.10, delta=1e-5)

    def test_media_de_dois_passos_no_p4(self):
        # passo 1: caixa 0,20 abaixo do alvo (vertical 0,20); passo 2: 0,40 abaixo (vertical 0,40); média 0,30
        env = env_do_p4_e_p0()
        alvo = torch.tensor([[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]])
        m = metricas.erro_p4(None, env)
        with mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=alvo):
            m(env)
            env.scene["box"].data.root_link_pos_w = torch.tensor([[1.0, 2.10, 0.10], [5.0, 5.0, 5.0]])
            m(env)
        m.reset()
        self.assertAlmostEqual(env.extras["log"]["Episodio/erro_p4_vertical"].item(), 0.30, delta=1e-5)

    def test_reset_so_do_env_0_nao_zera_a_soma_do_env_1(self):
        # os dois envs no P4 e com caixa (CARREGAR); o env 1 tem erro grande, então a soma dele é visível
        env = env_do_p4_e_p0()
        env.v3_alvo_idx = torch.tensor([P4, P4])
        alvo = torch.tensor([[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]])
        m = metricas.erro_p4(None, env)
        with mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=alvo):
            m(env)
        m.reset(torch.tensor([0]))
        self.assertEqual(m.n.tolist(), [0.0, 1.0])
        self.assertGreater(m.soma[1].abs().sum().item(), 0.0)


class FracaoDoCursoTest(unittest.TestCase):
    def test_maior_fracao_do_curso_com_a_conta_do_potencial(self):
        # junta 0: |0,5 − 0|/1 = 0,5; junta 1: |2,5 − 1|/1 = 1,5 → o pico é 1,5 (o mesmo de antes)
        robo = types.SimpleNamespace(data=types.SimpleNamespace(
            joint_pos_limits=torch.tensor([[[-1.0, 1.0], [0.0, 2.0]]]), joint_pos=torch.tensor([[0.5, 2.5]])))
        env = types.SimpleNamespace(scene={"robot": robo})
        asset = types.SimpleNamespace(name="robot", joint_ids=[0, 1])
        self.assertAlmostEqual(metricas.fracao_do_curso(env, asset).item(), 1.5, places=6)

    def test_usa_a_potencial_curso(self):
        # P13: uma conta só para o curso; a métrica chama `potencial.curso` (esta assinatura falha antes do código)
        robo = types.SimpleNamespace(data=types.SimpleNamespace(
            joint_pos_limits=torch.tensor([[[-1.0, 1.0]]]), joint_pos=torch.tensor([[0.5]])))
        env = types.SimpleNamespace(scene={"robot": robo})
        asset = types.SimpleNamespace(name="robot", joint_ids=[0])
        with mock.patch.object(metricas, "curso", wraps=potencial.curso) as curso:
            metricas.fracao_do_curso(env, asset)
        curso.assert_called_once()


if __name__ == "__main__":
    unittest.main()
