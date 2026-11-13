"""Testes de g1_v3/algoritmo.py, núcleo normaliza_por_grupo (D9 do plano, §11; g1_limpo/algoritmo.py:155-224).
Escritos antes do código.

Convenções: `grupo` é o índice de TAREFAS (§6.2; ANDAR = 0), o argmax das 6 últimas colunas do crítico
(D9). `adv` e `grupo` são (N,) e a saída é (N,). O desvio é o `std()` do torch, COM correção de Bessel
(n−1), o padrão do torch, como no g1_limpo (`a.std()`); por isso a saída tem `std()` = 1.

Grupo pequeno (< 2 amostras), como no g1_limpo: o ANDAR fica com a vantagem CRUA; os outros grupos
são normalizados pela "manipulação inteira" (todas as amostras não-ANDAR do lote, o próprio grupo
ralo incluído), que é o fallback que evita o NaN do std de uma amostra. Os testes repetem essa conta.
"""
import types
import unittest
import unittest.mock

import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules.normalization import EmpiricalNormalization

from g1_v3.algoritmo import PISO_DESVIO, PPOPorTarefa, fixa_normalizador, normaliza_por_grupo

ANDAR, PEGAR, CARREGAR = 0, 1, 2


class TestNormalizaPorGrupo(unittest.TestCase):
    def test_tres_grupos_media_zero_e_desvio_um(self):
        # três grupos com escalas e médias diferentes: cada um sai com média 0 e desvio 1
        g = torch.Generator().manual_seed(0)
        blocos = [(ANDAR, 40, 0.0, 0.5), (PEGAR, 30, 3.0, 2.0), (CARREGAR, 20, -1.0, 10.0)]
        adv = torch.cat([torch.randn(n, generator=g) * esc + loc for _, n, loc, esc in blocos])
        grupo = torch.cat([torch.full((n,), t, dtype=torch.long) for t, n, _, _ in blocos])
        out = normaliza_por_grupo(adv, grupo)
        self.assertEqual(out.shape, adv.shape)
        for t, _, _, _ in blocos:
            m = out[grupo == t]
            self.assertAlmostEqual(float(m.mean()), 0.0, places=4)
            self.assertAlmostEqual(float(m.std()), 1.0, places=4)

    def test_grupo_de_uma_amostra_nao_andar_usa_a_manipulacao_inteira(self):
        # CARREGAR com uma amostra só (7,0). O fallback é a média e o desvio de todas as não-ANDAR
        # do lote (2,0; 4,0; 7,0), e o valor sai normalizado por eles, sem NaN (g1_limpo:`n < 2`).
        adv = torch.tensor([0.5, -0.5, 1.0, 2.0, 4.0, 7.0])
        grupo = torch.tensor([ANDAR, ANDAR, ANDAR, PEGAR, PEGAR, CARREGAR])
        out = normaliza_por_grupo(adv, grupo)
        self.assertTrue(torch.isfinite(out).all())
        manip = adv[grupo != ANDAR]
        esperado = (7.0 - manip.mean()) / (manip.std() + 1e-8)
        self.assertAlmostEqual(float(out[5]), float(esperado), places=5)
        # o PEGAR, com duas amostras, segue a regra do grupo normal (média 0)
        self.assertAlmostEqual(float(out[3] + out[4]), 0.0, places=5)

    def test_grupo_de_uma_amostra_andar_fica_com_a_vantagem_crua(self):
        # o ANDAR é o único grupo que não usa o fallback: a amostra sozinha sai como veio (0,8)
        adv = torch.tensor([0.8, 1.0, 2.0, 3.0])
        grupo = torch.tensor([ANDAR, PEGAR, PEGAR, PEGAR])
        out = normaliza_por_grupo(adv, grupo)
        self.assertAlmostEqual(float(out[0]), 0.8, places=6)


class TestFixaNormalizador(unittest.TestCase):
    def _nz(self):
        # canal 0 vivo (desvio ~1), canal 1 constante (desvio 0), canais 2 e 3 um one-hot raro
        nz = EmpiricalNormalization(4)
        x = torch.zeros(1000, 4)
        x[:, 0] = torch.linspace(-1.7, 1.7, 1000)
        x[:3, 2] = 1.0
        nz.update(x)
        return nz

    def test_piso_no_canal_constante_e_fatia_fixa_em_media_zero_desvio_um(self):
        nz = self._nz()
        self.assertLess(float(nz._std[0, 1]), 1e-6)  # sem o conserto: ganho de 100
        var_antes = nz._var.clone()
        fixa_normalizador(nz, [(slice(2, 4), 1.0)])
        self.assertAlmostEqual(float(nz._std[0, 0]), float(var_antes[0, 0]) ** 0.5, places=6)  # canal vivo: intacto
        self.assertAlmostEqual(float(nz._std[0, 1]), PISO_DESVIO)
        self.assertEqual(float(nz._var[0, 1]), float(var_antes[0, 1]))  # o piso não mexe na estatística
        self.assertEqual(nz._mean[0, 2:].tolist(), [0.0, 0.0])
        self.assertEqual(nz._std[0, 2:].tolist(), [1.0, 1.0])
        saida = nz(torch.tensor([[0.0, 1.0, 1.0, 0.0]]))
        self.assertAlmostEqual(float(saida[0, 2]), 1.0 / 1.01, places=5)  # o slot aceso entra como ~1, e não ~18
        self.assertLess(float(saida[0, 1]), 1.0 / PISO_DESVIO)  # o canal constante que muda: ganho < 20

    def test_par_com_desvio_0_1_fixa_media_zero_e_escala_a_entrada(self):
        nz = self._nz()
        fixa_normalizador(nz, [(slice(2, 4), 0.1)])
        self.assertEqual(nz._mean[0, 2:].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(float(nz._std[0, 2]), 0.1, places=6)
        self.assertAlmostEqual(float(nz._var[0, 2]), 0.01, places=6)
        saida = nz(torch.tensor([[0.0, 0.0, 0.1, 0.0]]))
        self.assertAlmostEqual(float(saida[0, 2]), 0.1 / 0.11, places=5)  # 0,1 / (0,1 + 0,01), ~0,909

    def test_sem_normalizador_nao_faz_nada(self):
        fixa_normalizador(torch.nn.Identity(), [(slice(0, 2), 1.0)])

    def test_process_env_step_fixa_depois_do_update_do_pai(self):
        from rsl_rl.algorithms import PPO
        alg = object.__new__(PPOPorTarefa)
        nz_a, nz_c = self._nz(), self._nz()
        alg.actor = unittest.mock.Mock(obs_normalizer=nz_a)
        alg.critic = unittest.mock.Mock(obs_normalizer=nz_c)
        alg.fatias_fixas = {"actor": [(slice(2, 4), 1.0)]}
        with unittest.mock.patch.object(PPO, "process_env_step", return_value="pai") as pai:
            self.assertEqual(alg.process_env_step(1, 2, 3, {}), "pai")
        pai.assert_called_once_with(1, 2, 3, {})
        self.assertEqual(nz_a._std[0, 2:].tolist(), [1.0, 1.0])
        self.assertAlmostEqual(float(nz_c._std[0, 1]), PISO_DESVIO)  # crítico sem fatia: só o piso
        self.assertLess(float(nz_c._std[0, 2]), 1.0)

    def test_compute_returns_agrupa_por_tarefa_e_alvo(self):
        # 4 envs, observação do crítico com 20 colunas: alvo one-hot em 0:10, 4 colunas de ruído, tarefa em 14:20.
        # ANDAR com "nenhum" (o slot 9, o último da fatia) cai no grupo 0; PEGAR (alvos 3 e 7) cai em 1 + 1*10 + alvo.
        obs = torch.zeros(1, 4, 20)
        for env, (alvo, tarefa) in enumerate([(9, ANDAR), (9, ANDAR), (3, PEGAR), (7, PEGAR)]):
            obs[0, env, alvo] = 1.0
            obs[0, env, 14 + tarefa] = 1.0
        st = types.SimpleNamespace(observations={"critic": obs}, returns=torch.ones(1, 4),
                                   values=torch.zeros(1, 4), advantages=None)
        alg = object.__new__(PPOPorTarefa)
        alg.storage = st
        alg.fatia_alvo = slice(0, 10)
        with unittest.mock.patch.object(PPO, "compute_returns"), \
                unittest.mock.patch("g1_v3.algoritmo.normaliza_por_grupo",
                                    side_effect=lambda adv, g: adv) as norm:
            alg.compute_returns(None)
        self.assertEqual(norm.call_args[0][1].tolist(), [0, 0, 14, 18])

    def test_compute_returns_sem_fatia_alvo_agrupa_so_pela_tarefa(self):
        obs = torch.zeros(1, 4, 20)
        for env, tarefa in enumerate([ANDAR, ANDAR, PEGAR, PEGAR]):
            obs[0, env, 14 + tarefa] = 1.0
        st = types.SimpleNamespace(observations={"critic": obs}, returns=torch.ones(1, 4),
                                   values=torch.zeros(1, 4), advantages=None)
        alg = object.__new__(PPOPorTarefa)
        alg.storage = st
        with unittest.mock.patch.object(PPO, "compute_returns"), \
                unittest.mock.patch("g1_v3.algoritmo.normaliza_por_grupo",
                                    side_effect=lambda adv, g: adv) as norm:
            alg.compute_returns(None)
        self.assertEqual(norm.call_args[0][1].tolist(), [0, 0, 1, 1])



N_FATIA = 13  # o `alvo_macro` da M7
D_OBS = N_FATIA + 4 + 6  # a fatia, quatro colunas de enchimento e o one-hot da tarefa nas 6 últimas


def obs_critico(casos):
    """(len(casos), D_OBS): a fatia do alvo acesa no slot e o one-hot da tarefa, por caso (slot, tarefa)."""
    obs = torch.zeros(len(casos), D_OBS)
    for i, (slot, tarefa) in enumerate(casos):
        obs[i, slot] = 1.0
        obs[i, D_OBS - 6 + tarefa] = 1.0
    return obs


def alg_com_fatia():
    alg = object.__new__(PPOPorTarefa)
    alg.fatia_alvo = slice(0, N_FATIA)
    return alg


class TestGrupoEMediaDoGrupo(unittest.TestCase):
    """M7 e M1 (13/11): o grupo do ANDAR e a média de V por grupo que a queda e a perda somam."""

    def test_grupo_do_andar_so_e_zero_com_nenhum(self):
        alg = alg_com_fatia()
        # ANDAR: nenhum (12), B4 fechado (10), perda (11); PEGAR: P4 aberto (4) e P4 fechado (9)
        g = alg._grupo(obs_critico([(12, ANDAR), (10, ANDAR), (11, ANDAR), (4, PEGAR), (9, PEGAR)])).tolist()
        self.assertEqual(g[0], 0)
        self.assertEqual(g[1:3], [1 + 10, 1 + 11])  # grupos próprios, diferentes de 0 e entre si
        self.assertEqual(g[3:], [1 + PEGAR * N_FATIA + 4, 1 + PEGAR * N_FATIA + 9])
        self.assertNotEqual(g[3], g[4])

    def test_grupo_sem_fatia_alvo_e_a_tarefa(self):
        alg = object.__new__(PPOPorTarefa)
        self.assertEqual(alg._grupo(obs_critico([(12, ANDAR), (4, PEGAR), (4, CARREGAR)])).tolist(), [0, 1, 2])

    def test_compute_returns_guarda_a_media_de_v_por_grupo_e_mantem_o_ausente(self):
        # 2 passos, 3 envs: env 0 no grupo 0 (ANDAR nenhum); envs 1 e 2 no PEGAR P4 aberto (grupo 18)
        obs = torch.stack([obs_critico([(12, ANDAR), (4, PEGAR), (4, PEGAR)])] * 2)
        st = types.SimpleNamespace(observations={"critic": obs}, returns=torch.ones(2, 3, 1),
                                   values=torch.tensor([[1.0, 2.0, 4.0], [3.0, 6.0, 8.0]]).unsqueeze(-1),
                                   advantages=None)
        alg = alg_com_fatia()
        alg.storage = st
        alg.v_grupo = torch.full((1 + 6 * N_FATIA,), 9.0)  # o rollout anterior
        with unittest.mock.patch.object(PPO, "compute_returns"):
            alg.compute_returns(None)
        self.assertEqual(alg.v_grupo.shape, (1 + 6 * N_FATIA,))
        self.assertAlmostEqual(float(alg.v_grupo[0]), 2.0)  # (1 + 3) / 2
        self.assertAlmostEqual(float(alg.v_grupo[1 + PEGAR * N_FATIA + 4]), 5.0)  # (2 + 4 + 6 + 8) / 4
        self.assertEqual(float(alg.v_grupo[5]), 9.0)  # grupo ausente: o valor anterior

    def _passo(self, v_grupo, dones, time_outs, perdeu):
        # 4 envs: 0, 1 e 3 no ANDAR nenhum (grupo 0, V̄ −5); 2 no PEGAR P4 aberto (grupo 18, V̄ 3)
        alg = alg_com_fatia()
        alg.gamma = 0.99
        alg.v_grupo = v_grupo
        alg.transition = types.SimpleNamespace(
            observations={"critic": obs_critico([(12, ANDAR), (12, ANDAR), (4, PEGAR), (12, ANDAR)])})
        alg.actor = unittest.mock.Mock(obs_normalizer=torch.nn.Identity())
        alg.critic = unittest.mock.Mock(obs_normalizer=torch.nn.Identity())
        extras = {"time_outs": torch.tensor(time_outs), "v3_perdeu_agora": torch.tensor(perdeu)}
        rewards = torch.zeros(4)
        with unittest.mock.patch.object(PPO, "process_env_step") as pai:
            alg.process_env_step("obs", rewards, torch.tensor(dones), extras)
        return pai.call_args[0][1]

    def _v_grupo(self):
        v = torch.zeros(1 + 6 * N_FATIA)
        v[0], v[1 + PEGAR * N_FATIA + 4] = -5.0, 3.0
        return v

    def test_queda_soma_gama_vezes_o_v_negativo_do_grupo_e_nada_no_v_positivo(self):
        r = self._passo(self._v_grupo(), [1, 0, 1, 0], [False] * 4, [False] * 4)
        self.assertAlmostEqual(float(r[0]), 0.99 * -5.0, places=5)  # queda no grupo de V̄ −5
        self.assertEqual(float(r[2]), 0.0)  # queda no grupo de V̄ 3
        self.assertEqual(float(r[1]), 0.0)  # sem queda
        self.assertEqual(float(r[3]), 0.0)

    def test_time_out_nao_soma(self):
        r = self._passo(self._v_grupo(), [1, 0, 0, 0], [True, False, False, False], [False] * 4)
        self.assertEqual(float(r.abs().sum()), 0.0)

    def test_pulso_da_perda_soma_sem_terminar_o_episodio(self):
        r = self._passo(self._v_grupo(), [0, 0, 0, 0], [False] * 4, [False, True, False, False])
        self.assertAlmostEqual(float(r[1]), 0.99 * -5.0, places=5)
        self.assertEqual(float(r[[0, 2, 3]].abs().sum()), 0.0)

    def test_queda_e_perda_no_mesmo_passo_somam_uma_vez(self):
        r = self._passo(self._v_grupo(), [1, 0, 0, 0], [False] * 4, [True, False, False, False])
        self.assertAlmostEqual(float(r[0]), 0.99 * -5.0, places=5)

    def test_sem_v_grupo_no_primeiro_rollout_nada_muda(self):
        r = self._passo(None, [1, 1, 1, 1], [False] * 4, [True] * 4)
        self.assertEqual(float(r.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
