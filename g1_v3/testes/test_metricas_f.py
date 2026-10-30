"""Testes de g1_v3/metricas.py da F9 (revisão 29/10): M4 (deslocamento depois do apoio), M5 (média só dos envs
com evento) e P3 (velocidade de chegada lida de `v3_v_chegada`). Env falso com `types.SimpleNamespace`.

Os termos de evento devolvem a FLAG do evento no episódio (`Episode_Metrics/<nome>` = fração dos episódios com
evento, que é o `torch.mean` do mjlab sobre os envs que resetam). O valor por episódio vai no `reset`, em
`extras["log"]["Episodio/<nome>"]`, só para os envs que fecham com evento.
"""
import types
import unittest

import torch

from g1_v3 import metricas
from g1_v3.alvos import B1, B2, B3


class MetricasFTest(unittest.TestCase):

    def test_deslocamento_depois_do_apoio_nao_conta_o_b2(self):
        # M4: o B2 é antes do fecho; mesmo com a caixa 2 cm fora do ponto, a métrica dá 0
        box = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=torch.tensor(
            [[0.0, 0.0, 0.5], [0.02, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])))
        env = types.SimpleNamespace(num_envs=4, device="cpu", scene={"box": box}, extras={"log": {}},
                                    v3_alvo_idx=torch.tensor([B2] * 4), v3_ponto_apoio=torch.zeros(4, 3),
                                    v3_perdeu=torch.zeros(4, dtype=torch.bool))
        desl = metricas.deslocamento_depois_do_apoio(None, env)
        self.assertEqual(float(desl(env).mean()), 0.0)
        desl.reset(torch.arange(4))
        self.assertNotIn("Episodio/deslocamento_depois_do_apoio", env.extras["log"])

    def test_deslocamento_depois_do_apoio_no_b3_com_caixa_2cm(self):
        # M4 no B3 com a caixa 2 cm do ponto: só o env 1 tem evento (flag 0/1); o log do reset é 0,02 (M5)
        box = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=torch.tensor(
            [[0.0, 0.0, 0.5], [0.02, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])))
        env = types.SimpleNamespace(num_envs=4, device="cpu", scene={"box": box}, extras={"log": {}},
                                    v3_alvo_idx=torch.tensor([B2, B3, B2, B2]), v3_ponto_apoio=torch.zeros(4, 3),
                                    v3_perdeu=torch.zeros(4, dtype=torch.bool))
        desl = metricas.deslocamento_depois_do_apoio(None, env)
        self.assertTrue(torch.equal(desl(env), torch.tensor([0.0, 1.0, 0.0, 0.0])))
        desl.reset(torch.arange(4))
        self.assertTrue(torch.allclose(env.extras["log"]["Episodio/deslocamento_depois_do_apoio"],
                                       torch.tensor([0.02]), atol=1e-6))

    def test_tempo_no_alvo_loga_so_o_episodio_com_o_alvo(self):
        # M5: 4 envs, 1 com o alvo B1 por 0,8 s: a flag é 1 só nele, e o log é 0,8 (não 0,2, a média com zeros)
        alvo = torch.tensor([0, B1, 0, 0])
        env = types.SimpleNamespace(num_envs=4, device="cpu", step_dt=0.1, v3_alvo_idx=alvo, extras={"log": {}},
                                    v3_tarefa=torch.full((4,), metricas.BOTAR, dtype=torch.long))
        tempo = metricas.tempo_no_alvo(types.SimpleNamespace(params={"alvo": B1}), env)
        for _ in range(8):
            out = tempo(env, alvo=B1)  # o `alvo` vem do `params` do MetricsTermCfg, como no mjlab
        self.assertTrue(torch.equal(out, torch.tensor([0.0, 1.0, 0.0, 0.0])))
        tempo.reset(torch.arange(4))
        self.assertTrue(torch.allclose(env.extras["log"]["Episodio/tempo_B1"], torch.tensor([0.8]), atol=1e-5))

    def test_velocidade_de_chegada_le_o_buffer_e_loga_so_os_com_toque(self):
        # P3 + M5: o valor vem de `v3_v_chegada` (−1 = sem toque). Fecham o env 0 (toque de 0,8 m/s) e o env 1
        # (sem toque). O env 3 tem toque mas não fecha no reset: não entra no log.
        env = types.SimpleNamespace(num_envs=4, device="cpu", extras={"log": {}},
                                    v3_v_chegada=torch.tensor([0.8, -1.0, -1.0, 0.3]))
        m = metricas.velocidade_de_chegada(None, env)
        m(env)
        m.reset(torch.tensor([0, 1]))
        self.assertTrue(torch.allclose(env.extras["log"]["Episodio/velocidade_de_chegada"], torch.tensor([0.8])))

    def test_phi_medio_do_alvo_loga_a_media_do_episodio_com_o_alvo(self):
        # env 0 com o alvo P0 em 2 passos com Φ 0,5 → log 0,5; env 1 está em B1 e não entra
        env = types.SimpleNamespace(num_envs=2, device="cpu", extras={"log": {}},
                                    v3_alvo_idx=torch.tensor([0, B1]), v3_phi=torch.tensor([0.5, 0.1]),
                                    v3_tarefa=torch.full((2,), metricas.BOTAR, dtype=torch.long))
        phi = metricas.phi_medio_do_alvo(types.SimpleNamespace(params={"alvo": 0}), env)
        for _ in range(2):
            out = phi(env, alvo=0)
        self.assertTrue(torch.equal(out, torch.tensor([1.0, 0.0])))
        phi.reset(torch.arange(2))
        self.assertTrue(torch.allclose(env.extras["log"]["Episodio/phi_P0"], torch.tensor([0.5])))

    def test_lote_sem_evento_nao_loga(self):
        # nenhum env do lote com evento: a chave `Episodio/...` não sai
        env = types.SimpleNamespace(num_envs=2, device="cpu", extras={"log": {}},
                                    v3_v_chegada=torch.tensor([-1.0, -1.0]))
        m = metricas.velocidade_de_chegada(None, env)
        m(env)
        m.reset(torch.arange(2))
        self.assertNotIn("Episodio/velocidade_de_chegada", env.extras["log"])


if __name__ == "__main__":
    unittest.main()
