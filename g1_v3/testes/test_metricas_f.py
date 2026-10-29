"""Testes de g1_v3/metricas.py da F9 (revisão 29/10): M4 (deslocamento depois do apoio), M5 (média só dos envs
com evento) e P3 (velocidade de chegada lida de `v3_v_chegada`). Env falso com `types.SimpleNamespace`.

A média dos envs é o que o mjlab loga para uma métrica `reduce="last"` (`metrics_manager.py:113-124`): `torch.mean`
sobre os envs que resetam, de um valor que cada env guarda no último passo.
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
        env = types.SimpleNamespace(num_envs=4, device="cpu", scene={"box": box},
                                    v3_alvo_idx=torch.tensor([B2] * 4), v3_ponto_apoio=torch.zeros(4, 3),
                                    v3_perdeu=torch.zeros(4, dtype=torch.bool))
        desl = metricas.deslocamento_depois_do_apoio(None, env)
        self.assertEqual(float(desl(env).mean()), 0.0)

    def test_deslocamento_depois_do_apoio_no_b3_com_caixa_2cm(self):
        # M4 no B3 com a caixa 2 cm do ponto: só o env 1 tem evento, e a média dos 4 envs é 0,02 (M5)
        box = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=torch.tensor(
            [[0.0, 0.0, 0.5], [0.02, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])))
        env = types.SimpleNamespace(num_envs=4, device="cpu", scene={"box": box},
                                    v3_alvo_idx=torch.tensor([B2, B3, B2, B2]), v3_ponto_apoio=torch.zeros(4, 3),
                                    v3_perdeu=torch.zeros(4, dtype=torch.bool))
        desl = metricas.deslocamento_depois_do_apoio(None, env)
        self.assertAlmostEqual(float(desl(env).mean()), 0.02, places=6)

    def test_tempo_no_alvo_media_so_dos_envs_com_o_alvo(self):
        # M5: 4 envs, 1 com o alvo B1 por 0,8 s: a métrica logada é 0,8, e não 0,2 (a média diluída pelos zeros)
        alvo = torch.tensor([0, B1, 0, 0])
        env = types.SimpleNamespace(num_envs=4, device="cpu", step_dt=0.1, v3_alvo_idx=alvo,
                                    v3_tarefa=torch.full((4,), metricas.BOTAR, dtype=torch.long))
        tempo = metricas.tempo_no_alvo(None, env)
        for _ in range(8):
            out = tempo(env, alvo=B1)  # o `alvo` vem do `params` do MetricsTermCfg, como no mjlab
        self.assertAlmostEqual(float(out.mean()), 0.8, places=5)

    def test_velocidade_de_chegada_le_o_buffer_e_so_media_os_com_toque(self):
        # P3 + M5: o valor vem de `v3_v_chegada` (−1 = sem toque); 4 envs, 1 com toque de 0,8 m/s dá 0,8
        env = types.SimpleNamespace(num_envs=4, device="cpu",
                                    v3_v_chegada=torch.tensor([-1.0, 0.8, -1.0, -1.0]))
        self.assertAlmostEqual(float(metricas.velocidade_de_chegada(env).mean()), 0.8, places=5)


if __name__ == "__main__":
    unittest.main()
