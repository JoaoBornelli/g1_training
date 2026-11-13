"""Testes da qualidade do CARREGAR no terreno (G7 S3; decisão P2 do dono, §6.3 "Fase G"; S §8.2; E §4.4).

Env falso com `types.SimpleNamespace`, como os vizinhos. Na célula `terreno` o alvo da qualidade é a pelve +
`caixa_transporte_b` no frame da pelve (z incluso); no plano fica o alvo de hoje (chão sob a pelve + 0,90).
"""
import math
import types
import unittest

import torch

from g1_v3 import recompensas
from g1_v3.knobs import ATIVO

K = ATIVO.alvos
CELULA_TERRENO = 5
TB = (0.25, 0.0, 0.111)  # `caixa_transporte_b` de teste: 0,25 m à frente da pelve e 0,111 m acima dela
PELVE_XY = (1.0, 2.0)
PELVE_Z = 0.789
CHAO_SOB_PELVE = -0.15  # o chão sob a pelve fica 0,15 m abaixo do pé: a quina do degrau


def env_falso(celula, yaw, caixa_z):
    """Env de 1 env: pelve com rumo `yaw`, caixa rígida de identidade em (base + R·TB, `caixa_z`)."""
    c, s = math.cos(yaw), math.sin(yaw)
    caixa_xy = (PELVE_XY[0] + c * TB[0] - s * TB[1], PELVE_XY[1] + s * TB[0] + c * TB[1])
    quat_pelve = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])
    robo = types.SimpleNamespace(data=types.SimpleNamespace(
        root_link_pos_w=torch.tensor([[PELVE_XY[0], PELVE_XY[1], PELVE_Z]]), root_link_quat_w=quat_pelve))
    caixa = types.SimpleNamespace(data=types.SimpleNamespace(
        root_link_pos_w=torch.tensor([[caixa_xy[0], caixa_xy[1], caixa_z]]),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
    mao = types.SimpleNamespace(data=types.SimpleNamespace(found=torch.ones(1, 1)))
    cena = {"robot": robo, "box": caixa, "palma_E": mao, "palma_D": mao,
            "altura_pelve": types.SimpleNamespace(data=types.SimpleNamespace(
                heights=torch.tensor([[PELVE_Z - CHAO_SOB_PELVE]])))}
    ref = types.SimpleNamespace(caixa_transporte_b=lambda a: torch.tensor([TB], dtype=torch.float32))
    cur = types.SimpleNamespace(id_celula=lambda tarefa, tipo: CELULA_TERRENO
                                if (tarefa, tipo) == ("CARREGAR", "terreno") else -1)
    return types.SimpleNamespace(
        device="cpu", scene=cena, v3_ref=ref, v3_cur=cur, v3_meia=torch.tensor([0.1]),
        v3_alvo_idx=torch.zeros(1, dtype=torch.long), v3_caixa_pega_w=torch.zeros(1, 3),
        v3_alvo_w=torch.zeros(1, 3), v3_celula=torch.tensor([celula], dtype=torch.long),
        v3_degrau=torch.zeros(1, dtype=torch.long))


# Altura da caixa no alvo de hoje (chão sob a pelve + 0,90) e no transporte da pelve (pelve + 0,111)
Z_ALVO_DE_HOJE = CHAO_SOB_PELVE + K.z_transporte
Z_TRANSPORTE_PELVE = PELVE_Z + TB[2]


class QualidadeNoTerreno(unittest.TestCase):
    def test_terreno_caixa_em_pelve_mais_transporte_da_qualidade_1(self):
        # G7 S3 (P2): na célula `terreno`, a caixa rígida em pelve + transporte (rumo 90°) dá qualidade 1. Hoje o
        # alvo é o chão sob a pelve + 0,90, 0,15 m abaixo: d = 0,15 e q = exp(−9) ≈ 0. Falha antes do conserto.
        env = env_falso(CELULA_TERRENO, yaw=math.pi / 2, caixa_z=Z_TRANSPORTE_PELVE)
        self.assertAlmostEqual(recompensas._qualidade(env)[0].item(), 1.0, places=4)


class QualidadeNoPlano(unittest.TestCase):
    def test_plano_caixa_no_alvo_de_hoje_da_qualidade_1(self):
        # Regressão: fora da célula `terreno` (sem célula, aqui −1) o alvo é o de hoje, chão sob a pelve + 0,90.
        env = env_falso(-1, yaw=math.pi / 2, caixa_z=Z_ALVO_DE_HOJE)
        self.assertAlmostEqual(recompensas._qualidade(env)[0].item(), 1.0, places=4)

    def test_plano_caixa_no_transporte_da_pelve_nao_conta(self):
        # No plano o alvo continua sendo o de hoje: a caixa na pelve + transporte fica 0,15 m longe e q ≈ 0.
        env = env_falso(-1, yaw=math.pi / 2, caixa_z=Z_TRANSPORTE_PELVE)
        self.assertLess(recompensas._qualidade(env)[0].item(), 0.01)


if __name__ == "__main__":
    unittest.main()
