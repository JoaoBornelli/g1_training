"""Testes da terminação `batente` (H3 do plano; §6.3 "Fase H: núcleos puros (30/10)", §12.4 H3; S §10;
caso 7 do registro, P13 e P14). Escritos antes do código: o AttributeError de `n_batente` e `batente` é o
esperado até a lane H3.

Núcleo: tensores sintéticos, E = 1, centro 0 e meio 1 nas 29 juntas. Invólucro: env falso (`SimpleNamespace`),
com `scene["robot"].data` (joint_pos e joint_pos_limits), `v3_degrau`, `v3_tarefa` e `device`.
"""
import types
import unittest

import torch

from g1_v3 import terminacoes
from g1_v3.knobs import ATIVO, JUNTAS, TAREFAS

T = ATIVO.terminacao
DEGRAU_MIN = ATIVO.refino.degrau_batente  # 1
J_TORNOZELO = JUNTAS.index("left_ankle_roll_joint")
J_YAW_E = JUNTAS.index("left_hip_yaw_joint")
J_YAW_D = JUNTAS.index("right_hip_yaw_joint")
TODAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")


def _q(valor=0.0, junta=J_TORNOZELO):
    q = torch.zeros(1, len(JUNTAS))
    q[0, junta] = float(valor)
    return q


def _yaw(valor, lado=J_YAW_E):
    # (E,2) com o hip_yaw esquerdo e o direito; só o `lado` recebe `valor`
    y = torch.zeros(1, 2)
    y[0, 0 if lado == J_YAW_E else 1] = float(valor)
    return y


def batente_nucleo(q, q_yaw, degrau):
    return terminacoes.n_batente(q, torch.zeros(len(JUNTAS)), torch.ones(len(JUNTAS)), T.batente_frac,
                                 q_yaw, T.hip_yaw_max, torch.tensor([degrau]), DEGRAU_MIN).item()


class NBatente(unittest.TestCase):

    def test_junta_a_1_01_do_meio_curso_no_degrau_1_termina(self):
        # H3: 1,01 no degrau 1 → True (teste a 1,01, P14)
        self.assertTrue(batente_nucleo(_q(1.01), _yaw(0.0), 1))

    def test_junta_a_menos_1_01_no_degrau_1_termina(self):
        # H3: −1,01 no degrau 1 → True (|q − c|/m, nos dois lados do curso)
        self.assertTrue(batente_nucleo(_q(-1.01), _yaw(0.0), 1))

    def test_junta_a_1_01_no_degrau_0_nao_termina(self):
        # H3: 1,01 no degrau 0 → False (a liga é do degrau 1 em diante)
        self.assertFalse(batente_nucleo(_q(1.01), _yaw(0.0), 0))

    def test_junta_exata_no_batente_nao_termina(self):
        # H3: 1,0 exato no degrau 1 → False (`>` estrito, P14)
        self.assertFalse(batente_nucleo(_q(1.0), _yaw(0.0), 1))

    def test_hip_yaw_a_1_01_no_degrau_1_termina(self):
        # H3: |hip_yaw| 1,01 no degrau 1 → True (o segundo gatilho, S §10)
        self.assertTrue(batente_nucleo(_q(0.0), _yaw(1.01), 1))

    def test_hip_yaw_a_0_99_nao_termina(self):
        # H3: |hip_yaw| 0,99 → False
        self.assertFalse(batente_nucleo(_q(0.0), _yaw(0.99), 1))

    def test_hip_yaw_a_menos_1_01_termina(self):
        # H3: −1,01 no hip_yaw → True (|hip_yaw|)
        self.assertTrue(batente_nucleo(_q(0.0), _yaw(-1.01), 1))

    def test_hip_yaw_a_1_01_no_degrau_0_nao_termina(self):
        # H3: |hip_yaw| 1,01 no degrau 0 → False
        self.assertFalse(batente_nucleo(_q(0.0), _yaw(1.01), 0))

    def test_centro_e_meio_por_env_sao_aceitos(self):
        # H3: centro e meio podem ser (E,29). Com centro 0,5 e meio 1, q = 1,52 passa do batente (curso 1,02)
        centro = torch.full((1, len(JUNTAS)), 0.5)
        meio = torch.ones(1, len(JUNTAS))
        termina = terminacoes.n_batente(_q(1.52), centro, meio, T.batente_frac, _yaw(0.0), T.hip_yaw_max,
                                        torch.tensor([1]), DEGRAU_MIN)
        self.assertTrue(termina.item())


class Batente(unittest.TestCase):

    def _env(self, q=None, lim=None, degrau=1, tarefa="ANDAR"):
        # Env falso: joint_pos (E,29), joint_pos_limits (E,29,2); `lim` padrão = (−1; 1) em todas as juntas
        if q is None:
            q = torch.zeros(1, len(JUNTAS))
        if lim is None:
            lim = torch.tensor([-1.0, 1.0]).expand(1, len(JUNTAS), 2).clone()
        robo = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=q, joint_pos_limits=lim))
        return types.SimpleNamespace(
            scene={"robot": robo},
            v3_degrau=torch.tensor([degrau]),
            v3_tarefa=torch.tensor([TAREFAS.index(tarefa)]),
            device="cpu",
        )

    def test_tornozelo_a_centro_mais_1_01_meio_no_degrau_1_termina(self):
        # H3: o left_ankle_roll_joint a centro + 1,01·meio (limites ±1) no degrau 1 → True
        q = _q(1.01, J_TORNOZELO)
        self.assertTrue(terminacoes.batente(self._env(q=q), TODAS).item())

    def test_limite_assimetrico_1_52_termina_e_1_49_nao(self):
        # H3: limites (−0,5; 1,5) numa junta → centro 0,5 e meio 1,0. q = 1,52 → curso 1,02 → True; 1,49 → False
        lim = torch.tensor([-0.5, 1.5]).expand(1, len(JUNTAS), 2).clone()
        self.assertTrue(terminacoes.batente(self._env(q=_q(1.52), lim=lim), TODAS).item())
        self.assertFalse(terminacoes.batente(self._env(q=_q(1.49), lim=lim), TODAS).item())

    def test_tarefa_fora_de_tarefas_nao_termina(self):
        # H3: a tarefa fora de `tarefas` → False, mesmo com a junta além do batente
        env = self._env(q=_q(1.01), tarefa="ANDAR")
        self.assertFalse(terminacoes.batente(env, ("PEGAR",)).item())


if __name__ == "__main__":
    unittest.main()
