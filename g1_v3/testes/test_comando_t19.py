"""T19 (27/11): o comando "frente + rumo" do ANDAR de terreno no `TwistV3`: `_resample_command` liga o rumo e grava o alvo, e
`_update_command` chama o do fabricante (wz = rigidez × erro de rumo) e zera o giro abaixo de `giro_min`.
`TwistV3` sem o `__init__` (que pede a cena e o fabricante), como os vizinhos: só o que os dois métodos leem.
"""
import math
import types
import unittest
from unittest import mock

import torch

from g1_v3 import balanceador, comando
from g1_v3.knobs import ATIVO, TAREFAS

K = ATIVO.comando
ANDAR, CARREGAR = TAREFAS.index("ANDAR"), TAREFAS.index("CARREGAR")
FATIA_ANDAR, FATIA_B = 0, 1
TETO_GIRO = ATIVO.comando.giro_tetos[-1]


def _twist(yaw, coluna, tarefa, fatia):
    # um env por linha. `yaw` (rad) é o rumo de partida, gravado no `qpos` (w, x, y, z); `coluna` é o `terrain_types` do episódio
    n = len(yaw)
    cur = balanceador.Curriculo(ATIVO.balanceador, ATIVO.niveis, K, ATIVO.terreno)
    cur._nivel["CARREGAR"] = 1  # um nível em que o `sorteia_velocidade` do CARREGAR de terreno anda
    # o free joint começa na coluna 5 do `qpos` (endereço 5..11: pos em 5..7, quat w x y z em 8..11), com lixo nas colunas 0..4:
    # uma leitura de `qpos[:, 3:7]` sem o endereço lê o lixo e o rumo sai errado
    qpos = torch.zeros(n, 12)
    qpos[:, :5] = 3.0
    qpos[:, 8], qpos[:, 11] = torch.cos(torch.tensor(yaw) / 2), torch.sin(torch.tensor(yaw) / 2)
    env = types.SimpleNamespace(
        device="cpu", num_envs=n, v3_cur=cur, v3_gerador=torch.Generator().manual_seed(0),
        v3_tarefa=torch.tensor(tarefa), v3_fatia=torch.tensor(fatia), v3_celula=torch.full((n,), -1, dtype=torch.long),
        scene={"terrain": types.SimpleNamespace(terrain_types=torch.tensor(coluna))},
        sim=types.SimpleNamespace(data=types.SimpleNamespace(qpos=qpos)))
    obj = comando.TwistV3.__new__(comando.TwistV3)
    obj._env, obj.k = env, K
    obj.cfg = types.SimpleNamespace(heading_command=True, heading_control_stiffness=1.0,
                                    ranges=types.SimpleNamespace(ang_vel_z=(-TETO_GIRO, TETO_GIRO)))
    obj.robot = types.SimpleNamespace(
        indexing=types.SimpleNamespace(free_joint_q_adr=torch.arange(5, 12)),
        data=types.SimpleNamespace(forward_vec_b=torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1), heading_w=torch.zeros(n)))
    obj.vel_command_b, obj.vel_command_w = torch.zeros(n, 3), torch.zeros(n, 3)
    obj.time_left, obj._agora = torch.zeros(n), torch.zeros(n)
    obj.heading_target, obj.heading_error = torch.zeros(n), torch.zeros(n)
    obj.is_heading_env, obj.is_standing_env, obj.is_world_env = (torch.zeros(n, dtype=torch.bool) for _ in range(3))
    return obj


class ResampleDoRumo(unittest.TestCase):

    def _resample(self):
        # 0..2: ANDAR de terreno, 3: ANDAR no plano (coluna 0), 4: CARREGAR na fatia B de terreno
        obj = _twist(yaw=[0.3, 2.0, -3.0, 1.0, 1.0], coluna=[3, 4, 5, 0, 2], tarefa=[ANDAR] * 4 + [CARREGAR],
                     fatia=[FATIA_ANDAR] * 4 + [FATIA_B])
        obj.is_heading_env[:] = True  # o resample zera as flags dos envs que reamostra
        obj._resample_command(torch.arange(5))
        return obj

    def test_so_o_andar_de_terreno_liga_o_rumo(self):
        obj = self._resample()
        self.assertEqual(obj.is_heading_env.tolist(), [True, True, True, False, False])

    def test_alvo_e_o_multiplo_de_pi_sobre_2_mais_proximo_do_rumo_de_partida(self):
        # 0,3 → 0; 2,0 (1,27 × π/2) → π/2; −3,0 (−1,91 × π/2) → −π
        obj = self._resample()
        self.assertEqual([round(x / (math.pi / 2)) for x in obj.heading_target[:3].tolist()], [0, 1, -2])
        for x in obj.heading_target[:3].tolist():
            self.assertAlmostEqual(x / (math.pi / 2), round(x / (math.pi / 2)), places=5)

    def test_comando_do_terreno_e_so_frente_e_o_resto_sai_como_antes(self):
        obj = self._resample()
        self.assertTrue(bool((obj.vel_command_b[:3, 1:] == 0).all()))
        self.assertTrue(bool((obj.vel_command_b[:3, 0] >= 0).all()))
        self.assertGreater(float(obj.vel_command_b[:3, 0].max()), 0.0)
        # o CARREGAR de B segue com o comando fixo no trecho, sem rumo 
        self.assertEqual(float(obj.time_left[4]), float("inf"))

    def test_no_meio_do_episodio_o_alvo_fica(self):
        # `_agora` > 0: a reamostragem de 3 a 8 s mantém o rumo-alvo e só renova o comando
        obj = _twist(yaw=[2.0], coluna=[4], tarefa=[ANDAR], fatia=[FATIA_ANDAR])
        obj.heading_target[:] = 0.7
        obj._agora[:] = 5.0
        with mock.patch.object(comando, "sorteia_velocidade", return_value=torch.tensor([[0.4, 0.0, 0.0]])):  # com linear: o rumo liga
            obj._resample_command(torch.arange(1))
        self.assertAlmostEqual(float(obj.heading_target[0]), 0.7, places=6)
        self.assertTrue(bool(obj.is_heading_env[0]))

    def test_so_reamostra_os_envs_pedidos(self):
        obj = _twist(yaw=[2.0, 2.0], coluna=[4, 4], tarefa=[ANDAR] * 2, fatia=[FATIA_ANDAR] * 2)
        with mock.patch.object(comando, "sorteia_velocidade", return_value=torch.tensor([[0.4, 0.0, 0.0]])):
            obj._resample_command(torch.tensor([1]))
        self.assertEqual(obj.is_heading_env.tolist(), [False, True])
        self.assertEqual(float(obj.heading_target[0]), 0.0)

    def _resample_com_linear(self, vx, yaw=2.0):
        # um env de ANDAR de terreno cujo sorteio de velocidade devolve `vx` (o resto do comando fica 0)
        obj = _twist(yaw=[yaw], coluna=[4], tarefa=[ANDAR], fatia=[FATIA_ANDAR])
        with mock.patch.object(comando, "sorteia_velocidade", return_value=torch.tensor([[vx, 0.0, 0.0]])):
            obj._resample_command(torch.arange(1))
        return obj

    def test_comando_linear_parado_no_terreno_nao_liga_o_rumo_e_o_alvo_fica_gravado(self):
        # 27/11, T19 §12.2: `andar_parado` ou `linear_min` zera o linear; "parado" não vira "gira até alinhar"
        obj = self._resample_com_linear(0.0)
        self.assertFalse(bool(obj.is_heading_env[0]))
        self.assertAlmostEqual(float(obj.heading_target[0]) / (math.pi / 2), 1.0, places=5)  # o alvo vale para o segmento seguinte
        obj.heading_target[:] = 0.5  # erro de rumo de 0,5 rad (o robô está em 0)
        obj._update_command()
        self.assertEqual(float(obj.vel_command_b[0, 2]), 0.0)

    def test_comando_linear_acima_do_limiar_liga_o_rumo_e_o_wz_segue_o_erro(self):
        obj = self._resample_com_linear(K.linear_min)
        self.assertTrue(bool(obj.is_heading_env[0]))
        obj.heading_target[:] = 0.3
        obj._update_command()
        self.assertAlmostEqual(float(obj.vel_command_b[0, 2]), 0.3, places=6)


class UpdateDoRumo(unittest.TestCase):

    def _update(self, erro, rumo_ligado, wz0=0.0):
        # rumo do robô 0 e alvo = `erro` (rad): o erro de rumo de cada env
        n = len(erro)
        obj = _twist(yaw=[0.0] * n, coluna=[0] * n, tarefa=[ANDAR] * n, fatia=[FATIA_ANDAR] * n)
        obj.heading_target[:] = torch.tensor(erro)
        obj.is_heading_env[:] = torch.tensor(rumo_ligado)
        obj.vel_command_b[:, 2] = wz0
        obj._update_command()
        return obj.vel_command_b[:, 2].tolist()

    def test_wz_segue_o_erro_de_rumo_com_teto_no_giro_e_zera_abaixo_de_giro_min(self):
        # rigidez 1,0: wz = erro; teto ±0,5; |wz| < `giro_min` (0,1) vira 0 (a regra da T15)
        wz = self._update([0.3, -0.3, 0.05, -0.09, 2.0, -2.0, 0.12], [True] * 7)
        for obtido, esperado in zip(wz, [0.3, -0.3, 0.0, 0.0, TETO_GIRO, -TETO_GIRO, 0.12]):
            self.assertAlmostEqual(obtido, esperado, places=6)

    def test_o_erro_de_rumo_dá_a_volta_pelo_lado_curto(self):
        # alvo −π + 0,2 com o robô em π − 0,1: o erro é +0,3 rad (pelo lado curto), e não −6
        n = 1
        obj = _twist(yaw=[0.0] * n, coluna=[0] * n, tarefa=[ANDAR] * n, fatia=[FATIA_ANDAR] * n)
        obj.robot.data.heading_w[:] = math.pi - 0.1
        obj.heading_target[:] = -math.pi + 0.2
        obj.is_heading_env[:] = True
        obj._update_command()
        self.assertAlmostEqual(float(obj.vel_command_b[0, 2]), 0.3, places=5)

    def test_fora_do_terreno_o_wz_sorteado_nao_muda(self):
        wz = self._update([1.0, 1.0, 0.05], [False, False, False], wz0=0.3)
        for x in wz:
            self.assertAlmostEqual(x, 0.3, places=6)
        # um wz sorteado pequeno de um env sem rumo também fica (a regra `giro_min` é do sorteio, não do `_update_command`)
        self.assertAlmostEqual(self._update([0.0], [False], wz0=0.05)[0], 0.05, places=6)

    def test_o_resto_do_fabricante_nao_age(self):
        # sem `is_standing_env` nem `is_world_env` o `_update_command` não mexe no vx e vy
        obj = _twist(yaw=[0.0, 0.0], coluna=[3, 3], tarefa=[ANDAR] * 2, fatia=[FATIA_ANDAR] * 2)
        obj.vel_command_b[:] = torch.tensor([[0.4, 0.1, 0.0], [0.2, -0.1, 0.0]])
        antes = obj.vel_command_b[:, :2].clone()
        obj.is_heading_env[:] = True
        obj.heading_target[:] = 0.3
        obj._update_command()
        self.assertTrue(bool((obj.vel_command_b[:, :2] == antes).all()))

class NotaDoSegmentoDeRumo(unittest.TestCase):
    """27/11, T19 §12.1: o `_abre` não copia o giro de rumo para o `seg_cmd`. `_abre` e `_fecha` reais, sem o `__init__`."""

    DUR = 4.0

    def _abre(self, rumo, wz=0.3):
        n = 1
        env = types.SimpleNamespace(
            v3_incl_max_seg=torch.zeros(n), v3_celula=torch.tensor([0]), v3_tarefa=torch.tensor([ANDAR]),
            v3_ancora=torch.tensor([[0.0, 0.0, 0.0, 1.5]]), v3_perdeu=torch.zeros(n, dtype=torch.bool),
            v3_cur=types.SimpleNamespace(id_celula=lambda tarefa, tipo: -1, restricao=lambda: 1.0, registra=mock.Mock()))
        obj = comando.TwistV3.__new__(comando.TwistV3)
        obj._env, obj.k = env, K
        obj._agora, obj.t_ab = torch.zeros(n), torch.zeros(n)
        obj.vel_command_b = torch.tensor([[0.4, 0.0, wz]])
        obj.is_heading_env = torch.tensor([rumo])
        obj.seg_cmd, obj.seg_cel = torch.zeros(n, 3), torch.full((n,), -1, dtype=torch.long)
        obj.seg_carrega, obj.seg_visto = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.long)
        obj.command_counter = torch.zeros(n, dtype=torch.long)
        obj.seg_proj, obj.seg_int = torch.zeros(n, 2), torch.zeros(n, 2)
        obj.seg_nota, obj.seg_dur = torch.zeros(n, 3), torch.zeros(n)
        obj.ancora_feita = torch.zeros(n, dtype=torch.bool)
        obj._abre(torch.arange(n))
        return obj

    def test_env_de_rumo_abre_o_segmento_sem_comando_de_giro(self):
        self.assertEqual(self._abre(True).seg_cmd[0].tolist(), [torch.tensor(0.4).item(), 0.0, 0.0])

    def test_env_sem_rumo_copia_o_wz(self):
        self.assertEqual(self._abre(False).seg_cmd[0].tolist(), [torch.tensor(0.4).item(), 0.0, torch.tensor(0.3).item()])

    def test_segmento_de_rumo_com_linear_entregue_e_giro_medio_zero_tira_nota_1(self):
        # o robô anda a 0,4 m/s (o linear comandado) e corrige o rumo em menos de 1 s: o giro médio do segmento é ~0
        obj = self._abre(True)
        obj.seg_nota[0] = torch.tensor([0.4, 0.0, 0.0]) * self.DUR
        obj.seg_dur[0] = self.DUR
        obj.seg_proj[0], obj.seg_int[0] = torch.tensor([0.4 * self.DUR, 0.0]), torch.tensor([0.4 * self.DUR, 0.3 * 0.5])
        obj.pose_ult, obj._agora = torch.zeros(1, 3), torch.tensor([5.0])
        obj.metrics = {k: torch.zeros(1) for k in ("eficiencia_min", "eficiencia_media", "segmentos")}
        obj._fecha(torch.tensor([0]), torch.tensor([False]))
        celula, n, soma = obj._env.v3_cur.registra.call_args.args
        self.assertEqual((celula, n), (0, 1))
        self.assertAlmostEqual(soma, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
