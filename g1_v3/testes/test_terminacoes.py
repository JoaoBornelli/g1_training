"""Testes dos núcleos de terminação da D6 (terminacoes.py). Escritos antes do código: o ImportError é esperado.

Os casos são os da linha "Teste" de D6 (plano §11), com os knobs `Terminacao` (S §10). Tensores sintéticos.
"""
import math
import types
import unittest

import torch

from g1_v3 import balanceador, eventos, terminacoes
from g1_v3.knobs import ATIVO, TAREFAS

K = ATIVO.terminacao
D_TRAV = ATIVO.terreno.dist_travessia
F = torch.float32
ANDAR = TAREFAS.index("ANDAR")  # 0
PEGAR = TAREFAS.index("PEGAR")  # 1


def t(*valores):
    return torch.tensor(valores, dtype=F)


def queda(tarefa, incl_rad, z_pelve, z_chao_pelve, z_joelho=0.5, z_chao_joelho=0.0):
    return terminacoes.n_queda(torch.tensor([tarefa]), t(incl_rad), t(z_pelve), t(z_chao_pelve),
                               t(z_joelho), t(z_chao_joelho), K).tolist()


class TestQueda(unittest.TestCase):
    def test_inclinacao_09_rad_no_andar_cai(self):
        # 0,9 rad > incl_marcha (0,8 rad, S §10).
        self.assertEqual(queda(ANDAR, 0.9, 0.78, 0.0), [True])

    def test_inclinacao_09_rad_no_pegar_nao_cai(self):
        # 0,9 rad = 51,6° < incl_manip_graus (70°): na manipulação a queda é só por inclinação de 70°.
        self.assertEqual(queda(PEGAR, 0.9, 0.78, 0.0), [False])
        self.assertLess(0.9 * 180 / math.pi, K.incl_manip_graus)

    def test_pelve_a_015_do_terreno_com_origem_do_subterreno_em_14_cai(self):
        # z_pelve_w = 1,55; chão do subterreno a 1,40 m: 0,15 m < base_min (0,20 m) → cai.
        self.assertEqual(queda(ANDAR, 0.0, 1.55, 1.40), [True])

    def test_mesma_pelve_com_chao_no_mundo_nao_cai(self):
        # Com z_chao_pelve = 0 (o mundo), a mesma pelve está a 1,55 m e não cai: a conta é do terreno.
        self.assertEqual(queda(ANDAR, 0.0, 1.55, 0.0), [False])


class TestFimDaCadeiaNaTravessia(unittest.TestCase):
    """27/11, T19 §13.3: o episódio de terreno do ANDAR acaba (`time_out`) quando a base passa de `dist_travessia` da origem da célula.
    T19 §13.8: a posição lida é a do `twist.pose_ult` (x, y e rumo no mundo, do passo anterior), a mesma do registro do `sorteia_fatia`."""

    def _fim(self, fatia, coluna, d, passo, max_ep=1000, fim=False, d_raiz=None, dz=0.0):
        # `d` é a distância no plano xy do `pose_ult` à origem; `d_raiz` a do `root_link_pos_w` (igual a `d` se None); a base fica
        # `dz` acima da origem da célula. A 3ª coluna do `pose_ult` é o rumo (0), e não o z
        origem = torch.tensor([[10.0, -5.0, -dz]])
        dr = d if d_raiz is None else d_raiz
        env = types.SimpleNamespace(
            episode_length_buf=torch.tensor([passo]), max_episode_length=max_ep, v3_fim=torch.tensor([fim]),
            v3_fatia=torch.tensor([fatia]),
            command_manager=types.SimpleNamespace(get_term=lambda nome: types.SimpleNamespace(
                pose_ult=torch.tensor([[10.0 + d * 0.6, -5.0 + d * 0.8, 0.0]]))),
            scene=_Cena(origem, types.SimpleNamespace(terrain_types=torch.tensor([coluna])),
                        torch.tensor([[10.0 + dr * 0.6, -5.0 + dr * 0.8, 0.0]])))
        return bool(terminacoes.fim_da_cadeia(env)[0])

    def test_andar_em_terreno_com_d_acima_da_travessia_no_passo_10_termina(self):
        self.assertTrue(self._fim(fatia=0, coluna=2, d=D_TRAV + 0.1, passo=10))

    def test_d_abaixo_da_travessia_nao_termina(self):
        self.assertFalse(self._fim(fatia=0, coluna=2, d=D_TRAV - 0.1, passo=10))

    def test_no_plano_nao_termina_mesmo_com_d_5_m(self):
        self.assertFalse(self._fim(fatia=0, coluna=0, d=5.0, passo=10))

    def test_fatia_b_com_coluna_de_terreno_nao_termina_com_d_5_m(self):
        self.assertFalse(self._fim(fatia=1, coluna=2, d=5.0, passo=10))

    def test_no_passo_2_nao_termina_com_d_5_m(self):
        # a guarda do mjlab: no reset a posição lida ainda é a velha
        self.assertFalse(self._fim(fatia=0, coluna=2, d=5.0, passo=2))

    def test_no_passo_3_termina_com_d_5_m(self):
        # a guarda é `> 2`: com `> 3` o passo 3 não terminaria
        self.assertTrue(self._fim(fatia=0, coluna=2, d=5.0, passo=3))

    def test_d_logo_acima_da_travessia_termina(self):
        # o limiar é o `dist_travessia`, e não um valor acima dele
        self.assertTrue(self._fim(fatia=0, coluna=2, d=D_TRAV + 0.02, passo=10))

    def test_o_z_nao_entra(self):
        # base 1,0 m acima da origem com `dist_travessia` − 0,1 m no plano: a norma 3D passaria de `dist_travessia` e terminaria
        self.assertGreater(math.hypot(D_TRAV - 0.1, 1.0), D_TRAV)
        self.assertFalse(self._fim(fatia=0, coluna=2, d=D_TRAV - 0.1, passo=10, dz=1.0))

    def test_segue_o_pose_ult_e_nao_o_root_link_pos_w(self):
        # a leitura de `root_link_pos_w` inverte os dois casos
        self.assertTrue(self._fim(fatia=0, coluna=2, d=D_TRAV + 0.1, passo=10, d_raiz=D_TRAV - 0.1))
        self.assertFalse(self._fim(fatia=0, coluna=2, d=D_TRAV - 0.1, passo=10, d_raiz=D_TRAV + 0.1))

    def test_as_regras_de_hoje_seguem(self):
        self.assertTrue(self._fim(fatia=0, coluna=0, d=0.0, passo=1000, max_ep=1000))  # `max_episode_length`
        self.assertTrue(self._fim(fatia=0, coluna=0, d=0.0, passo=10, fim=True))  # `v3_fim`


class TestTerminoERegistroLeemAMesmaPosicao(unittest.TestCase):
    """T19 §13.8: a ordem de um passo do mjlab (`docs/treino/10-tarefa-andar.md` §2.11) é terminações, recompensas, reset (o
    `sorteia_fatia` é o primeiro evento), `sim.forward` e o `compute` dos comandos, que grava o `pose_ult`. No passo k, o
    `root_link_pos_w` é a posição k e o `pose_ult` a k−1. No passo em que o término dispara, o registro tem de contar a travessia."""

    def test_no_passo_do_termino_o_registro_conta_visita_e_travessia(self):
        T, dt = ATIVO.terreno, 0.02
        origem = torch.tensor([[10.0, -5.0, 0.0]])
        xs = [D_TRAV - 0.05, D_TRAV - 0.01, D_TRAV + 0.01, D_TRAV + 0.05]  # distância da base à origem da célula, passo a passo
        terreno = types.SimpleNamespace(terrain_types=torch.tensor([2]), terrain_levels=torch.zeros(1, dtype=torch.long),
                                        terrain_origins=torch.zeros(T.linhas, len(T.colunas), 3))
        twist = types.SimpleNamespace(pose_ult=None, soma_cmd_marcha=torch.tensor([2.5 / dt]))  # pedido de 2,5 m: a visita vem do d
        cur = balanceador.Curriculo(ATIVO.balanceador, ATIVO.niveis, ATIVO.comando, T)
        env = types.SimpleNamespace(
            device="cpu", step_dt=dt, v3_cur=cur, v3_gerador=torch.Generator().manual_seed(0), v3_fatia=torch.tensor([0]),
            v3_t_empurrao=torch.zeros(1), max_episode_length=1000, v3_fim=torch.tensor([False]),
            command_manager=types.SimpleNamespace(get_term=lambda nome: twist),
            termination_manager=types.SimpleNamespace(terminated=torch.tensor([False])))  # o `time_out` não entra em `terminated`
        for k in range(1, len(xs)):
            env.episode_length_buf = torch.tensor([10 + k])
            twist.pose_ult = origem + torch.tensor([[xs[k - 1], 0.0, 0.0]])  # x, y e rumo (0), gravados no passo k−1
            env.scene = _Cena(origem.clone(), terreno, origem + torch.tensor([[xs[k], 0.0, 0.78]]))
            if terminacoes.fim_da_cadeia(env)[0]:
                break
        else:
            self.fail("o término não disparou")
        self.assertEqual(k, 3)  # dispara no passo em que o `pose_ult` (a posição do passo anterior, xs[2]) passa de `dist_travessia`
        eventos.sorteia_fatia(env, torch.arange(1), T)
        self.assertEqual((cur.vis_ter[2], cur.trav_ter[2]), (1, 1))


class _Cena:
    # `env.scene["robot"].data.root_link_pos_w`, `env.scene["terrain"]` e `env.scene.env_origins`, como a cena do mjlab
    def __init__(self, origem, terreno, base):
        self.env_origins = origem
        self._itens = {"robot": types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=base)), "terrain": terreno}

    def __getitem__(self, nome):
        return self._itens[nome]


if __name__ == "__main__":
    unittest.main()
