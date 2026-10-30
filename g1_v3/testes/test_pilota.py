"""Testes do `pilota_v3.py` (D11 do plano, linha "Teste:").

Escritos contra o CONTRATO da docstring do `exporta_cena.py` ("Contrato com o pilota_v3") e contra
`knobs.Contrato.blocos_ator`, e NÃO contra o código do pilota: se o código não cumpre o contrato, o teste falha.

Roda sem física e sem o mjlab: a cena é falsa (um objeto com `qpos`, `qvel`, `xquat`, `sensordata`).
O pilota importa `torch` no topo (o `Ator`), então o venv precisa de torch.
"""
import importlib.util
import os
import py_compile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from g1_v3.knobs import ATIVO, TAREFAS, Contrato

RAIZ = Path(__file__).resolve().parents[2]
CONTRATO = Contrato()
DIM_ATOR = 1447
DIM_QUADRO = 113
HISTORICO = 5

NQ, NV, NCORPOS = 40, 40, 3
ID_BASE = 1
ADR_ANG = 10


def _carrega_pilota():
    """Carrega o `pilota_v3.py` da raiz por caminho, como o exportador faz."""
    spec = importlib.util.spec_from_file_location("pilota_v3_teste", RAIZ / "pilota_v3.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


PILOTA = _carrega_pilota()


def _cena_falsa(semente: int):
    """`d` (estado vivo, com os campos que `monta_observacao` lê) e `c` (os campos de cena.npz que ele usa)."""
    rng = np.random.default_rng(semente)
    d = SimpleNamespace(
        qpos=rng.standard_normal(NQ),
        qvel=rng.standard_normal(NV),
        xquat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (NCORPOS, 1)),  # base sem rotação: gravidade = (0, 0, −1)
        sensordata=rng.standard_normal(20),
    )
    c = {
        "ids_junta_qpos": np.arange(7, 36, dtype=np.int64),
        "ids_junta_qvel": np.arange(6, 35, dtype=np.int64),
        "q_default": 0.1 * rng.standard_normal(29),
        "qd_default": np.zeros(29),
        "adr_ang_vel": np.int64(ADR_ANG),
        "id_base": np.int64(ID_BASE),
    }
    return d, c


def _entradas(semente: int) -> dict:
    rng = np.random.default_rng(semente)
    return {
        "twist": rng.standard_normal(3),
        "tarefa": 2,
        "anterior": rng.standard_normal(29),
        "caixa": rng.standard_normal(11),
        "mapa": rng.standard_normal(882),
    }


def _chama(d, c, e, hist):
    return PILOTA.monta_observacao(d, c, e["twist"], e["tarefa"], e["anterior"], e["caixa"], e["mapa"], hist)


def _blocos(obs: np.ndarray) -> dict[str, np.ndarray]:
    """Fatia o vetor de 1447 pelos blocos de `Contrato.blocos_ator`: bloco com histórico ocupa canais × 5."""
    saida, i = {}, 0
    for nome, canais, _escala, historico in CONTRATO.blocos_ator:
        n = canais * max(historico, 1)
        saida[nome] = obs[i:i + n]
        i += n
    assert i == obs.shape[0], f"os blocos somam {i}, a observação tem {obs.shape[0]}"
    return saida


def _quadro_esperado(d, c, e) -> dict[str, np.ndarray]:
    """Quadro atual de cada bloco com histórico, já escalado pela escala de `Contrato.blocos_ator`."""
    um_hot = np.zeros(len(TAREFAS))
    um_hot[e["tarefa"]] = 1.0
    crus = {
        "giro_base": d.sensordata[ADR_ANG:ADR_ANG + 3],
        "gravidade": np.array([0.0, 0.0, -1.0]),
        "comando": e["twist"],
        "juntas": d.qpos[c["ids_junta_qpos"]] - c["q_default"],
        "vel_juntas": d.qvel[c["ids_junta_qvel"]] - c["qd_default"],
        "acao": e["anterior"],
        "tarefa": um_hot,
        "caixa": e["caixa"],
    }
    return {nome: crus[nome] * escala for nome, _canais, escala, h in CONTRATO.blocos_ator if h > 0}


class TestNomesPrefixados(unittest.TestCase):
    """O modelo compilado prefixa as entidades do robô (`robot/...`). Busca sem prefixo dá −1 (defeito silencioso)."""

    XML = ('<mujoco><worldbody><body name="robot/torso_link">'
           '<site name="robot/mid360" pos="0 0 0.4"/></body></worldbody></mujoco>')

    def test_acha_id_com_o_prefixo_do_robo(self):
        m = mujoco.MjModel.from_xml_string(self.XML)
        self.assertEqual(PILOTA.acha_id(m, mujoco.mjtObj.mjOBJ_SITE, "robot/mid360"), 0)
        self.assertEqual(PILOTA.acha_id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link"), 1)

    def test_nome_sem_prefixo_levanta_erro_claro(self):
        m = mujoco.MjModel.from_xml_string(self.XML)
        self.assertEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "mid360"), -1)
        with self.assertRaisesRegex(AssertionError, "robot/"):
            PILOTA.acha_id(m, mujoco.mjtObj.mjOBJ_SITE, "mid360")


class TestPilota(unittest.TestCase):

    def test_py_compile_do_pilota_e_do_exportador(self):
        with tempfile.TemporaryDirectory() as tmp:  # o .pyc vai para fora do repo
            for caminho in (RAIZ / "pilota_v3.py", RAIZ / "g1_v3" / "exporta_cena.py"):
                with self.subTest(arquivo=caminho.name):
                    py_compile.compile(str(caminho), cfile=os.path.join(tmp, caminho.name + "c"), doraise=True)

    @unittest.skipUnless((RAIZ / "registra_juntas_v3.py").exists(), "registra_juntas_v3.py ainda não existe")
    def test_py_compile_do_registra_juntas(self):
        with tempfile.TemporaryDirectory() as tmp:
            py_compile.compile(str(RAIZ / "registra_juntas_v3.py"),
                               cfile=os.path.join(tmp, "registra_juntas_v3.pyc"), doraise=True)

    def test_base_para_mundo_do_driver_e_o_do_alvos(self):
        # F9 R3: o driver não tem cópia numpy de base → mundo; usa `alvos.base_para_mundo` (carregado por `_AL`).
        # Antes do conserto: a função não existe em `alvos` e o driver ainda define `_base_para_mundo`.
        self.assertFalse(hasattr(PILOTA, "_base_para_mundo"))
        self.assertTrue(hasattr(PILOTA._AL, "base_para_mundo"))

    def test_forma_e_blocos_do_contrato(self):
        d, c = _cena_falsa(1)
        obs, hist = _chama(d, c, _entradas(2), None)
        self.assertEqual(obs.shape, (DIM_ATOR,))
        self.assertEqual(hist.shape, (HISTORICO, DIM_QUADRO))
        blocos = _blocos(obs)
        self.assertEqual([nome for nome, *_ in CONTRATO.blocos_ator], list(blocos))
        for nome, canais, _escala, historico in CONTRATO.blocos_ator:
            self.assertEqual(blocos[nome].shape, (canais * max(historico, 1),), nome)

    def test_hist_none_repete_o_quadro_nas_cinco_linhas(self):
        d, c = _cena_falsa(3)
        obs, _ = _chama(d, c, _entradas(4), None)
        for nome, canais, _escala, historico in CONTRATO.blocos_ator:
            if historico == 0:
                continue
            linhas = _blocos(obs)[nome].reshape(HISTORICO, canais)
            for k in range(1, HISTORICO):
                np.testing.assert_array_equal(linhas[k], linhas[0], err_msg=nome)

    def test_quadro_atual_tem_valores_e_escalas_do_contrato(self):
        d, c = _cena_falsa(5)
        e = _entradas(6)
        obs, _ = _chama(d, c, e, None)
        esperado = _quadro_esperado(d, c, e)
        for nome, valor in esperado.items():
            atual = _blocos(obs)[nome].reshape(HISTORICO, -1)[-1]
            np.testing.assert_allclose(atual, valor, rtol=0, atol=1e-12, err_msg=nome)

    def test_segundo_passo_mais_antigo_primeiro(self):
        d1, c = _cena_falsa(7)
        e1 = _entradas(8)
        obs1, hist1 = _chama(d1, c, e1, None)

        d2, _ = _cena_falsa(9)  # outra cena, com o mesmo `c`: o quadro novo é diferente
        e2 = _entradas(10)
        obs2, hist2 = _chama(d2, c, e2, hist1)
        self.assertEqual(hist2.shape, (HISTORICO, DIM_QUADRO))

        novo = _quadro_esperado(d2, c, e2)
        antigo = _quadro_esperado(d1, c, e1)
        for nome, _canais, _escala, historico in CONTRATO.blocos_ator:
            if historico == 0:
                continue
            linhas = _blocos(obs2)[nome].reshape(HISTORICO, -1)
            np.testing.assert_allclose(linhas[-1], novo[nome], rtol=0, atol=1e-12, err_msg=f"{nome}: último")
            np.testing.assert_allclose(linhas[0], antigo[nome], rtol=0, atol=1e-12, err_msg=f"{nome}: primeiro")

    def test_mapa_sai_igual_a_entrada(self):
        d, c = _cena_falsa(11)
        e = _entradas(12)
        obs, _ = _chama(d, c, e, None)
        np.testing.assert_array_equal(_blocos(obs)["mapa"], e["mapa"])

    def test_tarefa_one_hot_no_quadro_atual(self):
        d, c = _cena_falsa(13)
        e = _entradas(14)
        obs, _ = _chama(d, c, e, None)
        quadro_tarefa = _blocos(obs)["tarefa"].reshape(HISTORICO, len(TAREFAS))[-1]
        esperado = np.zeros(len(TAREFAS))
        esperado[e["tarefa"]] = 1.0
        np.testing.assert_array_equal(quadro_tarefa, esperado)


def _carrega_registra():
    """Carrega o `registra_juntas_v3.py` da raiz por caminho (ele importa o `pilota_v3` da raiz)."""
    spec = importlib.util.spec_from_file_location("registra_juntas_v3_teste", RAIZ / "registra_juntas_v3.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


class TestRegistraTerreno(unittest.TestCase):
    """G6: roteiros de terreno e a pose da cena no sub-terreno, núcleos puros do `registra_juntas_v3.py`."""

    COLUNAS = list(ATIVO.terreno.colunas)
    LINHA_MAX = dict(ATIVO.terreno.linha_max_carregar)

    @classmethod
    def setUpClass(cls):
        cls.R = _carrega_registra()

    def test_roteiro_de_terreno_e_um_so(self):
        # M9: `escada`, `rampa` e `obstaculo` eram a mesma string; agora é um só, `terreno`.
        fases = self.R.analisa_roteiro("terreno")
        self.assertEqual([f.tarefa for f in fases], ["ANDAR", "PEGAR", "CARREGAR", "CARREGAR"])
        self.assertEqual((fases[-1].segundos, fases[-1].vx), (8.0, 0.25))
        for antigo in ("escada", "rampa", "obstaculo"):
            self.assertNotIn(antigo, self.R.ROTEIROS)

    def test_resolve_terreno_sem_coluna_so_serve_ao_roteiro_sem_terreno(self):
        self.assertIsNone(self.R.resolve_terreno("c", None, None, 0, self.COLUNAS, self.LINHA_MAX))
        with self.assertRaises(SystemExit):
            self.R.resolve_terreno("terreno", None, None, 0, self.COLUNAS, self.LINHA_MAX)
        with self.assertRaises(SystemExit):
            self.R.resolve_terreno("c", None, 3, 0, self.COLUNAS, self.LINHA_MAX)
        # M7: uma borda sem coluna sairia calada no plano, com a borda pedida só no nome do registro.
        with self.assertRaises(SystemExit):
            self.R.resolve_terreno("c", None, None, 1, self.COLUNAS, self.LINHA_MAX)

    def test_afasta_sobe_a_laje_em_z_como_o_treino(self):
        # M3: o treino sobe a laje (`eventos.laje_fora`); o driver a jogava em +x, de lado com `--borda` ≠ 0.
        m = SimpleNamespace(body_mocapid=np.array([-1, 0]))
        d = SimpleNamespace(mocap_pos=np.zeros((1, 3)))
        with mock.patch.object(self.R.mujoco, "mj_forward"):
            self.R.limpa_a_cena(m, d, {"id_laje": np.int64(1)}, 20.0, laje=True, caixa=False)
        np.testing.assert_allclose(d.mocap_pos[0], [0.0, 0.0, 20.0])

    def test_andar_sobe_a_caixa_em_z_junto_com_a_laje_como_o_estaciona_cena(self):
        # D11: o `eventos.estaciona_cena` do treino põe a caixa APOIADA na laje, então ela sobe `afasta` em z,
        # com o xy e a velocidade zero. O driver a jogava em +x (com o xy original sobra a caixa fora da laje).
        m = SimpleNamespace(body_mocapid=np.array([-1, 0]), body_jntadr=np.array([-1, 0]),
                            jnt_qposadr=np.array([0]), jnt_dofadr=np.array([0]))
        d = SimpleNamespace(mocap_pos=np.zeros((1, 3)), qpos=np.array([0.5, -0.3, 0.8, 1.0, 0.0, 0.0, 0.0]),
                            qvel=np.ones(6))
        with mock.patch.object(self.R.mujoco, "mj_forward"):
            self.R.limpa_a_cena(m, d, {"id_laje": np.int64(1), "id_caixa": np.int64(1)}, 20.0, laje=True, caixa=True)
        np.testing.assert_allclose(d.qpos[:3], [0.5, -0.3, 20.8])
        np.testing.assert_allclose(d.qvel, np.zeros(6))

    def test_driver_nao_tem_copia_da_conta_de_base_e_quaternio(self):
        # R4: o xy e o quatérnio vêm do `alvos` e do `estimador`, os mesmos que o `pilota_v3` carrega.
        # Antes do conserto: `_rz` e `_qmul` existem no driver.
        self.assertFalse(hasattr(self.R, "_rz"))
        self.assertFalse(hasattr(self.R, "_qmul"))

    def test_resolve_terreno_recusa_coluna_sem_plataforma_linha_acima_do_teto_e_borda_fora(self):
        with self.assertRaises(SystemExit):
            self.R.resolve_terreno("c", "irregular", 0, 0, self.COLUNAS, self.LINHA_MAX)
        with self.assertRaises(SystemExit):
            self.R.resolve_terreno("c", "escada", 10, 0, self.COLUNAS, self.LINHA_MAX)
        with self.assertRaises(SystemExit):
            self.R.resolve_terreno("c", "escada", 0, 4, self.COLUNAS, self.LINHA_MAX)
        self.assertEqual(self.R.resolve_terreno("c", "obstaculos", 8, 0, self.COLUNAS, self.LINHA_MAX), 7)

    def test_pose_no_sub_terreno_leva_a_laje_a_0_75_rumo_a_borda(self):
        # Laje no reset a 0,50 m do centro (`prateleira_xy[0]`), z 0,55; sub-terreno em (3, 4, 0,3).
        laje = np.array([0.5, 0.0, 0.55])
        for borda, (dx, dy) in ((0, (1, 0)), (1, (0, 1)), (2, (-1, 0)), (3, (0, -1))):
            pos, _ = self.R.pose_no_sub_terreno(laje, np.array([1.0, 0, 0, 0]), np.zeros(3),
                                                np.array([3.0, 4.0, 0.3]), borda, 0.75, 0.5)
            np.testing.assert_allclose(pos, [3.0 + 0.75 * dx, 4.0 + 0.75 * dy, 0.85], atol=1e-9, err_msg=str(borda))

    def test_pose_no_sub_terreno_gira_o_quaternio_e_mantem_o_robo_em_relacao_a_caixa(self):
        caixa = np.array([0.5, 0.0, 0.55])
        robo = np.array([0.0, 0.0, 0.8])
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        args = (np.zeros(3), np.array([3.0, 4.0, 0.3]), 1, 0.75, 0.5)
        caixa2, q2 = self.R.pose_no_sub_terreno(caixa, q_id, *args)
        robo2, _ = self.R.pose_no_sub_terreno(robo, q_id, *args)
        np.testing.assert_allclose(q2, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], atol=1e-9)
        # A cena inteira gira junta: o vetor robô→caixa gira 90° em z, e o comprimento não muda.
        girado = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(robo2 - caixa2, girado @ (robo - caixa), atol=1e-9)

    def test_pose_no_sub_terreno_compoe_o_quaternio_do_robo_com_o_giro_da_cena(self):
        # Com rotação própria (yaw 0,3) o giro da cena entra à esquerda: yaw total = π/2 + 0,3.
        q_robo = np.array([np.cos(0.15), 0.0, 0.0, np.sin(0.15)])
        _, q = self.R.pose_no_sub_terreno(np.zeros(3), q_robo, np.zeros(3), np.zeros(3), 1, 0.0, 0.0)
        meio = (np.pi / 2 + 0.3) / 2
        np.testing.assert_allclose(q, [np.cos(meio), 0, 0, np.sin(meio)], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
