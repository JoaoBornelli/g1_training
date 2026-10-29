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

import numpy as np

from g1_v3.knobs import TAREFAS, Contrato

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


if __name__ == "__main__":
    unittest.main()
