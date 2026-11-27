"""Testes do núcleo puro escolhe_celula_terreno de g1_v3/eventos.py (D7 do plano, §6.3 "eventos.py (núcleo
puro; 29/10)"; S §11.2, §12; PD-20). Escritos antes do código.

Convenções: fatia é índice em FATIAS (0 ANDAR, 1 B, 2 C); a tabela de colunas e os tetos de CARREGAR
vêm de `ATIVO.terreno` (`colunas`, `linha_max_carregar` por nome). O gerador é fixo (semente 0) e E = 20 000
amostras: com esse tamanho, cada coluna recebe ~2 000 sorteios, o bastante para que o máximo de uma
faixa uniforme chegue ao teto com folga. Os casos seguem a linha "Teste" da D7.
"""
import math
import types
import unittest

import torch

from g1_v3 import balanceador, eventos
from g1_v3.eventos import escolhe_celula_terreno
from g1_v3.knobs import ATIVO

T = ATIVO.terreno
NOMES = list(T.colunas)
IRREGULAR = NOMES.index("irregular")  # coluna 6 (rugosidade aleatória, sem plataforma, só ANDAR)
LIMITE = {NOMES.index(nome): linha for nome, linha in T.linha_max_carregar}  # coluna -> linha máxima CARREGAR
ANDAR, B, C = 0, 1, 2
E = 20_000
NIVEL_MAX = 12  # linhas 0..12 do gerador (Terreno.linhas = 13, PD-20)


def sorteia(fatia, terreno, nivel, seed=0, linha_andar=None):
    # `nivel` é o do CARREGAR; `linha_andar` (por coluna) é a do ANDAR (27/11, T19); sem ela, todas as colunas em `nivel`
    g = torch.Generator().manual_seed(seed)
    f = torch.full((E,), fatia, dtype=torch.long)
    t = torch.full((E,), terreno, dtype=torch.bool)
    n = torch.full((E,), nivel, dtype=torch.long)
    la = torch.full((len(NOMES),), float(nivel), dtype=torch.float64) if linha_andar is None else linha_andar
    return escolhe_celula_terreno(f, t, n, la, T, g)


class TestEscolheCelulaTerreno(unittest.TestCase):
    def test_fatia_b_nunca_cai_no_irregular(self):
        # S §11.2: o `irregular` não tem plataforma, então a fatia B (que pede plataforma) nunca o sorteia
        coluna, _ = sorteia(B, True, NIVEL_MAX)
        self.assertFalse((coluna == IRREGULAR).any())
        self.assertTrue(set(coluna.tolist()) <= set(LIMITE))

    def test_carregar_nivel_12_respeita_o_teto_de_cada_coluna(self):
        # PD-20: escadas e rampas ≤ 9 (degrau de 0,15 m), obstáculos ≤ 8, caixas ≤ 7
        coluna, linha = sorteia(B, True, NIVEL_MAX)
        for c, teto in LIMITE.items():
            sel = linha[coluna == c]
            self.assertGreater(sel.numel(), 0, NOMES[c])
            self.assertLessEqual(int(sel.max()), teto, NOMES[c])
            self.assertEqual(int(sel.max()), teto, f"{NOMES[c]} não chegou ao teto {teto}")

    def test_andar_com_linha_12_chega_a_linha_12_e_nao_passa(self):
        # ANDAR de terreno com a linha da coluna em 12 (o máximo, S §11.2): o teto é `linhas − 1`, mesmo com o arredondamento estocástico
        coluna, linha = sorteia(ANDAR, True, NIVEL_MAX)
        self.assertEqual(int(linha.max()), NIVEL_MAX)
        self.assertTrue((linha == NIVEL_MAX).all())
        self.assertGreaterEqual(int(coluna.min()), 1)  # coluna 0 (plano) só no episódio sem terreno
        self.assertLessEqual(int(coluna.max()), len(NOMES) - 1)

    def test_andar_sorteia_so_as_duas_linhas_ao_redor_da_linha_da_coluna_com_media_igual_a_ela(self):
        # 27/11, T19: linha = floor(linha[coluna] + u): com 3,4 só saem 3 e 4, e a média é ~3,4 (arredondamento estocástico).
        # Cada coluna tem a sua linha; o teto é `linhas − 1`
        la = torch.tensor([0.0, 3.4, 0.0, 7.0, 12.9, 0.2, 1.0, 5.5, 0.0], dtype=torch.float64)
        coluna, linha = sorteia(ANDAR, True, 0, linha_andar=la)
        for c in range(1, len(NOMES)):
            sel = linha[coluna == c].double()
            self.assertGreater(sel.numel(), 0, NOMES[c])
            self.assertTrue(set(sel.tolist()) <= {math.floor(float(la[c])), min(math.floor(float(la[c])) + 1, NIVEL_MAX)}, NOMES[c])
            self.assertAlmostEqual(float(sel.mean()), min(float(la[c]), NIVEL_MAX), delta=0.06, msg=NOMES[c])
        self.assertEqual(int(linha[coluna == 4].max()), NIVEL_MAX)  # 12,9: o sorteio de 13 cai no teto

    def test_andar_no_plano_fica_na_coluna_0(self):
        # sem terreno no episódio: coluna 0 e linha 0, mesmo com nível 12 (D7, `flat`)
        coluna, linha = sorteia(ANDAR, False, NIVEL_MAX)
        self.assertTrue((coluna == 0).all())
        self.assertTrue((linha == 0).all())

    def test_fatia_c_sempre_zero(self):
        # a fatia C não tem terreno (S §7): (0, 0) qualquer que seja o nível e o episódio
        coluna, linha = sorteia(C, True, NIVEL_MAX)
        self.assertTrue((coluna == 0).all())
        self.assertTrue((linha == 0).all())


class _Cena:
    # `env.scene["terrain"]` e `env.scene.env_origins`, como a cena do mjlab
    def __init__(self, terreno, origens):
        self._terreno, self.env_origins = terreno, origens

    def __getitem__(self, nome):
        return self._terreno


class TestRegistroDaTravessia(unittest.TestCase):
    """27/11, T19: o `sorteia_fatia` registra a visita e a travessia do episódio de ANDAR em terreno que acabou, antes de sobrescrever
    a fatia e o terreno de `env_ids` (ele roda antes do reset do `twist`: `pose_ult` e `soma_cmd_marcha` ainda são do episódio que acabou)."""

    DT = 0.02

    def _roda(self, fatia, coluna, pedido, caiu, d):
        # um env por linha; o `pedido` (m) é a distância comandada no episódio, `d` (m) a distância final da origem da célula
        n = len(fatia)
        cur = balanceador.Curriculo(ATIVO.balanceador, ATIVO.niveis, ATIVO.comando, T)
        # origens diferentes de zero: a antiga (uma por env) e as da tabela, que o sorteio novo escreve em `env_origins`. Quem lê
        # `env_origins` depois da sobrescrita mede a distância até a célula nova (>= 100 m) e erra a travessia
        origens = torch.stack([10.0 + torch.arange(n), 20.0 - torch.arange(n), torch.zeros(n)], dim=1)
        pose_ult = torch.zeros(n, 3)
        pose_ult[:, :2] = origens[:, :2] + torch.stack([torch.tensor(d), torch.zeros(n)], dim=1)
        tabela = torch.zeros(T.linhas, len(NOMES), 3)
        tabela[..., 0] = 100.0 + 10.0 * torch.arange(len(NOMES))[None, :] + torch.arange(T.linhas)[:, None]
        tabela[..., 1] = -50.0
        terreno = types.SimpleNamespace(terrain_types=torch.tensor(coluna), terrain_levels=torch.zeros(n, dtype=torch.long),
                                        terrain_origins=tabela)
        env = types.SimpleNamespace(
            device="cpu", step_dt=self.DT, v3_cur=cur, v3_gerador=torch.Generator().manual_seed(0),
            v3_fatia=torch.tensor(fatia), v3_t_empurrao=torch.zeros(n), scene=_Cena(terreno, origens.clone()),
            command_manager=types.SimpleNamespace(get_term=lambda nome: types.SimpleNamespace(
                pose_ult=pose_ult, soma_cmd_marcha=torch.tensor(pedido) / self.DT)),
            termination_manager=types.SimpleNamespace(terminated=torch.tensor(caiu)))
        eventos.sorteia_fatia(env, torch.arange(n), T)
        return cur, env

    def test_visita_por_pedido_ou_queda_e_travessia_acima_de_3_m(self):
        # env 0: pedido curto, sem queda (não é visita); env 1: pedido curto, queda (visita, sem travessia); env 2: pedido de 4 m e
        # d = 3,5 m (visita e travessia); env 3: plano (coluna 0: não conta); env 4: fatia B (não conta); env 5: d = 3,5 m sem
        # visita (nem travessia: a taxa não passa de 1)
        cur, env = self._roda(fatia=[0, 0, 0, 0, 1, 0], coluna=[1, 2, 3, 0, 4, 5], pedido=[2.0, 1.0, 4.0, 5.0, 5.0, 2.0],
                              caiu=[False, True, False, False, False, False], d=[0.5, 0.4, 3.5, 4.0, 4.0, 3.5])
        self.assertEqual(cur.vis_ter, [0, 0, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(cur.trav_ter, [0, 0, 0, 1, 0, 0, 0, 0, 0])

    def test_o_registro_lê_o_episodio_antigo_e_nao_o_sorteio_novo(self):
        # o sorteio novo (fatia e coluna de cada env) sobrescreve `v3_fatia` e `terrain_types` depois do registro
        cur, env = self._roda(fatia=[0] * 8, coluna=[1] * 8, pedido=[4.0] * 8, caiu=[False] * 8, d=[3.5] * 8)
        self.assertEqual((cur.vis_ter[1], cur.trav_ter[1]), (8, 8))
        self.assertEqual(sum(cur.vis_ter), 8)

    def test_travessia_pede_d_acima_de_3_m(self):
        # pedido acima de `dist_travessia` é visita; com d = `dist_travessia` (estrito) ainda não é travessia
        cur, env = self._roda(fatia=[0], coluna=[2], pedido=[T.dist_travessia + 0.01], caiu=[False], d=[T.dist_travessia])
        self.assertEqual((cur.vis_ter[2], cur.trav_ter[2]), (1, 0))


if __name__ == "__main__":
    unittest.main()
