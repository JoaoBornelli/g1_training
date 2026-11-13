"""Testes da cena de B no terreno (G1 do plano; §6.3 "Fase G: núcleos puros", §12.3 G1; E §2 "Terreno"; S §11.2, §12).

Escritos antes do código: o AttributeError de `cena_na_plataforma` e os asserts de rumo e de laje são esperados até
o conserto. Os casos de regressão (sem coluna de terreno) passam hoje e têm de continuar passando.

Convenções: env falso com `types.SimpleNamespace`, como os testes vizinhos. `env_origins` é zero, salvo onde o teste
diz o contrário; o gerador é semeado. A plataforma de 2 m tem metade de 1,0 m (`Terreno.plataforma / 2`).
"""
import math
import unittest
from types import SimpleNamespace

import torch

from g1_v3 import estimador, eventos
from g1_v3.knobs import ATIVO

PI2 = math.pi / 2
TOL = 1e-5
N = 20_000  # amostras do núcleo: uma fração de 25% com erro de amostragem menor que 1%
META_PLATAFORMA = ATIVO.terreno.plataforma / 2  # 1,0 m (S §12)
META_LIMITE = 1.05  # laje inteira (0,75 + meia de 0,30) dentro de 1,05 m do centro, no eixo do rumo


def gira(rumo, x, y):
    # R(rumo)·(x, y), um vetor por env
    c, s = torch.cos(rumo), torch.sin(rumo)
    return c * x - s * y, s * x + c * y


class _Cenario(dict):  # o `env.scene`: `scene["robot"]`, `scene["table"]`… e `scene.env_origins`
    env_origins = None


class _Corpo:  # os `write_*` do mjlab: guarda a última pose escrita por env, sem sim
    def __init__(self, n, dim):
        self.pose = torch.zeros(n, dim)

    def write_mocap_pose_to_sim(self, pose, env_ids):
        self.pose[env_ids] = pose.float()

    write_root_link_pose_to_sim = write_mocap_pose_to_sim

    def write_root_link_velocity_to_sim(self, vel, env_ids):
        pass


class _Robo(_Corpo):
    def __init__(self, n):
        super().__init__(n, 7)
        # default_root_state: pos (0, 0, 0,8), quat identidade, velocidades zero (13 números, como o mjlab)
        pose0 = torch.tensor([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.data = SimpleNamespace(default_root_state=pose0.repeat(n, 1))


def env_falso(fatias, colunas=None, recuo=0.6, origens=None):
    """Env falso: `fatias` (0 ANDAR, 1 B, 2 C), `colunas` do terreno (0 = plano) e o recuo do `ref` (escalar ou (E,))."""
    n = len(fatias)
    cena = _Cenario(robot=_Robo(n), table=_Corpo(n, 7), box=_Corpo(n, 7),
                    terrain=SimpleNamespace(terrain_types=torch.tensor(colunas if colunas is not None else [0] * n)))
    cena.env_origins = torch.zeros(n, 3) if origens is None else origens
    cur = SimpleNamespace(nivel=lambda tarefa: 0, nivel_passou=lambda tarefa: -1,
                          subnivel=lambda tarefa: len(ATIVO.niveis.subnivel_extra) - 1)  # 13/11, M3: último subnível
    return SimpleNamespace(
        device="cpu", scene=cena, v3_cur=cur, v3_gerador=torch.Generator().manual_seed(0),
        v3_fatia=torch.tensor(fatias), v3_topo=torch.zeros(n), v3_massa=torch.zeros(n),
        v3_meia=torch.full((n,), 0.1), v3_caixa_xy=torch.zeros(n, 2), v3_rumo_cena=torch.zeros(n),
        v3_ref=SimpleNamespace(recuo=lambda topo, meia: torch.as_tensor(recuo, dtype=torch.float64)
                               .expand(topo.shape[0]).clone()),
    )


class CenaNaPlataforma(unittest.TestCase):
    # Núcleo puro (§6.3 "Fase G"): cena_na_plataforma(terreno_b, k_ter, k_cena, gerador) -> (rumo (E,), desloc (E,2)).

    def nucleo(self, terreno_b, seed=0):
        g = torch.Generator().manual_seed(seed)
        return eventos.cena_na_plataforma(terreno_b, ATIVO.terreno, ATIVO.cena, g)

    def test_com_terreno_rumo_e_multiplo_de_pi_meio(self):
        # Teste (G1): com terreno_b, o rumo é k·π/2 com k em {0, 1, 2, 3}, uma das 4 bordas da plataforma.
        rumo, _ = self.nucleo(torch.ones(N, dtype=torch.bool))
        k = rumo / PI2
        self.assertTrue(torch.allclose(k, k.round(), atol=1e-9))
        self.assertTrue(set(k.round().long().unique().tolist()) <= {0, 1, 2, 3})

    def test_as_quatro_bordas_aparecem_com_frequencia_parecida(self):
        # Teste (G1): com um gerador semeado e N grande, cada uma das 4 bordas sai em ~25% dos episódios.
        rumo, _ = self.nucleo(torch.ones(N, dtype=torch.bool))
        k = (rumo / PI2).round().long()
        for borda in range(4):
            fracao = (k == borda).double().mean().item()
            self.assertGreater(fracao, 0.23, f"borda {borda}")
            self.assertLess(fracao, 0.27, f"borda {borda}")

    def test_laje_a_0_75_do_centro_na_direcao_do_rumo(self):
        # Teste (G1): o centro da laje (desloc + R(rumo)·prateleira_xy) fica a 0,75 m do centro, rumo à borda.
        terreno_b = torch.ones(2000, dtype=torch.bool)
        rumo, desloc = self.nucleo(terreno_b)
        px, py = ATIVO.cena.prateleira_xy
        lx, ly = gira(rumo, torch.full_like(rumo, px), torch.full_like(rumo, py))
        centro_x, centro_y = desloc[:, 0] + lx, desloc[:, 1] + ly
        self.assertTrue(torch.allclose(centro_x.hypot(centro_y), torch.full_like(rumo, 0.75), atol=TOL))
        ex, ey = gira(rumo, torch.ones_like(rumo), torch.zeros_like(rumo))  # direção do rumo
        self.assertTrue(torch.allclose(centro_x, 0.75 * ex, atol=TOL))
        self.assertTrue(torch.allclose(centro_y, 0.75 * ey, atol=TOL))

    def test_sem_terreno_rumo_zero_e_desloc_zero(self):
        # Teste (G1): sem terreno_b, rumo 0 e desloc 0 (o layout plano de hoje, sem mudança).
        rumo, desloc = self.nucleo(torch.zeros(500, dtype=torch.bool))
        self.assertTrue((rumo == 0).all())
        self.assertTrue((desloc == 0).all())

    def test_mistura_so_os_episodios_de_terreno_saem_do_zero(self):
        # Teste (G1): num lote misto, só as linhas com terreno_b ganham rumo e desloc.
        terreno_b = torch.tensor([True, False, True, False])
        rumo, desloc = self.nucleo(terreno_b)
        self.assertTrue((rumo[~terreno_b] == 0).all())
        self.assertTrue((desloc[~terreno_b] == 0).all())
        self.assertTrue(((desloc[terreno_b] != 0).any(dim=-1)).all())


class PosicionaCenaNoTerreno(unittest.TestCase):

    def test_sem_terreno_a_laje_e_a_caixa_ficam_como_hoje(self):
        # Regressão (passa hoje): sem coluna de terreno, a laje fica em origem + prateleira_xy, a caixa em origem +
        # v3_caixa_xy (o jitter de hoje, de um lado em x e entre ±0,18 em y), e o rumo fica 0.
        origens = torch.tensor([[3.0, -2.0, 0.1]]).repeat(8, 1)
        env = env_falso([eventos.B, eventos.C] * 4, origens=origens)
        eventos.posiciona_cena(env, torch.arange(8))
        laje = env.scene["table"].pose
        caixa = env.scene["box"].pose
        prat = torch.tensor(ATIVO.cena.prateleira_xy)
        self.assertTrue(torch.allclose(laje[:, :2], origens[:, :2] + prat, atol=TOL))
        self.assertTrue(torch.allclose(caixa[:, :2], origens[:, :2] + env.v3_caixa_xy, atol=TOL))
        self.assertTrue(torch.allclose(laje[:, 2], origens[:, 2] + env.v3_topo - ATIVO.cena.prateleira_meia_z, atol=TOL))
        self.assertTrue(torch.allclose(caixa[:, 2], origens[:, 2] + env.v3_topo + env.v3_meia, atol=TOL))
        self.assertTrue((env.v3_rumo_cena == 0).all())
        dx = env.v3_caixa_xy[:, 0] - ATIVO.cena.caixa_xy[0]
        dy = env.v3_caixa_xy[:, 1] - ATIVO.cena.caixa_xy[1]
        self.assertTrue(((dx >= -TOL) & (dx <= ATIVO.cena.jitter_x_max[0] + TOL)).all())
        self.assertTrue(((dy >= ATIVO.cena.caixa_jitter_y[0] - TOL) & (dy <= ATIVO.cena.caixa_jitter_y[1] + TOL)).all())

    def test_terreno_b_gira_a_laje_e_a_caixa_vai_sobre_ela(self):
        # Teste (G1): fatia B em coluna de terreno: rumo k·π/2 gravado, laje a 0,75 m do centro no rumo, e a caixa
        # no frame da laje (x local em [−0,18; 0,02], y local em ±0,18), como no plano.
        n = 16
        env = env_falso([eventos.B] * n, colunas=[2] * n)
        eventos.posiciona_cena(env, torch.arange(n))
        rumo = env.v3_rumo_cena.double()
        k = rumo / PI2
        self.assertTrue(torch.allclose(k, k.round(), atol=1e-6))
        self.assertTrue((rumo != 0).any(), "o rumo não foi gravado")
        laje = env.scene["table"].pose[:, :2].double()
        lx, ly = gira(rumo, torch.full_like(rumo, 0.75), torch.zeros_like(rumo))
        self.assertTrue(torch.allclose(laje[:, 0], lx, atol=TOL))
        self.assertTrue(torch.allclose(laje[:, 1], ly, atol=TOL))
        caixa = env.scene["box"].pose[:, :2].double()
        rel = caixa - laje  # vetor laje → caixa, no mundo; no frame da laje (girado por −rumo):
        xl, yl = gira(-rumo, rel[:, 0], rel[:, 1])
        k_cx = ATIVO.cena.caixa_xy[0]
        self.assertTrue(((xl >= k_cx - ATIVO.cena.prateleira_xy[0] - TOL)
                         & (xl <= k_cx + ATIVO.cena.jitter_x_max[0] - ATIVO.cena.prateleira_xy[0] + TOL)).all())
        self.assertTrue(((yl.abs() <= ATIVO.cena.caixa_jitter_y[1] + TOL)).all())

    def test_so_a_fatia_b_no_terreno_ganha_a_plataforma(self):
        # Teste (G1): a fatia C fica no plano mesmo com coluna de terreno (S §12: a C fica toda no plano).
        env = env_falso([eventos.C] * 4, colunas=[3] * 4)
        eventos.posiciona_cena(env, torch.arange(4))
        self.assertTrue((env.v3_rumo_cena == 0).all())
        self.assertTrue(torch.allclose(env.scene["table"].pose[:, :2],
                                       torch.tensor(ATIVO.cena.prateleira_xy).expand(4, 2), atol=TOL))


class BasePegaComRumo(unittest.TestCase):
    FAIXA = dict(ATIVO.cena.reset_base_manipula)

    def test_rumo_pi_meio_robo_a_d_da_caixa_no_eixo_y_e_de_frente(self):
        # Teste (G1, `_base_da_pega` com rumo): rumo π/2, a caixa em (0; 0,6). O robô fica a d da caixa no eixo y,
        # e o yaw é π/2 ± o jitter da faixa (S4).
        env = env_falso([eventos.B], recuo=0.6)
        env.v3_rumo_cena[:] = PI2
        env.v3_caixa_xy[:] = torch.tensor([0.0, 0.6])
        eventos._base_da_pega(env, torch.tensor([0]), self.FAIXA)
        robo = env.scene["robot"].pose
        self.assertAlmostEqual(robo[0, 0].item(), 0.0, delta=TOL)  # sem deslocamento em x
        d = 0.6 - robo[0, 1].item()
        self.assertLessEqual(abs(d - 0.6), ATIVO.alvos.faixa_recuo + TOL)
        yaw = estimador.yaw(robo[:, 3:7].double())[0].item()
        self.assertLessEqual(abs(yaw - PI2), 0.2 + TOL)

    def test_rumo_zero_e_igual_a_hoje(self):
        # Regressão (passa hoje): rumo 0 põe o robô em caixa − (d, 0), com yaw só o jitter.
        env = env_falso([eventos.B], recuo=0.6)
        env.v3_caixa_xy[:] = torch.tensor([0.57, 0.0])
        eventos._base_da_pega(env, torch.tensor([0]), self.FAIXA)
        robo = env.scene["robot"].pose
        d = 0.57 - robo[0, 0].item()
        self.assertLessEqual(abs(d - 0.6), ATIVO.alvos.faixa_recuo + TOL)
        self.assertAlmostEqual(robo[0, 1].item(), 0.0, delta=TOL)
        yaw = estimador.yaw(robo[:, 3:7].double())[0].item()
        self.assertLessEqual(abs(yaw), 0.2 + TOL)

    def test_subnivel_0_do_pegar_tira_o_jitter_do_recuo(self):
        # 13/11, M3: com a laje de 0,75 m (subnível 0) o recuo é exato, sem a faixa_recuo
        n = 200
        env = env_falso([eventos.B] * n, recuo=0.6)
        env.v3_cur.subnivel = lambda tarefa: 0
        env.v3_caixa_xy[:] = torch.tensor([0.57, 0.0])
        eventos._base_da_pega(env, torch.arange(n), self.FAIXA)
        d = 0.57 - env.scene["robot"].pose[:, 0]
        self.assertLessEqual((d - 0.6).abs().max().item(), TOL)


class PlataformaDeTerreno(unittest.TestCase):
    # Propriedade do contrato (item 5 do G1): plataforma de 2 m (metade 1,0 m), laje inteira dentro de 1,05 m do centro
    # no eixo do rumo; robô dentro da plataforma e de frente para a borda. A distância do robô à borda NÃO entra
    # (decisão do dono, 29/10: 0,48–0,95 m, medida na fase I).

    def test_laje_inteira_e_robo_dentro_e_de_frente(self):
        n = 4000
        env = env_falso([eventos.B] * n, colunas=[2] * n, recuo=torch.linspace(0.48, 0.95, n))
        idx = torch.arange(n)
        eventos.posiciona_cena(env, idx)
        eventos._base_da_pega(env, idx, dict(ATIVO.cena.reset_base_manipula))
        rumo = env.v3_rumo_cena.double()
        laje = env.scene["table"].pose[:, :2].double()
        meia = ATIVO.cena.prateleira_meia_xy
        # a laje fica a 0,75 m do centro, no rumo
        self.assertTrue(torch.allclose(laje.norm(dim=-1), torch.full((n,), 0.75, dtype=torch.float64), atol=TOL))
        # os quatro cantos da laje, no eixo do rumo: até 1,05 m do centro
        for sx in (-meia, meia):
            for sy in (-meia, meia):
                eixo_rumo, _ = gira(-rumo, laje[:, 0] + sx, laje[:, 1] + sy)
                self.assertTrue((eixo_rumo.abs() <= META_LIMITE + TOL).all())
        # o robô dentro da plataforma
        robo = env.scene["robot"].pose[:, :2].double()
        self.assertTrue((robo.abs() <= META_PLATAFORMA + TOL).all())
        # o yaw aponta para fora do centro (rumo), ± o jitter; a diferença é dada no intervalo (−π, π]
        yaw = estimador.yaw(env.scene["robot"].pose[:, 3:7].double())
        diff = (yaw - rumo + math.pi) % (2 * math.pi) - math.pi
        self.assertTrue((diff.abs() <= 0.2 + TOL).all())


if __name__ == "__main__":
    unittest.main()
