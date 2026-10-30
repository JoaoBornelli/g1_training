"""Testes da DR do refino em `eventos` (H4 do plano; §6.3 "Fase H: núcleos puros (30/10)", §6.5, §6.8; P3, P4, P15 a
P20). Escritos antes do código: o AttributeError de `_dist`, `pd`, `forca_motor`, `solref`, `com_caixa`, `forca_caixa` e
`rfi` é esperado até a lane H4.

Convenções: env falso com `types.SimpleNamespace`, como os testes vizinhos; o gerador é semeado. Os invólucros dos `dr.*`
do mjlab são testados com o `dr` trocado por `mock` (o que importa é o que o invólucro passa). O `forca_caixa` roda sobre
um asset falso que guarda o último wrench escrito por env.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from mjlab.envs.mdp import dr

from g1_v3 import eventos
from g1_v3.knobs import TAREFAS, TAU_MAX

ORIG_PD, ORIG_FORCA, ORIG_COM = dr.pd_gains, dr.effort_limits, dr.body_com_offset  # os originais do mjlab (antes do mock)
ANDAR, CARREGAR = TAREFAS.index("ANDAR"), TAREFAS.index("CARREGAR")
MEIA = torch.tensor([0.07, 0.09, 0.11, 0.13])  # v3_meia dos 4 envs


class _Cur:  # o `env.v3_cur`: só `dr()`
    def __init__(self, d):
        self.d = d

    def dr(self):
        return self.d


def env_falso(dr_=None, seed=0):
    return SimpleNamespace(num_envs=4, device="cpu", v3_gerador=torch.Generator().manual_seed(seed),
                           v3_cur=_Cur(dr_ or {}), v3_meia=MEIA)


class _Asset:  # o `env.scene["box"]`: guarda o último wrench escrito por env e conta as escritas
    def __init__(self, n):
        self.f = torch.zeros(n, 3)
        self.escritas = 0

    def write_external_wrench_to_sim(self, forces, torques, env_ids, body_ids):
        self.escritas += 1
        self.f[env_ids.long()] = forces[:, 0]


class Distribuicao(unittest.TestCase):
    def test_amostras_no_intervalo_e_espalhadas(self):
        lo, hi = torch.tensor(0.9), torch.tensor(1.1)
        a = eventos._dist(env_falso()).sample(lo, hi, (5000,), "cpu")
        self.assertTrue(bool(((a >= 0.9) & (a <= 1.1)).all()))
        self.assertGreater(float(a.std()), 0.05)  # U[0,9; 1,1] tem desvio 0,058

    def test_mesma_semente_mesmo_sorteio_qualquer_torch_seed(self):
        lo, hi = torch.tensor(0.9), torch.tensor(1.1)
        a = eventos._dist(env_falso(seed=3)).sample(lo, hi, (100,), "cpu")
        torch.manual_seed(999)  # o RNG global não entra no sorteio
        b = eventos._dist(env_falso(seed=3)).sample(lo, hi, (100,), "cpu")
        self.assertTrue(torch.equal(a, b))

    def test_dtype_de_lo(self):
        lo, hi = torch.tensor(0.9, dtype=torch.float64), torch.tensor(1.1, dtype=torch.float64)
        self.assertEqual(eventos._dist(env_falso()).sample(lo, hi, (3,), "cpu").dtype, torch.float64)


class Rfi(unittest.TestCase):
    def env(self, s):
        robo = SimpleNamespace(indexing=SimpleNamespace(joint_v_adr=torch.arange(6, 35)))
        sim = SimpleNamespace(data=SimpleNamespace(qfrc_applied=torch.ones(4, 35)))
        return SimpleNamespace(num_envs=4, device="cpu", v3_gerador=torch.Generator().manual_seed(0),
                               v3_cur=_Cur({"rfi": s}), scene={"robot": robo}, sim=sim)

    def test_torque_limitado_pelo_tau_max_e_nao_zero(self):
        env = self.env(0.1)
        eventos.rfi(env, None)
        tau = env.sim.data.qfrc_applied[:, 6:35]
        lim = 0.1 * torch.tensor(TAU_MAX, dtype=torch.float32)
        self.assertTrue(bool((tau.abs() <= lim + 1e-6).all()))
        self.assertGreater(float(tau.abs().sum()), 0.0)

    def test_colunas_da_junta_livre_intactas(self):
        env = self.env(0.1)
        eventos.rfi(env, None)
        self.assertTrue(bool((env.sim.data.qfrc_applied[:, :6] == 1.0).all()))

    def test_escala_zero_escreve_zero(self):
        env = self.env(0.0)
        eventos.rfi(env, None)
        self.assertTrue(bool((env.sim.data.qfrc_applied[:, 6:35] == 0.0).all()))


class Solref(unittest.TestCase):
    def env(self):
        # geoms globais: robô 0–5, laje 6–8, caixa 9–10, terreno e resto 11–15 (fora dos asset_cfg)
        model = SimpleNamespace(geom_solref=torch.tensor([0.02, 1.0]).repeat(4, 16, 1))
        ent = lambda ids: SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.tensor(list(ids), dtype=torch.int)))
        scene = {"robot": ent(range(0, 6)), "table": ent(range(6, 9)), "box": ent(range(9, 11))}
        return SimpleNamespace(num_envs=4, device="cpu", v3_gerador=torch.Generator().manual_seed(0),
                               v3_cur=_Cur({"solref": [0.015, 0.03]}), scene=scene,
                               sim=SimpleNamespace(model=model))

    def test_um_u_por_env_nas_geoms_selecionadas(self):
        env = self.env()
        robo = SimpleNamespace(name="robot", geom_ids=[0, 1, 2, 3])
        laje = SimpleNamespace(name="table", geom_ids=[0, 1, 2])
        caixa = SimpleNamespace(name="box", geom_ids=[0, 1])
        eventos.solref(env, torch.arange(4), asset_cfg_robo=robo, asset_cfg_laje=laje, asset_cfg_caixa=caixa)
        s = env.sim.model.geom_solref[:, :, 0]
        sel = [0, 1, 2, 3, 6, 7, 8, 9, 10]
        for e in range(4):
            vals = s[e, sel]
            self.assertTrue(bool((vals == vals[0]).all()))  # o mesmo u em todas as geoms do env
            self.assertTrue(bool(((vals >= 0.015) & (vals <= 0.03)).all()))
        self.assertGreater(len({float(s[e, 0]) for e in range(4)}), 1)  # u diferente por env

    def test_fora_das_geoms_e_coluna_1_intactas(self):
        env = self.env()
        eventos.solref(env, torch.arange(4), asset_cfg_robo=SimpleNamespace(name="robot", geom_ids=[0, 1, 2, 3]),
                       asset_cfg_laje=SimpleNamespace(name="table", geom_ids=[0, 1, 2]),
                       asset_cfg_caixa=SimpleNamespace(name="box", geom_ids=[0, 1]))
        m = env.sim.model.geom_solref
        self.assertTrue(bool((m[:, [4, 5, 11, 12, 13, 14, 15], 0] == 0.02).all()))
        self.assertTrue(bool((m[:, :, 1] == 1.0).all()))


class InvolucrosDR(unittest.TestCase):
    def test_pd_escala_com_a_faixa_do_curriculo(self):
        env, robo = env_falso({"pd": [0.9, 1.1]}), SimpleNamespace(name="robot")
        with mock.patch.object(dr, "pd_gains") as m:
            eventos.pd(env, torch.arange(4), asset_cfg=robo)
        self.assertEqual(m.call_count, 1)
        kw = m.call_args.kwargs
        self.assertEqual(tuple(kw["kp_range"]), (0.9, 1.1))
        self.assertEqual(tuple(kw["kd_range"]), (0.9, 1.1))
        self.assertEqual(kw["operation"], "scale")
        self.assertIsInstance(kw["distribution"], dr.Distribution)
        self.assertIs(kw["asset_cfg"], robo)
        self.assertTrue(set(ORIG_PD.model_fields) <= set(eventos.pd.model_fields))

    def test_forca_motor_pernas_com_uma_faixa_bracos_com_outra(self):
        env = env_falso({"forca_motor": [0.9, 1.1], "forca_motor_bracos": [0.8, 1.1]})
        pernas = SimpleNamespace(name="robot", actuator_ids=[0, 1, 2, 3, 4, 5, 6])
        bracos = SimpleNamespace(name="robot", actuator_ids=[7, 8])
        with mock.patch.object(dr, "effort_limits") as m:
            eventos.forca_motor(env, torch.arange(4), asset_cfg_pernas=pernas, asset_cfg_bracos=bracos)
        self.assertEqual(m.call_count, 2)
        p, b = m.call_args_list
        self.assertEqual(tuple(p.kwargs["effort_limit_range"]), (0.9, 1.1))
        self.assertIs(p.kwargs["asset_cfg"], pernas)
        self.assertEqual(tuple(b.kwargs["effort_limit_range"]), (0.8, 1.1))
        self.assertIs(b.kwargs["asset_cfg"], bracos)
        for c in (p, b):
            self.assertEqual(c.kwargs["operation"], "scale")
            self.assertIsInstance(c.kwargs["distribution"], dr.Distribution)
        self.assertTrue(set(ORIG_FORCA.model_fields) <= set(eventos.forca_motor.model_fields))

    def test_com_caixa_faixa_em_meia_aresta_do_mundo(self):
        env, ids = env_falso({"com_caixa": [0.2, 0.1]}), torch.tensor([0, 2])
        with mock.patch.object(dr, "body_com_offset") as m:
            eventos.com_caixa(env, ids, asset_cfg=SimpleNamespace(name="box"))
        kw = m.call_args.kwargs
        self.assertEqual(kw["ranges"], {0: (-0.2, 0.2), 1: (-0.2, 0.2), 2: (-0.1, 0.1)})
        self.assertIsInstance(kw["distribution"], dr.Distribution)
        op = kw["operation"]
        self.assertIsInstance(op, dr.Operation)
        self.assertTrue(op.uses_defaults)
        out = op.combine(torch.zeros(2, 1, 3), torch.ones(2, 1, 3))  # base 0 e r = 1: a meia aresta de cada env
        meia = MEIA[ids].to(out.dtype)
        for eixo in range(3):
            self.assertTrue(torch.allclose(out[:, 0, eixo], meia))
        self.assertIn("body_ipos", eventos.com_caixa.model_fields)
        self.assertTrue(set(ORIG_COM.model_fields) <= set(eventos.com_caixa.model_fields))


class ForcaCaixa(unittest.TestCase):
    # os params do cfg: o EventManager passa todos a cada passo (mode="step"), como func(env, None, **params)
    PARAMS = {"asset_cfg": SimpleNamespace(name="box", body_ids=[0]), "cooldown_s": (0.0, 0.0),
              "force_range": (-11.5, 11.5), "torque_range": (0.0, 0.0), "duration_s": (0.1, 0.1)}

    def env(self, dr_, tarefa, pegou, perdeu):
        asset = _Asset(4)
        env = SimpleNamespace(num_envs=4, device="cpu", step_dt=0.02, scene={"box": asset}, v3_cur=_Cur(dr_),
                              v3_tarefa=torch.tensor(tarefa), v3_pegou=torch.tensor(pegou),
                              v3_perdeu=torch.tensor(perdeu))
        return env, asset

    def test_so_o_carregar_com_a_caixa_na_mao_recebe_forca(self):
        # env 0: CARREGAR com a caixa; 1: CARREGAR que perdeu; 2: ANDAR; 3: CARREGAR sem pegar
        env, asset = self.env({"forca_caixa": True}, [CARREGAR, CARREGAR, ANDAR, CARREGAR],
                              [True, True, False, False], [False, True, False, False])
        evento = eventos.forca_caixa(SimpleNamespace(params=self.PARAMS), env)
        evento(env, None, **self.PARAMS)
        self.assertGreater(float(asset.f[0].abs().sum()), 0.0)
        self.assertTrue(bool((asset.f[0].abs() <= 11.5).all()))  # por eixo, ±11,5 N (P3)
        self.assertTrue(bool((asset.f[1:] == 0.0).all()))

    def test_sem_a_trava_nada_e_escrito(self):
        env, asset = self.env({"forca_caixa": False}, [CARREGAR] * 4, [True] * 4, [False] * 4)
        evento = eventos.forca_caixa(SimpleNamespace(params=self.PARAMS), env)
        evento(env, None, **self.PARAMS)
        self.assertEqual(asset.escritas, 0)


if __name__ == "__main__":
    unittest.main()
