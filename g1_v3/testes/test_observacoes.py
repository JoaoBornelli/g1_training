"""Testes de g1_v3/observacoes.py (D2 do plano, §6.3, §6.4, §6.5; S §19.1; PD-25, PD-28). Escritos antes do código.

Casos: one-hot da tarefa; ordem dos termos do ator e do crítico (§6.5); atraso por termo (capacidade 3 no cfg,
PD-28); núcleo do mapa (`passo_de_percepcao`, chão plano); guarda de `atualiza_percepcao` (um cálculo por passo,
com `_calcula_percepcao` e `_calcula_caixa` trocados por contadores); `referencia_do_alvo` com alvo −1 (zero) e a referência carregada
por `knobs.caminho`.
"""
import dataclasses
import math
import types
import unittest
from unittest import mock

import torch

from g1_v3 import observacoes
from g1_v3.alvos import Referencias
from g1_v3.percepcao_lidar import FiltroDeElevacao
from g1_v3.knobs import ALVOS, ATIVO, TAREFAS, caminho

ATOR = ["giro_base", "gravidade", "comando", "juntas", "vel_juntas", "acao", "tarefa", "caixa", "mapa"]  # §6.5
CRITICO = ATOR + ["vel_lin_base", "caixa_verdade", "alvo_verdade", "alvo_macro", "referencia",
                  "niveis", "degrau", "massa", "forca_caixa", "tarefa_critico"]  # §6.5, o último é o grupo de D9
COM_ATRASO = ["giro_base", "gravidade", "juntas", "vel_juntas", "mapa"]  # §6.5: atraso 0–1 em cinco termos
K_PERC = ATIVO.percepcao
CPU = "cpu"


def chao_plano(R=None):
    # sensor a 0,8 m do chão, raios para baixo entre 8° e 60°: todos acertam o chão (dist ≤ 5,75 m < 6 m)
    R = K_PERC.raios_por_passo if R is None else R
    az = torch.linspace(0, 2 * math.pi, R)
    el = torch.linspace(math.radians(-60), math.radians(-8), R)
    dirs = torch.stack([torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], -1)
    dist = (0.8 / -torch.sin(el)).unsqueeze(0)  # (1,R)
    return dirs, dist


class ObservacoesTest(unittest.TestCase):

    def test_um_de_seis_one_hot_da_tarefa(self):
        # TAREFAS (knobs): 0 ANDAR, 3 BOTAR; 6 canais (§6.5 `tarefa`)
        env = types.SimpleNamespace(v3_tarefa=torch.tensor([0, 3], dtype=torch.long))
        out = observacoes.um_de_seis(env)
        self.assertEqual(out.shape, (2, 6))
        self.assertEqual(out.tolist(), [[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

    def test_ordem_dos_termos_do_ator_e_do_critico(self):
        grupos = observacoes.grupos(ATIVO)
        self.assertEqual(list(grupos["actor"].terms), ATOR)
        self.assertEqual(list(grupos["critic"].terms), CRITICO)

    def test_sem_percepcao_o_mapa_sai_dos_dois_grupos_e_a_ordem_fica(self):
        # §6.5: com `Percepcao.ligada` False, `mapa` não entra no ator nem no crítico; o resto mantém a ordem
        k = dataclasses.replace(ATIVO, percepcao=dataclasses.replace(ATIVO.percepcao, ligada=False))
        grupos = observacoes.grupos(k)
        self.assertEqual(list(grupos["actor"].terms), [t for t in ATOR if t != "mapa"])
        self.assertEqual(list(grupos["critic"].terms), [t for t in CRITICO if t != "mapa"])

    def test_cinco_termos_com_atraso_tem_capacidade_3_no_cfg(self):
        # PD-28: capacidade 3 desde o build; o evento `atraso` (D7) baixa o teto para 1 no startup
        termos = observacoes.grupos(ATIVO)["actor"].terms
        for nome in COM_ATRASO:
            self.assertEqual(termos[nome].delay_max_lag, 3, nome)

    def test_mapa_de_chao_plano_tem_alturas_zero_onde_ha_confianca(self):
        # núcleo `passo_de_percepcao` (§6.3): (E,882) = 441 alturas, depois 441 confianças
        filtro = FiltroDeElevacao(1, CPU, K_PERC)
        dirs, dist = chao_plano()
        E = 1
        z = torch.zeros(E)
        out = observacoes.passo_de_percepcao(
            filtro, dist, dirs,
            sensor_pos_w=torch.tensor([[0.0, 0.0, 0.8]]),
            sensor_rot_w=torch.eye(3).expand(E, 3, 3),
            base_pos_w=torch.zeros(E, 3), rumo=z, z_pe_min_w=z,
            incl_imu=torch.eye(3).expand(E, 3, 3),
            capsulas_w=torch.tensor([[[[-50.0, 50.0, 0.0], [-50.0, 50.0, 1.0]]]]),  # uma cápsula longe
            raios=torch.tensor([0.05]),
            dxy_b=torch.zeros(E, 2), dpsi=z, z_ref=z, dt=0.02,
            k=K_PERC, gerador=None)
        self.assertEqual(out.shape, (E, 882))
        h, conf = out[:, :441], out[:, 441:]
        vistas = conf > 0
        self.assertTrue(bool(vistas.any()))  # o chão foi visto em alguma célula
        self.assertLess(float(h[vistas].abs().max()), 0.01)

    def test_guarda_calcula_a_percepcao_uma_vez_por_passo(self):
        # `atualiza_percepcao` só chama `_calcula_percepcao` e `_calcula_caixa` uma vez por common_step_counter
        env = types.SimpleNamespace(common_step_counter=5, v3_percepcao_passo=-1,
                                    v3_mapa_obs=None, v3_caixa_obs=None)
        with mock.patch.object(observacoes, "_calcula_percepcao",
                               return_value=torch.zeros(1, 882)) as calc, \
             mock.patch.object(observacoes, "_calcula_caixa",
                               return_value=torch.zeros(1, 11)) as caixa:
            observacoes.atualiza_percepcao(env)
            observacoes.atualiza_percepcao(env)  # mesmo passo
            self.assertEqual(calc.call_count, 1)
            self.assertEqual(caixa.call_count, 1)
            env.common_step_counter = 6
            observacoes.atualiza_percepcao(env)  # passo seguinte
            self.assertEqual(calc.call_count, 2)
            self.assertEqual(caixa.call_count, 2)

    def test_referencia_com_alvo_menos_um_e_zero(self):
        # §6.6: zero (não NaN) quando o alvo é −1; `v3_ref` é o Referencias carregado do npz (§6.4)
        ref = Referencias(str(caminho(ATIVO.contrato.referencias)), CPU)
        indice = torch.tensor([-1, -1], dtype=torch.long)
        out = observacoes.referencia_do_alvo(indice, torch.tensor([0.3, 0.3]),
                                             torch.tensor([0.10, 0.10]), ref, ATIVO.alvos)
        self.assertEqual(out.shape, (2, 8))
        self.assertFalse(bool(torch.isnan(out).any()))
        self.assertEqual(float(out.abs().max()), 0.0)

    def test_referencia_no_b1_usa_o_topo_de_destino(self):
        # F9 S4: a referência do crítico é a do Φ (`recompensas.ref_do_alvo`); no B1 a pelve sai do topo de destino
        ref = Referencias(str(caminho(ATIVO.contrato.referencias)), CPU)
        env = types.SimpleNamespace(v3_alvo_idx=torch.tensor([ALVOS.index("B1")]), v3_topo=torch.tensor([0.30]),
                                    v3_topo_destino=torch.tensor([0.55]), v3_meia=torch.tensor([0.10]), v3_ref=ref)
        do_destino = observacoes.referencia_do_alvo(env.v3_alvo_idx, env.v3_topo_destino, env.v3_meia, ref, ATIVO.alvos)
        do_topo = observacoes.referencia_do_alvo(env.v3_alvo_idx, env.v3_topo, env.v3_meia, ref, ATIVO.alvos)
        self.assertFalse(torch.allclose(do_destino, do_topo))  # o teste discrimina os dois topos
        self.assertTrue(torch.allclose(observacoes.referencia(env), do_destino))

    def _calcula_caixa_com(self, tarefa, alvo_idx, aviso):
        # `_calcula_caixa` com a visibilidade, as cápsulas, o estimador e o alvo no frame trocados por falsos
        robo = mock.MagicMock()
        robo.find_sites.return_value = ([0], None)
        robo.data.site_pos_w = torch.zeros(1, 1, 3)
        robo.data.site_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        robo.data.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        caixa = types.SimpleNamespace(data=types.SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.5]]), root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
        estimador = mock.MagicMock()
        estimador.saida_b.return_value = torch.zeros(1, 8)
        env = types.SimpleNamespace(scene={"robot": robo, "box": caixa}, common_step_counter=3, step_dt=0.02,
                                    v3_meia=torch.tensor([0.1]), v3_odo=None, v3_estimador=estimador,
                                    v3_tarefa=torch.tensor([tarefa]), v3_alvo_idx=torch.tensor([alvo_idx]),
                                    v3_alvo_w=torch.zeros(1, 3), v3_aviso=torch.tensor([aviso]))
        with mock.patch.object(observacoes, "_capsulas_w", return_value=(torch.zeros(1, 33, 2, 3), torch.zeros(33))), \
                mock.patch.object(observacoes, "visivel", return_value=torch.ones(1, dtype=torch.bool)), \
                mock.patch.object(observacoes, "alvo_no_frame_da_base", return_value=torch.ones(1, 3)):
            return observacoes._calcula_caixa(env)[:, 8:11]  # o alvo: os 3 últimos canais da caixa

    def test_botar_sem_aviso_mostra_o_alvo_mesmo_sem_indice(self):
        # F9 R5: no BOTAR o canal do alvo zera com o latch do aviso, e não com o índice −1 (fecho do B4)
        alvo = self._calcula_caixa_com(TAREFAS.index("BOTAR"), -1, False)
        self.assertEqual(alvo.tolist(), [[1.0, 1.0, 1.0]])

    def test_botar_com_aviso_zera_o_alvo(self):
        alvo = self._calcula_caixa_com(TAREFAS.index("BOTAR"), ALVOS.index("B4"), True)
        self.assertEqual(float(alvo.abs().max()), 0.0)

    def test_fora_do_botar_alvo_sem_indice_zera(self):
        # a regra de fora do BOTAR não muda: sem índice (−1), o alvo zera
        alvo = self._calcula_caixa_com(TAREFAS.index("PEGAR"), -1, False)
        self.assertEqual(float(alvo.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
