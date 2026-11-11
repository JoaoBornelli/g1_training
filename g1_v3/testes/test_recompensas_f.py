"""Testes do pouso devagar e do invariante depois do apoio no BOTAR (F4 do plano; §6.3 "Fase F: núcleos puros
(29/10)", §12.2 F4; S §9.1; D3 da Parte 0). Escritos antes do código: a F4 não cria núcleo novo, então estes casos
passam hoje (são as peças que o `progresso` do BOTAR vai usar, com `fator_pouso`, `portoes_do_alvo` e
`phi_do_alvo`, já na fase E). Estes casos são a guarda da F4: não podem quebrar quando ela entrar.

Convenções: tensores sintéticos, E = 1, sem env. `a` é a meia aresta da caixa (E,); `sigma` é o σ do degrau 0,
`Recompensa.sigma_pouso[0]` = 1,0 m/s.
"""
import math
import types
import unittest
from unittest import mock

import torch

from g1_v3 import potencial, recompensas
from g1_v3.knobs import ATIVO, TAREFAS

A = ATIVO.alvos
B2, B3, P3, B4 = 6, 7, 3, 8
R = ATIVO.recompensa
TODAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")


def t(x, dtype=torch.float32):
    return torch.tensor([x], dtype=dtype)


class PousoDevagar(unittest.TestCase):
    def test_pouso_a_0_9_m_s_abaixo_da_altura_da_0_44(self):
        # Teste (F4): pousar a 0,9 m/s com o fundo abaixo de altura_pouso dá fator 0,44 (± 0,01).
        fator = potencial.fator_pouso(t(0.9), t(0.03), R.sigma_pouso[0], R.altura_pouso, R.altura_pouso_inicio)
        self.assertAlmostEqual(fator.item(), 0.44, delta=0.01)


class InvarianteDepoisDoApoio(unittest.TestCase):
    def test_deslocamento_de_3_cm_depois_do_apoio_da_meio(self):
        # Teste (F4): no B3, deslocar 3 cm depois de apoiar leva a rampa do `desloc` a 0,5 (rampa(0,03; 0,02; 0,02)).
        indice = t(B3, torch.long)
        zeros = torch.zeros(1)
        portoes = recompensas.portoes_do_alvo(
            indice, zeros, t(0.03), zeros, t(30.0), torch.ones(1), zeros, zeros, zeros,
            torch.tensor([True]), t(0.1), A)
        self.assertAlmostEqual(portoes["desloc"].item(), 0.5, delta=1e-5)


class PhiDoB2(unittest.TestCase):
    def test_phi_no_b2_com_fator_0_44_e_0_44_vezes_o_nucleo(self):
        # Teste (F4; núcleo tanh desde 10/11): phi_do_alvo no B2 com o fator 0,44 dá 0,44 × (1 − tanh(d / 2σ))
        # (portões todos 1 com as condições B2).
        indice = t(B2, torch.long)
        zeros = torch.zeros(1)
        portoes = recompensas.portoes_do_alvo(
            indice, zeros, zeros, zeros, t(30.0), torch.ones(1), zeros, zeros, zeros,
            torch.tensor([True]), t(0.1), A)
        d, sigma = t(0.2), t(0.5)
        phi = recompensas.phi_do_alvo(indice, d, sigma, t(0.44), portoes)
        self.assertAlmostEqual(phi.item(), 0.44 * (1 - math.tanh(0.2)), delta=1e-5)  # 0,2 / (2 × 0,5)

    def test_phi_tanh_no_alvo_em_sigma_e_longe(self):
        # Teste (10/11): Φ = 1 − tanh(d / 2σ) em todo alvo, B4 incluído. No alvo vale 1; em d = σ, 1 − tanh(0,5) =
        # 0,53788; a 4σ, 1 − tanh(2) = 0,03597 (a gaussiana já dava ~1e-7 ali: era o canal morto).
        zeros = torch.zeros(1)
        for alvo, d, sigma, esperado in (
                (0, 0.0, 0.37, 1.0),
                (0, 0.37, 0.37, 0.53788),
                (1, 0.56, 0.14, 0.03597),
                (B4, 2.0, 2.0, 0.53788)):
            indice = t(alvo, torch.long)
            portoes = recompensas.portoes_do_alvo(
                indice, zeros, zeros, zeros, t(30.0), torch.ones(1), zeros, zeros, zeros,
                torch.tensor([True]), t(0.1), A)
            phi = recompensas.phi_do_alvo(indice, t(d), t(sigma), t(1.0), portoes)
            self.assertAlmostEqual(phi.item(), esperado, delta=1e-4)

    def test_phi_com_indice_menos_um_e_zero(self):
        # Teste (F4): −1 (nenhum alvo) dá Φ = 0 (PD-36, §6.6: nunca NaN).
        indice = t(-1, torch.long)
        zeros = torch.zeros(1)
        portoes = recompensas.portoes_do_alvo(
            indice, zeros, zeros, zeros, t(30.0), torch.ones(1), zeros, zeros, zeros,
            torch.tensor([True]), t(0.1), A)
        phi = recompensas.phi_do_alvo(indice, t(0.2), t(0.5), t(0.44), portoes)
        self.assertEqual(phi.item(), 0.0)


class F9PortaoDoTronco(unittest.TestCase):
    """F9 P2 / R1: o desvio do tronco que entra no portão é de um lado só, relu(incl − IK) (S §8.2); no B4 vale 1."""

    def _portoes(self, indice, tronco_graus, ik_graus, forca_N=0.0, massa=1.0):
        """Portões que `_phi_env` monta, com o tronco e a referência do tronco dados; o resto do env em repouso.
        `forca_N`: força normal de cada palma nas faces ±y da caixa (sensor `caixa_faces`, 12 fatias)."""
        class Cena(dict):
            env_origins = torch.zeros(1, 3)

        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        robo = types.SimpleNamespace(
            data=types.SimpleNamespace(root_link_pos_w=torch.zeros(1, 3), root_link_quat_w=quat,
                                       site_pos_w=torch.zeros(1, 2, 3)),
            find_sites=lambda nomes: ([0, 1], None))
        caixa = types.SimpleNamespace(data=types.SimpleNamespace(
            root_link_pos_w=torch.zeros(1, 3), root_link_quat_w=quat, root_link_lin_vel_w=torch.zeros(1, 3)))
        normal, achado = torch.zeros(1, 12, 3), torch.zeros(1, 12)
        normal[0, 0, 1], normal[0, 1, 1], achado[0, :2] = 1.0, -1.0, 1.0  # fatia 0: face +y; fatia 1: face −y
        cena = Cena(robot=robo, box=caixa, altura_pelve=types.SimpleNamespace(
            data=types.SimpleNamespace(heights=torch.zeros(1, 1))), caixa_faces=types.SimpleNamespace(
            data=types.SimpleNamespace(normal=normal, force=normal * forca_N, found=achado)))
        env = types.SimpleNamespace(
            device="cpu", scene=cena, v3_meia=t(0.1), v3_topo_destino=t(0.0), v3_degrau=t(0, torch.long),
            v3_caixa_pega_w=torch.zeros(1, 3), v3_perdeu=torch.zeros(1, dtype=torch.bool),
            v3_ponto_apoio=torch.zeros(1, 3), v3_alvo_idx=t(indice, torch.long), v3_sigma=t(0.3),
            v3_ref=None, v3_massa=t(massa), sim=types.SimpleNamespace(mj_model=types.SimpleNamespace(
                opt=types.SimpleNamespace(gravity=(0.0, 0.0, -9.81)))))
        ref_do = torch.zeros(1, 8)
        ref_do[:, 7] = math.radians(ik_graus)
        espiado = {}
        real = recompensas.portoes_do_alvo

        def espia(*args):
            espiado["g"] = real(*args)
            return espiado["g"]

        with mock.patch.object(recompensas, "ref_do_alvo", return_value=ref_do), \
                mock.patch.object(recompensas, "incl_torso", return_value=t(tronco_graus)), \
                mock.patch.object(recompensas, "_pes_e_com",
                                  return_value=(torch.zeros(1, 8, 2), torch.zeros(1, 2), t(True, torch.bool))), \
                mock.patch.object(recompensas, "_recuo_ref", return_value=torch.zeros(1)), \
                mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=torch.zeros(1, 3)), \
                mock.patch.object(recompensas, "distancia_do_alvo", return_value=torch.zeros(1)), \
                mock.patch.object(recompensas, "portoes_do_alvo", side_effect=espia):
            recompensas._phi_env(env, env.v3_alvo_idx, env.v3_sigma)
        return espiado["g"]

    def test_maos_paga_a_firmeza_da_pega_pelo_peso(self):
        # 11/11: maos = tanh(min F / F_pega), F_pega = margem_pega · m·g / (2μ), μ = DR.atrito_pad[0]. Com 1 kg e 20 N
        # por palma: F_pega = 2,5 · 9,81 / 1,2 = 20,44 N → tanh(0,9786) = 0,7526. Com 5 kg a mesma força vale 5× menos
        # no argumento: tanh(0,1957) = 0,1933. Sem força vale 0; fora de P2–B2 vale 1.
        self.assertAlmostEqual(self._portoes(P3, 0.0, 0.0, forca_N=20.0)["maos"].item(), 0.7526, delta=1e-3)
        self.assertAlmostEqual(self._portoes(P3, 0.0, 0.0, forca_N=20.0, massa=5.0)["maos"].item(), 0.1933, delta=1e-3)
        self.assertEqual(self._portoes(P3, 0.0, 0.0)["maos"].item(), 0.0)
        self.assertEqual(self._portoes(0, 0.0, 0.0)["maos"].item(), 1.0)

    def test_tronco_vertical_com_ik_de_30_graus_da_portao_1(self):
        # F9 P2: tronco vertical (0°) com a IK em 30° está do lado livre: o portão vale 1 (hoje, |0 − 30| dá 0)
        self.assertEqual(self._portoes(P3, 0.0, 30.0)["tronco"].item(), 1.0)

    def test_tronco_20_graus_alem_da_ik_da_a_rampa_de_hoje(self):
        # F9 P2: 20° além da IK, com faixa e Δ de 15°: rampa(20) = 1 − (20 − faixa)/Δ
        esperado = 1 - (20 - A.faixa_tronco_graus) / A.delta_tronco_graus
        self.assertAlmostEqual(self._portoes(P3, 20.0, 0.0)["tronco"].item(), esperado, places=5)

    def test_b4_da_portao_1_com_qualquer_tronco(self):
        # F9 P2: no B4 o núcleo já mede o tronco; o portão vale 1 mesmo 40° além da IK
        self.assertEqual(self._portoes(B4, 40.0, 0.0)["tronco"].item(), 1.0)


class F9CercaPelveSemAlvo(unittest.TestCase):
    """F9 S1: com `v3_alvo_idx` −1 (espera do PEGAR, depois do B4) a cerca da pelve vale a pose de pé."""

    def test_indice_menos_um_no_pegar_cobra_a_pose_de_pe(self):
        # Pelve a 0,789 m, pose de pé também a 0,789 m: custo 0. A referência do alvo (pelve 0) daria 0,709.
        ref = types.SimpleNamespace(pelve_de_pe=0.789, pelve_com_caixa=lambda a: torch.zeros(1))
        env = types.SimpleNamespace(
            device="cpu", num_envs=1, v3_cur=types.SimpleNamespace(restricao=lambda: 1.0),
            v3_tarefa=t(TAREFAS.index("PEGAR"), torch.long), v3_alvo_idx=t(-1, torch.long),
            v3_meia=t(0.1), v3_ref=ref,
            scene={"altura_pelve": types.SimpleNamespace(data=types.SimpleNamespace(heights=torch.tensor([[0.789]])))})
        with mock.patch.object(recompensas, "ref_do_alvo", return_value=torch.zeros(1, 8)):
            r = recompensas.cerca_pelve(env, TODAS)
        self.assertAlmostEqual(float(r[0]), 0.0, places=6)


class F9RecuoSolto(unittest.TestCase):
    """F9 M14: no BOTAR com o alvo B4 ou −1 a caixa já está solta, e a cerca do recuo vale zero."""

    def test_botar_com_b4_ou_menos_um_zera_a_cerca_do_recuo(self):
        # Caixa a 0,25 m do centro; o recuo da pega é 0,40. Sem o zero, relu(0,15 − 0,08) = 0,07.
        transporte = torch.tensor([[0.25, 0.0, 0.0]])

        class Referencia:
            def recuo(self, topo, a):
                return torch.full((1,), 0.40)

            def caixa_transporte_b(self, a):
                return transporte

        for indice in (B4, -1):
            with self.subTest(indice=indice):
                env = types.SimpleNamespace(
                    device="cpu",
                    scene={"robot": types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=torch.zeros(1, 3))),
                           "box": types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=transporte))},
                    v3_tarefa=t(TAREFAS.index("BOTAR"), torch.long), v3_alvo_idx=t(indice, torch.long),
                    v3_perdeu=torch.zeros(1, dtype=torch.bool), v3_topo=torch.zeros(1), v3_topo_destino=torch.zeros(1),
                    v3_meia=t(0.1), v3_ref=Referencia())
                r = recompensas.cerca_recuo(env, ("PEGAR", "CARREGAR", "BOTAR"))
                self.assertAlmostEqual(float(r[0]), 0.0, places=6)


class F9AlturaPeComComando(unittest.TestCase):
    """F9 P4: a `altura_pe` vale zero com o comando do twist zero (o ANDAR parado final pagava 1,0 com o pé parado)."""

    def _env(self, cmd):
        pes = types.SimpleNamespace(data=types.SimpleNamespace(heights=torch.full((1, 2), R.alvo_pe)))
        robo = types.SimpleNamespace(data=types.SimpleNamespace(body_link_vel_w=torch.zeros(1, 2, 6)),
                                     find_bodies=lambda nomes: ([0, 1], None))
        return types.SimpleNamespace(
            device="cpu", scene={"altura_pes": pes, "robot": robo},
            command_manager=types.SimpleNamespace(get_command=lambda nome: cmd),
            v3_tarefa=t(TAREFAS.index("ANDAR"), torch.long), v3_perdeu=torch.zeros(1, dtype=torch.bool))

    def test_pes_parados_com_comando_zero_nao_pagam(self):
        r = recompensas.altura_pe(self._env(torch.zeros(1, 3)), ("ANDAR", "CARREGAR"))
        self.assertAlmostEqual(float(r[0]), 0.0, places=6)

    def test_pes_parados_com_comando_de_marcha_pagam_um(self):
        r = recompensas.altura_pe(self._env(torch.tensor([[0.5, 0.0, 0.0]])), ("ANDAR", "CARREGAR"))
        self.assertAlmostEqual(float(r[0]), 1.0, places=6)


class F9RefDoAlvoPublica(unittest.TestCase):
    """F9 S4: a referência do alvo vira pública, `ref_do_alvo`: a fonte única da observação `referencia`."""

    def test_ref_do_alvo_e_publica(self):
        self.assertTrue(callable(getattr(recompensas, "ref_do_alvo", None)))
        self.assertFalse(hasattr(recompensas, "_ref_do_alvo"))


if __name__ == "__main__":
    unittest.main()
