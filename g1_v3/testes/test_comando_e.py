"""Testes dos núcleos puros da cadeia B de g1_v3/comando.py (E3 do plano; §6.3 "Fase E: núcleos puros (29/10)",
§12.1; S §8.2, §10, §11.3; PD-36, PD-50). Escritos antes do código: o ImportError é esperado até a lane E3.

Convenções: `indice` é long em ALVOS (P0..P4 = 0..4, B1..B4 = 5..8, −1 = nenhum). Estados sintéticos de E = 1, sem
env. O relógio do avanço é float64, para que 25 passos de 0,02 s somem 0,5 s dentro de `_EPS` de alvos.py.
`invariantes` entram como número: 1,0 quando o caso não os mede; o caso do P3 a 40° monta-os com a rampa de
`potencial` (o `portoes_do_alvo` da E4 ainda não existe).
"""
import types
import unittest
from unittest import mock

import torch

from g1_v3 import alvos, comando, curriculo, eventos, potencial, recompensas
from g1_v3.comando import CadeiaCmd, caixa_perdida_agora, ok_agora_pegar, sucesso_do_pegar
from g1_v3.knobs import ATIVO, Estimador

A = ATIVO.alvos
T = ATIVO.terminacao
P0, P1, P2, P3, P4, B1, B2, B3, B4 = range(9)
DT = ATIVO.contrato.decimacao * ATIVO.contrato.dt_fisica  # 0,02 s, passo da política
INCL_LIM = A.incl_inicio_graus  # 30°: limite de inclinação do P3 (S §8.2)
DELTA_INCL = A.delta_incl_graus  # 15°: tolerância da rampa de inclinação
# tempo de permanência por alvo, na ordem de ALVOS. A §6.2 só dá tempo a P4, B2, B3 e B4; P0–P3 e B1 ficam em 0
TEMPOS = torch.tensor([0, 0, 0, 0, A.tempo_p4, 0, A.tempo_b2, A.tempo_b3, A.tempo_b4], dtype=torch.float64)


def estado(indice, d_palmas=0.2, contatos=(False, False), apoio=True, incl=45.0, d_caixa=0.5, d_pelve=0.5,
           invariantes=1.0):
    # Um passo sintético (E = 1). Por padrão nenhuma condição de alvo vale, para o caso trocar só a sua.
    return dict(indice=torch.tensor([indice], dtype=torch.long),
                d_palmas=torch.full((1, 2), float(d_palmas)),
                contatos=torch.tensor([list(contatos)], dtype=torch.bool),
                apoio=torch.tensor([apoio], dtype=torch.bool),
                incl=torch.tensor([float(incl)]),
                d_caixa=torch.tensor([float(d_caixa)]),
                d_pelve=torch.tensor([float(d_pelve)]),
                invariantes=torch.tensor([float(invariantes)]))


def fase(i):
    # O estado em que só a condição do alvo `i` vale.
    if i in (P0, P1):
        return estado(i, d_palmas=0.01)
    if i == P2:
        return estado(i, contatos=(True, True))
    if i == P3:
        return estado(i, apoio=False, incl=10.0)
    return estado(i, d_caixa=0.01, d_pelve=0.01)  # P4


def ok(s):
    return ok_agora_pegar(s["indice"], s["d_palmas"], s["contatos"], s["apoio"], s["incl"], INCL_LIM,
                          s["d_caixa"], s["d_pelve"], s["invariantes"], A)


def perda(fundo=0.5, v_rel=0.0, fora=False, pegou=False, indice=P3, sobre_laje=True):
    return caixa_perdida_agora(torch.tensor([float(fundo)]), torch.tensor([float(v_rel)]),
                               torch.tensor([fora]), torch.tensor([pegou]),
                               torch.tensor([indice], dtype=torch.long), torch.tensor([sobre_laje]), T).item()


def sucesso(p4, falhou, perdeu):
    return sucesso_do_pegar(torch.tensor([p4]), torch.tensor([falhou]), torch.tensor([perdeu])).item()


# --- E11 (29/10): invólucros da cadeia com env falso (E = 1), sem a cena --------------------------------------

def _env_falso(idx, tarefa=comando.PEGAR):
    # Só os buffers que `_alvos` e `_abre_p0` leem ou escrevem.
    z, b = torch.zeros(1), torch.zeros(1, dtype=torch.bool)
    return types.SimpleNamespace(
        v3_alvo_idx=torch.tensor([idx], dtype=torch.long), v3_tarefa=torch.tensor([tarefa], dtype=torch.long),
        v3_meia=torch.tensor([0.1]), v3_ref=types.SimpleNamespace(pelve_com_caixa=lambda a: torch.zeros(1),
                                                                  pelve_de_pe=0.0),
        v3_p4_fechado=b.clone(), v3_pegou=b.clone(), v3_b4_fechado=b.clone(), v3_perdeu=b.clone(),
        v3_avancou=b.clone(), v3_relogio_alvo=z.clone(), v3_alvo_w=torch.zeros(1, 3), v3_caixa_pega_w=torch.zeros(1, 3),
        v3_ponto_apoio=torch.zeros(1, 3),
        v3_sigma=torch.ones(1), v3_phi=torch.zeros(1),
        v3_odo=types.SimpleNamespace(pos_w=torch.tensor([[1.0, 2.0, 0.0]]), quat_w=torch.tensor([[1.0, 0, 0, 0]])))


def _cadeia_sem_env(env):
    # `CadeiaCmd` sem o `__init__` (que pede a cena): só os atributos que os métodos lidos aqui usam
    obj = CadeiaCmd.__new__(CadeiaCmd)
    obj._env = env
    obj.pega = torch.zeros(1, dtype=torch.bool)
    obj.tempos = torch.tensor(comando._TEMPOS, dtype=torch.float32)
    obj.cfg = types.SimpleNamespace(est=Estimador())
    return obj


def _medidas():
    z = torch.zeros(1)
    return dict(h=z, pelve=z, base_p=torch.zeros(1, 3), base_q=torch.tensor([[1.0, 0, 0, 0]]), caixa_p=torch.zeros(1, 3),
                palmas_w=torch.zeros(1, 2, 3), palmas_c=torch.zeros(1, 2, 3), contatos=torch.ones(1, 2, dtype=torch.bool),
                apoio=torch.ones(1, dtype=torch.bool), incl_graus=z, tronco_graus=z,
                v_caixa=torch.zeros(1, 3), apoio_N=z.clone(), peso_N=torch.ones(1))


class TestCadeiaComEnvFalso(unittest.TestCase):

    def test_troca_de_alvo_poe_o_sigma_do_novo_alvo(self):
        # No avanço de P1 a P2 a cadeia põe o σ do P2 em `v3_sigma` (a renda usa o σ do alvo corrente)
        env = _env_falso(comando.P1)
        env.v3_sigma[:] = 0.15
        obj = _cadeia_sem_env(env)
        obj._sigma = lambda m, indice: torch.full((1,), 0.02)  # o σ do P2 (a conta da distância fica de fora)
        ok_todos = dict(desloc=torch.ones(1), incl=torch.ones(1), maos=torch.ones(1))
        with mock.patch.object(comando, "ok_agora_pegar", return_value=torch.tensor([True])), \
                mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=torch.zeros(1, 3)), \
                mock.patch.object(recompensas, "incl_lim", return_value=torch.tensor([30.0])), \
                mock.patch.object(recompensas, "portoes_do_alvo", return_value=ok_todos):
            obj._alvos(_medidas(), 0.02)
        self.assertEqual(env.v3_alvo_idx.tolist(), [comando.P2])
        self.assertAlmostEqual(env.v3_sigma.item(), 0.02, places=6)

    def test_sem_leitura_valida_o_pega_fica_sem_alvo(self):
        # E11 S5b: sem leitura válida (a caixa nunca vista, idade = idade_max) o PEGAR fica com alvo −1 e a pega não
        # abre; o relógio espera. Com leitura válida (idade 0,5 s), abre P0 com a pega na caixa estimada, no mundo
        env = _env_falso(-1)
        env.v3_estimador = types.SimpleNamespace(saida_b=lambda odo, q: torch.tensor([[0.3, 0, 0.1, 0, 0, 0, 0.1, 2.0]]),
                                                 tem_leitura=lambda: torch.tensor([False]))
        obj = _cadeia_sem_env(env)
        obj.pega[:] = True
        obj._sigma = lambda m, indice: torch.full((1,), 0.02)
        obj._abre_p0(_medidas())
        self.assertEqual(env.v3_alvo_idx.tolist(), [-1])
        self.assertTrue(obj.pega.all().item())
        self.assertEqual(env.v3_caixa_pega_w.tolist(), [[0.0, 0.0, 0.0]])
        env.v3_estimador = types.SimpleNamespace(saida_b=lambda odo, q: torch.tensor([[0.3, 0, 0.1, 0, 0, 0, 0.1, 0.5]]),
                                                 tem_leitura=lambda: torch.tensor([True]))
        obj._abre_p0(_medidas())
        self.assertEqual(env.v3_alvo_idx.tolist(), [comando.P0])
        # a caixa estimada no frame da base (0,3; 0; 0,1), com a base em (1; 2; 0) e rumo 0, vai a (1,3; 2; 0,1)
        self.assertEqual([round(v, 6) for v in env.v3_caixa_pega_w[0].tolist()], [1.3, 2.0, 0.1])
        self.assertEqual(env.v3_phi.tolist(), [-1.0])

    def test_compute_da_o_passo_da_odometria_uma_vez_com_a_pose_da_raiz(self):
        # PD-44: a cadeia dá o `passo` da `v3_odo` uma vez por passo, com a pose da raiz e a deriva. Um commit (fce6a42)
        # tirou essa chamada e a odometria do treino parou; sem ela o mapa não é transportado (registro 30/10, caso 8)
        env = _env_falso(comando.P0, tarefa=comando.ANDAR)
        env.v3_trecho = torch.zeros(1, dtype=torch.long)
        env.v3_aviso = torch.zeros(1, dtype=torch.bool)
        env.v3_fim = torch.zeros(1, dtype=torch.bool)
        env.v3_vz_caixa_ant = torch.zeros(1)
        env.v3_odo = mock.Mock()
        env.v3_cur = types.SimpleNamespace(degrau_global=lambda: 3, restricao=lambda: 1.0)  # H5: a deriva sai do degrau global
        env.scene = {"box": types.SimpleNamespace(data=types.SimpleNamespace(root_link_lin_vel_w=torch.zeros(1, 3)))}
        obj = _cadeia_sem_env(env)
        obj.robot = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=torch.tensor([[1.0, 2.0, 0.5]]),
                                                                     root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]])))
        obj.tab = None  # a tabela só vai ao `avanca_trecho`, que está mockado
        obj.relogio = torch.zeros(1)
        obj.alvo_s = torch.zeros(1)
        obj.pega = torch.zeros(1, dtype=torch.bool)
        obj.bota = torch.zeros(1, dtype=torch.bool)
        ordem = []
        env.v3_odo.passo.side_effect = lambda *a: ordem.append("odo")
        obj._mede = mock.Mock(return_value=_medidas())
        obj._perda = mock.Mock()
        obj._abre_p0 = mock.Mock()
        obj._alvos = mock.Mock()
        obj._detector = mock.Mock()
        obj._grava_chegada = mock.Mock(side_effect=lambda: ordem.append("chegada"))
        nao_avancou = (torch.zeros(1, dtype=torch.long), torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                       torch.zeros(1, dtype=torch.bool))
        with mock.patch.object(comando, "avanca_trecho", return_value=nao_avancou):
            obj.compute(DT)
        env.v3_odo.passo.assert_called_once_with(
            obj.robot.data.root_link_pos_w, obj.robot.data.root_link_quat_w,
            comando.deriva_do_degrau(obj.cfg.est, 3, ATIVO.balanceador.degraus))
        self.assertEqual(ordem, ["odo", "chegada"])  # a odometria anda antes das escritas do fim (grava_chegada)


class TestOkAgoraPegar(unittest.TestCase):

    def test_avanca_de_p0_a_p4_na_ordem_e_p4_pede_25_passos(self):
        # E3, linha "Teste": com estados sintéticos, o núcleo de avanço passa de P0 a P4 na ordem
        idx = torch.tensor([P0], dtype=torch.long)
        rel = torch.zeros(1, dtype=torch.float64)
        visitados, passos_em_p4 = [P0], 0
        for _ in range(60):
            if int(idx) == P4:
                passos_em_p4 += 1
            idx, rel, avancou = alvos.avanca(idx, ok(fase(int(idx))), rel, DT, TEMPOS)
            if avancou.item():
                visitados.append(int(idx))
            if int(idx) == B1:
                break
        self.assertEqual(visitados, [P0, P1, P2, P3, P4, B1])
        self.assertEqual(passos_em_p4, 25)  # tempo_p4 = 0,5 s = 25 passos de 0,02 s

    def test_p3_nao_avanca_com_a_caixa_a_40_graus(self):
        # E3, linha "Teste": não avança no P3 com a caixa a 40°, porque o invariante incl = rampa(40°) = 1/3 < 1
        # (limite de 30°). O invariante é incl · maos, com maos = 1 (duas palmas em contato).
        contatos = torch.tensor([[True, True]])
        inv = potencial.rampa(torch.tensor([40.0]), INCL_LIM, DELTA_INCL) * contatos.sum(-1) / 2
        s = estado(P3, apoio=False, incl=40.0, contatos=(True, True), invariantes=inv.item())
        self.assertFalse(ok(s).item())
        idx, _, avancou = alvos.avanca(s["indice"], ok(s), torch.zeros(1, dtype=torch.float64), DT, TEMPOS)
        self.assertEqual(int(idx), P3)
        self.assertFalse(avancou.item())

    def test_invariante_abaixo_de_1_bloqueia_o_p3_mesmo_com_o_degrau_ok(self):
        # E3, §6.3 e S §8.2: `ok_agora` exige invariantes == 1. Com uma palma fora do contato, maos = 0,5 e o P3
        # trava, embora o degrau (sem apoio, 10° ≤ 30°) esteja certo.
        inv = potencial.rampa(torch.tensor([10.0]), INCL_LIM, DELTA_INCL) * torch.tensor([[True, False]]).sum(-1) / 2
        self.assertFalse(ok(estado(P3, apoio=False, incl=10.0, contatos=(True, False), invariantes=inv.item())).item())
        self.assertTrue(ok(estado(P3, apoio=False, incl=10.0, contatos=(True, True), invariantes=1.0)).item())

    def test_p0_e_p1_exigem_as_duas_palmas_no_alvo(self):
        # E3, linha "Teste" (P0, P1: as duas palmas a ≤ 3 cm): uma palma a 4 cm já trava
        for i in (P0, P1):
            with self.subTest(indice=i):
                s = estado(i, d_palmas=0.01)
                s["d_palmas"] = torch.tensor([[0.01, 0.04]])
                self.assertFalse(ok(s).item())
                self.assertTrue(ok(fase(i)).item())

    def test_p2_exige_os_dois_contatos(self):
        # E3, linha "Teste" (P2: as duas palmas em contato): um contato só trava
        self.assertFalse(ok(estado(P2, contatos=(True, False))).item())
        self.assertTrue(ok(estado(P2, contatos=(True, True))).item())

    def test_p4_exige_caixa_e_pelve_dentro_da_tolerancia(self):
        # E3, linha "Teste" (P4: caixa a ≤ 5 cm e pelve a ≤ 3 cm): cada uma, fora do limite, trava
        self.assertFalse(ok(estado(P4, d_caixa=0.06, d_pelve=0.01)).item())
        self.assertFalse(ok(estado(P4, d_caixa=0.01, d_pelve=0.04)).item())
        self.assertTrue(ok(fase(P4)).item())

    def test_p4_zera_o_relogio_quando_a_condicao_cai(self):
        # E3, linha "Teste" (P4: "por 0,5 s"): se a condição cai no meio, o relógio volta a zero
        idx = torch.tensor([P4], dtype=torch.long)
        rel = torch.zeros(1, dtype=torch.float64)
        for _ in range(10):
            idx, rel, _ = alvos.avanca(idx, ok(fase(P4)), rel, DT, TEMPOS)
        idx, rel, avancou = alvos.avanca(idx, ok(estado(P4, d_caixa=0.2)), rel, DT, TEMPOS)
        self.assertEqual(rel.item(), 0.0)
        self.assertEqual(int(idx), P4)
        self.assertFalse(avancou.item())

    def test_b1_a_b4_e_nenhum_dao_falso_no_pegar(self):
        # E3, §6.3: B1..B4 e −1 dão False, mesmo com todas as condições de pega verdadeiras
        for i in (B1, B2, B3, B4, -1):
            with self.subTest(indice=i):
                s = estado(i, d_palmas=0.0, contatos=(True, True), apoio=False, incl=0.0, d_caixa=0.0, d_pelve=0.0)
                self.assertFalse(ok(s).item())


class TestCaixaPerdidaAgora(unittest.TestCase):

    def test_fundo_a_1_cm_do_chao_e_caixa_perdida(self):
        # E3, linha "Teste": o fundo a 0,01 m (abaixo da folga de 0,02 m) é caixa perdida
        self.assertTrue(perda(fundo=0.01))

    def test_fundo_exatamente_na_folga_nao_e_perda(self):
        # E3, §6.3: "caiu" é fundo < caixa_folga_chao (estrito): na folga exata, não cai
        self.assertFalse(perda(fundo=T.caixa_folga_chao))

    def test_pegou_com_v_rel_15_fora_do_alvo_e_perda(self):
        # E3, linha "Teste": pegou, com v_rel 1,5 m/s (acima de v_solta) e fora do alvo, é arremesso
        self.assertTrue(perda(pegou=True, v_rel=1.5, fora=True))

    def test_v_rel_exatamente_em_v_solta_nao_e_arremesso(self):
        # E3, §6.3: "v_rel > v_solta" é estrito: em 1,2 m/s não conta
        self.assertFalse(perda(pegou=True, v_rel=T.v_solta, fora=True))

    def test_pegou_dentro_do_alvo_nao_e_arremesso(self):
        # E3, §6.3: arremesso exige fora_do_alvo; dentro do alvo, mesmo rápido, não conta
        self.assertFalse(perda(pegou=True, v_rel=1.5, fora=False))

    def test_sem_pegar_nao_ha_arremesso(self):
        # E3, linha "Teste": sem pegou, a caixa solta rápido e fora do alvo não é perda
        self.assertFalse(perda(pegou=False, v_rel=1.5, fora=True))

    def test_fora_da_laje_nos_alvos_de_pega_e_perda(self):
        # E3, linha "Teste": P1 fora da laje é perda (e P0 e P2 também, pela regra 0 ≤ índice ≤ 2)
        for i in (P0, P1, P2):
            with self.subTest(indice=i):
                self.assertTrue(perda(indice=i, sobre_laje=False))

    def test_p4_fora_da_laje_nao_e_perda(self):
        # E3, linha "Teste": P4 fora da laje não é perda (a regra vale só de P0 a P2)
        self.assertFalse(perda(indice=P4, sobre_laje=False))

    def test_indice_menos_1_nao_e_perda_por_laje(self):
        # E3, §6.6: o índice −1 dá False, nunca NaN nem perda pela laje
        self.assertFalse(perda(indice=-1, sobre_laje=False))


class TestPerdaNoPrimeiroPasso(unittest.TestCase):

    def _perda_no_passo(self, buf):
        # `_perda` com a caixa "no chão" pelo `h_caixa` e a tarefa PEGAR, num env de E = 1 com `episode_length_buf` = buf.
        # A chave `caixa_perdida_vira_andar` está ligada, para que a perda apareça também em `v3_perdeu`.
        env = _env_falso(comando.P1)
        env.episode_length_buf = torch.tensor([buf], dtype=torch.long)
        env.v3_perdeu_agora = torch.zeros(1, dtype=torch.bool)
        env.v3_perdeu = torch.zeros(1, dtype=torch.bool)
        env.v3_phi = torch.zeros(1)
        env.v3_relogio_alvo = torch.zeros(1)
        env.v3_sigma = torch.ones(1)
        obj = _cadeia_sem_env(env)
        obj.cfg = types.SimpleNamespace(est=Estimador(), k=types.SimpleNamespace(caixa_perdida_vira_andar=True))
        obj._publica = mock.Mock()
        obj._sigma = lambda m, indice: torch.ones(1)
        m = _medidas()
        m["h_caixa"] = torch.zeros(1)  # fundo a −0,1 m, abaixo da folga: "caiu"
        m["v_rel"] = torch.zeros(1)
        m["sobre"] = torch.ones(1, dtype=torch.bool)
        obj._perda(m)
        return env

    def test_primeiro_passo_do_episodio_nao_marca_perda_pelo_h_caixa_stale(self):
        # Caso 6: no passo 0 o `h_caixa` é o do fim do episódio anterior (o `sense()` ainda não rodou sobre o estado novo).
        # Sem o portão de `episode_length_buf` > 0, a caixa "no chão" marcaria perda falsa e um PEGAR viraria ANDAR
        env = self._perda_no_passo(buf=0)
        self.assertFalse(env.v3_perdeu_agora.item())
        self.assertFalse(env.v3_perdeu.item())

    def test_a_partir_do_passo_1_a_mesma_medida_e_perda(self):
        # Caso 6, controle: no passo 1 o `sense()` já rodou sobre o episódio novo, e a mesma medida é perda de verdade
        env = self._perda_no_passo(buf=1)
        self.assertTrue(env.v3_perdeu_agora.item())
        self.assertTrue(env.v3_perdeu.item())


class TestSucessoDoPegar(unittest.TestCase):

    def test_queda_no_carregar_conta_como_falha(self):
        # E3, linha "Teste" (PD-50): a queda no CARREGAR (terminação que não é time_out) não conta como sucesso
        self.assertFalse(sucesso(p4=True, falhou=True, perdeu=False))

    def test_caixa_perdida_no_carregar_com_chave_ligada_nao_conta(self):
        # E3, linha "Teste" (PD-50): com a chave ligada a perda não termina; vira `v3_perdeu` e não conta
        self.assertFalse(sucesso(p4=True, falhou=False, perdeu=True))

    def test_perda_com_chave_desligada_entra_como_falha(self):
        # E3, §6.3 (PD-50): com a chave desligada a perda termina o episódio, então entra por `falhou`
        self.assertFalse(sucesso(p4=True, falhou=True, perdeu=False))

    def test_pega_fechada_sem_falha_nem_perda_conta(self):
        # E3, §6.3: P4 fechado, sem terminação e sem perda, é sucesso
        self.assertTrue(sucesso(p4=True, falhou=False, perdeu=False))

    def test_sem_p4_fechado_nao_conta(self):
        # E3, §6.3: sem P4 fechado não há sucesso, mesmo sem falha
        self.assertFalse(sucesso(p4=False, falhou=False, perdeu=False))


# --- F9 (29/10), lane 1: a cadeia com a tabela de `_monta` e um `v3_cur` falso (S2, S3, P3, P5, P6, P7, M6) -------

PEGAR_TRECHO, ANDAR_TRECHO, BOTAR_TRECHO = 2, 1, 7  # índices de `_monta`: fatia B = ANDAR, PEGAR, CARREGAR; C = …, BOTAR


def _cadeia_de_trechos(n=1):
    # `CadeiaCmd` com a tabela real e um env de E = n sem a cena. `v3_cur` só guarda os registros (num, sucessos).
    z = lambda *forma: torch.zeros(n, *forma)  # noqa: E731
    b = lambda: torch.zeros(n, dtype=torch.bool)  # noqa: E731
    cur = types.SimpleNamespace(registros=[], id_celula=lambda tarefa, tipo: 0, niveis_abertos=lambda nome: 0,
                                degrau=lambda nome: 0, restricao=lambda: 1.0)
    cur.registra = lambda celula, num, suc: cur.registros.append((num, suc))
    env = types.SimpleNamespace(
        device="cpu", v3_cur=cur, v3_gerador=None, v3_meia=torch.full((n,), 0.1), scene={}, step_dt=DT,
        v3_fatia=torch.full((n,), comando.FATIA_B, dtype=torch.long),
        v3_tarefa=torch.full((n,), comando.ANDAR, dtype=torch.long), v3_trecho=torch.zeros(n, dtype=torch.long),
        v3_alvo_idx=torch.full((n,), -1, dtype=torch.long), v3_phi=z(), v3_relogio_alvo=z(),
        v3_freio=torch.zeros(n, dtype=torch.long), v3_degrau=torch.zeros(n, dtype=torch.long),
        v3_alvo_w=z(3), v3_caixa_pega_w=z(3), v3_ponto_apoio=z(3), v3_sigma=torch.ones(n),
        v3_vz_caixa_ant=z(), v3_v_chegada=torch.full((n,), -1.0), v3_aviso=b(),
        v3_pegou=b(), v3_perdeu=b(), v3_p4_fechado=b(), v3_b4_fechado=b(), v3_perdeu_agora=b(), v3_avancou=b(),
        v3_fim=b(), v3_t_empurrao=torch.full((n,), float("-inf")),
        v3_ref=types.SimpleNamespace(pelve_com_caixa=lambda a: torch.zeros(n), pelve_de_pe=0.0,
                                     palmas_de_pe_b=torch.zeros(n, 2, 3),
                                     palmas_com_caixa_b=lambda a: torch.zeros(n, 2, 3)),
        termination_manager=types.SimpleNamespace(time_outs=b()),
        v3_odo=types.SimpleNamespace(pos_w=torch.zeros(n, 3), quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)),
        v3_estimador=types.SimpleNamespace(saida_b=lambda odo, q: torch.zeros(n, 8), tem_leitura=lambda: b()))
    obj = _cadeia_sem_env(env)
    obj.pega, obj.bota, obj.rel_fim = b(), b(), z()
    obj.relogio, obj.alvo_s = z(), z()
    obj.tab, obj.faixas = comando._monta(ATIVO.cadeia, "cpu")
    obj.tempos = torch.tensor(comando._TEMPOS, dtype=torch.float32)
    obj._twist = lambda: types.SimpleNamespace(_resample=lambda ids: None, vel_command_b=torch.zeros(n, 3))
    obj._sigma = lambda m, indice: torch.ones(n)
    obj._mede = lambda: None  # sem a cena; o reset não deve chamá-lo (M6), e o teste que confere isso o troca por um Mock
    return obj


def _abre(obj, trecho, m=None, ids=(0,)):
    # Abre o `trecho` nos `ids` como o `compute` (com `m`) ou o reset (sem `m`)
    ids = torch.tensor(ids, dtype=torch.long)
    obj._abre_trecho(ids, torch.full((len(ids),), trecho, dtype=torch.long), m)


def _posiciona(n=1):
    return mock.patch.object(eventos, "posiciona_destino", return_value=torch.zeros(n, 3))


def _alvos_botar(obj, idx, m, ok=True):
    # `_alvos` com o `ok_agora_botar` forçado e as recompensas de fora mockadas: o teste é da cadeia
    env = obj._env
    env.v3_alvo_idx[:] = idx
    env.v3_tarefa[:] = comando.BOTAR
    with mock.patch.object(comando, "ok_agora_botar", return_value=torch.tensor([ok])), \
            mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=torch.zeros(1, 3)), \
            mock.patch.object(recompensas, "incl_lim", return_value=torch.tensor([30.0])), \
            mock.patch.object(recompensas, "portoes_do_alvo",
                              return_value=dict(desloc=torch.ones(1), incl=torch.ones(1), maos=torch.ones(1))):
        obj._alvos(m, DT)


class TestCadeiaBotarEnvFalso(unittest.TestCase):

    def test_ponto_de_apoio_nasce_no_fecho_do_b2_e_nao_no_primeiro_apoio(self):
        # F9 S2/P1: o ponto de apoio é a caixa no fecho do B2 ("depois de apoiada", S §8.2). Um primeiro contato no B1
        # não grava nada; o B2 que fecha grava a caixa desse passo
        obj = _cadeia_de_trechos()
        m = _medidas()
        m["caixa_p"] = torch.tensor([[0.02, 0.0, 0.5]])
        _alvos_botar(obj, comando.B1, m, ok=False)
        self.assertEqual(obj._env.v3_ponto_apoio.tolist(), [[0.0, 0.0, 0.0]])
        obj._env.v3_relogio_alvo[:] = A.tempo_b2 - DT  # o B2 fecha neste passo
        m["caixa_p"] = torch.tensor([[0.03, 0.0, 0.5]])
        _alvos_botar(obj, comando.B2, m, ok=True)
        self.assertEqual(obj._env.v3_alvo_idx.tolist(), [comando.B3])
        self.assertEqual([round(v, 6) for v in obj._env.v3_ponto_apoio[0].tolist()], [0.03, 0.0, 0.5])

    def test_afasta_e_zero_com_a_palma_no_contato_e_o_b3_fecha_a_5_cm(self):
        # F9 S3: `afasta` mede da face, a + palma_face. Com a palma encostada dá 0; a 5 cm além (e 1 mm), o B3 fecha
        obj = _cadeia_de_trechos()
        a_face = 0.1 + A.palma_face

        def afasta_e_ok(y):
            obj._env.v3_alvo_idx[:] = comando.B3
            obj._env.v3_tarefa[:] = comando.BOTAR
            m = _medidas()
            m["palmas_c"] = torch.tensor([[[0.0, y, 0.0], [0.0, -y, 0.0]]])
            capturas = []
            real = comando.ok_agora_botar

            def captura(*args):
                out = real(*args)
                capturas.append((args[5].clone(), out.clone()))
                return out

            with mock.patch.object(comando, "ok_agora_botar", side_effect=captura), \
                    mock.patch.object(recompensas, "alvo_da_caixa_w", return_value=torch.zeros(1, 3)), \
                    mock.patch.object(recompensas, "incl_lim", return_value=torch.tensor([30.0])), \
                    mock.patch.object(recompensas, "portoes_do_alvo",
                                      return_value=dict(desloc=torch.ones(1), incl=torch.ones(1), maos=torch.ones(1))):
                obj._alvos(m, DT)
            return capturas[0]

        afasta, ok = afasta_e_ok(a_face)
        self.assertAlmostEqual(afasta.abs().max().item(), 0.0, places=6)
        self.assertFalse(ok.item())
        afasta, ok = afasta_e_ok(a_face + A.b3_afasta + 0.001)
        self.assertAlmostEqual(afasta.min().item(), A.b3_afasta + 0.001, places=6)
        self.assertTrue(ok.item())

    def test_primeiro_toque_grava_a_vz_de_antes_do_pouso_e_o_segundo_nao_muda(self):
        # F9 P3: o primeiro contato do apoio no BOTAR grava −vz do passo anterior (a descida); o segundo não muda
        obj = _cadeia_de_trechos()
        env = obj._env
        toque = {"v": torch.tensor([[True]])}
        env.scene = {"apoio_caixa": types.SimpleNamespace(compute_first_contact=lambda dt: toque["v"])}
        env.v3_tarefa[:] = comando.BOTAR
        env.v3_vz_caixa_ant[:] = -1.2
        obj._grava_chegada()
        self.assertAlmostEqual(env.v3_v_chegada.item(), 1.2, places=6)
        env.v3_vz_caixa_ant[:] = -0.4
        obj._grava_chegada()  # o segundo toque
        self.assertAlmostEqual(env.v3_v_chegada.item(), 1.2, places=6)

    def test_abertura_do_botar_volta_a_v_chegada_menos_1(self):
        # F9 P3: a abertura do BOTAR zera o buffer para o −1 (sem toque)
        obj = _cadeia_de_trechos()
        obj._env.v3_v_chegada[:] = 1.2
        with _posiciona():
            _abre(obj, BOTAR_TRECHO, _medidas())
        self.assertEqual(obj._env.v3_v_chegada.tolist(), [-1.0])

    def test_garante_v3_cria_v_chegada_com_menos_1(self):
        env = types.SimpleNamespace(num_envs=2, device="cpu", common_step_counter=0)
        curriculo.garante_v3(env)
        self.assertEqual(env.v3_v_chegada.tolist(), [-1.0, -1.0])

    def test_aviso_do_detector_e_latch_no_trecho(self):
        # F9 P5/R2: `v3_aviso` é o latch do aviso no trecho. Depois do aviso, a pose que sai não o desfaz
        obj = _cadeia_de_trechos()
        env = obj._env
        env.v3_tarefa[:] = comando.PEGAR
        env.v3_trecho[:] = PEGAR_TRECHO
        env.v3_alvo_idx[:] = comando.P0  # o trecho já tem alvo: não está à espera da caixa
        obj.robot = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=torch.zeros(1, 3),
                                                                     joint_vel=torch.zeros(1, 29)))
        avisos = iter([torch.tensor([True]), torch.tensor([False])])
        with mock.patch.object(comando.alvos, "detecta_fim", side_effect=lambda *a: (next(avisos), torch.zeros(1))), \
                mock.patch.object(comando, "z_pe_min", return_value=torch.zeros(1)):
            obj._detector(_medidas(), DT)
            self.assertTrue(env.v3_aviso.item())
            obj._detector(_medidas(), DT)  # a pose saiu: o latch fica
            self.assertTrue(env.v3_aviso.item())

    def test_aviso_so_entra_no_latch_em_trecho_por_detector_e_abertura_zera(self):
        # F9 P5: num trecho por tempo (ANDAR) o aviso não entra; a abertura do trecho zera o latch
        obj = _cadeia_de_trechos()
        env = obj._env
        env.v3_tarefa[:] = comando.ANDAR
        env.v3_trecho[:] = ANDAR_TRECHO
        obj.robot = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=torch.zeros(1, 3),
                                                                     joint_vel=torch.zeros(1, 29)))
        with mock.patch.object(comando.alvos, "detecta_fim", return_value=(torch.tensor([True]), torch.zeros(1))), \
                mock.patch.object(comando, "z_pe_min", return_value=torch.zeros(1)):
            obj._detector(_medidas(), DT)
        self.assertFalse(env.v3_aviso.item())
        env.v3_aviso[:] = True
        _abre(obj, ANDAR_TRECHO)
        self.assertFalse(env.v3_aviso.item())

    def test_pegar_sem_p0_nao_fica_pendente_nem_registra_no_botar(self):
        # F9 P6: o PEGAR só fica pendente de registro depois que o P0 abre. Sem leitura da caixa, o P0 não abre
        obj = _cadeia_de_trechos()
        env = obj._env
        with _posiciona():
            _abre(obj, PEGAR_TRECHO)
            obj._abre_p0(_medidas())
            self.assertFalse(obj.pega.any())
            _abre(obj, BOTAR_TRECHO, _medidas())
        self.assertEqual(env.v3_cur.registros, [])

    def test_pegar_com_p0_aberto_registra_no_botar(self):
        # F9 P6, controle: com o P0 aberto (leitura válida) o PEGAR fica pendente e registra na abertura do BOTAR
        obj = _cadeia_de_trechos()
        env = obj._env
        env.v3_estimador = types.SimpleNamespace(saida_b=lambda odo, q: torch.tensor([[0.3, 0, 0.1, 0, 0, 0, 0.1, 0.5]]),
                                                 tem_leitura=lambda: torch.tensor([True]))
        env.v3_odo = types.SimpleNamespace(pos_w=torch.tensor([[1.0, 2.0, 0.0]]), quat_w=torch.tensor([[1.0, 0, 0, 0]]))
        with _posiciona():
            _abre(obj, PEGAR_TRECHO)
            obj._abre_p0(_medidas())
            self.assertTrue(obj.pega.all().item())
            _abre(obj, BOTAR_TRECHO, _medidas())
        self.assertEqual(env.v3_cur.registros, [(1, 0)])  # P4 não fechou: registra falha

    def test_botar_pendente_no_reset_conta_sucesso_so_sem_queda(self):
        # F9 P7: o BOTAR com B4 fechado que acaba no `time_out` (reset por tempo) conta sucesso; a queda, falha
        for time_out, esperado in ((True, [(1, 1)]), (False, [(1, 0)])):
            with self.subTest(time_out=time_out):
                obj = _cadeia_de_trechos()
                env = obj._env
                obj.bota[:] = True
                env.v3_b4_fechado[:] = True
                env.termination_manager.time_outs[:] = time_out
                obj._resample(torch.tensor([0]))
                self.assertEqual(env.v3_cur.registros, esperado)

    def test_reset_nao_mede_e_botar_sem_medida_levanta_erro(self):
        # F9 M6: o reset não mede (o BOTAR nunca abre no reset); um BOTAR sem medida é erro claro
        obj = _cadeia_de_trechos()
        obj._mede = mock.Mock(return_value=None)
        obj._resample(torch.tensor([0]))
        obj._mede.assert_not_called()
        with self.assertRaises(ValueError), _posiciona():
            _abre(obj, BOTAR_TRECHO, None)


if __name__ == "__main__":
    unittest.main()
