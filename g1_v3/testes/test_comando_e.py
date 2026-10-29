"""Testes dos núcleos puros da cadeia B de g1_v3/comando.py (E3 do plano; §6.3 "Fase E: núcleos puros (29/10)",
§12.1; S §8.2, §10, §11.3; PD-36, PD-50). Escritos antes do código: o ImportError é esperado até a lane E3.

Convenções: `indice` é long em ALVOS (P0..P4 = 0..4, B1..B4 = 5..8, −1 = nenhum). Estados sintéticos de E = 1, sem
env. O relógio do avanço é float64, para que 25 passos de 0,02 s somem 0,5 s dentro de `_EPS` de alvos.py.
`invariantes` entram como número: 1,0 quando o caso não os mede; o caso do P3 a 40° monta-os com a rampa de
`potencial` (o `portoes_do_alvo` da E4 ainda não existe).
"""
import unittest

import torch

from g1_v3 import alvos, potencial
from g1_v3.comando import caixa_perdida_agora, ok_agora_pegar, sucesso_do_pegar
from g1_v3.knobs import ATIVO

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


if __name__ == "__main__":
    unittest.main()
