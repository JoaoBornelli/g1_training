"""Testes dos núcleos puros da cadeia C, o BOTAR, de g1_v3/comando.py (F1 do plano; §6.3 "Fase F: núcleos puros
(29/10)", §12.2 F1; S §8.2, §10, §11.3; PD-39, PD-50). Escritos antes do código: o AttributeError de
`ok_agora_botar` e `sucesso_do_botar` é esperado até a lane F1.

Convenções: `indice` é long em ALVOS (P0..P4 = 0..4, B1..B4 = 5..8, −1 = nenhum). Estados sintéticos de E = 1, sem
env. A cadeia usa `alvos.avanca` com o relógio em float64, para que os passos de 0,02 s somem os tempos dentro de
`_EPS` de alvos.py. `V_PARADA_B3` = 0,05 m/s é o `Alvos.v_parada_b3` da PD-39: o campo ainda não existe no knobs
(a lane F1 cria); o teste usa o número direto para não depender dele.
"""
import unittest

import torch

from g1_v3 import alvos, comando
from g1_v3.knobs import ATIVO

A = ATIVO.alvos
P0, P1, P2, P3, P4, B1, B2, B3, B4 = range(9)
B_IDX = (B1, B2, B3, B4)
DT = ATIVO.contrato.decimacao * ATIVO.contrato.dt_fisica  # 0,02 s, passo da política
V_PARADA_B3 = 0.05  # PD-39 (Alvos.v_parada_b3, a criar na F1)
# tempo de permanência por alvo, na ordem de ALVOS (mesma tabela da E3): P0..P3 e B1 = 0
TEMPOS = torch.tensor([0, 0, 0, 0, A.tempo_p4, 0, A.tempo_b2, A.tempo_b3, A.tempo_b4], dtype=torch.float64)


def ok(indice, d_horiz=0.0, d_caixa=0.0, apoio=10.0, peso=10.0, afasta=(0.20, 0.20), apoiada=True,
       vel=0.0, dpelve=0.0, dtronco=0.0, invariantes=1.0):
    # Um passo sintético (E = 1). Os defaults cumprem as condições de B1..B4 ao mesmo tempo; o caso troca só a sua.
    return comando.ok_agora_botar(
        torch.tensor([indice], dtype=torch.long),
        torch.tensor([float(d_horiz)]),
        torch.tensor([float(d_caixa)]),
        torch.tensor([float(apoio)]),
        torch.tensor([float(peso)]),
        torch.tensor([list(afasta)]),
        torch.tensor([apoiada]),
        torch.tensor([[float(vel), 0.0, 0.0]]),
        torch.tensor([float(dpelve)]),
        torch.tensor([float(dtronco)]),
        torch.tensor([float(invariantes)]),
        A,
    ).item()


class OkAgoraBotar(unittest.TestCase):
    def test_defaults_cumprem_b1_a_b4(self):
        # Teste (F1): os defaults cumprem as quatro condições; ok vale em B1, B2, B3 e B4.
        for i in B_IDX:
            self.assertTrue(ok(i), f"índice {i}")

    def test_b1_horizontal_dentro_e_fora(self):
        # Teste (F1): B1, d_horiz ≤ tol_b1_horizontal (3 cm); o B1 não olha a distância da caixa.
        self.assertTrue(ok(B1, d_horiz=A.tol_b1_horizontal - 0.001))
        self.assertFalse(ok(B1, d_horiz=A.tol_b1_horizontal + 0.001))
        self.assertTrue(ok(B1, d_horiz=0.0, d_caixa=0.5, apoio=0.0))

    def test_b2_caixa_e_apoio_dentro_e_fora(self):
        # Teste (F1): B2, caixa a ≤ 0,10 m do alvo e apoio ≥ 0,8·peso (peso de 10 N no caso).
        self.assertTrue(ok(B2, d_caixa=A.raio_b2 - 0.001, apoio=A.apoio_b2_mg * 10.0))
        self.assertFalse(ok(B2, d_caixa=A.raio_b2 + 0.001))
        self.assertFalse(ok(B2, apoio=A.apoio_b2_mg * 10.0 - 0.5))

    def test_b3_palmas_apoio_e_velocidade_dentro_e_fora(self):
        # Teste (F1): B3, as duas palmas a ≥ 5 cm dos planos das faces, caixa apoiada e ‖v‖ < 0,05 m/s (PD-39).
        self.assertTrue(ok(B3, afasta=(A.b3_afasta + 0.01, A.b3_afasta + 0.01), vel=V_PARADA_B3 - 0.01))
        self.assertFalse(ok(B3, afasta=(A.b3_afasta - 0.01, A.b3_afasta + 0.01)))  # uma palma só
        self.assertFalse(ok(B3, apoiada=False))
        self.assertFalse(ok(B3, vel=V_PARADA_B3 + 0.01))

    def test_b4_pelve_e_tronco_dentro_e_fora(self):
        # Teste (F1): B4, |pelve − pelve_de_pe| ≤ 3 cm e |tronco| ≤ 5°; o sinal não importa.
        self.assertTrue(ok(B4, dpelve=-(A.tol_pelve_b4 - 0.001), dtronco=-(A.tol_tronco_b4_graus - 0.5)))
        self.assertFalse(ok(B4, dpelve=A.tol_pelve_b4 + 0.001))
        self.assertFalse(ok(B4, dtronco=A.tol_tronco_b4_graus + 0.5))

    def test_invariante_menor_que_1_bloqueia_b1_a_b4(self):
        # Teste (F1): `& (invariantes == 1)`: com o produto 0,99 nenhum alvo B vale (S §8.2).
        for i in B_IDX:
            self.assertFalse(ok(i, invariantes=0.99), f"índice {i}")

    def test_p0_a_p4_e_nenhum_dao_false(self):
        # Teste (F1): P0..P4 e −1 devolvem False, mesmo com todas as condições B cumpridas.
        for i in (P0, P1, P2, P3, P4, -1):
            self.assertFalse(ok(i), f"índice {i}")


def _cumpre(i):
    # Só as condições de B_i ficam verdadeiras; as dos outros B ficam falsas (o `ok` do passo da cadeia).
    base = dict(d_horiz=1.0, d_caixa=1.0, apoio=0.0, afasta=(0.0, 0.0), apoiada=False, vel=1.0, dpelve=1.0,
                dtronco=90.0)
    cumpre = {
        B1: dict(d_horiz=0.0),
        B2: dict(d_caixa=0.0, apoio=10.0),
        B3: dict(afasta=(0.2, 0.2), apoiada=True, vel=0.0),
        B4: dict(dpelve=0.0, dtronco=0.0),
    }[i]
    return {**base, **cumpre}


class CadeiaBotar(unittest.TestCase):
    def test_cadeia_b1_a_b4_com_os_tempos_do_knobs(self):
        # Teste (F1): a cadeia B1 → B4 avança com `alvos.avanca` e os tempos de Alvos: B1 em 1 passo (tempo 0),
        # B2 em 15 passos (0,3 s), B3 em 25 (0,5 s), B4 em 25 (0,5 s); depois do B4 o índice vira −1.
        idx = torch.tensor([B1], dtype=torch.long)
        rel = torch.zeros(1, dtype=torch.float64)
        visitados, passos = [], []
        n = 0
        while idx.item() != -1 and n < 500:
            i = idx.item()
            okn = ok(i, **_cumpre(i))
            n += 1
            idx, rel, av = alvos.avanca(idx, torch.tensor([okn]), rel, DT, TEMPOS)
            if av.item():
                visitados.append(i)
                passos.append(n)
                n = 0
        self.assertEqual(visitados, [B1, B2, B3, B4])
        self.assertEqual(passos, [1, 15, 25, 25])
        self.assertEqual(idx.item(), -1)
        self.assertEqual(rel.dtype, torch.float64)

    def test_nao_avanca_no_b2_com_invariante_menor_que_1(self):
        # Teste (F1): no B2, com invariantes 0,99 a cada passo, o índice fica em B2 por 100 passos (sem avanço).
        idx = torch.tensor([B2], dtype=torch.long)
        rel = torch.zeros(1, dtype=torch.float64)
        for _ in range(100):
            okn = ok(B2, invariantes=0.99)
            idx, rel, av = alvos.avanca(idx, torch.tensor([okn]), rel, DT, TEMPOS)
            self.assertFalse(av.item())
        self.assertEqual(idx.item(), B2)
        # controle: com invariante 1,0 o mesmo B2 avança em 15 passos
        idx = torch.tensor([B2], dtype=torch.long)
        rel = torch.zeros(1, dtype=torch.float64)
        for _ in range(15):
            okn = ok(B2, invariantes=1.0)
            idx, rel, av = alvos.avanca(idx, torch.tensor([okn]), rel, DT, TEMPOS)
        self.assertEqual(idx.item(), B3)

    def test_ok_falso_no_meio_zera_o_relogio_do_b2(self):
        # Teste (F1): a permanência em B2 é contínua; um passo com ok falso zera o relógio (§8.2).
        idx = torch.tensor([B2], dtype=torch.long)
        rel = torch.zeros(1, dtype=torch.float64)
        for _ in range(10):
            idx, rel, _ = alvos.avanca(idx, torch.tensor([True]), rel, DT, TEMPOS)
        idx, rel, _ = alvos.avanca(idx, torch.tensor([False]), rel, DT, TEMPOS)
        self.assertEqual(rel.item(), 0.0)
        for _ in range(14):
            idx, rel, _ = alvos.avanca(idx, torch.tensor([True]), rel, DT, TEMPOS)
        self.assertEqual(idx.item(), B2)
        idx, rel, _ = alvos.avanca(idx, torch.tensor([True]), rel, DT, TEMPOS)
        self.assertEqual(idx.item(), B3)


def sucesso(b4, falhou, perdeu):
    return comando.sucesso_do_botar(torch.tensor([b4]), torch.tensor([falhou]), torch.tensor([perdeu])).item()


class SucessoDoBotar(unittest.TestCase):
    def test_sucesso_so_com_b4_fechado_sem_falha_e_sem_perda(self):
        # Teste (F1): sucesso = B4 fechado & ~falhou & ~perdeu (PD-50).
        self.assertTrue(sucesso(True, False, False))
        self.assertFalse(sucesso(False, False, False))

    def test_queda_no_andar_final_nao_e_sucesso(self):
        # Teste (F1): a queda no ANDAR final (falhou) tira o sucesso mesmo com B4 fechado.
        self.assertFalse(sucesso(True, True, False))

    def test_caixa_perdida_nao_e_sucesso(self):
        # Teste (F1): com a caixa perdida (PD-50), o B4 fechado não vira sucesso.
        self.assertFalse(sucesso(True, False, True))


if __name__ == "__main__":
    unittest.main()
