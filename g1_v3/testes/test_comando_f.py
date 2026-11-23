"""Testes dos núcleos puros da cadeia C, o BOTAR, de g1_v3/comando.py (F1 do plano; §6.3 "Fase F: núcleos puros
(29/10)", §12.2 F1; S §8.2, §10, §11.3; PD-39, PD-50). Escritos antes do código: o AttributeError de
`ok_agora_botar` e `sucesso_do_botar` é esperado até a lane F1.

Convenções: `indice` é long em ALVOS (P0..P3 = 0..3, B1..B3 = 4..6, −1 = nenhum; 23/11, T14: o B1 e o B2 antigos viraram o
B1, o B3 virou o B2 e o B4 virou o B3). Estados sintéticos de E = 1, sem env. A cadeia usa `alvos.avanca` com o relógio em
float64, para que os passos de 0,02 s somem os tempos dentro de `_EPS` de alvos.py. `V_PARADA` = 0,05 m/s é o
`Alvos.v_parada` da PD-39; o teste usa o número direto para não depender dele.
"""
import dataclasses
import unittest

import torch

from g1_v3 import alvos, balanceador, comando
from g1_v3.knobs import ATIVO

A = ATIVO.alvos
P0, P1, P2, P3, B1, B2, B3 = range(7)
B_IDX = (B1, B2, B3)
DT = ATIVO.contrato.decimacao * ATIVO.contrato.dt_fisica  # 0,02 s, passo da política
V_PARADA = 0.05  # PD-39 (Alvos.v_parada)
# tempo de permanência por alvo, na ordem de ALVOS (mesma tabela da E3): P0 e P1 = 0
TEMPOS = torch.tensor([0, 0, A.tempo_p2, A.tempo_p3, A.tempo_b1, A.tempo_b2, A.tempo_b3], dtype=torch.float64)


def ok(indice, d_caixa=0.0, apoio=10.0, peso=10.0, afasta=(0.20, 0.20), apoiada=True,
       vel=0.0, dpelve=0.0, dtronco=0.0, dpalmas=0.0, invariantes=1.0, k=A):
    # Um passo sintético (E = 1). Os defaults cumprem as condições de B1..B3 ao mesmo tempo; o caso troca só a sua.
    return comando.ok_agora_botar(
        torch.tensor([indice], dtype=torch.long),
        torch.tensor([float(d_caixa)]),
        torch.tensor([float(apoio)]),
        torch.tensor([float(peso)]),
        torch.tensor([list(afasta)]),
        torch.tensor([apoiada]),
        torch.tensor([[float(vel), 0.0, 0.0]]),
        torch.tensor([float(dpelve)]),
        torch.tensor([float(dtronco)]),
        torch.tensor([float(dpalmas)]),
        torch.tensor([float(invariantes)]),
        k,
    ).item()


def efetivo(z):
    # O `Alvos` efetivo da zona (como `CadeiaCmd._alvos`: final + z·(largo − final)) só com as linhas do B1
    return dataclasses.replace(A, **{campo: balanceador.no_degrau(getattr(A, campo), largo, z, 1)
                                     for nome, campo, largo in A.zona if nome == "B1"})


class OkAgoraBotar(unittest.TestCase):
    def test_defaults_cumprem_b1_a_b3(self):
        # Teste (F1): os defaults cumprem as três condições; ok vale em B1, B2 e B3.
        for i in B_IDX:
            self.assertTrue(ok(i), f"índice {i}")

    def test_b1_caixa_e_apoio_dentro_e_fora(self):
        # Teste (F1); 23/11, T14: B1, caixa a ≤ 0,10 m do alvo e apoio ≥ 0,8·peso (peso de 10 N no caso).
        self.assertTrue(ok(B1, d_caixa=A.raio_b1 - 0.001, apoio=A.apoio_b1_mg * 10.0))
        self.assertFalse(ok(B1, d_caixa=A.raio_b1 + 0.001))
        self.assertFalse(ok(B1, apoio=A.apoio_b1_mg * 10.0 - 0.5))

    def test_b1_pede_a_caixa_parada(self):
        # 13/11, A4: com distância e apoio cumpridos, ‖v_caixa‖ 0,10 não fecha o B1; parada, fecha. O ponto de apoio
        # (gravado no fecho do B1) sai então sem a caixa deslizando.
        self.assertFalse(ok(B1, vel=0.10))
        self.assertTrue(ok(B1, vel=0.0))

    def test_b1_a_zona_afrouxa_o_raio_mas_nunca_o_peso(self):
        # 23/11, T14: com o apoio em 0,79 do peso, o B1 não fecha em nenhum z da zona; o raio, sim, afrouxa (0,15 m só
        # fecha com z perto de 1). A lição do P4 antigo: a zona não afrouxa a grandeza física da etapa.
        for z in (0.0, 0.5, 1.0):
            with self.subTest(z=z):
                k = efetivo(z)
                self.assertEqual(k.apoio_b1_mg, A.apoio_b1_mg)
                self.assertFalse(ok(B1, apoio=0.79 * 10.0, k=k))
                self.assertTrue(ok(B1, apoio=0.80 * 10.0, k=k))
        self.assertFalse(ok(B1, d_caixa=0.15, k=efetivo(0.0)))
        self.assertTrue(ok(B1, d_caixa=0.15, k=efetivo(1.0)))

    def test_b2_palmas_apoio_e_velocidade_dentro_e_fora(self):
        # Teste (F1): B2, as duas palmas a ≥ 5 cm dos planos das faces, caixa apoiada e ‖v‖ < 0,05 m/s (PD-39).
        self.assertTrue(ok(B2, afasta=(A.b2_afasta + 0.01, A.b2_afasta + 0.01), vel=V_PARADA - 0.01))
        self.assertFalse(ok(B2, afasta=(A.b2_afasta - 0.01, A.b2_afasta + 0.01)))  # uma palma só
        self.assertFalse(ok(B2, apoiada=False))
        self.assertFalse(ok(B2, vel=V_PARADA + 0.01))

    def test_b3_pelve_e_tronco_dentro_e_fora(self):
        # Teste (F1): B3, |pelve − pelve_de_pe| ≤ 3 cm e |tronco| ≤ 5°; o sinal não importa.
        self.assertTrue(ok(B3, dpelve=-(A.tol_pelve_b3 - 0.001), dtronco=-(A.tol_tronco_b3_graus - 0.5)))
        self.assertFalse(ok(B3, dpelve=A.tol_pelve_b3 + 0.001))
        self.assertFalse(ok(B3, dtronco=A.tol_tronco_b3_graus + 0.5))

    def test_b3_pede_as_maos_ao_lado_do_tronco(self):
        # 13/11, A9: com pelve e tronco dentro, a maior distância palma–pose de pé de 0,06 m (> 5 cm) não fecha o B3; 0,04 fecha.
        self.assertFalse(ok(B3, dpalmas=0.06))
        self.assertTrue(ok(B3, dpalmas=0.04))

    def test_invariante_menor_que_1_bloqueia_b1_a_b3(self):
        # Teste (F1): `& (invariantes == 1)`: com o produto 0,99 nenhum alvo B vale (S §8.2).
        for i in B_IDX:
            self.assertFalse(ok(i, invariantes=0.99), f"índice {i}")

    def test_p0_a_p3_e_nenhum_dao_false(self):
        # Teste (F1): P0..P3 e −1 devolvem False, mesmo com todas as condições B cumpridas.
        for i in (P0, P1, P2, P3, -1):
            self.assertFalse(ok(i), f"índice {i}")


def _cumpre(i):
    # Só as condições de B_i ficam verdadeiras; as dos outros B ficam falsas (o `ok` do passo da cadeia).
    base = dict(d_caixa=1.0, apoio=0.0, afasta=(0.0, 0.0), apoiada=False, vel=1.0, dpelve=1.0, dtronco=90.0)
    cumpre = {
        B1: dict(d_caixa=0.0, apoio=10.0, vel=0.0),  # 13/11, A4: o B1 pede a caixa parada
        B2: dict(afasta=(0.2, 0.2), apoiada=True, vel=0.0),
        B3: dict(dpelve=0.0, dtronco=0.0),
    }[i]
    return {**base, **cumpre}


class CadeiaBotar(unittest.TestCase):
    def test_cadeia_b1_a_b3_com_os_tempos_do_knobs(self):
        # Teste (F1); 23/11, T14: a cadeia B1 → B3 avança com `alvos.avanca` e os tempos de Alvos: B1 em 15 passos (0,3 s),
        # B2 em 25 (0,5 s), B3 em 25 (0,5 s); depois do B3 o índice vira −1.
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
        self.assertEqual(visitados, [B1, B2, B3])
        self.assertEqual(passos, [15, 25, 25])
        self.assertEqual(idx.item(), -1)
        self.assertEqual(rel.dtype, torch.float64)

    def test_nao_avanca_no_b1_com_invariante_menor_que_1(self):
        # Teste (F1): no B1, com invariantes 0,99 a cada passo, o índice fica em B1 por 100 passos (sem avanço).
        idx = torch.tensor([B1], dtype=torch.long)
        rel = torch.zeros(1, dtype=torch.float64)
        for _ in range(100):
            okn = ok(B1, invariantes=0.99)
            idx, rel, av = alvos.avanca(idx, torch.tensor([okn]), rel, DT, TEMPOS)
            self.assertFalse(av.item())
        self.assertEqual(idx.item(), B1)
        # controle: com invariante 1,0 o mesmo B1 avança em 15 passos
        idx = torch.tensor([B1], dtype=torch.long)
        rel = torch.zeros(1, dtype=torch.float64)
        for _ in range(15):
            okn = ok(B1, invariantes=1.0)
            idx, rel, av = alvos.avanca(idx, torch.tensor([okn]), rel, DT, TEMPOS)
        self.assertEqual(idx.item(), B2)

    def test_ok_falso_no_meio_zera_o_relogio_do_b1(self):
        # Teste (F1): a permanência em B1 é contínua; um passo com ok falso zera o relógio (§8.2).
        idx = torch.tensor([B1], dtype=torch.long)
        rel = torch.zeros(1, dtype=torch.float64)
        for _ in range(10):
            idx, rel, _ = alvos.avanca(idx, torch.tensor([True]), rel, DT, TEMPOS)
        idx, rel, _ = alvos.avanca(idx, torch.tensor([False]), rel, DT, TEMPOS)
        self.assertEqual(rel.item(), 0.0)
        for _ in range(14):
            idx, rel, _ = alvos.avanca(idx, torch.tensor([True]), rel, DT, TEMPOS)
        self.assertEqual(idx.item(), B1)
        idx, rel, _ = alvos.avanca(idx, torch.tensor([True]), rel, DT, TEMPOS)
        self.assertEqual(idx.item(), B2)


def sucesso(b3, falhou, perdeu):
    return comando.sucesso_do_botar(torch.tensor([b3]), torch.tensor([falhou]), torch.tensor([perdeu])).item()


class SucessoDoBotar(unittest.TestCase):
    def test_sucesso_so_com_b3_fechado_sem_falha_e_sem_perda(self):
        # Teste (F1): sucesso = B3 fechado & ~falhou & ~perdeu (PD-50).
        self.assertTrue(sucesso(True, False, False))
        self.assertFalse(sucesso(False, False, False))

    def test_queda_no_andar_final_nao_e_sucesso(self):
        # Teste (F1): a queda no ANDAR final (falhou) tira o sucesso mesmo com B3 fechado.
        self.assertFalse(sucesso(True, True, False))

    def test_caixa_perdida_nao_e_sucesso(self):
        # Teste (F1): com a caixa perdida (PD-50), o B3 fechado não vira sucesso.
        self.assertFalse(sucesso(True, False, True))


if __name__ == "__main__":
    unittest.main()
