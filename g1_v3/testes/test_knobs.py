"""Testes do knobs.py (A2 do plano). Escritos antes do código (lane de testes, A6).

Os números vêm do plano (§6.2, A6) e da interface real (I §2.2). Não ajustar a tolerância
para fazer passar: se o valor sai daqui, o contrato com o robô mudou.
"""
import dataclasses
import re
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from g1_v3.knobs import ATIVO, JUNTAS, TAREFAS, Contrato, Knobs, Recompensa, dim_ator, q_de_pe

RAIZ = Path(__file__).resolve().parents[2]


class TestKnobs(unittest.TestCase):

    def test_dim_ator(self):
        # I §2.1: 1 447 números por passo (o contrato do ator no robô real).
        self.assertEqual(dim_ator(), 1447)

    def test_juntas_e_tarefas(self):
        # I §2.2: 29 juntas, na ordem dos motores do SDK.
        self.assertEqual(len(JUNTAS), 29)
        # One-hot: ANDAR é o índice 0 e BOTAR o índice 3 (I §2.2).
        self.assertEqual(TAREFAS[0], "ANDAR")
        self.assertEqual(TAREFAS[3], "BOTAR")

    def test_cada_junta_casa_uma_regex_de_pd_e_uma_de_tau_max(self):
        # Uma junta sem regex, ou com duas, ganha ganho ou torque errado sem aviso.
        # fullmatch: a regex precisa casar o nome inteiro (ex.: waist_roll não casa waist_r).
        pd = Contrato().pd
        tau = Recompensa().tau_max
        for junta in JUNTAS:
            casa_pd = [r for r, _kp, _kd in pd if re.fullmatch(r, junta)]
            casa_tau = [r for r, _v in tau if re.fullmatch(r, junta)]
            self.assertEqual(len(casa_pd), 1, f"{junta} casa {len(casa_pd)} regex de pd")
            self.assertEqual(len(casa_tau), 1, f"{junta} casa {len(casa_tau)} regex de tau_max")

    def test_q_de_pe_forma_e_valores(self):
        q = q_de_pe()
        self.assertEqual(q.shape, (29,))
        # I §2.2: joelho 0,162; cotovelo 0,600; ombros de pé com roll de ±0,200.
        # Tolerância de 1 mrad: a tabela tem 3 casas; o driver usa o valor exato do arquivo.
        for nome in ("left_knee_joint", "right_knee_joint"):
            self.assertAlmostEqual(q[JUNTAS.index(nome)], 0.162, delta=0.001, msg=nome)
        for nome in ("left_elbow_joint", "right_elbow_joint"):
            self.assertAlmostEqual(q[JUNTAS.index(nome)], 0.600, delta=0.001, msg=nome)
        self.assertAlmostEqual(q[JUNTAS.index("left_shoulder_roll_joint")], 0.200, delta=0.001)
        self.assertAlmostEqual(q[JUNTAS.index("right_shoulder_roll_joint")], -0.200, delta=0.001)

    def test_knobs_e_contrato_congelados(self):
        # Um knob mutável deixa um termo mudar o número no meio do treino.
        self.assertTrue(dataclasses.is_dataclass(Knobs))
        self.assertTrue(Knobs.__dataclass_params__.frozen)
        # O nome do primeiro campo do agregador não está na §6; lê-o do próprio dataclass.
        campos = dataclasses.fields(Knobs)
        self.assertGreater(len(campos), 0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            setattr(Knobs(), campos[0].name, None)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            Contrato().escala_acao = 1.0  # §6.2: escala_acao = 0,25

    def test_dr_e_refino(self):
        # §6.2, contrato de 29/10: o DR (§6.8) e o Refino (§6.9) entram no agregador.
        k = Knobs()
        self.assertFalse(k.dr.com_caixa)  # PD-38: dr.body_com_offset corrompeu a heap; nasce desligada
        self.assertEqual(tuple(k.dr.atraso_refino), (0, 3))
        self.assertEqual(k.dr.degrau_atraso_refino, 3)  # PD-41
        self.assertEqual(k.refino.degrau_batente, 1)
        # Cada regra resolve para o par (início, fim) da seção que ela aponta: o número mora num lugar só.
        esperado = {"laje_joelho_N": (50.0, 10.0), "incl_lim": (30.0, 15.0),
                    "chegada": (1.0, 0.25), "sigma_pouso": (1.0, 0.35)}
        achado = {}
        for regra, tarefas, secao, campo in k.refino.regras:
            s = getattr(k, secao)
            par = (tuple(getattr(s, c) for c in campo) if isinstance(campo, tuple)
                   else tuple(getattr(s, campo)))
            achado[regra] = par
            self.assertTrue(set(tarefas) <= set(TAREFAS), regra)
        self.assertEqual(achado, esperado)

    def test_nenhum_none_no_ativo(self):
        # None = "sai da Parte 0": nenhum pode chegar ao treino. Anda tuplas e tuplas de tuplas.
        def achar_none(nome, valor):
            if valor is None:
                return [nome]
            if isinstance(valor, tuple):
                return [n for i, v in enumerate(valor) for n in achar_none(f"{nome}[{i}]", v)]
            return []
        achados = []
        for secao in dataclasses.fields(ATIVO):
            obj = getattr(ATIVO, secao.name)
            for campo in dataclasses.fields(obj):
                achados += achar_none(f"{secao.name}.{campo.name}", getattr(obj, campo.name))
        self.assertEqual(achados, [], f"None no ATIVO: {achados}")

    def test_knobs_nao_importa_torch_nem_mjlab(self):
        # Subprocesso limpo: o sys.modules deste processo já tem o que o unittest carregou.
        # Conta só os módulos que o próprio import de g1_v3.knobs trouxe.
        # mujoco entra junto: a regra do plano §1.5 proíbe mjlab e mujoco nos módulos puros.
        codigo = (
            "import sys\n"
            "antes = set(sys.modules)\n"
            "import g1_v3.knobs\n"
            "novos = set(sys.modules) - antes\n"
            "proibidos = ('torch', 'mjlab', 'mujoco')\n"
            "print(sorted(m for m in novos if m.split('.')[0] in proibidos))\n"
        )
        r = subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
