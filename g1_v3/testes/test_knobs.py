"""Testes do knobs.py (A2 do plano). Escritos antes do código (lane de testes, A6).

Os números vêm do plano (§6.2, A6) e da interface real (I §2.2). Não ajustar a tolerância
para fazer passar: se o valor sai daqui, o contrato com o robô mudou.
"""
import ast
import builtins
import dataclasses
import re
import subprocess
import symtable
import sys
import unittest
from pathlib import Path

import numpy as np

from g1_v3.knobs import (ATIVO, DR, JUNTAS, TAREFAS, TAU_MAX, Contrato, Estimador, Knobs, Recompensa, Refino,
                         dim_ator, q_de_pe)

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

    def test_tolerancias_do_b4_leem_o_detector(self):
        # F9 R3 (I §4.2: são as mesmas): um literal só. Antes do conserto, `Alvos.tol_pelve_b4` é literal (0,03)
        # e o teste falha na checagem do AST; os valores (0,03 m e 5°) não mudam.
        arvore = ast.parse((RAIZ / "g1_v3" / "knobs.py").read_text(encoding="utf-8"))
        defaults = {}
        for no in ast.walk(arvore):
            if isinstance(no, ast.ClassDef) and no.name == "Alvos":
                defaults = {st.target.id: st.value for st in no.body if isinstance(st, ast.AnnAssign)}
        for nome in ("tol_pelve_b4", "tol_tronco_b4_graus"):
            self.assertIsInstance(defaults[nome], ast.Attribute, f"{nome} repete um literal")
        self.assertEqual(ATIVO.alvos.tol_pelve_b4, 0.03)
        self.assertEqual(ATIVO.alvos.tol_tronco_b4_graus, 5.0)
        self.assertEqual(ATIVO.alvos.tol_pelve_b4, ATIVO.detector.tol_pelve)
        self.assertEqual(ATIVO.alvos.tol_tronco_b4_graus, ATIVO.detector.tol_tronco_graus)

    def test_dr_e_refino(self):
        # §6.2, contrato de 29/10: o DR (§6.8) e o Refino (§6.9) entram no agregador.
        k = Knobs()
        self.assertFalse(k.dr.com_caixa)  # PD-38: dr.body_com_offset corrompeu a heap; nasce desligada
        self.assertEqual(tuple(k.dr.atraso_refino), (0, 3))
        self.assertEqual(k.dr.degrau_atraso_refino, 3)  # PD-41
        self.assertEqual(k.refino.degrau_batente, 1)
        # H1 (30/10, P1): as 4 regras do refino moram nas suas seções (o número mora num lugar só).
        self.assertEqual(tuple(k.recompensa.laje_joelho_N), (50.0, 10.0))
        self.assertEqual((k.alvos.incl_inicio_graus, k.alvos.incl_fim_graus), (30.0, 15.0))
        self.assertEqual(tuple(k.terminacao.chegada), (1.0, 0.25))
        self.assertEqual(tuple(k.recompensa.sigma_pouso), (1.0, 0.35))

    def test_tau_max_por_junta_na_ordem_de_juntas(self):
        # §12.4 H1 (P20): TAU_MAX tem 29 valores na ordem de JUNTAS, montados da tabela por regex no import.
        # Joelho e hip_roll 139; hip_pitch, hip_yaw e waist_yaw 88; cintura e tornozelo 50; ombro, cotovelo e
        # wrist_roll 25; wrist_pitch e wrist_yaw 5.
        esperado = (88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50, 88, 50, 50,
                    25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5)
        self.assertEqual(len(TAU_MAX), 29)
        self.assertEqual(tuple(TAU_MAX), tuple(float(v) for v in esperado))

    def test_valores_novos_da_fase_h(self):
        # §12.4 H1 e §6.2 (30/10): custos com o sinal do cfg; a DR da força da caixa; o viés de orientação.
        r = Recompensa()
        self.assertEqual(r.sacudida, -0.05)  # P2
        self.assertEqual(r.aperto, -0.05)  # caso 9 (30/10): forma linear, peso pela escada
        self.assertEqual(r.sacudida_norma, (5.04, 31.56))  # P8, P10
        self.assertEqual(DR().forca_caixa_N, (-11.5, 11.5))  # P3: por eixo
        self.assertLessEqual(11.5 * 3 ** 0.5, 20.0)  # ‖F‖ do impulso por eixo, ≤ 19,9 N
        e = Estimador()
        self.assertEqual(e.ori_graus, ((1.5, 2.5), (5.3, 15.3)))  # P21a
        self.assertEqual(e.ori_leitura_graus, (0.0, 1.0))  # P21b
        self.assertEqual(e.vies_pos, (0.0, 0.02))  # P21c
        self.assertEqual(e.rajada_s, (0.3, 1.0))  # DR §3.5
        self.assertEqual(e.rajada_taxa, (0.0, 0.035))  # P21e
        # Os dois campos saíram na fase H (P21b): `_magnitude` não os lê mais.
        self.assertFalse(hasattr(e, "sigma_ori_graus"))
        self.assertFalse(hasattr(e, "ori_persistente"))

    def test_refino_sem_os_campos_mortos(self):
        # §12.4 H1 (P1): saem Refino.regras, fator_por_degrau e tarefas_fator; fica degrau_batente
        campos = {f.name for f in dataclasses.fields(Refino)}
        self.assertEqual(campos, {"degrau_batente"})

    def test_colunas_do_terreno(self):
        # D1 (§6.2, 29/10): as 9 colunas na ordem do gerador; o eventos lê esta ordem.
        t = Knobs().terreno
        self.assertEqual(t.colunas, ("plano", "escada", "escada_inv", "rampa", "rampa_inv",
                                     "escada_aleatoria", "irregular", "obstaculos", "caixas"))
        self.assertEqual(len(t.colunas), 9)
        # PD-20: toda coluna com plataforma de 2 m é uma coluna do gerador.
        nomes = [n for n, _linha in t.linha_max_carregar]
        self.assertTrue(set(nomes) <= set(t.colunas))
        # plano e irregular não têm plataforma de 2 m: fora do CARREGAR.
        self.assertNotIn("plano", nomes)
        self.assertNotIn("irregular", nomes)
        # O balanceador usa só o máximo dos valores (9).
        self.assertEqual(max(v for _n, v in t.linha_max_carregar), 9)
        # P3 (R1, 30/10): a escada aleatória para na linha 6 (degrau até 0,15 m com caixa; primitive_terrains.py:1061).
        self.assertEqual(dict(t.linha_max_carregar)["escada_aleatoria"], 6)

    def test_fase_g_laje_do_centro_e_incl_max_do_terreno(self):
        # G1 e G3 (§6.2, fase G, 29/10): a laje do centro a 0,75 m da plataforma (S §12); a caixa a ≤ 15° no CARREGAR
        # do terreno (S §11.3). Um número só: a lane G3 lê o mesmo `Comando.incl_max_terreno_graus`.
        self.assertEqual(ATIVO.terreno.laje_do_centro, 0.75)
        self.assertEqual(ATIVO.comando.incl_max_terreno_graus, 15.0)

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


class TestNomesGlobais(unittest.TestCase):
    """B1 (revisão E11): `K` lido em `recompensas.incl_lim` sem existir no módulo derrubava o primeiro reset
    (NameError). Checagem estática com `symtable`, sem importar os módulos: todo nome global lido numa função
    (ou no nível do módulo) tem de ser definido no módulo ou ser builtin."""

    DUNDERS_DO_MODULO = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__",
                         "__builtins__", "__cached__"}

    def test_nenhum_nome_global_indefinido_em_g1_v3(self):
        pacote = RAIZ / "g1_v3"
        achados = []
        for arq in sorted(pacote.rglob("*.py")):
            if set(arq.relative_to(pacote).parts) & {"testes", "parte0"}:
                continue
            modulo = symtable.symtable(arq.read_text(encoding="utf-8"), str(arq), "exec")
            definidos = {s.get_name() for s in modulo.get_symbols()
                         if s.is_assigned() or s.is_imported() or s.is_namespace()} | self.DUNDERS_DO_MODULO

            def visita(tabela):
                for s in tabela.get_symbols():
                    nome = s.get_name()
                    if tabela is not modulo and (s.is_assigned() or not s.is_global()):
                        continue
                    if s.is_referenced() and nome not in definidos and not hasattr(builtins, nome):
                        achados.append(f"{arq.relative_to(RAIZ)}:{tabela.get_name()}: {nome}")
                for filho in tabela.get_children():
                    visita(filho)

            visita(modulo)
        self.assertEqual(achados, [])


if __name__ == "__main__":
    unittest.main()
