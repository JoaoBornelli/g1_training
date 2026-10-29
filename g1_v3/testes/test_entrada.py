"""Testes de g1_v3/container/entrada.sh em modo seco (D13 do plano, §11; PD-22). Escritos antes do código.

SECO=1 pula GPU e versões e só imprime o comando final: o teste roda o script num subprocesso, sem GPU,
sem rede e sem treino. SAIDA aponta para um diretório temporário; a pasta do experimento é
`$SAIDA/g1_v3` (D13, passo 3). Os nomes `2026-01-01_00-00-00_x` e `2026-01-02_00-00-00_x` seguem o formato
das pastas do mjlab (data e hora, depois `_$RUN`); a "mais nova" é a de nome maior.
Os casos vêm da linha "Teste" da D13. `Mjlab-G1-V3 ` tem espaço final para não casar com `-SemPercepcao`.
"""
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = RAIZ / "g1_v3" / "container" / "entrada.sh"
VELHA = "2026-01-01_00-00-00_x"
NOVA = "2026-01-02_00-00-00_x"


class TestEntrada(unittest.TestCase):
    def setUp(self):
        # sem o script, os testes de "código ≠ 0" passariam por acaso (bash devolve 127): falha antes
        self.assertTrue(ENTRADA.is_file(), f"falta {ENTRADA} (D13)")
        self._tmp = tempfile.TemporaryDirectory()
        self.saida = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def pasta(self, nome, checkpoint=False):
        # uma pasta de corrida em $SAIDA/g1_v3; com `checkpoint`, tem um model_*.pt (vazio basta)
        d = Path(self.saida) / "g1_v3" / nome
        d.mkdir(parents=True, exist_ok=True)
        if checkpoint:
            (d / "model_100.pt").write_bytes(b"")
        return d

    def roda(self, **variaveis):
        # ambiente limpo: não herda RUN nem RETOMA do shell de quem roda o teste
        env = {"PATH": os.environ.get("PATH", ""), "SECO": "1", "SAIDA": self.saida}
        env.update(variaveis)
        return subprocess.run(["bash", str(ENTRADA)], env=env, capture_output=True, text=True, timeout=60)

    def test_sintaxe_bash(self):
        r = subprocess.run(["bash", "-n", str(ENTRADA)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_sem_run_falha(self):
        self.assertNotEqual(self.roda(RETOMA="0").returncode, 0)

    def test_sem_retoma_falha(self):
        self.assertNotEqual(self.roda(RUN="x").returncode, 0)

    def test_retoma_fora_de_0_e_1_falha(self):
        self.assertNotEqual(self.roda(RUN="x", RETOMA="2").returncode, 0)

    def test_retoma0_com_pasta_do_mesmo_run_falha(self):
        self.pasta(VELHA)
        self.assertNotEqual(self.roda(RUN="x", RETOMA="0").returncode, 0)

    def test_retoma0_limpo_monta_comando_de_corrida_nova(self):
        r = self.roda(RUN="x", RETOMA="0")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Mjlab-G1-V3 ", r.stdout)
        self.assertIn("--agent.run-name x", r.stdout)
        self.assertIn("--log-root " + self.saida, r.stdout)
        self.assertNotIn("--agent.resume", r.stdout)

    def test_retoma1_escolhe_a_mais_nova_com_checkpoint(self):
        # a mais nova (NOVA) não tem checkpoint: a retomada cai na VELHA, que tem
        self.pasta(VELHA, checkpoint=True)
        self.pasta(NOVA)
        r = self.roda(RUN="x", RETOMA="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--agent.resume", r.stdout)
        self.assertIn("^" + VELHA + "$", r.stdout)
        self.assertNotIn("^" + NOVA + "$", r.stdout)

    def test_retoma1_sem_pasta_falha(self):
        self.assertNotEqual(self.roda(RUN="x", RETOMA="1").returncode, 0)

    def test_retoma1_so_pasta_sem_checkpoint_falha(self):
        # uma sessão que caiu antes do primeiro checkpoint não conta como retomável
        self.pasta(VELHA)
        self.assertNotEqual(self.roda(RUN="x", RETOMA="1").returncode, 0)

    def test_percepcao0_escolhe_a_task_sem_percepcao(self):
        r = self.roda(RUN="x", RETOMA="0", PERCEPCAO="0")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Mjlab-G1-V3-SemPercepcao", r.stdout)

    def test_texto_do_script_sem_push_gh_nem_pip(self):
        # o pod não empurra nada, não usa `gh` e não instala pacote (contrato da imagem, PD-22).
        # Bordas de palavra: "high " não pode casar com `gh `.
        texto = ENTRADA.read_text(encoding="utf-8")
        for padrao in (r"git\s+push", r"\bgh\s", r"\bpip\b"):
            self.assertIsNone(re.search(padrao, texto), padrao)


if __name__ == "__main__":
    unittest.main()
