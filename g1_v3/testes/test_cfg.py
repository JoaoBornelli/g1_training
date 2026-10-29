"""Testes da montagem: cena (D1), make_env_cfg, registro e tarefa (D10). Escritos antes do código: o ImportError é esperado.

Só este arquivo chama make_env_cfg() e o registro. Nenhum teste constrói ManagerBasedRlEnv nem compila a cena;
a caixa e a laje são compiladas só como MjSpec (sem física). Números: as linhas "Teste" de D1 e D10 e a §6.3.
"""
import dataclasses
import re
import unittest

import numpy as np
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from g1_v3 import algoritmo, cena, env_cfg, tarefa
from g1_v3.knobs import ATIVO, JUNTAS

# Os 11 sensores de contato do g1_limpo/cena.py (palmas 2, dorsos 2, prateleira 3, apoio, caixa, auto, pés).
SENSORES_G1_LIMPO = ("palma_E", "palma_D", "dorso_E", "dorso_D", "corpo_prateleira", "palma_prateleira",
                     "dorso_prateleira", "apoio_caixa", "caixa_robo", "auto_colisao", "pes_chao")
# Os 7 que a D1 acrescenta (altura, nao_pes_chao, caixa_faces, lidar só com Percepcao.ligada).
SENSORES_NOVOS = ("altura_pes", "altura_pelve", "altura_joelhos", "altura_caixa", "nao_pes_chao",
                  "caixa_faces", "lidar")

NOMES_COLUNAS = ("plano", "escada", "escada_inv", "rampa", "rampa_inv", "escada_aleatoria",
                 "irregular", "obstaculos", "caixas")
# (coluna, topo do range do gerador, teto de CARREGAR da S §11.2, tolerância) — linha "Teste" de D1.
TETOS_CARREGAR = (
    ("escada", 0.20, 0.15, 0.001),
    ("escada_inv", 0.20, 0.15, 0.001),
    ("rampa", 0.36, 0.27, 0.001),
    ("rampa_inv", 0.36, 0.27, 0.001),
    ("escada_aleatoria", 0.20, 0.15, 0.001),
    ("obstaculos", 0.15, 0.10, 0.005),
    ("caixas", 0.15, 0.10, 0.005),
)

# §6.5, na ordem do dict. Termos de fases futuras ainda não existem; o teste checa só o que existe.
ATOR = ("giro_base", "gravidade", "comando", "juntas", "vel_juntas", "acao", "tarefa", "caixa", "mapa")
CRITICO = ATOR + ("vel_lin_base", "caixa_verdade", "alvo_verdade", "alvo_macro", "referencia", "niveis",
                  "degrau", "massa", "forca_caixa", "tarefa_critico")
RECOMPENSAS = ("rastreio_linear", "rastreio_angular", "passo", "vivo", "altura_pe", "escorrego", "vz",
               "giro_xy", "vel_junta", "acel_junta", "taxa_acao", "energia", "limite_junta", "desvio_bracos",
               "desvio_cintura", "desvio_quadril", "tronco_plano", "altura_base", "cerca_sola", "cerca_giro",
               "cerca_tronco", "cerca_com", "cerca_torque", "cerca_pelve", "cerca_recuo", "contato_fora",
               "contato_laje", "freio", "progresso", "sacudida", "aperto", "perda_caixa", "terminacao")
TERMINACOES = ("time_out", "queda", "caixa_perdida", "chegada", "batente")
EVENTOS = ("fatia", "base", "juntas", "cena", "carga", "atrito_pad", "atrito_laje", "vies_imu", "pd",
           "forca_motor", "solref", "com_caixa", "filtros",            # reset
           "atrito_pe", "massa_tronco", "vies_encoder", "tamanho_caixa", "atraso",  # startup
           "empurrao", "forca_caixa", "rfi")                            # intervalo

# Tarefas de cada recompensa da D5. "todas" = ANDAR, PEGAR, CARREGAR, BOTAR (§6.3).
TODAS = ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")
ANDAR_CARREGAR = ("ANDAR", "CARREGAR")
TAREFAS_D5 = {
    "rastreio_linear": TODAS, "rastreio_angular": TODAS, "passo": ANDAR_CARREGAR, "vivo": ANDAR_CARREGAR,
    "altura_pe": ANDAR_CARREGAR, "escorrego": TODAS, "vz": TODAS, "giro_xy": TODAS, "vel_junta": TODAS,
    "acel_junta": TODAS, "taxa_acao": TODAS, "energia": TODAS, "limite_junta": TODAS,
    "desvio_bracos": ANDAR_CARREGAR, "desvio_cintura": ANDAR_CARREGAR, "desvio_quadril": ANDAR_CARREGAR,
    "tronco_plano": ANDAR_CARREGAR, "altura_base": ANDAR_CARREGAR, "cerca_sola": ("PEGAR", "BOTAR"),
    "cerca_giro": TODAS, "cerca_tronco": TODAS, "cerca_com": TODAS, "cerca_torque": TODAS,
    "cerca_pelve": TODAS, "cerca_recuo": ("PEGAR", "CARREGAR", "BOTAR"), "contato_fora": TODAS,
    "terminacao": TODAS,
}


def _teto_do_gerador(coluna, topo, linha, R):
    """t = linha/(R−1). O `random_spread_boxes` (caixas) escala a altura por 0,2 + 0,8·t (primitive_terrains.py:794)."""
    t = linha / (R - 1)
    return topo * (0.2 + 0.8 * t) if coluna == "caixas" else topo * t


def _na_ordem_da_spec(nomes_cfg, ordem_spec):
    """Só nomes da spec, na mesma ordem (as fases futuras podem ainda faltar)."""
    return list(nomes_cfg) == [n for n in ordem_spec if n in nomes_cfg] and set(nomes_cfg) <= set(ordem_spec)


class TestCena(unittest.TestCase):
    def test_nomes_dos_sensores(self):
        # D1: os 11 do g1_limpo e mais 7; com Percepcao.ligada, o lidar entra (18 no total).
        nomes = [s.name for s in cena.sensores(ATIVO)]
        self.assertEqual(sorted(nomes), sorted(SENSORES_G1_LIMPO + SENSORES_NOVOS))

    def test_lidar_so_com_percepcao_ligada(self):
        sem = dataclasses.replace(ATIVO, percepcao=dataclasses.replace(ATIVO.percepcao, ligada=False))
        self.assertNotIn("lidar", [s.name for s in cena.sensores(sem)])

    def test_todo_sensor_da_cena_faz_build(self):
        # B2: caixa_faces tem global_frame=True e exige "tangent" (contact_sensor.py:232-237); o build levantava ValueError.
        for sensor in cena.sensores(ATIVO):
            with self.subTest(sensor=sensor.name):
                sensor.build()

    def test_nao_pes_chao_tem_o_corpo_do_robo_como_primario(self):
        # S1: o exclude filtra os nomes primários; com mode="subtree" o primário era só "pelvis", e a subárvore
        # pegava os dois pés. Primário por corpo, com os *_ankle_roll_link fora.
        sensor = next(s for s in cena.sensores(ATIVO) if s.name == "nao_pes_chao")
        self.assertEqual(sensor.primary.mode, "body")
        self.assertEqual(sensor.primary.pattern, ".*")
        self.assertEqual(sensor.primary.exclude, (r".*_ankle_roll_link",))

    def test_colunas_do_terreno_na_ordem_da_tabela(self):
        self.assertEqual(tuple(c["nome"] for c in cena.COLUNAS), NOMES_COLUNAS)
        self.assertEqual(tuple(ATIVO.terreno.colunas), NOMES_COLUNAS)

    def test_sub_terrenos_com_as_chaves_das_colunas(self):
        sub = cena.terreno_cfg(ATIVO).terrain_generator.sub_terrains
        self.assertEqual(list(sub), list(ATIVO.terreno.colunas))
        self.assertTrue(sub["irregular"].scale_with_difficulty)

    def test_teto_de_carregar_pela_linha_maxima(self):
        colunas = {c["nome"]: c for c in cena.COLUNAS}
        R = ATIVO.terreno.linhas
        for nome, topo, teto, tol in TETOS_CARREGAR:
            with self.subTest(coluna=nome):
                linha = colunas[nome]["linha_max_carregar"]
                self.assertAlmostEqual(_teto_do_gerador(nome, topo, linha, R), teto, delta=tol)

    def test_grupo_da_caixa_e_da_laje_e_mobilia(self):
        # Compila só o MjSpec da caixa e da laje (sem física): toda geom no grupo de mobília.
        for spec in (cena.spec_caixa(ATIVO.cena), cena.spec_laje(ATIVO.cena)):
            grupos = np.asarray(spec.compile().geom_group)
            self.assertGreater(grupos.size, 0)
            self.assertTrue((grupos == ATIVO.cena.grupo_mobilia).all())

    def test_kp_kd_de_cada_junta_vem_do_contrato(self):
        # Cada junta cai em exatamente uma linha de Contrato.pd, e cada atuador leva o kp e o kd dessa linha.
        pd = ATIVO.contrato.pd
        esperado = {}
        for junta in JUNTAS:
            linhas = [p for p in pd if re.fullmatch(p[0], junta)]
            self.assertEqual(len(linhas), 1, junta)
            esperado[junta] = (linhas[0][1], linhas[0][2])
        cobertas = set()
        for atuador in cena.robot_cfg().articulation.actuators:
            for junta in JUNTAS:
                if any(re.fullmatch(e, junta) for e in atuador.target_names_expr):
                    cobertas.add(junta)
                    with self.subTest(junta=junta):
                        self.assertEqual((atuador.stiffness, atuador.damping), esperado[junta])
        self.assertEqual(cobertas, set(JUNTAS))


class TestEnvCfg(unittest.TestCase):
    def test_make_env_cfg_monta(self):
        self.assertIsNotNone(env_cfg.make_env_cfg())

    def test_observacoes_na_ordem_da_spec(self):
        cfg = env_cfg.make_env_cfg()
        self.assertEqual(list(cfg.observations["actor"].terms), list(ATOR))
        self.assertEqual(list(cfg.observations["critic"].terms), list(CRITICO))

    def test_comandos_cadeia_e_twist(self):
        self.assertEqual(list(env_cfg.make_env_cfg().commands), ["cadeia", "twist"])

    def test_recompensas_na_ordem_da_spec(self):
        self.assertTrue(_na_ordem_da_spec(env_cfg.make_env_cfg().rewards, RECOMPENSAS))

    def test_terminacoes_na_ordem_da_spec(self):
        self.assertTrue(_na_ordem_da_spec(env_cfg.make_env_cfg().terminations, TERMINACOES))

    def test_eventos_na_ordem_da_spec(self):
        self.assertTrue(_na_ordem_da_spec(env_cfg.make_env_cfg().events, EVENTOS))

    def test_pesos_iguais_ao_knobs(self):
        # O mjlab multiplica o peso por dt no manager (scale_by_dt): o cfg guarda o peso cru (§6.9: o do degrau 5).
        cfg = env_cfg.make_env_cfg()
        for nome in TAREFAS_D5:
            with self.subTest(termo=nome):
                self.assertEqual(cfg.rewards[nome].weight, getattr(ATIVO.recompensa, nome))

    def test_tarefas_de_cada_recompensa_da_d5(self):
        cfg = env_cfg.make_env_cfg()
        for nome, tarefas in TAREFAS_D5.items():
            with self.subTest(termo=nome):
                self.assertEqual(set(cfg.rewards[nome].params["tarefas"]), set(tarefas))

    def test_ultimo_termo_do_critico_e_tarefa_critico(self):
        self.assertEqual(list(env_cfg.make_env_cfg().observations["critic"].terms)[-1], "tarefa_critico")


class TestRegistro(unittest.TestCase):
    def test_as_duas_tasks_no_registro(self):
        self.assertIn(tarefa.TASK_ID, list_tasks())
        self.assertIn(tarefa.TASK_ID_SEM_PERCEPCAO, list_tasks())
        self.assertEqual(tarefa.TASK_ID, "Mjlab-G1-V3")
        self.assertEqual(tarefa.TASK_ID_SEM_PERCEPCAO, "Mjlab-G1-V3-SemPercepcao")

    def test_sem_percepcao_tira_o_mapa_do_ator(self):
        # Decisão do dono (§6.3): na SemPercepcao o termo `mapa` fica fora do ator; na task normal está dentro.
        self.assertIn("mapa", load_env_cfg(tarefa.TASK_ID).observations["actor"].terms)
        self.assertNotIn("mapa", load_env_cfg(tarefa.TASK_ID_SEM_PERCEPCAO).observations["actor"].terms)


class TestTarefa(unittest.TestCase):
    def test_rl_cfg(self):
        rl = tarefa.rl_cfg()
        self.assertFalse(rl.actor.obs_normalization)
        self.assertFalse(rl.critic.obs_normalization)
        self.assertEqual(rl.num_steps_per_env, ATIVO.balanceador.passos_por_iteracao)
        self.assertEqual(rl.experiment_name, "g1_v3")

    def test_classe_do_algoritmo_e_a_da_v3(self):
        self.assertEqual(tarefa.rl_cfg().algorithm.class_name, algoritmo.CAMINHO)
        self.assertEqual(load_rl_cfg(tarefa.TASK_ID).algorithm.class_name, algoritmo.CAMINHO)


if __name__ == "__main__":
    unittest.main()
