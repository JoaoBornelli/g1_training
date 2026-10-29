"""Os checks da Parte 1 da v3 (D12 do plano, seções 1 a 15).

    python -m g1_v3.smoke

Imprime `N ok / M falhas` e sai com código 1 se houver falha. Roda na fase I, pelo dono: ele constrói envs e
roda física, e a implementação não roda o smoke.

⚠ Uma seção que levanta exceção vira uma falha (nome da seção + mensagem), e o smoke segue para a próxima.
⚠ Cada seção fecha o env que abriu (`try/finally`). A seção 14 tem 16 mundos; as outras, 4.
⚠ Zero import de `g1_limpo`. Os imports do mjlab e do `g1_v3` ficam DENTRO das seções: um módulo quebrado vira
   a falha da seção dele, e não derruba o smoke inteiro.
⚠ Os números vêm do `knobs` e do plano (§6.2, §6.4, §6.5, §6.8; I §2.1, §2.4). Nada aqui é a verdade do treino.
"""
from __future__ import annotations

import dataclasses
import importlib
import math
import os
import re
import sys
import tempfile
import time

import torch

# SEMENTE FIXA, como no g1_limpo: sem ela o sorteio dos checks muda de uma execução para outra.
torch.manual_seed(0)

_ok = 0
_falhas: list[str] = []
_atual = ""  # nome da seção em curso: cada falha sai com ele


def check(nome: str, cond, detalhe: str = "") -> None:
    global _ok
    if bool(cond):
        _ok += 1
    else:
        _falhas.append(f"[{_atual}] {nome}{f'  ({detalhe})' if detalhe else ''}")


def _roda(nome: str, fn) -> None:
    """Roda uma seção. Exceção vira falha com o nome da seção; o smoke segue."""
    global _atual
    _atual = nome
    print(f"\n--- {nome}")
    try:
        fn()
    except Exception as e:  # uma seção quebrada não derruba as outras
        _falhas.append(f"[{nome}] exceção {type(e).__name__}: {e}")


def _abre_env(n: int, k=None, play: bool = False):
    """Env de `n` mundos em CPU, seed 0. Quem chama fecha com `try/finally`.

    ⚠ `cfg.seed` = 0: o `v3_gerador` sai da seed do cfg (§6.4). Sem ela o smoke não repete.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from g1_v3.env_cfg import make_env_cfg
    from g1_v3.knobs import ATIVO

    cfg = make_env_cfg(k or ATIVO, play=play)
    cfg.scene.num_envs = n
    cfg.seed = 0
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def _zeros(env):
    from g1_v3.knobs import JUNTAS
    return torch.zeros(env.num_envs, len(JUNTAS), device=env.device)


def _grade_recorte():
    """(441,) x e y de cada célula do recorte, no frame da base (I §2.4).

    Ordem do recorte (`percepcao_lidar.FiltroDeElevacao`): linha = x de +1,0 a −1,0; dentro da linha, y de +1,0
    a −1,0. As 210 primeiras células (x ≥ 0,1) são a metade da frente.
    """
    from g1_v3.knobs import ATIVO
    p = ATIVO.percepcao
    eixo = (p.n_recorte // 2 - torch.arange(p.n_recorte)).float() * p.celula
    gx, gy = torch.meshgrid(eixo, eixo, indexing="ij")
    return gx.reshape(-1), gy.reshape(-1)


# =============================================================================
def s01_cfg_e_registro():
    from mjlab.tasks.registry import list_tasks, load_env_cfg
    from g1_v3 import tarefa  # ⚠ o import REGISTRA as duas tasks (tarefa.py): é o efeito que esta seção mede
    from g1_v3.env_cfg import make_env_cfg
    from g1_v3.knobs import ATIVO

    cfg = make_env_cfg(ATIVO)
    play = make_env_cfg(ATIVO, play=True)
    check("cfg de treino monta", cfg is not None)
    check("cfg de play monta, com episódio longo (o v3_fim acaba a cadeia)", play.episode_length_s > 1e6)
    tasks = list_tasks()
    check("Mjlab-G1-V3 registrada", tarefa.TASK_ID in tasks)
    check("Mjlab-G1-V3-SemPercepcao registrada", tarefa.TASK_ID_SEM_PERCEPCAO in tasks)
    check("load_env_cfg devolve o cfg da task", load_env_cfg(tarefa.TASK_ID) is not None)


def s02_env_nasce_com_1447_canais():
    from g1_v3.knobs import dim_ator

    env = _abre_env(4)
    try:
        obs, _ = env.reset()
        check("env em CPU", str(env.device) == "cpu", str(env.device))
        check("4 mundos", env.num_envs == 4, str(env.num_envs))
        n = obs["actor"].shape[-1]
        check("ator com 1 447 canais (I §2.1)", n == 1447 == dim_ator(), f"{n}")
    finally:
        env.close()


def s03_blocos_e_historico():
    from g1_v3.knobs import Contrato

    c = Contrato()
    env = _abre_env(4)
    try:
        obs, _ = env.reset()
        nomes = list(env.observation_manager.active_terms["actor"])
        check("ordem dos blocos = Contrato.blocos_ator", nomes == [b[0] for b in c.blocos_ator], str(nomes))
        off = 0
        for nome, canais, _escala, hist in c.blocos_ator:
            tam = canais * max(hist, 1)
            if hist > 1:
                # ⚠ layout tempo-maior, mais antigo primeiro (observation_manager.py:60-65; exporta_cena.py:99-101)
                v =obs["actor"][:, off:off + tam].reshape(obs["actor"].shape[0], hist, canais)
                iguais = all(torch.equal(v[:, 0], v[:, i]) for i in range(1, hist))
                check(f"{nome}: os {hist} lugares do histórico iguais depois do reset", iguais)
            off += tam
        check("soma dos blocos = 1 447", off == 1447, str(off))
    finally:
        env.close()


def s04_pd_e_acao_zero():
    import mujoco
    from g1_v3.knobs import ATIVO, JUNTAS, q_de_pe

    env = _abre_env(4)
    try:
        env.reset()
        robo = env.scene["robot"]
        # o índice reordena por nome; o check abaixo confere que a ordem do robô é a de JUNTAS (os índices de
        # `recompensas.py` supõem essa ordem)
        idx = [robo.joint_names.index(j) for j in JUNTAS]
        check("ordem das juntas = JUNTAS", list(robo.joint_names) == list(JUNTAS))
        qpe = torch.as_tensor(q_de_pe(ATIVO.contrato), dtype=torch.float32, device=env.device)
        check("default_joint_pos = q_de_pé", torch.allclose(robo.data.default_joint_pos[:, idx], qpe.expand(4, -1), atol=1e-4))

        env.step(_zeros(env))
        # ⚠ JointPositionAction subtrai o encoder_bias antes de escrever o alvo (envs/mdp/actions/actions.py:222-223):
        # o alvo com ação zero é q_de_pé − viés, e não q_de_pé puro.
        alvo = robo.data.joint_pos_target[:, idx]
        bias = robo.data.encoder_bias[:, idx]
        check("ação zero → alvo do PD = q_de_pé (− viés do encoder)", torch.allclose(alvo, qpe.expand(4, -1) - bias, atol=1e-4))

        # kp e kd do modelo: gainprm[0] = kp e biasprm[2] = −kd (utils/spec.py:307-313); o atuador é achado pelo joint
        m = env.sim.mj_model
        achados, errados = 0, []
        for act in robo.actuators:
            for jn in act.target_names:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
                kp, kd = next((kp, kd) for rgx, kp, kd in ATIVO.contrato.pd if re.fullmatch(rgx, jn))
                for aid in range(m.nu):
                    if m.actuator_trnid[aid, 0] != jid:
                        continue
                    achados += 1
                    ok = (abs(m.actuator_gainprm[aid, 0] - kp) < 1e-6 and abs(m.actuator_biasprm[aid, 1] + kp) < 1e-6
                          and abs(-m.actuator_biasprm[aid, 2] - kd) < 1e-6)
                    if not ok:
                        errados.append(jn)
        check("um atuador por junta (29)", achados == 29, str(achados))
        check("kp e kd do modelo = Contrato.pd", not errados, ", ".join(errados[:5]))
    finally:
        env.close()


def s05_mapa_troca_local_directions():
    env = _abre_env(4)
    try:
        env.reset()
        lidar = env.scene.sensors["lidar"]
        env.step(_zeros(env))
        d0 = lidar.data.distances.clone()
        # ⚠ copy_ no mesmo tensor, como `observacoes.atualiza_percepcao` faz com `direcoes(passo + 1)`: a troca vale no passo seguinte
        novo = lidar._local_directions * torch.tensor([1.0, 1.0, -1.0], device=lidar._local_directions.device)
        lidar._local_directions.copy_(novo)
        env.step(_zeros(env))
        d1 = lidar.data.distances
        check("trocar _local_directions muda as distâncias no passo seguinte", not torch.allclose(d0, d1))
    finally:
        env.close()


def s06_mapa_parado_e_com_passo():
    from g1_v3.estimador import yaw
    from g1_v3.knobs import ATIVO

    p = ATIVO.percepcao
    env = _abre_env(4)
    try:
        env.reset()
        robo = env.scene["robot"]
        plano = env.scene["terrain"].terrain_types == 0
        check("ao menos um mundo no plano (a seção mede o plano)", bool(plano.any()))
        gx, gy = _grade_recorte()
        perto = (gx ** 2 + gy ** 2).sqrt() < 0.8
        frente = gx > 0.05  # 10 linhas com x ≥ 0,1: 210 células (I §2.4 mediu 205 de 210)

        # parado: a ação zero deixa o robô parado (o `TwistV3._update_command` é `pass`, comando.py:440-441)
        for _ in range(20):
            env.step(_zeros(env))
        # ⚠ o mapa é 441 alturas e depois 441 confianças (percepcao_lidar.recorte, E,882); vista = confiança > limiar_visto
        mapa = env.v3_mapa_obs
        conf = mapa[:, 441:]
        vista = conf > p.limiar_visto
        check("parado 20 passos: nenhuma célula a menos de 0,8 m vista (o lidar não vê o chão perto, I §2.4)",
              not bool(vista[plano][:, perto].any()))

        # base movida 1,0 m para a frente em 50 passos de 2 cm: escrita da pose, `sim.forward()` e um passo.
        # ⚠ ler a pose depois de escrevê-la exige `sim.forward()` (entity/data.py:39). A odometria da v3 lê a pose
        # verdadeira depois do forward (comando.py:254), e cada passo de 2 cm entra como um Δ pequeno.
        for _ in range(50):
            pose = robo.data.root_link_pose_w.clone()
            psi = yaw(pose[:, 3:7])
            pose[:, 0] += 0.02 * torch.cos(psi)
            pose[:, 1] += 0.02 * torch.sin(psi)
            robo.write_root_link_pose_to_sim(pose)
            env.sim.forward()
            env.step(_zeros(env))
        mapa = env.v3_mapa_obs
        h, conf = mapa[:, :441], mapa[:, 441:]
        vista = conf > p.limiar_visto
        frac = vista[:, frente].float().mean(dim=1)
        check("≥ 90% das células da metade da frente vistas (I §2.4: 205 de 210)",
              bool((frac[plano] >= 0.9).all()), f"mínimo {frac[plano].min().item():.3f}")
        alt_ok = ((h[:, frente].abs() <= 0.02) | ~vista[:, frente]).all(dim=1)
        check("alturas das células vistas a ±2 cm", bool(alt_ok[plano].all()))
    finally:
        env.close()


def s07_terreno_na_coluna_e_linha_certas():
    from g1_v3.knobs import ATIVO

    env = _abre_env(4)
    try:
        ter = env.scene["terrain"]
        n_colunas = len(ATIVO.terreno.colunas)
        tabela = ter.terrain_origins  # (linhas, colunas, 3), indexada [linha, coluna] como o mjlab (terrain_entity.py:232)
        for _ in range(3):
            env.reset()
            tipos, niveis = ter.terrain_types, ter.terrain_levels
            nivel_max = env.v3_cur.nivel_terreno("ANDAR")  # ANDAR é a única fatia no começo (S §11.1)
            check("coluna dentro de 0..8", bool(((tipos >= 0) & (tipos < n_colunas)).all()))
            check("linha ≤ nível de terreno da tarefa (ANDAR)", bool((niveis <= nivel_max).all()),
                  f"máx {int(niveis.max())} > {nivel_max}")
            check("env_origins = terrain_origins[linha, coluna]",
                  torch.allclose(env.scene.env_origins, tabela[niveis, tipos], atol=1e-5))
    finally:
        env.close()


def s08_lags_constantes_no_episodio():
    env = _abre_env(4)
    try:
        env.reset()
        robo = env.scene["robot"]
        # ⚠ `_delay_buffer` e `_group_obs_term_delay_buffer` são atributos privados do mjlab 1.5.3, que o entrada.sh
        # pina (actuator.py:219, observation_manager.py:397). Sem API pública para o lag: se o mjlab mudar, esta seção quebra.
        bufs_atuador = [a._delay_buffer for a in robo.actuators if getattr(a, "_delay_buffer", None) is not None]
        bufs_obs = [env.observation_manager._group_obs_term_delay_buffer["actor"][t]
                    for t in ("giro_base", "gravidade", "juntas", "vel_juntas", "mapa")]
        check("atuadores com buffer de atraso", len(bufs_atuador) > 0, str(len(bufs_atuador)))
        # eventos.atraso: 1 passo da política × decimação 4 = 4 passos de física nos atuadores; 1 passo nas observações
        check("lag do atuador: teto 4 (0..4)", all(b.max_lag == 4 for b in bufs_atuador),
              str({b.max_lag for b in bufs_atuador}))
        check("lag da observação: teto 1 (0..1)", all(b.max_lag == 1 for b in bufs_obs),
              str({b.max_lag for b in bufs_obs}))

        antes = [b.current_lags.clone() for b in bufs_atuador + bufs_obs]
        zeros = _zeros(env)
        for _ in range(10):
            env.step(zeros)
        sem_reset = env.episode_length_buf == 10  # mundos que não resetaram: o lag é do mesmo episódio
        check("há mundo sem reset para medir", bool(sem_reset.any()))
        depois = [b.current_lags for b in bufs_atuador + bufs_obs]
        constantes = all(torch.equal(a[sem_reset], d[sem_reset]) for a, d in zip(antes, depois))
        check("lag constante no episódio (atuador e observação)", constantes)
    finally:
        env.close()


def s09_recompensas_finitas_e_vivo():
    env = _abre_env(4)
    try:
        env.reset()
        env.v3_tarefa.zero_()  # ANDAR (índice 0 de knobs.TAREFAS): ⚠ fixa a tarefa só para testar os termos
        env.v3_perdeu.zero_()
        ruins = []
        for nome, termo in env.cfg.rewards.items():
            valor = termo.func(env, **termo.params).float()
            if not bool(torch.isfinite(valor).all()):
                ruins.append(nome)
        check("todo termo de recompensa finito no reset", not ruins, ", ".join(ruins))
        vivo = env.cfg.rewards["vivo"]
        # reward_buf = núcleo × peso × dt (o mjlab escala por dt, managers/reward_manager.py:39); núcleo do vivo = 1
        buf = vivo.weight * env.step_dt * vivo.func(env, **vivo.params).float()
        check("vivo = 0,15·dt no ANDAR", torch.allclose(buf, torch.full_like(buf, 0.15 * env.step_dt)))
        # de pé, ação zero: o `nao_pes_chao` tem um primário por corpo (menos os pés), e nenhum corpo fora dos pés toca o
        # chão. Sem a exclusão dos pés, o termo valeria 1 em todo passo (cena.py:389-390).
        for _ in range(5):
            env.step(_zeros(env))
        fora = env.cfg.rewards["contato_fora"]
        check("de pé, ação zero: contato_fora = 0", bool((fora.func(env, **fora.params) == 0).all()))
    finally:
        env.close()


def s10_queda_por_inclinacao_e_altura():
    # ⚠ a `queda` usa a tupla TODAS (env_cfg.py:30); a seção testa só os dois ramos do n_queda.
    from g1_v3 import terminacoes
    from g1_v3.env_cfg import TODAS

    def queda_depois_de(altura=None, inclinacao=None):
        """Escreve a pose, roda `forward` e `sense` e lê a `queda` direto, sem `step`.

        ⚠ o raycast (`altura_pelve`) só atualiza no `sense` (raycast_sensor.py:691); o `step` calcula a terminação
        antes dele. Por isso o check chama a função, e não o `termination_manager`.
        """
        env = _abre_env(4)
        try:
            env.reset()
            env.v3_tarefa.zero_()  # ANDAR
            robo = env.scene["robot"]
            pose = robo.data.root_link_pose_w.clone()
            if inclinacao is not None:  # rolamento em x: quaternion wxyz (cos θ/2, sin θ/2, 0, 0)
                pose[:, 3] = math.cos(inclinacao / 2)
                pose[:, 4] = math.sin(inclinacao / 2)
                pose[:, 5:7] = 0.0
            if altura is not None:
                pose[:, 2] = env.scene.env_origins[:, 2] + altura
            robo.write_root_link_pose_to_sim(pose)
            env.sim.forward()
            env.sim.sense()
            return terminacoes.queda(env, TODAS).clone()
        finally:
            env.close()

    check("de pé: sem queda", not bool(queda_depois_de().any()))
    check("inclinação de 0,9 rad (> 0,8 de marcha): queda", bool(queda_depois_de(inclinacao=0.9).all()))
    check("base 0,10 m acima do chão (< 0,20 de base_min): queda", bool(queda_depois_de(altura=0.10).all()))


def s11_so_andar_no_comeco():
    env = _abre_env(4)
    try:
        for _ in range(3):
            env.reset()
            check("fatia ANDAR no começo (S §11.1)", bool((env.v3_fatia == 0).all()))
            check("tarefa ANDAR no começo", bool((env.v3_tarefa == 0).all()))
        check("currículo na iteração 0", env.v3_cur.it == 0, str(env.v3_cur.it))
    finally:
        env.close()


def s12_save_e_load_do_runner():
    from mjlab.rl import RslRlVecEnvWrapper
    from g1_v3 import tarefa
    from g1_v3.runner import RunnerV3

    env = _abre_env(4)
    try:
        env.reset()
        wrapper = RslRlVecEnvWrapper(env)
        agente = dataclasses.asdict(tarefa.rl_cfg())  # ⚠ como o mjlab/scripts/train.py:150 passa o agente
        with tempfile.TemporaryDirectory() as tmp:
            # ⚠ cria o PPO inteiro (rsl_rl) só para o save e o load: é o custo do check
            runner = RunnerV3(wrapper, agente, log_dir=tmp, device="cpu")
            cur = env.v3_cur
            s0 = cur.estado()
            caminho = os.path.join(tmp, "model_0.pt")
            runner.save(caminho)
            cur.it += 5  # muda o estado depois do save: o load tem de desfazer a mudança
            runner.load(caminho)
            check("save → load devolve o mesmo v3_cur.estado()", env.v3_cur.estado() == s0)
    finally:
        env.close()


def s13_class_name_do_algoritmo():
    from rsl_rl.algorithms import PPO
    from g1_v3 import algoritmo, tarefa

    modulo, _, nome = algoritmo.CAMINHO.partition(":")
    cls = getattr(importlib.import_module(modulo), nome)
    check("CAMINHO resolve para PPOPorTarefa", cls is algoritmo.PPOPorTarefa)
    check("é subclasse do PPO do rsl_rl", issubclass(cls, PPO))
    check("rl_cfg aponta para o CAMINHO", tarefa.rl_cfg().algorithm.class_name == algoritmo.CAMINHO)


def s14_200_passos_aleatorios_sem_nan():
    from g1_v3.knobs import JUNTAS

    env = _abre_env(16)  # ⚠ 16 mundos em CPU: a seção mais cara do smoke
    try:
        obs, _ = env.reset()
        ruins = 0
        for _ in range(200):
            acao = torch.rand(16, len(JUNTAS), device=env.device) * 2 - 1
            obs, rew, _term, _trunc, _info = env.step(acao)
            if not (torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
                    and torch.isfinite(rew).all()):
                ruins += 1
        check("200 passos com ação aleatória em 16 mundos sem NaN", ruins == 0, f"{ruins} passos com NaN")
    finally:
        env.close()


def s15_tempo_por_passo():
    """Só imprime (fase 5c, medido em CPU): não conta como check."""
    from g1_v3.knobs import ATIVO

    sem = dataclasses.replace(ATIVO, percepcao=dataclasses.replace(ATIVO.percepcao, ligada=False))
    for rotulo, k in (("com Percepcao.ligada", ATIVO), ("sem Percepcao.ligada", sem)):
        env = _abre_env(4, k=k)
        try:
            env.reset()
            zeros = _zeros(env)
            for _ in range(10):  # aquece
                env.step(zeros)
            t0 = time.perf_counter()
            for _ in range(50):
                env.step(zeros)
            ms = (time.perf_counter() - t0) * 1000 / 50
            print(f"  {rotulo}: {ms:.1f} ms por passo (4 mundos, CPU)")
        finally:
            env.close()


SECOES = (
    ("1 cfg monta e a task registra", s01_cfg_e_registro),
    ("2 env de 4 mundos em CPU, ator com 1 447 canais", s02_env_nasce_com_1447_canais),
    ("3 blocos do ator e histórico depois do reset", s03_blocos_e_historico),
    ("4 ação zero → alvo do PD; kp e kd do modelo", s04_pd_e_acao_zero),
    ("5 trocar _local_directions muda as distâncias", s05_mapa_troca_local_directions),
    ("6 mapa parado e com passo de 1,0 m", s06_mapa_parado_e_com_passo),
    ("7 terreno na coluna e na linha certas", s07_terreno_na_coluna_e_linha_certas),
    ("8 lag do atuador e da observação", s08_lags_constantes_no_episodio),
    ("9 recompensas finitas no reset; vivo = 0,15·dt", s09_recompensas_finitas_e_vivo),
    ("10 queda por inclinação e por altura", s10_queda_por_inclinacao_e_altura),
    ("11 só a fatia ANDAR no começo", s11_so_andar_no_comeco),
    ("12 save → load do runner", s12_save_e_load_do_runner),
    ("13 class_name do algoritmo", s13_class_name_do_algoritmo),
    ("14 200 passos aleatórios em 16 mundos sem NaN", s14_200_passos_aleatorios_sem_nan),
    ("15 tempo por passo (imprime)", s15_tempo_por_passo),
)


def main() -> int:
    for nome, fn in SECOES:
        _roda(nome, fn)
    print(f"\n{_ok} ok / {len(_falhas)} falhas")
    for f in _falhas:
        print(f"  FALHA {f}")
    return 1 if _falhas else 0


if __name__ == "__main__":
    sys.exit(main())
