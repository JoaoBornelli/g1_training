"""Os checks da v3 (D12 e E10 do plano, seções 1 a 29: Partes 1 a 3).

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


# =============================================================================
# Parte 2 (E10): cadeia B. Os buffers `v3_*` são forçados na seção que diz; o resto roda como no treino.
def _trecho_de(cad, tarefa: str, n: int, device, fatia: int = 1, ultimo: bool = False):
    """(n,) long: o primeiro trecho da `tarefa` dentro da `fatia` (1 = FATIA_B, o padrão; 2 = FATIA_C), ou o último.

    ⚠ O ANDAR aparece duas vezes na fatia C (a espera e o ANDAR parado final): `ultimo` escolhe o final.
    """
    from g1_v3.knobs import TAREFAS
    ini = int(cad.tab.inicio[fatia])
    fim = int(cad.tab.inicio[fatia + 1]) if fatia < 2 else len(cad.tab.tarefa)
    idx = (cad.tab.tarefa == TAREFAS.index(tarefa)).nonzero().flatten()
    idx = idx[(idx >= ini) & (idx < fim)]
    i = int(idx[-1] if ultimo else idx[0])
    return torch.full((n,), i, dtype=torch.long, device=device)


def _abre_trecho_de(env, tarefa: str, fatia: int = 1, ultimo: bool = False):
    """Força a `fatia` (B por padrão) e abre o trecho de `tarefa` em todos os mundos, pelo `_abre_trecho` do `compute`.

    ⚠ Força: o reset sorteou um trecho de ANDAR da fatia A, e aqui `v3_fatia` vai à `fatia`. Devolve o termo `cadeia`.
    """
    cad = env.command_manager.get_term("cadeia")
    n = env.num_envs
    env.v3_fatia[:] = fatia
    ids = torch.arange(n, device=env.device)
    cad._abre_trecho(ids, _trecho_de(cad, tarefa, n, env.device, fatia, ultimo), cad._mede())
    return cad


def _poe_caixa_parada(env, pose):
    """Põe a caixa na `pose` (n, 7) com velocidade zero: a escrita da s27 (e do `_abre_botar_c`, M2)."""
    env.scene["box"].write_root_link_pose_to_sim(pose)
    env.scene["box"].write_root_link_velocity_to_sim(torch.zeros(pose.shape[0], 6, device=pose.device))


def _abre_botar_c(env):
    """Fatia C forçada e o trecho de BOTAR aberto em todos os mundos (o caminho do `compute`). Devolve o `cadeia`.

    ⚠ Estado forçado: o reset sorteou ANDAR (fatia A). O BOTAR abre a laje de destino (`posiciona_destino`). A caixa
    fica parada sobre a soltura (M2, F9): sem isto ela fica longe do robô, perde a laje quando ela sai e cai 20 m.
    Quem chama faz o `env.reset()` antes.
    """
    from g1_v3.comando import FATIA_C
    cad = _abre_trecho_de(env, "BOTAR", FATIA_C)
    pose = torch.zeros(env.num_envs, 7, device=env.device)
    pose[:, :3] = env.v3_alvo_w
    pose[:, 3] = 1.0
    _poe_caixa_parada(env, pose)
    env.sim.forward()  # ⚠ a leitura das poses depois da escrita da laje pede o forward (como em s06)
    return cad


def _abre_cargar_no_curriculo(env):
    """Abre as 4 células do CARREGAR no `v3_cur` (estado forçado, como o currículo as abre depois do portão do ANDAR).

    ⚠ Sem isto o `twist` sorteia uma célula fechada e `torch.multinomial` sobre vetor vazio levanta `RuntimeError`
    (balanceador.py:229-230; E11 M1). `_abre_cel` é privado, como o `_dr` da s24.
    """
    cur = env.v3_cur
    for tipo in ("reto", "giro_lugar", "arco", "terreno"):
        cur._abre_cel("CARREGAR", tipo)


def _le_caixa_valida(env):
    """Dá ao estimador uma leitura válida da caixa AGORA: a pose verdadeira, com idade 0 (estado forçado).

    ⚠ S5b (decisão do dono, 29/10): o PEGAR só abre depois da primeira leitura válida. As seções que abrem o PEGAR
    forçado não esperam a câmera achar a caixa: a visão fica para a s20. Sem a leitura, a pega cai na base.
    """
    est, caixa = env.v3_estimador, env.scene["box"].data
    ids = torch.arange(env.num_envs, device=env.device)
    est._r_t[ids] = est.t
    est._r_pos[ids] = caixa.root_link_pos_w
    est._r_quat[ids] = caixa.root_link_quat_w
    est._r_meia[ids] = env.v3_meia


def _assenta_fatia_b(env, passos: int = 10):
    """Põe a cena da fatia B (`posiciona_cena`) e deixa o robô parado `passos` passos antes de abrir o trecho.

    ⚠ E11 M3: com a cena estacionada, a s17 passava com a pega na base (S5b) e a s18 via o Φ do PD assentando.
    ⚠ O trecho do reset é o ANDAR da fatia A (`episodio_s`, 20 s): não termina nesses passos, então a cadeia não troca.
    """
    from g1_v3 import eventos
    from g1_v3.comando import FATIA_B
    env.v3_fatia[:] = FATIA_B
    eventos.posiciona_cena(env, torch.arange(env.num_envs, device=env.device))
    env.sim.forward()  # ⚠ a leitura das poses depois da escrita pede o forward (como em s06)
    zeros = _zeros(env)
    for _ in range(passos):
        env.step(zeros)


def _por_termo(env) -> dict:
    """{termo: (E,) valor do passo}, lido do `reward_manager` (peso × função, sem o dt: só o sinal importa)."""
    linhas = [env.reward_manager.get_active_iterable_terms(i) for i in range(env.num_envs)]
    return {nome: torch.tensor([linha[k][1][0] for linha in linhas]) for k, (nome, _) in enumerate(linhas[0])}


# =============================================================================
def s16_fatia_b_forcada_topo_da_laje_e_caixa_sobre_ela():
    from g1_v3 import eventos
    from g1_v3.comando import FATIA_B
    from g1_v3.knobs import ATIVO

    env = _abre_env(4)
    try:
        env.reset()
        ids = torch.arange(env.num_envs, device=env.device)
        # ⚠ força a fatia B: o reset sorteou a fatia, e `posiciona_cena` lê `v3_fatia` e põe a laje e a caixa de novo
        env.v3_fatia[:] = FATIA_B
        eventos.posiciona_cena(env, ids)
        env.sim.forward()  # ⚠ a leitura das poses depois da escrita pede o forward (como em s06)
        topo_w = env.scene.env_origins[:, 2] + env.v3_topo
        # ⚠ a laje é mocap: a pose lida é a do corpo (`root_link_pos_w`), que o forward atualiza a partir do mocap.
        laje_z = env.scene["table"].data.root_link_pos_w[:, 2]
        check("laje: topo = origem + v3_topo", torch.allclose(laje_z + ATIVO.cena.prateleira_meia_z, topo_w, atol=1e-4),
              f"{(laje_z + ATIVO.cena.prateleira_meia_z - topo_w).abs().max().item():.2e}")
        caixa_z = env.scene["box"].data.root_link_pos_w[:, 2]
        check("caixa sobre a laje: centro = topo + v3_meia", torch.allclose(caixa_z, topo_w + env.v3_meia, atol=1e-4))
    finally:
        env.close()


def s17_abertura_do_pegar_poe_alvo_em_p0_e_sigma():
    from g1_v3.knobs import TAREFAS
    from g1_v3.alvos import P0

    env = _abre_env(4)
    try:
        env.reset()
        _assenta_fatia_b(env)  # ⚠ E11 M3: a cena se põe e assenta antes da abertura, como no treino
        _le_caixa_valida(env)  # S5b: o PEGAR abre depois da primeira leitura válida (estado do estimador forçado)
        cad = _abre_trecho_de(env, "PEGAR")  # ⚠ força: abre o trecho de PEGAR como o `compute` abriria
        m = cad._mede()
        check("PEGAR aberto em todos os mundos (`cad.pega`)", bool(cad.pega.all()))
        check("alvo corrente = P0", bool((env.v3_alvo_idx == P0).all()), str(env.v3_alvo_idx.tolist()))
        check("alvo da caixa = pega estimada (v3_alvo_w = v3_caixa_pega_w)",
              torch.allclose(env.v3_alvo_w, env.v3_caixa_pega_w, atol=1e-6))
        # a pega sai do frame da base pelo estimador: com a leitura da pose verdadeira, a pega cai na caixa (< 2 cm)
        erro = (env.v3_caixa_pega_w - env.scene["box"].data.root_link_pos_w).norm(dim=-1)
        check("pega estimada a < 2 cm da caixa (a base não pode valer por ela)", bool((erro < 0.02).all()),
              f"erro {erro.tolist()}")
        # σ = distância inicial das palmas a P0 (com o piso de `_SIGMA_MIN`, como em `_sigma`)
        sigma = cad._sigma(m, torch.full_like(env.v3_alvo_idx, P0))
        check("σ = distância inicial das palmas a P0", torch.allclose(env.v3_sigma, sigma, atol=1e-5))
        check("relógio do alvo zerado", bool((env.v3_relogio_alvo == 0).all()))
        check("tarefa publicada = PEGAR", bool((env.v3_tarefa == TAREFAS.index("PEGAR")).all()))
    finally:
        env.close()


def s18_pegar_com_acao_zero_nada_paga_positivo():
    from g1_v3.knobs import TAREFAS

    env = _abre_env(4)
    try:
        env.reset()
        # ⚠ E11 M3: o PD assenta antes da abertura (sem isso o Φ do assentamento paga); a leitura é forçada (S5b).
        # ⚠ DÚVIDA: 10 passos podem não bastar para o PD assentar. Se a soma passar de 1e-6, subir `passos` antes de
        # afrouxar a tolerância.
        _assenta_fatia_b(env)
        _le_caixa_valida(env)
        _abre_trecho_de(env, "PEGAR")  # ⚠ força o trecho de PEGAR; o `avanca_trecho` pode sair dele antes de 50 passos
        zeros = _zeros(env)
        soma, fora = 0.0, 0
        for _ in range(50):
            env.step(zeros)
            fora += int((env.v3_tarefa != TAREFAS.index("PEGAR")).sum())
            for v in _por_termo(env).values():
                soma += float(v.clamp(min=0.0).sum())  # o que paga positivo neste passo, somado em mundos e termos
        check("50 passos no PEGAR (a precondição da seção)", fora == 0, f"{fora} mundo-passo fora do PEGAR")
        # ⚠ `progresso` e `rastreio` (c = 0: deslize ≤ 0, potencial.py:106) e `vivo`/`passo` (mascarados) não pagam
        check("soma dos termos que pagam positivo = 0 (ação zero)", soma <= 1e-6, f"soma {soma:.3g}")
    finally:
        env.close()


def s19_caixa_no_chao_paga_uma_vez_e_vira_andar_com_a_chave():
    from g1_v3.knobs import ATIVO, TAREFAS

    def roda(vira_andar: bool) -> dict:
        # a chave vai ao cfg inteiro: a cadeia e a terminação (`params`, E11 S9) leem o mesmo `k`
        k = dataclasses.replace(ATIVO, cadeia=dataclasses.replace(ATIVO.cadeia, caixa_perdida_vira_andar=vira_andar))
        env = _abre_env(4, k=k)
        try:
            env.reset()
            n = env.num_envs
            _le_caixa_valida(env)  # S5b: o PEGAR abre depois da primeira leitura válida (estado forçado)
            _abre_trecho_de(env, "PEGAR")
            caixa = env.scene["box"]
            pose = caixa.data.root_link_pose_w.clone()
            pose[:, 2] = env.scene.env_origins[:, 2] + env.v3_meia  # ⚠ força: a caixa no chão (fundo a 0 m)
            caixa.write_root_link_pose_to_sim(pose)
            caixa.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device))
            env.sim.forward()
            zeros = _zeros(env)
            pulsos = torch.zeros(n, dtype=torch.long, device=env.device)
            pagos = torch.zeros(n, dtype=torch.long, device=env.device)
            termina = torch.zeros(n, dtype=torch.bool, device=env.device)
            for _ in range(5):
                _o, _r, term, _t, _i = env.step(zeros)
                pulsos += env.v3_perdeu_agora.long()
                pagos += (_por_termo(env)["perda_caixa"] != 0).long()
                termina |= term.bool()
                if bool(term.any()):
                    break  # a terminação reseta o mundo no passo: os pulsos seguintes não valem
            return dict(pulsos=pulsos, pagos=pagos, termina=termina, tarefa=env.v3_tarefa.clone(),
                        cfg_tem="caixa_perdida" in env.cfg.terminations)
        finally:
            env.close()

    on, off = roda(True), roda(False)
    check("chave ligada: a perda pulsa uma vez em cada mundo", bool((on["pulsos"] == 1).all()), str(on["pulsos"].tolist()))
    check("chave ligada: o pagamento de `perda_caixa` cai só no pulso", bool((on["pagos"] == on["pulsos"]).all()),
          f"pagos {on['pagos'].tolist()}")
    check("chave ligada: a tarefa vira ANDAR", bool((on["tarefa"] == TAREFAS.index("ANDAR")).all()))
    check("chave ligada: a cadeia não termina na perda", not bool(on["termina"].any()))
    check("chave desligada: termina no passo da perda", bool(off["termina"].all()))
    check("cfg tem o termo `caixa_perdida` (S §10: sem a chave, termina)", on["cfg_tem"] and off["cfg_tem"])


def s20_estimador_ve_caixa_a_frente_e_idade_cresce_com_bracos():
    from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
    from g1_v3.knobs import ATIVO, q_de_pe

    env = _abre_env(4)
    try:
        env.reset()
        robo, caixa = env.scene["robot"], env.scene["box"]
        dev, n = env.device, env.num_envs
        # caixa 0,45 m à frente da base, na altura do eixo óptico da D435i (frame da base). A câmera fica em
        # `d435_pos` = (0,058; 0,018; 0,430) m (torso ≈ pelve, ⚠ DÚVIDA), com pitch 47,6°: o eixo óptico passa por
        # (0,45; 0; 0,0), e a caixa cai no centro do FOV (vertical ±21°; `Estimador.fov_*`).
        ofs = torch.tensor([0.45, 0.0, 0.0], device=dev)

        def repoe_caixa():
            # ⚠ força: a caixa é reposta a cada passo, senão cairia (0,4 s de queda livre são 0,8 m) antes do estimador
            pose = torch.zeros(n, 7, device=dev)
            pose[:, :3] = robo.data.root_link_pos_w + quat_apply(robo.data.root_link_quat_w, ofs.expand(n, 3))
            pose[:, 3] = 1.0
            caixa.write_root_link_pose_to_sim(pose)
            caixa.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev))

        zeros = _zeros(env)
        for _ in range(20):
            repoe_caixa()
            env.step(zeros)
        # a checagem de FOV usa o site `d435` (o eixo óptico é o +x), com os mesmos limites de `estimador.visivel`
        sit = robo.find_sites(("d435",))[0]
        cam_p = robo.data.site_pos_w[:, sit].squeeze(1)
        cam_q = robo.data.site_quat_w[:, sit].squeeze(1)
        rel = quat_apply_inverse(cam_q, caixa.data.root_link_pos_w - cam_p)  # no frame da câmera
        k_est = ATIVO.estimador
        az = torch.atan2(rel[:, 1], rel[:, 0]).abs()
        el = torch.atan2(rel[:, 2], rel[:, 0]).abs()
        dist = rel.norm(dim=-1)
        no_fov = ((az <= math.radians(k_est.fov_h_graus) / 2) & (el <= math.radians(k_est.fov_v_graus) / 2)
                  & (dist >= k_est.alcance[0]) & (dist <= k_est.alcance[1]))
        check("centro da caixa no FOV da D435i e na faixa do depth (Estimador.fov_* e alcance)", bool(no_fov.all()),
              f"az {az.tolist()}, el {el.tolist()}, d {dist.tolist()}")
        s = env.v3_estimador.saida_b(env.v3_odo, robo.data.root_link_quat_w)
        check("estimador vê a caixa à frente: x na base ≈ 0,45 m", bool(((s[:, 0] - 0.45).abs() < 0.03).all()),
              f"x {s[:, 0].tolist()}")
        check("leitura nova: idade ≤ 0,1 s", bool((s[:, 7] <= 0.1).all()), f"idade {s[:, 7].tolist()}")
        # braços na frente: q_com_caixa segurada pelo PD (ação = (q − q_de_pé)/escala); a caixa segue no mesmo lugar
        q = env.v3_ref.q_com_caixa(env.v3_meia).to(dev, torch.float32)
        qde = torch.as_tensor(q_de_pe(ATIVO.contrato), dtype=torch.float32, device=dev)
        acao = (q - qde) / ATIVO.contrato.escala_acao
        idade0 = s[:, 7].clone()
        for _ in range(30):  # 0,6 s com os braços à frente
            repoe_caixa()
            env.step(acao)
        s2 = env.v3_estimador.saida_b(env.v3_odo, robo.data.root_link_quat_w)
        # ⚠ DÚVIDA: a oclusão pelas cápsulas dos braços (`visivel`, estimador.py:107) decide se a idade cresce. Com a
        # caixa no eixo óptico, os braços a tapam: a seção mede a oclusão; o dono mede a geometria na fase I.
        check("braços à frente: a idade cresce ≥ 0,4 s em 0,6 s", bool(((s2[:, 7] - idade0) >= 0.4).all()),
              f"Δ {(s2[:, 7] - idade0).tolist()}")
    finally:
        env.close()


def s21_pose_de_pe_com_caixa_dispara_aviso_depois_de_meio_segundo():
    from g1_v3.knobs import ATIVO, TAREFAS, q_de_pe

    env = _abre_env(4)
    try:
        env.reset()
        _abre_cargar_no_curriculo(env)  # M1: o trecho que vem depois do aviso é o CARREGAR; a célula não pode estar fechada
        _le_caixa_valida(env)  # S5b: o PEGAR abre depois da primeira leitura válida (estado forçado)
        cad = _abre_trecho_de(env, "PEGAR")
        # M1: o trecho acaba pelo aviso do detector (por_detector), e não pelo atraso sorteado em U[0; 1] s, que
        # venceria em ~99% dos mundos. ⚠ `inf` segura o trecho no PEGAR durante os 60 passos.
        cad.alvo_s[:] = float("inf")
        robo, dev, a = env.scene["robot"], env.device, env.v3_meia
        q = env.v3_ref.q_com_caixa(a).to(dev, torch.float32)  # ⚠ a ordem é a de JUNTAS (s04)
        qde = torch.as_tensor(q_de_pe(ATIVO.contrato), dtype=torch.float32, device=dev)
        # ⚠ força a pose de pé com caixa: braços em q_com_caixa e a pelve na altura de `pelve_com_caixa`.
        # ⚠ Limite conhecido (I §4.2): o detector lê só as palmas e a pelve (alvos.detecta_fim), não a caixa. Que o PD
        # alcance q_com_caixa e que a pelve fique em 3 cm: medir na fase I.
        robo.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.sim.forward()
        env.sim.sense()
        pose = robo.data.root_link_pose_w.clone()
        pose[:, 2] += env.v3_ref.pelve_com_caixa(a).to(dev) - env.scene["altura_pelve"].data.heights[:, 0]
        robo.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        env.sim.sense()
        acao = (q - qde) / ATIVO.contrato.escala_acao  # segura os braços: o PD segue q_com_caixa
        avisos = []
        for _ in range(60):
            env.step(acao)
            avisos.append(env.v3_aviso.clone())
        m = torch.stack(avisos)  # (60, E)
        check("precondição: a tarefa segue PEGAR durante os 60 passos (o trecho é por detector)",
              bool((env.v3_tarefa == TAREFAS.index("PEGAR")).all()))
        check("nenhum aviso nos primeiros 24 passos (0,48 s: a janela de 0,5 s ainda não fechou)", not bool(m[:24].any()))
        check("aviso depois de 0,5 s, com a pose de pé com caixa", bool(m.any(dim=0).all()),
              f"primeiro aviso por mundo: {m.float().argmax(0).tolist()}")
    finally:
        env.close()


def s22_laje_sai_com_o_carregar_que_anda():
    from g1_v3 import eventos
    from g1_v3.comando import FATIA_B
    from g1_v3.knobs import ATIVO

    env = _abre_env(4)
    try:
        env.reset()
        _abre_cargar_no_curriculo(env)  # M1: o `twist` sorteia célula do CARREGAR; fechada, `multinomial` falha
        n = env.num_envs
        ids = torch.arange(n, device=env.device)
        env.v3_fatia[:] = FATIA_B  # ⚠ força a fatia B; a laje começa no topo (posiciona_cena)
        eventos.posiciona_cena(env, ids)
        env.sim.forward()
        laje = env.scene["table"]
        origem_z = env.scene.env_origins[:, 2]
        cad = env.command_manager.get_term("cadeia")
        t_carregar = _trecho_de(cad, "CARREGAR", n, env.device)
        andando = torch.zeros(n, dtype=torch.bool, device=env.device)
        antes = torch.zeros(n, device=env.device)
        for _ in range(20):  # o comando do CARREGAR é sorteado na abertura: repete até algum mundo andar
            antes = laje.data.root_link_pos_w[:, 2].clone()
            cad._abre_trecho(ids, t_carregar, cad._mede())
            andando = cad._twist().vel_command_b.abs().sum(-1) > 0
            if bool(andando.any()):
                break
        env.sim.forward()
        depois = laje.data.root_link_pos_w[:, 2]
        estacionada = origem_z + ATIVO.cena.estaciona_z - ATIVO.cena.prateleira_meia_z
        esperado = torch.where(andando, estacionada, antes)
        check("algum mundo sorteou o CARREGAR andando (senão a seção não mede)", bool(andando.any()))
        check("laje sai (estaciona_z) só onde o CARREGAR anda; onde para, fica", torch.allclose(depois, esperado, atol=1e-4),
              f"andando {andando.tolist()}")
    finally:
        env.close()


def s23_tamanho_do_critico_confere_com_6_5():
    from g1_v3.knobs import dim_ator

    env = _abre_env(4)
    try:
        obs, _ = env.reset()
        # §6.5: os nove do ator com o mesmo histórico (1 447), mais vel_lin_base (3×5), caixa_verdade (7),
        # alvo_verdade (3), alvo_macro (10), referencia (8), niveis (4), degrau (1), massa (1), forca_caixa (1),
        # tarefa_critico (6). Total: 1 503.
        esperado = dim_ator() + 3 * 5 + 7 + 3 + 10 + 8 + 4 + 1 + 1 + 1 + 6
        check("crítico = 1 447 do ator + 56 de §6.5 (= 1 503)", obs["critic"].shape[-1] == esperado,
              f"{obs['critic'].shape[-1]} != {esperado}")
    finally:
        env.close()


def s24_atraso_do_atuador_depois_do_portao_do_pegar():
    from g1_v3 import curriculo

    env = _abre_env(4)
    try:
        env.reset()
        # ⚠ força o estado: a trava `portao0` do DR (balanceador.py:74) é a que o fecho liga quando o PEGAR passa o
        # portão do nível 0. A trava é privada do `Curriculo`: a seção a escreve direto.
        env.v3_cur._dr["portao0"] = True
        curriculo.v3(env, torch.arange(env.num_envs, device=env.device))  # põe o teto nos buffers (eventos.atraso)
        check("DR: atraso_max = atraso_portao[1] = 2 passos da política", env.v3_cur.dr()["atraso_max"] == 2)
        bufs = [a._delay_buffer for a in env.scene["robot"].actuators if getattr(a, "_delay_buffer", None) is not None]
        check("teto do atuador = 2 × decimação 4 = 8 passos de física", all(b.max_lag == 8 for b in bufs),
              str({b.max_lag for b in bufs}))
        vistos = []
        for _ in range(6):
            env.reset()  # o lag é sorteado no 1º passo depois do reset (observacoes.py:50)
            env.step(_zeros(env))
            vistos += [b.current_lags.clone() for b in bufs]
        lags = torch.cat(vistos)
        check("lag do atuador em {0..8} depois do portão", bool(((lags >= 0) & (lags <= 8)).all()),
              f"mín {int(lags.min())}, máx {int(lags.max())}")
        check("o teto subiu: algum lag passa de 4 (o teto antes do portão era 4)", bool((lags > 4).any()))
    finally:
        env.close()


# =============================================================================
# Parte 3 (F8): fatia C, o BOTAR. Os trechos vêm de `cad.tab`; o estado é forçado onde a seção diz.
def s25_fatia_c_forcada_laje_de_destino_a_frente():
    from g1_v3.comando import FATIA_C
    from g1_v3.knobs import ATIVO, TAREFAS

    env = _abre_env(4)
    try:
        env.reset()
        _abre_botar_c(env)
        check("fatia C forçada", bool((env.v3_fatia == FATIA_C).all()))
        check("tarefa publicada = BOTAR", bool((env.v3_tarefa == TAREFAS.index("BOTAR")).all()))
        topo_w = env.scene.env_origins[:, 2] + env.v3_topo_destino
        # ⚠ a laje é mocap: o centro do corpo fica `prateleira_meia_z` abaixo do topo (como em s16)
        laje = env.scene["table"].data
        check("laje de destino: topo = origem + v3_topo_destino (± 1e-4)",
              torch.allclose(laje.root_link_pos_w[:, 2] + ATIVO.cena.prateleira_meia_z, topo_w, atol=1e-4),
              f"{(laje.root_link_pos_w[:, 2] + ATIVO.cena.prateleira_meia_z - topo_w).abs().max().item():.2e}")
        # x no frame da base, pelo rumo do corpo (`heading_w`, o mesmo de `comando._mede`)
        base = env.scene["robot"].data
        dxy = laje.root_link_pos_w[:, :2] - base.root_link_pos_w[:, :2]
        x_base = torch.cos(base.heading_w) * dxy[:, 0] + torch.sin(base.heading_w) * dxy[:, 1]
        check("laje de destino à frente da base (x > 0 no rumo do corpo)", bool((x_base > 0).all()),
              f"x {x_base.tolist()}")
    finally:
        env.close()


def s26_alvo_de_soltura_no_topo_mais_a():
    from g1_v3.alvos import B1

    env = _abre_env(4)
    try:
        env.reset()
        _abre_botar_c(env)
        check("alvo corrente = B1", bool((env.v3_alvo_idx == B1).all()), str(env.v3_alvo_idx.tolist()))
        # ⚠ `_abre_botar` põe o centro da caixa a `a` acima do ponto de soltura, que está no topo da laje
        esperado = env.scene.env_origins[:, 2] + env.v3_topo_destino + env.v3_meia
        check("alvo de soltura: z = origem + v3_topo_destino + v3_meia (± 1e-4)",
              torch.allclose(env.v3_alvo_w[:, 2], esperado, atol=1e-4),
              f"{(env.v3_alvo_w[:, 2] - esperado).abs().max().item():.2e}")
    finally:
        env.close()


def s27_caixa_solta_sobre_a_laje_5_cm_nao_termina_10_cm_termina():
    def roda(altura: float) -> dict:
        """Caixa parada, fundo `altura` acima do topo, sobre o ponto de soltura; ação zero por até 0,3 s."""
        env = _abre_env(4)
        try:
            env.reset()
            _abre_botar_c(env)
            env.v3_degrau[:] = 0  # ⚠ estado forçado: o degrau 0 põe o limite da chegada em 1,0 m/s (início)
            n, dev, dt = env.num_envs, env.device, env.step_dt
            caixa = env.scene["box"]
            # ⚠ `v3_alvo_w` é o centro a `a` acima do topo: o fundo fica `altura` acima do topo se o centro sobe `altura`
            pose = torch.zeros(n, 7, device=dev)
            pose[:, :3] = env.v3_alvo_w
            pose[:, 2] += altura
            pose[:, 3] = 1.0
            _poe_caixa_parada(env, pose)
            env.sim.forward()
            topo_w = env.scene.env_origins[:, 2] + env.v3_topo_destino
            gap0 = caixa.data.root_link_pos_w[:, 2] - env.v3_meia - topo_w
            # queda livre de `altura` (sem contato antes): o toque é no passo em que a caixa percorre `altura`
            g = float(-env.sim.mj_model.opt.gravity[2])
            k_toque = math.ceil(math.sqrt(2 * altura / g) / dt) - 1
            zeros = _zeros(env)
            passos = math.ceil(0.3 / dt)
            cheg, toque, robo = [], [], []
            for _ in range(passos):
                env.step(zeros)
                # ⚠ o termo lê-se pelo `termination_manager` (o valor do último `compute`); depois do passo, o mundo que
                # terminou já foi reiniciado, então as poses só valem antes do disparo
                cheg.append(env.termination_manager.get_term("chegada").clone())
                toque.append(env.scene["apoio_caixa"].data.found.amax(-1) > 0)
                # M3 (F9): o toque do robô é a caixa × subárvore do pelve (`caixa_robo`), e não só as palmas
                robo.append(env.scene["caixa_robo"].data.found.amax(-1) > 0)
            return dict(gap0=gap0, cheg=torch.stack(cheg), toque=torch.stack(toque), robo=torch.stack(robo),
                        k_toque=k_toque, altura=altura)
        finally:
            env.close()

    d5, d10 = roda(0.05), roda(0.10)
    # 5 cm: queda livre de 0,99 m/s, abaixo do limite do degrau 0 (1,0 m/s): não termina
    check("5 cm: fundo a 5 cm do topo", bool(torch.allclose(d5["gap0"], torch.full_like(d5["gap0"], 0.05), atol=1e-4)))
    check("5 cm: a caixa toca a laje nos 0,3 s", bool(d5["toque"].any(dim=0).all()))
    check("5 cm: a chegada nunca dispara (0,99 m/s < 1,0)", not bool(d5["cheg"].any()),
          f"disparos {int(d5['cheg'].sum())}")
    check("5 cm: o robô não toca a caixa antes do toque", not bool(d5["robo"][:d5["k_toque"]].any()))
    # 10 cm: queda livre de 1,40 m/s: dispara no passo do toque (± 2 passos, pelo tempo da queda livre)
    check("10 cm: fundo a 10 cm do topo", bool(torch.allclose(d10["gap0"], torch.full_like(d10["gap0"], 0.10), atol=1e-4)))
    primeiro = d10["cheg"].float().argmax(dim=0)
    check("10 cm: a chegada dispara em todo mundo (1,40 m/s)", bool(d10["cheg"].any(dim=0).all()))
    # Veredito F9: a conta dá o passo exato (`k_toque`); a folga de ± 2 passos é só para o arredondamento do disparo.
    check("10 cm: o disparo é no passo do toque (± 2 passos da queda livre)",
          bool(((primeiro - d10["k_toque"]).abs() <= 2).all()), f"primeiro {primeiro.tolist()}, toque {d10['k_toque']}")
    check("10 cm: o robô não toca a caixa antes do toque", not bool(d10["robo"][:d10["k_toque"]].any()))


def s28_one_hot_fica_em_botar_durante_o_atraso():
    from g1_v3.knobs import ATIVO, TAREFAS

    env = _abre_env(4)
    try:
        env.reset()
        cad = _abre_botar_c(env)
        n, dt, dev = env.num_envs, env.step_dt, env.device
        i_botar = TAREFAS.index("BOTAR")
        atraso_max = ATIVO.cadeia.atraso_botar_s[1]
        margem = 2 * dt  # Veredito F9: a margem de 2 passos serve; o defeito da s28 é a base de 0,5 s (M1), não a margem
        zeros = _zeros(env)
        aviso0 = torch.full((n,), -1, dtype=torch.long, device=dev)  # passo do primeiro aviso, −1 = não veio
        troca = torch.full((n,), -1, dtype=torch.long, device=dev)  # passo em que o trecho sai do BOTAR, −1 = não saiu
        ok = torch.ones(n, dtype=torch.bool, device=dev)
        # ⚠ para quando todo mundo trocou, ou no tempo máximo (aviso em 1,5 s + atraso máximo + margem)
        for p in range(math.ceil((1.5 + atraso_max + margem) / dt)):
            env.step(zeros)
            aviso0 = torch.where((aviso0 < 0) & env.v3_aviso, torch.full_like(aviso0, p), aviso0)
            # depois do primeiro aviso e antes da troca, o one-hot e a tarefa ficam em BOTAR; o passo da troca é o próprio
            pre = (aviso0 >= 0) & (troca < 0)
            botar = env.v3_tarefa == i_botar
            troca = torch.where((troca < 0) & ~botar, torch.full_like(troca, p), troca)
            ok &= ~pre | (botar & (cad.command[:, i_botar] > 0.5)) | (troca == p)
            if bool((troca >= 0).all()):
                break
        t_aviso = (aviso0 + 1).float() * dt
        t_troca = (troca + 1).float() * dt
        # ⚠ o aviso é a pose de pé sem caixa com 0,5 s de estabilidade: não pode vir antes disso (± 1 passo)
        check("aviso do BOTAR em até 1,5 s (a pose de pé sem caixa)", bool(((aviso0 >= 0) & (t_aviso <= 1.5)).all()),
              f"aviso em {t_aviso.tolist()} s")
        check("aviso não antes de 0,5 s de pose estável", bool((t_aviso >= 0.5 - dt).all()), f"{t_aviso.tolist()} s")
        # M1 (F9): o limite conta do aviso MEDIDO (o primeiro passo com v3_aviso), e não de 0,5 s fixos
        atraso = t_troca - t_aviso
        check("trecho sai do BOTAR até atraso máximo + margem depois do aviso",
              bool(((troca >= 0) & (aviso0 >= 0) & (atraso <= atraso_max + margem)).all()),
              f"troca {atraso.tolist()} s depois do aviso, limite {atraso_max + margem:.2f} s")
        check("tarefa e one-hot em BOTAR de todo passo do aviso até a troca", bool(ok.all()))
    finally:
        env.close()


def s29_andar_parado_final_acaba_em_time_out():
    from g1_v3.comando import FATIA_C
    from g1_v3.knobs import ATIVO

    env = _abre_env(4)
    try:
        env.reset()
        # ⚠ estado forçado: o ANDAR parado final da fatia C (o último trecho ANDAR da tabela), aberto direto
        _abre_trecho_de(env, "ANDAR", FATIA_C, ultimo=True)
        n, dt = env.num_envs, env.step_dt
        zeros = _zeros(env)
        fim = torch.full((n,), -1, dtype=torch.long, device=env.device)
        queda_no_fim = torch.zeros(n, dtype=torch.bool, device=env.device)
        for p in range(math.ceil(2 * ATIVO.cadeia.andar_final_s / dt)):
            env.step(zeros)
            # ⚠ `time_outs` é o valor do último `compute`; a `queda` lê-se no mesmo passo
            novo = env.termination_manager.time_outs & (fim < 0)
            fim = torch.where(novo, torch.full_like(fim, p), fim)
            queda_no_fim |= novo & env.termination_manager.get_term("queda")
            if bool((fim >= 0).all()):
                break
        check("ANDAR final: time_out em todo mundo", bool((fim >= 0).all()), f"passos {fim.tolist()}")
        t_fim = (fim + 1).float() * dt
        # ⚠ o v3_fim sai no `compute` do passo k e o time_out aparece no passo k + 1: a tolerância é de 2 passos
        check("time_out no fim do ANDAR (andar_final_s ± 2 passos)",
              bool(((t_fim - ATIVO.cadeia.andar_final_s).abs() <= 2 * dt).all()), f"{t_fim.tolist()} s")
        check("queda False no passo do time_out", not bool(queda_no_fim.any()))
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
    ("16 fatia B forçada: topo da laje e caixa sobre ela", s16_fatia_b_forcada_topo_da_laje_e_caixa_sobre_ela),
    ("17 abertura do PEGAR: alvo em P0, σ e pega", s17_abertura_do_pegar_poe_alvo_em_p0_e_sigma),
    ("18 PEGAR com ação zero: nada paga positivo", s18_pegar_com_acao_zero_nada_paga_positivo),
    ("19 caixa no chão: paga uma vez e vira ANDAR; sem a chave, termina",
     s19_caixa_no_chao_paga_uma_vez_e_vira_andar_com_a_chave),
    ("20 estimador vê a caixa à frente; idade cresce com os braços", s20_estimador_ve_caixa_a_frente_e_idade_cresce_com_bracos),
    ("21 pose de pé com caixa: aviso depois de 0,5 s", s21_pose_de_pe_com_caixa_dispara_aviso_depois_de_meio_segundo),
    ("22 laje sai com o CARREGAR que anda", s22_laje_sai_com_o_carregar_que_anda),
    ("23 tamanho do crítico confere com §6.5", s23_tamanho_do_critico_confere_com_6_5),
    ("24 atraso do atuador em {0..8} depois do portão do PEGAR", s24_atraso_do_atuador_depois_do_portao_do_pegar),
    ("25 fatia C forçada: laje de destino à frente no BOTAR", s25_fatia_c_forcada_laje_de_destino_a_frente),
    ("26 alvo de soltura no topo + a", s26_alvo_de_soltura_no_topo_mais_a),
    ("27 caixa de 5 cm não termina; de 10 cm termina", s27_caixa_solta_sobre_a_laje_5_cm_nao_termina_10_cm_termina),
    ("28 one-hot fica em BOTAR durante o atraso", s28_one_hot_fica_em_botar_durante_o_atraso),
    ("29 ANDAR parado final acaba em time_out", s29_andar_parado_final_acaba_em_time_out),
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
