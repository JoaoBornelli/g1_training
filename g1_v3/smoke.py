"""Os checks da v3 (D12 e E10 do plano, seções 1 a 39: Partes 1 a 4).

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


def _segura_base(env, pose):
    """Estado forçado: a base na `pose` (E,7) com velocidade zero, antes de cada passo.

    ⚠ Com o `Contrato.pd` (tornozelo kp 40) a ação zero NÃO segura o G1 de pé: sem DR ele cai em ~1,1 s (tronco a
    38° no passo 50, `queda` no 52; triagem da fase I). As seções que medem outra coisa com o robô parado seguram a base.
    """
    robo = env.scene["robot"]
    robo.write_root_link_pose_to_sim(pose)
    robo.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))


def _poe_curriculo(env, degrau, travas=()):
    """Põe o degrau das 4 tarefas (um int, ou a tupla na ordem de `balanceador.TAREFAS`) e as travas do DR, pelo
    `estado()` do currículo (§6.8). Quem chama dá o `env.reset()`: o reset publica o degrau da tarefa
    (comando.py:450-461) e os eventos de reset leem o `dr()` novo. Exige um reset antes (o `v3_cur` nasce nele)."""
    from g1_v3.balanceador import TAREFAS
    cur = env.v3_cur
    estado = cur.estado()
    degraus = degrau if isinstance(degrau, tuple) else (degrau,) * len(TAREFAS)
    for t, d in zip(TAREFAS, degraus):
        estado["tarefas"][t]["degrau"] = d
    for k in ("portao0", "vies", "subida"):
        estado["dr"][k] = k in travas
    cur.carrega(estado)


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
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{jn}")  # o modelo prefixa a entidade
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

        # parado: a base segura (S0); a ação zero sozinha não segura o G1 de pé
        pose0 = robo.data.root_link_pose_w.clone()
        for _ in range(20):
            _segura_base(env, pose0)
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
            _segura_base(env, pose)  # ⚠ sem a velocidade zero o robô cai no passo 33–38 e o reset zera o mapa
            env.sim.forward()
            env.step(_zeros(env))
        mapa = env.v3_mapa_obs
        h, conf = mapa[:, :441], mapa[:, 441:]
        vista = conf > p.limiar_visto
        frac = vista[:, frente].float().mean(dim=1)
        check("≥ 90% das células da metade da frente vistas (I §2.4: 205 de 210)",
              bool((frac[plano] >= 0.9).all()), f"mínimo {frac[plano].min().item():.3f}")
        # ⚠ o ruído do lidar está ligado (σ_alcance 2 cm; 0,5° a 2–3 m): ±2 cm por célula é 1σ. O σ de cada célula sai
        # da confiança (PD-6: c = σ_ref²/(σ_ref² + P)); a média fica nos ±2 cm do plano. Fase I: média 0,9 cm, máx 1,5σ.
        sig = (p.sigma_ref ** 2 * (1 / conf.clamp_min(1e-6) - 1)).sqrt()
        hv, vv = h[:, frente], vista[:, frente]
        dentro = ((hv.abs() <= 3 * sig[:, frente]) | ~vv).all(dim=1)
        media = (hv.abs() * vv).sum(dim=1) / vv.sum(dim=1).clamp_min(1)
        check("alturas das células vistas: |h| ≤ 3σ da célula e média ≤ 2 cm",
              bool((dentro & (media <= 0.02))[plano].all()), f"média {media.tolist()}")
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
        # ⚠ o manager instancia os termos de classe (`sacudida`) numa CÓPIA do cfg; o `env.cfg.rewards` guarda a classe
        for nome in env.reward_manager.active_terms:
            termo = env.reward_manager.get_term_cfg(nome)
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
            _poe_base(env, pose)
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
    from g1_v3.knobs import ATIVO
    env.v3_fatia[:] = FATIA_B
    ids = torch.arange(env.num_envs, device=env.device)
    eventos.posiciona_cena(env, ids)
    # ⚠ como o reset do treino: a base atrás da caixa (`_base_da_pega`, E11 S4). Sem isto ela fica onde o ANDAR a
    # sorteou (±0,5 m, rumo ±π), às vezes contra a laje
    eventos.reset_base_por_fatia(env, ids, dict(ATIVO.cena.reset_base_andar), dict(ATIVO.cena.reset_base_manipula))
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
        # F9 P6 (E11 S5b): o P0 abre em `_abre_p0`, que só o `compute` chama; sem ele o `cad.pega` não sobe.
        m = cad._mede()
        cad._abre_p0(m)
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
        pose0 = env.scene["robot"].data.root_link_pose_w.clone()
        _abre_trecho_de(env, "PEGAR")  # ⚠ força o trecho de PEGAR; o `avanca_trecho` pode sair dele antes de 50 passos
        zeros = _zeros(env)
        liquido, outros, fora = torch.zeros(env.num_envs, device=env.device), {}, 0
        for _ in range(50):
            _segura_base(env, pose0)
            env.step(zeros)
            fora += int((env.v3_tarefa != TAREFAS.index("PEGAR")).sum())
            for nome, v in _por_termo(env).items():
                if nome == "progresso":
                    liquido += v.to(liquido.device)
                elif float(v.clamp(min=0.0).sum()) > 0:
                    outros[nome] = outros.get(nome, 0.0) + float(v.clamp(min=0.0).sum())
        check("50 passos no PEGAR (a precondição da seção)", fora == 0, f"{fora} mundo-passo fora do PEGAR")
        check("nenhum termo além do `progresso` paga positivo (ação zero)", not outros, str(outros))
        # ⚠ A renda por segundo (10/11) paga `renda_phi`·Φ enquanto há alvo: pairar no P0 paga > 0 (a soma antiga ≤ 0
        # era do potencial com γ). O teste só vale com o P0 aberto; sem leitura válida o alvo é −1 e a renda é 0.
        check("progresso paga renda > 0 em cada mundo (ação zero, base segura, P0 aberto)", bool((liquido > 0).all()),
              str(liquido.tolist()))
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
    check("chave desligada: termina um passo depois da perda (caso 4)", bool(off["termina"].all()))
    check("cfg tem o termo `caixa_perdida` (S §10: sem a chave, termina)", on["cfg_tem"] and off["cfg_tem"])


def s20_estimador_ve_caixa_a_frente_e_idade_cresce_fora_do_fov():
    from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
    from g1_v3.knobs import ATIVO

    env = _abre_env(4)
    try:
        env.reset()
        robo, caixa = env.scene["robot"], env.scene["box"]
        dev, n = env.device, env.num_envs
        # caixa 0,45 m à frente da base, na altura do eixo óptico da D435i (frame da base). A câmera fica em
        # `d435_pos` = (0,058; 0,018; 0,430) m (torso ≈ pelve, ⚠ DÚVIDA), com pitch 47,6°: o eixo óptico passa por
        # (0,45; 0; 0,0), e a caixa cai no centro do FOV (vertical ±21°; `Estimador.fov_*`).
        ofs = torch.tensor([0.45, 0.0, 0.0], device=dev)
        atras = torch.tensor([-0.45, 0.0, 0.0], device=dev)  # fora do FOV da D435i

        def repoe_caixa(o):
            # ⚠ força: a caixa é reposta a cada passo, senão cairia (0,4 s de queda livre são 0,8 m) antes do estimador
            pose = torch.zeros(n, 7, device=dev)
            pose[:, :3] = robo.data.root_link_pos_w + quat_apply(robo.data.root_link_quat_w, o.expand(n, 3))
            pose[:, 3] = 1.0
            caixa.write_root_link_pose_to_sim(pose)
            caixa.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev))

        zeros = _zeros(env)
        pose0 = robo.data.root_link_pose_w.clone()
        for _ in range(20):
            _segura_base(env, pose0)
            repoe_caixa(ofs)
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
        # ⚠ fase I: em `q_com_caixa` os braços NÃO tapam a câmera (palmas ao lado da caixa, 0–2 dos 9 pontos
        # ocluídos). A idade se mede sem leitura: a caixa atrás da base por 0,6 s.
        idade0 = s[:, 7].clone()
        for _ in range(30):
            _segura_base(env, pose0)
            repoe_caixa(atras)
            env.step(zeros)
        s2 = env.v3_estimador.saida_b(env.v3_odo, robo.data.root_link_quat_w)
        check("caixa fora do FOV: a idade cresce ≥ 0,4 s em 0,6 s", bool(((s2[:, 7] - idade0) >= 0.4).all()),
              f"Δ {(s2[:, 7] - idade0).tolist()}")
    finally:
        env.close()


def s21_pose_de_pe_com_caixa_dispara_aviso_depois_de_meio_segundo():
    from g1_v3.knobs import ATIVO, JUNTAS, TAREFAS, escala_por_junta, q_de_pe

    env = _abre_env(4)
    try:
        env.reset()
        _abre_cargar_no_curriculo(env)  # M1: o trecho que vem depois do aviso é o CARREGAR; a célula não pode estar fechada
        _le_caixa_valida(env)  # S5b: o PEGAR abre depois da primeira leitura válida (estado forçado)
        cad = _abre_trecho_de(env, "PEGAR")
        # M1: o trecho acaba pelo aviso (por_detector; 13/11, M6: o fecho do P4), e não pelo atraso sorteado em U[0; 1] s, que
        # venceria em ~99% dos mundos. ⚠ `inf` segura o trecho no PEGAR durante os 60 passos.
        cad.alvo_s[:] = float("inf")
        robo, dev, a = env.scene["robot"], env.device, env.v3_meia
        q = env.v3_ref.q_com_caixa(a).to(dev, torch.float32)  # ⚠ a ordem é a de JUNTAS (s04)
        qde = torch.as_tensor(q_de_pe(ATIVO.contrato), dtype=torch.float32, device=dev)
        # ⚠ força a pose de pé com caixa: braços em q_com_caixa e a pelve na altura de `pelve_com_caixa`.
        # ⚠ Limite conhecido (I §4.2; 13/11, M6): o aviso do PEGAR é o fecho do P4 (`v3_p4_fechado`), que pede a caixa a
        # 5 cm do transporte e a pelve a 3 cm por 0,5 s, e não mais o detector de pose (alvos.detecta_fim, só do pilota_v3).
        robo.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.sim.forward()
        env.sim.sense()
        pose = robo.data.root_link_pose_w.clone()
        pose[:, 2] += env.v3_ref.pelve_com_caixa(a).to(dev) - env.scene["altura_pelve"].data.heights[:, 0]
        robo.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        env.sim.sense()
        # ⚠ o alvo do PD é q_de_pé + escala·a − viés (s04): a ação soma o viés para o alvo ser q_com_caixa
        idx = [robo.joint_names.index(j) for j in JUNTAS]
        acao = (q - qde + robo.data.encoder_bias[:, idx]) / torch.tensor(escala_por_junta(), device=q.device, dtype=q.dtype)
        avisos = []
        for _ in range(60):
            # ⚠ fase I: a ação zero não segura a base (S0), e a cintura (kp 40) não segura os braços à frente
            # (tronco +8° em 0,36 s). A seção mede o aviso do trecho: base e juntas reescritas a cada passo.
            _segura_base(env, pose)
            robo.write_joint_state_to_sim(q, torch.zeros_like(q))
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
    from g1_v3 import balanceador
    from g1_v3.alvos import B1
    from g1_v3.knobs import ATIVO

    env = _abre_env(4)
    try:
        env.reset()
        _abre_botar_c(env)
        check("alvo corrente = B1", bool((env.v3_alvo_idx == B1).all()), str(env.v3_alvo_idx.tolist()))
        # ⚠ `_abre_botar` põe o centro da caixa a `a` acima do ponto de soltura, que está no topo da laje
        esperado = env.scene.env_origins[:, 2] + env.v3_topo_destino + env.v3_meia
        # ⚠ P22: o `_abre_botar` soma o viés do alvo U[±vz] (vz pelo degrau global; 5 mm no degrau 0)
        vz = balanceador.no_degrau(*ATIVO.estimador.vies_alvo_botar_z, env.v3_cur.degrau_global(),
                                   ATIVO.balanceador.degraus)
        dz = (env.v3_alvo_w[:, 2] - esperado).abs()
        check("alvo de soltura: z = origem + v3_topo_destino + v3_meia (± viés do alvo, P22)",
              bool((dz <= vz + 1e-4).all()), f"{dz.max().item():.2e} > {vz:.3f}")
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
            # ⚠ P22: o `v3_alvo_w` traz o viés do alvo (até 5 mm em z no degrau 0): o z sai do topo verdadeiro
            pose[:, :2] = env.v3_alvo_w[:, :2]
            pose[:, 2] = env.scene.env_origins[:, 2] + env.v3_topo_destino + env.v3_meia + altura
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
        pose0 = env.scene["robot"].data.root_link_pose_w.clone()
        for p in range(math.ceil((1.5 + atraso_max + margem) / dt)):
            _segura_base(env, pose0)  # S0: a ação zero cai em ~1,1 s, antes do aviso
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
        pose0 = env.scene["robot"].data.root_link_pose_w.clone()
        for p in range(math.ceil(2 * ATIVO.cadeia.andar_final_s / dt)):
            _segura_base(env, pose0)  # S0: sem isto a `queda` (~1,1 s) chega antes do fim do ANDAR de 1,5 s
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


# =============================================================================
# Parte 3b (G5): CARREGAR no terreno. A fatia B é forçada numa escada, na linha mais alta que `linha_max_carregar`
# permite. A origem do sub-terreno é o topo da plataforma (mjlab primitive_terrains.py:233-236, a caixa do meio).
def _abre_b_na_escada(env) -> float:
    """Fatia B forçada numa escada na linha de `linha_max_carregar`, com a cena e a base postas. Devolve o degrau (m).

    ⚠ Estado forçado, como o `sorteia_fatia` escreve: nível, tipo e origem do sub-terreno. Quem chama faz o `reset` antes.
    ⚠ O degrau da linha é `tetos_andar.degrau × linha/(linhas−1)`: o mjlab põe `difficulty = linha/(num_rows−1)`
       (terrain_generator.py:253-258) e `step = 0 + difficulty × teto` (primitive_terrains.py:79-81). Na linha 9: 0,15 m.
    """
    from g1_v3 import eventos
    from g1_v3.comando import FATIA_B
    from g1_v3.knobs import ATIVO

    t, k = ATIVO.terreno, ATIVO.cena
    linha = dict(t.linha_max_carregar)["escada"]
    coluna = t.colunas.index("escada")
    ids = torch.arange(env.num_envs, device=env.device)
    ter = env.scene["terrain"]
    ter.terrain_levels[ids] = linha
    ter.terrain_types[ids] = coluna
    env.scene.env_origins[ids] = ter.terrain_origins[linha, coluna]  # como `sorteia_fatia` (eventos.py:146-148)
    env.v3_fatia[:] = FATIA_B
    eventos.posiciona_cena(env, ids)
    eventos.reset_base_por_fatia(env, ids, dict(k.reset_base_andar), dict(k.reset_base_manipula))
    env.sim.forward()  # ⚠ a leitura das poses depois da escrita pede o forward (como em s06)
    return dict(t.tetos_andar)["degrau"] * linha / (t.linhas - 1)


def _geometria_escada():
    """(n degraus, step_width, meia plataforma do topo) da escada do cfg, pelo cálculo do `pyramid_stairs`.

    ⚠ primitive_terrains.py:88-100: n = int((lado − 2·border − plataforma)/(2·step_width)); o topo tem
       `lado − 2·border − 2·n·step_width` de lado, e o primeiro anel, logo fora dele, tem `step_width` de largura.
       Linha 9: n 6, meia plataforma 1,20 m (lado 8, border 1,0, plataforma 2,0).
    """
    from g1_v3.cena import terreno_cfg

    g = terreno_cfg().terrain_generator
    e = g.sub_terrains["escada"]
    lado = g.size[0]
    n = int((lado - 2 * e.border_width - e.platform_width) / (2 * e.step_width))
    return n, e.step_width, (lado - 2 * e.border_width - 2 * n * e.step_width) / 2


def _poe_base(env, pose):
    """Põe a base na `pose` (n, 7) com velocidade zero, e roda `forward` e `sense` (M8: a escrita da s10 e a do _base_alem).

    ⚠ o raycast das alturas só atualiza no `sense` (raycast_sensor.py:691).
    """
    robo = env.scene["robot"]
    robo.write_root_link_pose_to_sim(pose)
    robo.write_root_link_velocity_to_sim(torch.zeros(pose.shape[0], 6, device=pose.device))
    env.sim.forward()
    env.sim.sense()


def _base_alem_da_borda(env, z):
    """Põe a base parada além da borda da escada, no rumo da cena, a `z` (E,); roda `forward` e `sense`.

    ⚠ 1,35 m do centro = a meia plataforma do topo (1,20 m) + meio degrau (`_geometria_escada`, do cfg). O primeiro
       anel fica um degrau abaixo da plataforma (primitive_terrains.py:133-135).
    """
    _, sw, meia = _geometria_escada()
    robo = env.scene["robot"]
    rumo = env.v3_rumo_cena
    alem = meia + sw / 2
    pose = robo.data.root_link_pose_w.clone()
    pose[:, :2] = env.scene.env_origins[:, :2] + alem * torch.stack([torch.cos(rumo), torch.sin(rumo)], -1).float()
    pose[:, 2] = z
    _poe_base(env, pose)
    return pose


def _caixa_no_peito(env, pose):
    """Põe a caixa no peito da base `pose` (n,7): `caixa_transporte_b` (a IK) levado ao mundo, velocidade zero."""
    from mjlab.utils.lab_api.math import quat_apply

    caixa_b = env.v3_ref.caixa_transporte_b(env.v3_meia).to(pose.dtype)
    alvo = torch.zeros(pose.shape[0], 7, device=pose.device)
    alvo[:, :3] = pose[:, :3] + quat_apply(pose[:, 3:7], caixa_b)
    alvo[:, 3] = 1.0
    _poe_caixa_parada(env, alvo)
    env.sim.forward()
    env.sim.sense()


def s30_env_da_fatia_b_no_terreno_nasce_na_plataforma():
    from g1_v3.estimador import yaw
    from g1_v3.knobs import ATIVO

    env = _abre_env(4)
    try:
        env.reset()
        _abre_b_na_escada(env)
        base, origem = env.scene["robot"].data, env.scene.env_origins[:, :2]
        d_base = (base.root_link_pos_w[:, :2] - origem).abs().amax(-1)
        check("base dentro do quadrado da plataforma (‖xy da base − origem‖∞ < plataforma/2)",
              bool((d_base < ATIVO.terreno.plataforma / 2).all()), f"{d_base.tolist()}")
        rumo = env.v3_rumo_cena
        laje = env.scene["table"].data.root_link_pos_w[:, :2] - origem
        L = ATIVO.terreno.laje_do_centro
        check("centro da laje a laje_do_centro do centro (± 1e-3)",
              bool(((laje.norm(dim=-1) - L).abs() <= 1e-3).all()), f"{laje.norm(dim=-1).tolist()}")
        na_direcao = torch.stack([torch.cos(rumo), torch.sin(rumo)], -1) * L
        check("centro da laje na direção de v3_rumo_cena (± 1e-3)", bool(torch.allclose(laje, na_direcao, atol=1e-3)))
        lo, hi = dict(ATIVO.cena.reset_base_manipula)["yaw"]
        d_yaw = torch.remainder(yaw(base.root_link_quat_w) - rumo + math.pi, 2 * math.pi) - math.pi
        check("rumo do robô (estimador.yaw) para fora, na faixa yaw da base",
              bool(((d_yaw >= lo - 1e-3) & (d_yaw <= hi + 1e-3)).all()), f"{d_yaw.tolist()}")
    finally:
        env.close()


def s31_topo_da_laje_e_origem_de_escada_alta():
    from g1_v3.knobs import ATIVO

    env = _abre_env(4)
    try:
        env.reset()
        degrau = _abre_b_na_escada(env)
        n, _, _ = _geometria_escada()
        origem_z = env.scene.env_origins[:, 2]
        print(f"  origem z na linha mais alta da escada: {origem_z.tolist()}")
        # primitive_terrains.py:233-236: a origem fica (n + 1) degraus acima do chão do sub-terreno
        check(f"origem de escada alta: z = (n + 1)·degrau ± 1e-3 (n {n}, degrau {degrau:.3f} m)",
              bool(((origem_z - (n + 1) * degrau).abs() <= 1e-3).all()), f"{origem_z.tolist()}")
        topo_w = origem_z + env.v3_topo
        laje_z = env.scene["table"].data.root_link_pos_w[:, 2]
        check("laje: topo = origem + v3_topo (± 1e-4)",
              torch.allclose(laje_z + ATIVO.cena.prateleira_meia_z, topo_w, atol=1e-4),
              f"{(laje_z + ATIVO.cena.prateleira_meia_z - topo_w).abs().max().item():.2e}")
    finally:
        env.close()


def s32_robo_um_degrau_abaixo_com_caixa_no_peito_nao_perde_a_caixa():
    from g1_v3.knobs import ATIVO

    env = _abre_env(4)
    try:
        env.reset()
        degrau = _abre_b_na_escada(env)
        _abre_cargar_no_curriculo(env)  # M1: o twist sorteia célula do CARREGAR; fechada, `multinomial` falha
        _abre_trecho_de(env, "CARREGAR")
        z = env.scene["robot"].data.root_link_pos_w[:, 2] - degrau  # um degrau abaixo da base de hoje
        pose = _base_alem_da_borda(env, z)
        _caixa_no_peito(env, pose)
        print(f"  origem z {env.scene.env_origins[:, 2].tolist()}, base z {pose[:, 2].tolist()}, degrau {degrau:.3f} m")
        # ⚠ `step` devolve (obs, rew, terminated, truncated, info), um bool por mundo. Se um mundo terminou ou truncou
        # aqui, o reset dele já rodou antes da cadeia: `v3_perdeu_agora` vira o do episódio novo, não o do passo testado.
        _o, _r, term, trunc, _i = env.step(_zeros(env))  # UM passo, ação zero (a caixa não está presa às mãos)
        passo_limpo = not bool(term.any()) and not bool(trunc.any())
        check("nenhum mundo terminou nem truncou no passo (senão os checks abaixo não valem)", passo_limpo,
              f"terminou {term.tolist()}, truncou {trunc.tolist()}")
        if not passo_limpo:
            return
        if ATIVO.cadeia.caixa_perdida_vira_andar:
            # Com a chave ligada, `n_caixa_perdida` é sempre False: o termo não se aplica (nem ok, nem falha)
            print("  `caixa_perdida` não se aplica: caixa_perdida_vira_andar ligada (a perda vira ANDAR, sem termo)")
        else:
            perdeu = env.termination_manager.get_term("caixa_perdida")
            check("um degrau abaixo, caixa no peito: `caixa_perdida` False (D4)", not bool(perdeu.any()),
                  f"{perdeu.tolist()}")
        # ⚠ o pulso do PASSO (a cadeia o escreve depois da terminação), não o `v3_perdeu` do reset (M1)
        check("um degrau abaixo, caixa no peito: v3_perdeu_agora False no passo", not bool(env.v3_perdeu_agora.any()),
              f"{env.v3_perdeu_agora.tolist()}")
    finally:
        env.close()


def s33_altura_da_pelve_mede_contra_o_terreno_local():
    from g1_v3 import terminacoes
    from g1_v3.env_cfg import TODAS

    env = _abre_env(4)
    try:
        env.reset()
        degrau = _abre_b_na_escada(env)
        _abre_cargar_no_curriculo(env)
        _abre_trecho_de(env, "CARREGAR")
        pelve = env.v3_ref.pelve_de_pe
        origem_z = env.scene.env_origins[:, 2]
        z_degrau = origem_z - degrau  # chão do anel, um degrau abaixo do topo da plataforma (origem)
        pose = _base_alem_da_borda(env, z_degrau + pelve)  # a base de pé sobre o degrau
        _caixa_no_peito(env, pose)
        sp = env.scene["altura_pelve"].data
        alt = sp.heights[:, 0]
        print(f"  origem z {origem_z.tolist()}, z do degrau {z_degrau.tolist()}, "
              f"chão medido pelo sensor {(sp.frame_pos_w[:, 0, 2] - alt).tolist()}")
        check("altura da pelve (sensor) = pelve_de_pe ± 5 cm: mede contra o terreno local (D3, D4)",
              bool(((alt - pelve).abs() <= 0.05).all()), f"{alt.tolist()}")
        sj = env.scene["altura_joelhos"].data  # (E,2): o chão medido por cada joelho tem de ser o do degrau
        chao_j = sj.frame_pos_w[:, :, 2] - sj.heights
        check("altura do joelho (sensor): chão medido = z do degrau ± 5 cm (contra o terreno local)",
              bool(((chao_j - z_degrau[:, None]).abs() <= 0.05).all()), f"{chao_j.tolist()}")
        check("queda False com a base de pé sobre o degrau", not bool(terminacoes.queda(env, TODAS).any()))
    finally:
        env.close()


# =============================================================================
# Parte 4 (H): recompensas, terminações e DR do refino (H1 a H6). O degrau é forçado pelo `estado()` do currículo.
def s34_degrau_5_pesa_laje_chegada_e_pouso():
    from g1_v3 import recompensas
    from g1_v3.alvos import P3
    from g1_v3.knobs import ATIVO

    R, T = ATIVO.recompensa, ATIVO.terminacao
    env = _abre_env(4)
    try:
        env.reset()
        # M2: o PEGAR aberto com a caixa presa (alvo em P3, máscara aberta); só o ANDAR zera a sacudida e o aperto
        _assenta_fatia_b(env)
        _le_caixa_valida(env)
        _abre_trecho_de(env, "PEGAR")
        env.v3_alvo_idx[:] = P3
        env.v3_degrau[:] = 5  # ⚠ depois da abertura: o `_publica` escreve o degrau do currículo (0) no trecho novo
        for nome, par, esperado in (("laje_joelho_N", R.laje_joelho_N, 10.0), ("chegada", T.chegada, 0.25),
                                    ("sigma_pouso", R.sigma_pouso, 0.35)):
            v = recompensas.por_degrau(env, *par)
            check(f"por_degrau({nome}) = {esperado} no degrau 5", bool(torch.allclose(v, torch.full_like(v, esperado), atol=1e-6)),
                  f"{v.tolist()}")
        for _ in range(5):
            env.step(_zeros(env))
        vals = _por_termo(env)
        for nome in ("sacudida", "aperto"):
            # ⚠ o valor do `reward_manager` é peso × custo: o custo sai dividindo pelo peso (negativo)
            peso = env.cfg.rewards[nome].weight
            custo = vals[nome] / peso
            check(f"{nome}: peso negativo", peso < 0, str(peso))
            check(f"{nome}: custo finito e ≥ 0 depois de 5 passos parados", bool(torch.isfinite(custo).all() and (custo >= 0).all()),
                  f"{custo.tolist()}")
    finally:
        env.close()


def s35_batente_so_com_degrau_1_e_contagem_por_junta():
    from g1_v3 import potencial, terminacoes
    from g1_v3.env_cfg import TODAS
    from g1_v3.knobs import ATIVO, JUNTAS

    env = _abre_env(16)
    try:
        env.reset()
        robo = env.scene["robot"]
        lim = robo.data.joint_pos_limits  # (E,29,2) do modelo
        j = robo.joint_names.index("left_knee_joint")
        meio = (lim[:, j, 1] - lim[:, j, 0]) / 2
        q = robo.data.joint_pos.clone()
        q[:, j] = lim[:, j, 1] + 0.02 * meio  # 2 % do meio curso além do limite de cima: curso 1,02
        robo.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.sim.forward()
        env.sim.sense()
        env.v3_degrau[:] = 0
        check("batente desligado no degrau 0", not bool(terminacoes.batente(env, TODAS).any()))
        env.v3_degrau[:] = 1
        check("batente ligado no degrau 1", bool(terminacoes.batente(env, TODAS).all()))

        # MEDIDA (sem check): 200 passos aleatórios com o degrau 1 da tarefa; conta, por junta, os passos além do batente.
        # ⚠ P13: se um tornozelo dominar, o ajuste por família é decisão do dono (a S §15 pede ≤ 5 % de terminação nova).
        _poe_curriculo(env, 1)
        env.reset()
        contagem = torch.zeros(robo.data.joint_pos.shape[1], device=env.device)
        for _ in range(200):
            env.step(torch.rand(16, len(JUNTAS), device=env.device) * 2 - 1)
            lim = robo.data.joint_pos_limits
            curso = potencial.curso(robo.data.joint_pos, (lim[..., 0] + lim[..., 1]) / 2, (lim[..., 1] - lim[..., 0]) / 2)
            contagem += (curso > ATIVO.terminacao.batente_frac).float().sum(dim=0)
        print("  passos além do batente, por junta (de 16 × 200 = 3 200 por junta):")
        for nome, c in zip(robo.joint_names, contagem.tolist()):
            if c > 0:
                print(f"    {nome}: {int(c)}")
    finally:
        env.close()


def _geoms_do_solref(env):
    """(geoms que `eventos.solref` escreve, geoms do terreno). As primeiras vêm das três `asset_cfg` do evento do cfg
    (P17, mesmo conjunto que o evento usa); as do terreno, pelo regex do mjlab (`terrain$`)."""
    import mujoco

    # ⚠ o manager resolve uma CÓPIA do cfg: no `env.cfg.events` o `geom_ids` fica slice(None) (as 72 geoms do robô)
    ev = env.event_manager.get_term_cfg("solref")  # P17: o evento do manager (env_cfg.py:170)
    gid = torch.cat([env.scene[c.name].indexing.geom_ids[c.geom_ids]
                     for c in (ev.params["asset_cfg_robo"], ev.params["asset_cfg_laje"], ev.params["asset_cfg_caixa"])])
    m = env.sim.mj_model
    # ⚠ as geoms do terreno se chamam terrain_0 … terrain_N; com `terrain$` o tid ficava vazio e o check, vazio
    tid = [i for i in range(m.ngeom) if re.fullmatch(r"terrain_\d+", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]
    return gid, torch.tensor(tid, dtype=torch.long)


def s36_dr_do_refino_zero_no_degrau_0_e_faixa_no_5():
    from g1_v3.knobs import TAREFAS, TAU_MAX

    env = _abre_env(4)
    try:
        env.reset()
        robo = env.scene["robot"]
        ctrl_de = [a.global_ctrl_ids for a in robo.actuators]  # 9 grupos; os grupos 7 e 8 são os braços (P18)
        todos = torch.cat(ctrl_de)
        gid, tid = _geoms_do_solref(env)

        def razao(campo, ctrl, col):  # (E, n): o campo do modelo de cada mundo sobre o default
            return getattr(env.sim.model, campo)[:, ctrl, col] / env.sim.get_default_field(campo)[ctrl, col]

        # degrau global 0: tudo no default (H1: `pares(ini, fim)` dá o início)
        for nome, campo, col in (("kp", "actuator_gainprm", 0), ("kd", "actuator_biasprm", 2)):
            r = razao(campo, todos, col)
            check(f"degrau 0: {nome} = default", bool(torch.allclose(r, torch.ones_like(r), atol=1e-6)))
        check("degrau 0: forcerange = default (9 grupos)",
              all(bool(torch.allclose(r, torch.ones_like(r), atol=1e-6))
                  for r in (razao("actuator_forcerange", c, 1) for c in ctrl_de)))
        sr = env.sim.model.geom_solref[:, gid, 0]
        check("degrau 0: solref das geoms de contato = 0,02", bool(torch.allclose(sr, torch.full_like(sr, 0.02), atol=1e-6)))

        # degrau 5 nas 4 tarefas, com as travas do refino ligadas
        _poe_curriculo(env, 5, ("portao0", "subida"))
        env.reset()
        for nome, campo, col in (("kp", "actuator_gainprm", 0), ("kd", "actuator_biasprm", 2)):
            r = razao(campo, todos, col)
            check(f"degrau 5: {nome} em [0,9; 1,1] do default", bool(((r >= 0.9) & (r <= 1.1)).all()))
            check(f"degrau 5: {nome} não todo igual a 1", not bool(torch.allclose(r, torch.ones_like(r), atol=1e-6)))
        for i, c in enumerate(ctrl_de):
            lo = 0.9 if i < 7 else 0.8  # ⚠ P18: braços (grupos 7 e 8) com a faixa de 0,8
            r = razao("actuator_forcerange", c, 1)
            check(f"degrau 5: forcerange do grupo {i} em [{lo}; 1,1] do default", bool(((r >= lo) & (r <= 1.1)).all()),
                  f"{r.tolist()}")
        sr = env.sim.model.geom_solref[:, gid, 0]
        check("degrau 5: solref das geoms de contato em [0,015; 0,03]", bool(((sr >= 0.015) & (sr <= 0.03)).all()))
        check("degrau 5: solref igual entre as geoms de um mundo", bool(torch.allclose(sr, sr[:, :1].expand_as(sr))))
        st = env.sim.model.geom_solref[:, tid, 0]
        check("degrau 5: solref do terreno = 0,02", bool(torch.allclose(st, torch.full_like(st, 0.02), atol=1e-6)))

        # RFI: ruído de torque por junta (≤ 0,1·TAU_MAX) e nada na junta livre
        for _ in range(10):
            env.step(_zeros(env))
        pose0 = robo.data.root_link_pose_w.clone()  # os pés no chão: o contato pé × terreno do fim se lê daqui
        qf = env.sim.data.qfrc_applied
        tau = qf[:, robo.indexing.joint_v_adr]
        teto = 0.1 * torch.tensor(TAU_MAX, dtype=tau.dtype, device=tau.device)
        check("RFI: |τ| ≤ 0,1·TAU_MAX por junta", bool((tau.abs() <= teto + 1e-6).all()))
        check("RFI: torque não todo zero", bool(tau.abs().max() > 0))
        check("junta livre sem força (6 colunas)", bool((qf[:, robo.indexing.free_joint_v_adr] == 0).all()))

        # forca_caixa (M3): o env 0 em CARREGAR com `pegou` e sem `perdeu` é o único em que a força pode agir. Os outros
        # envs ficam como estão (ANDAR, do currículo). A tarefa e as flags são reescritas a cada passo.
        caixa, carregar = env.scene["box"], TAREFAS.index("CARREGAR")
        forca_no_0, sem_forca = False, True
        for _ in range(450):  # o intervalo da força é de 3 a 8 s (até 400 passos)
            # ⚠ S0: sem a base segura o robô cai a cada ~1,1 s, e o reset sorteia de novo o intervalo do impulso
            # (`apply_body_impulse.reset`): a força nunca chega
            _segura_base(env, pose0)
            env.v3_tarefa[0] = carregar
            env.v3_pegou[0] = True
            env.v3_perdeu[0] = False
            env.step(_zeros(env))
            w = caixa.data.body_external_wrench.flatten(1)
            forca_no_0 |= bool((w[0] != 0).any())
            sem_forca &= bool((w[1:] == 0).all())
        check("CARREGAR e pegou no env 0, 450 passos: wrench da caixa ≠ 0 em algum passo", forca_no_0)
        check("outros envs: wrench da caixa zero em todo passo (máscara da forca_caixa)", sem_forca)

        # M4 (P17): contato pé × terreno (pé priority 1, terreno 0): o solref do contato é o da geom de pé. O contato
        # com a laje (priority 3) ou a caixa não entra: a laje manda nele.
        import mujoco
        m = env.sim.mj_model
        pe = torch.tensor([i for i in range(m.ngeom) if re.search("_foot[1-7]_collision$",
                                                                   mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")],
                          dtype=torch.long)
        ct, n = env.sim.data.contact, int(env.sim.data.nacon[0])
        g = ct.geom[:n]  # (n, 2): as duas geoms de cada contato ativo
        e_pe, e_ter = torch.isin(g, pe), torch.isin(g, tid)
        sel = (e_pe[:, 0] & e_ter[:, 1]) | (e_pe[:, 1] & e_ter[:, 0])
        pe_geom = torch.where(e_pe[:, 0], g[:, 0], g[:, 1])[sel]
        sr_contato = ct.solref[:n, 0][sel]
        sr_geom = env.sim.model.geom_solref[ct.worldid[:n][sel], pe_geom, 0]
        check("contatos pé × terreno existem", bool(sel.any()), f"{int(sel.sum())} contatos")
        check("contato pé × terreno: solref do contato = solref da geom de pé",
              bool(sel.any()) and bool(torch.allclose(sr_contato, sr_geom, atol=1e-6)))
    finally:
        env.close()


def s37_com_caixa_ligado_sem_erro_de_heap():
    from g1_v3.knobs import ATIVO, JUNTAS

    # ⚠ PD-38: `com_caixa` só entra no cfg com `DR.com_caixa`; a seção liga a chave só para este cfg
    k = dataclasses.replace(ATIVO, dr=dataclasses.replace(ATIVO.dr, com_caixa=True))
    env = _abre_env(16, k=k)
    try:
        env.reset()
        env.v3_cur.k_dr = k.dr  # R4/M1: o `dr()` do currículo lê o `k_dr` (balanceador.py:60); sem isto o CoM fica zero
        _poe_curriculo(env, 0, ("subida",))
        env.reset()
        box = env.scene["box"]
        ip = env.sim.model.body_ipos[:, int(box.indexing.body_ids[0])]  # (E,3): o CoM da caixa no modelo de cada mundo
        a = env.v3_meia
        check("com_caixa: |xy| do CoM ≤ 0,3·a", bool((ip[:, :2].abs() <= 0.3 * a[:, None] + 1e-6).all()))
        check("com_caixa: |z| do CoM ≤ 0,2·a", bool((ip[:, 2].abs() <= 0.2 * a + 1e-6).all()))
        check("com_caixa: CoM não todo zero", bool(ip.abs().sum() > 0))
        ruins = 0
        for _ in range(200):
            obs, _rew, _term, _trunc, _info = env.step(torch.rand(16, len(JUNTAS), device=env.device) * 2 - 1)
            if not (torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()):
                ruins += 1
        check("com_caixa: 200 passos aleatórios em 16 mundos com observação finita", ruins == 0, f"{ruins} passos")
    finally:
        env.close()


def s38_save_e_load_com_degraus_e_travas():
    from mjlab.rl import RslRlVecEnvWrapper
    from g1_v3 import tarefa
    from g1_v3.runner import RunnerV3

    env = _abre_env(4)
    try:
        env.reset()
        _poe_curriculo(env, (5, 3, 4, 2), ("portao0", "subida"))  # ANDAR, PEGAR, CARREGAR, BOTAR: degrau global 2
        cur = env.v3_cur
        wrapper = RslRlVecEnvWrapper(env)
        agente = dataclasses.asdict(tarefa.rl_cfg())  # ⚠ como o mjlab/scripts/train.py:150 passa o agente
        with tempfile.TemporaryDirectory() as tmp:
            runner = RunnerV3(wrapper, agente, log_dir=tmp, device="cpu")  # ⚠ o `_embrulha_log` roda no __init__
            antes = (cur.estado(), cur.dr(), cur.degrau_global())
            caminho = os.path.join(tmp, "model_0.pt")
            runner.save(caminho)
            _poe_curriculo(env, 0)  # muda o estado depois do save: o load tem de desfazer a mudança
            runner.load(caminho)
            depois = (cur.estado(), cur.dr(), cur.degrau_global())
            check("save → load: estado(), dr() e degrau_global() iguais", depois == antes)
            check("degrau global = 2 depois do load", cur.degrau_global() == 2, str(cur.degrau_global()))
    finally:
        env.close()


def s39_odometria_anda_com_a_base():
    env = _abre_env(4)
    try:
        env.reset()
        robo = env.scene["robot"]
        # ⚠ achado de 30/10: o fce6a42 tirou o `v3_odo.passo` do `compute`; sem ele a odometria fica parada
        pose = robo.data.root_link_pose_w.clone()
        pose[:, 0] += 0.30
        robo.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        env.step(_zeros(env))
        erro = (env.v3_odo.pos_w[:, :2] - robo.data.root_link_pose_w[:, :2]).norm(dim=-1)
        check("odometria acompanha a base a < 0,02 m", bool((erro < 0.02).all()), f"{erro.tolist()}")
        dxy = env.v3_odo.delta_b()[0]
        check("Δ da odometria no passo do deslocamento passa de 0,25 m", bool((dxy.norm(dim=-1) > 0.25).all()),
              f"{dxy.norm(dim=-1).tolist()}")
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
    ("20 estimador vê a caixa à frente; idade cresce fora do FOV", s20_estimador_ve_caixa_a_frente_e_idade_cresce_fora_do_fov),
    ("21 pose de pé com caixa: aviso depois de 0,5 s", s21_pose_de_pe_com_caixa_dispara_aviso_depois_de_meio_segundo),
    ("22 laje sai com o CARREGAR que anda", s22_laje_sai_com_o_carregar_que_anda),
    ("23 tamanho do crítico confere com §6.5", s23_tamanho_do_critico_confere_com_6_5),
    ("24 atraso do atuador em {0..8} depois do portão do PEGAR", s24_atraso_do_atuador_depois_do_portao_do_pegar),
    ("25 fatia C forçada: laje de destino à frente no BOTAR", s25_fatia_c_forcada_laje_de_destino_a_frente),
    ("26 alvo de soltura no topo + a", s26_alvo_de_soltura_no_topo_mais_a),
    ("27 caixa de 5 cm não termina; de 10 cm termina", s27_caixa_solta_sobre_a_laje_5_cm_nao_termina_10_cm_termina),
    ("28 one-hot fica em BOTAR durante o atraso", s28_one_hot_fica_em_botar_durante_o_atraso),
    ("29 ANDAR parado final acaba em time_out", s29_andar_parado_final_acaba_em_time_out),
    ("30 env da fatia B no terreno nasce na plataforma", s30_env_da_fatia_b_no_terreno_nasce_na_plataforma),
    ("31 topo da laje = origem + h, origem de escada alta", s31_topo_da_laje_e_origem_de_escada_alta),
    ("32 degrau abaixo com caixa no peito: sem perda (D4)", s32_robo_um_degrau_abaixo_com_caixa_no_peito_nao_perde_a_caixa),
    ("33 altura da pelve e do joelho contra o terreno local", s33_altura_da_pelve_mede_contra_o_terreno_local),
    ("34 degrau 5: laje, chegada e pouso; sacudida e aperto finitos", s34_degrau_5_pesa_laje_chegada_e_pouso),
    ("35 batente só com degrau ≥ 1; contagem por junta (imprime)", s35_batente_so_com_degrau_1_e_contagem_por_junta),
    ("36 DR do refino: default no degrau 0; faixa no 5; RFI e caixa", s36_dr_do_refino_zero_no_degrau_0_e_faixa_no_5),
    ("37 com_caixa ligado: sem erro de heap (PD-38)", s37_com_caixa_ligado_sem_erro_de_heap),
    ("38 save → load com degraus e travas do DR", s38_save_e_load_com_degraus_e_travas),
    ("39 odometria anda com a base", s39_odometria_anda_com_a_base),
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
