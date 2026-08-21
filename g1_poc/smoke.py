"""Smoke do g1_poc — roda na CPU, sem GPU, em segundos.

Ele NÃO mede valor de recompensa e NÃO mede convergência. Ele mede que o cfg monta,
que os contratos batem, e que os invariantes que já custaram blocos a este
repositório continuam de pé.

    python -m g1_poc.smoke

Um teste falho é um `AssertionError` com o número medido na mensagem.
"""
from __future__ import annotations


import torch

import g1_poc  # noqa: F401  (registra a task)
from g1_poc import cena as C
from g1_poc.comando import COMANDO_DIM, ALVO, FACE, DIR, VALIDA
from g1_poc import curriculo as CU
from g1_poc.curriculo import NIVEL_MAX
from g1_poc.env_cfg import OBS_ATOR, OBS_CRITICO, make_g1_poc_env_cfg
from g1_poc.knobs import Knobs

N_ENVS = 8
N_PASSOS = 5
OK, FALHA = [], []


def ok(msg: str) -> None:
    OK.append(msg)
    print(f"  ok   {msg}")


def checa(cond: bool, msg: str) -> None:
    if cond:
        ok(msg)
    else:
        FALHA.append(msg)
        print(f"  FALHA {msg}")


def termo_obs(env, grupo: str, nome: str):
    """O TermCfg de observação como o manager o tem, e não como o cfg o declarou.

    ⚠ Os dois não são o mesmo objeto. O manager faz `deepcopy` do cfg e RESOLVE os
    `SceneEntityCfg` nele: no cfg `site_names=PALM_SITES` continua com
    `site_ids=slice(None)` (os 6 sites do robô), e no manager já é `[4, 5]` (as duas
    palmas). Chamar a função com os params do CFG rebenta em `palmas - alvos`,
    porque `palmas` sai [B,6,3] contra [B,2,3]. Vale igual para as recompensas, e lá
    o acesso é público: `reward_manager.get_term_cfg(nome).params`.
    """
    m = env.observation_manager
    return m._group_obs_term_cfgs[grupo][m.active_terms[grupo].index(nome)]


def main() -> int:
    k = Knobs()
    print("== 1. o cfg monta ==")
    cfg = make_g1_poc_env_cfg(k)
    cfg.scene.num_envs = N_ENVS
    ok("make_g1_poc_env_cfg()")

    print("== 2. contrato do comando ==")
    checa(COMANDO_DIM == 10, f"COMANDO_DIM == 10 (medido {COMANDO_DIM})")
    fatias = [ALVO, FACE, DIR, VALIDA]
    cobertura = sorted(i for f in fatias for i in range(f.start, f.stop))
    checa(cobertura == list(range(COMANDO_DIM)),
          "as 4 fatias cobrem o comando sem buraco nem sobreposição")
    nomes = list(cfg.commands.keys())
    checa(nomes[0] == "caixa_alvo",
          f"`caixa_alvo` vem ANTES do `twist` (ordem medida: {nomes})")

    print("== 3. instancia e dá passos ==")
    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg, device="cpu")
    obs, _ = env.reset()
    ok(f"reset() com {N_ENVS} envs em CPU")

    print("== 4. contrato da observação ==")
    n_ator = obs["actor"].shape[-1]
    n_critico = obs["critic"].shape[-1]
    checa(n_ator == OBS_ATOR, f"ator == {OBS_ATOR} (medido {n_ator})")
    checa(n_critico == OBS_CRITICO, f"crítico == {OBS_CRITICO} (medido {n_critico})")
    # ⚠ §5.1 REVERTIDA em 20/08: o canal volta ao ator, POR ÚLTIMO, com Unoise
    # ±0,5 (receita do fabricante). A escola "sem o canal" exige histórico de
    # observação, que não temos — medido: 530 iterações com 30% de dados de andar
    # e a duração do episódio de locomoção caiu de 112 para 26 passos.
    _ator_terms = list(cfg.observations["actor"].terms)
    checa(_ator_terms[-1] == "base_lin_vel",
          f"`base_lin_vel` está no ator, POR ÚLTIMO (medido {_ator_terms[-1]!r}) — "
          f"canal novo no fim é o contrato da cirurgia")
    checa(cfg.observations["actor"].terms["base_lin_vel"].noise is not None,
          "e com ruído (±0,5, a tolerância do estimador de bordo)")
    checa("base_lin_vel" in cfg.observations["critic"].terms,
          "`base_lin_vel` LIMPO está no crítico (privilégio legítimo)")

    print("== 5. termos e terminações ==")
    # 13 da fundação do `velocity` + `self_collisions`, que o env_cfg CRIA (a
    # fundação não tem esse termo) + os 9 de tarefa (load em 20/08).
    n_rew = len(cfg.rewards)
    checa(n_rew == 23, f"23 termos de recompensa (medido {n_rew}: {sorted(cfg.rewards)})")
    # 14 de fundação (`velocity` do G1) + 9 de tarefa. Nenhum termo inventado no
    # eixo da locomoção — a §21 confere isso contra o cfg do fabricante.
    tarefa = ("staged", "precise_pos", "precise_ori", "squeeze", "unload",
              "postura_ereta", "sustentacao", "load", "joint_vel_hinge")
    faltam = [t for t in tarefa if t not in cfg.rewards]
    checa(not faltam, f"os 8 termos de tarefa existem (faltam: {faltam})")
    # a fundação traz 3 (`time_out`, `fell_over`, `out_of_terrain_bounds`), o env_cfg
    # tira o `out_of_terrain_bounds` e põe as 2 nossas. NÃO existe `nonfinite` no
    # mjlab — ver `tasks/velocity/velocity_env_cfg.py:377`.
    n_term = len(cfg.terminations)
    checa(n_term == 4, f"4 terminações (medido {n_term}: {sorted(cfg.terminations)})")
    # ⚠ o resample do comando NUNCA pode caber dentro do episódio: com o range
    # igual à duração, o time_left cruzava zero no passo 999 e zerava
    # episode_success/pegou/alvo UM PASSO antes do time_out — o nível lia sucesso
    # 0 em todo episódio que chegava ao fim (medido it 5306: pegou 0,97, sucesso 0,00)
    checa(cfg.commands["caixa_alvo"].resampling_time_range[0] > k.episodio.duracao_s,
          f"o resample do `caixa_alvo` não cabe no episódio (range "
          f"{cfg.commands['caixa_alvo'].resampling_time_range[0]:.0f} s > "
          f"{k.episodio.duracao_s:.0f} s) — senão o wipe do passo 999 volta")
    checa(cfg.terminations["time_out"].time_out is True,
          "`time_out` tem time_out=True (sem isso o rsl_rl trata como fracasso)")

    print("== 6. os passos são finitos ==")
    acao = torch.zeros(N_ENVS, env.action_manager.total_action_dim, device=env.device)
    for _ in range(N_PASSOS):
        obs, rew, *_ = env.step(acao)
    checa(bool(torch.isfinite(obs["actor"]).all()), "obs[actor] finita")
    checa(bool(torch.isfinite(obs["critic"]).all()), "obs[critic] finita")
    checa(bool(torch.isfinite(rew).all()), "recompensa finita")
    for nome, col in zip(env.reward_manager.active_terms,
                         env.reward_manager._step_reward.unbind(-1)):
        if not bool(torch.isfinite(col).all()):
            FALHA.append(f"termo não-finito: {nome}")
            print(f"  FALHA termo não-finito: {nome}")

    print("== 7. o bit `caixa_valida` ==")
    cmd = env.command_manager.get_term("caixa_alvo")
    # zera o bit em TODOS os envs. `_update_command` não recalcula o bit — ele só o
    # LÊ, e é ele que propaga o zero para as fatias `face_alvo` e `dir_alvo`.
    cmd._command[:, 9] = 0.0
    cmd._update_command()

    # ⚠ Duas armadilhas de MEDIÇÃO, e as duas fazem este teste passar/reprovar por
    # motivo errado:
    #   1. `observation_manager.compute()` devolve o `_obs_buffer` CACHEADO do passo
    #      anterior (`observation_manager.py:311`) — a obs de quando o bit era 1.
    #   2. `palmas_para_caixa` e `caixa_para_alvo` levam `Unoise(±0.01)`, somado
    #      DEPOIS da função. Zero mais ruído não é zero, e o ruído é de propósito.
    # O invariante é da FUNÇÃO. Portanto ela é chamada direto, com os params
    # resolvidos do manager, sem passar pelo buffer nem pelo ruído.
    zerados = ("palmas_para_caixa", "caixa_para_alvo", "face_alvo", "dir_alvo")
    for nome in zerados:
        tc = termo_obs(env, "actor", nome)
        v = tc.func(env, **tc.params)
        checa(bool((v.abs() < 1e-6).all()),
              f"com bit=0, `{nome}` é zero (medido max {float(v.abs().max()):.3e})")

    # ⚠ O teste MAIS importante da lista. Com o bit em 0 os canais são zerados, e
    # um vetor zerado dá exp(0) = 1. Sem multiplicar por `caixa_valida`, "não
    # existe caixa" pagaria o valor MÁXIMO.
    for nome in ("staged", "precise_pos", "precise_ori", "squeeze", "unload",
                 "postura_ereta", "sustentacao", "load"):
        tc = env.reward_manager.get_term_cfg(nome)
        v = tc.func(env, **tc.params)
        checa(bool((v.abs() < 1e-6).all()),
              f"com bit=0, `{nome}` é zero (medido max {float(v.abs().max()):.3e})")

    print("== 8. o twist zera na manipulação ==")
    env.poc_twist_zero = torch.ones(N_ENVS, dtype=torch.bool, device=env.device)
    twist = env.command_manager.get_term("twist")
    twist._update_command()
    checa(bool((twist.vel_command_b.abs() < 1e-6).all()),
          "com `poc_twist_zero`, o twist é zero")

    print("== 9. cena e grupos ==")
    checa(set(cfg.scene.entities) == {"robot", "box", "table"},
          f"3 entidades (medido {sorted(cfg.scene.entities)})")
    nomes_sensor = {s.name for s in cfg.scene.sensors}
    for n in C.SENSOR_PALMA + C.SENSOR_DORSO + (
            C.SENSOR_APOIO, C.SENSOR_CORPO_PRATELEIRA, C.SENSOR_AUTO_COLISAO, C.SENSOR_PES):
        checa(n in nomes_sensor, f"sensor `{n}` presente")

    print("== 10. geometria da §18 ==")
    kc = k.cena
    fundo_laje = kc.prateleira_topo_piso - 2 * kc.prateleira_meia_z
    checa(fundo_laje >= -1e-9,
          f"a laje APOIA no chão no piso da faixa (fundo = {fundo_laje:+.3f} m)")
    # O x da caixa é 0,32 (§3.1). NÃO é a tangência exata da borda perto, que daria
    # 0,30 — são 2 cm de folga. Portanto o invariante não é um número derivado, e sim
    # o que a folga tem de garantir: a caixa apoiada por inteiro, o jitter (só
    # positivo, para dentro) sem jogá-la fora, e ela do lado PERTO do robô, porque o
    # centro da prateleira é inalcançável (lição 16/07).
    caixa_x, meia = kc.caixa_xy[0], kc.caixa_meia_aresta[0]
    borda_perto = kc.prateleira_xy[0] - kc.prateleira_meia_xy
    borda_longe = kc.prateleira_xy[0] + kc.prateleira_meia_xy
    checa(caixa_x - meia >= borda_perto - 1e-9,
          f"a caixa nasce apoiada por inteiro (face em {caixa_x - meia:.2f} m, "
          f"borda perto em {borda_perto:.2f} m)")
    face_max = caixa_x + max(k.celulas.jitter_x_max) + meia
    checa(face_max <= borda_longe + 1e-9,
          f"o jitter não tira a caixa da prateleira (face em {face_max:.2f} m, "
          f"borda longe em {borda_longe:.2f} m)")
    checa(caixa_x <= kc.prateleira_xy[0],
          f"a caixa nasce do lado perto do robô (x = {caixa_x:.2f} m, centro da "
          f"prateleira em {kc.prateleira_xy[0]:.2f} m)")
    # o alvo do `pegar` tem de estar ACIMA da caixa em repouso por mais que o raio
    ka, kt = k.alvo, k.tol
    repouso = kc.prateleira_topo_teto + kc.caixa_meia_aresta[2]
    subida_min = ka.pegar_z[0] - repouso
    checa(subida_min > kt.raio_sucesso,
          f"a subida mínima ({subida_min*100:.0f} cm) é maior que a esfera de "
          f"sucesso ({kt.raio_sucesso*100:.0f} cm) — senão o sucesso dispara sem erguer")

    print("== 11. o alvo do `pegar` é do MUNDO ==")
    # duas amostras com o robô em alturas diferentes devem dar o MESMO alvo z.
    z0 = cmd._command[:, 2].clone()
    robot = env.scene["robot"]
    pose = robot.data.root_link_pose_w.clone()
    pose[:, 2] -= 0.20                      # agacha 20 cm
    robot.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    cmd._update_command()
    checa(bool(torch.allclose(z0, cmd._command[:, 2])),
          "agachar 20 cm NÃO move o alvo (âncora de mundo, ADR-0001)")

    print("== 11b. o anti-hack do `unload` ==")
    # ⚠ O `unload` paga por `1 − F_apoio/m·g`. Sem gate, DERRUBAR a caixa da prateleira
    # paga o MÁXIMO: sem tampo embaixo o apoio é zero e a fração vale 1. É o caminho
    # mais curto do robô, e este teste é o que o fecha.
    cmd._command[:, 9] = 1.0                # o teste do bit zerou; restaura
    cmd._update_command()
    tc_unload = env.reward_manager.get_term_cfg("unload")
    caixa = env.scene["box"]
    pose_caixa = caixa.data.root_link_pose_w.clone()
    pose_caixa[:, 2] = 0.05                 # no chão, longe da prateleira
    caixa.write_root_link_pose_to_sim(pose_caixa)
    env.sim.forward()
    v_caida = tc_unload.func(env, **tc_unload.params)
    checa(bool((v_caida.abs() < 1e-6).all()),
          f"derrubar a caixa NÃO paga `unload` (medido max "
          f"{float(v_caida.abs().max()):.3e})")

    print("== 11c. as rampas e os gates dos termos novos ==")
    tc_ereta = env.reward_manager.get_term_cfg("postura_ereta")
    v = tc_ereta.func(env, **tc_ereta.params)
    checa(bool((v.abs() < 1e-6).all()),
          f"sem preensão bimanual, `postura_ereta` é zero (medido max "
          f"{float(v.abs().max()):.3e}) — agachar para alcançar sai de graça")
    # a forma da rampa em duas partes, sobre a fórmula
    kt2, kr2 = k.tol, k.reward
    def rampa2(z):
        fl = min(max((z - (kt2.pelve_min - kr2.postura_ereta_rampa))
                     / kr2.postura_ereta_rampa, 0.0), 1.0)
        ff = min(max((z - (kt2.pelve_min - kr2.postura_ereta_rampa_fina))
                     / kr2.postura_ereta_rampa_fina, 0.0), 1.0)
        return 0.5 * fl + 0.5 * ff
    for z, esperado in ((0.20, 0.0), (0.425, 0.25), (0.65, 1.0), (0.75, 1.0)):
        f = rampa2(z)
        checa(abs(f - esperado) < 1e-9,
              f"rampa da pelve: z = {z:.3f} -> {f:.3f} (esperado {esperado:.2f})")
    checa(rampa2(0.61) > rampa2(0.57) + 0.25,
          "a parte FINA é íngreme: 4 cm perto do fecho valem mais que 25% da rampa")
    # a sustentação é rampa no cronômetro do comando
    tc_sus = env.reward_manager.get_term_cfg("sustentacao")
    cmd._sustenta[:] = 0.5 * cmd.cfg.sustenta_pegar_s
    v = tc_sus.func(env, **tc_sus.params)
    esperado_s = 0.5 * float(cmd.valida.max())
    checa(bool(((v - 0.5 * cmd.valida).abs() < 1e-6).all()),
          "meio cronômetro paga meia `sustentacao` (× valida)")
    cmd._sustenta[:] = 0.0
    # o σ do reaching é por elo, com piso
    checa(bool((env.poc_reach_inicial >= kr2.reaching_std - 1e-6).all()),
          f"`poc_reach_inicial` respeita o piso de {kr2.reaching_std} "
          f"(mín medido {float(env.poc_reach_inicial.min()):.3f})")
    checa(bool(torch.isfinite(env.poc_reach_inicial).all()),
          "`poc_reach_inicial` é finito")

    print("== 12. currículo ==")
    checa(NIVEL_MAX == 6, f"NIVEL_MAX == 6 (medido {NIVEL_MAX})")
    checa("forma" in cfg.curriculum and "nivel" in cfg.curriculum,
          "os termos `forma` e `nivel` existem")
    ordem = list(cfg.curriculum)
    checa(ordem.index("command_vel") < ordem.index("forma")
          and ordem.index("nivel") < ordem.index("forma"),
          f"`nivel` vem ANTES de `forma` (ordem: {ordem}) — ele "
          f"leem a forma do episódio que ACABOU, e `forma` a sobrescreve. Medido "
          f"20/08: com `nivel` depois, p_up = 0,7·p e locomoção rebaixava o nível")
    checa("hinge" in cfg.curriculum,
          "o cronograma do `joint_vel_hinge` existe (a pose entra DEPOIS da tarefa)")

    print("== 13. dívidas do treino antigo que não se repetem ==")
    checa("base_com" not in cfg.events,
          "`base_com` está FORA (dr.body_com_offset corrompe a heap)")
    fr = cfg.events["caixa_friction"]
    checa(fr.mode == "reset",
          f"a DR de atrito é por EPISÓDIO (medido mode={fr.mode!r})")
    checa(fr.params.get("shared_random") is True,
          "a DR de atrito é COMPARTILHADA entre as palmas")
    checa(fr.params["asset_cfg"].name == "box",
          f"a DR de atrito é na CAIXA (medido {fr.params['asset_cfg'].name!r})")
    ordem_ev = list(cfg.events)
    checa(ordem_ev.index("reset_cena") < ordem_ev.index("afasta_cena"),
          "`reset_cena` roda antes de `afasta_cena`")
    # ⚠ Este assert INVERTEU em 21/08. Ele guardava o −10,0 que vinha de
    # `manipulation`/`tracking`, e esse valor é dez vezes o do fabricante
    # (`velocity_env_cfg.py:317`). Andar precisa de amplitude no quadril e no
    # joelho; uma penalidade forte de limite empurra as juntas para o meio da faixa
    # e ACHATA o balanço. O `peak_height_mean` entre 0,007 e 0,023 no bloco 2 é
    # consistente com isso. Agora o guarda é contra voltar a endurecê-lo sem medida.
    checa(cfg.rewards["dof_pos_limits"].weight == -1.0,
          f"`dof_pos_limits` == −1 (o valor do fabricante; medido "
          f"{cfg.rewards['dof_pos_limits'].weight})")

    print("== 14. a tabela de células (§10.1) ==")
    cel = k.celulas
    tc_nivel = env.curriculum_manager.get_term_cfg("nivel")
    todos = torch.arange(N_ENVS, device=env.device)
    jit = k.cena.prateleira_jitter_z
    teto = k.cena.prateleira_topo_teto

    def cena_no_nivel(n: int):
        tc_nivel.params["nivel_forcado"] = n
        env._reset_idx(todos)
        env.sim.forward()
        return env.poc_topo.clone(), env.poc_massa.clone()

    topo0, massa0 = cena_no_nivel(0)
    checa(bool(((topo0 >= teto - jit - 1e-6) & (topo0 <= teto + jit + 1e-6)).all()),
          f"nível 0 é NO-OP: topo em 0,55 ± jitter (medido {float(topo0.min()):.3f}"
          f" a {float(topo0.max()):.3f})")
    checa(bool((massa0 - k.cena.caixa_massa).abs().max() < 1e-6),
          f"nível 0 é NO-OP: carga fixa em {k.cena.caixa_massa:.1f} kg")

    for n in range(NIVEL_MAX + 1):
        topo, massa = cena_no_nivel(n)
        checa(float(topo.min()) >= cel.topo_min[n] - jit - 1e-6,
              f"nível {n}: topo >= {cel.topo_min[n]:.2f} − jitter "
              f"(medido {float(topo.min()):.3f})")
        checa(float(topo.max()) <= teto + jit + 1e-6,
              f"nível {n}: o TETO continua {teto:.2f} (medido {float(topo.max()):.3f})")
        checa(float(massa.max()) <= cel.carga_max[n] + 1e-6,
              f"nível {n}: carga <= {cel.carga_max[n]:.1f} kg "
              f"(medido {float(massa.max()):.2f})")
        checa(float(massa.min()) >= k.cena.caixa_massa - 1e-6,
              f"nível {n}: o PISO da carga continua {k.cena.caixa_massa:.1f} kg")

    t0, _ = cena_no_nivel(0)
    t4, c4 = cena_no_nivel(4)
    checa(float(t4.min()) < float(t0.min()) - 0.20,
          f"promover 0 -> 4 BAIXA a prateleira ({float(t0.min()):.3f} -> "
          f"{float(t4.min()):.3f})")
    checa(float(c4.max()) > 2.0,
          f"promover 0 -> 4 SOBE a carga (máx medido {float(c4.max()):.2f} kg)")
    tc_nivel.params["nivel_forcado"] = None

    print("== 15. a promoção usa a forma que ACABOU (bug de 20/08) ==")
    tc_n = env.curriculum_manager.get_term_cfg("nivel")
    env.poc_nivel[:] = 2
    env.poc_manipula[:] = True                      # a forma do episódio que acabou
    cmd.episode_success.copy_(torch.ones(N_ENVS, device=env.device))
    tc_n.func(env, todos, **tc_n.params)            # o termo, ISOLADO do `forma`
    checa(bool((env.poc_nivel == 3).all()),
          f"manipulação + sucesso promove TODOS (medido "
          f"{int((env.poc_nivel == 3).sum())}/{N_ENVS})")
    env.poc_manipula[:] = False                     # locomoção que acabou
    cmd.episode_success.copy_(torch.zeros(N_ENVS, device=env.device))
    tc_n.func(env, todos, **tc_n.params)
    checa(bool((env.poc_nivel == 3).all()),
          "episódio de LOCOMOÇÃO não move o nível (nem para baixo)")

    print("== 16. o currículo de comando é o do fabricante ==")
    tc_cv = env.curriculum_manager.get_term_cfg("command_vel")
    from mjlab.tasks.velocity import mdp as vel_mdp
    checa(tc_cv.func is vel_mdp.commands_vel,
          "o termo é o `commands_vel` do mjlab, e não um gate próprio")
    checa("twist_ranges" not in env.curriculum_manager.active_terms,
          "o gate por competência do twist SAIU (era invenção de 21/08)")
    checa("action_rate" not in env.curriculum_manager.active_terms,
          "o cronograma do `action_rate` SAIU: o fabricante roda −0,10 fixo")
    est = tc_cv.params["velocity_stages"]
    checa([e["step"] for e in est] == [0, 5000 * 24, 10000 * 24],
          f"os degraus são os do fabricante (medido {[e['step'] for e in est]})")
    checa(est[0]["lin_vel_x"] == (-1.0, 1.0) and est[0]["ang_vel_z"] == (-0.5, 0.5),
          "e o degrau 0 é (−1,0; 1,0) / (−0,5; 0,5)")

    print("== 17. a máquina de elo (§7) ==")
    from g1_poc.comando import PEGAR, REORIENTAR, CARREGAR, BOTAR
    cmd._command[:, 9] = 1.0
    cmd.manipula[:] = True
    env.poc_manipula[:] = True

    # 17a. cadeia `pegar`->`botar`: força o fecho e valida a troca de elo
    cmd._cadeia[:] = 3
    cmd._elo_idx[:] = 0
    cmd._elo_id[:] = PEGAR
    cmd.pegou[:] = 0.0
    cmd.episode_success.copy_(torch.zeros(N_ENVS, device=env.device))
    cmd._sustenta[:] = cmd.cfg.sustenta_pegar_s + 1.0   # sustentado
    cmd._sust_alvo[:] = cmd.cfg.sustenta_pegar_s
    # checa que `_avanca_elo` está definido
    checa(hasattr(cmd, '_avanca_elo'), "existe método `_avanca_elo`")
    # força a troca de elo manualmente para testar a lógica, já que a troca real depende de todas as 4 condições
    cmd._avanca_elo(todos)
    checa(bool((cmd._elo_id == BOTAR).all()),
          "após `_avanca_elo`, o elo avança para `botar`")
    checa(bool((cmd._sustenta.abs() < 1e-6).all()), "o cronômetro zera na troca")
    checa(bool((env.poc_topo >= cmd.cfg.botar_topo_piso - 1e-6).all()),
          f"o topo novo respeita o piso da colocação ({cmd.cfg.botar_topo_piso})")

    # 17b. o `unload` é ZERO no elo `botar`, e o `load` só paga nele
    tc_unl = env.reward_manager.get_term_cfg("unload")
    v = tc_unl.func(env, **tc_unl.params)
    checa(bool((v.abs() < 1e-6).all()),
          f"`unload` mascarado fora do `pegar` (medido max {float(v.abs().max()):.2e})")
    tc_load = env.reward_manager.get_term_cfg("load")
    cmd._elo_id[:] = PEGAR
    v = tc_load.func(env, **tc_load.params)
    checa(bool((v.abs() < 1e-6).all()), "`load` é zero fora do `botar`")
    cmd._elo_id[:] = BOTAR

    # 17c-17d. validação de `caixa_largada` por ramo
    tc_cl = env.termination_manager.get_term_cfg("caixa_largada")
    env.poc_pegou[:] = 0.0
    checa(bool(~tc_cl.func(env, **tc_cl.params).any()),
          "sem preensão, `caixa_largada` nunca dispara")
    env.poc_pegou[:] = 1.0
    cmd._elo_id[:] = CARREGAR
    # coloca a caixa longe das palmas para disparar `escapou`. O z é FIXADO acima
    # de z_min para isolar o ramo: nos níveis altos a prateleira baixa deixa o
    # repouso em ~0,17 m, abaixo do z_min de 0,20 — `caiu` dispararia junto e o
    # teste do botar viraria moeda.
    caixa = env.scene["box"]
    pose_c = caixa.data.root_link_pose_w.clone()
    pose_c[:, 0] += 1.0  # distancia na horizontal
    pose_c[:, 2] = 0.50  # acima de z_min = 0,20
    caixa.write_root_link_pose_to_sim(pose_c)
    env.sim.forward()
    checa(bool(tc_cl.func(env, **tc_cl.params).all()),
          "no `carregar`, `caixa_largada` dispara com preensão e escapada")
    # no `botar` a caixa CONTINUA deslocada (escapou=True), mas o gate do elo
    # desarma o ramo: afastar as mãos é o objetivo. sucesso=0 e pegou=1 de propósito —
    # é o caso que o gate antigo (`pegou & ~sucesso`) errava.
    cmd._elo_id[:] = BOTAR
    cmd.episode_success.copy_(torch.zeros(N_ENVS, device=env.device))
    checa(bool(~tc_cl.func(env, **tc_cl.params).any()),
          "no `botar`, afastar as mãos NÃO termina (soltar é o objetivo)")
    # mas CAIR termina em qualquer elo, inclusive no `botar`
    pose_c[:, 2] = 0.05                       # abaixo de z_min = 0,20
    caixa.write_root_link_pose_to_sim(pose_c)
    env.sim.forward()
    checa(bool(tc_cl.func(env, **tc_cl.params).all()),
          "no `botar`, a caixa no chão TERMINA (`caiu` vale em qualquer elo)")

    # 17e. regressão POSITIVA do `load`: com contato real ele PAGA no botar.
    # Sem contato o valor-base é 0 em todos os elos e a máscara não é exercitada.
    env._reset_idx(todos)
    env.sim.forward()
    cmd._command[:, 9] = 1.0
    # ⚠ 21/08: escrever o bit à mão não basta mais. O `_update_command` o RECALCULA
    # todo passo como `manipula & ~aguardando` (§11.2), e o `env.step` abaixo passa
    # por ele. Com o `frac_locomocao` default agora em 1,0 a forma sorteada é
    # locomoção, o bit voltaria a 0 e o `load` mediria 0,000. A forma e a janela têm
    # de ser fixadas junto com o bit.
    cmd.manipula[:] = True
    cmd._espera[:] = 0.0
    # o alvo é ONDE a caixa está, apoiada; 3 passos assentam o contato caixa<->tampo
    for _ in range(3):
        env.step(torch.zeros(N_ENVS, env.action_manager.total_action_dim,
                             device=env.device))
    cmd._command[:, 0:3] = env.scene["box"].data.root_link_pos_w
    tc_load = env.reward_manager.get_term_cfg("load")
    cmd._elo_id[:] = BOTAR
    v = tc_load.func(env, **tc_load.params)
    checa(float(v.max()) > 0.9,
          f"`load` paga com a caixa APOIADA no alvo, no elo botar (medido {float(v.max()):.3f})")
    cmd._elo_id[:] = PEGAR
    v = tc_load.func(env, **tc_load.params)
    checa(bool((v.abs() < 1e-6).all()), "e é zero fora do botar — a máscara existe")

    # restaura a cena para as seções seguintes
    env._reset_idx(todos)

    print("== 18. o controlador da forma (§11) ==")
    tc_f = env.curriculum_manager.get_term_cfg("forma")
    # a álgebra do controlador, sobre a fórmula
    def f_de(tl, tm, alvo):
        return alvo * tm / max(tl * (1.0 - alvo) + alvo * tm, 1e-6)
    checa(abs(f_de(24.0, 961.0, 0.30) - 0.945) < 0.005,
          f"não anda (Tl=24): sorteia {f_de(24.0, 961.0, 0.30):.3f} de locomoção")
    checa(abs(f_de(961.0, 961.0, 0.30) - 0.30) < 1e-9,
          "marcha madura (Tl=Tm): o sorteio relaxa para o alvo 0,30")
    checa(f_de(0.0, 961.0, 0.0) == 0.0 and abs(f_de(24.0, 961.0, 1.0) - 1.0) < 1e-9,
          "os extremos do play (alvo 0 e 1) saem exatos, sem clamp")
    # o termo mede a forma ANTIGA e sorteia a nova
    env.poc_dur_loco = torch.full((), 24.0, device=env.device)
    env.poc_dur_manip = torch.full((), 961.0, device=env.device)
    env.episode_length_buf[:] = 500
    saida = tc_f.func(env, todos, **tc_f.params)
    checa(float(saida["frac_loco_sorteio"]) > 0.90,
          f"com Tl na EMA em 24, o sorteio despeja locomoção "
          f"(medido {float(saida['frac_loco_sorteio']):.3f})")
    env.episode_length_buf[:] = 0   # restaura o que a seção sujou

    print("== 19. o balanço automático de forma (§10.4) ==")
    balanco = tc_f.params["balanco"]
    checa(balanco is not None, "o balanço está LIGADO no treino (auto_balanco=True)")
    checa(Knobs().episodio.frac_locomocao == 1.0,
          f"o bloco começa em locomoção PURA (default medido "
          f"{Knobs().episodio.frac_locomocao})")
    if balanco is not None:
        # ⚠ Testado numa STUB, e não no `env`. Custou as três falhas de 21/08: as
        # seções 16 e 18 mexem em `common_step_counter` e nas EMAs para exercitar o
        # gate do twist, e o `poc_alvo_ultimo` do balanço fica preso num passo global
        # alto. Qualquer teste que mexa no contador depois disso volta atrás no tempo
        # e cai no retorno antecipado — os checks passariam ou falhariam por
        # contaminação, e não pelo que eles medem.
        #
        # `_alvo_locomocao` é função PURA sobre quatro atributos. Uma stub isola.
        import types
        P = balanco["passo"]
        IM, ID = balanco["iters_min"] * 24, balanco["iters_entre_degraus"] * 24

        def stub(dur, passo):
            e = types.SimpleNamespace(common_step_counter=passo, poc_dur_loco=dur)
            CU._alvo_locomocao(e, 1.0, balanco)   # cria o estado
            return e

        # nasce em locomoção pura, e a carência é relativa ao INÍCIO
        e = stub(900.0, 0)
        checa(abs(e.poc_alvo_loco - 1.0) < 1e-9,
              f"o alvo nasce em locomoção PURA (medido {e.poc_alvo_loco:.3f})")
        e.common_step_counter = IM - 24
        CU._alvo_locomocao(e, 1.0, balanco)
        checa(abs(e.poc_alvo_loco - 1.0) < 1e-9,
              "a carência segura o 1º degrau: a `dur_loco_ema` nasce NEUTRA em 1000")
        # ⚠ o mesmo teste com o contador ALTO no primeiro contato: é o caso do
        # RESUME, e é o bug que o smoke de 21/08 pegou
        e2 = stub(1000.0, 3000 * 24)
        e2.common_step_counter += ID * 2
        CU._alvo_locomocao(e2, 1.0, balanco)
        checa(abs(e2.poc_alvo_loco - 1.0) < 1e-9,
              "num RESUME (contador alto, EMA neutra) a carência AINDA segura")

        # passada a carência, com os dois sinais bons, desce um degrau
        e.common_step_counter = IM + ID
        CU._alvo_locomocao(e, 1.0, balanco)
        desceu = e.poc_alvo_loco
        checa(abs(desceu - (1.0 - P)) < 1e-9,
              f"com dur=900 ele desce um degrau (medido {desceu:.3f})")

        # a duração degradada faz SUBIR de volta — a assimetria é o ponto
        e.poc_dur_loco = 100.0
        e.common_step_counter += ID * 2
        CU._alvo_locomocao(e, 1.0, balanco)
        checa(e.poc_alvo_loco > desceu,
              f"com a duração degradada ele DEVOLVE chão "
              f"(medido {e.poc_alvo_loco:.3f})")

        # ⚠ o segundo sinal (erro de giro) SAIU em 21/08: número chutado, e ele
        # travou a rampa por 390 iterações com o yaw comprovadamente bom
        checa("erro_giro_alvo" not in balanco,
              "o balanço tem UM sinal só, como o `terrain_levels_vel` do fabricante")

        # o piso é respeitado
        e3 = stub(900.0, 0)
        e3.poc_dur_loco = 900.0
        e3.poc_alvo_loco = balanco["alvo_min"]
        e3.common_step_counter = IM + ID * 4
        CU._alvo_locomocao(e3, 1.0, balanco)
        checa(abs(e3.poc_alvo_loco - balanco["alvo_min"]) < 1e-9,
              f"e o piso {balanco['alvo_min']:.2f} não é furado")

        # sem balanço a fatia é a constante de antes (é o modo do play)
        checa(CU._alvo_locomocao(env, 0.42, None) == 0.42,
              "`balanco=None` devolve a fatia FIXA — é o que o play usa")

    print("== 20. a janela de espera (§11.2) ==")
    lo, hi = cmd.cfg.espera_s
    checa(hi > 0.0, f"a janela existe no treino ({lo:g}-{hi:g} s)")
    env._reset_idx(todos)
    # força TODO env a manipular, para o bit poder ser 1 depois da espera
    env.poc_manipula[:] = True
    cmd._resample_command(todos)
    cmd._espera[:] = 0.40
    cmd._update_command()
    checa(bool(env.poc_aguardando.all()), "dentro da janela, todos aguardam")
    checa(bool((cmd.command[:, 9] < 0.5).all()),
          "o BIT vai a 0 na espera — os nove termos de tarefa se desligam sozinhos")
    checa(bool(env.poc_twist_zero.all()),
          "o twist é zerado: `parado` é velocidade linear E angular zero")
    checa(bool((cmd._sustenta < 1e-9).all()),
          "o cronômetro de sustentação NÃO acumula na espera")
    t_elo = cmd._elo_t.clone()
    cmd._update_command()
    checa(bool((cmd._elo_t - t_elo).abs().max() < 1e-9),
          "o cronômetro do elo também congela (senão o `carregar` ganharia 1 s)")
    # passada a janela, o objetivo chega
    cmd._espera[:] = 0.0
    cmd._update_command()
    checa(not bool(env.poc_aguardando.any()), "fora da janela, ninguém aguarda")
    checa(bool((cmd.command[:, 9] > 0.5).all()),
          "e o BIT vai a 1: a descontinuidade É o sinal de que o objetivo chegou")
    env._reset_idx(todos)

    print("== 20b. o comando é SORTEADO na borda de abertura ==")
    # ⚠ Sem isto o comando é ZERO exatamente quando ele passa a valer: a linha
    # `vel_command_b[zero] = 0.0` é escrita NO LUGAR e destrói o sorteio, e o timer
    # do twist só volta em 3 a 8 s. O `carregar` fecha em 6 s.
    env._reset_idx(todos)
    tw = env.command_manager.get_term("twist")
    cmd = env.command_manager.get_term("caixa_alvo")
    env.poc_manipula[:] = False           # locomoção: o twist manda
    cmd._resample_command(todos)
    cmd._espera[:] = 0.40                 # dentro da janela
    cmd._update_command(); tw._update_command()
    checa(tw.cfg.init_velocity_prob == 0.0,
          "`init_velocity_prob` é 0: o sorteio na borda roda no MEIO do episódio, e "
          "o `_resample_command` do fabricante escreveria a velocidade da base no sim")
    checa(bool((tw.command.abs().sum(dim=-1) < 1e-9).all()),
          "na janela de espera o comando é zero nos três eixos")
    cmd._espera[:] = 0.0                  # a janela acabou
    cmd._update_command(); tw._update_command()
    checa(float(tw.command.abs().sum(dim=-1).max()) > 1e-6,
          f"e na saída da janela ele é SORTEADO, não fica em zero "
          f"(máx medido {float(tw.command.abs().sum(dim=-1).max()):.3f})")
    n_zero = int((tw.command.abs().sum(dim=-1) < 1e-9).sum())
    checa(n_zero <= max(1, N_ENVS // 4),
          f"e a maioria recebe comando não nulo ({n_zero}/{N_ENVS} em zero — "
          f"os `standing` do fabricante)")
    env._reset_idx(todos)

    print("== 21. PARIDADE da locomoção com o fabricante ==")
    # ⚠ Esta seção é o guarda-corpo pedido em 21/08: a locomoção tem de ser
    # EXATAMENTE o `velocity` do fabricante, com UMA mudança — a janela de espera.
    # Ela existe para que o próximo desvio bem-intencionado seja pego aqui, e não
    # 1200 iterações depois num painel.
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
    fab = unitree_g1_flat_env_cfg(play=False)

    FUNDACAO = ("track_linear_velocity", "track_angular_velocity", "upright", "pose",
                "body_ang_vel", "angular_momentum", "dof_pos_limits", "action_rate_l2",
                "air_time", "foot_clearance", "foot_swing_height", "foot_slip",
                "soft_landing", "self_collisions")
    difs = [n for n in FUNDACAO
            if cfg.rewards[n].weight != fab.rewards[n].weight]
    checa(not difs, f"os {len(FUNDACAO)} pesos de fundação batem com o fabricante "
                    f"(divergem: {difs})")

    checa(cfg.actions["joint_pos"].scale == fab.actions["joint_pos"].scale,
          "a escala de ação é o `G1_ACTION_SCALE` puro (o ×0,8 saiu em 21/08)")

    tw, tw_f = cfg.commands["twist"], fab.commands["twist"]
    difs = [c for c in ("rel_standing_envs", "rel_heading_envs", "rel_forward_envs",
                        "heading_command", "heading_control_stiffness",
                        "resampling_time_range")
            if getattr(tw, c) != getattr(tw_f, c)]
    checa(not difs, f"os parâmetros do twist batem (divergem: {difs})")
    difs = [c for c in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
            if tuple(getattr(tw.ranges, c)) != tuple(getattr(tw_f.ranges, c))]
    checa(not difs, f"as faixas do twist batem (divergem: {difs})")

    cv, cv_f = cfg.curriculum["command_vel"], fab.curriculum["command_vel"]
    checa(cv.func is cv_f.func and
          cv.params["velocity_stages"] == cv_f.params["velocity_stages"],
          "o currículo de comando é o mesmo termo e a mesma tabela")

    difs = [n for n in fab.terminations if n not in cfg.terminations]
    checa(not difs, f"nenhuma terminação do fabricante foi removida (faltam: {difs})")

    # o que SOBRA em relação ao fabricante. Tudo aqui é da CAIXA, e tudo é mascarado
    # pelo bit `caixa_valida` — portanto na locomoção o valor é exatamente zero.
    TAREFA = {"staged", "precise_pos", "precise_ori", "squeeze", "unload", "load",
              "postura_ereta", "sustentacao", "joint_vel_hinge"}
    extras = set(cfg.rewards) - set(fab.rewards)
    checa(extras == TAREFA,
          f"os únicos termos EXTRA são os 9 da caixa (medido {sorted(extras)})")

    extras_ev = set(cfg.events) - set(fab.events)
    checa(extras_ev == {"caixa_friction", "entrega_do_navegador", "reset_cena",
                        "carga_caixa", "afasta_cena"},
          f"os eventos extra são só de cena/caixa (medido {sorted(extras_ev)})")
    faltam_ev = set(fab.events) - set(cfg.events)
    checa(faltam_ev == {"base_com"},
          f"o ÚNICO evento do fabricante removido é o `base_com` "
          f"(dr.body_com_offset corrompe a heap) — medido {sorted(faltam_ev)}")

    extras_cu = set(cfg.curriculum) - set(fab.curriculum)
    checa(extras_cu == {"nivel", "forma", "hinge"},
          f"o currículo extra é só da caixa e da forma (medido {sorted(extras_cu)})")

    # A ÚNICA mudança deliberada na locomoção
    checa(cfg.commands["caixa_alvo"].espera_s[1] > 0.0,
          "e a UMA mudança deliberada existe: a janela de espera (§11.2)")

    env.close()

    print()
    print(f"== {len(OK)} ok, {len(FALHA)} falhas ==")
    for f in FALHA:
        print(f"  FALHA: {f}")
    print()
    print("NÃO coberto por este smoke, e é declarado:")
    print("  - valor de recompensa e convergência")
    print("  - a física do `reorientar` (empurrar a caixa apoiada) — só a sonda/play medem")
    print("  - GPU, DDP e escala")
    return 1 if FALHA else 0


if __name__ == "__main__":
    raise SystemExit(main())
