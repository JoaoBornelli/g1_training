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
    checa("base_lin_vel" not in cfg.observations["actor"].terms,
          "`base_lin_vel` NÃO está no ator (não é medível no robô real)")
    checa("base_lin_vel" in cfg.observations["critic"].terms,
          "`base_lin_vel` está no crítico (privilégio legítimo)")

    print("== 5. termos e terminações ==")
    # 13 da fundação do `velocity` + `self_collisions`, que o env_cfg CRIA (a
    # fundação não tem esse termo) + os 8 de tarefa (postura_ereta e sustentacao em 20/08).
    n_rew = len(cfg.rewards)
    checa(n_rew == 22, f"22 termos de recompensa (medido {n_rew}: {sorted(cfg.rewards)})")
    tarefa = ("staged", "precise_pos", "precise_ori", "squeeze", "unload",
              "postura_ereta", "sustentacao", "joint_vel_hinge")
    faltam = [t for t in tarefa if t not in cfg.rewards]
    checa(not faltam, f"os 8 termos de tarefa existem (faltam: {faltam})")
    # a fundação traz 3 (`time_out`, `fell_over`, `out_of_terrain_bounds`), o env_cfg
    # tira o `out_of_terrain_bounds` e põe as 2 nossas. NÃO existe `nonfinite` no
    # mjlab — ver `tasks/velocity/velocity_env_cfg.py:377`.
    n_term = len(cfg.terminations)
    checa(n_term == 4, f"4 terminações (medido {n_term}: {sorted(cfg.terminations)})")
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
                 "postura_ereta", "sustentacao"):
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
    checa(ordem.index("twist_ranges") < ordem.index("forma")
          and ordem.index("nivel") < ordem.index("forma"),
          f"`twist_ranges` e `nivel` vêm ANTES de `forma` (ordem: {ordem}) — os dois "
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
    checa(cfg.rewards["dof_pos_limits"].weight == -10.0,
          f"`dof_pos_limits` == −10 (medido {cfg.rewards['dof_pos_limits'].weight})")

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

    print("== 16. o gate por competência do twist (§10.3) ==")
    tc_tw = env.curriculum_manager.get_term_cfg("twist_ranges")
    degrau = k.cronograma.locomocao[1]["step"]
    env.common_step_counter = degrau + k.cronograma.twist_iters_entre_degraus * 24 + 1
    env.poc_estagio_twist = 0
    env.poc_duracao_loco = torch.zeros((), device=env.device)
    env.poc_twist_ultimo_degrau = 0
    tc_tw.func(env, todos, **tc_tw.params)
    checa(env.poc_estagio_twist == 0,
          "passo global acima do degrau mas duração baixa: o estágio SEGURA")
    tw_cfg = env.command_manager.get_term("twist").cfg
    checa(tuple(tw_cfg.ranges.lin_vel_x) == tuple(k.cronograma.locomocao[0]["lin_vel_x"]),
          f"e a faixa fica no estágio 0 (medido {tw_cfg.ranges.lin_vel_x})")
    env.poc_duracao_loco = torch.tensor(float(env.max_episode_length), device=env.device)
    tc_tw.func(env, todos, **tc_tw.params)
    checa(env.poc_estagio_twist == 1,
          "com a duração no alvo, o estágio SOBE")
    tc_tw.func(env, todos, **tc_tw.params)
    checa(env.poc_estagio_twist == 1,
          "e NÃO sobe de novo na mesma janela (teto de 1 degrau por 12 iterações)")
    env.common_step_counter += k.cronograma.twist_iters_entre_degraus * 24 + 1
    env.poc_duracao_loco = torch.tensor(0.0, device=env.device)
    tc_tw.func(env, todos, **tc_tw.params)
    checa(env.poc_estagio_twist == 0,
          "com a duração degradada, o estágio DESCE (histerese de 0,8×alvo)")

    env.close()

    print()
    print(f"== {len(OK)} ok, {len(FALHA)} falhas ==")
    for f in FALHA:
        print(f"  FALHA: {f}")
    print()
    print("NÃO coberto por este smoke, e é declarado:")
    print("  - valor de recompensa e convergência")
    print("  - as 4 cadeias e a troca de elo (passo 4 da §17)")
    print("  - o movimento da prateleira quando o `pegar` fecha (passo 4)")
    print("  - GPU, DDP e escala")
    return 1 if FALHA else 0


if __name__ == "__main__":
    raise SystemExit(main())
