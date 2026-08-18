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
    n_rew = len(cfg.rewards)
    checa(n_rew == 18, f"18 termos de recompensa (medido {n_rew}: {sorted(cfg.rewards)})")
    n_term = len(cfg.terminations)
    checa(n_term == 5, f"5 terminações (medido {n_term}: {sorted(cfg.terminations)})")
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
    # força metade dos envs em cada estado e mede
    cmd._command[:, 9] = 0.0
    cmd._update_command()
    env.observation_manager.compute()
    o0 = env.observation_manager.compute()["actor"]
    idx = {}
    inicio = 0
    for nome, termo in cfg.observations["actor"].terms.items():
        d = env.observation_manager.group_obs_term_dim["actor"][
            list(cfg.observations["actor"].terms).index(nome)]
        largura = int(d[0]) if hasattr(d, "__len__") else int(d)
        idx[nome] = (inicio, inicio + largura)
        inicio += largura
    zerados = ("palmas_para_caixa", "caixa_para_alvo", "face_alvo", "dir_alvo")
    for nome in zerados:
        a, b = idx[nome]
        checa(bool((o0[:, a:b].abs() < 1e-6).all()),
              f"com bit=0, `{nome}` é zero")

    # ⚠ O teste MAIS importante da lista. Com o bit em 0 os canais são zerados, e
    # um vetor zerado dá exp(0) = 1. Sem multiplicar por `caixa_valida`, "não
    # existe caixa" pagaria o valor MÁXIMO.
    from g1_poc import recompensas as R
    for nome, fn, params in (
        ("staged", R.staged, cfg.rewards["staged"].params),
        ("precise_pos", R.precise_pos, cfg.rewards["precise_pos"].params),
        ("precise_ori", R.precise_ori, cfg.rewards["precise_ori"].params),
        ("squeeze", R.squeeze, cfg.rewards["squeeze"].params),
    ):
        v = fn(env, **params)
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
    caixa_x = kc.caixa_xy[0]
    borda_perto = kc.prateleira_xy[0] - kc.prateleira_meia_xy
    checa(abs(caixa_x - (borda_perto + kc.caixa_meia_aresta[0])) < 1e-6,
          f"a caixa nasce na borda perto do robô (x = {caixa_x:.2f} m)")
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

    print("== 12. currículo ==")
    checa(NIVEL_MAX == 6, f"NIVEL_MAX == 6 (medido {NIVEL_MAX})")
    checa("forma" in cfg.curriculum and "nivel" in cfg.curriculum,
          "os termos `forma` e `nivel` existem")
    ordem = list(cfg.curriculum)
    checa(ordem.index("forma") < ordem.index("nivel"),
          "`forma` vem antes de `nivel` (os eventos leem `poc_manipula`)")
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
    print("  - a tabela de células do nível (passo 4)")
    print("  - GPU, DDP e escala")
    return 1 if FALHA else 0


if __name__ == "__main__":
    raise SystemExit(main())
