"""PASSO 2, parte B — o PORTÃO. Confere meu caminho em lote contra o original.

    python g1_residual/referencia.py

Pré-requisito: rodar o `dump_referencia.py` no ambiente do BFM-Zero primeiro.

**Por que este arquivo existe.** Eu reescrevi a montagem da obs do BFM de numpy /
um-robô para torch / N-robôs. Erro nessa reescrita **não levanta exceção**: o ator
recebe números na escala ou na ordem errada e devolve uma ação plausível mas errada.
É a mesma classe de falha do bug de índice do normalizador que já apareceu neste
projeto. A única defesa é comparar com o caminho que funciona.

Quatro checagens, de baixo para cima:

    A  o ATOR         obs do rastro -> minha ação == ação do rastro
    B  os CAMPOS      qpos/qvel do rastro -> campos do mjlab == contas cruas
    C  o ESTADO       o `ObsBFM` reproduz os 64 números do rastro
    D  o LAYOUT       os 372 números do histórico estão na ordem que eu suponho

A checagem A é a que pega normalizador esquecido e peso mal carregado. A D pega
ordem de concatenação errada, e ela roda só com o rastro, sem simulador.
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from g1_residual.bfm import PASSOS_HISTORICO, AtorBFM  # noqa: E402
from g1_residual.obs_bfm import ESCALA_ANG_VEL, ObsBFM  # noqa: E402

RASTRO = pathlib.Path(__file__).resolve().parent / "peso" / "referencia.npz"
TOL_ATOR = 2e-4
TOL_OBS = 1e-5

falhas: list[str] = []


def check(nome: str, ok: bool, detalhe: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FALHOU'}  {nome}" + (f"   ({detalhe})" if detalhe else ""))
    if not ok:
        falhas.append(nome)


def main() -> None:
    assert RASTRO.is_file(), (
        f"não achei {RASTRO}.\nRode primeiro, no ambiente do BFM-Zero:\n"
        f"  cd ~/Documents/BFM-Zero && ~/miniconda3/envs/bmf0/bin/python "
        f"{pathlib.Path(__file__).parent}/dump_referencia.py")
    r = np.load(RASTRO)
    K = r["state"].shape[0]
    print(f"rastro: {K} passos, comportamento {r['comportamento']}\n")

    ator = AtorBFM(device="cpu")
    z = torch.as_tensor(r["z"]).float().reshape(1, -1)

    # ---------------------------------------------------------------- A: ator
    print("-- A: o ator (obs do rastro -> minha ação) --")
    a_meu = ator(torch.as_tensor(r["state"]).float(),
                 torch.as_tensor(r["last_action"]).float(),
                 torch.as_tensor(r["history_actor"]).float(),
                 z.expand(K, -1))
    a_ref = torch.as_tensor(r["acao"]).float()
    erro = (a_meu - a_ref).abs()
    check("ação bate com a do caminho original",
          float(erro.max()) < TOL_ATOR,
          f"erro máx {float(erro.max()):.2e}, médio {float(erro.mean()):.2e}")
    # um controle NEGATIVO: sem o normalizador a ação tem que sair bem diferente.
    # Se este passar, o normalizador não está sendo aplicado e a checagem A é cega.
    salvo = ator._normalizador
    try:
        # `nn.Identity` e não lambda: atributo de submódulo do torch só aceita
        # Module. O `forward` dela devolve a entrada intacta, dicionário incluído.
        ator._normalizador = torch.nn.Identity()
        a_cru = ator(torch.as_tensor(r["state"]).float(),
                     torch.as_tensor(r["last_action"]).float(),
                     torch.as_tensor(r["history_actor"]).float(), z.expand(K, -1))
    finally:
        ator._normalizador = salvo
    d_cru = float((a_cru - a_ref).abs().max())
    check("controle negativo: sem normalizador a ação MUDA",
          d_cru > 10 * TOL_ATOR, f"erro sem normalizador {d_cru:.2e}")

    # ------------------------------------------------------- B e C: mjlab
    print("\n-- B: campos do mjlab contra as contas cruas --")
    import g1_multitask  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.lab_api.math import quat_apply

    cfg = load_env_cfg(g1_multitask.TASK_ID)
    cfg.scene.num_envs = K
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    robot = env.scene["robot"]

    qpos = torch.as_tensor(r["qpos"]).float()          # [K, 36]
    qvel = torch.as_tensor(r["qvel"]).float()          # [K, 35]
    # cada ambiente recebe UM passo do rastro -> as K checagens saem de uma vez
    origem = env.scene.env_origins
    pose = torch.cat([qpos[:, 0:3] + origem, qpos[:, 3:7]], dim=-1)
    robot.write_root_link_pose_to_sim(pose)
    # ⚠️ O `qvel[3:6]` do freejoint do MuJoCo é a angular no frame LOCAL do corpo.
    # Verificado contra o `qvel` cru do mjlab: `root_link_ang_vel_b` bate a 3e-08 e
    # o `_w` erra 8e-02 com o robô inclinado. Mas o
    # `write_root_link_velocity_to_sim` recebe MUNDO — escrever o valor local nele
    # gerava uma diferença de 1,5e-02 que parecia bug do `obs_bfm.py` e era do
    # teste. Rotaciona antes de escrever.
    ang_w = quat_apply(qpos[:, 3:7], qvel[:, 3:6])
    robot.write_root_link_velocity_to_sim(torch.cat([qvel[:, 0:3], ang_w], dim=-1))
    robot.write_joint_state_to_sim(qpos[:, 7:36], qvel[:, 6:35])
    env.sim.forward()

    d = robot.data
    check("joint_pos == qpos[7:36]",
          float((d.joint_pos - qpos[:, 7:36]).abs().max()) < TOL_OBS,
          f"máx {float((d.joint_pos - qpos[:, 7:36]).abs().max()):.2e}")
    check("joint_vel == qvel[6:35]",
          float((d.joint_vel - qvel[:, 6:35]).abs().max()) < TOL_OBS,
          f"máx {float((d.joint_vel - qvel[:, 6:35]).abs().max()):.2e}")
    # o rastro guarda o `state`, e nele a gravidade são as posições 58:61
    grav_ref = torch.as_tensor(r["state"][:, 58:61]).float()
    e_grav = float((d.projected_gravity_b - grav_ref).abs().max())
    check("projected_gravity_b == gravidade do rastro", e_grav < 1e-4,
          f"máx {e_grav:.2e}")
    # e a angular são 61:64, JÁ multiplicadas por 0.25
    av_ref = torch.as_tensor(r["state"][:, 61:64]).float()
    e_av = float((d.root_link_ang_vel_b * ESCALA_ANG_VEL - av_ref).abs().max())
    check("root_link_ang_vel_b * 0.25 == ang_vel do rastro", e_av < 1e-4,
          f"máx {e_av:.2e}")

    print("\n-- C: o ObsBFM reproduz os 64 números --")
    plant = torch.load(pathlib.Path(__file__).resolve().parent / "peso"
                       / "bfm_ator.pt", map_location="cpu",
                       weights_only=True)["plant"]
    obs = ObsBFM(env, plant["default_joint_pos"])
    estado_meu, _, _ = obs.monta()
    estado_ref = torch.as_tensor(r["state"]).float()
    e = (estado_meu - estado_ref).abs()
    check("state[64] bate", float(e.max()) < 1e-4,
          f"máx {float(e.max()):.2e} | dof_pos {float(e[:, :29].max()):.1e} "
          f"dof_vel {float(e[:, 29:58].max()):.1e} "
          f"grav {float(e[:, 58:61].max()):.1e} "
          f"ang {float(e[:, 61:64].max()):.1e}")

    print("\n-- D: o layout dos 372 números do histórico --")
    # Só com o rastro, sem simulador. O slot 0 de cada componente tem que ser o
    # valor MAIS RECENTE, e para dof_pos/dof_vel/grav ele aparece também no
    # `state`. Se a ordem do concat estiver errada, isto explode.
    h = torch.as_tensor(r["history_actor"]).float()
    P = PASSOS_HISTORICO
    fatias = {"acao": (0, P * 29), "ang_vel": (P * 29, P * 29 + P * 3),
              "dof_pos": (P * 29 + P * 3, P * 29 + P * 3 + P * 29),
              "dof_vel": (P * 29 + P * 3 + P * 29, P * 29 + P * 3 + 2 * P * 29),
              "grav": (P * 29 + P * 3 + 2 * P * 29, 372)}
    check("as fatias somam 372", fatias["grav"][1] == 372)
    # no passo 1 em diante o slot 0 de dof_pos casa com algum estado já visto
    hp = h[:, fatias["dof_pos"][0]:fatias["dof_pos"][0] + 29]
    dist = (hp[1:].unsqueeze(1) - estado_ref[:, :29].unsqueeze(0)).abs().amax(-1)
    check("slot 0 de dof_pos é um estado do rastro",
          float(dist.min(dim=-1).values.max()) < 1e-4,
          f"pior casamento {float(dist.min(dim=-1).values.max()):.2e}")
    ha = h[:, fatias["acao"][0]:fatias["acao"][0] + 29]
    la = torch.as_tensor(r["last_action"]).float()
    check("slot 0 do histórico de ação == last_action",
          float((ha - la).abs().max()) < 1e-4,
          f"máx {float((ha - la).abs().max()):.2e}")

    # ------------------------------------------------------------------ E
    print("\n-- E: a EVOLUÇÃO do histórico, passo a passo --")
    # Esta é a checagem que faltava, e é a que pegou o bug de verdade. As A-D olham
    # UM passo; o histórico é o único estado que se acumula ao longo do tempo, e o
    # cronograma dos 4 slots não é `[t, t-1, t-2, t-3]`. Medido:
    #
    #     slot0 = x_t     slot1 = x_{t-1}     slot2 = x_{t-1}     slot3 = x_{t-2}
    #
    # A duplicata em 1 e 2 vem do `step()` do BFM montar a obs duas vezes por passo
    # de controle, e o "antes da física" no passo t ser igual ao "depois" em t-1.
    cfg1 = load_env_cfg(g1_multitask.TASK_ID)
    cfg1.scene.num_envs = 1
    env1 = ManagerBasedRlEnv(cfg=cfg1, device="cpu")
    env1.reset()
    rb = env1.scene["robot"]
    obs1 = ObsBFM(env1, plant["default_joint_pos"])
    acoes = torch.as_tensor(r["acao"]).float()
    h_ref = torch.as_tensor(r["history_actor"]).float()
    la_ref = torch.as_tensor(r["last_action"]).float()
    pior_h = pior_la = 0.0
    for t in range(K):
        p = torch.cat([qpos[t:t + 1, 0:3] + env1.scene.env_origins,
                       qpos[t:t + 1, 3:7]], dim=-1)
        rb.write_root_link_pose_to_sim(p)
        aw = quat_apply(qpos[t:t + 1, 3:7], qvel[t:t + 1, 3:6])
        rb.write_root_link_velocity_to_sim(
            torch.cat([qvel[t:t + 1, 0:3], aw], dim=-1))
        rb.write_joint_state_to_sim(qpos[t:t + 1, 7:36], qvel[t:t + 1, 6:35])
        env1.sim.forward()
        _, ultima, hist = obs1.monta()
        pior_h = max(pior_h, float((hist[0] - h_ref[t]).abs().max()))
        pior_la = max(pior_la, float((ultima[0] - la_ref[t]).abs().max()))
        obs1.guarda_acao(acoes[t:t + 1])
    check("history_actor[372] bate nos 40 passos", pior_h < 1e-4,
          f"pior erro {pior_h:.2e}")
    check("last_action[29] bate nos 40 passos", pior_la < 1e-4,
          f"pior erro {pior_la:.2e}")

    print("\n" + "=" * 64)
    if falhas:
        print(f"PORTÃO FECHADO — {len(falhas)} falha(s): {falhas}")
        print("NÃO suba nada para o Kaggle até isto passar.")
        sys.exit(1)
    print("PORTÃO ABERTO — o caminho em lote reproduz o original")


if __name__ == "__main__":
    main()
