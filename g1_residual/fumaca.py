"""TESTES 0 e 1 — o BFM sozinho, com o residual em ZERO. Nenhum treino.

    python g1_residual/fumaca.py [num_envs] [passos]

Com ação zerada o residual é zero e `c` é zero, então `z = prior` e o que roda é o
BFM puro dentro do NOSSO env. É o teste de fumaça da peça montada, e ele responde
duas perguntas que decidem a arquitetura antes de gastar GPU:

    TESTE 0   o BFM fica de pé no nosso env, com a nossa cena e a nossa DR?
    TESTE 1   o BFM segura uma caixa que ele nunca viu no treino?

O teste 1 é o importante. O BFM-Zero foi treinado em LaFAN, **sem carga nas mãos**.
Se ele derruba a caixa ou cai com ela, o residual tem que consertar equilíbrio além
de fazer a tarefa, e o clamp de 0,35 rad na perna passa a ser pouco.

O teste 0 também compara `z`: o `crouch-0` (prior do `pegar`) contra o
`move-ego-0-0` (ficar de pé), e a média das 10 sementes contra a semente 0. Isso não
é capricho — medido, as sementes de `move-ego-0-0` estão a ~60° umas das outras
(cos médio 0,500), porque "não se mexa" tem muitas soluções. Para comportamento
difuso a média pode ser pior que uma semente só.
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import g1_residual  # noqa: E402, F401
from g1_multitask import tasks as T  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
PASSOS = int(sys.argv[2]) if len(sys.argv) > 2 else 300

Z_DE_PE = 0.65
"""O mesmo limiar do `terminations.de_pe`: pelve >= 0,65 m."""


def monta(dev: str):
    cfg = load_env_cfg(g1_residual.TASK_ID)
    cfg.scene.num_envs = N
    env = ManagerBasedRlEnv(cfg=cfg, device=dev)
    return env, env.action_manager.get_term("joint_pos")


def roda(env, termo, tarefa: int, z_nome: str, semente: int | None,
         passos: int = PASSOS) -> dict:
    """Força tarefa e `z`, roda com ação zerada, devolve o resumo."""
    base = termo._base
    base.prior[tarefa] = base.M[base.nomes.index(z_nome)] if semente is None else \
        base._projeta(termo._ator.z_tabela[z_nome][semente].unsqueeze(0))[0]

    env.reset()
    env.tarefa_sorteada[:] = tarefa
    if hasattr(env, "task_dist"):
        env.task_dist = torch.zeros(T.NUM_TASKS, device=env.device)
        env.task_dist[tarefa] = 1.0
    env.reset()                      # 2º reset: já com a tarefa fixada

    acao = torch.zeros(env.num_envs, env.action_manager.total_action_dim,
                       device=env.device)
    robot = env.scene["robot"]
    caixa = env.scene["box"]
    vivo_ate = torch.zeros(env.num_envs, device=env.device)
    z_min = torch.full((env.num_envs,), 9.9, device=env.device)
    caixa_min = torch.full((env.num_envs,), 9.9, device=env.device)
    quedas = 0
    for i in range(passos):
        env.step(acao)
        pelve = robot.data.root_link_pos_w[:, 2]
        z_min = torch.minimum(z_min, pelve)
        cx = caixa.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        caixa_min = torch.minimum(caixa_min, cx)
        de_pe = pelve >= Z_DE_PE
        vivo_ate = torch.where(de_pe, torch.full_like(vivo_ate, i + 1), vivo_ate)
        quedas += int((~de_pe).sum() == env.num_envs)
    return {
        "z": f"{z_nome}[{'média' if semente is None else semente}]",
        "de_pe_no_fim": float((robot.data.root_link_pos_w[:, 2] >= Z_DE_PE).float().mean()),
        "passos_de_pe": float(vivo_ate.mean()),
        "pelve_min": float(z_min.mean()),
        "caixa_min": float(caixa_min.mean()),
    }


def main() -> None:
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"{N} envs, {PASSOS} passos ({PASSOS * 0.02:.1f} s), {dev}\n")
    env, termo = monta(dev)

    print("== TESTE 0: o BFM fica de pé no nosso env? (tarefa `pegar`) ==")
    print(f"{'z':28s} {'de pé no fim':>13s} {'passos de pé':>13s} {'pelve mín':>10s}")
    linhas = []
    # ⚠️ O nome `crouch-N` é a ALTURA ALVO, não a intensidade. Medido: `crouch-0`
    # significa "vai ao chão" e desaba em 10 passos (pelve 0,125 m). Por isso a
    # varredura inclui as variantes e o `move-ego-0-0` como controle.
    for nome, sem in (("crouch-0", None), ("crouch-0.25", None),
                      ("move-ego-low0.5-0-0", None), ("move-ego-low0.6-0-0.7", None),
                      ("move-ego-0-0", None), ("move-ego-0-0", 0),
                      ("raisearms-m-m", None), ("sitonground", None)):
        r = roda(env, termo, T.PEGAR, nome, sem)
        linhas.append(r)
        print(f"{r['z']:28s} {r['de_pe_no_fim']:12.1%} "
              f"{r['passos_de_pe']:13.0f} {r['pelve_min']:10.3f}")
    melhor = max(linhas, key=lambda r: r["passos_de_pe"])
    print(f"\n  melhor: {melhor['z']}  ({melhor['passos_de_pe']:.0f} de {PASSOS} passos)")
    if melhor["passos_de_pe"] < 0.5 * PASSOS:
        print("  ⚠️ o BFM NÃO fica de pé aqui. Antes de treinar, investigue:\n"
              "     ganhos, cena (a prateleira empurra?), DR, ou o `z`.")

    print("\n== TESTE 1: o BFM segura a caixa? (tarefa `parado c/ caixa`) ==")
    print("   Ele nasce com as palmas TOCANDO a caixa, força normal zero.")
    print("   Segurar é o que a tarefa ensina, então cair a caixa com ação nula")
    print("   é o esperado. O que interessa é QUANTO ele aguenta de pé.")
    r1 = roda(env, termo, T.PARADO_CAIXA, "raisearms-m-m", None)
    print(f"\n  de pé no fim {r1['de_pe_no_fim']:.1%} | "
          f"passos de pé {r1['passos_de_pe']:.0f} de {PASSOS} | "
          f"pelve mín {r1['pelve_min']:.3f} | caixa mín {r1['caixa_min']:.3f} m")
    print("\n  Leitura: se `passos de pé` aqui for MUITO menor que no teste 0, a")
    print("  carga derruba o BFM e o clamp de 0,35 rad na perna é pouco. Se for")
    print("  parecido, o equilíbrio dele aguenta peso e o residual só faz tarefa.")


if __name__ == "__main__":
    main()
