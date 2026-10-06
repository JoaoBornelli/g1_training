"""Mede quanto as juntas SE MOVEM com o robô parado, por junta, em GRAUS.

    python g1_multitask/variacao_pose.py reference_checkpoints/model_2000_stand_step.pt --task Stand-Step
    python g1_multitask/variacao_pose.py ~/Downloads/model_0.pt --task Multitask --tarefa parado

Roda headless, em CPU, sem viewer, pelo MESMO caminho do `play`:
`load_env_cfg(play=True)`, `RslRlVecEnvWrapper`, `runner.load(load_cfg={"actor": True})`,
`get_inference_policy` — a política determinística (a MÉDIA), sem o `std` de exploração.

**A grandeza é VARIAÇÃO, não desvio.** As duas respondem coisas diferentes, e confundi-las
foi o erro da primeira versão deste arquivo:

    desvio    |alvo − default|             quão LONGE da pose padrão ele comanda
    variação  espalhamento do alvo no tempo  quanto ele MEXE enquanto está parado

O desvio pode ser grande e constante — postura própria, imóvel. A variação é o que
distingue **controle ativo** de **saída congelada**: um robô parado de verdade micro-corrige
sem parar, porque a política reage ao que o corpo faz. Uma rede não treinada devolve
quase a mesma coisa a cada passo (a obs muda pouco e ela é uma função suave), então mede
variação perto de zero **mesmo ficando de pé** — que é de graça, pelo
`use_default_offset` + PD na pose padrão do G1 (base 0,784 m, joelho 0,3 rad).

**Duas variações, porque são causas diferentes:**

    ALVO    o que a POLÍTICA comanda      atividade da rede
    JUNTA   onde a junta de fato ficou    física; mexe um pouco mesmo com alvo fixo

**Por que a referência é o `Stand-Step` it 2000** (`reference_checkpoints/README.md`, a
skill de ficar-de-pé separada). Ela é controle limpo contra o `parado` do multi-tarefa, por
três motivos lidos no código:

1. a skill Stand **não tem comando de velocidade** — `skills/stand/env.py` é "ficar de pé
   parado, a fundação de equilíbrio pura". A variação medida não carrega locomoção;
2. no `play` o `build_base_env` **remove o `push_robot`**, e o `push_force` do Stand-Step
   só entra com `not play`. Sem perturbação externa nos dois lados;
3. caixa e prateleira ficam fora de alcance útil de propósito (fix da muleta, 15/07).

⚠️ A variação é medida **dentro de segmentos sem reset**, por env. Pooling ingênuo sobre a
janela inteira somaria a diferença ENTRE episódios (a pose de reset é sorteada,
`_RESET_BASE_POSE_RANGE`) à variação de dentro de um, e isso infla justamente quem recupera
bem. Segmento curto é descartado — std de 5 amostras é ruído.

⚠️ O alvo sai de `raw_action × scale`: é a via pública, e dá o mesmo número que
`processed_actions − offset` quando `use_default_offset = True` e `clip is None`. O script
confere as duas e aborta se alguma cair.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import asdict

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import (  # noqa: E402
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)

SKILLS = ("Stand", "Stand-Step", "Lift")
"""Tasks do `g1_training`. Id `Mjlab-Lift-Box-Unitree-G1-<nome>`, igual ao `play.py` da
raiz."""

MIN_SEGMENTO = 30
"""Passos mínimos num segmento para ele contar. Abaixo disso o desvio-padrão no tempo é
ruído de amostra, não medida de movimento."""


def _task_id(task: str, tarefa: str | None) -> str:
    """Resolve o id e, no multi-tarefa, fixa a tarefa reusando o `_registra` do play.

    Reusa em vez de duplicar: o orquestrador `_Fixo` é o único jeito que o currículo
    respeita (escrever em `env.tarefa_sorteada` não tem efeito — o `_amostrar` sobrescreve
    no reset), e duas cópias dessa lógica divergem."""
    if task == "Multitask":
        import g1_multitask  # noqa: F401  (o import registra a task)
        from g1_multitask.play import NOMES, _registra

        if tarefa is None:
            return g1_multitask.TASK_ID
        return _registra(NOMES[tarefa], None)

    if tarefa is not None:
        raise SystemExit("--tarefa só existe no --task Multitask")
    import g1_training  # noqa: F401  (o import registra Stand/Stand-Step/Lift)

    return f"Mjlab-Lift-Box-Unitree-G1-{task}"


def roda(ckpt: pathlib.Path, task: str, tarefa: str | None, envs: int,
         passos: int, descartar: int, device: str) -> dict:
    """Roda a política e devolve as séries temporais brutas, em GRAUS.

    `alvo` é o desvio comandado em relação à pose padrão; `junta` é a posição real da
    junta. Os dois em [T, N, 29]. `dones` em [T, N]."""
    task_id = _task_id(task, tarefa)

    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = envs
    agent_cfg = load_rl_cfg(task_id)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    politica = runner.get_inference_policy(device=device)

    termo = env.unwrapped.action_manager.get_term("joint_pos")
    assert termo.cfg.use_default_offset, (
        "sem `use_default_offset` o `offset` não é a pose padrão e `raw × scale` deixa de "
        "ser desvio de pose")
    assert termo.cfg.clip is None, (
        "há clip no termo: `raw × scale` passa a superestimar o alvo de verdade")
    robo = env.unwrapped.scene["robot"]
    ids = termo.target_ids

    obs, _ = env.reset()
    alvo, junta, dones = [], [], []
    for k in range(descartar + passos):
        with torch.no_grad():
            acao = politica(obs)
        obs, _, done, _ = env.step(acao)
        if k >= descartar:
            alvo.append((termo.raw_action * termo.scale).clone())
            junta.append(robo.data.joint_pos[:, ids].clone())
            dones.append(done.clone())

    env.close()
    return {"nomes": list(termo.target_names),
            "alvo": torch.rad2deg(torch.stack(alvo)).cpu(),
            "junta": torch.rad2deg(torch.stack(junta)).cpu(),
            "dones": torch.stack(dones).cpu().bool()}


def _segmentos(dones: torch.Tensor) -> list[tuple[int, int, int]]:
    """(env, início, fim) dos trechos sem reset, com pelo menos `MIN_SEGMENTO` passos."""
    T, N = dones.shape
    out = []
    for n in range(N):
        ini = 0
        for t in range(T):
            if dones[t, n]:
                if t + 1 - ini >= MIN_SEGMENTO:
                    out.append((n, ini, t + 1))
                ini = t + 1
        if T - ini >= MIN_SEGMENTO:
            out.append((n, ini, T))
    return out


def _por_junta(serie: torch.Tensor, segs: list[tuple[int, int, int]]) -> dict:
    """Estatísticas de movimento por junta, agregando segmentos pela MEDIANA.

    Mediana e não média entre segmentos: um segmento que pegou uma quase-queda tem
    espalhamento muito maior e arrastaria a média."""
    stds, amps, jits = [], [], []
    for n, a, b in segs:
        s = serie[a:b, n, :]                      # [t, 29]
        stds.append(s.std(dim=0))
        amps.append(s.quantile(0.95, dim=0) - s.quantile(0.05, dim=0))
        jits.append((s[1:] - s[:-1]).abs().median(dim=0).values)
    return {"std": torch.stack(stds).median(dim=0).values,
            "amp": torch.stack(amps).median(dim=0).values,
            "jitter": torch.stack(jits).median(dim=0).values}


def relatorio(d: dict, rotulo: str) -> None:
    nomes, segs = d["nomes"], _segmentos(d["dones"])
    T, N, _ = d["alvo"].shape
    print(f"\n{'=' * 86}")
    print(f"{rotulo}")
    print(f"{T} passos × {N} envs   |   {len(segs)} segmentos sem reset "
          f"({int(d['dones'].sum())} resets na janela)")
    print(f"{'=' * 86}")
    if not segs:
        print(f"nenhum segmento com {MIN_SEGMENTO}+ passos — o robô está caindo sem "
              f"parar. Aumente --passos ou olhe o `play`, não esta tabela.")
        return

    A, J = _por_junta(d["alvo"], segs), _por_junta(d["junta"], segs)
    print(f"{'junta':<28} {'ALVO std':>9} {'ALVO amp':>9} {'ALVO/passo':>11} "
          f"{'JUNTA std':>10}")
    for i in A["amp"].argsort(descending=True).tolist():
        print(f"{nomes[i]:<28} {float(A['std'][i]):>8.3f}° {float(A['amp'][i]):>8.3f}° "
              f"{float(A['jitter'][i]):>10.3f}° {float(J['std'][i]):>9.3f}°")

    print("\nRESUMO entre as 29 juntas (mediana / p95 / máx):")
    for nome, v in (("variação do ALVO  (std)", A["std"]),
                    ("variação do ALVO  (amp p05-p95)", A["amp"]),
                    ("movimento por PASSO do alvo", A["jitter"]),
                    ("variação da JUNTA (std)", J["std"])):
        print(f"  {nome:<34} {float(v.quantile(0.50)):>7.3f}° "
              f"{float(v.quantile(0.95)):>7.3f}° {float(v.max()):>7.3f}°")

    # Contexto, não a medida: o desvio SUSTENTADO em relação à pose padrão. Serve para
    # ler a variação — 15° de postura fixa com 0,1° de variação é postura, não controle.
    desvio = d["alvo"].abs().flatten()
    print(f"\n  (contexto) desvio |alvo-default|      "
          f"{float(desvio.quantile(0.50)):>7.2f}° {float(desvio.quantile(0.95)):>7.2f}° "
          f"{float(desvio.max()):>7.2f}°")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--task", default="Stand-Step", choices=(*SKILLS, "Multitask"))
    p.add_argument("--tarefa", default=None,
                   help="só com --task Multitask: fixa uma das 7")
    p.add_argument("--envs", type=int, default=8,
                   help="mais envs = mais segmentos, e cobre mais poses de reset")
    p.add_argument("--passos", type=int, default=200)
    p.add_argument("--descartar", type=int, default=50,
                   help="passos jogados fora no começo (transiente do reset)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt = pathlib.Path(args.checkpoint).expanduser()
    assert ckpt.is_file(), f"não achei {ckpt}"

    d = roda(ckpt, args.task, args.tarefa, args.envs, args.passos,
             args.descartar, args.device)
    rotulo = f"{ckpt.name}   task={args.task}" + (
        f"  tarefa={args.tarefa}" if args.tarefa else "")
    relatorio(d, rotulo)


if __name__ == "__main__":
    main()
