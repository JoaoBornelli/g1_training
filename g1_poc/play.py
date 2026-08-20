"""Visualiza a cena e um checkpoint do g1_poc.

Quatro modos:

    # 1. só a cena, sem política — a verificação da §18
    python -m g1_poc.play --geometria

    # 2. um checkpoint treinado, com a forma sorteada como no treino
    python -m g1_poc.play --checkpoint CAMINHO/model_999.pt

    # 3. só a forma de MANIPULAÇÃO — a caixa está lá e o twist fica zerado
    python -m g1_poc.play --pegar --checkpoint CAMINHO/model_999.pt
    python -m g1_poc.play --pegar --sem-jitter --checkpoint ...   # o caso NOMINAL

    # 4. só a forma de LOCOMOÇÃO — a cena sai da frente e o twist manda
    python -m g1_poc.play --andar --checkpoint CAMINHO/model_999.pt
    python -m g1_poc.play --andar --vx 1.0  --checkpoint ...   # frente, determinístico
    python -m g1_poc.play --andar --giro 0.5 --checkpoint ...  # girar no lugar

O `--sem-jitter` existe para separar duas perguntas que o sorteio mistura: "ele não
pega nunca" e "ele não pega quando a caixa está longe". Com o jitter default a caixa
nasce em `x = 0,32 + U(0 … 0,20)`, e 35% dos episódios caem em `x > 0,45` — a faixa
que a §3.1 chama de inalcançável. Sem jitter ela nasce sempre em (0,32 ; 0,00).

O modo `--geometria` é o portão do passo 0 da §17. Ele custa minutos e o repositório
já perdeu um bloco por não fazê-lo. A lição de 16/07 é: **ataque a geometria, e não
o sintoma.**

⚠ Este arquivo chama `run_play` DIRETO, e nunca `mjlab.scripts.play.main()`. Aquele
reparseia `sys.argv` com tyro, e o primeiro posicional dele é o TASK ID
(`scripts/play.py:308`, via `literal_type_from_choices`). As flags daqui vazavam para
lá, e o erro saía como `Missing value for argument 'value'` — o tyro reclamando do
task id que ninguém passou. Mesmo molde do `g1_multitask/play.py`.
"""
from __future__ import annotations

import argparse
import pathlib

import g1_poc  # noqa: F401  (registra a task)
from mjlab.tasks.registry import register_mjlab_task

from g1_poc.env_cfg import make_g1_poc_env_cfg
from g1_poc.knobs import Knobs

TASK_MANIPULA = g1_poc.TASK_ID + "-Manipula"
TASK_ANDAR = g1_poc.TASK_ID + "-Andar"
"""Ids SEPARADOS por variante. O `register_mjlab_task` NÃO sobrescreve: com o mesmo id
ele levanta `ValueError: Task ... is already registered`. Mesmo padrão do
`g1_multitask/play.py`."""


def imprime_geometria() -> None:
    k = Knobs()
    kc, ka, kt = k.cena, k.alvo, k.tol
    repouso_alto = kc.prateleira_topo_teto + kc.caixa_meia_aresta[2]
    repouso_baixo = kc.prateleira_topo_piso + kc.caixa_meia_aresta[2]
    print("== a verificação da §18, em números ==")
    print(f"  1. topo da prateleira, faixa       : {kc.prateleira_topo_piso:.2f} a "
          f"{kc.prateleira_topo_teto:.2f} m")
    print(f"     fundo da laje no piso           : "
          f"{kc.prateleira_topo_piso - 2*kc.prateleira_meia_z:+.3f} m  "
          f"(0,000 = apoia no chão)")
    print(f"  2. a prateleira ocupa x de         : "
          f"{kc.prateleira_xy[0]-kc.prateleira_meia_xy:.2f} a "
          f"{kc.prateleira_xy[0]+kc.prateleira_meia_xy:.2f} m")
    print(f"     a caixa nasce em x              : {kc.caixa_xy[0]:.2f} m")
    print(f"  3. centro da caixa, prateleira alta: {repouso_alto:.2f} m")
    print(f"     centro da caixa, prateleira baixa: {repouso_baixo:.2f} m")
    print(f"  4. alvo do `pegar`, z              : {ka.pegar_z[0]:.2f} a "
          f"{ka.pegar_z[1]:.2f} m  (MUNDO)")
    print(f"     subida mínima da prateleira alta: "
          f"{(ka.pegar_z[0]-repouso_alto)*100:.0f} cm")
    print(f"     subida mínima da prateleira baixa: "
          f"{(ka.pegar_z[0]-repouso_baixo)*100:.0f} cm")
    print(f"     esfera de sucesso                : {kt.raio_sucesso*100:.0f} cm")
    print()
    print("  Confira no viewer, nesta ordem:")
    print("   a) o tampo NÃO cobre a caixa na prateleira baixa")
    print("   b) a pelve, o tronco e a coxa NÃO tocam o tampo no agachamento")
    print("   c) os pads tocam as FACES da caixa, e não as quinas")
    print("   d) subir o topo para 0,55 m com a caixa a 0,82 m não toca a caixa")
    print("      nem os antebraços (folga esperada: 0,17 m)")
    print()
    print("  Tabela de células (§10.1):")
    cel = k.celulas
    for n in range(len(cel.topo_min)):
        print(f"     nível {n}: topo {cel.topo_min[n]:.2f} m, carga até {cel.carga_max[n]:.1f} kg, "
              f"jitter_x até {cel.jitter_x_max[n]:.2f} m, rotação até {cel.ang_max_deg[n]:.0f}°")


def _registra(task_id: str, ajusta, nivel: int | None = None) -> str:
    """Registra uma task de play com os knobs mutados por `ajusta`.

    Os dois cronogramas por passo global (`twist_ranges`, `hinge`, `action_rate`) já
    saem no `play=True`, portanto as faixas do `Comando` valem como escritas aqui — no
    treino o `mdp.commands_vel` as sobrescreveria.
    """
    k = Knobs()
    ajusta(k)
    env_cfg = make_g1_poc_env_cfg(k, play=True)
    if nivel is not None:
        # o termo `nivel` FICA no play; forçar aqui congela a célula
        env_cfg.curriculum["nivel"].params["nivel_forcado"] = int(nivel)
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg,
        play_env_cfg=env_cfg,
        rl_cfg=g1_poc.rl_cfg(),
    )
    return task_id


def _ajusta_manipula(sem_jitter: bool):
    """`frac_locomocao = 0`: a cena NUNCA é afastada e o twist fica zerado.

    ⚠ Sem isto os modos de manipulação são um sorteio. Com o default de 0,30, três de
    cada dez episódios são de LOCOMOÇÃO, e neles o `afasta_cena` sobe a caixa e a
    prateleira a 5 m — o viewer abre sem mobília nenhuma e não há o que conferir.

    `sem_jitter` fixa a caixa em (0,32 ; 0,00), sem giro, e a prateleira em 0,55. É o
    caso NOMINAL: serve para separar "não pega nunca" de "não pega quando está longe".

    ⚠ O jitter x vem da CÉLULA desde 20/08; o campo `cena.caixa_jitter_x` foi removido.
    """
    def ajusta(k: Knobs) -> None:
        k.episodio.frac_locomocao = 0.0
        if sem_jitter:
            k.celulas.jitter_x_max = (0.0,) * 7
            k.cena.caixa_jitter_y = (0.0, 0.0)
            k.cena.caixa_jitter_yaw_deg = 0.0
            k.cena.prateleira_jitter_z = 0.0
    return ajusta


def _ajusta_andar(vx: float | None, giro: float | None):
    """`frac_locomocao = 1`: só a forma de locomoção, e o twist manda de verdade.

    Três knobs, e os três são necessários:

    1. `frac_locomocao = 1` — o `afasta_cena` sobe a mobília 5 m, o bit `caixa_valida`
       vai a 0 e o `poc_twist_zero` LIBERA o twist. Com o default de 0,30 sete de cada
       dez episódios seriam de manipulação, com o twist zerado: o robô fica parado.
    2. `rel_standing = 0` — um em cada dez envs recebe comando ZERO e fica de pé sem
       andar. Num play de 1 env isso é 10% de chance de não ver nada.
    3. `rel_heading = 0` quando a velocidade é pinada — com `heading_command=True` os
       envs de heading calculam o `ang_vel_z` pelo erro de rumo e IGNORAM a faixa
       pedida. É o que faria um `--giro` fixo não girar.
    """
    def ajusta(k: Knobs) -> None:
        k.episodio.frac_locomocao = 1.0
        k.comando.rel_standing = 0.0
        if vx is not None or giro is not None:
            k.comando.lin_vel_x = (vx or 0.0, vx or 0.0)
            k.comando.lin_vel_y = (0.0, 0.0)
            k.comando.ang_vel_z = (giro or 0.0, giro or 0.0)
            k.comando.rel_heading = 0.0
            k.comando.rel_forward = 0.0
    return ajusta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--geometria", action="store_true",
                   help="imprime os números da §18 e abre a cena, sem política")
    p.add_argument("--pegar", action="store_true",
                   help="só a forma de manipulação: a caixa está lá e o twist é zero")
    p.add_argument("--andar", action="store_true",
                   help="só a forma de locomoção: a cena sai da frente e o twist manda")
    p.add_argument("--sem-jitter", action="store_true",
                   help="--pegar/--geometria: caixa fixa em (0,32; 0,00), o caso nominal")
    p.add_argument("--vx", type=float, default=None,
                   help="--andar: pina a velocidade para frente, em m/s")
    p.add_argument("--giro", type=float, default=None,
                   help="--andar: pina o giro no lugar, em rad/s")
    p.add_argument("--nivel", type=int, default=None,
                   help="--pegar/--geometria: força a célula do nível (§10.1); default = promoção por sucesso")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="caminho de um model_*.pt treinado")
    p.add_argument("--envs", type=int, default=1)
    p.add_argument("--sem-quedas", action="store_true",
                   help="desliga as terminações (o episódio não acaba na queda)")
    p.add_argument("--video", action="store_true")
    p.add_argument("--video-length", type=int, default=500)
    args = p.parse_args()

    modos = [m for m in ("geometria", "pegar", "andar") if getattr(args, m)]
    if len(modos) > 1:
        raise SystemExit(f"escolha UM modo: {modos} são exclusivos")
    if (args.vx is not None or args.giro is not None) and not args.andar:
        raise SystemExit("--vx e --giro só valem com --andar")
    if args.sem_jitter and args.andar:
        raise SystemExit("--sem-jitter não tem efeito no --andar: a cena é afastada")
    if args.nivel is not None:
        if not (0 <= args.nivel <= 6):
            raise SystemExit("--nivel: 0 <= valor <= 6")
        if args.andar:
            raise SystemExit("--nivel não tem efeito com --andar: a mobília é afastada")
    if not args.geometria and not args.checkpoint:
        raise SystemExit("passe --checkpoint CAMINHO, ou use --geometria")

    # o id acompanha o cfg: com e sem jitter são cenas DIFERENTES, e o
    # `register_mjlab_task` não sobrescreve.
    sufixo = "-Nominal" if args.sem_jitter else ""
    if args.nivel is not None:
        sufixo += f"-N{args.nivel}"

    ckpt = None
    if args.geometria:
        if args.checkpoint:
            print("[PLAY] --geometria: o checkpoint é IGNORADO")
        imprime_geometria()
        task_id = _registra(TASK_MANIPULA + sufixo, _ajusta_manipula(args.sem_jitter),
                            nivel=args.nivel)
    else:
        ckpt = pathlib.Path(args.checkpoint).expanduser()
        if not ckpt.is_file():
            raise SystemExit(f"não achei {ckpt}")
        if args.pegar:
            task_id = _registra(TASK_MANIPULA + sufixo,
                                _ajusta_manipula(args.sem_jitter), nivel=args.nivel)
        elif args.andar:
            task_id = _registra(TASK_ANDAR, _ajusta_andar(args.vx, args.giro))
        else:
            task_id = g1_poc.TASK_ID

    # No modo geometria as terminações saem: com o `contato_ilegal` e o `fell_over` de
    # pé a cena reseta antes de dar tempo de olhar. No `--andar` elas FICAM: cair
    # andando é justamente o que se quer medir.
    sem_quedas = args.sem_quedas or args.geometria

    from mjlab.scripts.play import PlayConfig, run_play

    print(f"[PLAY] task {task_id}")
    if args.geometria:
        print("[PLAY] GEOMETRIA — ação zero; o que você vê é a pose padrão no PD")
    if args.geometria or args.pegar:
        print("[PLAY] frac_locomocao = 0 — a cena NUNCA é afastada, o twist é zero")
        print("[PLAY] caixa em x = 0,32 FIXO (caso nominal)" if args.sem_jitter
              else "[PLAY] caixa em x = 0,32 + U(0; 0,20) — 35% cai em x > 0,45")
    if args.pegar:
        print("[PLAY] olhe nesta ordem: as palmas chegam nas FACES -> a caixa sobe ->")
        print("[PLAY]   ela vem para o corpo (o alvo é x 0,20-0,30, ATRAS da caixa)")
    if args.andar:
        print("[PLAY] ANDAR — frac_locomocao = 1: a mobília sobe 5 m e o bit vai a 0")
        if args.vx is not None or args.giro is not None:
            print(f"[PLAY] twist PINADO: vx = {args.vx or 0.0} m/s, "
                  f"giro = {args.giro or 0.0} rad/s (heading desligado)")
        else:
            print("[PLAY] twist SORTEADO nas faixas do `Comando`")
    if sem_quedas:
        print("[PLAY] terminações DESLIGADAS — o episódio não acaba quando ele cai")

    run_play(task_id, PlayConfig(
        agent="zero" if args.geometria else "trained",
        checkpoint_file=None if ckpt is None else str(ckpt),
        num_envs=args.envs,
        viewer="native",
        video=args.video,
        video_length=args.video_length,
        no_terminations=sem_quedas,
    ))


if __name__ == "__main__":
    main()
