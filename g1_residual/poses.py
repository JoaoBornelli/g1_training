"""Escolher o `z` OLHANDO. Os 41 comportamentos do BFM, navegáveis com `<` e `>`.

    python g1_residual/poses.py --lista               # os 41, agrupados
    python g1_residual/poses.py --visor               # abre no 1o, navega com < >
    python g1_residual/poses.py --visor --so-baixas   # só os candidatos a agachar
    python g1_residual/poses.py --visor --em crouch-0.25
    python g1_residual/poses.py --medir               # varredura sem janela

No visor, **duas dimensões**, porque o envelope shippado tem duas:

    <  ou  >     COMPORTAMENTO anterior / próximo   (41 nomes)
    [  ou  ]     SEMENTE anterior / próxima          (10 + a média)
    ENTER        reseta o episódio
    SPACE        pausa      seta-direita  1 passo      -/=  velocidade
    Ctrl+arrasto empurra o robô

O nome e a semente saem no console a cada troca.

⚠️ **São 410 poses, não 41.** Cada um dos 41 comportamentos vem com **10 sementes**,
e elas não são cópias: o `base_z.py` mede as sementes de `move-ego-0-0` a ~60° umas
das outras (cos médio 0,500), porque "não se mexa" tem muitas soluções. Navegar só
a semente 0 esconde 90% do envelope. A média das 10 entra como uma 11a posição, e
ela **só vale onde as 10 concordam** — para comportamento difuso a média pode ser
pior que qualquer semente.

**Por que este arquivo.** O `play.py` visualiza a POLÍTICA treinada. Aqui não há
política: a ação é ZERO, então `c = 0`, o residual é 0 e `z = prior`. O que aparece
é o **BFM puro** rodando um comportamento escolhido a dedo. É o A/B que decide qual
`z` vira prior de cada nível de altura.

**Por que não uso o `run_play`.** Ele constrói o visor com
`NativeMujocoViewer(env, policy).run()` e **não repassa `key_callback`**, então não
dá para navegar. Aqui monto o env e o visor na mão, com o callback.

**O que já se sabe, e por que a varredura antiga não basta.** O docstring do
`base_z.py` registra: `crouch-N` é ALTURA ALVO, não intensidade — `crouch-0` quer
dizer "vai ao chão" e não se sustenta sem residual; `sitonground` senta de fato.
Mas aquela varredura cobriu só `crouch-0`, `move-ego-0-0` e `sitonground`. A
família `low` e o `crouch-0.25` — os candidatos a agachamento EM PÉ — nunca foram
medidos.

⚠️ **Um conflito que a medição expõe.** O critério `de_pe` exige pelve >= 0,65 m
(`terminations.de_pe`). Pelos nomes, `move-ego-low0.6` mira 0,60 e `crouch-0.25`
mira 0,25 — **os dois abaixo do limiar**. Se agachar reprova o `de_pe`, e o `de_pe`
está no sucesso de 6 das 7 tarefas, então alcançar caixa no chão exige **dobrar na
cintura**, não agachar. O `--medir` mede pelve, joelho e cintura juntos, que é o que
separa agachar / curvar / ajoelhar.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import g1_residual  # noqa: E402  (o import registra a task)
import g1_residual.base_z as BZ  # noqa: E402
from g1_multitask import tasks as T  # noqa: E402
from g1_residual.env_residual import OrquestradorPegar  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.viewer.native.keys import (KEY_COMMA, KEY_LEFT_BRACKET,  # noqa: E402
                                      KEY_PERIOD, KEY_RIGHT_BRACKET)

NOMES = {
    "parado": T.PARADO, "andar": T.ANDAR, "pegar": T.PEGAR, "botar": T.BOTAR,
    "reorientar": T.REORIENTAR, "parado_caixa": T.PARADO_CAIXA,
    "andar_caixa": T.ANDAR_CAIXA,
}

Z_DE_PE = 0.65
"""O limiar do `terminations.de_pe`. Está aqui para aparecer na tabela."""

COMP_ANT, COMP_PROX = KEY_COMMA, KEY_PERIOD
"""`<` e `>` chegam como COMMA e PERIOD: o GLFW reporta a tecla sem o shift. Os dois
já são bindados pelo visor para `PREV_ENV`/`NEXT_ENV`, mas com 1 robô isso é no-op e
o callback do usuário roda depois de qualquer jeito (`viewer.py:486`)."""

SEM_ANT, SEM_PROX = KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET
"""`[` e `]`. Livres — o `_safe_key_callback` não os usa."""


# ------------------------------------------------------------------- catálogo
def _grupo(n: str) -> str:
    """Classifica pelo NOME. É leitura de nome, não medição — daí o `--medir`."""
    if n == "sitonground":
        return "SENTA"
    if n.startswith("crouch") or "-low" in n:
        return "BAIXA  <-- candidato"
    if n.startswith(("rotate-z", "spin-arms")):
        return "GIRA"
    if n.startswith("raisearms"):
        return "BRACO (parado)"
    if n.startswith("move-arms"):
        return "BRACO (andando)"
    return "PARADO" if n.endswith("-0-0") else "ANDA"


def _candidatos(z_tabela) -> list[str]:
    return [n for n in sorted(z_tabela)
            if _grupo(n).startswith(("BAIXA", "SENTA")) or n == "move-ego-0-0"]


def lista(z_tabela) -> None:
    por_grupo: dict[str, list[str]] = {}
    for n in sorted(z_tabela):
        por_grupo.setdefault(_grupo(n), []).append(n)
    print(f"{len(z_tabela)} comportamentos.\n")
    for g in sorted(por_grupo, key=lambda k: (not k.startswith("BAIXA"), k)):
        print(g)
        for n in por_grupo[g]:
            print(f"   {n}")
        print()
    print("Convenção dos nomes, deduzida (o `--medir` confirma):")
    print("  crouch-<h>                 altura ALVO da pelve, em metros")
    print("  move-ego-low<h>-<rumo>-<v> pelve baixada a <h>, rumo em graus, v m/s")
    print("  move-ego-<rumo>-<v>        locomoção, pelve normal")
    print("  raisearms-<E>-<D>          braço esquerdo/direito, nível l ou m")
    print(f"\nlimiar `de_pe` do nosso critério: pelve >= {Z_DE_PE} m")


# ------------------------------------------------------------------- montagem
def _monta(tarefa: int, envs: int, sem_terminacao: bool,
           escala_delta: float = 0.0):
    """Env com a tarefa FIXA. Sem registrar task nova: eu não uso o `run_play`, então
    não preciso de id no registro — e assim não colido com o `-PlayCustom` do
    `play.py`.

    `escala_delta = 0.0` desliga o residual, que é o que o visor e a medição querem
    (ação zero já bastaria, mas assim fica explícito). O `autoridade.py` passa o valor
    real do treino, porque lá o ponto É empurrar com o residual."""
    env_cfg = g1_residual.build_env_residual(play=True, escala_delta=escala_delta)

    # ⚠️ Escrever em `env.tarefa_sorteada` não sobrevive: o `_amostrar` do currículo
    # sobrescreve no reset. A única forma que ele respeita é mexer no `abertas`.
    class _Uma(OrquestradorPegar):
        def __init__(self, cfg, env):
            super(OrquestradorPegar, self).__init__(cfg, env)
            self.abertas = [tarefa]
            env.tarefa_sorteada[:] = tarefa

    env_cfg.curriculum["orquestrador"].func = _Uma
    env_cfg.scene.num_envs = envs
    if sem_terminacao:
        env_cfg.terminations = {}
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    return env, env.action_manager.get_term("joint_pos")


def _troca_prior(termo, tarefa: int, nome: str, semente: int | None) -> None:
    base = termo._base
    z = BZ.BaseZ._de(termo._ator.z_tabela, nome, semente)
    base.prior[tarefa] = z.to(base.prior.device)


# ---------------------------------------------------------------------- visor
def visor(z_tabela, ciclo: list[str], inicio: int, semente: int,
          tarefa: int, envs: int) -> None:
    from mjlab.viewer import NativeMujocoViewer

    env, termo = _monta(tarefa, envs, sem_terminacao=True)
    venv = RslRlVecEnvWrapper(env, clip_actions=g1_residual._rl_cfg().clip_actions)
    forma = env.action_space.shape
    dev = env.device

    # Quantas sementes cada comportamento tem. Lido do tensor, não fixado em 10 —
    # se um release trouxer número diferente, o visor acompanha.
    n_sem = {n: int(z_tabela[n].shape[0]) for n in ciclo}
    MEDIA = -1
    """Posição extra depois da última semente. Ver o aviso no docstring do módulo."""

    def _sementes(nome: str) -> list[int]:
        return list(range(n_sem[nome])) + [MEDIA]

    s0 = MEDIA if semente < 0 else min(semente, n_sem[ciclo[inicio]] - 1)
    # `n = None` de propósito: assim o PRIMEIRO `__call__` conta como troca e aplica
    # o prior inicial. Com `n = inicio` nada aplicaria até a primeira tecla.
    est: dict[str, int | None] = {"n": None, "s": None,
                                  "pn": inicio, "ps": s0}

    def anuncia(i: int, s: int) -> None:
        nome = ciclo[i]
        ss = "média" if s == MEDIA else f"{s}"
        print(f"\n[{i + 1}/{len(ciclo)}]  {nome:26} semente {ss:>5}"
              f"/{n_sem[nome]}   ({_grupo(nome)})", flush=True)

    class PoliticaZero:
        """Ação zero, e é AQUI que o prior troca.

        ⚠️ O `key_callback` roda na thread do visor, e o próprio mjlab avisa que ela
        não deve tocar o env (`viewer.py:450`). Então a tecla só anota os índices, e
        a troca acontece neste `__call__`, que roda na thread do sim."""

        def __call__(self, obs):
            del obs
            if (est["pn"], est["ps"]) != (est["n"], est["s"]):
                est["n"], est["s"] = est["pn"], est["ps"]
                nome = ciclo[est["n"]]
                # a semente pedida pode não existir no comportamento novo
                if est["s"] != MEDIA and est["s"] >= n_sem[nome]:
                    est["s"] = est["ps"] = n_sem[nome] - 1
                _troca_prior(termo, tarefa, nome,
                             None if est["s"] == MEDIA else est["s"])
                anuncia(est["n"], est["s"])
            return torch.zeros(forma, device=dev)

    def tecla(k: int) -> None:
        if k == COMP_ANT:
            est["pn"] = (est["pn"] - 1) % len(ciclo)
        elif k == COMP_PROX:
            est["pn"] = (est["pn"] + 1) % len(ciclo)
        elif k in (SEM_ANT, SEM_PROX):
            ss = _sementes(ciclo[est["pn"]])
            i = ss.index(est["ps"]) if est["ps"] in ss else 0
            est["ps"] = ss[(i + (1 if k == SEM_PROX else -1)) % len(ss)]

    total = sum(n_sem[n] for n in ciclo)
    print(f"[POSES] {len(ciclo)} comportamentos x sementes = {total} poses "
          f"(+{len(ciclo)} médias), cena de `{T.NAMES[tarefa]}`, {envs} robô(s)")
    print("[POSES] residual DESLIGADO e ação ZERO — isto é o BFM puro")
    print("[POSES] terminações desligadas: se cair, fica no chão e você vê se levanta")
    print("[POSES] `<` `>` trocam o COMPORTAMENTO. `[` `]` trocam a SEMENTE.")
    print("[POSES] ENTER reseta, SPACE pausa.")
    print("[POSES] a troca é a QUENTE, sem reset — dá para ver a transição. "
          "ENTER se quiser começar limpo.")
    NativeMujocoViewer(venv, PoliticaZero(), key_callback=tecla).run()
    env.close()


# -------------------------------------------------------------------- medição
def _juntas(robot, padrao: str) -> list[int]:
    """Índices das juntas que casam o padrão. Defensivo: o nome do atributo de nomes
    de junta varia entre versões do mjlab, então se não achar, avisa em vez de
    devolver coluna errada."""
    import re
    for attr in ("joint_names", "joint_name_list"):
        nomes = getattr(robot, attr, None)
        if nomes:
            return [i for i, n in enumerate(nomes) if re.search(padrao, n)]
    return []


def medir(nomes: list[str], passos: int, tarefa: int, semente: int,
          todas_sementes: bool) -> None:
    env, termo = _monta(tarefa, envs=1, sem_terminacao=True)
    base = termo._base
    robot = env.scene["robot"]
    tab = termo._ator.z_tabela
    # Os pares a medir. Com `--todas-sementes` a varredura cresce 10x, e é onde ela
    # vale: o `base_z.py` mede as sementes de um mesmo comportamento a ~60° uma da
    # outra, então "o comportamento X agacha" pode ser verdade em 3 das 10.
    pares = ([(n, s) for n in nomes for s in range(int(tab[n].shape[0]))]
             if todas_sementes
             else [(n, None if semente < 0 else semente) for n in nomes])
    i_joelho, i_cintura = _juntas(robot, "knee"), _juntas(robot, "waist")
    if not i_joelho:
        print("⚠️ não achei junta de joelho pelo nome — as colunas de joelho e "
              "cintura sairão zeradas. O resto da tabela vale.")

    acao = torch.zeros(1, env.action_manager.total_action_dim)
    print(f"\n{len(pares)} poses x {passos} passos, ação zero, 1 robô.")
    print(f"{'comportamento':26} {'sem':>4} {'pelve0':>7} {'pelveMIN':>9} "
          f"{'pelveFIM':>9} {'joelhoMAX':>10} {'cintMAX':>8} {'de_pe%':>7}  veredito")
    print("-" * 110)
    for nome, sem in pares:
        # ⚠️ `base.prior` é estado COMPARTILHADO. Sem restaurar, o `z` de um
        # comportamento vaza para o seguinte — foi assim que uma varredura antiga
        # mediu o `pegar` com o prior do `sitonground`.
        salvo = base.prior[tarefa].clone()
        _troca_prior(termo, tarefa, nome, sem)
        try:
            env.reset()
            z0 = float(robot.data.root_link_pos_w[0, 2])
            zmin, jmax, cmax, de_pe = z0, 0.0, 0.0, 0
            for _ in range(passos):
                env.step(acao)
                pz = float(robot.data.root_link_pos_w[0, 2])
                zmin = min(zmin, pz)
                de_pe += int(pz >= Z_DE_PE)
                q = robot.data.joint_pos[0]
                if i_joelho:
                    jmax = max(jmax, float(q[i_joelho].abs().max()))
                if i_cintura:
                    cmax = max(cmax, float(q[i_cintura].abs().max()))
            zfim = float(robot.data.root_link_pos_w[0, 2])
            frac = de_pe / passos
            # os três juntos separam os três modos de descer
            if zfim < 0.35:
                v = "FOI AO CHAO / ajoelhou"
            elif frac > 0.9:
                v = "fica DE PE (nao agacha)"
            elif zmin > 0.45:
                v = "AGACHA EM PE  <-- serve"
            else:
                v = "desce e nao segura"
            ss = "méd" if sem is None else str(sem)
            print(f"{nome:26} {ss:>4} {z0:7.3f} {zmin:9.3f} {zfim:9.3f} "
                  f"{jmax:10.3f} {cmax:8.3f} {frac:7.1%}  {v}")
        finally:
            base.prior[tarefa] = salvo
    env.close()
    print("\n`AGACHA EM PE` é o que serve de prior para os níveis baixos de altura.")
    print(f"`de_pe%` abaixo de 100% já indica conflito com o critério "
          f"(pelve >= {Z_DE_PE}).")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lista", action="store_true", help="imprime os 41 agrupados")
    p.add_argument("--visor", action="store_true",
                   help="abre o visor; `<` e `>` navegam")
    p.add_argument("--medir", action="store_true",
                   help="varredura sem janela: pelve, joelho, cintura")
    p.add_argument("--so-baixas", action="store_true",
                   help="restringe o ciclo aos candidatos a agachar (+ controle)")
    p.add_argument("--todas-sementes", action="store_true",
                   help="com --medir, varre as 10 sementes de cada comportamento. "
                        "Multiplica a varredura por 10 — use junto de --so-baixas")
    p.add_argument("--em", type=str, default=None, metavar="NOME",
                   help="em qual comportamento começar")
    p.add_argument("--semente", type=int, default=0,
                   help="qual das 10 sementes. -1 usa a MÉDIA, que só vale onde as "
                        "10 concordam (ver a tabela no docstring de PRIOR)")
    p.add_argument("--tarefa", choices=sorted(NOMES), default="parado",
                   help="qual cena montar. `parado` não tem interferência da caixa")
    p.add_argument("--envs", type=int, default=1)
    p.add_argument("--passos", type=int, default=250)
    args = p.parse_args()

    z_tabela = torch.load(
        pathlib.Path(__file__).resolve().parent / "peso" / "bfm_ator.pt",
        weights_only=True, map_location="cpu")["z"]
    tarefa = NOMES[args.tarefa]

    if args.lista:
        lista(z_tabela)
        return

    ciclo = _candidatos(z_tabela) if args.so_baixas else sorted(z_tabela)
    inicio = 0
    if args.em:
        assert args.em in ciclo, (
            f"`{args.em}` não está no ciclo — rode `--lista`, ou tire `--so-baixas`")
        inicio = ciclo.index(args.em)

    if args.medir:
        # sem `--so-baixas` a varredura fica nos candidatos: os 41 x 250 passos já
        # são 10 250 passos, e com `--todas-sementes` viraria 102 500.
        medir(ciclo if args.so_baixas else _candidatos(z_tabela),
              args.passos, tarefa, args.semente, args.todas_sementes)
        return
    if args.visor:
        visor(z_tabela, ciclo, inicio, args.semente, tarefa, args.envs)
        return
    p.print_help()


if __name__ == "__main__":
    main()
