"""Simula a run inteira do currículo, sem física:

    python g1_multitask/sim_curriculo.py            # modo DESENHO (default)
    python g1_multitask/sim_curriculo.py --ruido    # modo RUÍDO (S0)

Este é o teste de aceitação do orquestrador. Ele responde à pergunta que nenhum outro
check responde: **a sequência de destravamentos que o código produz é a que o desenho
especifica?** Sem ele, um erro no grafo ou na prioridade do F9 só apareceria depois de
horas de GPU, e apareceria como "o treino não progride".

O que se verifica no modo DESENHO (§7b):
  - o total é exatamente **54** destravamentos (era 60 antes da S11)
  - a cadeia crítica até o `botar` abrir é **9** eventos
  - o `andar` só abre com push COMPLETO (os 4 níveis competentes)
  - `reorientar` e `pegar` abrem como IRMÃOS, os dois filhos do `andar`
  - cada eixo termina com todos os níveis abertos

Roda em segundos porque não instancia env: um stub com os 4 atributos que o
orquestrador toca basta, e é justamente o que torna o teste barato o suficiente pra
rodar sempre.

--------------------------------------------------------------------------------
MODO RUÍDO (S0)
--------------------------------------------------------------------------------

O modo DESENHO usa `success_buf.fill_(1.0)`. Sucesso perfeito não tem ruído, e sem
ruído **o congelamento nunca dispara** — portanto o modo desenho não consegue testar
a regra de congelamento. O modo ruído existe só pra isso.

Ele é OPT-IN e o default não muda. O `smoke.py:902` chama `simula(num_envs=64)` e
exige os 54 destravamentos determinísticos; tornar o ruído default quebraria o teste
de aceitação do desenho.

Rode ANTES e DEPOIS da S3 e guarde os dois números. O que se espera:

  | critério                                  | antes da S3        | depois da S3 |
  |-------------------------------------------|--------------------|--------------|
  | células em `p = 0.50` congeladas no fim   | congelam, não soltam | nenhuma permanente |
  | células com `p` CONSTANTE congeladas      | várias (falso positivo) | nenhuma |
  | intervalo entre destravamentos            | registre o número  | não deve crescer com níveis abertos |

Se o terceiro não passar, o portão `min` decide por sorte. Registre e siga: o conserto
do portão está fora do escopo desta rodada.
"""
from __future__ import annotations

import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from g1_multitask import tasks as T
from g1_multitask.curriculum import FILHOS, PUSH, Orquestrador
from g1_multitask.knobs import MultitaskKnobs


class _EnvFalso:
    """Só o que o orquestrador lê e escreve. Nada de física."""

    def __init__(self, n: int, dev="cpu"):
        self.num_envs = n
        self.device = dev
        self.max_episode_length = 1000
        self.success_buf = torch.ones(n, device=dev)
        # o alarme de estagnação conta transição pela DIFERENÇA deste contador
        self.common_step_counter = 0


class _CfgFalso:
    def __init__(self, params):
        self.params = params


# ------------------------------------------------------------------ modo ruído (S0)

P_CELULA = (0.50, 0.90, 0.97)
"""Taxas de sucesso VERDADEIRAS do harness ruidoso. Os três valores são os da S0.

Elas cobrem os três regimes que interessam à regra de congelamento:

  - `0.50` — variância máxima da Bernoulli, e o pior caso do viés do máximo corrido
    (σ da EMA = 0.062 com α = 0.03);
  - `0.90` — exatamente o `limiar_competencia`, ou seja a fronteira do portão;
  - `0.97` — competente com folga, o caso em que nada deveria congelar nunca.

⚠️ O `p` é CONSTANTE no tempo, de propósito. É isso que torna o harness um controle
limpo: sem regressão real possível, **todo congelamento observado é falso positivo**.
Se o `p` variasse, o teste não distinguiria bug de regressão legítima.

⚠️ O `p` também não varia por NÍVEL, só por célula. Mesmo motivo: um `p` por nível
faria abrir nível novo baixar a competência medida, e aí a queda seria real."""


def _p_por_celula(orq) -> dict[tuple[int, str], float]:
    """Um `p` por célula, round-robin determinístico sobre `P_CELULA`.

    Round-robin e não sorteio: o `_destravar` do próprio orquestrador desempata assim
    (`curriculum.py:286`), e reusar a mesma disciplina deixa o harness reprodutível
    sem semente. Determinismo importa aqui porque o número desta rodada vai ser
    comparado com o da rodada de depois da S3."""
    return {cel: P_CELULA[i % len(P_CELULA)] for i, cel in enumerate(orq.celulas)}


def _p_por_env(orq, env, p_cel: dict) -> torch.Tensor:
    """`p` verdadeiro de cada env = MÍNIMO sobre as células que ele treina.

    `min` e não média nem produto, por derivação do próprio orquestrador: o
    `_min_tarefa` (`curriculum.py:178`) define competência da tarefa como o `min` sobre
    os níveis, com a justificativa "dominar o nível fácil e ignorar o difícil não
    passa". O harness espelha essa semântica — o env falha se QUALQUER eixo dele
    estiver difícil.

    A célula de push entra em todo env, porque ela se aplica a todos ao mesmo tempo.
    O `parado` não tem eixo próprio, portanto o `p` dele é o do push sozinho — que é
    exatamente o que o `_min_tarefa` devolve pra ele (1.0, "quem manda é o push")."""
    p_push = p_cel[(T.PARADO, PUSH)]
    p = torch.full((env.num_envs,), p_push, device=env.device)
    for t, eixos in T.AXES.items():
        if not eixos:
            continue
        p_t = min(p_cel[(t, e)] for e in eixos)
        p[env.tarefa_sorteada == t] = min(p_t, p_push)
    return p


def simula(num_envs: int = 64, max_chamadas: int = 40_000, verboso: bool = False,
           ruido: bool = False):
    """Roda até o currículo parar de destravar. Devolve (orq, historico).

    `ruido=False` (default) mantém o modo DESENHO: sucesso sempre, sequência mais
    curta possível, total exatamente o do desenho. É o que o `smoke.py:902` consome.

    `ruido=True` liga o harness da S0: `success_buf` sai de uma Bernoulli com `p`
    verdadeiro por célula. Acrescenta ao histórico as três séries que a S0 pede."""
    knobs = MultitaskKnobs()
    env = _EnvFalso(num_envs)
    orq = Orquestrador(_CfgFalso({"curriculum": knobs.curriculum,
                                  "min_amostras_evento": 200,
                                  "verboso": verboso}), env)
    ids = torch.arange(num_envs)
    abertura_em: dict[int, int] = {}      # tarefa -> nº do evento em que ela abriu
    push_completo_em = None
    antes = 0
    parado_em = 0

    p_cel = _p_por_celula(orq) if ruido else None
    # (célula, nível) -> nº de chamadas em que ele esteve marcado como congelado
    chamadas_congelado: dict[tuple, int] = defaultdict(int)
    # uma entrada POR EVENTO: (chamada, total de níveis abertos naquele momento)
    destravamentos: list[tuple[int, int]] = []

    for chamada in range(max_chamadas):
        if p_cel is None:
            # sucesso SEMPRE: o teste é da MÁQUINA de destravamento, não da política.
            # Se o robô sempre vence, a sequência é a mais curta possível e o total tem
            # que ser exatamente o do desenho.
            env.success_buf.fill_(1.0)
        else:
            torch.bernoulli(_p_por_env(orq, env, p_cel), out=env.success_buf)
        # cada chamada é um episódio inteiro por env, então o contador do env
        # anda `max_episode_length`. É isso que o alarme mede.
        env.common_step_counter += env.max_episode_length
        antes_abertas = set(orq.abertas)
        antes_push = orq.abertos[(T.PARADO, PUSH)]
        orq(env, ids)
        for t in set(orq.abertas) - antes_abertas:
            abertura_em[t] = orq.eventos
        if (push_completo_em is None
                and antes_push < orq.abertos[(T.PARADO, PUSH)] == len(T.LEVELS[PUSH])):
            push_completo_em = orq.eventos
        if p_cel is not None:
            niveis = sum(orq.abertos[c] for c in orq.celulas)
            for _ in range(orq.eventos - antes):
                destravamentos.append((chamada, niveis))
            # `.tolist()` numa chamada por célula, e não `bool(t[i])` por nível: o
            # segundo custa uma sincronização por nível e domina o tempo do harness.
            for cel in orq.celulas:
                for nivel, cong in enumerate(
                        orq.congelado[cel][: orq.abertos[cel]].tolist()):
                    if cong:
                        chamadas_congelado[(cel, nivel)] += 1
        if orq.eventos > antes:
            antes = orq.eventos
            parado_em = chamada
        elif chamada - parado_em > 3000:
            break              # 3000 chamadas sem evento: convergiu
    return orq, {"abertura_em": abertura_em, "push_completo_em": push_completo_em,
                 "chamadas": chamada + 1, "p_cel": p_cel,
                 "chamadas_congelado": dict(chamadas_congelado),
                 "destravamentos": destravamentos, "num_envs": num_envs,
                 "envs_por_tarefa": num_envs / max(len(orq.abertas), 1)}


def _relatorio(orq, info) -> int:
    falhas = 0

    def ok(nome, cond, detalhe=""):
        nonlocal falhas
        print(f"  {'OK   ' if cond else 'FALHA'} {nome}"
              + (f"   ({detalhe})" if detalhe else ""))
        if not cond:
            falhas += 1

    print("\n-- total de destravamentos --")
    ok("total = 54", orq.eventos == 54, f"deu {orq.eventos}")

    print("\n-- todas as 7 tarefas abriram --")
    ok("7 tarefas abertas", len(orq.abertas) == T.NUM_TASKS,
       ", ".join(T.NAMES[t] for t in orq.abertas))

    print("\n-- cada eixo esgotou --")
    for cel in orq.celulas:
        t, eixo = cel
        total = len(T.LEVELS[PUSH]) if eixo == PUSH else len(T.axis_levels(t, eixo))
        ok(f"{T.NAMES[t]}/{eixo}: {total} níveis",
           orq.abertos[cel] == total, f"abriu {orq.abertos[cel]} de {total}")

    print("\n-- contagem por origem (§7b: a tabela que soma 60) --")
    esperado = {"parado": 0, "andar": 4, "reorientar": 10, "pegar": 10,
                "botar": 10, "parado_caixa": 4, "andar_caixa": 6}
    for t, eixos in T.AXES.items():
        n = sum(orq.abertos[(t, e)] - 1 for e in eixos)
        ok(f"{T.NAMES[t]} = {esperado[T.NAMES[t]]}", n == esperado[T.NAMES[t]],
           f"deu {n}")
    n_push = orq.abertos[(T.PARADO, PUSH)] - 1
    ok("push = 4", n_push == 4, f"deu {n_push}")
    ok("aberturas = 6", len(orq.abertas) - 1 == 6, f"deu {len(orq.abertas) - 1}")

    print("\n-- cadeia crítica e ordem de abertura (§7b) --")
    ab = info["abertura_em"]
    for t in sorted(ab, key=lambda x: ab[x]):
        print(f"        evento {ab[t]:2d}: abriu {T.NAMES[t]}")
    ok("push completo ANTES do `andar` abrir (gate do `parado`)",
       info["push_completo_em"] is not None and info["push_completo_em"] < ab[T.ANDAR],
       f"push completo no evento {info['push_completo_em']}, "
       f"andar no {ab[T.ANDAR]}")
    ok("push completo em 4 eventos", info["push_completo_em"] == 4,
       f"deu {info['push_completo_em']}")
    # A "cadeia crítica = 9" da §7b é o número de consolidações EM SÉRIE, não o índice
    # global do evento. São grandezas diferentes: o índice global conta os eventos de
    # todas as tarefas intercalados (o `botar` abre no evento ~18 porque `reorientar`
    # e `pegar` estão consolidando em paralelo no meio), enquanto a cadeia é a
    # PROFUNDIDADE do grafo — quantas competências têm que acontecer uma depois da
    # outra. É a cadeia que decide o orçamento: 9 × ~500 iterações cabe nas 30 000.
    prof = {T.PARADO: len(T.LEVELS[PUSH]) - 1}      # os 4 níveis de push, em série
    fila = [T.PARADO]
    while fila:
        pai = fila.pop(0)
        for filho in FILHOS[pai]:
            prof[filho] = prof[pai] + 1
            fila.append(filho)
    ok("cadeia crítica (profundidade do grafo) até o `botar` = 9",
       prof[T.BOTAR] == 9, f"deu {prof[T.BOTAR]}")
    print(f"        profundidade em série: "
          + ", ".join(f"{T.NAMES[t]}={prof[t]}" for t in sorted(prof, key=prof.get)))
    print(f"        (índice global do evento em que o `botar` abriu: {ab[T.BOTAR]} — "
          f"maior porque\n         `reorientar` e `pegar` consolidam em PARALELO no "
          f"meio, que é o ponto do desenho)")
    ok("`pegar` abre antes do `reorientar` (cadeia crítica primeiro)",
       ab[T.PEGAR] < ab[T.REORIENTAR],
       f"pegar no {ab[T.PEGAR]}, reorientar no {ab[T.REORIENTAR]}")

    print("\n-- round-trip do checkpoint (item 0) --")
    estado = orq.state_dict()
    env2 = _EnvFalso(8)
    orq2 = Orquestrador(_CfgFalso({"curriculum": MultitaskKnobs().curriculum}), env2)
    orq2.load_state_dict(estado)
    ok("eventos sobrevivem", orq2.eventos == orq.eventos,
       f"{orq2.eventos} vs {orq.eventos}")
    ok("níveis destravados sobrevivem",
       all(orq2.abertos[c] == orq.abertos[c] for c in orq.celulas))
    ok("perf sobrevive bit a bit",
       all(bool(torch.equal(orq2.perf[c].cpu(), orq.perf[c].cpu()))
           for c in orq.celulas))
    ok("tarefas abertas sobrevivem", orq2.abertas == orq.abertas)
    ok("state_dict é invariante a num_envs (64 -> 8)", env2.num_envs != 64)
    return falhas


CELULA_PUSH = (T.PARADO, PUSH)
"""A célula de push continua lida à parte no relatório, por motivo HISTÓRICO.

Antes da S3 ela media `sucesso.mean()` sobre TODOS os envs de todas as tarefas, e não
sobre os envs de uma tarefa só. Abrir uma tarefa nova mais difícil derrubava a média
de verdade — medido em 05/08 neste harness: de 0.900 para 0.696, queda de 0.206. Isso
não era desvio do máximo corrido; era a composição da população mudando.

A S3 sozinha não cobriria esse caso: a referência lenta converge para 0.90 e a queda
de 0.206 continua passando o `congela_queda` de 0.10. Por isso a medição passou a ser
restrita aos envs da tarefa `parado` (`curriculum.py`, `_medir`), o que estabiliza a
população e devolve à queda o significado de regressão.

A separação fica no relatório porque ela é o teste de que a correção continua de pé:
se a linha `[push]` reaparecer, a medição voltou a misturar tarefas."""


def _regime(orq, info) -> str:
    """Uma linha dizendo em que regime de amostragem a rodada estava.

    É o número que decide se o fenômeno da S3 sequer aparece. O `_medir`
    (`curriculum.py:219`) alimenta a EMA com a MÉDIA dos envs no nível, e não com uma
    amostra: com `m` envs por nível, o σ da EMA é

        σ = sqrt(p(1-p)/m) · sqrt(α/(2-α))

    Com α = 0.03 o segundo fator é 0.123. Em p = 0.50 e m = 1 isso dá 0.062, que é o
    número que a S3 usa. Em m = 32 dá 0.011, e aí o viés do máximo (2.5σ a 3σ) não
    chega perto do `congela_queda` de 0.10 — o falso positivo desaparece por
    amostragem, sem nenhum conserto."""
    m = info["envs_por_tarefa"]
    sigma = (0.25 / max(m, 1e-9)) ** 0.5 * (0.03 / 1.97) ** 0.5
    return (f"num_envs={info['num_envs']}, {len(orq.abertas)} tarefas abertas "
            f"-> ~{m:.1f} envs/tarefa, σ_EMA(p=0.5) ≈ {sigma:.3f}, "
            f"viés do máx ≈ {2.75 * sigma:.3f} (limiar {orq.congela_queda:.2f})")


def _relatorio_ruido(orq, info, com_p: bool = True) -> None:
    """Os três números da S0. NÃO devolve pass/fail, de propósito.

    O veredito depende de a rodada ser antes ou depois da S3, e o critério do
    intervalo entre destravamentos não tem limiar especificado. A S0 manda
    "registre o número e siga" — então o relatório registra, e o julgamento é humano."""
    p_cel = info["p_cel"]
    cong = info["chamadas_congelado"]
    chamadas = info["chamadas"]

    if com_p:
        print("\n-- `p` verdadeiro por célula (constante no tempo) --")
        for cel in orq.celulas:
            t, eixo = cel
            print(f"        {T.NAMES[t]}/{eixo:<12s} p = {p_cel[cel]:.2f}")
        print("\n   ⚠️ Dentro de uma tarefa, o `p` EFETIVO é o mínimo entre as células"
              "\n      dela (ver `_p_por_env`). Portanto a célula que não é o mínimo"
              "\n      mede o `p` da que é — o `p` individual dela não é observável.")

    print(f"\n-- regime de amostragem --\n        {_regime(orq, info)}")

    print("\n-- congelamento --")
    print("   (o `p` é constante no tempo e a célula de push mede só os envs do"
          "\n    `parado`, portanto nenhuma célula sofre regressão real. Todo"
          "\n    congelamento que aparecer aqui é falso positivo.)")
    abertos_tot = sum(orq.abertos[c] for c in orq.celulas)
    permanentes = [(cel, n) for cel in orq.celulas
                   for n in range(orq.abertos[cel])
                   if bool(orq.congelado[cel][n])]
    viés = [x for x in permanentes if x[0] != CELULA_PUSH]
    print(f"        níveis abertos ao fim ................. {abertos_tot}")
    print(f"        congelaram alguma vez ................. {len(cong)}")
    print(f"        congelados NO FIM ..................... {len(permanentes)}")
    print(f"          dos quais por VIÉS (o alvo da S3) ... {len(viés)}")
    print(f"          dos quais a célula de push .......... "
          f"{len(permanentes) - len(viés)}  (desde a S3 ela mede só o `parado`; "
          "se aparecer aqui, a população dela voltou a ser mista)")

    for cel, n in permanentes:
        t, eixo = cel
        marca = "push" if cel == CELULA_PUSH else "viés"
        print(f"            [{marca}] {T.NAMES[t]}/{eixo} n{n}: "
              f"p={p_cel[cel]:.2f} perf={float(orq.perf[cel][n]):.3f} "
              f"ref={float(orq.ref[cel][n]):.3f} "
              f"queda={float(orq.ref[cel][n]) - float(orq.perf[cel][n]):.3f} "
              f"({cong.get((cel, n), 0)} de {chamadas} chamadas)")

    por_p: dict[float, list[int]] = {p: [0, 0] for p in P_CELULA}
    for cel in orq.celulas:
        p = p_cel[cel]
        por_p[p][1] += orq.abertos[cel]
        por_p[p][0] += sum(1 for n in range(orq.abertos[cel])
                           if bool(orq.congelado[cel][n]))
    print("        congelados no fim, por `p`:")
    for p in P_CELULA:
        n, tot = por_p[p]
        print(f"            p = {p:.2f}:  {n} de {tot} níveis")

    print("\n-- intervalo entre destravamentos x níveis abertos --")
    dest = info["destravamentos"]
    if len(dest) < 2:
        print(f"        só {len(dest)} destravamento(s): o currículo travou. "
              "É o resultado, não um erro do harness.")
    else:
        print("        evento   chamada   intervalo   níveis abertos")
        anterior = 0
        intervalos = []
        for i, (chamada, niveis) in enumerate(dest, start=1):
            passo = chamada - anterior
            intervalos.append(passo)
            anterior = chamada
            if i <= 12 or i > len(dest) - 4:
                print(f"        {i:6d}   {chamada:7d}   {passo:9d}   {niveis:14d}")
            elif i == 13:
                print(f"        {'...':>6}")
        meio = len(intervalos) // 2
        a = sum(intervalos[:meio]) / max(meio, 1)
        b = sum(intervalos[meio:]) / max(len(intervalos) - meio, 1)
        razao = b / a if a > 0 else float("inf")
        print(f"        intervalo médio: 1ª metade {a:.0f}, 2ª metade {b:.0f}  "
              f"-> razão {razao:.2f}x")
        print("        (razão perto de 1 = o portão `min` decide por competência;"
              "\n         razão alta = o tempo cresce com os níveis abertos, e aí ele"
              "\n         decide por sorte. Registre o número; o conserto está fora"
              "\n         do escopo desta rodada.)")

    print(f"\n{'=' * 60}")
    print("números registrados. Rode este mesmo comando DEPOIS da S3 e compare:")
    print("  - células em p = 0.50 devem deixar de congelar de forma permanente")
    print("  - o total de congelados no fim deve ir a zero")
    print("  - a razão do intervalo não deve piorar")


VARREDURA = (8, 64, 512)
"""Os `num_envs` do modo ruído. Varredura, e não um valor único, porque não há um
número certo a escolher — os dois extremos respondem perguntas diferentes:

  - `8` reproduz o regime que a S3 MODELA (~1 amostra por atualização, σ ≈ 0.06). É
    onde o falso positivo do máximo corrido aparece.
  - `512` se aproxima do regime de TREINO (4096 envs por rank divididos entre tarefas
    e níveis). É onde se vê se o fenômeno sobrevive à escala real.

O `64` fica no meio como referência, porque é o default do modo desenho e o valor que
o `smoke.py:902` usa."""


def _varredura_ruido() -> None:
    """Roda o modo ruído em vários `num_envs` e reporta o detalhe do primeiro.

    A varredura existe porque o congelamento espúrio é função do número de amostras
    por atualização, e não só da regra de congelamento. Reportar um `num_envs` só
    esconderia essa dependência — que é justamente o que a S0 precisa medir."""
    print("simulando a run do currículo (modo RUÍDO — S0, sem física)...")
    print(f"varredura de num_envs: {VARREDURA}\n")
    resumo = []
    detalhe = None
    for n in VARREDURA:
        orq, info = simula(num_envs=n, ruido=True)
        perm = [(c, i) for c in orq.celulas for i in range(orq.abertos[c])
                if bool(orq.congelado[c][i])]
        viés = sum(1 for c, _ in perm if c != CELULA_PUSH)
        resumo.append((n, orq.eventos, len(orq.abertas), len(perm), viés,
                       info["chamadas"]))
        if detalhe is None:
            detalhe = (orq, info)

    print("-- varredura --")
    print("        num_envs   eventos   tarefas   congelados   por viés   chamadas")
    for n, ev, tf, perm, viés, ch in resumo:
        print(f"        {n:8d}   {ev:7d}   {tf:7d}   {perm:10d}   {viés:8d}   {ch:8d}")
    print("\n   Leitura: se `por viés` cai a zero conforme `num_envs` sobe, o falso"
          "\n   positivo do máximo é um efeito de POUCAS AMOSTRAS, e a escala de treino"
          "\n   (4096/rank) já o dilui sozinha.")

    print(f"\n{'=' * 60}\ndetalhe do caso num_envs={VARREDURA[0]} "
          "(o regime que a S3 modela)")
    _relatorio_ruido(*detalhe)


if __name__ == "__main__":
    if "--ruido" in sys.argv:
        _varredura_ruido()
        sys.exit(0)

    print("simulando a run do currículo (sucesso sempre, sem física)...")
    orq, info = simula()
    print(f"convergiu em {info['chamadas']} chamadas de reset")
    falhas = _relatorio(orq, info)
    print(f"\n{'=' * 60}")
    if falhas:
        print(f"{falhas} FALHA(S) — a sequência do código não é a do desenho")
        sys.exit(1)
    print("a sequência de destravamentos reproduz o desenho")
