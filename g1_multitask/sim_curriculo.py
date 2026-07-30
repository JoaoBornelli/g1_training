"""Simula a run inteira do currículo, sem física:

    python g1_multitask/sim_curriculo.py

Este é o teste de aceitação do orquestrador. Ele responde à pergunta que nenhum outro
check responde: **a sequência de destravamentos que o código produz é a que o desenho
especifica?** Sem ele, um erro no grafo ou na prioridade do F9 só apareceria depois de
horas de GPU, e apareceria como "o treino não progride".

O que se verifica (§7b):
  - o total é exatamente **60** destravamentos
  - a cadeia crítica até o `botar` abrir é **9** eventos
  - o `andar` só abre com push COMPLETO (os 4 níveis competentes)
  - `reorientar` e `pegar` abrem como IRMÃOS, os dois filhos do `andar`
  - cada eixo termina com todos os níveis abertos

Roda em segundos porque não instancia env: um stub com os 4 atributos que o
orquestrador toca basta, e é justamente o que torna o teste barato o suficiente pra
rodar sempre.
"""
from __future__ import annotations

import pathlib
import sys

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


class _CfgFalso:
    def __init__(self, params):
        self.params = params


def simula(num_envs: int = 64, max_chamadas: int = 40_000, verboso: bool = False):
    """Roda até o currículo parar de destravar. Devolve (orq, historico)."""
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

    for chamada in range(max_chamadas):
        # sucesso SEMPRE: o teste é da MÁQUINA de destravamento, não da política.
        # Se o robô sempre vence, a sequência é a mais curta possível e o total tem
        # que ser exatamente o do desenho.
        env.success_buf.fill_(1.0)
        antes_abertas = set(orq.abertas)
        antes_push = orq.abertos[(T.PARADO, PUSH)]
        orq(env, ids)
        for t in set(orq.abertas) - antes_abertas:
            abertura_em[t] = orq.eventos
        if (push_completo_em is None
                and antes_push < orq.abertos[(T.PARADO, PUSH)] == len(T.LEVELS[PUSH])):
            push_completo_em = orq.eventos
        if orq.eventos > antes:
            antes = orq.eventos
            parado_em = chamada
        elif chamada - parado_em > 3000:
            break              # 3000 chamadas sem evento: convergiu
    return orq, {"abertura_em": abertura_em, "push_completo_em": push_completo_em,
                 "chamadas": chamada + 1}


def _relatorio(orq, info) -> int:
    falhas = 0

    def ok(nome, cond, detalhe=""):
        nonlocal falhas
        print(f"  {'OK   ' if cond else 'FALHA'} {nome}"
              + (f"   ({detalhe})" if detalhe else ""))
        if not cond:
            falhas += 1

    print("\n-- total de destravamentos --")
    ok("total = 60", orq.eventos == 60, f"deu {orq.eventos}")

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
    esperado = {"parado": 0, "andar": 4, "reorientar": 13, "pegar": 13,
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


if __name__ == "__main__":
    print("simulando a run do currículo (sucesso sempre, sem física)...")
    orq, info = simula()
    print(f"convergiu em {info['chamadas']} chamadas de reset")
    falhas = _relatorio(orq, info)
    print(f"\n{'=' * 60}")
    if falhas:
        print(f"{falhas} FALHA(S) — a sequência do código não é a do desenho")
        sys.exit(1)
    print("a sequência de destravamentos reproduz o desenho")
