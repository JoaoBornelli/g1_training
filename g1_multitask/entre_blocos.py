"""Relatório entre um bloco de 2k-3k iterações e o próximo:

    python g1_multitask/entre_blocos.py                 # pega a run mais recente
    python g1_multitask/entre_blocos.py logs/g1_lifting_box/2026-07-31_multitask

Este arquivo é a razão pela qual a `observability.py` existe. O log default do mjlab
(`Episode_Reward/<termo>`) é média sobre TODOS os envs, e com 7 tarefas intercaladas um
termo gateado aparece com ~1/7 da magnitude real — olhando só ele não há como decidir
nada. Aqui os números vêm separados por tarefa.

Cinco seções, na ordem em que se olha:

  1. PROGRESSO   — destravamentos, tarefas abertas, nível de cada eixo
  2. SUCESSO     — taxa por CÉLULA (tarefa × eixo × nível): é a régua do
                   currículo, e a célula de `min` mais baixo é o que trava
  3. CONTRIBUIÇÃO— peso × valor por tarefa × termo, ordenado
  4. ALARMES     — estagnação, células congeladas, platô
  5. DECISÃO     — a tabela A/B/C: o que pode mudar entre blocos e o que não pode

A seção 5 não é enfeite. Sem ela ao lado dos números, a tentação é mudar algo de
Categoria C (largura da obs, definição de sucesso) sem perceber que isso joga o
checkpoint no lixo.
"""
from __future__ import annotations

import pathlib
import sys

from tensorboard.backend.event_processing import event_accumulator

LARG = 78


def ultima_run(raiz: pathlib.Path) -> pathlib.Path | None:
    cands = [p for p in raiz.rglob("events.out.tfevents.*")]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime).parent


def resolve(alvo: pathlib.Path) -> pathlib.Path:
    """Aceita tanto o diretório do evento quanto qualquer pai dele.

    O `EventAccumulator` exige o diretório que CONTÉM o `events.out.tfevents.*`, e o
    mjlab põe o arquivo num subdiretório do log_dir — apontar pro log_dir daria
    "Could not find directory" sem dizer o motivo."""
    if not alvo.exists():
        raise SystemExit(f"não existe: {alvo}")
    if any(alvo.glob("events.out.tfevents.*")):
        return alvo
    achado = ultima_run(alvo)
    if achado is None:
        raise SystemExit(f"nenhum events.out.tfevents.* em {alvo} nem abaixo dele")
    return achado


def carrega(dir_run: pathlib.Path) -> dict[str, list[tuple[int, float]]]:
    ea = event_accumulator.EventAccumulator(
        str(dir_run), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    return {tag: [(e.step, e.value) for e in ea.Scalars(tag)]
            for tag in ea.Tags()["scalars"]}


def _ultimo(series, tag, default=None):
    if tag not in series or not series[tag]:
        return default
    return series[tag][-1][1]


def titulo(txt):
    print(f"\n{'=' * LARG}\n {txt}\n{'=' * LARG}")


def relatorio(alvo: pathlib.Path) -> None:
    dir_run = resolve(alvo)
    s = carrega(dir_run)
    print(f"run: {dir_run}")
    passos = max((v[-1][0] for v in s.values() if v), default=0)
    print(f"iterações no log: {passos}")

    # -------------------------------------------------------------- 1. progresso
    titulo("1. PROGRESSO DO CURRÍCULO")
    eventos = _ultimo(s, "Curriculum/orquestrador/eventos")
    abertas = _ultimo(s, "Curriculum/orquestrador/tarefas_abertas")
    push = _ultimo(s, "Curriculum/orquestrador/push_nivel")
    if eventos is None:
        print("  (nenhuma chave `Curriculum/orquestrador/*` no log — o currículo "
              "não rodou ou o nome do termo mudou)")
    else:
        print(f"  destravamentos: {eventos:.0f} de 60      "
              f"tarefas abertas: {abertas:.0f} de 7      push: nível {push:.0f} de 4")
        print("\n  níveis abertos por eixo:")
        for tag in sorted(t for t in s if t.endswith("/abertos")):
            nome = tag.replace("Curriculum/orquestrador/", "").replace("/abertos", "")
            m = _ultimo(s, tag.replace("/abertos", "/min"))
            print(f"      {nome:<28s} {_ultimo(s, tag):.0f} níveis   "
                  f"min={m:.3f}" if m is not None else f"      {nome}")

    # ---------------------------------------------------------------- 2. sucesso
    titulo("2. SUCESSO POR CÉLULA (tarefa × eixo × nível) — a régua do currículo")
    perf = {t: _ultimo(s, t) for t in s if "/perf_n" in t}
    if not perf:
        print("  (sem chaves de perf — ver seção 1)")
    else:
        por_celula: dict[str, list[tuple[int, float]]] = {}
        for tag, v in perf.items():
            base = tag.split("/perf_n")[0].replace("Curriculum/orquestrador/", "")
            por_celula.setdefault(base, []).append((int(tag.split("_n")[-1]), v))
        for base in sorted(por_celula):
            niveis = sorted(por_celula[base])
            linha = "  ".join(f"n{i}={v:.2f}" for i, v in niveis)
            pior = min(v for _, v in niveis)
            marca = "  <== abaixo de 0.90, é o que trava" if pior < 0.90 else ""
            print(f"      {base:<28s} {linha}{marca}")

    # ----------------------------------------------------------- 3. contribuição
    titulo("3. CONTRIBUIÇÃO POR TAREFA × TERMO")
    print("  `peso × valor médio`, e o `_total` é a soma. Um termo cuja |contribuição|\n"
          "  supera o total da tarefa está mandando mais que a tarefa.\n")
    contrib: dict[str, dict[str, float]] = {}
    for tag, v in s.items():
        if "/contrib/" not in tag.lower() and not tag.startswith("Curriculum/contrib/"):
            continue
        resto = tag.replace("Curriculum/contrib/", "")
        if "/" not in resto:
            continue
        tarefa, termo = resto.split("/", 1)
        contrib.setdefault(tarefa, {})[termo] = v[-1][1]
    if not contrib:
        print("  (sem chaves `Curriculum/contrib/*` — o relatório de contribuição só\n"
              "   emite depois de `min_amostras` passos acumulados)")
    for tarefa in sorted(contrib):
        termos = contrib[tarefa]
        total = termos.pop("_total", None)
        print(f"\n  {tarefa}   total={total:+.3f}" if total is not None
              else f"\n  {tarefa}")
        for termo, v in sorted(termos.items(), key=lambda kv: -abs(kv[1]))[:8]:
            flag = ("   <== supera o total" if total is not None
                    and abs(v) > abs(total) > 0 else "")
            print(f"      {termo:<28s} {v:+8.4f}{flag}")

    # ---------------------------------------------------------------- 4. alarmes
    titulo("4. ALARMES")
    achou = False
    if _ultimo(s, "Curriculum/orquestrador/ALARME_estagnacao"):
        print("  🔴 ESTAGNAÇÃO: passou o limite de transições sem destravamento.\n"
              "     Ver seção 2: a célula com min mais baixo é a que trava.")
        achou = True
    congeladas = {t: _ultimo(s, t) for t in s if t.endswith("/congeladas")}
    for tag, v in congeladas.items():
        if v and v > 0:
            nome = tag.replace("Curriculum/orquestrador/", "").replace("/congeladas", "")
            print(f"  🟡 CONGELADA: {nome} tem {v:.0f} nível(is) congelado(s) "
                  f"(queda > 0.10 do pico)")
            achou = True
    platôs = [t for t in s if "/plato_n" in t and _ultimo(s, t)]
    if platôs:
        print(f"  ⚪ PLATÔ (diagnóstico, não portão) em {len(platôs)} célula(s)")
        achou = True
    if not achou:
        print("  nenhum alarme")

    # ---------------------------------------------------------------- 5. decisão
    titulo("5. O QUE PODE MUDAR ANTES DO PRÓXIMO BLOCO")
    print("""
  Categoria A — GRÁTIS, retoma do checkpoint e segue
      peso de reward · ligar/desligar um gate · `ema_alpha` · acrescentar ruído a um
      termo · acrescentar DR de atrito da caixa
      Só é grátis porque o sucesso mora em `env.success_buf` como FATO FÍSICO: mexer
      em peso não move a régua e não invalida nenhuma EMA nem nenhum limiar.

  Categoria B — WARM-START, o ator reaprende a usar o canal novo
      enxertar canais novos na observação (6 tensores, §9b) · reinicializar só o crítico

  Categoria C — DO ZERO, o checkpoint vai pro lixo
      mudar a LARGURA da obs (hoje 151) · mudar o espaço de ação · mudar a DEFINIÇÃO
      de sucesso (`knobs.Tolerancia`, seção de sucesso)

  Em observação desde a calibração (T12), decidir com os números da seção 3:
      `box_shake`     — supera o sinal de tarefa nas 3 tarefas que carregam
      `table_contact` — idem em `parado c/ caixa` e `andar c/ caixa`
      `arm_vel`       — idem em `andar c/ caixa`; braço que carrega é estrutura
    Se com política treinada ainda superarem, é peso: Categoria A.
""")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        alvo = pathlib.Path(sys.argv[1])
    else:
        raiz = pathlib.Path(__file__).resolve().parent.parent / "logs"
        alvo = ultima_run(raiz) if raiz.exists() else None
        if alvo is None:
            print("nenhuma run encontrada em ./logs — passe o diretório como argumento")
            sys.exit(1)
    relatorio(alvo)
