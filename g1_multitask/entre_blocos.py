"""Relatório entre um bloco de 2k-3k iterações e o próximo:

    python g1_multitask/entre_blocos.py                 # texto + gráfico
    python g1_multitask/entre_blocos.py --grafico       # SÓ o gráfico (8 painéis)
    python g1_multitask/entre_blocos.py logs/g1_residual/<run> --grafico

**No Kaggle, uma célula só:**

    !python g1_multitask/entre_blocos.py --grafico
    from IPython.display import Image; Image('painel_<run>.png')

O `--grafico` existe porque a saída em texto tem ~200 linhas, e entre blocos ninguém lê
200 linhas. Os 8 painéis respondem, em ordem: está vivo · aprende · o currículo andou ·
o que trava o push · de quem é o gargalo · **o sucesso é real** · alguma tarefa no vale ·
o movimento faz sentido.

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


# ============================================================ GRÁFICO (--grafico)
# Existe porque a saída em texto tem ~200 linhas e ninguém lê 200 linhas entre blocos.
# Cada painel responde UMA pergunta, e o título diz qual. A ordem é a ordem em que se
# olha: primeiro "está vivo", por último "quem domina a reward".
TAREFAS = ("parado", "andar", "pegar", "botar", "reorientar",
           "parado_caixa", "andar_caixa")
CORES = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
         "#9467bd", "#8c564b", "#17becf")


def acha(series: dict, *padroes: str) -> list[tuple[int, float]] | None:
    """Primeira série cujo tag casa um dos padrões, em ordem de preferência.

    Os nomes de tag do rsl_rl e do mjlab mudam entre versões (`Train/mean_reward`,
    `Train/mean_reward/time`, ...), então buscar por substring é mais robusto que
    hardcodar — e um painel sem dado avisa em vez de quebrar."""
    import re
    for pad in padroes:
        for tag in series:
            if re.search(pad, tag):
                return series[tag]
    return None


def _plot(ax, serie, **kw):
    if not serie:
        return False
    xs = [p[0] for p in serie]
    ys = [p[1] for p in serie]
    ax.plot(xs, ys, **kw)
    return True


def _vazio(ax, msg="sem dado neste log"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9,
            color="#888", transform=ax.transAxes)


def grafico(alvo: pathlib.Path, saida: pathlib.Path | None = None) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")               # sem display: Kaggle e headless
    import matplotlib.pyplot as plt

    dir_run = resolve(alvo)
    s = carrega(dir_run)
    fig, axes = plt.subplots(4, 2, figsize=(15, 17))
    fig.suptitle(f"{dir_run.name}   —   {len(s)} séries", fontsize=13, y=0.995)
    A = axes.flatten()

    # 1. ESTÁ VIVO? comprimento do episódio contra quem mata o episódio
    ax = A[0]
    ok = _plot(ax, acha(s, r"episode_length"), color="k", lw=2, label="passos/episódio")
    ax.set_ylabel("passos"); ax.set_title("1. Está vivo? (episódio e quem o mata)")
    ax2 = ax.twinx()
    for nome, cor in (("time_out", "#2ca02c"), ("fell_over", "#d62728"),
                      ("largou", "#ff7f0e"), ("fora_da_area", "#9467bd")):
        _plot(ax2, acha(s, rf"Termination/{nome}$"), color=cor, lw=1, alpha=.8,
              label=nome)
    ax2.set_ylabel("terminações")
    if not ok:
        _vazio(ax)
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

    # 2. ESTÁ APRENDENDO? reward e sucesso JUNTOS — reward caindo com sucesso subindo
    #    é DILUIÇÃO (tarefa nova abriu), não regressão. Foi exatamente a confusão de
    #    31/07 na iteração 1499: reward 18,75 -> 6,73 com 3 tarefas novas abrindo.
    ax = A[1]
    _plot(ax, acha(s, r"mean_reward$", r"mean_reward"), color="#1f77b4", lw=2,
          label="reward")
    ax.set_ylabel("reward", color="#1f77b4")
    ax.set_title("2. Aprende? (reward caindo + sucesso subindo = diluição)")
    ax2 = ax.twinx()
    _plot(ax2, acha(s, r"Metrics/sucesso"), color="#d62728", lw=2, label="sucesso")
    ax2.set_ylabel("sucesso", color="#d62728"); ax2.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)

    # 3. O CURRÍCULO ANDOU? eventos é a única medida de progresso REAL
    ax = A[2]
    for tag, cor, rot in ((r"orquestrador/eventos", "k", "eventos (de 60)"),
                          (r"tarefas_abertas", "#2ca02c", "tarefas abertas"),
                          (r"push_nivel", "#ff7f0e", "nível do push")):
        _plot(ax, acha(s, tag), color=cor, lw=2, drawstyle="steps-post", label=rot)
    ax.set_title("3. O currículo andou?"); ax.legend(fontsize=8)

    # 4. O QUE TRAVA O PUSH? o portão é o mínimo sobre os níveis abertos
    ax = A[3]
    algum = False
    for k in range(5):
        algum |= _plot(ax, acha(s, rf"parado_push/perf_n{k}$"), lw=1.6,
                       color=CORES[k], label=f"n{k}")
    _plot(ax, acha(s, r"parado_push/min$"), color="k", lw=2.4, label="min (o portão)")
    ax.axhline(0.90, color="r", ls="--", lw=1, label="portão 0,90")
    ax.set_ylim(-0.02, 1.05)
    # ⚠️ n0..n3 são HISTÓRICOS e ficam planos de propósito: o push é eixo GLOBAL com UM
    # nível corrente, então só o nível atual recebe medição (`curriculum.py:229`). Sem
    # esta nota o gráfico engana — quatro retas planas parecem bug.
    ax.set_title("4. O que trava o push? (n0-n3 são histórico; só o corrente mede)")
    if not algum:
        _vazio(ax)
    ax.legend(fontsize=8, ncol=3)

    # 5. DE QUEM É O GARGALO? o `min` de cada tarefa contra o portão
    ax = A[4]
    algum = False
    for t, cor in zip(TAREFAS, CORES):
        # ⚠️ casar `/{t}_` cru pegaria `parado_caixa_peso` dentro de `parado`. O nome
        # da célula é `<tarefa>_<eixo>`, então descasco pelo EIXO e comparo exato.
        alvos = []
        for tag in s:
            if not tag.endswith("/min"):
                continue
            cel = tag.split("/")[-2]
            for eixo in ("altura", "peso", "distancia", "heading", "giro", "push"):
                if cel.endswith("_" + eixo) and cel[: -len(eixo) - 1] == t:
                    alvos.append(tag)
                    break
        if not alvos:
            continue
        # a tarefa trava pelo PIOR eixo dela: mínimo sobre os eixos, por iteração
        passos = sorted({p[0] for tag in alvos for p in s[tag]})
        curva = []
        for x in passos:
            vs = [v for tag in alvos for (px, v) in s[tag] if px == x]
            if vs:
                curva.append((x, min(vs)))
        algum |= _plot(ax, curva, color=cor, lw=1.8, label=t)
    ax.axhline(0.90, color="r", ls="--", lw=1)
    ax.set_ylim(-0.02, 1.05)
    # `parado` não aparece: ele não tem eixo próprio, quem manda nele é o painel 4.
    ax.set_title("5. De quem é o gargalo? (min por tarefa; `parado` está no painel 4)")
    if not algum:
        _vazio(ax)
    ax.legend(fontsize=8, ncol=2)

    # 6. O SUCESSO É REAL? o tripwire de 31/07. Se a competência sobe e a condição
    #    FÍSICA da tarefa fica em zero, o crédito é falso — foi o caso do `pegar`
    #    marcando 0,98 com `grasp = 0`.
    ax = A[5]
    algum = False
    for t, cor in zip(TAREFAS, CORES):
        algum |= _plot(ax, acha(s, rf"contrib/{t}/cond_fisica$"), color=cor, lw=1.8,
                       label=f"{t} cond")
        algum |= _plot(ax, acha(s, rf"contrib/{t}/atribuicao_divergente$"), color=cor,
                       lw=1.2, ls=":", label=f"{t} DIVERG")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("6. O sucesso é real? (cheia=condição física · pontilhada=crédito falso)")
    if not algum:
        _vazio(ax, "sem `cond_fisica` — log anterior ao conserto de 31/07")
    ax.legend(fontsize=7, ncol=2)

    # 7. ALGUMA TAREFA ESTÁ NO VALE? total negativo => morrer cedo rende mais
    ax = A[6]
    algum = False
    for t, cor in zip(TAREFAS, CORES):
        algum |= _plot(ax, acha(s, rf"contrib/{t}/_total$"), color=cor, lw=1.8, label=t)
    ax.axhline(0.0, color="k", lw=1)
    ax.set_title("7. Alguma tarefa no vale? (total < 0 = morrer cedo rende)")
    ax.set_xlabel("iteração")
    if not algum:
        _vazio(ax)
    ax.legend(fontsize=8, ncol=2)

    # 8. O MOVIMENTO FAZ SENTIDO? `taxa_alvo` é o alvo COMPOSTO (BFM + residual), que
    #    o `action_rate_l2` não vê. É o número do braço sacudindo.
    ax = A[7]
    ok = _plot(ax, acha(s, r"Metrics/taxa_alvo"), color="#8c564b", lw=2,
               label="taxa do alvo composto")
    _plot(ax, acha(s, r"Metrics/deriva_parado"), color="#1f77b4", lw=1.5,
          label="deriva do parado (m)")
    ax.set_title("8. O movimento faz sentido? (tremor e deriva)")
    ax.set_xlabel("iteração")
    ax2 = ax.twinx()
    _plot(ax2, acha(s, r"mean_noise_std", r"action_std", r"noise_std"), color="#7f7f7f",
          lw=1, ls="--", label="std da ação")
    ax2.set_ylabel("std", color="#7f7f7f")
    if not ok:
        _vazio(ax)
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

    for a in A:
        a.grid(alpha=.25)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.2, w_pad=3.0)
    saida = saida or (pathlib.Path.cwd() / f"painel_{dir_run.name}.png")
    fig.savefig(saida, dpi=110)
    plt.close(fig)
    print(f"\n[GRÁFICO] {saida}")
    print("  no Kaggle, mostre inline com:")
    print(f"  from IPython.display import Image; Image('{saida}')")
    return saida


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    so_grafico = "--grafico" in sys.argv
    if args:
        alvo = pathlib.Path(args[0])
    else:
        raiz = pathlib.Path(__file__).resolve().parent.parent / "logs"
        alvo = ultima_run(raiz) if raiz.exists() else None
        if alvo is None:
            print("nenhuma run encontrada em ./logs — passe o diretório como argumento")
            sys.exit(1)
    if so_grafico:
        grafico(alvo)
    else:
        relatorio(alvo)
        grafico(alvo)      # o texto continua, o gráfico vem de brinde
