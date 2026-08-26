"""Lê o log do TensorBoard e imprime as tabelas que decidem o bloco.

    python g1_limpo/leitura.py                 # o run mais recente sob ./logs
    python g1_limpo/leitura.py CAMINHO/run     # um run específico
    python g1_limpo/leitura.py --demo          # autoteste da aritmética, sem log

⚠ RODE COMO ARQUIVO, e não com `-m`. O `-m g1_limpo.leitura` importa o
`g1_limpo/__init__.py`, que registra a task e exige o `mjlab`. Este arquivo só precisa
do `tensorboard`, portanto ele lê um log baixado do Drive numa máquina sem GPU e sem
`mjlab` — que é a máquina onde este projeto é analisado.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO, e nem do próprio pacote.

POR QUE ELE EXISTE — A DILUIÇÃO. O `Episode_Reward/<termo>` do `rsl_rl` é a soma do
episódio dividida por `max_episode_length_s` (`reward_manager.py:107-110`), e NÃO pela
duração real. Com episódios de 2,05 s num teto de 20 s, todo valor sai multiplicado
por 0,1027:

    taxa por segundo = Episode_Reward × max_episode_length_s / (passos_médios × dt)

Ler o painel sem desfazer isso foi o que deixou dois freios consumirem 55% do sinal
positivo por 5000 iterações sem ninguém ver. Um `action_rate_l2 = −0,34` no painel é
um custo real de **−3,31/s**.

⚠ E o `Episode_Metrics/*` (o `MetricsManager`) NÃO tem essa diluição: ele divide por
`step_count` (`metrics_manager.py:113-126`). Métrica em [0,1] fica em [0,1]. Só a
recompensa precisa da correção — misturar as duas é o erro fácil aqui.
"""
from __future__ import annotations

import pathlib
import sys

# --- constantes do env. Se mudarem no `knobs.py`, mudam aqui. ---
DT = 0.02            # 50 Hz de controle (timestep 0,005 × decimation 4)
MAX_EP_S = 20.0      # `episode_length_s`
PASSOS_CHEIOS = MAX_EP_S / DT        # 1000

# =============================================================================
# AS CHAVES, e elas são um CONTRATO com o produtor.
#
# ⚠ Uma chave errada aqui NÃO levanta erro: a linha só não aparece, e o bloco segue
# sem o portão. O `smoke.py` confere estes nomes contra quem os produz.
#
# ⚠ CICATRIZ: a escada do `g1_poc` usa `Policy/mean_noise_std`, e essa chave NÃO
# EXISTE no `rsl_rl` 5.4.0 — o logger escreve `Policy/mean_std`
# (`rsl_rl/utils/logger.py:211`). Aquela linha da escada nunca disparou.
CH_STD = "Policy/mean_std"
CH_DURACAO = "Train/mean_episode_length"        # em PASSOS, não em segundos
CH_RECOMPENSA = "Train/mean_reward"
CH_RAZAO = "Metrics/twist/razao_marcha"
CH_VOO = "Episode_Metrics/tempo_de_voo"
CH_PICO = "Episode_Metrics/pico_de_altura"
CH_ESCORREGO = "Episode_Metrics/velocidade_de_escorrego"
CH_POUSO = "Episode_Metrics/forca_de_pouso"
CH_NIVEL = "Curriculum/nivel"

# =============================================================================
# A ESCADA DE CORTE DA F1. Pare o bloco se uma linha falhar.
#   (iteração, chave, comparador, alvo, o que significa falhar)
ESCADA = [
    (200, CH_STD, ">=", 0.85,
     "as penalidades dominam desde o começo; algum peso ficou grande"),

    # ⚠ SOBREVIVÊNCIA, e só isso. Um robô IMÓVEL marca 1000 passos. Foi por medir
    # "andar" com este número que o balanço de forma entregou a locomoção sem marcha
    # existir. Ele é condição NECESSÁRIA, nunca suficiente.
    (1000, CH_DURACAO, ">=", 150.0,
     "o robô não sobrevive ao episódio; nada abaixo disto se interpreta"),

    # ⚠ O PORTÃO DE VERDADE DA F1. Adimensional, e nasce em 0,0 para a estátua.
    (2000, CH_RAZAO, ">=", 0.50,
     "o robô NÃO ANDA: ele não fecha metade da velocidade comandada. "
     "PRÉ-REGISTRADO: mover `escala_acao_mult` para 0,8, e NADA MAIS no bloco"),

    # ⚠ O pé sai do chão? É a pergunta que o peso 0 do `air_time` apagava do painel,
    # e a razão de as métricas terem saído dos termos de recompensa.
    (1000, CH_VOO, ">", 0.0,
     "nenhum pé deixa o chão: não existe marcha, existe arrasto"),
]


def _acumulador(run: pathlib.Path):
    from tensorboard.backend.event_processing import event_accumulator

    arquivos = sorted(run.rglob("events.out.tfevents*"))
    if not arquivos:
        raise SystemExit(f"nenhum arquivo de evento em {run}")
    acc = event_accumulator.EventAccumulator(
        str(arquivos[-1]), size_guidance={event_accumulator.SCALARS: 0})
    acc.Reload()
    return acc


def _serie(acc, chave: str) -> list[tuple[int, float]]:
    if chave not in acc.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in acc.Scalars(chave)]


def _em(serie, it: int) -> float | None:
    """O último valor com passo <= `it`. `None` se a run não chegou lá."""
    validos = [v for s, v in serie if s <= it]
    return validos[-1] if validos else None


def por_segundo(valor_do_painel: float, passos_medios: float) -> float:
    """Desfaz a diluição do `Episode_Reward/*`.

    ⚠ `passos_medios` vem em PASSOS (`Train/mean_episode_length`), e não em segundos.
    Confundir os dois erra por 50×.
    """
    if passos_medios <= 0:
        return float("nan")
    return valor_do_painel * PASSOS_CHEIOS / passos_medios


def _tabela_de_recompensa(acc, it: int, passos: float) -> None:
    chaves = [t for t in acc.Tags().get("scalars", [])
              if t.startswith("Episode_Reward/")]
    if not chaves:
        print("  (nenhum Episode_Reward/* no log)")
        return
    linhas = []
    for ch in chaves:
        v = _em(_serie(acc, ch), it)
        if v is None:
            continue
        linhas.append((por_segundo(v, passos), v, ch.split("/", 1)[1]))
    linhas.sort(key=lambda x: x[0], reverse=True)

    pos = sum(t for t, _, _ in linhas if t > 0)
    neg = sum(t for t, _, _ in linhas if t < 0)
    print(f"  {'termo':<26}{'painel':>10}{'POR SEGUNDO':>14}{'% do lado':>11}")
    print("  " + "-" * 61)
    for taxa, painel, nome in linhas:
        lado = pos if taxa > 0 else neg
        frac = f"{abs(taxa) / abs(lado) * 100:.1f}%" if lado else "—"
        print(f"  {nome:<26}{painel:>10.4f}{taxa:>14.3f}{frac:>11}")
    print("  " + "-" * 61)
    print(f"  {'SOMA positiva':<26}{'':>10}{pos:>14.3f}")
    print(f"  {'SOMA negativa':<26}{'':>10}{neg:>14.3f}")
    print(f"  {'LÍQUIDO':<26}{'':>10}{pos + neg:>14.3f}")
    if pos and abs(neg) > 0.5 * pos:
        print(f"  ⚠ os freios consomem {abs(neg) / pos * 100:.0f}% do sinal positivo")


def _tabela_de_marcha(acc, it: int) -> None:
    for rot, ch, un in (
        ("razão de marcha", CH_RAZAO, ""),
        ("tempo de voo", CH_VOO, " s"),
        ("pico do pé", CH_PICO, " m"),
        ("escorrego do pé", CH_ESCORREGO, " m/s"),
        ("força de pouso", CH_POUSO, " N"),
        ("duração", CH_DURACAO, " passos"),
        ("nível", CH_NIVEL, ""),
    ):
        v = _em(_serie(acc, ch), it)
        print(f"  {rot:<20}{'ausente' if v is None else f'{v:.4f}{un}'}")


def _escada(acc) -> int:
    falhas = 0
    print(f"  {'it':>6}  {'chave':<34}{'medido':>10}{'alvo':>10}  veredito")
    print("  " + "-" * 78)
    for it, ch, comp, alvo, porque in ESCADA:
        serie = _serie(acc, ch)
        ultimo = max((s for s, _ in serie), default=-1)
        v = _em(serie, it)
        if ultimo < it:
            estado = "ainda não" if serie else "CHAVE AUSENTE"
            print(f"  {it:>6}  {ch:<34}{'—':>10}{alvo:>10.2f}  {estado}")
            if not serie:
                falhas += 1
                print(f"          ⚠ {ch} não existe no log — portão CEGO")
            continue
        ok = (v >= alvo) if comp == ">=" else (v > alvo)
        print(f"  {it:>6}  {ch:<34}{v:>10.3f}{alvo:>10.2f}  "
              f"{'ok' if ok else 'FALHOU'}")
        if not ok:
            falhas += 1
            print(f"          ✗ {porque}")
    return falhas


def _demo() -> int:
    """Autoteste da aritmética, sem log e sem GPU.

    Os números são os do defeito que este arquivo existe para não repetir: episódios
    de 2,05 s (102,5 passos) num teto de 20 s.
    """
    print("--- demo: a aritmética da diluição")
    passos = 102.5
    fator = PASSOS_CHEIOS / passos
    erros = 0

    def confere(nome, medido, esperado, tol=1e-9):
        nonlocal erros
        ok = abs(medido - esperado) < tol
        print(f"  {'ok ' if ok else 'FALHOU'} {nome}: {medido:.4f} "
              f"(esperado {esperado:.4f})")
        if not ok:
            erros += 1

    confere("o fator de diluição de 2,05 s", fator, 1000.0 / 102.5)
    confere("action_rate_l2 de −0,34 no painel", por_segundo(-0.34, passos),
            -0.34 * fator)
    confere("um episódio CHEIO não sofre correção",
            por_segundo(1.0, PASSOS_CHEIOS), 1.0)
    confere("a correção é LINEAR nos passos",
            por_segundo(1.0, 250.0) / por_segundo(1.0, 500.0), 2.0)
    print(f"\n  ⚠ −0,34 no painel é −{abs(-0.34 * fator):.2f}/s de verdade.")
    return erros


def main(argv: list[str]) -> int:
    if "--demo" in argv:
        return _demo()

    alvo = [a for a in argv if not a.startswith("-")]
    if alvo:
        run = pathlib.Path(alvo[0])
    else:
        raiz = pathlib.Path("logs")
        candidatos = sorted((p for p in raiz.rglob("*") if p.is_dir()),
                            key=lambda p: p.stat().st_mtime)
        if not candidatos:
            raise SystemExit("nenhum run sob ./logs — passe o caminho, "
                             "ou rode --demo")
        run = candidatos[-1]

    acc = _acumulador(run)
    passos_serie = _serie(acc, CH_DURACAO)
    it = max((s for s, _ in passos_serie), default=0)
    passos = _em(passos_serie, it) or PASSOS_CHEIOS

    print("=" * 80)
    print(f"g1_limpo — {run}")
    print(f"iteração {it}  ·  {passos:.1f} passos médios "
          f"({passos * DT:.2f} s de {MAX_EP_S:.0f} s)")
    print("=" * 80)

    print("\n--- recompensa, DES-DILUÍDA (por segundo de episódio vivo)")
    _tabela_de_recompensa(acc, it, passos)
    print("\n--- marcha")
    _tabela_de_marcha(acc, it)
    print("\n--- escada de corte da F1")
    falhas = _escada(acc)
    print("\n" + "=" * 80)
    print("a escada passa" if not falhas else f"{falhas} linha(s) da escada FALHARAM")
    return falhas


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0)
