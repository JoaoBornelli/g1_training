"""Item 6 da Parte 0 (plano B5; S §16 item 6, §19 conflito 5).

Rastreio em comando baixo. Para c = 0,05; 0,10; 0,25; 0,5; 1,0 (linear e giro):
- ganho por segundo de executar (rastreio normalizado = 1), contra a estátua (rastreio 0, passo 0)
  e contra a marcha no lugar (rastreio 0, passo 0);
- custo por segundo de começar e manter a marcha: custos_por_s do ANDAR andando menos o do ANDAR parado.

Critério: executar vence a estátua e a marcha no lugar em todo c.

Rodar da raiz do worktree:
    python -m g1_v3.parte0.item6_comando_baixo
"""
from pathlib import Path

import torch

from g1_v3 import potencial as P
from g1_v3.knobs import ATIVO, caminho
from g1_v3.parte0.comum import custos_por_s, le_csv

ANDAR_ANDANDO = ["5000_botar", "5350_botar"]  # fase `andar`, 300 passos, 0,65 m/s
ANDAR_PARADO = ["5000_andar_parado", "5350_andar_parado", "5800_andar_parado"]  # fase `parado`
COMANDOS = (0.05, 0.10, 0.25, 0.5, 1.0)
SAIDA = Path(__file__).resolve().parent / "saida" / "item6_comando_baixo.txt"


def _trecho(nome: str, fase: str) -> dict:
    """Só as linhas da `fase` (o play do ANDAR andando traz também PEGAR, BOTAR e espera)."""
    d = le_csv(caminho(nome + ".csv"))
    m = d["fase"] == fase
    return {k: v[m] for k, v in d.items()}


def _ganhos_por_s(c: float, linear: bool) -> tuple[float, float]:
    """(executar, parado) em valor por segundo: peso × rastreio normalizado.

    Executar: erro 0 e v = c, rastreio 1. Parado (estátua e marcha no lugar): erro = c e v = 0, rastreio 0.

    ⚠ Unidade (plano, cabeçalho da B5, 29/10): o mjlab soma w·x·dt por passo, então o retorno por
    segundo é w·média(x). O ganho por segundo é w·rastreio, sem dividir por DT (o mesmo que custos_por_s).
    """
    R, C = ATIVO.recompensa, ATIVO.comando
    peso = R.rastreio_linear if linear else R.rastreio_angular
    sigma_zero = R.sigma_zero_lin if linear else R.sigma_zero_ang
    t = lambda x: torch.tensor([x], dtype=torch.float64)  # noqa: E731
    r_exec = P.rastreio_normalizado(t(0.0), t(c), t(c * c), R.sigma_rastreio, sigma_zero,
                                    C.limiar_canal, R.piso_rastreio).item()
    r_parado = P.rastreio_normalizado(t(c * c), t(c), t(0.0), R.sigma_rastreio, sigma_zero,
                                      C.limiar_canal, R.piso_rastreio).item()
    return peso * r_exec, peso * r_parado


def _num(x: float) -> str:
    return f"{x:+.2f}".replace(".", ",")


def main() -> None:
    out = []
    emite = out.append

    # taxa_acao e energia saem NaN: os CSV não têm a_* nem alvo_pd_* (comum.custos_por_s)
    custo = {n: custos_por_s(_trecho(n, "andar")) for n in ANDAR_ANDANDO}
    custo |= {n: custos_por_s(_trecho(n, "parado")) for n in ANDAR_PARADO}
    media_andando = sum(custo[n]["total"] for n in ANDAR_ANDANDO) / len(ANDAR_ANDANDO)
    media_parado = sum(custo[n]["total"] for n in ANDAR_PARADO) / len(ANDAR_PARADO)
    # custo da marcha = parado − andando, positivo; é o número que entra no critério
    marcha_juntos = -(media_andando - media_parado)

    emite("Item 6 (S §16, §19 conflito 5): rastreio em comando baixo")
    emite("")
    emite("Custo por segundo de cada play (valor por s; negativo = custo; NaN = sem CSV)")
    emite(f"  {'arquivo':<20}{'fase':<8}{'taxa_acao':>10}{'vel_junta':>11}{'acel_junta':>12}"
          f"{'energia':>9}{'total':>10}")
    for nome in ANDAR_ANDANDO + ANDAR_PARADO:
        c = custo[nome]
        fase = "andar" if nome in ANDAR_ANDANDO else "parado"
        emite(f"  {nome:<20}{fase:<8}{c['taxa_acao']:>10.3f}{c['vel_junta']:>11.3f}"
              f"{c['acel_junta']:>12.3f}{c['energia']:>9.3f}{c['total']:>10.3f}")
    emite(f"  média andando ({len(ANDAR_ANDANDO)} arquivos): {media_andando:.3f}; "
          f"média parado ({len(ANDAR_PARADO)}): {media_parado:.3f}")
    emite(f"  custo da marcha por s, todos juntos: {marcha_juntos:.3f}")
    emite("  ⚠ taxa_acao e energia ficam fora do total: os CSV não têm a_* nem alvo_pd_*.")
    emite("")

    tudo_ok = True
    for c in COMANDOS:
        emite(f"c = {str(c).replace('.', ',')} (valor por s)")
        emite(f"  {'tipo':<8}{'escopo':<16}{'ganho':>9}{'custo':>9}{'exec−estátua':>14}"
              f"{'exec−marcha':>13}  veredito")
        for linear, tipo in ((True, "linear"), (False, "giro")):
            ganho_exec, ganho_parado = _ganhos_por_s(c, linear)
            escopos = [("todos juntos", marcha_juntos)]
            escopos += [(n, -(custo[n]["total"] - media_parado)) for n in ANDAR_ANDANDO]
            for escopo, marcha in escopos:
                # estátua: sem custo; marcha no lugar: paga a marcha; ambas sem ganho (rastreio 0)
                net_exec = ganho_exec - marcha
                net_estatua = ganho_parado
                net_marcha = ganho_parado - marcha
                vs_estatua = net_exec - net_estatua
                vs_marcha = net_exec - net_marcha
                ok = vs_estatua > 0 and vs_marcha > 0
                tudo_ok &= ok
                nota = "  (custo do linear: sem CSV de giro)" if not linear else ""
                emite(f"  {tipo:<8}{escopo:<16}{_num(ganho_exec):>9}{_num(-marcha):>9}"
                      f"{_num(vs_estatua):>14}{_num(vs_marcha):>13}  {'OK' if ok else 'FALHA'}{nota}")
        emite("")

    emite("Passo: não entra no ganho (conta 0). É ≥ 0, então o ganho real é maior e o critério fica")
    emite("a favor. A marcha no lugar paga o custo da marcha (não há CSV dela).")
    emite("Suposições (não definidas no plano; o número depende delas):")
    emite("  3. taxa_acao e energia são NaN (sem a_* e alvo_pd_*); o custo está incompleto. Para")
    emite(f"     virar o linear, esses termos teriam de somar mais de {_num(_ganhos_por_s(0.05, True)[0])[1:]} por s.")
    emite("  4. O custo do ANDAR andando é de um play a 0,65 m/s, usado igual para todo c.")
    emite("     Não há play em 0,05 a 0,25 m/s.")
    emite("  5. Não há CSV de giro: o giro usa o custo do linear.")
    emite("  6. Não há CSV de marcha no lugar: assumida com o custo da marcha e ganho 0.")
    emite("  7. Passo fora do ganho: a definição por segundo depende de cadência e de FK (item 8).")
    emite("  8. Por arquivo: cada andando é comparado com a média dos três parados (5800 não tem par).")
    emite(f"Resultado: {'todas as linhas OK' if tudo_ok else 'HÁ FALHA'}.")

    texto = "\n".join(out)
    print(texto)
    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
