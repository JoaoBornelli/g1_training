"""Item 4 da Parte 0 (plano B5; S §11, S §16 Parte 0 item 4): balanceadores em mesa.

O `Curriculo` recebe curvas sintéticas de sucesso por iteração (logísticas, uma por célula, com meio e
inclinação diferentes) e uma célula presa em 0,3. Critérios:
  1. nenhuma fatia trava a outra: toda fatia aberta recebe ≥ piso_tarefa em toda iteração;
  2. com todas as fatias no portão, as frações vão a 1/3 ± 0,02.

Fase 1: a célula presa fica em 0,3 até as três fatias abrirem. Fase 2: a presa volta à curva normal.
Sem a volta, a fatia dela nunca passa no portão e o critério 2 não tem como ocorrer.

Roda com: python -m g1_v3.parte0.item4_balanceador   (grava saida/item4_balanceador.txt)
"""
import math
import os

from g1_v3.balanceador import FATIA_DE, LAYOUT, Curriculo
from g1_v3.knobs import ATIVO

PMAX = 0.95  # teto da logística: abaixo de 1,0, para o portão (0,75) ser alcançável e não trivial
TRIALS = 100  # envs por célula aberta e por iteração; n_sucesso = round(TRIALS · p), sem sorteio
PRESA = LAYOUT.index(("CARREGAR", "arco"))  # escolha do script: o plano não diz qual célula fica presa
P_PRESA = 0.3  # S §16 item 4: "uma presa"
TABELA_CADA = 50  # iterações entre linhas da tabela
CAP_ITERS = 3000  # teto de cada fase; estourar é FALHA
TOL = 1e-9
SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saida", "item4_balanceador.txt")


def curva(i):
    # meio e inclinação por célula (ordem do LAYOUT): meio 10 + 15·i; inclinação 0,06 a 0,14
    return 10 + 15 * i, 0.06 + 0.01 * ((7 * i) % 9)


def sucesso(it, i, presa_ativa):
    if presa_ativa and i == PRESA:
        return P_PRESA
    meio, inclinacao = curva(i)
    return PMAX / (1 + math.exp(-inclinacao * (it - meio)))


def passo(cur, presa_ativa):
    # só as células abertas medem, como no env; o fecho aplica a EMA e as aberturas e subidas
    for i, c in enumerate(cur.celulas):
        if c.aberta:
            cur.registra(i, TRIALS, round(TRIALS * sucesso(cur.it, i, presa_ativa)))
    cur.fecha_iteracao()


def fatias_abertas(cur):
    return [any(c.aberta and FATIA_DE[c.tarefa] == f for c in cur.celulas) for f in range(3)]


def linha(cur, fr):
    niv = [cur.nivel(t) for t in ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")]
    deg = [cur.degrau(t) for t in ("ANDAR", "PEGAR", "CARREGAR", "BOTAR")]
    ab = sum(c.aberta for c in cur.celulas)
    return (f"{cur.it:>5}  {fr[0]:>7.3f} {fr[1]:>7.3f} {fr[2]:>7.3f}  "
            f"{niv[0]:>2} {niv[1]:>2} {niv[2]:>2} {niv[3]:>2}  "
            f"{deg[0]:>2} {deg[1]:>2} {deg[2]:>2} {deg[3]:>2}  {ab:>3}/{len(cur.celulas)}")


def main():
    cur = Curriculo(ATIVO.balanceador, ATIVO.niveis, ATIVO.comando, ATIVO.terreno)
    piso = ATIVO.balanceador.piso_tarefa
    tabela = []  # linhas da tabela, a cada TABELA_CADA iterações e no fim de cada fase
    min_fatia, it_min, viol, n_it = 1.0, None, 0, 0

    def mede(fim_de_fase=False):
        nonlocal min_fatia, it_min, viol, n_it
        fr = cur.fracoes_fatias().tolist()
        n_it += 1
        for f, aberta in enumerate(fatias_abertas(cur)):
            if aberta and fr[f] < piso - TOL:
                viol += 1
            if aberta and fr[f] < min_fatia:
                min_fatia, it_min = fr[f], cur.it
        if cur.it % TABELA_CADA == 0 or fim_de_fase:
            tabela.append(linha(cur, fr))
        return fr

    # fase 1: a célula presa fica em 0,3 até as três fatias abrirem
    it_tres = None
    while cur.it < CAP_ITERS:
        mede()
        if all(fatias_abertas(cur)):
            it_tres = cur.it
            break
        passo(cur, presa_ativa=True)
    if it_tres is not None:
        mede(fim_de_fase=True)

    # fase 2: a presa volta à curva; para no primeiro it com todas as células abertas no portão
    it_portao, fr_portao, desvio = None, None, None
    if it_tres is not None:
        ini = cur.it
        while cur.it - ini < CAP_ITERS:
            fr = mede()
            if all(c.s >= ATIVO.balanceador.portao for c in cur.celulas if c.aberta):
                it_portao, fr_portao = cur.it, fr
                desvio = max(abs(x - 1 / 3) for x in fr)
                break
            passo(cur, presa_ativa=False)
        mede(fim_de_fase=True)

    # critérios: OK/FALHA com o número
    crit = []
    ok1 = viol == 0 and it_min is not None
    crit.append(f"CRITERIO 1 (piso das fatias abertas >= {piso:.2f}): {'OK' if ok1 else 'FALHA'}: "
                f"menor fração = {min_fatia:.4f} em it={it_min}; violações = {viol} em {n_it} medidas "
                f"(fatia × iteração)")
    ok_ab = it_tres is not None
    crit.append(f"ABERTURA (três fatias abertas): {'OK' if ok_ab else 'FALHA'}: "
                f"todas abertas em it={it_tres}" if ok_ab else
                f"ABERTURA (três fatias abertas): FALHA: não abriram até it={cur.it}")
    if it_portao is None:
        crit.append("CRITERIO 2 (1/3 ± 0,02 com todas no portão): FALHA: nunca todas no portão "
                    f"até it={cur.it}")
    else:
        ok2 = desvio <= 0.02
        crit.append(f"CRITERIO 2 (1/3 ± 0,02 com todas no portão): {'OK' if ok2 else 'FALHA'}: "
                    f"em it={it_portao}, frações = ({fr_portao[0]:.4f}, {fr_portao[1]:.4f}, "
                    f"{fr_portao[2]:.4f}); desvio máximo = {desvio:.4f}")

    cab = (f"{'it':>5}  {'ANDAR':>7} {'B':>7} {'C':>7}  {'niv A':>2} {'P':>2} {'C':>2} {'B':>2}  "
           f"{'deg A':>2} {'P':>2} {'C':>2} {'B':>2}  {'cel':>6}")
    linhas = [
        "Item 4 (B5): balanceadores em mesa, Curriculo com curvas sintéticas",
        f"Curvas: logística, PMAX {PMAX}, meio 10 + 15·i, inclinação 0,06–0,14; "
        f"célula presa: {LAYOUT[PRESA]} em {P_PRESA} até as três fatias abrirem",
        f"Fase 1 (presa em {P_PRESA}): {it_tres} iterações até as três fatias abrirem",
        f"Fase 2 (presa volta à curva): {'-' if it_portao is None else it_portao - it_tres} iterações "
        f"até todas as células abertas passarem no portão ({ATIVO.balanceador.portao})",
        f"Iterações totais: {cur.it}",
        "",
        "Colunas: frações das fatias (ANDAR, B, C); níveis (ANDAR, PEGAR, CARREGAR, BOTAR); "
        "degraus do refino (mesma ordem); células abertas",
        cab,
        *tabela,
        "",
        *crit,
    ]
    texto = "\n".join(linhas) + "\n"
    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    with open(SAIDA, "w", encoding="utf-8") as fh:
        fh.write(texto)
    print(texto, end="")


if __name__ == "__main__":
    main()
