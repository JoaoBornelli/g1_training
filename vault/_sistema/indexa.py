"""Escreve em cada nota pai a lista estatica dos filhos, para um agente percorrer o vault.

    python vault/_sistema/indexa.py            # atualiza os blocos de filhos
    python vault/_sistema/indexa.py --semear   # preenche `resumo:` que falta, a partir do corpo

O bloco fica entre os marcadores abaixo, sob o titulo `## Filhos`, e e REESCRITO a cada
execucao: nao edite a lista a mao, edite o `resumo:` do filho.
"""
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
INI, FIM = "<!-- filhos:inicio -->", "<!-- filhos:fim -->"
QUERY = re.compile(r"```dataview\n.*?```\n?", re.S)
# property `cluster` pela pasta: o grafo pinta por ela. Do mais especifico ao mais geral.
CLUSTERS = [
    ("G1/Abordagens/PPO/Currículos/C3 g1_limpo/Linha bloco", "c3-bloco"),
    ("G1/Abordagens/PPO/Currículos/C3 g1_limpo/Linha zero", "c3-zero"),
    ("G1/Abordagens/PPO/Currículos/C3 g1_limpo/Refino v2", "c3-refino"),
    ("G1/Abordagens/PPO/Currículos/C3 g1_limpo", "c3"),
    ("G1/Abordagens/PPO/Currículos/C4 v3", "c4"),
    ("G1/Abordagens/PPO/Currículos/C2 g1_poc", "c2"),
    ("G1/Abordagens/PPO/Currículos/C1 multitask", "c1"),
    ("G1/Abordagens/PPO/Tarefas isoladas", "tarefas-isoladas"),
    ("G1/Abordagens/PPO", "ppo"),
    ("G1/Abordagens/BFM-Zero", "bfm"),
    ("G1/Abordagens/Controle determinístico", "controle"),
    ("G1/Abordagens", "abordagens"),
    ("G1/Percepção", "percepcao"),
    ("G1/Cena de depósito", "cena"),
    ("G1/Transversais", "transversais"),
]
SECAO_RESUMO = ("Resultado", "Fato", "Decisão", "Objetivo", "Veredito da frente", "Veredito")


def _notas():
    return [p for p in RAIZ.rglob("*.md")
            if not {".obsidian", "_sistema"} & set(p.relative_to(RAIZ).parts)]


def _partes(texto):
    m = re.match(r"---\n(.*?)\n---\n", texto, re.S)
    return (m.group(1), texto[m.end():]) if m else ("", texto)


def _campo(fm, nome):
    m = re.search(rf"^{nome}:\s*(.*)$", fm, re.M)
    return m.group(1).strip().strip('"') if m else ""


def _pai(fm):
    m = re.search(r"\[\[([^\]|#]+)", _campo(fm, "pai"))
    return m.group(1).strip() if m else None


def _primeira_frase(txt):
    txt = re.sub(r"[*`|]", "", txt).strip()
    txt = re.sub(r"^\s*[-#>].*?(?=\w)", "", txt)
    frase = re.split(r"(?<=[.!?])\s", " ".join(txt.split()), maxsplit=1)[0]
    return frase[:220].rstrip(" ,;:") if frase else ""


def _resumo_do_corpo(corpo):
    for s in SECAO_RESUMO:
        m = re.search(rf"^## {re.escape(s)}\n(.*?)(?=^## |\Z)", corpo, re.S | re.M)
        if m and _primeira_frase(m.group(1)):
            return _primeira_frase(m.group(1))
    corpo = corpo.split("\n---\n", 1)[-1]  # notas originais: corpo depois do separador
    for par in re.split(r"\n\s*\n", corpo):
        if par.strip() and not par.lstrip().startswith(("#", "```", "<!--")):
            return _primeira_frase(par)
    return ""


def _cluster(p):
    rel = p.relative_to(RAIZ).with_suffix("").as_posix()
    for pre, nome in CLUSTERS:
        if rel == pre or rel.startswith(pre + "/") or rel + ".md" == pre + ".md":
            return nome
    return "raiz"


def clusteriza():
    """Escreve `cluster:` no frontmatter de cada nota, pela pasta. Nota nova recebe sozinha."""
    n = 0
    for p in _notas():
        t = p.read_text()
        fm, corpo = _partes(t)
        if not fm:
            continue
        c = _cluster(p)
        fm2 = re.sub(r"^cluster:.*$", f"cluster: {c}", fm, flags=re.M) if re.search(r"^cluster:", fm, re.M) else fm + f"\ncluster: {c}"
        if fm2 != fm:
            p.write_text(f"---\n{fm2}\n---\n{corpo}"); n += 1
    return n


def semear():
    n = 0
    for p in _notas():
        t = p.read_text()
        fm, corpo = _partes(t)
        if fm and not _campo(fm, "resumo"):
            r = _resumo_do_corpo(corpo).replace('"', "'")
            if r:
                p.write_text(t.replace("\n---\n", f'\nresumo: "{r}"\n---\n', 1)); n += 1
    print(f"resumo semeado em {n} notas")


def indexa():
    print(f"cluster escrito em {clusteriza()} notas")
    info = {}
    for p in _notas():
        fm, corpo = _partes(p.read_text())
        info[p.stem] = dict(p=p, pai=_pai(fm), serie=corpo.lstrip().startswith("> **Deriva de"), tipo=_campo(fm, "tipo"), estado=_campo(fm, "estado"),
                            periodo=_campo(fm, "periodo"), resumo=_campo(fm, "resumo"))
    filhos = {}
    for nome, i in info.items():
        if i["pai"]:
            filhos.setdefault(i["pai"], []).append(nome)
    n = 0
    for nome, i in info.items():
        t = i["p"].read_text()
        t2 = QUERY.sub("", t)
        t2 = re.sub(rf"\n*## Filhos\n+{re.escape(INI)}.*?{re.escape(FIM)}\n?", "\n", t2, flags=re.S)
        t2 = re.sub(r"\n*## Filhos\s*\Z", "\n", t2)
        if nome in filhos:
            linhas = []
            for f in sorted(filhos[nome], key=lambda k: (not info[k]["serie"], info[k]["tipo"] != "frente", info[k]["tipo"] != "abordagem", k)):
                c = info[f]
                meta = " · ".join(x for x in (c["tipo"], c["estado"], c["periodo"]) if x)
                marca = "PRÓXIMA NA SÉRIE: " if c["serie"] else ""
                linhas.append(f"- {marca}**{f}** ({meta}) — {c['resumo'] or 'sem resumo'}")
            t2 = t2.rstrip() + f"\n\n## Filhos\n{INI}\n" + "\n".join(linhas) + f"\n{FIM}\n"
        else:
            t2 = t2.rstrip() + "\n"
        if t2 != t:
            i["p"].write_text(t2); n += 1
    print(f"blocos de filhos atualizados em {n} notas")


if __name__ == "__main__":
    semear() if "--semear" in sys.argv else indexa()
