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
    info = {}
    for p in _notas():
        fm, _ = _partes(p.read_text())
        info[p.stem] = dict(p=p, pai=_pai(fm), tipo=_campo(fm, "tipo"), estado=_campo(fm, "estado"),
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
            for f in sorted(filhos[nome], key=lambda k: (info[k]["tipo"] != "frente", info[k]["tipo"] != "abordagem", k)):
                c = info[f]
                meta = " · ".join(x for x in (c["tipo"], c["estado"], c["periodo"]) if x)
                linhas.append(f"- **{f}** ({meta}) — {c['resumo'] or 'sem resumo'}")
            t2 = t2.rstrip() + f"\n\n## Filhos\n{INI}\n" + "\n".join(linhas) + f"\n{FIM}\n"
        else:
            t2 = t2.rstrip() + "\n"
        if t2 != t:
            i["p"].write_text(t2); n += 1
    print(f"blocos de filhos atualizados em {n} notas")


if __name__ == "__main__":
    semear() if "--semear" in sys.argv else indexa()
