"""Checa o vault: toda nota tem tipo, estado e um pai que existe."""
import re
import sys
from collections import Counter
from pathlib import Path

TIPOS = {"frente", "abordagem", "achado", "decisao", "pendencia", "estudo"}
ESTADOS = {"em-desenho", "ativa", "validada", "rejeitada", "parada",
           "superada", "aberta", "fechada"}
RAIZ = "G1"
IGNORA = {".obsidian", "_sistema"}


def _frontmatter(texto):
    m = re.match(r"---\n(.*?)\n---", texto, re.S)
    if not m:
        return {}
    return dict(re.findall(r"^(\w+):\s*(.*)$", m.group(1), re.M))


def _alvo(valor):
    m = re.search(r"\[\[([^\]|#]+)", valor or "")
    return m.group(1).strip() if m else None


def checa(raiz: Path) -> list[str]:
    notas = [p for p in raiz.rglob("*.md")
             if not IGNORA & set(p.relative_to(raiz).parts)
             and p.name not in {"CLAUDE.md", "inbox.md", "Home.md"}]
    nomes = Counter(p.stem for p in notas)
    erros = [f"nome duplicado: {n}" for n, c in nomes.items() if c > 1]
    for p in notas:
        fm, n = _frontmatter(p.read_text()), p.stem
        if fm.get("tipo") not in TIPOS:
            erros.append(f"{n}: tipo {fm.get('tipo')!r} inválido")
        if fm.get("estado") not in ESTADOS:
            erros.append(f"{n}: estado {fm.get('estado')!r} inválido")
        if fm.get("estado") == "superada":
            alvo = (fm.get("superada_por") or "").strip().strip('"[]')
            if not alvo:
                erros.append(f"{n}: superada sem superada_por")
            elif alvo not in nomes:
                erros.append(f"{n}: superada_por inexistente {alvo}")
        if not fm.get("resumo", "").strip().strip('"'):
            erros.append(f"{n}: sem resumo")
        if n == RAIZ and p.parent == raiz:
            continue
        pai = _alvo(fm.get("pai"))
        if not pai:
            erros.append(f"{n}: sem pai")
        elif pai not in nomes:
            erros.append(f"{n}: pai inexistente [[{pai}]]")
    return erros


if __name__ == "__main__":
    erros = checa(Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent))
    print("\n".join(erros) or "vault ok")
    sys.exit(1 if erros else 0)
