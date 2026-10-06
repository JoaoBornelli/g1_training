from pathlib import Path
from checa import checa

def _nota(d, nome, fm):
    (d / f"{nome}.md").write_text("---\n" + fm + "\n---\n" + "palavra " * 50 + "\n")

def test_vault_valido(tmp_path):
    _nota(tmp_path, "G1", "tipo: frente\nestado: ativa\nverificado: 2026-10-06\nresumo: raiz")
    _nota(tmp_path, "PPO", 'tipo: frente\npai: "[[G1]]"\nestado: ativa\nverificado: 2026-10-06\nresumo: x')
    assert checa(tmp_path) == []

def test_erros(tmp_path):
    _nota(tmp_path, "G1", "tipo: frente\nestado: ativa\nverificado: 2026-10-06")
    _nota(tmp_path, "solta", "tipo: achado\nestado: ativa\nverificado: 2026-10-06")
    _nota(tmp_path, "orfa", 'tipo: achado\npai: "[[Nada]]"\nestado: ativa\nverificado: 2026-10-06')
    _nota(tmp_path, "velha", 'tipo: achado\npai: "[[G1]]"\nestado: superada\nverificado: 2026-10-06')
    _nota(tmp_path, "rara", 'tipo: ideia\npai: "[[G1]]"\nestado: ok\nverificado: 2026-10-06')
    (tmp_path / "sub").mkdir()
    _nota(tmp_path / "sub", "G1", 'tipo: frente\npai: "[[G1]]"\nestado: ativa\nverificado: 2026-10-06')
    (tmp_path / "rasa.md").write_text('---\ntipo: achado\npai: "[[G1]]"\nestado: ativa\nverificado: 2026-10-06\nresumo: x\n---\nduas linhas\n')
    erros = "\n".join(checa(tmp_path))
    for trecho in ["solta: sem pai", "orfa: pai inexistente", "velha: superada sem superada_por",
                   "rara: tipo", "rara: estado", "nome duplicado: G1", "solta: sem resumo", "rasa: sem conteudo"]:
        assert trecho in erros


if __name__ == "__main__":
    import tempfile
    for teste in (test_vault_valido, test_erros):
        with tempfile.TemporaryDirectory() as d:
            teste(Path(d))
    print("2 passed")
