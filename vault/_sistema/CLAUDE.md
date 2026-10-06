# Vault G1 — regras para o Claude

Spec: `../specs/2026-10-05-vault-g1-design.md`.

## Modelo de nota

- Frontmatter: `tipo`, `pai`, `estado`, `verificado` (AAAA-MM-DD). Templates em `_sistema/_templates/`.
- `tipo`: frente, abordagem, achado, decisao, pendencia, estudo.
- `estado`: em-desenho, ativa, validada, rejeitada, parada, superada, aberta, fechada.
- `pai` é obrigatório (exceto `G1`). A nota fica na pasta do pai. Nome de arquivo único no vault.
- `abordagem` tem: Hipótese, Método, Resultado, Por que parou, O que herda, Filhos.
- Links transversais (herança, achado reaproveitado) vão no corpo.

## Operação

- Antes de propor uma abordagem, leia o nó da frente e os filhos dele. Verifique se já foi testada.
- Anotação solta vai para `_sistema/inbox.md` na mesma resposta em que surge.
- No começo da sessão, esvazie a inbox: pendure cada item no pai certo.
- Sem pai na árvore: proponha um nó novo ao usuário. Não crie nota solta.
- Afirmação invalidada: `estado: superada` + `superada_por: <nome da nota nova, sem link>`. Não apague a nota.
- Ao reverificar um achado, atualize `verificado`.
- Depois de criar ou mover nota, rode `python vault/_sistema/checa.py`. Ele precisa imprimir `vault ok`.
- A memória do Claude (`~/.claude/memory`) guarda só uma linha e o caminho no vault.

## Grafo

- Na raiz fica só `G1.md`. Cada seção é uma pasta em `G1/` e forma um cluster.
- Só o `pai` (frontmatter) cria aresta. No corpo, cite outra nota pelo nome em texto, sem `[[link]]`.
- Seção nova = pasta nova em `G1/` + uma cor nova em `.obsidian/graph.json`.
