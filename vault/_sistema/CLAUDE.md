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
- Só o `pai` (frontmatter) cria aresta. Abordagens feitas em sequência, cada uma para consertar a anterior, ficam em SÉRIE: a seguinte é filha da anterior (Currículos → C1 → C2 → C3 → C4); o corpo abre com "Deriva de X: motivo". `deriva_de` fica para ligar uma abordagem a outra fora da série. No corpo, cite outra nota pelo nome em texto, sem `[[link]]`.
- Seção nova = pasta nova em `G1/` + uma cor nova em `.obsidian/graph.json`.

## Fonte das notas

- A memória do Claude NÃO é fonte para o vault: o teto de 150 palavras corta o estudo.
- Escreva a partir das fontes primárias: código e docstrings, corpos de commit, `.npz`/logs com números, specs, relatórios e o relato do usuário.
- Cada experimento vira uma nota filha (Hipótese, Método com arquivos, Resultado com números, Por que parou, O que herda). O relato do usuário prevalece sobre a memória.

## Resumo e lista de filhos

- Toda nota tem `resumo:` no frontmatter: uma linha com o que foi testado e o veredito. O `checa.py` acusa a falta.
- A lista `## Filhos` de cada nota pai é gerada por `python vault/_sistema/indexa.py`, entre os marcadores `filhos:inicio`/`filhos:fim`. Não edite a lista à mão; edite o `resumo` do filho e rode o script.
- Para percorrer o vault: comece em `G1.md`, leia a lista de filhos e desça só pelos ramos que interessam.
- Ordem depois de criar ou mudar notas: `indexa.py`, depois `checa.py`.
- A property `cluster` (pela pasta) é escrita pelo `indexa.py` e é o que o grafo usa para pintar (`[cluster:c3-bloco]` em `.obsidian/graph.json`). Cluster novo: acrescente em `CLUSTERS` no `indexa.py` e um grupo de cor no `graph.json`.
