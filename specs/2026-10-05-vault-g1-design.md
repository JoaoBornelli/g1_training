# Vault G1 — grafo de conhecimento do projeto

Data: 2026-10-05 · Estado: aprovado em conversa, aguarda revisão da spec

## Problema

Os mesmos problemas voltam. Anotações feitas durante as discussões se perdem.
O conhecimento do G1 está espalhado em `~/.claude/memory` (~140 arquivos `g1-*`,
`mjlab-*`, `bfm-*`) e em `docs/memoria/`, sem uma estrutura que mostre o que foi
testado dentro de quê.

## Objetivo

Um vault Obsidian do projeto G1 inteiro, organizado como uma árvore de abordagens,
que:

1. mostra o que foi testado dentro de cada frente e o estado de cada tentativa;
2. cresce com ordem: toda nota tem um pai, nada fica solto;
3. marca o conteúdo obsoleto sem apagá-lo;
4. mostra o grafo visual dos nós e dos links;
5. o Claude consulta antes de propor e alimenta quando algo surge.

## Decisões

- **Lugar:** `g1_training/vault/`, versionado no git do repo.
- **Substitui `docs/memoria/`.** A memória do Claude guarda só ponteiros:
  uma linha com o fato e o caminho no vault.
- **Seção nova** = nó filho da raiz `G1`. A estrutura não muda.

## Modelo de nota

Toda nota tem frontmatter:

```yaml
tipo: frente | abordagem | achado | decisao | pendencia | estudo
pai: "[[nó pai]]"          # obrigatório, exceto na raiz G1
estado: em-desenho | ativa | validada | rejeitada | parada | superada | aberta | fechada
verificado: AAAA-MM-DD
superada_por: "[[nota]]"   # só quando estado = superada
```

- `frente` e `abordagem` formam a árvore. A pasta repete a árvore.
- `abordagem` tem as seções: **Hipótese**, **Método**, **Resultado**,
  **Por que parou**, **O que herda**.
- `achado`, `decisao`, `pendencia` e `estudo` são notas pequenas, penduradas
  numa abordagem ou numa frente pelo `pai`.
- Links transversais (herança entre currículos, achado reaproveitado) ficam no
  corpo da nota.
- `Transversais/` guarda o que várias frentes usam: princípios de recompensa,
  infra (mjlab, Warp, Kaggle/Colab) e o enunciado do comportamento.

## Árvore semente

```
G1
├── BFM-Zero
│   ├── Goal Reaching            (overdrive por alvo → rejeitado)
│   ├── Reward Optimization      (z é direção sem magnitude → parada)
│   ├── Few-shot (do paper)      (resultado a registrar)
│   └── Residual PPO sobre o BFM (pose → z → rejeitado)
├── PPO
│   ├── Tarefas isoladas
│   │   ├── Locomoção
│   │   └── Lift Box (g1_training/)
│   ├── Currículos
│   │   ├── C1 multitask (g1_multitask/)
│   │   ├── C2 g1_poc
│   │   ├── C3 g1_limpo (ativo; melhor model_23250)
│   │   └── C4 v3 (em-desenho)
│   └── Estudos
│       └── ResMimic (estudo, não treinado)
├── Controle determinístico
│   ├── IK + PD do braço
│   └── Autônomo fase 1
├── Percepção
│   ├── Visão da caixa
│   ├── Câmera D435i
│   └── Lidar simulado → SLAM
├── Cena de depósito
└── Transversais
    ├── Princípios de recompensa
    ├── Infra
    └── Enunciado do comportamento
```

## Crescimento e obsolescência

- `inbox.md` recebe a anotação solta na mesma resposta em que ela surge.
  No começo de cada sessão, o Claude esvazia a inbox e pendura cada item no pai.
- Sem pai na árvore, o Claude propõe um nó novo; não cria nota solta.
- Afirmação invalidada: `estado: superada` + `superada_por`. A nota fica.
- `Home.md` (Dataview): pendências abertas, abordagens ativas e em desenho,
  achados com `verificado` acima de 3 semanas.
- Páginas de frente usam Dataview para listar os filhos por estado. Ninguém
  edita essas listas à mão.

## Grafo

- Revisão 06/10: na raiz fica só `G1.md`; cada seção é uma pasta em `G1/` e forma um cluster (`G1/Abordagens/` é o macro de BFM-Zero, PPO e Controle). Sistema (Home, inbox, CLAUDE.md, checa.py, templates) fica em `_sistema/`, fora do grafo.
- Só o `pai` cria aresta. Relações transversais e `superada_por` são texto, sem `[[link]]`.
- Cor por cluster (pasta), não por estado. O estado aparece no Dataview.
- Texto original abaixo, substituído nos pontos acima:

- Graph View nativo, com cor por `estado` (`.obsidian/graph.json`).
- Notas `superada` ficam em cinza.
- ExcaliBrain opcional, para a vista hierárquica pai e filho.
- O usuário ativa o Dataview uma vez.

## Papel do Claude

Regras em `vault/CLAUDE.md`, citadas no `CLAUDE.md` do repo:

- Ler o nó da frente e os filhos antes de propor uma abordagem.
- Gravar na inbox ou no nó certo na mesma resposta.
- Seguir o modelo de nota e o tratamento de obsoleto acima.

## Migração

1. **Esqueleto:** árvore semente, templates, `inbox.md`, `Home.md`, config do
   grafo, `vault/CLAUDE.md`. Mover `docs/memoria/` para os nós e reescrever os
   links que apontam para `docs/memoria/`.
2. **Memória, por frente:** cada arquivo `g1-*`/`mjlab-*`/`bfm-*` vira nota do
   vault; a memória fica com uma linha e o caminho. O usuário revisa uma frente
   por vez.

## Fora do escopo

- Plugins além de Dataview e ExcaliBrain.
- Sincronizar com o vault Obsidian geral de projetos.
- Automação por hook. Revisitar se a inbox não for esvaziada na prática.
