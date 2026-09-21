# Cuidado com a caixa: o impacto e o aperto desnecessário

Anotado em 2026-09-21 como **correção futura**, a pedido do dono. Nada disto está
implementado. As duas entradas saem de medição no código e nos CSV do `registra_juntas`,
e nenhuma delas foi tocada pelas mudanças de 20 e 21 de setembro.

O contexto é o requisito que o dono declarou nesta data: o robô manipula caixas com
conteúdo, e o movimento tem de ser controlado para não danificar o conteúdo nem causar
acidente. A razão principal de desacelerar é **segurança de pessoas**, e ela já foi
atendida pelo peso do `velocidade_por_regime`. As duas entradas abaixo são a outra metade:
o cuidado com a carga.

---

## 1. O impacto na caixa é medido e não é cobrado

### O que existe

`metricas.impacto_da_caixa` publica o **pico** de `|F_apoio_z| / (m·g)` no episódio, por
env. Caixa apoiada em repouso lê 1,0. Ela tem `reduce="max"` e tem `reset`.

Ela **não tem peso**. É só sentinela. O docstring registra a decisão do dono de 03/09:
soltar a caixa de uns 5 cm é permitido, e "se começar a jogar de mais alto vira problema".

### A medição

A métrica é **plana em toda a linhagem**, e não podia ser diferente: nada a otimiza.

| run | iteração | `impacto_da_caixa` |
|---|---|---|
| bloco22 | 21 510 | 6,27 |
| zero02 | 3 451 | 8,66 |
| zero02 | 5 477 | 6,79 |

Vinte e cinco mil iterações não a moveram. O robô entrega a caixa ao apoio com **6 a 9
vezes o peso dela**.

A cinemática da caixa confirma, medida por diferenciação dos CSV:

| checkpoint | fase | caixa \|v\| p95 | máxima | aceleração máxima |
|---|---|---|---|---|
| `model_24999` | pegar | 0,49 m/s | 1,58 | 2,5 g |
| `model_24999` | carregar | 0,48 | 0,51 | 1,0 g |
| `model_24999` | botar | 0,31 | 0,91 | 1,0 g |
| `model_5477` | carregar | 0,92 | 3,30 | 6,5 g |

A maturidade consertou o carregar, de 6,5 g para 1,0 g. Ela **não** consertou a pega, que
segue arrancando a caixa a 1,58 m/s.

### A correção proposta

Promover a métrica a termo de recompensa, com peso negativo e dobradiça:

    excesso = relu(pico / m·g − limiar)
    custo   = excesso²

com `limiar` na ordem de 1,5, ou seja o robô pode entregar a caixa com meia vez o peso
dela de sobra, e paga o quadrado do que passar disso.

Três cuidados que a medição já indica:

- **A dobradiça, e não a reta.** Abaixo do limiar o custo é zero, para não taxar o apoio
  normal. É o mesmo idioma do `limite_de_junta` e do `velocidade_por_regime`.
- **O pico, e não a média.** A métrica já publica o pico corrente; a média de um pico
  monótono é um piso, defeito que a própria métrica já corrigiu com `reduce="max"`.
- **O termo lê o PICO ACUMULADO**, então ele cobra uma vez e continua cobrando até o fim
  do episódio. Isso pode ser o certo (o dano já ocorreu) ou virar imposto fixo sem
  derivada. Decidir antes de escrever: cobrar o pico acumulado, ou o excesso instantâneo.

---

## 2. O `squeeze` paga por esmagar

### O que existe

`recompensas.squeeze` devolve `tanh(min(F_esq, F_dir) / F_ref)`, com peso **+1,0**. Ele
**recompensa** força de palma. A referência é derivada, e não escolhida:

    F_ref = m·g / (2·μ)

que é a força de aperto que o atrito precisa para segurar a caixa. Com a caixa de 1 kg e
μ = 0,8, `F_ref` = 6,13 N.

### A medição

O `tanh` satura, e depois da saturação apertar mais **não paga mais e não custa nada**:

| aperto | caixa de 1 kg | ganho marginal |
|---|---|---|
| 6,1 N (= `F_ref`) | 0,762 | 0,420 |
| 12,3 N | 0,964 | 0,071 |
| 30,7 N | 0,9999 | 0,0002 |
| 61,3 N | 1,0000 | 0,0000 |
| 69,0 N | 1,0000 | 0,0000 |

O medido é **69 N onde 8,5 N bastariam** (ver
`~/.claude/memory/g1-limpo-caixa-esmagada-squeeze-cego.md`). A partir de uns 30 N o robô
está numa planície: a preensão está garantida e o custo de apertar mais é exatamente
zero. Esmagar é a estratégia mais segura contra deixar cair, e o desenho atual a torna
gratuita.

⚠ E o problema **cresce com o nível**: `carga_max` vai a 5 kg, onde `F_ref` = 30,7 N e a
saturação só chega perto de 150 N.

### A correção proposta

Manter o `tanh` como incentivo, que é o que fez a preensão nascer, e acrescentar a
dobradiça do excesso acima de um múltiplo de `F_ref`:

    custo = relu(F / (k · F_ref) − 1)²

com `k` na ordem de 2 a 3, ou seja o robô pode apertar duas a três vezes o necessário e
paga o quadrado do que passar. O peso entra negativo, num termo separado ou como segunda
parcela do mesmo termo.

⚠ A escolha entre termo novo e segunda parcela não é livre: o `CLAUDE.md` do repo manda
consertar a forma de um termo que já existe antes de acrescentar outro. A segunda parcela
dentro do `squeeze` é a forma preferida, e ela tem a vantagem de o sinal chegar pela mesma
coluna da tabela por estado.

---

## Ordem sugerida

O impacto primeiro. Ele mede o dano diretamente e está plano há vinte e cinco mil
iterações, o que prova que nenhum outro termo o cobre por procuração. O aperto depois,
porque sem o teto do aperto o robô pode responder ao preço do impacto apertando mais para
não deixar cair.

## O que NÃO entra aqui

A velocidade da mão e o regime de limite foram resolvidos em 21/09 e estão em
`Obsidian-documents/Robotics/G1-Limpo-Status.md`. Este plano é só a carga.
