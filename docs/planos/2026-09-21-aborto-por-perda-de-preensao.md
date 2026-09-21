# Aborto por perda de preensão

Anotado em 2026-09-21 como **trabalho futuro**, a pedido do dono. Nada implementado.
Vale junto com o segundo estágio de `2026-09-21-botar-referencia-de-pe.md`, que entra
quando o REORIENTAR sair de inerte.

## O problema

Quando a caixa escapa, o robô mergulha atrás dela. No robô real isso não pode acontecer:
ele cai ou bate nas coisas em volta.

O mergulho não é ruído de exploração. Ele é a resposta ótima ao que a tabela paga: o elo
não muda quando a caixa cai, o estado continua PEGAR_COM ou CARREGAR, e nessas colunas o
`precise_pos` vale 1,0 (`knobs.py:1236`) contra um alvo ancorado no peito. Caixa caindo é
caixa se afastando do alvo, e seguir a caixa para baixo é o único jeito de recuperar o
termo.

E o robô nunca viu o depois: o `caixa_largada` termina o episódio quando a caixa chega
perto do chão. Tudo que a política sabe desse caso vem dos ~0,4 s de queda.

## A divisão de responsabilidade (decisão do dono, 2026-09-21)

O laço de recuperação **não é treinado**. Ele é plano, e plano vive no controlador
externo:

    pega -> deixa cair -> aborta -> reposiciona -> pega de novo -> repete

Cada passo já é um elo que existe. O contrato disso já está escrito: o one-hot que o ator
lê é o elo PUBLICADO, "o que o operador manda" (`observacoes.py:um_de_cinco`).

| passo | o controlador comanda |
|---|---|
| pega | PEGAR |
| aborta | ANDAR, twist zero |
| reposiciona | ANDAR, twist para a caixa |
| pega de novo | PEGAR |

---

## Parte 1 — a habilidade que falta

**Escopo:** o robô recebe ANDAR com comando zero em QUALQUER instante e para de pé, sem
cair e sem seguir a caixa. Uma habilidade, não um plano. Ela vale para todo aborto, não
só para a caixa caída.

Hoje essa transição nunca é treinada, porque o `caixa_largada` termina antes. A política
sabe ANDAR parado — é 30% do treino — mas sempre a partir de um corpo calmo, nunca a
partir de um movimento de manipulação comprometido.

**A rota:** o `caixa_largada` deixa de terminar e passa a trocar o estado. Duas coisas
acontecem sozinhas quando o elo publicado vira ANDAR:

1. o one-hot de 5 slots muda, e o ator o lê todo passo;
2. os 10 canais de caixa vão a **zero exato** pelo gate `cmd[ELO] != ANDAR`
   (`observacoes.py:caixa_no_frame_da_base`).

Ou seja, no estado de aborto **a caixa some da entrada do ator**. Não há atrás de que
mergulhar. O gatilho pode ser privilegiado: o ambiente já é (terminações,
`forca_de_apoio`, `limpo_massa`, laje mocap).

⚠⚠ **O RISCO É A PRECIFICAÇÃO, e este projeto já apanhou dela.** Ver
`~/.claude/memory/g1-limpo-piso-da-estatua-travava-a-pega.md`: ficar parado rendia 145
contra 102 de tentar, e a pega travou. Hoje derrubar custa o episódio inteiro — é a multa
implícita da terminação. Se o aborto cair num estado que ainda paga, derrubar vira lucro.

Duas colunas descartadas de saída:

| coluna | por que não |
|---|---|
| CAUDA | paga `postura_ereta` ×8 e `pose` ×8; está precificada para um fim BEM-SUCEDIDO |
| ANDAR | mantém `track_linear_velocity` e `track_angular_velocity` em 1,0, e rastrear comando zero parado colhe renda |

**A forma defendida:** uma décima primeira coluna POBRE — todo termo positivo em zero,
penalidades vivas, `fell_over` ainda terminando. O gradiente vira "minimizar custo", e o
mínimo de `self_collisions`, `contato_*`, `action_rate_l2` e `joint_acc` é ficar parado de
pé sem encostar em nada. Nenhum termo novo, nenhuma cabeça nova: uma coluna nas onze
tuplas da `PesoPorEstado`.

⚠ **Efeito de segunda ordem a MEDIR, não estimar.** O `caixa_largada` dispara em ~0,29
por episódio. Tirá-lo da lista de terminações faz esses episódios rodarem até o
`time_out`, o que move `dur_manip`, que move o `resolve_sorteio`, que move a divisão
locomoção × manipulação.

---

## Parte 2 — o sinal para o controlador autônomo

**Visão está descartada** (decisão do dono, 2026-09-21). Ela falha exatamente na hora do
aborto: a caixa fica ocluída pelos próprios braços durante a manipulação, sai do quadro ao
cair abaixo da câmera, e o mergulho borra a imagem. O robô sente mais do que vê.

**O G1 real não tem célula de carga na palma.** Os sensores `palma_E`/`palma_D` e o
`palmas_em_contato` são canal do CRÍTICO, por decisão de transferência. O ator vê o molde
do fabricante, o one-hot de 5 elos e os 10 canais de caixa, e nada mais
(`observacoes.py`). A preensão é inferida do torque nas juntas.

**Ponto técnico que fecha o desenho:** o sinal é de UM QUADRO SÓ. Sob controle PD a carga
aparece como diferença entre o alvo comandado e a posição atingida, e ela some no instante
em que a caixa sai. Não precisa de memória, o que importa porque a política é um MLP sem
recorrência — o mesmo motivo pelo qual o canal `GIRO` existe.

### As duas restrições de desenho

1. **A cabeça tem de ser SUPERVISIONADA** contra o rótulo privilegiado —
   `min(F_esq, F_dir)` acima de um limiar, que o `squeeze` já calcula
   (`recompensas._forca_das_palmas`). Uma saída livre que dispara a troca de estado é um
   botão de fuga: o PPO o aperta sempre que o aborto pagar mais que brigar com a tarefa.
   Supervisionada, ela fica presa ao fato.
2. **A borda de descida não pode ser lida sozinha.** No BOTAR bem-sucedido a preensão cai
   de 1 para 0 de propósito, porque soltar é a tarefa. O gatilho é
   `grasp caiu ∧ (elo ≠ BOTAR ∨ ~perto)`. Isso fica na máquina de estados externa, e é
   bom que fique: é regra de plano.

⚠ A supervisão fecha o hack da cabeça, mas NÃO fecha o da precificação: com o aborto
barato, a política ainda tem incentivo a tornar o rótulo verdadeiro, isto é, a derrubar
a caixa de verdade. A Parte 1 continua sendo pré-requisito.

### O baseline a medir primeiro

Se o sinal é de um quadro só, uma regra sobre a deflexão das juntas do braço talvez já
separe preensão de não-preensão. Custa zero de treino e roda no G1 real com os mesmos
dados.

A cabeça ganha o lugar dela se a separação depender da massa e da pose — e vai depender:
o `carga_max` vai a 5 kg e a `caixa_meia_aresta_faixa` vai de 0,07 a 0,13 m
(`knobs.py:30`), então um limiar fixo precisaria de calibração por configuração. Aprender
é o jeito de não calibrar.

**O que impede medir hoje:** os CSV do `registra_juntas` trazem `q_*` das 29 juntas, mas
não o alvo comandado nem o torque. Sem uma das duas colunas não dá para separar "o robô
mandou diferente" de "a junta cedeu sob carga". Acrescentar `tau_*` ao gravador é mudança
no INSTRUMENTO, não no treino, e com ela o `ckp9100_m2_t055_i2_carregar.csv` contra um
trecho de ANDAR responde em uma conta.

### Onde a perda auxiliar caberia

O `PPOPorElo` já é subclasse de `PPO` neste repo (`algoritmo.py`) e já sobrescreve
`compute_returns`. Há onde pôr uma segunda perda sem tocar no framework. A saída em si
custa uma dimensão a mais no ator, que o gerenciador de ações não mapeia para junta
nenhuma — e a perda tem de mirar a MÉDIA, não a amostra ruidosa.

---

## Ordem

| | o quê | por quê |
|---|---|---|
| 1 | `caixa_largada` troca o estado para a coluna pobre em vez de terminar | o comportamento passa a existir; a caixa some da entrada do ator |
| 2 | medir o efeito em `dur_manip` e na divisão loco × manipulação | tirar uma terminação de 0,29 move a estatística de episódio |
| 3 | `tau_*` no `registra_juntas` e o teste de separabilidade por regra | decide se a cabeça é necessária |
| 4 | cabeça supervisionada de preensão | o estimador de campo, sem célula de carga |

Uma cabeça perfeita que joga o robô num estado onde ele não sabe o que fazer não melhora
nada. O comportamento vem antes do estimador.
