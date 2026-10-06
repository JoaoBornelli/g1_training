# Avaliação da v3 — faz sentido rodar assim?

Data: 2026-10-02. Avalia o desenho de `specs/g1-v3-comportamento.md` contra o enunciado
(`specs/g1-limpo-comportamento.md`). Base: `vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md` e
`docs/relatorios/2026-10-01-inventario-v3.md` (os códigos [Cn], [Fn], [An] apontam itens dele).

## 1. Veredito

O desenho cobre os objetivos do enunciado: cada comportamento pedido tem um mecanismo (§10 do
rascunho), o produto não muda (um modelo, troca por one-hot, transições em cadeia) e a v3 tira
os 15 remendos de economia de renda. A parte de locomoção leva os mecanismos com marcha medida
[C1–C10], então ela deve formar como no bloco 1.

O risco está na parte nova, que nunca rodou neste repo: o potencial, os alvos macro e os
balanceadores. Nenhum deles tem medida própria; o resume do 21700 apoia o refino, mas rodou com a
recompensa antiga. Por isso a v3 deve passar por checagens baratas, sem GPU, antes do primeiro
treino longo (§4).

Um ponto já decidido não passa na medida: a terminação no batente desde o começo (§2).

## 2. Achado medido: a terminação no batente desde o começo mata o treino

Conta feita com o `cena.npz` (escala da ação, default) e o `init_std` 1,0 do molde: o alvo do PD
é `q_default + escala · a`, com `a ~ N(0, 1)` no começo do treino.

| junta | escala | do default ao batente | P(alvo além do batente) por passo |
|---|---|---|---|
| `ankle_roll` (cada) | 0,439 rad | 15,0° | 55,1% |
| `waist_roll`, `waist_pitch` | 0,439 rad | 29,8° | 23,6% |
| `ankle_pitch` (cada) | 0,439 rad | 29,2° | 14,4% |
| `hip_roll` (cada) | 0,351 rad | 30,0° | 6,8% |

- **Lendo o alvo comandado** (a condição que eu propus em 02/10): 92,7% dos passos têm alguma
  junta com alvo além do batente. Com std 1,0, nenhum episódio passa de 5 passos; com std 0,5,
  nenhum passa de 1 s; com std 0,3, 0,8% passam de 1 s. Mandar o alvo além do batente é o uso
  normal do PD para gerar torque. Esta leitura está REFUTADA.
- **Lendo o ângulo medido:** nos plays das políticas treinadas, o ângulo passa do batente em até
  6,2% dos passos (20500), e nos plays do 23250 em 4 de 7, a primeira vez no passo 78 (1,6 s),
  quase sempre no tornozelo, empurrado pelo chão [F18]. Um episódio do melhor checkpoint de hoje
  terminaria em 1,6 s. Do zero, com std 1,0, a taxa só pode ser maior.
- **O `hip_yaw` acima de 1 rad** pelo alvo comandado: 13,1% por passo com std 1,0 (as duas
  juntas); 0,1% dos episódios passam de 1 s. O [C33] foi validado num resume, com std baixo.

Histórico na linha v2 (as duas terminações viveram nos notebooks, e não no `env_cfg.py`):

| terminação | onde e quando | resultado |
|---|---|---|
| `NoBatente` nas três juntas da cintura, ângulo medido, `frac_max` 1,00, sem os tornozelos | notebook do ZERO, de 20/09 (`d4e5e4e`, zero02) a 22/09 | a zero03 (it 451) e a zero04 (it 650) morreram com ela dominante: todo episódio de manipulação acabava em ~7 passos, o PEGAR nunca treinou; saiu na zero05 (`9b7dfba`); a rampa ficou |
| `hip_yaw` acima de 1,0 rad | notebook de RESUME, fora do git, 18/09 | o `s_C` caiu de 0,73 a 0,02 e voltou a 0,6 em ~400 its; ela seguia disparando em ~1% dos fins de episódio |

A terminação por junta funciona num resume, com a política já treinada, e mata o treino do
zero. É a lógica do refino: ela entra depois do aprender. A `zero01`, só com a rampa, andou e
pegou.

Opções:

| opção | o que faz | custo |
|---|---|---|
| A | rampa desde o começo; a terminação no batente entra no refino | o robô explora perto do batente no começo, pagando a rampa |
| B | terminação pelo ângulo medido, com margem além do batente do MJCF (por exemplo 5°) e sem os tornozelos | a terminação só pega o abuso grosso; o tornozelo fica só na rampa |
| C | como está (desde o começo) | a medida acima diz que o treino morre no passo 1 |

Recomendação: A, que é também a primeira intuição do dono ("ela pode matar a exploração
inicial"). O `hip_yaw` acima de 1 rad pelo ângulo medido pode ficar desde o começo, mas vale a
mesma conta antes de fechar.

## 3. Riscos do desenho, por mecanismo

| # | mecanismo | risco | o que fazer |
|---|---|---|---|
| R1 | batente e `hip_yaw` | §2 | decidir entre A e B |
| R2 | potencial | com `Φ(s') − Φ(s)` e desconto γ 0,99, uma oscilação de amplitude Δ rende Δ(1 − γ) por ciclo: pairar com ruído não paga exatamente zero. A forma de Ng (`γΦ(s') − Φ(s)`) zera os ciclos, mas cobra (1 − γ)Φ por passo parado, e com Φ acumulado (alvos alcançados + proximidade) segurar o P4 por 0,5 s custaria perto de um alvo inteiro | na spec: forma `γΦ(s') − Φ(s)` com Φ limitado a 1 por alvo, e um bônus fixo por alvo alcançado |
| R3 | potencial e terminação | "o ganho já pago fica": esticar-se até uma caixa longe ganha proximidade e depois cai | custo fixo da queda maior que o maior ganho de um mergulho; a faixa do tronco zera a proximidade fora dela; conferir com números |
| R4 | balanceador de tarefas | a regra "a fatia seguinte abre quando TODAS as subtarefas abertas passam" acopla as fatias: um giro no lugar preso em 0,6 impede a C de abrir | abrir a fatia seguinte pelo núcleo da anterior (ANDAR: reto e parado; B: PEGAR na laje alta), e não por todas as subtarefas |
| R5 | refino | o gatilho "tudo no portão" trava se um nível nunca chegar a 0,75 (a laje de 0,04 m com 5 kg) | refino por fatia, ou o gatilho conta só os níveis abertos há mais de N its |
| R6 | alvos macro | o P0 (palmas à frente da caixa) é novo e não tem pose de IK; as tolerâncias de cada alvo não têm número | o P0 sai da geometria da caixa; a spec fixa as tolerâncias |
| R7 | observação | a v3 ainda não define os canais do alvo macro corrente nem do nível | a spec define, por append no fim [C16] |
| R8 | gerador de IK | o treino não pode resolver IK por env a cada reset (segundos por pose) | o alvo sai de interpolação da tabela (`ref_v3_*.npz`) por altura, tamanho e massa |
| R9 | CARREGAR | frear com a caixa vem só da generalização do ANDAR (decisão do dono) | conferir no play: CARREGAR a 0,5 m/s seguido de CARREGAR parado |
| R10 | impacto | na fase aprender a caixa pode cair de até 20 cm (1,98 m/s) sem terminar; soltar de 10–20 cm pode virar hábito antes do refino | acompanhar a velocidade de chegada no log; o refino aperta |
| R11 | pacote novo | é código novo inteiro; erro de implementação parece erro de desenho | smoke do dono com checagem por mecanismo, como no `g1_limpo` |
| R12 | orçamento | fase 0 (~1000 its), abertura dos níveis, 5 degraus de refino com ≥ 100 its cada, dentro de 30 000 its [F34] | sem medida; a primeira corrida diz |
| R13 | pose de referência no refino (05/10) | pagar por passo a proximidade da pose da IK reabre a renda por estado numa arquitetura que paga por tarefa completa: estátua na pose no PEGAR e no BOTAR, que não têm relógio; alvo preferido; pose contra tarefa fora da tabela; o contador do refino não volta e prende o degrau que quebrou; renda inobservável pelo crítico | discutir a forma antes da implementação (spec §11.5): pagar a pose só no avanço do alvo, ou como condição de avanço; contador do refino que volta um degrau |

## 4. O que medir primeiro

Antes de escrever o pacote, sem GPU:

1. **Batente** — feito (§2). Falta a decisão entre A e B.
2. **Potencial em trajetória roteirizada:** reproduzir, só com cinemática, a interpolação entre
   as poses de IK de P1 a P4 e de B1 a B4, e calcular Φ passo a passo. Esperado: Φ sobe sem
   descer no caminho certo; pairar rende zero; tombar a caixa ou soltar uma mão derruba Φ.
3. **Derivada no ponto de partida:** a proximidade de cada alvo, no estado em que o alvo
   começa, tem derivada não nula (σ = distância inicial [C20]).
4. **Escada de custos:** queda > perda da caixa > ganho da volta à pose de pé; queda > ganho de
   um mergulho até a caixa longe; com os pesos que a spec propuser.
5. **Balanceadores em simulação de mesa:** curvas de sucesso sintéticas, uma delas presa;
   conferir que nenhuma fatia trava a abertura das outras (R4) e que as frações convergem a 1/3.

Depois do smoke do dono, na primeira corrida:

6. Primeiras 200 its: terminações por causa, std por junta, `razao_marcha`.
7. Fase 0 em ~1000 its: `razao_marcha` ≥ 0,50 [C1].
8. No play: espera sem agachar [A21]; tombo da caixa na subida [A20]; frear com a caixa (R9).

## 5. Pontos abertos que a spec decide

O gatilho do refino; a medida de sucesso do ANDAR (proposta: razão projetada [P3, A11]); o
batente virtual das juntas de curso largo; o freio com o balanceador; a massa da cadeia C; e do
inventário, A7 (altura de transporte), A8 (faixa de comando), A9 (σ do `track_lin`), A13 (braços
abertos parado), A17 (teto da rampa de junta), A19 (massa só no crítico).
