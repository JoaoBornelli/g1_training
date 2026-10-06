---
tipo: achado
pai: "[[C1 multitask]]"
estado: ativa
verificado: 2026-10-06
resumo: "Decidido e implementado em 2026-08-06, branch exp/curriculo-teste (commits ee83202 + 0126410)."
---
# g1-multitask-orcamento-igual-por-tarefa (original)

Corpo original da memória `g1-multitask-orcamento-igual-por-tarefa`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Decidido e implementado em 2026-08-06, branch `exp/curriculo-teste` (commits `ee83202`
+ `0126410`).

## O problema

Depois do bloco 1 da auditoria, os tetos por tarefa iam de **2,50 (`botar`) a 7,00
(`parado c/ caixa`)** — 2,8×. Só de sinal de tarefa era **1,0 contra 5,5**. Duas
consequências, e as duas são de escala e não de forma:

1. **As penalidades não são gateadas.** `table_contact −1,5`, `com_balance −2,0`,
   `back_penalty −0,5`, `action_rate_l2 −0,25` cobram o mesmo de todas as tarefas. O
   `botar` enfrentava a mesma pilha de custo com um quinto do sinal.
2. **O rsl_rl normaliza a vantagem UMA vez sobre o rollout inteiro**
   (`ppo.py:186-188`), com as 7 tarefas no mesmo pote. Escala de recompensa menor sai
   com vantagem menor e é comprimida contra zero no gradiente — escala relativa entre
   tarefas **é** taxa de aprendizado relativa entre tarefas.

## A mudança estrutural

O `gated` de `rewards.py` deixou de aplicar máscara 0/1 e passou a aplicar um **vetor
de peso**: fora do escopo zero, dentro o fator da tarefa (`escala = {tarefa: fator}`).

Isto era necessário, não cosmético: um `RewardTermCfg` tem UM peso e os termos são
compartilhados — `box_at_peito` vale em 3 tarefas, `hold_still` em 3, os dois de
rastreio em 4. Baixar o peso de `box_at_peito` para conter o `parado c/ caixa`
baixaria junto o `pegar`. O grau de liberdade que faltava é **(termo × tarefa)**.

## O alvo é derivado, não escolhido

`orcamento_tarefa = 4.0` = `track_linear_velocity + track_angular_velocity` do
`unitree_g1_flat_env_cfg` (2,0 + 2,0). É o orçamento de tarefa que o fabricante dá ao
G1 na locomoção, e o único do desenho que não foi digitado por nós — os pesos do
bloco 2 saíram todos da §14. O resto do cfg do fabricante é `upright 1,0` + `pose
1,0`, que são piso postural.

Os 7 fatores são **calculados** de 9 pesos e 9 gates (`_equaliza_orcamento` em
`env.py`), nunca digitados. Digitados ficariam errados em silêncio no dia em que
alguém mudasse um peso — que é exatamente o que o bloco 1 fez ao tirar o `reaching`
do gate do `botar`.

Resultado: `parado 1,000 · andar 1,000 · pegar 0,800 · botar 4,000 · reorientar 2,000
· parado c/ caixa 0,727 · andar c/ caixa 0,727`. Tetos nominais 5,50 em todas menos
`pegar` (5,00 — ele não tem termo de postura, decisão de 03/08 porque a postura
cobrava de quem agacha).

**A escala é uniforme dentro da tarefa**, portanto só muda a relação ENTRE tarefas; a
proporção entre os termos de uma mesma tarefa fica intacta.

## Dois pontos de desenho fixados no mesmo dia

**`reaching` é PERMANENTE, não anela.** O item 22 do doc previa catraca até 0,01 e
três lugares do código descreviam isso como se estivesse ligado. Nunca foi
implementado (nada em `curriculum.py` muta peso de reward) e a decisão é não
implementar: é o único termo que diz "ponha as mãos na caixa", pré-requisito físico
do `pegar` e do `reorientar`. Fatia depois da equalização: 20% do orçamento no
`pegar`, 50% no `reorientar` — inalteradas, porque a escala é uniforme dentro da
tarefa.

**`botar_fracao_solta` 0,0 → 0,5** (commit `44cbf15`). Com 0,0 o termo era indiferente
entre segurar no destino e soltar no destino — medido 3,695 nos dois — e o critério
de sucesso exige `~preensao`. Argmax da recompensa num estado que o critério reprova,
mesma classe do C1/C2. `f = 1.0` estaria errado: o termo viraria `kernel × (1 −
preensão)`, zero durante todo o transporte. O transporte no `botar` já é medido pelo
`box_at_prateleira` (destino ← caixa, escala grossa 0,30 desde o bloco 1); não é
trabalho do `reaching`, que mede palma→caixa e no `botar` já nasce satisfeito.

## O que fica em aberto

- **Aceitação:** `_std_vantagem_por_tarefa` (`runner.py`) mede desvio de vantagem por
  tarefa — antes: 0,79 / 1,11 / 1,15. Se a dispersão não encolher, o teto nominal não
  era o proxy certo para escala efetiva, e o lever passa a ser a vantagem medida.
- **O anelamento com catraca do `reaching` está documentado mas NÃO implementado**
  (verificado 2026-08-06: nada em `curriculum.py` mexe em peso de reward). Se for
  implementado, muda o orçamento de `pegar` e `reorientar` em runtime — tem que
  recalcular a escala junto.
- Quebra a escala do crítico de checkpoints anteriores.

Ver g1-multitask-spec-validacao-2026-08, g1-multitask-curriculum-design,
g1-lift-progress-alvo-e-a-caixa.
