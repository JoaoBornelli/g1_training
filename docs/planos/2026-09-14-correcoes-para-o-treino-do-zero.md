# Correções para o treino do zero — g1_limpo

Data: 2026-09-14
Estado: proposta, nada implementado além das mudanças A (ver §0)

Três comportamentos a corrigir, medidos no `model_14000`:

1. velocidade de movimentação exagerada na manipulação
2. o robô se apoia na caixa para botar
3. manipulação com as mãos tortas

Este documento traz o plano, a verificação de que ele fecha cada um, os buracos que
ele abre, e uma auditoria do que a arquitetura de pagar por segundo incentiva hoje.

---

## 0. O que já está aplicado

| # | mudança | onde |
|---|---|---|
| A1 | `altura_carregar` 1.02 → 0.85 | `knobs.py:200`, `comando.py:294` |
| A2 | `peito_b.z` 0.222 → 0.052 | `knobs.py:178`, `comando.py:291` |
| A3 | check do piso do alvo | `smoke.py` |
| A4 | `--roteiro carregar` | `registra_juntas.py` |
| A5 | `limpa_laje` no CARREGAR | `registra_juntas.py` |

A1 e A3 são provisórios: a mudança M7 troca o valor fixo por sorteio.

---

## 1. As sete mudanças

### M1 — `velocidade_por_regime`: `mean` → `amax`

`recompensas.py:831`

```python
return torch.amax(torch.relu(v.abs() / vmax - 1.0) ** 2, dim=1)
```

A média sobre 29 juntas divide o sinal das poucas que correm. Medido no
`juntas14k.csv`: p50 de **1 junta** acima do `vmax` por passo, p90 de 3, de 29.

A forma (dobradiça quadrática, zero abaixo do limite) já está certa e fica.

### M2 — `faixa_de_pose.perna` no BOTAR: 1.3 → 0

`knobs.py:1206`

Agachar até a laje exige `hip_pitch` 2.03 e `knee` 1.50. A faixa de 1.3 é mais
apertada que a geometria, portanto ela cobra o movimento que a tarefa exige.

Com `tol = 1.3` a faixa morde em três juntas só — `hip_pitch`, `hip_yaw` e `knee`
para cima. As outras nove não alcançam 1.3 nem no batente. Ou seja: ela cobra o
agachamento e quase nada mais.

A perna não fica sem guarda. O `dof_pos_limits` (peso −1.0) continua, e ele é
assimétrico — sabe que o `knee` tem 0.756 de curso para baixo e 2.211 para cima.
A faixa é simétrica e não consegue expressar isso.

É pré-requisito de M5 e de M6.

### M3 — `apoiada` vira faixa

`comando.py:1303`

```python
apoiada = ((forca >= c.fracao_do_peso_apoiada * peso)
           & (forca <= peso + c.folga_apoiada_N))
```

Knob novo: `folga_apoiada_N`, em newton **absoluto**. A caixa vai de 1 a 5 kg; a
capacidade do robô de empurrar não muda com ela.

### M4 — `& ~apoiada` no fecho do PEGAR

`comando.py:1320`

```python
fecha[m] = (perto[m] & alinhado[m] & de_pe[m] & ~apoiada[m])
```

Pegar é a caixa sair da laje. Sem isto, M7 não pode descer abaixo de 0.80.

### M5 — `upright` entra na `PesoPorEstado`

Linha nova, com **BOTAR = 4** e 1 em todo o resto, **inclusive a CAUDA**.

O `upright` já mede inclinação do `torso_link` (`unitree_g1_flat_env_cfg:147`),
peso 1.0, `exp(−sin²(inclinação)/0.2)`. Deitar o tronco custa no máximo 1/s contra
uma renda de ~16/s no BOTAR.

⚠ A CAUDA fica em 1 de propósito. Ver §3, buraco B4.

### M6 — `faixa_de_pose` entra na `PesoPorEstado`

Linha nova, com **BOTAR = 2** e 1 em todo o resto.

É o conserto das mãos tortas, e ele faltava na lista original. A coluna BOTAR
multiplica os sete de manipulação por 2 e a faixa fica em 1 — o preço não escala
com o pagamento, e é exatamente no BOTAR que os braços vão ao batente.

Exige M2: sem ela, M6 dobra também o imposto sobre o agachamento.

### M7 — `altura_carregar` sorteado por episódio

Faixa **0.75 a 1.0**, buffer por env, sorteado no reset, no padrão do `limpo_topo`.

O `alvo_b` já é observável (canal 9 da observação), portanto a randomização é
aprendível e não vira ruído.

O check A3 muda de forma: ele passa a exigir M4 sempre que o mínimo da faixa ficar
abaixo de 0.80.

---

## 2. Verificação contra os três problemas

### Problema 1 — velocidade exagerada: **M1 conserta a forma, não garante a magnitude**

Medido no `juntas14k.csv` (MuJoCo clássico, `--impratio 10`, cadeia do carregar):

| fase | `mean` | `amax` | razão |
|---|---|---|---|
| espera#1 | 0.22/s | 3.04/s | 14× |
| pegar#1 | 0.15/s | 1.49/s | 9.6× |
| espera#2 | 0.19/s | 2.44/s | 13× |
| carregar#1 | 0.07/s | 1.79/s | 26× |
| **manipulação** | **0.11/s** | **1.78/s** | **15.5×** |

⚠ **Esta gravação é muito mais calma que o treino.** O p50 de velocidade de junta
aqui é 0.10 rad/s; a métrica `velocidade_de_junta` do painel marcava RMS **2.873**.
Um robô só e bem-comportado contra 4096 envs com exploração e quedas.

A razão medida (15.5×) depende de quantas juntas correm ao mesmo tempo. Aqui é 1 a
3. No treino serão mais, portanto a razão real fica menor — a estimativa honesta é
**3× a 8×**, não 15×.

O termo custa hoje **−2.6/s** no painel. O prêmio por fechar 1 s antes:

| | teto ao vivo |
|---|---|
| durante o BOTAR | ~35/s (15.8 dos sete ×2 + 13.8 congelado + 4 rastreio + 2 postura) |
| na CAUDA | ~58/s (29.6 congelado + 16 `postura_ereta`×8 + 8 `pose`×8 + 4 rastreio) |

**Fechar 1 s antes vale ~23/s.** Com `amax` a 3×, o termo vai a 8/s e perde. A 8×,
vai a 21/s e empata.

**Veredito:** M1 é necessária e não é suficiente sozinha. O peso é o botão da
magnitude e ele é livre — calibrar no primeiro bloco contra o painel. E a raiz
(§4, A1) continua intocada.

### Problema 2 — se apoiar na caixa: **M3 fecha**

O mecanismo não é economia de postura. É compra do fecho.

```
apoiada = forca >= 0.5 × m·g          limiar de 4.9 N com a caixa de 1 kg
```

O robô pesa ~343 N. Apoiar 1.5% do peso do corpo já dispara. E o fecho do BOTAR é a
porta da CAUDA, que paga ~58/s contra ~35/s do BOTAR.

Escorar vale a entrada antecipada na cauda. Custa ~1/s de `upright`. Fica 23× mais
barato que o que compra.

M3 não cobra: **tira o efeito**. Escorar acima do teto passa a BLOQUEAR o fecho.

M2 e M5 completam, redirecionando para o agachamento — que é como um humano põe uma
caixa numa laje baixa.

**Veredito:** fecha, desde que `folga_apoiada_N` seja calibrado (ver B2).

### Problema 3 — mãos tortas: **M6 fecha o caso do BOTAR**

Medido na cadeia do carregar, desvio máximo por fase contra a faixa:

| junta | pegar#1 | faixa | excesso |
|---|---|---|---|
| `left_wrist_pitch` | 1.41 | 0.6 | **0.81** |
| `right_wrist_yaw` | 0.94 | 0.5 | **0.44** |
| `left_wrist_yaw` | 0.76 | 0.5 | 0.26 |
| `right_wrist_roll` | 1.10 | 0.9 | 0.20 |
| `right_shoulder_yaw` | **0.28** | 0.8 | — |

O `faixa_de_pose` **já consertou os casos graves**: o `right_shoulder_yaw` saiu de
2.65 (no batente de 2.62) para 0.28. O que sobra são os punhos, 0.2 a 0.8 rad fora,
custando ~0.70/s — 5% da renda.

Na cadeia do botar o quadro era muito pior: `right_shoulder_yaw` travado em 2.63 e
`right_wrist_roll` em 2.15, além do curso de 1.97. É a coluna BOTAR: pagamento ×2,
preço ×1.

**Veredito:** M6 fecha o BOTAR. O resíduo dos punhos pede a `escala` (1.5 → ~1.0),
que torna a exponencial mais íngreme sem mexer no peso. Não está no plano — medir
depois de M6.

---

## 3. Buracos que o plano abre

### B1 — `amax` capta transiente de contato

Um passo de contato produz velocidade de junta aparente alta. Hoje a média sobre 29
dilui isso. Com `amax` e peso alto, o termo passa a cobrar ruído que a política não
escolheu.

**Mitigação:** manter o peso em −2.0 no primeiro bloco e ler a variância do termo
antes de subir.

### B2 — `folga_apoiada_N` apertada demais mata o BOTAR

Se o teto ficar abaixo do que um pouso normal produz, o BOTAR nunca fecha e a cadeia
inteira morre. É o maior risco do lote.

**Mitigação:** começar generoso (30 N) e apertar por medição. O `impacto_da_caixa`
já grava o pico por episódio com `reduce="max"` — o número sai do primeiro bloco.

### B3 — `& ~apoiada` e a laje no chão

Nos níveis 4 a 6 a laje fica a 0.04 m. A caixa continua apoiada no geom da laje,
portanto o sensor continua valendo. Se a caixa cair da laje, `apoio_caixa` = 0 e
`~apoiada` fica trivialmente verdadeiro — mas o `perto` falha, porque o alvo está a
0.75–1.0 m e a caixa no chão está a ~0.1 m. Sem buraco.

### B4 — `upright` na CAUDA aumentaria o prêmio da corrida

Subir o `upright` na CAUDA acrescenta até 3/s a um estado que já paga ~58/s, e isso
aumenta o ganho de fechar cedo — brigando com M1.

Por isso M5 sobe **só a coluna BOTAR**. A CAUDA já tem `postura_ereta`×8 e `pose`×8,
que pagam por estar de pé e na pose default; `upright` ali é redundante.

### B5 — M6 sem M2 dobra o imposto do agachamento

A coluna BOTAR de M6 multiplica a faixa inteira por 2, incluindo a linha da perna.
Sem M2 o robô fica numa pinça: um termo proíbe dobrar a perna, o outro proíbe deitar
o tronco, e a laje continua baixa. Política travada.

**M2 tem de entrar junto com M6, não depois.**

### B6 — M7 facilita o `de_pe`

Com alvo em 0.75 o robô alcança sem se esticar, portanto o portão `de_pe` do fecho
do PEGAR fica mais fácil e o elo fecha antes. Isso soma à corrida. Efeito pequeno
contra o ganho de generalização.

---

## 4. Auditoria: o que o pagar por segundo incentiva hoje

### A1 — o episódio não acaba no sucesso. É a raiz da pressa.

Episódio fixo de 20 s. Terminar a cadeia converte segundos pobres (a tarefa, ~35/s)
em segundos ricos (a cauda, ~58/s).

Fechar 1 s antes vale ~23/s. Nenhuma das sete mudanças toca nisso; M1 tenta vencer
esse prêmio por fora.

**Duas saídas, e as duas são cirurgia:**

- terminar o episódio no fecho da cadeia, com bootstrap de valor no corte
- pagar a cauda a partir de um tempo fixo, e não do fecho

A segunda foi levantada e recusada: `renda_congelada` garante o arranque, e o treino
vai rodar do zero. Fica registrada como a correção de raiz, para quando o modelo
souber fazer as tarefas.

### A2 — `renda_congelada` congela UM INSTANTE

```python
self.congelado += torch.where(fechou_agora, self.soma_anterior, 0)
```

`soma_anterior` é a soma dos sete no passo ANTERIOR ao fecho. Ela é anuitizada pelo
resto do episódio.

Consequência: o robô é pago por um **pico**, não por um estado sustentado. Ele pode
ficar perfeito num passo, deixar o fecho disparar ali, e relaxar no passo seguinte —
a anuidade já está travada.

**Conserto barato:** congelar uma média móvel (EMA de ~0.2 s) em vez do instante. Um
buffer, uma linha, e remove a classe inteira de exploração por pico.

### A3 — `descarga` tem clamp em zero

```python
descarga = (1.0 - f / peso).clamp(0.0, 1.0)
```

Entre `f = m·g` e `f = ∞` a derivada é zero. Empurrar a caixa para baixo é
invisível ao `unload` e ao `load`. Escorar é grátis — e é a forma mais ROBUSTA de
garantir `load = 1` e `apoiada = True`.

M3 tira o proveito. O canal morto fica.

### A4 — `squeeze` é cego acima de ~3×F_ref

`tanh(F / F_ref)` com `F_ref = m·g/(2µ)` = 6.13 N na caixa de 1 kg:

| força | `squeeze` |
|---|---|
| 6.1 N | 0.76 |
| 18.4 N | 0.995 |
| **68.9 N** (medido) | **1.0000** |

Esmagar não paga, mas também não custa. E é a maneira mais robusta de garantir a
preensão, portanto a política não tem motivo para parar.

**Conserto na mesma forma de M3:** faixa em vez de saturação — máximo em `F_ref`,
caindo acima. Não está neste plano.

### A5 — os preços de movimento estão fora da tabela

`knobs.py:1123` declara: `upright`, `terminacao`, `contato_*`, `joint_acc`,
`action_rate_l2`, `velocidade_por_regime` e `renda_congelada` ficam fora do
`PesoPorEstado`.

No BOTAR o pagamento é ×2 e todos esses ficam em ×1. Correr, escorar e sacudir saem
2× mais baratos justo no estado em que ele mais corre, escora e sacode.

M5 e M6 corrigem dois. Os outros ficam, e `velocidade_por_regime` tem um aviso
explícito no código contra entrar na tabela — mexer ali é desvio declarado, não
descuido.

---

## 5. Ordem de implementação

| bloco | mudanças | por quê juntas |
|---|---|---|
| 1 | M2 + M6 | B5: uma sem a outra trava a política |
| 2 | M3 + M4 | mesma função, `_fecha_elo_corrente` |
| 3 | M5 | independente |
| 4 | M1 | independente, e é a que mais depende de calibração |
| 5 | M7 | depende de M4 |

Cada bloco fecha com `smoke.py` verde.

## 6. Invariantes novos no smoke

| invariante | por quê |
|---|---|
| `folga_apoiada_N > 0` e o teto acima do peso máximo da caixa | B2 |
| M4 presente sempre que o mínimo do alvo < 0.80 | A3 muda de forma |
| `faixa_de_pose.perna[BOTAR] == 0` enquanto a linha da faixa no `PesoPorEstado` for > 1 | B5 |
| a coluna CAUDA de `upright` continua em 1 | B4 |
| a faixa de `altura_carregar` é observável (`alvo_b` no bloco 9) | M7 |

## 7. Números que só o treino entrega

| knob | de onde sai |
|---|---|
| `folga_apoiada_N` | `impacto_da_caixa`, `reduce="max"`, bloco 1 |
| peso de `velocidade_por_regime` | `Episode_Reward/velocidade_por_regime` depois de M1 |
| coluna BOTAR de `upright` | painel do bloco 1 |
| `faixa_de_pose.escala` | resíduo dos punhos depois de M6 |
