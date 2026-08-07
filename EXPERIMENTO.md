# Experimento: locomoção por rastreio de velocidade

| campo | valor |
|---|---|
| branch | `exp/locomocao-velocidade` |
| base | `exp/curriculo-teste` |
| data de abertura | 2026-08-07 |
| estado | desenho fechado, implementação não iniciada |

Registro escrito em ASD-STE100.

---

## O que este experimento testa

**Hipótese.** O `andar` do multi-tarefa não aprende porque o comando de velocidade vem
do alvo. A hipótese é que um comando **sorteado**, no padrão do mjlab, destrava a
locomoção — e que três descalibrações de recompensa desaparecem junto, sem ajuste.

**Critério de refutação.** Se `locomover/perf` continuar num platô abaixo de 0,10 depois
de 1 000 iterações com comando sorteado, a hipótese cai. A causa passa a ser outra.

**Escopo desta run.** Ela é teste rápido, não treino final. Ela responde duas perguntas:
o robô pega a caixa, e o currículo faz sentido. Ela não busca política final.

**Regra de trabalho.** Reusar mecanismo existente. Não inventar termo novo. Inventar
dificultou os treinos passados.

---

## 1. O diagnóstico que motivou

O treino travou. O `andar` não passou de 5% de sucesso entre as iterações 421 e 790.

| medição | valor |
|---|---|
| comando efetivo | 0,052 m/s |
| `v_max` configurado | 0,5 m/s |
| altura máxima do pé | 5 mm |
| `andar/perf_n0` | 0,04 a 0,06, estável |
| iterações para chegar a 0,90 | ~7 850 |

A extrapolação linear previu 0,0458 na iteração 790. A medição deu 0,04 a 0,06. A
extrapolação geométrica previu 0,165 e errou.

Os 5% não são caminhada. O robô vagueia 0,91 m em 20 s. O alvo fica a 1,0 m. O raio de
chegada é 0,25 m. O acerto vem da deriva.

## 2. A causa raiz

O comando do `andar` vinha do alvo. A linha `v_alvo *= cos(erro_rumo)` fecha um ciclo:

```
robô não gira -> comando ≈ 0 -> nunca recebe ordem de andar -> não aprende -> não gira
```

O mjlab quebra esse ciclo. Ele sorteia `lin_vel_x ~ U(−1, 1)` sempre. O robô recebe a
ordem mesmo sem saber executar.

**Três descalibrações são um sintoma só.**

| termo | por que parecia errado | com comando ±1,0 |
|---|---|---|
| `track_linear` std 0,5 | span de 4% a 0,1 m/s | span de 98% |
| `foot_clearance` −2,0 | o passo custa 2× o ganho | proporção do mjlab volta |
| `action_rate` −0,25 | reflexo caro | já corrigido para −0,10 |

A correção do `action_rate` foi confirmada por medição em 06/08. O portão do push fechou
em 0,9449, com um empurrão 2,4 vezes mais forte que o anterior.

Restaurar a faixa de comando é uma mudança. Afinar os três termos seriam três mudanças,
com uma run cada.

## 3. A estrutura nova

Três grupos, cinco tarefas. Todas usam o mesmo código de locomoção. Manipulação é
locomoção com comando fixado em zero.

```
LOCOMOÇÃO
  └─ locomover              velocidade: 1,0 / 1,5 / 2,0

MANIPULAÇÃO                 comando zero
  ├─ pegar                  altura · peso
  ├─ botar                  altura · peso
  └─ reorientar             giro · altura

LOCO-MANIPULAÇÃO
  └─ locomover_carregando   velocidade · peso
```

| origem | destravamentos |
|---|---|
| `locomover` | 4 |
| `pegar` | 11 |
| `locomover_carregando` | 7 |
| `reorientar` | 10 |
| `botar` | 10 |
| **total** | **42** |

## 4. As recompensas

### Termos globais

Estes termos valem nas cinco tarefas, com peso fixo.

| termo | peso | o que mede |
|---|---|---|
| `upright` | +1,0 | gravidade projetada no tronco |
| `foot_clearance` | −2,0 | altura do pé × velocidade do pé |
| `foot_swing_height` | −0,25 | pico de altura no pouso |
| `feet_slip` | −0,1 | velocidade do pé em contato |
| `soft_landing_feet` | −1e-5 | força de impacto |
| `self_collisions` | −1,0 | auto-colisão |
| `dof_pos_limits` | −0,5 | violação de limite de junta |
| `action_rate_l2` | **−0,10** | mudança de ação entre passos |
| `body_ang_vel` | −0,05 | taxa angular do tronco |
| `angular_momentum` | −0,02 | momento angular |
| `joint_acc` | −2,5e-7 | aceleração de junta |
| `table_contact` | −1,5 | contato com a prateleira |
| `back_penalty` | −0,5 | contato nas costas |
| `terminacao` | −4,0 real | queda |

Os quatro termos de marcha se desligam sozinhos. O gate deles é `‖cmd‖ > 0,05`.
Manipulação tem comando zero, então eles ficam inertes sem nenhum gate por tarefa.

⚠️ **`track_linear_velocity` e `track_angular_velocity` NÃO são globais.** O `env.py:430`
os gateia em quatro tarefas. Eles estão desligados no `pegar`, no `botar` e no
`reorientar`.

A razão é orçamentária. Com comando zero os dois kernels valem 1,0 exato com o robô
imóvel. Ligá-los na manipulação daria 4,0 por ficar em pé, contra 4,0 de tarefa. O sinal
de manipulação viraria metade do orçamento.

A base de locomoção é compartilhada. O pagamento por ela não pode ser.

### Orçamento de tarefa

Cada tarefa recebe um fator. O fator escala os termos próprios dela para somar **4,0**.

O alvo 4,0 vem do próprio mjlab. Ele é `track_linear + track_angular`.

### Termos por tarefa

| tarefa | termos próprios | soma | fator |
|---|---|---|---|
| `locomover` | `track_lin` 2,0 · `track_ang` 2,0 | 4,0 | 1,000 |
| `pegar` | `lift` 2,0 · `reaching` 1,0 · `box_at_peito` 1,0 · `grasp` 0,5 | 4,5 | **0,889** |
| `botar` | `box_at_prateleira` 1,0 | 1,0 | 4,000 |
| `reorientar` | `orienta_face` 1,0 · `reaching` 1,0 | 2,0 | 2,000 |
| `locomover_carregando` | `track_lin` 2,0 · `track_ang` 2,0 · `box_at_peito` 1,0 | 5,0 | **0,800** |

O `locomover` **tem** termos próprios. São os dois de rastreio, gateados nele. Eles somam
4,0, que é o alvo, então o fator dele é 1,000.

No `locomover_carregando` os mesmos dois entram, mais o `box_at_peito`. O `exige_grasp`
os gateia ali: eles só pagam com a preensão ativa.

### Anti-hacks

| termo | peso | escopo |
|---|---|---|
| `com_balance` | −2,0 | ligado quando `‖cmd‖ ≈ 0` |
| `box_shake` | −0,15 | desligado no `reorientar` |

## 5. O comando

Entra o `UniformVelocityCommandCfg` do mjlab.

| campo | valor |
|---|---|
| `lin_vel_x`, `lin_vel_y` | `(-1, 1)` |
| `ang_vel_z` | `(-0.5, 0.5)` |
| `resampling_time_range` | `(3, 8)` |
| `rel_standing_envs` | `0,1` |
| `rel_heading_envs` | `0,3` |
| `rel_forward_envs` | `0,2` |

O comando é `[vx, vy, ωz]`. O terceiro campo é taxa de guinada. Ele não é orientação.

O `0` não é nível da escada. Ele é `rel_standing_envs`, presente em todos os níveis.

O sorteio é contínuo dentro do teto. Ele não escolhe entre cinco valores discretos.

Uma subclasse zera o comando nas três tarefas de manipulação.

**O mjlab tem eixo de velocidade**, o `command_vel`
(`tasks/velocity/mdp/curriculums.py:83-108`). Ele avança nas iterações 5 000 e 10 000,
de `(−1,1)` para `(−1,5 · 2,0)` e depois `(−2,0 · 3,0)`. Ele é cronograma por contagem de
passos. Ele não mede desempenho. Nosso eixo mede.

## 6. A postura

O `variable_posture` escolhe o `std` pelo comando, não pela tarefa.

| junta | parado | andando | correndo |
|---|---|---|---|
| joelho | 0,05 | 0,35 | 0,60 |
| ombro | 0,05 | 0,15 | 0,50 |

A fusão obriga essa troca. Depois dela, o comando varia dentro do episódio. Um gate por
tarefa não consegue mais expressar o regime.

### O termo é BONIFICAÇÃO, não penalidade

| onde | peso |
|---|---|
| `knobs.py:178` | `postura = +0,5` |
| mjlab `velocity_env_cfg.py:296` | `pose weight = +1,0` |

A função devolve `exp(−média(err²/std²))`. O valor fica em `(0, 1]`. Ele nunca é negativo.

Sair da pose padrão **para de pagar**. Isso não cria dívida. Nada é proibido.

O bônus é pequeno perto da tarefa. O `pegar` soma 4,0 depois da equalização; a postura
vale 0,5. O robô sai da pose, porque pegar paga oito vezes mais.

Depois da pega os termos de tarefa saturam. Sobra o bônus de postura como única
recompensa marginal. O robô aprende a **parar nela no final**.

A postura é atrator de repouso. Ela não é obstáculo no caminho.

⚠️ O registro de 03/08 diz que o termo "cobrava 0,5 de quem agacha". Isso é custo de
oportunidade, não punição. Se 12% do orçamento chega para enviesar contra o agachamento é
pergunta empírica.

### O que muda

**O `variable_posture` não muda.** Nem escala, nem peso, nem mecanismo.

O código passa de quatro termos de postura para dois.

| tarefa | escopo |
|---|---|
| `locomover` | corpo todo |
| `locomover_carregando` | pernas e cintura |
| `pegar`, `botar`, `reorientar` | **não decidido** |

O escopo é a segunda dimensão, e ela continua sendo parâmetro. O `env.py:516-522` já a
separa: o `std` responde ao regime de velocidade, o escopo responde a se a mão está
ocupada. O `variable_posture` resolve só a primeira.

O `variable_posture` é uma classe. O `gated()` chama `inner(env, **kw)`, então uma classe
passada como `inner` produz uma instância. A saída é uma subclasse que multiplica a
máscara de tarefa no resultado.

## 7. O push

O push vira evento fixo do mjlab. Ele fica sempre ligado, com magnitude fixa, em todos os
envs.

O eixo de currículo sai. Saem também `push_fator`, `push_nivel`, `_push_competente` e
`_destravar_push`.

Sai a força sustentada. O `JANELA_LIVRE_S` de 0,5 s permanece, porque as tarefas de
preensão nascem com a palma apenas tocando a caixa.

## 8. O critério de sucesso

O critério base vale para as cinco tarefas:

```
erro_lin = (1/T) ∫ ‖v_cmd_xy − v_xy‖ dt        # m/s
erro_ang = (1/T) ∫ |ωz_cmd − ωz|   dt          # rad/s

base = não_caiu & (erro_lin ≤ TOL_V) & (erro_ang ≤ TOL_W)
```

O critério é físico. Ele mede metros por segundo. A recompensa calcula `exp(−erro²/σ²)`.
As duas funções são diferentes, então o `σ` da recompensa não move a régua.

Com comando zero, o critério mede a deriva. Ele reprova os 0,91 m que o critério atual
aprova.

| tarefa | condição adicional |
|---|---|
| `locomover` | nenhuma |
| `locomover_carregando` | preensão · caixa no peito |
| `pegar` | preensão · caixa no peito · de pé |
| `botar` | caixa no alvo · sem preensão · caixa quieta · de pé |
| `reorientar` | ângulo < 10° · desvio xy < 0,05 · apoiada |

O `de_pé` sai do critério de locomoção. O `pose` e o `upright` já produzem a postura, de
forma contínua.

**Alternativa rejeitada antes da implementação:** o critério por deslocamento. Com
reamostragem a cada 3 a 8 s, as direções se cancelam. Um robô parado passaria.

**Risco aceito:** o `TOL_V` fica fixado sem medição do deslocamento de pelve durante o
agachamento. Decisão do dono do experimento, com base em observação no `play`.

## 9. Os desbloqueios

### O grafo

```
locomover  (nasce aberta)
   ├──> pegar
   │      └──> locomover_carregando
   │                 └──> botar
   └──> reorientar
```

### Quando um destravamento acontece

O orquestrador roda a cada reset. Ele percorre as tarefas abertas. Três condições valem
para cada uma.

| # | condição | valor | código |
|---|---|---|---|
| 1 | episódios desde o último evento **daquela tarefa** | ≥ 200 | `curriculum.py:477` |
| 2 | nenhuma célula da tarefa está congelada | — | `curriculum.py:480` |
| 3 | `_min_tarefa(t)` ≥ limiar | 0,90 | `curriculum.py:482` |

O `_min_tarefa(t)` é o mínimo sobre **todos os níveis já abertos de todos os eixos** da
tarefa. A EMA usa `alpha = 0,03`.

A tarefa destrava quando o pior nível aberto dela chega a 0,90.

A condição 1 conta **episódios daquela tarefa**, não iterações. O `_medir` soma quantos
envs da tarefa terminaram naquele reset (`curriculum.py:248`). O contador zera no evento
(`curriculum.py:504`). A função dele é impedir destravamento por ruído.

A condição 1 não limita na prática. Com 4 096 envs, 200 episódios de uma tarefa passam em
poucas iterações. O portão que limita é o 0,90.

O congelamento tem histerese. Uma queda maior que 0,10 contra a referência lenta congela.
Uma volta a menos de 0,05 descongela.

### O que destrava, em que ordem

O `_destravar` segue uma prioridade fixa.

1. **Tarefa nova primeiro.** Se a tarefa tem filho fechado, ele abre. O eixo espera.
2. **Depois, um eixo.** O código escolhe o eixo com maior folga, ou seja, o de melhor
   competência mínima. O empate cai em round-robin sobre `AXIS_ORDER`.

Um destravamento por evento, por tarefa. Duas tarefas podem destravar na mesma chamada.

### A sequência do `locomover`

| ordem | o que abre |
|---|---|
| 1 | tarefa `pegar` |
| 2 | tarefa `reorientar` |
| 3 | velocidade n1 (teto 1,5) |
| 4 | velocidade n2 (teto 2,0) |

O `pegar` abre com teto de velocidade 1,0. Ele não espera o robô correr a 2,0 m/s.

## 10. Mudanças de 07/08

### `arm_vel` sai

O termo tinha peso −0,002 sobre as juntas de braço.

Três motivos, e o próprio docstring dele (`knobs.py:198-212`) já registrava o primeiro:

1. Nas tarefas que carregam, o braço é estrutura, não gesto. Punir velocidade de braço
   compete com a tarefa. O docstring o deixou "no radar do bloco 1".
2. O docstring o chamou de "peso muito baixo". A medição contradiz: `contrib/parado/arm_vel
   = −0,0691` é o **segundo maior custo**, atrás só do `action_rate`.
3. O `variable_posture` passa a moldar o braço por posição (`std_walking` de ombro 0,15,
   cotovelo 0,15, punho 0,3). O `arm_vel` molda por velocidade. Os dois fazem a mesma
   coisa. E manipulação, que não tem termo de postura, ficaria só com o `arm_vel` freando
   o braço justo onde o braço é a tarefa.

### `botar` ganha escala grossa

O `box_at_prateleira` usa `std` único de **0,05**, herdado da Lift, onde ele servia à
precisão final. A caixa a 30 cm do alvo rende `exp(−0,09/0,0025) ≈ 0`. Não há gradiente
até a caixa já estar quase no lugar.

É o mesmo defeito do `track_linear` que travou o `andar`.

As outras duas tarefas de manipulação já usam escala dupla:

| termo | grosso | fino |
|---|---|---|
| `reaching` | 1,0 | 0,25 |
| `orienta_face` | 30° | 5° |
| `box_at_prateleira` | **0,0 (off)** | 0,05 |

O mecanismo já existe. O `box_at_prateleira` aceita `std_grosso`, com default `0.0`:

```python
k = height_kernel(err_sq, std)
if std_grosso > 0.0:
    k = 0.5 * height_kernel(err_sq, std_grosso) + 0.5 * k
```

**Ação: ADIADA.** A mudança está certa e não é desta run. O `botar` fica a quatro
destravamentos de distância e não é alcançado num teste rápido.

### `hold_still` sai

Ele era `_grasp × kernel(err², 0,25) × exp(−‖ω_pelve‖²/0,5²)`, com peso 0,5, nas tarefas
`PARADO_CAIXA`, `ANDAR_CAIXA` e `PEGAR`.

**Ele é redundante com o `track_angular_velocity`.** Os dois leem `root_link_ang_vel`, e
o root do G1 é a pelve. A norma de ω é invariante de frame.

| ‖ω_pelve‖ | `hold_still` perde | `track_ang` perde | razão |
|---|---|---|---|
| 0,5 | 0,32 | 0,79 | 2,5× |
| 1,0 (rebolado) | 0,49 | 1,73 | 3,5× |

O `hold_still` tem escala mais apertada. O `track_ang` tem quatro vezes o peso. O peso
ganha.

O docstring do `hold_still` lista quatro termos cegos ao rebolado de pelve:
`body_ang_vel`, `angular_momentum`, `feet_slip` e `posture`. Ele acerta nos quatro. Ele
não lista o `track_angular_velocity`, que é o que cobre.

A explicação: `hold_still_bonus` mora em `g1_training/skills/lift/`. A skill Lift é de
tarefa única e não tem comando de velocidade. O termo foi importado para um contexto que
já tinha a cobertura.

**Buraco que sobra.** O `pegar` não recebe `track_*`. Sem o `hold_still`, nada nele
enxerga rotação de pelve. Mitigação de custo zero: logar `‖ω_pelve‖` como diagnóstico.

**Consequência ligada, ADIADA.** O `hold_still` carregava a escala grossa do erro
caixa→peito (`gate_std = 0,25`). Sem ele o `box_at_peito` fica com escala única de 0,05.
Isso importa no `locomover_carregando`, que não é alcançado nesta run. O `pegar` tem
gradiente contínuo pelo `lift` e pelo `reaching`.

## 11. Decisões em aberto

### `exige_grasp` no `locomover_carregando`

O gate quebra a simetria com o `locomover`. Sem ele, a tarefa fica literalmente
`locomover` mais `box_at_peito` e `hold_still`.

O `largou` só encerra o episódio quando a caixa cai abaixo de 0,30 m. Existe um estado
intermediário: segurar a caixa mal, a 0,40 m, e andar.

| desenho | perder a preensão custa | fator |
|---|---|---|
| com `exige_grasp` | 4,0 de rastreio + `box_at_peito` | 0,727 |
| sem `exige_grasp` | só `box_at_peito` | 2,667 |

Simetria contra pressão de preensão. Não decidido.

### Escopo da postura na manipulação

O `variable_posture` resolve o `std` pelo comando. O escopo — quais juntas o termo mede —
continua sendo parâmetro por tarefa.

As três tarefas de manipulação não têm escopo definido. Não decidido.

## 12. O que sai do código

| arquivo | remoções |
|---|---|
| `commands.py` | `DesiredTwistCommand` inteiro |
| `tasks.py` | `distancia_andar`, `rumo`, `push`; `hold_still` de `TERMOS_DE_TAREFA`; tarefas `PARADO`, `ANDAR`, `PARADO_CAIXA`, `ANDAR_CAIXA` |
| `metrics.py` | `chegou_andar`, `alinhado`, `chegou`, `sustenta_andar_s`, `de_pé` na locomoção |
| `curriculum.py` | célula de push e as funções dela |
| `env.py` | dois dos quatro termos de postura; `arm_vel`; `hold_still` |
| `knobs.py` | `v_max`, `v_max_carga_cheia`, `a_max`, `w_max`, `alpha_max`, `d_morto_andar`, `morto_angular_rad`, `andar_raio`, `andar_raio_chega`, `heading_gain`, `arm_vel`, `hold_still` |

O `alpha_max` estava rotulado "PALPITE A VALIDAR". Ele deixa de existir.

## 13. O que permanece

- `afasta_cena`. A causa muda, mas o robô ainda esbarra na prateleira.
- As três guardas: `caixa_quieta` no `botar`, `desvio_xy` e `apoiada` no `reorientar`.
- A instrumentação `contrib/<tarefa>/<termo>`. Ela achou as três descalibrações.
- A observação continua com 154 números. O checkpoint atual carrega.

## 14. O custo escondido

Os dois portões de teste quebram.

| arquivo | acoplamento |
|---|---|
| `smoke.py` | 37 usos de `NUM_TASKS`, 28 de `PARADO`, 27 de `twist`, 22 de `ANDAR` |
| `sim_curriculo.py` | a premissa inteira é "o `andar` só abre com push completo" |

## 15. O que a reforma não conserta

A manipulação guarda 30 dos 42 destravamentos. A reforma arruma a frente da cadeia. Ela
não reduz o volume.

O orçamento continua sendo o limite duro. O treino tem 98 milhões de passos. O mjlab
gasta 2,95 bilhões só para andar.

## 16. Pendências

| item | estado |
|---|---|
| implementação | não iniciada |
| `smoke.py` e `sim_curriculo.py` | não atualizados |
| notebook da Kaggle | falta converter de Dataset para `git clone` |
| limpeza do log | não iniciada |
| `exige_grasp` e `hold_still` | decisão pendente |

A bagunça do log tem causa conhecida. Em `rsl_rl/utils/logger.py:186-202`, o mesmo laço
escreve no TensorBoard e monta a linha do console, na ordem de inserção do dict. Não há
ordenação. A limpeza deve reagrupar a saída, e não cortar chaves.
