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
  ├─ pegar                  altura
  ├─ botar                  altura
  └─ reorientar             giro

LOCO-MANIPULAÇÃO
  └─ locomover_carregando   velocidade
```

**Um eixo por tarefa.** A versão de 07/08 desta tabela dava dois eixos a quatro
tarefas. O segundo eixo era o `peso`, e ele deixou de ser eixo — ver a §9.

| origem | destravamentos |
|---|---|
| `locomover` | 2 |
| `pegar` | 6 |
| `reorientar` | 4 |
| `locomover_carregando` | 2 |
| `botar` | 6 |
| aberturas de tarefa | 4 |
| **total** | **24** |

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
| `rel_standing_envs` | `0,1` — **dividido**: 0,05 parado, 0,05 giro parado |
| `rel_heading_envs` | `0,3` |
| `rel_forward_envs` | `0,2` |
| `heading_command` | `True` |
| `ranges.heading` | `(−π, π)` |
| `heading_control_stiffness` | `0,5` |

O comando é `[vx, vy, ωz]`. O terceiro campo é taxa de guinada. Ele não é orientação.

O `0` não é nível da escada. Ele é `rel_standing_envs`, presente em todos os níveis.

O sorteio é contínuo dentro do teto. Ele não escolhe entre cinco valores discretos.

⚠️ **O sorteio contínuo vale para 70% dos envs.** Os três campos `heading_command`,
`ranges.heading` e `heading_control_stiffness` faltavam nesta tabela, e sem eles o
`rel_heading_envs` fica inerte em silêncio: o `is_heading_env` só é escrito dentro do
`if self.cfg.heading_command` (`velocity_command.py:80-84`). Com eles ligados, nos 30%
de envs de heading o `ωz` **não é sorteado** — ele é `clip(0,5 × erro_de_rumo, ±0,5)`,
recalculado a cada passo.

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

**O código passa de quatro termos de postura para UM** (decidido 07/08). Um
`variable_posture`, corpo todo, **sem gate**, peso +0,5. É o desenho do fabricante: um
`pose` só, com `joint_names=(".*",)` e nenhuma máscara.

O escopo por tarefa deixa de existir. Ele era a segunda dimensão do desenho antigo, e a
aritmética mostra que ele é desnecessário.

### Por que o escopo não é necessário

O escopo existia para impedir que a postura travasse o braço nas tarefas que manipulam.
Ela não trava. Ela deixa de pagar, e a tarefa paga oito vezes mais.

Mais que isso: **o termo de corpo inteiro se desliga sozinho** onde o braço é a tarefa.

Com comando zero, o `variable_posture` usa o regime `standing`. O cfg do g1 põe
`std_standing = {".*": 0,05}` para todas as juntas, braços inclusive
(`config/g1/env_cfgs.py:107`).

Um ombro deslocado 0,5 rad para alcançar a caixa dá `0,25 / 0,0025 = 100`. Com 8 juntas
de braço de 29, a média fica em ~27,6, e o termo vale `exp(−27,6) ≈ 1e−12`.

Isso é zero em float32. O gradiente também é zero. O termo não pode enviesar nada.

### A consequência declarada

O termo fica inerte nas **quatro** tarefas com caixa, e não só nas três de manipulação.

No `locomover_carregando` o comando não é zero, então o regime é `walking` e o ombro
recebe `std = 0,15`. Mas os braços seguram a caixa no peito. Um ombro a 1,0 rad dá
`1 / 0,0225 = 44`, média ~12, e `exp(−12) ≈ 6e−6`. Zero também.

Portanto o `locomover_carregando` perde o `posture_carrega`, que hoje paga 0,5 por manter
a perna perto da pose enquanto carrega.

O que sobra ali: os quatro termos de marcha, porque o comando não é zero; mais o
`upright`, o `body_ang_vel` e o `angular_momentum`, que são globais.

A alternativa rejeitada era parar em dois termos — corpo todo no `locomover`, perna no
`locomover_carregando`. Um termo é mais simples e é o que o fabricante entrega.

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

O currículo FICA. As opções "mjlab puro" e "medir sem destravar" foram descartadas. A
razão é do dono do experimento: o robô não aprende as tarefas por exploração, e ele
precisa de base.

O que muda é a estrutura. Ela fica mais simples.

### O grafo — quatro camadas

```
1  locomover                        (nasce aberta)
2  pegar        reorientar          abrem JUNTOS
3  locomover_carregando             pais: pegar E reorientar
4  botar                            pai: locomover_carregando
```

A ordem tem razão física. O `reorientar` exige aproximar, tocar e aplicar força. O
`pegar` exige tudo isso, mais força normal suficiente e mais erguer.

O grafo é declarado por PAIS, e não por FILHOS:

```python
PAIS = {
    locomover:            (),
    pegar:                (locomover,),
    reorientar:           (locomover,),
    locomover_carregando: (pegar, reorientar),      # junção AND
    botar:                (locomover_carregando,),
}
```

O `pegar` e o `reorientar` têm o mesmo pai, então abrem no mesmo evento. Isso substitui
a regra F9 de "um filho por evento".

### A regra do evento

```
evento da tarefa T:
    condição:  episódios(T) ≥ 200  E  perf[T][topo] ≥ 0,90

    ação 1:    se a DR de peso de T ainda está fechada:  abre a DR
               senão:                                    abertos[T] += 1

    ação 2:    para cada F que tem T entre os pais:
                   se TODO pai P de F tem eventos[P] ≥ 1:  abre F no nível 0
```

As duas ações rodam no MESMO evento. A tarefa nova começa no nível 0 enquanto a tarefa
mãe avança. Nada serializa.

O portão do filho é `eventos[P] ≥ 1`. Ele é um inteiro por tarefa, monotônico. O
primeiro evento de uma tarefa dispara exatamente quando ela chega a 0,90 na configuração
mais fácil dela.

⚠️ O portão do filho nunca olha nível difícil. A cadeia não trava atrás de um nível duro.

### O peso não é eixo. São dois níveis de DR

| nível de DR | massa da caixa |
|---|---|
| 0 | 1 kg fixo |
| 1 | `U(1, 5)` kg |

**O sucesso não é atrelado ao peso.** O critério é "fez o que tinha que fazer", qualquer
que seja a massa.

Portanto o peso não tem célula, não tem EMA e não tem portão próprio. Ele é um booleano
por tarefa, dentro do `state_dict`, que a DR lê.

O booleano vira `True` no PRIMEIRO evento da tarefa. É a ação 1 acima. A tarefa nunca
recebe as duas dificuldades no mesmo passo: o 1º evento alarga a carga, o 2º avança o
eixo específico.

O objetivo declarado é ensinar o nível fácil antes de entregar a tarefa completa.

⚠️ Depois do alargamento, `perf[(t, eixo)][k]` é a taxa de sucesso marginalizada sobre a
faixa de carga. Isso é aceito, e não é o travamento de 06/08. Aquele vinha de
marginalizar sobre um EIXO que ficava progressivamente mais difícil, com o PLR dando 30%
ao pior nível. Aqui a distribuição para de se mover depois do alargamento, e a política
converge contra ela. O portão de 0,90 passa a significar "0,90 sobre a faixa inteira de
carga", que é a tarefa real.

⚠️ É PESO, não INÉRCIA. O `dr.body_mass` corrompe a heap, então a carga é força externa
em −z. A caixa de 5 kg tem inércia de 1 kg. A DR randomiza carga ESTÁTICA.

### O portão

| # | condição | valor |
|---|---|---|
| 1 | episódios desde o último evento daquela tarefa | ≥ 200 |
| 2 | `perf[T][topo]` ≥ limiar | 0,90 |

A EMA usa `alpha = 0,03`. A condição 1 conta episódios daquela tarefa, não iterações, e
existe para impedir destravamento por ruído.

O gate lê o **nível corrente**, e não o `_min_tarefa` (mínimo sobre todos os níveis
abertos). Consequência: o eixo avança mesmo se um nível anterior regrediu. O piso `ρ/L`
continua sorteando os níveis antigos, então a regressão aparece no log. Ela vira
observabilidade, não portão.

### O que sai do orquestrador

| some | por quê |
|---|---|
| `FILHOS` e a prioridade 1 do `_destravar` | viram `PAIS` mais o teste AND |
| a regra F9 "um filho por evento" | os filhos abrem juntos |
| `AXIS_ORDER`, `self.rr` e o round-robin | um eixo por tarefa, nada a desempatar |
| `_min_tarefa`, `_min_cel`, `_push_competente` | o gate vira `perf[T][topo]` |
| o condicionamento do `_medir` | não há outro eixo para condicionar |
| o congelamento: `ref`, `congelado`, `congela_queda`, `descongela` | ele bloqueava a abertura do filho, e a referência já é EMA lenta desde a S3 |
| a célula `(PARADO, PUSH)` e o eixo `push` | o push vira evento fixo |
| os eixos `rumo`, `distancia_andar` e `peso` | o comando é sorteado; o peso vira DR |
| `sim_curriculo.py`, 472 linhas | a ordem cabe no bloco da regra do evento |

### O que fica

O sorteio de tarefa. O PLR com piso `ρ/L`. A EMA por célula. O portão 0,90. O gate de
200 episódios. O `state_dict`.

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

## 10b. Limpeza de 07/08 — segunda passada

Enquadramento novo do dono do experimento: ele reescreve o treino quase inteiro, e
reaproveita só algumas peças. Seis decisões fecharam nesta passada.

### `d_morto` sai inteiro, e o freio de z vai junto

O `d_morto` tem dois papéis. O papel A gera o comando (`v = 0` dentro do raio). Ele
morre com o `DesiredTwistCommand`.

O papel B abre o freio de z do `track_linear_velocity_freio_z`. Ele **já é inerte onde
importa**: o `env.py:430` gateia o termo em `(PARADO, ANDAR, ANDAR_CAIXA,
PARADO_CAIXA)`, e o `pegar` está fora. Agachar acontece só no `pegar`.

Consequência: volta o `vel_mdp.track_linear_velocity` do fabricante, sem cópia.
**Nenhum `command_threshold` substituto é necessário.**

### `rumo` sai

Saem junto o `erro_rumo_deg`, o `alinhado` e o `heading_gain`. O `ωz` é o único
mecanismo de giro.

### `fora_da_area` sai

Com comando sorteado, ela reprova o rastreio bom. Uma janela de 1 m/s por 5 s já cobre
os 5 m do raio, e ela é `time_out=False` — custa −4,0 e zera o `não_caiu`.

O `unitree_g1_flat_env_cfg` remove o `out_of_terrain_bounds` (`env_cfgs.py:209`). No
plano, o fabricante não termina por distância. O `nonfinite` cobre deriva
descontrolada.

### Giro parado ENTRA, com piso 0,15

O giro parado é o comando `vx = 0`, `vy = 0`, `ωz ≠ 0`.

| fato medido | valor |
|---|---|
| chance de o sorteio produzi-lo sozinho | 0,05 × 0,05 = **0,25%** |
| gate dos quatro termos de marcha | `‖cmd_xy‖ + \|ωz\| > 0,05` |
| sorteios de `ωz` abaixo de 0,05 | 10% |
| piso adotado | **0,15** |

O `ωz` **conta** no gate dos quatro termos de marcha (`velocity/mdp/rewards.py:233`,
`:263`, `:306`, `:338`, `:377`). Portanto o robô que gira parado continua pagando
`foot_clearance`, `foot_swing_height`, `feet_slip` e `soft_landing`. Esses quatro
termos não precisam de mudança.

O piso é obrigatório. Sem ele, 10% dos envs de giro ficam com o gate fechado e sem
sinal de tarefa nenhum.

O piso 0,15 é derivado, não digitado: o `rel_forward_envs` trava `lin_vel_x ≥ 0,3` num
teto de 1,0, ou seja 30% do teto. Trinta por cento do teto de `ωz` (0,5) dá 0,15.

A fração sai do `rel_standing_envs`, e não de uma fração nova. O regime parado é o mais
simples dos três, e reduzi-lo custa menos que reduzir dado de marcha.

⚠️ Armadilha de implementação: o `_update_command` zera o `is_standing_env` a cada
passo (`velocity_command.py:136`). Os envs de giro têm de sair dessa máscara.

### `heading_command` fica LIGADO

Ele é nativo do mjlab e já vem ligado na task que o fabricante entrega para o G1.

| item | onde |
|---|---|
| campo `heading_command` | `velocity_command.py:283` |
| `heading_command=True` | `velocity_env_cfg.py:184` |
| `rel_heading_envs=0.3` | `velocity_env_cfg.py:182` |
| `heading=(-π, π)` | `velocity_env_cfg.py:191` |
| override no g1 flat | nenhum |

O ganho é 0,5 e o teto é 0,5, portanto o `ωz` satura quando o erro passa de 1,0 rad
(57°). O `heading_target` sai de `U(−π, π)`, então em 68% dos sorteios o erro inicial
passa de 1,0 rad. O env começa saturado, e o `ωz` decai a zero conforme o robô gira.

O modo sorteado nunca produz essa trajetória. Ele pede taxa constante, e nunca pede
"pare de girar nesta orientação".

⚠️ **Isto não é o ciclo que travou o `andar`.** O ciclo antigo era
`v_alvo *= cos(erro_rumo)`: sem giro, o comando ia a zero. O heading tem o sinal
contrário — sem giro, o erro permanece grande e o `ωz` permanece no teto. A ordem
satura em vez de desaparecer.

**Interação com o giro parado.** A ordem de execução é resample → heading → zeramento
do standing. O heading sobrescreve o `ωz` do piso. O resultado é "gire parado até
apontar para X, depois pare" — que é o caso de uso da navegação que motivou o giro
parado.

## 11. Decisões em aberto

### ~~`exige_grasp` no `locomover_carregando`~~ — DECIDIDO 07/08

**O `exige_grasp` fica LIGADO.** O `locomover_carregando` exige preensão: os pads das
duas palmas tocando a caixa.

Isso é exatamente o que o `_grasp` mede
(`g1_training/skills/lift/rewards.py`): as duas palmas em contato E nenhum verso em
contato. Não há limiar de força. O spawn `SPAWN_SEGURANDO` já nasce com `_grasp = 1`,
então o gate não é obstáculo de aquisição — ele é pressão para não largar.

O que a decisão fecha: os dois termos de rastreio são multiplicados pela preensão nesta
tarefa. Largar a caixa custa **4,0 de rastreio mais o `box_at_peito`**.

O buraco que ela tapa: o `largou` só encerra o episódio quando a caixa cai abaixo de
0,30 m. Existia o estado intermediário de segurar mal, a 0,40 m, e andar. Sem o gate, esse
estado paga o rastreio inteiro.

O fator de orçamento não muda com a decisão. O `_equaliza_orcamento` soma pesos de termo,
e o `exige_grasp` é multiplicador de runtime. Com o `hold_still` fora (§10), os termos
próprios somam `2,0 + 2,0 + 1,0 = 5,0`, e o fator continua **0,800**, como a §4 registra.

⚠️ A simetria com o `locomover` quebra de propósito. As duas tarefas partilham os dois
termos de rastreio, e só nesta eles dependem da preensão.

### ~~Escopo da postura na manipulação~~ — DECIDIDO 07/08

O escopo por tarefa deixa de existir. Fica **um** `variable_posture`, corpo todo, sem
gate. A derivação está na §6.

**Não há decisão em aberto neste documento.**

## 12. O que sai do código

| arquivo | remoções |
|---|---|
| `commands.py` | `DesiredTwistCommand` inteiro; `_quintica`; `erro_rumo_deg` do `LiftTargetCommand` |
| `rewards.py` | `track_linear_velocity_freio_z` — volta o termo do fabricante |
| `terminations.py` | `fora_da_area` |
| `tasks.py` | eixos `distancia_andar`, `rumo` e `push`; `AXIS_ORDER`; `peso` sai de `T.AXES` — a tabela de 2 valores dele fica, porque é a DR; `hold_still` de `TERMOS_DE_TAREFA`; tarefas `PARADO`, `ANDAR`, `PARADO_CAIXA`, `ANDAR_CAIXA` |
| `metrics.py` | `chegou_andar`, `alinhado`, `chegou`, `sustenta_andar_s`, `de_pé` na locomoção |
| `curriculum.py` | célula de push e as funções dela; `FILHOS` e a prioridade 1 do `_destravar`; `self.rr` e o round-robin; `_min_tarefa`, `_min_cel`, `_push_competente`; o condicionamento do `_medir`; o congelamento (`ref`, `congelado`, `_congelamento`) |
| `sim_curriculo.py` | o arquivo inteiro, 472 linhas |
| `events.py` | `payload_por_nivel` vira `payload_dr` — o teto sai do booleano, não de `env.nivel["peso"]` |
| `env.py` | **três dos quatro** termos de postura e o gate do que sobra; o helper `_so_pernas`; `arm_vel`; `hold_still`; o override de `track_linear_velocity.func`; o registro da terminação `fora_da_area` |
| `knobs.py` | `v_max`, `v_max_carga_cheia`, `a_max`, `w_max`, `alpha_max`, `d_morto_andar`, `d_morto_manipula`, `d_freio_extra`, `morto_angular_rad`, `andar_raio`, `andar_raio_chega`, `andar_raio_mantem`, `alinhado_chega_deg`, `alinhado_mantem_deg`, `area_raio`, `heading_gain`, `arm_vel`, `hold_still`; `postura_std_parado`, `postura_std_manipula` e `postura_joints`; o bloco `Push` inteiro; `congela_queda`, `descongela_dist_pico` e `ema_alpha_lenta` do bloco de currículo |

O `alpha_max` estava rotulado "PALPITE A VALIDAR". Ele deixa de existir.

⚠️ O `erro_rumo_deg` mora no `LiftTargetCommand`, não no `DesiredTwistCommand`. A linha
"`DesiredTwistCommand` inteiro" não o cobre. Ele tem dois leitores vivos
(`metrics.py:141` e `:269`), os dois no `alinhado`, que também sai.

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

O `sim_curriculo.py` não é reescrito. Ele sai (§9). Com o grafo por camadas e um eixo
por tarefa, a ordem dos destravamentos cabe no bloco da regra do evento.

Sobra o `smoke.py`, com 1 832 linhas. Ele é o único portão de teste que resta, e ele
está acoplado a nomes que deixam de existir.

## 15. O que a reforma não conserta

A manipulação guarda 16 dos 24 destravamentos. A reforma arruma a frente da cadeia e
corta o volume quase pela metade. Ela não muda a ordem de grandeza.

O orçamento continua sendo o limite duro. O treino tem 98 milhões de passos. O mjlab
gasta 2,95 bilhões só para andar.

## 16. Pendências

| item | estado |
|---|---|
| desenho | **fechado** — §10b e §9 registram a limpeza de 07/08 |
| implementação | não iniciada |
| `smoke.py` | não atualizado; o `sim_curriculo.py` sai inteiro |
| notebook da Kaggle | falta converter de Dataset para `git clone` |
| limpeza do log | não iniciada |
| escopo da postura | decidido — não existe; um termo, corpo todo (§6) |
| `exige_grasp` | decidido — fica ligado (§11) |
| `hold_still` | decidido — sai (§10) |

**Nenhuma decisão de desenho continua em aberto.** O que resta é implementação.

A bagunça do log tem causa conhecida. Em `rsl_rl/utils/logger.py:186-202`, o mesmo laço
escreve no TensorBoard e monta a linha do console, na ordem de inserção do dict. Não há
ordenação. A limpeza deve reagrupar a saída, e não cortar chaves.
