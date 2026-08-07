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

## 3. Três descalibrações, um sintoma

| termo | por que parecia errado | com comando ±1,0 |
|---|---|---|
| `track_linear` std 0,5 | span de 4% a 0,1 m/s | span de 98% |
| `foot_clearance` −2,0 | o passo custa 2× o ganho | proporção do mjlab volta |
| `action_rate` −0,25 | reflexo caro | já corrigido para −0,10 |

A correção do `action_rate` foi confirmada por medição em 06/08. O portão do push fechou
em 0,9449, com um empurrão 2,4 vezes mais forte que o anterior.

Restaurar a faixa de comando é uma mudança. Afinar os três termos seriam três mudanças,
com uma run cada.

## 4. A estrutura nova

Três grupos, cinco tarefas.

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

Todas as tarefas usam o mesmo código de locomoção. Manipulação é locomoção com comando
fixado em zero.

Cadeia de dependência: `locomover` nasce aberta. Dela saem `pegar` e `reorientar`, que
são irmãos. Do `pegar` sai `locomover_carregando`. Dele sai `botar`.

| origem | destravamentos |
|---|---|
| `locomover` | 2 |
| `locomover_carregando` | 6 |
| `pegar` | 10 |
| `botar` | 10 |
| `reorientar` | 10 |
| aberturas de tarefa | 4 |
| **total** | **42** |

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

## 6. A postura

O `variable_posture` escolhe o `std` pelo comando, não pela tarefa.

| junta | parado | andando | correndo |
|---|---|---|---|
| joelho | 0,05 | 0,35 | 0,60 |
| ombro | 0,05 | 0,15 | 0,50 |

A fusão obriga essa troca. Depois dela, o comando varia dentro do episódio. Um gate por
tarefa não consegue mais expressar o regime.

Manipulação não recebe termo de postura. O `pegar` já saiu do gate em 03/08, porque o
`std` 0,5 punia o agachamento. O `std` 0,05 é vinte vezes mais apertado.

O código passa de quatro termos de postura para dois.

| tarefa | escopo |
|---|---|
| `locomover` | corpo todo |
| `locomover_carregando` | pernas e cintura |

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

## 9. O que sai do código

| arquivo | remoções |
|---|---|
| `commands.py` | `DesiredTwistCommand` inteiro |
| `tasks.py` | `distancia_andar`, `rumo`, `push`; tarefas `PARADO`, `ANDAR`, `PARADO_CAIXA`, `ANDAR_CAIXA` |
| `metrics.py` | `chegou_andar`, `alinhado`, `chegou`, `sustenta_andar_s`, `de_pé` na locomoção |
| `curriculum.py` | célula de push e as funções dela |
| `env.py` | dois dos quatro termos de postura |
| `knobs.py` | `v_max`, `v_max_carga_cheia`, `a_max`, `w_max`, `alpha_max`, `d_morto_andar`, `morto_angular_rad`, `andar_raio`, `andar_raio_chega`, `heading_gain` |

O `alpha_max` estava rotulado "PALPITE A VALIDAR". Ele deixa de existir.

## 10. O que permanece

- `afasta_cena`. A causa muda, mas o robô ainda esbarra na prateleira.
- As três guardas: `caixa_quieta` no `botar`, `desvio_xy` e `apoiada` no `reorientar`.
- `exige_grasp` no `locomover_carregando`.
- A instrumentação `contrib/<tarefa>/<termo>`. Ela achou as três descalibrações.
- A observação continua com 154 números. O checkpoint atual carrega.

## 11. O custo escondido

Os dois portões de teste quebram.

| arquivo | acoplamento |
|---|---|
| `smoke.py` | 37 usos de `NUM_TASKS`, 28 de `PARADO`, 27 de `twist`, 22 de `ANDAR` |
| `sim_curriculo.py` | a premissa inteira é "o `andar` só abre com push completo" |

## 12. O que a reforma não conserta

A manipulação guarda 30 dos 42 destravamentos. A reforma arruma a frente da cadeia. Ela
não reduz o volume.

O orçamento continua sendo o limite duro. O treino tem 98 milhões de passos. O mjlab
gasta 2,95 bilhões só para andar.

## 13. Pendências

| item | estado |
|---|---|
| implementação | não iniciada |
| `smoke.py` e `sim_curriculo.py` | não atualizados |
| notebook da Kaggle | falta converter de Dataset para `git clone` |
| limpeza do log | não iniciada |

A bagunça do log tem causa conhecida. Em `rsl_rl/utils/logger.py:186-202`, o mesmo laço
escreve no TensorBoard e monta a linha do console, na ordem de inserção do dict. Não há
ordenação. A limpeza deve reagrupar a saída, e não cortar chaves.
