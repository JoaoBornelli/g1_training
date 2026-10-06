---
tipo: achado
pai: "[[C2 g1_poc]]"
estado: ativa
verificado: 2026-10-06
---
# g1-poc-orcamento-30k-iteracoes (original)

Corpo original da memória `g1-poc-orcamento-30k-iteracoes`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Verificado no código em 2026-08-24.

```
mjlab/tasks/velocity/config/g1/rl_cfg.py:45      max_iterations = 30_000
                                        :32      entropy_coef   = 0.01
                                        :35      learning_rate  = 1.0e-3
                                        :44      num_steps_per_env = 24
```

O `g1_poc` usa `unitree_g1_ppo_runner_cfg()` **sem mudança** (só o `experiment_name`),
portanto herda todos esses hiperparâmetros — inclusive a expectativa de 30 mil iterações.

## A consequência de escala

Com 4096 envs × 24 passos = 98.304 amostras/iteração:

| orçamento | iterações | passos |
|---|---:|---:|
| fabricante (declarado) | 30.000 | **2,95 G** |
| bloco5 (24/08) | 5.000 | 0,49 G |
| onde o bloco5 estava ao diagnosticar | 997 | 0,098 G = **3,3%** |

E os degraus do `commands_vel` do fabricante estão em **5000 e 10000 iterações**
(`velocity_env_cfg.py:403-405`). Ou seja: **uma run de 5000 termina exatamente onde o
currículo de comando do fabricante começa.** Runs de 3000 nunca veem degrau nenhum.

**Regra prática:** antes de declarar defeito de recompensa por "não aprendeu X", checar em
que fração de 30.000 a run está. Abaixo de ~15% a ausência de marcha não é evidência de
defeito.

## Correção de leitura do `peak_height_mean`

⚠ `peak_height_mean` é medido **no pouso** (`velocity/mdp/rewards.py:313`), portanto um
robô que DESPENCA registra valores grandes. No bloco5 ele valia 0,050 na it 60 **com
`fell_over` em ~500** e caiu para 0,0024 na it 997 com `fell_over` em 0,125.

Isso NÃO é "a política desaprendeu a levantar o pé" — é a transição de *esperneando* para
*de pé controlado*. Eu afirmei o contrário em 24/08 e retifiquei no mesmo dia. Nunca houve
marcha para desaprender.

## O harness do play está correto (não culpar o `--vx`)

`play.py:144-173` (`_ajusta_andar`) faz as cinco coisas necessárias para o `--vx` valer:
`auto_balanco=False`, `frac_locomocao=1.0`, `rel_standing=0.0`, `espera_s=(0,0)`,
`rel_heading=0.0` e `rel_forward=0.0`, além de pinar `lin_vel_x=(vx,vx)` com `lin_vel_y` e
`ang_vel_z` em zero. Testado por observação: com `--andar --vx 0.5` na it 997 o robô fica
**perfeitamente imóvel** — e como o play roda a ação MÉDIA (sem ruído), isso também
confirma que o "giro" de 0,75 rad/s visto no treino é **jitter do ruído de exploração**, e
não comportamento aprendido.

## Experimento de controle pré-registrado (não executado ainda)

Rodar a tarefa do fabricante sem modificação, mesma máquina e mesmo `num_envs`:
`Mjlab-Velocity-Flat-Unitree-G1`, ~1000 iterações. Comparar na MESMA iteração:

| resultado na it 1000 | veredito | ação |
|---|---|---|
| `peak_height ≲ 0,01` e `track_lin` norm ≈ 0,30 | é CEDO | subir `max_iterations` para ≥15000; não mexer em recompensa |
| `peak_height ≳ 0,03` | o poc tem DEFEITO | bissecção; 1º suspeito = intervalo do push |

Ganho colateral: hoje o projeto **não tem curva de referência**, e é por isso que cada
leitura vira debate.

Relacionado: g1-poc-portao-forma-mede-sobrevivencia,
g1-track-lin-parado-colhe-27pct, g1-poc-recompensa-loco-prefere-parar,
g1-mjlab-velocity-anatomia, g1-treino-gargalo-banda-nao-flops.
