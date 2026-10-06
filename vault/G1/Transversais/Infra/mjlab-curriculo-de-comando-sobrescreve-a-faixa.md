---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
resumo: "o commands_vel do mjlab escreve cfg.ranges a cada tique, entao ler a faixa do twist antes do launch mostra um numero que o robo nao recebe no passo 0"
cluster: transversais
---
`mdp.commands_vel` (`velocity/mdp/curriculums.py:99-108`) **escreve**
`cfg.ranges.lin_vel_x/lin_vel_y/ang_vel_z` a cada tique, varrendo a tabela de estágios.
Ler `cfg.env.commands["twist"].ranges` antes do `launch_training` mostra a faixa BASE,
que o estágio 0 sobrescreve no primeiro tique. Num treino do zero isso engana.

A tabela do fabricante (`velocity_env_cfg.py:405-409`), em `common_step_counter`:

| estágio | passo | `lin_vel_x` | `ang_vel_z` |
|---|---|---|---|
| 0 | 0 | (−1,0; 1,0) | **(−0,5; 0,5)** |
| 1 | 5000×24 | (−1,5; 2,0) | (−0,7; 0,7) — no g1_limpo vira ±1,6 |
| 2 | 10000×24 | (−2,0; 3,0) | — |

`lin_vel_y` não aparece em estágio nenhum: vem da base e nunca é tocado.

**Why:** com 24 passos por env, `5000×24` é a **iteração 5000** — uma sessão Kaggle do
zero (~5478 its) CRUZA o estágio 1, e a queda de recompensa ali não é regressão.

Relacionado: g1-mjlab-velocity-anatomia, g1-limpo-giro-canal-e-envelope
