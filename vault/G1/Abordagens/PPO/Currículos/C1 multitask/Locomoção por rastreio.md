---
tipo: abordagem
pai: "[[C1 multitask]]"
estado: validada
verificado: 2026-10-06
periodo: 07–11/08/26
resumo: "Andar como rastreio de velocidade funcionou; a régua angular sem recompensa travou o currículo 4931 iterações."
---
## Hipótese
Andar e ficar parado são a mesma tarefa: rastreio de velocidade com o `UniformVelocityCommand` do fabricante.

## Método
Reforma de `EXPERIMENTO.md` §9–10b: 7 tarefas → 5; o `DesiredTwistCommand` próprio sai inteiro; entra o comando do fabricante com 3 acréscimos (zero na manipulação, teto por env, giro parado com |wz| ≥ 0,15). Currículo em 4 camadas com junção AND: `locomover → (pegar E reorientar) → locomover_carregando → botar`. Peso vira DR em 2 níveis. Bug herdado: `_passos_por_iter` era `num_envs × 24`, 4096× grande.

## Resultado
Bloco 1, 7300 iterações: episódio 1000/1000, `fell_over` 0,0, erro linear 0,197 (tolerância 0,25). Mas o currículo disparou UM evento (it 2368) e travou 4931 iterações: o critério exigia erro angular ≤ 0,35 e nenhum termo pagava para reduzi-lo; a política convergiu em 0,64. O piso real era 0,62–0,73 (oscilação de marcha + push dentro da integral); `tol_w` → 0,70 deu `sucesso` 0 → 0,2884.

## Por que parou
Locomoção resolvida; o gargalo passou ao `pegar`.

## O que herda
- Toda régua de currículo precisa de um termo de recompensa correspondente.
- `tol_w` 0,35 nem testava rastreio angular: um robô parado marca ~0,25 e passa.
- `action_rate_l2` −0,25 travava o portão parado → andar; −0,10 (valor do mjlab) abriu o `andar` na it 421 e as quedas caíram de 30% para 1,8%.
- A faixa de comando era a causa raiz: `v_alvo *= cos(erro_rumo)` fechava um ciclo (sem girar não há comando, sem comando não anda). O mjlab sorteia `lin_vel_x` sempre.
- `std` 0,5 do kernel dá span de 98% a 1,0 m/s e 4,2% a 0,104 m/s.
