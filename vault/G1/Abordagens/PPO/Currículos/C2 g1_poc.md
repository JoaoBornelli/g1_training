---
tipo: abordagem
pai: "[[Currículos]]"
estado: superada
superada_por: C3 g1_limpo
verificado: 2026-10-06
periodo: 18/08–25/08/26
resumo: "Uma tarefa, dois comandos; o alvo da caixa define o comportamento. Pegou e reorientou SEM ANDAR; portão do balanço media sobrevivência. Superado pelo C3."
---
## Hipótese
Uma tarefa só, com dois comandos (`twist(3)` e `caixa_alvo(10)`), e o ALVO da caixa define o comportamento. "Pegar", "carregar", "botar" e "reorientar" não são tarefas: são posições de alvo. Sem one-hot, sem orçamento equalizado, sem gate por tarefa.

## Método
Módulo `g1_poc/` (43 commits, branch `exp/g1-poc`), especificação `ESPECIFICACAO-g1_poc.md`. Parte de `unitree_g1_flat_env_cfg()` e MUTA: a fundação de locomoção do fabricante entra sem reescrita. 20 termos de recompensa (13 do mjlab + `self_collisions` + 6 de tarefa) contra 43 knobs do C1; currículo de 90 linhas contra 786. Treino no Kaggle e no Colab, blocos 1 a 5.

Filhos: Desenho de uma tarefa, Bloco 1 a pega, Sonda do pegar, Máquina de elo, Bloco 2 colapso por giro, Paridade com o fabricante, Portão do balanço.

## Resultado
- Abriu `pegar` E `reorientar`: no `model_5100` a caixa sobe 16,5 cm estável, erro 0,011 contra raio 0,05, as 4 condições do fecho juntas em 63,4%. Reorientava bem e levava ao alvo.
- Primeiro sucesso do projeto: `episode_success` 0,0060 na it 3080, destravado pelo `unload`.
- Fez tudo isso SEM ANDAR: a marcha nunca formou (`peak_height_mean` 2,7–3 mm contra alvo de 0,10).
- Não aprendeu a GIRAR; o bloco 2 colapsou por giro.

## Por que parou
O portão do balanço entregava as transições da locomoção para a manipulação ao medir SOBREVIVÊNCIA, não marcha: um robô parado passava. O robô pegava a caixa no nível 1 e nunca aprendia a andar. Em 25/08 o trabalho foi refeito limpo no C3 g1_limpo, com locomoção primeiro e fatia 100%.

## O que herda
- O princípio validado: UM objetivo (botar a caixa em algum lugar com uma orientação) dispensa orçamento por tarefa, grafo de pais e congelamento.
- `squeeze` + `unload`: o gradiente tem de estar na coordenada do aperto, e pagar por descarregar.
- Terminar em vez de penalizar (trajetória inválida ACABA).
- Medir FORÇA em vez de recompensa (a sonda).
- Locomoção = a tarefa `velocity` do fabricante, sem mexer.
- Freios gateados por competência, não por passo global.
- Ler o log desfazendo `Episode_Reward / max_episode_length_s` e a diluição pelos resets (`leitura.py`).

## Filhos
<!-- filhos:inicio -->
- **Bloco 1 a pega** (abordagem · validada · 18–20/08/26) — squeeze saturou e o robô prensava a caixa; unload deu o 1º sucesso do projeto (it 3080) e a caixa subiu 16,5 cm no model_5100.
- **Bloco 2 colapso por giro** (abordagem · rejeitada · 21/08/26) — O robô aprendeu a girar e caía; o termo giro_indevido foi criado e apagado no mesmo dia.
- **Máquina de elo** (abordagem · validada · 20/08/26) — Cadeias de elos sem reset entre eles, σ por elo e load no botar; pegar e reorientar fecharam.
- **Paridade com o fabricante** (abordagem · validada · 21/08/26) — Locomoção igual ao velocity do fabricante, com comando zero no início; base_lin_vel volta ao ator.
- **Portão do balanço** (abordagem · rejeitada · 20–24/08/26) — O controlador de forma passava a fatia de andar à manipulação ao medir sobrevivência; o robô nunca andou.
- **Sonda do pegar** (abordagem · validada · 19–20/08/26) — Mede força em vez de recompensa no play; três erros de agregação em que um extremo mascarava a distribuição.
- **Desenho de uma tarefa** (decisao · superada · 18/08/26) — twist + caixa_alvo de 10 números com bit caixa_valida; terminar em vez de penalizar; currículo por env de 90 linhas.
- **Resample zerava o sucesso** (achado · ativa · 20/08/26) — O comando resampleava um passo antes do time_out e zerava o sucesso; explica a diferença treino × sonda.
- **g1-locomocao-primeiro-fatia-100** (achado · ativa) — por que o g1_multitask andou e o g1_poc nao: o multitask treinou locomocao SOZINHA com 100% dos dados (abertas=[LOCOMOVER]) por dezenas de milhares de iteracoes antes de abrir manipulacao; o poc entregou 70% das transicoes a manipulacao na it ~420
- **g1-poc-auditoria-incentivos** (achado · ativa) — Auditoria de incentivos do g1_poc (20/08): reaching sem σ variável = sorte do nível 3+; postura_ereta subcalibrada 3×; botar sem termo que pague por soltar
- **g1-poc-auditoria-travas** (achado · ativa) — Auditoria de travas g1_poc (20/08): nivel lia a forma do episódio SEGUINTE (p_up=0,7·p); Risco 3 invertido (pico a 0,30 m); T1 da caixa caindo REFUTADA por mim
- **g1-poc-orcamento-30k-iteracoes** (achado · ativa) — o PPO do fabricante do G1 declara max_iterations=30_000; runs de 3000-5000 do g1_poc são 10-17% do orçamento e terminam ANTES do 1º degrau do currículo de comando
- **g1-poc-recompensa-loco-prefere-parar** (achado · ativa) — g1_poc bloco4: em locomoção PURA a recompensa por passo caiu ~6x enquanto a duração subiu 50x e o pico do pé caiu 15x — alongar o episódio paga mais que andar
- **g1-track-lin-parado-colhe-27pct** (achado · ativa) — track_linear_velocity: um robô IMÓVEL colhe 0,275 (27%) do termo, só pela geometria da caixa de comando ±1,0 — ler o valor bruto como progresso é armadilha
<!-- filhos:fim -->
