---
tipo: abordagem
pai: "[[Currículos]]"
estado: superada
superada_por: C3 g1_limpo
verificado: 2026-10-06
periodo: 18/08–25/08/26
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
```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT file.name
```
