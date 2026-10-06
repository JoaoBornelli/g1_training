---
tipo: abordagem
pai: "[[Tracking]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 06/10/26
resumo: "O BFM não abaixou até a pega na laje a 0,04 m (relato do dono, 06/10)."
cluster: bfm
---
## Hipótese
Desacoplar o 24800 do robô e pôr o BFM no meio: o 24800 (One-Hot e comandos) decide a pose, o BFM executa com o movimento humano do LaFAN e o 24800 corrige vendo o robô. O tracking pode passar onde o goal falhou.

## Método
Limite superior da malha fechada: o BFM recebe a trajetória inteira do 24800, com o futuro conhecido. Com o laço fechado só existe uma pose futura por passo, e uma janela de 1 faz o `tracking_inference` igual ao `goal_inference` (`fb/model.py:143-148`, a média de B nos próximos `seq_length` estados).

- `registra_juntas.py --topo 0.04` grava o rollout do 24800 com a laje a 0,04 m.
- `experimentos/bfm_tracking_24800/csv_para_npz.py` converte o CSV para `qpos`/`qvel`.
- `~/Documents/BFM-Zero/imitate_rl.py` com `ROLLOUT=<npz>` roda o tracking, sem caixa. Ambiente conda `bmf0`.

## Resultado
O BFM não abaixou até a pega na laje a 0,04 m (relato do dono, 06/10). A previsão era essa: abaixo de 0,35 m comandado, o Mapa comandado-alcançado dá queda ou robô deitado.

## Por que parou
A mesma causa da frente: o B só conhece o LaFAN, e o LaFAN não tem agachamento parado com as mãos perto do chão. O modo de inferência não muda o B. A malha fechada também não resolve, porque mover o alvo gira o `z` e não empurra o corpo (Goal servo).

## O que herda
- Malha fechada com uma pose por passo é goal por passo. O tracking só ganha quando o futuro é conhecido.
- O caminho que resta para essa pose é fine-tune do FB-CPR com rollouts do 24800 no buffer. Ele pede Isaac Lab e GPU.
