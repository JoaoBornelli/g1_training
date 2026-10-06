---
tipo: achado
pai: "[[C1 multitask]]"
estado: ativa
verificado: 2026-10-06
periodo: 18/08/26
---
## Fato
89% do tempo de iteração é física (`Collection` 4,558 s contra `Learning` 0,533 s); GPU melhor só ajuda por banda de memória.

## Evidência
T4 320 GB/s (base); L4 300 GB/s (não ganha); RTX 6000 Ada 960 GB/s (~3×); A100 1555 GB/s (~4,9×). Kaggle: 2× T4, 30 h/semana, 12 h/sessão.

## Implicação
Colab Pro com L4 não é upgrade. O orçamento real da tarefa é ~30 mil iterações; runs de 5 mil terminam onde o currículo começa.
