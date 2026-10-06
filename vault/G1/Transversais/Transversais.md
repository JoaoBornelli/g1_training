---
tipo: frente
pai: "[[G1]]"
estado: ativa
verificado: 2026-10-06
resumo: "Conhecimento que várias frentes usam."
cluster: transversais
---
Conhecimento que mais de uma frente usa. Três ramos:

- **Princípios de recompensa:** regras de desenho dadas pelo dono ou medidas no treino (incentivo antes de penalização, σ = distância inicial, gradiente no ponto de operação, termos que pagam estar parado).
- **Infra:** fatos do mjlab, do MuJoCo Warp, do rsl_rl e da Kaggle/Colab que enganam a leitura do log ou quebram o treino.
- **Enunciado do comportamento:** o documento do dono com o comportamento esperado do G1; todo conserto cita a regra que serve.

Também moram aqui o caminho para o robô real e o artifact "Como o G1 aprende".

## Filhos
<!-- filhos:inicio -->
- **Infra** (frente · ativa) — mjlab, Warp, Kaggle/Colab, gargalo de banda.
- **Princípios de recompensa** (frente · ativa) — Incentivo, não penalização; σ = distância inicial; gradiente no ponto de operação.
- **Enunciado do comportamento** (decisao · ativa) — Enunciado do problema escrito pelo dono: o comportamento esperado do G1 em cada etapa (andar, pegar, carregar, botar). Todo conserto cita a regra que serve.
- **g1-limpo-artifact-como-o-g1-aprende** (achado · ativa) — o artifact público 'Como o G1 Aprende' — endereço, o fio condutor do depósito e o problema central em dois lados
- **g1-limpo-caminho-para-o-robo-real** (achado · ativa) — o que falta para o treino do g1_limpo rodar no G1 físico — lidar, mãos Dex3 e DR agressiva; anotado pelo dono em 2026-09-16
<!-- filhos:fim -->
