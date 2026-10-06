---
tipo: frente
pai: "[[Transversais]]"
estado: ativa
verificado: 2026-10-06
resumo: "Incentivo, não penalização; σ = distância inicial; gradiente no ponto de operação."
cluster: transversais
---
Incentivo, não penalização; σ = distância inicial; gradiente no ponto de operação.

## Filhos
<!-- filhos:inicio -->
- **g1-air-time-o-incentivo-desligado** (achado · ativa) — air_time é o único termo POSITIVO de marcha do mjlab e o cfg do G1 o zera — mas ele NÃO é o diferenciador: o módulo que andou também o tinha desligado
- **g1-postura-penhasco-std-standing** (achado · ativa) — O 4º regime da postura cobra 0,93/s por TERMINAR a tarefa: std_standing=0,05 zera `pose` com os braços erguidos
- **g1-principio-incentivo-nao-penalizacao** (achado · ativa) — principio de desenho de recompensa no g1_training, dado pelo usuario em 2026-08-25: toda acao precisa de INCENTIVO proprio, senao ficar parado custa menos que explorar; e nao se controla por PENALIZACAO — incentiva-se a acao correta em vez de punir a transgressao
- **g1-sigma-e-a-distancia-inicial** (achado · ativa) — O sigma do kernel de alcance tem de ser a distancia inicial daquele env, nao um numero fixo -- com sigma fixo pequeno a derivada no repouso e ZERO e o robo nao tem pista de onde ir
<!-- filhos:fim -->
