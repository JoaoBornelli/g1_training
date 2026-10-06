---
tipo: frente
pai: "[[Transversais]]"
estado: ativa
verificado: 2026-10-06
resumo: "Incentivo, não penalização; σ = distância inicial; gradiente no ponto de operação."
cluster: transversais
---
Regras de desenho de recompensa do projeto. As três primeiras estão no `CLAUDE.md` do repo.

1. **Recompense antes de penalizar.** Penalidade só diz o que não fazer; o espaço do não fazer é infinito. Dada pelo dono em 25/08.
2. **Toda ação precisa de gradiente no ponto de operação.** Um termo com derivada zero onde o robô está hoje é canal morto, por mais alto que seja o peso. Medir a derivada antes de propor o peso.
3. **Mais termo não traz o efeito.** Consertar a forma de um termo que existe antes de criar outro.
4. **σ = distância inicial daquele env**, não um número fixo: com σ fixo o kernel nasce em 1e−5 e não puxa.
5. **Gateie todo termo que paga por estar parado** (piso da estátua; std_standing que zera a pose; rastreio que paga comando nulo).
6. **Régua de currículo precisa de termo correspondente**; portão de habilidade mede a habilidade, não a sobrevivência.
7. **Penalidade sempre ligada é vencida** (o robô corre, trava a perna); amortecer por escala de ação, gate de fase ou escopo estreito.
8. `air_time` não é o diferenciador da marcha: o módulo que andou também o tinha desligado.

## Filhos
<!-- filhos:inicio -->
- **g1-air-time-o-incentivo-desligado** (achado · ativa) — air_time é o único termo POSITIVO de marcha do mjlab e o cfg do G1 o zera — mas ele NÃO é o diferenciador: o módulo que andou também o tinha desligado
- **g1-postura-penhasco-std-standing** (achado · ativa) — O 4º regime da postura cobra 0,93/s por TERMINAR a tarefa: std_standing=0,05 zera `pose` com os braços erguidos
- **g1-principio-incentivo-nao-penalizacao** (achado · ativa) — principio de desenho de recompensa no g1_training, dado pelo usuario em 2026-08-25: toda acao precisa de INCENTIVO proprio, senao ficar parado custa menos que explorar; e nao se controla por PENALIZACAO — incentiva-se a acao correta em vez de punir a transgressao
- **g1-sigma-e-a-distancia-inicial** (achado · ativa) — O sigma do kernel de alcance tem de ser a distancia inicial daquele env, nao um numero fixo -- com sigma fixo pequeno a derivada no repouso e ZERO e o robo nao tem pista de onde ir
<!-- filhos:fim -->
