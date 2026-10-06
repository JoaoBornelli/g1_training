---
tipo: achado
pai: "[[Princípios de recompensa]]"
estado: ativa
verificado: 2026-10-06
resumo: "air_time é o único termo POSITIVO de marcha do mjlab e o cfg do G1 o zera — mas ele NÃO é o diferenciador: o módulo que andou também o tinha desligado"
---
`air_time` é o único termo positivo de marcha do mjlab e o único com derivada não-nula no repouso. O cfg do G1 o zera (`config/g1/env_cfgs.py:155`). Os quatro freios de passada valem zero com o pé plantado: eles moldam o passo depois que ele existe, e nenhum o cria.

REFUTADO como diferenciador (medido 25/08, montando os dois cfgs): o `g1_multitask`, que ANDOU, também tinha o `air_time` desligado. A fuga do ótimo local foi por exploração. Ligar o termo seria desviar da única configuração com marcha medida.

REFUTADO também `ang_vel_z = (0, 0)`: com guinada zerada a estátua tem erro zero e colhe o kernel cheio (2,000/s contra 1,711/s). Zerar a guinada torna o viés anti-marcha 8× maior. Fica `(−0,5; 0,5)`.

A bissecção real entre o módulo que andou e o que não andou são dois itens: o termo postural (função diferente, não só peso) e a terminação por queda (−200, ou −4,0 por evento ao dt de 0,02).

Relacionado: g1-principio-incentivo-nao-penalizacao, g1-limpo-desenho, g1-postura-penhasco-std-standing
