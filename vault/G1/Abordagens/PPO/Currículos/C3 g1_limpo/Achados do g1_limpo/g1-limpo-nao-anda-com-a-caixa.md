---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "O robo nao anda carregando a caixa por INCENTIVO, nao por capacidade: andar acrescenta 5,2% a uma renda que ja e 94% ligada a caixa"
---
MEDIDO 2026-09-10, `model_10200`, cauda `CARREGAR` com a caixa, n = 16 663.

**As pernas nao tentam.** Com o mesmo comando, `|qd|` p50 e 0,039 contra 0,578 na locomocao pura — 15×. **97,2% dos passos tem os dois pes no chao**, contra 30%, e nao ha alternancia de passo. A pose e um agachamento de suporte travado.

**A conta que a politica faz.** Parado, a cauda paga **28,64/s**, e **93,9% e renda ligada a caixa** (congelada 13,49 + viva 13,40). Andar bem a 1 m/s acrescenta **+1,49/s = +5,2%**. Break-even: se andar aumenta o risco de terminar em mais de **5,2 pontos**, parar e a resposta certa.

Na locomocao pura `track_lin` e 34% da renda; na cauda, andar vale 5%. Exposicao (11,3%) nao explica.

⚠ **Refinado 09-10:** a renda age pelo canal de TERMINACAO, nao por preferencia de passo — ela e constante na cauda (σ temporal 1e-6), e largar a caixa custa 62,3 unidades descontadas. Ver g1-limpo-cauda-carregar-piso-sem-derivada.

Relacionado: g1-limpo-cauda-carregar-piso-sem-derivada, g1-limpo-piso-da-estatua-travava-a-pega, g1-limpo-plano-tres-tarefas, g1-poc-recompensa-loco-prefere-parar
