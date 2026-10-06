---
tipo: achado
pai: "[[Controle determinístico]]"
estado: ativa
verificado: 2026-10-06
resumo: "G1 em agosto/26 (relato do usuário 2026-08-25): treino preliminar dá andar E pegar a caixa SEPARADAMENTE, não encadeado; frente nova = abordagens determinísticas de controle; orçamento de GPU levantado no RunPods"
---
Estado da frente de robótica no fechamento de agosto/26, relatado pelo usuário em 2026-08-25:

- **Treino preliminar entrega as duas competências, mas SEPARADAMENTE:** o robô anda, e o robô pega
  a caixa — não a cadeia completa. Corrige qualquer leitura de que `pegar → carregar → depositar`
  já roda ponta a ponta; o encadeamento existe no currículo, não no resultado.
- **Frente nova: abordagens determinísticas para o controle**, sob investigação em paralelo ao RL.
  Encaixa no que já foi medido para o braço (IK numérica + servo de posição com os ganhos oficiais —
  g1-arm-ik-pd-control) e é a rota natural depois das armadilhas de recompensa de agosto.
- **Orçamento de GPU levantado no RunPods** para contratar treino de modelos de inferência em cloud.
  Responde ao gargalo já medido: 89% do tempo de treino é simulação física, banda de memória manda,
  placa intermediária não muda o jogo (g1-treino-gargalo-banda-nao-flops), e o orçamento real da
  tarefa é 30.000 iterações — runs de 5.000 terminam onde o currículo começa
  (g1-poc-orcamento-30k-iteracoes).
- **Possível (a confirmar pelo usuário):** conversa preliminar com **Douglas** sobre o que é
  necessário para usar os robôs da UDESC — ajuda com sim-to-real e como testar. Era o follow-up
  agendado para o início de agosto em okr-julho-2026-draft; não confirmado.

**Why:** "anda e pega" é um marco vendável, mas dizê-lo como cadeia completa seria falso, e a
diferença é justamente o que os próximos runs precisam provar. E a decisão de GPU deixou de ser
"decidir se vale" para "aprovar o orçamento levantado".

**How to apply:** ao reportar, separar as duas competências e nomear o encadeamento como o próximo
alvo. O pedido ao líder é aprovação do orçamento RunPods para um run no orçamento real da tarefa —
ou redução explícita do escopo da demonstração para o que cabe em 5.000 iterações.

Relaciona: g1-locomocao-primeiro-fatia-100, g1-modulo-novo-heranca-comportamental,
g1-poc-portao-forma-mede-sobrevivencia, g1-poc-recompensa-loco-prefere-parar,
g1-lift-progress-alvo-e-a-caixa, unitree-g1-sim-mujoco-not-isaac.
