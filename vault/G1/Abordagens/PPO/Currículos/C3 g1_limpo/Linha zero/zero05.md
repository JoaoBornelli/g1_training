---
tipo: abordagem
pai: "[[Linha zero]]"
estado: rejeitada
verificado: 2026-10-06
periodo: 22–23/09/26
resumo: "Sem NoBatente: o freio −15 custava −14,8/s com o std perto de 1 e comprimiu o std a 0,45; freio forte corta a exploração."
cluster: c3-zero
---
- **Lugar:** Colab (`[9b7dfba]`) e um notebook só-Kaggle novo (`[350ce87]`, dataset
  `g1-limpo-zero`), os dois com `RUN = zero05`, sem `NoBatente`, cintura (40; 0,10; 0,90).
- **O que mudou:** só a saída da terminação.
- **Medido:** com o std perto de 1 o freio a −15 custava **−14,8/s**, 64% dos custos ligados
  ao ruído; o std caiu a **0,45**, contra 0,61 na task de fábrica; com o robô de pé a taxa
  líquida ficava em ~0/s e a marcha atrasava `[65072eb] [plano 09-23 §1]`. Com σ 0,45 o freio
  custa −0,67/s (memória `g1-limpo-freio-nao-taxa-a-marcha`). A marcha treinada da linha bloco
  (`model_22300`) custa 0,0000/s a 0,5 e 0,8 m/s e 0,0003/s a 1,0 m/s na tabela de ANDAR
  (mesma memória, 22/09).
- **Por que terminou:** freio forte desde a it 0 corta a exploração; o freio vai a −6 só nos
  notebooks do zero `[65072eb] [8e405db]`. Última iteração não registrada.
- **Lição:** o freio a −15 não taxa a marcha pronta, mas cobra a exploração que forma a
  marcha.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
