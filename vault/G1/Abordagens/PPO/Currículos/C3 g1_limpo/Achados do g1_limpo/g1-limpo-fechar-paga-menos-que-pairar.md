---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "Fechar o PEGAR rende menos do que pairar no alvo (16,5 vs 25/s), e 'fechou mas não avançou' paga o dobro (32/s): explica os fechos que aparecem e somem"
cluster: c3
---
MEDIDO no `model_1600` da `zero11`, renda por segundo: pairar no alvo sem fechar 25,0; espera pós-fecho 23,2; CARREGAR 16,5; **PEGAR_COM+fechou 32,3** (congelado + os sete ao vivo: o avanço exige `perto`, e o mapa de estados não conhece `fechou`). O `renda_congelada` paga sem condição e cumulativo.

Consequência: o crítico aprende a evitar o fecho quando ele aparece; tombar a caixa é um jeito de não fechar. Padrão nas zero08 e zero11.

**Why:** o próximo estágio tem de pagar mais do que o anterior, senão o robô para antes.
**How to apply:** tirar a renda de ficar parado do PEGAR_COM (rastreio 1→0, `pose` 4→1): pairar cai a 18,6. Quem não avança no fim da espera vai à cauda CARREGAR. `renda_congelada` ×1,7 REJEITADO (29/09): sobe o prêmio de pressa.
**Detalhe:** `g1_training/docs/relatorios/2026-09-29-tombo-da-caixa-achados-e-correcoes.md`

Relacionado: g1-limpo-ang-cego-ao-tombo g1-limpo-renda-congelada-paga-10x-o-deficit g1-limpo-escala-da-acao-do-punho
