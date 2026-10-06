---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o velocidade_por_regime a −15 cobra ~0 de uma marcha treinada (0,0003/s a 1 m/s), mas custa −14,8/s com o ruído do começo do treino (σ 0,96)"
cluster: c3
---
MEDIDO 22/09 (`registra_juntas`, velocidade por diferença finita a 50 Hz, peso −15):
marcha do `model_22300` custa 0,0000/s a 0,5 m/s, 0,0000/s a 0,8 e 0,0003/s a 1,0 na
tabela de ANDAR. Na tabela de PARADO a mesma marcha custaria 1,2 a 4,2/s. Env de
locomoção não cai nela: `_elo` ANDAR e `_espera` zero (`comando.py:1283`).

MEDIDO no começo (zero05, 22/09): com σ 0,96 o freio custa −14,8/s, 9× todo o ganho;
com σ 0,45, −0,67/s. Ele não taxa a marcha pronta, mas comprime o ruído da exploração
(ver g1-limpo-std-cai-cedo). O bloco 1 andou sem freio.

**Why:** o freio a −15 cobra a exploração que forma a marcha.
**How to apply:** o −2 deixava o robô rápido demais. IMPLEMENTADO 23/09, smoke pendente: soma das dobradiças, piso −6/29 por junta, preço ×1,5 por degrau fora do ANDAR, degrau por env atrás do `nivel`. A `zero07` testa do zero.
**Detalhe:** `g1_training/docs/planos/2026-09-23-freio-em-curriculo-pelo-nivel.md`

Relacionado: g1-limpo-velocidade-de-junta-medida, g1-limpo-bloco1-marcha-formou
