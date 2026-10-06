---
tipo: frente
pai: "[[PPO]]"
estado: ativa
verificado: 2026-10-06
resumo: "Estudos em paralelo ao treino."
cluster: ppo
---
Trabalhos e métodos estudados em paralelo ao treino, sem treino próprio aqui. Cada estudo registra o que o trabalho faz, o que serve ao G1 e o que não serve.

Origem principal: a revisão de literatura de 05/10 sobre aperto e deslize em humanoide com caixa (`docs/relatorios/2026-10-05-dr-de-atrito-pega-e-transferencia.md` §7), feita para decidir o preço do aperto e a DR de atrito. Ideia aberta na inbox (06/10): SONIC (NVlabs GR00T-WholeBodyControl) como executor de movimento humano, com o PPO emitindo alvos esparsos e um residual estilo ResMimic para a força.

## Filhos
<!-- filhos:inicio -->
- **Literatura de aperto em humanoide** (estudo · ativa · 05/10/26) — Revisão de 05/10: ninguém dosa o aperto sem tátil; três mecanismos (teto de força no contato, mola virtual no PD, torque L2 pequeno); todo G1 real usa mão de borracha.
- **ResMimic** (estudo · ativa) — Mímica humana + residual de interação: G1 real com 4,5 kg (o corpo encosta); contato pago sem teto. Serve como ideia de residual de força; não treinado aqui.
<!-- filhos:fim -->
