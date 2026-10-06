---
tipo: estudo
pai: "[[Estudos]]"
estado: ativa
verificado: 2026-10-06
resumo: "Mímica humana + residual de interação: G1 real com 4,5 kg (o corpo encosta); contato pago sem teto. Serve como ideia de residual de força; não treinado aqui."
cluster: ppo
---
ResMimic ([arXiv 2510.05070](https://arxiv.org/abs/2510.05070)): G1 real carregando caixa de até 4,5 kg com uma política de mímica de movimento humano mais um residual treinado para a interação com o objeto.

- **O que mostra:** o G1 real carrega 4,5 kg, acima do limite de ~2,5 kg do punho, porque o corpo encosta na caixa (antebraço e tronco dividem a carga).
- **Recompensa de contato:** paga força maior sem teto. É o oposto do freio de aperto que testamos no 24350, e do teto de força do HDMI.
- **O que serve ao G1:** a arquitetura base de movimento humano + residual de força. É a parte "residual" da ideia SONIC registrada na inbox (06/10).
- **O que não serve direto:** depende de uma referência de mímica do movimento humano, que não temos para a pega no chão (mesma falta do LaFAN no BFM-Zero).

Estudado, não treinado. Fonte: revisão de literatura de 05/10.
