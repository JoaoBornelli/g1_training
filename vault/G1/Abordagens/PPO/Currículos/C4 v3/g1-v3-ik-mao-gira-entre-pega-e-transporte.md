---
tipo: achado
pai: "[[C4 v3]]"
estado: ativa
verificado: 2026-10-06
resumo: "MEDIDO 05/10: nas tabelas de IK da v3 a mão (eixo palma-a-palma) aponta −22° a −78° na pega e +13° no transporte; com a caixa presa e nivelada a IK é impossível de seguir; causa provável do tronco deitado e das mãos na horizontal"
---
MEDIDO 05/10: a inclinação dos dedos é −55° a −78° na pega a 0,04 m, −22° a −30° a 0,55 m, e +13° de pé com a caixa a 0,90 m (P4). Da pega ao transporte a IK gira a mão 35° a 91°. Com a caixa presa e nivelada, isso é impossível: o robô fixa a mão perto da horizontal; com o punho a 0,075 rad/unidade no `g1_limpo`, o antebraço não desce, e o tronco deita (relato do dono, 05/10).

**Why:** a IK resolveu cada pose sozinha, sem o ângulo da mão em relação à caixa.
**How to apply:** APROVADO (dono, 05/10): mão a −30° na pega, no CARREGAR (parado e andando) e no BOTAR (spec §8.1); 14 de 15 pegas; caixas de 14 a 26 cm testadas de pé. Agachar mais REJEITADO.

Relacionado: g1-limpo-escala-da-acao-do-punho g1-limpo-botar-deita-o-tronco g1-limpo-v3-alvos-macro
