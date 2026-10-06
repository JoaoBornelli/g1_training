---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o limite de junta do MJCF é restrição mole e o ankle_roll sai 50% do curso; a rampa que cobra tem de acabar ALÉM de onde a junta vive"
cluster: c3
---
MEDIDO no MuJoCo clássico (`carrega085.csv`, 1250 passos): o `ankle_roll` chega a
**150% do curso** do MJCF e o `ankle_pitch` a 120%, empurrados pelo contato com o
chão. O `solimp` default (`[0.9, 0.95, 0.001]`) cede na proporção da força. A política usa
curso que o G1 real não tem. É a limitação 11.

O `limite_de_junta` cobra o efeito: rampa exponencial sobre a fração do curso, com `k`,
teto e limiar POR FAMÍLIA. O dono recusou mexer no solver.

**Why:** o teto tem de acabar ALÉM de onde a junta vive. Junta no teto tem derivada ZERO, e o termo vira imposto fixo. Duas tabelas reprovaram assim: 95,7% e 59,1% dos passos no teto.
**How to apply:** no ZERO a rampa é 0,85 a 1,00; a cintura é 0,90 a 1,00, ainda não treinada — ver g1-limpo-rampa-estreita-nao-ensina. A tabela larga conserta política viciada.

Relacionado: g1-limpo-faixa-de-pose, g1-limpo-inercia-real-da-caixa
