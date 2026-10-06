---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "A escala da ação do wrist_pitch/yaw é 0,075 rad por unidade (6x menor que o braço): a política não explora nem aprende o giro do punho que mantém a caixa reta ao erguer"
---
`G1_ACTION_SCALE` do mjlab = 0,25 × torque_limite ÷ rigidez. Para o punho (motor 4010, 5 N·m) dá **0,075 rad** por unidade; para ombro, cotovelo e wrist_roll dá 0,439. Ruído de exploração no punho: 2° (std 0,49), contra 12° no cotovelo.

MEDIDO no `model_1600` da `zero11`, caixa no alto: a rede já grita no punho (`left_wrist_yaw` −5,25, `right_wrist_pitch` −3,04), e o punho fica 0,23 rad atrás do alvo (~3,9 dos 5 N·m). Sonda: `wrist_pitch` +0,45 rad endireita a caixa 24° em 0,5 s; −0,45 tomba +42°. Erguer pelo ombro/cotovelo tomba a caixa 1,6 rad por metro; só o punho desacopla.

**Why:** a escala foi calibrada para locomoção, onde o punho não importa.
**How to apply:** escala do punho = cotovelo REJEITADO pelo dono (29/09): não mexer no padrão do mjlab. O `pose` ×4 → ×1 no PEGAR_COM barateou o punho 4×.

Relacionado: g1-limpo-ang-cego-ao-tombo g1-limpo-punho-vai-ao-batente
