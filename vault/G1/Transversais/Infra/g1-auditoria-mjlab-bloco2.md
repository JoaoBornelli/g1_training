---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
---
# g1-auditoria-mjlab-bloco2 (original)

Corpo original da memória `g1-auditoria-mjlab-bloco2`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Verificado 2026-08-06, branch `exp/curriculo-teste`. Auditoria que usou os treinos
oficiais do mjlab como referência levantou 4 itens de "linha dupla". **Dois caem.**

## A4 — pose inicial idêntica: REJEITADO

`reset_robot_joints` usa `position_range = (0.0, 0.0)`. O
`unitree_g1_flat_env_cfg` usa **exatamente o mesmo**. Não é divergência.

A divergência real está no `reset_base`: o fabricante sorteia yaw em ±π e x/y em
±0,5 m, o nosso é ±0,2 rad e ±10 cm. E o nosso está certo — **a observação do ator é
toda em frame de base, então yaw absoluto no mundo não é observável**. O fabricante
randomiza yaw porque o terreno dele é irregular e girar muda a rampa que se encara.
Em terreno plano é operação nula. Volta à mesa junto com terreno irregular.

## A5 — DR de `startup` para `reset`: REJEITADO

O fabricante usa `startup` nos três (`foot_friction`, `encoder_bias`, `base_com`), e
há razão física além do precedente: os quatro termos de DR randomizam propriedades
**do robô** — atrito da sola, viés de encoder, CoM da base, atrito da palma. Num robô
real esses valores são fixos. `startup` com N envs dá N robôs distintos, cada um
consistente consigo mesmo. Reamostrar por episódio simularia um encoder cujo viés muda
a cada 20 s.

`dr.geom_friction` aceita `env_ids`, então `reset` é mecanicamente possível se algum
dia se quiser DR mais agressiva. Não é defeito hoje.

## A6 — ruído de percepção: ADIADO (decisão do dono)

`box_pos_b` e `box_rot_b` têm ruído zero. É gap real de sim-to-real, mas a magnitude
**não é derivável do repositório**: todos os canais ruidosos do fabricante são
propriocepção (encoder ±0,010 rad contra kernel `std_walking` mediana 0,150 rad, razão
1/15; IMU ±0,05 em vetor unitário), e `box_pos_b` é percepção. Erro de visão é
propriedade do hardware pretendido. Aplicar a razão 1/15 fielmente daria 3 mm —
cosmético.

Adiado para a branch de sim-to-real porque **esta rodada mede o currículo**, e ruído
acrescenta variância ao `success_buf`, que é o sinal que o PLR usa para congelar
célula. Mesmo raciocínio que manteve a DR de atrito da S13 conservadora.

⚠️ Eu tinha sugerido aplicar só o `box_rot_b` (±0,05, derivado do `projected_gravity`)
por parecer inócuo. **Medido e descartado:** ±0,05 no rep 6D dá mediana **2,75°** e p95
**4,20°** de erro de orientação, contra tolerância de 10° e kernel fino de 5° do
`reorientar` — 55% do std fino na mediana. Criaria exatamente a variância que motivou
o adiamento.

A LATÊNCIA que o item pedia precisa de buffer de histórico de observação (o A7), então
não é linha dupla de qualquer jeito.

## A11 — `table_contact` cobrava o pé: REAL, corrigido (`7340f96`)

O sensor `body_table` casa `.*_collision` no robô, e **14 dos 31 geoms desse padrão são
de pé** (`left_foot1_collision` … `right_foot7_collision`). O comentário dele no
`base_env` diz "sem pads" — os pads de palma escapam, os pés não. Eram −1,5 por passo,
sem gate, nas 7 tarefas.

Contradizia a seção 5 do próprio `env.py`, onde o `feet_ground_contact` perdeu o
`secondary` **de propósito** para que pisar na prateleira conte como chão. E o dano
cresce para onde o currículo vai: o eixo `altura` desce de 0,55 m a 0,00 m e a
prateleira ocupa x de 0,20 a 0,80 — num nível baixo ela é um degrau na frente dos pés.

Corrigido com `exclude=(r"_foot\d+_collision",)` no `ContactMatch`
(`contact_sensor.py:79`), sem tocar em `g1_training/`, que a Lift partilha. Sensor de
31 para 17 geoms, tronco intacto.

Ver g1-multitask-orcamento-igual-por-tarefa, g1-multitask-spec-validacao-2026-08.
