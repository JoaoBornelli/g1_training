---
tipo: abordagem
pai: "[[Tarefas isoladas]]"
estado: validada
verificado: 2026-10-06
periodo: 15–24/07/26
---
## Hipótese
O robô parado prensa a caixa entre as mãos e a sustenta na altura comandada. Agachar EMERGE de ter que alcançar a caixa mais baixo; não é prescrito.

## Método
`skills/lift/` (`env.py`, `rewards.py`, `knobs.py`, 10 configs datados). Cadeia de recompensa MONOTÔNICA `reaching → grasp → lift → sustain_precise`, com anti-hacks (`back_penalty`, `table_contact`, `com_balance`) e fundação escopada (`upright` cheio, postura só em perna e cintura; braços livres). Sequência de fine-tunes, cada um com warm-start do anterior: ver os filhos.

## Resultado
Pegou, ergueu reto e sustentou. Currículo de altura de prateleira 0,55 → 0,45 → 0,35 → 0,25 com peso de 0,5 a 5 kg (a memória registra pega até prateleira de 0,05 m). Checkpoints: `model_6550_lift_altura_normal`, `model_7300` (2 alturas), `model_13200_lift_55-00`, `model_15700`.

## Por que parou
Skill isolada não encadeia: esqueceu o Stand e não anda. Superada pelos currículos. Defeito descoberto depois, já no multitask: o `progress` do `lift` vale sempre 1 quando o alvo é a própria caixa (ver Lift não mede erguer).

## O que herda
A cadeia monotônica anti-vale, o PLR de altura, o peso por força externa e o princípio anti-blanket seguiram para o `g1_multitask` e o `g1_limpo`.

## Filhos
```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT file.name
```
