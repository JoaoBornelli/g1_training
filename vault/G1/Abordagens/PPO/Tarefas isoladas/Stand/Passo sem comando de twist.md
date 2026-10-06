---
tipo: achado
pai: "[[Stand]]"
estado: ativa
verificado: 2026-10-06
periodo: 15/07/26
resumo: "Termos de marcha do mjlab dependem do comando twist; andar não emerge de alvo longe."
cluster: tarefas-isoladas
---
## Fato
Os 4 termos de marcha do mjlab (`foot_clearance`, `air_time`, `foot_slip`, `soft_landing`) são gateados pelo comando `twist`: sem comando de velocidade eles dão KeyError ou ficam inertes.

## Evidência
`skills/stand/configs/step_recovery.py`: o passo vem do DRIVER (push forte + `upright` + penalidades anti-dinâmica afrouxadas), não dos termos de marcha.

## Implicação
Andar NÃO emerge de "alvo longe" como o agachar emergiu. Marcha é skill cíclica de exploração dura e vem de warm-start da tarefa `velocity`.
