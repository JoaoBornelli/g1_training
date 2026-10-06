---
tipo: abordagem
pai: "[[Tarefas isoladas]]"
estado: validada
verificado: 2026-10-06
periodo: 15–16/07/26
---
## Hipótese
Uma política que só fica de pé e recupera de empurrão serve de fundação de equilíbrio e de warm-start para o Lift.

## Método
`skills/stand/`. Dois configs: `baseline` (stand calmo, só a fundação de equilíbrio, números do `src/G1_lifting_box` original) e `step_recovery` (Stand-Step: passo protetivo emergente sob empurrão forte). Checkpoint de referência: `reference_checkpoints/model_2000_stand_step.pt`.

## Resultado
Fica de pé e dá passo protetivo sob empurrão.

## Por que parou
Cumpriu o papel de fundação; o trabalho foi para o Lift.

## O que herda


## Filhos
```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT file.name
```
