---
tipo: abordagem
pai: "[[Tarefas isoladas]]"
estado: validada
verificado: 2026-10-06
periodo: 15–16/07/26
resumo: "Ficar de pé e recuperar de empurrão (Stand e Stand-Step); fundação para o Lift."
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
<!-- filhos:inicio -->
- **Muleta na mesa** (achado · ativa · 15/07/26) — Com a mesa ao alcance o robô se escora para satisfazer o upright; mesa fora de alcance no Stand.
- **Passo sem comando de twist** (achado · ativa · 15/07/26) — Termos de marcha do mjlab dependem do comando twist; andar não emerge de alvo longe.
- **Stand de graça no mjlab** (achado · ativa · 26/08/26) — No mjlab o G1 fica de pé sem aprender; de pé no play não prova aprendizado.
<!-- filhos:fim -->
