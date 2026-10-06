---
tipo: abordagem
pai: "[[Currículos]]"
estado: superada
superada_por: C2 g1_poc
verificado: 2026-10-06
periodo: 30/07–18/08/26
---
## Hipótese
Uma política só, condicionada a comando (one-hot da tarefa dentro do vetor de comando), aprende todas as tarefas da cadeia se um orquestrador abrir cada tarefa quando a anterior atinge competência.

## Método
Módulo `g1_multitask/` (63 commits), branch `exp/locomocao-velocidade`. Treino na Kaggle (2× T4, 4096 envs). Desenho completo no handoff `docs/handoff/2026-08-18-multitask-inventario.md` e no ADR `docs/adrs/0001-curriculo-do-pegar-rampa-ponte-e-bonus.md`.

Quatro fases, cada uma um filho: Fase 1 com o residual, Specs e auditoria, Locomoção por rastreio, Bloco 4 do pegar.

## Resultado
- Andou e resistiu a empurrão simulado: `locomover/perf_n0` 0,9143, `sucesso` 0 → 0,2884 depois do `tol_w` 0,70.
- O `pegar` ergueu a caixa só 3,0 cm dos 5,5 do nível 0 (`cond_fisica` 0,0000 exato).
- `reorientar` chegou a `perf` 0,158. `locomover_carregando` e `botar` nunca abriram.
- Não aprendeu a GIRAR.
- O `std` da política fugiu (0,43 → 0,98) e matou o currículo (`ALARME_estagnacao` com 7,9e8 transições).

Resumo do dono (25/08): anda E pega, mas SEPARADO — o encadeamento existe no currículo, não no resultado.

## Por que parou
O gargalo `pegar` (pai de `locomover_carregando` e `botar`) não fechou em 22 mil iterações, e a fuga do `std` era bloqueante. A maquinaria de currículo (células, 12–60 destravamentos, congelamento, orçamento por tarefa) cresceu mais que o resultado. O trabalho seguiu no C2 g1_poc com UM objetivo.

## O que herda
- Régua de currículo sem termo de recompensa correspondente é parede (ver Locomoção por rastreio).
- Separar alvo de NAVEGAÇÃO de alvo de TAREFA.
- Âncora em z de MUNDO, não no frame da base (agachar encurtava o percurso).
- `unload` como ponte contínua do aperto; `sucesso_denso` contínuo atravessa o desconto de 2 s.
- `parado` e `andar` são a MESMA tarefa no mjlab (rastreio de velocidade com `rel_standing_envs`).
- Ler o log desfazendo o orçamento: `cru = contrib / peso efetivo`.
- Piso plano é premissa do projeto (decisão do dono).

## Filhos
```dataview
TABLE tipo, estado, verificado FROM "" WHERE contains(pai, this.file.link) AND estado != "superada" SORT file.name
```
