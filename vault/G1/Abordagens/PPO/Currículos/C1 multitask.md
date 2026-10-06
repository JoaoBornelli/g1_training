---
tipo: abordagem
pai: "[[Currículos]]"
estado: superada
superada_por: C2 g1_poc
verificado: 2026-10-06
periodo: 30/07–18/08/26
resumo: "Uma política com one-hot e orquestrador por célula. Andou e pegou SEPARADO; o pegar parou em 3 cm e o std fugiu. Superado pelo C2."
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
<!-- filhos:inicio -->
- **Bloco 4 do pegar** (abordagem · rejeitada · 10–18/08/26) — Rampa de altura, unload e sucesso denso: lift cru 0,06 → 0,39, mas ergue 3,0 cm de 3,5 exigidos.
- **Fase 1 com o residual** (abordagem · rejeitada · 30/07–03/08/26) — Env de 7 tarefas usado pelo residual e pela monolítica; consertos de cena e critério; hack de sentar.
- **Locomoção por rastreio** (abordagem · validada · 07–11/08/26) — Andar como rastreio de velocidade funcionou; a régua angular sem recompensa travou o currículo 4931 iterações.
- **Specs e auditoria** (abordagem · validada · 06/08/26) — 16 specs e auditoria por agentes: orçamento igual de 4,0 e 5 defeitos (lift sem erguer, botar pagando segurar).
- **Desenho do currículo** (decisao · superada · 30/07–18/08/26) — Orquestrador de células (tarefa, eixo), regra do evento a 0,90 e tabela de tarefas e eixos do C1.
- **Gargalo é banda** (achado · ativa · 18/08/26) — 89% do tempo é física; GPU só ajuda por banda (A100 é o salto; L4 não).
- **Std foge** (achado · ativa · 17–18/08/26) — O std sobe de 0,43 a 0,98 por entropia ∝ 1/σ e zera todas as tarefas acima de ~0,50.
- **g1-curriculo-congelamento-escala** (achado · ativa) — congelamento espúrio do currículo multitask some com num_envs; a célula de push media população mista e travava no limiar absoluto — corrigido 05/08 medindo só o `parado`
- **g1-multitask-caixa-fora-de-alcance** (achado · ativa) — Bug raiz do bloco 1: multitask recombinou box_xy=0.50 (o 'longe demais' que a Lift abandonou 16/07) com o jitter do config ativo — caixa fisicamente fora de alcance em ~81% dos episódios; explica pegar=0.0 e o teto 0.17 do reorientar
- **g1-multitask-orcamento-igual-por-tarefa** (achado · ativa) — O gate do g1_multitask carrega escala por tarefa (não máscara 0/1) para igualar o sinal de tarefa em 4.0 — alvo derivado do cfg do fabricante
- **g1-multitask-spec-validacao-2026-08** (achado · ativa) — SPEC S0-S15 do g1_multitask implementada 05/08 — as 4 decisões que divergiram da spec e os 2 números que ela não deu
<!-- filhos:fim -->
