---
tipo: achado
pai: "[[Infra]]"
estado: ativa
verificado: 2026-10-06
resumo: "CAUSA ACHADA 05/10: elliptic+impratio 10 divergiu 2× (15/07, 17/09) por 3 defeitos upstream — Hessiano do cone indefinido perto do ápice (mujoco_warp #1657, fix PR #1660 de 23/09, AUSENTE no 3.10.0.3 instalado), busca de linha esgotada com impratio 10 e ls 20 (mujoco #3628, PR aberto), warm-start do Warp propaga o NaN e o reset não limpa"
---
`elliptic` + `impratio=10` matou o treino em 15/07 e em 17/09 (it ~755). Causa, verificada em 05/10: (1) o Warp clampa `t³` separado de `t` no Hessiano do cone elíptico; com contato quase parado o Hessiano fica indefinido e o Cholesky dá NaN (mujoco_warp #1657; fix PR #1660 de 23/09; o 3.10.0.3 instalado ainda tem o clamp em `solver.py:2495`); (2) com impratio 10 e `ls_iterations=20` a busca de linha esgota num ponto de sela (mujoco #3628, PR #3651 aberto); (3) o Warp sempre parte de `qacc_warmstart`, e o reset do mjlab não zera esse campo: o NaN sobrevive ao reset.

**Why:** o piramidal não passa por (1) nem (2); o elíptico só volta com Warp pós-#1660 e `ls_iterations` > 20.
**How to apply:** manter piramidal; ligar o `nan_guard` do mjlab na v3; testar elíptico só com sonda curta.
**Detalhe:** g1_training/vault/G1/Abordagens/PPO/Currículos/C3 g1_limpo/g1-limpo-cone-eliptico-nan.md

Relacionado: mjlab-warp-contato-e-dr g1-limpo-inercia-real-da-caixa
