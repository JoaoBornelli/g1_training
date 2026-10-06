---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "como a politica model_6999 do g1_limpo segura a caixa, medido em CPU em 2026-09-08 — altura 1,02 m, x 0,25, postura assimetrica na perna esquerda; e a sonda de nivel forcado que nao agiu"
cluster: c3
---
Medido 2026-09-08, `model_6999`, 64 envs, CPU, no hold com a caixa.

**Caixa na mão:** `caixa_b.x` p50 0,254 (0,228 a 0,272 conforme a meia-aresta); `y` centrado; **z p50 1,025 m** (p10 0,913; p90 1,076). O knob `altura_carregar = 0,95` ficava 7,5 cm ABAIXO de onde a caixa repousa. Era esse o desvio que fazia `perto` oscilar no limiar de 0,10 m. Corrigido para 1,02 e `peito_b.z = 0,222`.

**Postura de pé com a caixa:** `dq` p90 0,69 rad, dominado por `left_hip_pitch` (0,73) e `left_knee` (0,61) contra `right_hip_pitch` 0,49. Postura assimétrica: a perna esquerda dobrada. Ao alcançar baixo, `auto_colisao` chega a 121 N; `apoio_caixa` fica em 0.

**Cautela:** a sonda que forçou `nivel` 4, 5 e 6 deu `limpo_topo` idêntico (p50 0,43 m). O forçar de nível não agiu. `de_pe_tol_rad = 0,69` e `joelho_z_min = 0,05` valem para laje a ~0,43 m; re-medir no nível 6 real (0,04 m).

**Why:** números de knob têm de ser medidos, e a condição da medição tem de estar escrita.
**How to apply:** antes de apertar `de_pe_tol_rad`, rode a sonda com o nível de fato variando (confira `limpo_topo` por nível, filtrado ao PEGAR).

Relacionado: g1-limpo-plano-tres-tarefas, g1-limpo-botar-paga-nao-botar
