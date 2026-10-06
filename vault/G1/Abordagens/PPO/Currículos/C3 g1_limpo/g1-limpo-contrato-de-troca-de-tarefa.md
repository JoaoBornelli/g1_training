---
tipo: achado
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "A spec está em g1_training/docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md, na branch exp/g1-limpo-v2."
---
# g1-limpo-contrato-de-troca-de-tarefa (original)

Corpo original da memória `g1-limpo-contrato-de-troca-de-tarefa`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

A spec está em `g1_training/docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md`, na branch `exp/g1-limpo-v2`. Estado em 2026-09-03: **IMPLEMENTADA** na `exp/g1-limpo-v2`, sem push. Plano: `g1_training/docs/planos/2026-09-03-plano-contrato-de-troca-v2.md`. Portões: smoke 517 ok / 0 falhas (base 425), paridade 86 campos, inspetor 0 falhas nos 5 elos e 3 cadeias × 7 níveis. A renda do `BOTAR` ficou monótona, medida: pairar 10,7 < apoiada 13,0 < espera final 17,9 por segundo. O crítico tem 131 canais, e não 119: o crítico do fabricante já traz 12 canais de pé. Falta o push do dono e o treino do zero. A v14 troca `ANG` por `giro_b` (vetor de giro, 3 canais; ator 114, crítico 119) e fecha o contrato do `REORIENTAR`: quatro primitivas de 90°, direção horizontal, termina largando a caixa, mesa de 0,45 a 0,55 m independente do nível, carga variando como hoje; o treino dele fica fora da v2 (`voltas_max = 0`). A v13 (§6.6.1, §6.6.2, §13 sétima rodada) conserta três buracos que a revisão de 2026-09-03 achou: o `BOTAR` paga mais por não apoiar (g1-limpo-botar-paga-nao-botar); a espera final não tem incentivo para largar (`pose` com σ 0,05 vale zero com os braços fora); `caixa_z_min` fixo em 0,10 não acusa queda de caixa maior. O plano A do tamanho é first-party: `mjlab.envs.mdp.dr.geom_size` escreve `geom_size` e recalcula `rbound` e `aabb`; só falta um wrapper para o cubo.

A `exp/g1-limpo` é a referência que treinou a `bloco7`. Ela não muda mais.

A run `bloco8` roda a v2 desde 2026-09-03. Ela seguiu na Kaggle a partir da iteração 1050, com 4096 envs e 4,8 s por iteração. A locomoção formou na 324, o portão da forma abriu na 502, e a marcha MADUROU na 1503: por env de locomoção a eficiência mascarada é 0,745 contra o limiar de 0,50, a `razao_marcha` 0,55 e o voo 0,147 s, com queda em 1,9% das terminações. O portão está saturado no piso, portanto `alvo_loco_min = 0,30` é hoje a trava do orçamento de manipulação — o portão não é mais.

A manipulação nunca fecha o `PEGAR`, e REGREDIU entre a 1213 e a 1503 enquanto a marcha formava: `squeeze` 92% -> 84%, `staged` 1,48 -> 1,33, palmas em contato 0,97 -> 0,84, e `caixa_largada` DOBROU para 16,7% das terminações. O que ele FAZ, decomposto na 1503: preensao 0,843, descarga 0,933, rampa da pelve 0,718. Ele pega, ERGUE a caixa da mesa e fica de pé com ela. O que falta é subir os ~40 cm até `altura_carregar = 0,95`: a caixa está a 86% da distância inicial do alvo. `sucesso`, `sustentacao`, `load` e `largou` seguem em zero, portanto `CARREGAR` e `BOTAR` nunca receberam UM passo.

⚠ BURACO DE INCENTIVO MEDIDO na 1503, e ele viola a regra do dono. `precise_pos_sigma = 0,05 m` é a METADE de `tol_pos = 0,10 m`. No limiar de fecho do elo o termo paga `exp(−4) = 0,018`, portanto o peso 2,0 entrega 0,036/s no ponto exato do fecho. A 0,385 m ele lê 1e−26. Ele não é rampa e não é aceite: está morto. Consequência: dos 8,5 de peso que pagam o transporte, só o `staged` (3,0) tem derivada onde a caixa está. O `trazer` sozinho rende 4,96 por metro. E a pelve está em 0,665 m contra `pelve_alvo = 0,75`: o robô agacha e deixa +0,44/s na mesa. O `alinhado` NÃO é o problema — Δθ medido 8,2° contra tolerância de 25°. Verificado 2026-09-04.

⚠ `ELOS_SORTEAVEIS = (REORIENTAR, PEGAR)` sorteia uniforme, portanto METADE da manipulação nasce no `REORIENTAR` — que com `reorientar_inerte = True` é um temporizador de 0,3 s. O preço é o orçamento: as cadeias com `CARREGAR` ou `BOTAR` valem 5,6% da metade do `PEGAR`, ou 4% dos envs. Os canais que essa janela falseia estão em g1-leitura-log-e-kaggle.

O defeito: a observação publica a caixa a 5 m no `ANDAR`. A política pode ler "ando" da distância. Isso impede trocar de tarefa em voo.

Quatro mudanças: gate dos 7 canais de caixa quando o elo publicado é `ANDAR`; o `VALIDA` sai da observação e o `meia_aresta` (tamanho da caixa) entra — 112 canais, significado diferente; a espera publica `ANDAR` em vez de `PEGAR`; a cadeia 3 vira `(PEGAR, CARREGAR[v=0], BOTAR)`; depois do `BOTAR` uma espera final publica `ANDAR` até o fim do episódio, com `escapou` desarmado em `caixa_largada`. Mais um ramo de locomoção: `rel_turning_envs = 0,10`, com `|wz|` mínimo. Toda espera é `espera_s = (0,5, 1,5)`; um knob só. Nenhuma transição fica como aposta. DR de tamanho da caixa desde a FASE 1: `geom_size` + `rbound` + `aabb` por mundo, escritos no startup como o `foot_friction`. A caixa segue box primitivo. Fixo na run. Mesh (`VariantEntityCfg`) é plano B. A rede vê o tamanho. Cubo, 0,07 a 0,13 m, K = 8. `rel_turning_envs = 0,10`, `|wz| ≥ 0,2`.

Princípio do dono: treinar só andar, pegar, carregar e botar. O resto é composição do controlador via one-hot. `PEGAR → BOTAR` direto não é treinado.

Decisões: andar até a mesa não é deste modelo; `REORIENTAR` fica sorteável; DR e LIDAR são fases posteriores; o visualizador não é tocado.

**Why:** Um modelo só, troca por one-hot, sem handover. A espera publicando `ANDAR` treina `ANDAR(v=0) → PEGAR` em todo episódio de manipulação.
**How to apply:** O checkpoint fica inválido. Treino do zero na v2. Não toque em recompensa, cadeia, currículo ou reset.

Relacionado: g1-limpo-janela-de-espera, g1-limpo-desenho, g1-mjlab-velocity-anatomia
