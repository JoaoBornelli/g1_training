# v3: sanidade da spec e plano de treino

Data: 2026-10-06. Leitores: agentes que vão implementar a v3 (pacote `g1_v3/`).

Este documento junta a spec da v3 com os achados de uma revisão de sanidade. Ele não substitui a spec. Onde este documento e a spec diferem, a spec vale até o dono decidir os pontos da seção 6.

Siglas das fontes:

| Sigla | Arquivo |
|---|---|
| E | `specs/g1-limpo-comportamento.md` (enunciado) |
| CV | `specs/g1-v3-comportamento.md` |
| S | `specs/g1-v3.md` |
| DEC | `vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md` |
| REV | `docs/relatorios/2026-10-05-revisao-spec-v3.md` |
| DR | `docs/relatorios/2026-10-05-dr-para-a-v3.md` |
| ZERO17 | `docs/relatorios/2026-10-01-aprendizados-zero17-para-v3.md` |
| AVAL | `docs/relatorios/2026-10-02-avaliacao-v3.md` |
| INV | `docs/relatorios/2026-10-01-inventario-v3.md` |
| PERC | `docs/planos/2026-10-02-percepcao-lidar-na-v3.md` |
| INBOX | `vault/_sistema/inbox.md` |
| LS | `/home/joaobornelli/orca/workspaces/g1_training/lidar_sim` |
| FASE1 | `LS/specs/2026-10-05-autonomo-fase1.md` |
| SIM2REAL | `LS/docs/planos/2026-10-05-autonomo-do-sim-ao-real.md` |

Ordem de autoridade (S:7-15): enunciado > CV > DEC > inventário > avaliação > DR. Atenção: o CV ainda tem regras antigas (problema 4.2.4). Até o CV ser corrigido, siga a S nos pontos listados lá.

---

## 1. Objetivo e critério de sucesso

### 1.1 Objetivo

Um treino só (v3) que aprende as quatro tarefas, ANDAR, PEGAR, CARREGAR e BOTAR, com um modelo só (requisito duro, DEC:29). A política tem de ser generalista o bastante para entrar no Unitree G1 real com pouco esforço. O robô vai manipular caixas num depósito comercial.

Fatos de contorno:

- Pacote novo `g1_v3/`, treinado do zero. mjlab primeiro, depois Isaac Lab. A lógica fica em funções puras (alvos, potencial, balanceador, estimador) para rodar igual nos simuladores e no robô (S:34-37, S:89-93).
- Orçamento: 30 000 iterações (S:38). Meta de código: ~5 000 linhas, contra 17 893 do g1_limpo (S:23-39, S:649-650).
- Diagnóstico que motiva a v3: no g1_limpo a recompensa faz dois trabalhos, sequência e qualidade; 15 de 26 hacks do catálogo são renda por estado (DEC:20-25). A v3 separa: o controlador manda a sequência (one-hot, comando, alvos), e a recompensa só mede o avanço até o alvo corrente.
- A laje é o único suporte no treino (CV:33, S:61-75). REORIENTAR, LEVANTAR, prateleira e vizinhas ficam para depois (CV:394-489).
- Nenhum código da v3 existe ainda (S:3-4).

### 1.2 Critério de sucesso

O treino passa quando os aceites da S:513-594 passam. Resumo:

| Aceite | Critério |
|---|---|
| Locomoção | razao_marcha ≥ 0,50 em ~1000 its; nenhuma terminação nova > 5% |
| Espera sem agachar | joelho ~0,16 rad antes da abertura do PEGAR |
| Pega reta | inclinação da caixa ≤ limite do degrau de P3 a P4 |
| BOTAR de um comando | um comando BOTAR leva até B4 |
| Caixa perdida | comportamento da chave CAIXA_PERDIDA_VIRA_ANDAR |
| Frear com a caixa | CARREGAR 0,5 m/s → parado |
| Comando baixo | vx 0,10 por 5 s e wz 0,15 no lugar por 5 s: razão ≥ 0,50, giro desloca < 0,10 m |
| Passo com a caixa | 0,25 e 0,5 m/s |
| Rumo | < 3° com k=2 do driver |
| Quatro do registra_juntas (9 de 10 rodadas) | sem batente (> 95% do curso); BOTAR na laje 0,04 m com 5 kg (caixa a ≤ 0,10 m, ≤ 15°, chegada ≤ 0,99 m/s); giro de 180° ± 15° com caixa a 0,8 rad/s por 4 s; empurrão 0,5 m/s em 4 situações, caixa volta a ≤ 15° em 1 s |
| Refino | sucesso não fica abaixo do portão por mais de 300 its |

O ciclo do galpão é teste do dono depois do treino, não aceite (S:588-589).

Esta revisão propõe aceites a mais (detalhe na seção 4):

- μ 0,3 × 5 kg, CARREGAR de 7 s a 0,5 m/s, caixa não desce > 2 cm (4.3.1).
- Pega com erro de chegada 0,15 m e 20° (4.3.8).
- Atuador do braço saturado < 5% dos passos; |τ| médio em 1 s ≤ 60% do limite no CARREGAR com 5 kg (4.3.7).
- Velocidade da palma p99 ≤ 1,0 m/s; ≤ 0,5 m/s com caixa (4.4.9).
- razao_marcha ≥ 0,50 com atraso de 1 passo (4.3.6).
- Frear com a caixa estendido até B4 (4.4.11).

### 1.3 O que a v3 entrega ao controlador

O controlador (LS/autonomo) espera (SIM2REAL:5):

- Toda tarefa começa e termina de pé.
- PEGAR termina em P4: caixa no peito, nivelada, robô de pé.
- BOTAR termina em B4: robô de pé, mãos fora da caixa, caixa assentada e parada. A etapa SOLTAR do controlador sai.
- A tarefa de manipulação garante: sem bater, sem impacto, sem esmagar.

---

## 2. Fluxo de aprendizado proposto e ordem das tarefas

### 2.1 Visão geral

```
Parte 0  (calibração e sondas, sem treino longo)
   │
Fase 0   100% ANDAR  ──► portão do ANDAR (reto e parado ≥ 0,75 no balanceador)
   │
Fase 1   fatias por env: ANDAR | B | C, balanceador de tarefas
   │        B abre: reto e parado do ANDAR no portão
   │        C abre: PEGAR no portão do 1º nível
   │        dentro de cada tarefa: balanceador de subtarefas/níveis
   │
Refino   por tarefa: degraus 0..5 (contato, nivelamento, chegada,
         batente, hip_yaw, estimador, DR); sobe ≥ 0,75 / desce < 0,50 (proposta)
```

Ordem do treino: locomoção → B → C (DEC:349). Partes 0 a 4 da S em sequência, sem paralelismo (S:53, S:596-642).

### 2.2 Parte 0 antes do primeiro treino

A Parte 0 (S:601-621) mede e calibra sem treinar. Itens da spec mais os que esta revisão pede:

1. Potencial em trajetória roteirizada: pairar rende zero (S:603-604).
2. Derivada no ponto de partida de cada termo e de cada fator do Φ. Acrescentar: dk_força/dF em F = 230 N (4.2.2); derivada de g_invariantes a θ_lim + 5° (4.3.2).
3. Escada de custos: queda > perda > volta à pose de pé; queda > mergulho a caixa longe (S:347-348). Acrescentar as duas desigualdades de 4.3.4 (desistir no BOTAR; "pega e larga").
4. Balanceadores sintéticos.
5. a 9. Itens da S (comando baixo, passo mínimo L_min, faixa da IK, CoM).
10. IK dos alvos P0, P1, P3, B1, B2, B3 com mão −30°; a pega na laje 0,04 m com caixa 14 cm que não fecha (S:621). Acrescentar: tabelas até 0,70 m se o dono aceitar 4.4.12.

Acrescentar à Parte 0, antes da Parte 1:

- Contrato de atuação único e teste de paridade de kp/kd (4.2.3).
- Concordância do índice do alvo observável com o índice verdadeiro (4.2.1).
- Medida do tombo da caixa pequena com mão −30° (4.3.9).

### 2.3 Fase 0: só ANDAR

- 100% dos envs em ANDAR (CV:39-44).
- Subtarefas na ordem: reto 4 direções + parado → comando baixo (0,05-0,25) → giro andando → giro no lugar (S:371-376).
- Níveis de velocidade: frente 0,25 → 1,5 m/s em passos de 0,25; trás e lado até 1,0; giro 0,25 → 1,0 rad/s. Nível k sorteia entre 0,25 e o teto. TERRENO entra como nível a mais (06/10).
- Mapa de elevação ligado desde a it 0 (S:75-87).
- Saída da fase 0: hoje há duas regras (CV:43 razao_marcha > 0,50; S:361 portão do balanceador). Proposta: usar o portão 0,75 do balanceador em reto e parado, a mesma condição que abre a B. A razao_marcha ≥ 0,50 fica como aceite (4.2.4).

### 2.4 Fase 1: fatias por env

Cada env recebe a fatia no reset e a mantém até o próximo reset (CV:47-50, S:354-355).

| Fatia | Sequência | Pior caso |
|---|---|---|
| ANDAR | comando sorteado | ≤ 20 s |
| B | ANDAR parado (0,5-1,0 s) → PEGAR → CARREGAR (4-7 s, comando fixo no trecho) | 15,0 s |
| C | ANDAR parado → PEGAR → CARREGAR parado (0,5-1,0 s) → BOTAR → ANDAR parado (0-1,5 s depois de B4, one-hot fica em BOTAR) + 1,5 s | 15,7 s (CV) / 17,2 s (S:177-178) |

Teto do episódio: 20 s, time_out no fim. PEGAR e BOTAR sem relógio. Pega da C só em laje ≥ 0,30 m (CV:153).

Balanceador de tarefas (S:357-362): F_T = piso + (1 − n·piso)·(D_T + ε)/Σ(D + ε), piso 0,20, ε 0,02. D_T = média dos déficits das subtarefas abertas.

Balanceador de subtarefas (S:364-369): mesma fórmula, piso 0,10, portão 0,75, d_i = max(0, portão − s_i). A subtarefa seguinte abre quando as abertas da mesma tarefa passam. ANDAR e CARREGAR(B) têm balanceadores independentes.

Níveis por tarefa:

- CARREGAR(B): reto 4 direções, giro no lugar nos dois lados e arco, com os níveis de velocidade do ANDAR (CV:93-97).
- PEGAR: (0,55 m; 1 kg) → (0,45; 2) → (0,30; 3) → (0,15; 4) → (0,04; 5). Altura ∈ [h_k; 0,55] e massa ∈ [1; m_k] sorteadas independentes (S:375).
- BOTAR: altura de destino 0,55 → 0,04; massa da faixa dominada pelo PEGAR (S:376).

Mudanças propostas nesta fase (detalhe na seção 4):

- Tamanho da caixa vira eixo de nível do PEGAR: 0,09-0,11 primeiro, depois 0,07-0,13 (4.3.9).
- Eixo "chegada" (lateral e rumo) depois do nível de distância (4.3.8).
- Atrito da caixa por priority: [0,5; 0,6] após o 1º nível do PEGAR, descendo até 0,3 (4.3.1).
- Freio por nível ligado ao nível sorteado do env, não ao número de níveis abertos (4.3.5).
- Sucesso do PEGAR = P4 + 1,0 s sem terminação, igual em B e C (4.4.2).
- Sucesso do ANDAR/CARREGAR no balanceador = razão em [0,8; 1,2] (4.4.4).
- Na C, P4 não fechado até t = 13 s termina o episódio e conta como falha do PEGAR (4.4.8).
- Opcional, decisão do dono: CARREGAR andando antes do BOTAR na C (4.4.11).

### 2.5 Refino por tarefa

Começa quando o último nível abriu e o sucesso médio dos níveis ≥ 0,75 por 100 its. Cada degrau (0 a 5) pede o mesmo por 100 its (CV:493-516, S:393-413).

O que aperta por degrau (S:402-411):

| Regra | Começo → fim | Tarefas |
|---|---|---|
| Contato com laje | 50 → 10 N | PEGAR, BOTAR |
| Nivelamento | 30° → 15° | PEGAR, CARREGAR, BOTAR |
| Aperto | EM ABERTO | — |
| Sacudida | peso 0 → a definir | CARREGAR |
| Chegada da caixa | 1,98 → 0,99 m/s | BOTAR |
| Terminação no batente e hip_yaw > 1 rad | desligada → ligada | todas |
| Erro do estimador e DR | baixo → medido | todas |
| Pose de referência | peso 0 → a definir, forma EM ABERTO | PEGAR, CARREGAR, BOTAR |
| DR do PD, atraso, atrito seco | ver seção 3.6 | todas |

Mudança proposta: o contador desce um degrau quando o sucesso fica < 0,50 por 100 its (regra do DR:172). Hoje ele nunca desce (S:398-399). Ver 4.4.3.

Mudanças propostas que tiram coisas do refino e põem antes:

- Atraso da ação e do sensor 0-1 passo desde a fase 0 (4.3.6).
- Viés de tamanho e posição do estimador ±0,5 cm desde a abertura do PEGAR (4.4.6).
- Erro do alvo de soltura a partir de um nível do BOTAR (4.4.13).
- Desvio de hip_yaw no PEGAR/BOTAR desde o começo, antes da terminação do refino (4.3.3).

### 2.6 Estado no checkpoint

O checkpoint guarda médias de sucesso, frações, níveis abertos e contadores de refino (S:465-468). Acrescentar: bordas atuais de cada faixa de DR (se 4.3.1 e 4.4.3 entrarem).

---

## 3. Características de cada tarefa

### 3.1 Comum a todas

**Observação do ator** (S:772-793). Por passo, 121 canais:

| Bloco | Canais | Escala / ruído |
|---|---|---|
| Vel. angular da base | 3 | ×0,2; ±0,2 |
| Gravidade projetada | 3 | ±0,05 |
| Comando vx, vy, wz | 3 | sem ruído |
| q − q_de_pé | 29 | ±0,01 |
| q̇ | 29 | ×0,05; ±1,5 |
| Última ação | 29 | — |
| One-hot | 6 | sem ruído; fora do normalizador |
| Bloco da caixa | 17 | ruído do estimador; zero no ANDAR |
| Fase da marcha sin/cos | 2 | — |

Histórico de 5 → 605. Mapa de elevação 21×21 × (altura, confiança) = 882, sem histórico. Total 1 487. Sem velocidade linear da base, sem normalizador empírico.

Bloco da caixa, 17 canais (S:134-139): posição (3), giro de endireitar (3), meia aresta (1); alvo macro corrente: 2 palmas (6), altura da pelve (1), inclinação do tronco (1); idade da leitura da caixa (1) e do alvo de soltura (1). Tudo no frame da base. Canal novo entra no fim [C16].

**Observação do crítico** (S:145-149, S:790-792): a do ator sem ruído + velocidade linear + privilegiados do molde + pose verdadeira da caixa e do alvo + índice do alvo (P0-P4, B1-B4) + nível de cada tarefa + degrau de refino + massa da caixa + força normal total robô→caixa. Proposto: mapa perfeito. Proposta desta revisão: velocidade média da janela de rastreio, 3 canais (4.5.4).

**Ação** (S:151-157): alvo de posição das 29 juntas = q_de_pé + 0,25·a, sem clip. q_de_pé = pose 0 de `ref_v3_de_pe.npz` (joelho 0,16, cotovelo 0,60), não o default da Unitree. PD da Unitree: quadril 100, joelho 150, tornozelo 40, cintura 200, braços 40; amortecimento 2-10 sem valor por junta (S:687). Ver 4.2.3.

**Fase da marcha** (S:755-770): T = clamp(2·L_min/|v_cmd|, 0,8; 1,4) s; L_min 0,20 m a calibrar; giro no lugar T = 0,8 s; φ ← φ + dt/T. φ no reset e com comando zero não está definido (4.4.14).

**Reguladores comuns** (S:302-317): escorrego −0,2; vz −2; giro xy −0,05; vel. de junta −0,001; acel. de junta −2,5e−7; action_rate −0,05; energia −2e−5; limite de junta −5 (mole a 0,9 do curso); contato fora dos pés filtrado pelo parceiro −1.

**Terminações comuns** (S:329-337): time_out (não é falha); junta no batente e |hip_yaw| > 1 rad só no refino (opção A, AVAL §2); rampa no batente desde o começo.

**Custo de terminação**: −200·dt (= −4) mais o resto do episódio; o ganho já pago fica (S:345-346).

**DR desde o começo** (S:477-481): atrito do pé 0,3-1,0; massa do tronco −1 a +3 kg; viés de encoder ±0,015; empurrão ±0,5 m/s a cada 5 s em todas as tarefas; reset ±0,5 m com rumo livre; base_com fora (heap).

### 3.2 ANDAR

| Item | Valor |
|---|---|
| Obs | comum; bloco da caixa zerado [C13]; mapa ligado |
| Ação | comum |
| Recompensa | rastreio normalizado pelo comando: linear 1,0, angular 0,5, σ 0,5 (S:717-753); marcha por fase × rastreio 0,5; alive 0,15; altura do pé 1,0 (alvo 0,1 m); desvio dos braços −0,1 (pose de pé sem caixa); cintura −1, hip_roll/hip_yaw −1, tronco plano −5, altura da base −10 (alvo 0,78 m); reguladores comuns. Sem freio por nível |
| Rastreio normalizado | r = (exp(−e²/σ²) − exp(−c²/σ²))/(1 − exp(−c²/σ²)), piso −1; c < 0,02 vira exp(−v²/σ²) − 1; medido sobre a média de um período (≥ 0,8 s) |
| Terminação | inclinação > 0,8 rad ou base < 0,2 m |
| DR | comum; terreno como nível |
| Sucesso | por segmento: razão deslocamento projetado/pedido ≥ 0,50 sem queda (S:381-384); parado = deriva de posição e rumo abaixo do limiar. Proposta: no balanceador, razão em [0,8; 1,2] (4.4.4) |

### 3.3 PEGAR

| Item | Valor |
|---|---|
| Obs | comum; bloco da caixa ativo; alvo macro P0..P4 |
| Alvos (S:228-233) | P0 frente: palmas 5 cm à frente da face, y = ±(a + 0,035), tol 3 cm. P1 entrada: palmas no centro das faces, 3 cm. P2 pega: contato nas duas mãos, sem força mínima. P3 livre: caixa 5 cm acima da laje, sem apoio, nivelada. P4 peito: caixa a ≤ 5 cm do alvo de transporte (z = 0,90 m), pelve a ≤ 3 cm, 0,5 s |
| Recompensa | só potencial: r = γΦ(s') − Φ(s), γ 0,99; Φ = k_principal·(½ + ½·k_postura)·g_invariantes·g_faixa; σ = distância no início do alvo [C20]; B_alvo fixo por alvo. Nenhum termo positivo por passo (alive e altura do pé = 0). Reguladores comuns + contato com laje −2,0 por sensor (rampa a partir de 50 N) + freio por nível −6/29 + rastreio só de deriva |
| Invariantes | P0→P2: caixa anda < meia aresta na laje; caixa fora da laje = perdida. P2→B2: nivelada (30° → 15°) e duas mãos. Quebra zera Φ e impede avanço |
| Faixa (S:271-275) | tronco ±15°, pés ±8 cm em relação à caixa, CoM no meio dos pés |
| Terminação | fell_over a 70° ou joelho < 5 cm; caixa fora da laje antes da pega; caixa perdida (armada na 1ª pega) |
| DR | comum + massa 1-5 kg pelo nível; atrito da caixa por priority após o 1º nível; forma do estimador desde o começo |
| Sucesso | P4 fechado e o trecho seguinte sem terminação. Proposta: P4 + 1,0 s (4.4.2) |

### 3.4 CARREGAR

| Item | Valor |
|---|---|
| Obs | comum; alvo macro = pose de transporte |
| Recompensa | rastreio × qualidade (CarregarComQualidade: caixa no peito, nivelada, duas mãos); marcha por fase; alive 0,15; altura do pé; desvio dos braços (pose de pé com caixa, mão −30°); cintura, hip, tronco, altura da base; freio por nível; Sacudida no refino |
| Terminação | inclinação > 0,8 rad ou base < 0,2 m; caixa perdida |
| Duração | B: 4-7 s, comando sorteado na abertura e fixo. C: parado 0,5-1,0 s |
| DR | comum + atrito da caixa; impulso na caixa nos níveis seguintes (DR:146, :168) |
| Sucesso | como o ANDAR, com a caixa |

O controlador decide a duração do trecho, então terminar antes não compra passos (CV:227-229).

### 3.5 BOTAR

| Item | Valor |
|---|---|
| Comando | uma vez; abaixar, apoiar, soltar e voltar de pé ficam dentro (CV:183-188) |
| Alvos (S:233-236) | B1 sobre o alvo: caixa 5 cm acima, nivelada, ≤ 3 cm na horizontal. B2 apoiada: caixa ≤ 0,10 m do alvo, nivelada, chegada abaixo do limiar; apoio ≥ 0,8·m·g e força total robô→caixa < F_solta por 0,3 s, palmas nas faces. B3 solta: palmas ≥ 5 cm das faces, caixa parada, 0,5 s. B4 de pé: pose de pé sem caixa, pelve ≤ 3 cm, tronco ≤ 5°, 0,5 s |
| Recompensa | igual ao PEGAR (só potencial, B_alvo) |
| Terminação | fell_over 70° ou joelho < 5 cm; caixa perdida; chegada da caixa à laje > 1,98 m/s (0,99 no refino) |
| DR | como PEGAR; laje de destino aparece na abertura do BOTAR [C38] |
| Sucesso | B4 fechado e ANDAR parado final sem terminação. Proposta: B4 + 1,0 s (4.4.2) |

Problemas graves nesta tarefa: gate de força do B2 sem gradiente (4.2.2), B3 sem a caixa no lugar (4.4.5), hip_yaw sem custo (4.3.3).

### 3.6 DR por fase (consolidado)

| Quando | O que entra |
|---|---|
| Desde a it 0 | atrito do pé 0,3-1,0; massa do tronco −1/+3 kg; viés de encoder ±0,015; ruído Unitree; empurrão; reset ±0,5 m; massa e tamanho da caixa pelo nível; forma do estimador (atraso, retenção, oclusão, idade). Proposto: atraso da ação e do sensor 0-1 passo (DR:165) |
| Locomoção no portão | CoM do tronco (retestar base_com), rugosidade, ruído correlacionado (DR §5) |
| PEGAR no 1º nível | atrito da caixa por priority; CoM da caixa ±0,2; atraso 0-2. Proposto: viés do estimador ±0,5 cm |
| Níveis seguintes | μ descendo (proposto até 0,3); CoM ±0,3; impulso na caixa; RFI 0,05; viés do estimador até ±2 cm; erro do alvo de soltura |
| Refino | ganho do PD 0,9-1,1× (depois de alinhar ao deploy); força do motor; solref; atraso 0-3; RFI 0,1; atrito seco 0-0,05 N·m; erro do estimador até o medido; gravidade ±0,1 |

Sem erro em one-hot e comando (CV:587). A força de contato fica só na recompensa e no crítico.

---

## 4. Problemas encontrados, por gravidade

Marcas: **conferido na fonte** = a evidência foi lida no arquivo citado; **só do resumo** = vem do resumo, sem leitura direta.

### 4.1 Legenda de gravidade

- **Bloqueia**: o treino não entrega o objetivo, ou a política não roda no real, se nada mudar.
- **Alta**: regressão provável, renda falsa ou lacuna sim-to-real com medida que a sustenta.
- **Média**: risco real com solução barata.
- **Baixa**: higiene ou escopo.

### 4.2 Bloqueia

#### 4.2.1 O índice do alvo macro (8 canais do ator) e o `pegou` avançam por contato e força que só o simulador mede

Marca: conferido na fonte.

Evidência: S:134-136 (alvo macro sai do gerador); S:161-164 (gerador e estimador rodam no robô); S:228-236 (P2 = contato nas duas mãos; P3 = sem apoio; B2 = apoio ≥ 0,8·m·g e força < F_solta); CV:167, CV:543-548 (fonte real por alvo, sem regra de avanço); PERC:170 (caixa na mão = `pegou` do sim); LS/autonomo/fecho.py:83-87 ("Sem critério de torque: o aperto lateral domina e o torque dos ombros não cai ao pousar", run 20261005-132852).

Problema: o gerador roda no robô, mas P1→P2→P3 e B1→B2→B3 dependem de força de contato, apoio e "sem apoio". O G1 sem Dex3 não mede isso, e a v3 tirou o tau_est do ator (CV:581). No real, os 8 canais do alvo mudam num instante que o robô não reproduz. É um canal privilegiado que entra no ator por caminho indireto. O mesmo vale para o `pegou`. O torque já falhou como sinal de apoio.

Solução:
1. Divida o gerador em duas funções puras. (a) Recompensa e sucesso (Φ, B_alvo, fecho) leem a verdade. (b) `avanca_alvo(propriocepcao, estimativa, alvo_atual, t)` publica o índice no ator e lê só o que o real mede: palmas pela FK, caixa pelo estimador (ou FK das palmas depois da pega), altura da caixa sobre o topo da laje (mapa/estimador), tempo no alvo.
2. Critérios geométricos, por exemplo: P2 = palmas a ≤ 1 cm das faces estimadas por 0,2 s; P3 = caixa ≥ 3 cm acima do topo. Não use torque para apoio.
3. O treino usa (b) nos canais do ator.
4. Parte 0: concordância com o índice verdadeiro em rollouts roteirizados, meta ≥ 95% dos passos e atraso ≤ 2 passos.
5. Se a meta falhar: tire o índice do ator; mande só os alvos de pega e de transporte; o índice fica no crítico.
6. O `pegou` do estimador usa o mesmo critério.

#### 4.2.2 O gate de força do B2 não tem gradiente

Marca: conferido na fonte.

Evidência: S:234 (B2: força total < F_solta por 0,3 s); S:254-262 (força total 200-260 N com qualquer massa; F_solta = 0,25·m·g); S:285-286 (k_principal do B2 = caixa); `vault/G1/Abordagens/PPO/Currículos/C3 g1_limpo/Pendências/Aperto no limite do motor.md`:11 (preço do aperto −0,05 e −0,15 não moveu a força).

Problema: com a caixa apoiada, k_caixa ≈ 1 e a força não entra em gradiente. Para fechar o B2, a força tem de cair 95-99% (2,45 N com 1 kg, 12 N com 5 kg) com as palmas nas faces. Uma condição binária tão longe do ponto de operação não é achada pela exploração. Resultado esperado: o BOTAR trava no B2 e o time_out encerra sem custo.

Solução (forma de termo existente, sem termo novo):
- k_B2 = k_caixa · k_força, com k_força = exp(−(max(0, F − F_solta)/σ_F)²); σ_F = força total no instante em que o B2 começa (regra C20).
- Piso: F_solta = max(0,25·m·g; 10 N), acima do ruído de contato do Warp.
- Parte 0 item 2: medir dk_força/dF em F = 230 N (não zero) e confirmar que (apoio ≥ 0,8·m·g, F < F_solta) é alcançável pelo roteiro de IK com palmas nas faces.

#### 4.2.3 Contrato de atuação treino↔deploy com duas fontes

Marca: conferido na fonte.

Evidência: S:153-156 (q_de_pé da IK com joelho 0,16 e cotovelo 0,60, contra default Unitree 0,30 e 0,97); S:687 (PD 100/150/40/200/40; "amortecimento 2 a 10" sem valor por junta); mjlab `asset_zoo/robots/unitree_g1/g1_constants.py`:112-123 (kp = armadura·(2π·10 Hz)²); DR:77, DR:153 (braço a 14,25 N·m/rad no mjlab; "alinhar primeiro"); CV:581; a Parte 1 (S:613-620) não tem o passo de alinhar.

Problema: o ator infere o aperto por τ = kp·(q_de_pé + 0,25a − q) (S:264-268). Braço com kp 40 no deploy e ~14 no treino: a mesma ação aperta ~2,8× mais no real, muito fora da DR 0,9-1,1×. Na perna, a rigidez muda a resposta ao impacto. O kd não tem número. O yaml de deploy do unitree_rl_lab usa outro default_joint_pos: ação e observação ficam 0,14 rad deslocadas no joelho e 0,37 rad no cotovelo. Nenhum teste do sim mostra isso.

Solução:
1. Antes da Parte 1, o dono escolhe: treinar com kp/kd do yaml de deploy (`unitree_rl_lab/deploy/robots/g1_29dof/config`) ou levar o kp do mjlab ao deploy (decisão 6.1).
2. Arquivo único no `g1_v3`: kp, kd, effort_limit, armadura, q_de_pé, escala, ordem das 29 juntas, dt, decimation. O cfg do mjlab, o exportador de `contrato_deploy.json` e `laco.py`/driver leem esse arquivo.
3. Testes: kp/kd do MjModel compilado = contrato; paridade de observação < 1e-6 a partir de um estado salvo.
4. Corrigir S §6.3 e §19 com kp e kd numéricos por junta.
5. DR de ganho 0,9-1,1× só depois do teste 3 passar.

#### 4.2.4 O CV, com autoridade maior que a S, tem regras antigas que mudam o treino

Gravidade original: alta. Fica aqui porque a ordem de autoridade faz o implementador seguir a regra errada.

Marca: conferido na fonte.

Evidência: CV:175 (P2: força entre F_ref e k·F_ref); CV:193 (B2: apoio ~1·m·g); CV:199 (P2 até B3 com força na faixa); CV:365 (P4 por massa, joelho 0,20-0,37); CV:367 e CV:507 (air_time no refino); CV:510 (aperto 8× → 2,5× F_ref); CV:523 (P2 "força na faixa", B2 0,8·m·g); CV:43 (fase 0 sai por razao_marcha > 0,50). Contra: S:7-15, S:232-262, S:197-200, S:768-770, S:357-362.

Problema: quem implementa pelo CV recebe (a) faixa de força até o B3, impossível junto com o gate do B2 e com o B3; (b) apoio ~1·m·g contra 0,8·m·g; (c) P4 por massa contra pose única; (d) aperto em F_ref, que o dono tirou; (e) air_time contra marcha por fase. O CV se contradiz (:193 contra :523). Duas regras de saída da fase 0.

Solução: atualizar CV:175, 193, 199, 365, 367, 507, 510, 523 com o texto de S:232-262, S:197-200, S:768-770 (duas mãos e nivelada de P2 a B2, sem faixa de força; apoio ≥ 0,8·m·g; pose única com caixa; aperto EM ABERTO; sem air_time). Uma regra só para a fase 0 (proposta: portão 0,75 do balanceador em reto e parado). razao_marcha ≥ 0,50 fica como aceite. Até o CV ser corrigido, os implementadores seguem a S nesses pontos.

Nota: o resumo da spec registrou outras divergências textuais (bloco da caixa 10→15 em CV:522 vs 17 em S:139; lidar desligado em CV:35/DEC:220 vs ligado em S:78; portão razao_marcha mantido em S:626 vs "sai" em S:678; B3 "0,5 s" e B2 "~1·m·g" em CV:194-195). Corrigir junto. Valor que vale: S (decisões mais recentes).

### 4.3 Alta

#### 4.3.1 Faixa de DR do atrito da caixa não cobre a palma de plástico (μ 0,3-0,5)

Marca: conferido na fonte.

Evidência: S:250-252 (palma de plástico μ 0,3-0,5; 82 N por palma contra ~100 N do braço); S:482 (adota o DR §5, sem número); DR:100, DR:167-168 ([0,6; 1,0] → borda até 0,45); `vault/.../C4 v3/g1-v3-palma-de-plastico-5kg.md`:12 ([0,3; 0,6]); `vault/.../Achados do g1_limpo/g1-limpo-caixa-presa-pelo-pulso.md`:9 (com priority, 0,45 solta 5 kg; 0,3 solta 3 e 5 kg).

Problema: a política nunca vê μ < 0,45. O real terá 0,3-0,5. μ 0,3 solta 3 e 5 kg. CARREGAR dura 4-7 s.

Solução: faixa final [0,3; 0,6] por priority em S:482 e DR:100/168. Cronograma: [0,5; 0,6] depois do 1º nível do PEGAR → 0,4 → 0,3 nos níveis de massa ≥ 3 kg, sempre depois da recompensa do resultado. Priority da caixa acima de todo geom do robô. Aceite: μ 0,3 × 5 kg, CARREGAR 7 s a 0,5 m/s, 9 de 10 sem a caixa descer > 2 cm.

#### 4.3.2 Invariantes e faixa entram no Φ como portão binário

Marca: conferido na fonte.

Evidência: S:242-245 ("Quebrar um invariante zera a proximidade… endireitar a devolve"); S:285 (Φ = … · g_invariantes · g_faixa); S:271-275; S:405 (30° → 15° no refino).

Problema: entre 0° e o limite, g = 1 e nivelar melhor não paga. Acima, g = 0 e a derivada para endireitar é zero. Na fase de aprender o robô leva a caixa a 25-29° de graça. Quando o degrau aperta para 15°, essa região passa a Φ = 0 sem derivada; o sucesso cai e o contador para.

Solução (forma do fator existente): g = clip(1 − (θ − θ_lim)/Δθ, 0, 1), θ_lim = limite do degrau corrente, Δθ fixo (ex. 15°). O invariante duro que impede o avanço segue em θ_lim. Mesmo para g_faixa. Parte 0 item 2: derivada a θ_lim + 5°.

#### 4.3.3 hip_yaw sem custo no PEGAR e no BOTAR até o refino

Marca: conferido na fonte.

Evidência: S:310-312 (desvio de hip_roll/hip_yaw −1 só em ANDAR e CARREGAR; limite mole a 0,9 do curso); S:337, S:409 (|hip_yaw| > 1 rad termina só no refino); `vault/.../C3 g1_limpo/Pendências/Perna virada no BOTAR.md`:14 (hip_yaw E −0,69 a −0,91 no 21700; rampa a partir de 40° paga 900 its sem mover a pose no 20500).

Problema: no PEGAR e no BOTAR nenhum termo vê hip_yaw abaixo de 0,9 do curso. A linha zero aprendeu −0,91 no BOTAR. No refino, uma terminação dura a 1 rad liga sobre um comportamento a ~0,1 rad da borda, sem gradiente prévio.

Solução: estender o termo de desvio de hip_roll/hip_yaw (−1), que já existe, ao PEGAR e ao BOTAR, com alvo = hip_yaw da IK do alvo corrente (tabelas theta-30). Não usar rampa de limite sozinha. Antes de ligar a terminação: p99 de |hip_yaw| no BOTAR < 0,8 rad no play.

#### 4.3.4 Desistir sai barato no PEGAR e no BOTAR

Marca: conferido na fonte.

Evidência: S:296-300 (nenhum termo positivo por passo); S:345 (custo −4 + resto do episódio; ganho pago fica); S:347-348 (escada só compara queda, perda e volta); S:606 (Parte 0 item 3).

Problema: no PEGAR e no BOTAR o resto do episódio é só regulador até 20 s. No BOTAR, a única renda futura é o ANDAR parado final (~1,7). Com os B_alvo pagos, cair num B1/B2 difícil custa −4 e evita até ~15 s de regulador. Na perda: P0-P2 pagam B_alvo e ficam; "pega, sobe um pouco, larga" pode render.

Solução: duas desigualdades na escada da Parte 0, medidas no ponto de operação:
1. custo da queda > reguladores de 20 s parado no BOTAR + B_alvo restantes;
2. custo da perda > Σ B_alvo de P0-P3 já pagos + Φ máximo.

Se (2) não fecha com custo fixo, a perda devolve o B_alvo pago desde P2 (sem termo novo). Conferir com trajetória roteirizada "pega e larga".

#### 4.3.5 Freio por nível repete a catraca da zero17

Marca: conferido na fonte.

Evidência: S:315; S:388-391 (degrau = número de níveis abertos); ZERO17:403-426 (freio_degrau 1,75 → 3,0; fell_over 0,4 → 1,33; s_C 0,53 → 0,41; "nunca desce"); INV:110 (C8: freio por velocidade de junta, piso na locomoção).

Problema: níveis abertos só aumentam, então o freio é catraca por construção. Níveis fáceis recebem o freio máximo. No CARREGAR até 1,5 m/s, a spec não diz se o piso da locomoção do C8 continua.

Solução: degrau do freio = nível sorteado no reset daquele env (altura e massa no PEGAR/BOTAR; teto de velocidade no CARREGAR). No CARREGAR, manter o piso do regime de locomoção do C8. Mudança só na função do degrau. Registrar em S:390.

#### 4.3.6 Atraso da ação e do sensor só no refino; o real tem 10-30 ms

Marca: conferido na fonte.

Evidência: S:482 (adota DR §5, mas diz atraso no refino); CV:515, CV:585; DR:136 (10-30 ms), DR:165 (0-1 passo no primeiro treino), DR:194 (atraso mata a marcha se ligado tarde).

Problema: marcha e contato aprendem com atraso zero. Pôr 0-3 passos no refino muda a dinâmica já fixada; razao_marcha cai; o contador não volta. A S cita o DR e o contradiz na mesma linha.

Solução: atraso da ação 0-1 passo (sorteio por episódio) e 0-1 passo no bloco de propriocepção desde a fase 0. Subida a 0-2 e 0-3 por nível e refino (DR:167, :169). Corrigir S:482, CV:515, CV:585. Aceite: razao_marcha ≥ 0,50 com atraso 1 passo em ~1000 its.

#### 4.3.7 O aperto medido leva os motores do braço ao limite de 25 N·m

Marca: conferido na fonte.

Evidência: mjlab `g1_constants.py`:91-110 (effort_limit 25 N·m no 5020, 5 N·m no 4010); S:259-268; `vault/.../Pendências/Aperto no limite do motor.md`:12 (shoulder_yaw no limite); ZERO17 §3b (punho E além de ±5 N·m).

Problema: saturado, τ deixa de ser kp·(alvo − q) e o ator perde a via para sentir o toque. Sustentar ~25 N·m por 4-7 s aquece o 5020 no real; o firmware corta o torque e a caixa cai. O sim não modela temperatura.

Solução: (a) conferir pico e contínuo do 5020 e 4010 na ficha (DR:4 "não verificado"); (b) medir fração de passos com braço saturado (meta < 5%) e |τ| médio em 1 s no CARREGAR com 5 kg (meta ≤ 60% do limite); (c) se falhar, o desenho do aperto (EM ABERTO) usa mais apoio no tronco (permitido, S:250-252), não mais pressão lateral.

#### 4.3.8 Sem chegada desalinhada nem chegada andando

Marca: conferido na fonte.

Evidência: S:474 ("jitter curto [C37]", sem número); S:172-173; E:129-131; INBOX:7 (06/10: chegada desalinhada, poucos passos laterais; "o freio de postura parada hoje pune passos no PEGAR").

Problema: no real o robô chega do ANDAR com velocidade residual e erro do SLAM e da estação. A v3 só tem distância como nível. Sem número para a faixa do reset, não dá para garantir que a tolerância da estação (5 cm, 0,1 rad) fica dentro do treino. Freio e faixa de pés ±8 cm punem o passo de correção.

Solução: fixar na §12 dx, dy, rumo e yaw da caixa no nível 0. Eixo de nível "chegada" no PEGAR (e no alvo de soltura do BOTAR) depois da distância: lateral 0 → ±0,10 → ±0,20 m; rumo 0 → ±10° → ±25°. Tolerância da estação = faixa − margem do SLAM (ATE p95) e do estimador, com teste que falha se a estação aceitar pose fora. Com alvo P0, isentar passos de correção do freio. Aceite: 9 de 10 pegas com erro 0,15 m e 20°. Levar a decisão do INBOX:7 ao CV.

#### 4.3.9 Caixa pequena tomba na pega, e a v3 mantém meia aresta 0,07-0,08 desde a it 0

Marca: conferido na fonte.

Evidência: ZERO17:487-489 (0,07 e 0,08 tombam ~70° com 0,5-1 kg; 0,09-0,10 funcionam); S:474 (0,07-0,13 por mundo); S:212 (laje 0,04 m com caixa 14 cm não fecha na IK).

Problema: ~1/3 dos mundos tem caixa que a política atual tomba, e a IK não fecha (0,04 m; 14 cm). Renda zero sem gradiente numa fração fixa desde a it 0.

Solução: tamanho vira eixo do nível do PEGAR: 0,09-0,11 primeiro; 0,07-0,08 e 0,12-0,13 depois do 1º nível. Tirar (laje 0,04; meia aresta 0,07) do sorteio até a Parte 0 item 10 fechar a IK. Medir de novo o tombo com mão −30°.

#### 4.3.10 O detector da caixa sem marcador não existe; latência e taxa sem número

Marca: conferido na fonte.

Evidência: S:489-502 (atraso = "latência medida", sem número; real = RGB sem marcador + depth); CV:545 ("tag CAIXA ou as CNNs"); vault box-vision (CNNs com tag, RANSAC no depth).

Problema: os 7 canais da caixa e a idade pedem detecção, segmentação e tamanho de uma caixa de papelão comum. O g1-box-vision foi medido com tag. Sem latência e taxa reais, atraso e idade do treino são chute. ±2 cm na meia aresta é 57% da folga de 3,5 cm do P0 e vira erro de força.

Solução: fonte real em 3 estágios: (1) máscara pelo depth acima do plano da laje vindo do mapa, ou detector leve sem tag; (2) RANSAC de planos no depth recortado dá centro, eixos e arestas; (3) a rede só desambigua a face de cima. Medir na bancada com a D435i real: latência p50/p95, taxa, erro de posição e aresta a 0,4-1,5 m. Até medir: treinar com atraso 2-4 passos e leitura retida a 15 Hz.

#### 4.3.11 O fecho do controlador (P4, B4) usa a pose da caixa e o q_default antigo

Marca: conferido na fonte.

Evidência: LS/autonomo/fecho.py:59-64 (de_pe compara com `contrato.q_default`), :71-87 (assentada pela pose e velocidade da caixa), :97-110; S:233-236; S:172-173.

Problema: no treino a troca PEGAR→CARREGAR e BOTAR→ANDAR vem do fecho pela verdade. No real, fecho.py decide pela caixa estimada, que fica na zona cega ou atrás dos braços. de_pe usa q_default, e a v3 usa q_de_pé. Troca cedo leva a estado fora da distribuição.

Solução: reescrever fecho.py para a v3 com grandezas observáveis. P4: caixa pela FK das palmas a ≤ 5 cm do alvo de transporte e pelve pela FK a ≤ 3 cm, por 0,5 s. B4: max|q − q_de_pé| < limiar e palmas a ≥ 5 cm das faces estimadas, por 0,5 s. "Assentada" vem da última leitura válida antes da soltura + mapa. q_de_pé do contrato único. No treino, sortear 0-0,5 s de atraso na troca.

### 4.4 Média

#### 4.4.1 O mapa de elevação nasce frio no reset da B e da C

Marca: conferido na fonte.

Evidência: PERC:26-31 (zona cega < ~1,0-1,3 m, preenchida pela memória), :42-47 (célula nunca vista = confiança 0), :93-96 (chão mais perto visto a 0,89 m), :349; S:172-173.

Problema: no reset da B e da C o robô nasce a ~0,5 m da laje, dentro da zona cega; quase todas as 441 células ficam com confiança 0 no PEGAR. No real o robô viu a laje ao se aproximar. A política aprende a ignorar o mapa na manipulação, ou depende de um estado que o real não tem.

Solução: no reset da B e da C, pré-encher o mapa com 1-2 s de leituras simuladas ao longo de uma aproximação virtual de 1,5-2 m (só raycast e filtro). 20% dos resets com mapa frio. Na fase 5c, medir fração de células com confiança > 0,5 em 4 casos (parado, depois do reset, andando a 0,5 m/s, caixa no peito) e fazer ablação com mapa zerado na manipulação.

#### 4.4.2 O sucesso do PEGAR depende do trecho seguinte, diferente em B e C

Marca: conferido na fonte.

Evidência: S:383, S:384, S:361-362, S:172-173.

Problema: na B o trecho seguinte é CARREGAR andando 4-7 s até 1,5 m/s; na C, parado 0,5-1,0 s. Queda no CARREGAR conta como falha do PEGAR e prende os níveis do PEGAR e a abertura da C, contra S:362.

Solução: sucesso do PEGAR = P4 + 1,0 s sem terminação, igual em B e C. BOTAR = B4 + 1,0 s.

#### 4.4.3 O refino em catraca contradiz o DR

Marca: conferido na fonte.

Evidência: S:398-399 ("nunca desce"), S:452 (risco 4), S:465 (proposta de voltar um degrau), S:596-598; CV:500; DR:39, DR:172.

Problema: cada degrau aperta várias coisas juntas. Se o sucesso cai, o contador congela no degrau difícil. O aceite de 300 its não diz o que fazer.

Solução: o contador desce um degrau com sucesso < 0,50 por 100 its (DR:172). Registrar no DEC; corrigir S:399 e CV:500.

#### 4.4.4 O portão do ANDAR e do CARREGAR abre o nível seguinte com metade da velocidade

Marca: conferido na fonte.

Evidência: S:382 (razão ≥ 0,50), S:366-368 (portão 0,75), S:373; ZERO17 §4 (CARREGAR a 0,35 m/s com pedido 0,5).

Problema: abre o nível de 1,25 m/s com o robô a ~0,6 m/s. O CARREGAR lento da zero17 passaria.

Solução: separar réguas. Balanceador: razão em [0,8; 1,2] e desvio de rumo abaixo do limiar. Aceite: ≥ 0,50.

#### 4.4.5 O B3 não exige a caixa no lugar

Marca: conferido na fonte.

Evidência: S:235; ZERO17:481-486 (soltura arrasta 7,3 / 11,4 cm; caixa_largada só termina fora de 0,18 m).

Problema: caixa arrastada 11 cm ou tombada, mas parada, fecha o B3.

Solução: condição do B3 + "caixa a ≤ 3 cm e ≤ 15° da pose do fecho do B2". Mesma grandeza no k_principal do B3. Sem termo novo.

#### 4.4.6 Viés da câmera e erro de tamanho só no refino

Marca: conferido na fonte.

Evidência: CV:581; S:494-499.

Problema: força ≈ kp × penetração; 2 cm de erro de tamanho mudam a força em dezenas de N. A política fixa "apertar até a medida" antes do refino.

Solução: viés de tamanho e posição ±0,5 cm por episódio desde a abertura do PEGAR, crescendo até ±2 cm por nível (DR:3.5). Aceite na Parte 0: viés −2 cm no tamanho sobe a força total no P4 < 30%.

#### 4.4.7 Terminação no batente: decisão A vs B com texto aberto

Marca: só do resumo (AVAL §2, :4; DEC 02/10).

Evidência: AVAL §2 (P(alvo além do batente) por passo com std 1,0: ankle_roll 55,1%; 92,7% dos passos têm alguma junta além; pelo ângulo medido 6,2%; hip_yaw > 1 rad 13,1%). DEC 02/10 escolheu A (rampa desde o começo, terminação no refino). AVAL:4 ainda marca A vs B em aberto. O batente virtual das juntas de curso largo segue "a confirmar" (CV:353-355).

Solução: tratar A como decidida; fechar AVAL:4 com referência ao DEC. Definir o batente virtual por junta antes da Parte 1.

#### 4.4.8 Fome de BOTAR

Marca: só do resumo.

Evidência: DEC:40-41; S:171-178; S:357-359; ZERO17 §2 (funil da C na it 6320: 55% chegam ao CARREGAR, 35% ao BOTAR, 17% sucesso).

Problema: amostra do BOTAR = fração da C × alcance do funil, dividida pelos níveis. PEGAR lento deixa ao BOTAR menos tempo; time_out não é falha; o BOTAR que não termina não entra na média e o portão fica ruidoso.

Solução: na C, P4 não fechado até t = 13 s termina e conta como falha do PEGAR. Registrar o funil (fração de C que chega ao B1) e usar como peso do déficit D_BOTAR. Mudança no código de cadeia e de balanceador.

#### 4.4.9 Sem limite de velocidade da mão e sem camada de segurança no deploy

Marca: conferido na fonte.

Evidência: grep de "segurança|emergência|ISO" em S e CV: zero; S:386-391; INV F36 (mão a 2,58 m/s; ISO/TS 15066: 0,25).

Problema: braços balançam sem freio no ANDAR; empurrão ou recuperação gera mão > 1 m/s com 5 kg. Sem aceite de segurança, sem camada que limite q̇, τ, inclinação antes do modo amortecido.

Solução: (1) aceite: p99 da velocidade da palma ≤ 1,0 m/s em todas as tarefas, ≤ 0,5 m/s com caixa, com empurrões; (2) camada de segurança do deploy fora da política: clamp de |q̇| e τ, inclinação > 40° → amortecido, rampa de partida de 2 s até q_de_pé; testar no sim antes do G1.

#### 4.4.10 Manipulação entre vizinhas

Marca: conferido na fonte.

Evidência: E:131-135 (caixa isolada; vizinhas para o futuro); INBOX:7 (06/10, dono reabre); grep "vizinh" em S/CV: zero; S:474.

Problema: no depósito a caixa quase nunca está isolada. O mapa de 10 cm não separa caixas a < 10 cm. Enunciado e proposta não conciliados; o controlador não tem regra provisória.

Solução: decisão do dono (6.6). (a) Provisório: controlador só manda PEGAR com folga lateral ≥ meia aresta + 0,10 m, e BOTAR em região livre ≥ aresta + 0,20 m. (b) Nível "vizinhas": 0 → 1 → 2 caixas inertes, folga 0,30 → 0,20 → 0,12 m; invariante P0-P2 a cada vizinha (< 2 cm); crítico vê as vizinhas; ator recebe folga livre de cada lado (2 canais no fim, [C16]). Atualizar E:131-135.

#### 4.4.11 A transição CARREGAR andando → parado → BOTAR não é treinada

Marca: conferido na fonte.

Evidência: S:180-182 (risco aceito 02/10); S:173; S:540-543.

Problema: no deploy todo BOTAR vem de um CARREGAR com frenagem. No treino, só de CARREGAR parado curto logo depois da pega. O aceite não cobre BOTAR depois da frenagem.

Solução: reabrir com o dono (6.7). Na C: CARREGAR com comando sorteado 1,5-3 s (níveis da B já abertos) → parado 0,5-1,0 s → BOTAR; laje de destino à frente da pose do fim do trecho [C38]; conferir pior caso contra 20 s; abrir só depois do CARREGAR(B) passar no reto a 0,5 m/s. Mínimo: estender o aceite S:540-543 até B4.

#### 4.4.12 PEGAR e BOTAR param em 0,55 m; a bancada do REORIENTAR vai até 0,70 m

Marca: conferido na fonte.

Evidência: E:24-26 (ciclo pela bancada); E:124-125 (lajes 0-0,55 m; bancada 0,40-0,70 m); S:375-376.

Problema: quando o REORIENTAR entrar, BOTAR e PEGAR a 0,55-0,70 m ficam fora da distribuição. Tabelas theta-30 não têm essas alturas.

Solução: estender as tabelas a 0,70 m (Parte 0 item 10) e pôr nível 0,70 antes do 0,55. Ou registrar no CV que a bancada fica ≤ 0,55 m até a fatia R entrar.

#### 4.4.13 O alvo de soltura e a altura da laje de destino não têm modelo de erro na §13

Marca: conferido na fonte.

Evidência: S:484-502 (estimador só da caixa), S:136; CV:545, CV:584 (erro só no refino); DR:3 (laje ±3 cm z, ±5 cm xy).

Problema: 3 cm em z é da ordem da folga de 10 mm da IK e da tolerância do B1/B2 e muda a velocidade de chegada.

Solução: estender estimador.py com o alvo "suporte" (topo e borda no frame da base, idade, validade). Fonte real: ajuste de plano no mapa (confiança > 0,5) dentro do retângulo previsto pelo SLAM, refinado pelo RANSAC do depth. Erro do DR:3 fixo por episódio a partir de um nível do BOTAR. Medir no lidar_sim o erro do topo pelo mapa de 10 cm.

#### 4.4.14 A fase da marcha pede relógio idêntico no driver

Marca: conferido na fonte.

Evidência: S:758-768; LS/autonomo/rastreio.py.

Problema: φ é estado integrado. Outro dt, outra regra de reset, ou o comando antes da correção do rastreio.py tira a fase da distribuição sem sintoma. Com comando zero, T = 1,4 s, mas não se diz se φ corre, para ou zera, nem na troca de tarefa.

Solução: `fase_marcha(phi, cmd, dt)` como função pura no `g1_v3`, carregada por caminho no driver, dt 0,02 s, integrada sobre o comando que entra na observação. Fixar na §19 φ no reset e com comando zero. Incluir φ no teste de paridade. Remover rastreio.py quando a v3 seguir o twist.

#### 4.4.15 A odometria do mapa e do estimador é o ESKF offline em numpy

Marca: só do resumo.

Evidência: PERC:66 (predição a 50 Hz), PERC:170 (real = ESKF); LS/percepcao/eskf.py, odometria_lidar.py.

Problema: ESKF e KISS-ICP em numpy nunca rodaram em tempo real no computador do G1. A latência entra no atraso do mapa sem número.

Solução: núcleo C++ do kiss-icp (ou RKO-LIO) a 10 Hz com predição do ESKF a 200 Hz, processo à parte com timestamp. Medir latência p95 no computador do G1; se > 50 ms, subir o atraso do mapa no treino.

### 4.5 Baixa

#### 4.5.1 Caixa só cúbica

Marca: conferido na fonte. Evidência: E:99; S:134-139, S:191-196.

O enunciado define cubo, então a spec é coerente. No depósito a maioria das caixas é retangular. Pergunta de escopo (6.8). Se aceito: 3 meias dimensões (+2 canais no fim), gerador de IK em largura × altura, nível de razão de aspecto 1:1 → 1:1,5 → 1:2 depois do 1º nível do PEGAR.

#### 4.5.2 Ruído do estimador e da odometria "baixo" desde o começo, sem número

Marca: conferido na fonte. Evidência: S:494-499; DR:165.

Sem número, a implementação põe zero. Fixar na §13: posição σ 0,5 cm + 0,5% da distância; orientação σ 2°; perda 0,5% por leitura em rajadas de 0,1-0,3 s; deriva 0,5 cm/m. Subida por nível do PEGAR com a regra de bordas do DR:172.

#### 4.5.3 Densidade do lidar: treino 128-256 raios, real 20 mil pontos

Marca: conferido na fonte. Evidência: PERC:47, :76, :228-233, :345.

Com 100× mais pontos o Kalman converge mais rápido no real e a escala dos 441 canais de confiança muda. Na função pura, limitar a 1 ponto por célula por leitura (ou subamostrar a nuvem real). Na fase 6, histograma de confiança treino × real, KS < 0,1.

#### 4.5.4 O rastreio é pago sobre a média de ≥ 0,8 s, e o crítico não vê essa média

Marca: só do resumo. Evidência: S:738-741, S:790-793.

Pôr a velocidade média da janela (3 canais) no fim da observação do crítico. O ator não muda.

#### 4.5.5 Ordem dos 1 487 canais tem de bater bit a bit no driver

Marca: conferido na fonte. Evidência: S:772-793.

`monta_observacao_v3` no `g1_v3`, carregada por caminho no laço e no driver. Teste de paridade da observação inteira, cobrindo troca de tarefa e reset.

### 4.6 Pendências da spec que não viraram problema, mas bloqueiam números

- O aperto (S:247-253), desenho do dono; F_solta 0,25·m·g a calibrar.
- Forma do incentivo à pose de referência no refino e o risco R13 (S:421-463).
- Pesos B_alvo, ProgressoDoAlvo, custo da perda, Sacudida, pose (S:321-325, S:407).
- Tolerâncias dos alvos e larguras da faixa (S:226, S:271-274).
- Batente virtual das juntas de curso largo e gatilho de cada degrau do refino (CV:353-355).
- Conflito limite de junta da Unitree × rampa do ZERO no PEGAR/BOTAR (REV:45-46).
- Custo do passo com 1 487 canais (fase 5c, S:792).

---

## 5. O que implementar fora do treino

Contrato = o que a política espera receber. Viabilidade: hoje, com o que existe. Ordem: 1 = antes da Parte 1; 2 = durante o treino; 3 = antes do G1 real.

| Módulo | Contrato | O que já existe | Como fazer | Viabilidade | Dificuldade | Ordem |
|---|---|---|---|---|---|---|
| Contrato de atuação | kp, kd, effort, armadura, q_de_pé, escala 0,25, ordem das 29 juntas, dt, decimation num arquivo | mjlab kp = armadura·(2π·10)²; yaml do unitree_rl_lab com outro default | arquivo único lido por cfg, exportador e driver; testes de paridade (4.2.3) | alta | média | 1 |
| Observação | 1 487 canais na ordem da §19.1; histórico 5; mapa sem histórico | `pilota.monta_observacao` (114 canais, g1_limpo) | `monta_observacao_v3` carregada por caminho (4.5.5) | alta | baixa | 1 |
| Fase da marcha | sin/cos 2πφ; dt 0,02; T pelo comando; regra no reset e com comando 0 | não existe no driver | função pura `fase_marcha` (4.4.14) | alta | baixa | 1 |
| Avanço do alvo macro (8 canais) | palmas, pelve, tronco do alvo corrente no frame da base; troca por critério observável | gerador planejado; avanço por força verdadeira (S:228-236) | `avanca_alvo` geométrico, mesma função no treino (4.2.1) | média | alta | 1 |
| Estado próprio | IMU e juntas no frame da base, 50 Hz | unitree_sdk2; `Propriocepcao` do autonomo | adaptador sdk2 (I1); q − q_de_pé do contrato | alta | baixa | 3 |
| Vel. linear da base | só no crítico e no navegador | — | navegador via SLAM | alta | baixa | 3 |
| Comando (vx, vy, wz) | 3 canais sem ruído; c < 0,02 = parado | navegador + rastreio.py | publicar direto; remover rastreio.py | alta | baixa | 3 |
| One-hot e máquina de estados | 6 slots; BOTAR fica 0-1,5 s depois do B4 | missao.py com SOLTAR | tirar SOLTAR; troca pelo fecho observável | alta | baixa | 3 |
| Fecho do controlador | P4/B4 por FK e q_de_pé | fecho.py com q_default e pose da caixa | `fecho_observavel()` (4.3.11) | alta | média | 3 |
| Estimador da caixa (7 + idade) | posição, giro, meia aresta; erro, atraso, idade | g1-box-vision com tag; CNNs; RANSAC; sim da D435i | segmentação sem tag + RANSAC; medir latência e taxa (4.3.10) | baixa (sem detector sem marcador) | alta | 2 (medir) / 3 |
| Caixa na mão | FK das palmas depois do `pegou` | `pegou` do sim | FK dos braços pelo MJCF + `pegou` observável | alta | baixa | 1 |
| Alvo de soltura / laje | topo e borda no frame da base; ±3 cm z, ±5 cm xy | proposta "tag MESA ou lidar"; ausente da §13 | alvo "suporte" no estimador: plano no mapa + RANSAC (4.4.13) | média | média | 2 |
| Mapa de elevação (882) | 21×21, 10 cm, altura relativa ao pé mais baixo, confiança; 50/10 Hz | plano PERC; sim do MID-360 | `percepcao_lidar.py`; 1 ponto por célula; pré-encher no reset (4.4.1, 4.5.3) | média | média | 1 (função pura) / 3 |
| Lidar MID-360 | 10 Hz, T_BASE_SENSOR | sim feito (não commitado) | SDK Livox + filtro do corpo | alta | média | 3 |
| SLAM / odometria | (x, y, rumo) por passo; ATE ~1,5 cm/4 m | ESKF e KISS-ICP em numpy, offline | kiss-icp C++ / RKO-LIO + ESKF 200 Hz (4.4.15) | média | média | 3 |
| Câmera D435i | depth 30 Hz; Min-Z 0,175-0,28 m | pose e sim | librealsense + extrínseco medido | alta | baixa | 3 |
| Navegação / estação | pose dentro da faixa do reset do PEGAR | estação 5 cm / 0,1 rad; faixa do reset sem número | fixar faixa na spec; teste da estação contra ela (4.3.8) | alta | média | 2 |
| Vizinhas | nenhum canal hoje | — | provisório: controlador exige folga (4.4.10) | alta | média | 3 |
| Camada de segurança | clamp q̇, τ; inclinação > 40° → amortecido; rampa de partida 2 s | I2-I4 no SIM2REAL, sem código | módulo no driver, testado no sim (4.4.9) | alta | média | 3 |
| Medidas no real | kp/kd do driver, resposta a degrau, limite de torque, μ do pad, rigidez da caixa, latência da visão, extrínsecos (DR:4) | — | bancada | alta | média | 2 |

Regra de replicação para todos os módulos que a política consome: uma função pura (torch/numpy, sem simulador), carregada por caminho (`importlib.util.spec_from_file_location`) no treino, no sim do real e no robô; nunca copiada (PERC §2).

---

## 6. Decisões que ficam para o dono

| # | Decisão | Opções | Referência |
|---|---|---|---|
| 6.1 | Qual kp/kd vale | (a) treinar com o yaml de deploy; (b) levar o kp do mjlab ao deploy. Também: kd por junta | 4.2.3 |
| 6.2 | Índice do alvo macro no ator | avanço geométrico observável; se a concordância < 95%, tirar do ator | 4.2.1 |
| 6.3 | Forma de k_força no B2 e piso do F_solta | k_B2 = k_caixa·k_força; F_solta ≥ 10 N. Liga com o desenho do aperto (EM ABERTO) | 4.2.2, S:247-253 |
| 6.4 | Atrito da caixa | faixa final [0,3; 0,6] e cronograma | 4.3.1 |
| 6.5 | Atraso desde o começo | adotar DR:165 (0-1 passo desde a fase 0) | 4.3.6 |
| 6.6 | Vizinhas | (a) provisório no controlador; (b) nível no treino | 4.4.10 |
| 6.7 | CARREGAR andando antes do BOTAR | reabrir o risco aceito em 02/10 | 4.4.11 |
| 6.8 | Caixa retangular | aceitar 3 dimensões e nível de aspecto, ou manter cubo | 4.5.1 |
| 6.9 | Altura até 0,70 m | nível 0,70 ou bancada ≤ 0,55 m até a fatia R | 4.4.12 |
| 6.10 | Refino pode descer | adotar DR:172 | 4.4.3 |
| 6.11 | Saída da fase 0 | portão 0,75 do balanceador ou razao_marcha | 4.2.4 |
| 6.12 | Chegada desalinhada | faixa numérica do reset e eixo "chegada" | 4.3.8 |
| 6.13 | Desenho do aperto | já EM ABERTO; considerar limite térmico do braço | 4.3.7, S:247-253 |
| 6.14 | Forma da pose de referência no refino (R13) | pose no avanço, condição de avanço, dentro de Φ | S:435-463 |
| 6.15 | Chave CAIXA_PERDIDA_VIRA_ANDAR | testar com volta atrás; custo da perda depois de 4.3.4 | S:339-343 |
| 6.16 | Proposta SONIC (INBOX:6) | teste rápido em MuJoCo ou não; nova frente ao lado de BFM-Zero | INBOX:6 |
| 6.17 | Arrasto na soltura | raio ~3 cm desde o início ou só no refino | ZERO17:8 item 7; 4.4.5 |

---

## 7. Achados refutados

- Troca de alvo cria queda de Φ e pairar perto da tolerância: refutado; Φ depende do estado aumentado (índice no crítico, S:144-146), a soma telescopa (Ng, Harada, Russell 1999), e a Parte 0 item 1 confere que pairar rende zero.
- O pulso a μ 1,0 esconde a DR mesmo com priority: refutado; com priority todos os contatos usam o μ da caixa (`g1-limpo-caixa-presa-pelo-pulso.md`:9).
- Impulso na caixa ficou fora da spec: refutado; S:482 adota o DR §5, que põe força na caixa nos níveis seguintes (DR:146, :168).
- Alturas até 0,85 m (bancada, palete): refutado; enunciado limita lajes a 0-0,55 m (E:124); sobra só a bancada do REORIENTAR até 0,70 m (4.4.12).
- Pilha de caixas como suporte: refutado; fora do enunciado (E:123-135).
- Quase todas as células do mapa no ponto cego, política ignora o mapa: refutado em parte; PERC:26-31 preenche pela memória e o chão mais perto fica a 0,89 m (PERC:93); sobra o mapa frio no reset (4.4.1).
- Torque do braço como substituto de contato e de apoio: refutado para o apoio; o torque dos ombros não cai ao pousar (fecho.py:83-87).
- hip_yaw com rampa própria a partir de 0,6 rad: refutado; rampa a partir de 40° foi paga 900 its sem mover a pose (Perna virada no BOTAR.md:14).
- Freio no CARREGAR só acima de |cmd| + margem: refutado; o freio é por velocidade de junta, não da base (INV:110, C8).
- Rampa de invariante g = clip((θ_aprender − θ)/(θ_aprender − θ_degrau)): refutada; divide por zero na fase de aprender; trocada por rampa de largura fixa (4.3.2).
