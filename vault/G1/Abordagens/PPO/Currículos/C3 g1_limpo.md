---
tipo: abordagem
pai: "[[Currículos]]"
deriva_de: "[[C2 g1_poc]]"
estado: ativa
verificado: 2026-10-06
periodo: 25/08–05/10/26
resumo: "Máquina de elo com locomoção primeiro, em duas linhas: a linha bloco (resume encadeado) gerou o 21700, e o refino v2 dele deu o model_23250, MELHOR modelo; a linha zero (12 recomeços) não produziu política competente."
cluster: c3
---
> **Deriva de C2 g1_poc:** o C2 pegou e reorientou mas nunca andou, porque o portão do balanço media sobrevivência e entregou a fatia da locomoção à manipulação. O C3 põe a locomoção PRIMEIRO, com fatia de 95% e portão que mede marcha, e mantém a máquina de elo do C2.

## Hipótese
Refazer limpo, sem importar código dos módulos anteriores, preservando os dois resultados medidos (o C2 pegava e reorientava; o C1 andava e resistia a empurrão): locomoção PRIMEIRO com fatia de 95%, um objetivo na manipulação, cadeias de elos que trocam dentro do episódio, incentivo e não penalização.

## Método
Módulo `g1_limpo/` (185 commits, 26 specs, 33 planos e relatórios), branches `exp/g1-limpo`, `exp/g1-limpo-v2`, `exp/g1-limpo-v3`. Desenho em Desenho do g1_limpo. Três eras, cada uma um filho:

- **Linha bloco** (26/08–18/09): 1 recomeço do zero e ~21 blocos em resume encadeado até o `model_21700`.
- **Linha zero** (14/09–01/10): ~12 recomeços do zero (zero01–zero17), currículo de cadeia na zero17.
- **Refino v2** (01–05/10): resume do `model_21700` com o código da zero17 → `model_23250`.

## Resultado
- `model_23250`: ciclo completo andar → pegar → carregar → botar em lajes de 0,55 a 0,15 m com 1 e 3 kg, levantando sem derrubar; funil da cadeia C de 44–65%.
- A linha zero nunca formou uma política competente: "a recompensa atual sustenta e refina uma política competente, mas não a produz a partir do zero" (relatório de 01/10).
- Pendências em 05/10: ver Estado do C3 em 05-10.

## Por que parou
O C3 continua como referência (o `model_23250` é o melhor modelo). O trabalho novo foi para o C4 v3, que refaz o treino simples com as lições daqui: treino em degraus com resume, restrições em níveis que sobem com o fecho e podem descer, DR desde o começo.

## O que herda
Máquina de elo, currículo de cadeia, one-hot fora da normalização, R4/R4b (dificuldade do BOTAR pelo `s_C`), métricas sem peso do funil, registrador de juntas, e a lição central: o caminho de aprendizado importa tanto quanto o ótimo da recompensa.

## Filhos
<!-- filhos:inicio -->
- **Achados do g1_limpo** (frente · ativa) — Um achado por memória g1-limpo-*: o fato medido, o porquê e como aplicar. Complementa as notas por run das três linhas.
- **Pendências do C3** (frente · ativa) — As 12 pendências abertas do C3 em 05/10; cada uma é uma nota pendencia que sai da Home quando fecha.
- **Linha bloco** (abordagem · validada · 26/08–18/09/26) — Um treino do zero e ~21 blocos em resume encadeado, cada um mudando a recompensa sobre uma política que já andava; termina no model_21700, base do melhor modelo.
- **Linha zero** (abordagem · rejeitada · 14/09–01/10/26) — ~12 recomeços do zero (zero01–zero17), cada um com todas as restrições desde a it 0; nenhum produziu política competente; a zero17 trouxe o currículo de cadeia.
- **Refino v2** (abordagem · ativa · 01–05/10/26) — Resume do model_21700 com o código da zero17: model_23250 é o MELHOR modelo; as continuações e os resumes do 24350 (freio em catraca, preço do aperto) não melhoraram.
- **2026-09-21-botar-alinhado-pune-endireitar** (achado · ativa) — No BOTAR o alinhado congelado punia ENDIREITAR a caixa (chega tombada 30–52°); σ_ori preso no piso; levou ao FACE_DE_PE.
- **Desenho do g1_limpo** (decisao · ativa · 25/08–04/09/26) — Fases F0–F6, cinco estados com one-hot, cadeias de dois elos, currículo de três relógios, sete incentivos; depois o contrato de troca v2 e os gradientes v2.1.
- **Estado do C3 em 05-10** (achado · ativa · 05/10/26) — O que funciona (model_23250, 24350/24800), as 12 pendências abertas e o que motivou a v3.
- **g1-limpo-ang-cego-ao-tombo** (achado · ativa) — O ANG do PEGAR media uma face lateral e era cego ao tombo palma-a-palma; desde 28/09 mede o eixo de cima; MEDIDO 01/10 nos plays: a subida tomba 42° a 59° (5800, 6050) contra ≤ 24° no 5350; o log caixa_na_pega (8–9°) dilui
- **g1-limpo-bloco1-marcha-formou** (achado · ativa) — g1_limpo bloco 1: a marcha formou na it 800 com razao_marcha 0,468; e o portao da forma abre logo depois
- **g1-limpo-botar-veredito-do-dono-21700** (achado · ativa) — Veredito do dono no model_21700 (18/09): tudo satisfatório menos o BOTAR — tronco 65°, braço a 55% do alcance, hip_yaw E −34°; hipótese de hábito herdado do resume.
- **g1-limpo-cone-eliptico-nan** (achado · ativa) — Cone elíptico + impratio 10 diverge para NaN: Hessiano indefinido (mujoco_warp #1657), busca de linha esgotada (#3628) e warm-start; usar pirâmide.
- **g1-limpo-contrato-de-troca-de-tarefa** (achado · ativa) — spec APROVADA do contrato de troca de tarefa em voo; branch exp/g1-limpo-v2; gate da caixa, VALIDA fora da obs, espera publica ANDAR, girar no lugar
- **g1-limpo-desenho** (achado · ativa) — g1_limpo é o módulo ativo do G1 — o desenho vive na spec e no plano do repo; aqui ficam as âncoras que não se derivam do código
- **g1-limpo-limite-de-junta-curso-largo** (achado · ativa) — Rampa de limite a 85% do curso não morde em junta de curso largo: hip_yaw chega a 151° no BOTAR do 20500; a IK pede além da rampa.
- **g1-limpo-mesa-contato-zero** (achado · ativa) — Contato do corpo com a mesa: terminação (28/08) REVERTIDA em 01/09 para multa em RAMPA (−2, 50→100 N, 3 sensores); a terminação matava 76% da manipulação por ruído de exploração; v3 baixa o joelho para 10 N (dono, 01/10)
- **g1-limpo-piso-da-estatua** (achado · ativa) — Ficar parado rendia 145 contra 102 de explorar no PEGAR (rastreio pagava comando nulo); rastreio por elo destravou a pega. VALIDADO.
- **g1-limpo-registro-de-checkpoints** (achado · ativa) — Registro dos checkpoints que completam o ciclo, com sha256, commit provável e veredito do dono: 5800, 9100, 17999, 20500, 21700, 23250 (melhor).
- **g1-limpo-std-da-acao-por-junta** (achado · ativa) — o std da ação do g1_limpo é UM vetor de 29, sem estado; parou em 0,78 por 2800 iterações e caiu para 0,67 quando o ruído passou a custar renda proporcional
<!-- filhos:fim -->
