---
tipo: decisao
pai: "[[C4 v3]]"
estado: ativa
verificado: 2026-10-06
resumo: "Decisões da v3 com motivo e fonte: alvos macro de IK, locomoção da Unitree, palma de plástico com 5 kg, soltar exige afrouxar, Isaac Lab."
---
# G1 v3 — decisões e propostas da discussão de 01/10

Fonte: a conversa de 01/10 com o dono, depois do zero17. A v3 é um treino refeito do
zero, mais simples, só com o que funcionou e é estritamente necessário. Nada daqui foi
implementado. A spec da v3 nasce desta lista.

## Por que a v3

Medido em 01/10 no `g1_limpo`: 17 893 linhas (6 296 de smoke), 31 termos de recompensa,
tabela por estado de 110 pesos, ~214 campos de knob, 4 currículos, 174 commits desde 25/08,
12 treinos do zero. O robô falha nas quatro tarefas: não gira no lugar, a caixa tomba na
pega, bota mal, carrega agachado (pelve 0,68 m contra 0,79 de pé).

Diagnóstico: a recompensa faz dois trabalhos na mesma moeda, a SEQUÊNCIA (renda congelada,
tabela, teto ≥ piso, fecho, currículo de cadeia) e a QUALIDADE do movimento (pose, forma,
orientação, contato). Cada conserto de um quebra a conta do outro. Pela contagem, 15 das 26
linhas do catálogo do enunciado (§5) são economia de renda por estado.

## Decisões do dono

1. **Um modelo só** é requisito duro.
2. **Poses de IK viram alvos macro** das tarefas de manipulação. Sem mímica: o robô
   generaliza em peso, posição e tamanho da caixa. O alvo definitivo segue por tarefa:
   rastrear uma velocidade, ou botar a caixa num alvo e orientação.
3. **PEGAR, BOTAR e REORIENTAR são tarefas à parte**, cada uma com sua cadeia e seu nível.
   Reverte o "sem nível por tarefa" do R4b.
4. **Validados pelo dono:** o currículo do freio de velocidade por nível
   (`docs/planos/2026-09-23-freio-em-curriculo-pelo-nivel.md`) e as poses de IK da
   manipulação (`g1_limpo/ik/ref_botar.npz`, IK de pé).
5. **Toda tarefa é exercitada** com fatia parecida de envs: andar reto, girar, pegar,
   carregar, botar e, depois, reorientar.
6. **As transições são o ganho do `g1_limpo` sobre o `g1_multitask`.** Nenhum episódio
   nasce com a caixa na mão.
7. **As cadeias:**
   - ANDAR: só comando de velocidade.
   - B: ANDAR parado → PEGAR → CARREGAR, com as mesmas subtarefas do ANDAR (reto, giro
     andando, giro no lugar, parado).
   - C: ANDAR parado → PEGAR → CARREGAR parado → BOTAR → ANDAR parado por tempo
     determinado.
   - R, depois: ANDAR parado → REORIENTAR × k → PEGAR.
8. **Girar no lugar precisa de envs suficientes.** Hoje o ramo `turning` realiza
   `0,10 × (1 − 0,10) ≈ 9%` dos sorteios de locomoção (`knobs.py:551`), com ~30% da frota
   em locomoção: ~3% da frota.
9. **Tolerância de nivelamento para o alvo avançar: 15°** (01/10).
10. **Pose de pé parado = IK** (01/10): sem caixa joelho ~0,15; com caixa a IK de pé com caixa
    para a massa; andando, o keyframe do molde. Na v3 a duração da espera é do controlador,
    então o risco da estátua que motivou a rejeição no `g1_limpo` não existe.
11. **Alvo macro sem travar a exploração** (01/10, em discussão): o dono pensa em alvo de pelve
    e alvo das mãos; o resto se discute.
12. **Contato com a mesa ou a laje: rampa com joelho em 10 N** (01/10; hoje 50 N). Segue
    multa em rampa, como desde 01/09; não volta a terminação.
13. **Nenhuma junta chega ao limite; a penalização pode ser grave** (01/10). Limiar único em fração do curso
    REJEITADO pelo dono (01/10): cursos muito diferentes; braços e punhos já foram ao batente.
    Proposta em aberto: rampa por junta e lado, de largura angular min(15°, 30% do meio-curso),
    terminação no batente, IK gerada fora das rampas.
14. **As 15 poses da `ref_botar.npz` são o comportamento desejado** (01/10, vistas de frente e
    de lado): o tronco curvado na laje baixa com os braços estendidos é o certo. Folhas em
    `~/Downloads/ik_v3/`.
15. **Referências de postura da v3 aprovadas** (01/10, vistas de frente e de lado):
    `g1_limpo/ik/ref_v3_manipulacao.npz` (15 poses, PEGAR e BOTAR pela altura) e
    `g1_limpo/ik/ref_v3_de_pe.npz` (de pé sem caixa, mãos no keyframe: pelve 0,789, joelho
    0,16; e com a caixa no alvo: 3 kg a 0,85–1,00 m; 1 e 5 kg a 0,95 m). Elas SUBSTITUEM a aprovação do item 14: a tabela antiga punha a perna dentro da
    laje em 12 das 15 poses (até 54 mm), porque o gerador deixava a laje fora da folga desde
    16/09. Regras novas: folga robô × laje de 10 mm; cintura até 25° (rampa do batente de 5°).
    Consequências: tronco 7°–20° mais curvado e pernas mais retas nas lajes de 0,15 a 0,45 m.
    Na laje de 0,04 m (caixa a 4 cm do chão, confirmado pelo dono) o robô fica 9–11 cm mais
    atrás, e a caixa de 14 cm só fecha rente à borda da laje. Geradas por
    `~/Downloads/ik_v3/ik_cintura25.py`, `gera_de_pe.py` e `gera_de_pe_sem_caixa.py`, partindo das poses aprovadas; as
    folhas estão em `~/Downloads/ik_v3/ref_v3_lado.png` e `ref_v3_frente.png`. Os scripts
    entram no pacote da v3 na implementação.
16. **Durações dentro dos 20 s** (01/10): espera 0,5–1,0 s; CARREGAR da B 4–7 s; CARREGAR
    parado da C 0,5–1,0 s; ANDAR parado final 1,5 s; a pega da C só em laje ≥ 0,30 m.
    Estimativa a 0,25 m/s de mão: PEGAR 3–7 s, BOTAR 3,2–7,2 s; pior caso B 15,0 s, C 15,7 s.
17. **Refino progressivo e automático** (dono, 01/10): primeiro o robô aprende a completar as
    tarefas (pegar e botar em todos os níveis, andar com a caixa); depois o movimento é
    refinado, com as penalidades entrando aos poucos, como o freio. Teoria do dono: exigir o certo
    desde o começo cobra a exploração e trava em platôs. Apoio medido: freio −15 do início
    comprimiu o std; terminação da mesa matou 76% da manipulação; batente da cintura com rampa
    estreita (zero03). Limite medido: estratégia aprendida solta vira hábito, e apertar de uma vez
    falhou (resume com terminação da mesa, 230 its). Resultado (dono, 02/10): o resume do HEAD
    do `exp/g1-limpo-v2` sobre o 21700 (bloco anterior à v2) corrigiu boa parte dos
    comportamentos errados; ainda não está perfeito, mas progride. Esse aperto foi de uma vez.
    Corrigiu: a perna girada ao contrário no BOTAR; a caixa fica reta quase o tempo todo; bota
    em lajes mais baixas. Falta: anda com a caixa, mas o movimento é robótico.
    DECIDIDO (dono, 02/10): as terminações entram desde o começo, a do batente inclusive.
    DECIDIDO (dono, 02/10): a fase aprender termina quando UM modelo faz todas as tarefas,
    pegar e botar a caixa em todas as alturas; aí o refino começa e aperta aos poucos. A junta
    no batente também entra no refino, porque pode matar a exploração inicial. Proibido desde
    o começo fica só: queda do robô, caixa largada e caixa acima de 30° (o alvo não avança).
    Leitura minha, a confirmar: o `hip_yaw` e as juntas de curso largo ganham batente virtual
    dentro do refino. Na fase aprender, o freio sobe por nível, como hoje.
    DECIDIDO (dono, 01/10): a atração para a postura da IK (mãos e pelve) vale cheia desde o
    começo, porque ela incentiva a completar a tarefa; a faixa do tronco e dos pés em volta da
    IK também vale cheia desde o começo. As duas ficam fora do refino.
18. **Alvo macro** (01/10): mãos e pelve atraem para a IK em todos os alvos; tronco e pés em
    faixa em volta da IK; força das mãos em faixa entre `F_ref` e k·`F_ref`.
19. **Batente** (01/10): rampa por junta e lado, largura min(15°, 30% do meio-curso), ~50/s no
    batente, terminação no batente, a rampa nunca cobra a IK (cintura de 25° a 30°).
    DECIDIDO (dono, 02/10): a rampa no batente entra no refino; na fase aprender só o limite
    físico do MJCF age. DECIDIDO (dono, 02/10): a terminação no batente vale desde o começo.
    DECIDIDO (dono, 02/10): a rampa também vale desde o começo; a terminação custa mais que
    ficar na rampa. REFUTADO 02/10 (avaliação §2): ler o alvo comandado termina 92,7% dos
    passos com std 1,0; ler o ângulo medido termina 4 de 7 plays do 23250 em 1,6 s. A
    terminação no batente desde o começo volta à mesa do dono.
    Histórico: `NoBatente` na cintura no notebook do zero (zero02–04, 20–22/09) matou a
    manipulação e saiu em `9b7dfba`; `hip_yaw` > 1 rad num resume (18/09) custou ~400 its e
    funcionou. Terminação por junta serve no resume, não no zero.
    DECIDIDO (dono, 02/10): opção A. A rampa vale desde o começo; a terminação no batente
    entra no refino.
    Medido antes: na `zero04` a
    terminação no batente (`NoBatente`, −4) levou a cintura ao batente em 7 passos e o PEGAR
    nunca treinou; a `zero01`, só com a rampa, andou e pegou. O chão empurra o tornozelo além
    do batente do MJCF (`ankle_roll` 150%, `ankle_pitch` 120% do curso).
20. **Impacto** (01/10): o dono fechou "o pico da queda de 5 cm". O teste de queda mostrou que o
    pico lido por passo de controle é aliasado (5 cm leem 2,5 × m·g, 8 cm leem 9,0; real 11,8 e
    16,1). DECIDIDO (dono, 01/10): velocidade vertical da caixa no passo antes do primeiro
    contato, limiar 0,99 m/s (queda de 5 cm). Script: `teste_queda.py` do scratchpad de 01/10.

## Propostas em discussão (não aprovadas)

- **Pagar o progresso, não o estado.** O alvo macro paga `Φ(s') − Φ(s)`, com `Φ` = alvos
  já alcançados + proximidade ao seguinte (potencial; Ng, Harada e Russell, 1999). A soma no
  episódio é `Φ_fim − Φ_início`: parar, pairar e agachar antes pagam zero. A terminação custa
  um valor fixo e o resto do episódio, e o ganho já pago FICA (corrigido 01/10: devolver tudo
  faria o que está em jogo crescer ao longo da cadeia). Substitui a renda congelada, a tabela
  por estado e o teto ≥ piso. Comportamento completo: `specs/g1-v3-comportamento.md`.
- **O alvo macro sai da caixa real** (posição, giro, tamanho e alvo da tarefa), em poucas
  grandezas de espaço de tarefa: mãos nas faces, pose da caixa, pelve e tronco pela IK da
  altura. As juntas ficam livres.
- **Avanço por tolerância**, não por relógio.
- **O que pode regredir entra em todo alvo seguinte:** caixa nivelada e duas mãos.
- **O ganho por alvo cobre o custo do trecho**, medido no ponto de operação (freio,
  `action_rate`).
- **Duração dada pelo controlador** em toda tarefa que paga por passo: o CARREGAR da B, o
  CARREGAR parado da C e o ANDAR parado final da C. Fechar antes não compra passos.
- **Cadeia por env, não por episódio.** Fatia de env = fatia de passo, qualquer que seja a
  duração do episódio. SUBSTITUÍDO 02/10: o 1/3 fixo virou o equilíbrio do balanceador de
  tarefas (abaixo); o env troca de fatia só no reset.
- **Sorteador de comando único** para o ANDAR e o CARREGAR da B: reto, giro andando, giro
  no lugar e parado; as frações seguem o balanceador da fatia (proposta 02/10 em
  `specs/g1-v3-comportamento.md` §2: abertura em ordem, mais envs para a subtarefa com menos
  sucesso, piso para as aprendidas, iguais no portão). DECIDIDO (dono, 02/10): o mesmo
  balanceador serve às alturas, às velocidades e, depois, aos ângulos do REORIENTAR; abre só
  o nível mais fácil, e o seguinte abre quando o aberto está dominado; o ANDAR e o CARREGAR
  da B têm balanceadores independentes. DECIDIDO (dono, 02/10): a massa faz parte do nível;
  cada nível abre alturas mais baixas e massas maiores. Frear com a caixa vem da
  generalização do ANDAR: o comando do CARREGAR não muda dentro do trecho (conferir no play:
  `carregar:CARREGAR:4:0.5,para:CARREGAR:2`). Proposta: na C, a massa vem da faixa que o
  PEGAR já domina, como a laje da pega (o BOTAR só treina quando a pega da C fecha). O
  CARREGAR sorteia o comando na abertura, sem herdar o
  twist zero da pega (medido: a cauda herdava ~2,9 s parada).
- **O giro no lugar precisa de gradiente além da fatia.** No estágio 0 o |wz| sorteado
  fica abaixo do piso σ 0,707 do rastreio angular, e a estátua rende igual ao giro
  (medido 01/10). O |wz| do giro no lugar fica acima do piso, ou o piso desce; medir na spec.
- **Na C, a laje da pega sai da faixa já dominada do PEGAR.** A fronteira do PEGAR treina
  na B; o BOTAR só treina quando a pega da C fecha (hoje 25–41%).
- **Locomoção primeiro:** fase 0 com 100% de ANDAR até `razao_marcha` ≥ 0,50.
- **Ficam:** vantagem normalizada por grupo de tarefa; one-hot fora do normalizador desde a
  it 0. **Sai:** o orçamento igual por tarefa de agosto e o balanceador por sucesso.
- **Caixa visível na espera**, como diz o enunciado §4.2; hoje os canais vão a zero
  (`observacoes.py:144`).
- **Log:** fração medida de passos por tarefa (ANDAR, PEGAR, CARREGAR, BOTAR).
- **Deriva de rumo na marcha reta** (medido 02/10, 14 plays do 21850 ao 23250, trechos de
  vx 0,8 e 0,5 com laço de rumo k = 0,5 do `registra_juntas`): o rumo assenta até 13° à
  direita do rumo de referência (mediana ~6°), e o desvio da linha de referência vai de
  −1,23 a +0,66 m em ~4 m (corrigido 02/10: o 0,8–1,1 m de antes era contra o rumo do
  início do trecho, que inclui a volta do laço). A velocidade
  lateral do corpo é pequena (−0,01 a −0,05 m/s); a deriva vem do rumo. No equilíbrio do laço,
  o erro de rumo vale b/k, com b o viés de giro da política: b ≈ −0,04 a −0,09 rad/s. O laço
  converte erro de rumo em comando de giro com ganho k, e o kernel σ 0,707 do rastreio de
  giro vira um kernel de rumo de σ/k = 1,41 rad (81°): 10° de erro custam 1,5% do termo.
  Com k = 2, o kernel de rumo fica em 20°, 10° custam 22%, e a derivada a 10° sobe 12,7×.
  TESTE do dono (02/10, `23250_k2.csv`, mesmo checkpoint, `--rumo-k 2`): o rumo assenta em
  −2,1° com desvio-padrão 0,3° nos últimos 3 s, contra −6,1° e −6,5° (dp 0,7–0,9°) com
  k = 0,5; desvio da linha −0,20 m contra −0,40 m. Previsão era 1–3°: confirmada (n = 1).
  O viés implícito fica em −0,055 a −0,073 rad/s. Sobra: o resíduo b/k e o passo lateral
  de −0,02 m/s, que o laço de rumo não vê. DECIDIDO (dono, 02/10): k = 2 vira o padrão do
  `registra_juntas` para os testes longos. O treino segue com o 0,5 do molde até a v3.
- **Dois balanceadores com a mesma regra** (dono, 02/10; regra em proposta): um reparte a
  frota entre as fatias abertas, outro reparte cada fatia entre as subtarefas dela. Quem sabe
  menos recebe mais, quem sabe mantém piso; no fim 1/3 por fatia. Substitui a partição fixa
  por índice; a fração continua em envs, não em episódios [F29]. O déficit da tarefa é a
  média dos déficits das subtarefas, sem peso pela amostragem. Exemplo do dono conferido em
  `specs/g1-v3-comportamento.md` §2 (41/59 → 25/75 → 50/50).
  DECIDIDO (dono, 02/10): piso de tarefa 0,20.
- **Garantias de comportamento** (dono, 02/10; tabela em `specs/g1-v3-comportamento.md` §10).
  DECIDIDO: caixa sem sacudir = penalidade pela aceleração linear e angular da caixa, no
  refino, em toda tarefa de manipulação (medido no 23250: CARREGAR linear p50 2,9 m/s²,
  angular p50 17 rad/s²). Caixa longe: alvos de pelve e pés pela IK em relação à caixa,
  distância como nível. Caixa perdida → ANDAR parado com volta à pose de pé da IK: o dono
  quer testar, com chave para voltar à terminação. LEVANTAR: depois do REORIENTAR; começa de
  pé, empurrão forte derruba, o comando troca para LEVANTAR pela inclinação, objetivo é a
  pose de pé da IK. DECIDIDO (dono, 02/10): passo miúdo pelo `air_time` com duração mínima
  no refino; a caixa perdida tem chave que devolve a terminação de hoje. LEVANTAR: o dono
  pediu mini cadeia; DECIDIDO (dono, 02/10): referência FIRM, alvos macro L0 queda (a política
  age, com custo de impacto e solavanco), L1 estirado de bruços ou de costas (pose não
  ditada), L2 de pé pela IK (o caminho é dele); slot próprio reservado desde o começo (§10.1).
  O LEVANTAR é FUTURO, junto com o REORIENTAR (dono, 02/10).
- **Laje e prateleira** (dono, 02/10; `specs/g1-v3-comportamento.md` §11). FUTURO: a
  prateleira sai da v3, com arquitetura própria como o REORIENTAR; a v3 fica só com a laje.
  Antes: DECIDIDO: mãos
  entram pela frente na laje desde o começo; prateleira (cubo oco sem a frente, um pouco
  maior que a caixa) é nível tardio da v3; deslizar a caixa no piso da prateleira é
  permitido, registrado no enunciado §4.3 e §4.5. A grade do MID-360 é o que deixa a
  política distinguir os suportes.
  MEDIDO 02/10 (IK do BOTAR na prateleira): lateral 15 cm cabe, 7,5 cm é o limite; o teto
  limita a prateleira baixa: com 15 cm de teto, 9 de 15 poses; com 30 cm, 13 de 15, e o
  tronco fica igual ou abaixo do da laje aberta aprovada.
  REPROVADO (dono, 02/10): a estante tem vários níveis; só os braços entram no vão, e a
  cabeça não passa sobre o cubo. Com o peito no plano da frente, a pose de pé alcança o
  centro da caixa de ~0,67 a ~1,3 m; os níveis baixos pedem outra família de pose.
- **Grade do MID-360 na v3 só com a laje** (02/10). DECIDIDO (dono, 02/10): a chave
  `PERCEPCAO_LIDAR` fica desligada na primeira v3. O dono teme que a grade
  vire atalho: ANDAR nunca tem laje, o parado quase sempre tem laje e caixa. Análise: com a
  laje só, a grade traz pouco além do oráculo da caixa e dos alvos da IK (borda e altura da
  laje); ela separa as tarefas pela cena, e no galpão o ANDAR passa perto de móveis, que o
  treino nunca mostrou; precedente medido: os canais da caixa vazavam para o ANDAR [C13];
  custo de 1440 canais contra 114. A grade entra
  com suporte variado ou com obstáculos no ANDAR, por append no fim da entrada (precedente:
  `g1_poc/expande_checkpoint.py`).
- **Abertura por tarefa e refino por tarefa** (02/10; `specs/g1-v3-comportamento.md` §2 e §12).
  Dono: cada tarefa tem seu nível. Proposta: a fatia abre quando a tarefa de que depende
  domina o primeiro nível (B: reto e parado do ANDAR; C: PEGAR no primeiro nível); o refino de
  cada tarefa começa com o último nível aberto e o sucesso médio dos níveis ≥ 0,75 por 100 its.
  Lacunas da spec com proposta em §12.2.
- **Alvo no mundo e estimador** (02/10; `specs/g1-v3-comportamento.md` §13). Dono: precisa ser
  resolvido agora, o que muda a decisão de 24/09. Proposta: a política segue recebendo tudo no
  frame da base; um estimador fora dela ancora o alvo no mundo e o reescreve pela odometria;
  no treino o oráculo vira estimador simulado (viés, ruído, atraso, retenção, oclusão,
  deriva), com a forma desde o começo e as magnitudes no refino; caixa na mão pela cinemática.
  DECIDIDO (dono, 02/10): assim, com erro em praticamente tudo, crescendo no refino. A caixa no
  real: câmera RGB sem marcador + depth, um algoritmo de detecção e outro de orientação.
- **BOTAR de um comando só** (dono, 02/10): no play o robô bota e fica agachado com as mãos na
  caixa porque, no treino de hoje, o fecho do BOTAR publica ANDAR (`comando.py:1767`) e não
  exige soltar [A6]. Na v3, abaixar, apoiar, soltar e voltar de pé ficam dentro do BOTAR.
- **Locomoção pela receita da Unitree** (dono, 02/10; `specs/g1-v3.md` §19). DECIDIDO: a
  receita do `unitree_rl_lab` (G1 29dof Velocity, `4960b84`) fica pura, sem importar o código;
  faixa de 1 m/s nas quatro direções e ±0,5 rad/s; sorteio pelo balanceador da v3; termos
  posturais da Unitree só no ANDAR (e no CARREGAR sem os braços); a observação da Unitree
  (histórico de 5, sem velocidade linear, sem normalizador, com a angular) vale em todas as
  tarefas. Saem o freio por nível, o `giro_sem_gingado`, o laço de rumo no treino e o portão
  da marcha. CORRIGIDO 05/10: a marcha por fase tira a cadência alta, mas o passo fica 0,4·v
  (0,10 m a 0,25 m/s).
- **Rastreio em comando baixo, desvio da receita** (dono, 05/10; `specs/g1-v3.md` §19,
  conflito 5). Todos os modelos até hoje giram de forma estranha ou só executam com comando
  mais alto que o necessário: parado, o robô colhe `exp(−c²/σ²)` do rastreio (96% a 0,1 com
  σ 0,5). DECIDIDO: rastreio normalizado pelo comando (parado 0, perfeito 1), sobre a
  velocidade média no período da marcha; marcha por fase multiplicada pelo rastreio; comando
  baixo como subtarefa do balanceador; checagem na Parte 0, item 6.
- **Respostas de 05/10 à revisão** (`docs/relatorios/2026-10-05-revisao-spec-v3.md`):
  - velocidade: 1,5 m/s para a frente; trás e lado em 1,0; níveis 0,25 → 1,5 em passos de 0,25;
  - `alive` zero no PEGAR e no BOTAR;
  - passo miúdo com a caixa: corrigir desde o começo, não no refino;
  - caixa nivelada: 15° no fim do treino em TODA etapa, porque o conteúdo não pode sofrer
    impacto. REJEITADO: "caixa de pé só no BOTAR" e a caixa inclinada contra o peito (05/10);
    o antebraço segue proibido; o apoio no tronco segue permitido (enunciado §3);
  - uma pose de pé só: a IK de pé vira o default (offset da ação e alvo do desvio);
  - REJEITADO: o B3 exigir força total zero; o B3 fica como está ("se não acrescenta, não
    implementa");
  - de P0 a P2 a caixa anda menos de meia aresta; caixa fora da laje antes da pega = perdida;
  - freio por nível em TODAS as tarefas de manipulação (PEGAR, CARREGAR, BOTAR);
  - o enunciado é atualizado nos 9 pontos da revisão (B9);
  - contato com a caixa é livre só na janela da pega (P2 até B3, e no CARREGAR); fora dela
    (espera, P0, P1, B4, espera final) ele custa; mão no chão custa;
  - passo mínimo desde o começo: o período da marcha cresce quando a velocidade cai, e a fase
    entra na observação (seno e cosseno, 2 canais no fim);
  - uma pose alvo de pé com caixa para toda massa (a de 3 kg a 0,90 m, que já existe); o robô
    se adapta; as poses de 1 e 5 kg a 0,90 m deixam de ser necessárias;
  - a troca BOTAR → ANDAR parado vem 0–1,5 s depois do B4, sorteada;
  - dentro de um nível, altura e massa sorteadas de forma independente até o limite do nível;
    a laje de 0,04 m basta;
  - o ciclo do galpão é teste do dono depois do treino, e não aceite da spec;
  - EM ABERTO, o dono vai arquitetar: o aperto. Requisito dele: apertar o suficiente para a
    caixa não escorregar nem cair, e nunca esmagar a caixa, contra o corpo, entre as mãos ou
    contra laje ou parede. O P2 não exige força mínima;
  - aceites do C5 aprovados: sem batente (95% do curso), BOTAR a 0,04 m com 5 kg, giro de
    180° com caixa, empurrão com caixa; 9 de 10 rodadas;
  - giro de 180° em no máximo 4 s (0,79 rad/s): teto do giro sobe para 1,0 rad/s, com e sem
    caixa, níveis 0,25 → 1,0 (substitui o ±0,5 de 02/10).
- **Mão na IK, da pega ao transporte** (05/10; fotos e tabelas em `~/Downloads/ik_v3/mao_fixa/`).
  MEDIDO: a IK aprovada gira a mão 35°–91° da pega ao peito; com a caixa presa e nivelada isso
  é impossível, e o robô pega com a mão quase horizontal e deita o tronco. Levantar da laje de
  0,04 m com a mão da pega (−56°) leva o punho a 2° do batente. Varredura na laje de 0,04 m:
  a pega só fecha com a mão a −40° ou mais para baixo; de pé, −40° deixa o punho a ~10° do
  batente. Agachar mais REJEITADO pelo dono: a pelve a 0,45 m só fecha com o `hip_pitch` a 95%
  do curso, e o tronco só melhora de 85° para 79°; fica a pelve da IK aprovada. DECIDIDO pelo
  dono: mão a −30° contra o solo, da pega ao transporte; pés até ~2 cm fora do alvo na laje de
  0,04 m são aceitáveis. Tabela gerada (`ref_v3_manipulacao_theta-30.npz`, `ref_v3_de_pe_theta-30.npz`
  na mesma pasta): 14 de 15 pegas aceitas; a laje de 0,04 m com a caixa de 14 cm não fecha (pés a
  5,2 cm); de pé, nenhuma junta perto do batente. Custo: o tronco deita mais que na IK aprovada,
  +5° a +11° de 0,15 a 0,45 m (ex.: 0,45 m, 40° → 51°). APROVADAS pelo dono ("parece bom as
  poses"): as tabelas `theta-30` substituem as de 01/10 como referência da v3 (spec §8.1).
  Andar com a caixa parte da mesma pose de pé com a mão a −30°, para coerência entre os
  movimentos (dono, 05/10): no CARREGAR o desvio postural, braços incluídos, mira essa pose.
  Testado por tamanho: de pé, 14, 20 e 26 cm fecham sem junta perto do batente
  (`ref_v3_de_pe_theta-30_tamanhos.npz`; fotos `tamanho_<cm>_{lado,frente}.png`).
- **Pose de referência no refino** (dono, 05/10): o robô RECEBE por se aproximar da pose da IK do
  alvo corrente, mirando o alvo, SEM zona morta (REJEITADA a zona morta proposta). EM ABERTO, a
  discutir antes da implementação: como incentivar sem quebrar a tarefa (spec §11.5). RISCO
  registrado (avaliação R13): pagar a proximidade por passo reabre a renda por estado.
- **Respostas de 02/10 à §18 da spec:** branch `g1_v3`; a faixa inclui o CoM centrado nos pés;
  Φ como proposto; `hip_yaw` e batente virtual no refino; massa da C da faixa dominada; altura
  de transporte fixa em 0,90 m; CARREGAR da B com reto nas 4 direções, giro no lugar e arco.
- **Juntas de curso largo fora do batente** (medido 01/10, 26 plays em CSV na raiz do repo,
  contra as 22 poses de `ref_v3_*.npz`): o `hip_yaw` usa até 16° na IK e chega a 132° no play
  do 20500, com 26% do tempo acima de 1 rad; no 21700 o máximo é 52°. O punho em roll usa
  17° na IK e 66° no p99 dos plays; o ombro em roll, 20° e 62°; o ombro em yaw, 9° e 47°; a
  cintura em yaw, 5° e 28°. Todos ficam a mais de 40° do batente, onde a rampa da decisão 19
  não age. O `ankle_pitch` e o `waist_pitch` passam do batente em 3° e 1,5°.

## Enunciado atualizado (05/10)

FEITO em 05/10, com o OK do dono: §1 composições (REORIENTAR futuro), ordem de aprendizado
(balanceadores), fora do objetivo (estimador real, grade do lidar); §2 meia aresta, massa
desconhecida, estimador no lugar do oráculo, aperto, 15° em toda etapa; §3 pés, locomoção com
comando baixo, contato com a caixa só da pega até soltar, progressão pelo potencial; §4.1
faixa; §4.2 esperas; §4.5 BOTAR de um comando e proibido do tronco; §4.6 REORIENTAR futuro;
§5 linha do `air_time` corrigida e linha do comando baixo; §6 exceção da energia.

## Próximos passos

1. FEITO (01/10): inventário em `docs/relatorios/2026-10-01-inventario-v3.md`. 36 fatos,
   49 consertos validados, 15 remendos que saem, 27 rejeitados, 16 regras de processo,
   26 abertos para a spec, 8 contradições resolvidas.
2. Spec da v3: RASCUNHO em `specs/g1-v3.md` (02/10), com as perguntas abertas na §18.
3. Alvos macro do PEGAR e do BOTAR gerados e julgados no viewer, sem treino.
4. Pacote novo, só com as peças do inventário.
5. Treino em degraus: locomoção (com giro no lugar) → B → C.
