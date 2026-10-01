# Platô do BOTAR na zero17 — diagnóstico verificado

Data: 30/09. Método: dois workflows só-leitura (4 agentes de diagnóstico, 3 céticos e 1
crítico de completude). Nada rodou simulador de dinâmica, smoke ou treino. As medidas são
do log do TensorBoard, dos checkpoints (sondas de rede), de replay cinemático (`mj_forward`)
dos plays e de fórmulas do código reimplementadas. Scripts e saídas (temporários):
`/tmp/claude-1002/-home-joaobornelli-Documents-g1-training/7edc1d25-9383-4df4-a40a-9e392434fa57/scratchpad/wf_botar/`
e `.../scratchpad/wf_verif/`. Resultados em JSON: `.../scratchpad/wf_botar_result.json` e
`.../scratchpad/wf_verif_result.json`.

## 0. Resposta

- O plano zero18 (`docs/planos/2026-09-30-plano-zero18.md`) NÃO resolve o platô do BOTAR.
- "Botar para na mesa e o comando troca para parado" JÁ EXISTE: no fecho do BOTAR o
  `soltou` vira 1, o publicado vira ANDAR com twist zero, e a CAUDA paga `postura_ereta` ×8
  e `pose` ×8 (`comando.py:1715-1718`, `838-846`; `knobs.py:1309, 1313`).
- O fecho do BOTAR NÃO exige ficar de pé: é `perto ∧ alinhado ∧ apoiada` por 0,5 s
  (`comando.py:1559-1560`). O de pé saiu na v3.4. Ele continua no fecho do PEGAR e do
  CARREGAR da cadeia C (`comando.py:1557-1558`).
- A trava está ANTES do fecho do BOTAR, e em três lugares: o CARREGAR da C, a abertura do
  BOTAR e o próprio BOTAR. Além disso, o crítico e o ator "esqueceram" o BOTAR entre o
  `model_4650` e o `model_5000`.

## 1. Achados, com o veredito da verificação

| # | Achado | Tipo | Veredito |
|---|---|---|---|
| A1 | O crítico inverteu o valor de descer no BOTAR: V(alvo) − V(congelado) = +15,1 (4000), +11,4 (4150), +0,1 (4500), +4,7 (4650), −8,7 (5000), −8,9 (5200). O valor do fecho (V_CAUDA − V_BOTAR) caiu de +11 a +15 (4000–4650) para +0,9 (5000) | medido (sonda de rede) | novo, do crítico de completude |
| A2 | O ator perdeu o reflexo de descer na abertura do BOTAR: altura comandada das palmas −10,0 cm (4000), −10,9 (4150), −7,0 (4500), −4,4 (4650), +4,2 (5200). Com a caixa mais baixa, todo checkpoint comanda subir de volta, e mais forte na fase 4 | medido (sonda de rede) | novo |
| A3 | A abertura do BOTAR teleporta a laje para base + R_yaw·(0,50 ± 0,10; ± 0,10), topo0 ± 0,10, quatérnion identidade, sem checar o corpo (`comando.py:1799-1887`). Ela nasce dentro da coxa, do quadril ou da canela em 31–72% das aberturas, e a fração cresce com o nível. Força de kN no 1º subpasso, Δv da pelve ~1,2–1,4 m/s (ordem de grandeza, sem dinâmica), multa saturada em qualquer limiar. A cláusula de arremesso do `caixa_largada` (v_rel > 1,2 m/s com a caixa a mais de 0,18 m do alvo) liga em 36–48% dos toques | código + medido | confirmado com ressalvas |
| A4 | O CARREGAR da cadeia C é uma estátua: com twist zero, o rastreio ×3,5 da coluna CARREGAR (`knobs.py:1311-1312`, pensado para andar com a caixa) paga 14/s por ficar parado. Fechar vale −0,8 a +5,4 contra ficar (γ 0,99). Com rastreio ×1 no CARREGAR parado, fechar vale +15 a +21 | código + medido | novo |
| A5 | Poucas C chegam ao BOTAR na fase 4: entre ~1% e ~18%, conforme a premissa (o log não mede direto). O fecho do BOTAR ocorre em ≤ 0,46% das C. Na fase 1 o `s_C` teve média de 4–7% | medido (log), inferido (fração) | parcial: a faixa é larga |
| A6 | No BOTAR, descer paga +0,29 a +0,37/s por cm em todas as rotas (só braços, agachar, dobrar o quadril). Não há bloqueio físico nem limite de equilíbrio no ponto congelado (CoM com ~6 cm de margem, nada encosta na laje) | medido | confirmado |
| A7 | Para descer pelos braços a mão tem de girar 35–57° sobre a face da caixa (pega de berço, punho 3,4 cm abaixo do fundo), e nenhum termo paga esse giro. Com a pega rígida, a caixa tomba junto e a rota tem vale | medido | confirmado, com correção dos números |
| A8 | O `upright` (peso 1, fora da tabela por estado, `knobs.py:1294`) põe o ótimo do tronco em ~3°, contra 24° da IK do BOTAR. Isso contradiz o §3 e o §4.5 (o tronco deve inclinar para alvo baixo) e premia a cintura compensando a pelve | medido | confirmado |
| A9 | O registrador monta a observação com `giro_w` = 0 (`registra_juntas.py:370` → `pilota.py:211-214`). No treino o comando publica o giro de correção do tombo em todo regime (`comando.py:2097-2106`). O ator em play fica CEGO ao tombo; a sonda mostra correção de +45 a +56° nas palmas no CARREGAR. Todo tombo de caixa medido em play está confundido | código + medido | novo, do cético V2 |
| A10 | O `5000_c.csv` bate com a dinâmica do `model_5200` (RMSE de qacc 27 contra 84 do `model_5000`) | medido | novo |

## 2. Correções de afirmações minhas anteriores

| Afirmação anterior | Correção |
|---|---|
| "O `passo_final` alto mostra que a C chega ao BOTAR e não fecha" (spec do teto do `p_C`, §1) | O `passo_final` médio da C fica em 0,5–0,6, quase todo no passo 1 (CARREGAR). A maioria das C não chega ao BOTAR |
| "O congelamento vem de falta de equilíbrio e da coxa apoiada na laje" (lote zero18, §C) | Refutado no ponto congelado: CoM com ~6 cm de margem e nenhum contato. A coxa encosta no PEGAR e no CARREGAR (~1 mm), e não no BOTAR |
| "A palma escorregar custa 5,7× o ganho de descer" | Só se as duas palmas saem 1 cm. Com uma palma, 1,5–2,8×. O escorregão observado acontece parado também, e não é custo de descer |
| "A caixa tomba de 19° a 45° no CARREGAR" (plays desta sessão) | Confundido pelo A9: o play roda cego ao tombo |
| "A política nunca amostrou o fecho do BOTAR" | Amostrou na fase 1 (`s_C` médio 4–7%). Deixou de amostrar a partir do fim da fase 1 |

## 3. O que o plano zero18 faz pelo BOTAR

| Item | Efeito no BOTAR |
|---|---|
| 1.1 sorteio do twist, 1.2 métrica de giro, 1.3 `pose` parado | neutro: a cadeia C inteira roda com twist zero |
| 1.4 multa de contato de 5 a 30 N | inerte no ponto congelado; o toque do teleporte já satura em qualquer limiar |
| 1.5 teto do `p_C` em 0,30 | protege o nível da pega, mas corta a prática do BOTAR para ~0,45× |
| 1.6 postura de pé com caixa (IK) | indireto: a referência do BOTAR em h = 0,65 não muda |
| 1.7 métrica `caixa_no_carregar` | mede o estado CARREGAR, que mistura a cauda de B andando; não isola o elo da C |
| 1.8 registrador | útil, mas falta o `giro_w` (A9) |

## 4. Opções de conserto

Todas precisam de spec e de "implementa". Custos estimados pelos agentes.

| Id | Mudança | Ataca | Enunciado | Linhas | Risco |
|---|---|---|---|---|---|
| R1 | DECISÃO DO DONO (01/10): fica a DR atual do BOTAR (laje 0,50 ± 0,10 em x, ± 0,10 em y, topo0 ± 0,10; alvo ± 0,05), que simula "parou em frente à mesa de destino". O conserto é o sorteio em x só afastar a laje: dx ∈ [0; 0,10]. Medido (`scratchpad/r1_guarda/guarda.py`): toque 39% → 0,6% de pé no default; 33% → 4,7% na pose do fim do CARREGAR da zero17. O alvo vai de 0,30–0,40 m para 0,35–0,40 m à frente. Uma guarda exata pela perna dá 0%, mas empurra a laje até 15–23 cm com o robô agachado. As opções "laje girada com o rumo" e "mesma mesa com altura nova" foram descartadas | A3 | §2; §3 Contato; §4.5 entrada | 1 | o toque que sobra vem da postura agachada |
| R2 | IMPLEMENTADO 01/10 (variante b, sem commit, smoke pendente; `specs/g1-limpo-rastreio-carregar-elo.md`): rastreio ×1 só no elo CARREGAR aberto da cadeia C; a cauda de B/R e a C desviada seguem em ×3,5 | A4 | §3 "sem estátua"; §4.4 | +111 −6 (47 de smoke) | o crítico não separa o elo da C da cauda parada das fases 1–2 |
| R3 | `upright` na tabela por estado, com a coluna BOTAR reduzida (medir a derivada antes do peso) | A8 | §3; §4.5 | a medir | a memória da bloco20 registra que `upright` ×4 no BOTAR matou o `s_C` — a direção aqui é a oposta |
| R4 | BOTAR raso primeiro: topo = teto − U(0; prof), com `prof` crescendo pelo `s_C` ou por fase. Tira o BOTAR do nível do PEGAR | A1, A2, A5 | §1 ordem de aprendizado; §4.5 | +10 a +12 | lajes acima de 0,55 m ficam fora do §2 (decisão do dono); a folga tem de passar de ~5 cm por causa da pega de berço |
| R5 | Resume do `model_4650`, anterior à inversão do crítico, com as sondas A1/A2 como alarme por checkpoint | A1, A2 | — | 0 no repo | sem R1–R2, a inversão pode se repetir |
| R6 | Métricas sem peso por episódio C: entrou no CARREGAR, chegou ao BOTAR, caiu na reconferência; terminações nos 10 passos depois da abertura do BOTAR; `fell_over` por elo. Registrador com `giro_w`, espera publicando o elo interno, laje fixa na C | medida | — | +20 a +30 | nenhum no treino |

Não recomendadas agora:
- SEGURA do BOTAR (depois do fecho, mãos na caixa): ataca o pós-fecho, e a trava está antes.
- Dois sub-elos "descer até pairar" e "apoiar": cria o atalho de pairar (proibido no §4.5)
  e custa 20–40 linhas; R4 dá o mesmo efeito de visitas com ~10.
- Subir o `p_C` ou afrouxar o piso do σ_alcance: sem R1–R2, mais C só gera mais estátua e
  mais chutes.

## 5. Outras lacunas registradas

- A espera com caixa publica o elo interno, e o `VALIDA` saiu da observação: o crítico não
  distingue a espera do elo aberto (`comando.py:838-840`, `observacoes.py:106-111`).
- O `PPOPorElo` agrupa a vantagem pelo elo interno: o grupo CARREGAR mistura o elo parado da
  C com a cauda de B que anda (`algoritmo.py:176-177, 209-223`).
- O degrau do freio é catraca: na fase 4 o nível caiu de 2,85 para 1,48 e o freio ficou em
  2,3, o maior preço do log (`curriculo.py:165-176`).
- A laje do BOTAR herda o nível do PEGAR (topo0 ± 0,10), contra o §2 (alturas independentes).
- O treino roda com `impratio` 2,0 e o play com 1,0: o contato do treino é 2× menos mole.
- O horizonte efetivo do GAE é ~0,34 s (γ 0,99, λ 0,95): o ganho de apoiar e fechar só chega
  pelo crítico.
