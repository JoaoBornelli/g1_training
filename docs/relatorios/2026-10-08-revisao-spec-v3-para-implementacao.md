# Revisão da spec v3 para a implementação

Revisão de 08/10, a pedido do dono: "quero revisar a spec v3 para começar a me organizar para a
implementação". Leitura inteira de `specs/g1-v3.md` (937 linhas) depois das decisões de 08/10
(§6.5 limites de junta, §8.1 tabela do agachamento, §9.2 rampa a 95%, §9.3 cercas de postura).

Fontes: `specs/g1-v3.md`, `vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md`,
`docs/planos/2026-10-02-percepcao-lidar-na-v3.md`, `docs/relatorios/2026-10-05-revisao-spec-v3.md`,
`g1_limpo/ik/agachado/ref_agachado.txt`.

## 0. Veredito

A spec está coerente nas decisões de hoje, com quatro resíduos mecânicos (seção A). Ela ainda não
está pronta para a Parte 1 por três motivos, em ordem de bloqueio:

1. **Seis decisões do dono em aberto** (seção B). Duas bloqueiam a Parte 2 (nível do chão;
   `k_postura` no Φ); as outras bloqueiam a Parte 3 e a 4.
2. **A Parte 0 precisa ser refeita** (seção C): quatro itens novos (pesos das cercas, peso da
   rampa, poses seguráveis sob torque, poses dos alvos sem pose) e dois itens já feitos.
3. **O plano de percepção é o caminho mais longo** (seção D): a Parte 1 só começa depois das
   fases 1 a 4 do plano do lidar, e o custo dos 882 canais do mapa é medido antes do treino.

Risco novo que a leitura inteira mostrou (seção E, item 1): os geradores e as tabelas de IK
vivem em `g1_limpo/ik/`, fora do repo e fora do git.

## A. Resíduos das edições de 08/10 (correção mecânica, sem decisão nova)

| onde | o que ficou | correção |
|---|---|---|
| §8.1 "Mão fixa em relação à caixa" | o parágrafo descreve a tabela `theta-30` inteira (14 de 15 fecham, tronco por tamanho, pelve da IK de 01/10) e só no fim diz que as pegas saíram | reduzir ao que ainda vale: a mão a −30° e a pose de pé com caixa; mover o resto para o registro de decisões |
| §16, Parte 0, item 9 | "Largura da faixa do CoM, a partir das poses da IK" | FEITO em 08/10: piso de 3 cm à borda das solas (§9.3); marcar |
| §16, Parte 0, item 10 | "a pega que falha (laje de 0,04 m, caixa de 14 cm)" | continua a mesma pega, agora com o pé a 3,5 cm (era 5,2); manter |
| §18, item 2 e item 6 | "a faixa inclui o CoM, centrado no meio dos pés"; "o batente virtual das juntas de curso largo entra no refino" | SUPERADOS em 08/10: o CoM é piso à borda (§9.3); a rampa pela banda cobre o curso largo desde o começo (§9.2); marcar os dois |

Corrigido já na revisão: a linha "aproximação da pose de referência" da tabela do refino (§11.5)
estava viva; agora está riscada, com a nota de que as cercas não têm degrau.

APLICADO em 09/10: os quatro itens da tabela (parágrafo "Mão fixa" reduzido; item 9 FEITO; item 10
com as pegas que não fecham na tabela de 09/10; §18 itens 2 e 6 SUPERADOS).

## B. Decisões do dono em aberto

| # | decisão | onde | o que bloqueia | o que a spec diz hoje |
|---|---|---|---|---|
| B1 | **DECIDIDO (dono, 08/10): NÃO.** O chão fica só como referência na tabela; a laje de 0,04 m segue o nível mais baixo; a cena não precisa do caso "sem laje". Texto original: **Nível do chão no PEGAR e no BOTAR.** A tabela da IK tem a caixa no chão (`topo` = 0), e a pose do dono nasceu para "pegar e botar uma caixa no chão". Os níveis da §11.2 param em 0,04 m | §11.2, §12 | Parte 2 (níveis do PEGAR) e Parte 3 (níveis do BOTAR); a cena precisa do caso "sem laje" na pega | PEGAR: (0,55; 1 kg) → … → (0,04; 5). BOTAR: 0,55 → … → 0,04. Nada do chão |
| B2 | **DECIDIDO (dono, 08/10): opção 2.** A pelve sai do Φ e vira cerca de ±8 cm na §9.3 (a sexta). Texto original: **`k_postura` dentro do Φ.** O Φ tem "a atração da pelve à IK, cheia desde o começo". Com as cercas da §9.3 (tronco, CoM) e sem tracking, a pelve continua como atração dentro do Φ, ou vira cerca (±8 cm, proposta de 08/10 não aprovada), ou sai? | §9.1, §8.2 | Parte 2 (`potencial.py`) | `Φ = k_principal · (½ + ½·k_postura) · g_invariantes · g_faixa` |
| B3 | **DECIDIDO.** Pelve: B2. Recuo: vira cerca de ±8 cm (dono, 08/10; a sétima da §9.3). Largura dos pés: SEM cerca (dono, 09/10: "para algumas poses ele precisa abrir a base"). Texto original: **Cercas da pelve, do recuo e da largura dos pés.** Propostas em 08/10 com os valores da tabela (pelve ±8 cm; recuo ±8 cm; largura ≥ IK − 8 cm). O dono aprovou só as cinco da §9.3 | §9.3 | — | sete cercas; largura livre |
| B4 | **DECIDIDO (dono, 09/10): recompensa pelo menor aperto que segura**, `w·exp(−aperto/F_ref)`, só com a caixa presa; teto por massa rejeitado. Texto original: **Aperto.** O dono vai desenhar (05/10) | §8.2 | Parte 4 (refino); a Parte 2 roda sem força mínima; `w` sai da Parte 0 | desenhado |
| B5 | **REMOVIDO (dono, 09/10): o gate `F_solta` saiu**; o invariante de B2 a B4 (caixa parada depois de apoiada) cobre o arrasto. No mesmo dia tinha virado 2 N fixo. Texto original: **`F_solta`.** Proposta 0,25·m·g, a calibrar junto com o aperto | §8.2 | Parte 3 (gate do B2) | 2 N |
| B6 | **Medir a flexão do `hip_pitch` no robô** (145° da tabela contra 154° da página) | §6.5 | nada no simulador; a tabela vale até a medida | PENDENTE |

Decisão implícita a confirmar: as cercas valem em TODA tarefa, inclusive no ANDAR, onde a
receita da Unitree já tem "tronco plano" (−5) e "desvio de `hip_yaw`" (−1). No ANDAR a cerca do
tronco coincide com o tronco plano (IK = 0°) e a do giro da coxa dobra com o desvio de `hip_yaw`.
Dois termos sobre a mesma grandeza: o dono aceita, ou a cerca só vale na manipulação?

## C. Parte 0 refeita (checagens sem GPU, antes de qualquer código)

A lista da §16 com o estado de hoje (itens 11 a 16 levados à §16 em 09/10). Os itens novos vêm da §9.3 e da tabela do agachamento.

| # | item | estado | observação |
|---|---|---|---|
| 1 | Φ ao longo da interpolação P0–P4 e B1–B4 | aberto | depende de B2 |
| 2 | derivada da proximidade no estado inicial de cada alvo | aberto | |
| 3 | escada de custos e `B_alvo` | aberto | precisa dos pesos das cercas (item 11) |
| 4 | balanceadores em simulação de mesa | aberto | puro; pode começar hoje |
| 5 | batente | feito 02/10 | |
| 6 | rastreio em comando baixo | aberto | nos CSV de play que já existem |
| 7 | limite mole a 0,95 contra as tabelas da §8.1 | **feito 09/10** | as 21 poses ficam a ≤ 90% do curso (`hip_pitch` no chão e na laje de 0,04 m); a rampa não cobra nenhuma |
| 8 | período da marcha pela velocidade | aberto | |
| 9 | largura da faixa do CoM | **feito 08/10** | piso de 3 cm à borda (§9.3) |
| 10 | poses dos alvos sem pose (P0, P1, P3, B1, B2, B3) e a pega que falha | aberto | P0 e B1 saem da pose da pega com as palmas deslocadas; P3 e B2 da mesma pose com a caixa 5 cm acima |
| 11 | **NOVO: pesos das sete cercas e da rampa** | **medido 09/10** (`cercas_nos_plays.py`); sola só fora da marcha (DECIDIDO 09/10); pesos esperam o item 3 | derivada no ponto de operação dos CSV de play (21700, 23250, 24800): quanto cada cerca cobraria por passo hoje, contra o que a tarefa paga |
| 12 | **NOVO: a IK é segurável sob torque?** | **feito 09/10** | em física com os servos PD da cena (`segura_pose.py`): cintura 91–99% no chão e na laje de 0,30 m (satura em 2 poses), 85–91% em 0,04 e 0,15, 67–83% em 0,45 e 0,55; punho 48–57%. Antes: na laje de 0,30 m a cintura pede 94–97% do torque ESTÁTICO com 5 kg (tabela de 09/10); o movimento soma a isso. Rodar as 21 poses em física (MuJoCo clássico, `teste_queda.py` ou o `pilota` da v3) com o PD da Unitree e 5 kg, e ver se o robô segura a pose por 2 s |
| 13 | **NOVO: a tabela contra as cercas** | aberto | tabela de 09/10: a pose do dono paga só no CoM (na borda); a IK para 0,2–1,0 cm dentro da cerca do CoM em 10 poses (aceito, 09/10); MEDIDO 09/10: paga também o torque (cintura 81–97% do chão à laje de 0,45 m); as outras cinco não pagam; falta o custo com os pesos do item 11 |
| 14 | **NOVO: o ANDAR com duas cercas sobre a mesma grandeza** | **feito 09/10**: custo zero no ANDAR (tronco p95 4–7°, hip_yaw p95 ≤ 9°); os dois termos ficam; a sola é a única cerca que toca a marcha (toe-off) e fica desligada nela (09/10) | medir nos CSV do ANDAR o que a cerca do tronco e a do giro cobrariam por cima do tronco plano e do desvio de `hip_yaw` (decisão implícita da seção B) |
| 15 | **NOVO (09/10): aperto** | aberto | derivada de `r_aperto` a 250 N; o ganho de afrouxar nunca cobre a perda da caixa; fixa `w` e confirma `F_ref` |
| 16 | **NOVO (09/10): `F_solta`** | **sem objeto (09/10)** | palmas na posição da IK (1–2 mm dentro da face): 2–8,5 N por face, ~3 N/mm; 2 N total = sem contato; ABERTO: distância numérica de "nas faces" no B2 |

## D. Dependências e ordem

```
plano de percepção, fases 1–4 (filtro puro; no sim do real)   <- caminho mais longo
        |
Parte 0 (itens 1–14, sem GPU) ----------------------------------+
        |                                                        |
Parte 1: pacote g1_v3, locomoção, lidar no treino (fase 5), custo dos 882 canais medido (5c)
        |
Parte 2: cadeia B   <- B1, B2, B3
        |
Parte 3: cadeia C   <- B5
        |
Parte 4: refino     <- B4
```

- A Parte 0 e as fases 1 a 4 do plano de percepção não dependem uma da outra: podem andar
  juntas.
- A Parte 1 copia as tabelas para `g1_v3/ik/`. As tabelas e os geradores precisam entrar no
  repo antes disso (seção E, item 1).
- O mapa de elevação (882 canais, sem histórico) entra no ator desde o começo (dono, 06/10). O
  custo do passo e o tamanho da primeira camada se medem na fase 5c antes de treinar. Se o custo
  for alto, a decisão volta ao dono antes da Parte 1.

## E. Riscos vistos na leitura inteira

1. **Geradores e tabelas fora do repo — FEITO em 08/10.** Tudo foi movido para `g1_limpo/ik/` (scripts na raiz da pasta; `mao_fixa/`, `agachado/`, `entre_pernas/`, `poses_manuais/`, `prateleira/`, `sombra/`, `fotos_v3/`); os imports viraram `g1_limpo.ik.*` e os caminhos são relativos ao arquivo. Texto original: `ik_agachado.py`, `ik_mao_fixa.py`, `monta_pose.py`,
   `ik_entre_pernas.py`, `gera_de_pe.py`, `ik_cintura25.py`, `ref_agachado.npz`,
   `ref_v3_de_pe_theta-30_tamanhos.npz` e `poses_manuais/agachado 00.npz` vivem em
   `g1_limpo/ik/`, sem git. A spec §8.1 aponta para esses caminhos. Proposta: mover para
   `g1_limpo/ik/` agora (os scripts importam `g1_limpo.ik.gera_botar` e `ik_mao_fixa`), e na
   Parte 1 copiar o que fica para `g1_v3/ik/`.
2. **Torque da cintura.** A cerca de torque (80%) cobra a própria IK nas lajes de 0,15 a 0,45 m.
   Isso é desejado (o treino fica mais de pé que a IK), mas se a pose não for segurável (item
   12 da Parte 0), o alvo P2 nessas lajes pode ser inalcançável com 5 kg, e o balanceador vai
   mandar envs para um nível que não fecha. A Parte 0 responde antes.
3. **A cena precisa de "sem laje".** Hoje a laje sai da cena só no CARREGAR [C39]. Com o nível do
   chão (B1), a pega e a soltura acontecem sem laje; `posiciona_cena` e `afasta_cena` precisam
   do caso. Sem B1, nada muda.
4. **O aceite "sem batente" (§15) coincide com o limite mole (95%).** Uma política que estaciona
   uma junta em 95% passa no aceite e paga zero. Se o dono quer folga entre o aceite e o custo,
   o aceite desce para 93%.
5. **`k_postura` e a observação.** O ator recebe "altura da pelve (1), inclinação do tronco (1)"
   do alvo macro (§6.1). Se B2 tirar a pelve do Φ, o canal da pelve na observação continua
   válido (ele é informação, não recompensa), mas vale decidir junto.
6. **Contagem de código (§17) é de 02/10.** Ela não conta a §19.1 (observação nova), o mapa de
   elevação, o estimador com a forma, nem as cercas. Refazer na Parte 1, como a spec já diz.
7. **`pilota.py` e `registra_juntas.py` da v3** (§5) são novos e sem spec própria: o laço de rumo
   k = 2, o relógio da marcha, o gerador de alvos e o estimador no laço. Eles são o portão de
   quase todo aceite da §15. Uma spec curta deles evita que o coder invente.

## F. Ordem sugerida para o dono se organizar

1. Responder B1, B2 e B3 (uma sessão de decisão; nenhuma precisa de medida nova).
2. Mover os geradores e as tabelas de IK para o repo (E1) e commitar.
3. Aplicar as correções mecânicas da seção A.
4. Parte 0: itens 4, 6, 7, 11, 12, 13 e 14 não dependem de B1–B3 e podem começar; os itens 1,
   2, 3 e 10 esperam B2.
5. Em paralelo, o plano de percepção, fases 1 a 4.
6. Portão da Parte 0 com o dono; depois a Parte 1.
