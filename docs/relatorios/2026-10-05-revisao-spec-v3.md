# Revisão da spec v3 contra o enunciado

Revisão de 05/10, a pedido do dono: "verificar se ela está atendendo todos os objetivos que
eu defini na documentação de referência".

Fontes: `specs/g1-v3.md` (a spec), `specs/g1-limpo-comportamento.md` (o enunciado),
`specs/g1-v3-comportamento.md` (CV), `vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md`, os relatórios de 05/10
(`2026-10-05-dr-de-atrito-pega-e-transferencia.md`, `2026-10-05-dr-para-a-v3.md`) e a receita
da Unitree no commit `4960b84` (`velocity_env_cfg.py`, `rewards.py`, `unitree.py`).

## 0. Veredito

Atualização de 05/10: o dono respondeu item a item (`vault/G1/Abordagens/PPO/Currículos/C4 v3/g1-v3-decisoes.md`, "Respostas
de 05/10 à revisão"), e as respostas estão aplicadas na spec. Em aberto: o aperto (B4), que o
dono vai desenhar, e a atualização do enunciado (B9).

A spec cobre o objetivo do enunciado na estrutura: um modelo, as cadeias, os alvos macro, o
potencial, o balanceador, o refino e o estimador. Ela ainda não está pronta para implementar:

1. A §19 (receita da Unitree) entrou por cima e não foi propagada. Dez trechos da spec
   contradizem a §19 (seção A).
2. Cinco termos da receita da Unitree quebram regras do enunciado quando valem fora do ANDAR
   (B1 a B5).
3. A decisão de 05/10 (palma de plástico) contradiz o enunciado e a §8.2 (B4).
4. O enunciado está desatualizado em nove pontos que a v3 já decidiu (B9).

## A. Contradições internas: a §19 não foi propagada

Correção mecânica, sem decisão nova. A §19 diz que substitui a §9.2, a §9.3 e o comando na
locomoção, mas o resto da spec segue o desenho anterior.

| # | trecho | diz | a §19 diz |
|---|---|---|---|
| A1 | §5 (`exporta_cena`, `pilota`), §6.1, §16 Parte 1 | observação de 122 canais | 119 por passo, 595 com histórico (§19.1) |
| A2 | §9.2, §9.3 | `giro_sem_gingado`, freio, `AlturaDeBalanco`, `soft_landing`, `upright`, rastreio 2,0, `limite_de_junta` −1,0 | rastreio 1,0 e 0,5; `dof_pos_limits` −5 (mole 0,9); marcha por fase; sem freio |
| A3 | §11.2, linha ANDAR | vx até 2,0 m/s; wz até 1,6 rad/s; giro no lugar com \|wz\| ≥ 0,75 | linear 0,25 → 1,0 m/s; giro 0,25 → 0,5 rad/s. O giro no lugar a 0,75 nunca abre |
| A4 | §5 (`velocidade_por_regime`, `garante_freio`), §11.4 | freio por nível | o freio sai |
| A5 | §15 "locomoção", §16 Parte 1 | portão `razao_marcha` ≥ 0,50 | o portão da marcha sai |
| A6 | §15 "rumo", §16 Parte 1 | laço de rumo k = 2 no treino | o laço sai do treino; o k = 2 fica no driver |
| A7 | §10 | só `fell_over` a 70° | ANDAR e CARREGAR: 0,8 rad e base abaixo de 0,2 m |
| A8 | §12 | empurrão a cada 1–3 s; atrito do pé 0,3–1,2; sem massa do tronco | empurrão a cada 5 s; atrito 0,3–1,0; tronco −1 a +3 kg |
| A9 | §6.3 | ação "a do molde" | escala 0,25, offset no default da Unitree, ganhos do PD da Unitree |
| A10 | §11.5, §17 | linha do `air_time`; contagem com os módulos que saíram | o `air_time` sai; a contagem muda |

Ponto sem resposta em A2: no PEGAR e no BOTAR, qual limite de junta vale, a rampa da tabela do
ZERO ou o `dof_pos_limits` da Unitree? Os dois juntos cobram duas vezes.

## B. Conflitos com o enunciado

### B1. Estátua no PEGAR e no BOTAR (enunciado §3 "Sem estátua"; catálogo §5, linha "Estátua")

A receita da Unitree paga por passo: `alive` +0,15/s, e o rastreio `exp(−erro²/0,25)` paga
1,0 + 0,5 por segundo com comando zero e o robô parado. A §9.2 diz "o rastreio só cobra
deriva", mas a forma da Unitree paga renda.

O PEGAR e o BOTAR não têm relógio, e o episódio acaba no fim da cadeia (§7). Se esses termos
valem ali, cada segundo de estátua no PEGAR ou no BOTAR paga ~1,65. A §19, conflito 3, tira
desses trechos só os termos posturais.

Proposta: no PEGAR e no BOTAR, `alive` zero e rastreio só como custo de deriva. Termo positivo
por passo só nos trechos com duração fixa (ANDAR, esperas, CARREGAR).

### B2. Contato fora dos pés (enunciado §3 "Contato")

O termo da Unitree conta todo corpo sem "ankle" no nome, com limiar de 1 N. A §19 exclui as
mãos. Resultado:

- a mão no chão e na laje fica sem custo, e o enunciado diz "as mãos nunca" tocam o chão;
- o pulso, o antebraço e o tronco na caixa pagam −1 por passo. O enunciado permite o tronco
  em carga alta. A medida de 05/10 achou ~110 N por lado no pulso e no tronco.

Proposta: filtrar pelo parceiro do contato, e não pelo corpo. Contato com a caixa fica fora da
conta; contato da mão com o chão e com a laje fica dentro.

### B3. Passo miúdo em velocidade baixa (enunciado §3 "Pés"; pedido do dono "independente da velocidade")

A §19 diz que a marcha por fase resolve o passo miúdo. Isso está errado em velocidade baixa,
e o erro é meu. O período de 0,8 s fixa 2,5 passos por segundo em qualquer velocidade:

| velocidade | passo |
|---|---|
| 0,25 m/s | 0,10 m |
| 0,5 m/s | 0,20 m |
| 1,0 m/s | 0,40 m |

A marcha por fase resolve a cadência alta sem voo, que era o hack do catálogo. Ela não dá passo
de tamanho normal abaixo de ~0,7 m/s. O `air_time` saiu do refino com base nesta afirmação.

Opções: aceitar o passo curto em baixa velocidade; período que cresce quando a velocidade cai
(sai da receita pura); ou devolver o `air_time` com duração mínima ao refino.

### B4. Palma de plástico (decisão de 05/10) contra o enunciado e a §8.2

A decisão de 05/10 diz: pagar o resultado da pega, e não a força nos pads; caixa de pé só no
BOTAR; caixa inclinada contra o peito e apoio no antebraço permitidos com 5 kg e μ 0,3.

| onde | diz hoje |
|---|---|
| enunciado §2 e §3 | face de cima normal ao solo "o tempo todo" |
| enunciado §4.3 | proibido "pegar com o antebraço" |
| spec §8.2, P2 | força normal nas DUAS mãos entre `F_ref` e k·`F_ref` |
| spec §8.2, invariantes | caixa nivelada de P2 a B2; força na faixa de P2 a B1 |
| spec §9.2 | `CarregarComQualidade` exige a caixa nivelada |
| spec §11.5 | aperto 8 → 2,5 × `F_ref`; o relatório propõe dobradiça k = 3 sobre a força total, com o `F_ref` do pior μ da faixa |
| spec §12 | atrito da caixa só no refino; o relatório e a memória de 05/10 põem a DR de μ por nível, depois da recompensa do resultado |

Decisão do dono: mudar o enunciado (inclinação no CARREGAR, antebraço em carga alta) e a §8.2,
ou manter o enunciado e limitar a decisão de 05/10.

### B5. Duas poses de pé (enunciado §3 "De pé, como um humano" e "Troca de tarefa")

| junta | default da Unitree (offset da ação, alvo do desvio no ANDAR) | IK de pé sem caixa (B4, aceite da espera) |
|---|---|---|
| joelho | 0,30 | 0,16 |
| cotovelo | 0,97 | 0,60 |
| ombro pitch / roll | 0,30 / 0,25 | 0,20 / 0,20 |
| punho roll | ±0,15 | 0 |

Na troca BOTAR (B4) → ANDAR parado, o cotovelo anda 0,37 rad. O enunciado pede pose que muda
pouco na troca e joelho perto de 0,15 parado.

Proposta: uma pose de pé só. O alvo do desvio da Unitree passa a ser a IK de pé, ou a IK de pé
é refeita com os braços do default da Unitree.

### B6. B3 não exige "sem contato com o robô" (enunciado §1 "Sucesso do ciclo", §4.5 "Fecho")

O B3 pede palmas a 5 cm ou mais das faces. O pulso ou o antebraço ainda podem tocar a caixa.
Proposta: o B3 exige também força total robô → caixa igual a zero, com o sensor do B2.

### B7. Caixa arrastada ou derrubada antes da pega (enunciado §4.3, proibido "arrastar além do ajuste")

Nada mede o deslocamento da caixa na laje de P0 a P2. Os alvos seguem a caixa, então empurrá-la
não custa. A caixa perdida só arma na primeira pega: derrubar a caixa da laje antes dela não
termina o episódio.

Proposta: invariante de P0 a P2, deslocamento da caixa abaixo de meia aresta (a regra do
REORIENTAR); caixa fora da laje antes da pega conta como caixa perdida.

### B8. Transições do exemplo do galpão (enunciado §1, passos 4 e 8; §4.4 "frear em poucos passos")

Nunca treinadas: CARREGAR andando → CARREGAR parado → BOTAR, e giro no lugar → andando dentro
do CARREGAR. Na B o comando não muda no trecho; na C o parado vem só depois do PEGAR. O BOTAR
sempre abre com o robô parado há 0,5–1,0 s. O dono decidiu pela generalização do ANDAR (02/10).
Risco aceito; o aceite da §15 cobre só o frear.

### B9. Enunciado desatualizado

A regra do repo julga toda mudança contra o enunciado. Estes pontos do enunciado contradizem a
v3 decidida:

| enunciado | diz | v3 |
|---|---|---|
| §1 Composições | cadeia com REORIENTAR | REORIENTAR é futuro |
| §1 Ordem de aprendizado | fases do `curriculo-de-cadeia` | dois balanceadores |
| §1 Fora do objetivo; §2 Caixa | oráculo; "estimador vem depois"; lidar fora | estimador simulado desde o começo |
| §2 Caixa | aresta 0,07–0,13 m | meia aresta |
| §3 Progressão na cadeia | renda congelada | potencial |
| §4.2 Espera | 0,5–1,5 s; espera final "até o fim" | 0,5–1,0 s; espera final de 1,5 s |
| §4.5 Proibido | "deitar o tronco para alcançar" | "no lugar de estender o braço" |
| §5, linha `air_time` | "marcha só forma com incentivo de voo" | errado [X1] |
| §6 Decisões fechadas | "torque e energia: por enquanto não" | a receita da Unitree tem energia −2e−5 |

### B10. A massa no gerador de alvo (enunciado §2 "O robô não conhece essas alturas")

O P4 usa a pose de pé com caixa da massa `m` (§8.1). O controlador (§6.4) e o estimador (§13) não
dão a massa no robô real. O efeito nos canais do ator é pequeno: pelve 0,780–0,789 m, tronco 0.
Proposta: os canais do ator saem de uma pose fixa; a massa entra só no Φ e no crítico.

### B11. Sem correr no PEGAR e no BOTAR (enunciado §3 "Movimento controlado")

A CV §10 garante "sem correr" pelo freio por nível, validado pelo dono (decisão 4). A §19 tira
o freio da locomoção e não diz se ele fica na manipulação. Sem ele, só a velocidade de chegada
da caixa e o `joint_vel` −0,001 da Unitree seguram a pressa.

### B12. BOTAR depois do B4, no play (CV §4.2)

No treino, a cadeia troca para ANDAR parado no fecho do B4. No play, o one-hot fica em BOTAR até o
controlador trocar. O estado "BOTAR depois do B4" nunca é treinado. Proposta: a troca acontece
0–1,5 s depois do B4, sorteado.

## C. Lacunas menores

| # | lacuna | proposta |
|---|---|---|
| C1 | a Parte 0 (§16) não lista as poses de pé de 1 e 5 kg a 0,90 m (§18.7; a tabela tem as duas a 0,85), o `F_solta` (§8.2) e a largura da faixa do CoM | itens 6, 7 e 8 da Parte 0 |
| C2 | a amostragem dentro de um nível não está definida (altura × massa; a C pega de 0,30 para cima com a massa dominada) | altura uniforme em [h_k; 0,55], massa uniforme em [1; m_k], independentes |
| C3 | o enunciado §2 vai até a laje de 0 m; a v3 para em 0,04 m | confirmar que 0,04 basta |
| C4 | a §15 não tem o ciclo do galpão, que é o objetivo do produto | aceite: o exemplo do enunciado §1 no play, de ponta a ponta, mais uma volta (pegar, botar, pegar outra) |
| C5 | a §15 não cobre: junta sem batente, BOTAR na laje de 0,04 m, giro de 180° com caixa, empurrão com caixa | um cenário por regra |
| C6 | impacto da caixa contra o próprio robô (enunciado §2) | só a `Sacudida` no refino cobre; aceitar ou medir |

## D. O que está coberto

| enunciado | mecanismo na spec |
|---|---|
| §1 um modelo, troca por one-hot sem handover | §4, §6.4 |
| §1 sucesso do episódio de treino | §11.3 |
| §2 transporte a 0,90 m no mundo | §8.2 P4 |
| §2 tolerância de 0,10 m no BOTAR | §8.2 B2 |
| §3 postura de manipulação, agachar só para alvo baixo, CoM | faixa da §8.2 com o CoM |
| §3 sem batente | rampa desde o começo; terminação no refino (opção A) |
| §3 pega com as duas mãos, pelas faces laterais | P0, P1, P2 e invariantes (ver B4) |
| §3 empurrão | evento da Unitree em todo env |
| §4.3 mergulho, pairar, tombar ao erguer | faixa, potencial, invariantes |
| §4.5 BOTAR: soltar de altura, pairar, ficar apoiado, empurrar depois de soltar | chegada da caixa, potencial, B4, B3 |
| §4.5 BOTAR de um comando (dono, 02/10) | B1 → B4 dentro do BOTAR |
| catálogo §5: pairar, fechar sem avançar, renda congelada, nível sobre competência zero | o potencial, os invariantes e o balanceador com portão (a linha "Estátua" depende de B1) |
