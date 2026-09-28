# O `ANG` do PEGAR era cego ao tombo (28/09)

## O defeito

Até 28/09 o PEGAR e o CARREGAR usavam o regime `FACE_CONGELADA`: o `ANG` comparava a
normal de UMA face lateral (−x) com a normal dela na abertura. Um vetor não muda quando
a caixa gira em torno dele. A pega bimanual segura as faces laterais, e o giro mais fácil
é em torno do eixo palma-a-palma — exatamente a normal lateral. O `precise_ori` e o
`alinhado` do fecho não viam esse tombo, e cobravam a guinada, que o dono libera.

## Medições

`model_2000` da `zero09` (retomada da `zero08`, `precise_ori` já híbrido), currículo
natural, 64 envs, 1000 passos, PEGAR_COM (912 amostras):

| grandeza | p50 |
|---|---|
| tombo real (Z da caixa contra a vertical) | 93° |
| `ANG` publicado | 62° |
| guinada da face lateral | 46° |
| razão `precise_ori / staged` | 0,057 |

Mesmo checkpoint, PEGAR forçado, 59 643 amostras (`mede_juntas.py`, código do HEAD):

| grandeza | p10 / p50 / p90 |
|---|---|
| tombo no mundo | 68° / 81° / 94° |
| tombo no frame do torso | 69° / 83° / 95° |
| giro da caixa DENTRO da mão esquerda desde a pega | 4° / 12° / 37° |

O torso não explica o tombo, e o escorregamento na mão explica pouco. A caixa gira COM
os antebraços ao erguer: `elbow` −0,73 / −1,11 rad (p50, esq./dir.), `wrist_roll`
−0,78 / −0,56, `wrist_pitch` só −0,17 / −0,23. Correlação com o tombo: `elbow` −0,53 /
−0,40, `left_shoulder_roll` −0,49, `right_shoulder_yaw` +0,57.

Leitura: endireitar pelo punho pediria ~1,3 rad de `wrist_pitch`, contra a faixa de
0,6 rad no PEGAR_COM. A rota sem custo é erguer mantendo a atitude do antebraço
(ombro + cotovelo), que a faixa `braco_pos` (1,5) não cobra.

## A mudança

Requisito do dono: a face de baixo continua para baixo; a guinada é livre.

- `FACE_CONGELADA` sai. Todo elo menos o REORIENTAR usa `FACE_DE_PE`: o eixo que estava
  para cima na mesa (`_cima_b`, capturado por `_captura_cima`) contra a vertical.
- A captura roda na abertura do elo e quando a tarefa liga, só fora do CARREGAR e do
  BOTAR: no avanço de cadeia eles herdam o eixo do PEGAR (capturar ali pegaria a caixa
  tombada — o defeito de 21/09). No reset o eixo volta ao Z.
- σ no regime de pé = max(erro inicial × fator, `tol_ang_deg`).

## O que fica aberto

- Avaliação por 3 agentes (28/09): a mudança conserta o portão, mas provavelmente não
  destrava o PEGAR sozinha. Todo termo de segurar paga igual com a caixa tombada; o
  `precise_ori` é ~2% da renda do PEGAR_COM.
- Campo: o fecho do BOTAR não prova pouso estável (`apoiada` a 50% do peso); 25° não tem
  base física para conteúdo desconhecido; nada mede o giro da caixa; o estimador do real
  não vê o tombo na mão sem a tag ou força/torque de punho.
