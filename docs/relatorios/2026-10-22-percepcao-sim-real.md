# Percepção da v3: onde o sinal do treino pode divergir do robô real

Data: 22/10/2026. Base: `specs/g1-v3-interface-real.md` (I), `docs/planos/2026-10-14-implementacao-g1-v3.md`
(plano), `specs/g1-v3.md` (S). Só análise; nenhuma solução foi decidida.

## 0. As duas fronteiras

| fronteira | o que o treino simula | o que roda igual no treino e no real |
|---|---|---|
| lidar → mapa (882 números) | os raios crus do MID-360, a odometria | `percepcao_lidar.py` (filtro do corpo, Kalman, recorte) |
| câmera → caixa (11 números) | só a SAÍDA do estimador, com erros sorteados | nada; o box_vision real não roda no treino |

Na primeira, o risco é o raio e a pose que entram no código comum. Na segunda, o risco é a
distribuição de erro inteira.

Gravidade: **A** = a política vê um sinal fora da faixa do treino em operação normal. **M** = fora da
faixa em uma situação comum (giro, PEGAR, superfície ruim). **B** = efeito pequeno ou raro.

## 1. Lidar → mapa

| # | ponto | no treino | no real | efeito no mapa | grav. |
|---|---|---|---|---|---|
| L1 | densidade de pontos | 1 025 pontos por 0,1 s (PD-5) | ~20 mil por 0,1 s (I §2.4 passo 1) | o Kalman usa `σ_m²/n` (plano C1). Com n ~20× maior, a confiança vai a 1 quase na primeira leitura e a altura fica menos ruidosa. O canal de confiança (441 números) sai da distribuição do treino | A |
| L2 | cadência e deskew | 205 raios a cada 20 ms, cada trecho na pose do passo | o SDK entrega pontos com carimbo por ponto; a I §2.4 descreve atualização a cada 0,1 s | se o driver aplica uma pose só ao quadro de 0,1 s: 1 m/s dá 10 cm de borrão; 1 rad/s a 2 m dá 20 cm | A |
| L3 | odometria no giro | erro cresce só com `\|Δp\|` (plano D2b) | o ESKF deriva também no giro, e é no giro que o LIO mais falha | no `giro_lugar`, `\|Δp\|` ≈ 0, então o treino nunca vê erro de rumo; no real, o erro de Δψ gira o mapa e duplica bordas | A |
| L4 | saltos da odometria | passeio aleatório contínuo | relocalização, salto após perda de rastreio | o Kalman funde o salto como um degrau falso; nada no treino parecido | M |
| L5 | inclinação do IMU no mapa | giro sorteado POR LEITURA, σ 0,5° (plano §6.3) | viés lento e correlacionado (o mesmo ±1° fixo que a DR põe na gravidade) | ruído por leitura se cancela na média do Kalman; viés fixo não. 1° dá ~3,5 cm de inclinação falsa a 2 m. O treino usa dois modelos diferentes para o mesmo IMU | M |
| L6 | extrínseca do MID-360 | pose exata do MJCF no `torso_link` | erro de montagem e de calibração | mesmo efeito do L5, fixo por robô; não está na DR | M |
| L7 | superfície | perda aleatória de 1 %, sem ponto falso | buraco em superfície preta, brilhante ou de vidro; multipercurso; ponto falso | o Kalman faz média sem rejeitar outlier: um ponto falso move a célula | M |
| L8 | filtro do corpo fora de 0,7 m | as cápsulas envolvem as geoms que o raycast acerta, então nada vaza | o pé, a mão e os cabos reais diferem das cápsulas | ponto do próprio pé ou da perna vira obstáculo perto da base | M |
| L9 | atraso do caminho do lidar | zero | pacote + transformação + filtro | a DR de atraso cobre só a ação e a propriocepção; o mapa chega atrasado sem treino para isso | M |
| L10 | zero de altura | sites dos pés verdadeiros | FK com encoder (viés ±0,015 rad, ~1 cm na perna) | erro pequeno e constante | B |
| L11 | caixa nas mãos | a caixa fica dentro de 0,7 m e sai; a sombra à frente aparece (82 de 210 células) | igual, se a caixa fica dentro de 0,7 m | coberto | B |

## 2. Câmera → caixa

| # | ponto | no treino | no real | efeito no bloco da caixa | grav. |
|---|---|---|---|---|---|
| C1 | distribuição do erro | números do H5 (mediana 5,3°, perda 2,2 %, viés ±2 cm) | não medida na D435i do G1 | se o erro real é maior, a política age sobre leitura que nunca viu | A |
| C2 | erro sistemático | erro sorteado, independente da geometria | erro preso à face, ao ângulo de vista e à distância | o mesmo erro em toda leitura de uma pose; o treino só vê erro que muda | M |
| C3 | leitura parcial | "visível" = centro no FOV e ≥ 3 de 9 pontos sem oclusão; com isso, a leitura sai boa | com a caixa parcial ou com a mão na frente, o detector entrega um centro ERRADO em vez de nenhum | no P2–P4 (o momento crítico), o treino vê leitura velha e certa; o real pode dar leitura nova e torta | A |
| C4 | sincronia câmera × odometria | a caixa é estática no mundo; o atraso de 33 ms não muda a posição dela | a caixa vai ao mundo pela pose do robô no instante da foto; a cabeça gira com a cintura no PEGAR | erro = velocidade angular × erro de sincronia × distância | M |
| C5 | perda em movimento | perda aleatória e rajadas de 0,3–1,0 s no refino | borrão no RGB ao andar e girar, luz, reflexo | a perda real depende do movimento; o treino a sorteia sem relação com ele | M |
| C6 | extrínseca da D435i | pose exata na cabeça | erro de montagem mais FK da cintura | viés de posição; o ±2 cm do refino pode cobrir ou não | M |
| C7 | ruído do depth | 2 % da distância | cresce com o quadrado da distância | a 2–3 m o real pode passar de 2 %; perto, fica abaixo | B |
| C8 | simetria do cubo | redução depois do ruído (a confirmar no código) | redução na saída do box_vision | perto de 45° a leitura salta entre rotações equivalentes; se o treino reduz antes do ruído, ele nunca vê o salto | B |
| C9 | mais de uma caixa ou objeto parecido | uma caixa | falso positivo, troca de caixa | salto da posição; fora do escopo da v3 | B |

## 3. Comum às duas

- A odometria entra nos dois blocos: no mapa (transporte) e na caixa e no alvo (mundo → base). Os
  pontos L3 e L4 também movem a caixa e o alvo do BOTAR no frame da base.
- Nenhum dos números de ruído do lidar, da odometria e do estimador foi medido no robô.

## 4. Os pontos A, em ordem

1. **L1:** a densidade real muda o canal de confiança.
2. **L2:** a cadência e o deskew no driver real ainda não estão definidos.
3. **L3:** a odometria do treino não erra no giro no lugar.
4. **C3:** a leitura parcial da caixa no PEGAR; o real erra, e o treino não.
5. **C1:** a distribuição do estimador ainda é hipótese.

## 5. Correções (28/10, APROVADAS pelo dono e aplicadas no plano e nas specs)

Enunciado: E §2 "Terreno" (o mapa); I §2.3 (a caixa). Regra: corrigir o modelo que causa a
diferença, e não imitar no robô real uma limitação do treino. A decimação de 1 em 20 pontos no
driver, sugerida antes, SAI: ela joga fora dado real para copiar o custo do raycast do treino.

### 5.0 Raio do corte e quem usa o filtro do corpo

FK sem física (`scratchpad/braco_esticado.py`), braço esquerdo, distância do MID-360 ao ponto mais
longe da mão: frente 0,64 m; lado (T) 0,74 m; cima 0,45 m; baixo 0,77 m; máximo em 20 000 poses nos
limites de junta 0,80 m. De pé com os braços soltos (`ref_v3_de_pe` pose 0), a mão já fica a 0,74 m,
fora do corte de 0,7 m. Lidar sobre a sola: 1,30 m de pé; 0,61–0,64 m no agachamento fundo.

- `raio_min` 0,7 → 0,85 m (0,80 m medido + 5 cm).
- Um só `filtra_corpo` (raio + cápsulas) para os dois consumidores: o mapa e a entrada da odometria
  real (KISS-ICP). O raio tira o braço e a caixa sem depender da forma das cápsulas; as cápsulas
  tiram as pernas e os pés quando aparecem. Nos dois casos, um ponto que anda com o robô não entra
  no casamento de nuvens.
- Custo: no agachamento fundo, o chão a 0,43–0,58 m na horizontal some; com a laje de 0,55 m, o topo
  da laje bem perto do robô some. O treino tem o mesmo corte.

### 5.1 L1 — a confiança depende do número de pontos

Causa: o modelo `R = σ_m²/n` (PD-42) supõe que os pontos de uma leitura têm erros independentes.
Eles não têm: o erro de inclinação do IMU, a extrínseca e o viés do encoder (zero da altura) são os
mesmos para todos os pontos da leitura, e os dois últimos são os mesmos para todas as leituras do
episódio. A média de n pontos não reduz esse erro. O treino já sorteia um erro de inclinação por
leitura (`ruido_incl_graus`), e o filtro do treino também o ignora.

Correção:

1. Variância da medida por célula: `R = σ_leitura² + σ_ponto²/n`, com `σ_leitura = σ_incl · d̄`
   (o erro comum da leitura, d̄ = distância média dos pontos da célula) e `σ_ponto²` = o `σ_m²` de
   hoje.
2. Piso da variância: `P ≥ σ_fixo²`, `σ_fixo` 2 cm (±1° fixo do IMU a 2 m ≈ 2 cm; viés de encoder
   no pé ≈ 1 cm). É o erro que nenhuma média remove.
3. Um modelo de IMU só: os pontos do mapa giram pela mesma orientação com viés que a observação da
   gravidade usa (evento `vies_imu`, ±1° por episódio), mais o ruído por leitura. Hoje são dois
   modelos para o mesmo sensor (L5).

Efeito (fórmula da PD-6, σ_ref 5 cm, célula a 2 m):

| | 1 leitura | várias leituras | + 1 m de transporte |
|---|---|---|---|
| hoje, treino (n = 1; 4 leituras) | 0,854 | 0,959 | 0,831 |
| hoje, real (n = 20; 80 leituras) | 0,992 | 1,000 | 0,862 |
| corrigido, treino | 0,774 | 0,862 | 0,758 |
| corrigido, real | 0,862 | 0,862 | 0,758 |

Com a correção, a confiança de uma célula vista satura no mesmo valor nos dois, depois de 1 leitura
no real e de 2 no treino. O que muda a confiança depois disso é o transporte (PD-7), que depende só
do movimento. A densidade só muda a altura: o real tem menos ruído, o que fica dentro do que o
treino viu.

Teste novo (C0, funções puras, sem simulador): chão plano com degrau, a mesma trajetória a 1 m/s,
pontos a 205 e a 4 000 por passo com o mesmo modelo de ruído; depois de 2 leituras por célula, o
recorte tem altura igual em ± 1 cm e confiança igual em ± 0,03.

### 5.2 L2 — cadência e distorção de movimento

Causa: o treino calcula cada ponto na pose exata do instante em que o raio sai. O real só faz o
mesmo se o driver usar a pose de cada ponto.

Correção:

1. Driver real: cada ponto vai ao mundo pela pose do sensor no carimbo de tempo dele (orientação pelo
   IMU, posição pela odometria, montagem pela FK das juntas daquele instante). É o deskew padrão dos
   métodos de lidar com IMU. Com isso, um ponto de terreno parado cai no mesmo lugar no real e no
   treino.
2. Cadência: a cada passo de 20 ms, o driver entrega ao filtro os pontos que chegaram desde o passo
   anterior. A I §2.4 ("a cada 0,1 s") passa a dizer isso, como a PD-5.
3. Atraso: o termo do mapa entra na DR de atraso por episódio, como a propriocepção (0–1 passo no
   início). No real, os pontos chegam ao filtro com o atraso do caminho; hoje o treino entrega o mapa
   sem atraso (L9).

### 5.3 L3 — a odometria do treino não erra no giro

Causa: o modelo da D2b só tem erro proporcional à distância andada. O modelo de movimento de
odometria dos livros de robótica (Thrun, *Probabilistic Robotics*, cap. 5) tem quatro termos: o erro
de posição e o de rumo crescem com a translação E com a rotação.

Correção: na `Odometria.passo`,

- variância de posição por eixo += `deriva_odo²·|Δp| + deriva_pos_giro²·|Δψ|`;
- variância de rumo += `deriva_rumo²·|Δp| + deriva_rumo_giro²·|Δψ|`.

Valores (hipótese até medir no G1): `deriva_rumo_giro` início 0,0035 rad/√rad (0,5° por volta),
refino 0,014 (2° por volta); `deriva_pos_giro` início 0,004 m/√rad (1 cm por volta), refino 0,008
(2 cm por volta). Teste: 10 000 envs, uma volta no lugar → desvio do rumo 2° ± 0,1 e da posição
2 cm ± 0,1 no refino. Como o mapa e o bloco da caixa usam a mesma odometria, o erro de giro aparece
nos dois, como no real.

### 5.4 C3 e C1

- C3: contrato do detector (dono, 28/10): sem certeza, ele entrega "sem leitura", nunca um centro
  deslocado. O treino supõe isso. O critério de "certeza" do detector real deve ser o da PD-11
  (centro no FOV do RGB, 0,28–3,0 m, ≥ 3 de 9 pontos sem oclusão), ou a PD-11 muda para o critério
  do detector quando ele for medido.
- C1: fica como está até o G1 chegar (dono, 28/10).

### 5.5 Onde as correções tocam

| correção | plano | specs | código (antes → depois) |
|---|---|---|---|
| 5.0 raio e consumidores | PD-10, §6.2 `raio_min` | I §1, §2.4; S §13 | knob só |
| 5.1 modelo da medida | PD-6, PD-42, §6.2 (`sigma_fixo`), §6.3 `pontos_no_frame_do_mapa` recebe o viés do IMU, C0, C1, D2 | I §2.4; S §19.1 | C1 ~350 → ~360; C0 + 1 teste (~30) |
| 5.2 cadência e atraso | PD-5, §6.8 (atraso do mapa), D2 | I §2.4 | ~0 (o termo já tem atraso no mjlab, a conferir na D2) |
| 5.3 odometria | PD-18, §6.2, D2b | S §13 | D2b ~60 → ~64; teste + 1 caso |
| 5.4 contrato | — | I §2.3 | — |
