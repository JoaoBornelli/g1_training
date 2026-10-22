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
