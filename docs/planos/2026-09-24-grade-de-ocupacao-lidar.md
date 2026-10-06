# Plano — a grade de ocupação do MID-360 como percepção da política
> **Substituído para a v3** pelo `docs/planos/2026-10-02-percepcao-lidar-na-v3.md` (2026-10-02): camadas de 5 cm, padrão de raios rotativo, contrato de replicação e chave.

Data: 2026-09-24. Nenhum código foi escrito ainda. O trabalho começa no worktree
`lidar_sim` (branch `JoaoBornelli/lidar_sim`) e só chega ao `g1_limpo` na fase 5.

## 1. O objetivo

A política ganha uma entrada de PERCEPÇÃO do ambiente: o que existe em volta do robô,
visto pelo MID-360 da cabeça. Hoje ela recebe a pose da caixa e o alvo por oráculo, e não
vê a mesa. Com a percepção, ela passa a ver a laje, o teto de uma prateleira e as paredes
laterais, e pode escolher COMO pegar e botar a caixa sem bater.

A percepção é do AMBIENTE, e não do objeto. O oráculo da caixa (10 canais) continua. A
localização da caixa por visão é outro trabalho (`g1-box-vision`).

## 2. A decisão central: uma função só, nos três lugares

```
TREINO (mjlab, E envs)                  SIMULADOR DO REAL / REAL (E = 1)
raycast do mjlab, R raios               mj_multiRay ou SDK do Livox
   → pontos em mundo                       → pontos no frame do sensor
   → convertidos ao frame do sensor        │
           └────────────┬──────────────────┘
                        ▼
      filtro do próprio corpo (cápsulas + caixa na mão)
                        ▼
      pontos_para_grade(pontos, roll, pitch, altura) → grade (E, 800)
                        ▼
      transporte + decaimento entre leituras (10 Hz → 50 Hz)
                        ▼
                     política
```

A função `pontos_para_grade` e o filtro do corpo são PUROS: torch, sem import de
simulador. O treino, o simulador do real e o robô real os importam do mesmo módulo. Esta
é a regra da memória `g1-limpo-lidar-sim-desenho`, e é ela que faz a política transferir.

## 3. A grade

Um volume fixo em torno do robô, dividido em cubos de 0,15 m. Cada índice do vetor é um
cubo; o valor é 1 quando ao menos um ponto do lidar caiu dentro dele, e 0 nos outros casos.

| eixo | faixa, no frame da base | cubos |
|---|---|---|
| x (frente) | −0,3 a +1,2 m | 10 |
| y (esquerda) | −0,6 a +0,6 m | 8 |
| z (altura) | 0 a 1,5 m, desde o CHÃO | 10 |

- Alinhada ao YAW da base: a frente da grade é a frente do robô. Roll e pitch saem, e o
  chão fica sempre na camada de baixo.
- A origem em z é o chão, e não a pelve, para a grade não andar com o agachamento.
- Índice linear: `ix·80 + iy·10 + iz`. Saída `(E, 800)`, float, 0,0 ou 1,0.
- Binária de propósito: 128 raios no treino e 20 mil no real dão a mesma grade.
- 0 quer dizer "nenhum ponto": ar livre e região não vista saem iguais. O traçado do raio
  para separar os dois fica fora deste plano.

Os tamanhos são ponto de partida. A fase 1 mede se 1,2 m à frente e 1,5 m de altura
cobrem a mesa e uma prateleira ao alcance.

## 4. A pose do sensor: uma constante só

O `mid360_joint` do URDF rev 1.0: site no `torso_link`, `pos = (0,0002835, 0,00003,
0,428434)`, `rpy = (π, 0,0511, 0)`. É a pose que o `estudo_lidar/lidar_parado.py:43` já
usa. Ela vira UMA constante `T_BASE_SENSOR` num módulo compartilhado, e dela saem:

- o site do raycast no treino;
- a transformação sensor → base da função;
- o site do simulador do real.

Um site posto à mão em outra pose faria a grade do treino divergir da do real em silêncio.

## 5. As fases

Cada fase tem um portão. A seguinte só começa com o portão verde.

### Fase 1 — a função pura (worktree `lidar_sim`)

**Arquivo novo:** `percepcao/grade.py`.

- `T_BASE_SENSOR`, `CELULA`, `ORIGEM`, `DIMS`: as constantes.
- `pontos_para_grade(pontos, roll, pitch, altura, validos=None) → (E, 800)`:
  1. sensor → base, pela constante;
  2. desfaz roll e pitch da base; soma a altura, e z passa a contar do chão;
  3. `floor((p − ORIGEM) / CELULA)` e a máscara `dentro`;
  4. `scatter_` com os inválidos levados a uma 801ª célula descartada. Sem isto, um ponto
     inválido no índice 0 pode apagar um válido, porque o `scatter_` não garante ordem.

**Portão:** teste sem simulador, com pontos sintéticos.
- um plano horizontal a 0,55 m ocupa só a camada `iz = 3`;
- o mesmo plano, com a base inclinada 10° em pitch, ocupa a mesma camada;
- um ponto fora do volume não ocupa nada, e um inválido no índice 0 não apaga um válido.

### Fase 2 — no simulador do real, robô parado

**Arquivo:** `estudo_lidar/lidar_parado.py` ganha a chamada à função e um desenho dos
cubos ocupados no viewer.

- os pontos do `mj_multiRay` saem no frame do sensor (hoje eles saem em mundo);
- roll e pitch vêm do quaternion da base, e a altura do estado, pelo mesmo caminho que
  a IMU e as pernas darão no real: nada além desses três números.

**Portão:** no viewer, a laje da mesa aparece nas células certas. Inclinar o robô não
move a mesa na grade. Arrastar a coluna mocap move os cubos dela.

### Fase 3 — o filtro do próprio corpo

**Arquivo novo:** `percepcao/corpo.py`.

- A lista de cápsulas por elo (ombros, braços, antebraços, mãos, tronco, coxas), gerada
  das geometrias de colisão do URDF, com margem de 2 a 3 cm.
- `filtra_corpo(pontos, poses_dos_elos, caixa=None) → validos`: distância de cada ponto a
  cada segmento; o ponto a menos do raio sai. ~12 cápsulas × 20 mil pontos, vetorizado.
- A caixa na mão entra como mais um volume, ativo só com `pegou`. Antes da pega a caixa é
  ambiente e fica na grade; depois da pega ela sai, porque a política já a conhece pelo
  oráculo e uma "parede" que anda com as mãos seria ruído.
- No simulador, o `mj_multiRay` passa a INCLUIR o robô (hoje o `torso_link` é excluído),
  para o raio parar no ombro como no real. O filtro descarta o ponto que parou nele.

**Portão:** robô com a caixa no peito. Os pontos sobre braços e caixa somem; a mesa
continua; a parte da mesa escondida pela caixa fica vazia, como no real.

### Fase 4 — o tempo: 10 Hz para 50 Hz

**Arquivo:** `percepcao/grade.py` ganha `transporta(grade, deslocamento_da_base, decai)`.

- Entre duas leituras do lidar, a grade é deslocada pelo movimento da base (odometria) e
  decai: `grade = max(nova, decai · transportada)` na leitura, `decai · transportada` entre
  elas. Com a grade binária, o decaimento vira um limiar de idade por célula.
- É isto que guarda a prateleira de destino quando a caixa sobe ao peito e a esconde, e o
  chão da zona cega de 1,0 m à frente depois que o robô passa por ele.

**Portão:** no simulador, com o robô andando pelo `pilota.py`, a mesa fica parada no mundo
enquanto a grade anda; ela não "pisca" entre leituras.

### Fase 5 — no treino (`g1_limpo`)

Só aqui o código toca o treino, e ele recebe as fases 1 a 4 prontas.

- **Padrão de raios:** `Mid360PatternCfg`, uma quarta classe ao lado do `GridPatternCfg`
  do mjlab, com R direções subamostradas do `mid360.csv`. R sai de uma medição: disparar
  contra um plano e contar as células ao alcance que ficam vazias. O alvo é nenhuma célula
  vazia dentro de 1,2 m. O ponto de partida é R = 128 a 256.
- **Sensor:** `RayCastSensorCfg` com `frame` no site da constante, `include_geom_groups`
  com terreno, mobília e robô, e `debug_vis` para conferir.
- **Ritmo:** o sensor lê a cada 5 passos (10 Hz a 50 Hz de política), e o transporte da
  fase 4 roda nos outros 4. Sem isto a política treina com grade fresca a cada 20 ms e
  recebe uma de até 100 ms no real.
- **Observação:** um termo novo no fim do `actor` e do `critic`, depois dos 114 canais de
  hoje; `fatia_do_elo` e `N_CAIXA` continuam valendo pelo contrato de append. A
  observação vai a 914.
- **Custo:** medir o passo com e sem o sensor, na CPU e depois na GPU do Colab, antes de
  treinar. O raycast é da ordem do `terrain_scan` do fabricante (160 raios); o `scatter_`
  é um kernel só.

**Portão:** smoke com as checagens das fases 1 a 4 portadas; a mesa na grade do env igual
à do simulador do real para a mesma pose; custo do passo medido e aceito pelo dono.

**Nova linhagem do zero.** A observação muda de tamanho, e um checkpoint de 114 entradas
não carrega. A percepção é entrada permanente, então ela nasce com a próxima linhagem do
zero, e a `zero07` fica como referência sem percepção. Com a mesa, a grade é quase
constante e a política aprende a pesá-la pouco; ela ganha função quando o suporte variar.

## 6. Fora deste plano

- **A prateleira.** É outro currículo: a caixa tem teto, e o robô tem de puxar antes de
  erguer. Ela entra como variante do suporte (a cena de depósito já lista palete, mesa e
  estante), com a batida paga por termo de contato que já existe, e não por termo novo.
  Plano próprio, depois da fase 5.
- **Três estados por célula** (ocupado, livre, não visto), com traçado do raio.
- **SLAM.** A grade é local e alinhada ao yaw; ela não precisa de posição global.
- **Ruído do sensor.** O MID-360 tem erro de ~2 cm e pontos espúrios. Ele entra como
  ruído nos pontos, antes da função, quando a fase 5 treinar; o valor sai da ficha técnica.

## 7. Riscos

| risco | onde aparece | o que fazer |
|---|---|---|
| a convenção de roll e pitch não bate com a IMU do G1 | fase 2 e real | o teste da inclinação da fase 2; conferir a ordem Rx, Ry contra a IMU antes do real |
| R raios deixam células vazias que o real enche | fase 5 | a medição contra o plano define R |
| o filtro do corpo corta a mesa perto da mão | fase 3 | a margem das cápsulas é knob; o portão mede a mesa perto da mão |
| o custo do raycast derruba as iterações por hora | fase 5 | medir antes; o Colab é limitado por banda (memória `g1-treino-gargalo-banda-nao-flops`) |
| a política ignora a grade com a mesa fixa | treino | esperado; a grade ganha função com o suporte variado, que é o plano seguinte |

## 8. Tamanho

| fase | arquivos | linhas novas, estimativa |
|---|---|---|
| 1 | `percepcao/grade.py`, teste | ~60 + ~40 |
| 2 | `lidar_parado.py` | ~30 |
| 3 | `percepcao/corpo.py`, teste | ~80 + ~40 |
| 4 | `percepcao/grade.py` | ~40 |
| 5 | `g1_limpo`: padrão, sensor, observação, smoke | ~120 |

Nada é removido: é capacidade nova. No `g1_limpo`, o sensor é o único termo de observação
novo, e nenhum termo de recompensa entra.

## 9. Anotado para depois: a caixa fora de vista (2026-09-24)

Decisão do dono: primeiro o robô executa as tarefas de maneira satisfatória; percepção,
oclusão e estimador entram depois. Nada disto é implementado agora.

**O problema.** O ator recebe a pose EXATA da caixa em todo passo fora do ANDAR
(`observacoes.caixa_no_frame_da_base`, 10 canais, sem ruído nem atraso). No real a câmera
da cabeça perde a caixa em três momentos: fora do campo de visão na aproximação, coberta
pelas mãos na pega e encostada no peito, abaixo da câmera, no carregar.

**O caminho proposto.** Um estimador entre a câmera e a política, função pura igual nos
três lugares:

| estado | fonte da pose da caixa |
|---|---|
| sobre a mesa, visível | câmera, pela tag |
| sobre a mesa, fora de vista | última pose vista, transportada pela odometria da base |
| na mão (`pegou`) | cinemática direta das mãos + deslocamento medido na pega |

No treino:
- o ator recebe a estimativa: visibilidade simulada por cone da câmera e raycast até a
  caixa (o mesmo raycast do lidar); invisível, o canal mantém e transporta a última pose;
- um canal novo com a IDADE da estimativa;
- ruído e atraso na pose vista;
- o crítico continua com a verdade, como o `elo_interno`.

Muda a observação do ator: pede nova linhagem do zero, e pode nascer junto com a grade.

**A fonte da pose vista, decisão do dono (2026-09-24).** Duas CNNs leves na câmera: uma
detecta onde a caixa está na imagem, a outra infere a orientação dela. A distância sai da
fusão com o depth da câmera ou com o lidar, e dá a posição relativa à câmera. A política
continua treinada com os canais de hoje, e o sinal é PIORADO no treino para ficar realista
(ruído, atraso, perda por oclusão), em vez de o simulador rodar as CNNs.
