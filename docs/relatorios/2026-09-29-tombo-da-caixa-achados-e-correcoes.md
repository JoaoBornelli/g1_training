# Tombo da caixa no PEGAR: achados e correções possíveis

Data: 2026-09-29. Branch `exp/g1-limpo-v2`, HEAD `829ee03`. Nada abaixo está aplicado.

## SOLUÇÃO CONSOLIDADA (revisão de 29/09, depois de 3 agentes; substitui a seção 4)

Todas alteram código existente. Nenhum termo novo, nenhum peso subido.

| # | alteração | onde | código | achado |
|---|---|---|---|---|
| S1 | coluna PEG_COM: `track_linear` 1→0, `track_angular` 1→0, `pose` 4→1 | `knobs.py` tabela | 3 números | A: pairar 25,0 → 18,6/s; fechar passa a compensar em ESPERA, BOTAR e CARREGAR |
| S2 | quem não avança no fim da espera vai à cauda: `cauda = acabou & ~avanca & ~ja_em_cauda` | `comando.py:742` | 1 token | B e E: o estado "fechou e não avançou" (32/s) deixa de existir |
| S3 | helper `_alinha` (corpo do `precise_ori`); `staged` = `alcança × (1 + trazer × alinha)`; `precise_pos` × alinha | `recompensas.py:594-636` | 8 → 8 | C: erguer tombando 17°→60° passa de +0,10 a −2,43/s |
| S4 | σ fixo na tolerância no regime de pé | `comando.py:1844` | 1 linha | F |
| S5 | RUN zero12; assert de clone; tabela na impressão digital | notebooks | +2 | I |

- S1: o rastreio no PEG_COM voltou em 08/09 (`8332caf`) para a velocidade de junta parado; o freio faz isso desde 23/09, e o `rumo` cobre o giro com 8,5× a alavanca. O `pose` ×4 era para o punho (`c64ee98`); a `faixa_de_pose` faz isso junta a junta desde `c9578ae`. O `forma_postural` FICA: tirá-lo cria queda de renda no primeiro toque.
- S2: o portão `perto` fica (e o `_forcado` do viewer). Removê-lo abre o atalho de baixar a caixa na espera para escolher a altura do BOTAR. Desviar para o CARREGAR faz baixar a caixa ser perda.
- S3 e S4 entram juntas: sem a S4, tombar na espera alarga o σ que a S3 passa a ler no CARREGAR.
- REJEITADOS: `renda_congelada` ×1,7 (sobe o prêmio de pressa que o freio contém); remover o portão; portão 40°; janela gaussiana σ 25°; escala do punho; `condim 4` (o `CollisionCfg` em `cena.py:286` sobrescreve o `_add_pad`; o `box_box` já gera até 8 pontos que resistem 1,1 a 1,6 N·m; contaria a torção duas vezes).
- Smoke a ajustar: 1554 (`pose` PEG_COM = 1), 1570 (rastreio zero no PEGAR_COM), 4941-4946 (G1 1c usa ESPERA_COM), item 6 5266-5321 (sem `perto` → CARREGAR; com `perto` → BOTAR), comentário 4172.
- Riscos a vigiar: pressa COBERTA pelo freio em currículo (+3,2 no prêmio pede o degrau 3, 1,5³ = 3,4×; o teto é o degrau 6, 11,4×, e o degrau sobe com os fechos; ler `freio_degrau`); perna com `pose` ×1 no PEGAR_COM (`altura_da_pelve`, `de_pe`); REORIENTAR sai de inerte um dia → gatear a S3 pelo regime.

Fontes: medições na CPU sobre `model_1600` e `model_1950` (zero11), `model_2000` (zero09),
`model_1750` (zero08); três revisões por agente (código, reward hacking, movimento);
leitura do código. Toda afirmação sem "MEDIDO" ou "LIDO" é conta de modelo.

## 0. Resumo

1. O robô ergue a caixa e ela tomba junto: **1,6 rad por metro de subida**. É cinemática:
   ombro, cotovelo e punho giram no mesmo eixo, o palma-a-palma. Só o punho desacopla, e a
   escala da ação dele é 6× menor (padrão do mjlab; o dono não quer mudar).
2. **Fechar o PEGAR paga menos do que pairar no alvo sem fechar** (23 → 16,5/s contra 25/s).
   Quando os fechos aparecem, o crítico aprende a fugir deles. Explica os fechos que somem.
3. **"Fechou mas não avançou" paga o dobro** (32/s): os sete ao vivo mais o congelado. É o
   estado mais lucrativo do PEGAR, e é um buraco.
4. A janela de orientação nos termos de aproximação ajuda **só na forma híbrida**, com σ fixo.
   O portão do PEGAR a 40° e a janela gaussiana σ 25° foram reprovados.

## 1. O problema observado

| run | o que aconteceu |
|---|---|
| zero08 | fechos até `s_B` 0,115 (it ~1430); depois `caixa_na_pega` 78° e `s_B` 0,007 |
| zero11 (retomada de zero10 ~1300) | fechos `s_B` 0,07 (its 1446–1568, caixa a ~22°); depois 60° e `s_B` 0,002 |
| viewer (dono) | sobe até o alvo, tomba, desce à mesa, endireita; repete a cada ~1,6 s |

## 2. Medições

### 2.1 `model_1600` (pico) e `model_1950` (colapso), PEGAR forçado, 70 envs, 10 por nível

| nível | 1600: fecho/ep | 1600: tombo p50 | 1950: fecho/ep | 1950: tombo p50 |
|---|---|---|---|---|
| 0 | 0,35 | 25° | 0,03 | 60° |
| 1–2 | 0,04–0,07 | 23–29° | 0 | 60–64° |
| 3–5 | 0 | 5–16° | 0 | 52–68° |

O colapso é igual em todos os níveis. O piso do currículo não é a causa (REFUTADO).

### 2.2 `model_1600`: o que muda entre a caixa baixa e a caixa no alvo (90 278 amostras)

| dist ao alvo | tombo p50 | `alinhado` vale | caixa na mesa | total/s |
|---|---|---|---|---|
| > 0,30 m | 7° | 83% | 65% | 15,7 |
| 0,20–0,30 | 14° | 86% | 21% | 21,3 |
| 0,10–0,20 | 27° | 43% | 0% | 23,7 |
| 0,05–0,10 | 36° | 10% | 0% | 24,7 |
| < 0,05 | 39° | 3% | 0% | 25,0 |

- A recompensa **cresce** com a aproximação: nenhum termo paga para baixar a caixa.
- Em 781 aproximações, o cronômetro do fecho fica em 0 em 90% (p90 = 0,08 s de 0,50).
- O tombo faz pico no ponto mais próximo (35,6°) e cai ao afastar (28°): subir e tombar
  são o mesmo movimento.

### 2.3 Renda por estado (`model_1600`, por segundo)

| estado | renda | composição principal |
|---|---|---|
| PEGAR_COM, pairando no alvo | 25,0 | staged 5,6, pose 3,5, precise_pos 2,8, forma 2,5, unload 1,9, precise_ori 1,7 |
| PEGAR_COM + fechou (não avançou) | **32,3** | congelado 14,6 + staged 2,2 + pose 3,3 + rastreio 3,8 |
| ESPERA_COM + fechou | 23,2 | congelado 15,0 + pose 3,3 + rastreio 3,8 |
| CARREGAR + fechou | 16,5 | congelado 14,7 + rastreio 6,5 − action_rate 4,2 |
| BOTAR | 20,7 | congelado 14,9 + precise_pos 2,7 + load 1,8 |
| CAUDA + fechou | −76 | freio −85,7 no instante de soltar (15 amostras; por desenho) |

### 2.4 O punho (`model_1600`, caixa no alto)

- Escala da ação: `wrist_pitch`/`wrist_yaw` 0,075 rad/unidade; ombro, cotovelo, `wrist_roll`
  0,439. Fórmula do mjlab: 0,25 × torque ÷ rigidez (`g1_constants.py:286`). Nunca foi
  sobrescrita em nenhum módulo (LIDO).
- A rede já grita no punho: `left_wrist_yaw` −5,25 (p90 6,08), `right_wrist_pitch` −3,04.
  O punho fica 0,23 rad atrás do alvo (~3,9 dos 5 N·m).
- Sonda (ordem somada por 0,5 s): `wrist_pitch` +0,45 rad → −24° de tombo; −0,45 → **+42°**;
  cotovelo +0,44 → −26°; deriva sem ordem −7,7°. A política manda o `wrist_pitch` para o
  lado negativo, o que tomba.
- O giro da caixa DENTRO da mão é 12° p50: a caixa segue a mão. Não é escorregamento.
- IK: existe pose reta no alvo com qualquer inclinação de mão (0,85 a 0,95 m).

## 3. Achados, por gravidade

### A. Fechar o PEGAR paga menos do que pairar (agente de hacking; renda de 2.3)

Pairar no alvo rende 25/s até o fim do episódio. Fechar rende 23,2/s por 0,5 s de espera
e depois 16,5/s no CARREGAR (com a política de hoje, que não anda bem com a caixa; um
carregador perfeito chegaria a ~26). Mais o risco de `caixa_largada`, que termina o
episódio. O crítico, quando vê fechos, aprende a evitá-los; tombar a caixa é um jeito de
não fechar. **Não verificado por comparação de retorno** (a simulação foi adiada).

### B. "Fechou mas não avançou" paga o dobro (LIDO; novo)

- `renda_congelada` paga o congelado todo passo, sem condição, cumulativo
  (`recompensas.py:1206-1240`).
- O avanço para o próximo elo exige `perto` no fim da espera (`comando.py:711-716`).
- O mapa de estados não conhece `fechou` (`comando.py:145-152`): o env fica em PEGAR_COM,
  com os sete ao vivo, mais o congelado. MEDIDO: 32,3/s, 533 amostras em 15 fechos.
- O caminho: fechar, baixar a caixa 1 cm na espera (os sete valem zero ali), e ficar.

### C. Subir tomba a caixa, e só o punho desacopla (medições 2.2 e 2.4; agente de movimento)

- Todo movimento que ergue gira a caixa no mesmo sentido. O modo reto ombro↓ cotovelo↑
  ergue 0,06 m/rad e empurra a caixa 17 cm para o tronco. Pernas saturadas (pelve 0,748).
- Gradiente que a política sente por unidade de ação: erguer tombando +1,4; endireitar pelo
  punho +0,11. O caminho certo é 13× mais fraco, com 2° de ruído.
- Nenhuma mudança de recompensa cria gradiente de "subir reto" nos canais fortes.

### D. O `ANG` pelo eixo de cima funciona como medida (28/09) e o `precise_ori` ×4 não basta

Na zero11 a caixa ficou reta enquanto baixa (17°) e tombou ao subir (60°). O termo tem
derivada viva; o caminho acoplado ganha.

### E. Recaptura do eixo de cima com a caixa na mão (agente de hacking; LIDO)

No ramo B, `_sigma_pendente` fica verdadeiro depois do fecho (`_avanca_elo_force`), o
`liga` roda com elo = PEGAR e `_captura_cima` (`comando.py:1858`) grava o eixo da caixa
torta. O CARREGAR e o BOTAR herdam. É o defeito de 21/09 por outro caminho.

### F. O σ de orientação é controlável pela política (agente de hacking; LIDO)

σ = max(tombo na abertura, 25°) é recalculado a cada elo (`comando.py:1844`). Tombar na
espera é grátis e alarga o σ. Hoje afeta só o `precise_ori`; com uma janela nos termos de
aproximação vira exploit (+1,6/s no CARREGAR, +3,1/s no BOTAR).

### G. Pivô no tronco (agente de movimento; LIDO)

Pads com `condim=3` (`cena.py:269`): sem atrito de torção. Empurrar a caixa contra o peito
a endireita dentro da mão. A janela lê a caixa, não a mão. Latente hoje (o giro na mão é
12°); cresce com qualquer incentivo de orientação mais forte. Só funciona no simulador.

### H. Fechar com a caixa na quina da laje (agente de hacking)

Com o portão a 40°, a caixa em pé na quina tem o centro a 0,753 m e passa no `perto`.
Com 25° não passa. Só importa se o portão alargar.

### I. Impressão digital cega (agente de código)

`pesos.json` guarda pesos e os knobs do freio. Janela, σ e portão não mudam peso: uma
retomada com código diferente passa pela guarda.

### J. Currículo `nivel` (achado anterior, 28/09)

O piso `abertos = buf.max()+1` mantém aberto todo nível já alcançado; o nível médio é
equilíbrio de sorteio, não competência. Não causa o tombo (2.1), mas distorce o painel.

## 4. Correções possíveis

| # | correção | onde | linhas | efeito | risco | estado |
|---|---|---|---|---|---|---|
| 1 | `renda_congelada` 1,0 → ~1,7 | `knobs.py:731` | 1 | congelado 15 → 25: fechar ≥ pairar | **só com a 2**: sozinha, o estado B passa a render ~45/s | recomendada |
| 2 | "fechou e não avançou" deixa de pagar os sete ao vivo: o mapa de estados trata `fechou & ~aguardando` como ESPERA_COM | `comando.py:145-152`, `_aplica_espera`; smoke sintético do mapa | +2 | fecha o buraco B: o estado passa a 23/s | muda a função pura; o smoke a testa com tensores | recomendada, junto com a 1 |
| 3 | janela **híbrida** no `staged` (`trazer × janela`) e no `precise_pos`; `precise_ori = alcançar × janela` | `recompensas.py:594-635` | −1 | erguer tombando deixa de render; derivada no alvo 1,5 → 3,9/rad; sem ótimo na mesa | não cria caminho nos canais fortes (C); depende do punho por ruído | recomendada |
| 4 | σ fixo na tolerância no regime de pé (sem `max` com o tombo inicial) | `comando.py:1844` | 0 | fecha o exploit F | quem chega tombado ao BOTAR perde a gaussiana; a metade linear cobre | obrigatória com a 3 |
| 5 | `& ~self.fechou[ids]` no `_captura_cima` | `comando.py:1858` | 0 | fecha E | nenhum | recomendada |
| 6 | `& ~apoiada` no fecho do PEGAR | `comando.py:1372` | 0 | fecha H | nenhum | opcional |
| 7 | RUN zero12; assert de clone; knobs novos no `pesos.json` | 2 notebooks | +3 | fecha I | nenhum | obrigatória com qualquer uma |
| 8 | portão do PEGAR a 40° | — | — | fecharia torto pagando menos | aprova o tombo de hoje; congela-o; sem refinamento no CARREGAR | **REJEITADA** (3 agentes) |
| 9 | janela gaussiana σ 25° | — | — | — | ótimo 3 cm acima da mesa; canal morto a 60° | **REJEITADA** (2 agentes) |
| 10 | escala da ação do punho = cotovelo | `env_cfg.py:226` | +1 | ruído 2° → 12°; gradiente 6× | mexe no padrão do mjlab | **REJEITADA** (dono) |
| 11 | pads `condim` 3 → 4 | `cena.py:269` | 0 | atrito de torção: fecha G; mais real | muda a física da pega; medir antes | para depois |
| 12 | piso do currículo só até o nível conquistado | `curriculo.py:196` | ±2 | painel honesto | nenhum no tombo | separada |
| 13 | isentar a CAUDA do freio | — | — | — | a CAUDA é parada por desenho (v3.4) | **REJEITADA** (eu) |

### Dependências

- 1 exige 2. A 2 sozinha vale (fecha o buraco), mas não faz fechar valer mais do que pairar.
- 3 exige 4. A 3 sem a 4 abre o exploit F.
- 7 acompanha qualquer mudança de forma.

### Ordem proposta, uma mudança por vez

1. **Renda pós-fecho** (2 + 1 + 5 + 7): o fecho passa a valer mais do que pairar e menos do
   que avançar. É a correção do achado A/B, que explica os fechos que somem.
2. **Janela híbrida** (3 + 4): tira a renda de erguer tombado. Só depois de ver a 1 no treino.
3. Pads (11) e currículo (12): separadas.

## 5. O que exige simulação (adiado)

- **Retorno de quem fecha contra quem paira** no `model_1600`: decide o achado A com dado,
  e não com conta. ~15 min na CPU.
- Sondas ombro × cotovelo (9 combinações): mostra se existe direção que aproxima e
  endireita ao mesmo tempo. ~20 min.

## 6. Scripts de medição (scratchpad da sessão)

`mede_nivel.py`, `mede_renda.py`, `mede_sobe_desce.py`, `mede_punho.py`, `ik_reta.py`,
`mede_fechar_vs_pairar.py` (escrito, não rodado).
