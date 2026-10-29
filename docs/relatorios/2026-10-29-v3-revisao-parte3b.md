# Revisão da Parte 3b da v3 (G7)

**Veredito: NADA DERRUBA O ENV E A CENA DE B NO TERRENO ESTÁ CERTA; MAS A s33 FALHA SEMPRE, A MÉTRICA DA D10 MEDE OUTRA COISA, O REGISTRO DE TERRENO NASCE SEM CAIXA E, NA ESCADA, A QUALIDADE DO CARREGAR CORTA O RASTREIO A ~30%.**

| severidade | achados |
|---|---|
| bloqueia o treino | 0 |
| defeito silencioso | 3 |
| regra do plano | 4 |
| menor | 9 |

Escopo: `git diff 33f4e50..HEAD -- g1_v3 pilota_v3.py registra_juntas_v3.py`, seis commits (`ee2707f` a `bbfae26`).
Código: 8 arquivos, +393 −33. Testes: 9 arquivos, +694 −2. Li inteiros: `eventos.py`, `comando.py`, `metricas.py`,
`curriculo.py`, `terminacoes.py`, `balanceador.py`, `registra_juntas_v3.py`; `knobs.py` (`Terreno`, `Comando`, `Cena`,
`Alvos`); `cena.py` (`terreno_cfg` e sensores); `exporta_cena.py` (contrato, `constroi_env`, `exporta`); `smoke.py` (30 a
33 e os auxiliares). API do mjlab no `.venv`: `terrain_generator.py`, `primitive_terrains.py` (escadas, escada
aleatória, caixas), `heightfield_terrains.py` (rampa, obstáculos), `terrain_entity.py`, `terrain_height_sensor.py`,
`command_manager.py`, `manager_based_rl_env.py`. Nenhum env foi construído. Fora do env, só: a `function` de cada
sub-terreno num `MjSpec` em CPU com `mj_ray` para ler alturas, `alvos.Referencias`, `balanceador.Curriculo`,
`comando.sorteia_velocidade` e um MJCF de três corpos para a colisão mocap × estático. As contas estão em
`/tmp/claude-1002/-home-joaobornelli-Documents-g1-training/5d540910-028a-4291-8d9f-065788e279ae/scratchpad/g7_contas.py`
e `/tmp/claude-1002/-home-joaobornelli-Documents-g1-training/5d540910-028a-4291-8d9f-065788e279ae/scratchpad/g7_contas2.py`.

Suíte (`unittest discover`): 384 testes, todos ok, 1 skip. `pyflakes` nos 9 módulos tocados: sem achado. Nenhum teste
da G chama um invólucro com env: S1 a S3 passam verdes hoje.

Conferido sem achado:
- A premissa da G1 vale: em todas as 7 colunas com plataforma, `terrain_origins` é o centro do sub-terreno na altura do
  plano central. Escada (n + 1)·h (`primitive_terrains.py:234`); invertida −(n + 1)·h (`:411`); aleatória a soma dos
  degraus (`:1159`); rampa e rampa invertida ±z_pf (`heightfield_terrains.py:285-286`); obstáculos e caixas 0. Medido por
  raio: o chão sob a origem é o z da origem nas 7.
- O núcleo `cena_na_plataforma` e o `posiciona_cena` seguem a §6.3: laje a R(rumo)·(0,75; 0), caixa e robô no mesmo
  frame, `base_para_mundo` em toda conta. O `_base_da_pega` põe a base em caixa − R(rumo)·(d, 0), rumo + o jitter.
- Sorteio sem terreno: o `return` antes do `_uniforme` (`eventos.py:112-113`) só muda a posição do fluxo do gerador. Os
  uniformes seguintes seguem i.i.d.; nada correlaciona envs com e sem terreno. O custo é um `.any()` que sincroniza a
  GPU por reset.
- A G2 (`balanceador.py:146-148`) é a S §11.2: o terreno do CARREGAR abre no último nível de velocidade, com as
  abertas no portão e o terreno do ANDAR já dominado no nível 0 (`nivel_ter ≥ 1` só vem depois de a célula passar).
- Laje × terreno: a laje é mocap e não colide com o terreno estático (MJCF de prova: `ncon` 0, `body_weldid` 0).
- O reset do exportador nunca cai em terreno: o currículo do play é o novo (só ANDAR aberto, `fracoes_fatias` = [1, 0,
  0], conferido), a célula `terreno` do ANDAR está fechada e o `sorteia_fatia` dá coluna 0, linha 0, z 0. O defeito é
  outro (S2).
- `qvel_inicial` do exportador é zero: o reset grava velocidade zero na raiz, nas juntas (`velocity_range` (0, 0)) e na
  caixa. A transformação do `pose_no_sub_terreno` está completa (M4 só troca o comentário por um assert).

Ordem do `env.step` (`manager_based_rl_env.py:419-464`): física, terminação, recompensa, métricas, reset (`_reset_idx`:
eventos, depois `command_manager.reset`, a cadeia antes do twist), `sim.forward()`, `command_manager.compute` (cadeia,
depois twist), eventos de intervalo, `sense`, observação. O `env.reset()` roda `command_manager.compute(dt=0)` depois
do `forward` (`:369-374`).

---

## 1. Colunas com plataforma, na linha máxima do CARREGAR

Laje: centro a 0,75 m, meia 0,30 (de 0,45 a 1,05 m no eixo do rumo, ±0,30 de lado), espessura 0,04, topo pelo PEGAR
(fatia B: U[h_k; 0,55], até 0,04 no nível 4). "Plano até" é a distância do centro em que o chão sai 5 mm da origem,
medida por raio a cada 5 mm, nas 4 bordas e a ±0,30 de lado.

| coluna (linha; t) | origem z | plano até | primeiro degrau ou rampa fora | terreno sob a laje (4 bordas) |
|---|---|---|---|---|
| escada (9; 0,75) | +1,050 (6 degraus de 0,15 + 0,15) | 1,20 m nas 4 bordas: plataforma de 2,4 m (8 − 2·1,0 de borda − 2·6·0,30) | anel de 0,30 m, 0,15 m abaixo | 0,000: não corta |
| escada_inv (9) | −1,050 | 1,20 m | anel de 0,30 m, 0,15 m ACIMA | 0,000: não corta; o degrau começa 15 cm além da laje |
| rampa (9) | +0,555 | 1,79 m no eixo; 1,63–1,68 m a ±0,30 de lado. O plano é uma estrela (`hf_raw` = h·xx·yy cortado em z_pf), e não um quadrado | rampa de 0,27 (−4,1 cm em 15 cm) | 0,000 |
| rampa_inv (9) | −0,555 | idem | rampa de 0,27 subindo | 0,000 |
| escada_aleatoria (9) | +0,07 a +0,44 (40 sementes; média 0,26) | 2,15 m: plataforma de 2,7 m, e o anel de dentro fica no nível dela (o topo do plano é a soma dos 3 degraus) | anel de 0,80 m; degrau U(0; 0,20)·(0,5 + 0,5·t), até 0,175 m (R1) | 0,000 |
| obstaculos (8; 0,667) | 0,000 | 0,962 m (vértice 49 do hfield de 80); o hfield sobe até o vértice 50, a 1,063 m | obstáculos de 0,3 ou 0,7 m, ±0,045 ou ±0,095 m, colados na borda | a ponta de 0,962 a 1,05 m pode ficar na subida: até +0,083 m. 22,5% das (célula, borda) com > 0,5 cm, 10% com > 4 cm. Corta com topo < ~0,12 m (só o PEGAR no nível 4) |
| caixas (7; 0,583) | 0,000 | 2,0 m no papel; as caixas giradas invadem (R2): 76% das células têm > 1 cm dentro de \|x\|,\|y\| ≤ 0,95 m | caixas de 0,1–1,0 × 0,1–2,0 m, altura U(0; 0,10), giro livre | até +0,093 m; 12,5% > 0,5 cm, 10% > 4 cm |

Veredito: a laje a 0,75 m não corta o terreno nas escadas, nas rampas nem na aleatória, em nenhuma borda. Nos
obstáculos e nas caixas a ponta da laje pode entrar no terreno, sem efeito físico (mocap). A caixa não chega lá nos
obstáculos (face de longe ≤ 0,90 m < 0,962). Nas caixas, a zona da caixa tem terreno > 1 cm em 7,5% das (célula, borda)
e a zona dos pés em 2,7% (150 células, linha 7).

## 2. O robô pela S4 e a fração fora da plataforma

No eixo do rumo, a base fica em 0,75 − 0,50 + 0,32 + dx − d = 0,57 + dx − d, com d = `recuo(topo, a)` ± 0,08 e
`recuo` de 0,331 a 0,438 m na grade inteira. De lado, dy ±0,18; rumo ±0,2 rad.

| PEGAR | x da base | à borda de 1,0 m |
|---|---|---|
| nível 0 (topo 0,55) | +0,15 a +0,52 | 0,48 a 0,85 m |
| nível 2 | +0,05 a +0,52 | 0,48 a 0,95 m |
| nível 4 (topo 0,04–0,55) | +0,05 a +0,40 (p5–p95 +0,12 a +0,31) | 0,60 a 0,95 m (p5–p95 0,69 a 0,88) |

A faixa 0,48–0,95 m da decisão do dono confere. A borda real fica mais longe: +0,20 m na escada, +0,79 m na rampa (no
eixo), +1,15 m na aleatória; −0,04 m nos obstáculos.

Fração do CARREGAR com a base fora, marcha ideal, 4–7 s, as quatro direções do `_anda` (frente é rumo à borda):

| velocidade | borda | frente | trás | lados | média |
|---|---|---|---|---|---|
| 0,25 m/s (viab. D10) | 1,0 m (a métrica de hoje) | 0,41 | 0,12 | 0,25 | 0,26 |
| 0,25 m/s | escada real, 1,20 m | 0,26 | 0,04 | 0,13 | 0,14 |
| 0,25 m/s | rampa real, 1,79 m | 0,02 | 0,00 | 0,00 | 0,006 |
| 0,25 m/s | aleatória real, 2,15 m | 0 | 0 | 0 | 0 |
| U[0,25; 1,0] (o treino) | 1,0 m | 0,73 | 0,58 | 0,65 | 0,65 |
| U[0,25; 1,0] | escada 1,20 | 0,66 | 0,51 | 0,58 | 0,58 |
| U[0,25; 1,0] | rampa 1,79 | 0,48 | 0,36 | 0,42 | 0,42 |
| U[0,25; 1,0] | aleatória 2,15 | 0,36 | 0,27 | 0,31 | 0,31 |

⚠ O treino não anda a 0,25 m/s no terreno. A célula `terreno` do CARREGAR só abre no nível 5 de velocidade
(`balanceador.py:146`), e `sorteia_velocidade("terreno", 5)` dá U[0,25; 1,0] m/s, média 0,63 (`comando.py:113-114`,
PD-12; conferido). O caminho mediano é 3,35 m; 34% dos CARREGAR de terreno saem da célula de 8 m e 59% passam do fim da
escada (3 m). A preocupação da D10 (pouco terreno) não vale com essa velocidade (P5).

Viés da métrica de hoje (S1): ela loga a média transversal das frações CORRENTES dos envs no CARREGAR, e não a fração
do episódio. A 0,25 m/s: por episódio 0,259, logada 0,054 (frente: 0,411 contra 0,107).

## 3. A caixa perdida e a queda descendo a escada (viab. D4)

- `caixa_perdida` (`comando.py:355-358`): caiu = `altura_caixa` − a < 0,02. O sensor é um `TerrainHeightSensor` no
  centro da caixa, anel de 5 cm, grupo 0, redução "min" (o ponto de terreno mais alto sob o anel). Lê o terreno local e
  não lê `env_origins`. Caixa no peito: centro a pelve + 0,111 = 0,90 m sobre o pé, 0,25 m à frente; fundo a 0,77–0,83
  m. Descendo, o chão sob a caixa é igual ou mais baixo: folga ≥ 0,77 m. Subindo (escada_inv para fora, ou de costas na
  escada): no máximo um degrau acima (0,25 + 0,05 < 0,30): folga ≥ 0,62 m. Rampa: +0,08 m, ≥ 0,69 m. Obstáculo: +0,095,
  ≥ 0,67 m. O `arremesso` lê velocidade relativa e o `fora_da_laje` só P0–P2: nada lê o terreno.
- `queda` (`terminacoes.py:76-89`): no CARREGAR (marcha), inclinação > 0,8 rad pela gravidade projetada, ou pelve −
  chão < 0,20, com o chão do `altura_pelve` (anel de 5 cm, "min"). Local. Com um pé em cada degrau, a pelve fica ≥ 0,789
  − 0,15 = 0,64 m sobre o degrau de cima. No PEGAR e no BOTAR, o joelho pelo `altura_joelhos`, também local.
- Até quantos degraus: todos. Nenhuma das duas lê a origem. A escada da linha 9 tem 6 anéis (o último a −0,90 m) e a
  borda da célula a −1,05 m; as duas seguem certas em todos e fora da célula. O limite de 5–6 degraus da D4 (o g1_limpo
  media contra a origem) não existe mais. A s32, que devia provar isso, não prova (M1).

## 4. A ordem do `twist` e o `v3_incl_max_seg`

Num passo comum: a física roda; as terminações e o reset passam; `sim.forward()`; a cadeia publica o CARREGAR e chama
`twist._resample` (contador + 1); o `compute` do twist chama `_update_metrics` primeiro (`command_manager.py:109-110`):
`mudou` fecha o segmento velho com o máximo até o passo anterior, `_abre` zera (`comando.py:635`), e só então
`_mede_incl` lê a caixa (`:696`), já com a física do passo e o `forward`. No reset: `TwistV3.reset` fecha com o máximo
antigo, reamostra e `_abre` zera; o `compute` seguinte (do `step` ou do `env.reset`) mede a caixa do reset, em pé (0°).

Veredito: mede o passo certo e zera antes da primeira medida. Defasagem de um passo nas pontas, sem efeito: a primeira
medida do CARREGAR é a caixa no fim do último passo do PEGAR (que fechou o P4 com a inclinação no limite do degrau), e o
estado do passo em que o `time_out` reseta não entra.

## 5. Formas e dtypes

| escrita | buffer | valor | veredito |
|---|---|---|---|
| `eventos.py:334` `v3_caixa_xy[ids]` | f32 (E,2) | f64 `.to(dtype)` | ok (o conserto da G1) |
| `eventos.py:335` `v3_rumo_cena[ids]` | f32 (E,) | f64 `.to(dtype)` | ok |
| `eventos.py:253-264` pose da laje, `:272-274` da caixa | tensor local f32 | xy e rumo f64 por fatia `[:, k]` | ok: fatia básica é `copy_` e converte |
| `eventos.py:174-180` pose da base | `root.dtype` | `.to(root.dtype)` nos dois pedaços | ok |
| `comando.py:635` e `:641` `v3_incl_max_seg` | f32 (E,) | 0.0; `maximum(f32, f32)` | ok |
| `metricas.py:385-386` buffers próprios | f32 | `bool.float()` | ok |
| `smoke.py:1046-1048` terreno forçado | long; f32 | int; f32 | ok |

Nenhuma outra escrita de dtype trocado em buffer `v3_*`. O teste da G1 usa `v3_caixa_xy` e `v3_rumo_cena` em float32 no
env falso, e por isso pega o `index_put`.

---

## Defeito silencioso

| # | arquivo:linha | o que está errado | cenário | correção | plano |
|---|---|---|---|---|---|
| S1 | `g1_v3/metricas.py:384`, `:387` | `fracao_fora_da_plataforma` mede (a) contra a meia plataforma de 1,0 m em toda coluna, e a real é 1,20 (escadas), 1,63–1,79 (rampas), 2,15 (aleatória), 0,96 (obstáculos); (b) devolve a média transversal das frações correntes (`_media_dos_com_evento`), e não a do episódio | Escada, frente, 0,25 m/s: no terreno de verdade 0,26; com a borda errada 0,41; logado ~0,11. Rampa: real 0,02, logado ~0,11. Média das 4 direções, borda de 1,0: por episódio 0,26, logado 0,054. A decisão da D10 (fase I) sai desta métrica | (b): o `reset(env_ids)` guarda a média das frações dos episódios que fecham, e o `__call__` devolve essa constante (+4). (a): P1 | §6.3 G4; viab. D10 |
| S2 | `g1_v3/exporta_cena.py:449-460`; `registra_juntas_v3.py:62`, `:328-348` | O exportador exporta o reset do currículo novo do play: só o ANDAR abre, a fatia é ANDAR, a cena estaciona (laje e caixa 20 m acima, `eventos.py:281-290`) e a base sai com xy ±0,5 m e rumo ±π. O comentário da `:62` ("a cena é exportada do reset do PEGAR") é falso | `--roteiro escada --coluna escada --linha 9 --borda 1`: a caixa vai à origem do sub-terreno + 20 m; o robô fica perto do centro, num rumo qualquer; a câmera (alcance 3 m) nunca vê a caixa e o PEGAR espera em pé para sempre. Vale também para os roteiros `c` e `carregar` (E9, F7) | O `exporta` força a fatia B no plano depois do `env.reset()`, pelo caminho do treino (o padrão do `_abre_b_na_escada` da smoke, sem a coluna): `v3_fatia[:] = B`, `posiciona_cena`, `reset_base_por_fatia`, `reinicia_filtros`, `forward`. O `origem_reset` passa a ser a da cena B (+~7) | §12.3 G6; D11 |
| S3 | `g1_v3/recompensas.py:426`, `:485-490` | No CARREGAR, a qualidade do rastreio mede a caixa contra o P4 = chão sob a pelve + 0,90. Na escada o chão sob a pelve pula 0,15 m na quina e a pelve desce contínua | A caixa rígida a pelve + 0,111 dá d = \|h − 0,789\|, que varre 0 a 0,15 m em cada degrau. q = exp(−(d/0,05)²) vale 0,105 em d = 0,075; média 0,30 no degrau de 0,15, 0,44 no obstáculo de 0,10, 0,62 no degrau da linha 4, se a pelve desce uniforme no degrau. A parte positiva do rastreio (linear e giro) cai ~3× com a caixa perfeita, e a derivada pede a pelve a pular 0,15 m na quina. Rampa: d ≤ 0,0135, q ≥ 0,93. A `cerca_pelve` usa o mesmo h, mas a margem de 0,08 a deixa quase muda (≤ 0,0075/s) | P2 | S §8.2, PD-21, PD-37; E §4.4 ("a caixa fica no alvo de transporte o tempo todo"); CLAUDE.md (gradiente) |

## Regra do plano

| # | arquivo:linha | o que está errado | cenário | correção | plano |
|---|---|---|---|---|---|
| R1 | `g1_v3/knobs.py:345-346` com `g1_v3/cena.py:262` | Na escada aleatória o degrau é U(0; 0,20)·(0,5 + 0,5·t) (`primitive_terrains.py:1061`), e não t·0,20 como a PD-20 conta. Na linha 9 vai até 0,175 m, acima do teto de 0,15 m com caixa | 1 em 7 células de terreno da fatia B; 14% dos anéis dela passam de 0,15 m. E o primeiro degrau fica a 2,15 m do centro: a 0,25 m/s o CARREGAR nunca sai do plano nela | Linha 6 (t 0,5: até 0,15 m). P3 | E §2; S §11.2; PD-20 |
| R2 | `g1_v3/cena.py:265` | As caixas giram livres; a fuga da plataforma testa as meias medidas sem o giro (`primitive_terrains.py:806-829`). Uma caixa de 0,1 × 2,0 m girada 90° entra até 0,95 m na plataforma | Linha 7, 150 células: 76% com terreno > 1 cm dentro de \|x\|,\|y\| ≤ 0,95; 51% com > 5 cm. Zona dos pés da S4: 2,7% das (célula, borda); zona da caixa: 7,5%. O terreno é fixo no startup: as mesmas células erram em todo o treino | P4 | S §12 ("plataforma plana"); E §2 ("o PEGAR só no plano") |
| R3 | plano §12.3 G5, §6.3 "Fase G", §6.4; viab. D10; contrato do `exporta_cena.py:58-60` | Texto: (a) o "+1,4 m" da s31 é a linha 12 do ANDAR; na linha 9 a origem é 1,05 m; (b) "plataforma de 2 m" vale só nos obstáculos (1,92) e nas caixas: escadas 2,4, rampas estrela de 1,63–1,79 m de meia, aleatória 4,3; (c) o código põe `incl_max_seg, terreno` antes do `k` em `sucesso_do_segmento`; (d) a D10 conta 0,25 m/s e o treino sorteia U[0,25; 1,0]; (e) o contrato diz "o reset do exportador é plano" e omite que é ANDAR | Uma lane nova lê o contrato e erra (a s32 e a s33 erraram por (b)) | O orquestrador escreve antes do despacho (lane 7) | §6.3, §6.4, §12.3 |
| R4 | `registra_juntas_v3.py:157-183` | `_rz`, `_qmul` e a conta de `pose_no_sub_terreno` repetem `alvos.base_para_mundo` (o xy) e `estimador._quat_de_yaw` + `estimador._qmul` (o giro do quatérnio; a mesma composição está em `estimador.py:177`). O driver já carrega os dois (`pilota_v3.py:53-54`) e já chama `_AL.base_para_mundo` (`:324`). A F9 R3 pediu uma fonte | Duas fontes da mesma conta divergem sem aviso. +102 linhas no driver | xy por `pv._AL.base_para_mundo`, z por soma, quatérnio por `pv._ES._qmul(pv._ES._quat_de_yaw(...), q)`; saem `_rz` e `_qmul`. O `test_pilota.py:249` troca `R._rz` por `_AL` (510 → ~495) | §1.5; F9 R3; CLAUDE.md |

## Menor

| # | arquivo:linha | o que está errado | cenário | correção |
|---|---|---|---|---|
| M1 | `g1_v3/smoke.py:1064` | `alem` = plataforma/2 + 0,15 = 1,15 m. Na escada da linha 9 o plano vai até 1,20 m (medido: chão = origem em 1,19 m; −0,15 em 1,21 m) | s33: a base fica sobre o plano, 0,15 m baixa. `altura_pelve` lê 0,64 (≠ 0,789 ± 0,05) e o chão do joelho lê a origem (≠ origem − 0,15): os dois checks falham SEMPRE. s32: os pés entram 0,09–0,13 m no plano e a caixa fica sobre o anel (1,40 m), o contrário do roteiro. O check do termo `caixa_perdida` lê o `v3_perdeu_agora` do `env.reset()` (as terminações rodam antes da cadeia) e, com `caixa_perdida_vira_andar` = True, nunca dispara; se a penetração derrubar o robô, o reset zera o `v3_perdeu`. A s32 passa sem testar | Meia plataforma pelo cfg: `(tamanho − 2·border − 2·n·step_width)/2` do `cena.terreno_cfg().terrain_generator.sub_terrains["escada"]`, + `step_width`/2 = 1,35 m. A s32 checa só `v3_perdeu` e a queda falsa no passo |
| M2 | `g1_v3/smoke.py:1124-1126` | A DÚVIDA conta 10 degraus. São 6 (`border_width` 1,0 do preset, `config.py:51`): origem 1,05 m, 5 cm acima do "> 1,0" | Passa hoje, por pouco e pelo motivo errado | Check `== (n + 1)·degrau ± 1e-3`, com n do cfg; apagar a marca |
| M3 | `registra_juntas_v3.py:220`, `:326` | `--afasta` move a laje em +x do mundo; o treino a sobe 20 m (`eventos.laje_fora`) | Com `--borda` ≠ 0 a laje sai de lado, 10 m: fora do lidar (6 m), sem efeito na marcha de 2 m | Somar em z, como o treino; a marca sai |
| M4 | `registra_juntas_v3.py:325` | O comentário supõe `qvel_inicial` zero | Confere hoje (seção "sem achado") | `assert not np.any(c["qvel_inicial"])` no lugar do comentário |
| M5 | `g1_v3/metricas.py:27` | Importa o `comando` só pelo `FATIA_B` | Acopla a métrica a um módulo pesado | `FATIAS.index("B")` do `knobs`, já importado |
| M6 | `g1_v3/comando.py:309-314`, `:397-398` | No B4 da perda, σ e Φ seguem `m["h"]`; o ok segue `m["pelve"]` (sobre o pé mais baixo) | Com um pé em cada degrau divergem até 0,15 m (tolerância 0,03): Φ ≈ 1 e o B4 não fecha. Na fatia B nada paga o fecho do B4 depois da perda (o B_alvo só sem `v3_perdeu`), e a fatia C é plana | Sem conserto agora; unificar se o BOTAR for ao terreno |
| M7 | `registra_juntas_v3.py:186-191` | `--borda` sem `--coluna` é ignorada em silêncio | O registro sai no plano com a borda pedida no nome | Recusar `--borda` ≠ 0 sem `--coluna` |
| M8 | `g1_v3/smoke.py:1054-1072` | `_base_alem_da_borda` repete a escrita da base do `queda_depois_de` da s10 (`smoke.py:340-350`) | Duas cópias | Um `_poe_base(env, pose)`, como o `_poe_caixa_parada` |
| M9 | `registra_juntas_v3.py:85-91` | Três nomes (`escada`, `rampa`, `obstaculo`) para a mesma string; a coluna vem de `--coluna` | O nome do roteiro não diz o que roda | Um roteiro `terreno` |

---

## Pendências

Veredito de cada suspeita do orquestrador.

| item | veredito |
|---|---|
| G1: `env_origins` = centro da plataforma, na altura dela | Confirmado nas 7 colunas, no código e por raio. Mas a plataforma não tem 2 m em 5 delas (seção 1; R3, S1). |
| G1: outra escrita de dtype trocado | Nenhuma (seção 5). O env falso do teste é float32 e pega o `index_put`. |
| G1: o sorteio sem terreno correlaciona envs? | Não. Só muda a posição do fluxo; os uniformes seguem i.i.d. |
| G1: a laje corta o primeiro degrau nas invertidas? | Não. O degrau da escada_inv começa a 1,20 m e a rampa_inv a ≥ 1,63 m; a laje acaba em 1,05. A premissa "5 cm além da plataforma" só vale nos obstáculos e nas caixas, onde o corte é inofensivo (mocap; seção 1). |
| G3: ordem do `_mede_incl` | Certa (seção 4). |
| G3: σ do B4 por `h`, ok pela pelve sobre o pé | Confirmado; sem efeito na fatia B (M6). |
| G3: `d_pelve_p4` por `h` | Certo no plano da plataforma. Nas caixas que invadem a plataforma, ver R2. |
| G4: "base" = `root_link_pos_w` | OK (a pelve). O defeito é a meia plataforma e a média (S1). |
| G4: condição do terreno do CARREGAR | OK: é a S §11.2. |
| G6: duplicação no driver | Confirmada (R4). |
| G6: o reset do exportador cai em terreno? | Nunca: coluna 0, linha 0, z 0. Mas é a fatia ANDAR, e a cena exportada não tem caixa à frente do robô (S2). |
| G6: `--afasta` de lado | Confirmado, sem efeito prático (M3). |
| G6: `qvel_inicial` zero | Confere (M4). |
| G5: origem da escada ~1,65 m | A conta da suspeita erra: 6 degraus, 1,05 m. O plano (1,4 m) também: é a linha 12. A s31 passa com 5 cm (M2, R3). |
| G5: s32 a plataforma/2 + 0,15 | Confirmado e pior: o primeiro anel começa a 1,20 m, e a base fica sobre o plano. A s33 falha sempre; a s32 passa sem testar; a caixa no peito fica sobre o anel, e não sobre a plataforma (M1). |
| G5: smoke 1069 → 1234 e os auxiliares | `_caixa_no_peito` reusa `_poe_caixa_parada`. `_abre_b_na_escada` repete só as 3 escritas do `sorteia_fatia` (estado forçado; a s07 só lê). `_base_alem_da_borda` repete a s10 (M8). |

## Pendências do dono

| # | pergunta | opções | recomendação |
|---|---|---|---|
| P1 | O que é "fora da plataforma" na `fracao_fora_da_plataforma` (S1)? | A: \|xy − origem\|∞ > 1,0 m (hoje; erra 5 das 7 colunas). B: a meia plataforma real por coluna, uma tabela de 7 números tirada do cfg (a rampa é estrela: um quadrado só aproxima). C: o chão sob a pelve (`altura_pelve`) a mais de 1 cm da origem: o robô sobre terreno que não é o plano | C. Uma linha, sem tabela, exata nas escadas, nas rampas e na aleatória, e mede o que a D10 quer: passos de CARREGAR no terreno. Nos obstáculos e nas caixas conta só os passos sobre obstáculo. Com (b) do S1 em qualquer opção. |
| P2 | Alvo de transporte da qualidade do CARREGAR no terreno (S3). | A: chão sob a pelve + 0,90 (hoje). B: no frame da pelve, pelve + `caixa_transporte_b` (z incluso), só na célula `terreno`; A no plano. C: sola mais baixa (`z_pe_min`) + 0,90 | B (+2 linhas em `_qualidade`). No plano fica A, que pega o andar agachado (E §4.4). C também pula: o chão cai 0,15 m quando o pé da frente pousa, com a pelve ainda alta. |
| P3 | Escada aleatória na linha 9 (R1). | A: manter (até 0,175 m). B: `linha_max_carregar` = 6 (até 0,15 m) | B. É o teto da E §2. |
| P4 | Caixas que invadem a plataforma (R2). | A: manter. B: `box_yaw_range=(0, 0)` na coluna `caixas` (`cena.py:265`, 1 linha): a fuga fica exata; o ANDAR perde o giro das caixas. C: tirar `caixas` do `linha_max_carregar` (a fatia B não a usa) | B. Mantém a coluna no CARREGAR e o PEGAR no plano. |
| P5 | O CARREGAR de terreno anda a U[0,25; 1,0] m/s (PD-12; seção 2). 34% saem da célula de 8 m; a conta da D10 a 0,25 m/s não descreve o treino. | A: manter a PD-12. B: teto próprio do CARREGAR no terreno (campo novo) | A. Com essa velocidade o robô passa 31–58% do trecho fora do plano (pela borda real). Medir com a métrica consertada (S1, P1). |

## DÚVIDAS no código

| marca | veredito |
|---|---|
| `g1_v3/exporta_cena.py:58-60` (nada força a coluna `plano`) | Respondida: o reset do play cai sempre na coluna 0, z 0. Mas é a fatia ANDAR (S2). Reescrever o contrato depois do conserto. |
| `g1_v3/smoke.py:1124` (1,4 × 1,65 m) | A conta da marca erra: são 6 degraus, 1,05 m (M2). |
| `registra_juntas_v3.py:326` (`--afasta` de lado) | Confirmada, sem efeito prático; somar em z (M3). |
| `g1_v3/metricas.py:85` (F9: média do último passo × lote do reset) | Confirmada para valores correntes: na G4 a média transversal dá 0,054 contra 0,259 por episódio (S1). Vale também para `tempo_*` e `phi_*` (F9 M5), fora deste escopo. |

---

## Lanes de correção

Arquivos disjuntos dentro de cada onda. Contagem antes → depois (estimada).

**Onda 0 (orquestrador, Opus):** as decisões P1 a P5; a lane 7 escreve o plano antes dos testes.

**Onda 1 (testes, TDD, Haiku)**

| lane | arquivos | correções | contagem |
|---|---|---|---|
| 5 — testes | `g1_v3/testes/test_metricas_g.py`, `g1_v3/testes/test_pilota.py`, `g1_v3/testes/test_recompensas_g.py` (novo) | S1: a métrica loga a média dos episódios que fecham, e "fora" pela definição do P1; S3: a qualidade no terreno pelo frame da pelve (P2); R4: `pose_no_sub_terreno` sem `_rz`; M3: afasta em z; M7 | +~70 linhas de teste |

**Onda 2 (código, em paralelo)**

| lane | arquivos | correções | contagem |
|---|---|---|---|
| 1 — métrica | `g1_v3/metricas.py` | S1 (P1); M5 | 470 → ~474 |
| 2 — exportador | `g1_v3/exporta_cena.py` | S2 (fatia B forçada no plano); o texto do contrato (R3 e) | 641 → ~648 |
| 3 — recompensa | `g1_v3/recompensas.py` | S3 (P2) | 776 → ~778 |
| 4 — driver | `registra_juntas_v3.py` | R4; M3; M4; M7; M9; o comentário da `:62` | 510 → ~490 |
| 6 — terreno | `g1_v3/cena.py`, `g1_v3/knobs.py` | R2 (P4); R1 (P3) | 421 → 421; 483 → 483 |

**Onda 3**

| lane | arquivos | correções | contagem |
|---|---|---|---|
| 8 — smoke | `g1_v3/smoke.py` | M1; M2; M8 | 1234 → ~1230 |

**Lane 7 (orquestrador, Opus; antes da onda 1)**

| lane | arquivos | correções |
|---|---|---|
| 7 — plano | `docs/planos/2026-10-14-implementacao-g1-v3.md` | R3: §12.3 G5 (origem 1,05 m na linha 9; a base da s32 a 1,35 m); §6.3 "Fase G" (a plataforma real por coluna; a ordem dos argumentos do `sucesso_do_segmento`; a definição da G4 pelo P1 e a média por episódio; a qualidade no terreno pelo P2); PD-20 (P3, P4); §6.4 (o exportador força a fatia B); viab. D10 com a velocidade do treino (P5); §20: linha desta revisão. |

Ordem: lane 7, lane 5 (os testes falham), lanes 1 a 4 e 6, lane 8. Nenhum id bloqueia o treino. Antes do smoke da fase
I: M1 (a s33 falha sempre). Antes da medida da D10: S1. Antes de usar o registro de terreno: S2. Antes do treino de
terreno do CARREGAR: P2 (S3).
