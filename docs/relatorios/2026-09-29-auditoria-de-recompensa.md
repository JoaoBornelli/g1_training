# Auditoria de recompensa — g1_limpo (HEAD 7a23ea9, 29/09/2026)

Fonte das regras: `specs/g1-limpo-comportamento.md`. Cada achado passou por três céticos. Quando os céticos corrigiram a conta, este relatório usa a conta corrigida. Achados que descrevem o mesmo defeito estão fundidos. A severidade é a do auditor, e a nota do cético vem ao lado quando ele discorda.

Classes: A = sem gradiente; B = o proibido paga; C = ficar paga mais que avançar; D = penalidade taxa o certo; E = proibido sem preço; F = recompensa depende de estado não observado; G = contradição entre etapas, entre spec e código, ou entre doc e código.

## 0. Correções aplicadas (29/09, RUN zero14)

- Achado 4: CORRIGIDO na forma `unload × _trazer`, e não `postura_ereta × traz`. A `postura_ereta` chama o `unload` e herda o fator. A rampa da pelve é baixa na pega (~0,35 a 0,7), e a versão só na `postura_ereta` rendia ~+0,4/s. Contas, e não medida, sobre a economia do `model_1750`: pairar a 2 mm 15,2 → 13,5/s; erguer reto +4,9 → +6,6/s sobre pairar; erguer tombada 30° +1,8 → +3,0/s; gradiente no início da subida 0,08 → 0,14/s por cm.
- Achado 2: CORRIGIDO. O `PosturaPorElo` usa o regime parado quando `limpo_twist_zerado = 1`; o laço de rumo não troca mais o σ do punho e da cintura.
- Caminho B (`_alinha(aproximacao=True)` no `staged` e no `precise_pos`): já estava no HEAD auditado (`7a23ea9`, RUN zero13). A zero14 o leva junto.
- Leitura no log da zero14: no começo o `unload` e a `postura_ereta` caem, e isso não é regressão. O sinal de sucesso é `renda_congelada` e `sucesso` acima de zero.

- Achado novo do play, fora desta auditoria (29/09, RUN zero15): a caixa DEITAVA 90° com a face de cima para o peito, 43° aos 2 s e 89° aos 7,5 s (`juntas.csv`, zero12 it ~3100). O `caixa_na_pega` do log marcava ~8°, porque ele se dilui entre todos os envs. Consertos: a reta da aproximação passa a `max(0, 1 − Δθ/90°)` e zera a 90°; a âncora desce para z 0,75–0,85; o fecho do PEGAR exige `~apoiada`.

## 1. Tabela

| # | Título | Etapa | Classe | Veredito | Severidade | Arquivo:linha |
|---|---|---|---|---|---|---|
| 1 | Pairar no PEGAR_COM contra a cauda CARREGAR: fechar vence só pelo bônus da espera | PEGAR_COM → ESPERA_COM → CARREGAR | C | PLAUSIVEL | trava (cético: não medida) | knobs.py:1254-1272; recompensas.py:1240-1290 |
| 2 | O `pose` lê o regime pelo comando: o laço de rumo põe espera, pega e CAUDA no σ de walking | ESPERA, PEGAR, CAUDA ×8 | A/G | CONFIRMADO · CORRIGIDO zero14 | trava (cético: degrada a volta à pose) | recompensas.py:118-138; comando.py:1296-1308 |
| 3 | A cadeia C pula o CARREGAR: PEGAR→BOTAR é treinado e CARREGAR→BOTAR nunca | cadeias | G | CONFIRMADO | trava (impacto não medido) | comando.py:155-166, 718 |
| 4 | `unload` e `postura_ereta` pagam 4/s nos primeiros 2 mm e nada depois | PEGAR_COM | A | PLAUSIVEL · CORRIGIDO zero14 | trava | recompensas.py:739, 796 |
| 5 | A estátua com a caixa colhe ~77% da renda de andar; o ganho é ~+5 a +7/s, não +14/s | CARREGAR | C | PLAUSIVEL | trava | knobs.py:1223-1225, 1265-1266 |
| 6 | Na CAUDA a caixa some da observação, e a terminação por caixa fora do raio depende dela | CAUDA | F/E | PLAUSIVEL | trava | observacoes.py:139-141; comando.py:779; terminacoes.py:117-120 |
| 7 | REORIENTAR inerte: fecha por temporizador, derivada zero na ação | REORIENTAR | A | CONFIRMADO | trava (cético: tarefa sem aprendizado, decisão do dono) | comando.py:1382; knobs.py:338, 1068 |
| 8 | O fecho do REORIENTAR nunca sobe `_fechos`, mesmo com `reorientar_inerte=False` | REORIENTAR | C | PLAUSIVEL | trava (cético: latente, 0 hoje) | comando.py:1505; recompensas.py:1240 |
| 9 | O fecho do BOTAR não exige mãos livres e congela mais renda a quem fecha segurando | BOTAR → CAUDA | G + C | CONFIRMADO | paga hack | comando.py:1386, 1541-1544; recompensas.py:533-541, 1281-1285 |
| 10 | O impacto da caixa não tem preço; soltar de até 0,18 m sobre o alvo não termina nem custa | BOTAR | E + A | CONFIRMADO | paga hack | terminacoes.py:108-120; recompensas.py:799-831 |
| 11 | O `squeeze` paga apertar além do necessário; no CARREGAR o aperto custa 0 | PEGAR, CARREGAR | B / E | CONFIRMADO | paga hack | recompensas.py:690, 740-744, 774; knobs.py:947-954, 1261 |
| 12 | `apoiada` sem teto e `descarga` com clamp: prensar a caixa fica fora de todo termo | BOTAR | E | CONFIRMADO | paga hack (cético: E, não B) | comando.py:1365-1367; recompensas.py:690, 828 |
| 13 | O fecho do PEGAR não confere preensão nem as duas mãos | PEGAR | B | PLAUSIVEL | paga hack | comando.py:1384, 1515-1520 |
| 14 | Segurar com uma mão paga o mesmo que a pega bimanual no CARREGAR | CARREGAR | B | CONFIRMADO | paga hack | knobs.py:1261-1262; recompensas.py:620-633 |
| 15 | Contato com a laje: grátis abaixo de 50 N, derivada zero acima de 100 N, cinco partes fora do sensor | PEGAR, BOTAR | E | CONFIRMADO | paga hack | recompensas.py:189-190; cena.py:133-146; knobs.py:473-475, 1124 |
| 16 | A ESPERA_SEM zera os dois rastreios: deriva de rumo e de posição saem de graça | ESPERA_SEM (e PEGAR) | A/G | CONFIRMADO | paga hack (cético: A, não B) | knobs.py:1265-1266; comando.py:1318-1327 |
| 17 | Na ESPERA_COM a orientação e a sacudida da caixa são grátis | ESPERA_COM | A | PLAUSIVEL | paga hack (cético: janela curta) | knobs.py:1254-1264; comando.py:724-725 |
| 18 | A anuidade da `renda_congelada` paga o fecho antecipado acima do que o freio cobra | PEGAR, BOTAR | C | PLAUSIVEL | paga hack | recompensas.py:1227-1291; knobs.py:862-863 |
| 19 | Mão no chão e antebraço ou dorso na caixa não têm preço nem terminação | todas | E | PLAUSIVEL | paga hack (cético: só E provado) | terminacoes.py:149-155; recompensas.py:544-580 |
| 20 | A direção pedida do REORIENTAR é viva: andar em volta da caixa alinha a face | REORIENTAR | B | PLAUSIVEL | paga hack (latente) | comando.py:1908-1925; recompensas.py:665 |
| 21 | A renda congelada, o estado de espera e o `pegou` não são observados pelo crítico | todas as cadeias | F | CONFIRMADO | custa eficiência | observacoes.py:86; env_cfg.py:619-639; recompensas.py:1240-1298 |
| 22 | O sucesso conta no fecho; cair ou largar depois não desconta no nível nem no balanceador | currículo | G | CONFIRMADO | custa eficiência | comando.py:884-900, 742-760; curriculo.py:150-162 |
| 23 | O piso `frac_uniforme` sorteia sobre [0, max global] e o max nunca decai | currículo | A | CONFIRMADO | custa eficiência | curriculo.py:195-200 |
| 24 | O fecho e o sucesso do BOTAR não exigem robô de pé no fim | BOTAR → espera final | G | CONFIRMADO | custa eficiência | comando.py:1386, 1526-1530; curriculo.py:157 |
| 25 | A espera inicial esconde a caixa, contra o spec | ESPERA_SEM | G | CONFIRMADO | custa eficiência | observacoes.py:139-141; comando.py:779 |
| 26 | A cadeia (B contra C) não é observada no PEGAR | PEGAR | F | PLAUSIVEL | custa eficiência | observacoes.py:59-97; comando.py:160-170 |
| 27 | A laje do BOTAR sobe até 0,65 m, fora do spec e da tabela de IK | BOTAR | G + A | CONFIRMADO | custa eficiência | comando.py:1627-1641; recompensas.py:948 |
| 28 | Arrastar a caixa depois do fecho sai de graça entre 0,10 e 0,18 m | BOTAR → CAUDA | E | PLAUSIVEL | custa eficiência | terminacoes.py:117-119 |
| 29 | Na CAUDA, largar e sair da caixa só é pago pelo `pose` diluído em 29 juntas | CAUDA | A | PLAUSIVEL | custa eficiência | knobs.py:1263, 1267; recompensas.py:155, 791-796 |
| 30 | Arrastar a caixa na laje até o peito paga `staged` e `precise_pos` | PEGAR_COM | B | PLAUSIVEL | custa eficiência | recompensas.py:616, 632 |
| 31 | Acima de h = 0,68 a `forma_postural` pede o tronco a 22° | PEGAR_COM | G | PLAUSIVEL | custa eficiência | recompensas.py:948, 979 |
| 32 | No CARREGAR o tombo da caixa custa só 0,955/s por rad | PEGAR → CARREGAR | A/G | CONFIRMADO | custa eficiência | knobs.py:1255, 1260; recompensas.py:632 |
| 33 | A sacudida da caixa não tem canal | CARREGAR | E/A | CONFIRMADO | custa eficiência | recompensas.py:620-633; knobs.py:932 |
| 34 | O teto do `limite_de_junta` corta a derivada até o batente | todas | A | PLAUSIVEL | custa eficiência (inativo hoje) | recompensas.py:311-312 |
| 35 | A laje do BOTAR nasce por mocap a 0,10-0,30 m da pelve | troca → BOTAR | G | PLAUSIVEL | custa eficiência | comando.py:1643-1653, 1711-1713 |
| 36 | O docstring diz que `NoBatente` está ligada; ela não está | todas | G | CONFIRMADO | custa eficiência | knobs.py:1446-1448; terminacoes.py:158-196; smoke.py:660 |
| 37 | A `FaixaDePose` taxa o agachamento certo do BOTAR na laje baixa | BOTAR | D | PLAUSIVEL | custa eficiência | knobs.py:1365-1370; recompensas.py:253-255 |
| 38 | O `track_lin` tem σ fixo 0,5: o topo do envelope tem derivada quase zero | ANDAR, CARREGAR | A | PLAUSIVEL | custa eficiência | mjlab velocity_env_cfg.py:282; recompensas.py:432 |
| 39 | O freio taxa a frenagem no CARREGAR ao comando cair a zero | CARREGAR | D | PLAUSIVEL | custa eficiência | recompensas.py:1190-1237 |
| 40 | A altura de transporte é sorteada em 0,85-0,95 m; o spec fixa 0,95 m | CARREGAR | G | CONFIRMADO | custa eficiência | knobs.py:210, 219; comando.py:1179-1180 |
| 41 | O x,y do alvo fica no referencial da pelve, não do peito | CARREGAR | G | PLAUSIVEL | custa eficiência | comando.py:1726-1748 |
| 42 | Cair segurando custa menos que largar | CARREGAR | E | PLAUSIVEL | custa eficiência | env_cfg.py:237, 280-300 |
| 43 | O `air_time` segue em 0: a marcha só recebe punição | ANDAR | A/D | PLAUSIVEL | custa eficiência | knobs.py:438, 447-449 |
| 44 | Girar no lugar rende pouco mais que ficar parado | ANDAR | C | PLAUSIVEL | custa eficiência | recompensas.py:397-462 |
| 45 | A faixa de comando é ±1,0 lateral e −1,5 em ré, não ±2,0 nas quatro direções | ANDAR | G | CONFIRMADO | custa eficiência | env_cfg.py:496-531; mjlab velocity_env_cfg.py:191-192 |
| 46 | Braços abertos parado quase não custam | ANDAR, ESPERA_SEM | A | CONFIRMADO | custa eficiência | knobs.py:397-427, 1367 |
| 47 | O portão da forma é cego ao giro, à deriva e ao erro lateral | currículo | G | CONFIRMADO | custa eficiência | comando.py:2219-2251, 2166-2178; curriculo.py:456-470 |
| 48 | Com o comando a zero, os termos de pé desligam | ANDAR | A | PLAUSIVEL | custa eficiência | mjlab rewards.py (active); velocity_env_cfg.py:340-371 |
| 49 | O comentário do freio diz "média"; o código soma | velocidade_por_regime | G | CONFIRMADO | custa eficiência | knobs.py:790, 800-802; recompensas.py:1233 |
| 50 | O ANDAR standing não tem laço de rumo | ANDAR | A | PLAUSIVEL | custa eficiência | mjlab velocity_command.py:136-138; comando.py:313 |
| 51 | A massa da caixa define o F_ref e ninguém a observa | PEGAR, CARREGAR | F | PLAUSIVEL | custa eficiência | eventos.py:344-354; recompensas.py:594, 690 |
| 52 | O spec diz 0,3 s e 50% para o REORIENTAR inerte; o código faz 0,5 s e 5% | REORIENTAR | G | CONFIRMADO | custa eficiência | knobs.py:1060-1075; env_cfg.py:82 |
| 53 | k fixo em 1 e primitivas fora do spec no REORIENTAR | REORIENTAR | G | PLAUSIVEL | custa eficiência (latente) | comando.py:161-165; knobs.py:313-338 |

## 2. Renda por segundo: ficar contra avançar, por etapa

Unidade: renda/s, peso = valor/s, dt 0,02. Horizonte do PPO ~2 s (γ ≈ 0,99; γ não foi lido). Base medida no model_1600/zero11 (relatório 2026-09-29 §2.3), ajustada à HEAD (S1, S2, S3).

### 2.1 ANDAR

| Situação | Ficar | Avançar | Ganho |
|---|---|---|---|
| Comando de marcha (distribuição do treino) | ≈ 3,2 (piso medido no smoke 3,863) | ≈ 4,8-5,3 | +1,6 a +2,1 |
| Giro no lugar, \|wz\| ≥ 0,707 | 4,74 | ≈ 5,0-5,3 | +0,3 a +0,6 (achado 44) |
| Comando 2,0 m/s | 0,000 | a 1,0 m/s: 0,037 | derivada morre de 0 a ~1,2 m/s (achado 38) |
| Standing, cmd 0 | ≈ 6 | — | deriva de 0,077 rad/s custa 0,024/s; passo no lugar ≈ 0,005/s (achados 48, 50) |

### 2.2 Espera inicial (ESPERA_SEM) → PEGAR_SEM

- ESPERA_SEM: pose 1 + upright 1 ≈ 2,0/s. Rastreio e os sete termos valem 0. A deriva custa ≈ 0.
- PEGAR_SEM no VALIDA (σ = d0, alcançar 0,368): ≈ 4,7/s (smoke mediu 4,86 em 23/09).
- A troca é por temporizador. A política não escolhe ficar. Não há classe C. O defeito é a deriva grátis (achado 16) e o mesmo estado do ator render ~6/s no ANDAR e ~2/s na espera.

### 2.3 PEGAR

| Estado | Renda/s |
|---|---|
| PEGAR_SEM, mãos longe | ≈ 4,6 |
| PEGAR_SEM, mãos nas faces sem tocar | ≈ 9,7 |
| PEGAR_COM apertando na mesa | ≈ 13,7 (medido 12,7 na zero12) |
| Pairando 3 mm acima da mesa | ≈ 17,7 |
| Erguido até o alvo, tombo ~27° | ≈ 19,9 |
| Erguido reto | ≈ 22,4 |

A cadeia é monótona. Dentro da subida, o gradiente é raso: +6/s em 29 cm, e 4 desses 6 saem nos primeiros 2 mm (achado 4). Parado com a mão na caixa já colhe ~60% do teto.

### 2.4 PEGAR_COM → ESPERA_COM → CARREGAR ou BOTAR

- Pairar no alvo: ≈ 17,4/s a 39° de tombo; ≈ 19/s reto.
- Fechar: ESPERA_COM ≈ 23,2/s por 0,5 a 1,5 s; depois CARREGAR ≈ 16,5/s (política antiga) ou BOTAR ≈ 20,7/s.
- Com horizonte de 2 s: fechar ≈ (23,2 + 16,5)/2 = 19,9 contra 17,4 a 19. Margem de +0,9 a +2,5/s, antes do risco de `caixa_largada`. Sem desconto, o empate fica em T ≈ 7,4 s. MARGINAL (achado 1).
- O estado "fechou sem avançar" (32,3/s) foi eliminado pela S2.
- A espera paga mais que o BOTAR seguinte, mas é temporizada. Não é explorável.

### 2.5 CARREGAR

- ESPERA_COM ≈ C + 9 ≈ 22,8/s contra CARREGAR parado com cmd 0 ≈ C + 19 ≈ 32,8/s. A progressão sobe.
- Estátua sob comando de marcha: ≈ 25,4/s. Andar perfeito: ≈ 32,8/s. Ganho bruto ≈ +5 a +7,4/s, não +14/s. Break-even de risco ≈ 15 a 23%, não 45% (achado 5).
- O CARREGAR é cauda terminal. Não existe elo seguinte (achado 3).

### 2.6 BOTAR → CAUDA

| Estado | Renda/s |
|---|---|
| ESPERA_COM antes do BOTAR | ≈ 18,6 |
| Abertura do BOTAR, caixa no peito | ≈ 25,8 |
| Pairar a 10 cm / 2 cm | 33,0 / 34,6 |
| Apoiada no alvo, segurando | 38,6 |
| Apoiada com as mãos fora antes do fecho (ordem do spec) | 28,3 |
| CAUDA agachado | ≈ 38,5 a 43,8 |
| CAUDA de pé, mãos ao lado | ≈ 50,4 a 58,9 |

- O teto do que falta passa o piso do que foi feito em todo degrau, com uma exceção: soltar antes do fecho perde ~10/s ao vivo e ~7,6 a 7,9/s congelados até o fim do episódio (achado 9).
- Soltar de altura dentro de 0,18 m assenta mais cedo e custa 0 (achado 10).
- Com erro de rumo > 5,7°, o `pose` passa a walking: agachado cai de 2,4 para ~0,02/s (achado 2).
- Na CAUDA, recolher o braço rende ~+1/s e arrisca a terminação que custa a anuidade restante. Ficar com a mão na caixa é o ótimo local (achados 6, 29).

### 2.7 REORIENTAR

- Mão na caixa, caixa no lugar: ≈ 10,7/s. O temporizador fecha em 0,5 s.
- Depois do fecho vem a ESPERA_SEM, ~0/s mais pose, e o congelado soma 0 (`ganho` exclui o REORIENTAR). Com o elo inerte isto está certo. Com `reorientar_inerte=False`, o giro real perderia ~10,7/s no fecho (achado 8).
- A hipótese "ficar travado com a caixa deslocada 0,11 m rende 7,2/s" foi REFUTADA: o alvo segue a caixa todo passo (seção 4).

## 3. Achados

### 1. Pairar no PEGAR_COM contra a cauda CARREGAR
- **Etapa e transição:** PEGAR_COM → ESPERA_COM → CARREGAR (cadeias B e C desviada).
- **Regra violada:** §3 "teto do que falta fazer > piso do que já foi feito"; "Sem estátua"; princípio 6.
- **Classe:** C.
- **Arquivo:linha:** knobs.py:1254-1272; recompensas.py:1240-1290.
- **Conta (corrigida):** pairar 25,0 − 3,8 − 2,6 = 18,6/s; com a S3 ≈ 17,4/s (39°), ≈ 19/s reto. Fechar com horizonte de 2 s: (23,2 + 16,5)/2 ≈ 19,9/s. Margem +0,9 a +2,5/s. Sem desconto, pairar vence se o resto do episódio passa de 7,4 s. Os valores 23,2 e 16,5 são pré-S1, da política antiga. Com o CARREGAR em ~22,6 (número do plano S1), fechar vence em qualquer horizonte.
- **Veredito:** PLAUSIVEL. Margem fina, não inversão demonstrada.
- **Severidade:** trava o aprendizado (céticos: não medida).
- **Conserto:** não subir `renda_congelada` (×1,7 REJEITADO). Medir antes o retorno de fechar contra pairar (sonda de ~15 min). Se confirmar, tirar o `precise_pos` da coluna PEG_COM do teto de pairar.
- **Linhas:** 0 → 0 (medição; depois 1 número).

### 2. O `pose` lê o regime pelo comando
Funde: "pose da CAUDA ×8 e da ESPERA lê o regime pelo comando" e "o pose ainda escolhe o regime pelo COMANDO".
- **Etapa e transição:** ESPERA_SEM, ESPERA_COM, pega, CAUDA ×8.
- **Regra violada:** §3 Pose de referência; §4.2 "volta à pose de pé"; §5 "Regime vem do estado"; princípio 5.
- **Classe:** A/G (e F: o ator não vê o erro de rumo).
- **Arquivo:linha:** recompensas.py:118-138 (`PosturaPorElo`, `standing = total_speed < 0,05`); comando.py:1296-1308 (wz = 0,5·wrap(rumo_ref − rumo)); contraste com recompensas.py:1211 (`velocidade_por_regime` já lê o estado).
- **Conta:** erro de rumo > 0,1 rad (5,7°) dá \|wz\| > 0,05 e o `pose` passa a `std_walking`. Medido no 21700: 17,3% da espera, 28% do pouso, 31,3% da pega. Perto de pé: standing 0,978 contra walking 0,656; na CAUDA ×8, 7,83 contra 5,25/s, degrau de 2,6/s sem derivada. Agachado depois do BOTAR: standing 0,296 (×8 = 2,37/s) contra walking 0,0024 (0,019/s): canal morto justo em "levantar depois de largar". Os valores de 0,978/0,656/0,296 não foram refeitos pelos céticos; o limiar foi. O comentário em knobs.py:424-426 ("twist forçado a zero no PEGAR") está desatualizado.
- **Veredito:** CONFIRMADO.
- **Severidade:** trava o aprendizado (cético: degrada a volta à pose).
- **Conserto:** em `PosturaPorElo.__call__`, `standing |= env.limpo_twist_zerado > 0.5`, e walking/running com `~standing`. É o conserto de 21/09 aplicado ao termo que ficou de fora.
- **Linhas:** 155 → 157 a 158.
- **Estado:** CORRIGIDO na zero14 (29/09). Ver §0.

### 3. A cadeia C pula o CARREGAR
- **Etapa e transição:** CARREGAR → BOTAR (nunca treinada); PEGAR → BOTAR (treinada, proibida).
- **Regra violada:** §1 Composições treinadas: "ANDAR (parado) → PEGAR → CARREGAR → BOTAR → espera final"; "PEGAR → BOTAR sem CARREGAR não é treinada".
- **Classe:** G.
- **Arquivo:linha:** comando.py:155-166 (CADEIAS C = (PEGAR, BOTAR)); comando.py:718 (`tem_prox` exige elo != CARREGAR).
- **Conta:** frequência de CARREGAR→BOTAR = 0 por construção. Frequência de PEGAR→BOTAR = p_C > 0 (o valor 0,20 do balanceador não foi conferido; a conclusão não depende dele). O controlador real compõe ...→CARREGAR andando→parar→BOTAR, estado que a política nunca viu.
- **Veredito:** CONFIRMADO.
- **Severidade:** trava o aprendizado (céticos: não medida).
- **Conserto:** cadeia C = (PEGAR, CARREGAR, BOTAR). Fecho temporal curto do CARREGAR só nessa cadeia, reusando `_fecha_elo_corrente` e `_sustain_alvo_de`. Sem termo novo.
- **Linhas:** +3 a +8 em comando.py; 0 em recompensas.py.

### 4. `unload` e `postura_ereta` saturam em 2 mm
- **Etapa e transição:** PEGAR_COM, subida da mesa ao alvo.
- **Regra violada:** §4.3 "Ergue a caixa reta até o alvo de transporte"; princípios 1 e 5.
- **Classe:** A.
- **Arquivo:linha:** recompensas.py:739 (`descarga = clamp(1 − F/mg)`), :796 (`rampa × descarga`).
- **Conta:** a descarga salta de 0 a 1 em 2 mm (teste com a caixa teleportada). Com a pelve alta, 2·1 + 2·1 = +4/s aos 3 mm, e depois d/dz_caixa = 0 nos dois termos (a rampa lê a pelve). Não verificado pelos céticos: a divisão 46%/66% do ganho total e o 0,1/s por cm. O gradiente de subida restante vem de `staged` e `precise_pos`.
- **Veredito:** PLAUSIVEL.
- **Severidade:** trava o aprendizado.
- **Conserto:** na `postura_ereta`, trocar `descarga` por `descarga × traz` (fator que já existe no `staged`). Os céticos alertam que isso pode duplicar o gradiente do `staged`: revisar a dose.
- **Linhas:** +1.
- **Estado:** CORRIGIDO na zero14 (29/09). Ver §0.

### 5. A estátua com a caixa colhe ~77% da renda de andar
- **Etapa e transição:** CARREGAR, parado contra andando sob comando.
- **Regra violada:** §3 "Sem estátua"; §4.4 "Rastreia a velocidade com a caixa"; §5 "Não andou com a caixa".
- **Classe:** C (por break-even, não violação literal: a estátua rende menos que andar).
- **Arquivo:linha:** knobs.py:1223-1225 (docstring "parado = 13,79 + 3"), 1265-1266.
- **Conta (corrigida):** a estátua colhe `track_ang` 7·k_ang (k_ang ≈ 0,65 a 1, porque wz_cmd ≈ 0 com σ 0,707) e `track_lin` 7·exp(−v²/0,25) (2,0 a 0,56 m/s). Ganho bruto de andar: +4,4 a +9,4/s, centro ≈ +5 a +7. Dado medido no model_10200 escalado por 3,5: ≈ +5,2/s. Break-even de risco ≈ 14 a 30%, contra 45% do docstring. O líquido depois de action_rate, foot_*, freio e limite_de_pelve não foi medido.
- **Veredito:** PLAUSIVEL.
- **Severidade:** trava o aprendizado.
- **Conserto:** subir `track_lin` na coluna CARREGAR de 3,5 para ~6, ou baixar `track_ang` no CARREGAR para 1,5. Corrigir o docstring de `knobs.PesoPorEstado`.
- **Linhas:** 0 (números).

### 6. Na CAUDA a caixa some da observação
- **Etapa e transição:** CAUDA (espera final depois do BOTAR).
- **Regra violada:** §4.5 Proibido "Empurrar a caixa depois de soltar"; princípio 3. A citação de §4.2 "caixa já visível" é da espera INICIAL, não da final: não serve de apoio.
- **Classe:** F/E.
- **Arquivo:linha:** observacoes.py:139-141 (canais × (ELO publicado != ANDAR)); comando.py:779 (`publica_andar = soltou | ...`); terminacoes.py:117-120.
- **Conta:** depois do fecho os 10 canais da caixa valem 0 no ator e no crítico. A terminação dispara com \|\|caixa − alvo\|\| > `precise_pos_sigma`, sem limiar de velocidade. A perda é −4 mais a anuidade restante (~50/s × tempo restante; ~25 com < 0,5 s). O ganho de recolher o braço é ~1/s. O risco de arrasto depende do achado 9.
- **Veredito:** PLAUSIVEL.
- **Severidade:** trava o aprendizado.
- **Conserto:** o gate de `caixa_no_frame_da_base` lê o elo INTERNO quando `soltou = 1`. O one-hot publicado continua ANDAR. Com o achado 9 resolvido, o risco de arrasto cai junto.
- **Linhas:** 1 → 1 a 2.

### 7. REORIENTAR inerte fecha por temporizador
- **Etapa e transição:** REORIENTAR → PEGAR.
- **Regra violada:** princípios 1 e 5; §4.6 Fecho (girar 90°, erro < 10°).
- **Classe:** A.
- **Arquivo:linha:** comando.py:1382; knobs.py:1068 (`reorientar_inerte=True`), 338 (`voltas_max=(0,)*7`), 1063.
- **Conta (corrigida):** fecha = perto(d=0) & (alinhado | True) & ativo = ativo. O elo fecha em `sustenta_outros_s` = 0,5 s (não 0,3 s). d(fecho)/d(ação) = 0. Sorteado em 5% das cadeias de manipulação.
- **Veredito:** CONFIRMADO.
- **Severidade:** trava o aprendizado do REORIENTAR (céticos: decisão do dono documentada no §4.6; não trava as outras etapas).
- **Conserto:** decisão do dono. Para o incentivo existir: `reorientar_inerte=False` e `voltas_max=(0,0,1,1,1,1,1)`, junto com os achados 8, 20 e 53. Sem termo novo. Obs.: o spec não tem §8.3; a referência do knob é órfã, a fonte é o §4.6.
- **Linhas:** 0 (números).

### 8. O fecho do REORIENTAR nunca sobe `_fechos`
- **Etapa e transição:** REORIENTAR → REORIENTAR/PEGAR.
- **Regra violada:** princípio 6; §4.6 Cadeia.
- **Classe:** C.
- **Arquivo:linha:** comando.py:1505 (`ganho = origem != REORIENTAR`); recompensas.py:1281-1284.
- **Conta (corrigida):** hoje (inerte) o efeito é 0, e a exclusão é correta. A linha 1505 não lê `c.reorientar_inerte`, e contradiz o comentário da linha 1502 ("exceto o do REORIENTAR inerte"). Com o interruptor em False, o giro real fecha sem congelar. A perda estimada de ~10,7/s não foi derivada.
- **Veredito:** PLAUSIVEL.
- **Severidade:** latente (auditor: trava; céticos: não age hoje).
- **Conserto:** `ganho = (origem != REORIENTAR) | ~bool(c.reorientar_inerte)`.
- **Linhas:** 1 → 1.

### 9. O fecho do BOTAR não exige mãos livres
Funde três achados: "congela sem conferir sem contato nem de pé", "renda congelada paga segurar no fecho", "congela mais renda a quem fecha segurando".
- **Etapa e transição:** BOTAR → CAUDA.
- **Regra violada:** §4.5 Fecho "apoiada, parada por 0,5 s, sem contato com o robô"; §4.5 Proibido "Ficar apoiado depois de soltar"; §1 Sucesso; princípio 6.
- **Classe:** G + C.
- **Arquivo:linha:** comando.py:1385-1386 (fecha = perto & alinhado & apoiada); comando.py:1541-1544 (`_soltou` declarado no fecho); recompensas.py:533-541 (`_alcancar`), 1281-1285 (congela a soma do passo anterior). O docstring em recompensas.py:526 ("mãos livres") não é imposto pelo código.
- **Conta (corrigida):** alc(0,058 m, σ 0,08) = 0,591; alc(0,15 m) = 0,030. Congelado segurando ≈ 17,9 a 18,3/s; com as mãos soltas ≈ 10,3/s. Soltar antes do fecho, na ordem do spec, custa ≈ 7,6 a 7,9/s até o fim do episódio e ≈ 10/s ao vivo no sustain. A mão que empurra soma na força da laje: ∂apoiada/∂empurrar ≥ 0. O "de pé" saiu de propósito na v3.4 (reprovava 99,975% pelo joelho); fica só o "sem contato" (ver achado 24). O p50 de 1,24·m·g citado não foi achado nos docs; o achado vale sem ele.
- **Veredito:** CONFIRMADO.
- **Severidade:** paga hack.
- **Conserto:** no ramo BOTAR de `_fecha_elo_corrente`, fazer AND com `~(found_E | found_D)` dos sensores de palma que já existem. Para a renda não cair ao soltar: `staged` e `precise_ori` no BOTAR deixam de exigir `alcancar` depois de apoiada e perto, reusando a máscara do `load`. Sem termo novo.
- **Linhas:** +2 a +3 em comando.py; ~+1 em recompensas.py.

### 10. O impacto da caixa não tem preço
Funde: "impacto sem preço, soltar de até 18 cm" e "soltar de altura é o caminho mais rápido para o fecho".
- **Etapa e transição:** BOTAR (também PEGAR).
- **Regra violada:** §2 "não pode sofrer impacto"; §4.5 Proibido "Soltar de altura"; §5 "Soltou rápido | Impacto sem preço".
- **Classe:** E + A.
- **Arquivo:linha:** terminacoes.py:108-120 (`soltou_fora` exige `~no_alvo`, raio 0,18); recompensas.py:799-831 (`load` não lê velocidade nem pico); comando.py:1365-1367.
- **Conta:** queda de 0,18 m: v = 1,88 m/s > v_solta 1,2, e não termina. Fora do raio a queda termina acima de h = 0,073 m. `load` = perto para todo F ≥ m·g: d(load)/d(pico) = 0 e d(load)/d(v_z) = 0. O `load` paga com d < tol_pos 0,10 m. Medido: `impacto_da_caixa` plano em 6,27 / 8,66 / 6,79 × m·g por 25k its. Ressalva: a métrica é o pico do EPISÓDIO e inclui a pega (2,5 g).
- **Veredito:** CONFIRMADO.
- **Severidade:** paga hack.
- **Conserto:** consertar a forma do `load`, não criar termo. Duas formas escritas: dobradiça no pico, `load = (1−descarga)·perto·(1 − relu(pico/(m·g) − 1,5)/k)`, ou sino na velocidade vertical `exp(−(v_z/0,1)²)` mais v_z baixo como condição de `apoiada`. A escolha entre pico e excesso instantâneo está pendente no plano 2026-09-21-cuidado-com-a-caixa. O pico acumulado no episódio cobraria no `load` um impacto da pega.
- **Linhas:** ~+3.

### 11. O `squeeze` paga apertar além do necessário
Funde: "squeeze paga apertar além", "aperto além é PAGO: squeeze, unload e postura_ereta", "esmagar não tem preço no CARREGAR".
- **Etapa e transição:** PEGAR (pago); CARREGAR (sem preço).
- **Regra violada:** §2 "amassado por aperto"; §3 "sem aperto além do necessário"; §4.3 Proibido "Apertar além do necessário"; §5 "Esmagou a caixa".
- **Classe:** B no PEGAR; E no CARREGAR.
- **Arquivo:linha:** recompensas.py:690 (`tanh(F/F_ref)`), 740-744 (porteiro do `unload`), 774 (`postura_ereta` via `unload`); knobs.py:947-954 (débito registrado), 1261.
- **Conta:** F_ref = 1,0·9,81/1,6 = 6,13 N. tanh(1) = 0,762; tanh(3) = 0,995. No PEGAR os três termos somam peso 1 + 2 + 2 = 5: apertar 3× em vez de 1× rende +1,17/s com a caixa erguida (derivado) e +0,23/s na mesa (regime medido de 4,9×). Em 4,9·F_ref a derivada é sech² = 2,2e-4: nada puxa de volta. O máximo do termo fica em F → ∞. No CARREGAR as três colunas valem 0: 68,9 N (11× os 6,13 N) custam 0.
- **Veredito:** CONFIRMADO.
- **Severidade:** paga hack.
- **Conserto:** trocar o `tanh` por um pico em F_ref, por ex. `clamp(F/F_ref, 0, 1) − ½·relu(F/F_ref − 1,5)`; `unload` e `postura_ereta` herdam pelo mesmo helper. No CARREGAR, juntar com o achado 14 (gate de preensão com queda acima de ~2·F_ref). Cuidado: o termo é congelável; mexer na forma altera o valor do fecho num resume.
- **Linhas:** ~2 trocadas, +1 no helper.

### 12. `apoiada` sem teto e `descarga` com clamp
- **Etapa e transição:** BOTAR.
- **Regra violada:** §3 "Caixa íntegra: sem aperto além do necessário"; §4.5 "Apoia a caixa sem impacto".
- **Classe:** E (céticos: B só no sentido fraco; prensar paga igual, não mais).
- **Arquivo:linha:** comando.py:1365-1367 (`apoiada = F_z ≥ 0,5·m·g`); recompensas.py:690 (squeeze × `_fora_do_botar` = 0); recompensas.py:828.
- **Conta:** caixa de 1 kg: limiar 4,9 N. Para F ≥ 9,81 N, `load` = 1·perto e dload/dF = 0. Prensar com 1,24 ou 7·m·g rende o mesmo que apoiar com 1·m·g; custo 0. A aresta a 7·m·g fica fora do `perto`. O plano de 14/09 (apoiada como faixa) não foi aplicado.
- **Veredito:** CONFIRMADO.
- **Severidade:** paga hack (corrigida: proibido sem preço).
- **Conserto:** `0,5·m·g ≤ F_z ≤ m·g + folga_N` no `apoiada`, e a mesma faixa no `(1 − descarga)` do `load`, com rampa descendo acima de m·g + folga.
- **Linhas:** 1 → 2 em `_fecha_elo_corrente`; 1 → 2 em `load`.

### 13. O fecho do PEGAR não confere preensão
Funde: "não exige as duas mãos: o fecho escorado congela a renda" e "não confere preensão nem as duas mãos".
- **Etapa e transição:** PEGAR → ESPERA_COM.
- **Regra violada:** §4.3 Fecho "Caixa presa nas duas mãos"; Proibido "com o antebraço"; §3 Pega "nunca em uma mão só"; §5 "Escorar comprava o fecho sem preensão".
- **Classe:** B.
- **Arquivo:linha:** comando.py:1383-1384 (`fecha = perto & alinhado & de_pe`); comando.py:1515-1520 (força `_pegou=True`, comentário A3).
- **Conta:** o portão não lê força. Caminho escorado: `squeeze`, `unload`, `postura_ereta` ≈ 0; congelado ≈ 13 contra ≈ 18 da pega certa (−5/s), mas o fecho abre ESPERA_COM e CARREGAR com ganho > 17/s. Os valores 14,6 e 9,6/s vêm da memória "Escorar COMPRA o fecho", que trata do BOTAR; não derivam do PEGAR. A viabilidade física do escorado com ANG ≤ 25° no alvo do peito não foi medida.
- **Veredito:** PLAUSIVEL (portão confirmado; magnitude não).
- **Severidade:** paga hack.
- **Conserto:** somar `preso = min(F_n_E, F_n_D) ≥ F_ref` (mesmo `_forca_das_palmas`). Risco: o comentário A3 diz que o sensor de palma pode nunca disparar; medir a taxa de disparo antes, para não deixar o PEGAR inerte até o time_out.
- **Linhas:** +2.

### 14. Segurar com uma mão paga igual no CARREGAR
- **Etapa e transição:** CARREGAR.
- **Regra violada:** §4.4 Proibido "Segurar com uma mão"; §3 Pega.
- **Classe:** B.
- **Arquivo:linha:** knobs.py:1261-1262 (squeeze e unload na coluna 7 = 0); recompensas.py:620-633 (`precise_pos` não lê palmas).
- **Conta:** `precise_pos` = 3·exp(−(0,05/0,18)²)·1 = 2,78/s com uma ou duas mãos. Diferença de renda = 0. Só `caixa_largada` separa, e ela pune a queda, não a mão única.
- **Veredito:** CONFIRMADO.
- **Severidade:** paga hack.
- **Conserto:** multiplicar o `precise_pos` por `tanh(min(F_E, F_D)/F_ref)` só no CARREGAR. Junto, a queda acima de ~2·F_ref do achado 11.
- **Linhas:** +3.

### 15. Contato com a laje
Funde três achados: "abaixo de 50 N grátis, acima de 100 N sem derivada, partes fora do sensor", "tocar a laje abaixo de 50 N no BOTAR", "tocar a laje com mão, dorso ou tronco no PEGAR".
- **Etapa e transição:** PEGAR, BOTAR (termo global).
- **Regra violada:** §3 Contato "Nada do robô toca a laje"; §4.3 e §4.5 Proibido "Tocar a laje".
- **Classe:** E (+ canal morto abaixo do joelho da rampa).
- **Arquivo:linha:** recompensas.py:189-190 (`clamp((F−50)/50)`); knobs.py:473-475 (−2 cada), 1124, 1130; cena.py:99, 133-146 (grupos).
- **Conta:** custo/s = 2·clamp((F − 50)/50, 0, 1) por grupo. F < 50 N: 0, derivada 0 (≈ 14,6% do peso do robô de ~343 N; cobre o peso inteiro de uma caixa de 1 a 5 kg). 50-100 N: 0,04/s por N. F > 100 N: 2/s fixo, derivada 0. Antebraço, punho, cotovelo, canela, pé: 0/s a qualquer força. O motivo da exclusão (a terminação matava a exploração) caiu quando a terminação virou multa em 01/09 (terminacoes.py:30). O pé só alcança a laje nos níveis de topo 0,04 m. Nenhuma medição mostra o robô escorando abaixo de 50 N.
- **Veredito:** CONFIRMADO (mecanismo); a exploração no ponto de operação não foi medida.
- **Severidade:** paga hack.
- **Conserto:** pôr antebraço, punho, canela e pé nos grupos que existem. Trocar a rampa por `relu(F − F0)/escala` sem teto, com F0 ≈ 5 a 10 N (ou joelho_N 5 e saturação 50). Rever o peso junto. Medir antes a força palma-laje no BOTAR com o sensor `contato_palma`.
- **Linhas:** ~4 trocadas.

### 16. A ESPERA_SEM zera os dois rastreios
Funde três achados: "ESPERA_SEM zera os dois rastreios", "a espera inicial não paga rastreio", "deriva, giro e passo sem comando grátis na ESPERA_SEM e no PEGAR".
- **Etapa e transição:** ESPERA_SEM; em parte PEGAR_SEM, PEGAR_COM, REORIENTAR_SEM.
- **Regra violada:** §4.2 "ANDAR com comando zero, parado"; §4.1 Proibido "deriva sem comando"; §3 Locomoção; princípio 1.
- **Classe:** A e G (contra a coluna ANDAR). Céticos: não é B; a deriva não rende, só custa 0.
- **Arquivo:linha:** knobs.py:1265-1266 (ESP_SEM = 0,0; as linhas 1286-1287 do auditor estão desatualizadas); comando.py:1318-1327.
- **Conta:** rastreio × 0 = 0; os sete termos e a `forma_postural` × 0 = 0, então o fator de rumo multiplica zero. Deriva de 0,2 m/s custa 0,30/s no ANDAR e 0 na espera. O mesmo estado do ator (one-hot ANDAR, caixa zerada) rende ~6/s no ANDAR e ~2/s na espera. O zero veio do antigo `rastreio_por_elo`, cujo motivo (estátua do PEGAR) não vale para uma janela de temporizador. O −0,38 rad/s foi medido na pega do 19300, não na espera; no 20500 a deriva caiu. Custo indireto que sobra: `pose` ×1, `faixa_de_pose` e o freio acima do vmax standing. No PEGAR o rumo morde pelo `staged`; a deriva em xy não tem preço.
- **Veredito:** CONFIRMADO (mecanismo); magnitude na espera não medida.
- **Severidade:** paga hack (corrigida: custa eficiência).
- **Conserto:** 0,0 → 1,0 nas duas células ESP_SEM. A espera tem duração sorteada: no máximo ~6 de renda por episódio, sem piso de estátua. Conferir que a espera não passa a pagar mais que o início do PEGAR_SEM (princípio 6).
- **Linhas:** 0 (2 números).

### 17. Na ESPERA_COM a caixa é grátis
- **Etapa e transição:** ESPERA_COM (depois do fecho do PEGAR, antes de CARREGAR ou BOTAR).
- **Regra violada:** §2 "A face de cima fica normal ao solo o tempo todo"; §3 Caixa nivelada; §3 Troca de tarefa.
- **Classe:** A.
- **Arquivo:linha:** knobs.py:1254-1264 (ESP_COM = 0 nos sete termos); comando.py:724-725 (o avanço confere só `_perto`).
- **Conta (corrigida):** R = 4·pose + rastreio + upright. Tombar pelo ombro ou cotovelo: ∂R/∂θ = 0. Tombar pelo punho tem preço pelo `pose` ×4. A janela é de 0,5 a 1,5 s. Depois, o BOTAR volta a cobrar (~1,3/rad ou mais) e o CARREGAR cobra 0,32/rad. O defeito real é a entrada tombada sem gate.
- **Veredito:** PLAUSIVEL.
- **Severidade:** paga hack (céticos: grátis por pouco tempo, não pago).
- **Conserto:** somar `alinhado` ao gate de avanço em `_aplica_espera`. Opcional: `precise_pos` = 1 na coluna ESP_COM (conferir a `renda_congelada`, que lê os sete termos pelo nome).
- **Linhas:** +1 (e 1 número).

### 18. O fecho antecipado é pago acima do freio
Funde: "correr compensa: a anuidade paga o fecho antecipado" e "a renda congelada paga por fechar CEDO".
- **Etapa e transição:** PEGAR, BOTAR (fecho).
- **Regra violada:** §3 Movimento controlado "demorar é aceitável, correr não"; princípio 2.
- **Classe:** C (B no texto de um achado; céticos: C/D, pressa sem preço).
- **Arquivo:linha:** recompensas.py:1227-1237 (dobradiça zero até vmax), 1240-1291; knobs.py:862-863.
- **Conta (corrigida):** o ganho de antecipar o fecho em Δt é (r_depois − r_antes)·Δt, pago UMA vez, e vale r_next, não S. PEGAR: 4,6·Δt; conta do dono: 6,9 por 0,5 s. Com γ 0,99, fechar 1 s antes vale ≈ 0,79·S. Freio por junta = 6/29 × 1,5^degrau: 0,207 (d0), 0,31 (d1), 0,466 (d2). Empate do dono: 0,41 por junta. A pressa vence nos degraus 0 e 1 por ~2×, e perde a partir do degrau 2. A razão "10 a 20×" não se sustenta. O "freio −0,01/s" citado vem do YAM e sai da conta. O "~14" do knobs foi medido no model_10200, antes da tabela.
- **Veredito:** PLAUSIVEL.
- **Severidade:** paga hack nos degraus 0-1.
- **Conserto:** não mexer na `renda_congelada` (catraca e ×1,7 REJEITADOS no §6). Os achados 10 e 33 devem morder a velocidade da caixa ou da mão. Medir o `freio_degrau` médio contra o prêmio.
- **Linhas:** 0.

### 19. Mão no chão e antebraço na caixa sem preço
- **Etapa e transição:** todas (mão no chão); PEGAR e CARREGAR (antebraço).
- **Regra violada:** §3 Contato; §4.3 Proibido "com o antebraço".
- **Classe:** E.
- **Arquivo:linha:** terminacoes.py:149-155 (`Caiu`: 70° e joelho z < 0,05); recompensas.py:544-580 (só os pads de palma); cena.py:345-437.
- **Conta:** nenhum sensor lê mão×chão ou antebraço×caixa; o sensor dorso×caixa existe e só alimenta métrica. Custo 0/s. O ganho do antebraço no `unload` não foi medido: o `unload` usa `min(F_palma)`, então exige força nas duas palmas.
- **Veredito:** PLAUSIVEL (ausência de preço confirmada; hack não).
- **Severidade:** paga hack (corrigida: proibido sem preço).
- **Conserto:** somar `pad z < folga` ao `Caiu`. Pôr os geoms de antebraço no sensor dorso_E/D e usar a força deles como divisor da preensão no `unload`.
- **Linhas:** ~+3.

### 20. A direção pedida do REORIENTAR é viva
- **Etapa e transição:** REORIENTAR (latente: o elo é inerte).
- **Regra violada:** §4.6 Comportamento; princípio 2.
- **Classe:** B.
- **Arquivo:linha:** comando.py:1908-1925 (`viva` = pelve − caixa recalculada todo passo), 434-440; recompensas.py:665.
- **Conta:** um passo lateral de 1 m com a caixa a 0,32 m gira a direção pedida em 72,3°, sem girar a caixa. Renda realizada hoje ≈ 0. O ~1,1/rad do `precise_ori` não foi verificado.
- **Veredito:** PLAUSIVEL.
- **Severidade:** paga hack quando o REORIENTAR for ligado.
- **Conserto:** capturar a direção uma vez na abertura do elo, junto do `_captura_cima`, ou derivá-la do eixo da primitiva.
- **Linhas:** igual ou menos.

### 21. Renda congelada e estado de espera não observados pelo crítico
Funde quatro achados: "a renda congelada não aparece no crítico", "o valor congelado do BOTAR depende do instante do fecho", "a renda congelada é 42% do CARREGAR", "renda_congelada e ESPERA_COM não observáveis".
- **Etapa e transição:** todas as cadeias.
- **Regra violada:** §5 "Renda congelada ... inobservável | canal no crítico"; §3 Progressão na cadeia.
- **Classe:** F.
- **Arquivo:linha:** observacoes.py:86 (`um_de_cinco_interno`); env_cfg.py:619-639; recompensas.py:1240-1298; smoke.py:2426 (ordem do grupo critic).
- **Conta (corrigida):** o congelado é a soma dos sete termos no passo anterior ao fecho, por env: 13,8 a 16,5 após o PEGAR, ~30 na cauda C; no BOTAR varia de ~10,4 a ~18,3 conforme a mão (achado 9). No CARREGAR medido: congelado 14,7 de 16,5/s = 89%, não 42% (o §5 dá 57% no agregado). Na ESPERA_COM, publicado = interno = PEGAR: o crítico não separa o fim do PEGAR_COM ao vivo do início da espera, nem vê o relógio da espera. Pairar contra fechado é separável pela pose da caixa. O erro de ±4 no valor (±2/s × 2 s) é estimativa.
- **Veredito:** CONFIRMADO.
- **Severidade:** custa eficiência.
- **Conserto:** canal só no crítico, no fim do grupo: `renda_congelada.congelado/~30`; trocar o one-hot de 5 elos pelo `limpo_estado` de 10 estados. Não é o multiplicador por posição rejeitado no §6. Ajustar o smoke da linha 2426.
- **Linhas:** +8 a +12.

### 22. O sucesso conta no fecho
- **Etapa e transição:** currículo `nivel` e balanceador B/C.
- **Regra violada:** §1 Sucesso do ciclo (robô e caixa não caem); §5 "nível saltou sobre competência zero".
- **Classe:** G.
- **Arquivo:linha:** comando.py:884-900 (`concluiu`), 742-760 (a cauda não limpa `fechou`); curriculo.py:150-162.
- **Conta:** cadeia B (n = 1): o fecho do PEGAR dá `concluiu = True` até o reset; cair no CARREGAR ainda dá +1 no nível e em s_B. Cadeia C: queda na CAUDA depois do BOTAR também conta. O passeio ±1 equilibra em P(fechar) = 0,5; o sucesso real no equilíbrio é 0,5·(1 − q), q não medido.
- **Veredito:** CONFIRMADO.
- **Severidade:** custa eficiência.
- **Conserto:** `concluiu ∧ ¬terminated` no currículo e no balanceador (usar `terminated`, não `reset_buf`, para não contar o time_out como falha).
- **Linhas:** +2.

### 23. O max do `frac_uniforme` nunca decai
- **Etapa e transição:** currículo `nivel`.
- **Regra violada:** §5 "frac_uniforme empurra a média"; princípio 1.
- **Classe:** A.
- **Arquivo:linha:** curriculo.py:195-200 (`abertos = int(buf.max()) + 1`).
- **Conta:** 20% dos resets sorteiam k em [0, max]. P(nenhum reset volta ao topo) = (1 − 0,2/7)^N ≈ 5e-7 com N = 500. Fração dos episódios de cadeia acima do nível 0: 0,2·6/7 = 17% (15% com max = 3). Os envs de locomoção também seguram o max.
- **Veredito:** CONFIRMADO.
- **Severidade:** custa eficiência.
- **Conserto:** max só dos envs não sorteados (`buf[~sorteado].max()`), ou o maior nível com sucesso acima de um limiar (relatório de 29/09, item 12).
- **Linhas:** ±2.

### 24. O sucesso do BOTAR não exige robô de pé no fim
- **Etapa e transição:** BOTAR → espera final.
- **Regra violada:** §4.5 Fecho "Robô de pé"; §1 "termina de pé, livre".
- **Classe:** G.
- **Arquivo:linha:** comando.py:1386, 1526-1530 (sucesso no fecho); curriculo.py:157.
- **Conta:** d(sucesso)/d(pelve) = 0. Agachado até o time_out conta +1 no nível. A pose de pé só entra como renda da cauda.
- **Veredito:** CONFIRMADO.
- **Severidade:** custa eficiência.
- **Conserto:** não voltar o `de_pe` ao fecho do apoio. Medir o sucesso da cadeia C em `_aplica_espera`, no passo em que a espera final acaba: `fechou & soltou & pelve ≥ pelve_alvo & palmas sem contato`. Gravar dentro do mesmo episódio (evitar o atraso de métrica do comentário comando.py:1520-1525).
- **Linhas:** ~+2.

### 25. A espera inicial esconde a caixa
- **Etapa e transição:** ESPERA_SEM → PEGAR.
- **Regra violada:** §4.2 "de 0,5 a 1,5 s, com a caixa já visível".
- **Classe:** G.
- **Arquivo:linha:** observacoes.py:139-141; comando.py:779 (não 792).
- **Conta:** aguardando & ¬pegou → publicado ANDAR → os 10 canais valem 0. A caixa aparece no passo em que o one-hot vira PEGAR. A recompensa não muda. Decisão de 02/09 (memória janela-de-espera), não registrada no §6.
- **Veredito:** CONFIRMADO.
- **Severidade:** custa eficiência.
- **Conserto:** decisão do dono: tirar "com a caixa já visível" do §4.2, ou o gate da caixa ler o elo interno na espera inicial.
- **Linhas:** 0 ou 1.

### 26. A cadeia B contra C não é observada no PEGAR
- **Etapa e transição:** PEGAR (cadeias B e C).
- **Regra violada:** §3 Progressão na cadeia; §5 "canal no crítico".
- **Classe:** F.
- **Arquivo:linha:** observacoes.py:59-97; comando.py:160-170, 1600-1609 (mesmo alvo nas duas cadeias).
- **Conta:** V ≈ p_C·V_C + (1 − p_C)·V_B. A diferença ΔV (~8 a 28) não foi derivada: depende de γ, da espera e de p_C.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência (variância do valor, sem caminho de hack).
- **Conserto:** junto do achado 21, um bit `passo < n−1` só no crítico, derivado de `_passo` e `n_elos_da_cadeia`.
- **Linhas:** +2 a +3.

### 27. A laje do BOTAR sobe até 0,65 m
- **Etapa e transição:** BOTAR.
- **Regra violada:** §2 Lajes "entre 0 e 0,55 m".
- **Classe:** G + A.
- **Arquivo:linha:** comando.py:1627-1641 (teto `_TOPO_TETO_FISICO = 0,80`); recompensas.py:948 (`h.clamp(h[0], h[-1])`).
- **Conta:** topo_max = min(0,55 + 0,10; fundo − 0,05; 0,80) = 0,65. Chave até 0,78; a tabela vai até h = 0,68. De 0,68 a 0,78, dref/dh = 0. Cerca de 25% das aberturas a partir de topo0 = 0,55 passam de 0,55. O termo segue com gradiente na pose; falta só o ajuste à altura.
- **Veredito:** CONFIRMADO.
- **Severidade:** custa eficiência.
- **Conserto:** teto = min(`_TOPO_TETO_FISICO`, `prateleira_topo_teto`).
- **Linhas:** 0 a +1.

### 28. Arrastar a caixa depois do fecho sai de graça
- **Etapa e transição:** BOTAR → CAUDA.
- **Regra violada:** §4.5 Proibido "Empurrar a caixa depois de soltar"; Fecho "parada por 0,5 s".
- **Classe:** E.
- **Arquivo:linha:** terminacoes.py:117-119; `renda_congelada` (valor fixo).
- **Conta:** entre 0,10 m (fecho) e 0,18 m (raio) o custo é 0 e a derivada é 0; acima de 0,18 m, degrau de terminação (pedido do dono, não reaberto). Antes de soltar, só v_rel > 1,2 m/s termina: arrastar devagar é livre em qualquer distância. Não há medição de arrasto na cauda.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência.
- **Conserto:** resolvido pelo achado 9: com o fecho só com a mão fora, a caixa está solta quando a renda congela.
- **Linhas:** 0.

### 29. Na CAUDA, largar e sair só é pago pelo `pose` diluído
- **Etapa e transição:** CAUDA.
- **Regra violada:** §4.5 "solta e volta à pose de pé, mãos ao lado"; princípio 1; §5 "Cauda: ficou apoiado depois de soltar".
- **Classe:** A.
- **Arquivo:linha:** knobs.py:1263, 1267; recompensas.py:155, 791-796; env_cfg.py:685.
- **Conta (corrigida):** `rampa_cauda` = clamp((z − 0,32)/(0,78 − 0,32)): 8/0,46 = 17,4/s por m, cheia em 0,78 m (não 0,75). Braço: 8·p·2e/(σ²·29) ≈ 0,3/σ² por s por junta. `faixa_de_pose` CAUDA cobra 0 até 0,6 rad. Nada lê o contato palma-caixa na CAUDA. Que a pelve apoiada esteja ≥ 0,78 m não foi medido (a memória registra 0,33 a 0,62 no fecho).
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência.
- **Conserto:** quase todo pelo achado 9. Se sobrar: `braco_pos` da coluna CAUDA na `FaixaDePose` de 0,6 para 0,3.
- **Linhas:** 0 (1 número).

### 30. Arrastar a caixa na laje até o peito paga
- **Etapa e transição:** PEGAR_COM.
- **Regra violada:** §4.3 Proibido "Arrastar a caixa pela laje além do ajuste de pega".
- **Classe:** B.
- **Arquivo:linha:** recompensas.py:616 (`traz`), 632.
- **Conta:** σ_trazer ≈ 0,29: arrastar 0,15 m em x (d 0,29 → 0,216) rende `staged` +0,62/s e `precise_pos` +0,49/s, total ≈ +1,1/s, sem custo de tombo e sem terminação antes do fecho. A subida paga também o `unload`, que o arrasto não paga. O Δx por nível não foi medido.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência.
- **Conserto:** no `staged`, `traz × descarga`. Risco: a descarga vale ≈ 0,0005 na laje e zera o gradiente de `trazer` antes de sair da laje; o `alcançar` continua.
- **Linhas:** 1 → 1.

### 31. Acima de h = 0,68 a `forma_postural` pede o tronco a 22°
- **Etapa e transição:** PEGAR_COM perto do alvo.
- **Regra violada:** §3 Postura "tronco ereto, inclina só quando o alvo exige"; §4.3.
- **Classe:** G.
- **Arquivo:linha:** recompensas.py:948, 979 (chave = `caixa_z` fora do BOTAR).
- **Conta:** última linha do .npz: tronco 22,06°, pelve 0,72 m (lido por um cético). Tronco a 0°: perda ≈ 0,12/s, derivada 0,0056/s por grau rumo a 22°; pelve ereta soma ≈ 0,10/s; total ≈ 0,22/s contra `de_pe` e `postura_ereta`. Não se sabe se o robô ainda está em PEGAR_COM nessa altura.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência.
- **Conserto:** chave = min(`caixa_z`, topo + meia) fora do BOTAR, ou zerar a forma com descarga = 1.
- **Linhas:** 1 → 1.

### 32. No CARREGAR o tombo da caixa custa pouco
Funde: "o tombo exigido no PEGAR fica quase sem preço no CARREGAR" e "a orientação tem gradiente de só 0,017/s por grau".
- **Etapa e transição:** PEGAR → CARREGAR.
- **Regra violada:** §2 Orientação "o tempo todo"; §4.4 "nivelada".
- **Classe:** A/G (canal fraco, não morto).
- **Arquivo:linha:** knobs.py:1255, 1260 (`precise_ori` CARREGAR = 0); recompensas.py:632.
- **Conta:** CARREGAR: d(3·reta)/dθ = 3/π = 0,955/s por rad (0,0167/s por grau); 60° custam 1,0/s contra 14/s de rastreio. PEGAR_COM a 20°: ≈ 5,5/s por rad. Nenhuma terminação lê o tombo. Não há medida de tombo no CARREGAR.
- **Veredito:** CONFIRMADO (preço baixo sai do código); impacto não medido.
- **Severidade:** custa eficiência.
- **Conserto:** `precise_ori` = 1 na coluna CARREGAR. Isto É renda de parado nova (≈ +1/s): o piso parado vai de 16,8 a ~17,8/s; o ganho de andar fica ≈ +14 pela conta do docstring. Não usar a troca do flag de `_alinha` para híbrido: com ele a derivada em θ = 0 cai à metade, e o flag também atinge o PEGAR, desfazendo a decisão de 29/09 (commit 7a23ea9).
- **Linhas:** 0 (1 número).

### 33. A sacudida da caixa não tem canal
Funde: "nenhum termo lê a aceleração ou velocidade da caixa no CARREGAR" e "sacudir sem preço: σ 0,18 m".
- **Etapa e transição:** CARREGAR (e erguer no PEGAR).
- **Regra violada:** §2 "sacudida (aceleração da caixa)"; §3 Caixa íntegra; §4.4 Proibido "Sacudir a caixa".
- **Classe:** E/A.
- **Arquivo:linha:** recompensas.py:620-633; knobs.py:718, 932, 1254-1272.
- **Conta:** custo de desvio d: 3·(1 − exp(−(d/0,18)²)). d = 1,6 cm: 0,024/s; 2 cm: 0,037/s; 5 cm: 0,22/s. A 2 Hz, 1,6 cm é ~2,5 m/s². d/d(aceleração) = 0. O alvo anda com a base: sacudida em fase com o tronco custa 0 em qualquer σ. Medido: 2,5 g na pega e 1,0 g no carregar (model_24999).
- **Veredito:** CONFIRMADO.
- **Severidade:** custa eficiência.
- **Conserto:** multiplicar o `precise_pos` do CARREGAR por exp(−\|\|a_caixa − a_base\|\|²/σ_a²) (ou v), reusando a forma do conserto de impacto. Só estreitar o σ para ~0,06-0,10 dá preço à oscilação relativa, não à aceleração no mundo.
- **Linhas:** +1 a +4.

### 34. O teto do `limite_de_junta` corta a derivada
- **Etapa e transição:** todas.
- **Regra violada:** §2 Robô "chegar ao limite de uma junta é penalizado"; §3 Sem batente.
- **Classe:** A.
- **Arquivo:linha:** recompensas.py:311-312 (`expm1(k·min(excesso, teto))`); knobs.py:1400-1404, 1450-1458.
- **Conta:** hip_yaw: rampa de 40° a 63°; depois custo fixo expm1(3) = 19,1/s com derivada 0 até o batente (158°). Cintura: 53,6/s fixo além de frac 1,0. Hoje inativo: −0,01/s no zero12. A comparação "cair paga mais que ficar" não foi derivada.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência (latente).
- **Conserto:** continuar linear depois do teto: expm1(k·min(e, teto)) + k·e^(k·teto)·relu(e − teto).
- **Linhas:** 1 trocada.

### 35. A laje do BOTAR nasce perto da perna
- **Etapa e transição:** CARREGAR/ESPERA_COM → BOTAR.
- **Regra violada:** §3 Troca de tarefa "sem tranco".
- **Classe:** G.
- **Arquivo:linha:** comando.py:284, 1643-1653, 1711-1713.
- **Conta:** borda perto = 0,50 − 0,30 + U(±0,10) = 0,10 a 0,30 m da pelve (média 0,20, igual ao reset). O teto protege só a caixa. Sem teste de interseção com a perna. Penetração só na cauda do sorteio, não medida.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência.
- **Conserto:** piso na distância da borda (raio da coxa + folga), ou conferir a interseção antes do mocap. Muda o sorteio, não a recompensa.
- **Linhas:** ~+2.

### 36. `NoBatente` descrita como ligada
- **Etapa e transição:** todas.
- **Regra violada:** §7 (uma regra, uma fonte); §3 Sem batente.
- **Classe:** G (documental).
- **Arquivo:linha:** knobs.py:1446-1448; terminacoes.py:158-196; smoke.py:660; env_cfg.py (sem registro).
- **Conta:** terminações ativas: `caixa_largada`, `fell_over`, `time_out`. `NoBatente`: 0 usos.
- **Veredito:** CONFIRMADO.
- **Severidade:** custa eficiência.
- **Conserto:** corrigir o docstring e o comentário do smoke; apagar a classe.
- **Linhas:** ~−39.

### 37. A `FaixaDePose` taxa o agachamento certo do BOTAR
- **Etapa e transição:** BOTAR, laje baixa.
- **Regra violada:** §3 Postura "agachar é permitido para alvo baixo"; §4.5; princípio 4.
- **Classe:** D.
- **Arquivo:linha:** knobs.py:1365-1370 (perna BOTAR 1,3); recompensas.py:253-255.
- **Conta (corrigida):** os números 2,03/1,50 rad não têm fonte no repo e medem o default errado. Com a IK do BOTAR (hip_pitch frac 0,88 → q ≈ −2,21; default ≈ −0,31 assumido): excesso 0,59, custo ≈ 0,48/s, derivada ≈ 0,49/s por rad no sentido de não agachar. Faixa possível: 0,32 a 1,0/s. O joelho não tem medição.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência (baixa).
- **Conserto:** coluna BOTAR de perna = 0, ou tolerância seguindo a referência de IK. `forma_postural` e `limite_de_junta` já guardam a perna.
- **Linhas:** 0 (1 número).

### 38. O `track_lin` tem σ fixo 0,5
Funde: "track_lin sem derivada na estátua para comando ≥ 1,2 m/s" (CARREGAR) e "rastreio linear σ fixo: topo do envelope com derivada quase zero" (ANDAR).
- **Etapa e transição:** ANDAR; CARREGAR andando.
- **Regra violada:** §4.1 faixa até 2 m/s, "aprende o máximo"; §4.4 "várias velocidades"; princípios 1 e 5.
- **Classe:** A.
- **Arquivo:linha:** mjlab velocity_env_cfg.py:279-282 (std √0,25); recompensas.py:432; knobs.py:1265.
- **Conta (corrigida):** peso 2 (ANDAR): cmd 2,0 e v 1,0: R = 0,037/s (não 0,074), derivada 0,29; v 0: ~0. Peso 7 (CARREGAR), estátua: derivada 10,3 (c 0,5), 1,03 (c 1,0), 0,010 (c 1,5), 1e-5 (c 2,0). Com o robô já andando a 1,0 m/s e cmd 2,0 o canal vive (1,03). Na média medida o termo roda a 59% do máximo. O ponto de operação no topo não foi medido.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência.
- **Conserto:** a forma do `giro_sem_gingado`: σ = max(\|cmd_xy\|·f, 0,5). Com f = 0,5 e cmd 2,0, v 1,0: R = 0,74, derivada 1,47 (os valores 1,56/1,6 do auditor exigem σ = 2,0). Fazer junto do achado 45.
- **Linhas:** +6 a +10, ou 0 se o `giro_sem_gingado` for generalizado (−1 função).

### 39. O freio taxa a frenagem no CARREGAR
- **Etapa e transição:** CARREGAR, comando cai a zero.
- **Regra violada:** §4.4 "frear em poucos passos"; §5 "o freio taxa o ruído, não a marcha"; princípio 4.
- **Classe:** D.
- **Arquivo:linha:** recompensas.py:1190-1237.
- **Conta:** sem histerese, o regime vira standing (pernas 1,5 rad/s contra 4,0-6,5). Joelho a 4 rad/s: 0,575/s por junta no degrau 0; 1,29 no degrau 2; 6,55 no degrau 6. Com 6 a 12 juntas: 3,5-6,9/s (d0) a 39-79/s (d6) no transiente. No ANDAR custa 0. Velocidade de junta na frenagem, degrau e duração não medidos.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência.
- **Conserto:** no CARREGAR, aplicar a máscara "estado ≠ ANDAR" só às juntas de braço e punho; ou ~0,5 s de histerese no regime standing.
- **Linhas:** +2 a +4.

### 40. A altura de transporte é sorteada
- **Etapa e transição:** CARREGAR.
- **Regra violada:** §2 "O z do alvo é 0,95 m no mundo".
- **Classe:** G.
- **Arquivo:linha:** knobs.py:210, 219; comando.py:1179-1180.
- **Conta:** U(0,85; 0,95): média 0,90, −0,05 m; piso −0,10 m. `peito_b.z` de 0,052 a 0,152. O incentivo a agachar não foi medido.
- **Veredito:** CONFIRMADO (contradição).
- **Severidade:** custa eficiência.
- **Conserto:** faixa (0,95; 0,95), ou corrigir o spec. Atualizar o comentário (`peito_b.z = 0,152`) e o smoke. O piso não pode descer abaixo de 0,80.
- **Linhas:** 0.

### 41. O alvo fica no referencial da pelve
- **Etapa e transição:** CARREGAR.
- **Regra violada:** §2 "x e y no referencial do peito" (leitura ambígua: `peito_b` é o ponto do peito dado na base).
- **Classe:** G.
- **Arquivo:linha:** comando.py:1726-1748.
- **Conta:** pelve a 20°: x = 0,270 contra 0,250 (+2 cm). O erro vem do pitch e roll da pelve, não do waist_yaw. O custo no `precise_pos` não foi derivado.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência (pequena).
- **Conserto:** `torso_link` com só o yaw (`quat_apply_yaw`). Efeito colateral: o alvo passa a seguir o waist_yaw.
- **Linhas:** ±2.

### 42. Cair segurando custa menos que largar
- **Etapa e transição:** CARREGAR.
- **Regra violada:** §4.4 "Deixar a caixa cair de altura"; §5 "Segurou e caiu"; §1.
- **Classe:** E.
- **Arquivo:linha:** env_cfg.py:233-237, 280-300; terminacoes.py:52-120.
- **Conta (corrigida):** as duas terminações custam −4. Largar termina em ~0,12 s (v_rel 1,2) ou ≤ 0,45 s; não há "largar e reequilibrar". Cair segurando recebe renda até os 70°: diferença ≈ renda/s × (t_tombo − 0,12) ≈ +3 a +11, teto, não medida. O retorno futuro perdido domina as duas.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência (segunda ordem).
- **Conserto:** `upright` como multiplicador do `precise_pos` no CARREGAR.
- **Linhas:** +1 a +2.

### 43. O `air_time` segue em 0
- **Etapa e transição:** ANDAR.
- **Regra violada:** §3 Pés "passo normal"; §4.1 Proibido "passo miúdo, pé arrastando"; §5 "air_time desligado".
- **Classe:** A/D.
- **Arquivo:linha:** knobs.py:438, 447-449; smoke.py:602.
- **Conta:** bloco 1: folga 23,8 mm contra 0,10 m, voo 154 ms, escorrego 0,258 m/s. `swing_height` ≈ 0,29/s de piso, 0,076/s por cm; `foot_slip` ≈ 0,013/s. Tudo é punição, 1 a 6% do teto de ~5/s. As derivadas de `swing_height` e `clearance` não foram verificadas na fonte. Contra: a memória refutou o `air_time` como causa de andar; o zero é decisão declarada da F1.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência (qualidade da marcha).
- **Conserto:** peso 0,5-1,0 no `air_time` do molde (janela 0,05-0,5 s). Desvia da única configuração com marcha medida: risco.
- **Linhas:** 0 (1 número).

### 44. Girar no lugar rende pouco mais que ficar
- **Etapa e transição:** ANDAR, ramo turning.
- **Regra violada:** §3 Sem estátua; §3 "gira também parado"; princípio 6.
- **Classe:** C.
- **Arquivo:linha:** recompensas.py:397-462 (σ = max(\|cmd\|, 0,707)); comando.py:2103-2150.
- **Conta (corrigida):** \|cmd\| ≥ 0,707: parado paga 0,74, girando 2,0, ganho bruto +1,26/s, derivada 0,92 em cmd 1,6. Caso fraco real: \|cmd\| em [0,2; 0,707] (~36% do ramo), parado paga 1,21 a 1,85, ganho bruto 0,15 a 0,79/s. O líquido 0,3-0,6/s desconta custos estimados e ignora termos não lidos.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência.
- **Conserto:** `sigma_fator` 0,6 e sem piso (não girar paga 0,06). Medir a derivada no ponto real antes: o kernel estreito paga menos ao giro lento.
- **Linhas:** 0.

### 45. A faixa de comando não é a mesma nas quatro direções
- **Etapa e transição:** ANDAR.
- **Regra violada:** §4.1 "até 2 m/s, a mesma faixa nas quatro direções".
- **Classe:** G (e A fora da faixa: sem amostra, sem gradiente).
- **Arquivo:linha:** env_cfg.py:496-531; mjlab velocity_env_cfg.py:191-192, 405-408.
- **Conta:** vx ∈ [−1,5; 2,0], vy ∈ [−1,0; 1,0]. Ré 75%, lateral 50% do pedido.
- **Veredito:** CONFIRMADO.
- **Severidade:** custa eficiência.
- **Conserto:** no bloco que reescreve os estágios, lin_vel_x e lin_vel_y = (−2,0; 2,0), ou registrar no spec a faixa pedida. Fazer junto do achado 38. O corte do 3º estágio em 10/09 foi porque o robô perdia a caixa a 3 m/s: a cauda B e R sorteia do mesmo envelope.
- **Linhas:** ~+2.

### 46. Braços abertos parado quase não custam
- **Etapa e transição:** ANDAR parado, ESPERA_SEM.
- **Regra violada:** §4.1 Proibido "braços abertos"; §3 Pose de referência.
- **Classe:** A (canal fraco).
- **Arquivo:linha:** knobs.py:397-427 (std_standing ombro/cotovelo 1,00), 1367 (`braco_pos` ANDAR 0,7, ESP_SEM 0,9).
- **Conta:** dois shoulder_roll a 0,5 rad: custo 0,017/s, derivada 0,034/s por rad; a 0,7: 0,033/s. `faixa_de_pose` = 0 até 0,7 rad. No walking a mesma pose custa 53%. Não há medição do robô de braços abertos.
- **Veredito:** CONFIRMADO (preço quase zero).
- **Severidade:** custa eficiência.
- **Conserto:** std_standing de ombro e cotovelo para ~0,3 (o braço já sai da máscara em pegou ∧ ¬soltou), ou `braco_pos` ANDAR/ESP_SEM para ~0,4.
- **Linhas:** 0.

### 47. O portão da forma é cego ao giro
- **Etapa e transição:** portão da forma (currículo).
- **Regra violada:** §4.1 locomoção forma antes da manipulação; §3 anda nas quatro direções e gira.
- **Classe:** G.
- **Arquivo:linha:** comando.py:2219-2251, 2166-2178; curriculo.py:456-470.
- **Conta:** giro puro: ‖cmd_xy‖ = 0 → segmento descartado. Standing: descartado. cmd (0; 1), v (0,5; 1): e = 1,0, erro lateral invisível. v = 2·cmd: e = 2, sem teto. d(e)/d(erro_yaw) = 0; `error_vel_yaw` ~2,5 sem portão.
- **Veredito:** CONFIRMADO.
- **Severidade:** custa eficiência.
- **Conserto:** projeção sobre o twist inteiro (vx, vy, wz com 1 rad/s ≈ 1 m/s), e = clamp(proj/ped, max 1) − \|erro_perp\|/ped.
- **Linhas:** ~+4.

### 48. Com o comando a zero, os termos de pé desligam
- **Etapa e transição:** ANDAR, frenagem e parada.
- **Regra violada:** §4.1 "freia em poucos passos e fixa os pés"; §3 Pés "parado, sem passo".
- **Classe:** A.
- **Arquivo:linha:** mjlab rewards.py (`active = total_command > threshold`); velocity_env_cfg.py:340-371; recompensas.py:1233.
- **Conta:** cmd 0: slip, clearance, swing, soft_landing = 0; freio = 0 no ANDAR; `air_time` = 0. Resíduo de 0,3 m/s custa 0,61/s no `track_lin`; 0,05 m/s custa 0,02/s. Passo no lugar ≈ 0,005/s via `pose` (estimado).
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência.
- **Conserto:** `foot_slip` com `command_threshold = −1`. Limites: 0,1 m/s de deslize custa 0,001/s (peso teria de subir 10-100×), e pisar no lugar com o pé no ar segue grátis. É só penalidade.
- **Linhas:** 0 (1 parâmetro).

### 49. O comentário do freio diz "média"
- **Etapa e transição:** `velocidade_por_regime`.
- **Regra violada:** §7 passo 2 (verificável no código).
- **Classe:** G (documental).
- **Arquivo:linha:** knobs.py:790, 800-802; recompensas.py:1233 (`excesso.sum`).
- **Conta:** junta a 2× vmax: soma → 0,207/s; média com o peso novo → 0,0071/s (29×). O mesmo bloco já corrige em knobs.py:851-855; o erro só aparece para quem lê só o cabeçalho.
- **Veredito:** CONFIRMADO.
- **Severidade:** custa eficiência (documental).
- **Conserto:** "média" → "soma" na linha 790; atualizar ou apagar os exemplos com peso −2,0.
- **Linhas:** 0 ou menos.

### 50. O ANDAR standing não tem laço de rumo
- **Etapa e transição:** ANDAR parado.
- **Regra violada:** §4.1 "sem deriva de rumo nem de posição".
- **Classe:** A.
- **Arquivo:linha:** mjlab velocity_command.py:136-138; comando.py:313 (`elos_parados` sem ANDAR no treino).
- **Conta:** wz_cmd = 0 fixo. Deriva de 0,077 rad/s: kernel 0,988, custo 0,024/s, derivada 0,61. O rumo integrado tem derivada 0. A deriva de 4,4°/s foi medida na marcha, não parado.
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência.
- **Conserto:** estender `_zera_twist_nos_parados` aos standing do ANDAR (rumo_ref congelado ao parar, wz = 0,5·erro).
- **Linhas:** ~+2.

### 51. A massa da caixa não é observada
- **Etapa e transição:** PEGAR, CARREGAR, BOTAR.
- **Regra violada:** §4.1 (observação sem massa); princípio 1; classe F.
- **Classe:** F.
- **Arquivo:linha:** eventos.py:344-354; recompensas.py:594, 690, 738, 827; observacoes.py:25, 104; knobs.py:312.
- **Conta:** F_ref 6,13 N (1 kg) contra 30,66 N (5 kg). Com F = 10 N: squeeze 0,926 contra 0,315 (2,9×). Mesma renda com 5 kg exige 50 N. No nível 1 a faixa é zero; o efeito só existe a partir do nível 2. A ligação com a caixa esmagada é especulativa (o `tanh` satura).
- **Veredito:** PLAUSIVEL.
- **Severidade:** custa eficiência.
- **Conserto:** `limpo_massa` (ou m·g/(2µ)) como canal só do crítico, no fim do grupo, como o `elo_interno`.
- **Linhas:** ~+3.

### 52. Spec e código divergem no REORIENTAR inerte
- **Etapa e transição:** REORIENTAR.
- **Regra violada:** §4.6 Particularidades ("0,3 s ... metade da manipulação").
- **Classe:** G (documental).
- **Arquivo:linha:** knobs.py:1060-1075; env_cfg.py:82.
- **Conta:** código: 0,5 s e 5%. Custo de tempo ≈ 0,13% (o comentário do knob diz 0,3 s e 0,08%, também obsoleto).
- **Veredito:** CONFIRMADO.
- **Severidade:** custa eficiência (documental).
- **Conserto:** spec §4.6 para 0,5 s e 5%; comentário do knob para 0,5 s ≈ 0,13%.
- **Linhas:** 0 de código.

### 53. k fixo e primitivas fora do spec no REORIENTAR
- **Etapa e transição:** REORIENTAR × k → PEGAR.
- **Regra violada:** §4.6 Cadeia (k sorteado), Entrada (3 primitivas); §6 (só tombo para trás).
- **Classe:** G.
- **Arquivo:linha:** comando.py:161-165; knobs.py:313-338.
- **Conta:** k = 1 sempre. As 6 primitivas (±X, ±Y, ±Z) existem só em comentário; `voltas_max = 0`, então 0 primitivas são sorteadas e o §6 não é violado no código que roda. Com o elo inerte, k maior só soma temporizador.
- **Veredito:** PLAUSIVEL (latente).
- **Severidade:** custa eficiência quando o REORIENTAR for ligado.
- **Conserto:** ao ligar o REORIENTAR: sortear k em CADEIAS com (R,P), (R,R,P), (R,R,R,P), ou repetir o elo em `_aplica_espera`; restringir o eixo Y ao tombo para trás.
- **Linhas:** ~+2.

## 4. Achados refutados (não reabrir)

1. **"REORIENTAR paga ~10,7/s de graça no ponto de nascimento; precise_pos = 3,0/s constante"** (recompensas.py:597, 620, 635). O REORIENTAR é inerte: fecha em 0,5 s faça o robô o que fizer, em 5% dos envs. Uma constante com derivada zero na ação só desloca a linha de base do valor. A única parte com gradiente (`staged` via `alcançar`) paga a mão rumo à caixa, que é a ponte certa para o PEGAR. Volta a valer como C/A só com `reorientar_inerte=False`.
2. **"Empurrar a caixa mais de 0,10 m trava o REORIENTAR e rende mais que avançar ao PEGAR"** (comando.py:910, 1382). No REORIENTAR o ALVO recebe a pose da caixa todo passo (comando.py:1206-1208). `perto` é sempre verdadeiro e d_alvo ≈ 0: o elo não trava e a conta de 7,2/s parte de um estado que não existe. "Deslocar a caixa não tem preço no REORIENTAR" seria outro achado, não este.
3. **"O fecho do REORIENTAR não exige apoiada, nivelada, sem contato e mão solta; tolerância 25° e não 10°"** (comando.py:1382; knobs.py:1012). Com `reorientar_inerte=True` o fecho é só `perto`; o ramo com `alinhado` não roda no treino (só no smoke). O §4.6 já declara o elo inerte. Vale como pendência de desenho ao ligar o REORIENTAR.
4. **"precise_ori ×4 paga o máximo com a caixa parada na mesa; erguer só tira dele"** (recompensas.py:665-670; knobs.py:1260). O termo é plano em z; só o caminho que tomba a caixa perde. A própria conta do auditor dá +2,1/s para erguer tombado, e erguer reto paga +7,4/s. `precise_ori 4 → 0` foi REJEITADO em 29/09 (memória g1-limpo-pega-parada-na-mesa); a trava medida é cinemática (escala da ação do punho).
5. **"Sem comando, a deriva de rumo e de posição no CARREGAR parado é quase grátis"** (comando.py:1291-1308). O CARREGAR fica fora de propósito do `_zera_twist_nos_parados`; 80% dos envs têm o laço de heading do fabricante (`rel_heading_envs` 0,8); o `Giro.portao_da_renda` multiplica a renda congelável pelo kernel de giro. O −0,34 rad/s é do 19300, antes do conserto; no 20500 o CARREGAR mediu −0,1°/s. Sobra só a fatia standing, sem deriva medida.
