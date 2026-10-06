---
tipo: achado
pai: "[[Linha bloco]]"
estado: ativa
verificado: 2026-10-06
periodo: 26/08–16/09/26
resumo: "Achados datados que não pertencem a um run: Entity.data obsoleto, diluição do Episode_Reward, unload quase booleano, alcance refutado, escorar compra o fecho."
---
### 26/08–04/09
- **26/08 — `Entity.data.*` obsoleto no reset e no resample do comando** (caixa sem origem do env, topo do BOTAR 0,300 nos 8 envs, alvo do ANDAR 0,000); conserto pelo `_pendente` transcrito do g1_poc [a1c19d9]. Ler `.data` depois de `write_mocap_pose_to_sim` sem passo devolve buffer velho [81c87f2].
- **26/08 — `reward_manager.py:122` pula termo com peso 0**: o `Metrics/air_time_mean` do fabricante não existe; métricas de marcha viram `MetricsTermCfg` próprios [4cca18c].
- **26/08 — `feet_swing_height` sem `reset()`**: o pico vaza entre episódios e infla com queda; `AlturaDeBalanco` conserta [13fb96b]. O `pico_de_altura` próprio é diluído ~3,6× (zero de env sem pouso na média) [mem: g1-limpo-pico-altura-diluido, 27/08].
- **26/08 — Diluição do `Episode_Reward`**: soma do episódio / `max_episode_length_s`; com episódios de 2,05 s num teto de 20 s o fator é 0,1027 (`action_rate_l2` −0,34 no painel = −3,31/s real); chave certa do rsl-rl 5.4.0 é `Policy/mean_std`, não `Policy/mean_noise_std` [30ae991].
- **26/08 — Curva do `unload`**: 9,80 N → 0,00 N e termo 0,0005 → 1,0000, transição em 2 mm; teleporte não modela partilha de carga [604cc9b].
- **26/08 — BOTAR: o limite físico vence o knob**: laje em 0,300 contra `fundo − folga` 0,067; o piso cede; nos níveis altos a laje sai em 0,196 e 0,208 [ab3f9d1; a0923c6]. Um `except Exception` no verificador apagou o check e reportou sucesso por omissão nos 7 níveis [a0923c6].
- **26/08 — CARREGAR fechava parado** (`perto` era subconjunto da condição do PEGAR sobre o mesmo alvo); agora `perto & andou ≥ 0,50 m`; o cronômetro do `sustentacao` não zerava no avanço [c6f7e65]. Lição: testar a condição de fecho contra a do elo anterior [vault:g1-limpo-desenho].
- **27/08 — `com_over_feet` não entra**: mergulho de 30 cm custa 0,125/s contra 11,5/s de tarefa (1,1%) [f4ccb6c]. Checagem "todo sensor tem consumidor" achou 4 órfãos (`dorso_E/D` como dívida real) [f4ccb6c].
- **27/08 — Viewer com cadeia**: laje de 0,538 m a 4,98 m em t = 3,5 s com o robô parado [71c383a]; um splice por índice duplicou a função `viewer` e Python manteve a antiga em silêncio [71c383a].
- **31/08 — Tolerância do apoio da caixa**: 9,80, 11,79 e 13,36 N contra m·g 9,81 (penetração inicial varia por run) [369af74; 0228523].
- **02/09 — Primeira observação do episódio com twist de até 2 m/s no modo entrega** (`reset()` chama `compute(dt=0)` sem eventos de intervalo): 1,9682 → 0,0000 depois do conserto [78379ff].
- **03/09 — Pose escrita em corpo livre PERSISTE** (caixa solta 20 cm acima fica 25 passos lá; F_apoio zero); estado apoiada = pinar 2 mm abaixo → 0,98 mg [f9f299f].
- **03/09 — `limpo_soltou` publicado um passo atrasado** (terminações e recompensas rodam antes do comando): o robô que solta e recua morria por `escapou` no passo do sucesso [a3b0ea9].
- **03/09 — One-hot publicado troca no meio do episódio e a política vê**: escrever `_command[:, ELO] = ANDAR` mostra `[1,0,0,0,0]` no passo seguinte; rastreio passa a pagar 1,83/s [31daa9e].
- **04/09 — `descarga` alta prova caixa fora da mesa** (corrige afirmação errada do mesmo dia) e três canais FALSOS com `reorientar_inerte = True` (`avancos`, `fatia_cadeia`, piso de `precise_pos` 0,8%) [vault:g1-leitura-log-e-kaggle].
- **Smoke ao longo da janela:** 98 (F0) → 148 (F1) → 191 (F2) → 217 (F3) → 243 (F4) → 277 (F5) → 291 (F6) → 301 → 306 → 310 → 325 → 327 → 348 → 352 → 357 → 368 → 382 → 398 → 402 → 423 → 425 → 517 → 520 → 524 → 561 ok / 0 falhas (commits na ordem). Pendência em 04/09: falha intermitente nas seções 19 e 28 do smoke (sem seed) [6051a60; mem: g1-limpo-auditoria-gradientes].

### 08/09–16/09
- **08/09 — semente fixa no smoke** (`c3c9d08`): a seção 26 falhava 1 de 5 execuções com o código
  intocado; "um portão que falha por acaso não gateia".
- **08/09 — revisão independente achou 14 desvios** no diff da v3.1 (`60b8fdf`), entre eles: o fecho do
  PEGAR tem de marcar `pegou=True` porque o sensor de palma pode nunca disparar (o tronco escora a maior
  parte do contato); `fatia_cadeia` 0,10 → 0,008 (inalcançável com `p_C` no piso).
- **08/09 — `auto_colisao` 121 N** no nível 4 é o robô esbarrando nele mesmo ao alcançar baixo, não a
  laje; `apoio_caixa` 0 N (`34a356f`). A sonda que forçou nível 4/5/6 deu `limpo_topo` idêntico (p50
  0,43): o forçar de nível não agiu.
- **09/09 — notebook do Kaggle entra no repo** (`c1801ea`): a corrida rodou uma cópia velha cujos asserts
  nunca dispararam.
- **10/09 — `pilota` e `exporta_cena`** (`1fa55ed`, `749f157`): checkpoint dirigido no MuJoCo clássico;
  `--paridade` compara os 114 canais de observação nos cinco elos. O 0,7 rad/s que o `pilota` anunciava
  vem do ramo `play` do molde (`80244b2`).
- **10/09 — cauda CARREGAR sem derivada** (memória, `model_10200`): renda viva 13,20/s = 3,96
  estruturais + 3,96 saturados + 5,28 com gradiente; a cauda não tem destino.
- **11/09 — fórmula obsoleta no comentário do G2** (`a6259fe`): o comentário dizia `1 − exp`, o código
  roda clamp; custos reais 2,0/s no limite e 8,0/s no dobro (4,1×); o "1,08/s da pressa" é 2,17/s.
- **11/09 — punho: aritmética corrigida** (`fe50a1e`): 6 × (1,6/0,30)² = 170,7; /21 = 8,13;
  `exp(−8,13) = 0,0003`.
- **11/09 — `registra_juntas`** (`3d81758` … `30828fb`): loga as 29 juntas por fase no MuJoCo clássico,
  com botões de atrito, `impratio` e `--topo`.
- **11/09 — regras de minimização de código e de desenho de recompensa** entram no `CLAUDE.md` (`ea0f592`).
- **11/09 — Colab × Kaggle** (`dbcc299`, spec `notebook-colab`): Colab roda 8192 envs (16 GB), Kaggle não;
  `SEG_POR_ITER` 9,0 / 7,0; cópia ao Drive a cada 10 min (a Kaggle caiu por internet e levou 1500 its).
  A bloco8 mediu 8,54 s/iter no Colab com 8192.
- **14/09 — alcance REFUTADO** (memória `g1-limpo-botar-e-bang-bang-nao-pressa`): IK com pés planos e
  caixa de 5 kg alcança com margem 0,35 (laje 0,55) e 0,25 (laje 0,05); curvar e agachar alcançam o mesmo
  ponto. A pose extrema é escolha. O `fell_over` de 70° mede a pelve, não o tronco.
- **14/09 — escorar compra o fecho** (handoff; memória): `apoiada` dispara a 4,9 N (1,5% do peso de ~343 N);
  CAUDA paga ~58/s contra ~35/s do BOTAR; fechar 1 s antes vale ~23; escorar custava ~1/s de `upright`.
- **14/09 — quatro achados de arquitetura** (handoff, plano): o episódio não acaba no sucesso e a renda
  congelada vira anuidade; `renda_congelada` congela um instante; `descarga` com `clamp(0,1)` torna o
  escoro invisível; `squeeze` cego acima de ~3×F_ref (`tanh(F/6,13)` = 1,0000 aos 68,9 N).
- **14/09 — separação estável × experimental** (`96be97a`): o notebook clona por REF; o oficial aponta
  para a tag; o resume pega o `model_*.pt` de maior número por `rglob`, e um treino do zero na mesma
  pasta sobrescreveria o checkpoint bom. Conversão do painel: `taxa/s = painel × 20 / segundos do episódio`.
- **15/09 — oito sondas de forma postural** commitadas sem mudança (`d117342`): carregam o `g1.xml` cru
  (sem caixa, laje e pads), assumem pelve vertical; `ikc.py` põe o alvo da palma em mundo.
- **16/09 — notebook do zero** (`618373f`): `limite_de_junta` forçado a (20,0; 0,15) nas 14 famílias, a
  rampa acaba no batente; 5478 its em 10,5 h a 6,9 s/iter; slug `g1-limpo-zero` para não retomar a
  linhagem errada.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
