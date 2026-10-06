---
tipo: decisao
pai: "[[C3 g1_limpo]]"
estado: ativa
verificado: 2026-10-06
periodo: 25/08–04/09/26
resumo: "Fases F0–F6, cinco estados com one-hot, cadeias de dois elos, currículo de três relógios, sete incentivos; depois o contrato de troca v2 e os gradientes v2.1."
---
#### Por que ele existe (25/08)

Quarto módulo de treino, isolado, que preserva os dois resultados comportamentais medidos e descarta o resto [24121d7]:

- `g1_poc` abriu `pegar` e `reorientar`, reorientava bem e erguia ao alvo; nunca andou.
- `g1_multitask` andou e resistiu a empurrão; nunca girou, e o código acumulou mecanismo.

Três restrições duras [24121d7; spec §2]:

- **R1** zero import de código do projeto (`g1_training`, `g1_poc`, `g1_multitask`); só o mjlab (framework). Exceção: `paridade.py`, descartável [a1c19d9].
- **R2** um objetivo na manipulação, cadeias de no máximo DOIS elos, elos trocam DENTRO do episódio.
- **R3** incentivo, não penalização; contínuo em vez de booleano.

Achado que fundamenta a estrutura [24121d7; spec §9]: o multitask andou porque a fase 1 tinha `abertas = [LOCOMOVER]` (100% dos dados à locomoção por 22.296 iterações); o `g1_poc` abriu o portão medindo SOBREVIVÊNCIA, com o robô imóvel, e entregou 70% das transições à manipulação por volta da it 420. Como a fatia é governada pelo tempo de vida do episódio, com Tl = 24 e Tm = 961 um sorteio de 0,30 entrega 1,06% das transições ao andar. Daí: currículo SEQUENCIAL e fatia resolvida das durações, `f = alvo·Tm / (Tl·(1−alvo) + alvo·Tm)`.

Fundação: molde `mjlab.tasks.velocity`, porque os σ por junta do `variable_posture` só existem dentro de `unitree_g1_rough_env_cfg` e redigitá-los é risco sem teste [24121d7; spec §3]. `air_time` fica DESLIGADO (ausente no módulo que andou, 0,0 no que não andou) [24121d7; spec §5.4].

#### Plano em sete fases (25/08) [a874616; plano]

F0 esqueleto/cena/ação/paridade · F1 locomoção pura + leitura (escada de corte) · F2 one-hot de 5 slots, twist zerado nos elos parados · F3 os sete incentivos, cadeia de 1 elo · F4 máquina de elo (3 cadeias de 2) · F5 balanço de forma, portão da razão de marcha, pisos · F6 currículo de nível e tabela de cadeias.

Decisões fechadas: `variable_posture` do fabricante peso 1,0; `terminacao = −200` (−4,0 por queda a dt 0,02); `dof_pos_limits −1,0`; `escala_acao_mult 1,0` com 0,8 pré-registrado como primeiro número a mover se a F1 falhar; `ang_vel_z = (−0,5; 0,5)`; orçamento do bloco 1 ≥ 5000 iterações; estado do currículo VAI para o checkpoint [a874616; plano "Decisões fechadas"]. ⚠ A spec §8.1 ainda diz "Nada do currículo vai para o checkpoint"; o commit [237881b] implementou o contrário (ver Lacunas).

Ordem planejada dos blocos [plano "A ordem de execução dos blocos"]:

| bloco | o que abre | portão |
|---|---|---|
| 1 | locomoção pura (fatia 0,95) | `Metrics/twist/razao_marcha ≥ 0,50` |
| 2 | os sete incentivos na fatia de 5% | `staged` sobe, `precise_pos` sai do zero |
| 3 | o balanço abre, rampa desce a 0,30 | fatia logada bate com §9.3 |
| 4 | cadeias de 2 elos | `Metrics/alvo_caixa/sucesso > 0` |

F2 a F6 não são blocos separados: são camadas que o `knobs.py` liga, e todas já estavam ativas desde o bloco 1 [e4e5ff1].

#### Estados (elos) e one-hot [spec §4.2; b6cbc7c]

| slot | estado | twist | caixa nas mãos |
|---|---|---|---|
| 0 | `ANDAR` | sim | não |
| 1 | `REORIENTAR` | zero | não |
| 2 | `PEGAR` | zero | não → sim |
| 3 | `CARREGAR` | sim | sim |
| 4 | `BOTAR` | zero | sim |

- One-hot escrito POR PASSO, lido do comando; entra por último nos dois grupos (append, nunca inserção). Ator 99 → 104, crítico 111 → 116 [b6cbc7c]. Sem noise/scale.
- Alvos [a1c19d9]: ANDAR = twist, mobília e caixa a +5 m; REORIENTAR = orientação (alvo é a caixa); PEGAR e CARREGAR = MESMO alvo, x,y relativos ao robô (`peito_b = (0,25; 0; 0,15)`), z ABSOLUTO 0,95 m (impede satisfazer agachando) [vault:g1-limpo-desenho]; BOTAR = ponto lateral num topo novo, travado no fundo da caixa.
- REORIENTAR: face alvo constante `(−1,0,0)` marcada por placa verde inerte (`contype = conaffinity = density = 0`); sorteia a orientação de nascimento; `ang_max_deg` deletado; dificuldade em `voltas_max`, `eixo_vertical`, `desalinho_max_deg` [24364cf; 20fb477].
- Postura (`PosturaPorElo`) devolve 1,0 (neutra) nos elos de manipulação: `exp(−média(erro²/σ²))` sobre 29 juntas colapsa para qualquer σ (standing 0,000 já a 10% da faixa; nem run×5 sobrevive a 40%: 0,338) [b4a2add].
- `fatia_loco = 0,95`, nunca 1,00: com 1,00 os slots de manipulação são constantes e o normalizador (`rsl_rl/modules/normalization.py:48`, sem clamp) faria o 1,0 entrar como 100,0 [a90f739].

#### Máquina de elo e cadeias [spec §4.1; 60c0f5b; 6e04bf9:comando.py]

Cadeias originais (transcritas de `g1_poc/comando.py:60,63`):

```
0 (PEGAR,)
1 (REORIENTAR, PEGAR)
2 (PEGAR, CARREGAR)
3 (PEGAR, BOTAR)
```

`prob_por_nivel` original (nível 0: 0,80 / 0,10 / 0,05 / 0,05; nível 3: 0,40 / 0,25 / 0,20 / 0,15; nível 4: 0,20 / 0,25 / 0,30 / 0,25; nível 5: 0,15 / 0,25 / 0,35 / 0,25) [knobs.py em 6e04bf9].

Fecho por elo [spec §4.1, depois c6f7e65]: REORIENTAR perto & alinhado; PEGAR perto & alinhado & de pé (sustain 0,5 s, medido [8921bf8]); CARREGAR perto & ANDOU ≥ 0,50 m (antes era só `perto`, ver c6f7e65); BOTAR perto & alinhado & apoiada (fração do peso).

Invariante central [60c0f5b]: quem decide loco × manipulação é o CURRÍCULO; a cadeia só escolhe qual transição praticar e tem de começar no elo já sorteado. A 1ª versão sobrescrevia `_elo` com o 1º elo da cadeia e apagava a fatia de 95% (todos viravam PEGAR). Depois do conserto: fatia 0,9414. Mais seis defeitos consertados no mesmo commit (sensor "apoiada" inexistente dentro de `try` que deixava `apoiada = True`; limiar de 2 N fixo → fração do peso; `prob_por_nivel` não chegava ao cfg — máquina INERTE; `CADEIAS[-1]` para env sem cadeia; `dt` escrito à mão; laço Python por env).

#### Currículo: três relógios [spec §8; a9d900d; ce8984b; 814a322]

| parte | o que move | gatilho |
|---|---|---|
| `command_vel` | faixa de velocidade | passo global (it 0 / 5000 / 10000) |
| `forma` | fatia loco × cadeia | portão `razao_marcha_ema ≥ 0,50` (depois `eficiencia_min`, ver bloco 1/2) |
| `nivel` | dificuldade e distribuição de cadeias | passeio ±1 por sucesso de CADEIA, clamp(0, 6) |

- Rampa: 0,95 → 0,30, passo 0,02 a cada 12 iterações (33 degraus, ≥ 396 its); carência 200 its contada do início do balanço; histerese 0,80 × limiar [spec §9.2].
- Estado inicial assimétrico: durações neutras, `razao_marcha` pessimista em 0,0 [a9d900d].
- Ordem do dict: `command_vel → forma → nivel → elo` (forma e nível leem o episódio que acabou) [a9d900d].
- Só episódio de cadeia move o nível; ponto fixo p = 0,5 por construção (difunde, não converge) [ce8984b; 693ae0f].
- Tabela de níveis [spec §8.1]: `topo_min` 0,55 / 0,45 / 0,30 / 0,15 / 0,04 / 0,04 / 0,04; `carga_max` 1 a 5 kg; `voltas_max` 0,0,1,1,1,1,1; eixo vertical do nível 4.
- Estado do currículo no checkpoint via `runner_cls` (EMAs, fatia, carência, portão, nível e elo por env) [237881b].
- Defeito grave achado por code review antes de qualquer run: a carência e a rampa contavam PASSOS, não iterações (erro de 24×). Medido com 128 envs: o contador subiu em 48,8% dos passos em 400 passos. `carencia_iters = 200` era atingido em ~17 its e a rampa colapsava em ~34. Conserto: derivar de `env.common_step_counter`; um degrau por JANELA [814a322; 6e04bf9].

#### Os sete incentivos (F3) [2179e3c; 218e0cb; spec §6]

`staged = alcancar × (1 + trazer)` +3,0 · `precise_pos` σ fixo 0,05 +2,0 · `precise_ori` +1,0 · `squeeze = tanh(min(F_E, F_D)/F_ref)` +1,0 · `unload = 1 − F_apoio/mg` +2,0 · `postura_ereta = rampa × preensão × descarga` +2,0 · `sustentacao` +0,5. Soma 11,5/s. Gate por VALIDA (sem ele `exp(0) = 1` pagaria o máximo em ANDAR; medido: os sete valem exatamente 0 em ANDAR).

σ = distância inicial por env [218e0cb]: a palma nasce a 0,339 m da caixa (0,211 a 0,481); σ fixo 0,10 dá kernel 1e−05 e derivada zero. Com σ = d₀ o kernel nasce em 0,368 e a derivada fica entre 3,49 e 1,53/m; medido em 48 envs, derivada mínima > 1,0/m. Distância até a SUPERFÍCIE (ao centro o kernel saturava em 0,674; com a meia-aresta chega a 0,887). Pré-registrado: se o alcance não aparecer, `sigma_fator` → 1,5, nunca o peso.

Preço declarado [52d20b9]: o piso do elo parado subiu de 5,81/s para 8,03/s (staged paga 1,95/s à estátua); manipulação paga mais que locomoção (8,03 contra 4,44).

#### Estado ao fim da implementação (26–27/08)

Smoke 291 → 301 → 306 ok / 0 falhas; paridade 85/0; inspetor 0 falhas; "NENHUMA RUN FOI RODADA" [e4e5ff1; 6e04bf9; 604cc9b; fa8cee2]. O `unload` medido salta de 0 a 1 em 2 mm de erguida (contato rígido): quase booleano como rampa de altura [604cc9b]. O venv local não roda PPO (`select_gpus` sem placa; mjlab 1.5.1 incompatível com rsl-rl-lib 5.4.0); o treino é remoto com mjlab 1.5.3 [cc26eda].

#### Contrato de troca de tarefa (v2, 02–03/09)

Nasce do defeito "a observação publica a caixa a 5 m no ANDAR (`|caixa_b| = 4,32`)": a política pode ler "ando" da distância; ao entregar a tarefa com a caixa perto o robô "sambou" [1f803e8; handoff]. Spec v5 → v17 em `docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md`, branch `exp/g1-limpo-v2` (a `exp/g1-limpo` fica intocada, é a que treinou a `bloco7`) [1f803e8; handoff]. Itens: gate dos canais de caixa quando o one-hot PUBLICADO é ANDAR; VALIDA sai da obs; espera publica ANDAR (interno segue PEGAR); quatro comportamentos (andar, pegar, carregar, botar) e cadeia 3 vira `(PEGAR, CARREGAR[v=0], BOTAR)` [a48a169]; espera final após o BOTAR [e9c1043]; toda espera 0,5–1,5 s e `rel_turning_envs = 0,10` com `|wz| ≥ 0,2` [ba12e3d; 996fac2]; DR de tamanho da caixa 0,07–0,13 m, K = 8, por `geom_size` no startup, caixa segue box primitivo [6cde03b; 7ec5851]; `ANG` vira `giro_b` (3 canais), ator 114 [7fd3592]; renda monótona do BOTAR com `load` 2,0 e `largou` 1,0 [8812b4c; d63961f]. Implementado em 03/09 [d3ea735 … f9f299f]: smoke 517 ok, crítico 131 canais (12 de pé do fabricante a mais) [2ef82ec]; renda do BOTAR MEDIDA pairar 10,70 < apoiada 13,00 < espera final 17,86/s [2ef82ec]; `REORIENTAR` precisa de interruptor `reorientar_inerte` (1 em 6 envs nascia fora dos 25°) [5b24601]; força de apoio projetada no eixo vertical (`|f_z|`) e métrica `impacto_da_caixa` [f58b6fb]. Treino do zero; checkpoint antigo inválido [1f803e8; handoff].

#### Gradientes v2.1 (04/09) [53f81e9; 6051a60; 8f7eca1]

Regra do dono: em todo instante existe incentivo da posição atual até o alvo, e do fecho de um elo ao seguinte a renda não cai. P1 `precise_pos` σ 0,05 → 0,18, peso 2,0 → 3,0 (pagava exp(−4) = 0,018 no limiar; agora 0,73; derivada no aceite 2,5 → 16,1/m); P2 `sustentacao` lê `_sust` do comando; P3 `renda_congelada` (congela no fecho a soma dos termos dependentes do elo; `load` sai); P4 rastreio gateado por `limpo_twist_zerado`; P5 twist fixo no CARREGAR andando (|v| ≥ 0,3); P6 `pelve_margem` 0,03; P7 `caiu` sem a arma do `pegou`; P8 REORIENTAR inerte a 5% do sorteio (cadeias 2+3 de 2,8% para 5,3%). Smoke 524 → 561 ok.

> Fonte: levantamento de 06/10/2026 sobre commits, specs, planos, relatórios e handoffs (relatórios c3_A–D). `[hash]` = commit; `mem:` = só memória; `SESS` = só transcrição de sessão.
