# g1_v3: nível do ANDAR, entrada da fatia B e relógio de marcha pela receita da Unitree (dono, 16/11)

Esta spec é o contrato do lote de 16/11. Ela registra as edições exatas. O motivo e os números de cada mudança estão no
desenho `/home/joaobornelli/workbench/projects/g1-treino-v3/desenho-curriculo-andar-unitree.md` (fora do repo) e no
inventário `/home/joaobornelli/workbench/projects/g1-treino-v3/inventario-limpo-bloco1-vs-v3.md`. Leia a seção da sua
tarefa no desenho antes de editar.

## 1. Decisões do dono (16/11)

1. Relógio de marcha na forma da Unitree: período fixo de 0,8 s, sem recomeço, peso 0,5, fase fora do ator.
2. Nível do ANDAR pela receita da Unitree, com as decisões P1 a P8 da §6 do desenho:
   - P1: a nota do segmento usa o núcleo cru da Unitree;
   - P2: a faixa começa em 0,5;
   - P3: 50 iterações seguidas;
   - P4: 10 % de comando parado;
   - P5: a fatia B abre no nível 5 do ANDAR;
   - P6: o `passo` perde o portão de velocidade;
   - P7: o ANDAR tem piso de 0,30 dos envs;
   - P8: a regra das iterações seguidas vale para as 4 tarefas.
3. O subnível de 0,75 m do PEGAR e do BOTAR sai (anotação de 16/11 no `vault/_sistema/inbox.md`).
4. O `altura_pe` fica como está.
5. A próxima corrida é do zero. Checkpoint antigo não precisa carregar.

## 2. Regras do lote (valem para toda tarefa)

- Trabalhe só no worktree `/home/joaobornelli/workbench/projects/g1-treino-v3/g1_training`, branch `g1_v3`.
- Commit local no fim da tarefa, com os arquivos da tarefa por caminho explícito. PROIBIDO: push, `gh`, qualquer
  operação no remoto, `git stash`, `Co-Authored-By` e qualquer trailer de coautoria. Depois do commit, rode
  `git log -1 --format=%B | grep -ci co-authored` e confirme 0.
- Durante as edições, não rode teste nem import. Faça todas as edições da tarefa, confira com `git diff` contra esta
  spec, e rode os testes uma vez, no fim. Não rode treino.
- Comando dos testes: `.venv/bin/python -m unittest discover -s g1_v3/testes -t .`
  A falha `test_mid360.TestMid360.test_primeira_direcao_bate_com_o_csv` é antiga (CSV fora do repo) e não conta.
  Qualquer outra falha conta.
- Um teste que fixa um valor ou um comportamento antigo desta spec é atualizado para o novo. Não enfraqueça um assert
  para ele passar. Teste de função ou de célula apagada sai junto com ela.
- Todo teste novo falha contra o código de antes da tarefa. Diga no relato como você conferiu (por leitura do caso).
- Não mude nada fora do que a tarefa lista. Não refatore além do que a tarefa pede. Todo número novo mora em
  `g1_v3/knobs.py`, com um comentário na linha: "16/11, <item>: <motivo curto>".
- Antes de mexer num termo de recompensa, knob ou terminação, faça `grep` do nome em `vault/_sistema/por-termo.md` e leia
  a linha. Se ela citar uma armadilha que esta spec não trata, pare e relate.
- Regra do repo (`CLAUDE.md`): modifique antes de adicionar. Nenhuma função nova neste lote.
- Ordem do passo no mjlab: física → terminações → recompensas → reset → comandos (`compute` da cadeia e do `twist`) →
  eventos de intervalo → observações.
- Relate no fim: arquivos mudados; contagem de funções do `g1_v3` sem testes antes → depois
  (`grep -c "def " g1_v3/*.py`, somando os arquivos fora de `testes/`); comando de teste; as últimas 15 linhas da saída;
  hash do commit; o que você não conseguiu cumprir.

Linha de base: commit `0ca0827` (código igual ao `967f690`), 476 funções no `g1_v3` sem testes. Meta: ≤ 476.

## 3. Tarefa T1: relógio de marcha na forma da Unitree (item R)

Arquivos: `g1_v3/knobs.py`, `g1_v3/comando.py`, `g1_v3/recompensas.py`, `g1_v3/eventos.py`, `g1_v3/testes/`.

Referência: `unitree_rl_lab`, termo `feet_gait` (nota de 13/11 no `vault/_sistema/inbox.md`): período 0,8 s, fase pelo
tempo do episódio, pés defasados em 0 e 0,5, apoio com fase < 0,55, recompensa × (‖comando‖ > 0,1), peso 0,5, sem fase
na observação do ator.

1. `knobs.Comando`:
   - saem `cadencia_base`, `cadencia_por_v` e `giro_para_v`;
   - entram `periodo_marcha: float = 0.8` (período do relógio, s) e `limiar_marcha: float = 0.1` (‖comando‖ acima do
     qual o relógio paga);
   - `fase_apoio` fica 0,55.
2. `knobs.Recompensa.passo`: 1.3 → 0.5 (peso do `feet_gait`; até 1,0/s nos dois pés).
3. `comando.TwistV3._update_metrics`, bloco 5 (relógio):
   - `on = cmd.norm(dim=1) > k.limiar_marcha` (norma do comando inteiro: vx, vy e wz);
   - `env.v3_fase_marcha[:] = torch.remainder(self._agora / k.periodo_marcha, 1.0)`;
   - `env.v3_marcha_on[:] = on`;
   - saem o período pela velocidade e o recomeço no limiar;
   - `soma_acerto`, `passos_on` e a métrica `acerto_relogio` ficam, com o `on` novo.
   Atualize o comentário do bloco.
4. `comando.TwistV3.reset`: saem as duas linhas que zeram `v3_fase_marcha` e `v3_marcha_on`. A fase vem do `_agora`, que
   o reset já zera, e o `compute` dos comandos regrava os dois buffers antes das observações.
5. `recompensas.n_passo(contato, fase, fase_apoio)`: sai o parâmetro `on`.
   `apoio = fases < fase_apoio`; devolve `(contato == apoio).float()`. Atualize o docstring.
6. `recompensas.passo`:
   - `on = cmd.norm(dim=-1) > C.limiar_marcha`, com o comando do passo corrente;
   - `acerto = n_passo(contato, env.v3_fase_marcha, C.fase_apoio)`;
   - `env.v3_acerto_marcha[:] = acerto.mean(-1)`;
   - devolve `_fim(env, tarefas, acerto.sum(-1) * on.float(), paga=True)`.
   Saem o portão de velocidade (`g_lin`, `g_ang`, `portao`) e o apoio duplo pago com comando zero. Reescreva o docstring:
   a forma é a do `feet_gait`; marchar no lugar paga até 1,0/s, e o nível do ANDAR (T2) não sobe sem rastreio.
7. `eventos.empurrao`: sai a linha `env.v3_fase_marcha[env_ids] = 0.0`.
8. Testes (`test_marcha_relogio.py`, `test_eventos_empurrao.py`, `test_recompensas.py` e outros que fixem o relógio velho):
   - a fase vale `(t / 0,8) mod 1` para comando ligado e desligado, sem recomeço no limiar;
   - o empurrão não muda a fase;
   - o `passo` vale 0 com ‖comando‖ ≤ 0,1, e paga a soma dos acertos com ‖comando‖ > 0,1;
   - o robô parado com comando de 0,5 m/s recebe o `passo` pelos acertos, sem portão;
   - `n_passo` com os dois pés no chão acerta 0,55 de um ciclo, em média, por pé.

## 4. Tarefa T2: nível do ANDAR, entrada da fatia B e subnível (itens A, B e S)

Arquivos: `g1_v3/knobs.py`, `g1_v3/balanceador.py`, `g1_v3/comando.py`, `g1_v3/curriculo.py` (só docstring, se citar
as células velhas), `specs/g1-v3-comportamento.md` (§2 e §6), `g1_v3/testes/`.

### 4.1 Knobs

1. `knobs.Comando`:
   - sai `baixo`;
   - entram `andar_inicio: float = 0.5` (teto inicial de frente, trás e lado no ANDAR, m/s),
     `andar_giro_inicio: float = 0.5` (teto inicial do giro no ANDAR, rad/s), `andar_passo: float = 0.1` (quanto cada
     nível soma aos tetos), `andar_parado: float = 0.10` (fração dos sorteios do ANDAR com comando zero) e
     `sigma_nota: float = 0.5` (σ do núcleo da nota do segmento, o da Unitree);
   - `frente_tetos`, `giro_tetos`, `tras_lado_teto_max`, `piso_velocidade`, `terreno_v_max` e `razao_portao` ficam
     (o CARREGAR segue com eles; os tetos finais do ANDAR são `frente_tetos[-1]`, `tras_lado_teto_max` e
     `giro_tetos[-1]`).
2. `knobs.Balanceador`:
   - sai `iters_min_nivel`;
   - entram `iters_nivel: int = 50` (iterações seguidas no portão para subir um nível, nas 4 tarefas),
     `portao_andar: float = 0.80` (portão da célula do ANDAR: nota média ≥ 0,8, o critério da Unitree),
     `nivel_abre_b: int = 5` (nível do ANDAR que abre a fatia B), `piso_andar: float = 0.30` (fração mínima da fatia
     ANDAR depois que a B abre) e `histerese_rampa: float = 0.70` (abaixo desta nota do ANDAR, o teto da rampa desce);
   - `portao` 0,75 fica para as células das outras tarefas.
3. `knobs.Niveis` (item S): `subnivel_extra` (0.20, 0.10, 0.0) → (0.10, 0.0); `subnivel_longe` (0.0, 0.10, 0.20) →
   (0.10, 0.20); `subnivel_jitter` (0.0, 0.0, 1.0) → (0.0, 1.0). Comentário: "16/11: sai a laje de 0,75 m (no
   model_2200 a caixa se apoia nela no P4)".

### 4.2 Células do ANDAR (`balanceador.py`)

1. `LAYOUT`: as células `reto`, `parado`, `baixo`, `giro_andando` e `giro_lugar` do ANDAR saem. Entra
   `("ANDAR", "andar")` no começo. Fica `("ANDAR", "terreno")`. As do CARREGAR, PEGAR e BOTAR não mudam.
2. `ABERTAS_NO_COMECO = (("ANDAR", "andar"),)`. `VELOCIDADE["ANDAR"] = ("andar",)`.
3. `Curriculo.max_nivel["ANDAR"] = round((k_com.frente_tetos[-1] − k_com.andar_inicio) / k_com.andar_passo)` (= 10).
   O do CARREGAR fica `len(k_com.frente_tetos) − 1`.
4. Portão por tarefa: o portão de uma célula do ANDAR é `portao_andar`; o das outras, `portao`. Vale em `_passa`,
   `_passa_media` e `_deficit` (e, por eles, nas frações). Sem função nova: um dicionário no `__init__`
   (`{"ANDAR": k.portao_andar}`, com `.get(tarefa, k.portao)`) basta. Atualize os docstrings do módulo e de `Celula`.

### 4.3 Regra de subida (`_sobe`, item A4 e P8)

1. Sai a carência por tempo (`self.it − self._ult[t] < iters_min_nivel`) e sai o `_ult` inteiro (também no `_abre` e no
   estado).
2. Entra o contador `self._seguidas = dict.fromkeys(TAREFAS, 0)`. Em cada fecho, para cada tarefa:
   - nível abaixo do máximo: `ok` = todas as células de `VELOCIDADE[t]` abertas e `_passa_media` delas;
   - nível no máximo, tarefa em `CADEIA` e linha de terreno abaixo do máximo: `ok` = célula `terreno` aberta e
     `_passa` dela;
   - senão: `ok` = falso.
   `_seguidas[t]` soma 1 com `ok` e volta a 0 sem `ok`. Com `_seguidas[t] ≥ iters_nivel`, a tarefa sobe como hoje
   (subnível, nível ou linha de terreno), zera o `n_medidas` das células como hoje e zera `_seguidas[t]`.
3. A carência da tarefa que abre tarde (A8 de 13/11) fica coberta: o contador da tarefa nova começa em 0.
4. Atualize o comentário de ordem do `fecha_iteracao` e o docstring de `_sobe`.

### 4.4 Entrada e fatia da B (`_abre`, `fracoes_fatias`, `fecha_iteracao`; itens B1 a B3)

1. `_abre`: a fatia B (CARREGAR `reto` e PEGAR `pegar`) abre quando `self._nivel["ANDAR"] >= self.k.nivel_abre_b`.
   Sai o portão pela média do `reto` e do `parado`. O resto do `_abre` (cadeias, terreno, fatia C) fica.
   Efeito: a B abre no fecho seguinte ao da subida do ANDAR ao nível 5.
2. Rampa condicionada: entra `self._rampa = [0, 0, 0]` (degraus da rampa de cada fatia). Em `fecha_iteracao`, depois
   do laço que grava `_it_abertura` e antes de `_sobe`: para cada fatia f ∈ {1, 2} aberta, quando
   `(self.it − self._it_abertura[f])` é múltiplo positivo de `rampa_fatia_its`:
   - nota do ANDAR = `s` da célula `("ANDAR", "andar")`;
   - nota ≥ `portao_andar`: `_rampa[f] += 1`;
   - nota < `histerese_rampa`: `_rampa[f] = max(0, _rampa[f] − 1)`;
   - entre os dois: nada muda.
   Em `fracoes_fatias`, o teto vira `min(1.0, rampa_fatia_inicio + rampa_fatia_passo * self._rampa[f])`. O excedente
   continua voltando ao ANDAR.
3. Piso do ANDAR: em `fracoes_fatias`, depois do bloco da rampa, se a B ou a C está aberta e `out[0] < piso_andar`, a
   falta (`piso_andar − out[0]`) sai das fatias abertas 1 e 2 na proporção das frações delas, e `out[0]` vira
   `piso_andar`. A soma continua 1.
4. `_trava_dr`: o `"vies"` liga quando a célula `("ANDAR", "andar")` passa no portão.
5. `estado` e `carrega`: a chave `"ult"` vira `"seguidas"`; entra `"rampa": list(self._rampa)`. O `carrega` aceita
   estado sem as chaves novas (`.get` com o padrão 0 e `[0, 0, 0]`).

### 4.5 Sorteio e nota do ANDAR (`comando.py`, itens A1 a A3)

1. `sorteia_velocidade(tipo, nivel, tarefa, n, gerador, k)`: com `tarefa == "ANDAR"` (tipos `andar` e `terreno`):
   - `frente = min(k.andar_inicio + k.andar_passo * nivel, k.frente_tetos[-1])`;
   - `lado = min(k.andar_inicio + k.andar_passo * nivel, k.tras_lado_teto_max)`; `tras = lado`;
   - `giro = min(k.andar_giro_inicio + k.andar_passo * nivel, k.giro_tetos[-1])`;
   - no `terreno`: `frente` e `tras` com teto `k.terreno_v_max`;
   - vx ~ U(−tras; frente), vy ~ U(−lado; lado), wz ~ U(−giro; giro), independentes, pelo gerador;
   - com U < `k.andar_parado`, o comando inteiro vale zero.
   O CARREGAR (`reto`, `giro_lugar`, `arco`, `terreno`) fica como hoje. Saem os ramos `parado` e `baixo` e o tipo
   `giro_andando`. Atualize o docstring (o parâmetro `tarefa` passa a mudar a faixa).
2. `TwistV3`: entram os buffers `seg_nota` (E,) e `seg_dur` (E,), zerados onde o `seg_proj` é zerado (`_abre` e fim do
   `_fecha`).
3. `TwistV3._update_metrics`, no bloco 4, depois de gravar `v_media_b`:
   - `apos = (self._agora >= env.v3_ancora[:, 3]).float()`;
   - `e_lin = (cmd[:, :2] − v_media_b[:, :2])²` somado nos dois eixos; `e_ang = (cmd[:, 2] − v_media_b[:, 2])²`;
   - `seg_nota += 0.5 * (exp(−e_lin / σ²) + exp(−e_ang / σ²)) * apos * dt`, com σ = `k.sigma_nota`;
   - `seg_dur += apos * dt`.
   O canal sem comando entra no mesmo núcleo (c = 0): ficar parado vale 1, e a deriva cobra pelo erro.
4. `TwistV3._fecha`:
   - `nota = seg_nota / seg_dur.clamp(min=1e-6)` dos `ids`;
   - `x = torch.where(seg_carrega, suc.float(), nota) * (~falha).float()` (o CARREGAR segue com o sucesso de hoje);
   - um segmento sem caixa só registra com `seg_dur > 0` ou com `falha`
     (`reg = (reg & ((seg_dur > 0) | seg_carrega)) | falha`);
   - `env.v3_cur.registra(int(c), int(m.sum()), float(x[m].sum()))`.
5. `Curriculo.registra(celula_id, n, soma)`: o terceiro argumento passa a ser a soma das notas (float; 0 ou 1 por
   segmento nas tarefas com sucesso booleano). A EMA de `fecha_iteracao` não muda: `x = soma / n`. Atualize os
   docstrings.

### 4.6 Enunciado (`specs/g1-v3-comportamento.md`)

1. §2, fase 0: 100 % dos envs andam. O comando do ANDAR é sorteado uniforme na faixa do nível (vx, vy e wz
   independentes, 10 % parado). A faixa começa em 0,5 m/s e 0,5 rad/s e soma 0,1 por nível até os tetos (frente 1,5,
   trás e lado 1,0, giro 1,0). A fase 0 termina quando o ANDAR chega ao nível 5; aí a B abre. Depois, o teto da B sobe
   só com o ANDAR no portão, e o ANDAR fica com pelo menos 30 % dos envs.
2. §6, linha do ANDAR e CARREGAR: no ANDAR, o nível controla a faixa do comando e o nível TERRENO; o sucesso é a nota
   média do segmento pelo núcleo da Unitree (exp(−erro²/0,25), média do linear e do giro, 0 na queda), com portão 0,8;
   o nível sobe com o portão por 50 iterações seguidas e não desce. O CARREGAR fica como está, com a mesma regra das
   iterações seguidas.

### 4.7 Testes

Atualize os testes que fixam as células velhas do ANDAR, a carência por tempo e o subnível de três valores. Testes
novos, todos com o `Curriculo` puro, sem simulador, salvo o último:

1. Estátua: com a nota do ANDAR em 0,69 em todo fecho, o nível do ANDAR fica em 0 por 3000 fechos, e a B fica fechada.
2. Nota 0,85 em todo fecho: o nível sobe a cada 50 fechos até 10. A B abre no fecho seguinte ao da subida ao nível 5.
3. Contagem que zera: nota 0,85 por 40 fechos, 0,0 por 2 fechos, 0,85 depois. O nível não sobe no fecho 50; ele sobe
   só depois de 50 fechos seguidos com `s ≥ 0,8`, contados da volta.
4. PEGAR pela mesma regra: a célula `pegar` com `s ≥ 0,75` por 50 fechos seguidos sobe o subnível (de 0 para 1) e,
   depois, o nível.
5. Rampa e piso: com a B aberta, a B em `s = 0` e a nota do ANDAR 0,85, o teto da B sobe 0,02 a cada 12 fechos; com a
   nota em 0,6, ele desce; a fração do ANDAR nunca fica abaixo de 0,30, e a soma das frações é 1.
6. Sorteio do ANDAR (com gerador fixo): no nível 0, |vx|, |vy| ≤ 0,5 e |wz| ≤ 0,5; no nível 10, vx até 1,5, vy até 1,0
   e wz até 1,0; ~10 % dos comandos zerados; no `terreno`, vx ≤ `terreno_v_max`.
7. Nota do segmento (no padrão de env falso dos testes de comando): rastreio perfeito dá ~1; robô parado com comando
   (0,5; 0; 0) depois da âncora dá (e⁻¹ + 1)/2 ≈ 0,684; queda dá 0.

## 5. Verificação do lote (orquestrador)

1. Suíte completa, com só a falha antiga do `test_mid360`.
2. Contagem de funções do `g1_v3` sem testes: 476 → ≤ 476.
3. Treino local de 2 its em CPU, 16 envs, com `--agent.logger tensorboard` e `WANDB_MODE=disabled`: sem erro nem NaN.
4. `git log 0ca0827..HEAD --format=%B | grep -ci co-authored` = 0.
5. Revisão (opus, esforço médio) do lote contra esta spec.
6. Na corrida do zero, olhar: `Curriculum/v3/nivel_ANDAR` parado até `s_ANDAR_andar` passar de 0,8 e subindo no máximo
   a cada 50 its; `frac_ANDAR` em 1,0 até o nível 5 e depois ≥ 0,30; `Metrics/twist/acerto_relogio` acima de 0,55.

## 6. Tarefa T3: rastreio da Unitree, com velocidade instantânea (dono, 16/11)

Motivo (inbox do vault, 16/11, corrida unitree1): a nota do ANDAR parou em 0,75 a 0,78, e o robô desliza torcendo a
pelve. O erro instantâneo de giro é ~7 vezes o de um robô parado (`error_vel_yaw` 4,13 → ~1,65 rad/s), em todas as
corridas da v3. O rastreio e a nota leem a média de 0,8 s, que anula a torção. A Unitree e o g1_limpo usam a velocidade
instantânea. Decisões do dono: rastreio igual ao da Unitree (núcleo cru, velocidade instantânea), pesos 1,0 e 0,5;
o `vivo` fica 1,0; o sinal de progresso do CARREGAR (M5) fica normalizado.

Arquivos: `g1_v3/knobs.py`, `g1_v3/recompensas.py`, `g1_v3/comando.py`, `specs/g1-v3-comportamento.md` (alvo do link no
vault), `g1_v3/testes/`. As regras da §2 valem. Linha de base: commit `3b4ab10`, 476 funções. Meta: ≤ 476, nenhuma
função nova.

1. `knobs.Recompensa`:
   - `rastreio_linear`: 2.0 → 1.0; `rastreio_angular`: 2.0 → 0.5. Comentário: "16/11, T3: pesos da Unitree; o 2,0 de
     11/11 compensava a renda negativa do rastreio normalizado".
   - `sigma_rastreio` fica 0,5.
   - `piso_rastreio`, `sigma_zero_lin`, `sigma_zero_ang` e `sigma_zero_ang_inicio` ficam. Atualize o comentário de cada
     um: só o sinal de progresso do CARREGAR (M5) os usa.
2. `knobs.Comando`: sai `janela_media_s`.
3. `recompensas._rastreio(env, linear, normalizado=False)` (a função existente ganha o parâmetro):
   - velocidade instantânea da base, no frame da base, como o comando: `v_xy = robot.data.root_link_lin_vel_b[:, :2]`,
     `wz = robot.data.root_link_ang_vel_b[:, 2]`, com `robot = env.scene["robot"]`;
   - `normalizado=False` (rastreio da Unitree): linear `exp(−‖cmd_xy − v_xy‖² / σ²)`; angular `exp(−(cmd_z − wz)² / σ²)`;
     σ = `R.sigma_rastreio`. O canal sem comando entra no mesmo núcleo (c = 0);
   - `normalizado=True`: a conta de hoje (`potencial.rastreio_normalizado`, com o σ₀ e o piso), com a velocidade
     instantânea no lugar da média.
   Reescreva o docstring.
4. `recompensas.rastreio_linear`, `rastreio_angular` e `_rastreio_com_qualidade`: chamam `_rastreio(env, linear)` (forma
   da Unitree). A qualidade do CARREGAR (E4) segue escalando a parte positiva. Reescreva os docstrings (não citam mais
   o rastreio normalizado nem o σ₀).
5. `recompensas.progresso` (sinal do CARREGAR, M5): `r_lin, r_ang = _rastreio(env, True, normalizado=True),
   _rastreio(env, False, normalizado=True)`. O resto da conta do sinal não muda.
6. `comando.TwistV3`:
   - saem `self.v_media_b`, `self._hist` e `self._i` do `__init__`, as três linhas que os atualizam no `_update_metrics`
     e as duas que os zeram no `reset`;
   - a nota do segmento (bloco "16/11, A1") usa `vel` e `wz`, a velocidade instantânea que o bloco 2 já lê:
     `e_lin = ((cmd[:, :2] − vel) ** 2).sum(dim=1)`, `e_ang = (cmd[:, 2] − wz) ** 2`. Atualize o comentário (sem "média
     de 0,8 s").
   Confira com `grep` que nada mais em `g1_v3/` e em `pilota_v3.py` lê `v_media_b` ou `janela_media_s`.
7. Enunciado (`specs/g1-v3-comportamento.md`): nas três linhas que citam o rastreio normalizado do ANDAR (§5.2, a linha
   "Não gira no lugar" da §7 e a linha "Andar normal" da §10), troque pela forma da Unitree: núcleo cru, velocidade
   instantânea, pesos 1,0 e 0,5 (dono, 16/11). Registre que o rastreio normalizado fica só no sinal de progresso do
   CARREGAR, e que o problema do comando baixo (conflito 5) volta a ficar em aberto para o refino.
8. Testes:
   - o rastreio da Unitree: robô parado com comando (0,5; 0; 0) recebe exp(−1) no linear e 1 no giro; com o comando
     cumprido, 1; a oscilação instantânea da velocidade de giro (±1,65 rad/s em torno do comando) baixa o termo angular
     para perto de 0 (o caso que a média de 0,8 s escondia);
   - o `normalizado=True` dá o valor de hoje com a velocidade instantânea (parado com comando dá 0);
   - a nota do segmento com a velocidade instantânea: rastreio perfeito dá 1; parado com (0,5; 0; 0) dá ≈ 0,684; queda
     dá 0; a oscilação do giro baixa a nota;
   - os pesos 1,0 e 0,5 no cfg montado (`test_cfg` ou equivalente);
   - atualize os testes que fixam o rastreio normalizado no termo de recompensa, o `v_media_b` ou a janela.

Verificação do orquestrador depois da T3: suíte; contagem de funções ≤ 476; treino local de 2 its em CPU
(`--gpu-ids None`, `--agent.logger tensorboard`, `WANDB_MODE=disabled`); `grep -ci co-authored` = 0; revisão (opus, esforço
médio). Na corrida do zero, olhar: `Metrics/twist/error_vel_yaw` abaixo de ~1,0 nas primeiras 300 its (erro de giro
perto de 0,4 rad/s), depois `tempo_de_voo` acima de 0,1 s e `razao_marcha` perto de 0,47 até a it 800.

## 7. Tarefa T4: fase do relógio no ator e nota pelo mínimo dos canais (dono, 16/11)

Motivo: auditoria "a tarefa certa paga mais" (`/home/joaobornelli/workbench/projects/g1-treino-v3/auditoria-tarefa-certa-paga-mais.md`,
§2 achados 1 e 4). O `passo` paga a fase de um relógio que o ator não vê: andar em fase rende 4,30/s, com fase aleatória
3,81/s, parado 3,48/s, em antifase 3,40/s. A nota do segmento soma o linear e o giro meio a meio, e andar sem girar passa
no portão nos níveis 0 a 2. Decisão do dono: a fase vai ao ator (reverte a decisão de 14/10; o robô real calcula a fase
no laço da política, pelo tempo, sem sensor) e a nota passa a ser o mínimo dos dois canais.

Arquivos: `g1_v3/knobs.py`, `g1_v3/observacoes.py`, `g1_v3/comando.py`, `pilota_v3.py`, `g1_v3/exporta_cena.py`,
`specs/g1-v3-comportamento.md` (alvo do link no vault), `g1_v3/testes/`. As regras da §2 valem. Linha de base: commit
`a89ea9b`, 476 funções. Meta: ≤ 476, nenhuma função nova.

### 7.1 Fase no ator (item R1)

1. `knobs.Contrato.blocos_ator`: entra `("fase", 2, 1.0, 5)` logo depois de `("comando", 3, 1.0, 5)`. O ator passa de
   1447 para 1457 canais. Atualize os comentários que citam 1 447 (`knobs.py`, `dim_ator`).
2. `observacoes.privilegiados(env, fase=False)` (a função existente ganha o parâmetro):
   - `fase=True`: devolve (E,2) com o seno e o cosseno de 2π·`v3_fase_marcha`;
   - `fase=False`: devolve (E,4) com `v3_marcha_on`, `v3_h`, a lacuna e `log1p(v3_sigma)` (o seno e o cosseno saem
     daqui, porque o crítico passa a recebê-los pelo termo `fase`).
   Reescreva o docstring: o termo `fase` vai ao ator e ao crítico; o `privilegiados` fica só no crítico.
3. `observacoes.grupos`: a lista `nove` ganha `("fase", privilegiados, {"fase": True})` logo depois de `comando`. O termo
   não tem ruído nem atraso (não entra em `ruido_obs` nem em `_ATRASO`), passa pelo normalizador e tem histórico 5 (vem
   do `blocos_ator`). O crítico recebe o termo pela mesma lista; o `tarefa_critico` continua por último.
4. `pilota_v3.monta_observacao(..., fase)`: entra o parâmetro `fase: float`, a fase do relógio em [0, 1). O dicionário
   `valores` ganha `"fase": [sin(2π·fase), cos(2π·fase)]`. Atualize o docstring (1457 canais, o quadro com histórico
   cresce 2 canais).
5. Laço do `pilota_v3.main`: um contador de passos da política zera em todo `restaura` (onde o `hist` vira `None`) e
   soma 1 antes de cada `monta_observacao`; `fase = (passos · dt_politica / periodo_marcha) mod 1`, com
   `dt_politica = dt_fisica · decimacao` e `periodo_marcha` do knobs. É a mesma conta do env: o `_agora` do `twist` soma
   dt antes de gravar a fase, então a primeira observação depois do reset tem fase dt/0,8.
6. `exporta_cena.py`: a conferência contra o env passa `fase = v3_fase_marcha` do env 0 para `monta_observacao`. Atualize
   os textos que citam 1447 e a lista dos blocos.

### 7.2 Nota pelo mínimo dos canais (item R2)

1. `comando.TwistV3`: `seg_nota` passa a (E,2), o linear e o giro, cada um com o seu núcleo, acumulados como hoje (com
   `apos` e dt). No `_fecha`: `nota = (seg_nota / seg_dur[:, None].clamp(min=1e-6)).min(dim=1).values`. O resto do
   `_fecha` não muda. Atualize o comentário do bloco.
2. Enunciado (`specs/g1-v3-comportamento.md`, §6, linha do ANDAR): a nota é o mínimo do linear e do giro (cada um a
   média do núcleo no segmento), e não a média dos dois.

### 7.3 Testes

- O ator tem 1457 canais (`dim_ator` e o cfg montado); o termo `fase` vem logo depois do `comando` e tem histórico 5.
- `privilegiados(env, fase=True)` dá o seno e o cosseno certos para três fases (0, 0,25, 0,5); `fase=False` dá 4 canais.
- `pilota_v3.monta_observacao` com a fase põe o seno e o cosseno na fatia do bloco `fase`; o teste de paridade do
  `test_pilota.py` passa com a assinatura nova.
- A nota do segmento: rastreio perfeito dá 1; parado com comando (0,5; 0; 0) dá exp(−1) ≈ 0,368 (antes ≈ 0,684); andar
  certo no linear sem girar, com comando de giro 0,5 rad/s, dá exp(−1) ≈ 0,368; queda dá 0.
- Atualize os testes que fixam 1447, a forma (E,6) do `privilegiados` e a nota pela média.

Verificação do orquestrador depois da T4: suíte; contagem de funções ≤ 476; treino local de 2 its em CPU; `grep -ci
co-authored` = 0; revisão (opus, esforço médio) da T4. Na corrida do zero, olhar também `acerto_relogio` acima de 0,55
e subindo.

## 8. Tarefa T5: giro da nota pelo erro médio do segmento (dono, 17/11)

Motivo: medida no pod com `g1_v3/mede_deriva.py` (checkpoint da it 2250 da unitree2, 128 envs, 3000 passos, ação
amostrada, 1173 segmentos do ANDAR). O canal linear dá 0,886. O canal do giro instantâneo dá 0,701, e o giro pelo ω_z
médio do segmento dá 0,956. A nota (o mínimo) dá 0,700, com 1 % dos segmentos ≥ 0,8; com o giro pelo erro médio, ela dá
0,877, com 90 % dos segmentos ≥ 0,8. O erro de giro instantâneo é o balanço da pelve dentro da passada (período 0,8 s):
com ele, o portão exige a pelve quase sem rotação (±2,5° por passada), e o nível do ANDAR não sai do 0 (s 0,616 na it
2600). Decisão do dono: o canal do giro da nota mede o giro do segmento (o erro médio de ω_z depois da âncora); o canal
linear continua com o núcleo instantâneo. A recompensa não muda: o `rastreio_angular` continua a cobrar o giro
instantâneo (T3), então a torção da pelve continua a custar na recompensa.

Arquivos: `g1_v3/comando.py`, `g1_v3/mede_deriva.py`, `specs/g1-v3-comportamento.md` (link; o arquivo real é
`vault/G1/Abordagens/PPO/Currículos/C4 v3/Comportamento esperado da v3.md`), `g1_v3/testes/test_marcha_relogio.py`.
As regras da §2 valem, com "17/11, T5" nos comentários novos. Nenhum knob novo: σ continua `Comando.sigma_nota`.
Linha de base: commit `14d017b`, 476 funções no `g1_v3` sem testes. Meta: ≤ 476, nenhuma função nova.

### 8.1 Nota do segmento (`comando.TwistV3`)

1. `_update_metrics`, bloco da nota (hoje `e_lin`, `e_ang` e `self.seg_nota += torch.exp(...)`):
   - coluna 0 do `seg_nota`: igual a hoje, exp(−‖c_xy − v_xy‖²/σ²)·apos·dt;
   - coluna 1 do `seg_nota`: o erro de giro com sinal, sem núcleo, (c_z − ω_z)·apos·dt;
   - `seg_dur` não muda. `cmd`, `vel` e `wz` continuam os do bloco 2 (instantâneos).
2. `_fecha`: `media = seg_nota[ids] / seg_dur[ids, None].clamp(min=1e-6)`; linear = `media[:, 0]`;
   giro = exp(−`media[:, 1]`²/σ²), com σ = `k.sigma_nota`; `nota` = mínimo(linear, giro). O resto do `_fecha` não muda
   (`x`, `reg`, `registra`, eficiência, zeragem).
3. Comentários: a linha do `seg_nota` no `__init__` diz [linear: Σ núcleo·dt; giro: Σ (c_z − ω_z)·dt], depois da âncora.
   O comentário do bloco da nota e o do `_fecha` dizem a regra nova e o motivo curto: o balanço da pelve dentro da passada
   se anula na média do segmento; a recompensa continua a cobrar o giro instantâneo.

### 8.2 Script de medida (`g1_v3/mede_deriva.py`)

Com a T5, as colunas de estudo da T5 repetem o portão. O script volta a medir só a regra do `_fecha`:
1. Saem o `wz_soma` (declaração, acúmulo no laço e zeragem), o `wz_med`, o `giro_t5` e o `x_t5`.
2. No `fecha_medindo`: `media` como no `_fecha`; grava `nota_lin` = `media[:, 0]`, `nota_giro` = exp(−`media[:, 1]`²/σ²),
   `erro_giro` = `media[:, 1]` (rad/s, com sinal) e `x` = where(carrega, sucesso, mínimo(`nota_lin`, `nota_giro`))·(1 − falha).
3. Tabela: as colunas "nota hoje", "nota T5", "%seg≥portão hoje", "%seg≥portão T5", "lin", "giro" e "giro T5" viram
   "nota", "%seg≥portão", "lin" e "giro". As colunas de deriva (`|rumo|`, `dpos`, viés) não mudam.
4. Gráfico `nota.png`: por célula, o painel 0 é o histograma de `x` e o painel 1 sobrepõe os histogramas de `nota_lin` e
   `nota_giro`; os dois com a linha do portão.
5. Docstring: a frase "16/11, T5 em estudo: ..." vira "17/11, T5: o giro da nota é o núcleo do erro médio de ω_z no
   segmento".

### 8.3 Enunciado (`specs/g1-v3-comportamento.md`, §6, linha "ANDAR e CARREGAR")

O trecho "o mínimo entre o linear e o giro, cada um a média do núcleo no segmento, e não a média dos dois [T4, 16/11]"
passa a "o mínimo entre o linear (a média do núcleo no segmento) e o giro (o núcleo do erro médio de giro no segmento, que
ignora o balanço da pelve dentro da passada), e não a média dos dois [T4, 16/11; T5, 17/11]".

### 8.4 Testes (`g1_v3/testes/test_marcha_relogio.py`, classe `NotaDoSegmento`)

- Comentário da classe: a regra nova (linear pela média do núcleo; giro pelo núcleo do erro médio).
- O helper `_nota` calcula `acumulada` pela regra nova: mínimo(média do canal 0, exp(−(média do canal 1)²/0,25)).
- Os casos com velocidade constante ficam com os valores de hoje: rastreio perfeito dá 1; parado com comando
  (0,5; 0; 0) dá e⁻¹; andar certo no linear sem girar, com comando de giro 0,5 rad/s, dá e⁻¹; queda registra 0.
- `test_oscilacao_do_giro_baixa_a_nota` sai. No lugar entram dois casos, os dois com `obj.t_ab[:] = -10.0` antes dos
  passos, para que os 100 passos contem depois da âncora (número par de passos), linear perfeito (v_x = 0,5) e comando
  (0,5; 0; 0,3), conferidos pelo valor registrado (`_fecha` e o mock do `registra`):
  - `test_balanco_do_giro_em_torno_do_comando_nao_baixa_a_nota`: ω_z = 0,3 ± 1,65 alternado → registra 1,0 (places=5).
    No código de antes, registra ≈ 0;
  - `test_vies_de_giro_com_balanco_cobra_pelo_erro_medio`: ω_z = 0,5 ± 1,65 alternado (erro médio −0,2) → registra
    exp(−0,16) ≈ 0,8521 (places=3). No código de antes, registra ≈ 0.
- Não mexa nos testes do `rastreio_angular` da recompensa (`test_recompensas*.py`): a recompensa continua instantânea.

Verificação do orquestrador depois da T5: suíte; contagem de funções ≤ 476; fumaça do `mede_deriva.py` em CPU (8 envs,
400 passos); treino local de 2 its em CPU; `grep -ci co-authored` = 0; revisão (opus, esforço médio) da T5. Na retomada
(RETOMA = 1 a partir da it 2600), olhar `s_ANDAR_andar` acima de 0,8 em ~60 its e `nivel_ANDAR` subindo depois de 50 its
seguidas no portão; `error_vel_yaw` e `rastreio_angular` não devem piorar.

## 9. Tarefa T6: registro da nota do ANDAR sem a regra do instante da âncora e empurrão com intervalo sorteado (dono, 17/11)

Motivo: com a T5, o `s_ANDAR_andar` do treino parou em 0,72 a 0,75, e a medida dava 0,879 por segmento. No TB do treino,
o `x` de cada iteração tirado da EMA (x = s_ant + Δs/0,05) tem p50 0,851, mas 15,7 % das iterações têm x ≈ 0; a média
(0,716) é o `s`. Antes da T5 era igual: p50 0,636, 16 % de zeros, `s` 0,60. As iterações com x ≈ 0 voltam a cada 251
passos (concentração de fase 0,95 nas duas corridas), o período do empurrão (5 s). Duas causas juntas:
1. O empurrão chega a todos os envs no mesmo passo. O `event_manager` do mjlab só reinicia o relógio de intervalo dos
   termos que são classe (`mjlab/managers/event_manager.py:226`); o `eventos.empurrao` é função, então o relógio corre do
   início do treino, igual em todos os envs, e o intervalo fixo (5, 5) o mantém em fase para sempre.
2. O empurrão leva a âncora 1,5 s à frente, e o `sucesso_do_segmento` só registra o segmento que fecha depois da âncora
   (`registra = t_fecho >= t_ancora`, PD-15). A queda registra sempre (`| falha` no `_fecha`). Por 1,5 s depois de cada
   empurrão, nenhum segmento normal registra; com 4096 envs, essas iterações só têm quedas, x = 0, e a EMA, que pesa cada
   iteração igual, toma o zero.
Decisão do dono: o empurrão deixa de ser um relógio de 5 s, e o ANDAR registra pela nota, sem a regra do instante da
âncora. A nota já exclui o trecho do empurrão, porque só soma os passos depois da âncora (`apos`).

Arquivos: `g1_v3/knobs.py`, `g1_v3/env_cfg.py`, `g1_v3/comando.py`, `g1_v3/mede_deriva.py`,
`specs/g1-v3-comportamento.md` (link; o arquivo real é `vault/G1/Abordagens/PPO/Currículos/C4 v3/Comportamento esperado
da v3.md`), `g1_v3/testes/test_marcha_relogio.py`, `g1_v3/testes/test_cfg.py`. As regras da §2 valem, com "17/11, T6"
nos comentários novos. Linha de base: commit desta seção, 476 funções no `g1_v3` sem testes. Meta: ≤ 476, nenhuma função
nova.

### 9.1 Registro do ANDAR (`comando.TwistV3._fecha`)

1. A linha `reg = (reg & ((self.seg_dur[ids] > 0) | self.seg_carrega[ids])) | falha` passa a
   `reg = torch.where(self.seg_carrega[ids], reg, self.seg_dur[ids] > 0) | falha`.
   - ANDAR: registra o segmento com tempo depois da âncora (`seg_dur > 0`) ou que acaba em queda.
   - Tarefas com caixa: igual a hoje (`reg` do `sucesso_do_segmento`, ou a queda).
2. `sucesso_do_segmento` não muda: as tarefas com caixa seguem a PD-15.
3. Comentário do bloco: a regra nova e o motivo curto (depois de cada empurrão, só as quedas registravam, e a EMA tomava
   zeros).

### 9.2 Empurrão com intervalo sorteado (`knobs.DR`, `env_cfg`)

1. `DR.periodo_empurrao_s: float = 5.0` passa a `periodo_empurrao_s: tuple = (3.0, 7.0)`, com o comentário: intervalo
   sorteado por env, média de 5 s (a da Unitree); o relógio de um termo de intervalo que é função não reinicia no reset
   do episódio (`mjlab/managers/event_manager.py:226`), e o intervalo fixo deixava todos os envs empurrados no mesmo
   passo.
2. `env_cfg.py`: `interval_range_s=k.dr.periodo_empurrao_s`.

### 9.3 Script de medida (`g1_v3/mede_deriva.py`)

A coluna `reg` espelha a regra nova do `_fecha`: `torch.where(carrega, reg, seg_dur > 0) | falha`.

### 9.4 Enunciado (`specs/g1-v3-comportamento.md`)

Na linha "Resistir a empurrões", "±0,5 m/s a cada 5 s" passa a "±0,5 m/s a cada 3 a 7 s, sorteado por env [T6, 17/11]".

### 9.5 Testes

- `test_marcha_relogio.py`, classe `NotaDoSegmento`, caso novo `test_empurrao_depois_da_ancora_nao_tira_o_registro`:
  comando (0,5; 0; 0) e rastreio perfeito; 100 passos (0,5 s depois da âncora); depois `obj._env.v3_t_empurrao[:] = 1.9`
  e mais 1 passo (a âncora vai a 3,4 s, à frente do relógio); `_fecha` sem queda registra 1 segmento com nota 1,0
  (places=5). No código de antes, `t_fecho < t_ancora` e o `registra` não é chamado.
- `test_cfg.py`: o evento `empurrao` do cfg de treino tem `interval_range_s == knobs.ATIVO.dr.periodo_empurrao_s`, com o
  limite de baixo menor que o de cima. No código de antes, o intervalo é (5, 5).

Verificação do orquestrador depois da T6: suíte; contagem de funções ≤ 476; fumaça do `mede_deriva.py` em CPU; treino
local de 2 its em CPU; `grep -ci co-authored` = 0; revisão (opus, esforço médio). Na retomada (RETOMA = 1), olhar
`s_ANDAR_andar` acima de 0,8 e `nivel_ANDAR` subindo; o `x` por iteração tirado da EMA sem iterações em zero.

## 10. Tarefa T7: comando do ANDAR sorteado na elipse dos tetos (dono, 17/11)

Motivo: com a T6, a medida e o treino batem (x por fecho 0,780 na medida, 0,763 no TB), e o nível 4 parou em x ≈ 0,76.
No `segmentos.csv` do `mede_deriva` (checkpoint da it 5605, nível 4, 5329 segmentos), a `nota_lin` cai com a velocidade
total |v| = |(vx, vy)|: 0,91 a 0,93 até 0,3 m/s, 0,872 de 0,3 a 0,6, 0,765 de 0,6 a 0,9 e 0,583 de 0,9 a 1,27. A faixa
acima de 0,9 só existe na diagonal: o sorteio põe vx e vy independentes em ±teto (quadrado), e a diagonal passa do teto
em até 41 %; ela é 18 % dos segmentos. Decisão do dono: o (vx, vy) do ANDAR passa a ser uniforme na elipse dos tetos. Os
tetos de cada eixo não mudam (frente 1,5, trás e lado 1,0); só saem os cantos da diagonal. Sem os segmentos com
|v| > 0,9, a `nota_lin` média do nível 4 seria ~0,838 (estimativa pelos mesmos segmentos).

Arquivos: `g1_v3/comando.py`, `g1_v3/testes/test_comando.py`, `specs/g1-v3-comportamento.md` (link; o arquivo real é
`vault/G1/Abordagens/PPO/Currículos/C4 v3/Comportamento esperado da v3.md`). Comentários novos com "17/11, T7". Meta:
nenhuma função nova.

### 10.1 Sorteio (`comando.sorteia_velocidade`, ramo do ANDAR)

1. As linhas `v[:, 0] = -tras + (frente + tras) * U` e `v[:, 1] = lado * (2U − 1)` passam a:
   - r = √U e ângulo θ ~ U(−π/2; π/2): ponto uniforme na meia-unidade de disco com x ≥ 0;
   - semieixo x = `frente` com chance frente/(frente + tras), senão −`tras` (chance pela área de cada meia elipse);
   - vx = semieixo x · r · cos θ; vy = `lado` · r · sen θ.
2. O wz, o `andar_parado` e o teto do `terreno` não mudam. O CARREGAR não muda.
3. O docstring diz "vx e vy uniformes na elipse dos tetos, wz independente".

### 10.2 Enunciado (`specs/g1-v3-comportamento.md`, §2, fase 0)

"(vx, vy e wz independentes, 10% parado)" passa a "((vx, vy) uniforme na elipse dos tetos e wz independente, 10%
parado) [T7, 17/11]".

### 10.3 Testes (`test_comando.py`)

- Caso novo `test_andar_sorteia_vx_vy_na_elipse_dos_tetos`, no nível 10: com os envs ativos, (vx/semieixo)² + vy²/lado²
  ≤ 1; a fração com esse valor ≤ 0,5 fica em 0,5 ± 0,01 (uniforme na área); a fração com vx > 0 fica em
  frente/(frente + tras) = 0,6 ± 0,01. No código de antes, o máximo chega a 2.
- O comentário do `test_andar_nivel_0_fica_em_0_5_com_10_por_cento_parado` muda para o disco; os asserts não mudam.

Verificação do orquestrador: suíte; nenhuma função nova; `grep -ci co-authored` = 0. Na retomada (RETOMA = 1), olhar o
`s_ANDAR_andar` subir acima de 0,8 no nível 4 e o `nivel_ANDAR` chegar a 5.

## 11. Tarefa T8: o P3 só fecha com a caixa levantada (dono, 18/11)

**Medida (model_8150, `g1_v3/coleta.py`, 256 envs × 3000 passos, ação média e amostrada).** Em 369 fechos do P3, o
fundo da caixa estava a 0–4 mm da laje (p50 0,000 m). A maior altura da visita inteira (P3 + P4) foi p50 1,2–1,6 cm e
p99 3,3 cm; nenhuma passou de 5 cm. O P4 abria com a caixa na laje e a renda da etapa (24·h por s, τ 3 s) pagava
~72 por visita com o robô parado. `s_PEGAR` = 0 em 3904 medidas.

**Causa.** O fecho do P3 era `~apoio & incl`, com tempo 0: um passo de 0,02 s sem contato caixa × laje bastava, sem
altura. O enunciado pede "a caixa 5 cm acima da laje, nivelada, sem apoio" (Comportamento esperado, tabela dos alvos).

**Mudança.**
- `ok_agora_pegar` recebe `subida` (fundo da caixa sobre o topo da laje, pela verdade: `h_caixa − meia − v3_topo`).
  P3: `~apoio & incl & subida ≥ p3_alto − tol_caixa_p3`. P4: a condição antiga `& ~apoio`.
- `_TEMPOS[P3]` = `tempo_p3`. Knobs novos em `Alvos`: `tol_caixa_p3` = 0,02 m e `tempo_p3` = 0,3 s.
- Contagem de funções: 472 → 472. Testes: `test_p3_exige_a_caixa_levantada_por_tempo_p3` e o caso do P4 apoiado.

**O que olhar no TB.** `Episodio/tempo_P3` sobe (o P3 deixa de fechar no toque); `Episodio/phi_P3` deve subir com a
pega firme; `forca_robo_caixa` sobe. Se o `phi_P3` não subir em ~1000 its, o gradiente do P3 não basta.

**Aberto (não entra na T8).** O sucesso do segmento do CARREGAR (`sucesso_do_segmento`) só exige `~perdeu`: com a
caixa deixada na laje depois do fecho do P4 e o robô a < `v_solta` (1,2 m/s), o segmento conta como sucesso.

## 12. Tarefa T9: fatias em partes iguais e rampa só por tempo (dono, 18/11)

**Estado medido (model_8150).** `rampa` = [0, 6, 0]: o teto da B estava em 0,17, 3900 its depois da abertura (it 4222).
Na retomada com a T8 o João viu `frac_ANDAR` = 0,95: o teto voltou a 0,05.

**Causa.** O teto da B só subia com a nota do ANDAR ≥ 0,80 e descia com ela < 0,70 (§4.4, itens B2 e B3). O ANDAR sobe
de nível ao ficar 50 its no portão, e no nível novo a nota cai. A B só ganhava envs nessas 50 its e os devolvia a cada
nível novo. A regra veio do g1_limpo, onde o ANDAR era fácil; na v3 os últimos níveis do ANDAR são difíceis.

**Decisão do dono.** Aprender o básico de uma tarefa abre a seguinte; depois os refinos rodam em paralelo, com os envs
divididos em partes iguais. A fração de uma fatia não depende mais da nota de outra.

**Mudança.** Revoga os itens 2 e 3 da §4.4.
- `fracoes_fatias`: cada fatia aberta recebe 1/n (2 fatias: 1/2; 3 fatias: 1/3). O déficit sai da conta das fatias; a
  conta das células dentro da tarefa (`fracoes_celulas`) não muda.
- Rampa: a fatia nova entra com teto `rampa_fatia_inicio` (0,05) e o teto sobe `rampa_fatia_passo` (0,02) a cada
  `rampa_fatia_its` (12) fechos, sem condição e sem descida. O que o teto corta volta em partes iguais às fatias de
  teto maior.
- Saem os knobs `piso_andar` e `histerese_rampa`. O estado do checkpoint não muda de formato (`rampa`, `it_abertura`).
- A abertura não muda: a B no nível `nivel_abre_b` (5) do ANDAR, a C com o PEGAR nível 0 no portão.
- Contagem de funções do `balanceador.py`: 34 → 34.

**O episódio da fatia.** A fatia B é uma cadeia num episódio só: espera, PEGAR, CARREGAR. A C é a cadeia inteira até o
BOTAR. Não existe sorteio de tarefa dentro da fatia: com a B em 1/2, o PEGAR começa em metade dos envs.

**Retomada do model_8150 (simulada no `Curriculo`).** `frac_B` 0,17 na retomada, 0,31 depois de 80 fechos e 0,50
depois de 200.

**O que olhar no TB.** `Curriculum/frac_B` sobe 0,02 a cada 12 its até 0,50. `s_ANDAR_andar` no nível 6: se cair muito
com 50 % dos envs, o custo da divisão igual aparece aqui.

## 13. Tarefa T10: conquistas mantidas na renda da cadeia e fecho do P2 com tempo (dono, 18/11)

**Medida (model_9150, T8 + T9, `g1_v3/coleta.py`, ação amostrada, 386 episódios da fatia B).** 198 visitas ao P3 e 0
fechos. No 1º passo do P3, 1 % dos envs tem as duas palmas na caixa; de 0,1 s em diante, 71 a 78 % dos passos sem
nenhuma palma. Portão `maos` médio 0,08; Φ médio 0,03. Apertar firme (Fmin > 20 N) paga +1,0 por s sobre não apertar.
A caixa deriva 8 cm (p50) na horizontal; 27 das 46 perdas saem do P3.

**Causa.** (1) O P2 fecha com um passo de contato das duas palmas (tempo 0): o robô toca, recebe o prêmio de entrada
do P3 (~54) e tira as mãos. (2) O portão `maos` multiplica o Φ de P3 a B2: com uma palma fora o Φ vale 0, e nenhum
termo puxa a palma de volta.

**Regra (dono).** Uma conquista é um estado que exige ação contínua e que as etapas seguintes ainda precisam. Ela
paga uma renda própria, somada, enquanto vale. O prêmio de avanço (`renda_etapa`·n·h) não muda.

**Fórmula.** `r = renda_etapa·n·h + renda_phi·Φ_n + renda_conquista·(q_palmas + q_pega)`
- `Φ_n`: como hoje, com o portão `maos` em 1 (os portões `desloc`, `incl`, `tronco`, `recuo` e `com` não mudam).
- `q_palmas = 1 − tanh(d_palmas / (largura_tanh·sigma_palmas))`; `d_palmas` = média da distância das duas palmas a
  `alvos.palmas_alvo_caixa(P2, a, k)`, no frame da caixa.
- `q_pega = tanh(min(F_esq, F_dir) / F_pega)`: a expressão que hoje alimenta o portão `maos` no `_phi_env`.
- As conquistas somam nas etapas P3, P4, B1 e B2 e no CARREGAR com o P4 fechado. Valem 0 nas outras, depois da perda
  da caixa e no CARREGAR sem o P4 fechado.

**Mudança.**
- `recompensas._phi_env`: devolve `(Φ, conquistas)`, com `conquistas` (E,) = `q_palmas + q_pega`; passa 1 no
  argumento `maos` de `portoes_do_alvo`.
- `recompensas.n_progresso`: recebe `conquistas` e `w_c`; devolve `w_etapa·etapa·h + w_phi·Φ + w_c·conquistas`.
- `recompensas.progresso`: zera as conquistas fora das etapas acima e as passa ao `n_progresso`. O h, a recarga e o
  `v3_phi` não mudam.
- `recompensas.portoes_do_alvo`: não muda. A cadeia continua a passar contatos/2 no `maos` do invariante.
- `comando._TEMPOS`: a posição do P2 vira `A.tempo_p2`.
- Knobs novos: `Recompensa.renda_conquista` = 2,0; `Recompensa.sigma_palmas` = 0,08 (m); `Alvos.tempo_p2` = 0,3 (s).
- `coleta.py`: lê o par e grava as colunas `q_palmas` e `q_pega`.
- Contagem de funções de `g1_v3/*.py`: 477 → 477. Observações e formato do checkpoint não mudam.

**Limite do peso.** O fecho do B2 perde as duas conquistas. Para a abertura do B3 pagar mais que o máximo do B2 (a
regra de 09/11), `renda_conquista` < 2,3: `42 + 3·0,54 > 36 + 3 + 2·w_c`.

**Testes.**
- `progresso` no P3 com Φ = 0 e conquistas = 2 paga `18·h + 4`; no P2 e no B3 as conquistas não entram.
- CARREGAR com o P4 fechado soma as conquistas; depois da perda não soma.
- Os 7 fechos de P2 → P3 a B3 → B4: abertura (h = 1, Φ = 0,54) > máximo anterior (h = 1, Φ = 1, conquistas cheias),
  com os knobs do repo.
- `_phi_env` no P3 com uma palma fora: o Φ não vai a 0 e `q_palmas` cai.
- O P2 só fecha depois de `tempo_p2` de contato das duas palmas.
- Os testes que trocam `_phi_env` por um Φ fixo passam a devolver o par.

**O que olhar.** Coleta ~1000 its depois da retomada: passos do P3 sem palma na caixa abaixo de 30 % (hoje 71 a 78 %);
Fmin > 5 N no P3 acima de 50 % (hoje 11 %); deriva da caixa p50 abaixo de 3 cm (hoje 8 cm); fechos do P3 acima de 0.

**Aberto (não entra na T10).** O `sigma_palmas` não tem medida. O custo da perda da caixa soma −4,8 por perda no
retorno (240 × 0,02 s). A recarga do h lê o pico do Φ.

## 14. Tarefa T11: a mão alcança a caixa e o P2 paga o que o fecho dele testa (dono, 18/11)

**Regra (dono).** A mão tem de tocar a caixa. O braço e o tronco também podem tocar. A almofada (`*_palm_pad`) é a
superfície de contato da mão na simulação.

**Medidas (model_10600 e model_10850, T10).**
- P2: 232 visitas e 0 fechos; as duas almofadas tocam em 9 % do tempo, em sequências de 1 a 3 passos (máximo 9).
- O site das palmas fica a 3,6 a 4,1 cm da face lateral (mediana); o alvo é 3,5 cm. Nessa faixa a almofada toca em
  25 a 35 % do tempo; com o site de 1,5 a 3,0 cm, em 70 a 80 %.
- Geometria: a face da almofada fica a 2,3 cm do eixo da mão (`_PALM_DZ` + meia espessura); a cápsula do punho
  (`*_wrist_collision`) tem raio 3,5 cm. Com a mão plana, o punho toca e a almofada fica a 1,2 cm da face. O 0,035 do
  `palma_face` veio do gerador de IK, que o escolheu para o punho encostar (`g1_limpo/ik/gera_botar.py:54-60`).
- A malha da mão (`left_rubber_hand`) tem os dedos de 3,9 a 4,5 cm do eixo, do lado da palma; a cápsula da mão do
  fabricante (`left_hand_collision`, apagada pelo `add_pads_de_palma`) alcança de 3,5 a 5,5 cm.
- Força nas duas faces laterais, de qualquer peça do robô: 23 % do tempo do P2, mediana da menor força 16,8 N, maior
  sequência 14 passos. O punho já alcança a caixa e o contato dele também é curto.
- Renda no P2 depois de 2 s: 3,02/s sem palma, 3,44/s com as duas. As conquistas não pagam no P2.

**Causa.** O fecho do P2 testa o contato das duas almofadas. A almofada fica atrás do punho e não alcança a caixa com
a mão plana; o alvo das palmas é a distância do punho, e não a da almofada; e nenhum termo do P2 paga o contato.

A tarefa tem duas partes, em commits separados.

### 14.1 T11a: a almofada na profundidade da mão e o alvo saído da geometria

**O que a T11a conserta e o que não conserta.** Ela conserta a superfície que o sensor do fecho lê: a almofada passa
a alcançar a caixa. Ela não conserta o toque curto: o punho já alcança a caixa hoje e o contato dele dura no máximo
14 passos; o contato segue a fase do relógio de marcha (21 % contra 4 % de contato duplo entre faixas da fase).

**Mudança.**
- `cena.py`: `_PALM_DZ` = 0,037 (era 0,015): a face da almofada da palma fica a 0,045 do eixo, 1,0 cm à frente da
  cápsula do punho. A almofada do dorso fica onde está: constante nova `_DORSO_DZ` = 0,015, usada só na chamada do
  `back_pad` em `add_pads_de_palma`. `_PAD_HALF` e `_PALM_X` não mudam (a massa e a posição ao longo da mão ficam).
- `knobs.Alvos`: `palma_face` = 0,045 (era 0,035): a distância do site à face com a almofada encostada. Knob novo
  `folga_p1` = 0,02 (m): a folga da almofada à face no P0 e no P1.
- `alvos.palmas_alvo_caixa`: `y` = `a + palma_face + folga_p1` no P0 e no P1 (ao lado da face, sem tocar);
  `a + palma_face` no P2 (encostada); `a + palma_face + b3_afasta` no B3. O `x` e o `z` não mudam.
- `comando.py`: só o comentário da linha do `afasta` (o B3 mede de `a + palma_face`; o limiar fica 10 mm mais longe
  do centro da caixa que antes).
- `specs/g1-v3.md`, §8.2: as linhas do P0, do P1 e do P2 da tabela, com a nota "18/11, T11a".
- Nenhuma recompensa, portão, fecho ou tempo muda. A `q_palmas` continua a medir do alvo do P2, que agora é a mão
  encostada. O formato das observações e o do checkpoint não mudam, e o ator não lê o alvo das palmas. Os canais do
  crítico e as recompensas que leem `referencia_do_alvo` mudam de valor (as palmas da referência).
- Contagem de funções de `g1_v3/*.py`: 477 → 477.

**Margem e valor da face (decisão do dono, 18/11: 0,045).**
- Com a face a 0,045, a margem sobre o punho é 10 mm. Pela geometria, o punho volta a tocar antes da almofada com a
  base da mão inclinada para a caixa acima de ~4,7° (caixa de meia aresta 0,13) a ~9,8° (0,07), ou com a flexão do
  `wrist_yaw` acima de ~10,3°. Uma guinada da caixa vale +θ para uma mão e −θ para a outra.
- A primeira versão da T11a usou 0,040: margem de 5 mm, com os limites em ~2,4°, ~5,1° e ~5,1°. O dono trocou para
  0,045 antes de qualquer coleta ou treino com 0,040.
- Os dedos da malha vão de 0,039 a 0,045: a face da almofada fica na ponta deles. A folga da almofada ao quadril na
  pose de pé, medida por cinemática, é 5,6 cm com 0,045 (6,0 cm com 0,040; 7,1 cm antes da T11a).

**Alternativas avaliadas.**
- Almofada a 0,040: acima; trocada por 0,045 pela margem.
- Almofada mais perto dos dedos (x de 0,14 a 0,18): muda o braço de alavanca do `wrist_yaw`, que limita o aperto
  (40 a 62 N por palma); fora desta tarefa.
- Reduzir ou excluir a cápsula do punho: o punho real tem de 2,4 a 3,3 cm de raio na malha; tirar a peça deixaria o
  punho atravessar a caixa. Rejeitada.
- Restaurar a cápsula da mão do fabricante (`*_hand_collision`): ela não separa a palma do dorso, o motivo da
  almofada (`cena.add_pads_de_palma`). Rejeitada nesta tarefa.
- Fechar o P2 pela força nas faces, de qualquer peça: o robô fecharia só com o punho. Rejeitada pela regra do dono
  (a mão tem de tocar).

**Testes.**
- `g1_v3/testes/test_alvos.py`: com `a` = 0,10, P0 = (−0,15; ±0,165; 0), P1 = (0; ±0,165; 0),
  P2 = (0; ±0,145; 0), B3 = (0; ±0,195; 0). O mesmo teste confere `folga_p1` < `tol_palma`: a política que vai direto
  ao contato ainda fecha o P1.
- `g1_v3/testes/test_alvos.py`, teste de geometria: `Alvos.palma_face` = `cena._PALM_DZ + cena._PAD_HALF[1]`, e esse
  valor é maior que 0,035, o raio da cápsula `left_wrist_collision` do `g1.xml`.
- Os testes que leem `A.palma_face` (`test_comando_e.py`, `test_recompensas_f.py`) seguem pela fórmula.
- O teste de geometria monta o robô com `add_pads_de_palma`, compila o modelo e mede, com o punho reto, o alcance
  de cada peça ao longo da normal da palma a partir do sítio: o da almofada é `palma_face` e passa do alcance da
  cápsula do punho em mais de 5 mm, nas duas mãos; a cápsula `*_hand_collision` não existe no modelo. O teste é
  geométrico: ele não roda a física do contato.
- `g1_v3/coleta.py`: colunas `F_pad_E` e `F_pad_D`, a força (N) do sensor de cada almofada com a caixa. A força na
  face (`F_esq`, `F_dir`) menos a da almofada dá a parte do punho e do braço.

**Efeitos fora do PEGAR.**
- CARREGAR e BOTAR usam as mesmas mãos. Com a caixa presa, cada site fica 1,0 cm mais longe do centro da caixa. O
  detector de "PEGAR concluído" tem 5 cm de tolerância nas palmas.
- As tabelas de IK (`g1_limpo/ik/`, `FOLGA_PALMA` = 0,035) não mudam. A pose de pé com caixa delas põe o site 10 mm
  mais perto da caixa que a almofada encostada: seguir essa pose aperta a caixa 10 mm por lado.
- Autocolisão: a almofada saiu 2,2 cm para o lado da coxa, e a autocolisão entra no `contato_fora` em toda tarefa.
  Medido por cinemática (18/11): na pose de pé a folga da almofada ao quadril cai de 7,1 para 5,6 cm, sem toque. A
  marcha com balanço de braço não tem medida.
- A cena exportada do `pilota_v3` precisa de nova exportação (`python -m g1_v3.exporta_cena --saida ...`). Uma cena
  exportada antes da T11a roda sem erro, com a almofada velha e o alvo novo.

**O que olhar, SEM treino.** Coleta do model_10850 com a cena nova: `python -m g1_v3.coleta`, 256 envs × 3000 passos,
`--amostra`, a mesma forma da base. Os valores de hoje vêm da coleta do model_10850 com a cena velha.

| Medida (passos com `alvo` = 2 no PEGAR, salvo indicação) | Hoje | Passa | Reprova |
|---|---|---|---|
| `cont_E` e `cont_D` ao mesmo tempo | 9,2 % | ≥ 20 % | < 15 % |
| Sequências de contato duplo: p50 e máximo | 1 e 9 passos | registrar | — |
| Invariante `desloc·incl·maos` em 1 | 7,4 % | registrar | — |
| `p_desloc` igual a 1 | 81 % | ≥ 60 % | < 50 % |
| Inclinação da caixa, p90 | 4,1° | ≤ 8° | > 15° |
| Velocidade da caixa, p50 | 0,10 m/s | ≤ 0,15 | > 0,25 |
| Perdas da caixa por episódio da fatia B | 10 em 378 | ≤ 20 em 378 | > 40 em 378 |
| Fechos do P2 por visita | 0 em 232 | registrar | — |
| `contato_fora` no ANDAR, passos com custo | 0,37 % | ≤ 0,8 % | > 1,5 % |
| Quedas no ANDAR | 10 | ≤ 20 | > 30 |

- Entre "passa" e "reprova" no contato duplo (15 a 20 %): repetir a coleta uma vez e usar a média das duas.
- Contato duplo abaixo de 15 %: NÃO concluir que a geometria não era a causa. Ler antes qual peça toca (a força do
  sensor de cada almofada contra a força na face) e o `site_*_y` com e sem contato. Os contatos de hoje são de quina,
  com a mão girada.
- Uma linha da caixa ou do ANDAR em "reprova": a T11b não entra; a almofada empurra a caixa ou bate no corpo, e o
  valor da face volta à mesa.
- Pendente, pede o "implementa" do dono: duas colunas na coleta com a força do sensor de cada almofada.

### 14.2 T11b: as conquistas pagam também no P2

**Regra da cadeia.** Em toda etapa, a variável que o fecho testa tem um termo de recompensa dentro da etapa, e os
dois leem a mesma régua.

**Mudança.**
- `recompensas.progresso`: a máscara das conquistas passa a incluir o P2, pela expressão exata
  `mantem = ~sem_caixa & ~b4_fechado & (fecho_p4 | _em(idx, _CAIXA_ALVO) | (idx == alvos.P2))`. A tupla
  `_CAIXA_ALVO` NÃO muda: ela também decide a distância do alvo, a `sacudida`, o `aperto` e uma métrica.
- No P2, a `q_pega` entra multiplicada pela fração das almofadas em contato (`_contato_palmas(env)`, a régua do
  fecho): o aperto só com o punho não recebe a conquista. Do P3 em diante a `q_pega` segue como hoje.
- A `q_palmas` soma no P2 sem fator. Ela mede a distância do site ao alvo do P2 (a mão encostada).
- Docstrings e comentários que mudam: `progresso`, `_phi_env` e o knob `renda_conquista`.
- Nenhum termo, portão ou knob novo. `renda_conquista` segue em 2,0. O `aperto` (custo) continua sem valer no P2.
- Contagem de funções: 477 → 477.

**Números.**
- `q_palmas` no P2: 1,00 com a mão no alvo e 0,88 no alvo do P1 (2 cm fora). Ela soma de 1,75 a 2,0/s quase
  constantes no P2; nos últimos 5 mm ela vale ~0,06/s. Ela não é o gradiente do contato.
- `q_pega` no P2, caixa de 1 kg (`F_pega` = 20,4 N): com as duas almofadas e 16,8 N na menor força, paga
  `2·tanh(16,8/20,4)` = 1,35/s; com uma almofada, 0,68/s; só com o punho, 0. Hoje paga 0 nos três casos.
- Regra de 09/11 (todo fecho sobe a renda), no mesmo estado físico dos dois lados do fecho:
  P1 → P2: a renda da etapa vai de 6·h para 12, e as conquistas só somam depois; o ganho é de 6 ou mais menos a
  queda do Φ, no máximo 3: sobe ao menos 3/s.
  P2 → P3: a renda da etapa vai de 12·h para 18; as conquistas do P3 são maiores ou iguais às do P2 (a `q_pega`
  perde o fator das almofadas); o Φ cai no máximo 3: sobe ao menos 3/s, com qualquer valor dos portões.
- O limite `renda_conquista` < 2,3 (fecho B2 → B3) não muda.

**Testes** (`g1_v3/testes/test_recompensas_e.py`).
- `test_conquistas_so_somam_onde_a_pega_tem_de_continuar`: com `q_palmas` = 0,3 e `q_pega` = 0,9 (valores diferentes,
  para o teste distinguir qual conquista recebe o fator), o P2 paga 14,4 com as duas almofadas, 13,5 com uma e 12,6
  com nenhuma; o P1 e o B3 seguem sem as conquistas; o P3 sem almofada paga a pega inteira (20,4); o P2 sem a caixa
  (perda, ou CARREGAR aberto depois de um PEGAR que parou no P2) paga 0.
- `test_todo_fecho_sobe_a_renda_com_as_conquistas`: entram os fechos P1 → P2 e P2 → P3, no mesmo estado dos dois
  lados e com os portões em 0 e em 1.

**O que o código permite (revisão independente de 18/11, sem achado que bloqueie).**
- A força no P2 não tem custo: o `aperto` e a `sacudida` só valem de P3 a B2. A parcela da pega cresce até ~2·`F_pega`
  (~41 N com 1 kg) e fica plana depois; nada paga nem cobra força acima disso.
- A `q_pega` lê a força de qualquer peça nas faces laterais, e o fator lê o toque da almofada em qualquer face da
  caixa. Uma almofada no topo com o punho na face lateral recebe a parcela. O fecho do P2 lê a mesma régua.
- As conquistas do P2 não passam pelos portões: com a caixa deslocada ou inclinada o Φ cai a 0 e o fecho fica
  bloqueado, e as conquistas seguem pagando até 4/s. A diferença entre o estado certo e o deslocado continua 3·Φ.

**O que olhar, COM treino.** Retomada do model_10850 com a T11a e a T11b; coleta de 256 × 3000 ~500 its depois.

| Medida | Hoje | Passa | Reprova |
|---|---|---|---|
| `cont_E` e `cont_D` no P2 | 9,2 % | ≥ 50 % | < 30 % |
| Fechos do P2 por visita | 0 % | ≥ 30 % | < 10 % |
| `s_ANDAR_andar` 400 its depois da retomada | — | ≥ o valor de antes − 0,05 | abaixo disso |

- Entre "passa" e "reprova": mais 500 its e nova coleta.
- Reprova com a modulação do contato pela fase igual à de hoje: a causa do toque curto é o ritmo dos braços; o passo
  seguinte é o teste da fase.

**Ordem.** T11a → push → coleta sem treino no pod → leitura → T11b → push → retomada com `RETOMA = 1`.

**Aberto (não entra na T11).** A posição da almofada ao longo da mão. O relógio do fecho que zera e o invariante de
deslocamento que exige o valor exato 1. A fase do relógio de marcha no PEGAR. A recompensa da pega em uma etapa só
(`design_draft.md` do projeto, opção 3), que só entra se o P3 ou o P4 pararem depois da T11.

## 15. Tarefa T12: o relógio da marcha congela com comando zero, e a mão colide pelo casco da malha

Enunciado: `specs/g1-v3-comportamento.md` §4.1 (P2: as duas palmas nas faces) e §4.3 (de P2 a B2, a caixa nivelada e
segura). A T12 junta duas mudanças por decisão do dono (19/11), numa retomada do `model_12500`.

### 15.1 Causa medida

**Relógio.** Coleta do `model_12500` (amostrada, 256 × 3000, `medidas-p2-model10600.md` do projeto): o P2 fecha em
84 % das visitas e o P3 em 1 de 190. A caixa fica 3 cm ou mais no ar só nas fases 0,4 a 0,7 do relógio da marcha
(11 a 39 % do tempo; abaixo de 2 % nas outras); 52 % da energia da altura da caixa no P3 fica entre 1,0 e 1,5 Hz
(relógio: 1,25 Hz). A janela dura ~0,24 s, menos que o `tempo_p3` de 0,30 s. O comando no P3 é zero, e o relógio
corre pelo tempo do episódio com qualquer comando (`comando.py:787`); o ator vê a fase (`observacoes.py:311`).
Teste no `registra_juntas` (flag `--fase-parada`, 19/11): com a fase parada, a energia em 1,0–1,5 Hz cai de 83 % para
1 % e a base para (0,043 → 0,002 m/s). A Unitree (`unitree_rl_lab`, `gait_phase` e `feet_gait`) usa a mesma forma:
fase pelo tempo, sem ler o comando; ela só treina locomoção.

**Mão.** A almofada da palma fica a 4,5 cm da normal da palma, 3,2 a 3,5 cm à frente da palma visível (1,0 a 1,3 cm),
sobre a palma e não sobre os dedos, e é invisível no viewer (grupo 3). O dono viu a caixa se mover sem a mão tocar.
O casco convexo da malha da mão segue a mão real: a frente da mão vai de 1,0 cm na palma a 4,5 cm na ponta dos dedos
curvados, e o casco cobre esse vão. Medida local (19/11, script de teste que troca a geometria e mantém os nomes): o
`mujoco_warp` aceita o casco; em CPU, 34,0 → 37,2 s por iteração (64 envs, ruído de ±6 s; não vale para a GPU); no
`registra_juntas` com o `model_12500`, as duas palmas têm força em 78 % do PEGAR (fase normal) e 93 % (fase parada).

### 15.2 Relógio congelado (`comando.TwistV3`)

- Acumulador próprio da fase, `self._fase_t` (s), criado ao lado do `_agora` e zerado no `reset`. O `_agora` não muda:
  ele conta o tempo da âncora e da abertura (`comando.py:669`, `:759`, `:778`).
- A cada passo: `self._fase_t += dt · on`, com o `on` que já existe (`‖cmd‖ > limiar_marcha`, `comando.py:786`).
  `env.v3_fase_marcha = remainder(self._fase_t / periodo_marcha, 1)`.
- Com o comando zero a fase fica no valor em que estava; quando o comando volta, ela continua dali.
- O `passo` (`recompensas.py:597`) e o bloco `marcha_on` da observação não mudam.

**Driver** (`pilota_v3.py:818-819`, `registra_juntas_v3.py:405-409`; hoje o driver soma `passos` todo passo): `passos` só soma quando o twist do passo
anterior passa de `limiar_marcha` (o env soma com o comando que a política viu no passo t, e a fase entra na
observação de t+1). O flag `--fase-parada` sai do `registra_juntas`: o comportamento vira o padrão. A coluna
`relogio` do CSV fica.

### 15.3 Casco da mão (`cena.add_pads_de_palma`)

- Os quatro geoms mantêm o nome (`*_palm_pad`, `*_hand_back_pad`), o grupo (`_GRUPO_PAD`), o condim, o atrito e o
  rgba. Assim o `CollisionCfg` (`cena.py:205-220`), os sensores, o DR de atrito, a coleta e o `registra_juntas` não
  mudam de nome.
- A forma passa de caixa a malha. Os vértices vêm das malhas visuais do corpo da mão (`*_rubber_hand`, grupo 2), no
  frame do corpo, lidos de uma cópia compilada do spec em qpos0. O corte é o plano do sítio `*_palm` perpendicular à
  normal da palma (o y do corpo com o sinal do lado): vértices com projeção ≥ 0 formam a palma, ≤ 0 o dorso.
- Cada metade vira uma malha (`spec.add_mesh`) com os vértices do casco (`scipy.spatial.ConvexHull`) e
  `maxhullvert` 64 (constante em `cena.py`).
- Saem `_PAD_HALF`, `_PALM_DZ`, `_DORSO_DZ`, `_PALM_X` e a função `_add_pad`.
- `knobs.Alvos.palma_face` fica em 0,045: é o alcance do casco da palma na normal da palma em qpos0 (a ponta dos
  dedos). O comentário passa a dizer isso.

**Coleta** (`coleta.py:76-79` e `:93-96`): a folga da almofada à face passa a usar os vértices da malha do geom
(`mesh_vert` do modelo) no lugar dos 8 cantos do `_PAD_HALF`. As colunas não mudam.

**Contagem de funções:** `cena.py` −1 (`_add_pad`); nenhuma função nova no resto.

### 15.4 Testes

- `test_marcha_relogio.py`: `test_fase_e_o_tempo_sobre_08_com_comando_ligado_e_desligado` passa a exigir a fase parada com o comando zero
  (50 passos com cmd 0 → fase 0) e a volta a cada 0,8 s com comando; `test_sem_recomeco_quando_o_comando_passa_do_limiar`
  passa a 10 passos sem comando e 1 com → fase `1 · dt / 0,8` (continua de onde parou); novo caso: comando, zero e
  comando de novo → a fase soma só os passos com comando.
- `test_alvos.py`, `test_palma_face_e_a_face_da_almofada_a_frente_do_punho`: o alcance do casco da palma na normal,
  medido do sítio no modelo compilado, vale `palma_face` com tolerância de 1 mm; passa o raio da cápsula
  `*_wrist_collision` por mais de 5 mm; os geoms são `mjGEOM_MESH`; o casco do dorso fica todo do outro lado do plano.
  Sai a igualdade com `_PALM_DZ + _PAD_HALF[1]`.
- Driver: um teste que confere a contagem dos `passos` com o twist do passo anterior (comando, zero, comando).
- Paridade do `exporta_cena` (`--paridade`): roda e passa com o casco e com o relógio novo.
- Suíte inteira no fim do lote; treino local de 2 iterações em CPU sem erro nem NaN.

### 15.5 O que olhar, COM treino

Retomada do `model_12500` (`RETOMA = 1`); coleta de 256 × 3000 com `--amostra` ~500 its depois.

| Medida | Hoje (12500) | Passa | Reprova |
|---|---|---|---|
| Tempo por iteração no pod | 9,13 s | ≤ 11,4 s (+25 %) | acima: `maxhullvert` 32 |
| Energia da altura da caixa no P3 em 1,0–1,5 Hz | 52 % | ≤ 20 % | ≥ 40 % |
| Fechos do P3 por visita | 0,5 % | ≥ 30 % | < 5 % |
| Contato duplo no P2 | 46 % | ≥ 30 % | < 20 % |
| `s_ANDAR_andar` 400 its depois | 0,731 | ≥ 0,664 | abaixo |

- Previsão que a medida pode refutar: com o relógio parado no PEGAR, a subida da caixa deixa de seguir o passo e passa
  a durar mais que 0,30 s. Espero uma queda inicial do P3: o teste do `registra_juntas` mostrou que a política atual
  só levanta a caixa no pulso da fase.
- Reprova na energia com o relógio parado: o ritmo vem do histórico de ações, e não da fase; o congelamento sai.
- Reprova no `s_ANDAR_andar`: treino do zero com a T12.

**Ordem.** Spec → "implementa" do dono → lote → revisão → push com o OK → retomada.

### 15.6 O casco sai (20/11)

Retomada da T12 no pod (pasta `2026-11-19_19-25-45_unitree2`, its 12500–13500): o P2 deixou de fechar (antes, 84 % das
visitas) e o P3 não abriu. Coleta local em CPU do `model_13500` (32 envs × 1200 passos, amostrada): o sensor da palma
funciona com o casco, mas o contato duplo no P2 cai a 9,5 % (era 46 %), as sequências não passam de 0,08 s e a força
da palma tem p50 0 (era 9–11 N). No `registra_juntas` (sem DR, ação média) o mesmo model segura com as duas palmas,
com 1–3 N. O casco encosta pela curva da ponta dos dedos, um contato em linha que o ruído e a DR desfazem.

Decisão do dono: a almofada volta à da T11a (`cena.py`, `coleta.py`, `knobs.py` e `test_alvos.py` como no `bb5afa3`);
o relógio congelado (§15.2) fica. A almofada plana na face do casco (plano de 14,1°) pede a mão inclinada e o
`palma_face` em ~0,022; fica para uma mudança separada.
