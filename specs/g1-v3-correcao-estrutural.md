# g1_v3: correção estrutural do treino (dono, 13/11)

Esta spec é o contrato do lote de 13/11. Ela registra as edições exatas. O motivo e os números de cada mudança estão no
desenho `/home/joaobornelli/workbench/projects/g1-treino-v3/desenho-correcao-estrutural.md` (fora do repo; leia a seção
da sua tarefa antes de editar).

## 1. Decisões do dono (13/11)

1. O João confirmou o desenho com M1 a M7 e os ajustes A1 a A9, e pediu a implementação com sonnet.
2. Critérios do desenho: toda parte do treino tem gradiente que puxa para o sucesso; progredir paga mais que ficar parado.
3. A próxima corrida é do zero. Checkpoint antigo não precisa carregar.

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
  para ele passar. Teste de função apagada sai junto com ela.
- Todo teste novo falha contra o código de antes da tarefa. Diga no relato como você conferiu (por leitura do caso).
- Não mude nada fora do que a tarefa lista. Não refatore além do que a tarefa pede. Todo número novo mora em
  `g1_v3/knobs.py`, com um comentário na linha: "13/11, <item>: <motivo curto>".
- Antes de mexer num termo de recompensa, knob ou terminação, faça `grep` do nome em `vault/_sistema/por-termo.md` e leia
  a linha. Se ela citar uma armadilha que esta spec não trata, pare e relate.
- Regra do repo (`CLAUDE.md`): modifique antes de adicionar. Função nova só onde a tarefa permite.
- Ordem do passo no mjlab (vale para M4 e M5): física → terminações → recompensas → reset → comandos (`compute` da
  cadeia e do `twist`) → eventos de intervalo → observações. A recompensa do passo t lê os buffers que os comandos
  gravaram no passo t − 1.
- Relate no fim: arquivos mudados; contagem de funções do `g1_v3` sem testes antes → depois
  (`grep -c "def " g1_v3/*.py`, somando os arquivos fora de `testes/`); comando de teste; as últimas 15 linhas da saída;
  hash do commit; o que você não conseguiu cumprir.

Linha de base: commit `3a01bc1`, 473 funções no `g1_v3` sem testes.

## 3. Tarefa T1: ajustes pequenos (M2, A1, A2, A4, A5, A6, A7, A8)

Arquivos: `g1_v3/knobs.py`, `g1_v3/tarefa.py`, `g1_v3/recompensas.py`, `g1_v3/comando.py`, `g1_v3/balanceador.py`,
`g1_v3/testes/`.

1. M2 (par de ruído que aprendeu):
   - `Recompensa.taxa_acao`: −0.10 → −0.01.
   - `tarefa.rl_cfg`: `cfg.algorithm.entropy_coef = 0.001`. Reescreva o parágrafo do docstring sobre o par: 0,001 com
     −0,01 é o par da corrida que formou marcha (12/11); o par −0,10/0,01 (unico3, 13/11) deixou o saldo por segundo de
     vida negativo, e a corrida não saiu do nível 0.
2. A1 e A6 (máscaras dos portões), em `recompensas.py`: troque `_P2_B2` por duas máscaras:
   `_INCL = _ix("P2", "P3", "P4", "B1", "B2", "B3")` e `_MAOS = _ix("P3", "P4", "B1", "B2")`.
   Em `portoes_do_alvo`: `incl` vale a rampa só em `_INCL`; `maos` vale o argumento só em `_MAOS`; fora, 1.
   Atualize o docstring. Efeito esperado: o Φ do P2 deixa de zerar sem força (A1); o Φ e o fecho do B3 passam a
   exigir a caixa nivelada (A6, spec de comportamento §4.5). O fecho do P2 continua exigindo as duas palmas em contato
   (`ok_agora_pegar`, `p2`).
3. A2: `_JANELA_PEGA = _ix("P1", "P2", "P3", "P4", "B1", "B2", "B3")` (o alvo do P1 é a palma encostada). Atualize o
   docstring de `contato_fora` ("P1 a B3 e o CARREGAR").
4. A4: em `comando.ok_agora_botar`, o `b2` passa a exigir também `v_caixa.norm(dim=-1) < k.v_parada_b3`. O ponto de
   apoio (`v3_ponto_apoio`, gravado no fecho do B2) sai então com a caixa parada.
5. A5:
   - `Alvos.delta_b2b4`: 0.02 → 0.06.
   - Bug em `CadeiaCmd._alvos` (`comando.py:388-398`): o invariante recebe o deslocamento do apoio no lugar do
     deslocamento da pega e zero no lugar do deslocamento do apoio, então o fecho do B3 e do B4 ignora o arrasto. Passe
     os dois certos a `portoes_do_alvo`: `desloc_pega` = ‖caixa − `v3_caixa_pega_w`‖ (xy) fora do BOTAR e sem perda,
     zero no resto; `desloc_apoio` = ‖caixa − `v3_ponto_apoio`‖ (xy) no BOTAR sem perda, zero no resto. Corrija o
     comentário F9 S2/P1 do bloco.
6. A7: em `recompensas._com_combinado`, a massa da caixa entra só com `v3_pegou & ~v3_perdeu` e o índice do alvo entre
   P3 e B3, inclusive (o CARREGAR fica com o índice no P4). Atualize o docstring.
7. A8: em `Curriculo._abre`, grave `self._ult[t] = self.it` quando a primeira célula da tarefa abre: PEGAR e CARREGAR
   na abertura da fatia B; BOTAR na abertura da fatia C. Corrija o comentário da criação de `_ult`: o ANDAR conta desde
   a iteração 0; as tarefas que abrem depois contam da abertura (13/11, A8; antes, a primeira medida da célula, com 1
   ou 2 episódios, subia o nível sem carência).

Testes:
- M2: o knob e o `entropy_coef` do `rl_cfg()`.
- A1: `portoes_do_alvo` no P2 com `maos` 0 dá `maos` 1; no P3 com `maos` 0 dá 0.
- A6: `portoes_do_alvo` no B3 com a inclinação acima de limite + delta dá `incl` 0; no B4, 1.
- A2: o P1 está na janela da pega (o teste lê a máscara ou o termo, como os testes do arquivo já fazem).
- A4: `ok_agora_botar` no B2 com distância e apoio cumpridos: ‖v_caixa‖ 0,10 dá False; 0,0 dá True.
- A5: `portoes_do_alvo` no B3 com `desloc_apoio` 0,05 dá `desloc` 0,5; com 0,10 dá 0. O conserto da chamada em `_alvos`
  vai para a revisão se não houver teste de cadeia que o alcance; diga no relato.
- A7: teste se o arquivo de testes já tiver um env falso que alcance `_com_combinado`; senão, diga no relato.
- A8: `Curriculo` puro: abra a fatia B numa iteração i > `iters_min_nivel` e registre só sucessos do PEGAR em seguida;
  o nível do PEGAR não sobe antes de i + `iters_min_nivel`.

Commit: `feat(v3): par de ruído, máscaras dos portões, apoio parado no B2 e carência das tarefas novas`

## 4. Tarefa T2: o trecho acaba no fecho (M6), as mãos no B4 (A9) e o apoio no B2 (A3)

Arquivos: `g1_v3/comando.py`, `g1_v3/alvos.py`, `g1_v3/recompensas.py`, `g1_v3/knobs.py`, `g1_v3/smoke.py`,
`g1_v3/testes/`.

1. M6:
   - `CadeiaCmd._detector`: o latch `v3_aviso` passa a receber o fecho do alvo do trecho, só nos trechos por aviso
     (`self.tab.por_detector[env.v3_trecho]`): no PEGAR, `v3_p4_fechado`; no BOTAR, `v3_b4_fechado`. Nada mais. Se o
     método ficar com até três linhas, ponha o corpo no `compute` e apague o método.
   - Apague `alvos.detecta_fim`, o buffer `self.rel_fim` (criação e reset) e os testes do detector (`test_alvos.py`).
   - `knobs.Detector`: apague `estavel_s` e `juntas_paradas`. A classe fica com as três tolerâncias, que o B4 usa.
     Tire o `D` da linha de import de `comando.py` se ele ficar sem uso.
   - O atraso sorteado continua (`atraso_pegar_s`, `atraso_botar_s`): o trecho avança com o aviso e o relógio do aviso
     ≥ atraso, como hoje (`avanca_trecho` não muda).
   - Atualize os docstrings e comentários que descrevem o detector (`TabelaTrechos.por_detector`, `avanca_trecho`,
     `_abre_p0`, a classe `CadeiaCmd`, o comentário S5b) e o texto do `smoke.py` que cita `detecta_fim`.
   - O canal de alvo do ator no BOTAR continua zerado pelo latch (`observacoes.py:193`): não mude.
2. A9 (mãos ao lado do tronco no B4, spec de comportamento §4.5):
   - Knob novo `Alvos.tol_palmas_b4: float = Detector.tol_palma` (0,05 m), logo depois de `tol_tronco_b4_graus`.
   - `recompensas.distancia_do_alvo`: argumento novo `dpalmas_b4` (E,), a maior das duas distâncias
     ‖palma_b − `ref.palmas_de_pe_b`‖ no frame da base. No B4:
     d = √((Δpelve/tol_pelve_b4)² + (Δtronco/tol_tronco_b4_graus)² + (dpalmas_b4/tol_palmas_b4)²).
   - `comando.ok_agora_botar`: argumento novo `dpalmas` (E,); o `b4` exige também `dpalmas <= k.tol_palmas_b4`.
     O B4 da caixa perdida usa o mesmo núcleo.
   - Todos os chamadores calculam `dpalmas`: `_phi_env` (recompensas), `CadeiaCmd._alvos` e o σ do alvo
     (`CadeiaCmd._dist`/`_sigma`). Palmas no frame da base: `quat_apply_inverse(base_q, palmas_w − base_p)`, a conta
     que o `_detector` de hoje faz.
3. A3 (apoio do B2 no Φ):
   - Knob novo `Alvos.escala_apoio_b2: float = 0.10` (m).
   - `recompensas.distancia_do_alvo`: argumento novo `deficit_apoio` (E,) em [0, 1] =
     relu(1 − apoio_N / (apoio_b2_mg · peso_N)). No B2: d = hypot(d_caixa, escala_apoio_b2 · deficit_apoio).
     Os outros alvos não mudam.
   - Os chamadores calculam o déficit com a mesma medida de `apoio_N` e `peso_N` do `CadeiaCmd._mede`. Em `_phi_env`,
     leia os mesmos sensores, sem função nova.

Testes:
- M6: com o `avanca_trecho` (puro), um trecho por aviso nunca avança sem o latch, e avança depois do atraso com ele.
  Se o arquivo de testes da cadeia tiver env que alcance o `compute`, teste também: o PEGAR não sai sem
  `v3_p4_fechado`, e sai com ele depois do atraso.
- A9: `ok_agora_botar` no B4 com pelve e tronco dentro: `dpalmas` 0,06 dá False; 0,04 dá True. `distancia_do_alvo` no
  B4 cresce com `dpalmas_b4`.
- A3: `distancia_do_alvo` no B2: déficit 1 dá hypot(d_caixa; 0,10); déficit 0 dá d_caixa.

Commit: `feat(v3): o trecho acaba no fecho do P4 e do B4; mãos no B4; apoio no Φ do B2`

## 5. Tarefa T3: a renda da etapa cai sem progresso (M5)

Arquivos: `g1_v3/recompensas.py`, `g1_v3/comando.py`, `g1_v3/curriculo.py`, `g1_v3/knobs.py`, `g1_v3/testes/`, e
qualquer chamador de `n_progresso` (faça `grep` no repo inteiro, `parte0/` incluso).

1. Knobs novos em `Recompensa`, logo depois de `renda_etapa`: `tau_renda: float = 3.0` (s), `recarga_renda: float = 5.0`,
   `qualidade_larga: tuple = (0.05, 0.25)` (m: `x_lim` e `delta` da `potencial.rampa`).
2. Buffers novos em `curriculo.garante_v3`: `v3_h` (E,) float32 com valor inicial 1.0; `v3_phi_melhor` (E,) float32 com
   0.0.
3. `n_progresso(phi, etapa, w_phi, w_etapa, h)`: `w_etapa·etapa·h + w_phi·phi`. Docstring com a M5.
4. Núcleo novo (a única função nova da T3), na seção dos núcleos de `recompensas.py`:

   ```python
   def n_pressa(h, ganho, dt, tau, recarga) -> torch.Tensor:
       """h·exp(−dt/τ) + recarga·ganho, com teto 1 (13/11, M5). `ganho` = relu(Φ − melhor Φ da etapa)."""
       return torch.clamp(h * math.exp(-dt / tau) + recarga * ganho, max=1.0)
   ```

5. `progresso(env, tarefas)`: a etapa e o Φ pago ficam como hoje. A renda passa a `n_progresso(Φ pago, etapa,
   renda_phi, renda_etapa, h_ef)`, com:
   - Etapa com alvo e Φ pago (P1 a P4, B1 a B4): `ganho = relu(Φ − v3_phi_melhor)`; `v3_h = n_pressa(v3_h, ganho,
     step_dt, tau_renda, recarga_renda)`; `v3_phi_melhor = max(v3_phi_melhor, Φ)`; `h_ef = v3_h`. No P0 a etapa vale 0,
     e o h não importa.
   - CARREGAR sem perda: `v3_h += (step_dt/tau_renda)·(sinal − v3_h)`; `h_ef = v3_h`. O `sinal`, em [0, 1]:
     - com um canal comandado (|c| ≥ `limiar_canal`; linear c = ‖cmd_xy‖, angular c = |wz|): a média, nos canais
       comandados, de `clamp(rastreio, 0, 1)`, com o rastreio de `_rastreio(env, linear)` (sem a qualidade);
     - sem canal comandado: `(1 + r_lin)·(1 + r_ang)`, que vale 1 parado (no ramo sem comando, r = exp(−v²/σ₀²) − 1);
     - vezes `potencial.rampa(d, *qualidade_larga)`, com `d` = a distância da caixa ao alvo de transporte que `_qualidade`
       já calcula. Faça `_qualidade` devolver também o `d` e ajuste o chamador; não duplique a conta.
   - Depois do fecho do P4 (etapa 5 fora do CARREGAR) e do B4 (etapa 9): `h_ef = 1`. Com a M6, esses trechos duram só o
     atraso e o ANDAR final, e o robô não muda a duração deles.
   - Caixa perdida e sem alvo: renda 0, como hoje.
   - O `v3_phi` continua gravado como hoje (métricas).
6. Recomeço do h em `comando.py`: `v3_h = 1` e `v3_phi_melhor = 0` no reset do episódio (`CadeiaCmd._resample`), na troca
   de alvo (`troca` em `_alvos`), na abertura do P0 (`_abre_p0`), na abertura do B1 (`_abre_botar`), na abertura de
   todo trecho (`_abre_trecho`) e na perda da caixa (`_perda`).

Testes:
- `n_progresso` com h: 6·n·h + 3·Φ.
- `n_pressa`: com ganho 0, h cai por exp(−dt/τ) a cada passo; 3 s sem ganho levam h de 1 a e⁻¹; ganho 0,2 com
  recarga 5 devolve h a 1; teto 1.
- Salto do fecho: para n de 1 a 8, h e Φ em [0, 1], 6·(n + 1) − (6·n·h + 3·Φ) ≥ 3.
- O sinal do CARREGAR, se houver env falso que alcance `progresso`; senão, diga no relato.

Commit: `feat(v3): renda da etapa cai sem progresso (M5)`

## 6. Tarefa T4: recompensa de marcha por relógio (M4)

Arquivos: `g1_v3/knobs.py`, `g1_v3/comando.py` (`TwistV3`), `g1_v3/recompensas.py`, `g1_v3/potencial.py`,
`g1_v3/eventos.py`, `g1_v3/curriculo.py`, `g1_v3/testes/`.

1. Knobs:
   - `Recompensa.passo`: 0.5 → 1.3 ("13/11, M4: relógio de marcha do unitree_rl_lab, `feet_gait`; 1,3 por pé").
   - Apague `Recompensa.L_min` e `Recompensa.fator_T` e a função `potencial.L_alvo` (sem outro uso; confira com `grep`).
     `recompensas.comprimento_do_passo` fica: a `metricas.py` usa.
   - Novos em `Comando`, logo depois de `limiar_canal`: `fase_apoio: float = 0.55`; `cadencia_base: float = 1.4`
     (passos/s); `cadencia_por_v: float = 1.2` (passos/s por m/s); `giro_para_v: float = 0.3` (m/s por rad/s).
2. Buffers novos em `curriculo.garante_v3`: `v3_fase_marcha` (E,) float32 0.0; `v3_marcha_on` (E,) bool False;
   `v3_acerto_marcha` (E,) float32 0.0.
3. Relógio, no `TwistV3._update_metrics` (roda uma vez por passo), com o comando `self.vel_command_b`:

   ```python
   c_lin, c_ang = cmd[:, :2].norm(dim=1), cmd[:, 2].abs()
   on = (c_lin >= k.limiar_canal) | (c_ang >= k.limiar_canal)
   v = torch.where(c_lin >= k.limiar_canal, c_lin, k.giro_para_v * c_ang)
   T = 2.0 / (k.cadencia_base + k.cadencia_por_v * v)  # período: dois passos
   fase = torch.where(on & ~env.v3_marcha_on, torch.zeros_like(c_lin), env.v3_fase_marcha)  # recomeço no limiar
   env.v3_fase_marcha[:] = torch.where(on, torch.remainder(fase + dt / T, 1.0), torch.zeros_like(fase))
   env.v3_marcha_on[:] = on
   ```

   - Métrica nova `acerto_relogio` do `twist` (em `_METRICAS`): a soma de `v3_acerto_marcha` nos passos com `on`, sobre
     o número desses passos, como a `razao_marcha` faz. Zera no `reset`.
   - `TwistV3.reset`: `v3_fase_marcha = 0` e `v3_marcha_on = False` nos `env_ids`.
4. `eventos.empurrao`: `env.v3_fase_marcha[env_ids] = 0.0` (recomeço no empurrão).
5. `recompensas.n_passo` muda de corpo e de assinatura (o nome fica):

   ```python
   def n_passo(contato, fase, on, fase_apoio) -> torch.Tensor:
       """(E,2) 1 onde o contato do pé bate com o relógio (13/11, M4; unitree_rl_lab `feet_gait`). Pé esquerdo na `fase`,
       direito na fase + 0,5; apoio com fase < `fase_apoio`. Com `on` False (comando zero), os dois pés em apoio."""
       fases = torch.stack([fase, torch.remainder(fase + 0.5, 1.0)], dim=-1)
       apoio = (fases < fase_apoio) | ~on.unsqueeze(-1)
       return (contato == apoio).float()
   ```

6. `recompensas.passo(env, tarefas)` muda de corpo:
   - `on` do comando atual (`limiar_canal` nos dois canais), `contato = env.scene["pes_chao"].data.found > 0` (E,2),
     `acerto = n_passo(contato, env.v3_fase_marcha, on, C.fase_apoio)`.
   - Portão de velocidade, com `v = twist.v_media_b` (a mesma média do rastreio): com comando linear,
     `clamp((v_xy · cmd_xy)/c_lin², 0, 1)`; senão, com comando de giro, `clamp(v_z·sinal(wz)/c_ang, 0, 1)`; sem comando,
     1. Denominadores com `clamp(min=_EPS)`.
   - `env.v3_acerto_marcha[:] = acerto.mean(-1)`; a recompensa é `acerto.sum(-1)·portão`, com `_fim(..., paga=True)`.
   - Docstring: forma do `unitree_rl_lab`, relógio parado em apoio duplo com comando zero, portão contra a marcha no
     lugar (spec de comportamento §4.1), período pela velocidade.
7. Apague o que ficar sem uso no `recompensas.py` (`_PES_SITIOS` do arquivo, se for o caso).

Testes:
- `n_passo`: fase 0,2 com `on`: E em apoio e D (0,7) em balanço; contatos (True, False) dão (1, 1); (True, True) dão
  (1, 0). Com `on` False: (True, True) dão (1, 1); (False, True) dão (0, 1).
- Período: v 0,5 m/s dá T = 1,0 s; v 1,0 dá T ≈ 0,769 s; giro 1 rad/s sem comando linear usa v = 0,3.
- Portão: velocidade igual ao comando dá 1; parado dá 0; acima do comando dá 1; sem comando dá 1.
- Relógio do `twist`, se houver env nos testes do comando: recomeça em 0 quando o comando passa do limiar, e o
  empurrão zera a fase. Senão, diga no relato.

Commit: `feat(v3): marcha por relógio com portão de velocidade (M4)`

## 7. Tarefa T5: estado da cadeia no crítico (M7) e queda e perda sem saída (M1)

Arquivos: `g1_v3/observacoes.py`, `g1_v3/runner.py`, `g1_v3/algoritmo.py`, `g1_v3/comando.py`, `g1_v3/smoke.py`,
`g1_v3/testes/`.

1. `observacoes.alvo_macro`: one-hot de 13 posições (`N_ALVO_MACRO = len(knobs.ALVOS) + 4`). Posições 0 a 8: o índice
   de `ALVOS` (P0 a B4); 9: P4 fechado; 10: B4 fechado; 11: perda; 12: nenhum. Prioridade: `v3_perdeu` → 11;
   `v3_b4_fechado` → 10; `v3_p4_fechado` com o índice no P4 → 9; índice < 0 → 12; senão, o índice. Docstring.
2. Termo novo do crítico, `observacoes.privilegiados(env)` (E,6): seno e cosseno de 2π·`v3_fase_marcha`;
   `v3_marcha_on` em float; `v3_h`; lacuna = relu(`v3_phi_melhor` − `v3_phi`.clamp(min=0)); `v3_sigma`. Entra no dict do
   crítico logo antes de `tarefa_critico`. O one-hot da tarefa continua nas 6 últimas colunas. `runner._FIXOS` ganha
   `"privilegiados": 1.0`.
3. `algoritmo.PPOPorTarefa`:
   - Método novo `_grupo(self, obs_critico)` (a única função nova da T5 além do termo do item 2): a tarefa é o argmax
     das 6 últimas colunas; sem `fatia_alvo`, o grupo é a tarefa; com ela, `slot` = argmax da fatia, `n` = largura da
     fatia, e o grupo é 0 para o ANDAR com o slot "nenhum" (`n − 1`) e `1 + tarefa·n + slot` no resto. O
     `compute_returns` passa a usar `_grupo` (sem mudar a trava da soma do one-hot).
   - No `compute_returns`, depois da normalização: guarde `self.v_grupo` (tensor de `1 + 6·n` posições, no device dos
     valores, começa em zeros), com a média de `storage.values` de cada grupo presente no rollout. Grupo ausente fica com
     o valor anterior.
   - `process_env_step(self, obs, rewards, dones, extras)` (assinatura explícita): antes do `super()`, com `fatia_alvo` e
     `v_grupo` presentes, some `self.gamma · clamp(self.v_grupo[g], max=0)` em `rewards` onde
     `ruim = (dones > 0) & ~time_outs`, ou onde `extras["v3_perdeu_agora"]` vale True. O grupo `g` sai de
     `self._grupo(self.transition.observations["critic"])`: as observações do passo, e não as do reset. Sem a chave
     `time_outs` em `extras`, trate como zeros. Mantenha a fixação do normalizador que já existe depois do `super()`.
     Docstring: M1, por que a média do grupo (o V do próprio estado cria ponto fixo r/(1 − γ)) e por que vale também na
     perda da caixa.
4. `comando.CadeiaCmd.compute`: depois de `self._perda(m)`, publique `env.extras["v3_perdeu_agora"] = env.v3_perdeu_agora`.
5. `smoke.py`: as dimensões do crítico (`alvo_macro` 13, `privilegiados` 6) e o que mais citar o one-hot de 10.

Testes:
- `alvo_macro` com um env falso (`types.SimpleNamespace` com os buffers): cada caso da prioridade.
- `_grupo`: ANDAR com "nenhum" dá 0; ANDAR com "B4 fechado" e com "perda" dão grupos próprios, diferentes de 0;
  PEGAR no P4 aberto e no P4 fechado dão grupos diferentes.
- `v_grupo` e o termo do `process_env_step`: com um `PPOPorTarefa` montado só com os atributos que o método lê (como os
  testes do algoritmo já fazem, se fazem), a queda num grupo de V̄ −5 soma γ·(−5); num grupo de V̄ 3, soma 0; o
  `time_out` não soma; o pulso da perda soma. Se não der para montar sem o runner, diga no relato.
- `privilegiados` com env falso: forma (E,6) e os valores.

Commit: `feat(v3): estado da cadeia no crítico e nos grupos; queda e perda sem saída (M7, M1)`

## 8. Tarefa T6: laje alta primeiro, por subníveis (M3)

Arquivos: `g1_v3/knobs.py`, `g1_v3/balanceador.py`, `g1_v3/eventos.py`, `g1_v3/curriculo.py`, `g1_v3/testes/`.

1. Knobs novos em `Niveis`: `subnivel_extra: tuple = (0.20, 0.10, 0.0)` (m acima do topo do nível 0);
   `subnivel_longe: tuple = (0.0, 0.10, 0.20)` (teto do `longe` do destino, m); `subnivel_jitter: tuple = (0.0, 0.0, 1.0)`
   (fração da `faixa_recuo` na base da pega). O último subnível é o nível 0 de hoje.
2. `Curriculo`:
   - Estado novo `self._sub = {"PEGAR": 0, "BOTAR": 0}` e o acessor `subnivel(tarefa)`, que devolve o último subnível
     para as outras tarefas.
   - `_sobe`: no PEGAR e no BOTAR com o nível 0 e o subnível abaixo do último, a mesma condição de subida (células
     abertas, média no portão, carência) sobe o subnível, e não o nível. A subida de subnível grava `_ult` e zera
     `n_medidas` das células da tarefa, como a subida de nível.
   - `nivel_passou(t)` devolve −1 no PEGAR e no BOTAR enquanto o subnível está abaixo do último.
   - `estado()` e `carrega()` levam o subnível (`"sub"` por tarefa); estado sem a chave carrega como o último subnível.
   - O índice do nível não muda: as travas de DR, a `restricao()`, o freio, o `jitter_x_max` e o teto dos sorteios
     continuam lendo o mesmo número.
3. `eventos`:
   - `sorteia_topo_e_massa`: parâmetro novo `extra` (m, padrão 0.0), somado ao topo das fatias B e C. O chamador passa
     `subnivel_extra[s]` com o PEGAR no nível 0, e 0.0 no resto. No nível 0 o topo de B e C já vale o topo do nível 0, então
     o subnível fixa 0,75, 0,65 e 0,55 m.
   - `_base_da_pega`: o jitter de `d` é multiplicado por `subnivel_jitter[s]` com o PEGAR no nível 0, e por 1 no resto.
   - `sorteia_destino`: parâmetro novo do subnível do BOTAR. No nível 0 e abaixo do último subnível: `longe` =
     u·`subnivel_longe[s]` e topo = `botar[0]` + `subnivel_extra[s]`. No resto, como hoje.
4. Log do currículo (`curriculo.v3`): `subnivel_PEGAR` e `subnivel_BOTAR`.

Testes:
- `Curriculo` puro: com sucessos do PEGAR, o subnível sobe 0 → 1 → 2 com a carência entre as subidas, o nível fica 0, e
  o `nivel_passou("PEGAR")` vale −1 até o subnível 2 passar no portão. O mesmo no BOTAR. Ida e volta do `estado`.
- `sorteia_topo_e_massa` com `extra` 0,20 no nível 0: topo 0,75 em B e C, 0 no ANDAR.
- `sorteia_destino`: subnível 0 dá `longe` 0 e topo 0,75; o último subnível dá o sorteio de hoje.

Commit: `feat(v3): laje alta primeiro, por subníveis no nível 0 (M3)`

## 9. Verificação do lote (orquestrador)

1. Suíte completa, uma vez, no fim.
2. Treino local curto (16 envs, 2 iterações): sem erro nem NaN; o one-hot da tarefa nas 6 últimas colunas do crítico;
   `alvo_macro` com 13 posições; `privilegiados` com 6.
3. Contagem de funções antes → depois (473 na linha de base).
4. `git log 3a01bc1..HEAD --format=%B | grep -ci co-authored` = 0.
5. Uma revisão (opus, esforço médio) do lote contra esta spec. Se houver achado, um despacho de correção.
