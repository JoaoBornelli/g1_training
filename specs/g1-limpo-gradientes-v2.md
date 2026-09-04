# Spec de implementação — gradientes v2.1 do g1_limpo

Fonte do "porquê": `docs/planos/2026-09-04-proposta-gradientes-g1-limpo.md` (v2.1), §3 e §5. Auditoria: `docs/planos/2026-09-04-auditoria-gradientes-g1-limpo.md`. Este arquivo é o contrato de implementação. Ele fixa nomes, arquivos e checks.

## 0. Regras para quem implementa

- Repo `g1_training`, branch `exp/g1-limpo-v2`. Só a pasta `g1_limpo/` e `g1_limpo/ARQUITETURA.md`.
- Python: `.venv/bin/python`. Smoke: `cd g1_training && .venv/bin/python -m g1_limpo.smoke`. Ele imprime `N ok / M falhas`. Sem GPU: só o smoke roda aqui.
- `g1_limpo/` não importa `g1_training`, `g1_poc` nem `g1_multitask`. Import entre módulos do próprio `g1_limpo` é permitido.
- Estilo: igual ao existente. Docstring em português. Um `⚠` por fato medido ou por armadilha. Nenhum comentário decorativo.
- TDD: escrever o check no smoke primeiro; rodar; ver a falha; implementar; rodar; ver passar. Checks existentes que afirmam o contrato VELHO (listados abaixo) mudam para o contrato novo. Nenhum check é apagado sem substituto.
- Simplicidade: remover o que a spec manda remover. Não acrescentar termo, knob ou buffer fora desta lista.
- Git: NÃO commitar, NÃO `stash`, NÃO `reset`, NÃO `checkout`. Não tocar em arquivo untracked (`docs/handoff/`, `record_rl.py`, `rl_rollout.npz`, `g1_multitask/variacao_pose.py`, `g1_poc/*.patch`, `g1_poc/patches3/`, `*.ipynb`, `reference_checkpoints/`, `ver_play.py`).
- Ao terminar: reportar o `N ok / M falhas` antes e depois, a lista de arquivos tocados, e qualquer desvio desta spec com o motivo.

## 1. Lote C1 — números, remoções e o relógio da `sustentacao`

### P1 — `knobs.Tarefa`
- `precise_pos: 2.0 → 3.0`. `precise_pos_sigma: 0.05 → 0.18`.
- Docstring da classe: a soma dos sete pesos vira 12,5/s. Corrigir o comentário do `precise_pos_sigma`: ele é rampa de aceite, raio 0,18 > `tol_pos` 0,10, e paga 0,73 no limiar.

### P2 — `sustentacao` lê o relógio do comando
`comando.py`:
- Novo buffer `self._sustain_alvo` (float, por env), alocado em `__init__` ao lado de `_sust`.
- Ele é escrito em TODA abertura de elo: no `_resample_command` (reset) e no ramo de avanço de `_avanca_elo_force`. O valor é a regra que hoje vive inline em `_avanca_elo`: PEGAR → `sustenta_pegar_s`; CARREGAR → `_segurar[env]` se `_segura_parado`, senão `carregar_s`; REORIENTAR e BOTAR → `sustenta_outros_s`; ANDAR → 0.
- `_avanca_elo` passa a ler `self._sustain_alvo[nao_fechou]` em vez de recalcular. Uma fonte.
- Remover `self.avancou`: a alocação (`:364`), o zeramento (`:681`) e a escrita (`:1007`).

`recompensas.py`:
- A classe `sustentacao` vira função:
  ```python
  def sustentacao(env, nome_do_comando):
      t = _t(env, nome_do_comando)
      return (t._sust / t._sustain_alvo.clamp(min=1e-6)).clamp(0.0, 1.0) * _valida(env, nome_do_comando)
  ```
- Docstring: um ⚠ dizendo que ela paga pela condição de FECHO do elo, lida do comando, e por que o relógio próprio saiu (condição mais frouxa que o fecho; `avancou` pegajoso zerava o termo para sempre após o primeiro avanço).

`env_cfg.py`: params de `sustentacao` viram `{"nome_do_comando": _cmd}`.
`knobs.py`: remover `Tarefa.sustenta_s` e o comentário dele.
`ARQUITETURA.md`: atualizar as menções (`:77`, `:1256`, `:1556`, `:1713-1718`) para a função nova. Edição mínima.

### P6 — margem da pelve
- `knobs.Tarefa.pelve_margem: float = 0.03`, com um comentário de uma linha: a rampa satura acima do limiar de `de_pe`, para a política não parar na borda.
- `env_cfg.py`: `postura_ereta` recebe `pelve_alvo=tr.pelve_alvo + tr.pelve_margem`. O comando continua recebendo `tr.pelve_alvo`.

### P7 — `caiu` desarmado
`terminacoes.caixa_largada`:
- `return caiu | (escapou & (pegou > 0.5))`. Se `pegou is None`, `escapou` é falso e `caiu` vale sozinho.
- Docstring: a arma vale só para `escapou`. `caiu` no reset é falso porque o fundo da caixa está a `topo ≥ 0,04 > folga 0,02`.

### P8 — REORIENTAR inerte sorteado a 5% (decisão do dono, 2026-09-04: opção b)

Motivo: o one-hot do elo tem um canal para o REORIENTAR. O normalizador do rsl_rl é `(x − μ)/(σ + 0,01)` sem clamp; um canal constante tem `σ = 0` e, ao acender, entra como ×100. É a mesma regra que fixa `fatia_loco = 0,95`. Portanto o REORIENTAR NÃO sai do sorteio; ele cai para 5% enquanto inerte.

- `knobs.Cadeia.prob_reorientar_inerte: float = 0.05`, com dois `⚠`: a regra do normalizador, e o custo (0,3 s em 5% dos envs de manipulação = 0,08% do tempo; cadeias 2+3 vão de 2,8% para 5,3% dos envs).
- `env_cfg.py`: `ELOS_SORTEAVEIS = (CMD.REORIENTAR, CMD.PEGAR)` volta a ser constante de módulo. `elos_sorteaveis(k)` e `ELOS_SORTEAVEIS_COM_REORIENTAR` (criados pelo C1) SAEM. O termo `elo` recebe um param novo `pesos_manip`: `(p, 1 − p)` com `p = k.cadeia.prob_reorientar_inerte if k.cadeia.reorientar_inerte else 0.5`.
- `curriculo.sorteia_elo(..., elos_manip, pesos_manip=None, ...)`: se `pesos_manip` vier, `k = torch.multinomial(torch.tensor(pesos_manip), n, replacement=True)`; senão uniforme como hoje. Um `⚠` no docstring com a regra do normalizador.
- Checks pré-existentes que o C1 mudou para o contrato 0% VOLTAM: `smoke.py` (~1052) afirma `ELOS_SORTEAVEIS == (REORIENTAR, PEGAR)`; o check do C1 "nenhum env nasce em REORIENTAR" é SUBSTITUÍDO por: com `reorientar_inerte=True`, sobre ≥ 4096 sorteios de manipulação a fração em REORIENTAR fica em [0,03; 0,07]; com `reorientar_inerte=False`, em [0,45; 0,55]. O check `smoke.py:3595` ("o interruptor está ligado e chega ao comando") volta a testar algo: acrescentar que `cfg.curriculum["elo"].params["pesos_manip"][0] == k.cadeia.prob_reorientar_inerte`.

### P9 — `impacto_da_caixa`
`metricas.termos()`: `MetricsTermCfg(func=impacto_da_caixa, params={...}, reduce="max")`. Corrigir o docstring da classe: com `reduce="mean"` o painel publicava a média de um pico monótono, um piso.

### Smoke do lote C1
Mudar:
- `smoke.py:1052` (`ELOS_SORTEAVEIS == (REORIENTAR, PEGAR)`) FICA como está (P8-b mantém a tupla).

Acrescentar, numa seção nova `--- v2.1: gradientes`:
1. `precise_pos(d = tol_pos) ≥ 0,5`, calculado em fórmula com os knobs: `exp(−(0,10/0,18)²) = 0,73`. E a derivada do par caixa→alvo em `d = tol_pos` é maior que em `d = 0,45`: `2,7·(2d/σ²)·exp(−(d/σ)²)` com σ 0,45, mais `3,0·(2d/0,18²)·exp(−(d/0,18)²)`; em 0,10 dá ≈ 16,1, em 0,45 dá ≈ 2,4.
2. Num env sintético com `elo=PEGAR` e `cadeia=2`: `sustentacao` vale `_sust/_sustain_alvo`, com `_sustain_alvo == sustenta_pegar_s`; após `forca_avanco` para CARREGAR, no passo seguinte `sustentacao == 0` e `_sustain_alvo == carregar_s`; o atributo `avancou` não existe mais no termo de comando (`not hasattr`).
3. `caixa_largada`: com a caixa teleportada ao chão (z do centro = meia + 0,01) e `limpo_pegou = 0`, o termo é verdadeiro; no reset ele é falso.
4. (substituído por P8-b) Com `reorientar_inerte=True`, a fração de envs de manipulação em REORIENTAR sobre ≥ 4096 sorteios está em [0,03; 0,07].
5. `cfg.metrics["impacto_da_caixa"].reduce == "max"`.
6. `postura_ereta` recebe `pelve_alvo = tr.pelve_alvo + tr.pelve_margem`, e a rampa em `z = tr.pelve_alvo` vale `< 1,0` (derivada viva no limiar do fecho).

## 2. Lote C2 — `renda_congelada`, o gate do rastreio, o twist fixo, a régua

### P3 — `renda_congelada` entra, `load` sai
`comando.py`:
- Buffer `self._fechos` (long, por env), zero no `_resample_command`.
- Em `_avanca_elo_force`, ANTES de mudar `self._elo`: `origem = self._elo[ids]`; `ganho = origem != REORIENTAR`; incrementar `self._fechos[ids]` em `(ganho & (pode | (tem & ~pode))).long()`. Isto é: todo fecho, avanço ou terminal, exceto o inerte.

`recompensas.py`:
- Classe `renda_congelada` conforme a proposta §3 P3: `__init__` resolve os índices dos `termos` em `env.reward_manager.active_terms`; buffers `soma_anterior`, `congelado`, `fechos_anterior`; `__call__` congela `soma_anterior` onde `_fechos` subiu, atualiza `fechos_anterior`, lê `env.reward_manager._step_reward[:, idx].sum(-1)` para `soma_anterior`, devolve `congelado × _valida`; `reset(env_ids)` zera os três.
- Docstring com dois ⚠: (a) congela a SOMA em número, não os termos vivos contra o alvo velho, e por quê; (b) depende de `_step_reward` (privado do mjlab) e de ser o ÚLTIMO termo do dict, e o smoke fixa os dois.
- Remover `load`. `largou` vira `soltou × (1 − exp(−(d_palma/σ_solta)²))`; os params `sensor_apoio` e `raio_mult` saem dele. Docstring: o `× load` era gate de "apoiada"; na espera final o BOTAR já fechou com `apoiada`, e `caiu` termina se a caixa cair depois.

`env_cfg.py`:
- Remover a fiação de `load`.
- `TERMOS_CONGELAVEIS = ("staged", "precise_pos", "precise_ori", "squeeze", "unload", "postura_ereta", "sustentacao", "track_linear_velocity", "track_angular_velocity")` como constante de módulo.
- `cfg.rewards["renda_congelada"] = RewardTermCfg(func=RC.renda_congelada, weight=tr.renda_congelada, params={"nome_do_comando": _cmd, "termos": TERMOS_CONGELAVEIS})`, inserido DEPOIS de todos os outros — no fim da seção 3i. Afirmar no smoke que é o último.

`knobs.py`: remover `Tarefa.load`, `Tarefa.load_raio_mult` e o bloco de comentário deles. Acrescentar `renda_congelada: float = 1.0` com um comentário de duas linhas. `largou` e `sigma_solta` ficam.
`ARQUITETURA.md`: linha `:310` (3i) passa a listar `largou` e `renda_congelada`.

### P4 — um gate só para o rastreio
`comando.py`:
- Em `__init__`, alocar `env.limpo_twist_zerado = torch.zeros(n, device=d)` ao lado de `limpo_aguardando` (`:414`).
- Em `_zera_twist_nos_parados`, DEPOIS de calcular `parados` (incluindo o segurar-parado) e ANTES do `return` cedo: `self._env.limpo_twist_zerado.copy_(parados.float())`. O `return` cedo tem de vir depois da publicação, senão o buffer fica velho quando ninguém está parado.

`recompensas.py`:
- `rastreio_por_elo(env, *, func, **kwargs)` → `return func(env, **kwargs) * (1.0 - env.limpo_twist_zerado)`.
- Remover `_anda_neste_elo`. Docstring de `rastreio_por_elo`: a regra é "twist zerado PELA TAREFA não rende rastreio"; standing SORTEADO continua rendendo; as duas esperas e o segurar-parado deixam de render.

`env_cfg.py`: no laço `for _nome_rastreio in (...)`, os params passam a ser só `{"func": ...}` mais os originais do molde. `ELOS_QUE_ANDAM` fica para `PosturaPorElo` e `reset_base_por_elo`.

### P5 — twist fixo no CARREGAR-andando
`comando.py`:
- Buffers `self._twist_carregar` (float, `(n, 3)`) e `self._twist_valido` (bool, `(n,)`), alocados em `__init__`. `_twist_valido` vai a `False` no `_resample_command` e para os `ids` que avançam em `_avanca_elo_force`.
- Em `_zera_twist_nos_parados`, depois de zerar os parados: `anda_c = (self._elo == CARREGAR) & ~self._segura_parado(todos)`; para `anda_c & ~_twist_valido`: sortear `v_x ∈ U(0,3; 1,0)`, `v_y = 0`, `w_z = 0`, gravar, marcar válido; para todo `anda_c`: `tw.vel_command_b[anda_c] = self._twist_carregar[anda_c]`.
- Um ⚠ no docstring: sem isto 10% dos envs recebem standing e não fecham `andou`, e o resample de 3 a 8 s inverte o sentido antes de `andou` valer.

### P10 — a régua da caixa
`metricas.py`:
- Classe `aproxima_caixa`: buffer `self.minimo` por env, iniciado em 1,0 no `reset`; a cada passo `d = ‖caixa − ALVO‖ / sigma_trazer.clamp(min=1e-6)` do termo de comando `alvo_caixa`; `self.minimo = min(self.minimo, d)` só onde `VALIDA > 0,5`; devolve `self.minimo`. `MetricsTermCfg(..., reduce="last")`. Leitura: 1,0 = não saiu de onde o elo abriu; 0 = no alvo.
- Função `renda_manipulacao`: soma de `env.reward_manager._step_reward[:, idx]` sobre `TERMOS_CONGELAVEIS + ("renda_congelada",)`; os índices resolvidos uma vez (classe com `__init__`). `reduce` default. É a régua da regra 1 no treino.
- Ambas entram em `termos()`.

### Smoke do lote C2
Mudar:
- `smoke.py:1101` (params `elos_que_andam` nos dois rastreios) → afirmar que os dois rastreios NÃO têm `elos_que_andam` nos params e que `RC.rastreio_por_elo` é o `func`.

Acrescentar na seção `--- v2.1: gradientes`:
7. `list(cfg.rewards)[-1] == "renda_congelada"`; `"load" not in cfg.rewards`; `cfg.rewards["largou"].params` sem `sensor_apoio` e `raio_mult`.
8. Num env sintético `elo=PEGAR, cadeia=3`, 32 envs: `renda_congelada == 0` em todos; `_fechos == 0`. Um passo. `forca_avanco` (→ CARREGAR). Um passo: `_fechos == 1` e `renda_congelada` ≈ a soma dos `TERMOS_CONGELAVEIS` lida de `_step_reward` no passo ANTERIOR ao avanço (guardar essa soma antes do `forca_avanco`; tolerância 1e−4). `forca_avanco` (→ BOTAR), um passo: `_fechos == 2`. `forca_avanco` (fecho terminal), um passo: `_fechos == 3` e `renda_congelada` subiu de novo.
9. Regra 1 nas transições: no MESMO env sintético, a soma total das recompensas do passo seguinte a cada `forca_avanco` é ≥ a do passo anterior − 1e−3, para as três transições de 8. Ler `env.reward_manager._reward_buf.sum()` ou a soma das colunas de `_step_reward`.
10. Com `elo=REORIENTAR` forçado e `reorientar_inerte=True` (usar `elo=` explícito, pois o sorteio não o produz mais): após o fecho inerte, `_fechos == 0` e `renda_congelada == 0`.
11. `rastreio_por_elo` = 0 quando `limpo_twist_zerado = 1` e igual ao termo do molde quando `= 0`. Num env `elo=PEGAR` durante a espera inicial (`_passa_janela` NÃO chamado): `limpo_twist_zerado == 1` e `track_linear_velocity == 0`. Num env `elo=CARREGAR, cadeia=3` (segurar parado): `limpo_twist_zerado == 1`. Num env `elo=ANDAR`: `limpo_twist_zerado == 0`.
12. Num env `elo=CARREGAR, cadeia=2`: `‖vel_command_b[:, :2]‖ ≥ 0,3` em todo env, e igual em dois passos consecutivos.
13. `cfg.metrics` tem `aproxima_caixa` com `reduce="last"` e `renda_manipulacao`. Num env `elo=PEGAR` depois da janela, `aproxima_caixa` vale ≈ 1,0 no primeiro passo ativo (σ = d₀).

## 2b. Lote C3 — P8-b

Depois do C2. Implementa a seção P8 acima e desfaz o P8 a 0% que o C1 aplicou (`elos_sorteaveis(k)`, `ELOS_SORTEAVEIS_COM_REORIENTAR`, o check "nenhum env nasce em REORIENTAR"). Toca `knobs.py`, `curriculo.py`, `env_cfg.py`, `smoke.py`.

## 3. Lote R — revisão

Ler o diff completo (`git diff`) contra a proposta §3 e esta spec. Verificar:
- Cada item P1–P10 está implementado como escrito, ou o desvio está justificado no relatório do coder.
- Nada foi acrescentado fora da lista. Nada listado para remover ficou.
- `renda_congelada` é o último termo; lê a soma do passo ANTERIOR; não incrementa no REORIENTAR inerte.
- `_sustain_alvo` é escrito em TODA abertura de elo (reset e avanço), e `_avanca_elo` lê dele.
- `limpo_twist_zerado` é publicado antes do `return` cedo.
- `caiu` sem `pegou`; `escapou` com `pegou`.
- Smoke verde; nenhum check antigo apagado sem substituto.
- Docstrings no estilo do módulo; nenhum comentário decorativo.
Saída: APROVADO ou lista de correções com arquivo e linha.
