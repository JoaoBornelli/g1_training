# Contrato de troca de tarefa v2 — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar na branch `exp/g1-limpo-v2` a spec v14 do contrato de troca de tarefa: gate da caixa, esperas publicando `ANDAR`, cadeia 3 de três elos, renda monótona do `BOTAR`, DR de tamanho, `giro_b`, crítico com elo interno, giro no lugar, `REORIENTAR` inerte.

**Architecture:** Um termo de comando (`AlvoCaixaCmd`) separa o elo **publicado** (o que a rede vê) do elo **interno** (a mecânica do episódio). As recompensas de caixa leem o interno; a observação lê o publicado e gateia os canais de caixa. Tudo o mais é append: canais novos entram no fim, termos novos entram por `env_cfg.py`, knobs novos entram em `knobs.py`. O `smoke.py` é a bateria de testes; cada task acrescenta os seus checks e deixa o smoke verde.

**Tech Stack:** Python 3.12, `mjlab` 1.5.1 (local, `.venv`) / 1.5.3 (Kaggle), `mujoco_warp`, `rsl_rl` 5.4, PyTorch em CPU para os checks.

**Spec:** `docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md` (v14, aprovada em 2026-09-03). O plano cita a spec por seção; o executor lê as duas.

## Global Constraints

Copiadas da spec. Valem para toda task.

- Branch de trabalho: `exp/g1-limpo-v2`. A `exp/g1-limpo` **não** recebe nada.
- Commits: `git -c core.hooksPath=/dev/null commit`. **Sem** `Co-Authored-By`. **Sem** `git push` (o dono faz).
- `g1_limpo/` **não importa** `g1_training`, `g1_poc` nem `g1_multitask`. Só `mjlab`, `mujoco`, `rsl_rl`, `torch`.
- Nunca commitar os untracked do dono: `docs/handoff/`, `record_rl.py`, `rl_rollout.npz`, `g1_multitask/variacao_pose.py`, `g1_poc/*.patch`, `g1_poc/patches3/`, `*.ipynb`, `reference_checkpoints/`, `ver_play.py`. Nunca `git stash`, nunca `git reset`.
- Tudo roda em CPU: `smoke.py`, `paridade.py`, `inspeciona.py`. Nenhum treino aqui.
- `git diff exp/g1-limpo -- g1_limpo/curriculo.py` fica **vazio** ao fim.
- Em `g1_limpo/recompensas.py` só mudam `_alcancar`, `squeeze`, `unload`, `rastreio_por_elo` (limpeza) e entram `load` e `largou`. Em `g1_limpo/terminacoes.py` só muda `caixa_largada`. Em `g1_limpo/eventos.py` **não** mudam `entrega_tarefa_no_viewer` nem `avanca_elo_no_viewer`.
- Pesos existentes em `knobs.py` não mudam. Knobs novos e seus valores (spec §13): `Tarefa.load = 2.0`, `Tarefa.largou = 1.0`, `Tarefa.load_raio_mult = 2.0`, `Tarefa.sigma_solta = 0.10`, `Terminacao.caixa_folga_chao = 0.02` (substitui `caixa_z_min`), `Alvo.espera_s = (0.5, 1.5)`, `Cena.caixa_meia_aresta_faixa = (0.07, 0.13)`, `Cena.caixa_n_variantes = 8`, `Marcha.rel_turning_envs = 0.10`, `Marcha.turning_wz_min = 0.2`, `Nivel.voltas_max = (0,)*7`, `Nivel.eixo_vertical = (False,)*7`.
- Observação: ator **114** canais, crítico **119**. Layout do comando: canal novo entra **por último** (`GIRO = slice(9, 12)`, `DIM = 12`).
- O layout da observação é **append**: ator termina em `[..., elo(5), caixa(10)]`; crítico em `[..., elo(5), caixa(10), elo_interno(5)]`.
- Comando para rodar o smoke, sempre da raiz do repo (ele lê caminhos relativos):

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.smoke 2>&1 | tail -25
```

  Ele imprime `N ok / M falhas` e sai com código 1 se houver falha. Um check novo que **falha antes** e **passa depois** é o TDD deste repositório.

---

## Mapa de arquivos

| arquivo | responsabilidade nesta v2 |
|---|---|
| `g1_limpo/comando.py` | cadeia 3; regras por cadeia; publicado nas duas esperas (`_soltou`, `_segurar`); `_pegou` gateado; canal `GIRO`; leitura do tamanho por env; publica `limpo_elo_interno`, `limpo_soltou`; ramo `is_turning_env` no twist |
| `g1_limpo/observacoes.py` | gate pelo publicado; `giro_b`; `meia_aresta`; `VALIDA` fora; `um_de_cinco_interno`; `fatia_do_elo_interno`; `N_CAIXA = 10` |
| `g1_limpo/eventos.py` | evento de startup `tamanho_caixa`; `posiciona_cena` e `afasta_cena` leem o tamanho por env |
| `g1_limpo/terminacoes.py` | `caixa_largada` com `caiu` por tamanho e `escapou & ~soltou` |
| `g1_limpo/recompensas.py` | `_alcancar ≡ 1` no `BOTAR`/`soltou`; máscaras de `squeeze` e `unload`; `load`; `largou`; `rastreio_por_elo` sem `aguardando` |
| `g1_limpo/metricas.py` | `fracao_esperando` lê `aguardando ∨ soltou` |
| `g1_limpo/algoritmo.py` | `PPOPorElo` agrupa pelo `elo_interno` do crítico |
| `g1_limpo/knobs.py` | os knobs novos listados acima |
| `g1_limpo/env_cfg.py` | fiação: evento, termos, obs do crítico, params novos |
| `g1_limpo/smoke.py` | os checks novos (spec §11.1, itens 1 a 24) e os checks antigos que mudam |
| `g1_limpo/ARQUITETURA.md`, `g1_limpo/__init__.py` (docstring) | documentação do layout novo |

---

### Task 0: Linha de base

**Files:**
- Nenhum arquivo muda. Só leitura.

- [ ] **Step 1: Confirmar a branch e o estado limpo do que é rastreado**

```bash
cd /home/joaobornelli/Documents/g1_training && git -c core.hooksPath=/dev/null branch --show-current && git -c core.hooksPath=/dev/null status --short | grep -v '^??'
```

Esperado: `exp/g1-limpo-v2` e nenhuma linha depois (só untracked `??`, que não aparecem pelo `grep -v`).

- [ ] **Step 2: Rodar o smoke e anotar a contagem de base**

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.smoke 2>&1 | tail -3
```

Esperado: `425 ok / 0 falhas` (ou o número que a HEAD `31daa9e..d63961f` imprime; anote-o). Se houver falha aqui, PARE: a base está quebrada e nenhuma task começa.

- [ ] **Step 3: Rodar a paridade**

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.paridade 2>&1 | tail -5
```

Esperado: saída sem `FALHA`/`✗`. Anote a última linha.

---

### Task 1: Cadeia 3 = `(PEGAR, CARREGAR, BOTAR)` com regras por cadeia

Spec §6.5. O `CARREGAR` do meio tem twist zero e fecha por `perto` sustentado pela espera sorteada. Nada de índice `3` no corpo do comando: a marca "esta cadeia segura parado" é derivada de `CADEIAS`.

**Files:**
- Modify: `g1_limpo/comando.py` (bloco `CADEIAS`, `__init__`, `_resample_command`, `recebe_tarefa`, `_zera_twist_nos_parados`, `_fecha_elo_corrente`, `_avanca_elo`)
- Modify: `g1_limpo/knobs.py` (`Alvo.espera_s`, docstring de `Cadeia`)
- Modify: `g1_limpo/smoke.py` (bloco "o AVANÇO: forçado à mão", seção 19; seção nova 22)

**Interfaces:**
- Produces: `comando._SEGURA_PARADO: torch.BoolTensor[len(CADEIAS)]`; `AlvoCaixaCmd._segurar: Tensor[n]` (segundos); `AlvoCaixaCmd._segura_parado(ids) -> BoolTensor`.
- Consumes: nada novo.

- [ ] **Step 1: Escrever os checks que falham**

No fim de `g1_limpo/smoke.py`, **antes** da linha `# =============================================================================` que precede o `print()` final, acrescente:

```python
# ==================== 22. a cadeia 3 tem TRÊS elos e segura parado (spec §6.5) ======
secao("22. a cadeia 3: (PEGAR, CARREGAR, BOTAR), o CARREGAR do meio segura parado")
from g1_limpo import comando as CMD                                       # noqa: E402

check("9. CADEIAS[3] é (PEGAR, CARREGAR, BOTAR)",
      CMD.CADEIAS[3] == (CMD.PEGAR, CMD.CARREGAR, CMD.BOTAR), str(CMD.CADEIAS[3]))
check("9. o teto de elos é DERIVADO e vale 3", CMD._TETO_ELOS == 3)
check("9. a marca de segurar parado é derivada de CADEIAS: só a cadeia 3 a tem",
      CMD._SEGURA_PARADO.tolist() == [False, False, False, True],
      str(CMD._SEGURA_PARADO.tolist()))
check("as outras três cadeias não mudaram",
      CMD.CADEIAS[:3] == ((CMD.PEGAR,), (CMD.REORIENTAR, CMD.PEGAR),
                          (CMD.PEGAR, CMD.CARREGAR)))
check("toda espera é a MESMA faixa: espera_s = (0,5, 1,5)",
      tuple(k.alvo.espera_s) == (0.5, 1.5), str(k.alvo.espera_s))

# --- rodando: a cadeia 3 percorre os três elos com a caixa PINADA na âncora ---
# `elo=CARREGAR` liga o `segura_caixa` + `pina_caixa` (a caixa fica no peito a cada
# passo); `cadeia=3` vence e o elo de abertura é o PEGAR. Com a caixa na âncora, o
# PEGAR fecha sozinho depois da espera + 0,5 s; o CARREGAR de segurar parado fecha
# por `perto` sustentado pela espera sorteada; o BOTAR nunca fecha (a caixa pinada
# no ar não é `apoiada`).
try:
    import torch as _t22

    _c22 = make_env_cfg(k, inspecao=True, elo=CMD.CARREGAR, cadeia=3)
    _c22.scene.num_envs = 16
    _e22 = ManagerBasedRlEnv(cfg=_c22, device="cpu")
    _e22.reset()
    _n22 = _e22.action_manager.total_action_dim
    _t22c = _e22.command_manager.get_term("alvo_caixa")
    _tw22 = _e22.command_manager.get_term("twist")
    _dt22 = _e22.step_dt
    _t1 = _t22.full((_e22.num_envs,), -1, dtype=_t22.long)
    _t2 = _t22.full((_e22.num_envs,), -1, dtype=_t22.long)
    _twist_no_carregar = 0.0
    for _i in range(240):
        _e22.step(_t22.zeros(_e22.num_envs, _n22))
        _p = _t22c._passo
        _t1 = _t22.where((_t1 < 0) & (_p >= 1), _t22.full_like(_t1, _i), _t1)
        _t2 = _t22.where((_t2 < 0) & (_p >= 2), _t22.full_like(_t2, _i), _t2)
        if bool(((_p == 1)).any()):
            _twist_no_carregar = max(_twist_no_carregar,
                                     float(_tw22.vel_command_b[_p == 1].abs().max()))
    check("9. a máquina de elo percorre PEGAR -> CARREGAR -> BOTAR sozinha",
          bool((_t22c._passo == 2).all()) and bool((_t22c._elo == CMD.BOTAR).all()),
          f"passo {_t22c._passo.tolist()[:8]}")
    check("9. e `fechou` NÃO marca no BOTAR com a caixa no ar",
          not bool(_t22c.fechou.any()))
    _seg = (_t2 - _t1).float() * _dt22
    check("11. o CARREGAR de segurar parado dura a ESPERA sorteada (0,5 a 1,5 s)",
          bool((_seg >= k.alvo.espera_s[0] - 2 * _dt22).all())
          and bool((_seg <= k.alvo.espera_s[1] + 3 * _dt22).all()),
          f"durações medidas {[round(float(x), 2) for x in _seg[:8]]} s")
    check("10. no CARREGAR da cadeia 3 o twist é ZERO em todo passo",
          _twist_no_carregar == 0.0, f"máximo medido {_twist_no_carregar:.4f}")
    del _e22

    # --- controle: na cadeia 2 o CARREGAR ANDA e fecha por distância ---
    _c22b = make_env_cfg(k, inspecao=True, elo=CMD.CARREGAR, cadeia=2)
    _c22b.scene.num_envs = 16
    _e22b = ManagerBasedRlEnv(cfg=_c22b, device="cpu")
    _e22b.reset()
    _n22b = _e22b.action_manager.total_action_dim
    _t22d = _e22b.command_manager.get_term("alvo_caixa")
    _tw22b = _e22b.command_manager.get_term("twist")
    _twist_c2 = 0.0
    for _ in range(240):
        _e22b.step(_t22.zeros(_e22b.num_envs, _n22b))
        if bool((_t22d._passo == 1).any()):
            _twist_c2 = max(_twist_c2,
                            float(_tw22b.vel_command_b[_t22d._passo == 1].abs().max()))
    check("11. na cadeia 2 o CARREGAR NÃO fecha com o robô parado — `andou` continua",
          bool((_t22d._passo == 1).all()), f"passo {_t22d._passo.tolist()[:8]}")
    check("10. e na cadeia 2 o twist RELIGA no CARREGAR",
          _twist_c2 > 0.0, f"máximo medido {_twist_c2:.4f}")
    del _e22b
except Exception as _e22x:      # noqa: BLE001
    _falhas.append(f"a cadeia 3 não pôde ser exercitada: "
                   f"{type(_e22x).__name__}: {_e22x}")
```

E no bloco existente da seção 19 ("o AVANÇO: forçado à mão", ~linha 2106), troque as três linhas que assumem `(PEGAR, BOTAR)`:

```python
    _ca.commands["alvo_caixa"].cadeia_forcada = 3        # (PEGAR, BOTAR)
```
por
```python
    _ca.commands["alvo_caixa"].cadeia_forcada = 3        # (PEGAR, CARREGAR, BOTAR)
```
e
```python
    check("o avanço muda o elo, e o novo é o 2º da cadeia forçada",
          bool((_tac._elo == CMD.BOTAR).all()) and bool((_elo_antes == CMD.PEGAR).all()),
          f"{_elo_antes.tolist()[:3]} -> {_tac._elo.tolist()[:3]}")
```
por
```python
    check("o avanço muda o elo, e o novo é o 2º da cadeia forçada",
          bool((_tac._elo == CMD.CARREGAR).all()) and bool((_elo_antes == CMD.PEGAR).all()),
          f"{_elo_antes.tolist()[:3]} -> {_tac._elo.tolist()[:3]}")
    # o 2º avanço leva ao BOTAR; o invariante da laje abaixo é medido nesse instante
    _tac.forca_avanco(_ids_a)
    check("o segundo avanço leva ao BOTAR, e `_passo` vai a 2",
          bool((_tac._elo == CMD.BOTAR).all()) and bool((_tac._passo == 2).all()))
```
e apague a check `"o `_passo` foi para 1"` (ela virou a de cima) e a check `"no avanço para CARREGAR o twist RELIGA"` inteira com o seu comentário (a seção 22 a mede na cadeia certa).

- [ ] **Step 2: Rodar o smoke e ver as falhas novas**

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.smoke 2>&1 | grep -E '✗|falhas'
```

Esperado: falhas em `9. CADEIAS[3]`, `9. o teto`, `9. a marca` (AttributeError vira falha registrada), `toda espera é a MESMA faixa`, e o bloco 22 acusa exceção ou `passo` errado.

- [ ] **Step 3: `knobs.py` — a espera e a docstring da cadeia**

Em `class Alvo`, troque `espera_s: tuple[float, float] = (0.3, 1.0)` por:

```python
    # ⚠ TODA espera é a MESMA faixa (spec §6.3, §6.5): a espera inicial em ANDAR
    # publicado antes do PEGAR, e o "segurar parado" do CARREGAR da cadeia 3. Sorteada
    # para a política não contar passos. Decisão do dono, 02/09 (quarta rodada).
    espera_s: tuple[float, float] = (0.5, 1.5)
```

Em `class Cadeia`, troque no docstring `⚠ TETO DE 2 ELOS. As cadeias são:` e a linha `índice 3: (PEGAR, BOTAR)` por:

```
    ⚠ O TETO É DERIVADO de `CADEIAS` (hoje 3). As cadeias são:
      índice 0: (PEGAR,)                 -> 1 elo (cadeia curta da F3)
      índice 1: (REORIENTAR, PEGAR)
      índice 2: (PEGAR, CARREGAR)        -> andar com a caixa
      índice 3: (PEGAR, CARREGAR, BOTAR) -> pegar, SEGURAR PARADO, botar (spec §6.5)
```

- [ ] **Step 4: `comando.py` — a cadeia e a marca derivada**

Troque o bloco `CADEIAS` (comentário "Teto de 2 elos" incluído) por:

```python
# --- as cadeias de elo (F4). O teto é DERIVADO (`_TETO_ELOS`), nunca redigitado. ---
# índice 0: cadeia de 1 elo (PEGAR, já treina desde F3)
# índice 1, 2: cadeias de 2 elos
# índice 3: (PEGAR, CARREGAR, BOTAR) — pegar, SEGURAR PARADO, botar (spec §6.5). O
#           controlador de campo nunca manda BOTAR a partir de PEGAR; ele passa por
#           CARREGAR com v = 0. A cadeia treina exatamente isso.
CADEIAS = (
    (PEGAR,),
    (REORIENTAR, PEGAR),
    (PEGAR, CARREGAR),
    (PEGAR, CARREGAR, BOTAR),
)
```

Logo depois do laço que preenche `_ELO_EM`, acrescente:

```python
# ⚠ A CADEIA DE SEGURAR PARADO (spec §6.5): aquela em que o CARREGAR é seguido do BOTAR.
# Nela o CARREGAR tem twist ZERO e fecha por `perto` sustentado pela espera sorteada, em
# vez de `andou`. DERIVADA de `CADEIAS`, e o índice 3 não aparece no corpo do termo — uma
# tabela paralela escrita à mão sai de sincronia no dia em que uma cadeia mudar.
_SEGURA_PARADO = torch.tensor(
    [any(c[i] == CARREGAR and c[i + 1] == BOTAR for i in range(len(c) - 1))
     for c in CADEIAS], dtype=torch.bool)
```

- [ ] **Step 5: `comando.py` — o buffer `_segurar` e o sorteio**

No `__init__`, logo depois de `self._pos_no_elo = torch.zeros(n, 3, device=d)`, acrescente:

```python
        # ⚠ O SUSTAIN do CARREGAR de segurar parado (spec §6.5 item 3): a espera
        # sorteada do MESMO knob `espera_s`, por env. Só a cadeia marcada em
        # `_SEGURA_PARADO` o lê; as outras usam `carregar_s`.
        self._segurar = torch.zeros(n, device=d)
```

No `_resample_command`, logo depois do bloco que escreve `self._espera[env_ids] = torch.where(...)`, acrescente:

```python
        # o "segurar parado" da cadeia 3 usa a MESMA faixa; sorteio próprio, por env
        self._segurar[env_ids] = lo + (hi - lo) * torch.rand(n, device=d)
```

No `recebe_tarefa`, logo depois de `self._espera[ids] = lo + (hi - lo) * torch.rand(len(ids), device=d)`, acrescente:

```python
        self._segurar[ids] = lo + (hi - lo) * torch.rand(len(ids), device=d)
```

- [ ] **Step 6: `comando.py` — a máscara e as três regras por cadeia**

Acrescente este método logo antes de `_zera_twist_nos_parados`:

```python
    def _segura_parado(self, ids: torch.Tensor) -> torch.Tensor:
        """Máscara: o env está no CARREGAR da cadeia de SEGURAR PARADO (spec §6.5).

        ⚠ `_cadeia == −1` no `ANDAR`: o `tem` impede indexar a tabela com −1, que em
        Python devolveria a ÚLTIMA cadeia — o mesmo defeito que `n_elos_da_cadeia`
        já guarda.
        """
        cad = self._cadeia[ids]
        tem = cad >= 0
        seg = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
        if bool(tem.any()):
            seg[tem] = _SEGURA_PARADO.to(self.device)[cad[tem]]
        return seg & (self._elo[ids] == CARREGAR)
```

Em `_zera_twist_nos_parados`, troque

```python
        parados = torch.isin(self._elo, torch.tensor(self.cfg.elos_parados,
                                                     device=self.device))
```
por
```python
        parados = torch.isin(self._elo, torch.tensor(self.cfg.elos_parados,
                                                     device=self.device))
        # ⚠ REGRA POR CADEIA (spec §6.5 item 2): o CARREGAR de segurar parado também
        # tem twist zero. Na cadeia 2 o CARREGAR continua andando.
        parados = parados | self._segura_parado(
            torch.arange(self.num_envs, device=self.device))
```

Em `_fecha_elo_corrente`, no ramo `elif elo_tipo == CARREGAR:`, troque `fecha[m] = perto[m] & andou` por:

```python
                # ⚠ REGRA POR CADEIA (spec §6.5 item 3): em SEGURAR PARADO não há
                # `andou` — o twist é zero e a condição é `perto`, sustentado pela espera
                # sorteada (ver `_avanca_elo`). Sem `perto`, o BOTAR começaria com a
                # caixa em qualquer lugar.
                segura = self._segura_parado(ids)[m]
                fecha[m] = torch.where(segura, perto[m], perto[m] & andou)
```

Em `_avanca_elo`, troque

```python
                elif elo_tipo == CARREGAR:
                    sustain_alvo[m] = self.cfg.carregar_s
```
por
```python
                elif elo_tipo == CARREGAR:
                    # ⚠ em SEGURAR PARADO o sustain É a espera sorteada (spec §6.5)
                    segura = self._segura_parado(nao_fechou)[m]
                    seg_s = self._segurar[nao_fechou][m]
                    sustain_alvo[m] = torch.where(
                        segura, seg_s, torch.full_like(seg_s, self.cfg.carregar_s))
```

E no `__init__` troque o comentário `self._passo = torch.zeros(n, dtype=torch.long, device=d)  # 0 ou 1` por `# 0 .. _TETO_ELOS-1`.

- [ ] **Step 7: Rodar o smoke até verde**

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.smoke 2>&1 | tail -6
```

Esperado: `N ok / 0 falhas`, com N ≥ base + 9. Se a check `11. ... dura a ESPERA sorteada` falhar por 1 passo, a causa é o `dt` de acumulação do `_sust`: o sustain acumula `dt` por passo em que `fecha` vale, portanto a duração é `ceil(espera/dt)` passos — a folga de `3 * dt` na check já cobre isso; se ainda falhar, imprima `_seg` e compare com `_t22c._segurar`.

- [ ] **Step 8: Commit**

```bash
cd /home/joaobornelli/Documents/g1_training && git -c core.hooksPath=/dev/null add g1_limpo/comando.py g1_limpo/knobs.py g1_limpo/smoke.py && git -c core.hooksPath=/dev/null commit -m "feat(limpo): cadeia 3 vira (PEGAR, CARREGAR, BOTAR) e o CARREGAR do meio segura parado

Spec §6.5. A marca de segurar parado e derivada de CADEIAS (_SEGURA_PARADO):
twist zero e fecho por perto sustentado pela espera sorteada do knob
espera_s, agora (0,5, 1,5) para toda espera. Na cadeia 2 o CARREGAR segue
andando e fechando por distancia. Smoke: secao 22 percorre os tres elos
com a caixa pinada e mede a duracao do segurar parado."
```

---

### Task 2: O publicado nas duas esperas

Spec §6.0, §6.3, §6.6.3, §6.4. O one-hot publicado é `ANDAR` enquanto `aguardando ∨ soltou`; o `VALIDA` segue derivado do interno e a espera final **não** o zera; `_pegou` só arma com o objetivo ativo; `escapou` desarmado depois de `soltou`; `fracao_esperando` conta as duas esperas.

**Files:**
- Modify: `g1_limpo/comando.py` (`__init__`, `_aplica_espera`, `_publica_pegou`, `_avanca_elo_force`, `_resample_command`, `recebe_tarefa`, `_update_command`)
- Modify: `g1_limpo/terminacoes.py` (`caixa_largada`)
- Modify: `g1_limpo/metricas.py` (`fracao_esperando`)
- Modify: `g1_limpo/recompensas.py` (`rastreio_por_elo`, só a limpeza)
- Modify: `g1_limpo/smoke.py` (seção nova 23)

**Interfaces:**
- Produces: `env.limpo_soltou: Tensor[n] float` (1.0 depois do fecho do `BOTAR`); `env.limpo_elo_interno: LongTensor[n]` (referência a `AlvoCaixaCmd._elo`); `AlvoCaixaCmd._soltou: BoolTensor[n]`.
- Consumes: Task 1 (`_segurar`).

- [ ] **Step 1: Escrever os checks que falham**

No fim de `g1_limpo/smoke.py`, antes do bloco final de impressão, acrescente:

```python
# ============ 23. as DUAS esperas publicam ANDAR, o VALIDA lê o interno (spec §6.3, §6.6)
secao("23. as duas esperas publicam ANDAR")
from g1_limpo import comando as CMD                                       # noqa: E402
from g1_limpo import observacoes as OB_                                   # noqa: E402
from g1_limpo import terminacoes as TE_                                   # noqa: E402
from g1_limpo import recompensas as RC_                                   # noqa: E402

check("7. o publicado é recalculado do INTERNO e das duas esperas",
      "aguardando | self._soltou" in inspect.getsource(CMD.AlvoCaixaCmd._aplica_espera)
      and "self._elo" in inspect.getsource(CMD.AlvoCaixaCmd._aplica_espera),
      "ler o que se escreveu no passo anterior deixa o canal preso (02/09)")
check("20. o `_pegou` só arma com o objetivo ATIVO",
      "self._espera <= 0.0" in inspect.getsource(CMD.AlvoCaixaCmd._publica_pegou),
      "um toque por exploração na espera inicial armaria `escapou` e mataria o episódio")
check("o `rastreio_por_elo` não lê mais `limpo_aguardando` — o publicado já é ANDAR",
      "limpo_aguardando" not in inspect.getsource(RC_.rastreio_por_elo))

try:
    import torch as _t23

    # --- a espera INICIAL, vista pela observação ---
    _c23 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _c23.scene.num_envs = 32
    _e23 = ManagerBasedRlEnv(cfg=_c23, device="cpu")
    _o23, _ = _e23.reset()
    _n23 = _e23.action_manager.total_action_dim
    _t23c = _e23.command_manager.get_term("alvo_caixa")
    _fat23 = OB_.fatia_do_elo(_o23["actor"].shape[-1])
    _hot0 = _o23["actor"][:, _fat23].argmax(-1)
    check("4. na observação do RESET o one-hot publicado é ANDAR",
          bool((_hot0 == CMD.ANDAR).all()), str(_hot0.tolist()[:8]))
    check("4. e o elo INTERNO é PEGAR no reset",
          bool((_t23c._elo == CMD.PEGAR).all())
          and bool((_e23.limpo_elo_interno == CMD.PEGAR).all()))
    check("3. o VALIDA é ZERO na espera inicial",
          float(_t23c.command[:, CMD.VALIDA].max()) == 0.0)
    _borda = _t23.full((_e23.num_envs,), -1, dtype=_t23.long)
    for _i in range(int(k.alvo.espera_s[1] / _e23.step_dt) + 5):
        _o23 = _e23.step(_t23.zeros(_e23.num_envs, _n23))[0]
        _hot = _o23["actor"][:, _fat23].argmax(-1)
        _borda = _t23.where((_borda < 0) & (_hot == CMD.PEGAR),
                            _t23.full_like(_borda, _i), _borda)
    check("4. na borda o one-hot publicado vira PEGAR em todos os envs",
          bool((_borda >= 0).all()), str(_borda.tolist()[:8]))
    _bs = _borda.float() * _e23.step_dt
    check("4. e a borda cai dentro da faixa de espera_s",
          float(_bs.min()) >= k.alvo.espera_s[0] - 2 * _e23.step_dt
          and float(_bs.max()) <= k.alvo.espera_s[1] + 2 * _e23.step_dt,
          f"{float(_bs.min()):.2f} .. {float(_bs.max()):.2f} s")
    check("3. depois da borda o VALIDA é UM",
          float(_t23c.command[:, CMD.VALIDA].min()) == 1.0)
    check("6. o piso de locomoção não conta a espera: `limpo_elo` segue PEGAR",
          bool((_e23.limpo_elo == CMD.PEGAR).all()),
          "a fatia lê o interno do currículo, não o publicado")
    del _e23

    # --- a espera FINAL, forçada à mão na cadeia 3 ---
    _c23b = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR, cadeia=3)
    _c23b.scene.num_envs = 8
    _e23b = ManagerBasedRlEnv(cfg=_c23b, device="cpu")
    _e23b.reset()
    _n23b = _e23b.action_manager.total_action_dim
    _passa_janela(_e23b, _n23b, _t23)
    _t23d = _e23b.command_manager.get_term("alvo_caixa")
    _ids23 = _t23.arange(_e23b.num_envs)
    _t23d.forca_avanco(_ids23)            # -> CARREGAR
    _t23d.forca_avanco(_ids23)            # -> BOTAR
    check("12. antes do fecho do BOTAR o publicado é BOTAR e `soltou` é falso",
          bool((_t23d.command[:, CMD.ELO] == CMD.BOTAR).all())
          and not bool(_t23d._soltou.any()))
    _t23d.forca_avanco(_ids23)            # fecha o BOTAR -> espera final
    check("12. no MESMO passo do fecho o publicado vira ANDAR, sem atraso",
          bool((_t23d.command[:, CMD.ELO] == CMD.ANDAR).all()))
    check("12. o interno fica BOTAR, `fechou` e `soltou` marcam, sucesso = 1",
          bool((_t23d._elo == CMD.BOTAR).all()) and bool(_t23d.fechou.all())
          and bool(_t23d._soltou.all())
          and float(_t23d.metrics["sucesso"].min()) == 1.0)
    check("3. e o VALIDA é UM na espera final — ela NÃO zera os incentivos",
          float(_t23d.command[:, CMD.VALIDA].min()) == 1.0,
          "spec §6.0: o VALIDA deriva do interno; é isso que fecha o buraco da renda")
    _o23b = _e23b.step(_t23.zeros(_e23b.num_envs, _n23b))[0]
    _hotf = _o23b["actor"][:, OB_.fatia_do_elo(_o23b["actor"].shape[-1])].argmax(-1)
    check("12. a observação mostra ANDAR na espera final",
          bool((_hotf == CMD.ANDAR).all()))
    check("12. `limpo_soltou` é publicado como 1,0",
          float(_e23b.limpo_soltou.min()) == 1.0)
    check("a métrica `fracao_esperando` conta a espera final",
          float(_e23b.limpo_aguardando.max()) == 0.0
          and float(MT_.fracao_esperando(_e23b).min()) == 1.0,
          "sem isto a espera final não aparece no painel")

    # --- a terminação: `escapou` DESARMADO na espera final, `caiu` ARMADO ---
    _t23d._pegou[:] = True
    _e23b.step(_t23.zeros(_e23b.num_envs, _n23b))          # publica limpo_pegou = 1
    _cx23 = _e23b.scene["box"]
    _pt = _cx23.data.root_link_pos_w.clone()
    _pt[:, 0] += 1.0                                           # 1 m à frente, mesma altura
    _cx23.write_root_link_pose_to_sim(_t23.cat([_pt, _cx23.data.root_link_quat_w], -1))
    _cx23.write_root_link_velocity_to_sim(_t23.zeros(_e23b.num_envs, 6))
    _e23b.step(_t23.zeros(_e23b.num_envs, _n23b))
    _par = dict(_c23b.terminations["caixa_largada"].params)
    _longe = TE_.caixa_largada(_e23b, **_par)
    check("12. afastar a caixa das palmas na espera final NÃO termina (escapou desarmado)",
          not bool(_longe.any()), str(_longe.tolist()))
    _pt2 = _cx23.data.root_link_pos_w.clone()
    _pt2[:, 2] = _e23b.scene.env_origins[:, 2] + 0.02            # no chão
    _cx23.write_root_link_pose_to_sim(_t23.cat([_pt2, _cx23.data.root_link_quat_w], -1))
    _cx23.write_root_link_velocity_to_sim(_t23.zeros(_e23b.num_envs, 6))
    _e23b.step(_t23.zeros(_e23b.num_envs, _n23b))
    _caiu = TE_.caixa_largada(_e23b, **_par)
    check("12. derrubar a caixa na espera final TERMINA (caiu armado)",
          bool(_caiu.all()), str(_caiu.tolist()))
    _t23d._soltou[:] = False                                    # antes do fecho...
    _cx23.write_root_link_pose_to_sim(_t23.cat([_pt, _cx23.data.root_link_quat_w], -1))
    _cx23.write_root_link_velocity_to_sim(_t23.zeros(_e23b.num_envs, 6))
    _e23b.step(_t23.zeros(_e23b.num_envs, _n23b))
    check("12. ... e ANTES do fecho afastar as palmas continua terminando (escapou armado)",
          bool(TE_.caixa_largada(_e23b, **_par).all()))
    del _e23b
except Exception as _e23x:      # noqa: BLE001
    _falhas.append(f"as duas esperas não puderam ser medidas: "
                   f"{type(_e23x).__name__}: {_e23x}")
```

Acrescente também, junto dos imports do topo do `smoke.py` (depois de `from g1_limpo.knobs import Knobs`): `from g1_limpo import metricas as MT_`. O check lê a função `MT_.fracao_esperando` direto, que é o que o `MetricsManager` chama.

- [ ] **Step 2: Rodar e ver as falhas**

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.smoke 2>&1 | grep -E '✗|falhas'
```

Esperado: falham `7.`, `20.`, `rastreio_por_elo não lê`, e o bloco 23 acusa `AttributeError: ... limpo_elo_interno` ou `_soltou`.

- [ ] **Step 3: `comando.py` — os buffers e as publicações**

No `__init__`, logo depois de `env.limpo_aguardando = torch.zeros(n, device=d)`, acrescente:

```python
        # ⚠ A ESPERA FINAL (spec §6.6): depois do fecho do BOTAR, o publicado é ANDAR
        # até o fim do episódio; o interno segue BOTAR. `soltou` desarma o `escapou` da
        # terminação e liga o `largou` da recompensa.
        self._soltou = torch.zeros(n, dtype=torch.bool, device=d)
        env.limpo_soltou = self._soltou.float()
        # ⚠ O ELO INTERNO, publicado para o crítico e para as recompensas de caixa. É
        # uma REFERÊNCIA a `_elo`, que só é escrito in-place (`self._elo[ids] = ...`),
        # portanto ela nunca fica obsoleta; `_update_command` a republica por segurança.
        env.limpo_elo_interno = self._elo
```

Troque `_aplica_espera` inteiro por:

```python
    def _aplica_espera(self) -> None:
        """Decrementa a espera e escreve o PUBLICADO e o `VALIDA` (spec §6.0).

        ⚠⚠ TUDO É RECALCULADO DO INTERNO, e não lido do próprio canal. Uma versão
        anterior fazia `where(aguardando, 0, self._command[:, VALIDA])` — DESTRUTIVO: no
        passo seguinte lia o zero que ela mesma tinha escrito, e o bit nunca voltava a 1.
        Medido no smoke em 02/09.

            publicado = ANDAR   se aguardando ∨ soltou, senão o interno
            VALIDA    = (interno ≠ ANDAR) ∧ ¬aguardando

        ⚠ A espera FINAL (`soltou`) publica ANDAR mas NÃO zera o VALIDA: os incentivos
        do estado "caixa apoiada no alvo" continuam pagando depois do fecho do BOTAR. É
        o que fecha o buraco da renda (spec §6.6.1). A v12 dizia o contrário e estava
        errada.

        ⚠ Publica `env.limpo_aguardando` e `env.limpo_soltou` para as métricas e para a
        terminação. Sem elas, "o robô não espera" e "a janela não existe" leem igual.
        """
        self._espera.sub_(self._env.step_dt).clamp_(min=0.0)
        aguardando = self._espera > 0.0
        self._env.limpo_aguardando.copy_(aguardando.float())
        self._env.limpo_soltou = self._soltou.float()
        self._env.limpo_elo_interno = self._elo
        publica_andar = aguardando | self._soltou
        self._command[:, ELO] = torch.where(
            publica_andar, torch.full_like(self._elo, ANDAR), self._elo).float()
        base = (self._elo != ANDAR).float()
        self._command[:, VALIDA] = base * (~aguardando).float()
```

Em `_publica_pegou`, troque `if tocou is not None:\n            self._pegou |= tocou` por:

```python
        if tocou is not None:
            # ⚠ SÓ ARMA COM O OBJETIVO ATIVO (spec §6.3). Na espera inicial um toque
            # por exploração armaria `escapou` com as palmas longe, e o episódio
            # morreria por ter esperado. Lê `_espera` direto, e não o `VALIDA`, porque
            # este método roda ANTES de `_aplica_espera` na passada.
            ativo = (self._elo != ANDAR) & (self._espera <= 0.0)
            self._pegou |= tocou & ativo
```

Em `_avanca_elo_force`, troque o bloco `f = ids[tem & ~pode]` por:

```python
        f = ids[tem & ~pode]
        if len(f):
            self.fechou[f] = True
            self.metrics["sucesso"][f] = 1.0
            # ⚠ A ESPERA FINAL (spec §6.6): quem fecha no BOTAR publica ANDAR daqui até
            # o fim do episódio, NO MESMO PASSO do fecho — sem esperar o `_aplica_espera`
            # do passo seguinte. O interno segue BOTAR.
            solta = f[self._elo[f] == BOTAR]
            if len(solta):
                self._soltou[solta] = True
                self._command[solta, ELO] = float(ANDAR)
```

No `_resample_command`, logo depois de `self._pegou[env_ids] = False`, acrescente `self._soltou[env_ids] = False`.

No `recebe_tarefa`, troque a última linha `self._command[ids, VALIDA] = 0.0` por:

```python
        self._command[ids, VALIDA] = 0.0
        # ⚠ E O PUBLICADO JÁ NASCE ANDAR (spec §6.4): a espera acabou de ser armada, e o
        # `_aplica_espera` só a veria no passo seguinte.
        self._soltou[ids] = False
        self._command[ids, ELO] = float(ANDAR)
```

- [ ] **Step 4: `terminacoes.py` — o guarda**

Em `caixa_largada`, troque as duas últimas linhas por:

```python
    escapou = (dist > dist_max).all(dim=-1)
    # ⚠ O GUARDA DA ESPERA FINAL (spec §6.6.3): depois do fecho do BOTAR as mãos TÊM de
    # sair da caixa — `escapou` dispararia por fazer a coisa certa. `caiu` continua
    # armado: largar é permitido, derrubar não.
    soltou = getattr(env, "limpo_soltou", None)
    if soltou is not None:
        escapou = escapou & (soltou < 0.5)
    return (caiu | escapou) & (pegou > 0.5)
```

- [ ] **Step 5: `metricas.py` — a fração conta as duas esperas**

Troque o corpo de `fracao_esperando` por:

```python
    v = getattr(env, "limpo_aguardando", None)
    if v is None:
        return torch.zeros(env.num_envs, device=env.device)
    # ⚠ AS DUAS ESPERAS (spec §6.4): a inicial (`aguardando`) e a final (`soltou`), que
    # são os passos em que um episódio de manipulação publica ANDAR.
    s = getattr(env, "limpo_soltou", None)
    if s is None:
        return v
    return torch.clamp(v + s, max=1.0)
```

E no docstring dela troque a primeira linha por `"""1 enquanto o env publica ANDAR dentro de um episódio de manipulação: a espera inicial ou a final. Por env.`.

- [ ] **Step 6: `recompensas.py` — a limpeza do rastreio**

Em `rastreio_por_elo`, apague as cinco linhas:

```python
    # ⚠ `getattr` com default, e não acesso direto: o atributo nasce no `__init__` do
    # termo de comando, e um módulo montado sem ele (ou o inspetor, que zera a janela)
    # tem de ler "ninguém aguarda" em vez de estourar.
    aguardando = getattr(env, "limpo_aguardando", None)
    if aguardando is not None:
        anda = anda | (aguardando > 0.5)
```

e no docstring troque o parágrafo que começa em `⚠⚠ A JANELA DE ESPERA CONTA COMO ELO QUE ANDA (02/09).` até o fim do parágrafo `⚠ O RISCO RESIDUAL, declarado: ...` por:

```
    ⚠ A ESPERA NÃO PRECISA DE LINHA PRÓPRIA (desde a v2, spec §6.3): o comando publica
    `ANDAR` durante as duas esperas, portanto `_anda_neste_elo` já devolve verdadeiro
    ali e o rastreio paga por manter velocidade zero — que é o contrato "fique parado,
    e ainda não existe tarefa". A versão de 02/09 lia `limpo_aguardando` aqui porque o
    publicado ainda era `PEGAR`; ficou redundante e saiu.
```

- [ ] **Step 7: Rodar o smoke até verde**

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.smoke 2>&1 | tail -6
```

Esperado: `0 falhas`. ⚠ Checks antigos que podem acusar e o que fazer: na seção 16b, `"no PRIMEIRO passo de um elo de manipulação o VALIDA é ZERO"` segue verdadeira; `"os dois track_* PAGAM durante a janela"` segue verdadeira (o publicado é ANDAR). Se `16c` (entrega no viewer) acusar `VALIDA` ou `ELO`, é porque `recebe_tarefa` agora escreve `ELO = ANDAR`: leia o check e ajuste a expectativa para "publicado ANDAR e interno PEGAR no prazo" — o comportamento novo é o certo.

- [ ] **Step 8: Commit**

```bash
cd /home/joaobornelli/Documents/g1_training && git -c core.hooksPath=/dev/null add g1_limpo/comando.py g1_limpo/terminacoes.py g1_limpo/metricas.py g1_limpo/recompensas.py g1_limpo/smoke.py && git -c core.hooksPath=/dev/null commit -m "feat(limpo): as duas esperas publicam ANDAR; o VALIDA segue derivado do interno

Spec §6.0, §6.3, §6.6. O one-hot publicado e ANDAR enquanto aguardando ou
soltou; o interno nao se move. Depois do fecho do BOTAR, soltou marca no
mesmo passo, o publicado vira ANDAR e o VALIDA fica 1: os incentivos do
estado apoiado continuam pagando. _pegou so arma com o objetivo ativo.
caixa_largada: escapou desarmado por soltou, caiu armado. fracao_esperando
conta as duas esperas. rastreio_por_elo perde a leitura de aguardando,
agora redundante. Smoke: secao 23."
```

---

### Task 3: Tamanho da caixa por mundo, e `caiu` por tamanho

Spec §6.7. Evento de startup `tamanho_caixa`: sorteia um dos K meio-lados por env, escreve `geom_size`, `geom_rbound`, `geom_aabb` por mundo (o caminho de DR do próprio mjlab), publica `env.limpo_meia_aresta`. Todo consumidor do tamanho lê dali. `caixa_z_min` vira `caixa_folga_chao`.

**Files:**
- Modify: `g1_limpo/knobs.py` (`Cena`, `Terminacao`)
- Modify: `g1_limpo/eventos.py` (novo `tamanho_caixa`; `posiciona_cena`; `afasta_cena`; `__all__`)
- Modify: `g1_limpo/comando.py` (`_meia`, `alvos_das_palmas`, ramo `BOTAR`, `_laje_para`)
- Modify: `g1_limpo/terminacoes.py` (`caixa_largada`)
- Modify: `g1_limpo/env_cfg.py` (evento; params da terminação)
- Modify: `g1_limpo/smoke.py` (linha ~308; linha ~1844; seção nova 24)

**Interfaces:**
- Produces: `env.limpo_meia_aresta: Tensor[n, 3]` (metros, os três eixos iguais); `AlvoCaixaCmd._meia(ids) -> Tensor[len(ids), 3]`; `eventos.tamanho_caixa(env, env_ids, *, faixa, n_variantes, asset_cfg)`; `terminacoes.caixa_largada(env, folga_chao, dist_max, meia_aresta_ref)`.
- Consumes: Task 2 (`limpo_soltou`).

- [ ] **Step 1: Escrever os checks que falham**

Troque o check da linha ~308:

```python
check("o `caixa_z_min` é a MEIA-ARESTA — a caixa apoiada no chão",
      k.terminacao.caixa_z_min == k.cena.caixa_meia_aresta[2],
      "é o piso físico, e não uma tolerância escolhida")
```
por
```python
check("o `caiu` lê o TAMANHO da caixa: a folga do chão é menor que a laje mais baixa",
      0.0 < k.terminacao.caixa_folga_chao < k.cena.prateleira_topo_piso,
      f"folga {k.terminacao.caixa_folga_chao} vs piso da laje {k.cena.prateleira_topo_piso}")
```

Troque o check da linha ~1844:

```python
    check("os dois alvos ficam nas FACES laterais — separados por 2×meia-aresta",
          float((_sep5 - 2.0 * k.cena.caixa_meia_aresta[1]).abs().max()) < 1e-5,
          f"separação medida {float(_sep5.mean()):.4f} m")
```
por
```python
    check("os dois alvos ficam nas FACES laterais — separados por 2×meia-aresta DO ENV",
          float((_sep5 - 2.0 * _e5.limpo_meia_aresta[:, 1]).abs().max()) < 1e-5,
          f"separação medida {float(_sep5.mean()):.4f} m")
```

E acrescente no fim (antes da impressão final):

```python
# ==================== 24. o TAMANHO da caixa por mundo, e o `caiu` por tamanho (spec §6.7)
secao("24. tamanho da caixa por mundo")
from g1_limpo import eventos as EV_                                       # noqa: E402

check("13. o evento `tamanho_caixa` existe, é de STARTUP e declara os três campos",
      "tamanho_caixa" in cfg.events and cfg.events["tamanho_caixa"].mode == "startup"
      and tuple(getattr(EV_.tamanho_caixa, "model_fields", ()))
      == ("geom_size", "geom_rbound", "geom_aabb"),
      "sem `requires_model_fields` o mjlab não expande os campos por mundo")
check("13. a faixa e o K são os da spec",
      tuple(k.cena.caixa_meia_aresta_faixa) == (0.07, 0.13) and k.cena.caixa_n_variantes == 8)
try:
    import torch as _t24

    _c24 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _c24.scene.num_envs = 64
    _e24 = ManagerBasedRlEnv(cfg=_c24, device="cpu")
    _e24.reset()
    _n24 = _e24.action_manager.total_action_dim
    _cx = _e24.scene["box"]
    _loc, _ = _cx.find_geoms([C.BOX_GEOM])
    _g = int(_cx.indexing.geom_ids[_loc[0]])
    _size = _e24.sim.model.geom_size[:, _g]                      # (64, 3)
    _a = _size[:, 0]
    _K = _t24.linspace(*k.cena.caixa_meia_aresta_faixa, k.cena.caixa_n_variantes)
    _no_k = (_a.unsqueeze(-1) - _K.unsqueeze(0)).abs().min(-1).values < 1e-6
    check("13. `geom_size` difere entre mundos e só toma os K valores",
          bool(_no_k.all()) and len(_t24.unique(_a)) >= 5,
          f"{len(_t24.unique(_a))} valores distintos em 64 envs: {sorted(set(round(float(x),4) for x in _a))}")
    check("13. a caixa é CUBO: os três eixos iguais",
          float((_size - _a.unsqueeze(-1)).abs().max()) < 1e-7)
    check("13. `geom_rbound` acompanha: a·√3",
          float((_e24.sim.model.geom_rbound[:, _g] - _a * math.sqrt(3.0)).abs().max()) < 1e-6)
    check("13. `geom_aabb` acompanha: meia-caixa (a, a, a)",
          float((_e24.sim.model.geom_aabb[:, _g, 1] - _size).abs().max()) < 1e-7)
    _bm = _e24.sim.model.body_mass
    _bid = int(_cx.indexing.body_ids[0])
    check("13. `body_mass` da caixa NÃO mudou — independência do peso",
          float((_bm[..., _bid] - float(k.cena.caixa_massa)).abs().max()) < 1e-6,
          f"{_bm[..., _bid].flatten()[:4].tolist()}")
    check("13. `limpo_meia_aresta` bate com `geom_size` env a env",
          float((_e24.limpo_meia_aresta - _size).abs().max()) < 1e-7)
    # o colisor LÊ o tamanho: a caixa repousa com o centro a `a` acima do topo
    _passa_janela(_e24, _n24, _t24)
    _rep = (_cx.data.root_link_pos_w[:, 2] - _e24.limpo_topo - _a)
    check("13. a caixa repousa a `a` acima da laje em TODO env — o colisor lê o tamanho novo",
          float(_rep.abs().max()) < 5e-3,
          f"desvio máximo {float(_rep.abs().max())*1000:.1f} mm")
    # 15. todo consumidor lê o tamanho por env
    _t24c = _e24.command_manager.get_term("alvo_caixa")
    _alv = _t24c.alvos_das_palmas(_t24.arange(_e24.num_envs))
    _sep = (_alv[:, 0] - _alv[:, 1]).norm(dim=-1)
    check("15. `alvos_das_palmas` separa as palmas por 2a DO ENV",
          float((_sep - 2.0 * _a).abs().max()) < 1e-5)
    # 19. o `caiu` por tamanho
    _t24c._pegou[:] = True
    _e24.step(_t24.zeros(_e24.num_envs, _n24))
    _par24 = dict(_c24.terminations["caixa_largada"].params)
    _q = _cx.data.root_link_quat_w
    _pf = _cx.data.root_link_pos_w.clone()
    _pf[:, 2] = _e24.scene.env_origins[:, 2] + _a                # deitada no chão
    _cx.write_root_link_pose_to_sim(_t24.cat([_pf, _q], -1))
    _cx.write_root_link_velocity_to_sim(_t24.zeros(_e24.num_envs, 6))
    _e24.step(_t24.zeros(_e24.num_envs, _n24))
    check("19. deitada no chão, a caixa de QUALQUER tamanho dispara `caiu`",
          bool(TE_.caixa_largada(_e24, **_par24).all()))
    _pl = _pf.clone()
    _pl[:, 2] = _e24.scene.env_origins[:, 2] + k.cena.prateleira_topo_piso + _a
    _cx.write_root_link_pose_to_sim(_t24.cat([_pl, _q], -1))
    _cx.write_root_link_velocity_to_sim(_t24.zeros(_e24.num_envs, 6))
    _e24.step(_t24.zeros(_e24.num_envs, _n24))
    check("19. apoiada na laje mais baixa, a caixa MENOR não dispara `caiu`",
          not bool(TE_.caixa_largada(_e24, **_par24).any()))
    del _e24
except Exception as _e24x:      # noqa: BLE001
    _falhas.append(f"o tamanho por mundo não pôde ser medido: "
                   f"{type(_e24x).__name__}: {_e24x}")
```

- [ ] **Step 2: Rodar e ver as falhas**

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.smoke 2>&1 | grep -E '✗|falhas'
```

Esperado: falha em `o caiu lê o TAMANHO` (AttributeError `caixa_folga_chao`), `13.` e o bloco 24 com exceção.

- [ ] **Step 3: `knobs.py`**

Em `class Cena`, logo depois de `caixa_meia_aresta: tuple[float, float, float] = (0.10, 0.10, 0.10)`, acrescente:

```python
    # ⚠ DR DE TAMANHO desde a FASE 1 (spec §6.7, decisão do dono 02/09). K meio-lados
    # discretos, cubo, sorteados por env UMA vez no startup e escritos por mundo em
    # `geom_size` + `geom_rbound` + `geom_aabb` pelo caminho de DR do mjlab. O
    # `caixa_meia_aresta` acima segue sendo o spec de REFERÊNCIA (paridade, inspetor) e
    # a variante do meio da faixa.
    caixa_meia_aresta_faixa: tuple[float, float] = (0.07, 0.13)
    caixa_n_variantes: int = 8
```

Em `class Terminacao`, troque o campo `caixa_z_min` e o seu docstring por:

```python
    caixa_folga_chao: float = 0.02
    """Folga, em metros, do FUNDO da caixa ao chão abaixo da qual ela CAIU.
    ⚠ Substitui `caixa_z_min = 0,10`, que era a meia-aresta de UMA caixa. Com o tamanho
    variando (spec §6.7), o limiar fixo não acusava a queda da caixa de 0,13 m (centro
    em 0,13) e ficava a 1 cm de acusar a de 0,07 m na laje a 0,04 m. Agora
    `caiu = z_centro − meia_aresta_env < folga`: "o fundo está a menos de 2 cm do chão".
    Menor que `prateleira_topo_piso` (0,04), senão a laje mais baixa dispararia."""
```

- [ ] **Step 4: `eventos.py` — o evento de startup e os dois consumidores**

Acrescente aos imports do topo:

```python
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
```

Acrescente `"tamanho_caixa"` a `__all__`. Acrescente esta função logo antes de `carga_caixa`:

```python
@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def tamanho_caixa(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    *,
    faixa: tuple[float, float],
    n_variantes: int,
    asset_cfg: SceneEntityCfg,
) -> None:
    """DR de TAMANHO da caixa: K meio-lados discretos, por mundo, UMA vez no startup.

    Spec §6.7. É o caminho de DR do próprio mjlab: `requires_model_fields` faz o
    `load_managers` expandir os três campos para `(nworld, ngeom)` ANTES dos eventos de
    startup e do primeiro `forward`, e o kernel de broadphase do `mujoco_warp` nasce já
    indexando por mundo. O `mjlab.envs.mdp.dr.geom_size` faz isto para um box, mas
    sorteia cada eixo de forma independente; aqui a caixa é CUBO, portanto a escrita é
    própria e as duas fórmulas do box são repetidas (`rbound = a·√3`, `aabb_half = a`).

    ⚠ `body_mass` e `body_inertia` NÃO são tocados: a independência do peso vem daí, e a
    inércia fica a da caixa de 0,10 m — inconsistência declarada, do mesmo tipo da que
    `carga_caixa` já aceita.

    Publica `env.limpo_meia_aresta` (n, 3). Todo consumidor do tamanho lê dali:
    `comando._meia`, `posiciona_cena`, `afasta_cena`, `terminacoes.caixa_largada`,
    `observacoes.caixa_no_frame_da_base`.
    """
    dev = env.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=dev, dtype=torch.int)
    else:
        env_ids = env_ids.to(dev, dtype=torch.int)
    n = len(env_ids)
    valores = torch.linspace(float(faixa[0]), float(faixa[1]), int(n_variantes), device=dev)
    a = valores[torch.randint(int(n_variantes), (n,), device=dev)]           # (n,)

    caixa: Entity = env.scene[asset_cfg.name]
    gid = caixa.indexing.geom_ids[asset_cfg.geom_ids]                        # (1,)
    env_grid, geom_grid = torch.meshgrid(env_ids, gid, indexing="ij")
    tam = a.unsqueeze(-1).unsqueeze(-1).expand(n, len(gid), 3)
    env.sim.model.geom_size[env_grid, geom_grid] = tam
    env.sim.model.geom_rbound[env_grid, geom_grid] = (a * math.sqrt(3.0)).unsqueeze(-1)
    env.sim.model.geom_aabb[env_grid, geom_grid, 1] = tam

    if not hasattr(env, "limpo_meia_aresta"):
        env.limpo_meia_aresta = torch.full((env.num_envs, 3), float(faixa[0]), device=dev)
    env.limpo_meia_aresta[env_ids.long()] = a.unsqueeze(-1).expand(n, 3)
```

Em `posiciona_cena`, troque `pose_caixa[:, 2] = topo + caixa_meia_z` por:

```python
    # ⚠ o tamanho é POR ENV (spec §6.7); o knob é só o fallback de um env montado sem o
    # evento `tamanho_caixa` (não existe no pacote, mas o fallback é explícito)
    meia = getattr(env, "limpo_meia_aresta", None)
    meia_z = meia[env_ids, 2] if meia is not None else torch.full((n,), caixa_meia_z, device=dev)
    pose_caixa[:, 2] = topo + meia_z
```

Em `afasta_cena`, troque `pose_c[:, 2] = afasta_z + caixa_meia_z    # apoiada na laje erguida` por:

```python
    meia = getattr(env, "limpo_meia_aresta", None)
    meia_z = meia[env_ids, 2] if meia is not None else torch.full((n,), caixa_meia_z, device=dev)
    pose_c[:, 2] = afasta_z + meia_z          # apoiada na laje erguida
```

- [ ] **Step 5: `comando.py` — os quatro sítios**

Acrescente este método logo antes de `alvos_das_palmas`:

```python
    def _meia(self, ids: torch.Tensor) -> torch.Tensor:
        """[k, 3] — a meia-aresta da caixa DE CADA ENV (spec §6.7).

        Lê `env.limpo_meia_aresta`, publicado pelo evento de startup `tamanho_caixa`. O
        knob `caixa_meia_aresta` é só o fallback de um env montado sem o evento.
        """
        meia = getattr(self._env, "limpo_meia_aresta", None)
        if meia is not None:
            return meia[ids]
        return torch.full((len(ids), 3), float(self.cfg.caixa_meia_aresta), device=self.device)
```

Em `alvos_das_palmas`, troque `off[:, 1] = self.cfg.caixa_meia_aresta` por `off[:, 1] = self._meia(ids)[:, 1]`.

No ramo `BOTAR` de `_aplica_elo`, troque `fundo = self.caixa.data.root_link_pos_w[m, 2] - c.caixa_meia_z` por `fundo = self.caixa.data.root_link_pos_w[m, 2] - self._meia(m)[:, 2]`, e `a[:, 2] = topo + c.caixa_meia_z` por `a[:, 2] = topo + self._meia(m)[:, 2]`.

Em `_laje_para`, troque `pc[:, 2] = topo_t + c.caixa_meia_z` por `pc[:, 2] = topo_t + self._meia(ids)[:, 2]`.

- [ ] **Step 6: `terminacoes.py` — `caiu` por tamanho**

Troque a assinatura e o `caiu`:

```python
def caixa_largada(env: "ManagerBasedRlEnv", folga_chao: float,
                  dist_max: float, meia_aresta_ref: float) -> torch.Tensor:
```
e
```python
    caiu = (caixa[:, 2] - env.scene.env_origins[:, 2]) < z_min
```
por
```python
    # ⚠ POR TAMANHO (spec §6.7): "o fundo da caixa está a menos de `folga_chao` do chão".
    # Com o limiar fixo de 0,10 a caixa de 0,13 m deitada no chão nunca acusava queda.
    meia = getattr(env, "limpo_meia_aresta", None)
    meia_z = meia[:, 2] if meia is not None else torch.full_like(caixa[:, 2], meia_aresta_ref)
    caiu = (caixa[:, 2] - env.scene.env_origins[:, 2] - meia_z) < folga_chao
```

No docstring da função acrescente uma linha: `⚠ Limitação declarada: uma caixa que cai TOMBADA sobre uma aresta tem o centro em a·√2 e escapa ao caiu; fora da espera final o escapou a pega.`

- [ ] **Step 7: `env_cfg.py` — o evento e os params**

Logo antes de `cfg.events["posiciona_cena"] = EventTermCfg(`, acrescente:

```python
    # ⚠ DR DE TAMANHO (spec §6.7): startup, por mundo, K meio-lados, cubo. É o caminho
    # de DR do próprio mjlab (`requires_model_fields` expande os campos antes do primeiro
    # forward). Vem ANTES do `posiciona_cena` só por leitura; startup e reset são modos
    # separados.
    cfg.events["tamanho_caixa"] = EventTermCfg(
        func=EV.tamanho_caixa, mode="startup",
        params={"faixa": c.caixa_meia_aresta_faixa,
                "n_variantes": c.caixa_n_variantes,
                "asset_cfg": SceneEntityCfg("box", geom_names=(C.BOX_GEOM,))},
    )
```

Troque os params de `caixa_largada`:

```python
        params={"z_min": k.terminacao.caixa_z_min,
                "dist_max": k.terminacao.caixa_dist_max})
```
por
```python
        params={"folga_chao": k.terminacao.caixa_folga_chao,
                "dist_max": k.terminacao.caixa_dist_max,
                "meia_aresta_ref": c.caixa_meia_aresta[2]})
```

- [ ] **Step 8: Rodar o smoke até verde, e a paridade**

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.smoke 2>&1 | tail -6 && .venv/bin/python -m g1_limpo.paridade 2>&1 | tail -3
```

Esperado: `0 falhas` e a paridade igual à da Task 0 (o spec de referência não mudou). ⚠ Se o check `13. a caixa repousa a a acima da laje` falhar com desvio de ~30 mm em alguns envs, o colisor NÃO leu o tamanho: confira que `cfg.events["tamanho_caixa"].func.model_fields` existe e que `env.sim.model.geom_size.shape[0] == num_envs` depois do reset. Se `shape[0] == 1`, o `requires_model_fields` não foi visto pelo `EventManager` — o decorador tem de estar na função referenciada em `func=`, não num wrapper.

- [ ] **Step 9: Commit**

```bash
cd /home/joaobornelli/Documents/g1_training && git -c core.hooksPath=/dev/null add g1_limpo/knobs.py g1_limpo/eventos.py g1_limpo/comando.py g1_limpo/terminacoes.py g1_limpo/env_cfg.py g1_limpo/smoke.py && git -c core.hooksPath=/dev/null commit -m "feat(limpo): tamanho da caixa por mundo no startup, e caiu por tamanho

Spec §6.7. Evento tamanho_caixa (requires_model_fields) escreve geom_size,
geom_rbound e geom_aabb por mundo, K=8 meio-lados de 0,07 a 0,13 m, cubo,
e publica limpo_meia_aresta. Todo consumidor le dali: alvos_das_palmas, o
ramo BOTAR, _laje_para, posiciona_cena, afasta_cena. caixa_z_min vira
caixa_folga_chao = 0,02: o fundo da caixa a menos de 2 cm do chao. Smoke:
secao 24 prova que o colisor le o tamanho (a caixa repousa a `a` acima da
laje em todo env) e que caiu dispara para a caixa grande no chao."
```

---

### Task 4: A observação — gate, `giro_b`, `meia_aresta`, `VALIDA` fora, crítico com o interno

Spec §4, §4.1, §6.1, §6.2, §8.3 (contrato). O comando ganha o canal `GIRO` (mundo); a observação publica `[caixa_b, alvo_b, giro_b, meia_aresta]` gateados pelo publicado; o crítico ganha `elo_interno`; o `PPOPorElo` agrupa por ele.

**Files:**
- Modify: `g1_limpo/comando.py` (layout; `_atualiza_face`; `__all__`)
- Modify: `g1_limpo/observacoes.py` (tudo)
- Modify: `g1_limpo/env_cfg.py` (obs do crítico)
- Modify: `g1_limpo/algoritmo.py` (`compute_returns`)
- Modify: `g1_limpo/smoke.py` (9b `fatia_do_elo(112)`; 16 `_NOSSA_OBS`; 18 "canais DEPOIS do one-hot"; seção nova 25)

**Interfaces:**
- Produces: `comando.GIRO = slice(9, 12)`, `comando.DIM = 12`; `observacoes.N_CAIXA = 10`; `observacoes.um_de_cinco_interno(env, command_name)`; `observacoes.fatia_do_elo_interno(dim_total) -> slice`.
- Consumes: Task 2 (`limpo_elo_interno`), Task 3 (`limpo_meia_aresta`).

- [ ] **Step 1: Escrever os checks que falham**

Na seção 9b troque

```python
check("`fatia_do_elo` devolve o penúltimo bloco, de N_SLOTS canais",
      _OB.fatia_do_elo(112) == slice(99, 104)
      and _OB.fatia_do_elo(200) == slice(200 - _OB.N_CAIXA - _OB.N_SLOTS,
                                        200 - _OB.N_CAIXA),
      f"em 112 devolveu {_OB.fatia_do_elo(112)}")
```
por
```python
check("`fatia_do_elo` devolve o penúltimo bloco do ATOR, de N_SLOTS canais",
      _OB.fatia_do_elo(114) == slice(99, 104)
      and _OB.fatia_do_elo(200) == slice(200 - _OB.N_CAIXA - _OB.N_SLOTS,
                                        200 - _OB.N_CAIXA),
      f"em 114 devolveu {_OB.fatia_do_elo(114)}")
check("`fatia_do_elo_interno` devolve o ÚLTIMO bloco do CRÍTICO",
      _OB.fatia_do_elo_interno(119) == slice(114, 119))
check("o `PPOPorElo` agrupa pelo elo INTERNO do crítico, não pelo publicado do ator",
      'observations["critic"]' in inspect.getsource(ALG.PPOPorElo.compute_returns)
      and "fatia_do_elo_interno" in inspect.getsource(ALG.PPOPorElo.compute_returns),
      "spec §6.1: a espera final tem retorno de manipulação com one-hot de ANDAR")
```

Na seção 16 troque `_NOSSA_OBS = ["elo", "caixa"]` e o check `"os NOSSOS vêm depois..."` por:

```python
_NOSSA_OBS = {"actor": ["elo", "caixa"], "critic": ["elo", "caixa", "elo_interno"]}
check("os NOSSOS vêm depois, na ordem das fases; o crítico ganha `elo_interno` no fim",
      all(nossa_obs[g][len(fab_obs[g]):] == _NOSSA_OBS[g]
          for g in ("actor", "critic")),
      "append de colunas; inserir no meio desloca todo peso da 1ª camada em silêncio")
```

Na seção 18 troque o check `"os canais da caixa entram DEPOIS do one-hot, nos dois grupos"` por:

```python
check("os canais da caixa entram DEPOIS do one-hot, nos dois grupos",
      list(cfg.observations["actor"].terms)[-2:] == ["elo", "caixa"]
      and list(cfg.observations["critic"].terms)[-3:] == ["elo", "caixa", "elo_interno"],
      str(list(cfg.observations["actor"].terms)))
```

E acrescente no fim:

```python
# ======== 25. a OBSERVAÇÃO: gate, giro_b, meia_aresta, VALIDA fora, crítico (spec §4, §6.1)
secao("25. a observação nova")
check("3. `N_CAIXA` é 10: caixa_b(3) alvo_b(3) giro_b(3) meia_aresta(1)", OB_.N_CAIXA == 10)
check("3. o VALIDA NÃO está na observação",
      "VALIDA" not in inspect.getsource(OB_.caixa_no_frame_da_base))
check("o canal GIRO é o ÚLTIMO do comando (append), e DIM é 12",
      CMD.GIRO == slice(9, 12) and CMD.DIM == 12)
try:
    import torch as _t25
    from mjlab.utils.lab_api.math import quat_mul as _qmul

    # --- dimensões ---
    _c25 = make_env_cfg(k, inspecao=True, elo=CMD.ANDAR)
    _c25.scene.num_envs = 16
    _e25 = ManagerBasedRlEnv(cfg=_c25, device="cpu")
    _o25, _ = _e25.reset()
    _n25 = _e25.action_manager.total_action_dim
    check("3. o ator tem 114 canais e o crítico 119",
          _o25["actor"].shape[-1] == 114 and _o25["critic"].shape[-1] == 119,
          f"ator {_o25['actor'].shape[-1]}, crítico {_o25['critic'].shape[-1]}")
    _int = _o25["critic"][:, OB_.fatia_do_elo_interno(119)]
    check("o `elo_interno` do crítico é um one-hot",
          bool(_t25.allclose(_int.sum(-1), _t25.ones(16))))
    # --- 1. o gate: caixa PERTO e publicado ANDAR -> os 10 canais são zero ---
    _cx25 = _e25.scene["box"]
    _p25 = _e25.scene["robot"].data.root_link_pos_w.clone()
    _p25[:, 0] += 0.5
    for _ in range(3):
        _cx25.write_root_link_pose_to_sim(_t25.cat([_p25, _cx25.data.root_link_quat_w], -1))
        _cx25.write_root_link_velocity_to_sim(_t25.zeros(16, 6))
        _o25 = _e25.step(_t25.zeros(16, _n25))[0]
    _cx_slice_a = _o25["actor"][:, 114 - OB_.N_CAIXA:114]
    _cx_slice_c = _o25["critic"][:, 119 - 5 - OB_.N_CAIXA:119 - 5]
    check("1. com a caixa a 0,5 m e o publicado em ANDAR, os 10 canais são EXATAMENTE zero (ator)",
          float(_cx_slice_a.abs().max()) == 0.0, f"máximo {float(_cx_slice_a.abs().max())}")
    check("1. ... e no crítico também",
          float(_cx_slice_c.abs().max()) == 0.0)
    del _e25

    # --- 2. a invariante, e 3. meia_aresta, na borda da espera ---
    _c25b = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR)
    _c25b.scene.num_envs = 16
    _e25b = ManagerBasedRlEnv(cfg=_c25b, device="cpu")
    _o25b, _ = _e25b.reset()
    _n25b = _e25b.action_manager.total_action_dim
    _fat = OB_.fatia_do_elo(114)
    _cxs = slice(114 - OB_.N_CAIXA, 114)
    _viol = False
    _borda_ok = False
    _hot_ant = _o25b["actor"][:, _fat].argmax(-1)
    _cx_ant = _o25b["actor"][:, _cxs]
    check("4. no reset: publicado ANDAR e canais de caixa zero",
          bool((_hot_ant == CMD.ANDAR).all()) and float(_cx_ant.abs().max()) == 0.0)
    for _ in range(int(k.alvo.espera_s[1] / _e25b.step_dt) + 5):
        _o25b = _e25b.step(_t25.zeros(16, _n25b))[0]
        _hot = _o25b["actor"][:, _fat].argmax(-1)
        _cxo = _o25b["actor"][:, _cxs]
        _norma = _cxo[:, :3].norm(dim=-1)
        _viol |= bool(((_hot == CMD.ANDAR) & (_norma > 0)).any())
        _viol |= bool(((_hot != CMD.ANDAR) & (_norma == 0)).any())
        _borda_ok |= bool(((_hot_ant == CMD.ANDAR) & (_hot == CMD.PEGAR) & (_norma > 0.1)).any())
        _hot_ant = _hot
    check("2. NUNCA existe |caixa_b| = 0 com publicado ≠ ANDAR, nem ≠ 0 com ANDAR", not _viol)
    check("4. na borda os canais ACENDEM no mesmo passo em que o one-hot vira PEGAR", _borda_ok)
    _meia_obs = _o25b["actor"][:, 113]
    check("3. o último canal é `meia_aresta` e bate com `limpo_meia_aresta` env a env",
          float((_meia_obs - _e25b.limpo_meia_aresta[:, 0]).abs().max()) < 1e-6,
          f"{_meia_obs[:4].tolist()} vs {_e25b.limpo_meia_aresta[:4, 0].tolist()}")
    # --- 23. giro_b: em PEGAR a face está CONGELADA -> zero na abertura, e cresce ao torcer
    _t25c = _e25b.command_manager.get_term("alvo_caixa")
    _giro0 = _o25b["actor"][:, 114 - 4:114 - 1]
    check("23. em PEGAR, na abertura, giro_b é ~0 (face congelada)",
          float(_giro0.norm(dim=-1).max()) < 2e-2, f"{float(_giro0.norm(dim=-1).max()):.4f}")
    _cx25b = _e25b.scene["box"]
    _ang = math.radians(20.0)
    # ⚠ a torção é RELATIVA ao quatérnion da abertura (a face está congelada nele): a
    # caixa nasce com um desalinho de até ±15°, portanto um yaw ABSOLUTO de 20° não daria
    # |giro| = 20°. Compõe-se `qz(20°) ⊗ q0`.
    _q0 = _cx25b.data.root_link_quat_w.clone()
    _qz = _qmul(_t25.tensor([math.cos(_ang / 2), 0.0, 0.0, math.sin(_ang / 2)]).expand(16, 4), _q0)
    for _ in range(3):
        _cx25b.write_root_link_pose_to_sim(_t25.cat([_cx25b.data.root_link_pos_w, _qz], -1))
        _cx25b.write_root_link_velocity_to_sim(_t25.zeros(16, 6))
        _o25b = _e25b.step(_t25.zeros(16, _n25b))[0]
    _giro1 = _o25b["actor"][:, 114 - 4:114 - 1]
    check("23. torcida 20° em Z, |giro_b| ≈ 0,35 e bate com ANG",
          float((_giro1.norm(dim=-1) - _t25c.command[:, CMD.ANG]).abs().max()) < 1e-4
          and abs(float(_giro1.norm(dim=-1).mean()) - _ang) < 0.03,
          f"|giro| {float(_giro1.norm(dim=-1).mean()):.3f}, ANG {float(_t25c.command[:, CMD.ANG].mean()):.3f}")
    check("23. ... e o eixo é Z", float(_giro1[:, :2].abs().max()) < 0.05)
    del _e25b

    # --- 23. giro_b no REORIENTAR: direção VIVA; caixa girada 90° em Z pede giro em Z ---
    # ⚠ SEM jitter em y na caixa: a direção viva é "da caixa para o robô", e com a caixa
    # deslocada em y ela deixa de ser −X puro — o eixo do giro do tombo ganharia uma
    # componente em x e o ângulo do giro em Z deixaria de ser 90°. Com dy = 0 os dois
    # casos são exatos.
    _kk25 = dataclasses.replace(k, cena=dataclasses.replace(k.cena, caixa_jitter_y=(0.0, 0.0)))
    _c25c = make_env_cfg(_kk25, inspecao=True, elo=CMD.REORIENTAR)
    _c25c.scene.num_envs = 8
    _e25c = ManagerBasedRlEnv(cfg=_c25c, device="cpu")
    _e25c.reset()
    _n25c = _e25c.action_manager.total_action_dim
    _cx25c = _e25c.scene["box"]
    _t25d = _e25c.command_manager.get_term("alvo_caixa")

    def _giro_com(quat):
        for _ in range(int(k.alvo.espera_s[1] / _e25c.step_dt) + 5):
            _cx25c.write_root_link_pose_to_sim(
                _t25.cat([_cx25c.data.root_link_pos_w, quat.expand(8, 4)], -1))
            _cx25c.write_root_link_velocity_to_sim(_t25.zeros(8, 6))
            _o = _e25c.step(_t25.zeros(8, _n25c))[0]
        return _o["actor"][:, 114 - 4:114 - 1]

    _h = math.pi / 4
    _g_mais = _giro_com(_t25.tensor([math.cos(_h), 0.0, 0.0, math.sin(_h)]))   # yaw +90°
    _g_menos = _giro_com(_t25.tensor([math.cos(_h), 0.0, 0.0, -math.sin(_h)]))  # yaw −90°
    _g_pitch = _giro_com(_t25.tensor([math.cos(_h), 0.0, math.sin(_h), 0.0]))   # pitch +90°
    check("23. caixa girada 90° em Z: |giro_b| ≈ π/2 e o eixo é Z",
          abs(float(_g_mais.norm(dim=-1).mean()) - math.pi / 2) < 0.05
          and float(_g_mais[:, :2].abs().max()) < 0.1,
          f"{_g_mais[0].tolist()}")
    check("23. o SINAL troca com o sentido do giro",
          float((_g_mais[:, 2] * _g_menos[:, 2]).max()) < 0.0)
    check("23. caixa tombada 90° em Y: o eixo é Y",
          abs(float(_g_pitch[:, 1].abs().mean()) - math.pi / 2) < 0.05
          and float(_g_pitch[:, [0, 2]].abs().max()) < 0.1,
          f"{_g_pitch[0].tolist()}")
    del _e25c
except Exception as _e25x:      # noqa: BLE001
    _falhas.append(f"a observação nova não pôde ser medida: "
                   f"{type(_e25x).__name__}: {_e25x}")
```

- [ ] **Step 2: Rodar e ver as falhas**

Esperado: falham 9b (`fatia_do_elo(114)` devolve `slice(101, 106)` com `N_CAIXA = 8`), 16, 18, e a seção 25 (`GIRO` não existe, dimensão 112).

- [ ] **Step 3: `comando.py` — o canal `GIRO`**

Troque o bloco do layout por:

```python
# --- o layout, por nome. Nenhum índice solto no resto do pacote. ---
ALVO = slice(0, 3)
FACE = slice(3, 6)
ANG = 6
VALIDA = 7
ELO = 8
# ⚠ O VETOR DE GIRO (spec §8.3), em MUNDO: eixo × ângulo da rotação que leva a normal
# ATUAL da face pedida à direção pedida. `|GIRO| == ANG`. A observação o leva ao frame
# da base (`giro_b`). Entrou POR ÚLTIMO, como manda o contrato de append.
GIRO = slice(9, 12)
DIM = 12
```

Atualize o docstring do módulo (a lista `[0:3] ALVO ... [8] ELO`) acrescentando a linha `[9:12] GIRO    eixo × ângulo do giro pedido, em MUNDO (spec §8.3)`. Acrescente `"GIRO"` a `__all__`.

Em `_atualiza_face`, troque as duas últimas linhas

```python
        cos = (normal_w * desejada).sum(-1).clamp(-1.0, 1.0)
        self._command[ids, ANG] = torch.acos(cos)
```
por
```python
        cos = (normal_w * desejada).sum(-1).clamp(-1.0, 1.0)
        ang = torch.acos(cos)
        self._command[ids, ANG] = ang
        # ⚠ O VETOR DE GIRO (spec §8.3): eixo `normal × desejada` normalizado, vezes o
        # ângulo. Diz para que LADO girar, o que o escalar `ANG` não diz — sem ele um MLP
        # sem memória não aprende a reorientar. Antiparalelo (ang ≈ π) não tem eixo
        # definido: usa-se Z, que é "meia volta pivotando na laje". Em ang ≈ 0 o
        # produto vetorial some e `where` põe Z também — vezes zero, dá zero.
        eixo = torch.cross(normal_w, desejada, dim=-1)
        norma = eixo.norm(dim=-1, keepdim=True)
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(eixo)
        eixo = torch.where(norma > 1e-6, eixo / norma.clamp(min=1e-6), ez)
        self._command[ids, GIRO] = eixo * ang.unsqueeze(-1)
```

- [ ] **Step 4: `observacoes.py` — a observação nova**

Troque `N_CAIXA = 8` por:

```python
# caixa_b(3) alvo_b(3) giro_b(3) meia_aresta(1). Spec §4.1.
N_CAIXA = 10
```

Acrescente a `__all__`: `"um_de_cinco_interno", "fatia_do_elo_interno"`. Acrescente depois de `fatia_do_elo`:

```python
def fatia_do_elo_interno(dim_total: int) -> slice:
    """A fatia do one-hot do elo INTERNO dentro da observação do CRÍTICO.

    É o ÚLTIMO bloco: o `env_cfg` o acrescenta depois de `caixa`, só no grupo `critic`
    (spec §6.1). O `PPOPorElo` agrupa por ele.
    """
    return slice(dim_total - N_SLOTS, dim_total)
```

Acrescente depois de `um_de_cinco`:

```python
def um_de_cinco_interno(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    """O one-hot do elo INTERNO — só para o CRÍTICO (spec §6.1).

    ⚠ NÃO ENTRA NO ATOR: em campo ninguém o manda. O crítico o recebe porque a espera
    final rende ~18/s e um env `standing` da locomoção rende 6/s com a MESMA observação
    de ator; sem o interno a função de valor confunde os dois. Ator-crítico assimétrico.
    `aguardando` e `soltou` não precisam de canal: são `interno ≠ publicado`.
    """
    elo = env.command_manager.get_term(command_name)._elo.clamp(0, N_SLOTS - 1)
    return torch.nn.functional.one_hot(elo, num_classes=N_SLOTS).float()
```

Troque `caixa_no_frame_da_base` inteira por:

```python
def caixa_no_frame_da_base(env, command_name: str) -> torch.Tensor:
    """Os canais da caixa, TODOS no frame da base. 10 canais, GATEADOS (spec §4.1, §6.1).

        [0:3]  caixa − base, no frame da base
        [3:6]  alvo  − base, no frame da base
        [6:9]  giro_b: eixo × ângulo do giro pedido, no frame da base (spec §8.3)
        [9]    meia_aresta: o meio-lado da caixa deste env, em metros (spec §6.7)

    ⚠⚠ O GATE: quando o elo PUBLICADO é `ANDAR`, os 10 canais são ZERO — mesmo com a
    caixa a 0,5 m. Não existe terceiro estado. É a invariante que substitui o bit
    `VALIDA`, que SAIU da observação (spec §6.2). Sem o gate a política aprendia "ando"
    da distância da caixa (5 m no ANDAR), e sambava em campo com a caixa perto (§5).
    Lê o PUBLICADO (`comando[:, ELO]`), e não o interno: é o que o operador manda.

    ⚠ TUDO NO FRAME DA BASE, e não em mundo. Coordenada de mundo carrega a ORIGEM DO
    ENV, diferente em cada um dos 4096, e o rumo. No frame da base o problema é o mesmo
    em todo env.

    ⚠ O σ NÃO ENTRA AQUI: ele diz "este env é fácil ou difícil", e a política
    condicionaria a ação à forma da RECOMPENSA em vez de à tarefa.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse

    from g1_limpo.comando import ALVO, ANDAR, ELO, GIRO

    cmd = env.command_manager.get_command(command_name)
    robo = env.scene["robot"]
    p, q = robo.data.root_link_pos_w, robo.data.root_link_quat_w
    caixa_b = quat_apply_inverse(q, env.scene["box"].data.root_link_pos_w - p)
    alvo_b = quat_apply_inverse(q, cmd[:, ALVO] - p)
    giro_b = quat_apply_inverse(q, cmd[:, GIRO])
    meia = getattr(env, "limpo_meia_aresta", None)
    assert meia is not None, ("o evento `tamanho_caixa` não publicou `limpo_meia_aresta`; "
                              "um canal constante em zero envenena o normalizador")
    canais = torch.cat([caixa_b, alvo_b, giro_b, meia[:, :1]], dim=-1)
    vivo = (cmd[:, ELO].long() != ANDAR).float().unsqueeze(-1)
    return canais * vivo
```

Atualize o docstring de `fatia_do_elo`: `MEDIDO: com a observação de ator em 114, isto devolve slice(99, 104)`.

- [ ] **Step 5: `env_cfg.py` — o crítico**

Logo depois do laço `for grupo in ("actor", "critic"): cfg.observations[grupo].terms["caixa"] = ...`, acrescente:

```python
    # ------------------------------------------- 3h. o elo INTERNO, só no crítico (v2)
    # ⚠ SÓ NO `critic`, e POR ÚLTIMO. Spec §6.1: a espera final rende ~18/s e um env
    # `standing` rende 6/s com a mesma observação de ator; o crítico precisa separá-los.
    # Não vai para o robô. O `PPOPorElo` agrupa por esta fatia (`fatia_do_elo_interno`).
    cfg.observations["critic"].terms["elo_interno"] = ObservationTermCfg(
        func=OB.um_de_cinco_interno, params={"command_name": "alvo_caixa"},
    )
```

- [ ] **Step 6: `algoritmo.py` — agrupar pelo interno**

Troque `from g1_limpo.observacoes import fatia_do_elo` por `from g1_limpo.observacoes import fatia_do_elo_interno` e, em `compute_returns`, troque

```python
        atores = st.observations["actor"]
        bloco = atores[..., fatia_do_elo(atores.shape[-1])]
```
por
```python
        # ⚠ O ELO INTERNO DO CRÍTICO, e não o publicado do ator (spec §6.1): nas duas
        # esperas o ator vê ANDAR, mas a espera final carrega retorno de MANIPULAÇÃO —
        # agrupá-la com a locomoção inflaria o desvio da locomoção, que é exatamente o
        # defeito que esta classe existe para evitar.
        criticos = st.observations["critic"]
        bloco = criticos[..., fatia_do_elo_interno(criticos.shape[-1])]
```

E na mensagem do `assert` troque `alguém acrescentou observação DEPOIS do canal caixa?` por `alguém acrescentou observação ao crítico DEPOIS do elo_interno?`.

- [ ] **Step 7: Rodar o smoke até verde**

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.smoke 2>&1 | tail -6
```

Esperado: `0 falhas`. ⚠ Se algum check antigo citar `112` (grep: `grep -n '112' g1_limpo/smoke.py`), troque para `114` e leia o contexto: a seção 18 tem um check de dimensão da observação. Se o check `23. caixa girada 90° em Z ... o eixo é Z` falhar com eixo em Y, confira a convenção do quatérnion do mjlab (`w, x, y, z`) na escrita da pose — o teste escreve `(w, 0, 0, sin)` para yaw.

- [ ] **Step 8: Commit**

```bash
cd /home/joaobornelli/Documents/g1_training && git -c core.hooksPath=/dev/null add g1_limpo/comando.py g1_limpo/observacoes.py g1_limpo/env_cfg.py g1_limpo/algoritmo.py g1_limpo/smoke.py && git -c core.hooksPath=/dev/null commit -m "feat(limpo): observacao v2 — gate pelo publicado, giro_b, meia_aresta, VALIDA fora, critico com elo interno

Spec §4, §6.1, §6.2, §8.3. Os 10 canais de caixa sao zero quando o
publicado e ANDAR, mesmo com a caixa a 0,5 m. ANG (1) vira giro_b (3):
eixo x angulo do giro pedido, no frame da base — diz para que lado
girar. meia_aresta entra no fim do termo caixa. Ator 114, critico 119: o
critico ganha o one-hot do elo interno e o PPOPorElo agrupa por ele.
Smoke: secao 25 (gate com a caixa perto, invariante, borda, giro_b em Z e
em Y, sinal)."
```

---

### Task 5: A renda do `BOTAR` é monótona

Spec §6.6.1, §6.6.2. Máscaras de `squeeze` e `unload` no `BOTAR`; `alcança ≡ 1` no `BOTAR` ou `soltou`; termos novos `load` (2,0) e `largou` (1,0). Pairar < apoiar < fechar < largar.

**Files:**
- Modify: `g1_limpo/knobs.py` (`Tarefa`)
- Modify: `g1_limpo/recompensas.py` (`_alcancar`, `squeeze`, `unload`, novos `load` e `largou`, `__all__`)
- Modify: `g1_limpo/env_cfg.py` (dois termos)
- Modify: `g1_limpo/smoke.py` (seção 18 "soma dos pesos"; seção nova 26)

**Interfaces:**
- Produces: `recompensas.load(env, nome_do_comando, sensor_apoio, raio_mult)`; `recompensas.largou(env, nome_do_comando, sensor_apoio, raio_mult, sigma_solta)`; `recompensas._elo_interno(env, nome)`.
- Consumes: Task 2 (`limpo_soltou`, `_elo`), Task 3 (nada direto).

- [ ] **Step 1: Escrever os checks que falham**

Na seção 18, o check `"a soma dos pesos é 11,5/s"` continua (os SETE somam 11,5). Acrescente logo depois dele:

```python
check("os dois termos do BOTAR existem: `load` = 2,0 e `largou` = 1,0 (spec §6.6.2)",
      cfg.rewards["load"].weight == 2.0 and cfg.rewards["largou"].weight == 1.0,
      str({n: cfg.rewards[n].weight for n in ("load", "largou") if n in cfg.rewards}))
```

E acrescente no fim:

```python
# ============ 26. a RENDA DO BOTAR é MONÓTONA: pairar < apoiar < fechar < largar (spec §6.6)
secao("26. a renda do BOTAR")
check("16. o `load` é o espelho do `unload`: só no BOTAR, gateado por `perto`",
      "== BOTAR" in inspect.getsource(RC_.load) and "raio_mult" in inspect.getsource(RC_.load))
check("17. `squeeze` e `unload` são MASCARADOS no BOTAR (o precedente do g1_poc)",
      "!= BOTAR" in inspect.getsource(RC_.squeeze)
      and "!= BOTAR" in inspect.getsource(RC_.unload))
check("17. `alcança ≡ 1` no BOTAR ou em `soltou`",
      "== BOTAR" in inspect.getsource(RC_._alcancar)
      and "limpo_soltou" in inspect.getsource(RC_._alcancar))
try:
    import torch as _t26

    _c26 = make_env_cfg(k, inspecao=True, elo=CMD.PEGAR, cadeia=3)
    _c26.scene.num_envs = 8
    _e26 = ManagerBasedRlEnv(cfg=_c26, device="cpu")
    _e26.reset()
    _n26 = _e26.action_manager.total_action_dim
    _passa_janela(_e26, _n26, _t26)
    _t26c = _e26.command_manager.get_term("alvo_caixa")
    _ids26 = _t26.arange(8)
    _nm26 = list(_c26.rewards)
    _cx26 = _e26.scene["box"]
    _q26 = _cx26.data.root_link_quat_w.clone()

    def _renda(passos=4, alvo_dz=None):
        """Soma dos termos por segundo, com a caixa mantida em alvo + dz (ou livre)."""
        for _ in range(passos):
            if alvo_dz is not None:
                _p = _t26c.command[:, CMD.ALVO].clone()
                _p[:, 2] += alvo_dz
                _cx26.write_root_link_pose_to_sim(_t26.cat([_p, _q26], -1))
                _cx26.write_root_link_velocity_to_sim(_t26.zeros(8, 6))
            _e26.step(_t26.zeros(8, _n26))
        _sr = _e26.reward_manager._step_reward
        return (float(_sr.mean(0).sum()),
                {n: float(_sr[:, _nm26.index(n)].mean())
                 for n in ("staged", "precise_pos", "load", "largou", "unload", "squeeze",
                           "postura_ereta", "track_linear_velocity", "pose")})

    # 17. alcança no PEGAR com a caixa longe é ~0; no BOTAR é 1
    _pl = _cx26.data.root_link_pos_w.clone()
    _pl[:, 0] += 1.0
    for _ in range(3):
        _cx26.write_root_link_pose_to_sim(_t26.cat([_pl, _q26], -1))
        _cx26.write_root_link_velocity_to_sim(_t26.zeros(8, 6))
        _e26.step(_t26.zeros(8, _n26))
    _alc_pegar = float(RC_._alcancar(_e26, "alvo_caixa").max())
    _t26c.forca_avanco(_ids26)            # -> CARREGAR
    _t26c.forca_avanco(_ids26)            # -> BOTAR (laje nova sob a caixa; alvo lateral)
    _e26.step(_t26.zeros(8, _n26))
    _alc_botar = float(RC_._alcancar(_e26, "alvo_caixa").min())
    check("17. `alcança` < 0,1 no PEGAR com a caixa a 1 m, e == 1 no BOTAR na mesma pose",
          _alc_pegar < 0.1 and _alc_botar == 1.0, f"pegar {_alc_pegar:.3f}, botar {_alc_botar:.3f}")
    # 17. as máscaras, com uma força de palma FINGIDA (o robô pinado não aperta nada)
    _orig = RC_._forca_das_palmas
    RC_._forca_das_palmas = lambda env, sensores, asset_cfg: _t26.full((env.num_envs,), 20.0)
    try:
        _sq_botar = float(RC_.squeeze(_e26, "alvo_caixa", C.SENSOR_PALMA, k.tarefa.squeeze_mu,
                                      cfg.rewards["squeeze"].params["asset_cfg"]).max())
        _t26c._elo[:] = CMD.PEGAR
        _sq_pegar = float(RC_.squeeze(_e26, "alvo_caixa", C.SENSOR_PALMA, k.tarefa.squeeze_mu,
                                      cfg.rewards["squeeze"].params["asset_cfg"]).min())
        _t26c._elo[:] = CMD.BOTAR
    finally:
        RC_._forca_das_palmas = _orig
    check("17. com a mesma força de palma, `squeeze` é 0 no BOTAR e > 0 no PEGAR",
          _sq_botar == 0.0 and _sq_pegar > 0.5, f"botar {_sq_botar:.3f}, pegar {_sq_pegar:.3f}")

    # A: pairar 2 cm acima do alvo, sem apoio
    _rA, _dA = _renda(passos=6, alvo_dz=0.02)
    check("18. pairando, `load` é 0", abs(_dA["load"]) < 1e-6, f"{_dA['load']:.4f}")
    check("17. pairando no BOTAR, `unload` e `postura_ereta` são 0 (mascarados)",
          abs(_dA["unload"]) < 1e-9 and abs(_dA["postura_ereta"]) < 1e-9)
    # C: apoiada no alvo (solta a caixa sobre a laje e deixa assentar 5 passos)
    _pC = _t26c.command[:, CMD.ALVO].clone()
    _cx26.write_root_link_pose_to_sim(_t26.cat([_pC, _q26], -1))
    _cx26.write_root_link_velocity_to_sim(_t26.zeros(8, 6))
    _rC, _dC = _renda(passos=5)
    check("18. apoiada no alvo, `load` é ~1 × 2,0", _dC["load"] > 1.6, f"{_dC['load']:.3f}")
    check("12. e ainda NÃO fechou (0,3 s de sustain)", not bool(_t26c.fechou.any()))
    # apoiada a 25 cm do alvo -> load 0
    _pD = _pC.clone()
    _pD[:, 1] += 0.25
    _cx26.write_root_link_pose_to_sim(_t26.cat([_pD, _q26], -1))
    _cx26.write_root_link_velocity_to_sim(_t26.zeros(8, 6))
    _rD, _dD = _renda(passos=4)
    check("18. apoiada a 25 cm do alvo, `load` é 0 — o gate de posição", abs(_dD["load"]) < 1e-6)
    # prensada com o dobro do peso -> load segue 1 (clamp)
    _pE = _pC.clone()
    _pE[:, 2] -= 0.004
    _cx26.write_root_link_pose_to_sim(_t26.cat([_pE, _q26], -1))
    _cx26.write_root_link_velocity_to_sim(_t26.zeros(8, 6))
    _t26c._sust[:] = 0.0
    _rE, _dE = _renda(passos=3, alvo_dz=-0.004)
    _F = float(_t26.norm(_e26.scene[C.SENSOR_APOIO].data.force, dim=-1).mean())
    _mg = float((_e26.limpo_massa * 9.81).mean())
    check("18. prensada (F > m·g), `load` continua 1 — o `clamp`",
          _F > 1.2 * _mg and _dE["load"] > 1.8, f"F/mg {_F/_mg:.2f}, load {_dE['load']:.3f}")
    # ⚠ se `F/mg` sair abaixo de 1,2, a penetração de 4 mm não bastou para este `solref`:
    # aumente o `−0.004` dos dois lugares acima para `−0.008`. O que o check afirma é o
    # `clamp`, e não o valor da força.
    # volta a apoiar no alvo e deixa o BOTAR FECHAR sozinho -> espera final
    _cx26.write_root_link_pose_to_sim(_t26.cat([_pC, _q26], -1))
    _cx26.write_root_link_velocity_to_sim(_t26.zeros(8, 6))
    _rC2, _dC2 = _renda(passos=5)
    for _ in range(int(k.cadeia.sustenta_outros_s / _e26.step_dt) + 3):
        _e26.step(_t26.zeros(8, _n26))
    check("12. apoiada no alvo o BOTAR FECHA sozinho e `soltou` marca",
          bool(_t26c.fechou.all()) and bool(_t26c._soltou.all()))
    _rF, _dF = _renda(passos=4)
    check("18. na espera final, com as palmas longe, `largou` ≥ 0,95 × 1,0",
          _dF["largou"] >= 0.95, f"{_dF['largou']:.3f}")
    check("16. a RENDA É MONÓTONA: pairar < apoiada < espera final (palmas longe)",
          _rA < _rC2 < _rF,
          f"pairar {_rA:.2f}  apoiada {_rC2:.2f}  espera final {_rF:.2f}  (/s)")
    check("16. o rastreio entra na espera final e não antes",
          _dC2["track_linear_velocity"] == 0.0 and _dF["track_linear_velocity"] > 1.0)
    print(f"  renda do BOTAR: pairar {_rA:.2f}  apoiada {_rC2:.2f}  espera final {_rF:.2f} /s")
    del _e26
except Exception as _e26x:      # noqa: BLE001
    _falhas.append(f"a renda do BOTAR não pôde ser medida: "
                   f"{type(_e26x).__name__}: {_e26x}")
```

- [ ] **Step 2: Rodar e ver as falhas**

Esperado: falham `os dois termos do BOTAR existem`, `16.`, `17.` (fontes) e a seção 26 com `KeyError: 'load'`.

- [ ] **Step 3: `knobs.py` — `Tarefa`**

Logo depois de `sustentacao: float = 0.5       # ficou lá`, acrescente:

```python
    # ⚠ A RENDA DO BOTAR (spec §6.6.2, decisão do dono 03/09). Sem estes dois, pairar a
    # caixa a 1 cm da laje rendia 16,5/s e apoiá-la 12,5/s: o BOTAR não fechava. O
    # `g1_poc` tinha `load` + as máscaras de `squeeze`/`unload` e a reescrita as perdeu.
    load: float = 2.0              # clamp(F_apoio/m·g) × perto — o espelho do `unload`, só no BOTAR
    largou: float = 1.0            # soltou × load × (1 − exp(−(d_palma/σ_solta)²)) — tirar as mãos
    load_raio_mult: float = 2.0    # `perto` do load = d ≤ raio_mult × tol_pos (g1_poc: 2)
    sigma_solta: float = 0.10      # m; palmas a 10 cm rendem 63%, a 20 cm 98%
```

E no docstring de `Tarefa` acrescente: `Mais `load` (2,0) e `largou` (1,0), só no BOTAR e na espera final — spec §6.6.2.`

- [ ] **Step 4: `recompensas.py` — as peças**

Acrescente `"load", "largou"` a `__all__`. Acrescente logo depois de `_valida`:

```python
def _elo_interno(env, nome: str) -> torch.Tensor:
    """O elo INTERNO do termo de comando (spec §6.0): o que paga lê o interno."""
    return env.command_manager.get_term(nome)._elo


def _fora_do_botar(env, nome: str) -> torch.Tensor:
    """1 fora do BOTAR, 0 nele. A máscara do g1_poc para `squeeze` e `unload`."""
    from g1_limpo.comando import BOTAR
    return (_elo_interno(env, nome) != BOTAR).float()
```

Troque `_alcancar` inteira por:

```python
def _alcancar(env, nome: str) -> torch.Tensor:
    """`exp(−(d_palma/σ_alcance)²)`. O kernel de aproximação da mão.

    No passo em que o elo abre ele vale `exp(−1) = 0,368` por construção, porque
    `σ = d₀`. MEDIDO: 0,3679 a 0,3708 em 32 envs.

    ⚠ `alcança ≡ 1` no BOTAR e na espera final (`soltou`) — spec §6.6.2 item 3, §8.3.
    No BOTAR as mãos já estão na caixa: σ cai no piso de 0,08 m e o kernel vale 1 por
    construção; ele não carrega informação ali, só paga 3/s por MANTER as mãos na caixa,
    que é o freio contra largar. Com `≡ 1`, `staged` vira `3 × (1 + trazer)` e
    `precise_ori` vira `alinha`: pagam pela caixa, indiferentes às mãos.
    """
    from g1_limpo.comando import BOTAR
    t = _t(env, nome)
    ids = torch.arange(env.num_envs, device=t.sigma_alcance.device)
    d = t.dist_palma_caixa(ids)
    kernel = torch.exp(-(d / t.sigma_alcance.clamp(min=1e-6)) ** 2)
    um = t._elo == BOTAR
    soltou = getattr(env, "limpo_soltou", None)
    if soltou is not None:
        um = um | (soltou > 0.5)
    return torch.where(um, torch.ones_like(kernel), kernel)
```

Em `squeeze`, troque o `return` por:

```python
    # ⚠ ZERO NO BOTAR (spec §6.6.2 item 2; g1_poc: "apertar durante o botar paga contra
    # soltar, −1,0/s medido"). Pagar por segurar é pagar contra a tarefa de largar.
    return (torch.tanh(f / _forca_ref(env, mu)) * _valida(env, nome_do_comando)
            * _fora_do_botar(env, nome_do_comando))
```

Em `unload`, troque o `return` por:

```python
    # ⚠ ZERO NO BOTAR (spec §6.6.2 item 2; g1_poc: "ligado no botar, pagaria 2,0/s para
    # NÃO botar"). `postura_ereta` é `rampa × unload` e zera junto, sem linha própria.
    return (descarga * preensao * _valida(env, nome_do_comando)
            * _fora_do_botar(env, nome_do_comando))
```

Acrescente no fim do arquivo (antes de `class sustentacao` ou depois dela):

```python
def load(env, nome_do_comando: str, sensor_apoio: str, raio_mult: float) -> torch.Tensor:
    """`clamp(F_apoio/m·g) × perto` — o espelho do `unload`, SÓ no `BOTAR` (spec §6.6.2).

    Paga quando a laje carrega o peso da caixa perto do alvo. Sem ele o BOTAR não tinha
    quem pagasse por apoiar: a condição de fecho exige `apoiada` e nenhuma recompensa a
    pagava — pairar rendia mais (spec §6.6.1). É a peça do `g1_poc`
    (`g1_poc/recompensas.py::load`), com o mesmo peso e o mesmo gate de posição.

    ⚠ `clamp` em 1: prensar a caixa contra a laje não rende mais. ⚠ Gate de posição
    `d ≤ raio_mult × tol_pos`: fecha o hack de largar a caixa em qualquer lugar do tampo.
    ⚠ SEM gate de preensão: soltar É o objetivo. ⚠ Continua pagando na espera final,
    porque o interno segue BOTAR — é isso que torna fechar melhor do que pairar.
    """
    from g1_limpo.comando import BOTAR
    t = _t(env, nome_do_comando)
    f = torch.norm(env.scene[sensor_apoio].data.force, dim=-1).squeeze(-1)
    peso = (env.limpo_massa * 9.81).clamp(min=1e-6)
    fracao = (f / peso).clamp(0.0, 1.0)
    perto = (_dist_caixa_alvo(env, nome_do_comando) <= raio_mult * t.cfg.tol_pos).float()
    no_botar = (t._elo == BOTAR).float()
    return fracao * perto * no_botar * _valida(env, nome_do_comando)


def largou(env, nome_do_comando: str, sensor_apoio: str, raio_mult: float,
           sigma_solta: float) -> torch.Tensor:
    """`soltou × load × (1 − exp(−(d_palma/σ_solta)²))` — tirar as mãos (spec §6.6.2).

    Paga por afastar as palmas da caixa APOIADA no alvo, só na espera final. Sem ele a
    espera final não ensinava a largar: `pose` em `standing` (σ 0,05) vale zero com os
    braços fora e `action_rate` paga por não mover — o ótimo era congelar com as mãos na
    caixa. Como `_alcancar ≡ 1` tirou o freio, um peso pequeno basta.
    """
    soltou = getattr(env, "limpo_soltou", None)
    if soltou is None:
        return torch.zeros(env.num_envs, device=env.device)
    t = _t(env, nome_do_comando)
    d = t.dist_palma_caixa(torch.arange(env.num_envs, device=env.device))
    longe = 1.0 - torch.exp(-(d / max(sigma_solta, 1e-6)) ** 2)
    return ((soltou > 0.5).float() * longe
            * load(env, nome_do_comando, sensor_apoio, raio_mult))
```

- [ ] **Step 5: `env_cfg.py` — os dois termos**

Logo depois do bloco `cfg.rewards["sustentacao"] = RewardTermCfg(...)`, acrescente:

```python
    # ------------------------------------------- 3i. a renda do BOTAR (v2, spec §6.6.2)
    # ⚠ Os DOIS termos que faltavam para o BOTAR fechar. `load` é o espelho do `unload`,
    # só no BOTAR; `largou` paga por tirar as mãos na espera final. Ambos leem o elo
    # INTERNO e o VALIDA. O smoke prova a monotonia da renda (§11.1 item 16).
    cfg.rewards["load"] = RewardTermCfg(
        func=RC.load, weight=tr.load,
        params={"nome_do_comando": _cmd, "sensor_apoio": C.SENSOR_APOIO,
                "raio_mult": tr.load_raio_mult})
    cfg.rewards["largou"] = RewardTermCfg(
        func=RC.largou, weight=tr.largou,
        params={"nome_do_comando": _cmd, "sensor_apoio": C.SENSOR_APOIO,
                "raio_mult": tr.load_raio_mult, "sigma_solta": tr.sigma_solta})
```

- [ ] **Step 6: Rodar o smoke até verde**

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.smoke 2>&1 | tail -8
```

Esperado: `0 falhas` e a linha `renda do BOTAR: pairar ~10..12  apoiada ~12..14  espera final ~17..19 /s`. ⚠ Se `apoiada` sair ABAIXO de `pairar`, o `load` não está pagando: confira que `_sust` não fechou o BOTAR antes do tempo (a check 12 acusa) e que `F_apoio` ≥ 0,9·mg no estado C (imprima `_F`). Se `12. apoiada no alvo o BOTAR FECHA sozinho` falhar, confira `alinhado`: o teste mantém o quatérnion `_q26` da abertura, portanto `ANG ≈ 0`; e `perto` com `tol_pos = 0,10` no alvo exato.

- [ ] **Step 7: Commit**

```bash
cd /home/joaobornelli/Documents/g1_training && git -c core.hooksPath=/dev/null add g1_limpo/knobs.py g1_limpo/recompensas.py g1_limpo/env_cfg.py g1_limpo/smoke.py && git -c core.hooksPath=/dev/null commit -m "feat(limpo): a renda do BOTAR vira monotona — load, largou, mascaras, alcanca == 1

Spec §6.6.1, §6.6.2. Sem isto pairar a caixa a 1 cm da laje rendia mais
que apoia-la, e a espera final da v12 alargava a queda: o BOTAR nunca
fecharia. Peças com precedente no g1_poc: squeeze e unload zeram no
BOTAR (postura_ereta zera junto); alcanca vale 1 no BOTAR e em soltou;
load = clamp(F_apoio/mg) x perto, peso 2,0, so no BOTAR e na espera
final; largou = soltou x load x (1 - exp(-(d/0,10)^2)), peso 1,0. Smoke:
secao 26 mede pairar < apoiada < espera final com o robo travado."
```

---

### Task 6: Girar no lugar — `is_turning_env` com precedência explícita

Spec §9. Um ramo novo no sorteador do twist: `lin = 0` todo passo, `|wz| ≥ 0,2`, fora do `heading`. Precedência `standing > turning > forward > heading`.

**Files:**
- Modify: `g1_limpo/knobs.py` (`Marcha`)
- Modify: `g1_limpo/comando.py` (`TwistComRazaoDeMarcha`, `TwistComRazaoDeMarchaCfg`)
- Modify: `g1_limpo/env_cfg.py` (passa os dois knobs)
- Modify: `g1_limpo/smoke.py` (seção nova 27)

**Interfaces:**
- Produces: `TwistComRazaoDeMarcha.is_turning_env: BoolTensor[n]`; cfg fields `rel_turning_envs`, `turning_wz_min`.

- [ ] **Step 1: Escrever os checks que falham**

```python
# ==================== 27. o RAMO DE GIRO no sorteador do twist (spec §9)
secao("27. girar no lugar")
check("21. os knobs do giro são os da spec",
      k.marcha.rel_turning_envs == 0.10 and k.marcha.turning_wz_min == 0.2)
check("21. o cfg do twist recebe os dois",
      cfg.commands["twist"].rel_turning_envs == 0.10
      and cfg.commands["twist"].turning_wz_min == 0.2)
try:
    import torch as _t27

    _c27 = make_env_cfg(k, elo=CMD.ANDAR)          # cfg de TREINO, elo forçado
    _c27.scene.num_envs = 512
    _e27 = ManagerBasedRlEnv(cfg=_c27, device="cpu")
    _e27.reset()
    _n27 = _e27.action_manager.total_action_dim
    _tw27 = _e27.command_manager.get_term("twist")
    _todos = _t27.arange(512)
    _cont = {"turning": 0, "standing": 0, "forward": 0, "heading": 0, "n": 0}
    _ok_wz = _ok_lin = _ok_heading = True
    for _ in range(8):
        _tw27._resample_command(_todos)
        _tu = _tw27.is_turning_env
        _cont["turning"] += int(_tu.sum()); _cont["n"] += 512
        _cont["standing"] += int(_tw27.is_standing_env.sum())
        _cont["forward"] += int(_tw27.is_forward_env.sum())
        _cont["heading"] += int(_tw27.is_heading_env.sum())
        if bool(_tu.any()):
            _ok_wz &= bool((_tw27.vel_command_b[_tu, 2].abs() >= k.marcha.turning_wz_min - 1e-6).all())
            _ok_lin &= float(_tw27.vel_command_b[_tu, :2].abs().max()) == 0.0
            _ok_heading &= not bool(_tw27.is_heading_env[_tu].any())
    _frac = _cont["turning"] / _cont["n"]
    check("21. a fração REALIZADA de turning é 0,09 ± 0,02 (0,10 × 0,90, fora do standing)",
          abs(_frac - 0.09) < 0.02, f"{_frac:.3f} em {_cont['n']} sorteios")
    check("21. |wz| ≥ 0,2 em todo env turning", _ok_wz)
    check("21. lin = 0 em todo env turning, no resample", _ok_lin)
    check("21. nenhum env turning está em heading", _ok_heading)
    check("21. o standing continua ~0,10 — o turning não o comeu",
          abs(_cont["standing"] / _cont["n"] - 0.10) < 0.02,
          f"{_cont['standing']/_cont['n']:.3f}")
    # lin continua ZERO passo a passo, e wz NÃO é reescrito pelo heading
    _tu = _tw27.is_turning_env.clone()
    _wz0 = _tw27.vel_command_b[:, 2].clone()
    for _ in range(3):
        _e27.step(_t27.zeros(512, _n27))
    check("21. lin = 0 nos envs turning em TODO passo",
          float(_tw27.vel_command_b[_tu, :2].abs().max()) == 0.0)
    check("21. e wz dos envs turning não muda entre passos (fora do heading)",
          float((_tw27.vel_command_b[_tu, 2] - _wz0[_tu]).abs().max()) < 1e-6)
    del _e27
except Exception as _e27x:      # noqa: BLE001
    _falhas.append(f"o ramo de giro não pôde ser medido: {type(_e27x).__name__}: {_e27x}")
```

- [ ] **Step 2: Rodar e ver as falhas**

Esperado: falham `21. os knobs` (AttributeError) e a seção com exceção.

- [ ] **Step 3: `knobs.py` — `Marcha`**

Depois de `pedido_min_segmento: float = 0.5` acrescente:

```python
    # ⚠ GIRAR NO LUGAR (spec §9, decisão do dono 02/09). O sorteador do fabricante quase
    # nunca produz `lin ≈ 0 ∧ wz ≠ 0`: standing não gira, forward não pode, heading gira
    # ANDANDO. Um ramo `turning`: lin = 0 todo passo, |wz| ≥ turning_wz_min. A precedência
    # é `standing > turning > forward > heading`, portanto a fração REALIZADA é
    # rel_turning_envs × (1 − rel_standing_envs) ≈ 0,09.
    rel_turning_envs: float = 0.10
    turning_wz_min: float = 0.2
```

- [ ] **Step 4: `comando.py` — o ramo**

Em `TwistComRazaoDeMarchaCfg`, depois de `pedido_min_segmento: float = 0.5` e do seu docstring, acrescente:

```python
    rel_turning_envs: float = 0.0
    """Fração dos envs que só GIRAM: `lin = 0` todo passo, `|wz| ≥ turning_wz_min`
    (spec §9). Sorteada a cada re-sorteio de comando, como as flags do fabricante.
    Precedência: `standing > turning > forward > heading`."""
    turning_wz_min: float = 0.2
```

Em `TwistComRazaoDeMarcha.__init__`, depois de `self.metrics["eficiencia_media"] = z.clone()`, acrescente:

```python
        # ⚠ O RAMO DE GIRO (spec §9). Flag por env, re-sorteada com as do fabricante.
        self.is_turning_env = torch.zeros_like(self.is_standing_env)
```

Acrescente estes dois métodos à classe, antes de `_fecha_segmento`:

```python
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # ⚠ AS FLAGS DO FABRICANTE SÃO SORTEIOS INDEPENDENTES, não uma partição (spec §9):
        # `standing` zera tudo todo passo; `heading` reescreve `wz` todo passo; `forward`
        # escreve só no resample. O `turning` entra com precedência EXPLÍCITA:
        # standing > turning > forward > heading. Ele cede ao standing (que zeraria o
        # wz todo passo), vence o forward (roda depois do super) e SAI do heading (senão
        # o heading reescreve o wz dele no passo seguinte).
        r = torch.empty(len(env_ids), device=self.device)
        turning = r.uniform_(0.0, 1.0) <= self.cfg.rel_turning_envs
        turning &= ~self.is_standing_env[env_ids]
        self.is_turning_env[env_ids] = turning
        ids = env_ids[turning]
        if len(ids) == 0:
            return
        self.is_heading_env[ids] = False
        self.is_forward_env[ids] = False
        lo, hi = self.cfg.ranges.ang_vel_z
        teto = max(abs(float(lo)), abs(float(hi)))
        mag = torch.empty(len(ids), device=self.device).uniform_(
            float(self.cfg.turning_wz_min), teto)
        sinal = torch.where(torch.rand(len(ids), device=self.device) < 0.5, -1.0, 1.0)
        self.vel_command_b[ids, 0] = 0.0
        self.vel_command_b[ids, 1] = 0.0
        self.vel_command_b[ids, 2] = sinal * mag
        self.vel_command_w[ids] = self.vel_command_b[ids]

    def _update_command(self) -> None:
        super()._update_command()
        # ⚠ `lin = 0` TODO PASSO nos envs turning, como o fabricante faz com o standing.
        ids = self.is_turning_env.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.vel_command_b[ids, :2] = 0.0
```

- [ ] **Step 5: `env_cfg.py` — passar os knobs**

Na construção `cfg.commands["twist"] = CMD.TwistComRazaoDeMarchaCfg(**campos, limiar_comando=..., pedido_min_segmento=...)`, acrescente `rel_turning_envs=k.marcha.rel_turning_envs, turning_wz_min=k.marcha.turning_wz_min`.

- [ ] **Step 6: Rodar o smoke até verde**

Esperado: `0 falhas`. ⚠ Se `21. o standing continua ~0,10` falhar, o `_zera_twist_nos_parados` não interfere (elo ANDAR), então confira que o super `_resample_command` continua sendo chamado ANTES do ramo. Se `wz dos envs turning não muda entre passos` falhar, o heading ainda reescreve: confira `self.is_heading_env[ids] = False`.

- [ ] **Step 7: Commit**

```bash
cd /home/joaobornelli/Documents/g1_training && git -c core.hooksPath=/dev/null add g1_limpo/knobs.py g1_limpo/comando.py g1_limpo/env_cfg.py g1_limpo/smoke.py && git -c core.hooksPath=/dev/null commit -m "feat(limpo): ramo de giro no lugar no sorteador do twist, com precedencia explicita

Spec §9. is_turning_env: lin = 0 em todo passo, |wz| >= 0,2, fora do
heading; rel_turning_envs = 0,10, que realiza ~0,09 fora do standing.
Precedencia standing > turning > forward > heading. As penalidades de
marcha gateiam por |lin| + |wz|, portanto girar paga foot_slip e cai no
regime walking (sem hack de arrastar o pe). Smoke: secao 27 mede as
fracoes realizadas e a estabilidade do wz."
```

---

### Task 7: O `REORIENTAR` fica inerte na run da v2

Spec §8.3. `voltas_max = 0` em todo nível e `eixo_vertical` falso: a caixa nasce dentro da tolerância e o elo fecha em 0,3 s sem trabalho. O slot continua sorteado. A tabela antiga fica em comentário para o dia em que virar foco.

**Files:**
- Modify: `g1_limpo/knobs.py` (`Nivel`)
- Modify: `g1_limpo/smoke.py` (seção nova 28)

- [ ] **Step 1: Escrever o check que falha**

```python
# ==================== 28. o REORIENTAR está INERTE na v2 (spec §8.3)
secao("28. o REORIENTAR inerte")
check("24. `voltas_max` é zero e `eixo_vertical` é falso em TODO nível",
      all(v == 0 for v in k.nivel.voltas_max) and not any(k.nivel.eixo_vertical),
      f"{k.nivel.voltas_max} / {k.nivel.eixo_vertical}")
check("o REORIENTAR CONTINUA sorteável — o slot não pode ficar constante",
      CMD.REORIENTAR in ELOS_SORTEAVEIS)
try:
    import torch as _t28

    _c28 = make_env_cfg(k, inspecao=True, elo=CMD.REORIENTAR)
    _c28.scene.num_envs = 16
    _e28 = ManagerBasedRlEnv(cfg=_c28, device="cpu")
    _e28.reset()
    _n28 = _e28.action_manager.total_action_dim
    _t28c = _e28.command_manager.get_term("alvo_caixa")
    _p0 = _e28.scene["box"].data.root_link_pos_w.clone()
    _passa_janela(_e28, _n28, _t28)
    for _ in range(int(k.cadeia.sustenta_outros_s / _e28.step_dt) + 3):
        _e28.step(_t28.zeros(16, _n28))
    _dp = (_e28.scene["box"].data.root_link_pos_w - _p0).norm(dim=-1)
    check("24. um env de cadeia 1 avança para o PEGAR em `sustenta_outros_s` sem a caixa se mover",
          bool((_t28c._elo == CMD.PEGAR).all()) and float(_dp.max()) < 0.01,
          f"elo {_t28c._elo.tolist()[:6]}, deslocamento máx {float(_dp.max())*1000:.1f} mm")
    del _e28
except Exception as _e28x:      # noqa: BLE001
    _falhas.append(f"o REORIENTAR inerte não pôde ser medido: {type(_e28x).__name__}: {_e28x}")
```

- [ ] **Step 2: Rodar e ver a falha**

Esperado: falha `24. voltas_max é zero`.

- [ ] **Step 3: `knobs.py` — `Nivel`**

Troque as duas linhas

```python
    voltas_max: tuple[int, ...] = (0, 0, 1, 1, 1, 1, 1)
    eixo_vertical: tuple[bool, ...] = (False, False, False, False, True, True, True)
```
por
```python
    # ⚠ O REORIENTAR ESTÁ INERTE NA RUN DA v2 (spec §8.3, decisão do dono 03/09): a
    # caixa nasce sempre dentro da tolerância de fechamento e o elo fecha em 0,3 s sem
    # trabalho. O slot continua sorteado (senão o normalizador o vê constante). Para o
    # cubo isto não muda nada em PEGAR, CARREGAR ou BOTAR. Quando a reorientação virar
    # foco, a tabela sai do nível e vai para um bloco `Reorientar` próprio (spec §8.3):
    #     voltas_max     = (0, 0, 1, 1, 1, 1, 1)       <- a de antes, por nível
    #     eixo_vertical  = (False, False, False, False, True, True, True)
    voltas_max: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0)
    eixo_vertical: tuple[bool, ...] = (False,) * 7
```

- [ ] **Step 4: Rodar o smoke até verde**

Esperado: `0 falhas`. O check antigo `"o eixo do reorientar satura no nível 4"` continua verdadeiro (zero em todo nível).

- [ ] **Step 5: Commit**

```bash
cd /home/joaobornelli/Documents/g1_training && git -c core.hooksPath=/dev/null add g1_limpo/knobs.py g1_limpo/smoke.py && git -c core.hooksPath=/dev/null commit -m "feat(limpo): REORIENTAR inerte na run da v2 — voltas_max zero em todo nivel

Spec §8.3. A rede recebe o contrato (giro_b) mas o treino da reorientacao
fica para depois: e tarefa dificil e pode custar o resto. Com a caixa
nascendo dentro da tolerancia o elo fecha em 0,3 s sem trabalho, e o slot
do one-hot segue sorteado. A tabela antiga fica em comentario."
```

---

### Task 8: Fechamento — inspetor, paridade, documentação, memória

**Files:**
- Modify: `g1_limpo/ARQUITETURA.md` (layout do comando e da observação, cadeias, tabela por elo)
- Modify: `g1_limpo/__init__.py` (docstring)
- Modify: `docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md` (estado)
- Modify: `~/.claude/memory/g1-limpo-contrato-de-troca-de-tarefa.md`

- [ ] **Step 1: Rodar o inspetor para cada elo e para a cadeia 3**

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.inspeciona --help 2>&1 | head -30
```

Leia as opções. Depois rode, para cada `elo` em `andar reorientar pegar carregar botar`, o inspetor com esse elo (a opção de elo conforme o `--help`), e para a cadeia 3 use o nome que `inspeciona._nomes_de_cadeia()` deriva de `CADEIAS` (imprima com `.venv/bin/python -c "from g1_limpo import inspeciona as I; print(I._nomes_de_cadeia())"`). Esperado: nenhuma linha `✗`; a tabela ANTES/DEPOIS mostra três elos para a cadeia 3.

- [ ] **Step 2: Rodar a paridade e o smoke inteiro uma última vez**

```bash
cd /home/joaobornelli/Documents/g1_training && .venv/bin/python -m g1_limpo.paridade 2>&1 | tail -3 && .venv/bin/python -m g1_limpo.smoke 2>&1 | tail -3
```

Esperado: paridade igual à Task 0; smoke `N ok / 0 falhas` com N ≥ base + 60.

- [ ] **Step 3: Confirmar as travas da spec §12 no diff**

```bash
cd /home/joaobornelli/Documents/g1_training && git -c core.hooksPath=/dev/null diff exp/g1-limpo -- g1_limpo/curriculo.py | wc -l && git -c core.hooksPath=/dev/null diff exp/g1-limpo -- g1_limpo/eventos.py | grep -E '^[-+].*def (entrega_tarefa_no_viewer|avanca_elo_no_viewer)' | wc -l && git -c core.hooksPath=/dev/null diff exp/g1-limpo -- g1_limpo/knobs.py | grep -E '^-\s+[a-z_]+:'
```

Esperado: `0`, `0`, e a terceira saída lista só as linhas removidas `-    espera_s: ...`, `-    caixa_z_min: ...`, `-    voltas_max: ...`, `-    eixo_vertical: ...` — os únicos knobs existentes que mudaram, todos decididos na spec §13. Nenhum peso de `Recompensa` nem dos sete de `Tarefa` aparece com `-`.

- [ ] **Step 4: `ARQUITETURA.md` e `__init__.py`**

Em `g1_limpo/ARQUITETURA.md`:
- no bloco do layout do comando (linhas ~916-919), acrescente `GIRO   = slice(9, 12)   # eixo × ângulo do giro pedido, em MUNDO (spec §8.3)` e troque `DIM    = 9` por `DIM    = 12`;
- na linha ~877 troque `return cat([caixa_b, alvo_b, cmd[:, ANG], cmd[:, VALIDA]])` por `return cat([caixa_b, alvo_b, giro_b, meia_aresta]) * (publicado != ANDAR)`;
- na linha ~899 troque `N_CAIXA = 8` por `N_CAIXA = 10`;
- no bloco `CADEIAS` (linhas ~1118-1121) troque `(PEGAR, BOTAR))` por `(PEGAR, CARREGAR, BOTAR))   # segurar parado (spec §6.5)`;
- na tabela por elo (linha ~928, coluna `VALIDA`) acrescente uma linha de nota: `A espera inicial publica ANDAR com VALIDA = 0; a espera final publica ANDAR com VALIDA = 1 (spec §6.0).`;
- na linha ~308 (`3g | 7 termos de recompensa`) acrescente `| 3i | load, largou (só BOTAR / espera final) | leem o elo interno |`.

Em `g1_limpo/__init__.py`, no docstring, troque `cadeias de no máximo dois elos` por `cadeias de até três elos` e a linha das cadeias por `(PEGAR)  (REORIENTAR->PEGAR)  (PEGAR->CARREGAR)  (PEGAR->CARREGAR->BOTAR)`.

Confira que nada mais cita o layout velho:

```bash
cd /home/joaobornelli/Documents/g1_training && grep -nE 'DIM *= *9|N_CAIXA *= *8|\(PEGAR, BOTAR\)|caixa_z_min' g1_limpo/*.py g1_limpo/ARQUITETURA.md
```

Esperado: nenhuma linha (fora de comentários históricos que digam explicitamente "antes").

- [ ] **Step 5: Estado da spec e memória**

Na spec, troque no cabeçalho `Nada implementado. Próximo passo: o plano de implementação na branch exp/g1-limpo-v2.` por `Implementada na branch exp/g1-limpo-v2 em <data> (plano: docs/planos/2026-09-03-plano-contrato-de-troca-v2.md); smoke N ok / 0 falhas. Próximo passo: o dono faz o push, o Kaggle faz o pull, treino do zero (§12).`

Na memória `~/.claude/memory/g1-limpo-contrato-de-troca-de-tarefa.md`, troque `nada implementado; o plano de implementação é o passo seguinte.` por `implementada em <data> na exp/g1-limpo-v2 (N checks de smoke); falta o push do dono e o treino do zero.`

- [ ] **Step 6: Commit final**

```bash
cd /home/joaobornelli/Documents/g1_training && git -c core.hooksPath=/dev/null add g1_limpo/ARQUITETURA.md g1_limpo/__init__.py docs/planos/2026-09-02-contrato-de-troca-de-tarefa.md && git -c core.hooksPath=/dev/null commit -m "docs(limpo): ARQUITETURA e spec refletem a v2 implementada

Layout do comando com GIRO (DIM 12), N_CAIXA 10, cadeia 3 de tres elos,
os dois elos publicado/interno nas esperas, load e largou. A spec marca
a implementacao e o proximo passo: push, pull no Kaggle, treino do zero."
```

Não faça `git push`. Avise o dono: a branch `exp/g1-limpo-v2` está pronta para o push e para a run de treino do zero, com as sentinelas da spec §12 (`descarga`, `rampa`, `seg_proj/seg_pedido`, `Episode_Reward/load`, `Metrics/alvo_caixa/passo_final`, `sucesso` da cadeia 3).

---

## Auto-revisão do plano contra a spec

**Cobertura da §11.1 (os 24 itens):** 1, 2, 3, 4 → Task 4 (seção 25) e Task 2 (seção 23); 5 → seções 16b/17 existentes (rastreio paga na espera, sete zero) + Task 2; 6 → Task 2 (`limpo_elo`); 7 → Task 2 (fonte); 8 → Task 8 (diff); 9, 10, 11 → Task 1 (seção 22); 12 → Tasks 2 e 5; 13, 14 (o teste do `unload` existente), 15 → Task 3 (seção 24); 16, 17, 18 → Task 5 (seção 26); 19 → Task 3; 20 → Task 2 (fonte); 21 → Task 6; 22 → Task 4 (fonte); 23 → Task 4; 24 → Task 7.

**Cobertura da §6:** 6.0/6.2/6.3 → Task 2; 6.1 → Task 4; 6.4 → Task 2 (`fracao_esperando`, `recebe_tarefa`, `rastreio`); 6.5 → Task 1; 6.6 → Tasks 2 e 5; 6.7 → Task 3; §8.3 contrato → Task 4, inerte → Task 7; §9 → Task 6.

**Consistência de nomes:** `_SEGURA_PARADO`, `_segura_parado`, `_segurar` (Task 1) usados só no comando; `limpo_soltou`, `limpo_elo_interno`, `_soltou` (Task 2) lidos em `terminacoes`, `recompensas`, `observacoes`; `limpo_meia_aresta`, `_meia` (Task 3) lidos em `comando`, `eventos`, `terminacoes`, `observacoes`; `GIRO`, `N_CAIXA = 10`, `fatia_do_elo_interno`, `um_de_cinco_interno` (Task 4) lidos em `algoritmo`, `env_cfg`, `smoke`; `load`, `largou`, `_fora_do_botar`, `_elo_interno` (Task 5); `is_turning_env`, `rel_turning_envs`, `turning_wz_min` (Task 6). `caixa_largada(env, folga_chao, dist_max, meia_aresta_ref)` é a assinatura única a partir da Task 3; a Task 2 ainda usa a antiga com `z_min` e a Task 3 a troca junto com o `env_cfg`.
