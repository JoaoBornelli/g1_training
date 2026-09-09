# g1_limpo — soltar a caixa fora do alvo termina (v3.2)

Data: 2026-09-09. Branch: `exp/g1-limpo-v2`. Base: `c1801ea` (dois-bits v3.1 + notebook).
Retomada: o checkpoint da `bloco11` interrompida (~7300), subido como versão nova do dataset.

Contrato de observação NÃO muda: ator 114, crítico 131.

## 0. O problema, medido

`bloco11`, iterações 7044 → 7299: `s_C = 0,0000`, `load = 0,0014`, `impacto_da_caixa` 4,97 → 7,70,
`caixa_largada` 10,5% → 38,5% das terminações, `avancos` 0,59 (o maior já visto).

O robô chega ao BOTAR e ATIRA a caixa no alvo. Por quê, lendo o código:

| paga POSIÇÃO da caixa, sem exigir a mão | exigiria a mão, mas vale ZERO no BOTAR |
|---|---|
| `staged` até 6/s — `_alcancar ≡ 1` no BOTAR (`recompensas.py`, ramo `um`) | `squeeze` — `_fora_do_botar` |
| `precise_pos` até 3/s — nunca teve gate de mão | `unload` — `_fora_do_botar` |
| `precise_ori` até 1/s — `_alcancar ≡ 1` | `postura_ereta` — herda de `unload` |

A partir de "caixa no peito, 0,45 m acima do alvo", soltar reduz `d_alvo` mais rápido que descer.
Nada distingue os dois caminhos. O gradiente aponta para soltar.

A cláusula `escapou` de `caixa_largada` (`dist_palmas > 0,45 m`) não pega o arremesso curto: a caixa
pousa na laje ANTES de se afastar 0,45 m das palmas.

Segundo buraco, dormente: com `soltou`, `_alcancar ≡ 1` de novo, a caixa vai a +5 m, `trazer → 0`,
e `staged = 1 × (1 + 0) = 1,0` → **3,0/s pelo resto do episódio**, com a caixa a 5 m e mascarada
da observação. Dormente porque o BOTAR nunca fecha.

## 1. Princípio (decisão do dono, 2026-09-09)

**Em manipulação, soltar a caixa fora do alvo é terminação.** A caixa só pode ser solta quando está
no alvo. É o princípio que `terminacoes.py` já enuncia: "largar a caixa NÃO tem como ser pago".
Penalidade máxima = a que já existe: `terminacao` (−4,0 por evento) mais toda a renda futura perdida.
Nenhum número novo.

Simplicidade: **duas funções tocadas, duas linhas de lógica, knobs ±0, 28 termos, 3 terminações.**

## 2. `terminacoes.caixa_largada` — a cláusula `escapou` é REESCRITA

Sai:

```python
palmas = env.scene["robot"].data.site_pos_w[:, env.limpo_ids_palma, :]
dist = torch.norm(palmas - caixa.unsqueeze(1), dim=-1)
escapou = (dist > dist_max).all(dim=-1)
```

Entra, no mesmo lugar:

```python
v_caixa = env.scene["box"].data.root_link_lin_vel_w
v_base  = env.scene["robot"].data.root_link_lin_vel_w
v_rel   = torch.norm(v_caixa - v_base, dim=-1)
d_alvo  = torch.norm(caixa - alvo, dim=-1)              # alvo = _command[:, ALVO], como recompensas._alvo
no_alvo = d_alvo <= raio_solta                          # raio_solta = precise_pos_sigma (0,18), REUSO
soltou_fora = (v_rel > v_solta) & ~no_alvo
return caiu | (soltou_fora & (pegou > 0.5) & (soltou < 0.5))
```

- `caiu` FICA como está.
- A exceção `soltou < 0.5` FICA: depois do fecho do BOTAR as mãos têm de sair.
- O detector é CINEMÁTICO, não de contato: as palmas piscam (`palmas_em_contato = 0,56`; o tronco
  escora 94,7%), e um detector de contato mataria o hold. Uma caixa na mão move-se com a base; uma
  caixa solta ou atirada não.
- A terminação é do ENV e só existe no treino. O robô real não a computa; ele só aprende a não soltar.
- Params: `folga_chao` (fica), `dist_max` SAI, `v_solta` e `raio_solta` ENTRAM (ver §4). `env_cfg.py:262`.

⚠ Escopo "depois do PEGAR fechar": NÃO acrescentar guarda a priori. Com `v_solta` bem medido, a pega
não dispara (a caixa está parada na laje; o push é 0,5 m/s). A medição do §5 decide. Se o máximo de
`v_rel` na pega com push passar de `v_solta`, aí entra `& ((elo_interno != PEGAR) | fechou)`, com
sinais que já existem — e o motivo vai no comentário.

## 3. `recompensas._alcancar` — o ramo `soltou` devolve 0

Hoje:
```python
um = t._elo == BOTAR
if soltou is not None:
    um = um | (soltou > 0.5)
return torch.where(um, torch.ones_like(kernel), kernel)
```

Vira:
```python
no_botar = (t._elo == BOTAR) & (soltou < 0.5)   # BOTAR ativo: ≡ 1. A terminação do §2 GARANTE a mão
depois   = soltou > 0.5                          # cauda pós-fecho: ≡ 0. Só `renda_congelada` paga
return torch.where(depois, torch.zeros_like(kernel),
                   torch.where(no_botar, torch.ones_like(kernel), kernel))
```

- O `≡ 1` no BOTAR ativo FICA: a premissa "as mãos estão na caixa" deixa de ser assumida e passa a
  ser imposta pelo §2.
- Atualizar o docstring: o ⚠ que diz "`alcança ≡ 1` no BOTAR e na espera final" passa a dizer só
  "no BOTAR ativo; na cauda é 0, e o motivo é este".
- Efeito: na cauda, `staged → 0` e `precise_ori → 0`. `precise_pos` já é ~0 (σ 0,18 contra 5 m).

## 4. Knobs (`knobs.Terminacao`)

| sai | entra |
|---|---|
| `caixa_dist_max = 0.45` | `v_solta` (m/s) — MEDIDO (§5), com a tabela ao lado |
| — | `raio_solta` NÃO é knob novo: `env_cfg` passa `k.recompensa.precise_pos_sigma` |

Saldo: −1 +1 = 0.

## 5. Medição ANTES de escrever `v_solta` — uma sonda, CPU, `model_6999`

Reuse o andaime de `scratchpad/sonda_botar_v2.py` (env + wrapper + runner + `episode_length_s = 20`).
UM env, 32 envs, ~20 s de sim. Três leituras de `v_rel = ‖v_caixa − v_base‖`:

| condição | como montar | o que ler |
|---|---|---|
| a) caixa na mão | `cadeia_forcada = 0` (B), passos com `limpo_pegou = 1` — hold e cauda andando | **p99** e máx |
| b) caixa solta do peito | pinar a caixa a `alvo + (0,0,0.45)`, zerar velocidade, soltar (não escrever mais a pose), ler `v_rel` nos passos 1, 3, 5 | a curva da queda livre (~0,2 / 0,6 / 1,0 m/s) |
| c) push na pega | `cadeia_forcada = 0`, passos com `limpo_pegou = 0` e a caixa na laje; o `push_robot` está ativo | **máx** de `v_rel` |

`v_solta` fica ACIMA de max(a.p99, c.máx) e ABAIXO de b.passo5. Palpite a confirmar: 0,6–0,8.
Se c.máx > a.p99 e perto de `v_solta`, o escopo do §2 entra. Registrar as três leituras no knob.

## 6. Smoke — NÃO RODAR. O dono roda localmente.

Escrever os checks; não executar `smoke.py`.

Migram: todo check que lê `caixa_dist_max` / `dist_max`, e o que afirma `alcanca ≡ 1` com `soltou`.

Entram, seção `--- v3.2: soltar termina`:

1. Caixa pinada a 0,45 m acima do alvo, na mão (pegou=1), depois solta: `caixa_largada` True no
   passo em que `v_rel` cruza `v_solta` (≤ 5 passos).
2. Caixa descendo escrita à mão a 0,2 m/s (abaixo de `v_solta`), pegou=1: `caixa_largada` False.
3. Caixa solta a 0,10 m acima do alvo (dentro de `raio_solta`): `caixa_largada` False.
4. Cauda: `soltou=1`, caixa e laje a +5 m: `_alcancar == 0`, `_step_reward[staged] == 0`,
   `_step_reward[precise_ori] == 0`.
5. BOTAR ativo, caixa na mão: `_alcancar == 1` (o ≡ 1 fica).
6. Pega: caixa na laje, pegou=0, base com velocidade escrita de 0,5 m/s (o push): `caixa_largada` False.
7. `caixa_dist_max` ausente de `knobs.Terminacao`; `v_solta` presente; `raio_solta` do cfg ==
   `precise_pos_sigma`.

⚠ Ao escrever pose/velocidade de corpo, usar `write_root_link_pose_to_sim` /
`write_root_link_velocity_to_sim` e dar UM `step` antes de ler (`xpos` só atualiza no forward).
Atribuição direta em `.data.*` não gruda — já quebrou 5 checks nesta base.

## 7. Notebook (`g1_limpo/kaggle/g1_limpo_kaggle.ipynb`)

`RUN = "bloco12"`. `META` = it0 + 2000 (ler do checkpoint que subir). Um discriminador:
`hasattr(k.terminacao, "v_solta") and not hasattr(k.terminacao, "caixa_dist_max")`.
Editar por `json` com `assert alvo in fonte` antes de cada `.replace`, e `compile()` de cada célula.

## 8. O que NÃO muda

Pesos; `load`; as máscaras `_fora_do_botar`; `caiu`; `CADEIAS`; balanceador; contrato 114/131;
os 70° do `fell_over`; `joelho_z_min`; `de_pe_tol_rad`; G2.

## 9. Commits

NÃO commitar antes do smoke do dono. Deixar a árvore com as edições e reportar. Depois do smoke
verde: `fix(limpo): soltar a caixa fora do alvo termina; a cauda não paga manipulação` e
`test(limpo): ...` separados. Sem `Co-Authored-By`. Sem push.
