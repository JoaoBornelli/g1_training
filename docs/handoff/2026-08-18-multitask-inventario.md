# Handoff — multitask G1: inventário do desenho e problemas abertos

| campo | valor |
|---|---|
| data | 2026-08-18 |
| branch | `exp/locomocao-velocidade` · HEAD `6c29464` (igual ao remoto) |
| `dt` do passo | **0,02 s** (`decimation 4` × `timestep 0,005`) = 50 Hz |
| horizonte do desconto | `gamma 0,99` a 50 Hz = **2,0 s** |
| envs | 4096 · `num_steps_per_env 24` = 98 304 transições por iteração |
| próximo ponto de partida | `model_13700.pt` |

Tudo abaixo saiu do código, não de memória. Comandos de extração no fim (§11).

---

# PARTE I — O DESENHO

## 1. O currículo

Uma política, condicionada em `task_onehot` dentro do vetor de comando. O orquestrador
(`curriculum.py::Orquestrador`) é um termo de currículo. Ele roda no reset, **antes** dos
eventos e do comando (`manager_based_rl_env.py:554` contra `:560` e `:581`).

### 1.1 Estrutura

O estado do currículo é uma grade de **células**. Uma célula é `(tarefa, eixo)`, e cada
célula tem uma lista de **níveis**. Só uma tarefa nasce aberta: `locomover`.

```
célula = (tarefa, eixo)
  ├── abertos[célula]   -> quantos níveis podem ser sorteados
  ├── perf[célula][i]   -> EMA de sucesso do nível i   (alpha = 0,03)
  └── amostras[célula][i]
```

### 1.2 A regra do evento

```
condição:  episódios(T) >= 200   E   perf[T][nível topo] >= 0,90
ação 1:    se a DR de peso está FECHADA -> abre a DR
           senão                        -> abertos[T] += 1
ação 2:    abre todo filho cujos pais já tiveram o 1º evento
```

As duas ações rodam no MESMO evento. A tarefa nova começa no nível 0 enquanto a tarefa
mãe avança. Nada serializa.

⚠️ Uma tarefa sem eixo a avançar e sem DR a abrir cairia no `continue` e **nunca teria
evento** — o que travava a cadeia inteira em silêncio. A condição carrega
`or self.eventos_tarefa[t] == 0` por causa disso.

### 1.3 Amostragem

- **Tarefa:** inversa à competência, com piso. `piso_amostragem = 0,15` por tarefa
  aberta; o resto se reparte na proporção de `1 − perf[topo]`. Medido com 3 tarefas
  abertas: 0,517 / 0,333 / 0,150.
- **Nível dentro da célula:** entre os níveis abertos.

### 1.4 Os 12 destravamentos

Derivados, não digitados (`tasks.py::unlock_count`):

| fonte | quantidade |
|---|---:|
| aberturas de tarefa (`NUM_TASKS − 1`) | 4 |
| DR de peso (uma por tarefa em `COM_DR_PESO`) | 4 |
| níveis do eixo `alvo` do `pegar` (5 − 1) | 4 |
| níveis dos outros eixos (congelados em 1 nível) | 0 |
| **total** | **12** |

### 1.5 Parâmetros

| knob | valor | o que faz |
|---|---|---|
| `min_amostras_evento` | 200 | episódios mínimos na tarefa antes de um evento poder disparar |
| `limiar_competencia` | 0,90 | o portão do evento |
| `ema_alpha` | 0,03 | suavização do `perf` |
| `piso_amostragem` | 0,15 | fração mínima de cada tarefa aberta |
| `platô_amostras` | 2000 | acima disso o nível recebe a flag `plato_n<i> = 1` |
| `alarme_transicoes` | 2,0e8 | acima disso liga `ALARME_estagnacao` (≈ 2035 iterações) |
| `rho` / `focus_beta` | 0,3 / 1,0 | herdados do desenho anterior |

---

## 2. As 5 tarefas e seus eixos

Níveis disponíveis (`tasks.LEVELS`):

```
velocidade : (1.0, 1.5, 2.0)                      m/s
altura     : (0.55, 0.45, 0.35, 0.25, 0.15, 0.05, 0.0)   m — topo da prateleira
giro       : (15, 45, 90, 180, 360)               graus
alvo       : (0.2, 0.4, 0.6, 0.8, 1.0)            fração da rampa de altura do pegar
peso       : (1.0, 5.0)                           kg — é a DR, não eixo
```

Congelamento ativo (`NIVEIS_ATIVOS`): `velocidade 1` · `altura 1` · `giro 1` ·
**`alvo 5`**. Só o `pegar` tem eixo com margem para avançar neste bloco. É de propósito:
concentra os 12 destravamentos onde o problema está.

| # | tarefa | eixo | níveis ativos | DR de peso | episódio | sustentação |
|---|---|---|---|---|---:|---:|
| 0 | `locomover` | `velocidade` | `(1.0,)` | — | 20 s | 0 s (fecha no `time_out`) |
| 1 | `pegar` | **`alvo`** | `(0.2, 0.4, 0.6, 0.8, 1.0)` | sim | 10 s | 2,0 s |
| 2 | `reorientar` | `giro` | `(15°,)` | sim | 10 s | 2,0 s |
| 3 | `locomover_carregando` | `velocidade` | `(1.0,)` | sim | 20 s | 3,0 s |
| 4 | `botar` | `altura` | `(0.55,)` | sim | 10 s | 2,0 s |

O episódio é por tarefa via terminação própria (`time_out_por_tarefa`), porque
`max_episode_length` do mjlab é escalar. O global fica em 20 s.

### 2.1 O eixo `alvo` — como a rampa funciona

O eixo do `pegar` é **fração**, não metro:

```python
alvo_z = plr_rest_z + frac × (_TOPO_RAMPA_Z − plr_rest_z)      # _TOPO_RAMPA_Z = 0,91 m
```

Os dois extremos são **z de mundo**. `plr_rest_z` é onde a caixa descansa (propriedade
da prateleira, escrita no reset). `_TOPO_RAMPA_Z` é constante: `pelve_de_pé(0,76) + 0,15`.
Nunca lê a pose do robô — foi por isso que a versão ancorada na base deixava o robô
encurtar o percurso agachando.

Nível 0 (`frac = 0,2`) pede **5,5 cm** de subida. Nível 4 (`frac = 1,0`) pede 27 cm.
Fração e não metro para compor com o eixo `altura` quando ele reabrir.

---

## 3. O critério de sucesso — a régua

Vive em `metrics.py`, e é **fato físico**, não soma de reward. É isso que faz "mudar peso"
ser Categoria A: peso não move a régua.

### 3.1 Base — vale para as 5 tarefas

```
erro_lin = (1/T) ∫ ‖v_cmd_xy − v_xy‖ dt          <= tol_v = 0,25   m/s
erro_ang = (1/T) ∫ |ωz_cmd − ωz|   dt            <= tol_w = 0,70   rad/s
base     = nunca_caiu  &  passos >= 50  &  as duas acima
```

Com comando zero — as três tarefas de manipulação — a base mede **deriva e rebolado de
pelve**.

### 3.2 Condição adicional por tarefa

| tarefa | condição |
|---|---|
| `locomover` | nenhuma. A base já é a tarefa dele. |
| `pegar` | `erguida & parado_de_pé & preensão` |
| `reorientar` | `erro_ângulo < 10° & desvio_xy < 0,05 & apoiada` |
| `locomover_carregando` | `no_peito & preensão` |
| `botar` | `no_alvo & ~preensão & caixa_quieta & parado_de_pé` |

Onde:

- `erguida` = `box_z >= alvo_z − alvo_tol_z`, com `alvo_tol_z = 0,02`. **Piso em z, não
  esfera 3D** — a esfera reprovava o robô por erguer MAIS que o pedido.
- `parado_de_pé` = `z_pelve >= 0,65` **e** inclinação `<= 20°`. Os 0,65 m contra 0,76 do
  keyframe permitem agachar 11 cm, não permitem ficar dobrado.
- `preensão` = `_grasp > 0,5`: duas palmas tocando, **nenhum** verso. Sem isso o robô
  aninha a caixa no vão dos antebraços e ergue sem pegar.
- `no_peito` / `no_alvo` = distância `< caixa_no_alvo = 0,10 m`.
- `caixa_quieta` = `‖v_caixa‖ < 0,05 m/s`.

O sucesso é do **episódio**: uma vez conquistado, persiste até o reset.

### 3.3 A tolerância inteira

| knob | valor |
|---|---|
| `tol_v` | 0,25 m/s |
| `tol_w` | **0,70 rad/s** |
| `de_pe_z` / `de_pe_tilt_rad` | 0,65 m / 0,349 rad (20°) |
| `caixa_no_alvo` | 0,10 m |
| `caixa_quieta_v` | 0,05 m/s |
| `reorienta_angulo_deg` / `reorienta_xy` | 10,0° / 0,05 m |
| `alvo_tol_z` | 0,02 m |
| `largou_z` | 0,30 m |
| `sustenta_pegar_s` / `carregar` / `botar` / `reorienta` | 2,0 / 3,0 / 2,0 / 2,0 s |

---

## 4. Recompensas — o sinal de TAREFA

### 4.1 O orçamento igual

`_equaliza_orcamento` iguala o sinal de tarefa em **4,0 por passo em toda tarefa**. O
fator é `4,0 / Σ pesos dos termos gateados naquela tarefa`. **Penalidades não são
escaladas** — manter o denominador parado é o que iguala a razão sinal/penalidade.

| tarefa | orçamento | **fator** |
|---|---:|---:|
| `locomover` | 4,00 | 1,0000 |
| `pegar` | 4,50 | **0,8889** |
| `reorientar` | 2,00 | **2,0000** |
| `locomover_carregando` | 5,00 | 0,8000 |
| `botar` | 1,00 | **4,0000** |

### 4.2 Por tarefa, com peso efetivo

**`locomover`** — soma 4,00

| termo | peso | efetivo | mede |
|---|---:|---:|---|
| `track_linear_velocity` | 2,00 | 2,000 | rastreio da velocidade linear comandada |
| `track_angular_velocity` | 2,00 | 2,000 | erro de heading nos envs com heading, `ωz` nos outros |

**`pegar`** — soma 4,00

| termo | peso | efetivo | mede |
|---|---:|---:|---|
| `lift` | 2,00 | **1,778** | `preensão × upright_caixa × progress`, progresso de repouso até `alvo_z` |
| `reaching` | 1,00 | 0,889 | aproximação das palmas à caixa, 2 escalas (grossa + fina) |
| `grasp` | 0,50 | 0,444 | bônus de preensão estabelecida — **booleano** |
| `unload` | 1,00 | 0,889 | `preensão × (1 − F_apoio/mg) × [caixa acima do repouso]` |

**`reorientar`** — soma 4,00

| termo | peso | efetivo | mede |
|---|---:|---:|---|
| `reaching` | 1,00 | **2,000** | idem |
| `orienta_face` | 1,00 | **2,000** | gira a face alvo para a direção alvo sem arrastar a caixa |

**`locomover_carregando`** — soma 4,00

| termo | peso | efetivo | mede |
|---|---:|---:|---|
| `track_linear_velocity` | 2,00 | 1,600 | idem |
| `track_angular_velocity` | 2,00 | 1,600 | idem |
| `box_at_peito` | 1,00 | 0,800 | `preensão × gaussiana(erro caixa → alvo do peito)` |

**`botar`** — soma 4,00

| termo | peso | efetivo | mede |
|---|---:|---:|---|
| `box_at_prateleira` | 1,00 | **4,000** | gaussiana do erro caixa → alvo, **sem** fator de preensão |

⚠️ O `botar` tem **um único** termo carregando os 4,0 inteiros. Qualquer mudança nele move
100% do sinal daquela tarefa.

### 4.3 O bônus de sucesso — fora do orçamento

| termo | peso | gate | mede |
|---|---:|---|---|
| `sucesso_denso` | **5,00** | `pegar` `reorientar` `locomover_carregando` `botar` | 1,0 enquanto a `condicao_tarefa` vale |

Fica **fora** de `TERMOS_DE_TAREFA` de propósito: dentro, o orçamento do `pegar` iria a
9,5 e o bônus se diluiria a 2,1. Peso 5,0 vale direto, sem escala.

O `locomover` fica fora do gate: a condição dele é `ones_like` e ele coletaria o bônus
inteiro de graça.

⚠️ **`sucesso_denso` NÃO passa pelo `_base`.** Ele é só `condicao_tarefa`. O `Sucesso`
(que alimenta o currículo) é `condicao_tarefa & _base`, e o `_base` é quem carrega o
`tol_w`. Por isso o reward pode pagar enquanto o currículo se recusa a creditar.

---

## 5. Penalidades e postura

Nenhuma é escalada pelo orçamento. As "GLOBAL" valem em toda tarefa.

### 5.1 Postura — positivas, fora do orçamento

| termo | peso | gate | mede |
|---|---:|---|---|
| `upright` | +1,00 | GLOBAL | manter a base ereta |
| `posture` | +0,50 | GLOBAL | desvio da pose default, com tolerância dependente da velocidade |

### 5.2 Penalidades globais

| termo | peso | mede |
|---|---:|---|
| `terminacao` | **−200** | episódio terminado que não é `time_out`. ⚠️ Com `dt = 0,02` o efetivo por evento é **−4,0** |
| `foot_clearance` | −2,00 | desvio da altura-alvo de pé no ar, ponderado pela velocidade do pé |
| `table_contact` | −1,50 | qualquer parte do corpo **exceto as mãos** tocando a mesa (apoiar-se) |
| `self_collisions` | −1,00 | auto-colisão |
| `dof_pos_limits` | −0,50 | junta cruzando o limite mole |
| `back_penalty` | −0,50 | tocar a caixa com o **verso** da mão (gradiente "vire a palma") |
| `foot_swing_height` | −0,25 | desvio da altura de balanço, avaliado no pouso |
| `action_rate_l2` | −0,10 | taxa de mudança da ação. **`ActionRateJuntas`: fator 0,25 nos 14 canais de braço** |
| `feet_slip` | −0,10 | pé escorregando (velocidade xy em contato) |
| `body_ang_vel` | −0,05 | velocidade angular do corpo |
| `angular_momentum` | −0,02 | momento angular de corpo inteiro (favorece balanço natural de braço) |
| `soft_landing_table` | −1,0e−4 | força de pouso na mesa |
| `soft_landing_feet` | −1,0e−5 | força de pouso dos pés |
| `joint_acc` | −2,5e−7 | aceleração de junta |

### 5.3 Penalidades gateadas por tarefa

| termo | peso | gate | mede |
|---|---:|---|---|
| `com_balance` | −2,00 | `pegar` `reorientar` `botar` | CoM derivando à frente do apoio dos pés (dump de peso) |
| `box_shake` | −0,15 | `locomover` `locomover_carregando` `botar` | soma dos quadrados da velocidade angular da caixa |
| `box_shake_pegar` | −0,15 | `pegar` | idem, mas **só cobra depois** da caixa chegar perto do alvo (`std = 0,10`) |

---

## 6. Terminações

| termo | `time_out` | parâmetros | gate |
|---|---|---|---|
| `time_out` | **True** | `limites_s` por tarefa (20/10/10/20/10 s) | todas |
| `fell_over` | False | `limit_angle = 1,2217 rad` (70°) | todas |
| `nonfinite` | False | — | todas |
| `largou` | False | `z_min = 0,30` | `locomover_carregando` |
| `caixa_caiu` | False | `margem = 0,10`, `shelf_half_z = 0,02` | `reorientar` |

---

## 7. PPO e runtime

| parâmetro | valor |
|---|---|
| `gamma` / `lam` | 0,99 / 0,95 → horizonte de **2,0 s** a 50 Hz |
| `schedule` / `desired_kl` | `adaptive` / 0,01 |
| `learning_rate` | 1e-3 default; **5e-4 no degrau de retomada** |
| `entropy_coef` | **0,01** — ver problema A1 |
| `clip_param` | 0,2 |
| `num_learning_epochs` / `num_mini_batches` | 5 / 4 |
| `num_steps_per_env` | 24 |
| `save_interval` | 50 iterações (~4 min) · checkpoint 5,7 MB |
| distribuição | `GaussianDistribution`, `std_type "scalar"`, `std_param` é **vetor de 29** |
| obs | 151 no ator · +12 privilegiados no crítico · ação 29 |

---

# PARTE II — PROBLEMAS

## 8. Abertos

### A1 · o `std` foge — **BLOQUEANTE**

**Sintoma.** 0,43 → 0,78 (com reaquecimento) → 0,98. No relançamento limpo,
0,43 → 0,70 em **300 iterações**.

**Por que mata.** O `std` alto dobra a oscilação de guinada, e o `_base` mede o erro CRU:

| `std` | `locomover/erro_vel_ang` |
|---:|---:|
| 0,43 | 0,591 |
| 0,78 | 1,153 |
| 0,98 | 1,131 |

`erro ≈ 0,591 + 1,606·(std − 0,43)` cruza `tol_w = 0,70` em **`std` ≈ 0,498**. Acima
disso nenhuma tarefa pontua: em it 23548, `Episode_Metrics/sucesso = 0,0000`, todo
`perf_n0 = 0`, e `ALARME_estagnacao = 1` com **7,9e8** transições sem destravar contra o
limite de 2,0e8 — cerca de **8000 iterações**.

**Por que sobe.** Três fatos somados:

1. Entropia de gaussiana é `Σ log σ`, então a força do `entropy_coef = 0,01` é **∝ 1/σ**:
   quanto menor o `std`, mais forte o empurrão. As taxas confirmam — 0,0009/it partindo de
   0,43 contra 0,00016/it partindo de 0,78.
2. `std_param` é vetor de **29** (`rsl_rl/modules/distribution.py:166`). Cada junta tem o
   seu.
3. `action_rate_bracos = 0,25` cortou a ¾ o freio nos 14 canais de braço. O corte foi
   deliberado — o `action_rate` era 82% do sinal de tarefa do `pegar` — mas era ele que
   opunha a entropia ali.

**Não se autocorrige:** `locomover/track_angular_velocity` desabou 0,467 → 0,096, ou seja
o termo que pagava precisão de guinada sumiu. Realimentação positiva.

**Conserto proposto, não aplicado.** Teto no `std`, espelho do `_sobe_std` existente:

```python
param.data.clamp_(max=0.45)      # todo update, mesmo gancho do runner
```

Cabe porque `std_range` do rsl_rl é lista Python comum e **não entra no `state_dict`**.
`clamp_` no `.data` mantém o gradiente vivo: se a política quiser baixar de 0,45, pode.
0,45 e não 0,50 para ter margem. Categoria A. Junto, 3 linhas logando o `std` separado
**pernas × braços** — sem elas não se sabe se o teto deve ser global ou por grupo.

**Alternativas descartadas:** devolver `action_rate_bracos` a 1,0 re-bloqueia o `lift`;
trocar a régua do `_base` para o `erro_vel_ang_filt` é o conserto certo a longo prazo (o
filtrado deu 0,143–0,305 mesmo com `std` 0,98) mas invalida a EMA acumulada.

⚠️ O `model_13700.pt` sai com `std = 0,4769` — **0,02** do precipício, contra 0,07 do
22296.

### A2 · `pegar/cond_fisica = 0,0000` exato, com a caixa no ar

Valores CRUS em it 23548 (contrib ÷ efetivo, ver §4.1):

| termo | contrib | cru | leitura |
|---|---:|---:|---|
| `grasp` | 0,3193 | **0,718** | preensão em 72% dos passos |
| `unload` | 0,5683 | **0,639** | caixa fora do apoio em ~89% desses |
| `lift` | 0,7013 | **0,394** | `progress` 0,55 = **3,0 cm** dos 5,5 |
| `reaching` | 0,6037 | 0,679 | palmas na caixa |

Sinal de tarefa 2,19 de 4,00 — era 1,07 no 22296. `pegar/_total` **+0,32** líquido, era
≈ −0,08. **A ponte do `unload` funciona.**

Mas o `cond_fisica` é zero exato. A `preensão` vale, senão o `lift` seria 0. Sobra:

- **`erguida`** exige `progress >= 1 − 0,02/0,055 = 0,64`, ou 3,5 cm. Ele para em 3,0 cm.
  **Erra por 5 mm.**
- **`parado_de_pé`**: o `upright` cru é 0,944, então a inclinação está boa. Sobra a altura
  da pelve, e ela não está instrumentada (ver A3).

Pista física: `pegar/dof_pos_limits = −0,0704`. Os braços batem no limite de junta ao
erguer. Os 3,0 cm podem ser teto mecânico da pose, não falta de incentivo.

### A3 · instrumento cego

`Episode_Metrics/agachamento_pegar` sai **`nan`**: a função devolve NaN fora do `pegar`
(`observability.py:78`) e o `MetricsManager` propaga. Logo o fator que zera o
`cond_fisica` é indecidível pelo log.

**Conserto:** decompor `cond_fisica` em três colunas novas do `env.diag_soma`, uma por
fator. O `Relatorio` já faz média por tarefa com `index_add_`, então sai como
`pegar/erguida`, `pegar/de_pe`, `pegar/preensao`. Só-log, Categoria A.

### A4 · o `entre_blocos.py` apodreceu

| painel | problema |
|---|---|
| 3 | plota `tarefas_abertas` duas vezes (copy-paste); rótulo diz "de 60", hoje são 12 |
| 4 | 100% dedicado a `parado_push` — tarefa e eixo que **não existem mais** |
| 5 | eixos `("velocidade","altura","giro")` → **cego para o eixo `alvo`** |
| 8 | usa `Metrics/taxa_alvo`, chave do experimento residual |

E falta o que importa: `std` contra o teto, `erro_vel_ang` por tarefa contra o `tol_w`, e
a cadeia `grasp → unload → lift → cond_fisica`.

### A5 · a pega é bamba

`box_shake_pegar` cru = 1,41 com o gate quase aberto ⇒ **|ω_caixa| ≈ 1,45 rad/s**.
Reserva: termo `squeeze` por força de palma com `min(f_esq, f_dir)` e `F_ref ≈ 4,9 N`.
Aplicar só se o `unload` subir e o `lift` não seguir.

### A6 · débito declarado

`NIVEIS_ATIVOS["velocidade"] = 1` ⇒ o `locomover` não treina acima de 1,0 m/s neste
bloco. É de propósito, para não diluir os 12 destravamentos.

### A7 · a escala do `Episode_Termination/*` não é fração

`time_out = 6,4167` prova. Então `fell_over = 0,2917` e `caixa_caiu = 0,3333` **não** são
"29%" e "33%". Só a razão entre eles é segura de ler.

### A8 · infra — Colab free não é upgrade

`Collection 4,558 s` contra `Learning 0,533 s`: **89% do tempo é física**. GPU melhor só
ajuda por **banda de memória**.

| GPU | banda | veredito |
|---|---:|---|
| T4 (Kaggle, Colab free) | 320 GB/s | linha de base |
| **L4 (Colab Pro)** | **300 GB/s** | **não ganha nada** |
| A100 40 GB | 1555 GB/s | ~4,9× — o único salto real |
| RTX 6000 Ada | 960 GB/s | ~3× |

Mesmo zerando o `Learning`, o teto de ganho é 10%. Kaggle = 2× T4, 30 h/semana,
12 h/sessão. Colab free = 1× T4 sem garantia, sessão curta. Use o Colab por **cota**.

---

## 9. Resolvidos — não reabrir

| # | problema | causa | conserto | validação |
|---|---|---|---|---|
| R1 | caixa fora de alcance | `box_xy = 0,50` (default abandonado pela Lift em 16/07) recombinado com `box_jitter_x` calibrado sobre 0,30. P(alcançável) ≈ 0,19, e o `cond_fisica` do `reorientar` mediu 0,17 | `box_xy = 0,30`; `table_jitter_xy` 0,15 → 0,05 | `grasp` 0 → 0,25 · `lift` 0 → 0,32 |
| R2 | robô segurava a caixa **agachado** | `alvo_peito_w` no frame da base: agachar ENCURTAVA o percurso, a linha de chegada descia com a pelve | z ancorado no mundo (0,91 m) + escala dupla 0,30/0,05 | `box_at_peito` 0,0000 → 0,1644 |
| R3 | `tol_w = 0,35` inalcançável | piso real 0,62–0,73 por tarefa, uniforme com e sem comando: oscilação de marcha + `push_robot` moram dentro da integral | `tol_w` → 0,70 | **`sucesso` 0 → 0,2884** · `locomover/perf_n0` → **0,9143** |
| R4 | `pegar` e `reorientar` nunca abriam, **em silêncio** | com um nível só, o `locomover` não tinha eixo nem DR, e a regra do evento caía no `continue` | `or self.eventos_tarefa[t] == 0` | smoke: a cadeia abre as 5 |
| R5 | resume ignorava o congelamento, **em silêncio** | `load_state_dict` restaurava `abertos` cru, então `abertos = 2` sobrevivia e a célula continuava **sorteando** 1,5 m/s | o comprimento da célula manda; corta com aviso | `locomover_velocidade/abertos = 1` onde o ckpt tinha 2 |
| R6 | platô do `pegar` por 22 mil iterações | 6 causas: `_grasp` booleano; nenhum eixo graduava QUANTO erguer; `cond_fisica` só diagnóstico; critério era esfera 3D; episódio de 20 s desperdiçava 17; mover o braço custava 2× | `unload` + eixo `alvo` + `sucesso_denso` + piso em z + episódio por tarefa + `action_rate` ×0,25 nos braços | `lift` cru 0,062 → 0,394 · `unload` 9,70 N → 0,00 N |
| R7 | `CollisionCfg.__init__() missing 2 args` | mjlab do ambiente subiu para 1.6.0 | `pip install mjlab==1.5.3` | — |
| R8 | `AssertionError` no lançamento | notebook afirmava `total_unlocks() == 24`; o congelamento derrubou para 12 | assert corrigido | 10 asserts passam local |
| R9 | as duas células do notebook podiam divergir | `load_checkpoint` digitado literal | usa o `CHECKPOINT` derivado | — |

---

## 10. Como ler o log

### 10.1 Desfazer o orçamento — obrigatório

**`valor cru = contrib / peso efetivo`** (tabela em §4.2). Ler cru me fez afirmar
`grasp = 0,32` quando era 0,718 — erro de 2,2×.

Verificação de graça: `sucesso_denso` não é escalado, então `sucesso_denso / 5,0` tem de
ser **igual** ao `cond_fisica` da mesma tarefa. Conferido em it 23548 no `reorientar`:
`2,0466 / 5,0 = 0,4093` contra `cond_fisica = 0,4093`. Exato.

⚠️ `contrib` = `peso × valor` **sem `dt`** (`reward_manager.py:132` faz `value / scale`).
`Episode_Reward/*` e `_reward_buf` **têm** `dt`. É por isso que `terminacao = −200` no
knob vale −4,0 por evento.

### 10.2 Os números que decidem, em ordem

1. `Mean action std` — passando de **0,50**, nada mais importa (A1).
2. `contrib/<tarefa>/erro_vel_ang` contra 0,70, com o `erro_vel_ang_filt` ao lado. Cru
   reprovando e filtrado baixo = oscilação de marcha, não falta de rastreio.
3. `Episode_Metrics/sucesso` e `orquestrador/*/perf_n0`. Zero com `ALARME_estagnacao`
   ligado = currículo morto.
4. `contrib/pegar/{grasp,unload,lift,cond_fisica}`, cru. A cadeia da tarefa.
5. `contrib/<tarefa>/_total`. É a soma de **todos** os termos, penalidades inclusas.
   Negativo = morrer cedo rende mais.
6. `atribuicao_divergente`. Tem de ser **zero**. Diferente de zero = crédito falso.

### 10.3 Filtros do TensorBoard

| pergunta | regex |
|---|---|
| o `std` está subindo? | `std` |
| a régua está alcançável? | `contrib/.*erro_vel_ang` |
| o currículo andou? | `perf_n0\|eventos\|tarefas_abertas\|ALARME` |
| a cadeia do `pegar` | `contrib/pegar/(grasp\|unload\|lift\|cond_fisica)` |

---

## 11. Inventário de checkpoints

Lido do disco em 18/08 (`infos.curriculum.orquestrador` e
`actor_state_dict["distribution.std_param"]`).

| ckpt | `std_param` médio | `perf[locomover/velocidade]` | `perf[reorientar/giro]` |
|---|---:|---|---|
| `model_13700.pt` | **0,4769** | 0 · 0 · 0 | 0 |
| `model_17297.pt` | 0,4287 | 0 · 0 · 0 | 0 |
| `model_22296.pt` | 0,4284 | **0,9218** · 0,679 · 0 | **0,158** |

Iguais no resto: `eventos = 3`, `abertas = [locomover, pegar, reorientar]`,
`eventos_tarefa = {locomover: 1}`, `dr_peso` todo `False`,
`abertos = {'0|velocidade': 2, '1|altura': 1, ...}`.

Dois fatos que isso fecha:

1. **A era do `tol_w = 0,35` não produziu sucesso nenhum.** O 22296 é o único checkpoint
   com `perf` diferente de zero em toda a história do experimento.
2. **Os três têm a célula do `pegar` como `'1|altura'`**, e hoje ela é `'1|alvo'`. O
   `load_state_dict` itera as células de HOJE e dá `continue` na chave que falta, então o
   `pegar` recomeça do zero em **qualquer um deles**. Documentado na docstring; não é bug.

**Custo de voltar do 22296 para o 13700:** perde `perf[reorientar] = 0,158` (longe do
portão de 0,90) e a EMA do `locomover`, que é **inerte** — `eventos_tarefa[locomover]` já
é 1 e o `NIVEIS_ATIVOS` congela o eixo dele, então ele nunca dispara outro evento. O custo
real é 8600 iterações de peso.

**Risco concreto:** `unload` e `lift` são os DOIS multiplicados por `_grasp`. Checkpoint
com preensão fraca enfraquece justamente os termos novos. No 22296 o `grasp` cru era
0,851; no 12298 era 0,36, e o 13700 está 1400 iterações depois do 12298.

**Ganho em troca:** política menos convertida no ótimo local "agacha, mão na caixa, não
ergue".

**Como decidir em 50 iterações em vez de 2000:** `contrib/pegar/grasp ÷ 0,444` no
primeiro print. ≥ 0,5 e os termos novos têm com o que trabalhar; ~0,35 e o `unload` vai
pagar pouco.

---

## 12. Regras que saíram desta sessão

1. **Reaquecer o `std` só serve para gradiente ESPARSO.** O do `unload` é denso — o robô
   já está com as mãos na caixa. Reaquecer custou uma run inteira. LR baixa no degrau
   sim; `std` alto não.
2. **Nunca ler `contrib/*` sem desfazer o orçamento** (§10.1).
3. **`sucesso_denso` e `Sucesso` medem coisas diferentes de propósito** (§4.3). O reward
   pode pagar enquanto o currículo se recusa a creditar — foi o que houve com o
   `reorientar` (`sucesso_denso` 2,05 com `perf_n0 = 0`).
4. **Categoria A × C.** Peso de reward é A: retoma do checkpoint. Largura da obs e
   definição de sucesso são C: recomeça.
5. **Checkpoint mais recente não é checkpoint melhor.** As duas últimas runs pioraram a
   política. A eleição por "maior iteração" no notebook assume monotonicidade que os
   blocos 4-a e 4-b violaram.

---

## 13. Pendências, em ordem

1. **Teto no `std`** (A1). Bloqueia tudo. Categoria A, ~10 linhas no `runner.py`.
2. **Log do `std` pernas × braços** (A1). 3 linhas; decide se o teto é global ou por grupo.
3. **Decompor `cond_fisica`** em `erguida` / `de_pé` / `preensão` (A3). Sem isso o A2 fica
   indecidível.
4. **Consertar `entre_blocos.py`** (A4).
5. Reserva condicional, só com evidência: `squeeze` (A5) · `upright` → 0,2 no `pegar` ·
   autoridade ao `soft_landing_table`.
6. Limpeza: `EXPERIMENTO.md` ainda diz "implementação não iniciada", 11 dias e 4 blocos
   depois.

---

## 14. Como regerar as tabelas deste doc

```python
# fatores do orçamento e peso efetivo por tarefa
import sys, os; sys.path.insert(0, "."); os.environ.setdefault("MUJOCO_GL", "egl")
import g1_multitask
from g1_multitask import tasks as T
from mjlab.scripts.train import TrainConfig
cfg = TrainConfig.from_task(g1_multitask.TASK_ID)

orc = {t: 0.0 for t in range(T.NUM_TASKS)}
for n in T.TERMOS_DE_TAREFA:
    if n in cfg.env.rewards:
        for t in cfg.env.rewards[n].params["tasks"]:
            orc[t] += abs(cfg.env.rewards[n].weight)
for t in range(T.NUM_TASKS):
    print(T.NAMES[t], orc[t], 4.0 / orc[t])
```

Fontes: `tasks.py` (eixos e destravamentos) · `knobs.py` (pesos e tolerâncias) ·
`env.py` (gates e `_equaliza_orcamento`) · `metrics.py` (a régua) ·
`curriculum.py` (a regra do evento) · `observability.py` (as chaves de log) ·
`docs/adrs/0001-curriculo-do-pegar-rampa-ponte-e-bonus.md` (as alternativas rejeitadas).
