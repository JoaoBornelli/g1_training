# g1_limpo — dois bits: a cena, o alvo e a cadeia (v3.1)

Data: 2026-09-08. Branch: `exp/g1-limpo-v2`. Base: `c3c9d08`. Retomada: `model_6999.pt` (bloco10).

v3.1 incorpora duas revisões (a do PM, por estado da máquina; uma independente, por leitura do código inteiro). O changelog está no §9.

Contrato de observação NÃO muda: ator 114, crítico 131. O warm-start carrega.

## 0. Princípio

Um problema se resolve REMOVENDO ou MODIFICANDO a função existente. Termo, gate ou tabela novos só entram se um velho sai.

| | hoje | depois |
|---|---|---|
| termos de recompensa | 30 | **28** (`largou`, `pose_de_braco`, `sustentacao` saem; `load` volta) |
| terminações | 3 | 3 (`fell_over` ganha uma cláusula) |
| cadeias | 4 | **3** |
| variantes de CARREGAR | 3 | **0** (o estado fica; fecho e twist fixo saem) |
| tabela `prob_por_nivel` | 7 × 4 | **sai** |

### 0.1 O que foi medido e justifica cada mudança

Sondas em CPU, `model_5950`, 64 envs, 20 s (`scratchpad/sonda_*_v2.py`, `sonda_abertura.py`, `sonda_relogio.py`).

- **BOTAR é alcançado e nunca fecha.** 46,3% chegam; 0 de 190 fecham.
- **A abertura do BOTAR mata.** Sem espera entre CARREGAR e BOTAR, o alvo salta 0,83 m (p50) a 2,29 m (p90) com VALIDA ligado. `action_rate_l2` a −25/s no passo; palmas soltam no mesmo passo; RMS de junta 0,99 → 6,88 rad/s; 82% morrem em 1 s. A abertura do PEGAR, com espera e máscara, tem 0% de queda.
- **O alvo ancora na origem do env** e o robô já não está lá.
- **O robô deriva com twist zero.** Base a 0,29 m (p50) no início do segurar-parado e 0,85 m no fim; p90 4,65 m. O alvo persegue a base todo passo: derivar é grátis.
- **`push_robot` ativo em todo env**: 1–3 s, até 0,5 m/s. Cobrar o passo cobra a recuperação. As penalidades de pé NÃO mudam.
- **O relógio do hold zera a 99%**: `perto` oscila no limiar de 0,10 m (71 de 71 resets). E a caixa fica a ~0,15 m da pelve — `peito_b.x = 0,25` está 10 cm à frente de onde ela encosta.
- **Pairar no BOTAR paga igual a apoiar.** Nada paga `apoiada`; `unload` e `postura_ereta` pagam por NÃO apoiar.
- **G2 inerte** por 1022 iterações: `1 − exp(−x)` satura (derivada 0,27 em x = 1,3).
- **Joelho de lado**: `hip_yaw` ±158°, `hip_roll` −30°/+170°; `dof_pos_limits` só no limite; `std_standing = {".*": 0,05}` vale ~0 e não puxa. E `PosturaPorElo` devolve 1,0 fora de `ELOS_QUE_ANDAM` — o termo nem age no PEGAR.

## 1. Os dois bits

Cena e alvo são função de: **o twist é zero?** e **a caixa está na mão?** (`_pegou ∧ ¬_soltou`).

| twist | caixa na mão | laje | caixa | alvo da caixa |
|---|---|---|---|---|
| = 0 | não | presente | na laje, **mascarada** | do elo |
| = 0 | sim | presente | nas mãos, visível | **congelado em mundo** |
| ≠ 0 | sim | +5 m | nas mãos, visível | **referenciado no robô**, todo passo |
| ≠ 0 | não | +5 m | +5 m, junto | nenhum |

### 1.1 Twist zero — `comando._zera_twist_nos_parados`

```python
parados = (torch.isin(self._elo, elos_parados) & ~self._soltou) | (self._espera > 0.0)
```

- `elos_parados` continua `(REORIENTAR, PEGAR, BOTAR)`.
- `& ~_soltou`: na cauda pós-BOTAR o `_elo` fica BOTAR (§2.3) e o twist tem de fluir.
- **Sai** `| _segura_parado(...)`.
- **Sai** o bloco P5 inteiro (`anda_c`, `novos`, `_twist_carregar`, `_twist_valido`), os buffers e os resets. O CARREGAR recebe o twist do fabricante **sem filtro** — decisão do dono: a cauda de B é `normal`.
- ⚠ **ORDEM NO PASSO** (revisão, item 32): hoje `_zera_twist_nos_parados` (~795) roda ANTES de `_aplica_espera` (~805). Ela passa a rodar DEPOIS, para ler `_elo`, `_espera` e `_soltou` do passo corrente. Sem isso `limpo_twist_zerado` fica um passo atrasado e o gate de §1.2 lê valor velho.

### 1.2 Alvo — o quadro de referência é decidido pelo twist

`_alvo_ancorado_na_base` passa a rodar em TRÊS momentos, e só neles:

1. **Abertura do elo** (`_aplica_elo`, ramos PEGAR e CARREGAR) — como hoje.
2. **Fim da espera** (`_aplica_espera`, bloco `liga`, junto com `_recalcula_sigmas` e `_pos_no_elo`) — **novo para PEGAR e CARREGAR**. Com push ativo, 0,5–1,5 s de espera movem o robô; o alvo tem de nascer da pose de quando a tarefa abre, não do reset.
3. **Todo passo, só com twist ≠ 0**: `anda = (_elo == CARREGAR) & (limpo_twist_zerado < 0.5)`.

Com twist zero, `_command[:, ALVO]` fica com o que foi escrito em 1 ou 2. É o ponto à frente do robô onde o alvo estaria no instante em que o twist zerou. Nenhum buffer novo.

⚠ A reancoragem por passo do REORIENTAR (~791, alvo = a própria caixa) **não muda**: ela ancora na caixa, não na base.

**`peito_b.x`** (revisão, item 4): medir a posição da caixa segurada no `model_5950` (p50 de `caixa_b.x` no hold, por faixa de `meia_aresta`) e ajustar. Candidato: `peito_b.x = 0,10 + meia_x`. Registrar a medição no knob. Sem isso `perto` vive no limiar.

### 1.3 Laje — `comando._laje_para`

Assinatura vira `_laje_para(ids, topo, *, sobe_caixa=False, xy=None)`. `xy=None` usa `org + prateleira_xy`.

| chamador | `topo` | `sobe_caixa` | `xy` |
|---|---|---|---|
| reset do ANDAR | `afasta_z` | `True` | default |
| fim da espera com twist a ligar (§2.3) | `afasta_z` | `~pegou \| soltou` | default |
| abertura do BOTAR (§1.4) | `limpo_topo ± δ` sob o guarda | `False` | `base_xy + quat_apply_yaw(base_q, (prateleira_xy[0] + δ, δ))` |

**Sai** a chamada `_laje_para(m, afasta_z, sobe_caixa=False)` do ramo CARREGAR de `_aplica_elo`.

`quat_apply_yaw` existe em `mjlab/utils/lab_api/math.py:673`.

### 1.4 Abertura do BOTAR — ramo BOTAR de `_aplica_elo`

Usa a pose **corrente** da base (revisão, item 5: `_pos_no_elo` é o do PEGAR e o robô se moveu ao alcançar).

```python
base_p, base_q = root_link_pos_w[m], root_link_quat_w[m]
topo0 = self._env.limpo_topo[m]
dtopo = c.botar_delta_topo * (2*torch.rand(k) - 1)
teto  = torch.clamp(fundo - c.botar_folga_laje, max=_TOPO_TETO_FISICO)     # guarda de hoje; 0,80 vira constante
topo  = torch.minimum(topo0 + dtopo, teto).clamp(min=c.prateleira_topo_piso)
dxy   = c.botar_delta_xy * (2*torch.rand(k, 2) - 1)
xy_laje = base_p[:, :2] + quat_apply_yaw(base_q, (prateleira_xy[0] + dxy[:,0], dxy[:,1]))
self._laje_para(m, topo, xy=xy_laje)
# ⚠ o ALVO fica na BORDA PERTO do tampo, não no centro (revisão, item 14). Hoje `botar_x`
# 0,30–0,40 contra laje em 0,50 põe o alvo a 0,10–0,20 m da borda. No centro, o robô
# alcançaria por cima de 20 cm de tampo — defeito datado em 16/07 — e a 0,04 m de topo
# a distância passa de ALCANCE_R = 0,85.
a[:, :2] = xy_laje - quat_apply_yaw(base_q, (c.botar_recuo_borda + dxy[:,0]*0.5, 0)) + lateral pequeno
a[:, 2]  = topo + meia_z
```

Knobs: **saem** `botar_x`, `botar_y`, `botar_topo_piso`, `botar_topo_teto`. **Entram** `botar_delta_topo = 0.10`, `botar_delta_xy = 0.10`, `botar_recuo_borda = 0.15`. Saldo 4 → 3.

⚠ O guarda `teto = fundo − botar_folga_laje` **fica**.
⚠ A laje nasce no **fim da espera** com a caixa, no mesmo passo em que o alvo é escrito (§2.3). A laje não é observada.
⚠ **Penetração** (revisão, item 27): `write_mocap_pose_to_sim` não checa contato. Depois de implementar, medir em CPU a força em `apoio_caixa` e `auto_colisao` no passo do avanço para BOTAR, níveis 0 e 4. Se houver pico, `prateleira_xy[0]` na chamada sobe para 0,55.

## 2. A cadeia

### 2.1 Regra única

**Todo elo é precedido por uma espera. A cadeia termina em espera → cauda.**

```
A:  ANDAR(normal)
B:  espera → PEGAR → espera(caixa) → CARREGAR com twist normal
C:  espera → PEGAR → espera(caixa) → BOTAR → espera → cauda ANDAR (interno BOTAR)
R:  espera → REORIENTAR → espera → PEGAR → espera(caixa) → CARREGAR com twist normal
```

`CADEIAS = ((PEGAR,), (REORIENTAR, PEGAR), (PEGAR, BOTAR))` — B, R, C. **Sai** `(PEGAR, CARREGAR)` e o CARREGAR de dentro das tuplas. `_N_ELOS`, `_ELO_EM`, `_PRIMEIRO_ELO` derivam. **Sai** `_SEGURA_PARADO`.

O CARREGAR é o **estado de cauda** de quem fechou o PEGAR e não vai botar: VALIDA 1, alvo no peito por passo, twist do fabricante, sem fecho. Fica no one-hot e na recompensa.

**Interruptor** (revisão, item 31): `prob_por_nivel = ()` era o desliga da máquina de elo (F0–F3). Vira knob `cadeia.ativa: bool = True`. `smoke.py` e `inspeciona.py` montam cfgs com ele em `False` onde hoje passam a tabela vazia.

### 2.2 Fecho → espera — `comando._avanca_elo_force`

O fecho **arma a espera** e não avança:

```python
self.fechou[ids] = True
self._fechos[ids] += (ganho & tem).long()        # ⚠ o guarda `tem` FICA (revisão, item 6): sem ele o inspetor credita fecho a env de ANDAR
self._espera[ids] = sorteio na faixa da espera inicial
self._sigma_pendente[ids] = True
self._sust[ids] = 0.0                            # (revisão, item 9) senão `_sust` congela no valor do fecho
solta = ids[self._elo[ids] == BOTAR]
self._soltou[solta] = True; self._env.limpo_soltou[solta] = 1.0
```

`_elo` e `_passo` **não mudam** aqui. `_avanca_elo` já pula `fechou`.

`fechou` passa a significar "o elo corrente fechou". **Cadeia concluída** = `fechou ∧ (_passo == n_elos − 1)`. Esse predicado ganha um método `concluiu(ids)` e é a ÚNICA definição de sucesso (revisão, itens 20–21): `metrics["sucesso"]`, `curriculo.nivel` (hoje lê `cmd.fechou` sozinho) e o balanceador leem o mesmo.

`metrics["sucesso"][ids] = 1.0` é escrito **aqui**, no instante do fecho terminal (revisão, item 1: métrica escrita no `_resample` sai um episódio atrasada).

⚠ **Viewer** (revisão, item 5): `eventos.avanca_elo_no_viewer` chama `forca_avanco` a cada dt. Ele passa a chamar o caminho de avanço de §2.3 (fim de espera forçado), senão rearma a espera todo passo e congela.

### 2.3 Fim da espera → próximo elo ou cauda — `comando._aplica_espera`

```python
acabou   = self.fechou & ~aguardando
tem_prox = self._passo + 1 < n_elos
avanca   = acabou & tem_prox
if pegou: avanca &= self._perto(ids)        # checagem ÚNICA; `_perto` extraído de `_fecha_elo_corrente` (revisão, item 29)
ids = envs[avanca]:
    _passo += 1; _elo = _ELO_EM[cad, _passo]; fechou = False
    _sust = 0; _sustain_alvo = _sustain_alvo_de(ids)
    _aplica_elo(ids)                        # BOTAR: laje ±δ e alvo (§1.4)
    _recalcula_sigmas(ids); _pos_no_elo = pose atual; _sigma_pendente = False
    metrics["avancos"][ids] += 1            # (revisão, item 28) mudou de lugar
    metricas.aproxima_caixa: zerar o mínimo corrente destes ids   # (revisão, item 24)

cauda = acabou & ~tem_prox & ~ja_em_cauda
ids = envs[cauda]:
    se pegou ∧ ¬soltou:  _elo = CARREGAR; _alvo_ancorado_na_base(ids)     # B, R
    senão:               _elo FICA BOTAR (revisão, item 3: o crítico vê interno BOTAR + publicado ANDAR e prevê a renda)
    _laje_para(ids, afasta_z, sobe_caixa=~pegou | soltou)
```

⚠ Quem falha `perto` no fim da espera não fica preso: `acabou` é reavaliado todo passo.

**Publicação:**

```python
publica_andar = self._soltou | (aguardando & ~self._pegou)
```

- Antes do PEGAR: `pegou = 0` → ANDAR → `observacoes.caixa_no_frame_da_base` zera os 10 canais.
- Espera com caixa: publica o interno (PEGAR) → caixa visível.
- Depois do BOTAR: `soltou` → ANDAR → caixa mascarada. O robô "deixa de a considerar".

`VALIDA = (interno ≠ ANDAR) ∧ ¬aguardando` **não muda**. Na cauda pós-BOTAR VALIDA fica 1, mas a caixa está a +5 m e os kernels valem 0.

### 2.4 Fecho por elo — `comando._fecha_elo_corrente`

| elo | vira |
|---|---|
| REORIENTAR | igual |
| PEGAR | `perto ∧ alinhado ∧ de_pe` |
| CARREGAR | **ramo sai** |
| BOTAR | `perto ∧ alinhado ∧ apoiada ∧ de_pe` |

**`de_pe`** vira pose padrão de pernas e cintura:

```python
juntas = complemento de JUNTAS_BRACO
dq = (q[:, juntas] − q_default[:, juntas]).abs().amax(dim=-1)
de_pe = dq <= c.de_pe_tol_rad
```

⚠ **`de_pe_tol_rad` É MEDIDO EM TRÊS CONDIÇÕES** (revisão, itens 6 e 15): parado sem caixa; segurando parado com caixa; e **no PEGAR dos níveis 4–6** (laje a 0,04 m, que exige agachar). Tomar o p90 de `dq` no instante em que o robô ESTÁ DE PÉ com a caixa erguida (não durante o agachamento): `de_pe` é condição de FECHO, depois de erguer. Fallback `0,35`. Se o p90 do nível 6 for muito maior, é sinal de que o robô fecha o PEGAR agachado hoje — registrar.

`pelve_alvo` **fica** (`postura_ereta`). **Saem**: `andou`, `carregar_dist_m`, `carregar_s`, `_segurar`, `_segura_parado()`, ramo CARREGAR de `_sustain_alvo_de`.

### 2.5 Sorteio da cadeia — balanceador

**Sai** `prob_por_nivel`. `nivel` continua governando a dificuldade física.

```python
p_C = clamp((1 − s_C) / ((1 − s_B) + (1 − s_C) + 1e-6), piso, 1 − piso);  p_B = 1 − p_C
```

- `compat` continua: elo REORIENTAR → só R; elo PEGAR → B ou C. A fração do REORIENTAR já vem de `pesos_dos_sorteaveis` no sorteio de ELO — **não** há `p_R` (revisão, item 26).
- `s_B`, `s_C`: médias móveis de `concluiu` por cadeia. ⚠ **Atualizadas uma vez por iteração de PPO**, não por reset (revisão, item 19): acumular `(n_concluiu, n_episodios)` por cadeia nos resets e aplicar `s ← (1−α)s + α·(n_concluiu/n_episodios)` no mesmo relógio que `knobs.Forma.passos_por_iteracao` usa. `balanceador_alpha = 0.05` por iteração.
- ⚠ **Semente `s_C = 1, s_B = 0`** (revisão, item 17): `p_C` nasce no piso (0,20) e ABRE conforme C falha. Sem isso 50% da manipulação cai no BOTAR na iteração 0.
- ⚠ **Checkpoint** (revisão, item 18): `s_B`, `s_C` entram nos escalares que `RunnerComEstadoDeCurriculo` salva e restaura (`runner.py:43-45`, `CHAVES_ESCALARES`). Senão todo resume volta a `p_C = 0,20`.
- Knobs **novos**: `balanceador_piso = 0.20`, `balanceador_alpha = 0.05`. Saldo: 28 números → 2.

**Limitação declarada:** `concluiu` em B mede "PEGAR fechou", não a qualidade do carregar. A qualidade fica visível em `Metrics/twist/eficiencia_*` da cauda. Uma condição a mais no sucesso misturaria taxa de queda com conclusão (revisão, item 16) e enviesaria `p_C` para 0,5.

### 2.6 `renda_congelada`

- `return self.congelado * _valida(...)` vira `return self.congelado`. Com esperas entre elos, o `× _valida` zerava a renda ganha em toda espera.
- `TERMOS_CONGELAVEIS` (`env_cfg.py:96-99`): **saem** `track_linear_velocity` e `track_angular_velocity` (revisão, item 12: no fecho do PEGAR o twist é zero e eles pagam ~4/s por "rastrear zero"; congelar isso e pagar de novo ao vivo na cauda conta o mesmo canal duas vezes). **Sai** `sustentacao`; **entra** `load`. A lista fica só com incentivos de manipulação.

### 2.7 Termos — `recompensas.py`, `knobs.py`, `env_cfg.py`

| termo | ação | motivo |
|---|---|---|
| `largou` (+ `sigma_solta`) | **sai** | a cauda é ANDAR com twist; sair andando já tira as mãos; `escapou` já desarma por `soltou` |
| `pose_de_braco` (+ 2 knobs) | **sai** | redundante com §3.1 em standing; na cauda pós-BOTAR a espera em standing traz os braços antes do twist ligar |
| `sustentacao` | **sai** | 0,19 no painel; redundante com o fecho; e na cauda ficaria travado em 1,0 (revisão, item 9) |
| `load` | **volta**: `(1 − descarga) × perto × _valida`, só em BOTAR (gate `~_fora_do_botar`, que já existe), peso 2,0 | nada paga `apoiada`; pairar a 1 cm do alvo vale o mesmo que apoiar |
| `unload`, `postura_ereta` | **zeram em BOTAR** (mesmo gate) | pagam por NÃO apoiar; em BOTAR são o incentivo errado. É a máscara que o `g1_poc` tinha |
| `squeeze` | igual | pega e apoio precisam de preensão |
| `rastreio_por_elo` | `engajado = env.limpo_pegou` (sem `× _valida`) | na espera com caixa VALIDA é 0 e o rastreio de v=0 pagava zero: derivar grátis (revisão do PM, item 2) |

Aritmética do BOTAR depois disto (por segundo, pesos aplicados, congelado F à parte): pairar a 1 cm = `staged 6 + precise_pos 3 + ori 1 + squeeze 1 = 11`; apoiado antes do fecho = `11 + load 2 = 13`; depois do fecho `F += 13`, cauda = `F + 13 + rastreio`. Apoiar vence pairar em todo instante.

## 3. Postura, terminação e velocidade

### 3.1 `pose` em standing — `knobs.std_standing` e `recompensas.PosturaPorElo`

Duas quebras que a revisão achou (itens 2 e 3): `PosturaPorElo.__call__` devolve **1,0 fora de `ELOS_QUE_ANDAM`** — o termo nunca age no PEGAR, que é onde o joelho vai ao chão; e `variable_posture.__call__` do molde devolve um **escalar** — não dá para zerar 14 juntas por fora.

Vira: `PosturaPorElo.__call__` **reimplementa o cálculo** (não chama o do molde), sem neutralização por elo:

```python
std   = std por regime (standing/walking/running), como o molde
err2  = ((q − q_default) / std) ** 2                       # (n, 29)
ativa = ones(29); ativa[JUNTAS_BRACO] = 0 onde (pegou ∧ ¬soltou)   # braço trabalhando sai da conta
return exp(−(err2 × ativa).sum(-1) / ativa.sum(-1))         # média sobre as juntas ATIVAS
```

`std_standing` vira dict por padrão (formato de `std_walking`):

```python
{r".*hip_yaw.*": 0.30, r".*hip_roll.*": 0.30,
 r".*hip_pitch.*": 0.50, r".*knee.*": 0.50, r".*ankle_pitch.*": 0.50, r".*ankle_roll.*": 0.30,
 r".*waist.*": 0.30,
 r".*shoulder.*": 1.00, r".*elbow.*": 1.00, r".*wrist.*": 1.00}
```

⚠ Números são ponto de partida. Validar num script de CPU sem env, **com o divisor real** (15 juntas de perna+cintura quando os braços saem): `exp(−mean)` ≥ 0,8 a 0,1 rad uniforme; ≤ 0,3 a 0,6 rad. Registrar a tabela no knob.

⚠ `smoke.py:426-430`, `smoke.py:1160-1167` e `env_cfg.py:114-124` (`colhe_sigmas_de_postura`) afirmam que `std_standing` é o objeto do molde e vale `{".*": 0.05}` (revisão, item 4). Migrar os três.

### 3.2 `fell_over` — `env_cfg.py` + `terminacoes.py`

Mesmo slot `cfg.terminations["fell_over"]`, função do `g1_limpo`:

```python
def caiu(env, limit_angle, joelho_z_min, asset_cfg):
    tombou   = bad_orientation(env, limit_angle, asset_cfg)     # 70° ficam
    z_joelho = altura dos corpos *knee* − origem do env
    return tombou | (z_joelho.amin(-1) < joelho_z_min)
```

Knob novo `joelho_z_min`. ⚠ **MEDIR NO AGACHAMENTO** (revisão, item 15): p10 da altura do joelho no PEGAR dos níveis 4–6 no `play`, não andando. Isto é TERMINAÇÃO: um limiar alto mata o agachamento legítimo. O knob fica **abaixo** do p10 do agachamento e **acima** do joelho no chão (~0,05). Fallback `0,10`. Sem sensor novo: `body_pos_w`.

### 3.3 G2 — `recompensas.velocidade_por_regime`

```python
return torch.clamp(torch.mean((v / vmax) ** 2, dim=1), max=4.0)
```

Derivada 1 até 2× o limite (o dobro do operável); teto −8,0/s acima. A revisão (item 13) aponta derivada zero no teto. Aceito: acima de 2× vmax o robô está caindo ou abrindo com violência, e −8/s por 5 passos (−0,8) é menor que `terminacao` (−4). O que importa é a derivada em x ∈ [0,5 ; 3], e ela é 1. Peso −2,0 fica. Atualizar docstring e knob.

⚠ Colateral: `walking`/`running` custam ~0,4/s em vez de ~0,1/s. Se `eficiencia_media` cair mais, `vel_max_walking`/`running` sobem 1,5×. Não mexer agora.

### 3.4 `PPOPorElo` — `algoritmo.py:120`

Hoje agrupa em **dois** grupos: `argmax == ANDAR` e o resto. Com a cauda CARREGAR longa (retorno alto, variância baixa) no mesmo grupo que PEGAR e BOTAR (curtos, variáveis), a vantagem do elo de trabalho é dividida pelo desvio da cauda (revisão, item 25).

Vira: agrupar pelos **cinco slots** de `elo_interno`. O invariante do one-hot (`algoritmo.py:96-100`) já é conferido. Nenhum canal novo.

## 4. O que NÃO muda

- Observações e o contrato 114/131.
- `forma` (loco × manip) e o piso de 30%. `nivel` e a dificuldade física.
- As quatro penalidades de pé e `command_threshold` (o push exige passo).
- `push_robot`. Os 70° do `fell_over`.
- `track_linear_velocity`, `track_angular_velocity`.
- A mecânica de congelar de `renda_congelada` (só o `× _valida` e a lista mudam).
- `caixa_largada`, `caiu`, `escapou`.
- F1 (σ no fim da espera): agora roda em toda espera.
- `ELOS_SORTEAVEIS`, `pesos_dos_sorteaveis`, `prob_reorientar_inerte`.

## 5. Smoke — rodar UMA vez, no fim

**Cai e migra**: tudo que referencia `prob_por_nivel`, `cadeia_forcada = 2`, `_segura_parado`, `_SEGURA_PARADO`, `andou`, `carregar_dist_m`, `carregar_s`, `_segurar`, `_twist_carregar`, `_twist_valido`, `largou`, `sigma_solta`, `pose_de_braco`, `sustentacao`, `botar_x/y/topo_piso/topo_teto`, `std_standing == {".*": 0.05}` (`smoke.py:426-430`, `:1160-1167`), a contagem 30, e `inspeciona.py:271` (`botar_topo_piso`). `smoke.py:4175` já afirma `largou`/`pose_de_braco` fora de `TERMOS_CONGELAVEIS` — fica.

**Novos** (seção `--- v3.1: dois bits`):

1. `CADEIAS` tem 3 entradas sem CARREGAR; `_N_ELOS = (1, 2, 2)`.
2. Twist zero ⟹ laje presente; twist ≠ 0 ⟹ laje a `afasta_z`, caixa junto iff `~pegou | soltou`.
3. Alvo congelado: PEGAR com twist zero, base movida 0,3 m à mão → `_command[ALVO]` inalterado. CARREGAR com twist ≠ 0 → muda.
4. Depois de um fecho: `fechou = True`, `_espera > 0`, `_elo` inalterado, `_sust = 0`, `_fechos` só sobe com `tem`.
5. Publicação: `pegou=0 ∧ aguardando` → ANDAR e canais da caixa zero; `pegou=1 ∧ ¬soltou ∧ aguardando` → interno e canais ≠ 0; `soltou` → ANDAR.
6. Fim da espera com caixa e `perto` falso → não avança; `perto` verdadeiro no passo seguinte → avança. `avancos` incrementa AQUI.
7. BOTAR abre com laje a `‖xy_laje − base_xy‖ ≈ prateleira_xy[0] ± δ`, `|topo − topo0| ≤ δ_topo`, alvo a `botar_recuo_borda` do centro para a borda perto.
8. `renda_congelada` paga durante `aguardando` depois de um fecho; `TERMOS_CONGELAVEIS` sem rastreio, sem `sustentacao`, com `load`.
9. CARREGAR nunca fecha.
10. `p_C ∈ [piso, 1−piso]`; semente dá `p_C = piso`; `s_C = 0, s_B = 1` dá `1 − piso`; EMA atualiza só na borda de iteração.
11. `concluiu` = 1 só com `fechou ∧ último elo`; `metrics["sucesso"]` escrita no fecho; `curriculo.nivel` lê `concluiu`, não `fechou`.
12. G2: v = 0 → 0; v = vmax → 1,0; v = 3·vmax → 4,0.
13. `de_pe`: default → True; joelho a +0,8 rad → False; pelve baixa com pernas default → True.
14. `fell_over`: de pé → False; joelho a z = 0,05 → True; pelve a 75° → True; joelho a `joelho_z_min + 0,05` → False.
15. `PosturaPorElo`: com `pegou ∧ ¬soltou` e braços a 1,2 rad, valor igual ao de braços na default; em PEGAR o termo NÃO é 1,0 constante; a média usa o divisor das juntas ativas.
16. `load` em BOTAR: caixa pairando → 0; apoiada no alvo → ~1; fora de BOTAR → 0. `unload` em BOTAR → 0.
17. Cauda pós-BOTAR: `_elo == BOTAR`, `limpo_twist_zerado == 0`, publicado ANDAR.
18. `PPOPorElo` produz 5 grupos quando os 5 elos estão presentes.
19. `cadeia.ativa = False` reproduz o comportamento de hoje com `prob_por_nivel = ()`.
20. Contagem: 28 termos, 3 terminações.
21. `s_B`, `s_C` sobrevivem a `save` → `load` do runner.
22. Viewer: `--avanca-elo` avança (não congela na espera).

Meta: 0 falhas.

## 6. Retomada

- `model_6999.pt`, LR 5e-4, 2000 iterações.
- Notebook: `esperado 30` → 28; `largou`/`pose_de_braco`/`sustentacao` ausentes; `load` presente; `len(CADEIAS) == 3`; `prob_por_nivel` ausente; `botar_delta_topo` presente; `s_B`/`s_C` no checkpoint.
- **Primeiro log** (não antes de +150): `sucesso` acima de 0,46 com `fatia_cadeia` alta é o sinal do BOTAR; `impacto_da_caixa` e `velocidade_de_junta` em queda; `fell_over` **não** pode subir (o `de_pe` e o joelho endurecem); `eficiencia_media` pelo colateral do G2; `Curriculum` deve mostrar `p_C` subindo do piso.
- **Medições pós-implementação, antes do resume** (CPU, `model_6999`): penetração da laje no avanço (§1.4); `dq` e altura do joelho nos níveis 4–6 (§2.4, §3.2); `caixa_b.x` no hold (§1.2).

## 7. Commits

Atômicos, Conventional Commits, `git -c core.hooksPath=/dev/null`, **sem `Co-Authored-By`**, sem push. Um por seção: §1, §2, §3, §5, notebook à parte.

## 8. Refutados da revisão independente, com motivo

| item | por quê não entra |
|---|---|
| 10 (cauda reancora por passo → deriva grátis) | na cauda o movimento é **comandado**; desviar do twist custa rastreio. "Deriva" só existe com twist zero |
| 11 (10% de comando zero na cauda de B paga ~19,5/s parado) | o robô é comandado a ficar parado com a caixa e o faz. Andar paga o mesmo mais rastreio. Não há preferência por parar |
| 13 (teto do G2 com derivada zero) | ver §3.3: o operável é x ∈ [0,5 ; 3] e ali a derivada é 1 |
| 22 (`pose_de_braco` protege os braços na cauda) | a espera pós-BOTAR roda em standing com σ 1,0 e traz os braços ANTES do twist ligar; na marcha os braços seguem `std_walking` como em A |

Item 23 (`fracao_esperando` mistura estados): aceito como **docstring**: a métrica passa a chamar-se o que mede — "sem tarefa ativa" (`aguardando ∨ soltou`). Sem código novo.

## 9. Changelog v3 → v3.1

Graves: `load` volta e `sustentacao` sai; `unload`/`postura_ereta` zeram em BOTAR; `engajado = pegou`; cauda pós-BOTAR mantém `_elo = BOTAR`; `PosturaPorElo` reimplementa o cálculo e perde a neutralização por elo; `PPOPorElo` em 5 grupos; `sucesso` escrito no fecho e unificado com `nivel`; ordem `_aplica_espera` → `_zera_twist`. Médios: `peito_b.x` medido; laje na pose corrente; alvo na borda perto; `de_pe_tol`/`joelho_z_min` medidos no agachamento; rastreios fora de `TERMOS_CONGELAVEIS`; `_fechos` com `tem`; `s_B/s_C` no checkpoint, por iteração, semeados no piso; `cadeia.ativa`; `_perto` extraído; `avancos` e `aproxima_caixa` no avanço; viewer. Cortados: `p_R`; a instrução no-op de `TERMOS_CONGELAVEIS`.
