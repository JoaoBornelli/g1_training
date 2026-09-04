# Auditoria de gradientes — g1_limpo (bloco8, iteração 1503)

Data: 2026-09-04. Branch: `exp/g1-limpo-v2`. Só leitura. Nenhum arquivo de código foi alterado.

Regra do dono: em todo instante o robô tem um incentivo da posição atual até o alvo do elo, e do fecho de um elo até o elo seguinte. Um trecho sem incentivo é um buraco.

Regra do dono para pares: o gradiente macro e o preciso têm a mesma forma. O macro tem raio grande e peso menor. O preciso tem raio pequeno e peso maior.

## 1. Fontes lidas

- `g1_limpo/recompensas.py` — os 13 termos próprios e os 3 wrappers do molde.
- `g1_limpo/env_cfg.py:180-300, 470-600` — a fiação, os `ELOS_QUE_ANDAM`, o `rastreio_por_elo`.
- `g1_limpo/knobs.py` — `Recompensa`, `Tarefa`, `Alvo`, `Cena`, `Nivel`, `Cadeia`, `Terminacao`.
- `g1_limpo/comando.py` — `_fecha_elo_corrente`, `_avanca_elo`, `_avanca_elo_force`, `_aplica_elo`, `_aplica_espera`, `_recalcula_sigmas`, `_alvo_ancorado_na_base`, `alvos_das_palmas`.
- `g1_limpo/terminacoes.py`, `g1_limpo/eventos.py:142-215`.
- `mjlab/tasks/velocity/mdp/rewards.py`, `velocity_env_cfg.py:183-193, 276-296`.

## 2. Geometria do PEGAR no nível 0

- Caixa no reset: `x = 0,32 + U(0; 0,20)`, `y = U(−0,18; 0,18)`, `z = topo + meia`. Topo 0,55 ± 0,02. Meia-aresta 0,035 a 0,065. Centro da caixa em z ≈ 0,59 a 0,62.
- Base no reset: `x ∈ (−0,10; 0)`, pelve em z ≈ 0,80.
- Alvo do PEGAR: `base + R(0,25; 0; ·)`, `z = 0,95` absoluto. Alvo em (≈0,15 a 0,25; ≈0; 0,95).
- `d₀` caixa→alvo ≈ 0,36 a 0,50 m. A componente vertical é ≈ 0,35 m e domina.
- `d_palma₀` ≈ 0,34 m (medido: 0,21 a 0,48).

## 3. Inventário — todo gradiente, por elo

Notação: `σ` é o raio do kernel. `w` é o peso por segundo. "Derivada" é `d(recompensa)/d(variável)` no ponto indicado.

### 3.1 PEGAR (twist = 0; fecho = perto ≤ 0,10 m ∧ Δθ ≤ 25° ∧ z_pelve ≥ 0,75, sustentado 0,5 s)

| movimento | termo | forma | σ | w | estado |
|---|---|---|---|---|---|
| mãos → faces laterais | `staged`·alcancar, `precise_ori`·alcancar | `exp(−(d_palma/σ)²)`, d_palma = média das 2 palmas às suas faces | `d_palma₀` ≈ 0,34; piso 0,08 | 3,0 + 1,0 | OK. Derivada na abertura 8,7/m. A 2 cm, 24/m. |
| apertar | `squeeze` | `tanh(min(F_E, F_D)/6,13 N)` | contínuo | 1,0 | OK desde 0,1 N. |
| tirar da mesa | `unload` | `(1 − F_apoio/m·g) × preensao` | degrau em 2 mm (medido) | 2,0 | Degrau. O `trazer` cobre a subida. |
| caixa → peito, MACRO | `staged`·trazer | `exp(−(d/σ)²)` | `d₀` ≈ 0,45 | 3,0 | OK. Derivada 4,96/m em d = 0,385 (hoje). 2,5/m em d = 0,10. Paga 95% em d = 0,10. |
| caixa → peito, PRECISO | `precise_pos` | `exp(−(d/σ)²)` | **0,05 fixo** | 2,0 | **BURACO H1.** 1e−26 em 0,385; 1e−7 em 0,20; **0,018 em 0,10 (o limiar)**. |
| ficar de pé | `postura_ereta` | `rampa(z; 0,45 → 0,75) × unload` | linear | 2,0 | OK. 6,7/m × 0,79. Satura em 0,75 = limiar de `de_pe` (H9). |
| não girar | `precise_ori` | `alcancar × exp(−(Δθ/σ)²)`, face congelada na abertura | 0,20 rad | 1,0 | OK. Termo de manter. Δθ medido 8,2°. |
| ficar lá | `sustentacao` | `clamp(t/1 s)` dentro de tol | 0,10 m, 25° | 0,5 | **BUG H2**: morta em todo env que já avançou um elo. |

Medido na 1503, por env de manipulação: preensao 0,843; descarga 0,933; rampa 0,718 (pelve 0,665 m); trazer ≈ 0,48 (d/d₀ ≈ 0,86); alinha ≈ 0,60.

### 3.2 Transição PEGAR → CARREGAR (cadeias 2 e 3)

`_avanca_elo_force` chama `_aplica_elo` e depois `_recalcula_sigmas`. O σ do `trazer` vira `max(d_alvo; 0,08)`. Com `d_alvo ≤ 0,10`, o `trazer` volta para `e⁻¹ = 0,37` (ou 0,37 a 1,0 se d < 0,08). Antes do fecho ele valia 0,95.

| termo | antes do fecho | depois | Δ por segundo |
|---|---|---|---|
| `staged`·trazer | 0,9 × 1,95 × 3 = 5,27 | 1,0 × 1,37 × 3 = 4,1 | **−1,2** |
| `sustentacao` | 0,5 | 0 (reset) | **−0,5** |
| `pose` | 1,0 (neutro) | `variable_posture` com braços fora ≈ 0 | **−1,0** |
| `rastreio` | 0 | `exp(−v²/0,25)` com cmd 0,5–1,0 e robô parado | +0,04 a +1,5 |
| saldo | | | **≈ −1,5 a −2,5** |

**BURACO H3.** Fechar o PEGAR reduz a renda. Um env que fica em `d = 0,11` não fecha e mantém `trazer = 0,94`, `sustentacao = 0,5`, `pose = 1,0`.

O módulo já corrigiu o mesmo defeito para o `alcancar` no BOTAR (`_alcancar ≡ 1`, docstring). O `trazer` não recebeu a correção.

### 3.3 CARREGAR, cadeia 2 (andar 0,50 m; fecho = perto ∧ andou ≥ 0,50 m, 1,5 s)

| movimento | termo | forma | σ | w | estado |
|---|---|---|---|---|---|
| andar | `track_linear_velocity` | `exp(−‖Δv‖²/0,25)` | std 0,5 m/s | 2,0 | OK. Fraco em |e| ≥ 0,7 m/s (H10). |
| girar | `track_angular_velocity` | `exp(−Δw²/0,5)` | std 0,71 rad/s | 2,0 | OK. |
| deslocar 0,50 m | nenhum | — | — | — | **H8.** A condição lê deslocamento. Nenhum termo paga deslocamento. 10% dos envs têm cmd = 0 (`rel_standing_envs`) e não fecham até o resample (3 a 8 s). |
| caixa no peito | `trazer` (σ ≈ 0,09), `precise_pos`, `sustentacao` | | | | Igual ao PEGAR: H1 e H2. |
| postura | `pose` | `variable_posture` | σ 0,05 | 1,0 | **H6.** Canal morto: 0,000 com derivada zero (medição do próprio `PosturaPorElo`). CARREGAR está em `ELOS_QUE_ANDAM`. |
| apertar, descarga, pelve | `squeeze`, `unload`, `postura_ereta` | | | | OK, continuam. |

### 3.4 CARREGAR, cadeia 3 (segurar parado; twist = 0; fecho = perto, sustentado 0,5 a 1,5 s)

| item | estado |
|---|---|
| `rastreio` | Publicado = CARREGAR ∈ `ELOS_QUE_ANDAM`. Twist zerado. Paga **4,0/s por velocidade zero**. |
| `pose` | Ativo, standing σ 0,05, braços fora → 0. |
| manipulação | Igual ao PEGAR. |

**H7.** Este elo paga o piso da estátua de 8,3/s que o `rastreio_por_elo` existe para remover. É o estado mais confortável do ambiente, e ele antecede o BOTAR.

### 3.5 Transição CARREGAR → BOTAR

| termo | antes | depois | Δ por segundo |
|---|---|---|---|
| `squeeze` | 0,84 | 0 (`fora_do_botar`) | −0,84 |
| `unload` | 1,57 | 0 | −1,57 |
| `postura_ereta` | 1,13 | 0 (é `rampa × unload`) | −1,13 |
| `rastreio` (cadeia 3) | 4,0 | 0 | −4,0 |
| `pose` | ≈ 0 | 1,0 (neutro) | +1,0 |
| `staged`·trazer | 0,9 × 1,37 × 3 ≈ 3,7 | 1,0 × 1,37 × 3 = 4,1 | +0,4 |
| `load`, `largou` | 0 | 0 na abertura | 0 |
| saldo | | | **≈ −6,1** (cadeia 3), **≈ −2,1** (cadeia 2) |

**BURACO H4.** Um env em `d = 0,11` no CARREGAR de segurar parado não fecha e mantém a renda maior.

### 3.6 BOTAR (twist = 0; fecho = perto ∧ alinhado ∧ apoiada ≥ 0,5 m·g, 0,3 s)

| movimento | termo | forma | σ | w | estado |
|---|---|---|---|---|---|
| caixa → laje, MACRO | `staged`·trazer, `alcancar ≡ 1` | `exp(−(d/σ)²)` | `d₀` | 3,0 | OK. |
| caixa → laje, PRECISO | `precise_pos` | | 0,05 | 2,0 | **H1**, igual. |
| apoiar | `load` | `clamp(F_z/m·g) × [d ≤ 0,20]` | degrau em 2 mm; gate booleano 0,20 | 2,0 | Degrau. A descida é paga pelo `trazer`. Monotonia medida: pairar 10,7 < apoiada 13,0 < espera 17,9. |
| não girar | `precise_ori` = alinha | | 0,20 rad | 1,0 | OK. |

### 3.7 Espera final (publica ANDAR; VALIDA = 1; interno = BOTAR; twist = 0)

| movimento | termo | forma | σ | w | estado |
|---|---|---|---|---|---|
| largar | `largou` | `soltou × (1 − exp(−(d_palma/0,10)²)) × load` | 0,10 | 1,0 | OK. Derivada zero em d = 0 (início quadrático), 7,8/m em 5 cm, satura em 20 cm. |
| voltar à pose de referência | `pose` (publicado ANDAR → ativo) | `variable_posture`, standing σ 0,05 | | 1,0 | **BURACO H5.** Com os braços fora o termo vale 0 com derivada 0. Depois de 20 cm de `largou`, nada puxa os braços. |
| ficar parado | `rastreio` | 4,0/s por v = 0 | | | Platô. |

### 3.8 ANDAR (fecho: nenhum; o episódio corre até o time_out)

| movimento | termo | forma | σ | w | estado |
|---|---|---|---|---|---|
| rastrear v | `track_linear_velocity` | `exp(−‖Δv‖²/0,25)` | 0,5 m/s | 2,0 | OK. Derivada 0,29/(m/s) em e = 1,0; 2,9 em e = 0,5. **H10**: sem kernel macro. |
| rastrear wz | `track_angular_velocity` | `exp(−Δw²/0,5)` | 0,71 rad/s | 2,0 | OK. |
| de pé | `upright` | `exp(−tilt²/0,2)` | | 1,0 | OK. |
| pose | `variable_posture` | por regime | | 1,0 | OK. |
| passo | `air_time` | | | 0,0 | Desligado por decisão (F1). Clearance, swing_height, slip, soft_landing são freios gated por cmd > 0,05. Nenhum incentivo positivo de passo. |

Comandos: lin ±1,0 m/s, wz ±0,5 rad/s até 5000 iterações; 10% standing; 10% turning; resample 3 a 8 s.

### 3.9 Espera inicial (publica ANDAR; VALIDA = 0; twist = 0; 0,5 a 1,5 s)

Toda manipulação × 0. `rastreio` 4,0/s + `pose` ≈ 1,0 → ≈ 5,8/s. Na abertura do PEGAR a renda cai para ≈ 2,5/s. É degrau de timer, não de escolha. Não é buraco.

### 3.10 REORIENTAR inerte (0,3 s)

Alvo = a própria caixa. `trazer = 1`, `precise_pos = 1`. Ao avançar, os dois caem. É degrau de timer. Não é buraco.

## 4. Buracos, em ordem de efeito

| # | buraco | prova | elos | conserto (decisão do dono) |
|---|---|---|---|---|
| H1 | `precise_pos_sigma = 0,05` é a metade de `tol_pos = 0,10` | `exp(−4) = 0,018` no limiar; 1e−26 onde a caixa está | PEGAR, CARREGAR, BOTAR | σ ≥ tol_pos. Com 0,15: 0,64 no limiar. Com 0,30: 0,89 no limiar, 0,20 a 0,385 m. |
| H2 | `sustentacao` morta após o primeiro avanço | `comando.py:1007` escreve `avancou` só nos ids que avançam; `:681` só zera no reset. `recompensas.py:530` zera `t` enquanto `avancou` for True. | PEGAR da cadeia 1; CARREGAR; BOTAR | `self.avancou[:] = False` no início de `_avanca_elo`. Uma linha. |
| H3 | Fechar o PEGAR custa ≈ −2/s | σ do `trazer` recomputado no avanço; `sustentacao` reset; `pose` deixa de ser neutro | PEGAR → CARREGAR | Não recomputar `σ_trazer` quando `d_alvo ≤ tol_pos`; ou piso do σ = tol_pos. |
| H4 | Fechar o CARREGAR custa −2 a −6/s | `squeeze`, `unload`, `postura_ereta` → 0; `rastreio` desliga na cadeia 3 | CARREGAR → BOTAR | Renda do BOTAR na abertura ≥ renda do CARREGAR no fecho. |
| H5 | Nenhum gradiente para voltar à pose de referência | `pose` standing σ 0,05 = 0 com derivada 0 (medido em `PosturaPorElo`) | espera final | Um termo de pose de braços com σ largo, ativo em `soltou`. |
| H6 | `pose` é canal morto no CARREGAR | `ELOS_QUE_ANDAM = (ANDAR, CARREGAR)`; braços fora → 0 | CARREGAR | Tirar CARREGAR do gate do `PosturaPorElo`, ou σ próprio. |
| H7 | `rastreio` paga 4,0/s por ficar parado no segurar-parado | publicado = CARREGAR; twist zerado | CARREGAR cadeia 3 | Gate do `rastreio_por_elo` também por `_segura_parado`. |
| H8 | `andou ≥ 0,50 m` sem termo que pague deslocamento | tracking paga velocidade, não posição; 10% standing | CARREGAR cadeia 2 | Termo de deslocamento desde `_pos_no_elo`; ou excluir standing no CARREGAR. |
| H9 | `postura_ereta` satura em 0,75 = limiar de `de_pe` | rampa `clamp(0, 1)` em `pelve_alvo`; condição `≥ pelve_alvo` | PEGAR | `pelve_alvo` da rampa 3 a 5 cm acima do limiar do fecho. |
| H10 | ANDAR sem kernel macro | derivada 0,29/(m/s) em e = 1,0 | ANDAR | A marcha formou. Registro, sem ação. |

## 5. Contra a regra do macro e do preciso

| par | macro (σ, w) | preciso (σ, w) | veredito |
|---|---|---|---|
| caixa → alvo | 0,45, **3,0** | 0,05, **2,0** | Raio: certo. Peso: **invertido**. E o raio preciso não cobre o aceite de 0,10. |
| mãos → faces | 0,34, 4,0 | piso 0,08 do mesmo kernel + `squeeze` (força) | Sem par explícito. Sem buraco: o piso faz o papel do preciso. |
| velocidade | 0,5, 2,0 | — | Sem macro. |

## 6. O que NÃO é buraco, e eu verifiquei

- `alinhado`: Δθ medido 8,2° contra 25°. O termo é de manter, e ele mantém.
- `contato_tronco` é multa de contato com a MESA, não com a caixa. O alvo no peito não está em zona penalizada.
- `unload` lê força, e o contato é rígido. `descarga = 0,933` prova caixa fora da mesa.
- Espera inicial e REORIENTAR: degraus de timer, sem escolha da política.

## 7. Estado medido que a auditoria explica

Na 1503, por env de manipulação: `staged` 1,33 de 2,0; `precise_pos` 0,7%; `sustentacao` 0,06%; `sucesso` 0. A caixa está a 86% da distância inicial. O único termo com derivada ali é o `trazer` (4,96/m). Os 5,5 de peso restantes (`precise_pos`, `sustentacao`, `load`, `largou`) estão atrás de limiares que a caixa não vê. E `action_rate_l2` custa −2,16/s.
