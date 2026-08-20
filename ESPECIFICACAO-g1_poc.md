# Especificação — `g1_poc`

**Data:** 2026-08-18
**Responsável:** João Bornelli
**Robô:** Unitree G1, 29 DoF, base flutuante
**Base:** mjlab 1.5.1 · MuJoCo Warp · rsl_rl 5.4.0
**Estilo:** frases curtas, voz ativa, termos constantes

---

## 0. Objetivo

O robô deve executar sete comportamentos com uma caixa de 20 cm.

| # | comportamento | como a especificação o cobre |
|---|---|---|
| 1 | ficar de pé | forma de locomoção, `twist = 0` |
| 2 | ficar de pé com a caixa | elo `carregar`, `twist = 0` |
| 3 | andar | forma de locomoção, `twist` sorteado |
| 4 | andar carregando a caixa | elo `carregar`, `twist` sorteado |
| 5 | pegar em alturas e pesos, e erguer | elo `pegar` |
| 6 | pôr a caixa num alvo | elo `botar` |
| 7 | reorientar a caixa | elo `reorientar` |

Esta é uma POC. O objetivo é provar que os sete comportamentos são possíveis. O objetivo não
é acerto de 100%.

### Critério de aceite da POC

| # | critério |
|---|---|
| 1 | O robô ergue a caixa de 1 kg de uma prateleira a 0,55 m, fica de pé, e sustenta 1,0 s. Taxa ≥ 50%. |
| 2 | O robô repete o item 1 com a prateleira a 0,04 m e carga de 5 kg. Taxa ≥ 30%. |
| 3 | O robô anda 2 m com a caixa sem largá-la. Taxa ≥ 50%. |
| 4 | O robô põe a caixa na prateleira. Taxa ≥ 30%. |
| 5 | O robô gira a caixa 90° em torno do eixo vertical. Taxa ≥ 30%. |

### Fora do escopo

Duas transições precisam de um comando de **destino**, e não de um comando de velocidade:

- `andar` → `pegar`: o robô anda até a caixa.
- `carregar` → `botar`: o robô anda até a prateleira.

As duas são navegação. Elas exigem um comando novo, uma penalidade de colisão, e uma
prateleira observada. Elas ficam para um bloco posterior.

---

## 1. Princípios

1. **Uma tarefa.** Não existe one-hot de tarefa. Não existe orçamento equalizado. Não existe
   gate por tarefa.
2. **O alvo é um comando, e o robô o observa.** Sempre.
3. **O comportamento vem do par `(twist, alvo)`.** "Pegar", "botar" e "reorientar" são
   posições de alvo.
4. **Reuse o mjlab.** Escreva apenas o que não existe.
5. **Termine em vez de penalizar.** Uma trajetória inválida acaba. Ela não paga multa.
6. **Ataque a geometria antes da penalidade.** Esta é a lição de 16/07 do repositório.

---

## 2. Estrutura de arquivos

```
g1_poc/
├── __init__.py          registra a task no gym
├── knobs.py             todos os números, num dataclass
├── cena.py              robô, caixa, prateleira, sensores
├── env_cfg.py           monta o ManagerBasedRlEnvCfg
├── comando.py           CaixaAlvoCommand: o alvo, a cadeia, o sucesso
├── recompensas.py       reaching bimanual, staged, precise_pos, precise_ori
├── postura.py           o quarto regime do variable_posture
├── curriculo.py         o nível por env, e os dois cronogramas
├── terminacoes.py       caixa_largada, contato_ilegal, de_pe
├── observacoes.py       palmas_para_caixa (com clamp), caixa_para_alvo
├── smoke.py             valida o cfg na CPU
├── train.py
└── play.py
```

Estimativa: 400 a 500 linhas novas.

---

## 3. Cena

### 3.1 Robô

Reuse `g1_training/common/robot.py` sem mudança de física.

| item | valor | origem |
|---|---|---|
| entidade | `get_lift_box_robot_cfg()` | `common/robot.py:79` |
| pose inicial | `KNEES_BENT_KEYFRAME` | mjlab |
| pads de palma | 7 × 1,6 × 9 cm, `condim` 3, μ 1,0 | `common/robot.py:28` |
| pads de dorso | iguais, `condim` 1 | `common/robot.py:28` |
| prioridade dos pads de palma | 1 | necessário: sem ela o μ da caixa vence |

**Conserto de documentação.** O docstring de `common/robot.py:20-27` diz que a palma é a face
−Z local. O código desloca o pad em **Y**, e o código está certo. O mesh da mão mede
13,2 × 6,7 × 10,7 cm, portanto a face fina é Y. Com o deslocamento em Y as duas palmas ficam
viradas uma para a outra. Corrija o texto.

**Nota de física.** O `condim = 1` em `.*_collision` vale só contra outro `condim = 1`. Contra
a caixa, que é `condim = 3`, o MuJoCo usa `condim = max` e `friction = max`. Verificado: o
contato antebraço–caixa nasce com `dim 3` e μ = 1,0. Portanto o antebraço agarra.

### 3.2 Caixa

| item | valor |
|---|---|
| forma | cubo, meia-aresta 0,10 m |
| massa | 1,0 kg |
| `condim` / `friction` | 3 / (1,0 · 0,02 · 0,001) |
| junta | livre, 6 DoF |
| carga extra | força vertical `−(m − 1)·g`, por `write_external_wrench_to_sim` |

A carga usa força externa, e nunca `dr.body_mass`. O `dr.body_mass` corrompe a heap. Isto
está medido no repositório.

**Limitação declarada.** A caixa de 5 kg tem a inércia de 1 kg. A carga endurece a estática.
Ela não endurece a dinâmica.

### 3.3 Prateleira

| item | valor |
|---|---|
| forma | laje, 60 × 60 × 4 cm |
| tipo | mocap, sem junta livre |
| centro xy | (0,50 ; 0,00) |
| topo | sorteado por nível, de 0,04 m a 0,55 m |
| grupo de geom | 2 |

O grupo 2 é obrigatório. O `foot_height_scan` do fabricante lê o grupo 0. Sem o regrupamento
ele lê a prateleira como chão.

O piso de 0,04 m existe porque a laje tem 4 cm. Com o topo em 0,04 m a laje apoia no chão, e
não o atravessa.

### 3.4 Caixa e prateleira: posições

| ponto | valor |
|---|---|
| caixa no reset | (0,32 ; 0,00 ; topo + 0,10), com jitter |
| jitter x da caixa | 0,00 a +0,20 m |
| jitter y da caixa | −0,18 a +0,18 m |
| jitter de yaw da caixa | ±15° |
| jitter do topo da prateleira | ±0,02 m |

O x de 0,32 m é a borda perto do robô. O centro da prateleira, em 0,50 m, é inalcançável. Isto
está medido no repositório em 16/07.

### 3.5 Sensores

| sensor | primário | secundário | campos |
|---|---|---|---|
| `palma_E`, `palma_D` | `left/right_palm_pad` | `box_geom` | `found`, `force` |
| `dorso_E`, `dorso_D` | `*_hand_back_pad` | `box_geom` | `found` |
| `apoio_caixa` | `box_geom` | `table_geom` | `found`, `force` |
| `corpo_prateleira` | pelve, tronco, coxa | `table_geom` | `found`, `force` |
| `auto_colisao` | pelve | pelve | `found`, `force` |
| `pes_chao` | tornozelos | qualquer | `found`, `force`, `track_air_time` |

O `corpo_prateleira` **exclui** antebraço, mão, pad e pé. Uma pega baixa põe o antebraço perto
do tampo, e esse contato é normal.

### 3.6 Física

| item | valor | motivo |
|---|---|---|
| `timestep` | 0,005 s | mjlab |
| `decimation` | 4 | mjlab. Controle a 50 Hz. |
| `cone` | `pyramidal` | O `elliptic` divergiu para NaN neste repositório em 15/07. |
| `impratio` | 1,0 | idem |
| `njmax` / `nconmax` | 800 / 300 | do repositório |

**Experimento posterior.** A tarefa de manipulação do mjlab usa `elliptic` e `impratio = 10`.
Esse par modela melhor o cone de atrito de uma pega. Teste depois da POC, com uma terminação
`nonfinite` de rede — ela ainda **não existe** e precisa ser criada sobre o
`nonfinite_state` de `g1_training/common`. Ver §12.

---

## 4. Ação

| item | valor |
|---|---|
| termo | `JointPositionActionCfg` |
| dimensão | 29 |
| juntas | todas |
| escala | `G1_ACTION_SCALE × 0,8` |
| offset | pose padrão (`use_default_offset=True`) |
| clamp | nenhum |

---

## 5. Observação

### 5.1 Ator — 112 canais

| termo | dim | ruído | origem |
|---|---:|---|---|
| `base_ang_vel` | 3 | mjlab | `velocity` |
| `projected_gravity` | 3 | mjlab | `velocity` |
| `joint_pos` | 29 | `Unoise ±0,01` | `velocity` |
| `joint_vel` | 29 | `Unoise ±1,5` | `velocity` |
| `last_action` | 29 | — | `velocity` |
| `twist_cmd` | 3 | — | `velocity` |
| `palmas_para_caixa` | 6 | `Unoise ±0,01` | novo |
| `caixa_para_alvo` | 3 | `Unoise ±0,01` | `manipulation` |
| `face_alvo` | 3 | — | do repositório |
| `dir_alvo` | 3 | — | do repositório |
| **`caixa_valida`** | **1** | — | **novo** |

`enable_corruption = True`.

### 5.1.1 `caixa_valida` — o bit que diz se existe caixa

O canal vale 0 ou 1. Ele vem da fatia `[9:10]` do comando `caixa_alvo`.

| valor | significado | os quatro canais da caixa |
|---|---|---|
| 1 | existe tarefa de caixa | valores reais |
| 0 | não existe caixa | **zerados**: `palmas_para_caixa`, `caixa_para_alvo`, `face_alvo`, `dir_alvo` |

**Por que o bit existe.** No robô real o robô nem sempre vê a caixa. Ao ficar de pé e ao andar
não existe caixa nenhuma.

Sem o bit, "não existe caixa" e "a caixa está exatamente no alvo" produzem a mesma
observação: erro de alvo igual a zero. São duas situações opostas.

**O bit não tem clamp e não tem ruído.** Ele é um comando, e o supervisor o escreve.

**Aviso de contrato.** A largura da observação é um contrato com o checkpoint. Acrescentar um
canal depois invalida todos os checkpoints. O bit entra antes do primeiro bloco.

### 5.2 Crítico — 125 canais

O crítico recebe os 112 canais do ator, sem ruído, mais 13 canais privilegiados:

| termo | dim |
|---|---:|
| `base_lin_vel` | 3 |
| força normal das duas palmas | 2 |
| força de apoio da prateleira | 1 |
| velocidade linear da caixa | 3 |
| velocidade angular da caixa | 3 |
| topo da prateleira | 1 |

⚠ São **13**, e não 10: o `base_lin_vel` conta. Ele SAI do ator (num humanoide real não
é medido de forma confiável) e volta aqui, o que é privilégio legítimo — o crítico é
descartado no deploy. Uma versão anterior desta seção somava 122 por esquecê-lo.

`enable_corruption = False`.

---

## 6. Comandos

Existem dois comandos. Os dois estão presentes em todo episódio. Nenhum é zerado.

### 6.1 `twist` — 3 números

`UniformVelocityCommandCfg` do mjlab, com três acréscimos do repositório.

| parâmetro | valor |
|---|---|
| faixas | do cronograma, ver §10.2 |
| `heading_command` | `True` |
| `ranges.heading` | −π a +π |
| `heading_control_stiffness` | 0,5 |
| `rel_standing_envs` | 0,10 |
| `rel_heading_envs` | 0,30 |
| `resampling_time_range` | 3 a 8 s |
| giro no lugar | 0,50 dos envs parados recebem \|ωz\| ≥ 0,15 |

O `heading_command` e o `ranges.heading` andam juntos. Sem os dois, o `rel_heading_envs` fica
inerte em silêncio.

Nos elos de manipulação o `twist` é forçado a zero, **depois** do sorteio do mjlab.

### 6.2 `caixa_alvo` — 10 números

| fatia | conteúdo | frame |
|---|---|---|
| `[0:3]` | a posição do alvo | mundo |
| `[3:6]` | `face_alvo`: qual das 6 faces | caixa |
| `[6:9]` | `dir_alvo`: para onde a normal dessa face aponta | base |
| `[9:10]` | `caixa_valida`: 0 ou 1 | — |

O supervisor escreve **13 números**: 3 do `twist` e 10 do `caixa_alvo`.

O alvo **não** usa quatérnio. A caixa é um cubo, e o cubo tem simetria de 4 voltas no eixo
vertical. Um alvo em quatérnio pediria uma rotação que a caixa já satisfaz.

`Δθ` é o ângulo entre a normal da face, em mundo, e o `dir_alvo`, em mundo. A função
`erro_angulo_deg` do repositório já calcula `Δθ`. Reuse.

O termo `caixa_alvo` também guarda:

| campo | conteúdo |
|---|---|
| `cadeia` | a lista de elos deste episódio |
| `elo` | o índice do elo corrente |
| `sustenta` | o tempo em que a condição do elo vale |
| `episode_success` | 1 quando o último elo fecha. Travado. |

---

## 7. Elos e cadeias

### 7.1 Os quatro elos

Um elo é um par `(modo do twist, alvo da caixa)`.

| elo | `twist` | alvo da caixa | o elo fecha quando |
|---|---|---|---|
| `pegar` | 0 | x 0,20–0,30 · y ±0,05 · **z 0,78–0,85**, mundo | 4 condições, 1,0 s |
| `reorientar` | 0 | a posição atual da caixa · `dir_alvo` girado | 2 condições, 0,5 s |
| `carregar` | sorteado | base + (0,25 ; 0,00 ; 0,15), frame da **base** | passam 6 s |
| `botar` | 0 | topo novo da prateleira + 0,10 · x 0,30–0,40 · y ±0,12 | 2 condições, 0,5 s |

**As duas convenções de frame ficam.** O alvo de erguer é do mundo. O alvo de carregar é do
corpo. Esta distinção vem do ADR-0001 e ela está correta.

O alvo do `pegar` é uma altura absoluta. Agachar não o move. Portanto o robô tem de ficar de
pé para fechar o elo.

### 7.2 As condições de fecho

**`pegar` — 4 condições, durante 1,0 s:**

1. `‖caixa − alvo‖ < 0,05 m`
2. `Δθ < 20°`
3. a pelve está acima de 0,65 m
4. a inclinação do tronco é menor que 20°

As condições 3 e 4 são a função `de_pe` do repositório. Reuse.

**`reorientar` — 2 condições, durante 0,5 s:**

1. `‖caixa − alvo‖ < 0,05 m`
2. `Δθ < 20°`

**`botar` — 3 condições, durante 0,5 s:**

1. `‖caixa − alvo‖ < 0,05 m`
2. `Δθ < 20°`
3. **a prateleira carrega ao menos 80% do peso da caixa**: `F_apoio ≥ 0,8 · m · g`

A terceira condição existe porque "botar" significa **soltar**. Sem ela o robô fecha o elo com a
caixa ainda apertada nas mãos, apenas posicionada.

O sensor é o `apoio_caixa`, e ele já tem o campo `force`. É a grandeza inversa do `unload`.

**`carregar` — 1 condição:** passam 6 s. O elo fecha com sucesso se
`‖caixa − alvo‖ < 0,05 m` nesse instante.

### 7.3 A prateleira se move quando o `pegar` fecha

Uma regra, três destinos:

| cadeia | destino da prateleira |
|---|---|
| `pegar` | não se move |
| `reorientar` → `pegar` | não se move |
| `pegar` → `carregar` | **+5 m**. O chão fica livre. |
| `pegar` → `botar` | **um topo novo**, sorteado na faixa do nível |

O movimento é seguro por construção. No instante do movimento o robô está de pé, e o fundo da
caixa está a 0,72 m. O topo novo fica no máximo em 0,55 m. A folga é de 0,17 m.

Escreva a pose com `write_mocap_pose`. Chame `sim.forward()` depois da escrita.

**Isto não é um teleporte da caixa.** O treino move só a mobília.

### 7.4 As quatro cadeias

| cadeia | elos | prova o item |
|---|---|---|
| `pegar` | 1 | 5 |
| `reorientar` → `pegar` | 2 | 7 e 5 |
| `pegar` → `carregar` | 2 | 2 e 4 |
| `pegar` → `botar` | 2 | 6 |

Não existe cadeia de 3 elos. Uma cadeia de 3 elos exigiria navegação.

### 7.5 O episódio

| evento | efeito |
|---|---|
| um elo fecha, e há elo seguinte | o treino escreve o elo seguinte no comando |
| o último elo fecha | `episode_success = 1`, **travado**. O episódio **continua**. |
| passa o `time_out` | o episódio termina |
| uma terminação de falha dispara | o episódio termina |

**O episódio não termina no sucesso.** Terminar cedo descartaria a recompensa do tempo
restante, e isso **puniria** o sucesso.

O robô que fecha a cadeia aos 4 s coleta a taxa máxima nos 16 s restantes. O robô que encosta e
para coleta a taxa do estado B. A diferença é 31% do retorno do episódio, e é este número que
faz o robô querer terminar.

É o que o `lift_cube` do mjlab faz: ele trava `episode_success` e continua até o `time_out`.

O treino **não** faz reset entre elos. O robô, a caixa e as velocidades continuam.

---

## 8. Recompensas — 20 termos

São **13 da fundação + `self_collisions` + 6 de tarefa**. O `self_collisions` não existe
no `velocity_env_cfg` do mjlab: o `env_cfg` o CRIA, sobre o sensor `auto_colisao`.
Portanto 13 + 1 + 6 = 20, e o `smoke.py` confere o número.

Os pesos são por segundo. O mjlab divide o `dt` de volta.

⚠ **Para ler um `Episode_Reward/*` do log, o divisor é `max_episode_length_s`, e não o
número de passos** (`reward_manager.py:108`). O valor médio da função por passo é

```
média(func) = Episode_Reward × max_episode_length_s / (duração_real_s × peso)
```

Com o episódio em 20 s e a duração real em 13,6 s, o fator é 1,47. Errar isto dá um
fator 20 e inverte o diagnóstico: no bloco 1 o `squeeze` parecia estar em 0,047 quando
estava em **0,945**, ou seja saturado.

### 8.1 Fundação — 13 termos, pesos do `velocity` do G1

| termo | peso | parâmetro |
|---|---:|---|
| `track_linear_velocity` | +2,0 | `std` = √0,25 |
| `track_angular_velocity` | +2,0 | `std` = √0,50 |
| `upright` | +1,0 | `std` = √0,20, no `torso_link` |
| `pose` | +1,0 | 4 regimes, ver §9 |
| `foot_clearance` | −2,0 | alvo 0,10 m |
| `foot_swing_height` | −0,25 | alvo 0,10 m |
| `foot_slip` | −0,1 | — |
| `soft_landing` | −1e-5 | nos pés |
| `body_ang_vel` | −0,05 | no `torso_link` |
| `angular_momentum` | −0,02 | — |
| `action_rate_l2` | −0,10 | cronograma, ver §10.3 |
| `dof_pos_limits` | **−10,0** | é o valor que `manipulation` e `tracking` usam |
| `self_collisions` | −1,0 | limiar 10 N |

Os cinco termos de marcha se auto-gateiam pelo comando. Eles multiplicam por
`(‖v_cmd‖ + |ω_cmd| > 0,05)`. Com o `twist` em zero eles valem zero.

**Este auto-gate substitui toda a máquina de gates por tarefa.**

### 8.2 Tarefa — 6 termos

| termo | peso | forma | origem |
|---|---:|---|---|
| `staged` | +3,0 | `reaching × (1 + bringing)` | `manipulation` |
| `precise_pos` | +2,0 | `exp(−‖caixa − alvo‖² / 0,05²)` | `manipulation` |
| `precise_ori` | +1,0 | `reaching × exp(−Δθ² / 0,40²)` | novo |
| **`squeeze`** | **+1,0** | `tanh( min(F_n_esq , F_n_dir) / F_ref )` | **novo** |
| **`unload`** | **+2,0** | `clamp(1 − F_apoio/m·g) × preensão × não_caiu` | **§8.5, ligado 19/08** |
| `joint_vel_hinge` | −0,01 | `(\|v\| − 0,5)⁺²`, cronograma | `manipulation` |

O `unload` estava **em reserva** na §8.5, com o gatilho "se o `squeeze` subir e o
`precise_pos` não seguir". O gatilho disparou no bloco 1 e ele subiu para cá. Ver
§8.2.2.

**`reaching` é bimanual.** O `reaching` do mjlab mede um site só. Use o `reaching_kernel` do
repositório: ele mede as duas palmas contra as duas faces laterais, com
`lateral_offset = 0,10 m`.

```
reaching = exp( −média(‖palma_E − face_E‖² , ‖palma_D − face_D‖²) / 0,20² )
```

**`bringing` tem σ variável.** O σ do mjlab é fixo em 0,30 m. No nível 0 a caixa sobe 0,17 m,
e um σ de 0,30 m já está saturado.

```
bringing_std = max(0,10 ; ‖alvo − caixa‖ no começo do elo)
bringing = exp( −‖caixa − alvo‖² / bringing_std² )
```

Esta é a única função do mjlab que a especificação modifica.

**A estrutura multiplicativa é o anti-hack.** O `bringing` só paga através do `reaching`.
Levar a caixa ao alvo sem as mãos nela não paga.

**Os cinco termos de tarefa multiplicam por `caixa_valida`.**

```
staged      ×= caixa_valida
precise_pos ×= caixa_valida
precise_ori ×= caixa_valida
squeeze     ×= caixa_valida
unload      ×= caixa_valida
```

Só o `joint_vel_hinge` fica fora: ele é qualidade de movimento, e vale com caixa ou sem.

Isto é obrigatório. Com o bit em 0 os canais da caixa são zerados, e um vetor zerado dá
`exp(−0 / σ²) = 1`. Sem a multiplicação, "não existe caixa" pagaria o valor **máximo**.

O gate é sobre um valor de **comando**, e não sobre uma tarefa. É o mesmo idioma dos cinco
termos de marcha, que multiplicam por `(‖v_cmd‖ + |ω_cmd| > 0,05)`.

### 8.2.1 `squeeze` — o único termo com gradiente na coordenada do aperto

Este é o termo mais importante da especificação. Sem ele o treino repete a falha de hoje.

**O diagnóstico.** O `lift` de hoje paga +0,34/s por cada centímetro de subida, no nível 0. O
gradiente existe e é grande. O robô mesmo assim não subiu 1 cm.

O motivo: o gradiente está na coordenada errada.

| derivada da recompensa em relação a… | valor |
|---|---|
| a altura da caixa | grande |
| **a força de aperto** | **zero** |

A política só age em alvos de junta. Para a caixa subir, a força de atrito tem de vencer o
peso. Antes disso a caixa não se move, e nenhuma recompensa muda. É um degrau, e não uma
rampa.

O `reaching` não conserta isto. A palma não penetra a caixa. Portanto o `reaching` satura no
contato, e apertar mais forte não o move.

**A forma:**

```
F_n = componente NORMAL ao pad da força de contato palma ↔ caixa
squeeze = tanh( min(F_n_esquerda , F_n_direita) / F_ref )
F_ref   = m·g / (2·μ)     ≈ 6 N para 1 kg
```

A força de palma cresce de forma contínua com a penetração comandada. Portanto a derivada em
relação ao aperto é positiva de 0 N até `F_ref`. É exatamente o vão que hoje não paga nada.

O `min` das duas palmas exige aperto **simétrico**. Uma palma sozinha vale zero.

**O anti-hack.** Apertar a caixa **para baixo** contra a prateleira também gera força de palma.
O ADR-0001 já registrou este risco.

O conserto é a projeção na normal do pad. A palma aponta na horizontal, em ±Y local. Apertar
para baixo gera força **tangencial**, e não normal.

⚠ **A projeção não fechou o hack, e a versão anterior desta seção afirmava que fechava.**
Medido no bloco 1, iteração 1884: força de palma em **647% de `F_ref`** e apoio da
prateleira em **138% do peso** ao mesmo tempo. A projeção impede que o empurrão para
baixo seja **pago** como aperto — ela não impede que ele **aconteça de graça**. O robô faz
as duas coisas juntas: aperta na horizontal (pago) e prensa para baixo (grátis, e escora
os braços). Foi preciso um segundo termo, o `unload` da §8.2.2.

`squeeze` multiplica por `caixa_valida`.

### 8.2.2 `unload` — a ponte contínua, ligada em 19/08

O `squeeze` resolve o vão de 0 N a `F_ref` e **satura logo depois**. Com a força em 6,47×
`F_ref`, `tanh(6,47) = 0,99999` e a derivada é **1e-5**: ele deixa de guiar. O bloco 1
parou exatamente aí — 1884 iterações, `episode_success` **zero**, mãos nas faces, caixa
imóvel.

```
unload = clamp(1 − F_apoio/m·g , 0, 1) × preensão_bimanual × não_caiu × caixa_valida
```

**Por que a força de apoio, e não a altura.** Ela é a única grandeza da cena que responde
de forma contínua ao ato de erguer: cai de `m·g` a zero **antes** de a caixa se mover. A
altura é degrau; o apoio é rampa. O g1_multitask fechou o mesmo platô com ela — medido
lá: 9,70 N apoiada → 0,00 N erguida, fração 0,011 → 1,000, sem exploração extra, porque o
gradiente é denso e o robô já está com as mãos na caixa.

**É a coordenada, e não o tamanho do gradiente.** Medido no bloco 1, o gradiente de
`d(recompensa)/d(posição da caixa)` era **8,67 por metro** — grande. Mas
`d(posição da caixa)/d(ação)` era **zero**, porque a caixa estava prensada contra o tampo.
É o mesmo erro de coordenada que a §8.2.1 diagnosticou para o aperto, um nível adiante.

**Ele SOMA ao `squeeze`, e não o substitui.** O aperto já está resolvido; o que faltava era
pagar por descarregar. Peso 2,0, igual ao `precise_pos`, porque é sinal de tarefa.

**Os dois gates, e o que cada um fecha:**

| gate | sem ele |
|---|---|
| preensão bimanual (as duas palmas em contato) | DERRUBAR a caixa paga o máximo: sem tampo embaixo, `F_apoio = 0` e a fração vale 1 |
| `não_caiu` (`z > repouso − 3 cm`) | empurrá-la para fora do tampo paga igual a erguê-la |

⚠ O segundo gate é de **queda**, e não de subida. Exigir a caixa **acima** do repouso
recriaria o degrau que este termo existe para remover: o apoio cai enquanto a altura ainda
não mudou, e é nessa faixa que está o gradiente que falta. O repouso é `env.poc_topo` por
env — não uma constante —, porque o currículo alarga a faixa da prateleira no passo 4.

### 8.3 O gradiente de incentivo

Nível 0. Alvo em (0,25 ; 0,00 ; 0,82). A caixa em repouso está a 0,18 m dele. `twist` em zero.

| estado do robô | `staged` | `precise_pos` | `precise_ori` | `squeeze` | fundação | **total/s** |
|---|---:|---:|---:|---:|---:|---:|
| A · mãos longe | 0,19 | 0,00 | 0,05 | 0,00 | 6,0 | **6,2** |
| B · mãos na caixa, caixa parada | 4,06 | 0,00 | 0,99 | 1,00 | 6,0 | **12,1** |
| C · meio do percurso | 5,28 | 0,07 | 0,99 | 1,00 | 6,0 | **13,3** |
| D · caixa no alvo | 5,94 | 2,00 | 0,99 | 1,00 | 6,0 | **14,9** |

A recompensa cresce de A até D sem vale.

O retorno do episódio de 20 s:

| comportamento | retorno |
|---|---:|
| encosta e para (estado B por 18 s) | ≈ **229** |
| ergue em 4 s e segura (estado D por 16 s) | ≈ **300** |

**Erguer paga 31% mais no episódio.** Parar no estado B deixa 71 pontos na mesa.

Compare com hoje: o ADR-0001 mediu 1,07 de sinal de tarefa no estado B, de um orçamento de
4,00. E o `box_at_peito` foi **retirado** do `pegar`. Portanto nenhum termo pagava por a caixa
**estar** num lugar.

Três garantias, e elas são de aritmética:

1. A recompensa é monótona de A até D. Não existe vale.
2. A derivada em relação à força de aperto é positiva de 0 N a `F_ref`.
3. Chegar ao alvo e **ficar** lá é o estado que mais paga.

Uma coisa que a aritmética **não** garante: que a política encontre a solução. Isso é
exploração. A evidência de que é possível é a skill Lift, que ergueu 5 kg nesta mesma cena.

### 8.4 O que não existe nesta especificação

| termo do treino atual | por que sai |
|---|---|
| `grasp` | é booleano. O `squeeze` cobre a mesma faixa, de forma contínua. |
| `box_at_peito`, `box_at_prateleira` | são o `precise_pos`, com outro alvo |
| `box_shake`, `box_shake_pegar` | é o `precise_ori` |
| `back_penalty` | é o `reaching` bimanual |
| `com_balance` | é o `upright`, mais a terminação `fell_over` |
| `table_contact` | é a terminação `contato_ilegal` |
| `sucesso_denso` | é o `precise_pos`. E ele não pode usar a função da régua. |
| `terminacao` (−200) | a falha passa a ser fim de episódio |
| `hold_still` | é o regime `standing` da `pose`, ver §9 |
| `hip_deviation` | são os σ laterais apertados, ver §9 |
| a equalização de orçamento | existe uma tarefa só |
| `T.gated` e `TERMOS_DE_TAREFA` | o auto-gate pelo comando |

### 8.5 Em reserva

Estes termos não entram agora. Eles têm gatilho definido.

| termo | quando ligar |
|---|---|
| ~~`unload`~~ | **LIGADO em 19/08.** O gatilho era "se o `squeeze` subir e o `precise_pos` não seguir", e foi exatamente o que mediu no bloco 1: `squeeze` 0,945 (saturado) contra `precise_pos` 0,0002. Subiu para a §8.2.2. |
| `box_shake` | se o `precise_ori` não bastar para conter a rotação durante o transporte |
| `com_balance` | se o robô se inclinar para frente escorado na caixa, e o `upright` não pegar |
| separar o `bringing` do `staged` em termo próprio | se, com o `unload`, a caixa sair da prateleira e não chegar ao alvo. Hoje "chegar" e "trazer" dividem um peso único (3,0) e não há como pesá-los em separado. |

Regra: se aparecer um hack, volte **um** termo, e não seis.

O `unload` é a prova de que a reserva funciona: o gatilho estava escrito antes do treino,
a medição o disparou, e entrou **um** termo.

---

## 9. Postura — o quarto regime

### 9.1 O problema

O `variable_posture` do mjlab escolhe o regime pela velocidade comandada. Ele tem três
regimes: `standing`, `walking`, `running`.

O elo `pegar` roda com `twist = 0`. Portanto ele cai em `standing`. O `std_standing` do G1 é
`{".*": 0,05}`.

Um ombro deslocado 0,5 rad dá `0,25 / 0,0025 = 100`. Portanto o termo cobra do robô por
esticar o braço.

Uma prateleira a 0,04 m exige um agachamento acima de 1,5 rad no joelho. O `std_running` do G1
dá 0,6 rad ao joelho. Portanto o termo cobra do robô por agachar.

Este é o defeito que o repositório mediu em 17/07: *"posture 0,8 briga com o squat"*.

### 9.2 A solução

Acrescente um quarto regime. Ele lê a **demanda da caixa**, e não a velocidade.

```
demanda_caixa = 10 · ‖caixa − alvo‖ + 6 · Δθ            (Δθ em radianos)

demanda_caixa ≥ 1,5   →  std_manipulando
demanda_caixa < 1,5   →  os 3 regimes do mjlab, pela velocidade comandada
```

Os três dicionários do G1 ficam intocados. A marcha validada não muda.

Custo: cerca de 10 linhas, numa subclasse de `variable_posture`.

### 9.3 Os valores de `std_manipulando`

| junta | σ | motivo |
|---|---:|---|
| `knee` | 1,20 | o agachamento fundo |
| `hip_pitch` | 1,00 | o agachamento fundo |
| `ankle_pitch` | 0,50 | acompanha o agachamento |
| `hip_roll`, `hip_yaw` | 0,20 | o equilíbrio lateral fica apertado |
| `ankle_roll` | 0,15 | o equilíbrio lateral fica apertado |
| `waist_pitch` | 0,40 | o tronco inclina para alcançar |
| `waist_yaw` | 0,60 | o tronco gira para pôr a caixa ao lado |
| `waist_roll` | 0,15 | o equilíbrio lateral fica apertado |
| `shoulder_pitch`, `elbow` | 1,00 | o braço é a tarefa |
| `shoulder_roll` | 0,60 | o aperto lateral |
| `shoulder_yaw`, `wrist` | 0,40 | livres |

A regra é uma frase: **as juntas do plano sagital abrem, e as juntas laterais ficam
apertadas.**

O robô agacha e alcança. O robô não abre as pernas para os lados.

### 9.4 O gradiente que levanta o robô

A demanda cai a zero quando a caixa chega ao alvo.

| momento | demanda | regime | σ do joelho |
|---|---:|---|---:|
| a caixa está na prateleira baixa | alta | `std_manipulando` | 1,20 |
| a caixa chega ao alvo | ≈ 0 | `standing` | 0,05 |

O robô agacha para pegar. Depois a `pose` o puxa para a pose de pé.

Isto é um gradiente, e não um critério. O robô não precisa descobrir que deve levantar.

---

## 10. Currículo

Três partes. A separação é deliberada: o que a tarefa **pede** adapta por sucesso; o quão
**limpo** o movimento deve ser não adapta, porque apertar sempre baixa o sucesso.

### 10.1 Parte A — o nível, por env

Um inteiro por env, de 0 a 6. Ele seleciona uma célula.

| nível | topo da prateleira | carga | rotação | cadeias e frações |
|---|---|---|---|---|
| 0 | 0,55 | 1 kg | 0° | `pegar` 100% |
| 1 | 0,45 – 0,55 | 1 – 2 kg | 0° | `pegar` 100% |
| 2 | 0,30 – 0,55 | 1 – 3 kg | 0° | `pegar` 100% |
| 3 | 0,15 – 0,55 | 1 – 4 kg | 0 – 45° | `pegar` 50% · `reorientar`→`pegar` 50% |
| 4 | 0,04 – 0,55 | 1 – 5 kg | 0 – 90° | 40% · 25% · `pegar`→`carregar` 35% |
| 5 | 0,04 – 0,55 | 1 – 5 kg | 0 – 180° | 30% · 20% · 25% · `pegar`→`botar` 25% |
| 6 | 0,04 – 0,55 | 1 – 5 kg | 0 – 180°, eixo horizontal | iguais ao nível 5 |

Três regras da tabela:

- O **máximo** do topo continua 0,55 m em todos os níveis. O robô treina a altura que ele
  domina.
- O **mínimo** da carga continua 1 kg em todos os níveis. O mesmo motivo.
- O nível **acrescenta** cadeias. Ele não substitui cadeias.

Na cadeia `pegar` → `botar` o treino sorteia **duas** alturas, e elas usam faixas **diferentes**:

| altura | faixa | motivo |
|---|---|---|
| topo da pega | a faixa do nível, 0,04 – 0,55 m | pegar do chão e da prateleira |
| topo da colocação | **0,30 – 0,80 m** | uma mesa real tem 0,70 a 0,80 m |

A caixa a 0,80 m mais a meia-aresta de 0,10 m põe o alvo em 0,90 m. O peito do robô de pé está
em 0,91 m. Portanto o alcance existe.

Sem esta separação, o treino nunca veria a altura de uma mesa real.

**A regra de promoção:**

```
sobe  = episode_success
desce = ~episode_success
nivel = clamp(nivel + sobe − desce , 0 , 6)
```

Três linhas. Sem EMA, sem contador, sem limiar, sem grafo, sem evento de destravamento.

Duas propriedades:

1. **O nível equilibra onde a taxa de sucesso é ≈ 50%.** É um passeio aleatório ±1 com
   probabilidade de subir igual a `p(sucesso)`. O ponto fixo é `p = 0,5`.
2. **O rebaixamento é o anti-esquecimento.** Os envs se espalham pelos níveis. Sempre há envs
   nos casos fáceis. Não existe piso a escolher.

Só os episódios de manipulação movem o nível.

### 10.2 Parte B — a locomoção, por passo global

Use `mdp.commands_vel` do mjlab.

| passo | `lin_vel_x` | `lin_vel_y` | `ang_vel_z` |
|---|---|---|---|
| 0 | (−0,5 ; 1,0) | (−0,3 ; 0,3) | (−0,5 ; 0,5) |
| **8 000 × 24** | (−0,8 ; 1,5) | (−0,5 ; 0,5) | (−1,0 ; 1,0) |
| **12 000 × 24** | (−1,0 ; 2,0) | (−0,6 ; 0,6) | (−1,5 ; 1,5) |

⚠ **Os degraus estavam em 1000 e 2500, e a premissa que os justificava era falsa.**
Esta seção dizia "a locomoção do treino atual já funciona; o passo 0 pode começar nas
faixas de hoje" — o que pressupõe **warm-start de uma política que anda**. O bloco 1
rodou do ZERO (`resume = False`), portanto o robô nunca teve marcha alguma, e o
cronograma avançou dois estágios sozinho: na iteração 5099 ele recebia comando de
**2,0 m/s e 1,5 rad/s** com `peak_height_mean = 2,7 mm` — ou seja, arrastando os pés.

Cair em meio segundo com esse comando não é falha de aprendizado; é o comando ser
impossível. E isso se realimenta, porque **a fatia de dados de locomoção é governada
pelo tempo de vida do episódio, não pelo sorteio**:

| duração do episódio de andar | transições de locomoção |
|---|---:|
| 24 passos (0,5 s — medido) | **1,1%** |
| 100 passos | 4,3% |
| 400 passos | 15,1% |
| 961 (igual à manipulação) | **30,0%** |

O `frac_locomocao = 0,30` é por **EPISÓDIO** e o PPO aprende de **PASSO**. Com 7
episódios de manipulação de 961 passos contra 3 de locomoção de 24, andar recebe
`72 / 6.799` = 1,1% do gradiente — e os termos de marcha (`foot_clearance`,
`foot_swing_height`, `foot_slip`) são gateados por comando, logo valem **zero** nos
99% em que o twist está zerado. O sorteio só entrega os 30% prometidos quando as duas
formas têm tempo de vida parecido.

**Isto é o mesmo defeito de fase do `hinge`.** Dois dos três cronogramas por passo
global já saíram de fase, o que reforça a dívida registrada na §10.3: gatear por
COMPETÊNCIA. Aqui o gate natural é o `error_vel_xy` ou o `peak_height_mean` — só abrir
o teto seguinte quando o robô rastrear o teto atual.

⚠ Adiar o cronograma **não basta** para o robô aprender a andar: ele tira o comando
impossível do caminho, o que é pré-requisito, mas 1,1% de dados continua pouco. O
rompimento do laço exige um bloco com `frac_locomocao = 1,0`, e ali o risco é
esquecimento do `pegar` — mensurável em segundos com a `sonda.py`.

### 10.3 Parte C — a qualidade de movimento, por passo global

Use `mdp.reward_curriculum` do mjlab.

| passo | `joint_vel_hinge` | `action_rate_l2` |
|---|---|---|
| 0 | −0,01 | −0,10 |
| 1500 × 24 | −0,10 | −0,10 |
| 3000 × 24 | −0,10 | −0,25 |
| **10 000 × 24** | **−1,00** | −0,25 |

**Regra de leitura: não olhe para a pose antes de 1500 iterações.** Antes disso o freio solto é
deliberado.

⚠ **O degrau de −1,00 estava em 3000 × 24 e foi para 10 000 × 24 em 19/08, medido.** O aviso
abaixo previa o risco e ele se materializou — só que pior do que o previsto, porque o sucesso
não "caiu": ele **ainda não existia** quando o freio chegou.

Na iteração 3080, com o degrau ativo, o `joint_vel_hinge` mais o `action_rate_l2` passaram a
consumir **99,1% de todas as penalidades e 100% de todo o sinal positivo** — líquido −0,03,
`Mean reward` −0,37. E o `episode_success` tinha acabado de sair de zero, em 0,0060 — o
primeiro do projeto.

⚠ **Uma atribuição causal foi RETIRADA daqui.** Este texto afirmava que o freio provocava o
hack de escorar o tronco, porque o `contato_ilegal` subira de 6,4% para 17,5% das
terminações. **Não se sustenta:** outra run, com o **mesmo** freio de −1,00 e *menos*
avançada pós-degrau (34 iterações contra 80), mediu **3,4%**. A escora a 17,5% era
transitório de adaptação, e não consequência do freio. A conta do orçamento acima está
medida e vale; a causa do hack fica **não estabelecida**.

**Como ler `Mean reward`:** ele é a soma por episódio SEM a divisão por
`max_episode_length_s`, portanto `Mean reward = líquido × max_episode_length_s`. Com 20 s,
um líquido de −2,54 aparece como −50,3. Verificado nos dois blocos.

**A causa é de FASE, e não de valor.** O cronograma é por passo global e pressupõe a tarefa
resolvida em 3000 iterações; ela levou 3080 só para começar. A §17 põe "refino de pose" no
passo 6, o **último**, e o freio chegou cinco passos adiantado.

**Aviso, agora com precedente.** A carga chega a 5 kg no nível 4, e o `joint_vel_hinge` aperta
no passo 1500 × 24. Os dois podem coincidir. Se o sucesso cair nesse ponto, adie o aperto até o
nível médio da população passar de 4.

**A correção estrutural, ainda não feita:** gatear a pose por **competência** em vez de por
passo global — apertar só quando o `episode_success` passar de um limiar. Isso resolve a classe
do problema; adiar o degrau resolve esta instância. Enquanto o gate não existir, todo bloco que
passe de 10 000 iterações precisa reconferir a fase.

---

## 11. A forma do episódio

O treino sorteia a forma no reset de cada env.

| forma | fração | `twist` | prateleira e caixa | `caixa_valida` | canais da caixa |
|---|---|---|---|---|---|
| locomoção | 0,30 | sorteado | **+5 m** | **0** | **zerados** |
| manipulação | 0,70 | por elo | na cena | **1** | reais |

Na forma de locomoção o bit é 0. Portanto os quatro canais da caixa são zerados, e os três
termos de tarefa valem zero. O robô só rastreia velocidade.

O afastamento de 5 m continua, por dois motivos:

1. A prateleira sai da frente. O robô anda sem obstáculo.
2. O robô não consegue tocar a caixa. Ele não coleta `reaching` durante um episódio de
   locomoção.

**Este estado é idêntico ao estado de deploy sem caixa.** No robô real, ficar de pé e andar
usam o bit em 0 e os canais zerados. O treino e o deploy vêem a mesma coisa.

A locomoção é ensaiada em 30% dos envs, em todos os níveis. Ela nunca sai do treino.

### 11.1 A entrega do navegador

No robô real o navegador entrega um robô com velocidade residual e com erro de rumo. No treino
o elo `pegar` começaria com o robô paralisado.

Portanto a forma de manipulação sorteia no reset:

| grandeza | faixa |
|---|---|
| velocidade linear da base | ±0,25 m/s |
| velocidade angular da base | ±0,4 rad/s |
| erro de rumo em relação à caixa | ±20° |

Isto custa duas linhas no `reset_base`. Ele treina a entrega de bastão sem treinar navegação.

---

## 12. Terminações — 4 termos

| termo | condição | `time_out` |
|---|---|---|
| `time_out` | passam 20 s | **True** |
| `fell_over` | a inclinação do tronco passa de 70° | False |
| `caixa_largada` | a caixa está abaixo de 0,20 m, ou a caixa está a mais de 0,40 m das duas palmas | False |
| `contato_ilegal` | pelve, tronco ou coxa toca a prateleira com mais de 50 N | False |

São 4 e não 5. A fundação do `velocity` traz **três** terminações — `time_out`,
`fell_over` e `out_of_terrain_bounds` (`velocity_env_cfg.py:377`) —, o `env_cfg` remove o
`out_of_terrain_bounds` (o terreno é plano e a mobília tem pose absoluta) e acrescenta as
duas nossas.

⚠ **Não existe terminação `nonfinite` no mjlab**, e uma versão anterior desta tabela a
listava. O que existe é a função `nonfinite_state` em `g1_training/common`, disponível e
**não usada aqui** — ver o risco 6 da §19, que a menciona como rede para o teste de
`cone = elliptic`.

O `time_out = True` é obrigatório. Sem ele o rsl_rl trata o fim do tempo como fracasso.

O `caixa_largada` só vale depois de o elo `pegar` fechar.

---

## 13. PPO

Use `mjlab/tasks/velocity/config/g1/rl_cfg.py` sem mudança.

| campo | valor |
|---|---|
| `num_steps_per_env` | 24 |
| `gamma` / `lam` | 0,99 / 0,95 |
| `learning_rate` | 1,0e-3, `schedule = "adaptive"`, `desired_kl = 0,01` |
| `clip_param` | 0,2 |
| `entropy_coef` | 0,01 |
| épocas / minibatches | 5 / 4 |
| ator e crítico | MLP (512, 256, 128), ELU |
| distribuição | Gaussiana, `init_std = 1,0`, `std_type = "scalar"` |
| normalização de obs | `EmpiricalNormalization`, nos dois grupos |

Três notas:

1. `num_envs` não tem default útil. O default do mjlab é **1**. Passe o valor no CLI.
2. O `learning_rate` de warm-start é **5e-4**. Ponha esse valor em `knobs.py`. O ADR-0001
   declarou esta mitigação, e ela nunca existiu no código.
3. Não congele canais de comando no normalizador. Todos os canais de comando desta
   especificação são grandezas contínuas com faixa útil, e o normalizador empírico as trata
   bem.

---

## 14. Reuso

| origem | o que |
|---|---|
| `mjlab/tasks/velocity` | observação proprioceptiva, ação, `UniformVelocityCommand`, `track_linear_velocity`, `track_angular_velocity`, `upright`, `variable_posture`, `feet_clearance`, `feet_swing_height`, `feet_slip`, `soft_landing`, `body_angular_velocity_penalty`, `angular_momentum_penalty`, `self_collision_cost`, `action_rate_l2`, `joint_pos_limits`, DR de startup, `push_robot`, `commands_vel`, `terrain_levels_vel` (como molde) |
| `mjlab/tasks/manipulation` | `LiftingCommand` (o alvo e o `episode_success` travado), `staged_position_reward`, `bring_object_reward`, `joint_velocity_hinge_penalty`, `illegal_contact` |
| `mjlab/envs/mdp` | `reward_curriculum`, `termination_curriculum`, `write_external_wrench_to_sim`, `geom_friction` |
| `g1_training/common` | `robot.py` (os pads), `box.py` (a caixa e a prateleira mocap), `reaching_kernel` bimanual com `lateral_offset`, `apply_box_payload`, `nonfinite_state`, o fix broadcast-safe de `reset_joints_by_offset` |
| `g1_multitask` | `erro_angulo_deg`, `FACE_AXES`, `de_pe`, o portão de assinatura de checkpoint do `runner.py` |
| novo | `CaixaAlvoCommand` (o alvo, o bit `caixa_valida` e a cadeia), o quarto regime da `postura`, o nível por env, o σ variável do `bringing` |

---

## 15. O que sai do repositório

| item | linhas | motivo |
|---|---:|---|
| `g1_residual/` | ~3 000 | Desligado por dentro: `ESCALA_C = 0` e `dim_c = 0`. A premissa foi anulada por medição. Arquive. |
| `g1_multitask/entre_blocos.py` | 405 | Reporta tarefas que não existem. O painel de gargalo filtra fora o eixo do `pegar`. |
| `g1_multitask/calibra.py` | 336 | Reporta um número obsoleto e o esconde como "decidido". |
| `g1_multitask/curriculum.py` | 543 | Substituído por ~50 linhas. |
| `g1_multitask/tasks.py` | 243 | Não existem tarefas. |
| `g1_multitask/knobs.py` | 835 | Vai para ~250. |
| `g1_training/skills/place/` | stub | Nada implementado. |
| `reset_segurando` | — | As cadeias alcançam `carregar` e `botar` através do `pegar`. O teleporte deixa de ser necessário. |

Números que hoje vivem no meio do código e devem entrar em `knobs.py`:
`min_amostras_evento`, `STD_AO_ABRIR_TAREFA`, `JANELA_LIVRE_S`, `_RESET_BASE_POSE_RANGE`,
`_TOPO_RAMPA_Z`, a margem do `unload`, a escala do `target_pos_b`, e os hiperparâmetros de PPO.

Um treino deve ser reproduzível por `git diff` de um arquivo de config.

---

## 16. `smoke.py` — o que ele verifica na CPU

1. A task registra, e o nome não colide.
2. O cfg instancia, e 5 passos rodam com 8 envs.
3. `obs[actor]` tem 112 canais. `obs[critic]` tem 125.
4. Cada coluna de `_step_reward` é finita. Ele nomeia a coluna que falhar.
5. O comando `caixa_alvo` tem 10 números, e as quatro fatias cobrem tudo sem sobreposição.
6. Existem 20 termos de recompensa (13 + `self_collisions` + 6) e 4 terminações, e os 6
   termos de tarefa existem por nome.
7. Os cinco termos de marcha valem zero quando o `twist` é zero.
8. Com `caixa_valida = 0`: os quatro canais da caixa são zero, **e** `staged`, `precise_pos`,
   `precise_ori`, `squeeze` e `unload` são zero. Este teste é o mais importante da lista.
   ⚠ Ele é feito chamando a FUNÇÃO com os params do manager. Dois motivos, e os dois já
   deram falso resultado: o `observation_manager.compute()` devolve o `_obs_buffer`
   CACHEADO do passo anterior (`observation_manager.py:311`), e o ruído `Unoise` entra
   DEPOIS da função. Os `SceneEntityCfg` do cfg também não servem — o manager faz
   `deepcopy` e resolve a cópia, então no cfg `site_ids` continua `slice(None)`.
8b. Derrubar a caixa **não** paga `unload`. É o gate que impede o caminho mais curto:
   sem tampo embaixo, `F_apoio = 0` e a fração valeria 1.
9. A prateleira está no grupo de geom 2.
10. As quatro cadeias montam, e cada elo escreve um alvo diferente.
11. A transição de elo não faz reset: a pose do robô e a pose da caixa continuam.
12. O nível sobe com `episode_success = 1` e desce com 0.
13. O quarto regime da postura ativa quando `demanda_caixa ≥ 1,5`.
14. A prateleira se move quando o `pegar` fecha, e a folga vertical é positiva.
15. O `squeeze` cresce quando a força normal de palma cresce, e vale zero com uma palma só.
16. O `squeeze` **não** cresce quando a caixa é apertada para baixo contra a prateleira.
    ⚠ Isto é sobre o que o termo PAGA. Medido no bloco 1: o robô prensa a caixa de todo
    modo, porque prensar é grátis e escora os braços. Quem cobra por isso é o `unload`,
    e a medição de campo é a `sonda.py` — o smoke não pega comportamento.
17. O episódio **não** termina quando o último elo fecha. O `episode_success` fica travado em 1.
18. O elo `botar` não fecha enquanto a prateleira carregar menos de 80% do peso.

---

## 17. Ordem de execução

| passo | o que roda | portão de aceite |
|---|---|---|
| 0 | verificação visual no `play`, sem treinar | ver §18 |
| 1 | nível 0 · `pegar` · prateleira 0,55 · 1 kg | `Contrib/squeeze` sai de zero **antes** de `Contrib/precise_pos`. Depois a taxa de sucesso passa de 0,50, em ≤ 1 000 iterações. |
| 2 | níveis 1 e 2 · a prateleira desce até 0,30 · carga até 3 kg | o nível médio da população passa de 2 |
| 3 | nível 3 · `reorientar` entra | o nível médio passa de 3 |
| 4 | nível 4 · a prateleira desce até 0,04 · carga até 5 kg · `carregar` entra | o nível médio passa de 4 |
| 5 | níveis 5 e 6 · `botar` entra | os cinco critérios da §0 fecham |
| 6 | refino de pose | só aqui |

Duas regras de disciplina, e o repositório já pagou por elas:

1. **Uma mudança por bloco.**
2. **Warm-start sempre com `learning_rate = 5e-4`.**

---

## 18. Verificação antes do primeiro bloco

Ela roda na CPU, no `play`, e custa minutos. A lição de 16/07 do repositório é
*"ataque a geometria, e não o sintoma"*.

1. Ponha o topo da prateleira em 0,04 m. Confirme que a laje apoia no chão, sem atravessar.
2. Ponha a caixa em (0,32 ; 0,00 ; 0,14). Confirme que o tampo não cobre a caixa.
3. Mande o robô agachar até a caixa, com uma pose escrita à mão. Confirme que a pelve, o
   tronco e a coxa não tocam o tampo.
4. Ponha as duas palmas nas faces laterais da caixa. Confirme que os pads tocam as faces, e
   não as quinas.
5. Ponha a caixa em (0,25 ; 0,00 ; 0,82) com o robô de pé. Mova o topo da prateleira para
   0,55 m. Confirme que a laje não toca a caixa nem os antebraços.
6. Ponha o alvo do `botar` em (0,32 ; 0,12 ; topo + 0,10). Confirme que o robô alcança esse
   ponto sem girar a base mais de 20°.

Se o passo 3 falhar, suba o piso do topo da prateleira de 0,04 m para 0,15 m. Ou mova o centro
dela de x = 0,50 m para x = 0,60 m.

---

## 19. Riscos declarados

| # | risco | mitigação |
|---|---|---|
| 1 | O nível 6 gira a caixa no eixo horizontal. O robô não tem mãos. Tombar a caixa pode exigir mãos. | Aceite o nível 6 como opcional. Os critérios da §0 não o exigem. |
| 2 | A carga de 5 kg com a inércia de 1 kg. A DR endurece a estática, e não a dinâmica. | Declarado. A skill Lift ergueu 5 kg com o mesmo mecanismo. |
| 3 | A prateleira a 0,04 m fica à frente do tronco no agachamento. O `contato_ilegal` pode travar o nível 4. | Suba o piso para 0,15 m, ou mova o centro para x = 0,60 m. |
| 4 | O pulso é mole: 1,68 N·m/rad, limite de 5 N·m. A palma inclina sob carga, e o contato plano vira linha. | Meça `Metrics/` da força de palma. Se a pega falhar por inclinação, aumente a espessura do pad. |
| 5 | ~~A troca do gate `_grasp` booleano por `squeeze` contínuo mais forma multiplicativa não está validada para um humanoide bimanual.~~ **MATERIALIZOU-SE no bloco 1.** O `squeeze` saturou em 647% de `F_ref` (derivada 1e-5) e o robô prensou a caixa contra o tampo: apoio em 138% do peso, sucesso zero em 1884 iterações. | A mitigação era esta e funcionou como escrita: entrou **um** termo da §8.5, o `unload` (§8.2.2). O que resta na reserva continua valendo, e a régua de campo é a `sonda.py`. |
| 6 | O `cone = pyramidal` modela pior o cone de atrito de uma pega. | Teste `elliptic` com `impratio = 10` depois da POC, com uma terminação `nonfinite` de rede — a criar, ela não existe (§12). |
| 7 | A pose da caixa é verdade absoluta do simulador, com ruído de ±0,01 m. Falta latência, falta viés, e falta perda de rastreio. | Fora do escopo da POC. Entra no bloco de sim-to-real. O bit `caixa_valida` já cria o canal para tratar a perda de rastreio. |
| 8 | O bit `caixa_valida` só é sorteado em 0 na forma de locomoção. O treino nunca o vê mudar **dentro** de um episódio. | No robô real ele muda entre os passos 1 e 2. Sorteie a troca dentro do episódio no bloco de sim-to-real. |
| 9 | A altura de colocação vai a 0,80 m, e a de pega vai a 0,04 m. As duas pontas ficam nos extremos do alcance. | Se o `botar` a 0,80 m não fechar, baixe o teto para 0,70 m. Uma bancada tem 0,90 m e fica fora do alcance de qualquer forma. |

---

## 20. Relação com o ADR-0001

O ADR-0001 continua válido como histórico. Esta especificação mantém quatro decisões dele e
reverte duas.

**Mantém:**

- O alvo de erguer é uma altitude do **mundo**. Agachar não o move.
- O alvo de carregar é do **corpo**.
- O sucesso é medido por fato físico, e não por soma de recompensa.
- O episódio de manipulação é curto.

**Reverte:**

- O alvo do `pegar` volta à observação. O ADR-0001 o manteve fora, e escreveu a rampa em
  fração para compensar. A rampa sai. O alvo é uma altura absoluta, sorteada em
  0,78 – 0,85 m, e o robô a vê.
- O custo de braço volta a ≈ −0,25. Os 82% medidos pelo ADR-0001 eram sintoma de o robô não
  coletar `lift` nem `sustain`, e não causa.

---

## 21. Deploy no robô real

Esta seção não faz parte da POC. Ela existe porque o desenho da observação é um contrato, e o
contrato tem de servir ao deploy desde o começo.

### 21.1 A interface

A política é uma função:

```
ação(29) = π( propriocepção , twist(3) , caixa_alvo(10) )
```

Ela não tem estado de tarefa. Ela não tem fase. Ela não sabe o que é uma cadeia.

O supervisor escreve **13 números** a 50 Hz. A cadeia mora no supervisor.

### 21.2 A tabela completa dos comportamentos

| comportamento | `twist` | alvo, posição | `face_alvo` e `dir_alvo` | `caixa_valida` |
|---|---|---|---|---|
| ficar de pé | 0 · 0 · 0 | zerado | zerados | **0** |
| andar | vx · vy · ωz | zerado | zerados | **0** |
| girar no lugar | 0 · 0 · ωz | zerado | zerados | **0** |
| pegar | 0 · 0 · 0 | 0,25 · 0,00 · 0,82 do chão | os medidos | 1 |
| ficar de pé com a caixa | 0 · 0 · 0 | base + (0,25 ; 0 ; 0,15) | os medidos | 1 |
| andar com a caixa | vx · vy · ωz | base + (0,25 ; 0 ; 0,15) | os medidos | 1 |
| botar | 0 · 0 · 0 | topo da mesa + 0,10 | os medidos | 1 |
| reorientar | 0 · 0 · 0 | a posição medida da caixa | a face e a direção desejadas | 1 |

"Os medidos" significa: copie a orientação medida da caixa. O robô entende isso como **não gire
a caixa**.

### 21.3 A sequência `andar` → `pegar` → `andar` → `botar`

| passo | `caixa_valida` | posição da caixa vem de | `twist` | alvo | fim do passo |
|---|---|---|---|---|---|
| 1 · andar até a caixa | 0 | nada. Zerada. | do navegador | zerado | o navegador: a caixa está a 0,35 m à frente |
| 2 · pegar | 1 | **a câmera** | 0 | 0,25 · 0 · 0,82 | a caixa está a menos de 0,05 m do alvo, o robô está de pé, por 1,0 s |
| 3 · andar com a caixa | 1 | **as mãos** | do navegador | no peito | o navegador: a mesa está ao alcance |
| 4 · botar | 1 | **as mãos** | 0 | topo da mesa + 0,10 | a caixa está a menos de 0,05 m do alvo |

O passo 1 usa o bit em 0. O robô ignora a caixa por completo, e o estado é idêntico ao da forma
de locomoção do treino.

### 21.4 De onde vem a pose da caixa

Três casos, e o terceiro é o mais importante:

| caso | fonte |
|---|---|
| não existe caixa | nenhuma. Zere os canais e ponha o bit em 0. |
| a caixa está à vista, e o robô ainda não a pegou | a câmera |
| a caixa está nas mãos, e a câmera não a vê | **as mãos.** A caixa está no meio das duas palmas, com o deslocamento conhecido do pad. |

Depois de pegar, a câmera não é necessária. A pose sai da cinemática direta.

Se a câmera perder a caixa antes da pega, mantenha a última leitura e atualize com o movimento
do próprio robô.

### 21.5 O que a percepção precisa entregar

A política nunca vê posição global. Ela vê duas diferenças:

```
palmas_para_caixa  =  posição das palmas  −  posição da caixa
caixa_para_alvo    =  posição da caixa    −  o alvo
```

As duas são relativas ao robô. Não é necessário mapa nem localização global.

**Uma exceção.** O alvo do `pegar` tem altura absoluta, 0,82 m. Portanto a câmera precisa
informar **a que altura a caixa está do chão**. É o único número que não é puramente relativo.

### 21.6 Combinações que não funcionam

Os 13 números são independentes. Algumas combinações não têm treino por trás:

| combinação | resultado |
|---|---|
| velocidade alta **e** alvo com 5 cm de precisão | não funciona |
| andar **e** pegar do chão ao mesmo tempo | não foi treinado |
| `caixa_valida = 1` com a caixa fora de alcance | não foi treinado |

Regra: combine velocidade alta com o bit em 0, ou com o alvo no peito. Combine alvo preciso com
velocidade zero.

### 21.7 O que exportar

| item | vai para o robô |
|---|---|
| a MLP do ator (512, 256, 128) | **sim** |
| a média e a variância do normalizador de observação | **sim** |
| a ordem exata dos 112 termos de observação | **sim.** Ela é um contrato. |
| as escalas de cada termo | **sim** |
| a MLP do crítico e os 10 canais privilegiados | não. Descarte. |
| o nível, o `episode_success`, o currículo | não. São do treino. |

O normalizador é a armadilha silenciosa. O repositório já registrou este defeito no BFM: sem o
normalizador o ator recebe números crus e devolve lixo, e não levanta exceção.

Escreva um teste que compara a saída da política no simulador com a saída no alvo, para a mesma
observação. A diferença deve ser zero.
