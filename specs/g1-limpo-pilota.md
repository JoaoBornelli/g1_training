# g1_limpo — `pilota`: dirigir um checkpoint no MuJoCo clássico (v1)

Ferramenta de análise. **Não toca em nada do treino.**

## §0 A MEDIÇÃO que justifica

Mesma cena, mesma máquina (Core 7 250U, sem GPU), 2026-09-10:

| motor | ms por subpasso | vs tempo real |
|---|---|---|
| `mujoco==3.10.0` clássico | **0,065** | **77×** |
| MuJoCo Warp em CPU (mjlab) | 6,229 | 0,80× |
| | | **96× mais lento** |

O `play.py` inteiro roda a **0,35×** do tempo real (17,3 passos/s, medido headless, sem
viewer). A física é 49% disso; os 28 termos de recompensa são 5%. Não há ajuste que
conserte: o Warp compila kernels para GPU e o caminho de CPU é emergência.

Orçamento do `pilota`: física 0,26 ms + MLP ~0,2 ms por passo de controle de 20 ms =
**~40× o tempo real**.

## §1 O QUE É, E O QUE NÃO É

**É:** carregar um `model_*.pt` escolhido à mão, pôr o robô numa cena idêntica à do
treino, e dirigir com o teclado o `twist` (3 canais) e o `elo` (one-hot de 5).

**NÃO é:** máquina de estados. Sem espera, sem `sustain`, sem fecho de elo, sem
cadeia, sem currículo, sem recompensa, sem terminação. **Quem troca o elo é o
teclado.** Um alvo é gerado por elo, e mais nada.

**NÃO altera** nenhum arquivo existente de `g1_limpo`. Só acrescenta dois arquivos.

## §2 OS DOIS ARQUIVOS

### 2.1 `g1_limpo/exporta_cena.py` — roda uma vez, precisa do mjlab

```
python -m g1_limpo.exporta_cena --saida ~/g1_pilota/
```

Constrói o env com `make_env_cfg(play=True)`, `num_envs=1`, `device="cpu"`, e grava:

| arquivo | conteúdo |
|---|---|
| `cena.mjb` | `mujoco.mj_saveModel(env.sim.mj_model, ...)` — ~71 MB |
| `cena.npz` | tudo abaixo |

No `.npz`, lido do env e **não digitado à mão**:

- `q_default` — `robot.data.default_joint_pos[0]`, 29 valores
- `escala_acao` — `action_manager._terms["joint_pos"]._scale[0]`, 29 valores
- `nomes_juntas` — `robot.joint_names`, na ordem do modelo
- `ids_atuador` — o mapa junta → `ctrl`, para não supor que é identidade
- `meia_aresta` — `env.limpo_meia_aresta[0]`
- `peito_b`, `prateleira_xy`, `altura_carregar` — dos knobs
- `id_caixa`, `id_laje` — os índices de corpo, por `mj_name2id`

⚠ O `.mjb` **NÃO vai para o git** (71 MB). Acrescente `*.mjb` ao `.gitignore` se ainda
não estiver. O gerador é que é versionado.

### 2.2 `pilota.py`, na raiz do repo — não importa `g1_limpo`

```
python pilota.py --cena ~/g1_pilota/ --checkpoint ~/Downloads/model_9499.pt
```

Depende só de `mujoco`, `torch`, `numpy`. Carrega o `.mjb`, o `.npz` e o `.pt`, abre o
`mujoco.viewer.launch_passive`, e roda o laço.

⚠ Ele fica na RAIZ, e não em `g1_limpo/`: a regra do módulo é que `g1_limpo` não
importe código do projeto, e aqui o sentido é o inverso — o `pilota` não importa
`g1_limpo`. Mantê-lo fora deixa isso óbvio e permite rodá-lo num venv magro.

## §3 A OBSERVAÇÃO — 114 canais, nesta ordem

Da `observation_manager.active_terms["actor"]`, conferido:

| # | bloco | dim | como calcular |
|---|---|---|---|
| 1 | `base_lin_vel` | 3 | velocidade linear da base, **no frame da base** |
| 2 | `base_ang_vel` | 3 | velocidade angular da base, no frame da base |
| 3 | `projected_gravity` | 3 | gravidade no frame da base |
| 4 | `joint_pos` | 29 | `q − q_default` (é `joint_pos_rel`) |
| 5 | `joint_vel` | 29 | `qd − qd_default`; o default é zero |
| 6 | `actions` | 29 | a ação do passo ANTERIOR, crua |
| 7 | `command` | 3 | o `twist`: `vx, vy, wz` — **do teclado** |
| 8 | `elo` | 5 | one-hot do elo — **do teclado** |
| 9 | `caixa` | 10 | ver §4 |

⚠ Escalas e ruído: os termos do molde podem ter `scale`. **Leia do cfg no exportador**
e grave no `.npz`; não suponha 1,0. Ruído de observação é de treino e não entra.

⚠ Depois de concatenar, aplique o **normalizador do checkpoint**:
`obs = (obs − mean) / sqrt(var + eps)`, com `mean` e `var` de
`sd["actor_state_dict"]["obs_normalizer._mean"]` e `._var`. Sem isso a rede recebe
entrada fora de escala e o robô cai no primeiro passo.

### Os 10 canais da caixa (`observacoes.py:119-141`)

```
caixa_b = R⁻¹ · (pos_caixa_w − pos_base_w)     [3]
alvo_b  = R⁻¹ · (alvo_w      − pos_base_w)     [3]
giro_b  = R⁻¹ · giro_w                          [3]
meia_aresta                                      [1]
tudo × (elo_publicado != ANDAR)
```

`R` é a rotação da base. Com o elo em `ANDAR` os dez canais são **zero**, e é o gate do
treino — reproduza-o, não o contorne.

`giro_w` = 0 nesta versão (nenhum giro pedido).

## §4 O GERADOR DE ALVO — vinte linhas, sem estado

| elo | `alvo_w` |
|---|---|
| `ANDAR` (0) | irrelevante, os canais zeram |
| `REORIENTAR` (1) | a posição da caixa |
| `PEGAR` (2) | `pos_base_w + R · peito_b`, `peito_b = (0,25, 0, 0,222)` |
| `CARREGAR` (3) | igual ao `PEGAR` — âncora no peito |
| `BOTAR` (4) | ponto sobre a laje: `xy` de `prateleira_xy` relativo à origem, `z` = topo da laje lido do modelo |

Nenhum temporizador, nenhuma condição de fecho. O elo muda quando a tecla é apertada.

## §5 A PARIDADE — é o portão, roda ANTES de o viewer abrir

O risco único do lote é errar a convenção de um canal: o robô cai no primeiro passo e
não diz por quê.

`g1_limpo/exporta_cena.py --paridade` compara, no MESMO estado, os 114 canais que o
`pilota` monta contra os que o `observation_manager` produz.

Procedimento: construa o env do mjlab, `reset()`, avance 5 passos com ação zero, leia
`observation_manager.compute()["actor"][0]`. Do mesmo `mjdata`, monte a observação com
a rotina do `pilota` (importe-a, não a duplique) e compare.

| resultado | ação |
|---|---|
| todos os 114 com `\|Δ\| < 1e-5` | passa |
| qualquer canal fora | **relate qual bloco e o Δ**, e PARE |

Sem paridade verde, não commite o `pilota`.

⚠ Faça a rotina de observação viver num terceiro lugar importável pelos dois, ou no
próprio `pilota.py` e importada pelo exportador. **Não escreva a conta duas vezes** —
duas cópias divergem e a paridade passa a testar a si mesma.

## §6 O TECLADO

⚠⚠ O TETO NÃO É O DO CURRÍCULO. Decisão do dono, 2026-09-10: "como o comando aqui é
binário não quero vel máxima no andar". Uma tecla é liga-desliga, e uma rampa até os
2,0 m/s do currículo entrega corrida quando ele quer caminhada. O envelope de treino
(`lin_vel_x` até 2,0; `lin_vel_y` 1,0; `ang_vel_z` 0,7) é a faixa em que a política foi
treinada, e **não** a faixa em que se dirige.

Tetos por bandeira, com estes padrões:

| bandeira | padrão | teto do treino |
|---|---|---|
| `--vx-max` | **1,0** | 2,0 |
| `--vy-max` | **0,5** | 1,0 |
| `--wz-max` | **0,5** | 0,7 |

O passo por tecla é `teto / 5`, e não 0,1 fixo: assim cinco toques levam ao teto seja
qual for o teto, e baixar o teto não vira uma rampa longa.

⚠ Os tetos são ajustáveis AO VIVO, com `-` e `=`. Sem isso o dono reiniciaria o viewer
para calibrar o valor de caminhada, e o valor certo só se acha dirigindo.

| tecla | efeito |
|---|---|
| ↑ ↓ | `vx` ± passo, saturado em `±vx_max` |
| ← → | `vy` ± passo, saturado em `±vy_max` |
| `,` `.` | `wz` ± passo, saturado em `±wz_max` |
| espaço | zera o `twist` |
| `-` `=` | os três tetos × 0,8 e × 1,25, juntos, limitados ao envelope do treino |
| `0`–`4` | o elo: ANDAR, REORIENTAR, PEGAR, CARREGAR, BOTAR |
| `r` | reset: `qpos0` e `twist` zerado |
| `[` `]` | fator de tempo ÷2 e ×2, para câmera lenta |

⚠ O `-`/`=` NÃO passa do envelope do treino. Acima dele a política está fora da
distribuição e o que se vê não diz nada sobre o treino. Sature em 2,0 / 1,0 / 0,7 e
imprima um aviso ao bater no limite.

⚠ O teto RECORTA o valor corrente, não só os próximos toques: baixar o teto com o robô
a 1,0 m/s tem de derrubar o comando para o teto novo no mesmo passo.

O `mujoco.viewer.launch_passive` entrega teclado por `key_callback`.

Imprima na tela, todo passo, uma linha: elo por nome, `twist` pedido, `twist` medido,
os tetos correntes e o fator de tempo.

## §7 O LAÇO

```
por passo de controle (dt = 0,02 s):
    obs   = monta_observacao(d, twist, elo, acao_anterior)
    obs_n = (obs − mean) / sqrt(var + 1e-8)
    acao  = ator(obs_n)                      # torch.no_grad, float32, cpu
    d.ctrl[ids_atuador] = q_default + escala_acao * acao
    4 × mujoco.mj_step(m, d)                 # decimation = 4, physics_dt = 0,005
    acao_anterior = acao
    dorme o que faltar para o tempo real × fator
```

⚠ `decimation = 4` e `physics_dt = 0,005` vêm do cfg. Grave os dois no `.npz`; não os
escreva à mão.

⚠ A ação vai ao `ctrl` **sem clamp nosso**. Os atuadores são servos de posição com
`ctrlrange` no modelo, e o MuJoCo já satura. Um clamp a mais mudaria o comportamento
em relação ao treino.

## §8 O que este lote NÃO faz

Não toca `comando.py`, `recompensas.py`, `terminacoes.py`, `knobs.py`, `env_cfg.py`,
`observacoes.py`, `curriculo.py`, `runner.py`, `algoritmo.py`, `play.py`, o notebook,
nem qualquer spec. Não roda `smoke.py`. Não muda o contrato de observação.

Fora de escopo, declarado: a máquina de estados (espera, sustain, fecho, cadeia), a
caixa reposicionável, o giro pedido, gravação de vídeo, joystick.

## §9 Commits

Atômicos, Conventional Commits, mensagem em português, **sem trailer de atribuição**
(`git -c core.hooksPath=/dev/null`). Três:

1. `feat(limpo): exporta_cena — a cena de treino em .mjb para o MuJoCo clássico`
2. `feat(pilota): dirigir um checkpoint no MuJoCo clássico, twist e elo no teclado`
3. `docs(specs): g1-limpo pilota v1`

Nunca commite o `.mjb` nem: `docs/handoff/`, `docs/memoria/`, `record_rl.py`,
`rl_rollout.npz`, `g1_multitask/variacao_pose.py`, `g1_poc/*.patch`, `g1_poc/patches3/`,
`reference_checkpoints/*.pt`, `ver_play.py`. Nunca `git push`.
