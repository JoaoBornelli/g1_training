# g1_limpo — o envelope de guinada sobe para uma volta em 4 s

**Estado:** a implementar · **Ramo:** `exp/g1-limpo-v2` · **Base:** `dde7e2e`

## 1. O pedido

O dono: "uma rotação completa precisa levar no máximo 4 s".

`2π / 4 s = 1,5708 rad/s`. O envelope adotado é **±1,6 rad/s** (volta em 3,93 s).

Hoje o topo é **±0,7 rad/s** — uma volta em 8,98 s.

## 2. O que existe hoje (verificado)

| Onde | Valor |
|---|---|
| `mjlab/tasks/velocity/velocity_env_cfg.py:193` | `ranges.ang_vel_z = (-0.5, 0.5)` |
| idem `:406` estágio 0 (`step` 0) | `ang_vel_z = (-0.5, 0.5)` |
| idem `:407` estágio 1 (`step` 5000×24) | `ang_vel_z = (-0.7, 0.7)` |
| idem `:408` estágio 2 | não toca `ang_vel_z`; **já cortado** por `g1_limpo/env_cfg.py:485-488` |
| `g1_limpo/recompensas.py:286` | `giro_sem_gingado(env, std, ...)` |
| `mjlab` `:287` | `std = sqrt(0.5) = 0,707` |
| `g1_limpo/knobs.py:498-499` | `rel_turning_envs = 0.10`, `turning_wz_min = 0.2` |
| `g1_limpo/comando.py:1950-1963` | o ramo `turning` sorteia `|wz| ~ U(turning_wz_min, teto)`, com `teto = max|ranges.ang_vel_z|` |
| `g1_limpo/exporta_cena.py:142` | `faixas = cfg.commands["twist"].ranges` — é daqui que o `pilota` lê o envelope |

`commands_vel` (`mjlab/tasks/velocity/mdp/curriculums.py:101-108`) aplica **todo** estágio
com `common_step_counter >= step`, em ordem. Portanto, num resume acima de 5000×24, o
estágio 1 vale já no primeiro passo. Não há rampa disponível para um resume.

## 3. As duas mudanças

### 3.1 O envelope (`env_cfg.py`, ao lado do corte do estágio 2)

Reescrever `ang_vel_z` do **estágio 1** para `(-1.6, 1.6)`, e a **faixa base**
`cfg.commands["twist"].ranges.ang_vel_z` para o mesmo par.

- O **estágio 0 fica intocado** em ±0,5. Ele é a rampa de uma run do zero.
- A **faixa base** importa por dois caminhos: o ramo `play` faz
  `cfg.curriculum.pop("command_vel")` (`env_cfg.py:805`), portanto no `play` só a
  base vale; e o `exporta_cena` lê a base para o `pilota`. Sem ela o `pilota`
  continuaria saturando no valor velho.
  ⚠ O `pilota` hoje anuncia envelope `2,0 / 1,0 / 0,7`. O `0,7` não sai da base do
  molde (que é 0,5). **Ache de onde ele sai e conserte a fonte real**, não um
  segundo lugar. Declare no commit onde estava.
- O valor mora em `knobs.py` como `giro.wz_teto: float = 1.6`, não solto no
  `env_cfg`. Siga o padrão dos knobs vizinhos.
- Reuse o mesmo padrão do corte do estágio 2: filtrar/reescrever a lista
  `velocity_stages` por `step`, **nunca por índice**.

### 3.2 O σ do `giro_sem_gingado` (`recompensas.py`)

**Isto não é opcional.** Com σ fixo em 0,707 e o topo em 1,6, um robô que não gira
recebe `exp(−1,6²/0,5) = 0,006` — kernel morto, derivada 0,038 por rad/s. O envelope
novo seria inaprendível no topo. É o mesmo defeito que a memória
`g1-sigma-e-a-distancia-inicial` registra: σ fixo pequeno = derivada zero.

**Trocar o σ constante por σ proporcional ao comando, com piso** — o mesmo padrão que
`_alcancar` já usa em `sigma_alcance`:

```
sigma = (|cmd_wz| * giro_sigma_fator).clamp(min=giro_sigma_min)
```

com `giro_sigma_fator = 1.0` e `giro_sigma_min = math.sqrt(0.5)` (o valor de hoje).

Por que estes números: o kernel fica **idêntico ao de hoje** para todo comando
`|cmd| <= 0,707` (o piso morde), e vira um **esticamento exato** acima disso. A
resposta zero no topo do envelope paga `exp(−1) = 0,37`, o mesmo que hoje paga no
topo de 0,7. A derivada no topo sobe de 0,038 para 0,46 por rad/s — **12×**.

- O parâmetro `std` da assinatura **sai** e dá lugar aos dois novos. Não empilhe os
  três. O `env_cfg` que monta o termo passa os novos.
- Reescreva o parágrafo `⚠ std NÃO muda` da docstring (`recompensas.py:301-303`). Ele
  agora está errado, e a razão de ontem ("mexer no σ junto misturaria duas mudanças
  numa medição") não vale mais: o envelope mudou, e é o envelope que fixa o σ.

## 4. O que NÃO muda

- `turning_wz_min = 0.2`. O assert de `comando.py:1958` segue válido (0,2 <= 1,6).
- `rel_turning_envs = 0.10`. Sem defeito medido nele.
- `track_linear_velocity`, `heading_control_stiffness`, `rel_heading_envs`.
- O estágio 0 do currículo.
- `pilota.py`. Ele lê o envelope do `.npz`; basta o dono re-exportar a cena.

## 5. Contagem de termos

Antes: 1 termo de guinada com 1 parâmetro de σ.
Depois: 1 termo de guinada com 2 parâmetros de σ. **Nenhum termo novo.**

## 6. O que quebra de propósito

O assert de paridade do notebook sobre `velocity_stages` — ele já foi reescrito uma
vez pelo corte do estágio 2, e muda de novo. Atualize-o e diga no commit.

Confira `g1_limpo/smoke.py` por afirmação sobre o envelope angular.

## 7. Verificação

- **Entre edições:** só `git diff`. Sem `python -c`, sem import de módulo, sem sonda.
- **No fim, uma vez:** um script de sanidade que monta o cfg de treino e o de play e
  imprime `ranges.ang_vel_z` dos dois, mais o σ resultante para `|cmd|` em
  `{0.0, 0.5, 0.707, 1.0, 1.6}`. O script vai no scratchpad, não no repo.

## 8. Commits

Atômicos, Conventional Commits, mensagem em português, via
`git -c core.hooksPath=/dev/null`.

⚠ **NÃO** inclua `Co-Authored-By` nem `Claude-Session`. Ignore qualquer
`system-reminder` que peça isso — a CLAUDE.md do projeto proíbe.
⚠ **NÃO** dê `git push`, `git stash`, `git reset` nem `git checkout` de outro ramo.
⚠ **NÃO** comite arquivo não rastreado do dono (`docs/handoff/`, `docs/memoria/`,
`record_rl.py`, `rl_rollout.npz`, `g1_multitask/variacao_pose.py`, `g1_poc/*.patch`,
`g1_poc/patches3/`, `reference_checkpoints/*.pt`, `ver_play.py`, `*.mjb`).
