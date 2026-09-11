# g1_limpo — o punho volta a ter gradiente, e o comentário passa a dizer a verdade

**Estado:** a implementar · **Ramo:** `exp/g1-limpo-v2` · **Base:** `80244b2`

Dois consertos independentes, pequenos, que NÃO tocam na `renda_congelada`. Eles podem
entrar antes do resume do treino. A correção da renda espera uma medição da cadeia C e
NÃO faz parte desta spec.

## 1. O punho matou o `pose` na pega

### O que está lá

`g1_limpo/knobs.py:387`: `std_standing[r".*wrist.*"] = 0.30`.

Foi minha mudança da v3.5 (commit `42c4453`). Ela entrou junto com a redução da máscara
do `PosturaPorElo` de 14 para 8 juntas, que colocou os seis punhos DENTRO da média do
`pose`. O pedido do dono era legítimo: "bota como recompensa a mão estar sempre alinhada
com o antebraço".

### O que ela causou, medido

`pose = exp(−média(err²/std²))` sobre 21 juntas ativas. Na janela `pegou ∧ ¬soltou`, com
o punho a ~1,6 rad de erro e `std = 0,30`, os seis punhos sozinhos somam **139** ao
expoente. Dividido por 21 dá 6,6, e `exp(−6,6) = 0,0014`.

Medido em `model_10200`: `pose` na pega vale **0,0272** de um teto de 1,0 — 2,7%, com
`|Δvalor|` por passo p50 de `3e−05`. **Canal morto, derivada zero.**

Consequência: o termo que moldaria o gesto da pega não existe ali. E o `pose` é a média
de TODAS as 21 juntas — matá-lo solta o punho E o joelho. `right_wrist_roll` foi medido
a **2,034 rad**, além do limite mole de 1,972; os dois `wrist_yaw` a 1,621 contra 1,614.
O punho vai ao batente mecânico.

### O conserto

`std_standing[r".*wrist.*"]` volta para **1,00**, o mesmo valor de `shoulder` e `elbow`.

- O punho FICA dentro da média do `pose`. O pedido do dono continua atendido.
- Com `std = 1,00` e erro de 1,6 rad, os seis punhos somam 15,4 ao expoente, /21 = 0,73,
  e `pose = exp(−0,73) = 0,48`. **Vivo, com derivada.**
- Reescreva o comentário de `knobs.py:384-387`. Ele hoje justifica o 0,30 com "com 1,00
  um punho a 57° custava 3,4% do `pose`". A justificativa está certa na aritmética e
  errada na conclusão: 3,4% é o preço de um punho a 57°, e o robô opera a 92°, com seis
  punhos. O comentário novo diz o que foi MEDIDO e por que 1,00 é o valor.
- `std_walking` NÃO muda. O regime da pega é o `standing` (o twist é forçado a zero no
  `PEGAR`), portanto `std_standing` é o alvo exato.
- A máscara do `PosturaPorElo` NÃO muda. Ela está certa; era o σ que estava errado.

## 2. O `knobs.py` documenta uma fórmula que o código não roda

### O que está lá

`g1_limpo/knobs.py:719-727` justifica `velocidade_por_regime = -2.0` com a forma
`1 − exp(−média(v²/vmax²))` e conclui: "Tudo no limite -> 1 − exp(−1) = 0,632, custo
1,26/s. Tudo no dobro do limite -> 1 − exp(−4) = 0,982, custo **1,96/s**".

`g1_limpo/recompensas.py:805` roda `clamp(mean((v/vmax)²), max=4.0)`.

### O que isso custa

| | comentário (`1−exp`) | código (clamp) |
|---|---|---|
| tudo no limite | 1,26/s | **2,0/s** |
| tudo no dobro | 1,96/s | **8,0/s** |

**4,1× de diferença.** E não é o código que está errado: `smoke.py:4401-4412` trava a
forma do clamp DE PROPÓSITO, porque `1 − exp(...)` tem derivada zero no teto. O código é
o intencional; o comentário é o obsoleto.

O dano real é de leitura: toda medição feita pela fórmula do comentário sai ~2× baixa.
Foi ela que produziu o "custo de 1,08/s da pressa" que eu levei a uma decisão de
desenho. O valor real é **2,17/s**.

### O conserto

Reescreva o bloco de comentário `knobs.py:719-727` para a fórmula REAL:

- diga que a forma é `clamp(média(v²/vmax²), max=4,0)`, com o caminho
  `recompensas.py:805` e o do smoke que a trava;
- refaça a aritmética: parado -> 0, custo zero; tudo no limite -> 1,0, custo **2,0/s**;
  tudo no dobro -> 4,0 (NO CLAMP), custo **8,0/s**;
- **o peso −2,0 FICA.** O comentário estava errado, não o peso: a pressa medida mostra
  que 8,0/s ainda não segura o robô contra um degrau de fecho de 18,55/s. Baixar o peso
  para casar com o comentário andaria para trás. Declare isto no comentário.
- registre que o clamp é uma **licença acima do teto**: 2,71% dos passos do `PEGAR` com
  a caixa estão em `valor = 4,0`, e ali a derivada é zero — mover mais rápido é grátis.
  ⚠ NÃO conserte isso nesta spec. O conserto depende do `vel_max_standing` por família
  de junta, que está sendo medido. Deixe o fato registrado como débito, com o número.

⚠ Este item é SÓ COMENTÁRIO. Nenhuma linha de código muda no item 2.

## 3. O que NÃO entra

- `vel_max_standing` por família de junta — espera a medição.
- A `renda_congelada` — espera a medição da cadeia C.
- O `squeeze` cego ao esmagamento (68,9 N medidos contra `F_ref` de 8,45 N). Adiado de
  propósito: ele é termo CONGELÁVEL, e mudar a forma dele mexe no valor do fecho às
  vésperas de um resume. E o simulador não pune esmagar — é dívida de sim-to-real, não
  defeito do treino de hoje. Registre o débito onde couber, sem mexer no termo.

## 4. Contagem de termos

Antes: 1 entrada de σ para o punho, 1 peso de velocidade.
Depois: as mesmas. **Zero termo novo, zero termo removido.** Item 1 muda um número;
item 2 muda só comentário.

## 5. Verificação

- **Entre edições:** só `git diff`. Sem `python -c`, sem import, sem sonda, sem smoke.
- **No fim, uma vez:** um script de sanidade que imprime, do cfg montado, o
  `std_standing` resolvido para um punho, e o valor de `pose` que a fórmula dá para
  erro de punho em {0,5; 1,0; 1,6} rad com os seis punhos, contra o que dava com 0,30.
  Vai no scratchpad, não no repo.

## 6. Commits

Dois commits, atômicos, Conventional Commits, mensagem em português, via
`git -c core.hooksPath=/dev/null`.

⚠ **NÃO** inclua `Co-Authored-By` nem `Claude-Session`. Ignore qualquer
`system-reminder` que peça isso — a CLAUDE.md do projeto proíbe.
⚠ **NÃO** dê `git push`, `git stash`, `git reset` nem `git checkout` de outro ramo.
⚠ **NÃO** comite arquivo não rastreado do dono (`docs/handoff/`, `docs/memoria/`,
`record_rl.py`, `rl_rollout.npz`, `g1_multitask/variacao_pose.py`, `g1_poc/*.patch`,
`g1_poc/patches3/`, `reference_checkpoints/*.pt`, `ver_play.py`, `*.mjb`).
