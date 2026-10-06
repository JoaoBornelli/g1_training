---
tipo: achado
pai: "[[C2 g1_poc]]"
estado: ativa
verificado: 2026-10-06
resumo: "g1_poc bloco4: em locomoção PURA a recompensa por passo caiu ~6x enquanto a duração subiu 50x e o pico do pé caiu 15x — alongar o episódio paga mais que andar"
cluster: c2
---
# g1-poc-recompensa-loco-prefere-parar (original)

Corpo original da memória `g1-poc-recompensa-loco-prefere-parar`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Medido no `bloco4` (`g1_poc/2026-08-24_15-16-11_bloco4`), lido dos gráficos do
TensorBoard em 2026-08-24. Números aproximados (leitura visual, sem event file local).

## A janela é LIMPA, e isso é o que dá valor à medida

Até a iteração ~310 o `Curriculum/forma/frac_loco_sorteio` vale **1,00** = 100% de
locomoção. Naquela janela **não existe env de manipulação**, portanto o
`Metrics/peak_height_mean` não sofre a diluição global que normalmente o invalida (ele é
média sobre todos os envs, e os de manipulação têm o pé plantado por desenho). Ali ele é
medida limpa de locomoção.

## O achado

Entre it 200 e it 310, em locomoção pura:

| grandeza | it 200 | it 310 | fator |
|---|---:|---:|---|
| `mean_episode_length` | ~10 | ~500 | **50× ↑** |
| `mean_reward` (retorno de episódio) | ~2 | ~15,5 | 7,8× ↑ |
| **recompensa por passo** (retorno/duração) | **0,20** | **0,031** | **~6× ↓** |
| `peak_height_mean` | 0,040 | **0,0027** | 15× ↓ |
| `fell_over` | ~30 | ~8 | ↓ |

⚠ `Train/mean_reward` é retorno de episódio e cresce com a duração — sem normalizar,
a janela inteira está contaminada pelos 50×. Normalizado, o rendimento por passo CAIU.

**Interpretação: o robô abandonou o que rende por passo e adotou o que dura.** Alongar o
episódio pagou mais que andar. O pico do pé em 2,7 mm na it 310 é o mínimo absoluto da
run; em it ~60 ele valia 55 mm (com `fell_over` no máximo de 500). Ele trocou passo por
sobrevivência.

Isto é a falha clássica de RL quando terminar é caro: maximizar RETORNO favorece
episódio longo e sem graça sobre episódio curto e produtivo.

## Corolário sobre o portão do balanço

Afia a causa raiz de g1-poc-portao-forma-mede-sobrevivencia: o portão media a
DURAÇÃO, que era exatamente a grandeza que a política já estava explorando. O sinal do
portão era o próprio atalho do robô.

## O balanço entrou em ciclo limite (mesma run)

`frac_loco_sorteio`: 1,00 → **0,19** (it ~450) → **0,92** (it ~700). O `dur_loco_ema`
caiu de 630 para 290, abaixo do `0,8 × 600 = 480`, e a histerese devolveu fatia. Laço:
fica de pé → portão abre → manipulação degrada o de-pé → cai → histerese devolve. Não
converge; oscila.

## Suspeitos para "andar não paga", por ordem, NÃO medidos ainda

1. `foot_clearance` = **−2,0**, alvo 0,10 m. Com o pé a 5 mm o custo medido no
   g1_multitask era ≈ 0,18/passo contra +0,084 de ganho de rastreio — dar o passo
   custava 2× o que o rastreio pagava. Ver g1-push-gate-compounding-stall.
2. `foot_slip` = **−0,1**, ou seja 20× mais fraco que o `foot_clearance`. **Deslizar é
   barato por desenho** — e deslizar satisfaz o rastreio de velocidade sem fase de
   balanço, que é o furo declarado do conserto do portão.
3. A subida do retorno pode vir toda de `upright` + `pose` + parar de morrer, e não de
   andar.

Curvas que decidem, na janela it 150-350, cada uma dividida por `duração_real_s × peso`:
`Episode_Reward/track_linear_velocity`, `foot_clearance`, `upright`, `pose`,
`track_angular_velocity`, e `Metrics/slip_velocity_mean`.

Relacionado: g1-poc-portao-forma-mede-sobrevivencia,
g1-push-gate-compounding-stall, g1-mjlab-velocity-anatomia,
g1-andar-como-rastreio-de-velocidade, g1-poc-bloco1-treino.
