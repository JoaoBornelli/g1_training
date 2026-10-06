---
tipo: achado
pai: "[[C2 g1_poc]]"
estado: ativa
verificado: 2026-10-06
resumo: "Calculado e conferido contra medição em 2026-08-24 (g1_poc bloco5, it 440)."
---
# g1-track-lin-parado-colhe-27pct (original)

Corpo original da memória `g1-track-lin-parado-colhe-27pct`, copiado sem edição em 2026-09-04. A memória guarda o resumo e aponta para este arquivo.

---

Calculado e conferido contra medição em 2026-08-24 (g1_poc bloco5, it 440).

## A armadilha

`Episode_Reward/track_linear_velocity` marcava **0,3998** e parecia progresso. O robô
estava IMÓVEL (`peak_height_mean` = 0,0029, `razao_marcha_ema` = 0,0004).

Normalizado pela regra do `leitura.py` (`× max_episode_length_s / (duração_real_s × peso)`
= `× 20 / (13,54 × 2,0)`): **0,2953**.

Previsão para um robô **perfeitamente imóvel**, sobre a distribuição do fabricante
(`lin_vel_x`, `lin_vel_y ~ U(−1,1)`, `std² = 0,25`, `rel_standing_envs = 0,10`):

```
E||v_cmd_xy||                        = 0,7654 m/s
E[exp(−||v_cmd||²/0,25)]  não-standing = 0,1942
0,9 × 0,1942 + 0,10 × 1,0            = 0,2748      <-- previsto
                            medido    = 0,2953
```

Batem em 7%. **Ficar parado colhe ~27% do termo de graça**, porque a caixa de comando
±1,0 contém muitos comandos de norma pequena, e os 10% de `standing` pagam o kernel
cheio (erro zero rende máximo com qualquer std).

Consequência prática: `track_linear_velocity` bruto **não** serve para dizer se o robô
anda. Use `razao_marcha_ema` (que é gateada por comando ativo e vale 0 no imóvel) ou
compare o valor normalizado contra o piso de 0,275.

## A conta do incentivo no mesmo ponto

```
ganho de andar (imóvel -> rastreio perfeito)  = 2,0 × (1 − 0,194) = +1,61 /s
d(track_lin)/dv em v = 0                     = 1,18 por (m/s)
custo feet_clearance a h = 2,9 mm            = 2,0 × |0,0029 − 0,10| = 0,194 por (m/s) de pé
                                               -> ~0,14 /s numa marcha
```

⚠ **Isto REFUTA a hipótese de que o `foot_clearance` a −2,0 é o bloqueio por custo**: o
ganho é ~11× o custo. A conta antiga de g1-push-gate-compounding-stall ("dar o passo
custa 2× o que o rastreio paga") valia num regime de comando de **0,1 m/s**; o comando
agora é ±1,0 e a proporção do fabricante voltou.

## O que sobra como suspeito: derivada zero na elevação do pé

A partir da postura parada, os três termos de marcha valem **zero**, portanto não há
gradiente para levantar o pé:

| termo | valor com o pé plantado | por quê |
|---|---|---|
| `feet_clearance` = Σ\|h − 0,10\| × ‖v_pé_xy‖ | 0 | `v_pé = 0` (`velocity/mdp/rewards.py:241-268`) |
| `feet_swing_height` | 0 | gateado em `first_contact`; sem pouso não há custo |
| `foot_slip` | 0 | idem, precisa de contato + velocidade |

Eles moldam o passo DEPOIS que ele existe. Nenhum o cria. É a terceira instância da
mesma classe neste projeto: g1-pegar-plato-do-grasp-booleano (força de aperto →
`squeeze`), o `unload` (força de apoio), e agora a elevação do pé.

⚠ **NÃO agir sobre isto ainda.** Contra-hipótese forte: o `velocity` do fabricante
produz marcha com exatamente estes termos, e ele treina muito mais — o 1º degrau do
`commands_vel` dele está em 5000 iterações, contra as 440 medidas aqui (9%). O platô em
`mean_reward = 16` pode ser o platô do "de pé" que quebra depois.

Critério de decisão em ~1500 iterações:

| observação | veredito |
|---|---|
| `razao_marcha` e `peak_height` subindo | era cedo; funciona |
| `razao_marcha` < 0,05 e `peak_height` ~0,003 | falta gradiente na elevação do pé; agir |
| `razao_marcha` subindo com `peak_height` ~0,003 | deslize; o alvo é o `foot_slip` (−0,1) |

Relacionado: g1-poc-portao-forma-mede-sobrevivencia,
g1-poc-recompensa-loco-prefere-parar, g1-mjlab-velocity-anatomia,
g1-push-gate-compounding-stall, g1-andar-como-rastreio-de-velocidade.
