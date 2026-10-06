---
tipo: abordagem
pai: "[[Tarefas isoladas]]"
estado: validada
verificado: 2026-10-06
periodo: 15–24/07/26
resumo: "Pegar e sustentar a caixa parado, agachar emergente. Validado até prateleira baixa e 5 kg; superado porque não encadeia."
cluster: tarefas-isoladas
---
## Hipótese
O robô parado prensa a caixa entre as mãos e a sustenta na altura comandada. Agachar EMERGE de ter que alcançar a caixa mais baixo; não é prescrito.

## Método
`skills/lift/` (`env.py`, `rewards.py`, `knobs.py`, 10 configs datados). Cadeia de recompensa MONOTÔNICA `reaching → grasp → lift → sustain_precise`, com anti-hacks (`back_penalty`, `table_contact`, `com_balance`) e fundação escopada (`upright` cheio, postura só em perna e cintura; braços livres). Sequência de fine-tunes, cada um com warm-start do anterior: ver os filhos.

## Resultado
Pegou, ergueu reto e sustentou. Currículo de altura de prateleira 0,55 → 0,45 → 0,35 → 0,25 com peso de 0,5 a 5 kg (a memória registra pega até prateleira de 0,05 m). Checkpoints: `model_6550_lift_altura_normal`, `model_7300` (2 alturas), `model_13200_lift_55-00`, `model_15700`.

## Por que parou
Skill isolada não encadeia: esqueceu o Stand e não anda. Superada pelos currículos. Defeito descoberto depois, já no multitask: o `progress` do `lift` vale sempre 1 quando o alvo é a própria caixa (ver Lift não mede erguer).

## O que herda
A cadeia monotônica anti-vale, o PLR de altura, o peso por força externa e o princípio anti-blanket seguiram para o `g1_multitask` e o `g1_limpo`.

## Filhos
<!-- filhos:inicio -->
- **Caixa na borda** (abordagem · validada · 16/07/26) — Caixa no centro da mesa era inalcançável em pé; mover para a borda resolveu sem penalidade nova.
- **Currículo de altura com PLR** (abordagem · validada · 16–17/07/26) — Prateleira mocap e PLR de altura 0,55 → 0,25; nível inalcançável colapsa o currículo.
- **Generalizar e rehearsal** (abordagem · validada · 16/07/26) — Variar a caixa e empurrar sob carga; rehearsal de 20% contra o esquecimento do Stand.
- **Movimento suave** (abordagem · validada · 17–24/07/26) — Corre, bate, abre espacate e rebola; amortecer por escala de ação e gate, nunca penalidade sempre ligada.
- **Orientação da caixa** (abordagem · validada · 15–16/07/26) — Deitar a caixa pagava altura de graça; gate de 10° e depois kernel suave de orientação.
- **Peso variável** (abordagem · validada · 17/07/26) — DR de massa corrompe a heap; peso por força externa (payload) de 0,5 a 5 kg.
- **Lift não mede erguer** (achado · ativa · 06/08/26) — Com alvo = caixa o progress do lift vale sempre 1 e dá NaN; separar alvo de navegação e de tarefa.
- **O vale da cadeia** (achado · ativa · 15/07/26) — reaching × (1 − contato) zerava no toque; a cadeia de recompensa tem de ser monotônica.
<!-- filhos:fim -->
