---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o knobs.py documentava 1-exp para velocidade_por_regime mas o codigo roda um clamp; comentario corrigido em 2026-09-11, peso -2,0 mantido, clamp segue como licenca acima do teto"
cluster: c3
---
ACHADO 2026-09-10, CORRIGIDO 2026-09-11 (`exp/g1-limpo-v2`, commit `a6259fe`) — so o comentario; nenhuma linha de codigo mudou.

`knobs.py` justificava `velocidade_por_regime = −2.0` com `1 − exp(−media(v²/vmax²))` e "no dobro do limite custa 1,96/s". O codigo (`recompensas.py:805`) roda `clamp(mean((v/vmax)²), max=4.0)`: no limite custa **2,0/s**, no dobro **8,0/s** — 4,1× a derivacao do comentario. O `smoke.py:4401-4412` trava a forma do clamp de proposito (a `1 − exp` tinha derivada zero no teto): **o codigo e o intencional**.

**O peso −2,0 FICA.** Baixa-lo para casar com a aritmetica velha andaria para tras: nem 8,0/s segura o robo contra um degrau de fecho de 18,55/s.

Medicao feita pela formula velha sai ~2× baixa — o "1,08/s" da pressa e 2,17/s.

**Debito aberto:** o clamp e licenca acima do teto — 2,71% dos passos do `PEGAR` com a caixa estao em `valor = 4,0`, com derivada zero. Conserto espera `vel_max_standing` por familia de junta.

Relacionado: g1-limpo-velocidade-de-junta-medida, g1-limpo-renda-congelada-paga-10x-o-deficit
