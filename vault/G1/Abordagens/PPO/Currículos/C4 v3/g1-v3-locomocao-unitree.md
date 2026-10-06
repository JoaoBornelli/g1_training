---
tipo: decisao
pai: "[[C4 v3]]"
estado: ativa
verificado: 2026-10-06
resumo: "DECISÃO 02/10: a locomoção da v3 segue a receita do unitree_rl_lab (G1 29dof Velocity), reescrita sem importar; 05/10: frente 1,5 m/s, trás e lado 1,0, giro ±1,0 rad/s; rastreio normalizado; observação com histórico de 5"
cluster: c4
---
DECISÃO DO DONO (02/10): a locomoção da v3 é reescrita a partir da `Unitree-G1-29dof-Velocity` do `unitree_rl_lab` (commit `4960b84`), pura, sem importar o código. Faixa (dono, 05/10): frente 1,5 m/s, trás e lado 1,0, giro ±1,0 rad/s (180° em ≤ 4 s); sorteio pelo balanceador da v3; posturais só no ANDAR (CARREGAR sem os braços); a observação dela em todas as tarefas.

DESVIOS (dono, 05/10): rastreio normalizado pelo comando g1-limpo-giro-no-lugar-empata-no-estagio-0; período da marcha cresce quando a velocidade cai (com 0,8 s fixo o passo era 0,4·v), com a fase na observação; `alive` zero no PEGAR e no BOTAR.

**Why:** a receita da Unitree anda bem no G1 real, e o treino vai para o Isaac Lab.
**How to apply:** a locomoção segue a §19 da spec; os desvios estão na §19 e nas decisões de 05/10.
**Detalhe:** g1_training/docs/relatorios/2026-10-05-revisao-spec-v3.md

Relacionado: g1-limpo-v3-alvos-macro g1-treino-vai-para-isaac-lab
