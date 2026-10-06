---
tipo: decisao
pai: "[[C4 v3]]"
estado: ativa
verificado: 2026-10-06
resumo: "DECISÃO 02/10: a v3 treina primeiro no mjlab (MuJoCo) e muda em breve para o Isaac Lab; a lógica fica em funções puras sem import do mjlab"
cluster: c4
---
DECISÃO DO DONO (02/10): o treino da v3 roda primeiro no mjlab (MuJoCo Warp) e muda em breve para o Isaac Lab.

**Why:** o simulador vai trocar; código preso ao mjlab teria de ser reescrito.
**How to apply:** na v3, alvos, potencial, balanceador e estimador são funções puras (`torch`/`numpy`, sem mjlab); só o `env_cfg` e os termos ficam no mjlab e serão reescritos.
**Detalhe:** g1_training/specs/g1-v3.md §1 e §5

Relacionado: g1-limpo-v3-alvos-macro unitree-g1-sim-mujoco-not-isaac
