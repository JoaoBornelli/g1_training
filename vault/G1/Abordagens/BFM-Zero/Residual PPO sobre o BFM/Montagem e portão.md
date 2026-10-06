---
tipo: abordagem
pai: "[[Residual PPO sobre o BFM]]"
estado: validada
verificado: 2026-10-06
periodo: 30/07/26
resumo: "Ator do BFM em lote no mjlab, validado contra o fabricante (erro 1e-6); registra 7 armadilhas silenciosas da obs."
cluster: bfm
---
## Hipótese
Dá para rodar o ator do BFM em lote na GPU dentro do mjlab, com a obs idêntica à do fabricante.

## Método
- `extrai_ator.py`: do checkpoint de 3,15 GB sobram 122 MB (`_actor` 2048×6 + `_obs_normalizer`), mais a tabela dos 41 comportamentos com `z`.
- `obs_bfm.py`: a obs do BFM reescrita de numpy/um robô para torch/N robôs.
- `referencia.py`, o PORTÃO: compara com um rastro do caminho original (`dump_referencia.py`).
- `fumaca.py`: o BFM puro no nosso env, residual em zero.

## Resultado
Portão: ator erro máx 1,39e-06; `joint_pos`/`joint_vel` 0; `state[64]` 4,5e-08; histórico de 372 exato em 40 passos. Fumaça: `move-ego-0-0` 100% de pé em 150 passos; nenhum agachamento do BFM sobrevive sozinho (`crouch-0` 0% de pé, pelve 0,12 m; `crouch-N` é ALTURA alvo, não intensidade). Com caixa nas mãos o BFM fica de pé 150/150.

## Por que parou
Concluída; abriu o treino.

## O que herda
Armadilhas silenciosas que o portão pegou:
- A pose padrão do BFM difere da nossa em até 34,4° (joelho 0,300 × 0,669; cotovelo 0,000 × 0,600).
- Escala de ação: razão 0,800 em 25 juntas; `hip_pitch` 1,973 e `hip_roll` 1,264. É vetor, não escalar.
- `ang_vel` × 0,25 e `last_action` × 5,0.
- O histórico do BFM é `[t, t−1, t−1, t−2]`, não `[t, t−1, t−2, t−3]`. Errar isso fez o robô "dançar".
- `qvel[3:6]` do freejoint é frame LOCAL, mas `write_root_link_velocity_to_sim` recebe MUNDO.
- O normalizador do BFM é obrigatório; sem ele o ator devolve lixo sem exceção.
- `FBcprAuxModelConfig.device` aceita só `cpu`/`cuda`, não `cuda:0`.
