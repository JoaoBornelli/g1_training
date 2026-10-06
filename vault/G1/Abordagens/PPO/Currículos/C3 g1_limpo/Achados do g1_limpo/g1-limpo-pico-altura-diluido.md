---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "Episode_Metrics/pico_de_altura do g1_limpo é diluído ~3,6×; o número físico é o Metrics/peak_height_mean do fabricante"
---
O `pico_de_altura` do `g1_limpo/metricas.py` **subestima a folga do pé por ~3,6×**.
Verificado 2026-08-27 lendo os dois códigos.

A causa é o denominador. O fabricante divide pelo NÚMERO DE POUSOS
(`velocity/mdp/rewards.py:313-317`), logo env sem pouso não entra em numerador nem
denominador. A nossa usa `_media_por_env`, que devolve **0,0** para env sem pouso, e esse
zero entra na média sobre os envs. O fator é a fração de envs que pousam por passo, medida
em ~0,28.

Erro oposto no lado do fabricante: a classe `feet_swing_height` dele não define `reset()`,
então o pico vaza entre episódios e INFLA o valor. O vazamento só importa com queda em
massa; com `fell_over` baixo ele é desprezível.

**Why:** eu li 2,8 mm como física e afirmei "o pé não sai do chão". O valor real era
10,3 mm.

**How to apply:** com `fell_over` baixo, use o `Metrics/peak_height_mean` do fabricante
como o número físico. Não conserte a métrica no meio de um run — o denominador novo torna a
curva incomparável. Conserte entre blocos, trocando o denominador por contagem de pousos.

Relacionado: g1-mjlab-velocity-anatomia, g1-leitura-log-e-kaggle,
g1-limpo-bloco1-marcha-formou
