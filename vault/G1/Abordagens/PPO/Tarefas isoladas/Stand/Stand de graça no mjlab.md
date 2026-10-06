---
tipo: achado
pai: "[[Stand]]"
estado: ativa
verificado: 2026-10-06
periodo: 26/08/26
resumo: "No mjlab o G1 fica de pé sem aprender; de pé no play não prova aprendizado."
---
## Fato
No mjlab o G1 fica de pé sem aprender nada.

## Evidência
`use_default_offset = True` faz `alvo = ação × escala + default_joint_pos`, e a pose padrão já está em pé (base 0,784 m, joelho 0,3 rad), segurada por atuador de posição. O `play` roda a média determinística: no `model_0.pt`, `|a|` médio 0,0735 (1,4° por junta).

## Implicação
Um checkpoint que fica de pé no `play` não prova aprendizado. Para separar aprendizado de PD: cair e levantar, resistir a empurrão, e comparar com `model_0.pt` na mesma tarefa.
