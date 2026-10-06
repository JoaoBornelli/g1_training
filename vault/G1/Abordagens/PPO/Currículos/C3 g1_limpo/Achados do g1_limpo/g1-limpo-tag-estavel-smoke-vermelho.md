---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "a tag estavel-bloco17 nasce com 2 falhas no smoke; as duas são contadores que não subiram com o faixa_de_pose, não defeito de código"
cluster: c3
---
A tag `estavel-bloco17` — o commit que produziu o `model_14000` — roda o smoke com
**636 ok / 2 falhas** (medido 2026-09-14). Nenhuma das duas é defeito de código.

Causa única: o `faixa_de_pose` (`c9578ae`) entrou na tag e dois CONTADORES não
subiram junto.

1. `len(cfg.rewards) == 28`, e são 29. O 29º é o `faixa_de_pose`.
2. `knobs.py`.count("knee") == 4, e são 7. A tabela nova do `faixa_de_pose` traz uma
   ocorrência e o docstring dela traz duas. O check contava a PALAVRA no arquivo
   inteiro, portanto menção em docstring contava.

Conserto na `exp/g1-limpo-alvo-e-inercia`: o contador vai a 29, e o do `knee` passa a
contar o PADRÃO `.*knee.*` (cinco tabelas) em vez da palavra.

**Why:** um portão que nasce vermelho deixa de gatear — a falha nova some no ruído.
**How to apply:** termo de recompensa novo sobe o contador no mesmo commit.

Relacionado: g1-limpo-faixa-de-pose, g1-limpo-inercia-real-da-caixa
