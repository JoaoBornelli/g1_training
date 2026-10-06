---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o alinhado do BOTAR congela a normal da caixa TOMBADA nas mãos e passa a exigir que ela mantenha o tombo; o sigma_ori trava no piso e o gradiente morre"
cluster: c3
---
MEDIDO 2026-09-21, `model_9100`, três alturas de laje. A caixa chega ao BOTAR tombada
29,7° (laje 0,55) e 52,2° (laje 0,35) e termina a 0,0°: a laje a nivela. O
`_congela_face` captura a normal no instante TOMBADO, então `alinhado` (25°) exige
manter o tombo. O giro punido é o endireitamento.

E o `sigma_ori` trava no piso de 0,20 rad SEMPRE: `_congela_face` e `_recalcula_sigmas`
rodam no mesmo passo (`comando.py:720-721`) e o erro inicial é zero por construção. Nos
33° medidos, `precise_ori` = 2e−4, derivada 1e−4 por grau.

⚠ NÃO remover o `alinhado`: o dono quer orientação pré-determinada. Mas isso só vale
quando o REORIENTAR sair de inerte. Hoje: referência = vertical do mundo.

**Why:** σ = erro inicial não se aplica a alvo congelado, que nasce com erro zero.
**How to apply:** trocar a REFERÊNCIA, nunca o peso nem o σ sozinho.
**Detalhe:** `g1_training/vault/G1/Abordagens/PPO/Currículos/C3 g1_limpo/2026-09-21-botar-alinhado-pune-endireitar.md`

Relacionado: g1-limpo-botar-deita-o-tronco, g1-limpo-ik-referencia-botar
