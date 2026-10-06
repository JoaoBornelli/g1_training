---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "MEDIDO por sonda de rede (30/09 e 01/10): o crítico desvaloriza descer e fechar no BOTAR e o ator manda subir a caixa; o resume K1b + R1 levou 48% das C ao BOTAR sem nenhum fecho; o fecho não exige de pé"
---
Sondas de rede, sem simulador, nos estados do play antigo. V(alvo) − V(congelado) no BOTAR: +15 (model_4000), +4,7 (4650), −8,7 (5000 de 30/09), −3,1 (5000 do resume de 01/10). O ator comanda subir as palmas em todo ponto da descida (+20 cm no alvo). V_CAUDA − V_BOTAR no alvo: +14,7 (4650), −3,6 (agora). O fecho é `perto ∧ alinhado ∧ apoiada`; de pé NÃO entra (comando.py:1559-1560).

**Why:** o crítico esquece estados que a política não visita, e sem fecho nada paga descer.
**How to apply:** MEDIDO 01/10 com o alvo real no CSV: o `model_5800` (R4) põe a caixa a 4 cm do alvo, tombo 4°; o 5350 parava a 12 cm, o 5000(1) a 20 cm. O R4b (laje com profundidade pelo `s_C`) levou o `s_C` a 0,20 na it 6320; ~50% das C que chegam ao BOTAR terminam com sucesso.
**Detalhe:** docs/relatorios/2026-09-30-plato-do-botar.md

Relacionado: g1-limpo-teleporte-da-laje-no-corpo g1-limpo-carregar-parado-e-estatua g1-limpo-botar-e-bang-bang-nao-pressa
