---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "no g1_limpo o BOTAR nunca fecha; a politica achou duas fugas em sequencia — arremessar e segurar para sempre — e as duas vinham do _alcancar identico a 1"
cluster: c3
---
`BOTAR` é ALCANÇADO e nunca FECHA. A política achou duas fugas em sequência.

**Arremessar** (v3.1, `bloco11`): `impacto` 7,70, `s_C` 0,0000. v3.2 fez soltar fora do alvo TERMINAR.

**Segurar para sempre** (v3.2, `bloco12` it 7499): `time_out` 66,4%, `load` 0,0001, `s_C` 0,0000, reward 48 → 152. A renda triplicou e nada foi apoiado.

**Mecanismo comum:** `_alcancar ≡ 1` no BOTAR fazia `staged` e `precise_ori` pagarem 3,99/s sem exigir a mão. Fechar congelava a mesma soma: o fecho ganhava zero e arriscava morte. `precise_pos` nunca teve portão de mão — é desenho (aceite).

**v3.3:** tira o `≡ 1` do `_alcancar`. O robô passa a APOIAR (`load` 0 → 0,441/s) e o BOTAR segue sem fechar.

**O bloqueio era o `de_pe`, MEDIDO** (`model_9499`, 62 109 passos em BOTAR): dos passos com três portões verdadeiros, 99,975% falham só nele, sempre pelo JOELHO (`dq` p50 1,457 contra limiar 0,69). Causa geométrica: a laje fica a 0,30–0,55 m, apoiar exige agachar, `de_pe` proíbe agachar. O PEGAR fecha porque o alvo dele é o PEITO — o robô ergue a caixa e se levanta antes de fechar.

**v3.4/v3.5 RESOLVEU (`bloco14`, it 9847):** `s_C` **0,0000 → 0,1952**, `sucesso` 0,167 → 0,306. `de_pe` sai do fecho do BOTAR e fica no PEGAR; a cauda vira parada com a cena intacta; `VALIDA` lê o elo publicado.

**Why:** todo termo que paga por estado da caixa tem de exigir a contribuição do robô.
**How to apply:** se o `load` continuar em zero, o próximo alvo é o peso do `load`, não o `_alcancar`.

Relacionado: g1-limpo-plano-tres-tarefas, g1-principio-incentivo-nao-penalizacao
