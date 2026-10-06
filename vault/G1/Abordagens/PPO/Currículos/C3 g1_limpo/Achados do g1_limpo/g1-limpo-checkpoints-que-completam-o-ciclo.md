---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "MEDIDO 01–02/10 nos plays + veredito do dono: quais checkpoints pegam, botam e saem andando; MELHOR até 02/10 = model_23250 (21700 refinado no v2)"
cluster: c3
---
MELHOR MODELO (dono, 02/10): `model_23250` (sha 7967fa59d9b4), o `model_21700` da linha bloco refinado no `exp/g1-limpo-v2` (`39b8b82`, config do zero17), run `21700_zero_teste`. Bota em 0,55, 0,35 e 0,15 m com 1 e 3 kg.

MEDIDO 01/10 (roteiro `botar`): completam o ciclo `5800`, `5350`, `20500`, `21700`, `ckp9100`, `17999`, `19300`, `m24999`; a linha bloco falha em laje ≤ 0,15 m (menos o 21700). Veredito do dono: `20500` perna toda errada; `21700` perna torta; `5800` agachado, coxa na laje, só laje alta.

**Why:** o play é triagem, não veredito (antes de 01/10 o registrador era cego ao giro).
**How to apply:** partir do `model_23250` para qualquer refino na linha v2.
**Detalhe:** g1_training/vault/G1/Abordagens/PPO/Currículos/C3 g1_limpo/g1-limpo-registro-de-checkpoints.md (commit e sha256 de cada um)

Relacionado: g1-limpo-botar-esquecido-na-fase-4 g1-limpo-registrador-cego-ao-giro g1-limpo-pendencias
