---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "No CARREGAR parado o rastreio ×3,5 pagava 14/s por ficar; o R2 criou desvio pago e multiplicador inobservável; o K1b (×1 no regime parado, em toda cadeia) o substitui, commitado e em treino"
cluster: c3
---
MEDIDO 30/09: no CARREGAR com twist zero, o rastreio ×3,5 (knobs.py:1311-1312) paga 14/s por ficar parado; fechar vale ≈ 0. O R2 (×1 só no elo aberto da C) criou um desvio pago (a cauda desviada ficava em ×3,5) e um multiplicador inobservável, que o §6 rejeita. VERIFICADO 01/10.

IMPLEMENTADO e commitado 01/10: o K1b põe ×1 no CARREGAR em regime parado, em toda cadeia, e a C desviada fica sempre parada. O R2 saiu. Junto entraram o R1 (dx da laje só afasta) e as métricas K12.

**Why:** estátua e desvio pago ferem o §3 "Progressão na cadeia" e o §6.
**How to apply:** o resume saiu do `model_4650` com LR 5e-4. Depois da fase 4, ler `c_chegou_botar/c_episodio` e `c_desvio_*`.
**Detalhe:** docs/relatorios/2026-10-01-monotonia-do-curriculo.md

Relacionado: g1-limpo-fechar-paga-menos-que-pairar g1-limpo-botar-esquecido-na-fase-4 g1-limpo-cauda-andando-herda-twist-zero
