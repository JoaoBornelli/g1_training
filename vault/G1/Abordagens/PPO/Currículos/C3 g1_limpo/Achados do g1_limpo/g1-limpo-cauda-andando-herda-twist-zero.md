---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "CÓDIGO 01/10: a cauda CARREGAR que anda herda vx = vy = 0 do regime parado por ~2,9 s, e o rastreio ×3,5 paga a estátua nesse trecho"
---
CÓDIGO 01/10 (verificador do workflow): `_zera_twist_nos_parados` escreve vx = vy = 0 nos envs parados (comando.py:1501-1502), e nada re-sorteia o twist quando o env passa de parado a andando. A cauda CARREGAR andando (fase ≥ 3, e 10% na fase 2) fica ~2,9 s com comando linear zero, até o re-sorteio do fabricante (segmentos de 3 a 8 s). Nesse trecho o rastreio ×3,5 paga a estátua.

**Why:** tira prática de andar com a caixa e agrava o desvio pago da cadeia C. O smoke não detecta, porque confere o máximo sobre todos os envs.
**How to apply:** conserto candidato K13, re-sortear o twist na entrada da cauda andando; ainda não avaliado. Medir a fração com o K12.
**Detalhe:** docs/relatorios/2026-10-01-monotonia-do-curriculo.md §2.3

Relacionado: g1-limpo-carregar-parado-e-estatua mjlab-curriculo-de-comando-sobrescreve-a-faixa
