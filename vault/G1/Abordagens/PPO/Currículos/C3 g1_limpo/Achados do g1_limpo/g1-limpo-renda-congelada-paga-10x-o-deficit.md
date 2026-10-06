---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "A renda_congelada ACUMULA por fecho; o desperdicio e o empilhamento, nao o fecho isolado. Sobrepaga 10,3x na cadeia B e so 1,2x por fecho na C"
---
MEDIDO 2026-09-10/11 em `model_10200`. Cadeia B: 25 852 amostras, 32 fechos. Cadeia C: 192 000 amostras, 215+135 fechos.

`renda_congelada` **soma** a renda de cada fecho e paga ate o fim do episodio.

| | cadeia B (15% dos passos) | cadeia C (45%) |
|---|---|---|
| deficit real na cauda | 0,68/s | **6,6/s** |
| sobrepagamento **marginal** | 10,3× | **1,22×** |
| piso na cauda | 13,79/s | **21,76/s** (dois fechos empilhados) |

⚠ **O 10,3× e da cadeia B e NAO generaliza.** Por fecho a renda e quase honesta na C. O desperdicio ali e o EMPILHAMENTO: dois fechos somam 21,76/s enquanto o fecho mais recente vale 7,92/s.

**A forma proposta:** `congelado = max(0, M − soma_viva)`, com `M` = maximo corrente da renda total de manipulacao. Ela e uma catraca: a renda nunca cai, e nunca empilha. Da 13,79/s nas duas caudas contra 27,0 (B) e 23,3 (C).

O unico buraco legitimo e a janela de espera do `PEGAR`: 1,02 s com soma viva **exatamente zero** em 100% de 10 703 amostras.

Relacionado: g1-limpo-cauda-carregar-piso-sem-derivada, g1-limpo-nao-anda-com-a-caixa, g1-limpo-botar-paga-nao-botar
