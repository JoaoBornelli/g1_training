---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o robô mergulha atrás da caixa caída porque perder a caixa custa o mesmo que cair; v3 02/10: a perda troca para ANDAR parado (teste com volta atrás), trecho pobre"
cluster: c3
---
O mergulho atrás da caixa caída é resposta ÓTIMA: o elo não muda quando a caixa cai, `precise_pos` segue pagando, e `caixa_largada` termina com o mesmo custo da queda.

DECISÃO DO DONO 21/09: o laço pega→cai→aborta→reposiciona→pega fica no controlador externo. Falta treinar UMA transição: receber ANDAR parado em qualquer instante e parar de pé. No ANDAR os canais da caixa vão a zero.
⚠ Aborto que paga faz derrubar virar lucro: o trecho depois da perda é POBRE.

v3 (dono, 02/10): a perda troca para ANDAR parado, com chave para voltar à terminação. A volta à pose de pé da IK paga como potencial, uma vez; custo fixo da perda maior que esse ganho; queda custa mais que perda.

**Why:** cabeça perfeita num estado sem comportamento treinado não melhora nada.
**How to apply:** o comportamento antes do estimador.
**Detalhe:** `docs/planos/2026-09-21-aborto-por-perda-de-preensao.md`; `specs/g1-v3-comportamento.md` §10.1

Relacionado: g1-limpo-piso-da-estatua-travava-a-pega, g1-limpo-v3-alvos-macro
