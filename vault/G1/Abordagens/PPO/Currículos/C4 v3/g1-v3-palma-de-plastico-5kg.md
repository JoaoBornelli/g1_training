---
tipo: decisao
pai: "[[C4 v3]]"
estado: ativa
verificado: 2026-10-06
resumo: "DECISÃO DO DONO 05/10: o robô real fica com a palma de plástico (μ 0,3–0,5 contra papelão), a massa máxima continua 5 kg, e o robô tem de aprender outro jeito de segurar quando o atrito lateral não basta; a recompensa da v3 paga o resultado, não o aperto lateral"
---
DECISÃO DO DONO (05/10): palma de plástico (μ 0,3–0,5 contra papelão), massa até 5 kg. Com μ 0,3, 5 kg pedem 82 N por palma e o braço entrega ~100 N; o robô pode apoiar a caixa nivelada no tronco (enunciado §3). REJEITADO pelo dono (05/10): caixa inclinada contra o peito e apoio no antebraço; a caixa fica a 15° no fim do treino em toda etapa, porque o conteúdo não pode sofrer impacto.

**Why:** sem acesso ao robô real não há como medir μ nem trocar o material.
**How to apply:** v3 paga o RESULTADO da pega (a caixa fica com o robô e acompanha as mãos), não a força normal nos pads; o freio de aperto é teto sobre a força total; DR de μ por `priority` da caixa em [0,3; 0,6], por nível, DEPOIS da recompensa do resultado.
**Detalhe:** docs/relatorios/2026-10-05-dr-de-atrito-pega-e-transferencia.md §6

Relacionado: g1-limpo-aperto-fixo-nao-acompanha-massa g1-limpo-v3-alvos-macro g1-limpo-caixa-presa-pelo-pulso
