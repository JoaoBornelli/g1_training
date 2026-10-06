---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: rejeitada
verificado: 2026-10-06
resumo: "entrega de tarefa por evento no viewer foi REJEITADA; a task de entrega é PEGAR com espera alongada"
cluster: c3
---
A task `Mjlab-G1-Limpo-Entrega` do `g1_limpo` é `PEGAR` forçado com `espera_s` alongada. Verificado 2026-09-02.

Entrega por evento REJEITADA (2026-09-02): um evento trocava o elo `ANDAR → PEGAR` ao vivo, repunha a mobília e zerava o twist. O robô sambava. Duas causas. Primeira: `elo = ANDAR` com a caixa a 0,6 m não existe no treino; a caixa sempre está a 5 m no `ANDAR`. Segunda: a caixa teleportava 4,5 m em um passo de observação.

Duas lições ficam. O `reset_base_por_elo` escolhe a faixa pelo elo do currículo; quem precisa de outra faixa tem de ser conhecido antes dos eventos. Um evento de intervalo não roda no `reset()`; zerar o twist nele deixa a primeira observação com comando de até 2 m/s.

**Why:** A janela de espera já produz "parado, sem tarefa, depois pegar". Não precisa de evento.
**How to apply:** Para olhar a transição, alongue `espera_s` no cfg de play. Não crie evento de troca de elo.

Relacionado: g1-limpo-contrato-de-troca-de-tarefa, g1-limpo-janela-de-espera
