---
tipo: pendencia
pai: "[[Pendências do C3]]"
estado: aberta
verificado: 2026-10-06
periodo: 05/10/26
resumo: "Com s_C ≥ 0,50 a laje do BOTAR sai uniforme de 0,04 a 0,64 m e o R4b não dá mais fatia à laje que falha."
cluster: c3
---
Com s_C ≥ 0,50 a laje do BOTAR sai uniforme de 0,04 a 0,64 m e o R4b não dá mais fatia à laje que falha.

Marcado pelo dono como 'A DISCUTIR MELHOR' (REL-APR §8 item 6).

**Como funciona hoje:** o topo da laje do BOTAR é `teto_raso − prof·U`, com `prof` crescendo de 0,10 m (`s_C` = 0) até o chão (`s_C` ≥ 0,50). Com `s_C` ≥ 0,50 a laje sai uniforme de 0,04 a 0,64 m (média 0,34).

**Problema:** o `s_C` estabiliza em ~0,50 no refino do 23250, e daí em diante o R4b não dá mais fatia à altura que falha; a laje baixa continua com a mesma probabilidade que a alta. Logo depois de cada resume as EMAs caem por ~1 episódio e o R4b encolhe a laje nesse vale.

**Próximo passo:** sorteio da altura ponderado pelo sucesso por faixa (estilo PLR do Lift Box); decisão do dono pendente.
