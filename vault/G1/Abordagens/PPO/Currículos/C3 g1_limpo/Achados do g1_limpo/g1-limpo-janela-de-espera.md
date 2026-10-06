---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "g1_limpo tem a janela de espera do g1_poc; decisão de 2026-09-02: ela passa a publicar ANDAR e o VALIDA sai da observação"
---
O `g1_limpo` tem a janela de espera desde 2026-09-02. Ela sorteia 0,3 a 1,0 s por episódio de manipulação e mantém o objetivo desligado. No `ANDAR` ela é zero. O rastreio de velocidade paga durante a janela; medido 3,816/s na janela e 0,000/s depois.

O ganho real foi o fecho de elo: `_fecha_elo_corrente` exige o objetivo ativo. Sem isso o `REORIENTAR` fechava no passo zero.

DECISÃO (2026-09-02, não implementada): durante a janela o one-hot publicado passa a ser `ANDAR`, e não `PEGAR`. O elo interno segue `PEGAR`. Tudo que a política e a recompensa leem usa o publicado; a mecânica usa o interno. Com isso o `VALIDA` vale `publicado ≠ ANDAR` e sai da observação.

**Why:** A observação durante a espera fica idêntica a um env `standing` da locomoção. A transição `ANDAR(v=0) → PEGAR` fica treinada em todo episódio de manipulação.
**How to apply:** O publicado é recalculado do interno a cada passo, nunca do próprio publicado. Ler o que se escreveu deixa o canal preso.

Relacionado: g1-limpo-contrato-de-troca-de-tarefa, g1-limpo-cadeia-da-pega
