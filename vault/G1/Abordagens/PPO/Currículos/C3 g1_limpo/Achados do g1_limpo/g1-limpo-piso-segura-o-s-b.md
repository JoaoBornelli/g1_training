---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "MEDIDO 30/09 por simulação: o piso uniforme de 20% do nível segura o sucesso abaixo de 0,5 até o nível médio chegar perto de 3; o portão s_B ≥ 0,50 da fase 2 exige a pega no nível 3"
cluster: c3
---
MEDIDO 30/09 (simulação do `curriculo.nivel`: passeio ±1, depois o piso uniforme de 20% em todo reset): o sucesso estacionário é ≈ 0,5 − 0,16 × (3 − nível médio). O piso manda envs acima do equilíbrio, e o passeio compensa descendo. O `s_B ≥ 0,50` da troca 1 → 2 só chega com o nível médio perto de 3: a pega sobrevive em 50% no nível 3 (laje 0,15–0,55 m, até 4 kg).

Nível por tarefa REJEITADO (30/09, dono): o BOTAR vem sempre depois de um PEGAR na mesma caixa, e o nível único fica.

**Why:** a spec de cadeia §5 e eu dissemos "perto de 0,5 depois do nível 0". Com o piso, isso é falso.
**How to apply:** prever a fase 2 pelo `Curriculum/nivel`, e não pela reta do `s_B`.

Relacionado: g1-limpo-curriculo-de-cadeia
