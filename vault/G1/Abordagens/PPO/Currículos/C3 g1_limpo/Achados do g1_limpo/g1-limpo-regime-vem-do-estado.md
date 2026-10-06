---
tipo: achado
pai: "[[Achados do g1_limpo]]"
estado: ativa
verificado: 2026-10-06
resumo: "o laço de rumo tirava a manipulação do regime standing e afrouxava o freio de velocidade em 2,7x; corrigido em 21/09 mas ainda não treinado"
---
`velocidade_por_regime` escolhia a tabela de limite pelo COMANDO, e o laço de rumo de
17/09 punha a manipulação no regime `walking`, 2,7× mais frouxo, em 31,3% dos passos da
pega: 58% do freio perdido.

CORRIGIDO 21/09, duas mudanças que se MULTIPLICAM: o regime vem de
`env.limpo_twist_zerado`, e o peso foi de −2,0 para −15,0. O regime escolhe a TABELA, o
peso o CUSTO. Motivo: SEGURANÇA DE PESSOAS. A mão chega a 2,58 m/s na pega, e a ISO/TS
15066 usa 0,25. A rajada passa a custar 8,64 contra 6,9 que a pressa compra. O limite NÃO mudou: 91% da pega já custa zero.

⚠ ESCRITO E NÃO TREINADO; é WARM-START, baixar a LR para 5e-4.
⚠ ABERTO: `pose` lê o regime pelo mesmo caminho.

**Why:** um gate por comando mede intenção de viajar, e correção de rumo não é isso.
**How to apply:** docstring que declara premissa não prova que ela ainda vale.
**Detalhe:** `Obsidian-documents/Robotics/G1-Limpo-Status.md`

Relacionado: g1-limpo-deriva-de-rumo-com-a-caixa
