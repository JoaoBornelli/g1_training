---
tipo: achado
pai: "[[Visão da caixa]]"
estado: ativa
verificado: 2026-10-06
resumo: "No g1-box-vision o RANSAC de planos no depth dá o frame da caixa sem erro grosseiro mesmo amassada; o que quebra é o rótulo do eixo por tamanho"
cluster: percepcao
---
Medido 2026-09-10 no worktree `orca/workspaces/box-vision/g1-box-vision`. RANSAC de
planos no depth recupera o frame da caixa. Sob degradação (ruído estéreo até 1%,
amassado até 8 mm) o erro de FRAME, módulo as 24 rotações, NUNCA passou de 20°;
pior p90 = 13,15°. Sem ruído, 0,08°. A fronteira que o user precisa é 45°.

Quebra só o RÓTULO do eixo. Ordenar os eixos pela aresta medida inverte quando o
erro de aresta (7 a 15 mm sob degradação) se aproxima da diferença entre duas
arestas. Uma permutação vale 90°.

**Why:** Caixa real é amassada e de dimensão desconhecida. Grupo de simetria: cubo
24, paralelepípedo distinto 4.

**How to apply:** CORRIGIDO 2026-09-17: o depth SAIU do escopo, então este caminho
está parado e a rede é a hipótese. Guarde o número para o dia em que o depth voltar.

**Detalhe:** sonda em `scratchpad/sonda_ransac.py`

Relacionado: g1-box-vision-redesenho g1-vision-cube-orientation
