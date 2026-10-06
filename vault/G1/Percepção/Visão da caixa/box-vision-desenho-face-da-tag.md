---
tipo: achado
pai: "[[Visão da caixa]]"
estado: ativa
verificado: 2026-10-06
resumo: "g1-box-vision etapa 1 travada 2026-09-17: duas CNNs em série estimam orientação da caixa só por RGB"
---
A hipótese: **uma rede leve infere a orientação da caixa a partir de RGB**. Etapas:
simulador → webcam → G1. Duas CNNs em série: detector (regressa `u,v,lado`) →
orientador (recorte 96×96, saída 6D).

**HIPÓTESE VALIDADA.** Estado em 2026-09-20, com 8000 amostras: detector contém a
caixa em 97,8%; orientador com mediana **5,3°**, p90 15,3° e 0,2% acima de 45°. O
orçamento era 45°.

Travado: só RGB; uma caixa; CUBO, logo perda = mínimo sobre as 24 rotações; rótulo
relativo ao RAIO, não à câmera.

**How to apply:** O teste ao vivo com câmera FIXA acha o que o agregado esconde:
ele revelou que a caixa nunca caía na borda do quadro no treino, e lá o detector
perdia 3 de 4. Rode `ao_vivo.py` antes de confiar numa mediana. Não reabra o
caminho geométrico: ele precisava de depth, que saiu do escopo.

**Detalhe:** `g1-box-vision/CLAUDE.md`

Relacionado: box-vision-ransac-dispensa-a-rede g1-box-vision-redesenho box-vision-skin-procedural
