---
tipo: abordagem
pai: "[[Percepção]]"
estado: ativa
verificado: 2026-10-06
resumo: "Orientação da caixa por RGB: duas CNNs, memória das faces e HMM (85% com ORB); o RANSAC de planos no depth dá o frame da caixa sem a rede."
cluster: percepcao
---
## Hipótese
A orientação da caixa sai de uma imagem RGB por rede neural, sem marcador, para o robô real (onde não há oráculo).

## Método
Duas linhas, em ordem: `Vision_training/` (jul/26: CNN regride a orientação de um cubo com loss simétrico octaédrico) e o repo `g1-box-vision`, refeito do zero em 09/26 (o dono proibiu ancorar no `box_scene.py` de julho): duas CNNs em série estimam a face da tag e a orientação só por RGB; memória das 6 faces com 4 estados (ver a tag encerra a busca); skin procedural por amostra com a tag pintada na textura; continuidade entre quadros por HMM sobre as 24 respostas.

## Resultado
- Continuidade: HMM só com transição acerta 70%; com aparência (ORB), 85%.
- Retreinada com skin procedural, a tag ficou perto da linha de base.
- **O RANSAC de planos no depth dá o frame da caixa sem erro grosseiro, mesmo amassada**: a rede deixa de ser necessária para o frame; ela serve só para dizer qual face é qual.

## Por que parou
Ativa como estudo; a v3 treina com estimador simulado da caixa, e o caminho para o real é câmera sem marcador + depth.

## O que herda
- `pupil-apriltags`: destruir um Detector corrompe o heap; um detector só por processo.
- Câmera D435i fixa a 47,6°: vê o chão até 2,4 m.

## Filhos
<!-- filhos:inicio -->
- **box-vision-continuidade-por-aparencia** (achado · ativa) — g1-box-vision: continuidade da orientação entre quadros — HMM sobre as 24 respostas é o melhor (70% só transição, 85% com ORB/SIFT, 78% com histograma); rejeição de 45° é a pior peça (medido 2026-09-28)
- **box-vision-desenho-face-da-tag** (achado · ativa) — g1-box-vision etapa 1 travada 2026-09-17: duas CNNs em série estimam orientação da caixa só por RGB
- **box-vision-memoria-das-faces** (achado · ativa) — g1-box-vision etapa 3: vetor de 6 faces com 4 estados; ver a tag encerra a busca, dedução vira provável
- **box-vision-ransac-dispensa-a-rede** (achado · ativa) — No g1-box-vision o RANSAC de planos no depth dá o frame da caixa sem erro grosseiro mesmo amassada; o que quebra é o rótulo do eixo por tamanho
- **box-vision-skin-procedural** (achado · ativa) — g1-box-vision: skin procedural por amostra, 6 faces independentes, tag = único código de barras pintado na textura; retreinada 2026-09-25: orientação 7,6° mediana; cabeça de tag ~na linha de base
- **g1-box-vision-redesenho** (achado · ativa) — Repo g1-box-vision refaz visão caixa→orientação do zero; user proíbe ancorar no box_scene.py de julho
- **g1-vision-cube-orientation** (achado · ativa) — Experimento de visão (Vision_training/) — imagem RGB do cubo → CNN regride orientação (gizmo); loss simétrico octaédrico; tag adiada
- **pupil-apriltags-del-corrompe-heap** (achado · ativa) — pupil-apriltags 1.0.4: destruir um Detector corrompe o heap (famílias destruídas antes do detector); use um detector só por processo
<!-- filhos:fim -->
