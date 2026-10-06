---
tipo: abordagem
pai: "[[Percepção]]"
estado: ativa
verificado: 2026-10-06
resumo: "RANSAC dispensa a rede."
cluster: percepcao
---
## Hipótese
CNN em RGB dá a orientação da caixa.

## Método
Vision_training/, face da tag, skin procedural.

## Resultado
RANSAC dispensa a rede.

## Por que parou


## O que herda

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
