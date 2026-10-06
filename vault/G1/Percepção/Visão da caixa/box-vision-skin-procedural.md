---
tipo: achado
pai: "[[Visão da caixa]]"
estado: ativa
verificado: 2026-10-06
resumo: "g1-box-vision: skin procedural por amostra, 6 faces independentes, tag = único código de barras pintado na textura; retreinada 2026-09-25: orientação 7,6° mediana; cabeça de tag ~na linha de base"
---
Implementado no `main.py` e retreinado em 2026-09-25: cada amostra gera uma caixa nova (papelão, impressão, fita, etiqueta só com texto, desgaste), seis faces independentes giradas em 90° ao acaso, e mesa em cinco famílias. Há um código de barras por caixa, e ele é a tag, pintada na textura. O rótulo sai pelo mapa cube map medido (sem espelho; convenção OpenGL). Tag visível só com a face a menos de 65° da câmera.

**Why:** caixa uniforme não generaliza para foto real; Tobin 2017 mede queda forte com menos de ~1000 texturas únicas, então pool pequeno repetido não serve.
Resultado: orientação mediana 7,6°, p90 27,7° (lisa 6,4°/21,0°). A cabeça de tag deu 84,0% com base de ~82% (18% positivos): quase não aprendeu.

**How to apply:** avalie a tag por recall/precisão, não por acurácia. Troca de textura por `tex_data` + `mjr_uploadTexture` usa o `_mjr_context` privado, com fallback de recriar o `Renderer` com `close()`.
**Detalhe:** `g1-box-vision/CLAUDE.md` (Skin procedural)

Relacionado: box-vision-desenho-face-da-tag box-vision-memoria-das-faces
