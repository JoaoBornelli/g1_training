---
tipo: achado
pai: "[[Visão da caixa]]"
estado: ativa
verificado: 2026-10-06
resumo: "pupil-apriltags 1.0.4: destruir um Detector corrompe o heap (famílias destruídas antes do detector); use um detector só por processo"
cluster: percepcao
---
O `pupil-apriltags` 1.0.4.post11 destrói as famílias de tag antes do detector no `Detector.__del__` (`bindings.py:419-449`). O `apriltag_detector_destroy` depois mexe na memória liberada. Medido em 2026-10-01: destruir um detector e criar outro aborta o processo na terceira vez, com `malloc(): mismatching next->prev_size`. Num pytest, o aborto aparece num teste qualquer, longe da causa.

**Why:** o erro só aparece quando o coletor de lixo roda o `__del__`; o mesmo código passa isolado e aborta na suíte.
**How to apply:** crie UM detector por processo e nunca o destrua. Na saída, `atexit.register(setattr, det, "tag_detector_ptr", None)` impede o destrutor. Feito em `percepcao/tags.py` (worktree `lidar_sim`).

Relacionado: g1-slam-percepcao-desenho box-vision-desenho-face-da-tag
