---
tipo: achado
pai: "[[Visão da caixa]]"
estado: ativa
verificado: 2026-10-06
resumo: "Repo g1-box-vision refaz visão caixa→orientação do zero; user proíbe ancorar no box_scene.py de julho"
---
O worktree `orca/workspaces/box-vision/g1-box-vision` refaz a percepção da caixa do
zero (verificado 2026-09-03). O escopo novo tem duas partes: detectar a caixa E
estimar a orientação da face com a tag.

O experimento de julho em `unitree-g1-sim/Vision_training/box_scene.py` continua no
disco e fechou com 2,1°. O user PROÍBE usá-lo como âncora do desenho novo. Motivo
declarado: "não quero viés".

**Why:** O user projeta de primeiros princípios. A solução de julho carrega suposições
embutidas: o recorte vem da verdade do simulador, a saída é 6D, a loss é octaédrica.

**How to apply:** Discuta o desenho sem citar `box_scene.py` como resposta. Leia
aquele código só se o user pedir.

Relacionado: g1-vision-cube-orientation feedback-como-trabalhar
