"""Baseline do multi-tarefa — §14 do doc, **já passado pela calibração**.

`KNOBS = MultitaskKnobs()` sem nenhum override, e isso é RESULTADO, não omissão: a
calibração da Tarefa 12 (`python g1_multitask/calibra.py`) rodou sobre estes valores
e a conclusão foi manter todos. Cada decisão está escrita ao lado do número no
`knobs.py` — "conferido, mantido" com o número na mão é resposta válida.

O que a calibração mudou de verdade foi UM número, e ele já entrou aqui pelo default:
o `orienta_face` ganhou uma **segunda escala** (`angulo_std_grosso_deg = 30.0`),
porque com escala única de 5° o nível 15° do eixo de giro daria `exp(−(15/5)²) =
1.2e−4` — gradiente nenhum até o robô já estar dentro de ~10°. Com as duas escalas o
termo vale **0.457** na tolerância de 10° e **0.389** no reset do nível 15°, ambos
medidos.

Os 3 achados que ficaram em observação, e não em mudança:
  - `box_shake` supera o sinal de tarefa nas 3 tarefas que carregam (razão 2× a 62×),
    mas a medição é com ação nula, que é o pior caso de ω. Ver `Reward.box_shake`.
  - `table_contact` idem, e ali o flag CONFIRMA que o termo funciona.
  - `sustain_std` vale 0.018 na tolerância de sucesso, e fica: nenhuma tarefa precisa
    fechar aquele vão usando só esse termo.

Manter este arquivo intacto de agora em diante: comparar dois treinos é `git diff`
entre dois configs, e pra isso o baseline precisa parar de mudar.
"""
from __future__ import annotations

from ..knobs import MultitaskKnobs

KNOBS = MultitaskKnobs()
