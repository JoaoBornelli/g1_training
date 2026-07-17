"""Fine-tune 2026-07-16: REHEARSAL anti-esquecimento (mistura ficar-de-pé no treino).

Motivo (user 07-16): testou o checkpoint da lift no Stand-Step e ele DESAPRENDEU
a ficar de pé — esquecimento catastrófico entre skills (o fine-tuning da lift
corroeu o balanço puro). Antes de baixar a mesa (currículo de altura, que exige
agachar com equilíbrio sólido), restaurar E manter o ficar-de-pé.

Mecanismo (warm-start do generalize): `rehearsal_fraction=0.2` → 20% dos envs
spawnam a caixa a 5m (fora de alcance, no chão). Nesses envs os rewards de tarefa
zeram sozinhos (nada pra pegar) e só a fundação (upright/postura/slip) + o push
ficam ativos → praticam ficar de pé/recuperar empurrão A CADA batch, junto com a
lift nos outros 80%. Sem ping-pong (não troca grasp por stand), sem mascarar
reward (a caixa longe faz o trabalho), e a obs (box_pos_b longe) ensina a
política a condicionar "sem caixa = fico de pé". Dura pelo currículo de altura.

CAIXA E MESA vão juntas pra longe (mesma máscara por-env, evento combinado
reset_scene_with_rehearsal) → cena LIMPA de só ficar de pé: nada na frente pra
alcançar nem esbarrar num passo protetivo.
"""
from dataclasses import replace

from ..knobs import LiftKnobs
from .c2026_07_16_generalize import KNOBS as _PREV

KNOBS = replace(_PREV,
    scene=replace(_PREV.scene, rehearsal_fraction=0.2, rehearsal_far_x=5.0),
)
