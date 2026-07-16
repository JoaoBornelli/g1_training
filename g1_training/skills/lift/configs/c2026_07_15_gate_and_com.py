"""Fine-tune 2026-07-15: gate de orientação da caixa endurecido pra ~10°
(fecha o hack de tombar a caixa pra ganhar altura de centro de graça) + reward
de CoM-sobre-os-pés (pune o corpo inteiro derivando pra frente, escorado na
caixa). ARQUIVADO — números exatos do `src/G1_lifting_box` antes da mudança
de geometria de 07-16 (caixa ainda no centro da mesa). Mantido pra referência/
revert; não é o config ativo."""
from ..knobs import LiftKnobs

KNOBS = LiftKnobs()  # todos os defaults de knobs.py SÃO esses números (box_xy=(0.50,0.0))
