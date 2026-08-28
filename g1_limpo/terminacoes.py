"""As terminações próprias do g1_limpo.

`time_out` e `fell_over` vêm do molde e não aparecem aqui. O molde traz um terceiro, o
`out_of_terrain_bounds`, que o `env_cfg` remove: o terreno é plano e a mobília tem pose
absoluta.

**Princípio: TERMINAR EM VEZ DE PENALIZAR.** Uma trajetória inválida acaba; ela não paga
multa. É o que a tarefa `tracking` do mjlab faz, e é o que o `g1_poc` adotou trocando
quatro penalidades por duas terminações (`g1_poc/terminacoes.py:8-14`).

⚠ A REESCRITA PERDEU O PRINCÍPIO JUNTO COM OS TERMOS. Até 27/08 o `g1_limpo` não tinha
terminação própria NENHUMA — só as duas do molde. E o freio do escoro entrou primeiro como
penalidade (`contato_prateleira = -1.5`), que rodou 405 iterações do bloco 2 e mostrou o
problema da forma: o contato do tronco caiu monotonicamente (7,5% -> 3,8% -> 2,0% dos
passos) e a manipulação caiu com ele (`staged` 0,36 -> 0,17), porque uma multa que o robô
pode pagar é uma multa que ele orça. A terminação não é orçável.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def contato_ilegal(env: "ManagerBasedRlEnv", sensor_name: str,
                   limiar_N: float) -> torch.Tensor:
    """Qualquer parte do corpo toca a mesa com força acima do limiar.

    ⚠ LIMIAR DE FORÇA, e não booleano — e é ele que torna a lista de CORPO INTEIRO
    segura. Roçar o tampo ao alcançar não termina; APOIAR o peso termina. Com um
    booleano, cobrir as 33 geoms tornaria pose baixa inganhável, que é a classe de erro
    do `botar_topo_piso` neste módulo.

    ⚠ 50 N é MEDIDO no `g1_poc` (`knobs.py:328`), onde a mesma terminação rodou. Não é
    escolha nova.

    ⚠ E há um registro do g1_poc que se aplica direto aqui: quando o movimento ficou
    caro (o `action_rate` degrau para −1,00), o `contato_ilegal` subiu de 6,4% para 17,5%
    das terminações — "com movimento caro, escorar o tronco na prateleira economiza
    esforço". O `action_rate_l2` do g1_limpo está em −2,04, a maior penalidade do
    conjunto. A mesma pressão existe, portanto ESPERE esta terminação disparar bastante
    no começo, e leia a fração dela antes de concluir que o robô piorou.

    ⚠ `amax` sobre os slots, e não `sum`: a pergunta é "algum ponto de contato passa de
    50 N", não "a soma de todos passa". Com `reduce="netforce"` e `num_slots=1` os dois
    coincidem hoje; o `amax` continua certo se alguém subir os slots.
    """
    f = env.scene[sensor_name].data.force
    assert f is not None, f"sensor '{sensor_name}' precisa do field 'force'."
    return torch.norm(f, dim=-1).amax(dim=-1) > limiar_N
