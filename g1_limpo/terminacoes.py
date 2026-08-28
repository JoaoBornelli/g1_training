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

As DUAS terminações próprias, e o que cada uma fecha:

    contato_ilegal   escorar o corpo na mesa
    caixa_largada    derrubar a caixa DEPOIS de tê-la pegado

A segunda entrou em 28/08 e ela é a metade que faltava do porteiro do `unload`: o
porteiro tira o pagamento de "derrubar sem pegar", e esta tira o de "pegar e largar".
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def caixa_largada(env: "ManagerBasedRlEnv", z_min: float,
                  dist_max: float) -> torch.Tensor:
    """A caixa caiu no chão, ou ela escapou das DUAS palmas.

    ⚠ ELA É ARMADA PELA PRIMEIRA PREENSÃO, e nunca antes. Sem a arma, todo episódio
    começaria terminando: no reset a caixa está na laje e as palmas estão longe, que é
    a condição de `escapou`. A arma é `env.limpo_pegou`, escrita pelo comando quando as
    duas palmas registram força pela primeira vez no episódio.

    ⚠ Ela FECHA o atalho que o porteiro do `unload` abre pela metade. Com o porteiro,
    derrubar a caixa deixa de pagar; com esta terminação, derrubar a caixa DEPOIS de
    tê-la pegado acaba o episódio. Os dois juntos cobrem "nunca pegou" e "pegou e
    largou". O `g1_poc` tem a mesma terminação e a mesma arma.

    ⚠ `escapou` exige as DUAS palmas longe (`all`), e não uma. Uma mão que solta para
    reposicionar é parte de uma pega, não o fim dela.
    """
    caixa = env.scene["box"].data.root_link_pos_w
    pegou = getattr(env, "limpo_pegou", None)
    if pegou is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=caixa.device)

    palmas = env.scene["robot"].data.site_pos_w[:, env.limpo_ids_palma, :]
    dist = torch.norm(palmas - caixa.unsqueeze(1), dim=-1)          # [B,2]
    # ⚠ o z é RELATIVO à origem do env: com `env_spacing` os envs não estão todos em
    # z = 0, e um limiar absoluto acusaria queda no env errado.
    caiu = (caixa[:, 2] - env.scene.env_origins[:, 2]) < z_min
    escapou = (dist > dist_max).all(dim=-1)
    return (caiu | escapou) & (pegou > 0.5)


def contato_ilegal(env: "ManagerBasedRlEnv", sensor_name: str,
                   limiar_N: float) -> torch.Tensor:
    """Uma parte do corpo que não deve escorar toca a mesa com força acima do limiar.

    ⚠ QUEM ESTÁ NA LISTA importa tanto quanto o limiar, e a lista de CORPO INTEIRO
    falhou por medição — ver o bloco de `cena.CORPOS_QUE_NAO_ESCORAM`. Em resumo: ela
    cobria punho e cotovelo, que TÊM de chegar perto do tampo para pegar, e não cobria
    os pads, que eram a superfície que escorava. O sinal estava invertido, e ~75% dos
    episódios de manipulação morriam na aproximação.

    ⚠ LIMIAR DE FORÇA, e não booleano. Roçar o tampo ao alcançar não termina; APOIAR o
    peso termina. Com um booleano, a lista tornaria pose baixa inganhável, que é a
    classe de erro do `botar_topo_piso` neste módulo.

    ⚠ 50 N é MEDIDO no `g1_poc` (`knobs.py:328`), onde a mesma terminação rodou. Não é
    escolha nova.

    ⚠ E há um registro do g1_poc que se aplica direto aqui: quando o movimento ficou
    caro (o `action_rate` degrau para −1,00), o `contato_ilegal` subiu de 6,4% para 17,5%
    das terminações — "com movimento caro, escorar o tronco na prateleira economiza
    esforço". O PESO do `action_rate_l2` no g1_limpo é −0,10 (o do fabricante), mas o
    `Episode_Reward` dele mede −2,3/s, a maior conta do conjunto. A mesma pressão
    existe, portanto ESPERE esta terminação disparar, e leia a fração dela antes de
    concluir que o robô piorou.

    ⚠ `amax` sobre os slots, e não `sum`: a pergunta é "algum ponto de contato passa de
    50 N", não "a soma de todos passa". Com `reduce="netforce"` e `num_slots=1` os dois
    coincidem hoje; o `amax` continua certo se alguém subir os slots.
    """
    f = env.scene[sensor_name].data.force
    assert f is not None, f"sensor '{sensor_name}' precisa do field 'force'."
    return torch.norm(f, dim=-1).amax(dim=-1) > limiar_N
