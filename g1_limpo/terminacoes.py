"""As terminações próprias do g1_limpo.

`time_out` e `fell_over` vêm do molde e não aparecem aqui. O molde traz um terceiro, o
`out_of_terrain_bounds`, que o `env_cfg` remove: o terreno é plano e a mobília tem pose
absoluta.

**Princípio: TERMINAR SÓ O QUE NÃO TEM COMO SER PAGO.** É a forma REVISADA, em 01/09, do
"terminar em vez de penalizar" que o `g1_poc` adotou. A revisão vem de medição, e ela
distingue dois casos que antes eram tratados como um:

  · **encostar na mesa TEM como ser pago** — o robô alivia o peso e segue a tarefa.
    Terminar ali mata a exploração antes de ela refinar a pega. Virou multa em rampa.
  · **largar a caixa NÃO tem** — com ela no chão a tarefa acabou. Continua terminação.

⚠ E o argumento "multa que o robô pode pagar é multa que ele ORÇA" NÃO foi abandonado —
ele foi medido e não se realizou. A previsão era `contato_tronco` cada vez mais negativo
com `staged` parado; o medido foi `contato_tronco` em −0,09 (7% da conta do
`action_rate`) com `staged` DOBRANDO. O risco existe e o discriminador fica registrado.

A ÚNICA terminação própria é a `caixa_largada`: derrubar a caixa depois de tê-la
pegado. Ela é a metade que faltava do porteiro do `unload` — o porteiro tira o pagamento
de "derrubar sem pegar", e ela tira o de "pegar e largar".

⚠ O CONTATO COM A MESA SAIU DAQUI em 01/09 e virou MULTA (`recompensas.contato_mesa`).
Não é abandono do princípio: largar a caixa não tem como ser pago — com ela no chão a
tarefa acabou —, enquanto encostar na mesa tem. E a medição decidiu: com a terminação,
76% dos episódios de manipulação morriam na mesa, e o `play` mostrou que a ação MÉDIA
nem se aproximava, portanto aqueles 76% eram RUÍDO de exploração. A terminação matava a
exploração antes de ela refinar a pega. Depois da troca, `descarga` foi de 0,0 a 0,994.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def caixa_largada(env: "ManagerBasedRlEnv", folga_chao: float,
                  dist_max: float, meia_aresta_ref: float) -> torch.Tensor:
    """A caixa caiu no chão, ou ela escapou das DUAS palmas depois de pega.

    ⚠ `caiu` NÃO EXIGE A ARMA (v2.1, spec P7). Até 04/09 as duas condições dependiam
    de `pegou`, e uma caixa derrubada da mesa ANTES da primeira preensão não terminava:
    o env ficava morto até o time_out, pagando ~2/s sem aprender nada por até 18 s.
    `caiu` no RESET continua falso mesmo sem a arma: o fundo da caixa fica a
    `topo ≥ 0,04 > folga_chao = 0,02` do chão.

    ⚠ `escapou` CONTINUA ARMADO pela primeira preensão, e nunca antes. A arma é
    `env.limpo_pegou`, escrita pelo comando quando as duas palmas registram força pela
    primeira vez no episódio. Sem ela, todo episódio começaria terminando por
    `escapou`: no reset a caixa está na laje e as palmas estão longe.

    ⚠ Ela FECHA o atalho que o porteiro do `unload` abre pela metade. Com o porteiro,
    derrubar a caixa deixa de pagar; com esta terminação, derrubar a caixa DEPOIS de
    tê-la pegado acaba o episódio. Os dois juntos cobrem "nunca pegou" e "pegou e
    largou". O `g1_poc` tem a mesma terminação e a mesma arma.

    ⚠ `escapou` exige as DUAS palmas longe (`all`), e não uma. Uma mão que solta para
    reposicionar é parte de uma pega, não o fim dela.

    ⚠ Limitação declarada: uma caixa que cai TOMBADA sobre uma aresta tem o centro em
    a·√2 e escapa ao caiu; fora da espera final o escapou a pega.
    """
    caixa = env.scene["box"].data.root_link_pos_w
    # ⚠ o z é RELATIVO à origem do env: com `env_spacing` os envs não estão todos em
    # z = 0, e um limiar absoluto acusaria queda no env errado.
    # ⚠ POR TAMANHO (spec §6.7): "o fundo da caixa está a menos de `folga_chao` do chão".
    # Com o limiar fixo de 0,10 a caixa de 0,13 m deitada no chão nunca acusava queda.
    meia = getattr(env, "limpo_meia_aresta", None)
    meia_z = meia[:, 2] if meia is not None else torch.full_like(caixa[:, 2], meia_aresta_ref)
    caiu = (caixa[:, 2] - env.scene.env_origins[:, 2] - meia_z) < folga_chao

    pegou = getattr(env, "limpo_pegou", None)
    if pegou is None:
        return caiu

    palmas = env.scene["robot"].data.site_pos_w[:, env.limpo_ids_palma, :]
    dist = torch.norm(palmas - caixa.unsqueeze(1), dim=-1)          # [B,2]
    escapou = (dist > dist_max).all(dim=-1)
    # ⚠ O GUARDA DA ESPERA FINAL (spec §6.6.3): depois do fecho do BOTAR as mãos TÊM de
    # sair da caixa — `escapou` dispararia por fazer a coisa certa. `caiu` continua
    # armado: largar é permitido, derrubar não.
    soltou = getattr(env, "limpo_soltou", None)
    if soltou is not None:
        escapou = escapou & (soltou < 0.5)
    return caiu | (escapou & (pegou > 0.5))
