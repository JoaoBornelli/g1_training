"""As terminações próprias do g1_limpo.

`time_out` vem do molde e não aparece aqui. O molde traz um terceiro, o
`out_of_terrain_bounds`, que o `env_cfg` remove: o terreno é plano e a mobília tem pose
absoluta.

⚠ `fell_over` GANHA UMA CLÁUSULA (spec `g1-limpo-dois-bits.md` §3.2): o MESMO slot
`cfg.terminations["fell_over"]` passa a chamar `caiu`, deste módulo, que faz o
`bad_orientation` do molde (70°, intocado) E acrescenta o joelho no chão — o caso
que a orientação sozinha não vê: o robô agacha de LADO, sem tombar, e o joelho
encosta.

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

from mjlab.envs.mdp import bad_orientation
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["caixa_largada", "Caiu"]


def caixa_largada(env: "ManagerBasedRlEnv", folga_chao: float, v_solta: float,
                  raio_solta: float, meia_aresta_ref: float,
                  nome_do_comando: str = "alvo_caixa") -> torch.Tensor:
    """A caixa caiu no chão, ou foi solta/atirada fora do alvo depois de pega.

    ⚠⚠ REESCRITA v3.2 (spec `g1-limpo-soltar-termina.md` §2). A cláusula antiga
    (`escapou`, distância das palmas > 0,45 m) não pegava o arremesso curto: a caixa
    pousa no alvo antes de se afastar 0,45 m da mão. `soltou_fora` troca a distância
    pela VELOCIDADE relativa da caixa à base — caixa rápida e fora do alvo foi solta,
    perto ou longe da mão.

    ⚠ `caiu` NÃO EXIGE A ARMA (v2.1, spec P7): sem ela, derrubar a caixa ANTES da
    primeira preensão não terminava, e o env ficava morto até o `time_out`.

    ⚠ `soltou_fora` CONTINUA ARMADO só pela primeira preensão (`env.limpo_pegou`),
    e DESARMA na espera final (`env.limpo_soltou`): depois do fecho do `BOTAR` as
    mãos TÊM de sair, e isso não pode terminar o episódio.

    ⚠⚠ `apos_pegar` GUARDA a janela entre o TOQUE (arma `pegou`) e o FECHO do `PEGAR`:
    a caixa ainda está na laje ali, e um tropeço ou push forte na base não pode matar
    a pega. `elo != PEGAR` não bastaria: a espera depois do `REORIENTAR` fechar também
    toca a caixa sem o `PEGAR` ter fechado.
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

    # ⚠ `alvo` é `_command[:, ALVO]`, o MESMO ponto que `recompensas._alvo` lê — uma
    # fonte só para "onde a caixa deve ficar".
    from g1_limpo.comando import ALVO, BOTAR, CARREGAR, PEGAR
    alvo = env.command_manager.get_command(nome_do_comando)[:, ALVO]
    t = env.command_manager.get_term(nome_do_comando)
    v_caixa = env.scene["box"].data.root_link_lin_vel_w
    v_base = env.scene["robot"].data.root_link_lin_vel_w
    v_rel = torch.norm(v_caixa - v_base, dim=-1)
    d_alvo = torch.norm(caixa - alvo, dim=-1)
    no_alvo = d_alvo <= raio_solta          # raio_solta = tarefa.precise_pos_sigma, REUSO
    soltou_fora = (v_rel > v_solta) & ~no_alvo

    # ⚠ hold (PEGAR fechado), CARREGAR, BOTAR — não "elo != PEGAR" (ver docstring).
    apos_pegar = (t._elo == CARREGAR) | (t._elo == BOTAR) | ((t._elo == PEGAR) & t.fechou)

    soltou = getattr(env, "limpo_soltou", None)
    if soltou is not None:
        soltou_fora = soltou_fora & (soltou < 0.5)
    return caiu | (soltou_fora & (pegou > 0.5) & apos_pegar)


class Caiu:
    """`fell_over`, com uma cláusula a mais (spec `g1-limpo-dois-bits.md` §3.2).

    ⚠ MESMO SLOT `cfg.terminations["fell_over"]`. A 1ª cláusula é o `bad_orientation`
    do molde, INTOCADA — os 70° ficam. A 2ª pega o caso que a orientação sozinha não
    vê: o robô AGACHA DE LADO, sem tombar, e o joelho encosta no chão.

    ⚠ ISTO É TERMINAÇÃO, não recompensa: um `joelho_z_min` alto mata o AGACHAMENTO
    LEGÍTIMO que os níveis altos exigem (a laje a 0,04 m). O knob fica ABAIXO do p10
    do agachamento medido (spec §2.4) e ACIMA do joelho no chão (~0,05).

    ⚠ SEM SENSOR NOVO: `body_link_pose_w` (via `find_bodies`), o mesmo caminho que
    `_meia`/`_ids_palma` já usam para sítios e para a meia-aresta por env.

    ⚠⚠ VIROU CLASSE (revisão independente, item A11): a função rodava
    `find_bodies((".*_knee_link",))` — uma busca por REGEX — A CADA PASSO, dentro
    de uma terminação chamada todo passo de todo env. O `__init__` resolve os ids
    UMA VEZ, o mesmo padrão que `PosturaPorElo` já usa para o braço.
    """

    def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        robo = env.scene[asset_cfg.name]
        ids_joelho, _ = robo.find_bodies((".*_knee_link",))
        self._ids_joelho = ids_joelho

    def __call__(self, env: "ManagerBasedRlEnv", limit_angle: float,
                joelho_z_min: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        tombou = bad_orientation(env, limit_angle, asset_cfg)
        robo = env.scene[asset_cfg.name]
        z_joelho = (robo.data.body_link_pose_w[:, self._ids_joelho, 2]
                   - env.scene.env_origins[:, 2:3])
        return tombou | (z_joelho.amin(dim=-1) < joelho_z_min)
