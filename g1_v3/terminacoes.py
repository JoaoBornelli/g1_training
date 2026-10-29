"""Terminações da v3 da fase D: o fim da cadeia (`time_out`) e a queda (plano §6.5, D6; PD-21).

Cada termo é um núcleo puro sobre tensores (`n_*`) mais um invólucro fino que lê o env. O teste
chama o núcleo com tensores sintéticos. `fim_da_cadeia`, `queda` (D) e `caixa_perdida` (E);
`batente` é da fase H.

⚠ TERMINAR SÓ O QUE NÃO TEM COMO SER PAGO (princípio do g1_limpo/terminacoes.py). A queda é o
caso que o robô não recupera. Nas tarefas de andar, a inclinação de marcha (0,8 rad) e a pelve
a menos de 0,20 m do terreno; na manipulação, só 70° ou o joelho a menos de 0,05 m. Inclinar
para pegar não é queda.

⚠ Toda altura é medida contra o TERRENO, e não contra a origem do mundo (PD-21). A origem do
subterreno pode estar a +1,4 m (escada, caixas). Medir contra o mundo daria "caiu" em cima do
degrau e "de pé" no buraco.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from g1_v3.knobs import ATIVO, TAREFAS, Terminacao

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# índices em TAREFAS (one-hot da I §2.2): o núcleo recebe o índice long, não o nome
_MARCHA = (TAREFAS.index("ANDAR"), TAREFAS.index("CARREGAR"))  # S §10: inclinação e pelve
_MANIP = (TAREFAS.index("PEGAR"), TAREFAS.index("BOTAR"))  # S §10: inclinação de 70° e joelho


def n_queda(tarefa, incl_rad, z_pelve_w, z_chao_pelve, z_joelho_w, z_chao_joelho,
            k: Terminacao) -> torch.Tensor:
    """(E,) bool. Cai se a inclinação passa do limite da tarefa ou a altura cai abaixo do mínimo.

    ANDAR e CARREGAR: incl > `incl_marcha` ou (z_pelve − z_chao_pelve) < `base_min`.
    PEGAR e BOTAR: incl > `incl_manip_graus` ou (z_joelho − z_chao_joelho) < `joelho_min`.
    Os dois z de chão vêm do terreno (`altura_pelve`, `altura_joelhos`), não do mundo (PD-21).
    `z_joelho_w` e `z_chao_joelho` são do joelho MAIS BAIXO: o invólucro escolhe o joelho.
    ⚠ O limite de manipulação está em graus e a inclinação em rad: converte-se aqui, uma vez.
    REORIENTAR e LEVANTAR não estão no env: caem em nenhum ramo e dão False.
    """
    marcha = (incl_rad > k.incl_marcha) | (z_pelve_w - z_chao_pelve < k.base_min)
    manip = (incl_rad > math.radians(k.incl_manip_graus)) | (z_joelho_w - z_chao_joelho < k.joelho_min)
    na_marcha = torch.isin(tarefa, torch.tensor(_MARCHA, device=tarefa.device))
    na_manip = torch.isin(tarefa, torch.tensor(_MANIP, device=tarefa.device))
    return (na_marcha & marcha) | (na_manip & manip)


def n_fim_da_cadeia(passos, passos_max, fim) -> torch.Tensor:
    """(E,) bool. Acaba no tempo máximo do episódio (S §7) ou quando a cadeia acabou (`v3_fim`)."""
    return (passos >= passos_max) | fim


def fim_da_cadeia(env: ManagerBasedRlEnv) -> torch.Tensor:
    """`time_out=True` no env_cfg (§6.5). Não se mascara por tarefa: a cadeia inteira acaba aqui."""
    return n_fim_da_cadeia(env.episode_length_buf, env.max_episode_length, env.v3_fim)


def n_caixa_perdida(perdeu_agora, vira_andar: bool) -> torch.Tensor:
    """(E,) bool. A caixa caiu neste passo e a chave `caixa_perdida_vira_andar` está desligada (PD-27).

    Ligada, a perda vira ANDAR (`v3_tarefa`, no `comando.cadeia`) e o episódio segue: nunca termina aqui.
    ⚠ `perdeu_agora` é o instante da perda; o termo acaba no passo em que ela acontece, não depois.
    """
    return perdeu_agora & (not vira_andar)


def queda(env: ManagerBasedRlEnv, tarefas: tuple[str, ...]) -> torch.Tensor:
    """Invólucro de `n_queda`. Lê o robô e os sensores de altura da `cena.sensores` (PD-21).

    Altura do chão = z do frame − altura medida pelo sensor: o `TerrainHeightSensor` devolve a
    folga acima do terreno, e o z de chão sai da mesma conta que o núcleo faz.
    ⚠ `curriculo.mascara` é importado aqui dentro: o `curriculo.py` é de outra lane (D4) e o
    teste deste módulo não pode depender dele para importar.
    """
    from g1_v3.curriculo import mascara

    robo = env.scene["robot"]
    incl = torch.acos((-robo.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))

    sp = env.scene["altura_pelve"].data
    z_pelve = sp.frame_pos_w[:, 0, 2]
    z_chao_pelve = z_pelve - sp.heights[:, 0]

    sj = env.scene["altura_joelhos"].data
    z_j = sj.frame_pos_w[:, :, 2]  # (E,2): joelhos esquerdo e direito
    i = sj.heights.argmin(dim=-1, keepdim=True)  # o joelho mais baixo do terreno
    z_joelho = z_j.gather(-1, i).squeeze(-1)
    z_chao_joelho = (z_j - sj.heights).gather(-1, i).squeeze(-1)

    cai = n_queda(env.v3_tarefa, incl, z_pelve, z_chao_pelve, z_joelho, z_chao_joelho,
                  ATIVO.terminacao)
    # ⚠ torch.where, e não multiplicação: um NaN de um env fora da tarefa não pode vazar.
    return torch.where(mascara(env, tarefas), cai, torch.zeros_like(cai))


def caixa_perdida(env: ManagerBasedRlEnv, tarefas: tuple[str, ...]) -> torch.Tensor:
    """Invólucro de `n_caixa_perdida`. Lê `v3_perdeu_agora` e a chave `ATIVO.cadeia` (PD-27).

    ⚠ `curriculo.mascara` é importado aqui dentro, pelo mesmo motivo de `queda`.
    """
    from g1_v3.curriculo import mascara

    cai = n_caixa_perdida(env.v3_perdeu_agora, ATIVO.cadeia.caixa_perdida_vira_andar)
    # ⚠ torch.where, e não multiplicação: o mesmo motivo de `queda`.
    return torch.where(mascara(env, tarefas), cai, torch.zeros_like(cai))
