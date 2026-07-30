"""Terminações próprias do multi-tarefa, e a definição de "de pé".

Já vêm herdadas e não precisam de nada: `time_out` (20 s), `fell_over`
(`bad_orientation` com 1.2217 rad = 70° exatos, do fabricante) e `nonfinite` (do
`base_env`). As três daqui são as da §6b/D que não existem em lugar nenhum.

**As três são GATEADAS por tarefa**, e o gate não é detalhe: a distinção
`pegar` × `carregar` é o ponto do desenho. No `pegar`, largar no meio da tentativa
deve permitir **nova tentativa no mesmo episódio**; no `botar`, soltar **é** o
objetivo. Só quando carregar é o estado EXIGIDO é que largar é falha.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

ONEHOT = slice(9, 17)
_ROBOT = SceneEntityCfg("robot")


def _mascara(env: "ManagerBasedRlEnv", tasks, gate_command: str) -> torch.Tensor:
    """[B] bool — a tarefa ativa está em `tasks`. Mesma fonte que o gate de reward."""
    onehot = env.command_manager.get_term(gate_command).command[:, ONEHOT]
    idx = torch.as_tensor(list(tasks), dtype=torch.long, device=onehot.device)
    return onehot[:, idx].sum(dim=-1) > 0


# ------------------------------------------------------------------ item 19
def de_pe(env: "ManagerBasedRlEnv", z_min: float = 0.65,
          tilt_max_rad: float = 0.349, asset_cfg: SceneEntityCfg = _ROBOT
          ) -> torch.Tensor:
    """[B] bool — `z_pelve >= 0.65` **E** inclinação <= 20° (§14).

    **NÃO é terminação.** É pré-condição dos critérios de SUCESSO do `pegar`, do
    `botar` e do `parado`, e por isso vive aqui e não em `metrics.py`: quem termina
    e quem tem sucesso olham a mesma definição de "de pé".

    Por que precisa de número próprio em vez de reusar o `fell_over`: o `fell_over` é
    70°, folgadíssimo — ele dá "de pé" pra um robô dobrado a 65°. Com esse limite, o
    motivo de ter acrescentado a condição (garantir que o estado final do `pegar` é o
    estado inicial canônico do `andar c/ caixa` e do `botar`) não se cumpriria.

    Os 0.65 m se comparam com 0.76 m, que é a pelve no keyframe `KNEES_BENT`
    (medido) — ou seja permite agachar 11 cm, não permite ficar dobrado."""
    robo: Entity = env.scene[asset_cfg.name]
    z = robo.data.root_link_pos_w[:, 2]
    # projeção da gravidade no eixo z do corpo: 1 = ereto, cos(tilt) em geral
    up = robo.data.projected_gravity_b[:, 2]
    return (z >= z_min) & (up <= -torch.cos(torch.tensor(tilt_max_rad, device=z.device)))


# ------------------------------------------------------------------ item 21
def largou(env: "ManagerBasedRlEnv", tasks, z_min: float = 0.30,
           box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
           gate_command: str = "lift_target") -> torch.Tensor:
    """[B] bool — a caixa caiu abaixo de `z_min` (§14: 0.30 m).

    Gateada só nas tarefas em que CARREGAR é o estado exigido (`parado c/ caixa`,
    `andar c/ caixa`). 🔴 Fora do `pegar` e do `botar` de propósito: no `pegar`
    largar no meio deve dar nova chance no mesmo episódio, e no `botar` soltar é o
    objetivo."""
    caixa: Entity = env.scene[box_cfg.name]
    return (caixa.data.root_link_pos_w[:, 2] < z_min) & _mascara(env, tasks, gate_command)


def caixa_caiu(env: "ManagerBasedRlEnv", tasks, margem: float = 0.10,
               box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
               table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
               shelf_half_z: float = 0.02,
               gate_command: str = "lift_target") -> torch.Tensor:
    """[B] bool — a caixa está ABAIXO da superfície da prateleira. Só no `reorientar`.

    ⚠️ **Tem que significar "caiu", não "não está apoiada".** No nível topo/fundo do
    eixo de giro a solução é **erguer e rolar** a caixa entre as palmas — no meio da
    manobra ela está no ar, ACIMA da prateleira. Um critério de "não apoiada"
    encerraria o episódio exatamente na única solução que existe pra aquele nível.

    Daí o teste ser puramente vertical e pra BAIXO: `z_caixa < topo − margem`. Com
    `margem` = meia-altura da caixa, só dispara quando o centro dela passou de vez
    por baixo do plano da prateleira, o que só acontece se ela saiu pela borda.

    Nota: no nível de altura 0.00 a prateleira está no chão e esta terminação nunca
    dispara. Correto — não há de onde cair."""
    caixa: Entity = env.scene[box_cfg.name]
    mesa: Entity = env.scene[table_cfg.name]
    topo = mesa.data.root_link_pos_w[:, 2] + shelf_half_z
    abaixo = caixa.data.root_link_pos_w[:, 2] < (topo - margem)
    return abaixo & _mascara(env, tasks, gate_command)


def fora_da_area(env: "ManagerBasedRlEnv", raio: float = 5.0,
                 asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """[B] bool — o robô se afastou mais de `raio` do spawn (§14: 5 m).

    O spawn é a `env_origin` daquele ambiente, não um buffer gravado no reset: a
    origem é fixa por construção, e o jitter de base é de 10 cm — desprezível contra
    5 m. Sem gate: vale em todas as tarefas.

    Só a distância no PLANO. Subir ou descer não é sair da área."""
    robo: Entity = env.scene[asset_cfg.name]
    d = (robo.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).norm(dim=-1)
    return d > raio
