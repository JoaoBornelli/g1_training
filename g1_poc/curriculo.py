"""O currículo do g1_poc (§10).

Três partes, e a separação é deliberada:

    A · adaptativa, por env   — o que a TAREFA pede. Adapta por sucesso.
    B · agendada, por passo   — as faixas do twist. `mdp.commands_vel` do mjlab.
    C · agendada, por passo   — a qualidade de movimento. `mdp.reward_curriculum`.

O que a tarefa pede pode adaptar por sucesso. O quão LIMPO o movimento tem de ser
não pode: apertar sempre baixa o sucesso.

ESTADO DESTE ARQUIVO — passo 2 da §17:
    `sorteia_forma` está pronto (os eventos e o comando dependem dele).
    `nivel_caixa` está aqui em forma MÍNIMA: ele só existe para o smoke exercitar a
    regra de promoção. A tabela de células e as cadeias entram no passo 4, depois do
    portão do passo 3.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

NIVEL_MAX = 6


def sorteia_forma(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    frac_locomocao: float,
) -> dict[str, torch.Tensor]:
    """Sorteia a forma do episódio e escreve `env.poc_manipula`.

    ⚠ Isto tem de ser um termo de CURRÍCULO, e não um evento nem um comando. No
    reset a ordem do mjlab é currículo → eventos → comando. O `afasta_cena` e o
    `carga_caixa` precisam da forma ANTES de rodarem, e o comando precisa dela
    depois. O currículo é o único ponto que serve aos dois.
    """
    if not hasattr(env, "poc_manipula"):
        env.poc_manipula = torch.ones(
            env.num_envs, dtype=torch.bool, device=env.device)
    sorteio = torch.rand(len(env_ids), device=env.device)
    env.poc_manipula[env_ids] = sorteio >= frac_locomocao
    return {"frac_manipula": env.poc_manipula.float().mean()}


def nivel_caixa(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
) -> dict[str, torch.Tensor]:
    """Sobe ou desce o nível de cada env, no molde do `terrain_levels_vel`.

        sobe  = episode_success
        desce = ~episode_success
        nivel = clamp(nivel + sobe − desce, 0, NIVEL_MAX)

    Três linhas. Sem EMA, sem contador de episódios, sem limiar, sem grafo, sem
    evento de destravamento.

    Duas propriedades, e elas são o motivo de a regra ser assim:

    1. **O nível equilibra onde a taxa de sucesso é ≈ 50%.** É um passeio aleatório
       ±1 com probabilidade de subir igual a p(sucesso). O ponto fixo é p = 0,5.
       Nenhum limiar é escolhido à mão.
    2. **O rebaixamento É o anti-esquecimento.** Os envs se espalham pelos níveis, e
       sempre há envs nos casos fáceis. O piso de 0,15 do orquestrador antigo deixa
       de existir: ele é consequência da dinâmica.

    Só os episódios de MANIPULAÇÃO movem o nível. Um episódio de locomoção é ensaio.
    """
    if not hasattr(env, "poc_nivel"):
        env.poc_nivel = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device)

    manipula = env.poc_manipula[env_ids]
    sucesso = env.command_manager.get_term(command_name).episode_success[env_ids] > 0.5

    delta = torch.where(sucesso, 1, -1)
    delta = torch.where(manipula, delta, torch.zeros_like(delta))
    env.poc_nivel[env_ids] = torch.clamp(
        env.poc_nivel[env_ids] + delta, 0, NIVEL_MAX)

    niveis = env.poc_nivel.float()
    return {
        "nivel_medio": niveis.mean(),
        "nivel_max": niveis.max(),
        "nivel_min": niveis.min(),
    }
