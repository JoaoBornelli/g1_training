"""O que a política vê a mais que a receita do fabricante.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Só `mjlab`, que é framework.

CONTRATO DO LAYOUT: canal novo entra sempre POR ÚLTIMO, e nos DOIS grupos, na MESMA
ordem. Assim migrar um checkpoint é um APPEND de colunas, e nunca uma inserção no
meio — uma inserção no meio desloca todo peso da primeira camada em silêncio, e a
política sai andando de lado sem uma linha de erro.

ESCOPO DESTA FASE (F2): o one-hot dos 5 elos. Os canais da caixa (posição do alvo,
face pedida, erro angular) entram na F3, depois deste, pelo mesmo contrato.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["um_de_cinco", "um_de_cinco_interno", "caixa_no_frame_da_base",
           "fatia_do_elo", "fatia_do_elo_interno", "N_SLOTS", "N_CAIXA"]

# caixa_b(3) alvo_b(3) giro_b(3) meia_aresta(1). Spec §4.1.
N_CAIXA = 10

N_SLOTS = 5


def fatia_do_elo(dim_total: int) -> slice:
    """A fatia do one-hot do elo dentro da observação CONCATENADA do ator.

    ⚠ CONTADA DO FIM, e isso é o contrato e não uma conveniência. O `env_cfg` acrescenta
    os canais próprios por APPEND, nesta ordem: `elo` e depois `caixa`
    (`env_cfg.py:456` e `:465`). Portanto o elo é o penúltimo bloco, sempre.

    Contar do INÍCIO exigiria somar as dimensões dos termos do molde, e um upgrade do
    `mjlab` que acrescente um canal deslocaria tudo em silêncio. Contando do fim, só
    quebra quem acrescentar termo DEPOIS do `caixa` — e aí o invariante de one-hot no
    consumidor pega na primeira iteração.

    MEDIDO: com a observação de ator em 114, isto devolve `slice(99, 104)`, que é onde o
    `observation_manager` põe o termo. O `smoke` afirma isso contra o manager vivo.
    """
    fim = dim_total - N_CAIXA
    return slice(fim - N_SLOTS, fim)


def fatia_do_elo_interno(dim_total: int) -> slice:
    """A fatia do one-hot do elo INTERNO dentro da observação do CRÍTICO.

    É o ÚLTIMO bloco: o `env_cfg` o acrescenta depois de `caixa`, só no grupo `critic`
    (spec §6.1). O `PPOPorElo` agrupa por ele.
    """
    return slice(dim_total - N_SLOTS, dim_total)


def um_de_cinco(env: "ManagerBasedRlEnv", command_name: str,
                canal_do_elo: int) -> torch.Tensor:
    """O one-hot do elo corrente: 5 slots, um por estado.

        0 ANDAR   1 REORIENTAR   2 PEGAR   3 CARREGAR   4 BOTAR

    ⚠ ELE É LIDO DO COMANDO, POR PASSO, e não de um buffer de reset. Isso é o que
    permite o elo TROCAR DENTRO do episódio na F4 sem reset e sem resample — e era a
    única incompatibilidade real entre a máquina de elo do `g1_poc` e o one-hot do
    `g1_multitask`. Ela é de uma linha, e é esta.

    ⚠ SEM `noise` E SEM `scale`, e isso é decisão. Ruído num one-hot produziria
    frações entre slots, isto é, estados que não existem; e `scale` num canal que já
    está em [0,1] só desalinharia a escala contra o normalizador.

    ⚠ O one-hot NÃO leva o crédito do andar. O `g1_poc` já tinha o equivalente
    funcional — o bit `caixa_valida` mais o twist forçado a zero — e não andou. A
    razão de engenharia dele é outra, e basta: ele diz QUAL objetivo está ativo, e
    gateia os sete termos de tarefa da F3, que sem gate pagariam o máximo com os
    canais de caixa zerados, porque `exp(0) = 1`.
    """
    comando = env.command_manager.get_command(command_name)
    assert comando is not None, f"comando '{command_name}' não existe"
    elo = comando[:, canal_do_elo].long().clamp(0, N_SLOTS - 1)
    return torch.nn.functional.one_hot(elo, num_classes=N_SLOTS).float()


def um_de_cinco_interno(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    """O one-hot do elo INTERNO — só para o CRÍTICO (spec §6.1).

    ⚠ NÃO ENTRA NO ATOR: em campo ninguém o manda. O crítico o recebe porque a espera
    final rende ~18/s e um env `standing` da locomoção rende 6/s com a MESMA observação
    de ator; sem o interno a função de valor confunde os dois. Ator-crítico assimétrico.
    `aguardando` e `soltou` não precisam de canal: são `interno ≠ publicado`.
    """
    elo = env.command_manager.get_term(command_name)._elo.clamp(0, N_SLOTS - 1)
    return torch.nn.functional.one_hot(elo, num_classes=N_SLOTS).float()


def caixa_no_frame_da_base(env, command_name: str) -> torch.Tensor:
    """Os canais da caixa, TODOS no frame da base. 10 canais, GATEADOS (spec §4.1, §6.1).

        [0:3]  caixa − base, no frame da base
        [3:6]  alvo  − base, no frame da base
        [6:9]  giro_b: eixo × ângulo do giro pedido, no frame da base (spec §8.3)
        [9]    meia_aresta: o meio-lado da caixa deste env, em metros (spec §6.7)

    ⚠⚠ O GATE: quando o elo PUBLICADO é `ANDAR`, os 10 canais são ZERO — mesmo com a
    caixa a 0,5 m. Não existe terceiro estado. É a invariante que substitui o bit
    `VALIDA`, que SAIU da observação (spec §6.2). Sem o gate a política aprendia "ando"
    da distância da caixa (5 m no ANDAR), e sambava em campo com a caixa perto (§5).
    Lê o PUBLICADO (`comando[:, ELO]`), e não o interno: é o que o operador manda.

    ⚠ TUDO NO FRAME DA BASE, e não em mundo. Coordenada de mundo carrega a ORIGEM DO
    ENV, diferente em cada um dos 4096, e o rumo. No frame da base o problema é o mesmo
    em todo env.

    ⚠ O σ NÃO ENTRA AQUI: ele diz "este env é fácil ou difícil", e a política
    condicionaria a ação à forma da RECOMPENSA em vez de à tarefa.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse

    from g1_limpo.comando import ALVO, ANDAR, ELO, GIRO

    cmd = env.command_manager.get_command(command_name)
    robo = env.scene["robot"]
    p, q = robo.data.root_link_pos_w, robo.data.root_link_quat_w
    caixa_b = quat_apply_inverse(q, env.scene["box"].data.root_link_pos_w - p)
    alvo_b = quat_apply_inverse(q, cmd[:, ALVO] - p)
    giro_b = quat_apply_inverse(q, cmd[:, GIRO])
    meia = getattr(env, "limpo_meia_aresta", None)
    # ⚠ O `ObservationManager` chama cada termo UMA vez na construção, para medir a
    # dimensão — e isso acontece ANTES dos eventos de startup (`load_managers` monta os
    # managers e só depois o `__init__` do env roda o `startup`). Nessa sondagem o
    # `limpo_meia_aresta` ainda não existe, e a resposta certa é um zero de forma (n, 1).
    # Em todo passo real o evento já rodou; o smoke afirma que o canal bate com
    # `limpo_meia_aresta` env a env (spec §11.1 item 3), portanto um zero "vazado" para o
    # treino seria pego ali.
    if meia is None:
        meia = torch.zeros(env.num_envs, 3, device=cmd.device)
    canais = torch.cat([caixa_b, alvo_b, giro_b, meia[:, :1]], dim=-1)
    vivo = (cmd[:, ELO].long() != ANDAR).float().unsqueeze(-1)
    return canais * vivo
