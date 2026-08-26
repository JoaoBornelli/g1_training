"""Os relógios do currículo. FASE F0/F1: só o dono do NÍVEL.

Três relógios, e eles não se falam:

    A. `command_vel`  agendado por passo global. É o do fabricante, e fica como está.
    B. `forma`        a fatia locomoção × cadeia.        -> F5
    C. `nivel`        a dificuldade do objetivo.          -> aqui (F6 põe o passeio)

⚠ ORDEM NO DICT É CONTRATO, e ela tem teste. No reset o mjlab roda
CURRÍCULO -> EVENTOS -> COMANDO. Portanto:

    o `nivel` escreve `env.limpo_nivel`  (currículo, primeiro)
    os eventos posicionam a cena LENDO esse nível
    o comando sorteia o alvo LENDO esse mesmo nível

Invertida, a coisa quebra em silêncio: num bug medido em 20/08 a probabilidade de
subir caía de `p` para `0,7·p`, o ponto fixo do nível saía de 0,5 para 0,714, e um
episódio de LOCOMOÇÃO rebaixava o nível em 70% das vezes.

ESCOPO DESTA FASE (F2): o nível é FIXO (o `forcado` do knob, ou 0), e o `sorteia_elo`
entrou. O passeio aleatório ±1 por sucesso entra na F6, e o controlador de fatia na
F5 — os dois mudam SÓ este arquivo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["nivel", "garante_nivel", "sorteia_elo", "garante_elo"]

# ⚠ ESTE ARQUIVO NÃO IMPORTA `comando.py`, e não é estilo: `comando.py` importa
# `garante_nivel` daqui, portanto o import de volta seria um ciclo. Os ids de elo
# chegam por PARÂMETRO, do `env_cfg`, que é quem importa os dois.
#
# ⚠ SÓ O `REORIENTAR` E O `PEGAR` entram no sorteio, e a razão é FÍSICA, não
# preferência: o `CARREGAR` e o `BOTAR` começam com a caixa NAS MÃOS, e ninguém põe a
# caixa na mão de um robô no reset — eles existem apenas como 2º elo de uma cadeia,
# alcançados por `_avanca_elo` depois de o `PEGAR` fechar. Isso é F4.
#
# ⚠ CONSEQUÊNCIA DECLARADA: os slots 3 e 4 do one-hot ficam CONSTANTES EM ZERO até a
# F4, e o normalizador do `rsl_rl` os verá acender de repente. MEDIDO no fonte
# (`rsl_rl/modules/normalization.py:48`): a saída é `(x − _mean) / (_std + 1e−2)`, sem
# clamp. Com o canal constante em 0, `_std -> 0` e `_mean -> 0`, portanto o primeiro
# 1,0 entra na rede como **100,0**.
#
# Por que aceitar isso em vez de sortear os quatro elos: pôr um env em "CARREGAR" com
# a caixa na prateleira ensinaria à política que o slot 3 significa "caixa na
# prateleira, ande" — e a F4 teria de DESAPRENDER isso. O choque do normalizador é um
# transiente de algumas centenas de iterações num warm-start que já tem transiente; o
# significado errado de um slot é permanente.
#
# PRÉ-REGISTRADO para a F4: zerar `_var`/`_std` desses dois canais no checkpoint, ou
# aceitar o transiente. Decidir com o painel, não agora.


def garante_nivel(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Cria `env.limpo_nivel` se ele ainda não existe, e devolve o buffer.

    ⚠ Ele precisa existir antes do PRIMEIRO reset, porque os eventos o leem. O
    currículo roda inteiro antes dos eventos, portanto chamar isto no termo de
    currículo é suficiente — mas os consumidores usam esta mesma função, e não um
    `getattr` solto, para que a criação fique num lugar só.
    """
    if not hasattr(env, "limpo_nivel"):
        env.limpo_nivel = torch.zeros(env.num_envs, dtype=torch.long,
                                      device=env.device)
    return env.limpo_nivel


def nivel(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    n_niveis: int,
    forcado: int | None = None,
) -> float:
    """Escreve o nível dos envs que resetaram. Devolve a média, para o log.

    F0/F1: fixo. F6 troca o corpo desta função pelo passeio ±1 por sucesso, e a
    assinatura não muda.
    """
    buf = garante_nivel(env)
    if forcado is not None:
        buf[env_ids] = int(max(0, min(n_niveis - 1, forcado)))
    # sem `forcado`, o nível permanece onde está. Na F6 é aqui que o passeio entra:
    #     sobe = sucesso ; desce = ~sucesso ; buf = clamp(buf + sobe - desce, 0, N-1)
    return float(buf.float().mean())


# =============================================================================
# O ELO — quem faz o quê neste episódio
# =============================================================================
def garante_elo(env: "ManagerBasedRlEnv", elo_loco: int = 0) -> torch.Tensor:
    """Cria `env.limpo_elo` se ainda não existe, e devolve o buffer.

    ⚠ Ele nasce em `elo_loco` para TODOS os envs, e isso importa: os eventos leem este
    buffer, e no primeiro reset o termo de currículo já rodou — mas se alguém montar um
    env sem o termo (o inspetor força o elo por outro caminho), o default seguro é
    locomoção, que não depende de mobília nenhuma.
    """
    if not hasattr(env, "limpo_elo"):
        env.limpo_elo = torch.full((env.num_envs,), int(elo_loco),
                                   dtype=torch.long, device=env.device)
    return env.limpo_elo


def sorteia_elo(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    elo_loco: int,
    elos_manip: tuple[int, ...],
    fatia_loco: float,
    forcado: int | None = None,
) -> float:
    """Sorteia o elo POR ENV. Devolve a fatia de locomoção medida, para o log.

    ⚠ ELE RODA NO CURRÍCULO, e não num evento, porque a ORDEM É CONTRATO: no reset o
    mjlab faz `curriculum_manager.compute` (`manager_based_rl_env.py:554`) ->
    `event_manager.apply(mode="reset")` (`:560`) -> `command_manager.reset`, que
    resampleia (`:581`). O elo tem de existir antes dos eventos, porque o reset de pose
    da base depende dele; e antes do comando, porque o alvo depende dele.

    Num evento, o elo chegaria DEPOIS de o reset de pose já ter acontecido.

    ⚠ A FATIA NUNCA É 1,00, e o motivo não é o treino — é o normalizador. Com 1,00 os
    slots de manipulação do one-hot são constantes em zero, e
    `rsl_rl/modules/normalization.py:48` divide por `_std + 1e−2` sem clamp: ao
    acender, 1,0 entra na rede como 100,0. Com 0,95, 5% dos episódios são de
    manipulação desde o passo 0 e os slots sorteáveis nunca são constantes.

    F5 troca o `fatia_loco` fixo pelo controlador de fatia, gateado pela
    `razao_marcha`. A assinatura não muda.
    """
    buf = garante_elo(env, elo_loco)
    if forcado is not None:
        buf[env_ids] = int(forcado)
        return float(forcado)

    n = len(env_ids)
    if n:
        sorteio = torch.rand(n, device=env.device)
        # o índice do elo de manipulação, uniforme entre os sorteáveis
        k = torch.randint(len(elos_manip), (n,), device=env.device)
        tabela = torch.tensor(elos_manip, dtype=torch.long, device=env.device)
        buf[env_ids] = torch.where(sorteio < fatia_loco,
                                   torch.full((n,), int(elo_loco),
                                              dtype=torch.long, device=env.device),
                                   tabela[k])
    return float((buf == int(elo_loco)).float().mean())
