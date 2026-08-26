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

__all__ = ["nivel", "garante_nivel", "sorteia_elo", "garante_elo",
           "forma", "garante_forma", "resolve_sorteio"]

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
    frac_uniforme: float = 0.0,
) -> float:
    """Escreve o nível dos envs que resetaram. Devolve a média, para o log.

    F0/F1: fixo. F6 troca o corpo desta função pelo passeio ±1 por sucesso, e a
    assinatura não muda.
    """
    buf = garante_nivel(env)
    if forcado is not None:
        buf[env_ids] = int(max(0, min(n_niveis - 1, forcado)))
        return float(buf.float().mean())

    # sem `forcado`, o nível permanece onde está. Na F6 é aqui que o passeio ±1 entra:
    #     sobe = sucesso ; desce = ~sucesso ; buf = clamp(buf + sobe - desce, 0, N-1)

    # ------------------------------------------------------ O PISO DE NÍVEL (F5)
    # ⚠ Uma fração dos envs é sorteada UNIFORMEMENTE sobre os níveis abertos. É seguro
    # barato: o rebaixamento ±1 espalha os envs, mas é DISTRIBUIÇÃO e não garantia — se
    # a política ficar boa, os envs empilham no topo e o nível 0 sai do treino.
    #
    # ⚠ ELE NÃO É O `rho = 0,30` DO `g1_multitask`. Aquele era piso sobre TAREFAS, e com
    # 5 tarefas ocupava 0,75 do sorteio: o teto da locomoção ficava em 0,55 contra os
    # 0,945 que a fatia de 30% exigia, e a fatia alvo virava inalcançável. Piso de NÍVEL
    # e piso de FATIA são eixos ORTOGONAIS — este não toca a divisão loco × cadeia.
    #
    # ⚠ Até a F6 ele é INERTE por construção: sem o passeio, o nível aberto é só o 0, e
    # sortear uniformemente sobre {0} devolve 0. Ele entra agora para a F6 não precisar
    # mexer em duas coisas ao mesmo tempo.
    if frac_uniforme > 0.0 and len(env_ids):
        abertos = int(buf.max()) + 1
        sorteia = torch.rand(len(env_ids), device=env.device) < frac_uniforme
        if bool(sorteia.any()):
            k = torch.randint(abertos, (int(sorteia.sum()),), device=env.device)
            buf[env_ids[sorteia]] = k
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

    # ⚠ DESDE A F5 A FATIA VEM DO CONTROLADOR, e não do knob. O knob passa a ser só o
    # valor de partida (e o valor usado quando o controlador está desligado).
    #
    # ⚠ E o que chega aqui é o SORTEIO JÁ RESOLVIDO, não a fatia alvo. Os dois são
    # coisas diferentes: `alvo` é fatia de TRANSIÇÕES, `sorteio` é probabilidade por
    # EPISÓDIO, e `resolve_sorteio` converte um no outro usando as durações medidas.
    # Usar o alvo direto aqui é a armadilha de 40× deste projeto.
    est = getattr(env, "limpo_forma", None)
    if est is not None and "sorteio" in est:
        fatia_loco = est["sorteio"]

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


# =============================================================================
# O BALANÇO DE FORMA (F5) — o mecanismo central do módulo
# =============================================================================
def resolve_sorteio(alvo: float, dur_loco: float, dur_manip: float,
                    lo: float, hi: float) -> float:
    """Converte uma fatia de TRANSIÇÕES alvo na probabilidade de SORTEIO que a entrega.

        f = alvo·Tm / (Tl·(1−alvo) + alvo·Tm)

    ⚠ ESTA FUNÇÃO É O CORAÇÃO DA F5, e confundir os dois lados dela é a armadilha
    medida deste projeto. O sorteio é por EPISÓDIO; o PPO aprende por TRANSIÇÃO. Um
    episódio de locomoção que morre em 24 passos e um de manipulação que dura 961
    contribuem 40× diferente com o MESMO sorteio:

        Tl    Tm    sorteio 0,30 entrega   para entregar 0,30, sortear
        24   961          1,06%                     0,9449
       150   500         11,4%                      0,5882
      1000   500         46,2%                      0,1765

    O `g1_poc` entregou 70% das transições à manipulação achando que entregava 30%.

    ⚠ Ela é PURA de propósito: nenhum tensor, nenhum env. É o que permite testá-la
    contra a tabela da spec sem simulador — e é o único jeito de saber que a
    aritmética está certa antes de gastar GPU.
    """
    tl = max(float(dur_loco), 1.0)
    tm = max(float(dur_manip), 1.0)
    a = min(max(float(alvo), 0.0), 1.0)
    denom = tl * (1.0 - a) + a * tm
    if denom <= 0.0:
        return hi
    return min(max(a * tm / denom, lo), hi)


def garante_forma(env: "ManagerBasedRlEnv", f) -> dict:
    """Cria o estado do balanço, se ainda não existe. Devolve o dicionário.

    ⚠ Ele vive em `env.limpo_forma`, um dict de floats e não de tensores, porque ele é
    ESCALAR (uma fatia para todos os envs) e porque ele vai para o CHECKPOINT — ver
    `runner.RunnerComEstadoDeCurriculo`.

    ⚠ O estado inicial é ASSIMÉTRICO, e isso é decisão medida:
      · as DURAÇÕES nascem NEUTRAS (episódio cheio). Elas governam a FATIA, e um erro
        ali só desafina o sorteio por ~tau.
      · a `razao_marcha` nasce PESSIMISTA em 0,0. Ela governa o PORTÃO, e um portão que
        nasce aprovando entrega a locomoção ANTES de existir marcha. Foi exatamente o
        que a `dur_loco_ema` neutra em 1000 passos fez: ela dava nota máxima à estátua,
        porque estátua não cai.
    """
    if not hasattr(env, "limpo_forma"):
        env.limpo_forma = {
            "alvo": float(f.alvo_loco_max),
            "dur_loco": float(f.dur_inicial_passos),
            "dur_manip": float(f.dur_inicial_passos),
            "razao": 0.0,                 # PESSIMISTA
            "iters_balanco": 0.0,         # conta de quando o BALANÇO começou
            "abriu": 0.0,                 # 1.0 depois de o portão abrir a 1ª vez
        }
    return env.limpo_forma


def forma(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    *,
    f,
    elo_loco: int,
    nome_do_twist: str = "twist",
) -> float:
    """O controlador de fatia. Devolve o SORTEIO resolvido, para o log.

    Ele roda como termo de currículo, DEPOIS do `elo` e do `nivel` no dict — a ordem é
    contrato, e está testada.

    ⚠ ORDEM NO DICT: `command_vel` -> `elo` -> `nivel` -> `forma`. Invertida, um bug
    medido em 20/08 fazia a probabilidade de subir de nível cair de `p` para `0,7·p`, o
    ponto fixo saía de 0,5 para 0,714, e um episódio de LOCOMOÇÃO rebaixava o nível em
    70% das vezes.
    """
    st = garante_forma(env, f)
    if not f.controla:
        st["sorteio"] = resolve_sorteio(st["alvo"], st["dur_loco"], st["dur_manip"],
                                        f.sorteio_min, f.sorteio_max)
        return st["sorteio"]

    st["iters_balanco"] += 1.0

    # ---------------------------------------------------- 1. as EMAs de duração
    # ⚠ Medidas dos episódios QUE ACABARAM, e separadas por forma. É o que faz a
    # aritmética acima usar números reais em vez de um chute.
    if len(env_ids):
        dur = env.episode_length_buf[env_ids].float()
        elo = garante_elo(env, elo_loco)[env_ids]
        eh_loco = elo == elo_loco
        a = f.ema
        if bool(eh_loco.any()):
            st["dur_loco"] = a * st["dur_loco"] + (1 - a) * float(dur[eh_loco].mean())
        if bool((~eh_loco).any()):
            st["dur_manip"] = a * st["dur_manip"] + (1 - a) * float(dur[~eh_loco].mean())

    # ------------------------------------------------------ 2. o sinal do portão
    # ⚠ UM SINAL SÓ, e ele é a `razao_marcha` — adimensional, imune ao alargamento das
    # faixas de comando na iteração 5000, e ZERO para a estátua.
    try:
        tw = env.command_manager.get_term(nome_do_twist)
        sinal = float(tw.metrics["razao_marcha"].mean())
    except (KeyError, AttributeError):
        sinal = st["razao"]
    st["razao"] = f.ema * st["razao"] + (1 - f.ema) * sinal

    # ----------------------------------------------------- 3. o portão e a rampa
    # ⚠ CARÊNCIA contada de quando o BALANÇO começou, e não de passo global. Com passo
    # global, retomar um checkpoint depois da carência abriria o portão no passo 1.
    if st["iters_balanco"] >= f.carencia_iters:
        # ⚠ HISTERESE ASSIMÉTRICA: lento para avançar, rápido para defender.
        if st["razao"] < f.histerese * f.limiar_portao:
            st["alvo"] = min(st["alvo"] + f.alvo_passo, f.alvo_loco_max)
        elif st["razao"] >= f.limiar_portao:
            st["abriu"] = 1.0
            if int(st["iters_balanco"]) % max(int(f.iters_entre_degraus), 1) == 0:
                st["alvo"] = max(st["alvo"] - f.alvo_passo, f.alvo_loco_min)

    # o alvo NUNCA sai da faixa — nem por acumulação de ponto flutuante
    st["alvo"] = min(max(st["alvo"], f.alvo_loco_min), f.alvo_loco_max)

    st["sorteio"] = resolve_sorteio(st["alvo"], st["dur_loco"], st["dur_manip"],
                                    f.sorteio_min, f.sorteio_max)
    return st["sorteio"]
