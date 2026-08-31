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
    nome_do_comando: str | None = None,
) -> float:
    """Escreve o nível dos envs que resetaram. Devolve a média, para o log.

    Passeio aleatório ±1: sobe com sucesso de cadeia, desce sem, `clamp(0, N−1)`. Mais
    o PISO, que sorteia uma fração dos envs uniformemente sobre os níveis abertos.
    """
    buf = garante_nivel(env)
    if forcado is not None:
        buf[env_ids] = int(max(0, min(n_niveis - 1, forcado)))
        return float(buf.float().mean())

    # -------------------------------------------------- O PASSEIO ALEATÓRIO (F6)
    # Sobe com sucesso, desce sem. O nível se EQUILIBRA onde a taxa de sucesso é ~50%,
    # e é por isso que não existe limiar escolhido à mão: o ponto fixo do passeio ±1 é
    # `p = 0,5` por construção.
    #
    # ⚠ SÓ EPISÓDIOS DE CADEIA MOVEM O NÍVEL. Um episódio de LOCOMOÇÃO não tem cadeia,
    # não tem o que fechar, e portanto "fracassaria" sempre — com a fatia de locomoção
    # em 95%, o nível seria empurrado ao piso por episódios que nem tentaram a tarefa.
    # Foi o bug medido em 20/08 na forma espelhada: a probabilidade de subir caía de `p`
    # para `0,7·p` e o ponto fixo saía de 0,5 para 0,714.
    #
    # ⚠ Ele lê o comando AQUI porque a ordem do currículo é `forma -> nivel -> elo`, e
    # o `command_manager.reset` só roda depois de TODO o currículo
    # (`manager_based_rl_env.py:581`). Portanto `fechou` e `_cadeia` ainda são do
    # episódio que ACABOU. Se o `elo` rodasse antes, isto leria o episódio seguinte.
    if nome_do_comando is not None and len(env_ids):
        try:
            cmd = env.command_manager.get_term(nome_do_comando)
            de_cadeia = cmd._cadeia[env_ids] >= 0
            sucesso = cmd.fechou[env_ids]
        except (KeyError, AttributeError):
            de_cadeia = sucesso = None
        if de_cadeia is not None and bool(de_cadeia.any()):
            passo = torch.where(sucesso, 1, -1) * de_cadeia.long()
            buf[env_ids] = (buf[env_ids] + passo).clamp(0, n_niveis - 1)

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
            # ⚠ O PASSO em que o balanço começou, e não um contador próprio. Ver o
            # bloco `passos_por_iteracao` do `knobs.Forma`: um `contador += 1` neste
            # termo conta PASSOS (o `_reset_idx` roda quando QUALQUER env reseta), e a
            # carência de 200 "iterações" era atingida em ~17.
            "passo_inicial": -1.0,        # -1 = ainda não começou
            "iters_balanco": 0.0,         # DERIVADO, para o log e o checkpoint
            "ultimo_degrau": -1.0,        # a iteração do último degrau da rampa
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

    # ⚠ A ITERAÇÃO É DERIVADA DO CONTADOR DE PASSOS DO ENV, e não incrementada aqui.
    # `common_step_counter` conta passos (`manager_based_rl_env.py:431`) e o mjlab o
    # PERSISTE no checkpoint (`mjlab/rl/runner.py:73`), portanto a rampa fica
    # resume-safe de graça e monotônica por construção.
    passo = float(getattr(env, "common_step_counter", 0))
    if st["passo_inicial"] < 0.0:
        st["passo_inicial"] = passo
    st["iters_balanco"] = ((passo - st["passo_inicial"])
                           / max(int(f.passos_por_iteracao), 1))

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
    # ⚠ UM SINAL SÓ, e desde 27/08 ele é a `eficiencia_min` — a projeção da velocidade
    # real na direção comandada, pontuada POR SEGMENTO de comando e reduzida pelo pior.
    # Adimensional, imune ao alargamento das faixas na it 5000, e ZERO para a estátua.
    #
    # ⚠ ELE NÃO É MAIS A `razao_marcha`, e o motivo é medido. Aquela é `1 − Σ‖v_cmd −
    # v_real‖/Σ‖v_cmd‖`, uma soma de NORMAS: norma nunca cancela, portanto ruído de média
    # zero sempre a infla. No bloco 1 o `std` foi de 0,43 para 0,61 e a razão caiu de
    # 0,514 para 0,426 — com DURAÇÃO e QUEDA parados e o `play` determinístico andando
    # bem. O portão congelou na banda morta e a rampa deu UM degrau em 1341 iterações.
    # A `razao_marcha` fica no log como diagnóstico, para comparar as duas curvas.
    #
    # ⚠ O `min` e não a média: um robô que anda reto e não gira mostra média alta e
    # mínimo baixo, e é o mínimo que responde "sabe andar". O `error_vel_yaw` está em
    # ~2,5 há 5000 iterações e nenhum portão olhava para ele.
    #
    # ⚠⚠ A MÉDIA É SÓ DOS ENVS QUE ANDAM, e a máscara nasceu de um defeito MEDIDO em
    # 31/08. Até então a média era sobre TODOS os envs. O twist é forçado a zero nos
    # elos de manipulação, portanto `seg_pedido` nunca alcança `pedido_min_segmento` e
    # `eficiencia_min` fica em ZERO EXATO naqueles envs. O portão se envenenava com a
    # própria rampa:
    #
    #     rampa desce -> fatia de manipulação cresce -> média DILUÍDA cai
    #          ^                                                 |
    #          +------- portão abre <- média sobe <- rampa REVERTE
    #
    # A aritmética fecha em três casas no bloco 6, iteração 785: eficiência dos envs que
    # andam ~0,80, fatia de manipulação 0,272, média diluída prevista 0,582 contra
    # 0,5844 medida. E o ponto fixo do laço é `efic × (1 − fatia) = limiar`, isto é
    # fatia <= 0,375 com `limiar_portao = 0,50`. O destino `alvo_loco_min = 0,30` era
    # INALCANÇÁVEL por construção, e a rampa parou em ~0,79 de 33 degraus possíveis.
    #
    # ⚠ A máscara é `segmentos > 0`, e não o canal do elo. Ela se autodescreve — "a
    # eficiência de quem foi PEDIDO para andar" — e não acopla este termo ao layout do
    # comando de caixa. Um env de `CARREGAR` tem twist ativo e entra na conta, que é o
    # correto: ali andar É a tarefa.
    #
    # ⚠ DECISÃO DO DONO (31/08): o portão olha SÓ para as tarefas de andar. Conforme o
    # robô fica bom em andar, a manipulação ganha chão até 30%, INDEPENDENTE de o robô
    # estar conseguindo pegar a caixa. Sucesso de manipulação move o `nivel`, e nunca a
    # `forma`.
    try:
        tw = env.command_manager.get_term(nome_do_twist)
        efic = tw.metrics["eficiencia_min"]
        pedidos = tw.metrics["segmentos"] > 0
        # ⚠ Sem nenhum env pedido a andar, MANTÉM o sinal anterior em vez de ler 0. Ler
        # zero fecharia o portão por ausência de dado, que é o defeito acima com outro
        # nome.
        sinal = float(efic[pedidos].mean()) if bool(pedidos.any()) else st["razao"]
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
            # ⚠ UM DEGRAU POR JANELA, e não um por chamada. Este termo roda várias
            # vezes por iteração de PPO (uma por passo em que algum env reseta), logo um
            # `% iters_entre_degraus == 0` desceria a rampa muitas vezes na mesma
            # iteração. O `ultimo_degrau` garante uma descida por janela.
            janela = int(st["iters_balanco"]) // max(int(f.iters_entre_degraus), 1)
            if janela > st["ultimo_degrau"]:
                st["ultimo_degrau"] = float(janela)
                st["alvo"] = max(st["alvo"] - f.alvo_passo, f.alvo_loco_min)

    # o alvo NUNCA sai da faixa — nem por acumulação de ponto flutuante
    st["alvo"] = min(max(st["alvo"], f.alvo_loco_min), f.alvo_loco_max)

    st["sorteio"] = resolve_sorteio(st["alvo"], st["dur_loco"], st["dur_manip"],
                                    f.sorteio_min, f.sorteio_max)
    return st["sorteio"]
