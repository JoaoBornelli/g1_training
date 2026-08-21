"""O currículo do g1_poc (§10).

Quatro partes, e a separação é deliberada:

    A · adaptativa, por env   — DOIS adaptativos: a FORMA (por duração medida,
                                controlador de `sorteia_forma`) e o NÍVEL (por
                                sucesso, `nivel_caixa`).
    B · agendada, por passo   — as faixas do twist. Gate por COMPETÊNCIA (§10.3).
    C · agendada, por passo   — a qualidade de movimento. `mdp.reward_curriculum`.
    D · tabela de células     — altura, carga, jitter, rotação por nível (§10.1).

O que a tarefa pede pode adaptar por sucesso. O quão LIMPO o movimento tem de ser
não pode: apertar sempre baixa o sucesso.

ORDEM IMPORTA no dict de currículo: `twist_ranges` e `nivel` leem a forma do
episódio que ACABOU, e vêm ANTES de `forma`; `sorteia_forma` a sobrescreve com o
sorteio do episódio novo. Os eventos leem a forma NOVA e rodam DEPOIS de todo o
currículo, portanto não são afetados. (Bug medido 20/08: com `nivel` depois de
`forma`, a promoção era gateada pela forma do episódio SEGUINTE.)

ESTADO DESTE ARQUIVO — passo 4 da §17:
    tabela de células ligada (§10.1), gate por competência no lugar de `commands_vel`,
    faltam as cadeias (MACRO 2).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

NIVEL_MAX = 6


def _alvo_locomocao(
    env: ManagerBasedRlEnv,
    piso_inicial: float,
    balanco: dict | None,
) -> float:
    """A fatia de locomoção alvo, como ESTADO (§10.4).

    Sem `balanco` ela é a constante de antes — é o modo que o `play` usa para pinar
    0,0 (`--pegar`) ou 1,0 (`--andar`).

    Com `balanco` ela começa em `piso_inicial` (1,0 = locomoção pura, a caixa não
    existe) e desce até `alvo_min` por competência. E SOBE de volta quando a
    competência cai.

    A assimetria é o ponto. O bloco 2 mediu `dur_loco_ema` subindo a 65 na it 260 e
    caindo a 11 na it 700, com `peak_height_mean` indo de 0,024 a 0,0069 no mesmo
    intervalo — e nada devolveu chão para a locomoção. Aqui devolve.

    Dois sinais, e os DOIS têm de passar para descer:

        dur_loco_ema     >= dur_loco_alvo      (sobrevive ao episódio)
        erro_giro_ema    <= erro_giro_alvo     (controla o próprio yaw)

    Basta UM piorar além da histerese para subir. O `erro_giro_ema` é mantido pela
    recompensa `giro_indevido`, que roda todo passo; o `getattr` pessimista de 1e3
    cobre o primeiro reset, onde ela ainda não escreveu nada.

    ⚠ A carência `iters_min` não é opcional. A `dur_loco_ema` nasce NEUTRA em
    `max_episode_length` (1000 passos, ver `sorteia_forma`), portanto ela já passa o
    limiar de 600 na iteração 0 com dado que não existe. Sem carência o balanço
    desceria 12 degraus antes da primeira medida real.
    """
    if balanco is None:
        return piso_inicial

    if not hasattr(env, "poc_alvo_loco"):
        env.poc_alvo_loco = float(piso_inicial)
        # o relógio começa na carência, e não em zero
        env.poc_alvo_ultimo = int(balanco["iters_min"]) * 24

    passo_g = int(env.common_step_counter)
    if passo_g - env.poc_alvo_ultimo < int(balanco["iters_entre_degraus"]) * 24:
        return env.poc_alvo_loco

    dur = float(env.poc_dur_loco)
    giro = float(getattr(env, "poc_erro_giro_ema", 1e3))
    dur_alvo = float(balanco["dur_loco_alvo"])
    giro_alvo = float(balanco["erro_giro_alvo"])
    frac = float(balanco["desce_frac"])
    mn, mx = float(balanco["alvo_min"]), float(balanco["alvo_max"])
    passo = float(balanco["passo"])

    apto = (dur >= dur_alvo) and (giro <= giro_alvo)
    # histerese: o retorno exige piorar 25% além do limiar (1/0,8), e não só cruzá-lo
    caiu = (dur < frac * dur_alvo) or (giro > giro_alvo / frac)

    if apto and env.poc_alvo_loco > mn:
        env.poc_alvo_loco = max(mn, env.poc_alvo_loco - passo)
        env.poc_alvo_ultimo = passo_g
    elif caiu and env.poc_alvo_loco < mx:
        env.poc_alvo_loco = min(mx, env.poc_alvo_loco + passo)
        env.poc_alvo_ultimo = passo_g
    return env.poc_alvo_loco


def sorteia_forma(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    frac_locomocao: float,
    frac_loco_min: float,
    frac_loco_max: float,
    ema: float,
    balanco: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Sorteia a forma do episódio e escreve `env.poc_manipula`.

    Desde 20/08 é um CONTROLADOR: `frac_locomocao` é a fatia de TRANSIÇÕES alvo,
    e o sorteio é resolvido a partir das durações medidas (ver knobs.Episodio).

    ⚠ Isto tem de ser um termo de CURRÍCULO, e não um evento nem um comando. No
    reset a ordem do mjlab é currículo → eventos → comando. O `afasta_cena` e o
    `carga_caixa` precisam da forma ANTES de rodarem, e o comando precisa dela
    depois. O currículo é o único ponto que serve aos dois.

    ⚠ `frac_manipula_pop` é a fração POPULACIONAL, e NÃO o sorteio. Em regime ela é
    a fatia de TRANSIÇÕES — que é o que o PPO aprende — e essa fatia é governada
    pelo TEMPO DE VIDA do episódio, não pelo sorteio:

        0,30 × 24 passos / (0,30 × 24 + 0,70 × 961) = 1,06%

    É exatamente o laço que o controlador quebra: ele fixa a fatia no alvo.

    ⚠ Este termo SOBRESCREVE `env.poc_manipula` com a forma do episódio NOVO.
    Quem precisa da forma do episódio que ACABOU (`nivel`, `twist_ranges`) tem de
    vir ANTES dele no dict de currículo. Medido 20/08: com `nivel` depois de
    `forma`, a promoção era gateada pela forma do episódio SEGUINTE — `p_up`
    caía de p para 0,7·p, e um episódio de LOCOMOÇÃO rebaixava o nível em 70%
    das vezes.

    ⚠ A EMA daqui INCLUI os envs parados; a do `twist_por_competencia` os exclui.
    São filtros diferentes para perguntas diferentes — não unificar.
    """
    if not hasattr(env, "poc_manipula"):
        env.poc_manipula = torch.ones(
            env.num_envs, dtype=torch.bool, device=env.device)
        # as durações nascem NEUTRAS (episódio cheio): o sorteio começa no alvo e
        # se ajusta por medição em ~τ. Nascer pessimista (24) despejaria locomoção
        # antes de existir amostra.
        env.poc_dur_loco = torch.full((), float(env.max_episode_length),
                                      device=env.device)
        env.poc_dur_manip = torch.full((), float(env.max_episode_length),
                                       device=env.device)

    # --- mede as durações dos episódios que ACABARAM (a forma ainda é a antiga:
    # este termo lê ANTES de sobrescrever; episode_length_buf zera só no fim) ---
    if len(env_ids) > 0:
        antiga = env.poc_manipula[env_ids]
        loco = env_ids[~antiga]
        manip = env_ids[antiga]
        if len(loco) > 0:
            amostra = env.episode_length_buf[loco].float().mean()
            env.poc_dur_loco = ema * env.poc_dur_loco + (1.0 - ema) * amostra
        if len(manip) > 0:
            amostra = env.episode_length_buf[manip].float().mean()
            env.poc_dur_manip = ema * env.poc_dur_manip + (1.0 - ema) * amostra

    # --- o controlador: f = alvo·Tm / (Tl·(1−alvo) + alvo·Tm) ---
    # Fixa a FATIA DE TRANSIÇÕES em `frac_locomocao` resolvendo o sorteio a partir
    # das durações medidas. Tl = 24 e Tm = 961 dão f = 0,945; Tl = Tm dá f = alvo.
    # Sem integrador: o mapa é estático e as EMAs dão a inércia — não oscila.
    # ⚠ o alvo é ESTADO desde 21/08 (§10.4), e é resolvido DEPOIS das EMAs deste
    # passo — o balanço lê `poc_dur_loco` acabado de atualizar.
    alvo = _alvo_locomocao(env, frac_locomocao, balanco)
    tl = float(env.poc_dur_loco)
    tm = float(env.poc_dur_manip)
    f = alvo * tm / max(tl * (1.0 - alvo) + alvo * tm, 1e-6)
    # ⚠ o clamp só vale no MEIO: o play pina o alvo em 0 ou 1, e a álgebra sai
    # exata nos extremos (alvo 0 → f = 0; alvo 1 → f = 1). Clampar ali devolveria
    # 10% de locomoção ao `--pegar` e o viewer abriria sem mobília.
    if 0.0 < alvo < 1.0:
        f = min(max(f, frac_loco_min), frac_loco_max)

    sorteio = torch.rand(len(env_ids), device=env.device)
    env.poc_manipula[env_ids] = sorteio >= f
    dev = env.device
    return {
        "frac_manipula_pop": env.poc_manipula.float().mean(),
        "frac_loco_sorteio": torch.tensor(f, device=dev),
        "dur_loco_ema": torch.tensor(tl, device=dev),
        "dur_manip_ema": torch.tensor(tm, device=dev),
        # o alvo é o número que o balanço move. Sem ele no log não há como saber se
        # a rampa andou, e foi essa cegueira que custou o bloco 2.
        "alvo_loco": torch.tensor(alvo, device=dev),
        "erro_giro_ema": torch.tensor(
            float(getattr(env, "poc_erro_giro_ema", 0.0)), device=dev),
    }


def nivel_caixa(env, env_ids, command_name, nivel_forcado: int | None = None):
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

    # ⚠ Atalho de MEDIÇÃO, não de treino. O `play`, a `sonda` e o `smoke` fixam o
    # nível para conferir a célula (§10.1); no treino fica em None. Forçar
    # `env.poc_nivel` de fora não funciona: este termo roda no reset e aplicaria o
    # delta ±1 por cima.
    if nivel_forcado is not None:
        env.poc_nivel[:] = int(nivel_forcado)
        return _metricas_nivel(env)

    # ⚠ A forma do episódio que ACABOU. Este termo roda ANTES de `sorteia_forma`
    # (ordem do dict, tarefa do env_cfg) — depois dele, a máscara já seria a do
    # episódio NOVO, e a promoção viraria moeda enviesada (medido 20/08:
    # p_up = 0,7·p; ponto fixo saía de 0,5 para 0,714). No primeiríssimo reset o
    # buffer ainda não existe (quem o cria é o `sorteia_forma`): sem forma que
    # tenha ACABADO, nada se promove.
    manipula = getattr(env, "poc_manipula", None)
    if manipula is None:
        return _metricas_nivel(env)
    manipula = manipula[env_ids]

    sucesso = env.command_manager.get_term(command_name).episode_success[env_ids] > 0.5

    delta = torch.where(sucesso, 1, -1)
    delta = torch.where(manipula, delta, torch.zeros_like(delta))
    env.poc_nivel[env_ids] = torch.clamp(
        env.poc_nivel[env_ids] + delta, 0, NIVEL_MAX)

    return _metricas_nivel(env)


def _metricas_nivel(env) -> dict[str, torch.Tensor]:
    """Média, extremos e as duas pontas do histograma.

    ⚠ A média sozinha mente duas vezes: `nivel_medio` "sai de zero" já com
    p = 0,006 (0,0042 — 17 envs de 4096 no nível 1), e `nivel_max` satura em 6 com
    UM env sortudo. As frações dizem onde a POPULAÇÃO está.
    """
    niveis = env.poc_nivel.float()
    return {
        "nivel_medio": niveis.mean(),
        "nivel_max": niveis.max(),
        "nivel_min": niveis.min(),
        "nivel_frac_0": (env.poc_nivel == 0).float().mean(),
        "nivel_frac_3mais": (env.poc_nivel >= 3).float().mean(),
    }


def peso_por_competencia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    stages: list,
    sinal: str,
    alvo: float,
    desce_frac: float,
    iters_entre_degraus: int,
) -> dict[str, torch.Tensor]:
    """Aperta o peso de um termo só quando a habilidade que ele ameaça EXISTE.

    Substitui o `mdp.reward_curriculum` nos dois freios de movimento. Ele avança por
    passo global e só por isso — e o bloco 1 mostrou o que isso custa. Este é o
    mesmo desenho do `twist_por_competencia`: **o passo global é o PISO do degrau, e
    o gatilho é a competência medida.**

    Por que o defeito é de FASE, e não de valor:

    - it 3080: o `hinge` bateu −1,00 e os dois freios passaram a consumir 99,1% de
      todas as penalidades e 100% do sinal positivo. A recompensa líquida virou
      −0,03. No mesmo passo o `contato_ilegal` foi de 6,4% para 17,5% das
      terminações: com movimento caro, escorar o tronco na prateleira economiza
      esforço.
    - it 5000: com o `hinge` recuado para −0,10 e o `action_rate` em −0,25, os dois
      ainda custavam −6,08/s contra +11,6/s de todo o positivo.
    - A §17 põe "refino de pose" no passo 6, o ÚLTIMO. O freio chegava cinco passos
      adiantado, porque o cronograma não sabe em que passo o treino está.

    **Cada freio é gateado pela competência da habilidade que ELE ameaça:**

    | termo | sinal | por quê |
    |---|---|---|
    | `action_rate_l2` | `duracao_loco` | ele suprime a marcha, e quem não anda cai em 24 passos |
    | `joint_vel_hinge` | `nivel_medio` | desde 21/08 ele só vale na manipulação |

    ⚠ Tem de vir DEPOIS de `twist_ranges` e de `nivel` no dict de currículo: o
    `duracao_loco` é a EMA que o `twist_por_competencia` mantém, e o `nivel_medio` é
    o buffer que o `nivel_caixa` escreve.

    ⚠ Nada disto vai para o checkpoint. Depois de um resume o gate recomeça no
    estágio 0 e recalibra. É o comportamento seguro: um freio recomeça SOLTO, e não
    apertado.
    """
    if not hasattr(env, "poc_estagio_peso"):
        env.poc_estagio_peso = {}
        env.poc_peso_ultimo_degrau = {}
    est = env.poc_estagio_peso.get(reward_name, 0)
    ultimo = env.poc_peso_ultimo_degrau.get(reward_name, 0)

    if sinal == "duracao_loco":
        medido = float(getattr(env, "poc_duracao_loco", torch.zeros(())))
        limiar = alvo * env.max_episode_length
    elif sinal == "nivel_medio":
        nivel = getattr(env, "poc_nivel", None)
        medido = 0.0 if nivel is None else float(nivel.float().mean())
        limiar = alvo
    else:
        raise ValueError(f"sinal desconhecido: {sinal!r}")

    passo = env.common_step_counter
    pode = passo - ultimo >= iters_entre_degraus * 24
    if (pode and est + 1 < len(stages)
            and passo >= stages[est + 1]["step"]
            and medido >= limiar):
        est += 1
        ultimo = passo
    elif pode and est > 0 and medido < desce_frac * limiar:
        est -= 1
        ultimo = passo
    env.poc_estagio_peso[reward_name] = est
    env.poc_peso_ultimo_degrau[reward_name] = ultimo

    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    term_cfg.weight = stages[est]["weight"]

    dev = env.device
    return {
        "weight": torch.tensor(float(term_cfg.weight), device=dev),
        "estagio": torch.tensor(float(est), device=dev),
        "sinal_medido": torch.tensor(medido, device=dev),
        "sinal_limiar": torch.tensor(float(limiar), device=dev),
    }


def twist_por_competencia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list,
    duracao_min_frac: float,
    desce_frac: float,
    ema: float,
    iters_entre_degraus: int,
) -> dict[str, torch.Tensor]:
    """Avança as faixas do twist só quando o robô SUSTENTA o teto atual (§10.3).

    Substitui o `mdp.commands_vel`, que avança por passo global e só por isso —
    na it 5099 o robô recebia 2,0 m/s com `peak_height_mean = 2,7 mm`. O sinal de
    competência é a DURAÇÃO do episódio de locomoção: direto (quem não anda cai em
    24 passos) e é a MESMA grandeza que governa a fatia de transições.

    Regra: sobe quando (passo global >= degrau) E (EMA >= duracao_min_frac ×
    episódio cheio) E (passou `iters_entre_degraus` desde o último degrau);
    DESCE quando EMA < desce_frac × alvo. O degrau global é PISO, não gatilho.

    ⚠ ANTES de `forma` no dict: `sorteia_forma` sobrescreve `env.poc_manipula`, e
    aqui precisamos da forma do episódio que ACABOU.
    ⚠ `episode_length_buf[env_ids]` só zera no FIM do `_reset_idx` — aqui ainda
    vale a duração final. Medido.
    ⚠ Os envs PARADOS (`is_standing_env`) saem da EMA: ficar de pé até o time_out
    entregaria 8% do alvo sem andar. Os de giro no lugar CONTAM — girar é andar.
    ⚠ A EMA daqui EXCLUI os parados; a do `sorteia_forma` os inclui. São filtros
    diferentes para perguntas diferentes — não unificar.
    ⚠ A EMA nasce PESSIMISTA (zero) e é tensor no device (uma sync por reset já
    basta para o degrau; 48 syncs/iteração não).
    ⚠ Nada disto vai para o checkpoint: depois de um resume o gate recomeça em
    (0, 0) e recalibra em ~3τ ≈ 12 iterações. Seguro por construção.
    """
    if not hasattr(env, "poc_estagio_twist"):
        env.poc_estagio_twist = 0
        env.poc_duracao_loco = torch.zeros((), device=env.device)
        env.poc_twist_ultimo_degrau = 0

    manipula = getattr(env, "poc_manipula", None)
    if manipula is not None and len(env_ids) > 0:
        loco = env_ids[~manipula[env_ids]]
        if len(loco) > 0:
            parado = env.command_manager.get_term(command_name).is_standing_env
            loco = loco[~parado[loco]]
        if len(loco) > 0:
            amostra = env.episode_length_buf[loco].float().mean()
            env.poc_duracao_loco = ema * env.poc_duracao_loco + (1.0 - ema) * amostra

    alvo = duracao_min_frac * env.max_episode_length
    dur = float(env.poc_duracao_loco)
    est = env.poc_estagio_twist
    passo = env.common_step_counter
    pode = passo - env.poc_twist_ultimo_degrau >= iters_entre_degraus * 24
    if (pode and est + 1 < len(velocity_stages)
            and passo >= velocity_stages[est + 1]["step"]
            and dur >= alvo):
        est += 1
        env.poc_twist_ultimo_degrau = passo
    elif pode and est > 0 and dur < desce_frac * alvo:
        est -= 1
        env.poc_twist_ultimo_degrau = passo
    env.poc_estagio_twist = est

    cfg = env.command_manager.get_term(command_name).cfg
    estagio = velocity_stages[est]
    for chave in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        if estagio.get(chave) is not None:
            setattr(cfg.ranges, chave, estagio[chave])

    dev = env.device
    return {
        "estagio": torch.tensor(float(est), device=dev),
        "duracao_loco_ema": torch.tensor(dur, device=dev),
        "duracao_alvo": torch.tensor(alvo, device=dev),
        "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1], device=dev),
        "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1], device=dev),
    }
