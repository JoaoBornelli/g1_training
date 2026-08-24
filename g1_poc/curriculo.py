"""O currículo do g1_poc (§10).

Quatro partes, e a separação é deliberada:

    A · adaptativa, por env   — DOIS adaptativos: a FORMA (por duração medida,
                                controlador de `sorteia_forma`) e o NÍVEL (por
                                sucesso, `nivel_caixa`).
    B · agendada, por passo   — as faixas do twist. É o `mdp.commands_vel` do
                                fabricante, por PASSO GLOBAL, sem gate (21/08).
    C · agendada, por passo   — a qualidade de movimento. `mdp.reward_curriculum`.
    D · tabela de células     — altura, carga, jitter, rotação por nível (§10.1).

O que a tarefa pede pode adaptar por sucesso. O quão LIMPO o movimento tem de ser
não pode: apertar sempre baixa o sucesso.

ORDEM IMPORTA no dict de currículo: `nivel` lê a forma do episódio que ACABOU, e vem
ANTES de `forma`; `sorteia_forma` a sobrescreve com o sorteio do episódio novo. Os eventos leem a forma NOVA e rodam DEPOIS de todo o
currículo, portanto não são afetados. (Bug medido 20/08: com `nivel` depois de
`forma`, a promoção era gateada pela forma do episódio SEGUINTE.)

ESTADO DESTE ARQUIVO — 21/08:
    tabela de células ligada (§10.1); o eixo da LOCOMOÇÃO voltou ao fabricante
    (`commands_vel`, sem gate próprio, sem cronograma de `action_rate`).
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

    UM sinal, e é a razão de marcha dos envs de locomoção:

        razao_marcha_ema >= razao_marcha_alvo     (o robô move como mandado)

    ⚠ **O sinal era a DURAÇÃO do episódio até 24/08, e aquilo media a coisa errada.**
    A duração mede sobrevivência, e ficar de pé sobrevive: um robô imóvel marca 1000
    passos contra um limiar de 600, portanto ele PASSAVA o portão sem nunca ter
    andado. A fatia descia 0,02 a cada 12 iterações até 0,30, e a manipulação ficava
    com 70% das transições com a marcha inexistente. E o caminho de volta exigia
    `dur < 480`, ou seja o robô voltar a CAIR: "parou de andar" era invisível ao
    controlador, e só "começou a cair" era visível. É a causa medida de "o robô pega
    a caixa no nível 1 e nunca aprende a andar".

    O docstring desta função já citava o fabricante como base e usava outra grandeza.
    O `terrain_levels_vel` promove por DISTÂNCIA ANDADA
    (`velocity/mdp/curriculums.py:42-48`) e rebaixa quem anda menos de metade da
    distância comandada (`:52-54`). Duração de episódio não aparece lá. A regra "um
    sinal só" estava certa; a grandeza estava trocada.

    A razão de marcha é a mesma fração do fabricante, com a ordem de integração
    trocada — metade da VELOCIDADE comandada, em vez de metade da DISTÂNCIA
    comandada. Ela é adimensional, portanto o limiar não se move quando o
    `commands_vel` alarga as faixas na iteração 5000. Quem a produz é o
    `TwistPoc._update_metrics`; quem a monta é o `sorteia_forma`.

    ⚠ Sinal TROCADO, e não acrescentado. Um portão com dois sinais conjuntivos foi
    invenção que já quebrou uma vez (o `erro_giro_ema`, abaixo). A sobrevivência sai
    do portão inteira: ela continua no log, e continua governando a fatia de
    transições pelo tempo de vida, que é outro trabalho.

    ⚠ Em 21/08 havia um SEGUNDO sinal, `erro_giro_ema <= 0,30`, e ele saiu. Duas
    razões, e as duas importam:

    1. O número era chutado. `|ωz − ωz_cmd|` é um erro INSTANTÂNEO, e numa marcha
       normal o tronco contra-rotaciona a cada passada — o piso desse número é a
       oscilação da própria marcha, e não a qualidade do rastreio. Medido: ele ficou
       plano em 0,587 por 390 iterações enquanto a `razao_giro` marcava 0,373
       (dezoito vezes o bloco 2). Dois sinais diziam que o yaw estava bem; só o meu
       limiar dizia que não, e ele travava a rampa para sempre.

    2. O fabricante promove por UM sinal. O `terrain_levels_vel` mede a distância
       caminhada e mais nada. Um portão com dois sinais conjuntivos foi invenção
       minha, e a invenção é que quebrou.

    A `razao_giro` continua existindo, mas onde ela deve estar: como portão de
    LEITURA no `leitura.py`. Um número que um humano confere não é maquinaria no
    laço de treino.

    ⚠ A carência `iters_min` FICA, e agora ela é cinto e não corda. A
    `razao_marcha_ema` nasce PESSIMISTA em 0,0 (ver `sorteia_forma`), portanto ela
    reprova o portão na iteração 0 por construção — era exatamente a estreia NEUTRA
    da `dur_loco_ema` em 1000 passos que fazia o portão abrir com dado que não
    existia. Um sinal que nasce reprovando não precisa de carência; a carência
    continua aqui porque ela custa nada e cobre o caso de resume.
    """
    if balanco is None:
        return piso_inicial

    passo_g = int(env.common_step_counter)
    if not hasattr(env, "poc_alvo_loco"):
        env.poc_alvo_loco = float(piso_inicial)
        # ⚠ A carência é medida a partir de QUANDO O BALANÇO COMEÇOU, e nunca de um
        # passo global absoluto. Custou o smoke de 21/08, e o bug era real no treino:
        #
        # num RESUME o `common_step_counter` volta do checkpoint, mas o
        # `poc_alvo_loco` e as EMAs NÃO — o runner só salva o contador. Retomando na
        # iteração 3000 o contador vale 72000, qualquer carência absoluta já estaria
        # vencida, e o balanço desceria no primeiro reset com a `dur_loco_ema` ainda
        # no valor NEUTRO de 1000 passos. Ou seja: exatamente o falso positivo que a
        # carência existe para impedir, e pior, num ponto do treino em que ninguém
        # olharia mais para ela.
        env.poc_alvo_inicio = passo_g
        env.poc_alvo_ultimo = passo_g

    if passo_g - env.poc_alvo_inicio < int(balanco["iters_min"]) * 24:
        return env.poc_alvo_loco
    if passo_g - env.poc_alvo_ultimo < int(balanco["iters_entre_degraus"]) * 24:
        return env.poc_alvo_loco

    razao = float(env.poc_razao_marcha)
    razao_alvo = float(balanco["razao_marcha_alvo"])
    frac = float(balanco["desce_frac"])
    mn, mx = float(balanco["alvo_min"]), float(balanco["alvo_max"])
    passo = float(balanco["passo"])

    apto = razao >= razao_alvo
    # histerese: o retorno exige cair 20% abaixo do limiar, e não só cruzá-lo
    caiu = razao < frac * razao_alvo

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
    twist_command_name: str,
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
    Quem precisa da forma do episódio que ACABOU (`nivel`) tem de vir ANTES dele no
    dict de currículo. Medido 20/08: com `nivel` depois de
    `forma`, a promoção era gateada pela forma do episódio SEGUINTE — `p_up`
    caía de p para 0,7·p, e um episódio de LOCOMOÇÃO rebaixava o nível em 70%
    das vezes.

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
        # ⚠ A razão de marcha nasce PESSIMISTA, e a assimetria com as durações acima
        # é deliberada. As durações governam a FATIA (um erro ali só desafina o
        # sorteio por ~τ); a razão governa o PORTÃO, e um portão que nasce aprovando
        # entrega a locomoção antes de existir marcha. Foi exatamente o que a
        # `dur_loco_ema` neutra em 1000 passos fez até 24/08.
        env.poc_razao_marcha = torch.zeros((), device=env.device)

    # --- mede as durações dos episódios que ACABARAM (a forma ainda é a antiga:
    # este termo lê ANTES de sobrescrever; episode_length_buf zera só no fim) ---
    if len(env_ids) > 0:
        antiga = env.poc_manipula[env_ids]
        loco = env_ids[~antiga]
        manip = env_ids[antiga]
        if len(loco) > 0:
            amostra = env.episode_length_buf[loco].float().mean()
            env.poc_dur_loco = ema * env.poc_dur_loco + (1.0 - ema) * amostra
            # --- a razão de marcha, só dos envs de LOCOMOÇÃO que acabaram ---
            #
            # ⚠ Os dois somatórios são POPULACIONAIS antes da divisão, e não uma
            # média de razões por env. Um env que recebeu comando quase zero pelo
            # episódio inteiro tem demanda ≈ 0 e razão indefinida; ele pesaria igual
            # a um env que recebeu 1 m/s numa média de razões. Somar os dois lados e
            # dividir uma vez pondera cada env pela demanda que ele de fato recebeu.
            twist = env.command_manager.get_term(twist_command_name)
            erro = float(twist.metrics["marcha_erro"][loco].sum())
            demanda = float(twist.metrics["marcha_demanda"][loco].sum())
            if demanda > 1e-6:
                amostra_r = max(0.0, min(1.0, 1.0 - erro / demanda))
                env.poc_razao_marcha = (ema * env.poc_razao_marcha
                                        + (1.0 - ema) * amostra_r)
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
        # ⚠ O sinal do PORTÃO desde 24/08. É o único número do log que mede andar
        # separado por forma: o `Metrics/peak_height_mean` do fabricante é média
        # GLOBAL e os envs de manipulação, com o pé plantado por desenho, dominam a
        # população viva (`frac_manipula_pop` mediu 0,96-0,99 no bloco 1).
        "razao_marcha_ema": env.poc_razao_marcha.clone(),
        # o alvo é o número que o balanço move. Sem ele no log não há como saber se
        # a rampa andou, e foi essa cegueira que custou o bloco 2.
        "alvo_loco": torch.tensor(alvo, device=dev),
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
    Desenho: **o passo global é o PISO do degrau, e
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
    | `joint_vel_hinge` | `nivel_medio` | desde 21/08 ele só vale na manipulação |

    ⚠ O `action_rate_l2` SAIU desta tabela em 21/08. O fabricante roda −0,10 fixo,
    sem cronograma, e a medida da it 488 fecha o caso: a −0,10 e σ 0,54 o termo já
    cobra −1,49/s, que é o piso de ruído. Só o `hinge` continua aqui.

    ⚠ Tem de vir DEPOIS de `nivel` no dict de currículo: o `nivel_medio` é o buffer
    que o `nivel_caixa` escreve.

    ⚠ Nada disto vai para o checkpoint. Depois de um resume o gate recomeça no
    estágio 0 e recalibra. É o comportamento seguro: um freio recomeça SOLTO, e não
    apertado.
    """
    if not hasattr(env, "poc_estagio_peso"):
        env.poc_estagio_peso = {}
        env.poc_peso_ultimo_degrau = {}
    est = env.poc_estagio_peso.get(reward_name, 0)
    ultimo = env.poc_peso_ultimo_degrau.get(reward_name, 0)

    # ⚠ O sinal `duracao_loco` SAIU em 21/08, junto com o `twist_por_competencia`
    # que o produzia e com o gate do `action_rate` que o consumia. Deixar um sinal
    # sem produtor é convite a religá-lo lendo zero para sempre.
    if sinal == "nivel_medio":
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


# ⚠ `twist_por_competencia` SAIU em 21/08. Ele era um gate por competência sobre as
# faixas do twist, com degraus em 8000 e 12000 iterações — invenção minha.
#
# O fabricante usa `mdp.commands_vel`: por PASSO GLOBAL, degraus em 0, 5000 e 10000,
# sem gate nenhum (`velocity_env_cfg.py:393-407`). O `env_cfg` chama aquele agora.
#
# Com ele saiu o buffer `env.poc_duracao_loco`, que só o gate do `action_rate` lia — e
# aquele gate saiu junto, porque o fabricante roda `action_rate_l2 = −0,10` fixo.
