"""O currículo do g1_poc (§10).

Quatro partes, e a separação é deliberada:

    A · adaptativa, por env   — o que a TAREFA pede. Adapta por sucesso.
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

    ⚠ `frac_manipula_pop` é a fração POPULACIONAL, e NÃO o sorteio. Em regime ela é
    a fatia de TRANSIÇÕES — que é o que o PPO aprende — e essa fatia é governada
    pelo TEMPO DE VIDA do episódio, não pelo sorteio:

        0,30 × 24 passos / (0,30 × 24 + 0,70 × 961) = 1,06%

    O sorteio é `Episodio.frac_locomocao`; esta métrica é o resultado.

    ⚠ Este termo SOBRESCREVE `env.poc_manipula` com a forma do episódio NOVO.
    Quem precisa da forma do episódio que ACABOU (`nivel`, `twist_ranges`) tem de
    vir ANTES dele no dict de currículo. Medido 20/08: com `nivel` depois de
    `forma`, a promoção era gateada pela forma do episódio SEGUINTE — `p_up`
    caía de p para 0,7·p, e um episódio de LOCOMOÇÃO rebaixava o nível em 70%
    das vezes.
    """
    if not hasattr(env, "poc_manipula"):
        env.poc_manipula = torch.ones(
            env.num_envs, dtype=torch.bool, device=env.device)
    sorteio = torch.rand(len(env_ids), device=env.device)
    env.poc_manipula[env_ids] = sorteio >= frac_locomocao
    return {"frac_manipula_pop": env.poc_manipula.float().mean()}


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
