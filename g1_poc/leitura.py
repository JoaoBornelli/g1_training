"""Lê o log do TensorBoard e imprime as tabelas que decidem o bloco.

    python g1_poc/leitura.py                 # o run mais recente
    python g1_poc/leitura.py CAMINHO/run     # um run específico
    python g1_poc/leitura.py --demo          # os números da it 5000 do bloco 1

⚠ Rode como ARQUIVO, e não com `-m`. O `-m g1_poc.leitura` importa o
`g1_poc/__init__.py`, que registra a task e exige o mjlab. Este arquivo só precisa do
`tensorboard`, portanto ele lê um log do Drive numa máquina sem GPU e sem mjlab.

**Por que este arquivo existe.** O `Episode_Reward/*` do rsl_rl é a soma do episódio
dividida por `max_episode_length_s`. Com episódios de 2,05 s num teto de 20 s, todo
valor sai dividido por 0,1027 — e aí `action_rate_l2 = −0,34` parece pequeno quando o
custo real é **−3,31/s**.

Ler o painel sem desfazer essa normalização foi o que deixou dois freios consumirem
55% do sinal positivo por 5000 iterações sem ninguém ver.

    taxa por segundo = Episode_Reward × max_episode_length_s / (passos_médios × dt)

O `--demo` roda a análise nos números medidos na it 5000 do bloco 1. Ele é o
autoteste: se a aritmética quebrar, ele quebra sem GPU e sem log.
"""
from __future__ import annotations

import pathlib
import sys

# --- constantes do env. Se mudarem no `knobs`, mudam aqui. ---
DT = 0.02          # 50 Hz de controle
MAX_EP_S = 20.0    # `Episodio.duracao_s`
PASSOS_CHEIOS = MAX_EP_S / DT   # 1000

# A escada de corte (§17). Pare o bloco se uma linha falhar.
#   (iteração, chave, comparador, alvo, o que significa falhar)
#
# ⚠ 21/08: o `peak_height_mean >= 0,02` SAIU da escada e virou aviso na seção de
# marcha. Ele é padrão de QUALIDADE de marcha, e esta POC mede VIABILIDADE — um robô
# estável a 0,4 m/s serve para pegar a caixa. Medido no bloco 3: 3 mm de elevação com
# `slip_velocity` 0,09 (o pé não escorrega) e `landing_force` 165 N (existe impacto),
# ou seja passos curtos e rasos, e não arrasto. Cortar o bloco por isso seria cortar
# por um alvo que a POC não pediu.
#
# ⚠ E o `erro_giro_ema` saiu junto, com o termo que o produzia. A `razao_giro` fica:
# ela é a mesma pergunta, medida com dois números que o fabricante já loga.
ESCADA = [
    (200, "Policy/mean_noise_std", ">=", 0.85,
     "as penalidades ainda dominam; algum termo ficou pesado"),

    # ⚠ A duração é portão de SOBREVIVÊNCIA, e só isso. Ela é condição necessária e
    # NÃO mede marcha: um robô imóvel marca 1000 passos. Foi por medir andar com este
    # número que o balanço entregava a locomoção sem marcha existir (24/08) — ver
    # `curriculo._alvo_locomocao`.
    (1000, "Curriculum/forma/dur_loco_ema", ">=", 150.0,
     "o robô de locomoção não sobrevive ao episódio"),
    # O PORTÃO DA MARCHA (24/08). Este é o número que o balanço consome, e o alvo é o
    # mesmo `razao_marcha_alvo` dos knobs. Se ele não chegar a 0,50, a fatia fica em
    # 1,00 para sempre e o bloco não produz dado de manipulação nenhum.
    (2000, "Curriculum/forma/razao_marcha_ema", ">=", 0.50,
     "o robô não anda: ele não fecha metade da velocidade comandada, e o balanço "
     "nunca vai liberar fatia para a manipulação"),
    # ⚠ O PORTÃO DO GIRO, acrescentado em 21/08. Ele é o que faltava.
    #
    # O bloco 2 mostrou `track_angular_velocity = 0,0054` contra `0,275` do linear
    # desde a iteração 187 — 51× de diferença, com PESO IGUAL (2,0 nos dois) e com
    # tolerância MAIOR no giro (σ 0,707 contra 0,500). Só um erro de rastreio grande
    # e permanente produz isso. A escada tinha portão de σ, de altura do pé e de
    # duração, e não tinha portão de giro; o número que explicava o colapso estava
    # no log 1000 iterações antes de alguém olhar.
    (400, "Derivado/razao_giro", ">=", 0.10,
     "o yaw não é rastreado — confira o yaw ±3,14 do reset da locomoção"),
    (3000, "Derivado/sucesso_manipulacao", ">=", 0.30,
     "os consertos machucaram a manipulação; o `model_5000` do bloco 1 fazia 0,37"),
]


def derivados(v: dict[str, float]) -> dict[str, float]:
    """Grandezas que o log não tem, e que a leitura precisa.

    ⚠ Os `Metrics/caixa_alvo/*` são multiplicados pelo bit `caixa_valida` DENTRO do
    termo de comando, e o `CommandTerm.reset` os reporta como média sobre os envs que
    resetaram. Portanto eles saem diluídos pela COMPOSIÇÃO DOS RESETS, e não pela
    fração populacional: no bloco 1, 8,89% dos resets eram de manipulação contra
    70,1% da população, porque o episódio de manipulação é 24× mais longo.

    Ler `episode_success = 0,033` como "3% de sucesso" foi o erro. O valor
    condicionado é 37%.
    """
    saida = dict(v)
    frac = v.get("Metrics/caixa_alvo/frac_manipula")
    # ⚠ Guarda do divisor zero, 21/08. Com o balanço automático em `frac_locomocao
    # = 1,0` (locomoção pura) NÃO existe reset de manipulação, portanto
    # `frac_manipula` é exatamente 0 e a desdiluição não é definida. Isso é o estado
    # NORMAL do começo de um bloco, e não um erro — a leitura simplesmente omite os
    # derivados de manipulação.
    if frac and frac > 1e-6:
        for k in ("episode_success", "pegou", "no_alvo", "fecha_de_pe",
                  "fecha_angulo", "fecha_todas"):
            chave = f"Metrics/caixa_alvo/{k}"
            if chave in v:
                saida[f"Derivado/{k}_manipulacao"] = v[chave] / frac
        if "Metrics/caixa_alvo/episode_success" in v:
            saida["Derivado/sucesso_manipulacao"] = (
                v["Metrics/caixa_alvo/episode_success"] / frac)

    # A RAZÃO DO GIRO (21/08). É o único número que separa "o yaw é rastreado" de
    # "o yaw é ignorado", e ele não existe em nenhuma chave crua.
    #
    # Os dois termos têm peso IGUAL (2,0) e o do giro tem σ MAIOR (0,707 contra
    # 0,500). Portanto, com rastreio comparável, a razão deveria ficar próxima de
    # 1. No bloco 2 ela ficou em 0,0054/0,275 = 0,020 — cinquenta vezes abaixo.
    lin = v.get("Episode_Reward/track_linear_velocity")
    ang = v.get("Episode_Reward/track_angular_velocity")
    if lin and abs(lin) > 1e-9 and ang is not None:
        saida["Derivado/razao_giro"] = ang / lin
    return saida


def _ultimo_valor(acc, tag: str):
    try:
        eventos = acc.Scalars(tag)
    except KeyError:
        return None
    return eventos[-1].value if eventos else None


def le_run(run: pathlib.Path) -> tuple[dict[str, float], int]:
    """Devolve o último valor de cada escalar do run, e a iteração."""
    from tensorboard.backend.event_processing import event_accumulator

    acc = event_accumulator.EventAccumulator(
        str(run), size_guidance={event_accumulator.SCALARS: 0})
    acc.Reload()
    tags = acc.Tags()["scalars"]
    valores = {t: _ultimo_valor(acc, t) for t in tags}
    valores = {k: v for k, v in valores.items() if v is not None}
    passo = 0
    if tags:
        passo = acc.Scalars(tags[0])[-1].step
    return valores, passo


def _tem_evento(d: pathlib.Path) -> bool:
    return any(d.glob("events.out.tfevents.*"))


def acha_run(raiz: pathlib.Path | None = None) -> pathlib.Path:
    """Resolve para um diretório de RUN, aceitando três formas de entrada.

    1. o próprio diretório da run (tem `events.out.tfevents.*`) — devolve como está;
    2. o diretório do EXPERIMENTO (`.../logs/g1_poc`) — devolve a run mais recente;
    3. nada — usa `logs/rsl_rl/g1_poc` relativo ao cwd.

    A run mais recente é a alfabeticamente última, e isso é correto porque o
    `launch_training` nomeia o diretório como `<timestamp>_<run_name>`
    (`scripts/train.py:187-189`). É a mesma regra que o `get_checkpoint_path` do mjlab
    usa para o resume.

    ⚠ No Colab cada sessão cria um diretório novo. Portanto apontar para o diretório
    do experimento é o uso normal, e não a exceção.
    """
    raiz = raiz or pathlib.Path("logs/rsl_rl/g1_poc")
    if not raiz.exists():
        raise SystemExit(f"sem logs em {raiz}. Rodou o treino?")
    if _tem_evento(raiz):
        return raiz
    runs = sorted(p for p in raiz.iterdir() if p.is_dir() and _tem_evento(p))
    if not runs:
        subdirs = [p.name for p in raiz.iterdir() if p.is_dir()]
        raise SystemExit(
            f"{raiz} não tem event file nem run com event file.\n"
            f"  subdiretórios: {subdirs}")
    if len(runs) > 1:
        print(f"  {len(runs)} runs em {raiz}; usando a mais recente. "
              f"As outras: {[r.name for r in runs[:-1]]}")
    return runs[-1]


def analisa(v: dict[str, float]) -> None:
    passos = v.get("Train/mean_episode_length")
    if not passos:
        raise SystemExit("falta `Train/mean_episode_length` no log.")
    t_ep_s = passos * DT
    fator = MAX_EP_S / t_ep_s

    print("=" * 74)
    print(f"  episódio médio : {passos:.1f} passos = {t_ep_s:.2f} s "
          f"de um teto de {MAX_EP_S:.0f} s")
    print(f"  fator de leitura: × {fator:.2f}   "
          f"(o rsl_rl divide por {1/fator:.4f})")
    for chave, rotulo in (("Train/mean_reward", "recompensa média"),
                          ("Policy/mean_noise_std", "σ da política"),
                          ("Loss/value_function", "perda do crítico")):
        if chave in v:
            print(f"  {rotulo:16s}: {v[chave]:.4f}")

    # ---------------------------------------------------- recompensa por segundo
    termos = {k[len("Episode_Reward/"):]: val * fator
              for k, val in v.items() if k.startswith("Episode_Reward/")}
    pos = {k: x for k, x in termos.items() if x > 0}
    neg = {k: x for k, x in termos.items() if x < 0}
    soma_pos, soma_neg = sum(pos.values()), sum(neg.values())

    print()
    print("  RECOMPENSA POR SEGUNDO")
    print("  " + "-" * 52)
    for k, x in sorted(pos.items(), key=lambda kv: -kv[1]):
        print(f"    {k:28s} {x:+8.3f}")
    print(f"    {'soma positiva':28s} {soma_pos:+8.3f}")
    print()
    for k, x in sorted(neg.items(), key=lambda kv: kv[1]):
        frac = 100.0 * x / soma_neg if soma_neg else 0.0
        print(f"    {k:28s} {x:+8.3f}   {frac:5.1f}% da penalidade")
    print(f"    {'soma negativa':28s} {soma_neg:+8.3f}")
    if soma_pos:
        print()
        print(f"    penalidade / positivo = {abs(soma_neg)/soma_pos*100:.1f}%")
        if abs(soma_neg) > 0.35 * soma_pos:
            print("    ⚠ acima de 35%: algum freio está dominando. Veja quais dois "
                  "termos somam o topo da lista.")

    # ---------------------------------------------------- terminações
    term = {k[len("Episode_Termination/"):]: val
            for k, val in v.items() if k.startswith("Episode_Termination/")}
    total = sum(term.values())
    if total:
        print()
        print("  TERMINAÇÕES")
        print("  " + "-" * 52)
        for k, x in sorted(term.items(), key=lambda kv: -kv[1]):
            print(f"    {k:28s} {x:8.2f}   {100*x/total:5.1f}%")
        caiu = 100 * term.get("fell_over", 0.0) / total
        if caiu > 50:
            print(f"    ⚠ `fell_over` é {caiu:.0f}% das terminações. O robô cai; "
                  "tudo o mais é consequência.")

    # ---------------------------------------------------- manipulação
    frac_reset = v.get("Metrics/caixa_alvo/frac_manipula")
    if frac_reset and frac_reset > 1e-6:
        print()
        print("  MANIPULAÇÃO, condicionada aos resets de manipulação "
              f"(fração {frac_reset:.4f})")
        print("  " + "-" * 52)
        print("    ⚠ os `Metrics/caixa_alvo/*` são multiplicados pelo bit e diluídos")
        print("      pela composição dos resets. Dividir pela fração desdilui.")
        for k in ("episode_success", "pegou", "no_alvo", "fecha_de_pe",
                  "fecha_angulo", "fecha_todas"):
            chave = f"Metrics/caixa_alvo/{k}"
            if chave in v:
                print(f"    {k:28s} {v[chave]:8.4f} -> {v[chave]/frac_reset:6.1%}")

    # ---------------------------------------------------- marcha
    print()
    print("  MARCHA — é aqui que se vê se existe passo")
    print("  " + "-" * 52)
    # ⚠ A `razao_marcha_ema` vem PRIMEIRO desde 24/08, e ela é o único número desta
    # seção que mede andar SEPARADO POR FORMA. O `peak_height_mean` do fabricante é
    # média GLOBAL sobre todos os envs, e os de manipulação — com o pé plantado por
    # desenho — dominam a população viva (`frac_manipula_pop` mediu 0,96-0,99 no
    # bloco 1). Ele também INFLA, porque o `peak_heights` do mjlab 1.5.1 não zera no
    # reset de episódio. Dois vieses opostos, num número que não separa as formas:
    # ele fica como indício de QUALIDADE, e não como medida de "anda ou não anda".
    for k, rot, alvo in (("Curriculum/forma/razao_marcha_ema", "razão de marcha", 0.50),
                         ("Metrics/peak_height_mean", "pico do pé (m)", 0.02),
                         ("Metrics/landing_force_mean", "força no pouso (N)", None),
                         ("Metrics/slip_velocity_mean", "escorregão (m/s)", None),
                         ("Curriculum/forma/dur_loco_ema", "sobrevida loco (passos)", 150.0),
                         ("Curriculum/forma/dur_manip_ema", "duração manip (passos)", None),
                         ("Curriculum/command_vel/lin_vel_x_max", "teto de vx (m/s)", None)):
        if k in v:
            marca = ""
            if alvo is not None:
                marca = "  ok" if v[k] >= alvo else f"  ⚠ abaixo de {alvo:g}"
            print(f"    {rot:28s} {v[k]:8.4f}{marca}")
    if "Curriculum/forma/razao_marcha_ema" in v:
        print("      (0 = parado, 1 = rastreia o comando. É o sinal do PORTÃO do")
        print("       balanço de forma: abaixo de 0,50 a fatia NÃO desce de 1,00.)")

    # ---------------------------------------------------- os gates
    print()
    print("  GATES POR COMPETÊNCIA")
    print("  " + "-" * 52)
    # ⚠ 21/08: só o `hinge` sobrou. O gate do twist e o do `action_rate` saíram —
    # o fabricante usa `commands_vel` por passo global e `action_rate_l2 = −0,10` fixo.
    for nome in ("hinge",):
        med = v.get(f"Curriculum/{nome}/sinal_medido")
        lim = v.get(f"Curriculum/{nome}/sinal_limiar")
        peso = v.get(f"Curriculum/{nome}/weight")
        est = v.get(f"Curriculum/{nome}/estagio")
        if med is None and est is None:
            continue
        linha = f"    {nome:16s} estágio {est if est is not None else '-'}"
        if peso is not None:
            linha += f"  peso {peso:+.3f}"
        if med is not None and lim is not None:
            linha += f"  sinal {med:.1f} / {lim:.1f}"
        print(linha)

    # ------------------------------------------------ giro e balanço de forma
    # Este bloco é de 21/08, e ele existe por um motivo específico: o bloco 2
    # colapsou por GIRO, e nenhuma das seções acima mostrava giro. A `razao_giro`
    # ficou em 0,020 desde a it 187 — cinquenta vezes abaixo do esperado — e o
    # diagnóstico só saiu na it 1216, olhando um play.
    print()
    print("  GIRO — o eixo com um guardião só")
    print("  " + "-" * 52)
    razao = v.get("Derivado/razao_giro")
    if razao is not None:
        marca = "  ok" if razao >= 0.10 else "  ⚠ o yaw NÃO é rastreado"
        print(f"    track_ang / track_lin       {razao:8.4f}{marca}")
        print("      (pesos iguais e σ maior no giro: com rastreio comparável")
        print("       esta razão fica perto de 1. No bloco 2 deu 0,020.)")
    for k, rot, alvo, maior_e_melhor in (
            ("Metrics/giro_wz_abs", "|ωz| médio (rad/s)", None, False),
            ("Metrics/giro_frac_sem_comando", "fração sem comando de giro", None, True)):
        if k in v:
            marca = ""
            if alvo is not None:
                ok = v[k] >= alvo if maior_e_melhor else v[k] <= alvo
                marca = "  ok" if ok else f"  ⚠ fora de {alvo:g}"
            print(f"    {rot:28s} {v[k]:8.4f}{marca}")

    print()
    print("  BALANÇO DE FORMA (§10.4)")
    print("  " + "-" * 52)
    alvo_loco = v.get("Curriculum/forma/alvo_loco")
    if alvo_loco is None:
        print("    sem `alvo_loco` no log: balanço automático DESLIGADO")
    else:
        if alvo_loco >= 0.999:
            fase = "locomoção PURA (a caixa não existe ainda)"
        elif alvo_loco <= 0.301:
            fase = "regime (a rampa terminou)"
        else:
            fase = "na RAMPA"
        print(f"    alvo de locomoção           {alvo_loco:8.4f}  {fase}")
        print(f"    fração populacional         "
              f"{1.0 - v.get('Curriculum/forma/frac_manipula_pop', 0.0):8.4f}"
              "  (tem de seguir o alvo)")
        # ⚠ O SINAL QUE MOVE A RAMPA. Sem esta linha não há como saber POR QUE o alvo
        # está onde está, e foi essa cegueira que custou o bloco 2. Até 24/08 o sinal
        # era a duração do episódio — sobrevivência — e o alvo descia com o robô
        # imóvel. Agora ele desce só quando a marcha existe.
        razao_m = v.get("Curriculum/forma/razao_marcha_ema")
        if razao_m is not None:
            if razao_m >= 0.50:
                veredito = "abre: a rampa DESCE"
            elif razao_m < 0.40:
                veredito = "⚠ histerese: a rampa SOBE de volta"
            else:
                veredito = "zona morta: a rampa fica parada"
            print(f"    razão de marcha (portão)    {razao_m:8.4f}  {veredito}")
    for k, rot in (("Curriculum/forma/frac_loco_sorteio", "sorteio de locomoção"),
                   ("Metrics/caixa_alvo/aguardando", "fração na espera (§11.2)"),
                   ("Metrics/caixa_alvo/espera_s", "espera restante (s)")):
        if k in v:
            print(f"    {rot:28s} {v[k]:8.4f}")


def escada(v: dict[str, float], iteracao: int) -> None:
    print()
    print("  ESCADA DE CORTE (§17)")
    print("  " + "-" * 52)
    falhou = False
    for it, chave, cmp, alvo, texto in ESCADA:
        if iteracao < it:
            print(f"    it {it:5d}  {chave:38s}  (ainda não vence)")
            continue
        val = v.get(chave)
        if val is None:
            print(f"    it {it:5d}  {chave:38s}  SEM DADO")
            continue
        ok = val >= alvo if cmp == ">=" else val <= alvo
        marca = "ok  " if ok else "FALHA"
        print(f"    it {it:5d}  {chave:38s}  {val:8.4f} {cmp} {alvo:<8.4g} {marca}")
        if not ok:
            falhou = True
            print(f"             -> {texto}")
    if falhou:
        print()
        print("  ⚠ Uma linha falhou. Pare o bloco: o resto das iterações não vai")
        print("    responder a pergunta que a linha faz.")


# ----------------------------------------------------------------- o autoteste
DEMO = {
    "Train/mean_episode_length": 102.68,
    "Train/mean_reward": 9.79,
    "Policy/mean_noise_std": 0.46,
    "Loss/value_function": 0.6129,
    "Episode_Reward/staged": 0.4280,
    "Episode_Reward/track_linear_velocity": 0.1693,
    "Episode_Reward/postura_ereta": 0.1227,
    "Episode_Reward/unload": 0.1222,
    "Episode_Reward/precise_pos": 0.0818,
    "Episode_Reward/upright": 0.0806,
    "Episode_Reward/squeeze": 0.0691,
    "Episode_Reward/pose": 0.0525,
    "Episode_Reward/precise_ori": 0.0305,
    "Episode_Reward/track_angular_velocity": 0.0291,
    "Episode_Reward/sustentacao": 0.0038,
    "Episode_Reward/action_rate_l2": -0.3398,
    "Episode_Reward/joint_vel_hinge": -0.2845,
    "Episode_Reward/body_ang_vel": -0.0087,
    "Episode_Reward/dof_pos_limits": -0.0072,
    "Episode_Reward/angular_momentum": -0.0061,
    "Episode_Reward/foot_swing_height": -0.0019,
    "Episode_Reward/foot_clearance": -0.0017,
    "Episode_Reward/self_collisions": -0.0005,
    "Episode_Reward/foot_slip": -0.0001,
    "Episode_Termination/fell_over": 34.5833,
    "Episode_Termination/time_out": 2.5417,
    "Episode_Termination/contato_ilegal": 0.2500,
    "Episode_Termination/caixa_largada": 0.1250,
    "Metrics/caixa_alvo/frac_manipula": 0.0889,
    "Metrics/caixa_alvo/episode_success": 0.0330,
    "Metrics/caixa_alvo/pegou": 0.0370,
    "Metrics/caixa_alvo/no_alvo": 0.0447,
    "Metrics/caixa_alvo/fecha_de_pe": 0.0536,
    "Metrics/caixa_alvo/fecha_angulo": 0.0242,
    "Metrics/caixa_alvo/fecha_todas": 0.0155,
    "Metrics/peak_height_mean": 0.0042,
    "Metrics/landing_force_mean": 188.9893,
    "Metrics/slip_velocity_mean": 0.0934,
    "Curriculum/forma/dur_loco_ema": 35.2659,
    "Curriculum/forma/dur_manip_ema": 860.8481,
    "Curriculum/command_vel/lin_vel_x_max": 1.0,
    "Curriculum/hinge/weight": -0.1000,
    "Curriculum/action_rate/weight": -0.2500,
}

# O SEGUNDO autoteste: a it 1216 do bloco 2. Ele existe para travar a lição de
# 21/08, e não para reproduzir uma tabela.
#
# O bloco 2 colapsou por GIRO, e a escada não tinha portão de giro. A
# `razao_giro` valia 0,020 desde a iteração 187 — cinquenta vezes abaixo do
# esperado, com pesos IGUAIS e σ MAIOR no giro — e o diagnóstico só saiu na
# iteração 1216, olhando um play. Se algum dia esta entrada passar no portão do
# giro, o portão está errado.
DEMO_GIRO = {
    "Train/mean_episode_length": 41.53,
    "Train/mean_reward": 5.72,
    "Policy/mean_noise_std": 0.8523,
    "Episode_Reward/track_linear_velocity": 0.0754,
    "Episode_Reward/track_angular_velocity": 0.0015,
    "Episode_Termination/fell_over": 69.8817,
    "Episode_Termination/time_out": 3.0662,
    "Episode_Termination/contato_ilegal": 0.4708,
    "Episode_Termination/caixa_largada": 0.0,
    "Metrics/peak_height_mean": 0.0109,
    "Metrics/caixa_alvo/frac_manipula": 0.0280,
    "Metrics/caixa_alvo/episode_success": 0.0,
    "Metrics/caixa_alvo/pegou": 0.0,
    "Metrics/caixa_alvo/erro_posicao": 0.0055,
    "Metrics/caixa_alvo/fecha_angulo": 0.0041,
    "Metrics/caixa_alvo/fecha_de_pe": 0.0048,
    "Metrics/caixa_alvo/fecha_todas": 0.0001,
    "Curriculum/forma/dur_loco_ema": 10.7137,
    "Curriculum/forma/dur_manip_ema": 866.3829,
    "Curriculum/command_vel/lin_vel_x_max": 1.0,
    "Curriculum/nivel/nivel_medio": 0.0,
}


def main(argv: list[str]) -> int:
    if "--demo-giro" in argv:
        print("### DEMO — a it 1216 do bloco 2: o colapso por GIRO ###")
        v = derivados(DEMO_GIRO)
        analisa(v)
        escada(v, 1216)
        razao = v.get("Derivado/razao_giro", 1.0)
        assert razao < 0.10, f"o portão do giro tem de reprovar o bloco 2 (razão {razao})"
        assert DEMO_GIRO["Episode_Termination/caixa_largada"] == 0.0
        print()
        print("  autoteste do giro ok: a razão é "
              f"{razao:.4f} e o portão reprova, como tem de reprovar.")
        return 0
    if "--demo" in argv:
        print("### DEMO — os números medidos na it 5000 do bloco 1 ###")
        d = derivados(DEMO)
        analisa(d)
        escada(d, 5000)
        return 0
    alvo = [a for a in argv if not a.startswith("-")]
    run = acha_run(pathlib.Path(alvo[0])) if alvo else acha_run()
    print(f"### run: {run} ###")
    v, passo = le_run(run)
    v = derivados(v)
    analisa(v)
    escada(v, passo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
