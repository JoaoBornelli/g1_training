"""Monta o env do g1_limpo sobre o molde de locomoção do FABRICANTE.

⚠ ZERO IMPORT DE CÓDIGO DO PROJETO. Só `mjlab`, que é framework.

Por que o molde e não `ManagerBasedRlEnvCfg` do zero: as três tabelas de σ por
junta do `variable_posture` do G1 existem APENAS dentro de
`unitree_g1_rough_env_cfg` (`mjlab/tasks/velocity/config/g1/env_cfgs.py:107-146`).
Não há constante exportada. Sob "sem importar código do projeto" elas podem ser
COLHIDAS do cfg do fabricante, porque `mjlab` é biblioteca. Redigitá-las custaria
~40 linhas de calibração — 14 padrões de junta × 3 regimes — sem nenhum teste que
pegue um dígito trocado. E um σ de joelho de 0,35 digitado 0,035 não quebra nada:
ele achata o passo, e a run morre 1200 iterações depois num painel.

ESCOPO ATÉ AQUI (F0 + F1): esqueleto, cena, ação, física, remoções, a tabela de
recompensa da locomoção, as métricas de marcha e a `razao_marcha`. O one-hot e os
gates entram na F2; os sete incentivos de manipulação, na F3.
"""
from __future__ import annotations

import dataclasses
import math

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import is_terminated, joint_acc_l2
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from g1_limpo import cena as C
from g1_limpo import comando as CMD
from g1_limpo import curriculo as CU
from g1_limpo import eventos as EV
from g1_limpo import metricas as MT
from g1_limpo import observacoes as OB
from g1_limpo import recompensas as RC
from g1_limpo.knobs import ATIVO, Knobs

__all__ = ["make_env_cfg", "colhe_sigmas_de_postura", "aplica_pesos", "ELO_DE_TREINO"]

# ⚠ O ELO QUE O TREINO ABRE, e ele é da FASE, não um número de ajuste — por isso vive
# aqui e não no `knobs.py`.
#
# F1 = LOCOMOÇÃO PURA. A mobília sobe a +5 m, `valida = 0`, e não existe alvo de caixa.
# É a hipótese central do módulo, e ela foi VALIDADA por medição: o `g1_multitask`
# andou porque a fase 1 dele não tinha manipulação nenhuma (fatia de 100% para a
# locomoção), e o `g1_poc` não andou porque entregou 70% das transições à manipulação
# por volta da iteração 420, com o robô imóvel.
#
# Desde a F2 ele é o elo da MAIORIA, não de todos: o `curriculo.sorteia_elo` dá
# `fatia_loco` dos envs a ele e o resto aos elos sorteáveis. A F5 põe o controlador.
ELO_DE_TREINO = CMD.ANDAR

# Os elos com twist ATIVO. Eles governam três coisas de uma vez, e é de propósito que
# a lista seja uma só: a faixa de yaw do reset da base, a neutralidade da postura, e
# quais elos NÃO têm o twist zerado pelo comando.
ELOS_QUE_ANDAM = (CMD.ANDAR, CMD.CARREGAR)

# Os elos que o sorteio pode entregar num RESET. O `CARREGAR` e o `BOTAR` ficam fora
# porque começam com a caixa NAS MÃOS — eles só existem como 2º elo de uma cadeia, e
# isso é F4. Ver `curriculo.sorteia_elo` para o preço declarado (os slots 3 e 4 do
# one-hot ficam constantes até lá) e a mitigação pré-registrada.
ELOS_SORTEAVEIS = (CMD.REORIENTAR, CMD.PEGAR)


def aplica_pesos(cfg, r) -> None:
    """Escreve a tabela de pesos da F1 sobre a do molde, termo a termo.

    ⚠ O `assert` é o ponto. Um nome de termo que o `mjlab` renomeie num upgrade
    passaria como `cfg.rewards["nome_velho"].weight = x` num dict novo — criando um
    termo ÓRFÃO sem `func`, ou pior, deixando o termo real com o peso do molde. O
    `assert` transforma isso num erro na montagem, e não num painel estranho 3000
    iterações depois.
    """
    for nome, peso in dataclasses.asdict(r).items():
        if nome == "altura_de_balanco":      # não é peso, é o alvo em metros
            continue
        assert nome in cfg.rewards, (
            f"termo de recompensa '{nome}' não existe no molde; "
            f"existentes: {sorted(cfg.rewards)}")
        cfg.rewards[nome].weight = peso


def colhe_sigmas_de_postura(cfg) -> dict:
    """Colhe os três dicionários de σ do G1 do cfg do fabricante.

    Eles são calibrados POR ROBÔ e NÃO são redigitados aqui. O `smoke.py` prova por
    identidade de objeto que eles foram colhidos, e não copiados.
    """
    p = cfg.rewards["pose"].params
    return {
        "std_standing": p["std_standing"],
        "std_walking": p["std_walking"],
        "std_running": p["std_running"],
    }


def make_env_cfg(
    k: Knobs | None = None,
    play: bool = False,
    *,
    inspecao: bool = False,
    elo: int | None = None,
    cadeia: int | None = None,
    avanca_apos_s: float | None = None,
) -> ManagerBasedRlEnvCfg:
    """Monta o cfg.

    `inspecao=True`  trava o robô e desliga as terminações. SÓ para revisão visual.
    `elo`            força o elo (`comando.ANDAR`..`comando.BOTAR`). `None` =
                     `ELO_DE_TREINO`, que na F1 é o `ANDAR`.

    ⚠ A LAJE é movida pelo TERMO DE COMANDO, e não por este cfg: o `ANDAR` e o
    `CARREGAR` a mandam para +5 m, e o `BOTAR` a põe num topo novo. É lá que a F4 vai
    fazer o mesmo no avanço de elo, portanto o caminho é um só.
    """
    k = k or ATIVO
    c = k.cena
    # ⚠ `elo` EXPLÍCITO e `ELO_DE_TREINO` são coisas diferentes, e confundi-los mata a
    # F2 em silêncio. `elo=X` é o inspetor e o `play`: ali o elo é FORÇADO, igual em
    # todos os envs. `elo=None` é o TREINO: ali o elo é SORTEADO por env, e forçar o
    # `ELO_DE_TREINO` no comando anularia o sorteio inteiro sem uma linha de erro.
    elo_explicito = elo is not None
    elo_alvo = ELO_DE_TREINO if elo is None else int(elo)
    anda = elo_alvo in ELOS_QUE_ANDAM
    segurando = elo_alvo in (CMD.CARREGAR, CMD.BOTAR)

    # ------------------------------------------------ 0. a fundação do fabricante
    # `unitree_g1_flat_env_cfg` já resolve os sítios, os grupos de colisão e os σ do
    # G1, e a variante `flat` já remove de graça o `terrain_scan`, o `height_scan`, o
    # `out_of_terrain_bounds` e o currículo `terrain_levels`. Isso não é desvio do
    # fabricante: é a receita flat dele.
    cfg = unitree_g1_flat_env_cfg(play=play)

    # ------------------------------------------------------------- 1. a cena
    cfg.scene.entities = C.entidades(k)
    cfg.scene.extent = c.extent

    # ⚠ OS SENSORES DO FABRICANTE FICAM TODOS. Os nossos são ADIÇÃO, não substituição.
    #
    # Eu tentei substituir o `feet_ground_contact` e o `self_collision` pelos nossos
    # (`pes_chao` e `auto_colisao`, que têm força). Isso EXPLODE na montagem: o termo
    # de observação `foot_air_time` DO FABRICANTE referencia `feet_ground_contact`
    # POR NOME, e o reward `self_collisions` referencia `self_collision`. O g1_poc
    # sobrevive a essa remoção porque ele também apaga os termos de obs dependentes —
    # mas isso MUDA a observação, e o objetivo da F1 é reproduzir a locomoção do
    # fabricante, não uma variante dela.
    #
    # Preço de manter os dois: um sensor de contato de pé duplicado (o booleano dele
    # + o nosso com força). É barato, e mantém intacto o contrato que fez o robô
    # andar.
    #
    # O `foot_height_scan` lê o grupo 0, e a mobília está no grupo 2 (`cena.regroup`),
    # portanto ele não vê a prateleira como chão.
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + C.sensores()

    # física de manipulação. Cicatriz de 15/07: `elliptic` com `impratio=10` divergiu
    # para NaN no reset parcial; `pyramidal` com 1,0 é o par que roda.
    cfg.sim.njmax = c.njmax
    cfg.sim.nconmax = c.nconmax
    cfg.sim.mujoco.impratio = c.impratio
    cfg.sim.mujoco.cone = c.cone

    # ------------------------------------------------------------ 2. a ação
    # ⚠ `G1_ACTION_SCALE` tem 16 chaves, e elas são PADRÕES REGEX, não nomes de
    # junta. Multiplicar o dict inteiro preserva a calibração relativa entre grupos.
    acao = cfg.actions["joint_pos"]
    assert isinstance(acao, JointPositionActionCfg), type(acao)
    acao.scale = {padrao: v * c.escala_acao_mult
                  for padrao, v in G1_ACTION_SCALE.items()}

    # ------------------------------------------ 2b. a recompensa da locomoção (F1)
    # Os DOIS termos que o molde não tem, vindos do `g1_multitask` — o módulo que
    # ANDOU. Entram antes de `aplica_pesos`, que exige que todo nome exista.
    #
    # ⚠ `terminacao = −200` NÃO custa 200. Com `scale_rewards_by_dt = True` e
    # `dt = 0,02`, o passo que termina paga `−200 × 0,02 = −4,0`. E `is_terminated`
    # exclui o `time_out` (ele lê `termination_manager.terminated`), portanto
    # terminar o episódio pelo tempo NÃO é punido — só cair é.
    cfg.rewards["terminacao"] = RewardTermCfg(func=is_terminated, weight=0.0)
    cfg.rewards["joint_acc"] = RewardTermCfg(func=joint_acc_l2, weight=0.0)

    # ⚠ O `feet_swing_height` do fabricante NÃO tem `reset`, e `reward_manager.py:174`
    # só chama `reset` em termo de classe que tenha. Logo o `peak_heights` dele
    # atravessa o fim do episódio, e o pico do pé que estava no ar quando o robô caiu
    # entra no episódio seguinte. Ver `recompensas.AlturaDeBalanco`.
    cfg.rewards["foot_swing_height"].func = RC.AlturaDeBalanco
    cfg.rewards["foot_swing_height"].params["target_height"] = k.recompensa.altura_de_balanco

    # ⚠ A POSTURA FICA NEUTRA NOS ELOS DE MANIPULAÇÃO (F2). Não é um 4º regime de σ —
    # medi, e nenhum σ resolve. Com o twist em zero o regime é SEMPRE `standing`
    # (`walking_threshold` do G1 é 0,05, medido), cujo σ é uma entrada só, `.*` = 0,05
    # para as 29 juntas. A 10% da faixa de junta o termo já vale 0,000, com GRADIENTE
    # ZERO — é canal morto, não penalidade forte. Nem `running×5` sobrevive a 40%.
    # Ver `recompensas.PosturaPorElo` para a tabela medida.
    cfg.rewards["pose"].func = RC.PosturaPorElo
    cfg.rewards["pose"].params.update(
        canal_do_elo=CMD.ELO, nome_do_comando="alvo_caixa",
        elos_que_andam=ELOS_QUE_ANDAM)

    aplica_pesos(cfg, k.recompensa)

    # ------------------------------------------------ 2c. as métricas de marcha
    # ⚠ Elas saem de DENTRO dos termos de recompensa de propósito:
    # `reward_manager.py:122` pula termo com peso 0, portanto a medição do fabricante
    # morre junto com o incentivo. O `air_time` já vem com peso 0,0 no molde — o
    # `Metrics/air_time_mean` dele NÃO EXISTE. Ver `metricas.py`.
    #
    # O `mean_action_acc` do molde FICA: ele já é `MetricsTermCfg` e não depende de peso.
    cfg.metrics.update(MT.termos())

    # ------------------------------------------ 2d. a régua: `razao_marcha`
    # ⚠ O twist é RECONSTRUÍDO como subclasse, campo a campo por `dataclasses.fields`,
    # e não editado. Copiar campos à mão perderia em silêncio qualquer campo que um
    # upgrade de `mjlab` adicione — e um `rel_standing_envs` perdido mudaria 10% dos
    # envs sem uma linha de log.
    antigo = cfg.commands["twist"]
    campos = {f.name: getattr(antigo, f.name)
              for f in dataclasses.fields(antigo)}
    cfg.commands["twist"] = CMD.TwistComRazaoDeMarchaCfg(
        **campos, limiar_comando=k.marcha.limiar_comando)

    # -------------------------------------------------------- 3. os eventos
    # ⚠ `base_com` (`dr.body_com_offset`) SAI, e não é preferência: ele corrompe a
    # heap em CPU E em GPU (illegal memory access), provado por A/B com 256 envs e
    # `CUDA_LAUNCH_BLOCKING=1`, e derruba a task do PRÓPRIO fabricante. O preço é
    # perder ±2,5 cm de randomização de CoM no torso, e o preço está declarado.
    cfg.events.pop("base_com", None)

    # Sem evento de reset, uma entidade fica no world-origin para TODOS os envs.
    #
    # ⚠ UM evento por entidade. Dois eventos que escrevem a pose da MESMA entidade no
    # MESMO reset não se somam: o segundo APAGA o primeiro, sem erro e sem log.
    # Portanto `posiciona_cena` faz a prateleira E a caixa, e não existe um segundo
    # evento tocando nenhuma das duas.
    n = k.nivel
    cfg.events["posiciona_cena"] = EventTermCfg(
        func=EV.posiciona_cena, mode="reset",
        params={"topo_min": n.topo_min, "jitter_x_max": n.jitter_x_max,
                "topo_teto": c.prateleira_topo_teto, "jitter_z": c.prateleira_jitter_z,
                "prateleira_xy": c.prateleira_xy,
                "prateleira_meia_z": c.prateleira_meia_z,
                "caixa_xy": c.caixa_xy, "caixa_jitter_y": c.caixa_jitter_y,
                "caixa_meia_z": c.caixa_meia_aresta[2],
                "voltas_max": n.voltas_max, "eixo_vertical": n.eixo_vertical,
                "desalinho_max_deg": n.desalinho_max_deg},
    )
    cfg.events["carga_caixa"] = EventTermCfg(
        func=EV.carga_caixa, mode="reset",
        params={"carga_max": n.carga_max, "massa_base": c.caixa_massa},
    )
    if segurando:
        # ⚠ SÓ PARA INSPEÇÃO: o `carregar` e o `botar` começam com a caixa nas mãos,
        # porque no treino eles são o 2º elo de uma cadeia. Roda DEPOIS do
        # `posiciona_cena` (dict ordenado por inserção), portanto ele reposiciona.
        cfg.events["segura_caixa"] = EventTermCfg(
            func=EV.segura_caixa, mode="reset",
            params={"peito_b": k.alvo.peito_b},
        )
        if inspecao:
            # ⚠ E na inspeção ela é PINADA a cada passo. Nada a segura de verdade:
            # sem o pino ela cai em ~0,4 s, empurra o robô, e o clamp do `botar`
            # (que mede o FUNDO da caixa) passa a medir uma caixa no chão.
            dt_ = cfg.sim.mujoco.timestep * cfg.decimation
            cfg.events["pina_caixa"] = EventTermCfg(
                func=EV.segura_caixa, mode="interval",
                interval_range_s=(dt_, dt_),
                params={"peito_b": k.alvo.peito_b},
            )

    # O reset da base, POR ENV desde a F2. O evento do fabricante é substituído por um
    # DESPACHANTE que o chama uma vez por subconjunto de elo — ele não reimplementa
    # amostragem nenhuma (ver `eventos.reset_base_por_elo`).
    #
    # ⚠ Com `elo` forçado (inspetor, play) o buffer de elo é uniforme, portanto o
    # despachante cai num único subconjunto e o comportamento é o de antes.
    cfg.events["reset_base"] = EventTermCfg(
        func=EV.reset_base_por_elo, mode="reset",
        params={"elos_que_andam": ELOS_QUE_ANDAM,
                "faixa_loco": dict(c.reset_base_loco),
                "faixa_manipula": dict(c.reset_base_manipula),
                "velocidade": {}},
    )
    _ = anda

    # ------------------------------------------------------ 3b. o currículo
    # ⚠ ORDEM DE RESET: currículo -> eventos -> comando. O `nivel` escreve
    # `env.limpo_nivel`, e os dois consumidores abaixo o leem. Invertida, a coisa
    # quebra em silêncio.
    # ⚠⚠ A ORDEM DESTE DICT É CONTRATO, E ELA MUDOU NA F5. Ela é
    #
    #        command_vel  ->  forma  ->  nivel  ->  elo
    #
    # e a razão é que o `forma` e o `nivel` medem o episódio que ACABOU, enquanto o
    # `elo` escreve o do episódio que COMEÇA. Os dois primeiros precisam ler
    # `env.limpo_elo` ANTES de o terceiro sobrescrevê-lo:
    #
    #   · o `forma` atribui as durações medidas ao lado certo (loco ou manipulação);
    #   · o `nivel` precisa saber se o episódio era de locomoção para NÃO mover o nível
    #     por causa dele.
    #
    # ⚠ Na F2 o `elo` vinha primeiro e não havia problema, porque ninguém lia o elo
    # antigo. A F5 introduziu dois leitores, e com a ordem velha os dois lêem o elo do
    # episódio SEGUINTE. É o mesmo bug medido em 20/08 com o `nivel` e a `forma`: a
    # probabilidade de subir caía de `p` para `0,7·p`, o ponto fixo saía de 0,5 para
    # 0,714, e um episódio de LOCOMOÇÃO rebaixava o nível em 70% das vezes.
    #
    # O `elo` continua sendo termo de CURRÍCULO (e não evento) porque o reset de pose e
    # o alvo o leem — e todo termo de currículo roda antes de todo evento.
    cfg.curriculum["forma"] = CurriculumTermCfg(
        func=CU.forma,
        params={"f": k.forma, "elo_loco": CMD.ANDAR, "nome_do_twist": "twist"},
    )
    cfg.curriculum["nivel"] = CurriculumTermCfg(
        func=CU.nivel,
        params={"n_niveis": n.n_niveis, "forcado": n.forcado,
                "frac_uniforme": k.piso.frac_nivel_uniforme,
                "nome_do_comando": "alvo_caixa"},
    )
    cfg.curriculum["elo"] = CurriculumTermCfg(
        func=CU.sorteia_elo,
        params={"elo_loco": CMD.ANDAR, "elos_manip": ELOS_SORTEAVEIS,
                "fatia_loco": k.forma.fatia_loco,
                "forcado": elo_alvo if elo_explicito else None},
    )

    # -------------------------------------------------------- 3c. o comando
    # ⚠ O comando é a ÚNICA fonte de verdade do alvo, e o desenho de debug mora
    # DENTRO dele. Um visualizador que reimplementasse o sorteio seria uma segunda
    # fonte, e ela mentiria no dia em que as duas divergissem.
    cfg.commands["alvo_caixa"] = CMD.AlvoCaixaCmdCfg(
        peito_b=k.alvo.peito_b,
        altura_carregar=k.alvo.altura_carregar,
        botar_x=k.alvo.botar_x, botar_y=k.alvo.botar_y,
        botar_topo_piso=k.alvo.botar_topo_piso,
        botar_topo_teto=k.alvo.botar_topo_teto,
        botar_folga_laje=k.alvo.botar_folga_laje,
        afasta_z=c.afasta_z,
        prateleira_xy=c.prateleira_xy,
        prateleira_meia_z=c.prateleira_meia_z,
        prateleira_topo_piso=c.prateleira_topo_piso,
        prateleira_meia_xy=c.prateleira_meia_xy,
        caixa_meia_z=c.caixa_meia_aresta[2],
        face_alvo_b=c.face_alvo_b,
        sitios_palma=C.PALM_SITES,
        caixa_meia_aresta=c.caixa_meia_aresta[0],
        sigma_fator=k.tarefa.sigma_fator,
        sigma_min=k.tarefa.sigma_min,
        # ------------------------------------------------------- a cadeia (F4)
        # ⚠ SEM ESTAS LINHAS A MÁQUINA DE ELO É INERTE, e em silêncio: o campo
        # `prob_por_nivel` do cfg tem default `()`, e o sorteio cai no ramo "não há
        # cadeia". Foi o que aconteceu na primeira entrega — toda cadeia saía 0 e
        # nenhum env avançava, sem nenhum erro.
        prob_por_nivel=k.cadeia.prob_por_nivel,
        sustenta_pegar_s=k.cadeia.sustenta_pegar_s,
        sustenta_outros_s=k.cadeia.sustenta_outros_s,
        carregar_s=k.cadeia.carregar_s,
        # as tolerâncias de FECHAMENTO são as mesmas da régua de sustentação da F3:
        # um elo que "fecha" com tolerância diferente da que a recompensa paga
        # ensinaria duas coisas contraditórias.
        tol_pos=k.tarefa.tol_pos,
        tol_ang_deg=k.tarefa.tol_ang_deg,
        pelve_alvo=k.tarefa.pelve_alvo,
        nome_sensor_apoio=C.SENSOR_APOIO,
        elo_forcado=elo_alvo if elo_explicito else None,
        # ⚠ A cadeia forçada, para o inspetor e o play. Ela VENCE o `elo_forcado`: o elo
        # passa a ser o 1º da cadeia. Os dois juntos são pedidos contraditórios, e é a
        # cadeia que manda porque ela carrega mais informação.
        cadeia_forcada=cadeia,
        debug_vis=True,
    )

    # ------------------------------------------------- 3e. o one-hot (F2)
    # ⚠ POR ÚLTIMO, e nos DOIS grupos, na MESMA ordem. Canal novo é APPEND de colunas;
    # uma inserção no meio desloca todo peso da primeira camada em silêncio.
    #
    # ⚠ Sem `noise` e sem `scale`: ruído num one-hot produziria frações entre slots,
    # isto é, estados que não existem.
    for grupo in ("actor", "critic"):
        cfg.observations[grupo].terms["elo"] = ObservationTermCfg(
            func=OB.um_de_cinco,
            params={"command_name": "alvo_caixa", "canal_do_elo": CMD.ELO},
        )

    # ------------------------------------------------- 3f. a caixa na obs (F3)
    # ⚠ DEPOIS do one-hot, pelo mesmo contrato de append. E nos dois grupos.
    for grupo in ("actor", "critic"):
        cfg.observations[grupo].terms["caixa"] = ObservationTermCfg(
            func=OB.caixa_no_frame_da_base,
            params={"command_name": "alvo_caixa"},
        )

    # ------------------------------------------- 3g. os sete incentivos (F3)
    # ⚠ TODOS positivos e contínuos (R3), e todos gateados por `VALIDA` — sem o gate
    # um env de `ANDAR` pagaria o MÁXIMO, porque com os canais de caixa zerados
    # `exp(0) = 1`.
    tr = k.tarefa
    _cmd = "alvo_caixa"
    cfg.rewards["staged"] = RewardTermCfg(
        func=RC.staged, weight=tr.staged,
        params={"nome_do_comando": _cmd})
    cfg.rewards["precise_pos"] = RewardTermCfg(
        func=RC.precise_pos, weight=tr.precise_pos,
        params={"nome_do_comando": _cmd, "sigma": tr.precise_pos_sigma})
    cfg.rewards["precise_ori"] = RewardTermCfg(
        func=RC.precise_ori, weight=tr.precise_ori,
        params={"nome_do_comando": _cmd})
    cfg.rewards["squeeze"] = RewardTermCfg(
        func=RC.squeeze, weight=tr.squeeze,
        params={"nome_do_comando": _cmd, "sensores": C.SENSOR_PALMA,
                "forca_ref": tr.forca_ref})
    cfg.rewards["unload"] = RewardTermCfg(
        func=RC.unload, weight=tr.unload,
        params={"nome_do_comando": _cmd, "sensor_apoio": C.SENSOR_APOIO})
    cfg.rewards["postura_ereta"] = RewardTermCfg(
        func=RC.postura_ereta, weight=tr.postura_ereta,
        params={"nome_do_comando": _cmd, "sensores_palma": C.SENSOR_PALMA,
                "sensor_apoio": C.SENSOR_APOIO, "forca_ref": tr.forca_ref,
                "pelve_alvo": tr.pelve_alvo, "pelve_piso": tr.pelve_piso})
    cfg.rewards["sustentacao"] = RewardTermCfg(
        func=RC.sustentacao, weight=tr.sustentacao,
        params={"nome_do_comando": _cmd, "tol_pos": tr.tol_pos,
                "tol_ang": math.radians(tr.tol_ang_deg),
                "sustenta_s": tr.sustenta_s})

    # ---------------------------------------------------- 3d. modo INSPEÇÃO
    if inspecao or play:
        # ⚠ O CONTROLADOR DE FATIA FICA DESLIGADO na inspeção e no play. Ele é um laço
        # fechado que se move a cada iteração; num inspetor ele faria a fatia mudar
        # entre duas invocações e a tabela deixaria de ser reproduzível.
        import dataclasses as _dc
        cfg.curriculum["forma"].params["f"] = _dc.replace(k.forma, controla=False)
    if inspecao and avanca_apos_s is not None:
        # ⚠ O AVANÇO DE ELO NO VISUALIZADOR, como EVENTO DE INTERVALO. O `run_play` do
        # mjlab roda o próprio laço e não expõe gancho por passo — foi por isso que a
        # primeira tentativa deixou o `--avanca-elo` como no-op. O evento é o mesmo
        # idioma do `trava_robo`, e o `forca_avanco` é idempotente.
        #
        # O primeiro disparo em `avanca_apos_s` dá tempo de ver o estado ANTES.
        cfg.events["avanca_elo"] = EventTermCfg(
            func=EV.avanca_elo_no_viewer, mode="interval",
            interval_range_s=(avanca_apos_s, avanca_apos_s),
            params={"nome_do_comando": "alvo_caixa"},
        )

    if inspecao:
        # ⚠ SÓ PARA INSPEÇÃO. Trava o robô na pose de reset, para a cena ficar
        # PARADA enquanto se confere alvo e eixo. Sem isto um robô sem política cai
        # em meio segundo.
        #
        # O intervalo é o próprio `dt`, portanto o evento dispara a CADA passo.
        dt = cfg.sim.mujoco.timestep * cfg.decimation
        cfg.events["trava_robo"] = EventTermCfg(
            func=EV.trava_robo, mode="interval",
            interval_range_s=(dt, dt),
            params={},
        )
        # sem terminação: o robô travado dispararia `fell_over` ou `nonfinite` e o
        # episódio reiniciaria sem parar
        cfg.terminations = {}

    # ------------------------------------------------------- 4. ramo de play
    if play:
        # ⚠ `randomize_terrain` entra no FIM do dict de eventos, portanto roda DEPOIS
        # dos eventos de cena, e mexe na ORIGEM do env — ele dessincroniza mobília de
        # pose absoluta.
        cfg.events.pop("randomize_terrain", None)
        # ⚠ `commands_vel` MUTA o cfg COMPARTILHADO: `CommandManager.__init__` faz
        # `self.cfg = cfg` SEM deepcopy (ao contrário do `RewardManager`, que faz), e
        # `commands_vel` escreve em `cfg.ranges.lin_vel_x`. No play ele reescreve a
        # faixa a cada reset e APAGA qualquer velocidade pinada à mão.
        cfg.curriculum.pop("commands_vel", None)
        cfg.curriculum.pop("command_vel", None)   # o nome varia por versão do mjlab

    return cfg
