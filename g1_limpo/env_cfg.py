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

ESCOPO DESTA FASE (F0): esqueleto, cena, ação, física, remoções. A tabela de
recompensa é a DO FABRICANTE, sem mudança — quem a ajusta é a F1.
"""
from __future__ import annotations

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg

from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from g1_limpo import cena as C
from g1_limpo import comando as CMD
from g1_limpo import curriculo as CU
from g1_limpo import eventos as EV
from g1_limpo.knobs import ATIVO, Knobs

__all__ = ["make_env_cfg", "colhe_sigmas_de_postura"]


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
) -> ManagerBasedRlEnvCfg:
    """Monta o cfg.

    `inspecao=True`  trava o robô e desliga as terminações. SÓ para revisão visual.
    `elo`            força o elo (`comando.ANDAR`..`comando.BOTAR`). `None` = `PEGAR`,
                     o único que a F0/F1 treina.

    ⚠ A LAJE é movida pelo TERMO DE COMANDO, e não por este cfg: o `ANDAR` e o
    `CARREGAR` a mandam para +5 m, e o `BOTAR` a põe num topo novo. É lá que a F4 vai
    fazer o mesmo no avanço de elo, portanto o caminho é um só.
    """
    k = k or ATIVO
    c = k.cena
    elo_alvo = CMD.PEGAR if elo is None else int(elo)
    anda = elo_alvo in (CMD.ANDAR, CMD.CARREGAR)
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
                "caixa_jitter_yaw_deg": c.caixa_jitter_yaw_deg,
                "caixa_meia_z": c.caixa_meia_aresta[2]},
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

    # O reset da base. Na F0 vale o range de MANIPULAÇÃO, ou o de LOCOMOÇÃO se a
    # mobília estiver afastada. A bifurcação POR ENV entra na F2, junto com o one-hot
    # que define a forma.
    cfg.events["reset_base"].params["pose_range"] = dict(
        c.reset_base_loco if anda else c.reset_base_manipula)

    # ------------------------------------------------------ 3b. o currículo
    # ⚠ ORDEM DE RESET: currículo -> eventos -> comando. O `nivel` escreve
    # `env.limpo_nivel`, e os dois consumidores abaixo o leem. Invertida, a coisa
    # quebra em silêncio.
    cfg.curriculum["nivel"] = CurriculumTermCfg(
        func=CU.nivel,
        params={"n_niveis": n.n_niveis, "forcado": n.forcado},
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
        prateleira_meia_xy=c.prateleira_meia_xy,
        caixa_meia_z=c.caixa_meia_aresta[2],
        ang_max_deg=n.ang_max_deg,
        elo_forcado=elo_alvo,
        debug_vis=True,
    )

    # ---------------------------------------------------- 3d. modo INSPEÇÃO
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
