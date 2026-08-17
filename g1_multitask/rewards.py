"""Recompensas do multi-tarefa.

Divisão deliberada entre o que é REUSO e o que é escrito aqui:

- **Locomoção: reuso total.** Os 5 termos de marcha do fabricante entram por
  config, apontados pro comando `"twist"`, sem uma linha reescrita. Eles são
  código testado e a semântica não mudou.
- **Kernels de manipulação: reuso como PRIMITIVA.** `height_kernel`,
  `reaching_kernel`, `_grasp`, `_contact`, `_box_target_err_sq` vêm de
  `g1_training/skills/lift/rewards.py` por import.
- **Termos de tarefa: escritos aqui.** Os NÚMEROS da Lift foram calibrados pra UMA
  tarefa com o robô parado; numa política que também anda a relação entre eles
  muda. E `botar`/`reorientar` não existem em lugar nenhum. Ver `calibra.py`.

Contato e dinâmica da caixa podem entrar aqui — reward não roda no robô real. O que
NÃO pode é entrar na observação do ator.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME
from mjlab.entity import Entity
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.utils.lab_api.math import quat_apply

from g1_training.skills.lift.rewards import _grasp, height_kernel

from .tasks import LEVELS as _LEVELS
from .tasks import ONEHOT_DIM

T_LEVELS_ALVO = _LEVELS["alvo"]

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

ONEHOT = slice(9, 17)
ALVO = slice(0, 3)


# --------------------------------------------------------------------- gate (T7)
class gated:
    """Envolve outro termo com a máscara das tarefas em `tasks` (item 20).

    Gate por MÁSCARA, nunca por peso. Dois motivos: por peso o termo desapareceria
    do log (o `RewardManager` pula termo com `weight == 0.0`) e o anelamento não
    teria o que mutar; por máscara o valor continua sendo computado e logado, só
    não pontua fora do escopo.

    A máscara sai do MESMO one-hot que o log por tarefa usa (`observability.py`) —
    uma fonte só, senão gate e log podem discordar sobre qual tarefa estava ativa.

    Desde 06/08 ela é um vetor de PESO e não mais 0/1: fora do escopo continua zero,
    dentro vale o fator da tarefa (`escala`). O motivo está no `__call__`.

    **Classe e não função** porque vários termos do mjlab (`posture`,
    `variable_posture`, `electrical_power_cost`, `feets_swing_height`) são CLASSES
    que o manager auto-instancia com `(cfg, env)` pra resolver dicionários de `std`
    em tensor. Uma função-wrapper chamaria `inner(env, ...)` e construiria uma
    instância em vez de chamar o termo — que foi exatamente o `TypeError:
    posture.__init__() got an unexpected keyword argument 'std'` que apareceu aqui.
    O manager instancia ESTA classe, e ela instancia a de dentro.
    """

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        alvo = cfg.params["inner"]
        # o termo interno recebe o MESMO cfg: as classes do mjlab leem só as chaves
        # que lhes interessam de `cfg.params` e toleram as nossas (`inner`, `tasks`).
        self._inner = alvo(cfg=cfg, env=env) if isinstance(alvo, type) else alvo
        # A máscara é um VETOR DE PESO, não um booleano: fora do escopo é 0, dentro é
        # o fator da tarefa (`escala`, default 1.0). Ver `escala` no `__call__`.
        escala = cfg.params.get("escala") or {}
        self._peso = torch.zeros(ONEHOT_DIM, device=env.device)
        for t in cfg.params["tasks"]:
            self._peso[t] = float(escala.get(t, 1.0))
        self._exige_grasp = torch.tensor(
            list(cfg.params.get("exige_grasp", ())), dtype=torch.long,
            device=env.device)

    def __call__(self, env: "ManagerBasedRlEnv", inner, tasks,
                 gate_command: str = "lift_target",
                 exige_grasp: tuple[int, ...] = (),
                 escala: dict[int, float] | None = None,
                 grasp_palm=None, grasp_back=None, **kw) -> torch.Tensor:
        """`gate_command` e NÃO `command_name`: vários termos internos (`lift_reward`,
        `hold_still_bonus`, `orienta_face`) têm um `command_name` PRÓPRIO, e usar a
        mesma chave fazia o gate engolir o parâmetro do termo de dentro —
        `TypeError: lift_reward() missing 1 required positional argument`.

        `exige_grasp` (06/08) multiplica o termo pela PREENSÃO, mas só nas tarefas
        listadas ali. Nas outras o fator é 1.

        ⚠️ Ele existe por causa de um hack medido: em `parado c/ caixa` e
        `andar c/ caixa`, `track_linear_velocity` e `track_angular_velocity` valem
        2.0 + 2.0 com o robô imóvel e o comando em zero. São 4.0 de um teto de 7.0 —
        **79% do orçamento pago por ficar em pé**. O robô podia aninhar a caixa no vão
        dos antebraços contra o tronco, ficar parado, e receber 5.5 de 7.0 com
        `_grasp = 0`, sem manipular nada e sem risco. A caixa fica acima do
        `largou_z`, então nem o `largou` terminava.

        Com o fator, o piso só existe se ele estiver de fato segurando.

        `escala` (06/08) é `{tarefa: fator}`. Ele multiplica o termo POR TAREFA, e
        existe porque um `RewardTermCfg` tem UM peso só enquanto os termos são
        compartilhados: `box_at_peito` vale no `pegar` e nas duas tarefas com caixa,
        `track_*` valem em quatro. Sem fator por tarefa não há como igualar orçamento
        — mexer no peso do termo move todas as tarefas dele juntas.

        Quem preenche é o `_equaliza_orcamento` do `env.py`, por cálculo. Ausente, o
        fator é 1.0 e o gate volta a ser a máscara binária de antes."""
        del inner, tasks, escala              # já resolvidos no __init__
        onehot = env.command_manager.get_term(gate_command).command[:, ONEHOT]
        mascara = onehot @ self._peso
        if len(self._exige_grasp):
            g = _grasp(env, grasp_palm, grasp_back)
            m_g = onehot[:, self._exige_grasp].sum(dim=-1)
            mascara = mascara * (1.0 - m_g + m_g * g)
        return self._inner(env, **kw) * mascara

    def reset(self, env_ids=None):
        """Repassa o reset pro termo de dentro, se ele tiver estado.

        O `RewardManager` coleta pra reset só quem tem `reset` (`:174`), e o que ele
        vê é ESTE objeto — não o de dentro. Sem este repasse, envolver um termo com
        estado (o `feets_swing_height` guarda o pico de altura do passo, por exemplo)
        perderia o reset dele em silêncio. Hoje o único termo-classe que envolvemos é
        o `posture`, que não tem estado; isto é pra não virar armadilha depois."""
        interno = getattr(self._inner, "reset", None)
        if callable(interno):
            interno(env_ids=env_ids)


def soft_landing_rotulado(env: "ManagerBasedRlEnv", rotulo: str, **kw
                          ) -> torch.Tensor:
    """`soft_landing` do fabricante com a chave de log ROTULADA (06/08).

    ⚠️ O termo do fabricante escreve `env.extras["log"]["Metrics/landing_force_mean"]`
    com o nome FIXO (`velocity/mdp/rewards.py:373`). Nós instalamos DOIS: um nos pés e
    um no impacto do tronco contra a mesa. Os nomes dos TERMOS são distintos — o
    comentário do `env.py` diz isso e está certo — mas a chave de LOG é a mesma nos
    dois, e o `RewardManager` avalia na ordem de inserção. O `soft_landing_table` roda
    depois, então o número publicado como "força de pouso" é sempre o do TRONCO.

    Não muda recompensa nenhuma: os dois pesos são ~0 (−1e−5 e −1e−4). É observabilidade,
    e é o que se lê entre blocos — uma métrica que mente sobre qual corpo bateu custa
    mais que as quatro linhas daqui.

    Renomeia DEPOIS da chamada em vez de reescrever a função: o cálculo é do fabricante
    e não há motivo para copiá-lo por causa de uma string."""
    custo = vel_mdp.soft_landing(env, **kw)
    log = env.extras["log"]
    if "Metrics/landing_force_mean" in log:
        log[f"Metrics/landing_force_{rotulo}"] = log.pop("Metrics/landing_force_mean")
    return custo


def action_rate_l2_juntas(env: "ManagerBasedRlEnv", n_juntas: int = 29
                          ) -> torch.Tensor:
    """`action_rate_l2` do fabricante, mas só nos canais de JUNTA (item 6).

    A ação tem 49 números: 29 de residual de junta e 20 de comportamento (`c`). O
    termo do fabricante soma a diferença passo-a-passo dos **49**
    (`mjlab/envs/mdp/rewards.py:63`), então ele cobra jitter nos 20 canais de `c`.

    Com `ESCALA_C = 0` esses 20 canais **não fazem nada**, e cobrá-los é pura
    distorção: a política paga por ruído em dimensões sem efeito. E a saída dela para
    esse custo é encolher o `std` de TODOS os canais, inclusive dos 29 que importam —
    medido na run de 31/07, `std` caindo de 0,96 para 0,70 enquanto o episódio
    encurtava de 765 para 18 passos.

    O sinal de que o custo era ruído: `action_rate_l2` marcava −3,07 / −3,10 / −3,10 /
    −3,14 em quatro tarefas de física completamente diferente. Termo que não distingue
    andar de agachar não está medindo comportamento.

    ⚠️ **Substituída pela `ActionRateJuntas` em 17/08**, que faz o mesmo e ainda pesa
    os canais de braço. Fica aqui como referência da versão sem peso."""
    a = env.action_manager.action[:, :n_juntas]
    p = env.action_manager.prev_action[:, :n_juntas]
    return torch.sum(torch.square(a - p), dim=-1)


class ActionRateJuntas:
    """`action_rate_l2` nos canais de junta, com FATOR próprio nos braços. (17/08)

    O termo plano cobrava −0,88 no `pegar` contra 1,07 de todo o sinal de tarefa
    coletado — **82%** — numa tarefa cujo conteúdo é mover os braços. E a saída da
    política para esse custo é encolher o `std`, o que reduz a exploração exatamente
    onde ela falta.

    Só os braços recebem o fator. O peso global fica: o termo existe para conter jitter
    de perna na marcha, e ali ele funciona (`contrib/locomover/action_rate = −1,02` com
    a locomoção fechada).

    **Classe e não função** porque os índices de junta precisam do `env`, e resolvê-los
    a cada passo custaria uma busca por regex por chamada.

    ⚠️ **Ela confere que a ordem da AÇÃO bate com a ordem das JUNTAS.** Os índices vêm
    de `find_joints`, que devolve a ordem do modelo; o vetor de ação segue a ordem do
    termo de ação. Hoje as duas coincidem (`target_ids == range(29)`), mas um dia o
    termo pode passar a mirar um subconjunto — e aí pesar pelo índice errado
    penalizaria a junta errada, em silêncio. Daí o assert."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        p = cfg.params
        self._n = int(p.get("n_juntas", 29))
        fator = float(p.get("fator_bracos", 1.0))
        padroes = tuple(p.get("padroes_bracos", ()))
        peso = torch.ones(self._n, device=env.device)
        if fator != 1.0 and padroes:
            termo = env.action_manager.get_term("joint_pos")
            alvos = getattr(termo, "target_ids", None)
            if alvos is not None:
                esperado = list(range(self._n))
                reais = [int(i) for i in list(alvos)[: self._n]]
                assert reais == esperado, (
                    "a ordem da ação não é a ordem das juntas — pesar por índice "
                    f"penalizaria a junta errada. target_ids[:{self._n}] = {reais}")
            idx, nomes = env.scene["robot"].find_joints(list(padroes))
            idx = [i for i in idx if i < self._n]
            assert idx, f"nenhuma junta casou {padroes}"
            peso[idx] = fator
            print(f"[REWARD] action_rate: {len(idx)} canais de braço × {fator}")
        self._peso = peso

    def __call__(self, env: "ManagerBasedRlEnv", **kw) -> torch.Tensor:
        del kw
        a = env.action_manager.action[:, : self._n]
        p = env.action_manager.prev_action[:, : self._n]
        return torch.sum(self._peso * torch.square(a - p), dim=-1)


# --------------------------------------------------------- tarefa: pegar (T8)
_PELVE_DE_PE_Z = float(KNEES_BENT_KEYFRAME.pos[2])
"""Altura da pelve DE PÉ (keyframe `KNEES_BENT`, 0.76 m) — a MESMA referência de
onde o `de_pe_z = 0.65` da régua deriva. Importada do keyframe, não digitada."""


_TOPO_RAMPA_Z = _PELVE_DE_PE_Z + 0.15
"""Topo da rampa do eixo `alvo` do `pegar`: **altitude FIXA de 0,91 m no mundo.**

⚠️⚠️ **NÃO é "a altura do peito". É uma altitude constante que POR ACASO coincide com a
altura do peito quando o robô está de pé.** A distinção é o defeito que custou um bloco
inteiro de treino, e o nome antigo — `_PEITO_DE_PE_Z` — convidava à confusão:

  - **alvo que SEGUE o robô** (`pelve_atual + 0,15`) — foi o que o `lift` usou até 10/08.
    Agachar baixa o alvo, encurta o percurso e `progress = 1,0` fica alcançável com a
    caixa mais baixa. O argmax vira **levar o peito até a caixa**, e foi exatamente o que
    o robô aprendeu: `box_at_peito = 0,51` com `cond_fisica = 0,0000`.
  - **alvo FIXO no mundo** (esta constante) — agachar não move nada. A única forma de
    subir o `progress` é **erguer a caixa**.

Esta constante é lida uma vez, de `KNEES_BENT_KEYFRAME.pos[2] + 0,15`, e nunca consulta
a pose corrente. Nenhuma função da cadeia do `pegar` (`lift_altura`, `box_shake_pegar`,
`condicao_tarefa`) toca em `root_link_pos_w` do robô para formar o alvo.

O `0,15` é o `alvo_peito_b[2]` da §14. Ele entra só para DERIVAR o número em vez de
digitá-lo — não para acoplar o alvo ao peito.

⚠️ Sem termo de `env_origins`, de propósito: as origens de env diferem em x e y, nunca
em z (ver `events.afasta_cena`), e o `plr_rest_z` também é gravado sem elas
(`common/events.py:164`). Somar aqui e não lá desalinharia numerador e denominador."""


def alvo_peito_w(env: "ManagerBasedRlEnv", alvo_peito_b) -> torch.Tensor:
    """O alvo do peito, do frame da BASE pro mundo. [B, 3]

    Ele é CONSTANTE na base — é por isso que o `alvo_pos` do comando não precisa
    transmiti-lo (§9). Mas o erro da caixa se mede no mundo, então converte aqui.

    ⚠️ **Voltou a ser 100% frame da base em 17/08**, e a âncora de mundo que viveu um
    dia aqui foi removida. Ela existia para matar um argmax agachado no `pegar`: o alvo
    descia com a pelve, então levar o peito até a caixa pagava o mesmo que erguer a
    caixa. O `pegar` **não usa mais este termo** — o alvo dele é altura de mundo, via
    `alvo_z_pegar`. Sobrou o `locomover_carregando`, e ali a âncora de mundo era pior:
    a pelve oscila na marcha, e não existe canal de altura de mundo na observação (nem
    no ator nem no crítico). Frame da base é o único alvo que o robô consegue calcular
    de `box_pos_b`.

    O agachamento não compensa no `locomover_carregando` por dois outros motivos: ele
    tem de rastrear velocidade (4,0 dos 5,0 de orçamento dele), e a terminação `largou`
    encerra o episódio se a caixa cair."""
    robot: Entity = env.scene["robot"]
    alvo_b = torch.as_tensor(alvo_peito_b,
                             device=robot.data.root_link_pos_w.device)
    alvo_b = alvo_b.expand(robot.data.root_link_pos_w.shape[0], 3)
    return robot.data.root_link_pos_w + quat_apply(robot.data.root_link_quat_w,
                                                   alvo_b)


def alvo_z_pegar(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Altura-alvo do `pegar`. **Altitude de MUNDO, por env.** [B]

        alvo_z = repouso + fração × (TOPO_RAMPA_Z − repouso)

    ⚠️⚠️ **As DUAS pontas são do MUNDO. Nenhuma segue o robô.** O `repouso` é onde a
    caixa descansa, ou seja uma propriedade da PRATELEIRA (`plr_rest_z`, gravado pelo
    `reset_scene_plr`, com o jitter de ±2 cm). O topo é a constante `_TOPO_RAMPA_Z`.
    Esta função não lê `root_link_pos_w` do robô, e não pode passar a ler: um alvo que
    acompanha a pelve faz agachar encurtar o percurso, e o argmax vira levar o peito
    até a caixa em vez de erguer a caixa. Ver `_TOPO_RAMPA_Z`.

    A fração vem do eixo `alvo` do currículo (`env.alvo_frac`).

    **Por que fração e não metro.** Ela compõe com o eixo `altura`: quando a prateleira
    descer, o repouso desce e a rampa se re-escala sozinha — o nível 1,0 continua sendo
    a mesma altitude de 0,91 m, e o percurso a vencer cresce, que é o correto.

    ⚠️ **O robô NÃO observa este alvo, e isso é seguro.** O `progress` do `lift` satura
    em 1,0 e o critério é um PISO em z — erguer mais nunca reduz a recompensa nem
    reprova. Portanto a política não precisa saber em que nível está: ela maximiza
    subindo, sempre. Foi essa monotonicidade que permitiu tirar o alvo da observação em
    vez de alargá-la (Categoria C).

    Fallback para o `alvo_frac` ausente: nível mais fácil. O termo de recompensa é
    construído antes do termo de currículo."""
    rest = env.plr_rest_z
    frac = getattr(env, "alvo_frac", None)
    if frac is None:
        frac = torch.full_like(rest, float(T_LEVELS_ALVO[0]))
    return rest + frac * (_TOPO_RAMPA_Z - rest).clamp(min=0.05)


def lift_altura(env: "ManagerBasedRlEnv", object_name: str,
                rest_z_attr: str, palm_sensors, back_sensors,
                upright_std: float = 0.1) -> torch.Tensor:
    """Progresso de altura da caixa, do repouso até `alvo_z_pegar`. [B]

    ⚠️ **Chamava-se `lift_ao_peito` até 17/08, e o alvo era o peito.** Agora ele é a
    altura graduada pelo eixo `alvo` do currículo. O motivo: com denominador de 0,26 m
    fixo, erguer 5 cm dava `progress = 0,19` e o portão do currículo pede 0,90 — ou
    seja o robô tinha de erguer 23,4 cm antes de QUALQUER evento, e ele conseguia
    0,4 cm. Com a rampa, o nível 0 fecha a tarefa com 5 cm. Ver `alvo_z_pegar`.

    ⚠️ **Existe porque o `lift_reward` da Lift estava medindo nada aqui.** Ele lê a
    altura-alvo de `command[:, 2]`, e no `pegar` o `alvo_pos` do comando É A PRÓPRIA
    CAIXA (`commands.py`, `alvo = caixa` para PEGAR e REORIENTAR). Numerador e
    denominador viravam o mesmo número:

        progress = (box_z − rz) / (target_z − rz)      com target_z ≡ box_z

    Medido em 06/08 movendo a caixa à mão: `progress = 1.0000` a +0.10 m, +0.30 m e
    +0.60 m, e `nan` quando `box_z == rz` exatamente. Ou seja o termo de peso 2.0 — o
    maior sinal de tarefa do `pegar` — era `2.0 × grasp × upright`: pagava por encostar
    as palmas e manter a caixa nivelada, nunca por erguer.

    ⚠️ E o NaN não era teórico. A S1 ligou `rest_z_attr="plr_rest_z"`, e o
    `reset_scene_plr` grava ali a posição EXATA de repouso — então `box_z == rz` no
    primeiro passo de todo episódio. `_grasp = 0` não protege: `0.0 * nan = nan` em
    IEEE, e a terminação `nonfinite` checa o ESTADO, não a recompensa.

    **A correção é separar alvo de NAVEGAÇÃO de alvo de TAREFA.** O comando continua
    apontando para a caixa, porque é isso que o twist precisa para o robô se aproximar.
    A altura-alvo do progresso passa a ser o peito, que é onde a caixa tem de chegar.

    O denominador ganha um piso: erguer 1 mm não pode valer progresso 1.0.

    O resto é idêntico ao `lift_reward` — preensão × orientação × progresso, com o
    mesmo kernel suave de `upright` que fecha o hack de tombar."""
    obj: Entity = env.scene[object_name]
    box_z = obj.data.root_link_pos_w[:, 2]
    rz = getattr(env, rest_z_attr)
    alvo_z = alvo_z_pegar(env)
    # piso no denominador: sem ele, uma caixa que nasce à altura do alvo daria 0/0
    span = (alvo_z - rz).clamp(min=0.02)
    progress = torch.clamp((box_z - rz) / span, 0.0, 1.0)

    world_up = torch.zeros(box_z.shape[0], 3, device=box_z.device)
    world_up[:, 2] = 1.0
    box_up = quat_apply(obj.data.root_link_quat_w, world_up)
    upright = torch.exp(-(1.0 - box_up[:, 2]) / upright_std)
    return _grasp(env, palm_sensors, back_sensors) * upright * progress


def unload(env: "ManagerBasedRlEnv", object_name: str, sensor_apoio: str,
           palm_sensors, back_sensors, rest_z_attr: str = "plr_rest_z",
           margem: float = 0.02, g: float = 9.81) -> torch.Tensor:
    """Fração do peso da caixa que SAIU da prateleira. [B] (17/08)

    **A ponte contínua entre tocar e erguer.** O `_grasp` é booleano: tocar paga, e
    apertar paga zero até a caixa se mover. Medido no bloco 3, com 22 mil iterações:
    preensão em 0,851 e a caixa subindo **4 mm**. O robô mora no platô pago.

    A força normal da prateleira contra a caixa cai de `m·g` para zero **antes de a
    caixa sair do lugar**. É a única grandeza da cena que responde ao aperto de forma
    contínua, e por isso ela preenche exatamente o vão.

        unload = preensão × clamp(1 − F_apoio_z / (m·g)) × [box_z >= repouso − margem]

    ⚠️ **Os dois fatores de gate são obrigatórios, e cada um fecha um hack medível.**
    Sem `preensão`, empurrar a caixa da borda zera o apoio da MESA (o sensor casa
    caixa↔mesa, não caixa↔chão) e paga o termo inteiro sem manipular nada. Sem o teste
    de altura, a caixa no chão continua pagando.

    ⚠️ O peso é `env.peso_amostrado × g`, e NÃO `box_mass`: a DR de carga aplica força
    externa em vez de mudar a massa (`dr.body_mass` corrompe a heap), então a massa do
    modelo não reflete a carga sorteada.

    ⚠️ O sensor precisa do campo `"force"`. Com `reduce="netforce"` ele vem em
    `[B, N, 3]` no frame GLOBAL (`sensor/contact_sensor.py:196`), então a componente z
    é o apoio vertical direto. O `abs()` é indiferença de sinal, não conserto.

    Reward-only: força de contato é grandeza privilegiada do sim e não entra na obs."""
    obj: Entity = env.scene[object_name]
    forca = env.scene[sensor_apoio].data.force
    assert forca is not None, (
        f"o sensor '{sensor_apoio}' precisa do campo 'force' — ver env.py")
    apoio_z = forca[:, :, 2].sum(dim=1).abs()
    peso = (env.peso_amostrado * g).clamp(min=1e-3)
    fracao = (1.0 - apoio_z / peso).clamp(0.0, 1.0)
    acima = obj.data.root_link_pos_w[:, 2] >= (getattr(env, rest_z_attr) - margem)
    return _grasp(env, palm_sensors, back_sensors) * fracao * acima.float()


def box_shake_pegar(env: "ManagerBasedRlEnv", object_name: str,
                    palm_sensors, back_sensors,
                    std: float = 0.10) -> torch.Tensor:
    """`box_shake` que só cobra DEPOIS de a caixa chegar perto do alvo. [B] (17/08)

    O termo plano brigava com a tarefa: medido no bloco 3, ele subia junto com o `lift`
    e cancelava o ganho dele. Erguer uma caixa por abraço gira a caixa — a rotação é
    parte da manobra, não hack.

    O gate é `preensão × exp(−(alvo_z − box_z)²/std²)`, com o erro clampado em zero
    para cima: passar do alvo não reabre a cobrança. Portanto erguer é de graça e
    sacudir a caixa já erguida custa.

    Mesmo desenho do `hold_still_bonus` da Lift, e pelo mesmo motivo declarado lá: não
    taxar o `reach`/`lift`, que exigem exatamente o movimento que o termo puniria."""
    obj: Entity = env.scene[object_name]
    w2 = torch.sum(torch.square(obj.data.root_link_ang_vel_w), dim=-1)
    falta = (alvo_z_pegar(env) - obj.data.root_link_pos_w[:, 2]).clamp(min=0.0)
    porta = _grasp(env, palm_sensors, back_sensors) * torch.exp(-(falta ** 2) / std ** 2)
    return w2 * porta


def box_at_peito(env: "ManagerBasedRlEnv", std: float, object_name: str,
                 alvo_peito_b, palm_sensors, back_sensors,
                 std_grosso: float = 0.0) -> torch.Tensor:
    """Preensão × gaussiana do erro caixa->alvo do peito (§6b, +1).

    Adaptado do `sustain_precise_reward` da Lift, com UMA diferença de fundo: lá o
    alvo vinha do vetor de comando; aqui sai do `alvo_peito_w` — o xy ACOMPANHA o
    robô (andar com a caixa no peito continua pontuando, que é o que o `andar
    c/ caixa` precisa), e o z é ancorado no MUNDO em altura de peito DE PÉ
    (10/08): agachar com a caixa parou de pagar, em qualquer tarefa com caixa.

    Exige preensão (`_grasp`) porque a caixa parada no lugar certo sem estar na mão
    não é a tarefa. Kernels e gate de preensão vêm da Lift por import.

    ⚠️ **Dupla escala desde 11/08 — mesmo conserto, mesmo modo de falha do
    `box_at_prateleira`.** Com a âncora do peito em MUNDO (0.91 m), o std único de
    0.05 valia `e⁻²⁵ = zero exato` com a caixa a 25 cm — o caminho vertical inteiro
    ficou sem o segundo pagador (só o `lift` pagava, com o span dobrado), e o
    `pegar` se acomodou em "mãos na caixa, sem erguer": medido no bloco 3,
    `lift = 0.02` e `std_vantagem/pegar` colapsado de 0.18 pra 0.075. A grossa
    (0.30) paga ~0.25 a 25 cm; o fino continua mandando perto do alvo. Mesmo 50/50
    do `reaching`, `orienta_face` e `box_at_prateleira`."""
    obj: Entity = env.scene[object_name]
    err_sq = torch.sum(torch.square(
        alvo_peito_w(env, alvo_peito_b) - obj.data.root_link_pos_w), dim=-1)
    kernel = height_kernel(err_sq, std)
    if std_grosso > 0.0:
        kernel = 0.5 * height_kernel(err_sq, std_grosso) + 0.5 * kernel
    return _grasp(env, palm_sensors, back_sensors) * kernel


# ---------------------------------------------------------- tarefa: botar (T8)
def box_at_prateleira(env: "ManagerBasedRlEnv", std: float, object_name: str,
                      command_name: str, palm_sensors, back_sensors,
                      fracao_solta: float = 0.0,
                      std_grosso: float = 0.0) -> torch.Tensor:
    """Gaussiana do erro caixa->alvo, SEM fator de preensão (§4, §6b).

        (1 − f) × kernel  +  f × kernel × (1 − preensão)        f = fracao_solta

    Com o default `f = 0.0` isto é o kernel puro que o doc especifica, e o
    raciocínio dele fecha: transportando, a caixa está longe da prateleira →
    recompensa baixa; aproximando e baixando → sobe; **soltando → continua alta**,
    porque não há fator de preensão. Sem vale, e "não largar no caminho" emerge do
    próprio termo, sem penalidade de queda nem gate espacial.

    ⚠️ Isso só vale porque a caixa **começa na mão** — a condição de spawn
    "segurando" do `andar c/ caixa`, `parado c/ caixa` e `botar`. Com a caixa
    nascendo em cima da prateleira, o `botar` começaria perto do alvo e o termo não
    ensinaria nada.

    `f > 0` existe como lever: o kernel puro deixa a política INDIFERENTE entre
    segurar no alvo e soltar no alvo (as duas pontuam igual), e o sucesso exige soltar.

    ⚠️ **A afirmação de que isso era só indiferença estava ERRADA, e foi corrigida em
    06/08.** Ela foi escrita contando este termo sozinho. Com o `reaching` ligado no
    `botar` — como estava até agora — soltar custava de 0.54 a 0.85 por passo, porque
    as palmas se afastam da caixa. Contra um orçamento de 3.5, eram 15% a 24% cobrados
    por OBEDECER ao critério de sucesso, que exige `~preensao`. O argmax da recompensa
    era o estado que o critério reprova. O conserto foi tirar o `reaching` do gate do
    `botar`, não mexer em `f`.

    ⚠️ **DUAS ESCALAS desde 06/08, e sem elas a tarefa não tinha shaping.** Com o
    `std` único de 0.05 — herdado da Lift, onde ele serve à PRECISÃO FINAL — o termo
    valia `exp(−0.397²/0.05²) = 4e−28` no spawn do `botar`. A caixa nasce na mão a
    ~0.40 m da prateleira, e o gradiente era indistinguível de zero em float32 nos
    primeiros 33 cm dos 40. O `knobs.py` já previa este caso por escrito: "se algum
    dia uma tarefa tiver que ATINGIR o alvo partindo de longe sem `lift`/`reaching`
    ligados, este número volta pra mesa". O `botar` é essa tarefa.

    A escala grossa mantém sinal de longe; a fina premia a colocação. Mesmo desenho
    do `orienta_face` e do `reaching_reward`, pelo mesmo motivo.
    """
    obj: Entity = env.scene[object_name]
    alvo = env.command_manager.get_term(command_name).command[:, ALVO]
    err_sq = torch.sum(torch.square(alvo - obj.data.root_link_pos_w), dim=-1)
    k = height_kernel(err_sq, std)
    if std_grosso > 0.0:
        k = 0.5 * height_kernel(err_sq, std_grosso) + 0.5 * k
    soltou = 1.0 - _grasp(env, palm_sensors, back_sensors)
    return (1.0 - fracao_solta) * k + fracao_solta * k * soltou


# ----------------------------------------------------- tarefa: reorientar (T8)
def orienta_face(env: "ManagerBasedRlEnv", command_name: str,
                 std_grosso_deg: float, std_fino_deg: float,
                 xy_std: float) -> torch.Tensor:
    """Gira a face alvo pra direção alvo SEM arrastar a caixa (§6b, +1).

        (0.5 × kernel(erro, grosso) + 0.5 × kernel(erro, fino)) × kernel(desvio_xy)

    Três decisões, cada uma com motivo:

    **1. Duas escalas de ângulo, não uma.** Com `std` único de 5° o nível 15° do eixo
    de giro daria `exp(−(15/5)²) = 1.2e−4` no reset — gradiente nenhum até o robô já
    estar dentro de ~10°. A escala grossa mantém sinal de longe e a fina premia a
    precisão final. Mesmo desenho monotônico anti-vale do `reaching_reward` da Lift,
    pelo mesmo motivo.

    **2. A posição entra no KERNEL, nunca como penalidade por passo.** Se deslocar
    custasse a cada passo, "não tocar na caixa" pontuaria melhor que girar com 3 cm
    de deriva, e a manobra certa seria castigada enquanto acontece.

    **3. NÃO exige que a caixa esteja apoiada.** No nível topo/fundo a solução é
    erguer e rolar a caixa entre as palmas — no meio da manobra ela está no ar.
    Exigir apoio no reward reprovaria a única solução que existe. O apoio é exigido
    só no critério de SUCESSO, que é terminal."""
    termo = env.command_manager.get_term(command_name)
    erro = termo.erro_angulo_deg()
    ang = (0.5 * height_kernel(erro ** 2, std_grosso_deg)
           + 0.5 * height_kernel(erro ** 2, std_fino_deg))
    return ang * height_kernel(termo.desvio_xy() ** 2, xy_std)
