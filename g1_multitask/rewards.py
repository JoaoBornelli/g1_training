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

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from g1_training.skills.lift.rewards import _grasp, height_kernel

from .tasks import ONEHOT_DIM

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
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
    andar de agachar não está medindo comportamento."""
    a = env.action_manager.action[:, :n_juntas]
    p = env.action_manager.prev_action[:, :n_juntas]
    return torch.sum(torch.square(a - p), dim=-1)


# -------------------------------------------------------------- locomoção (T6)
def track_linear_velocity_freio_z(
    env: "ManagerBasedRlEnv",
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Cópia de `vel_mdp.track_linear_velocity` com UMA linha diferente (item 11).

    Original:   `z_error = actual_z²`            — sempre punido
    Aqui:       `z_error = actual_z² × (fora do d_morto)`

    Por quê: dentro do raio de chegada o robô PRECISA se mover em z — agachar pra
    pegar a caixa no chão, erguer pra levar ao peito. Punir velocidade em z ali
    briga diretamente com a tarefa. Fora do raio, o robô está andando, e velocidade
    vertical é pulo — aí a punição é o que se quer.

    Cópia inteira e não wrapper porque a linha a mudar é no MEIO do cálculo; um
    wrapper teria que recomputar tudo pra subtrair o termo."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    fora = (~env.command_manager.get_term(command_name).dentro_do_morto()).float()
    z_error = torch.square(actual[:, 2]) * fora        # <-- a única linha diferente
    return torch.exp(-(xy_error + z_error) / std**2)


# --------------------------------------------------------- tarefa: pegar (T8)
def alvo_peito_w(env: "ManagerBasedRlEnv", alvo_peito_b) -> torch.Tensor:
    """O alvo do peito, do frame da base pro mundo. [B, 3]

    Ele é CONSTANTE na base — é por isso que `alvo_pos` do `pegar` não precisa
    transmiti-lo no vetor de comando (§9). Mas o erro da caixa se mede no mundo,
    então converte aqui."""
    robot: Entity = env.scene["robot"]
    alvo_b = torch.tensor(alvo_peito_b, device=robot.data.root_link_pos_w.device)
    alvo_b = alvo_b.expand(robot.data.root_link_pos_w.shape[0], 3)
    return robot.data.root_link_pos_w + quat_apply(robot.data.root_link_quat_w, alvo_b)


def lift_ao_peito(env: "ManagerBasedRlEnv", object_name: str, alvo_peito_b,
                  rest_z_attr: str, palm_sensors, back_sensors,
                  upright_std: float = 0.1) -> torch.Tensor:
    """Progresso de altura da caixa, do repouso ATÉ O PEITO. [B]

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
    alvo_z = alvo_peito_w(env, alvo_peito_b)[:, 2]
    # piso no denominador: sem ele, uma caixa que nasce à altura do peito daria 0/0
    span = (alvo_z - rz).clamp(min=0.05)
    progress = torch.clamp((box_z - rz) / span, 0.0, 1.0)

    world_up = torch.zeros(box_z.shape[0], 3, device=box_z.device)
    world_up[:, 2] = 1.0
    box_up = quat_apply(obj.data.root_link_quat_w, world_up)
    upright = torch.exp(-(1.0 - box_up[:, 2]) / upright_std)
    return _grasp(env, palm_sensors, back_sensors) * upright * progress


def box_at_peito(env: "ManagerBasedRlEnv", std: float, object_name: str,
                 alvo_peito_b, palm_sensors, back_sensors) -> torch.Tensor:
    """Preensão × gaussiana do erro caixa->alvo do peito (§6b, +1).

    Adaptado do `sustain_precise_reward` da Lift, com UMA diferença de fundo: lá o
    alvo vinha do vetor de comando; aqui é constante na base. Isso muda o
    significado — o alvo ACOMPANHA o robô, então andar com a caixa no peito
    continua pontuando, o que é exatamente o que o `andar c/ caixa` precisa.

    Exige preensão (`_grasp`) porque a caixa parada no lugar certo sem estar na mão
    não é a tarefa. Kernels e gate de preensão vêm da Lift por import."""
    obj: Entity = env.scene[object_name]
    err_sq = torch.sum(torch.square(
        alvo_peito_w(env, alvo_peito_b) - obj.data.root_link_pos_w), dim=-1)
    return _grasp(env, palm_sensors, back_sensors) * height_kernel(err_sq, std)


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
