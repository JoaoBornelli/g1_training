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
        self._tasks = torch.tensor(
            list(cfg.params["tasks"]), dtype=torch.long, device=env.device)

    def __call__(self, env: "ManagerBasedRlEnv", inner, tasks,
                 gate_command: str = "lift_target", **kw) -> torch.Tensor:
        """`gate_command` e NÃO `command_name`: vários termos internos (`lift_reward`,
        `hold_still_bonus`, `orienta_face`) têm um `command_name` PRÓPRIO, e usar a
        mesma chave fazia o gate engolir o parâmetro do termo de dentro —
        `TypeError: lift_reward() missing 1 required positional argument`."""
        del inner, tasks                      # já resolvidos no __init__
        onehot = env.command_manager.get_term(gate_command).command[:, ONEHOT]
        return self._inner(env, **kw) * onehot[:, self._tasks].sum(dim=-1)

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
                      fracao_solta: float = 0.0) -> torch.Tensor:
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

    `f > 0` existe como lever pra a Tarefa 12: o kernel puro deixa a política
    INDIFERENTE entre segurar no alvo e soltar no alvo (as duas pontuam igual), e o
    sucesso exige soltar. Indiferença não é incentivo contrário, mas se o robô ficar
    segurando, `f` é o botão."""
    obj: Entity = env.scene[object_name]
    alvo = env.command_manager.get_term(command_name).command[:, ALVO]
    err_sq = torch.sum(torch.square(alvo - obj.data.root_link_pos_w), dim=-1)
    k = height_kernel(err_sq, std)
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
