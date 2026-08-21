"""Os 9 termos de tarefa do g1_poc (§8.2).

Os 13 termos de fundação vêm do `velocity` do mjlab, sem uma linha aqui.

Desenho:
    staged      = reaching × (1 + bringing)          — o anti-hack é a FORMA
    precise_pos = exp(−‖caixa − alvo‖² / 0,05²)
    precise_ori = reaching × exp(−Δθ² / σ²)          — σ variável por elo
    squeeze     = tanh( min(F_n_esq, F_n_dir) / F_ref )
    unload      = 1 − F_apoio/m·g                    — a PONTE do platô do grasp
    postura_ereta = rampa2(pelve) × preensão × descarga — condição 3 do fecho
    sustentacao = rampa(t / alvo_elo)                 — alvo por elo, 1,0/0,5 s
    load        = clamp(F_apoio/m·g)                 — espelho do unload, só botar
    joint_vel_hinge = (|v| − v_max)⁺²

⚠ O `unload` entrou em 19/08, depois de o bloco 1 rodar 1884 iterações com sucesso
ZERO. O `squeeze` saturou em 6× `F_ref` (derivada 1e-5) e o robô PRENSAVA a caixa
contra a prateleira — apoio em 138% do peso. Ele soma ao `squeeze`, não o substitui.
Ver o docstring de `unload`.

Os oito primeiros (staged, precise_pos, precise_ori, squeeze, unload, postura_ereta,
sustentacao, load) multiplicam por `caixa_valida`. **Isto é obrigatório**: com o bit em 0
os canais da caixa são zerados, e um vetor zerado dá exp(0) = 1. Sem a
multiplicação, "não existe caixa" pagaria o valor MÁXIMO.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.lab_api.string import resolve_matching_names_values

from g1_poc.observacoes import alvos_das_palmas

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


# --------------------------------------------------------------------- auxiliares
def _valida(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).command[:, 9]


def _preensao(env: ManagerBasedRlEnv, palm_sensors: tuple[str, ...]) -> torch.Tensor:
    """[B] bool — as DUAS palmas em contato com a caixa.

    Gate compartilhado por `unload` e `postura_ereta`. Uma palma sozinha não conta:
    sem preensão bimanual os dois termos pagariam por caminhos que não são erguer.
    """
    pega: torch.Tensor | None = None
    for nome in palm_sensors:
        found = env.scene[nome].data.found
        assert found is not None, f"sensor '{nome}' precisa do field 'found'."
        aqui = (found > 0).any(dim=-1)
        pega = aqui if pega is None else (pega & aqui)
    assert pega is not None, "palm_sensors vazio"
    return pega


def _erro_pos_sq(env: ManagerBasedRlEnv, command_name: str, object_name: str):
    obj: Entity = env.scene[object_name]
    alvo = env.command_manager.get_term(command_name).command[:, 0:3]
    return torch.sum(torch.square(alvo - obj.data.root_link_pos_w), dim=-1)


def _reaching(env, object_name, lateral_offset, std, asset_cfg):
    """O `reaching` BIMANUAL.

    O `reaching` do `lift_cube` do mjlab mede UM site (um braço com garra). Aqui
    cada palma mira a SUA face lateral, e a média das duas distâncias entra no
    kernel: uma mão atrasada derruba o gradiente, portanto as duas se aproximam
    juntas. O máximo do termo é a pose PRÉ-GRASP, com as mãos flanqueando a caixa.

    ⚠ O σ é `max(std, distância inicial palma→face do elo)` — `env.poc_reach_inicial`,
    escrito pelo comando no começo do elo. Com σ fixo o gradiente de aproximação cai
    1391× entre a prateleira a 0,55 m e a 0,04 m (medido 20/08): os níveis 3+ do
    currículo viravam sorte. É a MESMA correção que a §8.2 fez no `bringing`.
    `std` vira o PISO (e o fallback quando o buffer não existe — smoke chama a
    função fora do laço do env).
    """
    robot: Entity = env.scene[asset_cfg.name]
    palmas = robot.data.site_pos_w[:, asset_cfg.site_ids]
    alvos = alvos_das_palmas(env, object_name, lateral_offset)
    d2 = torch.sum(torch.square(palmas - alvos), dim=-1)
    sigma = getattr(env, "poc_reach_inicial", None)
    if sigma is None:
        return torch.exp(-d2.mean(dim=-1) / std**2)
    return torch.exp(-d2.mean(dim=-1) / sigma.clamp(min=std) ** 2)


# ----------------------------------------------------------- os 8 termos
def staged(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    reaching_std: float,
    lateral_offset: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """`reaching × (1 + bringing)` — a forma do `lift_cube` do mjlab.

    O `bringing` só paga ATRAVÉS do `reaching`. Levar a caixa ao alvo sem as mãos
    nela não paga. O anti-hack é a forma, e não uma penalidade.

    O σ do `bringing` é VARIÁVEL, e é a única mudança que fazemos numa função do
    mjlab. O σ fixo de 0,30 m já está saturado no nível 0, onde a caixa sobe
    0,17 m: exp(−0,17²/0,30²) = 0,72 contra 1,00 no alvo. Com
    σ = distância comandada, o termo cobre o percurso todo.
    """
    reaching = _reaching(env, object_name, lateral_offset, reaching_std, asset_cfg)
    std = env.poc_dist_inicial
    bringing = torch.exp(-_erro_pos_sq(env, command_name, object_name) / std**2)
    return reaching * (1.0 + bringing) * _valida(env, command_name)


def precise_pos(
    env: ManagerBasedRlEnv, command_name: str, object_name: str, std: float
) -> torch.Tensor:
    """Gaussiana APERTADA no alvo. É o termo que paga por a caixa ESTAR no lugar.

    O treino atual não tem nenhum termo assim no `pegar`: o `box_at_peito` foi
    retirado dele pelo ADR-0001, e sobrou só o progresso escalar.
    """
    err = _erro_pos_sq(env, command_name, object_name)
    return torch.exp(-err / std**2) * _valida(env, command_name)


def precise_ori(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    std: float,
    lateral_offset: float,
    reaching_std: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """`reaching × exp(−Δθ²/σ²)`.

    Δθ é o ângulo entre a normal da face alvo, em MUNDO, e o `dir_alvo`. A
    simetria do cubo se resolve sozinha: girar em torno da normal não move o vetor.

    No nível 0 o `dir_alvo` é a normal ATUAL, portanto o termo pede "erga sem
    torcer". É isto que substitui o `box_shake`: erguer torto deixa de pagar, em
    vez de custar.

    ⚠ Com σ fixo de 0,40 rad, 90° dá 2,0e-7 — o `reorientar` dos níveis 4+ era
    sorte; mesmo idioma do `bringing`/`reaching`.
    """
    cmd = env.command_manager.get_term(command_name)
    theta = cmd.erro_ang()
    reaching = _reaching(env, object_name, lateral_offset, reaching_std, asset_cfg)
    sigma = getattr(env, "poc_ori_inicial", None)
    if sigma is None:
        sigma = torch.full_like(theta, std)
    return reaching * torch.exp(-torch.square(theta) / sigma.clamp(min=std) ** 2) * _valida(env, command_name)


def squeeze(
    env: ManagerBasedRlEnv,
    command_name: str,
    palm_sensors: tuple[str, str],
    massa_attr: str,
    mu: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """`tanh( min(F_n_esq, F_n_dir) / F_ref )` — o termo que faltava.

    **Este é o termo mais importante do arquivo.** Sem ele o treino repete a falha
    de hoje.

    O diagnóstico: o `lift` de hoje paga +0,34/s por centímetro de subida no nível
    0. O gradiente existe e é grande. O robô mesmo assim não subiu 1 cm. O motivo é
    que o gradiente está na coordenada ERRADA:

        d(recompensa)/d(altura da caixa)  = grande
        d(recompensa)/d(força de aperto)  = ZERO

    A política só age em alvos de junta. Antes de a força de atrito vencer o peso a
    caixa não se move, e nenhuma recompensa muda. É um degrau, não uma rampa.

    O `reaching` não conserta isto: a palma não penetra a caixa, portanto ele satura
    no contato.

    A força de palma cresce de forma CONTÍNUA com a penetração comandada. Portanto
    este termo tem derivada positiva em toda a faixa de 0 N a F_ref.

    ANTI-HACK: usa só a componente NORMAL AO PAD. Apertar a caixa para BAIXO contra
    a prateleira gera força TANGENCIAL, e não normal — o ADR-0001 registrou esse
    risco. A projeção o fecha sem precisar de um segundo termo.

    A normal do pad vem da ORIENTAÇÃO DO SITE, e não do campo `normal` do sensor.
    Motivo: os sensores de palma usam `reduce="netforce"`, que soma todos os contatos
    num wrench só — e aí "a normal do contato" perde significado. A geometria da mão
    é conhecida e fixa: o mesh mede 13,2 × 6,7 × 10,7 cm, portanto a face fina é Y, e
    a palma esquerda olha para −Y local e a direita para +Y local (é o que o offset
    dos pads em `common/robot.py` produz, e é por isso que as duas palmas ficam
    viradas uma para a outra).

    A força vem no frame GLOBAL, porque `netforce` implica `global_frame`.

    O `min` das duas palmas exige aperto SIMÉTRICO: uma palma sozinha vale zero.
    """
    robot: Entity = env.scene[asset_cfg.name]
    # ordem dos sites = ordem de PALM_SITES = (esquerda, direita). O G1 declara
    # `left_palm` antes de `right_palm` no XML, e a skill Lift já depende disso.
    quat = robot.data.site_quat_w[:, asset_cfg.site_ids]            # [B,2,4]
    locais = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], device=quat.device, dtype=quat.dtype)
    normais = quat_apply(quat, locais.expand(quat.shape[0], 2, 3))  # [B,2,3]

    forcas = []
    for i, nome in enumerate(palm_sensors):
        f = env.scene[nome].data.force
        assert f is not None, f"sensor '{nome}' precisa do field 'force'."
        # `f` é [B, slots, 3] no frame global. Projeta na normal da palma.
        # `abs` em vez de sinal: errar a convenção zeraria o termo em silêncio, e a
        # componente que queremos excluir é a TANGENCIAL, que a projeção já remove.
        f_n = torch.sum(f * normais[:, i].unsqueeze(1), dim=-1).abs().sum(dim=-1)
        forcas.append(f_n)
    f_min = torch.minimum(forcas[0], forcas[1])
    # F_ref = m·g / (2·μ). A massa é POR ENV (a DR de carga a sorteia).
    massa = getattr(env, massa_attr)
    f_ref = (massa * 9.81 / (2.0 * mu)).clamp(min=1e-3)
    resultado = torch.tanh(f_min / f_ref) * _valida(env, command_name)
    # ⚠ Fora do `botar`: apertar durante o `botar` paga contra soltar (−1,0/s medido).
    elo = getattr(env, "poc_elo", None)
    fora_botar = torch.ones_like(resultado) if elo is None else (elo != 3).float()
    return resultado * fora_botar


def unload(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    support_sensor: str,
    palm_sensors: tuple[str, str],
    massa_attr: str,
    caixa_meia_z: float,
    tol_queda: float,
) -> torch.Tensor:
    """`1 − F_apoio/m·g`, gateado por preensão bimanual e por "a caixa não caiu".

    **É a ponte contínua do platô do grasp, e o `squeeze` sozinho não a dá.** Medido
    neste pacote na iteração 1884: a força de palma chega a 6× `F_ref`, onde
    `tanh(6,47) = 0,99999` e a derivada é 1e-5 — o `squeeze` satura e deixa de guiar.
    Ao mesmo tempo o apoio da prateleira ficava em 138% do peso: o robô PRENSAVA a
    caixa contra o tampo, porque isso escora os braços e nada cobrava por descarregá-la.

    A força de apoio é a única grandeza da cena que responde de forma contínua ao ato
    de erguer: ela cai de `m·g` a zero ANTES de a caixa se mover. Medido no
    g1_multitask, que fechou o mesmo platô com ela: 9,70 N apoiada -> 0,00 N erguida.

    Os dois gates são anti-hack, e cada um fecha um caminho distinto:

    1. **preensão bimanual** — sem ele, DERRUBAR a caixa da prateleira paga o máximo:
       sem tampo embaixo, `F_apoio = 0` e a fração vale 1.
    2. **a caixa não caiu** — sem ele, empurrá-la para fora do tampo paga igual a
       erguê-la.

    ⚠ O segundo gate é de QUEDA, e não de subida. Exigir a caixa ACIMA do repouso
    recriaria exatamente o degrau que este termo existe para remover: o apoio cai
    enquanto a altura ainda não mudou, e é nessa faixa que está o gradiente que falta.
    """
    f = env.scene[support_sensor].data.force
    assert f is not None, f"sensor '{support_sensor}' precisa do field 'force'."
    # `reduce="netforce"` -> força no frame GLOBAL, portanto z é o apoio vertical.
    apoio_z = f[..., 2].abs().sum(dim=-1)
    massa = getattr(env, massa_attr)
    peso = (massa * 9.81).clamp(min=1e-3)
    fracao = (1.0 - apoio_z / peso).clamp(0.0, 1.0)

    pega = _preensao(env, palm_sensors)

    # o repouso é o topo SORTEADO da prateleira, e não uma constante: o `reset_cena`
    # grava `env.poc_topo` por env, e o currículo alarga a faixa no passo 4.
    obj: Entity = env.scene[object_name]
    repouso = env.poc_topo + caixa_meia_z
    nao_caiu = obj.data.root_link_pos_w[:, 2] > (repouso - tol_queda)

    # ⚠ SÓ no elo `pegar` (20/08). Nos outros a caixa já saiu da prateleira, e no
    # `botar` este termo é o OPOSTO do fecho (F_apoio >= 0,8·m·g): ligado lá, pagaria
    # 2,0/s para NÃO botar. A máscara vem antes de qualquer mexida em `poc_topo`.
    elo = getattr(env, "poc_elo", None)
    no_pegar = torch.ones_like(fracao) if elo is None else (elo == 0).float()
    return fracao * pega.float() * nao_caiu.float() * no_pegar * _valida(env, command_name)


def load(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    support_sensor: str,
    massa_attr: str,
    raio_sucesso: float,
    raio_mult: float,
) -> torch.Tensor:
    """`clamp(F_apoio/m·g)` — o espelho do `unload`, SÓ no elo `botar` (§8.2.5).

    Sem ele o `botar` não tem quem pague por soltar: `squeeze` e `unload` apontam
    contra, e com as máscaras deles o saldo vira exatamente ZERO — o fecho
    (`F_apoio >= 0,8·m·g`) seria descoberto por sorte. Medido: satisfazer a 3ª
    condição custava −3,0/s antes das máscaras.

    O gate de posição (`erro < 2·raio`) fecha o hack de largar a caixa em qualquer
    lugar do tampo. Sem gate de preensão, de propósito: soltar É o objetivo, e o
    termo continua pagando depois do fecho — é o estado colocado que mais paga.
    """
    elo = getattr(env, "poc_elo", None)
    if elo is None:
        return torch.zeros(env.num_envs, device=env.device)
    f = env.scene[support_sensor].data.force
    assert f is not None
    apoio_z = f[..., 2].abs().sum(dim=-1)
    peso = (getattr(env, massa_attr) * 9.81).clamp(min=1e-3)
    fracao = (apoio_z / peso).clamp(0.0, 1.0)
    err = torch.sqrt(_erro_pos_sq(env, command_name, object_name))
    perto = (err < raio_mult * raio_sucesso).float()
    return fracao * perto * (elo == 3).float() * _valida(env, command_name)


def postura_ereta(
    env: ManagerBasedRlEnv,
    command_name: str,
    palm_sensors: tuple[str, ...],
    support_sensor: str,
    massa_attr: str,
    pelve_min: float,
    rampa: float,
    rampa_fina: float,
    frac_descarga: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Rampa contínua na altura da pelve, gateada por preensão E descarga (§8.2.3).

    A condição 3 do fecho do `pegar` (§7.2) exige `pelve >= 0,65 m` e nada pagava
    por ela na coordenada certa: quem precifica a pelve é só a `pose`, a ~0,73/m
    (o default do G1 é o KNEES_BENT_KEYFRAME, pelve em 0,76 m). O `precise_pos` é
    INDIFERENTE à pelve abaixo do alvo, e CONTRÁRIO acima (−16,2/m no fecho) —
    por isso a rampa tem uma parte FINA, íngreme perto de `pelve_min`.

    A rampa em DUAS partes (medido 20/08):
        fracao = 0,5·clamp((z − (pelve_min − rampa)) / rampa)
               + 0,5·clamp((z − (pelve_min − rampa_fina)) / rampa_fina)
    longa 0,20→0,65 (o nível 4 pega com a pelve a 0,267 m — sem ela, zona morta em
    33% das pegas) e fina 0,57→0,65 (14,7/m com peso 2,0, contra os −16,2/m do
    `precise_pos` no fecho). Satura em `pelve_min`: a régua não pede mais, e pagar
    por mais convidaria a ponta dos pés.

    Os DOIS gates, e o que cada um fecha (mesmo idioma do `unload`):
    - preensão bimanual: antes de ter a caixa, agachar para a pega baixa sai de
      graça — sem este gate o termo brigaria com o `staged`.
    - descarga (`F_apoio < frac_descarga·m·g`): sem ele, encostar as palmas e ficar
      de pé com a caixa APOIADA paga a rampa inteira — +2,0/s exatamente no platô
      "encosta e para" que o bloco 1 mediu.
    """
    robot: Entity = env.scene[asset_cfg.name]
    z = robot.data.root_link_pos_w[:, 2]
    f_longa = ((z - (pelve_min - rampa)) / rampa).clamp(0.0, 1.0)
    f_fina = ((z - (pelve_min - rampa_fina)) / rampa_fina).clamp(0.0, 1.0)
    fracao = 0.5 * f_longa + 0.5 * f_fina

    f = env.scene[support_sensor].data.force
    assert f is not None, f"sensor '{support_sensor}' precisa do field 'force'."
    apoio_z = f[..., 2].abs().sum(dim=-1)
    peso = (getattr(env, massa_attr) * 9.81).clamp(min=1e-3)
    descarregada = apoio_z < frac_descarga * peso

    pega = _preensao(env, palm_sensors)
    return fracao * pega.float() * descarregada.float() * _valida(env, command_name)


def sustentacao(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Rampa no TEMPO dentro da condição de fecho (§8.2.4).

    O fecho exige as 4 condições por 1,0 s ININTERRUPTO, e nenhum termo
    diferenciava 0,98 s de 0,00 s — o degrau que fazia o push (o único fator que
    degradava o sucesso, medido) decidir o currículo. Esta é a rampa na coordenada
    que faltava: o cronômetro do próprio comando.

    O denominador é o alvo do ELO corrente, não uma constante — senão os elos de
    0,5 s pagariam só metade da rampa.
    """
    cmd = env.command_manager.get_term(command_name)
    fracao = (cmd._sustenta / cmd._sust_alvo.clamp(min=1e-6)).clamp(0.0, 1.0)
    return fracao * _valida(env, command_name)


class hinge_por_forma:  # noqa: N801 (idioma do mjlab)
    """Dobradiça de velocidade de junta, POR FORMA e POR GRUPO DE JUNTA (§8.2.6).

    Grátis abaixo do teto; quadrática acima. A forma vem do
    `joint_velocity_hinge_penalty` do `lift_cube`.

    ⚠ A versão anterior tinha `max_vel = 0,5 rad/s` no corpo TODO, nas duas formas.
    Três fatos a condenam, e o bloco 1 mediu o efeito:

    1. **A tarefa `velocity` do mjlab NÃO tem este termo.** Nenhum. A marcha
       validada do G1 roda sem ele. O termo veio do `lift_cube`, que é um braço YAM
       de 6 DoF sobre uma mesa — não há marcha para atrapalhar lá.
    2. **0,5 rad/s é uma ordem de grandeza abaixo da marcha.** Os limites de
       velocidade das juntas do G1 são 20 a 37 rad/s. Um joelho em fase de balanço
       passa de 0,5 rad/s sem esforço, portanto o termo cobrava exatamente o balanço.
    3. **O repositório já tinha resolvido isso, e de outro jeito.** A skill Lift usava
       `arm_vel = −0,002` com escopo `.*(shoulder|elbow|wrist).*` — NUNCA a perna. O
       comentário dela: *"NÃO inclui perna — ela precisa de velocidade pra
       agachar/equilibrar."*

    Medido na it 5000 do bloco 1: o termo custava **−2,77/s**, e com o
    `action_rate_l2` somava 96% de toda a penalidade e 55% do sinal positivo. A
    assinatura no comportamento é `peak_height_mean = 0,0042` — **o pé subia 4 mm.**
    Sem fase de balanço não há passo, e o episódio de locomoção morria em 35 passos.

    O desenho novo:

    - **Locomoção (bit = 0): o termo é ZERO.** Exatamente o que o fabricante faz.
    - **Manipulação (bit = 1): teto POR GRUPO DE JUNTA.** O plano sagital da perna
      fica largo (agachar e levantar saem de graça); o braço fica apertado, para o
      movimento ser controlado; as juntas laterais ficam no meio.

    A regra em uma frase: **o que a tarefa exige mover fica livre; o que ela não
    exige, custa.**
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Entity = env.scene[asset_cfg.name]
        _, joint_names = asset.find_joints(asset_cfg.joint_names)
        _, _, tetos = resolve_matching_names_values(
            data=cfg.params["max_vel_manipulando"],
            list_of_strings=joint_names,
        )
        self.max_vel = torch.tensor(tetos, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        max_vel_manipulando,
        caixa_command_name: str,
        asset_cfg: SceneEntityCfg = _ROBOT,
    ) -> torch.Tensor:
        del max_vel_manipulando  # resolvido no __init__
        robot: Entity = env.scene[asset_cfg.name]
        v = robot.data.joint_vel[:, asset_cfg.joint_ids]
        excesso = (v.abs() - self.max_vel).clamp_min(0.0)
        custo = torch.square(excesso).sum(dim=-1)
        bit = env.command_manager.get_term(caixa_command_name).command[:, 9]
        env.extras["log"]["Metrics/hinge_excesso_manip"] = (
            excesso.sum(dim=-1) * bit).mean()
        return custo * bit
