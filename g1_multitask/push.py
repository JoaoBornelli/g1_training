"""O eixo `push` do currículo virando física (S2).

**O problema que isto resolve.** O eixo existia no papel e não existia na física.
`knobs.Push` estava definido e nenhum arquivo o lia. O que rodava era o `push_robot`
herdado do fabricante pelo `build_base_env`, com o range do fabricante, CONSTANTE. O
`curriculum.py` avançava `push_nivel` de 0 a 4 e a dificuldade não mudava — ou seja, a
Fase 0 media a mesma coisa cinco vezes. E ela é o portão de `parado` para `andar`.

**A escada.** Os números saem de `knobs.Push`; nenhum é novo aqui.

    nível  fator  velocidade                força sustentada  duração
    0      0.00   zero                      não               —
    1      0.35   35% dos 6 componentes     não               —
    2      0.70   70%                       não               —
    3      1.00   100%                      até 50 N          duracao_curta_s
    4      1.00   100%                      até 50 N          duracao_longa_s

Níveis 0 a 2 são **push**. Níveis 3 e 4 são **push and hold**.

**Reuso.** Nenhuma dinâmica nova: `empurrao` é o `push_by_setting_velocity` do
fabricante com o range escalado, e `empurrao_sustentado` envolve o
`apply_body_impulse` do mjlab — o mesmo que o `skills/stand/env.py:50` já fia.

**Não há conflito de wrench com o peso da caixa.** O `payload_por_nivel` escreve
`xfrc_applied` na **caixa**; o `empurrao_sustentado` escreve na **pelve**. Entidades
diferentes, buffers diferentes. Fica registrado porque é a primeira coisa que alguém
vai suspeitar ao ver duas forças externas no mesmo env.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import events as base_events
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

from . import tasks as T

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_ROBO = SceneEntityCfg("robot")
_PELVE = SceneEntityCfg("robot", body_names="pelvis")

JANELA_LIVRE_S = 0.5
"""Segundos no início do episódio em que NENHUM empurrão age.

⚠️ Não é folga arbitrária. As três tarefas de `SPAWN_SEGURANDO` nascem com as palmas
apenas TOCANDO a caixa, com força normal zero. O `pregrasp.py` mede que, com ação nula,
a caixa cai 22 cm em 0.5 s. Um empurrão dentro desse intervalo torna o episódio
não-ganhável antes de a política ter tido a chance de fechar a preensão — e o
currículo leria isso como incompetência.
"""

_NIVEL_HOLD = 3
"""A partir deste nível o push vira **push and hold**. Abaixo dele só há o chute de
velocidade. Sai da escada da §14: força sustentada existe nos níveis 3 e 4."""


def _fora_da_janela(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """[num_envs] bool — True onde o episódio já passou da janela livre."""
    return env.episode_length_buf * env.step_dt >= JANELA_LIVRE_S


def empurrao(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    push,
    asset_cfg: SceneEntityCfg = _ROBO,
) -> None:
    """Chute de velocidade na base, com magnitude escalada por `env.push_fator`.

    É o `push_by_setting_velocity` do fabricante (`mjlab/envs/mdp/events.py:316`) com
    UMA diferença: o range é multiplicado pelo fator por env que o currículo sorteou.
    A dinâmica é idêntica — soma um delta à velocidade do root e escreve de volta.

    Os 6 componentes escalam pelo MESMO fator, como a §14 especifica. Os ranges de
    `knobs.Push` são simétricos, então o código usa o limite superior e simetriza; ler
    os dois lados daria o mesmo resultado com o dobro de linhas.

    Use com `mode="interval"`."""
    env_ids = resolve_env_ids(env, env_ids)
    ids = env_ids[_fora_da_janela(env)[env_ids]]
    if len(ids) == 0:
        return

    asset = env.scene[asset_cfg.name]
    limites = torch.tensor(
        [push.vel_x_full[1], push.vel_y_full[1], push.vel_z_full[1],
         push.roll_full[1], push.pitch_full[1], push.yaw_full[1]],
        device=env.device,
    )
    fator = env.push_fator[ids].unsqueeze(-1)                       # [n, 1]
    delta = (torch.rand(len(ids), 6, device=env.device) * 2.0 - 1.0) * limites * fator
    asset.write_root_link_velocity_to_sim(
        asset.data.root_link_vel_w[ids] + delta, env_ids=ids)


class _CfgInterno:
    """Cfg mínimo para instanciar o `apply_body_impulse`. Ele só lê `params`."""

    def __init__(self, params: dict):
        self.params = params


class empurrao_sustentado:
    """Força segurada na pelve nos níveis 3 e 4 — o "and hold" da escada (S2).

    Envolve o `apply_body_impulse` do mjlab, que já traz o ciclo completo
    (cooldown -> dispara -> sustenta -> expira) com timer independente por env. A
    fiação é a mesma do `skills/stand/env.py:50`, que já roda em treino.

    O que o wrapper acrescenta:

      1. **Gate de nível.** Nos níveis 0 a 2 ele retorna sem escrever nada.
         ⚠️ Não usa `force_range = (0, 0)` para desligar: isso custaria uma escrita de
         wrench por env em todo passo, sem nenhum efeito físico.
      2. **Duração por nível.** `duracao_curta_s` no nível 3, `duracao_longa_s` no 4.
         Do 3 pro 4 o fator NÃO muda — só a duração alonga, como a §14 diz.
      3. **Janela livre.** Zera o wrench dos envs que ainda estão nos primeiros
         `JANELA_LIVRE_S`. A escrita é condicional: só ocorre se houver env na janela.

    ⚠️ **DIVERGÊNCIA da S2, registrada.** A spec pede a magnitude escalada por
    `env.push_fator` (por env). O `apply_body_impulse` sorteia a força com um
    `force_range` GLOBAL (`events.py:524`) e não expõe o wrench escrito para reescala,
    então um fator por env exigiria reimplementar os timers — e a S2 manda reusar.

    O wrapper escala pelo fator do NÍVEL, que é o teto do sorteio de `push_fator`. A
    propriedade que a S2 quer do sorteio continua valendo: o `apply_body_impulse` já
    sorteia cada componente em `U(−F, +F)`, portanto o nível alto CONTÉM o baixo, e o
    eixo não esquece a carga leve. O que se perde é a correlação dentro do episódio —
    aqui cada impulso sorteia sua magnitude, em vez de o env inteiro herdar uma.

    Use com `mode="step"`."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        p = cfg.params["push"]
        self._asset = env.scene[_PELVE.name]
        self._pelve = SceneEntityCfg("robot", body_names="pelvis")
        self._pelve.resolve(env.scene)
        self._body_ids = self._pelve.body_ids
        self._n_bodies = (len(self._body_ids)
                          if isinstance(self._body_ids, list)
                          else self._asset.num_bodies)
        # O `apply_body_impulse` captura `cooldown_s` e `asset_cfg` no __init__ e
        # ignora os kwargs homônimos do __call__ — por isso eles entram aqui.
        self._inner = base_events.apply_body_impulse(
            _CfgInterno({"asset_cfg": self._pelve, "cooldown_s": p.cooldown_s}), env)

    def __call__(self, env: "ManagerBasedRlEnv", env_ids, push) -> None:
        # O nível é GLOBAL (um `push_nivel` escalar no orquestrador), e
        # `env.push_nivel_t` guarda a cópia por env. `.max()` recupera o global sem
        # acoplar este evento ao objeto do currículo.
        nivel = int(env.push_nivel_t.max())
        if nivel < _NIVEL_HOLD:
            return

        fator = float(T.LEVELS["push"][nivel])
        forca = push.force_full * fator
        duracao = (push.duracao_longa_s if nivel >= len(T.LEVELS["push"]) - 1
                   else push.duracao_curta_s)
        self._inner(env, env_ids, force_range=(-forca, forca), torque_range=(0.0, 0.0),
                    duration_s=duracao, cooldown_s=push.cooldown_s,
                    asset_cfg=self._pelve)

        # Janela livre: silencia quem acabou de nascer. Condicional de propósito —
        # depois dos primeiros passos do episódio nenhum env cai aqui, e o custo some.
        na_janela = ~_fora_da_janela(env)
        if bool(na_janela.any()):
            ids = na_janela.nonzero(as_tuple=False).squeeze(-1)
            zeros = torch.zeros(len(ids), self._n_bodies, 3, device=env.device)
            self._asset.write_external_wrench_to_sim(
                zeros, zeros, env_ids=ids, body_ids=self._body_ids)

    def reset(self, env_ids=None):
        pass    # o ciclo do `apply_body_impulse` é contínuo, não por episódio
