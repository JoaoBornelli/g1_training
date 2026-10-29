"""As métricas da locomoção e do mapa, como `MetricsTermCfg` — fora dos termos de recompensa.

⚠ Só importa `mjlab` e `torch` do framework, mais `recompensas.comprimento_do_passo` (§6.3) e,
só para a anotação, `knobs.Knobs`.

POR QUE ELAS SAEM DA RECOMPENSA. O `reward_manager` pula termo com peso 0: desligar uma penalidade
apaga a medição dela, em silêncio. Medir num manager separado desacopla as duas coisas: o peso
decide o INCENTIVO, o `metrics` decide a MEDIÇÃO.

⚠ Cada métrica é POR ENV. O `MetricsManager` divide por `step_count`: uma média por toque (passo_medio)
precisa de `reduce="last"` com a média corrente, como `media_por_estado` no g1_limpo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_yaw

from g1_v3.recompensas import comprimento_do_passo

if TYPE_CHECKING:
    from g1_v3.knobs import Knobs

__all__ = [
    "termos",
    "tempo_de_voo",
    "pico_de_altura",
    "velocidade_de_escorrego",
    "velocidade_de_junta",
    "fracao_do_curso",
    "forca_de_pouso",
    "passo_medio",
    "fracao_mapa_visto",
]

# nomes da cena v3 (`cena.py`): o sensor de contato dos pés, o de altura e os sítios. Os
# nomes são transcritos; um errado aparece como métrica ausente, sem erro.
PES_CHAO = "pes_chao"
ALTURA_DO_PE = "altura_pes"
SITIOS_DOS_PES = ("left_foot", "right_foot")
COMANDO = "twist"


def _media_por_env(valor: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
    """Média de `valor` sobre os pés onde `mascara`, por env. Zero se nenhum pé.

    ⚠ `clamp(min=1)` no denominador, e não `+1e−6`: com nenhum pé mascarado o numerador é
    exatamente 0, portanto o resultado é 0 e não um NaN nem um número enorme.
    """
    m = mascara.float()
    return (valor * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)


def tempo_de_voo(env, sensor_name: str = PES_CHAO) -> torch.Tensor:
    """Tempo de voo médio dos pés que estão NO AR, por env, em segundos.

    ⚠ É a métrica central da F1. Robô imóvel dá 0,0 — nenhum pé no ar, numerador zero.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    t = sensor.data.current_air_time
    assert t is not None, f"{sensor_name} precisa de track_air_time=True"
    return _media_por_env(t, t > 0.0)


class pico_de_altura:
    """Altura de pico do pé no balanço, medida NO POUSO.

    ⚠ Buffer PRÓPRIO, independente do da recompensa: se lesse o buffer do termo de recompensa,
    desligar o peso dele congelaria a métrica.

    ⚠ E ela TEM `reset`: sem ele o pico do pé que estava no ar quando o robô caiu sobreviveria
    ao episódio.
    """

    def __init__(self, cfg, env):
        n_pes = env.scene[cfg.params.get("height_sensor_name", ALTURA_DO_PE)].num_frames
        self.picos = torch.zeros((env.num_envs, n_pes), device=env.device)
        self.dt = env.step_dt

    def __call__(self, env, sensor_name: str = PES_CHAO,
                 height_sensor_name: str = ALTURA_DO_PE) -> torch.Tensor:
        contato: ContactSensor = env.scene[sensor_name]
        alturas = env.scene[height_sensor_name].data.heights
        no_ar = contato.data.found == 0
        self.picos = torch.where(no_ar, torch.maximum(self.picos, alturas), self.picos)
        pousou = contato.compute_first_contact(dt=self.dt)
        valor = _media_por_env(self.picos, pousou)
        self.picos = torch.where(pousou, torch.zeros_like(self.picos), self.picos)
        return valor

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.picos[env_ids] = 0.0


def velocidade_de_escorrego(env, sensor_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Velocidade horizontal do pé APOIADO. Um pé apoiado deveria estar parado."""
    robo: Entity = env.scene[asset_cfg.name]
    contato: ContactSensor = env.scene[sensor_name]
    assert contato.data.found is not None
    v = torch.norm(robo.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
    return _media_por_env(v, contato.data.found > 0)


def velocidade_de_junta(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """RMS de `joint_vel` sobre as 29 juntas, por env. Sem peso."""
    robo: Entity = env.scene[asset_cfg.name]
    v = robo.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sqrt((v ** 2).mean(dim=-1))


def fracao_do_curso(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """O MAIOR `|frac|` sobre as 29 juntas, por env. 1,0 é o batente do MJCF.

        frac = (q − centro do curso) / meio curso

    ⚠ Ela NÃO DUPLICA o custo de limite de junta: o custo satura no teto da rampa, e esta lê a
    POSIÇÃO. Use com `reduce="max"`: a média sobre o episódio esconde o pico.
    """
    robo: Entity = env.scene[asset_cfg.name]
    lim = robo.data.joint_pos_limits[0, asset_cfg.joint_ids]              # (n, 2)
    centro = lim.mean(dim=-1)
    meio = (lim[:, 1] - lim[:, 0]) / 2.0
    q = robo.data.joint_pos[:, asset_cfg.joint_ids]
    return (((q - centro) / meio).abs()).amax(dim=-1)


def forca_de_pouso(env, sensor_name: str = PES_CHAO) -> torch.Tensor:
    """Módulo da força no instante do pouso, em newtons. Proxy de impacto."""
    contato: ContactSensor = env.scene[sensor_name]
    assert contato.data.force is not None
    f = torch.norm(contato.data.force, dim=-1)
    return _media_por_env(f, contato.compute_first_contact(dt=env.step_dt))


class passo_medio:
    """Média, na corrida, do comprimento do passo L no toque, por env, em metros. Sem peso.

    L é o do termo `passo` (PD-16): a projeção, na direção do comando, de (pé que toca − pé de
    apoio), menos a mesma projeção na pose de pé. Passo de lado parado dá o que o comando dá.

    ⚠ Só conta toque com comando ‖cmd_xy‖ ≥ `limiar`, como o termo `passo`: sem comando não há
    passo a medir.

    ⚠ `reduce="last"` no cfg: o valor devolvido é a média CORRENTE desde o reset. A média por passo
    do manager diluiria o toque (que acontece em poucos passos) por `step_count`.

    ⚠ A ordem é (esquerdo, direito) nos três: o `pes_chao` segue a ordem do corpo no MJCF
    (`g1.xml`: `left_ankle_roll_link` antes de `right_ankle_roll_link`), os sítios são
    `SITIOS_DOS_PES` e `pes_de_pe_b` é `PES` em `dados/gera_referencias.py`.
    """

    def __init__(self, cfg, env):
        self.soma = torch.zeros(env.num_envs, device=env.device)
        self.n = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env, sensor_name: str, asset_cfg: SceneEntityCfg,
                 command_name: str, limiar: float) -> torch.Tensor:
        robo: Entity = env.scene[asset_cfg.name]
        contato: ContactSensor = env.scene[sensor_name]
        pousou = contato.compute_first_contact(dt=env.step_dt)                 # (E,2)
        p = robo.data.site_pos_w[:, asset_cfg.site_ids]                        # (E,2,3)
        cmd = env.command_manager.get_command(command_name)[:, :2]
        vivo = cmd.norm(dim=-1) >= limiar                                      # (E,)
        c_b = F.pad(cmd / cmd.norm(dim=-1, keepdim=True).clamp(min=1e-6), (0, 1))
        c_w = quat_apply_yaw(robo.data.root_link_quat_w, c_b)
        p0 = env.v3_ref.pes_de_pe_b                                            # (2,3), frame da pelve
        for i, j in ((0, 1), (1, 0)):
            ok = pousou[:, i] & vivo
            L = comprimento_do_passo(p[:, i], p[:, j], c_w, p0[i], p0[j], c_b)
            self.soma = self.soma + torch.where(ok, L, torch.zeros_like(L))
            self.n = self.n + ok.float()
        return self.soma / self.n.clamp(min=1.0)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.soma[env_ids] = 0.0
        self.n[env_ids] = 0.0


def fracao_mapa_visto(env, limiar: float) -> torch.Tensor:
    """Fração das células do recorte com confiança > `limiar`, por env. Sem peso.

    ⚠ Lê `v3_mapa_obs` (E,882): 441 alturas e depois 441 confianças (I §2.4). Sem o mapa
    (percepção desligada), zero.
    """
    mapa = getattr(env, "v3_mapa_obs", None)
    if mapa is None:
        return torch.zeros(env.num_envs, device=env.device)
    conf = mapa[:, mapa.shape[-1] // 2:]
    return (conf > limiar).float().mean(dim=-1)


def termos(k: Knobs) -> dict[str, MetricsTermCfg]:
    """Os termos, montados aqui e em nenhum outro lugar.

    ⚠ O `SceneEntityCfg` TEM DE VIVER EM `params`: `manager_base` só resolve o que está em
    `term_cfg.params`. Como argumento default ele nunca é resolvido, e a métrica mede o robô
    inteiro no lugar dos pés.

    Fora da lista do g1_limpo, de propósito (decisão do dono, 29/10): `momento_angular` e
    `atraso_medio`.
    """
    return {
        "tempo_de_voo": MetricsTermCfg(func=tempo_de_voo, params={"sensor_name": PES_CHAO}),
        "pico_de_altura": MetricsTermCfg(
            func=pico_de_altura,
            params={"sensor_name": PES_CHAO, "height_sensor_name": ALTURA_DO_PE}),
        "velocidade_de_escorrego": MetricsTermCfg(
            func=velocidade_de_escorrego,
            params={"sensor_name": PES_CHAO,
                    "asset_cfg": SceneEntityCfg("robot", site_names=SITIOS_DOS_PES)}),
        "velocidade_de_junta": MetricsTermCfg(
            func=velocidade_de_junta,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}),
        # ⚠ O PICO, não a média: a pergunta é se alguma junta saiu do curso.
        "fracao_do_curso": MetricsTermCfg(
            func=fracao_do_curso, reduce="max",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}),
        "forca_de_pouso": MetricsTermCfg(func=forca_de_pouso, params={"sensor_name": PES_CHAO}),
        "passo_medio": MetricsTermCfg(
            func=passo_medio, reduce="last",
            params={"sensor_name": PES_CHAO,
                    "asset_cfg": SceneEntityCfg("robot", site_names=SITIOS_DOS_PES),
                    "command_name": COMANDO, "limiar": k.comando.limiar_canal}),
        "fracao_mapa_visto": MetricsTermCfg(
            func=fracao_mapa_visto, params={"limiar": k.percepcao.limiar_visto}),
    }
