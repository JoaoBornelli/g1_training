"""As métricas da locomoção e do mapa, como `MetricsTermCfg` — fora dos termos de recompensa.

⚠ Só importa `mjlab` e `torch` do framework, mais `recompensas.comprimento_do_passo` (§6.3),
`alvos` (os índices P0–B4 e `inclinacao_da_caixa`), `knobs.TAREFAS` (os índices das tarefas) e, só para a
anotação, `knobs.Knobs`.

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

from g1_v3.alvos import B3, B4, P0, P1, P2, P3, inclinacao_da_caixa
from g1_v3.knobs import TAREFAS
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
    "phi_medio_do_alvo",
    "tempo_no_alvo",
    "inclinacao_da_caixa_no_p3",
    "forca_robo_caixa",
    "deslocamento_da_caixa",
    "velocidade_de_chegada",
    "deslocamento_depois_do_apoio",
    "idade_das_leituras",
    "fracao_avisos",
]

# nomes da cena v3 (`cena.py`): o sensor de contato dos pés, o de altura e os sítios. Os
# nomes são transcritos; um errado aparece como métrica ausente, sem erro.
PES_CHAO = "pes_chao"
ALTURA_DO_PE = "altura_pes"
SITIOS_DOS_PES = ("left_foot", "right_foot")
COMANDO = "twist"
CAIXA = "box"
CAIXA_ROBO = "caixa_robo"
# a ordem é a dos índices de `alvos` (P0 = 0 … B4 = 8): o nome do termo sai daqui
_NOMES_ALVOS = ("P0", "P1", "P2", "P3", "P4", "B1", "B2", "B3", "B4")


def _media_por_env(valor: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
    """Média de `valor` sobre os pés onde `mascara`, por env. Zero se nenhum pé.

    ⚠ `clamp(min=1)` no denominador, e não `+1e−6`: com nenhum pé mascarado o numerador é
    exatamente 0, portanto o resultado é 0 e não um NaN nem um número enorme.
    """
    m = mascara.float()
    return (valor * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)


def _media_dos_com_evento(valor: torch.Tensor, com_evento: torch.Tensor) -> torch.Tensor:
    """Média de `valor` sobre os envs com `com_evento`, igual em todo env (E,). Zero sem evento.

    ⚠ O mjlab tira a média sobre todos os envs que resetam (`metrics_manager.py:113-124`): o zero dos envs sem
    evento diluiria a métrica (F9 M5). Uma constante tem média ela mesma, então o reset dá a média dos com evento.
    """
    return torch.ones_like(valor) * _media_por_env(valor[None], com_evento[None])  # um env só no eixo: a média global
    # ⚠ DÚVIDA: a média é sobre TODOS os envs no último passo, e o reset do mjlab a tira sobre o lote que reseta.
    # Igual quando o lote é o todo; se resetarem em passos diferentes, o valor pode diferir da média do lote.


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


PEGAR, CARREGAR, BOTAR = (TAREFAS.index(n) for n in ("PEGAR", "CARREGAR", "BOTAR"))  # índices de `knobs.TAREFAS`


def _com_caixa(env) -> torch.Tensor:
    """(E,) bool: a tarefa do env tem caixa (PEGAR, CARREGAR, BOTAR; a máscara da E5 e da S §10)."""
    t = env.v3_tarefa
    return (t == PEGAR) | (t == CARREGAR) | (t == BOTAR)


class phi_medio_do_alvo:
    """Média do Φ (`v3_phi`) nos passos em que o alvo é `alvo` e a tarefa tem caixa, por env. Sem peso.

    ⚠ Φ é o potencial do passo ANTERIOR (§6.4): o valor do passo em que o `v3_alvo_idx` é `alvo`.
    ⚠ `reduce="last"` no cfg: o valor é a média CORRENTE desde o reset, como `passo_medio`.
    """

    def __init__(self, cfg, env):
        self.soma = torch.zeros(env.num_envs, device=env.device)
        self.n = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env, alvo: int) -> torch.Tensor:
        ok = (env.v3_alvo_idx == alvo) & _com_caixa(env)
        self.soma = self.soma + torch.where(ok, env.v3_phi, torch.zeros_like(self.soma))
        self.n = self.n + ok.float()
        return _media_dos_com_evento(self.soma / self.n.clamp(min=1.0), self.n > 0)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.soma[env_ids] = 0.0
        self.n[env_ids] = 0.0


class tempo_no_alvo:
    """Segundos em que o alvo corrente é `alvo` e a tarefa tem caixa, até o instante, por env. Sem peso.

    ⚠ `reduce="last"` no cfg: o valor é o total corrente, e o último passo é o do episódio.
    """

    def __init__(self, cfg, env):
        self.passos = torch.zeros(env.num_envs, device=env.device)
        self.dt = env.step_dt

    def __call__(self, env, alvo: int) -> torch.Tensor:
        self.passos = self.passos + ((env.v3_alvo_idx == alvo) & _com_caixa(env)).float()
        return _media_dos_com_evento(self.passos * self.dt, self.passos > 0)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.passos[env_ids] = 0.0


def inclinacao_da_caixa_no_p3(env) -> torch.Tensor:
    """Inclinação da caixa (rad) no passo em que o alvo é P3, zero fora dele. Sem peso.

    ⚠ `reduce="max"` no cfg: o PICO no P3 (S §14), e não a média do episódio.
    """
    incl = inclinacao_da_caixa(env.scene[CAIXA].data.root_link_quat_w)
    return torch.where(env.v3_alvo_idx == P3, incl, torch.zeros_like(incl))


def forca_robo_caixa(env, sensor_name: str = CAIXA_ROBO) -> torch.Tensor:
    """Força total robô → caixa, em newtons: a soma dos módulos das fatias do sensor, por env.

    ⚠ Soma de MÓDULOS, e não do vetor: as duas palmas apertam em sentidos opostos (`cena.py`) e a
    soma vetorial se cancela. O `caixa_robo` tem 12 fatias, com força (E, 12, 3).
    """
    f = env.scene[sensor_name].data.force
    return torch.norm(f, dim=-1).sum(dim=-1)


class deslocamento_da_caixa:
    """Quanto a caixa andou no plano xy desde o primeiro passo de P0, nos passos de P0 a P2, por env (m). Sem peso.

    ⚠ A âncora é a posição no PRIMEIRO passo em que o alvo é P0 do episódio. Fora de P0–P2 o valor é zero.
    ⚠ Só xy, como o portão do `recompensas` (E7, M4): a queda da caixa não conta como deslocamento na laje.
    ⚠ `reduce="max"` no cfg: o maior deslocamento do episódio na laje.
    """

    def __init__(self, cfg, env):
        self.ancora = torch.zeros(env.num_envs, 3, device=env.device)
        self.marcada = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env) -> torch.Tensor:
        alvo = env.v3_alvo_idx
        pos = env.scene[CAIXA].data.root_link_pos_w
        novo = (alvo == P0) & ~self.marcada
        self.ancora = torch.where(novo[:, None], pos, self.ancora)
        self.marcada = self.marcada | novo
        na_laje = (alvo == P0) | (alvo == P1) | (alvo == P2)
        desl = torch.norm((pos - self.ancora)[:, :2], dim=-1)
        return torch.where(self.marcada & na_laje, desl, torch.zeros_like(desl))

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.marcada[env_ids] = False


def velocidade_de_chegada(env) -> torch.Tensor:
    """Velocidade de descida da caixa (m/s, positiva) no primeiro contato caixa × laje, por env. Sem peso.

    Lê `v3_v_chegada` (§6.4, F9 P3), o número que o B_alvo[B2] e a `chegada` julgam; −1 = sem toque.
    ⚠ `reduce="last"` no cfg: a média é a dos envs com toque (F9 M5).
    """
    v = env.v3_v_chegada
    return _media_dos_com_evento(v, v >= 0.0)


class deslocamento_depois_do_apoio:
    """Maior deslocamento da caixa no plano xy desde `v3_ponto_apoio`, nos passos de B3 a B4 sem `v3_perdeu`. Sem peso.

    ⚠ O ponto nasce no fecho do B2 (F9 P1): o B2 não entra. Só xy, como `deslocamento_da_caixa`.
    ⚠ `reduce="last"` no cfg, com o maior no próprio buffer: o `max` do mjlab diluiria os envs sem apoio (F9 M5).
    """

    def __init__(self, cfg, env):
        self.maior = torch.zeros(env.num_envs, device=env.device)
        self.marcada = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env) -> torch.Tensor:
        alvo = env.v3_alvo_idx
        ok = ((alvo == B3) | (alvo == B4)) & ~env.v3_perdeu
        desl = torch.norm((env.scene[CAIXA].data.root_link_pos_w - env.v3_ponto_apoio)[:, :2], dim=-1)
        self.maior = torch.where(ok, torch.maximum(self.maior, desl), self.maior)
        self.marcada = self.marcada | ok
        return _media_dos_com_evento(self.maior, self.marcada)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.maior[env_ids] = 0.0
        self.marcada[env_ids] = False


def idade_das_leituras(env) -> torch.Tensor:
    """Idade (s) média da leitura do estimador da caixa, por env. Sem peso.

    A idade é a coluna 7 da `saida_b` (E,8) do `v3_estimador`, com a odometria `v3_odo`.
    ⚠ Zero onde a tarefa não tem caixa (ANDAR): ali a idade é sempre idade_max e polui a média.
    ⚠ Se o `v3_estimador` ainda não existe no env (a fase E ainda não o criou), devolve zeros:
    zero aqui quer dizer "sem medida", e não "leitura nova".
    """
    est = getattr(env, "v3_estimador", None)
    odo = getattr(env, "v3_odo", None)
    if est is None or odo is None:
        return torch.zeros(env.num_envs, device=env.device)
    idade = est.saida_b(odo, env.scene["robot"].data.root_link_quat_w)[:, 7]
    return torch.where(_com_caixa(env), idade, torch.zeros_like(idade))


def fracao_avisos(env) -> torch.Tensor:
    """Fração dos passos com o aviso do detector de fim (`v3_aviso`), por env. Sem peso.

    Sem `reduce` no cfg: o padrão é `sum / step_count`, que é a fração pedida.
    """
    return env.v3_aviso.float()


def termos(k: Knobs) -> dict[str, MetricsTermCfg]:
    """Os termos, montados aqui e em nenhum outro lugar.

    ⚠ O `SceneEntityCfg` TEM DE VIVER EM `params`: `manager_base` só resolve o que está em
    `term_cfg.params`. Como argumento default ele nunca é resolvido, e a métrica mede o robô
    inteiro no lugar dos pés.

    Fora da lista do g1_limpo, de propósito (decisão do dono, 29/10): `momento_angular` e
    `atraso_medio`.
    """
    m = {
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
    # Manipulação (E7, S §14): diagnóstico, sem peso. O Φ e o tempo saem por alvo: nove de cada.
    for i, nome in enumerate(_NOMES_ALVOS):
        m[f"phi_{nome}"] = MetricsTermCfg(func=phi_medio_do_alvo, reduce="last", params={"alvo": i})
        m[f"tempo_{nome}"] = MetricsTermCfg(func=tempo_no_alvo, reduce="last", params={"alvo": i})
    m["inclinacao_caixa_p3"] = MetricsTermCfg(func=inclinacao_da_caixa_no_p3, reduce="max")
    m["forca_robo_caixa"] = MetricsTermCfg(func=forca_robo_caixa, params={"sensor_name": CAIXA_ROBO})
    m["deslocamento_caixa_laje"] = MetricsTermCfg(func=deslocamento_da_caixa, reduce="max")
    m["velocidade_de_chegada"] = MetricsTermCfg(func=velocidade_de_chegada, reduce="last")
    m["deslocamento_depois_do_apoio"] = MetricsTermCfg(func=deslocamento_depois_do_apoio, reduce="last")
    m["idade_leituras"] = MetricsTermCfg(func=idade_das_leituras)
    m["fracao_avisos"] = MetricsTermCfg(func=fracao_avisos)
    return m
