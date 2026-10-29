"""Observações do ator e do crítico da v3 (D2 do plano; §6.3, §6.4, §6.5, §6.6; S §19.1; PD-25, PD-28).

⚠ CONTRATO DO LAYOUT (herdado do `g1_limpo/observacoes.py`): a ordem dos termos é o contrato. Um termo novo
entra SEMPRE por último, e nos dois grupos na mesma ordem. Inserir no meio desloca o peso da primeira camada
em silêncio. O teste `test_ordem_dos_termos_do_ator_e_do_critico` trava a ordem da §6.5.

⚠ O mjlab cria o `DelayBuffer` com a capacidade do cfg (`observation_manager.py:465-475`): por isso os cinco
termos com atraso nascem com `delay_max_lag` 3, e o evento `atraso` (D7) baixa o teto para 1 no startup.

Os termos `n_*` não existem aqui: as observações não têm núcleo de recompensa. Os núcleos puros são
`passo_de_percepcao` (o mapa, sobre tensores) e `referencia_do_alvo`; os demais termos são funções finas
que leem o env. A guarda `atualiza_percepcao` roda o mapa uma vez por passo para o env inteiro (o ator e o
crítico têm cada um um termo `mapa`, e sem a guarda o filtro andaria o dobro).

⚠ DÚVIDAS PENDENTES (ver relatório da D2): ruídos e capacidade de atraso ficam como constantes deste módulo,
e não no `knobs.py`, porque o `knobs` não tem esses campos; os divisores de normalização do crítico também.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.envs import mdp as mdp_env
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.lab_api.math import (axis_angle_from_quat, matrix_from_quat, quat_apply_inverse,
                                      quat_conjugate, quat_mul)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from g1_v3 import alvos, knobs
from g1_v3.balanceador import TAREFAS as CADEIA
from g1_v3.percepcao_lidar import filtra_corpo, pontos_no_frame_do_mapa

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from g1_v3.knobs import Contrato, Percepcao

K = knobs.ATIVO
ANDAR = knobs.TAREFAS.index("ANDAR")
N_TAREFAS = len(knobs.TAREFAS)  # 6 canais one-hot (§6.5 `tarefa`)
N_CAIXA = 11  # I §2.1: canais da caixa do ator (§6.5 `caixa`)
N_ALVO_MACRO = len(knobs.ALVOS) + 1  # P0–B4 e "nenhum" (§6.5 `alvo_macro`)

# ⚠ Ruído do ator, ±, aplicado ANTES da escala (pipeline do mjlab: compute → noise → clip → scale → delay →
# history). Valores de §6.5. Não estão no `knobs` (o Contrato só tem escala e histórico): decisão pendente.
_RUIDO = {"giro_base": 0.2, "gravidade": 0.05, "juntas": 0.01, "vel_juntas": 1.5}
# Atraso 0–1 passo sorteado por episódio nos cinco termos de §6.5 (PD-25). A capacidade 3 é a do cfg (PD-28).
_ATRASO = ("giro_base", "gravidade", "juntas", "vel_juntas", "mapa")
_ATRASO_CAP = 3
# ⚠ 10⁹ passos: o sorteio do lag acontece no 1º passo depois do reset e fica (`delay_buffer.py:247-258`).
_SEM_SORTEIO = 10**9

# Divisores do crítico (§6.5): degrau /5, massa /5 kg. Não há campo no `knobs` para eles.
_ESC_DEGRAU = 5.0
_ESC_MASSA_KG = 5.0

# Pés e cantos: `cantos_sola` (PD-35) dá a ponta mais baixa da sola sem depender de sites no XML.
_PES = ("left_ankle_roll_link", "right_ankle_roll_link")
with np.load(knobs.caminho("g1_v3/dados/capsulas.npz")) as _caps:  # PD-10: 33 cápsulas do MJCF
    _CORPO = [str(s) for s in _caps["corpo"]]
    _P0 = torch.as_tensor(_caps["p0"], dtype=torch.float32)
    _P1 = torch.as_tensor(_caps["p1"], dtype=torch.float32)
    _RAIO = torch.as_tensor(_caps["raio"], dtype=torch.float32)
_CORPO_UNICOS = list(dict.fromkeys(_CORPO))  # `find_bodies` não aceita nome repetido com preserve_order
_IDX_CORPO = torch.tensor([_CORPO_UNICOS.index(n) for n in _CORPO], dtype=torch.long)

# Índices de ALVOS: P0–P3 e B1–B3 usam a pose de IK da pelve e do tronco; P4 usa a pose com caixa; B4, de pé.
_IK = [knobs.ALVOS.index(n) for n in ("P0", "P1", "P2", "P3", "B1", "B2", "B3")]
_P4 = knobs.ALVOS.index("P4")
_B4 = knobs.ALVOS.index("B4")


# --- núcleos puros (sobre tensores; o teste chama estes) -----------------------------------------------------

def passo_de_percepcao(filtro, dist, dirs_s, sensor_pos_w, sensor_rot_w, base_pos_w, rumo, z_pe_min_w,
                       incl_imu, capsulas_w, raios, dxy_b, dpsi, z_ref, dt, k: Percepcao, gerador
                       ) -> torch.Tensor:
    """Uma leitura do lidar → mapa de 882 canais (I §2.4, PERC §0). Ordem: pontos → corpo → transporte → Kalman.

    ⚠ `prediz` tem de vir ANTES de `atualiza`: o transporte move o mapa que a leitura nova vai preencher.
    ⚠ `gerador` None desliga o ruído e a perda (teste determinístico, §6.3).
    """
    pontos, pontos_w, validos = pontos_no_frame_do_mapa(dist, dirs_s, sensor_pos_w, sensor_rot_w, base_pos_w,
                                                        rumo, z_pe_min_w, incl_imu, k, gerador)
    validos = filtra_corpo(pontos_w, validos, capsulas_w, raios, sensor_pos_w, k.raio_min, k.margem_corpo)
    filtro.prediz(dxy_b, dpsi, z_ref, dt)
    filtro.atualiza(pontos, validos, dist)
    return filtro.recorte()


def referencia_do_alvo(indice, h, a, ref: alvos.Referencias, k: knobs.Alvos) -> torch.Tensor:
    """Canais da referência do alvo corrente (E,8): palmas (6), pelve (1) e tronco (1). §6.6.

    ⚠ Zero (nunca NaN) com alvo −1: `torch.where` com ramo zero, e não multiplicação (NaN × 0 = NaN).
    ⚠ `torch.where` avalia os dois ramos; por isso `ref.pelve_z` roda para todo env mesmo fora de P0–B3.
    """
    palmas = alvos.palmas_alvo_caixa(indice, a, k).reshape(-1, 6)
    ik = torch.isin(indice, torch.tensor(_IK, device=indice.device))
    p4 = indice == _P4
    b4 = indice == _B4
    z = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
    pelve = torch.where(ik, ref.pelve_z(h, a),
                        torch.where(p4, ref.pelve_com_caixa(a), torch.where(b4, z + ref.pelve_de_pe, z)))
    tronco = torch.where(ik, ref.tronco(h, a), z)
    return torch.cat([palmas, pelve[:, None], tronco[:, None]], -1).to(palmas.dtype)


# --- invólucros (leem o env; o crítico e o ator usam os mesmos) ------------------------------------------------

def um_de_seis(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One-hot de `v3_tarefa` em TAREFAS (6 canais, §6.5 `tarefa`). ⚠ Sem ruído: ruído faria estados mistos."""
    return torch.nn.functional.one_hot(env.v3_tarefa, num_classes=N_TAREFAS).float()


def caixa_zero(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Zeros (E,11) enquanto a fase D não tem a caixa estimada. A fase E troca por `caixa` (E8)."""
    return torch.zeros(env.num_envs, N_CAIXA, device=env.device)


def atualiza_percepcao(env: ManagerBasedRlEnv) -> None:
    """Guarda: `_calcula_percepcao` roda uma vez por `common_step_counter`; no mesmo passo, só volta (§6.3)."""
    if env.v3_percepcao_passo == env.common_step_counter:
        return
    env.v3_mapa_obs = _calcula_percepcao(env)
    env.v3_percepcao_passo = env.common_step_counter


def mapa(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Termo `mapa` do ator e do crítico: a guarda e o buffer. Histórico 0 e atraso 0–1 (§6.5)."""
    atualiza_percepcao(env)
    return env.v3_mapa_obs


def _calcula_percepcao(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Lê o env e chama `passo_de_percepcao`. Também gira o padrão do MID-360 para o próximo passo (PERC §5).

    ⚠ O padrão lido agora é `direcoes(passo)`, e o copiado para o sensor é `direcoes(passo + 1)`: o raio
    do passo seguinte já sai na direção certa. ⚠ `z_ref` usa o pé mais baixo do frame da odometria (PD-7):
    a altura estimada da base mais a altura relativa do pé, medida no sim.
    """
    robo = env.scene["robot"]
    lidar = env.scene.sensors["lidar"]
    passo = env.common_step_counter
    dirs_s = env.v3_padrao.direcoes(passo)
    lidar._local_directions.copy_(env.v3_padrao.direcoes(passo + 1))
    base_pos, base_quat = robo.data.root_link_pos_w, robo.data.root_link_quat_w
    z_pe = _z_pe_min(robo, env.v3_ref)
    capsulas_w, raios = _capsulas_w(robo)
    dxy_b, dpsi, _dz = env.v3_odo.delta_b()
    z_ref = env.v3_odo.pos_w[:, 2] + (z_pe - base_pos[:, 2])
    # ⚠ DÚVIDA (relatório): o gerador de ruído não tem dono na §6. `None` = sem ruído de lidar no treino.
    return passo_de_percepcao(env.v3_filtro, lidar.data.distances, dirs_s, lidar.data.pos_w,
                              matrix_from_quat(lidar.data.quat_w), base_pos, _rumo(base_quat), z_pe,
                              env.v3_incl_imu, capsulas_w, raios, dxy_b, dpsi, z_ref, env.step_dt,
                              K.percepcao, None)


def _rumo(q: torch.Tensor) -> torch.Tensor:
    """Yaw do quaternion wxyz (E,4) → (E,) rad. Mesma conta do `_yaw` do estimador."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _z_pe_min(robo, ref: alvos.Referencias) -> torch.Tensor:
    """Altura (E,) da ponta mais baixa das duas solas no mundo: cantos de `cantos_sola` nos tornozelos."""
    ids, _ = robo.find_bodies(_PES, preserve_order=True)
    pos = robo.data.body_link_pos_w[:, ids]  # (E,2,3)
    rot = matrix_from_quat(robo.data.body_link_quat_w[:, ids])  # (E,2,3,3)
    cantos = torch.einsum("eaij,cj->eaci", rot, ref.cantos_sola.to(pos))  # (E,2,4,3)
    return (pos[:, :, None] + cantos)[..., 2].amin(dim=(1, 2))


def _capsulas_w(robo) -> tuple[torch.Tensor, torch.Tensor]:
    """Cápsulas do corpo no mundo: (E,33,2,3) com os extremos p0, p1, e os raios (33,). PD-10.

    ⚠ `capsulas.npz` repete o nome do corpo (33 cápsulas, menos corpos): a pose sai por índice, e não por
    `find_bodies` direto, que devolveria um corpo só uma vez.
    """
    ids, _ = robo.find_bodies(_CORPO_UNICOS, preserve_order=True)
    idx = _IDX_CORPO.to(robo.data.body_link_pos_w.device)
    pos = robo.data.body_link_pos_w[:, ids][:, idx]  # (E,33,3)
    rot = matrix_from_quat(robo.data.body_link_quat_w[:, ids][:, idx])  # (E,33,3,3)
    p0 = pos + torch.einsum("eaij,aj->eai", rot, _P0.to(pos))
    p1 = pos + torch.einsum("eaij,aj->eai", rot, _P1.to(pos))
    return torch.stack([p0, p1], dim=2), _RAIO.to(pos)


def _no_frame_da_base(env: ManagerBasedRlEnv, p_w: torch.Tensor) -> torch.Tensor:
    """Vetor de mundo (E,3) no frame da base: `R(q_base)⁻¹ · (p − p_base)`. ⚠ Só para pontos, não velocidades."""
    robo = env.scene["robot"]
    return quat_apply_inverse(robo.data.root_link_quat_w, p_w - robo.data.root_link_pos_w)


def caixa_verdade(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Caixa real (E,7): posição (3), rotvec (3) e meia aresta (1), no frame da base. Zero no ANDAR. §6.5.

    ⚠ DÚVIDA (relatório): a §6.5 não diz a composição dos 7 canais; escolhi o análogo verdadeiro das três
    primeiras partes do `saida_b` do estimador (pos_b, rotvec, meia, sem a idade).
    """
    robo, caixa = env.scene["robot"], env.scene["box"]
    rel = quat_mul(quat_conjugate(robo.data.root_link_quat_w), caixa.data.root_link_quat_w)
    val = torch.cat([_no_frame_da_base(env, caixa.data.root_link_pos_w), axis_angle_from_quat(rel),
                     env.v3_meia[:, None]], -1)
    return torch.where((env.v3_tarefa == ANDAR)[:, None], torch.zeros_like(val), val)


def alvo_verdade(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Alvo corrente (E,3) no frame da base; zero com alvo −1 (§6.6). `v3_alvo_w` é do comando (§6.4)."""
    val = _no_frame_da_base(env, env.v3_alvo_w)
    return torch.where((env.v3_alvo_idx >= 0)[:, None], val, torch.zeros_like(val))


def alvo_macro(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One-hot (E,10) de P0–B4 e "nenhum". O "nenhum" é o último canal (índice −1 vira 9)."""
    idx = env.v3_alvo_idx
    slot = torch.where(idx < 0, torch.full_like(idx, N_ALVO_MACRO - 1), idx)
    return torch.nn.functional.one_hot(slot, num_classes=N_ALVO_MACRO).float()


def referencia(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Canais (E,8) da referência do alvo, por `referencia_do_alvo` (§6.6). Usa `v3_topo`, `v3_meia` e `v3_ref`."""
    return referencia_do_alvo(env.v3_alvo_idx, env.v3_topo, env.v3_meia, env.v3_ref, K.alvos)


def niveis(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Nível/máximo de cada tarefa da cadeia (E,4). O balanceador é global, então o valor é o mesmo em todo env.

    ⚠ DÚVIDA (relatório): o divisor é o último índice do nível (`balanceador.Curriculo._max`, mesma conta),
    e a §6.5 diz só "nível/n". Lido do `knobs`, sem reabrir o `Curriculo`.
    """
    teto = {"ANDAR": len(K.comando.frente_tetos) - 1, "CARREGAR": len(K.comando.frente_tetos) - 1,
            "PEGAR": len(K.niveis.pegar) - 1, "BOTAR": len(K.niveis.botar) - 1}
    val = torch.tensor([[env.v3_cur.nivel(t) / teto[t] for t in CADEIA]], dtype=torch.float32,
                       device=env.device)
    return val.expand(env.num_envs, -1).clone()


def degrau(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Degrau do refino da tarefa corrente, em (E,1), dividido por 5 (§6.5)."""
    return env.v3_degrau.float()[:, None] / _ESC_DEGRAU


def massa(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Massa da caixa (E,1), em kg dividida por 5 (§6.5). `v3_massa` é escrito pelo `eventos`."""
    return env.v3_massa[:, None] / _ESC_MASSA_KG


def forca_caixa(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Força externa na caixa (E,1), em N dividida por 100 (§6.5).

    ⚠ DÚVIDA (relatório): não há buffer `v3_*` para a força na §6.4; o evento `forca_caixa` só entra na fase H.
    Até lá, zero: o canal existe com a forma certa e não tem o que ler.
    """
    return torch.zeros(env.num_envs, 1, device=env.device)


def tarefa_critico(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One-hot (E,6) da tarefa, sem histórico. É o último termo: o grupo da vantagem (D9) lê os seis últimos."""
    return torch.nn.functional.one_hot(env.v3_tarefa, num_classes=N_TAREFAS).float()


# --- grupos ----------------------------------------------------------------------------------------------------

def grupos(k: Contrato) -> dict[str, ObservationGroupCfg]:
    """Os grupos `actor` (com corrupção) e `critic` (sem corrupção), com os termos de §6.5 na ordem.

    Escala e histórico vêm do `Contrato.blocos_ator` (I §2.1). O crítico usa os mesmos nove termos do ator,
    sem ruído e sem atraso, e depois os canais privilegiados.
    """
    escala = {n: e for n, _c, e, _h in k.blocos_ator}
    historico = {n: h for n, _c, _e, h in k.blocos_ator}

    def termo(nome, func, params=None, ator=False, hist_padrao=0):
        # ator: ruído e atraso, nos termos das listas. Crítico: nem um nem outro (§6.5).
        ruido = _RUIDO.get(nome) if ator else None
        atraso = ator and nome in _ATRASO
        cfg = ObservationTermCfg(
            func=func, params=params or {}, scale=escala.get(nome, 1.0),
            history_length=historico.get(nome, hist_padrao),
            noise=Unoise(n_min=-ruido, n_max=ruido) if ruido else None,
        )
        if atraso:
            cfg.delay_min_lag = 0
            cfg.delay_max_lag = _ATRASO_CAP
            cfg.delay_update_period = _SEM_SORTEIO
            cfg.delay_per_env_phase = False
        return cfg

    nove = [  # (nome, função, params do ator); o crítico tira o viés do juntas (joint_pos sem `biased`)
        ("giro_base", mdp_env.builtin_sensor, {"sensor_name": "robot/imu_ang_vel"}),
        ("gravidade", mdp_env.projected_gravity, {}),
        ("comando", mdp_env.generated_commands, {"command_name": "twist"}),
        ("juntas", mdp_env.joint_pos_rel, {"biased": True}),
        ("vel_juntas", mdp_env.joint_vel_rel, {}),
        ("acao", mdp_env.last_action, {}),
        ("tarefa", um_de_seis, {}),
        ("caixa", caixa_zero, {}),
        ("mapa", mapa, {}),
    ]
    ator = {n: termo(n, f, p, ator=True) for n, f, p in nove}
    critico = {n: termo(n, f, {} if n == "juntas" else p) for n, f, p in nove}
    critico.update({  # sem histórico, salvo `vel_lin_base` (histórico 5, §6.5)
        "vel_lin_base": termo("vel_lin_base", mdp_env.builtin_sensor, {"sensor_name": "robot/imu_lin_vel"},
                              hist_padrao=5),
        "caixa_verdade": termo("caixa_verdade", caixa_verdade),
        "alvo_verdade": termo("alvo_verdade", alvo_verdade),
        "alvo_macro": termo("alvo_macro", alvo_macro),
        "referencia": termo("referencia", referencia),
        "niveis": termo("niveis", niveis),
        "degrau": termo("degrau", degrau),
        "massa": termo("massa", massa),
        "forca_caixa": termo("forca_caixa", forca_caixa),
        "tarefa_critico": termo("tarefa_critico", tarefa_critico),
    })
    return {
        "actor": ObservationGroupCfg(terms=ator, concatenate_terms=True, enable_corruption=True),
        "critic": ObservationGroupCfg(terms=critico, concatenate_terms=True, enable_corruption=False),
    }
