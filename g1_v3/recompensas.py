"""Termos de recompensa da locomoção da v3, fases D e E (plano §6.5, D5, E4; PD-16, PD-19, PD-35 a PD-37, PD-45,
PD-48, PD-49).

Cada termo é um núcleo puro sobre tensores (`n_*`, ou o nome do §6.3 quando ele não tem `n_`) mais um
invólucro fino que lê o env. O teste chama o núcleo com tensores sintéticos. Nesta fase: os rastreios, o
passo, o vivo, a altura do pé, o escorregão, as taxas e a energia, os limites e os desvios, o tronco, a
altura da base, as seis cercas (sola, giro, tronco, CoM, torque, pelve, recuo) e o contato fora dos pés.
Fase E (E4): o progresso (Φ do alvo corrente), a qualidade do rastreio no CARREGAR, o contato com a laje, o
freio, a perda da caixa, e o contato com a caixa fora da janela da pega. A sacudida e o aperto são da fase H (H2):
custos, sem `paga`, com o fator d/5 do degrau da tarefa no próprio termo.
⚠ Todo termo recebe `tarefas` e vale zero fora delas com `torch.where` (§6.6). Nunca multiplicar pela
máscara: NaN × 0 = NaN, e o NaN de um env fora da tarefa chega à recompensa e ao ator.
⚠ Os termos que pagam positivo (`vivo`, `altura_pe`, `passo`, rastreios) zeram depois da caixa perdida
(`v3_perdeu`, S §10). Os pesos moram no `knobs` (no `RewardTermCfg`): o termo devolve a grandeza, sem sinal.
Unidades: ângulos em graus nas cercas (a cerca de giro recebe rad, §6.3), comprimentos em metros, torque
em fração de τ_max ou de `tau_par` (§6.3).
"""
from __future__ import annotations

import math
import re

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from g1_v3 import alvos, balanceador, estimador, potencial
from g1_v3.cena import MESA_POR_GRUPO
from g1_v3.curriculo import mascara
from g1_v3.knobs import ALVOS, ATIVO, JUNTAS, TAREFAS, TAU_MAX

R = ATIVO.recompensa
C = ATIVO.comando
K = ATIVO.alvos
_EPS = 1e-6  # tolerância de "sobre a reta" no fecho convexo do CoM (metros)

_PES_CORPOS = ("left_ankle_roll_link", "right_ankle_roll_link")  # PD-21: altura e velocidade do pé
_BRACOS = tuple(i for i, j in enumerate(JUNTAS) if re.search(r"shoulder|elbow|wrist", j))  # 14
_CINTURA = tuple(i for i, j in enumerate(JUNTAS) if "waist" in j)  # 3
_QUADRIL = tuple(i for i, j in enumerate(JUNTAS) if re.search(r"hip_(roll|yaw)", j))  # 4
_YAW_E, _YAW_D = JUNTAS.index("left_hip_yaw_joint"), JUNTAS.index("right_hip_yaw_joint")
_VMAX = tuple(  # E4: tetos de velocidade por junta, um tuplo por regime (standing, andando, correndo)
    tuple(next(v for rx, v in tab if re.fullmatch(rx, j)) for j in JUNTAS)
    for tab in (R.vel_max_parado, R.vel_max_andando, R.vel_max_correndo))
_PALMAS_SITIOS = ("left_palm", "right_palm")  # E4: ordem = `palmas_alvo_caixa` (0 = +y, 1 = −y)
_PALMAS_SENSORES = ("palma_E", "palma_D")  # E4: sensores de contato das palmas com a caixa (cena.py)


def _ix(*nomes):
    """Índices de ALVOS pelos nomes (E4)."""
    return tuple(ALVOS.index(n) for n in nomes)


_P4, _B1, _B2, _B4 = ALVOS.index("P4"), ALVOS.index("B1"), ALVOS.index("B2"), ALVOS.index("B4")
_P0_P2, _B3_B4 = _ix("P0", "P1", "P2"), _ix("B3", "B4")  # E4: portão do deslocamento (pega e apoio)
_INCL, _MAOS = _ix("P2", "P3", "P4", "B1", "B2", "B3"), _ix("P3", "P4", "B1", "B2")  # 13/11, A1/A6: incl; mãos
_POUSO, _P3_B3 = _ix("B1", "B2"), _ix("P3", "P4", "B1", "B2", "B3")  # E4: pouso; 13/11, A7: caixa no CoM
_PALMA_ALVO, _CAIXA_ALVO = _ix("P0", "P1", "P2", "B3"), _ix("P3", "P4", "B1", "B2")  # E4: distância por grupo
_B1_B3, _JANELA_PEGA = _ix("B1", "B2", "B3"), _ix("P1", "P2", "P3", "P4", "B1", "B2", "B3")  # E4: topo; 13/11, A2: janela (P1 = palma encostada)
_PARES = torch.tensor([[JUNTAS.index(a), JUNTAS.index(b)] for a, b in R.pares_torque],
                      dtype=torch.long)  # PD-45: (waist_pitch, waist_roll) e os dois pares do tornozelo


# ---------------------------------------------------------------- núcleos (tensores, sem env)

def comprimento_do_passo(p_toque_w, p_apoio_w, c_w, p0_toque_b, p0_apoio_b, c_b) -> torch.Tensor:
    """L = (p_toque − p_apoio)·ĉ_w − (p0_toque − p0_apoio)·ĉ_b (PD-16). Posições (E,3) ou (3,); ĉ unitário.

    ⚠ A segunda parcela tira a largura da pose de pé. Sem ela, o passo de lado recebe a largura da base
    (0,23 m) de graça, e o termo paga um passo que o robô não deu.
    """
    return ((p_toque_w - p_apoio_w) * c_w).sum(-1) - ((p0_toque_b - p0_apoio_b) * c_b).sum(-1)


def n_passo(contato, fase, on, fase_apoio) -> torch.Tensor:
    """(E,2) 1 onde o contato do pé bate com o relógio (13/11, M4; unitree_rl_lab `feet_gait`). Pé esquerdo na `fase`,
    direito na fase + 0,5; apoio com fase < `fase_apoio`. Com `on` False (comando zero), os dois pés em apoio."""
    fases = torch.stack([fase, torch.remainder(fase + 0.5, 1.0)], dim=-1)
    apoio = (fases < fase_apoio) | ~on.unsqueeze(-1)
    return (contato == apoio).float()


def n_vivo(perdeu) -> torch.Tensor:
    """1 enquanto a caixa não caiu (S §9.2). Zera com `v3_perdeu` (§6.6)."""
    return 1.0 - perdeu.float()


def n_altura_pe(h, v_xy, alvo, std, tanh_k) -> torch.Tensor:
    """exp(−Σ_pés (h − alvo)²·tanh(k·‖v_xy‖)/std): o pé parado vale 1 e o pé que anda no ar é premiado (S §9.2).

    h (E,2) altura do pé sobre o terreno; v_xy (E,2,2) velocidade horizontal dos tornozelos. O `tanh` faz o
    termo valer zero para o pé parado no chão, que não precisa de altura (foot_clearance_reward).
    """
    custo = (h - alvo).pow(2) * torch.tanh(tanh_k * v_xy.norm(dim=-1)) / std
    return torch.exp(-custo.sum(-1))


def n_escorrego(v_xy, contato) -> torch.Tensor:
    """Σ_pés ‖v_xy‖ · contato (S §9.2): o pé que desliza no chão. v_xy (E,2,2); contato (E,2) bool."""
    v = v_xy.norm(dim=-1)
    return torch.where(contato, v, torch.zeros_like(v)).sum(-1)


def n_vz(v_z) -> torch.Tensor:
    """v_z² da base (receita da Unitree; mjlab `lin_vel_z_l2`)."""
    return v_z.pow(2)


def n_giro_xy(g_xy) -> torch.Tensor:
    """ω_x² + ω_y² da base, no frame da base (mjlab `ang_vel_xy_l2`)."""
    return g_xy.pow(2).sum(-1)


def n_vel_junta(qd) -> torch.Tensor:
    """Σ q̇² nas 29 juntas (mjlab `joint_vel_l2`)."""
    return qd.pow(2).sum(-1)


def n_acel_junta(qdd) -> torch.Tensor:
    """Σ q̈² nas 29 juntas (mjlab `joint_acc_l2`)."""
    return qdd.pow(2).sum(-1)


def n_taxa_acao(a, a_ant) -> torch.Tensor:
    """Σ (a − a_anterior)² sobre a saída crua da política (mjlab `action_rate_l2`)."""
    return (a - a_ant).pow(2).sum(-1)


def n_energia(tau, qd) -> torch.Tensor:
    """Σ |τ·q̇| com τ = `qfrc_actuator` das juntas (S §9.2)."""
    return (tau * qd).abs().sum(-1)


def n_limite_junta(q, lo, hi, mole) -> torch.Tensor:
    """Excesso além do limite mole de `mole` do curso (S §9.2): o núcleo é `potencial.excesso_de_junta`."""
    return potencial.excesso_de_junta(q, (lo + hi) / 2.0, (hi - lo) / 2.0, mole)


def n_desvio(q, alvo, idx) -> torch.Tensor:
    """Σ |q − alvo| nas juntas `idx` (braços, cintura ou quadril, S §9.2). q e alvo (E,29)."""
    return (q[:, list(idx)] - alvo[:, list(idx)]).abs().sum(-1)


def n_tronco_plano(g_xy) -> torch.Tensor:
    """Gravidade projetada no plano da base, ‖g_xy‖² (mjlab `flat_orientation_l2`). g_xy (E,2)."""
    return g_xy.pow(2).sum(-1)


def n_altura_base(h_pelve, alvo) -> torch.Tensor:
    """(h_pelve − alvo)² com `alvo_base` (0,78 m, S §9.2). h_pelve: altura sobre o terreno."""
    return (h_pelve - alvo).pow(2)


def n_cerca_sola(incl_graus, contato, lim_graus) -> torch.Tensor:
    """Σ_pés em contato relu(incl − lim) (S §9.3): a sola inteira no chão. incl (E,2); contato (E,2) bool."""
    return torch.where(contato, torch.relu(incl_graus - lim_graus), torch.zeros_like(incl_graus)).sum(-1)


def n_cerca_giro(q_yaw_E, q_yaw_D, lim_graus) -> torch.Tensor:
    """relu(max(−q_E, 0)° − lim) + relu(max(q_D, 0)° − lim) (S §9.3). Entrada em rad; lim em graus.

    ⚠ Só o lado que gira para dentro conta: o hip_yaw de cada perna tem o sinal oposto no giro do tronco.
    """
    dE = torch.rad2deg(torch.clamp(-q_yaw_E, min=0.0))
    dD = torch.rad2deg(torch.clamp(q_yaw_D, min=0.0))
    return torch.relu(dE - lim_graus) + torch.relu(dD - lim_graus)


def n_cerca_tronco(incl_graus, tronco_ik_graus, lim_graus) -> torch.Tensor:
    """relu(incl − (tronco_IK + lim)) (S §9.3). tronco_IK é 0 no ANDAR e no CARREGAR (§6.6 da D5)."""
    return torch.relu(incl_graus - (tronco_ik_graus + lim_graus))


def n_cerca_com(com_xy, cantos_xy, dois_pes, lim) -> torch.Tensor:
    """relu(lim − margem)·dois_pes (PD-35; S §9.3). com_xy (E,2): CoM combinado; cantos_xy (E,K,2); dois_pes (E,)."""
    m = margem_com(com_xy, cantos_xy)
    return torch.where(dois_pes, torch.relu(lim - m), torch.zeros_like(m))


def n_cerca_torque(tau, tau_max, pares, tau_par, lim, banda) -> torch.Tensor:
    """Σ_soltas relu(|τ|/τ_max − lim)/banda + Σ_pares relu((|τ_p| + |τ_r|)/tau_par − lim)/banda (PD-45, S §9.3).

    tau (E,29); tau_max (29,); pares (P,2) long. ⚠ Nos pares, os dois motores 5020 movem o pitch e o roll
    juntos: a soma é o que o motor real aguenta. A junta do par não entra como solta (sem conta dupla).
    """
    f = tau.abs() / tau_max
    no_par = torch.zeros(tau.shape[-1], dtype=torch.bool, device=tau.device)
    no_par[pares.reshape(-1)] = True
    soltas = torch.where(no_par, torch.zeros_like(f), torch.relu(f - lim)).sum(-1)
    par = torch.relu((tau[:, pares[:, 0]].abs() + tau[:, pares[:, 1]].abs()) / tau_par - lim).sum(-1)
    return (soltas + par) / banda


def n_cerca_pelve(h_pelve, pelve_ik, lim) -> torch.Tensor:
    """relu(|h − ik| − lim) (S §9.3). h e ik em metros sobre o terreno."""
    return torch.relu((h_pelve - pelve_ik).abs() - lim)


def n_cerca_recuo(recuo, recuo_ik, lim) -> torch.Tensor:
    """relu(|r − ik| − lim) (S §9.3). `recuo` é a distância no plano, pelve → caixa, em m (29/10)."""
    return torch.relu((recuo - recuo_ik).abs() - lim)


def n_contato_fora(forca_N, colisoes, limiar_N) -> torch.Tensor:
    """Contatos do robô fora dos pés com força > limiar, mais os corpos em autocolisão (S §9.2).

    forca_N (E,S): norma da força de cada contato com o chão; colisoes (E,): corpos em autocolisão.
    """
    return (forca_N > limiar_N).float().sum(-1) + colisoes.float()


def n_terminacao(terminado, tempo_esgotado) -> torch.Tensor:
    """1 numa terminação que não é `time_out` (S §9.2). Peso −200 no knob (× dt, na cfg)."""
    return (terminado & ~tempo_esgotado).float()


def _em(indice, grupo):
    """(E,) bool: o índice do alvo está no grupo. `isin`, e não máscara por multiplicação (§6.6)."""
    return torch.isin(indice, torch.tensor(grupo, device=indice.device))


def distancia_do_alvo(indice, palmas_c, caixa_w, caixa_alvo_w, dpelve, dtronco_graus, dpalmas_b4, deficit_apoio, a,
                      k) -> torch.Tensor:
    """d do alvo corrente, na medida de §6.3 (E4). P0–P2 e B3: média das duas palmas no frame da caixa; P3 e B1:
    ‖caixa − alvo‖; P4 (11/11, §6.4): √(‖caixa − alvo‖² + (Δpelve·tol_caixa/tol_pelve)²), a caixa e a pelve;
    B2 (13/11, A3): hypot(‖caixa − alvo‖, escala_apoio_b2 · deficit_apoio), o apoio que falta entra na distância;
    B4: √((Δpelve/tol)² + (Δtronco/tol)² + (dpalmas_b4/tol_palmas_b4)²) (adimensional; 13/11, A9: as mãos ao lado do
    tronco); −1: 0 (nunca NaN, §6.6).

    palmas_c (E,2,3) no frame da caixa (PD-29); dpelve e dtronco_graus (E,); dpalmas_b4 (E,): a maior das duas
    distâncias palma–`palmas_de_pe_b` no frame da base; deficit_apoio (E,) em [0, 1] = relu(1 − apoio/(apoio_b2_mg·peso)).
    Mesma conta de `parte0/item1_phi` quando os dois últimos valem 0.
    """
    d_palma = (palmas_c - alvos.palmas_alvo_caixa(indice, a, k)).norm(dim=-1).mean(-1)
    d_caixa = (caixa_w - caixa_alvo_w).norm(dim=-1)
    d_pose = torch.hypot(torch.hypot(dpelve / k.tol_pelve_b4, dtronco_graus / k.tol_tronco_b4_graus),
                         dpalmas_b4 / k.tol_palmas_b4)
    d_p4 = torch.hypot(d_caixa, dpelve * (k.tol_caixa_p4 / k.tol_pelve_p4))
    d_b2 = torch.hypot(d_caixa, k.escala_apoio_b2 * deficit_apoio)
    d = torch.where(_em(indice, _PALMA_ALVO), d_palma,
                    torch.where(indice == _P4, d_p4,
                                torch.where(indice == _B2, d_b2, torch.where(_em(indice, _CAIXA_ALVO), d_caixa, d_pose))))
    return torch.where(indice >= 0, d, torch.zeros_like(d))


def portoes_do_alvo(indice, desloc_pega, desloc_apoio, incl_graus, incl_lim_graus, maos, desvio_tronco_graus,
                    desvio_recuo, desvio_com, dois_pes, a, k) -> dict:
    """Portões de Φ, cada um em [0, 1] (E4; §6.2 `Alvos`, PD-19, PD-35). 1 fora do intervalo em que o portão vale.

    "desloc": P0–P2 pela caixa na pega, B3–B4 pelo ponto de apoio; "incl": P2–B3 (13/11, A1/A6: o B3 exige a caixa
    nivelada, §4.5), com `incl_lim_graus` do degrau;
    "maos": P3–B2 (13/11, A1: o Φ do P2 puxa a palma à face até o contato, sem exigir força), o valor `maos` (E,) em
    [0, 1] de quem chama: a cadeia passa contatos/2 (PD-19; o invariante exige 1) e o `_phi_env` passa a firmeza da
    pega, tanh(min F / F_pega) (11/11; S §9.1: rampa por grandeza).
    "tronco": faixa do tronco, do lado de fora só, 1 no B4 (F9 P2); "recuo": fora
    do B4; "com": só com os dois pés (PD-35). Desvios são não negativos (quem chama tira o módulo; o tronco entra
    com relu(incl − IK), S §8.2). Os intervalos são os de `parte0/item1_phi`.
    "com": `desvio_com` = relu(faixa_com − margem), 1 com margem ≥ faixa_com e 0 com margem ≤ 0 (R2, 29/10).
    """
    rampa, um = potencial.rampa, torch.ones_like(desloc_pega)
    desloc = torch.where(_em(indice, _P0_P2), rampa(desloc_pega, k.desloc_p0p2_frac * a, k.delta_p0p2_frac * a),
                         torch.where(_em(indice, _B3_B4), rampa(desloc_apoio, k.desloc_b2b4, k.delta_b2b4), um))
    return dict(
        desloc=desloc,
        incl=torch.where(_em(indice, _INCL), rampa(incl_graus, incl_lim_graus, k.delta_incl_graus), um),
        maos=torch.where(_em(indice, _MAOS), maos, um),
        tronco=torch.where(indice == _B4, um, rampa(desvio_tronco_graus, k.faixa_tronco_graus, k.delta_tronco_graus)),
        recuo=torch.where(indice == _B4, um, rampa(desvio_recuo, k.faixa_recuo, k.delta_recuo)),
        com=torch.where(dois_pes, rampa(desvio_com, 0.0, k.delta_com), um),
    )


def phi_do_alvo(indice, d, sigma, fator, portoes: dict) -> torch.Tensor:
    """Φ = (1 − tanh(d / (largura·σ))) · fator · Π portões (S §9.1; E4). `fator` é o pouso no B1 e no B2, 1 nos outros.

    −1 → 0. σ tem piso de `_EPS` só no denominador: com σ = 0 fora do alvo o gradiente daria NaN (§6.6).
    10/11: a gaussiana ficava plana perto do alvo e depois de ~2σ (canal morto); a tanh inclina em toda distância.
    O núcleo fica em [0, 1]: com um portão em zero, o Φ nunca sobe.
    """
    nuc = 1 - torch.tanh(d / (K.largura_tanh * sigma.clamp(min=_EPS)))
    phi = potencial.phi(nuc * fator, list(portoes.values()))
    return torch.where(indice >= 0, phi, torch.zeros_like(phi))


def n_progresso(phi, etapa, w_phi, w_etapa, h) -> torch.Tensor:
    """Renda por segundo: w_etapa·etapa·h + w_phi·Φ. Cresce com o tempo parado no alvo, como os custos (que são por
    segundo); o termo antigo (diferença de Φ + prêmio por avanço) pagava um valor fixo e o robô parava na frente da caixa.
    13/11, M5: o h em [0, 1] só multiplica a parte da etapa e cai sem progresso (`n_pressa`): parado no P4, a renda de
    ~25/s pagava mais que fechar. O Φ fica fora do h para o gradiente do alvo não sumir; o fecho sobe a renda em ≥ 3/s."""
    return w_etapa * etapa * h + w_phi * phi


def n_pressa(h, ganho, dt, tau, recarga) -> torch.Tensor:
    """h·exp(−dt/τ) + recarga·ganho, com teto 1 (13/11, M5). `ganho` = relu(Φ − melhor Φ da etapa)."""
    return torch.clamp(h * math.exp(-dt / tau) + recarga * ganho, max=1.0)


def n_qualidade(d_transporte, incl_graus, incl_lim_graus, contatos, k) -> torch.Tensor:
    """Fator do rastreio no CARREGAR (E4, PD-37): nucleo(‖caixa − alvo_transporte‖; tol_caixa_p4) · rampa(incl) ·
    (mãos em contato)/2. Entra em `aplica_qualidade`, só na parte positiva do rastreio."""
    return (potencial.nucleo(d_transporte, k.tol_caixa_p4)
            * potencial.rampa(incl_graus, incl_lim_graus, k.delta_incl_graus) * contatos.float().sum(-1) / 2.0)


def aplica_qualidade(r, q) -> torch.Tensor:
    """q·relu(r) + min(r, 0): a qualidade escala só o positivo; o rastreio negativo fica como está (PD-37)."""
    return q * torch.relu(r) + torch.clamp(r, max=0.0)


def n_perda_caixa(perdeu_agora, custo) -> torch.Tensor:
    """custo no passo seguinte à detecção da perda (S §10; E4). Positivo: o peso do cfg é −1."""
    return perdeu_agora.float() * custo


def margem_com(com_xy, cantos_xy) -> torch.Tensor:
    """Distância com sinal do CoM às arestas do fecho convexo das solas (positiva dentro; PD-35). (E,).

    com_xy (E,2); cantos_xy (E,K,2). ⚠ Sem `ConvexHull`: uma reta por par de cantos é aresta do fecho quando
    todos os cantos ficam de um lado dela. Pares de cantos iguais (a mesma quina dos dois pés) dão reta nula e
    saem pela `comp`. Sem aresta válida, devolve 1e3 (dentro de tudo).
    """
    n_cantos = cantos_xy.shape[1]
    ii, jj = torch.triu_indices(n_cantos, n_cantos, 1, device=cantos_xy.device)
    a, b = cantos_xy[:, ii], cantos_xy[:, jj]  # (E,P,2)
    t = b - a
    nrm = torch.stack([-t[..., 1], t[..., 0]], dim=-1)  # normal de cada reta
    comp = nrm.norm(dim=-1)
    u = nrm / comp.clamp(min=_EPS).unsqueeze(-1)
    s = ((cantos_xy.unsqueeze(1) - a.unsqueeze(2)) * u.unsqueeze(2)).sum(-1)  # (E,P,n_cantos)
    dentro = (s >= -_EPS).all(-1)
    fora = (s <= _EPS).all(-1)
    valida = (dentro | fora) & (comp > _EPS)
    sinal = dentro.to(s.dtype) * 2.0 - 1.0  # normal para dentro: +u se todos estão do lado de +u
    margem = sinal * ((com_xy.unsqueeze(1) - a) * u).sum(-1)  # (E,P)
    return torch.where(valida, margem, torch.full_like(margem, 1e3)).min(dim=-1).values


def n_sacudida(dv_lin, dw, dt, norma_lin, norma_ang) -> torch.Tensor:
    """(E,) sacudida adimensional (PD-34): Σ(Δv/(dt·norma_lin))² + Σ(Δω/(dt·norma_ang))², sem raiz. Δ = velocidade
    da caixa no mundo menos a da chamada anterior (m/s, rad/s); dt = env.step_dt. Com o p50 medido, vale 2 (P8, P10)."""
    return (dv_lin / (dt * norma_lin)).pow(2).sum(-1) + (dw / (dt * norma_ang)).pow(2).sum(-1)


def forcas_por_face(normal_w, forca_w, achado, R_caixa) -> tuple[torch.Tensor, torch.Tensor]:
    """(F_esq, F_dir) (E,) em N, das 12 fatias do `caixa_faces` (P12). A normal aponta da caixa (primário) para o robô.
    n_c = R_caixaᵀ·normal_w no frame da caixa; a face é o eixo de maior |n_c|, com o sinal de n_c nele. F_n =
    |⟨forca_w, normal_w⟩|. F_esq: fatias com achado na face +y (palma esquerda); F_dir: na −y. As faces ±x (frente e
    fundo, onde o tronco apoia) e ±z não entram (S §8.2). `torch.where`: uma fatia sem achado vale 0, mesmo com NaN."""
    n_c = torch.einsum("eij,esi->esj", R_caixa, normal_w)
    mais_y = n_c.abs().argmax(-1) == 1
    f_n = (forca_w * normal_w).sum(-1).abs()
    zero = torch.zeros_like(f_n)
    esq = torch.where(achado & mais_y & (n_c[..., 1] > 0), f_n, zero).sum(-1)
    dir_ = torch.where(achado & mais_y & (n_c[..., 1] < 0), f_n, zero).sum(-1)
    return esq, dir_


# ---------------------------------------------------------------- invólucros (leem o env)

def _fim(env, tarefas, valor, paga=False):
    """Vale zero fora de `tarefas` (§6.6). `paga`: também zera depois da caixa perdida (S §10)."""
    if paga:
        valor = torch.where(env.v3_perdeu, torch.zeros_like(valor), valor)
    return torch.where(mascara(env, tarefas), valor, torch.zeros_like(valor))


def _incl_graus(quat):
    """Inclinação (graus) do eixo z de um corpo em relação à vertical. quat (E,K,4) em wxyz → (E,K)."""
    x, y = quat[..., 1], quat[..., 2]
    return torch.rad2deg(torch.acos((1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)))


def _rastreio(env, linear):
    """Rastreio normalizado (S §19, conflito 5; PD-49). Linear: c = ‖cmd_xy‖; angular: c = |wz|."""
    cmd = env.command_manager.get_command("twist")  # (E,3): vx, vy, wz (§6.3, comando)
    v = env.command_manager.get_term("twist").v_media_b  # (E,3): média de 0,8 s no frame da base
    if linear:
        c = cmd[:, :2].norm(dim=-1)
        erro2 = (cmd[:, :2] - v[:, :2]).pow(2).sum(-1)
        v2 = v[:, :2].pow(2).sum(-1)
        sigma_zero = R.sigma_zero_lin
    else:
        c = cmd[:, 2].abs()
        erro2 = (cmd[:, 2] - v[:, 2]).pow(2)
        v2 = v[:, 2].pow(2)
        sigma_zero = balanceador.no_degrau(R.sigma_zero_ang_inicio, R.sigma_zero_ang, env.v3_cur.restricao(), 1)
    return potencial.rastreio_normalizado(erro2, c, v2, R.sigma_rastreio, sigma_zero, C.limiar_canal,
                                          R.piso_rastreio)


def _q_alvo(env, q):
    """Alvo das juntas dos desvios: q_de_pé no ANDAR e q_com_caixa(a) no CARREGAR (D5)."""
    ref = env.v3_ref
    carrega = (env.v3_tarefa == TAREFAS.index("CARREGAR")).unsqueeze(-1)
    return torch.where(carrega, ref.q_com_caixa(env.v3_meia).to(q), ref.q_de_pe.to(q))


def _com_combinado(env):
    """CoM de robô + caixa presa (E,3). A caixa entra só quando está na mão (`v3_pegou` e não `v3_perdeu`) e o alvo
    está entre P3 e B3 (13/11, A7: com o índice no B4 a caixa já está na laje; o CARREGAR fica com o índice no P4).

    ⚠ Robô: `subtree_com` e `body_subtreemass` da raiz (a pelve, com toda a subárvore do robô). Os campos do modelo
    têm forma (mundos, corpos): indexa a raiz pela coluna e expande para (E,), porque antes da DR de massa há 1 mundo.
    (29/10: o `root_com_pos_w` era o CoM só da pelve, e o `body_mass[body_ids]` indexava o eixo dos mundos.)
    """
    robo = env.scene["robot"]
    raiz = robo.indexing.root_body_id
    com_robo = env.sim.data.subtree_com[:, raiz]  # (E,3), mundo
    m_robo = env.sim.model.body_subtreemass[:, raiz].expand(com_robo.shape[0])  # (E,)
    na_mao = env.v3_pegou & ~env.v3_perdeu & _em(env.v3_alvo_idx, _P3_B3)
    m_caixa = torch.where(na_mao, env.v3_massa, torch.zeros_like(env.v3_massa))
    m_tot = m_robo + m_caixa
    com = (m_robo.unsqueeze(-1) * com_robo + m_caixa.unsqueeze(-1) * env.scene["box"].data.root_com_pos_w)
    return com / m_tot.unsqueeze(-1)


def _cantos_mundo(pose, cantos):
    """Cantos das solas no mundo (E,2·K,3). pose (E,2,7): pos + quat wxyz dos tornozelos; cantos (K,3)."""
    E = pose.shape[0]
    q = pose[..., 3:7].unsqueeze(-2).expand(-1, -1, cantos.shape[0], -1).reshape(-1, 4)
    v = cantos.to(pose).expand(E, 2, -1, -1).reshape(-1, 3)
    pos = pose[..., :3].unsqueeze(-2)
    return (quat_apply(q, v).reshape(E, 2, -1, 3) + pos).reshape(E, -1, 3)


def por_degrau(env, ini, fim):
    """Valor do degrau da tarefa corrente, `balanceador.no_degrau` (§6.9; P1). Chamam: `incl_lim`, o pouso do `_phi_env`,
    o fator do B2 no `progresso`, `contato_laje`, `sacudida`, `aperto` e `terminacoes.chegada` (esta, de outro módulo)."""
    return balanceador.no_degrau(ini, fim, env.v3_degrau.float(), ATIVO.balanceador.degraus)


def por_restricao(env, ini, so_andar=False):
    """Fator (E,) das restrições que seguem o progresso (11/11; specs/g1-v3-treino-unico.md §2): `ini` com
    `restricao` 0 e 1 com `restricao` 1. `so_andar`: o fator vale só nos envs da tarefa ANDAR; nos outros, 1."""
    f = torch.full((env.num_envs,), balanceador.no_degrau(ini, 1.0, env.v3_cur.restricao(), 1), device=env.device)
    if so_andar:
        f = torch.where(env.v3_tarefa == TAREFAS.index("ANDAR"), f, torch.ones_like(f))
    return f


def incl_lim(env):
    """`incl_lim` dos invariantes: 30° → 15° pelo degrau (§6.9)."""
    return por_degrau(env, K.incl_inicio_graus, K.incl_fim_graus)


def incl_torso(env):
    """Inclinação (graus) do torso do robô, do vertical (E,). Mesma medida da `cerca_tronco`."""
    robo = env.scene["robot"]
    return _incl_graus(robo.data.body_link_pose_w[:, robo.find_bodies(("torso_link",))[0], 3:7])[:, 0]


def _pes_e_com(env):
    """(cantos_xy (E,8,2), com_xy (E,2), dois_pes (E,)) para a margem do CoM (PD-35). O CoM inclui a caixa presa."""
    robo = env.scene["robot"]
    pose = robo.data.body_link_pose_w[:, robo.find_bodies(_PES_CORPOS)[0]]  # (E,2,7)
    cantos = _cantos_mundo(pose, env.v3_ref.cantos_sola)
    dois = (env.scene["pes_chao"].data.found > 0).all(-1)
    return cantos[..., :2], _com_combinado(env)[:, :2], dois


def _contato_palmas(env):
    """(E,2) bool: palma esquerda e direita tocam a caixa (sensores `palma_E` e `palma_D`; PD-19)."""
    return torch.stack([env.scene[n].data.found[:, 0] > 0 for n in _PALMAS_SENSORES], -1)


def alvo_da_caixa_w(env, indice):
    """Posição-alvo da caixa (E,3) no mundo para o alvo `indice` (`alvos.caixa_alvo_w`). O chão é local: z da pelve
    menos `altura_pelve`, como a cadeia (S10, 29/10). Pública: o progresso, a qualidade e a cadeia a usam."""
    robo = env.scene["robot"]
    chao = robo.data.root_link_pos_w[:, 2] - env.scene["altura_pelve"].data.heights[:, 0]
    return alvos.caixa_alvo_w(indice, env.v3_caixa_pega_w, env.v3_alvo_w, robo.data.root_link_pos_w,
                              estimador.yaw(robo.data.root_link_quat_w), chao, env.v3_meia, env.v3_ref, K)


def _topo_do(env, indice):
    """Topo da laje de referência: o de destino nos B1–B3, o da pega nos outros (E,)."""
    return torch.where(_em(indice, _B1_B3), env.v3_topo_destino, env.v3_topo)


def ref_do_alvo(env, indice):
    """(E,8) da referência do alvo `indice` (palmas, pelve, tronco; `observacoes.referencia_do_alvo`). Lazy.
    Pública (F9 S4): a observação `referencia` do crítico chama esta, a mesma fonte da recompensa."""
    from g1_v3.observacoes import referencia_do_alvo

    return referencia_do_alvo(indice, _topo_do(env, indice), env.v3_meia, env.v3_ref, K)


def _recuo_ref(env, indice):
    """Recuo de referência (m) do alvo `indice`: o transporte no P4 e no CARREGAR, `ref.recuo` nos outros (E4; B4 não
    usa). O transporte é `ref.caixa_transporte_b(a)`, a referência da IK.
    É o `ik` da `cerca_recuo` nas quatro tarefas (S7, 29/10): uma conta só."""
    ref, a = env.v3_ref, env.v3_meia
    transporte = ref.caixa_transporte_b(a)[:, :2].norm(dim=-1)
    carrega = env.v3_tarefa == TAREFAS.index("CARREGAR")
    return torch.where((indice == _P4) | carrega, transporte, ref.recuo(_topo_do(env, indice), a).to(transporte))


def _phi_env(env, indice, sigma):
    """Φ (E,) do alvo `indice` no estado corrente, pelos núcleos (E4; −1 → 0). `indice` pode ser o alvo velho no
    passo do avanço (D1), e `sigma` é o σ dele (S3, 29/10): posições e referências saem do próprio índice passado."""
    robo, caixa = env.scene["robot"], env.scene["box"]
    a, ref = env.v3_meia, env.v3_ref
    caixa_w, quat = caixa.data.root_link_pos_w, caixa.data.root_link_quat_w
    xy = caixa_w[:, :2]
    rec = (xy - robo.data.root_link_pos_w[:, :2]).norm(dim=-1)  # recuo medido, como em `cerca_recuo`
    palmas_w = robo.data.site_pos_w[:, robo.find_sites(_PALMAS_SITIOS)[0]]  # (E,2,3)
    R_caixa = alvos.frame_da_caixa(quat, estimador.yaw(robo.data.root_link_quat_w))
    palmas_c = torch.einsum("eij,eki->ekj", R_caixa, palmas_w - caixa_w[:, None])
    ref_do = ref_do_alvo(env, indice)
    dpelve = env.scene["altura_pelve"].data.heights[:, 0] - ref_do[:, 6]
    # 13/11, A9: as palmas no frame da base contra a pose de pé (o B4 as mede); A3: o déficit de apoio do B2, com os
    # mesmos sensores e o mesmo peso de `CadeiaCmd._mede` (`forca_de_apoio`: |Fz| somada)
    palmas_b = quat_apply_inverse(robo.data.root_link_quat_w[:, None].expand(-1, 2, -1),
                                  palmas_w - robo.data.root_link_pos_w[:, None])
    dpalmas = (palmas_b - env.v3_ref.palmas_de_pe_b.to(palmas_b.dtype)).norm(dim=-1).amax(-1)  # ref é float64
    apoio_N = env.scene["apoio_caixa"].data.force[..., 2].abs().sum(-1)
    peso_N = env.v3_massa * float(-env.sim.mj_model.opt.gravity[2])
    deficit = torch.relu(1.0 - apoio_N / (K.apoio_b2_mg * peso_N))
    d = distancia_do_alvo(indice, palmas_c, caixa_w, alvo_da_caixa_w(env, indice), dpelve,
                          incl_torso(env) - torch.rad2deg(ref_do[:, 7]), dpalmas, deficit, a, K)
    # fator de pouso: B1 e B2, com o fundo da caixa sobre a laje de destino (D3 da Parte 0)
    fundo = caixa_w[:, 2] - a - (env.scene.env_origins[:, 2] + env.v3_topo_destino)
    pouso = potencial.fator_pouso(caixa.data.root_link_lin_vel_w[:, 2], fundo, por_degrau(env, *R.sigma_pouso),
                                  R.altura_pouso, R.altura_pouso_inicio)
    fator = torch.where(_em(indice, _POUSO), pouso, torch.ones_like(pouso))
    cantos, com, dois = _pes_e_com(env)
    # desloc do apoio = 0 com `v3_perdeu`: o B4 da perda não tem ponto de apoio, e o portão vale 1 (S8, 29/10)
    desloc_apoio = torch.where(env.v3_perdeu, torch.zeros_like(xy[:, 0]), (xy - env.v3_ponto_apoio[:, :2]).norm(dim=-1))
    # mãos: firmeza da pega, e não a contagem dos contatos (11/11, `knobs.margem_pega`). F_pega = margem · m·g / (2μ)
    faces = env.scene["caixa_faces"].data
    f_pega = R.margem_pega * env.v3_massa * float(-env.sim.mj_model.opt.gravity[2]) / (2.0 * ATIVO.dr.atrito_pad[0])
    maos = torch.tanh(potencial.custo_aperto(
        *forcas_por_face(faces.normal, faces.force, faces.found > 0, R_caixa), f_pega))
    portoes = portoes_do_alvo(
        indice, (xy - env.v3_caixa_pega_w[:, :2]).norm(dim=-1), desloc_apoio,
        torch.rad2deg(alvos.inclinacao_da_caixa(quat)), incl_lim(env), maos,
        torch.relu(incl_torso(env) - torch.rad2deg(ref_do[:, 7])), (rec - _recuo_ref(env, indice)).abs(),
        torch.relu(K.faixa_com - margem_com(com, cantos)), dois, a, K)
    return phi_do_alvo(indice, d, sigma, fator, portoes)


def _qualidade(env):
    """(qualidade, d) no CARREGAR, cada um (E,): a caixa perto do transporte, inclinação e as duas mãos (E4); `d` é a
    distância da caixa ao alvo de transporte (a qualidade larga da M5 usa o mesmo `d`).
    ⚠ G7 S3 (P2): na célula `terreno` o alvo é a pelve + `caixa_transporte_b` no frame da pelve (z incluso). O chão
    sob a pelve pula 0,15 m na quina do degrau; no plano fica o alvo de hoje (chão + 0,90, E §4.4)."""
    caixa, robo = env.scene["box"], env.scene["robot"]
    pelve = robo.data.root_link_pos_w
    tb = env.v3_ref.caixa_transporte_b(env.v3_meia)
    pelve_tb = torch.cat([alvos.base_para_mundo(pelve[:, :2], estimador.yaw(robo.data.root_link_quat_w), tb[:, :2]),
                          (pelve[:, 2] + tb[:, 2]).unsqueeze(-1)], -1)
    terreno = env.v3_celula == env.v3_cur.id_celula("CARREGAR", "terreno")
    alvo = torch.where(terreno[:, None], pelve_tb, alvo_da_caixa_w(env, torch.full_like(env.v3_alvo_idx, _P4)))
    d = (caixa.data.root_link_pos_w - alvo).norm(dim=-1)
    incl = torch.rad2deg(alvos.inclinacao_da_caixa(caixa.data.root_link_quat_w))
    return n_qualidade(d, incl, incl_lim(env), _contato_palmas(env), K), d


def _rastreio_com_qualidade(env, linear):
    """Rastreio; no CARREGAR a qualidade escala só a parte positiva (PD-37)."""
    r = _rastreio(env, linear)
    carrega = env.v3_tarefa == TAREFAS.index("CARREGAR")
    return torch.where(carrega, aplica_qualidade(r, _qualidade(env)[0]), r)


def rastreio_linear(env, tarefas):
    """Invólucro de `potencial.rastreio_normalizado`, linear. Todas as tarefas (§6.6 da D5). CARREGAR: qualidade (E4)."""
    return _fim(env, tarefas, _rastreio_com_qualidade(env, True), paga=True)


def rastreio_angular(env, tarefas):
    """Invólucro do rastreio do giro. σ₀ = `sigma_zero_ang` (PD-49). CARREGAR: qualidade (E4, PD-37)."""
    return _fim(env, tarefas, _rastreio_com_qualidade(env, False), paga=True)


def passo(env, tarefas):
    """Marcha por relógio, na forma do `feet_gait` do unitree_rl_lab (13/11, M4). ANDAR e CARREGAR.

    O relógio mora no `TwistV3._update_metrics` (fase por pé, pés defasados em meio ciclo, período pela velocidade
    comandada). Cada pé paga 1 quando o contato bate com a fase. Sem comando, o relógio fica parado em apoio duplo e o
    termo paga cada pé no chão. ⚠ Portão de velocidade (spec de comportamento §4.1): o termo vale × a fração da
    velocidade comandada que o robô cumpre (média de 0,8 s), então a marcha no lugar não recebe o termo inteiro.
    Sem comando o portão vale 1. A taxa de acerto, sem o portão, vai para `v3_acerto_marcha` (métrica `acerto_relogio`).
    """
    cmd = env.command_manager.get_command("twist")
    v = env.command_manager.get_term("twist").v_media_b
    c_lin, c_ang = cmd[:, :2].norm(dim=-1), cmd[:, 2].abs()
    lin, ang = c_lin >= C.limiar_canal, c_ang >= C.limiar_canal
    contato = env.scene["pes_chao"].data.found > 0  # (E,2)
    acerto = n_passo(contato, env.v3_fase_marcha, lin | ang, C.fase_apoio)
    g_lin = torch.clamp((v[:, :2] * cmd[:, :2]).sum(-1) / c_lin.pow(2).clamp(min=_EPS), 0.0, 1.0)
    g_ang = torch.clamp(v[:, 2] * torch.sign(cmd[:, 2]) / c_ang.clamp(min=_EPS), 0.0, 1.0)
    portao = torch.where(lin, g_lin, torch.where(ang, g_ang, torch.ones_like(g_lin)))
    env.v3_acerto_marcha[:] = acerto.mean(-1)
    return _fim(env, tarefas, acerto.sum(-1) * portao, paga=True)


def vivo(env, tarefas):
    """1 por passo enquanto a caixa não caiu (S §9.2). ANDAR e CARREGAR."""
    return _fim(env, tarefas, n_vivo(env.v3_perdeu))


def altura_pe(env, tarefas):
    """Altura do pé sobre o terreno (PD-21) e a velocidade horizontal dos tornozelos. ANDAR e CARREGAR.
    F9 P4: zero com o comando do twist zero (o ANDAR parado final pagava 1,0/s com os pés parados).
    """
    robo = env.scene["robot"]
    h = env.scene["altura_pes"].data.heights  # (E,2): folga do pé acima do terreno
    v = robo.data.body_link_vel_w[:, robo.find_bodies(_PES_CORPOS)[0], :2]  # (E,2,2): [lin, ang] → lin xy
    cmd = env.command_manager.get_command("twist")
    anda = cmd.norm(dim=-1) > 0
    valor = n_altura_pe(h, v, R.alvo_pe, R.std_pe, R.tanh_pe)
    return _fim(env, tarefas, torch.where(anda, valor, torch.zeros_like(valor)), paga=True)


def escorrego(env, tarefas):
    """Escorregão do pé em contato, pela velocidade horizontal do tornozelo (S §9.2). Todas as tarefas."""
    robo = env.scene["robot"]
    v = robo.data.body_link_vel_w[:, robo.find_bodies(_PES_CORPOS)[0], :2]
    contato = env.scene["pes_chao"].data.found > 0  # (E,2)
    return _fim(env, tarefas, n_escorrego(v, contato))


def vz(env, tarefas):
    """v_z² da base (world). Todas as tarefas."""
    return _fim(env, tarefas, n_vz(env.scene["robot"].data.root_link_lin_vel_w[:, 2]))


def giro_xy(env, tarefas):
    """ω_x² + ω_y² da base, no frame da base. Todas as tarefas."""
    return _fim(env, tarefas, n_giro_xy(env.scene["robot"].data.root_link_ang_vel_b[:, :2])) * por_restricao(env, R.estilo_inicio)


def vel_junta(env, tarefas):
    """Σ q̇² nas 29 juntas. Todas as tarefas."""
    return _fim(env, tarefas, n_vel_junta(env.scene["robot"].data.joint_vel)) * por_restricao(env, R.estilo_inicio)


def acel_junta(env, tarefas):
    """Σ q̈² nas 29 juntas. Todas as tarefas."""
    return _fim(env, tarefas, n_acel_junta(env.scene["robot"].data.joint_acc)) * por_restricao(env, R.estilo_inicio)


def taxa_acao(env, tarefas):
    """Taxa de variação da ação crua da política (mjlab `action_rate_l2`). Todas as tarefas."""
    am = env.action_manager
    return _fim(env, tarefas, n_taxa_acao(am.action, am.prev_action))


def energia(env, tarefas):
    """Energia Σ|τ·q̇| com τ de `qfrc_actuator`, que o mjlab já devolve só com as 29 juntas (sem a base
    livre). Todas as tarefas."""
    robo = env.scene["robot"]
    return _fim(env, tarefas, n_energia(robo.data.qfrc_actuator, robo.data.joint_vel)) * por_restricao(env, R.estilo_inicio)


def limite_junta(env, tarefas):
    """Excesso de junta além de `limite_mole` (95%) do curso do modelo (S §9.2). Todas as tarefas."""
    robo = env.scene["robot"]
    lim = robo.data.joint_pos_limits  # (29,2) do modelo
    return _fim(env, tarefas, n_limite_junta(robo.data.joint_pos, lim[..., 0], lim[..., 1], R.limite_mole))


def desvio_bracos(env, tarefas):
    """Desvio dos 14 braços de `q_de_pe` (ANDAR) ou de `q_com_caixa` (CARREGAR)."""
    q = env.scene["robot"].data.joint_pos
    return _fim(env, tarefas, n_desvio(q, _q_alvo(env, q), _BRACOS)) * por_restricao(env, R.estilo_inicio)


def desvio_cintura(env, tarefas):
    """Desvio das 3 juntas da cintura (mesmo alvo dos braços)."""
    q = env.scene["robot"].data.joint_pos
    return _fim(env, tarefas, n_desvio(q, _q_alvo(env, q), _CINTURA)) * por_restricao(env, R.estilo_inicio)


def desvio_quadril(env, tarefas):
    """Desvio de hip_roll e hip_yaw das duas pernas (mesmo alvo dos braços)."""
    q = env.scene["robot"].data.joint_pos
    return _fim(env, tarefas, n_desvio(q, _q_alvo(env, q), _QUADRIL)) * por_restricao(env, R.estilo_inicio)


def tronco_plano(env, tarefas):
    """Tronco plano: gravidade projetada no plano da base (mjlab `flat_orientation_l2`). ANDAR e CARREGAR."""
    return _fim(env, tarefas, n_tronco_plano(env.scene["robot"].data.projected_gravity_b[:, :2]))


def altura_base(env, tarefas):
    """Altura da pelve sobre o terreno, contra `alvo_base` (S §9.2). ANDAR e CARREGAR."""
    h = env.scene["altura_pelve"].data.heights[:, 0]
    return _fim(env, tarefas, n_altura_base(h, R.alvo_base))


def cerca_sola(env, tarefas):
    """Inclinação da sola de cada pé em contato, além de `lim_sola_graus` (S §9.3). Todas as tarefas."""
    robo = env.scene["robot"]
    incl = _incl_graus(robo.data.body_link_pose_w[:, robo.find_bodies(_PES_CORPOS)[0], 3:7])
    contato = env.scene["pes_chao"].data.found > 0
    return _fim(env, tarefas, n_cerca_sola(incl, contato, R.lim_sola_graus))


def cerca_giro(env, tarefas):
    """Giro do hip_yaw de cada perna para dentro, além de `lim_giro_graus` (S §9.3). Todas as tarefas."""
    q = env.scene["robot"].data.joint_pos
    return _fim(env, tarefas, n_cerca_giro(q[:, _YAW_E], q[:, _YAW_D], R.lim_giro_graus)) * por_restricao(env, 0.0, so_andar=True)


def cerca_tronco(env, tarefas):
    """Inclinação do torso além de `lim_tronco_graus` sobre a referência do tronco (S §9.3).

    ⚠ tronco_IK = 0 no ANDAR e no CARREGAR; no PEGAR e no BOTAR, o tronco do alvo corrente (rad → graus).
    """
    incl = incl_torso(env)
    manip = (env.v3_tarefa == TAREFAS.index("PEGAR")) | (env.v3_tarefa == TAREFAS.index("BOTAR"))
    ik = torch.where(manip, torch.rad2deg(ref_do_alvo(env, env.v3_alvo_idx)[:, 7]), torch.zeros_like(incl))
    return _fim(env, tarefas, n_cerca_tronco(incl, ik, R.lim_tronco_graus)) * por_restricao(env, 0.0, so_andar=True)


def cerca_com(env, tarefas):
    """Margem do CoM combinado às arestas das solas, só com os dois pés em contato (PD-35; S §9.3)."""
    cantos, com, dois = _pes_e_com(env)
    return _fim(env, tarefas, n_cerca_com(com, cantos, dois, R.lim_com)) * por_restricao(env, 0.0, so_andar=True)


def cerca_torque(env, tarefas):
    """Torque de cada junta solta e de cada par acoplado, sobre o limite de `lim_torque` (PD-45; S §9.3)."""
    tau = env.scene["robot"].data.qfrc_actuator  # (E,29): o mjlab já tira a base livre
    tau_max = torch.tensor(TAU_MAX, dtype=tau.dtype, device=tau.device)
    pares = _PARES.to(tau.device)
    return _fim(env, tarefas, n_cerca_torque(tau, tau_max, pares, R.tau_par, R.lim_torque, R.banda_torque)) * por_restricao(env, 0.0, so_andar=True)


def cerca_pelve(env, tarefas):
    """Altura da pelve sobre o terreno contra a referência da tarefa (S §9.3).

    ik: `pelve_de_pe` no ANDAR; `pelve_com_caixa(a)` no CARREGAR; `ref_do_alvo` no PEGAR e no BOTAR com alvo.
    ⚠ F9 S1: com −1 (espera do PEGAR, BOTAR depois do B4) a referência do alvo tem pelve 0; vale a pose de pé.
    """
    h = env.scene["altura_pelve"].data.heights[:, 0]
    ref = env.v3_ref
    carrega = env.v3_tarefa == TAREFAS.index("CARREGAR")
    ik = torch.where(carrega, ref.pelve_com_caixa(env.v3_meia).to(h), torch.full_like(h, ref.pelve_de_pe))
    manip = (env.v3_tarefa == TAREFAS.index("PEGAR")) | (env.v3_tarefa == TAREFAS.index("BOTAR"))
    manip = manip & (env.v3_alvo_idx >= 0)
    ik = torch.where(manip, ref_do_alvo(env, env.v3_alvo_idx)[:, 6].to(h), ik)
    return _fim(env, tarefas, n_cerca_pelve(h, ik, R.lim_pelve)) * por_restricao(env, 0.0, so_andar=True)


def cerca_recuo(env, tarefas):
    """Recuo do centro da caixa além de `lim_recuo` da referência (S §9.3). PEGAR, CARREGAR e BOTAR.

    recuo = ‖xy da caixa − xy da pelve‖ no mundo: a mesma medida da `grade_recuo` (`dados/gera_referencias.py:109`),
    e não o x no frame da base (29/10). ik: `_recuo_ref` (o transporte no CARREGAR e no P4; `ref.recuo` nos outros).
    ⚠ F9 M14: no BOTAR com o alvo B4 ou −1 a caixa já está solta, e a cerca vale zero.
    """
    robo = env.scene["robot"]
    recuo = (env.scene["box"].data.root_link_pos_w[:, :2] - robo.data.root_link_pos_w[:, :2]).norm(dim=-1)
    custo = n_cerca_recuo(recuo, _recuo_ref(env, env.v3_alvo_idx).to(recuo), R.lim_recuo)
    solta = (env.v3_tarefa == TAREFAS.index("BOTAR")) & ((env.v3_alvo_idx == _B4) | (env.v3_alvo_idx < 0))
    return _fim(env, tarefas, torch.where(solta, torch.zeros_like(custo), custo))


def contato_fora(env, tarefas):
    """Contatos do robô com o chão fora dos pés (> 1 N), corpos em autocolisão (S §9.2) e, fora da janela da pega
    (P1 a B3 e o CARREGAR), contatos do robô com a caixa (E4). Todas as tarefas."""
    forca = env.scene["nao_pes_chao"].data.force.norm(dim=-1)  # (E,1)
    colisoes = (env.scene["auto_colisao"].data.found > 0).sum(-1)
    base = n_contato_fora(forca, colisoes, R.contato_fora_N)
    caixa = (env.scene["caixa_robo"].data.force.norm(dim=-1) > R.contato_fora_N).float().sum(-1)  # (E,12) slots
    janela = _em(env.v3_alvo_idx, _JANELA_PEGA) | (env.v3_tarefa == TAREFAS.index("CARREGAR"))
    return _fim(env, tarefas, base + torch.where(janela, torch.zeros_like(caixa), caixa))


def progresso(env, tarefas):
    """Renda por segundo: `renda_etapa`·etapa·h + `renda_phi`·Φ pago (proposta-renda-em-escada.md). A etapa sobe com a
    cadeia. Índices de alvo: P0..P4 = 0..4, B1..B4 = 5..8, −1 = sem alvo.

    | condição (nesta precedência)                              | etapa | Φ pago    |
    | `v3_perdeu`                                               | 0     | 0         |
    | `~v3_p4_fechado` e tarefa ≠ PEGAR (sem a caixa na mão)    | 0     | 0         |
    | `v3_b4_fechado` (ANDAR final)                             | 9     | 0         |
    | `v3_p4_fechado` e alvo < B1 (CARREGAR, fecho do P4)       | 5     | 0         |
    | alvo ≥ 0                                                  | alvo  | Φ do alvo |
    | resto                                                     | 0     | 0         |

    Por quê: a etapa 5 só vale com a pega feita (`v3_p4_fechado`). A tarefa CARREGAR sozinha não prova a pega: o aviso do
    PEGAR é o fecho do P4 e o do BOTAR é o do B4 (M6). Depois da perda a renda é 0, sem Φ: com Φ do B4 pago, fechar o
    B4 (que marca `v3_b4_fechado`) derrubaria a renda para 0, e perder a caixa no P0 renderia mais que ficar. O
    `v3_p4_fechado` só zera no reset do episódio: o `alvo < B1` impede que ele cubra o BOTAR da fatia C. Grava `v3_phi`
    = Φ do alvo corrente (−1 sem alvo), que as métricas `phi_<alvo>` leem.

    13/11, M5: o h (`v3_h`) só multiplica a etapa. P0 a B4 com Φ pago: `n_pressa` (cai com τ, recarrega com o ganho do Φ
    sobre `v3_phi_melhor`). CARREGAR: média móvel (τ) de um sinal em [0, 1], o rastreio dos canais comandados (ou o deslize
    parado, `(1 + r_lin)(1 + r_ang)`) vezes a qualidade larga da caixa. Depois do fecho do P4 (fora do CARREGAR) e do B4,
    h = 1: o robô não muda a duração desses trechos. Os buffers recomeçam em `comando.py` (reset, troca de alvo, abertura de
    trecho, perda). Lê o que os comandos gravaram no passo anterior."""
    idx = env.v3_alvo_idx
    phi_alvo = _phi_env(env, idx, env.v3_sigma)  # 0 onde idx < 0
    p4_fechado, b4_fechado = env.v3_p4_fechado, env.v3_b4_fechado
    sem_caixa = env.v3_perdeu | (~p4_fechado & (env.v3_tarefa != TAREFAS.index("PEGAR")))
    fecho_p4 = p4_fechado & (idx < _B1)
    etapa = torch.where(sem_caixa, 0, torch.where(b4_fechado, 9, torch.where(fecho_p4, 5, idx.clamp(min=0))))
    paga_phi = ~(sem_caixa | b4_fechado | fecho_p4)
    dt, tau = env.step_dt, R.tau_renda
    com_alvo = paga_phi & (idx >= 0)
    carrega = fecho_p4 & ~sem_caixa & (env.v3_tarefa == TAREFAS.index("CARREGAR"))
    h_alvo = n_pressa(env.v3_h, torch.relu(phi_alvo - env.v3_phi_melhor), dt, tau, R.recarga_renda)
    cmd = env.command_manager.get_command("twist")
    com_lin, com_ang = cmd[:, :2].norm(dim=-1) >= C.limiar_canal, cmd[:, 2].abs() >= C.limiar_canal
    r_lin, r_ang = _rastreio(env, True), _rastreio(env, False)
    comandado = (com_lin * r_lin.clamp(0, 1) + com_ang * r_ang.clamp(0, 1)) / (com_lin.float() + com_ang).clamp(min=1)
    sinal = torch.where(com_lin | com_ang, comandado, ((1 + r_lin) * (1 + r_ang)).clamp(0, 1))
    sinal = sinal * potencial.rampa(_qualidade(env)[1], *R.qualidade_larga)
    h_carga = env.v3_h + (dt / tau) * (sinal - env.v3_h)
    env.v3_h[:] = torch.where(com_alvo, h_alvo, torch.where(carrega, h_carga, env.v3_h))
    env.v3_phi_melhor[:] = torch.where(com_alvo, torch.maximum(env.v3_phi_melhor, phi_alvo), env.v3_phi_melhor)
    h_ef = torch.where(com_alvo | carrega, env.v3_h, torch.ones_like(env.v3_h))
    r = n_progresso(torch.where(paga_phi, phi_alvo, torch.zeros_like(phi_alvo)), etapa.float(), R.renda_phi,
                    R.renda_etapa, h_ef)
    env.v3_phi[:] = torch.where(idx >= 0, phi_alvo, torch.full_like(phi_alvo, -1.0))
    return _fim(env, tarefas, r)


def contato_laje(env, tarefas):
    """Contato com a laje: soma da rampa de força dos três sensores da laje (`cena.MESA_POR_GRUPO`), com o joelho
    pelo degrau (§6.9). Cópia de `g1_limpo` `contato_mesa`. PEGAR e BOTAR."""
    joelho = por_degrau(env, R.laje_joelho_N[0], R.laje_joelho_N[1])
    sat = torch.clamp(R.laje_saturacao_N - joelho, min=_EPS)
    soma = 0.0
    for nome, _ in MESA_POR_GRUPO:
        forca = env.scene[nome].data.force.norm(dim=-1).amax(dim=-1)  # (E,)
        soma = soma + torch.clamp((forca - joelho) / sat, 0.0, 1.0)
    return _fim(env, tarefas, soma)


def freio(env, tarefas):
    """Freio: `velocidade_por_regime` do g1_limpo (E4). Regime pelo comando (`twist`), com a dobradiça do excesso
    de velocidade sobre o teto da tabela; o preço cresce por degrau (`freio_fator ** v3_freio`) e pela rampa
    `min(s/portão, 1)`, com o `s` médio das células abertas da tarefa. PEGAR, CARREGAR, BOTAR.

    """
    cmd = env.command_manager.get_command("twist")
    total = cmd[:, :2].norm(dim=-1) + cmd[:, 2].abs()
    parado = total < R.freio_limiar_andar
    andando = ~parado & (total < R.freio_limiar_correr)
    correndo = ~(parado | andando)
    v = env.scene["robot"].data.joint_vel.abs()
    vp, va, vc = (torch.tensor(t, dtype=v.dtype, device=v.device) for t in _VMAX)
    vmax = (parado.float()[:, None] * vp + andando.float()[:, None] * va + correndo.float()[:, None] * vc)
    custo = torch.relu(v / vmax - 1.0).pow(2).sum(-1)
    # Rampa: o preço cheio só vale quando a tarefa passa no portão. Sem ela, o freio do PEGAR (punhos a 3,4 rad/s)
    # custava mais que viver e a política colapsou na it 5140. Tarefa sem célula aberta: rampa 0.
    abertas = [[c.s for c in env.v3_cur.celulas if c.tarefa == t and c.aberta] for t in TAREFAS]
    s_t = torch.tensor([sum(a) / len(a) if a else 0.0 for a in abertas], dtype=v.dtype, device=v.device)
    rampa = (s_t / ATIVO.balanceador.portao).clamp(max=1.0)[env.v3_tarefa]
    return _fim(env, tarefas, custo * R.freio_fator ** env.v3_freio.to(v.dtype) * rampa)


class sacudida:
    """Sacudida da caixa (H2; PD-34): termo-classe do mjlab, `__init__(cfg, env)` e `reset(env_ids)` (P9). Custo, sem
    `paga`: o peso negativo vem do knob. ⚠ As recompensas rodam antes do `sim.forward()`: o cvel está um subpasso atrasado
    em TODA chamada, e o Δ entre duas chamadas é de um step_dt exato. Um buffer escrito pela cadeia (pós-forward) errava o
    Δt em 25 % (P9). Por isso a `v_ant` é da classe, e não buffer do env (§6.4)."""

    def __init__(self, cfg, env):
        self.v_ant = torch.zeros(env.num_envs, 6, device=env.device)  # [lin xyz, ang xyz] da caixa na chamada anterior

    def reset(self, env_ids=None):
        self.v_ant[slice(None) if env_ids is None else env_ids] = 0.0

    def custo(self, env) -> torch.Tensor:
        """(E,) sem fator e sem máscara; atualiza a `v_ant`. A métrica `aceleracao_da_caixa` (H5b) reusa esta conta."""
        v = env.scene["box"].data.root_link_vel_w
        d = v - self.v_ant
        self.v_ant = v.clone()
        return n_sacudida(d[:, :3], d[:, 3:], env.step_dt, *R.sacudida_norma)

    def __call__(self, env, tarefas):
        custo = self.custo(env)  # antes da máscara: a v_ant anda mesmo fora da tarefa
        presa = _em(env.v3_alvo_idx, _CAIXA_ALVO)  # P3, P4, B1, B2 (P11: sem `~apoio`)
        valor = custo * por_degrau(env, 0.0, 1.0)
        return _fim(env, tarefas, torch.where(presa, valor, torch.zeros_like(valor)))


def aperto(env, tarefas):
    """Aperto das palmas na caixa (H2; S §8.2): `potencial.custo_aperto` sobre as forças das faces ±y (P12), com o fator
    d/5 do degrau. Custo, sem `paga`; a mesma máscara da sacudida (P3, P4, B1, B2)."""
    s = env.scene["caixa_faces"].data
    R_caixa = alvos.frame_da_caixa(env.scene["box"].data.root_link_quat_w,
                                   estimador.yaw(env.scene["robot"].data.root_link_quat_w))
    valor = potencial.custo_aperto(*forcas_por_face(s.normal, s.force, s.found > 0, R_caixa), R.F_ref)
    valor = valor * por_degrau(env, 0.0, 1.0)
    presa = _em(env.v3_alvo_idx, _CAIXA_ALVO)
    return _fim(env, tarefas, torch.where(presa, valor, torch.zeros_like(valor)))


def perda_caixa(env, tarefas):
    """Perda da caixa no passo seguinte à detecção (`v3_perdeu_agora` é escrito no `compute`, depois das recompensas): só o custo (E4; S §10). Peso −1 no cfg."""
    return _fim(env, tarefas, n_perda_caixa(env.v3_perdeu_agora, R.perda_caixa))


def terminacao(env, tarefas):
    """1 numa terminação que não é `time_out` (S §9.2). Todas as tarefas."""
    tm = env.termination_manager
    return _fim(env, tarefas, n_terminacao(tm.terminated, tm.time_outs))
