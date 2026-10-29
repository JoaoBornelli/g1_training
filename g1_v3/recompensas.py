"""Termos de recompensa da locomoção da v3, fase D (plano §6.5, D5; PD-16, PD-35, PD-45, PD-48, PD-49).

Cada termo é um núcleo puro sobre tensores (`n_*`, ou o nome do §6.3 quando ele não tem `n_`) mais um
invólucro fino que lê o env. O teste chama o núcleo com tensores sintéticos. Nesta fase: os rastreios, o
passo, o vivo, a altura do pé, o escorregão, as taxas e a energia, os limites e os desvios, o tronco, a
altura da base, as seis cercas (sola, giro, tronco, CoM, torque, pelve, recuo) e o contato fora dos pés.
O contato com a laje, o freio, o progresso, a sacudida, o aperto e a perda da caixa são das fases E e H.

⚠ Todo termo recebe `tarefas` e vale zero fora delas com `torch.where` (§6.6). Nunca multiplicar pela
máscara: NaN × 0 = NaN, e o NaN de um env fora da tarefa chega à recompensa e ao ator.
⚠ Os termos que pagam positivo (`vivo`, `altura_pe`, `passo`, rastreios) zeram depois da caixa perdida
(`v3_perdeu`, S §10). Os pesos moram no `knobs` (no `RewardTermCfg`): o termo devolve a grandeza, sem sinal.
Unidades: ângulos em graus nas cercas (a cerca de giro recebe rad, §6.3), comprimentos em metros, torque
em fração de τ_max ou de `tau_par` (§6.3).
"""
from __future__ import annotations

import re

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_yaw

from g1_v3 import potencial
from g1_v3.curriculo import mascara
from g1_v3.knobs import ATIVO, JUNTAS, TAREFAS

R = ATIVO.recompensa
C = ATIVO.comando
_EPS = 1e-6  # tolerância de "sobre a reta" no fecho convexo do CoM (metros)

_PES_CORPOS = ("left_ankle_roll_link", "right_ankle_roll_link")  # PD-21: altura e velocidade do pé
_PES_SITIOS = ("left_foot", "right_foot")  # PD-16: ordem = `pes_de_pe_b` (0 esquerdo, 1 direito)
_BRACOS = tuple(i for i, j in enumerate(JUNTAS) if re.search(r"shoulder|elbow|wrist", j))  # 14
_CINTURA = tuple(i for i, j in enumerate(JUNTAS) if "waist" in j)  # 3
_QUADRIL = tuple(i for i, j in enumerate(JUNTAS) if re.search(r"hip_(roll|yaw)", j))  # 4
_YAW_E, _YAW_D = JUNTAS.index("left_hip_yaw_joint"), JUNTAS.index("right_hip_yaw_joint")
_TAU_MAX = tuple(next(v for rx, v in R.tau_max if re.fullmatch(rx, j)) for j in JUNTAS)  # S §9.3
_PARES = torch.tensor([[JUNTAS.index(a), JUNTAS.index(b)] for a, b in R.pares_torque],
                      dtype=torch.long)  # PD-45: (waist_pitch, waist_roll) e os dois pares do tornozelo


# ---------------------------------------------------------------- núcleos (tensores, sem env)

def comprimento_do_passo(p_toque_w, p_apoio_w, c_w, p0_toque_b, p0_apoio_b, c_b) -> torch.Tensor:
    """L = (p_toque − p_apoio)·ĉ_w − (p0_toque − p0_apoio)·ĉ_b (PD-16). Posições (E,3) ou (3,); ĉ unitário.

    ⚠ A segunda parcela tira a largura da pose de pé. Sem ela, o passo de lado recebe a largura da base
    (0,23 m) de graça, e o termo paga um passo que o robô não deu.
    """
    return ((p_toque_w - p_apoio_w) * c_w).sum(-1) - ((p0_toque_b - p0_apoio_b) * c_b).sum(-1)


def n_passo(L, L_alvo, r_lin, c_xy, limiar) -> torch.Tensor:
    """min(relu(L)/L_alvo, 1)² · max(r_lin, 0), zero com ‖cmd_xy‖ < limiar (PD-48; S §9.2).

    ⚠ Forma do quadrado (PD-48): linear por toque paga v/L_alvo por segundo, o mesmo para todo L abaixo do
    alvo (derivada zero em L). Com o quadrado, a derivada em L é positiva, e o gradiente aponta para o passo
    do alvo. ⚠ L_alvo vai com `max(·, 1e-6)` só no denominador: com comando zero o clamp de 1 cobre o resto.
    """
    r = torch.clamp(torch.relu(L) / torch.clamp(L_alvo, min=1e-6), max=1.0).pow(2) * torch.clamp(r_lin, min=0.0)
    return torch.where(c_xy < limiar, torch.zeros_like(r), r)


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
    """relu(lim − margem)·dois_pes (PD-35). Margem = distância com sinal do CoM às arestas do fecho convexo.

    com_xy (E,2): CoM combinado (robô + caixa presa); cantos_xy (E,K,2): os K cantos das solas no mundo (K = 8);
    dois_pes (E,) bool. A margem é positiva dentro. ⚠ Sem `ConvexHull`: uma reta por par de cantos é aresta
    do fecho quando todos os cantos ficam de um lado dela. Pares de cantos iguais (a mesma quina dos dois
    pés) dão reta nula e saem pela `comp`.
    """
    K = cantos_xy.shape[1]
    ii, jj = torch.triu_indices(K, K, 1, device=cantos_xy.device)
    a, b = cantos_xy[:, ii], cantos_xy[:, jj]  # (E,P,2)
    t = b - a
    nrm = torch.stack([-t[..., 1], t[..., 0]], dim=-1)  # normal de cada reta
    comp = nrm.norm(dim=-1)
    u = nrm / comp.clamp(min=_EPS).unsqueeze(-1)
    s = ((cantos_xy.unsqueeze(1) - a.unsqueeze(2)) * u.unsqueeze(2)).sum(-1)  # (E,P,K)
    dentro = (s >= -_EPS).all(-1)
    fora = (s <= _EPS).all(-1)
    valida = (dentro | fora) & (comp > _EPS)
    sinal = dentro.to(s.dtype) * 2.0 - 1.0  # normal para dentro: +u se todos estão do lado de +u
    margem = sinal * ((com_xy.unsqueeze(1) - a) * u).sum(-1)  # (E,P)
    margem = torch.where(valida, margem, torch.full_like(margem, 1e3)).min(dim=-1).values
    return torch.where(dois_pes, torch.relu(lim - margem), torch.zeros_like(margem))


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


def _ref_alvo(env):
    """(E,8) da referência do alvo corrente: palmas 6, pelve 6, tronco 7 (§6.3). Lazy: `observacoes` é de outra lane."""
    from g1_v3.observacoes import referencia_do_alvo

    return referencia_do_alvo(env.v3_alvo_idx, env.v3_topo, env.v3_meia, env.v3_ref, ATIVO.alvos)


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
        sigma_zero = R.sigma_zero_ang
    return potencial.rastreio_normalizado(erro2, c, v2, R.sigma_rastreio, sigma_zero, C.limiar_canal,
                                          R.piso_rastreio)


def _q_alvo(env, q):
    """Alvo das juntas dos desvios: q_de_pé no ANDAR e q_com_caixa(a) no CARREGAR (D5)."""
    ref = env.v3_ref
    carrega = (env.v3_tarefa == TAREFAS.index("CARREGAR")).unsqueeze(-1)
    return torch.where(carrega, ref.q_com_caixa(env.v3_meia).to(q), ref.q_de_pe.to(q))


def _com_combinado(env):
    """CoM de robô + caixa presa (E,3). A caixa entra só quando está na mão (`v3_pegou` e não `v3_perdeu`).

    ⚠ Robô: `subtree_com` e `body_subtreemass` da raiz (a pelve, com toda a subárvore do robô). Os campos do modelo
    têm forma (mundos, corpos): indexa a raiz pela coluna e expande para (E,), porque antes da DR de massa há 1 mundo.
    (29/10: o `root_com_pos_w` era o CoM só da pelve, e o `body_mass[body_ids]` indexava o eixo dos mundos.)
    """
    robo = env.scene["robot"]
    raiz = robo.indexing.root_body_id
    com_robo = env.sim.data.subtree_com[:, raiz]  # (E,3), mundo
    m_robo = env.sim.model.body_subtreemass[:, raiz].expand(com_robo.shape[0])  # (E,)
    m_caixa = torch.where(env.v3_pegou & ~env.v3_perdeu, env.v3_massa, torch.zeros_like(env.v3_massa))
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


def rastreio_linear(env, tarefas):
    """Invólucro de `potencial.rastreio_normalizado`, linear. Todas as tarefas (§6.6 da D5)."""
    return _fim(env, tarefas, _rastreio(env, True), paga=True)


def rastreio_angular(env, tarefas):
    """Invólucro do rastreio do giro. σ₀ = `sigma_zero_ang` (PD-49)."""
    return _fim(env, tarefas, _rastreio(env, False), paga=True)


def passo(env, tarefas):
    """Comprimento do passo no toque, na forma de PD-48. Só ANDAR e CARREGAR; zero sem toque neste passo.

    ⚠ "Toque" é o `first_contact` do `pes_chao` (mjlab, `compute_first_contact`), e não o contato deste
    passo: o pé que já estava no chão não paga outra vez. Com os dois pés no mesmo passo, vale o esquerdo.
    """
    robo = env.scene["robot"]
    cmd = env.command_manager.get_command("twist")
    c_xy = cmd[:, :2]
    n_cmd = c_xy.norm(dim=-1)
    c_b = torch.cat([c_xy / n_cmd.clamp(min=_EPS).unsqueeze(-1), torch.zeros_like(n_cmd).unsqueeze(-1)], dim=-1)
    c_w = quat_apply_yaw(robo.data.root_link_quat_w, c_b)  # o mesmo comando, girado pelo rumo da base
    pos = robo.data.site_pos_w[:, robo.find_sites(_PES_SITIOS)[0]]  # (E,2,3) sites dos pés, mundo
    p0 = env.v3_ref.pes_de_pe_b.to(pos)  # (2,3) pose de pé, frame da pelve (PD-16)
    primeiro = env.scene["pes_chao"].compute_first_contact(env.step_dt)  # (E,2): pousou agora
    L_esq = comprimento_do_passo(pos[:, 0], pos[:, 1], c_w, p0[0], p0[1], c_b)
    L_dir = comprimento_do_passo(pos[:, 1], pos[:, 0], c_w, p0[1], p0[0], c_b)
    L = torch.where(primeiro[:, 0], L_esq, L_dir)
    r = n_passo(L, potencial.L_alvo(n_cmd, R.L_min, R.fator_T), _rastreio(env, True), n_cmd, C.limiar_canal)
    r = torch.where(primeiro.any(-1), r, torch.zeros_like(r))
    return _fim(env, tarefas, r, paga=True)


def vivo(env, tarefas):
    """1 por passo enquanto a caixa não caiu (S §9.2). ANDAR e CARREGAR."""
    return _fim(env, tarefas, n_vivo(env.v3_perdeu))


def altura_pe(env, tarefas):
    """Altura do pé sobre o terreno (PD-21) e a velocidade horizontal dos tornozelos. ANDAR e CARREGAR."""
    robo = env.scene["robot"]
    h = env.scene["altura_pes"].data.heights  # (E,2): folga do pé acima do terreno
    v = robo.data.body_link_vel_w[:, robo.find_bodies(_PES_CORPOS)[0], :2]  # (E,2,2): [lin, ang] → lin xy
    return _fim(env, tarefas, n_altura_pe(h, v, R.alvo_pe, R.std_pe, R.tanh_pe), paga=True)


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
    return _fim(env, tarefas, n_giro_xy(env.scene["robot"].data.root_link_ang_vel_b[:, :2]))


def vel_junta(env, tarefas):
    """Σ q̇² nas 29 juntas. Todas as tarefas."""
    return _fim(env, tarefas, n_vel_junta(env.scene["robot"].data.joint_vel))


def acel_junta(env, tarefas):
    """Σ q̈² nas 29 juntas. Todas as tarefas."""
    return _fim(env, tarefas, n_acel_junta(env.scene["robot"].data.joint_acc))


def taxa_acao(env, tarefas):
    """Taxa de variação da ação crua da política (mjlab `action_rate_l2`). Todas as tarefas."""
    am = env.action_manager
    return _fim(env, tarefas, n_taxa_acao(am.action, am.prev_action))


def energia(env, tarefas):
    """Energia Σ|τ·q̇| com τ de `qfrc_actuator`, que o mjlab já devolve só com as 29 juntas (sem a base
    livre). Todas as tarefas."""
    robo = env.scene["robot"]
    return _fim(env, tarefas, n_energia(robo.data.qfrc_actuator, robo.data.joint_vel))


def limite_junta(env, tarefas):
    """Excesso de junta além de `limite_mole` (95%) do curso do modelo (S §9.2). Todas as tarefas."""
    robo = env.scene["robot"]
    lim = robo.data.joint_pos_limits  # (29,2) do modelo
    return _fim(env, tarefas, n_limite_junta(robo.data.joint_pos, lim[..., 0], lim[..., 1], R.limite_mole))


def desvio_bracos(env, tarefas):
    """Desvio dos 14 braços de `q_de_pe` (ANDAR) ou de `q_com_caixa` (CARREGAR)."""
    q = env.scene["robot"].data.joint_pos
    return _fim(env, tarefas, n_desvio(q, _q_alvo(env, q), _BRACOS))


def desvio_cintura(env, tarefas):
    """Desvio das 3 juntas da cintura (mesmo alvo dos braços)."""
    q = env.scene["robot"].data.joint_pos
    return _fim(env, tarefas, n_desvio(q, _q_alvo(env, q), _CINTURA))


def desvio_quadril(env, tarefas):
    """Desvio de hip_roll e hip_yaw das duas pernas (mesmo alvo dos braços)."""
    q = env.scene["robot"].data.joint_pos
    return _fim(env, tarefas, n_desvio(q, _q_alvo(env, q), _QUADRIL))


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
    return _fim(env, tarefas, n_cerca_giro(q[:, _YAW_E], q[:, _YAW_D], R.lim_giro_graus))


def cerca_tronco(env, tarefas):
    """Inclinação do torso além de `lim_tronco_graus` sobre a referência do tronco (S §9.3).

    ⚠ tronco_IK = 0 no ANDAR e no CARREGAR; no PEGAR e no BOTAR, o tronco do alvo corrente (rad → graus).
    """
    robo = env.scene["robot"]
    incl = _incl_graus(robo.data.body_link_pose_w[:, robo.find_bodies(("torso_link",))[0], 3:7])[:, 0]
    manip = (env.v3_tarefa == TAREFAS.index("PEGAR")) | (env.v3_tarefa == TAREFAS.index("BOTAR"))
    ik = torch.where(manip, torch.rad2deg(_ref_alvo(env)[:, 7]), torch.zeros_like(incl))
    return _fim(env, tarefas, n_cerca_tronco(incl, ik, R.lim_tronco_graus))


def cerca_com(env, tarefas):
    """Margem do CoM combinado às arestas das solas, só com os dois pés em contato (PD-35; S §9.3)."""
    robo = env.scene["robot"]
    pose = robo.data.body_link_pose_w[:, robo.find_bodies(_PES_CORPOS)[0]]  # (E,2,7)
    cantos = _cantos_mundo(pose, env.v3_ref.cantos_sola)  # (E,8,3)
    dois = (env.scene["pes_chao"].data.found > 0).all(-1)
    com = _com_combinado(env)
    return _fim(env, tarefas, n_cerca_com(com[:, :2], cantos[..., :2], dois, R.lim_com))


def cerca_torque(env, tarefas):
    """Torque de cada junta solta e de cada par acoplado, sobre o limite de `lim_torque` (PD-45; S §9.3)."""
    tau = env.scene["robot"].data.qfrc_actuator  # (E,29): o mjlab já tira a base livre
    tau_max = torch.tensor(_TAU_MAX, dtype=tau.dtype, device=tau.device)
    pares = _PARES.to(tau.device)
    return _fim(env, tarefas, n_cerca_torque(tau, tau_max, pares, R.tau_par, R.lim_torque, R.banda_torque))


def cerca_pelve(env, tarefas):
    """Altura da pelve sobre o terreno contra a referência da tarefa (S §9.3).

    ik: `pelve_de_pe` no ANDAR; `pelve_com_caixa(a)` no CARREGAR; `referencia_do_alvo` no PEGAR e no BOTAR.
    """
    h = env.scene["altura_pelve"].data.heights[:, 0]
    ref = env.v3_ref
    carrega = env.v3_tarefa == TAREFAS.index("CARREGAR")
    ik = torch.where(carrega, ref.pelve_com_caixa(env.v3_meia).to(h), torch.full_like(h, ref.pelve_de_pe))
    manip = (env.v3_tarefa == TAREFAS.index("PEGAR")) | (env.v3_tarefa == TAREFAS.index("BOTAR"))
    ik = torch.where(manip, _ref_alvo(env)[:, 6].to(h), ik)
    return _fim(env, tarefas, n_cerca_pelve(h, ik, R.lim_pelve))


def cerca_recuo(env, tarefas):
    """Recuo do centro da caixa além de `lim_recuo` da referência (S §9.3). PEGAR, CARREGAR e BOTAR.

    recuo = ‖xy da caixa − xy da pelve‖ no mundo: a mesma medida da `grade_recuo` (`dados/gera_referencias.py:109`),
    e não o x no frame da base (29/10). ik: `R.recuo_carregar` no CARREGAR; no PEGAR e no BOTAR, `ref.recuo(topo, meia)`.
    """
    robo = env.scene["robot"]
    recuo = (env.scene["box"].data.root_link_pos_w[:, :2] - robo.data.root_link_pos_w[:, :2]).norm(dim=-1)
    alvo = env.v3_ref.recuo(env.v3_topo, env.v3_meia).to(recuo)
    carrega = env.v3_tarefa == TAREFAS.index("CARREGAR")
    ik = torch.where(carrega, torch.full_like(recuo, R.recuo_carregar), alvo)
    return _fim(env, tarefas, n_cerca_recuo(recuo, ik, R.lim_recuo))


def contato_fora(env, tarefas):
    """Contatos do robô com o chão fora dos pés (> 1 N) e corpos em autocolisão (S §9.2). Todas as tarefas."""
    forca = env.scene["nao_pes_chao"].data.force.norm(dim=-1)  # (E,1)
    colisoes = (env.scene["auto_colisao"].data.found > 0).sum(-1)
    return _fim(env, tarefas, n_contato_fora(forca, colisoes, R.contato_fora_N))


def terminacao(env, tarefas):
    """1 numa terminação que não é `time_out` (S §9.2). Todas as tarefas."""
    tm = env.termination_manager
    return _fim(env, tarefas, n_terminacao(tm.terminated, tm.time_outs))
