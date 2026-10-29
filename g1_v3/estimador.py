"""Estimador simulado da v3: odometria (D2b), caixa (I §2.3), visibilidade da D435i e simetria do cubo (plano §6.3;
PD-11, PD-18; S §13).

⚠ MÓDULO PURO (§1.5): sem mjlab nem mujoco. Quaternion wxyz; giro só em z na odometria. `deriva` = magnitudes correntes.
"""
from __future__ import annotations

import math

import torch


def yaw(q):  # (E,4) wxyz → (E,) rad
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                       1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _quat_de_yaw(psi):  # (E,) → (E,4) wxyz, só giro em z
    z = torch.zeros_like(psi)
    return torch.stack([torch.cos(psi / 2), z, z, torch.sin(psi / 2)], dim=-1)


def _qmul(a, b):  # produto de quaternions wxyz, (E,4) (broadcast)
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def _conj(q):  # conjugado = inverso de um quaternion unitário
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)


def _gira(q, v, inversa=False):  # v (E,M,3) girado por q (E,4); inversa = R(q)ᵀ, o frame do sensor
    u = q[:, None, 1:].expand_as(v)
    if inversa:
        u = -u
    t = 2 * torch.cross(u, v, dim=-1)
    return v + q[:, None, :1] * t + torch.cross(u, t, dim=-1)


def _quat_de_rotvec(v):  # (E,3) rad → (E,4) wxyz
    ang = v.norm(dim=-1, keepdim=True)
    fator = torch.where(ang > 1e-12, torch.sin(ang / 2) / ang.clamp_min(1e-12), torch.full_like(ang, 0.5))
    return torch.cat([torch.cos(ang / 2), v * fator], dim=-1)


def _canonico(q):  # chave do quaternion: sinal com a primeira componente não nula positiva
    s = next((x for x in q if abs(x) > 1e-9), 1.0)
    return tuple(round(x * (1.0 if s > 0 else -1.0), 6) + 0.0 for x in q)


def _grupo_do_cubo():  # as 24 rotações do cubo (grupo octaédrico), por fecho de 90° em z e 120° em (1,1,1)
    def mul(a, b):
        return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])
    gens = [(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), (0.5, 0.5, 0.5, 0.5)]
    ident = (1.0, 0.0, 0.0, 0.0)
    grupo = {_canonico(ident): ident}
    fila = [ident]
    while fila:
        a = fila.pop()
        for g in gens:
            b = mul(a, g)
            if _canonico(b) not in grupo:
                grupo[_canonico(b)] = b
                fila.append(b)
    return torch.tensor(list(grupo.values()), dtype=torch.float32)  # (24,4)


_CUBO = _grupo_do_cubo()


def reduz_simetria_cubo(q_rel):  # (E,4) wxyz → (E,3): rotvec de menor ângulo entre as 24 rotações do cubo (I §2.3)
    g = _CUBO.to(device=q_rel.device, dtype=q_rel.dtype)
    p = _qmul(q_rel[:, None, :], g[None])  # (E,24,4): o cubo é simétrico, então q·g dá a mesma caixa
    p = torch.where(p[..., :1] < 0, -p, p)  # w ≥ 0: ângulo em [0, π]
    v = p[..., 1:]
    norma = v.norm(dim=-1)
    ang = 2 * torch.atan2(norma, p[..., 0])  # (E,24)
    rv = v / norma.clamp_min(1e-12)[..., None] * ang[..., None]
    i = ang.argmin(dim=1)
    return rv[torch.arange(len(q_rel), device=q_rel.device), i]


def _dist_segmentos(p0, p1, q0, q1):  # distância mínima entre p0p1 e q0q1, (...,3) cada
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = (d1 * d1).sum(-1).clamp_min(1e-12)
    e = (d2 * d2).sum(-1).clamp_min(1e-12)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    den = (a * e - b * b).clamp_min(1e-12)
    s_ = (b * f - c * e) / den  # solução sem limites; os dois candidatos com um parâmetro fixado no limite
    t_ = (b * s_ + f) / e
    s1 = s_.clamp(0, 1)
    t1 = ((b * s1 + f) / e).clamp(0, 1)
    t2 = t_.clamp(0, 1)
    s2 = ((b * t2 - c) / a).clamp(0, 1)
    d_a = (p0 + s1[..., None] * d1) - (q0 + t1[..., None] * d2)
    d_b = (p0 + s2[..., None] * d1) - (q0 + t2[..., None] * d2)
    return torch.minimum(d_a.norm(dim=-1), d_b.norm(dim=-1))


def visivel(cam_pos_w, cam_rot_w, pontos_w, capsulas_w, raios, k):  # (E,) bool (PD-11)
    """Centro da caixa no FOV do RGB e na faixa do depth, e ≥ `pontos_visiveis_min` dos 9 pontos sem oclusão.
    cam_pos_w (E,3); cam_rot_w (E,4), eixo óptico = +x do site d435; pontos_w (E,9,3); capsulas_w (E,K,2,3); raios (K,).
    """
    d = _gira(cam_rot_w, pontos_w - cam_pos_w[:, None], inversa=True)  # (E,9,3) no frame da câmera
    meia_h, meia_v = math.radians(k.fov_h_graus) / 2, math.radians(k.fov_v_graus) / 2
    dist = d.norm(dim=-1)
    no_fov = (torch.atan2(d[..., 1], d[..., 0]).abs() <= meia_h) & (torch.atan2(d[..., 2], d[..., 0]).abs() <= meia_v)
    ok = no_fov & (dist >= k.alcance[0]) & (dist <= k.alcance[1])  # (E,9)
    E, M = pontos_w.shape[:2]
    K_ = capsulas_w.shape[1]
    p0 = cam_pos_w[:, None, None].expand(E, M, K_, 3)
    p1 = pontos_w[:, :, None].expand(E, M, K_, 3)
    ocluido = (_dist_segmentos(p0, p1, capsulas_w[:, None, :, 0].expand(E, M, K_, 3),
                               capsulas_w[:, None, :, 1].expand(E, M, K_, 3)) < raios).any(dim=-1)  # (E,9)
    return ok[:, 0] & ((ok & ~ocluido).sum(-1) >= k.pontos_visiveis_min)


class Odometria:
    """Pose estimada (`pos_w`, `quat_w`); a VERDADEIRA fica à parte, para tirar o Δ do real.
    ⚠ O erro de rumo entra na posição absoluta, como no robô real, mas o Δ sai no frame da base
    verdadeira: somado no mundo, o Δ do mapa giraria pelo erro absoluto de rumo a cada passo (PD-18).
    """

    def __init__(self, n, device, k, gerador):
        self.device = device
        self.k = k
        self.gerador = gerador  # None desliga o ruído (teste determinístico)
        self.pos_w = torch.zeros(n, 3, device=device)
        self.quat_w = torch.zeros(n, 4, device=device)
        self.quat_w[:, 0] = 1.0
        self._pos_v = self.pos_w.clone()
        self._quat_v = self.quat_w.clone()
        self._dxy = torch.zeros(n, 2, device=device)
        self._dpsi = torch.zeros(n, device=device)
        self._dz = torch.zeros(n, device=device)

    def reinicia(self, ids, pos_w, quat_w):
        # pose estimada = a verdadeira no reset; Δ do passo = 0
        self.pos_w[ids] = pos_w
        self.quat_w[ids] = quat_w
        self._pos_v[ids] = pos_w
        self._quat_v[ids] = quat_w
        self._dxy[ids] = 0.0
        self._dpsi[ids] = 0.0
        self._dz[ids] = 0.0

    def passo(self, pos_w, quat_w, deriva: dict):
        # incremento VERDADEIRO no frame da base do passo anterior: R(−ψ_ant)·Δp, Δψ e Δz
        dp = pos_w - self._pos_v
        psi_ant = yaw(self._quat_v)
        c, s = torch.cos(psi_ant), torch.sin(psi_ant)
        dxy = torch.stack([c * dp[:, 0] + s * dp[:, 1], -s * dp[:, 0] + c * dp[:, 1]], dim=-1)
        dz = dp[:, 2]
        dpsi = torch.atan2(torch.sin(yaw(quat_w) - psi_ant), torch.cos(yaw(quat_w) - psi_ant))
        if self.gerador is not None:
            # PD-18: variância = deriva²·|Δp| + deriva_giro²·|Δψ|, por eixo (modelo de Thrun, cap. 5)
            z = torch.randn(len(dp), 4, generator=self.gerador, device=self.gerador.device).to(self.device)
            norma, giro = dp.norm(dim=-1), dpsi.abs()
            s_xyz = (deriva["odo"] ** 2 * norma + deriva["pos_giro"] ** 2 * giro).sqrt()
            s_psi = (deriva["rumo"] ** 2 * norma + deriva["rumo_giro"] ** 2 * giro).sqrt()
            dxy = dxy + s_xyz[:, None] * z[:, :2]
            dz = dz + s_xyz * z[:, 2]
            dpsi = dpsi + s_psi * z[:, 3]
        # integra pela pose ESTIMADA: pos += (R(ψ_est)·dxy_b, dz); ψ_est += dψ
        psi_est = yaw(self.quat_w)
        c, s = torch.cos(psi_est), torch.sin(psi_est)
        self.pos_w = self.pos_w + torch.stack([c * dxy[:, 0] - s * dxy[:, 1],
                                               s * dxy[:, 0] + c * dxy[:, 1], dz], dim=-1)
        # roll e pitch saem do IMU (os verdadeiros); só o yaw é o estimado
        self.quat_w = _qmul(_quat_de_yaw(psi_est + dpsi - yaw(quat_w)), quat_w)
        self._pos_v = pos_w.clone()
        self._quat_v = quat_w.clone()
        self._dxy, self._dpsi, self._dz = dxy, dpsi, dz

    def delta_b(self):
        # (dxy_b (E,2), dψ (E,), dz (E,)) do último passo, com o erro de deriva
        return self._dxy, self._dpsi, self._dz


def alvo_no_frame_da_base(alvo_w, odo: Odometria):  # (E,3): frame de rumo da base ESTIMADA (plano §6.3)
    d = alvo_w - odo.pos_w
    c, s = torch.cos(yaw(odo.quat_w)), torch.sin(yaw(odo.quat_w))
    return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], dim=-1)


def do_frame_da_base(p_b, odo: Odometria):  # (E,3): o inverso de `alvo_no_frame_da_base` (plano §6.3, E11 S5a)
    c, s = torch.cos(yaw(odo.quat_w)), torch.sin(yaw(odo.quat_w))
    x = c * p_b[:, 0] - s * p_b[:, 1]
    y = s * p_b[:, 0] + c * p_b[:, 1]
    return torch.stack([odo.pos_w[:, 0] + x, odo.pos_w[:, 1] + y, odo.pos_w[:, 2] + p_b[:, 2]], dim=-1)


class EstimadorCaixa:
    """Leitura da caixa (I §2.3). A amostra cai na grade k/taxa_hz desde o `reinicia` (no mundo, com o
    ruído e a perda), entrega em t + latencia_s, e a saída é a última válida já entregue. A idade é t − instante,
    cortada em idade_max; sem leitura válida, idade_max. A saída é convertida para a base pela odometria atual."""

    def __init__(self, n, device, k, gerador):
        self.n, self.device, self.k, self.gerador = n, device, k, gerador
        self.t = 0.0
        q = k.latencia_s * k.taxa_hz
        m = int(math.ceil(q)) + 3  # amostras em voo (≈ latência × taxa) e folga
        f = dict(device=device)
        self._t0 = torch.full((n,), float("nan"), **f)  # início da grade; NaN = depois do reinicia, sem passo
        self._kp = torch.zeros(n, dtype=torch.long, **f)  # próximo índice da grade
        self._tp = torch.zeros(n, **f)  # instante e estado do passo anterior (interpolação na grade)
        self._pos_p = torch.zeros(n, 3, **f)
        self._quat_p = torch.zeros(n, 4, **f)
        self._quat_p[:, 0] = 1.0
        self._q_t = torch.full((n, m), float("nan"), **f)  # fila: amostras ainda não entregues
        self._q_pos = torch.zeros(n, m, 3, **f)
        self._q_quat = torch.zeros(n, m, 4, **f)
        self._q_meia = torch.zeros(n, m, **f)
        self._r_t = torch.full((n,), float("-inf"), **f)  # registro: última amostra entregue
        self._r_pos = torch.zeros(n, 3, **f)
        self._r_quat = torch.zeros(n, 4, **f)
        self._r_quat[:, 0] = 1.0
        self._r_meia = torch.zeros(n, **f)
        self._z_meia = torch.zeros(n, **f)  # vieses persistentes por env (N(0,1); a escala vem da magnitude)
        self._z_ori = torch.zeros(n, 3, **f)
        self.reinicia(torch.arange(n, device=device))

    def _sorteia(self, shape, normal):  # gerador None: sem ruído (normal → 0) e sem perda (uniforme → 1)
        if self.gerador is None:
            return torch.zeros(shape, device=self.device) if normal else torch.ones(shape, device=self.device)
        g = self.gerador
        x = (torch.randn(shape, generator=g, device=g.device) if normal
             else torch.rand(shape, generator=g, device=g.device))
        return x.to(self.device)

    def reinicia(self, ids):
        self._t0[ids] = float("nan")
        self._kp[ids] = 0
        self._q_t[ids] = float("nan")
        self._r_t[ids] = float("-inf")
        self._r_pos[ids] = 0.0
        self._r_quat[ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._r_meia[ids] = 0.0
        self._z_meia[ids] = self._sorteia((len(ids),), True)
        self._z_ori[ids] = self._sorteia((len(ids), 3), True)

    def passo(self, t, caixa_pos_w, caixa_quat_w, meia, visivel, magnitude: dict, dist=None):
        """`magnitude`: escalares {sigma_pos, sigma_pos_rel, sigma_ori_graus, perda, vies_meia, ori_persistente};
        chave ausente vale 0. σ da posição = sigma_pos + sigma_pos_rel·dist, por eixo (dist (E,) ou escalar)."""
        self.t = t
        nova = torch.isnan(self._t0)  # primeiro passo depois do reinicia: início da grade e estado anterior = o atual
        self._t0[nova] = t
        self._tp[nova] = t
        self._pos_p[nova] = caixa_pos_w[nova]
        self._quat_p[nova] = caixa_quat_w[nova]
        rel = magnitude.get("sigma_pos_rel", 0.0)
        if rel and dist is None:
            raise ValueError("sigma_pos_rel exige `dist` (distância da câmera à caixa)")
        sp = torch.as_tensor(magnitude.get("sigma_pos", 0.0) + rel * (0.0 if dist is None else dist),
                             dtype=torch.float32, device=self.device).expand(self.n)[:, None]
        so = math.radians(magnitude.get("sigma_ori_graus", 0.0))
        z_ori = self._z_ori if magnitude.get("ori_persistente", 0.0) > 0 else self._sorteia((self.n, 3), True)
        erro_ori = _quat_de_rotvec(z_ori * so)
        meia_s = meia + self._z_meia * magnitude.get("vies_meia", 0.0)
        perda = magnitude.get("perda", 0.0)
        ar = torch.arange(self.n, device=self.device)
        # amostras da grade com instante ≤ t: interpolação linear entre o passo anterior e o atual
        while True:
            g = self._t0 + self._kp / self.k.taxa_hz
            due = g <= t + 1e-9
            if not bool(due.any()):
                break
            a = ((g - self._tp) / (t - self._tp).clamp_min(1e-9)).clamp(0, 1)[:, None]
            pos = self._pos_p + a * (caixa_pos_w - self._pos_p) + sp * self._sorteia((self.n, 3), True)
            q1 = caixa_quat_w * torch.where((self._quat_p * caixa_quat_w).sum(-1, keepdim=True) < 0, -1.0, 1.0)
            qi = self._quat_p * (1 - a) + q1 * a
            qs = _qmul(erro_ori, qi / qi.norm(dim=-1, keepdim=True))
            valido = visivel & (self._sorteia((self.n,), False) >= perda)
            ins = due & valido
            slot = torch.isnan(self._q_t).float().argmax(dim=1)  # primeiro slot livre
            self._q_t[ar, slot] = torch.where(ins, g, self._q_t[ar, slot])
            self._q_pos[ar, slot] = torch.where(ins[:, None], pos, self._q_pos[ar, slot])
            self._q_quat[ar, slot] = torch.where(ins[:, None], qs, self._q_quat[ar, slot])
            self._q_meia[ar, slot] = torch.where(ins, meia_s, self._q_meia[ar, slot])
            self._kp = self._kp + due.long()
        self._tp = torch.full_like(self._tp, t)
        self._pos_p = caixa_pos_w.clone()
        self._quat_p = caixa_quat_w.clone()
        # entrega: amostras com instante ≤ t − latência; o registro guarda a mais recente
        entregue = self._q_t <= t - self.k.latencia_s + 1e-9
        if bool(entregue.any()):
            best = self._q_t.masked_fill(~entregue, float("-inf")).argmax(dim=1)
            tem = entregue.any(dim=1)
            self._r_t = torch.where(tem, self._q_t[ar, best], self._r_t)
            self._r_pos = torch.where(tem[:, None], self._q_pos[ar, best], self._r_pos)
            self._r_quat = torch.where(tem[:, None], self._q_quat[ar, best], self._r_quat)
            self._r_meia = torch.where(tem, self._q_meia[ar, best], self._r_meia)
            self._q_t = self._q_t.masked_fill(entregue, float("nan"))

    def saida_b(self, odo: Odometria, base_quat_w):  # (E,8): pos_b 3, rotvec 3, meia 1, idade 1
        seen = torch.isfinite(self._r_t)
        idade = (self.t - self._r_t).clamp(max=self.k.idade_max)  # −∞ (nunca vista) → idade_max
        d = self._r_pos - odo.pos_w
        c, s = torch.cos(yaw(odo.quat_w)), torch.sin(yaw(odo.quat_w))
        pos_b = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], dim=-1)
        rv = reduz_simetria_cubo(_qmul(_conj(base_quat_w), self._r_quat))
        out = torch.cat([pos_b, rv, self._r_meia[:, None], idade[:, None]], dim=-1)
        out[~seen, :7] = 0.0
        return out
