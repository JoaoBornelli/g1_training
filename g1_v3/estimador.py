"""Estimador simulado da v3: odometria (D2b; plano §6.3; PD-18; S §13).

⚠ MÓDULO PURO (§1.5): sem mjlab nem mujoco. Quaternion wxyz; giro só em z. `deriva` = magnitudes correntes.
"""
from __future__ import annotations

import torch


def _yaw(q):  # (E,4) wxyz → (E,) rad
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                       1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _quat_de_yaw(psi):  # (E,) → (E,4) wxyz, só giro em z
    z = torch.zeros_like(psi)
    return torch.stack([torch.cos(psi / 2), z, z, torch.sin(psi / 2)], dim=-1)


def _qmul(a, b):  # produto de quaternions wxyz, (E,4)
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


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
        psi_ant = _yaw(self._quat_v)
        c, s = torch.cos(psi_ant), torch.sin(psi_ant)
        dxy = torch.stack([c * dp[:, 0] + s * dp[:, 1], -s * dp[:, 0] + c * dp[:, 1]], dim=-1)
        dz = dp[:, 2]
        dpsi = torch.atan2(torch.sin(_yaw(quat_w) - psi_ant), torch.cos(_yaw(quat_w) - psi_ant))
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
        psi_est = _yaw(self.quat_w)
        c, s = torch.cos(psi_est), torch.sin(psi_est)
        self.pos_w = self.pos_w + torch.stack([c * dxy[:, 0] - s * dxy[:, 1],
                                               s * dxy[:, 0] + c * dxy[:, 1], dz], dim=-1)
        # roll e pitch saem do IMU (os verdadeiros); só o yaw é o estimado
        self.quat_w = _qmul(_quat_de_yaw(psi_est + dpsi - _yaw(quat_w)), quat_w)
        self._pos_v = pos_w.clone()
        self._quat_v = quat_w.clone()
        self._dxy, self._dpsi, self._dz = dxy, dpsi, dz

    def delta_b(self):
        # (dxy_b (E,2), dψ (E,), dz (E,)) do último passo, com o erro de deriva
        return self._dxy, self._dpsi, self._dz
