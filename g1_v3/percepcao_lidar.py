"""Filtro de elevação do lidar, puro (I §2.4; PERC §0, §2, §5, §7; PD-5 a PD-10, PD-42).

Só math, numpy, torch e stdlib (plano §1.5). Nada de mjlab, mujoco nem outro módulo do g1_v3: o
sim do real e o robô carregam este arquivo por caminho (PD-43). Os números vêm do `Percepcao`,
recebido como argumento.

A cada passo de 20 ms: `prediz` transporta o mapa pela odometria, `atualiza` põe os pontos novos
e `recorte` devolve as 441 alturas e as 441 confianças que entram na política.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class PadraoMid360:
    """Direções do MID-360 no treino: 200 trechos de 20 ms com 205 raios (PD-5, PERC §5).

    O trecho se repete a cada 200 passos, como a varredura do sensor real.
    """

    def __init__(self, caminho_npz: str, device: str):
        with np.load(caminho_npz) as z:
            self.dirs = torch.as_tensor(z["dirs"], dtype=torch.float32, device=device)  # (200,205,3)

    def direcoes(self, passo: int) -> torch.Tensor:
        return self.dirs[passo % self.dirs.shape[0]]  # (205,3) no frame do sensor


def _giro(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues em lote: vetor de giro (E,3), em rad, → matriz (E,3,3). O clamp evita 0/0 em v = 0."""
    th = torch.linalg.norm(v, dim=-1).clamp_min(1e-12)
    k = v / th[:, None]
    z = torch.zeros_like(k[:, 0])
    kx = torch.stack([z, -k[:, 2], k[:, 1], k[:, 2], z, -k[:, 0], -k[:, 1], k[:, 0], z], -1)
    kx = kx.reshape(-1, 3, 3)
    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    return eye + torch.sin(th)[:, None, None] * kx + (1 - torch.cos(th))[:, None, None] * (kx @ kx)


def pontos_no_frame_do_mapa(dist, dirs_s, sensor_pos_w, sensor_rot_w, base_pos_w, rumo,
                            z_pe_min_w, incl_imu, k, gerador):
    """Uma leitura → (pontos no frame do mapa (E,R,3), pontos_w (E,R,3), validos (E,R)).

    `gerador` é um torch.Generator, ou None: com None não se sorteia nada (teste determinístico).
    Com gerador, sorteia o ruído de alcance e de ângulo, a perda e um giro de inclinação por leitura
    (PERC §2). O alcance vale sempre. `incl_imu` (E,3,3) é o giro fixo do IMU do episódio (PD-42),
    o mesmo da observação da gravidade. Os giros agem no raio inteiro, em volta do sensor.
    """
    E, R = dist.shape
    dev = dist.device
    d = dist
    dirs = dirs_s[None].expand(E, R, 3)
    giro = incl_imu
    validos = torch.ones(E, R, dtype=torch.bool, device=dev)
    if gerador is not None:
        d = d + torch.randn(E, R, generator=gerador, device=dev) * k.ruido_alcance
        dirs = dirs + torch.randn(E, R, 3, generator=gerador, device=dev) * math.radians(k.ruido_angulo_graus)
        dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
        validos = torch.rand(E, R, generator=gerador, device=dev) >= k.perda
        sorteio = torch.randn(E, 3, generator=gerador, device=dev) * math.radians(k.ruido_incl_graus)
        giro = _giro(sorteio) @ incl_imu
    validos = validos & (d >= k.alcance[0]) & (d <= k.alcance[1])  # sem acerto: 100 m, fora
    raios = torch.einsum("eij,erj->eri", sensor_rot_w, dirs) * d[..., None]  # sensor → mundo
    pontos_w = sensor_pos_w[:, None] + torch.einsum("eij,erj->eri", giro, raios)
    rel = pontos_w - base_pos_w[:, None]
    c, s = torch.cos(rumo)[:, None], torch.sin(rumo)[:, None]  # frame do rumo (I §2.4 passo 2)
    x = c * rel[..., 0] + s * rel[..., 1]
    y = -s * rel[..., 0] + c * rel[..., 1]
    z = pontos_w[..., 2] - z_pe_min_w[:, None]  # zero no pé mais baixo (I §2.4 passo 3)
    return torch.stack([x, y, z], -1), pontos_w, validos


def filtra_corpo(pontos, validos, capsulas, raios, sensor_pos, raio_min, margem):
    """Tira os pontos do próprio corpo (PD-10). Pontos, cápsulas e sensor no MESMO frame.

    Sai o ponto a menos de raio + margem de uma cápsula, e o ponto a menos de `raio_min` do sensor.
    ⚠ O tensor (E,R,K,3) pesa E·R·K·3 floats: com 4 096 envs, roda em lotes se o custo aparecer.
    """
    p0, p1 = capsulas[:, :, 0], capsulas[:, :, 1]  # (E,K,3)
    ab = p1 - p0
    ap = pontos[:, :, None] - p0[:, None]  # (E,R,K,3)
    t = (ap * ab[:, None]).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)[:, None]
    perto = p0[:, None] + t.clamp(0.0, 1.0)[..., None] * ab[:, None]  # ponto mais perto do segmento
    dentro_cap = (torch.linalg.norm(pontos[:, :, None] - perto, dim=-1) < raios + margem).any(-1)
    dentro_sensor = torch.linalg.norm(pontos - sensor_pos[:, None], dim=-1) < raio_min
    return validos & ~dentro_cap & ~dentro_sensor


class FiltroDeElevacao:
    """Mapa interno de 51 × 51 células de 10 cm, centrado na base (PD-8), com Kalman por célula.

    `vista` (bool) é a máscara de célula já medida. A variância de uma célula não vista é "∞" no
    plano, e aqui é só `vista = False`: ⚠ inf × 0 dá NaN, por isso nada usa inf.
    """

    def __init__(self, n, device, k):
        self.k = k
        self.m = k.n_interno
        self.c0 = self.m // 2  # índice da célula da base
        self.s2_ref = k.sigma_ref ** 2
        self.piso = k.sigma_fixo ** 2  # PD-42: a incerteza nunca fica abaixo disto
        self.h = torch.zeros(n, self.m * self.m, device=device)
        self.P = torch.zeros(n, self.m * self.m, device=device)
        self.vista = torch.zeros(n, self.m * self.m, dtype=torch.bool, device=device)
        self.zref = torch.zeros(n, device=device)
        self.o = torch.zeros(n, 2, device=device)  # base no mapa, a menos de meia célula do centro
        self.psi = torch.zeros(n, device=device)  # rumo da base nos eixos do mapa (o mapa não gira)
        ax = (torch.arange(self.m, device=device) - self.c0).float() * k.celula
        gx, gy = torch.meshgrid(ax, ax, indexing="ij")
        self.cx, self.cy = gx.reshape(-1), gy.reshape(-1)  # centros no frame do rumo, (2601,)
        # Ordem do recorte (I §2.4): linha = x de +1,0 a −1,0; dentro da linha, y de +1,0 a −1,0.
        meia = self.c0 + k.n_recorte // 2
        q = meia - torch.arange(k.n_recorte, device=device)
        self.ordem = (q[:, None] * self.m + q[None, :]).reshape(-1)  # (441,) índices no mapa

    def _indice(self, x, y):
        """Índice plano da célula que contém (x, y), e se cai dentro do mapa interno."""
        cel = self.k.celula
        i = torch.floor(x / cel + 0.5).long() + self.c0  # centro da célula em múltiplos de 10 cm
        j = torch.floor(y / cel + 0.5).long() + self.c0
        ok = (i >= 0) & (i < self.m) & (j >= 0) & (j < self.m)
        return i.clamp(0, self.m - 1) * self.m + j.clamp(0, self.m - 1), ok

    def _no_mapa(self, x, y):
        """(x, y) no frame do rumo → eixos do mapa: R(ψ)·p + o."""
        c, s = torch.cos(self.psi)[:, None], torch.sin(self.psi)[:, None]
        return c * x - s * y + self.o[:, 0:1], s * x + c * y + self.o[:, 1:2]

    def reinicia(self, ids):
        """Zera o mapa dos envs `ids` (nunca vista) e esquece o z_ref anterior (PD-7)."""
        self.h[ids] = 0.0
        self.P[ids] = 0.0
        self.vista[ids] = False
        self.zref[ids] = 0.0
        self.o[ids] = 0.0
        self.psi[ids] = 0.0

    def prediz(self, dxy_rumo, dpsi, z_ref, dt):
        """Transporte do mapa pela odometria (PD-7): no plano, na vertical e na variância.

        ⚠ a célula mais perto a cada passo congelava o mapa: a 50 Hz o passo de 2–3 cm é menor que meia célula (PD-7; dono, 29/10).
        O mapa não gira e anda só em células inteiras: a base guarda o resto `o` (< meia célula) e o
        rumo `psi` nos eixos do mapa, e o recorte gira na leitura. Fora do mapa vira "nunca vista". Depois, nas
        células vistas, h −= z_ref − z_ref_anterior. Δxy está nos eixos do rumo do passo anterior.
        """
        k = self.k
        a = torch.cat(self._no_mapa(dxy_rumo[:, 0:1], dxy_rumo[:, 1:2]), 1)  # o + R(ψ)·Δxy, eixos do mapa
        d = torch.round(a / k.celula) * k.celula  # o mapa anda só em células inteiras, e não gira
        self.o, self.psi = a - d, self.psi + dpsi
        src, dentro = self._indice(self.cx + d[:, 0:1], self.cy + d[:, 1:2])
        v = self.vista.gather(1, src) & dentro
        h = self.h.gather(1, src)
        P = self.P.gather(1, src)
        h = torch.where(v, h - (z_ref - self.zref)[:, None], torch.zeros_like(h))
        cresce = k.q_desloc * dxy_rumo.norm(dim=1) + k.q_giro * dpsi.abs() + k.q_tempo * dt
        P = torch.where(v, P + cresce[:, None], torch.zeros_like(P))
        self.h, self.P, self.vista = h, P, v
        self.zref = z_ref.clone()

    def atualiza(self, pontos, validos, dist):
        """Kalman por célula com a média dos z da leitura (PD-6, PD-42).

        Por célula: n, z médio e d̄ (distância média). R = (σ_incl·d̄)² + σ_m²/n, com
        σ_m² = σ_alcance² + (d̄·σ_ângulo)². Célula nunca vista recebe h = z, P = R. Depois,
        P ≥ σ_fixo² nas células vistas.
        """
        k = self.k
        M = self.m * self.m
        E = pontos.shape[0]
        idx, dentro = self._indice(*self._no_mapa(pontos[..., 0], pontos[..., 1]))
        # ⚠ O balde M é o descarte: inválido ou fora do mapa nunca entra numa média.
        idx = torch.where(validos & dentro, idx, torch.full_like(idx, M))
        z = pontos[..., 2]
        zeros = torch.zeros(E, M + 1, device=z.device)
        n = zeros.scatter_add(1, idx, torch.ones_like(z))[:, :M]
        zm = zeros.scatter_reduce(1, idx, z, "mean", include_self=False)[:, :M]
        dm = zeros.scatter_reduce(1, idx, dist, "mean", include_self=False)[:, :M]
        sig_m2 = k.ruido_alcance ** 2 + (dm * math.radians(k.ruido_angulo_graus)) ** 2
        R = (math.radians(k.ruido_incl_graus) * dm) ** 2 + sig_m2 / n.clamp_min(1.0)
        medido = n > 0
        ganho = self.P / (self.P + R)  # R > 0 sempre (σ_alcance > 0): sem divisão por zero
        h_k = torch.where(self.vista, self.h + ganho * (zm - self.h), zm)
        P_k = torch.where(self.vista, (1 - ganho) * self.P, R)
        self.h = torch.where(medido, h_k, self.h)
        self.P = torch.where(medido, P_k, self.P)
        self.vista = self.vista | medido
        self.P = torch.where(self.vista, self.P.clamp_min(self.piso), self.P)

    def recorte(self) -> torch.Tensor:
        """(E,882): 441 alturas (±altura_clip) e depois 441 confianças (PD-6); nunca vista = 0 e 0."""
        i, _ = self._indice(*self._no_mapa(self.cx[self.ordem], self.cy[self.ordem]))  # gira só aqui; ≤ 1,5 m, dentro
        v = self.vista.gather(1, i)
        clip = self.k.altura_clip
        h = torch.where(v, self.h.gather(1, i).clamp(-clip, clip), torch.zeros_like(v, dtype=self.h.dtype))
        P = self.P.gather(1, i)
        c = torch.where(v, self.s2_ref / (self.s2_ref + P), torch.zeros_like(P))
        return torch.cat([h, c], dim=1)
