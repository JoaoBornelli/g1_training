"""Alvos macro da v3: referências da IK, frame da caixa, alvos geométricos, avanço e detector de fim.

Spec: S §8 (alvos macro), §8.2 (tolerâncias); I §4.2 (detector de fim); PD-29 (frame da caixa),
PD-36 (troca de tarefa), PD-47 (laje de destino). Plano §6.3. Módulo puro (§1.5): não importa o knobs,
recebe as seções como argumento `k`. Convenções: quaternion wxyz; `indice` long (E,) na ordem de
ALVOS, −1 = nenhum; tempos (9,) na mesma ordem.
"""
from __future__ import annotations

import math

import numpy as np
import torch

# Ordem de ALVOS do knobs.py, copiada aqui: este módulo não importa o knobs (§1.5).
P0, P1, P2, P3, P4, B1, B2, B3, B4 = range(9)
# Tolerância numérica do relógio em float: 25 passos de 0,02 s somam 0,4999… e não podem perder o aviso.
# Não é número de treino (§1.5), só a folga de ponto flutuante.
_EPS = 1e-9


def _pos(nos: torch.Tensor, x: torch.Tensor):
    # Índices do nó de baixo e de cima e o peso do de cima. x é preso nas bordas da grade (clamp, S §8.1).
    x = x.clamp(float(nos[0]), float(nos[-1]))
    i0 = (torch.bucketize(x, nos, right=True) - 1).clamp(0, len(nos) - 2)
    w = (x - nos[i0]) / (nos[i0 + 1] - nos[i0])
    return i0, i0 + 1, w


def _bilin(grade, topos, meias, h, a):
    # Bilinear na grade (h, a) de forma (6, 3); h e a são (E,). Nós e grade vão no dtype de h.
    g = grade.to(h)
    i0, i1, wh = _pos(topos.to(h), h)
    j0, j1, wa = _pos(meias.to(a), a)
    baixo = g[i0, j0] * (1 - wa) + g[i0, j1] * wa
    cima = g[i1, j0] * (1 - wa) + g[i1, j1] * wa
    return baixo * (1 - wh) + cima * wh


class Referencias:
    """Referências macro da IK (decisão de 14/10). Lê `referencias.npz`; nunca `qpos` de pega.

    A grade (h, a) vem de `ref_agachado.npz` (S §8.1). As três poses de pé com caixa (uma por meia
    aresta, 14/20/26 cm) são interpoladas só em `a`. Tudo em float64 no device pedido.
    """

    def __init__(self, caminho: str, device: str):
        d = np.load(caminho, allow_pickle=False)

        def f(chave):
            return torch.as_tensor(np.asarray(d[chave]), dtype=torch.float64, device=device)

        self.topos = f("topos")  # (6,) altura do apoio, m
        self.meias = f("meias")  # (3,) meia aresta da caixa, m
        self._pelve = f("grade_pelve")  # (6,3)
        self._tronco = f("grade_tronco")  # (6,3) rad
        self._recuo = f("grade_recuo")  # (6,3) m
        self._caixa_tb = f("caixa_transporte_b")  # (3,3)
        self._palmas_cc = f("palmas_com_caixa_b")  # (3,2,3)
        self._q_cc = f("q_com_caixa")  # (3,29)
        self._pelve_cc = f("pelve_com_caixa")  # (3,)
        self.palmas_de_pe_b = f("palmas_de_pe_b")  # (2,3)
        self.pes_de_pe_b = f("pes_de_pe_b")  # (2,3), frame da pelve (PD-16)
        self.cantos_sola = f("cantos_sola")  # (4,3), frame do *_ankle_roll_link (PD-35)
        self.q_de_pe = f("q_de_pe")  # (29,)
        self.pelve_de_pe = float(d["pelve_de_pe"])  # 0,789
        self.site_pe_de_pe = float(d["site_pe_de_pe"])

    def pelve_z(self, h, a):
        return _bilin(self._pelve, self.topos, self.meias, h, a)

    def tronco(self, h, a):
        return _bilin(self._tronco, self.topos, self.meias, h, a)

    def recuo(self, h, a):
        return _bilin(self._recuo, self.topos, self.meias, h, a)

    def caixa_transporte_b(self, a):
        return self._na_meia(self._caixa_tb, a)

    def palmas_com_caixa_b(self, a):
        return self._na_meia(self._palmas_cc, a)

    def q_com_caixa(self, a):
        return self._na_meia(self._q_cc, a)

    def pelve_com_caixa(self, a):
        return self._na_meia(self._pelve_cc, a)

    def _na_meia(self, tabela, a):
        # Linear só em `a` sobre as 3 linhas da pose de pé com caixa (uma por caixa).
        i0, i1, w = _pos(self.meias.to(a), a)
        t = tabela.to(a)
        w = w.reshape((-1,) + (1,) * (t.dim() - 1))
        return t[i0] * (1 - w) + t[i1] * w


def _matriz_do_quat(q):
    # wxyz (E,4) → matriz de rotação (E,3,3). Normaliza: o quaternion vem da simulação com erro de ponto flutuante.
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def frame_da_caixa(quat_caixa_w, rumo_base):
    """Frame da caixa (E,3,3), colunas x, y, z (PD-29).

    z = a normal de face mais perto da vertical do mundo (com sinal para cima); x = a normal horizontal
    de face mais alinhada ao rumo da base; y = z × x. ⚠ x sai só das faces horizontais: a face de cima
    ou de baixo nunca vira a frente, ou a caixa inclinada trocaria a frente pelo topo.
    """
    r = _matriz_do_quat(quat_caixa_w)
    cols = r.transpose(1, 2)  # (E,3,3): cols[:, i] = normal da face i da caixa, no mundo
    iz = r[:, 2, :].abs().argmax(-1)  # índice da coluna mais vertical
    z = cols[torch.arange(len(iz)), iz]
    z = z * (1 - 2 * (z[:, 2:3] < 0).to(z))  # z aponta para cima
    cand = torch.cat([cols, -cols], 1)  # (E,6,3): as 6 normais, ±
    u = torch.stack([torch.cos(rumo_base), torch.sin(rumo_base), torch.zeros_like(rumo_base)], -1)
    dot = (cand * u.unsqueeze(1)).sum(-1)  # (E,6)
    excluidas = (torch.arange(6, device=dot.device) % 3).unsqueeze(0) == iz.unsqueeze(1)
    dot = dot.masked_fill(excluidas, float("-inf"))
    x = cand[torch.arange(len(iz)), dot.argmax(-1)]
    y = torch.linalg.cross(z, x, dim=-1)
    return torch.stack([x, y, z], -1)


def inclinacao_da_caixa(quat_caixa_w):
    """Ângulo (E,) em rad entre a face mais vertical da caixa e a vertical. Zero com uma face na vertical."""
    r = _matriz_do_quat(quat_caixa_w)
    cos_max = r[:, 2, :].abs().max(-1).values  # cosseno da face mais vertical
    return torch.acos(cos_max.clamp(0.0, 1.0))


def palmas_alvo_caixa(indice, a, k):
    """Palmas-alvo (E,2,3) no frame da caixa (S §8.2). Palma 0 = +y (esquerda), palma 1 = −y.

    P0: x = −(a + p0_frente), y = ±(a + palma_face); P1 e P2: x = 0; B3: y afastado por b3_afasta.
    ⚠ Alvos sem uso (P3, P4, B1, B2, B4) devolvem 0, nunca NaN (§6.6): o termo os zera pela máscara.
    """
    usa = (indice == P0) | (indice == P1) | (indice == P2) | (indice == B3)
    zero = torch.zeros_like(a)
    x = torch.where(indice == P0, -(a + k.p0_frente), zero)
    y = a + k.palma_face + (indice == B3).to(a) * k.b3_afasta
    lado = a.new_tensor([1.0, -1.0])
    x2 = x.unsqueeze(1).expand(-1, 2)
    y2 = y.unsqueeze(1) * lado
    z2 = zero.unsqueeze(1).expand(-1, 2)
    out = torch.stack([x2, y2, z2], -1)
    return torch.where(usa.view(-1, 1, 1), out, torch.zeros_like(out))


def base_para_mundo(base_xy, rumo, p):
    """Frame da base → mundo (E,2): base + R(rumo)·p (F9 R3). Núcleo único: eventos, caixa_alvo_w e o driver chamam."""
    c, s = torch.cos(rumo), torch.sin(rumo)
    return base_xy + torch.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]], -1)


def caixa_alvo_w(indice, caixa_na_pega_w, alvo_soltura_w, base_pos_w, rumo, chao_z, a, ref, k):
    """Posição-alvo da caixa (E,3) no mundo (S §8.2).

    P3 = caixa na pega + (0,0,p3_alto); P4 = base + R(rumo)·caixa_transporte_b(a), com z = chao_z +
    z_transporte (o z da tabela é descartado); B1 = soltura + (0,0,b1_alto); B2 = soltura. Os outros: 0.
    """
    ez = caixa_na_pega_w.new_tensor([0.0, 0.0, 1.0])
    tb = ref.caixa_transporte_b(a)
    xy4 = base_para_mundo(base_pos_w[:, :2], rumo, tb[:, :2])
    p4 = torch.cat([xy4, (chao_z + k.z_transporte).unsqueeze(-1)], -1)
    candidatos = (
        (P3, caixa_na_pega_w + k.p3_alto * ez),
        (P4, p4),
        (B1, alvo_soltura_w + k.b1_alto * ez),
        (B2, alvo_soltura_w),
    )
    out = torch.zeros_like(caixa_na_pega_w)
    for alvo, val in candidatos:
        out = torch.where((indice == alvo).unsqueeze(-1), val, out)
    return out


def laje_destino(recuo, theta, y, meia_xy, borda):
    """Alvo e centro da caixa na laje de destino no frame da base (PD-47). Devolve ((E,2), (E,2)).

    Alvo em (recuo + e(θ), y), com e(θ) = borda·cos θ + meia_xy·sen|θ| − borda: o canto mais perto
    fica à mesma distância da base quando a laje gira. Centro = alvo + R(θ)·(meia_xy − borda, 0).
    """
    e = borda * torch.cos(theta) + meia_xy * torch.sin(theta.abs()) - borda
    alvo = torch.stack([recuo + e, y], -1)
    d = meia_xy - borda
    centro = alvo + d * torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    return alvo, centro


def avanca(indice, ok_agora, relogio, dt, tempos):
    """Avança o alvo quando `ok_agora` vale pelo tempo do alvo (S §8.2). Devolve (indice, relogio, avancou).

    `ok_agora` falso zera o relógio (o alvo só avança com a condição sustentada). Depois de B4 o índice
    vira −1 (nenhum): o controlador troca a tarefa e a PD-36 zera o progresso nesse passo.
    ⚠ Alvo sem tempo (P2, tempo 0) avança no primeiro passo com `ok_agora`.
    """
    valido = indice >= 0
    tempo = tempos.to(relogio)[indice.clamp(min=0)]
    rel = torch.where(ok_agora, relogio + dt, torch.zeros_like(relogio))
    avancou = ok_agora & valido & (rel >= tempo - _EPS)
    prox = indice + 1
    prox = torch.where(prox > B4, torch.full_like(prox, -1), prox)
    novo_indice = torch.where(avancou, prox, indice)
    novo_relogio = torch.where(avancou, torch.zeros_like(rel), rel)
    return novo_indice, novo_relogio, avancou


def detecta_fim(pelve_sobre_pe, tronco, palmas_b, vel_juntas, ref_pelve, ref_palmas_b,
                relogio, dt, k):
    """Aviso de fim de PEGAR ou BOTAR (I §4.2). Devolve (aviso (E,) bool, novo_relogio (E,)).

    Pose dentro das tolerâncias (pelve ±3 cm, tronco ±5°, cada palma ±5 cm) e juntas paradas
    (média de |q̇| abaixo de juntas_paradas), sustentadas por estavel_s. ⚠ Usa a média do instante, e
    não a média móvel de estavel_s: o relógio é o único estado (E,) desta função. Ver dúvida no relatório.
    """
    pelve_ok = (pelve_sobre_pe - ref_pelve).abs() <= k.tol_pelve
    tronco_ok = tronco.abs() <= math.radians(k.tol_tronco_graus)
    palmas_ok = (torch.linalg.norm(palmas_b - ref_palmas_b, dim=-1) <= k.tol_palma).all(-1)
    parado = vel_juntas.abs().mean(-1) < k.juntas_paradas
    ok = pelve_ok & tronco_ok & palmas_ok & parado
    rel = torch.where(ok, relogio + dt, torch.zeros_like(relogio))
    return rel >= k.estavel_s - _EPS, rel
