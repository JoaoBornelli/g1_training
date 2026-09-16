"""Gera a pose de REFERÊNCIA do BOTAR por IK, numa grade de laje × tamanho de caixa.

    python -m g1_limpo.ik.gera_botar --cena ~/g1_pilota/ --saida ref_botar.npz

⚠ SÓ O `botar`. O portão de escopo (15/09, `model_15999`, três lajes) mediu o `pegar`
IDÊNTICO nas três alturas — pelve 0,773, tronco 14°, CoM 0,02 m dos pés — e sem
dispersão. Não há forma a corrigir ali. O BOTAR é o defeito: o tronco vai a 84° a 99°,
e o robô DEITA para pousar a caixa.

⚠⚠ O MODELO É O `cena.mjb` EXPORTADO, e não uma remontagem por `cena.py`. Ele É a cena
compilada do treino: tem a caixa, a laje e os `*_palm_pad`. Remontar as specs à mão
reintroduziria a armadilha 1 do plano — MJCF diferente entre gerador e treino — que é
exatamente o que as oito sondas antigas sofrem ao ler o `g1.xml` cru. O hash do
arquivo vai na saída.

⚠ `mj_forward` e `mj_kinematics` apenas. NUNCA `mj_step`: a pose é cinemática, e um
passo de física a derrubaria antes de o solver convergir.

AS TRÊS GRANDEZAS que o termo vai rastrear, medidas na pose resolvida:

    pelve_z       altura da pelve no mundo
    tronco_incl   ângulo do eixo z do `torso_link` contra a vertical, em graus
    com_offset    distância horizontal do CoM COMBINADO ao meio dos pés

⚠ O CoM é COMBINADO: robô mais os 5 kg da caixa, no CENTRO dela. Somar a massa no
ponto médio das palmas erra — foi o que inflou o `com_offset` do `pegar` de 0,02 para
0,17 na primeira medição.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mujoco
import numpy as np

# ⚠ O ALVO DO BOTAR, transcrito de `comando.py:1592-1640`. A laje nasce a
# `_AVANCO_LAJE_BOTAR` à frente da base, e o alvo RECUA `botar_recuo_borda` dela em
# direção ao robô — no centro do tampo ele alcançaria por cima de 20 cm de pedra.
AVANCO_LAJE = 0.50
RECUO_BORDA = 0.15
MASSA_CAIXA = 5.0
# ⚠ O PESO DA TAREFA DE REPOUSO. Ele entra no MESMO vetor de resíduo das outras, e o
# amortecimento do solver é 2e-2. ⚠⚠ ELE VALE 0,005, e não 0,05: o resíduo de repouso
# é `peso × |q − q_default|` em RADIANOS, e os outros são em METROS. Com 0,05 um desvio
# de 1 rad custa o mesmo que 5 cm de erro de pé, e as 29 juntas somadas venciam as 6
# tarefas de mão e as 9 de pé — o solver parava num compromisso, com o pé 6 cm dentro
# do chão. MEDIDO: 0,05 dá resíduo 0,020; 0,005 dá 0,0005.
PESO_REPOUSO = 0.005
# ⚠⚠ O site da palma fica FORA da face da caixa por esta folga, e ela é o RAIO DA
# CÁPSULA DO PUNHO (`*_wrist_collision`, r = 0,035, no `wrist_pitch_link`). MEDIDO no
# modelo: o pad tem o centro a 0,015 do site (`cena._PALM_DZ`) e meia-espessura 0,008,
# logo a face externa do pad está a 0,023 do site — mas a cápsula do punho, no mesmo
# eixo, sai 0,035. Com o pad plano na caixa a cápsula entra 12 mm nela. Com 0,023 o
# solver parava num compromisso: mão 11 mm fora do alvo e punho 1 mm dentro. Com 0,035
# o PUNHO encosta na caixa e o pad fica 12 mm da face, plano. É o que o modelo da cena
# permite sem peça dentro de peça. A NORMAL do pad (eixo y do punho) é uma tarefa —
# sem ela o punho girava livre e um canto do pad entrava 5 mm na caixa.
FOLGA_PALMA = 0.035
# ⚠ O SITE DO PÉ NÃO É A SOLA. MEDIDO no keyframe: o site fica em z = 0,0237 e a base
# das cápsulas do pé em 0,0257 — o site está 2,0 mm ABAIXO da sola. Pedir `site_z = 0`
# enterrava a sola nesses 2 mm, e foi o "pés atravessando o chão" da inspeção visual.
Z_SITE_ALVO = -0.0020
# ⚠ LARGURA MÍNIMA DO APOIO. A tarefa antiga só tinha TETO (0,40 m) e simetria: nada
# impedia os dois pés de se juntarem, e a inspeção visual pegou isso. O keyframe do
# fabricante apoia com 0,237 m; 0,20 deixa folga para o IK escolher.
LARGURA_MIN = 0.20
LARGURA_MAX = 0.40
# ⚠⚠ A PELVE NÃO PODE PASSAR DE 70°: é `bad_orientation` (`env_cfg.py:286`), e o
# episódio TERMINA como queda. O termo entra como dobradiça no resíduo, em radianos,
# a 60° — dez graus de folga contra a terminação. O portão aceita até 65°: cinco graus
# de tolerância para o solver, e ainda cinco antes da queda.
PELVE_MAX = np.radians(60.0)
PELVE_PORTAO = 65.0
# ⚠⚠ NENHUMA PEÇA DENTRO DE OUTRA. É o que trava a transferência para o robô real: a
# referência de hoje põe o antebraço 9 cm dentro da coxa, a canela 4 cm dentro da caixa
# e o pad da palma 2 cm dentro da caixa, com o punho torto. O termo mede a distância
# ASSINADA entre geoms de colisão com `mj_geomDistance`, que ignora `contype`, e cobra
# uma dobradiça quando a folga cai abaixo de `FOLGA_MIN`. Pares: robô × robô e robô ×
# caixa. A LAJE FICA FORA por decisão (16/09): só as poses com a caixa importam agora.
# Robô × caixa tem folga ZERO: a caixa é o que as mãos seguram, encostar é o normal,
# entrar não é. Robô × robô pede `FOLGA_MIN`. Um par só entra no resíduo quando chega a `BANDA_ATIVA` da folga; fora dela a
# derivada é zero e o par é custo.
FOLGA_MIN = 0.010
BANDA_ATIVA = 0.020
DIST_MAX = 0.05

LAJES = (0.04, 0.15, 0.30, 0.45, 0.55)
MEIAS = (0.07, 0.10, 0.13)


def _hash(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()[:16]


class Cena:
    """O `cena.mjb` com a laje e a caixa reposicionáveis, e a caixa redimensionável."""

    def __init__(self, pasta: Path):
        self.caminho = pasta / "cena.mjb"
        self.m = mujoco.MjModel.from_binary_path(str(self.caminho))
        self.d = mujoco.MjData(self.m)
        self.hash = _hash(self.caminho)
        n = lambda t, s: mujoco.mj_name2id(self.m, t, s)          # noqa: E731
        self.bid_pelve = n(mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
        self.bid_torso = n(mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
        self.bid_caixa = n(mujoco.mjtObj.mjOBJ_BODY, "box/box")
        self.bid_laje = n(mujoco.mjtObj.mjOBJ_BODY, "table/table")
        self.gid_caixa = n(mujoco.mjtObj.mjOBJ_GEOM, "box/box_geom")
        self.sid = {s: n(mujoco.mjtObj.mjOBJ_SITE, f"robot/{s}")
                    for s in ("left_foot", "right_foot", "left_palm", "right_palm")}
        self.adr_caixa = int(self.m.jnt_qposadr[int(self.m.body_jntadr[self.bid_caixa])])
        self.meia_laje = float(
            self.m.geom_size[n(mujoco.mjtObj.mjOBJ_GEOM, "table/table_geom")][2])
        # ⚠ as 29 juntas do robô ocupam `qpos[7:36]`; a caixa fica em `[36:43]`
        self.n_juntas = 29
        self.lo = self.m.jnt_range[1:1 + self.n_juntas, 0].copy()
        self.hi = self.m.jnt_range[1:1 + self.n_juntas, 1].copy()
        # ⚠⚠ A POSE DE REPOUSO, e sem ela o IK deita o robô. O sistema tem 32 variáveis
        # e 16 tarefas: ele é SUBDETERMINADO, e o solver escolhe qualquer ponto do
        # espaço nulo. A primeira versão semeava `zeros(29)` e saía com o tronco a 110°
        # — PIOR que o robô, que faz 84° a 99°. A seção 2.2 do plano pede esta tarefa,
        # com peso BAIXO, e eu a tinha omitido.
        self.q_rep = np.asarray(np.load(pasta / "cena.npz")["q_default"], dtype=float)
        assert len(self.q_rep) == self.n_juntas
        self._ft = np.zeros(6)
        self._pares_de_folga()
        self.pes_alvo: dict[str, tuple[float, float]] | None = None
        self.n_pe = 9                 # entradas de pé no resíduo; 11 com `pes_de`
        # ⚠ SEM ROTAÇÃO INTERNA DO JOELHO (16/09, inspeção visual): ela junta e torce as
        # pernas, e o robô real perde a base. Os dois `hip_yaw` têm eixo (0, 0, 1): na
        # esquerda o positivo gira o joelho para FORA, na direita é o negativo. O sinal
        # da rotação interna é, portanto, esquerda < 0 e direita > 0.
        self.i_yaw = {lado: int(self.m.jnt_qposadr[
            n(mujoco.mjtObj.mjOBJ_JOINT, f"robot/{lado}_hip_yaw_joint")]) - 7
            for lado in ("left", "right")}

    def _pares_de_folga(self) -> None:
        """Os pares de geoms que o termo de folga vigia: robô × robô e robô × caixa.
        Fora ficam o mesmo corpo, pai-filho, e os pares ESTRUTURAIS — os
        que já ficam abaixo da folga na pose de repouso, como cotovelo × punho (−22 mm)
        e pelve × quadril (−5 mm): eles são o desenho do robô, e não um defeito."""
        m = self.m
        nome = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""   # noqa: E731
        robo = [g for g in range(m.ngeom) if m.geom_contype[g] and nome(g).startswith("robot/")]
        pares, folga = [], []
        for i, a in enumerate(robo):
            for b in robo[i + 1:]:
                ba, bb = m.geom_bodyid[a], m.geom_bodyid[b]
                if ba == bb or m.body_parentid[ba] == bb or m.body_parentid[bb] == ba:
                    continue
                pares.append((a, b)); folga.append(FOLGA_MIN)
            pares.append((a, self.gid_caixa)); folga.append(0.0)
        self.pares = np.array(pares); self.folga_par = np.array(folga)
        self.cenario(5.0, 0.10)                       # caixa e laje longe do robô
        self.poe(self.q_rep, 0.78, 0.0, 0.0, np.array([5.0, 0.0, 5.0]))
        vivos = self.folgas() >= self.folga_par
        self.estruturais = [(nome(a), nome(b)) for a, b in self.pares[~vivos]]
        self.pares = self.pares[vivos]; self.folga_par = self.folga_par[vivos]

    def pes_de(self, npz: Path, k: int) -> None:
        """Copia a POSE DAS PERNAS da pose `k` de um npz anterior: o y e o rumo de cada
        pé, no frame da pelve. Pedido de 16/09: a pose 3 saiu com separação lateral, pés
        paralelos e alinhados, e ela vira o alvo dos pés em todas as alturas.
        ⚠ O x dos pés contra a pelve NÃO é copiado: ele é o recuo do agachamento e muda
        com a altura. A primeira versão copiava (+0,176 m) e na laje de 0,55 a perna não
        alcançava — a pelve travava em 0,65 e o erro se espalhava por todas as tarefas.
        Os dois pés ficam ALINHADOS entre si em x; o recuo fica livre."""
        ref = np.load(npz)
        assert ref["hash_mjcf"] == self.hash, "npz de outro MJCF"
        self.d.qpos[:] = ref["qpos"][k]
        mujoco.mj_kinematics(self.m, self.d)
        self.pes_alvo = {}
        for s in ("left_foot", "right_foot"):
            p = self.d.site_xpos[self.sid[s]]; R = self.d.site_xmat[self.sid[s]].reshape(3, 3)
            self.pes_alvo[s] = (float(p[1]), float(np.arctan2(R[1, 0], R[0, 0])))
        self.n_pe = 11

    def folgas(self, idx: np.ndarray | None = None) -> np.ndarray:
        """Distância assinada de cada par vigiado, limitada por cima em `DIST_MAX`."""
        pares = self.pares if idx is None else self.pares[idx]
        return np.array([mujoco.mj_geomDistance(self.m, self.d, int(a), int(b), DIST_MAX, self._ft)
                         for a, b in pares])

    def cenario(self, topo: float, a: float) -> None:
        """Põe o TOPO da laje em `topo` e redimensiona o cubo.

        ⚠ A laje do `cena.mjb` NÃO é mocap (`body_mocapid` = −1): ela é um corpo estático
        em z = 0,53. Antes desta função o gerador e o viewer a deixavam FIXA, com o topo
        em 0,55 m para TODAS as alturas — e a folga contra ela não media nada. Corpo
        estático se move por `body_pos`, que a cinemática lê a cada chamada.
        As duas fórmulas do box são as de `eventos.tamanho_caixa`."""
        self.m.body_pos[self.bid_laje] = (AVANCO_LAJE, 0.0, topo - self.meia_laje)
        self.m.geom_size[self.gid_caixa] = (a, a, a)
        self.m.geom_rbound[self.gid_caixa] = a * np.sqrt(3.0)
        # ⚠ No MuJoCo CLÁSSICO o `geom_aabb` é `(ngeom, 6)`: centro e meia-extensão
        # lado a lado. No `mujoco_warp` do treino ele é `(nworld, ngeom, 2, 3)`, e é por
        # isso que `eventos.tamanho_caixa` indexa `[..., 1]`. Copiar aquele índice para
        # cá dava `ValueError` — e um índice errado que NÃO explodisse seria pior.
        self.m.geom_aabb[self.gid_caixa, 3:6] = (a, a, a)

    def poe(self, q: np.ndarray, pelve_z: float, roll: float, pitch: float,
            caixa_xyz: np.ndarray) -> None:
        d = self.d
        d.qpos[:3] = (0.0, 0.0, pelve_z)
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([roll, pitch, 0.0]), b"xyz")
        d.qpos[3:7] = quat
        d.qpos[7:7 + self.n_juntas] = q
        d.qpos[self.adr_caixa:self.adr_caixa + 3] = caixa_xyz
        d.qpos[self.adr_caixa + 3:self.adr_caixa + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)


def residuo(c: Cena, caixa_xyz: np.ndarray, a: float,
            peso_rep: float = PESO_REPOUSO, ativos: np.ndarray | None = None) -> np.ndarray:
    """O vetor de erro das tarefas. Zero é a pose pedida."""
    d = c.d
    e: list[float] = []
    # pés: no chão e horizontais
    for s in ("left_foot", "right_foot"):
        p = d.site_xpos[c.sid[s]]
        R = d.site_xmat[c.sid[s]].reshape(3, 3)
        e += [p[2] - Z_SITE_ALVO, R[0, 2], R[1, 2]]
    # pés simétricos em y, e a largura do apoio DENTRO da faixa
    ly = d.site_xpos[c.sid["left_foot"]][1]; ry = d.site_xpos[c.sid["right_foot"]][1]
    larg = ly - ry
    if c.pes_alvo is None:
        e += [ly + ry, max(0.0, larg - LARGURA_MAX), max(0.0, LARGURA_MIN - larg)]
    else:
        # pés no y e no rumo da referência, e alinhados entre si em x (substitui a
        # simetria e a faixa de largura); o recuo contra a pelve fica livre
        for s, (y0, yaw0) in c.pes_alvo.items():
            p = d.site_xpos[c.sid[s]]; R = d.site_xmat[c.sid[s]].reshape(3, 3)
            e += [p[1] - y0, np.arctan2(R[1, 0], R[0, 0]) - yaw0]
        e += [d.site_xpos[c.sid["left_foot"]][0] - d.site_xpos[c.sid["right_foot"]][0]]
    # palmas nas faces laterais da caixa, no FRAME DELA (a caixa nasce sem giro aqui,
    # portanto o frame é o do mundo transladado)
    lado = a + FOLGA_PALMA
    e += list(d.site_xpos[c.sid["left_palm"]] - (caixa_xyz + (0.0, +lado, 0.0)))
    e += list(d.site_xpos[c.sid["right_palm"]] - (caixa_xyz + (0.0, -lado, 0.0)))
    # a normal do pad (eixo y do punho) alinhada à face da caixa, nas duas mãos: o
    # giro EM TORNO da normal continua livre
    for s in ("left_palm", "right_palm"):
        R = d.site_xmat[c.sid[s]].reshape(3, 3)
        e += [R[0, 1], R[2, 1]]
    # CoM COMBINADO sobre o meio dos pés
    m_robo = float(c.m.body_mass[1:1 + 30].sum())
    com = ((m_robo * d.subtree_com[c.bid_pelve] + MASSA_CAIXA * caixa_xyz)
           / (m_robo + MASSA_CAIXA))
    mid = 0.5 * (d.site_xpos[c.sid["left_foot"]] + d.site_xpos[c.sid["right_foot"]])
    e += [com[0] - mid[0], com[1] - mid[1]]
    # pelve abaixo do limite de queda (dobradiça: zero enquanto obedece)
    e += [max(0.0, inclinacao(c, c.bid_pelve) - PELVE_MAX)]
    # sem rotação interna do joelho (dobradiça: zero em neutro ou para fora)
    q = d.qpos[7:7 + c.n_juntas]
    e += [max(0.0, -q[c.i_yaw["left"]]), max(0.0, q[c.i_yaw["right"]])]
    # folga entre peças: dobradiça por par ATIVO (o conjunto é fixo dentro da iteração,
    # para que o jacobiano numérico compare o mesmo par consigo mesmo)
    if ativos is not None and len(ativos):
        e += list(np.maximum(0.0, c.folga_par[ativos] - c.folgas(ativos)))
    # ⚠ PESO BAIXO, de propósito. Ela não disputa com as mãos nem com o CoM: ela só
    # escolhe UM ponto dentro do espaço nulo que as outras tarefas deixam livre.
    e += list(peso_rep * (d.qpos[7:7 + c.n_juntas] - c.q_rep))
    return np.array(e)


def resolve(c: Cena, topo: float, a: float, semente: int,
            iters: int = 240, margem: float = 0.10,
            peso_rep: float = PESO_REPOUSO) -> tuple[float, float, np.ndarray]:
    """Newton amortecido sobre 32 variáveis: 29 juntas, pelve z, roll e pitch da raiz.

    ⚠ O jacobiano é NUMÉRICO, herdado do `ikc.py`. Ele custa 32 cinemáticas por
    iteração. Trocar por `mj_jacSite` vale quando o portão da seção 3 reprovar por
    precisão; enquanto ele passar, o numérico já está validado.
    """
    c.cenario(topo, a)
    caixa_xyz = np.array([AVANCO_LAJE - RECUO_BORDA, 0.0, topo + a])
    rng = np.random.default_rng(semente)
    q0 = np.clip(c.q_rep, c.lo + 1e-3, c.hi - 1e-3)
    ruido = (rng.random(c.n_juntas) - 0.5) * (0.7 if semente else 0.0)
    st = np.concatenate([np.clip(q0 + ruido, c.lo + 1e-3, c.hi - 1e-3),
                         [0.78], [0.0, 0.0]])          # 0,78 = a pelve do keyframe
    for _ in range(iters):
        c.poe(st[:29], st[29], st[30], st[31], caixa_xyz)
        ativos = np.flatnonzero(c.folgas() < c.folga_par + BANDA_ATIVA)
        f0 = residuo(c, caixa_xyz, a, peso_rep, ativos)
        J = np.zeros((len(f0), 32))
        for k in range(32):
            s = st.copy(); s[k] += 1e-5
            c.poe(s[:29], s[29], s[30], s[31], caixa_xyz)
            J[:, k] = (residuo(c, caixa_xyz, a, peso_rep, ativos) - f0) / 1e-5
        passo = J.T @ np.linalg.solve(J @ J.T + 2e-2 * np.eye(len(f0)), -f0)
        st = st + np.clip(passo, -0.15, 0.15)
        folga = margem * (c.hi - c.lo) / 2.0
        st[:29] = np.clip(st[:29], c.lo + folga, c.hi - folga)
        st[29] = np.clip(st[29], 0.25, 1.05)
    c.poe(st[:29], st[29], st[30], st[31], caixa_xyz)
    f = residuo(c, caixa_xyz, a, peso_rep)
    # ⚠⚠ OS DOIS RESÍDUOS, e a versão anterior só devolvia o da MÃO. O comentário dela
    # dizia "pé no chão o `clip` garante" — e não garante: nada no `clip` de limite de
    # junta põe a sola no chão. Uma pose com o pé 5 cm enterrado passava como `ok`, e a
    # inspeção visual foi quem pegou. O portão agora vê os dois.
    n = c.n_pe
    return float(np.abs(f[:n]).max()), float(np.abs(f[n:n + 6]).max()), st


def inclinacao(c: Cena, bid: int) -> float:
    """Ângulo do eixo z do corpo contra a vertical, em radianos. É a MESMA grandeza que
    o `bad_orientation` lê da raiz via `projected_gravity_b`."""
    return float(np.arccos(np.clip(c.d.xmat[bid].reshape(3, 3)[2, 2], -1.0, 1.0)))


def grandezas(c: Cena, a: float, caixa_xyz: np.ndarray) -> tuple[float, float, float, float]:
    d = c.d
    pelve_z = float(d.xpos[c.bid_pelve][2])
    tronco = float(np.degrees(inclinacao(c, c.bid_torso)))
    m_robo = float(c.m.body_mass[1:1 + 30].sum())
    com = ((m_robo * d.subtree_com[c.bid_pelve] + MASSA_CAIXA * caixa_xyz)
           / (m_robo + MASSA_CAIXA))
    mid = 0.5 * (d.site_xpos[c.sid["left_foot"]] + d.site_xpos[c.sid["right_foot"]])
    return (pelve_z, tronco, float(np.hypot(com[0] - mid[0], com[1] - mid[1])),
            float(np.degrees(inclinacao(c, c.bid_pelve))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cena", required=True, help="pasta com cena.mjb e cena.npz")
    ap.add_argument("--saida", default="ref_botar.npz")
    ap.add_argument("--sementes", type=int, default=6)
    ap.add_argument("--margem", type=float, default=0.10,
                    help="fração do curso proibida em cada ponta. O criterio de "
                         "rejeicao do plano e |frac| > 0,95, ou seja margem 0,05")
    ap.add_argument("--iters", type=int, default=240)
    ap.add_argument("--peso-repouso", type=float, default=PESO_REPOUSO)
    ap.add_argument("--lajes", type=float, nargs="*", default=list(LAJES))
    ap.add_argument("--meias", type=float, nargs="*", default=list(MEIAS))
    ap.add_argument("--pernas-de", help="npz anterior de onde copiar a pose das pernas")
    ap.add_argument("--pernas-pose", type=int, default=4, help="índice da pose no npz")
    args = ap.parse_args()

    c = Cena(Path(args.cena).expanduser())
    if args.pernas_de:
        c.pes_de(Path(args.pernas_de).expanduser(), args.pernas_pose)
        print(f"[gera_botar] pernas da pose {args.pernas_pose} de {args.pernas_de}: "
              + "  ".join(f"{s} y {y:+.3f} rumo {np.degrees(w):+.1f}°"
                          for s, (y, w) in c.pes_alvo.items()))
    print(f"[gera_botar] {c.caminho}  hash {c.hash}")
    print(f"[gera_botar] {len(c.pares)} pares de folga vigiados; {len(c.estruturais)} "
          f"estruturais fora: {sorted({a.split('/')[-1] + ' x ' + b.split('/')[-1] for a, b in c.estruturais})}")
    print(f"{'topo':>6} {'meia':>6} {'res_pe':>8} {'res_mao':>8} {'pelve_z':>8} "
          f"{'tronco':>8} {'com_off':>8} {'pelve_i':>8} {'folga_mm':>8}  {'veredito'}")
    linhas = []
    for topo in args.lajes:
        for a in args.meias:
            melhor = None
            for s in range(args.sementes):
                e_pe, e_mao, st = resolve(c, topo, a, s, args.iters, args.margem, args.peso_repouso)
                # ⚠ o pior dos dois manda: uma pose com a mão perfeita e o pé enterrado
                # não é melhor que uma com os dois medianos.
                if melhor is None or max(e_pe, e_mao) < max(melhor[0], melhor[1]):
                    melhor = (e_pe, e_mao, st)
            e_pe, e_mao, st = melhor
            caixa_xyz = np.array([AVANCO_LAJE - RECUO_BORDA, 0.0, topo + a])
            c.poe(st[:29], st[29], st[30], st[31], caixa_xyz)
            g1, g2, g4, gp = grandezas(c, a, caixa_xyz)
            frac = float((np.abs(st[:29] - (c.hi + c.lo) / 2) / ((c.hi - c.lo) / 2)).max())
            # ⚠ o portão da folga é "nada DENTRO de nada": a pior distância assinada
            # entre pares vigiados não pode ser negativa (2 mm de tolerância do solver)
            folga = float(c.folgas().min())
            # separação lateral dos pés: a pelve está em yaw 0, logo é o |Δy| dos sites
            larg = float(abs(c.d.site_xpos[c.sid["left_foot"]][1]
                             - c.d.site_xpos[c.sid["right_foot"]][1]))
            ok = (e_pe < 0.01 and e_mao < 0.02 and frac < 0.95 and gp < PELVE_PORTAO
                  and folga > -0.002)
            print(f"{topo:6.2f} {a:6.2f} {e_pe:8.4f} {e_mao:8.4f} {g1:8.3f} {g2:8.1f} "
                  f"{g4:8.3f} {gp:8.1f} {folga * 1000:8.1f}  {'ok' if ok else 'REJEITADA'}  |frac|max {frac:.2f}")
            linhas.append(dict(topo=topo, meia=a, residuo=e_mao, residuo_pe=e_pe,
                               pelve_z=g1, pelve_incl=gp, folga_min=folga, pes_larg=larg,
                               tronco_incl=g2, com_offset=g4, frac_max=frac,
                               aceita=ok, qpos=c.d.qpos.copy()))
    aceitas = [l for l in linhas if l["aceita"]]
    print(f"\n{len(aceitas)} de {len(linhas)} poses aceitas")
    np.savez(args.saida, hash_mjcf=c.hash,
             **{k: np.array([l[k] for l in linhas]) for k in
                ("topo", "meia", "residuo", "residuo_pe", "pelve_z", "pelve_incl", "folga_min",
                 "pes_larg", "tronco_incl",
                 "com_offset", "frac_max", "aceita", "qpos")})
    print(f"-> {args.saida}")


if __name__ == "__main__":
    main()
