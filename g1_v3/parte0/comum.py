"""Utilitários da Parte 0 (plano B4; S §9.2, §9.3, §8.2; S §16 Parte 0).

`le_csv` lê um play; `fk` é a cinemática sem física (PD-3) do g1.xml do mjlab; `custos_por_s` e
`cercas_por_passo` medem os reguladores e as cercas; `trajetoria_pegar` e `trajetoria_botar` são os
roteiros do item 1 da B5. Origem: `g1_limpo/ik/cercas_nos_plays.py` (copiado; nada importado de
`g1_limpo`). Módulo da Parte 0: pode importar `g1_v3.knobs`, `g1_v3.alvos` e `g1_v3.potencial`.
"""
import csv
import functools
import importlib.util
import re
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial import ConvexHull

from g1_v3 import alvos as A
from g1_v3.knobs import ATIVO, JUNTAS, caminho

DT = ATIVO.contrato.dt_fisica * ATIVO.contrato.decimacao  # 0,02 s: passo da política
MASSA_CAIXA = 5.0  # S §9.3: caixa de 5 kg nas medidas do CoM
MEIA_LAJE = 0.02  # laje_z é o centro da laje; topo = laje_z + 0,02
A_CAIXA = 0.10  # meia aresta da caixa (S §8.2)
TOPO_PEGAR, TOPO_BOTAR = 0.30, 0.15  # lajes do roteiro (B5, item 1)
CONTATO_M = 0.02  # pé em contato: site a menos de 2 cm da altura de pé


def le_csv(caminho_csv) -> dict[str, np.ndarray]:
    """Coluna numérica vira float; coluna de texto (`fase`, `elo`, `ckpt`) vira str."""
    with open(caminho_csv, newline="") as fh:
        cab, *corpo = list(csv.reader(fh))
    out = {}
    for j, nome in enumerate(cab):
        col = [linha[j] for linha in corpo]
        try:
            out[nome] = np.array(col, dtype=float)
        except ValueError:
            out[nome] = np.array(col, dtype=str)
    return out


@functools.lru_cache(maxsize=1)
def _modelo() -> mujoco.MjModel:
    raiz = Path(importlib.util.find_spec("mjlab").submodule_search_locations[0])
    return mujoco.MjModel.from_xml_path(str(raiz / "asset_zoo/robots/unitree_g1/xmls/g1.xml"))


def fk(q, raiz=None) -> dict[str, np.ndarray]:
    """`raiz` (T,7) = pose da pelve [x,y,z,qw,qx,qy,qz]. Devolve pes, palmas, pelve, tronco (rad),
    pe_pos, pe_rot, com, por passo."""
    m = _modelo()
    d = mujoco.MjData(m)
    q = np.atleast_2d(np.asarray(q, dtype=float))
    raiz = (np.tile(m.qpos0[:7], (len(q), 1)) if raiz is None
            else np.atleast_2d(np.asarray(raiz, dtype=float)))
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JUNTAS]
    sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
           for n in ("left_foot", "right_foot", "left_palm", "right_palm")]
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
           for n in ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link")]
    out = {k: [] for k in ("pes", "palmas", "pelve", "tronco", "pe_pos", "pe_rot", "com")}
    for i in range(len(q)):
        d.qpos[:7] = raiz[i]
        d.qpos[qadr] = q[i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        out["pes"].append(d.site_xpos[sid[:2]])
        out["palmas"].append(d.site_xpos[sid[2:]])
        out["pelve"].append(d.xpos[bid[0]].copy())  # índice inteiro devolve view: copiar
        out["tronco"].append(np.arccos(np.clip(d.xmat[bid[1]].reshape(3, 3)[2, 2], -1, 1)))
        out["pe_pos"].append(d.xpos[bid[2:]])
        out["pe_rot"].append(d.xmat[bid[2:]].reshape(2, 3, 3))
        out["com"].append(d.subtree_com[bid[0]].copy())
    return {k: np.array(v) for k, v in out.items()}


def _colunas(dados, prefixo):
    """(T,29) na ordem de JUNTAS, ou None: plays antigos não gravaram `a_*` nem `alvo_pd_*`."""
    chaves = [prefixo + j for j in JUNTAS]
    return np.stack([dados[c] for c in chaves], 1) if all(c in dados for c in chaves) else None


def _por_junta(tabela, nomes):
    """Valor da primeira regex da tabela que casa (fullmatch) com cada nome de junta."""
    return np.array([next(v for rx, v in tabela if re.fullmatch(rx, n)) for n in nomes], dtype=float)


def _ganhos():
    # ganho do PLAY (mjlab), não o do treino: os alvo_pd dos CSV saíram de uma política com estes ganhos
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION as GA
    atu = GA.actuators
    kp = np.array([_atuador(atu, n).stiffness for n in JUNTAS], dtype=float)
    kd = np.array([_atuador(atu, n).damping for n in JUNTAS], dtype=float)
    return kp, kd


def _atuador(atuadores, nome):
    return next(a for a in atuadores if any(re.fullmatch(rx, nome) for rx in a.target_names_expr))


def _dt(dados) -> float:
    return float(np.median(np.diff(dados["t"])))


def custos_por_s(dados, k=ATIVO) -> dict[str, float]:
    """Valor por segundo no retorno do mjlab (S §9.2, pesos da v3; negativo = custo). NaN sem `a_*` ou `alvo_pd_*`.

    ⚠ O mjlab soma valor·peso·dt por passo (`scale_by_dt`), então o retorno por segundo é peso·média(valor):
    soma por passo ÷ número de passos, e não ÷ duração (isso daria 1/dt = 50 vezes mais).
    """
    r = k.recompensa
    q = _colunas(dados, "q_")
    dt = _dt(dados)
    dur = len(q)  # passos: retorno por s = peso·média(valor) (ver a docstring)
    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    out = {"taxa_acao": np.nan, "vel_junta": r.vel_junta * np.sum(qd ** 2) / dur,
           "acel_junta": r.acel_junta * np.sum(qdd ** 2) / dur, "energia": np.nan}
    a = _colunas(dados, "a_")
    if a is not None:
        out["taxa_acao"] = r.taxa_acao * np.sum(np.diff(a, axis=0) ** 2) / dur
    alvo = _colunas(dados, "alvo_pd_")
    if alvo is not None:
        kp, kd = _ganhos()
        out["energia"] = r.energia * np.sum(np.abs((kp * (alvo - q) - kd * qd) * qd)) / dur
    out["total"] = float(np.nansum(list(out.values())))
    return out


def cercas_por_passo(dados, ref, k=ATIVO) -> dict[str, np.ndarray]:
    """Sete cercas da S §9.3 por passo (custo da dobradiça; graus, m e fração de τ_max). Sola e recuo
    só no PEGAR e no BOTAR; fora da pega, a referência é a pose de pé."""
    r = k.recompensa
    q = _colunas(dados, "q_")
    raiz = np.stack([dados[c] for c in ("raiz_x", "raiz_y", "raiz_z",
                                        "raiz_qw", "raiz_qx", "raiz_qy", "raiz_qz")], 1)
    f = fk(q, raiz)
    T = len(q)
    pega = np.isin(dados["fase"], ["pegar", "botar"])
    topo = dados["laje_z"] + MEIA_LAJE
    h = torch.as_tensor(topo, dtype=torch.float64)
    a = torch.full_like(h, A_CAIXA)
    pelve_ik = np.where(pega, ref.pelve_z(h, a).numpy(), ref.pelve_de_pe)
    tronco_ik = np.where(pega, np.degrees(ref.tronco(h, a).numpy()), 0.0)
    recuo_ik = ref.recuo(h, a).numpy()

    incl = np.degrees(np.arccos(np.clip(f["pe_rot"][:, :, 2, 2], -1, 1)))
    contato = f["pes"][:, :, 2] < ref.site_pe_de_pe + CONTATO_M
    sola = (np.maximum(0, incl - r.lim_sola_graus) * contato).sum(1) * pega

    iq = [JUNTAS.index("left_hip_yaw_joint"), JUNTAS.index("right_hip_yaw_joint")]
    giro_q = np.degrees(np.stack([-q[:, iq[0]], q[:, iq[1]]], 1))
    giro = np.maximum(0, giro_q - r.lim_giro_graus).sum(1)

    tronco = np.maximum(0, np.degrees(f["tronco"]) - (tronco_ik + r.lim_tronco_graus))

    caixa = np.stack([dados[c] for c in ("caixa_x", "caixa_y", "caixa_z")], 1)
    solta = caixa[:, 2] - A_CAIXA > topo + 0.01
    presa = solta if "fn_tot" not in dados else solta & (dados["fn_tot"] > 20.0)
    m = _modelo()
    m_robo = m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
    mc = np.where(presa, MASSA_CAIXA, 0.0)
    com = (m_robo * f["com"] + mc[:, None] * caixa) / (m_robo + mc)[:, None]
    cantos = ref.cantos_sola.numpy()
    com_cerca = np.zeros(T)
    for i in np.flatnonzero(contato.all(1)):  # margem ao casco das 8 quinas das solas (PD-35)
        pts = np.concatenate([f["pe_pos"][i, p] + cantos @ f["pe_rot"][i, p].T for p in (0, 1)])[:, :2]
        eq = ConvexHull(pts).equations
        com_cerca[i] = max(0.0, r.lim_com - np.min(-(eq[:, :2] @ com[i, :2] + eq[:, 2])))

    tor = np.full(T, np.nan)
    alvo = _colunas(dados, "alvo_pd_")
    if alvo is not None:
        kp, kd = _ganhos()
        tmax = _por_junta(r.tau_max, JUNTAS)
        # corte em τ_max: o g1.xml do mjlab não tem forcerange; τ_max é o effort_limit do mjlab
        tau = np.clip(kp * (alvo - q) - kd * np.gradient(q, _dt(dados), axis=0), -tmax, tmax)
        frac = np.abs(tau) / tmax
        idx = {n: i for i, n in enumerate(JUNTAS)}
        pares = [(idx[p], idx[s]) for p, s in r.pares_torque]
        sozinha = np.ones(len(JUNTAS), bool)
        for i, j in pares:
            sozinha[[i, j]] = False
        cu = np.maximum(0, frac[:, sozinha] - r.lim_torque) / r.banda_torque
        cp = [np.maximum(0, (np.abs(tau[:, i]) + np.abs(tau[:, j])) / r.tau_par - r.lim_torque)
              / r.banda_torque for i, j in pares]
        tor = cu.sum(1) + np.sum(cp, 0)

    pelve = np.maximum(0, np.abs(f["pelve"][:, 2] - pelve_ik) - r.lim_pelve)
    dist = np.linalg.norm(caixa[:, :2] - f["pelve"][:, :2], axis=1)
    recuo = np.where(pega, np.maximum(0, np.abs(dist - recuo_ik) - r.lim_recuo), 0.0)
    return {"sola": sola, "giro": giro, "tronco": tronco, "com": com_cerca,
            "torque": tor, "pelve": pelve, "recuo": recuo}


# --- trajetórias roteirizadas (B5, item 1) ---
# Mundo: chão em z 0, base na origem, rumo 0; a caixa não gira. Estado = palmas (2,3), caixa (3,),
# pelve (1,), tronco (1,), com smoothstep entre os extremos de cada trecho.

@functools.lru_cache(maxsize=1)
def _ref() -> A.Referencias:
    return A.Referencias(str(caminho(ATIVO.contrato.referencias)), "cpu")


def _t(v) -> torch.Tensor:
    return torch.tensor([v], dtype=torch.float64)


def _estado(palmas, caixa, pelve, tronco) -> np.ndarray:
    return np.concatenate([np.ravel(palmas), caixa, [pelve, tronco]])


def _caixa_alvo(indice, caixa_pega, soltura, a, ref) -> np.ndarray:
    z = torch.zeros(1, 3, dtype=torch.float64)
    out = A.caixa_alvo_w(torch.tensor([indice]), torch.as_tensor(caixa_pega, dtype=torch.float64)[None],
                         torch.as_tensor(soltura, dtype=torch.float64)[None], z,
                         _t(0.0), _t(0.0), _t(a), ref, ATIVO.alvos)
    return out[0].numpy()


def _palmas(indice, caixa, a) -> np.ndarray:
    return caixa + A.palmas_alvo_caixa(torch.tensor([indice]), _t(a), ATIVO.alvos)[0].numpy()


def _pose_de_pe(ref) -> np.ndarray:
    return ref.palmas_de_pe_b.numpy() + [0.0, 0.0, ref.pelve_de_pe]


def _roteiro(inicio: np.ndarray, trechos) -> dict:
    """`trechos`: (nome, duração s, índice do alvo, estado final). Cada trecho vira `round(dur/DT)`
    passos (no mínimo 1: o contato em P2 dura um passo). Interpolação smoothstep 3s²−2s³ (s = j/n)."""
    estados, indices, nomes = [inicio], [trechos[0][2]], [trechos[0][0]]
    atual = inicio
    for nome, dur, idx, fim in trechos:
        n = max(1, round(dur / DT))
        for j in range(1, n + 1):
            s = j / n
            # velocidade zero nas pontas: a reta fazia o BOTAR pousar a 0,6 m/s (S §9.1 fator_pouso)
            estados.append(atual + (fim - atual) * (3 * s * s - 2 * s ** 3))
            indices.append(idx)
            nomes.append(nome)
        atual = fim
    e = np.array(estados)
    return {"t": np.arange(len(e)) * DT, "dt": DT, "indice": np.array(indices),
            "trecho": np.array(nomes), "palmas": e[:, :6].reshape(-1, 2, 3),
            "caixa": e[:, 6:9], "pelve": e[:, 9], "tronco": e[:, 10]}


def trajetoria_pegar() -> dict:
    """Roteiro do PEGAR a 50 Hz (laje 0,30 m, a 0,10). Dict: t, dt, indice (0..8 em ALVOS), trecho,
    palmas (N,2,3), caixa (N,3), pelve, tronco. Trechos: P0 1,0 s; P1 0,5 s; P2 1 passo; P3 0,5 s; P4 1,0 s + 0,5 s."""
    ref, a = _ref(), A_CAIXA
    recuo = float(ref.recuo(_t(TOPO_PEGAR), _t(a))[0])
    pelve = float(ref.pelve_z(_t(TOPO_PEGAR), _t(a))[0])
    tronco = float(ref.tronco(_t(TOPO_PEGAR), _t(a))[0])
    caixa0 = np.array([recuo, 0.0, TOPO_PEGAR + a])
    caixa1 = _caixa_alvo(A.P3, caixa0, np.zeros(3), a, ref)
    caixa4 = _caixa_alvo(A.P4, np.zeros(3), np.zeros(3), a, ref)
    p1 = _palmas(A.P1, caixa0, a)
    # P4 é a pose de pé com caixa (S §8.2 "P4 peito"): pelve e tronco de pé, e não os da pega (tronco 51°)
    pelve4 = float(ref.pelve_com_caixa(_t(a))[0])
    inicio = _estado(_pose_de_pe(ref), caixa0, ref.pelve_de_pe, 0.0)
    return _roteiro(inicio, [
        ("ate_P0", 1.0, A.P0, _estado(_palmas(A.P0, caixa0, a), caixa0, pelve, tronco)),
        ("P0_P1", 0.5, A.P1, _estado(p1, caixa0, pelve, tronco)),
        ("P2", DT, A.P2, _estado(p1, caixa0, pelve, tronco)),
        ("P3", 0.5, A.P3, _estado(_palmas(A.P1, caixa1, a), caixa1, pelve, tronco)),
        ("P4", 1.0, A.P4, _estado(_palmas(A.P1, caixa4, a), caixa4, pelve4, 0.0)),
        ("P4_espera", ATIVO.alvos.tempo_p4, A.P4, _estado(_palmas(A.P1, caixa4, a), caixa4, pelve4, 0.0)),
    ])


def trajetoria_botar() -> dict:
    """Roteiro do BOTAR (laje 0,15 m), começa no transporte; mesmo formato. Permanências do knobs."""
    ref, a, k = _ref(), A_CAIXA, ATIVO.alvos
    caixa4 = _caixa_alvo(A.P4, np.zeros(3), np.zeros(3), a, ref)
    recuo = float(ref.recuo(_t(TOPO_BOTAR), _t(a))[0])
    pelve = float(ref.pelve_z(_t(TOPO_BOTAR), _t(a))[0])
    tronco = float(ref.tronco(_t(TOPO_BOTAR), _t(a))[0])
    alvo_b, centro_b = A.laje_destino(_t(recuo), _t(0.0), _t(0.0), a, ATIVO.cena.destino_borda)
    soltura = np.array([centro_b[0, 0].item(), centro_b[0, 1].item(), TOPO_BOTAR + a])
    caixa_b1 = _caixa_alvo(A.B1, np.zeros(3), soltura, a, ref)
    p1_b1, p1_b2 = _palmas(A.P1, caixa_b1, a), _palmas(A.P1, soltura, a)
    b3, pe = _palmas(A.B3, soltura, a), _pose_de_pe(ref)
    # o BOTAR começa no fim do P4 da pega: pose de pé com caixa (tronco 0), e não a pose da pega (tronco 51°)
    inicio = _estado(_palmas(A.P1, caixa4, a), caixa4, float(ref.pelve_com_caixa(_t(a))[0]), 0.0)
    return _roteiro(inicio, [
        ("ate_B1", 1.0, A.B1, _estado(p1_b1, caixa_b1, pelve, tronco)),
        ("B1_B2", 0.5, A.B2, _estado(p1_b2, soltura, pelve, tronco)),
        ("B2_espera", k.tempo_b2, A.B2, _estado(p1_b2, soltura, pelve, tronco)),
        ("B3", 0.5, A.B3, _estado(b3, soltura, pelve, tronco)),
        ("B3_espera", k.tempo_b3, A.B3, _estado(b3, soltura, pelve, tronco)),
        ("B4", 1.0, A.B4, _estado(pe, soltura, ref.pelve_de_pe, 0.0)),
        ("B4_espera", k.tempo_b4, A.B4, _estado(pe, soltura, ref.pelve_de_pe, 0.0)),
    ])
