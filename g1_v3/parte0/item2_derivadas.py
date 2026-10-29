"""Item 2 da Parte 0 (S §16, item 2; plano B5): derivadas da proximidade e das rampas.

Confere, nos roteiros do item 1 (`comum.trajetoria_pegar` e `comum.trajetoria_botar`):
(a) dΦ/dd no estado inicial de cada alvo, com d = σ e Φ = nucleo(d, σ) · Πg (plano §6.4, `progresso`);
(b) d(rampa)/dx em x_lim + Δ/2 para cada rampa da §6.2 `Alvos` (S §9.1, PD-23).
Critério: toda derivada ≠ 0. A derivada é por autograd do torch em float64; a diferença finita
central sai ao lado só como conferência. Imprime a tabela dos Δ da PD-23 para o dono.
Roda com `python -m g1_v3.parte0.item2_derivadas` na raiz do repo. Sem simulador.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

from g1_v3 import alvos as A
from g1_v3 import potencial as P
from g1_v3.knobs import ATIVO
from g1_v3.parte0 import comum as C

K = ATIVO.alvos
A_CAIXA = C.A_CAIXA  # meia aresta do roteiro (S §8.2)
SAIDA = Path(__file__).resolve().parent / "saida" / "item2_derivadas.txt"
# σ de alvo métrico tem piso de 1 cm (plano §6.4: "distância atual (mínimo 1 cm)"). B4 é adimensional: sem piso.
PISO_M = 0.01
ALVOS = [("P0", A.P0), ("P1", A.P1), ("P2", A.P2), ("P3", A.P3), ("P4", A.P4),
         ("B1", A.B1), ("B2", A.B2), ("B3", A.B3), ("B4", A.B4)]


def _t(v) -> torch.Tensor:
    return torch.tensor([v], dtype=torch.float64)


def _d0(ind, traj, i, ref) -> float:
    """Distância que a proximidade mede para o alvo `ind` no passo `i` (plano §6.4, `progresso`)."""
    caixa = traj["caixa"][i]
    if ind == A.B4:  # adimensional: √((Δpelve/tol)² + (Δtronco/tol)²), com as tolerâncias do B4
        dp = (traj["pelve"][i] - ref.pelve_de_pe) / K.tol_pelve_b4
        dt = traj["tronco"][i] / math.radians(K.tol_tronco_b4_graus)
        return math.hypot(dp, dt)
    if ind in (A.P0, A.P1, A.P2, A.B3):  # palmas: média das duas distâncias, no frame da caixa
        alvo = caixa + A.palmas_alvo_caixa(torch.tensor([ind]), _t(A_CAIXA), K)[0].numpy()
        return float(np.mean(np.linalg.norm(traj["palmas"][i] - alvo, axis=-1)))
    # P3, P4, B1, B2: distância da caixa ao alvo. A pega é a caixa do início do PEGAR; a soltura, a do fim do BOTAR
    pega, soltura = traj["caixa"][0], traj["caixa"][-1]
    z = torch.zeros(1, 3, dtype=torch.float64)
    alvo = A.caixa_alvo_w(torch.tensor([ind]), torch.as_tensor(pega, dtype=torch.float64)[None],
                          torch.as_tensor(soltura, dtype=torch.float64)[None], z,
                          _t(0.0), _t(0.0), _t(A_CAIXA), ref, K)[0].numpy()
    return float(np.linalg.norm(caixa - alvo))


def _phi(d, sigma):
    # Gates (Πg) = 1 no início: o roteiro não quebra invariante (caixa parada, tronco e recuo na IK).
    return P.phi(P.nucleo(d, sigma), [torch.ones((), dtype=torch.float64)])


def main() -> int:
    ref = C._ref()  # carrega referencias.npz (o mesmo que os roteiros usam)
    pegar, botar = C.trajetoria_pegar(), C.trajetoria_botar()
    linhas, falhas = [], []

    linhas.append("Item 2 da Parte 0 (S §16, item 2; plano B5)")
    linhas.append("")
    linhas.append("A. dΦ/dd no estado inicial de cada alvo (d = σ; Φ = nucleo(d, σ), Πg = 1)")
    linhas.append("   derivada por autograd (float64); FD = diferença finita central, h = 1e-6·σ")
    for nome, ind in ALVOS:
        traj, roteiro = (pegar, "PEGAR") if nome[0] == "P" else (botar, "BOTAR")
        i = int(np.flatnonzero(traj["indice"] == ind)[0])  # primeiro passo em que o alvo aparece
        d0 = _d0(ind, traj, i, ref)
        sigma = max(d0, 0.0 if nome == "B4" else PISO_M)
        if sigma <= 0.0:
            falhas.append(f"dΦ/dd {nome}")
            linhas.append(f"{nome:<3} {roteiro:<6} d0={d0:.4f}  σ=0  FALHA (σ nulo: sem derivada)")
            continue
        d = torch.tensor(sigma, dtype=torch.float64, requires_grad=True)
        auto, = torch.autograd.grad(_phi(d, sigma), d)
        h = 1e-6 * sigma
        fd = (_phi(torch.tensor(sigma + h, dtype=torch.float64), sigma).item()
              - _phi(torch.tensor(sigma - h, dtype=torch.float64), sigma).item()) / (2 * h)
        der = auto.item()
        ok = der != 0.0 and math.isfinite(der)
        if not ok:
            falhas.append(f"dΦ/dd {nome}")
        unid = "" if nome != "B4" else " (adim.)"
        linhas.append(f"{nome:<3} {roteiro:<6} d0={d0:.4f}  σ={sigma:.4f}{unid}  dΦ/dd={der:+.4e}  "
                      f"FD={fd:+.4e}  {'OK' if ok else 'FALHA'} ({der:+.4e})")

    linhas.append("")
    linhas.append("B. d(rampa)/dx em x_lim + Δ/2 (rampa = clip(1 − (x − x_lim)/Δ, 0, 1); esperado −1/Δ)")
    rampas = [(f"P0–P2, a={a:.2f} m", K.desloc_p0p2_frac * a, K.delta_p0p2_frac * a)
              for a in ref.meias.tolist()]  # Δ = a/2 (PD-23); x_lim = a
    rampas += [
        ("P2–B2 inclinação, fase 30°", K.incl_inicio_graus, K.delta_incl_graus),  # spec §8.2: 30° na fase aprender
        ("P2–B2 inclinação, fim 15°", K.incl_fim_graus, K.delta_incl_graus),  # spec §8.2: 15° no fim do treino
        ("B2–B4 deslocamento (m)", K.desloc_b2b4, K.delta_b2b4),
        ("tronco (graus)", K.faixa_tronco_graus, K.delta_tronco_graus),
        ("recuo (m)", K.faixa_recuo, K.delta_recuo),
        ("CoM (m)", K.faixa_com, K.delta_com),
    ]
    for nome, lim, delta in rampas:
        x = torch.tensor(lim + delta / 2, dtype=torch.float64, requires_grad=True)
        der, = torch.autograd.grad(P.rampa(x, lim, delta), x)
        der = der.item()
        ok = der != 0.0 and math.isfinite(der)
        if not ok:
            falhas.append(f"rampa {nome}")
        linhas.append(f"{nome:<27} x_lim={lim:.4f}  Δ={delta:.4f}  d(rampa)/dx={der:+.4f} "
                      f"(−1/Δ={-1 / delta:+.4f})  {'OK' if ok else 'FALHA'}")

    linhas.append("")
    linhas.append("C. Tabela dos Δ da PD-23 (para o dono confirmar ou trocar; lidos do knobs)")
    linhas.append(f"   deslocamento P0–P2: a/2 → {K.delta_p0p2_frac:.1f} × a; com a = "
                  + ", ".join(f"{a:.2f}" for a in ref.meias.tolist()) + " m")
    linhas.append(f"   deslocamento B2–B4: {K.delta_b2b4 * 100:.0f} cm")
    linhas.append(f"   tronco: {K.delta_tronco_graus:.0f}°")
    linhas.append(f"   recuo: {K.delta_recuo * 100:.0f} cm")
    linhas.append(f"   CoM: {K.delta_com * 100:.0f} cm")
    linhas.append(f"   inclinação: {K.delta_incl_graus:.0f}°")
    linhas.append("")
    total = len(ALVOS) + len(rampas)
    linhas.append(f"Resultado: {total - len(falhas)} de {total} derivadas ≠ 0 (9 de Φ, {len(rampas)} de rampa)"
                  + ("" if not falhas else f"; FALHAS: {', '.join(falhas)}"))

    texto = "\n".join(linhas)
    print(texto)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(texto + "\n", encoding="utf-8")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
