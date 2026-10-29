"""Item 8 da Parte 0 (plano B5; S §16 item 8, S §19 conflito 6; PD-16, PD-48).

Passo nos plays, pela FK dos pés (comum.fk, sem física). Toque = pé com altura < 2 cm do chão
(z 0 do mundo). Passo = L da PD-16 em cada toque. Compara com `L_alvo` (potencial.L_alvo) na
velocidade medida de cada play, e confere a PD-16 (passo de lado parado) e a saturação do termo.

O CSV não grava o comando: a velocidade e a direção de cada trecho são as medidas na base (deslocamento
da raiz dividido pelo tempo da fase), e a direção do comando é a dessa velocidade.

Rodar da raiz do worktree:
    python -m g1_v3.parte0.item8_passo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from g1_v3 import potencial as P
from g1_v3.knobs import ATIVO, caminho
from g1_v3.parte0 import comum

ANDAR = ["5000_botar", "5350_botar"]  # fase `andar`, 300 passos, 0,65 m/s
CARREGAR = ["23250_topo004", "24800_mu06_m3", "24800_mu10_m1", "24800_mu10_m5"]  # fase `carregar`, 0,30 m/s
TOQUE_M = 0.02  # "pé a < 2 cm do chão" (B5). ⚠ comum.CONTATO_M é 2 cm acima da altura de pé (−2 mm): 2 mm a mais
SAIDA = Path(__file__).resolve().parent / "saida" / "item8_passo.txt"
LIMITE_L = 0.60  # m: teto de L_alvo do critério (B5)
FOLGA_MAX = 1.2  # L_alvo ≤ 1,2 × o maior passo medido (B5)
NA_D2 = "N/A (passo miúdo, D2)"  # plays de CARREGAR: o critério 1,2 × maior passo não vale (D2, dono 29/10)


def _yaw(q) -> float:
    qw, qx, qy, qz = q
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def passo_L(p_toque, p_apoio, f, c_w, yaw, pes_b) -> float:
    """L da PD-16 num toque (xy). `f` = pé que toca (0 esq., 1 dir.); `c_w` = comando no mundo;
    `yaw` = rumo da pelve. Projeção de (pé que toca − pé de apoio) na direção do comando, menos a
    mesma projeção na pose de pé (`pes_b`, frame da pelve).

    ⚠ O passo de lado recebe a largura da base de graça sem o desconto da pose de pé (PD-16).
    """
    c, s = np.cos(yaw), np.sin(yaw)
    c_b = np.array([c * c_w[0] + s * c_w[1], -s * c_w[0] + c * c_w[1]])  # ĉ_b = R_z(−yaw)·ĉ_w
    de_pe = (pes_b[f, :2] - pes_b[1 - f, :2]) @ c_b
    return float((p_toque - p_apoio) @ c_w - de_pe)


def r_passo(L: float, L_alvo: float) -> float:
    """Forma do termo `passo` (PD-48): min(relu(L)/L_alvo, 1)². Peso e rastreio ficam fora."""
    return min(max(L, 0.0) / L_alvo, 1.0) ** 2


def analisa(nome: str, fase: str, pes_b) -> dict:
    """Toques, passo e fração de voo da `fase` de um play."""
    d = comum.le_csv(caminho(nome + ".csv"))
    raiz = np.stack([d[c] for c in ("raiz_x", "raiz_y", "raiz_z", "raiz_qw", "raiz_qx", "raiz_qy", "raiz_qz")], 1)
    pes = comum.fk(comum._colunas(d, "q_"), raiz)["pes"]  # (T, pé esq./dir., xyz)
    toca = pes[:, :, 2] < TOQUE_M
    idx = np.where(d["fase"] == fase)[0]
    a, b = idx[0], idx[-1]
    dur = d["t"][b] - d["t"][a]
    v_w = np.array([d["raiz_x"][b] - d["raiz_x"][a], d["raiz_y"][b] - d["raiz_y"][a]]) / dur
    v = float(np.linalg.norm(v_w))
    c_w = v_w / v
    L = [passo_L(pes[i, f, :2], pes[i, 1 - f, :2], f, c_w, _yaw(raiz[i, 3:]), pes_b)
         for i in range(a + 1, b + 1) for f in (0, 1) if toca[i, f] and not toca[i - 1, f]]
    voo = float(np.mean(~toca[a:b + 1].any(axis=1)))
    return {"nome": nome, "v": v, "L": np.array(L), "voo": voo, "cad": len(L) / dur}


def main() -> None:
    k = ATIVO.recompensa
    ref = comum._ref()
    pes_b = ref.pes_de_pe_b.numpy()
    niveis = ATIVO.comando.frente_tetos

    def l_alvo(v: float) -> float:
        return float(P.L_alvo(torch.tensor([v], dtype=torch.float64), k.L_min, k.fator_T)[0])

    crit = []  # (nome, ok)
    linhas = []

    def crit_txt(nome, ok, num):
        crit.append(ok)
        return f"{'OK' if ok else 'FALHA'} ({num})"

    linhas.append(f"item 8, passo: L_min {k.L_min} m, fator_T {k.fator_T}, toque < {TOQUE_M} m do chão")
    linhas.append(f"{'arquivo':<18}{'trecho':<10}{'v med':>7}{'toques':>8}{'L med':>8}{'L max':>8}"
                  f"{'L_alvo':>8}{'voo':>7}{'toq/s':>7}  {'L_alvo ≤ 1,2·max':<30}{'L_alvo ≤ 0,60':<22}")
    plays = [(n, "andar", "ANDAR andando") for n in ANDAR] + [(n, "carregar", "CARREGAR") for n in CARREGAR]
    res = []
    for nome, fase, trecho in plays:
        r = analisa(nome, fase, pes_b)
        r["trecho"] = trecho
        res.append(r)
        L, la = r["L"], l_alvo(r["v"])
        r["carregar"] = fase == "carregar"
        # ⚠ sem toque (o pé nunca sobe 2 cm), o maior passo medido é 0: o critério reprova, não some
        mx = float(L.max()) if len(L) else 0.0
        c1 = NA_D2 if r["carregar"] else crit_txt(nome, la <= FOLGA_MAX * mx, f"{la:.3f} ≤ {FOLGA_MAX * mx:.3f}")
        c2 = crit_txt(nome, la <= LIMITE_L, f"{la:.3f} ≤ {LIMITE_L:.2f}")
        med = f"{L.mean():8.3f}" if len(L) else f"{'-':>8}"
        linhas.append(f"{nome:<18}{trecho:<10}{r['v']:7.3f}{len(L):8d}{med}{mx:8.3f}"
                      f"{la:8.3f}{r['voo']:7.3f}{r['cad']:7.2f}  {c1:<30}{c2:<22}")

    # todos juntos: plays cujo v médio cai no mesmo nível de velocidade (o nível mais perto)
    linhas.append("todos juntos, por nível de velocidade (plays no nível mais perto do v médio):")
    for n_v in niveis:
        grupo = [r for r in res if min(niveis, key=lambda x: abs(x - r["v"])) == n_v]
        if not grupo:
            continue
        L = np.concatenate([r["L"] for r in grupo])
        v = float(np.mean([r["v"] for r in grupo]))
        la, mx = l_alvo(v), float(L.max())
        c1 = NA_D2 if all(r["carregar"] for r in grupo) else crit_txt(
            f"nível {n_v}", la <= FOLGA_MAX * mx, f"{la:.3f} ≤ {FOLGA_MAX * mx:.3f}")
        c2 = crit_txt(f"nível {n_v}", la <= LIMITE_L, f"{la:.3f} ≤ {LIMITE_L:.2f}")
        linhas.append(f"  nível {n_v:<5} plays {len(grupo)}  v {v:.3f}  toques {len(L)}  L med {L.mean():.3f}  "
                      f"L max {mx:.3f}  L_alvo {la:.3f}  {c1:<30}{c2}")

    linhas.append("L_alvo por nível de velocidade (sem CSV nesse nível: o passo não foi medido):")
    linhas.append("  " + "  ".join(f"{n_v}: {l_alvo(n_v):.3f}" for n_v in niveis))

    # passo de lado parado: pés na largura da base, comando lateral; o termo tem que pagar 0
    lado = []
    for f in (0, 1):
        L = passo_L(pes_b[f, :2], pes_b[1 - f, :2], f, np.array([0.0, 1.0]), 0.0, pes_b)
        lado.append((f, L, r_passo(L, l_alvo(0.25))))
    frente = passo_L(pes_b[0, :2], pes_b[1, :2], 0, np.array([1.0, 0.0]), 0.0, pes_b)
    ok_lado = all(abs(L) < 1e-9 and r == 0.0 for _f, L, r in lado)
    linhas.append("passo de lado parado (pés na largura da base, comando lateral (0,1)):")
    for f, L, r in lado:
        linhas.append(f"  toque {'esquerdo' if f == 0 else 'direito'}: L {L:.4f} m, r {r:.4f}")
    linhas.append(f"  comando para a frente na pose de pé: L {frente:.4f} m")
    linhas.append("  " + crit_txt("lado", ok_lado, "paga 0 nos dois pés" if ok_lado else "paga > 0"))

    # saturação: passo maior que L_alvo não paga mais, em todo nível
    sat = []
    for n_v in niveis:
        la = l_alvo(n_v)
        sat.append(all(r_passo(k_ * la, la) == 1.0 for k_ in (1.0, 1.5, 2.0, 3.0)) and r_passo(0.5 * la, la) == 0.25)
    linhas.append("saturação do termo (r = 1 em L ≥ L_alvo, r = 0,25 em L_alvo/2), nos níveis:")
    linhas.append("  " + crit_txt("sat", all(sat), f"{sum(sat)} de {len(sat)} níveis"))

    linhas.append(f"critérios: {sum(crit)} OK, {len(crit) - sum(crit)} FALHA")
    texto = "\n".join(linhas)
    print(texto)
    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
