"""Item 15 da Parte 0 (S §8.2, §11.5, §16 item 15; plano B5, PD-34).

Propõe os pesos do `aperto` e da `sacudida`:
1. Derivada de `custo_aperto` em 117,5 N e em F_ref (autograd; diferença central como conferência);
   critério: derivada em 117,5 N ≥ 20% da derivada em F_ref.
2. Ganho acumulado de afrouxar de 117,5 N para 49 N em 20 s, com w = 1 e por w (pior caso: solta de uma vez).
3. Peso máximo w ≤ 0,5 × perda_caixa (em retorno) ÷ (ganho de afrouxar em 20 s, em retorno, por w); w proposto.
4. `sacudida` pela regra da PD-34: custo no p50 = 10% do ganho máximo do rastreio linear por s.

Unidade: retorno do mjlab (`scale_by_dt`, × dt). Um termo de valor x e peso w soma w·x·dt por passo.
"por passo" = retorno ÷ dt (o número que vai ao knobs para o valor do termo); para o peso, o próprio w.

Rodar da raiz do worktree: `python -m g1_v3.parte0.item15_aperto`. Grava `saida/item15_aperto.txt`.
"""
import math
import re
from pathlib import Path

import numpy as np
import torch

from g1_v3 import potencial as P
from g1_v3.knobs import ATIVO, caminho
from g1_v3.parte0 import comum as C

R = ATIVO.recompensa
DT = C.DT
F_OPERACAO = 117.5  # N por face, S §8.2 (110–125 N no model_24800)
F_SOLTA = 49.0  # N, S §8.2 (força que segura 5 kg com μ 0,5)
JANELA = 20.0  # s, S §16 item 15
FRAC_PERDA = 0.5  # S §16 item 15: w ≤ 0,5 × perda ÷ ganho
FRAC_DERIVADA = 0.20  # S §8.2: derivada em 117,5 N ≥ 20% da de F_ref
FRAC_SACUDIDA = 0.10  # PD-34: custo no p50 = 10% do ganho máximo do rastreio linear por s
NORM_LIN, NORM_ANG = 2.9, 17.0  # m/s², rad/s²: p50 do CARREGAR do 23250 (S §11.5, PD-34)
SAIDA = Path(__file__).resolve().parent / "saida" / "item15_aperto.txt"
ITEM3 = SAIDA.with_name("item3_escada.txt")

linhas_saida = []


def emite(texto=""):
    linhas_saida.append(texto)


def f4(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def linha(rotulo, por_passo, retorno, n=4):
    emite(f"  {rotulo:<56} {f4(por_passo, n):>12} {f4(retorno, n):>12}")


def criterio(nome, ok, detalhe):
    emite(f"  critério {nome}: {'OK' if ok else 'FALHA'} ({detalhe})")


def custo_vec(F):
    """custo_aperto(F, F, F_ref) para um escalar ou um array (F_esq = F_dir: as duas faces juntas)."""
    t = torch.as_tensor(np.asarray(F, dtype=np.float64))
    return P.custo_aperto(t, t, R.F_ref).numpy()


def custo(F):
    return float(custo_vec(F))


def derivada(F0):
    """d custo_aperto(F, F, F_ref) / dF em F0, por autograd (float64)."""
    F = torch.tensor(float(F0), dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(P.custo_aperto(F, F, R.F_ref), F)
    return float(g)


def derivada_central(F0, h=1e-3):
    return (custo(F0 + h) - custo(F0 - h)) / (2 * h)


def ganho_rampa_20s():
    """Ganho por unidade de w se a força descer linear de 117,5 N a 49 N em 20 s (conferência, sem critério)."""
    t = np.linspace(0.0, JANELA, 2001)
    F = F_OPERACAO + (F_SOLTA - F_OPERACAO) * t / JANELA
    ganho_s = custo(F_OPERACAO) - custo_vec(F)  # retorno por s, por w, em relação a segurar 117,5 N
    return float(np.sum((ganho_s[1:] + ganho_s[:-1]) / 2 * np.diff(t)))


def perda_do_item3():
    """perda_caixa por passo (knob) lida da saída do item 3; o retorno é o knob × dt."""
    m = re.search(r"perda_caixa = .*?por passo \(knob\) = retorno/dt = ([0-9.]+)", ITEM3.read_text(encoding="utf-8"))
    if m is None:
        raise SystemExit(f"perda_caixa não achada em {ITEM3}")
    return float(m.group(1))


def floor_1_sig(x):
    """Arredonda para baixo com 1 algarismo significativo (o peso proposto não passa do máximo)."""
    e = 10.0 ** math.floor(math.log10(x))
    return math.floor(x / e + 1e-9) * e


def custo_sacudida(a_lin, alfa):
    """PD-34: (‖a_lin‖/2,9)² + (‖α‖/17)², por unidade de w, por passo (valor do termo)."""
    return (a_lin / NORM_LIN) ** 2 + (alfa / NORM_ANG) ** 2


def qmul(a, b):
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)


def p50_carregar_23250():
    """Conferência (sem critério): p50 de ‖a_lin‖ e ‖α‖ da caixa no CARREGAR do 23250_topo004."""
    d = C.le_csv(caminho("23250_topo004.csv"))
    m = d["fase"] == "carregar"
    dt = float(np.median(np.diff(d["t"][m])))
    p = np.stack([d["caixa_x"], d["caixa_y"], d["caixa_z"]], axis=1)[m]
    q = np.stack([d["caixa_qw"], d["caixa_qx"], d["caixa_qy"], d["caixa_qz"]], axis=1)[m]
    a = np.gradient(np.gradient(p, dt, axis=0), dt, axis=0)
    rel = qmul(q[1:], q[:-1] * np.array([1.0, -1.0, -1.0, -1.0]))  # q_{k+1} ⊗ conj(q_k): giro no mundo
    rel = rel * np.where(rel[:, :1] < 0, -1.0, 1.0)
    omega = 2 * rel[:, 1:] / dt
    alfa = np.gradient(omega, dt, axis=0)
    return float(np.nanmedian(np.linalg.norm(a, axis=1))), float(np.nanmedian(np.linalg.norm(alfa, axis=1)))


def main():
    fref = R.F_ref
    emite("Item 15 da Parte 0: peso do aperto (r_aperto) e da sacudida (S §8.2, §11.5, §16 item 15; PD-34)")
    emite(f"unidade: retorno do mjlab (× dt = {DT} s). Por passo = retorno ÷ dt; para o peso w, o próprio w.")
    emite(f"ponto de operação: {F_OPERACAO} N por face (model_24800); F_ref = {fref} N (knobs)")
    emite()

    # --- 1. derivada ---
    d_op, d_ref = derivada(F_OPERACAO), derivada(fref)
    razao = d_op / d_ref
    emite("1. Derivada de custo_aperto(F, F, F_ref) por N (autograd; diferença central como conferência)")
    emite(f"  F = {F_OPERACAO} N: custo = {custo(F_OPERACAO):.4f}, derivada = {d_op:.6f} /N "
          f"(central {derivada_central(F_OPERACAO):.6f})")
    emite(f"  F = {fref} N (F_ref): custo = {custo(fref):.4f}, derivada = {d_ref:.6f} /N "
          f"(central {derivada_central(fref):.6f})")
    emite(f"  razão das derivadas = {razao * 100:.1f}%")
    criterio("(a) derivada em 117,5 N ≥ 20% da de F_ref", razao >= FRAC_DERIVADA, f"{razao * 100:.1f}%")
    emite()

    # --- 2. ganho de afrouxar ---
    c_op, c_sol = custo(F_OPERACAO), custo(F_SOLTA)
    ganho_s = c_op - c_sol  # retorno por s com w = 1 (valor por s = w·Δcusto)
    ganho_20 = JANELA * ganho_s  # retorno em 20 s com w = 1: solta de uma vez e segura 49 N
    ganho_rampa = ganho_rampa_20s()
    emite("2. Ganho acumulado de afrouxar de 117,5 N para 49 N (w = 1; por w = o mesmo número × w)")
    emite(f"  {'':<56} {'por passo':>12} {'retorno':>12}")
    linha(f"custo(117,5 N) − custo(49 N), por s (w = 1)", ganho_s, ganho_s * DT)
    linha("ganho em 20 s, w = 1 (solta de uma vez, segura 49 N)", None, ganho_20)
    linha("info: ganho em 20 s, w = 1, descida linear em 20 s", None, ganho_rampa)
    emite()

    # --- 3. peso máximo ---
    perda_passo = perda_do_item3()
    perda_ret = perda_passo * DT
    w_max = FRAC_PERDA * perda_ret / ganho_20
    w = floor_1_sig(w_max)
    ganho_w = w * ganho_20
    emite("3. Peso máximo: w ≤ 0,5 × perda_caixa (retorno) ÷ ganho de afrouxar em 20 s (retorno, por w)")
    emite(f"  {'':<56} {'por passo':>12} {'retorno':>12}")
    linha("perda_caixa (item 3; por passo = knob, retorno = knob × dt)", perda_passo, perda_ret)
    linha("w máximo = 0,5 × perda ÷ ganho(w = 1)", w_max, FRAC_PERDA * perda_ret)
    linha("w proposto (1 algarismo, para baixo): ganho em 20 s", w, ganho_w)
    hold = w * c_op * JANELA
    linha("info: custo de segurar 117,5 N por 20 s com w proposto", None, -hold)
    criterio("(c) w proposto ≤ w máximo", w <= w_max, f"w = {w}, máximo = {w_max:.5f}")
    emite()

    # --- 4. sacudida (PD-34) ---
    a_med, alfa_med = p50_carregar_23250()
    c_p50 = custo_sacudida(NORM_LIN, NORM_ANG)  # p50 da S §11.5: custo por unidade de w
    gan_max_s = R.rastreio_linear * 1.0  # rastreio_normalizado ≤ 1 (S §19); ganho máximo por s
    w_sac = FRAC_SACUDIDA * gan_max_s / c_p50
    ratio = w_sac * c_p50 / gan_max_s
    emite("4. Sacudida (PD-34): custo no p50 com w = 10% do ganho máximo do rastreio linear por s")
    emite(f"  {'':<56} {'por passo':>12} {'retorno':>12}")
    linha("custo no p50 por unidade de w: (2,9/2,9)² + (17/17)²", c_p50, c_p50 * DT, 3)
    linha("ganho máximo do rastreio linear por s (w = knob × 1)", gan_max_s, gan_max_s * DT, 3)
    linha("w sacudida (PD-34)", w_sac, w_sac * c_p50 * DT, 3)
    criterio("(d) custo no p50 com w = 10% do ganho máximo por s", abs(ratio - FRAC_SACUDIDA) < 1e-9,
             f"{ratio * 100:.1f}%")
    emite(f"  conferência (sem critério): p50 medido no CARREGAR do 23250_topo004: "
          f"‖a_lin‖ = {a_med:.2f} m/s² (spec {NORM_LIN}), ‖α‖ = {alfa_med:.2f} rad/s² (spec {NORM_ANG})")
    c_med = custo_sacudida(a_med, alfa_med)
    w_alt = FRAC_SACUDIDA * gan_max_s / c_med
    emite(f"  com o p50 medido: custo no p50 = {c_med:.3f} por w; w daria {w_alt:.4f} (não entra no critério)")
    emite()

    emite("Propostas para o knobs (Recompensa; hoje None): aperto = "
          f"{w}, sacudida = {w_sac:g}. O termo paga −w·custo; o H2 multiplica por d/5 (fator_por_degrau).")
    emite()
    emite("Suposições")
    emite("  1. F_esq = F_dir = F no ponto de operação; a derivada é d/dF de custo_aperto(F, F, F_ref).")
    emite("  2. Ganho de afrouxar = pior caso: a força cai de 117,5 N para 49 N de uma vez e fica 20 s.")
    emite(f"     A descida linear em 20 s (info) dá {100 * ganho_rampa / ganho_20:.0f}% do ganho; o critério usa o pior caso.")
    emite("  3. perda_caixa = 1,2 × (2) do item 3, lida de saida/item3_escada.txt; retorno = knob × dt.")
    emite("  4. Ganho máximo do rastreio linear por s = rastreio_linear (1,0) × 1 (rastreio_normalizado ≤ 1).")
    emite("  5. A sacudida usa 2,9 e 17 (p50 da S §11.5) como normas no p50; o custo no p50 é 2 (teste da H2).")
    emite("  6. w proposto arredondado para baixo com 1 algarismo; a sacudida fica no valor exato de PD-34.")
    emite("  7. Os pesos são os do topo (degrau 5, d = 5); o degrau 0 multiplica por d/5 no H2, não aqui.")
    emite()
    emite("Dúvidas")
    emite("  a. Ganho de afrouxar: degrau (pior caso, usado) ou rampa em 20 s (info)? A diferença é 2,6×.")
    emite(f"  b. Segurar 117,5 N por 20 s com w = {w} custa {hold:.4f} em retorno, acima da perda "
          f"({perda_ret:.4f}). O critério (c) não olha esse custo; confirmar se é aceitável.")
    emite("  c. 'Ganho máximo do rastreio linear por s' = 1,0 (peso do knob × 1). Confirmar essa leitura.")
    emite(f"  d. O p50 da spec (2,9 e 17) não reproduz no 23250_topo004 (medido: {a_med:.2f} e {alfa_med:.2f}, "
          f"por diferença finita da posição e do quaternion). Com o medido, w = {w_alt:.4f}, não {w_sac:g}. "
          "Confirmar qual p50 vale.")
    emite()


if __name__ == "__main__":
    main()
    emite(f"Linhas deste script: {len(Path(__file__).read_text(encoding='utf-8').splitlines())}")
    texto = "\n".join(linhas_saida)
    print(texto)
    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(texto + "\n", encoding="utf-8")
