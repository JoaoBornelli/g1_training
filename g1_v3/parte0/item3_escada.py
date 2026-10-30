"""Item 3 da Parte 0 (S §16 item 3; plano B5, PD-30 a PD-33, PD-36; S §9.2–9.4, §10).

Propõe os pesos e checa cada desigualdade:
1. `B_alvo` por alvo (P0–P4, B1–B4) = 1,5 × custo dos reguladores no trecho até o alvo (PD-31).
2. `w_p` do progresso = 1,5 × o maior custo por s ÷ ganho de Φ por s, nos trechos de viagem do roteiro (PD-32).
3. Troca de alvo (D1, dono 29/10): no passo do avanço o `progresso` paga B_alvo + w_p·(γΦ(s′, i_velho) − Φ(s, i_velho)).
   Checagem: esse passo tem de pagar mais que ficar parado perto do alvo, w_p·(γ−1)·Φ(s, i_velho).
4. Escada da S §10 com os pesos propostos; `perda_caixa` = 1,2 × lado direito de (2); `devolve_b_alvo` (PD-33).
5. Pesos das cercas (PD-30) a partir de `cercas_por_passo` nos PEGAR.

Unidade: retorno do mjlab (`scale_by_dt`). Custo por s = w·média(x) (`custos_por_s`); custo do trecho =
custo por s × duração; bônus de um passo soma B·dt, então B_alvo = 1,5·C/dt. Os lados da escada vão em
retorno (×dt), e o dt cancela onde os dois lados são termos de passo.

Rodar da raiz do worktree: `python -m g1_v3.parte0.item3_escada`. Grava `saida/item3_escada.txt`.
"""
from pathlib import Path

import numpy as np

from g1_v3.knobs import ATIVO, caminho
from g1_v3.parte0 import comum as C
from g1_v3.parte0 import item1_phi as I1
from g1_v3.parte0 import item15_aperto as I15

R = ATIVO.recompensa
DT = C.DT
GAMMA = R.gamma
MARGEM_B = 1.5  # PD-31 (B_alvo) e PD-32 (w_p)
MARGEM_PERDA = 1.2  # PD-33
FRAC_CERCA = 0.5  # PD-30
VEL_BOTAR = 0.25  # m/s, S §7 (pior caso a 0,25 m/s de mão)
EPISODIO = 20.0  # s, S §7 (tempo do episódio para os reguladores da queda)
PHI_MAX = 1.0  # Φ ∈ [0, 1] (S §9.1)
QUEDA = -R.terminacao * DT  # S §10: toda terminação custa −200 × dt
COLUNAS = ("taxa_acao", "vel_junta", "acel_junta", "energia", "total")
CERCAS = {"sola": "cerca_sola", "giro": "cerca_giro", "tronco": "cerca_tronco", "com": "cerca_com",
          "torque": "cerca_torque", "pelve": "cerca_pelve", "recuo": "cerca_recuo"}
UNIDADES = {"sola": "°", "giro": "°", "tronco": "°", "com": "m", "torque": "fração", "pelve": "m", "recuo": "m"}
SAIDA = Path(__file__).resolve().parent / "saida" / "item3_escada.txt"
NOMES = I1.NOMES

# Trechos (B5, tabela por coluna `fase`). PEGAR com a_* e alvo_pd_* completos: os 5 de baixo e o 23250_topo004.
TRECHOS = {
    "pegar": ["23250_k2", "23250_topo004", "24800_botar_t015", "24800_mu06_m3", "24800_mu10_m1", "24800_mu10_m5"],
    "carregar": ["23250_topo004", "24800_mu06_m3", "24800_mu10_m1", "24800_mu10_m5"],
    "botar": ["23250_k2", "24800_botar_t015"],
    "andar": ["5000_botar", "5350_botar"],
    "parado": ["5000_andar_parado", "5350_andar_parado", "5800_andar_parado"],
}
PEGAR = TRECHOS["pegar"]

linhas_saida = []


def emite(texto=""):
    linhas_saida.append(texto)


def f3(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.4f}"


def _carrega(nome, fase):
    """Só as linhas da `fase` do CSV (o play do ANDAR andando traz também PEGAR, BOTAR e espera)."""
    d = C.le_csv(caminho(nome + ".csv"))
    m = d["fase"] == fase
    return {k: v[m] for k, v in d.items()}


def custos_por_arquivo() -> dict:
    """custos_por_s de cada (fase, arquivo). `completo` = taxa_acao e energia existem (a_* e alvo_pd_*)."""
    out = {}
    for fase, arquivos in TRECHOS.items():
        for n in arquivos:
            sub = _carrega(n, fase)
            c = C.custos_por_s(sub)
            out[(fase, n)] = dict(c, linhas=len(sub["t"]),
                                  completo=bool(np.isfinite(c["taxa_acao"]) and np.isfinite(c["energia"])))
    return out


def media_fase(fase, custos) -> tuple:
    """Média dos arquivos completos da fase; sem nenhum completo, média dos parciais (marcada)."""
    lote = [(n, v) for (f, n), v in custos.items() if f == fase]
    comp = [(n, v) for n, v in lote if v["completo"]]
    usados = comp or lote
    med = {k: float(np.mean([v[k] for _, v in usados])) for k in COLUNAS}
    return med, [n for n, _ in usados], bool(comp)


def pernas(tr, phi, fase) -> list:
    """Trechos (runs de mesmo nome) do roteiro. dur = passos × dt: a primeira linha é o estado inicial,
    não um passo. `prev` = Φ da linha anterior ao trecho (None no começo do roteiro)."""
    nomes, idx = tr["trecho"], tr["indice"]
    ini = np.r_[0, np.flatnonzero(nomes[1:] != nomes[:-1]) + 1]
    fim = np.r_[ini[1:] - 1, len(nomes) - 1]
    out = []
    for a, b in zip(ini, fim):
        passos = (b - a + 1) - (1 if a == 0 else 0)
        out.append(dict(fase=fase, nome=str(nomes[a]), alvo=int(idx[a]), espera=str(nomes[a]).endswith("_espera"),
                        dur=passos * DT, phi0=float(phi[a]), phi1=float(phi[b]),
                        prev=float(phi[a - 1]) if a > 0 else None, a=int(a)))
    return out


def phi_velho_no_novo(tr, res, refs, pts, a, src):
    """D1: Φ do alvo VELHO no estado do passo do avanço (s′, linha a). O mesmo potencial de item1.avalia com o
    índice trocado só nesse passo; alvo velho, caixa do alvo e σ vêm de `src` = (trajetória, resultado, linha)."""
    tv, rv, r = src
    idx = np.array(tr["indice"], dtype=int)
    idx[a] = tv["indice"][r]
    alvo = tr["caixa"][res["fim"][res["trecho"]]].copy()
    alvo[a] = tv["caixa"][rv["fim"][rv["trecho"][r]]]
    sig = np.array(res["sigma"], dtype=float)
    sig[a] = rv["sigma"][r]
    return float(I1.avalia(dict(tr, indice=idx), refs, pts, alvo=alvo, sigma=sig)["phi"][a])


def main():
    custos = custos_por_arquivo()
    med_p, usados_p, comp_p = media_fase("pegar", custos)
    med_b, usados_b, comp_b = media_fase("botar", custos)
    c_p, c_b = -med_p["total"], -med_b["total"]  # módulo do custo por s (positivo)

    # --- roteiro do item 1 (reuso por import) ---
    refs = I1.referencias()
    pe, bo = C.trajetoria_pegar(), C.trajetoria_botar()
    for tr in (pe, bo):
        tr["incl"] = np.zeros(len(tr["t"]))
    pts_pe, pts_bo = I1.pontos_de(pe), I1.pontos_de(bo)
    res_pe, res_bo = I1.avalia(pe, refs, pts_pe), I1.avalia(bo, refs, pts_bo)
    phi_pe, phi_bo = res_pe["phi"], res_bo["phi"]
    pl_pe, pl_bo = pernas(pe, phi_pe, "PEGAR"), pernas(bo, phi_bo, "BOTAR")
    pl_bo[0]["prev"] = float(phi_pe[-1])  # entrada no B1: o roteiro do BOTAR começa no fim do P4
    viagem = [p for p in pl_pe + pl_bo if not p["espera"]]
    T_pe = (len(pe["t"]) - 1) * DT  # duração do roteiro do PEGAR (com as esperas)
    for p in viagem:  # Φ do alvo velho no passo do avanço (D1); o B1 vem do fim do PEGAR
        if p["prev"] is None:
            continue
        tr, res, pts = (pe, res_pe, pts_pe) if p["fase"] == "PEGAR" else (bo, res_bo, pts_bo)
        src = (tr, res, p["a"] - 1) if p["a"] > 0 else (pe, res_pe, len(pe["t"]) - 1)
        p["phi2"] = phi_velho_no_novo(tr, res, refs, pts, p["a"], src)

    # --- custos e quocientes por trecho de viagem (PD-32) ---
    for p in viagem:
        p["c"] = c_p if p["fase"] == "PEGAR" else c_b
        p["C"] = p["c"] * p["dur"]
        p["B"] = MARGEM_B * p["C"] / DT
        d_phi = p["phi1"] - p["phi0"]
        p["d_phi"] = d_phi
        # w_p por passo que faz w_p·ΔΦ·dt (retorno do progresso) cobrir C (retorno dos reguladores), sem o 1,5
        p["quoc"] = p["C"] / (d_phi * DT) if d_phi > 1e-9 else np.nan
    quocientes = [p for p in viagem if np.isfinite(p["quoc"])]
    decisivo = max(quocientes, key=lambda p: p["quoc"])
    w_p = MARGEM_B * decisivo["quoc"]

    # --- ganho médio por s do progresso (PD-30): subida de Φ nos trechos de viagem do PEGAR ÷ T_pe ---
    subida_pe = sum(max(p["phi1"] - p["phi0"], 0.0) for p in pl_pe if not p["espera"])
    G = w_p * subida_pe * DT / T_pe  # retorno por s: w_p·ΔΦ·dt somado ÷ T
    # soma literal do progresso no PEGAR (D1): o passo do avanço usa Φ(s′, i_velho); B nas trocas P1–P4
    nx = phi_pe[1:].copy()
    trocas_pe = [p for p in viagem if p["fase"] == "PEGAR" and p["prev"] is not None]
    for p in trocas_pe:
        nx[p["a"] - 1] = p["phi2"]
    soma_w = w_p * float(np.sum(GAMMA * nx - phi_pe[:-1]))
    soma_b = sum(p["B"] for p in trocas_pe)
    lit_pe = (soma_w + soma_b) * DT / T_pe  # retorno/s

    # --- cercas (PD-30) ---
    ref = C._ref()
    acc = {k: [] for k in CERCAS}
    for n in PEGAR:
        r = C.cercas_por_passo(_carrega(n, "pegar"), ref)
        for k in CERCAS:
            acc[k].append(r[k])
    cercas = {}
    for k in CERCAS:
        x = np.concatenate(acc[k])
        fin = x[np.isfinite(x)]
        age = fin[fin > 0]  # D4: média só nos passos em que a cerca age
        media = float(age.mean()) if age.size else np.nan
        cercas[k] = dict(media=media, pct=float((fin > 0).mean()) if fin.size else np.nan,
                         nans=int((~np.isfinite(x)).sum()), peso=(-FRAC_CERCA * G / media) if media > 0 else None)

    # --- troca de alvo (D1): avanço = B + w_p·(γΦ(s′, velho) − Φ(s, velho)); parado = w_p·(γ−1)·Φ(s, velho) ---
    trocas = []
    for p in viagem:
        if p["prev"] is None:
            trocas.append((p, None))
            continue
        avanco = p["B"] + w_p * (GAMMA * p["phi2"] - p["prev"])
        parado = w_p * (GAMMA - 1) * p["prev"]
        trocas.append((p, (avanco, parado)))

    # --- escada da S §10 (todos os lados em retorno, × dt) ---
    B_pos = {NOMES[p["alvo"]]: p["B"] for p in viagem}  # um B por alvo; os P e B aparecem uma vez
    soma_p03 = sum(B_pos[n] for n in ("P0", "P1", "P2", "P3"))
    soma_b = sum(B_pos[n] for n in ("B1", "B2", "B3", "B4"))
    volta_r = w_p * PHI_MAX * DT
    mergulho_r = (B_pos["P0"] + w_p * PHI_MAX) * DT
    reguladores_r = c_b * EPISODIO
    w_ap = -R.aperto  # peso proposto = o do knob (caso 9); o teto sai da escada, abaixo
    K_ap = I15.custo(I15.F_OPERACAO) * I15.JANELA  # aperto evitado por unidade de w (retorno)
    r_ap_r = w_ap * K_ap  # aperto linear evitado: segurar 117,5 N por 20 s (R2, P2)
    r_sac_r = -R.sacudida * I15.custo_sacudida(*R.sacudida_norma) * EPISODIO  # sacudida evitada, p50, degrau 5
    lado1_dir = reguladores_r + soma_b * DT + r_ap_r
    perda_knob_r = R.perda_caixa * DT  # perda do knob (por passo × dt); o knob não muda aqui
    # (2a) parte fixa: o que a perda deixa recolher de novo (ΣB P0–P3 + Φ máx·w_p); a perda vale 1,2 × isso
    fixo_r = (soma_p03 + w_p * PHI_MAX) * DT
    ineq = [
        ("queda > perda do knob", QUEDA, perda_knob_r),
        ("perda do knob > volta", perda_knob_r, volta_r),
        ("queda > mergulho", QUEDA, mergulho_r),
        ("(1) queda > reguladores até 20 s no BOTAR + B restantes + r_aperto", QUEDA, lado1_dir),
        ("(2a) perda do knob > 1,2 × (ΣB P0–P3 + Φ máx·w_p): o que a perda deixa recolher", perda_knob_r,
         MARGEM_PERDA * fixo_r),
    ]
    fecha_fixo = all(e > d for _, e, d in ineq[:4])
    devolve = not fecha_fixo
    # (2b) por s de episódio restante: positivos que a perda zera > 1,2 × (sacudida evitada/s + aperto evitado/s).
    # passo e altura_pe (paga=True) ficam de fora: lado conservador (o positivo do lado do que a perda custa é menor).
    pos_s = R.vivo + R.rastreio_linear + R.rastreio_angular  # positivos que a perda zera, por s (valor 1)
    sac_s = -R.sacudida * I15.custo_sacudida(*R.sacudida_norma)  # sacudida evitada por s, no p50, degrau 5
    ap_s = w_ap * I15.custo(I15.F_OPERACAO)  # aperto evitado por s (117,5 N, F_ref 49 N)
    ok_2b = pos_s > MARGEM_PERDA * (sac_s + ap_s)
    teto_2b = (pos_s / MARGEM_PERDA - sac_s) / I15.custo(I15.F_OPERACAO)  # w < (pos_s/1,2 − sac_s)/(117,5/49)
    # (1) inalterada: teto de w pela desigualdade (1), com o aperto no retorno de 20 s
    teto_1 = (QUEDA - reguladores_r - soma_b * DT) / K_ap
    # informação (não é critério; R2 da H8): a versão da S §10, sem os positivos, com o aperto antigo (−0,02) e o do knob
    lado_s10_r = fixo_r + r_sac_r + r_ap_r
    lado_s10_antigo_r = fixo_r + r_sac_r + 0.02 * K_ap

    # ================= saída =================
    emite("Item 3 da Parte 0: escada de custos, B_alvo, w_p e pesos das cercas (S §9.4, §10, §16 item 3)")
    emite(f"unidade: retorno do mjlab (× dt = {DT} s); custo/s = w·média(x); C = custo/s × duração.")
    emite("")

    emite("1. Custo por s dos reguladores (custos_por_s; valor por s, negativo = custo)")
    emite(f"  {'trecho':<10}{'arquivo':<19}{'linhas':>7}{'completo':>10}{'taxa_acao':>11}{'vel_junta':>11}"
          f"{'acel_junta':>12}{'energia':>10}{'total':>10}")
    for fase, arquivos in TRECHOS.items():
        for n in arquivos:
            v = custos[(fase, n)]
            emite(f"  {fase:<10}{n:<19}{v['linhas']:>7}{('sim' if v['completo'] else 'NÃO'):>10}"
                  f"{f3(v['taxa_acao']):>11}{f3(v['vel_junta']):>11}{f3(v['acel_junta']):>12}"
                  f"{f3(v['energia']):>10}{f3(v['total']):>10}")
        med, _, comp = media_fase(fase, custos)
        emite(f"  {fase:<10}{'todos juntos':<19}{'':>7}{('sim' if comp else 'parcial'):>10}"
              f"{f3(med['taxa_acao']):>11}{f3(med['vel_junta']):>11}{f3(med['acel_junta']):>12}"
              f"{f3(med['energia']):>10}{f3(med['total']):>10}")
    incompletos = [f"{f}/{n}" for (f, n), v in custos.items() if not v["completo"]]
    emite(f"  incompletos (sem a_* ou alvo_pd_*: taxa_acao e energia NaN; total = só vel e acel): "
          + ", ".join(incompletos))
    emite(f"  médias usadas: PEGAR = {usados_p}; BOTAR = {usados_b} (ANDAR e CARREGAR não entram no roteiro).")
    emite("")

    emite("2. Trechos de viagem do roteiro (item 1) e B_alvo_i = 1,5·C/dt (PD-31); ganho e quociente (PD-32)")
    emite(f"  {'alvo':<5}{'trecho':<9}{'fase':<7}{'dur s':>7}{'Φ ini':>8}{'Φ fim':>8}{'ΔΦ':>8}"
          f"{'custo/s ret':>12}{'C ret':>9}{'B passo':>9}{'B·dt ret':>10}{'quoc passo':>12}")
    for p in viagem:
        quoc = f"{p['quoc']:.4f}" if np.isfinite(p["quoc"]) else "–"
        emite(f"  {NOMES[p['alvo']]:<5}{p['nome']:<9}{p['fase']:<7}{p['dur']:>7.2f}{p['phi0']:>8.4f}"
              f"{p['phi1']:>8.4f}{p['phi1'] - p['phi0']:>+8.4f}{-p['c']:>+12.4f}{p['C']:>9.4f}{p['B']:>9.4f}"
              f"{p['B'] * DT:>10.4f}{quoc:>12}")
    emite("  ⚠ durações: roteiro do item 1 (Suposições 4 e 5). C ret = custo/s × dur; B passo = 1,5·C/dt;")
    emite("    quoc passo = C/(ΔΦ·dt), w_p por passo que cobre o custo (sem o 1,5); ΔΦ ≤ 0 = –.")
    emite(f"  w_p (por passo, knob) = 1,5 × max(quoc) = 1,5 × {decisivo['quoc']:.4f} = {w_p:.4f} "
          f"(decide {decisivo['fase']} {NOMES[decisivo['alvo']]}, trecho {decisivo['nome']}); "
          f"retorno do progresso no trecho = w_p·ΔΦ·dt = {w_p * decisivo['d_phi'] * DT:.4f} ≥ 1,5·C = {MARGEM_B * decisivo['C']:.4f}")
    emite("")

    emite("3. Troca de alvo (D1), por passo: avanço = B_alvo + w_p·(γΦ(s′, velho) − Φ(s, velho)) > parado = w_p·(γ−1)·Φ(s, velho)")
    emite(f"  {'alvo':<5}{'Φ velho s':>10}{'Φ velho s′':>11}{'avanço':>11}{'avanço ret':>11}{'parado':>11}  veredito")
    for p, par in trocas:
        nome = NOMES[p["alvo"]]
        if par is None:
            emite(f"  {nome:<5}{'–':>10}{p['phi0']:>11.4f}{'–':>11}{'–':>11}{'–':>11}  n/a (sem alvo velho)")
            continue
        av, pa = par
        ok = av > pa
        emite(f"  {nome:<5}{p['prev']:>10.4f}{p['phi2']:>11.4f}{av:>11.4f}{av * DT:>11.4f}{pa:>11.4f}  "
              f"{'OK' if ok else 'FALHA'}")
    emite("  P0 entra do −1 (PD-36 não dá o Φ do −1): sem alvo velho, não checado. B1 entra do fim do P4.")
    emite("")

    emite("4. Escada da S §10 (cada desigualdade compara retorno com retorno; passo = retorno ÷ dt)")
    emite(f"  {'desigualdade':<66}{'esq ret':>9}{'dir ret':>9}{'esq passo':>11}{'dir passo':>11}  veredito")
    for nome, esq, dir_ in ineq:
        emite(f"  {nome:<66}{esq:>9.4f}{dir_:>9.4f}{esq / DT:>11.4f}{dir_ / DT:>11.4f}  "
              f"{'OK' if esq > dir_ else 'FALHA'}")
    emite(f"  queda (knob terminacao, por passo) = {-R.terminacao:.1f}; reguladores BOTAR até 20 s = "
          f"{reguladores_r:+.4f} ret; ΣB P0–P3 = {soma_p03:.4f} passo; ΣB B1–B4 = {soma_b:.4f} passo; w_p = {w_p:.4f}")
    emite(f"  (2a) parte fixa: perda do knob {perda_knob_r:.4f} ret > 1,2 × (ΣB P0–P3 + Φ máx·w_p) = "
          f"{MARGEM_PERDA * fixo_r:.4f} (sem a sacudida, o aperto nem os positivos)")
    emite(f"  (2b) por s de episódio restante: positivos que a perda zera = {pos_s:.4f}/s "
          f"((vivo {R.vivo:g} + rastreio linear {R.rastreio_linear:g} + angular {R.rastreio_angular:g}), valor 1)")
    emite(f"       sacudida evitada = {sac_s:.4f}/s (−sacudida {R.sacudida:g} × custo p50 "
          f"{I15.custo_sacudida(*R.sacudida_norma):.1f}); aperto evitado = {ap_s:.4f}/s (w {w_ap:g} × {I15.custo(I15.F_OPERACAO):.4f})")
    emite(f"       {pos_s:.4f} > 1,2 × ({sac_s:.4f} + {ap_s:.4f}) = {MARGEM_PERDA * (sac_s + ap_s):.4f}: "
          f"{'OK' if ok_2b else 'FALHA'}")
    emite(f"  teto do peso do aperto pela (2b): w < (pos_s/1,2 − sac_s)/(117,5/49) = {teto_2b:.4f}; "
          f"peso do knob {w_ap:g} {'cabe' if w_ap < teto_2b else 'passa'}")
    emite(f"  teto pela desigualdade (1), inalterada: {teto_1:+.4f} (o peso do knob {'cabe' if w_ap < teto_1 else 'passa'})")
    emite(f"  informação, S §10 sem os positivos (não é critério; falha com o aperto antigo, −0,02): "
          f"sacudida sozinha {r_sac_r:.4f} já passa da perda do knob {perda_knob_r:.4f}; "
          f"lado com o aperto antigo {lado_s10_antigo_r:.4f}; com o do knob {lado_s10_r:.4f}")
    emite(f"  devolve_b_alvo: knob = {R.devolve_b_alvo}, calculado = {devolve} "
          f"(True só se as quatro primeiras desigualdades da escada não fecham)")
    emite("")

    emite("5. Cercas de postura (PD-30, D4): média por passo só nos passos em que a cerca age (> 0), peso = −0,5·G ÷ média")
    emite(f"  G (ganho médio por s do progresso no roteiro do PEGAR) = {G:.5f}; 0,5·G = {FRAC_CERCA * G:.5f}")
    emite(f"  {'cerca':<8}{'unid.':<7}{'média (>0)':>13}{'% passos que age':>17}{'NaN':>6}  peso proposto")
    for k in CERCAS:
        c = cercas[k]
        peso = f"{c['peso']:+.5f}" if c["peso"] is not None else "indefinido (sem custo no PEGAR)"
        emite(f"  {k:<8}{UNIDADES[k]:<7}{c['media']:>13.4f}{100 * c['pct']:>16.1f}%{c['nans']:>6}  {peso}")
    emite("  torque sem valor no 23250_k2 (sem alvo_pd_*): NaN fora da média.")
    emite("")

    emite("Diagnósticos")
    emite(f"  soma literal do progresso no PEGAR (D1; B_P0 não entra: o roteiro começa em P0), retorno/s = "
          f"{lit_pe:+.5f} (w_p {soma_w * DT / T_pe:+.5f}; B {soma_b * DT / T_pe:+.5f}). G = {G:.5f} retorno/s (PD-30).")
    ini_b2 = [i for i, nm in enumerate(bo["trecho"]) if nm == "B1_B2"][0]
    dist = float(np.sum(np.linalg.norm(np.diff(bo["caixa"][:ini_b2 + 1], axis=0), axis=1)))
    dur_viagem_b = sum(p["dur"] for p in viagem if p["fase"] == "BOTAR")
    emite(f"  BOTAR por S §7: distância da caixa P4 → soltura (roteiro) = {dist:.3f} m; ÷ {VEL_BOTAR} m/s = "
          f"{dist / VEL_BOTAR:.2f} s; roteiro das viagens B1–B4 = {dur_viagem_b:.2f} s.")
    emite(f"  PEGAR nos CSV: a fase pegar dura {_dur('23250_k2', 'pegar'):.1f} s e {_dur('23250_topo004', 'pegar'):.1f} s "
          f"(não 0,5–2,2 s por alvo; sem a divisão por alvo nos CSV).")
    emite("")

    emite("Suposições (não definidas no plano; o número depende delas)")
    suposicoes = [
        "1. C de cada alvo = custo/s × duração do trecho de viagem que chega a ele; as esperas (_espera) ficam fora "
        "(o bônus já foi pago na chegada).",
        "2. Reguladores = custos_por_s total (vel_junta, acel_junta, taxa_acao, energia). As cercas não entram no C: "
        "o peso delas sai nesta conta (PD-30).",
        "3. Custo/s do PEGAR = média dos 5 arquivos completos; do BOTAR = 24800_botar_t015 (único completo). "
        "23250_k2 fica fora das médias (sem taxa e energia).",
        "4. Duração de cada trecho de viagem = roteiro do item 1 (P0 1,0 s; P1 0,5 s; P2 1 passo; P3 0,5 s; P4 1,0 s; "
        "B1 1,0 s; B2 0,5 s; B3 0,5 s; B4 1,0 s). O plano manda medir no CSV e, no BOTAR, usar distância/0,25: não há "
        "a divisão por trecho; o total do BOTAR por distância está no diagnóstico.",
        "5. A fase pegar dos CSV não tem a divisão por alvo (só fase, alvo_x/y/z e oh_*); não usei os tempos medidos.",
        "6. Φ novo = Φ da primeira linha do trecho; Φ velho = Φ da linha anterior. P0 não tem alvo velho (n/a). "
        "B1 usa o último Φ do PEGAR (o BOTAR começa no P4). Φ velho s′ (D1) = o mesmo Φ com o índice do alvo velho "
        "só no passo do avanço, com a caixa e o σ do trecho que termina.",
        "7. Ganho de Φ por s (PD-32) = subida de Φ dentro do trecho (Φ fim − Φ ini) ÷ duração. Trecho sem subida "
        "(P2) fica fora do máximo.",
        "8. G (PD-30, retorno/s) = w_p × Σ ΔΦ·dt dos trechos de viagem do PEGAR ÷ duração total do roteiro do PEGAR "
        "(com esperas). A subida de Φ é mantida (PD-30); a soma literal sai no diagnóstico.",
        "9. Cercas: média por passo sobre os 6 PEGAR, só nos passos em que a cerca age (> 0; D4), torque NaN fora; "
        "peso = −0,5·G ÷ média (sinal negativo, como custos_por_s).",
        "10. Escada (1): queda = |terminacao|·dt; reguladores até os 20 s do episódio no BOTAR = custo/s × 20 "
        "(limite superior: o plano não fixa quando o BOTAR começa); B_alvo restantes = B1–B4 (queda logo no início); "
        "+ aperto evitado.",
        "11. Escada (2), por coordenação (30/10): (2a) parte fixa = perda do knob > 1,2 × (ΣB P0–P3 + Φ máx·w_p), o que a "
        "perda deixa recolher; (2b) por s = positivos que a perda zera (vivo + rastreio linear + angular, valor 1; passo "
        "e altura_pe ficam de fora, lado conservador) > 1,2 × (sacudida evitada/s + aperto evitado/s). Os positivos são o "
        "que a perda custa, não o que ela rende. A perda do knob é lida, não recalculada.",
        "12. Volta à pose de pé = w_p·Φ máx (paga uma vez, como potencial, sem bônus). Mergulho = B_P0 + w_p·Φ máx.",
        "13. perda_caixa (peso −1) paga perda·dt no passo da perda, como a terminação paga −200·dt.",
        "14. devolve_b_alvo = True só se (queda > perda), (perda > volta), (queda > mergulho) e (1) não fecham com a "
        "perda fixa.",
    ]
    emite("\n".join("  " + s for s in suposicoes))


def _dur(nome, fase):
    return len(_carrega(nome, fase)["t"]) * DT


if __name__ == "__main__":
    main()
    emite(f"Linhas deste script: {len(Path(__file__).read_text(encoding='utf-8').splitlines())}")
    texto = "\n".join(linhas_saida)
    print(texto)
    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(texto + "\n", encoding="utf-8")
