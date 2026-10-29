"""Item 3 da Parte 0 (S §16 item 3; plano B5, PD-30 a PD-33, PD-36; S §9.2–9.4, §10).

Propõe os pesos e checa cada desigualdade:
1. `B_alvo` por alvo (P0–P4, B1–B4) = 1,5 × custo dos reguladores no trecho até o alvo (PD-31).
2. `w_p` do progresso = 1,5 × o maior custo por s ÷ ganho de Φ por s, nos trechos de viagem do roteiro (PD-32).
3. Troca de alvo: B_alvo_i > w_p·(Φ_velho − γ·Φ_novo), medido no roteiro (checagem pedida pelo orquestrador).
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
                        prev=float(phi[a - 1]) if a > 0 else None))
    return out


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
    phi_pe = I1.avalia(pe, refs, I1.pontos_de(pe))["phi"]
    phi_bo = I1.avalia(bo, refs, I1.pontos_de(bo))["phi"]
    pl_pe, pl_bo = pernas(pe, phi_pe, "PEGAR"), pernas(bo, phi_bo, "BOTAR")
    pl_bo[0]["prev"] = float(phi_pe[-1])  # entrada no B1: o roteiro do BOTAR começa no fim do P4
    viagem = [p for p in pl_pe + pl_bo if not p["espera"]]
    T_pe = (len(pe["t"]) - 1) * DT  # duração do roteiro do PEGAR (com as esperas)

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
    lit_pe = w_p * float(np.sum(GAMMA * phi_pe[1:] - phi_pe[:-1])) * DT / T_pe  # retorno/s da soma literal de r

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
        media = float(fin.mean()) if fin.size else np.nan
        cercas[k] = dict(media=media, pct=float((fin > 0).mean()) if fin.size else np.nan,
                         nans=int((~np.isfinite(x)).sum()), peso=(-FRAC_CERCA * G / media) if media > 0 else None)

    # --- troca de alvo (checagem pedida pelo orquestrador) ---
    trocas = []
    for p in viagem:
        if p["prev"] is None:
            trocas.append((p, None))
            continue
        lim = w_p * (p["prev"] - GAMMA * p["phi0"])
        trocas.append((p, lim))

    # --- escada da S §10 (todos os lados em retorno, × dt) ---
    B_pos = {NOMES[p["alvo"]]: p["B"] for p in viagem}  # um B por alvo; os P e B aparecem uma vez
    soma_p03 = sum(B_pos[n] for n in ("P0", "P1", "P2", "P3"))
    soma_b = sum(B_pos[n] for n in ("B1", "B2", "B3", "B4"))
    perda_passo = MARGEM_PERDA * (soma_p03 + w_p * PHI_MAX)  # knob perda_caixa (passo)
    perda_r = perda_passo * DT
    volta_r = w_p * PHI_MAX * DT
    mergulho_r = (B_pos["P0"] + w_p * PHI_MAX) * DT
    reguladores_r = c_b * EPISODIO
    lado1_dir = reguladores_r + soma_b * DT
    lado2_dir = (soma_p03 + w_p * PHI_MAX) * DT
    ineq = [
        ("queda > perda", QUEDA, perda_r),
        ("perda > volta", perda_r, volta_r),
        ("queda > mergulho", QUEDA, mergulho_r),
        ("(1) queda > reguladores até 20 s no BOTAR + B_alvo restantes", QUEDA, lado1_dir),
        ("(2) perda > B_alvo P0–P3 + Φ máx·w_p (perda = 1,2 × lado direito)", perda_r, lado2_dir),
    ]
    fecha_fixo = all(e > d for _, e, d in ineq[:4])
    devolve = not fecha_fixo

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

    emite("3. Troca de alvo, por passo no mesmo passo: B_alvo_i > w_p·(Φ_velho − γ·Φ_novo)")
    emite(f"  {'alvo':<5}{'Φ velho':>9}{'Φ novo':>9}{'w_p·(Φv−γΦn)':>15}{'B_alvo':>10}{'B·dt ret':>10}"
          f"{'1,5× (info)':>13}  veredito")
    for p, lim in trocas:
        nome = NOMES[p["alvo"]]
        if lim is None:
            emite(f"  {nome:<5}{'–':>9}{p['phi0']:>9.4f}{'–':>15}{p['B']:>10.4f}{p['B'] * DT:>10.4f}"
                  f"{'–':>13}  n/a (sem alvo velho)")
            continue
        ok = p["B"] > lim
        emite(f"  {nome:<5}{p['prev']:>9.4f}{p['phi0']:>9.4f}{lim:>15.4f}{p['B']:>10.4f}{p['B'] * DT:>10.4f}"
              f"{MARGEM_B * lim:>13.4f}  {'OK' if ok else 'FALHA'}")
    emite("  P0 entra do −1 (PD-36 não dá o Φ do −1): sem alvo velho, não checado. B1 entra do fim do P4.")
    emite("")

    emite("4. Escada da S §10 (cada desigualdade compara retorno com retorno; passo = retorno ÷ dt)")
    emite(f"  {'desigualdade':<66}{'esq ret':>9}{'dir ret':>9}{'esq passo':>11}{'dir passo':>11}  veredito")
    for nome, esq, dir_ in ineq:
        emite(f"  {nome:<66}{esq:>9.4f}{dir_:>9.4f}{esq / DT:>11.4f}{dir_ / DT:>11.4f}  "
              f"{'OK' if esq > dir_ else 'FALHA'}")
    emite(f"  queda (knob terminacao, por passo) = {-R.terminacao:.1f}; reguladores BOTAR até 20 s = "
          f"{reguladores_r:+.4f} ret; ΣB P0–P3 = {soma_p03:.4f} passo; ΣB B1–B4 = {soma_b:.4f} passo; w_p = {w_p:.4f}")
    emite(f"  perda_caixa = 1,2 × lado direito de (2): retorno {perda_r:.4f}; por passo (knob) = retorno/dt = "
          f"{perda_passo:.4f}")
    emite(f"  devolve_b_alvo = {devolve} (True só se a escada (queda > perda > volta, queda > mergulho, (1)) "
          f"não fecha com a perda fixa)")
    emite("")

    emite("5. Cercas de postura (PD-30): média por passo no PEGAR (w = 1), peso = −0,5·G ÷ média")
    emite(f"  G (ganho médio por s do progresso no roteiro do PEGAR) = {G:.5f}; 0,5·G = {FRAC_CERCA * G:.5f}")
    emite(f"  {'cerca':<8}{'unid.':<7}{'média/passo':>13}{'% passos>0':>12}{'NaN':>6}  peso proposto")
    for k in CERCAS:
        c = cercas[k]
        peso = f"{c['peso']:+.5f}" if c["peso"] is not None else "indefinido (sem custo no PEGAR)"
        emite(f"  {k:<8}{UNIDADES[k]:<7}{c['media']:>13.4f}{100 * c['pct']:>11.1f}%{c['nans']:>6}  {peso}")
    emite("  torque sem valor no 23250_k2 (sem alvo_pd_*): NaN fora da média.")
    emite("")

    emite("Diagnósticos")
    emite(f"  soma literal do progresso no PEGAR, retorno/s = {lit_pe:+.5f} (negativo: por isso PD-30 usa a "
          f"subida de Φ, Suposição 8). G = {G:.5f} retorno/s (conferir: G antigo 0,03369).")
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
        "B1 usa o último Φ do PEGAR (o BOTAR começa no P4).",
        "7. Ganho de Φ por s (PD-32) = subida de Φ dentro do trecho (Φ fim − Φ ini) ÷ duração. Trecho sem subida "
        "(P2) fica fora do máximo.",
        "8. G (PD-30, retorno/s) = w_p × Σ ΔΦ·dt dos trechos de viagem do PEGAR ÷ duração total do roteiro do PEGAR "
        "(com esperas). PD-30 não diz se é a soma literal de r; a soma literal é negativa (diagnóstico).",
        "9. Cercas: média por passo sobre os 6 PEGAR (torque NaN fora); peso = −0,5·G ÷ média (sinal negativo, como "
        "custos_por_s).",
        "10. Escada (1): queda = |terminacao|·dt; reguladores até os 20 s do episódio no BOTAR = custo/s × 20 "
        "(limite superior: o plano não fixa quando o BOTAR começa); B_alvo restantes = B1–B4 (queda logo no início).",
        "11. Escada (2): perda = 1,2 × (ΣB P0–P3 + w_p·Φ máx), Φ máx = 1; r_aperto = 0 (item 15 ainda não feito).",
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
