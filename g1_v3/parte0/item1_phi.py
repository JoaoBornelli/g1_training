"""Item 1 da Parte 0 (S §16 item 1; plano B5 e §6.3): Φ e a recompensa por passo no caminho roteirizado.

Entrada: as trajetórias sintéticas de `comum.trajetoria_pegar` e `comum.trajetoria_botar` (sem CSV).
Confere os quatro critérios da linha do item 1 do plano: Φ não desce dentro de cada alvo; pairar a
1 cm por 2 s soma ≤ 0; tombar a caixa 40° no P3 leva Φ a ≤ a rampa; soltar uma mão divide Φ por 2.
Pesos sem medida (saem do item 3): w_p = 1 e B_alvo = 0. Imprime a tabela e grava
`g1_v3/parte0/saida/item1_phi.txt`. Roda com `python -m g1_v3.parte0.item1_phi` na raiz do repo.
"""
import math
import os

import numpy as np
import torch

from g1_v3 import alvos as A
from g1_v3 import potencial as P
from g1_v3.knobs import ATIVO
from g1_v3.parte0 import comum as C

K = ATIVO.alvos
R = ATIVO.recompensa
NOMES = ("P0", "P1", "P2", "P3", "P4", "B1", "B2", "B3", "B4")  # ordem de ALVOS (knobs)
A_CAIXA = C.A_CAIXA
DT = C.DT
EPS = 1e-9  # folga de ponto flutuante nos critérios de ≤ e de razão
SIGMA_MIN = 0.01  # σ de alvo métrico com piso de 1 cm (plano E3); B4 é adimensional, sem piso
N_HOLD = round(2.0 / DT)  # 2 s de pausa a 50 Hz
OFFSET = 0.01  # 1 cm do alvo, na grandeza do próprio alvo (palmas, caixa ou pelve)
# ⚠ Pesos sem medida: o item 3 mede. Até lá, 1 e 0 (a B5 manda).
W_P = 1.0 if R.progresso is None else R.progresso
B_ALVO = np.array([0.0 if b is None else b for b in R.b_alvo])
# Grupos pela §6.3 e pela E4 do plano: qual grandeza o k_principal mede e quais invariantes valem.
PALMA = [0, 1, 2, 7]  # k pelas palmas: P0–P2 e B3
CAIXA = [3, 4, 5, 6]  # k pela caixa: P3, P4, B1, B2 (B4 usa a pose)
POUSO = [5, 6]  # B1 e B2: k × fator_pouso
DESLOC_PEGA = [0, 1, 2]  # invariante P0–P2: a caixa desloca menos de a na laje
MAOS = [2, 3, 4, 5, 6]  # invariante P2–B2: caixa nivelada e duas mãos (g_mãos)
DESLOC_BOTAR = [7, 8]  # invariante B2–B4: caixa parada, desloca pouco do ponto de apoio
METRICOS = [0, 1, 2, 3, 4, 5, 6, 7]  # σ métrico, com piso de 1 cm; o B4 é a pose adimensional, sem piso
ALVO_P1 = A.palmas_alvo_caixa(torch.tensor([1]), torch.tensor([A_CAIXA], dtype=torch.float64), K).numpy()[0]


def _t(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


def _repete(estado):
    """Um estado repetido N_HOLD vezes (a pausa de 2 s)."""
    return np.repeat(np.asarray(estado)[None], N_HOLD, 0)


def _g(x, lim, delta):
    """Rampa da §6.2 `Alvos` por array (numpy). Quem chama escolhe o x_lim e o Δ."""
    return P.rampa(_t(x), lim, delta).numpy()


def referencias() -> dict:
    """Referência de cada alvo para a faixa do Φ (E4): tronco (rad) e recuo (m). P0–P3 e B1–B3: a pega
    no topo da laje; P4: pose de pé com caixa (tronco pela FK de `q_com_caixa`, recuo do
    `caixa_transporte_b`); B4: pose de pé sem caixa (tronco 0 e sem recuo, como nas cercas da Parte 0)."""
    ref = C._ref()
    a = torch.tensor([A_CAIXA], dtype=torch.float64)
    tr_p = float(ref.tronco(_t([C.TOPO_PEGAR]), a)[0])
    tr_b = float(ref.tronco(_t([C.TOPO_BOTAR]), a)[0])
    rc_p = float(ref.recuo(_t([C.TOPO_PEGAR]), a)[0])
    rc_b = float(ref.recuo(_t([C.TOPO_BOTAR]), a)[0])
    tr_c = float(C.fk(ref.q_com_caixa(a).numpy())["tronco"][0])
    rc_c = float(np.linalg.norm(ref.caixa_transporte_b(a).numpy()[0, :2]))
    return dict(pelve_pe=ref.pelve_de_pe,
                tronco=np.array([tr_p] * 4 + [tr_c] + [tr_b] * 3 + [0.0]),
                recuo=np.array([rc_p] * 4 + [rc_c] + [rc_b] * 3 + [np.nan]))


def _trechos(idx):
    """Início, fim e número do trecho de cada passo; trecho = sequência de mesmo índice de alvo."""
    ini = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
    fim = np.r_[ini[1:] - 1, len(idx) - 1]
    trecho = np.searchsorted(ini, np.arange(len(idx)), side="right") - 1
    return ini, fim, trecho


def _primeiro_trecho(idx, alvo):
    """(início, fim) do primeiro trecho do alvo no caminho, ou None se o caminho não passa por ele."""
    ini, fim, _ = _trechos(idx)
    hit = np.flatnonzero(idx[ini] == alvo)
    return (ini[hit[0]], fim[hit[0]]) if len(hit) else None


def pontos_de(tr) -> dict:
    """Pontos de referência do caminho: caixa no início (desloc. P0–P2) e caixa no fim da descida do B2
    (ponto de apoio, desloc. B2–B4). ⚠ Apoio = fim do trecho B2, onde a caixa chega à laje."""
    b2 = _primeiro_trecho(tr["indice"], 6)
    apoio = tr["caixa"][b2[1]] if b2 else tr["caixa"][0]
    return dict(caixa0=tr["caixa"][0], apoio=apoio)


def _distancia(tr, alvo, refs):
    """d na medida do alvo corrente (E4): palmas (média das duas) em P0–P2 e B3; distância da caixa
    ao alvo em P3, P4, B1, B2; em B4, √((Δpelve/0,03)² + (Δtronco/5°)²)."""
    idx = tr["indice"]
    rel = tr["palmas"] - tr["caixa"][:, None, :]
    alvo_rel = A.palmas_alvo_caixa(torch.as_tensor(idx, dtype=torch.long),
                                   _t(np.full(len(idx), A_CAIXA)), K).numpy()
    d_palma = np.linalg.norm(rel - alvo_rel, axis=2).mean(1)
    d_caixa = np.linalg.norm(tr["caixa"] - alvo, axis=1)
    d_pose = np.hypot((tr["pelve"] - refs["pelve_pe"]) / K.tol_pelve_b4,
                      np.degrees(tr["tronco"] - refs["tronco"][idx]) / K.tol_tronco_b4_graus)
    return np.where(np.isin(idx, PALMA), d_palma, np.where(np.isin(idx, CAIXA), d_caixa, d_pose))


def avalia(tr, refs, pontos, alvo=None, sigma=None) -> dict:
    """Φ por passo de `tr` (arrays: indice, palmas (T,2,3), caixa (T,3), pelve, tronco em rad, incl em graus).
    `alvo` (T,3) é a caixa do alvo; None = caixa no fim do trecho corrente (o roteiro chega no alvo).
    `sigma` None = d no início de cada trecho, com piso (E3). Devolve Φ, σ e d por passo."""
    idx = np.asarray(tr["indice"], dtype=int)
    caixa, palmas = tr["caixa"], tr["palmas"]
    ini, fim, trecho = _trechos(idx)
    if alvo is None:
        alvo = caixa[fim[trecho]]
    d = _distancia(tr, alvo, refs)
    if sigma is None:
        sig0 = d[ini][trecho]
        sigma = np.where(np.isin(idx, METRICOS), np.maximum(sig0, SIGMA_MIN), sig0)
    # k_principal: nucleo(d, σ), vezes o fator de pouso no B1 e no B2 (velocidade vertical por diferença finita)
    z = caixa[:, 2]
    v = np.diff(z, prepend=z[0]) / DT
    pouso = P.fator_pouso(_t(v), _t(z - A_CAIXA - C.TOPO_BOTAR), R.sigma_pouso[0], R.altura_pouso,
                              R.altura_pouso_inicio).numpy()
    k = P.nucleo(_t(d), _t(sigma)).numpy() * np.where(np.isin(idx, POUSO), pouso, 1.0)
    # g: desloc. P0–P2 (x_lim = 1,0·a, Δ = 0,5·a); desloc. B2–B4; inclinação (x_lim = 30°, a fase aprender)
    maos = np.isin(idx, MAOS)
    x_ini = np.linalg.norm(caixa[:, :2] - pontos["caixa0"][:2], axis=1)
    x_apo = np.linalg.norm(caixa[:, :2] - pontos["apoio"][:2], axis=1)
    g_desl = np.where(np.isin(idx, DESLOC_PEGA),
                      _g(x_ini, K.desloc_p0p2_frac * A_CAIXA, K.delta_p0p2_frac * A_CAIXA), 1.0)
    g_desl = np.where(np.isin(idx, DESLOC_BOTAR), _g(x_apo, K.desloc_b2b4, K.delta_b2b4), g_desl)
    g_incl = np.where(maos, _g(tr["incl"], K.incl_inicio_graus, K.delta_incl_graus), 1.0)
    # g_mãos = contatos/2; contato = palma a até tol_palma do ponto do P1 (⚠ o plano não dá o contato em número)
    contato = np.linalg.norm(palmas - caixa[:, None, :] - ALVO_P1, axis=2) <= K.tol_palma
    g_maos = np.where(maos, contato.sum(1) / 2.0, 1.0)
    # faixa: tronco e recuo (x = |desvio| da referência); o recuo não vale no B4. O CoM não entra (ver a saída)
    # F9 P2 (29/10): um lado só e 1 no B4; a saída gravada em `parte0/saida/item1_phi.txt` é da forma antiga (o dono aceitou: o B_alvo não muda)
    g_tron = np.where(idx == 8, 1.0,_g(np.maximum(np.degrees(tr["tronco"] - refs["tronco"][idx]), 0.0),
                                         K.faixa_tronco_graus, K.delta_tronco_graus))
    rec_ref = refs["recuo"][idx]
    g_rec = np.where(np.isnan(rec_ref), 1.0,
                     _g(np.abs(np.linalg.norm(caixa[:, :2], axis=1) - np.nan_to_num(rec_ref)),
                        K.faixa_recuo, K.delta_recuo))
    phi = P.phi(_t(k), [_t(g) for g in (g_desl, g_incl, g_maos, g_tron, g_rec)]).numpy()
    return dict(indice=idx, phi=phi, sigma=sigma, d=d, ini=ini, fim=fim, trecho=trecho)


def recompensa(res) -> np.ndarray:
    """r por passo (E4): r[t] = w_p·(γΦ(t) − Φ(t−1)) + B_alvo·avançou, com r[0] = 0 (PD-36 não se aplica aqui)."""
    idx, phi = res["indice"], res["phi"]
    av = np.r_[False, idx[1:] != idx[:-1]]
    r = np.zeros(len(phi))
    r[1:] = W_P * P.progresso(phi[1:], phi[:-1], R.gamma) + B_ALVO[idx[1:]] * av[1:]
    return r


def _ok(b):
    return "OK" if b else "FALHA"


def tabela(nome, tr, res) -> list:
    """Uma linha por trecho de alvo: Φ no início e no fim, menor ΔΦ dentro do trecho e Σr do trecho."""
    r = recompensa(res)
    linhas = [f"{nome}  (laje {'0,30' if nome == 'PEGAR' else '0,15'} m, a = {A_CAIXA:.2f} m, {len(tr['t'])} passos)",
              f"  {'trecho':<11} {'alvo':<4} {'passos':>6} {'Φ início':>9} {'Φ fim':>9} {'min ΔΦ':>10} {'Σ r':>10}"]
    for a0, a1 in zip(res["ini"], res["fim"]):
        phi = res["phi"]
        dmin = f"{np.min(np.diff(phi[a0:a1 + 1])):+.2e}" if a1 > a0 else "–"
        linhas.append(f"  {tr['trecho'][a0]:<11} {NOMES[res['indice'][a0]]:<4} {a1 - a0 + 1:>6} "
                      f"{phi[a0]:>9.4f} {phi[a1]:>9.4f} {dmin:>10} {r[a0:a1 + 1].sum():>+10.4f}")
    return linhas


def criterios(pe, bo, rpe, rbo, pp, pb, refs):
    """Os quatro critérios da linha do item 1 do plano. Devolve as linhas e a lista de OK/FALHA."""
    linhas, oks = [], []

    # 1. Φ não desce dentro de cada alvo, no caminho certo (PEGAR e BOTAR)
    piores = []
    for nome, res in (("PEGAR", rpe), ("BOTAR", rbo)):
        for a0, a1 in zip(res["ini"], res["fim"]):
            if a1 > a0:
                piores.append((float(np.min(np.diff(res["phi"][a0:a1 + 1]))), nome, NOMES[res["indice"][a0]]))
    pior = min(piores)
    oks.append(pior[0] >= -EPS)
    linhas.append(f"1. Φ não desce dentro de cada alvo: {_ok(oks[-1])}; pior ΔΦ = {pior[0]:+.2e} "
                  f"({pior[1]}, alvo {pior[2]}); tolerância {-EPS:.0e}")

    # 2. Pairar a 1 cm do alvo por 2 s soma ≤ 0. Estado = fim do trecho do alvo, deslocado 1 cm na grandeza
    # dele; índice fixo (o avanço durante a pausa não é simulado, ver as premissas)
    somas = []
    for i in range(len(NOMES)):
        tr, res, pts = (pe, rpe, pp) if i < 5 else (bo, rbo, pb)
        a0, a1 = _primeiro_trecho(tr["indice"], i)
        pal, cai = _repete(tr["palmas"][a1]), _repete(tr["caixa"][a1])
        pel, tro = np.full(N_HOLD, tr["pelve"][a1]), np.full(N_HOLD, tr["tronco"][a1])
        if i in PALMA:
            pal = pal + np.array([OFFSET, 0.0, 0.0])
        if i in CAIXA:
            cai = cai + np.array([OFFSET, 0.0, 0.0])
        if i == 8:
            pel = pel + OFFSET
        hold = dict(indice=np.full(N_HOLD, i), palmas=pal, caixa=cai, pelve=pel, tronco=tro,
                    incl=np.zeros(N_HOLD))
        sig = np.full(N_HOLD, res["sigma"][a0])
        rh = avalia(hold, refs, pts, alvo=_repete(tr["caixa"][a1]), sigma=sig)
        somas.append((NOMES[i], res["sigma"][a0], rh["phi"][-1], float(recompensa(rh).sum())))
    maior = max(s[3] for s in somas)
    oks.append(maior <= EPS)
    linhas.append(f"2. Pairar a 1 cm por 2 s soma ≤ 0: {_ok(oks[-1])}; maior soma = {maior:+.2e}")
    for nome, sig, fi, soma in somas:
        linhas.append(f"     {nome}: σ = {sig:.4f}, Φ parado = {fi:.4f}, Σ r = {soma:+.2e}")

    # 3. Tombar a caixa 40° no P3: Φ ≤ a rampa de inclinação em 40°. A inclinação sai da função do plano
    a0, a1 = _primeiro_trecho(pe["indice"], 3)
    quat = torch.tensor([[math.cos(math.radians(20)), math.sin(math.radians(20)), 0.0, 0.0]], dtype=torch.float64)
    inc = float(np.degrees(A.inclinacao_da_caixa(quat).numpy()[0]))

    def no_p3(graus):
        tr1 = dict(indice=pe["indice"][a1:a1 + 1], palmas=pe["palmas"][a1:a1 + 1], caixa=pe["caixa"][a1:a1 + 1],
                   pelve=pe["pelve"][a1:a1 + 1], tronco=pe["tronco"][a1:a1 + 1], incl=np.array([graus]))
        return float(avalia(tr1, refs, pp, alvo=pe["caixa"][a1:a1 + 1], sigma=np.array([rpe["sigma"][a0]]))["phi"][0])

    phi_reto, phi_tomb = no_p3(0.0), no_p3(inc)
    rampa_v = float(_g(inc, K.incl_inicio_graus, K.delta_incl_graus))
    oks.append(phi_tomb <= rampa_v + EPS)
    linhas.append(f"3. Tombar a caixa {inc:.1f}° no P3: {_ok(oks[-1])}; Φ reto = {phi_reto:.4f}, "
                  f"Φ tombada = {phi_tomb:.4f} ≤ rampa({inc:.0f}°) = {rampa_v:.4f}")

    # 4. Soltar uma mão divide Φ por 2 (estado = P3; a palma 1 (−y) sai 10 cm da face, g_mãos cai de 1 para 1/2)
    pal_u = pe["palmas"][a1:a1 + 1].copy()
    pal_u[0, 1, 1] += 0.10 * np.sign(pal_u[0, 1, 1] - pe["caixa"][a1, 1])
    uma = dict(indice=pe["indice"][a1:a1 + 1], palmas=pal_u, caixa=pe["caixa"][a1:a1 + 1],
               pelve=pe["pelve"][a1:a1 + 1], tronco=pe["tronco"][a1:a1 + 1], incl=np.zeros(1))
    phi_uma = float(avalia(uma, refs, pp, alvo=pe["caixa"][a1:a1 + 1], sigma=np.array([rpe["sigma"][a0]]))["phi"][0])
    razao = phi_uma / phi_reto
    oks.append(abs(razao - 0.5) <= EPS)
    linhas.append(f"4. Soltar uma mão divide Φ por 2: {_ok(oks[-1])}; Φ duas mãos = {phi_reto:.4f}, "
                  f"Φ uma mão = {phi_uma:.4f}, razão = {razao:.6f} (esperado 0,5)")
    return linhas, oks


PREMISSAS = [
    "Premissas (o plano não define estes pontos; cada um é dúvida para o dono):",
    "  pesos: w_p = 1 e B_alvo = 0 (knobs `progresso` e `b_alvo` são None; saem do item 3).",
    "  σ do alvo = d no início do trecho de mesmo índice, com piso de 1 cm; B4 sem piso (como o item 2).",
    "  k: palmas = média das duas distâncias (caixa sem giro, então a distância é no frame da caixa); caixa = 3D.",
    "  desloc. P0–P2: x_lim = 1,0·a = 0,10 m, Δ = 0,5·a; B3 e B4: x_lim 2 cm, Δ 2 cm, ponto de apoio = fim do B2.",
    "  inclinação: x_lim = 30° (início da fase aprender), Δ = 15°. Só se usa no P2–B2 e no tombar.",
    "  contato da palma = a até 3 cm do ponto do P1 (o plano diz 'em contato', sem número).",
    "  faixa: tronco e recuo. P4 usa a FK da pose de pé com caixa; o roteiro do P4 usa o tronco e a pelve da pega.",
    "  faixa do CoM NÃO entra: precisa das juntas e dos pés (PD-35), que o roteiro sintético não tem.",
    "  fator de pouso: σ = 1,0 m/s (início), altura 5 cm, v por diferença finita do roteiro.",
    "  pairar: índice fixo no alvo. Se a pausa dispara o avanço (ok_agora), a soma muda; ok_agora não é simulado.",
    "  tombar e soltar mão: as palmas não giram com a caixa; só muda o termo de inclinação e o de contato.",
]


def main():
    refs = referencias()
    pe, bo = C.trajetoria_pegar(), C.trajetoria_botar()
    for tr in (pe, bo):
        tr["incl"] = np.zeros(len(tr["t"]))
    pp, pb = pontos_de(pe), pontos_de(bo)
    rpe, rbo = avalia(pe, refs, pp), avalia(bo, refs, pb)

    linhas = ["Item 1 da Parte 0: Φ e progresso no caminho roteirizado (S §8.2, §9.1, §16 item 1)",
              f"pesos: w_p = {W_P:g}, B_alvo = {B_ALVO.tolist()} (sem medida; item 3). γ = {R.gamma}; "
              f"a = {A_CAIXA} m; passo {DT} s.", ""]
    linhas += tabela("PEGAR", pe, rpe) + [""] + tabela("BOTAR", bo, rbo) + [""]
    crit, oks = criterios(pe, bo, rpe, rbo, pp, pb, refs)
    linhas += ["Critérios (linha do item 1 do plano):"] + crit + [""]
    linhas += [f"Resultado: {sum(oks)} OK / {len(oks) - sum(oks)} FALHA"] + PREMISSAS

    saida = "\n".join(linhas) + "\n"
    print(saida, end="")
    pasta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saida")
    os.makedirs(pasta, exist_ok=True)
    with open(os.path.join(pasta, "item1_phi.txt"), "w", encoding="utf-8") as fh:
        fh.write(saida)


if __name__ == "__main__":
    main()
