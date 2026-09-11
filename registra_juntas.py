"""Roda a cadeia do BOTAR no MuJoCo clássico e grava as 29 juntas num CSV.

    python registra_juntas.py --cena ~/g1_pilota/ --checkpoint ~/Downloads/model_10500.pt

Irmão do `pilota.py`: mesma cena, mesma observação de 114 canais, mesmo ator. A
diferença é que aqui NÃO tem teclado — um roteiro troca o elo sozinho a cada N
segundos, e cada passo vira uma linha do CSV.

⚠ Ele importa do `pilota.py` e por isso mora na RAIZ, não em `g1_limpo/`. Nada aqui
depende do `mjlab`: o MuJoCo clássico roda ~40× tempo real nesta CPU, contra 96× mais
LENTO por substep no Warp.

O que sai, e nada mais:
  <saida>.csv           passo, t, fase, elo, one-hot, e o ÂNGULO CRU das 29 juntas
  <saida>.limites.csv   junta, lo, hi, default — a régua para ler o CSV
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import mujoco
import numpy as np

from pilota import ELOS, Ator, carrega_cena, monta_observacao, restaura

# ⚠ A cena é exportada do reset do PEGAR: a caixa e a laje já estão à frente do robô.
# Por isso o roteiro abre na espera e não numa aproximação.
ROTEIRO_PADRAO = ("espera:ANDAR:1.0,"
                  "pegar:PEGAR:7.0,"
                  "espera:BOTAR:1.0,"
                  "botar:BOTAR:9.0,"
                  "espera:ANDAR:1.0,"
                  "andar:ANDAR:6.0:0.8")


class Fase:
    """Um trecho do roteiro: rótulo, elo publicado, duração e o vx pedido."""

    def __init__(self, rotulo: str, elo: int, segundos: float, vx: float):
        self.rotulo, self.elo, self.segundos, self.vx = rotulo, elo, segundos, vx

    @property
    def anda(self) -> bool:
        """Fase de marcha: elo ANDAR com velocidade pedida. É ela que limpa a cena."""
        return self.elo == 0 and abs(self.vx) > 1e-6


def analisa_roteiro(texto: str) -> list[Fase]:
    """`"rotulo:ELO:segundos[:vx], ..."` -> lista de `Fase`.

    ⚠ O elo é o NOME (`PEGAR`), e não o índice: um índice trocado é um erro silencioso
    que só aparece como "o robô não faz nada".
    """
    fases: list[Fase] = []
    for pedaco in texto.split(","):
        pedaco = pedaco.strip()
        if not pedaco:
            continue
        campos = pedaco.split(":")
        if len(campos) not in (3, 4):
            raise SystemExit(f"fase malformada: {pedaco!r} — use rotulo:ELO:segundos[:vx]")
        rotulo, nome, seg = campos[0], campos[1].upper(), campos[2]
        if nome not in ELOS:
            raise SystemExit(f"elo {nome!r} não existe; use um de {', '.join(ELOS)}")
        vx = float(campos[3]) if len(campos) == 4 else 0.0
        fases.append(Fase(rotulo, ELOS.index(nome), float(seg), vx))
    if not fases:
        raise SystemExit("roteiro vazio")
    return fases


def enderecos_da_caixa(m: mujoco.MjModel, id_caixa: int) -> tuple[int, int]:
    """`(qpos_adr, dof_adr)` da junta livre da caixa."""
    jid = int(m.body_jntadr[id_caixa])
    if jid < 0:
        raise SystemExit("a caixa não tem junta — não dá para afastá-la")
    return int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])


def limpa_a_cena(m: mujoco.MjModel, d: mujoco.MjData, c, afasta: float) -> None:
    """Manda a laje E a caixa para longe, como o reset do ANDAR faz no treino.

    ⚠ AS DUAS JUNTAS, e é o pedido do dono: tirar só a laje deixaria a caixa caindo
    no caminho. No `ANDAR` os dez canais da caixa zeram no gate da observação, então
    a pose delas não muda o que a política vê — só a física.
    """
    id_mocap = int(m.body_mocapid[int(c.id_laje)])
    if id_mocap >= 0:
        d.mocap_pos[id_mocap, 0] += afasta
    else:
        print("⚠ a laje não é mocap; ela ficou onde estava")
    adr_q, adr_v = enderecos_da_caixa(m, int(c.id_caixa))
    d.qpos[adr_q] += afasta
    d.qvel[adr_v:adr_v + 6] = 0.0
    mujoco.mj_forward(m, d)


def regua_das_juntas(m: mujoco.MjModel, c) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Nomes, faixa `(29, 2)` e default `(29,)`, na ordem do `nomes_juntas` da cena.

    ⚠ A JUNTA É ACHADA PELO ENDEREÇO DE `qpos`, e NÃO pelo nome. O `cena.npz` grava
    `left_hip_pitch_joint`, mas o modelo compilado chama `robot/left_hip_pitch_joint`
    — o `mj_name2id` devolvia −1 nas 29, a faixa saía toda NaN e o resumo vinha VAZIO,
    sem erro nenhum. O `ids_junta_qpos` veio do próprio exportador e aponta a junta
    certa, com prefixo ou sem.
    """
    nomes = [str(n) for n in c.nomes_juntas]
    ids_q = np.asarray(c.ids_junta_qpos, dtype=np.int64)
    por_adr = {int(m.jnt_qposadr[j]): j for j in range(m.njnt)}
    faixa = np.zeros((len(nomes), 2))
    faltando = []
    for i in range(len(nomes)):
        jid = por_adr.get(int(ids_q[i]), -1)
        if jid < 0 or not bool(m.jnt_limited[jid]):
            # ⚠ Junta sem limite vira NaN, e NÃO 0: um zero calado viraria `frac`
            # inventado, e o resumo apontaria batente onde não há.
            faixa[i] = (np.nan, np.nan)
            faltando.append(nomes[i])
        else:
            faixa[i] = m.jnt_range[jid]
    if faltando:
        print(f"⚠ sem faixa no modelo ({len(faltando)}): {', '.join(faltando[:6])}"
              + (" ..." if len(faltando) > 6 else ""))
    return nomes, faixa, np.asarray(c.q_default, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cena", required=True, help="pasta com cena.mjb e cena.npz")
    ap.add_argument("--checkpoint", required=True, help="o model_*.pt")
    ap.add_argument("--saida", default="juntas", help="prefixo dos arquivos de saída")
    ap.add_argument("--roteiro", default=ROTEIRO_PADRAO,
                    help="rotulo:ELO:segundos[:vx] separados por vírgula")
    ap.add_argument("--voltas", type=int, default=1, help="quantas vezes repetir o roteiro")
    ap.add_argument("--afasta", type=float, default=10.0,
                    help="metros em +x para onde laje e caixa vão na marcha")
    ap.add_argument("--sem-viewer", action="store_true", help="roda o mais rápido que der")
    args = ap.parse_args()

    fases = analisa_roteiro(args.roteiro)
    m, c = carrega_cena(args.cena)
    d = mujoco.MjData(m)
    ator = Ator(args.checkpoint)
    if ator.dim_entrada != int(c.dim_obs):
        raise SystemExit(f"o checkpoint espera {ator.dim_entrada} canais e a cena monta "
                         f"{int(c.dim_obs)}. Checkpoint de outra fase?")

    nomes, faixa, q_def = regua_das_juntas(m, c)
    ids_q = np.asarray(c.ids_junta_qpos, dtype=np.int64)
    ids_atuador = np.asarray(c.ids_atuador, dtype=np.int64)
    q_default_acao = np.asarray(c.q_default_acao, dtype=np.float64)
    escala_acao = np.asarray(c.escala_acao, dtype=np.float64)
    decimation, physics_dt = int(c.decimation), float(c.physics_dt)
    dt = physics_dt * decimation

    restaura(m, d, c)
    acao = np.zeros(ator.dim_saida)
    twist = np.zeros(3)
    linhas: list[dict] = []

    print(f"[registra] cena {args.cena}  checkpoint iter={ator.iteracao}  dt={dt*1000:.0f} ms")
    print(f"[registra] roteiro: " + "  ".join(
        f"{f.rotulo}({ELOS[f.elo]},{f.segundos:g}s"
        + (f",vx={f.vx:g}" if f.vx else "") + ")" for f in fases)
        + f"   × {args.voltas}")

    viewer = None
    if not args.sem_viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d, key_callback=lambda k: None)

    passo = 0
    try:
        for volta in range(args.voltas):
            for fase in fases:
                twist[:] = (fase.vx, 0.0, 0.0)
                if fase.anda:
                    limpa_a_cena(m, d, c, args.afasta)
                n = max(1, int(round(fase.segundos / dt)))
                for _ in range(n):
                    if viewer is not None and not viewer.is_running():
                        raise KeyboardInterrupt
                    t0 = time.perf_counter()

                    obs, _ = monta_observacao(m, d, c, twist, fase.elo, acao)
                    acao = ator(obs)
                    d.ctrl[ids_atuador] = q_default_acao + escala_acao * acao
                    for _ in range(decimation):
                        mujoco.mj_step(m, d)
                    # ⚠ obrigatório: o `mj_step` deixa xpos/xquat/sensordata atrasados
                    # um subpasso, e é depois dele que a observação e o log são lidos.
                    mujoco.mj_forward(m, d)

                    q = d.qpos[ids_q]
                    ln = {"passo": passo, "t": round(passo * dt, 4),
                          "fase": fase.rotulo, "elo": ELOS[fase.elo]}
                    for k in range(len(ELOS)):
                        ln[f"oh_{ELOS[k].lower()}"] = int(k == fase.elo)
                    for i, nome in enumerate(nomes):
                        ln[f"q_{nome}"] = float(q[i])
                    linhas.append(ln)
                    passo += 1

                    if viewer is not None:
                        viewer.sync()
                        atraso = dt - (time.perf_counter() - t0)
                        if atraso > 0:
                            time.sleep(atraso)
                    if passo % 200 == 0:
                        print(f"\r  passo {passo}  fase {fase.rotulo:<8s}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[registra] interrompido — gravando o que já rodou")
    finally:
        if viewer is not None:
            viewer.close()

    if not linhas:
        raise SystemExit("nada gravado")

    saida = Path(args.saida).expanduser()
    csv_grande = saida.with_suffix(".csv")
    with open(csv_grande, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    csv_regua = saida.with_suffix(".limites.csv")
    with open(csv_regua, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["junta", "lo", "hi", "default"])
        for i, nome in enumerate(nomes):
            w.writerow([nome, faixa[i, 0], faixa[i, 1], q_def[i]])

    print(f"\n[registra] {len(linhas)} linhas × {len(linhas[0])} colunas -> {csv_grande}")
    print(f"[registra] régua das juntas -> {csv_regua}")


if __name__ == "__main__":
    main()
